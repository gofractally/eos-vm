#include <eosio/vm/signals.hpp>
#include <chrono>
#include <csignal>
#include <future>
#include <thread>
#include <iostream>
#include <poll.h>

#include <catch2/catch.hpp>

struct test_exception {};

TEST_CASE("Testing signals", "[invoke_with_signal_handler]") {
   bool okay = false;
   eosio::vm::growable_allocator code_alloc;
   eosio::vm::wasm_allocator wasm_alloc;
   try {
      eosio::vm::invoke_with_signal_handler([&]() {
         volatile auto i = *wasm_alloc.get_base_ptr<unsigned char>();
      }, [](int sig) {
         throw test_exception{};
      }, code_alloc, &wasm_alloc);
   } catch(test_exception&) {
      okay = true;
   }
   CHECK(okay);
}

TEST_CASE("Testing throw", "[signal_handler_throw]") {
   eosio::vm::growable_allocator code_alloc;
   eosio::vm::wasm_allocator wasm_alloc;
   CHECK_THROWS_AS(eosio::vm::invoke_with_signal_handler([](){
      eosio::vm::throw_<eosio::vm::wasm_exit_exception>( "Exiting" );
   }, [](int){}, code_alloc, &wasm_alloc), eosio::vm::wasm_exit_exception);
}

static volatile sig_atomic_t sig_handled;

static void handle_signal(int sig) {
   sig_handled = 42 + sig;
}

static void handle_signal_sigaction(int sig, siginfo_t* info, void* uap) {
   sig_handled = 142 + sig;
}

TEST_CASE("Test signal handler forwarding", "[signal_handler_forward]") {
   // reset backup signal handlers
   auto guard = eosio::vm::scope_guard{[]{
      std::signal(SIGSEGV, SIG_DFL);
      std::signal(SIGBUS, SIG_DFL);
      std::signal(SIGFPE, SIG_DFL);
      eosio::vm::setup_signal_handler_impl(); // This is normally only called once
   }};
   {
      std::signal(SIGSEGV, &handle_signal);
      std::signal(SIGBUS, &handle_signal);
      std::signal(SIGFPE, &handle_signal);
      eosio::vm::setup_signal_handler_impl();
      sig_handled = 0;
      std::raise(SIGSEGV);
      CHECK(sig_handled == 42 + SIGSEGV);
#ifndef __linux__
      sig_handled = 0;
      std::raise(SIGBUS);
      CHECK(sig_handled == 42 + SIGBUS);
#endif
      sig_handled = 0;
      std::raise(SIGFPE);
      CHECK(sig_handled == 42 + SIGFPE);
   }
   {
      struct sigaction sa;
      sa.sa_sigaction = &handle_signal_sigaction;
      sigemptyset(&sa.sa_mask);
      sa.sa_flags = SA_NODEFER | SA_SIGINFO;
      sigaction(SIGSEGV, &sa, nullptr);
      sigaction(SIGBUS, &sa, nullptr);
      sigaction(SIGFPE, &sa, nullptr);
      eosio::vm::setup_signal_handler_impl();
      sig_handled = 0;
      std::raise(SIGSEGV);
      CHECK(sig_handled == 142 + SIGSEGV);
#ifndef __linux__
      sig_handled = 0;
      std::raise(SIGBUS);
      CHECK(sig_handled == 142 + SIGBUS);
#endif
      sig_handled = 0;
      std::raise(SIGFPE);
      CHECK(sig_handled == 142 + SIGFPE);
   }
}

static int signal_pipefds[2];
static void signal_write_pipe(int) {
   write(signal_pipefds[1], "", 1);
}

TEST_CASE("Test signal mask longjmp restoration", "[signal_mask_longjmp_restoration]") {
   REQUIRE(pipe(signal_pipefds) == 0);
   auto pipe_guard = eosio::vm::scope_guard{[&]{
      close(signal_pipefds[0]);
      close(signal_pipefds[1]);
   }};

   struct sigaction sa, old_sa;
   sa.sa_handler = signal_write_pipe;
   sigemptyset(&sa.sa_mask);
   REQUIRE(sigaction(SIGTERM, &sa, &old_sa) == 0);
   auto resore_orig_sigterm_guard = eosio::vm::scope_guard{[&]{
      sigaction(SIGTERM, &old_sa, nullptr);
   }};

   std::promise<void> thread_running;
   std::promise<void> thread_may_return;
   std::thread t([&] {
      //block everything except SIGTERM in this thread
      sigset_t suspend_mask;
      sigfillset(&suspend_mask);
      sigdelset(&suspend_mask, SIGTERM);
      pthread_sigmask(SIG_SETMASK, &suspend_mask, nullptr);

      thread_running.set_value();
      thread_may_return.get_future().get();
   });
   auto thread_guard = eosio::vm::scope_guard{[&] {
      thread_may_return.set_value();
      t.join();
   }};

   thread_running.get_future().get();

   sigset_t old_mask;
   //block SIGTERM on main thread
   sigset_t set;
   sigemptyset(&set);
   sigaddset(&set, SIGTERM);
   REQUIRE(pthread_sigmask(SIG_BLOCK, &set, &old_mask) == 0);
   auto mask_guard = eosio::vm::scope_guard{[&] {
      pthread_sigmask(SIG_SETMASK, &old_mask, nullptr);
   }};

   eosio::vm::growable_allocator code_alloc;
   eosio::vm::wasm_allocator wasm_alloc;
   eosio::vm::invoke_with_signal_handler([&]() {
      volatile auto i = *wasm_alloc.get_base_ptr<unsigned char>();
   }, [](int sig){}, code_alloc, &wasm_alloc);

   //send SIGTERM to the process, it should be delivered to the signal handling thread
   REQUIRE(kill(getpid(), SIGTERM) == 0);

   pollfd pfd{
      .fd = signal_pipefds[0],
      .events = POLLIN,
   };
   int r = poll(&pfd, 1, 500);
   REQUIRE(r == 1);
   REQUIRE((pfd.revents & POLLIN) != 0);

   char buf;
   REQUIRE(read(signal_pipefds[0], &buf, 1) == 1);
}
