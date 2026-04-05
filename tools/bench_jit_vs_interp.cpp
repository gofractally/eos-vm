#include <eosio/vm/backend.hpp>
#include <eosio/vm/error_codes.hpp>
#include <eosio/vm/watchdog.hpp>

#include <chrono>
#include <iostream>
#include <string>

using namespace eosio;
using namespace eosio::vm;

// Host function for CoreMark-minimal: returns current time in milliseconds
int64_t clock_ms() {
   auto now = std::chrono::steady_clock::now();
   return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
}

using rhf_t = registered_host_functions<standalone_function_t>;

struct result {
   long long startup_ns;
   long long exec_ns;
   float     return_val;
};

template<typename Impl>
result run_bench(wasm_code& code, wasm_allocator& wa, const std::string& func, int iterations) {
   using backend_t = backend<rhf_t, Impl>;
   auto t1 = std::chrono::high_resolution_clock::now();
   backend_t bkend(code, &wa);
   rhf_t::resolve(bkend.get_module());
   auto t2 = std::chrono::high_resolution_clock::now();

   long long total_exec = 0;
   float last_ret = 0;
   for (int i = 0; i < iterations; ++i) {
      bkend.initialize(nullptr);
      auto t3 = std::chrono::high_resolution_clock::now();
      auto ret = bkend.call_with_return("env", func);
      auto t4 = std::chrono::high_resolution_clock::now();
      total_exec += std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count();
      if (ret)
         last_ret = ret->to_f32();
   }

   return {
      std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(),
      total_exec / iterations,
      last_ret
   };
}

int main(int argc, char** argv) {
   wasm_allocator wa;

   if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " <wasm_file> [function] [iterations]\n";
      std::cerr << "  function:   exported function name (default: \"run\")\n";
      std::cerr << "  iterations: number of runs to average (default: 1)\n";
      return 1;
   }

   std::string func = argc >= 3 ? argv[2] : "run";
   int iterations    = argc >= 4 ? std::atoi(argv[3]) : 1;
   if (iterations < 1) iterations = 1;

   // Register host functions
   rhf_t::add<&clock_ms>("env", "clock_ms");

   auto code = read_wasm(argv[1]);

   std::cout << "File: " << argv[1] << "\n";
   std::cout << "Function: " << func << "\n";
   std::cout << "Iterations: " << iterations << "\n\n";

   result interp{}, jit_r{};
   bool interp_ok = false, jit_ok = false;

   std::cout << "Running interpreter..." << std::flush;
   try {
      interp = run_bench<interpreter>(code, wa, func, iterations);
      interp_ok = true;
      std::cout << " done\n";
   } catch (const std::exception& ex) {
      std::cout << " FAILED: " << ex.what() << "\n";
   }

   std::cout << "Running JIT..." << std::flush;
   try {
      jit_r = run_bench<jit>(code, wa, func, iterations);
      jit_ok = true;
      std::cout << " done\n";
   } catch (const std::exception& ex) {
      std::cout << " FAILED: " << ex.what() << "\n";
   }

   std::cout << "\n" << std::string(60, '-') << "\n";
   printf("%-14s %14s %14s %10s\n", "", "Startup (ms)", "Exec (ms)", "Result");
   std::cout << std::string(60, '-') << "\n";
   if (interp_ok)
      printf("Interpreter    %14.2f %14.2f %10.2f\n",
             interp.startup_ns / 1e6, interp.exec_ns / 1e6, interp.return_val);
   if (jit_ok)
      printf("JIT            %14.2f %14.2f %10.2f\n",
             jit_r.startup_ns / 1e6, jit_r.exec_ns / 1e6, jit_r.return_val);

   if (interp_ok && jit_ok) {
      std::cout << std::string(60, '-') << "\n";
      if (jit_r.return_val > 0 && interp.return_val > 0) {
         printf("JIT speedup    %14s %14s %9.1fx\n",
                "", "", (double)jit_r.return_val / interp.return_val);
      } else {
         printf("JIT/Interp     %13.1fx startup, %.1fx exec\n",
                (double)jit_r.startup_ns / interp.startup_ns,
                (double)interp.exec_ns / jit_r.exec_ns);
      }
   }

   return 0;
}
