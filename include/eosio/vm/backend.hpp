#pragma once

#include <eosio/vm/allocator.hpp>
#include <eosio/vm/bitcode_writer.hpp>
#include <eosio/vm/config.hpp>
#include <eosio/vm/debug_visitor.hpp>
#include <eosio/vm/execution_context.hpp>
#include <eosio/vm/interpret_visitor.hpp>
#include <eosio/vm/null_writer.hpp>
#include <eosio/vm/parser.hpp>
#include <eosio/vm/types.hpp>

#ifdef __x86_64__
#include <eosio/vm/x86_64.hpp>
#endif

#include <atomic>
#include <exception>
#include <iostream>
#include <optional>
#include <string_view>
#include <system_error>
#include <vector>

namespace eosio { namespace vm {

#ifdef __x86_64__
   struct jit {
      template<typename Host>
      using context = jit_execution_context<Host>;
      template<typename Host, typename Options, typename DebugInfo>
      using parser = binary_parser<machine_code_writer<jit_execution_context<Host>, detail::has_max_stack_bytes<Options>>, Options, DebugInfo>;
      static constexpr bool is_jit = true;
   };

   struct jit_profile {
      template<typename Host>
      using context = jit_execution_context<Host, true>;
      template<typename Host, typename Options, typename DebugInfo>
      using parser = binary_parser<machine_code_writer<context<Host>, detail::has_max_stack_bytes<Options>>, Options, DebugInfo>;
      static constexpr bool is_jit = true;
   };
#endif

   struct interpreter {
      template<typename Host>
      using context = execution_context<Host>;
      template<typename Host, typename Options, typename DebugInfo>
      using parser = binary_parser<bitcode_writer, Options, DebugInfo>;
      static constexpr bool is_jit = false;
   };

   struct null_backend {
      template<typename Host>
      using context = null_execution_context<Host>;
      template<typename Host, typename Options, typename DebugInfo>
      using parser = binary_parser<null_writer, Options, DebugInfo>;
      static constexpr bool is_jit = false;
   };

   template <typename HostFunctions = std::nullptr_t, typename Impl = interpreter, typename Options = default_options, typename DebugInfo = null_debug_info>
   class backend {
      using host_t     = detail::host_type_t<HostFunctions>;
      using context_t  = typename Impl::template context<HostFunctions>;
      using parser_t   = typename Impl::template parser<HostFunctions, Options, DebugInfo>;
      template<typename XDebugInfo>
      using parser_tpl   = typename Impl::template parser<HostFunctions, Options, XDebugInfo>;
      void construct(host_t* host=nullptr) {
         mod.finalize();
         if (exec_ctx_created_by_backend) {
            ctx->set_wasm_allocator(memory_alloc);
         }
         // Now data required by JIT is finalized; create JIT module
         // such that memory used in parsing can be released.
         if constexpr (Impl::is_jit) {
            assert(mod.allocator._base == nullptr);
         }
         if (exec_ctx_created_by_backend) {
            ctx->initialize_globals();
         }
         if constexpr (!std::is_same_v<HostFunctions, std::nullptr_t>)
            HostFunctions::resolve(mod);
         // FIXME: should not hard code knowledge of null_backend here
         if (exec_ctx_created_by_backend) {
            if constexpr (!std::is_same_v<Impl, null_backend>)
               initialize(host);
         }
      }
    public:
      backend(wasm_code&& code, host_t& host, wasm_allocator* alloc, const Options& options = Options{})
         : memory_alloc(alloc), ctx(new context_t{parse_module(code, options), detail::choose_stack_limit(options)}) {
         ctx->set_max_pages(detail::get_max_pages(options));
         construct(&host);
      }
      backend(wasm_code&& code, wasm_allocator* alloc, const Options& options = Options{})
         : memory_alloc(alloc), ctx(new context_t{parse_module(code, options), detail::choose_stack_limit(options)}) {
         ctx->set_max_pages(detail::get_max_pages(options));
         construct();
      }
      backend(wasm_code& code, host_t& host, wasm_allocator* alloc, const Options& options = Options{})
         : memory_alloc(alloc), ctx(new context_t{parse_module(code, options), detail::choose_stack_limit(options)}) {
         ctx->set_max_pages(detail::get_max_pages(options));
         construct(&host);
      }
      backend(wasm_code& code, wasm_allocator* alloc, const Options& options = Options{})
         : memory_alloc(alloc), ctx(new context_t{(parse_module(code, options)), detail::choose_stack_limit(options)}) {
         ctx->set_max_pages(detail::get_max_pages(options));
         construct();
      }
      template <typename XDebugInfo>
      backend(wasm_code& code, wasm_allocator* alloc, const Options& options, XDebugInfo& debug)
         : memory_alloc(alloc), ctx(new context_t{(parse_module(code, options, debug)), detail::choose_stack_limit(options)}) {
         ctx->set_max_pages(detail::get_max_pages(options));
         construct();
      }
      backend(wasm_code_ptr& ptr, size_t sz, host_t& host, wasm_allocator* alloc, const Options& options = Options{})
         : memory_alloc(alloc), ctx(new context_t{parse_module2(ptr, sz, options, true), detail::choose_stack_limit(options)}) { // single parsing. original behavior {
         ctx->set_max_pages(detail::get_max_pages(options));
         construct(&host);
      }
      // Leap:
      //  * Contract validation only needs single parsing as the instantiated module is not cached.
      //  * JIT execution needs single parsing only.
      //  * Interpreter execution requires two-passes parsing to prevent memory mappings exhaustion
      //  * Leap reuses execution context per thread; is_exec_ctx_created_by_backend is set
      //  to false when a backend is constructued
      backend(wasm_code_ptr& ptr, size_t sz, wasm_allocator* alloc, const Options& options = Options{}, bool single_parsing = true, bool is_exec_ctx_created_by_backend = true)
         : memory_alloc(alloc), exec_ctx_created_by_backend(is_exec_ctx_created_by_backend), initial_max_call_depth(detail::choose_stack_limit(options)), initial_max_pages(detail::get_max_pages(options)) {
         if (exec_ctx_created_by_backend) {
            ctx = new context_t{parse_module2(ptr, sz, options, single_parsing), initial_max_call_depth};
            ctx->set_max_pages(initial_max_pages);
         } else {
            parse_module2(ptr, sz, options, single_parsing);
         }
         construct();
      }

      ~backend() {
         if (exec_ctx_created_by_backend && ctx) {
            // delete only if the context was created by the backend
            delete ctx;
         }
      }

      module& parse_module(wasm_code& code, const Options& options) {
         mod.allocator.use_default_memory();
         return parser_t{ mod.allocator, options }.parse_module(code, mod, debug);
      }

      template <typename XDebugInfo>
      module& parse_module(wasm_code& code, const Options& options, XDebugInfo& debug) {
         mod.allocator.use_default_memory();
         return parser_tpl<XDebugInfo>{ mod.allocator, options }.parse_module(code, mod, debug);
      }

      module& parse_module2(wasm_code_ptr& ptr, size_t sz, const Options& options, bool single_parsing) {
         if (single_parsing) {
            mod.allocator.use_default_memory();
            return parser_t{ mod.allocator, options }.parse_module2(ptr, sz, mod, debug);
         } else {
            // To prevent large number of memory mappings used, two-passes of
            // parsing are performed.
            wasm_code_ptr orig_ptr = ptr;
            size_t largest_size = 0;

            // First pass: finds max size of memory required by parsing.
            {
               // Memory used by this pass is freed when going out of the scope
               module first_pass_module;
               first_pass_module.allocator.use_default_memory();
               parser_t{ first_pass_module.allocator, options }.parse_module2(ptr, sz, first_pass_module, debug);
               first_pass_module.finalize();
               largest_size = first_pass_module.allocator.largest_used_size();
            }

            // Second pass: uses actual required memory for final parsing
            mod.allocator.use_fixed_memory(largest_size);
            return parser_t{ mod.allocator, options }.parse_module2(orig_ptr, sz, mod, debug);
         }
      }

      void set_context(context_t* ctx_ptr) {
         // execution context can be only set when it is not already created by the backend
         assert(!exec_ctx_created_by_backend);
         ctx = ctx_ptr;
      }

      inline void reset_max_call_depth() {
         assert(!exec_ctx_created_by_backend);
         ctx->set_max_call_depth(initial_max_call_depth);
      }

      inline void reset_max_pages() {
         assert(!exec_ctx_created_by_backend);
         ctx->set_max_pages(initial_max_pages);
      }

      template <typename... Args>
      inline auto operator()(stack_manager& alt_stack, host_t& host, const std::string_view& mod, const std::string_view& func, Args... args) {
         return call(alt_stack, host, mod, func, args...);
      }

      template <typename... Args>
      inline auto operator()(host_t& host, const std::string_view& mod, const std::string_view& func, Args... args) {
         return call(host, mod, func, args...);
      }

      template <typename... Args>
      inline bool operator()(const std::string_view& mod, const std::string_view& func, Args... args) {
         return call(mod, func, args...);
      }

      // Only dynamic options matter.  Parser options will be ignored.
      inline backend& initialize(host_t* host, const Options& new_options) {
         ctx->set_max_call_depth(detail::choose_stack_limit(new_options));
         ctx->set_max_pages(detail::get_max_pages(new_options));
         initialize(host);
         return *this;
      }

      inline backend& initialize(host_t* host=nullptr) {
         if (memory_alloc) {
            ctx->reset();
            ctx->execute_start(host, interpret_visitor(*ctx));
         }
         return *this;
      }

      inline backend& initialize(stack_manager& alt_stack, host_t* host=nullptr) {
         if (memory_alloc) {
            ctx->reset();
            ctx->execute_start(alt_stack, host, interpret_visitor(*ctx));
         }
         return *this;
      }

      inline backend& initialize(host_t& host) {
         return initialize(&host);
      }

      template <typename... Args>
      inline bool call_indirect(host_t* host, uint32_t func_index, Args... args) {
         if constexpr (eos_vm_debug) {
            ctx->execute_func_table(host, debug_visitor(*ctx), func_index, args...);
         } else {
            ctx->execute_func_table(host, interpret_visitor(*ctx), func_index, args...);
         }
         return true;
      }

      template <typename... Args>
      inline bool call(host_t* host, uint32_t func_index, Args... args) {
         if constexpr (eos_vm_debug) {
            ctx->execute(host, debug_visitor(*ctx), func_index, args...);
         } else {
            ctx->execute(host, interpret_visitor(*ctx), func_index, args...);
         }
         return true;
      }

      template <typename... Args>
      inline bool call(stack_manager& alt_stack, host_t& host, const std::string_view& mod, const std::string_view& func, Args... args) {
         if constexpr (eos_vm_debug) {
            ctx->execute(alt_stack, &host, debug_visitor(*ctx), func, args...);
         } else {
            ctx->execute(alt_stack, &host, interpret_visitor(*ctx), func, args...);
         }
         return true;
      }

      template <typename... Args>
      inline bool call(host_t& host, const std::string_view& mod, const std::string_view& func, Args... args) {
         if constexpr (eos_vm_debug) {
            ctx->execute(&host, debug_visitor(*ctx), func, args...);
         } else {
            ctx->execute(&host, interpret_visitor(*ctx), func, args...);
         }
         return true;
      }

      template <typename... Args>
      inline bool call(const std::string_view& mod, const std::string_view& func, Args... args) {
         if constexpr (eos_vm_debug) {
            ctx->execute(nullptr, debug_visitor(*ctx), func, args...);
         } else {
            ctx->execute(nullptr, interpret_visitor(*ctx), func, args...);
         }
         return true;
      }

      template <typename... Args>
      inline auto call_with_return(host_t& host, const std::string_view& mod, const std::string_view& func, Args... args ) {
         if constexpr (eos_vm_debug) {
            return ctx->execute(&host, debug_visitor(*ctx), func, args...);
         } else {
            return ctx->execute(&host, interpret_visitor(*ctx), func, args...);
         }
      }

      template <typename... Args>
      inline auto call_with_return(const std::string_view& mod, const std::string_view& func, Args... args) {
         if constexpr (eos_vm_debug) {
            return ctx->execute(nullptr, debug_visitor(*ctx), func, args...);
         } else {
            return ctx->execute(nullptr, interpret_visitor(*ctx), func, args...);
         }
      }

      template<typename Watchdog, typename F>
      inline void timed_run(Watchdog&& wd, F&& f) {
         std::atomic<bool>       _timed_out = false;
         auto reenable_code = scope_guard{[&](){
            if (_timed_out) {
               mod.allocator.enable_code(Impl::is_jit);
            }
         }};
         try {
            auto wd_guard = wd.scoped_run([this,&_timed_out]() {
               _timed_out = true;
               mod.allocator.disable_code();
            });
            static_cast<F&&>(f)();
         } catch(wasm_memory_exception&) {
            if (_timed_out) {
               throw timeout_exception{ "execution timed out" };
            } else {
               throw;
            }
         }
      }

      template <typename Watchdog>
      inline void execute_all(Watchdog&& wd, host_t& host) {
         timed_run(static_cast<Watchdog&&>(wd), [&]() {
            for (int i = 0; i < mod.exports.size(); i++) {
               if (mod.exports[i].kind == external_kind::Function) {
                  std::string s{ (const char*)mod.exports[i].field_str.data(), mod.exports[i].field_str.size() };
                  ctx->execute(host, interpret_visitor(*ctx), s);
               }
            }
         });
      }

      template <typename Watchdog>
      inline void execute_all(Watchdog&& wd) {
         timed_run(static_cast<Watchdog&&>(wd), [&]() {
            for (int i = 0; i < mod.exports.size(); i++) {
               if (mod.exports[i].kind == external_kind::Function) {
                  std::string s{ (const char*)mod.exports[i].field_str.data(), mod.exports[i].field_str.size() };
                  ctx->execute(nullptr, interpret_visitor(*ctx), s);
               }
            }
         });
      }

      inline void set_wasm_allocator(wasm_allocator* alloc) {
         memory_alloc = alloc;
         ctx->set_wasm_allocator(memory_alloc);
      }

      inline wasm_allocator* get_wasm_allocator() { return memory_alloc; }
      inline module&         get_module() { return mod; }
      inline void            exit(const std::error_code& ec) { ctx->exit(ec); }
      inline auto&           get_context() { return *ctx; }

      const DebugInfo& get_debug() const { return debug; }

    private:
      wasm_allocator* memory_alloc = nullptr; // non owning pointer
      module          mod;
      DebugInfo       debug;
      context_t*      ctx = nullptr;
      bool            exec_ctx_created_by_backend = true; // true if execution context is created by backend (legacy behavior), false if provided by users (Leap uses this)
      uint32_t        initial_max_call_depth = 0;
      uint32_t        initial_max_pages = 0;
   };
}} // namespace eosio::vm
