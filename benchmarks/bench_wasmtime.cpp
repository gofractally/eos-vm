#ifdef BENCH_HAS_WASMTIME
#include "bench_hosts.hpp"
#include <wasmtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>

static wasm_trap_t* wt_identity(void*, wasmtime_caller_t*, const wasmtime_val_t* args, size_t, wasmtime_val_t* results, size_t) {
   results[0].kind = WASMTIME_I32;
   results[0].of.i32 = bench_host_identity(args[0].of.i32);
   return nullptr;
}
static wasm_trap_t* wt_accumulate(void*, wasmtime_caller_t*, const wasmtime_val_t* args, size_t, wasmtime_val_t*, size_t) {
   bench_host_accumulate(args[0].of.i64);
   return nullptr;
}
static wasm_trap_t* wt_mix(void*, wasmtime_caller_t*, const wasmtime_val_t* args, size_t, wasmtime_val_t* results, size_t) {
   results[0].kind = WASMTIME_I64;
   results[0].of.i64 = bench_host_mix(args[0].of.i64, args[1].of.i64);
   return nullptr;
}
static wasm_trap_t* wt_mem_op(void*, wasmtime_caller_t*, const wasmtime_val_t* args, size_t, wasmtime_val_t*, size_t) {
   bench_host_mem_op(args[0].of.i32, args[1].of.i32);
   return nullptr;
}
static wasm_trap_t* wt_multi(void*, wasmtime_caller_t*, const wasmtime_val_t* args, size_t, wasmtime_val_t* results, size_t) {
   results[0].kind = WASMTIME_I64;
   results[0].of.i64 = bench_host_multi(args[0].of.i64, args[1].of.i64, args[2].of.i32, args[3].of.i32);
   return nullptr;
}

double run_wasmtime(const std::vector<uint8_t>& wasm, const char* func, uint32_t n) {
   wasm_engine_t* engine = wasm_engine_new();
   wasmtime_store_t* store = wasmtime_store_new(engine, nullptr, nullptr);
   wasmtime_context_t* ctx = wasmtime_store_context(store);

   wasmtime_module_t* module = nullptr;
   wasmtime_error_t* err = wasmtime_module_new(engine, wasm.data(), wasm.size(), &module);
   if (err) { fprintf(stderr, "wasmtime module error\n"); wasmtime_error_delete(err); wasmtime_store_delete(store); wasm_engine_delete(engine); return -1; }

   // Build func types
   auto make_ft = [](std::initializer_list<wasm_valkind_t> params, std::initializer_list<wasm_valkind_t> results) -> wasm_functype_t* {
      wasm_valtype_vec_t pv, rv;
      wasm_valtype_vec_new_uninitialized(&pv, params.size());
      wasm_valtype_vec_new_uninitialized(&rv, results.size());
      size_t i = 0;
      for (auto k : params) pv.data[i++] = wasm_valtype_new(k);
      i = 0;
      for (auto k : results) rv.data[i++] = wasm_valtype_new(k);
      return wasm_functype_new(&pv, &rv);
   };

   wasm_functype_t* fts[] = {
      make_ft({WASM_I32}, {WASM_I32}),
      make_ft({WASM_I64}, {}),
      make_ft({WASM_I64, WASM_I64}, {WASM_I64}),
      make_ft({WASM_I32, WASM_I32}, {}),
      make_ft({WASM_I64, WASM_I64, WASM_I32, WASM_I32}, {WASM_I64}),
   };
   wasmtime_func_callback_t cbs[] = {wt_identity, wt_accumulate, wt_mix, wt_mem_op, wt_multi};

   wasmtime_extern_t imports[5];
   for (int i = 0; i < 5; i++) {
      wasmtime_func_new(ctx, fts[i], cbs[i], nullptr, nullptr, &imports[i].of.func);
      imports[i].kind = WASMTIME_EXTERN_FUNC;
   }

   wasmtime_instance_t instance;
   wasm_trap_t* trap = nullptr;
   err = wasmtime_instance_new(ctx, module, imports, 5, &instance, &trap);
   if (err || trap) {
      fprintf(stderr, "wasmtime instantiate error\n");
      if (err) wasmtime_error_delete(err);
      if (trap) wasm_trap_delete(trap);
      for (auto& ft : fts) wasm_functype_delete(ft);
      wasmtime_module_delete(module);
      wasmtime_store_delete(store);
      wasm_engine_delete(engine);
      return -1;
   }

   wasmtime_extern_t exp;
   bool found = wasmtime_instance_export_get(ctx, &instance, func, strlen(func), &exp);
   if (!found || exp.kind != WASMTIME_EXTERN_FUNC) {
      fprintf(stderr, "wasmtime: export %s not found\n", func);
      for (auto& ft : fts) wasm_functype_delete(ft);
      wasmtime_module_delete(module);
      wasmtime_store_delete(store);
      wasm_engine_delete(engine);
      return -1;
   }

   wasmtime_val_t arg = {.kind = WASMTIME_I32, .of = {.i32 = static_cast<int32_t>(n)}};
   wasmtime_val_t result;

   auto t1 = std::chrono::high_resolution_clock::now();
   err = wasmtime_func_call(ctx, &exp.of.func, &arg, 1, &result, 1, &trap);
   auto t2 = std::chrono::high_resolution_clock::now();
   if (err || trap) { fprintf(stderr, "wasmtime call error\n"); if (err) wasmtime_error_delete(err); if (trap) wasm_trap_delete(trap); }

   for (auto& ft : fts) wasm_functype_delete(ft);
   wasmtime_module_delete(module);
   wasmtime_store_delete(store);
   wasm_engine_delete(engine);
   return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

double run_wasmtime_compute(const std::vector<uint8_t>& wasm, const char* func, uint32_t n) {
   wasm_engine_t* engine = wasm_engine_new();
   wasmtime_store_t* store = wasmtime_store_new(engine, nullptr, nullptr);
   wasmtime_context_t* ctx = wasmtime_store_context(store);

   wasmtime_module_t* module = nullptr;
   wasmtime_error_t* err = wasmtime_module_new(engine, wasm.data(), wasm.size(), &module);
   if (err) { fprintf(stderr, "wasmtime compute module error\n"); wasmtime_error_delete(err); wasmtime_store_delete(store); wasm_engine_delete(engine); return -1; }

   wasmtime_instance_t instance;
   wasm_trap_t* trap = nullptr;
   err = wasmtime_instance_new(ctx, module, nullptr, 0, &instance, &trap);
   if (err || trap) {
      fprintf(stderr, "wasmtime compute instantiate error\n");
      if (err) wasmtime_error_delete(err);
      if (trap) wasm_trap_delete(trap);
      wasmtime_module_delete(module);
      wasmtime_store_delete(store);
      wasm_engine_delete(engine);
      return -1;
   }

   wasmtime_extern_t exp;
   bool found = wasmtime_instance_export_get(ctx, &instance, func, strlen(func), &exp);
   if (!found || exp.kind != WASMTIME_EXTERN_FUNC) {
      fprintf(stderr, "wasmtime compute: export %s not found\n", func);
      wasmtime_module_delete(module);
      wasmtime_store_delete(store);
      wasm_engine_delete(engine);
      return -1;
   }

   wasmtime_val_t arg = {.kind = WASMTIME_I32, .of = {.i32 = static_cast<int32_t>(n)}};
   wasmtime_val_t result;

   auto t1 = std::chrono::high_resolution_clock::now();
   err = wasmtime_func_call(ctx, &exp.of.func, &arg, 1, &result, 1, &trap);
   auto t2 = std::chrono::high_resolution_clock::now();
   if (err || trap) { fprintf(stderr, "wasmtime compute call error\n"); if (err) wasmtime_error_delete(err); if (trap) wasm_trap_delete(trap); }

   wasmtime_module_delete(module);
   wasmtime_store_delete(store);
   wasm_engine_delete(engine);
   return std::chrono::duration<double, std::milli>(t2 - t1).count();
}
#endif
