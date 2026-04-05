#ifdef BENCH_HAS_WASMER
#include "bench_hosts.hpp"
#include <wasmer.h>
#include <chrono>
#include <cstdio>
#include <cstring>

static wasm_trap_t* ws_identity(const wasm_val_vec_t* args, wasm_val_vec_t* results) {
   results->data[0] = WASM_I32_VAL(bench_host_identity(args->data[0].of.i32));
   return nullptr;
}
static wasm_trap_t* ws_accumulate(const wasm_val_vec_t* args, wasm_val_vec_t*) {
   bench_host_accumulate(args->data[0].of.i64);
   return nullptr;
}
static wasm_trap_t* ws_mix(const wasm_val_vec_t* args, wasm_val_vec_t* results) {
   results->data[0] = WASM_I64_VAL(bench_host_mix(args->data[0].of.i64, args->data[1].of.i64));
   return nullptr;
}
static wasm_trap_t* ws_mem_op(const wasm_val_vec_t* args, wasm_val_vec_t*) {
   bench_host_mem_op(args->data[0].of.i32, args->data[1].of.i32);
   return nullptr;
}
static wasm_trap_t* ws_multi(const wasm_val_vec_t* args, wasm_val_vec_t* results) {
   results->data[0] = WASM_I64_VAL(bench_host_multi(args->data[0].of.i64, args->data[1].of.i64,
                                                     args->data[2].of.i32, args->data[3].of.i32));
   return nullptr;
}

double run_wasmer(const std::vector<uint8_t>& wasm, const char* func, uint32_t n) {
   wasm_engine_t* engine = wasm_engine_new();
   wasm_store_t* store = wasm_store_new(engine);

   wasm_byte_vec_t binary = {wasm.size(), const_cast<wasm_byte_t*>(reinterpret_cast<const wasm_byte_t*>(wasm.data()))};
   wasm_module_t* module = wasm_module_new(store, &binary);
   if (!module) { fprintf(stderr, "wasmer module error\n"); wasm_store_delete(store); wasm_engine_delete(engine); return -1; }

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
   wasm_func_callback_t cbs[] = {ws_identity, ws_accumulate, ws_mix, ws_mem_op, ws_multi};

   wasm_extern_t* imports_arr[5];
   wasm_func_t* funcs[5];
   for (int i = 0; i < 5; i++) {
      funcs[i] = wasm_func_new(store, fts[i], cbs[i]);
      imports_arr[i] = wasm_func_as_extern(funcs[i]);
   }
   wasm_extern_vec_t imports = {5, imports_arr};

   wasm_instance_t* instance = wasm_instance_new(store, module, &imports, nullptr);
   if (!instance) {
      fprintf(stderr, "wasmer instantiate error\n");
      for (int i = 0; i < 5; i++) { wasm_func_delete(funcs[i]); wasm_functype_delete(fts[i]); }
      wasm_module_delete(module);
      wasm_store_delete(store);
      wasm_engine_delete(engine);
      return -1;
   }

   wasm_extern_vec_t exports;
   wasm_instance_exports(instance, &exports);

   // Exports are in order: bench_identity(0), bench_accumulate(1), bench_mix(2),
   // bench_mem_op(3), bench_multi(4), bench_mixed(5), memory(6)
   const char* export_names[] = {"bench_identity", "bench_accumulate", "bench_mix", "bench_mem_op", "bench_multi", "bench_mixed"};
   int func_idx = -1;
   for (int i = 0; i < 6; i++) {
      if (strcmp(export_names[i], func) == 0) { func_idx = i; break; }
   }
   if (func_idx < 0 || static_cast<size_t>(func_idx) >= exports.size) {
      fprintf(stderr, "wasmer: export %s not found\n", func);
      wasm_extern_vec_delete(&exports);
      wasm_instance_delete(instance);
      for (int i = 0; i < 5; i++) { wasm_func_delete(funcs[i]); wasm_functype_delete(fts[i]); }
      wasm_module_delete(module);
      wasm_store_delete(store);
      wasm_engine_delete(engine);
      return -1;
   }

   const wasm_func_t* run_func = wasm_extern_as_func(exports.data[func_idx]);

   wasm_val_t arg = WASM_I32_VAL(static_cast<int32_t>(n));
   wasm_val_t res;
   wasm_val_vec_t args_vec = {1, &arg};
   wasm_val_vec_t res_vec = {1, &res};

   auto t1 = std::chrono::high_resolution_clock::now();
   wasm_trap_t* trap = wasm_func_call(run_func, &args_vec, &res_vec);
   auto t2 = std::chrono::high_resolution_clock::now();
   if (trap) { fprintf(stderr, "wasmer call error\n"); wasm_trap_delete(trap); }

   wasm_extern_vec_delete(&exports);
   wasm_instance_delete(instance);
   for (int i = 0; i < 5; i++) { wasm_func_delete(funcs[i]); wasm_functype_delete(fts[i]); }
   wasm_module_delete(module);
   wasm_store_delete(store);
   wasm_engine_delete(engine);
   return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

double run_wasmer_compute(const std::vector<uint8_t>& wasm, const char* func, uint32_t n) {
   wasm_engine_t* engine = wasm_engine_new();
   wasm_store_t* store = wasm_store_new(engine);

   wasm_byte_vec_t binary = {wasm.size(), const_cast<wasm_byte_t*>(reinterpret_cast<const wasm_byte_t*>(wasm.data()))};
   wasm_module_t* module = wasm_module_new(store, &binary);
   if (!module) { fprintf(stderr, "wasmer compute module error\n"); wasm_store_delete(store); wasm_engine_delete(engine); return -1; }

   wasm_extern_vec_t imports = {0, nullptr};
   wasm_instance_t* instance = wasm_instance_new(store, module, &imports, nullptr);
   if (!instance) {
      fprintf(stderr, "wasmer compute instantiate error\n");
      wasm_module_delete(module);
      wasm_store_delete(store);
      wasm_engine_delete(engine);
      return -1;
   }

   wasm_extern_vec_t exports;
   wasm_instance_exports(instance, &exports);

   // Find the function by iterating exports
   wasm_exporttype_vec_t export_types;
   wasm_module_exports(module, &export_types);

   int func_idx = -1;
   for (size_t i = 0; i < export_types.size; i++) {
      const wasm_name_t* name = wasm_exporttype_name(export_types.data[i]);
      if (name->size == strlen(func) && memcmp(name->data, func, name->size) == 0) {
         func_idx = static_cast<int>(i);
         break;
      }
   }
   wasm_exporttype_vec_delete(&export_types);

   if (func_idx < 0 || static_cast<size_t>(func_idx) >= exports.size) {
      fprintf(stderr, "wasmer compute: export %s not found\n", func);
      wasm_extern_vec_delete(&exports);
      wasm_instance_delete(instance);
      wasm_module_delete(module);
      wasm_store_delete(store);
      wasm_engine_delete(engine);
      return -1;
   }

   const wasm_func_t* run_func = wasm_extern_as_func(exports.data[func_idx]);

   wasm_val_t arg = WASM_I32_VAL(static_cast<int32_t>(n));
   wasm_val_t res;
   wasm_val_vec_t args_vec = {1, &arg};
   wasm_val_vec_t res_vec = {1, &res};

   auto t1 = std::chrono::high_resolution_clock::now();
   wasm_trap_t* trap = wasm_func_call(run_func, &args_vec, &res_vec);
   auto t2 = std::chrono::high_resolution_clock::now();
   if (trap) { fprintf(stderr, "wasmer compute call error\n"); wasm_trap_delete(trap); }

   wasm_extern_vec_delete(&exports);
   wasm_instance_delete(instance);
   wasm_module_delete(module);
   wasm_store_delete(store);
   wasm_engine_delete(engine);
   return std::chrono::duration<double, std::milli>(t2 - t1).count();
}
#endif
