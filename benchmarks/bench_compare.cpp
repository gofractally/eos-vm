// Comparative benchmark: eos-vm vs other WASM runtimes
// Focuses on host function call overhead — the dominant cost in blockchain-style workloads.
//
// Build with cmake options:
//   -DENABLE_BENCH_WASM3=ON    (fetches wasm3 source)
//   -DENABLE_BENCH_WAMR=ON     (fetches WAMR source)
//   -DENABLE_BENCH_WASMTIME=ON (downloads prebuilt wasmtime)
//   -DENABLE_BENCH_WASMER=ON   (downloads prebuilt wasmer)

#include <eosio/vm/backend.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifdef BENCH_HAS_WASM3
#include <wasm3.h>
#include <m3_env.h>
#endif

#ifdef BENCH_HAS_WAMR
#include <wasm_export.h>
#endif

#ifdef BENCH_HAS_WASMTIME
#include <wasmtime.h>
#endif

#ifdef BENCH_HAS_WASMER
#include <wasmer.h>
#endif

using namespace eosio;
using namespace eosio::vm;

// ============================================================================
// Host functions — simulate psibase-style workloads
// ============================================================================

// Trivial: pure call overhead (like getResult with empty result)
static int32_t host_identity(int32_t x) { return x; }

// Accumulator: void return with one arg (like writeConsole length tracking)
static uint64_t g_accumulator = 0;
static void host_accumulate(int64_t val) { g_accumulator += static_cast<uint64_t>(val); }

// Hash mixing: 2 args -> 1 result (like cryptographic operations)
static int64_t host_mix(int64_t a, int64_t b) {
   uint64_t ua = static_cast<uint64_t>(a);
   uint64_t ub = static_cast<uint64_t>(b);
   ua ^= ub;
   ua ^= ua >> 33;
   ua *= 0xff51afd7ed558ccdULL;
   ua ^= ua >> 33;
   ua *= 0xc4ceb9fe1a85ec53ULL;
   ua ^= ua >> 33;
   return static_cast<int64_t>(ua);
}

// Memory read: 2 i32 args simulating (ptr, len) — like kvGet/kvPut with span
static uint64_t g_mem_sum = 0;
static void host_mem_op(int32_t ptr, int32_t len) {
   g_mem_sum += static_cast<uint64_t>(ptr) + static_cast<uint64_t>(len);
}

// Multi-arg: 4 args simulating a complex host call (like call(service, sender, data_ptr, data_len))
static uint64_t g_multi_sum = 0;
static int64_t host_multi(int64_t a, int64_t b, int32_t c, int32_t d) {
   g_multi_sum += static_cast<uint64_t>(a) + static_cast<uint64_t>(b);
   return static_cast<int64_t>(c + d);
}

// ============================================================================
// WASM binary builder
// ============================================================================
// Builds a module with 5 imported host functions and 5 exported benchmark
// functions. Each benchmark loops N times calling the corresponding host
// function, accumulating results to prevent dead-code elimination.

static std::vector<uint8_t> build_wasm() {
   std::vector<uint8_t> w;
   auto emit = [&](auto... bytes) { (w.push_back(static_cast<uint8_t>(bytes)), ...); };
   auto emit_u32 = [&](uint32_t v) {
      do { uint8_t b = v & 0x7f; v >>= 7; if (v) b |= 0x80; w.push_back(b); } while (v);
   };
   auto emit_str = [&](const char* s) {
      uint32_t len = static_cast<uint32_t>(strlen(s));
      emit_u32(len);
      for (uint32_t i = 0; i < len; i++) w.push_back(static_cast<uint8_t>(s[i]));
   };
   auto section = [&](uint8_t id, auto fn) {
      w.push_back(id);
      size_t size_pos = w.size();
      w.push_back(0);
      size_t start = w.size();
      fn();
      uint32_t size = static_cast<uint32_t>(w.size() - start);
      if (size < 128) {
         w[size_pos] = static_cast<uint8_t>(size);
      } else {
         std::vector<uint8_t> leb;
         uint32_t s = size;
         do { uint8_t b = s & 0x7f; s >>= 7; if (s) b |= 0x80; leb.push_back(b); } while (s);
         w.erase(w.begin() + static_cast<std::ptrdiff_t>(size_pos),
                 w.begin() + static_cast<std::ptrdiff_t>(size_pos) + 1);
         w.insert(w.begin() + static_cast<std::ptrdiff_t>(size_pos), leb.begin(), leb.end());
      }
   };

   // Magic + version
   emit(0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00);

   // Type section
   section(1, [&]() {
      emit_u32(6); // 6 types
      // 0: (i32) -> i32        identity
      emit(0x60); emit_u32(1); emit(0x7f); emit_u32(1); emit(0x7f);
      // 1: (i64) -> ()         accumulate
      emit(0x60); emit_u32(1); emit(0x7e); emit_u32(0);
      // 2: (i64, i64) -> i64   mix
      emit(0x60); emit_u32(2); emit(0x7e, 0x7e); emit_u32(1); emit(0x7e);
      // 3: (i32, i32) -> ()    mem_op
      emit(0x60); emit_u32(2); emit(0x7f, 0x7f); emit_u32(0);
      // 4: (i64, i64, i32, i32) -> i64   multi
      emit(0x60); emit_u32(4); emit(0x7e, 0x7e, 0x7f, 0x7f); emit_u32(1); emit(0x7e);
      // 5: (i32) -> i64        bench entry point
      emit(0x60); emit_u32(1); emit(0x7f); emit_u32(1); emit(0x7e);
   });

   // Import section — 5 host functions
   section(2, [&]() {
      emit_u32(5);
      emit_str("env"); emit_str("identity");   emit(0x00); emit_u32(0);
      emit_str("env"); emit_str("accumulate");  emit(0x00); emit_u32(1);
      emit_str("env"); emit_str("mix");         emit(0x00); emit_u32(2);
      emit_str("env"); emit_str("mem_op");      emit(0x00); emit_u32(3);
      emit_str("env"); emit_str("multi");       emit(0x00); emit_u32(4);
   });

   // Function section — 6 exported functions (indices 5-10)
   section(3, [&]() {
      emit_u32(6);
      for (int i = 0; i < 6; i++) emit_u32(5); // all type 5: (i32) -> i64
   });

   // Memory section — 1 page
   section(5, [&]() {
      emit_u32(1);  // 1 memory
      emit(0x00);   // no max
      emit_u32(1);  // 1 initial page
   });

   // Export section
   section(7, [&]() {
      emit_u32(7); // 6 functions + 1 memory
      emit_str("bench_identity");   emit(0x00); emit_u32(5);
      emit_str("bench_accumulate"); emit(0x00); emit_u32(6);
      emit_str("bench_mix");        emit(0x00); emit_u32(7);
      emit_str("bench_mem_op");     emit(0x00); emit_u32(8);
      emit_str("bench_multi");      emit(0x00); emit_u32(9);
      emit_str("bench_mixed");      emit(0x00); emit_u32(10);
      emit_str("memory");           emit(0x02); emit_u32(0);
   });

   // Code section — 6 function bodies
   section(10, [&]() {
      emit_u32(6);

      // Helper: emit a loop body that calls a function N times
      // Each bench function: param $n: i32, local $i: i32, local $acc: i64

      // --- bench_identity: loop calling identity(i), sum results ---
      // func 5
      {
         size_t bp = w.size(); w.push_back(0); size_t cs = w.size();
         emit_u32(2); emit_u32(1); emit(0x7f); emit_u32(1); emit(0x7e); // locals: i32, i64
         emit(0x03, 0x40); // loop
           emit(0x20); emit_u32(1); // local.get $i
           emit(0x10); emit_u32(0); // call identity
           emit(0xAD);              // i64.extend_i32_u
           emit(0x20); emit_u32(2); // local.get $acc
           emit(0x7C);              // i64.add
           emit(0x21); emit_u32(2); // local.set $acc
           // i++, loop if < n
           emit(0x20); emit_u32(1); emit(0x41, 0x01); emit(0x6A); emit(0x22); emit_u32(1);
           emit(0x20); emit_u32(0); emit(0x49); emit(0x0D, 0x00);
         emit(0x0B);
         emit(0x20); emit_u32(2); emit(0x0B); // return acc
         uint32_t sz = static_cast<uint32_t>(w.size() - cs);
         w[bp] = static_cast<uint8_t>(sz);
      }

      // --- bench_accumulate: loop calling accumulate(i64(i)) ---
      // func 6
      {
         size_t bp = w.size(); w.push_back(0); size_t cs = w.size();
         emit_u32(1); emit_u32(1); emit(0x7f); // local: i32
         emit(0x03, 0x40);
           emit(0x20); emit_u32(1); emit(0xAD); // local.get $i, extend
           emit(0x10); emit_u32(1); // call accumulate
           emit(0x20); emit_u32(1); emit(0x41, 0x01); emit(0x6A); emit(0x22); emit_u32(1);
           emit(0x20); emit_u32(0); emit(0x49); emit(0x0D, 0x00);
         emit(0x0B);
         emit(0x42, 0x00); emit(0x0B); // return 0
         uint32_t sz = static_cast<uint32_t>(w.size() - cs);
         w[bp] = static_cast<uint8_t>(sz);
      }

      // --- bench_mix: loop calling mix(acc, i64(i)), chain results ---
      // func 7
      {
         size_t bp = w.size(); w.push_back(0); size_t cs = w.size();
         emit_u32(2); emit_u32(1); emit(0x7f); emit_u32(1); emit(0x7e);
         emit(0x42, 0x01); emit(0x21); emit_u32(2); // acc = 1
         emit(0x03, 0x40);
           emit(0x20); emit_u32(2); // acc
           emit(0x20); emit_u32(1); emit(0xAD); // i64(i)
           emit(0x10); emit_u32(2); // call mix
           emit(0x21); emit_u32(2); // acc = result
           emit(0x20); emit_u32(1); emit(0x41, 0x01); emit(0x6A); emit(0x22); emit_u32(1);
           emit(0x20); emit_u32(0); emit(0x49); emit(0x0D, 0x00);
         emit(0x0B);
         emit(0x20); emit_u32(2); emit(0x0B);
         uint32_t sz = static_cast<uint32_t>(w.size() - cs);
         w[bp] = static_cast<uint8_t>(sz);
      }

      // --- bench_mem_op: loop calling mem_op(i*4, 16) — simulates span(ptr,len) pattern ---
      // func 8
      {
         size_t bp = w.size(); w.push_back(0); size_t cs = w.size();
         emit_u32(1); emit_u32(1); emit(0x7f);
         emit(0x03, 0x40);
           // ptr = i * 4 (shift left 2)
           emit(0x20); emit_u32(1); // i
           emit(0x41, 0x02); // const 2
           emit(0x74); // i32.shl
           // len = 16
           emit(0x41, 0x10); // const 16
           emit(0x10); emit_u32(3); // call mem_op
           emit(0x20); emit_u32(1); emit(0x41, 0x01); emit(0x6A); emit(0x22); emit_u32(1);
           emit(0x20); emit_u32(0); emit(0x49); emit(0x0D, 0x00);
         emit(0x0B);
         emit(0x42, 0x00); emit(0x0B);
         uint32_t sz = static_cast<uint32_t>(w.size() - cs);
         w[bp] = static_cast<uint8_t>(sz);
      }

      // --- bench_multi: loop calling multi(acc, i64(i), i, 42) ---
      // func 9
      {
         size_t bp = w.size(); w.push_back(0); size_t cs = w.size();
         emit_u32(2); emit_u32(1); emit(0x7f); emit_u32(1); emit(0x7e);
         emit(0x42, 0x01); emit(0x21); emit_u32(2); // acc = 1
         emit(0x03, 0x40);
           emit(0x20); emit_u32(2); // acc
           emit(0x20); emit_u32(1); emit(0xAD); // i64(i)
           emit(0x20); emit_u32(1); // i (as i32)
           emit(0x41, 0x2A); // const 42
           emit(0x10); emit_u32(4); // call multi
           emit(0x21); emit_u32(2); // acc = result
           emit(0x20); emit_u32(1); emit(0x41, 0x01); emit(0x6A); emit(0x22); emit_u32(1);
           emit(0x20); emit_u32(0); emit(0x49); emit(0x0D, 0x00);
         emit(0x0B);
         emit(0x20); emit_u32(2); emit(0x0B);
         uint32_t sz = static_cast<uint32_t>(w.size() - cs);
         w[bp] = static_cast<uint8_t>(sz);
      }

      // --- bench_mixed: interleaves all host calls per iteration (realistic workload) ---
      // func 10
      {
         size_t bp = w.size(); w.push_back(0); size_t cs = w.size();
         emit_u32(2); emit_u32(1); emit(0x7f); emit_u32(1); emit(0x7e);
         emit(0x42, 0x01); emit(0x21); emit_u32(2); // acc = 1
         emit(0x03, 0x40);
           // 1. identity(i) -> extend -> add to acc
           emit(0x20); emit_u32(1); emit(0x10); emit_u32(0); emit(0xAD);
           emit(0x20); emit_u32(2); emit(0x7C); emit(0x21); emit_u32(2);
           // 2. accumulate(acc)
           emit(0x20); emit_u32(2); emit(0x10); emit_u32(1);
           // 3. acc = mix(acc, i64(i))
           emit(0x20); emit_u32(2); emit(0x20); emit_u32(1); emit(0xAD);
           emit(0x10); emit_u32(2); emit(0x21); emit_u32(2);
           // 4. mem_op(i*4, 16)
           emit(0x20); emit_u32(1); emit(0x41, 0x02); emit(0x74);
           emit(0x41, 0x10); emit(0x10); emit_u32(3);
           // 5. acc += multi(acc, i64(i), i, 1)
           emit(0x20); emit_u32(2); emit(0x20); emit_u32(1); emit(0xAD);
           emit(0x20); emit_u32(1); emit(0x41, 0x01);
           emit(0x10); emit_u32(4);
           emit(0x20); emit_u32(2); emit(0x7C); emit(0x21); emit_u32(2);
           // i++, loop
           emit(0x20); emit_u32(1); emit(0x41, 0x01); emit(0x6A); emit(0x22); emit_u32(1);
           emit(0x20); emit_u32(0); emit(0x49); emit(0x0D, 0x00);
         emit(0x0B);
         emit(0x20); emit_u32(2); emit(0x0B);
         uint32_t sz = static_cast<uint32_t>(w.size() - cs);
         w[bp] = static_cast<uint8_t>(sz);
      }
   });

   return w;
}

// ============================================================================
// eos-vm runners
// ============================================================================

using rhf_t = registered_host_functions<standalone_function_t>;

static void register_eosvm_hosts() {
   static bool done = false;
   if (done) return;
   rhf_t::add<&host_identity>("env", "identity");
   rhf_t::add<&host_accumulate>("env", "accumulate");
   rhf_t::add<&host_mix>("env", "mix");
   rhf_t::add<&host_mem_op>("env", "mem_op");
   rhf_t::add<&host_multi>("env", "multi");
   done = true;
}

template<typename Impl>
static double run_eosvm(wasm_code& code, wasm_allocator& wa, const char* func, uint32_t n) {
   using backend_t = eosio::vm::backend<rhf_t, Impl>;
   backend_t bkend(code, &wa);
   rhf_t::resolve(bkend.get_module());
   bkend.initialize(nullptr);

   auto t1 = std::chrono::high_resolution_clock::now();
   bkend.call_with_return("env", func, n);
   auto t2 = std::chrono::high_resolution_clock::now();
   return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

// ============================================================================
// wasm3 runner
// ============================================================================
#ifdef BENCH_HAS_WASM3

// wasm3 host function trampolines (raw API)
static m3ApiRawFunction(w3_identity) {
   m3ApiReturnType(int32_t);
   m3ApiGetArg(int32_t, x);
   m3ApiReturn(host_identity(x));
}

static m3ApiRawFunction(w3_accumulate) {
   m3ApiGetArg(int64_t, val);
   host_accumulate(val);
   m3ApiSuccess();
}

static m3ApiRawFunction(w3_mix) {
   m3ApiReturnType(int64_t);
   m3ApiGetArg(int64_t, a);
   m3ApiGetArg(int64_t, b);
   m3ApiReturn(host_mix(a, b));
}

static m3ApiRawFunction(w3_mem_op) {
   m3ApiGetArg(int32_t, ptr);
   m3ApiGetArg(int32_t, len);
   host_mem_op(ptr, len);
   m3ApiSuccess();
}

static m3ApiRawFunction(w3_multi) {
   m3ApiReturnType(int64_t);
   m3ApiGetArg(int64_t, a);
   m3ApiGetArg(int64_t, b);
   m3ApiGetArg(int32_t, c);
   m3ApiGetArg(int32_t, d);
   m3ApiReturn(host_multi(a, b, c, d));
}

static double run_wasm3(const std::vector<uint8_t>& wasm, const char* func, uint32_t n) {
   IM3Environment env = m3_NewEnvironment();
   IM3Runtime runtime = m3_NewRuntime(env, 64 * 1024, nullptr);

   IM3Module module = nullptr;
   M3Result r = m3_ParseModule(env, &module, wasm.data(), static_cast<uint32_t>(wasm.size()));
   if (r) { fprintf(stderr, "wasm3 parse error: %s\n", r); return -1; }
   r = m3_LoadModule(runtime, module);
   if (r) { fprintf(stderr, "wasm3 load error: %s\n", r); return -1; }

   m3_LinkRawFunction(module, "env", "identity",   "i(i)",    w3_identity);
   m3_LinkRawFunction(module, "env", "accumulate", "v(I)",    w3_accumulate);
   m3_LinkRawFunction(module, "env", "mix",        "I(II)",   w3_mix);
   m3_LinkRawFunction(module, "env", "mem_op",     "v(ii)",   w3_mem_op);
   m3_LinkRawFunction(module, "env", "multi",      "I(IIii)", w3_multi);

   IM3Function f = nullptr;
   r = m3_FindFunction(&f, runtime, func);
   if (r) { fprintf(stderr, "wasm3 find %s: %s\n", func, r); m3_FreeRuntime(runtime); m3_FreeEnvironment(env); return -1; }

   auto t1 = std::chrono::high_resolution_clock::now();
   r = m3_CallV(f, (int32_t)n);
   auto t2 = std::chrono::high_resolution_clock::now();
   if (r) { fprintf(stderr, "wasm3 call error: %s\n", r); }

   m3_FreeRuntime(runtime);
   m3_FreeEnvironment(env);
   return std::chrono::duration<double, std::milli>(t2 - t1).count();
}
#endif

// ============================================================================
// WAMR runner
// ============================================================================
#ifdef BENCH_HAS_WAMR

static int32_t wamr_identity(wasm_exec_env_t, int32_t x) { return host_identity(x); }
static void wamr_accumulate(wasm_exec_env_t, int64_t v) { host_accumulate(v); }
static int64_t wamr_mix(wasm_exec_env_t, int64_t a, int64_t b) { return host_mix(a, b); }
static void wamr_mem_op(wasm_exec_env_t, int32_t p, int32_t l) { host_mem_op(p, l); }
static int64_t wamr_multi(wasm_exec_env_t, int64_t a, int64_t b, int32_t c, int32_t d) { return host_multi(a, b, c, d); }

static NativeSymbol wamr_natives[] = {
   {"identity",   (void*)wamr_identity,   "(i)i",    nullptr},
   {"accumulate", (void*)wamr_accumulate, "(I)",     nullptr},
   {"mix",        (void*)wamr_mix,        "(II)I",   nullptr},
   {"mem_op",     (void*)wamr_mem_op,     "(ii)",    nullptr},
   {"multi",      (void*)wamr_multi,      "(IIii)I", nullptr},
};

static double run_wamr(const std::vector<uint8_t>& wasm, const char* func, uint32_t n) {
   static bool inited = false;
   if (!inited) {
      wasm_runtime_init();
      wasm_runtime_register_natives("env", wamr_natives, sizeof(wamr_natives)/sizeof(wamr_natives[0]));
      inited = true;
   }

   char error_buf[256];
   wasm_module_t module = wasm_runtime_load(const_cast<uint8_t*>(wasm.data()),
                                            static_cast<uint32_t>(wasm.size()),
                                            error_buf, sizeof(error_buf));
   if (!module) { fprintf(stderr, "WAMR load error: %s\n", error_buf); return -1; }

   wasm_module_inst_t inst = wasm_runtime_instantiate(module, 64*1024, 64*1024,
                                                       error_buf, sizeof(error_buf));
   if (!inst) { fprintf(stderr, "WAMR inst error: %s\n", error_buf); wasm_runtime_unload(module); return -1; }

   wasm_exec_env_t exec_env = wasm_runtime_create_exec_env(inst, 64*1024);
   if (!exec_env) { fprintf(stderr, "WAMR exec_env error\n"); wasm_runtime_deinstantiate(inst); wasm_runtime_unload(module); return -1; }

   wasm_function_inst_t f = wasm_runtime_lookup_function(inst, func);
   if (!f) { fprintf(stderr, "WAMR: function %s not found\n", func); wasm_runtime_destroy_exec_env(exec_env); wasm_runtime_deinstantiate(inst); wasm_runtime_unload(module); return -1; }

   uint32_t argv[2] = {n, 0};

   auto t1 = std::chrono::high_resolution_clock::now();
   bool ok = wasm_runtime_call_wasm(exec_env, f, 1, argv);
   auto t2 = std::chrono::high_resolution_clock::now();
   if (!ok) { fprintf(stderr, "WAMR call error: %s\n", wasm_runtime_get_exception(inst)); }

   wasm_runtime_destroy_exec_env(exec_env);
   wasm_runtime_deinstantiate(inst);
   wasm_runtime_unload(module);
   return std::chrono::duration<double, std::milli>(t2 - t1).count();
}
#endif

// ============================================================================
// wasmtime runner
// ============================================================================
#ifdef BENCH_HAS_WASMTIME

static wasm_trap_t* wt_identity(void*, wasmtime_caller_t*, const wasmtime_val_t* args, size_t, wasmtime_val_t* results, size_t) {
   results[0].kind = WASMTIME_I32;
   results[0].of.i32 = host_identity(args[0].of.i32);
   return nullptr;
}
static wasm_trap_t* wt_accumulate(void*, wasmtime_caller_t*, const wasmtime_val_t* args, size_t, wasmtime_val_t*, size_t) {
   host_accumulate(args[0].of.i64);
   return nullptr;
}
static wasm_trap_t* wt_mix(void*, wasmtime_caller_t*, const wasmtime_val_t* args, size_t, wasmtime_val_t* results, size_t) {
   results[0].kind = WASMTIME_I64;
   results[0].of.i64 = host_mix(args[0].of.i64, args[1].of.i64);
   return nullptr;
}
static wasm_trap_t* wt_mem_op(void*, wasmtime_caller_t*, const wasmtime_val_t* args, size_t, wasmtime_val_t*, size_t) {
   host_mem_op(args[0].of.i32, args[1].of.i32);
   return nullptr;
}
static wasm_trap_t* wt_multi(void*, wasmtime_caller_t*, const wasmtime_val_t* args, size_t, wasmtime_val_t* results, size_t) {
   results[0].kind = WASMTIME_I64;
   results[0].of.i64 = host_multi(args[0].of.i64, args[1].of.i64, args[2].of.i32, args[3].of.i32);
   return nullptr;
}

static double run_wasmtime(const std::vector<uint8_t>& wasm, const char* func, uint32_t n) {
   wasm_engine_t* engine = wasm_engine_new();
   wasmtime_store_t* store = wasmtime_store_new(engine, nullptr, nullptr);
   wasmtime_context_t* ctx = wasmtime_store_context(store);

   wasmtime_module_t* module = nullptr;
   wasmtime_error_t* err = wasmtime_module_new(engine, wasm.data(), wasm.size(), &module);
   if (err) { fprintf(stderr, "wasmtime module error\n"); wasmtime_error_delete(err); wasmtime_store_delete(store); wasm_engine_delete(engine); return -1; }

   // Define host functions
   struct { const char* name; wasmtime_func_callback_t cb; wasm_valtype_vec_t params; wasm_valtype_vec_t results; } defs[] = {
      {"identity",   wt_identity,   {}, {}},
      {"accumulate", wt_accumulate, {}, {}},
      {"mix",        wt_mix,        {}, {}},
      {"mem_op",     wt_mem_op,     {}, {}},
      {"multi",      wt_multi,      {}, {}},
   };

   // Build func types
   wasm_valtype_t* i32_t = wasm_valtype_new(WASM_I32);
   wasm_valtype_t* i64_t = wasm_valtype_new(WASM_I64);

   // identity: (i32) -> i32
   wasm_valtype_t* id_p[] = {wasm_valtype_new(WASM_I32)};
   wasm_valtype_t* id_r[] = {wasm_valtype_new(WASM_I32)};
   wasm_valtype_vec_t id_pv = {1, id_p}, id_rv = {1, id_r};
   wasm_functype_t* ft_id = wasm_functype_new(&id_pv, &id_rv);

   // accumulate: (i64) -> ()
   wasm_valtype_t* ac_p[] = {wasm_valtype_new(WASM_I64)};
   wasm_valtype_vec_t ac_pv = {1, ac_p}, ac_rv = {0, nullptr};
   wasm_functype_t* ft_ac = wasm_functype_new(&ac_pv, &ac_rv);

   // mix: (i64, i64) -> i64
   wasm_valtype_t* mx_p[] = {wasm_valtype_new(WASM_I64), wasm_valtype_new(WASM_I64)};
   wasm_valtype_t* mx_r[] = {wasm_valtype_new(WASM_I64)};
   wasm_valtype_vec_t mx_pv = {2, mx_p}, mx_rv = {1, mx_r};
   wasm_functype_t* ft_mx = wasm_functype_new(&mx_pv, &mx_rv);

   // mem_op: (i32, i32) -> ()
   wasm_valtype_t* mo_p[] = {wasm_valtype_new(WASM_I32), wasm_valtype_new(WASM_I32)};
   wasm_valtype_vec_t mo_pv = {2, mo_p}, mo_rv = {0, nullptr};
   wasm_functype_t* ft_mo = wasm_functype_new(&mo_pv, &mo_rv);

   // multi: (i64, i64, i32, i32) -> i64
   wasm_valtype_t* mu_p[] = {wasm_valtype_new(WASM_I64), wasm_valtype_new(WASM_I64), wasm_valtype_new(WASM_I32), wasm_valtype_new(WASM_I32)};
   wasm_valtype_t* mu_r[] = {wasm_valtype_new(WASM_I64)};
   wasm_valtype_vec_t mu_pv = {4, mu_p}, mu_rv = {1, mu_r};
   wasm_functype_t* ft_mu = wasm_functype_new(&mu_pv, &mu_rv);

   wasm_functype_t* ftypes[] = {ft_id, ft_ac, ft_mx, ft_mo, ft_mu};
   wasmtime_func_callback_t cbs[] = {wt_identity, wt_accumulate, wt_mix, wt_mem_op, wt_multi};
   const char* names[] = {"identity", "accumulate", "mix", "mem_op", "multi"};

   wasmtime_extern_t imports[5];
   for (int i = 0; i < 5; i++) {
      wasmtime_func_new(ctx, ftypes[i], cbs[i], nullptr, nullptr, &imports[i].of.func);
      imports[i].kind = WASMTIME_EXTERN_FUNC;
   }

   wasmtime_instance_t instance;
   wasm_trap_t* trap = nullptr;
   err = wasmtime_instance_new(ctx, module, imports, 5, &instance, &trap);
   if (err || trap) {
      fprintf(stderr, "wasmtime instantiate error\n");
      if (err) wasmtime_error_delete(err);
      if (trap) wasm_trap_delete(trap);
      for (int i = 0; i < 5; i++) wasm_functype_delete(ftypes[i]);
      wasmtime_module_delete(module);
      wasmtime_store_delete(store);
      wasm_engine_delete(engine);
      return -1;
   }

   // Find export
   wasmtime_extern_t exp;
   bool found = wasmtime_instance_export_get(ctx, &instance, func, strlen(func), &exp);
   if (!found || exp.kind != WASMTIME_EXTERN_FUNC) {
      fprintf(stderr, "wasmtime: export %s not found\n", func);
      for (int i = 0; i < 5; i++) wasm_functype_delete(ftypes[i]);
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

   for (int i = 0; i < 5; i++) wasm_functype_delete(ftypes[i]);
   wasmtime_module_delete(module);
   wasmtime_store_delete(store);
   wasm_engine_delete(engine);
   wasm_valtype_delete(i32_t);
   wasm_valtype_delete(i64_t);
   return std::chrono::duration<double, std::milli>(t2 - t1).count();
}
#endif

// ============================================================================
// wasmer runner
// ============================================================================
#ifdef BENCH_HAS_WASMER

static wasm_trap_t* ws_identity(void*, const wasm_val_vec_t* args, wasm_val_vec_t* results) {
   results->data[0] = WASM_I32_VAL(host_identity(args->data[0].of.i32));
   return nullptr;
}
static wasm_trap_t* ws_accumulate(void*, const wasm_val_vec_t* args, wasm_val_vec_t*) {
   host_accumulate(args->data[0].of.i64);
   return nullptr;
}
static wasm_trap_t* ws_mix(void*, const wasm_val_vec_t* args, wasm_val_vec_t* results) {
   results->data[0] = WASM_I64_VAL(host_mix(args->data[0].of.i64, args->data[1].of.i64));
   return nullptr;
}
static wasm_trap_t* ws_mem_op(void*, const wasm_val_vec_t* args, wasm_val_vec_t*) {
   host_mem_op(args->data[0].of.i32, args->data[1].of.i32);
   return nullptr;
}
static wasm_trap_t* ws_multi(void*, const wasm_val_vec_t* args, wasm_val_vec_t* results) {
   results->data[0] = WASM_I64_VAL(host_multi(args->data[0].of.i64, args->data[1].of.i64,
                                               args->data[2].of.i32, args->data[3].of.i32));
   return nullptr;
}

static double run_wasmer(const std::vector<uint8_t>& wasm, const char* func, uint32_t n) {
   wasm_engine_t* engine = wasm_engine_new();
   wasm_store_t* store = wasm_store_new(engine);

   wasm_byte_vec_t binary = {wasm.size(), const_cast<wasm_byte_t*>(reinterpret_cast<const wasm_byte_t*>(wasm.data()))};
   wasm_module_t* module = wasm_module_new(store, &binary);
   if (!module) { fprintf(stderr, "wasmer module error\n"); wasm_store_delete(store); wasm_engine_delete(engine); return -1; }

   // Build function types and host funcs
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

   // Find the exported function by name — exports are in order, find by matching
   // We export: bench_identity(0), bench_accumulate(1), bench_mix(2), bench_mem_op(3), bench_multi(4), bench_mixed(5), memory(6)
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
#endif

// ============================================================================
// Main
// ============================================================================

int main() {
   register_eosvm_hosts();

   auto wasm_bytes = build_wasm();
   wasm_code code(wasm_bytes.begin(), wasm_bytes.end());
   wasm_allocator wa;

   const uint32_t N = 10'000'000;

   struct bench_def {
      const char* label;
      const char* func;
      uint32_t    calls_per_iter; // host calls per loop iteration
   };

   bench_def benches[] = {
      {"identity (1 call)",     "bench_identity",   1},
      {"accumulate (1 call)",   "bench_accumulate",  1},
      {"mix (1 call)",          "bench_mix",         1},
      {"mem_op (ptr,len)",      "bench_mem_op",      1},
      {"multi (4 args)",        "bench_multi",       1},
      {"mixed (5 calls/iter)",  "bench_mixed",       5},
   };

   // Column headers
   printf("Host-call benchmark — %u iterations per test\n", N);
   printf("Each test loops N times; \"mixed\" makes 5 host calls per iteration.\n\n");

   printf("%-24s %10s", "Test", "Interp");
#if defined(__x86_64__) || defined(__aarch64__)
   printf(" %10s %8s", "JIT", "Speedup");
#endif
#ifdef BENCH_HAS_WASM3
   printf(" %10s", "wasm3");
#endif
#ifdef BENCH_HAS_WAMR
   printf(" %10s", "WAMR");
#endif
#ifdef BENCH_HAS_WASMTIME
   printf(" %10s", "wasmtime");
#endif
#ifdef BENCH_HAS_WASMER
   printf(" %10s", "wasmer");
#endif
   printf("   (all times in ms)\n");
   printf("%s\n", std::string(120, '-').c_str());

   for (auto& b : benches) {
      double interp_ms = run_eosvm<interpreter>(code, wa, b.func, N);
      printf("%-24s %10.1f", b.label, interp_ms);

#if defined(__x86_64__) || defined(__aarch64__)
      double jit_ms = run_eosvm<jit>(code, wa, b.func, N);
      printf(" %10.1f %7.1fx", jit_ms, interp_ms / jit_ms);
#endif

#ifdef BENCH_HAS_WASM3
      double w3_ms = run_wasm3(wasm_bytes, b.func, N);
      printf(" %10.1f", w3_ms);
#endif

#ifdef BENCH_HAS_WAMR
      double wamr_ms = run_wamr(wasm_bytes, b.func, N);
      printf(" %10.1f", wamr_ms);
#endif

#ifdef BENCH_HAS_WASMTIME
      double wt_ms = run_wasmtime(wasm_bytes, b.func, N);
      printf(" %10.1f", wt_ms);
#endif

#ifdef BENCH_HAS_WASMER
      double ws_ms = run_wasmer(wasm_bytes, b.func, N);
      printf(" %10.1f", ws_ms);
#endif

      printf("\n");
   }

   printf("\n");

   // Summary: total host calls
   uint64_t total_calls = 0;
   for (auto& b : benches) total_calls += static_cast<uint64_t>(N) * b.calls_per_iter;
   printf("Total host calls across all tests: %llu\n", (unsigned long long)total_calls);

   return 0;
}
