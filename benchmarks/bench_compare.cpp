// Comparative benchmark: eos-vm vs other WASM runtimes
// Focuses on host function call overhead — the dominant cost in blockchain-style workloads.
//
// Build with cmake options:
//   -DENABLE_BENCH_WASM3=ON    (fetches wasm3 source)
//   -DENABLE_BENCH_WAMR=ON     (fetches WAMR source)
//   -DENABLE_BENCH_WASMTIME=ON (downloads prebuilt wasmtime)
//   -DENABLE_BENCH_WASMER=ON   (downloads prebuilt wasmer)

#include "bench_hosts.hpp"
#include <eosio/vm/backend.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#ifdef BENCH_HAS_COMPUTE
#include <eosio/vm/utils.hpp>

// Native baseline — same C code compiled natively
extern "C" {
   int64_t bench_sha256(int32_t iterations);
   int64_t bench_ecdsa_verify(int32_t iterations);
   int64_t bench_ecdsa_sign(int32_t iterations);
   int64_t bench_fib(int32_t n);
   int64_t bench_sort(int32_t iterations);
   int64_t bench_crc32(int32_t iterations);
   int64_t bench_matmul(int32_t iterations);
   int64_t bench_matmul_simd(int32_t iterations);
}
#endif

#ifdef BENCH_HAS_WASM3
#include <wasm3.h>
#include <m3_env.h>
#endif

#ifdef BENCH_HAS_WAMR
#include <wasm_export.h>
#endif

using namespace eosio;
using namespace eosio::vm;

// ============================================================================
// Host functions — simulate psibase-style workloads
// ============================================================================

static uint64_t g_accumulator = 0;
static uint64_t g_mem_sum = 0;
static uint64_t g_multi_sum = 0;

int32_t bench_host_identity(int32_t x) { return x; }
void bench_host_accumulate(int64_t val) { g_accumulator += static_cast<uint64_t>(val); }

int64_t bench_host_mix(int64_t a, int64_t b) {
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

void bench_host_mem_op(int32_t ptr, int32_t len) {
   g_mem_sum += static_cast<uint64_t>(ptr) + static_cast<uint64_t>(len);
}

int64_t bench_host_multi(int64_t a, int64_t b, int32_t c, int32_t d) {
   g_multi_sum += static_cast<uint64_t>(a) + static_cast<uint64_t>(b);
   return static_cast<int64_t>(c + d);
}

// ============================================================================
// WASM binary builder
// ============================================================================

std::vector<uint8_t> build_bench_wasm() {
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
      emit_u32(6);
      emit(0x60); emit_u32(1); emit(0x7f); emit_u32(1); emit(0x7f); // 0: (i32)->i32
      emit(0x60); emit_u32(1); emit(0x7e); emit_u32(0);             // 1: (i64)->()
      emit(0x60); emit_u32(2); emit(0x7e, 0x7e); emit_u32(1); emit(0x7e); // 2: (i64,i64)->i64
      emit(0x60); emit_u32(2); emit(0x7f, 0x7f); emit_u32(0);       // 3: (i32,i32)->()
      emit(0x60); emit_u32(4); emit(0x7e, 0x7e, 0x7f, 0x7f); emit_u32(1); emit(0x7e); // 4: (i64,i64,i32,i32)->i64
      emit(0x60); emit_u32(1); emit(0x7f); emit_u32(1); emit(0x7e); // 5: (i32)->i64
   });

   // Import section
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
      for (int i = 0; i < 6; i++) emit_u32(5);
   });

   // Memory section — 1 page
   section(5, [&]() { emit_u32(1); emit(0x00); emit_u32(1); });

   // Export section
   section(7, [&]() {
      emit_u32(7);
      emit_str("bench_identity");   emit(0x00); emit_u32(5);
      emit_str("bench_accumulate"); emit(0x00); emit_u32(6);
      emit_str("bench_mix");        emit(0x00); emit_u32(7);
      emit_str("bench_mem_op");     emit(0x00); emit_u32(8);
      emit_str("bench_multi");      emit(0x00); emit_u32(9);
      emit_str("bench_mixed");      emit(0x00); emit_u32(10);
      emit_str("memory");           emit(0x02); emit_u32(0);
   });

   // Code section
   section(10, [&]() {
      emit_u32(6);

      // bench_identity: loop calling identity(i), sum results
      {
         size_t bp = w.size(); w.push_back(0); size_t cs = w.size();
         emit_u32(2); emit_u32(1); emit(0x7f); emit_u32(1); emit(0x7e);
         emit(0x03, 0x40);
           emit(0x20); emit_u32(1); emit(0x10); emit_u32(0); emit(0xAD);
           emit(0x20); emit_u32(2); emit(0x7C); emit(0x21); emit_u32(2);
           emit(0x20); emit_u32(1); emit(0x41, 0x01); emit(0x6A); emit(0x22); emit_u32(1);
           emit(0x20); emit_u32(0); emit(0x49); emit(0x0D, 0x00);
         emit(0x0B);
         emit(0x20); emit_u32(2); emit(0x0B);
         w[bp] = static_cast<uint8_t>(w.size() - cs);
      }

      // bench_accumulate
      {
         size_t bp = w.size(); w.push_back(0); size_t cs = w.size();
         emit_u32(1); emit_u32(1); emit(0x7f);
         emit(0x03, 0x40);
           emit(0x20); emit_u32(1); emit(0xAD); emit(0x10); emit_u32(1);
           emit(0x20); emit_u32(1); emit(0x41, 0x01); emit(0x6A); emit(0x22); emit_u32(1);
           emit(0x20); emit_u32(0); emit(0x49); emit(0x0D, 0x00);
         emit(0x0B);
         emit(0x42, 0x00); emit(0x0B);
         w[bp] = static_cast<uint8_t>(w.size() - cs);
      }

      // bench_mix
      {
         size_t bp = w.size(); w.push_back(0); size_t cs = w.size();
         emit_u32(2); emit_u32(1); emit(0x7f); emit_u32(1); emit(0x7e);
         emit(0x42, 0x01); emit(0x21); emit_u32(2);
         emit(0x03, 0x40);
           emit(0x20); emit_u32(2); emit(0x20); emit_u32(1); emit(0xAD);
           emit(0x10); emit_u32(2); emit(0x21); emit_u32(2);
           emit(0x20); emit_u32(1); emit(0x41, 0x01); emit(0x6A); emit(0x22); emit_u32(1);
           emit(0x20); emit_u32(0); emit(0x49); emit(0x0D, 0x00);
         emit(0x0B);
         emit(0x20); emit_u32(2); emit(0x0B);
         w[bp] = static_cast<uint8_t>(w.size() - cs);
      }

      // bench_mem_op
      {
         size_t bp = w.size(); w.push_back(0); size_t cs = w.size();
         emit_u32(1); emit_u32(1); emit(0x7f);
         emit(0x03, 0x40);
           emit(0x20); emit_u32(1); emit(0x41, 0x02); emit(0x74);
           emit(0x41, 0x10); emit(0x10); emit_u32(3);
           emit(0x20); emit_u32(1); emit(0x41, 0x01); emit(0x6A); emit(0x22); emit_u32(1);
           emit(0x20); emit_u32(0); emit(0x49); emit(0x0D, 0x00);
         emit(0x0B);
         emit(0x42, 0x00); emit(0x0B);
         w[bp] = static_cast<uint8_t>(w.size() - cs);
      }

      // bench_multi
      {
         size_t bp = w.size(); w.push_back(0); size_t cs = w.size();
         emit_u32(2); emit_u32(1); emit(0x7f); emit_u32(1); emit(0x7e);
         emit(0x42, 0x01); emit(0x21); emit_u32(2);
         emit(0x03, 0x40);
           emit(0x20); emit_u32(2); emit(0x20); emit_u32(1); emit(0xAD);
           emit(0x20); emit_u32(1); emit(0x41, 0x2A);
           emit(0x10); emit_u32(4); emit(0x21); emit_u32(2);
           emit(0x20); emit_u32(1); emit(0x41, 0x01); emit(0x6A); emit(0x22); emit_u32(1);
           emit(0x20); emit_u32(0); emit(0x49); emit(0x0D, 0x00);
         emit(0x0B);
         emit(0x20); emit_u32(2); emit(0x0B);
         w[bp] = static_cast<uint8_t>(w.size() - cs);
      }

      // bench_mixed: 5 host calls per iteration
      {
         size_t bp = w.size(); w.push_back(0); size_t cs = w.size();
         emit_u32(2); emit_u32(1); emit(0x7f); emit_u32(1); emit(0x7e);
         emit(0x42, 0x01); emit(0x21); emit_u32(2);
         emit(0x03, 0x40);
           // 1. identity(i)
           emit(0x20); emit_u32(1); emit(0x10); emit_u32(0); emit(0xAD);
           emit(0x20); emit_u32(2); emit(0x7C); emit(0x21); emit_u32(2);
           // 2. accumulate(acc)
           emit(0x20); emit_u32(2); emit(0x10); emit_u32(1);
           // 3. mix(acc, i64(i))
           emit(0x20); emit_u32(2); emit(0x20); emit_u32(1); emit(0xAD);
           emit(0x10); emit_u32(2); emit(0x21); emit_u32(2);
           // 4. mem_op(i*4, 16)
           emit(0x20); emit_u32(1); emit(0x41, 0x02); emit(0x74);
           emit(0x41, 0x10); emit(0x10); emit_u32(3);
           // 5. multi(acc, i64(i), i, 1)
           emit(0x20); emit_u32(2); emit(0x20); emit_u32(1); emit(0xAD);
           emit(0x20); emit_u32(1); emit(0x41, 0x01);
           emit(0x10); emit_u32(4);
           emit(0x20); emit_u32(2); emit(0x7C); emit(0x21); emit_u32(2);
           // i++
           emit(0x20); emit_u32(1); emit(0x41, 0x01); emit(0x6A); emit(0x22); emit_u32(1);
           emit(0x20); emit_u32(0); emit(0x49); emit(0x0D, 0x00);
         emit(0x0B);
         emit(0x20); emit_u32(2); emit(0x0B);
         w[bp] = static_cast<uint8_t>(w.size() - cs);
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
   rhf_t::add<&bench_host_identity>("env", "identity");
   rhf_t::add<&bench_host_accumulate>("env", "accumulate");
   rhf_t::add<&bench_host_mix>("env", "mix");
   rhf_t::add<&bench_host_mem_op>("env", "mem_op");
   rhf_t::add<&bench_host_multi>("env", "multi");
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

static m3ApiRawFunction(w3_identity) {
   m3ApiReturnType(int32_t);
   m3ApiGetArg(int32_t, x);
   m3ApiReturn(bench_host_identity(x));
}
static m3ApiRawFunction(w3_accumulate) {
   m3ApiGetArg(int64_t, val);
   bench_host_accumulate(val);
   m3ApiSuccess();
}
static m3ApiRawFunction(w3_mix) {
   m3ApiReturnType(int64_t);
   m3ApiGetArg(int64_t, a);
   m3ApiGetArg(int64_t, b);
   m3ApiReturn(bench_host_mix(a, b));
}
static m3ApiRawFunction(w3_mem_op) {
   m3ApiGetArg(int32_t, ptr);
   m3ApiGetArg(int32_t, len);
   bench_host_mem_op(ptr, len);
   m3ApiSuccess();
}
static m3ApiRawFunction(w3_multi) {
   m3ApiReturnType(int64_t);
   m3ApiGetArg(int64_t, a);
   m3ApiGetArg(int64_t, b);
   m3ApiGetArg(int32_t, c);
   m3ApiGetArg(int32_t, d);
   m3ApiReturn(bench_host_multi(a, b, c, d));
}

static double run_wasm3(const std::vector<uint8_t>& wasm, const char* func, uint32_t n) {
   // wasm3 may take ownership of the buffer; make a copy
   auto wasm_copy = wasm;
   IM3Environment env = m3_NewEnvironment();
   IM3Runtime runtime = m3_NewRuntime(env, 64 * 1024, nullptr);

   IM3Module module = nullptr;
   M3Result r = m3_ParseModule(env, &module, wasm_copy.data(), static_cast<uint32_t>(wasm_copy.size()));
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

static int32_t wamr_identity(wasm_exec_env_t, int32_t x) { return bench_host_identity(x); }
static void wamr_accumulate(wasm_exec_env_t, int64_t v) { bench_host_accumulate(v); }
static int64_t wamr_mix(wasm_exec_env_t, int64_t a, int64_t b) { return bench_host_mix(a, b); }
static void wamr_mem_op(wasm_exec_env_t, int32_t p, int32_t l) { bench_host_mem_op(p, l); }
static int64_t wamr_multi(wasm_exec_env_t, int64_t a, int64_t b, int32_t c, int32_t d) { return bench_host_multi(a, b, c, d); }

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

   // WAMR modifies its input buffer; make a copy
   auto wasm_copy = wasm;
   char error_buf[256];
   wasm_module_t module = wasm_runtime_load(wasm_copy.data(),
                                            static_cast<uint32_t>(wasm_copy.size()),
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
// Compute benchmark runners (zero-import WASM modules)
// ============================================================================

template<typename Impl>
static void dump_code_sizes(const std::vector<uint8_t>& wasm_bytes, const char* label) {
   if constexpr (!Impl::is_jit) return;
   using backend_t = eosio::vm::backend<std::nullptr_t, Impl>;
   wasm_code code(wasm_bytes.begin(), wasm_bytes.end());
   wasm_allocator wa;
   backend_t bkend(code, &wa);
   bkend.initialize(nullptr);
   auto& mod = bkend.get_module();
   uint32_t total = 0;
   for (uint32_t i = 0; i < mod.code.size(); ++i) {
      uint32_t offset = mod.code[i].jit_code_offset;
      uint32_t next = (i + 1 < mod.code.size()) ? mod.code[i+1].jit_code_offset : offset + 1000;
      uint32_t size = next - offset;
      total += size;
      if (mod.code.size() <= 5) {
         fprintf(stderr, "  %s func[%u]: wasm=%u bytes, native=%u bytes (%.1fx)\n",
                 label, i, mod.code[i].size, size, (float)size / mod.code[i].size);
      }
   }
   fprintf(stderr, "  %s: %u functions, total native=%u bytes\n", label, (uint32_t)mod.code.size(), total);
}

template<typename Impl>
static double run_eosvm_compute(const std::vector<uint8_t>& wasm_bytes, const char* func, uint32_t n) {
   using backend_t = eosio::vm::backend<std::nullptr_t, Impl>;
   wasm_code code(wasm_bytes.begin(), wasm_bytes.end());
   wasm_allocator wa;
   backend_t bkend(code, &wa);
   bkend.initialize(nullptr);

   auto result = bkend.call_with_return("env", func, n);
   int64_t rv = result ? result->to_i64() : -1;

   auto t1 = std::chrono::high_resolution_clock::now();
   bkend.call_with_return("env", func, n);
   auto t2 = std::chrono::high_resolution_clock::now();
   fprintf(stderr, "  %s result=%ld\n", func, rv);
   return std::chrono::duration<double, std::milli>(t2 - t1).count();
}

#ifdef BENCH_HAS_WASM3
static double run_wasm3_compute(const std::vector<uint8_t>& wasm, const char* func, uint32_t n) {
   auto wasm_copy = wasm;
   IM3Environment env = m3_NewEnvironment();
   IM3Runtime runtime = m3_NewRuntime(env, 256 * 1024, nullptr);

   IM3Module module = nullptr;
   M3Result r = m3_ParseModule(env, &module, wasm_copy.data(), static_cast<uint32_t>(wasm_copy.size()));
   if (r) { fprintf(stderr, "wasm3 parse error: %s\n", r); return -1; }
   r = m3_LoadModule(runtime, module);
   if (r) { fprintf(stderr, "wasm3 load error: %s\n", r); return -1; }

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

#ifdef BENCH_HAS_WAMR
static double run_wamr_compute(const std::vector<uint8_t>& wasm, const char* func, uint32_t n) {
   static bool inited = false;
   if (!inited) {
      wasm_runtime_init();
      inited = true;
   }

   auto wasm_copy = wasm;
   char error_buf[256];
   wasm_module_t module = wasm_runtime_load(wasm_copy.data(),
                                            static_cast<uint32_t>(wasm_copy.size()),
                                            error_buf, sizeof(error_buf));
   if (!module) { fprintf(stderr, "WAMR load error: %s\n", error_buf); return -1; }

   wasm_module_inst_t inst = wasm_runtime_instantiate(module, 256*1024, 256*1024,
                                                       error_buf, sizeof(error_buf));
   if (!inst) { fprintf(stderr, "WAMR inst error: %s\n", error_buf); wasm_runtime_unload(module); return -1; }

   wasm_exec_env_t exec_env = wasm_runtime_create_exec_env(inst, 256*1024);
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
// Main
// ============================================================================

int main(int argc, char** argv) {
   register_eosvm_hosts();

   auto wasm_bytes = build_bench_wasm();
   wasm_code code(wasm_bytes.begin(), wasm_bytes.end());
   wasm_allocator wa;

   // Defaults
   uint32_t N = 10'000'000;
   bool run_interp = true, run_jit = true, run_jit2 = true;
   bool run_host = true, run_compute = true;

   // Parse args: --no-interp, --no-jit, --no-host, --no-compute, --iters=N
   for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];
      if (arg == "--no-interp")   run_interp = false;
      else if (arg == "--no-jit") run_jit = false;
      else if (arg == "--no-jit2") run_jit2 = false;
      else if (arg == "--no-host") run_host = false;
      else if (arg == "--no-compute") run_compute = false;
      else if (arg == "--jit2-only") { run_interp = false; run_jit = false; }
      else if (arg.rfind("--iters=", 0) == 0) N = static_cast<uint32_t>(std::stoul(arg.substr(8)));
      else if (arg == "--help" || arg == "-h") {
         printf("Usage: bench-compare [options]\n"
                "  --no-interp    Skip interpreter\n"
                "  --no-jit       Skip JIT1\n"
                "  --no-jit2      Skip JIT2\n"
                "  --jit2-only    Only run JIT2\n"
                "  --no-host      Skip host-call benchmarks\n"
                "  --no-compute   Skip compute benchmarks\n"
                "  --iters=N      Host-call iterations (default 10000000)\n");
         return 0;
      }
   }

   struct bench_def {
      const char* label;
      const char* func;
      uint32_t    calls_per_iter;
   };

   bench_def benches[] = {
      {"identity (1 call)",     "bench_identity",   1},
      {"accumulate (1 call)",   "bench_accumulate",  1},
      {"mix (1 call)",          "bench_mix",         1},
      {"mem_op (ptr,len)",      "bench_mem_op",      1},
      {"multi (4 args)",        "bench_multi",       1},
      {"mixed (5 calls/iter)",  "bench_mixed",       5},
   };

   // =========================================================================
   // Collect results into a table, then print with relative-to-fastest
   // =========================================================================

   struct runtime_info {
      const char* name;
      bool        enabled;
   };
   enum RT { RT_NATIVE, RT_INTERP, RT_JIT, RT_JIT2, RT_WASM3, RT_WAMR, RT_WASMTIME, RT_WASMER, RT_COUNT };

   runtime_info runtimes[RT_COUNT] = {
      {"native",        false},  // enabled only for compute benchmarks
      {"eos-vm interp", run_interp},
#ifdef __x86_64__
      {"eos-vm JIT",    run_jit},
      {"eos-vm JIT2",   run_jit2},
#else
      {"eos-vm JIT",    false},
      {"eos-vm JIT2",   false},
#endif
#ifdef BENCH_HAS_WASM3
      {"wasm3",         true},
#else
      {"wasm3",         false},
#endif
#ifdef BENCH_HAS_WAMR
      {"WAMR",          true},
#else
      {"WAMR",          false},
#endif
#ifdef BENCH_HAS_WASMTIME
      {"wasmtime",      true},
#else
      {"wasmtime",      false},
#endif
#ifdef BENCH_HAS_WASMER
      {"wasmer",        true},
#else
      {"wasmer",        false},
#endif
   };

   // Count active runtimes and compute column width
   int num_active = 0;
   for (int r = 0; r < RT_COUNT; r++)
      if (runtimes[r].enabled) num_active++;

   // Print a results table: ms values first, then relative-to-fastest
   auto print_table = [&](const char* title, const char* subtitle,
                          int num_tests, const char* labels[],
                          double results[][RT_COUNT]) {
      // Find the fastest for each test
      double fastest[16];
      for (int t = 0; t < num_tests; t++) {
         fastest[t] = 1e18;
         for (int r = 0; r < RT_COUNT; r++)
            if (runtimes[r].enabled && results[t][r] > 0 && results[t][r] < fastest[t])
               fastest[t] = results[t][r];
      }

      printf("\n%s\n", title);
      if (subtitle) printf("%s\n", subtitle);

      // --- Raw ms table ---
      printf("\n  %-26s", "");
      for (int r = 0; r < RT_COUNT; r++)
         if (runtimes[r].enabled)
            printf(" %14s", runtimes[r].name);
      printf("\n  %-26s", "");
      for (int r = 0; r < RT_COUNT; r++)
         if (runtimes[r].enabled)
            printf(" %14s", "(ms)");
      printf("\n  %s\n", std::string(26 + num_active * 15, '-').c_str());
      for (int t = 0; t < num_tests; t++) {
         printf("  %-26s", labels[t]);
         for (int r = 0; r < RT_COUNT; r++)
            if (runtimes[r].enabled)
               printf(" %14.1f", results[t][r]);
         printf("\n");
      }

      // --- Relative performance table ---
      printf("\n  %-26s", "relative to fastest");
      for (int r = 0; r < RT_COUNT; r++)
         if (runtimes[r].enabled)
            printf(" %14s", runtimes[r].name);
      printf("\n  %s\n", std::string(26 + num_active * 15, '-').c_str());
      for (int t = 0; t < num_tests; t++) {
         printf("  %-26s", labels[t]);
         for (int r = 0; r < RT_COUNT; r++) {
            if (!runtimes[r].enabled) continue;
            double ratio = results[t][r] / fastest[t];
            if (ratio < 1.05)
               printf("      \xe2\x96\xb6 %4.1fx ", ratio);  // arrow for winner
            else
               printf("        %4.1fx ", ratio);
         }
         printf("\n");
      }
   };

   // --- Host-call benchmarks ---
   const int num_host_tests = sizeof(benches) / sizeof(benches[0]);
   double host_results[6][RT_COUNT] = {};
   const char* host_labels[6];

   for (int t = 0; t < num_host_tests && run_host; t++) {
      host_labels[t] = benches[t].label;
      if (run_interp) {
         fprintf(stderr, "host[%d] interp...\n", t); fflush(stderr);
         host_results[t][RT_INTERP] = run_eosvm<interpreter>(code, wa, benches[t].func, N);
      }
#ifdef __x86_64__
      if (run_jit) {
         fprintf(stderr, "host[%d] jit...\n", t); fflush(stderr);
         host_results[t][RT_JIT] = run_eosvm<jit>(code, wa, benches[t].func, N);
      }
      if (run_jit2) {
         fprintf(stderr, "host[%d] jit2...\n", t); fflush(stderr);
         host_results[t][RT_JIT2] = run_eosvm<jit2>(code, wa, benches[t].func, N);
      }
#endif
#ifdef BENCH_HAS_WASM3
      host_results[t][RT_WASM3] = run_wasm3(wasm_bytes, benches[t].func, N);
#endif
#ifdef BENCH_HAS_WAMR
      host_results[t][RT_WAMR] = run_wamr(wasm_bytes, benches[t].func, N);
#endif
#ifdef BENCH_HAS_WASMTIME
      host_results[t][RT_WASMTIME] = run_wasmtime(wasm_bytes, benches[t].func, N);
#endif
#ifdef BENCH_HAS_WASMER
      host_results[t][RT_WASMER] = run_wasmer(wasm_bytes, benches[t].func, N);
#endif
   }

   if (run_host) {
      char host_title[128];
      snprintf(host_title, sizeof(host_title),
               "HOST-CALL BENCHMARK (%u iterations per test)", N);
      print_table(host_title,
                  "Measures wasm-to-native call transition overhead.",
                  num_host_tests, host_labels, host_results);
   }

   fprintf(stderr, "host-call done, starting compute...\n"); fflush(stderr);
   // --- Compute benchmarks ---
#ifdef BENCH_HAS_COMPUTE
  if (run_compute) {
   // Enable native column for compute benchmarks
   runtimes[RT_NATIVE].enabled = true;
   num_active++;

   using native_fn = int64_t (*)(int32_t);

   struct compute_def {
      const char* label;
      const char* wasm_path;
      const char* func;
      native_fn   native;
      uint32_t    iters;
   };
   compute_def compute_tests[] = {
      {"SHA-256 (64B, 10K)",   BENCH_SHA256_WASM, "bench_sha256",       bench_sha256,       10'000},
      {"ECDSA verify (k1)",    BENCH_ECDSA_WASM,  "bench_ecdsa_verify", bench_ecdsa_verify, 100},
      {"ECDSA sign (k1)",      BENCH_ECDSA_WASM,  "bench_ecdsa_sign",   bench_ecdsa_sign,   100},
      {"Fibonacci (1M)",       BENCH_MISC_WASM,   "bench_fib",          bench_fib,          1'000'000},
      {"Bubble sort (64, 1K)", BENCH_MISC_WASM,   "bench_sort",         bench_sort,         1'000},
      {"CRC32 (256B, 10K)",    BENCH_MISC_WASM,   "bench_crc32",        bench_crc32,        10'000},
      {"MatMul 8x8 (10K)",     BENCH_MISC_WASM,   "bench_matmul",       bench_matmul,       10'000},
      {"MatMul SIMD 8x8 (10K)",BENCH_MISC_WASM,   "bench_matmul_simd",  bench_matmul_simd,  10'000},
   };
   const int num_compute = sizeof(compute_tests) / sizeof(compute_tests[0]);
   double compute_results[16][RT_COUNT] = {};
   const char* compute_labels[16];

   // Pre-load WASM files
   std::vector<uint8_t> sha_wasm, ecdsa_wasm, misc_wasm;
   try { sha_wasm = eosio::vm::read_wasm(BENCH_SHA256_WASM); } catch (...) {}
   try { ecdsa_wasm = eosio::vm::read_wasm(BENCH_ECDSA_WASM); } catch (...) {}
   try { misc_wasm = eosio::vm::read_wasm(BENCH_MISC_WASM); } catch (...) {}

   auto get_wasm = [&](int t) -> std::vector<uint8_t>& {
      if (compute_tests[t].wasm_path == std::string(BENCH_SHA256_WASM)) return sha_wasm;
      if (compute_tests[t].wasm_path == std::string(BENCH_ECDSA_WASM)) return ecdsa_wasm;
      return misc_wasm;
   };

   for (int t = 0; t < num_compute; t++) {
      compute_labels[t] = compute_tests[t].label;
      auto& wasm = get_wasm(t);
      auto func = compute_tests[t].func;
      auto iters = compute_tests[t].iters;

      // Native baseline
      {
         auto t1 = std::chrono::high_resolution_clock::now();
         compute_tests[t].native(static_cast<int32_t>(iters));
         auto t2 = std::chrono::high_resolution_clock::now();
         compute_results[t][RT_NATIVE] = std::chrono::duration<double, std::milli>(t2 - t1).count();
      }

      if (wasm.empty()) continue;

      fprintf(stderr, "\n=== %s ===\n", compute_tests[t].label); fflush(stderr);
#ifdef __x86_64__
      dump_code_sizes<jit>(wasm, "jit1");
      dump_code_sizes<jit2>(wasm, "jit2");
#endif
      if (run_interp) {
         fprintf(stderr, "interp...\n"); fflush(stderr);
         compute_results[t][RT_INTERP] = run_eosvm_compute<interpreter>(wasm, func, iters);
      }
#ifdef __x86_64__
      if (run_jit) {
         fprintf(stderr, "jit1...\n"); fflush(stderr);
         compute_results[t][RT_JIT] = run_eosvm_compute<jit>(wasm, func, iters);
      }
      if (run_jit2) {
         fprintf(stderr, "jit2...\n"); fflush(stderr);
         compute_results[t][RT_JIT2] = run_eosvm_compute<jit2>(wasm, func, iters);
         fprintf(stderr, "jit2 done\n"); fflush(stderr);
      }
#endif
#ifdef BENCH_HAS_WASM3
      compute_results[t][RT_WASM3] = run_wasm3_compute(wasm, func, iters);
#endif
#ifdef BENCH_HAS_WAMR
      compute_results[t][RT_WAMR] = run_wamr_compute(wasm, func, iters);
#endif
#ifdef BENCH_HAS_WASMTIME
      compute_results[t][RT_WASMTIME] = run_wasmtime_compute(wasm, func, iters);
#endif
#ifdef BENCH_HAS_WASMER
      compute_results[t][RT_WASMER] = run_wasmer_compute(wasm, func, iters);
#endif
   }

   print_table("COMPUTE BENCHMARK (pure WASM, no host calls)",
               "Measures raw computation speed for crypto workloads. Native = same C code compiled natively.",
               num_compute, compute_labels, compute_results);

   // Disable native column again so it doesn't affect any further output
   runtimes[RT_NATIVE].enabled = false;
   num_active--;
  } // if (run_compute)
#endif

   printf("\n");
   return 0;
}
