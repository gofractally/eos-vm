#include <eosio/vm/backend.hpp>
#include <eosio/vm/error_codes.hpp>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace eosio;
using namespace eosio::vm;

// Stub host functions matching psibase's NativeFunctions signatures.
// Each does minimal work to isolate the call overhead.

struct host_context {
   // Simulated result buffer
   char result_buf[256];
   uint32_t result_size = 64;
   uint64_t checksum = 0;

   // Pattern 1: span<char> dest + uint32_t offset → uint32_t  (getResult, getKey)
   uint32_t getResult(span<char> dest, uint32_t offset) {
      uint32_t to_copy = 0;
      if (offset < result_size && dest.size() > 0) {
         to_copy = std::min((uint32_t)dest.size(), result_size - offset);
         memcpy(dest.data(), result_buf + offset, to_copy);
      }
      return result_size;
   }

   // Pattern 2: span<const char> → void  (writeConsole, abortMessage, setRetval)
   void writeConsole(span<const char> str) {
      for (size_t i = 0; i < str.size(); ++i)
         checksum += (uint8_t)str[i];
   }

   // Pattern 3: uint32_t + span<const char> + span<const char> → void  (kvPut)
   void kvPut(uint32_t handle, span<const char> key, span<const char> value) {
      checksum += handle + key.size() + value.size();
   }

   // Pattern 4: uint32_t + span<const char> → uint32_t  (kvGet, kvOpen)
   uint32_t kvGet(uint32_t handle, span<const char> key) {
      checksum += handle + key.size();
      return 42;
   }

   // Pattern 5: uint32_t + argument_proxy<uint64_t*> → int32_t  (clockTimeGet)
   int32_t clockTimeGet(uint32_t id, argument_proxy<uint64_t*> time) {
      *time = 1234567890ULL + id;
      return 0;
   }

   // Pattern 6: pure scalar → uint32_t  (getCurrentAction, kvGetTransactionUsage)
   uint32_t getCurrentAction() {
      return 7;
   }

   // Pattern 7: uint32_t + span<const char> + uint32_t → uint32_t (kvGreaterEqual)
   uint32_t kvGreaterEqual(uint32_t handle, span<const char> key, uint32_t matchKeySize) {
      checksum += handle + key.size() + matchKeySize;
      return 1;
   }
};

using rhf_t = registered_host_functions<host_context>;

// Build WASM module that benchmarks each call pattern.
// For span-taking functions, we pass a pointer into linear memory + length.
// Each bench function loops N times calling the host function.
static std::vector<uint8_t> build_wasm() {
   std::vector<uint8_t> w;
   auto emit = [&](auto... bytes) { (w.push_back(static_cast<uint8_t>(bytes)), ...); };
   auto emit_u32_leb = [&](uint32_t v) {
      do { uint8_t b = v & 0x7f; v >>= 7; if (v) b |= 0x80; w.push_back(b); } while (v);
   };
   auto emit_section = [&](uint8_t id, auto fn) {
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
         w.erase(w.begin() + size_pos, w.begin() + size_pos + 1);
         w.insert(w.begin() + size_pos, leb.begin(), leb.end());
      }
   };

   // Magic + version
   emit(0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00);

   // Type section
   // We need function types for each import and each bench function.
   // Import types (matching WASM-level signatures):
   //   0: (i32, i32, i32) -> i32        getResult(ptr, len, offset)
   //   1: (i32, i32) -> ()              writeConsole(ptr, len)
   //   2: (i32, i32, i32, i32, i32) -> ()  kvPut(handle, key_ptr, key_len, val_ptr, val_len)
   //   3: (i32, i32, i32) -> i32        kvGet(handle, key_ptr, key_len)
   //   4: (i32, i32) -> i32             clockTimeGet(id, time_ptr)
   //   5: () -> i32                     getCurrentAction()
   //   6: (i32, i32, i32, i32) -> i32   kvGreaterEqual(handle, key_ptr, key_len, matchKeySize)
   // Bench function type:
   //   7: (i32) -> i64                  bench(n) -> result
   emit_section(1, [&]() {
      emit_u32_leb(8); // 8 types

      // Type 0: (i32, i32, i32) -> i32
      emit(0x60); emit_u32_leb(3); emit(0x7f, 0x7f, 0x7f); emit_u32_leb(1); emit(0x7f);
      // Type 1: (i32, i32) -> ()
      emit(0x60); emit_u32_leb(2); emit(0x7f, 0x7f); emit_u32_leb(0);
      // Type 2: (i32, i32, i32, i32, i32) -> ()
      emit(0x60); emit_u32_leb(5); emit(0x7f, 0x7f, 0x7f, 0x7f, 0x7f); emit_u32_leb(0);
      // Type 3: (i32, i32, i32) -> i32
      emit(0x60); emit_u32_leb(3); emit(0x7f, 0x7f, 0x7f); emit_u32_leb(1); emit(0x7f);
      // Type 4: (i32, i32) -> i32
      emit(0x60); emit_u32_leb(2); emit(0x7f, 0x7f); emit_u32_leb(1); emit(0x7f);
      // Type 5: () -> i32
      emit(0x60); emit_u32_leb(0); emit_u32_leb(1); emit(0x7f);
      // Type 6: (i32, i32, i32, i32) -> i32
      emit(0x60); emit_u32_leb(4); emit(0x7f, 0x7f, 0x7f, 0x7f); emit_u32_leb(1); emit(0x7f);
      // Type 7: (i32) -> i64  (bench functions)
      emit(0x60); emit_u32_leb(1); emit(0x7f); emit_u32_leb(1); emit(0x7e);
   });

   // Import section: 7 imports
   emit_section(2, [&]() {
      emit_u32_leb(7);
      auto emit_import = [&](const char* name, uint8_t type_idx) {
         emit_u32_leb(3); emit('e', 'n', 'v');
         uint32_t len = strlen(name);
         emit_u32_leb(len);
         for (uint32_t i = 0; i < len; i++) emit(name[i]);
         emit(0x00); emit_u32_leb(type_idx);
      };
      emit_import("getResult", 0);       // import 0
      emit_import("writeConsole", 1);     // import 1
      emit_import("kvPut", 2);            // import 2
      emit_import("kvGet", 3);            // import 3
      emit_import("clockTimeGet", 4);     // import 4
      emit_import("getCurrentAction", 5); // import 5
      emit_import("kvGreaterEqual", 6);   // import 6
   });

   // Function section: 7 bench functions (indices 7-13)
   emit_section(3, [&]() {
      emit_u32_leb(7);
      for (int i = 0; i < 7; i++) emit_u32_leb(7); // all type 7: (i32)->i64
   });

   // Memory section: 1 page of memory for span pointers to reference
   emit_section(5, [&]() {
      emit_u32_leb(1);
      emit(0x00); emit_u32_leb(1); // min 1 page, no max
   });

   // Export section
   emit_section(7, [&]() {
      emit_u32_leb(7);
      auto emit_export = [&](const char* name, uint32_t func_idx) {
         uint32_t len = strlen(name);
         emit_u32_leb(len);
         for (uint32_t i = 0; i < len; i++) emit(name[i]);
         emit(0x00); emit_u32_leb(func_idx);
      };
      emit_export("bench_getResult", 7);
      emit_export("bench_writeConsole", 8);
      emit_export("bench_kvPut", 9);
      emit_export("bench_kvGet", 10);
      emit_export("bench_clockTimeGet", 11);
      emit_export("bench_getCurrentAction", 12);
      emit_export("bench_kvGreaterEqual", 13);
   });

   // Code section: each bench function is a loop calling the host function N times
   emit_section(10, [&]() {
      emit_u32_leb(7);

      // Helper lambda: emit a loop body that calls an import, drops/accumulates result
      auto emit_bench_body = [&](auto emit_call_body) {
         size_t body_start = w.size();
         w.push_back(0); // body size placeholder
         size_t code_start = w.size();

         emit_u32_leb(2);              // 2 local groups
         emit_u32_leb(1); emit(0x7f);  // $i: i32
         emit_u32_leb(1); emit(0x7e);  // $sum: i64

         // loop
         emit(0x03, 0x40);
           emit_call_body();
           // i++
           emit(0x20); emit_u32_leb(1); // local.get $i
           emit(0x41, 0x01);             // i32.const 1
           emit(0x6A);                   // i32.add
           emit(0x22); emit_u32_leb(1); // local.tee $i
           emit(0x20); emit_u32_leb(0); // local.get $n
           emit(0x49);                   // i32.lt_u
           emit(0x0D, 0x00);             // br_if 0
         emit(0x0B); // end loop

         emit(0x20); emit_u32_leb(2); // local.get $sum
         emit(0x0B); // end function

         uint32_t body_size = static_cast<uint32_t>(w.size() - code_start);
         w[body_start] = static_cast<uint8_t>(body_size);
      };

      // Helper to emit i32.const with proper signed LEB128
      auto emit_i32_const = [&](int32_t val) {
         emit(0x41);
         int32_t v = val;
         bool more = true;
         while (more) {
            uint8_t b = v & 0x7f;
            v >>= 7;
            if ((v == 0 && !(b & 0x40)) || (v == -1 && (b & 0x40)))
               more = false;
            else
               b |= 0x80;
            w.push_back(b);
         }
      };

      // bench_getResult: call getResult(ptr=0, len=64, offset=0), sum += i64.extend(result)
      emit_bench_body([&]() {
         emit_i32_const(0);              // dest ptr
         emit_i32_const(64);             // dest len
         emit_i32_const(0);              // offset
         emit(0x10); emit_u32_leb(0);   // call getResult
         emit(0xAD);                     // i64.extend_i32_u
         emit(0x20); emit_u32_leb(2);   // local.get $sum
         emit(0x7C);                     // i64.add
         emit(0x21); emit_u32_leb(2);   // local.set $sum
      });

      // bench_writeConsole: call writeConsole(ptr=0, len=16)
      emit_bench_body([&]() {
         emit_i32_const(0);              // str ptr
         emit_i32_const(16);             // str len
         emit(0x10); emit_u32_leb(1);   // call writeConsole
      });

      // bench_kvPut: call kvPut(handle=1, key_ptr=0, key_len=8, val_ptr=64, val_len=32)
      emit_bench_body([&]() {
         emit_i32_const(1);              // handle
         emit_i32_const(0);              // key_ptr
         emit_i32_const(8);              // key_len
         emit_i32_const(64);             // val_ptr
         emit_i32_const(32);             // val_len
         emit(0x10); emit_u32_leb(2);   // call kvPut
      });

      // bench_kvGet: call kvGet(handle=1, key_ptr=0, key_len=8)
      emit_bench_body([&]() {
         emit_i32_const(1);
         emit_i32_const(0);
         emit_i32_const(8);
         emit(0x10); emit_u32_leb(3);
         emit(0xAD);
         emit(0x20); emit_u32_leb(2);
         emit(0x7C);
         emit(0x21); emit_u32_leb(2);
      });

      // bench_clockTimeGet: call clockTimeGet(id=0, time_ptr=256)
      emit_bench_body([&]() {
         emit_i32_const(0);              // id=0
         emit_i32_const(256);            // time_ptr=256
         emit(0x10); emit_u32_leb(4);
         emit(0x1A);                     // drop result
      });

      // bench_getCurrentAction: call getCurrentAction()
      emit_bench_body([&]() {
         emit(0x10); emit_u32_leb(5);
         emit(0xAD);
         emit(0x20); emit_u32_leb(2);
         emit(0x7C);
         emit(0x21); emit_u32_leb(2);
      });

      // bench_kvGreaterEqual: call kvGreaterEqual(handle=1, key_ptr=0, key_len=8, matchKeySize=4)
      emit_bench_body([&]() {
         emit_i32_const(1);
         emit_i32_const(0);
         emit_i32_const(8);
         emit_i32_const(4);
         emit(0x10); emit_u32_leb(6);
         emit(0xAD);
         emit(0x20); emit_u32_leb(2);
         emit(0x7C);
         emit(0x21); emit_u32_leb(2);
      });
   });

   return w;
}

template<typename Impl>
double run_bench(wasm_code& code, wasm_allocator& wa, host_context& ctx,
                 const std::string& func, uint32_t n) {
   try {
      using backend_t = backend<rhf_t, Impl>;
      backend_t bkend(code, &wa);
      rhf_t::resolve(bkend.get_module());
      bkend.initialize(&ctx);

      auto t1 = std::chrono::high_resolution_clock::now();
      bkend.call_with_return(ctx, "env", func, n);
      auto t2 = std::chrono::high_resolution_clock::now();

      return std::chrono::duration<double, std::milli>(t2 - t1).count();
   } catch (const std::exception& e) {
      fprintf(stderr, "  ERROR: %s\n", e.what());
      return -1;
   }
}

int main() {
   wasm_allocator wa;
   host_context ctx;
   memset(ctx.result_buf, 0x42, sizeof(ctx.result_buf));

   rhf_t::add<&host_context::getResult>("env", "getResult");
   rhf_t::add<&host_context::writeConsole>("env", "writeConsole");
   rhf_t::add<&host_context::kvPut>("env", "kvPut");
   rhf_t::add<&host_context::kvGet>("env", "kvGet");
   rhf_t::add<&host_context::clockTimeGet>("env", "clockTimeGet");
   rhf_t::add<&host_context::getCurrentAction>("env", "getCurrentAction");
   rhf_t::add<&host_context::kvGreaterEqual>("env", "kvGreaterEqual");

   auto wasm = build_wasm();
   wasm_code code(wasm.begin(), wasm.end());

   const uint32_t N = 5'000'000;

   struct bench_entry {
      const char* name;
      const char* func;
      const char* pattern;
   };

   bench_entry benches[] = {
      {"getCurrentAction", "bench_getCurrentAction",  "()→u32"},
      {"getResult",        "bench_getResult",        "span<char>+u32→u32"},
      {"writeConsole",     "bench_writeConsole",      "span<const char>→void"},
      {"kvPut",            "bench_kvPut",             "u32+2×span→void"},
      {"kvGet",            "bench_kvGet",             "u32+span→u32"},
      {"clockTimeGet",     "bench_clockTimeGet",      "u32+arg_proxy→i32"},
      {"kvGreaterEqual",   "bench_kvGreaterEqual",    "u32+span+u32→u32"},
   };

   printf("Psibase-style host function benchmark — %u iterations each\n\n", N);
   printf("%-20s %-22s %10s %10s %8s %10s\n",
          "Function", "Pattern", "Interp ms", "JIT ms", "Speedup", "JIT Mc/s");
   printf("%s\n", std::string(84, '-').c_str());

   for (auto& b : benches) {
      double interp_ms = run_bench<interpreter>(code, wa, ctx, b.func, N);
      double jit_ms = run_bench<jit>(code, wa, ctx, b.func, N);
      double speedup = interp_ms / jit_ms;
      double mcalls = N / (jit_ms * 1000.0);

      printf("%-20s %-22s %10.1f %10.1f %7.1fx %9.1f\n",
             b.name, b.pattern, interp_ms, jit_ms, speedup, mcalls);
   }

   return 0;
}
