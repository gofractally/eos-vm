#include <eosio/vm/backend.hpp>
#include <eosio/vm/error_codes.hpp>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace eosio;
using namespace eosio::vm;

// --- Host functions that simulate real blockchain operations ---

// Trivial: just return the input (tests pure call overhead)
int32_t host_identity(int32_t x) { return x; }

// Light work: accumulate into a checksum
static uint64_t g_checksum = 0;
void host_accumulate(int64_t val) { g_checksum += static_cast<uint64_t>(val); }

// Medium work: hash-like mixing
int64_t host_mix(int64_t a, int64_t b) {
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

// Memory access: read N bytes from linear memory
static uint64_t g_mem_sum = 0;
void host_read_mem(int32_t offset, int32_t len) {
   // Simulates reading from linear memory (host side doesn't actually access it)
   g_mem_sum += static_cast<uint64_t>(offset) + static_cast<uint64_t>(len);
}

using rhf_t = registered_host_functions<standalone_function_t>;

// Build a WASM module that calls host functions in a tight loop
// The module has:
//   (import "env" "identity"   (func $identity   (param i32) (result i32)))
//   (import "env" "accumulate" (func $accumulate (param i64)))
//   (import "env" "mix"        (func $mix        (param i64 i64) (result i64)))
//   (import "env" "read_mem"   (func $read_mem   (param i32 i32)))
//   (export "bench_identity"   (func $bench_identity))
//   (export "bench_accumulate" (func $bench_accumulate))
//   (export "bench_mix"        (func $bench_mix))
//
// Each bench function loops N times calling the host function.
//
// Rather than hand-encoding WASM, we write it using the binary format.
static std::vector<uint8_t> build_host_call_wasm() {
   std::vector<uint8_t> w;
   auto emit = [&](auto... bytes) { (w.push_back(static_cast<uint8_t>(bytes)), ...); };
   auto emit_u32_leb = [&](uint32_t v) {
      do { uint8_t b = v & 0x7f; v >>= 7; if (v) b |= 0x80; w.push_back(b); } while (v);
   };
   auto emit_i32_leb = [&](int32_t v) {
      bool more = true;
      while (more) {
         uint8_t b = v & 0x7f; v >>= 7;
         if ((v == 0 && !(b & 0x40)) || (v == -1 && (b & 0x40))) more = false;
         else b |= 0x80;
         w.push_back(b);
      }
   };
   auto emit_section = [&](uint8_t id, auto fn) {
      w.push_back(id);
      size_t size_pos = w.size();
      w.push_back(0); // placeholder
      size_t start = w.size();
      fn();
      uint32_t size = static_cast<uint32_t>(w.size() - start);
      // Simple: if size < 128, single byte LEB
      if (size < 128) {
         w[size_pos] = static_cast<uint8_t>(size);
      } else {
         // Insert extra bytes for LEB128
         std::vector<uint8_t> leb;
         uint32_t s = size;
         do { uint8_t b = s & 0x7f; s >>= 7; if (s) b |= 0x80; leb.push_back(b); } while (s);
         w.erase(w.begin() + size_pos, w.begin() + size_pos + 1);
         w.insert(w.begin() + size_pos, leb.begin(), leb.end());
      }
   };

   // Magic + version
   emit(0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00);

   // Type section (section 1)
   emit_section(1, [&]() {
      emit_u32_leb(5); // 5 types
      // Type 0: (i32) -> i32   (identity)
      emit(0x60); emit_u32_leb(1); emit(0x7f); emit_u32_leb(1); emit(0x7f);
      // Type 1: (i64) -> ()    (accumulate)
      emit(0x60); emit_u32_leb(1); emit(0x7e); emit_u32_leb(0);
      // Type 2: (i64, i64) -> i64 (mix)
      emit(0x60); emit_u32_leb(2); emit(0x7e, 0x7e); emit_u32_leb(1); emit(0x7e);
      // Type 3: (i32, i32) -> () (read_mem)
      emit(0x60); emit_u32_leb(2); emit(0x7f, 0x7f); emit_u32_leb(0);
      // Type 4: (i32) -> i64   (bench functions)
      emit(0x60); emit_u32_leb(1); emit(0x7f); emit_u32_leb(1); emit(0x7e);
   });

   // Import section (section 2)
   emit_section(2, [&]() {
      emit_u32_leb(4); // 4 imports
      // import 0: env.identity (type 0)
      emit_u32_leb(3); emit('e','n','v'); emit_u32_leb(8); emit('i','d','e','n','t','i','t','y');
      emit(0x00); emit_u32_leb(0);
      // import 1: env.accumulate (type 1)
      emit_u32_leb(3); emit('e','n','v'); emit_u32_leb(10); emit('a','c','c','u','m','u','l','a','t','e');
      emit(0x00); emit_u32_leb(1);
      // import 2: env.mix (type 2)
      emit_u32_leb(3); emit('e','n','v'); emit_u32_leb(3); emit('m','i','x');
      emit(0x00); emit_u32_leb(2);
      // import 3: env.read_mem (type 3)
      emit_u32_leb(3); emit('e','n','v'); emit_u32_leb(8); emit('r','e','a','d','_','m','e','m');
      emit(0x00); emit_u32_leb(3);
   });

   // Function section (section 3) — 3 functions
   emit_section(3, [&]() {
      emit_u32_leb(3);
      emit_u32_leb(4); // bench_identity: type 4 (i32) -> i64
      emit_u32_leb(4); // bench_accumulate: type 4
      emit_u32_leb(4); // bench_mix: type 4
   });

   // Export section (section 7)
   emit_section(7, [&]() {
      emit_u32_leb(3);
      emit_u32_leb(14); emit('b','e','n','c','h','_','i','d','e','n','t','i','t','y');
      emit(0x00); emit_u32_leb(4); // func index 4
      emit_u32_leb(16); emit('b','e','n','c','h','_','a','c','c','u','m','u','l','a','t','e');
      emit(0x00); emit_u32_leb(5);
      emit_u32_leb(9); emit('b','e','n','c','h','_','m','i','x');
      emit(0x00); emit_u32_leb(6);
   });

   // Code section (section 10)
   emit_section(10, [&]() {
      emit_u32_leb(3); // 3 function bodies

      // Function 4: bench_identity(n: i32) -> i64
      // local $i: i32, local $sum: i64
      // loop: call identity(i), sum += result, i++, br_if i < n
      {
         size_t body_start = w.size();
         w.push_back(0); // body size placeholder
         size_t code_start = w.size();

         emit_u32_leb(2); // 2 local decl groups
         emit_u32_leb(1); emit(0x7f); // 1 x i32
         emit_u32_leb(1); emit(0x7e); // 1 x i64

         // loop $L
         emit(0x03, 0x40); // loop void
           // call identity(local.get $i)
           emit(0x20); emit_u32_leb(1); // local.get $i
           emit(0x10); emit_u32_leb(0); // call $identity (import 0)
           // sum += i64.extend_i32_u(result)
           emit(0xAD);                   // i64.extend_i32_u
           emit(0x20); emit_u32_leb(2); // local.get $sum
           emit(0x7C);                   // i64.add
           emit(0x21); emit_u32_leb(2); // local.set $sum
           // i++
           emit(0x20); emit_u32_leb(1); // local.get $i
           emit(0x41, 0x01);             // i32.const 1
           emit(0x6A);                   // i32.add
           emit(0x22); emit_u32_leb(1); // local.tee $i
           // br_if i < n
           emit(0x20); emit_u32_leb(0); // local.get $n (param 0)
           emit(0x49);                   // i32.lt_u
           emit(0x0D, 0x00);             // br_if 0 (loop)
         emit(0x0B); // end loop
         // return sum
         emit(0x20); emit_u32_leb(2); // local.get $sum
         emit(0x0B); // end function

         uint32_t body_size = static_cast<uint32_t>(w.size() - code_start);
         w[body_start] = static_cast<uint8_t>(body_size);
      }

      // Function 5: bench_accumulate(n: i32) -> i64
      {
         size_t body_start = w.size();
         w.push_back(0);
         size_t code_start = w.size();

         emit_u32_leb(1);
         emit_u32_leb(1); emit(0x7f); // 1 x i32 ($i)

         emit(0x03, 0x40); // loop
           // call accumulate(i64.extend_i32_u(local.get $i))
           emit(0x20); emit_u32_leb(1); // local.get $i
           emit(0xAD);                   // i64.extend_i32_u
           emit(0x10); emit_u32_leb(1); // call $accumulate
           // i++
           emit(0x20); emit_u32_leb(1);
           emit(0x41, 0x01);
           emit(0x6A);
           emit(0x22); emit_u32_leb(1);
           emit(0x20); emit_u32_leb(0);
           emit(0x49);
           emit(0x0D, 0x00);
         emit(0x0B);
         // return 0
         emit(0x42, 0x00); // i64.const 0
         emit(0x0B);

         uint32_t body_size = static_cast<uint32_t>(w.size() - code_start);
         w[body_start] = static_cast<uint8_t>(body_size);
      }

      // Function 6: bench_mix(n: i32) -> i64
      {
         size_t body_start = w.size();
         w.push_back(0);
         size_t code_start = w.size();

         emit_u32_leb(2);
         emit_u32_leb(1); emit(0x7f); // $i
         emit_u32_leb(1); emit(0x7e); // $acc

         emit(0x42, 0x01); // i64.const 1
         emit(0x21); emit_u32_leb(2); // local.set $acc

         emit(0x03, 0x40); // loop
           // acc = mix(acc, i64.extend_i32_u(i))
           emit(0x20); emit_u32_leb(2); // local.get $acc
           emit(0x20); emit_u32_leb(1); // local.get $i
           emit(0xAD);                   // i64.extend_i32_u
           emit(0x10); emit_u32_leb(2); // call $mix
           emit(0x21); emit_u32_leb(2); // local.set $acc
           // i++
           emit(0x20); emit_u32_leb(1);
           emit(0x41, 0x01);
           emit(0x6A);
           emit(0x22); emit_u32_leb(1);
           emit(0x20); emit_u32_leb(0);
           emit(0x49);
           emit(0x0D, 0x00);
         emit(0x0B);
         emit(0x20); emit_u32_leb(2); // return acc
         emit(0x0B);

         uint32_t body_size = static_cast<uint32_t>(w.size() - code_start);
         w[body_start] = static_cast<uint8_t>(body_size);
      }
   });

   return w;
}

template<typename Impl>
double run_bench(wasm_code& code, wasm_allocator& wa, const std::string& func, uint32_t n) {
   using backend_t = backend<rhf_t, Impl>;
   backend_t bkend(code, &wa);
   rhf_t::resolve(bkend.get_module());
   bkend.initialize(nullptr);

   auto t1 = std::chrono::high_resolution_clock::now();
   bkend.call_with_return("env", func, n);
   auto t2 = std::chrono::high_resolution_clock::now();

   double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
   return ms;
}

int main() {
   wasm_allocator wa;

   rhf_t::add<&host_identity>("env", "identity");
   rhf_t::add<&host_accumulate>("env", "accumulate");
   rhf_t::add<&host_mix>("env", "mix");
   rhf_t::add<&host_read_mem>("env", "read_mem");

   auto wasm = build_host_call_wasm();
   wasm_code code(wasm.begin(), wasm.end());

   const uint32_t N = 10'000'000;

   struct bench_entry {
      const char* name;
      const char* func;
   };

   bench_entry benches[] = {
      {"identity (trivial)", "bench_identity"},
      {"accumulate (void)",  "bench_accumulate"},
      {"mix (2-arg hash)",   "bench_mix"},
   };

   printf("Host function call benchmark — %u iterations each\n\n", N);
   printf("%-22s %12s %12s %10s %10s\n", "Test", "Interp (ms)", "JIT (ms)", "Speedup", "Mcalls/s");
   printf("%s\n", std::string(68, '-').c_str());

   for (auto& b : benches) {
      double interp_ms = run_bench<interpreter>(code, wa, b.func, N);
      double jit_ms = run_bench<jit>(code, wa, b.func, N);
      double speedup = interp_ms / jit_ms;
      double mcalls = N / (jit_ms * 1000.0);

      printf("%-22s %12.1f %12.1f %9.1fx %9.1f\n",
             b.name, interp_ms, jit_ms, speedup, mcalls);
   }

   return 0;
}
