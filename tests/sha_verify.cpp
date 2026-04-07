#include <eosio/vm/backend.hpp>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdio>

using namespace eosio::vm;

int main(int argc, char** argv) {
   const char* path = argc > 1 ? argv[1] : "benchmarks/bench_sha256.wasm";
   const char* func = argc > 2 ? argv[2] : "bench_sha256";
   uint32_t iters = argc > 3 ? atoi(argv[3]) : 10000;

   std::ifstream f(path, std::ios::binary);
   if (!f) { fprintf(stderr, "Cannot open %s\n", path); return 1; }
   std::vector<uint8_t> wasm((std::istreambuf_iterator<char>(f)), {});

   wasm_allocator wa;

   // Run with jit1
   {
      using backend_t = backend<std::nullptr_t, jit>;
      backend_t bkend(wasm, &wa);
      bkend.initialize(nullptr);

      auto t0 = std::chrono::high_resolution_clock::now();
      auto r = bkend.call_with_return("env", func, iters);
      auto t1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      printf("jit1: %ld  (%.1f ms)\n", r ? r->to_i64() : -999, ms);
   }

   // Run with jit2
   {
      using backend_t = backend<std::nullptr_t, jit2>;
      backend_t bkend(wasm, &wa);
      bkend.initialize(nullptr);

      auto t0 = std::chrono::high_resolution_clock::now();
      auto r = bkend.call_with_return("env", func, iters);
      auto t1 = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      printf("jit2: %ld  (%.1f ms)\n", r ? r->to_i64() : -999, ms);
   }

   return 0;
}
