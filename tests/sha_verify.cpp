#include <eosio/vm/backend.hpp>
#include <fstream>
#include <vector>
#include <cstdio>

using namespace eosio::vm;

int main(int argc, char** argv) {
   const char* path = argc > 1 ? argv[1] : "benchmarks/bench_sha256.wasm";

   // Load WASM
   std::ifstream f(path, std::ios::binary);
   if (!f) { fprintf(stderr, "Cannot open %s\n", path); return 1; }
   std::vector<uint8_t> wasm((std::istreambuf_iterator<char>(f)), {});

   wasm_allocator wa;

   // Run with jit
   {
      using backend_t = backend<std::nullptr_t, jit>;
      backend_t bkend(wasm, &wa);
      bkend.initialize(nullptr);
      auto r = bkend.call_with_return("env", "bench_sha256", (uint32_t)1);
      printf("jit1: %ld\n", r ? r->to_i64() : -999);
   }

   // Run with jit2
   {
      using backend_t = backend<std::nullptr_t, jit2>;
      backend_t bkend(wasm, &wa);
      bkend.initialize(nullptr);
      auto r = bkend.call_with_return("env", "bench_sha256", (uint32_t)1);
      printf("jit2: %ld\n", r ? r->to_i64() : -999);
   }

   return 0;
}
