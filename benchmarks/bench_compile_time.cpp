// Compile-time benchmark: measures WASM module compilation/parsing speed
#include "bench_hosts.hpp"
#include <eosio/vm/backend.hpp>
#include <eosio/vm/utils.hpp>
#include <chrono>
#include <cstdio>

using namespace eosio::vm;

int main(int argc, char** argv) {
   if (argc < 2) { fprintf(stderr, "Usage: %s <wasm_file> [iterations]\n", argv[0]); return 1; }
   int iters = (argc > 2) ? atoi(argv[2]) : 1000;

   auto wasm_bytes = read_wasm(argv[1]);
   wasm_code code_vec(wasm_bytes.begin(), wasm_bytes.end());

   printf("Module: %s (%zu bytes)\n\n", argv[1], wasm_bytes.size());

   // eos-vm JIT compile time
#if defined(__x86_64__) || defined(__aarch64__)
   {
      auto t1 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < iters; i++) {
         wasm_allocator wa;
         backend<std::nullptr_t, jit> bkend(code_vec, &wa);
      }
      auto t2 = std::chrono::high_resolution_clock::now();
      double us = std::chrono::duration<double, std::micro>(t2 - t1).count() / iters;
      printf("eos-vm JIT compile:  %8.1f us  (%d iterations)\n", us, iters);
   }
#endif

   // eos-vm interpreter parse time
   {
      auto t1 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < iters; i++) {
         wasm_allocator wa;
         backend<std::nullptr_t, interpreter> bkend(code_vec, &wa);
      }
      auto t2 = std::chrono::high_resolution_clock::now();
      double us = std::chrono::duration<double, std::micro>(t2 - t1).count() / iters;
      printf("eos-vm interp parse: %8.1f us  (%d iterations)\n", us, iters);
   }

   return 0;
}
