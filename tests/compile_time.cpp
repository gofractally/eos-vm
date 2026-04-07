#include <eosio/vm/backend.hpp>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdio>

using namespace eosio::vm;

int main(int argc, char** argv) {
   if (argc < 2) { fprintf(stderr, "Usage: %s file.wasm [file2.wasm ...]\n", argv[0]); return 1; }

   for (int a = 1; a < argc; a++) {
      std::ifstream f(argv[a], std::ios::binary);
      if (!f) { fprintf(stderr, "Cannot open %s\n", argv[a]); continue; }
      std::vector<uint8_t> wasm((std::istreambuf_iterator<char>(f)), {});

      wasm_allocator wa;

      // jit1 compile
      double jit1_ms;
      {
         using backend_t = backend<std::nullptr_t, jit>;
         auto t0 = std::chrono::high_resolution_clock::now();
         backend_t bkend(wasm, &wa);
         auto t1 = std::chrono::high_resolution_clock::now();
         jit1_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      }

      // jit2 compile
      double jit2_ms;
      uint32_t num_funcs;
      {
         using backend_t = backend<std::nullptr_t, jit2>;
         auto t0 = std::chrono::high_resolution_clock::now();
         backend_t bkend(wasm, &wa);
         auto t1 = std::chrono::high_resolution_clock::now();
         jit2_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
         num_funcs = bkend.get_module().code.size();
      }

      printf("%-50s %6zuKB %3u funcs  jit1=%6.2fms  jit2=%6.2fms  ratio=%.1fx\n",
             argv[a], wasm.size()/1024, num_funcs, jit1_ms, jit2_ms, jit2_ms/jit1_ms);
   }
   return 0;
}
