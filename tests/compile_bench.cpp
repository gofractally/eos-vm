// Compile-time benchmark: measures JIT compilation time for all runtimes
#include <wasmtime.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
   const char* path = argc > 1 ? argv[1] : "benchmarks/bench_sha256.wasm";
   int runs = argc > 2 ? atoi(argv[2]) : 10;

   std::ifstream f(path, std::ios::binary);
   if (!f) { fprintf(stderr, "Cannot open %s\n", path); return 1; }
   std::vector<uint8_t> wasm((std::istreambuf_iterator<char>(f)), {});
   printf("Module: %s (%zu bytes)\n\n", path, wasm.size());

   // Wasmtime (Cranelift) compile time
   {
      double total = 0;
      for (int i = 0; i < runs; i++) {
         wasm_engine_t* engine = wasm_engine_new();
         auto t0 = std::chrono::high_resolution_clock::now();
         wasmtime_module_t* module = nullptr;
         wasmtime_error_t* err = wasmtime_module_new(engine, wasm.data(), wasm.size(), &module);
         auto t1 = std::chrono::high_resolution_clock::now();
         if (err) { fprintf(stderr, "wasmtime error\n"); wasmtime_error_delete(err); }
         double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
         total += ms;
         if (module) wasmtime_module_delete(module);
         wasm_engine_delete(engine);
      }
      printf("wasmtime (Cranelift): %.2f ms avg (%d runs)\n", total / runs, runs);
   }

   return 0;
}
