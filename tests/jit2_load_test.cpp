#include <eosio/vm/backend.hpp>
#include <fstream>
#include <vector>
#include <cstdio>
using namespace eosio::vm;
int main(int argc, char** argv) {
   if (argc < 2) { fprintf(stderr, "Usage: %s file.wasm\n", argv[0]); return 1; }
   for (int a = 1; a < argc; a++) {
      std::ifstream f(argv[a], std::ios::binary);
      if (!f) { fprintf(stderr, "Cannot open %s\n", argv[a]); continue; }
      std::vector<uint8_t> wasm((std::istreambuf_iterator<char>(f)), {});
      wasm_allocator wa;
      try {
         using backend_t = backend<std::nullptr_t, jit2>;
         backend_t bkend(wasm, &wa);
         printf("OK: %s (%zu bytes, %lu funcs)\n", argv[a], wasm.size(), bkend.get_module().code.size());
      } catch (const std::exception& e) {
         printf("FAIL: %s — %s\n", argv[a], e.what());
      }
   }
   return 0;
}
