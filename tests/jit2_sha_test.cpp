#include <eosio/vm/backend.hpp>
#include <catch2/catch.hpp>
#include "utils.hpp"

using namespace eosio;
using namespace eosio::vm;

extern wasm_allocator wa;

BACKEND_TEST_CASE("jit2 SHA-256 correctness", "[jit2_sha]") {
   using backend_t = backend<std::nullptr_t, TestType>;
   auto code = read_wasm(std::string(BENCHMARK_WASM_DIR) + "/bench_sha256.wasm");
   backend_t bkend(code, &wa);
   auto result = bkend.call_with_return("env", "bench_sha256", (uint32_t)1);
   REQUIRE(result.has_value());
   // All backends should produce the same result
   CHECK(result->to_i64() == -68490791073730997LL);
}
