// Generated by spec_test_generator.  DO NOT MODIFY THIS FILE.

#include <algorithm>
#include <vector>
#include <iostream>
#include <iterator>
#include <cmath>
#include <cstdlib>
#include <catch2/catch.hpp>
#include <utils.hpp>
#include <wasm_config.hpp>
#include <eosio/vm/backend.hpp>

using namespace eosio;
using namespace eosio::vm;
extern wasm_allocator wa;

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_arith2_0_wasm>", "[simd_i64x2_arith2_0_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_arith2.0.wasm");
   backend_t bkend( code, &wa );

   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(1u,1u))->to_v128() == make_v128_i64(1u,1u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(1u,1u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(1u,1u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(1u,1u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(9223372036854775808u,9223372036854775808u))->to_v128() == make_v128_i64(9223372036854775808u,9223372036854775808u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(9223372036854775808u,9223372036854775808u))->to_v128() == make_v128_i64(9223372036854775808u,9223372036854775808u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(9223372036854775808u,9223372036854775808u))->to_v128() == make_v128_i64(9223372036854775808u,9223372036854775808u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(9223372036854775808u,9223372036854775808u))->to_v128() == make_v128_i64(9223372036854775808u,9223372036854775808u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(123u,123u))->to_v128() == make_v128_i64(123u,123u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(18446744073709551493u,18446744073709551493u))->to_v128() == make_v128_i64(123u,123u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(128u,128u))->to_v128() == make_v128_i64(128u,128u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(18446744073709551488u,18446744073709551488u))->to_v128() == make_v128_i64(128u,128u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(128u,128u))->to_v128() == make_v128_i64(128u,128u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(18446744073709551488u,18446744073709551488u))->to_v128() == make_v128_i64(128u,128u));
   CHECK(bkend.call_with_return("env", "i64x2.abs_with_const_0")->to_v128() == make_v128_i64(9223372036854775808u,9223372036854775807u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(9223372036854775808u,9223372036854775807u))->to_v128() == make_v128_i64(9223372036854775808u,9223372036854775807u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(0u,0u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(0u,0u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(0u,0u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.abs", make_v128_i64(0u,0u))->to_v128() == make_v128_i64(0u,0u));
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_arith2_1_wasm>", "[simd_i64x2_arith2_1_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_arith2.1.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_arith2_2_wasm>", "[simd_i64x2_arith2_2_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_arith2.2.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_arith2_3_wasm>", "[simd_i64x2_arith2_3_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_arith2.3.wasm");
   backend_t bkend( code, &wa );

   CHECK(bkend.call_with_return("env", "i64x2.abs-i64x2.abs", make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(1u,1u));
}

