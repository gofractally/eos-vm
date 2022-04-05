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

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_0_wasm>", "[simd_i16x8_arith2_0_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.0.wasm");
   backend_t bkend( code, &wa );

   CHECK(bkend.call_with_return("env", "i16x8.min_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(65535u,65535u,65535u,65535u,65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s", make_v128_i16(0,0,0,0,65535,65535,65535,65535), make_v128_i16(0,0,65535,65535,0,0,65535,65535))->to_v128() == make_v128_i16(0u,0u,65535u,65535u,65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(65535u,65535u,65535u,65535u,65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s", make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i16(65535u,65535u,65535u,65535u,65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(128,128,128,128,128,128,128,128))->to_v128() == make_v128_i16(65535u,65535u,65535u,65535u,65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s", make_v128_i16(123,123,123,123,123,123,123,123), make_v128_i16(123,123,123,123,123,123,123,123))->to_v128() == make_v128_i16(123u,123u,123u,123u,123u,123u,123u,123u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s", make_v128_i16(128,128,128,128,128,128,128,128), make_v128_i16(128,128,128,128,128,128,128,128))->to_v128() == make_v128_i16(128u,128u,128u,128u,128u,128u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u", make_v128_i16(0,0,0,0,65535,65535,65535,65535), make_v128_i16(0,0,65535,65535,0,0,65535,65535))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u", make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(128,128,128,128,128,128,128,128))->to_v128() == make_v128_i16(128u,128u,128u,128u,128u,128u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u", make_v128_i16(123,123,123,123,123,123,123,123), make_v128_i16(123,123,123,123,123,123,123,123))->to_v128() == make_v128_i16(123u,123u,123u,123u,123u,123u,123u,123u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u", make_v128_i16(128,128,128,128,128,128,128,128), make_v128_i16(128,128,128,128,128,128,128,128))->to_v128() == make_v128_i16(128u,128u,128u,128u,128u,128u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s", make_v128_i16(0,0,0,0,65535,65535,65535,65535), make_v128_i16(0,0,65535,65535,0,0,65535,65535))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s", make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(128,128,128,128,128,128,128,128))->to_v128() == make_v128_i16(128u,128u,128u,128u,128u,128u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s", make_v128_i16(123,123,123,123,123,123,123,123), make_v128_i16(123,123,123,123,123,123,123,123))->to_v128() == make_v128_i16(123u,123u,123u,123u,123u,123u,123u,123u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s", make_v128_i16(128,128,128,128,128,128,128,128), make_v128_i16(128,128,128,128,128,128,128,128))->to_v128() == make_v128_i16(128u,128u,128u,128u,128u,128u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(65535u,65535u,65535u,65535u,65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u", make_v128_i16(0,0,0,0,65535,65535,65535,65535), make_v128_i16(0,0,65535,65535,0,0,65535,65535))->to_v128() == make_v128_i16(0u,0u,65535u,65535u,65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(65535u,65535u,65535u,65535u,65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u", make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i16(65535u,65535u,65535u,65535u,65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(128,128,128,128,128,128,128,128))->to_v128() == make_v128_i16(65535u,65535u,65535u,65535u,65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u", make_v128_i16(123,123,123,123,123,123,123,123), make_v128_i16(123,123,123,123,123,123,123,123))->to_v128() == make_v128_i16(123u,123u,123u,123u,123u,123u,123u,123u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u", make_v128_i16(128,128,128,128,128,128,128,128), make_v128_i16(128,128,128,128,128,128,128,128))->to_v128() == make_v128_i16(128u,128u,128u,128u,128u,128u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u", make_v128_i16(0,0,0,0,65535,65535,65535,65535), make_v128_i16(0,0,65535,65535,0,0,65535,65535))->to_v128() == make_v128_i16(0u,0u,32768u,32768u,32768u,32768u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u", make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(128,128,128,128,128,128,128,128))->to_v128() == make_v128_i16(32832u,32832u,32832u,32832u,32832u,32832u,32832u,32832u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u", make_v128_i16(123,123,123,123,123,123,123,123), make_v128_i16(123,123,123,123,123,123,123,123))->to_v128() == make_v128_i16(123u,123u,123u,123u,123u,123u,123u,123u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u", make_v128_i16(128,128,128,128,128,128,128,128), make_v128_i16(128,128,128,128,128,128,128,128))->to_v128() == make_v128_i16(128u,128u,128u,128u,128u,128u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(123,123,123,123,123,123,123,123))->to_v128() == make_v128_i16(123u,123u,123u,123u,123u,123u,123u,123u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(65413,65413,65413,65413,65413,65413,65413,65413))->to_v128() == make_v128_i16(123u,123u,123u,123u,123u,123u,123u,123u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(128,128,128,128,128,128,128,128))->to_v128() == make_v128_i16(128u,128u,128u,128u,128u,128u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(65408,65408,65408,65408,65408,65408,65408,65408))->to_v128() == make_v128_i16(128u,128u,128u,128u,128u,128u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(128,128,128,128,128,128,128,128))->to_v128() == make_v128_i16(128u,128u,128u,128u,128u,128u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(65408,65408,65408,65408,65408,65408,65408,65408))->to_v128() == make_v128_i16(128u,128u,128u,128u,128u,128u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s_with_const_0")->to_v128() == make_v128_i16(32768u,32768u,16384u,16384u,16384u,16384u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s_with_const_1")->to_v128() == make_v128_i16(0u,0u,1u,1u,1u,1u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u_with_const_2")->to_v128() == make_v128_i16(32768u,32768u,16384u,16384u,16384u,16384u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u_with_const_3")->to_v128() == make_v128_i16(0u,0u,1u,1u,1u,1u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s_with_const_4")->to_v128() == make_v128_i16(65535u,65535u,32767u,32767u,32767u,32767u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s_with_const_5")->to_v128() == make_v128_i16(3u,3u,2u,2u,2u,2u,3u,3u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u_with_const_6")->to_v128() == make_v128_i16(65535u,65535u,32767u,32767u,32767u,32767u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u_with_const_7")->to_v128() == make_v128_i16(3u,3u,2u,2u,2u,2u,3u,3u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u_with_const_8")->to_v128() == make_v128_i16(49152u,49152u,24576u,24576u,24576u,24576u,49152u,49152u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u_with_const_9")->to_v128() == make_v128_i16(2u,2u,2u,2u,2u,2u,2u,2u));
   CHECK(bkend.call_with_return("env", "i16x8.abs_with_const_10")->to_v128() == make_v128_i16(32768u,32768u,32767u,32767u,16384u,16384u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s_with_const_11", make_v128_i16(65535,65535,16384,16384,32767,32767,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,16384u,16384u,16384u,16384u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s_with_const_12", make_v128_i16(3,3,2,2,1,1,0,0))->to_v128() == make_v128_i16(0u,0u,1u,1u,1u,1u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u_with_const_13", make_v128_i16(65535,65535,16384,16384,32767,32767,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,16384u,16384u,16384u,16384u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u_with_const_14", make_v128_i16(3,3,2,2,1,1,0,0))->to_v128() == make_v128_i16(0u,0u,1u,1u,1u,1u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s_with_const_15", make_v128_i16(65535,65535,16384,16384,32767,32767,32768,32768))->to_v128() == make_v128_i16(65535u,65535u,32767u,32767u,32767u,32767u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s_with_const_16", make_v128_i16(3,3,2,2,1,1,0,0))->to_v128() == make_v128_i16(3u,3u,2u,2u,2u,2u,3u,3u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u_with_const_17", make_v128_i16(65535,65535,16384,16384,32767,32767,32768,32768))->to_v128() == make_v128_i16(65535u,65535u,32767u,32767u,32767u,32767u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u_with_const_18", make_v128_i16(3,3,2,2,1,1,0,0))->to_v128() == make_v128_i16(3u,3u,2u,2u,2u,2u,3u,3u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u_with_const_19", make_v128_i16(65535,65535,16384,16384,32767,32767,32768,32768))->to_v128() == make_v128_i16(49152u,49152u,24576u,24576u,24576u,24576u,49152u,49152u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u_with_const_20", make_v128_i16(3,3,2,2,1,1,0,0))->to_v128() == make_v128_i16(2u,2u,2u,2u,2u,2u,2u,2u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s", make_v128_i16(32768,32768,32767,32767,16384,16384,65535,65535), make_v128_i16(65535,65535,16384,16384,32767,32767,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,16384u,16384u,16384u,16384u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s", make_v128_i16(0,0,1,1,2,2,128,128), make_v128_i16(0,0,2,2,1,1,128,128))->to_v128() == make_v128_i16(0u,0u,1u,1u,1u,1u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u", make_v128_i16(32768,32768,32767,32767,16384,16384,65535,65535), make_v128_i16(65535,65535,16384,16384,32767,32767,32768,32768))->to_v128() == make_v128_i16(32768u,32768u,16384u,16384u,16384u,16384u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u", make_v128_i16(0,0,1,1,2,2,128,128), make_v128_i16(0,0,2,2,1,1,128,128))->to_v128() == make_v128_i16(0u,0u,1u,1u,1u,1u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s", make_v128_i16(32768,32768,32767,32767,16384,16384,65535,65535), make_v128_i16(65535,65535,16384,16384,32767,32767,32768,32768))->to_v128() == make_v128_i16(65535u,65535u,32767u,32767u,32767u,32767u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s", make_v128_i16(0,0,1,1,2,2,128,128), make_v128_i16(0,0,2,2,1,1,128,128))->to_v128() == make_v128_i16(0u,0u,2u,2u,2u,2u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u", make_v128_i16(32768,32768,32767,32767,16384,16384,65535,65535), make_v128_i16(65535,65535,16384,16384,32767,32767,32768,32768))->to_v128() == make_v128_i16(65535u,65535u,32767u,32767u,32767u,32767u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u", make_v128_i16(0,0,1,1,2,2,128,128), make_v128_i16(0,0,2,2,1,1,128,128))->to_v128() == make_v128_i16(0u,0u,2u,2u,2u,2u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u", make_v128_i16(32768,32768,32767,32767,16384,16384,65535,65535), make_v128_i16(65535,65535,16384,16384,32767,32767,32768,32768))->to_v128() == make_v128_i16(49152u,49152u,24576u,24576u,24576u,24576u,49152u,49152u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u", make_v128_i16(0,0,1,1,2,2,128,128), make_v128_i16(0,0,2,2,1,1,128,128))->to_v128() == make_v128_i16(0u,0u,2u,2u,2u,2u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(32768,32768,32767,32767,16384,16384,65535,65535))->to_v128() == make_v128_i16(32768u,32768u,32767u,32767u,16384u,16384u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.abs", make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_10_wasm>", "[simd_i16x8_arith2_10_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.10.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_11_wasm>", "[simd_i16x8_arith2_11_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.11.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_12_wasm>", "[simd_i16x8_arith2_12_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.12.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_13_wasm>", "[simd_i16x8_arith2_13_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.13.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_14_wasm>", "[simd_i16x8_arith2_14_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.14.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_15_wasm>", "[simd_i16x8_arith2_15_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.15.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_16_wasm>", "[simd_i16x8_arith2_16_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.16.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_17_wasm>", "[simd_i16x8_arith2_17_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.17.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_18_wasm>", "[simd_i16x8_arith2_18_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.18.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_19_wasm>", "[simd_i16x8_arith2_19_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.19.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_20_wasm>", "[simd_i16x8_arith2_20_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.20.wasm");
   backend_t bkend( code, &wa );

   CHECK(bkend.call_with_return("env", "i16x8.min_s-i16x8.avgr_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s-i16x8.max_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s-i16x8.max_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s-i16x8.min_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s-i16x8.min_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.min_s-i16x8.abs", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.abs-i16x8.min_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u-i16x8.avgr_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u-i16x8.max_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u-i16x8.max_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u-i16x8.min_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u-i16x8.min_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.min_u-i16x8.abs", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.abs-i16x8.min_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s-i16x8.avgr_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(2u,2u,2u,2u,2u,2u,2u,2u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s-i16x8.max_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(2u,2u,2u,2u,2u,2u,2u,2u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s-i16x8.max_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(2u,2u,2u,2u,2u,2u,2u,2u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s-i16x8.min_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(2u,2u,2u,2u,2u,2u,2u,2u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s-i16x8.min_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(2u,2u,2u,2u,2u,2u,2u,2u));
   CHECK(bkend.call_with_return("env", "i16x8.max_s-i16x8.abs", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.abs-i16x8.max_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u-i16x8.avgr_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(2u,2u,2u,2u,2u,2u,2u,2u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u-i16x8.max_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(2u,2u,2u,2u,2u,2u,2u,2u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u-i16x8.max_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(2u,2u,2u,2u,2u,2u,2u,2u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u-i16x8.min_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(2u,2u,2u,2u,2u,2u,2u,2u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u-i16x8.min_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(2u,2u,2u,2u,2u,2u,2u,2u));
   CHECK(bkend.call_with_return("env", "i16x8.max_u-i16x8.abs", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.abs-i16x8.max_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u-i16x8.avgr_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(2u,2u,2u,2u,2u,2u,2u,2u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u-i16x8.max_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(2u,2u,2u,2u,2u,2u,2u,2u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u-i16x8.max_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(2u,2u,2u,2u,2u,2u,2u,2u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u-i16x8.min_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u-i16x8.min_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(2,2,2,2,2,2,2,2))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.avgr_u-i16x8.abs", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.abs-i16x8.avgr_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(32768u,32768u,32768u,32768u,32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i16x8.abs-i16x8.abs", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_3_wasm>", "[simd_i16x8_arith2_3_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.3.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_4_wasm>", "[simd_i16x8_arith2_4_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.4.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_5_wasm>", "[simd_i16x8_arith2_5_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.5.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_6_wasm>", "[simd_i16x8_arith2_6_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.6.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_7_wasm>", "[simd_i16x8_arith2_7_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.7.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_8_wasm>", "[simd_i16x8_arith2_8_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.8.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_arith2_9_wasm>", "[simd_i16x8_arith2_9_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_arith2.9.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

