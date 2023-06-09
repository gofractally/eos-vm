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

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_extmul_i8x16_0_wasm>", "[simd_i16x8_extmul_i8x16_0_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_extmul_i8x16.0.wasm");
   backend_t bkend( code, &wa );

   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u), make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(65535u,65535u,65535u,65535u,65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u), make_v128_i8(64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u))->to_v128() == make_v128_i16(4032u,4032u,4032u,4032u,4032u,4032u,4032u,4032u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u), make_v128_i8(64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u))->to_v128() == make_v128_i16(4096u,4096u,4096u,4096u,4096u,4096u,4096u,4096u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u), make_v128_i8(192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u))->to_v128() == make_v128_i16(4032u,4032u,4032u,4032u,4032u,4032u,4032u,4032u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u), make_v128_i8(192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u))->to_v128() == make_v128_i16(4096u,4096u,4096u,4096u,4096u,4096u,4096u,4096u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u), make_v128_i8(192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u))->to_v128() == make_v128_i16(4160u,4160u,4160u,4160u,4160u,4160u,4160u,4160u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(125u,125u,125u,125u,125u,125u,125u,125u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(126u,126u,126u,126u,126u,126u,126u,126u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(65408u,65408u,65408u,65408u,65408u,65408u,65408u,65408u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(126u,126u,126u,126u,126u,126u,126u,126u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(127u,127u,127u,127u,127u,127u,127u,127u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(128u,128u,128u,128u,128u,128u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u), make_v128_i8(127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u))->to_v128() == make_v128_i16(16129u,16129u,16129u,16129u,16129u,16129u,16129u,16129u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u), make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u))->to_v128() == make_v128_i16(16384u,16384u,16384u,16384u,16384u,16384u,16384u,16384u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u), make_v128_i8(129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u))->to_v128() == make_v128_i16(16256u,16256u,16256u,16256u,16256u,16256u,16256u,16256u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(65535u,65535u,65535u,65535u,65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u))->to_v128() == make_v128_i16(65409u,65409u,65409u,65409u,65409u,65409u,65409u,65409u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u))->to_v128() == make_v128_i16(128u,128u,128u,128u,128u,128u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_s", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u), make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(65535u,65535u,65535u,65535u,65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u), make_v128_i8(64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u))->to_v128() == make_v128_i16(4032u,4032u,4032u,4032u,4032u,4032u,4032u,4032u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u), make_v128_i8(64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u))->to_v128() == make_v128_i16(4096u,4096u,4096u,4096u,4096u,4096u,4096u,4096u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u), make_v128_i8(192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u))->to_v128() == make_v128_i16(4032u,4032u,4032u,4032u,4032u,4032u,4032u,4032u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u), make_v128_i8(192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u))->to_v128() == make_v128_i16(4096u,4096u,4096u,4096u,4096u,4096u,4096u,4096u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u), make_v128_i8(192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u))->to_v128() == make_v128_i16(4160u,4160u,4160u,4160u,4160u,4160u,4160u,4160u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(125u,125u,125u,125u,125u,125u,125u,125u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(126u,126u,126u,126u,126u,126u,126u,126u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(65408u,65408u,65408u,65408u,65408u,65408u,65408u,65408u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(126u,126u,126u,126u,126u,126u,126u,126u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(127u,127u,127u,127u,127u,127u,127u,127u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(128u,128u,128u,128u,128u,128u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u), make_v128_i8(127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u))->to_v128() == make_v128_i16(16129u,16129u,16129u,16129u,16129u,16129u,16129u,16129u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u), make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u))->to_v128() == make_v128_i16(16384u,16384u,16384u,16384u,16384u,16384u,16384u,16384u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u), make_v128_i8(129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u))->to_v128() == make_v128_i16(16256u,16256u,16256u,16256u,16256u,16256u,16256u,16256u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(65535u,65535u,65535u,65535u,65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u))->to_v128() == make_v128_i16(65409u,65409u,65409u,65409u,65409u,65409u,65409u,65409u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u))->to_v128() == make_v128_i16(128u,128u,128u,128u,128u,128u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_s", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u), make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(255u,255u,255u,255u,255u,255u,255u,255u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(65025u,65025u,65025u,65025u,65025u,65025u,65025u,65025u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u), make_v128_i8(64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u))->to_v128() == make_v128_i16(4032u,4032u,4032u,4032u,4032u,4032u,4032u,4032u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u), make_v128_i8(64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u))->to_v128() == make_v128_i16(4096u,4096u,4096u,4096u,4096u,4096u,4096u,4096u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u), make_v128_i8(192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u))->to_v128() == make_v128_i16(37056u,37056u,37056u,37056u,37056u,37056u,37056u,37056u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u), make_v128_i8(192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u))->to_v128() == make_v128_i16(36864u,36864u,36864u,36864u,36864u,36864u,36864u,36864u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u), make_v128_i8(192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u))->to_v128() == make_v128_i16(36672u,36672u,36672u,36672u,36672u,36672u,36672u,36672u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(125u,125u,125u,125u,125u,125u,125u,125u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(126u,126u,126u,126u,126u,126u,126u,126u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(128u,128u,128u,128u,128u,128u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(33150u,33150u,33150u,33150u,33150u,33150u,33150u,33150u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(32895u,32895u,32895u,32895u,32895u,32895u,32895u,32895u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(32640u,32640u,32640u,32640u,32640u,32640u,32640u,32640u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u), make_v128_i8(127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u))->to_v128() == make_v128_i16(16129u,16129u,16129u,16129u,16129u,16129u,16129u,16129u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u), make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u))->to_v128() == make_v128_i16(16384u,16384u,16384u,16384u,16384u,16384u,16384u,16384u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u), make_v128_i8(129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u))->to_v128() == make_v128_i16(16512u,16512u,16512u,16512u,16512u,16512u,16512u,16512u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(255u,255u,255u,255u,255u,255u,255u,255u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(65025u,65025u,65025u,65025u,65025u,65025u,65025u,65025u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u))->to_v128() == make_v128_i16(32385u,32385u,32385u,32385u,32385u,32385u,32385u,32385u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u))->to_v128() == make_v128_i16(32640u,32640u,32640u,32640u,32640u,32640u,32640u,32640u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_low_i8x16_u", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(65025u,65025u,65025u,65025u,65025u,65025u,65025u,65025u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u), make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(1u,1u,1u,1u,1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(255u,255u,255u,255u,255u,255u,255u,255u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(65025u,65025u,65025u,65025u,65025u,65025u,65025u,65025u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u,63u), make_v128_i8(64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u))->to_v128() == make_v128_i16(4032u,4032u,4032u,4032u,4032u,4032u,4032u,4032u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u), make_v128_i8(64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u,64u))->to_v128() == make_v128_i16(4096u,4096u,4096u,4096u,4096u,4096u,4096u,4096u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u,193u), make_v128_i8(192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u))->to_v128() == make_v128_i16(37056u,37056u,37056u,37056u,37056u,37056u,37056u,37056u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u), make_v128_i8(192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u))->to_v128() == make_v128_i16(36864u,36864u,36864u,36864u,36864u,36864u,36864u,36864u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u,191u), make_v128_i8(192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u,192u))->to_v128() == make_v128_i16(36672u,36672u,36672u,36672u,36672u,36672u,36672u,36672u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u,125u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(125u,125u,125u,125u,125u,125u,125u,125u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u,126u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(126u,126u,126u,126u,126u,126u,126u,126u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(128u,128u,128u,128u,128u,128u,128u,128u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u,130u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(33150u,33150u,33150u,33150u,33150u,33150u,33150u,33150u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(32895u,32895u,32895u,32895u,32895u,32895u,32895u,32895u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(32640u,32640u,32640u,32640u,32640u,32640u,32640u,32640u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u), make_v128_i8(127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u))->to_v128() == make_v128_i16(16129u,16129u,16129u,16129u,16129u,16129u,16129u,16129u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u), make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u))->to_v128() == make_v128_i16(16384u,16384u,16384u,16384u,16384u,16384u,16384u,16384u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u), make_v128_i8(129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u,129u))->to_v128() == make_v128_i16(16512u,16512u,16512u,16512u,16512u,16512u,16512u,16512u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u))->to_v128() == make_v128_i16(0u,0u,0u,0u,0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u,1u))->to_v128() == make_v128_i16(255u,255u,255u,255u,255u,255u,255u,255u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(65025u,65025u,65025u,65025u,65025u,65025u,65025u,65025u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u,127u))->to_v128() == make_v128_i16(32385u,32385u,32385u,32385u,32385u,32385u,32385u,32385u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u,128u))->to_v128() == make_v128_i16(32640u,32640u,32640u,32640u,32640u,32640u,32640u,32640u));
   CHECK(bkend.call_with_return("env", "i16x8.extmul_high_i8x16_u", make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u), make_v128_i8(255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u,255u))->to_v128() == make_v128_i16(65025u,65025u,65025u,65025u,65025u,65025u,65025u,65025u));
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_extmul_i8x16_1_wasm>", "[simd_i16x8_extmul_i8x16_1_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_extmul_i8x16.1.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_extmul_i8x16_10_wasm>", "[simd_i16x8_extmul_i8x16_10_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_extmul_i8x16.10.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_extmul_i8x16_11_wasm>", "[simd_i16x8_extmul_i8x16_11_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_extmul_i8x16.11.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_extmul_i8x16_12_wasm>", "[simd_i16x8_extmul_i8x16_12_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_extmul_i8x16.12.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_extmul_i8x16_2_wasm>", "[simd_i16x8_extmul_i8x16_2_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_extmul_i8x16.2.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_extmul_i8x16_3_wasm>", "[simd_i16x8_extmul_i8x16_3_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_extmul_i8x16.3.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_extmul_i8x16_4_wasm>", "[simd_i16x8_extmul_i8x16_4_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_extmul_i8x16.4.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_extmul_i8x16_5_wasm>", "[simd_i16x8_extmul_i8x16_5_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_extmul_i8x16.5.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_extmul_i8x16_6_wasm>", "[simd_i16x8_extmul_i8x16_6_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_extmul_i8x16.6.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_extmul_i8x16_7_wasm>", "[simd_i16x8_extmul_i8x16_7_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_extmul_i8x16.7.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_extmul_i8x16_8_wasm>", "[simd_i16x8_extmul_i8x16_8_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_extmul_i8x16.8.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i16x8_extmul_i8x16_9_wasm>", "[simd_i16x8_extmul_i8x16_9_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i16x8_extmul_i8x16.9.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

