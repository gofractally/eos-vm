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

BACKEND_TEST_CASE( "Testing wasm <simd_i32x4_extmul_i16x8_0_wasm>", "[simd_i32x4_extmul_i16x8_0_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i32x4_extmul_i16x8.0.wasm");
   backend_t bkend( code, &wa );

   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i32(0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(4294967295u,4294967295u,4294967295u,4294967295u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(16383,16383,16383,16383,16383,16383,16383,16383), make_v128_i16(16384,16384,16384,16384,16384,16384,16384,16384))->to_v128() == make_v128_i32(268419072u,268419072u,268419072u,268419072u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(16384,16384,16384,16384,16384,16384,16384,16384), make_v128_i16(16384,16384,16384,16384,16384,16384,16384,16384))->to_v128() == make_v128_i32(268435456u,268435456u,268435456u,268435456u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(49153,49153,49153,49153,49153,49153,49153,49153), make_v128_i16(49152,49152,49152,49152,49152,49152,49152,49152))->to_v128() == make_v128_i32(268419072u,268419072u,268419072u,268419072u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(49152,49152,49152,49152,49152,49152,49152,49152), make_v128_i16(49152,49152,49152,49152,49152,49152,49152,49152))->to_v128() == make_v128_i32(268435456u,268435456u,268435456u,268435456u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(49151,49151,49151,49151,49151,49151,49151,49151), make_v128_i16(49152,49152,49152,49152,49152,49152,49152,49152))->to_v128() == make_v128_i32(268451840u,268451840u,268451840u,268451840u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(32765,32765,32765,32765,32765,32765,32765,32765), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(32765u,32765u,32765u,32765u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(32766,32766,32766,32766,32766,32766,32766,32766), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(32766u,32766u,32766u,32766u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(4294934528u,4294934528u,4294934528u,4294934528u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(32770,32770,32770,32770,32770,32770,32770,32770), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(32766u,32766u,32766u,32766u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(32769,32769,32769,32769,32769,32769,32769,32769), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(32767u,32767u,32767u,32767u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(32767,32767,32767,32767,32767,32767,32767,32767), make_v128_i16(32767,32767,32767,32767,32767,32767,32767,32767))->to_v128() == make_v128_i32(1073676289u,1073676289u,1073676289u,1073676289u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i32(1073741824u,1073741824u,1073741824u,1073741824u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32769,32769,32769,32769,32769,32769,32769,32769))->to_v128() == make_v128_i32(1073709056u,1073709056u,1073709056u,1073709056u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i32(0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(4294967295u,4294967295u,4294967295u,4294967295u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(32767,32767,32767,32767,32767,32767,32767,32767))->to_v128() == make_v128_i32(4294934529u,4294934529u,4294934529u,4294934529u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i32(32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i32(0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(4294967295u,4294967295u,4294967295u,4294967295u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(16383,16383,16383,16383,16383,16383,16383,16383), make_v128_i16(16384,16384,16384,16384,16384,16384,16384,16384))->to_v128() == make_v128_i32(268419072u,268419072u,268419072u,268419072u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(16384,16384,16384,16384,16384,16384,16384,16384), make_v128_i16(16384,16384,16384,16384,16384,16384,16384,16384))->to_v128() == make_v128_i32(268435456u,268435456u,268435456u,268435456u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(49153,49153,49153,49153,49153,49153,49153,49153), make_v128_i16(49152,49152,49152,49152,49152,49152,49152,49152))->to_v128() == make_v128_i32(268419072u,268419072u,268419072u,268419072u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(49152,49152,49152,49152,49152,49152,49152,49152), make_v128_i16(49152,49152,49152,49152,49152,49152,49152,49152))->to_v128() == make_v128_i32(268435456u,268435456u,268435456u,268435456u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(49151,49151,49151,49151,49151,49151,49151,49151), make_v128_i16(49152,49152,49152,49152,49152,49152,49152,49152))->to_v128() == make_v128_i32(268451840u,268451840u,268451840u,268451840u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(32765,32765,32765,32765,32765,32765,32765,32765), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(32765u,32765u,32765u,32765u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(32766,32766,32766,32766,32766,32766,32766,32766), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(32766u,32766u,32766u,32766u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(4294934528u,4294934528u,4294934528u,4294934528u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(32770,32770,32770,32770,32770,32770,32770,32770), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(32766u,32766u,32766u,32766u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(32769,32769,32769,32769,32769,32769,32769,32769), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(32767u,32767u,32767u,32767u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(32767,32767,32767,32767,32767,32767,32767,32767), make_v128_i16(32767,32767,32767,32767,32767,32767,32767,32767))->to_v128() == make_v128_i32(1073676289u,1073676289u,1073676289u,1073676289u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i32(1073741824u,1073741824u,1073741824u,1073741824u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32769,32769,32769,32769,32769,32769,32769,32769))->to_v128() == make_v128_i32(1073709056u,1073709056u,1073709056u,1073709056u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i32(0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(4294967295u,4294967295u,4294967295u,4294967295u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(32767,32767,32767,32767,32767,32767,32767,32767))->to_v128() == make_v128_i32(4294934529u,4294934529u,4294934529u,4294934529u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i32(32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_s", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i32(0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(4294836225u,4294836225u,4294836225u,4294836225u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(16383,16383,16383,16383,16383,16383,16383,16383), make_v128_i16(16384,16384,16384,16384,16384,16384,16384,16384))->to_v128() == make_v128_i32(268419072u,268419072u,268419072u,268419072u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(16384,16384,16384,16384,16384,16384,16384,16384), make_v128_i16(16384,16384,16384,16384,16384,16384,16384,16384))->to_v128() == make_v128_i32(268435456u,268435456u,268435456u,268435456u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(49153,49153,49153,49153,49153,49153,49153,49153), make_v128_i16(49152,49152,49152,49152,49152,49152,49152,49152))->to_v128() == make_v128_i32(2415968256u,2415968256u,2415968256u,2415968256u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(49152,49152,49152,49152,49152,49152,49152,49152), make_v128_i16(49152,49152,49152,49152,49152,49152,49152,49152))->to_v128() == make_v128_i32(2415919104u,2415919104u,2415919104u,2415919104u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(49151,49151,49151,49151,49151,49151,49151,49151), make_v128_i16(49152,49152,49152,49152,49152,49152,49152,49152))->to_v128() == make_v128_i32(2415869952u,2415869952u,2415869952u,2415869952u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(32765,32765,32765,32765,32765,32765,32765,32765), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(32765u,32765u,32765u,32765u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(32766,32766,32766,32766,32766,32766,32766,32766), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(32766u,32766u,32766u,32766u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(32770,32770,32770,32770,32770,32770,32770,32770), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(2147581950u,2147581950u,2147581950u,2147581950u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(32769,32769,32769,32769,32769,32769,32769,32769), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(2147516415u,2147516415u,2147516415u,2147516415u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(2147450880u,2147450880u,2147450880u,2147450880u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(32767,32767,32767,32767,32767,32767,32767,32767), make_v128_i16(32767,32767,32767,32767,32767,32767,32767,32767))->to_v128() == make_v128_i32(1073676289u,1073676289u,1073676289u,1073676289u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i32(1073741824u,1073741824u,1073741824u,1073741824u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32769,32769,32769,32769,32769,32769,32769,32769))->to_v128() == make_v128_i32(1073774592u,1073774592u,1073774592u,1073774592u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i32(0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(4294836225u,4294836225u,4294836225u,4294836225u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(32767,32767,32767,32767,32767,32767,32767,32767))->to_v128() == make_v128_i32(2147385345u,2147385345u,2147385345u,2147385345u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i32(2147450880u,2147450880u,2147450880u,2147450880u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_low_i16x8_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(4294836225u,4294836225u,4294836225u,4294836225u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i32(0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(1u,1u,1u,1u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(0,0,0,0,0,0,0,0), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(1,1,1,1,1,1,1,1), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(4294836225u,4294836225u,4294836225u,4294836225u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(16383,16383,16383,16383,16383,16383,16383,16383), make_v128_i16(16384,16384,16384,16384,16384,16384,16384,16384))->to_v128() == make_v128_i32(268419072u,268419072u,268419072u,268419072u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(16384,16384,16384,16384,16384,16384,16384,16384), make_v128_i16(16384,16384,16384,16384,16384,16384,16384,16384))->to_v128() == make_v128_i32(268435456u,268435456u,268435456u,268435456u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(49153,49153,49153,49153,49153,49153,49153,49153), make_v128_i16(49152,49152,49152,49152,49152,49152,49152,49152))->to_v128() == make_v128_i32(2415968256u,2415968256u,2415968256u,2415968256u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(49152,49152,49152,49152,49152,49152,49152,49152), make_v128_i16(49152,49152,49152,49152,49152,49152,49152,49152))->to_v128() == make_v128_i32(2415919104u,2415919104u,2415919104u,2415919104u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(49151,49151,49151,49151,49151,49151,49151,49151), make_v128_i16(49152,49152,49152,49152,49152,49152,49152,49152))->to_v128() == make_v128_i32(2415869952u,2415869952u,2415869952u,2415869952u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(32765,32765,32765,32765,32765,32765,32765,32765), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(32765u,32765u,32765u,32765u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(32766,32766,32766,32766,32766,32766,32766,32766), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(32766u,32766u,32766u,32766u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(32768u,32768u,32768u,32768u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(32770,32770,32770,32770,32770,32770,32770,32770), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(2147581950u,2147581950u,2147581950u,2147581950u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(32769,32769,32769,32769,32769,32769,32769,32769), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(2147516415u,2147516415u,2147516415u,2147516415u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(2147450880u,2147450880u,2147450880u,2147450880u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(32767,32767,32767,32767,32767,32767,32767,32767), make_v128_i16(32767,32767,32767,32767,32767,32767,32767,32767))->to_v128() == make_v128_i32(1073676289u,1073676289u,1073676289u,1073676289u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i32(1073741824u,1073741824u,1073741824u,1073741824u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768), make_v128_i16(32769,32769,32769,32769,32769,32769,32769,32769))->to_v128() == make_v128_i32(1073774592u,1073774592u,1073774592u,1073774592u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(0,0,0,0,0,0,0,0))->to_v128() == make_v128_i32(0u,0u,0u,0u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(1,1,1,1,1,1,1,1))->to_v128() == make_v128_i32(65535u,65535u,65535u,65535u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(4294836225u,4294836225u,4294836225u,4294836225u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(32767,32767,32767,32767,32767,32767,32767,32767))->to_v128() == make_v128_i32(2147385345u,2147385345u,2147385345u,2147385345u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(32768,32768,32768,32768,32768,32768,32768,32768))->to_v128() == make_v128_i32(2147450880u,2147450880u,2147450880u,2147450880u));
   CHECK(bkend.call_with_return("env", "i32x4.extmul_high_i16x8_u", make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535), make_v128_i16(65535,65535,65535,65535,65535,65535,65535,65535))->to_v128() == make_v128_i32(4294836225u,4294836225u,4294836225u,4294836225u));
}

BACKEND_TEST_CASE( "Testing wasm <simd_i32x4_extmul_i16x8_1_wasm>", "[simd_i32x4_extmul_i16x8_1_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i32x4_extmul_i16x8.1.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i32x4_extmul_i16x8_10_wasm>", "[simd_i32x4_extmul_i16x8_10_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i32x4_extmul_i16x8.10.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i32x4_extmul_i16x8_11_wasm>", "[simd_i32x4_extmul_i16x8_11_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i32x4_extmul_i16x8.11.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i32x4_extmul_i16x8_12_wasm>", "[simd_i32x4_extmul_i16x8_12_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i32x4_extmul_i16x8.12.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i32x4_extmul_i16x8_2_wasm>", "[simd_i32x4_extmul_i16x8_2_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i32x4_extmul_i16x8.2.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i32x4_extmul_i16x8_3_wasm>", "[simd_i32x4_extmul_i16x8_3_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i32x4_extmul_i16x8.3.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i32x4_extmul_i16x8_4_wasm>", "[simd_i32x4_extmul_i16x8_4_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i32x4_extmul_i16x8.4.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i32x4_extmul_i16x8_5_wasm>", "[simd_i32x4_extmul_i16x8_5_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i32x4_extmul_i16x8.5.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i32x4_extmul_i16x8_6_wasm>", "[simd_i32x4_extmul_i16x8_6_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i32x4_extmul_i16x8.6.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i32x4_extmul_i16x8_7_wasm>", "[simd_i32x4_extmul_i16x8_7_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i32x4_extmul_i16x8.7.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i32x4_extmul_i16x8_8_wasm>", "[simd_i32x4_extmul_i16x8_8_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i32x4_extmul_i16x8.8.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i32x4_extmul_i16x8_9_wasm>", "[simd_i32x4_extmul_i16x8_9_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i32x4_extmul_i16x8.9.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

