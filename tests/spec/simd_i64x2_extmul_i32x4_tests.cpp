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

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_extmul_i32x4_0_wasm>", "[simd_i64x2_extmul_i32x4_0_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_extmul_i32x4.0.wasm");
   backend_t bkend( code, &wa );

   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(0,0,0,0), make_v128_i32(0,0,0,0))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(0,0,0,0), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(1,1,1,1), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(1u,1u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(0,0,0,0), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(1,1,1,1), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(1u,1u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(1073741823,1073741823,1073741823,1073741823), make_v128_i32(1073741824,1073741824,1073741824,1073741824))->to_v128() == make_v128_i64(1152921503533105152u,1152921503533105152u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(1073741824,1073741824,1073741824,1073741824), make_v128_i32(1073741824,1073741824,1073741824,1073741824))->to_v128() == make_v128_i64(1152921504606846976u,1152921504606846976u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(3221225473,3221225473,3221225473,3221225473), make_v128_i32(3221225472,3221225472,3221225472,3221225472))->to_v128() == make_v128_i64(1152921503533105152u,1152921503533105152u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(3221225472,3221225472,3221225472,3221225472), make_v128_i32(3221225472,3221225472,3221225472,3221225472))->to_v128() == make_v128_i64(1152921504606846976u,1152921504606846976u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(3221225471,3221225471,3221225471,3221225471), make_v128_i32(3221225472,3221225472,3221225472,3221225472))->to_v128() == make_v128_i64(1152921505680588800u,1152921505680588800u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(2147483645,2147483645,2147483645,2147483645), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(2147483645u,2147483645u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(2147483646,2147483646,2147483646,2147483646), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(2147483646u,2147483646u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(2147483648,2147483648,2147483648,2147483648), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(18446744071562067968u,18446744071562067968u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(2147483650,2147483650,2147483650,2147483650), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(2147483646u,2147483646u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(2147483649,2147483649,2147483649,2147483649), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(2147483647u,2147483647u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(2147483648,2147483648,2147483648,2147483648), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(2147483648u,2147483648u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(2147483647,2147483647,2147483647,2147483647), make_v128_i32(2147483647,2147483647,2147483647,2147483647))->to_v128() == make_v128_i64(4611686014132420609u,4611686014132420609u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(2147483648,2147483648,2147483648,2147483648), make_v128_i32(2147483648,2147483648,2147483648,2147483648))->to_v128() == make_v128_i64(4611686018427387904u,4611686018427387904u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(2147483648,2147483648,2147483648,2147483648), make_v128_i32(2147483649,2147483649,2147483649,2147483649))->to_v128() == make_v128_i64(4611686016279904256u,4611686016279904256u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(0,0,0,0))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(1u,1u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(2147483647,2147483647,2147483647,2147483647))->to_v128() == make_v128_i64(18446744071562067969u,18446744071562067969u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(2147483648,2147483648,2147483648,2147483648))->to_v128() == make_v128_i64(2147483648u,2147483648u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_s", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(1u,1u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(0,0,0,0), make_v128_i32(0,0,0,0))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(0,0,0,0), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(1,1,1,1), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(1u,1u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(0,0,0,0), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(1,1,1,1), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(1u,1u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(1073741823,1073741823,1073741823,1073741823), make_v128_i32(1073741824,1073741824,1073741824,1073741824))->to_v128() == make_v128_i64(1152921503533105152u,1152921503533105152u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(1073741824,1073741824,1073741824,1073741824), make_v128_i32(1073741824,1073741824,1073741824,1073741824))->to_v128() == make_v128_i64(1152921504606846976u,1152921504606846976u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(3221225473,3221225473,3221225473,3221225473), make_v128_i32(3221225472,3221225472,3221225472,3221225472))->to_v128() == make_v128_i64(1152921503533105152u,1152921503533105152u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(3221225472,3221225472,3221225472,3221225472), make_v128_i32(3221225472,3221225472,3221225472,3221225472))->to_v128() == make_v128_i64(1152921504606846976u,1152921504606846976u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(3221225471,3221225471,3221225471,3221225471), make_v128_i32(3221225472,3221225472,3221225472,3221225472))->to_v128() == make_v128_i64(1152921505680588800u,1152921505680588800u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(2147483645,2147483645,2147483645,2147483645), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(2147483645u,2147483645u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(2147483646,2147483646,2147483646,2147483646), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(2147483646u,2147483646u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(2147483648,2147483648,2147483648,2147483648), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(18446744071562067968u,18446744071562067968u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(2147483650,2147483650,2147483650,2147483650), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(2147483646u,2147483646u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(2147483649,2147483649,2147483649,2147483649), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(2147483647u,2147483647u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(2147483648,2147483648,2147483648,2147483648), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(2147483648u,2147483648u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(2147483647,2147483647,2147483647,2147483647), make_v128_i32(2147483647,2147483647,2147483647,2147483647))->to_v128() == make_v128_i64(4611686014132420609u,4611686014132420609u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(2147483648,2147483648,2147483648,2147483648), make_v128_i32(2147483648,2147483648,2147483648,2147483648))->to_v128() == make_v128_i64(4611686018427387904u,4611686018427387904u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(2147483648,2147483648,2147483648,2147483648), make_v128_i32(2147483649,2147483649,2147483649,2147483649))->to_v128() == make_v128_i64(4611686016279904256u,4611686016279904256u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(0,0,0,0))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(1u,1u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(2147483647,2147483647,2147483647,2147483647))->to_v128() == make_v128_i64(18446744071562067969u,18446744071562067969u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(2147483648,2147483648,2147483648,2147483648))->to_v128() == make_v128_i64(2147483648u,2147483648u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_s", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(1u,1u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(0,0,0,0), make_v128_i32(0,0,0,0))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(0,0,0,0), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(1,1,1,1), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(1u,1u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(0,0,0,0), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(1,1,1,1), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(4294967295u,4294967295u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(18446744065119617025u,18446744065119617025u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(1073741823,1073741823,1073741823,1073741823), make_v128_i32(1073741824,1073741824,1073741824,1073741824))->to_v128() == make_v128_i64(1152921503533105152u,1152921503533105152u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(1073741824,1073741824,1073741824,1073741824), make_v128_i32(1073741824,1073741824,1073741824,1073741824))->to_v128() == make_v128_i64(1152921504606846976u,1152921504606846976u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(3221225473,3221225473,3221225473,3221225473), make_v128_i32(3221225472,3221225472,3221225472,3221225472))->to_v128() == make_v128_i64(10376293544682848256u,10376293544682848256u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(3221225472,3221225472,3221225472,3221225472), make_v128_i32(3221225472,3221225472,3221225472,3221225472))->to_v128() == make_v128_i64(10376293541461622784u,10376293541461622784u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(3221225471,3221225471,3221225471,3221225471), make_v128_i32(3221225472,3221225472,3221225472,3221225472))->to_v128() == make_v128_i64(10376293538240397312u,10376293538240397312u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(2147483645,2147483645,2147483645,2147483645), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(2147483645u,2147483645u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(2147483646,2147483646,2147483646,2147483646), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(2147483646u,2147483646u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(2147483648,2147483648,2147483648,2147483648), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(2147483648u,2147483648u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(2147483650,2147483650,2147483650,2147483650), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(9223372043297226750u,9223372043297226750u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(2147483649,2147483649,2147483649,2147483649), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(9223372039002259455u,9223372039002259455u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(2147483648,2147483648,2147483648,2147483648), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(9223372034707292160u,9223372034707292160u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(2147483647,2147483647,2147483647,2147483647), make_v128_i32(2147483647,2147483647,2147483647,2147483647))->to_v128() == make_v128_i64(4611686014132420609u,4611686014132420609u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(2147483648,2147483648,2147483648,2147483648), make_v128_i32(2147483648,2147483648,2147483648,2147483648))->to_v128() == make_v128_i64(4611686018427387904u,4611686018427387904u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(2147483648,2147483648,2147483648,2147483648), make_v128_i32(2147483649,2147483649,2147483649,2147483649))->to_v128() == make_v128_i64(4611686020574871552u,4611686020574871552u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(0,0,0,0))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(4294967295u,4294967295u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(18446744065119617025u,18446744065119617025u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(2147483647,2147483647,2147483647,2147483647))->to_v128() == make_v128_i64(9223372030412324865u,9223372030412324865u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(2147483648,2147483648,2147483648,2147483648))->to_v128() == make_v128_i64(9223372034707292160u,9223372034707292160u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_low_i32x4_u", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(18446744065119617025u,18446744065119617025u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(0,0,0,0), make_v128_i32(0,0,0,0))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(0,0,0,0), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(1,1,1,1), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(1u,1u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(0,0,0,0), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(1,1,1,1), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(4294967295u,4294967295u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(18446744065119617025u,18446744065119617025u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(1073741823,1073741823,1073741823,1073741823), make_v128_i32(1073741824,1073741824,1073741824,1073741824))->to_v128() == make_v128_i64(1152921503533105152u,1152921503533105152u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(1073741824,1073741824,1073741824,1073741824), make_v128_i32(1073741824,1073741824,1073741824,1073741824))->to_v128() == make_v128_i64(1152921504606846976u,1152921504606846976u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(3221225473,3221225473,3221225473,3221225473), make_v128_i32(3221225472,3221225472,3221225472,3221225472))->to_v128() == make_v128_i64(10376293544682848256u,10376293544682848256u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(3221225472,3221225472,3221225472,3221225472), make_v128_i32(3221225472,3221225472,3221225472,3221225472))->to_v128() == make_v128_i64(10376293541461622784u,10376293541461622784u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(3221225471,3221225471,3221225471,3221225471), make_v128_i32(3221225472,3221225472,3221225472,3221225472))->to_v128() == make_v128_i64(10376293538240397312u,10376293538240397312u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(2147483645,2147483645,2147483645,2147483645), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(2147483645u,2147483645u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(2147483646,2147483646,2147483646,2147483646), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(2147483646u,2147483646u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(2147483648,2147483648,2147483648,2147483648), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(2147483648u,2147483648u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(2147483650,2147483650,2147483650,2147483650), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(9223372043297226750u,9223372043297226750u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(2147483649,2147483649,2147483649,2147483649), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(9223372039002259455u,9223372039002259455u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(2147483648,2147483648,2147483648,2147483648), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(9223372034707292160u,9223372034707292160u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(2147483647,2147483647,2147483647,2147483647), make_v128_i32(2147483647,2147483647,2147483647,2147483647))->to_v128() == make_v128_i64(4611686014132420609u,4611686014132420609u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(2147483648,2147483648,2147483648,2147483648), make_v128_i32(2147483648,2147483648,2147483648,2147483648))->to_v128() == make_v128_i64(4611686018427387904u,4611686018427387904u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(2147483648,2147483648,2147483648,2147483648), make_v128_i32(2147483649,2147483649,2147483649,2147483649))->to_v128() == make_v128_i64(4611686020574871552u,4611686020574871552u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(0,0,0,0))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(1,1,1,1))->to_v128() == make_v128_i64(4294967295u,4294967295u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(18446744065119617025u,18446744065119617025u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(2147483647,2147483647,2147483647,2147483647))->to_v128() == make_v128_i64(9223372030412324865u,9223372030412324865u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(2147483648,2147483648,2147483648,2147483648))->to_v128() == make_v128_i64(9223372034707292160u,9223372034707292160u));
   CHECK(bkend.call_with_return("env", "i64x2.extmul_high_i32x4_u", make_v128_i32(4294967295,4294967295,4294967295,4294967295), make_v128_i32(4294967295,4294967295,4294967295,4294967295))->to_v128() == make_v128_i64(18446744065119617025u,18446744065119617025u));
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_extmul_i32x4_1_wasm>", "[simd_i64x2_extmul_i32x4_1_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_extmul_i32x4.1.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_extmul_i32x4_10_wasm>", "[simd_i64x2_extmul_i32x4_10_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_extmul_i32x4.10.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_extmul_i32x4_11_wasm>", "[simd_i64x2_extmul_i32x4_11_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_extmul_i32x4.11.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_extmul_i32x4_12_wasm>", "[simd_i64x2_extmul_i32x4_12_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_extmul_i32x4.12.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_extmul_i32x4_2_wasm>", "[simd_i64x2_extmul_i32x4_2_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_extmul_i32x4.2.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_extmul_i32x4_3_wasm>", "[simd_i64x2_extmul_i32x4_3_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_extmul_i32x4.3.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_extmul_i32x4_4_wasm>", "[simd_i64x2_extmul_i32x4_4_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_extmul_i32x4.4.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_extmul_i32x4_5_wasm>", "[simd_i64x2_extmul_i32x4_5_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_extmul_i32x4.5.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_extmul_i32x4_6_wasm>", "[simd_i64x2_extmul_i32x4_6_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_extmul_i32x4.6.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_extmul_i32x4_7_wasm>", "[simd_i64x2_extmul_i32x4_7_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_extmul_i32x4.7.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_extmul_i32x4_8_wasm>", "[simd_i64x2_extmul_i32x4_8_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_extmul_i32x4.8.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_extmul_i32x4_9_wasm>", "[simd_i64x2_extmul_i32x4_9_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_extmul_i32x4.9.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

