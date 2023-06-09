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

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_cmp_0_wasm>", "[simd_i64x2_cmp_0_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_cmp.0.wasm");
   backend_t bkend( code, &wa );

   CHECK(bkend.call_with_return("env", "eq", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "eq", make_v128_i64(0u,0u), make_v128_i64(0u,0u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "eq", make_v128_i64(17361641481138401520u,17361641481138401520u), make_v128_i64(17361641481138401520u,17361641481138401520u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "eq", make_v128_i64(1085102592571150095u,1085102592571150095u), make_v128_i64(1085102592571150095u,1085102592571150095u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "eq", make_v128_i64(18446744073709551615u,0u), make_v128_i64(18446744073709551615u,0u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "eq", make_v128_i64(0u,18446744073709551615u), make_v128_i64(0u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "eq", make_v128_i64(50462976u,286263556u), make_v128_i64(50462976u,286263556u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "eq", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(1152921504606846975u,1152921504606846975u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "eq", make_v128_i64(1u,1u), make_v128_i64(2u,2u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "ne", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "ne", make_v128_i64(0u,0u), make_v128_i64(0u,0u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "ne", make_v128_i64(17361641481138401520u,17361641481138401520u), make_v128_i64(17361641481138401520u,17361641481138401520u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "ne", make_v128_i64(1085102592571150095u,1085102592571150095u), make_v128_i64(1085102592571150095u,1085102592571150095u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "ne", make_v128_i64(18446744073709551615u,0u), make_v128_i64(18446744073709551615u,0u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "ne", make_v128_i64(0u,18446744073709551615u), make_v128_i64(0u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "ne", make_v128_i64(50462976u,286263556u), make_v128_i64(50462976u,286263556u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(0u,0u), make_v128_i64(0u,0u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(17361641481138401520u,17361641481138401520u), make_v128_i64(17361641481138401520u,17361641481138401520u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(1085102592571150095u,1085102592571150095u), make_v128_i64(1085102592571150095u,1085102592571150095u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(18446744073709551615u,0u), make_v128_i64(18446744073709551615u,0u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(0u,18446744073709551615u), make_v128_i64(0u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(216736831865096452u,1876604746445072923u), make_v128_i64(216736831865096452u,1876604746445072923u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(9259542123273814144u,9259542123273814144u), make_v128_i64(9259542123273814144u,9259542123273814144u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(9259542123273814144u,9259542123273814144u), make_v128_i64(9259542123273814144u,9259542123273814144u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(9476278952713518845u,9151878496576798080u), make_v128_i64(9476278952713518845u,9151878496576798080u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(0u,0u), make_v128_i64(0u,0u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(18446744073709551615u,0u), make_v128_i64(18446744073709551615u,0u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(0u,18446744073709551615u), make_v128_i64(0u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(9223372036854775809u,18446744073709551615u), make_v128_i64(9223372036854775809u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(13862079653046386688u,13862009284302209024u), make_v128_f64(13862079653046386688u,13862009284302209024u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "lt_s", make_v128_i64(4607182418800017408u,4638637247447433216u), make_v128_f64(4607182418800017408u,4638637247447433216u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(0u,0u), make_v128_i64(0u,0u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(17361641481138401520u,17361641481138401520u), make_v128_i64(17361641481138401520u,17361641481138401520u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(1085102592571150095u,1085102592571150095u), make_v128_i64(1085102592571150095u,1085102592571150095u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(18446744073709551615u,0u), make_v128_i64(18446744073709551615u,0u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(0u,18446744073709551615u), make_v128_i64(0u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(216736831865096452u,1876604746445072923u), make_v128_i64(216736831865096452u,1876604746445072923u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(9259542123273814144u,9259542123273814144u), make_v128_i64(9259542123273814144u,9259542123273814144u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(9259542123273814144u,9259542123273814144u), make_v128_i64(9259542123273814144u,9259542123273814144u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(9476278952713518845u,9151878496576798080u), make_v128_i64(9476278952713518845u,9151878496576798080u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(0u,0u), make_v128_i64(0u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,0u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(0u,0u), make_v128_i64(0u,0u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(18446744073709551615u,0u), make_v128_i64(18446744073709551615u,0u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(0u,18446744073709551615u), make_v128_i64(0u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(9223372036854775809u,18446744073709551615u), make_v128_i64(9223372036854775809u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(13862079653046386688u,13862009284302209024u), make_v128_f64(13862079653046386688u,13862009284302209024u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "le_s", make_v128_i64(4607182418800017408u,4638637247447433216u), make_v128_f64(4607182418800017408u,4638637247447433216u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(0u,0u), make_v128_i64(0u,0u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(17361641481138401520u,17361641481138401520u), make_v128_i64(17361641481138401520u,17361641481138401520u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(1085102592571150095u,1085102592571150095u), make_v128_i64(1085102592571150095u,1085102592571150095u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(18446744073709551615u,0u), make_v128_i64(18446744073709551615u,0u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(0u,18446744073709551615u), make_v128_i64(0u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(216736831865096452u,1876604746445072923u), make_v128_i64(216736831865096452u,1876604746445072923u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(9259542123273814144u,9259542123273814144u), make_v128_i64(9259542123273814144u,9259542123273814144u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(9259542123273814144u,9259542123273814144u), make_v128_i64(9259542123273814144u,9259542123273814144u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(9476278952713518845u,9151878496576798080u), make_v128_i64(9476278952713518845u,9151878496576798080u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(0u,0u), make_v128_i64(0u,0u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(18446744073709551615u,0u), make_v128_i64(18446744073709551615u,0u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(0u,18446744073709551615u), make_v128_i64(0u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(9223372036854775809u,18446744073709551615u), make_v128_i64(9223372036854775809u,18446744073709551615u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(13862079653046386688u,13862009284302209024u), make_v128_f64(13862079653046386688u,13862009284302209024u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "gt_s", make_v128_i64(4607182418800017408u,4638637247447433216u), make_v128_f64(4607182418800017408u,4638637247447433216u))->to_v128() == make_v128_i64(0u,0u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(0u,0u), make_v128_i64(0u,0u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(17361641481138401520u,17361641481138401520u), make_v128_i64(17361641481138401520u,17361641481138401520u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(1085102592571150095u,1085102592571150095u), make_v128_i64(1085102592571150095u,1085102592571150095u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(18446744073709551615u,0u), make_v128_i64(18446744073709551615u,0u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(0u,18446744073709551615u), make_v128_i64(0u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(216736831865096452u,1876604746445072923u), make_v128_i64(216736831865096452u,1876604746445072923u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(9259542123273814144u,9259542123273814144u), make_v128_i64(9259542123273814144u,9259542123273814144u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(9259542123273814144u,9259542123273814144u), make_v128_i64(9259542123273814144u,9259542123273814144u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(9476278952713518845u,9151878496576798080u), make_v128_i64(9476278952713518845u,9151878496576798080u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(0u,18446744073709551615u))->to_v128() == make_v128_i64(0u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(0u,0u), make_v128_i64(0u,0u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(18446744073709551615u,18446744073709551615u), make_v128_i64(18446744073709551615u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(18446744073709551615u,0u), make_v128_i64(18446744073709551615u,0u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(0u,18446744073709551615u), make_v128_i64(0u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(9223372036854775809u,18446744073709551615u), make_v128_i64(9223372036854775809u,18446744073709551615u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(13862079653046386688u,13862009284302209024u), make_v128_f64(13862079653046386688u,13862009284302209024u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
   CHECK(bkend.call_with_return("env", "ge_s", make_v128_i64(4607182418800017408u,4638637247447433216u), make_v128_f64(4607182418800017408u,4638637247447433216u))->to_v128() == make_v128_i64(18446744073709551615u,18446744073709551615u));
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_cmp_1_wasm>", "[simd_i64x2_cmp_1_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_cmp.1.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_cmp_10_wasm>", "[simd_i64x2_cmp_10_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_cmp.10.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_cmp_2_wasm>", "[simd_i64x2_cmp_2_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_cmp.2.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_cmp_3_wasm>", "[simd_i64x2_cmp_3_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_cmp.3.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_cmp_4_wasm>", "[simd_i64x2_cmp_4_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_cmp.4.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_cmp_5_wasm>", "[simd_i64x2_cmp_5_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_cmp.5.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_cmp_6_wasm>", "[simd_i64x2_cmp_6_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_cmp.6.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_cmp_7_wasm>", "[simd_i64x2_cmp_7_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_cmp.7.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_cmp_8_wasm>", "[simd_i64x2_cmp_8_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_cmp.8.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <simd_i64x2_cmp_9_wasm>", "[simd_i64x2_cmp_9_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "simd_i64x2_cmp.9.wasm");
   CHECK_THROWS_AS(backend_t(code, nullptr), std::exception);
}

