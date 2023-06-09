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

BACKEND_TEST_CASE( "Testing wasm <bulk_0_wasm>", "[bulk_0_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "bulk.0.wasm");
   backend_t bkend( code, &wa );

}

BACKEND_TEST_CASE( "Testing wasm <bulk_1_wasm>", "[bulk_1_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "bulk.1.wasm");
   backend_t bkend( code, &wa );

}

BACKEND_TEST_CASE( "Testing wasm <bulk_10_wasm>", "[bulk_10_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "bulk.10.wasm");
   backend_t bkend( code, &wa );

}

BACKEND_TEST_CASE( "Testing wasm <bulk_11_wasm>", "[bulk_11_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "bulk.11.wasm");
   backend_t bkend( code, &wa );

}

BACKEND_TEST_CASE( "Testing wasm <bulk_12_wasm>", "[bulk_12_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "bulk.12.wasm");
   backend_t bkend( code, &wa );

bkend("env", "copy", UINT32_C(3), UINT32_C(0), UINT32_C(3));
   CHECK(bkend.call_with_return("env", "call", UINT32_C(3))->to_ui32() == UINT32_C(0));
   CHECK(bkend.call_with_return("env", "call", UINT32_C(4))->to_ui32() == UINT32_C(1));
   CHECK(bkend.call_with_return("env", "call", UINT32_C(5))->to_ui32() == UINT32_C(2));
bkend("env", "copy", UINT32_C(0), UINT32_C(1), UINT32_C(3));
   CHECK(bkend.call_with_return("env", "call", UINT32_C(0))->to_ui32() == UINT32_C(1));
   CHECK(bkend.call_with_return("env", "call", UINT32_C(1))->to_ui32() == UINT32_C(2));
   CHECK(bkend.call_with_return("env", "call", UINT32_C(2))->to_ui32() == UINT32_C(0));
bkend("env", "copy", UINT32_C(2), UINT32_C(0), UINT32_C(3));
   CHECK(bkend.call_with_return("env", "call", UINT32_C(2))->to_ui32() == UINT32_C(1));
   CHECK(bkend.call_with_return("env", "call", UINT32_C(3))->to_ui32() == UINT32_C(2));
   CHECK(bkend.call_with_return("env", "call", UINT32_C(4))->to_ui32() == UINT32_C(0));
bkend("env", "copy", UINT32_C(6), UINT32_C(8), UINT32_C(2));
bkend("env", "copy", UINT32_C(8), UINT32_C(6), UINT32_C(2));
bkend("env", "copy", UINT32_C(10), UINT32_C(0), UINT32_C(0));
bkend("env", "copy", UINT32_C(0), UINT32_C(10), UINT32_C(0));
   CHECK_THROWS_AS(bkend("env", "copy", UINT32_C(11), UINT32_C(0), UINT32_C(0)), std::exception);
   CHECK_THROWS_AS(bkend("env", "copy", UINT32_C(0), UINT32_C(11), UINT32_C(0)), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <bulk_2_wasm>", "[bulk_2_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "bulk.2.wasm");
   backend_t bkend( code, &wa );

bkend("env", "fill", UINT32_C(1), UINT32_C(255), UINT32_C(3));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(0))->to_ui32() == UINT32_C(0));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(1))->to_ui32() == UINT32_C(255));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(2))->to_ui32() == UINT32_C(255));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(3))->to_ui32() == UINT32_C(255));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(4))->to_ui32() == UINT32_C(0));
bkend("env", "fill", UINT32_C(0), UINT32_C(48042), UINT32_C(2));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(0))->to_ui32() == UINT32_C(170));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(1))->to_ui32() == UINT32_C(170));
bkend("env", "fill", UINT32_C(0), UINT32_C(0), UINT32_C(65536));
   CHECK_THROWS_AS(bkend("env", "fill", UINT32_C(65280), UINT32_C(1), UINT32_C(257)), std::exception);
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(65280))->to_ui32() == UINT32_C(0));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(65535))->to_ui32() == UINT32_C(0));
bkend("env", "fill", UINT32_C(65536), UINT32_C(0), UINT32_C(0));
   CHECK_THROWS_AS(bkend("env", "fill", UINT32_C(65537), UINT32_C(0), UINT32_C(0)), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <bulk_3_wasm>", "[bulk_3_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "bulk.3.wasm");
   backend_t bkend( code, &wa );

bkend("env", "copy", UINT32_C(10), UINT32_C(0), UINT32_C(4));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(9))->to_ui32() == UINT32_C(0));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(10))->to_ui32() == UINT32_C(170));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(11))->to_ui32() == UINT32_C(187));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(12))->to_ui32() == UINT32_C(204));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(13))->to_ui32() == UINT32_C(221));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(14))->to_ui32() == UINT32_C(0));
bkend("env", "copy", UINT32_C(8), UINT32_C(10), UINT32_C(4));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(8))->to_ui32() == UINT32_C(170));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(9))->to_ui32() == UINT32_C(187));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(10))->to_ui32() == UINT32_C(204));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(11))->to_ui32() == UINT32_C(221));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(12))->to_ui32() == UINT32_C(204));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(13))->to_ui32() == UINT32_C(221));
bkend("env", "copy", UINT32_C(10), UINT32_C(7), UINT32_C(6));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(10))->to_ui32() == UINT32_C(0));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(11))->to_ui32() == UINT32_C(170));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(12))->to_ui32() == UINT32_C(187));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(13))->to_ui32() == UINT32_C(204));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(14))->to_ui32() == UINT32_C(221));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(15))->to_ui32() == UINT32_C(204));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(16))->to_ui32() == UINT32_C(0));
bkend("env", "copy", UINT32_C(65280), UINT32_C(0), UINT32_C(256));
bkend("env", "copy", UINT32_C(65024), UINT32_C(65280), UINT32_C(256));
bkend("env", "copy", UINT32_C(65536), UINT32_C(0), UINT32_C(0));
bkend("env", "copy", UINT32_C(0), UINT32_C(65536), UINT32_C(0));
   CHECK_THROWS_AS(bkend("env", "copy", UINT32_C(65537), UINT32_C(0), UINT32_C(0)), std::exception);
   CHECK_THROWS_AS(bkend("env", "copy", UINT32_C(0), UINT32_C(65537), UINT32_C(0)), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <bulk_4_wasm>", "[bulk_4_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "bulk.4.wasm");
   backend_t bkend( code, &wa );

bkend("env", "init", UINT32_C(0), UINT32_C(1), UINT32_C(2));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(0))->to_ui32() == UINT32_C(187));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(1))->to_ui32() == UINT32_C(204));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(2))->to_ui32() == UINT32_C(0));
bkend("env", "init", UINT32_C(65532), UINT32_C(0), UINT32_C(4));
   CHECK_THROWS_AS(bkend("env", "init", UINT32_C(65534), UINT32_C(0), UINT32_C(3)), std::exception);
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(65534))->to_ui32() == UINT32_C(204));
   CHECK(bkend.call_with_return("env", "load8_u", UINT32_C(65535))->to_ui32() == UINT32_C(221));
bkend("env", "init", UINT32_C(65536), UINT32_C(0), UINT32_C(0));
bkend("env", "init", UINT32_C(0), UINT32_C(4), UINT32_C(0));
   CHECK_THROWS_AS(bkend("env", "init", UINT32_C(65537), UINT32_C(0), UINT32_C(0)), std::exception);
   CHECK_THROWS_AS(bkend("env", "init", UINT32_C(0), UINT32_C(5), UINT32_C(0)), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <bulk_5_wasm>", "[bulk_5_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "bulk.5.wasm");
   backend_t bkend( code, &wa );

bkend("env", "init_passive", UINT32_C(1));
bkend("env", "drop_passive");
bkend("env", "drop_passive");
   CHECK(!bkend.call_with_return("env", "init_passive", UINT32_C(0)));
   CHECK_THROWS_AS(bkend("env", "init_passive", UINT32_C(1)), std::exception);
bkend("env", "init_passive", UINT32_C(0));
bkend("env", "drop_active");
   CHECK(!bkend.call_with_return("env", "init_active", UINT32_C(0)));
   CHECK_THROWS_AS(bkend("env", "init_active", UINT32_C(1)), std::exception);
bkend("env", "init_active", UINT32_C(0));
}

BACKEND_TEST_CASE( "Testing wasm <bulk_6_wasm>", "[bulk_6_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "bulk.6.wasm");
   backend_t bkend( code, &wa );

}

BACKEND_TEST_CASE( "Testing wasm <bulk_7_wasm>", "[bulk_7_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "bulk.7.wasm");
   backend_t bkend( code, &wa );

}

BACKEND_TEST_CASE( "Testing wasm <bulk_8_wasm>", "[bulk_8_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "bulk.8.wasm");
   backend_t bkend( code, &wa );

   CHECK_THROWS_AS(bkend("env", "init", UINT32_C(2), UINT32_C(0), UINT32_C(2)), std::exception);
   CHECK_THROWS_AS(bkend("env", "call", UINT32_C(2)), std::exception);
bkend("env", "init", UINT32_C(0), UINT32_C(1), UINT32_C(2));
   CHECK(bkend.call_with_return("env", "call", UINT32_C(0))->to_ui32() == UINT32_C(1));
   CHECK(bkend.call_with_return("env", "call", UINT32_C(1))->to_ui32() == UINT32_C(0));
   CHECK_THROWS_AS(bkend("env", "call", UINT32_C(2)), std::exception);
bkend("env", "init", UINT32_C(1), UINT32_C(2), UINT32_C(2));
bkend("env", "init", UINT32_C(3), UINT32_C(0), UINT32_C(0));
bkend("env", "init", UINT32_C(0), UINT32_C(4), UINT32_C(0));
   CHECK_THROWS_AS(bkend("env", "init", UINT32_C(4), UINT32_C(0), UINT32_C(0)), std::exception);
   CHECK_THROWS_AS(bkend("env", "init", UINT32_C(0), UINT32_C(5), UINT32_C(0)), std::exception);
}

BACKEND_TEST_CASE( "Testing wasm <bulk_9_wasm>", "[bulk_9_wasm_tests]" ) {
   using backend_t = backend<standalone_function_t, TestType>;
   auto code = read_wasm( std::string(wasm_directory) + "bulk.9.wasm");
   backend_t bkend( code, &wa );

bkend("env", "init_passive", UINT32_C(1));
bkend("env", "drop_passive");
bkend("env", "drop_passive");
   CHECK(!bkend.call_with_return("env", "init_passive", UINT32_C(0)));
   CHECK_THROWS_AS(bkend("env", "init_passive", UINT32_C(1)), std::exception);
bkend("env", "init_passive", UINT32_C(0));
bkend("env", "drop_active");
   CHECK(!bkend.call_with_return("env", "init_active", UINT32_C(0)));
   CHECK_THROWS_AS(bkend("env", "init_active", UINT32_C(1)), std::exception);
bkend("env", "init_active", UINT32_C(0));
}

