#include <algorithm>
#include <vector>
#include <iterator>
#include <cstdlib>
#include <fstream>
#include <string>

#include <catch2/catch.hpp>

#include <eosio/vm/backend.hpp>
#include "wasm_config.hpp"
#include "utils.hpp"

using namespace eosio;
using namespace eosio::vm;

namespace {

void host_call() {}

struct dynamic_options {
   std::uint32_t max_stack_bytes;
   static constexpr std::uint32_t max_func_local_bytes = 0xFFFFFFFFu;
};

wasm_code max_stack_bytes_wasm_code = {
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x60,
  0x00, 0x00, 0x60, 0x01, 0x7f, 0x00, 0x02, 0x11, 0x01, 0x03, 0x65, 0x6e,
  0x76, 0x09, 0x68, 0x6f, 0x73, 0x74, 0x2e, 0x63, 0x61, 0x6c, 0x6c, 0x00,
  0x00, 0x03, 0x03, 0x02, 0x00, 0x01, 0x07, 0x15, 0x02, 0x05, 0x65, 0x6d,
  0x70, 0x74, 0x79, 0x00, 0x01, 0x09, 0x69, 0x33, 0x32, 0x2e, 0x73, 0x74,
  0x61, 0x63, 0x6b, 0x00, 0x02, 0x0a, 0x13, 0x02, 0x02, 0x00, 0x0b, 0x0e,
  0x00, 0x20, 0x00, 0x04, 0x40, 0x20, 0x00, 0x41, 0x7f, 0x6a, 0x10, 0x02,
  0x0b, 0x0b
};

}

BACKEND_TEST_CASE( "Test max stack bytes", "[max_stack_bytes]") {
   wasm_allocator wa;
   using rhf_t     = eosio::vm::registered_host_functions<standalone_function_t>;
   using backend_t = eosio::vm::backend<rhf_t, TestType, dynamic_options>;

   rhf_t::add<&host_call>("env", "host.call");

   backend_t bkend(max_stack_bytes_wasm_code, get_wasm_allocator(), dynamic_options{48});

   rhf_t::resolve(bkend.get_module());

   CHECK(!bkend.call_with_return("env", "i32.stack", (uint32_t)1));
   CHECK_THROWS_AS(bkend.call("env", "i32.stack", (uint32_t)2), std::exception);
}
