#include <eosio/vm/backend.hpp>
#include <catch2/catch.hpp>
#include "utils.hpp"

using namespace eosio;
using namespace eosio::vm;

extern wasm_allocator wa;

// (module
//   (func (export "add") (param i32 i32) (result i32)
//     local.get 0
//     local.get 1
//     i32.add)
// )
static std::vector<uint8_t> add_wasm = {
   0x00,0x61,0x73,0x6d, 0x01,0x00,0x00,0x00, // magic + version
   0x01,0x07,0x01,0x60,0x02,0x7f,0x7f,0x01,0x7f, // type: (i32,i32)->i32
   0x03,0x02,0x01,0x00,                           // func: type 0
   0x07,0x07,0x01,0x03,0x61,0x64,0x64,0x00,0x00,  // export "add" = func 0
   0x0a,0x09,0x01,0x07,0x00,0x20,0x00,0x20,0x01,0x6a,0x0b // code: get 0, get 1, i32.add, end
};

// (module
//   (func (export "max") (param i32 i32) (result i32)
//     local.get 0
//     local.get 1
//     i32.gt_s
//     if (result i32)
//       local.get 0
//     else
//       local.get 1
//     end)
// )
static std::vector<uint8_t> max_wasm = {
   0x00,0x61,0x73,0x6d, 0x01,0x00,0x00,0x00,
   0x01,0x07,0x01,0x60,0x02,0x7f,0x7f,0x01,0x7f,
   0x03,0x02,0x01,0x00,
   0x07,0x07,0x01,0x03,0x6d,0x61,0x78,0x00,0x00,
   0x0a,0x11,0x01,0x0f,0x00,
   0x20,0x00, 0x20,0x01, 0x4a,       // get 0, get 1, i32.gt_s
   0x04,0x7f,                         // if (result i32)
   0x20,0x00,                         // get 0
   0x05,                              // else
   0x20,0x01,                         // get 1
   0x0b,                              // end if
   0x0b                               // end func
};

// (module
//   (func (export "sum") (param i32) (result i32)
//     (local i32)
//     i32.const 0
//     local.set 1
//     block
//       loop
//         local.get 0
//         i32.eqz
//         br_if 1
//         local.get 1
//         local.get 0
//         i32.add
//         local.set 1
//         local.get 0
//         i32.const 1
//         i32.sub
//         local.set 0
//         br 0
//       end
//     end
//     local.get 1)
// )
static std::vector<uint8_t> sum_wasm = {
   0x00,0x61,0x73,0x6d, 0x01,0x00,0x00,0x00,
   0x01,0x06,0x01,0x60,0x01,0x7f,0x01,0x7f,
   0x03,0x02,0x01,0x00,
   0x07,0x07,0x01,0x03,0x73,0x75,0x6d,0x00,0x00,
   0x0a,0x23,0x01,0x21,0x01,0x01,0x7f,
   0x41,0x00,                   // i32.const 0
   0x21,0x01,                   // local.set 1
   0x02,0x40,                   // block
   0x03,0x40,                   // loop
   0x20,0x00,                   // local.get 0
   0x45,                        // i32.eqz
   0x0d,0x01,                   // br_if 1
   0x20,0x01,                   // local.get 1
   0x20,0x00,                   // local.get 0
   0x6a,                        // i32.add
   0x21,0x01,                   // local.set 1
   0x20,0x00,                   // local.get 0
   0x41,0x01,                   // i32.const 1
   0x6b,                        // i32.sub
   0x21,0x00,                   // local.set 0
   0x0c,0x00,                   // br 0
   0x0b,                        // end loop
   0x0b,                        // end block
   0x20,0x01,                   // local.get 1
   0x0b                         // end func
};

BACKEND_TEST_CASE("jit2 basic: i32.add", "[jit2_basic]") {
   using backend_t = backend<std::nullptr_t, TestType>;
   backend_t bkend(add_wasm, &wa);
   CHECK(bkend.call_with_return("env", "add", (uint32_t)3, (uint32_t)4)->to_ui32() == 7);
   CHECK(bkend.call_with_return("env", "add", (uint32_t)0, (uint32_t)0)->to_ui32() == 0);
   CHECK(bkend.call_with_return("env", "add", (uint32_t)100, (uint32_t)200)->to_ui32() == 300);
}

BACKEND_TEST_CASE("jit2 basic: if/else", "[jit2_basic]") {
   using backend_t = backend<std::nullptr_t, TestType>;
   backend_t bkend(max_wasm, &wa);
   CHECK(bkend.call_with_return("env", "max", (uint32_t)3, (uint32_t)7)->to_ui32() == 7);
   CHECK(bkend.call_with_return("env", "max", (uint32_t)10, (uint32_t)5)->to_ui32() == 10);
   CHECK(bkend.call_with_return("env", "max", (uint32_t)42, (uint32_t)42)->to_ui32() == 42);
}

BACKEND_TEST_CASE("jit2 basic: loop sum", "[jit2_basic]") {
   using backend_t = backend<std::nullptr_t, TestType>;
   backend_t bkend(sum_wasm, &wa);
   CHECK(bkend.call_with_return("env", "sum", (uint32_t)0)->to_ui32() == 0);
   CHECK(bkend.call_with_return("env", "sum", (uint32_t)1)->to_ui32() == 1);
   CHECK(bkend.call_with_return("env", "sum", (uint32_t)10)->to_ui32() == 55);
   CHECK(bkend.call_with_return("env", "sum", (uint32_t)100)->to_ui32() == 5050);
}
