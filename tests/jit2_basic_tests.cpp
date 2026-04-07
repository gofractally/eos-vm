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

// (module
//   (func (export "sum") (param i32) (result i32)
//     (local i32)
//     i32.const 0  local.set 1
//     block  loop
//       local.get 0  i32.eqz  br_if 1
//       local.get 1  local.get 0  i32.add  local.set 1
//       local.get 0  i32.const 1  i32.sub  local.set 0
//       br 0
//     end end
//     local.get 1))
static std::vector<uint8_t> sum_wasm = {
   0x00,0x61,0x73,0x6d, 0x01,0x00,0x00,0x00,           // magic + version
   0x01,0x06,0x01,0x60,0x01,0x7f,0x01,0x7f,             // type section: (i32)->i32
   0x03,0x02,0x01,0x00,                                   // func section: type 0
   0x07,0x07,0x01,0x03,0x73,0x75,0x6d,0x00,0x00,         // export "sum" = func 0
   0x0a,0x27,0x01,                                         // code section: size=39, 1 body
   0x25,                                                   // body size=37
   0x01,0x01,0x7f,                                         // 1 local of type i32
   0x41,0x00,                                               // i32.const 0
   0x21,0x01,                                               // local.set 1
   0x02,0x40,                                               // block (void)
   0x03,0x40,                                               // loop (void)
   0x20,0x00,                                               // local.get 0
   0x45,                                                     // i32.eqz
   0x0d,0x01,                                               // br_if 1
   0x20,0x01,                                               // local.get 1
   0x20,0x00,                                               // local.get 0
   0x6a,                                                     // i32.add
   0x21,0x01,                                               // local.set 1
   0x20,0x00,                                               // local.get 0
   0x41,0x01,                                               // i32.const 1
   0x6b,                                                     // i32.sub
   0x21,0x00,                                               // local.set 0
   0x0c,0x00,                                               // br 0
   0x0b,                                                     // end loop
   0x0b,                                                     // end block
   0x20,0x01,                                               // local.get 1
   0x0b                                                      // end func
};

BACKEND_TEST_CASE("jit2 basic: loop sum", "[jit2_basic]") {
   using backend_t = backend<std::nullptr_t, TestType>;
   backend_t bkend(sum_wasm, &wa);
   CHECK(bkend.call_with_return("env", "sum", (uint32_t)0)->to_ui32() == 0);
   CHECK(bkend.call_with_return("env", "sum", (uint32_t)1)->to_ui32() == 1);
   CHECK(bkend.call_with_return("env", "sum", (uint32_t)10)->to_ui32() == 55);
   CHECK(bkend.call_with_return("env", "sum", (uint32_t)100)->to_ui32() == 5050);
}

// (func (export "spill") (param i32) (result i32)
//   (local i32 i32 i32 i32 i32 i32 i32 i32 i32)  ;; 9 body locals (10 total)
//   ;; init: local[i] = param + i for i=1..9
//   ;; then push all 10, add them all → result = 10*param + 45
// )
static std::vector<uint8_t> spill_wasm = []() {
   std::vector<uint8_t> w;
   auto emit = [&](std::initializer_list<uint8_t> bytes) { for (auto b : bytes) w.push_back(b); };
   // Module header
   emit({0x00,0x61,0x73,0x6d, 0x01,0x00,0x00,0x00});
   // Type section: (i32)->i32
   emit({0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f});
   // Func section
   emit({0x03, 0x02, 0x01, 0x00});
   // Export "spill"
   emit({0x07, 0x09, 0x01, 0x05, 0x73, 0x70, 0x69, 0x6c, 0x6c, 0x00, 0x00});
   // Code section
   std::vector<uint8_t> body;
   body.push_back(0x01); body.push_back(0x09); body.push_back(0x7f); // 9 i32 locals
   // Init locals 1-9: local.get 0, i32.const i, i32.add, local.set i
   for (uint8_t i = 1; i <= 9; ++i) {
      body.push_back(0x20); body.push_back(0x00); // local.get 0
      body.push_back(0x41); body.push_back(i);    // i32.const i
      body.push_back(0x6a);                        // i32.add
      body.push_back(0x21); body.push_back(i);    // local.set i
   }
   // Push all 10 locals (forces 10 live values → spills with 8 regs)
   for (uint8_t i = 1; i <= 9; ++i) {
      body.push_back(0x20); body.push_back(i); // local.get i
   }
   body.push_back(0x20); body.push_back(0x00); // local.get 0
   // 9 i32.adds
   for (int i = 0; i < 9; ++i) body.push_back(0x6a);
   body.push_back(0x0b); // end
   // Encode code section
   uint8_t body_size = static_cast<uint8_t>(body.size());
   uint8_t section_size = 1 + 1 + body_size;
   w.push_back(0x0a);
   w.push_back(section_size);
   w.push_back(0x01); // 1 function
   w.push_back(body_size);
   w.insert(w.end(), body.begin(), body.end());
   return w;
}();

BACKEND_TEST_CASE("jit2 basic: spill test", "[jit2_basic]") {
   using backend_t = backend<std::nullptr_t, TestType>;
   backend_t bkend(spill_wasm, &wa);
   CHECK(bkend.call_with_return("env", "spill", (uint32_t)0)->to_ui32() == 45);
   CHECK(bkend.call_with_return("env", "spill", (uint32_t)10)->to_ui32() == 145);
   CHECK(bkend.call_with_return("env", "spill", (uint32_t)100)->to_ui32() == 1045);
}

// Spill test with control flow: if/else with spilled values live across branches
// (func (export "spill_if") (param i32) (result i32)
//   (local i32 i32 i32 i32 i32 i32 i32 i32 i32)
//   ;; Init 9 locals
//   ;; if (param > 5) { sum all } else { sum first 5 }
//   ;; After if: add local[9] (tests spilled value surviving across if/else)
// )
static std::vector<uint8_t> spill_if_wasm = []() {
   std::vector<uint8_t> w;
   auto emit = [&](std::initializer_list<uint8_t> bytes) { for (auto b : bytes) w.push_back(b); };
   emit({0x00,0x61,0x73,0x6d, 0x01,0x00,0x00,0x00});
   emit({0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f});
   emit({0x03, 0x02, 0x01, 0x00});
   emit({0x07, 0x0c, 0x01, 0x08, 0x73, 0x70, 0x69, 0x6c, 0x6c, 0x5f, 0x69, 0x66, 0x00, 0x00});
   std::vector<uint8_t> body;
   body.push_back(0x01); body.push_back(0x09); body.push_back(0x7f);
   // Init locals 1-9
   for (uint8_t i = 1; i <= 9; ++i) {
      body.push_back(0x20); body.push_back(0x00);
      body.push_back(0x41); body.push_back(i);
      body.push_back(0x6a);
      body.push_back(0x21); body.push_back(i);
   }
   // if (param > 5) → result i32
   body.push_back(0x20); body.push_back(0x00); // local.get 0
   body.push_back(0x41); body.push_back(0x05); // i32.const 5
   body.push_back(0x4a);                        // i32.gt_s
   body.push_back(0x04); body.push_back(0x7f);  // if (result i32)
   // then: sum locals 1-8
   for (uint8_t i = 1; i <= 8; ++i) {
      body.push_back(0x20); body.push_back(i);
      if (i > 1) body.push_back(0x6a);
   }
   body.push_back(0x05); // else
   // else: just local[1] + local[2]
   body.push_back(0x20); body.push_back(0x01);
   body.push_back(0x20); body.push_back(0x02);
   body.push_back(0x6a);
   body.push_back(0x0b); // end if
   // Add local[9] (spilled value must survive the if/else)
   body.push_back(0x20); body.push_back(0x09);
   body.push_back(0x6a);
   body.push_back(0x0b); // end func
   uint8_t body_size = static_cast<uint8_t>(body.size());
   uint8_t section_size = 1 + 1 + body_size;
   w.push_back(0x0a); w.push_back(section_size); w.push_back(0x01); w.push_back(body_size);
   w.insert(w.end(), body.begin(), body.end());
   return w;
}();

// Spill test with function call: values must survive across call
// Module with 2 functions:
//   func 0 (export "identity"): (param i32) (result i32) → returns param
//   func 1 (export "spill_call"): (param i32) (result i32)
//     has 10 live values, calls func 0 in the middle, uses spilled values after
static std::vector<uint8_t> spill_call_wasm = []() {
   std::vector<uint8_t> w;
   auto emit = [&](std::initializer_list<uint8_t> bytes) { for (auto b : bytes) w.push_back(b); };
   emit({0x00,0x61,0x73,0x6d, 0x01,0x00,0x00,0x00});
   // Type section: type 0 = (i32)->i32
   emit({0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f});
   // Func section: 2 functions, both type 0
   emit({0x03, 0x03, 0x02, 0x00, 0x00});
   // Export section: "identity"=func0, "spill_call"=func1
   emit({0x07, 0x19, 0x02,
         0x08, 0x69,0x64,0x65,0x6e,0x74,0x69,0x74,0x79, 0x00, 0x00,
         0x0a, 0x73,0x70,0x69,0x6c,0x6c,0x5f,0x63,0x61,0x6c,0x6c, 0x00, 0x01});
   // Code section: 2 function bodies
   // Body 0 (identity): local.get 0, end
   std::vector<uint8_t> body0 = {0x00, 0x20, 0x00, 0x0b};
   // Body 1 (spill_call): 9 body locals, init, call identity, use spilled values
   std::vector<uint8_t> body1;
   body1.push_back(0x01); body1.push_back(0x09); body1.push_back(0x7f); // 9 i32 locals
   // Init locals 1-9
   for (uint8_t i = 1; i <= 9; ++i) {
      body1.push_back(0x20); body1.push_back(0x00); // local.get 0
      body1.push_back(0x41); body1.push_back(i);    // i32.const i
      body1.push_back(0x6a);                          // i32.add
      body1.push_back(0x21); body1.push_back(i);    // local.set i
   }
   // Call identity(param) — this forces spilled values to survive the call
   body1.push_back(0x20); body1.push_back(0x00); // local.get 0
   body1.push_back(0x10); body1.push_back(0x00); // call func 0 (identity)
   body1.push_back(0x1a);                          // drop (discard result)
   // Now use all locals (some will be in spill slots that must survive the call)
   for (uint8_t i = 1; i <= 9; ++i) {
      body1.push_back(0x20); body1.push_back(i);
      if (i > 1) body1.push_back(0x6a);
   }
   // Add param (local 0)
   body1.push_back(0x20); body1.push_back(0x00);
   body1.push_back(0x6a);
   body1.push_back(0x0b); // end
   // Encode code section
   uint8_t b0_size = static_cast<uint8_t>(body0.size());
   uint8_t b1_size = static_cast<uint8_t>(body1.size());
   uint8_t section_size = 1 + 1 + b0_size + 1 + b1_size;
   w.push_back(0x0a); w.push_back(section_size);
   w.push_back(0x02); // 2 bodies
   w.push_back(b0_size); w.insert(w.end(), body0.begin(), body0.end());
   w.push_back(b1_size); w.insert(w.end(), body1.begin(), body1.end());
   return w;
}();

BACKEND_TEST_CASE("jit2 basic: spill across call", "[jit2_basic]") {
   using backend_t = backend<std::nullptr_t, TestType>;
   backend_t bkend(spill_call_wasm, &wa);
   // spill_call(10) = sum(11..19) + 10 = (11+12+13+14+15+16+17+18+19) + 10
   //                = 9*10 + 45 + 10 = 90+45+10 = 145
   CHECK(bkend.call_with_return("env", "spill_call", (uint32_t)10)->to_ui32() == 145);
   CHECK(bkend.call_with_return("env", "spill_call", (uint32_t)0)->to_ui32() == 45);
   CHECK(bkend.call_with_return("env", "spill_call", (uint32_t)100)->to_ui32() == 1045);
}

// Spill test with call_indirect + many live values
static std::vector<uint8_t> spill_indirect_wasm = []() {
   std::vector<uint8_t> w;
   auto emit = [&](std::initializer_list<uint8_t> bytes) { for (auto b : bytes) w.push_back(b); };
   emit({0x00,0x61,0x73,0x6d, 0x01,0x00,0x00,0x00});
   // Type section: type 0 = (i32)->i32
   emit({0x01, 0x06, 0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f});
   // Func section: 2 functions
   emit({0x03, 0x03, 0x02, 0x00, 0x00});
   // Table section: 1 table, min=1
   emit({0x04, 0x04, 0x01, 0x70, 0x00, 0x01});
   // Export section: "spill_ind"=func1
   emit({0x07, 0x0d, 0x01, 0x09, 0x73,0x70,0x69,0x6c,0x6c,0x5f,0x69,0x6e,0x64, 0x00, 0x01});
   // Element section: table[0] = func 0
   emit({0x09, 0x07, 0x01, 0x00, 0x41, 0x00, 0x0b, 0x01, 0x00});
   // Code section
   std::vector<uint8_t> body0 = {0x00, 0x20, 0x00, 0x0b}; // identity
   std::vector<uint8_t> body1;
   body1.push_back(0x01); body1.push_back(0x09); body1.push_back(0x7f); // 9 locals
   for (uint8_t i = 1; i <= 9; ++i) {
      body1.push_back(0x20); body1.push_back(0x00);
      body1.push_back(0x41); body1.push_back(i);
      body1.push_back(0x6a);
      body1.push_back(0x21); body1.push_back(i);
   }
   // call_indirect type=0, table_idx=0
   body1.push_back(0x20); body1.push_back(0x00); // param for call
   body1.push_back(0x41); body1.push_back(0x00); // table index 0
   body1.push_back(0x11); body1.push_back(0x00); body1.push_back(0x00); // call_indirect type=0 table=0
   body1.push_back(0x1a); // drop
   // Use all spilled locals
   for (uint8_t i = 1; i <= 9; ++i) {
      body1.push_back(0x20); body1.push_back(i);
      if (i > 1) body1.push_back(0x6a);
   }
   body1.push_back(0x20); body1.push_back(0x00);
   body1.push_back(0x6a);
   body1.push_back(0x0b);
   uint8_t b0_size = static_cast<uint8_t>(body0.size());
   uint8_t b1_size = static_cast<uint8_t>(body1.size());
   uint8_t section_size = 1 + 1 + b0_size + 1 + b1_size;
   w.push_back(0x0a); w.push_back(section_size);
   w.push_back(0x02);
   w.push_back(b0_size); w.insert(w.end(), body0.begin(), body0.end());
   w.push_back(b1_size); w.insert(w.end(), body1.begin(), body1.end());
   return w;
}();

BACKEND_TEST_CASE("jit2 basic: spill across call_indirect", "[jit2_basic]") {
   using backend_t = backend<std::nullptr_t, TestType>;
   backend_t bkend(spill_indirect_wasm, &wa);
   CHECK(bkend.call_with_return("env", "spill_ind", (uint32_t)10)->to_ui32() == 145);
   CHECK(bkend.call_with_return("env", "spill_ind", (uint32_t)0)->to_ui32() == 45);
}

BACKEND_TEST_CASE("jit2 basic: spill with if/else", "[jit2_basic]") {
   using backend_t = backend<std::nullptr_t, TestType>;
   backend_t bkend(spill_if_wasm, &wa);
   // param=10 (>5): sum(11..18) + 19 = (11+12+13+14+15+16+17+18) + 19 = 116 + 19 = 135
   // sum(11..18) = 8*10 + (1+2+3+4+5+6+7+8) = 80 + 36 = 116
   CHECK(bkend.call_with_return("env", "spill_if", (uint32_t)10)->to_ui32() == 135);
   // param=3 (<=5): (4+5) + 12 = 21
   CHECK(bkend.call_with_return("env", "spill_if", (uint32_t)3)->to_ui32() == 21);
}
