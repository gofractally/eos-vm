#pragma once

// Pass 1 of the two-pass optimizing JIT (jit2).
// Converts WASM stack machine operations to virtual-register IR.
//
// Phase 3: Delegates native codegen to machine_code_writer while building IR.
// Phase 4: Replace machine_code_writer with IR-based register-allocating codegen.
//
// The IR is built in parallel with the existing codegen. Once the IR pipeline
// is proven correct, the machine_code_writer delegation is removed and replaced
// with a new codegen that reads the IR and uses register allocation.

#include <eosio/vm/allocator.hpp>
#include <eosio/vm/exceptions.hpp>
#include <eosio/vm/jit_ir.hpp>
#include <eosio/vm/types.hpp>
#include <eosio/vm/x86_64.hpp>

#include <cstdint>

namespace eosio { namespace vm {

   template<typename Context, bool StackLimitIsBytes>
   class ir_writer {
      using mcw_t = machine_code_writer<Context, StackLimitIsBytes>;
    public:
      // Types must match machine_code_writer
      using branch_t = void*;
      using label_t  = void*;

      ir_writer(growable_allocator& alloc, std::size_t source_bytes, module& mod)
         : _allocator(alloc), _source_bytes(source_bytes), _mod(mod),
           _mcw(alloc, source_bytes, mod) {}

      ~ir_writer() = default;

      static constexpr uint32_t get_depth_for_type(uint8_t type) {
         return mcw_t::get_depth_for_type(type);
      }

      // ──── Prologue / epilogue ────

      void emit_prologue(const func_type& ft, const std::vector<local_entry>& locals, uint32_t funcnum) {
         // Phase 3: Skip IR building, just delegate to proven codegen.
         // Phase 4 will add: _func.init(_allocator, _source_bytes); etc.
         _mcw.emit_prologue(ft, locals, funcnum);
      }

      void emit_epilogue(const func_type& ft, const std::vector<local_entry>& locals, uint32_t funcnum) {
         _mcw.emit_epilogue(ft, locals, funcnum);
      }

      void finalize(function_body& body) {
         _mcw.finalize(body);
      }

      const void* get_addr() const { return _mcw.get_addr(); }
      const void* get_base_addr() const { return _mcw.get_base_addr(); }
      void set_stack_usage(std::uint64_t u) { _mcw.set_stack_usage(u); }

      // ──── Control flow (delegate to mcw, IR stubs) ────
      void emit_unreachable() { _mcw.emit_unreachable(); }
      void emit_nop() { _mcw.emit_nop(); }
      label_t emit_end() { return _mcw.emit_end(); }
      branch_t emit_return(uint32_t dc, uint8_t rt) { return _mcw.emit_return(dc, rt); }
      void emit_block() { _mcw.emit_block(); }
      label_t emit_loop() { return _mcw.emit_loop(); }
      branch_t emit_if() { return _mcw.emit_if(); }
      branch_t emit_else(branch_t if_loc) { return _mcw.emit_else(if_loc); }
      branch_t emit_br(uint32_t dc, uint8_t rt) { return _mcw.emit_br(dc, rt); }
      branch_t emit_br_if(uint32_t dc, uint8_t rt) { return _mcw.emit_br_if(dc, rt); }

      struct br_table_parser {
         typename mcw_t::br_table_generator mcw_btp;
         br_table_parser(typename mcw_t::br_table_generator btp) : mcw_btp(std::move(btp)) {}
         branch_t emit_case(uint32_t dc, uint8_t rt) { return mcw_btp.emit_case(dc, rt); }
         branch_t emit_default(uint32_t dc, uint8_t rt) { return mcw_btp.emit_default(dc, rt); }
      };
      br_table_parser emit_br_table(uint32_t table_size) {
         return br_table_parser(_mcw.emit_br_table(table_size));
      }

      // ──── Calls ────
      void emit_call(const func_type& ft, uint32_t funcnum) { _mcw.emit_call(ft, funcnum); }
      void emit_call_indirect(const func_type& ft, uint32_t fti) { _mcw.emit_call_indirect(ft, fti); }

      // ──── Parametric ────
      void emit_drop(uint8_t type) { _mcw.emit_drop(type); }
      void emit_select(uint8_t type) { _mcw.emit_select(type); }

      // ──── Local / global ────
      void emit_get_local(uint32_t li, uint8_t ty) { _mcw.emit_get_local(li, ty); }
      void emit_set_local(uint32_t li, uint8_t ty) { _mcw.emit_set_local(li, ty); }
      void emit_tee_local(uint32_t li, uint8_t ty) { _mcw.emit_tee_local(li, ty); }
      void emit_get_global(uint32_t gi) { _mcw.emit_get_global(gi); }
      void emit_set_global(uint32_t gi) { _mcw.emit_set_global(gi); }

      // ──── Memory loads ────
      void emit_i32_load(uint32_t o, uint32_t a) { _mcw.emit_i32_load(o, a); }
      void emit_i64_load(uint32_t o, uint32_t a) { _mcw.emit_i64_load(o, a); }
      void emit_f32_load(uint32_t o, uint32_t a) { _mcw.emit_f32_load(o, a); }
      void emit_f64_load(uint32_t o, uint32_t a) { _mcw.emit_f64_load(o, a); }
      void emit_i32_load8_s(uint32_t o, uint32_t a) { _mcw.emit_i32_load8_s(o, a); }
      void emit_i32_load16_s(uint32_t o, uint32_t a) { _mcw.emit_i32_load16_s(o, a); }
      void emit_i32_load8_u(uint32_t o, uint32_t a) { _mcw.emit_i32_load8_u(o, a); }
      void emit_i32_load16_u(uint32_t o, uint32_t a) { _mcw.emit_i32_load16_u(o, a); }
      void emit_i64_load8_s(uint32_t o, uint32_t a) { _mcw.emit_i64_load8_s(o, a); }
      void emit_i64_load16_s(uint32_t o, uint32_t a) { _mcw.emit_i64_load16_s(o, a); }
      void emit_i64_load32_s(uint32_t o, uint32_t a) { _mcw.emit_i64_load32_s(o, a); }
      void emit_i64_load8_u(uint32_t o, uint32_t a) { _mcw.emit_i64_load8_u(o, a); }
      void emit_i64_load16_u(uint32_t o, uint32_t a) { _mcw.emit_i64_load16_u(o, a); }
      void emit_i64_load32_u(uint32_t o, uint32_t a) { _mcw.emit_i64_load32_u(o, a); }

      // ──── Memory stores ────
      void emit_i32_store(uint32_t o, uint32_t a) { _mcw.emit_i32_store(o, a); }
      void emit_i64_store(uint32_t o, uint32_t a) { _mcw.emit_i64_store(o, a); }
      void emit_f32_store(uint32_t o, uint32_t a) { _mcw.emit_f32_store(o, a); }
      void emit_f64_store(uint32_t o, uint32_t a) { _mcw.emit_f64_store(o, a); }
      void emit_i32_store8(uint32_t o, uint32_t a) { _mcw.emit_i32_store8(o, a); }
      void emit_i32_store16(uint32_t o, uint32_t a) { _mcw.emit_i32_store16(o, a); }
      void emit_i64_store8(uint32_t o, uint32_t a) { _mcw.emit_i64_store8(o, a); }
      void emit_i64_store16(uint32_t o, uint32_t a) { _mcw.emit_i64_store16(o, a); }
      void emit_i64_store32(uint32_t o, uint32_t a) { _mcw.emit_i64_store32(o, a); }

      // ──── Memory management ────
      void emit_current_memory() { _mcw.emit_current_memory(); }
      void emit_grow_memory() { _mcw.emit_grow_memory(); }

      // ──── Constants ────
      void emit_i32_const(uint32_t v) { _mcw.emit_i32_const(v); }
      void emit_i64_const(uint64_t v) { _mcw.emit_i64_const(v); }
      void emit_f32_const(float v) { _mcw.emit_f32_const(v); }
      void emit_f64_const(double v) { _mcw.emit_f64_const(v); }

      // ──── Comparisons ────
      void emit_i32_eqz() { _mcw.emit_i32_eqz(); }
      void emit_i32_eq() { _mcw.emit_i32_eq(); }
      void emit_i32_ne() { _mcw.emit_i32_ne(); }
      void emit_i32_lt_s() { _mcw.emit_i32_lt_s(); }
      void emit_i32_lt_u() { _mcw.emit_i32_lt_u(); }
      void emit_i32_gt_s() { _mcw.emit_i32_gt_s(); }
      void emit_i32_gt_u() { _mcw.emit_i32_gt_u(); }
      void emit_i32_le_s() { _mcw.emit_i32_le_s(); }
      void emit_i32_le_u() { _mcw.emit_i32_le_u(); }
      void emit_i32_ge_s() { _mcw.emit_i32_ge_s(); }
      void emit_i32_ge_u() { _mcw.emit_i32_ge_u(); }
      void emit_i64_eqz() { _mcw.emit_i64_eqz(); }
      void emit_i64_eq() { _mcw.emit_i64_eq(); }
      void emit_i64_ne() { _mcw.emit_i64_ne(); }
      void emit_i64_lt_s() { _mcw.emit_i64_lt_s(); }
      void emit_i64_lt_u() { _mcw.emit_i64_lt_u(); }
      void emit_i64_gt_s() { _mcw.emit_i64_gt_s(); }
      void emit_i64_gt_u() { _mcw.emit_i64_gt_u(); }
      void emit_i64_le_s() { _mcw.emit_i64_le_s(); }
      void emit_i64_le_u() { _mcw.emit_i64_le_u(); }
      void emit_i64_ge_s() { _mcw.emit_i64_ge_s(); }
      void emit_i64_ge_u() { _mcw.emit_i64_ge_u(); }
      void emit_f32_eq() { _mcw.emit_f32_eq(); }
      void emit_f32_ne() { _mcw.emit_f32_ne(); }
      void emit_f32_lt() { _mcw.emit_f32_lt(); }
      void emit_f32_gt() { _mcw.emit_f32_gt(); }
      void emit_f32_le() { _mcw.emit_f32_le(); }
      void emit_f32_ge() { _mcw.emit_f32_ge(); }
      void emit_f64_eq() { _mcw.emit_f64_eq(); }
      void emit_f64_ne() { _mcw.emit_f64_ne(); }
      void emit_f64_lt() { _mcw.emit_f64_lt(); }
      void emit_f64_gt() { _mcw.emit_f64_gt(); }
      void emit_f64_le() { _mcw.emit_f64_le(); }
      void emit_f64_ge() { _mcw.emit_f64_ge(); }

      // ──── Integer arithmetic ────
      void emit_i32_clz() { _mcw.emit_i32_clz(); }
      void emit_i32_ctz() { _mcw.emit_i32_ctz(); }
      void emit_i32_popcnt() { _mcw.emit_i32_popcnt(); }
      void emit_i32_add() { _mcw.emit_i32_add(); }
      void emit_i32_sub() { _mcw.emit_i32_sub(); }
      void emit_i32_mul() { _mcw.emit_i32_mul(); }
      void emit_i32_div_s() { _mcw.emit_i32_div_s(); }
      void emit_i32_div_u() { _mcw.emit_i32_div_u(); }
      void emit_i32_rem_s() { _mcw.emit_i32_rem_s(); }
      void emit_i32_rem_u() { _mcw.emit_i32_rem_u(); }
      void emit_i32_and() { _mcw.emit_i32_and(); }
      void emit_i32_or() { _mcw.emit_i32_or(); }
      void emit_i32_xor() { _mcw.emit_i32_xor(); }
      void emit_i32_shl() { _mcw.emit_i32_shl(); }
      void emit_i32_shr_s() { _mcw.emit_i32_shr_s(); }
      void emit_i32_shr_u() { _mcw.emit_i32_shr_u(); }
      void emit_i32_rotl() { _mcw.emit_i32_rotl(); }
      void emit_i32_rotr() { _mcw.emit_i32_rotr(); }
      void emit_i64_clz() { _mcw.emit_i64_clz(); }
      void emit_i64_ctz() { _mcw.emit_i64_ctz(); }
      void emit_i64_popcnt() { _mcw.emit_i64_popcnt(); }
      void emit_i64_add() { _mcw.emit_i64_add(); }
      void emit_i64_sub() { _mcw.emit_i64_sub(); }
      void emit_i64_mul() { _mcw.emit_i64_mul(); }
      void emit_i64_div_s() { _mcw.emit_i64_div_s(); }
      void emit_i64_div_u() { _mcw.emit_i64_div_u(); }
      void emit_i64_rem_s() { _mcw.emit_i64_rem_s(); }
      void emit_i64_rem_u() { _mcw.emit_i64_rem_u(); }
      void emit_i64_and() { _mcw.emit_i64_and(); }
      void emit_i64_or() { _mcw.emit_i64_or(); }
      void emit_i64_xor() { _mcw.emit_i64_xor(); }
      void emit_i64_shl() { _mcw.emit_i64_shl(); }
      void emit_i64_shr_s() { _mcw.emit_i64_shr_s(); }
      void emit_i64_shr_u() { _mcw.emit_i64_shr_u(); }
      void emit_i64_rotl() { _mcw.emit_i64_rotl(); }
      void emit_i64_rotr() { _mcw.emit_i64_rotr(); }

      // ──── Float arithmetic ────
      void emit_f32_abs() { _mcw.emit_f32_abs(); }
      void emit_f32_neg() { _mcw.emit_f32_neg(); }
      void emit_f32_ceil() { _mcw.emit_f32_ceil(); }
      void emit_f32_floor() { _mcw.emit_f32_floor(); }
      void emit_f32_trunc() { _mcw.emit_f32_trunc(); }
      void emit_f32_nearest() { _mcw.emit_f32_nearest(); }
      void emit_f32_sqrt() { _mcw.emit_f32_sqrt(); }
      void emit_f32_add() { _mcw.emit_f32_add(); }
      void emit_f32_sub() { _mcw.emit_f32_sub(); }
      void emit_f32_mul() { _mcw.emit_f32_mul(); }
      void emit_f32_div() { _mcw.emit_f32_div(); }
      void emit_f32_min() { _mcw.emit_f32_min(); }
      void emit_f32_max() { _mcw.emit_f32_max(); }
      void emit_f32_copysign() { _mcw.emit_f32_copysign(); }
      void emit_f64_abs() { _mcw.emit_f64_abs(); }
      void emit_f64_neg() { _mcw.emit_f64_neg(); }
      void emit_f64_ceil() { _mcw.emit_f64_ceil(); }
      void emit_f64_floor() { _mcw.emit_f64_floor(); }
      void emit_f64_trunc() { _mcw.emit_f64_trunc(); }
      void emit_f64_nearest() { _mcw.emit_f64_nearest(); }
      void emit_f64_sqrt() { _mcw.emit_f64_sqrt(); }
      void emit_f64_add() { _mcw.emit_f64_add(); }
      void emit_f64_sub() { _mcw.emit_f64_sub(); }
      void emit_f64_mul() { _mcw.emit_f64_mul(); }
      void emit_f64_div() { _mcw.emit_f64_div(); }
      void emit_f64_min() { _mcw.emit_f64_min(); }
      void emit_f64_max() { _mcw.emit_f64_max(); }
      void emit_f64_copysign() { _mcw.emit_f64_copysign(); }

      // ──── Conversions ────
      void emit_i32_wrap_i64() { _mcw.emit_i32_wrap_i64(); }
      void emit_i32_trunc_s_f32() { _mcw.emit_i32_trunc_s_f32(); }
      void emit_i32_trunc_u_f32() { _mcw.emit_i32_trunc_u_f32(); }
      void emit_i32_trunc_s_f64() { _mcw.emit_i32_trunc_s_f64(); }
      void emit_i32_trunc_u_f64() { _mcw.emit_i32_trunc_u_f64(); }
      void emit_i64_extend_s_i32() { _mcw.emit_i64_extend_s_i32(); }
      void emit_i64_extend_u_i32() { _mcw.emit_i64_extend_u_i32(); }
      void emit_i64_trunc_s_f32() { _mcw.emit_i64_trunc_s_f32(); }
      void emit_i64_trunc_u_f32() { _mcw.emit_i64_trunc_u_f32(); }
      void emit_i64_trunc_s_f64() { _mcw.emit_i64_trunc_s_f64(); }
      void emit_i64_trunc_u_f64() { _mcw.emit_i64_trunc_u_f64(); }
      void emit_f32_convert_s_i32() { _mcw.emit_f32_convert_s_i32(); }
      void emit_f32_convert_u_i32() { _mcw.emit_f32_convert_u_i32(); }
      void emit_f32_convert_s_i64() { _mcw.emit_f32_convert_s_i64(); }
      void emit_f32_convert_u_i64() { _mcw.emit_f32_convert_u_i64(); }
      void emit_f32_demote_f64() { _mcw.emit_f32_demote_f64(); }
      void emit_f64_convert_s_i32() { _mcw.emit_f64_convert_s_i32(); }
      void emit_f64_convert_u_i32() { _mcw.emit_f64_convert_u_i32(); }
      void emit_f64_convert_s_i64() { _mcw.emit_f64_convert_s_i64(); }
      void emit_f64_convert_u_i64() { _mcw.emit_f64_convert_u_i64(); }
      void emit_f64_promote_f32() { _mcw.emit_f64_promote_f32(); }
      void emit_i32_reinterpret_f32() { _mcw.emit_i32_reinterpret_f32(); }
      void emit_i64_reinterpret_f64() { _mcw.emit_i64_reinterpret_f64(); }
      void emit_f32_reinterpret_i32() { _mcw.emit_f32_reinterpret_i32(); }
      void emit_f64_reinterpret_i64() { _mcw.emit_f64_reinterpret_i64(); }
      void emit_i32_trunc_sat_f32_s() { _mcw.emit_i32_trunc_sat_f32_s(); }
      void emit_i32_trunc_sat_f32_u() { _mcw.emit_i32_trunc_sat_f32_u(); }
      void emit_i32_trunc_sat_f64_s() { _mcw.emit_i32_trunc_sat_f64_s(); }
      void emit_i32_trunc_sat_f64_u() { _mcw.emit_i32_trunc_sat_f64_u(); }
      void emit_i64_trunc_sat_f32_s() { _mcw.emit_i64_trunc_sat_f32_s(); }
      void emit_i64_trunc_sat_f32_u() { _mcw.emit_i64_trunc_sat_f32_u(); }
      void emit_i64_trunc_sat_f64_s() { _mcw.emit_i64_trunc_sat_f64_s(); }
      void emit_i64_trunc_sat_f64_u() { _mcw.emit_i64_trunc_sat_f64_u(); }
      void emit_i32_extend8_s() { _mcw.emit_i32_extend8_s(); }
      void emit_i32_extend16_s() { _mcw.emit_i32_extend16_s(); }
      void emit_i64_extend8_s() { _mcw.emit_i64_extend8_s(); }
      void emit_i64_extend16_s() { _mcw.emit_i64_extend16_s(); }
      void emit_i64_extend32_s() { _mcw.emit_i64_extend32_s(); }

      // ──── SIMD ────
      void emit_v128_load(uint32_t o, uint32_t a) { _mcw.emit_v128_load(o, a); }
      void emit_v128_load8x8_s(uint32_t o, uint32_t a) { _mcw.emit_v128_load8x8_s(o, a); }
      void emit_v128_load8x8_u(uint32_t o, uint32_t a) { _mcw.emit_v128_load8x8_u(o, a); }
      void emit_v128_load16x4_s(uint32_t o, uint32_t a) { _mcw.emit_v128_load16x4_s(o, a); }
      void emit_v128_load16x4_u(uint32_t o, uint32_t a) { _mcw.emit_v128_load16x4_u(o, a); }
      void emit_v128_load32x2_s(uint32_t o, uint32_t a) { _mcw.emit_v128_load32x2_s(o, a); }
      void emit_v128_load32x2_u(uint32_t o, uint32_t a) { _mcw.emit_v128_load32x2_u(o, a); }
      void emit_v128_load8_splat(uint32_t o, uint32_t a) { _mcw.emit_v128_load8_splat(o, a); }
      void emit_v128_load16_splat(uint32_t o, uint32_t a) { _mcw.emit_v128_load16_splat(o, a); }
      void emit_v128_load32_splat(uint32_t o, uint32_t a) { _mcw.emit_v128_load32_splat(o, a); }
      void emit_v128_load64_splat(uint32_t o, uint32_t a) { _mcw.emit_v128_load64_splat(o, a); }
      void emit_v128_load32_zero(uint32_t o, uint32_t a) { _mcw.emit_v128_load32_zero(o, a); }
      void emit_v128_load64_zero(uint32_t o, uint32_t a) { _mcw.emit_v128_load64_zero(o, a); }
      void emit_v128_store(uint32_t o, uint32_t a) { _mcw.emit_v128_store(o, a); }
      void emit_v128_load8_lane(uint32_t o, uint32_t a, uint8_t l) { _mcw.emit_v128_load8_lane(o, a, l); }
      void emit_v128_load16_lane(uint32_t o, uint32_t a, uint8_t l) { _mcw.emit_v128_load16_lane(o, a, l); }
      void emit_v128_load32_lane(uint32_t o, uint32_t a, uint8_t l) { _mcw.emit_v128_load32_lane(o, a, l); }
      void emit_v128_load64_lane(uint32_t o, uint32_t a, uint8_t l) { _mcw.emit_v128_load64_lane(o, a, l); }
      void emit_v128_store8_lane(uint32_t o, uint32_t a, uint8_t l) { _mcw.emit_v128_store8_lane(o, a, l); }
      void emit_v128_store16_lane(uint32_t o, uint32_t a, uint8_t l) { _mcw.emit_v128_store16_lane(o, a, l); }
      void emit_v128_store32_lane(uint32_t o, uint32_t a, uint8_t l) { _mcw.emit_v128_store32_lane(o, a, l); }
      void emit_v128_store64_lane(uint32_t o, uint32_t a, uint8_t l) { _mcw.emit_v128_store64_lane(o, a, l); }
      void emit_v128_const(v128_t v) { _mcw.emit_v128_const(v); }
      void emit_i8x16_shuffle(const uint8_t* l) { _mcw.emit_i8x16_shuffle(l); }

#define FWD0(name) void name() { _mcw.name(); }
#define FWD1(name) void name(uint8_t a) { _mcw.name(a); }

      FWD1(emit_i8x16_extract_lane_s) FWD1(emit_i8x16_extract_lane_u) FWD1(emit_i8x16_replace_lane)
      FWD1(emit_i16x8_extract_lane_s) FWD1(emit_i16x8_extract_lane_u) FWD1(emit_i16x8_replace_lane)
      FWD1(emit_i32x4_extract_lane) FWD1(emit_i32x4_replace_lane)
      FWD1(emit_i64x2_extract_lane) FWD1(emit_i64x2_replace_lane)
      FWD1(emit_f32x4_extract_lane) FWD1(emit_f32x4_replace_lane)
      FWD1(emit_f64x2_extract_lane) FWD1(emit_f64x2_replace_lane)

      FWD0(emit_i8x16_swizzle) FWD0(emit_i8x16_splat) FWD0(emit_i16x8_splat)
      FWD0(emit_i32x4_splat) FWD0(emit_i64x2_splat) FWD0(emit_f32x4_splat) FWD0(emit_f64x2_splat)

      FWD0(emit_i8x16_eq) FWD0(emit_i8x16_ne) FWD0(emit_i8x16_lt_s) FWD0(emit_i8x16_lt_u)
      FWD0(emit_i8x16_gt_s) FWD0(emit_i8x16_gt_u) FWD0(emit_i8x16_le_s) FWD0(emit_i8x16_le_u)
      FWD0(emit_i8x16_ge_s) FWD0(emit_i8x16_ge_u)
      FWD0(emit_i16x8_eq) FWD0(emit_i16x8_ne) FWD0(emit_i16x8_lt_s) FWD0(emit_i16x8_lt_u)
      FWD0(emit_i16x8_gt_s) FWD0(emit_i16x8_gt_u) FWD0(emit_i16x8_le_s) FWD0(emit_i16x8_le_u)
      FWD0(emit_i16x8_ge_s) FWD0(emit_i16x8_ge_u)
      FWD0(emit_i32x4_eq) FWD0(emit_i32x4_ne) FWD0(emit_i32x4_lt_s) FWD0(emit_i32x4_lt_u)
      FWD0(emit_i32x4_gt_s) FWD0(emit_i32x4_gt_u) FWD0(emit_i32x4_le_s) FWD0(emit_i32x4_le_u)
      FWD0(emit_i32x4_ge_s) FWD0(emit_i32x4_ge_u)
      FWD0(emit_i64x2_eq) FWD0(emit_i64x2_ne) FWD0(emit_i64x2_lt_s) FWD0(emit_i64x2_gt_s)
      FWD0(emit_i64x2_le_s) FWD0(emit_i64x2_ge_s)
      FWD0(emit_f32x4_eq) FWD0(emit_f32x4_ne) FWD0(emit_f32x4_lt) FWD0(emit_f32x4_gt)
      FWD0(emit_f32x4_le) FWD0(emit_f32x4_ge)
      FWD0(emit_f64x2_eq) FWD0(emit_f64x2_ne) FWD0(emit_f64x2_lt) FWD0(emit_f64x2_gt)
      FWD0(emit_f64x2_le) FWD0(emit_f64x2_ge)
      FWD0(emit_v128_not) FWD0(emit_v128_and) FWD0(emit_v128_andnot) FWD0(emit_v128_or)
      FWD0(emit_v128_xor) FWD0(emit_v128_bitselect) FWD0(emit_v128_any_true)
      FWD0(emit_i8x16_abs) FWD0(emit_i8x16_neg) FWD0(emit_i8x16_popcnt)
      FWD0(emit_i8x16_all_true) FWD0(emit_i8x16_bitmask)
      FWD0(emit_i8x16_narrow_i16x8_s) FWD0(emit_i8x16_narrow_i16x8_u)
      FWD0(emit_i8x16_shl) FWD0(emit_i8x16_shr_s) FWD0(emit_i8x16_shr_u)
      FWD0(emit_i8x16_add) FWD0(emit_i8x16_add_sat_s) FWD0(emit_i8x16_add_sat_u)
      FWD0(emit_i8x16_sub) FWD0(emit_i8x16_sub_sat_s) FWD0(emit_i8x16_sub_sat_u)
      FWD0(emit_i8x16_min_s) FWD0(emit_i8x16_min_u) FWD0(emit_i8x16_max_s)
      FWD0(emit_i8x16_max_u) FWD0(emit_i8x16_avgr_u)
      FWD0(emit_i16x8_extadd_pairwise_i8x16_s) FWD0(emit_i16x8_extadd_pairwise_i8x16_u)
      FWD0(emit_i16x8_abs) FWD0(emit_i16x8_neg) FWD0(emit_i16x8_q15mulr_sat_s)
      FWD0(emit_i16x8_all_true) FWD0(emit_i16x8_bitmask)
      FWD0(emit_i16x8_narrow_i32x4_s) FWD0(emit_i16x8_narrow_i32x4_u)
      FWD0(emit_i16x8_extend_low_i8x16_s) FWD0(emit_i16x8_extend_high_i8x16_s)
      FWD0(emit_i16x8_extend_low_i8x16_u) FWD0(emit_i16x8_extend_high_i8x16_u)
      FWD0(emit_i16x8_shl) FWD0(emit_i16x8_shr_s) FWD0(emit_i16x8_shr_u)
      FWD0(emit_i16x8_add) FWD0(emit_i16x8_add_sat_s) FWD0(emit_i16x8_add_sat_u)
      FWD0(emit_i16x8_sub) FWD0(emit_i16x8_sub_sat_s) FWD0(emit_i16x8_sub_sat_u)
      FWD0(emit_i16x8_mul) FWD0(emit_i16x8_min_s) FWD0(emit_i16x8_min_u)
      FWD0(emit_i16x8_max_s) FWD0(emit_i16x8_max_u) FWD0(emit_i16x8_avgr_u)
      FWD0(emit_i16x8_extmul_low_i8x16_s) FWD0(emit_i16x8_extmul_high_i8x16_s)
      FWD0(emit_i16x8_extmul_low_i8x16_u) FWD0(emit_i16x8_extmul_high_i8x16_u)
      FWD0(emit_i32x4_extadd_pairwise_i16x8_s) FWD0(emit_i32x4_extadd_pairwise_i16x8_u)
      FWD0(emit_i32x4_abs) FWD0(emit_i32x4_neg) FWD0(emit_i32x4_all_true) FWD0(emit_i32x4_bitmask)
      FWD0(emit_i32x4_extend_low_i16x8_s) FWD0(emit_i32x4_extend_high_i16x8_s)
      FWD0(emit_i32x4_extend_low_i16x8_u) FWD0(emit_i32x4_extend_high_i16x8_u)
      FWD0(emit_i32x4_shl) FWD0(emit_i32x4_shr_s) FWD0(emit_i32x4_shr_u)
      FWD0(emit_i32x4_add) FWD0(emit_i32x4_sub) FWD0(emit_i32x4_mul)
      FWD0(emit_i32x4_min_s) FWD0(emit_i32x4_min_u) FWD0(emit_i32x4_max_s) FWD0(emit_i32x4_max_u)
      FWD0(emit_i32x4_dot_i16x8_s)
      FWD0(emit_i32x4_extmul_low_i16x8_s) FWD0(emit_i32x4_extmul_high_i16x8_s)
      FWD0(emit_i32x4_extmul_low_i16x8_u) FWD0(emit_i32x4_extmul_high_i16x8_u)
      FWD0(emit_i64x2_abs) FWD0(emit_i64x2_neg) FWD0(emit_i64x2_all_true) FWD0(emit_i64x2_bitmask)
      FWD0(emit_i64x2_extend_low_i32x4_s) FWD0(emit_i64x2_extend_high_i32x4_s)
      FWD0(emit_i64x2_extend_low_i32x4_u) FWD0(emit_i64x2_extend_high_i32x4_u)
      FWD0(emit_i64x2_shl) FWD0(emit_i64x2_shr_s) FWD0(emit_i64x2_shr_u)
      FWD0(emit_i64x2_add) FWD0(emit_i64x2_sub) FWD0(emit_i64x2_mul)
      FWD0(emit_i64x2_extmul_low_i32x4_s) FWD0(emit_i64x2_extmul_high_i32x4_s)
      FWD0(emit_i64x2_extmul_low_i32x4_u) FWD0(emit_i64x2_extmul_high_i32x4_u)
      FWD0(emit_f32x4_ceil) FWD0(emit_f32x4_floor) FWD0(emit_f32x4_trunc) FWD0(emit_f32x4_nearest)
      FWD0(emit_f32x4_abs) FWD0(emit_f32x4_neg) FWD0(emit_f32x4_sqrt)
      FWD0(emit_f32x4_add) FWD0(emit_f32x4_sub) FWD0(emit_f32x4_mul) FWD0(emit_f32x4_div)
      FWD0(emit_f32x4_min) FWD0(emit_f32x4_max) FWD0(emit_f32x4_pmin) FWD0(emit_f32x4_pmax)
      FWD0(emit_f64x2_ceil) FWD0(emit_f64x2_floor) FWD0(emit_f64x2_trunc) FWD0(emit_f64x2_nearest)
      FWD0(emit_f64x2_abs) FWD0(emit_f64x2_neg) FWD0(emit_f64x2_sqrt)
      FWD0(emit_f64x2_add) FWD0(emit_f64x2_sub) FWD0(emit_f64x2_mul) FWD0(emit_f64x2_div)
      FWD0(emit_f64x2_min) FWD0(emit_f64x2_max) FWD0(emit_f64x2_pmin) FWD0(emit_f64x2_pmax)
      FWD0(emit_i32x4_trunc_sat_f32x4_s) FWD0(emit_i32x4_trunc_sat_f32x4_u)
      FWD0(emit_f32x4_convert_i32x4_s) FWD0(emit_f32x4_convert_i32x4_u)
      FWD0(emit_i32x4_trunc_sat_f64x2_s_zero) FWD0(emit_i32x4_trunc_sat_f64x2_u_zero)
      FWD0(emit_f64x2_convert_low_i32x4_s) FWD0(emit_f64x2_convert_low_i32x4_u)
      FWD0(emit_f32x4_demote_f64x2_zero) FWD0(emit_f64x2_promote_low_f32x4)
#undef FWD0
#undef FWD1

      // ──── Bulk memory ────
      void emit_memory_init(std::uint32_t s) { _mcw.emit_memory_init(s); }
      void emit_data_drop(std::uint32_t s) { _mcw.emit_data_drop(s); }
      void emit_memory_copy() { _mcw.emit_memory_copy(); }
      void emit_memory_fill() { _mcw.emit_memory_fill(); }
      void emit_table_init(std::uint32_t s) { _mcw.emit_table_init(s); }
      void emit_elem_drop(std::uint32_t s) { _mcw.emit_elem_drop(s); }
      void emit_table_copy() { _mcw.emit_table_copy(); }

      // ──── Branch fixup ────
      void fix_branch(branch_t br, label_t lbl) { _mcw.fix_branch(br, lbl); }

    private:
      growable_allocator& _allocator;
      std::size_t _source_bytes;
      module& _mod;
      mcw_t _mcw;           // Delegate codegen (Phase 3)
      ir_function _func;    // IR being built (for Phase 4)
      uint32_t _current_block = 0;
   };

}} // namespace eosio::vm
