#pragma once

// Linear scan register allocator for jit2.
//
// Computes live intervals for virtual registers, then assigns physical
// registers using a linear scan. Vregs that can't get a register are
// assigned spill slots (rbp-relative stack slots).
//
// All data structures use growable_allocator — no malloc.

#include <eosio/vm/jit_ir.hpp>
#include <eosio/vm/allocator.hpp>

#include <cstdint>
#include <algorithm>

namespace eosio { namespace vm {

   // Physical register assignment
#ifdef __aarch64__
   enum class phys_reg : int8_t {
      none = -1,
      // X0, X1 reserved (temps for spill loads, like rax/rcx on x86)
      // X16, X17 reserved (scratch for large immediates / linker veneers)
      // X18 reserved (platform register)
      // X19 = context pointer, X20 = linear memory base, X21 = call depth (all callee-saved)
      // X29 = FP, X30 = LR, SP = stack pointer
      // Caller-saved (free, no save/restore):
      x2 = 0, x3 = 1, x4 = 2, x5 = 3, x6 = 4, x7 = 5,
      x8 = 6, x9 = 7, x10 = 8, x11 = 9, x12 = 10, x13 = 11, x14 = 12, x15 = 13,
      caller_saved_count = 14,
      // Callee-saved (must save/restore in prologue/epilogue):
      x22 = 14, x23 = 15, x24 = 16, x25 = 17, x26 = 18, x27 = 19, x28 = 20,
      count = 21,
   };
#else
   enum class phys_reg : int8_t {
      none = -1,
      // rax, rcx, rdx reserved (temps + implicit x86 usage in div/mul)
      // rsi = linear memory base, rdi = context pointer (both reserved)
      // Caller-saved (free, no save/restore):
      r8 = 0, r9 = 1, r10 = 2, r11 = 3,
      caller_saved_count = 4,
      // Callee-saved (must save/restore in prologue/epilogue):
      rbx = 4, r12 = 5, r13 = 6, r14 = 7, r15 = 8,
      count = 9,
   };
#endif

   class jit_regalloc {
    public:
      // Compute live intervals for all vregs in a function.
      // intervals array must have space for func.next_vreg entries.
      static void compute_live_intervals(ir_function& func, jit_scratch_allocator& alloc) {
         uint32_t num_vregs = func.next_vreg;
         if (num_vregs == 0) return;

         // Allocate intervals from scratch allocator
         func.intervals = alloc.alloc<ir_live_interval>(num_vregs);
         func.interval_count = num_vregs;

         // Initialize: start = MAX, end = 0
         for (uint32_t i = 0; i < num_vregs; ++i) {
            func.intervals[i].vreg = i;
            func.intervals[i].start = UINT32_MAX;
            func.intervals[i].end = 0;
            func.intervals[i].phys_reg = -1;
            func.intervals[i].phys_xmm = -1;
            func.intervals[i].spill_slot = -1;
            func.intervals[i].type = 0;
         }

         // Extend the return value vreg's interval to the end of the function.
         // The vstack's last entry holds the return value, which is read by
         // the epilogue but not by any IR instruction.
         // For v128 returns, the low vreg (vstack_top-2) is the actual value.
         if (func.vstack_top > 0) {
            bool is_v128_ret = func.type && func.type->return_count > 0
                               && func.type->return_type == types::v128;
            uint32_t ret_idx = is_v128_ret && func.vstack_top >= 2
                               ? func.vstack_top - 2 : func.vstack_top - 1;
            uint32_t ret_vreg = func.vstack[ret_idx];
            if (ret_vreg < num_vregs) {
               func.intervals[ret_vreg].end = func.inst_count;
            }
         }

         // Scan instructions to find first def and last use of each vreg
         for (uint32_t i = 0; i < func.inst_count; ++i) {
            const auto& inst = func.insts[i];

            // NOTE: Dead instructions are NOT skipped here. The register-allocated
            // codegen path still emits dead instructions to populate registers
            // (preventing stale data). Their vregs need valid intervals.

            // Destination vreg: defined at this instruction
            bool is_store = (inst.opcode >= ir_op::i32_store && inst.opcode <= ir_op::i64_store32);
            bool is_block_marker = (inst.opcode == ir_op::block_start || inst.opcode == ir_op::block_end);
            bool is_v128_op = (inst.opcode == ir_op::v128_op);
            if (!is_store && !is_block_marker && !is_v128_op && inst.dest != ir_vreg_none && inst.dest < num_vregs) {
               auto& iv = func.intervals[inst.dest];
               if (i < iv.start) iv.start = i;
               if (i > iv.end) iv.end = i;
               // Don't downgrade v128 type once set (prevents conflict between
               // const_v128 type and nop marker type for the same vreg)
               if (iv.type != types::v128)
                  iv.type = inst.type;
            }
            // Scalar-producing v128_ops store their result vreg in simd.addr
            if (is_v128_op && simd_produces_scalar(static_cast<simd_sub>(inst.dest))) {
               uint32_t result_vreg = inst.simd.addr;
               if (result_vreg != ir_vreg_none && result_vreg < num_vregs) {
                  auto& iv = func.intervals[result_vreg];
                  if (i < iv.start) iv.start = i;
                  if (i > iv.end) iv.end = i;
                  iv.type = types::i32; // scalar result
               }
            }
            // v128 operand/result vregs for XMM register allocation
            // Skip i8x16_shuffle: its immv128 union member overlaps v_src/v_dest fields
            if (is_v128_op && static_cast<simd_sub>(inst.dest) != simd_sub::i8x16_shuffle) {
               auto def_v128 = [&](uint16_t vreg) {
                  if (vreg != 0xFFFF && vreg < num_vregs) {
                     auto& iv = func.intervals[vreg];
                     if (i < iv.start) iv.start = i;
                     if (i > iv.end) iv.end = i;
                     iv.type = types::v128;
                  }
               };
               auto use_v128 = [&](uint16_t vreg) {
                  if (vreg != 0xFFFF && vreg < num_vregs) {
                     auto& iv = func.intervals[vreg];
                     if (i < iv.start) iv.start = i;
                     if (i > iv.end) iv.end = i;
                  }
               };
               use_v128(inst.simd.v_src1);
               use_v128(inst.simd.v_src2);
               def_v128(inst.simd.v_dest);
               // Bitselect's 3rd v128 source (mask) is stored in addr field
               if (static_cast<simd_sub>(inst.dest) == simd_sub::v128_bitselect) {
                  use_v128(static_cast<uint16_t>(inst.simd.addr));
               }
            }

            // Source vregs: must check per-opcode which union fields are vregs.
            // rr.src1/src2 are ONLY vregs for arithmetic/comparison/select ops.
            // For br/call/local/global ops, the union holds indices, not vregs.
            auto use_vreg = [&](uint32_t vreg) {
               if (vreg != ir_vreg_none && vreg < num_vregs) {
                  auto& iv = func.intervals[vreg];
                  if (i < iv.start) iv.start = i;
                  if (i > iv.end) iv.end = i;
               }
            };

            switch (inst.opcode) {
            // Binary ops: src1 and src2 are both vregs
            case ir_op::i32_add: case ir_op::i32_sub: case ir_op::i32_mul:
            case ir_op::i32_div_s: case ir_op::i32_div_u: case ir_op::i32_rem_s: case ir_op::i32_rem_u:
            case ir_op::i32_and: case ir_op::i32_or: case ir_op::i32_xor:
            case ir_op::i32_shl: case ir_op::i32_shr_s: case ir_op::i32_shr_u:
            case ir_op::i32_rotl: case ir_op::i32_rotr:
            case ir_op::i64_add: case ir_op::i64_sub: case ir_op::i64_mul:
            case ir_op::i64_div_s: case ir_op::i64_div_u: case ir_op::i64_rem_s: case ir_op::i64_rem_u:
            case ir_op::i64_and: case ir_op::i64_or: case ir_op::i64_xor:
            case ir_op::i64_shl: case ir_op::i64_shr_s: case ir_op::i64_shr_u:
            case ir_op::i64_rotl: case ir_op::i64_rotr:
            case ir_op::i32_eq: case ir_op::i32_ne: case ir_op::i32_lt_s: case ir_op::i32_lt_u:
            case ir_op::i32_gt_s: case ir_op::i32_gt_u: case ir_op::i32_le_s: case ir_op::i32_le_u:
            case ir_op::i32_ge_s: case ir_op::i32_ge_u:
            case ir_op::i64_eq: case ir_op::i64_ne: case ir_op::i64_lt_s: case ir_op::i64_lt_u:
            case ir_op::i64_gt_s: case ir_op::i64_gt_u: case ir_op::i64_le_s: case ir_op::i64_le_u:
            case ir_op::i64_ge_s: case ir_op::i64_ge_u:
            case ir_op::f32_add: case ir_op::f32_sub: case ir_op::f32_mul: case ir_op::f32_div:
            case ir_op::f32_min: case ir_op::f32_max: case ir_op::f32_copysign:
            case ir_op::f64_add: case ir_op::f64_sub: case ir_op::f64_mul: case ir_op::f64_div:
            case ir_op::f64_min: case ir_op::f64_max: case ir_op::f64_copysign:
            case ir_op::f32_eq: case ir_op::f32_ne: case ir_op::f32_lt: case ir_op::f32_gt:
            case ir_op::f32_le: case ir_op::f32_ge:
            case ir_op::f64_eq: case ir_op::f64_ne: case ir_op::f64_lt: case ir_op::f64_gt:
            case ir_op::f64_le: case ir_op::f64_ge:
               use_vreg(inst.rr.src1);
               use_vreg(inst.rr.src2);
               break;
            case ir_op::select:
               use_vreg(inst.sel.val1);
               use_vreg(inst.sel.val2);
               use_vreg(inst.sel.cond);
               break;

            // Unary ops: only src1 is a vreg
            case ir_op::i32_eqz: case ir_op::i64_eqz:
            case ir_op::i32_clz: case ir_op::i32_ctz: case ir_op::i32_popcnt:
            case ir_op::i64_clz: case ir_op::i64_ctz: case ir_op::i64_popcnt:
            case ir_op::i32_wrap_i64: case ir_op::i64_extend_s_i32: case ir_op::i64_extend_u_i32:
            case ir_op::i32_extend8_s: case ir_op::i32_extend16_s:
            case ir_op::i64_extend8_s: case ir_op::i64_extend16_s: case ir_op::i64_extend32_s:
            case ir_op::f32_abs: case ir_op::f32_neg: case ir_op::f32_sqrt:
            case ir_op::f32_ceil: case ir_op::f32_floor: case ir_op::f32_trunc: case ir_op::f32_nearest:
            case ir_op::f64_abs: case ir_op::f64_neg: case ir_op::f64_sqrt:
            case ir_op::f64_ceil: case ir_op::f64_floor: case ir_op::f64_trunc: case ir_op::f64_nearest:
            case ir_op::i32_reinterpret_f32: case ir_op::i64_reinterpret_f64:
            case ir_op::f32_reinterpret_i32: case ir_op::f64_reinterpret_i64:
            case ir_op::i32_trunc_s_f32: case ir_op::i32_trunc_u_f32:
            case ir_op::i32_trunc_s_f64: case ir_op::i32_trunc_u_f64:
            case ir_op::i64_trunc_s_f32: case ir_op::i64_trunc_u_f32:
            case ir_op::i64_trunc_s_f64: case ir_op::i64_trunc_u_f64:
            case ir_op::f32_convert_s_i32: case ir_op::f32_convert_u_i32:
            case ir_op::f32_convert_s_i64: case ir_op::f32_convert_u_i64:
            case ir_op::f64_convert_s_i32: case ir_op::f64_convert_u_i32:
            case ir_op::f64_convert_s_i64: case ir_op::f64_convert_u_i64:
            case ir_op::f32_demote_f64: case ir_op::f64_promote_f32:
            case ir_op::i32_trunc_sat_f32_s: case ir_op::i32_trunc_sat_f32_u:
            case ir_op::i32_trunc_sat_f64_s: case ir_op::i32_trunc_sat_f64_u:
            case ir_op::i64_trunc_sat_f32_s: case ir_op::i64_trunc_sat_f32_u:
            case ir_op::i64_trunc_sat_f64_s: case ir_op::i64_trunc_sat_f64_u:
               use_vreg(inst.rr.src1);
               break;

            // Loads: ri.src1 = addr vreg
            case ir_op::i32_load: case ir_op::i64_load: case ir_op::f32_load: case ir_op::f64_load:
            case ir_op::i32_load8_s: case ir_op::i32_load8_u:
            case ir_op::i32_load16_s: case ir_op::i32_load16_u:
            case ir_op::i64_load8_s: case ir_op::i64_load8_u:
            case ir_op::i64_load16_s: case ir_op::i64_load16_u:
            case ir_op::i64_load32_s: case ir_op::i64_load32_u:
               use_vreg(inst.ri.src1); // addr vreg
               break;

            // Stores: ri.src1 = addr vreg, dest = value vreg (both are uses)
            case ir_op::i32_store: case ir_op::i64_store: case ir_op::f32_store: case ir_op::f64_store:
            case ir_op::i32_store8: case ir_op::i32_store16:
            case ir_op::i64_store8: case ir_op::i64_store16: case ir_op::i64_store32:
               use_vreg(inst.ri.src1); // addr
               use_vreg(inst.dest);    // value (stored in dest field for stores)
               break;

            // Local set/tee: local.src1 is a vreg
            case ir_op::local_set: case ir_op::local_tee:
            case ir_op::global_set:
               use_vreg(inst.local.src1);
               break;

            // Branch with condition: br.src1 is a vreg
            case ir_op::br_if: case ir_op::if_:
               use_vreg(inst.br.src1);
               break;

            // br with return value: br.src1 might be a vreg
            case ir_op::br:
               use_vreg(inst.br.src1);
               break;

            // return: rr.src1 is the return value vreg
            case ir_op::return_:
               use_vreg(inst.rr.src1);
               break;

            // arg: rr.src1 is the argument vreg
            case ir_op::arg:
               use_vreg(inst.rr.src1);
               break;

            // memory_grow: rr.src1 is the pages vreg
            case ir_op::memory_grow:
               use_vreg(inst.rr.src1);
               break;

            // br_table: rr.src1 is the index vreg
            case ir_op::br_table:
               use_vreg(inst.rr.src1);
               break;

            // Mov: src1 is a vreg
            case ir_op::mov:
               use_vreg(inst.rr.src1);
               break;

            // v128_op: addr field may reference a GPR vreg; offset field
            // is a vreg only for shift/replace_lane ops, otherwise a literal.
            // For scalar-producing ops, addr is a DEST vreg (not a source).
            case ir_op::v128_op: {
               auto sub = static_cast<simd_sub>(inst.dest);
               // addr is a GP vreg for memory ops and scalar-dest ops, but
               // for bitselect it holds a v128 mask vreg (handled above)
               if (!simd_produces_scalar(sub) && sub != simd_sub::v128_bitselect
                   && inst.simd.addr != ir_vreg_none)
                  use_vreg(inst.simd.addr);
               if (simd_offset_is_vreg(sub))
                  use_vreg(inst.simd.offset);
               break;
            }

            // No source vregs
            case ir_op::nop: case ir_op::unreachable: case ir_op::drop:
            case ir_op::const_i32: case ir_op::const_i64:
            case ir_op::const_f32: case ir_op::const_f64: case ir_op::const_v128:
            case ir_op::local_get: case ir_op::global_get:
            case ir_op::memory_size:
            case ir_op::block: case ir_op::loop: case ir_op::end:
            case ir_op::else_: case ir_op::call: case ir_op::call_indirect:
            default:
               break;
            }
         }
      }

      // Perform linear scan register allocation.
      // Assigns phys_reg to each interval, or spill_slot if no register available.
      static uint32_t allocate_registers(ir_function& func) {
         if (func.interval_count == 0) return 0;

         // Build call bitmap for branchless crosses_call check
         const uint32_t bmp_words = (func.inst_count + 63) / 64;
         uint64_t call_bmp_stack[64]; // stack-allocate for small functions
         uint64_t* call_bmp = (bmp_words <= 64) ? call_bmp_stack : new uint64_t[bmp_words];
         std::memset(call_bmp, 0, bmp_words * sizeof(uint64_t));
         bool has_calls = false;
         for (uint32_t i = 0; i < func.inst_count; ++i) {
            if (func.insts[i].opcode == ir_op::call ||
                func.insts[i].opcode == ir_op::call_indirect ||
                func.insts[i].opcode == ir_op::memory_grow ||
                func.insts[i].opcode == ir_op::memory_size) {
               // memory_grow/size call native functions that clobber caller-saved regs
               call_bmp[i / 64] |= uint64_t(1) << (i % 64);
               has_calls = true;
            }
         }

         // Sort intervals by start position
         std::sort(func.intervals, func.intervals + func.interval_count,
                   [](const ir_live_interval& a, const ir_live_interval& b) {
                      return a.start < b.start;
                   });

         static constexpr int NUM_REGS = static_cast<int>(phys_reg::count);
         uint32_t active[NUM_REGS];
         bool reg_used[NUM_REGS] = {};
         int num_active = 0;
         uint32_t next_spill_slot = 0;

         for (uint32_t i = 0; i < func.interval_count; ++i) {
            auto& interval = func.intervals[i];
            if (interval.start == UINT32_MAX) continue;
            // v128/f32 vregs get XMM registers, not GPRs — handled in second pass
            // f64 stays in GPR for now (XMM f64 has encoding issues to debug)
            if (interval.type == types::v128 || interval.type == types::f32) continue;

            // Branchless bitmap check: any call in (start, end)?
            bool crosses_call = false;
            if (has_calls && interval.end > interval.start + 1) {
               uint32_t lo = interval.start + 1;
               uint32_t hi = interval.end; // exclusive
               uint32_t lo_word = lo / 64, hi_word = (hi - 1) / 64;
               if (lo_word == hi_word) {
                  // Same word: mask bits [lo%64, hi%64)
                  uint64_t mask = (hi % 64 == 0 ? ~uint64_t(0) : (uint64_t(1) << (hi % 64)) - 1)
                                & ~((uint64_t(1) << (lo % 64)) - 1);
                  crosses_call = (call_bmp[lo_word] & mask) != 0;
               } else {
                  // Scan words — still minimal branching (typically 1-2 words)
                  uint64_t acc = call_bmp[lo_word] & ~((uint64_t(1) << (lo % 64)) - 1);
                  for (uint32_t w = lo_word + 1; w < hi_word; ++w) acc |= call_bmp[w];
                  uint64_t hi_mask = hi % 64 == 0 ? ~uint64_t(0) : (uint64_t(1) << (hi % 64)) - 1;
                  acc |= call_bmp[hi_word] & hi_mask;
                  crosses_call = acc != 0;
               }
            }

            // Expire old intervals
            for (int j = 0; j < num_active; ) {
               auto& active_iv = func.intervals[active[j]];
               if (active_iv.end < interval.start) {
                  reg_used[active_iv.phys_reg] = false;
                  active[j] = active[--num_active];
               } else {
                  ++j;
               }
            }

            // Register coalescing: if this vreg is defined by a mov,
            // try to reuse the source's physical register (eliminates the mov).
            int hint_reg = -1;
            if (func.def_inst) {
               uint32_t di = func.def_inst[interval.vreg];
               if (di < func.inst_count && func.insts[di].opcode == ir_op::mov) {
                  uint32_t src_vreg = func.insts[di].rr.src1;
                  if (src_vreg != ir_vreg_none && src_vreg < func.next_vreg) {
                     // Find source's assigned register (already processed)
                     for (uint32_t k = 0; k < i; ++k) {
                        if (func.intervals[k].vreg == src_vreg && func.intervals[k].phys_reg >= 0) {
                           int sr = func.intervals[k].phys_reg;
                           if (!reg_used[sr] &&
                               (!crosses_call || sr >= static_cast<int>(phys_reg::caller_saved_count)))
                              hint_reg = sr;
                           break;
                        }
                     }
                  }
               }
            }

            int assigned = -1;
            if (hint_reg >= 0) {
               assigned = hint_reg;
            } else if (crosses_call) {
               // Must use callee-saved register (survives calls)
               for (int r = static_cast<int>(phys_reg::caller_saved_count); r < NUM_REGS; ++r) {
                  if (!reg_used[r]) {
                     assigned = r;
                     break;
                  }
               }
            } else {
               // Prefer caller-saved registers first (no save/restore overhead)
               for (int r = 0; r < NUM_REGS; ++r) {
                  if (!reg_used[r]) {
                     assigned = r;
                     break;
                  }
               }
            }

            if (assigned >= 0) {
               interval.phys_reg = static_cast<int8_t>(assigned);
               reg_used[assigned] = true;
               active[num_active++] = i;
               if (assigned >= static_cast<int>(phys_reg::caller_saved_count)) {
                  func.callee_saved_used |= (1 << (assigned - static_cast<int>(phys_reg::caller_saved_count)));
               }
            } else {
               interval.phys_reg = -1;
               interval.spill_slot = static_cast<int16_t>(next_spill_slot++);
            }
         }

         // ── Second pass: XMM register allocation for v128 vregs ──
         // xmm0-xmm3 reserved as temps, xmm4-xmm15 available (12 registers)
         static constexpr int NUM_XMM = 12;
         static constexpr int XMM_BASE = 4; // first allocatable = xmm4
         bool xmm_used[NUM_XMM] = {};
         uint32_t xmm_active[NUM_XMM];
         int num_xmm_active = 0;

         // Re-sort intervals (may have been modified by GPR pass)
         std::sort(func.intervals, func.intervals + func.interval_count,
                   [](const ir_live_interval& a, const ir_live_interval& b) {
                      return a.start < b.start;
                   });

         for (uint32_t i = 0; i < func.interval_count; ++i) {
            auto& interval = func.intervals[i];
            if (interval.start == UINT32_MAX) continue;
            // Assign XMM registers to v128 and f32 vregs.
            // f32 in XMM eliminates GPR↔XMM transfers on every float op.
            if (interval.type != types::v128 && interval.type != types::f32) continue;

            // Expire old XMM intervals
            for (int j = 0; j < num_xmm_active; ) {
               auto& active_iv = func.intervals[xmm_active[j]];
               if (active_iv.end < interval.start) {
                  xmm_used[active_iv.phys_xmm - XMM_BASE] = false;
                  xmm_active[j] = xmm_active[--num_xmm_active];
               } else {
                  ++j;
               }
            }

            // Assign XMM register
            int assigned = -1;
            for (int r = 0; r < NUM_XMM; ++r) {
               if (!xmm_used[r]) {
                  assigned = r;
                  break;
               }
            }

            if (assigned >= 0) {
               interval.phys_xmm = static_cast<int8_t>(assigned + XMM_BASE);
               xmm_used[assigned] = true;
               xmm_active[num_xmm_active++] = i;
            } else {
               // Spill: v128 spill slots use 16 bytes (2 x 8-byte slots)
               interval.phys_xmm = -1;
               interval.spill_slot = static_cast<int16_t>(next_spill_slot);
               next_spill_slot += 2; // 16 bytes
            }
         }

         if (bmp_words > 64) delete[] call_bmp;
         func.num_spill_slots = next_spill_slot;

         return next_spill_slot;
      }
   };

}} // namespace eosio::vm
