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
         if (func.vstack_top > 0) {
            uint32_t ret_vreg = func.vstack[func.vstack_top - 1];
            if (ret_vreg < num_vregs) {
               func.intervals[ret_vreg].end = func.inst_count;
            }
         }

         // Scan instructions to find first def and last use of each vreg
         for (uint32_t i = 0; i < func.inst_count; ++i) {
            const auto& inst = func.insts[i];

            // Destination vreg: defined at this instruction
            // Exception: for store instructions, dest holds the VALUE vreg (a use, not def)
            // — handled in the switch below instead.
            // block_start/block_end use dest for block_idx, not a vreg — skip them.
            bool is_store = (inst.opcode >= ir_op::i32_store && inst.opcode <= ir_op::i64_store32);
            bool is_block_marker = (inst.opcode == ir_op::block_start || inst.opcode == ir_op::block_end);
            if (!is_store && !is_block_marker && inst.dest != ir_vreg_none && inst.dest < num_vregs) {
               auto& iv = func.intervals[inst.dest];
               if (i < iv.start) iv.start = i;
               if (i > iv.end) iv.end = i;
               iv.type = inst.type;
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
                func.insts[i].opcode == ir_op::call_indirect) {
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

         if (bmp_words > 64) delete[] call_bmp;
         func.num_spill_slots = next_spill_slot;
         return next_spill_slot;
      }
   };

}} // namespace eosio::vm
