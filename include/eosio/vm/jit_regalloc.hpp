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
      // rax and rcx reserved as temporaries for spill loads
      // Caller-saved (free, no save/restore):
      rdx = 0, r8 = 1, r9 = 2, r10 = 3, r11 = 4,
      // Callee-saved disabled for debugging
      caller_saved_count = 5,
      count = 5,
   };

   class jit_regalloc {
    public:
      // Compute live intervals for all vregs in a function.
      // intervals array must have space for func.next_vreg entries.
      static void compute_live_intervals(ir_function& func, growable_allocator& alloc) {
         uint32_t num_vregs = func.next_vreg;
         if (num_vregs == 0) return;

         // Allocate intervals from growable_allocator
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

         // Scan instructions to find first def and last use of each vreg
         for (uint32_t i = 0; i < func.inst_count; ++i) {
            const auto& inst = func.insts[i];

            // Destination vreg: defined at this instruction
            if (inst.dest != ir_vreg_none && inst.dest < num_vregs) {
               auto& iv = func.intervals[inst.dest];
               if (i < iv.start) iv.start = i;
               if (i > iv.end) iv.end = i;
               iv.type = inst.type;
            }

            // Source vregs: used at this instruction
            // Note: for load/store instructions, rr.src2 == ri.imm (not a vreg!)
            if (inst.rr.src1 != ir_vreg_none && inst.rr.src1 < num_vregs) {
               auto& iv = func.intervals[inst.rr.src1];
               if (i < iv.start) iv.start = i;
               if (i > iv.end) iv.end = i;
            }
            // Only read src2 for instructions that use the rr union (not ri/br/call)
            bool has_src2 = (inst.opcode >= ir_op::i32_eqz && inst.opcode <= ir_op::i64_rotr)
                         || inst.opcode == ir_op::select;
            if (has_src2 && inst.rr.src2 != ir_vreg_none && inst.rr.src2 < num_vregs) {
               auto& iv = func.intervals[inst.rr.src2];
               if (i < iv.start) iv.start = i;
               if (i > iv.end) iv.end = i;
            }
         }
      }

      // Perform linear scan register allocation.
      // Assigns phys_reg to each interval, or spill_slot if no register available.
      static uint32_t allocate_registers(ir_function& func) {
         if (func.interval_count == 0) return 0;

         // Find call instruction positions — intervals spanning calls must be spilled
         // since we use caller-saved registers
         bool has_calls = false;
         for (uint32_t i = 0; i < func.inst_count; ++i) {
            if (func.insts[i].opcode == ir_op::call ||
                func.insts[i].opcode == ir_op::call_indirect) {
               has_calls = true;
               break;
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

            // Check if this interval crosses a call instruction
            bool crosses_call = false;
            if (has_calls) {
               for (uint32_t j = 0; j < func.inst_count; ++j) {
                  if ((func.insts[j].opcode == ir_op::call ||
                       func.insts[j].opcode == ir_op::call_indirect) &&
                      j > interval.start && j < interval.end) {
                     crosses_call = true;
                     break;
                  }
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

            int assigned = -1;
            if (crosses_call) {
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
            } else {
               interval.phys_reg = -1;
               interval.spill_slot = static_cast<int16_t>(next_spill_slot++);
            }
         }

         func.num_spill_slots = next_spill_slot;

         // Compute max liveness for diagnostics
         uint32_t max_live = 0;
         for (uint32_t pos = 0; pos < func.inst_count; ++pos) {
            uint32_t live = 0;
            for (uint32_t iv = 0; iv < func.interval_count; ++iv) {
               if (func.intervals[iv].start != UINT32_MAX &&
                   func.intervals[iv].start <= pos && pos <= func.intervals[iv].end)
                  ++live;
            }
            if (live > max_live) max_live = live;
         }
         fprintf(stderr, "regalloc func %u: %u vregs, %u in regs, %u spilled, max_live=%u\n",
                 func.func_index, func.interval_count,
                 func.interval_count - next_spill_slot, next_spill_slot, max_live);

         return next_spill_slot;
      }
   };

}} // namespace eosio::vm
