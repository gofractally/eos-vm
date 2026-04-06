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
      // rax(0) and rcx(1) reserved as temporaries for spill loads
      rdx = 0, r8 = 1, r9 = 2, r10 = 3, r11 = 4,
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
            // Use the rr union (src1, src2) — same layout as br, ri, etc.
            if (inst.rr.src1 != ir_vreg_none && inst.rr.src1 < num_vregs) {
               auto& iv = func.intervals[inst.rr.src1];
               if (i < iv.start) iv.start = i;
               if (i > iv.end) iv.end = i;
            }
            if (inst.rr.src2 != ir_vreg_none && inst.rr.src2 < num_vregs) {
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

         // Sort intervals by start position (use a simple insertion sort
         // since it's allocated from growable_allocator and we can't use std::sort
         // with non-contiguous memory... actually it IS contiguous)
         std::sort(func.intervals, func.intervals + func.interval_count,
                   [](const ir_live_interval& a, const ir_live_interval& b) {
                      return a.start < b.start;
                   });

         // Active intervals (currently assigned to a register)
         // Using a simple fixed-size array since we have at most 12 registers
         static constexpr int NUM_REGS = static_cast<int>(phys_reg::count);
         uint32_t active[NUM_REGS]; // index into intervals array
         bool reg_used[NUM_REGS] = {};
         int num_active = 0;
         uint32_t next_spill_slot = 0;

         for (uint32_t i = 0; i < func.interval_count; ++i) {
            auto& interval = func.intervals[i];

            // Skip unused vregs
            if (interval.start == UINT32_MAX) continue;

            // Expire old intervals
            for (int j = 0; j < num_active; ) {
               auto& active_iv = func.intervals[active[j]];
               if (active_iv.end < interval.start) {
                  // This interval has expired — free its register
                  reg_used[active_iv.phys_reg] = false;
                  active[j] = active[--num_active]; // swap-remove
               } else {
                  ++j;
               }
            }

            // Try to assign a register
            int assigned = -1;
            for (int r = 0; r < NUM_REGS; ++r) {
               if (!reg_used[r]) {
                  assigned = r;
                  break;
               }
            }

            if (assigned >= 0) {
               interval.phys_reg = static_cast<int8_t>(assigned);
               reg_used[assigned] = true;
               active[num_active++] = i;
            } else {
               // Spill: assign a stack slot
               // TODO: spill the interval that ends latest (better heuristic)
               interval.phys_reg = -1;
               interval.spill_slot = static_cast<int16_t>(next_spill_slot++);
            }
         }

         func.num_spill_slots = next_spill_slot;
         return next_spill_slot;
      }
   };

}} // namespace eosio::vm
