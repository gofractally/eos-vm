#pragma once

// IR optimization pass for jit2.
// Runs after IR building (Pass 1), before regalloc (Pass 1.5).
//
// 1. Constant folding: evaluate const-op-const at compile time
// 2. Dead code elimination: mark unused vregs as IR_DEAD
// 3. Simple strength reduction: identity ops, zero ops

#include <eosio/vm/jit_ir.hpp>
#include <cstring>

namespace eosio { namespace vm {

   struct jit_optimizer {

      // Run all optimization passes on a single function.
      static void optimize(ir_function& func, jit_scratch_allocator& scratch) {
         if (func.inst_count == 0 || func.next_vreg == 0) return;

         const uint32_t num_vregs = func.next_vreg;
         const uint32_t n = func.inst_count;

         // Allocate SSA use-chain arrays (persisted on ir_function for codegen)
         func.use_count = scratch.alloc<uint16_t>(num_vregs);
         func.def_inst  = scratch.alloc<uint32_t>(num_vregs);
         std::memset(func.use_count, 0, num_vregs * sizeof(uint16_t));
         for (uint32_t v = 0; v < num_vregs; ++v) func.def_inst[v] = UINT32_MAX;

         auto* use_count = func.use_count;
         auto* def_inst  = func.def_inst;

         // Allocate constant tracking arrays (persisted on ir_function for codegen)
         func.const_val = scratch.alloc<int64_t>(num_vregs);
         func.is_const  = scratch.alloc<uint8_t>(num_vregs);
         auto* const_val = func.const_val;
         auto* is_const  = func.is_const;
         std::memset(is_const, 0, num_vregs);

         // ── Phase 1: Constant propagation + folding ──
         for (uint32_t i = 0; i < n; ++i) {
            auto& inst = func.insts[i];
            if (inst.flags & IR_DEAD) continue;

            // Track definitions
            bool is_store_op = (inst.opcode >= ir_op::i32_store && inst.opcode <= ir_op::i64_store32);
            bool is_block_mk = (inst.opcode == ir_op::block_start || inst.opcode == ir_op::block_end);
            if (!is_store_op && !is_block_mk && inst.dest != ir_vreg_none && inst.dest < num_vregs) {
               // If this vreg was already defined, it's a phi merge from different
               // control flow paths. Kill constant status — the value depends on
               // which branch was taken at runtime.
               if (def_inst[inst.dest] != UINT32_MAX) {
                  is_const[inst.dest] = 0;
               }
               def_inst[inst.dest] = i;
            }

            // Track constants
            if (inst.opcode == ir_op::const_i32 || inst.opcode == ir_op::const_i64) {
               if (inst.dest < num_vregs) {
                  const_val[inst.dest] = inst.imm64;
                  is_const[inst.dest] = 1;
               }
               continue;
            }

            // Fold binary integer ops with two constant operands
            if (inst.rr.src1 < num_vregs && inst.rr.src2 < num_vregs &&
                is_const[inst.rr.src1] && is_const[inst.rr.src2]) {
               int64_t result;
               if (fold_binop(inst.opcode, const_val[inst.rr.src1], const_val[inst.rr.src2], result)) {
                  bool is32 = (inst.type == types::i32);
                  inst.opcode = is32 ? ir_op::const_i32 : ir_op::const_i64;
                  inst.imm64 = result;
                  if (inst.dest < num_vregs) {
                     const_val[inst.dest] = result;
                     is_const[inst.dest] = 1;
                  }
                  continue;
               }
            }

            // Fold unary ops with constant operand
            if (inst.rr.src1 < num_vregs && is_const[inst.rr.src1]) {
               int64_t result;
               if (fold_unop(inst.opcode, const_val[inst.rr.src1], result)) {
                  bool is32 = (inst.type == types::i32);
                  inst.opcode = is32 ? ir_op::const_i32 : ir_op::const_i64;
                  inst.imm64 = result;
                  if (inst.dest < num_vregs) {
                     const_val[inst.dest] = result;
                     is_const[inst.dest] = 1;
                  }
                  continue;
               }
            }

            // Strength reduction: identity operations
            // x + 0, x - 0, x | 0, x ^ 0, x << 0, x >> 0 → mov x
            // x * 1, x / 1 → mov x
            // x & 0, x * 0 → const 0
            // x | -1, x & -1 → depends
            if (inst.rr.src2 < num_vregs && is_const[inst.rr.src2]) {
               int64_t c = const_val[inst.rr.src2];
               bool reduced = false;
               switch (inst.opcode) {
               case ir_op::i32_add: case ir_op::i32_sub:
               case ir_op::i32_or: case ir_op::i32_xor:
               case ir_op::i32_shl: case ir_op::i32_shr_s: case ir_op::i32_shr_u:
               case ir_op::i32_rotl: case ir_op::i32_rotr:
               case ir_op::i64_add: case ir_op::i64_sub:
               case ir_op::i64_or: case ir_op::i64_xor:
               case ir_op::i64_shl: case ir_op::i64_shr_s: case ir_op::i64_shr_u:
               case ir_op::i64_rotl: case ir_op::i64_rotr:
                  if (c == 0) { // x op 0 → x
                     inst.opcode = ir_op::mov;
                     inst.rr.src2 = ir_vreg_none;
                     reduced = true;
                  }
                  break;
               case ir_op::i32_mul: case ir_op::i64_mul:
                  if (c == 1) { // x * 1 → x
                     inst.opcode = ir_op::mov;
                     inst.rr.src2 = ir_vreg_none;
                     reduced = true;
                  } else if (c == 0) { // x * 0 → 0
                     bool is32 = (inst.opcode == ir_op::i32_mul);
                     inst.opcode = is32 ? ir_op::const_i32 : ir_op::const_i64;
                     inst.imm64 = 0;
                     if (inst.dest < num_vregs) { const_val[inst.dest] = 0; is_const[inst.dest] = 1; }
                     reduced = true;
                  }
                  // NOTE: mul-by-power-of-2 → shl moved to codegen layer
                  // (try_emit_mul_as_shl) to avoid corrupting shared constant vregs.
                  break;
               case ir_op::i32_and: case ir_op::i64_and:
                  if (c == 0) { // x & 0 → 0
                     bool is32 = (inst.opcode == ir_op::i32_and);
                     inst.opcode = is32 ? ir_op::const_i32 : ir_op::const_i64;
                     inst.imm64 = 0;
                     if (inst.dest < num_vregs) { const_val[inst.dest] = 0; is_const[inst.dest] = 1; }
                     reduced = true;
                  }
                  break;
               default: break;
               }
               (void)reduced;
            }

            // Non-const dest: kill any tracked constant for this vreg
            if (inst.dest < num_vregs) {
               is_const[inst.dest] = 0;
            }
         }

         // ── Phase 2: Count vreg uses ──
         // Count the function return value as a use (read by epilogue, not by any IR instruction)
         // For v128 returns, the low vreg (vstack_top-2) is the actual value.
         if (func.vstack_top > 0) {
            bool is_v128_ret = func.type && func.type->return_count > 0
                               && func.type->return_type == types::v128;
            uint32_t ret_idx = is_v128_ret && func.vstack_top >= 2
                               ? func.vstack_top - 2 : func.vstack_top - 1;
            uint32_t ret_vreg = func.vstack[ret_idx];
            if (ret_vreg != ir_vreg_none && ret_vreg < num_vregs)
               use_count[ret_vreg]++;
         }
         for (uint32_t i = 0; i < n; ++i) {
            const auto& inst = func.insts[i];
            if (inst.flags & IR_DEAD) continue;

            auto count_use = [&](uint32_t vreg) {
               if (vreg != ir_vreg_none && vreg < num_vregs && use_count[vreg] < UINT16_MAX)
                  use_count[vreg]++;
            };

            bool is_store = (inst.opcode >= ir_op::i32_store && inst.opcode <= ir_op::i64_store32);
            bool is_block_marker = (inst.opcode == ir_op::block_start || inst.opcode == ir_op::block_end);

            // Source vregs — must check per-opcode which union fields are vregs
            switch (inst.opcode) {
            // Binary ops: src1 and src2
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
            case ir_op::i32_eq: case ir_op::i32_ne:
            case ir_op::i32_lt_s: case ir_op::i32_lt_u: case ir_op::i32_gt_s: case ir_op::i32_gt_u:
            case ir_op::i32_le_s: case ir_op::i32_le_u: case ir_op::i32_ge_s: case ir_op::i32_ge_u:
            case ir_op::i64_eq: case ir_op::i64_ne:
            case ir_op::i64_lt_s: case ir_op::i64_lt_u: case ir_op::i64_gt_s: case ir_op::i64_gt_u:
            case ir_op::i64_le_s: case ir_op::i64_le_u: case ir_op::i64_ge_s: case ir_op::i64_ge_u:
            case ir_op::f32_add: case ir_op::f32_sub: case ir_op::f32_mul: case ir_op::f32_div:
            case ir_op::f32_min: case ir_op::f32_max: case ir_op::f32_copysign:
            case ir_op::f64_add: case ir_op::f64_sub: case ir_op::f64_mul: case ir_op::f64_div:
            case ir_op::f64_min: case ir_op::f64_max: case ir_op::f64_copysign:
            case ir_op::f32_eq: case ir_op::f32_ne: case ir_op::f32_lt: case ir_op::f32_gt:
            case ir_op::f32_le: case ir_op::f32_ge:
            case ir_op::f64_eq: case ir_op::f64_ne: case ir_op::f64_lt: case ir_op::f64_gt:
            case ir_op::f64_le: case ir_op::f64_ge:
               count_use(inst.rr.src1);
               count_use(inst.rr.src2);
               break;
            // Unary ops: src1 only
            case ir_op::i32_eqz: case ir_op::i64_eqz:
            case ir_op::i32_clz: case ir_op::i32_ctz: case ir_op::i32_popcnt:
            case ir_op::i64_clz: case ir_op::i64_ctz: case ir_op::i64_popcnt:
            case ir_op::i32_wrap_i64:
            case ir_op::i32_trunc_s_f32: case ir_op::i32_trunc_u_f32:
            case ir_op::i32_trunc_s_f64: case ir_op::i32_trunc_u_f64:
            case ir_op::i64_extend_s_i32: case ir_op::i64_extend_u_i32:
            case ir_op::i64_trunc_s_f32: case ir_op::i64_trunc_u_f32:
            case ir_op::i64_trunc_s_f64: case ir_op::i64_trunc_u_f64:
            case ir_op::f32_convert_s_i32: case ir_op::f32_convert_u_i32:
            case ir_op::f32_convert_s_i64: case ir_op::f32_convert_u_i64:
            case ir_op::f32_demote_f64:
            case ir_op::f64_convert_s_i32: case ir_op::f64_convert_u_i32:
            case ir_op::f64_convert_s_i64: case ir_op::f64_convert_u_i64:
            case ir_op::f64_promote_f32:
            case ir_op::i32_reinterpret_f32: case ir_op::i64_reinterpret_f64:
            case ir_op::f32_reinterpret_i32: case ir_op::f64_reinterpret_i64:
            case ir_op::i32_trunc_sat_f32_s: case ir_op::i32_trunc_sat_f32_u:
            case ir_op::i32_trunc_sat_f64_s: case ir_op::i32_trunc_sat_f64_u:
            case ir_op::i64_trunc_sat_f32_s: case ir_op::i64_trunc_sat_f32_u:
            case ir_op::i64_trunc_sat_f64_s: case ir_op::i64_trunc_sat_f64_u:
            case ir_op::i32_extend8_s: case ir_op::i32_extend16_s:
            case ir_op::i64_extend8_s: case ir_op::i64_extend16_s: case ir_op::i64_extend32_s:
            case ir_op::f32_abs: case ir_op::f32_neg: case ir_op::f32_ceil: case ir_op::f32_floor:
            case ir_op::f32_trunc: case ir_op::f32_nearest: case ir_op::f32_sqrt:
            case ir_op::f64_abs: case ir_op::f64_neg: case ir_op::f64_ceil: case ir_op::f64_floor:
            case ir_op::f64_trunc: case ir_op::f64_nearest: case ir_op::f64_sqrt:
            case ir_op::mov:
               count_use(inst.rr.src1);
               break;
            // Loads: src1 is address vreg
            case ir_op::i32_load: case ir_op::i64_load: case ir_op::f32_load: case ir_op::f64_load:
            case ir_op::i32_load8_s: case ir_op::i32_load8_u: case ir_op::i32_load16_s: case ir_op::i32_load16_u:
            case ir_op::i64_load8_s: case ir_op::i64_load8_u: case ir_op::i64_load16_s: case ir_op::i64_load16_u:
            case ir_op::i64_load32_s: case ir_op::i64_load32_u:
               count_use(inst.ri.src1);
               break;
            // Stores: src1 is address, dest is value
            case ir_op::i32_store: case ir_op::i64_store: case ir_op::f32_store: case ir_op::f64_store:
            case ir_op::i32_store8: case ir_op::i32_store16:
            case ir_op::i64_store8: case ir_op::i64_store16: case ir_op::i64_store32:
               count_use(inst.ri.src1);
               count_use(inst.dest); // dest = value vreg for stores
               break;
            // Control flow
            case ir_op::if_:
            case ir_op::br_if:
               count_use(inst.br.src1);
               break;
            case ir_op::br:
               count_use(inst.br.src1);
               break;
            case ir_op::br_table:
               count_use(inst.rr.src1);
               break;
            case ir_op::return_:
               count_use(inst.rr.src1);
               break;
            // Calls: args tracked via arg instructions
            case ir_op::arg:
               count_use(inst.rr.src1);
               break;
            case ir_op::call:
            case ir_op::call_indirect:
               count_use(inst.call.src1);
               break;
            // Select: uses 3 vregs packed in sel union
            case ir_op::select:
               count_use(inst.sel.val1);
               count_use(inst.sel.val2);
               count_use(inst.sel.cond);
               break;
            // Local ops
            case ir_op::local_set:
            case ir_op::local_tee:
               count_use(inst.local.src1);
               break;
            case ir_op::global_set:
               count_use(inst.local.src1);
               break;
            case ir_op::memory_grow:
               count_use(inst.rr.src1);
               break;
            case ir_op::v128_op: {
               auto sub = static_cast<simd_sub>(inst.dest);
               // For scalar-producing ops, addr is a DEST vreg, not a source.
               // For bitselect, addr holds a v128 mask vreg (not a GP use).
               if (!simd_produces_scalar(sub) && sub != simd_sub::v128_bitselect
                   && inst.simd.addr != ir_vreg_none)
                  count_use(inst.simd.addr);
               if (simd_offset_is_vreg(sub))
                  count_use(inst.simd.offset);
               break;
            }
            default:
               // All opcode categories with vreg sources are handled above.
               // Unhandled opcodes (memory_fill, memory_copy, table ops, etc.)
               // use stack-based args — their vreg uses are counted via ir_op::arg.
               break;
            }
         }

         // The function epilogue reads the return vreg outside of IR.
         // Add an artificial use so DCE doesn't eliminate the mov that writes it.
         if (func.vstack_top > 0 && func.type && func.type->return_count > 0) {
            bool is_v128_ret = func.type->return_type == types::v128;
            uint32_t ret_idx = is_v128_ret && func.vstack_top >= 2
                               ? func.vstack_top - 2 : func.vstack_top - 1;
            uint32_t ret_vreg = func.vstack[ret_idx];
            if (ret_vreg < num_vregs)
               use_count[ret_vreg]++;
         }

         // ── Phase 3: Dead code elimination ──
         for (uint32_t i = 0; i < n; ++i) {
            auto& inst = func.insts[i];
            if (inst.flags & (IR_DEAD | IR_SIDE_EFFECT)) continue;
            if (inst.dest == ir_vreg_none || inst.dest >= num_vregs) continue;
            if (use_count[inst.dest] != 0) continue;

            // Only eliminate pure computation: const, binop, unop, comparison, mov
            bool is_pure = false;
            switch (inst.opcode) {
            case ir_op::const_i32: case ir_op::const_i64:
            case ir_op::const_f32: case ir_op::const_f64:
            case ir_op::mov:
               is_pure = true; break;
            default:
               // Arithmetic, comparison, conversion — check opcode ranges
               is_pure = (inst.opcode >= ir_op::i32_eqz && inst.opcode <= ir_op::i64_extend32_s)
                       || (inst.opcode >= ir_op::i32_add && inst.opcode <= ir_op::f64_copysign)
                       || (inst.opcode >= ir_op::f32_abs && inst.opcode <= ir_op::f64_sqrt);
               // Exclude trapping ops: div/rem can trap on div-by-zero or overflow,
               // trunc can trap on NaN/overflow — must not be eliminated even if unused
               if (is_pure) {
                  switch (inst.opcode) {
                  case ir_op::i32_div_s: case ir_op::i32_div_u:
                  case ir_op::i32_rem_s: case ir_op::i32_rem_u:
                  case ir_op::i64_div_s: case ir_op::i64_div_u:
                  case ir_op::i64_rem_s: case ir_op::i64_rem_u:
                  case ir_op::i32_trunc_s_f32: case ir_op::i32_trunc_u_f32:
                  case ir_op::i32_trunc_s_f64: case ir_op::i32_trunc_u_f64:
                  case ir_op::i64_trunc_s_f32: case ir_op::i64_trunc_u_f32:
                  case ir_op::i64_trunc_s_f64: case ir_op::i64_trunc_u_f64:
                     is_pure = false; break;
                  default: break;
                  }
               }
               break;
            }
            if (is_pure) {
               inst.flags |= IR_DEAD;
            }
         }

         // ── Phase 3.5: Copy propagation ──
         // When mov src → dest has use_count[dest] == 1, find the single consumer
         // and replace its reference from dest to src, eliminating the mov.
         // Skip the function body return vreg (read by epilogue outside IR).
         uint32_t return_vreg = ir_vreg_none;
         if (func.vstack_top > 0 && func.type && func.type->return_count > 0) {
            bool is_v128_ret = func.type->return_type == types::v128;
            uint32_t ret_idx = is_v128_ret && func.vstack_top >= 2
                               ? func.vstack_top - 2 : func.vstack_top - 1;
            return_vreg = func.vstack[ret_idx];
         }
         for (uint32_t i = 0; i < n; ++i) {
            auto& inst = func.insts[i];
            if (inst.flags & IR_DEAD) continue;
            if (inst.opcode != ir_op::mov) continue;
            uint32_t mv_dest = inst.dest;
            uint32_t mv_src = inst.rr.src1;
            if (mv_dest >= num_vregs || mv_src >= num_vregs) continue;
            if (mv_dest == mv_src) { inst.flags |= IR_DEAD; continue; } // self-mov
            if (use_count[mv_dest] != 1) continue;
            if (mv_dest == return_vreg) continue; // epilogue reads this outside IR
            // Skip phi-node merge vregs (written by multiple control flow paths).
            // Count non-dead definitions of mv_dest — only propagate if exactly 1.
            {
               uint32_t def_count = 0;
               for (uint32_t k = 0; k < n && def_count < 2; ++k) {
                  auto& ki = func.insts[k];
                  if (ki.flags & IR_DEAD) continue;
                  bool is_s = (ki.opcode >= ir_op::i32_store && ki.opcode <= ir_op::i64_store32);
                  bool is_b = (ki.opcode == ir_op::block_start || ki.opcode == ir_op::block_end);
                  if (!is_s && !is_b && ki.dest == mv_dest) def_count++;
               }
               if (def_count != 1) continue;
            }
            // Scan forward for the single consumer (bounded window)
            for (uint32_t j = i + 1; j < n && j < i + 32; ++j) {
               auto& u = func.insts[j];
               if (u.flags & IR_DEAD) continue;
               if (replace_vreg_use(u, mv_dest, mv_src)) {
                  inst.flags |= IR_DEAD;
                  use_count[mv_dest] = 0;
                  if (use_count[mv_src] < UINT16_MAX) use_count[mv_src]++;
                  break;
               }
            }
         }

         // ── Phase 4: Instruction fusion (cmp + branch) ──
         for (uint32_t i = 0; i + 1 < n; ++i) {
            auto& inst = func.insts[i];
            if (inst.flags & IR_DEAD) continue;
            if (!is_comparison(inst.opcode)) continue;
            if (inst.dest == ir_vreg_none || inst.dest >= num_vregs) continue;
            if (use_count[inst.dest] != 1) continue;

            auto& next = func.insts[i + 1];
            if (next.opcode == ir_op::if_ && next.br.src1 == inst.dest) {
               // OK — fuse with if_
            } else if (next.opcode == ir_op::br_if && next.br.src1 == inst.dest
                       && next.dest == 0) {  // dest = depth_change; only fuse when 0 (no multipop)
               // OK — fuse with br_if (no multipop)
            } else {
               continue;
            }

            inst.flags |= IR_FUSE_NEXT;
            next.flags |= IR_DEAD;
         }
      }

   private:
      static bool is_comparison(ir_op op) {
         return op == ir_op::i32_eqz || op == ir_op::i64_eqz
             || (op >= ir_op::i32_eq && op <= ir_op::i32_ge_u)
             || (op >= ir_op::i64_eq && op <= ir_op::i64_ge_u)
             // Float lt/le/gt/ge use ucomiss+jcc fusion in the codegen.
             // Float eq/ne are NOT fused (need JP handling for NaN).
             || op == ir_op::f32_lt || op == ir_op::f32_gt
             || op == ir_op::f32_le || op == ir_op::f32_ge
             || op == ir_op::f64_lt || op == ir_op::f64_gt
             || op == ir_op::f64_le || op == ir_op::f64_ge;
      }

      // Evaluate a binary integer op at compile time.
      static bool fold_binop(ir_op op, int64_t a, int64_t b, int64_t& result) {
         uint32_t a32 = static_cast<uint32_t>(a);
         uint32_t b32 = static_cast<uint32_t>(b);
         uint64_t ua = static_cast<uint64_t>(a);
         uint64_t ub = static_cast<uint64_t>(b);
         switch (op) {
         case ir_op::i32_add:   result = static_cast<int32_t>(a32 + b32); return true;
         case ir_op::i32_sub:   result = static_cast<int32_t>(a32 - b32); return true;
         case ir_op::i32_mul:   result = static_cast<int32_t>(a32 * b32); return true;
         case ir_op::i32_and:   result = static_cast<int32_t>(a32 & b32); return true;
         case ir_op::i32_or:    result = static_cast<int32_t>(a32 | b32); return true;
         case ir_op::i32_xor:   result = static_cast<int32_t>(a32 ^ b32); return true;
         case ir_op::i32_shl:   result = static_cast<int32_t>(a32 << (b32 & 31)); return true;
         case ir_op::i32_shr_u: result = static_cast<int32_t>(a32 >> (b32 & 31)); return true;
         case ir_op::i32_shr_s: result = static_cast<int32_t>(static_cast<int32_t>(a32) >> (b32 & 31)); return true;
         case ir_op::i32_eq:    result = (a32 == b32) ? 1 : 0; return true;
         case ir_op::i32_ne:    result = (a32 != b32) ? 1 : 0; return true;
         case ir_op::i32_lt_s:  result = (static_cast<int32_t>(a32) < static_cast<int32_t>(b32)) ? 1 : 0; return true;
         case ir_op::i32_lt_u:  result = (a32 < b32) ? 1 : 0; return true;
         case ir_op::i32_gt_s:  result = (static_cast<int32_t>(a32) > static_cast<int32_t>(b32)) ? 1 : 0; return true;
         case ir_op::i32_gt_u:  result = (a32 > b32) ? 1 : 0; return true;
         case ir_op::i32_le_s:  result = (static_cast<int32_t>(a32) <= static_cast<int32_t>(b32)) ? 1 : 0; return true;
         case ir_op::i32_le_u:  result = (a32 <= b32) ? 1 : 0; return true;
         case ir_op::i32_ge_s:  result = (static_cast<int32_t>(a32) >= static_cast<int32_t>(b32)) ? 1 : 0; return true;
         case ir_op::i32_ge_u:  result = (a32 >= b32) ? 1 : 0; return true;
         case ir_op::i64_add:   result = a + b; return true;
         case ir_op::i64_sub:   result = a - b; return true;
         case ir_op::i64_mul:   result = a * b; return true;
         case ir_op::i64_and:   result = a & b; return true;
         case ir_op::i64_or:    result = a | b; return true;
         case ir_op::i64_xor:   result = a ^ b; return true;
         case ir_op::i64_shl:   result = static_cast<int64_t>(ua << (ub & 63)); return true;
         case ir_op::i64_shr_u: result = static_cast<int64_t>(ua >> (ub & 63)); return true;
         case ir_op::i64_shr_s: result = a >> (b & 63); return true;
         case ir_op::i64_eq:    result = (a == b) ? 1 : 0; return true;
         case ir_op::i64_ne:    result = (a != b) ? 1 : 0; return true;
         case ir_op::i64_lt_s:  result = (a < b) ? 1 : 0; return true;
         case ir_op::i64_lt_u:  result = (ua < ub) ? 1 : 0; return true;
         case ir_op::i64_gt_s:  result = (a > b) ? 1 : 0; return true;
         case ir_op::i64_gt_u:  result = (ua > ub) ? 1 : 0; return true;
         case ir_op::i64_le_s:  result = (a <= b) ? 1 : 0; return true;
         case ir_op::i64_le_u:  result = (ua <= ub) ? 1 : 0; return true;
         case ir_op::i64_ge_s:  result = (a >= b) ? 1 : 0; return true;
         case ir_op::i64_ge_u:  result = (ua >= ub) ? 1 : 0; return true;
         // Skip div/rem — can't fold if divisor is 0
         default: return false;
         }
      }

      // Evaluate a unary integer op at compile time.
      static bool fold_unop(ir_op op, int64_t a, int64_t& result) {
         uint32_t a32 = static_cast<uint32_t>(a);
         switch (op) {
         case ir_op::i32_eqz:       result = (a32 == 0) ? 1 : 0; return true;
         case ir_op::i64_eqz:       result = (a == 0) ? 1 : 0; return true;
         case ir_op::i32_clz:       result = a32 ? __builtin_clz(a32) : 32; return true;
         case ir_op::i32_ctz:       result = a32 ? __builtin_ctz(a32) : 32; return true;
         case ir_op::i32_popcnt:    result = __builtin_popcount(a32); return true;
         case ir_op::i64_clz:       result = a ? __builtin_clzll(static_cast<uint64_t>(a)) : 64; return true;
         case ir_op::i64_ctz:       result = a ? __builtin_ctzll(static_cast<uint64_t>(a)) : 64; return true;
         case ir_op::i64_popcnt:    result = __builtin_popcountll(static_cast<uint64_t>(a)); return true;
         case ir_op::i32_wrap_i64:  result = static_cast<int32_t>(a32); return true;
         case ir_op::i64_extend_s_i32: result = static_cast<int32_t>(a32); return true;
         case ir_op::i64_extend_u_i32: result = a32; return true;
         case ir_op::i32_extend8_s:  result = static_cast<int32_t>(static_cast<int8_t>(a32)); return true;
         case ir_op::i32_extend16_s: result = static_cast<int32_t>(static_cast<int16_t>(a32)); return true;
         case ir_op::i64_extend8_s:  result = static_cast<int8_t>(a); return true;
         case ir_op::i64_extend16_s: result = static_cast<int16_t>(a); return true;
         case ir_op::i64_extend32_s: result = static_cast<int32_t>(a); return true;
         default: return false;
         }
      }

      // Replace a vreg reference in an instruction's source operands.
      // Returns true if the replacement was made, false if old_vreg not found.
      // Handles each opcode's specific union layout correctly.
      static bool replace_vreg_use(ir_inst& inst, uint32_t old_vreg, uint32_t new_vreg) {
         bool found = false;
         auto try_replace = [&](uint32_t& field) {
            if (field == old_vreg) { field = new_vreg; found = true; }
         };
         auto try_replace16 = [&](uint16_t& field) {
            if (field == old_vreg) { field = static_cast<uint16_t>(new_vreg); found = true; }
         };
         switch (inst.opcode) {
         // rr format: binary ops, comparisons, unary ops, mov, br_table, return, arg, drop, memory_grow
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
         case ir_op::i32_eq: case ir_op::i32_ne:
         case ir_op::i32_lt_s: case ir_op::i32_lt_u: case ir_op::i32_gt_s: case ir_op::i32_gt_u:
         case ir_op::i32_le_s: case ir_op::i32_le_u: case ir_op::i32_ge_s: case ir_op::i32_ge_u:
         case ir_op::i64_eq: case ir_op::i64_ne:
         case ir_op::i64_lt_s: case ir_op::i64_lt_u: case ir_op::i64_gt_s: case ir_op::i64_gt_u:
         case ir_op::i64_le_s: case ir_op::i64_le_u: case ir_op::i64_ge_s: case ir_op::i64_ge_u:
         case ir_op::f32_add: case ir_op::f32_sub: case ir_op::f32_mul: case ir_op::f32_div:
         case ir_op::f32_min: case ir_op::f32_max: case ir_op::f32_copysign:
         case ir_op::f64_add: case ir_op::f64_sub: case ir_op::f64_mul: case ir_op::f64_div:
         case ir_op::f64_min: case ir_op::f64_max: case ir_op::f64_copysign:
         case ir_op::f32_eq: case ir_op::f32_ne: case ir_op::f32_lt: case ir_op::f32_gt:
         case ir_op::f32_le: case ir_op::f32_ge:
         case ir_op::f64_eq: case ir_op::f64_ne: case ir_op::f64_lt: case ir_op::f64_gt:
         case ir_op::f64_le: case ir_op::f64_ge:
            try_replace(inst.rr.src1);
            try_replace(inst.rr.src2);
            break;
         // rr format: unary ops, mov, return, arg, br_table, memory_grow, drop
         case ir_op::i32_eqz: case ir_op::i64_eqz:
         case ir_op::i32_clz: case ir_op::i32_ctz: case ir_op::i32_popcnt:
         case ir_op::i64_clz: case ir_op::i64_ctz: case ir_op::i64_popcnt:
         case ir_op::i32_wrap_i64: case ir_op::i64_extend_s_i32: case ir_op::i64_extend_u_i32:
         case ir_op::i32_trunc_s_f32: case ir_op::i32_trunc_u_f32:
         case ir_op::i32_trunc_s_f64: case ir_op::i32_trunc_u_f64:
         case ir_op::i64_trunc_s_f32: case ir_op::i64_trunc_u_f32:
         case ir_op::i64_trunc_s_f64: case ir_op::i64_trunc_u_f64:
         case ir_op::f32_convert_s_i32: case ir_op::f32_convert_u_i32:
         case ir_op::f32_convert_s_i64: case ir_op::f32_convert_u_i64:
         case ir_op::f64_convert_s_i32: case ir_op::f64_convert_u_i32:
         case ir_op::f64_convert_s_i64: case ir_op::f64_convert_u_i64:
         case ir_op::f32_demote_f64: case ir_op::f64_promote_f32:
         case ir_op::i32_reinterpret_f32: case ir_op::i64_reinterpret_f64:
         case ir_op::f32_reinterpret_i32: case ir_op::f64_reinterpret_i64:
         case ir_op::i32_trunc_sat_f32_s: case ir_op::i32_trunc_sat_f32_u:
         case ir_op::i32_trunc_sat_f64_s: case ir_op::i32_trunc_sat_f64_u:
         case ir_op::i64_trunc_sat_f32_s: case ir_op::i64_trunc_sat_f32_u:
         case ir_op::i64_trunc_sat_f64_s: case ir_op::i64_trunc_sat_f64_u:
         case ir_op::i32_extend8_s: case ir_op::i32_extend16_s:
         case ir_op::i64_extend8_s: case ir_op::i64_extend16_s: case ir_op::i64_extend32_s:
         case ir_op::f32_abs: case ir_op::f32_neg: case ir_op::f32_ceil: case ir_op::f32_floor:
         case ir_op::f32_trunc: case ir_op::f32_nearest: case ir_op::f32_sqrt:
         case ir_op::f64_abs: case ir_op::f64_neg: case ir_op::f64_ceil: case ir_op::f64_floor:
         case ir_op::f64_trunc: case ir_op::f64_nearest: case ir_op::f64_sqrt:
         case ir_op::mov: case ir_op::return_: case ir_op::arg:
         case ir_op::br_table: case ir_op::memory_grow: case ir_op::drop:
            try_replace(inst.rr.src1);
            break;
         // ri format: loads (src1 = address vreg)
         case ir_op::i32_load: case ir_op::i64_load: case ir_op::f32_load: case ir_op::f64_load:
         case ir_op::i32_load8_s: case ir_op::i32_load8_u: case ir_op::i32_load16_s: case ir_op::i32_load16_u:
         case ir_op::i64_load8_s: case ir_op::i64_load8_u: case ir_op::i64_load16_s: case ir_op::i64_load16_u:
         case ir_op::i64_load32_s: case ir_op::i64_load32_u:
            try_replace(inst.ri.src1);
            break;
         // ri format: stores (src1 = address, dest = value)
         case ir_op::i32_store: case ir_op::i64_store: case ir_op::f32_store: case ir_op::f64_store:
         case ir_op::i32_store8: case ir_op::i32_store16:
         case ir_op::i64_store8: case ir_op::i64_store16: case ir_op::i64_store32:
            try_replace(inst.ri.src1);
            // dest is value vreg for stores — also replace
            try_replace(inst.dest);
            break;
         // br format: if_, br_if, br
         case ir_op::if_: case ir_op::br_if: case ir_op::br:
            try_replace(inst.br.src1);
            break;
         // call format
         case ir_op::call: case ir_op::call_indirect:
            try_replace(inst.call.src1);
            break;
         // local format: local_set, local_tee, global_set
         case ir_op::local_set: case ir_op::local_tee: case ir_op::global_set:
            try_replace(inst.local.src1);
            break;
         // select: 3 packed 16-bit vregs
         case ir_op::select:
            try_replace16(inst.sel.val1);
            try_replace16(inst.sel.val2);
            try_replace16(inst.sel.cond);
            break;
         default:
            break;
         }
         return found;
      }
   };

}} // namespace eosio::vm
