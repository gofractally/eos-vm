#pragma once

// IR definition for the two-pass optimizing JIT (jit2).
// Each WASM function is lowered to a flat array of ir_inst with virtual registers.
//
// ALL data structures are backed by growable_allocator (mmap).
// No malloc/free anywhere in the hot path.
// All buffers are bounded proportionally to WASM input size.

#include <eosio/vm/allocator.hpp>
#include <eosio/vm/types.hpp>
#include <eosio/vm/vector.hpp>

#include <cstdint>

namespace eosio { namespace vm {

   // IR opcodes. These mirror WASM operations but in register-transfer form.
   enum class ir_op : uint16_t {
      // Meta
      nop,
      unreachable,
      // Constants
      const_i32,
      const_i64,
      const_f32,
      const_f64,
      const_v128,
      // Local/global variable access
      local_get,
      local_set,
      local_tee,
      global_get,
      global_set,
      // Memory loads (dest = load(src1 + offset))
      i32_load, i64_load, f32_load, f64_load,
      i32_load8_s, i32_load8_u, i32_load16_s, i32_load16_u,
      i64_load8_s, i64_load8_u, i64_load16_s, i64_load16_u,
      i64_load32_s, i64_load32_u,
      // Memory stores (void: store src2 to [src1 + offset])
      i32_store, i64_store, f32_store, f64_store,
      i32_store8, i32_store16,
      i64_store8, i64_store16, i64_store32,
      // Memory management
      memory_size, memory_grow,
      // Parametric
      drop,
      select,
      // Comparison (result is i32 0 or 1)
      i32_eqz, i32_eq, i32_ne, i32_lt_s, i32_lt_u, i32_gt_s, i32_gt_u,
      i32_le_s, i32_le_u, i32_ge_s, i32_ge_u,
      i64_eqz, i64_eq, i64_ne, i64_lt_s, i64_lt_u, i64_gt_s, i64_gt_u,
      i64_le_s, i64_le_u, i64_ge_s, i64_ge_u,
      f32_eq, f32_ne, f32_lt, f32_gt, f32_le, f32_ge,
      f64_eq, f64_ne, f64_lt, f64_gt, f64_le, f64_ge,
      // Unary integer
      i32_clz, i32_ctz, i32_popcnt,
      i64_clz, i64_ctz, i64_popcnt,
      // Binary integer
      i32_add, i32_sub, i32_mul, i32_div_s, i32_div_u, i32_rem_s, i32_rem_u,
      i32_and, i32_or, i32_xor, i32_shl, i32_shr_s, i32_shr_u, i32_rotl, i32_rotr,
      i64_add, i64_sub, i64_mul, i64_div_s, i64_div_u, i64_rem_s, i64_rem_u,
      i64_and, i64_or, i64_xor, i64_shl, i64_shr_s, i64_shr_u, i64_rotl, i64_rotr,
      // Unary float
      f32_abs, f32_neg, f32_ceil, f32_floor, f32_trunc, f32_nearest, f32_sqrt,
      f64_abs, f64_neg, f64_ceil, f64_floor, f64_trunc, f64_nearest, f64_sqrt,
      // Binary float
      f32_add, f32_sub, f32_mul, f32_div, f32_min, f32_max, f32_copysign,
      f64_add, f64_sub, f64_mul, f64_div, f64_min, f64_max, f64_copysign,
      // Conversions
      i32_wrap_i64,
      i32_trunc_s_f32, i32_trunc_u_f32, i32_trunc_s_f64, i32_trunc_u_f64,
      i64_extend_s_i32, i64_extend_u_i32,
      i64_trunc_s_f32, i64_trunc_u_f32, i64_trunc_s_f64, i64_trunc_u_f64,
      f32_convert_s_i32, f32_convert_u_i32, f32_convert_s_i64, f32_convert_u_i64,
      f32_demote_f64,
      f64_convert_s_i32, f64_convert_u_i32, f64_convert_s_i64, f64_convert_u_i64,
      f64_promote_f32,
      i32_reinterpret_f32, i64_reinterpret_f64, f32_reinterpret_i32, f64_reinterpret_i64,
      // Saturating truncations
      i32_trunc_sat_f32_s, i32_trunc_sat_f32_u, i32_trunc_sat_f64_s, i32_trunc_sat_f64_u,
      i64_trunc_sat_f32_s, i64_trunc_sat_f32_u, i64_trunc_sat_f64_s, i64_trunc_sat_f64_u,
      // Sign extensions
      i32_extend8_s, i32_extend16_s,
      i64_extend8_s, i64_extend16_s, i64_extend32_s,
      // Control flow
      block,
      loop,
      br,
      br_if,
      br_table,
      if_,
      else_,
      end,
      return_,
      // Calls
      call,
      call_indirect,
      // Move (used at control flow merge points)
      mov,
      // Call argument (pseudo-instruction)
      arg,
      // SIMD (generic with sub-opcode in imm field)
      v128_op,
      // Bulk memory
      memory_init,
      data_drop,
      memory_copy,
      memory_fill,
      table_init,
      elem_drop,
      table_copy,
   };

   static constexpr uint32_t ir_vreg_none = UINT32_MAX;

   enum ir_flags : uint8_t {
      IR_NONE         = 0,
      IR_SIDE_EFFECT  = 1 << 0,
      IR_COMMUTATIVE  = 1 << 1,
      IR_DEAD         = 1 << 2,
   };

   // 16-byte IR instruction. POD type — no constructor/destructor.
   struct ir_inst {
      ir_op    opcode;
      uint8_t  type;
      uint8_t  flags;
      uint32_t dest;
      union {
         struct { uint32_t src1; uint32_t src2; } rr;
         struct { uint32_t src1; int32_t  imm; }  ri;
         int64_t  imm64;
         double   immf64;
         float    immf32;
         struct { uint32_t target; uint32_t src1; } br;
         struct { uint32_t index; uint32_t src1; }  call;
         struct { uint32_t index; uint32_t src1; }  local;
         v128_t   immv128;
      };
   };
   static_assert(sizeof(ir_inst) >= 16);
   static_assert(std::is_trivially_copyable_v<ir_inst>);
   static_assert(std::is_trivially_destructible_v<ir_inst>);

   struct ir_basic_block {
      uint32_t start;
      uint32_t end;
      uint32_t successors[2];
      uint8_t  num_successors;
      uint32_t loop_depth;
   };
   static_assert(std::is_trivially_copyable_v<ir_basic_block>);
   static_assert(std::is_trivially_destructible_v<ir_basic_block>);

   // Live interval for register allocation. POD.
   struct ir_live_interval {
      uint32_t vreg;
      uint32_t start;
      uint32_t end;
      int8_t   phys_reg;
      int8_t   phys_xmm;
      int16_t  spill_slot;
      uint8_t  type;
   };
   static_assert(std::is_trivially_copyable_v<ir_live_interval>);

   // Control stack entry for IR construction. POD.
   struct ir_control_entry {
      uint32_t block_idx;
      uint32_t stack_depth;
      uint8_t  result_type;
      uint8_t  is_loop;
      uint8_t  is_function;
      uint8_t  _pad;
      uint32_t merge_block;
   };
   static_assert(std::is_trivially_copyable_v<ir_control_entry>);

   // Per-function IR representation.
   // All buffers are backed by growable_allocator — no malloc/free.
   // Capacity is bounded by source_bytes at construction.
   struct ir_function {
      // Raw pointer + count arrays (allocated from growable_allocator)
      ir_inst*        insts       = nullptr;
      uint32_t        inst_count  = 0;
      uint32_t        inst_cap    = 0;

      ir_basic_block* blocks      = nullptr;
      uint32_t        block_count = 0;
      uint32_t        block_cap   = 0;

      // Virtual operand stack (bounded by source_bytes / min_instruction_size)
      uint32_t*       vstack      = nullptr;
      uint32_t        vstack_top  = 0;
      uint32_t        vstack_cap  = 0;

      // Control flow stack (bounded by max nesting depth)
      ir_control_entry* ctrl_stack     = nullptr;
      uint32_t          ctrl_stack_top = 0;
      uint32_t          ctrl_stack_cap = 0;

      // Live intervals (allocated lazily during regalloc, count = next_vreg)
      ir_live_interval* intervals      = nullptr;
      uint32_t          interval_count = 0;

      uint32_t        next_vreg   = 0;
      uint32_t        num_params  = 0;
      uint32_t        num_locals  = 0;
      uint32_t        func_index  = 0;
      const func_type* type       = nullptr;
      uint32_t        num_spill_slots = 0;

      // Initialize with bounded capacity from growable_allocator.
      // source_bytes = size of this function's WASM bytecode.
      // All allocations come from alloc, none from malloc.
      void init(growable_allocator& alloc, std::size_t source_bytes) {
         // Each WASM byte produces at most 3 IR instructions (e.g. store = addr+store+arg).
         // Add a minimum to handle tiny functions.
         inst_cap = static_cast<uint32_t>(source_bytes * 3 + 16);
         insts = alloc.alloc<ir_inst>(inst_cap);
         inst_count = 0;

         // Each control flow instruction creates at most 2 blocks.
         // source_bytes is an upper bound on instruction count.
         block_cap = static_cast<uint32_t>(source_bytes + 4);
         blocks = alloc.alloc<ir_basic_block>(block_cap);
         block_count = 0;

         // Virtual stack is bounded by source_bytes (each push needs at least 1 byte of WASM).
         vstack_cap = static_cast<uint32_t>(source_bytes + 4);
         vstack = alloc.alloc<uint32_t>(vstack_cap);
         vstack_top = 0;

         // Control stack is bounded by nesting depth. source_bytes is a safe upper bound.
         // In practice, WASM validators enforce max nesting depth.
         ctrl_stack_cap = static_cast<uint32_t>(source_bytes + 4);
         ctrl_stack = alloc.alloc<ir_control_entry>(ctrl_stack_cap);
         ctrl_stack_top = 0;

         next_vreg = 0;
         intervals = nullptr;
         interval_count = 0;
         num_spill_slots = 0;
      }

      // Release all buffers back to the allocator.
      // Must be called when this function's IR is no longer needed (after codegen).
      void release(growable_allocator& alloc) {
         alloc.reclaim(ctrl_stack, ctrl_stack_cap);
         alloc.reclaim(vstack, vstack_cap);
         alloc.reclaim(blocks, block_cap);
         alloc.reclaim(insts, inst_cap);
         insts = nullptr;
         inst_count = 0;
      }

      uint32_t alloc_vreg(uint8_t /*ty*/) {
         return next_vreg++;
      }

      void emit(ir_inst inst) {
         EOS_VM_ASSERT(inst_count < inst_cap, wasm_parse_exception, "IR instruction buffer overflow");
         insts[inst_count++] = inst;
      }

      uint32_t current_inst_index() const {
         return inst_count;
      }

      uint32_t new_block() {
         EOS_VM_ASSERT(block_count < block_cap, wasm_parse_exception, "IR block buffer overflow");
         uint32_t idx = block_count++;
         auto& b = blocks[idx];
         b.start = 0;
         b.end = 0;
         b.successors[0] = b.successors[1] = UINT32_MAX;
         b.num_successors = 0;
         b.loop_depth = 0;
         return idx;
      }

      void start_block(uint32_t block_idx) {
         if (block_idx < block_count) {
            blocks[block_idx].start = current_inst_index();
         }
      }

      void end_block(uint32_t block_idx) {
         if (block_idx < block_count) {
            blocks[block_idx].end = current_inst_index();
         }
      }

      // Virtual operand stack operations (bounded, no malloc)
      void vpush(uint32_t vreg) {
         EOS_VM_ASSERT(vstack_top < vstack_cap, wasm_parse_exception, "IR virtual stack overflow");
         vstack[vstack_top++] = vreg;
      }

      uint32_t vpop() {
         EOS_VM_ASSERT(vstack_top > 0, wasm_parse_exception, "IR virtual stack underflow");
         return vstack[--vstack_top];
      }

      void vstack_resize(uint32_t depth) {
         EOS_VM_ASSERT(depth <= vstack_top, wasm_parse_exception, "IR virtual stack resize underflow");
         vstack_top = depth;
      }

      uint32_t vstack_depth() const { return vstack_top; }
      uint32_t vstack_back() const {
         EOS_VM_ASSERT(vstack_top > 0, wasm_parse_exception, "IR virtual stack empty");
         return vstack[vstack_top - 1];
      }

      // Control flow stack operations (bounded, no malloc)
      void ctrl_push(ir_control_entry entry) {
         EOS_VM_ASSERT(ctrl_stack_top < ctrl_stack_cap, wasm_parse_exception, "IR control stack overflow");
         ctrl_stack[ctrl_stack_top++] = entry;
      }

      ir_control_entry ctrl_pop() {
         EOS_VM_ASSERT(ctrl_stack_top > 0, wasm_parse_exception, "IR control stack underflow");
         return ctrl_stack[--ctrl_stack_top];
      }

      ir_control_entry& ctrl_back() {
         EOS_VM_ASSERT(ctrl_stack_top > 0, wasm_parse_exception, "IR control stack empty");
         return ctrl_stack[ctrl_stack_top - 1];
      }

      ir_control_entry& ctrl_at(uint32_t depth) {
         EOS_VM_ASSERT(depth < ctrl_stack_top, wasm_parse_exception, "IR control stack depth out of range");
         return ctrl_stack[ctrl_stack_top - 1 - depth];
      }
   };

}} // namespace eosio::vm
