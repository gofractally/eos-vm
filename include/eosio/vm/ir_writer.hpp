#pragma once

// Pass 1 of the two-pass optimizing JIT (jit2).
// Converts WASM stack machine operations to virtual-register IR.
// Implements the same writer visitor interface as machine_code_writer.

#include <eosio/vm/allocator.hpp>
#include <eosio/vm/exceptions.hpp>
#include <eosio/vm/jit_codegen.hpp>
#include <eosio/vm/jit_ir.hpp>
#include <eosio/vm/types.hpp>

#include <cstdint>

namespace eosio { namespace vm {

   template<typename Context, bool StackLimitIsBytes>
   class ir_writer {
    public:
      // Types required by the parser
      using branch_t = uint32_t;  // index into fixup list
      using label_t  = uint32_t;  // block index

      ir_writer(growable_allocator& alloc, std::size_t source_bytes, module& mod)
         : _allocator(alloc), _source_bytes(source_bytes), _mod(mod),
           _codegen(alloc, mod) {
         _codegen.emit_entry_and_error_handlers();
      }

      ~ir_writer() {
         _codegen.finalize_code();
      }

      static constexpr uint32_t get_depth_for_type(uint8_t type) {
         return (type == types::v128) ? 2 : (type == types::pseudo ? 0 : 1);
      }

      // ──────── Function prologue/epilogue ────────

      void emit_prologue(const func_type& ft, const std::vector<local_entry>& locals, uint32_t funcnum) {
         _func = ir_function{};
         _func.init(_allocator, _source_bytes);
         _func.func_index = funcnum;
         _func.type = &ft;
         _func.num_params = static_cast<uint32_t>(ft.param_types.size());

         // Count total locals
         uint32_t local_count = 0;
         for (auto& le : locals) local_count += le.count;
         _func.num_locals = local_count;

         // Start first basic block
         uint32_t entry_block = _func.new_block();
         _func.start_block(entry_block);
         _current_block = entry_block;

         // Push implicit function-level block
         _func.ctrl_push({
            entry_block,
            0,                          // stack depth at entry
            ft.return_count ? ft.return_type : types::pseudo,
            0,                          // is_loop
            1,                          // is_function
            0,                          // _pad
            UINT32_MAX,                 // merge block
         });
      }

      void emit_epilogue(const func_type& ft, const std::vector<local_entry>& locals, uint32_t funcnum) {
         // The final end should have handled this
      }

      void finalize(function_body& body) {
         // Pass 2: compile this function's IR to x86_64, then release IR
         _codegen.compile_function(_func, body);
         _func.release(_allocator);
      }

      const void* get_addr() const { return nullptr; }
      const void* get_base_addr() const { return nullptr; }
      void set_stack_usage(std::uint64_t) {}

      // ──────── Control flow ────────

      void emit_unreachable() {
         emit_void(ir_op::unreachable);
      }

      void emit_nop() {}

      void emit_block() {
         uint32_t merge_block = _func.new_block();
         _func.ctrl_push({
            _current_block,
            _func.vstack_depth(),
            types::pseudo,
            0, 0, 0,
            merge_block,
         });
      }

      label_t emit_loop() {
         // End current block
         _func.end_block(_current_block);

         // Create loop header block
         uint32_t header = _func.new_block();
         _func.start_block(header);

         // Emit unconditional branch from previous block to header
         // (fallthrough)
         emit_br_to(header);

         _current_block = header;

         _func.ctrl_push({
            header,
            _func.vstack_depth(),
            types::pseudo,
            1, 0, 0,       // is_loop = true
            UINT32_MAX,
         });

         return header;
      }

      branch_t emit_if() {
         uint32_t cond = _func.vpop();

         // End current block
         _func.end_block(_current_block);

         // Create true block and merge block
         uint32_t true_block = _func.new_block();
         uint32_t merge_block = _func.new_block();

         // Emit conditional branch: if cond == 0, go to merge (or else)
         ir_inst inst{};
         inst.opcode = ir_op::br_if;
         inst.type = types::pseudo;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = ir_vreg_none;
         inst.br.target = merge_block; // patched to else if there's an else
         inst.br.src1 = cond;
         _func.emit(inst);

         _func.start_block(true_block);
         _current_block = true_block;

         _func.ctrl_push({
            _current_block,
            _func.vstack_depth(),
            types::pseudo,
            0, 0, 0,
            merge_block,
         });

         // Return index of the br_if instruction for fixup
         return static_cast<uint32_t>(_func.inst_count - 1);
      }

      branch_t emit_else(branch_t if_loc) {
         // End true block with branch to merge
         _func.end_block(_current_block);
         auto& ctrl = _func.ctrl_back();
         emit_br_to(ctrl.merge_block);

         // Create else block
         uint32_t else_block = _func.new_block();
         _func.start_block(else_block);
         _current_block = else_block;

         // Fix the if's branch to point to else block instead of merge
         if (if_loc < _func.inst_count) {
            _func.insts[if_loc].br.target = else_block;
         }

         // Restore stack to entry depth
         _func.vstack_resize(ctrl.stack_depth);

         return if_loc; // pass through for fix_branch at end
      }

      label_t emit_end() {
         _func.end_block(_current_block);

         auto ctrl = _func.ctrl_pop();

         if (ctrl.merge_block != UINT32_MAX) {
            _func.start_block(ctrl.merge_block);
            _current_block = ctrl.merge_block;
         }

         // Restore stack to entry depth + result
         if (ctrl.result_type != types::pseudo && _func.vstack_depth() > ctrl.stack_depth) {
            uint32_t result = _func.vstack_back();
            _func.vstack_resize(ctrl.stack_depth);
            _func.vpush(result);
         } else {
            _func.vstack_resize(ctrl.stack_depth);
         }

         return _current_block;
      }

      branch_t emit_return(uint32_t depth_change, uint8_t rt) {
         if (rt != types::pseudo && _func.vstack_depth() > 0) {
            // Move return value
            uint32_t val = _func.vpop();
            ir_inst inst{};
            inst.opcode = ir_op::return_;
            inst.type = types::pseudo;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = ir_vreg_none;
            inst.rr.src1 = val;
            inst.rr.src2 = ir_vreg_none;
            _func.emit(inst);
         } else {
            emit_void(ir_op::return_);
         }
         return 0;
      }

      branch_t emit_br(uint32_t depth_change, uint8_t rt) {
         auto& target_ctrl = _func.ctrl_at(get_branch_depth(depth_change, rt));

         if (rt != types::pseudo && _func.vstack_depth() > 0) {
            uint32_t val = _func.vpop();
            ir_inst inst{};
            inst.opcode = ir_op::br;
            inst.type = types::pseudo;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = ir_vreg_none;
            inst.br.target = get_br_target(target_ctrl);
            inst.br.src1 = val;
            _func.emit(inst);
         } else {
            ir_inst inst{};
            inst.opcode = ir_op::br;
            inst.type = types::pseudo;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = ir_vreg_none;
            inst.br.target = get_br_target(target_ctrl);
            inst.br.src1 = ir_vreg_none;
            _func.emit(inst);
         }

         _func.end_block(_current_block);
         uint32_t new_blk = _func.new_block();
         _func.start_block(new_blk);
         _current_block = new_blk;

         return 0;
      }

      branch_t emit_br_if(uint32_t depth_change, uint8_t rt) {
         uint32_t cond = _func.vpop();
         auto& target_ctrl = _func.ctrl_at(get_branch_depth(depth_change, rt));

         ir_inst inst{};
         inst.opcode = ir_op::br_if;
         inst.type = types::pseudo;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = ir_vreg_none;
         inst.br.target = get_br_target(target_ctrl);
         inst.br.src1 = cond;
         _func.emit(inst);

         // Continue in a new block (fallthrough)
         _func.end_block(_current_block);
         uint32_t new_blk = _func.new_block();
         _func.start_block(new_blk);
         _current_block = new_blk;

         return 0;
      }

      struct br_table_parser {
         ir_writer& writer;
         uint32_t   index_vreg;

         branch_t emit_case(uint32_t depth_change, uint8_t rt) {
            auto& target_ctrl = writer._func.ctrl_at(writer.get_branch_depth(depth_change, rt));
            ir_inst inst{};
            inst.opcode = ir_op::br_table;
            inst.type = types::pseudo;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = ir_vreg_none;
            inst.br.target = writer.get_br_target(target_ctrl);
            inst.br.src1 = index_vreg;
            writer._func.emit(inst);
            return 0;
         }

         branch_t emit_default(uint32_t depth_change, uint8_t rt) {
            return emit_case(depth_change, rt);
         }
      };

      br_table_parser emit_br_table(uint32_t table_size) {
         uint32_t idx = _func.vpop();
         return br_table_parser{*this, idx};
      }

      // ──────── Calls ────────

      void emit_call(const func_type& ft, uint32_t funcnum) {
         uint32_t n_args = static_cast<uint32_t>(ft.param_types.size());
         // Args are on vstack in order: vstack[top-n_args..top-1]
         // Read them directly from the vstack before popping.
         uint32_t base = _func.vstack_depth() - n_args;
         for (uint32_t i = 0; i < n_args; ++i) {
            ir_inst a{};
            a.opcode = ir_op::arg;
            a.type = ft.param_types[i];
            a.flags = IR_NONE;
            a.dest = ir_vreg_none;
            a.rr.src1 = _func.vstack[base + i];
            a.rr.src2 = i;
            _func.emit(a);
         }
         _func.vstack_resize(base); // pop all args at once

         ir_inst inst{};
         inst.opcode = ir_op::call;
         inst.flags = IR_SIDE_EFFECT;
         inst.call.index = funcnum;
         inst.call.src1 = n_args;

         if (ft.return_count) {
            inst.type = ft.return_type;
            uint32_t result = _func.alloc_vreg(ft.return_type);
            inst.dest = result;
            _func.emit(inst);
            _func.vpush(result);
         } else {
            inst.type = types::pseudo;
            inst.dest = ir_vreg_none;
            _func.emit(inst);
         }
      }

      void emit_call_indirect(const func_type& ft, uint32_t functypeidx) {
         uint32_t table_idx = _func.vpop();  // table index is on top

         uint32_t n_args = static_cast<uint32_t>(ft.param_types.size());
         uint32_t base = _func.vstack_depth() - n_args;
         for (uint32_t i = 0; i < n_args; ++i) {
            ir_inst a{};
            a.opcode = ir_op::arg;
            a.type = ft.param_types[i];
            a.flags = IR_NONE;
            a.dest = ir_vreg_none;
            a.rr.src1 = _func.vstack[base + i];
            a.rr.src2 = i;
            _func.emit(a);
         }
         _func.vstack_resize(base);

         ir_inst inst{};
         inst.opcode = ir_op::call_indirect;
         inst.flags = IR_SIDE_EFFECT;
         inst.call.index = functypeidx;
         inst.call.src1 = table_idx;

         if (ft.return_count) {
            inst.type = ft.return_type;
            uint32_t result = _func.alloc_vreg(ft.return_type);
            inst.dest = result;
            _func.emit(inst);
            _func.vpush(result);
         } else {
            inst.type = types::pseudo;
            inst.dest = ir_vreg_none;
            _func.emit(inst);
         }
      }

      // ──────── Parametric ────────

      void emit_drop(uint8_t type) {
         if (_func.vstack_depth() > 0) _func.vpop();
      }

      void emit_select(uint8_t type) {
         uint32_t cond = _func.vpop();
         uint32_t val2 = _func.vpop();
         uint32_t val1 = _func.vpop();
         uint32_t result = emit_ternary(ir_op::select, type, val1, val2, cond);
         _func.vpush(result);
      }

      // ──────── Local/global access ────────

      void emit_get_local(uint32_t localidx, uint8_t type) {
         uint32_t dest = _func.alloc_vreg(type);
         ir_inst inst{};
         inst.opcode = ir_op::local_get;
         inst.type = type;
         inst.flags = IR_NONE;
         inst.dest = dest;
         inst.local.index = localidx;
         inst.local.src1 = ir_vreg_none;
         _func.emit(inst);
         _func.vpush(dest);
      }

      void emit_set_local(uint32_t localidx, uint8_t type) {
         uint32_t val = _func.vpop();
         ir_inst inst{};
         inst.opcode = ir_op::local_set;
         inst.type = types::pseudo;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = ir_vreg_none;
         inst.local.index = localidx;
         inst.local.src1 = val;
         _func.emit(inst);
      }

      void emit_tee_local(uint32_t localidx, uint8_t type) {
         uint32_t val = _func.vpop();
         ir_inst inst{};
         inst.opcode = ir_op::local_tee;
         inst.type = type;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = val; // tee keeps the same vreg on stack
         inst.local.index = localidx;
         inst.local.src1 = val;
         _func.emit(inst);
         _func.vpush(val);
      }

      void emit_get_global(uint32_t globalidx) {
         uint32_t dest = _func.alloc_vreg(types::i64); // globals are 64-bit
         ir_inst inst{};
         inst.opcode = ir_op::global_get;
         inst.type = types::i64;
         inst.flags = IR_NONE;
         inst.dest = dest;
         inst.local.index = globalidx;
         inst.local.src1 = ir_vreg_none;
         _func.emit(inst);
         _func.vpush(dest);
      }

      void emit_set_global(uint32_t globalidx) {
         uint32_t val = _func.vpop();
         ir_inst inst{};
         inst.opcode = ir_op::global_set;
         inst.type = types::pseudo;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = ir_vreg_none;
         inst.local.index = globalidx;
         inst.local.src1 = val;
         _func.emit(inst);
      }

      // ──────── Memory loads ────────
#define EMIT_LOAD(name, op, ty) \
      void name(uint32_t offset, uint32_t alignment) { \
         uint32_t addr = _func.vpop(); \
         uint32_t dest = _func.alloc_vreg(ty); \
         ir_inst inst{}; \
         inst.opcode = ir_op::op; \
         inst.type = ty; \
         inst.flags = IR_NONE; \
         inst.dest = dest; \
         inst.ri.src1 = addr; \
         inst.ri.imm = static_cast<int32_t>(offset); \
         _func.emit(inst); \
         _func.vpush(dest); \
      }

      EMIT_LOAD(emit_i32_load,     i32_load,     types::i32)
      EMIT_LOAD(emit_i64_load,     i64_load,     types::i64)
      EMIT_LOAD(emit_f32_load,     f32_load,     types::f32)
      EMIT_LOAD(emit_f64_load,     f64_load,     types::f64)
      EMIT_LOAD(emit_i32_load8_s,  i32_load8_s,  types::i32)
      EMIT_LOAD(emit_i32_load16_s, i32_load16_s, types::i32)
      EMIT_LOAD(emit_i32_load8_u,  i32_load8_u,  types::i32)
      EMIT_LOAD(emit_i32_load16_u, i32_load16_u, types::i32)
      EMIT_LOAD(emit_i64_load8_s,  i64_load8_s,  types::i64)
      EMIT_LOAD(emit_i64_load16_s, i64_load16_s, types::i64)
      EMIT_LOAD(emit_i64_load32_s, i64_load32_s, types::i64)
      EMIT_LOAD(emit_i64_load8_u,  i64_load8_u,  types::i64)
      EMIT_LOAD(emit_i64_load16_u, i64_load16_u, types::i64)
      EMIT_LOAD(emit_i64_load32_u, i64_load32_u, types::i64)
#undef EMIT_LOAD

      // ──────── Memory stores ────────
#define EMIT_STORE(name, op) \
      void name(uint32_t offset, uint32_t alignment) { \
         uint32_t val = _func.vpop(); \
         uint32_t addr = _func.vpop(); \
         ir_inst inst{}; \
         inst.opcode = ir_op::op; \
         inst.type = types::pseudo; \
         inst.flags = IR_SIDE_EFFECT; \
         inst.dest = ir_vreg_none; \
         inst.ri.src1 = addr; \
         inst.ri.imm = static_cast<int32_t>(offset); \
         _func.emit(inst); \
         /* val is stored as an extra source - encode in a following arg inst */ \
         ir_inst arg{}; \
         arg.opcode = ir_op::arg; \
         arg.type = types::pseudo; \
         arg.flags = IR_NONE; \
         arg.dest = ir_vreg_none; \
         arg.rr.src1 = val; \
         arg.rr.src2 = ir_vreg_none; \
         _func.emit(arg); \
      }

      EMIT_STORE(emit_i32_store,   i32_store)
      EMIT_STORE(emit_i64_store,   i64_store)
      EMIT_STORE(emit_f32_store,   f32_store)
      EMIT_STORE(emit_f64_store,   f64_store)
      EMIT_STORE(emit_i32_store8,  i32_store8)
      EMIT_STORE(emit_i32_store16, i32_store16)
      EMIT_STORE(emit_i64_store8,  i64_store8)
      EMIT_STORE(emit_i64_store16, i64_store16)
      EMIT_STORE(emit_i64_store32, i64_store32)
#undef EMIT_STORE

      // ──────── Memory management ────────
      void emit_current_memory() { _func.vpush(emit_nullary(ir_op::memory_size, types::i32)); }
      void emit_grow_memory() {
         uint32_t pages = _func.vpop();
         uint32_t dest = _func.alloc_vreg(types::i32);
         ir_inst inst{};
         inst.opcode = ir_op::memory_grow;
         inst.type = types::i32;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = dest;
         inst.rr.src1 = pages;
         inst.rr.src2 = ir_vreg_none;
         _func.emit(inst);
         _func.vpush(dest);
      }

      // ──────── Constants ────────
      void emit_i32_const(uint32_t value) {
         uint32_t dest = _func.alloc_vreg(types::i32);
         ir_inst inst{};
         inst.opcode = ir_op::const_i32;
         inst.type = types::i32;
         inst.flags = IR_NONE;
         inst.dest = dest;
         inst.imm64 = static_cast<int64_t>(static_cast<int32_t>(value));
         _func.emit(inst);
         _func.vpush(dest);
      }

      void emit_i64_const(uint64_t value) {
         uint32_t dest = _func.alloc_vreg(types::i64);
         ir_inst inst{};
         inst.opcode = ir_op::const_i64;
         inst.type = types::i64;
         inst.flags = IR_NONE;
         inst.dest = dest;
         inst.imm64 = static_cast<int64_t>(value);
         _func.emit(inst);
         _func.vpush(dest);
      }

      void emit_f32_const(float value) {
         uint32_t dest = _func.alloc_vreg(types::f32);
         ir_inst inst{};
         inst.opcode = ir_op::const_f32;
         inst.type = types::f32;
         inst.flags = IR_NONE;
         inst.dest = dest;
         inst.immf32 = value;
         _func.emit(inst);
         _func.vpush(dest);
      }

      void emit_f64_const(double value) {
         uint32_t dest = _func.alloc_vreg(types::f64);
         ir_inst inst{};
         inst.opcode = ir_op::const_f64;
         inst.type = types::f64;
         inst.flags = IR_NONE;
         inst.dest = dest;
         inst.immf64 = value;
         _func.emit(inst);
         _func.vpush(dest);
      }

      // ──────── Unary operations ────────
#define EMIT_UNARY(name, op, in_ty, out_ty) \
      void name() { \
         uint32_t src = _func.vpop(); \
         _func.vpush(emit_unary_inst(ir_op::op, out_ty, src)); \
      }

      // i32 comparisons (eqz is unary)
      EMIT_UNARY(emit_i32_eqz, i32_eqz, types::i32, types::i32)
      EMIT_UNARY(emit_i64_eqz, i64_eqz, types::i64, types::i32)
      // Unary integer
      EMIT_UNARY(emit_i32_clz,    i32_clz,    types::i32, types::i32)
      EMIT_UNARY(emit_i32_ctz,    i32_ctz,    types::i32, types::i32)
      EMIT_UNARY(emit_i32_popcnt, i32_popcnt, types::i32, types::i32)
      EMIT_UNARY(emit_i64_clz,    i64_clz,    types::i64, types::i64)
      EMIT_UNARY(emit_i64_ctz,    i64_ctz,    types::i64, types::i64)
      EMIT_UNARY(emit_i64_popcnt, i64_popcnt, types::i64, types::i64)
      // Unary float
      EMIT_UNARY(emit_f32_abs,     f32_abs,     types::f32, types::f32)
      EMIT_UNARY(emit_f32_neg,     f32_neg,     types::f32, types::f32)
      EMIT_UNARY(emit_f32_ceil,    f32_ceil,    types::f32, types::f32)
      EMIT_UNARY(emit_f32_floor,   f32_floor,   types::f32, types::f32)
      EMIT_UNARY(emit_f32_trunc,   f32_trunc,   types::f32, types::f32)
      EMIT_UNARY(emit_f32_nearest, f32_nearest, types::f32, types::f32)
      EMIT_UNARY(emit_f32_sqrt,    f32_sqrt,    types::f32, types::f32)
      EMIT_UNARY(emit_f64_abs,     f64_abs,     types::f64, types::f64)
      EMIT_UNARY(emit_f64_neg,     f64_neg,     types::f64, types::f64)
      EMIT_UNARY(emit_f64_ceil,    f64_ceil,    types::f64, types::f64)
      EMIT_UNARY(emit_f64_floor,   f64_floor,   types::f64, types::f64)
      EMIT_UNARY(emit_f64_trunc,   f64_trunc,   types::f64, types::f64)
      EMIT_UNARY(emit_f64_nearest, f64_nearest, types::f64, types::f64)
      EMIT_UNARY(emit_f64_sqrt,    f64_sqrt,    types::f64, types::f64)
      // Conversions
      EMIT_UNARY(emit_i32_wrap_i64,       i32_wrap_i64,       types::i64, types::i32)
      EMIT_UNARY(emit_i32_trunc_s_f32,    i32_trunc_s_f32,    types::f32, types::i32)
      EMIT_UNARY(emit_i32_trunc_u_f32,    i32_trunc_u_f32,    types::f32, types::i32)
      EMIT_UNARY(emit_i32_trunc_s_f64,    i32_trunc_s_f64,    types::f64, types::i32)
      EMIT_UNARY(emit_i32_trunc_u_f64,    i32_trunc_u_f64,    types::f64, types::i32)
      EMIT_UNARY(emit_i64_extend_s_i32,   i64_extend_s_i32,   types::i32, types::i64)
      EMIT_UNARY(emit_i64_extend_u_i32,   i64_extend_u_i32,   types::i32, types::i64)
      EMIT_UNARY(emit_i64_trunc_s_f32,    i64_trunc_s_f32,    types::f32, types::i64)
      EMIT_UNARY(emit_i64_trunc_u_f32,    i64_trunc_u_f32,    types::f32, types::i64)
      EMIT_UNARY(emit_i64_trunc_s_f64,    i64_trunc_s_f64,    types::f64, types::i64)
      EMIT_UNARY(emit_i64_trunc_u_f64,    i64_trunc_u_f64,    types::f64, types::i64)
      EMIT_UNARY(emit_f32_convert_s_i32,  f32_convert_s_i32,  types::i32, types::f32)
      EMIT_UNARY(emit_f32_convert_u_i32,  f32_convert_u_i32,  types::i32, types::f32)
      EMIT_UNARY(emit_f32_convert_s_i64,  f32_convert_s_i64,  types::i64, types::f32)
      EMIT_UNARY(emit_f32_convert_u_i64,  f32_convert_u_i64,  types::i64, types::f32)
      EMIT_UNARY(emit_f32_demote_f64,     f32_demote_f64,     types::f64, types::f32)
      EMIT_UNARY(emit_f64_convert_s_i32,  f64_convert_s_i32,  types::i32, types::f64)
      EMIT_UNARY(emit_f64_convert_u_i32,  f64_convert_u_i32,  types::i32, types::f64)
      EMIT_UNARY(emit_f64_convert_s_i64,  f64_convert_s_i64,  types::i64, types::f64)
      EMIT_UNARY(emit_f64_convert_u_i64,  f64_convert_u_i64,  types::i64, types::f64)
      EMIT_UNARY(emit_f64_promote_f32,    f64_promote_f32,    types::f32, types::f64)
      EMIT_UNARY(emit_i32_reinterpret_f32, i32_reinterpret_f32, types::f32, types::i32)
      EMIT_UNARY(emit_i64_reinterpret_f64, i64_reinterpret_f64, types::f64, types::i64)
      EMIT_UNARY(emit_f32_reinterpret_i32, f32_reinterpret_i32, types::i32, types::f32)
      EMIT_UNARY(emit_f64_reinterpret_i64, f64_reinterpret_i64, types::i64, types::f64)
      // Saturating truncations
      EMIT_UNARY(emit_i32_trunc_sat_f32_s, i32_trunc_sat_f32_s, types::f32, types::i32)
      EMIT_UNARY(emit_i32_trunc_sat_f32_u, i32_trunc_sat_f32_u, types::f32, types::i32)
      EMIT_UNARY(emit_i32_trunc_sat_f64_s, i32_trunc_sat_f64_s, types::f64, types::i32)
      EMIT_UNARY(emit_i32_trunc_sat_f64_u, i32_trunc_sat_f64_u, types::f64, types::i32)
      EMIT_UNARY(emit_i64_trunc_sat_f32_s, i64_trunc_sat_f32_s, types::f32, types::i64)
      EMIT_UNARY(emit_i64_trunc_sat_f32_u, i64_trunc_sat_f32_u, types::f32, types::i64)
      EMIT_UNARY(emit_i64_trunc_sat_f64_s, i64_trunc_sat_f64_s, types::f64, types::i64)
      EMIT_UNARY(emit_i64_trunc_sat_f64_u, i64_trunc_sat_f64_u, types::f64, types::i64)
      // Sign extensions
      EMIT_UNARY(emit_i32_extend8_s,  i32_extend8_s,  types::i32, types::i32)
      EMIT_UNARY(emit_i32_extend16_s, i32_extend16_s, types::i32, types::i32)
      EMIT_UNARY(emit_i64_extend8_s,  i64_extend8_s,  types::i64, types::i64)
      EMIT_UNARY(emit_i64_extend16_s, i64_extend16_s, types::i64, types::i64)
      EMIT_UNARY(emit_i64_extend32_s, i64_extend32_s, types::i64, types::i64)
#undef EMIT_UNARY

      // ──────── Binary operations ────────
#define EMIT_BINARY(name, op, ty) \
      void name() { \
         uint32_t rhs = _func.vpop(); \
         uint32_t lhs = _func.vpop(); \
         _func.vpush(emit_binary_inst(ir_op::op, ty, lhs, rhs)); \
      }
#define EMIT_BINARY_COMM(name, op, ty) \
      void name() { \
         uint32_t rhs = _func.vpop(); \
         uint32_t lhs = _func.vpop(); \
         _func.vpush(emit_binary_inst(ir_op::op, ty, lhs, rhs, IR_COMMUTATIVE)); \
      }

      // i32 comparisons (binary)
      EMIT_BINARY(emit_i32_eq,   i32_eq,   types::i32)
      EMIT_BINARY(emit_i32_ne,   i32_ne,   types::i32)
      EMIT_BINARY(emit_i32_lt_s, i32_lt_s, types::i32)
      EMIT_BINARY(emit_i32_lt_u, i32_lt_u, types::i32)
      EMIT_BINARY(emit_i32_gt_s, i32_gt_s, types::i32)
      EMIT_BINARY(emit_i32_gt_u, i32_gt_u, types::i32)
      EMIT_BINARY(emit_i32_le_s, i32_le_s, types::i32)
      EMIT_BINARY(emit_i32_le_u, i32_le_u, types::i32)
      EMIT_BINARY(emit_i32_ge_s, i32_ge_s, types::i32)
      EMIT_BINARY(emit_i32_ge_u, i32_ge_u, types::i32)
      // i64 comparisons
      EMIT_BINARY(emit_i64_eq,   i64_eq,   types::i32)
      EMIT_BINARY(emit_i64_ne,   i64_ne,   types::i32)
      EMIT_BINARY(emit_i64_lt_s, i64_lt_s, types::i32)
      EMIT_BINARY(emit_i64_lt_u, i64_lt_u, types::i32)
      EMIT_BINARY(emit_i64_gt_s, i64_gt_s, types::i32)
      EMIT_BINARY(emit_i64_gt_u, i64_gt_u, types::i32)
      EMIT_BINARY(emit_i64_le_s, i64_le_s, types::i32)
      EMIT_BINARY(emit_i64_le_u, i64_le_u, types::i32)
      EMIT_BINARY(emit_i64_ge_s, i64_ge_s, types::i32)
      EMIT_BINARY(emit_i64_ge_u, i64_ge_u, types::i32)
      // f32 comparisons
      EMIT_BINARY(emit_f32_eq, f32_eq, types::i32)
      EMIT_BINARY(emit_f32_ne, f32_ne, types::i32)
      EMIT_BINARY(emit_f32_lt, f32_lt, types::i32)
      EMIT_BINARY(emit_f32_gt, f32_gt, types::i32)
      EMIT_BINARY(emit_f32_le, f32_le, types::i32)
      EMIT_BINARY(emit_f32_ge, f32_ge, types::i32)
      // f64 comparisons
      EMIT_BINARY(emit_f64_eq, f64_eq, types::i32)
      EMIT_BINARY(emit_f64_ne, f64_ne, types::i32)
      EMIT_BINARY(emit_f64_lt, f64_lt, types::i32)
      EMIT_BINARY(emit_f64_gt, f64_gt, types::i32)
      EMIT_BINARY(emit_f64_le, f64_le, types::i32)
      EMIT_BINARY(emit_f64_ge, f64_ge, types::i32)
      // i32 arithmetic
      EMIT_BINARY_COMM(emit_i32_add, i32_add, types::i32)
      EMIT_BINARY(emit_i32_sub,      i32_sub, types::i32)
      EMIT_BINARY_COMM(emit_i32_mul, i32_mul, types::i32)
      EMIT_BINARY(emit_i32_div_s,    i32_div_s, types::i32)
      EMIT_BINARY(emit_i32_div_u,    i32_div_u, types::i32)
      EMIT_BINARY(emit_i32_rem_s,    i32_rem_s, types::i32)
      EMIT_BINARY(emit_i32_rem_u,    i32_rem_u, types::i32)
      EMIT_BINARY_COMM(emit_i32_and, i32_and, types::i32)
      EMIT_BINARY_COMM(emit_i32_or,  i32_or,  types::i32)
      EMIT_BINARY_COMM(emit_i32_xor, i32_xor, types::i32)
      EMIT_BINARY(emit_i32_shl,      i32_shl, types::i32)
      EMIT_BINARY(emit_i32_shr_s,    i32_shr_s, types::i32)
      EMIT_BINARY(emit_i32_shr_u,    i32_shr_u, types::i32)
      EMIT_BINARY(emit_i32_rotl,     i32_rotl, types::i32)
      EMIT_BINARY(emit_i32_rotr,     i32_rotr, types::i32)
      // i64 arithmetic
      EMIT_BINARY_COMM(emit_i64_add, i64_add, types::i64)
      EMIT_BINARY(emit_i64_sub,      i64_sub, types::i64)
      EMIT_BINARY_COMM(emit_i64_mul, i64_mul, types::i64)
      EMIT_BINARY(emit_i64_div_s,    i64_div_s, types::i64)
      EMIT_BINARY(emit_i64_div_u,    i64_div_u, types::i64)
      EMIT_BINARY(emit_i64_rem_s,    i64_rem_s, types::i64)
      EMIT_BINARY(emit_i64_rem_u,    i64_rem_u, types::i64)
      EMIT_BINARY_COMM(emit_i64_and, i64_and, types::i64)
      EMIT_BINARY_COMM(emit_i64_or,  i64_or,  types::i64)
      EMIT_BINARY_COMM(emit_i64_xor, i64_xor, types::i64)
      EMIT_BINARY(emit_i64_shl,      i64_shl, types::i64)
      EMIT_BINARY(emit_i64_shr_s,    i64_shr_s, types::i64)
      EMIT_BINARY(emit_i64_shr_u,    i64_shr_u, types::i64)
      EMIT_BINARY(emit_i64_rotl,     i64_rotl, types::i64)
      EMIT_BINARY(emit_i64_rotr,     i64_rotr, types::i64)
      // f32 arithmetic
      EMIT_BINARY(emit_f32_add, f32_add, types::f32)
      EMIT_BINARY(emit_f32_sub, f32_sub, types::f32)
      EMIT_BINARY(emit_f32_mul, f32_mul, types::f32)
      EMIT_BINARY(emit_f32_div, f32_div, types::f32)
      EMIT_BINARY(emit_f32_min, f32_min, types::f32)
      EMIT_BINARY(emit_f32_max, f32_max, types::f32)
      EMIT_BINARY(emit_f32_copysign, f32_copysign, types::f32)
      // f64 arithmetic
      EMIT_BINARY(emit_f64_add, f64_add, types::f64)
      EMIT_BINARY(emit_f64_sub, f64_sub, types::f64)
      EMIT_BINARY(emit_f64_mul, f64_mul, types::f64)
      EMIT_BINARY(emit_f64_div, f64_div, types::f64)
      EMIT_BINARY(emit_f64_min, f64_min, types::f64)
      EMIT_BINARY(emit_f64_max, f64_max, types::f64)
      EMIT_BINARY(emit_f64_copysign, f64_copysign, types::f64)
#undef EMIT_BINARY
#undef EMIT_BINARY_COMM

      // ──────── SIMD operations (stubs - emit as generic v128_op) ────────
      // These all follow the same pattern: pop operands, push v128 result.
      // The sub-opcode is encoded in the imm field for later codegen.

#define EMIT_V128_LOAD(name, sub_op) \
      void name(uint32_t offset, uint32_t alignment) { \
         uint32_t addr = _func.vpop(); \
         uint32_t dest = _func.alloc_vreg(types::v128); \
         ir_inst inst{}; \
         inst.opcode = ir_op::v128_op; \
         inst.type = types::v128; \
         inst.flags = IR_NONE; \
         inst.dest = dest; \
         inst.ri.src1 = addr; \
         inst.ri.imm = sub_op; \
         _func.emit(inst); \
         _func.vpush(dest); \
      }

#define EMIT_V128_STORE(name, sub_op) \
      void name(uint32_t offset, uint32_t alignment) { \
         uint32_t val = _func.vpop(); \
         uint32_t addr = _func.vpop(); \
         ir_inst inst{}; \
         inst.opcode = ir_op::v128_op; \
         inst.type = types::pseudo; \
         inst.flags = IR_SIDE_EFFECT; \
         inst.dest = ir_vreg_none; \
         inst.ri.src1 = addr; \
         inst.ri.imm = sub_op; \
         _func.emit(inst); \
         ir_inst arg{}; arg.opcode = ir_op::arg; arg.type = types::pseudo; \
         arg.dest = ir_vreg_none; arg.rr.src1 = val; arg.rr.src2 = ir_vreg_none; \
         _func.emit(arg); \
      }

#define EMIT_V128_LANE_LOAD(name, sub_op) \
      void name(uint32_t offset, uint32_t alignment, uint8_t laneidx) { \
         uint32_t vec = _func.vpop(); \
         uint32_t addr = _func.vpop(); \
         uint32_t dest = _func.alloc_vreg(types::v128); \
         ir_inst inst{}; \
         inst.opcode = ir_op::v128_op; \
         inst.type = types::v128; \
         inst.flags = IR_NONE; \
         inst.dest = dest; \
         inst.ri.src1 = addr; \
         inst.ri.imm = sub_op; \
         _func.emit(inst); \
         ir_inst arg{}; arg.opcode = ir_op::arg; arg.type = types::pseudo; \
         arg.dest = ir_vreg_none; arg.rr.src1 = vec; arg.rr.src2 = laneidx; \
         _func.emit(arg); \
         _func.vpush(dest); \
      }

#define EMIT_V128_LANE_STORE(name, sub_op) \
      void name(uint32_t offset, uint32_t alignment, uint8_t laneidx) { \
         uint32_t vec = _func.vpop(); \
         uint32_t addr = _func.vpop(); \
         ir_inst inst{}; \
         inst.opcode = ir_op::v128_op; \
         inst.type = types::pseudo; \
         inst.flags = IR_SIDE_EFFECT; \
         inst.dest = ir_vreg_none; \
         inst.ri.src1 = addr; \
         inst.ri.imm = sub_op; \
         _func.emit(inst); \
         ir_inst arg{}; arg.opcode = ir_op::arg; arg.type = types::pseudo; \
         arg.dest = ir_vreg_none; arg.rr.src1 = vec; arg.rr.src2 = laneidx; \
         _func.emit(arg); \
      }

#define EMIT_V128_UNARY(name, sub_op, out_ty) \
      void name() { \
         uint32_t src = _func.vpop(); \
         uint32_t dest = _func.alloc_vreg(out_ty); \
         ir_inst inst{}; \
         inst.opcode = ir_op::v128_op; \
         inst.type = out_ty; \
         inst.flags = IR_NONE; \
         inst.dest = dest; \
         inst.ri.src1 = src; \
         inst.ri.imm = sub_op; \
         _func.emit(inst); \
         _func.vpush(dest); \
      }

#define EMIT_V128_BINARY(name, sub_op) \
      void name() { \
         uint32_t rhs = _func.vpop(); \
         uint32_t lhs = _func.vpop(); \
         uint32_t dest = _func.alloc_vreg(types::v128); \
         ir_inst inst{}; \
         inst.opcode = ir_op::v128_op; \
         inst.type = types::v128; \
         inst.flags = IR_NONE; \
         inst.dest = dest; \
         inst.rr.src1 = lhs; \
         inst.rr.src2 = rhs; \
         _func.emit(inst); \
         _func.vpush(dest); \
      }

#define EMIT_V128_TERNARY(name, sub_op) \
      void name() { \
         uint32_t c = _func.vpop(); \
         uint32_t b = _func.vpop(); \
         uint32_t a = _func.vpop(); \
         uint32_t dest = _func.alloc_vreg(types::v128); \
         ir_inst inst{}; \
         inst.opcode = ir_op::v128_op; \
         inst.type = types::v128; \
         inst.flags = IR_NONE; \
         inst.dest = dest; \
         inst.rr.src1 = a; \
         inst.rr.src2 = b; \
         _func.emit(inst); \
         ir_inst arg{}; arg.opcode = ir_op::arg; arg.type = types::pseudo; \
         arg.dest = ir_vreg_none; arg.rr.src1 = c; arg.rr.src2 = ir_vreg_none; \
         _func.emit(arg); \
         _func.vpush(dest); \
      }

#define EMIT_V128_EXTRACT(name, sub_op, out_ty) \
      void name(uint8_t laneidx) { \
         uint32_t src = _func.vpop(); \
         uint32_t dest = _func.alloc_vreg(out_ty); \
         ir_inst inst{}; \
         inst.opcode = ir_op::v128_op; \
         inst.type = out_ty; \
         inst.flags = IR_NONE; \
         inst.dest = dest; \
         inst.ri.src1 = src; \
         inst.ri.imm = (sub_op << 8) | laneidx; \
         _func.emit(inst); \
         _func.vpush(dest); \
      }

#define EMIT_V128_REPLACE(name, sub_op) \
      void name(uint8_t laneidx) { \
         uint32_t val = _func.vpop(); \
         uint32_t vec = _func.vpop(); \
         uint32_t dest = _func.alloc_vreg(types::v128); \
         ir_inst inst{}; \
         inst.opcode = ir_op::v128_op; \
         inst.type = types::v128; \
         inst.flags = IR_NONE; \
         inst.dest = dest; \
         inst.rr.src1 = vec; \
         inst.rr.src2 = val; \
         _func.emit(inst); \
         ir_inst arg{}; arg.opcode = ir_op::arg; arg.type = types::pseudo; \
         arg.dest = ir_vreg_none; arg.rr.src1 = laneidx; arg.rr.src2 = sub_op; \
         _func.emit(arg); \
         _func.vpush(dest); \
      }

#define EMIT_V128_SPLAT(name, sub_op) \
      void name() { \
         uint32_t src = _func.vpop(); \
         uint32_t dest = _func.alloc_vreg(types::v128); \
         ir_inst inst{}; \
         inst.opcode = ir_op::v128_op; \
         inst.type = types::v128; \
         inst.flags = IR_NONE; \
         inst.dest = dest; \
         inst.ri.src1 = src; \
         inst.ri.imm = sub_op; \
         _func.emit(inst); \
         _func.vpush(dest); \
      }

      // Use sub_op IDs starting from 1
      EMIT_V128_LOAD(emit_v128_load, 1)
      EMIT_V128_LOAD(emit_v128_load8x8_s, 2)
      EMIT_V128_LOAD(emit_v128_load8x8_u, 3)
      EMIT_V128_LOAD(emit_v128_load16x4_s, 4)
      EMIT_V128_LOAD(emit_v128_load16x4_u, 5)
      EMIT_V128_LOAD(emit_v128_load32x2_s, 6)
      EMIT_V128_LOAD(emit_v128_load32x2_u, 7)
      EMIT_V128_LOAD(emit_v128_load8_splat, 8)
      EMIT_V128_LOAD(emit_v128_load16_splat, 9)
      EMIT_V128_LOAD(emit_v128_load32_splat, 10)
      EMIT_V128_LOAD(emit_v128_load64_splat, 11)
      EMIT_V128_LOAD(emit_v128_load32_zero, 12)
      EMIT_V128_LOAD(emit_v128_load64_zero, 13)
      EMIT_V128_STORE(emit_v128_store, 14)
      EMIT_V128_LANE_LOAD(emit_v128_load8_lane, 15)
      EMIT_V128_LANE_LOAD(emit_v128_load16_lane, 16)
      EMIT_V128_LANE_LOAD(emit_v128_load32_lane, 17)
      EMIT_V128_LANE_LOAD(emit_v128_load64_lane, 18)
      EMIT_V128_LANE_STORE(emit_v128_store8_lane, 19)
      EMIT_V128_LANE_STORE(emit_v128_store16_lane, 20)
      EMIT_V128_LANE_STORE(emit_v128_store32_lane, 21)
      EMIT_V128_LANE_STORE(emit_v128_store64_lane, 22)

      void emit_v128_const(v128_t value) {
         uint32_t dest = _func.alloc_vreg(types::v128);
         ir_inst inst{};
         inst.opcode = ir_op::const_v128;
         inst.type = types::v128;
         inst.flags = IR_NONE;
         inst.dest = dest;
         inst.immv128 = value;
         _func.emit(inst);
         _func.vpush(dest);
      }

      void emit_i8x16_shuffle(const uint8_t* lanes) {
         uint32_t b = _func.vpop();
         uint32_t a = _func.vpop();
         uint32_t dest = _func.alloc_vreg(types::v128);
         ir_inst inst{};
         inst.opcode = ir_op::v128_op;
         inst.type = types::v128;
         inst.flags = IR_NONE;
         inst.dest = dest;
         inst.rr.src1 = a;
         inst.rr.src2 = b;
         _func.emit(inst);
         // Encode lane bytes as a following const
         ir_inst lanes_inst{};
         lanes_inst.opcode = ir_op::arg;
         lanes_inst.type = types::pseudo;
         lanes_inst.dest = ir_vreg_none;
         memcpy(&lanes_inst.immv128, lanes, 16);
         _func.emit(lanes_inst);
         _func.vpush(dest);
      }

      EMIT_V128_EXTRACT(emit_i8x16_extract_lane_s, 30, types::i32)
      EMIT_V128_EXTRACT(emit_i8x16_extract_lane_u, 31, types::i32)
      EMIT_V128_REPLACE(emit_i8x16_replace_lane, 32)
      EMIT_V128_EXTRACT(emit_i16x8_extract_lane_s, 33, types::i32)
      EMIT_V128_EXTRACT(emit_i16x8_extract_lane_u, 34, types::i32)
      EMIT_V128_REPLACE(emit_i16x8_replace_lane, 35)
      EMIT_V128_EXTRACT(emit_i32x4_extract_lane, 36, types::i32)
      EMIT_V128_REPLACE(emit_i32x4_replace_lane, 37)
      EMIT_V128_EXTRACT(emit_i64x2_extract_lane, 38, types::i64)
      EMIT_V128_REPLACE(emit_i64x2_replace_lane, 39)
      EMIT_V128_EXTRACT(emit_f32x4_extract_lane, 40, types::f32)
      EMIT_V128_REPLACE(emit_f32x4_replace_lane, 41)
      EMIT_V128_EXTRACT(emit_f64x2_extract_lane, 42, types::f64)
      EMIT_V128_REPLACE(emit_f64x2_replace_lane, 43)

      EMIT_V128_BINARY(emit_i8x16_swizzle, 44)
      EMIT_V128_SPLAT(emit_i8x16_splat, 45)
      EMIT_V128_SPLAT(emit_i16x8_splat, 46)
      EMIT_V128_SPLAT(emit_i32x4_splat, 47)
      EMIT_V128_SPLAT(emit_i64x2_splat, 48)
      EMIT_V128_SPLAT(emit_f32x4_splat, 49)
      EMIT_V128_SPLAT(emit_f64x2_splat, 50)

      // All the remaining SIMD binary/unary ops
      EMIT_V128_BINARY(emit_i8x16_eq, 51)
      EMIT_V128_BINARY(emit_i8x16_ne, 52)
      EMIT_V128_BINARY(emit_i8x16_lt_s, 53)
      EMIT_V128_BINARY(emit_i8x16_lt_u, 54)
      EMIT_V128_BINARY(emit_i8x16_gt_s, 55)
      EMIT_V128_BINARY(emit_i8x16_gt_u, 56)
      EMIT_V128_BINARY(emit_i8x16_le_s, 57)
      EMIT_V128_BINARY(emit_i8x16_le_u, 58)
      EMIT_V128_BINARY(emit_i8x16_ge_s, 59)
      EMIT_V128_BINARY(emit_i8x16_ge_u, 60)
      EMIT_V128_BINARY(emit_i16x8_eq, 61)
      EMIT_V128_BINARY(emit_i16x8_ne, 62)
      EMIT_V128_BINARY(emit_i16x8_lt_s, 63)
      EMIT_V128_BINARY(emit_i16x8_lt_u, 64)
      EMIT_V128_BINARY(emit_i16x8_gt_s, 65)
      EMIT_V128_BINARY(emit_i16x8_gt_u, 66)
      EMIT_V128_BINARY(emit_i16x8_le_s, 67)
      EMIT_V128_BINARY(emit_i16x8_le_u, 68)
      EMIT_V128_BINARY(emit_i16x8_ge_s, 69)
      EMIT_V128_BINARY(emit_i16x8_ge_u, 70)
      EMIT_V128_BINARY(emit_i32x4_eq, 71)
      EMIT_V128_BINARY(emit_i32x4_ne, 72)
      EMIT_V128_BINARY(emit_i32x4_lt_s, 73)
      EMIT_V128_BINARY(emit_i32x4_lt_u, 74)
      EMIT_V128_BINARY(emit_i32x4_gt_s, 75)
      EMIT_V128_BINARY(emit_i32x4_gt_u, 76)
      EMIT_V128_BINARY(emit_i32x4_le_s, 77)
      EMIT_V128_BINARY(emit_i32x4_le_u, 78)
      EMIT_V128_BINARY(emit_i32x4_ge_s, 79)
      EMIT_V128_BINARY(emit_i32x4_ge_u, 80)
      EMIT_V128_BINARY(emit_i64x2_eq, 81)
      EMIT_V128_BINARY(emit_i64x2_ne, 82)
      EMIT_V128_BINARY(emit_i64x2_lt_s, 83)
      EMIT_V128_BINARY(emit_i64x2_gt_s, 84)
      EMIT_V128_BINARY(emit_i64x2_le_s, 85)
      EMIT_V128_BINARY(emit_i64x2_ge_s, 86)
      EMIT_V128_BINARY(emit_f32x4_eq, 87)
      EMIT_V128_BINARY(emit_f32x4_ne, 88)
      EMIT_V128_BINARY(emit_f32x4_lt, 89)
      EMIT_V128_BINARY(emit_f32x4_gt, 90)
      EMIT_V128_BINARY(emit_f32x4_le, 91)
      EMIT_V128_BINARY(emit_f32x4_ge, 92)
      EMIT_V128_BINARY(emit_f64x2_eq, 93)
      EMIT_V128_BINARY(emit_f64x2_ne, 94)
      EMIT_V128_BINARY(emit_f64x2_lt, 95)
      EMIT_V128_BINARY(emit_f64x2_gt, 96)
      EMIT_V128_BINARY(emit_f64x2_le, 97)
      EMIT_V128_BINARY(emit_f64x2_ge, 98)
      EMIT_V128_UNARY(emit_v128_not, 99, types::v128)
      EMIT_V128_BINARY(emit_v128_and, 100)
      EMIT_V128_BINARY(emit_v128_andnot, 101)
      EMIT_V128_BINARY(emit_v128_or, 102)
      EMIT_V128_BINARY(emit_v128_xor, 103)
      EMIT_V128_TERNARY(emit_v128_bitselect, 104)
      EMIT_V128_UNARY(emit_v128_any_true, 105, types::i32)
      EMIT_V128_UNARY(emit_i8x16_abs, 106, types::v128)
      EMIT_V128_UNARY(emit_i8x16_neg, 107, types::v128)
      EMIT_V128_UNARY(emit_i8x16_popcnt, 108, types::v128)
      EMIT_V128_UNARY(emit_i8x16_all_true, 109, types::i32)
      EMIT_V128_UNARY(emit_i8x16_bitmask, 110, types::i32)
      EMIT_V128_BINARY(emit_i8x16_narrow_i16x8_s, 111)
      EMIT_V128_BINARY(emit_i8x16_narrow_i16x8_u, 112)
      EMIT_V128_BINARY(emit_i8x16_shl, 113)
      EMIT_V128_BINARY(emit_i8x16_shr_s, 114)
      EMIT_V128_BINARY(emit_i8x16_shr_u, 115)
      EMIT_V128_BINARY(emit_i8x16_add, 116)
      EMIT_V128_BINARY(emit_i8x16_add_sat_s, 117)
      EMIT_V128_BINARY(emit_i8x16_add_sat_u, 118)
      EMIT_V128_BINARY(emit_i8x16_sub, 119)
      EMIT_V128_BINARY(emit_i8x16_sub_sat_s, 120)
      EMIT_V128_BINARY(emit_i8x16_sub_sat_u, 121)
      EMIT_V128_BINARY(emit_i8x16_min_s, 122)
      EMIT_V128_BINARY(emit_i8x16_min_u, 123)
      EMIT_V128_BINARY(emit_i8x16_max_s, 124)
      EMIT_V128_BINARY(emit_i8x16_max_u, 125)
      EMIT_V128_BINARY(emit_i8x16_avgr_u, 126)
      EMIT_V128_UNARY(emit_i16x8_extadd_pairwise_i8x16_s, 127, types::v128)
      EMIT_V128_UNARY(emit_i16x8_extadd_pairwise_i8x16_u, 128, types::v128)
      EMIT_V128_UNARY(emit_i16x8_abs, 129, types::v128)
      EMIT_V128_UNARY(emit_i16x8_neg, 130, types::v128)
      EMIT_V128_BINARY(emit_i16x8_q15mulr_sat_s, 131)
      EMIT_V128_UNARY(emit_i16x8_all_true, 132, types::i32)
      EMIT_V128_UNARY(emit_i16x8_bitmask, 133, types::i32)
      EMIT_V128_BINARY(emit_i16x8_narrow_i32x4_s, 134)
      EMIT_V128_BINARY(emit_i16x8_narrow_i32x4_u, 135)
      EMIT_V128_UNARY(emit_i16x8_extend_low_i8x16_s, 136, types::v128)
      EMIT_V128_UNARY(emit_i16x8_extend_high_i8x16_s, 137, types::v128)
      EMIT_V128_UNARY(emit_i16x8_extend_low_i8x16_u, 138, types::v128)
      EMIT_V128_UNARY(emit_i16x8_extend_high_i8x16_u, 139, types::v128)
      EMIT_V128_BINARY(emit_i16x8_shl, 140)
      EMIT_V128_BINARY(emit_i16x8_shr_s, 141)
      EMIT_V128_BINARY(emit_i16x8_shr_u, 142)
      EMIT_V128_BINARY(emit_i16x8_add, 143)
      EMIT_V128_BINARY(emit_i16x8_add_sat_s, 144)
      EMIT_V128_BINARY(emit_i16x8_add_sat_u, 145)
      EMIT_V128_BINARY(emit_i16x8_sub, 146)
      EMIT_V128_BINARY(emit_i16x8_sub_sat_s, 147)
      EMIT_V128_BINARY(emit_i16x8_sub_sat_u, 148)
      EMIT_V128_BINARY(emit_i16x8_mul, 149)
      EMIT_V128_BINARY(emit_i16x8_min_s, 150)
      EMIT_V128_BINARY(emit_i16x8_min_u, 151)
      EMIT_V128_BINARY(emit_i16x8_max_s, 152)
      EMIT_V128_BINARY(emit_i16x8_max_u, 153)
      EMIT_V128_BINARY(emit_i16x8_avgr_u, 154)
      EMIT_V128_BINARY(emit_i16x8_extmul_low_i8x16_s, 155)
      EMIT_V128_BINARY(emit_i16x8_extmul_high_i8x16_s, 156)
      EMIT_V128_BINARY(emit_i16x8_extmul_low_i8x16_u, 157)
      EMIT_V128_BINARY(emit_i16x8_extmul_high_i8x16_u, 158)
      EMIT_V128_UNARY(emit_i32x4_extadd_pairwise_i16x8_s, 159, types::v128)
      EMIT_V128_UNARY(emit_i32x4_extadd_pairwise_i16x8_u, 160, types::v128)
      EMIT_V128_UNARY(emit_i32x4_abs, 161, types::v128)
      EMIT_V128_UNARY(emit_i32x4_neg, 162, types::v128)
      EMIT_V128_UNARY(emit_i32x4_all_true, 163, types::i32)
      EMIT_V128_UNARY(emit_i32x4_bitmask, 164, types::i32)
      EMIT_V128_UNARY(emit_i32x4_extend_low_i16x8_s, 165, types::v128)
      EMIT_V128_UNARY(emit_i32x4_extend_high_i16x8_s, 166, types::v128)
      EMIT_V128_UNARY(emit_i32x4_extend_low_i16x8_u, 167, types::v128)
      EMIT_V128_UNARY(emit_i32x4_extend_high_i16x8_u, 168, types::v128)
      EMIT_V128_BINARY(emit_i32x4_shl, 169)
      EMIT_V128_BINARY(emit_i32x4_shr_s, 170)
      EMIT_V128_BINARY(emit_i32x4_shr_u, 171)
      EMIT_V128_BINARY(emit_i32x4_add, 172)
      EMIT_V128_BINARY(emit_i32x4_sub, 173)
      EMIT_V128_BINARY(emit_i32x4_mul, 174)
      EMIT_V128_BINARY(emit_i32x4_min_s, 175)
      EMIT_V128_BINARY(emit_i32x4_min_u, 176)
      EMIT_V128_BINARY(emit_i32x4_max_s, 177)
      EMIT_V128_BINARY(emit_i32x4_max_u, 178)
      EMIT_V128_BINARY(emit_i32x4_dot_i16x8_s, 179)
      EMIT_V128_BINARY(emit_i32x4_extmul_low_i16x8_s, 180)
      EMIT_V128_BINARY(emit_i32x4_extmul_high_i16x8_s, 181)
      EMIT_V128_BINARY(emit_i32x4_extmul_low_i16x8_u, 182)
      EMIT_V128_BINARY(emit_i32x4_extmul_high_i16x8_u, 183)
      EMIT_V128_UNARY(emit_i64x2_abs, 184, types::v128)
      EMIT_V128_UNARY(emit_i64x2_neg, 185, types::v128)
      EMIT_V128_UNARY(emit_i64x2_all_true, 186, types::i32)
      EMIT_V128_UNARY(emit_i64x2_bitmask, 187, types::i32)
      EMIT_V128_UNARY(emit_i64x2_extend_low_i32x4_s, 188, types::v128)
      EMIT_V128_UNARY(emit_i64x2_extend_high_i32x4_s, 189, types::v128)
      EMIT_V128_UNARY(emit_i64x2_extend_low_i32x4_u, 190, types::v128)
      EMIT_V128_UNARY(emit_i64x2_extend_high_i32x4_u, 191, types::v128)
      EMIT_V128_BINARY(emit_i64x2_shl, 192)
      EMIT_V128_BINARY(emit_i64x2_shr_s, 193)
      EMIT_V128_BINARY(emit_i64x2_shr_u, 194)
      EMIT_V128_BINARY(emit_i64x2_add, 195)
      EMIT_V128_BINARY(emit_i64x2_sub, 196)
      EMIT_V128_BINARY(emit_i64x2_mul, 197)
      EMIT_V128_BINARY(emit_i64x2_extmul_low_i32x4_s, 198)
      EMIT_V128_BINARY(emit_i64x2_extmul_high_i32x4_s, 199)
      EMIT_V128_BINARY(emit_i64x2_extmul_low_i32x4_u, 200)
      EMIT_V128_BINARY(emit_i64x2_extmul_high_i32x4_u, 201)
      EMIT_V128_UNARY(emit_f32x4_ceil, 202, types::v128)
      EMIT_V128_UNARY(emit_f32x4_floor, 203, types::v128)
      EMIT_V128_UNARY(emit_f32x4_trunc, 204, types::v128)
      EMIT_V128_UNARY(emit_f32x4_nearest, 205, types::v128)
      EMIT_V128_UNARY(emit_f32x4_abs, 206, types::v128)
      EMIT_V128_UNARY(emit_f32x4_neg, 207, types::v128)
      EMIT_V128_UNARY(emit_f32x4_sqrt, 208, types::v128)
      EMIT_V128_BINARY(emit_f32x4_add, 209)
      EMIT_V128_BINARY(emit_f32x4_sub, 210)
      EMIT_V128_BINARY(emit_f32x4_mul, 211)
      EMIT_V128_BINARY(emit_f32x4_div, 212)
      EMIT_V128_BINARY(emit_f32x4_min, 213)
      EMIT_V128_BINARY(emit_f32x4_max, 214)
      EMIT_V128_BINARY(emit_f32x4_pmin, 215)
      EMIT_V128_BINARY(emit_f32x4_pmax, 216)
      EMIT_V128_UNARY(emit_f64x2_ceil, 217, types::v128)
      EMIT_V128_UNARY(emit_f64x2_floor, 218, types::v128)
      EMIT_V128_UNARY(emit_f64x2_trunc, 219, types::v128)
      EMIT_V128_UNARY(emit_f64x2_nearest, 220, types::v128)
      EMIT_V128_UNARY(emit_f64x2_abs, 221, types::v128)
      EMIT_V128_UNARY(emit_f64x2_neg, 222, types::v128)
      EMIT_V128_UNARY(emit_f64x2_sqrt, 223, types::v128)
      EMIT_V128_BINARY(emit_f64x2_add, 224)
      EMIT_V128_BINARY(emit_f64x2_sub, 225)
      EMIT_V128_BINARY(emit_f64x2_mul, 226)
      EMIT_V128_BINARY(emit_f64x2_div, 227)
      EMIT_V128_BINARY(emit_f64x2_min, 228)
      EMIT_V128_BINARY(emit_f64x2_max, 229)
      EMIT_V128_BINARY(emit_f64x2_pmin, 230)
      EMIT_V128_BINARY(emit_f64x2_pmax, 231)
      EMIT_V128_UNARY(emit_i32x4_trunc_sat_f32x4_s, 232, types::v128)
      EMIT_V128_UNARY(emit_i32x4_trunc_sat_f32x4_u, 233, types::v128)
      EMIT_V128_UNARY(emit_f32x4_convert_i32x4_s, 234, types::v128)
      EMIT_V128_UNARY(emit_f32x4_convert_i32x4_u, 235, types::v128)
      EMIT_V128_UNARY(emit_i32x4_trunc_sat_f64x2_s_zero, 236, types::v128)
      EMIT_V128_UNARY(emit_i32x4_trunc_sat_f64x2_u_zero, 237, types::v128)
      EMIT_V128_UNARY(emit_f64x2_convert_low_i32x4_s, 238, types::v128)
      EMIT_V128_UNARY(emit_f64x2_convert_low_i32x4_u, 239, types::v128)
      EMIT_V128_UNARY(emit_f32x4_demote_f64x2_zero, 240, types::v128)
      EMIT_V128_UNARY(emit_f64x2_promote_low_f32x4, 241, types::v128)

#undef EMIT_V128_LOAD
#undef EMIT_V128_STORE
#undef EMIT_V128_LANE_LOAD
#undef EMIT_V128_LANE_STORE
#undef EMIT_V128_UNARY
#undef EMIT_V128_BINARY
#undef EMIT_V128_TERNARY
#undef EMIT_V128_EXTRACT
#undef EMIT_V128_REPLACE
#undef EMIT_V128_SPLAT

      // ──────── Bulk memory operations ────────
      void emit_memory_init(std::uint32_t seg) { emit_bulk_mem(ir_op::memory_init, seg); }
      void emit_data_drop(std::uint32_t seg)   { emit_bulk_void(ir_op::data_drop, seg); }
      void emit_memory_copy()                   { emit_bulk_mem(ir_op::memory_copy, 0); }
      void emit_memory_fill()                   { emit_bulk_mem(ir_op::memory_fill, 0); }
      void emit_table_init(std::uint32_t seg)   { emit_bulk_mem(ir_op::table_init, seg); }
      void emit_elem_drop(std::uint32_t seg)    { emit_bulk_void(ir_op::elem_drop, seg); }
      void emit_table_copy()                    { emit_bulk_mem(ir_op::table_copy, 0); }

      // ──────── Branch fixup ────────
      void fix_branch(branch_t, label_t) {
         // In the IR, branches reference block indices directly.
         // No fixup needed since we resolve during IR construction.
      }

    private:
      uint32_t _current_block = 0;

      uint32_t get_branch_depth(uint32_t depth_change, uint8_t rt) {
         // depth_change encodes the branch target depth from the parser
         // For the original machine_code_writer, this includes stack adjustments.
         // For IR, we just need the control stack depth.
         // The depth_change from the parser already accounts for the stack.
         // We need to find which control entry this targets.
         uint32_t has_value = (rt != types::pseudo) ? 1 : 0;
         uint32_t depth = depth_change - has_value;
         // Convert from stack depth change to control stack index
         // This is approximate - the parser provides depth_change as
         // the number of stack items to pop. We need to map to control depth.
         // Actually, depth_change in the writer interface is the raw depth_change
         // from the parser, which is the number of values on the stack between
         // the current point and the branch target.
         // The real control depth is tracked separately by the parser.
         // For now, use depth_change as-is (the parser manages this).
         return 0; // TODO: This needs proper mapping from the parser
      }

      uint32_t get_br_target(ir_control_entry& ctrl) {
         if (ctrl.is_loop) {
            return ctrl.block_idx;  // loop: branch to header
         } else {
            return ctrl.merge_block;  // block/if: branch to merge
         }
      }

      void emit_br_to(uint32_t target_block) {
         ir_inst inst{};
         inst.opcode = ir_op::br;
         inst.type = types::pseudo;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = ir_vreg_none;
         inst.br.target = target_block;
         inst.br.src1 = ir_vreg_none;
         _func.emit(inst);
      }

      // ──────── Instruction emission helpers ────────
      void emit_void(ir_op op) {
         ir_inst inst{};
         inst.opcode = op;
         inst.type = types::pseudo;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = ir_vreg_none;
         inst.rr.src1 = ir_vreg_none;
         inst.rr.src2 = ir_vreg_none;
         _func.emit(inst);
      }

      uint32_t emit_nullary(ir_op op, uint8_t type) {
         uint32_t dest = _func.alloc_vreg(type);
         ir_inst inst{};
         inst.opcode = op;
         inst.type = type;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = dest;
         inst.rr.src1 = ir_vreg_none;
         inst.rr.src2 = ir_vreg_none;
         _func.emit(inst);
         return dest;
      }

      uint32_t emit_unary_inst(ir_op op, uint8_t type, uint32_t src) {
         uint32_t dest = _func.alloc_vreg(type);
         ir_inst inst{};
         inst.opcode = op;
         inst.type = type;
         inst.flags = IR_NONE;
         inst.dest = dest;
         inst.rr.src1 = src;
         inst.rr.src2 = ir_vreg_none;
         _func.emit(inst);
         return dest;
      }

      uint32_t emit_binary_inst(ir_op op, uint8_t type, uint32_t lhs, uint32_t rhs,
                                uint8_t extra_flags = IR_NONE) {
         uint32_t dest = _func.alloc_vreg(type);
         ir_inst inst{};
         inst.opcode = op;
         inst.type = type;
         inst.flags = extra_flags;
         inst.dest = dest;
         inst.rr.src1 = lhs;
         inst.rr.src2 = rhs;
         _func.emit(inst);
         return dest;
      }

      uint32_t emit_ternary(ir_op op, uint8_t type, uint32_t a, uint32_t b, uint32_t c) {
         uint32_t dest = _func.alloc_vreg(type);
         ir_inst inst{};
         inst.opcode = op;
         inst.type = type;
         inst.flags = IR_NONE;
         inst.dest = dest;
         inst.rr.src1 = a;
         inst.rr.src2 = b;
         _func.emit(inst);
         // Encode third operand as following arg
         ir_inst arg{};
         arg.opcode = ir_op::arg;
         arg.type = types::pseudo;
         arg.dest = ir_vreg_none;
         arg.rr.src1 = c;
         arg.rr.src2 = ir_vreg_none;
         _func.emit(arg);
         return dest;
      }

      void emit_bulk_mem(ir_op op, uint32_t seg) {
         // These pop 3 args from stack (dest, src, len)
         uint32_t len = _func.vpop();
         uint32_t src = _func.vpop();
         uint32_t dest = _func.vpop();
         ir_inst inst{};
         inst.opcode = op;
         inst.type = types::pseudo;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = ir_vreg_none;
         inst.rr.src1 = dest;
         inst.rr.src2 = src;
         _func.emit(inst);
         ir_inst arg{};
         arg.opcode = ir_op::arg;
         arg.type = types::pseudo;
         arg.dest = ir_vreg_none;
         arg.rr.src1 = len;
         arg.rr.src2 = seg;
         _func.emit(arg);
      }

      void emit_bulk_void(ir_op op, uint32_t seg) {
         ir_inst inst{};
         inst.opcode = op;
         inst.type = types::pseudo;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = ir_vreg_none;
         inst.ri.src1 = ir_vreg_none;
         inst.ri.imm = static_cast<int32_t>(seg);
         _func.emit(inst);
      }

      // ──────── State ────────
      growable_allocator& _allocator;
      std::size_t _source_bytes;
      module& _mod;
      jit_codegen<Context, StackLimitIsBytes> _codegen;
      ir_function _func;
   };

}} // namespace eosio::vm
