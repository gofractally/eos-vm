#pragma once

// Pass 1 of the two-pass optimizing JIT (jit2).
// Converts WASM stack machine operations to virtual-register IR.
//
// Phase 4: Standalone IR builder — no machine_code_writer dependency.
//   During parsing, builds IR for each function into an array.
//   After parsing completes, the destructor compiles all functions
//   via jit_codegen (Pass 2) which emits native x86_64 code.

#include <eosio/vm/allocator.hpp>
#include <eosio/vm/exceptions.hpp>
#include <eosio/vm/jit_codegen.hpp>
#include <eosio/vm/jit_ir.hpp>
#include <eosio/vm/types.hpp>

#include <cstdint>
#include <cstring>

namespace eosio { namespace vm {

   template<typename Context, bool StackLimitIsBytes>
   class ir_writer {
      using codegen_t = jit_codegen<Context, StackLimitIsBytes>;
    public:
      // Branch/label types — dummy values since IR tracks control flow directly.
      // The parser stores and passes these between emit_if/emit_else/emit_end/emit_br
      // but never interprets them. fix_branch is a no-op.
      using branch_t = uint32_t;
      using label_t  = uint32_t;

      ir_writer(growable_allocator& alloc, std::size_t source_bytes, module& mod)
         : _allocator(alloc), _source_bytes(source_bytes), _mod(mod) {
         // Mark the start of the code region. All allocations from here (IR buffers
         // during parsing + native code during compilation) are within this region.
         // end_code<true>() copies the native code to executable memory and resets
         // the allocator offset back to this point, reclaiming everything.
         _code_segment_base = alloc.start_code();
         // Allocate per-function IR storage from the code region.
         _num_functions = mod.code.size();
         _functions = alloc.alloc<ir_function>(_num_functions);
         for (uint32_t i = 0; i < _num_functions; ++i) {
            _functions[i] = ir_function{};
         }
      }

      ~ir_writer() {
         // Pass 2: Compile all functions from IR to native x86_64.
         // jit_codegen uses _code_segment_base (set in our constructor) as the
         // start of the code region. It allocates native code after the IR data,
         // then end_code<true>() copies native code to executable memory and
         // resets the allocator, reclaiming both IR and code buffers.
         codegen_t codegen(_allocator, _mod, _code_segment_base);
         codegen.emit_entry_and_error_handlers();
         for (uint32_t i = 0; i < _num_functions; ++i) {
            codegen.compile_function(_functions[i], _mod.code[i]);
         }
         codegen.finalize_code();
      }

      static constexpr uint32_t get_depth_for_type(uint8_t type) {
         if (type == types::v128) return 2;
         return type == types::pseudo ? 0 : 1;
      }

      // ──── Prologue / epilogue ────

      void emit_prologue(const func_type& ft, const std::vector<local_entry>& locals, uint32_t funcnum) {
         // Initialize IR for this function using the actual function body size
         _func = &_functions[funcnum];
         _func->init(_allocator, _mod.code[funcnum].size);
         _func->func_index = funcnum;
         _func->type = &ft;
         _func->num_params = ft.param_types.size();

         // Count total locals (params + body locals)
         uint32_t total = ft.param_types.size();
         for (const auto& local : locals) {
            total += local.count;
         }
         _func->num_locals = total;
         _unreachable = false;

         // Push function-level control entry
         ir_control_entry entry{};
         entry.block_idx = _func->new_block();
         entry.stack_depth = 0;
         entry.result_type = ft.return_count > 0 ? static_cast<uint8_t>(ft.return_type) : types::pseudo;
         entry.is_loop = 0;
         entry.is_function = 1;
         _func->ctrl_push(entry);
         _func->start_block(entry.block_idx);
      }

      void emit_epilogue(const func_type& /*ft*/, const std::vector<local_entry>& /*locals*/, uint32_t /*funcnum*/) {
         // Nothing to do — IR is complete after parsing
      }

      void finalize(function_body& /*body*/) {
         // Code offset is set during Pass 2 (jit_codegen::compile_function)
         _func = nullptr;
      }

      // Parser calls these for instruction map — return dummy values since
      // null_debug_info ignores them and addresses are set during Pass 2.
      const void* get_addr() const { return nullptr; }
      const void* get_base_addr() const { return nullptr; }
      void set_stack_usage(std::uint64_t /*u*/) { }

      // ──── Control flow ────
      void emit_unreachable() {
         if (!_unreachable) {
            ir_emit_nullary(ir_op::unreachable, types::pseudo);
         }
         _unreachable = true;
      }

      void emit_nop() { }

      label_t emit_end() {
         // Pop control entry
         if (_func->ctrl_stack_top > 0) {
            auto entry = _func->ctrl_pop();
            _func->end_block(entry.block_idx);
            // Restore stack to block entry depth + result
            if (_unreachable) {
               _func->vstack_resize(entry.stack_depth);
               if (entry.result_type != types::pseudo) {
                  uint32_t dest = _func->alloc_vreg(entry.result_type);
                  _func->vpush(dest);
               }
            }
            _unreachable = false;
         }
         return 0;
      }

      branch_t emit_return(uint32_t dc, uint8_t rt) {
         if (!_unreachable) {
            ir_inst inst{};
            inst.opcode = ir_op::return_;
            inst.type = types::pseudo;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = ir_vreg_none;
            if (rt != types::pseudo && _func->vstack_depth() > 0) {
               inst.rr.src1 = _func->vpop();
            } else {
               inst.rr.src1 = ir_vreg_none;
            }
            inst.rr.src2 = ir_vreg_none;
            _func->emit(inst);
         }
         _unreachable = true;
         return 0;
      }

      void emit_block() {
         ir_control_entry entry{};
         entry.block_idx = _func->new_block();
         entry.stack_depth = _func->vstack_depth();
         entry.result_type = types::pseudo;
         entry.is_loop = 0;
         entry.is_function = 0;
         _func->ctrl_push(entry);
      }

      label_t emit_loop() {
         ir_control_entry entry{};
         entry.block_idx = _func->new_block();
         _func->blocks[entry.block_idx].is_loop = 1;
         entry.stack_depth = _func->vstack_depth();
         entry.result_type = types::pseudo;
         entry.is_loop = 1;
         entry.is_function = 0;
         _func->ctrl_push(entry);
         _func->start_block(entry.block_idx);
         return 0;
      }

      branch_t emit_if() {
         if (!_unreachable) {
            uint32_t cond = _func->vpop();
            ir_inst inst{};
            inst.opcode = ir_op::if_;
            inst.type = types::pseudo;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = ir_vreg_none;
            inst.rr.src1 = cond;
            inst.rr.src2 = ir_vreg_none;
            _func->emit(inst);
         }
         ir_control_entry entry{};
         entry.block_idx = _func->new_block();
         entry.stack_depth = _func->vstack_depth();
         entry.result_type = types::pseudo;
         entry.is_loop = 0;
         entry.is_function = 0;
         _func->ctrl_push(entry);
         return 0;
      }

      branch_t emit_else(branch_t /*if_loc*/) {
         // Reset vstack to block entry depth
         if (_func->ctrl_stack_top > 0) {
            auto& entry = _func->ctrl_back();
            if (!_unreachable) {
               if (entry.result_type != types::pseudo && _func->vstack_depth() > entry.stack_depth) {
                  _func->vpop();
               }
            }
            _func->vstack_resize(entry.stack_depth);
         }
         _unreachable = false;
         return 0;
      }

      branch_t emit_br(uint32_t dc, uint8_t rt) {
         if (!_unreachable) {
            // Resolve depth count to target block index via control stack
            auto& target_entry = _func->ctrl_at(dc);
            ir_inst inst{};
            inst.opcode = ir_op::br;
            inst.type = rt;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = dc; // store depth change for multipop
            inst.br.target = target_entry.block_idx;
            if (rt != types::pseudo && _func->vstack_depth() > 0) {
               inst.br.src1 = _func->vstack_back();
            } else {
               inst.br.src1 = ir_vreg_none;
            }
            _func->emit(inst);
         }
         _unreachable = true;
         return 0;
      }

      branch_t emit_br_if(uint32_t dc, uint8_t rt) {
         if (!_unreachable) {
            auto& target_entry = _func->ctrl_at(dc);
            uint32_t cond = _func->vpop();
            ir_inst inst{};
            inst.opcode = ir_op::br_if;
            inst.type = rt;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = dc; // store depth change for multipop
            inst.br.target = target_entry.block_idx;
            inst.br.src1 = cond;
            _func->emit(inst);
         }
         return 0;
      }

      struct br_table_parser {
         ir_writer* _writer;
         br_table_parser(ir_writer* w) : _writer(w) {}
         branch_t emit_case(uint32_t dc, uint8_t rt) {
            if (!_writer->_unreachable) {
               auto& target_entry = _writer->_func->ctrl_at(dc);
               ir_inst inst{};
               inst.opcode = ir_op::br;  // each case is a branch
               inst.type = rt;
               inst.flags = IR_SIDE_EFFECT;
               inst.dest = dc;
               inst.br.target = target_entry.block_idx;
               inst.br.src1 = ir_vreg_none;
               _writer->_func->emit(inst);
            }
            return 0;
         }
         branch_t emit_default(uint32_t dc, uint8_t rt) {
            return emit_case(dc, rt);
         }
      };
      br_table_parser emit_br_table(uint32_t /*table_size*/) {
         if (!_unreachable) {
            uint32_t idx = _func->vpop();
            ir_inst inst{};
            inst.opcode = ir_op::br_table;
            inst.type = types::pseudo;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = ir_vreg_none;
            inst.rr.src1 = idx;
            inst.rr.src2 = ir_vreg_none;
            _func->emit(inst);
         }
         _unreachable = true;
         return br_table_parser(this);
      }

      // ──── Calls ────
      void emit_call(const func_type& ft, uint32_t funcnum) {
         if (!_unreachable) {
            for (uint32_t i = 0; i < ft.param_types.size(); ++i) {
               _func->vpop();
            }
            if (ft.return_count > 0) {
               uint32_t dest = _func->alloc_vreg(ft.return_type);
               ir_inst inst{};
               inst.opcode = ir_op::call;
               inst.type = ft.return_type;
               inst.flags = IR_SIDE_EFFECT;
               inst.dest = dest;
               inst.call.index = funcnum;
               inst.call.src1 = ir_vreg_none;
               _func->emit(inst);
               _func->vpush(dest);
            } else {
               ir_inst inst{};
               inst.opcode = ir_op::call;
               inst.type = types::pseudo;
               inst.flags = IR_SIDE_EFFECT;
               inst.dest = ir_vreg_none;
               inst.call.index = funcnum;
               inst.call.src1 = ir_vreg_none;
               _func->emit(inst);
            }
         }
      }

      void emit_call_indirect(const func_type& ft, uint32_t fti) {
         if (!_unreachable) {
            uint32_t table_idx = _func->vpop();
            for (uint32_t i = 0; i < ft.param_types.size(); ++i) {
               _func->vpop();
            }
            if (ft.return_count > 0) {
               uint32_t dest = _func->alloc_vreg(ft.return_type);
               ir_inst inst{};
               inst.opcode = ir_op::call_indirect;
               inst.type = ft.return_type;
               inst.flags = IR_SIDE_EFFECT;
               inst.dest = dest;
               inst.call.index = fti;
               inst.call.src1 = table_idx;
               _func->emit(inst);
               _func->vpush(dest);
            } else {
               ir_inst inst{};
               inst.opcode = ir_op::call_indirect;
               inst.type = types::pseudo;
               inst.flags = IR_SIDE_EFFECT;
               inst.dest = ir_vreg_none;
               inst.call.index = fti;
               inst.call.src1 = table_idx;
               _func->emit(inst);
            }
         }
      }

      // ──── Parametric ────
      void emit_drop(uint8_t type) {
         if (!_unreachable) {
            _func->vpop();
            if (type == types::v128) _func->vpop();
         }
      }

      void emit_select(uint8_t type) {
         if (!_unreachable) {
            uint32_t cond = _func->vpop();
            uint32_t val2 = _func->vpop();
            uint32_t val1 = _func->vpop();
            uint32_t dest = _func->alloc_vreg(type);
            ir_inst inst{};
            inst.opcode = ir_op::select;
            inst.type = type;
            inst.dest = dest;
            inst.rr.src1 = val1;
            inst.rr.src2 = val2;
            _func->emit(inst);
            _func->vpush(dest);
         }
      }

      // ──── Local / global ────
      void emit_get_local(uint32_t li, uint8_t ty) {
         if (!_unreachable) {
            uint32_t dest = _func->alloc_vreg(ty);
            ir_inst inst{};
            inst.opcode = ir_op::local_get;
            inst.type = ty;
            inst.dest = dest;
            inst.local.index = li;
            inst.local.src1 = ir_vreg_none;
            _func->emit(inst);
            _func->vpush(dest);
         }
      }

      void emit_set_local(uint32_t li, uint8_t ty) {
         if (!_unreachable) {
            uint32_t src = _func->vpop();
            ir_inst inst{};
            inst.opcode = ir_op::local_set;
            inst.type = ty;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = ir_vreg_none;
            inst.local.index = li;
            inst.local.src1 = src;
            _func->emit(inst);
         }
      }

      void emit_tee_local(uint32_t li, uint8_t ty) {
         if (!_unreachable) {
            uint32_t src = _func->vstack_back();
            ir_inst inst{};
            inst.opcode = ir_op::local_tee;
            inst.type = ty;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = ir_vreg_none;
            inst.local.index = li;
            inst.local.src1 = src;
            _func->emit(inst);
         }
      }

      void emit_get_global(uint32_t gi) {
         if (!_unreachable) {
            uint32_t dest = _func->alloc_vreg(types::i64);
            ir_inst inst{};
            inst.opcode = ir_op::global_get;
            inst.type = types::i64;
            inst.dest = dest;
            inst.local.index = gi;
            inst.local.src1 = ir_vreg_none;
            _func->emit(inst);
            _func->vpush(dest);
         }
      }

      void emit_set_global(uint32_t gi) {
         if (!_unreachable) {
            uint32_t src = _func->vpop();
            ir_inst inst{};
            inst.opcode = ir_op::global_set;
            inst.type = types::i64;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = ir_vreg_none;
            inst.local.index = gi;
            inst.local.src1 = src;
            _func->emit(inst);
         }
      }

      // ──── Memory loads ────
      void emit_i32_load(uint32_t o, uint32_t a)    { ir_load(ir_op::i32_load, types::i32, o); }
      void emit_i64_load(uint32_t o, uint32_t a)    { ir_load(ir_op::i64_load, types::i64, o); }
      void emit_f32_load(uint32_t o, uint32_t a)    { ir_load(ir_op::f32_load, types::f32, o); }
      void emit_f64_load(uint32_t o, uint32_t a)    { ir_load(ir_op::f64_load, types::f64, o); }
      void emit_i32_load8_s(uint32_t o, uint32_t a) { ir_load(ir_op::i32_load8_s, types::i32, o); }
      void emit_i32_load16_s(uint32_t o, uint32_t a){ ir_load(ir_op::i32_load16_s, types::i32, o); }
      void emit_i32_load8_u(uint32_t o, uint32_t a) { ir_load(ir_op::i32_load8_u, types::i32, o); }
      void emit_i32_load16_u(uint32_t o, uint32_t a){ ir_load(ir_op::i32_load16_u, types::i32, o); }
      void emit_i64_load8_s(uint32_t o, uint32_t a) { ir_load(ir_op::i64_load8_s, types::i64, o); }
      void emit_i64_load16_s(uint32_t o, uint32_t a){ ir_load(ir_op::i64_load16_s, types::i64, o); }
      void emit_i64_load32_s(uint32_t o, uint32_t a){ ir_load(ir_op::i64_load32_s, types::i64, o); }
      void emit_i64_load8_u(uint32_t o, uint32_t a) { ir_load(ir_op::i64_load8_u, types::i64, o); }
      void emit_i64_load16_u(uint32_t o, uint32_t a){ ir_load(ir_op::i64_load16_u, types::i64, o); }
      void emit_i64_load32_u(uint32_t o, uint32_t a){ ir_load(ir_op::i64_load32_u, types::i64, o); }

      // ──── Memory stores ────
      void emit_i32_store(uint32_t o, uint32_t a)   { ir_store(ir_op::i32_store, types::i32, o); }
      void emit_i64_store(uint32_t o, uint32_t a)   { ir_store(ir_op::i64_store, types::i64, o); }
      void emit_f32_store(uint32_t o, uint32_t a)   { ir_store(ir_op::f32_store, types::f32, o); }
      void emit_f64_store(uint32_t o, uint32_t a)   { ir_store(ir_op::f64_store, types::f64, o); }
      void emit_i32_store8(uint32_t o, uint32_t a)  { ir_store(ir_op::i32_store8, types::i32, o); }
      void emit_i32_store16(uint32_t o, uint32_t a) { ir_store(ir_op::i32_store16, types::i32, o); }
      void emit_i64_store8(uint32_t o, uint32_t a)  { ir_store(ir_op::i64_store8, types::i64, o); }
      void emit_i64_store16(uint32_t o, uint32_t a) { ir_store(ir_op::i64_store16, types::i64, o); }
      void emit_i64_store32(uint32_t o, uint32_t a) { ir_store(ir_op::i64_store32, types::i64, o); }

      // ──── Memory management ────
      void emit_current_memory() {
         if (!_unreachable) {
            uint32_t dest = _func->alloc_vreg(types::i32);
            ir_emit_nullary(ir_op::memory_size, types::i32);
            _func->vpush(dest);
         }
      }
      void emit_grow_memory() {
         if (!_unreachable) {
            uint32_t src = _func->vpop();
            uint32_t dest = _func->alloc_vreg(types::i32);
            ir_inst inst{};
            inst.opcode = ir_op::memory_grow;
            inst.type = types::i32;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = dest;
            inst.rr.src1 = src;
            inst.rr.src2 = ir_vreg_none;
            _func->emit(inst);
            _func->vpush(dest);
         }
      }

      // ──── Constants ────
      void emit_i32_const(uint32_t v) {
         if (!_unreachable) {
            uint32_t dest = _func->alloc_vreg(types::i32);
            ir_inst inst{};
            inst.opcode = ir_op::const_i32;
            inst.type = types::i32;
            inst.dest = dest;
            inst.imm64 = static_cast<int64_t>(static_cast<int32_t>(v));
            _func->emit(inst);
            _func->vpush(dest);
         }
      }

      void emit_i64_const(uint64_t v) {
         if (!_unreachable) {
            uint32_t dest = _func->alloc_vreg(types::i64);
            ir_inst inst{};
            inst.opcode = ir_op::const_i64;
            inst.type = types::i64;
            inst.dest = dest;
            inst.imm64 = static_cast<int64_t>(v);
            _func->emit(inst);
            _func->vpush(dest);
         }
      }

      void emit_f32_const(float v) {
         if (!_unreachable) {
            uint32_t dest = _func->alloc_vreg(types::f32);
            ir_inst inst{};
            inst.opcode = ir_op::const_f32;
            inst.type = types::f32;
            inst.dest = dest;
            std::memcpy(&inst.immf32, &v, 4);
            _func->emit(inst);
            _func->vpush(dest);
         }
      }

      void emit_f64_const(double v) {
         if (!_unreachable) {
            uint32_t dest = _func->alloc_vreg(types::f64);
            ir_inst inst{};
            inst.opcode = ir_op::const_f64;
            inst.type = types::f64;
            inst.dest = dest;
            std::memcpy(&inst.immf64, &v, 8);
            _func->emit(inst);
            _func->vpush(dest);
         }
      }

      // ──── Comparisons ────
      void emit_i32_eqz()  { ir_unop(ir_op::i32_eqz, types::i32); }
      void emit_i32_eq()   { ir_binop(ir_op::i32_eq, types::i32); }
      void emit_i32_ne()   { ir_binop(ir_op::i32_ne, types::i32); }
      void emit_i32_lt_s() { ir_binop(ir_op::i32_lt_s, types::i32); }
      void emit_i32_lt_u() { ir_binop(ir_op::i32_lt_u, types::i32); }
      void emit_i32_gt_s() { ir_binop(ir_op::i32_gt_s, types::i32); }
      void emit_i32_gt_u() { ir_binop(ir_op::i32_gt_u, types::i32); }
      void emit_i32_le_s() { ir_binop(ir_op::i32_le_s, types::i32); }
      void emit_i32_le_u() { ir_binop(ir_op::i32_le_u, types::i32); }
      void emit_i32_ge_s() { ir_binop(ir_op::i32_ge_s, types::i32); }
      void emit_i32_ge_u() { ir_binop(ir_op::i32_ge_u, types::i32); }
      void emit_i64_eqz()  { ir_unop(ir_op::i64_eqz, types::i32); }
      void emit_i64_eq()   { ir_binop(ir_op::i64_eq, types::i32); }
      void emit_i64_ne()   { ir_binop(ir_op::i64_ne, types::i32); }
      void emit_i64_lt_s() { ir_binop(ir_op::i64_lt_s, types::i32); }
      void emit_i64_lt_u() { ir_binop(ir_op::i64_lt_u, types::i32); }
      void emit_i64_gt_s() { ir_binop(ir_op::i64_gt_s, types::i32); }
      void emit_i64_gt_u() { ir_binop(ir_op::i64_gt_u, types::i32); }
      void emit_i64_le_s() { ir_binop(ir_op::i64_le_s, types::i32); }
      void emit_i64_le_u() { ir_binop(ir_op::i64_le_u, types::i32); }
      void emit_i64_ge_s() { ir_binop(ir_op::i64_ge_s, types::i32); }
      void emit_i64_ge_u() { ir_binop(ir_op::i64_ge_u, types::i32); }
      void emit_f32_eq()   { ir_binop(ir_op::f32_eq, types::i32); }
      void emit_f32_ne()   { ir_binop(ir_op::f32_ne, types::i32); }
      void emit_f32_lt()   { ir_binop(ir_op::f32_lt, types::i32); }
      void emit_f32_gt()   { ir_binop(ir_op::f32_gt, types::i32); }
      void emit_f32_le()   { ir_binop(ir_op::f32_le, types::i32); }
      void emit_f32_ge()   { ir_binop(ir_op::f32_ge, types::i32); }
      void emit_f64_eq()   { ir_binop(ir_op::f64_eq, types::i32); }
      void emit_f64_ne()   { ir_binop(ir_op::f64_ne, types::i32); }
      void emit_f64_lt()   { ir_binop(ir_op::f64_lt, types::i32); }
      void emit_f64_gt()   { ir_binop(ir_op::f64_gt, types::i32); }
      void emit_f64_le()   { ir_binop(ir_op::f64_le, types::i32); }
      void emit_f64_ge()   { ir_binop(ir_op::f64_ge, types::i32); }

      // ──── Integer arithmetic ────
      void emit_i32_clz()    { ir_unop(ir_op::i32_clz, types::i32); }
      void emit_i32_ctz()    { ir_unop(ir_op::i32_ctz, types::i32); }
      void emit_i32_popcnt() { ir_unop(ir_op::i32_popcnt, types::i32); }
      void emit_i32_add()    { ir_binop(ir_op::i32_add, types::i32); }
      void emit_i32_sub()    { ir_binop(ir_op::i32_sub, types::i32); }
      void emit_i32_mul()    { ir_binop(ir_op::i32_mul, types::i32); }
      void emit_i32_div_s()  { ir_binop(ir_op::i32_div_s, types::i32); }
      void emit_i32_div_u()  { ir_binop(ir_op::i32_div_u, types::i32); }
      void emit_i32_rem_s()  { ir_binop(ir_op::i32_rem_s, types::i32); }
      void emit_i32_rem_u()  { ir_binop(ir_op::i32_rem_u, types::i32); }
      void emit_i32_and()    { ir_binop(ir_op::i32_and, types::i32); }
      void emit_i32_or()     { ir_binop(ir_op::i32_or, types::i32); }
      void emit_i32_xor()    { ir_binop(ir_op::i32_xor, types::i32); }
      void emit_i32_shl()    { ir_binop(ir_op::i32_shl, types::i32); }
      void emit_i32_shr_s()  { ir_binop(ir_op::i32_shr_s, types::i32); }
      void emit_i32_shr_u()  { ir_binop(ir_op::i32_shr_u, types::i32); }
      void emit_i32_rotl()   { ir_binop(ir_op::i32_rotl, types::i32); }
      void emit_i32_rotr()   { ir_binop(ir_op::i32_rotr, types::i32); }
      void emit_i64_clz()    { ir_unop(ir_op::i64_clz, types::i64); }
      void emit_i64_ctz()    { ir_unop(ir_op::i64_ctz, types::i64); }
      void emit_i64_popcnt() { ir_unop(ir_op::i64_popcnt, types::i64); }
      void emit_i64_add()    { ir_binop(ir_op::i64_add, types::i64); }
      void emit_i64_sub()    { ir_binop(ir_op::i64_sub, types::i64); }
      void emit_i64_mul()    { ir_binop(ir_op::i64_mul, types::i64); }
      void emit_i64_div_s()  { ir_binop(ir_op::i64_div_s, types::i64); }
      void emit_i64_div_u()  { ir_binop(ir_op::i64_div_u, types::i64); }
      void emit_i64_rem_s()  { ir_binop(ir_op::i64_rem_s, types::i64); }
      void emit_i64_rem_u()  { ir_binop(ir_op::i64_rem_u, types::i64); }
      void emit_i64_and()    { ir_binop(ir_op::i64_and, types::i64); }
      void emit_i64_or()     { ir_binop(ir_op::i64_or, types::i64); }
      void emit_i64_xor()    { ir_binop(ir_op::i64_xor, types::i64); }
      void emit_i64_shl()    { ir_binop(ir_op::i64_shl, types::i64); }
      void emit_i64_shr_s()  { ir_binop(ir_op::i64_shr_s, types::i64); }
      void emit_i64_shr_u()  { ir_binop(ir_op::i64_shr_u, types::i64); }
      void emit_i64_rotl()   { ir_binop(ir_op::i64_rotl, types::i64); }
      void emit_i64_rotr()   { ir_binop(ir_op::i64_rotr, types::i64); }

      // ──── Float arithmetic ────
      void emit_f32_abs()      { ir_unop(ir_op::f32_abs, types::f32); }
      void emit_f32_neg()      { ir_unop(ir_op::f32_neg, types::f32); }
      void emit_f32_ceil()     { ir_unop(ir_op::f32_ceil, types::f32); }
      void emit_f32_floor()    { ir_unop(ir_op::f32_floor, types::f32); }
      void emit_f32_trunc()    { ir_unop(ir_op::f32_trunc, types::f32); }
      void emit_f32_nearest()  { ir_unop(ir_op::f32_nearest, types::f32); }
      void emit_f32_sqrt()     { ir_unop(ir_op::f32_sqrt, types::f32); }
      void emit_f32_add()      { ir_binop(ir_op::f32_add, types::f32); }
      void emit_f32_sub()      { ir_binop(ir_op::f32_sub, types::f32); }
      void emit_f32_mul()      { ir_binop(ir_op::f32_mul, types::f32); }
      void emit_f32_div()      { ir_binop(ir_op::f32_div, types::f32); }
      void emit_f32_min()      { ir_binop(ir_op::f32_min, types::f32); }
      void emit_f32_max()      { ir_binop(ir_op::f32_max, types::f32); }
      void emit_f32_copysign() { ir_binop(ir_op::f32_copysign, types::f32); }
      void emit_f64_abs()      { ir_unop(ir_op::f64_abs, types::f64); }
      void emit_f64_neg()      { ir_unop(ir_op::f64_neg, types::f64); }
      void emit_f64_ceil()     { ir_unop(ir_op::f64_ceil, types::f64); }
      void emit_f64_floor()    { ir_unop(ir_op::f64_floor, types::f64); }
      void emit_f64_trunc()    { ir_unop(ir_op::f64_trunc, types::f64); }
      void emit_f64_nearest()  { ir_unop(ir_op::f64_nearest, types::f64); }
      void emit_f64_sqrt()     { ir_unop(ir_op::f64_sqrt, types::f64); }
      void emit_f64_add()      { ir_binop(ir_op::f64_add, types::f64); }
      void emit_f64_sub()      { ir_binop(ir_op::f64_sub, types::f64); }
      void emit_f64_mul()      { ir_binop(ir_op::f64_mul, types::f64); }
      void emit_f64_div()      { ir_binop(ir_op::f64_div, types::f64); }
      void emit_f64_min()      { ir_binop(ir_op::f64_min, types::f64); }
      void emit_f64_max()      { ir_binop(ir_op::f64_max, types::f64); }
      void emit_f64_copysign() { ir_binop(ir_op::f64_copysign, types::f64); }

      // ──── Conversions ────
      void emit_i32_wrap_i64()       { ir_unop(ir_op::i32_wrap_i64, types::i32); }
      void emit_i32_trunc_s_f32()    { ir_unop(ir_op::i32_trunc_s_f32, types::i32); }
      void emit_i32_trunc_u_f32()    { ir_unop(ir_op::i32_trunc_u_f32, types::i32); }
      void emit_i32_trunc_s_f64()    { ir_unop(ir_op::i32_trunc_s_f64, types::i32); }
      void emit_i32_trunc_u_f64()    { ir_unop(ir_op::i32_trunc_u_f64, types::i32); }
      void emit_i64_extend_s_i32()   { ir_unop(ir_op::i64_extend_s_i32, types::i64); }
      void emit_i64_extend_u_i32()   { ir_unop(ir_op::i64_extend_u_i32, types::i64); }
      void emit_i64_trunc_s_f32()    { ir_unop(ir_op::i64_trunc_s_f32, types::i64); }
      void emit_i64_trunc_u_f32()    { ir_unop(ir_op::i64_trunc_u_f32, types::i64); }
      void emit_i64_trunc_s_f64()    { ir_unop(ir_op::i64_trunc_s_f64, types::i64); }
      void emit_i64_trunc_u_f64()    { ir_unop(ir_op::i64_trunc_u_f64, types::i64); }
      void emit_f32_convert_s_i32()  { ir_unop(ir_op::f32_convert_s_i32, types::f32); }
      void emit_f32_convert_u_i32()  { ir_unop(ir_op::f32_convert_u_i32, types::f32); }
      void emit_f32_convert_s_i64()  { ir_unop(ir_op::f32_convert_s_i64, types::f32); }
      void emit_f32_convert_u_i64()  { ir_unop(ir_op::f32_convert_u_i64, types::f32); }
      void emit_f32_demote_f64()     { ir_unop(ir_op::f32_demote_f64, types::f32); }
      void emit_f64_convert_s_i32()  { ir_unop(ir_op::f64_convert_s_i32, types::f64); }
      void emit_f64_convert_u_i32()  { ir_unop(ir_op::f64_convert_u_i32, types::f64); }
      void emit_f64_convert_s_i64()  { ir_unop(ir_op::f64_convert_s_i64, types::f64); }
      void emit_f64_convert_u_i64()  { ir_unop(ir_op::f64_convert_u_i64, types::f64); }
      void emit_f64_promote_f32()    { ir_unop(ir_op::f64_promote_f32, types::f64); }
      void emit_i32_reinterpret_f32(){ ir_unop(ir_op::i32_reinterpret_f32, types::i32); }
      void emit_i64_reinterpret_f64(){ ir_unop(ir_op::i64_reinterpret_f64, types::i64); }
      void emit_f32_reinterpret_i32(){ ir_unop(ir_op::f32_reinterpret_i32, types::f32); }
      void emit_f64_reinterpret_i64(){ ir_unop(ir_op::f64_reinterpret_i64, types::f64); }
      void emit_i32_trunc_sat_f32_s(){ ir_unop(ir_op::i32_trunc_sat_f32_s, types::i32); }
      void emit_i32_trunc_sat_f32_u(){ ir_unop(ir_op::i32_trunc_sat_f32_u, types::i32); }
      void emit_i32_trunc_sat_f64_s(){ ir_unop(ir_op::i32_trunc_sat_f64_s, types::i32); }
      void emit_i32_trunc_sat_f64_u(){ ir_unop(ir_op::i32_trunc_sat_f64_u, types::i32); }
      void emit_i64_trunc_sat_f32_s(){ ir_unop(ir_op::i64_trunc_sat_f32_s, types::i64); }
      void emit_i64_trunc_sat_f32_u(){ ir_unop(ir_op::i64_trunc_sat_f32_u, types::i64); }
      void emit_i64_trunc_sat_f64_s(){ ir_unop(ir_op::i64_trunc_sat_f64_s, types::i64); }
      void emit_i64_trunc_sat_f64_u(){ ir_unop(ir_op::i64_trunc_sat_f64_u, types::i64); }
      void emit_i32_extend8_s()      { ir_unop(ir_op::i32_extend8_s, types::i32); }
      void emit_i32_extend16_s()     { ir_unop(ir_op::i32_extend16_s, types::i32); }
      void emit_i64_extend8_s()      { ir_unop(ir_op::i64_extend8_s, types::i64); }
      void emit_i64_extend16_s()     { ir_unop(ir_op::i64_extend16_s, types::i64); }
      void emit_i64_extend32_s()     { ir_unop(ir_op::i64_extend32_s, types::i64); }

      // ──── SIMD (vstack tracking only, no IR instructions) ────
      void emit_v128_load(uint32_t o, uint32_t a)  { ir_simd_load(o); }
      void emit_v128_load8x8_s(uint32_t o, uint32_t a)  { ir_simd_load(o); }
      void emit_v128_load8x8_u(uint32_t o, uint32_t a)  { ir_simd_load(o); }
      void emit_v128_load16x4_s(uint32_t o, uint32_t a) { ir_simd_load(o); }
      void emit_v128_load16x4_u(uint32_t o, uint32_t a) { ir_simd_load(o); }
      void emit_v128_load32x2_s(uint32_t o, uint32_t a) { ir_simd_load(o); }
      void emit_v128_load32x2_u(uint32_t o, uint32_t a) { ir_simd_load(o); }
      void emit_v128_load8_splat(uint32_t o, uint32_t a) { ir_simd_load(o); }
      void emit_v128_load16_splat(uint32_t o, uint32_t a) { ir_simd_load(o); }
      void emit_v128_load32_splat(uint32_t o, uint32_t a) { ir_simd_load(o); }
      void emit_v128_load64_splat(uint32_t o, uint32_t a) { ir_simd_load(o); }
      void emit_v128_load32_zero(uint32_t o, uint32_t a)  { ir_simd_load(o); }
      void emit_v128_load64_zero(uint32_t o, uint32_t a)  { ir_simd_load(o); }
      void emit_v128_store(uint32_t o, uint32_t a) { ir_simd_store(o); }
      void emit_v128_load8_lane(uint32_t o, uint32_t a, uint8_t l)  { ir_simd_load_lane(); }
      void emit_v128_load16_lane(uint32_t o, uint32_t a, uint8_t l) { ir_simd_load_lane(); }
      void emit_v128_load32_lane(uint32_t o, uint32_t a, uint8_t l) { ir_simd_load_lane(); }
      void emit_v128_load64_lane(uint32_t o, uint32_t a, uint8_t l) { ir_simd_load_lane(); }
      void emit_v128_store8_lane(uint32_t o, uint32_t a, uint8_t l)  { ir_simd_store_lane(); }
      void emit_v128_store16_lane(uint32_t o, uint32_t a, uint8_t l) { ir_simd_store_lane(); }
      void emit_v128_store32_lane(uint32_t o, uint32_t a, uint8_t l) { ir_simd_store_lane(); }
      void emit_v128_store64_lane(uint32_t o, uint32_t a, uint8_t l) { ir_simd_store_lane(); }

      void emit_v128_const(v128_t v) {
         if (!_unreachable) {
            uint32_t dest = _func->alloc_vreg(types::v128);
            _func->vpush(dest);
            uint32_t dest2 = _func->alloc_vreg(types::v128);
            _func->vpush(dest2);
         }
      }
      void emit_i8x16_shuffle(const uint8_t* l) { ir_simd_binop(); }

// SIMD ops: track vstack (2 slots per v128) but don't build IR instructions.
#define SIMD_UNOP(name)  void name() { ir_simd_unop(); }
#define SIMD_BINOP(name) void name() { ir_simd_binop(); }
#define SIMD_SHIFT(name) void name() { ir_simd_shift(); }
#define SIMD_EXTRACT(name) void name(uint8_t a) { ir_simd_extract(); }
#define SIMD_REPLACE(name) void name(uint8_t a) { ir_simd_replace(); }
#define SIMD_SPLAT(name)   void name() { ir_simd_splat(); }
#define SIMD_RELOP(name)   void name() { ir_simd_binop(); }

      SIMD_EXTRACT(emit_i8x16_extract_lane_s) SIMD_EXTRACT(emit_i8x16_extract_lane_u) SIMD_REPLACE(emit_i8x16_replace_lane)
      SIMD_EXTRACT(emit_i16x8_extract_lane_s) SIMD_EXTRACT(emit_i16x8_extract_lane_u) SIMD_REPLACE(emit_i16x8_replace_lane)
      SIMD_EXTRACT(emit_i32x4_extract_lane) SIMD_REPLACE(emit_i32x4_replace_lane)
      SIMD_EXTRACT(emit_i64x2_extract_lane) SIMD_REPLACE(emit_i64x2_replace_lane)
      SIMD_EXTRACT(emit_f32x4_extract_lane) SIMD_REPLACE(emit_f32x4_replace_lane)
      SIMD_EXTRACT(emit_f64x2_extract_lane) SIMD_REPLACE(emit_f64x2_replace_lane)

      SIMD_BINOP(emit_i8x16_swizzle)
      SIMD_SPLAT(emit_i8x16_splat) SIMD_SPLAT(emit_i16x8_splat)
      SIMD_SPLAT(emit_i32x4_splat) SIMD_SPLAT(emit_i64x2_splat)
      SIMD_SPLAT(emit_f32x4_splat) SIMD_SPLAT(emit_f64x2_splat)

      SIMD_RELOP(emit_i8x16_eq) SIMD_RELOP(emit_i8x16_ne) SIMD_RELOP(emit_i8x16_lt_s) SIMD_RELOP(emit_i8x16_lt_u)
      SIMD_RELOP(emit_i8x16_gt_s) SIMD_RELOP(emit_i8x16_gt_u) SIMD_RELOP(emit_i8x16_le_s) SIMD_RELOP(emit_i8x16_le_u)
      SIMD_RELOP(emit_i8x16_ge_s) SIMD_RELOP(emit_i8x16_ge_u)
      SIMD_RELOP(emit_i16x8_eq) SIMD_RELOP(emit_i16x8_ne) SIMD_RELOP(emit_i16x8_lt_s) SIMD_RELOP(emit_i16x8_lt_u)
      SIMD_RELOP(emit_i16x8_gt_s) SIMD_RELOP(emit_i16x8_gt_u) SIMD_RELOP(emit_i16x8_le_s) SIMD_RELOP(emit_i16x8_le_u)
      SIMD_RELOP(emit_i16x8_ge_s) SIMD_RELOP(emit_i16x8_ge_u)
      SIMD_RELOP(emit_i32x4_eq) SIMD_RELOP(emit_i32x4_ne) SIMD_RELOP(emit_i32x4_lt_s) SIMD_RELOP(emit_i32x4_lt_u)
      SIMD_RELOP(emit_i32x4_gt_s) SIMD_RELOP(emit_i32x4_gt_u) SIMD_RELOP(emit_i32x4_le_s) SIMD_RELOP(emit_i32x4_le_u)
      SIMD_RELOP(emit_i32x4_ge_s) SIMD_RELOP(emit_i32x4_ge_u)
      SIMD_RELOP(emit_i64x2_eq) SIMD_RELOP(emit_i64x2_ne) SIMD_RELOP(emit_i64x2_lt_s) SIMD_RELOP(emit_i64x2_gt_s)
      SIMD_RELOP(emit_i64x2_le_s) SIMD_RELOP(emit_i64x2_ge_s)
      SIMD_RELOP(emit_f32x4_eq) SIMD_RELOP(emit_f32x4_ne) SIMD_RELOP(emit_f32x4_lt) SIMD_RELOP(emit_f32x4_gt)
      SIMD_RELOP(emit_f32x4_le) SIMD_RELOP(emit_f32x4_ge)
      SIMD_RELOP(emit_f64x2_eq) SIMD_RELOP(emit_f64x2_ne) SIMD_RELOP(emit_f64x2_lt) SIMD_RELOP(emit_f64x2_gt)
      SIMD_RELOP(emit_f64x2_le) SIMD_RELOP(emit_f64x2_ge)
      SIMD_UNOP(emit_v128_not) SIMD_BINOP(emit_v128_and) SIMD_BINOP(emit_v128_andnot) SIMD_BINOP(emit_v128_or)
      SIMD_BINOP(emit_v128_xor)
      void emit_v128_bitselect() {
         if (!_unreachable) {
            _func->vpop(); _func->vpop(); // mask
            _func->vpop(); _func->vpop(); // val2
            _func->vpop(); _func->vpop(); // val1
            uint32_t d1 = _func->alloc_vreg(types::v128); _func->vpush(d1);
            uint32_t d2 = _func->alloc_vreg(types::v128); _func->vpush(d2);
         }
      }
      void emit_v128_any_true() {
         if (!_unreachable) {
            _func->vpop(); _func->vpop();
            uint32_t d = _func->alloc_vreg(types::i32); _func->vpush(d);
         }
      }
      SIMD_UNOP(emit_i8x16_abs) SIMD_UNOP(emit_i8x16_neg) SIMD_UNOP(emit_i8x16_popcnt)
      void emit_i8x16_all_true() { ir_simd_test(); }
      void emit_i8x16_bitmask() { ir_simd_test(); }
      SIMD_BINOP(emit_i8x16_narrow_i16x8_s) SIMD_BINOP(emit_i8x16_narrow_i16x8_u)
      SIMD_SHIFT(emit_i8x16_shl) SIMD_SHIFT(emit_i8x16_shr_s) SIMD_SHIFT(emit_i8x16_shr_u)
      SIMD_BINOP(emit_i8x16_add) SIMD_BINOP(emit_i8x16_add_sat_s) SIMD_BINOP(emit_i8x16_add_sat_u)
      SIMD_BINOP(emit_i8x16_sub) SIMD_BINOP(emit_i8x16_sub_sat_s) SIMD_BINOP(emit_i8x16_sub_sat_u)
      SIMD_BINOP(emit_i8x16_min_s) SIMD_BINOP(emit_i8x16_min_u) SIMD_BINOP(emit_i8x16_max_s)
      SIMD_BINOP(emit_i8x16_max_u) SIMD_BINOP(emit_i8x16_avgr_u)
      SIMD_UNOP(emit_i16x8_extadd_pairwise_i8x16_s) SIMD_UNOP(emit_i16x8_extadd_pairwise_i8x16_u)
      SIMD_UNOP(emit_i16x8_abs) SIMD_UNOP(emit_i16x8_neg) SIMD_BINOP(emit_i16x8_q15mulr_sat_s)
      void emit_i16x8_all_true() { ir_simd_test(); }
      void emit_i16x8_bitmask() { ir_simd_test(); }
      SIMD_BINOP(emit_i16x8_narrow_i32x4_s) SIMD_BINOP(emit_i16x8_narrow_i32x4_u)
      SIMD_UNOP(emit_i16x8_extend_low_i8x16_s) SIMD_UNOP(emit_i16x8_extend_high_i8x16_s)
      SIMD_UNOP(emit_i16x8_extend_low_i8x16_u) SIMD_UNOP(emit_i16x8_extend_high_i8x16_u)
      SIMD_SHIFT(emit_i16x8_shl) SIMD_SHIFT(emit_i16x8_shr_s) SIMD_SHIFT(emit_i16x8_shr_u)
      SIMD_BINOP(emit_i16x8_add) SIMD_BINOP(emit_i16x8_add_sat_s) SIMD_BINOP(emit_i16x8_add_sat_u)
      SIMD_BINOP(emit_i16x8_sub) SIMD_BINOP(emit_i16x8_sub_sat_s) SIMD_BINOP(emit_i16x8_sub_sat_u)
      SIMD_BINOP(emit_i16x8_mul) SIMD_BINOP(emit_i16x8_min_s) SIMD_BINOP(emit_i16x8_min_u)
      SIMD_BINOP(emit_i16x8_max_s) SIMD_BINOP(emit_i16x8_max_u) SIMD_BINOP(emit_i16x8_avgr_u)
      SIMD_BINOP(emit_i16x8_extmul_low_i8x16_s) SIMD_BINOP(emit_i16x8_extmul_high_i8x16_s)
      SIMD_BINOP(emit_i16x8_extmul_low_i8x16_u) SIMD_BINOP(emit_i16x8_extmul_high_i8x16_u)
      SIMD_UNOP(emit_i32x4_extadd_pairwise_i16x8_s) SIMD_UNOP(emit_i32x4_extadd_pairwise_i16x8_u)
      SIMD_UNOP(emit_i32x4_abs) SIMD_UNOP(emit_i32x4_neg)
      void emit_i32x4_all_true() { ir_simd_test(); }
      void emit_i32x4_bitmask() { ir_simd_test(); }
      SIMD_UNOP(emit_i32x4_extend_low_i16x8_s) SIMD_UNOP(emit_i32x4_extend_high_i16x8_s)
      SIMD_UNOP(emit_i32x4_extend_low_i16x8_u) SIMD_UNOP(emit_i32x4_extend_high_i16x8_u)
      SIMD_SHIFT(emit_i32x4_shl) SIMD_SHIFT(emit_i32x4_shr_s) SIMD_SHIFT(emit_i32x4_shr_u)
      SIMD_BINOP(emit_i32x4_add) SIMD_BINOP(emit_i32x4_sub) SIMD_BINOP(emit_i32x4_mul)
      SIMD_BINOP(emit_i32x4_min_s) SIMD_BINOP(emit_i32x4_min_u) SIMD_BINOP(emit_i32x4_max_s) SIMD_BINOP(emit_i32x4_max_u)
      SIMD_BINOP(emit_i32x4_dot_i16x8_s)
      SIMD_BINOP(emit_i32x4_extmul_low_i16x8_s) SIMD_BINOP(emit_i32x4_extmul_high_i16x8_s)
      SIMD_BINOP(emit_i32x4_extmul_low_i16x8_u) SIMD_BINOP(emit_i32x4_extmul_high_i16x8_u)
      SIMD_UNOP(emit_i64x2_abs) SIMD_UNOP(emit_i64x2_neg)
      void emit_i64x2_all_true() { ir_simd_test(); }
      void emit_i64x2_bitmask() { ir_simd_test(); }
      SIMD_UNOP(emit_i64x2_extend_low_i32x4_s) SIMD_UNOP(emit_i64x2_extend_high_i32x4_s)
      SIMD_UNOP(emit_i64x2_extend_low_i32x4_u) SIMD_UNOP(emit_i64x2_extend_high_i32x4_u)
      SIMD_SHIFT(emit_i64x2_shl) SIMD_SHIFT(emit_i64x2_shr_s) SIMD_SHIFT(emit_i64x2_shr_u)
      SIMD_BINOP(emit_i64x2_add) SIMD_BINOP(emit_i64x2_sub) SIMD_BINOP(emit_i64x2_mul)
      SIMD_BINOP(emit_i64x2_extmul_low_i32x4_s) SIMD_BINOP(emit_i64x2_extmul_high_i32x4_s)
      SIMD_BINOP(emit_i64x2_extmul_low_i32x4_u) SIMD_BINOP(emit_i64x2_extmul_high_i32x4_u)
      SIMD_UNOP(emit_f32x4_ceil) SIMD_UNOP(emit_f32x4_floor) SIMD_UNOP(emit_f32x4_trunc) SIMD_UNOP(emit_f32x4_nearest)
      SIMD_UNOP(emit_f32x4_abs) SIMD_UNOP(emit_f32x4_neg) SIMD_UNOP(emit_f32x4_sqrt)
      SIMD_BINOP(emit_f32x4_add) SIMD_BINOP(emit_f32x4_sub) SIMD_BINOP(emit_f32x4_mul) SIMD_BINOP(emit_f32x4_div)
      SIMD_BINOP(emit_f32x4_min) SIMD_BINOP(emit_f32x4_max) SIMD_BINOP(emit_f32x4_pmin) SIMD_BINOP(emit_f32x4_pmax)
      SIMD_UNOP(emit_f64x2_ceil) SIMD_UNOP(emit_f64x2_floor) SIMD_UNOP(emit_f64x2_trunc) SIMD_UNOP(emit_f64x2_nearest)
      SIMD_UNOP(emit_f64x2_abs) SIMD_UNOP(emit_f64x2_neg) SIMD_UNOP(emit_f64x2_sqrt)
      SIMD_BINOP(emit_f64x2_add) SIMD_BINOP(emit_f64x2_sub) SIMD_BINOP(emit_f64x2_mul) SIMD_BINOP(emit_f64x2_div)
      SIMD_BINOP(emit_f64x2_min) SIMD_BINOP(emit_f64x2_max) SIMD_BINOP(emit_f64x2_pmin) SIMD_BINOP(emit_f64x2_pmax)
      SIMD_UNOP(emit_i32x4_trunc_sat_f32x4_s) SIMD_UNOP(emit_i32x4_trunc_sat_f32x4_u)
      SIMD_UNOP(emit_f32x4_convert_i32x4_s) SIMD_UNOP(emit_f32x4_convert_i32x4_u)
      SIMD_UNOP(emit_i32x4_trunc_sat_f64x2_s_zero) SIMD_UNOP(emit_i32x4_trunc_sat_f64x2_u_zero)
      SIMD_UNOP(emit_f64x2_convert_low_i32x4_s) SIMD_UNOP(emit_f64x2_convert_low_i32x4_u)
      SIMD_UNOP(emit_f32x4_demote_f64x2_zero) SIMD_UNOP(emit_f64x2_promote_low_f32x4)

#undef SIMD_UNOP
#undef SIMD_BINOP
#undef SIMD_SHIFT
#undef SIMD_EXTRACT
#undef SIMD_REPLACE
#undef SIMD_SPLAT
#undef SIMD_RELOP

      // ──── Bulk memory ────
      void emit_memory_init(std::uint32_t s) { ir_bulk_mem3(); }
      void emit_data_drop(std::uint32_t s) { }
      void emit_memory_copy() { ir_bulk_mem3(); }
      void emit_memory_fill() { ir_bulk_mem3(); }
      void emit_table_init(std::uint32_t s) { ir_bulk_mem3(); }
      void emit_elem_drop(std::uint32_t s) { }
      void emit_table_copy() { ir_bulk_mem3(); }

      // ──── Branch fixup ────
      // No-op: IR tracks control flow via the control stack, not code addresses.
      void fix_branch(branch_t /*br*/, label_t /*lbl*/) { }

    private:
      // ──── IR building helpers ────

      void ir_binop(ir_op op, uint8_t result_type) {
         if (_unreachable) return;
         uint32_t rhs = _func->vpop();
         uint32_t lhs = _func->vpop();
         uint32_t dest = _func->alloc_vreg(result_type);
         ir_inst inst{};
         inst.opcode = op;
         inst.type = result_type;
         inst.dest = dest;
         inst.rr.src1 = lhs;
         inst.rr.src2 = rhs;
         _func->emit(inst);
         _func->vpush(dest);
      }

      void ir_unop(ir_op op, uint8_t result_type) {
         if (_unreachable) return;
         uint32_t src = _func->vpop();
         uint32_t dest = _func->alloc_vreg(result_type);
         ir_inst inst{};
         inst.opcode = op;
         inst.type = result_type;
         inst.dest = dest;
         inst.rr.src1 = src;
         inst.rr.src2 = ir_vreg_none;
         _func->emit(inst);
         _func->vpush(dest);
      }

      void ir_emit_nullary(ir_op op, uint8_t type) {
         if (_unreachable) return;
         ir_inst inst{};
         inst.opcode = op;
         inst.type = type;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = ir_vreg_none;
         inst.rr.src1 = ir_vreg_none;
         inst.rr.src2 = ir_vreg_none;
         _func->emit(inst);
      }

      void ir_load(ir_op op, uint8_t result_type, uint32_t offset) {
         if (_unreachable) return;
         uint32_t addr = _func->vpop();
         uint32_t dest = _func->alloc_vreg(result_type);
         ir_inst inst{};
         inst.opcode = op;
         inst.type = result_type;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = dest;
         inst.ri.src1 = addr;
         inst.ri.imm = static_cast<int32_t>(offset);
         _func->emit(inst);
         _func->vpush(dest);
      }

      void ir_store(ir_op op, uint8_t type, uint32_t offset) {
         if (_unreachable) return;
         uint32_t val = _func->vpop();
         uint32_t addr = _func->vpop();
         ir_inst inst{};
         inst.opcode = op;
         inst.type = type;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = ir_vreg_none;
         inst.ri.src1 = addr;
         inst.ri.imm = static_cast<int32_t>(offset);
         _func->emit(inst);
      }

      // ──── SIMD vstack tracking ────
      void ir_simd_load(uint32_t) {
         if (_unreachable) return;
         _func->vpop();
         uint32_t d1 = _func->alloc_vreg(types::v128); _func->vpush(d1);
         uint32_t d2 = _func->alloc_vreg(types::v128); _func->vpush(d2);
      }
      void ir_simd_store(uint32_t) {
         if (_unreachable) return;
         _func->vpop(); _func->vpop();
         _func->vpop();
      }
      void ir_simd_load_lane() {
         if (_unreachable) return;
         _func->vpop(); _func->vpop();
         _func->vpop();
         uint32_t d1 = _func->alloc_vreg(types::v128); _func->vpush(d1);
         uint32_t d2 = _func->alloc_vreg(types::v128); _func->vpush(d2);
      }
      void ir_simd_store_lane() {
         if (_unreachable) return;
         _func->vpop(); _func->vpop();
         _func->vpop();
      }
      void ir_simd_unop() {
         if (_unreachable) return;
         _func->vpop(); _func->vpop();
         uint32_t d1 = _func->alloc_vreg(types::v128); _func->vpush(d1);
         uint32_t d2 = _func->alloc_vreg(types::v128); _func->vpush(d2);
      }
      void ir_simd_binop() {
         if (_unreachable) return;
         _func->vpop(); _func->vpop();
         _func->vpop(); _func->vpop();
         uint32_t d1 = _func->alloc_vreg(types::v128); _func->vpush(d1);
         uint32_t d2 = _func->alloc_vreg(types::v128); _func->vpush(d2);
      }
      void ir_simd_shift() {
         if (_unreachable) return;
         _func->vpop();
         _func->vpop(); _func->vpop();
         uint32_t d1 = _func->alloc_vreg(types::v128); _func->vpush(d1);
         uint32_t d2 = _func->alloc_vreg(types::v128); _func->vpush(d2);
      }
      void ir_simd_extract() {
         if (_unreachable) return;
         _func->vpop(); _func->vpop();
         uint32_t d = _func->alloc_vreg(types::i64); _func->vpush(d);
      }
      void ir_simd_replace() {
         if (_unreachable) return;
         _func->vpop();
         _func->vpop(); _func->vpop();
         uint32_t d1 = _func->alloc_vreg(types::v128); _func->vpush(d1);
         uint32_t d2 = _func->alloc_vreg(types::v128); _func->vpush(d2);
      }
      void ir_simd_splat() {
         if (_unreachable) return;
         _func->vpop();
         uint32_t d1 = _func->alloc_vreg(types::v128); _func->vpush(d1);
         uint32_t d2 = _func->alloc_vreg(types::v128); _func->vpush(d2);
      }
      void ir_simd_test() {
         if (_unreachable) return;
         _func->vpop(); _func->vpop();
         uint32_t d = _func->alloc_vreg(types::i32); _func->vpush(d);
      }
      void ir_bulk_mem3() {
         if (_unreachable) return;
         _func->vpop(); _func->vpop(); _func->vpop();
      }

      // ──── State ────
      growable_allocator& _allocator;
      std::size_t _source_bytes;
      module& _mod;
      void* _code_segment_base = nullptr; // Start of code region (from start_code)
      ir_function* _functions = nullptr;  // Per-function IR array
      uint32_t _num_functions = 0;
      ir_function* _func = nullptr;       // Current function being built
      bool _unreachable = false;
   };

}} // namespace eosio::vm
