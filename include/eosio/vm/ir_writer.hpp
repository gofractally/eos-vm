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
#ifdef __aarch64__
#include <eosio/vm/jit_codegen_a64.hpp>
#else
#include <eosio/vm/jit_codegen.hpp>
#endif
#include <eosio/vm/jit_ir.hpp>
#include <eosio/vm/jit_optimize.hpp>
#include <eosio/vm/jit_regalloc.hpp>
#include <eosio/vm/types.hpp>

#include <cstdint>
#include <cstring>
#include <exception>

namespace eosio { namespace vm {

   template<typename Context, bool StackLimitIsBytes>
   class ir_writer {
#ifdef __aarch64__
      using codegen_t = jit_codegen_a64<Context, StackLimitIsBytes>;
#else
      using codegen_t = jit_codegen<Context, StackLimitIsBytes>;
#endif
    public:
      // Branch/label types — dummy values since IR tracks control flow directly.
      // The parser stores and passes these between emit_if/emit_else/emit_end/emit_br
      // but never interprets them. fix_branch is a no-op.
      using branch_t = uint32_t;
      using label_t  = uint32_t;

      ir_writer(growable_allocator& alloc, std::size_t source_bytes, module& mod)
         : _allocator(alloc), _source_bytes(source_bytes), _mod(mod),
           _scratch(alloc) {
         _num_functions = mod.code.size();
         _functions = _scratch.alloc<ir_function>(_num_functions);
         for (uint32_t i = 0; i < _num_functions; ++i) {
            _functions[i] = ir_function{};
         }
      }

      ~ir_writer() {
         // If destructor runs during stack unwinding (parsing threw an exception),
         // skip compilation — some functions may not have been parsed.
         if (std::uncaught_exceptions() > 0) {
            return;
         }

         // Pass 2: Register allocation + code generation (fused per-function).
         // Processing each function's regalloc immediately before codegen keeps
         // the IR data hot in cache.
         codegen_t codegen(_allocator, _mod);
         codegen.emit_entry_and_error_handlers();
         for (uint32_t i = 0; i < _num_functions; ++i) {
            jit_optimizer::optimize(_functions[i], _scratch);
            jit_regalloc::compute_live_intervals(_functions[i], _scratch);
            jit_regalloc::allocate_registers(_functions[i]);
            codegen.compile_function(_functions[i], _mod.code[i]);
         }
         codegen.finalize_code();
         // Disarm the scratch allocator so its destructor won't restore the
         // watermark after we reset. Then reset to reclaim all pre-code memory.
         // The native code has been copied to the jit_allocator by end_code<true>().
         _scratch.disarm();
         _allocator.reset();
      }

      static constexpr uint32_t get_depth_for_type(uint8_t type) {
         if (type == types::v128) return 2;
         return type == types::pseudo ? 0 : 1;
      }

      // ──── Prologue / epilogue ────

      void emit_prologue(const func_type& ft, const std::vector<local_entry>& locals, uint32_t funcnum) {
         // Initialize IR for this function using the actual function body size
         _func = &_functions[funcnum];
         _func->init(_scratch, _mod.code[funcnum].size);
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
         entry.entered_unreachable = 0;
         entry.merge_vreg = ir_vreg_none;
         _func->ctrl_push(entry);
         _func->start_block(entry.block_idx);
      }

      void emit_epilogue(const func_type& /*ft*/, const std::vector<local_entry>& /*locals*/, uint32_t /*funcnum*/) {
         // Nothing to do — IR is complete after parsing
      }

      void finalize(function_body& /*body*/) {
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
         uint32_t block_idx = UINT32_MAX;
         if (_func->ctrl_stack_top > 0) {
            auto entry = _func->ctrl_pop();
            block_idx = entry.block_idx;

            // If we have a merge vreg from an if/else with result type,
            // emit a mov from the else-branch result to the merge vreg.
            // This must happen BEFORE end_block so that branches targeting
            // this block (e.g., else_'s forward branch) land AFTER the mov.
            if (entry.merge_vreg != ir_vreg_none) {
               if (!_unreachable && entry.result_type != types::pseudo &&
                   _func->vstack_depth() > entry.stack_depth) {
                  if (entry.result_type == types::v128) _func->vpop(); // extra v128 slot
                  uint32_t else_result = _func->vpop();
                  ir_inst mov{};
                  mov.opcode = ir_op::mov;
                  mov.type = entry.result_type;
                  mov.flags = IR_NONE;
                  mov.dest = entry.merge_vreg;
                  mov.rr.src1 = else_result;
                  mov.rr.src2 = ir_vreg_none;
                  _func->emit(mov);
               }
               _func->vstack_resize(entry.stack_depth);
               _func->vpush(entry.merge_vreg);
               if (entry.result_type == types::v128) {
                  uint32_t extra = _func->alloc_vreg(types::v128);
                  _func->vpush(extra);
               }
            } else if (_unreachable) {
               _func->vstack_resize(entry.stack_depth);
               if (entry.result_type != types::pseudo) {
                  uint32_t dest = _func->alloc_vreg(entry.result_type);
                  _func->vpush(dest);
                  if (entry.result_type == types::v128) {
                     uint32_t extra = _func->alloc_vreg(types::v128);
                     _func->vpush(extra);
                  }
               }
            }

            _func->end_block(entry.block_idx);
            // If the block was entered in unreachable code, it stays unreachable.
            // Only blocks entered reachably can make code reachable again.
            _unreachable = entry.entered_unreachable ? true : false;
         }
         return block_idx;
      }

      branch_t emit_return(uint32_t dc, uint8_t rt) {
         if (!_unreachable) {
            ir_inst inst{};
            inst.opcode = ir_op::return_;
            inst.type = rt;  // Store return type so codegen knows v128 vs scalar
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = ir_vreg_none;
            if (rt != types::pseudo && _func->vstack_depth() > 0) {
               if (rt == types::v128) _func->vpop(); // extra v128 slot
               inst.rr.src1 = _func->vpop();
            } else {
               inst.rr.src1 = ir_vreg_none;
            }
            inst.rr.src2 = ir_vreg_none;
            _func->emit(inst);
         }
         _unreachable = true;
         return UINT32_MAX; // return doesn't need branch fixup
      }

      void emit_block(uint8_t result_type = types::pseudo) {
         ir_control_entry entry{};
         entry.block_idx = _func->new_block();
         entry.stack_depth = _func->vstack_depth();
         entry.result_type = result_type;
         entry.is_loop = 0;
         entry.is_function = 0;
         entry.entered_unreachable = _unreachable ? 1 : 0;
         // Allocate merge vreg for blocks with result types (needed for br targets)
         if (result_type != types::pseudo) {
            entry.merge_vreg = _func->alloc_vreg(result_type);
         } else {
            entry.merge_vreg = ir_vreg_none;
         }
         _func->ctrl_push(entry);
      }

      label_t emit_loop(uint8_t result_type = types::pseudo) {
         ir_control_entry entry{};
         entry.block_idx = _func->new_block();
         _func->blocks[entry.block_idx].is_loop = 1;
         entry.stack_depth = _func->vstack_depth();
         entry.result_type = result_type;
         entry.is_loop = 1;
         entry.is_function = 0;
         entry.entered_unreachable = _unreachable ? 1 : 0;
         entry.merge_vreg = ir_vreg_none;
         _func->ctrl_push(entry);
         _func->start_block(entry.block_idx);
         return entry.block_idx;
      }

      branch_t emit_if(uint8_t result_type = types::pseudo) {
         ir_control_entry entry{};
         entry.block_idx = _func->new_block();
         _func->blocks[entry.block_idx].is_if = 1;
         entry.result_type = result_type;
         entry.is_loop = 0;
         entry.is_function = 0;
         entry.entered_unreachable = _unreachable ? 1 : 0;
         if (result_type != types::pseudo) {
            entry.merge_vreg = _func->alloc_vreg(result_type);
         } else {
            entry.merge_vreg = ir_vreg_none;
         }
         uint32_t inst_idx = UINT32_MAX;
         if (!_unreachable) {
            uint32_t cond = _func->vpop();
            entry.stack_depth = _func->vstack_depth(); // after popping condition
            ir_inst inst{};
            inst.opcode = ir_op::if_;
            inst.type = types::pseudo;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = ir_vreg_none;
            inst.br.src1 = cond;
            inst.br.target = entry.block_idx;  // default target (patched by fix_branch)
            inst_idx = _func->current_inst_index();
            _func->emit(inst);
         } else {
            entry.stack_depth = _func->vstack_depth();
         }
         _func->ctrl_push(entry);
         return inst_idx;
      }

      branch_t emit_else(branch_t if_inst_idx) {
         uint32_t else_inst_idx = UINT32_MAX;
         bool was_entered_unreachable = false;
         if (_func->ctrl_stack_top > 0) {
            auto& entry = _func->ctrl_back();
            was_entered_unreachable = entry.entered_unreachable;

            // If the block has a result type and the then-branch produced a value,
            // emit a mov to the merge vreg so both branches write the same destination.
            if (!_unreachable && entry.merge_vreg != ir_vreg_none &&
                _func->vstack_depth() > entry.stack_depth) {
               if (entry.result_type == types::v128) _func->vpop(); // extra v128 slot
               uint32_t then_result = _func->vpop();
               ir_inst mov{};
               mov.opcode = ir_op::mov;
               mov.type = entry.result_type;
               mov.flags = IR_NONE;
               mov.dest = entry.merge_vreg;
               mov.rr.src1 = then_result;
               mov.rr.src2 = ir_vreg_none;
               _func->emit(mov);
            }

            // Emit else_ instruction: then-block jumps to block end
            ir_inst inst{};
            inst.opcode = ir_op::else_;
            inst.type = types::pseudo;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = ir_vreg_none;
            inst.br.target = UINT32_MAX; // patched by fix_branch at end
            inst.br.src1 = ir_vreg_none;
            else_inst_idx = _func->current_inst_index();
            _func->emit(inst);
            // Patch the if_ instruction to branch HERE (else start)
            // We need a new block for the else, mark it with current inst position
            if (if_inst_idx < _func->inst_count) {
               // if_ should branch to the else-block, not the end-block
               // Create a new block for the else entry point
               uint32_t else_block = _func->new_block();
               _func->start_block(else_block);
               _func->insts[if_inst_idx].br.target = else_block;
            }
            // Clear is_if: the else handles the if_fixup pop, so the
            // block_end emitted by emit_end shouldn't pop again.
            if (entry.block_idx < _func->block_count) {
               _func->blocks[entry.block_idx].is_if = 0;
            }
            _func->vstack_resize(entry.stack_depth);
         }
         // Only become reachable if the if was entered in reachable code.
         // If the if was in dead code, the else body is also unreachable.
         _unreachable = was_entered_unreachable;
         return else_inst_idx;
      }

      branch_t emit_br(uint32_t dc, uint8_t rt, uint32_t label = UINT32_MAX) {
         uint32_t inst_idx = 0;
         if (!_unreachable) {
            // If the br carries a result value and the target block has a merge vreg,
            // emit a mov to copy the result to the merge vreg
            if (rt != types::pseudo && _func->vstack_depth() > 0 && label != UINT32_MAX) {
               uint32_t target_ctrl = _func->ctrl_stack_top - 1 - label;
               if (target_ctrl < _func->ctrl_stack_top) {
                  auto& target_entry = _func->ctrl_stack[target_ctrl];
                  if (target_entry.merge_vreg != ir_vreg_none && !target_entry.is_loop) {
                     // For v128, use the low vreg (vstack_top-2), not the high vreg
                     uint32_t src = (rt == types::v128 && _func->vstack_top >= 2)
                                    ? _func->vstack[_func->vstack_top - 2]
                                    : _func->vstack_back();
                     if (src != target_entry.merge_vreg) {
                        ir_inst mov{};
                        mov.opcode = ir_op::mov;
                        mov.type = rt;
                        mov.flags = IR_NONE;
                        mov.dest = target_entry.merge_vreg;
                        mov.rr.src1 = src;
                        mov.rr.src2 = ir_vreg_none;
                        _func->emit(mov);
                     }
                  }
               }
            }
            ir_inst inst{};
            inst.opcode = ir_op::br;
            inst.type = rt;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = dc; // depth change for multipop
            inst.br.target = UINT32_MAX; // patched by fix_branch
            if (rt != types::pseudo && _func->vstack_depth() > 0) {
               inst.br.src1 = (rt == types::v128 && _func->vstack_top >= 2)
                              ? _func->vstack[_func->vstack_top - 2]
                              : _func->vstack_back();
            } else {
               inst.br.src1 = ir_vreg_none;
            }
            inst_idx = _func->current_inst_index();
            _func->emit(inst);
         }
         _unreachable = true;
         return inst_idx;
      }

      branch_t emit_br_if(uint32_t dc, uint8_t rt, uint32_t label = UINT32_MAX) {
         uint32_t inst_idx = 0;
         if (!_unreachable) {
            uint32_t cond = _func->vpop();
            // If the br_if carries a result value and the target block has a merge vreg,
            // emit a mov to copy the result to the merge vreg.
            // This is safe because: if the branch is taken, the merge vreg is correct.
            // If not taken, the merge vreg will be overwritten by the fall-through path.
            if (rt != types::pseudo && _func->vstack_depth() > 0 && label != UINT32_MAX) {
               uint32_t target_ctrl = _func->ctrl_stack_top - 1 - label;
               if (target_ctrl < _func->ctrl_stack_top) {
                  auto& target_entry = _func->ctrl_stack[target_ctrl];
                  if (target_entry.merge_vreg != ir_vreg_none && !target_entry.is_loop) {
                     uint32_t src = (rt == types::v128 && _func->vstack_top >= 2)
                                    ? _func->vstack[_func->vstack_top - 2]
                                    : _func->vstack_back();
                     if (src != target_entry.merge_vreg) {
                        ir_inst mov{};
                        mov.opcode = ir_op::mov;
                        mov.type = rt;
                        mov.flags = IR_NONE;
                        mov.dest = target_entry.merge_vreg;
                        mov.rr.src1 = src;
                        mov.rr.src2 = ir_vreg_none;
                        _func->emit(mov);
                     }
                  }
               }
            }
            ir_inst inst{};
            inst.opcode = ir_op::br_if;
            inst.type = rt;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = dc; // depth change for multipop
            inst.br.target = UINT32_MAX; // patched by fix_branch
            inst.br.src1 = cond;
            inst_idx = _func->current_inst_index();
            _func->emit(inst);
         }
         return inst_idx;
      }

      struct br_table_parser {
         ir_writer* _writer;
         br_table_parser(ir_writer* w) : _writer(w) {}
         branch_t emit_case(uint32_t dc, uint8_t rt, uint32_t label = UINT32_MAX) {
            // Emit merge mov if target has a merge vreg
            if (rt != types::pseudo && label != UINT32_MAX) {
               auto* func = _writer->_func;
               uint32_t target_ctrl = func->ctrl_stack_top - 1 - label;
               if (target_ctrl < func->ctrl_stack_top) {
                  auto& target_entry = func->ctrl_stack[target_ctrl];
                  if (target_entry.merge_vreg != ir_vreg_none && !target_entry.is_loop &&
                      func->vstack_depth() > 0) {
                     uint32_t src = (rt == types::v128 && func->vstack_top >= 2)
                                    ? func->vstack[func->vstack_top - 2]
                                    : func->vstack_back();
                     if (src != target_entry.merge_vreg) {
                        ir_inst mov{};
                        mov.opcode = ir_op::mov;
                        mov.type = rt;
                        mov.flags = IR_NONE;
                        mov.dest = target_entry.merge_vreg;
                        mov.rr.src1 = src;
                        mov.rr.src2 = ir_vreg_none;
                        func->emit(mov);
                     }
                  }
               }
            }
            ir_inst inst{};
            inst.opcode = ir_op::br;
            inst.type = rt;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = dc;
            inst.br.target = UINT32_MAX; // patched by fix_branch
            inst.br.src1 = ir_vreg_none;
            uint32_t inst_idx = _writer->_func->current_inst_index();
            _writer->_func->emit(inst);
            return inst_idx;
         }
         branch_t emit_default(uint32_t dc, uint8_t rt, uint32_t label = UINT32_MAX) {
            return emit_case(dc, rt, label);
         }
      };
      br_table_parser emit_br_table(uint32_t table_size) {
         if (!_unreachable) {
            uint32_t idx = _func->vpop();
            ir_inst inst{};
            inst.opcode = ir_op::br_table;
            inst.type = types::pseudo;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = table_size;  // store case count for jit_codegen
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
            uint32_t nparams = ft.param_types.size();
            // Emit arg instructions for each parameter (needed by register mode
            // to push vreg values to the native stack before the call)
            // Pop params in reverse to get them in correct stack order
            uint32_t param_vregs[64]; // bounded by max params
            for (uint32_t i = 0; i < nparams; ++i) {
               uint32_t pi = nparams - 1 - i;
               if (ft.param_types[pi] == types::v128) _func->vpop(); // extra v128 slot
               param_vregs[pi] = _func->vpop();
            }
            for (uint32_t i = 0; i < nparams; ++i) {
               ir_inst arg{};
               arg.opcode = ir_op::arg;
               arg.type = types::pseudo;
               arg.flags = IR_NONE;
               arg.dest = ir_vreg_none;
               arg.rr.src1 = param_vregs[i];
               arg.rr.src2 = ir_vreg_none;
               _func->emit(arg);
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
               if (ft.return_type == types::v128) {
                  uint32_t dest2 = _func->alloc_vreg(ft.return_type);
                  _func->vpush(dest2);
               }
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
            uint32_t nparams = ft.param_types.size();
            // Emit arg instructions for each parameter
            uint32_t param_vregs[64];
            for (uint32_t i = 0; i < nparams; ++i) {
               uint32_t pi = nparams - 1 - i;
               if (ft.param_types[pi] == types::v128) _func->vpop(); // extra v128 slot
               param_vregs[pi] = _func->vpop();
            }
            for (uint32_t i = 0; i < nparams; ++i) {
               ir_inst arg{};
               arg.opcode = ir_op::arg;
               arg.type = types::pseudo;
               arg.flags = IR_NONE;
               arg.dest = ir_vreg_none;
               arg.rr.src1 = param_vregs[i];
               arg.rr.src2 = ir_vreg_none;
               _func->emit(arg);
            }
            // Emit arg for table index (must be on top of stack for call_indirect)
            {
               ir_inst arg{};
               arg.opcode = ir_op::arg;
               arg.type = types::pseudo;
               arg.flags = IR_NONE;
               arg.dest = ir_vreg_none;
               arg.rr.src1 = table_idx;
               arg.rr.src2 = ir_vreg_none;
               _func->emit(arg);
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
               if (ft.return_type == types::v128) {
                  uint32_t dest2 = _func->alloc_vreg(ft.return_type);
                  _func->vpush(dest2);
               }
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
            if (type == types::v128) {
               uint32_t low_vreg = _func->vpop(); // low vreg of v128
               // Emit a drop IR so codegen can pop 16 bytes from x86 stack if needed
               ir_inst inst{};
               inst.opcode = ir_op::drop;
               inst.type = types::v128;
               inst.flags = IR_NONE;
               inst.dest = ir_vreg_none;
               inst.rr.src1 = low_vreg; // track which vreg is dropped
               inst.rr.src2 = ir_vreg_none;
               _func->emit(inst);
            }
         }
      }

      void emit_select(uint8_t type) {
         if (!_unreachable) {
            uint32_t cond = _func->vpop();
            if (type == types::v128) _func->vpop(); // extra v128 slot for val2
            uint32_t val2 = _func->vpop();
            if (type == types::v128) _func->vpop(); // extra v128 slot for val1
            uint32_t val1 = _func->vpop();
            uint32_t dest = _func->alloc_vreg(type);
            ir_inst inst{};
            inst.opcode = ir_op::select;
            inst.type = type;
            inst.dest = dest;
            inst.sel.val1 = static_cast<uint16_t>(val1);
            inst.sel.val2 = static_cast<uint16_t>(val2);
            inst.sel.cond = static_cast<uint16_t>(cond);
            _func->emit(inst);
            _func->vpush(dest);
            if (type == types::v128) {
               uint32_t dest2 = _func->alloc_vreg(type);
               _func->vpush(dest2);
            }
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
            if (ty == types::v128) {
               uint32_t dest2 = _func->alloc_vreg(ty);
               _func->vpush(dest2);
            }
         }
      }

      void emit_set_local(uint32_t li, uint8_t ty) {
         if (!_unreachable) {
            if (ty == types::v128) _func->vpop(); // extra v128 slot
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
            // For v128, peek the low slot (2nd from top since v128 uses 2 slots)
            uint32_t src = (ty == types::v128 && _func->vstack_depth() >= 2)
                           ? _func->vstack[_func->vstack_top - 2]
                           : _func->vstack_back();
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
            uint8_t gtype = _mod.globals.at(gi).type.content_type;
            uint32_t dest = _func->alloc_vreg(gtype);
            ir_inst inst{};
            inst.opcode = ir_op::global_get;
            inst.type = gtype;
            inst.dest = dest;
            inst.local.index = gi;
            inst.local.src1 = ir_vreg_none;
            _func->emit(inst);
            _func->vpush(dest);
            if (gtype == types::v128) {
               _func->has_simd = true;
               uint32_t dest2 = _func->alloc_vreg(gtype);
               _func->vpush(dest2);
            }
         }
      }

      void emit_set_global(uint32_t gi) {
         if (!_unreachable) {
            uint8_t gtype = _mod.globals.at(gi).type.content_type;
            if (gtype == types::v128) _func->vpop(); // extra v128 slot
            uint32_t src = _func->vpop();
            ir_inst inst{};
            inst.opcode = ir_op::global_set;
            inst.type = gtype;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = ir_vreg_none;
            inst.local.index = gi;
            inst.local.src1 = src;
            _func->emit(inst);
         }
      }

      // ──── Memory loads ────
      // Parser passes (alignment, offset) — we only need offset for IR
      void emit_i32_load(uint32_t /*align*/, uint32_t offset)    { ir_load(ir_op::i32_load, types::i32, offset); }
      void emit_i64_load(uint32_t /*align*/, uint32_t offset)    { ir_load(ir_op::i64_load, types::i64, offset); }
      void emit_f32_load(uint32_t /*align*/, uint32_t offset)    { ir_load(ir_op::f32_load, types::f32, offset); }
      void emit_f64_load(uint32_t /*align*/, uint32_t offset)    { ir_load(ir_op::f64_load, types::f64, offset); }
      void emit_i32_load8_s(uint32_t /*a*/, uint32_t o) { ir_load(ir_op::i32_load8_s, types::i32, o); }
      void emit_i32_load16_s(uint32_t /*a*/, uint32_t o){ ir_load(ir_op::i32_load16_s, types::i32, o); }
      void emit_i32_load8_u(uint32_t /*a*/, uint32_t o) { ir_load(ir_op::i32_load8_u, types::i32, o); }
      void emit_i32_load16_u(uint32_t /*a*/, uint32_t o){ ir_load(ir_op::i32_load16_u, types::i32, o); }
      void emit_i64_load8_s(uint32_t /*a*/, uint32_t o) { ir_load(ir_op::i64_load8_s, types::i64, o); }
      void emit_i64_load16_s(uint32_t /*a*/, uint32_t o){ ir_load(ir_op::i64_load16_s, types::i64, o); }
      void emit_i64_load32_s(uint32_t /*a*/, uint32_t o){ ir_load(ir_op::i64_load32_s, types::i64, o); }
      void emit_i64_load8_u(uint32_t /*a*/, uint32_t o) { ir_load(ir_op::i64_load8_u, types::i64, o); }
      void emit_i64_load16_u(uint32_t /*a*/, uint32_t o){ ir_load(ir_op::i64_load16_u, types::i64, o); }
      void emit_i64_load32_u(uint32_t /*a*/, uint32_t o){ ir_load(ir_op::i64_load32_u, types::i64, o); }

      // ──── Memory stores ────
      void emit_i32_store(uint32_t /*a*/, uint32_t o)   { ir_store(ir_op::i32_store, types::i32, o); }
      void emit_i64_store(uint32_t /*a*/, uint32_t o)   { ir_store(ir_op::i64_store, types::i64, o); }
      void emit_f32_store(uint32_t /*a*/, uint32_t o)   { ir_store(ir_op::f32_store, types::f32, o); }
      void emit_f64_store(uint32_t /*a*/, uint32_t o)   { ir_store(ir_op::f64_store, types::f64, o); }
      void emit_i32_store8(uint32_t /*a*/, uint32_t o)  { ir_store(ir_op::i32_store8, types::i32, o); }
      void emit_i32_store16(uint32_t /*a*/, uint32_t o) { ir_store(ir_op::i32_store16, types::i32, o); }
      void emit_i64_store8(uint32_t /*a*/, uint32_t o)  { ir_store(ir_op::i64_store8, types::i64, o); }
      void emit_i64_store16(uint32_t /*a*/, uint32_t o) { ir_store(ir_op::i64_store16, types::i64, o); }
      void emit_i64_store32(uint32_t /*a*/, uint32_t o) { ir_store(ir_op::i64_store32, types::i64, o); }

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

      // ──── SIMD (vstack tracking + v128_op IR emission) ────
      void emit_v128_load(uint32_t /*align*/, uint32_t offset)  { ir_simd_load(simd_sub::v128_load, offset); }
      void emit_v128_load8x8_s(uint32_t /*align*/, uint32_t offset)  { ir_simd_load(simd_sub::v128_load8x8_s, offset); }
      void emit_v128_load8x8_u(uint32_t /*align*/, uint32_t offset)  { ir_simd_load(simd_sub::v128_load8x8_u, offset); }
      void emit_v128_load16x4_s(uint32_t /*align*/, uint32_t offset) { ir_simd_load(simd_sub::v128_load16x4_s, offset); }
      void emit_v128_load16x4_u(uint32_t /*align*/, uint32_t offset) { ir_simd_load(simd_sub::v128_load16x4_u, offset); }
      void emit_v128_load32x2_s(uint32_t /*align*/, uint32_t offset) { ir_simd_load(simd_sub::v128_load32x2_s, offset); }
      void emit_v128_load32x2_u(uint32_t /*align*/, uint32_t offset) { ir_simd_load(simd_sub::v128_load32x2_u, offset); }
      void emit_v128_load8_splat(uint32_t /*align*/, uint32_t offset) { ir_simd_load(simd_sub::v128_load8_splat, offset); }
      void emit_v128_load16_splat(uint32_t /*align*/, uint32_t offset) { ir_simd_load(simd_sub::v128_load16_splat, offset); }
      void emit_v128_load32_splat(uint32_t /*align*/, uint32_t offset) { ir_simd_load(simd_sub::v128_load32_splat, offset); }
      void emit_v128_load64_splat(uint32_t /*align*/, uint32_t offset) { ir_simd_load(simd_sub::v128_load64_splat, offset); }
      void emit_v128_load32_zero(uint32_t /*align*/, uint32_t offset)  { ir_simd_load(simd_sub::v128_load32_zero, offset); }
      void emit_v128_load64_zero(uint32_t /*align*/, uint32_t offset)  { ir_simd_load(simd_sub::v128_load64_zero, offset); }
      void emit_v128_store(uint32_t /*align*/, uint32_t offset) { ir_simd_store(simd_sub::v128_store, offset); }
      void emit_v128_load8_lane(uint32_t /*align*/, uint32_t offset, uint8_t lane)  { ir_simd_load_lane(simd_sub::v128_load8_lane, offset, lane); }
      void emit_v128_load16_lane(uint32_t /*align*/, uint32_t offset, uint8_t lane) { ir_simd_load_lane(simd_sub::v128_load16_lane, offset, lane); }
      void emit_v128_load32_lane(uint32_t /*align*/, uint32_t offset, uint8_t lane) { ir_simd_load_lane(simd_sub::v128_load32_lane, offset, lane); }
      void emit_v128_load64_lane(uint32_t /*align*/, uint32_t offset, uint8_t lane) { ir_simd_load_lane(simd_sub::v128_load64_lane, offset, lane); }
      void emit_v128_store8_lane(uint32_t /*align*/, uint32_t offset, uint8_t lane)  { ir_simd_store_lane(simd_sub::v128_store8_lane, offset, lane); }
      void emit_v128_store16_lane(uint32_t /*align*/, uint32_t offset, uint8_t lane) { ir_simd_store_lane(simd_sub::v128_store16_lane, offset, lane); }
      void emit_v128_store32_lane(uint32_t /*align*/, uint32_t offset, uint8_t lane) { ir_simd_store_lane(simd_sub::v128_store32_lane, offset, lane); }
      void emit_v128_store64_lane(uint32_t /*align*/, uint32_t offset, uint8_t lane) { ir_simd_store_lane(simd_sub::v128_store64_lane, offset, lane); }

      void emit_v128_const(v128_t v) {
         if (!_unreachable) {
            uint32_t dest = _func->alloc_vreg(types::v128);
            uint32_t dest2 = _func->alloc_vreg(types::v128);
            ir_inst inst{};
            inst.opcode = ir_op::const_v128;
            inst.type = types::v128;
            inst.flags = IR_NONE;
            inst.dest = dest; // v128 dest vreg for XMM register allocation
            inst.immv128 = v;
            _func->emit(inst);
            _func->vpush(dest);
            _func->vpush(dest2);
         }
      }
      void emit_i8x16_shuffle(const uint8_t* l) {
         if (!_unreachable) {
            // Pop two v128 operands (4 vregs): src2 (TOS), then src1
            _func->vpop(); uint32_t s2 = _func->vpop();
            _func->vpop(); uint32_t s1 = _func->vpop();
            // Emit arg instructions to push v128 sources from XMM/spill to x86 stack.
            // shuffle's immv128 overlaps the simd struct, so we can't store vregs there.
            // Push src1 first (NOS), then src2 (TOS) — matching stack-based layout.
            auto emit_v128_arg = [&](uint32_t vreg) {
               ir_inst arg{};
               arg.opcode = ir_op::arg;
               arg.type = types::v128;
               arg.flags = IR_NONE;
               arg.rr.src1 = vreg;
               arg.dest = ir_vreg_none;
               _func->emit(arg);
            };
            emit_v128_arg(s1);
            emit_v128_arg(s2);
            // Emit v128_op with shuffle sub-opcode, lanes stored in immv128
            uint32_t d1 = _func->alloc_vreg(types::v128);
            uint32_t d2 = _func->alloc_vreg(types::v128);
            ir_inst inst{};
            inst.opcode = ir_op::v128_op;
            inst.type = types::v128;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = static_cast<uint32_t>(simd_sub::i8x16_shuffle);
            std::memcpy(&inst.immv128, l, 16);
            _func->emit(inst);
            // Emit a nop with dest=d1 so regalloc creates a v128 interval for the
            // shuffle result (the shuffle instruction itself can't store v_dest
            // because immv128 overlaps the simd struct).
            {
               ir_inst marker{};
               marker.opcode = ir_op::nop;
               marker.type = types::v128;
               marker.flags = IR_NONE;
               marker.dest = d1;
               _func->emit(marker);
            }
            _func->vpush(d1); _func->vpush(d2);
         }
      }

// SIMD ops: track vstack (2 slots per v128) AND emit v128_op IR instructions.
// The sub-opcode is derived from the function name suffix matching simd_sub enum.
#define SIMD_UNOP(name)  void name() { ir_simd_unop(name##_sub); }
#define SIMD_BINOP(name) void name() { ir_simd_binop(name##_sub); }
#define SIMD_SHIFT(name) void name() { ir_simd_shift(name##_sub); }
#define SIMD_EXTRACT(name) void name(uint8_t a) { ir_simd_extract(name##_sub, a); }
#define SIMD_REPLACE(name) void name(uint8_t a) { ir_simd_replace(name##_sub, a); }
#define SIMD_SPLAT(name)   void name() { ir_simd_splat(name##_sub); }
#define SIMD_RELOP(name)   void name() { ir_simd_binop(name##_sub); }

// Map emit_xxx function names to simd_sub::xxx enum values.
// The _sub suffix is appended by the macro to form the enum name.
// We define bridging constants to handle the emit_ prefix stripping.
      static constexpr auto emit_i8x16_extract_lane_s_sub = simd_sub::i8x16_extract_lane_s;
      static constexpr auto emit_i8x16_extract_lane_u_sub = simd_sub::i8x16_extract_lane_u;
      static constexpr auto emit_i8x16_replace_lane_sub = simd_sub::i8x16_replace_lane;
      static constexpr auto emit_i16x8_extract_lane_s_sub = simd_sub::i16x8_extract_lane_s;
      static constexpr auto emit_i16x8_extract_lane_u_sub = simd_sub::i16x8_extract_lane_u;
      static constexpr auto emit_i16x8_replace_lane_sub = simd_sub::i16x8_replace_lane;
      static constexpr auto emit_i32x4_extract_lane_sub = simd_sub::i32x4_extract_lane;
      static constexpr auto emit_i32x4_replace_lane_sub = simd_sub::i32x4_replace_lane;
      static constexpr auto emit_i64x2_extract_lane_sub = simd_sub::i64x2_extract_lane;
      static constexpr auto emit_i64x2_replace_lane_sub = simd_sub::i64x2_replace_lane;
      static constexpr auto emit_f32x4_extract_lane_sub = simd_sub::f32x4_extract_lane;
      static constexpr auto emit_f32x4_replace_lane_sub = simd_sub::f32x4_replace_lane;
      static constexpr auto emit_f64x2_extract_lane_sub = simd_sub::f64x2_extract_lane;
      static constexpr auto emit_f64x2_replace_lane_sub = simd_sub::f64x2_replace_lane;
      static constexpr auto emit_i8x16_swizzle_sub = simd_sub::i8x16_swizzle;
      static constexpr auto emit_i8x16_splat_sub = simd_sub::i8x16_splat;
      static constexpr auto emit_i16x8_splat_sub = simd_sub::i16x8_splat;
      static constexpr auto emit_i32x4_splat_sub = simd_sub::i32x4_splat;
      static constexpr auto emit_i64x2_splat_sub = simd_sub::i64x2_splat;
      static constexpr auto emit_f32x4_splat_sub = simd_sub::f32x4_splat;
      static constexpr auto emit_f64x2_splat_sub = simd_sub::f64x2_splat;
      static constexpr auto emit_i8x16_eq_sub = simd_sub::i8x16_eq;
      static constexpr auto emit_i8x16_ne_sub = simd_sub::i8x16_ne;
      static constexpr auto emit_i8x16_lt_s_sub = simd_sub::i8x16_lt_s;
      static constexpr auto emit_i8x16_lt_u_sub = simd_sub::i8x16_lt_u;
      static constexpr auto emit_i8x16_gt_s_sub = simd_sub::i8x16_gt_s;
      static constexpr auto emit_i8x16_gt_u_sub = simd_sub::i8x16_gt_u;
      static constexpr auto emit_i8x16_le_s_sub = simd_sub::i8x16_le_s;
      static constexpr auto emit_i8x16_le_u_sub = simd_sub::i8x16_le_u;
      static constexpr auto emit_i8x16_ge_s_sub = simd_sub::i8x16_ge_s;
      static constexpr auto emit_i8x16_ge_u_sub = simd_sub::i8x16_ge_u;
      static constexpr auto emit_i16x8_eq_sub = simd_sub::i16x8_eq;
      static constexpr auto emit_i16x8_ne_sub = simd_sub::i16x8_ne;
      static constexpr auto emit_i16x8_lt_s_sub = simd_sub::i16x8_lt_s;
      static constexpr auto emit_i16x8_lt_u_sub = simd_sub::i16x8_lt_u;
      static constexpr auto emit_i16x8_gt_s_sub = simd_sub::i16x8_gt_s;
      static constexpr auto emit_i16x8_gt_u_sub = simd_sub::i16x8_gt_u;
      static constexpr auto emit_i16x8_le_s_sub = simd_sub::i16x8_le_s;
      static constexpr auto emit_i16x8_le_u_sub = simd_sub::i16x8_le_u;
      static constexpr auto emit_i16x8_ge_s_sub = simd_sub::i16x8_ge_s;
      static constexpr auto emit_i16x8_ge_u_sub = simd_sub::i16x8_ge_u;
      static constexpr auto emit_i32x4_eq_sub = simd_sub::i32x4_eq;
      static constexpr auto emit_i32x4_ne_sub = simd_sub::i32x4_ne;
      static constexpr auto emit_i32x4_lt_s_sub = simd_sub::i32x4_lt_s;
      static constexpr auto emit_i32x4_lt_u_sub = simd_sub::i32x4_lt_u;
      static constexpr auto emit_i32x4_gt_s_sub = simd_sub::i32x4_gt_s;
      static constexpr auto emit_i32x4_gt_u_sub = simd_sub::i32x4_gt_u;
      static constexpr auto emit_i32x4_le_s_sub = simd_sub::i32x4_le_s;
      static constexpr auto emit_i32x4_le_u_sub = simd_sub::i32x4_le_u;
      static constexpr auto emit_i32x4_ge_s_sub = simd_sub::i32x4_ge_s;
      static constexpr auto emit_i32x4_ge_u_sub = simd_sub::i32x4_ge_u;
      static constexpr auto emit_i64x2_eq_sub = simd_sub::i64x2_eq;
      static constexpr auto emit_i64x2_ne_sub = simd_sub::i64x2_ne;
      static constexpr auto emit_i64x2_lt_s_sub = simd_sub::i64x2_lt_s;
      static constexpr auto emit_i64x2_gt_s_sub = simd_sub::i64x2_gt_s;
      static constexpr auto emit_i64x2_le_s_sub = simd_sub::i64x2_le_s;
      static constexpr auto emit_i64x2_ge_s_sub = simd_sub::i64x2_ge_s;
      static constexpr auto emit_f32x4_eq_sub = simd_sub::f32x4_eq;
      static constexpr auto emit_f32x4_ne_sub = simd_sub::f32x4_ne;
      static constexpr auto emit_f32x4_lt_sub = simd_sub::f32x4_lt;
      static constexpr auto emit_f32x4_gt_sub = simd_sub::f32x4_gt;
      static constexpr auto emit_f32x4_le_sub = simd_sub::f32x4_le;
      static constexpr auto emit_f32x4_ge_sub = simd_sub::f32x4_ge;
      static constexpr auto emit_f64x2_eq_sub = simd_sub::f64x2_eq;
      static constexpr auto emit_f64x2_ne_sub = simd_sub::f64x2_ne;
      static constexpr auto emit_f64x2_lt_sub = simd_sub::f64x2_lt;
      static constexpr auto emit_f64x2_gt_sub = simd_sub::f64x2_gt;
      static constexpr auto emit_f64x2_le_sub = simd_sub::f64x2_le;
      static constexpr auto emit_f64x2_ge_sub = simd_sub::f64x2_ge;
      static constexpr auto emit_v128_not_sub = simd_sub::v128_not;
      static constexpr auto emit_v128_and_sub = simd_sub::v128_and;
      static constexpr auto emit_v128_andnot_sub = simd_sub::v128_andnot;
      static constexpr auto emit_v128_or_sub = simd_sub::v128_or;
      static constexpr auto emit_v128_xor_sub = simd_sub::v128_xor;
      static constexpr auto emit_i8x16_abs_sub = simd_sub::i8x16_abs;
      static constexpr auto emit_i8x16_neg_sub = simd_sub::i8x16_neg;
      static constexpr auto emit_i8x16_popcnt_sub = simd_sub::i8x16_popcnt;
      static constexpr auto emit_i8x16_narrow_i16x8_s_sub = simd_sub::i8x16_narrow_i16x8_s;
      static constexpr auto emit_i8x16_narrow_i16x8_u_sub = simd_sub::i8x16_narrow_i16x8_u;
      static constexpr auto emit_i8x16_shl_sub = simd_sub::i8x16_shl;
      static constexpr auto emit_i8x16_shr_s_sub = simd_sub::i8x16_shr_s;
      static constexpr auto emit_i8x16_shr_u_sub = simd_sub::i8x16_shr_u;
      static constexpr auto emit_i8x16_add_sub = simd_sub::i8x16_add;
      static constexpr auto emit_i8x16_add_sat_s_sub = simd_sub::i8x16_add_sat_s;
      static constexpr auto emit_i8x16_add_sat_u_sub = simd_sub::i8x16_add_sat_u;
      static constexpr auto emit_i8x16_sub_sub = simd_sub::i8x16_sub;
      static constexpr auto emit_i8x16_sub_sat_s_sub = simd_sub::i8x16_sub_sat_s;
      static constexpr auto emit_i8x16_sub_sat_u_sub = simd_sub::i8x16_sub_sat_u;
      static constexpr auto emit_i8x16_min_s_sub = simd_sub::i8x16_min_s;
      static constexpr auto emit_i8x16_min_u_sub = simd_sub::i8x16_min_u;
      static constexpr auto emit_i8x16_max_s_sub = simd_sub::i8x16_max_s;
      static constexpr auto emit_i8x16_max_u_sub = simd_sub::i8x16_max_u;
      static constexpr auto emit_i8x16_avgr_u_sub = simd_sub::i8x16_avgr_u;
      static constexpr auto emit_i16x8_extadd_pairwise_i8x16_s_sub = simd_sub::i16x8_extadd_pairwise_i8x16_s;
      static constexpr auto emit_i16x8_extadd_pairwise_i8x16_u_sub = simd_sub::i16x8_extadd_pairwise_i8x16_u;
      static constexpr auto emit_i16x8_abs_sub = simd_sub::i16x8_abs;
      static constexpr auto emit_i16x8_neg_sub = simd_sub::i16x8_neg;
      static constexpr auto emit_i16x8_q15mulr_sat_s_sub = simd_sub::i16x8_q15mulr_sat_s;
      static constexpr auto emit_i16x8_narrow_i32x4_s_sub = simd_sub::i16x8_narrow_i32x4_s;
      static constexpr auto emit_i16x8_narrow_i32x4_u_sub = simd_sub::i16x8_narrow_i32x4_u;
      static constexpr auto emit_i16x8_extend_low_i8x16_s_sub = simd_sub::i16x8_extend_low_i8x16_s;
      static constexpr auto emit_i16x8_extend_high_i8x16_s_sub = simd_sub::i16x8_extend_high_i8x16_s;
      static constexpr auto emit_i16x8_extend_low_i8x16_u_sub = simd_sub::i16x8_extend_low_i8x16_u;
      static constexpr auto emit_i16x8_extend_high_i8x16_u_sub = simd_sub::i16x8_extend_high_i8x16_u;
      static constexpr auto emit_i16x8_shl_sub = simd_sub::i16x8_shl;
      static constexpr auto emit_i16x8_shr_s_sub = simd_sub::i16x8_shr_s;
      static constexpr auto emit_i16x8_shr_u_sub = simd_sub::i16x8_shr_u;
      static constexpr auto emit_i16x8_add_sub = simd_sub::i16x8_add;
      static constexpr auto emit_i16x8_add_sat_s_sub = simd_sub::i16x8_add_sat_s;
      static constexpr auto emit_i16x8_add_sat_u_sub = simd_sub::i16x8_add_sat_u;
      static constexpr auto emit_i16x8_sub_sub = simd_sub::i16x8_sub;
      static constexpr auto emit_i16x8_sub_sat_s_sub = simd_sub::i16x8_sub_sat_s;
      static constexpr auto emit_i16x8_sub_sat_u_sub = simd_sub::i16x8_sub_sat_u;
      static constexpr auto emit_i16x8_mul_sub = simd_sub::i16x8_mul;
      static constexpr auto emit_i16x8_min_s_sub = simd_sub::i16x8_min_s;
      static constexpr auto emit_i16x8_min_u_sub = simd_sub::i16x8_min_u;
      static constexpr auto emit_i16x8_max_s_sub = simd_sub::i16x8_max_s;
      static constexpr auto emit_i16x8_max_u_sub = simd_sub::i16x8_max_u;
      static constexpr auto emit_i16x8_avgr_u_sub = simd_sub::i16x8_avgr_u;
      static constexpr auto emit_i16x8_extmul_low_i8x16_s_sub = simd_sub::i16x8_extmul_low_i8x16_s;
      static constexpr auto emit_i16x8_extmul_high_i8x16_s_sub = simd_sub::i16x8_extmul_high_i8x16_s;
      static constexpr auto emit_i16x8_extmul_low_i8x16_u_sub = simd_sub::i16x8_extmul_low_i8x16_u;
      static constexpr auto emit_i16x8_extmul_high_i8x16_u_sub = simd_sub::i16x8_extmul_high_i8x16_u;
      static constexpr auto emit_i32x4_extadd_pairwise_i16x8_s_sub = simd_sub::i32x4_extadd_pairwise_i16x8_s;
      static constexpr auto emit_i32x4_extadd_pairwise_i16x8_u_sub = simd_sub::i32x4_extadd_pairwise_i16x8_u;
      static constexpr auto emit_i32x4_abs_sub = simd_sub::i32x4_abs;
      static constexpr auto emit_i32x4_neg_sub = simd_sub::i32x4_neg;
      static constexpr auto emit_i32x4_extend_low_i16x8_s_sub = simd_sub::i32x4_extend_low_i16x8_s;
      static constexpr auto emit_i32x4_extend_high_i16x8_s_sub = simd_sub::i32x4_extend_high_i16x8_s;
      static constexpr auto emit_i32x4_extend_low_i16x8_u_sub = simd_sub::i32x4_extend_low_i16x8_u;
      static constexpr auto emit_i32x4_extend_high_i16x8_u_sub = simd_sub::i32x4_extend_high_i16x8_u;
      static constexpr auto emit_i32x4_shl_sub = simd_sub::i32x4_shl;
      static constexpr auto emit_i32x4_shr_s_sub = simd_sub::i32x4_shr_s;
      static constexpr auto emit_i32x4_shr_u_sub = simd_sub::i32x4_shr_u;
      static constexpr auto emit_i32x4_add_sub = simd_sub::i32x4_add;
      static constexpr auto emit_i32x4_sub_sub = simd_sub::i32x4_sub;
      static constexpr auto emit_i32x4_mul_sub = simd_sub::i32x4_mul;
      static constexpr auto emit_i32x4_min_s_sub = simd_sub::i32x4_min_s;
      static constexpr auto emit_i32x4_min_u_sub = simd_sub::i32x4_min_u;
      static constexpr auto emit_i32x4_max_s_sub = simd_sub::i32x4_max_s;
      static constexpr auto emit_i32x4_max_u_sub = simd_sub::i32x4_max_u;
      static constexpr auto emit_i32x4_dot_i16x8_s_sub = simd_sub::i32x4_dot_i16x8_s;
      static constexpr auto emit_i32x4_extmul_low_i16x8_s_sub = simd_sub::i32x4_extmul_low_i16x8_s;
      static constexpr auto emit_i32x4_extmul_high_i16x8_s_sub = simd_sub::i32x4_extmul_high_i16x8_s;
      static constexpr auto emit_i32x4_extmul_low_i16x8_u_sub = simd_sub::i32x4_extmul_low_i16x8_u;
      static constexpr auto emit_i32x4_extmul_high_i16x8_u_sub = simd_sub::i32x4_extmul_high_i16x8_u;
      static constexpr auto emit_i64x2_abs_sub = simd_sub::i64x2_abs;
      static constexpr auto emit_i64x2_neg_sub = simd_sub::i64x2_neg;
      static constexpr auto emit_i64x2_extend_low_i32x4_s_sub = simd_sub::i64x2_extend_low_i32x4_s;
      static constexpr auto emit_i64x2_extend_high_i32x4_s_sub = simd_sub::i64x2_extend_high_i32x4_s;
      static constexpr auto emit_i64x2_extend_low_i32x4_u_sub = simd_sub::i64x2_extend_low_i32x4_u;
      static constexpr auto emit_i64x2_extend_high_i32x4_u_sub = simd_sub::i64x2_extend_high_i32x4_u;
      static constexpr auto emit_i64x2_shl_sub = simd_sub::i64x2_shl;
      static constexpr auto emit_i64x2_shr_s_sub = simd_sub::i64x2_shr_s;
      static constexpr auto emit_i64x2_shr_u_sub = simd_sub::i64x2_shr_u;
      static constexpr auto emit_i64x2_add_sub = simd_sub::i64x2_add;
      static constexpr auto emit_i64x2_sub_sub = simd_sub::i64x2_sub;
      static constexpr auto emit_i64x2_mul_sub = simd_sub::i64x2_mul;
      static constexpr auto emit_i64x2_extmul_low_i32x4_s_sub = simd_sub::i64x2_extmul_low_i32x4_s;
      static constexpr auto emit_i64x2_extmul_high_i32x4_s_sub = simd_sub::i64x2_extmul_high_i32x4_s;
      static constexpr auto emit_i64x2_extmul_low_i32x4_u_sub = simd_sub::i64x2_extmul_low_i32x4_u;
      static constexpr auto emit_i64x2_extmul_high_i32x4_u_sub = simd_sub::i64x2_extmul_high_i32x4_u;
      static constexpr auto emit_f32x4_ceil_sub = simd_sub::f32x4_ceil;
      static constexpr auto emit_f32x4_floor_sub = simd_sub::f32x4_floor;
      static constexpr auto emit_f32x4_trunc_sub = simd_sub::f32x4_trunc;
      static constexpr auto emit_f32x4_nearest_sub = simd_sub::f32x4_nearest;
      static constexpr auto emit_f32x4_abs_sub = simd_sub::f32x4_abs;
      static constexpr auto emit_f32x4_neg_sub = simd_sub::f32x4_neg;
      static constexpr auto emit_f32x4_sqrt_sub = simd_sub::f32x4_sqrt;
      static constexpr auto emit_f32x4_add_sub = simd_sub::f32x4_add;
      static constexpr auto emit_f32x4_sub_sub = simd_sub::f32x4_sub;
      static constexpr auto emit_f32x4_mul_sub = simd_sub::f32x4_mul;
      static constexpr auto emit_f32x4_div_sub = simd_sub::f32x4_div;
      static constexpr auto emit_f32x4_min_sub = simd_sub::f32x4_min;
      static constexpr auto emit_f32x4_max_sub = simd_sub::f32x4_max;
      static constexpr auto emit_f32x4_pmin_sub = simd_sub::f32x4_pmin;
      static constexpr auto emit_f32x4_pmax_sub = simd_sub::f32x4_pmax;
      static constexpr auto emit_f64x2_ceil_sub = simd_sub::f64x2_ceil;
      static constexpr auto emit_f64x2_floor_sub = simd_sub::f64x2_floor;
      static constexpr auto emit_f64x2_trunc_sub = simd_sub::f64x2_trunc;
      static constexpr auto emit_f64x2_nearest_sub = simd_sub::f64x2_nearest;
      static constexpr auto emit_f64x2_abs_sub = simd_sub::f64x2_abs;
      static constexpr auto emit_f64x2_neg_sub = simd_sub::f64x2_neg;
      static constexpr auto emit_f64x2_sqrt_sub = simd_sub::f64x2_sqrt;
      static constexpr auto emit_f64x2_add_sub = simd_sub::f64x2_add;
      static constexpr auto emit_f64x2_sub_sub = simd_sub::f64x2_sub;
      static constexpr auto emit_f64x2_mul_sub = simd_sub::f64x2_mul;
      static constexpr auto emit_f64x2_div_sub = simd_sub::f64x2_div;
      static constexpr auto emit_f64x2_min_sub = simd_sub::f64x2_min;
      static constexpr auto emit_f64x2_max_sub = simd_sub::f64x2_max;
      static constexpr auto emit_f64x2_pmin_sub = simd_sub::f64x2_pmin;
      static constexpr auto emit_f64x2_pmax_sub = simd_sub::f64x2_pmax;
      static constexpr auto emit_i32x4_trunc_sat_f32x4_s_sub = simd_sub::i32x4_trunc_sat_f32x4_s;
      static constexpr auto emit_i32x4_trunc_sat_f32x4_u_sub = simd_sub::i32x4_trunc_sat_f32x4_u;
      static constexpr auto emit_f32x4_convert_i32x4_s_sub = simd_sub::f32x4_convert_i32x4_s;
      static constexpr auto emit_f32x4_convert_i32x4_u_sub = simd_sub::f32x4_convert_i32x4_u;
      static constexpr auto emit_i32x4_trunc_sat_f64x2_s_zero_sub = simd_sub::i32x4_trunc_sat_f64x2_s_zero;
      static constexpr auto emit_i32x4_trunc_sat_f64x2_u_zero_sub = simd_sub::i32x4_trunc_sat_f64x2_u_zero;
      static constexpr auto emit_f64x2_convert_low_i32x4_s_sub = simd_sub::f64x2_convert_low_i32x4_s;
      static constexpr auto emit_f64x2_convert_low_i32x4_u_sub = simd_sub::f64x2_convert_low_i32x4_u;
      static constexpr auto emit_f32x4_demote_f64x2_zero_sub = simd_sub::f32x4_demote_f64x2_zero;
      static constexpr auto emit_f64x2_promote_low_f32x4_sub = simd_sub::f64x2_promote_low_f32x4;

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
            _func->vpop(); uint32_t mask = _func->vpop(); // mask (low vreg)
            _func->vpop(); uint32_t val2 = _func->vpop(); // val2 (low vreg)
            _func->vpop(); uint32_t val1 = _func->vpop(); // val1 (low vreg)
            uint32_t d1 = _func->alloc_vreg(types::v128);
            uint32_t d2 = _func->alloc_vreg(types::v128);
            ir_inst inst{};
            inst.opcode = ir_op::v128_op;
            inst.type = types::v128;
            inst.flags = IR_SIDE_EFFECT;
            inst.dest = static_cast<uint32_t>(simd_sub::v128_bitselect);
            inst.simd.v_src1 = static_cast<uint16_t>(val1);
            inst.simd.v_src2 = static_cast<uint16_t>(val2);
            inst.simd.v_dest = static_cast<uint16_t>(d1);
            // Store mask vreg in addr field (unused for bitselect's memory addressing)
            inst.simd.addr = mask;
            _func->emit(inst);
            _func->vpush(d1); _func->vpush(d2);
         }
      }
      void emit_v128_any_true() {
         if (!_unreachable) {
            _func->has_simd = true;
            _func->vpop(); uint32_t s1 = _func->vpop();  // v128 (low vreg)
            uint32_t d = _func->alloc_vreg(types::i32);
            // Store result vreg in addr field so register path can pop scalar to vreg
            ir_simd_emit_with_offset(simd_sub::v128_any_true, 0, d, static_cast<uint16_t>(s1));
            _func->vpush(d);
         }
      }
      SIMD_UNOP(emit_i8x16_abs) SIMD_UNOP(emit_i8x16_neg) SIMD_UNOP(emit_i8x16_popcnt)
      void emit_i8x16_all_true() { ir_simd_test(simd_sub::i8x16_all_true); }
      void emit_i8x16_bitmask() { ir_simd_test(simd_sub::i8x16_bitmask); }
      SIMD_BINOP(emit_i8x16_narrow_i16x8_s) SIMD_BINOP(emit_i8x16_narrow_i16x8_u)
      SIMD_SHIFT(emit_i8x16_shl) SIMD_SHIFT(emit_i8x16_shr_s) SIMD_SHIFT(emit_i8x16_shr_u)
      SIMD_BINOP(emit_i8x16_add) SIMD_BINOP(emit_i8x16_add_sat_s) SIMD_BINOP(emit_i8x16_add_sat_u)
      SIMD_BINOP(emit_i8x16_sub) SIMD_BINOP(emit_i8x16_sub_sat_s) SIMD_BINOP(emit_i8x16_sub_sat_u)
      SIMD_BINOP(emit_i8x16_min_s) SIMD_BINOP(emit_i8x16_min_u) SIMD_BINOP(emit_i8x16_max_s)
      SIMD_BINOP(emit_i8x16_max_u) SIMD_BINOP(emit_i8x16_avgr_u)
      SIMD_UNOP(emit_i16x8_extadd_pairwise_i8x16_s) SIMD_UNOP(emit_i16x8_extadd_pairwise_i8x16_u)
      SIMD_UNOP(emit_i16x8_abs) SIMD_UNOP(emit_i16x8_neg) SIMD_BINOP(emit_i16x8_q15mulr_sat_s)
      void emit_i16x8_all_true() { ir_simd_test(simd_sub::i16x8_all_true); }
      void emit_i16x8_bitmask() { ir_simd_test(simd_sub::i16x8_bitmask); }
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
      void emit_i32x4_all_true() { ir_simd_test(simd_sub::i32x4_all_true); }
      void emit_i32x4_bitmask() { ir_simd_test(simd_sub::i32x4_bitmask); }
      SIMD_UNOP(emit_i32x4_extend_low_i16x8_s) SIMD_UNOP(emit_i32x4_extend_high_i16x8_s)
      SIMD_UNOP(emit_i32x4_extend_low_i16x8_u) SIMD_UNOP(emit_i32x4_extend_high_i16x8_u)
      SIMD_SHIFT(emit_i32x4_shl) SIMD_SHIFT(emit_i32x4_shr_s) SIMD_SHIFT(emit_i32x4_shr_u)
      SIMD_BINOP(emit_i32x4_add) SIMD_BINOP(emit_i32x4_sub) SIMD_BINOP(emit_i32x4_mul)
      SIMD_BINOP(emit_i32x4_min_s) SIMD_BINOP(emit_i32x4_min_u) SIMD_BINOP(emit_i32x4_max_s) SIMD_BINOP(emit_i32x4_max_u)
      SIMD_BINOP(emit_i32x4_dot_i16x8_s)
      SIMD_BINOP(emit_i32x4_extmul_low_i16x8_s) SIMD_BINOP(emit_i32x4_extmul_high_i16x8_s)
      SIMD_BINOP(emit_i32x4_extmul_low_i16x8_u) SIMD_BINOP(emit_i32x4_extmul_high_i16x8_u)
      SIMD_UNOP(emit_i64x2_abs) SIMD_UNOP(emit_i64x2_neg)
      void emit_i64x2_all_true() { ir_simd_test(simd_sub::i64x2_all_true); }
      void emit_i64x2_bitmask() { ir_simd_test(simd_sub::i64x2_bitmask); }
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
      void emit_memory_init(std::uint32_t s) {
         ir_bulk_mem3();
         if (!_unreachable) {
            ir_inst inst{}; inst.opcode = ir_op::memory_init;
            inst.flags = IR_SIDE_EFFECT; inst.dest = ir_vreg_none;
            _func->emit(inst);
         }
      }
      void emit_data_drop(std::uint32_t s) { }
      void emit_memory_copy() {
         ir_bulk_mem3(); // vpop the 3 args from vstack
         if (!_unreachable) {
            ir_inst inst{}; inst.opcode = ir_op::memory_copy;
            inst.flags = IR_SIDE_EFFECT; inst.dest = ir_vreg_none;
            _func->emit(inst);
         }
      }
      void emit_memory_fill() {
         ir_bulk_mem3();
         if (!_unreachable) {
            ir_inst inst{}; inst.opcode = ir_op::memory_fill;
            inst.flags = IR_SIDE_EFFECT; inst.dest = ir_vreg_none;
            _func->emit(inst);
         }
      }
      void emit_table_init(std::uint32_t s) { ir_bulk_mem3(); }
      void emit_elem_drop(std::uint32_t s) { }
      void emit_table_copy() { ir_bulk_mem3(); }

      // ──── Branch fixup ────
      // Patch a br/br_if IR instruction's target to the resolved block index.
      void fix_branch(branch_t inst_idx, label_t block_idx) {
         if (_func && inst_idx < _func->inst_count && block_idx != UINT32_MAX) {
            _func->insts[inst_idx].br.target = block_idx;
         }
      }

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
         inst.dest = val;  // reuse dest field for value vreg (stores have no result)
         inst.ri.src1 = addr;
         inst.ri.imm = static_cast<int32_t>(offset);
         _func->emit(inst);
      }

      // ──── SIMD vstack tracking + IR emission ────
      void ir_simd_emit(simd_sub sub) {
         ir_inst inst{};
         inst.opcode = ir_op::v128_op;
         inst.type = types::v128;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = static_cast<uint32_t>(sub);
         inst.simd.v_src1 = 0xFFFF;
         inst.simd.v_src2 = 0xFFFF;
         inst.simd.v_dest = 0xFFFF;
         _func->emit(inst);
      }
      void ir_simd_emit_with_offset(simd_sub sub, uint32_t offset, uint32_t addr_vreg = ir_vreg_none,
                                    uint16_t vs1 = 0xFFFF, uint16_t vd = 0xFFFF) {
         ir_inst inst{};
         inst.opcode = ir_op::v128_op;
         inst.type = types::v128;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = static_cast<uint32_t>(sub);
         inst.simd.offset = offset;
         inst.simd.addr = addr_vreg;
         inst.simd.v_src1 = vs1;
         inst.simd.v_src2 = 0xFFFF;
         inst.simd.v_dest = vd;
         _func->emit(inst);
      }
      void ir_simd_emit_with_offset_lane(simd_sub sub, uint32_t offset, uint8_t lane, uint32_t addr_vreg = ir_vreg_none,
                                          uint16_t vs1 = 0xFFFF, uint16_t vd = 0xFFFF) {
         ir_inst inst{};
         inst.opcode = ir_op::v128_op;
         inst.type = types::v128;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = static_cast<uint32_t>(sub);
         inst.simd.offset = offset;
         inst.simd.addr = addr_vreg;
         inst.simd.lane = lane;
         inst.simd.v_src1 = vs1;
         inst.simd.v_src2 = 0xFFFF;
         inst.simd.v_dest = vd;
         _func->emit(inst);
      }
      void ir_simd_emit_with_lane(simd_sub sub, uint8_t lane) {
         ir_inst inst{};
         inst.opcode = ir_op::v128_op;
         inst.type = types::v128;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = static_cast<uint32_t>(sub);
         inst.simd.v_src1 = 0xFFFF;
         inst.simd.v_src2 = 0xFFFF;
         inst.simd.v_dest = 0xFFFF;
         inst.simd.lane = lane;
         _func->emit(inst);
      }
      void ir_simd_load(simd_sub sub, uint32_t offset) {
         if (_unreachable) return;
         uint32_t addr = _func->vpop();  // address vreg
         uint32_t d1 = _func->alloc_vreg(types::v128);
         uint32_t d2 = _func->alloc_vreg(types::v128);
         ir_simd_emit_with_offset(sub, offset, addr, 0xFFFF, static_cast<uint16_t>(d1));
         _func->vpush(d1); _func->vpush(d2);
      }
      void ir_simd_store(simd_sub sub, uint32_t offset) {
         if (_unreachable) return;
         _func->vpop(); uint32_t s1 = _func->vpop();  // v128 value (low vreg)
         uint32_t addr = _func->vpop(); // address vreg
         ir_simd_emit_with_offset(sub, offset, addr, static_cast<uint16_t>(s1));
      }
      void ir_simd_load_lane(simd_sub sub, uint32_t offset, uint8_t lane) {
         if (_unreachable) return;
         _func->has_simd = true;
         _func->vpop(); uint32_t s1 = _func->vpop();  // v128 (low vreg)
         uint32_t addr = _func->vpop(); // address
         uint32_t d1 = _func->alloc_vreg(types::v128);
         uint32_t d2 = _func->alloc_vreg(types::v128);
         ir_simd_emit_with_offset_lane(sub, offset, lane, addr,
                                        static_cast<uint16_t>(s1), static_cast<uint16_t>(d1));
         _func->vpush(d1); _func->vpush(d2);
      }
      void ir_simd_store_lane(simd_sub sub, uint32_t offset, uint8_t lane) {
         if (_unreachable) return;
         _func->has_simd = true;
         _func->vpop(); uint32_t s1 = _func->vpop();  // v128 (low vreg)
         uint32_t addr = _func->vpop(); // address
         ir_simd_emit_with_offset_lane(sub, offset, lane, addr, static_cast<uint16_t>(s1));
      }
      void ir_simd_unop(simd_sub sub) {
         if (_unreachable) return;
         _func->has_simd = true;
         _func->vpop(); uint32_t s1 = _func->vpop(); // v128 src (low vreg)
         uint32_t d1 = _func->alloc_vreg(types::v128);
         uint32_t d2 = _func->alloc_vreg(types::v128);
         ir_inst inst{};
         inst.opcode = ir_op::v128_op;
         inst.type = types::v128;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = static_cast<uint32_t>(sub);
         inst.simd.v_src1 = static_cast<uint16_t>(s1);
         inst.simd.v_src2 = 0xFFFF;
         inst.simd.v_dest = static_cast<uint16_t>(d1);
         _func->emit(inst);
         _func->vpush(d1); _func->vpush(d2);
      }
      void ir_simd_binop(simd_sub sub) {
         if (_unreachable) return;
         _func->has_simd = true;
         _func->vpop(); uint32_t s2 = _func->vpop(); // v128 src2 (low vreg)
         _func->vpop(); uint32_t s1 = _func->vpop(); // v128 src1 (low vreg)
         uint32_t d1 = _func->alloc_vreg(types::v128);
         uint32_t d2 = _func->alloc_vreg(types::v128);
         ir_inst inst{};
         inst.opcode = ir_op::v128_op;
         inst.type = types::v128;
         inst.flags = IR_SIDE_EFFECT;
         inst.dest = static_cast<uint32_t>(sub);
         inst.simd.v_src1 = static_cast<uint16_t>(s1);
         inst.simd.v_src2 = static_cast<uint16_t>(s2);
         inst.simd.v_dest = static_cast<uint16_t>(d1);
         _func->emit(inst);
         _func->vpush(d1); _func->vpush(d2);
      }
      void ir_simd_shift(simd_sub sub) {
         if (_unreachable) return;
         _func->has_simd = true;
         uint32_t shift_amt = _func->vpop();  // shift amount (i32)
         _func->vpop(); uint32_t s1 = _func->vpop();  // v128 (low vreg)
         uint32_t d1 = _func->alloc_vreg(types::v128);
         uint32_t d2 = _func->alloc_vreg(types::v128);
         // Store shift amount vreg in offset field so codegen can load from register
         ir_simd_emit_with_offset_lane(sub, shift_amt, 0, ir_vreg_none,
                                        static_cast<uint16_t>(s1), static_cast<uint16_t>(d1));
         _func->vpush(d1); _func->vpush(d2);
      }
      void ir_simd_extract(simd_sub sub, uint8_t lane) {
         if (_unreachable) return;
         _func->has_simd = true;
         _func->vpop(); uint32_t s1 = _func->vpop();  // v128 (low vreg)
         uint32_t d = _func->alloc_vreg(types::i64);
         // Store result vreg in addr field so register path can pop scalar to vreg
         ir_simd_emit_with_offset_lane(sub, 0, lane, d, static_cast<uint16_t>(s1));
         _func->vpush(d);
      }
      void ir_simd_replace(simd_sub sub, uint8_t lane) {
         if (_unreachable) return;
         _func->has_simd = true;
         uint32_t scalar = _func->vpop();  // scalar value vreg
         _func->vpop(); uint32_t s1 = _func->vpop();  // v128 (low vreg)
         uint32_t d1 = _func->alloc_vreg(types::v128);
         uint32_t d2 = _func->alloc_vreg(types::v128);
         ir_simd_emit_with_offset_lane(sub, scalar, lane, ir_vreg_none,
                                        static_cast<uint16_t>(s1), static_cast<uint16_t>(d1));
         _func->vpush(d1); _func->vpush(d2);
      }
      void ir_simd_splat(simd_sub sub) {
         if (_unreachable) return;
         _func->has_simd = true;
         uint32_t scalar = _func->vpop();  // scalar value vreg
         uint32_t d1 = _func->alloc_vreg(types::v128);
         uint32_t d2 = _func->alloc_vreg(types::v128);
         ir_simd_emit_with_offset(sub, 0, scalar, 0xFFFF, static_cast<uint16_t>(d1));
         _func->vpush(d1); _func->vpush(d2);
      }
      void ir_simd_test(simd_sub sub) {
         if (_unreachable) return;
         _func->has_simd = true;
         _func->vpop(); uint32_t s1 = _func->vpop();  // v128 (low vreg)
         uint32_t d = _func->alloc_vreg(types::i32);
         // Store result vreg in addr field so register path can pop scalar to vreg
         ir_simd_emit_with_offset(sub, 0, d, static_cast<uint16_t>(s1));
         _func->vpush(d);
      }
      void ir_bulk_mem3() {
         if (_unreachable) return;
         // Pop 3 operands and emit arg instructions so register mode
         // pushes them to the native stack for the bulk memory handler.
         uint32_t v2 = _func->vpop(); // count (top)
         uint32_t v1 = _func->vpop(); // val/src
         uint32_t v0 = _func->vpop(); // dest
         uint32_t vregs[3] = {v0, v1, v2};
         for (int i = 0; i < 3; ++i) {
            ir_inst arg{};
            arg.opcode = ir_op::arg;
            arg.type = types::pseudo;
            arg.flags = IR_NONE;
            arg.dest = ir_vreg_none;
            arg.rr.src1 = vregs[i];
            arg.rr.src2 = ir_vreg_none;
            _func->emit(arg);
         }
      }

      // ──── State ────
      growable_allocator& _allocator;
      std::size_t _source_bytes;
      module& _mod;
      jit_scratch_allocator _scratch;       // All transient data (IR, regalloc, codegen aux)
      ir_function* _functions = nullptr;
      uint32_t _num_functions = 0;
      ir_function* _func = nullptr;
      bool _unreachable = false;
   };

}} // namespace eosio::vm
