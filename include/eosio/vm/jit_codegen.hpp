#pragma once

// Pass 2 of the two-pass optimizing JIT (jit2).
// Converts IR instructions to x86_64 machine code.
//
// Phase 3 (naive): All vregs spilled to stack slots.
//   Each IR instruction loads operands from slots, operates, stores result.
//   Correct but slow — same push/pop pattern as the original JIT.
//
// Phase 4 (register allocation): Linear scan assigns vregs to physical registers.
//   This is where the 3-4x performance improvement comes from.
//
// All code is emitted into the growable_allocator (mmap-backed, no malloc).

#include <eosio/vm/allocator.hpp>
#include <eosio/vm/exceptions.hpp>
#include <eosio/vm/jit_ir.hpp>
#include <eosio/vm/jit_regalloc.hpp>
#include <eosio/vm/softfloat.hpp>
#include <eosio/vm/x86_64_base.hpp>
#include <eosio/vm/types.hpp>
#include <eosio/vm/utils.hpp>
#include <eosio/vm/signals.hpp>

#include <cassert>
#include <cstdint>
#include <cstring>

namespace eosio { namespace vm {

   template<typename Context, bool StackLimitIsBytes>
   class jit_codegen : public x86_64_base<jit_codegen<Context, StackLimitIsBytes>> {
      using base = x86_64_base<jit_codegen<Context, StackLimitIsBytes>>;
      using base::code;
      using base::rax; using base::rcx; using base::rdx; using base::rbx;
      using base::rsp; using base::rbp; using base::rsi; using base::rdi;
      using base::r8; using base::r9; using base::r10; using base::r11;
      using base::r12; using base::r13; using base::r14; using base::r15;
      using base::eax; using base::ecx; using base::edx; using base::ebx;
      using base::esp; using base::ebp; using base::esi; using base::edi;
      using base::r8d; using base::r9d;
      using base::al; using base::cl; using base::dl;
      using base::ax;
      using base::xmm0; using base::xmm1;
      using typename base::general_register64;
      using typename base::general_register32;
      using typename base::disp_memory_ref;
      using typename base::simple_memory_ref;
      using typename base::sib_memory_ref;
      using typename base::Jcc;
      using typename base::imm8;
      using typename base::imm32;

    public:
      jit_codegen(growable_allocator& alloc, module& mod, void* code_segment_base = nullptr)
         : _allocator(alloc), _mod(mod) {
         // Allocate relocation table BEFORE starting the code region.
         // _code_base[0] must be the SysV ABI entry point.
         init_relocations();
         _code_segment_base = code_segment_base ? code_segment_base : _allocator.start_code();
      }

      // Emit the SysV ABI entry point (same as machine_code_writer)
      void emit_entry_and_error_handlers() {
         // Allocate generous buffers — no reclaim to avoid LIFO issues.
         // All memory is reclaimed in bulk by end_code<true>().

         // SysV ABI entry point
         auto* buf = _allocator.alloc<unsigned char>(256);
         code = buf;
         emit_sysv_abi_interface();

         // Error handlers
         buf = _allocator.alloc<unsigned char>(80);
         code = buf;
         fpe_handler = emit_error_handler(&on_fp_error);
         call_indirect_handler = emit_error_handler(&on_call_indirect_error);
         type_error_handler = emit_error_handler(&on_type_error);
         stack_overflow_handler = emit_error_handler(&on_stack_overflow);
         memory_handler = emit_error_handler(&on_memory_error);

         // Host function thunks
         const uint32_t num_imported = _mod.get_imported_functions_size();
         if (num_imported > 0) {
            const std::size_t host_functions_size = (42 + 10 * Context::async_backtrace()) * num_imported;
            buf = _allocator.alloc<unsigned char>(host_functions_size);
            code = buf;
            for (uint32_t i = 0; i < num_imported; ++i) {
               start_function(code, i);
               emit_host_call(i);
            }
         }
      }

      // Compile one function from IR to x86_64
      void compile_function(ir_function& func, function_body& body) {
         // Estimate output size: each IR instruction emits at most ~64 bytes
         const std::size_t est_size = static_cast<std::size_t>(func.inst_count) * 64 + 256;
         auto* buf = _allocator.alloc<unsigned char>(est_size);
         auto* code_start = buf;
         code = buf;

         // Allocate block address tracking (one entry per basic block)
         _block_addrs = _allocator.alloc<void*>(func.block_count);
         _block_fixups = _allocator.alloc<block_fixup*>(func.block_count);
         _num_blocks = func.block_count;
         for (uint32_t i = 0; i < func.block_count; ++i) {
            _block_addrs[i] = nullptr;
            _block_fixups[i] = nullptr;
         }

         // Build vreg → physical register mapping
         _vreg_map = nullptr;
         _num_vregs = 0;
         _num_spill_slots = 0;
         if (func.interval_count > 0 && func.intervals) {
            _num_vregs = func.next_vreg;
            _vreg_map = _allocator.alloc<int8_t>(_num_vregs);
            _spill_map = _allocator.alloc<int16_t>(_num_vregs);
            for (uint32_t v = 0; v < _num_vregs; ++v) {
               _vreg_map[v] = -1;
               _spill_map[v] = -1;
            }
            for (uint32_t iv = 0; iv < func.interval_count; ++iv) {
               auto& interval = func.intervals[iv];
               if (interval.vreg < _num_vregs) {
                  _vreg_map[interval.vreg] = interval.phys_reg;
                  _spill_map[interval.vreg] = interval.spill_slot;
               }
            }
            _num_spill_slots = func.num_spill_slots;
            _use_regalloc = true;
         } else {
            _use_regalloc = false;
         }

         start_function(code, func.func_index + _mod.get_imported_functions_size());

         // Emit function prologue
         emit_function_prologue(func);

         // Emit each IR instruction
         for (uint32_t i = 0; i < func.inst_count; ++i) {
            // Check if any blocks start at this instruction index
            for (uint32_t b = 0; b < func.block_count; ++b) {
               if (func.blocks[b].start != UINT32_MAX && func.blocks[b].start == i
                   && _block_addrs[b] == nullptr) {
                  mark_block_start(b);
               }
            }
            if (!_use_regalloc || !emit_ir_inst_reg(func, func.insts[i], i)) {
               emit_ir_inst(func, func.insts[i], i);
            }
            // Check if any blocks end at this instruction index
            for (uint32_t b = 0; b < func.block_count; ++b) {
               if (func.blocks[b].end != UINT32_MAX && func.blocks[b].end == i + 1) {
                  mark_block_end(func, b);
               }
            }
         }

         // Emit function epilogue (return)
         emit_function_epilogue(func);

         // Record code offset. Don't reclaim unused code buffer space —
         // block fixup nodes may be allocated after the code buffer.
         // All memory is reclaimed in bulk by end_code<true>().
         body.jit_code_offset = code_start - static_cast<unsigned char*>(_code_segment_base);

         // Clear per-function state
         _block_addrs = nullptr;
         _block_fixups = nullptr;
         _num_blocks = 0;
         _if_fixup_top = 0;
         _in_br_table = false;
         _br_table_case = 0;
         _br_table_size = 0;
      }

      void finalize_code() {
         _allocator.end_code<true>(_code_segment_base);

         // Patch element table entries (same as machine_code_writer destructor)
         auto num_functions = _mod.get_functions_total();
         for (auto& elem : _mod.elements) {
            for (auto& entry : elem.elems) {
               void* addr = call_indirect_handler;
               if (entry.index < num_functions && entry.index < _num_relocs) {
                  if (_relocs[entry.index].address) {
                     addr = _relocs[entry.index].address;
                  }
               }
               std::size_t offset = static_cast<char*>(addr) - static_cast<char*>(_code_segment_base);
               entry.code_ptr = _mod.allocator._code_base + offset;
            }
         }
      }

    private:
      // ──────── SysV ABI interface (identical to machine_code_writer) ────────
      void emit_sysv_abi_interface() {
         this->emit_push_raw(rbp);
         this->emit_mov(rsp, rbp);
         this->emit_sub(16, rsp);

         this->emit(base::TEST, r8, r8);
         this->emit(base::IA32_REX_W(0x0f, 0x45), r8, rsp);

         // save and set mxcsr
         this->emit_bytes(0x0f, 0xae, 0x5d, 0xfc); // stmxcsr [rbp-4]
         this->emit_movd(0x1f80, *(rbp - 8));
         this->emit_bytes(0x0f, 0xae, 0x55, 0xf8); // ldmxcsr [rbp-8]

         // copy args loop
         this->emit(base::TEST, r9, r9);
         void* loop_end = this->emit_branch8(base::JZ);
         void* loop = code;
         this->emit_mov(*rdx, rax);
         this->emit_add(8, rdx);
         this->emit_push_raw(rax);
         this->emit(base::DEC, r9);
         base::fix_branch8(this->emit_branch8(base::JNZ), loop);
         base::fix_branch8(loop_end, code);

         // load call depth counter
         this->emit_mov(rbx, *(rbp - 16));
         if constexpr (Context::async_backtrace()) {
            this->emit_mov(*(rdi + 16), ebx);
         } else {
            this->emit_mov(*rdi, ebx);
         }

         if constexpr (Context::async_backtrace()) {
            this->emit_mov(rbp, *(rdi + 8));
         }
         this->emit_call(rcx);
         if constexpr (Context::async_backtrace()) {
            this->emit_xor(edx, edx);
            this->emit_mov(rdx, *(rdi + 8));
         }

         this->emit_mov(*(rbp - 16), rbx);
         this->emit_bytes(0x0f, 0xae, 0x55, 0xfc); // ldmxcsr [rbp-4]

         // check vector result
         this->emit_mov(*(rbp + 16), edx);
         this->emit(base::TEST, edx, edx);
         void* is_vector = this->emit_branch8(base::JZ);
         this->emit_vpextrq(0, xmm0, rax);
         this->emit_vpextrq(1, xmm0, rdx);
         base::fix_branch8(is_vector, code);

         this->emit_mov(rbp, rsp);
         this->emit_pop_raw(rbp);
         this->emit(base::RET);
      }

      void* emit_error_handler(void (*handler)()) {
         void* result = code;
         this->emit_bytes(0x48, 0x83, 0xe4, 0xf0); // andq $-16, %rsp
         this->emit_bytes(0x48, 0xb8);
         this->emit_operand_ptr(handler);
         this->emit_bytes(0xff, 0xd0); // callq *%rax
         return result;
      }

      void emit_host_call(uint32_t funcnum) {
         uint32_t extra = 0;
         if constexpr (Context::async_backtrace()) {
            this->emit_bytes(0x55);             // pushq %rbp
            this->emit_bytes(0x48, 0x89, 0x27); // movq %rsp, (%rdi)
            extra = 8;
         }
         this->emit_bytes(0xba);
         this->emit_operand32(funcnum);     // mov $funcnum, %edx
         this->emit_bytes(0x57);             // pushq %rdi
         this->emit_bytes(0x56);             // pushq %rsi
         this->emit_bytes(0x48, 0x8d, 0x74, 0x24, 0x18 + extra); // lea 24(%rsp), %rsi
         // align stack
         this->emit_bytes(0x48, 0x89, 0xe1); // mov %rsp, %rcx
         this->emit_bytes(0x48, 0x83, 0xe4, 0xf0); // andq $-16, %rsp
         this->emit_bytes(0x51);             // push %rcx
         this->emit_bytes(0x51);             // push %rcx
         this->emit_mov(ebx, ecx);
         this->emit_bytes(0x48, 0xb8);
         this->emit_operand_ptr(&call_host_function);
         this->emit_bytes(0xff, 0xd0);       // callq *%rax
         // restore stack
         this->emit_bytes(0x48, 0x8b, 0x24, 0x24); // mov (%rsp), %rsp
         this->emit_bytes(0x5e);             // popq %rsi
         this->emit_bytes(0x5f);             // popq %rdi
         if constexpr (Context::async_backtrace()) {
            this->emit_bytes(0x31, 0xd2);       // xorl %edx, %edx
            this->emit_bytes(0x48, 0x89, 0x17); // movq %rdx, (%rdi)
            this->emit_bytes(0x5d);             // popq %rbp
         }
         this->emit_bytes(0xc3);             // retq
      }

      // ──────── Function prologue/epilogue ────────

      void emit_function_prologue(ir_function& func) {
         this->emit_push_raw(rbp);
         this->emit_mov(rsp, rbp);

         uint32_t body_locals = func.num_locals - func.num_params;
         _body_locals = body_locals;

         // Count callee-saved registers used
         _callee_saved_count = 0;
         _callee_saved_used = 0;
         if (_use_regalloc) {
            for (uint32_t v = 0; v < _num_vregs; ++v) {
               int8_t pr = _vreg_map[v];
               if (pr >= static_cast<int8_t>(phys_reg::caller_saved_count)) {
                  _callee_saved_used |= (1 << (pr - static_cast<int8_t>(phys_reg::caller_saved_count)));
               }
            }
            for (int i = 0; i < 4; ++i) if (_callee_saved_used & (1 << i)) _callee_saved_count++;
         }

         // Allocate space: body locals + spill slots + callee-saved saves
         uint32_t total_slots = body_locals + _num_spill_slots + _callee_saved_count;
         if (total_slots > 0) {
            this->emit_xor(eax, eax);
            for (uint32_t i = 0; i < total_slots; ++i) {
               this->emit_push_raw(rax);
            }
         }

         // Save callee-saved registers to the frame (after locals and spill slots)
         if (_use_regalloc) {
            int32_t save_offset = -static_cast<int32_t>((body_locals + _num_spill_slots + 1) * 8);
            if (_callee_saved_used & 1) { this->emit_mov(r12, *(rbp + save_offset)); save_offset -= 8; }
            if (_callee_saved_used & 2) { this->emit_mov(r13, *(rbp + save_offset)); save_offset -= 8; }
            if (_callee_saved_used & 4) { this->emit_mov(r14, *(rbp + save_offset)); save_offset -= 8; }
            if (_callee_saved_used & 8) { this->emit_mov(r15, *(rbp + save_offset)); save_offset -= 8; }
         }
      }

      // ──────── IR instruction emission (naive: everything via stack) ────────
      // For Phase 3, this emits the same push/pop stack-machine code as the
      // original JIT. The purpose is correctness verification — register
      // allocation in Phase 4 will replace this with register-based emission.

      void emit_ir_inst(ir_function& func, const ir_inst& inst, uint32_t idx) {
         if (inst.flags & IR_DEAD) return;

         switch (inst.opcode) {
         // ── Constants ──
         case ir_op::const_i32: {
            uint32_t val = static_cast<uint32_t>(inst.imm64);
            this->emit_mov(val, eax);
            this->emit_push_raw(rax);
            break;
         }
         case ir_op::const_i64: {
            uint64_t val = static_cast<uint64_t>(inst.imm64);
            this->emit_mov(val, rax);
            this->emit_push_raw(rax);
            break;
         }
         case ir_op::const_f32: {
            uint32_t bits;
            memcpy(&bits, &inst.immf32, 4);
            this->emit_mov(bits, eax);
            this->emit_push_raw(rax);
            break;
         }
         case ir_op::const_f64: {
            uint64_t bits;
            memcpy(&bits, &inst.immf64, 8);
            this->emit_mov(bits, rax);
            this->emit_push_raw(rax);
            break;
         }

         // ── Integer arithmetic (binary) ──
         case ir_op::i32_add: emit_binop_ra(inst, [this](auto d, auto s){ this->emit_add(s, d); }, true); break;
         case ir_op::i32_sub: emit_binop_ra(inst, [this](auto d, auto s){ this->emit_sub(s, d); }, true); break;
         case ir_op::i32_mul: emit_binop_ra(inst, [this](auto d, auto s){ this->emit(base::IMUL, s, d); }, true); break;
         case ir_op::i32_and: emit_binop_ra(inst, [this](auto d, auto s){ this->emit(base::AND_A, s, d); }, true); break;
         case ir_op::i32_or:  emit_binop_ra(inst, [this](auto d, auto s){ this->emit(base::OR_A, s, d); }, true); break;
         case ir_op::i32_xor: emit_binop_ra(inst, [this](auto d, auto s){ this->emit(base::XOR_A, s, d); }, true); break;

         case ir_op::i64_add: emit_binop_ra(inst, [this](auto d, auto s){ this->emit_add(s, d); }, false); break;
         case ir_op::i64_sub: emit_binop_ra(inst, [this](auto d, auto s){ this->emit_sub(s, d); }, false); break;
         case ir_op::i64_mul: emit_binop_ra(inst, [this](auto d, auto s){ this->emit(base::IMUL, s, d); }, false); break;
         case ir_op::i64_and: emit_binop_ra(inst, [this](auto d, auto s){ this->emit(base::AND_A, s, d); }, false); break;
         case ir_op::i64_or:  emit_binop_ra(inst, [this](auto d, auto s){ this->emit(base::OR_A, s, d); }, false); break;
         case ir_op::i64_xor: emit_binop_ra(inst, [this](auto d, auto s){ this->emit(base::XOR_A, s, d); }, false); break;

         // ── Shifts ──
         case ir_op::i32_shl:   emit_i32_shift(base::SHL_cl); break;
         case ir_op::i32_shr_s: emit_i32_shift(base::SAR_cl); break;
         case ir_op::i32_shr_u: emit_i32_shift(base::SHR_cl); break;
         case ir_op::i64_shl:   emit_i64_shift(base::SHL_cl); break;
         case ir_op::i64_shr_s: emit_i64_shift(base::SAR_cl); break;
         case ir_op::i64_shr_u: emit_i64_shift(base::SHR_cl); break;

         // ── Comparisons ──
         case ir_op::i32_eqz:
            this->emit_pop_raw(rax);
            this->emit(base::TEST, eax, eax);
            this->emit_setcc(base::JZ, al);
            this->emit_bytes(0x0f, 0xb6, 0xc0); // movzbl %al, %eax
            this->emit_push_raw(rax);
            break;

         case ir_op::i32_eq: emit_i32_relop(base::JE); break;
         case ir_op::i32_ne: emit_i32_relop(base::JNE); break;
         case ir_op::i32_lt_s: emit_i32_relop(base::JL); break;
         case ir_op::i32_lt_u: emit_i32_relop(base::JB); break;
         case ir_op::i32_gt_s: emit_i32_relop(base::JG); break;
         case ir_op::i32_gt_u: emit_i32_relop(base::JA); break;
         case ir_op::i32_le_s: emit_i32_relop(base::JLE); break;
         case ir_op::i32_le_u: emit_i32_relop(base::JBE); break;
         case ir_op::i32_ge_s: emit_i32_relop(base::JGE); break;
         case ir_op::i32_ge_u: emit_i32_relop(base::JAE); break;

         case ir_op::i64_eqz:
            this->emit_pop_raw(rax);
            this->emit(base::TEST, rax, rax);
            this->emit_setcc(base::JZ, al);
            this->emit_bytes(0x0f, 0xb6, 0xc0);
            this->emit_push_raw(rax);
            break;

         case ir_op::i64_eq: emit_i64_relop(base::JE); break;
         case ir_op::i64_ne: emit_i64_relop(base::JNE); break;
         case ir_op::i64_lt_s: emit_i64_relop(base::JL); break;
         case ir_op::i64_lt_u: emit_i64_relop(base::JB); break;
         case ir_op::i64_gt_s: emit_i64_relop(base::JG); break;
         case ir_op::i64_gt_u: emit_i64_relop(base::JA); break;
         case ir_op::i64_le_s: emit_i64_relop(base::JLE); break;
         case ir_op::i64_le_u: emit_i64_relop(base::JBE); break;
         case ir_op::i64_ge_s: emit_i64_relop(base::JGE); break;
         case ir_op::i64_ge_u: emit_i64_relop(base::JAE); break;

         // ── Return ──
         case ir_op::return_:
            if (inst.rr.src1 != ir_vreg_none) {
               this->emit_pop_raw(rax);
            }
            this->emit_mov(rbp, rsp);
            this->emit_pop_raw(rbp);
            this->emit(base::RET);
            break;

         // ── Unreachable ──
         case ir_op::unreachable:
            emit_error_handler(&on_unreachable);
            break;

         // ── Nop / control flow markers ──
         case ir_op::nop:
         case ir_op::arg:
            break;

         // The IR doesn't emit explicit block/loop/if/else/end instructions.
         // Block boundaries are tracked via ir_basic_block structures.
         // The jit_codegen marks block addresses when it sees the first
         // instruction of a block (using block.start) and patches forward
         // references when it sees the block end (using block.end).
         // We handle this by checking block boundaries before each instruction
         // in the main loop.
         case ir_op::block:
         case ir_op::loop:
            break;

         case ir_op::if_: {
            // Pop condition, test, emit forward conditional branch
            this->emit_pop_raw(rax);
            this->emit(base::TEST, eax, eax);
            void* branch = this->emit_branchcc32(base::JZ);
            // Store on if_fixup stack (patched by else_ or block end)
            push_if_fixup(branch);
            break;
         }

         case ir_op::else_: {
            uint32_t target_block = inst.br.target;
            // Then-block: emit jump to end (forward fixup to block end)
            if (target_block < _num_blocks) {
               void* jmp = emit_jmp32();
               auto* fixup = _allocator.alloc<block_fixup>(1);
               fixup->branch = jmp;
               fixup->next = _block_fixups[target_block];
               _block_fixups[target_block] = fixup;
            }
            // Patch the if_ branch to point HERE (else start)
            pop_if_fixup_to(code);
            break;
         }

         // ── Control flow (branches) ──
         case ir_op::br_table:
            _br_table_case = 0;
            _br_table_size = inst.dest;
            _in_br_table = true;
            break;

         case ir_op::br:
            if (_in_br_table) {
               bool is_default = (_br_table_case >= _br_table_size);
               if (is_default) {
                  // Default case: pop index, unconditional branch
                  this->emit_pop_raw(rax); // discard index
                  emit_branch_to_block(func, inst.br.target, inst.dest, inst.type);
                  _in_br_table = false;
               } else {
                  // Numbered case: compare index at (%rsp) without popping
                  // cmpl $case, (%rsp)
                  this->emit_bytes(0x81, 0x3c, 0x24);
                  this->emit_operand32(_br_table_case);
                  // If equal, pop index and branch
                  void* skip = this->emit_branchcc32(base::JNE);
                  this->emit_pop_raw(rax); // pop index
                  emit_branch_to_block(func, inst.br.target, inst.dest, inst.type);
                  base::fix_branch(skip, code);
                  _br_table_case++;
               }
            } else {
               emit_branch_to_block(func, inst.br.target, inst.dest, inst.type);
            }
            break;

         case ir_op::br_if:
            emit_cond_branch_to_block(func, inst.br.target, inst.dest, inst.type);
            break;


         // ── Calls ──
         case ir_op::call: {
            uint32_t funcnum = inst.call.index;  // absolute index (includes imports)
            const func_type& ft = _mod.get_function_type(funcnum);
            // Decrement call depth
            if constexpr (!StackLimitIsBytes) {
               this->emit(base::DECD, ebx);
               base::fix_branch(this->emit_branchcc32(base::JZ), stack_overflow_handler);
            }
            // Emit call (may need relocation)
            void* branch = this->emit_call32();
            register_call(branch, funcnum);
            // Pop params, push result
            emit_call_multipop(ft);
            // Increment call depth
            if constexpr (!StackLimitIsBytes) {
               this->emit(base::INCD, ebx);
            }
            break;
         }
         case ir_op::call_indirect: {
            uint32_t fti = inst.call.index;
            const func_type& ft = _mod.types[fti];
            // Pop table index
            this->emit_pop_raw(rax);
            // Bounds check
            uint32_t table_size = _mod.tables[0].limits.initial;
            this->emit_cmp(table_size, eax);
            base::fix_branch(this->emit_branchcc32(base::JAE), call_indirect_handler);
            // Compute table entry: each entry is 16 bytes {type_idx(4), pad(4), code_ptr(8)}
            // shlq $4, %rax
            this->emit_bytes(0x48, 0xc1, 0xe0, 0x04);
            if (_mod.indirect_table(0)) {
               this->emit_mov(*(rsi + wasm_allocator::table_offset()), rcx);
               this->emit_add(rcx, rax);
            } else {
               // lea table_offset(%rsi,%rax), %rax
               this->emit_bytes(0x48, 0x8d, 0x84, 0x06);
               this->emit_operand32(static_cast<uint32_t>(wasm_allocator::table_offset()));
            }
            // Type check: cmp $fti, (%rax)
            this->emit_bytes(0x81, 0x38); // cmp imm32, (%rax)
            this->emit_operand32(fti);
            base::fix_branch(this->emit_branchcc32(base::JNE), type_error_handler);
            // Call through function pointer
            if constexpr (!StackLimitIsBytes) {
               this->emit(base::DECD, ebx);
               base::fix_branch(this->emit_branchcc32(base::JZ), stack_overflow_handler);
            }
            // call *8(%rax)
            this->emit_bytes(0xff, 0x50, 0x08);
            emit_call_multipop(ft);
            if constexpr (!StackLimitIsBytes) {
               this->emit(base::INCD, ebx);
            }
            break;
         }

         // ── Local/global access ──
         case ir_op::local_get: {
            int32_t offset = get_frame_offset(func, inst.local.index);
            this->emit_mov(*(rbp + offset), rax);
            this->emit_push_raw(rax);
            break;
         }
         case ir_op::local_set: {
            int32_t offset = get_frame_offset(func, inst.local.index);
            this->emit_pop_raw(rax);
            this->emit_mov(rax, *(rbp + offset));
            break;
         }
         case ir_op::local_tee: {
            int32_t offset = get_frame_offset(func, inst.local.index);
            this->emit_pop_raw(rax);
            this->emit_mov(rax, *(rbp + offset));
            this->emit_push_raw(rax);
            break;
         }
         case ir_op::global_get: {
            uint32_t gi = inst.local.index;
            auto loc = emit_global_loc(gi);
            this->emit_mov(loc, rax);
            this->emit_push_raw(rax);
            break;
         }
         case ir_op::global_set: {
            uint32_t gi = inst.local.index;
            this->emit_pop_raw(rax);
            auto loc = emit_global_loc(gi);
            this->emit_mov(rax, loc);
            break;
         }

         // ── Memory loads ──
         case ir_op::i32_load:     emit_load(inst.ri.imm, base::MOV_A, eax); break;
         case ir_op::i64_load:     emit_load(inst.ri.imm, base::MOV_A, rax); break;
         case ir_op::f32_load:     emit_load(inst.ri.imm, base::MOV_A, eax); break;
         case ir_op::f64_load:     emit_load(inst.ri.imm, base::MOV_A, rax); break;
         case ir_op::i32_load8_s:  emit_load(inst.ri.imm, base::MOVSXB, eax); break;
         case ir_op::i32_load16_s: emit_load(inst.ri.imm, base::MOVSXW, eax); break;
         case ir_op::i32_load8_u:  emit_load(inst.ri.imm, base::MOVZXB, eax); break;
         case ir_op::i32_load16_u: emit_load(inst.ri.imm, base::MOVZXW, eax); break;
         case ir_op::i64_load8_s:  emit_load(inst.ri.imm, base::MOVSXB, rax); break;
         case ir_op::i64_load16_s: emit_load(inst.ri.imm, base::MOVSXW, rax); break;
         case ir_op::i64_load32_s: emit_load(inst.ri.imm, base::MOVSXD, rax); break;
         case ir_op::i64_load8_u:  emit_load(inst.ri.imm, base::MOVZXB, eax); break;
         case ir_op::i64_load16_u: emit_load(inst.ri.imm, base::MOVZXW, eax); break;
         case ir_op::i64_load32_u: emit_load(inst.ri.imm, base::MOV_A, eax); break;

         // ── Memory stores ──
         case ir_op::i32_store:   emit_store(inst.ri.imm, base::MOV_B, eax); break;
         case ir_op::i64_store:   emit_store(inst.ri.imm, base::MOV_B, rax); break;
         case ir_op::f32_store:   emit_store(inst.ri.imm, base::MOV_B, eax); break;
         case ir_op::f64_store:   emit_store(inst.ri.imm, base::MOV_B, rax); break;
         case ir_op::i32_store8:  emit_store(inst.ri.imm, base::MOVB_B, al); break;
         case ir_op::i32_store16: emit_store(inst.ri.imm, base::MOVW_B, this->ax); break;
         case ir_op::i64_store8:  emit_store(inst.ri.imm, base::MOVB_B, al); break;
         case ir_op::i64_store16: emit_store(inst.ri.imm, base::MOVW_B, this->ax); break;
         case ir_op::i64_store32: emit_store(inst.ri.imm, base::MOV_B, eax); break;

         // ── Memory management ──
         case ir_op::memory_size:
            this->emit_push_raw(rdi);
            this->emit_push_raw(rsi);
            this->emit_bytes(0x48, 0xb8);
            this->emit_operand_ptr(&current_memory);
            this->emit(base::CALL, rax);
            this->emit_pop_raw(rsi);
            this->emit_pop_raw(rdi);
            this->emit_push_raw(rax);
            break;

         case ir_op::memory_grow:
            this->emit_pop_raw(rax);  // pages
            this->emit_push_raw(rdi);
            this->emit_push_raw(rsi);
            this->emit_mov(eax, esi); // pages arg in esi
            this->emit_bytes(0x48, 0xb8);
            this->emit_operand_ptr(&grow_memory);
            this->emit(base::CALL, rax);
            this->emit_pop_raw(rsi);
            this->emit_pop_raw(rdi);
            this->emit_push_raw(rax);
            break;

         // ── Select/drop ──
         case ir_op::drop:
            this->emit_pop_raw(rax);
            break;

         case ir_op::select:
            this->emit_pop_raw(rax);  // condition
            this->emit_pop_raw(rcx);  // val2
            this->emit_pop_raw(rdx);  // val1
            this->emit(base::TEST, eax, eax);
            this->emit_bytes(0x0f, 0x44, 0xd1); // cmovz %ecx, %edx
            this->emit_push_raw(rdx);
            break;

         // ── Division/remainder ──
         case ir_op::i32_div_s:
            this->emit_pop_raw(rcx);
            this->emit_pop_raw(rax);
            this->emit_bytes(0x99);        // cdq (sign-extend eax to edx:eax)
            this->emit_bytes(0xf7, 0xf9);  // idiv ecx
            this->emit_push_raw(rax);
            break;
         case ir_op::i32_div_u:
            this->emit_pop_raw(rcx);
            this->emit_pop_raw(rax);
            this->emit_xor(edx, edx);
            this->emit_bytes(0xf7, 0xf1);  // div ecx
            this->emit_push_raw(rax);
            break;
         case ir_op::i32_rem_s:
            this->emit_pop_raw(rcx);
            this->emit_pop_raw(rax);
            this->emit_cmp(-1, ecx);
            {
               void* skip = this->emit_branch8(base::JE);
               this->emit_bytes(0x99);        // cdq
               this->emit_bytes(0xf7, 0xf9);  // idiv ecx
               void* done = this->emit_branch8(base::JMP_8);
               base::fix_branch8(skip, code);
               this->emit_xor(edx, edx);      // result = 0 for -1 divisor
               base::fix_branch8(done, code);
            }
            this->emit_push_raw(rdx);
            break;
         case ir_op::i32_rem_u:
            this->emit_pop_raw(rcx);
            this->emit_pop_raw(rax);
            this->emit_xor(edx, edx);
            this->emit_bytes(0xf7, 0xf1);  // div ecx
            this->emit_push_raw(rdx);
            break;
         case ir_op::i64_div_s:
            this->emit_pop_raw(rcx);
            this->emit_pop_raw(rax);
            this->emit_bytes(0x48, 0x99);        // cqo
            this->emit_bytes(0x48, 0xf7, 0xf9);  // idiv rcx
            this->emit_push_raw(rax);
            break;
         case ir_op::i64_div_u:
            this->emit_pop_raw(rcx);
            this->emit_pop_raw(rax);
            this->emit_xor(edx, edx);
            this->emit_bytes(0x48, 0xf7, 0xf1);  // div rcx
            this->emit_push_raw(rax);
            break;
         case ir_op::i64_rem_s:
            this->emit_pop_raw(rcx);
            this->emit_pop_raw(rax);
            this->emit_bytes(0x48, 0x83, 0xf9, 0xff); // cmp $-1, rcx
            {
               void* skip = this->emit_branch8(base::JE);
               this->emit_bytes(0x48, 0x99);        // cqo
               this->emit_bytes(0x48, 0xf7, 0xf9);  // idiv rcx
               void* done = this->emit_branch8(base::JMP_8);
               base::fix_branch8(skip, code);
               this->emit_xor(edx, edx);
               base::fix_branch8(done, code);
            }
            this->emit_push_raw(rdx);
            break;
         case ir_op::i64_rem_u:
            this->emit_pop_raw(rcx);
            this->emit_pop_raw(rax);
            this->emit_xor(edx, edx);
            this->emit_bytes(0x48, 0xf7, 0xf1);  // div rcx
            this->emit_push_raw(rdx);
            break;

         // ── Rotates ──
         case ir_op::i32_rotl: emit_i32_shift(base::ROL_cl); break;
         case ir_op::i32_rotr: emit_i32_shift(base::ROR_cl); break;
         case ir_op::i64_rotl: emit_i64_shift(base::ROL_cl); break;
         case ir_op::i64_rotr: emit_i64_shift(base::ROR_cl); break;

         // ── Unary integer ops ──
         case ir_op::i32_clz:
            this->emit_pop_raw(rax);
            this->emit_bytes(0xf3, 0x0f, 0xbd, 0xc0); // lzcnt eax, eax
            this->emit_push_raw(rax);
            break;
         case ir_op::i32_ctz:
            this->emit_pop_raw(rax);
            this->emit_bytes(0xf3, 0x0f, 0xbc, 0xc0); // tzcnt eax, eax
            this->emit_push_raw(rax);
            break;
         case ir_op::i32_popcnt:
            this->emit_pop_raw(rax);
            this->emit_bytes(0xf3, 0x0f, 0xb8, 0xc0); // popcnt eax, eax
            this->emit_push_raw(rax);
            break;
         case ir_op::i64_clz:
            this->emit_pop_raw(rax);
            this->emit_bytes(0xf3, 0x48, 0x0f, 0xbd, 0xc0); // lzcnt rax, rax
            this->emit_push_raw(rax);
            break;
         case ir_op::i64_ctz:
            this->emit_pop_raw(rax);
            this->emit_bytes(0xf3, 0x48, 0x0f, 0xbc, 0xc0); // tzcnt rax, rax
            this->emit_push_raw(rax);
            break;
         case ir_op::i64_popcnt:
            this->emit_pop_raw(rax);
            this->emit_bytes(0xf3, 0x48, 0x0f, 0xb8, 0xc0); // popcnt rax, rax
            this->emit_push_raw(rax);
            break;

         // ── Conversions ──
         case ir_op::i32_wrap_i64:
            this->emit_pop_raw(rax);
            this->emit_bytes(0x89, 0xc0); // mov eax, eax (zero-extend)
            this->emit_push_raw(rax);
            break;
         case ir_op::i64_extend_s_i32:
            this->emit_bytes(0x48, 0x63, 0x04, 0x24); // movsxd (%rsp), %rax
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i64_extend_u_i32:
            this->emit_bytes(0x8b, 0x04, 0x24); // mov (%rsp), %eax (zero-extends)
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i32_extend8_s:
            this->emit_bytes(0x0f, 0xbe, 0x04, 0x24); // movsbl (%rsp), %eax
            this->emit_bytes(0x89, 0x04, 0x24);        // mov %eax, (%rsp)
            break;
         case ir_op::i32_extend16_s:
            this->emit_bytes(0x0f, 0xbf, 0x04, 0x24); // movswl (%rsp), %eax
            this->emit_bytes(0x89, 0x04, 0x24);
            break;
         case ir_op::i64_extend8_s:
            this->emit_bytes(0x48, 0x0f, 0xbe, 0x04, 0x24); // movsbq (%rsp), %rax
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i64_extend16_s:
            this->emit_bytes(0x48, 0x0f, 0xbf, 0x04, 0x24); // movswq (%rsp), %rax
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i64_extend32_s:
            this->emit_bytes(0x48, 0x63, 0x04, 0x24); // movsxd (%rsp), %rax
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i32_reinterpret_f32:
         case ir_op::i64_reinterpret_f64:
         case ir_op::f32_reinterpret_i32:
         case ir_op::f64_reinterpret_i64:
            // Bit patterns are identical — no-op on stack machine
            break;

         // ── Native SSE float ops ──

         // f32 unary
         case ir_op::f32_abs:
            // andl $0x7fffffff, (%rsp) — clear sign bit
            this->emit_bytes(0x81, 0x24, 0x24, 0xff, 0xff, 0xff, 0x7f);
            break;
         case ir_op::f32_neg:
            // xorl $0x80000000, (%rsp) — flip sign bit
            this->emit_bytes(0x81, 0x34, 0x24, 0x00, 0x00, 0x00, 0x80);
            break;
         case ir_op::f32_ceil:
            // roundss $0x0a, (%rsp), %xmm0; movss %xmm0, (%rsp)
            this->emit_bytes(0x66, 0x0f, 0x3a, 0x0a, 0x04, 0x24, 0x0a);
            this->emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
            break;
         case ir_op::f32_floor:
            this->emit_bytes(0x66, 0x0f, 0x3a, 0x0a, 0x04, 0x24, 0x09);
            this->emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
            break;
         case ir_op::f32_trunc:
            this->emit_bytes(0x66, 0x0f, 0x3a, 0x0a, 0x04, 0x24, 0x0b);
            this->emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
            break;
         case ir_op::f32_nearest:
            this->emit_bytes(0x66, 0x0f, 0x3a, 0x0a, 0x04, 0x24, 0x08);
            this->emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
            break;
         case ir_op::f32_sqrt:
            // sqrtss (%rsp), %xmm0; movss %xmm0, (%rsp)
            this->emit_bytes(0xf3, 0x0f, 0x51, 0x04, 0x24);
            this->emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
            break;

         // f32 binary: movss 8(%rsp), %xmm0; OPss (%rsp), %xmm0; lea 8(%rsp),%rsp; movss %xmm0, (%rsp)
         case ir_op::f32_add: emit_f32_binop_sse(0x58); break;
         case ir_op::f32_sub: emit_f32_binop_sse(0x5c); break;
         case ir_op::f32_mul: emit_f32_binop_sse(0x59); break;
         case ir_op::f32_div: emit_f32_binop_sse(0x5e); break;
         case ir_op::f32_min: emit_f32_binop_sse(0x5d); break; // minss
         case ir_op::f32_max: emit_f32_binop_sse(0x5f); break; // maxss
         case ir_op::f32_copysign:
            // Copy sign from rhs to lhs: lhs = (lhs & 0x7fffffff) | (rhs & 0x80000000)
            this->emit_pop_raw(rcx); // rhs
            this->emit_pop_raw(rax); // lhs
            this->emit_bytes(0x81, 0xe1, 0x00, 0x00, 0x00, 0x80); // and $0x80000000, %ecx
            this->emit_bytes(0x25, 0xff, 0xff, 0xff, 0x7f);       // and $0x7fffffff, %eax
            this->emit(base::OR_A, ecx, eax);
            this->emit_push_raw(rax);
            break;

         // f64 unary
         case ir_op::f64_abs:
            // btr $63, (%rsp) — clear sign bit
            this->emit_bytes(0x48, 0x0f, 0xba, 0x34, 0x24, 0x3f);
            break;
         case ir_op::f64_neg:
            // btc $63, (%rsp) — flip sign bit
            this->emit_bytes(0x48, 0x0f, 0xba, 0x3c, 0x24, 0x3f);
            break;
         case ir_op::f64_ceil:
            this->emit_bytes(0x66, 0x0f, 0x3a, 0x0b, 0x04, 0x24, 0x0a);
            this->emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
            break;
         case ir_op::f64_floor:
            this->emit_bytes(0x66, 0x0f, 0x3a, 0x0b, 0x04, 0x24, 0x09);
            this->emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
            break;
         case ir_op::f64_trunc:
            this->emit_bytes(0x66, 0x0f, 0x3a, 0x0b, 0x04, 0x24, 0x0b);
            this->emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
            break;
         case ir_op::f64_nearest:
            this->emit_bytes(0x66, 0x0f, 0x3a, 0x0b, 0x04, 0x24, 0x08);
            this->emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
            break;
         case ir_op::f64_sqrt:
            this->emit_bytes(0xf2, 0x0f, 0x51, 0x04, 0x24);
            this->emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
            break;

         // f64 binary
         case ir_op::f64_add: emit_f64_binop_sse(0x58); break;
         case ir_op::f64_sub: emit_f64_binop_sse(0x5c); break;
         case ir_op::f64_mul: emit_f64_binop_sse(0x59); break;
         case ir_op::f64_div: emit_f64_binop_sse(0x5e); break;
         case ir_op::f64_min: emit_f64_binop_sse(0x5d); break;
         case ir_op::f64_max: emit_f64_binop_sse(0x5f); break;
         case ir_op::f64_copysign:
            this->emit_pop_raw(rcx); // rhs
            this->emit_pop_raw(rax); // lhs
            // movabs $0x8000000000000000, %rdx
            this->emit_bytes(0x48, 0xba); this->emit_operand64(0x8000000000000000ull);
            this->emit(base::AND_A, rdx, rcx); // rcx = sign of rhs
            this->emit_bytes(0x48, 0xba); this->emit_operand64(0x7fffffffffffffffull);
            this->emit(base::AND_A, rdx, rax); // rax = magnitude of lhs
            this->emit(base::OR_A, rcx, rax);
            this->emit_push_raw(rax);
            break;

         // ── Float comparisons (SSE cmpss/cmpsd) ──
         // cmpCCss: 0=eq, 1=lt, 2=le, 4=ne (unordered=false)
         case ir_op::f32_eq: emit_f32_relop_sse(0x00, false, false); break;
         case ir_op::f32_ne: emit_f32_relop_sse(0x00, false, true); break;  // eq + flip
         case ir_op::f32_lt: emit_f32_relop_sse(0x01, false, false); break;
         case ir_op::f32_gt: emit_f32_relop_sse(0x01, true, false); break;  // lt with swapped args
         case ir_op::f32_le: emit_f32_relop_sse(0x02, false, false); break;
         case ir_op::f32_ge: emit_f32_relop_sse(0x02, true, false); break;  // le with swapped args

         case ir_op::f64_eq: emit_f64_relop_sse(0x00, false, false); break;
         case ir_op::f64_ne: emit_f64_relop_sse(0x00, false, true); break;
         case ir_op::f64_lt: emit_f64_relop_sse(0x01, false, false); break;
         case ir_op::f64_gt: emit_f64_relop_sse(0x01, true, false); break;
         case ir_op::f64_le: emit_f64_relop_sse(0x02, false, false); break;
         case ir_op::f64_ge: emit_f64_relop_sse(0x02, true, false); break;

         // ── Float-to-int conversions ──
         case ir_op::i32_trunc_s_f32:
            // cvttss2si (%rsp), %eax; mov %rax, (%rsp)
            this->emit_bytes(0xf3, 0x0f, 0x2c, 0x04, 0x24);
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i32_trunc_u_f32:
            // cvttss2si (%rsp), %rax (64-bit to catch unsigned range)
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2c, 0x04, 0x24);
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i32_trunc_s_f64:
            this->emit_bytes(0xf2, 0x0f, 0x2c, 0x04, 0x24);
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i32_trunc_u_f64:
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2c, 0x04, 0x24);
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i64_trunc_s_f32:
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2c, 0x04, 0x24);
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i64_trunc_u_f32:
            // TODO: handle unsigned i64 range (>= 2^63)
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2c, 0x04, 0x24);
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i64_trunc_s_f64:
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2c, 0x04, 0x24);
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i64_trunc_u_f64:
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2c, 0x04, 0x24);
            this->emit_mov(rax, *rsp);
            break;

         // Saturating truncations (same as above for now — TODO: clamp)
         case ir_op::i32_trunc_sat_f32_s:
            this->emit_bytes(0xf3, 0x0f, 0x2c, 0x04, 0x24);
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i32_trunc_sat_f32_u:
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2c, 0x04, 0x24);
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i32_trunc_sat_f64_s:
            this->emit_bytes(0xf2, 0x0f, 0x2c, 0x04, 0x24);
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i32_trunc_sat_f64_u:
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2c, 0x04, 0x24);
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i64_trunc_sat_f32_s:
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2c, 0x04, 0x24);
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i64_trunc_sat_f32_u:
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2c, 0x04, 0x24);
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i64_trunc_sat_f64_s:
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2c, 0x04, 0x24);
            this->emit_mov(rax, *rsp);
            break;
         case ir_op::i64_trunc_sat_f64_u:
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2c, 0x04, 0x24);
            this->emit_mov(rax, *rsp);
            break;

         // ── Int-to-float conversions ──
         case ir_op::f32_convert_s_i32:
            // cvtsi2ssl (%rsp), %xmm0; movss %xmm0, (%rsp)
            this->emit_bytes(0xf3, 0x0f, 0x2a, 0x04, 0x24);
            this->emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
            break;
         case ir_op::f32_convert_u_i32:
            // Use 64-bit convert to handle unsigned i32 range
            // movl (%rsp), %eax (zero-extend to 64-bit)
            this->emit_bytes(0x8b, 0x04, 0x24);
            // cvtsi2ssq %rax, %xmm0
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2a, 0xc0);
            this->emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
            break;
         case ir_op::f32_convert_s_i64:
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2a, 0x04, 0x24);
            this->emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
            break;
         case ir_op::f32_convert_u_i64:
            // TODO: handle unsigned i64 range properly
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2a, 0x04, 0x24);
            this->emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
            break;
         case ir_op::f64_convert_s_i32:
            this->emit_bytes(0xf2, 0x0f, 0x2a, 0x04, 0x24);
            this->emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
            break;
         case ir_op::f64_convert_u_i32:
            this->emit_bytes(0x8b, 0x04, 0x24);
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2a, 0xc0);
            this->emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
            break;
         case ir_op::f64_convert_s_i64:
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2a, 0x04, 0x24);
            this->emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
            break;
         case ir_op::f64_convert_u_i64:
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2a, 0x04, 0x24);
            this->emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
            break;

         // ── Float-float conversions ──
         case ir_op::f32_demote_f64:
            // cvtsd2ss (%rsp), %xmm0; movss %xmm0, (%rsp)
            this->emit_bytes(0xf2, 0x0f, 0x5a, 0x04, 0x24);
            this->emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
            break;
         case ir_op::f64_promote_f32:
            // cvtss2sd (%rsp), %xmm0; movsd %xmm0, (%rsp)
            this->emit_bytes(0xf3, 0x0f, 0x5a, 0x04, 0x24);
            this->emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
            break;

         default:
            break;
         }
      }

      void emit_function_epilogue(ir_function& func) {
         if (_use_regalloc) {
            if (func.type->return_count != 0 && func.vstack_top > 0) {
               uint32_t result_vreg = func.vstack[func.vstack_top - 1];
               load_vreg_rax(result_vreg);
            }
         } else {
            if (func.type->return_count != 0) {
               this->emit_pop_raw(rax);
            }
         }
         // Restore callee-saved registers from frame
         if (_use_regalloc && _callee_saved_used) {
            int32_t save_offset = -static_cast<int32_t>((_body_locals + _num_spill_slots + 1) * 8);
            if (_callee_saved_used & 1) { this->emit_mov(*(rbp + save_offset), r12); save_offset -= 8; }
            if (_callee_saved_used & 2) { this->emit_mov(*(rbp + save_offset), r13); save_offset -= 8; }
            if (_callee_saved_used & 4) { this->emit_mov(*(rbp + save_offset), r14); save_offset -= 8; }
            if (_callee_saved_used & 8) { this->emit_mov(*(rbp + save_offset), r15); save_offset -= 8; }
         }
         // Restore frame
         this->emit_mov(rbp, rsp);
         this->emit_pop_raw(rbp);
         this->emit(base::RET);
      }

      // ──────── Register-based IR emission ────────
      // Uses physical registers for vreg values instead of push/pop.
      // rax and rcx are temporaries. Vregs in physical registers are
      // accessed directly; spilled vregs use fixed rbp-offset slots.
      //
      // Returns true if handled, false to fall back to stack-based emission.
      bool emit_ir_inst_reg(ir_function& func, const ir_inst& inst, uint32_t idx) {
         switch (inst.opcode) {
         case ir_op::nop:
         case ir_op::block:
         case ir_op::loop:
         case ir_op::drop:
            return true;

         // Control flow — uses vregs for conditions, block fixups for branches
         case ir_op::if_: {
            load_vreg_rax(inst.br.src1);
            this->emit(base::TEST, eax, eax);
            void* branch = this->emit_branchcc32(base::JZ);
            push_if_fixup(branch);
            return true;
         }
         case ir_op::else_: {
            uint32_t target_block = inst.br.target;
            if (target_block < _num_blocks) {
               void* jmp = emit_jmp32();
               auto* fixup = _allocator.alloc<block_fixup>(1);
               fixup->branch = jmp;
               fixup->next = _block_fixups[target_block];
               _block_fixups[target_block] = fixup;
            }
            pop_if_fixup_to(code);
            return true;
         }
         case ir_op::br: {
            if (_in_br_table) {
               bool is_default = (_br_table_case >= _br_table_size);
               if (is_default) {
                  // Default: discard index, branch unconditionally
                  this->emit_pop_raw(rax);
                  emit_branch_to_block(func, inst.br.target, 0, types::pseudo);
                  _in_br_table = false;
               } else {
                  this->emit_bytes(0x81, 0x3c, 0x24);
                  this->emit_operand32(_br_table_case);
                  void* skip = this->emit_branchcc32(base::JNE);
                  this->emit_pop_raw(rax);
                  emit_branch_to_block(func, inst.br.target, 0, types::pseudo);
                  base::fix_branch(skip, code);
                  _br_table_case++;
               }
            } else {
               emit_branch_to_block(func, inst.br.target, 0, types::pseudo);
            }
            return true;
         }
         case ir_op::br_if: {
            load_vreg_rax(inst.br.src1); // condition
            this->emit(base::TEST, eax, eax);
            // Branch to target block if nonzero
            if (inst.br.target < _num_blocks) {
               if (_block_addrs[inst.br.target] != nullptr) {
                  void* branch = this->emit_branchcc32(base::JNZ);
                  base::fix_branch(branch, _block_addrs[inst.br.target]);
               } else {
                  void* branch = this->emit_branchcc32(base::JNZ);
                  auto* fixup = _allocator.alloc<block_fixup>(1);
                  fixup->branch = branch;
                  fixup->next = _block_fixups[inst.br.target];
                  _block_fixups[inst.br.target] = fixup;
               }
            }
            return true;
         }
         case ir_op::br_table: {
            // Push index to x86 stack for case comparisons (same as stack mode)
            load_vreg_rax(inst.rr.src1);
            this->emit_push_raw(rax);
            _br_table_case = 0;
            _br_table_size = inst.dest;
            _in_br_table = true;
            return true;
         }
         case ir_op::unreachable:
            emit_error_handler(&on_unreachable);
            return true;

         // arg: push a vreg value to the x86 stack (for upcoming call)
         case ir_op::arg: {
            load_vreg_rax(inst.rr.src1);
            this->emit_push_raw(rax);
            return true;
         }

         // call: args already pushed by arg instructions
         case ir_op::call: {
            uint32_t funcnum = inst.call.index;
            const func_type& ft = _mod.get_function_type(funcnum);
            if constexpr (!StackLimitIsBytes) {
               this->emit(base::DECD, ebx);
               base::fix_branch(this->emit_branchcc32(base::JZ), stack_overflow_handler);
            }
            void* branch = emit_call32();
            register_call(branch, funcnum);
            // Args were pushed by arg instructions — pop them
            uint32_t arg_bytes = 0;
            for (uint32_t p = 0; p < ft.param_types.size(); ++p)
               arg_bytes += (ft.param_types[p] == types::v128) ? 16 : 8;
            if (arg_bytes > 0)
               this->emit_add(arg_bytes, rsp);
            if constexpr (!StackLimitIsBytes) {
               this->emit(base::INCD, ebx);
            }
            // Store result to dest vreg (result already in rax)
            if (ft.return_count > 0 && inst.dest != ir_vreg_none) {
               store_rax_vreg(inst.dest);
            }
            return true;
         }
         case ir_op::call_indirect:
            return false; // TODO

         // Global access
         case ir_op::global_get: {
            auto loc = emit_global_loc(inst.local.index);
            this->emit_mov(loc, rax);
            store_rax_vreg(inst.dest);
            return true;
         }
         case ir_op::global_set: {
            load_vreg_rax(inst.local.src1);
            auto loc = emit_global_loc(inst.local.index);
            this->emit_mov(rax, loc);
            return true;
         }

         // Memory management
         case ir_op::memory_size:
         case ir_op::memory_grow:
            return false; // fall back for these

         case ir_op::const_i32: {
            uint32_t val = static_cast<uint32_t>(inst.imm64);
            int8_t pr = get_phys(inst.dest);
            if (pr >= 0) {
               this->emit_mov(val, phys_to_reg32(pr));
            } else {
               this->emit_mov(val, eax);
               store_rax_vreg(inst.dest);
            }
            return true;
         }
         case ir_op::const_i64: {
            uint64_t val = static_cast<uint64_t>(inst.imm64);
            int8_t pr = get_phys(inst.dest);
            if (pr >= 0) {
               this->emit_mov(val, phys_to_reg64(pr));
            } else {
               this->emit_mov(val, rax);
               store_rax_vreg(inst.dest);
            }
            return true;
         }

         // Integer binary ops
         case ir_op::i32_add: return emit_binop_reg(inst, [this](auto d, auto s){ this->emit_add(s, d); }, true);
         case ir_op::i32_sub: return emit_binop_reg(inst, [this](auto d, auto s){ this->emit_sub(s, d); }, true);
         case ir_op::i32_mul: return emit_binop_reg(inst, [this](auto d, auto s){ this->emit(base::IMUL, s, d); }, true);
         case ir_op::i32_and: return emit_binop_reg(inst, [this](auto d, auto s){ this->emit(base::AND_A, s, d); }, true);
         case ir_op::i32_or:  return emit_binop_reg(inst, [this](auto d, auto s){ this->emit(base::OR_A, s, d); }, true);
         case ir_op::i32_xor: return emit_binop_reg(inst, [this](auto d, auto s){ this->emit(base::XOR_A, s, d); }, true);

         case ir_op::i64_add: return emit_binop_reg(inst, [this](auto d, auto s){ this->emit_add(s, d); }, false);
         case ir_op::i64_sub: return emit_binop_reg(inst, [this](auto d, auto s){ this->emit_sub(s, d); }, false);
         case ir_op::i64_mul: return emit_binop_reg(inst, [this](auto d, auto s){ this->emit(base::IMUL, s, d); }, false);
         case ir_op::i64_and: return emit_binop_reg(inst, [this](auto d, auto s){ this->emit(base::AND_A, s, d); }, false);
         case ir_op::i64_or:  return emit_binop_reg(inst, [this](auto d, auto s){ this->emit(base::OR_A, s, d); }, false);
         case ir_op::i64_xor: return emit_binop_reg(inst, [this](auto d, auto s){ this->emit(base::XOR_A, s, d); }, false);

         // Shifts/rotates with constant folding
         case ir_op::i32_shl:   return emit_shift_reg(func, inst, 4, true);
         case ir_op::i32_shr_s: return emit_shift_reg(func, inst, 7, true);
         case ir_op::i32_shr_u: return emit_shift_reg(func, inst, 5, true);
         case ir_op::i32_rotl:  return emit_shift_reg(func, inst, 0, true);
         case ir_op::i32_rotr:  return emit_shift_reg(func, inst, 1, true);
         case ir_op::i64_shl:   return emit_shift_reg(func, inst, 4, false);
         case ir_op::i64_shr_s: return emit_shift_reg(func, inst, 7, false);
         case ir_op::i64_shr_u: return emit_shift_reg(func, inst, 5, false);
         case ir_op::i64_rotl:  return emit_shift_reg(func, inst, 0, false);
         case ir_op::i64_rotr:  return emit_shift_reg(func, inst, 1, false);

         // Unary
         case ir_op::i32_eqz: {
            load_vreg_rax(inst.rr.src1);
            this->emit(base::TEST, eax, eax);
            this->emit_setcc(base::JZ, al);
            this->emit_bytes(0x0f, 0xb6, 0xc0); // movzbl
            store_rax_vreg(inst.dest);
            return true;
         }

         // Comparisons
         case ir_op::i32_eq: return emit_relop_reg(inst, base::JE, true);
         case ir_op::i32_ne: return emit_relop_reg(inst, base::JNE, true);
         case ir_op::i32_lt_s: return emit_relop_reg(inst, base::JL, true);
         case ir_op::i32_lt_u: return emit_relop_reg(inst, base::JB, true);
         case ir_op::i32_gt_s: return emit_relop_reg(inst, base::JG, true);
         case ir_op::i32_gt_u: return emit_relop_reg(inst, base::JA, true);
         case ir_op::i32_le_s: return emit_relop_reg(inst, base::JLE, true);
         case ir_op::i32_le_u: return emit_relop_reg(inst, base::JBE, true);
         case ir_op::i32_ge_s: return emit_relop_reg(inst, base::JGE, true);
         case ir_op::i32_ge_u: return emit_relop_reg(inst, base::JAE, true);

         // Local access
         case ir_op::local_get: {
            int32_t offset = get_frame_offset(func, inst.local.index);
            this->emit_mov(*(rbp + offset), rax);
            store_rax_vreg(inst.dest);
            return true;
         }
         case ir_op::local_set: {
            int32_t offset = get_frame_offset(func, inst.local.index);
            load_vreg_rax(inst.local.src1);
            this->emit_mov(rax, *(rbp + offset));
            return true;
         }
         case ir_op::local_tee: {
            int32_t offset = get_frame_offset(func, inst.local.index);
            load_vreg_rax(inst.local.src1);
            this->emit_mov(rax, *(rbp + offset));
            // Value stays in src register (tee doesn't consume)
            return true;
         }

         // Memory loads (addr in src1, offset in imm)
         case ir_op::i32_load: return emit_load_reg(inst, base::MOV_A, eax);
         case ir_op::i64_load: return emit_load_reg(inst, base::MOV_A, rax);
         case ir_op::i32_load8_u: return emit_load_reg(inst, base::MOVZXB, eax);
         case ir_op::i32_load16_u: return emit_load_reg(inst, base::MOVZXW, eax);
         case ir_op::i32_load8_s: return emit_load_reg(inst, base::MOVSXB, eax);
         case ir_op::i32_load16_s: return emit_load_reg(inst, base::MOVSXW, eax);

         // Memory stores (addr in src1, value in src2... wait, IR uses ri.src1=addr, stores val separately)
         case ir_op::i32_store: return emit_store_reg(inst, base::MOV_B, eax);
         case ir_op::i64_store: return emit_store_reg(inst, base::MOV_B, rax);
         case ir_op::i32_store8: return emit_store_reg(inst, base::MOVB_B, al);
         case ir_op::i32_store16: return emit_store_reg(inst, base::MOVW_B, ax);

         // Return
         case ir_op::return_: {
            if (inst.rr.src1 != ir_vreg_none) {
               load_vreg_rax(inst.rr.src1);
            }
            this->emit_mov(rbp, rsp);
            this->emit_pop_raw(rbp);
            this->emit(base::RET);
            return true;
         }

         // Conversions that are just register ops
         case ir_op::i32_wrap_i64: {
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x89, 0xc0); // mov eax, eax (zero-extend)
            store_rax_vreg(inst.dest);
            return true;
         }
         case ir_op::i64_extend_u_i32: {
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x89, 0xc0); // mov eax, eax
            store_rax_vreg(inst.dest);
            return true;
         }
         case ir_op::i64_extend_s_i32: {
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x48, 0x63, 0xc0); // movsxd eax, rax
            store_rax_vreg(inst.dest);
            return true;
         }

         // Reinterpret — no-op, just transfer the register
         case ir_op::i32_reinterpret_f32:
         case ir_op::i64_reinterpret_f64:
         case ir_op::f32_reinterpret_i32:
         case ir_op::f64_reinterpret_i64: {
            if (inst.rr.src1 != ir_vreg_none && inst.dest != ir_vreg_none) {
               load_vreg_rax(inst.rr.src1);
               store_rax_vreg(inst.dest);
            }
            return true;
         }

         // Unary integer ops
         case ir_op::i32_clz:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0xf3, 0x0f, 0xbd, 0xc0); // lzcnt eax, eax
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i32_ctz:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0xf3, 0x0f, 0xbc, 0xc0); // tzcnt eax, eax
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i32_popcnt:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0xf3, 0x0f, 0xb8, 0xc0); // popcnt eax, eax
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_eqz:
            load_vreg_rax(inst.rr.src1);
            this->emit(base::TEST, rax, rax);
            this->emit_setcc(base::JZ, al);
            this->emit_bytes(0x0f, 0xb6, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_clz:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0xf3, 0x48, 0x0f, 0xbd, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_ctz:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0xf3, 0x48, 0x0f, 0xbc, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_popcnt:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0xf3, 0x48, 0x0f, 0xb8, 0xc0);
            store_rax_vreg(inst.dest);
            return true;

         // i64 comparisons
         case ir_op::i64_eq: return emit_relop_reg(inst, base::JE, false);
         case ir_op::i64_ne: return emit_relop_reg(inst, base::JNE, false);
         case ir_op::i64_lt_s: return emit_relop_reg(inst, base::JL, false);
         case ir_op::i64_lt_u: return emit_relop_reg(inst, base::JB, false);
         case ir_op::i64_gt_s: return emit_relop_reg(inst, base::JG, false);
         case ir_op::i64_gt_u: return emit_relop_reg(inst, base::JA, false);
         case ir_op::i64_le_s: return emit_relop_reg(inst, base::JLE, false);
         case ir_op::i64_le_u: return emit_relop_reg(inst, base::JBE, false);
         case ir_op::i64_ge_s: return emit_relop_reg(inst, base::JGE, false);
         case ir_op::i64_ge_u: return emit_relop_reg(inst, base::JAE, false);

         // Division/remainder
         case ir_op::i32_div_s:
            load_vreg_rcx(inst.rr.src2);
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x99); // cdq
            this->emit_bytes(0xf7, 0xf9); // idiv ecx
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i32_div_u:
            load_vreg_rcx(inst.rr.src2);
            load_vreg_rax(inst.rr.src1);
            this->emit_xor(edx, edx);
            this->emit_bytes(0xf7, 0xf1); // div ecx
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i32_rem_s:
            load_vreg_rcx(inst.rr.src2);
            load_vreg_rax(inst.rr.src1);
            this->emit_cmp(-1, ecx);
            { void* skip = this->emit_branch8(base::JE);
              this->emit_bytes(0x99, 0xf7, 0xf9);
              void* done = this->emit_branch8(base::JMP_8);
              base::fix_branch8(skip, code);
              this->emit_xor(edx, edx);
              base::fix_branch8(done, code); }
            this->emit_mov(edx, eax);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i32_rem_u:
            load_vreg_rcx(inst.rr.src2);
            load_vreg_rax(inst.rr.src1);
            this->emit_xor(edx, edx);
            this->emit_bytes(0xf7, 0xf1);
            this->emit_mov(edx, eax);
            store_rax_vreg(inst.dest);
            return true;

         // Select
         case ir_op::select: {
            // select: cond=src2 (in rr union it's stored differently)
            // Actually in our IR: select has dest, src1=val1, src2=val2
            // and the condition was a separate vpop... hmm
            // Fall back for select — it has 3 operands which our IR doesn't capture well
            return false;
         }

         // i32 sign extensions
         case ir_op::i32_extend8_s:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x0f, 0xbe, 0xc0); // movsbl al, eax
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i32_extend16_s:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x0f, 0xbf, 0xc0); // movswl ax, eax
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_extend8_s:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x48, 0x0f, 0xbe, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_extend16_s:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x48, 0x0f, 0xbf, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_extend32_s:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x48, 0x63, 0xc0); // movsxd eax, rax
            store_rax_vreg(inst.dest);
            return true;

         // Additional loads
         case ir_op::f32_load: return emit_load_reg(inst, base::MOV_A, eax);
         case ir_op::f64_load: return emit_load_reg(inst, base::MOV_A, rax);
         case ir_op::i64_load8_s: return emit_load_reg(inst, base::MOVSXB, rax);
         case ir_op::i64_load16_s: return emit_load_reg(inst, base::MOVSXW, rax);
         case ir_op::i64_load32_s: return emit_load_reg(inst, base::MOVSXD, rax);
         case ir_op::i64_load8_u: return emit_load_reg(inst, base::MOVZXB, eax);
         case ir_op::i64_load16_u: return emit_load_reg(inst, base::MOVZXW, eax);
         case ir_op::i64_load32_u: return emit_load_reg(inst, base::MOV_A, eax);

         // Additional stores
         case ir_op::f32_store: return emit_store_reg(inst, base::MOV_B, eax);
         case ir_op::f64_store: return emit_store_reg(inst, base::MOV_B, rax);
         case ir_op::i64_store8: return emit_store_reg(inst, base::MOVB_B, al);
         case ir_op::i64_store16: return emit_store_reg(inst, base::MOVW_B, ax);
         case ir_op::i64_store32: return emit_store_reg(inst, base::MOV_B, eax);

         // Const float (just store bits)
         case ir_op::const_f32: {
            uint32_t bits;
            memcpy(&bits, &inst.immf32, 4);
            this->emit_mov(bits, eax);
            store_rax_vreg(inst.dest);
            return true;
         }
         case ir_op::const_f64: {
            uint64_t bits;
            memcpy(&bits, &inst.immf64, 8);
            this->emit_mov(bits, rax);
            store_rax_vreg(inst.dest);
            return true;
         }

         default:
            // Unknown op — emit unreachable trap so execution stops immediately
            fprintf(stderr, "REGALLOC: unhandled op %u at inst %u\n", (unsigned)inst.opcode, idx);
            emit_error_handler(&on_unreachable);
            return true; // don't fall back to stack mode
         }
      }

      // Register-based binary op helper
      template<typename F>
      bool emit_binop_reg(const ir_inst& inst, F op, bool is32) {
         int8_t pr_d = get_phys(inst.dest);
         int8_t pr_s1 = get_phys(inst.rr.src1);
         int8_t pr_s2 = get_phys(inst.rr.src2);

         if (pr_d >= 0 && pr_s1 >= 0 && pr_s2 >= 0) {
            // All in registers — optimal path
            if (pr_d == pr_s1) {
               // dest = src1, just apply op with src2
               if (is32) op(phys_to_reg32(pr_d), phys_to_reg32(pr_s2));
               else      op(phys_to_reg64(pr_d), phys_to_reg64(pr_s2));
            } else if (pr_d == pr_s2) {
               // dest = src2, need temp to avoid clobbering src2
               // Use rax as temp: rax = src1, op(rax, src2), mov rax to dest
               if (is32) { this->emit_mov(phys_to_reg32(pr_s1), eax); op(eax, phys_to_reg32(pr_s2)); this->emit_mov(eax, phys_to_reg32(pr_d)); }
               else      { this->emit_mov(phys_to_reg64(pr_s1), rax); op(rax, phys_to_reg64(pr_s2)); this->emit_mov(rax, phys_to_reg64(pr_d)); }
            } else {
               // dest != src1 && dest != src2: mov src1 to dest, op with src2
               if (is32) { this->emit_mov(phys_to_reg32(pr_s1), phys_to_reg32(pr_d)); op(phys_to_reg32(pr_d), phys_to_reg32(pr_s2)); }
               else      { this->emit_mov(phys_to_reg64(pr_s1), phys_to_reg64(pr_d)); op(phys_to_reg64(pr_d), phys_to_reg64(pr_s2)); }
            }
         } else {
            // Some spilled — use rax/rcx temps
            load_vreg_rcx(inst.rr.src2);
            load_vreg_rax(inst.rr.src1);
            if (is32) op(eax, ecx);
            else      op(rax, rcx);
            store_rax_vreg(inst.dest);
         }
         return true;
      }

      // Register-based comparison helper
      bool emit_relop_reg(const ir_inst& inst, Jcc cc, bool is32) {
         load_vreg_rcx(inst.rr.src2);
         load_vreg_rax(inst.rr.src1);
         if (is32) this->emit_cmp(ecx, eax);
         else      this->emit_cmp(rcx, rax);
         this->emit_setcc(cc, al);
         this->emit_bytes(0x0f, 0xb6, 0xc0); // movzbl %al, %eax
         store_rax_vreg(inst.dest);
         return true;
      }

      // Register-based shift with constant folding
      bool emit_shift_reg(ir_function& func, const ir_inst& inst, uint8_t reg_field, bool is32) {
         // Check for constant shift amount
         uint32_t src2_vreg = inst.rr.src2;
         if (src2_vreg != ir_vreg_none) {
            for (uint32_t j = func.inst_count; j > 0; --j) {
               auto& prev = func.insts[j - 1];
               if (prev.dest == src2_vreg) {
                  if (prev.opcode == ir_op::const_i32 || prev.opcode == ir_op::const_i64) {
                     uint8_t amt = static_cast<uint8_t>(prev.imm64 & (is32 ? 0x1f : 0x3f));
                     prev.flags |= IR_DEAD;
                     load_vreg_rax(inst.rr.src1);
                     if (is32) this->emit_bytes(0xc1, static_cast<uint8_t>(0xc0 | (reg_field << 3)), amt);
                     else      this->emit_bytes(0x48, 0xc1, static_cast<uint8_t>(0xc0 | (reg_field << 3)), amt);
                     store_rax_vreg(inst.dest);
                     return true;
                  }
                  break;
               }
            }
         }
         // Variable shift
         load_vreg_rcx(inst.rr.src2);
         load_vreg_rax(inst.rr.src1);
         if (is32) this->emit_bytes(0xd3, static_cast<uint8_t>(0xc0 | (reg_field << 3)));
         else      this->emit_bytes(0x48, 0xd3, static_cast<uint8_t>(0xc0 | (reg_field << 3)));
         store_rax_vreg(inst.dest);
         return true;
      }

      // Register-based memory load
      template<class I, class R>
      bool emit_load_reg(const ir_inst& inst, I instr, R reg) {
         uint32_t uoffset = static_cast<uint32_t>(inst.ri.imm);
         load_vreg_rax(inst.ri.src1); // addr
         if (uoffset & 0x80000000u) {
            this->emit_mov(uoffset, ecx);
            this->emit_add(rcx, rax);
            this->emit(instr, *(rax + rsi + 0), reg);
         } else {
            this->emit(instr, *(rax + rsi + uoffset), reg);
         }
         store_rax_vreg(inst.dest);
         return true;
      }

      // Register-based memory store
      // inst.dest = value vreg, inst.ri.src1 = addr vreg, inst.ri.imm = offset
      template<class I, class R>
      bool emit_store_reg(const ir_inst& inst, I instr, R reg) {
         uint32_t uoffset = static_cast<uint32_t>(inst.ri.imm);
         load_vreg_rax(inst.dest);   // value
         load_vreg_rcx(inst.ri.src1); // addr
         if (uoffset & 0x80000000u) {
            this->emit_mov(uoffset, edx);
            this->emit_add(rdx, rcx);
            this->emit(instr, *(rcx + rsi + 0), reg);
         } else {
            this->emit(instr, *(rcx + rsi + uoffset), reg);
         }
         return true;
      }

      // ──────── Block address tracking for control flow ────────
      struct block_fixup {
         void* branch;        // Code address to patch
         block_fixup* next;
      };

      // Record that a block's code starts at current position
      void mark_block_start(uint32_t block_idx) {
         if (block_idx < _num_blocks) {
            _block_addrs[block_idx] = code;
         }
      }

      // Record that a block's code ends at current position (for forward branches)
      void mark_block_end(ir_function& func, uint32_t block_idx) {
         if (block_idx >= _num_blocks) return;
         _block_addrs[block_idx] = code;
         // Patch all pending forward references to this block
         for (auto* f = _block_fixups[block_idx]; f; f = f->next) {
            base::fix_branch(f->branch, code);
         }
         _block_fixups[block_idx] = nullptr;
         // For if-blocks without else: patch the if_ conditional branch here
         if (func.blocks[block_idx].is_if) {
            pop_if_fixup_to(code);
         }
      }

      // Emit an unconditional 32-bit relative jump, return address to patch
      void* emit_jmp32() {
         this->emit_bytes(0xe9);
         return this->emit_branch_target32();
      }

      // Emit a 32-bit relative jump to a block.
      // For loops: jump to block start (backward, already known).
      // For non-loops: jump to block end (forward, may need fixup).
      void emit_branch_to_block(ir_function& func, uint32_t block_idx, uint32_t depth_change, uint8_t rt) {
         if (block_idx >= _num_blocks) return;
         // Multipop: adjust stack for the branch
         emit_branch_multipop(depth_change, rt);
         // Check if target address is already known
         if (_block_addrs[block_idx] != nullptr) {
            // Address known — emit direct jump (backward branch to loop)
            void* branch = emit_jmp32();
            base::fix_branch(branch, _block_addrs[block_idx]);
         } else {
            // Forward branch — emit placeholder and record fixup
            void* branch = emit_jmp32();
            auto* fixup = _allocator.alloc<block_fixup>(1);
            fixup->branch = branch;
            fixup->next = _block_fixups[block_idx];
            _block_fixups[block_idx] = fixup;
         }
      }

      // Emit conditional branch to a block
      void emit_cond_branch_to_block(ir_function& func, uint32_t block_idx, uint32_t depth_change, uint8_t rt) {
         if (block_idx >= _num_blocks) return;
         // Pop condition and test
         this->emit_pop_raw(rax);
         this->emit(base::TEST, eax, eax);
         // If no stack adjustment needed, emit simple conditional branch
         bool needs_multipop = (depth_change > 0);
         if (!needs_multipop) {
            if (_block_addrs[block_idx] != nullptr) {
               void* branch = this->emit_branchcc32(base::JNZ);
               base::fix_branch(branch, _block_addrs[block_idx]);
            } else {
               void* branch = this->emit_branchcc32(base::JNZ);
               auto* fixup = _allocator.alloc<block_fixup>(1);
               fixup->branch = branch;
               fixup->next = _block_fixups[block_idx];
               _block_fixups[block_idx] = fixup;
            }
         } else {
            // Complex: jz skip; multipop; jmp target; skip:
            void* skip = this->emit_branchcc32(base::JZ);
            emit_branch_to_block(func, block_idx, depth_change, rt);
            base::fix_branch(skip, code);
         }
      }

      void emit_branch_multipop(uint32_t depth_change, uint8_t rt) {
         if (depth_change == 0) return;
         if (rt != types::pseudo) {
            this->emit_mov(*rsp, rax);  // Save return value
            this->emit_add(static_cast<uint32_t>(depth_change * 8), rsp);
            this->emit_push_raw(rax);
         } else {
            this->emit_add(static_cast<uint32_t>(depth_change * 8), rsp);
         }
      }

      // ──────── Register cache (eliminates adjacent push/pop pairs) ────────
      // A 2-element cache of values that have been "pushed" but are still
      // in registers. When a pop is requested, check the cache first.
      static constexpr int REG_CACHE_SIZE = 2;
      struct cached_value {
         bool valid = false;
         general_register64 reg;
      };
      cached_value _reg_cache[REG_CACHE_SIZE];
      int _cache_top = 0;

      // Push a value: if cache has space, keep it in register
      void cached_push(general_register64 reg) {
         if (_cache_top < REG_CACHE_SIZE) {
            _reg_cache[_cache_top++] = {true, reg};
         } else {
            // Cache full — flush oldest and add new
            flush_cache();
            _reg_cache[_cache_top++] = {true, reg};
         }
      }

      // Pop a value into a register: check cache first
      void cached_pop(general_register64 dest) {
         if (_cache_top > 0 && _reg_cache[_cache_top - 1].valid) {
            auto& top = _reg_cache[--_cache_top];
            if (top.reg != dest) {
               this->emit_mov(top.reg, dest);
            }
            top.valid = false;
         } else {
            this->emit_pop_raw(dest);
         }
      }

      // Flush all cached values to the stack
      void flush_cache() {
         for (int i = 0; i < _cache_top; ++i) {
            if (_reg_cache[i].valid) {
               this->emit_push_raw(_reg_cache[i].reg);
               _reg_cache[i].valid = false;
            }
         }
         _cache_top = 0;
      }

      // ──────── Register allocation helpers ────────
      // Map phys_reg index to x86 register
      // Must match phys_reg enum: rdx=0, r8=1, r9=2, r10=3, r11=4
      // rax and rcx are reserved as temporaries for spill loads
      static constexpr general_register64 phys_to_reg64(int8_t pr) {
         // rax/rcx/rdx reserved as temps. Map: r8=0, r9=1, r10=2, r11=3, r12-r15=4-7
         constexpr general_register64 map[] = {
            general_register64(8),  // r8
            general_register64(9),  // r9
            general_register64(10), // r10
            general_register64(11), // r11
            general_register64(12), // r12 (callee-saved)
            general_register64(13), // r13 (callee-saved)
            general_register64(14), // r14 (callee-saved)
            general_register64(15), // r15 (callee-saved)
         };
         return map[pr];
      }
      static constexpr general_register32 phys_to_reg32(int8_t pr) {
         constexpr general_register32 map[] = {
            general_register32(8),  // r8d
            general_register32(9),  // r9d
            general_register32(10), // r10d
            general_register32(11), // r11d
            general_register32(12), // r12d
            general_register32(13), // r13d
            general_register32(14), // r14d
            general_register32(15), // r15d
         };
         return map[pr];
      }

      // Check if a vreg has a physical register assigned
      bool has_reg(uint32_t vreg) const {
         return _vreg_map && vreg < _num_vregs && _vreg_map[vreg] >= 0;
      }
      int8_t get_phys(uint32_t vreg) const {
         if (!_vreg_map || vreg >= _num_vregs) return -1;
         return _vreg_map[vreg];
      }

      // Load a vreg value into rax (temp register for operand loading)
      void load_vreg_rax(uint32_t vreg) {
         if (vreg == ir_vreg_none) return;
         int8_t pr = get_phys(vreg);
         if (pr >= 0) {
            this->emit_mov(phys_to_reg64(pr), rax);
         } else if (_spill_map && vreg < _num_vregs && _spill_map[vreg] >= 0) {
            int32_t off = get_spill_offset(_spill_map[vreg]);
            this->emit_mov(*(rbp + off), rax);
         } else if (_use_regalloc) {
            // Vreg has no register and no spill slot — shouldn't happen
            // This indicates a bug in live interval computation
            fprintf(stderr, "BUG: vreg %u has no register and no spill slot!\n", vreg);
         }
      }

      // Load a vreg value into rcx (second temp)
      void load_vreg_rcx(uint32_t vreg) {
         if (vreg == ir_vreg_none) return;
         int8_t pr = get_phys(vreg);
         if (pr >= 0) {
            this->emit_mov(phys_to_reg64(pr), rcx);
         } else if (_spill_map && vreg < _num_vregs && _spill_map[vreg] >= 0) {
            this->emit_mov(*(rbp + get_spill_offset(_spill_map[vreg])), rcx);
         }
      }

      // Store rax value to a vreg's home (register or spill slot)
      void store_rax_vreg(uint32_t vreg) {
         if (vreg == ir_vreg_none) return;
         int8_t pr = get_phys(vreg);
         if (pr >= 0) {
            this->emit_mov(rax, phys_to_reg64(pr));
         } else if (_spill_map && vreg < _num_vregs && _spill_map[vreg] >= 0) {
            this->emit_mov(rax, *(rbp + get_spill_offset(_spill_map[vreg])));
         }
      }

      // Get rbp-relative offset for a spill slot
      // Spill slots are after body locals: rbp - (body_locals + slot + 1) * 8
      int32_t get_spill_offset(int16_t slot) const {
         return -static_cast<int32_t>((_body_locals + static_cast<uint32_t>(slot) + 1) * 8);
      }

      // Find the spill slot for a vreg (search intervals)
      int16_t get_spill_slot(uint32_t vreg) const {
         // Linear search — could be optimized with a vreg→spill_slot map
         // but this is only called for spilled vregs (rare path)
         return -1; // TODO: look up from intervals
      }

      // ──────── SSE float helpers ────────
      void emit_f32_binop_sse(uint8_t op) {
         // movss 8(%rsp), %xmm0
         this->emit_bytes(0xf3, 0x0f, 0x10, 0x44, 0x24, 0x08);
         // OPss (%rsp), %xmm0
         this->emit_bytes(0xf3, 0x0f, op, 0x04, 0x24);
         // lea 8(%rsp), %rsp
         this->emit_bytes(0x48, 0x8d, 0x64, 0x24, 0x08);
         // movss %xmm0, (%rsp)
         this->emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
      }

      void emit_f64_binop_sse(uint8_t op) {
         this->emit_bytes(0xf2, 0x0f, 0x10, 0x44, 0x24, 0x08);
         this->emit_bytes(0xf2, 0x0f, op, 0x04, 0x24);
         this->emit_bytes(0x48, 0x8d, 0x64, 0x24, 0x08);
         this->emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
      }

      void emit_f32_relop_sse(uint8_t cmp_op, bool swap, bool flip) {
         if (swap) {
            this->emit_bytes(0xf3, 0x0f, 0x10, 0x04, 0x24);        // movss (%rsp), %xmm0
            this->emit_bytes(0xf3, 0x0f, 0xc2, 0x44, 0x24, 0x08, cmp_op); // cmpss 8(%rsp), %xmm0
         } else {
            this->emit_bytes(0xf3, 0x0f, 0x10, 0x44, 0x24, 0x08);  // movss 8(%rsp), %xmm0
            this->emit_bytes(0xf3, 0x0f, 0xc2, 0x04, 0x24, cmp_op);       // cmpss (%rsp), %xmm0
         }
         this->emit_bytes(0x66, 0x0f, 0x7e, 0xc0); // movd %xmm0, %eax
         if (!flip) {
            this->emit_bytes(0x83, 0xe0, 0x01);     // and $1, %eax
         } else {
            this->emit_bytes(0xff, 0xc0);            // inc %eax (0xffffffff→0, 0→1)
         }
         this->emit_bytes(0x48, 0x8d, 0x64, 0x24, 0x10); // lea 16(%rsp), %rsp
         this->emit_push_raw(rax);
      }

      void emit_f64_relop_sse(uint8_t cmp_op, bool swap, bool flip) {
         if (swap) {
            this->emit_bytes(0xf2, 0x0f, 0x10, 0x04, 0x24);
            this->emit_bytes(0xf2, 0x0f, 0xc2, 0x44, 0x24, 0x08, cmp_op);
         } else {
            this->emit_bytes(0xf2, 0x0f, 0x10, 0x44, 0x24, 0x08);
            this->emit_bytes(0xf2, 0x0f, 0xc2, 0x04, 0x24, cmp_op);
         }
         this->emit_bytes(0x66, 0x0f, 0x7e, 0xc0);
         if (!flip) {
            this->emit_bytes(0x83, 0xe0, 0x01);
         } else {
            this->emit_bytes(0xff, 0xc0);
         }
         this->emit_bytes(0x48, 0x8d, 0x64, 0x24, 0x10);
         this->emit_push_raw(rax);
      }

      void emit_operand64(uint64_t val) {
         for (int i = 0; i < 8; ++i) {
            this->emit_bytes(static_cast<uint8_t>(val >> (i * 8)));
         }
      }

      // ──────── If fixup stack ────────
      // ONLY used for if_ instructions (not blocks or loops).
      // The if_ conditional branch is stored here until else_ or block end patches it.
      static constexpr uint32_t MAX_IF_DEPTH = 256;
      void* _if_fixups[MAX_IF_DEPTH];
      uint32_t _if_fixup_top = 0;

      void push_if_fixup(void* branch) {
         if (_if_fixup_top < MAX_IF_DEPTH) {
            _if_fixups[_if_fixup_top++] = branch;
         }
      }
      void pop_if_fixup_to(void* target) {
         if (_if_fixup_top > 0) {
            void* branch = _if_fixups[--_if_fixup_top];
            if (branch && target) {
               base::fix_branch(branch, target);
            }
         }
      }

      // ──────── Global access helper ────────
      auto emit_global_loc(uint32_t globalidx) {
         auto offset = _mod.get_global_offset(globalidx);
         this->emit_mov(*(rsi + (wasm_allocator::globals_end() - 8)), rcx);
         if (offset > 0x7fffffff) {
            this->emit_mov(static_cast<std::uint64_t>(offset), rdx);
            this->emit_add(rdx, rcx);
            offset = 0;
         }
         return *(rcx + static_cast<std::int32_t>(offset));
      }

      // ──────── Frame offset calculation ────────
      // Compute the rbp-relative offset for a local variable.
      // Parameters are above rbp (positive offsets), locals below (negative).
      int32_t get_frame_offset(const ir_function& func, uint32_t local_idx) {
         const func_type* ft = func.type;
         if (local_idx < ft->param_types.size()) {
            // Parameter: above rbp. Caller pushes in WASM order (param0 first),
            // so param[N-1] is at rbp+16, param[N-2] at rbp+16+size(N-1), etc.
            int32_t offset = 16; // skip saved rbp + return address
            for (uint32_t i = ft->param_types.size(); i-- > 0; ) {
               if (i == local_idx) return offset;
               offset += (ft->param_types[i] == types::v128) ? 16 : 8;
            }
            return offset; // shouldn't reach
         } else {
            // Local: below rbp (negative offset)
            uint32_t li = local_idx - ft->param_types.size();
            const auto& locals = _mod.code[func.func_index].locals;
            int32_t offset = 0;
            uint32_t count = 0;
            for (uint32_t g = 0; g < locals.size(); ++g) {
               uint8_t size = (locals[g].type == types::v128) ? 16 : 8;
               if (li < count + locals[g].count) {
                  offset -= static_cast<int32_t>((li - count + 1) * size);
                  return offset;
               }
               count += locals[g].count;
               offset -= static_cast<int32_t>(locals[g].count * size);
            }
            return offset; // shouldn't reach here
         }
      }

      // ──────── Memory access helpers ────────
      template<class I, class R>
      void emit_load(int32_t offset, I instr, R reg) {
         uint32_t uoffset = static_cast<uint32_t>(offset);
         this->emit_pop_raw(rax);  // WASM address
         if (uoffset & 0x80000000u) {
            this->emit_mov(uoffset, ecx);
            this->emit_add(rcx, rax);
            this->emit(instr, *(rax + rsi + 0), reg);
         } else {
            this->emit(instr, *(rax + rsi + uoffset), reg);
         }
         this->emit_push_raw(rax);
      }

      template<class I, class R>
      void emit_store(int32_t offset, I instr, R reg) {
         uint32_t uoffset = static_cast<uint32_t>(offset);
         this->emit_pop_raw(rax);  // value
         this->emit_pop_raw(rcx);  // WASM address
         if (uoffset & 0x80000000u) {
            this->emit_mov(uoffset, edx);
            this->emit_add(rdx, rcx);
            this->emit(instr, *(rcx + rsi + 0), reg);
         } else {
            this->emit(instr, *(rcx + rsi + uoffset), reg);
         }
      }

      // ──────── Call helpers ────────
      // Emit a 32-bit relative call instruction, returns the address to patch
      void* emit_call32() {
         this->emit_bytes(0xe8);
         void* result = code;
         this->emit_operand32(0); // placeholder
         return result;
      }

      // Pop params and push result after a call
      void emit_call_multipop(const func_type& ft) {
         uint32_t total_size = 0;
         for (uint32_t i = 0; i < ft.param_types.size(); ++i) {
            total_size += (ft.param_types[i] == types::v128) ? 16 : 8;
         }
         if (total_size != 0) {
            this->emit_add(total_size, rsp);
         }
         if (ft.return_count != 0) {
            this->emit_push_raw(rax);
         }
      }

      // ──────── Binary op helpers ────────

      // Register-aware binary op: uses physical registers if available
      template<typename F>
      void emit_binop_ra(const ir_inst& inst, F op, bool is32) {
         int8_t pr_dest = get_phys(inst.dest);
         int8_t pr_src1 = get_phys(inst.rr.src1);
         int8_t pr_src2 = get_phys(inst.rr.src2);

            {
            // Fallback to stack-based
            this->emit_pop_raw(rcx);
            this->emit_pop_raw(rax);
            if (is32) op(eax, ecx);
            else      op(rax, rcx);
            this->emit_push_raw(rax);
         }
      }

      template<typename F>
      void emit_i32_binop(F op) {
         this->emit_pop_raw(rcx);
         this->emit_pop_raw(rax);
         op();
         this->emit_push_raw(rax);
      }

      template<typename F>
      void emit_i64_binop(F op) {
         this->emit_pop_raw(rcx);
         this->emit_pop_raw(rax);
         op();
         this->emit_push_raw(rax);
      }

      template<typename ShiftOp>
      void emit_i32_shift(ShiftOp op) {
         this->emit_pop_raw(rcx);
         this->emit_pop_raw(rax);
         this->emit(op, eax);
         this->emit_push_raw(rax);
      }

      template<typename ShiftOp>
      void emit_i64_shift(ShiftOp op) {
         this->emit_pop_raw(rcx);
         this->emit_pop_raw(rax);
         this->emit(op, rax);
         this->emit_push_raw(rax);
      }

      // Optimized shift: if the shift count (src2) is a constant, use immediate form
      void emit_shift_opt(ir_function& func, const ir_inst& inst, uint8_t reg_field, bool is32) {
         // Check if src2 is a const_i32/const_i64
         uint32_t src2_vreg = inst.rr.src2;
         bool found_const = false;
         uint8_t shift_amount = 0;

         if (src2_vreg != ir_vreg_none) {
            // Search backward for the const instruction that defines src2
            for (uint32_t j = func.inst_count; j > 0; --j) {
               auto& prev = func.insts[j - 1];
               if (prev.dest == src2_vreg) {
                  if (prev.opcode == ir_op::const_i32 || prev.opcode == ir_op::const_i64) {
                     shift_amount = static_cast<uint8_t>(prev.imm64 & (is32 ? 0x1f : 0x3f));
                     found_const = true;
                     prev.flags |= IR_DEAD; // Mark const as dead
                  }
                  break;
               }
            }
         }

         if (found_const) {
            // Const was already pushed to stack — discard it, keep the value
            this->emit_pop_raw(rcx); // discard the constant
            this->emit_pop_raw(rax); // the value to shift
            // Emit immediate shift: C1 /reg_field imm8 (32-bit) or 48 C1 /reg_field imm8 (64-bit)
            if (is32) {
               this->emit_bytes(0xc1, static_cast<uint8_t>(0xc0 | (reg_field << 3)), shift_amount);
            } else {
               this->emit_bytes(0x48, 0xc1, static_cast<uint8_t>(0xc0 | (reg_field << 3)), shift_amount);
            }
            this->emit_push_raw(rax);
         } else {
            // Variable shift — use cl register
            this->emit_pop_raw(rcx);
            this->emit_pop_raw(rax);
            // D3 /reg_field (32-bit) or 48 D3 /reg_field (64-bit)
            if (is32) {
               this->emit_bytes(0xd3, static_cast<uint8_t>(0xc0 | (reg_field << 3)));
            } else {
               this->emit_bytes(0x48, 0xd3, static_cast<uint8_t>(0xc0 | (reg_field << 3)));
            }
            this->emit_push_raw(rax);
         }
      }

      void emit_i32_relop(Jcc cc) {
         this->emit_pop_raw(rcx);  // rhs
         this->emit_pop_raw(rax);  // lhs
         this->emit_xor(edx, edx); // zero BEFORE cmp (xor clobbers flags)
         this->emit_cmp(ecx, eax);
         this->emit_setcc(cc, dl);
         this->emit_mov(edx, eax);
         this->emit_push_raw(rax);
      }

      void emit_i64_relop(Jcc cc) {
         this->emit_pop_raw(rcx);
         this->emit_pop_raw(rax);
         this->emit_xor(edx, edx);
         this->emit_cmp(rcx, rax);
         this->emit_setcc(cc, dl);
         this->emit_mov(edx, eax);
         this->emit_push_raw(rax);
      }

      // ──────── Function relocation ────────
      // Linked list node for pending forward-reference fixups.
      // Allocated from growable_allocator — no malloc, reclaimed by end_code.
      struct call_fixup {
         void* branch;       // Code address of the branch to patch
         call_fixup* next;   // Next pending fixup for the same target, or nullptr
      };

      // Each function has either a resolved address (non-null) or a linked list
      // of pending fixups (address is null, pending_fixups points to the list).
      struct func_reloc {
         void* address = nullptr;         // Resolved code address, or nullptr if unresolved
         call_fixup* pending = nullptr;   // Linked list of pending fixups
      };

      void init_relocations() {
         uint32_t total = _mod.get_functions_total();
         _relocs = _allocator.alloc<func_reloc>(total);
         _num_relocs = total;
         for (uint32_t i = 0; i < total; ++i) {
            _relocs[i] = func_reloc{};
         }
      }

      void register_call(void* branch_addr, uint32_t funcnum) {
         if (funcnum >= _num_relocs) return;
         auto& r = _relocs[funcnum];
         if (r.address) {
            // Already compiled — patch immediately
            base::fix_branch(branch_addr, r.address);
         } else {
            // Forward reference — add to linked list
            auto* fixup = _allocator.alloc<call_fixup>(1);
            fixup->branch = branch_addr;
            fixup->next = r.pending;
            r.pending = fixup;
         }
      }

      void start_function(void* func_start, uint32_t funcnum) {
         if (funcnum >= _num_relocs) return;
         auto& r = _relocs[funcnum];
         // Patch all pending forward references
         for (auto* f = r.pending; f; f = f->next) {
            base::fix_branch(f->branch, func_start);
         }
         r.address = func_start;
         r.pending = nullptr;
      }

      // ──────── Static callbacks (same as machine_code_writer) ────────
      static native_value call_host_function(Context* context, native_value* stack, uint32_t idx, uint32_t remaining_stack) {
         native_value result;
         vm::longjmp_on_exception([&]() {
            auto saved = context->_remaining_call_depth;
            context->_remaining_call_depth = remaining_stack;
            scope_guard g{[&](){ context->_remaining_call_depth = saved; }};
            result = context->call_host_function(stack, idx);
         });
         return result;
      }

      static int32_t current_memory(Context* context) { return context->current_linear_memory(); }
      static int32_t grow_memory(Context* context, int32_t pages) { return context->grow_linear_memory(pages); }
      static void on_memory_error() { throw_<wasm_memory_exception>("wasm memory out-of-bounds"); }
      static void on_unreachable() { vm::throw_<wasm_interpreter_exception>("unreachable"); }
      static void on_fp_error() { vm::throw_<wasm_interpreter_exception>("floating point error"); }
      static void on_call_indirect_error() { vm::throw_<wasm_interpreter_exception>("call_indirect out of range"); }
      static void on_type_error() { vm::throw_<wasm_interpreter_exception>("call_indirect incorrect function type"); }
      static void on_stack_overflow() { vm::throw_<wasm_interpreter_exception>("stack overflow"); }

      // ──────── State ────────
      growable_allocator& _allocator;
      module& _mod;
      void* _code_segment_base;
      void* fpe_handler = nullptr;
      void* call_indirect_handler = nullptr;
      void* type_error_handler = nullptr;
      void* stack_overflow_handler = nullptr;
      void* memory_handler = nullptr;
      func_reloc* _relocs = nullptr;
      uint32_t _num_relocs = 0;
      // Per-function block address tracking (set during compile_function)
      void** _block_addrs = nullptr;
      block_fixup** _block_fixups = nullptr;
      uint32_t _num_blocks = 0;
      // (if/else fixups stored in block_fixups, no separate stack)
      // br_table state
      bool _in_br_table = false;
      uint32_t _br_table_case = 0;
      uint32_t _br_table_size = 0;
      // Register allocation mapping
      int8_t* _vreg_map = nullptr;    // vreg → phys_reg (-1 = spilled)
      int16_t* _spill_map = nullptr;  // vreg → spill_slot (-1 = in register)
      uint32_t _num_vregs = 0;
      uint32_t _num_spill_slots = 0;
      uint32_t _body_locals = 0;
      bool _use_regalloc = false;
      uint32_t _callee_saved_used = 0;
      uint32_t _callee_saved_count = 0;
   };

}} // namespace eosio::vm
