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
      using base::al; using base::cl;
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
         _code_segment_base = code_segment_base ? code_segment_base : _allocator.start_code();
         init_relocations();
      }

      // Emit the SysV ABI entry point (same as machine_code_writer)
      void emit_entry_and_error_handlers() {
         // Allocate space for sysv_abi_interface
         auto* buf = _allocator.alloc<unsigned char>(256);
         code = buf;
         emit_sysv_abi_interface();
         _allocator.reclaim(code, buf + 256 - code);

         // Error handlers (5 * 16 bytes)
         buf = _allocator.alloc<unsigned char>(80);
         code = buf;
         fpe_handler = emit_error_handler(&on_fp_error);
         call_indirect_handler = emit_error_handler(&on_call_indirect_error);
         type_error_handler = emit_error_handler(&on_type_error);
         stack_overflow_handler = emit_error_handler(&on_stack_overflow);
         memory_handler = emit_error_handler(&on_memory_error);
         _allocator.reclaim(code, buf + 80 - code);

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
            _allocator.reclaim(code, buf + host_functions_size - code);
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

         start_function(code, func.func_index + _mod.get_imported_functions_size());

         // Emit function prologue
         emit_function_prologue(func);

         // Emit each IR instruction as stack-machine code (naive: all spilled)
         for (uint32_t i = 0; i < func.inst_count; ++i) {
            // Check if any blocks start at this instruction index
            for (uint32_t b = 0; b < func.block_count; ++b) {
               if (func.blocks[b].start == i && _block_addrs[b] == nullptr) {
                  mark_block_start(b);
               }
            }
            emit_ir_inst(func, func.insts[i], i);
            // Check if any blocks end at this instruction index
            for (uint32_t b = 0; b < func.block_count; ++b) {
               if (func.blocks[b].end == i + 1) {
                  mark_block_end(b);
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

         // Check stack depth
         if constexpr (!StackLimitIsBytes) {
            this->emit(base::DECD, ebx);
            base::fix_branch(this->emit_branchcc32(base::JZ), stack_overflow_handler);
         }

         // Allocate space for locals (params are pushed by caller)
         // Each local gets 8 bytes (or 16 for v128)
         uint32_t total_local_slots = func.num_locals;
         if (total_local_slots > 0) {
            // Initialize locals to zero
            this->emit_xor(eax, eax);
            for (uint32_t i = 0; i < total_local_slots; ++i) {
               this->emit_push_raw(rax);
            }
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
         case ir_op::i32_add: emit_i32_binop([this]{ this->emit_add(ecx, eax); }); break;
         case ir_op::i32_sub: emit_i32_binop([this]{ this->emit_sub(ecx, eax); }); break;
         case ir_op::i32_mul: emit_i32_binop([this]{ this->emit(base::IMUL, ecx, eax); }); break;
         case ir_op::i32_and: emit_i32_binop([this]{ this->emit(base::AND_A, ecx, eax); }); break;
         case ir_op::i32_or:  emit_i32_binop([this]{ this->emit(base::OR_A, ecx, eax); }); break;
         case ir_op::i32_xor: emit_i32_binop([this]{ this->emit(base::XOR_A, ecx, eax); }); break;

         case ir_op::i64_add: emit_i64_binop([this]{ this->emit_add(rcx, rax); }); break;
         case ir_op::i64_sub: emit_i64_binop([this]{ this->emit_sub(rcx, rax); }); break;
         case ir_op::i64_mul: emit_i64_binop([this]{ this->emit(base::IMUL, rcx, rax); }); break;
         case ir_op::i64_and: emit_i64_binop([this]{ this->emit(base::AND_A, rcx, rax); }); break;
         case ir_op::i64_or:  emit_i64_binop([this]{ this->emit(base::OR_A, rcx, rax); }); break;
         case ir_op::i64_xor: emit_i64_binop([this]{ this->emit(base::XOR_A, rcx, rax); }); break;

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
               // Return value is already on stack (top of stack)
               this->emit_pop_raw(rax);
            }
            // Restore frame
            this->emit_mov(rbp, rsp);
            this->emit_pop_raw(rbp);
            if constexpr (!StackLimitIsBytes) {
               this->emit(base::INCD, ebx);
            }
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
         case ir_op::end:
         case ir_op::if_:
         case ir_op::else_:
            break;

         // ── Control flow (branches) ──
         case ir_op::br:
            emit_branch_to_block(func, inst.br.target, inst.dest, inst.type);
            break;

         case ir_op::br_if:
            emit_cond_branch_to_block(func, inst.br.target, inst.dest, inst.type);
            break;

         case ir_op::br_table:
            // br_table is followed by individual br instructions for each case
            // The index was already popped; cases will use it
            // TODO: implement proper br_table with jump table
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
         case ir_op::call_indirect:
            // TODO: Full call_indirect with table lookup
            emit_error_handler(&on_call_indirect_error);
            break;

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

         // ── Float ops and remaining conversions ──
         // These require softfloat support; for now emit TODO stubs
         // that will be filled in when we integrate softfloat calling
         case ir_op::f32_abs: case ir_op::f32_neg: case ir_op::f32_ceil:
         case ir_op::f32_floor: case ir_op::f32_trunc: case ir_op::f32_nearest:
         case ir_op::f32_sqrt: case ir_op::f32_add: case ir_op::f32_sub:
         case ir_op::f32_mul: case ir_op::f32_div: case ir_op::f32_min:
         case ir_op::f32_max: case ir_op::f32_copysign:
         case ir_op::f64_abs: case ir_op::f64_neg: case ir_op::f64_ceil:
         case ir_op::f64_floor: case ir_op::f64_trunc: case ir_op::f64_nearest:
         case ir_op::f64_sqrt: case ir_op::f64_add: case ir_op::f64_sub:
         case ir_op::f64_mul: case ir_op::f64_div: case ir_op::f64_min:
         case ir_op::f64_max: case ir_op::f64_copysign:
         case ir_op::f32_eq: case ir_op::f32_ne: case ir_op::f32_lt:
         case ir_op::f32_gt: case ir_op::f32_le: case ir_op::f32_ge:
         case ir_op::f64_eq: case ir_op::f64_ne: case ir_op::f64_lt:
         case ir_op::f64_gt: case ir_op::f64_le: case ir_op::f64_ge:
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
            // TODO: Softfloat/SSE dispatch
            emit_error_handler(&on_unreachable);
            break;

         default:
            break;
         }
      }

      void emit_function_epilogue(ir_function& func) {
         // Pop return value if function has one
         if (func.type->return_count != 0) {
            this->emit_pop_raw(rax);
         }
         // Restore frame
         if (func.num_locals > func.num_params) {
            this->emit_mov(rbp, rsp);
         }
         this->emit_pop_raw(rbp);
         if constexpr (!StackLimitIsBytes) {
            this->emit(base::INCD, ebx);
         }
         this->emit(base::RET);
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
      void mark_block_end(uint32_t block_idx) {
         if (block_idx >= _num_blocks) return;
         _block_addrs[block_idx] = code;
         // Patch all pending forward references to this block
         for (auto* f = _block_fixups[block_idx]; f; f = f->next) {
            base::fix_branch(f->branch, code);
         }
         _block_fixups[block_idx] = nullptr;
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
      template<typename F>
      void emit_i32_binop(F op) {
         this->emit_pop_raw(rcx);  // rhs
         this->emit_pop_raw(rax);  // lhs
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
         this->emit_pop_raw(rcx);  // shift amount
         this->emit_pop_raw(rax);  // value
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

      void emit_i32_relop(Jcc cc) {
         this->emit_pop_raw(rcx);  // rhs
         this->emit_pop_raw(rax);  // lhs
         this->emit_cmp(ecx, eax);
         this->emit_xor(eax, eax);
         this->emit_setcc(cc, al);
         this->emit_push_raw(rax);
      }

      void emit_i64_relop(Jcc cc) {
         this->emit_pop_raw(rcx);
         this->emit_pop_raw(rax);
         this->emit_cmp(rcx, rax);
         this->emit_xor(eax, eax);
         this->emit_setcc(cc, al);
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
   };

}} // namespace eosio::vm
