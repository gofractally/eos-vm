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
      using base::xmm0; using base::xmm1; using base::xmm2; using base::xmm3;
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

      // Offset of _remaining_call_depth in the context (frame_info_holder)
      static constexpr int32_t call_depth_offset() {
         if constexpr (Context::async_backtrace()) return 16;
         else return 0;
      }

      // Emit: decrement call depth in context memory, branch to overflow if zero.
      // Uses ecx as temp to avoid clobbering rax (needed for call_indirect target).
      void emit_call_depth_dec() {
         if constexpr (!StackLimitIsBytes) {
            auto off = call_depth_offset();
            if (off) this->emit_mov(*(rdi + off), ecx);
            else     this->emit_mov(*rdi, ecx);
            this->emit(base::DECD, ecx);
            if (off) this->emit_mov(ecx, *(rdi + off));
            else     this->emit_mov(ecx, *rdi);
            this->emit(base::TEST, ecx, ecx);
            base::fix_branch(this->emit_branchcc32(base::JZ), stack_overflow_handler);
         }
      }

      // Emit: increment call depth in context memory
      void emit_call_depth_inc() {
         if constexpr (!StackLimitIsBytes) {
            auto off = call_depth_offset();
            if (off) this->emit_mov(*(rdi + off), ecx);
            else     this->emit_mov(*rdi, ecx);
            this->emit(base::INCD, ecx);
            if (off) this->emit_mov(ecx, *(rdi + off));
            else     this->emit_mov(ecx, *rdi);
         }
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
         // SIMD ops can emit up to ~128 bytes each (softfloat calls, shuffle, etc.)
         const std::size_t bytes_per_inst = func.has_simd ? 128 : 64;
         const std::size_t est_size = static_cast<std::size_t>(func.inst_count) * bytes_per_inst + 256;
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

         // Store SSA info for constant-operand fusion in emit_binop_reg
         _func_def_inst = func.def_inst;
         _func_use_count = func.use_count;
         _func_insts = func.insts;
         _func_inst_count = func.inst_count;

         start_function(code, func.func_index + _mod.get_imported_functions_size());

         // Emit function prologue
         emit_function_prologue(func);

         // Emit each IR instruction.
         // Block boundaries are encoded as block_start/block_end pseudo-instructions
         // in the IR stream — no O(blocks) scan needed per instruction.
         for (uint32_t i = 0; i < func.inst_count; ++i) {
            if (!_use_regalloc || !emit_ir_inst_reg(func, func.insts[i], i)) {
               emit_ir_inst(func, func.insts[i], i);
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

         // Call depth counter lives in context memory (frees rbx for regalloc)
         if constexpr (Context::async_backtrace()) {
            this->emit_mov(rbp, *(rdi + 8));
         }
         this->emit_call(rcx);
         if constexpr (Context::async_backtrace()) {
            this->emit_xor(edx, edx);
            this->emit_mov(rdx, *(rdi + 8));
         }
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
         // Load call depth from context memory (rdi) into ecx for call_host_function
         if constexpr (Context::async_backtrace())
            this->emit_mov(*(rdi + 16), ecx);
         else
            this->emit_mov(*rdi, ecx);
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

         // Count body local slots, accounting for v128 locals using 2 slots (16 bytes)
         uint32_t body_local_slots = 0;
         {
            uint32_t param_count = func.type->param_types.size();
            const auto& locals = _mod.code[func.func_index].locals;
            for (uint32_t g = 0; g < locals.size(); ++g) {
               uint32_t slots_per = (locals[g].type == types::v128) ? 2 : 1;
               body_local_slots += locals[g].count * slots_per;
            }
         }
         _body_locals = body_local_slots;

         // Callee-saved register usage was computed during regalloc
         _callee_saved_used = func.callee_saved_used;
         _callee_saved_count = __builtin_popcount(_callee_saved_used);

         // Allocate and zero-initialize: body locals + spill slots + callee-saved saves
         uint32_t total_slots = body_local_slots + _num_spill_slots + _callee_saved_count;
         if (total_slots > 0) {
            this->emit_xor(eax, eax);
            for (uint32_t i = 0; i < total_slots; ++i) {
               this->emit_push_raw(rax);
            }
         }

         // Save callee-saved registers to the frame (after locals and spill slots)
         if (_use_regalloc) {
            int32_t save_offset = -static_cast<int32_t>((body_local_slots + _num_spill_slots + 1) * 8);
            if (_callee_saved_used & 1)  { this->emit_mov(rbx, *(rbp + save_offset)); save_offset -= 8; }
            if (_callee_saved_used & 2)  { this->emit_mov(r12, *(rbp + save_offset)); save_offset -= 8; }
            if (_callee_saved_used & 4)  { this->emit_mov(r13, *(rbp + save_offset)); save_offset -= 8; }
            if (_callee_saved_used & 8)  { this->emit_mov(r14, *(rbp + save_offset)); save_offset -= 8; }
            if (_callee_saved_used & 16) { this->emit_mov(r15, *(rbp + save_offset)); save_offset -= 8; }
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

         case ir_op::const_v128: {
            // Push 16-byte constant: high qword first, then low qword
            uint64_t low, high;
            memcpy(&low, &inst.immv128, 8);
            memcpy(&high, reinterpret_cast<const char*>(&inst.immv128) + 8, 8);
            this->emit_mov(high, rax);
            this->emit_push_raw(rax);
            this->emit_mov(low, rax);
            this->emit_push_raw(rax);
            break;
         }

         case ir_op::v128_op:
            emit_simd_op(inst);
            break;

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
               if (inst.type == types::v128) {
                  // v128 return: load into xmm0 for caller
                  this->emit_vmovdqu(*rsp, xmm0);
               } else {
                  this->emit_pop_raw(rax);
               }
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

         // ── Mov (phi-node merge at control flow edges) ──
         case ir_op::mov:
            // In stack mode: pop src, push to dest position (nop since it's same slot)
            // The src is on top of stack and dest replaces it — no code needed.
            break;

         case ir_op::block:
         case ir_op::loop:
            break;
         case ir_op::block_start:
            mark_block_start(inst.dest);
            break;
         case ir_op::block_end:
            mark_block_end(func, inst.dest, inst.flags & 1);
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
               // is_if clearing now happens at IR build time (emit_else in ir_writer)
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
            emit_call_depth_dec();
            // Emit call (may need relocation)
            void* branch = this->emit_call32();
            register_call(branch, funcnum);
            // Pop params, push result
            emit_call_multipop(ft);
            emit_call_depth_inc();
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
            emit_call_depth_dec();
            // call *8(%rax)
            this->emit_bytes(0xff, 0x50, 0x08);
            emit_call_multipop(ft);
            emit_call_depth_inc();
            break;
         }

         // ── Local/global access ──
         case ir_op::local_get: {
            int32_t offset = get_frame_offset(func, inst.local.index);
            if (inst.type == types::v128) {
               // Load 16-byte v128 from frame and push to x86 stack
               auto addr = *(rbp + offset);
               this->emit_vmovdqu(addr, xmm0);
               this->emit_sub(16, rsp);
               this->emit_vmovdqu(xmm0, *rsp);
            } else {
               this->emit_mov(*(rbp + offset), rax);
               this->emit_push_raw(rax);
            }
            break;
         }
         case ir_op::local_set: {
            int32_t offset = get_frame_offset(func, inst.local.index);
            if (inst.type == types::v128) {
               // Pop 16-byte v128 from x86 stack and store to frame
               this->emit_vmovdqu(*rsp, xmm0);
               this->emit_add(16, rsp);
               auto addr = *(rbp + offset);
               this->emit_vmovdqu(xmm0, addr);
            } else {
               this->emit_pop_raw(rax);
               this->emit_mov(rax, *(rbp + offset));
            }
            break;
         }
         case ir_op::local_tee: {
            int32_t offset = get_frame_offset(func, inst.local.index);
            if (inst.type == types::v128) {
               // Copy TOS v128 (16 bytes) to frame without popping
               this->emit_vmovdqu(*rsp, xmm0);
               auto addr = *(rbp + offset);
               this->emit_vmovdqu(xmm0, addr);
            } else {
               this->emit_pop_raw(rax);
               this->emit_mov(rax, *(rbp + offset));
               this->emit_push_raw(rax);
            }
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
            if (inst.type == types::v128) {
               this->emit_add(16, rsp);
            } else {
               this->emit_pop_raw(rax);
            }
            break;

         case ir_op::select:
            if (inst.type == types::v128) {
               // select for v128: condition (8), val2 (16), val1 (16) on stack
               this->emit_pop_raw(rax);  // condition
               this->emit(base::TEST, eax, eax);
               // If zero, copy val2 over val1. val2 is at rsp+0 (16 bytes), val1 at rsp+16 (16 bytes).
               void* skip = this->emit_branch8(base::JNZ);
               // condition is zero: keep val2, copy it to val1 position
               this->emit_vmovdqu(*rsp, xmm0);
               this->emit_vmovdqu(xmm0, *(rsp + 16));
               base::fix_branch8(skip, code);
               // Remove val2, keep val1 (which may have been overwritten with val2)
               this->emit_add(16, rsp);
            } else {
               this->emit_pop_raw(rax);  // condition
               this->emit_pop_raw(rcx);  // val2
               this->emit_pop_raw(rdx);  // val1
               this->emit(base::TEST, eax, eax);
               this->emit_bytes(0x48, 0x0f, 0x44, 0xd1); // cmovz %rcx, %rdx (64-bit)
               this->emit_push_raw(rdx);
            }
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

         // Bulk memory operations
         case ir_op::memory_fill:
            // stack: [dest, val, count] with count on top
            this->emit_pop_raw(rcx);  // count
            this->emit_mov(rdi, r8);  // save rdi
            this->emit_pop_raw(rax);  // value (al used by stosb)
            this->emit_pop_raw(rdi);  // dest addr
            this->emit_add(rsi, rdi); // convert to native addr
            this->emit_bytes(0xf3, 0xaa); // rep stosb
            this->emit_mov(r8, rdi);  // restore rdi
            break;

         case ir_op::memory_copy:
            // stack: [dest, src, count] with count on top
            this->emit_pop_raw(rcx);  // count
            this->emit_mov(rsi, r8);  // save rsi (linear memory base)
            this->emit_mov(rdi, r9);  // save rdi (context)
            this->emit_pop_raw(rsi);  // src addr
            this->emit_pop_raw(rdi);  // dest addr
            this->emit_add(r8, rsi);  // convert to native
            this->emit_add(r8, rdi);
            // Handle overlapping: if dest > src, copy backward
            this->emit_mov(rdi, rax);
            this->emit_sub(rsi, rax);
            {
               void* fwd = this->emit_branchcc32(base::JBE);
               this->emit_cmp(rcx, rax);
               void* no_overlap = this->emit_branchcc32(base::JAE);
               // Overlapping backward: std, rep movsb, cld
               this->emit_add(rcx, rsi);
               this->emit(base::DEC, rsi);
               this->emit_add(rcx, rdi);
               this->emit(base::DEC, rdi);
               this->emit_bytes(0xfd); // std
               this->emit_bytes(0xf3, 0xa4); // rep movsb
               this->emit_bytes(0xfc); // cld
               void* done = emit_jmp32();
               base::fix_branch(fwd, code);
               base::fix_branch(no_overlap, code);
               // Forward copy
               this->emit_bytes(0xf3, 0xa4); // rep movsb
               base::fix_branch(done, code);
            }
            this->emit_mov(r8, rsi);  // restore rsi
            this->emit_mov(r9, rdi);  // restore rdi
            break;

         case ir_op::memory_init:
         case ir_op::data_drop:
         case ir_op::table_init:
         case ir_op::elem_drop:
         case ir_op::table_copy:
            // TODO: implement bulk memory/table ops
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
               if (func.type->return_type == types::v128) {
                  // v128 return: load 16 bytes into xmm0
                  this->emit_vmovdqu(*rsp, xmm0);
               } else {
                  this->emit_pop_raw(rax);
               }
            }
         }
         // Restore callee-saved registers from frame
         if (_use_regalloc && _callee_saved_used) {
            int32_t save_offset = -static_cast<int32_t>((_body_locals + _num_spill_slots + 1) * 8);
            if (_callee_saved_used & 1)  { this->emit_mov(*(rbp + save_offset), rbx); save_offset -= 8; }
            if (_callee_saved_used & 2)  { this->emit_mov(*(rbp + save_offset), r12); save_offset -= 8; }
            if (_callee_saved_used & 4)  { this->emit_mov(*(rbp + save_offset), r13); save_offset -= 8; }
            if (_callee_saved_used & 8)  { this->emit_mov(*(rbp + save_offset), r14); save_offset -= 8; }
            if (_callee_saved_used & 16) { this->emit_mov(*(rbp + save_offset), r15); save_offset -= 8; }
         }
         // Restore frame
         this->emit_mov(rbp, rsp);
         this->emit_pop_raw(rbp);
         this->emit(base::RET);
      }

      // ──────── Instruction fusion helpers ────────
      // Check if the next instruction is if_/br_if consuming dest_vreg.
      // If so, fuse the comparison with the branch: emit cmp+jcc directly,
      // mark the next instruction as dead, and return the fused branch info.
      // Called when a comparison has IR_FUSE_NEXT set.
      // The next instruction (if_/br_if) is already IR_DEAD.
      // Emit the branch directly from the flags set by the comparison.
      bool emit_fused_branch(ir_function& func, uint32_t idx, Jcc cc) {
         auto& next = func.insts[idx + 1];
         if (next.opcode == ir_op::if_) {
            // if_ branches to else when condition is FALSE → invert cc
            void* branch = this->emit_branchcc32(invert_cc(cc));
            push_if_fixup(branch);
            return true;
         }
         if (next.opcode == ir_op::br_if) {
            // br_if branches when condition is TRUE → use cc directly
            // Note: br_if with depth_change > 0 can't be fused (needs multipop)
            // but the optimizer already checked use_count == 1 and adjacency
            emit_branch_cc_to_block(func, next.br.target, next.dest, next.type, cc);
            return true;
         }
         return false;
      }

      // Invert a condition code (for if_ which branches on FALSE)
      static Jcc invert_cc(Jcc cc) {
         // x86 Jcc opcodes: even = false condition, odd = true condition.
         // XOR with 1 inverts (JE↔JNE, JL↔JGE, JB↔JAE, etc.)
         return Jcc{static_cast<uint8_t>(cc.opcode ^ 1)};
      }

      // Emit a conditional branch to a block (for fused cmp+br_if)
      void emit_branch_cc_to_block(ir_function& func, uint32_t block_idx,
                                    uint32_t depth_change, uint8_t rt, Jcc cc) {
         if (block_idx >= _num_blocks) return;
         // For fused br_if, no multipop needed (handled by the IR's depth_change)
         if (_block_addrs[block_idx] != nullptr) {
            // Backward branch (loop)
            void* branch = this->emit_branchcc32(cc);
            base::fix_branch(branch, _block_addrs[block_idx]);
         } else {
            // Forward branch
            void* branch = this->emit_branchcc32(cc);
            auto* fixup = _allocator.alloc<block_fixup>(1);
            fixup->branch = branch;
            fixup->next = _block_fixups[block_idx];
            _block_fixups[block_idx] = fixup;
         }
      }

      // Check if src2 is a compile-time constant. If so, load src1 into rax,
      // apply op with the immediate, store result. Returns true if handled.
      // Emit binop with constant src2: load src1 → rax, apply op with immediate, store.
      // Lambda receives int32_t immediate value.
      template<typename F>
      bool emit_binop_imm(const ir_inst& inst, F op_imm, bool is32) {
         if (!_func_def_inst || inst.rr.src2 >= _num_vregs) return false;
         uint32_t def = _func_def_inst[inst.rr.src2];
         if (def >= _func_inst_count) return false;
         auto& di = _func_insts[def];
         if (di.opcode != ir_op::const_i32 && di.opcode != ir_op::const_i64) return false;
         int32_t imm = static_cast<int32_t>(di.imm64);
         // For i64 operations, the imm32 is sign-extended to 64 bits by the CPU.
         // Reject if the sign-extended value doesn't match the original i64 constant.
         if (!is32 && static_cast<int64_t>(imm) != static_cast<int64_t>(di.imm64))
            return false;
         load_vreg_rax(inst.rr.src1);
         op_imm(imm);
         store_rax_vreg(inst.dest);
         if (_func_use_count && _func_use_count[inst.rr.src2] == 1)
            di.flags |= IR_DEAD;
         return true;
      }

      // ──────── Register-based IR emission ────────
      // Uses physical registers for vreg values instead of push/pop.
      // rax and rcx are temporaries. Vregs in physical registers are
      // accessed directly; spilled vregs use fixed rbp-offset slots.
      //
      // Returns true if handled, false to fall back to stack-based emission.
      bool emit_ir_inst_reg(ir_function& func, const ir_inst& inst, uint32_t idx) {
         if (inst.flags & IR_DEAD) return true;  // skip dead instructions
         switch (inst.opcode) {
         case ir_op::nop:
         case ir_op::block:
         case ir_op::loop:
         case ir_op::drop:
            return true;
         case ir_op::block_start:
            mark_block_start(inst.dest);
            return true;
         case ir_op::block_end:
            mark_block_end(func, inst.dest, inst.flags & 1);
            return true;

         // Mov (phi-node merge): dest = src1
         case ir_op::mov: {
            int8_t pr_dest = get_phys(inst.dest);
            int8_t pr_src  = get_phys(inst.rr.src1);
            if (pr_dest >= 0 && pr_src >= 0) {
               if (pr_dest != pr_src) {
                  this->emit_mov(phys_to_reg64(pr_src), phys_to_reg64(pr_dest));
               }
            } else if (pr_dest >= 0) {
               // src is spilled, dest is in register
               load_vreg_rax(inst.rr.src1);
               this->emit_mov(rax, phys_to_reg64(pr_dest));
            } else if (pr_src >= 0) {
               // src is in register, dest is spilled
               this->emit_mov(phys_to_reg64(pr_src), rax);
               store_rax_vreg(inst.dest);
            } else {
               // Both spilled
               load_vreg_rax(inst.rr.src1);
               store_rax_vreg(inst.dest);
            }
            return true;
         }

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
               // is_if clearing now happens at IR build time (emit_else in ir_writer)
            }
            pop_if_fixup_to(code);
            return true;
         }
         case ir_op::br: {
            // Result value transfer is handled by mov instructions emitted by ir_writer
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
            emit_call_depth_dec();
            void* branch = emit_call32();
            register_call(branch, funcnum);
            // Args were pushed by arg instructions — pop them
            uint32_t arg_bytes = 0;
            for (uint32_t p = 0; p < ft.param_types.size(); ++p)
               arg_bytes += (ft.param_types[p] == types::v128) ? 16 : 8;
            if (arg_bytes > 0)
               this->emit_add(arg_bytes, rsp);
            emit_call_depth_inc();
            // Store result to dest vreg (result already in rax)
            if (ft.return_count > 0 && inst.dest != ir_vreg_none) {
               store_rax_vreg(inst.dest);
            }
            return true;
         }
         case ir_op::call_indirect: {
            uint32_t fti = inst.call.index;
            const func_type& ft = _mod.types[fti];
            // Table index was pushed by an arg instruction — pop it
            this->emit_pop_raw(rax);
            // Bounds check
            uint32_t table_size = _mod.tables[0].limits.initial;
            this->emit_cmp(table_size, eax);
            base::fix_branch(this->emit_branchcc32(base::JAE), call_indirect_handler);
            // Table lookup
            this->emit_bytes(0x48, 0xc1, 0xe0, 0x04); // shlq $4, %rax
            if (_mod.indirect_table(0)) {
               this->emit_mov(*(rsi + wasm_allocator::table_offset()), rcx);
               this->emit_add(rcx, rax);
            } else {
               this->emit_bytes(0x48, 0x8d, 0x84, 0x06);
               this->emit_operand32(static_cast<uint32_t>(wasm_allocator::table_offset()));
            }
            this->emit_bytes(0x81, 0x38);
            this->emit_operand32(fti);
            base::fix_branch(this->emit_branchcc32(base::JNE), type_error_handler);
            emit_call_depth_dec();
            this->emit_bytes(0xff, 0x50, 0x08); // call *8(%rax)
            // Remove args, store result
            uint32_t arg_bytes = 0;
            for (uint32_t p = 0; p < ft.param_types.size(); ++p)
               arg_bytes += (ft.param_types[p] == types::v128) ? 16 : 8;
            if (arg_bytes > 0) this->emit_add(arg_bytes, rsp);
            emit_call_depth_inc();
            if (ft.return_count > 0 && inst.dest != ir_vreg_none) {
               store_rax_vreg(inst.dest);
            }
            return true;
         }

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

         // Memory management — handled in float/conversion section below

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

         // Integer binary ops (with const-immediate for add/sub)
         case ir_op::i32_add:
            if (emit_binop_imm(inst, [this](int32_t imm){ this->emit_add(imm, eax); }, true)) return true;
            return emit_binop_reg(inst, [this](auto d, auto s){ this->emit_add(s, d); }, true);
         case ir_op::i32_sub:
            if (emit_binop_imm(inst, [this](int32_t imm){ this->emit_sub(imm, eax); }, true)) return true;
            return emit_binop_reg(inst, [this](auto d, auto s){ this->emit_sub(s, d); }, true);
         case ir_op::i32_mul: return emit_binop_reg(inst, [this](auto d, auto s){ this->emit(base::IMUL, s, d); }, true);
         case ir_op::i32_and:
            if (emit_binop_imm(inst, [this](int32_t imm){ this->emit_and(imm, eax); }, true)) return true;
            return emit_binop_reg(inst, [this](auto d, auto s){ this->emit(base::AND_A, s, d); }, true);
         case ir_op::i32_or:
            if (emit_binop_imm(inst, [this](int32_t imm){ this->emit_or(imm, eax); }, true)) return true;
            return emit_binop_reg(inst, [this](auto d, auto s){ this->emit(base::OR_A, s, d); }, true);
         case ir_op::i32_xor:
            if (emit_binop_imm(inst, [this](int32_t imm){ this->emit_xor(imm, eax); }, true)) return true;
            return emit_binop_reg(inst, [this](auto d, auto s){ this->emit(base::XOR_A, s, d); }, true);

         case ir_op::i64_add:
            if (emit_binop_imm(inst, [this](int32_t imm){ this->emit_add(imm, rax); }, false)) return true;
            return emit_binop_reg(inst, [this](auto d, auto s){ this->emit_add(s, d); }, false);
         case ir_op::i64_sub:
            if (emit_binop_imm(inst, [this](int32_t imm){ this->emit_sub(imm, rax); }, false)) return true;
            return emit_binop_reg(inst, [this](auto d, auto s){ this->emit_sub(s, d); }, false);
         case ir_op::i64_mul: return emit_binop_reg(inst, [this](auto d, auto s){ this->emit(base::IMUL, s, d); }, false);
         case ir_op::i64_and:
            if (emit_binop_imm(inst, [this](int32_t imm){ this->emit_and(imm, rax); }, false)) return true;
            return emit_binop_reg(inst, [this](auto d, auto s){ this->emit(base::AND_A, s, d); }, false);
         case ir_op::i64_or:
            if (emit_binop_imm(inst, [this](int32_t imm){ this->emit_or(imm, rax); }, false)) return true;
            return emit_binop_reg(inst, [this](auto d, auto s){ this->emit(base::OR_A, s, d); }, false);
         case ir_op::i64_xor:
            if (emit_binop_imm(inst, [this](int32_t imm){ this->emit_xor(imm, rax); }, false)) return true;
            return emit_binop_reg(inst, [this](auto d, auto s){ this->emit(base::XOR_A, s, d); }, false);

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
            int8_t pr_s = get_phys(inst.rr.src1);
            if (pr_s >= 0)
               this->emit(base::TEST, phys_to_reg32(pr_s), phys_to_reg32(pr_s));
            else {
               load_vreg_rax(inst.rr.src1);
               this->emit(base::TEST, eax, eax);
            }
            if ((inst.flags & IR_FUSE_NEXT) && emit_fused_branch(func, idx, base::JZ)) return true;
            this->emit_setcc(base::JZ, al);
            this->emit_bytes(0x0f, 0xb6, 0xc0); // movzbl
            int8_t pr_d = get_phys(inst.dest);
            if (pr_d >= 0 && phys_to_reg64(pr_d) != rax)
               this->emit_mov(rax, phys_to_reg64(pr_d));
            else
               store_rax_vreg(inst.dest);
            return true;
         }

         // Comparisons
         case ir_op::i32_eq: return emit_relop_reg(func, inst, idx, base::JE, true);
         case ir_op::i32_ne: return emit_relop_reg(func, inst, idx, base::JNE, true);
         case ir_op::i32_lt_s: return emit_relop_reg(func, inst, idx, base::JL, true);
         case ir_op::i32_lt_u: return emit_relop_reg(func, inst, idx, base::JB, true);
         case ir_op::i32_gt_s: return emit_relop_reg(func, inst, idx, base::JG, true);
         case ir_op::i32_gt_u: return emit_relop_reg(func, inst, idx, base::JA, true);
         case ir_op::i32_le_s: return emit_relop_reg(func, inst, idx, base::JLE, true);
         case ir_op::i32_le_u: return emit_relop_reg(func, inst, idx, base::JBE, true);
         case ir_op::i32_ge_s: return emit_relop_reg(func, inst, idx, base::JGE, true);
         case ir_op::i32_ge_u: return emit_relop_reg(func, inst, idx, base::JAE, true);

         // Local access
         case ir_op::local_get: {
            int32_t offset = get_frame_offset(func, inst.local.index);
            int8_t pr = get_phys(inst.dest);
            if (pr >= 0) {
               this->emit_mov(*(rbp + offset), phys_to_reg64(pr));
            } else {
               this->emit_mov(*(rbp + offset), rax);
               store_rax_vreg(inst.dest);
            }
            return true;
         }
         case ir_op::local_set: {
            int32_t offset = get_frame_offset(func, inst.local.index);
            int8_t pr = get_phys(inst.local.src1);
            if (pr >= 0) {
               this->emit_mov(phys_to_reg64(pr), *(rbp + offset));
            } else {
               load_vreg_rax(inst.local.src1);
               this->emit_mov(rax, *(rbp + offset));
            }
            return true;
         }
         case ir_op::local_tee: {
            int32_t offset = get_frame_offset(func, inst.local.index);
            load_vreg_rax(inst.local.src1);
            this->emit_mov(rax, *(rbp + offset));
            // Value stays in src register (tee doesn't consume)
            return true;
         }

         // Memory loads
         case ir_op::i32_load: return emit_load_reg(inst, base::MOV_A, eax);
         case ir_op::i64_load: return emit_load_reg(inst, base::MOV_A, rax);
         case ir_op::i32_load8_u: return emit_load_reg(inst, base::MOVZXB, eax);
         case ir_op::i32_load16_u: return emit_load_reg(inst, base::MOVZXW, eax);
         case ir_op::i32_load8_s: return emit_load_reg(inst, base::MOVSXB, eax);
         case ir_op::i32_load16_s: return emit_load_reg(inst, base::MOVSXW, eax);
         // Memory stores
         case ir_op::i32_store: return emit_store_reg(inst, base::MOV_B, eax);
         case ir_op::i64_store: return emit_store_reg(inst, base::MOV_B, rax);
         case ir_op::i32_store8: return emit_store_reg(inst, base::MOVB_B, al);
         case ir_op::i32_store16: return emit_store_reg(inst, base::MOVW_B, ax);

         // Return
         case ir_op::return_: {
            if (inst.rr.src1 != ir_vreg_none) {
               load_vreg_rax(inst.rr.src1);
            }
            // Restore callee-saved registers before returning
            if (_callee_saved_used) {
               int32_t save_offset = -static_cast<int32_t>((_body_locals + _num_spill_slots + 1) * 8);
               if (_callee_saved_used & 1)  { this->emit_mov(*(rbp + save_offset), rbx); save_offset -= 8; }
               if (_callee_saved_used & 2)  { this->emit_mov(*(rbp + save_offset), r12); save_offset -= 8; }
               if (_callee_saved_used & 4)  { this->emit_mov(*(rbp + save_offset), r13); save_offset -= 8; }
               if (_callee_saved_used & 8)  { this->emit_mov(*(rbp + save_offset), r14); save_offset -= 8; }
               if (_callee_saved_used & 16) { this->emit_mov(*(rbp + save_offset), r15); save_offset -= 8; }
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
         case ir_op::i64_eqz: {
            int8_t pr_s = get_phys(inst.rr.src1);
            if (pr_s >= 0)
               this->emit(base::TEST, phys_to_reg64(pr_s), phys_to_reg64(pr_s));
            else {
               load_vreg_rax(inst.rr.src1);
               this->emit(base::TEST, rax, rax);
            }
            if ((inst.flags & IR_FUSE_NEXT) && emit_fused_branch(func, idx, base::JZ)) return true;
            this->emit_setcc(base::JZ, al);
            this->emit_bytes(0x0f, 0xb6, 0xc0);
            int8_t pr_d = get_phys(inst.dest);
            if (pr_d >= 0 && phys_to_reg64(pr_d) != rax)
               this->emit_mov(rax, phys_to_reg64(pr_d));
            else
               store_rax_vreg(inst.dest);
            return true;
         }
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
         case ir_op::i64_eq: return emit_relop_reg(func, inst, idx, base::JE, false);
         case ir_op::i64_ne: return emit_relop_reg(func, inst, idx, base::JNE, false);
         case ir_op::i64_lt_s: return emit_relop_reg(func, inst, idx, base::JL, false);
         case ir_op::i64_lt_u: return emit_relop_reg(func, inst, idx, base::JB, false);
         case ir_op::i64_gt_s: return emit_relop_reg(func, inst, idx, base::JG, false);
         case ir_op::i64_gt_u: return emit_relop_reg(func, inst, idx, base::JA, false);
         case ir_op::i64_le_s: return emit_relop_reg(func, inst, idx, base::JLE, false);
         case ir_op::i64_le_u: return emit_relop_reg(func, inst, idx, base::JBE, false);
         case ir_op::i64_ge_s: return emit_relop_reg(func, inst, idx, base::JGE, false);
         case ir_op::i64_ge_u: return emit_relop_reg(func, inst, idx, base::JAE, false);

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

         // Select: dest = cond ? val1 : val2  (3 source vregs packed in sel union)
         case ir_op::select: {
            load_vreg_rax(inst.sel.cond);  // condition
            this->emit_mov(rax, rdx);
            load_vreg_rcx(inst.sel.val2);  // val2
            load_vreg_rax(inst.sel.val1);  // val1
            this->emit(base::TEST, edx, edx);
            this->emit_bytes(0x48, 0x0f, 0x44, 0xc1); // cmovz rcx, rax
            store_rax_vreg(inst.dest);
            return true;
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

         // ── Bulk memory ops (args already pushed by arg instructions) ──
         case ir_op::memory_fill: {
            // Args on stack: [dest][val][count] with count on top
            // Use stack-based save for rdi instead of r8 (r8 may hold a vreg)
            this->emit_pop_raw(rcx);  // count
            this->emit_pop_raw(rax);  // value (al used by stosb)
            this->emit_pop_raw(rdx);  // dest addr
            this->emit_push_raw(rdi); // save rdi on stack
            this->emit_mov(edx, edi); // dest addr to edi
            this->emit_add(rsi, rdi); // convert to native addr
            this->emit_bytes(0xf3, 0xaa); // rep stosb
            this->emit_pop_raw(rdi);  // restore rdi
            return true;
         }
         case ir_op::memory_copy: {
            // Args on stack: [dest][src][count] with count on top
            this->emit_pop_raw(rcx);  // count
            this->emit_pop_raw(rax);  // src addr
            this->emit_pop_raw(rdx);  // dest addr
            // Save rsi, rdi on stack (don't use r8/r9 — may hold vregs)
            this->emit_push_raw(rsi);
            this->emit_push_raw(rdi);
            this->emit_mov(*(rsp + 8), rsi); // recover saved rsi (linear memory base)
            this->emit_add(rsi, rax); // src = rsi + src_addr (rsi = linear memory base)
            this->emit_add(rsi, rdx); // dest = rsi + dest_addr
            this->emit_mov(rax, rsi); // native src addr
            this->emit_mov(rdx, rdi); // native dest addr
            // Handle overlapping
            this->emit_mov(rdi, rax);
            this->emit_sub(rsi, rax); // rax = dest - src
            {
               void* fwd = this->emit_branchcc32(base::JBE);
               this->emit_cmp(rcx, rax);
               void* no_overlap = this->emit_branchcc32(base::JAE);
               // Overlapping backward
               this->emit_add(rcx, rsi);
               this->emit(base::DEC, rsi);
               this->emit_add(rcx, rdi);
               this->emit(base::DEC, rdi);
               this->emit_bytes(0xfd); // std
               this->emit_bytes(0xf3, 0xa4); // rep movsb
               this->emit_bytes(0xfc); // cld
               void* done = emit_jmp32();
               base::fix_branch(fwd, code);
               base::fix_branch(no_overlap, code);
               this->emit_bytes(0xf3, 0xa4); // rep movsb
               base::fix_branch(done, code);
            }
            this->emit_pop_raw(rdi);  // restore rdi
            this->emit_pop_raw(rsi);  // restore rsi
            return true;
         }
         case ir_op::memory_init:
         case ir_op::data_drop:
         case ir_op::table_init:
         case ir_op::elem_drop:
         case ir_op::table_copy:
            // Not yet implemented — args on stack, just discard
            return true;

         // ── Memory management (register mode) ──
         case ir_op::memory_size:
            this->emit_push_raw(rdi);
            this->emit_push_raw(rsi);
            this->emit_bytes(0x48, 0xb8);
            this->emit_operand_ptr(&current_memory);
            this->emit(base::CALL, rax);
            this->emit_pop_raw(rsi);
            this->emit_pop_raw(rdi);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::memory_grow:
            load_vreg_rax(inst.rr.src1);
            this->emit_push_raw(rdi);
            this->emit_push_raw(rsi);
            this->emit_mov(eax, esi); // pages arg in esi
            this->emit_bytes(0x48, 0xb8);
            this->emit_operand_ptr(&grow_memory);
            this->emit(base::CALL, rax);
            this->emit_pop_raw(rsi);
            this->emit_pop_raw(rdi);
            store_rax_vreg(inst.dest);
            return true;

         // ── Float unary ops (register mode) ──
         case ir_op::f32_abs:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x25);                       // and $0x7fffffff, eax
            this->emit_operand32(0x7fffffff);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f32_neg:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x35);                       // xor $0x80000000, eax
            this->emit_operand32(0x80000000);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f32_ceil:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);  // movq rax, xmm0
            this->emit_bytes(0x66, 0x0f, 0x3a, 0x0a, 0xc0, 0x0a); // roundss $0x0a, xmm0, xmm0
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);  // movq xmm0, rax
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f32_floor:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0x66, 0x0f, 0x3a, 0x0a, 0xc0, 0x09); // roundss $0x09
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f32_trunc:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0x66, 0x0f, 0x3a, 0x0a, 0xc0, 0x0b); // roundss $0x0b
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f32_nearest:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0x66, 0x0f, 0x3a, 0x0a, 0xc0, 0x08); // roundss $0x08
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f32_sqrt:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0xf3, 0x0f, 0x51, 0xc0);             // sqrtss xmm0, xmm0
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f64_abs:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x48, 0x0f, 0xba, 0xf0, 0x3f);      // btr $63, rax
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f64_neg:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x48, 0x0f, 0xba, 0xf8, 0x3f);      // btc $63, rax
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f64_ceil:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0x66, 0x0f, 0x3a, 0x0b, 0xc0, 0x0a); // roundsd $0x0a
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f64_floor:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0x66, 0x0f, 0x3a, 0x0b, 0xc0, 0x09); // roundsd $0x09
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f64_trunc:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0x66, 0x0f, 0x3a, 0x0b, 0xc0, 0x0b); // roundsd $0x0b
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f64_nearest:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0x66, 0x0f, 0x3a, 0x0b, 0xc0, 0x08); // roundsd $0x08
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f64_sqrt:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0xf2, 0x0f, 0x51, 0xc0);             // sqrtsd xmm0, xmm0
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);
            store_rax_vreg(inst.dest);
            return true;

         // ── Float binary ops (register mode) ──
         case ir_op::f32_add:      emit_f32_binop_reg(inst, 0x58); return true;
         case ir_op::f32_sub:      emit_f32_binop_reg(inst, 0x5c); return true;
         case ir_op::f32_mul:      emit_f32_binop_reg(inst, 0x59); return true;
         case ir_op::f32_div:      emit_f32_binop_reg(inst, 0x5e); return true;
         case ir_op::f32_min:      emit_f32_binop_reg(inst, 0x5d); return true;
         case ir_op::f32_max:      emit_f32_binop_reg(inst, 0x5f); return true;
         case ir_op::f64_add:      emit_f64_binop_reg(inst, 0x58); return true;
         case ir_op::f64_sub:      emit_f64_binop_reg(inst, 0x5c); return true;
         case ir_op::f64_mul:      emit_f64_binop_reg(inst, 0x59); return true;
         case ir_op::f64_div:      emit_f64_binop_reg(inst, 0x5e); return true;
         case ir_op::f64_min:      emit_f64_binop_reg(inst, 0x5d); return true;
         case ir_op::f64_max:      emit_f64_binop_reg(inst, 0x5f); return true;

         // ── Float copysign (register mode) ──
         case ir_op::f32_copysign:
            load_vreg_rax(inst.rr.src2);  // sign source
            this->emit_mov(eax, ecx);
            this->emit_bytes(0x81, 0xe1); // and $0x80000000, ecx
            this->emit_operand32(0x80000000);
            load_vreg_rax(inst.rr.src1);  // magnitude source
            this->emit_bytes(0x25);       // and $0x7fffffff, eax
            this->emit_operand32(0x7fffffff);
            this->emit(base::OR_A, ecx, eax);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f64_copysign:
            load_vreg_rax(inst.rr.src2);  // sign source
            this->emit_mov(rax, rcx);
            // and sign bit: mov $0x8000000000000000, rdx; and rdx, rcx
            this->emit_bytes(0x48, 0xba); // mov imm64, rdx
            this->emit_operand64(0x8000000000000000ULL);
            this->emit(base::AND_A, rdx, rcx);
            load_vreg_rax(inst.rr.src1);  // magnitude source
            // and magnitude: mov $0x7fffffffffffffff, rdx; and rdx, rax
            this->emit_bytes(0x48, 0xba);
            this->emit_operand64(0x7fffffffffffffffULL);
            this->emit(base::AND_A, rdx, rax);
            this->emit(base::OR_A, rcx, rax);
            store_rax_vreg(inst.dest);
            return true;

         // ── Float comparisons (register mode) ──
         // eq: cmpss $0, ne: cmpeqss + inc, lt: cmpss $1, gt: swap+cmpss $1, le: cmpss $2, ge: swap+cmpss $2
         case ir_op::f32_eq: emit_f32_relop_reg(inst, 0x00, false, false); return true;
         case ir_op::f32_ne: emit_f32_relop_reg(inst, 0x00, false, true);  return true;  // eq then inc
         case ir_op::f32_lt: emit_f32_relop_reg(inst, 0x01, false, false); return true;
         case ir_op::f32_gt: emit_f32_relop_reg(inst, 0x01, true,  false); return true;  // swap, use lt
         case ir_op::f32_le: emit_f32_relop_reg(inst, 0x02, false, false); return true;
         case ir_op::f32_ge: emit_f32_relop_reg(inst, 0x02, true,  false); return true;  // swap, use le
         case ir_op::f64_eq: emit_f64_relop_reg(inst, 0x00, false, false); return true;
         case ir_op::f64_ne: emit_f64_relop_reg(inst, 0x00, false, true);  return true;
         case ir_op::f64_lt: emit_f64_relop_reg(inst, 0x01, false, false); return true;
         case ir_op::f64_gt: emit_f64_relop_reg(inst, 0x01, true,  false); return true;
         case ir_op::f64_le: emit_f64_relop_reg(inst, 0x02, false, false); return true;
         case ir_op::f64_ge: emit_f64_relop_reg(inst, 0x02, true,  false); return true;

         // ── Float-to-int conversions (register mode) ──
         case ir_op::i32_trunc_s_f32:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);  // movq rax, xmm0
            this->emit_bytes(0xf3, 0x0f, 0x2c, 0xc0);         // cvttss2si eax, xmm0
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i32_trunc_u_f32:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2c, 0xc0);  // cvttss2si rax, xmm0 (64-bit to handle unsigned range)
            this->emit_bytes(0x89, 0xc0);                      // mov eax, eax (zero-extend)
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i32_trunc_s_f64:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0xf2, 0x0f, 0x2c, 0xc0);         // cvttsd2si eax, xmm0
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i32_trunc_u_f64:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2c, 0xc0);  // cvttsd2si rax, xmm0
            this->emit_bytes(0x89, 0xc0);                      // mov eax, eax
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_trunc_s_f32:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2c, 0xc0);  // cvttss2si rax, xmm0
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_trunc_u_f32:
            // For unsigned i64, handle values >= 2^63 specially
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);  // movq rax, xmm0
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2c, 0xc0);  // cvttss2si rax, xmm0
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_trunc_s_f64:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2c, 0xc0);  // cvttsd2si rax, xmm0
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_trunc_u_f64:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2c, 0xc0);  // cvttsd2si rax, xmm0
            store_rax_vreg(inst.dest);
            return true;

         // ── Saturating truncations (register mode) ──
         case ir_op::i32_trunc_sat_f32_s:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0xf3, 0x0f, 0x2c, 0xc0);         // cvttss2si eax, xmm0
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i32_trunc_sat_f32_u:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2c, 0xc0);  // cvttss2si rax, xmm0
            this->emit_bytes(0x89, 0xc0);                      // mov eax, eax
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i32_trunc_sat_f64_s:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0xf2, 0x0f, 0x2c, 0xc0);         // cvttsd2si eax, xmm0
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i32_trunc_sat_f64_u:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2c, 0xc0);  // cvttsd2si rax, xmm0
            this->emit_bytes(0x89, 0xc0);                      // mov eax, eax
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_trunc_sat_f32_s:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2c, 0xc0);  // cvttss2si rax, xmm0
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_trunc_sat_f32_u:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2c, 0xc0);  // cvttss2si rax, xmm0
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_trunc_sat_f64_s:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2c, 0xc0);  // cvttsd2si rax, xmm0
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_trunc_sat_f64_u:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2c, 0xc0);  // cvttsd2si rax, xmm0
            store_rax_vreg(inst.dest);
            return true;

         // ── Int-to-float conversions (register mode) ──
         case ir_op::f32_convert_s_i32:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0xf3, 0x0f, 0x2a, 0xc0);         // cvtsi2ss eax, xmm0
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);  // movq xmm0, rax
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f32_convert_u_i32:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x89, 0xc0);                      // mov eax, eax (zero-extend to 64-bit)
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2a, 0xc0);  // cvtsi2ss rax, xmm0
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f32_convert_s_i64:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2a, 0xc0);  // cvtsi2ss rax, xmm0
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f32_convert_u_i64:
            // For unsigned i64 → f32, use signed i64 convert (works for values < 2^63)
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0xf3, 0x48, 0x0f, 0x2a, 0xc0);  // cvtsi2ss rax, xmm0
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f64_convert_s_i32:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0xf2, 0x0f, 0x2a, 0xc0);         // cvtsi2sd eax, xmm0
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f64_convert_u_i32:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x89, 0xc0);                      // mov eax, eax (zero-extend)
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2a, 0xc0);  // cvtsi2sd rax, xmm0
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f64_convert_s_i64:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2a, 0xc0);  // cvtsi2sd rax, xmm0
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f64_convert_u_i64:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0xf2, 0x48, 0x0f, 0x2a, 0xc0);  // cvtsi2sd rax, xmm0
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);
            store_rax_vreg(inst.dest);
            return true;

         // ── Float-float conversions (register mode) ──
         case ir_op::f32_demote_f64:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);  // movq rax, xmm0
            this->emit_bytes(0xf2, 0x0f, 0x5a, 0xc0);         // cvtsd2ss xmm0, xmm0
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);  // movq xmm0, rax
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::f64_promote_f32:
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);  // movq rax, xmm0
            this->emit_bytes(0xf3, 0x0f, 0x5a, 0xc0);         // cvtss2sd xmm0, xmm0
            this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);  // movq xmm0, rax
            store_rax_vreg(inst.dest);
            return true;

         // ── i64 div/rem (register mode) ──
         case ir_op::i64_div_s:
            load_vreg_rcx(inst.rr.src2);
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x48, 0x99);        // cqo
            this->emit_bytes(0x48, 0xf7, 0xf9);  // idiv rcx
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_div_u:
            load_vreg_rcx(inst.rr.src2);
            load_vreg_rax(inst.rr.src1);
            this->emit_xor(edx, edx);
            this->emit_bytes(0x48, 0xf7, 0xf1);  // div rcx
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_rem_s:
            load_vreg_rcx(inst.rr.src2);
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x48, 0x83, 0xf9, 0xff); // cmp $-1, rcx
            { void* skip = this->emit_branch8(base::JE);
              this->emit_bytes(0x48, 0x99);        // cqo
              this->emit_bytes(0x48, 0xf7, 0xf9);  // idiv rcx
              void* done = this->emit_branch8(base::JMP_8);
              base::fix_branch8(skip, code);
              this->emit_xor(edx, edx);
              base::fix_branch8(done, code); }
            this->emit_bytes(0x48, 0x89, 0xd0); // mov rdx, rax
            store_rax_vreg(inst.dest);
            return true;
         case ir_op::i64_rem_u:
            load_vreg_rcx(inst.rr.src2);
            load_vreg_rax(inst.rr.src1);
            this->emit_xor(edx, edx);
            this->emit_bytes(0x48, 0xf7, 0xf1);  // div rcx
            this->emit_bytes(0x48, 0x89, 0xd0);  // mov rdx, rax
            store_rax_vreg(inst.dest);
            return true;

         default:
            // Unhandled in register mode — bridge to stack mode:
            // push source vregs, run stack-mode handler (uses push/pop), store result.
            // Stack-mode handlers only clobber rax/rcx/rdx/rsi/rdi (not r8-r15).
            {
               bool is_store = (inst.opcode >= ir_op::i32_store && inst.opcode <= ir_op::i64_store32);
               bool is_binary = (inst.opcode >= ir_op::i32_add && inst.opcode <= ir_op::f64_copysign);
               bool is_unary = (inst.opcode >= ir_op::i32_eqz && inst.opcode <= ir_op::i32_popcnt)
                            || (inst.opcode >= ir_op::i64_eqz && inst.opcode <= ir_op::i64_popcnt)
                            || (inst.opcode >= ir_op::i32_wrap_i64 && inst.opcode <= ir_op::i64_trunc_sat_f64_u)
                            || (inst.opcode >= ir_op::i32_extend8_s && inst.opcode <= ir_op::i64_extend32_s)
                            || (inst.opcode >= ir_op::f32_abs && inst.opcode <= ir_op::f64_sqrt);
               bool is_load = (inst.opcode >= ir_op::i32_load && inst.opcode <= ir_op::i64_load32_u);

               if (is_binary) {
                  load_vreg_rax(inst.rr.src1);
                  this->emit_push_raw(rax);
                  load_vreg_rax(inst.rr.src2);
                  this->emit_push_raw(rax);
               } else if (is_unary) {
                  load_vreg_rax(inst.rr.src1);
                  this->emit_push_raw(rax);
               } else if (is_load) {
                  load_vreg_rax(inst.ri.src1);
                  this->emit_push_raw(rax);
               } else if (is_store) {
                  load_vreg_rax(inst.ri.src1);
                  this->emit_push_raw(rax);
                  load_vreg_rax(inst.dest);
                  this->emit_push_raw(rax);
               }
               emit_ir_inst(func, inst, idx);
               if (!is_store && inst.dest != ir_vreg_none) {
                  this->emit_pop_raw(rax);
                  store_rax_vreg(inst.dest);
               }
               return true;
            }
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
               if (is32) op(phys_to_reg32(pr_d), phys_to_reg32(pr_s2));
               else      op(phys_to_reg64(pr_d), phys_to_reg64(pr_s2));
            } else if (pr_d == pr_s2) {
               if (is32) { this->emit_mov(phys_to_reg32(pr_s1), eax); op(eax, phys_to_reg32(pr_s2)); this->emit_mov(eax, phys_to_reg32(pr_d)); }
               else      { this->emit_mov(phys_to_reg64(pr_s1), rax); op(rax, phys_to_reg64(pr_s2)); this->emit_mov(rax, phys_to_reg64(pr_d)); }
            } else {
               if (is32) { this->emit_mov(phys_to_reg32(pr_s1), phys_to_reg32(pr_d)); op(phys_to_reg32(pr_d), phys_to_reg32(pr_s2)); }
               else      { this->emit_mov(phys_to_reg64(pr_s1), phys_to_reg64(pr_d)); op(phys_to_reg64(pr_d), phys_to_reg64(pr_s2)); }
            }
         } else {
            load_vreg_rcx(inst.rr.src2);
            load_vreg_rax(inst.rr.src1);
            if (is32) op(eax, ecx);
            else      op(rax, rcx);
            store_rax_vreg(inst.dest);
         }
         return true;
      }

      // Register-based comparison helper
      bool emit_relop_reg(ir_function& func, const ir_inst& inst, uint32_t idx, Jcc cc, bool is32) {
         int8_t pr_s1 = get_phys(inst.rr.src1);
         int8_t pr_s2 = get_phys(inst.rr.src2);

         // Fast path: both in physical registers — compare directly
         if (pr_s1 >= 0 && pr_s2 >= 0) {
            if (is32) this->emit_cmp(phys_to_reg32(pr_s2), phys_to_reg32(pr_s1));
            else      this->emit_cmp(phys_to_reg64(pr_s2), phys_to_reg64(pr_s1));
         }
         // Try const-immediate: cmp $imm, reg
         else if (_func_def_inst && inst.rr.src2 != ir_vreg_none && inst.rr.src2 < _num_vregs) {
            uint32_t def = _func_def_inst[inst.rr.src2];
            bool used_imm = false;
            if (def < _func_inst_count) {
               auto& di = _func_insts[def];
               if (di.opcode == ir_op::const_i32 || di.opcode == ir_op::const_i64) {
                  int32_t imm = static_cast<int32_t>(di.imm64);
                  if (pr_s1 >= 0) {
                     if (is32) this->emit_cmp(imm, phys_to_reg32(pr_s1));
                     else      this->emit_cmp(imm, phys_to_reg64(pr_s1));
                  } else {
                     load_vreg_rax(inst.rr.src1);
                     if (is32) this->emit_cmp(imm, eax);
                     else      this->emit_cmp(imm, rax);
                  }
                  if (_func_use_count && _func_use_count[inst.rr.src2] == 1)
                     di.flags |= IR_DEAD;
                  used_imm = true;
               }
            }
            if (!used_imm) {
               load_vreg_rcx(inst.rr.src2);
               load_vreg_rax(inst.rr.src1);
               if (is32) this->emit_cmp(ecx, eax);
               else      this->emit_cmp(rcx, rax);
            }
         } else {
            load_vreg_rcx(inst.rr.src2);
            load_vreg_rax(inst.rr.src1);
            if (is32) this->emit_cmp(ecx, eax);
            else      this->emit_cmp(rcx, rax);
         }
         if ((inst.flags & IR_FUSE_NEXT) && emit_fused_branch(func, idx, cc)) return true;
         this->emit_setcc(cc, al);
         this->emit_bytes(0x0f, 0xb6, 0xc0); // movzbl %al, %eax
         int8_t pr_dest = get_phys(inst.dest);
         if (pr_dest >= 0 && phys_to_reg64(pr_dest) != rax)
            this->emit_mov(rax, phys_to_reg64(pr_dest));
         else
            store_rax_vreg(inst.dest);
         return true;
      }

      // Register-based shift with constant folding
      bool emit_shift_reg(ir_function& func, const ir_inst& inst, uint8_t reg_field, bool is32) {
         // Check for constant shift amount
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

      // Try to fold addr = add(base, const) into the load displacement.
      // Returns the base vreg and updated offset, or the original if no folding.
      uint32_t try_fold_addr(uint32_t addr_vreg, uint32_t& uoffset) {
         if (!_func_def_inst || addr_vreg >= _num_vregs) return addr_vreg;
         uint32_t def = _func_def_inst[addr_vreg];
         if (def >= _func_inst_count) return addr_vreg;
         auto& di = _func_insts[def];
         if (di.opcode != ir_op::i32_add) return addr_vreg;
         // Only fold if the add result is single-use (this load is the only consumer)
         if (!_func_use_count || _func_use_count[addr_vreg] != 1) return addr_vreg;
         // Check both sides for a non-negative constant
         for (int side = 0; side < 2; ++side) {
            uint32_t cv = (side == 0) ? di.rr.src2 : di.rr.src1;
            uint32_t bv = (side == 0) ? di.rr.src1 : di.rr.src2;
            if (cv >= _num_vregs) continue;
            uint32_t cd = _func_def_inst[cv];
            if (cd >= _func_inst_count) continue;
            auto& ci = _func_insts[cd];
            if (ci.opcode != ir_op::const_i32) continue;
            int32_t cval = static_cast<int32_t>(ci.imm64);
            if (cval < 0) continue;
            uint64_t combined = static_cast<uint64_t>(uoffset) + static_cast<uint64_t>(static_cast<uint32_t>(cval));
            if (combined > INT32_MAX) continue;
            uoffset = static_cast<uint32_t>(combined);
            return bv;
         }
         return addr_vreg;
      }

      // Register-based memory load
      template<class I, class R>
      bool emit_load_reg(const ir_inst& inst, I instr, R reg) {
         uint32_t uoffset = static_cast<uint32_t>(inst.ri.imm);
         uint32_t addr_vreg = try_fold_addr(inst.ri.src1, uoffset);
         int8_t pr_addr = get_phys(addr_vreg);
         int8_t pr_dest = get_phys(inst.dest);

         // Use addr physical register as SIB index when available
         general_register64 addr = rax;
         if (pr_addr >= 0 && phys_to_reg64(pr_addr) != rax) {
            addr = phys_to_reg64(pr_addr);
         } else {
            load_vreg_rax(addr_vreg);
         }

         // WASM effective address = i32.add(base, offset), must wrap at 2^32.
         // When offset != 0, add it using 32-bit arithmetic (wraps and zero-extends)
         // before using as a 64-bit SIB index. Otherwise the 64-bit SIB displacement
         // doesn't wrap, causing out-of-bounds accesses on large i32 base values.
         if (uoffset != 0) {
            // Move addr to rcx (temp), add offset in 32-bit, use rcx as index
            if (addr != rcx) this->emit_mov(addr, rcx);
            this->emit_add(static_cast<int32_t>(uoffset), ecx); // 32-bit wrapping add
            this->emit(instr, *(rcx + rsi + 0), reg);
         } else {
            this->emit(instr, *(addr + rsi + 0), reg);
         }

         // Move result from reg (eax/rax) to dest physical register if different
         if (pr_dest >= 0 && phys_to_reg64(pr_dest) != rax) {
            this->emit_mov(rax, phys_to_reg64(pr_dest));
         } else {
            store_rax_vreg(inst.dest);
         }
         return true;
      }

      // Register-based memory store
      // inst.dest = value vreg, inst.ri.src1 = addr vreg, inst.ri.imm = offset
      template<class I, class R>
      bool emit_store_reg(const ir_inst& inst, I instr, R reg) {
         uint32_t uoffset = static_cast<uint32_t>(inst.ri.imm);
         uint32_t addr_vreg = inst.ri.src1; // stores: no folding (addr may be shared)
         int8_t pr_addr = get_phys(addr_vreg);

         // If addr is in rax, load it to rcx BEFORE loading value into rax
         // (otherwise loading value to rax would destroy the address)
         general_register64 addr = rcx;
         if (pr_addr >= 0 && phys_to_reg64(pr_addr) == rax) {
            load_vreg_rcx(addr_vreg);  // save addr to rcx before rax is overwritten
            load_vreg_rax(inst.dest);  // then load value to rax
         } else {
            load_vreg_rax(inst.dest);  // value to rax
            if (pr_addr >= 0) {
               addr = phys_to_reg64(pr_addr);
            } else {
               load_vreg_rcx(addr_vreg);
            }
         }

         // WASM effective address wraps at 2^32 — add offset in 32-bit.
         if (uoffset != 0) {
            // Move addr to rdx, add offset in 32-bit (wraps + zero-extends)
            this->emit_mov(addr, rdx);
            this->emit_add(static_cast<int32_t>(uoffset), edx);
            this->emit(instr, *(rdx + rsi + 0), reg);
         } else {
            this->emit(instr, *(addr + rsi + 0), reg);
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
      void mark_block_end(ir_function& /*func*/, uint32_t block_idx, bool is_if) {
         if (block_idx >= _num_blocks) return;
         _block_addrs[block_idx] = code;
         // Patch all pending forward references to this block
         for (auto* f = _block_fixups[block_idx]; f; f = f->next) {
            base::fix_branch(f->branch, code);
         }
         _block_fixups[block_idx] = nullptr;
         // For if-blocks without else: patch the if_ conditional branch here
         if (is_if) {
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
         if (rt == types::v128) {
            // v128 return: save 16-byte value, pop depth_change * 8 bytes, push it back
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(static_cast<uint32_t>(depth_change * 8), rsp);
            this->emit_sub(16, rsp);
            this->emit_vmovdqu(xmm0, *rsp);
         } else if (rt != types::pseudo) {
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
         // Map: r8=0..r11=3 (caller-saved), rbx=4, r12=5..r15=8 (callee-saved)
         constexpr general_register64 map[] = {
            general_register64(8),  // r8
            general_register64(9),  // r9
            general_register64(10), // r10
            general_register64(11), // r11
            general_register64(3),  // rbx (callee-saved, freed from call depth duty)
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
            general_register32(3),  // ebx (callee-saved)
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

      // ──────── Constant folding helpers ────────
      // Check if a vreg was defined by a const instruction and return its value.
      // If found, marks the const as dead (won't emit code for it).
      bool try_get_const_i32(ir_function& func, uint32_t vreg, int32_t& out) {
         if (vreg == ir_vreg_none) return false;
         // Search backward for the defining instruction
         for (uint32_t j = func.inst_count; j > 0; --j) {
            auto& prev = func.insts[j - 1];
            if (prev.dest == vreg) {
               if (prev.opcode == ir_op::const_i32) {
                  out = static_cast<int32_t>(prev.imm64);
                  prev.flags |= IR_DEAD;
                  return true;
               }
               return false; // defined by non-const
            }
         }
         return false;
      }

      // ──────── SSE float register-mode helpers ────────
      // Binary f32 op: xmm0 = src1, xmm1 = src2, OPss xmm1, xmm0 → result in rax
      void emit_f32_binop_reg(const ir_inst& inst, uint8_t op) {
         load_vreg_rax(inst.rr.src1);
         this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);  // movq rax, xmm0
         load_vreg_rax(inst.rr.src2);
         this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc8);  // movq rax, xmm1
         this->emit_bytes(0xf3, 0x0f, op, 0xc1);           // OPss xmm1, xmm0
         this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);  // movq xmm0, rax
         store_rax_vreg(inst.dest);
      }
      // Binary f64 op
      void emit_f64_binop_reg(const ir_inst& inst, uint8_t op) {
         load_vreg_rax(inst.rr.src1);
         this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);  // movq rax, xmm0
         load_vreg_rax(inst.rr.src2);
         this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc8);  // movq rax, xmm1
         this->emit_bytes(0xf2, 0x0f, op, 0xc1);           // OPsd xmm1, xmm0
         this->emit_bytes(0x66, 0x48, 0x0f, 0x7e, 0xc0);  // movq xmm0, rax
         store_rax_vreg(inst.dest);
      }
      // Float comparison: cmpss/cmpsd with predicate, result = 0 or 1
      void emit_f32_relop_reg(const ir_inst& inst, uint8_t cmp_op, bool swap, bool flip) {
         if (swap) {
            load_vreg_rax(inst.rr.src2);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);  // movq rax, xmm0
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc8);  // movq rax, xmm1
         } else {
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);  // movq rax, xmm0
            load_vreg_rax(inst.rr.src2);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc8);  // movq rax, xmm1
         }
         this->emit_bytes(0xf3, 0x0f, 0xc2, 0xc1, cmp_op);   // cmpss $imm, xmm1, xmm0
         this->emit_bytes(0x66, 0x0f, 0x7e, 0xc0);            // movd xmm0, eax
         if (!flip) {
            this->emit_bytes(0x83, 0xe0, 0x01);               // and $1, eax
         } else {
            this->emit_bytes(0xff, 0xc0);                      // inc eax
         }
         store_rax_vreg(inst.dest);
      }
      void emit_f64_relop_reg(const ir_inst& inst, uint8_t cmp_op, bool swap, bool flip) {
         if (swap) {
            load_vreg_rax(inst.rr.src2);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc8);
         } else {
            load_vreg_rax(inst.rr.src1);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc0);
            load_vreg_rax(inst.rr.src2);
            this->emit_bytes(0x66, 0x48, 0x0f, 0x6e, 0xc8);
         }
         this->emit_bytes(0xf2, 0x0f, 0xc2, 0xc1, cmp_op);   // cmpsd $imm, xmm1, xmm0
         this->emit_bytes(0x66, 0x0f, 0x7e, 0xc0);            // movd xmm0, eax
         if (!flip) {
            this->emit_bytes(0x83, 0xe0, 0x01);
         } else {
            this->emit_bytes(0xff, 0xc0);
         }
         store_rax_vreg(inst.dest);
      }

      // ──────── SSE float helpers (stack mode) ────────
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
         // WASM effective address wraps at 2^32 — add offset in 32-bit
         if (uoffset != 0) {
            this->emit_add(static_cast<int32_t>(uoffset), eax); // 32-bit wrapping add
         }
         this->emit(instr, *(rax + rsi + 0), reg);
         this->emit_push_raw(rax);
      }

      template<class I, class R>
      void emit_store(int32_t offset, I instr, R reg) {
         uint32_t uoffset = static_cast<uint32_t>(offset);
         this->emit_pop_raw(rax);  // value
         this->emit_pop_raw(rcx);  // WASM address
         // WASM effective address wraps at 2^32 — add offset in 32-bit
         if (uoffset != 0) {
            this->emit_add(static_cast<int32_t>(uoffset), ecx); // 32-bit wrapping add
         }
         this->emit(instr, *(rcx + rsi + 0), reg);
      }

      // ──────── SIMD helpers ────────
      // Pop a WASM address from the x86 stack, add offset, compute native address.
      // Result: *(rax + rsi + 0) is the effective memory address.
      // Uses ecx as temp for large offsets.
      void simd_pop_address(uint32_t offset) {
         this->emit_pop_raw(rax);  // WASM address (i32)
         if (offset != 0) {
            this->emit_add(static_cast<int32_t>(offset), eax);  // 32-bit wrapping add
         }
         // rsi holds linear memory base; effective address is rax + rsi
      }

      // v128 load: pop address, load 16 bytes into xmm0, push 16 bytes
      template<typename Op>
      void simd_loadop(Op op, uint32_t offset) {
         simd_pop_address(offset);
         this->emit(op, *(rax + rsi + 0), xmm0);
         this->emit_sub(16, rsp);
         this->emit_vmovdqu(xmm0, *rsp);
      }

      // v128 store: pop 16 bytes into xmm0, pop address, store 16 bytes
      void simd_storeop(uint32_t offset) {
         this->emit_vmovups(*rsp, xmm0);
         this->emit_add(16, rsp);
         this->emit_pop_raw(rax);  // WASM address
         if (offset != 0) {
            this->emit_add(static_cast<int32_t>(offset), eax);
         }
         this->emit_add(rsi, rax);  // native address = wasm_addr + memory_base
         this->emit_vmovups(xmm0, *rax);
      }

      // v128 load lane: pop v128 into xmm0, pop address, insert lane
      template<typename Op>
      void simd_load_laneop(Op op, uint32_t offset, uint8_t lane) {
         this->emit_vmovdqu(*rsp, xmm0);
         this->emit_add(16, rsp);
         this->emit_pop_raw(rax);  // WASM address
         if (offset != 0) {
            this->emit_add(static_cast<int32_t>(offset), eax);
         }
         this->emit_add(rsi, rax);
         this->emit(op, typename base::imm8{lane}, *rax, xmm0, xmm0);
         this->emit_sub(16, rsp);
         this->emit_vmovdqu(xmm0, *rsp);
      }

      // v128 store lane: pop v128, pop address, extract lane to memory
      template<typename Op>
      void simd_store_laneop(Op op, uint32_t offset, uint8_t lane) {
         this->emit_vmovdqu(*rsp, xmm0);
         this->emit_add(16, rsp);
         this->emit_pop_raw(rax);  // WASM address
         if (offset != 0) {
            this->emit_add(static_cast<int32_t>(offset), eax);
         }
         this->emit_add(rsi, rax);
         this->emit(op, typename base::imm8{lane}, *rax, xmm0);
      }

      // v128 extract lane: pop v128, extract scalar lane, push scalar
      template<typename Op>
      void simd_extract_laneop(Op op, uint8_t lane) {
         this->emit_vmovdqu(*rsp, xmm0);
         this->emit_add(16, rsp);
         this->emit(op, typename base::imm8{lane}, rax, xmm0);
         this->emit_push_raw(rax);
      }

      // v128 replace lane: pop scalar, load v128 from TOS, insert, store back
      template<typename Op>
      void simd_replace_laneop(Op op, uint8_t lane) {
         this->emit_pop_raw(rax);
         this->emit_vmovdqu(*rsp, xmm0);
         this->emit(op, typename base::imm8{lane}, rax, xmm0, xmm0);
         this->emit_vmovdqu(xmm0, *rsp);
      }

      // v128 splat: pop scalar, broadcast to v128, push v128
      template<typename Op>
      void simd_splatop(Op op) {
         this->emit(op, *rsp, xmm0);
         this->emit_sub(8, rsp);
         this->emit_vmovdqu(xmm0, *rsp);
      }

      // v128 unary: load TOS v128 into xmm0, apply op, store back
      template<typename Op>
      void simd_v128_unop(Op op) {
         this->emit(op, *rsp, xmm0);
         this->emit_vmovdqu(xmm0, *rsp);
      }

      // v128 binary (non-commutative): pop top v128 (xmm0), operate with NOS v128, store to NOS
      template<typename Op>
      void simd_v128_binop(Op op) {
         this->emit_vmovdqu(*rsp, xmm0);
         this->emit_add(16, rsp);
         this->emit_vmovdqu(*rsp, xmm1);
         this->emit(op, xmm0, xmm1, xmm0);
         this->emit_vmovdqu(xmm0, *rsp);
      }

      // v128 binary (commutative or right-operand form): pop top v128, operate with NOS from memory
      template<typename Op>
      void simd_v128_binop_r(Op op) {
         this->emit_vmovdqu(*rsp, xmm0);
         this->emit_add(16, rsp);
         this->emit(op, *rsp, xmm0, xmm0);
         this->emit_vmovdqu(xmm0, *rsp);
      }

      // v128 shift: pop i32 shift count, load v128 from TOS, shift, store back
      template<typename Op>
      void simd_v128_shiftop(Op op, uint8_t mask) {
         this->emit_pop_raw(rax);
         this->emit_bytes(0x83, 0xe0, mask);  // and $mask, %eax
         this->emit_vmovdqu(*rsp, xmm0);
         this->emit_vmovd(eax, xmm1);
         this->emit(op, xmm1, xmm0, xmm0);
         this->emit_vmovdqu(xmm0, *rsp);
      }

      // v128 comparison with cmp opcode
      template<typename Op>
      void simd_v128_irelop_cmp(Op op, bool switch_params, bool flip_result) {
         this->emit_vmovdqu(*rsp, xmm0);
         this->emit_add(16, rsp);
         if (!switch_params) {
            this->emit_vmovdqu(*rsp, xmm1);
            this->emit(op, xmm0, xmm1, xmm0);
         } else {
            this->emit(op, *rsp, xmm0, xmm0);
         }
         if (flip_result) {
            this->emit_const_ones(xmm1);
            this->emit(base::VPXOR, xmm1, xmm0, xmm0);
         }
         this->emit_vmovdqu(xmm0, *rsp);
      }

      // v128 comparison via min/max
      template<typename Op, typename Eq>
      void simd_v128_irelop_minmax(Op op, Eq eq, bool flip_result) {
         this->emit_vmovdqu(*rsp, xmm0);
         this->emit_add(16, rsp);
         this->emit(op, *rsp, xmm0, xmm1);
         this->emit(eq, xmm0, xmm1, xmm0);
         if (!flip_result) {
            this->emit_const_zero(xmm1);
            this->emit(base::VPCMPEQB, xmm1, xmm0, xmm0);
         }
         this->emit_vmovdqu(xmm0, *rsp);
      }

      // v128 test ops (all_true, bitmask) that produce an i32 result
      void simd_v128_test_all_true(auto eq_op) {
         this->emit_const_zero(xmm0);
         this->emit(eq_op, *rsp, xmm0, xmm0);
         this->emit_add(16, rsp);
         this->emit_xor(eax, eax);
         this->emit(base::VPTEST, xmm0, xmm0);
         this->emit(base::SETZ, al);
         this->emit_push_raw(rax);
      }

      void simd_v128_bitmask(auto pmovmskb_op) {
         this->emit_vmovdqu(*rsp, xmm0);
         this->emit_add(16, rsp);
         this->emit(pmovmskb_op, xmm0, rax);
         this->emit_push_raw(rax);
      }

      // Softfloat unary: call a v128_t -> v128_t function
      void simd_v128_unop_softfloat(v128_t (*fn)(v128_t)) {
         // The v128 argument is on the x86 stack as 16 bytes.
         // SysV ABI: v128_t is returned in rax:rdx (two uint64_t fields).
         // v128_t argument: passed as two uint64_t in rdi, rsi.
         // But rsi is the linear memory base! We must save/restore it.
         this->emit_push_raw(rdi);
         this->emit_push_raw(rsi);
         // Load v128 argument from stack (offset by the two pushes = +16)
         this->emit_mov(*(rsp + 16), rdi);
         this->emit_mov(*(rsp + 24), rsi);
         // Align stack
         this->emit_mov(rsp, rcx);
         this->emit_bytes(0x48, 0x83, 0xe4, 0xf0);  // and $-16, %rsp
         this->emit_push_raw(rcx);
         // Call
         this->emit_bytes(0x48, 0xb8);
         this->emit_operand_ptr(fn);
         this->emit_bytes(0xff, 0xd0);  // call *%rax
         // Restore stack
         this->emit_pop_raw(rsp);
         this->emit_pop_raw(rsi);
         this->emit_pop_raw(rdi);
         // Store result back to TOS v128
         this->emit_mov(rax, *rsp);
         this->emit_mov(rdx, *(rsp + 8));
      }

      // Softfloat binary: call a (v128_t, v128_t) -> v128_t function
      void simd_v128_binop_softfloat(v128_t (*fn)(v128_t, v128_t)) {
         // Two v128 args on stack: TOS = arg2 (16 bytes), NOS = arg1 (16 bytes)
         // SysV ABI: first arg in rdi:rsi, second in rdx:rcx
         // But rsi is linear memory base, so save/restore.
         this->emit_push_raw(rdi);
         this->emit_push_raw(rsi);
         // arg2 (TOS) at rsp+16, arg1 (NOS) at rsp+32
         this->emit_mov(*(rsp + 16), rdx);   // arg2.low
         this->emit_mov(*(rsp + 24), rcx);   // arg2.high
         this->emit_mov(*(rsp + 32), rdi);   // arg1.low
         this->emit_mov(*(rsp + 40), rsi);   // arg1.high
         // Align stack
         this->emit_mov(rsp, r8);
         this->emit_bytes(0x48, 0x83, 0xe4, 0xf0);  // and $-16, %rsp
         this->emit_push_raw(r8);
         // Call
         this->emit_bytes(0x48, 0xb8);
         this->emit_operand_ptr(fn);
         this->emit_bytes(0xff, 0xd0);
         // Restore stack
         this->emit_pop_raw(rsp);
         this->emit_pop_raw(rsi);
         this->emit_pop_raw(rdi);
         // Remove arg2 (16 bytes), store result to NOS (which becomes TOS)
         this->emit_add(16, rsp);
         this->emit_mov(rax, *rsp);
         this->emit_mov(rdx, *(rsp + 8));
      }

      // Main SIMD dispatch
      void emit_simd_op(const ir_inst& inst) {
         auto sub = static_cast<simd_sub>(inst.dest);
         switch (sub) {
         // ── Memory operations ──
         case simd_sub::v128_load: simd_loadop(base::VMOVDQU_A, inst.simd.offset); break;
         case simd_sub::v128_load8x8_s: simd_loadop(base::VPMOVSXBW, inst.simd.offset); break;
         case simd_sub::v128_load8x8_u: simd_loadop(base::VPMOVZXBW, inst.simd.offset); break;
         case simd_sub::v128_load16x4_s: simd_loadop(base::VPMOVSXWD, inst.simd.offset); break;
         case simd_sub::v128_load16x4_u: simd_loadop(base::VPMOVZXWD, inst.simd.offset); break;
         case simd_sub::v128_load32x2_s: simd_loadop(base::VPMOVSXDQ, inst.simd.offset); break;
         case simd_sub::v128_load32x2_u: simd_loadop(base::VPMOVZXDQ, inst.simd.offset); break;
         case simd_sub::v128_load8_splat: simd_loadop(base::VPBROADCASTB, inst.simd.offset); break;
         case simd_sub::v128_load16_splat: simd_loadop(base::VPBROADCASTW, inst.simd.offset); break;
         case simd_sub::v128_load32_splat: simd_loadop(base::VPBROADCASTD, inst.simd.offset); break;
         case simd_sub::v128_load64_splat: simd_loadop(base::VPBROADCASTQ, inst.simd.offset); break;
         case simd_sub::v128_load32_zero: simd_loadop(base::VMOVD_A, inst.simd.offset); break;
         case simd_sub::v128_load64_zero: simd_loadop(base::VMOVQ_A, inst.simd.offset); break;
         case simd_sub::v128_store: simd_storeop(inst.simd.offset); break;

         case simd_sub::v128_load8_lane: simd_load_laneop(base::VPINSRB, inst.simd.offset, inst.simd.lane); break;
         case simd_sub::v128_load16_lane: simd_load_laneop(base::VPINSRW, inst.simd.offset, inst.simd.lane); break;
         case simd_sub::v128_load32_lane: simd_load_laneop(base::VPINSRD, inst.simd.offset, inst.simd.lane); break;
         case simd_sub::v128_load64_lane: simd_load_laneop(base::VPINSRQ, inst.simd.offset, inst.simd.lane); break;
         case simd_sub::v128_store8_lane: simd_store_laneop(base::VPEXTRB, inst.simd.offset, inst.simd.lane); break;
         case simd_sub::v128_store16_lane: simd_store_laneop(base::VPEXTRW, inst.simd.offset, inst.simd.lane); break;
         case simd_sub::v128_store32_lane: simd_store_laneop(base::VPEXTRD, inst.simd.offset, inst.simd.lane); break;
         case simd_sub::v128_store64_lane: simd_store_laneop(base::VPEXTRQ, inst.simd.offset, inst.simd.lane); break;

         // ── Shuffle ──
         case simd_sub::i8x16_shuffle: {
            const uint8_t* lanes = reinterpret_cast<const uint8_t*>(&inst.immv128);
            auto emit_shuffle_operand = [this](const uint8_t* l) {
               for (int i = 0; i < 8; ++i)
                  this->emit_bytes(l[i] < 16 ? ~l[i] : l[i]);
            };
            // movabsq $lanes[0-7], %rax
            this->emit_bytes(0x48, 0xb8);
            emit_shuffle_operand(lanes);
            this->emit_vmovq(rax, xmm2);
            // movabsq $lanes[8-15], %rax
            this->emit_bytes(0x48, 0xb8);
            emit_shuffle_operand(lanes + 8);
            this->emit(base::VPINSRQ, typename base::imm8{1}, rax, xmm2, xmm2);

            this->emit_vmovdqu(*rsp, xmm0);
            this->emit(base::VPSHUFB, xmm2, xmm0, xmm1);
            this->emit_const_ones(xmm0);
            this->emit(base::VPXOR, xmm0, xmm2, xmm2);
            this->emit_add(16, rsp);
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit(base::VPSHUFB, xmm2, xmm0, xmm0);
            this->emit(base::VPOR, xmm1, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }

         case simd_sub::i8x16_swizzle: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit_mov(0x70707070u, eax);
            this->emit_vmovd(eax, xmm1);
            this->emit(base::VPSHUFD, typename base::imm8{0}, xmm1, xmm1);
            this->emit(base::VPADDUSB, xmm1, xmm0, xmm1);
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit(base::VPSHUFB, xmm1, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }

         // ── Extract/Replace lane ──
         case simd_sub::i8x16_extract_lane_s: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit_vpextrb(inst.simd.lane, xmm0, rax);
            this->emit(base::MOVSXB, al, eax);
            this->emit_push_raw(rax);
            break;
         }
         case simd_sub::i8x16_extract_lane_u: simd_extract_laneop(base::VPEXTRB, inst.simd.lane); break;
         case simd_sub::i8x16_replace_lane: simd_replace_laneop(base::VPINSRB, inst.simd.lane); break;
         case simd_sub::i16x8_extract_lane_s: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit_vpextrw(inst.simd.lane, xmm0, rax);
            this->emit(base::MOVSXW, this->ax, eax);
            this->emit_push_raw(rax);
            break;
         }
         case simd_sub::i16x8_extract_lane_u: simd_extract_laneop(base::VPEXTRW, inst.simd.lane); break;
         case simd_sub::i16x8_replace_lane: simd_replace_laneop(base::VPINSRW, inst.simd.lane); break;
         case simd_sub::i32x4_extract_lane: simd_extract_laneop(base::VPEXTRD, inst.simd.lane); break;
         case simd_sub::i32x4_replace_lane: simd_replace_laneop(base::VPINSRD, inst.simd.lane); break;
         case simd_sub::i64x2_extract_lane: simd_extract_laneop(base::VPEXTRQ, inst.simd.lane); break;
         case simd_sub::i64x2_replace_lane: simd_replace_laneop(base::VPINSRQ, inst.simd.lane); break;
         case simd_sub::f32x4_extract_lane: simd_extract_laneop(base::VPEXTRD, inst.simd.lane); break;
         case simd_sub::f32x4_replace_lane: simd_replace_laneop(base::VPINSRD, inst.simd.lane); break;
         case simd_sub::f64x2_extract_lane: simd_extract_laneop(base::VPEXTRQ, inst.simd.lane); break;
         case simd_sub::f64x2_replace_lane: simd_replace_laneop(base::VPINSRQ, inst.simd.lane); break;

         // ── Splat ──
         case simd_sub::i8x16_splat: simd_splatop(base::VPBROADCASTB); break;
         case simd_sub::i16x8_splat: simd_splatop(base::VPBROADCASTW); break;
         case simd_sub::i32x4_splat: simd_splatop(base::VPBROADCASTD); break;
         case simd_sub::i64x2_splat: simd_splatop(base::VPBROADCASTQ); break;
         case simd_sub::f32x4_splat: simd_splatop(base::VPBROADCASTD); break;
         case simd_sub::f64x2_splat: simd_splatop(base::VPBROADCASTQ); break;

         // ── i8x16 comparisons ──
         case simd_sub::i8x16_eq: simd_v128_irelop_cmp(base::VPCMPEQB, true, false); break;
         case simd_sub::i8x16_ne: simd_v128_irelop_cmp(base::VPCMPEQB, true, true); break;
         case simd_sub::i8x16_lt_s: simd_v128_irelop_cmp(base::VPCMPGTB, true, false); break;
         case simd_sub::i8x16_lt_u: simd_v128_irelop_minmax(base::VPMINUB, base::VPCMPEQB, false); break;
         case simd_sub::i8x16_gt_s: simd_v128_irelop_cmp(base::VPCMPGTB, false, false); break;
         case simd_sub::i8x16_gt_u: simd_v128_irelop_minmax(base::VPMAXUB, base::VPCMPEQB, false); break;
         case simd_sub::i8x16_le_s: simd_v128_irelop_cmp(base::VPCMPGTB, false, true); break;
         case simd_sub::i8x16_le_u: simd_v128_irelop_minmax(base::VPMAXUB, base::VPCMPEQB, true); break;
         case simd_sub::i8x16_ge_s: simd_v128_irelop_cmp(base::VPCMPGTB, true, true); break;
         case simd_sub::i8x16_ge_u: simd_v128_irelop_minmax(base::VPMINUB, base::VPCMPEQB, true); break;

         // ── i16x8 comparisons ──
         case simd_sub::i16x8_eq: simd_v128_irelop_cmp(base::VPCMPEQW, true, false); break;
         case simd_sub::i16x8_ne: simd_v128_irelop_cmp(base::VPCMPEQW, true, true); break;
         case simd_sub::i16x8_lt_s: simd_v128_irelop_cmp(base::VPCMPGTW, true, false); break;
         case simd_sub::i16x8_lt_u: simd_v128_irelop_minmax(base::VPMINUW, base::VPCMPEQW, false); break;
         case simd_sub::i16x8_gt_s: simd_v128_irelop_cmp(base::VPCMPGTW, false, false); break;
         case simd_sub::i16x8_gt_u: simd_v128_irelop_minmax(base::VPMAXUW, base::VPCMPEQW, false); break;
         case simd_sub::i16x8_le_s: simd_v128_irelop_cmp(base::VPCMPGTW, false, true); break;
         case simd_sub::i16x8_le_u: simd_v128_irelop_minmax(base::VPMAXUW, base::VPCMPEQW, true); break;
         case simd_sub::i16x8_ge_s: simd_v128_irelop_cmp(base::VPCMPGTW, true, true); break;
         case simd_sub::i16x8_ge_u: simd_v128_irelop_minmax(base::VPMINUW, base::VPCMPEQW, true); break;

         // ── i32x4 comparisons ──
         case simd_sub::i32x4_eq: simd_v128_irelop_cmp(base::VPCMPEQD, true, false); break;
         case simd_sub::i32x4_ne: simd_v128_irelop_cmp(base::VPCMPEQD, true, true); break;
         case simd_sub::i32x4_lt_s: simd_v128_irelop_cmp(base::VPCMPGTD, true, false); break;
         case simd_sub::i32x4_lt_u: simd_v128_irelop_minmax(base::VPMINUD, base::VPCMPEQD, false); break;
         case simd_sub::i32x4_gt_s: simd_v128_irelop_cmp(base::VPCMPGTD, false, false); break;
         case simd_sub::i32x4_gt_u: simd_v128_irelop_minmax(base::VPMAXUD, base::VPCMPEQD, false); break;
         case simd_sub::i32x4_le_s: simd_v128_irelop_cmp(base::VPCMPGTD, false, true); break;
         case simd_sub::i32x4_le_u: simd_v128_irelop_minmax(base::VPMAXUD, base::VPCMPEQD, true); break;
         case simd_sub::i32x4_ge_s: simd_v128_irelop_cmp(base::VPCMPGTD, true, true); break;
         case simd_sub::i32x4_ge_u: simd_v128_irelop_minmax(base::VPMINUD, base::VPCMPEQD, true); break;

         // ── i64x2 comparisons ──
         case simd_sub::i64x2_eq: simd_v128_irelop_cmp(base::VPCMPEQQ, true, false); break;
         case simd_sub::i64x2_ne: simd_v128_irelop_cmp(base::VPCMPEQQ, true, true); break;
         case simd_sub::i64x2_lt_s: simd_v128_irelop_cmp(base::VPCMPGTQ, true, false); break;
         case simd_sub::i64x2_gt_s: simd_v128_irelop_cmp(base::VPCMPGTQ, false, false); break;
         case simd_sub::i64x2_le_s: simd_v128_irelop_cmp(base::VPCMPGTQ, false, true); break;
         case simd_sub::i64x2_ge_s: simd_v128_irelop_cmp(base::VPCMPGTQ, true, true); break;

         // ── f32x4 comparisons (softfloat) ──
         case simd_sub::f32x4_eq: simd_v128_binop_softfloat(&_eosio_f32x4_eq); break;
         case simd_sub::f32x4_ne: simd_v128_binop_softfloat(&_eosio_f32x4_ne); break;
         case simd_sub::f32x4_lt: simd_v128_binop_softfloat(&_eosio_f32x4_lt); break;
         case simd_sub::f32x4_gt: simd_v128_binop_softfloat(&_eosio_f32x4_gt); break;
         case simd_sub::f32x4_le: simd_v128_binop_softfloat(&_eosio_f32x4_le); break;
         case simd_sub::f32x4_ge: simd_v128_binop_softfloat(&_eosio_f32x4_ge); break;

         // ── f64x2 comparisons (softfloat) ──
         case simd_sub::f64x2_eq: simd_v128_binop_softfloat(&_eosio_f64x2_eq); break;
         case simd_sub::f64x2_ne: simd_v128_binop_softfloat(&_eosio_f64x2_ne); break;
         case simd_sub::f64x2_lt: simd_v128_binop_softfloat(&_eosio_f64x2_lt); break;
         case simd_sub::f64x2_gt: simd_v128_binop_softfloat(&_eosio_f64x2_gt); break;
         case simd_sub::f64x2_le: simd_v128_binop_softfloat(&_eosio_f64x2_le); break;
         case simd_sub::f64x2_ge: simd_v128_binop_softfloat(&_eosio_f64x2_ge); break;

         // ── Logical ──
         case simd_sub::v128_not: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_const_ones(xmm1);
            this->emit(base::VPXOR, xmm1, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::v128_and: simd_v128_binop_r(base::VPAND); break;
         case simd_sub::v128_andnot: simd_v128_binop_r(base::VPANDN); break;
         case simd_sub::v128_or: simd_v128_binop_r(base::VPOR); break;
         case simd_sub::v128_xor: simd_v128_binop_r(base::VPXOR); break;

         case simd_sub::v128_bitselect: {
            // Stack: val1 (NOS+16), val2 (NOS), mask (TOS)
            this->emit_vmovdqu(*rsp, xmm2);          // mask
            this->emit_vmovdqu(*(rsp + 16), xmm1);   // val2
            this->emit_add(32, rsp);
            this->emit_vmovdqu(*rsp, xmm0);           // val1
            this->emit(base::VPAND, xmm0, xmm2, xmm0);
            this->emit(base::VPANDN, xmm1, xmm2, xmm1);
            this->emit(base::VPOR, xmm0, xmm1, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }

         case simd_sub::v128_any_true: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit_xor(eax, eax);
            this->emit(base::VPTEST, xmm0, xmm0);
            this->emit(base::SETNZ, al);
            this->emit_push_raw(rax);
            break;
         }

         // ── i8x16 arithmetic ──
         case simd_sub::i8x16_abs: simd_v128_unop(base::VPABSB); break;
         case simd_sub::i8x16_neg: {
            this->emit_const_zero(xmm0);
            this->emit(base::VPSUBB, *rsp, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i8x16_popcnt: {
            static const uint8_t popcnt4[] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};
            this->emit_bytes(0x48, 0xb8);
            this->emit_operand_ptr(&popcnt4);
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_vmovdqu(*rax, xmm3);
            this->emit_mov(0x0fu, eax);
            this->emit_vmovd(eax, xmm2);
            this->emit(base::VPBROADCASTB, xmm2, xmm2);
            this->emit(base::VPSRLQ_c, typename base::imm8{4}, xmm0, xmm1);
            this->emit(base::VPAND, xmm2, xmm0, xmm0);
            this->emit(base::VPAND, xmm2, xmm1, xmm1);
            this->emit(base::VPSHUFB, xmm0, xmm3, xmm0);
            this->emit(base::VPSHUFB, xmm1, xmm3, xmm1);
            this->emit(base::VPADDB, xmm0, xmm1, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i8x16_all_true: simd_v128_test_all_true(base::VPCMPEQB); break;
         case simd_sub::i8x16_bitmask: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit(base::VPMOVMSKB, xmm0, rax);
            this->emit_push_raw(rax);
            break;
         }
         case simd_sub::i8x16_narrow_i16x8_s: simd_v128_binop(base::VPACKSSWB); break;
         case simd_sub::i8x16_narrow_i16x8_u: simd_v128_binop(base::VPACKUSWB); break;
         case simd_sub::i8x16_shl: {
            this->emit_pop_raw(rax);
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_bytes(0x83, 0xe0, 0x07);  // and $7, %eax
            this->emit_vmovd(eax, xmm2);
            this->emit_const_ones(xmm1);
            this->emit(base::VPSLLD, xmm2, xmm1, xmm1);
            this->emit(base::VPBROADCASTB, xmm1, xmm1);
            this->emit(base::VPSLLW, xmm2, xmm0, xmm0);
            this->emit(base::VPAND, xmm0, xmm1, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i8x16_shr_s: {
            this->emit_pop_raw(rax);
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_bytes(0x83, 0xe0, 0x07);  // and $7, %eax
            this->emit_vmovd(eax, xmm2);
            this->emit_const_ones(xmm3);
            this->emit(base::VPSLLW_c, typename base::imm8{8}, xmm3, xmm3);
            this->emit(base::VPSLLW_c, typename base::imm8{8}, xmm0, xmm1);
            this->emit(base::VPSRAW_c, typename base::imm8{8}, xmm1, xmm1);
            this->emit(base::VPSLLW, xmm2, xmm3, xmm3);
            this->emit(base::VPANDN, xmm1, xmm3, xmm1);
            this->emit(base::VPAND, xmm3, xmm0, xmm0);
            this->emit(base::VPOR, xmm1, xmm0, xmm0);
            this->emit(base::VPSRAW, xmm2, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i8x16_shr_u: {
            this->emit_pop_raw(rax);
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_bytes(0x83, 0xe0, 0x07);  // and $7, %eax
            this->emit_vmovd(eax, xmm2);
            this->emit_const_ones(xmm1);
            this->emit(base::VPSLLW_c, typename base::imm8{8}, xmm1, xmm1);
            this->emit(base::VPSRLW, xmm2, xmm1, xmm1);
            this->emit(base::VPBROADCASTB, xmm1, xmm1);
            this->emit(base::VPSRLW, xmm2, xmm0, xmm0);
            this->emit(base::VPANDN, xmm0, xmm1, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i8x16_add: simd_v128_binop_r(base::VPADDB); break;
         case simd_sub::i8x16_add_sat_s: simd_v128_binop_r(base::VPADDSB); break;
         case simd_sub::i8x16_add_sat_u: simd_v128_binop_r(base::VPADDUSB); break;
         case simd_sub::i8x16_sub: simd_v128_binop(base::VPSUBB); break;
         case simd_sub::i8x16_sub_sat_s: simd_v128_binop(base::VPSUBSB); break;
         case simd_sub::i8x16_sub_sat_u: simd_v128_binop(base::VPSUBUSB); break;
         case simd_sub::i8x16_min_s: simd_v128_binop_r(base::VPMINSB); break;
         case simd_sub::i8x16_min_u: simd_v128_binop_r(base::VPMINUB); break;
         case simd_sub::i8x16_max_s: simd_v128_binop_r(base::VPMAXSB); break;
         case simd_sub::i8x16_max_u: simd_v128_binop_r(base::VPMAXUB); break;
         case simd_sub::i8x16_avgr_u: simd_v128_binop_r(base::VPAVGB); break;

         // ── i16x8 arithmetic ──
         case simd_sub::i16x8_extadd_pairwise_i8x16_s: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm0, xmm1);
            this->emit(base::VPMOVSXBW, xmm0, xmm0);
            this->emit(base::VPMOVSXBW, xmm1, xmm1);
            this->emit(base::VPHADDW, xmm1, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i16x8_extadd_pairwise_i8x16_u: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm0, xmm1);
            this->emit(base::VPMOVZXBW, xmm0, xmm0);
            this->emit(base::VPMOVZXBW, xmm1, xmm1);
            this->emit(base::VPHADDW, xmm1, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i16x8_abs: simd_v128_unop(base::VPABSW); break;
         case simd_sub::i16x8_neg: {
            this->emit_const_zero(xmm0);
            this->emit(base::VPSUBW, *rsp, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i16x8_q15mulr_sat_s: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit(base::VPMULHRSW, *rsp, xmm0, xmm0);
            this->emit_const_ones(xmm1);
            this->emit(base::VPSLLW_c, typename base::imm8{15}, xmm1, xmm1);
            this->emit(base::VPCMPEQW, xmm1, xmm0, xmm1);
            this->emit(base::VPXOR, xmm1, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i16x8_all_true: simd_v128_test_all_true(base::VPCMPEQW); break;
         case simd_sub::i16x8_bitmask: {
            this->emit_const_zero(xmm0);
            this->emit(base::VPCMPGTW, *rsp, xmm0, xmm1);
            this->emit(base::VPACKSSWB, xmm0, xmm1, xmm0);
            this->emit_add(16, rsp);
            this->emit(base::VPMOVMSKB, xmm0, rax);
            this->emit_push_raw(rax);
            break;
         }
         case simd_sub::i16x8_narrow_i32x4_s: simd_v128_binop(base::VPACKSSDW); break;
         case simd_sub::i16x8_narrow_i32x4_u: simd_v128_binop(base::VPACKUSDW); break;
         case simd_sub::i16x8_extend_low_i8x16_s: simd_v128_unop(base::VPMOVSXBW); break;
         case simd_sub::i16x8_extend_high_i8x16_s: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm0, xmm0);
            this->emit(base::VPMOVSXBW, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i16x8_extend_low_i8x16_u: simd_v128_unop(base::VPMOVZXBW); break;
         case simd_sub::i16x8_extend_high_i8x16_u: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm0, xmm0);
            this->emit(base::VPMOVZXBW, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i16x8_shl: simd_v128_shiftop(base::VPSLLW, 0x0f); break;
         case simd_sub::i16x8_shr_s: simd_v128_shiftop(base::VPSRAW, 0x0f); break;
         case simd_sub::i16x8_shr_u: simd_v128_shiftop(base::VPSRLW, 0x0f); break;
         case simd_sub::i16x8_add: simd_v128_binop_r(base::VPADDW); break;
         case simd_sub::i16x8_add_sat_s: simd_v128_binop_r(base::VPADDSW); break;
         case simd_sub::i16x8_add_sat_u: simd_v128_binop_r(base::VPADDUSW); break;
         case simd_sub::i16x8_sub: simd_v128_binop(base::VPSUBW); break;
         case simd_sub::i16x8_sub_sat_s: simd_v128_binop(base::VPSUBSW); break;
         case simd_sub::i16x8_sub_sat_u: simd_v128_binop(base::VPSUBUSW); break;
         case simd_sub::i16x8_mul: simd_v128_binop_r(base::VPMULLW); break;
         case simd_sub::i16x8_min_s: simd_v128_binop_r(base::VPMINSW); break;
         case simd_sub::i16x8_min_u: simd_v128_binop_r(base::VPMINUW); break;
         case simd_sub::i16x8_max_s: simd_v128_binop_r(base::VPMAXSW); break;
         case simd_sub::i16x8_max_u: simd_v128_binop_r(base::VPMAXUW); break;
         case simd_sub::i16x8_avgr_u: simd_v128_binop_r(base::VPAVGW); break;
         case simd_sub::i16x8_extmul_low_i8x16_s: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit_vmovdqu(*rsp, xmm1);
            this->emit(base::VPMOVSXBW, xmm0, xmm0);
            this->emit(base::VPMOVSXBW, xmm1, xmm1);
            this->emit(base::VPMULLW, xmm0, xmm1, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i16x8_extmul_high_i8x16_s: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit_vmovdqu(*rsp, xmm1);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm0, xmm0);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm1, xmm1);
            this->emit(base::VPMOVSXBW, xmm0, xmm0);
            this->emit(base::VPMOVSXBW, xmm1, xmm1);
            this->emit(base::VPMULLW, xmm0, xmm1, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i16x8_extmul_low_i8x16_u: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit_vmovdqu(*rsp, xmm1);
            this->emit(base::VPMOVZXBW, xmm0, xmm0);
            this->emit(base::VPMOVZXBW, xmm1, xmm1);
            this->emit(base::VPMULLW, xmm0, xmm1, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i16x8_extmul_high_i8x16_u: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit_vmovdqu(*rsp, xmm1);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm0, xmm0);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm1, xmm1);
            this->emit(base::VPMOVZXBW, xmm0, xmm0);
            this->emit(base::VPMOVZXBW, xmm1, xmm1);
            this->emit(base::VPMULLW, xmm0, xmm1, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }

         // ── i32x4 arithmetic ──
         case simd_sub::i32x4_extadd_pairwise_i16x8_s: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm0, xmm1);
            this->emit(base::VPMOVSXWD, xmm0, xmm0);
            this->emit(base::VPMOVSXWD, xmm1, xmm1);
            this->emit(base::VPHADDD, xmm1, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i32x4_extadd_pairwise_i16x8_u: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm0, xmm1);
            this->emit(base::VPMOVZXWD, xmm0, xmm0);
            this->emit(base::VPMOVZXWD, xmm1, xmm1);
            this->emit(base::VPHADDD, xmm1, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i32x4_abs: simd_v128_unop(base::VPABSD); break;
         case simd_sub::i32x4_neg: {
            this->emit_const_zero(xmm0);
            this->emit(base::VPSUBD, *rsp, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i32x4_all_true: {
            this->emit_const_zero(xmm0);
            this->emit(base::VPCMPEQD, *rsp, xmm0, xmm0);
            this->emit_add(16, rsp);
            this->emit_xor(eax, eax);
            this->emit(base::VPTEST, xmm0, xmm0);
            this->emit(base::SETZ, al);
            this->emit_push_raw(rax);
            break;
         }
         case simd_sub::i32x4_bitmask: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit(base::VMOVMSKPS, xmm0, rax);
            this->emit_push_raw(rax);
            break;
         }
         case simd_sub::i32x4_extend_low_i16x8_s: simd_v128_unop(base::VPMOVSXWD); break;
         case simd_sub::i32x4_extend_high_i16x8_s: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm0, xmm0);
            this->emit(base::VPMOVSXWD, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i32x4_extend_low_i16x8_u: simd_v128_unop(base::VPMOVZXWD); break;
         case simd_sub::i32x4_extend_high_i16x8_u: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm0, xmm0);
            this->emit(base::VPMOVZXWD, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i32x4_shl: simd_v128_shiftop(base::VPSLLD, 0x1f); break;
         case simd_sub::i32x4_shr_s: simd_v128_shiftop(base::VPSRAD, 0x1f); break;
         case simd_sub::i32x4_shr_u: simd_v128_shiftop(base::VPSRLD, 0x1f); break;
         case simd_sub::i32x4_add: simd_v128_binop_r(base::VPADDD); break;
         case simd_sub::i32x4_sub: simd_v128_binop(base::VPSUBD); break;
         case simd_sub::i32x4_mul: simd_v128_binop_r(base::VPMULLD); break;
         case simd_sub::i32x4_min_s: simd_v128_binop_r(base::VPMINSD); break;
         case simd_sub::i32x4_min_u: simd_v128_binop_r(base::VPMINUD); break;
         case simd_sub::i32x4_max_s: simd_v128_binop_r(base::VPMAXSD); break;
         case simd_sub::i32x4_max_u: simd_v128_binop_r(base::VPMAXUD); break;
         case simd_sub::i32x4_dot_i16x8_s: simd_v128_binop_r(base::VPMADDWD); break;
         case simd_sub::i32x4_extmul_low_i16x8_s: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit_vmovdqu(*rsp, xmm1);
            this->emit(base::VPMOVSXWD, xmm0, xmm0);
            this->emit(base::VPMOVSXWD, xmm1, xmm1);
            this->emit(base::VPMULLD, xmm0, xmm1, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i32x4_extmul_high_i16x8_s: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit_vmovdqu(*rsp, xmm1);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm0, xmm0);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm1, xmm1);
            this->emit(base::VPMOVSXWD, xmm0, xmm0);
            this->emit(base::VPMOVSXWD, xmm1, xmm1);
            this->emit(base::VPMULLD, xmm0, xmm1, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i32x4_extmul_low_i16x8_u: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit_vmovdqu(*rsp, xmm1);
            this->emit(base::VPMOVZXWD, xmm0, xmm0);
            this->emit(base::VPMOVZXWD, xmm1, xmm1);
            this->emit(base::VPMULLD, xmm0, xmm1, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i32x4_extmul_high_i16x8_u: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit_vmovdqu(*rsp, xmm1);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm0, xmm0);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm1, xmm1);
            this->emit(base::VPMOVZXWD, xmm0, xmm0);
            this->emit(base::VPMOVZXWD, xmm1, xmm1);
            this->emit(base::VPMULLD, xmm0, xmm1, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }

         // ── i64x2 arithmetic ──
         case simd_sub::i64x2_abs: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_const_zero(xmm1);
            this->emit(base::VPCMPGTQ, xmm0, xmm1, xmm1);  // xmm1 = 0 > x ? -1 : 0
            this->emit(base::VPXOR, xmm0, xmm1, xmm0);
            this->emit(base::VPSUBQ, xmm1, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i64x2_neg: {
            this->emit_const_zero(xmm0);
            this->emit(base::VPSUBQ, *rsp, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i64x2_all_true: {
            this->emit_const_zero(xmm0);
            this->emit(base::VPCMPEQQ, *rsp, xmm0, xmm0);
            this->emit_add(16, rsp);
            this->emit_xor(eax, eax);
            this->emit(base::VPTEST, xmm0, xmm0);
            this->emit(base::SETZ, al);
            this->emit_push_raw(rax);
            break;
         }
         case simd_sub::i64x2_bitmask: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit(base::VMOVMSKPD, xmm0, rax);
            this->emit_push_raw(rax);
            break;
         }
         case simd_sub::i64x2_extend_low_i32x4_s: simd_v128_unop(base::VPMOVSXDQ); break;
         case simd_sub::i64x2_extend_high_i32x4_s: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm0, xmm0);
            this->emit(base::VPMOVSXDQ, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i64x2_extend_low_i32x4_u: simd_v128_unop(base::VPMOVZXDQ); break;
         case simd_sub::i64x2_extend_high_i32x4_u: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit(base::VPSRLDQ_c, typename base::imm8{8}, xmm0, xmm0);
            this->emit(base::VPMOVZXDQ, xmm0, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i64x2_shl: simd_v128_shiftop(base::VPSLLQ, 0x3f); break;
         case simd_sub::i64x2_shr_s: {
            // (x >> n) | ((0 > x) << (64 - n))
            this->emit_pop_raw(rax);
            this->emit_bytes(0x83, 0xe0, 0x3f);  // and $0x3f, %eax
            this->emit_mov(64, ecx);
            this->emit_sub(eax, ecx);
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_vmovd(eax, xmm1);
            this->emit_vmovd(ecx, xmm3);
            this->emit_const_zero(xmm2);
            this->emit(base::VPCMPGTQ, xmm0, xmm2, xmm2);
            this->emit(base::VPSLLQ, xmm3, xmm2, xmm2);
            this->emit(base::VPSRLQ, xmm1, xmm0, xmm0);
            this->emit(base::VPOR, xmm0, xmm2, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i64x2_shr_u: simd_v128_shiftop(base::VPSRLQ, 0x3f); break;
         case simd_sub::i64x2_add: simd_v128_binop_r(base::VPADDQ); break;
         case simd_sub::i64x2_sub: simd_v128_binop(base::VPSUBQ); break;
         case simd_sub::i64x2_mul: {
            this->emit_vmovdqu(*rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit_vmovdqu(*rsp, xmm1);
            this->emit(base::VPMULUDQ, xmm0, xmm1, xmm2);
            this->emit(base::VPSHUFD, typename base::imm8{0xb1}, xmm0, xmm0);
            this->emit(base::VPMULLD, xmm0, xmm1, xmm0);
            this->emit(base::VPHADDD, xmm0, xmm0, xmm0);
            this->emit_const_zero(xmm1);
            this->emit(base::VPUNPCKLDQ, xmm0, xmm1, xmm0);
            this->emit(base::VPADDQ, xmm0, xmm2, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i64x2_extmul_low_i32x4_s: {
            this->emit(base::VPSHUFD, typename base::imm8{0x10}, *rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit(base::VPSHUFD, typename base::imm8{0x10}, *rsp, xmm1);
            this->emit(base::VPMULDQ, xmm0, xmm1, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i64x2_extmul_high_i32x4_s: {
            this->emit(base::VPSHUFD, typename base::imm8{0x32}, *rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit(base::VPSHUFD, typename base::imm8{0x32}, *rsp, xmm1);
            this->emit(base::VPMULDQ, xmm0, xmm1, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i64x2_extmul_low_i32x4_u: {
            this->emit(base::VPSHUFD, typename base::imm8{0x10}, *rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit(base::VPSHUFD, typename base::imm8{0x10}, *rsp, xmm1);
            this->emit(base::VPMULUDQ, xmm0, xmm1, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }
         case simd_sub::i64x2_extmul_high_i32x4_u: {
            this->emit(base::VPSHUFD, typename base::imm8{0x32}, *rsp, xmm0);
            this->emit_add(16, rsp);
            this->emit(base::VPSHUFD, typename base::imm8{0x32}, *rsp, xmm1);
            this->emit(base::VPMULUDQ, xmm0, xmm1, xmm0);
            this->emit_vmovdqu(xmm0, *rsp);
            break;
         }

         // ── f32x4 arithmetic (softfloat) ──
         case simd_sub::f32x4_ceil: simd_v128_unop_softfloat(&_eosio_f32x4_ceil); break;
         case simd_sub::f32x4_floor: simd_v128_unop_softfloat(&_eosio_f32x4_floor); break;
         case simd_sub::f32x4_trunc: simd_v128_unop_softfloat(&_eosio_f32x4_trunc); break;
         case simd_sub::f32x4_nearest: simd_v128_unop_softfloat(&_eosio_f32x4_nearest); break;
         case simd_sub::f32x4_abs: simd_v128_unop_softfloat(&_eosio_f32x4_abs); break;
         case simd_sub::f32x4_neg: simd_v128_unop_softfloat(&_eosio_f32x4_neg); break;
         case simd_sub::f32x4_sqrt: simd_v128_unop_softfloat(&_eosio_f32x4_sqrt); break;
         case simd_sub::f32x4_add: simd_v128_binop_softfloat(&_eosio_f32x4_add); break;
         case simd_sub::f32x4_sub: simd_v128_binop_softfloat(&_eosio_f32x4_sub); break;
         case simd_sub::f32x4_mul: simd_v128_binop_softfloat(&_eosio_f32x4_mul); break;
         case simd_sub::f32x4_div: simd_v128_binop_softfloat(&_eosio_f32x4_div); break;
         case simd_sub::f32x4_min: simd_v128_binop_softfloat(&_eosio_f32x4_min); break;
         case simd_sub::f32x4_max: simd_v128_binop_softfloat(&_eosio_f32x4_max); break;
         case simd_sub::f32x4_pmin: simd_v128_binop_softfloat(&_eosio_f32x4_pmin); break;
         case simd_sub::f32x4_pmax: simd_v128_binop_softfloat(&_eosio_f32x4_pmax); break;

         // ── f64x2 arithmetic (softfloat) ──
         case simd_sub::f64x2_ceil: simd_v128_unop_softfloat(&_eosio_f64x2_ceil); break;
         case simd_sub::f64x2_floor: simd_v128_unop_softfloat(&_eosio_f64x2_floor); break;
         case simd_sub::f64x2_trunc: simd_v128_unop_softfloat(&_eosio_f64x2_trunc); break;
         case simd_sub::f64x2_nearest: simd_v128_unop_softfloat(&_eosio_f64x2_nearest); break;
         case simd_sub::f64x2_abs: simd_v128_unop_softfloat(&_eosio_f64x2_abs); break;
         case simd_sub::f64x2_neg: simd_v128_unop_softfloat(&_eosio_f64x2_neg); break;
         case simd_sub::f64x2_sqrt: simd_v128_unop_softfloat(&_eosio_f64x2_sqrt); break;
         case simd_sub::f64x2_add: simd_v128_binop_softfloat(&_eosio_f64x2_add); break;
         case simd_sub::f64x2_sub: simd_v128_binop_softfloat(&_eosio_f64x2_sub); break;
         case simd_sub::f64x2_mul: simd_v128_binop_softfloat(&_eosio_f64x2_mul); break;
         case simd_sub::f64x2_div: simd_v128_binop_softfloat(&_eosio_f64x2_div); break;
         case simd_sub::f64x2_min: simd_v128_binop_softfloat(&_eosio_f64x2_min); break;
         case simd_sub::f64x2_max: simd_v128_binop_softfloat(&_eosio_f64x2_max); break;
         case simd_sub::f64x2_pmin: simd_v128_binop_softfloat(&_eosio_f64x2_pmin); break;
         case simd_sub::f64x2_pmax: simd_v128_binop_softfloat(&_eosio_f64x2_pmax); break;

         // ── Conversions (softfloat) ──
         case simd_sub::i32x4_trunc_sat_f32x4_s: simd_v128_unop_softfloat(&_eosio_i32x4_trunc_sat_f32x4_s); break;
         case simd_sub::i32x4_trunc_sat_f32x4_u: simd_v128_unop_softfloat(&_eosio_i32x4_trunc_sat_f32x4_u); break;
         case simd_sub::f32x4_convert_i32x4_s: simd_v128_unop_softfloat(&_eosio_f32x4_convert_i32x4_s); break;
         case simd_sub::f32x4_convert_i32x4_u: simd_v128_unop_softfloat(&_eosio_f32x4_convert_i32x4_u); break;
         case simd_sub::i32x4_trunc_sat_f64x2_s_zero: simd_v128_unop_softfloat(&_eosio_i32x4_trunc_sat_f64x2_s_zero); break;
         case simd_sub::i32x4_trunc_sat_f64x2_u_zero: simd_v128_unop_softfloat(&_eosio_i32x4_trunc_sat_f64x2_u_zero); break;
         case simd_sub::f64x2_convert_low_i32x4_s: simd_v128_unop_softfloat(&_eosio_f64x2_convert_low_i32x4_s); break;
         case simd_sub::f64x2_convert_low_i32x4_u: simd_v128_unop_softfloat(&_eosio_f64x2_convert_low_i32x4_u); break;
         case simd_sub::f32x4_demote_f64x2_zero: simd_v128_unop_softfloat(&_eosio_f32x4_demote_f64x2_zero); break;
         case simd_sub::f64x2_promote_low_f32x4: simd_v128_unop_softfloat(&_eosio_f64x2_promote_low_f32x4); break;

         default:
            break;
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
            if (ft.return_type == types::v128) {
               // v128 return: callee put result in xmm0, push 16 bytes
               this->emit_sub(16, rsp);
               this->emit_vmovdqu(xmm0, *rsp);
            } else {
               this->emit_push_raw(rax);
            }
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
      // SSA info from optimizer (for const-operand fusion)
      uint32_t* _func_def_inst = nullptr;
      uint16_t* _func_use_count = nullptr;
      ir_inst*  _func_insts = nullptr;
      uint32_t  _func_inst_count = 0;
   };

}} // namespace eosio::vm
