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
#include <variant>
#include <vector>

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
      jit_codegen(growable_allocator& alloc, module& mod)
         : _allocator(alloc), _mod(mod) {
         _code_segment_base = _allocator.start_code();
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

         start_function(code, func.func_index);

         // Emit function prologue
         emit_function_prologue(func);

         // Emit each IR instruction as stack-machine code (naive: all spilled)
         for (uint32_t i = 0; i < func.inst_count; ++i) {
            emit_ir_inst(func, func.insts[i], i);
         }

         // Record code offset
         _allocator.reclaim(code, buf + est_size - code);
         body.jit_code_offset = code_start - static_cast<unsigned char*>(_code_segment_base);
      }

      void finalize_code() {
         _allocator.end_code<true>(_code_segment_base);

         // Patch element table entries (same as machine_code_writer destructor)
         auto num_functions = _mod.get_functions_total();
         if (num_functions <= _function_relocations.size()) {
            for (auto& elem : _mod.elements) {
               for (auto& entry : elem.elems) {
                  void* addr = call_indirect_handler;
                  if (entry.index < num_functions) {
                     assert(entry.index < _function_relocations.size());
                     if (auto reloc = std::get_if<void*>(&_function_relocations[entry.index])) {
                        addr = *reloc;
                     }
                  }
                  std::size_t offset = static_cast<char*>(addr) - static_cast<char*>(_code_segment_base);
                  entry.code_ptr = _mod.allocator._code_base + offset;
               }
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

         // ── Nop / arg / block / loop / end / if / else ──
         case ir_op::nop:
         case ir_op::arg:  // args are handled by call emission
         case ir_op::block:
         case ir_op::loop:
         case ir_op::end:
         case ir_op::if_:
         case ir_op::else_:
            break;

         // ── Control flow (branches) ──
         case ir_op::br:
         case ir_op::br_if:
         case ir_op::br_table:
            // TODO: Proper branch emission requires block address mapping
            // For now, emit nothing (will be filled in during Phase 4)
            break;

         // ── Calls ──
         case ir_op::call:
         case ir_op::call_indirect:
            // TODO: Full call emission
            break;

         // ── Local/global access ──
         case ir_op::local_get:
         case ir_op::local_set:
         case ir_op::local_tee:
         case ir_op::global_get:
         case ir_op::global_set:
            // TODO: Frame-relative access
            break;

         // ── Memory operations ──
         case ir_op::i32_load:
         case ir_op::i64_load:
         case ir_op::f32_load:
         case ir_op::f64_load:
         case ir_op::i32_store:
         case ir_op::i64_store:
         case ir_op::f32_store:
         case ir_op::f64_store:
         case ir_op::memory_size:
         case ir_op::memory_grow:
            // TODO: Memory access emission
            break;

         // ── Select/drop ──
         case ir_op::select:
         case ir_op::drop:
            break;

         default:
            // All other opcodes handled as TODO stubs for Phase 3
            break;
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
      void register_call(void* ptr, uint32_t funcnum) {
         auto& vec = _function_relocations;
         if (funcnum >= vec.size()) vec.resize(funcnum + 1);
         if (void** addr = std::get_if<void*>(&vec[funcnum])) {
            base::fix_branch(ptr, *addr);
         } else {
            std::get<std::vector<void*>>(vec[funcnum]).push_back(ptr);
         }
      }

      void start_function(void* func_start, uint32_t funcnum) {
         auto& vec = _function_relocations;
         if (funcnum >= vec.size()) vec.resize(funcnum + 1);
         for (void* branch : std::get<std::vector<void*>>(vec[funcnum])) {
            base::fix_branch(branch, func_start);
         }
         vec[funcnum] = func_start;
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
      std::vector<std::variant<std::vector<void*>, void*>> _function_relocations;
   };

}} // namespace eosio::vm
