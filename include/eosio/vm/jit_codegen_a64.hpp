#pragma once

#ifdef __aarch64__

// Pass 2 of the two-pass optimizing JIT (jit2) — AArch64 backend.
// Converts IR instructions to ARM64 machine code.
//
// Register conventions (AAPCS64):
//   X0, X1    = temporaries (spill loads, like rax/rcx on x86)
//   X2-X15    = caller-saved, available for register allocation (14 regs)
//   X16       = scratch for large immediates
//   X19       = context pointer (callee-saved)
//   X20       = linear memory base (callee-saved)
//   X21       = call depth counter (callee-saved)
//   X22-X28   = callee-saved, available for register allocation (7 regs)
//   X29       = frame pointer (FP)
//   X30       = link register (LR)
//   SP        = stack pointer (must be 16-byte aligned at calls)

#include <eosio/vm/allocator.hpp>
#include <eosio/vm/exceptions.hpp>
#include <eosio/vm/jit_ir.hpp>
#include <eosio/vm/jit_regalloc.hpp>
#include <eosio/vm/softfloat.hpp>
#include <eosio/vm/types.hpp>
#include <eosio/vm/utils.hpp>
#include <eosio/vm/signals.hpp>

#include <cassert>
#include <cstdint>
#include <cstring>

namespace eosio { namespace vm {

   template<typename Context, bool StackLimitIsBytes>
   class jit_codegen_a64 {
    public:
      // Register numbers
      static constexpr uint32_t X0  = 0,  X1  = 1,  X2  = 2,  X3  = 3;
      static constexpr uint32_t X4  = 4,  X5  = 5,  X6  = 6,  X7  = 7;
      static constexpr uint32_t X8  = 8,  X9  = 9,  X10 = 10, X11 = 11;
      static constexpr uint32_t X12 = 12, X13 = 13, X14 = 14, X15 = 15;
      static constexpr uint32_t X16 = 16, X17 = 17;
      static constexpr uint32_t X19 = 19, X20 = 20, X21 = 21, X22 = 22;
      static constexpr uint32_t X23 = 23, X24 = 24, X25 = 25;
      static constexpr uint32_t X29 = 29, X30 = 30;
      static constexpr uint32_t XZR = 31, SP = 31, FP = 29;

      // Condition codes
      static constexpr uint32_t COND_EQ = 0,  COND_NE = 1;
      static constexpr uint32_t COND_HS = 2,  COND_LO = 3;   // unsigned >= / <
      static constexpr uint32_t COND_HI = 8,  COND_LS = 9;   // unsigned > / <=
      static constexpr uint32_t COND_GE = 10, COND_LT = 11;  // signed >= / <
      static constexpr uint32_t COND_GT = 12, COND_LE = 13;   // signed > / <=

      static constexpr uint32_t invert_condition(uint32_t cond) { return cond ^ 1; }

      static constexpr bool use_softfloat =
#ifdef EOS_VM_SOFTFLOAT
         true;
#else
         false;
#endif

      jit_codegen_a64(growable_allocator& alloc, module& mod, void* code_segment_base = nullptr)
         : _allocator(alloc), _mod(mod) {
         init_relocations();
         _code_segment_base = code_segment_base ? code_segment_base : _allocator.start_code();
      }

      static constexpr int32_t call_depth_offset() {
         if constexpr (Context::async_backtrace()) return 16;
         else return 0;
      }

      // ──────── Entry point and error handlers ────────

      void emit_entry_and_error_handlers() {
         // AAPCS64 entry point
         auto* buf = _allocator.alloc<unsigned char>(512);
         code = buf;
         emit_aapcs64_interface();

         // Error handlers
         buf = _allocator.alloc<unsigned char>(256);
         code = buf;
         fpe_handler = emit_error_handler(&on_fp_error);
         call_indirect_handler = emit_error_handler(&on_call_indirect_error);
         type_error_handler = emit_error_handler(&on_type_error);
         stack_overflow_handler = emit_error_handler(&on_stack_overflow);
         memory_handler = emit_error_handler(&on_memory_error);

         // Host function stubs
         const uint32_t num_imported = _mod.get_imported_functions_size();
         if (num_imported > 0) {
            const std::size_t host_functions_size = 256 * num_imported;
            buf = _allocator.alloc<unsigned char>(host_functions_size);
            code = buf;
            for (uint32_t i = 0; i < num_imported; ++i) {
               start_function(code, i);
               emit_host_call(i);
            }
         }
      }

      // ──────── Compile one function from IR to ARM64 ────────

      void compile_function(ir_function& func, function_body& body) {
         const std::size_t est_size = static_cast<std::size_t>(func.inst_count) * 64 + 512;
         auto* buf = _allocator.alloc<unsigned char>(est_size);
         auto* code_start = buf;
         code = buf;

         _block_addrs = _allocator.alloc<void*>(func.block_count);
         _block_fixups = _allocator.alloc<block_fixup*>(func.block_count);
         _num_blocks = func.block_count;
         for (uint32_t i = 0; i < func.block_count; ++i) {
            _block_addrs[i] = nullptr;
            _block_fixups[i] = nullptr;
         }

         // Build vreg → physical register mapping
         _vreg_map = nullptr;
         _spill_map = nullptr;
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

         _func_def_inst = func.def_inst;
         _func_use_count = func.use_count;
         _func_insts = func.insts;
         _func_inst_count = func.inst_count;

         start_function(code, func.func_index + _mod.get_imported_functions_size());

         emit_function_prologue(func);

         for (uint32_t i = 0; i < func.inst_count; ++i) {
            emit_ir_inst_reg(func, func.insts[i], i);
         }

         emit_function_epilogue(func);

         body.jit_code_offset = code_start - static_cast<unsigned char*>(_code_segment_base);

         _block_addrs = nullptr;
         _block_fixups = nullptr;
         _num_blocks = 0;
         _if_fixup_top = 0;
         _in_br_table = false;
      }

      void finalize_code() {
         _allocator.end_code<true>(_code_segment_base);


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

      // ──────── Instruction emission ────────

      void emit32(uint32_t instr) {
         std::memcpy(code, &instr, 4);
         code += 4;
      }

      // ──────── Branch fixup ────────

      static void fix_branch(void* branch, void* target) {
         auto* branch_bytes = static_cast<uint8_t*>(branch);
         auto* target_bytes = static_cast<uint8_t*>(target);
         int64_t offset = (target_bytes - branch_bytes) / 4;

         uint32_t instr;
         std::memcpy(&instr, branch, 4);

         if ((instr & 0xFC000000) == 0x14000000 || (instr & 0xFC000000) == 0x94000000) {
            // B / BL: imm26
            EOS_VM_ASSERT(offset <= 0x1FFFFFF && offset >= -0x2000000, wasm_parse_exception, "branch out of range");
            instr = (instr & 0xFC000000) | (static_cast<uint32_t>(offset) & 0x3FFFFFF);
         } else if ((instr & 0xFF000010) == 0x54000000) {
            // B.cond: imm19
            EOS_VM_ASSERT(offset <= 0x3FFFF && offset >= -0x40000, wasm_parse_exception, "branch out of range");
            instr = (instr & 0xFF00001F) | ((static_cast<uint32_t>(offset) & 0x7FFFF) << 5);
         } else if ((instr & 0x7F000000) == 0x34000000 || (instr & 0x7F000000) == 0x35000000 ||
                    (instr & 0xFF000000) == 0xB4000000 || (instr & 0xFF000000) == 0xB5000000) {
            // CBZ/CBNZ (32/64-bit): imm19
            EOS_VM_ASSERT(offset <= 0x3FFFF && offset >= -0x40000, wasm_parse_exception, "branch out of range");
            instr = (instr & 0xFF00001F) | ((static_cast<uint32_t>(offset) & 0x7FFFF) << 5);
         } else {
            EOS_VM_ASSERT(false, wasm_parse_exception, "unknown branch instruction to fix");
         }

         std::memcpy(branch, &instr, 4);
      }

      // ──────── Immediate encoding helpers ────────

      void emit_mov_imm32(uint32_t rd, uint32_t value) {
         if (value == 0) {
            emit32(0x2A1F03E0 | rd); // MOV Wd, WZR
            return;
         }
         uint16_t lo = value & 0xFFFF;
         uint16_t hi = (value >> 16) & 0xFFFF;
         if (hi == 0) {
            emit32(0x52800000 | (static_cast<uint32_t>(lo) << 5) | rd); // MOVZ Wd, #lo
         } else if (lo == 0) {
            emit32(0x52A00000 | (static_cast<uint32_t>(hi) << 5) | rd); // MOVZ Wd, #hi, LSL #16
         } else if ((value & 0xFFFF0000) == 0xFFFF0000) {
            emit32(0x12800000 | (static_cast<uint32_t>(static_cast<uint16_t>(~lo)) << 5) | rd); // MOVN
         } else {
            emit32(0x52800000 | (static_cast<uint32_t>(lo) << 5) | rd);
            emit32(0x72A00000 | (static_cast<uint32_t>(hi) << 5) | rd); // MOVK Wd, #hi, LSL #16
         }
      }

      void emit_mov_imm64(uint32_t rd, uint64_t value) {
         if (value == 0) {
            emit32(0xAA1F03E0 | rd); // MOV Xd, XZR
            return;
         }
         if (value <= 0xFFFFFFFF) {
            emit_mov_imm32(rd, static_cast<uint32_t>(value));
            return;
         }
         uint16_t chunks[4] = {
            static_cast<uint16_t>(value),
            static_cast<uint16_t>(value >> 16),
            static_cast<uint16_t>(value >> 32),
            static_cast<uint16_t>(value >> 48)
         };
         bool first = true;
         for (int i = 0; i < 4; ++i) {
            if (chunks[i] != 0) {
               if (first) {
                  emit32(0xD2800000 | (static_cast<uint32_t>(i) << 21) | (static_cast<uint32_t>(chunks[i]) << 5) | rd);
                  first = false;
               } else {
                  emit32(0xF2800000 | (static_cast<uint32_t>(i) << 21) | (static_cast<uint32_t>(chunks[i]) << 5) | rd);
               }
            }
         }
      }

      // ADD Xd, Xn, #imm (unsigned)
      void emit_add_imm(uint32_t rd, uint32_t rn, uint32_t imm) {
         if (imm <= 4095) {
            emit32(0x91000000 | (imm << 10) | (rn << 5) | rd);
         } else if ((imm & 0xFFF) == 0 && (imm >> 12) <= 4095) {
            emit32(0x91400000 | ((imm >> 12) << 10) | (rn << 5) | rd);
         } else {
            emit_mov_imm64(X16, imm);
            emit32(0x8B100000 | (rn << 5) | rd); // ADD Xd, Xn, X16
         }
      }

      // SUB Xd, Xn, #imm
      void emit_sub_imm(uint32_t rd, uint32_t rn, uint32_t imm) {
         if (imm <= 4095) {
            emit32(0xD1000000 | (imm << 10) | (rn << 5) | rd);
         } else if ((imm & 0xFFF) == 0 && (imm >> 12) <= 4095) {
            emit32(0xD1400000 | ((imm >> 12) << 10) | (rn << 5) | rd);
         } else {
            emit_mov_imm64(X16, imm);
            emit32(0xCB100000 | (rn << 5) | rd); // SUB Xd, Xn, X16
         }
      }

      // ADD with signed immediate
      void emit_add_signed_imm(uint32_t rd, uint32_t rn, int32_t imm) {
         if (imm >= 0) emit_add_imm(rd, rn, static_cast<uint32_t>(imm));
         else emit_sub_imm(rd, rn, static_cast<uint32_t>(-static_cast<int64_t>(imm)));
      }

      // CMP Wn, #imm (32-bit)
      void emit_cmp_imm32(uint32_t rn, uint32_t value) {
         if (value <= 4095) {
            emit32(0x7100001F | (value << 10) | (rn << 5)); // SUBS WZR, Wn, #value
         } else {
            emit_mov_imm32(X16, value);
            emit32(0x6B10001F | (rn << 5)); // CMP Wn, W16
         }
      }

      // CMP Xn, #imm (64-bit)
      void emit_cmp_imm64(uint32_t rn, uint32_t value) {
         if (value <= 4095) {
            emit32(0xF100001F | (value << 10) | (rn << 5)); // SUBS XZR, Xn, #value
         } else {
            emit_mov_imm32(X16, value);
            emit32(0xEB10001F | (rn << 5)); // CMP Xn, X16
         }
      }

      // CMP Wn, Wm (32-bit register)
      void emit_cmp_reg32(uint32_t rn, uint32_t rm) {
         emit32(0x6B00001F | (rm << 16) | (rn << 5)); // SUBS WZR, Wn, Wm
      }

      // CMP Xn, Xm (64-bit register)
      void emit_cmp_reg64(uint32_t rn, uint32_t rm) {
         emit32(0xEB00001F | (rm << 16) | (rn << 5)); // SUBS XZR, Xn, Xm
      }

      // CSET Xd, cond
      void emit_cset(uint32_t rd, uint32_t cond) {
         uint32_t inv = invert_condition(cond);
         emit32(0x9A9F07E0 | (inv << 12) | rd); // CSINC Xd, XZR, XZR, inv(cond)
      }

      // MOV Xd, Xm
      void emit_mov_reg(uint32_t rd, uint32_t rm) {
         if (rd != rm)
            emit32(0xAA0003E0 | (rm << 16) | rd); // ORR Xd, XZR, Xm
      }

      // MOV Wd, Wm (32-bit, zero-extends)
      void emit_mov_reg32(uint32_t rd, uint32_t rm) {
         if (rd != rm)
            emit32(0x2A0003E0 | (rm << 16) | rd); // ORR Wd, WZR, Wm
      }

      // LDR Xt, [Xn, #offset] (signed offset, using X16 as scratch if needed)
      void emit_ldr_offset(uint32_t rt, uint32_t rn, int32_t offset) {
         if (offset >= -256 && offset < 256) {
            // LDUR Xt, [Xn, #offset]
            emit32(0xF8400000 | ((static_cast<uint32_t>(offset) & 0x1FF) << 12) | (rn << 5) | rt);
         } else if (offset >= 0 && (offset % 8) == 0 && (offset / 8) <= 4095) {
            // LDR Xt, [Xn, #offset] (unsigned scaled)
            emit32(0xF9400000 | ((offset / 8) << 10) | (rn << 5) | rt);
         } else {
            emit_mov_imm64(X16, static_cast<uint64_t>(static_cast<int64_t>(offset)));
            emit32(0xF8706800 | (X16 << 16) | (rn << 5) | rt); // LDR Xt, [Xn, X16]
         }
      }

      // STR Xt, [Xn, #offset]
      void emit_str_offset(uint32_t rt, uint32_t rn, int32_t offset) {
         if (offset >= -256 && offset < 256) {
            emit32(0xF8000000 | ((static_cast<uint32_t>(offset) & 0x1FF) << 12) | (rn << 5) | rt);
         } else if (offset >= 0 && (offset % 8) == 0 && (offset / 8) <= 4095) {
            emit32(0xF9000000 | ((offset / 8) << 10) | (rn << 5) | rt);
         } else {
            emit_mov_imm64(X16, static_cast<uint64_t>(static_cast<int64_t>(offset)));
            emit32(0xF8306800 | (X16 << 16) | (rn << 5) | rt); // STR Xt, [Xn, X16]
         }
      }

      // LDR Xt, [FP, #offset]
      void emit_ldr_fp(uint32_t rt, int32_t offset) { emit_ldr_offset(rt, FP, offset); }
      // STR Xt, [FP, #offset]
      void emit_str_fp(uint32_t rt, int32_t offset) { emit_str_offset(rt, FP, offset); }

      // Push/pop via SP (16-byte aligned)
      void emit_push(uint32_t rt) {
         emit32(0xF81F0FE0 | rt); // STR Xt, [SP, #-16]!
      }
      void emit_pop(uint32_t rt) {
         emit32(0xF84107E0 | rt); // LDR Xt, [SP], #16
      }

      // ──────── AAPCS64 interface ────────

      void emit_aapcs64_interface() {
         // Args: X0=context, X1=memory, X2=data, X3=fun, X4=stack, X5=count, X6=vector_result
         // Save callee-saved
         emit32(0xA9BF7BFD); // STP X29, X30, [SP, #-16]!
         emit32(0x910003FD); // MOV X29, SP
         emit32(0xA9BF53F3); // STP X19, X20, [SP, #-16]!
         emit32(0xA9BF5BF5); // STP X21, X22, [SP, #-16]!
         emit32(0xA9BF63F7); // STP X23, X24, [SP, #-16]!
         emit32(0xA9BF6BF9); // STP X25, X26, [SP, #-16]!

         // Save vector_result flag
         emit32(0xAA0603F6); // MOV X22, X6

         // Set up context and memory base
         emit32(0xAA0003F3); // MOV X19, X0
         emit32(0xAA0103F4); // MOV X20, X1

         // Optional stack switch
         emit32(0xB4000044); // CBZ X4, +8
         emit32(0x91000000 | (X4 << 5) | SP); // MOV SP, X4

         // Save SP before arg push
         emit32(0x910003F7); // MOV X23, SP

         // Push args from data array
         void* skip_push = code;
         emit32(0xB4000000 | X5); // CBZ X5, skip (patched)

         // Allocate stack: SP -= count * 16
         emit32(0xCB2573FF); // SUB SP, SP, X5, UXTX #4

         // Copy data in reverse order
         emit32(0x8B050C42); // ADD X2, X2, X5, LSL #3
         emit32(0x910003E9); // MOV X9, SP
         emit32(0xAA0503E8); // MOV X8, X5
         void* loop_top = code;
         emit32(0xF85F8C4A); // LDR X10, [X2, #-8]!
         emit32(0xF801052A); // STR X10, [X9], #16
         emit32(0xF1000508); // SUBS X8, X8, #1
         {
            int32_t off = static_cast<int32_t>((static_cast<uint8_t*>(loop_top) - code)) / 4;
            emit32(0x54000001 | ((static_cast<uint32_t>(off) & 0x7FFFF) << 5)); // B.NE loop
         }
         fix_branch(skip_push, code);

         // Load call depth
         if constexpr (Context::async_backtrace()) {
            emit32(0xB9401275); // LDR W21, [X19, #16]
         } else {
            emit32(0xB9400275); // LDR W21, [X19]
         }

         if constexpr (Context::async_backtrace()) {
            emit32(0xF9000673); // STR X19, [X19, #8]
         }

         // BLR X3 (call WASM function)
         emit32(0xD63F0060);

         // Restore SP
         emit32(0x910002FF); // MOV SP, X23

         if constexpr (Context::async_backtrace()) {
            emit32(0xF900067F); // STR XZR, [X19, #8]
         }

         // Restore callee-saved via FP
         emit32(0xD10103A0 | SP); // SUB SP, X29, #64

         emit32(0xA8C16BF9); // LDP X25, X26, [SP], #16
         emit32(0xA8C163F7); // LDP X23, X24, [SP], #16
         emit32(0xA8C15BF5); // LDP X21, X22, [SP], #16
         emit32(0xA8C153F3); // LDP X19, X20, [SP], #16
         emit32(0xA8C17BFD); // LDP X29, X30, [SP], #16
         emit32(0xD65F03C0); // RET
      }

      // ──────── Error handlers ────────

      void* emit_error_handler(void (*handler)()) {
         void* result = code;
         // Align SP
         emit32(0x910003E8); // MOV X8, SP
         emit32(0x927CED08); // AND X8, X8, #~0xF
         emit32(0x9100011F); // MOV SP, X8
         emit_mov_imm64(X8, reinterpret_cast<uint64_t>(handler));
         emit32(0xD63F0100); // BLR X8
         return result;
      }

      void emit_branch_to_handler(uint32_t cond, void* handler) {
         int64_t offset = (static_cast<uint8_t*>(handler) - code) / 4;
         if (offset >= -0x40000 && offset < 0x40000) {
            emit32(0x54000000 | ((static_cast<uint32_t>(offset) & 0x7FFFF) << 5) | cond);
         } else {
            // Invert condition, skip trampoline
            void* skip = code;
            emit32(0x54000000 | invert_condition(cond)); // B.inv skip
            emit_mov_imm64(X8, reinterpret_cast<uint64_t>(handler));
            emit32(0xD61F0100); // BR X8
            fix_branch(skip, code);
         }
      }

      // ──────── Host call stubs ────────

      void emit_host_call(uint32_t funcnum) {
         // Save FP/LR
         emit32(0xA9BF7BFD); // STP X29, X30, [SP, #-16]!
         emit32(0x910003FD); // MOV X29, SP

         if constexpr (Context::async_backtrace()) {
            emit32(0xF9000273); // STR X19, [X19]
         }

         // Save X19/X20
         emit32(0xA9BF53F3); // STP X19, X20, [SP, #-16]!

         const auto& ft = _mod.get_function_type(funcnum);
         uint32_t num_params = ft.param_types.size();

         // Repack args from 16-byte stride to 8-byte stride buffer
         uint32_t buf_size = num_params > 0 ? ((num_params * 8 + 15) / 16) * 16 : 0;
         if (buf_size > 0) {
            emit_sub_imm(SP, SP, buf_size);
         }

         emit32(0x910003E1); // MOV X1, SP (buffer pointer)

         uint32_t extra = buf_size + 32; // buf + saved X19/X20 + saved FP/LR
         for (uint32_t i = 0; i < num_params; ++i) {
            uint32_t src_off = extra + i * 16;
            uint32_t dst_off = i * 8;
            // LDR X8, [SP, #src_off]
            if (src_off % 8 == 0 && src_off / 8 <= 4095) {
               emit32(0xF9400000 | ((src_off / 8) << 10) | (SP << 5) | X8);
            } else {
               emit_mov_imm32(X8, src_off);
               emit32(0xF8686BE8); // LDR X8, [SP, X8]
            }
            // STR X8, [X1, #dst_off]
            if (dst_off % 8 == 0 && dst_off / 8 <= 4095) {
               emit32(0xF9000000 | ((dst_off / 8) << 10) | (X1 << 5) | X8);
            } else {
               emit_mov_imm32(X16, dst_off);
               emit32(0xF8306828 | (X16 << 16)); // STR X8, [X1, X16]
            }
         }

         emit_mov_imm32(X2, funcnum);       // W2 = host function index
         emit32(0x2A1503E3);                // MOV W3, W21 (remaining call depth)
         emit32(0xAA1303E0);                // MOV X0, X19 (context)

         emit_mov_imm64(X8, reinterpret_cast<uint64_t>(&call_host_function));
         emit32(0xD63F0100); // BLR X8

         if (buf_size > 0) {
            emit_add_imm(SP, SP, buf_size);
         }

         emit32(0xA8C153F3); // LDP X19, X20, [SP], #16

         if constexpr (Context::async_backtrace()) {
            emit32(0xF900027F); // STR XZR, [X19]
         }

         emit32(0xA8C17BFD); // LDP X29, X30, [SP], #16
         emit32(0xD65F03C0); // RET
      }

      // ──────── Call depth checking ────────

      void emit_call_depth_dec() {
         if constexpr (!StackLimitIsBytes) {
            // SUBS W21, W21, #1
            emit32(0x71000400 | (1 << 10) | (X21 << 5) | X21);
            emit_branch_to_handler(COND_EQ, stack_overflow_handler);
         }
      }

      void emit_call_depth_inc() {
         if constexpr (!StackLimitIsBytes) {
            // ADD W21, W21, #1
            emit32(0x11000400 | (1 << 10) | (X21 << 5) | X21);
         }
      }

      // ──────── Function prologue/epilogue ────────

      void emit_function_prologue(ir_function& func) {
         // STP X29, X30, [SP, #-16]!
         emit32(0xA9BF7BFD);
         // MOV X29, SP
         emit32(0x910003FD);

         uint32_t body_locals = func.num_locals - func.num_params;
         _body_locals = body_locals;

         _callee_saved_used = func.callee_saved_used;
         _callee_saved_count = __builtin_popcount(_callee_saved_used);

         // Total frame slots: body locals + spill slots + callee-saved saves
         uint32_t total_slots = body_locals + _num_spill_slots + _callee_saved_count;
         // Round up to even for 16-byte alignment
         uint32_t aligned_slots = (total_slots + 1) & ~1u;

         if (aligned_slots > 0) {
            uint32_t frame_size = aligned_slots * 8;
            emit_sub_imm(SP, SP, frame_size);
            // Zero-initialize locals + spill slots
            uint32_t zero_slots = body_locals + _num_spill_slots;
            for (uint32_t i = 0; i < zero_slots; i += 2) {
               int32_t off = i * 8;
               if (i + 1 < zero_slots) {
                  // STP XZR, XZR, [SP, #off]
                  if (off % 8 == 0 && off / 8 <= 63) {
                     emit32(0xA9000000 | ((off / 8) << 15) | (XZR << 10) | (SP << 5) | XZR); // STP XZR, XZR, [SP, #off]
                  } else {
                     emit_str_offset(XZR, SP, off);
                     emit_str_offset(XZR, SP, off + 8);
                  }
               } else {
                  emit_str_offset(XZR, SP, off);
               }
            }
         }

         // Save callee-saved registers used by regalloc
         if (_use_regalloc && _callee_saved_used) {
            int32_t save_offset = (body_locals + _num_spill_slots) * 8;
            for (int i = 0; i < 7; ++i) {
               if (_callee_saved_used & (1 << i)) {
                  uint32_t reg = callee_saved_reg(i);
                  emit_str_offset(reg, SP, save_offset);
                  save_offset += 8;
               }
            }
         }
      }

      void emit_function_epilogue(ir_function& func) {
         if (_use_regalloc) {
            if (func.type->return_count != 0 && func.vstack_top > 0) {
               uint32_t result_vreg = func.vstack[func.vstack_top - 1];
               load_vreg_to(X0, result_vreg);
            }
         } else {
            if (func.type->return_count != 0) {
               emit_pop(X0);
            }
         }

         // Restore callee-saved registers
         if (_use_regalloc && _callee_saved_used) {
            int32_t save_offset = (_body_locals + _num_spill_slots) * 8;
            for (int i = 0; i < 7; ++i) {
               if (_callee_saved_used & (1 << i)) {
                  uint32_t reg = callee_saved_reg(i);
                  emit_ldr_offset(reg, SP, save_offset);
                  save_offset += 8;
               }
            }
         }

         // MOV SP, X29
         emit32(0x91000000 | (FP << 5) | SP);
         // LDP X29, X30, [SP], #16
         emit32(0xA8C17BFD);
         // RET
         emit32(0xD65F03C0);
      }

      // ──────── Register allocation helpers ────────

      static constexpr uint32_t phys_to_reg(int8_t pr) {
         constexpr uint32_t map[] = {
            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  // caller-saved (14)
            22, 23, 24, 25, 26, 27, 28                          // callee-saved (7)
         };
         return map[pr];
      }

      static constexpr uint32_t callee_saved_reg(int idx) {
         constexpr uint32_t regs[] = { 22, 23, 24, 25, 26, 27, 28 };
         return regs[idx];
      }

      int8_t get_phys(uint32_t vreg) const {
         if (!_vreg_map || vreg >= _num_vregs) return -1;
         return _vreg_map[vreg];
      }

      void load_vreg_to(uint32_t rd, uint32_t vreg) {
         if (vreg == ir_vreg_none) return;
         int8_t pr = get_phys(vreg);
         if (pr >= 0) {
            emit_mov_reg(rd, phys_to_reg(pr));
         } else if (_spill_map && vreg < _num_vregs && _spill_map[vreg] >= 0) {
            emit_ldr_offset(rd, SP, get_spill_offset(_spill_map[vreg]));
         } else {
            // vreg has no phys reg and no spill slot — should not happen
         }
      }

      void store_from_to_vreg(uint32_t rs, uint32_t vreg) {
         if (vreg == ir_vreg_none) return;
         int8_t pr = get_phys(vreg);
         if (pr >= 0) {
            emit_mov_reg(phys_to_reg(pr), rs);
         } else if (_spill_map && vreg < _num_vregs && _spill_map[vreg] >= 0) {
            emit_str_offset(rs, SP, get_spill_offset(_spill_map[vreg]));
         }
      }

      // Convenience: load into X0/X1 (temps)
      void load_vreg_x0(uint32_t vreg) { load_vreg_to(X0, vreg); }
      void load_vreg_x1(uint32_t vreg) { load_vreg_to(X1, vreg); }
      void store_x0_vreg(uint32_t vreg) { store_from_to_vreg(X0, vreg); }

      int32_t get_spill_offset(int16_t slot) const {
         return static_cast<int32_t>((_body_locals + static_cast<uint32_t>(slot)) * 8);
      }

      // Frame offset for locals: params above FP, locals below
      int32_t get_frame_offset(const ir_function& func, uint32_t local_idx) {
         const func_type* ft = func.type;
         if (local_idx < ft->param_types.size()) {
            // Params are above FP, pushed with 16-byte stride (ARM64 SP alignment)
            int32_t offset = 16; // skip saved FP/LR
            for (uint32_t i = ft->param_types.size(); i-- > 0; ) {
               if (i == local_idx) return offset;
               offset += (ft->param_types[i] == types::v128) ? 32 : 16;
            }
            return offset;
         } else {
            // Locals are in the SP-relative frame area
            uint32_t li = local_idx - ft->param_types.size();
            const auto& locals = _mod.code[func.func_index].locals;
            int32_t offset = 0;
            uint32_t count = 0;
            for (uint32_t g = 0; g < locals.size(); ++g) {
               uint8_t size = (locals[g].type == types::v128) ? 16 : 8;
               if (li < count + locals[g].count) {
                  offset += static_cast<int32_t>((li - count) * size);
                  return offset;
               }
               count += locals[g].count;
               offset += static_cast<int32_t>(locals[g].count * size);
            }
            return offset;
         }
      }

      // Load a local variable: params via FP, locals via SP
      void emit_local_load(const ir_function& func, uint32_t local_idx, uint32_t rd) {
         const func_type* ft = func.type;
         if (local_idx < ft->param_types.size()) {
            emit_ldr_offset(rd, FP, get_frame_offset(func, local_idx));
         } else {
            emit_ldr_offset(rd, SP, get_frame_offset(func, local_idx));
         }
      }

      void emit_local_store(const ir_function& func, uint32_t local_idx, uint32_t rs) {
         const func_type* ft = func.type;
         if (local_idx < ft->param_types.size()) {
            emit_str_offset(rs, FP, get_frame_offset(func, local_idx));
         } else {
            emit_str_offset(rs, SP, get_frame_offset(func, local_idx));
         }
      }

      // ──────── Global access ────────

      void emit_global_load(uint32_t gi, uint32_t rd) {
         auto offset = _mod.get_global_offset(gi);
         emit_ldr_offset(X16, X20, wasm_allocator::globals_end() - 8);
         emit_add_signed_imm(X16, X16, static_cast<int32_t>(offset));
         emit_ldr_offset(rd, X16, 0);
      }

      void emit_global_store(uint32_t gi, uint32_t rs) {
         auto offset = _mod.get_global_offset(gi);
         emit_ldr_offset(X16, X20, wasm_allocator::globals_end() - 8);
         emit_add_signed_imm(X16, X16, static_cast<int32_t>(offset));
         emit_str_offset(rs, X16, 0);
      }

      // ──────── Memory access helpers ────────

      // Compute native address: X0 = X20 + wasm_addr + offset
      // wasm_addr in rd, static offset added
      void emit_effective_addr(uint32_t rd, uint32_t addr_reg, uint32_t uoffset) {
         if (uoffset & 0x80000000u) {
            emit_mov_imm32(X16, uoffset);
            emit32(0x8B100000 | (addr_reg << 5) | rd); // ADD Xd, addr, X16
            emit32(0x8B140000 | (rd << 5) | rd);       // ADD Xd, Xd, X20
         } else if (uoffset > 0) {
            emit_add_imm(rd, addr_reg, uoffset);
            emit32(0x8B140000 | (rd << 5) | rd); // ADD Xd, Xd, X20
         } else {
            emit32(0x8B140000 | (addr_reg << 5) | rd); // ADD Xd, addr, X20
         }
      }

      // Memory load: various sizes and sign extensions
      void emit_mem_load(uint32_t rd, uint32_t addr_reg, uint32_t uoffset, uint32_t load_op) {
         emit_effective_addr(X16, addr_reg, uoffset);
         emit32(load_op | (X16 << 5) | rd);
      }

      void emit_mem_store(uint32_t rs, uint32_t addr_reg, uint32_t uoffset, uint32_t store_op) {
         emit_effective_addr(X16, addr_reg, uoffset);
         emit32(store_op | (X16 << 5) | rs);
      }

      // Load opcodes (all use [Xn] addressing, offset 0)
      static constexpr uint32_t LDR_W  = 0xB9400000; // LDR Wt, [Xn]
      static constexpr uint32_t LDR_X  = 0xF9400000; // LDR Xt, [Xn]
      static constexpr uint32_t LDRB   = 0x39400000; // LDRB Wt, [Xn]
      static constexpr uint32_t LDRH   = 0x79400000; // LDRH Wt, [Xn]
      static constexpr uint32_t LDRSB_W = 0x39C00000; // LDRSB Wt, [Xn]
      static constexpr uint32_t LDRSH_W = 0x79C00000; // LDRSH Wt, [Xn]
      static constexpr uint32_t LDRSB_X = 0x39800000; // LDRSB Xt, [Xn]
      static constexpr uint32_t LDRSH_X = 0x79800000; // LDRSH Xt, [Xn]
      static constexpr uint32_t LDRSW   = 0xB9800000; // LDRSW Xt, [Xn]

      // Store opcodes
      static constexpr uint32_t STR_W  = 0xB9000000; // STR Wt, [Xn]
      static constexpr uint32_t STR_X  = 0xF9000000; // STR Xt, [Xn]
      static constexpr uint32_t STRB   = 0x39000000; // STRB Wt, [Xn]
      static constexpr uint32_t STRH   = 0x79000000; // STRH Wt, [Xn]

      // ──────── Const-immediate helpers ────────

      bool try_get_const(uint32_t vreg, int64_t& out) {
         if (!_func_def_inst || vreg >= _num_vregs) return false;
         uint32_t def = _func_def_inst[vreg];
         if (def >= _func_inst_count) return false;
         auto& di = _func_insts[def];
         if (di.opcode != ir_op::const_i32 && di.opcode != ir_op::const_i64) return false;
         out = di.imm64;
         return true;
      }

      void kill_const_if_single_use(uint32_t vreg) {
         if (_func_use_count && vreg < _num_vregs && _func_use_count[vreg] == 1) {
            uint32_t def = _func_def_inst[vreg];
            if (def < _func_inst_count)
               _func_insts[def].flags |= IR_DEAD;
         }
      }

      // ──────── Block/branch tracking ────────

      struct block_fixup {
         void* branch;
         block_fixup* next;
      };

      void mark_block_start(uint32_t block_idx) {
         if (block_idx < _num_blocks) _block_addrs[block_idx] = code;
      }

      void mark_block_end(ir_function&, uint32_t block_idx, bool is_if) {
         if (block_idx >= _num_blocks) return;
         _block_addrs[block_idx] = code;
         for (auto* f = _block_fixups[block_idx]; f; f = f->next) {
            fix_branch(f->branch, code);
         }
         _block_fixups[block_idx] = nullptr;
         if (is_if) pop_if_fixup_to(code);
      }

      void* emit_branch_placeholder() {
         void* branch = code;
         emit32(0x14000000); // B (patched later)
         return branch;
      }

      void* emit_cond_branch_placeholder(uint32_t cond) {
         void* branch = code;
         emit32(0x54000000 | cond); // B.cond (patched later)
         return branch;
      }

      void emit_branch_to_block(ir_function&, uint32_t block_idx) {
         if (block_idx >= _num_blocks) return;
         if (_block_addrs[block_idx] != nullptr) {
            void* branch = emit_branch_placeholder();
            fix_branch(branch, _block_addrs[block_idx]);
         } else {
            void* branch = emit_branch_placeholder();
            auto* fixup = _allocator.alloc<block_fixup>(1);
            fixup->branch = branch;
            fixup->next = _block_fixups[block_idx];
            _block_fixups[block_idx] = fixup;
         }
      }

      void emit_cond_branch_to_block(uint32_t block_idx, uint32_t cond) {
         if (block_idx >= _num_blocks) return;
         if (_block_addrs[block_idx] != nullptr) {
            void* branch = emit_cond_branch_placeholder(cond);
            fix_branch(branch, _block_addrs[block_idx]);
         } else {
            void* branch = emit_cond_branch_placeholder(cond);
            auto* fixup = _allocator.alloc<block_fixup>(1);
            fixup->branch = branch;
            fixup->next = _block_fixups[block_idx];
            _block_fixups[block_idx] = fixup;
         }
      }

      // If fixup stack
      static constexpr uint32_t MAX_IF_DEPTH = 256;
      void* _if_fixups[MAX_IF_DEPTH];
      uint32_t _if_fixup_top = 0;

      void push_if_fixup(void* branch) {
         if (_if_fixup_top < MAX_IF_DEPTH) _if_fixups[_if_fixup_top++] = branch;
      }
      void pop_if_fixup_to(void* target) {
         if (_if_fixup_top > 0) {
            void* branch = _if_fixups[--_if_fixup_top];
            if (branch && target) fix_branch(branch, target);
         }
      }

      // ──────── Branch fusion ────────

      bool emit_fused_branch(ir_function& func, uint32_t idx, uint32_t cc) {
         auto& next = func.insts[idx + 1];
         if (next.opcode == ir_op::if_) {
            void* branch = emit_cond_branch_placeholder(invert_condition(cc));
            push_if_fixup(branch);
            return true;
         }
         if (next.opcode == ir_op::br_if) {
            emit_cond_branch_to_block(next.br.target, cc);
            return true;
         }
         return false;
      }

      // ──────── Emit call ────────

      void* emit_bl_placeholder() {
         void* branch = code;
         emit32(0x94000000); // BL (patched)
         return branch;
      }

      // ──────── SSE/NEON float helpers ────────

      // Move between GP and FP registers
      void emit_fmov_to_fp(uint32_t vd, uint32_t xn, bool is64) {
         if (is64) emit32(0x9E670000 | (xn << 5) | vd); // FMOV Dd, Xn
         else      emit32(0x1E270000 | (xn << 5) | vd); // FMOV Sd, Wn
      }
      void emit_fmov_from_fp(uint32_t xd, uint32_t vn, bool is64) {
         if (is64) emit32(0x9E660000 | (vn << 5) | xd); // FMOV Xd, Dn
         else      emit32(0x1E260000 | (vn << 5) | xd); // FMOV Wd, Sn
      }

      // ──────── Main IR instruction emission (register mode) ────────

      void emit_ir_inst_reg(ir_function& func, const ir_inst& inst, uint32_t idx) {
         if (inst.flags & IR_DEAD) return;

         switch (inst.opcode) {
         case ir_op::nop:
         case ir_op::block:
         case ir_op::loop:
         case ir_op::drop:
            break;

         // arg: push a vreg value to the stack (for upcoming call)
         case ir_op::arg:
            load_vreg_x0(inst.rr.src1);
            emit_push(X0);
            break;

         case ir_op::block_start:
            mark_block_start(inst.dest);
            break;
         case ir_op::block_end:
            mark_block_end(func, inst.dest, inst.flags & 1);
            break;

         // ── Constants ──
         case ir_op::const_i32: {
            uint32_t val = static_cast<uint32_t>(inst.imm64);
            int8_t pr = get_phys(inst.dest);
            if (pr >= 0) emit_mov_imm32(phys_to_reg(pr), val);
            else { emit_mov_imm32(X0, val); store_x0_vreg(inst.dest); }
            break;
         }
         case ir_op::const_i64: {
            uint64_t val = static_cast<uint64_t>(inst.imm64);
            int8_t pr = get_phys(inst.dest);
            if (pr >= 0) emit_mov_imm64(phys_to_reg(pr), val);
            else { emit_mov_imm64(X0, val); store_x0_vreg(inst.dest); }
            break;
         }
         case ir_op::const_f32: {
            uint32_t bits;
            memcpy(&bits, &inst.immf32, 4);
            int8_t pr = get_phys(inst.dest);
            if (pr >= 0) emit_mov_imm32(phys_to_reg(pr), bits);
            else { emit_mov_imm32(X0, bits); store_x0_vreg(inst.dest); }
            break;
         }
         case ir_op::const_f64: {
            uint64_t bits;
            memcpy(&bits, &inst.immf64, 8);
            int8_t pr = get_phys(inst.dest);
            if (pr >= 0) emit_mov_imm64(phys_to_reg(pr), bits);
            else { emit_mov_imm64(X0, bits); store_x0_vreg(inst.dest); }
            break;
         }

         // ── Mov ──
         case ir_op::mov: {
            int8_t pr_d = get_phys(inst.dest);
            int8_t pr_s = get_phys(inst.rr.src1);
            if (pr_d >= 0 && pr_s >= 0) emit_mov_reg(phys_to_reg(pr_d), phys_to_reg(pr_s));
            else { load_vreg_x0(inst.rr.src1); store_x0_vreg(inst.dest); }
            break;
         }

         // ── Integer binary ops (3-operand ARM64) ──
         case ir_op::i32_add: emit_binop(func, inst, 0x0B000000, true); break;  // ADD Wd
         case ir_op::i32_sub: emit_binop(func, inst, 0x4B000000, true); break;  // SUB Wd
         case ir_op::i32_mul: emit_binop3(inst, 0x1B007C00, true); break;       // MUL Wd (MADD Wd,Wn,Wm,WZR)
         case ir_op::i32_and: emit_binop(func, inst, 0x0A000000, true); break;  // AND Wd
         case ir_op::i32_or:  emit_binop(func, inst, 0x2A000000, true); break;  // ORR Wd
         case ir_op::i32_xor: emit_binop(func, inst, 0x4A000000, true); break;  // EOR Wd
         case ir_op::i64_add: emit_binop(func, inst, 0x8B000000, false); break;
         case ir_op::i64_sub: emit_binop(func, inst, 0xCB000000, false); break;
         case ir_op::i64_mul: emit_binop3(inst, 0x9B007C00, false); break;      // MUL Xd (MADD Xd,Xn,Xm,XZR)
         case ir_op::i64_and: emit_binop(func, inst, 0x8A000000, false); break;
         case ir_op::i64_or:  emit_binop(func, inst, 0xAA000000, false); break;
         case ir_op::i64_xor: emit_binop(func, inst, 0xCA000000, false); break;

         // ── Shifts ──
         case ir_op::i32_shl:   emit_binop_simple(inst, 0x1AC02000, true); break;  // LSLV Wd
         case ir_op::i32_shr_u: emit_binop_simple(inst, 0x1AC02400, true); break;  // LSRV Wd
         case ir_op::i32_shr_s: emit_binop_simple(inst, 0x1AC02800, true); break;  // ASRV Wd
         case ir_op::i32_rotl: emit_i32_rotl(inst); break;
         case ir_op::i32_rotr:  emit_binop_simple(inst, 0x1AC02C00, true); break;  // RORV Wd
         case ir_op::i64_shl:   emit_binop_simple(inst, 0x9AC02000, false); break;
         case ir_op::i64_shr_u: emit_binop_simple(inst, 0x9AC02400, false); break;
         case ir_op::i64_shr_s: emit_binop_simple(inst, 0x9AC02800, false); break;
         case ir_op::i64_rotl: emit_i64_rotl(inst); break;
         case ir_op::i64_rotr:  emit_binop_simple(inst, 0x9AC02C00, false); break;

         // ── Division (native on ARM64!) ──
         case ir_op::i32_div_s: emit_binop_simple(inst, 0x1AC00C00, true); break;  // SDIV Wd
         case ir_op::i32_div_u: emit_binop_simple(inst, 0x1AC00800, true); break;  // UDIV Wd
         case ir_op::i64_div_s: emit_binop_simple(inst, 0x9AC00C00, false); break;
         case ir_op::i64_div_u: emit_binop_simple(inst, 0x9AC00800, false); break;

         // ── Remainder (SDIV + MSUB) ──
         case ir_op::i32_rem_s: emit_i32_rem_s(inst); break;
         case ir_op::i32_rem_u: emit_i32_rem_u(inst); break;
         case ir_op::i64_rem_s: emit_i64_rem_s(inst); break;
         case ir_op::i64_rem_u: emit_i64_rem_u(inst); break;

         // ── Unary integer ──
         case ir_op::i32_clz:
            load_vreg_x0(inst.rr.src1);
            emit32(0x5AC01000); // CLZ W0, W0
            store_x0_vreg(inst.dest);
            break;
         case ir_op::i32_ctz:
            load_vreg_x0(inst.rr.src1);
            emit32(0x5AC00000); // RBIT W0, W0
            emit32(0x5AC01000); // CLZ W0, W0
            store_x0_vreg(inst.dest);
            break;
         case ir_op::i32_popcnt:
            load_vreg_x0(inst.rr.src1);
            // FMOV S0, W0; CNT V0.8B, V0.8B; ADDV B0, V0.8B; UMOV W0, V0.B[0]
            emit32(0x1E270000); // FMOV S0, W0
            emit32(0x0E205800); // CNT V0.8B, V0.8B
            emit32(0x0E31B800); // ADDV B0, V0.8B
            emit32(0x0E013C00); // UMOV W0, V0.B[0]
            store_x0_vreg(inst.dest);
            break;
         case ir_op::i64_clz:
            load_vreg_x0(inst.rr.src1);
            emit32(0xDAC01000); // CLZ X0, X0
            store_x0_vreg(inst.dest);
            break;
         case ir_op::i64_ctz:
            load_vreg_x0(inst.rr.src1);
            emit32(0xDAC00000); // RBIT X0, X0
            emit32(0xDAC01000); // CLZ X0, X0
            store_x0_vreg(inst.dest);
            break;
         case ir_op::i64_popcnt:
            load_vreg_x0(inst.rr.src1);
            emit32(0x9E670000); // FMOV D0, X0
            emit32(0x0E205800); // CNT V0.8B, V0.8B
            emit32(0x0E31B800); // ADDV B0, V0.8B
            emit32(0x0E013C00); // UMOV W0, V0.B[0]
            store_x0_vreg(inst.dest);
            break;

         // ── Comparisons ──
         case ir_op::i32_eqz: emit_eqz(func, inst, idx, true); break;
         case ir_op::i64_eqz: emit_eqz(func, inst, idx, false); break;
         case ir_op::i32_eq:  emit_cmp(func, inst, idx, COND_EQ, true); break;
         case ir_op::i32_ne:  emit_cmp(func, inst, idx, COND_NE, true); break;
         case ir_op::i32_lt_s: emit_cmp(func, inst, idx, COND_LT, true); break;
         case ir_op::i32_lt_u: emit_cmp(func, inst, idx, COND_LO, true); break;
         case ir_op::i32_gt_s: emit_cmp(func, inst, idx, COND_GT, true); break;
         case ir_op::i32_gt_u: emit_cmp(func, inst, idx, COND_HI, true); break;
         case ir_op::i32_le_s: emit_cmp(func, inst, idx, COND_LE, true); break;
         case ir_op::i32_le_u: emit_cmp(func, inst, idx, COND_LS, true); break;
         case ir_op::i32_ge_s: emit_cmp(func, inst, idx, COND_GE, true); break;
         case ir_op::i32_ge_u: emit_cmp(func, inst, idx, COND_HS, true); break;
         case ir_op::i64_eq:  emit_cmp(func, inst, idx, COND_EQ, false); break;
         case ir_op::i64_ne:  emit_cmp(func, inst, idx, COND_NE, false); break;
         case ir_op::i64_lt_s: emit_cmp(func, inst, idx, COND_LT, false); break;
         case ir_op::i64_lt_u: emit_cmp(func, inst, idx, COND_LO, false); break;
         case ir_op::i64_gt_s: emit_cmp(func, inst, idx, COND_GT, false); break;
         case ir_op::i64_gt_u: emit_cmp(func, inst, idx, COND_HI, false); break;
         case ir_op::i64_le_s: emit_cmp(func, inst, idx, COND_LE, false); break;
         case ir_op::i64_le_u: emit_cmp(func, inst, idx, COND_LS, false); break;
         case ir_op::i64_ge_s: emit_cmp(func, inst, idx, COND_GE, false); break;
         case ir_op::i64_ge_u: emit_cmp(func, inst, idx, COND_HS, false); break;

         // ── Select ──
         case ir_op::select: {
            load_vreg_x0(inst.sel.cond);
            load_vreg_to(X1, inst.sel.val1);
            load_vreg_to(X16, inst.sel.val2);
            emit_cmp_imm32(X0, 0);
            // CSEL X0, X1, X16, NE
            emit32(0x9A900020 | (COND_NE << 12) | (X16 << 16) | (X1 << 5) | X0);
            store_x0_vreg(inst.dest);
            break;
         }

         // ── Conversions ──
         case ir_op::i32_wrap_i64:
            load_vreg_x0(inst.rr.src1);
            emit_mov_reg32(X0, X0); // MOV W0, W0 (zero-extend)
            store_x0_vreg(inst.dest);
            break;
         case ir_op::i64_extend_s_i32:
            load_vreg_x0(inst.rr.src1);
            emit32(0x93407C00); // SXTW X0, W0
            store_x0_vreg(inst.dest);
            break;
         case ir_op::i64_extend_u_i32:
            load_vreg_x0(inst.rr.src1);
            emit_mov_reg32(X0, X0); // zero-extend by moving 32-bit
            store_x0_vreg(inst.dest);
            break;
         case ir_op::i32_extend8_s:
            load_vreg_x0(inst.rr.src1);
            emit32(0x13001C00); // SXTB W0, W0
            store_x0_vreg(inst.dest);
            break;
         case ir_op::i32_extend16_s:
            load_vreg_x0(inst.rr.src1);
            emit32(0x13003C00); // SXTH W0, W0
            store_x0_vreg(inst.dest);
            break;
         case ir_op::i64_extend8_s:
            load_vreg_x0(inst.rr.src1);
            emit32(0x93401C00); // SXTB X0, W0
            store_x0_vreg(inst.dest);
            break;
         case ir_op::i64_extend16_s:
            load_vreg_x0(inst.rr.src1);
            emit32(0x93403C00); // SXTH X0, W0
            store_x0_vreg(inst.dest);
            break;
         case ir_op::i64_extend32_s:
            load_vreg_x0(inst.rr.src1);
            emit32(0x93407C00); // SXTW X0, W0
            store_x0_vreg(inst.dest);
            break;

         // ── Reinterpret (no-op, same bit pattern) ──
         case ir_op::i32_reinterpret_f32:
         case ir_op::i64_reinterpret_f64:
         case ir_op::f32_reinterpret_i32:
         case ir_op::f64_reinterpret_i64:
            if (inst.rr.src1 != ir_vreg_none && inst.dest != ir_vreg_none) {
               load_vreg_x0(inst.rr.src1);
               store_x0_vreg(inst.dest);
            }
            break;

         // ── Control flow ──
         case ir_op::if_: {
            load_vreg_x0(inst.br.src1);
            emit_cmp_imm32(X0, 0);
            void* branch = emit_cond_branch_placeholder(COND_EQ);
            push_if_fixup(branch);
            break;
         }
         case ir_op::else_: {
            uint32_t target_block = inst.br.target;
            if (target_block < _num_blocks) {
               void* jmp = emit_branch_placeholder();
               auto* fixup = _allocator.alloc<block_fixup>(1);
               fixup->branch = jmp;
               fixup->next = _block_fixups[target_block];
               _block_fixups[target_block] = fixup;
            }
            pop_if_fixup_to(code);
            break;
         }
         case ir_op::br: {
            if (_in_br_table) {
               bool is_default = (_br_table_case >= _br_table_size);
               if (is_default) {
                  emit_pop(X0); // discard index
                  emit_branch_to_block(func, inst.br.target);
                  _in_br_table = false;
               } else {
                  // CMP W [SP], #case
                  emit_ldr_offset(X16, SP, 0);
                  emit_cmp_imm32(X16, _br_table_case);
                  void* skip = emit_cond_branch_placeholder(COND_NE);
                  emit_pop(X0); // pop index
                  emit_branch_to_block(func, inst.br.target);
                  fix_branch(skip, code);
                  _br_table_case++;
               }
            } else {
               emit_branch_to_block(func, inst.br.target);
            }
            break;
         }
         case ir_op::br_if: {
            load_vreg_x0(inst.br.src1);
            emit_cmp_imm32(X0, 0);
            emit_cond_branch_to_block(inst.br.target, COND_NE);
            break;
         }
         case ir_op::br_table: {
            load_vreg_x0(inst.rr.src1);
            emit_push(X0);
            _br_table_case = 0;
            _br_table_size = inst.dest;
            _in_br_table = true;
            break;
         }
         case ir_op::unreachable:
            emit_error_handler(&on_unreachable);
            break;

         // ── Return ──
         case ir_op::return_: {
            if (inst.rr.src1 != ir_vreg_none) {
               load_vreg_x0(inst.rr.src1);
            }
            // Restore callee-saved
            if (_callee_saved_used) {
               int32_t save_offset = (_body_locals + _num_spill_slots) * 8;
               for (int i = 0; i < 7; ++i) {
                  if (_callee_saved_used & (1 << i)) {
                     emit_ldr_offset(callee_saved_reg(i), SP, save_offset);
                     save_offset += 8;
                  }
               }
            }
            emit32(0x91000000 | (FP << 5) | SP); // MOV SP, X29
            emit32(0xA8C17BFD); // LDP X29, X30, [SP], #16
            emit32(0xD65F03C0); // RET
            break;
         }

         // ── Calls ──
         case ir_op::call: {
            uint32_t funcnum = inst.call.index;
            const func_type& ft = _mod.get_function_type(funcnum);
            emit_call_depth_dec();
            void* branch = emit_bl_placeholder();
            register_call(branch, funcnum);
            // Pop params
            uint32_t arg_bytes = 0;
            for (uint32_t p = 0; p < ft.param_types.size(); ++p)
               arg_bytes += (ft.param_types[p] == types::v128) ? 32 : 16;
            if (arg_bytes > 0) emit_add_imm(SP, SP, arg_bytes);
            emit_call_depth_inc();
            if (ft.return_count > 0 && inst.dest != ir_vreg_none) {
               store_x0_vreg(inst.dest);
            }
            break;
         }
         case ir_op::call_indirect: {
            uint32_t fti = inst.call.index;
            const func_type& ft = _mod.types[fti];
            // Pop table index
            emit_pop(X0);
            // Bounds check
            uint32_t table_size = _mod.tables[0].limits.initial;
            emit_cmp_imm32(X0, table_size);
            emit_branch_to_handler(COND_HS, call_indirect_handler);
            // Table entry: index * 16
            emit32(0xD37CEC00); // LSL X0, X0, #4
            if (_mod.indirect_table(0)) {
               int32_t toff = wasm_allocator::table_offset();
               if (toff >= 0) {
                  emit_ldr_offset(X8, X20, toff);
               } else {
                  emit_mov_imm64(X16, static_cast<uint64_t>(static_cast<int64_t>(toff)));
                  emit32(0x8B100000 | (X20 << 5) | X8); // ADD X8, X20, X16
                  emit32(0xF9400108); // LDR X8, [X8]
               }
               emit32(0x8B000000 | (X0 << 16) | (X8 << 5) | X0); // ADD X0, X8, X0
            } else {
               // table_offset is negative: X0 = X20 + X0 + table_offset
               int32_t toff = wasm_allocator::table_offset();
               emit32(0x8B000000 | (X0 << 16) | (X20 << 5) | X0); // ADD X0, X20, X0
               if (toff < 0) {
                  emit_sub_imm(X0, X0, static_cast<uint32_t>(-toff));
               } else {
                  emit_add_imm(X0, X0, static_cast<uint32_t>(toff));
               }
            }
            // Type check
            emit32(0xB9400008 | (X0 << 5)); // LDR W8, [X0]
            emit_cmp_imm32(X8, fti);
            emit_branch_to_handler(COND_NE, type_error_handler);
            // Load function pointer
            emit_ldr_offset(X8, X0, 8);
            emit_call_depth_dec();
            emit32(0xD63F0100); // BLR X8
            // Pop params
            uint32_t arg_bytes = 0;
            for (uint32_t p = 0; p < ft.param_types.size(); ++p)
               arg_bytes += (ft.param_types[p] == types::v128) ? 32 : 16;
            if (arg_bytes > 0) emit_add_imm(SP, SP, arg_bytes);
            emit_call_depth_inc();
            if (ft.return_count > 0 && inst.dest != ir_vreg_none) {
               store_x0_vreg(inst.dest);
            }
            break;
         }

         // ── Arg (push for upcoming call) ──
         // Note: arg opcode is handled at top as no-op in register mode
         // but we need it here for the stack push
         // Actually arg is already handled — but we need to push for calls
         // The 'arg' case at top breaks (no-op), so we override here:
         // Wait — arg is listed in the break-only section above. Let me fix this.
         // Actually we need arg to push. Let me handle it properly.

         // ── Local access ──
         case ir_op::local_get: {
            int8_t pr = get_phys(inst.dest);
            uint32_t rd = (pr >= 0) ? phys_to_reg(pr) : X0;
            emit_local_load(func, inst.local.index, rd);
            if (pr < 0) store_x0_vreg(inst.dest);
            break;
         }
         case ir_op::local_set: {
            int8_t pr = get_phys(inst.local.src1);
            uint32_t rs = (pr >= 0) ? phys_to_reg(pr) : X0;
            if (pr < 0) load_vreg_x0(inst.local.src1);
            emit_local_store(func, inst.local.index, rs);
            break;
         }
         case ir_op::local_tee: {
            load_vreg_x0(inst.local.src1);
            emit_local_store(func, inst.local.index, X0);
            break;
         }

         // ── Global access ──
         case ir_op::global_get: {
            emit_global_load(inst.local.index, X0);
            store_x0_vreg(inst.dest);
            break;
         }
         case ir_op::global_set: {
            load_vreg_x0(inst.local.src1);
            emit_global_store(inst.local.index, X0);
            break;
         }

         // ── Memory loads ──
         case ir_op::i32_load:     emit_load_op(inst, LDR_W); break;
         case ir_op::i64_load:     emit_load_op(inst, LDR_X); break;
         case ir_op::f32_load:     emit_load_op(inst, LDR_W); break;
         case ir_op::f64_load:     emit_load_op(inst, LDR_X); break;
         case ir_op::i32_load8_u:  emit_load_op(inst, LDRB); break;
         case ir_op::i32_load16_u: emit_load_op(inst, LDRH); break;
         case ir_op::i32_load8_s:  emit_load_op(inst, LDRSB_W); break;
         case ir_op::i32_load16_s: emit_load_op(inst, LDRSH_W); break;
         case ir_op::i64_load8_u:  emit_load_op(inst, LDRB); break;
         case ir_op::i64_load16_u: emit_load_op(inst, LDRH); break;
         case ir_op::i64_load32_u: emit_load_op(inst, LDR_W); break;
         case ir_op::i64_load8_s:  emit_load_op(inst, LDRSB_X); break;
         case ir_op::i64_load16_s: emit_load_op(inst, LDRSH_X); break;
         case ir_op::i64_load32_s: emit_load_op(inst, LDRSW); break;

         // ── Memory stores ──
         case ir_op::i32_store:   emit_store_op(inst, STR_W); break;
         case ir_op::i64_store:   emit_store_op(inst, STR_X); break;
         case ir_op::f32_store:   emit_store_op(inst, STR_W); break;
         case ir_op::f64_store:   emit_store_op(inst, STR_X); break;
         case ir_op::i32_store8:  emit_store_op(inst, STRB); break;
         case ir_op::i32_store16: emit_store_op(inst, STRH); break;
         case ir_op::i64_store8:  emit_store_op(inst, STRB); break;
         case ir_op::i64_store16: emit_store_op(inst, STRH); break;
         case ir_op::i64_store32: emit_store_op(inst, STR_W); break;

         // ── Memory management ──
         case ir_op::memory_size:
            emit_push(X19); emit_push(X20);
            emit_mov_reg(X0, X19);
            emit_mov_imm64(X8, reinterpret_cast<uint64_t>(&current_memory));
            emit32(0xD63F0100); // BLR X8
            emit_pop(X20); emit_pop(X19);
            store_x0_vreg(inst.dest);
            break;
         case ir_op::memory_grow:
            load_vreg_x0(inst.rr.src1);
            emit_push(X19); emit_push(X20);
            emit_mov_reg(X1, X0); // pages
            emit_mov_reg(X0, X19); // context
            emit_mov_imm64(X8, reinterpret_cast<uint64_t>(&grow_memory));
            emit32(0xD63F0100);
            emit_pop(X20); emit_pop(X19);
            store_x0_vreg(inst.dest);
            break;

         // ── Float unary ops ──
         case ir_op::f32_abs:
            load_vreg_x0(inst.rr.src1);
            emit32(0x121F7800); // AND W0, W0, #0x7FFFFFFF
            store_x0_vreg(inst.dest);
            break;
         case ir_op::f32_neg:
            load_vreg_x0(inst.rr.src1);
            emit32(0x52100008); // MOVZ W8, #0x8000, LSL #16
            emit32(0x4A080000); // EOR W0, W0, W8
            store_x0_vreg(inst.dest);
            break;
         case ir_op::f64_abs:
            load_vreg_x0(inst.rr.src1);
            emit32(0x927FFE00); // AND X0, X0, #0x7FFFFFFFFFFFFFFF
            store_x0_vreg(inst.dest);
            break;
         case ir_op::f64_neg:
            load_vreg_x0(inst.rr.src1);
            emit_mov_imm64(X8, 0x8000000000000000ULL);
            emit32(0xCA080000); // EOR X0, X0, X8
            store_x0_vreg(inst.dest);
            break;
         case ir_op::f32_sqrt:
            load_vreg_x0(inst.rr.src1);
            emit32(0x1E270000); // FMOV S0, W0
            emit32(0x1E21C000); // FSQRT S0, S0
            emit32(0x1E260000); // FMOV W0, S0
            store_x0_vreg(inst.dest);
            break;
         case ir_op::f64_sqrt:
            load_vreg_x0(inst.rr.src1);
            emit32(0x9E670000); // FMOV D0, X0
            emit32(0x1E61C000); // FSQRT D0, D0
            emit32(0x9E660000); // FMOV X0, D0
            store_x0_vreg(inst.dest);
            break;
         case ir_op::f32_ceil:   emit_f32_round(inst, 0x1E264000); break; // FRINTP S0, S0
         case ir_op::f32_floor:  emit_f32_round(inst, 0x1E264800); break; // FRINTM S0, S0
         case ir_op::f32_trunc:  emit_f32_round(inst, 0x1E265800); break; // FRINTZ S0, S0
         case ir_op::f32_nearest:emit_f32_round(inst, 0x1E264400); break; // FRINTN S0, S0 (ties to even)
         case ir_op::f64_ceil:   emit_f64_round(inst, 0x1E664000); break;
         case ir_op::f64_floor:  emit_f64_round(inst, 0x1E664800); break;
         case ir_op::f64_trunc:  emit_f64_round(inst, 0x1E665800); break;
         case ir_op::f64_nearest:emit_f64_round(inst, 0x1E664400); break;

         // ── Float binary ops ──
         case ir_op::f32_add: emit_f32_binop(inst, 0x1E202800); break; // FADD S0, S0, S1
         case ir_op::f32_sub: emit_f32_binop(inst, 0x1E203800); break;
         case ir_op::f32_mul: emit_f32_binop(inst, 0x1E200800); break;
         case ir_op::f32_div: emit_f32_binop(inst, 0x1E201800); break;
         case ir_op::f32_min: emit_f32_binop(inst, 0x1E205800); break; // FMINNM
         case ir_op::f32_max: emit_f32_binop(inst, 0x1E206800); break; // FMAXNM
         case ir_op::f64_add: emit_f64_binop(inst, 0x1E602800); break;
         case ir_op::f64_sub: emit_f64_binop(inst, 0x1E603800); break;
         case ir_op::f64_mul: emit_f64_binop(inst, 0x1E600800); break;
         case ir_op::f64_div: emit_f64_binop(inst, 0x1E601800); break;
         case ir_op::f64_min: emit_f64_binop(inst, 0x1E605800); break;
         case ir_op::f64_max: emit_f64_binop(inst, 0x1E606800); break;

         case ir_op::f32_copysign:
            load_vreg_x0(inst.rr.src1); // magnitude
            load_vreg_x1(inst.rr.src2); // sign
            emit32(0x121F7800); // AND W0, W0, #0x7FFFFFFF
            emit32(0x12010021); // AND W1, W1, #0x80000000
            emit32(0x2A010000); // ORR W0, W0, W1
            store_x0_vreg(inst.dest);
            break;
         case ir_op::f64_copysign:
            load_vreg_x0(inst.rr.src1);
            load_vreg_x1(inst.rr.src2);
            emit32(0x927FFE00); // AND X0, X0, #0x7FFFFFFFFFFFFFFF
            emit_mov_imm64(X8, 0x8000000000000000ULL);
            emit32(0x8A080021); // AND X1, X1, X8
            emit32(0xAA010000); // ORR X0, X0, X1
            store_x0_vreg(inst.dest);
            break;

         // ── Float comparisons ──
         case ir_op::f32_eq: emit_f32_cmp(inst, COND_EQ); break;
         case ir_op::f32_ne: emit_f32_cmp(inst, COND_NE); break;
         case ir_op::f32_lt: emit_f32_cmp(inst, COND_LO); break; // MI for ordered, but LO handles unordered
         case ir_op::f32_gt: emit_f32_cmp(inst, COND_GT); break;
         case ir_op::f32_le: emit_f32_cmp(inst, COND_LS); break;
         case ir_op::f32_ge: emit_f32_cmp(inst, COND_GE); break;
         case ir_op::f64_eq: emit_f64_cmp(inst, COND_EQ); break;
         case ir_op::f64_ne: emit_f64_cmp(inst, COND_NE); break;
         case ir_op::f64_lt: emit_f64_cmp(inst, COND_LO); break;
         case ir_op::f64_gt: emit_f64_cmp(inst, COND_GT); break;
         case ir_op::f64_le: emit_f64_cmp(inst, COND_LS); break;
         case ir_op::f64_ge: emit_f64_cmp(inst, COND_GE); break;

         // ── Float-to-int conversions ──
         case ir_op::i32_trunc_s_f32:     emit_fcvt(inst, 0x1E380000, true, true); break;   // FCVTZS W0, S0
         case ir_op::i32_trunc_u_f32:     emit_fcvt(inst, 0x1E390000, true, true); break;   // FCVTZU W0, S0
         case ir_op::i32_trunc_s_f64:     emit_fcvt(inst, 0x1E780000, true, false); break;  // FCVTZS W0, D0
         case ir_op::i32_trunc_u_f64:     emit_fcvt(inst, 0x1E790000, true, false); break;
         case ir_op::i64_trunc_s_f32:     emit_fcvt(inst, 0x9E380000, true, true); break;   // FCVTZS X0, S0
         case ir_op::i64_trunc_u_f32:     emit_fcvt(inst, 0x9E390000, true, true); break;
         case ir_op::i64_trunc_s_f64:     emit_fcvt(inst, 0x9E780000, true, false); break;
         case ir_op::i64_trunc_u_f64:     emit_fcvt(inst, 0x9E790000, true, false); break;
         case ir_op::i32_trunc_sat_f32_s: emit_fcvt(inst, 0x1E380000, true, true); break;
         case ir_op::i32_trunc_sat_f32_u: emit_fcvt(inst, 0x1E390000, true, true); break;
         case ir_op::i32_trunc_sat_f64_s: emit_fcvt(inst, 0x1E780000, true, false); break;
         case ir_op::i32_trunc_sat_f64_u: emit_fcvt(inst, 0x1E790000, true, false); break;
         case ir_op::i64_trunc_sat_f32_s: emit_fcvt(inst, 0x9E380000, true, true); break;
         case ir_op::i64_trunc_sat_f32_u: emit_fcvt(inst, 0x9E390000, true, true); break;
         case ir_op::i64_trunc_sat_f64_s: emit_fcvt(inst, 0x9E780000, true, false); break;
         case ir_op::i64_trunc_sat_f64_u: emit_fcvt(inst, 0x9E790000, true, false); break;

         // ── Int-to-float conversions ──
         case ir_op::f32_convert_s_i32:   emit_icvt(inst, 0x1E220000, true, true); break;   // SCVTF S0, W0
         case ir_op::f32_convert_u_i32:   emit_icvt(inst, 0x1E230000, true, true); break;
         case ir_op::f32_convert_s_i64:   emit_icvt(inst, 0x9E220000, false, true); break;
         case ir_op::f32_convert_u_i64:   emit_icvt(inst, 0x9E230000, false, true); break;
         case ir_op::f64_convert_s_i32:   emit_icvt(inst, 0x1E620000, true, false); break;
         case ir_op::f64_convert_u_i32:   emit_icvt(inst, 0x1E630000, true, false); break;
         case ir_op::f64_convert_s_i64:   emit_icvt(inst, 0x9E620000, false, false); break;
         case ir_op::f64_convert_u_i64:   emit_icvt(inst, 0x9E630000, false, false); break;

         // ── Float-float conversions ──
         case ir_op::f32_demote_f64:
            load_vreg_x0(inst.rr.src1);
            emit32(0x9E670000); // FMOV D0, X0
            emit32(0x1E624000); // FCVT S0, D0
            emit32(0x1E260000); // FMOV W0, S0
            store_x0_vreg(inst.dest);
            break;
         case ir_op::f64_promote_f32:
            load_vreg_x0(inst.rr.src1);
            emit32(0x1E270000); // FMOV S0, W0
            emit32(0x1E22C000); // FCVT D0, S0
            emit32(0x9E660000); // FMOV X0, D0
            store_x0_vreg(inst.dest);
            break;

         // ── Bulk memory ──
         case ir_op::memory_fill: {
            emit_pop(X2); // count
            emit_pop(X1); // value
            emit_pop(X0); // dest
            emit32(0x8B140000); // ADD X0, X0, X20 (native addr)
            emit_push(X19); emit_push(X20);
            emit_mov_imm64(X8, reinterpret_cast<uint64_t>(memset));
            emit32(0xD63F0100);
            emit_pop(X20); emit_pop(X19);
            break;
         }
         case ir_op::memory_copy: {
            emit_pop(X2); // count
            emit_pop(X1); // src
            emit_pop(X0); // dest
            emit32(0x8B140000); // ADD X0, X0, X20
            emit32(0x8B140021); // ADD X1, X1, X20
            emit_push(X19); emit_push(X20);
            emit_mov_imm64(X8, reinterpret_cast<uint64_t>(memmove));
            emit32(0xD63F0100);
            emit_pop(X20); emit_pop(X19);
            break;
         }

         case ir_op::memory_init:
         case ir_op::data_drop:
         case ir_op::table_init:
         case ir_op::elem_drop:
         case ir_op::table_copy:
            break; // TODO

         default:
            break;
         }
      }

      // ──────── Binary op helpers ────────

      // 2-operand binary: OP Xd, Xn, Xm (with const-imm folding for add/sub)
      void emit_binop(ir_function& func, const ir_inst& inst, uint32_t opcode, bool is32) {
         // Try const-immediate for add/sub
         bool is_add = (opcode == 0x0B000000 || opcode == 0x8B000000);
         bool is_sub = (opcode == 0x4B000000 || opcode == 0xCB000000);
         if (is_add || is_sub) {
            int64_t cval;
            if (try_get_const(inst.rr.src2, cval)) {
               uint32_t uval = static_cast<uint32_t>(cval & (is32 ? 0xFFFFFFFF : cval));
               if (uval <= 4095) {
                  load_vreg_x0(inst.rr.src1);
                  if (is_add) {
                     if (is32) emit32(0x11000000 | (uval << 10) | (X0 << 5) | X0);
                     else      emit32(0x91000000 | (uval << 10) | (X0 << 5) | X0);
                  } else {
                     if (is32) emit32(0x51000000 | (uval << 10) | (X0 << 5) | X0);
                     else      emit32(0xD1000000 | (uval << 10) | (X0 << 5) | X0);
                  }
                  kill_const_if_single_use(inst.rr.src2);
                  store_x0_vreg(inst.dest);
                  return;
               }
            }
         }
         // General case
         load_vreg_x0(inst.rr.src1);
         load_vreg_x1(inst.rr.src2);
         emit32(opcode | (X1 << 16) | (X0 << 5) | X0);
         store_x0_vreg(inst.dest);
      }

      // 3-operand (MUL = MADD Xd, Xn, Xm, XZR)
      void emit_binop3(const ir_inst& inst, uint32_t opcode, bool is32) {
         (void)is32;
         load_vreg_x0(inst.rr.src1);
         load_vreg_x1(inst.rr.src2);
         emit32(opcode | (X1 << 16) | (X0 << 5) | X0);
         store_x0_vreg(inst.dest);
      }

      // Simple 2-register binary: OP Xd, Xn, Xm
      void emit_binop_simple(const ir_inst& inst, uint32_t opcode, bool is32) {
         (void)is32;
         load_vreg_x0(inst.rr.src1);
         load_vreg_x1(inst.rr.src2);
         emit32(opcode | (X1 << 16) | (X0 << 5) | X0);
         store_x0_vreg(inst.dest);
      }

      // Rotate left = rotate right by (32/64 - amount)
      void emit_i32_rotl(const ir_inst& inst) {
         load_vreg_x0(inst.rr.src1);
         load_vreg_x1(inst.rr.src2);
         // NEG W1, W1 (SUB W1, WZR, W1)
         emit32(0x4B0103E1);
         // RORV W0, W0, W1
         emit32(0x1AC12C00);
         store_x0_vreg(inst.dest);
      }
      void emit_i64_rotl(const ir_inst& inst) {
         load_vreg_x0(inst.rr.src1);
         load_vreg_x1(inst.rr.src2);
         emit32(0xCB0103E1); // NEG X1, X1
         emit32(0x9AC12C00); // RORV X0, X0, X1
         store_x0_vreg(inst.dest);
      }

      // ── Remainder helpers ──
      void emit_i32_rem_s(const ir_inst& inst) {
         load_vreg_x0(inst.rr.src1);
         load_vreg_x1(inst.rr.src2);
         // Handle -1 divisor: rem(x, -1) = 0
         emit32(0x3100043F); // CMN W1, #1 (CMP W1, #-1)
         void* skip = emit_cond_branch_placeholder(COND_NE);
         emit_mov_imm32(X0, 0);
         void* done = emit_branch_placeholder();
         fix_branch(skip, code);
         emit32(0x1AC10C08); // SDIV W8, W0, W1
         emit32(0x1B018100); // MSUB W0, W8, W1, W0 (rem = a - (a/b)*b)
         fix_branch(done, code);
         store_x0_vreg(inst.dest);
      }
      void emit_i32_rem_u(const ir_inst& inst) {
         load_vreg_x0(inst.rr.src1);
         load_vreg_x1(inst.rr.src2);
         emit32(0x1AC10808); // UDIV W8, W0, W1
         emit32(0x1B018100); // MSUB W0, W8, W1, W0
         store_x0_vreg(inst.dest);
      }
      void emit_i64_rem_s(const ir_inst& inst) {
         load_vreg_x0(inst.rr.src1);
         load_vreg_x1(inst.rr.src2);
         emit32(0xB100043F); // CMN X1, #1
         void* skip = emit_cond_branch_placeholder(COND_NE);
         emit_mov_imm64(X0, 0);
         void* done = emit_branch_placeholder();
         fix_branch(skip, code);
         emit32(0x9AC10C08); // SDIV X8, X0, X1
         emit32(0x9B018100); // MSUB X0, X8, X1, X0
         fix_branch(done, code);
         store_x0_vreg(inst.dest);
      }
      void emit_i64_rem_u(const ir_inst& inst) {
         load_vreg_x0(inst.rr.src1);
         load_vreg_x1(inst.rr.src2);
         emit32(0x9AC10808); // UDIV X8, X0, X1
         emit32(0x9B018100); // MSUB X0, X8, X1, X0
         store_x0_vreg(inst.dest);
      }

      // ── Comparison helpers ──

      void emit_eqz(ir_function& func, const ir_inst& inst, uint32_t idx, bool is32) {
         int8_t pr = get_phys(inst.rr.src1);
         uint32_t rn;
         if (pr >= 0) { rn = phys_to_reg(pr); }
         else { load_vreg_x0(inst.rr.src1); rn = X0; }

         if (is32) emit_cmp_imm32(rn, 0);
         else      emit_cmp_imm64(rn, 0);

         if ((inst.flags & IR_FUSE_NEXT) && emit_fused_branch(func, idx, COND_EQ)) return;
         emit_cset(X0, COND_EQ);
         store_x0_vreg(inst.dest);
      }

      void emit_cmp(ir_function& func, const ir_inst& inst, uint32_t idx, uint32_t cond, bool is32) {
         // Try const-immediate
         int64_t cval;
         if (try_get_const(inst.rr.src2, cval)) {
            uint32_t uval = static_cast<uint32_t>(is32 ? (cval & 0xFFFFFFFF) : cval);
            if (uval <= 4095) {
               load_vreg_x0(inst.rr.src1);
               if (is32) emit_cmp_imm32(X0, uval);
               else      emit_cmp_imm64(X0, uval);
               kill_const_if_single_use(inst.rr.src2);
               if ((inst.flags & IR_FUSE_NEXT) && emit_fused_branch(func, idx, cond)) return;
               emit_cset(X0, cond);
               store_x0_vreg(inst.dest);
               return;
            }
         }
         load_vreg_x0(inst.rr.src1);
         load_vreg_x1(inst.rr.src2);
         if (is32) emit_cmp_reg32(X0, X1);
         else      emit_cmp_reg64(X0, X1);
         if ((inst.flags & IR_FUSE_NEXT) && emit_fused_branch(func, idx, cond)) return;
         emit_cset(X0, cond);
         store_x0_vreg(inst.dest);
      }

      // ── Memory load/store helpers ──

      void emit_load_op(const ir_inst& inst, uint32_t load_op) {
         uint32_t uoffset = static_cast<uint32_t>(inst.ri.imm);
         load_vreg_x0(inst.ri.src1);
         emit_effective_addr(X0, X0, uoffset);
         emit32(load_op | (X0 << 5) | X0);
         store_x0_vreg(inst.dest);
      }

      void emit_store_op(const ir_inst& inst, uint32_t store_op) {
         uint32_t uoffset = static_cast<uint32_t>(inst.ri.imm);
         load_vreg_x0(inst.dest);  // value
         load_vreg_x1(inst.ri.src1); // addr
         emit_effective_addr(X1, X1, uoffset);
         emit32(store_op | (X1 << 5) | X0);
      }

      // ── Float helpers ──

      void emit_f32_round(const ir_inst& inst, uint32_t round_op) {
         load_vreg_x0(inst.rr.src1);
         emit32(0x1E270000); // FMOV S0, W0
         emit32(round_op);   // FRINTx S0, S0
         emit32(0x1E260000); // FMOV W0, S0
         store_x0_vreg(inst.dest);
      }
      void emit_f64_round(const ir_inst& inst, uint32_t round_op) {
         load_vreg_x0(inst.rr.src1);
         emit32(0x9E670000); // FMOV D0, X0
         emit32(round_op);
         emit32(0x9E660000); // FMOV X0, D0
         store_x0_vreg(inst.dest);
      }

      void emit_f32_binop(const ir_inst& inst, uint32_t op) {
         load_vreg_x0(inst.rr.src1);
         load_vreg_x1(inst.rr.src2);
         emit32(0x1E270000);             // FMOV S0, W0
         emit32(0x1E270000 | (X1 << 5) | 1); // FMOV S1, W1
         emit32(op | (1 << 16));         // OP S0, S0, S1
         emit32(0x1E260000);             // FMOV W0, S0
         store_x0_vreg(inst.dest);
      }
      void emit_f64_binop(const ir_inst& inst, uint32_t op) {
         load_vreg_x0(inst.rr.src1);
         load_vreg_x1(inst.rr.src2);
         emit32(0x9E670000);             // FMOV D0, X0
         emit32(0x9E670000 | (X1 << 5) | 1); // FMOV D1, X1
         emit32(op | (1 << 16));         // OP D0, D0, D1
         emit32(0x9E660000);             // FMOV X0, D0
         store_x0_vreg(inst.dest);
      }

      void emit_f32_cmp(const ir_inst& inst, uint32_t cond) {
         load_vreg_x0(inst.rr.src1);
         load_vreg_x1(inst.rr.src2);
         emit32(0x1E270000);             // FMOV S0, W0
         emit32(0x1E270000 | (X1 << 5) | 1); // FMOV S1, W1
         emit32(0x1E212000);             // FCMP S0, S1
         emit_cset(X0, cond);
         store_x0_vreg(inst.dest);
      }
      void emit_f64_cmp(const ir_inst& inst, uint32_t cond) {
         load_vreg_x0(inst.rr.src1);
         load_vreg_x1(inst.rr.src2);
         emit32(0x9E670000);             // FMOV D0, X0
         emit32(0x9E670000 | (X1 << 5) | 1); // FMOV D1, X1
         emit32(0x1E612000);             // FCMP D0, D1
         emit_cset(X0, cond);
         store_x0_vreg(inst.dest);
      }

      // ── Float/int conversion helpers ──

      void emit_fcvt(const ir_inst& inst, uint32_t cvt_op, bool src_is32, bool fp_is32) {
         load_vreg_x0(inst.rr.src1);
         if (fp_is32) emit32(0x1E270000); // FMOV S0, W0
         else         emit32(0x9E670000); // FMOV D0, X0
         emit32(cvt_op); // FCVTZx Wd/Xd, S0/D0
         store_x0_vreg(inst.dest);
      }

      void emit_icvt(const ir_inst& inst, uint32_t cvt_op, bool src_is32, bool fp_is32) {
         load_vreg_x0(inst.rr.src1);
         emit32(cvt_op); // xCVTF S0/D0, W0/X0
         if (fp_is32) emit32(0x1E260000); // FMOV W0, S0
         else         emit32(0x9E660000); // FMOV X0, D0
         store_x0_vreg(inst.dest);
      }

      // ──────── Function relocation ────────

      struct call_fixup {
         void* branch;
         call_fixup* next;
      };

      struct func_reloc {
         void* address = nullptr;
         call_fixup* pending = nullptr;
      };

      void init_relocations() {
         uint32_t total = _mod.get_functions_total();
         _relocs = _allocator.alloc<func_reloc>(total);
         _num_relocs = total;
         for (uint32_t i = 0; i < total; ++i) _relocs[i] = func_reloc{};
      }

      void register_call(void* branch_addr, uint32_t funcnum) {
         if (funcnum >= _num_relocs) return;
         auto& r = _relocs[funcnum];
         if (r.address) {
            fix_branch(branch_addr, r.address);
         } else {
            auto* fixup = _allocator.alloc<call_fixup>(1);
            fixup->branch = branch_addr;
            fixup->next = r.pending;
            r.pending = fixup;
         }
      }

      void start_function(void* func_start, uint32_t funcnum) {
         if (funcnum >= _num_relocs) return;
         auto& r = _relocs[funcnum];
         for (auto* f = r.pending; f; f = f->next) {
            fix_branch(f->branch, func_start);
         }
         r.address = func_start;
         r.pending = nullptr;
      }

      // ──────── Static callbacks ────────

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

      unsigned char* code = nullptr;
      growable_allocator& _allocator;
      module& _mod;
      void* _code_segment_base = nullptr;
      void* fpe_handler = nullptr;
      void* call_indirect_handler = nullptr;
      void* type_error_handler = nullptr;
      void* stack_overflow_handler = nullptr;
      void* memory_handler = nullptr;
      func_reloc* _relocs = nullptr;
      uint32_t _num_relocs = 0;
      void** _block_addrs = nullptr;
      block_fixup** _block_fixups = nullptr;
      uint32_t _num_blocks = 0;
      bool _in_br_table = false;
      uint32_t _br_table_case = 0;
      uint32_t _br_table_size = 0;
      int8_t* _vreg_map = nullptr;
      int16_t* _spill_map = nullptr;
      uint32_t _num_vregs = 0;
      uint32_t _num_spill_slots = 0;
      uint32_t _body_locals = 0;
      bool _use_regalloc = false;
      uint32_t _callee_saved_used = 0;
      uint32_t _callee_saved_count = 0;
      uint32_t* _func_def_inst = nullptr;
      uint16_t* _func_use_count = nullptr;
      ir_inst*  _func_insts = nullptr;
      uint32_t  _func_inst_count = 0;
   };

}} // namespace eosio::vm

#endif // __aarch64__
