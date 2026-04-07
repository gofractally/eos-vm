#pragma once

// Reusable x86_64 encoding infrastructure for JIT code generators.
// Extracted from x86_64.hpp so that both the original machine_code_writer
// and the new two-pass jit_codegen can share the same encoding logic.

#include <eosio/vm/exceptions.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace eosio { namespace vm {

   // x86_64 assembler base class (CRTP).
   // Provides register types, instruction constant definitions,
   // REX/VEX prefix encoding, ModR/M+SIB+Disp encoding,
   // and the templated emit() system.
   //
   // Must be a template class because C++ requires static constexpr auto
   // members with template-dependent initializers to be in template classes.
   //
   // Usage: class my_emitter : public x86_64_base<my_emitter> { ... };
   //
   template<typename Derived = void>
   class x86_64_base {
    protected:
      unsigned char* code = nullptr;

    public:
      // ──────────────────── Register types ────────────────────
      enum class imm8 : uint8_t {};
      enum class imm32 : uint32_t {};

      enum general_register64 {
          rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi,
          r8, r9, r10, r11, r12, r13, r14, r15
      };
      enum general_register32 {
          eax, ecx, edx, ebx, esp, ebp, esi, edi,
          r8d, r9d, r10d, r11d, r12d, r13d, r14d, r15d
      };
      enum general_register16 {
          ax, cx, dx, bx, sp, bp, si, di,
          r8w, r9w, r10w, r11w, r12w, r13w, r14w, r15w
      };
      enum general_register8 {
          al, cl, dl, bl,
          r8b = 8, r9b, r10b, r11b, r12b, r13b, r14b, r15b
      };

      struct general_register {
         constexpr general_register(general_register32 reg) : reg(reg) {}
         constexpr general_register(general_register64 reg) : reg(reg) {}
         constexpr general_register(general_register16 reg) : reg(reg) {}
         constexpr general_register(general_register8 reg) : reg(reg) {}
         int reg;
      };

      // ──────────────────── Memory reference types ────────────────────
      struct simple_memory_ref { general_register64 reg; };
      inline friend simple_memory_ref operator*(general_register64 reg) { return { reg }; }

      struct register_add_expr { general_register64 reg; int32_t offset; };
      inline friend register_add_expr operator+(general_register64 reg, int32_t offset) { return { reg, offset }; }
      inline friend register_add_expr operator+(int32_t offset, general_register64 reg) { return { reg, offset }; }
      inline friend register_add_expr operator-(general_register64 reg, int32_t offset) { return { reg, -offset }; }

      struct disp_memory_ref {
         constexpr disp_memory_ref(general_register64 reg, int32_t offset) : reg(reg), offset(offset) {}
         constexpr disp_memory_ref(simple_memory_ref other) : reg(other.reg), offset(0) {}
         general_register64 reg;
         int32_t offset;
      };
      inline friend disp_memory_ref operator*(register_add_expr expr) { return { expr.reg, expr.offset }; }

      struct double_register_expr { general_register64 reg1; general_register64 reg2; int32_t offset; };
      friend double_register_expr operator+(general_register64 reg1, general_register64 reg2) { return { reg1, reg2, 0 }; }
      friend double_register_expr operator+(double_register_expr expr, int32_t offset) { return {expr.reg1, expr.reg2, expr.offset + offset}; }
      friend double_register_expr operator-(double_register_expr expr, int32_t offset) { return {expr.reg1, expr.reg2, expr.offset - offset}; }

      struct sib_memory_ref {
         general_register64 reg1;
         general_register64 reg2;
         int32_t offset;
      };
      friend sib_memory_ref operator*(double_register_expr expr) { return { expr.reg1, expr.reg2, expr.offset }; }

      template <typename T, int Sz>
      struct sized_memory_ref { T value; };

      template <typename T>
      static sized_memory_ref<T, 4> dword_ptr(const T& expr) { return { expr }; }
      template <typename T>
      static sized_memory_ref<T, 8> qword_ptr(const T& expr) { return { expr }; }

      template <typename T, int Sz>
      friend auto operator*(const sized_memory_ref<T, Sz>& expr) {
         return sized_memory_ref<decltype(*expr.value), Sz>{*expr.value};
      }

      enum xmm_register {
          xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7,
          xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15
      };

      struct any_register {
         constexpr any_register(general_register32 reg) : reg(reg) {}
         constexpr any_register(general_register64 reg) : reg(reg) {}
         constexpr any_register(general_register16 reg) : reg(reg) {}
         constexpr any_register(general_register8 reg) : reg(reg) {}
         constexpr any_register(general_register reg) : reg(reg.reg) {}
         constexpr any_register(xmm_register reg) : reg(reg) {}
         int reg;
      };

      // ──────────────────── Instruction encoding types ────────────────────
      template<int W, int N>
      struct IA32_t { uint8_t opcode[N]; };
      template<typename Base>
      struct IA32_opext { Base base; int opext; };
      template<int W, int N>
      inline constexpr friend IA32_opext<IA32_t<W, N>> operator/(IA32_t<W, N> base, int opext) {
         return {base, opext};
      }

      template<typename Base, typename T>
      struct IA32_imm { Base base; };
      template<typename T, typename B>
      static constexpr IA32_imm<B, T> add_immediate(B b) { return {b}; }

      template<typename Base, typename T>
      inline constexpr friend auto operator/(IA32_imm<Base, T> base, int opext) {
         return add_immediate<T>(base.base/opext);
      }

      template<typename... Op>
      static constexpr auto IA32(Op... op) {
         return IA32_t<0, sizeof...(Op)>{{ static_cast<uint8_t>(op)... }};
      }
      template<typename... Op>
      static constexpr auto IA32_REX_W(Op... op) {
         return IA32_t<1, sizeof...(Op)>{{ static_cast<uint8_t>(op)... }};
      }
      template<typename... Op>
      static constexpr auto IA32_WX(Op... op) {
         return IA32_t<-1, sizeof...(Op)>{{ static_cast<uint8_t>(op)... }};
      }
      template<typename... Op>
      static constexpr auto IA32_WX_imm8(Op... op) {
         return IA32_imm<IA32_t<-1, sizeof...(Op)>, imm8>{{{ static_cast<uint8_t>(op)... }}};
      }
      template<typename... Op>
      static constexpr auto IA32_WX_imm32(Op... op) {
         return IA32_imm<IA32_t<-1, sizeof...(Op)>, imm32>{{{ static_cast<uint8_t>(op)... }}};
      }

      struct Jcc { uint8_t opcode; };

      // ──────────────────── Instruction constants ────────────────────
      static constexpr auto ADD_A = IA32_WX(0x03);
      static constexpr auto ADD_B = IA32_WX(0x01);
      static constexpr auto ADD_imm8 = IA32_WX_imm8(0x83)/0;
      static constexpr auto ADD_imm32 = IA32_WX_imm32(0x81)/0;
      static constexpr auto AND_A = IA32_WX(0x23);
      static constexpr auto AND_imm8 = IA32_WX_imm8(0x83)/4;
      static constexpr auto AND_imm32 = IA32_WX_imm32(0x81)/4;
      static constexpr auto BSF = IA32_WX(0x0f, 0xbc);
      static constexpr auto BSR = IA32_WX(0x0f, 0xbd);
      static constexpr auto CALL = IA32(0xff)/2;
      static constexpr auto CDQ = IA32(0x99);
      static constexpr auto CQO = IA32_REX_W(0x99);
      static constexpr auto CLD = IA32(0xFC);
      static constexpr auto CMOVNZ = IA32_WX(0x0f, 0x45);
      static constexpr auto CMOVZ = IA32_WX(0x0f, 0x44);
      static constexpr auto CMP = IA32_WX(0x3b);
      static constexpr auto CMP_imm8 = IA32_WX_imm8(0x83)/7;
      static constexpr auto CMP_imm32 = IA32_WX_imm32(0x81)/7;
      static constexpr auto DEC = IA32_WX(0xFF)/1;
      static constexpr auto DECD = IA32(0xFF)/1;
      static constexpr auto DECQ = IA32_REX_W(0xFF)/1;
      static constexpr auto DIV = IA32_WX(0xf7)/6;
      static constexpr auto IDIV = IA32_WX(0xf7)/7;
      static constexpr auto IMUL = IA32_WX(0x0f, 0xaf);
      static constexpr auto INC = IA32_WX(0xFF)/0;
      static constexpr auto INCD = IA32(0xFF)/0;
      static constexpr auto JB = Jcc{0x72};
      static constexpr auto JAE = Jcc{0x73};
      static constexpr auto JE = Jcc{0x74};
      static constexpr auto JZ = Jcc{0x74};
      static constexpr auto JNE = Jcc{0x75};
      static constexpr auto JNZ = Jcc{0x75};
      static constexpr auto JBE = Jcc{0x76};
      static constexpr auto JA = Jcc{0x77};
      static constexpr auto JL = Jcc{0x7c};
      static constexpr auto JGE = Jcc{0x7d};
      static constexpr auto JLE = Jcc{0x7e};
      static constexpr auto JG = Jcc{0x7f};
      static constexpr auto JMP_8 = IA32(0xeb);
      static constexpr auto JMP_32 = IA32(0xe9);
      static constexpr auto LDMXCSR = IA32(0x0f, 0xae)/2;
      static constexpr auto LEA = IA32_WX(0x8d);
      static constexpr auto LEAVE = IA32(0xc9);
      static constexpr auto MOVB_A = IA32(0x8a);
      static constexpr auto MOVB_B = IA32(0x88);
      static constexpr auto MOVW_B = IA32(0x66, 0x89);
      static constexpr auto MOV_A = IA32_WX(0x8b);
      static constexpr auto MOV_B = IA32_WX(0x89);
      static constexpr auto MOVZXB = IA32(0x0f, 0xb6);
      static constexpr auto MOVZXW = IA32_WX(0x0f, 0xb7);
      static constexpr auto MOVSXB = IA32_WX(0x0f, 0xbe);
      static constexpr auto MOVSXW = IA32_WX(0x0f, 0xbf);
      static constexpr auto MOVSXD = IA32_WX(0x63);
      static constexpr auto NEG = IA32_WX(0xf7)/3;
      static constexpr auto NOT = IA32_WX(0xf7)/2;
      static constexpr auto LZCNT = IA32_WX(0xf3, 0x0f, 0xbd);
      static constexpr auto OR_A = IA32_WX(0x0B);
      static constexpr auto OR_imm8 = IA32_WX_imm8(0x83)/1;
      static constexpr auto OR_imm32 = IA32_WX_imm32(0x81)/1;
      static constexpr auto POPCNT = IA32_WX(0xf3, 0x0f, 0xb8);
      static constexpr auto ROL_cl = IA32_WX(0xd3)/0;
      static constexpr auto ROL_imm8 = IA32_WX(0xc1)/0;
      static constexpr auto ROR_cl = IA32_WX(0xd3)/1;
      static constexpr auto ROR_imm8 = IA32_WX(0xc1)/1;
      static constexpr auto SETZ = IA32(0x0f, 0x94);
      static constexpr auto SETNZ = IA32(0x0f, 0x95);
      static constexpr auto SAR_cl = IA32_WX(0xd3)/7;
      static constexpr auto SAR_imm8 = IA32_WX(0xc1)/7;
      static constexpr auto SHL_cl = IA32_WX(0xd3)/4;
      static constexpr auto SHL_imm8 = IA32_WX(0xc1)/4;
      static constexpr auto SHR_cl = IA32_WX(0xd3)/5;
      static constexpr auto SHR_imm8 = IA32_WX_imm8(0xc1)/5;
      static constexpr auto STD = IA32(0xFD);
      static constexpr auto STMXCSR = IA32(0x0f, 0xae)/3;
      static constexpr auto SUB_A = IA32_WX(0x2b);
      static constexpr auto SUB_imm8 = IA32_WX_imm8(0x83)/5;
      static constexpr auto SUB_imm32 = IA32_WX_imm32(0x81)/5;
      static constexpr auto TZCNT = IA32_WX(0xf3, 0x0f, 0xbc);
      static constexpr auto RET = IA32(0xc3);
      static constexpr auto TEST = IA32_WX(0x85);
      static constexpr auto XOR_A = IA32_WX(0x33);
      static constexpr auto XOR_imm8 = IA32_WX_imm8(0x83)/6;
      static constexpr auto XOR_imm32 = IA32_WX_imm32(0x81)/6;

      static constexpr auto reverse_condition(Jcc opcode) {
         switch(opcode.opcode) {
         case JA.opcode: return JBE;
         case JBE.opcode: return JA;
         case JB.opcode: return JAE;
         case JAE.opcode: return JB;
         case JE.opcode: return JNE;
         case JNE.opcode: return JE;
         case JL.opcode: return JGE;
         case JGE.opcode: return JL;
         case JG.opcode: return JLE;
         case JLE.opcode: return JG;
         default: EOS_VM_ASSERT(false, wasm_parse_exception, "Sorry, not implemented.");
         }
         __builtin_unreachable();
      }

      // ──────────────────── VEX encoding types ────────────────────
      enum VEX_mmmm { mmmm_none, mmmm_0F, mmmm_0F38, mmmm_0F3A };
      enum VEX_pp { pp_none, pp_66, pp_F3, pp_F2 };

      template<int SZ, VEX_pp PP, VEX_mmmm MMMM, int W, int OpExt = -1>
      struct VEX { uint8_t opcode; };
      using VEX_128_66_0F_W0 = VEX<128, pp_66, mmmm_0F, 0>;
      using VEX_128_66_0F = VEX_128_66_0F_W0;
      using VEX_128_66_0F_WIG = VEX_128_66_0F_W0;
      using VEX_128_66_0F_W1 = VEX<128, pp_66, mmmm_0F, 1>;
      using VEX_128_66_0F38_W0 = VEX<128, pp_66, mmmm_0F38, 0>;
      using VEX_128_66_0F38_WIG = VEX_128_66_0F38_W0;
      using VEX_128_66_0F38 = VEX_128_66_0F38_W0;
      using VEX_128_0F_W0 = VEX<128, pp_none, mmmm_0F, 0>;
      using VEX_128_0F_WIG = VEX_128_0F_W0;
      using VEX_128_66_0F3A_W0 = VEX<128, pp_66, mmmm_0F3A, 0>;
      using VEX_128_66_0F3A_W1 = VEX<128, pp_66, mmmm_0F3A, 1>;
      using VEX_128_F3_0F_W0 = VEX<128, pp_F3, mmmm_0F, 0>;
      using VEX_128_F3_0F_WIG = VEX_128_F3_0F_W0;

      // VEX-encoded instruction constants
      static constexpr auto VMOVD_A = VEX_128_66_0F_W0{0x6e};
      static constexpr auto VMOVD_B = VEX_128_66_0F_W0{0x7e};
      static constexpr auto VMOVQ_A = VEX_128_66_0F_W1{0x6e};
      static constexpr auto VMOVQ_B = VEX_128_66_0F_W1{0x7e};
      static constexpr auto VMOVDQU_A = VEX_128_F3_0F_WIG{0x6f};
      static constexpr auto VMOVDQU_B = VEX_128_F3_0F_WIG{0x7f};
      static constexpr auto VMOVUPS_A = VEX_128_66_0F_WIG{0x10};
      static constexpr auto VMOVUPS_B = VEX_128_66_0F_WIG{0x11};
      static constexpr auto VMOVMSKPS = VEX_128_0F_WIG{0x50};
      static constexpr auto VMOVMSKPD = VEX_128_66_0F_WIG{0x50};
      static constexpr auto VPABSB = VEX_128_66_0F38_WIG{0x1c};
      static constexpr auto VPABSW = VEX_128_66_0F38_WIG{0x1d};
      static constexpr auto VPABSD = VEX_128_66_0F38_WIG{0x1e};
      static constexpr auto VPACKSSWB = VEX_128_66_0F_WIG{0x63};
      static constexpr auto VPACKSSDW = VEX_128_66_0F_WIG{0x6b};
      static constexpr auto VPACKUSWB = VEX_128_66_0F_WIG{0x67};
      static constexpr auto VPACKUSDW = VEX_128_66_0F38{0x2B};
      static constexpr auto VPADDB = VEX_128_66_0F_WIG{0xfc};
      static constexpr auto VPADDW = VEX_128_66_0F_WIG{0xfd};
      static constexpr auto VPADDD = VEX_128_66_0F_WIG{0xfe};
      static constexpr auto VPADDQ = VEX_128_66_0F_WIG{0xd4};
      static constexpr auto VPADDSB = VEX_128_66_0F_WIG{0xec};
      static constexpr auto VPADDSW = VEX_128_66_0F_WIG{0xed};
      static constexpr auto VPADDUSB = VEX_128_66_0F_WIG{0xdc};
      static constexpr auto VPADDUSW = VEX_128_66_0F_WIG{0xdd};
      static constexpr auto VPAND = VEX_128_66_0F_WIG{0xdb};
      static constexpr auto VPANDN = VEX_128_66_0F_WIG{0xdf};
      static constexpr auto VPAVGB = VEX_128_66_0F_WIG{0xe0};
      static constexpr auto VPAVGW = VEX_128_66_0F_WIG{0xe3};
      static constexpr auto VPBROADCASTB = VEX_128_66_0F38_W0{0x78};
      static constexpr auto VPBROADCASTW = VEX_128_66_0F38_W0{0x79};
      static constexpr auto VPBROADCASTD = VEX_128_66_0F38_W0{0x58};
      static constexpr auto VPBROADCASTQ = VEX_128_66_0F38_W0{0x59};
      static constexpr auto VPCMPEQB = VEX_128_66_0F_WIG{0x74};
      static constexpr auto VPCMPEQW = VEX_128_66_0F_WIG{0x75};
      static constexpr auto VPCMPEQD = VEX_128_66_0F_WIG{0x76};
      static constexpr auto VPCMPEQQ = VEX_128_66_0F38_WIG{0x29};
      static constexpr auto VPCMPGTB = VEX_128_66_0F_WIG{0x64};
      static constexpr auto VPCMPGTW = VEX_128_66_0F_WIG{0x65};
      static constexpr auto VPCMPGTD = VEX_128_66_0F_WIG{0x66};
      static constexpr auto VPCMPGTQ = VEX_128_66_0F38_WIG{0x37};
      static constexpr auto VPEXTRB = VEX_128_66_0F3A_W0{0x14};
      static constexpr auto VPEXTRW = VEX_128_66_0F3A_W0{0x15};
      static constexpr auto VPEXTRD = VEX_128_66_0F3A_W0{0x16};
      static constexpr auto VPEXTRQ = VEX_128_66_0F3A_W1{0x16};
      static constexpr auto VPINSRB = VEX_128_66_0F3A_W0{0x20};
      static constexpr auto VPINSRW = VEX_128_66_0F_W0{0xc4};
      static constexpr auto VPINSRD = VEX_128_66_0F3A_W0{0x22};
      static constexpr auto VPINSRQ = VEX_128_66_0F3A_W1{0x22};
      static constexpr auto VPHADDW = VEX_128_66_0F38_WIG{0x01};
      static constexpr auto VPHADDD = VEX_128_66_0F38_WIG{0x02};
      static constexpr auto VPMADDWD = VEX_128_66_0F_WIG{0xf5};
      static constexpr auto VPMAXSB = VEX_128_66_0F38_WIG{0x3c};
      static constexpr auto VPMAXSW = VEX_128_66_0F_WIG{0xee};
      static constexpr auto VPMAXSD = VEX_128_66_0F38_WIG{0x3d};
      static constexpr auto VPMAXUB = VEX_128_66_0F{0xde};
      static constexpr auto VPMAXUW = VEX_128_66_0F38{0x3e};
      static constexpr auto VPMAXUD = VEX_128_66_0F38_WIG{0x3f};
      static constexpr auto VPMINSB = VEX_128_66_0F38{0x38};
      static constexpr auto VPMINSW = VEX_128_66_0F{0xea};
      static constexpr auto VPMINSD = VEX_128_66_0F38_WIG{0x39};
      static constexpr auto VPMINUB = VEX_128_66_0F{0xda};
      static constexpr auto VPMINUW = VEX_128_66_0F38{0x3a};
      static constexpr auto VPMINUD = VEX_128_66_0F38_WIG{0x3b};
      static constexpr auto VPMOVMSKB = VEX_128_66_0F_WIG{0xD7};
      static constexpr auto VPMOVSXBW = VEX_128_66_0F38_WIG{0x20};
      static constexpr auto VPMOVSXWD = VEX_128_66_0F38_WIG{0x23};
      static constexpr auto VPMOVSXDQ = VEX_128_66_0F38_WIG{0x25};
      static constexpr auto VPMOVZXBW = VEX_128_66_0F38_WIG{0x30};
      static constexpr auto VPMOVZXWD = VEX_128_66_0F38_WIG{0x33};
      static constexpr auto VPMOVZXDQ = VEX_128_66_0F38_WIG{0x35};
      static constexpr auto VPMULHW = VEX_128_66_0F_WIG{0xe5};
      static constexpr auto VPMULHUW = VEX_128_66_0F_WIG{0xe4};
      static constexpr auto VPMULHRSW = VEX_128_66_0F38_WIG{0x0b};
      static constexpr auto VPMULLW = VEX_128_66_0F_WIG{0xd5};
      static constexpr auto VPMULLD = VEX_128_66_0F38_WIG{0x40};
      static constexpr auto VPMULDQ = VEX_128_66_0F38_WIG{0x28};
      static constexpr auto VPMULUDQ = VEX_128_66_0F_WIG{0xf4};
      static constexpr auto VPOR = VEX_128_66_0F_WIG{0xeb};
      static constexpr auto VPSHUFB = VEX_128_66_0F38_WIG{0x00};
      static constexpr auto VPSHUFD = VEX_128_66_0F_WIG{0x70};
      static constexpr auto VPSLLW = VEX_128_66_0F_WIG{0xf1};
      static constexpr auto VPSLLW_c = VEX<128, pp_66, mmmm_0F, 0, 6>{0x71};
      static constexpr auto VPSLLD = VEX_128_66_0F_WIG{0xf2};
      static constexpr auto VPSLLQ = VEX_128_66_0F_WIG{0xf3};
      static constexpr auto VPSRAW = VEX_128_66_0F_WIG{0xe1};
      static constexpr auto VPSRAW_c = VEX<128, pp_66, mmmm_0F, 0, 4>{0x71};
      static constexpr auto VPSRAD = VEX_128_66_0F_WIG{0xe2};
      static constexpr auto VPSRLDQ_c = VEX<128, pp_66, mmmm_0F, 0, 3>{0x73};
      static constexpr auto VPSRLW = VEX_128_66_0F_WIG{0xd1};
      static constexpr auto VPSRLD = VEX_128_66_0F_WIG{0xd2};
      static constexpr auto VPSRLQ = VEX_128_66_0F_WIG{0xd3};
      static constexpr auto VPSRLQ_c = VEX<128, pp_66, mmmm_0F, 0, 2>{0x73};
      static constexpr auto VPSUBB = VEX_128_66_0F_WIG{0xf8};
      static constexpr auto VPSUBW = VEX_128_66_0F_WIG{0xf9};
      static constexpr auto VPSUBD = VEX_128_66_0F_WIG{0xfa};
      static constexpr auto VPSUBQ = VEX_128_66_0F_WIG{0xfb};
      static constexpr auto VPSUBSB = VEX_128_66_0F_WIG{0xe8};
      static constexpr auto VPSUBSW = VEX_128_66_0F_WIG{0xe9};
      static constexpr auto VPSUBUSB = VEX_128_66_0F_WIG{0xd8};
      static constexpr auto VPSUBUSW = VEX_128_66_0F_WIG{0xd9};
      static constexpr auto VPTEST = VEX_128_66_0F38_WIG{0x17};
      static constexpr auto VPUNPCKHBW = VEX_128_66_0F_WIG{0x68};
      static constexpr auto VPUNPCKHWD = VEX_128_66_0F_WIG{0x69};
      static constexpr auto VPUNPCKHDQ = VEX_128_66_0F_WIG{0x6a};
      static constexpr auto VPUNPCKLBW = VEX_128_66_0F_WIG{0x60};
      static constexpr auto VPUNPCKLWD = VEX_128_66_0F_WIG{0x61};
      static constexpr auto VPUNPCKLDQ = VEX_128_66_0F_WIG{0x62};
      static constexpr auto VPXOR = VEX_128_66_0F_WIG{0xef};

      // Scalar SSE via VEX
      static constexpr auto VADDSS = VEX<128, pp_F3, mmmm_0F, 0>{0x58};
      static constexpr auto VADDSD = VEX<128, pp_F2, mmmm_0F, 0>{0x58};
      static constexpr auto VSUBSS = VEX<128, pp_F3, mmmm_0F, 0>{0x5c};
      static constexpr auto VSUBSD = VEX<128, pp_F2, mmmm_0F, 0>{0x5c};
      static constexpr auto VMULSS = VEX<128, pp_F3, mmmm_0F, 0>{0x59};
      static constexpr auto VMULSD = VEX<128, pp_F2, mmmm_0F, 0>{0x59};
      static constexpr auto VDIVSS = VEX<128, pp_F3, mmmm_0F, 0>{0x5e};
      static constexpr auto VDIVSD = VEX<128, pp_F2, mmmm_0F, 0>{0x5e};
      static constexpr auto VSQRTSS = VEX<128, pp_F3, mmmm_0F, 0>{0x51};
      static constexpr auto VSQRTSD = VEX<128, pp_F2, mmmm_0F, 0>{0x51};
      static constexpr auto VMINSS = VEX<128, pp_F3, mmmm_0F, 0>{0x5d};
      static constexpr auto VMINSD = VEX<128, pp_F2, mmmm_0F, 0>{0x5d};
      static constexpr auto VMAXSS = VEX<128, pp_F3, mmmm_0F, 0>{0x5f};
      static constexpr auto VMAXSD = VEX<128, pp_F2, mmmm_0F, 0>{0x5f};

      // Packed float via VEX
      static constexpr auto VADDPS = VEX<128, pp_none, mmmm_0F, 0>{0x58};
      static constexpr auto VADDPD = VEX<128, pp_66, mmmm_0F, 0>{0x58};
      static constexpr auto VSUBPS = VEX<128, pp_none, mmmm_0F, 0>{0x5c};
      static constexpr auto VSUBPD = VEX<128, pp_66, mmmm_0F, 0>{0x5c};
      static constexpr auto VMULPS = VEX<128, pp_none, mmmm_0F, 0>{0x59};
      static constexpr auto VMULPD = VEX<128, pp_66, mmmm_0F, 0>{0x59};
      static constexpr auto VDIVPS = VEX<128, pp_none, mmmm_0F, 0>{0x5e};
      static constexpr auto VDIVPD = VEX<128, pp_66, mmmm_0F, 0>{0x5e};
      static constexpr auto VSQRTPS = VEX<128, pp_none, mmmm_0F, 0>{0x51};
      static constexpr auto VSQRTPD = VEX<128, pp_66, mmmm_0F, 0>{0x51};
      static constexpr auto VMINPS = VEX<128, pp_none, mmmm_0F, 0>{0x5d};
      static constexpr auto VMINPD = VEX<128, pp_66, mmmm_0F, 0>{0x5d};
      static constexpr auto VMAXPS = VEX<128, pp_none, mmmm_0F, 0>{0x5f};
      static constexpr auto VMAXPD = VEX<128, pp_66, mmmm_0F, 0>{0x5f};
      static constexpr auto VANDPS = VEX<128, pp_none, mmmm_0F, 0>{0x54};
      static constexpr auto VANDPD = VEX<128, pp_66, mmmm_0F, 0>{0x54};
      static constexpr auto VANDNPS = VEX<128, pp_none, mmmm_0F, 0>{0x55};
      static constexpr auto VANDNPD = VEX<128, pp_66, mmmm_0F, 0>{0x55};
      static constexpr auto VORPS = VEX<128, pp_none, mmmm_0F, 0>{0x56};
      static constexpr auto VORPD = VEX<128, pp_66, mmmm_0F, 0>{0x56};
      static constexpr auto VXORPS = VEX<128, pp_none, mmmm_0F, 0>{0x57};
      static constexpr auto VXORPD = VEX<128, pp_66, mmmm_0F, 0>{0x57};

      static constexpr auto VCMPPS = VEX<128, pp_none, mmmm_0F, 0>{0xc2};
      static constexpr auto VCMPPD = VEX<128, pp_66, mmmm_0F, 0>{0xc2};
      static constexpr auto VCMPSS = VEX<128, pp_F3, mmmm_0F, 0>{0xc2};
      static constexpr auto VCMPSD = VEX<128, pp_F2, mmmm_0F, 0>{0xc2};

      static constexpr auto VCVTDQ2PS = VEX<128, pp_none, mmmm_0F, 0>{0x5b};
      static constexpr auto VCVTDQ2PD = VEX<128, pp_F3, mmmm_0F, 0>{0xe6};
      static constexpr auto VCVTPS2PD = VEX<128, pp_none, mmmm_0F, 0>{0x5a};
      static constexpr auto VCVTPD2PS = VEX<128, pp_66, mmmm_0F, 0>{0x5a};
      static constexpr auto VCVTTPS2DQ = VEX<128, pp_F3, mmmm_0F, 0>{0x5b};
      static constexpr auto VCVTTPD2DQ = VEX<128, pp_66, mmmm_0F, 0>{0xe6};

      static constexpr auto VCVTSS2SD = VEX<128, pp_F3, mmmm_0F, 0>{0x5a};
      static constexpr auto VCVTSD2SS = VEX<128, pp_F2, mmmm_0F, 0>{0x5a};
      static constexpr auto VCVTSI2SS = VEX<128, pp_F3, mmmm_0F, 0>{0x2a};
      static constexpr auto VCVTSI2SD = VEX<128, pp_F2, mmmm_0F, 0>{0x2a};
      static constexpr auto VCVTTSS2SI = VEX<128, pp_F3, mmmm_0F, 0>{0x2c};
      static constexpr auto VCVTTSD2SI = VEX<128, pp_F2, mmmm_0F, 0>{0x2c};

      static constexpr auto VROUNDSS = VEX<128, pp_66, mmmm_0F3A, 0>{0x0a};
      static constexpr auto VROUNDSD = VEX<128, pp_66, mmmm_0F3A, 0>{0x0b};
      static constexpr auto VROUNDPS = VEX<128, pp_66, mmmm_0F3A, 0>{0x08};
      static constexpr auto VROUNDPD = VEX<128, pp_66, mmmm_0F3A, 0>{0x09};

      static constexpr auto VSHUFPS = VEX<128, pp_none, mmmm_0F, 0>{0xc6};
      static constexpr auto VSHUFPD = VEX<128, pp_66, mmmm_0F, 0>{0xc6};
      static constexpr auto VUNPCKLPS = VEX<128, pp_none, mmmm_0F, 0>{0x14};
      static constexpr auto VUNPCKHPS = VEX<128, pp_none, mmmm_0F, 0>{0x15};
      static constexpr auto VUNPCKLPD = VEX<128, pp_66, mmmm_0F, 0>{0x14};
      static constexpr auto VUNPCKHPD = VEX<128, pp_66, mmmm_0F, 0>{0x15};

      static constexpr auto VBLENDVPS = VEX<128, pp_66, mmmm_0F3A, 0>{0x4a};
      static constexpr auto VPBLENDVB = VEX<128, pp_66, mmmm_0F3A, 0>{0x4c};

      // ──────────────────── Low-level byte emission ────────────────────
      void emit_byte(uint8_t val) { *code++ = val; }
      void emit_bytes() {}
      template<class... T>
      void emit_bytes(uint8_t val0, T... vals) {
         emit_byte(val0);
         emit_bytes(vals...);
      }
      void emit_operand(imm8 val) { emit_byte(static_cast<uint8_t>(val)); }
      void emit_operand(imm32 val) { emit_operand32(static_cast<uint32_t>(val)); }
      void emit_operand16(uint16_t val) { memcpy(code, &val, sizeof(val)); code += sizeof(val); }
      void emit_operand32(uint32_t val) { memcpy(code, &val, sizeof(val)); code += sizeof(val); }
      void emit_operand64(uint64_t val) { memcpy(code, &val, sizeof(val)); code += sizeof(val); }
      void emit_operandf32(float val) { memcpy(code, &val, sizeof(val)); code += sizeof(val); }
      void emit_operandf64(double val) { memcpy(code, &val, sizeof(val)); code += sizeof(val); }
      template<class T>
      void emit_operand_ptr(T* val) { memcpy(code, &val, sizeof(val)); code += sizeof(val); }

      // ──────────────────── REX prefix encoding ────────────────────
      void emit_REX_prefix(bool W, bool R, bool X, bool B) {
         if(W || R || X || B) {
            emit_bytes(0x40 | (W << 3) | (R << 2) | (X << 1) | (B << 0));
         }
      }
      void emit_REX_prefix(bool W, general_register r_m, int reg) {
         emit_REX_prefix(W, reg & 8, false, r_m.reg & 8);
      }
      void emit_REX_prefix(bool W, disp_memory_ref mem, int reg) {
         emit_REX_prefix(W, reg & 8, false, mem.reg & 8);
      }
      void emit_REX_prefix(bool W, sib_memory_ref mem, int reg) {
         emit_REX_prefix(W, reg & 8, mem.reg1 & 8, mem.reg2 & 8);
      }
      template <typename T, int Sz>
      void emit_REX_prefix(bool W, sized_memory_ref<T, Sz> mem, int reg) {
         emit_REX_prefix(W, mem.value, reg);
      }

      // ──────────────────── VEX prefix encoding ────────────────────
      void emit_VEX_prefix(bool R, bool X, bool B, VEX_mmmm mmmm, bool W, int vvvv, bool L, VEX_pp pp) {
         if(X || B || (mmmm != mmmm_0F) || W) {
            emit_bytes(0xc4, (!R << 7) | (!X << 6) | (!B << 5) | mmmm, (W << 7) | ((vvvv ^ 0xF) << 3) | (L << 2) | pp);
         } else {
            emit_bytes(0xc5, (!R << 7) | ((vvvv ^ 0xF) << 3) | (L << 2) | pp);
         }
      }

      // ──────────────────── ModR/M + SIB + Disp encoding ────────────────────
      void emit_modrm_sib_disp(sib_memory_ref mem, int reg) {
         if (mem.reg1 == rsp || mem.reg2 == rbp) { EOS_VM_ASSERT(false, wasm_parse_exception, "Sorry, not implemented."); }
         auto sib = ((mem.reg1 & 7) << 3) | (mem.reg2 & 7);
         if(mem.offset == 0) {
            emit_bytes(0x00 | ((reg & 7) << 3) | 0x4, sib);
         } else if(mem.offset >= -0x80 && mem.offset <= 0x7f) {
            emit_bytes(0x40 | ((reg & 7) << 3) | 0x4, sib, mem.offset);
         } else {
            emit_bytes(0x80 | ((reg & 7) << 3) | 0x4, sib);
            emit_operand32(mem.offset);
         }
      }
      void emit_modrm_sib_disp(disp_memory_ref mem, int reg) {
         if(mem.offset == 0) {
            return emit_modrm_sib_disp(simple_memory_ref{mem.reg}, reg);
         } else if(mem.offset >= -0x80 && mem.offset <= 0x7f) {
            if(mem.reg == rsp) {
               emit_bytes(0x40 | ((reg & 7) << 3) | 0x04, 0x24, mem.offset);
            } else {
               emit_bytes(0x40 | ((reg & 7) << 3) | (mem.reg & 7), mem.offset);
            }
         } else {
            if(mem.reg == rsp) {
               emit_bytes(0x80 | ((reg & 7) << 3) | 0x04, 0x24);
            } else {
               emit_bytes(0x80 | ((reg & 7) << 3) | (mem.reg & 7));
            }
            emit_operand32(mem.offset);
         }
      }
      void emit_modrm_sib_disp(simple_memory_ref mem, int reg) {
         if(mem.reg == rsp) {
            emit_bytes(((reg & 7) << 3) | 0x04, 0x24);
         } else if(mem.reg == rbp) {
            emit_bytes(0x45 | ((reg & 7) << 3), 0x00);
         } else {
            emit_bytes(((reg & 7) << 3) | (mem.reg & 7));
         }
      }
      void emit_modrm_sib_disp(any_register r_m, int reg) {
         emit_bytes(0xc0 | ((reg & 7) << 3) | (r_m.reg & 7));
      }
      template<typename T, int Sz>
      void emit_modrm_sib_disp(sized_memory_ref<T, Sz> mem, int reg) {
         emit_modrm_sib_disp(mem.value, reg);
      }

      // ──────────────────── Operand size helpers ────────────────────
      template<int W>
      static constexpr bool get_operand_size(general_register32) { return W == 1; }
      template<int W>
      static constexpr bool get_operand_size(general_register64) { return W != 0; }
      template<int W, typename T>
      static constexpr bool get_operand_size(sized_memory_ref<T, 4>) { return W == 1; }
      template<int W, typename T>
      static constexpr bool get_operand_size(sized_memory_ref<T, 8>) { return W != 0; }
      template<int W, typename RM>
      static constexpr bool get_operand_size(RM) {
         static_assert(W != -1);
         return W != 0;
      }
      template<int W, typename RM>
      static constexpr bool get_operand_size(RM, general_register32) { return W == 1; }
      template<int W, typename RM>
      static constexpr bool get_operand_size(RM, general_register64) { return W != 0; }
      template<int W, typename RM>
      static constexpr bool get_operand_size(RM r_m, int) {
         return get_operand_size<W>(r_m);
      }

      // ──────────────────── IA32 emit templates ────────────────────
      template<int W, int N>
      void emit(IA32_t<W, N> opcode) {
         static_assert(W == 0 || W == 1);
         emit_REX_prefix(W, false, false, false);
         for(int i = 0; i < N; ++i) emit_bytes(opcode.opcode[i]);
      }
      template<int W, int N, typename RM>
      void emit(IA32_t<W, N> opcode, RM r_m) {
         emit(opcode, r_m, 0);
      }
      template<int W, int N, typename RM, typename Reg>
      void emit(IA32_t<W, N> opcode, RM r_m, Reg reg) {
         // Legacy prefixes (66h, 67h, F2h, F3h) must be emitted BEFORE REX.
         // The x86-64 rule: if REX is not immediately before the opcode, it is ignored.
         int prefix_count = 0;
         if constexpr (N >= 2) {
            while (prefix_count < N - 1 &&
                   (opcode.opcode[prefix_count] == 0x66 || opcode.opcode[prefix_count] == 0x67 ||
                    opcode.opcode[prefix_count] == 0xF2 || opcode.opcode[prefix_count] == 0xF3)) {
               emit_bytes(opcode.opcode[prefix_count]);
               ++prefix_count;
            }
         }
         emit_REX_prefix(get_operand_size<W>(r_m, reg), r_m, reg);
         for(int i = prefix_count; i < N; ++i) emit_bytes(opcode.opcode[i]);
         emit_modrm_sib_disp(r_m, reg);
      }
      template<int W, int N, typename RM, typename Reg>
      void emit(IA32_t<W, N> opcode, imm8 imm, RM r_m, Reg reg) {
         emit(opcode, r_m, reg);
         emit_bytes(static_cast<uint8_t>(imm));
      }
      template<typename T, typename RM>
      void emit(IA32_opext<T> opcode, RM r_m) {
         emit(opcode.base, r_m, opcode.opext);
      }
      template<typename T, typename RM>
      void emit(IA32_opext<T> opcode, imm8 imm, RM r_m) {
         emit(opcode.base, imm, r_m, opcode.opext);
      }
      template<typename T, typename I, typename... U>
      void emit(IA32_imm<T, I> opcode, I imm, U... u) {
         emit(opcode.base, u...);
         emit_operand(imm);
      }

      // ──────────────────── VEX emit templates ────────────────────
      template<int Sz, VEX_pp pp, VEX_mmmm mmmm, int W>
      void emit(VEX<Sz, pp, mmmm, W> opcode, sib_memory_ref src1, xmm_register src2) {
         emit_VEX_prefix(src2 & 8, src1.reg1 & 8, src1.reg2 & 8, mmmm, W, 0, Sz == 256, pp);
         emit_bytes(opcode.opcode);
         emit_modrm_sib_disp(src1, src2);
      }
      template<int Sz, VEX_pp pp, VEX_mmmm mmmm, int W>
      void emit(VEX<Sz, pp, mmmm, W> opcode, disp_memory_ref src1, xmm_register src2) {
         emit_VEX_prefix(src2 & 8, false, src1.reg & 8, mmmm, W, 0, Sz == 256, pp);
         emit_bytes(opcode.opcode);
         emit_modrm_sib_disp(src1, src2);
      }
      template<int Sz, VEX_pp pp, VEX_mmmm mmmm, int W>
      void emit(VEX<Sz, pp, mmmm, W> opcode, simple_memory_ref src1, xmm_register src2) {
         emit_VEX_prefix(src2 & 8, false, src1.reg & 8, mmmm, W, 0, Sz == 256, pp);
         emit_bytes(opcode.opcode);
         emit_modrm_sib_disp(src1, src2);
      }
      template<int Sz, VEX_pp pp, VEX_mmmm mmmm, int W>
      void emit(VEX<Sz, pp, mmmm, W> opcode, general_register src1, xmm_register src2) {
         emit_VEX_prefix(src2 & 8, false, src1.reg & 8, mmmm, W, 0, Sz == 256, pp);
         emit_bytes(opcode.opcode);
         emit_modrm_sib_disp(src1, src2);
      }
      template<int Sz, VEX_pp pp, VEX_mmmm mmmm, int W>
      void emit(VEX<Sz, pp, mmmm, W> opcode, xmm_register src1, xmm_register src2) {
         emit_VEX_prefix(src2 & 8, false, src1 & 8, mmmm, W, 0, Sz == 256, pp);
         emit_bytes(opcode.opcode);
         emit_modrm_sib_disp(src1, src2);
      }
      template<int Sz, VEX_pp pp, VEX_mmmm mmmm, int W>
      void emit(VEX<Sz, pp, mmmm, W> opcode, xmm_register src1, general_register src2) {
         emit_VEX_prefix(src2.reg & 8, false, src1 & 8, mmmm, W, 0, Sz == 256, pp);
         emit_bytes(opcode.opcode);
         emit_modrm_sib_disp(src1, src2.reg);
      }
      template<int Sz, VEX_pp pp, VEX_mmmm mmmm, int W>
      void emit(VEX<Sz, pp, mmmm, W> opcode, imm8 src1, simple_memory_ref src2, xmm_register dest) {
         emit_VEX_prefix(dest & 8, false, src2.reg & 8, mmmm, W, 0, Sz == 256, pp);
         emit_bytes(opcode.opcode);
         emit_modrm_sib_disp(src2, dest);
         emit_byte(static_cast<uint8_t>(src1));
      }
      template<int Sz, VEX_pp pp, VEX_mmmm mmmm, int W>
      void emit(VEX<Sz, pp, mmmm, W> opcode, imm8 src1, xmm_register src2, xmm_register dest) {
         emit_VEX_prefix(dest & 8, false, src2 & 8, mmmm, W, 0, Sz == 256, pp);
         emit_bytes(opcode.opcode);
         emit_modrm_sib_disp(src2, dest);
         emit_byte(static_cast<uint8_t>(src1));
      }
      template<int Sz, VEX_pp pp, VEX_mmmm mmmm, int W>
      void emit(VEX<Sz, pp, mmmm, W> opcode, imm8 src1, general_register src2, xmm_register dest) {
         emit_VEX_prefix(dest & 8, false, src2.reg & 8, mmmm, W, 0, Sz == 256, pp);
         emit_bytes(opcode.opcode);
         emit_modrm_sib_disp(src2, dest);
         emit_byte(static_cast<uint8_t>(src1));
      }
      template<int Sz, VEX_pp pp, VEX_mmmm mmmm, int W>
      void emit(VEX<Sz, pp, mmmm, W> opcode, imm8 src1, xmm_register src2, general_register dest) {
         emit_VEX_prefix(dest.reg & 8, false, src2 & 8, mmmm, W, 0, Sz == 256, pp);
         emit_bytes(opcode.opcode);
         emit_modrm_sib_disp(src2, dest.reg);
         emit_byte(static_cast<uint8_t>(src1));
      }
      template<int Sz, VEX_pp pp, VEX_mmmm mmmm, int W, int OpExt>
      void emit(VEX<Sz, pp, mmmm, W, OpExt> opcode, imm8 src1, xmm_register src2, xmm_register dest) {
         emit_VEX_prefix(false, false, src2 & 8, mmmm, W, dest, Sz == 256, pp);
         emit_bytes(opcode.opcode);
         emit_modrm_sib_disp(src2, OpExt);
         emit_byte(static_cast<uint8_t>(src1));
      }
      // 3-operand VEX (dest = op(src1_xmm, src2))
      template<int Sz, VEX_pp pp, VEX_mmmm mmmm, int W>
      void emit(VEX<Sz, pp, mmmm, W> opcode, simple_memory_ref src1, xmm_register src2, xmm_register dest) {
         emit_VEX_prefix(dest & 8, false, src1.reg & 8, mmmm, W, src2, Sz == 256, pp);
         emit_bytes(opcode.opcode);
         emit_modrm_sib_disp(src1, dest);
      }
      template<int Sz, VEX_pp pp, VEX_mmmm mmmm, int W>
      void emit(VEX<Sz, pp, mmmm, W> opcode, any_register src1, xmm_register src2, xmm_register dest) {
         emit_VEX_prefix(dest & 8, false, src1.reg & 8, mmmm, W, src2, Sz == 256, pp);
         emit_bytes(opcode.opcode);
         emit_modrm_sib_disp(src1, dest);
      }
      template<int Sz, VEX_pp pp, VEX_mmmm mmmm, int W, typename R_M>
      void emit(VEX<Sz, pp, mmmm, W> opcode, imm8 imm, R_M src1, xmm_register src2, xmm_register dest) {
         emit(opcode, src1, src2, dest);
         emit_byte(static_cast<uint8_t>(imm));
      }

      // ──────────────────── Condition code / branch emission ────────────────────
      void emit_setcc(Jcc opcode, general_register8 reg) {
         emit(IA32(0x0f, 0x20 + opcode.opcode), reg);
      }

      template<int W, int N>
      void* emit_branch8(IA32_t<W, N> opcode) {
         emit(opcode);
         return emit_branch_target8();
      }
      void* emit_branch8(Jcc opcode) {
         emit_bytes(opcode.opcode);
         return emit_branch_target8();
      }
      void* emit_branchcc32(Jcc opcode) {
         emit_bytes(0x0f, 0x10 + opcode.opcode);
         return emit_branch_target32();
      }

      void* emit_branch_target32() {
         void * result = code;
         emit_operand32(3735928555u - static_cast<uint32_t>(reinterpret_cast<uintptr_t>(code)));
         return result;
      }
      void* emit_branch_target8() {
         void* result = code;
         emit_bytes(0xcc);
         return result;
      }

      static void fix_branch(void* branch, void* target) {
         auto branch_ = static_cast<unsigned char*>(branch);
         auto target_ = static_cast<unsigned char*>(target);
         auto relative = static_cast<uint32_t>(target_ - branch_ - 4);
         memcpy(branch_, &relative, 4);
      }
      static void fix_branch8(void* branch, void* target) {
         auto branch_ = static_cast<unsigned char*>(branch);
         auto target_ = static_cast<unsigned char*>(target);
         auto relative = static_cast<uint8_t>(target_ - branch_ - 1);
         *branch_ = relative;
      }

      // ──────────────────── High-level emit helpers ────────────────────
      void emit_add(int32_t immediate, general_register64 dest) {
         if(immediate <= 0x7F && immediate >= -0x80) {
            emit(ADD_imm8, static_cast<imm8>(immediate), dest);
         } else {
            emit(ADD_imm32, static_cast<imm32>(immediate), dest);
         }
      }
      void emit_add(int32_t immediate, general_register32 dest) {
         if(immediate <= 0x7F && immediate >= -0x80) {
            emit(ADD_imm8, static_cast<imm8>(immediate), dest);
         } else {
            emit(ADD_imm32, static_cast<imm32>(immediate), dest);
         }
      }
      void emit_add(general_register32 src, general_register32 dest) { emit(ADD_A, src, dest); }
      void emit_add(general_register64 src, general_register64 dest) { emit(ADD_A, src, dest); }

      template<typename RM>
      void emit_sub(int32_t immediate, RM dest) {
         if(immediate <= 0x7F && immediate >= -0x80) {
            emit(SUB_imm8, static_cast<imm8>(immediate), dest);
         } else {
            emit(SUB_imm32, static_cast<imm32>(immediate), dest);
         }
      }
      void emit_sub(general_register32 src, general_register32 dest) { emit(SUB_A, src, dest); }
      void emit_sub(general_register64 src, general_register64 dest) { emit(SUB_A, src, dest); }

      void emit_cmp(general_register32 src, general_register32 dest) { emit(CMP, src, dest); }
      void emit_cmp(general_register64 src, general_register64 dest) { emit(CMP, src, dest); }
      template<typename RM>
      void emit_cmp(int32_t immediate, RM dest) {
         if(immediate <= 0x7F && immediate >= -0x80) {
            emit(CMP_imm8, static_cast<imm8>(immediate), dest);
         } else {
            emit(CMP_imm32, static_cast<imm32>(immediate), dest);
         }
      }

      template<typename RM>
      void emit_and(int32_t immediate, RM dest) {
         if(immediate <= 0x7F && immediate >= -0x80) {
            emit(IA32_WX(0x83)/4, static_cast<imm8>(immediate), dest);
         } else {
            emit(AND_imm32, static_cast<imm32>(immediate), dest);
         }
      }

      template<typename RM>
      void emit_or(int32_t immediate, RM dest) {
         if(immediate <= 0x7F && immediate >= -0x80) {
            emit(OR_imm8, static_cast<imm8>(immediate), dest);
         } else {
            emit(OR_imm32, static_cast<imm32>(immediate), dest);
         }
      }

      void emit_xor(general_register32 src, general_register32 dest) { emit(XOR_A, src, dest); }
      template<typename RM>
      void emit_xor(int32_t immediate, RM dest) {
         if(immediate <= 0x7F && immediate >= -0x80) {
            emit(XOR_imm8, static_cast<imm8>(immediate), dest);
         } else {
            emit(XOR_imm32, static_cast<imm32>(immediate), dest);
         }
      }

      void emit_call(general_register64 reg) { emit(IA32(0xff)/2, reg); }

      void emit_mov(uint8_t src, general_register8 dest) {
         emit_REX_prefix(false, false, false, dest & 7);
         emit_bytes(0xb0 | (dest & 7), src);
      }
      void emit_movd(uint32_t src, disp_memory_ref dest) {
         emit(IA32(0xc7)/0, dest);
         emit_operand32(src);
      }
      void emit_mov(uint32_t src, general_register32 dest) {
         emit_REX_prefix(false, false, false, dest & 8);
         emit_bytes(0xb8 | (dest & 7));
         emit_operand32(src);
      }
      void emit_mov(uint64_t src, general_register64 dest) {
         emit_REX_prefix(true, false, false, dest & 8);
         emit_bytes(0xb8 | (dest & 7));
         emit_operand64(src);
      }
      template<typename T>
      void emit_mov(T* src, general_register64 dest) {
         emit_REX_prefix(true, false, false, dest & 8);
         emit_bytes(0xb8 | (dest & 7));
         emit_operand_ptr(src);
      }
      void emit_mov(general_register32 src, disp_memory_ref dest) { emit(IA32(0x89), dest, src); }
      void emit_mov(general_register32 src, general_register32 dest) { emit(MOV_A, src, dest); }
      void emit_mov(disp_memory_ref mem, general_register32 reg) { emit(MOV_A, mem, reg); }
      void emit_mov(general_register64 src, general_register64 dest) { emit(MOV_A, src, dest); }
      void emit_mov(disp_memory_ref mem, general_register64 reg) { emit(MOV_A, mem, reg); }
      void emit_mov(general_register64 reg, disp_memory_ref mem) { emit(MOV_B, mem, reg); }
      void emit_mov(sib_memory_ref mem, general_register8 reg) { emit(MOVB_A, mem, reg); }

      // Push/pop without peephole (raw encoding)
      void emit_push_raw(general_register64 reg) {
         emit_REX_prefix(false, false, false, reg & 8);
         emit_bytes(0x50 | (reg & 7));
      }
      void emit_pop_raw(general_register64 reg) {
         emit_REX_prefix(false, false, false, reg & 8);
         emit_bytes(0x58 | (reg & 7));
      }

      // VEX mov helpers
      void emit_vmovdqu(disp_memory_ref mem, xmm_register reg) { emit(VMOVDQU_A, mem, reg); }
      void emit_vmovdqu(xmm_register reg, disp_memory_ref mem) { emit(VMOVDQU_B, mem, reg); }
      void emit_vmovups(disp_memory_ref mem, xmm_register reg) { emit(VMOVUPS_A, mem, reg); }
      void emit_vmovups(simple_memory_ref mem, xmm_register reg) { emit(VMOVUPS_A, mem, reg); }
      void emit_vmovups(xmm_register reg, disp_memory_ref mem) { emit(VMOVUPS_B, mem, reg); }
      void emit_vmovups(xmm_register reg, simple_memory_ref mem) { emit(VMOVUPS_B, mem, reg); }
      void emit_vmovd(general_register32 src, xmm_register dest) { emit(VMOVD_A, src, dest); }
      void emit_vmovq(general_register64 src, xmm_register dest) { emit(VMOVQ_A, src, dest); }

      template<typename T>
      void emit_vpextrb(uint8_t offset, xmm_register src, T dest) { emit(VPEXTRB, imm8{offset}, dest, src); }
      template<typename T>
      void emit_vpextrw(uint8_t offset, xmm_register src, T dest) { emit(VPEXTRW, imm8{offset}, dest, src); }
      template<typename T>
      void emit_vpextrd(uint8_t offset, xmm_register src, T dest) { emit(VPEXTRD, imm8{offset}, dest, src); }
      template<typename T>
      void emit_vpextrq(uint8_t offset, xmm_register src, T dest) { emit(VPEXTRQ, imm8{offset}, dest, src); }

      void emit_const_zero(xmm_register reg) { emit(VPXOR, reg, reg, reg); }
      void emit_const_ones(xmm_register reg) { emit(VPCMPEQB, reg, reg, reg); }
   };

}} // namespace eosio::vm
