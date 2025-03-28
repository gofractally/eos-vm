#pragma once

#include <eosio/vm/allocator.hpp>
#include <eosio/vm/exceptions.hpp>
#include <eosio/vm/signals.hpp>
#include <eosio/vm/softfloat.hpp>
#include <eosio/vm/types.hpp>
#include <eosio/vm/utils.hpp>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <variant>
#include <vector>
#include <cpuid.h>


namespace eosio { namespace vm {

   struct instruction_counter {
      template<int line>
      static std::uint64_t* ptr() {
         static std::uint64_t value = 0;
         static auto register_ = data().insert({line, &value});
         return &value;
      }
      static std::map<int, std::uint64_t*>& data() {
         static std::map<int, std::uint64_t*> result;
         return result;
      }
      static void print() {
         std::uint64_t total = 0;
         for(auto [line, count] : data()) {
            if(*count != 0) {
               printf("%d %lu\n", line, *count);
            }
            total += *count;
         }
         printf("total: %lu\n", total);
      }
   };

#if 0
#define COUNT_INSTR() do { emit_mov(instruction_counter::ptr<__LINE__>(), rax); emit(INC, *qword_ptr(rax)); } while(0)
#define COUNT_INSTR_NO_FLAGS() do { emit_mov(instruction_counter::ptr<__LINE__>(), rax); emit_mov(*rax, rcx); emit(LEA, *(rcx + 1), rcx); emit_mov(rcx, *rax); } while(0)
#else
#define COUNT_INSTR() ((void)0)
#define COUNT_INSTR_NO_FLAGS() ((void)0)
#endif

   // Random notes:
   // - branch instructions return the address that will need to be updated
   // - label instructions return the address of the target
   // - fix_branch will be called when the branch target is resolved
   // - parameters and locals are accessed relative to rbp. stack items
   //   are pushed and popped.
   // - Some sequences of instructions can be folded. Up to two previous
   //   instructions are tracked.
   //
   // - The base of memory is stored in rsi
   // - rdi hold the execution context
   // - rbx holds the remaining stack frames
   //
   template<typename Context, bool StackLimitIsBytes>
   class machine_code_writer {
    public:
      machine_code_writer(growable_allocator& alloc, std::size_t source_bytes, module& mod) :
         _mod(mod), _allocator(alloc), _code_segment_base(_allocator.start_code()) {
         _code_start = _allocator.alloc<unsigned char>(128);
         _code_end = _code_start + 128;
         code = _code_start;
         emit_sysv_abi_interface();
         assert(code <= _code_end);
         _allocator.reclaim(code, _code_end - code);

         const std::size_t code_size = 5 * 16; // 5 error handlers, each is 16 bytes.
         _code_start = _allocator.alloc<unsigned char>(code_size);
         _code_end = _code_start + code_size;
         code = _code_start;

         // always emit these functions
         fpe_handler = emit_error_handler(&on_fp_error);
         call_indirect_handler = emit_error_handler(&on_call_indirect_error);
         type_error_handler = emit_error_handler(&on_type_error);
         stack_overflow_handler = emit_error_handler(&on_stack_overflow);
         memory_handler = emit_error_handler(&on_memory_error);

         assert(code == _code_end); // verify that the manual instruction count is correct

         // emit host functions
         const uint32_t num_imported = mod.get_imported_functions_size();
         const std::size_t host_functions_size = (42 + 10 * Context::async_backtrace()) * num_imported;
         _code_start = _allocator.alloc<unsigned char>(host_functions_size);
         _code_end = _code_start + host_functions_size;
         // code already set
         for(uint32_t i = 0; i < num_imported; ++i) {
            start_function(code, i);
            emit_host_call(i);
         }
         assert(code == _code_end);
      }
      ~machine_code_writer() {
         _allocator.end_code<true>(_code_segment_base);
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

      void emit_sysv_abi_interface() {
         // jit_execution_context* context -> RDI
         // void* linear_memory -> RSI
         // native_value* data, -> RDX
         // native_value (*fun)(void*, void*) -> RCX
         // void* stack -> R8
         // uint64_t count -> R9
         // uint32_t vector_result -> (RBP + 16)
         emit_push(rbp);
         emit_mov(rsp, rbp);
         emit_sub(16, rsp);

         // switch stack
         emit(TEST, r8, r8);
         // cmovnz
         emit(IA32_REX_W(0x0f, 0x45), r8, rsp);

         // save and set mxcsr
         emit(STMXCSR, *(rbp - 4));
         emit_movd(0x1f80, *(rbp - 8));
         emit(LDMXCSR, *(rbp - 8));

         // copy args
         emit(TEST, r9, r9);
         void* loop_end = emit_branch8(JZ);
         void* loop = code;
         emit_mov(*rdx, rax);
         emit_add(8, rdx);
         emit_push(rax);
         emit(DEC, r9);
         fix_branch8(emit_branch8(JNZ), loop);
         fix_branch8(loop_end, code);

         // load call depth counter
         emit_mov(rbx, *(rbp - 16));
         if constexpr (Context::async_backtrace())
         {
            emit_mov(*(rdi + 16), ebx);
         }
         else
         {
            emit_mov(*rdi, ebx);
         }

         if constexpr (Context::async_backtrace()) {
            emit_mov(rbp, *(rdi + 8));
         }
         emit_call(rcx);
         if constexpr (Context::async_backtrace()) {
            emit_xor(edx, edx);
            emit_mov(rdx, *(rdi + 8));
         }

         emit_mov(*(rbp - 16), rbx);

         emit(LDMXCSR, *(rbp - 4));

         emit_mov(*(rbp + 16), edx);
         emit(TEST, edx, edx);
         void* is_vector = emit_branch8(JZ);
         emit_vpextrq(0, xmm0, rax);
         emit_vpextrq(1, xmm0, rdx);
         fix_branch8(is_vector, code);

         emit_mov(rbp, rsp);
         emit_pop(rbp);
         emit(RET);
      }

      static constexpr std::size_t max_prologue_size = 33;
      static constexpr std::size_t max_epilogue_size = 16;
      void emit_prologue(const func_type& /*ft*/, const std::vector<local_entry>& locals, uint32_t funcnum) {
         _ft = &_mod.types[_mod.functions[funcnum]];
         _params = function_parameters{_ft};
         _locals = function_locals{locals};
         // FIXME: This is not a tight upper bound
         const std::size_t instruction_size_ratio_upper_bound = use_softfloat?(Context::async_backtrace()?63:49):79;
         std::size_t code_size = max_prologue_size + _mod.code[funcnum].size * instruction_size_ratio_upper_bound + max_epilogue_size;
         _code_start = _allocator.alloc<unsigned char>(code_size);
         _code_end = _code_start + code_size;
         code = _code_start;
         start_function(code, funcnum + _mod.get_imported_functions_size());
         // pushq RBP
         emit_bytes(0x55);
         // movq RSP, RBP
         emit_bytes(0x48, 0x89, 0xe5);
         emit_check_stack_limit();
         uint64_t count = 0;
         for(uint32_t i = 0; i < locals.size(); ++i) {
            count += locals[i].count;
            if(locals[i].type == types::v128) {
               count += locals[i].count;
            }
         }
         _local_count = count;
         if (_local_count > 0) {
            // xor %rax, %rax
            emit_bytes(0x48, 0x31, 0xc0);
            if (_local_count > 14) { // only use a loop if it would save space
               if(count > 0xFFFFFFFFu) {
                  // movabsq $count, %rcx
                  emit_bytes(0x48, 0xb9);
                  emit_operand64(_local_count);
               } else {
                  // mov $count, %ecx
                  emit_bytes(0xb9);
                  emit_operand32(static_cast<uint32_t>(_local_count));
               }
               // loop:
               void* loop = code;
               // pushq %rax
               emit_bytes(0x50);
               // dec %ecx
               if(_local_count > 0xFFFFFFFFu) {
                  emit_bytes(0x48);
               }
               emit_bytes(0xff, 0xc9);
               // jnz loop
               emit_bytes(0x0f, 0x85);
               fix_branch(emit_branch_target32(), loop);
            } else {
               for (uint32_t i = 0; i < _local_count; ++i) {
                  // pushq %rax
                  emit_bytes(0x50);
               }
            }
         }
         assert((char*)code <= (char*)_code_start + max_prologue_size);
      }
      void emit_epilogue(const func_type& ft, const std::vector<local_entry>& locals, uint32_t /*funcnum*/) {
#ifndef NDEBUG
         void * epilogue_start = code;
#endif
         emit_check_stack_limit_end();
         if(ft.return_count != 0) {
            if(ft.return_type == types::v128) {
               emit_vmovups(*rsp, xmm0);
            } else {
               // pop RAX
               emit_pop(rax);
            }
         }
         if (_local_count > 0 || (ft.return_count != 0 && ft.return_type == types::v128)) {
            emit_mov(rbp, rsp);
         }
         emit_pop(rbp);
         emit(RET);
         assert((char*)code <= (char*)epilogue_start + max_epilogue_size);
      }
      static constexpr uint32_t get_depth_for_type(uint8_t type) {
         if(type == types::v128) {
            return 2;
         } else {
            return 1;
         }
      }

      void emit_unreachable() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(16);
         emit_error_handler(&on_unreachable);
      }
      void emit_nop() {}
      void* emit_end() { set_branch_target(); return code; }
      void* emit_return(uint32_t depth_change, uint8_t rt) {
         // Return is defined as equivalent to branching to the outermost label
         return emit_br(depth_change, rt);
      }
      void emit_block() {}
      void* emit_loop() { set_branch_target(); return code; }
      void* emit_if() {
         if (auto cond = try_pop_recent_op<condition_op>()) {
            COUNT_INSTR_NO_FLAGS();
            return emit_branchcc32(reverse_condition(cond->branchop));
         }
         COUNT_INSTR();
         auto icount = fixed_size_instr(9);
         emit_pop(rax);
         emit(TEST, eax, eax);
         return emit_branchcc32(JZ);
      }
      void* emit_else(void* if_loc) {
         COUNT_INSTR();
         auto icount = fixed_size_instr(5);
         // We never need to adjust the stack, even if there is a return type
         void* result = emit_br(0, types::pseudo);
         fix_branch(if_loc, code);
         set_branch_target();
         return result;
      }
      void* emit_br(uint32_t depth_change, uint8_t rt) {
         COUNT_INSTR();
         auto icount = variable_size_instr(5, 22);
         // add RSP, depth_change * 8
         emit_multipop(depth_change, rt);
         // jmp DEST
         emit_bytes(0xe9);
         return emit_branch_target32();
      }
      void* emit_br_if(uint32_t depth_change, uint8_t rt) {
         if (auto cond = try_pop_recent_op<condition_op>()) {
            COUNT_INSTR_NO_FLAGS(); // The previous flags are use be the conditional branch
            if (is_simple_multipop(depth_change, rt)) {
               return emit_branchcc32(cond->branchop);
            } else {
               void* skip = emit_branch8(reverse_condition(cond->branchop));
               emit_multipop(depth_change, rt);
               emit(JMP_32);
               void* result = emit_branch_target32();
               fix_branch8(skip, code);
               return result;
            }
         }
         COUNT_INSTR();
         auto icount = variable_size_instr(9, 27);
         emit_pop(rax);
         emit(TEST, eax, eax);

         if(is_simple_multipop(depth_change, rt)) {
            return emit_branchcc32(JNZ);
         } else {
            void* skip = emit_branch8(JZ);
            // add depth_change*8, %rsp
            emit_multipop(depth_change, rt);
            emit(JMP_32);
            void* result = emit_branch_target32();
            // SKIP:
            fix_branch8(skip, code);
            return result;
         }
      }

      // Generate a binary search.
      struct br_table_generator {
         void* emit_case(uint32_t depth_change, uint8_t rt) {
            while(true) {
               assert(!stack.empty() && "The parser is supposed to handle the number of elements in br_table.");
               auto [min, max, label] = stack.back();
               stack.pop_back();
               if (label) {
                  fix_branch(label, _this->code);
               }
               if (max - min > 1) {
                  // Note: the total number of times this block is executed
                  // is bounded by the number of cases.
                  auto icount = _this->fixed_size_instr(11);
                  // Emit a comparison to the midpoint of the current range
                  uint32_t mid = min + (max - min)/2;
                  // cmp i, %mid
                  _this->emit_bytes(0x3d);
                  _this->emit_operand32(mid);
                  // jae MID
                  _this->emit_bytes(0x0f, 0x83);
                  void* mid_label = _this->emit_branch_target32();
                  stack.push_back({mid,max,mid_label});
                  stack.push_back({min,mid,nullptr});
               } else {
                  auto icount = _this->variable_size_instr(0, 22);
                  assert(min == static_cast<uint32_t>(_i));
                  _i++;
                  if (is_simple_multipop(depth_change, rt)) {
                     if(label) {
                        return label;
                     } else {
                        // jmp TARGET
                        _this->emit_bytes(0xe9);
                        return _this->emit_branch_target32();
                     }
                  } else {
                     // jne NEXT
                    _this->emit_multipop(depth_change, rt);
                    // jmp TARGET
                    _this->emit_bytes(0xe9);
                    return _this->emit_branch_target32();
                  }
               }
            }

         }
         void* emit_default(uint32_t depth_change, uint8_t rt) {
            void* result = emit_case(depth_change, rt);
            assert(stack.empty() && "unexpected default.");
            return result;
         }
         machine_code_writer * _this;
         int _i = 0;
         struct stack_item {
            uint32_t min;
            uint32_t max;
            void* branch_target = nullptr;
         };
         // stores a stack of ranges to be handled.
         // the ranges are strictly contiguous and non-ovelapping, with
         // the lower values at the back.
         std::vector<stack_item> stack;
      };
      br_table_generator emit_br_table(uint32_t table_size) {
         COUNT_INSTR();
         emit_pop(rax);
         // Increase the size by one to account for the default.
         // The current algorithm handles this correctly, without
         // any special cases.
         return { this, 0, { {0, table_size+1, nullptr} } };
      }

      void register_call(void* ptr, uint32_t funcnum) {
         auto& vec = _function_relocations;
         if(funcnum >= vec.size()) vec.resize(funcnum + 1);
         if(void** addr = std::get_if<void*>(&vec[funcnum])) {
            fix_branch(ptr, *addr);
         } else {
            std::get<std::vector<void*>>(vec[funcnum]).push_back(ptr);
         }
      }
      void start_function(void* func_start, uint32_t funcnum) {
         auto& vec = _function_relocations;
         if(funcnum >= vec.size()) vec.resize(funcnum + 1);
         for(void* branch : std::get<std::vector<void*>>(vec[funcnum])) {
            fix_branch(branch, func_start);
         }
         vec[funcnum] = func_start;
      }

      void emit_call(const func_type& ft, uint32_t funcnum) {
         COUNT_INSTR();
         auto icount = variable_size_instr(15, 31);
         emit_check_call_depth();
         // callq TARGET
         emit_bytes(0xe8);
         void * branch = emit_branch_target32();
         emit_multipop(ft);
         register_call(branch, funcnum);
         emit_check_call_depth_end();
      }

      void emit_call_indirect(const func_type& ft, uint32_t functypeidx)
      {
         COUNT_INSTR();
         auto icount = variable_size_instr(37, 64);
         emit_check_call_depth();
         std::uint32_t table_size = _mod.tables[0].limits.initial;
         emit_pop(rax);
         emit_cmp(table_size, eax);
         fix_branch(emit_branchcc32(JAE), call_indirect_handler);
         emit(SHL_imm8, imm8{4}, rax);
         if (_mod.indirect_table(0))
         {
            emit_mov(*(rsi + wasm_allocator::table_offset()), rcx);
            emit_add(rcx, rax);
         }
         else
         {
            emit(LEA, *(rax + rsi + wasm_allocator::table_offset()), rax);
         }
         emit_cmp(functypeidx, *dword_ptr(rax));
         fix_branch(emit_branchcc32(JNE), type_error_handler);
         emit(CALL, *(rax + 8));
         emit_multipop(ft);
         emit_check_call_depth_end();
      }

      void emit_drop(uint8_t type) {
         COUNT_INSTR();
         auto icount = variable_size_instr(1, 4);
         if(type == types::v128) {
            emit_add(16, rsp);
         } else {
            emit_pop(rax);
         }
      }

      void emit_select(uint8_t type) {
         COUNT_INSTR();
         auto icount = variable_size_instr(9, 20);
         if(type == types::v128) {
            emit_pop(rax);
            emit(TEST, eax, eax);
            auto cond = emit_branch8(JNZ);
            emit_vmovups(*rsp, xmm0);
            emit_vmovups(xmm0, *(rsp + 16));
            fix_branch8(cond, code);
            emit_add(16, rsp);
         } else {
            emit_pop(rax);
            emit_pop(rcx);
            emit_pop(rdx);
            emit(TEST, eax, eax);
            if (type == types::i32 || type == types::f32) {
               emit(CMOVZ, ecx, edx);
            } else {
               emit(CMOVZ, rcx, rdx);
            }
            emit_push(rdx);
         }
      }

      void emit_get_local(uint32_t local_idx, uint8_t type) {
         auto start = code;
         COUNT_INSTR();
         auto icount = variable_size_instr(5, 17);
         // stack layout:
         //   param0    <----- %rbp + 8*(nparams + 1)
         //   param1
         //   param2
         //   ...
         //   paramN
         //   return address
         //   old %rbp    <------ %rbp
         //   local0    <------ %rbp - 8
         //   local1
         //   ...
         //   localN
         // v128 occupies two slots
         auto addr = *(rbp + get_frame_offset(local_idx));
         if (type != types::v128) {
            emit_mov(addr, rax);
            emit_push(rax);
            push_recent_op(start, get_local_op{addr});
         } else {
            emit_vmovups(addr, xmm0);
            emit_sub(16, rsp);
            emit_vmovups(xmm0, *rsp);
         }
      }

      void emit_set_local(uint32_t local_idx, uint8_t type) {
         COUNT_INSTR();
         auto icount = variable_size_instr(5, 17);
         auto addr = *(rbp + get_frame_offset(local_idx));
         if (type != types::v128) {
            emit_pop(rax);
            emit_mov(rax, addr);
         } else {
            emit_vmovups(*rsp, xmm0);
            emit_add(16, rsp);
            emit_vmovups(xmm0, addr);
         }
      }

      void emit_tee_local(uint32_t local_idx, uint8_t type) {
         COUNT_INSTR();
         auto icount = variable_size_instr(5, 13);
         auto addr = *(rbp + get_frame_offset(local_idx));
         if (type != types::v128) {
            emit_pop(rax);
            if (type == i32 || type == f32) {
               emit_mov(eax, addr);
            } else {
               emit_mov(rax, addr);
            }
            emit_push(rax);
         } else {
            emit_vmovups(*rsp, xmm0);
            emit_vmovups(xmm0, addr);
         }
      }

      auto emit_global_loc(uint32_t globalidx)
      {
         auto& gl = _mod.globals[globalidx];

         auto offset = _mod.get_global_offset(globalidx);
         emit_mov(*(rsi + (wasm_allocator::globals_end() - 8)), rcx);
         if (offset > 0x7fffffff) {
            // This isn't quite optimal, but realistically, no one should
            // ever have this many globals
            emit_mov(static_cast<std::uint64_t>(offset), rdx);
            emit_add(rdx, rcx);
            offset = 0;
         }
         return *(rcx + static_cast<std::int32_t>(offset));
      }

      void emit_get_global(uint32_t globalidx) {
         COUNT_INSTR();
         auto icount = variable_size_instr(11, 36);

         auto& gl = _mod.globals[globalidx];
         auto loc = emit_global_loc(globalidx);
         switch(gl.type.content_type) {
            case types::i32:
               emit_mov(loc, eax); // min = op + modr/m + disp8 = 3
               emit_push(rax);
               break;
            case types::i64:
               emit_mov(loc, rax);
               emit_push(rax);
               break;
            case types::f32:
               emit_mov(loc, eax);
               emit_push(rax);
               break;
            case types::f64:
               emit_mov(loc, rax);
               emit_push(rax);
               break;
            case types::v128:
               emit_vmovdqu(loc, xmm0); // 3 + disp32 = 7
               emit_push_v128(xmm0); // 4 + 5 = 9
               break;
            default: assert(!"Unknown global type");
         }
      }

      void emit_set_global(uint32_t globalidx) {
         COUNT_INSTR();
         auto icount = variable_size_instr(11, 36);

         auto& gl = _mod.globals[globalidx];
         if (gl.type.content_type == types::v128) {
            emit_pop_v128(xmm0);
         } else {
            emit_pop(rax);
         }
         auto loc = emit_global_loc(globalidx);
         switch(gl.type.content_type) {
            case types::i32: emit_mov(eax, loc); break;
            case types::i64: emit_mov(rax, loc); break;
            case types::f32: emit_mov(eax, loc); break;
            case types::f64: emit_mov(rax, loc); break;
            case types::v128: emit_vmovdqu(xmm0, loc); break;
            default: assert(!"Unknown global type");
         }
      }

      void emit_i32_load(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(5, 13);
         emit_load_impl2(offset, MOV_A, eax);
      }

      void emit_i64_load(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(6, 14);
         emit_load_impl2(offset, MOV_A, rax);
      }

      void emit_f32_load(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(5, 13);
         emit_load_impl2(offset, MOV_A, eax);
      }

      void emit_f64_load(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(6, 14);
         emit_load_impl2(offset, MOV_A, rax);
      }

      void emit_i32_load8_s(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(6, 14);
         emit_load_impl2(offset, MOVSXB, eax);
      }

      void emit_i32_load16_s(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(6, 14);
         emit_load_impl2(offset, MOVSXW, eax);
      }

      void emit_i32_load8_u(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(6, 14);
         emit_load_impl2(offset, MOVZXB, eax);
      }

      void emit_i32_load16_u(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(6, 14);
         emit_load_impl2(offset, MOVZXW, eax);
      }

      void emit_i64_load8_s(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(7, 15);
         emit_load_impl2(offset, MOVSXB, rax);
      }

      void emit_i64_load16_s(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(7, 15);
         emit_load_impl2(offset, MOVSXW, rax);
      }

      void emit_i64_load32_s(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(6, 14);
         emit_load_impl2(offset, MOVSXD, rax);
      }

      void emit_i64_load8_u(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(6, 14);
         emit_load_impl2(offset, MOVZXB, eax);
      }

      void emit_i64_load16_u(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(6, 14);
         emit_load_impl2(offset, MOVZXW, eax);
      }

     void emit_i64_load32_u(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(5, 13);
         emit_load_impl2(offset, MOV_A, eax);
      }

      void emit_i32_store(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(5, 13);
         emit_store_impl2(offset, MOV_B, eax);
      }

      void emit_i64_store(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(6, 14);
         emit_store_impl2(offset, MOV_B, rax);
      }

      void emit_f32_store(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(5, 13);
         emit_store_impl2(offset, MOV_B, eax);
      }

      void emit_f64_store(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(6, 14);
         emit_store_impl2(offset, MOV_B, rax);
      }

      void emit_i32_store8(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(5, 13);
         emit_store_impl2(offset, MOVB_B, al);
      }

      void emit_i32_store16(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(6, 14);
         emit_store_impl2(offset, MOVW_B, ax);
      }

      void emit_i64_store8(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(5, 13);
         emit_store_impl2(offset, MOVB_B, al);
      }

      void emit_i64_store16(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(6, 14);
         emit_store_impl2(offset, MOVW_B, ax);
      }

      void emit_i64_store32(uint32_t /*alignment*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = variable_size_instr(5, 13);
         emit_store_impl2(offset, MOV_B, eax);
      }

      void emit_current_memory() {
         COUNT_INSTR();
         auto icount = variable_size_instr(17, 35);
         emit_setup_backtrace();
         emit_push(rdi);
         emit_push(rsi);
         emit_mov(&current_memory, rax);
         emit(CALL, rax);
         emit_pop(rsi);
         emit_pop(rdi);
         emit_restore_backtrace();
         emit_push(rax);
      }
      void emit_grow_memory() {
         COUNT_INSTR();
         auto icount = variable_size_instr(20, 39);
         emit_pop(rax);
         emit_setup_backtrace();
         emit_push(rdi);
         emit_push(rsi);
         emit_mov(eax, esi);
         emit_mov(&grow_memory, rax);
         emit(CALL, rax);
         emit_pop(rsi);
         emit_pop(rdi);
         emit_restore_backtrace();
         emit_push(rax);
      }
      void emit_memory_init(std::uint32_t x) {
         COUNT_INSTR();
         auto icount = variable_size_instr(25, 43);
         emit_pop(r8);
         emit_pop(rcx);
         emit_pop(rdx);
         emit_setup_backtrace();
         emit_push(rdi);
         emit_push(rsi);
         emit_mov(x, esi);

         emit_mov(&init_memory, rax);
         emit(CALL, rax);

         emit_pop(rsi);
         emit_pop(rdi);
         emit_restore_backtrace();
      }
      void emit_data_drop(std::uint32_t x) {
         COUNT_INSTR();
         auto icount = variable_size_instr(21, 39);
         emit_setup_backtrace();
         emit_push(rdi);
         emit_push(rsi);
         emit_mov(x, esi);

         emit_mov(&drop_data, rax);
         emit(CALL, rax);

         emit_pop(rsi);
         emit_pop(rdi);
         emit_restore_backtrace();
      }
      void emit_table_init(std::uint32_t x) {
         COUNT_INSTR();
         auto icount = variable_size_instr(25, 43);
         emit_pop(r8);
         emit_pop(rcx);
         emit_pop(rdx);
         emit_setup_backtrace();
         emit_push(rdi);
         emit_push(rsi);
         emit_mov(x, esi);

         emit_mov(&init_table, rax);
         emit(CALL, rax);

         emit_pop(rsi);
         emit_pop(rdi);
         emit_restore_backtrace();
      }
      void emit_elem_drop(std::uint32_t x) {
         COUNT_INSTR();
         auto icount = variable_size_instr(21, 39);
         emit_setup_backtrace();
         emit_push(rdi);
         emit_push(rsi);
         emit_mov(x, esi);

         emit_mov(&drop_elem, rax);
         emit(CALL, rax);

         emit_pop(rsi);
         emit_pop(rdi);
         emit_restore_backtrace();
      }

      void emit_i32_const(uint32_t value) {
         auto start = code;
         if (value == 0) {
            COUNT_INSTR();
            auto icount = fixed_size_instr(3);
            emit_xor(eax, eax);
            emit_push(rax);
         } else {
            COUNT_INSTR();
            auto icount = fixed_size_instr(6);
            emit_mov(value, eax);
            emit_push(rax);
         }
         push_recent_op(start, i32_const_op{value});
      }

      void emit_i64_const(uint64_t value) {
         auto start = code;
         COUNT_INSTR();
         auto icount = fixed_size_instr(11);
         emit_mov(value, rax);
         emit_push(rax);
         push_recent_op(start, i64_const_op{value});
      }

      void emit_f32_const(float value) {
         COUNT_INSTR();
         auto icount = fixed_size_instr(6);
         emit_mov(std::bit_cast<std::uint32_t>(value), eax);
         emit_push(rax);
      }
      void emit_f64_const(double value) {
         COUNT_INSTR();
         auto icount = fixed_size_instr(11);
         emit_mov(std::bit_cast<std::uint64_t>(value), rax);
         emit_push(rax);
      }

      void emit_i32_eqz() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(9);
         emit_pop(rax);
         emit(XOR_A, ecx, ecx);
         emit(TEST, eax, eax);
         auto start = code;
         emit(SETZ, cl);
         emit_push(rcx);
         push_recent_op(start, condition_op{JZ});
      }

      // i32 relops
      void emit_i32_eq() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(10);
         emit_i32_relop(JE);
      }

      void emit_i32_ne() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(10);
         emit_i32_relop(JNE);
      }

      void emit_i32_lt_s() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(10);
         emit_i32_relop(JL);
      }

      void emit_i32_lt_u() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(10);
         emit_i32_relop(JB);
      }

      void emit_i32_gt_s() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(10);
         emit_i32_relop(JG);
      }

      void emit_i32_gt_u() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(10);
         emit_i32_relop(JA);
      }

      void emit_i32_le_s() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(10);
         emit_i32_relop(JLE);
      }

      void emit_i32_le_u() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(10);
         emit_i32_relop(JBE);
      }

      void emit_i32_ge_s() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(10);
         emit_i32_relop(JGE);
      }

      void emit_i32_ge_u() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(10);
         emit_i32_relop(JAE);
      }

      void emit_i64_eqz() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(11);
         // pop %rax
         emit_bytes(0x58);
         // xor %rcx, %rcx
         emit_bytes(0x48, 0x31, 0xc9);
         // test %rax, %rax
         emit_bytes(0x48, 0x85, 0xc0);
         // setz %cl
         emit_bytes(0x0f, 0x94, 0xc1);
         // push %rcx
         emit_bytes(0x51);
      }
      // i64 relops
      void emit_i64_eq() {
         auto icount = fixed_size_instr(12);
         // sete %dl
         emit_i64_relop(0x94);
      }

      void emit_i64_ne() {
         auto icount = fixed_size_instr(12);
         // sete %dl
         emit_i64_relop(0x95);
      }

      void emit_i64_lt_s() {
         auto icount = fixed_size_instr(12);
         // setl %dl
         emit_i64_relop(0x9c);
      }

      void emit_i64_lt_u() {
         auto icount = fixed_size_instr(12);
         // setl %dl
         emit_i64_relop(0x92);
      }

      void emit_i64_gt_s() {
         auto icount = fixed_size_instr(12);
         // setg %dl
         emit_i64_relop(0x9f);
      }

      void emit_i64_gt_u() {
         auto icount = fixed_size_instr(12);
         // seta %dl
         emit_i64_relop(0x97);
      }

      void emit_i64_le_s() {
         auto icount = fixed_size_instr(12);
         // setle %dl
         emit_i64_relop(0x9e);
      }

      void emit_i64_le_u() {
         auto icount = fixed_size_instr(12);
         // setbe %dl
         emit_i64_relop(0x96);
      }

      void emit_i64_ge_s() {
         auto icount = fixed_size_instr(12);
         // setge %dl
         emit_i64_relop(0x9d);
      }

      void emit_i64_ge_u() {
         auto icount = fixed_size_instr(12);
         // setae %dl
         emit_i64_relop(0x93);
      }

#ifdef EOS_VM_SOFTFLOAT
      // Make sure that the result doesn't contain any garbage bits in rax
      static uint64_t adapt_result(bool val) {
         return val?1:0;
      }
      static uint64_t adapt_result(float32_t val) {
         uint64_t result = 0;
         std::memcpy(&result, &val, sizeof(float32_t));
         return result;
      }
      static float64_t adapt_result(float64_t val) {
         return val;
      }

      template<auto F>
      static auto adapt_f32_unop(float32_t arg) {
        return ::to_softfloat32(static_cast<decltype(F)>(F)(::from_softfloat32(arg)));
      }
      template<auto F>
      static auto adapt_f32_binop(float32_t lhs, float32_t rhs) {
         return ::to_softfloat32(static_cast<decltype(F)>(F)(::from_softfloat32(lhs), ::from_softfloat32(rhs)));
      }
      template<auto F>
      static auto adapt_f32_cmp(float32_t lhs, float32_t rhs) {
         return adapt_result(static_cast<decltype(F)>(F)(::from_softfloat32(lhs), ::from_softfloat32(rhs)));
      }

      template<auto F>
      static auto adapt_f64_unop(float64_t arg) {
         return ::to_softfloat64(static_cast<decltype(F)>(F)(::from_softfloat64(arg)));
      }
      template<auto F>
      static auto adapt_f64_binop(float64_t lhs, float64_t rhs) {
         return ::to_softfloat64(static_cast<decltype(F)>(F)(::from_softfloat64(lhs), ::from_softfloat64(rhs)));
      }
      template<auto F>
      static auto adapt_f64_cmp(float64_t lhs, float64_t rhs) {
         return adapt_result(static_cast<decltype(F)>(F)(::from_softfloat64(lhs), ::from_softfloat64(rhs)));
      }

      static float32_t to_softfloat(float arg) { return ::to_softfloat32(arg); }
      static float64_t to_softfloat(double arg) { return ::to_softfloat64(arg); }
      template<typename T>
      static T to_softfloat(T arg) { return arg; }
      static float from_softfloat(float32_t arg) { return ::from_softfloat32(arg); }
      static double from_softfloat(float64_t arg) { return ::from_softfloat64(arg); }
      template<typename T>
      static T from_softfloat(T arg) { return arg; }

      template<typename T>
      using softfloat_arg_t = decltype(to_softfloat(T{}));

      template<auto F, typename T>
      static auto adapt_float_convert(softfloat_arg_t<T> arg) {
         auto result = to_softfloat(F(from_softfloat(arg)));
         if constexpr (sizeof(result) == 4 && sizeof(T) == 8) {
            uint64_t buffer = 0;
            std::memcpy(&buffer, &result, sizeof(result));
            return buffer;
         } else {
            return result;
         }
      }

      template<auto F, typename R, typename T>
      static constexpr auto choose_unop(R(*)(T)) {
         if constexpr(sizeof(R) == 4 && sizeof(T) == 8) {
            return static_cast<uint64_t(*)(softfloat_arg_t<T>)>(&adapt_float_convert<F, T>);
         } else {
            return static_cast<softfloat_arg_t<R>(*)(softfloat_arg_t<T>)>(&adapt_float_convert<F, T>);
         }
      }

      // HACK: avoid linking to softfloat if we aren't using it
      // and also avoid passing arguments in floating point registers,
      // since softfloat uses integer registers.
      template<auto F>
      constexpr auto choose_fn() {
         if constexpr (use_softfloat) {
            if constexpr (std::is_same_v<decltype(F), float(*)(float)>) {
               return &adapt_f32_unop<F>;
            } else if constexpr(std::is_same_v<decltype(F), float(*)(float,float)>) {
               return &adapt_f32_binop<F>;
            } else if constexpr(std::is_same_v<decltype(F), bool(*)(float,float)>) {
               return &adapt_f32_cmp<F>;
            } else if constexpr (std::is_same_v<decltype(F), double(*)(double)>) {
               return &adapt_f64_unop<F>;
            } else if constexpr(std::is_same_v<decltype(F), double(*)(double,double)>) {
               return &adapt_f64_binop<F>;
            } else if constexpr(std::is_same_v<decltype(F), bool(*)(double,double)>) {
               return &adapt_f64_cmp<F>;
            } else {
               return choose_unop<F>(F);
            }
         } else {
            return nullptr;
         }
      }

      template<auto F, typename R, typename... A>
      static R softfloat_trap_fn(A... a) {
         R result;
         longjmp_on_exception([&]() {
            result = F(a...);
         });
         return result;
      }

      template<auto F, typename R, typename... A>
      static constexpr auto make_softfloat_trap_fn(R(*)(A...)) -> R(*)(A...) {
         return softfloat_trap_fn<F, R, A...>;
      }

      template<auto F>
      static constexpr decltype(auto) softfloat_trap() {
         return *make_softfloat_trap_fn<F>(F);
      }

   #define CHOOSE_FN(name) choose_fn<&name>()
#else
      using float32_t = float;
      using float64_t = double;
   #define CHOOSE_FN(name) nullptr
#endif

      // --------------- f32 relops ----------------------
      void emit_f32_eq() {
         auto icount = softfloat_instr(25,45,59);
         emit_f32_relop(0x00, CHOOSE_FN(_eosio_f32_eq), false, false);
      }

      void emit_f32_ne() {
         auto icount = softfloat_instr(24,47,61);
         emit_f32_relop(0x00, CHOOSE_FN(_eosio_f32_eq), false, true);
      }

      void emit_f32_lt() {
         auto icount = softfloat_instr(25,45,59);
         emit_f32_relop(0x01, CHOOSE_FN(_eosio_f32_lt), false, false);
      }

      void emit_f32_gt() {
         auto icount = softfloat_instr(25,45,59);
         emit_f32_relop(0x01, CHOOSE_FN(_eosio_f32_lt), true, false);
      }

      void emit_f32_le() {
         auto icount = softfloat_instr(25,45,59);
         emit_f32_relop(0x02, CHOOSE_FN(_eosio_f32_le), false, false);
      }

      void emit_f32_ge() {
         auto icount = softfloat_instr(25,45,59);
         emit_f32_relop(0x02, CHOOSE_FN(_eosio_f32_le), true, false);
      }

      // --------------- f64 relops ----------------------
      void emit_f64_eq() {
         auto icount = softfloat_instr(25,47,61);
         emit_f64_relop(0x00, CHOOSE_FN(_eosio_f64_eq), false, false);
      }

      void emit_f64_ne() {
         auto icount = softfloat_instr(24,49,63);
         emit_f64_relop(0x00, CHOOSE_FN(_eosio_f64_eq), false, true);
      }

      void emit_f64_lt() {
         auto icount = softfloat_instr(25,47,61);
         emit_f64_relop(0x01, CHOOSE_FN(_eosio_f64_lt), false, false);
      }

      void emit_f64_gt() {
         auto icount = softfloat_instr(25,47,61);
         emit_f64_relop(0x01, CHOOSE_FN(_eosio_f64_lt), true, false);
      }

      void emit_f64_le() {
         auto icount = softfloat_instr(25,47,61);
         emit_f64_relop(0x02, CHOOSE_FN(_eosio_f64_le), false, false);
      }

      void emit_f64_ge() {
         auto icount = softfloat_instr(25,47,61);
         emit_f64_relop(0x02, CHOOSE_FN(_eosio_f64_le), true, false);
      }

      // --------------- i32 unops ----------------------

      bool has_tzcnt_impl() {
         unsigned a, b, c, d;
         return __get_cpuid_count(7, 0, &a, &b, &c, &d) && (b & bit_BMI) &&
                __get_cpuid(0x80000001, &a, &b, &c, &d) && (c & bit_LZCNT);
      }

      bool has_tzcnt() {
         static bool result = has_tzcnt_impl();
         return result;
      }

      void emit_i32_clz() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(has_tzcnt()?6:18);
         if(!has_tzcnt()) {
            emit_pop(rax);
            emit_mov(-1, ecx);
            emit(BSR, eax, eax);
            emit(CMOVZ, ecx, eax);
            emit_sub(31, eax);
            emit(NEG, eax);
            emit_push(rax);
         } else {
            emit_pop(rax);
            emit(LZCNT, eax, eax);
            emit_push(rax);
         }
      }

      void emit_i32_ctz() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(has_tzcnt()?6:13);
         if(!has_tzcnt()) {
            emit_pop(rax);
            emit_mov(32, ecx);
            emit(BSF, eax, eax);
            emit(CMOVZ, ecx, eax);
            emit_push(rax);
         } else {
            emit_pop(rax);
            emit(TZCNT, eax, eax);
            emit_push(rax);
         }
      }

      void emit_i32_popcnt() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(6);
         emit_pop(rax);
         emit(POPCNT, eax);
         emit_push(rax);
      }

      // --------------- i32 binops ----------------------

      void emit_i32_add() {
         if (auto local = try_pop_recent_op<get_local_op>()) {
            COUNT_INSTR();
            emit_pop(rax);
            emit(ADD_A, local->expr, eax);
            emit_push(rax);
            return;
         }
         if (auto constant = try_pop_recent_op<i32_const_op>()) {
            auto value = constant->value;
            COUNT_INSTR();
            emit_pop(rax);
            emit_add(value, eax);
            emit_push(rax);
            return;
         }
         COUNT_INSTR();
         auto icount = fixed_size_instr(5);
         emit_pop(rax);
         emit_pop(rcx);
         emit_add(ecx, eax);
         emit_push(rax);
      }
      void emit_i32_sub() {
         if (auto c = try_pop_recent_op<i32_const_op>()) {
            COUNT_INSTR();
            emit_pop(rax);
            emit_sub(c->value, eax);
            emit_push(rax);
            return;
         }
         COUNT_INSTR();
         auto icount = fixed_size_instr(5);
         emit_pop(rcx);
         emit_pop(rax);
         emit_sub(ecx, eax);
         emit_push(rax);
      }
      void emit_i32_mul() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(6);
         emit_pop(rax);
         emit_pop(rcx);
         emit(IMUL, ecx, eax);
         emit_push(rax);
      }
      void emit_i32_div_s() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(6);
         emit_pop(rcx);
         emit_pop(rax);
         emit(CDQ);
         emit(IDIV, ecx);
         emit_push(rax);
      }
      void emit_i32_div_u() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(7);
         emit_pop(rcx);
         emit_pop(rax);
         emit_xor(edx, edx);
         emit(DIV, ecx);
         emit_push(rax);
      }
      void emit_i32_rem_s() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(15);
         emit_pop(rcx);
         emit_pop(rax);
         emit_cmp(-1, ecx);
         void* minus1 = emit_branch8(JE);
         emit(CDQ);
         emit(IDIV, ecx);
         void* end = emit_branch8(JMP_8);
         fix_branch8(minus1, code);
         emit_xor(edx, edx);
         fix_branch8(end, code);
         emit_push(rdx);
      }
      void emit_i32_rem_u() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(7);
         emit_pop(rcx);
         emit_pop(rax);
         emit_xor(edx, edx);
         emit(DIV, ecx);
         emit_push(rdx);
      }
      void emit_i32_and() {
         if (auto c = try_pop_recent_op<i32_const_op>()) {
            COUNT_INSTR();
            emit_pop(rax);
            emit_and(c->value, eax);
            emit_push(rax);
            return;
         }
         COUNT_INSTR();
         auto icount = fixed_size_instr(5);
         emit_pop(rax);
         emit_pop(rcx);
         emit(AND_A, ecx, eax);
         emit_push(rax);
      }
      void emit_i32_or() {
         if (auto c = try_pop_recent_op<i32_const_op>()) {
            COUNT_INSTR();
            emit_pop(rax);
            emit_or(c->value, eax);
            emit_push(rax);
            return;
         }
         COUNT_INSTR();
         auto icount = fixed_size_instr(5);
         emit_pop(rax);
         emit_pop(rcx);
         emit(OR_A, ecx, eax);
         emit_push(rax);
      }
      void emit_i32_xor() {
         if (auto c = try_pop_recent_op<get_local_op>()) {
            COUNT_INSTR();
            emit_pop(rax);
            emit(XOR_A, c->expr, eax);
            emit_push(rax);
            return;
         }
         if (auto c = try_pop_recent_op<i32_const_op>()) {
            COUNT_INSTR();
            emit_pop(rax);
            if (c->value == 0xffffffffu) {
               emit(NOT, eax);
            } else {
               emit_xor(c->value, eax);
            }
            emit_push(rax);
            return;
         }
         COUNT_INSTR();
         auto icount = fixed_size_instr(5);
         emit_pop(rax);
         emit_pop(rcx);
         emit(XOR_A, ecx, eax);
         emit_push(rax);
      }
      void emit_i32_shl() {
         if (auto c = try_pop_recent_op<i32_const_op>()) {
            COUNT_INSTR();
            emit_pop(rax);
            emit(SHL_imm8, static_cast<imm8>(c->value & 0x1f), eax);
            emit_push(rax);
            return;
         }
         COUNT_INSTR();
         auto icount = fixed_size_instr(5);
         emit_pop(rcx);
         emit_pop(rax);
         emit(SHL_cl, eax);
         emit_push(rax);
      }
      void emit_i32_shr_s() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(5);
         emit_pop(rcx);
         emit_pop(rax);
         emit(SAR_cl, eax);
         emit_push(rax);
      }
      void emit_i32_shr_u() {
         if (auto c = try_pop_recent_op<i32_const_op>()) {
            COUNT_INSTR();
            emit_pop(rax);
            emit(SHR_imm8, static_cast<imm8>(c->value & 0x1f), eax);
            emit_push(rax);
            return;
         }
         COUNT_INSTR();
         auto icount = fixed_size_instr(5);
         emit_pop(rcx);
         emit_pop(rax);
         emit(SHR_cl, eax);
         emit_push(rax);
      }
      void emit_i32_rotl() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(5);
         emit_pop(rcx);
         emit_pop(rax);
         emit(ROL_cl, eax);
         emit_push(rax);
      }
      void emit_i32_rotr() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(5);
         emit_pop(rcx);
         emit_pop(rax);
         emit(ROR_cl, eax);
         emit_push(rax);
      }

      // --------------- i64 unops ----------------------

      void emit_i64_clz() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(has_tzcnt()?7:24);
         if(!has_tzcnt()) {
            // pop %rax
            emit_bytes(0x58);
            // mov $-1, %ecx
            emit_bytes(0x48, 0xc7, 0xc1, 0xff, 0xff, 0xff, 0xff);
            // bsr %eax, %eax
            emit_bytes(0x48, 0x0f, 0xbd, 0xc0);
            // cmovz %ecx, %eax
            emit_bytes(0x48, 0x0f, 0x44, 0xc1);
            // sub $63, %eax
            emit_bytes(0x48, 0x83, 0xe8, 0x3f);
            // neg %eax
            emit_bytes(0x48, 0xf7, 0xd8);
            // push %rax
            emit_bytes(0x50);
         } else {
            // popq %rax
            emit_bytes(0x58);
            // lzcntq %eax, %eax
            emit_bytes(0xf3, 0x48, 0x0f, 0xbd, 0xc0);
            // pushq %rax
            emit_bytes(0x50);
         }
      }

      void emit_i64_ctz() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(has_tzcnt()?7:17);
         if(!has_tzcnt()) {
            // pop %rax
            emit_bytes(0x58);
            // mov $64, %ecx
            emit_bytes(0x48, 0xc7, 0xc1, 0x40, 0x00, 0x00, 0x00);
            // bsf %eax, %eax
            emit_bytes(0x48, 0x0f, 0xbc, 0xc0);
            // cmovz %ecx, %eax
            emit_bytes(0x48, 0x0f, 0x44, 0xc1);
            // push %rax
            emit_bytes(0x50);
         } else {
            // popq %rax
            emit_bytes(0x58);
            // tzcntq %eax, %eax
            emit_bytes(0xf3, 0x48, 0x0f, 0xbc, 0xc0);
            // pushq %rax
            emit_bytes(0x50);
         }
      }

      void emit_i64_popcnt() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(7);
         // popq %rax
         emit_bytes(0x58);
         // popcntq %rax, %rax
         emit_bytes(0xf3, 0x48, 0x0f, 0xb8, 0xc0);
         // pushq %rax
         emit_bytes(0x50);
      }

      // --------------- i64 binops ----------------------

      void emit_i64_add() {
         if (auto c = try_pop_recent_op<i64_const_op>()) {
            COUNT_INSTR();
            emit_pop(rax);
            emit_mov(c->value, rcx);
            emit_add(rcx, rax);
            emit_push(rax);
            return;
         }
         COUNT_INSTR();
         auto icount = fixed_size_instr(6);
         emit_pop(rax);
         emit_pop(rcx);
         emit_add(rcx, rax);
         emit_push(rax);
      }
      void emit_i64_sub() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(6);
         emit_i64_binop(0x48, 0x29, 0xc8, 0x50);
      }
      void emit_i64_mul() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(7);
         emit_pop(rax);
         emit_pop(rcx);
         emit(IMUL, rcx, rax);
         emit_push(rax);
      }
      // cdq; idiv %rcx; pushq %rax
      void emit_i64_div_s() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(8);
         emit_i64_binop(0x48, 0x99, 0x48, 0xf7, 0xf9, 0x50);
      }
      void emit_i64_div_u() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(9);
         emit_i64_binop(0x48, 0x31, 0xd2, 0x48, 0xf7, 0xf1, 0x50);
      }
      void emit_i64_rem_s() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(25);
         // pop %rcx
         emit_bytes(0x59);
         // pop %rax
         emit_bytes(0x58);
         // cmp $-1, %rcx
         emit_bytes(0x48, 0x83, 0xf9, 0xff);
         // je MINUS1
         emit_bytes(0x0f, 0x84);
         void* minus1 = emit_branch_target32();
         // cqo
         emit_bytes(0x48, 0x99);
         // idiv %rcx
         emit_bytes(0x48, 0xf7, 0xf9);
         // jmp END
         emit_bytes(0xe9);
         void* end = emit_branch_target32();
         // MINUS1:
         fix_branch(minus1, code);
         // xor %edx, %edx
         emit_bytes(0x31, 0xd2);
         // END:
         fix_branch(end, code);
         // push %rdx
         emit_bytes(0x52);
      }
      void emit_i64_rem_u() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(9);
         emit_i64_binop(0x48, 0x31, 0xd2, 0x48, 0xf7, 0xf1, 0x52);
      }
      void emit_i64_and() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(6);
         emit_i64_binop(0x48, 0x21, 0xc8, 0x50);
      }
      void emit_i64_or() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(6);
         emit_i64_binop(0x48, 0x09, 0xc8, 0x50);
      }
      void emit_i64_xor() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(6);
         emit_i64_binop(0x48, 0x31, 0xc8, 0x50);
      }
      void emit_i64_shl() {
         if (auto c = try_pop_recent_op<i64_const_op>()) {
            COUNT_INSTR();
            emit_pop(rax);
            emit(SHL_imm8, static_cast<imm8>(c->value & 0x3f), rax);
            emit_push(rax);
            return;
         }
         COUNT_INSTR();
         auto icount = fixed_size_instr(6);
         emit_pop(rcx);
         emit_pop(rax);
         emit(SHL_cl, rax);
         emit_push(rax);
      }
      void emit_i64_shr_s() {
         if (auto c = try_pop_recent_op<i64_const_op>()) {
            COUNT_INSTR();
            emit_pop(rax);
            emit(SAR_imm8, static_cast<imm8>(c->value & 0x3f), rax);
            emit_push(rax);
            return;
         }
         COUNT_INSTR();
         auto icount = fixed_size_instr(6);
         emit_pop(rcx);
         emit_pop(rax);
         emit(SAR_cl, rax);
         emit_push(rax);
      }
      void emit_i64_shr_u() {
         if (auto c = try_pop_recent_op<i64_const_op>()) {
            COUNT_INSTR();
            emit_pop(rax);
            emit(SHR_imm8, static_cast<imm8>(c->value & 0x3f), rax);
            emit_push(rax);
            return;
         }
         COUNT_INSTR();
         auto icount = fixed_size_instr(6);
         emit_pop(rcx);
         emit_pop(rax);
         emit(SHR_cl, rax);
         emit_push(rax);
      }
      void emit_i64_rotl() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(6);
         emit_i64_binop(0x48, 0xd3, 0xc0, 0x50);
      }
      void emit_i64_rotr() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(6);
         emit_i64_binop(0x48, 0xd3, 0xc8, 0x50);
      }

      // --------------- f32 unops ----------------------

      void emit_f32_abs() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(7);
         // popq %rax; 
         emit_bytes(0x58);
         // andl 0x7fffffff, %eax
         emit_bytes(0x25);
         emit_operand32(0x7fffffff);
         // pushq %rax
         emit_bytes(0x50);
      }

      void emit_f32_neg() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(7);
         // popq %rax
         emit_bytes(0x58);
         // xorl 0x80000000, %eax
         emit_bytes(0x35);
         emit_operand32(0x80000000);
         // pushq %rax
         emit_bytes(0x50);
      }

      void emit_f32_ceil() {
         COUNT_INSTR();
         auto icount = softfloat_instr(12, 36, 54);
         if constexpr (use_softfloat) {
            if (_mod.eosio_fp) {
               return emit_softfloat_unop(CHOOSE_FN(_eosio_f32_ceil<false>));
            } else {
               return emit_softfloat_unop(CHOOSE_FN(_eosio_f32_ceil<true>));
            }
         }
         // roundss 0b1010, (%rsp), %xmm0
         emit_bytes(0x66, 0x0f, 0x3a, 0x0a, 0x04, 0x24, 0x0a);
         // movss %xmm0, (%rsp)
         emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
      }

      void emit_f32_floor() {
         COUNT_INSTR();
         auto icount = softfloat_instr(12, 36, 54);
         if constexpr (use_softfloat) {
            if (_mod.eosio_fp) {
               return emit_softfloat_unop(CHOOSE_FN(_eosio_f32_floor<false>));
            } else {
               return emit_softfloat_unop(CHOOSE_FN(_eosio_f32_floor<true>));
            }
         }
         // roundss 0b1001, (%rsp), %xmm0
         emit_bytes(0x66, 0x0f, 0x3a, 0x0a, 0x04, 0x24, 0x09);
         // movss %xmm0, (%rsp)
         emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
      }

      void emit_f32_trunc() {
         COUNT_INSTR();
         auto icount = softfloat_instr(12, 36, 54);
         if constexpr (use_softfloat) {
            if (_mod.eosio_fp) {
               return emit_softfloat_unop(CHOOSE_FN(_eosio_f32_trunc<false>));
            } else {
               return emit_softfloat_unop(CHOOSE_FN(_eosio_f32_trunc<true>));
            }
         }
         // roundss 0b1011, (%rsp), %xmm0
         emit_bytes(0x66, 0x0f, 0x3a, 0x0a, 0x04, 0x24, 0x0b);
         // movss %xmm0, (%rsp)
         emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
      }

      void emit_f32_nearest() {
         COUNT_INSTR();
         auto icount = softfloat_instr(12, 36, 54);
         if constexpr (use_softfloat) {
            if (_mod.eosio_fp) {
               return emit_softfloat_unop(CHOOSE_FN(_eosio_f32_nearest<false>));
            } else {
               return emit_softfloat_unop(CHOOSE_FN(_eosio_f32_nearest<true>));
            }
         }
         // roundss 0b1000, (%rsp), %xmm0
         emit_bytes(0x66, 0x0f, 0x3a, 0x0a, 0x04, 0x24, 0x08);
         // movss %xmm0, (%rsp)
         emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
      }

      void emit_f32_sqrt() {
         COUNT_INSTR();
         auto icount = softfloat_instr(10, 36, 54);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(_eosio_f32_sqrt));
         }
         // sqrtss (%rsp), %xmm0
         emit_bytes(0xf3, 0x0f, 0x51, 0x04, 0x24);
         // movss %xmm0, (%rsp)
         emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
      }

      // --------------- f32 binops ----------------------

      void emit_f32_add() {
         auto icount = softfloat_instr(21, 44, 58);
         emit_f32_binop(0x58, CHOOSE_FN(_eosio_f32_add));
      }
      void emit_f32_sub() {
         auto icount = softfloat_instr(21, 44, 58);
         emit_f32_binop(0x5c, CHOOSE_FN(_eosio_f32_sub));
      }
      void emit_f32_mul() {
         auto icount = softfloat_instr(21, 44, 58);
         emit_f32_binop(0x59, CHOOSE_FN(_eosio_f32_mul));
      }
      void emit_f32_div() {
         auto icount = softfloat_instr(21, 44, 58);
         emit_f32_binop(0x5e, CHOOSE_FN(_eosio_f32_div));
      }
      void emit_f32_min() {
         COUNT_INSTR();
         auto icount = softfloat_instr(47, 44, 58);
        if constexpr(use_softfloat) {
           if (_mod.eosio_fp) {
              emit_f32_binop_softfloat(CHOOSE_FN(_eosio_f32_min<false>));
           } else {
              emit_f32_binop_softfloat(CHOOSE_FN(_eosio_f32_min<true>));
           }
           return;
        }
        // mov (%rsp), %eax
        emit_bytes(0x8b, 0x04, 0x24);
        // test %eax, %eax
        emit_bytes(0x85, 0xc0);
        // je ZERO
        emit_bytes(0x0f, 0x84);
        void* zero = emit_branch_target32();
        // movss 8(%rsp), %xmm0
        emit_bytes(0xf3, 0x0f, 0x10, 0x44, 0x24, 0x08);
        // minss (%rsp), %xmm0
        emit_bytes(0xf3, 0x0f, 0x5d, 0x04, 0x24);
        // jmp DONE
        emit_bytes(0xe9);
        void* done = emit_branch_target32();
        // ZERO:
        fix_branch(zero, code);
        // movss (%rsp), %xmm0
        emit_bytes(0xf3, 0x0f, 0x10, 0x04, 0x24);
        // minss 8(%rsp), %xmm0
        emit_bytes(0xf3, 0x0f, 0x5d, 0x44, 0x24, 0x08);
        // DONE:
        fix_branch(done, code);
        // add $8, %rsp
        emit_bytes(0x48, 0x83, 0xc4, 0x08);
        // movss %xmm0, (%rsp)
        emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
      }
      void emit_f32_max() {
         COUNT_INSTR();
         auto icount = softfloat_instr(47, 44, 58);
        if(use_softfloat) {
           if (_mod.eosio_fp) {
              emit_f32_binop_softfloat(CHOOSE_FN(_eosio_f32_max<false>));
           } else {
              emit_f32_binop_softfloat(CHOOSE_FN(_eosio_f32_max<true>));
           }
           return;
        }
        // mov (%rsp), %eax
        emit_bytes(0x8b, 0x04, 0x24);
        // test %eax, %eax
        emit_bytes(0x85, 0xc0);
        // je ZERO
        emit_bytes(0x0f, 0x84);
        void* zero = emit_branch_target32();
        // movss (%rsp), %xmm0
        emit_bytes(0xf3, 0x0f, 0x10, 0x04, 0x24);
        // maxss 8(%rsp), %xmm0
        emit_bytes(0xf3, 0x0f, 0x5f, 0x44, 0x24, 0x08);
        // jmp DONE
        emit_bytes(0xe9);
        void* done = emit_branch_target32();
        // ZERO:
        fix_branch(zero, code);
        // movss 8(%rsp), %xmm0
        emit_bytes(0xf3, 0x0f, 0x10, 0x44, 0x24, 0x08);
        // maxss (%rsp), %xmm0
        emit_bytes(0xf3, 0x0f, 0x5f, 0x04, 0x24);
        // DONE:
        fix_branch(done, code);
        // add $8, %rsp
        emit_bytes(0x48, 0x83, 0xc4, 0x08);
        // movss %xmm0, (%rsp)
        emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
      }

      void emit_f32_copysign() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(16);
         // popq %rax; 
         emit_bytes(0x58);
         // andl 0x80000000, %eax
         emit_bytes(0x25);
         emit_operand32(0x80000000);
         // popq %rcx
         emit_bytes(0x59);
         // andl 0x7fffffff, %ecx
         emit_bytes(0x81, 0xe1);
         emit_operand32(0x7fffffff);
         // orl %ecx, %eax
         emit_bytes(0x09, 0xc8);
         // pushq %rax
         emit_bytes(0x50);
      }
      
      // --------------- f64 unops ----------------------

      void emit_f64_abs() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(15);
         // popq %rcx; 
         emit_bytes(0x59);
         // movabsq $0x7fffffffffffffff, %rax
         emit_bytes(0x48, 0xb8);
         emit_operand64(0x7fffffffffffffffull);
         // andq %rcx, %rax
         emit_bytes(0x48, 0x21, 0xc8);
         // pushq %rax
         emit_bytes(0x50);
      }

      void emit_f64_neg() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(15);
         // popq %rcx; 
         emit_bytes(0x59);
         // movabsq $0x8000000000000000, %rax
         emit_bytes(0x48, 0xb8);
         emit_operand64(0x8000000000000000ull);
         // xorq %rcx, %rax
         emit_bytes(0x48, 0x31, 0xc8);
         // pushq %rax
         emit_bytes(0x50);
      }

      void emit_f64_ceil() {
         COUNT_INSTR();
         auto icount = softfloat_instr(12, 38, 56);
         if constexpr (use_softfloat) {
            if (_mod.eosio_fp) {
               return emit_softfloat_unop(CHOOSE_FN(_eosio_f64_ceil<false>));
            } else {
               return emit_softfloat_unop(CHOOSE_FN(_eosio_f64_ceil<true>));
            }
         }
         // roundsd 0b1010, (%rsp), %xmm0
         emit_bytes(0x66, 0x0f, 0x3a, 0x0b, 0x04, 0x24, 0x0a);
         // movsd %xmm0, (%rsp)
         emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
      }

      void emit_f64_floor() {
         COUNT_INSTR();
         auto icount = softfloat_instr(12, 38, 56);
         if constexpr (use_softfloat) {
            if (_mod.eosio_fp) {
               return emit_softfloat_unop(CHOOSE_FN(_eosio_f64_floor<false>));
            } else {
               return emit_softfloat_unop(CHOOSE_FN(_eosio_f64_floor<true>));
            }
         }
         // roundsd 0b1001, (%rsp), %xmm0
         emit_bytes(0x66, 0x0f, 0x3a, 0x0b, 0x04, 0x24, 0x09);
         // movss %xmm0, (%rsp)
         emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
      }

      void emit_f64_trunc() {
         COUNT_INSTR();
         auto icount = softfloat_instr(12, 38, 56);
         if constexpr (use_softfloat) {
            if (_mod.eosio_fp) {
               return emit_softfloat_unop(CHOOSE_FN(_eosio_f64_trunc<false>));
            } else {
               return emit_softfloat_unop(CHOOSE_FN(_eosio_f64_trunc<true>));
            }
         }
         // roundsd 0b1011, (%rsp), %xmm0
         emit_bytes(0x66, 0x0f, 0x3a, 0x0b, 0x04, 0x24, 0x0b);
         // movss %xmm0, (%rsp)
         emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
      }

      void emit_f64_nearest() {
         COUNT_INSTR();
         auto icount = softfloat_instr(12, 38, 56);
         if constexpr (use_softfloat) {
            if (_mod.eosio_fp) {
               return emit_softfloat_unop(CHOOSE_FN(_eosio_f64_nearest<false>));
            } else {
               return emit_softfloat_unop(CHOOSE_FN(_eosio_f64_nearest<true>));
            }
         }
         // roundsd 0b1000, (%rsp), %xmm0
         emit_bytes(0x66, 0x0f, 0x3a, 0x0b, 0x04, 0x24, 0x08);
         // movss %xmm0, (%rsp)
         emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
      }

      void emit_f64_sqrt() {
         COUNT_INSTR();
         auto icount = softfloat_instr(10, 38, 56);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(_eosio_f64_sqrt));
         }
         // sqrtss (%rsp), %xmm0
         emit_bytes(0xf2, 0x0f, 0x51, 0x04, 0x24);
         // movss %xmm0, (%rsp)
         emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
      }

      // --------------- f64 binops ----------------------

      void emit_f64_add() {
         COUNT_INSTR();
         auto icount = softfloat_instr(21, 47, 61);
         emit_f64_binop(0x58, CHOOSE_FN(_eosio_f64_add));
      }
      void emit_f64_sub() {
         COUNT_INSTR();
         auto icount = softfloat_instr(21, 47, 61);
         emit_f64_binop(0x5c, CHOOSE_FN(_eosio_f64_sub));
      }
      void emit_f64_mul() {
         COUNT_INSTR();
         auto icount = softfloat_instr(21, 47, 61);
         emit_f64_binop(0x59, CHOOSE_FN(_eosio_f64_mul));
      }
      void emit_f64_div() {
         COUNT_INSTR();
         auto icount = softfloat_instr(21, 47, 61);
         emit_f64_binop(0x5e, CHOOSE_FN(_eosio_f64_div));
      }
      void emit_f64_min() {
         COUNT_INSTR();
         auto icount = softfloat_instr(49, 47, 61);
         if(use_softfloat) {
            if (_mod.eosio_fp) {
               emit_f64_binop_softfloat(CHOOSE_FN(_eosio_f64_min<false>));
            } else {
               emit_f64_binop_softfloat(CHOOSE_FN(_eosio_f64_min<true>));
            }
            return;
         }
         // mov (%rsp), %rax
         emit_bytes(0x48, 0x8b, 0x04, 0x24);
         // test %rax, %rax
         emit_bytes(0x48, 0x85, 0xc0);
         // je ZERO
         emit_bytes(0x0f, 0x84);
         void* zero = emit_branch_target32();
         // movsd 8(%rsp), %xmm0
         emit_bytes(0xf2, 0x0f, 0x10, 0x44, 0x24, 0x08);
         // minsd (%rsp), %xmm0
         emit_bytes(0xf2, 0x0f, 0x5d, 0x04, 0x24);
         // jmp DONE
         emit_bytes(0xe9);
         void* done = emit_branch_target32();
         // ZERO:
         fix_branch(zero, code);
         // movsd (%rsp), %xmm0
         emit_bytes(0xf2, 0x0f, 0x10, 0x04, 0x24);
         // minsd 8(%rsp), %xmm0
         emit_bytes(0xf2, 0x0f, 0x5d, 0x44, 0x24, 0x08);
         // DONE:
         fix_branch(done, code);
         // add $8, %rsp
         emit_bytes(0x48, 0x83, 0xc4, 0x08);
         // movsd %xmm0, (%rsp)
         emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
      }
      void emit_f64_max() {
         COUNT_INSTR();
         auto icount = softfloat_instr(49, 47, 61);
         if(use_softfloat) {
            if (_mod.eosio_fp) {
               emit_f64_binop_softfloat(CHOOSE_FN(_eosio_f64_max<false>));
            } else {
               emit_f64_binop_softfloat(CHOOSE_FN(_eosio_f64_max<true>));
            }
            return;
         }
         // mov (%rsp), %rax
         emit_bytes(0x48, 0x8b, 0x04, 0x24);
         // test %rax, %rax
         emit_bytes(0x48, 0x85, 0xc0);
         // je ZERO
         emit_bytes(0x0f, 0x84);
         void* zero = emit_branch_target32();
         // maxsd (%rsp), %xmm0
         emit_bytes(0xf2, 0x0f, 0x10, 0x04, 0x24);
         // maxsd 8(%rsp), %xmm0
         emit_bytes(0xf2, 0x0f, 0x5f, 0x44, 0x24, 0x08);
         // jmp DONE
         emit_bytes(0xe9);
         void* done = emit_branch_target32();
         // ZERO:
         fix_branch(zero, code);
         // movsd 8(%rsp), %xmm0
         emit_bytes(0xf2, 0x0f, 0x10, 0x44, 0x24, 0x08);
         // maxsd (%rsp), %xmm0
         emit_bytes(0xf2, 0x0f, 0x5f, 0x04, 0x24);
         // DONE:
         fix_branch(done, code);
         // add $8, %rsp
         emit_bytes(0x48, 0x83, 0xc4, 0x08);
         // movsd %xmm0, (%rsp)
         emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
      }

      void emit_f64_copysign() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(25);
         // popq %rcx; 
         emit_bytes(0x59);
         // movabsq 0x8000000000000000, %rax
         emit_bytes(0x48, 0xb8);
         emit_operand64(0x8000000000000000ull);
         // andq %rax, %rcx
         emit_bytes(0x48, 0x21, 0xc1);
         // popq %rdx
         emit_bytes(0x5a);
         // notq %rax
         emit_bytes(0x48, 0xf7, 0xd0);
         // andq %rdx, %rax
         emit_bytes(0x48, 0x21, 0xd0);
         // orq %rcx, %rax
         emit_bytes(0x48, 0x09, 0xc8);
         // pushq %rax
         emit_bytes(0x50);
      }

      // --------------- conversions --------------------


      void emit_i32_wrap_i64() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(4);
         emit_pop(rax);
         emit_mov(eax, eax);
         emit_push(rax);
      }

      void emit_i32_trunc_s_f32() {
         COUNT_INSTR();
         auto icount = softfloat_instr(33, 36, 54);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(softfloat_trap<&_eosio_f32_trunc_i32s>()));
         }
         // cvttss2si 8(%rsp), %eax
         emit_f2i(0xf3, 0x0f, 0x2c, 0x44, 0x24, 0x08);
         // mov %eax, (%rsp)
         emit_bytes(0x89, 0x04 ,0x24);
      }

      void emit_i32_trunc_u_f32() {
         COUNT_INSTR();
         auto icount = softfloat_instr(46, 36, 54);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(softfloat_trap<&_eosio_f32_trunc_i32u>()));
         }
         // cvttss2si 8(%rsp), %rax
         emit_f2i(0xf3, 0x48, 0x0f, 0x2c, 0x44, 0x24, 0x08);
         // mov %eax, (%rsp)
         emit_bytes(0x89, 0x04 ,0x24);
         // shr $32, %rax
         emit_bytes(0x48, 0xc1, 0xe8, 0x20);
         // test %eax, %eax
         emit_bytes(0x85, 0xc0);
         // jnz FP_ERROR_HANDLER
         emit_bytes(0x0f, 0x85);
         fix_branch(emit_branch_target32(), fpe_handler);
      }
      void emit_i32_trunc_s_f64() {
         COUNT_INSTR();
         auto icount = softfloat_instr(34, 38, 56);
         if constexpr (use_softfloat) {
            if (_mod.eosio_fp) {
               return emit_softfloat_unop(CHOOSE_FN(softfloat_trap<&_eosio_f64_trunc_i32s<false>>()));
            } else {
               return emit_softfloat_unop(CHOOSE_FN(softfloat_trap<&_eosio_f64_trunc_i32s<true>>()));
            }
         }
         // cvttsd2si 8(%rsp), %eax
         emit_f2i(0xf2, 0x0f, 0x2c, 0x44, 0x24, 0x08);
         // movq %rax, (%rsp)
         emit_bytes(0x48, 0x89, 0x04 ,0x24);
      }

      void emit_i32_trunc_u_f64() {
         COUNT_INSTR();
         auto icount = softfloat_instr(47, 38, 56);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(softfloat_trap<&_eosio_f64_trunc_i32u>()));
         }
         // cvttsd2si 8(%rsp), %rax
         emit_f2i(0xf2, 0x48, 0x0f, 0x2c, 0x44, 0x24, 0x08);
         // movq %rax, (%rsp)
         emit_bytes(0x48, 0x89, 0x04 ,0x24);
         // shr $32, %rax
         emit_bytes(0x48, 0xc1, 0xe8, 0x20);
         // test %eax, %eax
         emit_bytes(0x85, 0xc0);
         // jnz FP_ERROR_HANDLER
         emit_bytes(0x0f, 0x85);
         fix_branch(emit_branch_target32(), fpe_handler);
      }

      void emit_i32_trunc_sat_f32_s() {
         COUNT_INSTR();
         emit_softfloat_unop(CHOOSE_FN(_eosio_i32_trunc_sat_f32_s));
      }

      void emit_i32_trunc_sat_f32_u() {
         COUNT_INSTR();
         emit_softfloat_unop(CHOOSE_FN(_eosio_i32_trunc_sat_f32_u));
      }

      void emit_i32_trunc_sat_f64_s() {
         COUNT_INSTR();
         emit_softfloat_unop(CHOOSE_FN(_eosio_i32_trunc_sat_f64_s));
      }

      void emit_i32_trunc_sat_f64_u() {
         COUNT_INSTR();
         emit_softfloat_unop(CHOOSE_FN(_eosio_i32_trunc_sat_f64_u));
      }

      void emit_i32_extend8_s() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(7);
         emit(MOVSXB, *rsp, eax);
         emit_mov(eax, *rsp);
      }

      void emit_i32_extend16_s() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(7);
         emit(MOVSXW, *rsp, eax);
         emit_mov(eax, *rsp);
      }

      void emit_i64_extend_s_i32() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(8);
         // movslq (%rsp), %rax
         emit_bytes(0x48, 0x63, 0x04, 0x24);
         // mov %rax, (%rsp)
         emit_bytes(0x48, 0x89, 0x04, 0x24);
      }

      void emit_i64_extend_u_i32() { /* Nothing to do */ }

      void emit_i64_extend8_s() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(9);
         emit(MOVSXB, *rsp, rax);
         emit_mov(rax, *rsp);
      }

      void emit_i64_extend16_s() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(9);
         emit(MOVSXW, *rsp, rax);
         emit_mov(rax, *rsp);
      }

      void emit_i64_extend32_s() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(8);
         emit(MOVSXD, *rsp, rax);
         emit_mov(rax, *rsp);
      }

      void emit_i64_trunc_s_f32() {
         COUNT_INSTR();
         auto icount = softfloat_instr(35, 37, 55);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(softfloat_trap<&_eosio_f32_trunc_i64s>()));
         }
         // cvttss2si (%rsp), %rax
         emit_f2i(0xf3, 0x48, 0x0f, 0x2c, 0x44, 0x24, 0x08);
         // mov %rax, (%rsp)
         emit_bytes(0x48, 0x89, 0x04 ,0x24);
      }
      void emit_i64_trunc_u_f32() {
         COUNT_INSTR();
         auto icount = softfloat_instr(101, 37, 55);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(softfloat_trap<&_eosio_f32_trunc_i64u>()));
         }
         // mov $0x5f000000, %eax
         emit_bytes(0xb8);
         emit_operand32(0x5f000000);
         // movss (%rsp), %xmm0
         emit_bytes(0xf3, 0x0f, 0x10, 0x04, 0x24);
         // mov %eax, (%rsp)
         emit_bytes(0x89, 0x04, 0x24);
         // movss (%rsp), %xmm1
         emit_bytes(0xf3, 0x0f, 0x10, 0x0c, 0x24);
         // movaps %xmm0, %xmm2
         emit_bytes(0x0f, 0x28, 0xd0);
         // subss %xmm1, %xmm2
         emit_bytes(0xf3, 0x0f, 0x5c, 0xd1);
         // cvttss2siq %xmm2, %rax
         emit_f2i(0xf3, 0x48, 0x0f, 0x2c, 0xc2);
         // movabsq $0x8000000000000000, %rcx
         emit_bytes(0x48, 0xb9);
         emit_operand64(0x8000000000000000);
         // xorq %rax, %rcx
         emit_bytes(0x48, 0x31, 0xc1);
         // cvttss2siq %xmm0, %rax
         emit_bytes(0xf3, 0x48, 0x0f, 0x2c, 0xc0);
         // xor %rdx, %rdx
         emit_bytes(0x48, 0x31, 0xd2);
         // ucomiss %xmm0, %xmm1
         emit_bytes(0x0f, 0x2e, 0xc8);
         // cmovaq %rax, %rdx
         emit_bytes(0x48, 0x0f, 0x47, 0xd0);
         // cmovbeq %rcx, %rax
         emit_bytes(0x48, 0x0f, 0x46, 0xc1);
         // mov %rax, (%rsp)
         emit_bytes(0x48, 0x89, 0x04, 0x24);
         // bt $63, %rdx
         emit_bytes(0x48, 0x0f, 0xba, 0xe2, 0x3f);
         // jc FP_ERROR_HANDLER
         emit_bytes(0x0f, 0x82);
         fix_branch(emit_branch_target32(), fpe_handler);
      }
      void emit_i64_trunc_s_f64() {
         COUNT_INSTR();
         auto icount = softfloat_instr(35, 38, 56);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(softfloat_trap<&_eosio_f64_trunc_i64s>()));
         }
         // cvttsd2si (%rsp), %rax
         emit_f2i(0xf2, 0x48, 0x0f, 0x2c, 0x44, 0x24, 0x08);
         // mov %rax, (%rsp)
         emit_bytes(0x48, 0x89, 0x04 ,0x24);
      }
      void emit_i64_trunc_u_f64() {
         COUNT_INSTR();
         auto icount = softfloat_instr(109, 38, 56);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(softfloat_trap<&_eosio_f64_trunc_i64u>()));
         }
         // movabsq $0x43e0000000000000, %rax
         emit_bytes(0x48, 0xb8);
         emit_operand64(0x43e0000000000000);
         // movsd (%rsp), %xmm0
         emit_bytes(0xf2, 0x0f, 0x10, 0x04, 0x24);
         // movq %rax, (%rsp)
         emit_bytes(0x48, 0x89, 0x04, 0x24);
         // movsd (%rsp), %xmm1
         emit_bytes(0xf2, 0x0f, 0x10, 0x0c, 0x24);
         // movapd %xmm0, %xmm2
         emit_bytes(0x66, 0x0f, 0x28, 0xd0);
         // subsd %xmm1, %xmm2
         emit_bytes(0xf2, 0x0f, 0x5c, 0xd1);
         // cvttsd2siq %xmm2, %rax
         emit_f2i(0xf2, 0x48, 0x0f, 0x2c, 0xc2);
         // movabsq $0x8000000000000000, %rcx
         emit_bytes(0x48, 0xb9);
         emit_operand64(0x8000000000000000);
         // xorq %rax, %rcx
         emit_bytes(0x48, 0x31, 0xc1);
         // cvttsd2siq %xmm0, %rax
         emit_bytes(0xf2, 0x48, 0x0f, 0x2c, 0xc0);
         // xor %rdx, %rdx
         emit_bytes(0x48, 0x31, 0xd2);
         // ucomisd %xmm0, %xmm1
         emit_bytes(0x66, 0x0f, 0x2e, 0xc8);
         // cmovaq %rax, %rdx
         emit_bytes(0x48, 0x0f, 0x47, 0xd0);
         // cmovbeq %rcx, %rax
         emit_bytes(0x48, 0x0f, 0x46, 0xc1);
         // mov %rax, (%rsp)
         emit_bytes(0x48, 0x89, 0x04, 0x24);
         // bt $63, %rdx
         emit_bytes(0x48, 0x0f, 0xba, 0xe2, 0x3f);
         // jc FP_ERROR_HANDLER
         emit_bytes(0x0f, 0x82);
         fix_branch(emit_branch_target32(), fpe_handler);
      }

      void emit_i64_trunc_sat_f32_s() {
         COUNT_INSTR();
         emit_softfloat_unop(CHOOSE_FN(_eosio_i64_trunc_sat_f32_s));
      }

      void emit_i64_trunc_sat_f32_u() {
         COUNT_INSTR();
         emit_softfloat_unop(CHOOSE_FN(_eosio_i64_trunc_sat_f32_u));
      }

      void emit_i64_trunc_sat_f64_s() {
         COUNT_INSTR();
         emit_softfloat_unop(CHOOSE_FN(_eosio_i64_trunc_sat_f64_s));
      }

      void emit_i64_trunc_sat_f64_u() {
         COUNT_INSTR();
         emit_softfloat_unop(CHOOSE_FN(_eosio_i64_trunc_sat_f64_u));
      }

      void emit_f32_convert_s_i32() {
         COUNT_INSTR();
         auto icount = softfloat_instr(10, 36, 54);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(_eosio_i32_to_f32));
         }
         // cvtsi2ssl (%rsp), %xmm0
         emit_bytes(0xf3, 0x0f, 0x2a, 0x04, 0x24);
         // movss %xmm0, (%rsp)
         emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
      }
      void emit_f32_convert_u_i32() {
         COUNT_INSTR();
         auto icount = softfloat_instr(11, 36, 54);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(_eosio_ui32_to_f32));
         }
         // zero-extend to 64-bits
         // cvtsi2sslq (%rsp), %xmm0
         emit_bytes(0xf3, 0x48, 0x0f, 0x2a, 0x04, 0x24);
         // movss %xmm0, (%rsp)
         emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
      }
      void emit_f32_convert_s_i64() {
         COUNT_INSTR();
         auto icount = softfloat_instr(11, 38, 56);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(_eosio_i64_to_f32));
         }
         // cvtsi2sslq (%rsp), %xmm0
         emit_bytes(0xf3, 0x48, 0x0f, 0x2a, 0x04, 0x24);
         // movss %xmm0, (%rsp)
         emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
      }
      void emit_f32_convert_u_i64() {
         COUNT_INSTR();
         auto icount = softfloat_instr(55, 38, 56);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(_eosio_ui64_to_f32));
         }
        // movq (%rsp), %rax
        emit_bytes(0x48, 0x8b, 0x04, 0x24);
        // testq %rax, %rax
        emit_bytes(0x48, 0x85, 0xc0);
        // js LARGE
        emit_bytes(0x0f, 0x88);
        void * large = emit_branch_target32();
        // cvtsi2ssq %rax, %xmm0
        emit_bytes(0xf3, 0x48, 0x0f, 0x2a, 0xc0);
        // jmp done
        emit_bytes(0xe9);
        void* done = emit_branch_target32();
        // LARGE:
        fix_branch(large, code);
        // movq %rax, %rcx
        emit_bytes(0x48, 0x89, 0xc1);
        // shrq %rax
        emit_bytes(0x48, 0xd1, 0xe8);
        // andl $1, %ecx
        emit_bytes(0x83, 0xe1, 0x01);
        // orq %rcx, %rax
        emit_bytes(0x48, 0x09, 0xc8);
        // cvtsi2ssq %rax, %xmm0
        emit_bytes(0xf3, 0x48, 0x0f, 0x2a, 0xc0);
        // addss %xmm0, %xmm0
        emit_bytes(0xf3, 0x0f, 0x58, 0xc0);
        // DONE:
        fix_branch(done, code);
        // xorl %eax, %eax
        emit_bytes(0x31, 0xc0);
        // movl %eax, 4(%rsp)
        emit_bytes(0x89, 0x44, 0x24, 0x04);
        // movss %xmm0, (%rsp)
        emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
      }
      void emit_f32_demote_f64() {
         COUNT_INSTR();
         auto icount = softfloat_instr(16, 38, 56);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(_eosio_f64_demote));
         }
         // cvtsd2ss (%rsp), %xmm0
         emit_bytes(0xf2, 0x0f, 0x5a, 0x04, 0x24);
         // movss %xmm0, (%rsp)
         emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
         // Zero out the high 4 bytes
         // xor %eax, %eax
         emit_bytes(0x31, 0xc0);
         // mov %eax, 4(%rsp)
         emit_bytes(0x89, 0x44, 0x24, 0x04);
      }
      void emit_f64_convert_s_i32() {
         COUNT_INSTR();
         auto icount = softfloat_instr(10, 37, 55);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(_eosio_i32_to_f64));
         }
         // cvtsi2sdl (%rsp), %xmm0
         emit_bytes(0xf2, 0x0f, 0x2a, 0x04, 0x24);
         // movsd %xmm0, (%rsp)
         emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
      }
      void emit_f64_convert_u_i32() {
         COUNT_INSTR();
         auto icount = softfloat_instr(11, 37, 55);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(_eosio_ui32_to_f64));
         }
         //  cvtsi2sdq (%rsp), %xmm0
         emit_bytes(0xf2, 0x48, 0x0f, 0x2a, 0x04, 0x24);
         // movsd %xmm0, (%rsp)
         emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
      }
      void emit_f64_convert_s_i64() {
         COUNT_INSTR();
         auto icount = softfloat_instr(11, 38, 56);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(_eosio_i64_to_f64));
         }
         //  cvtsi2sdq (%rsp), %xmm0
         emit_bytes(0xf2, 0x48, 0x0f, 0x2a, 0x04, 0x24);
         // movsd %xmm0, (%rsp)
         emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
      }
      void emit_f64_convert_u_i64() {
         COUNT_INSTR();
         auto icount = softfloat_instr(49, 38, 56);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(_eosio_ui64_to_f64));
         }
        // movq (%rsp), %rax
        emit_bytes(0x48, 0x8b, 0x04, 0x24);
        // testq %rax, %rax
        emit_bytes(0x48, 0x85, 0xc0);
        // js LARGE
        emit_bytes(0x0f, 0x88);
        void * large = emit_branch_target32();
        // cvtsi2sdq %rax, %xmm0
        emit_bytes(0xf2, 0x48, 0x0f, 0x2a, 0xc0);
        // jmp done
        emit_bytes(0xe9);
        void* done = emit_branch_target32();
        // LARGE:
        fix_branch(large, code);
        // movq %rax, %rcx
        emit_bytes(0x48, 0x89, 0xc1);
        // shrq %rax
        emit_bytes(0x48, 0xd1, 0xe8);
        // andl $1, %ecx
        emit_bytes(0x83, 0xe1, 0x01);
        // orq %rcx, %rax
        emit_bytes(0x48, 0x09, 0xc8);
        // cvtsi2sdq %rax, %xmm0
        emit_bytes(0xf2, 0x48, 0x0f, 0x2a, 0xc0);
        // addsd %xmm0, %xmm0
        emit_bytes(0xf2, 0x0f, 0x58, 0xc0);
        // DONE:
        fix_branch(done, code);
        // movsd %xmm0, (%rsp)
        emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
      }
      void emit_f64_promote_f32() {
         COUNT_INSTR();
         auto icount = softfloat_instr(10, 37, 55);
         if constexpr (use_softfloat) {
            return emit_softfloat_unop(CHOOSE_FN(_eosio_f32_promote));
         }
         // cvtss2sd (%rsp), %xmm0
         emit_bytes(0xf3, 0x0f, 0x5a, 0x04, 0x24);
         // movsd %xmm0, (%rsp)
         emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
      }
      
      void emit_i32_reinterpret_f32() { /* Nothing to do */ }
      void emit_i64_reinterpret_f64() { /* Nothing to do */ }
      void emit_f32_reinterpret_i32() { /* Nothing to do */ }
      void emit_f64_reinterpret_i64() { /* Nothing to do */ }

#undef CHOOSE_FN

      void emit_v128_load(uint32_t align, uint32_t offset) {
         COUNT_INSTR();
         emit_v128_loadop(VMOVDQU_A, align, offset);
      }

      void emit_v128_load8x8_s(uint32_t align, uint32_t offset) {
         COUNT_INSTR();
         emit_v128_loadop(VPMOVSXBW, align, offset);
      }

      void emit_v128_load8x8_u(uint32_t align, uint32_t offset) {
         COUNT_INSTR();
         emit_v128_loadop(VPMOVZXBW, align, offset);
      }

      void emit_v128_load16x4_s(uint32_t align, uint32_t offset) {
         COUNT_INSTR();
         emit_v128_loadop(VPMOVSXWD, align, offset);
      }

      void emit_v128_load16x4_u(uint32_t align, uint32_t offset) {
         COUNT_INSTR();
         emit_v128_loadop(VPMOVZXWD, align, offset);
      }

      void emit_v128_load32x2_s(uint32_t align, uint32_t offset) {
         COUNT_INSTR();
         emit_v128_loadop(VPMOVSXDQ, align, offset);
      }

      void emit_v128_load32x2_u(uint32_t align, uint32_t offset) {
         COUNT_INSTR();
         emit_v128_loadop(VPMOVZXDQ, align, offset);
      }

      void emit_v128_load8_splat(uint32_t align, uint32_t offset) {
         COUNT_INSTR();
         emit_v128_loadop(VPBROADCASTB, align, offset);
      }

      void emit_v128_load16_splat(uint32_t align, uint32_t offset) {
         COUNT_INSTR();
         emit_v128_loadop(VPBROADCASTW, align, offset);
      }

      void emit_v128_load32_splat(uint32_t align, uint32_t offset) {
         COUNT_INSTR();
         emit_v128_loadop(VPBROADCASTD, align, offset);
      }

      void emit_v128_load64_splat(uint32_t align, uint32_t offset) {
         COUNT_INSTR();
         emit_v128_loadop(VPBROADCASTQ, align, offset);
      }

      void emit_v128_load32_zero(uint32_t align, uint32_t offset) {
         COUNT_INSTR();
         emit_v128_loadop(VMOVD_A, align, offset);
      }

      void emit_v128_load64_zero(uint32_t align, uint32_t offset) {
         COUNT_INSTR();
         emit_v128_loadop(VMOVQ_A, align, offset);
      }

      void emit_v128_store(uint32_t /*align*/, uint32_t offset) {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovups(*rsp, xmm0);
         emit_add(16, rsp);
         emit_pop_address(offset);
         emit_vmovups(xmm0, *rax);
      }

      void emit_v128_load8_lane(uint32_t align, uint32_t offset, uint8_t lane) {
         COUNT_INSTR();
         emit_v128_load_laneop(VPINSRB, align, offset, lane);
      }

      void emit_v128_load16_lane(uint32_t align, uint32_t offset, uint8_t lane) {
         COUNT_INSTR();
         emit_v128_load_laneop(VPINSRW, align, offset, lane);
      }

      void emit_v128_load32_lane(uint32_t align, uint32_t offset, uint8_t lane) {
         COUNT_INSTR();
         emit_v128_load_laneop(VPINSRD, align, offset, lane);
      }

      void emit_v128_load64_lane(uint32_t align, uint32_t offset, uint8_t lane) {
         COUNT_INSTR();
         emit_v128_load_laneop(VPINSRQ, align, offset, lane);
      }

      void emit_v128_store8_lane(uint32_t align, uint32_t offset, uint8_t lane) {
         COUNT_INSTR();
         emit_v128_store_laneop(VPEXTRB, align, offset, lane);
      }

      void emit_v128_store16_lane(uint32_t align, uint32_t offset, uint8_t lane) {
         COUNT_INSTR();
         emit_v128_store_laneop(VPEXTRW, align, offset, lane);
      }

      void emit_v128_store32_lane(uint32_t align, uint32_t offset, uint8_t lane) {
         COUNT_INSTR();
         emit_v128_store_laneop(VPEXTRD, align, offset, lane);
      }

      void emit_v128_store64_lane(uint32_t align, uint32_t offset, uint8_t lane) {
         COUNT_INSTR();
         emit_v128_store_laneop(VPEXTRQ, align, offset, lane);
      }

      void emit_v128_const(v128_t value) {
         COUNT_INSTR();
         auto icount = simd_instr();
         uint64_t low,high;
         memcpy(&high, reinterpret_cast<const char*>(&value) + 8, 8);
         memcpy(&low, &value, 8);
         emit_i64_const(high);
         emit_i64_const(low);
      }

      void emit_i8x16_shuffle(const uint8_t lanes[16])
      {
         COUNT_INSTR();
         auto icount = fixed_size_instr(72);
         auto emit_shuffle_operand = [this](const uint8_t lanes[8]) {
            for(int i = 0; i < 8; ++i)
            {
               if(lanes[i] < 16)
               {
                  emit_byte(~lanes[i]);
               }
               else
               {
                  emit_byte(lanes[i]);
               }
            }
         };
         // general case:
         // movabsq $lanes[0-7], %rax
         emit_bytes(0x48, 0xb8);
         emit_shuffle_operand(lanes);
         emit_vmovq(rax, xmm2);
         // movabsq $lanes[8-15], %rax
         emit_bytes(0x48, 0xb8);
         emit_shuffle_operand(lanes + 8);
         emit(VPINSRQ, imm8{1}, rax, xmm2, xmm2);

         emit_vmovdqu(*rsp, xmm0);
         emit(VPSHUFB, xmm2, xmm0, xmm1);
         emit_const_ones(xmm0);
         emit(VPXOR, xmm0, xmm2, xmm2);
         emit_add(16, rsp);
         emit_vmovdqu(*rsp, xmm0);
         emit(VPSHUFB, xmm2, xmm0, xmm0);
         emit(VPOR, xmm1, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i8x16_extract_lane_s(uint8_t l) {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         emit_vpextrb(l, xmm0, rax);
         emit(MOVSXB, al, eax);
         emit_push(rax);
      }

      void emit_i8x16_extract_lane_u(uint8_t l) {
         COUNT_INSTR();
         emit_v128_extract_laneop(VPEXTRB, l);
      }

      void emit_i8x16_replace_lane(uint8_t l) {
         COUNT_INSTR();
         emit_v128_replace_laneop(VPINSRB, l);
      }

      void emit_i16x8_extract_lane_s(uint8_t l) {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         emit_vpextrw(l, xmm0, rax);
         emit(MOVSXW, ax, eax);
         emit_push(rax);
      }

      void emit_i16x8_extract_lane_u(uint8_t l) {
         COUNT_INSTR();
         emit_v128_extract_laneop(VPEXTRW, l);
      }

      void emit_i16x8_replace_lane(uint8_t l) {
         COUNT_INSTR();
         emit_v128_replace_laneop(VPINSRW, l);
      }

      void emit_i32x4_extract_lane(uint8_t l) {
         COUNT_INSTR();
         emit_v128_extract_laneop(VPEXTRD, l);
      }

      void emit_i32x4_replace_lane(uint8_t l) {
         COUNT_INSTR();
         emit_v128_replace_laneop(VPINSRD, l);
      }

      void emit_i64x2_extract_lane(uint8_t l) {
         COUNT_INSTR();
         emit_v128_extract_laneop(VPEXTRQ, l);
      }

      void emit_i64x2_replace_lane(uint8_t l) {
         COUNT_INSTR();
         emit_v128_replace_laneop(VPINSRQ, l);
      }

      void emit_f32x4_extract_lane(uint8_t l) {
         COUNT_INSTR();
         emit_v128_extract_laneop(VPEXTRD, l);
      }

      void emit_f32x4_replace_lane(uint8_t l) {
         COUNT_INSTR();
         emit_v128_replace_laneop(VPINSRD, l);
      }

      void emit_f64x2_extract_lane(uint8_t l) {
         COUNT_INSTR();
         emit_v128_extract_laneop(VPEXTRQ, l);
      }

      void emit_f64x2_replace_lane(uint8_t l) {
         COUNT_INSTR();
         emit_v128_replace_laneop(VPINSRQ, l);
      }

      void emit_i8x16_swizzle() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         emit_mov(0x70707070, eax);
         emit_vmovd(eax, xmm1);
         emit(VPSHUFD, imm8{0}, xmm1, xmm1);
         emit(VPADDUSB, xmm1, xmm0, xmm1);
         emit_vmovdqu(*rsp, xmm0);
         emit(VPSHUFB, xmm1, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i8x16_splat() {
         COUNT_INSTR();
         emit_v128_splatop(VPBROADCASTB);
      }

      void emit_i16x8_splat() {
         COUNT_INSTR();
         emit_v128_splatop(VPBROADCASTW);
      }

      void emit_i32x4_splat() {
         COUNT_INSTR();
         emit_v128_splatop(VPBROADCASTD);
      }

      void emit_i64x2_splat() {
         COUNT_INSTR();
         emit_v128_splatop(VPBROADCASTQ);
      }

      void emit_f32x4_splat() {
         COUNT_INSTR();
         emit_v128_splatop(VPBROADCASTD);
      }

      void emit_f64x2_splat() {
         COUNT_INSTR();
         emit_v128_splatop(VPBROADCASTQ);
      }

      // ---------------- i8x16 compare ---------------------

      void emit_i8x16_eq() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPEQB, true, false);
      }

      void emit_i8x16_ne() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPEQB, true, true);
      }

      void emit_i8x16_lt_s() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPGTB, true, false);
      }

      void emit_i8x16_lt_u() {
         COUNT_INSTR();
         emit_v128_irelop_minmax(VPMINUB, VPCMPEQB, false);
      }

      void emit_i8x16_gt_s() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPGTB, false, false);
      }

      void emit_i8x16_gt_u() {
         COUNT_INSTR();
         emit_v128_irelop_minmax(VPMAXUB, VPCMPEQB, false);
      }

      void emit_i8x16_le_s() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPGTB, false, true);
      }

      void emit_i8x16_le_u() {
         COUNT_INSTR();
         emit_v128_irelop_minmax(VPMAXUB, VPCMPEQB, true);
      }

      void emit_i8x16_ge_s() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPGTB, true, true);
      }

      void emit_i8x16_ge_u() {
         COUNT_INSTR();
         emit_v128_irelop_minmax(VPMINUB, VPCMPEQB, true);
      }

      // ---------------- i16x8 compare --------------------

      void emit_i16x8_eq() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPEQW, true, false);
      }

      void emit_i16x8_ne() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPEQW, true, true);
      }

      void emit_i16x8_lt_s() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPGTW, true, false);
      }

      void emit_i16x8_lt_u() {
         COUNT_INSTR();
         emit_v128_irelop_minmax(VPMINUW, VPCMPEQW, false);
      }

      void emit_i16x8_gt_s() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPGTW, false, false);
      }

      void emit_i16x8_gt_u() {
         COUNT_INSTR();
         emit_v128_irelop_minmax(VPMAXUW, VPCMPEQW, false);
      }

      void emit_i16x8_le_s() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPGTW, false, true);
      }

      void emit_i16x8_le_u() {
         COUNT_INSTR();
         emit_v128_irelop_minmax(VPMAXUW, VPCMPEQW, true);
      }

      void emit_i16x8_ge_s() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPGTW, true, true);
      }

      void emit_i16x8_ge_u() {
         COUNT_INSTR();
         emit_v128_irelop_minmax(VPMINUW, VPCMPEQW, true);
      }

      // i32x4 compare

      void emit_i32x4_eq() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPEQD, true, false);
      }

      void emit_i32x4_ne() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPEQD, true, true);
      }

      void emit_i32x4_lt_s() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPGTD, true, false);
      }

      void emit_i32x4_lt_u() {
         COUNT_INSTR();
         emit_v128_irelop_minmax(VPMINUD, VPCMPEQD, false);
      }

      void emit_i32x4_gt_s() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPGTD, false, false);
      }

      void emit_i32x4_gt_u() {
         COUNT_INSTR();
         emit_v128_irelop_minmax(VPMAXUD, VPCMPEQD, false);
      }

      void emit_i32x4_le_s() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPGTD, false, true);
      }

      void emit_i32x4_le_u() {
         COUNT_INSTR();
         emit_v128_irelop_minmax(VPMAXUD, VPCMPEQD, true);
      }

      void emit_i32x4_ge_s() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPGTD, true, true);
      }

      void emit_i32x4_ge_u() {
         COUNT_INSTR();
         emit_v128_irelop_minmax(VPMINUD, VPCMPEQD, true);
      }

      // --------------- i64x2 compare ---------------------

      void emit_i64x2_eq() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPEQQ, true, false);
      }

      void emit_i64x2_ne() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPEQQ, true, true);
      }

      void emit_i64x2_lt_s() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPGTQ, true, false);
      }

      void emit_i64x2_gt_s() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPGTQ, false, false);
      }

      void emit_i64x2_le_s() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPGTQ, false, true);
      }

      void emit_i64x2_ge_s() {
         COUNT_INSTR();
         emit_v128_irelop_cmp(VPCMPGTQ, true, true);
      }

      // --------------------- f32x4 compare ------------------------

      void emit_f32x4_eq() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f32x4_eq);
      }

      void emit_f32x4_ne() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f32x4_ne);
      }

      void emit_f32x4_lt() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f32x4_lt);
      }

      void emit_f32x4_gt() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f32x4_gt);
      }

      void emit_f32x4_le() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f32x4_le);
      }

      void emit_f32x4_ge() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f32x4_ge);
      }

      // ------------------- f64x2 compare -------------------

      void emit_f64x2_eq() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f64x2_eq);
      }

      void emit_f64x2_ne() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f64x2_ne);
      }

      void emit_f64x2_lt() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f64x2_lt);
      }

      void emit_f64x2_gt() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f64x2_gt);
      }

      void emit_f64x2_le() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f64x2_le);
      }

      void emit_f64x2_ge() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f64x2_ge);
      }

      // --------------- v128 logical ops ----------------

      void emit_v128_not() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_const_ones(xmm1);
         emit(VPXOR, xmm1, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_v128_and() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPAND);
      }

      void emit_v128_andnot() {
         COUNT_INSTR();
         emit_v128_binop_r(VPANDN);
      }

      void emit_v128_or() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPOR);
      }

      void emit_v128_xor() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPXOR);
      }

      void emit_v128_bitselect() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm2);
         emit_vmovdqu(*(rsp + 16), xmm1);
         emit_add(32, rsp);
         emit_vmovdqu(*rsp, xmm0);
         // With AVX512: VPTERNLOGD 0xAC
         emit(VPAND, xmm0, xmm2, xmm0);
         emit(VPANDN, xmm1, xmm2, xmm1);
         emit(VPOR, xmm0, xmm1, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_v128_any_true() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         emit_xor(eax, eax);
         emit(VPTEST, xmm0, xmm0);
         emit(SETNZ, al);
         emit_push(rax);
      }

      void emit_i8x16_abs() {
         COUNT_INSTR();
         emit_v128_unop(VPABSB);
      }

      void emit_i8x16_neg() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_const_zero(xmm0);
         emit(VPSUBB, *rsp, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i8x16_popcnt() {
         COUNT_INSTR();
         auto icount = simd_instr();
         static const uint8_t popcnt4[] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};

         // movabsq $popcnt4, %rax
         emit_bytes(0x48, 0xb8);
         emit_operand_ptr(&popcnt4);
         emit_vmovdqu(*rsp, xmm0);
         emit_vmovdqu(*rax, xmm3);
         emit_mov(0x0f, al);
         emit_vmovd(eax, xmm2);
         emit(VPBROADCASTB, xmm2, xmm2);
         emit(VPSRLQ_c, imm8{4}, xmm0, xmm1);
         emit(VPAND, xmm2, xmm0, xmm0);
         emit(VPAND, xmm2, xmm1, xmm1);
         emit(VPSHUFB, xmm0, xmm3, xmm0);
         emit(VPSHUFB, xmm1, xmm3, xmm1);
         emit(VPADDB, xmm0, xmm1, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i8x16_all_true() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_const_zero(xmm0);
         emit(VPCMPEQB, *rsp, xmm0, xmm0);
         emit_add(16, rsp);
         emit_xor(eax, eax);
         emit(VPTEST, xmm0, xmm0);
         emit(SETZ, al);
         emit_push(rax);
      }

      void emit_i8x16_bitmask() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         emit(VPMOVMSKB, xmm0, rax);
         emit_push(rax);
      }

      void emit_i8x16_narrow_i16x8_s() {
         COUNT_INSTR();
         emit_v128_binop(VPACKSSWB);
      }

      void emit_i8x16_narrow_i16x8_u() {
         COUNT_INSTR();
         emit_v128_binop(VPACKUSWB);
      }

      void emit_i8x16_shl() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_pop(rax);
         emit_vmovdqu(*rsp, xmm0);
         emit_and(7, eax);
         emit_vmovd(eax, xmm2);
         emit_const_ones(xmm1);
         emit(VPSLLD, xmm2, xmm1, xmm1);
         emit(VPBROADCASTB, xmm1, xmm1);
         emit(VPSLLW, xmm2, xmm0, xmm0);
         emit(VPAND, xmm0, xmm1, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i8x16_shr_s() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_pop(rax);
         emit_vmovdqu(*rsp, xmm0);
         emit_and(7, eax);
         emit_vmovd(eax, xmm2);
         emit_const_ones(xmm3);
         emit(VPSLLW_c, imm8{8}, xmm3, xmm3);
         emit(VPSLLW_c, imm8{8}, xmm0, xmm1);
         emit(VPSRAW_c, imm8{8}, xmm1, xmm1);
         emit(VPSLLW, xmm2, xmm3, xmm3);
         emit(VPANDN, xmm1, xmm3, xmm1);
         emit(VPAND, xmm3, xmm0, xmm0);
         emit(VPOR, xmm1, xmm0, xmm0);
         emit(VPSRAW, xmm2, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i8x16_shr_u() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_pop(rax);
         emit_vmovdqu(*rsp, xmm0);
         emit_and(7, eax);
         emit_vmovd(eax, xmm2);
         emit_const_ones(xmm1);
         emit(VPSLLW_c, imm8{8}, xmm1, xmm1);
         emit(VPSRLW, xmm2, xmm1, xmm1);
         emit(VPBROADCASTB, xmm1, xmm1);
         emit(VPSRLW, xmm2, xmm0, xmm0);
         emit(VPANDN, xmm0, xmm1, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i8x16_add() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPADDB);
      }

      void emit_i8x16_add_sat_s() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPADDSB);
      }

      void emit_i8x16_add_sat_u() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPADDUSB);
      }

      void emit_i8x16_sub() {
         COUNT_INSTR();
         emit_v128_binop(VPSUBB);
      }

      void emit_i8x16_sub_sat_s() {
         COUNT_INSTR();
         emit_v128_binop(VPSUBSB);
      }

      void emit_i8x16_sub_sat_u() {
         COUNT_INSTR();
         emit_v128_binop(VPSUBUSB);
      }

      void emit_i8x16_min_s() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPMINSB);
      }

      void emit_i8x16_min_u() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPMINUB);
      }

      void emit_i8x16_max_s() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPMAXSB);
      }

      void emit_i8x16_max_u() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPMAXUB);
      }

      void emit_i8x16_avgr_u() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPAVGB);
      }

      // i16x8 ops

      void emit_i16x8_extadd_pairwise_i8x16_s() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit(VPSRLDQ_c, imm8{8}, xmm0, xmm1);
         emit(VPMOVSXBW, xmm0, xmm0);
         emit(VPMOVSXBW, xmm1, xmm1);
         emit(VPHADDW, xmm1, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i16x8_extadd_pairwise_i8x16_u() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit(VPSRLDQ_c, imm8{8}, xmm0, xmm1);
         emit(VPMOVZXBW, xmm0, xmm0);
         emit(VPMOVZXBW, xmm1, xmm1);
         emit(VPHADDW, xmm1, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i16x8_abs() {
         COUNT_INSTR();
         emit_v128_unop(VPABSW);
      }

      void emit_i16x8_neg() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_const_zero(xmm0);
         emit(VPSUBW, *rsp, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i16x8_q15mulr_sat_s() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         emit(VPMULHRSW, *rsp, xmm0, xmm0);
         emit_const_ones(xmm1);
         emit(VPSLLW_c, imm8{15}, xmm1, xmm1);
         emit(VPCMPEQW, xmm1, xmm0, xmm1);
         emit(VPXOR, xmm1, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i16x8_all_true() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_const_zero(xmm0);
         emit(VPCMPEQW, *rsp, xmm0, xmm0);
         emit_add(16, rsp);
         emit_xor(eax, eax);
         emit(VPTEST, xmm0, xmm0);
         emit(SETZ, al);
         emit_push(rax);
      }

      void emit_i16x8_bitmask() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_const_zero(xmm0);
         emit(VPCMPGTW, *rsp, xmm0, xmm1);
         emit(VPACKSSWB, xmm0, xmm1, xmm0);
         emit_add(16, rsp);
         emit(VPMOVMSKB, xmm0, rax);
         emit_push(rax);
      }

      void emit_i16x8_narrow_i32x4_s() {
         COUNT_INSTR();
         emit_v128_binop(VPACKSSDW);
      }

      void emit_i16x8_narrow_i32x4_u() {
         COUNT_INSTR();
         emit_v128_binop(VPACKUSDW);
      }

      void emit_i16x8_extend_low_i8x16_s() {
         COUNT_INSTR();
         emit_v128_unop(VPMOVSXBW);
      }

      void emit_i16x8_extend_high_i8x16_s() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit(VPSRLDQ_c, imm8{8}, xmm0, xmm0);
         emit(VPMOVSXBW, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i16x8_extend_low_i8x16_u() {
         COUNT_INSTR();
         emit_v128_unop(VPMOVZXBW);
      }

      void emit_i16x8_extend_high_i8x16_u() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit(VPSRLDQ_c, imm8{8}, xmm0, xmm0);
         emit(VPMOVZXBW, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i16x8_shl() {
         COUNT_INSTR();
         emit_v128_shiftop(VPSLLW, 0x0f);
      }

      void emit_i16x8_shr_s() {
         COUNT_INSTR();
         emit_v128_shiftop(VPSRAW, 0x0f);
      }

      void emit_i16x8_shr_u() {
         COUNT_INSTR();
         emit_v128_shiftop(VPSRLW, 0x0f);
      }

      void emit_i16x8_add() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPADDW);
      }

      void emit_i16x8_add_sat_s() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPADDSW);
      }

      void emit_i16x8_add_sat_u() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPADDUSW);
      }

      void emit_i16x8_sub() {
         COUNT_INSTR();
         emit_v128_binop(VPSUBW);
      }

      void emit_i16x8_sub_sat_s() {
         COUNT_INSTR();
         emit_v128_binop(VPSUBSW);
      }

      void emit_i16x8_sub_sat_u() {
         COUNT_INSTR();
         emit_v128_binop(VPSUBUSW);
      }

      void emit_i16x8_mul() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPMULLW);
      }

      void emit_i16x8_min_s() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPMINSW);
      }

      void emit_i16x8_min_u() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPMINUW);
      }

      void emit_i16x8_max_s() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPMAXSW);
      }

      void emit_i16x8_max_u() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPMAXUW);
      }

      void emit_i16x8_avgr_u() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPAVGW);
      }

      void emit_i16x8_extmul_low_i8x16_s() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit(VPMOVSXBW, *rsp, xmm1);
         emit_add(16, rsp);
         emit(VPMOVSXBW, *rsp, xmm0);
         emit(VPMULLW, xmm0, xmm1, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i16x8_extmul_high_i8x16_s() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm1);
         emit(VPSRLDQ_c, imm8{8}, xmm1, xmm1);
         emit(VPMOVSXBW, xmm1, xmm1);
         emit_add(16, rsp);
         emit_vmovdqu(*rsp, xmm0);
         emit(VPSRLDQ_c, imm8{8}, xmm0, xmm0);
         emit(VPMOVSXBW, xmm0, xmm0);
         emit(VPMULLW, xmm0, xmm1, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i16x8_extmul_low_i8x16_u() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit(VPMOVZXBW, *rsp, xmm1);
         emit_add(16, rsp);
         emit(VPMOVZXBW, *rsp, xmm0);
         emit(VPMULLW, xmm0, xmm1, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i16x8_extmul_high_i8x16_u() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm1);
         emit(VPSRLDQ_c, imm8{8}, xmm1, xmm1);
         emit(VPMOVZXBW, xmm1, xmm1);
         emit_add(16, rsp);
         emit_vmovdqu(*rsp, xmm0);
         emit(VPSRLDQ_c, imm8{8}, xmm0, xmm0);
         emit(VPMOVZXBW, xmm0, xmm0);
         emit(VPMULLW, xmm0, xmm1, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      // i32x4 ops

      void emit_i32x4_extadd_pairwise_i16x8_s() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit(VPSRLDQ_c, imm8{8}, xmm0, xmm1);
         emit(VPMOVSXWD, xmm0, xmm0);
         emit(VPMOVSXWD, xmm1, xmm1);
         emit(VPHADDD, xmm1, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i32x4_extadd_pairwise_i16x8_u() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit(VPSRLDQ_c, imm8{8}, xmm0, xmm1);
         emit(VPMOVZXWD, xmm0, xmm0);
         emit(VPMOVZXWD, xmm1, xmm1);
         emit(VPHADDD, xmm1, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i32x4_abs() {
         COUNT_INSTR();
         emit_v128_unop(VPABSD);
      }

      void emit_i32x4_neg() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_const_zero(xmm0);
         emit(VPSUBD, *rsp, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i32x4_all_true() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_const_zero(xmm0);
         emit(VPCMPEQD, *rsp, xmm0, xmm0);
         emit_add(16, rsp);
         emit_xor(eax, eax);
         emit(VPTEST, xmm0, xmm0);
         emit(SETZ, al);
         emit_push(rax);
      }

      void emit_i32x4_bitmask() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         emit(VMOVMSKPS, xmm0, rax);
         emit_push(rax);
      }

      void emit_i32x4_extend_low_i16x8_s() {
         COUNT_INSTR();
         emit_v128_unop(VPMOVSXWD);
      }

      void emit_i32x4_extend_high_i16x8_s() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit(VPSRLDQ_c, imm8{8}, xmm0, xmm0);
         emit(VPMOVSXWD, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i32x4_extend_low_i16x8_u() {
         COUNT_INSTR();
         emit_v128_unop(VPMOVZXWD);
      }

      void emit_i32x4_extend_high_i16x8_u() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit(VPSRLDQ_c, imm8{8}, xmm0, xmm0);
         emit(VPMOVZXWD, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i32x4_shl() {
         COUNT_INSTR();
         emit_v128_shiftop(VPSLLD, 0x1f);
      }

      void emit_i32x4_shr_s() {
         COUNT_INSTR();
         emit_v128_shiftop(VPSRAD, 0x1f);
      }

      void emit_i32x4_shr_u() {
         COUNT_INSTR();
         emit_v128_shiftop(VPSRLD, 0x1f);
      }

      void emit_i32x4_add() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPADDD);
      }

      void emit_i32x4_sub() {
         COUNT_INSTR();
         emit_v128_binop(VPSUBD);
      }

      void emit_i32x4_mul() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPMULLD);
      }

      void emit_i32x4_min_s() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPMINSD);
      }

      void emit_i32x4_min_u() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPMINUD);
      }

      void emit_i32x4_max_s() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPMAXSD);
      }

      void emit_i32x4_max_u() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPMAXUD);
      }

      void emit_i32x4_dot_i16x8_s() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPMADDWD);
      }

      void emit_i32x4_extmul_low_i16x8_s() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         emit_vmovdqu(*rsp, xmm1);
         emit(VPMOVSXWD, xmm0, xmm0);
         emit(VPMOVSXWD, xmm1, xmm1);
         emit(VPMULLD, xmm0, xmm1, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i32x4_extmul_high_i16x8_s() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         emit_vmovdqu(*rsp, xmm1);
         emit(VPMULLW, xmm0, xmm1, xmm2);
         emit(VPMULHW, xmm0, xmm1, xmm0);
         emit(VPUNPCKHWD, xmm0, xmm2, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i32x4_extmul_low_i16x8_u() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         emit_vmovdqu(*rsp, xmm1);
         emit(VPMOVZXWD, xmm0, xmm0);
         emit(VPMOVZXWD, xmm1, xmm1);
         emit(VPMULLD, xmm0, xmm1, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i32x4_extmul_high_i16x8_u() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         emit_vmovdqu(*rsp, xmm1);
         emit(VPMULLW, xmm0, xmm1, xmm2);
         emit(VPMULHUW, xmm0, xmm1, xmm0);
         emit(VPUNPCKHWD, xmm0, xmm2, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      // i64x2 ops

      void emit_i64x2_abs() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_const_zero(xmm1);
         emit(VPCMPGTQ, xmm0, xmm1, xmm1);
         emit(VPXOR, xmm0, xmm1, xmm0);
         emit(VPSUBQ, xmm1, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i64x2_neg() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_const_zero(xmm0);
         emit(VPSUBQ, *rsp, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i64x2_all_true() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_const_zero(xmm0);
         emit(VPCMPEQQ, *rsp, xmm0, xmm0);
         emit_add(16, rsp);
         emit_xor(eax, eax);
         emit(VPTEST, xmm0, xmm0);
         emit(SETZ, al);
         emit_push(rax);
      }

      void emit_i64x2_bitmask() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         emit(VMOVMSKPD, xmm0, rax);
         emit_push(rax);
      }

      void emit_i64x2_extend_low_i32x4_s() {
         COUNT_INSTR();
         emit_v128_unop(VPMOVSXDQ);
      }

      void emit_i64x2_extend_high_i32x4_s() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit(VPSRLDQ_c, imm8{8}, xmm0, xmm0);
         emit(VPMOVSXDQ, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i64x2_extend_low_i32x4_u() {
         COUNT_INSTR();
         emit_v128_unop(VPMOVZXDQ);
      }

      void emit_i64x2_extend_high_i32x4_u() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit(VPSRLDQ_c, imm8{8}, xmm0, xmm0);
         emit(VPMOVZXDQ, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i64x2_shl() {
         COUNT_INSTR();
         emit_v128_shiftop(VPSLLQ, 0x3f);
      }

      void emit_i64x2_shr_s() {
         COUNT_INSTR();
         auto icount = simd_instr();
         // (x >> n) | ((0 > x) << (64 - n))
         emit_pop(rax);
         emit_and(0x3f, eax);
         emit_mov(64, ecx);
         emit_sub(eax, ecx);
         emit_vmovdqu(*rsp, xmm0);
         emit_vmovd(eax, xmm1);
         emit_vmovd(ecx, xmm3);
         emit_const_zero(xmm2);
         emit(VPCMPGTQ, xmm0, xmm2, xmm2);
         emit(VPSLLQ, xmm3, xmm2, xmm2);
         emit(VPSRLQ, xmm1, xmm0, xmm0);
         emit(VPOR, xmm0, xmm2, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i64x2_shr_u() {
         COUNT_INSTR();
         emit_v128_shiftop(VPSRLQ, 0x3f);
      }

      void emit_i64x2_add() {
         COUNT_INSTR();
         emit_v128_binop_commutative(VPADDQ);
      }

      void emit_i64x2_sub() {
         COUNT_INSTR();
         emit_v128_binop(VPSUBQ);
      }

      void emit_i64x2_mul() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         emit_vmovdqu(*rsp, xmm1);
         emit(VPMULUDQ, xmm0, xmm1, xmm2);
         emit(VPSHUFD, imm8{0xb1}, xmm0, xmm0);
         emit(VPMULLD, xmm0, xmm1, xmm0);
         emit(VPHADDD, xmm0, xmm0, xmm0);
         emit_const_zero(xmm1);
         emit(VPUNPCKLDQ, xmm0, xmm1, xmm0);
         emit(VPADDQ, xmm0, xmm2, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i64x2_extmul_low_i32x4_s() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit(VPSHUFD, imm8{0x10}, *rsp, xmm0);
         emit_add(16, rsp);
         emit(VPSHUFD, imm8{0x10}, *rsp, xmm1);
         emit(VPMULDQ, xmm0, xmm1, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i64x2_extmul_high_i32x4_s() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit(VPSHUFD, imm8{0x32}, *rsp, xmm0);
         emit_add(16, rsp);
         emit(VPSHUFD, imm8{0x32}, *rsp, xmm1);
         emit(VPMULDQ, xmm0, xmm1, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i64x2_extmul_low_i32x4_u() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit(VPSHUFD, imm8{0x10}, *rsp, xmm0);
         emit_add(16, rsp);
         emit(VPSHUFD, imm8{0x10}, *rsp, xmm1);
         emit(VPMULUDQ, xmm0, xmm1, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      void emit_i64x2_extmul_high_i32x4_u() {
         COUNT_INSTR();
         auto icount = simd_instr();
         emit(VPSHUFD, imm8{0x32}, *rsp, xmm0);
         emit_add(16, rsp);
         emit(VPSHUFD, imm8{0x32}, *rsp, xmm1);
         emit(VPMULUDQ, xmm0, xmm1, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      // ------------- f32x4 arithmetic -------------------

      void emit_f32x4_ceil() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f32x4_ceil);
      }

      void emit_f32x4_floor() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f32x4_floor);
      }

      void emit_f32x4_trunc() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f32x4_trunc);
      }

      void emit_f32x4_nearest() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f32x4_nearest);
      }

      void emit_f32x4_abs() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f32x4_abs);
      }

      void emit_f32x4_neg() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f32x4_neg);
      }

      void emit_f32x4_sqrt() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f32x4_sqrt);
      }

      void emit_f32x4_add() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f32x4_add);
      }

      void emit_f32x4_sub() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f32x4_sub);
      }

      void emit_f32x4_mul() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f32x4_mul);
      }

      void emit_f32x4_div() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f32x4_div);
      }

      void emit_f32x4_min() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f32x4_min);
      }

      void emit_f32x4_max() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f32x4_max);
      }

      void emit_f32x4_pmin() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f32x4_pmin);
      }

      void emit_f32x4_pmax() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f32x4_pmax);
      }

      // ------------- f64x2 arithmetic -------------------

      void emit_f64x2_ceil() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f64x2_ceil);
      }

      void emit_f64x2_floor() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f64x2_floor);
      }

      void emit_f64x2_trunc() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f64x2_trunc);
      }

      void emit_f64x2_nearest() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f64x2_nearest);
      }

      void emit_f64x2_abs() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f64x2_abs);
      }

      void emit_f64x2_neg() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f64x2_neg);
      }

      void emit_f64x2_sqrt() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f64x2_sqrt);
      }

      void emit_f64x2_add() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f64x2_add);
      }

      void emit_f64x2_sub() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f64x2_sub);
      }

      void emit_f64x2_mul() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f64x2_mul);
      }

      void emit_f64x2_div() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f64x2_div);
      }

      void emit_f64x2_min() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f64x2_min);
      }

      void emit_f64x2_max() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f64x2_max);
      }

      void emit_f64x2_pmin() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f64x2_pmin);
      }

      void emit_f64x2_pmax() {
         COUNT_INSTR();
         emit_v128_binop_softfloat(&_eosio_f64x2_pmax);
      }

      // --------------- SIMD conversions ----------------

      void emit_i32x4_trunc_sat_f32x4_s() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_i32x4_trunc_sat_f32x4_s);
      }

      void emit_i32x4_trunc_sat_f32x4_u() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_i32x4_trunc_sat_f32x4_u);
      }

      void emit_f32x4_convert_i32x4_s() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f32x4_convert_i32x4_s);
      }

      void emit_f32x4_convert_i32x4_u() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f32x4_convert_i32x4_u);
      }

      void emit_i32x4_trunc_sat_f64x2_s_zero() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_i32x4_trunc_sat_f64x2_s_zero);
      }

      void emit_i32x4_trunc_sat_f64x2_u_zero() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_i32x4_trunc_sat_f64x2_u_zero);
      }

      void emit_f64x2_convert_low_i32x4_s() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f64x2_convert_low_i32x4_s);
      }

      void emit_f64x2_convert_low_i32x4_u() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f64x2_convert_low_i32x4_u);
      }

      void emit_f32x4_demote_f64x2_zero() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f32x4_demote_f64x2_zero);
      }

      void emit_f64x2_promote_low_f32x4() {
         COUNT_INSTR();
         emit_v128_unop_softfloat(&_eosio_f64x2_promote_low_f32x4);
      }

      void emit_error() { unimplemented(); }

      // bulk memory

      void emit_memory_fill() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(18);
         emit_pop(rcx);
         emit_mov(rdi, r8);
         emit_pop(rax);
         emit_pop(rdi);
         emit_add(rsi, rdi);

         // validate memory
         emit_mov(*(rcx + rdi - 1), dl);

         // rep stosb
         emit_bytes(0xf3);
         emit(IA32(0xaa));

         emit_mov(r8, rdi);
      }
      void emit_memory_copy() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(60);
         emit_pop(rcx);
         emit_mov(rsi, r8);
         emit_mov(rdi, r9);
         emit_pop(rsi);
         emit_pop(rdi);
         emit_add(r8, rsi);
         emit_add(r8, rdi);

         // TODO: reorder branches to help branch prediction

         // %rax = dest - src
         emit_mov(rdi, rax);
         emit_sub(rsi, rax);
         auto cmp1 = emit_branch8(JBE);

         // Validate dest
         emit_mov(*(rcx + rdi - 1), dl);

         emit(CMP, rcx, rax);
         auto cmp2 = emit_branch8(JAE);

         emit(LEA, *(rcx + rsi - 1), rsi);
         emit(LEA, *(rcx + rdi - 1), rdi);
         emit(STD);
         emit_bytes(0xf3);
         emit(IA32(0xa4));
         emit(CLD);

         emit(JMP_8);
         auto end = emit_branch_target8();
         fix_branch8(cmp1, code);
         // Validate src
         emit_mov(*(rcx + rsi - 1), dl);
         fix_branch8(cmp2, code);

         // rep movsb
         emit_bytes(0xf3);
         emit(IA32(0xa4));

         fix_branch8(end, code);

         emit_mov(r8, rsi);
         emit_mov(r9, rdi);
      }

      void emit_table_copy() {
         COUNT_INSTR();
         auto icount = fixed_size_instr(97);
         emit_pop(rcx);
         emit_mov(rsi, r8);
         emit_mov(rdi, r9);
         emit_pop(rsi);
         emit_pop(rdi);

         // Check bounds
         emit_mov(_mod.tables[0].limits.initial, eax);
         emit_sub(rcx, rax);
         emit_cmp(rax, rsi);
         fix_branch(emit_branchcc32(JG), memory_handler);
         emit_cmp(rax, rdi);
         fix_branch(emit_branchcc32(JG), memory_handler);

         // convert indices to pointers and size to bytes
         static_assert(sizeof(table_entry) == 16);
         if (_mod.indirect_table(0)) {
            emit_mov(*(r8 + wasm_allocator::table_offset()), rax);
         } else {
            emit(LEA, *(r8 + wasm_allocator::table_offset()), rax);
         }
         emit(SHL_imm8, imm8{4}, rsi);
         emit(SHL_imm8, imm8{4}, rdi);
         emit(SHL_imm8, imm8{4}, rcx);
         emit_add(rax, rsi);
         emit_add(rax, rdi);

         // check for overlap
         emit_mov(rdi, rax);
         emit_sub(rsi, rax);
         auto cmp1 = emit_branch8(JBE);
         emit(CMP, rcx, rax);
         auto cmp2 = emit_branch8(JAE);

         // reverse copy
         emit(LEA, *(rcx + rsi - 1), rsi);
         emit(LEA, *(rcx + rdi - 1), rdi);
         emit(STD);
         emit_bytes(0xf3);
         emit(IA32(0xa4));
         emit(CLD);

         emit(JMP_8);
         auto end = emit_branch_target8();
         fix_branch8(cmp1, code);
         fix_branch8(cmp2, code);

         // forward copy
         // rep movsb
         emit_bytes(0xf3);
         emit(IA32(0xa4));

         fix_branch8(end, code);

         emit_mov(r8, rsi);
         emit_mov(r9, rdi);
      }

      // --------------- random  ------------------------
      static void fix_branch(void* branch, void* target) {
         auto branch_ = static_cast<uint8_t*>(branch);
         auto target_ = static_cast<uint8_t*>(target);
         auto relative = static_cast<uint32_t>(target_ - (branch_ + 4));
         if((target_ - (branch_ + 4)) > 0x7FFFFFFFll ||
            (target_ - (branch_ + 4)) < -0x80000000ll) unimplemented();
         memcpy(branch, &relative, 4);
      }

      static void fix_branch8(void* branch, void* target) {
         auto branch_ = static_cast<uint8_t*>(branch);
         auto target_ = static_cast<uint8_t*>(target);
         auto relative = static_cast<uint8_t>(target_ - (branch_ + 1));
         if((target_ - (branch_ + 1)) > 0x7Fll ||
            (target_ - (branch_ + 1)) < -0x80ll) unimplemented();
         memcpy(branch, &relative, 1);
      }

      // A 64-bit absolute address is used for function calls whose
      // address is too far away for a 32-bit relative call.
      static void fix_branch64(void* branch, void* target) {
         memcpy(branch, &target, 8);
      }

      using fn_type = native_value(*)(void* context, void* memory);
      void finalize(function_body& body) {
         _allocator.reclaim(code, _code_end - code);
         body.jit_code_offset = _code_start - (unsigned char*)_code_segment_base;
      }

      // returns the current write address
      const void* get_addr() const {
         if (recent_ops[1].end == code) {
            if (recent_ops[0].end == recent_ops[1].start) {
               return recent_ops[0].start;
            } else {
               return recent_ops[1].start;
            }
         }
         return code;
      }

      const void* get_base_addr() const { return _code_segment_base; }

      void set_stack_usage(std::uint64_t usage)
      {
         if constexpr (StackLimitIsBytes)
         {
            // Add the base frame overhead
            usage += 16;
            if (usage >= 0x7fffffff)
            {
               unimplemented();
            }
            // overwrite prologue
            stack_usage = usage;
            static_assert(sizeof(stack_usage) == 4);
            std::memcpy(stack_limit_entry, &stack_usage, 4);
         }
      }

    private:

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
          al, cl, dl, bl, /*ah, ch, dh, bh,*//*spl, bpl, sil, dil*/
          r8b = 8, r9b, r10b, r11b, r12b, r13b, r14b, r15b
          // ah, ch, dh, and bh cannot be encoded with a REX prefix
          // spl, bpl, sil, and dil require a REX prefix.
          // We're not using any of these registers so disable access
          // movzx %ah, %r8d ;; Not encodable
      };
      struct general_register {
         constexpr general_register(general_register32 reg) : reg(reg) {}
         constexpr general_register(general_register64 reg) : reg(reg) {}
         constexpr general_register(general_register16 reg) : reg(reg) {}
         constexpr general_register(general_register8 reg) : reg(reg) {}
         int reg;
      };
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
      struct sib_memory_ref
      {
         general_register64 reg1;
         general_register64 reg2;
         int32_t offset;
      };
      friend sib_memory_ref operator*(double_register_expr expr) { return { expr.reg1, expr.reg2, expr.offset }; }
      template <typename T, int Sz>
      struct sized_memory_ref {
         T value;
      };
      template <typename T>
      static sized_memory_ref<T, 4> dword_ptr(const T& expr)
      {
         return { expr };
      }
      template <typename T>
      static sized_memory_ref<T, 8> qword_ptr(const T& expr)
      {
         return { expr };
      }
      template <typename T, int Sz>
      friend auto operator*(const sized_memory_ref<T, Sz>& expr)
      {
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

      void emit_add(general_register32 src, general_register32 dest) {
         emit(ADD_A, src, dest);
      }

      void emit_add(general_register64 src, general_register64 dest) {
         emit(ADD_A, src, dest);
      }

      template<typename RM>
      void emit_sub(int32_t immediate, RM dest) {
         if(immediate <= 0x7F && immediate >= -0x80) {
            emit(SUB_imm8, static_cast<imm8>(immediate), dest);
         } else {
            emit(SUB_imm32, static_cast<imm32>(immediate), dest);
         }
      }

      void emit_sub(general_register32 src, general_register32 dest) {
         emit(SUB_A, src, dest);
      }

      void emit_sub(general_register64 src, general_register64 dest) {
         emit(SUB_A, src, dest);
      }

      void emit_cmp(general_register32 src, general_register32 dest) {
         emit(CMP, src, dest);
      }

      void emit_cmp(general_register64 src, general_register64 dest) {
         emit(CMP, src, dest);
      }

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

      void emit_call(general_register64 reg) {
         emit(IA32(0xff)/2, reg);
      }

      void emit_enter(uint16_t offset) {
         emit_bytes(0xc8);
         emit_operand16(offset);
         emit_bytes(0x00);
      }

      void emit_ldmxcsr(disp_memory_ref mem) {
         emit_REX_prefix(false, mem, 0);
         emit_bytes(0x0f, 0xae);
         emit_modrm_sib_disp(mem, 2);
      }

      void emit_stmxcsr(disp_memory_ref mem) {
         emit_REX_prefix(false, mem, 0);
         emit_bytes(0x0f, 0xae);
         emit_modrm_sib_disp(mem, 3);
      }

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

      void emit_mov(general_register32 src, disp_memory_ref dest)
      {
         emit(IA32(0x89), dest, src);
      }

      void emit_mov(general_register32 src, general_register32 dest) {
         emit(MOV_A, src, dest);
      }

      void emit_mov(disp_memory_ref mem, general_register32 reg) {
         emit(MOV_A, mem, reg);
      }

      void emit_mov(general_register64 src, general_register64 dest) {
         emit(MOV_A, src, dest);
      }

      void emit_mov(disp_memory_ref mem, general_register64 reg) {
         emit(MOV_A, mem, reg);
      }

      void emit_mov(general_register64 reg, disp_memory_ref mem) {
         emit(MOV_B, mem, reg);
      }

      void emit_mov(sib_memory_ref mem, general_register8 reg) {
         emit(MOVB_A, mem, reg);
      }

      void emit_pop(general_register64 reg) {
#ifndef EOS_VM_VALIDATE_JIT_SIZE
         if (auto c = try_pop_recent_op<i32_const_op>()) {
            emit_mov(c->value, static_cast<general_register32>(reg));
            return;
         }
         if (auto c = try_pop_recent_op<i64_const_op>()) {
            emit_mov(c->value, reg);
            return;
         }
         if (auto push = try_undo_push()) {
            if (push->reg != reg) {
               emit_mov(push->reg, reg);
            }
            return;
         }
#endif
         emit_REX_prefix(false, false, false, reg & 8);
         emit_bytes(0x58 | (reg & 7));
      }

      void emit_push(general_register64 reg) {
         auto start = code;
         emit_REX_prefix(false, false, false, reg & 8);
         emit_bytes(0x50 | (reg & 7));
         push_recent_op(start, generic_register_push_op{reg});
      }

      void emit_push_v128(xmm_register reg) {
         auto start = code;
         emit_sub(16, rsp);
         emit_vmovdqu(reg, *rsp);
         push_recent_op(start, xmm_register_push_op{reg});
      }

      void emit_pop_v128(xmm_register reg) {
         if (auto push = try_pop_recent_op<xmm_register_push_op>()) {
            if (push->reg != reg) {
               emit(VMOVDQU_A, push->reg, reg);
            }
            return;
         }
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
      }

      void set_branch_target() {
         recent_ops[0] = {};
         recent_ops[1] = {};
      }

      void emit_xor(general_register32 src, general_register32 dest) {
         emit(XOR_A, src, dest);
      }

      void emit_xor(int32_t immediate, general_register32 dest) {
         if(immediate <= 0x7F && immediate >= -0x80) {
            emit(XOR_imm8, static_cast<imm8>(immediate), dest);
         } else {
            emit(XOR_imm32, static_cast<imm32>(immediate), dest);
         }
      }

      void emit_vmovdqu(disp_memory_ref mem, xmm_register reg) {
         emit(VMOVDQU_A, mem, reg);
      }

      void emit_vmovdqu(xmm_register reg, disp_memory_ref mem) {
         emit(VMOVDQU_B, mem, reg);
      }

      void emit_vmovups(disp_memory_ref mem, xmm_register reg) {
         emit(VMOVUPS_A, mem, reg);
      }
      
      void emit_vmovups(simple_memory_ref mem, xmm_register reg) {
         emit(VMOVUPS_A, mem, reg);
      }

      void emit_vmovups(xmm_register reg, disp_memory_ref mem) {
         emit(VMOVUPS_B, mem, reg);
      }

      void emit_vmovups(xmm_register reg, simple_memory_ref mem) {
         emit(VMOVUPS_B, mem, reg);
      }

      void emit_vmovd(general_register32 src, xmm_register dest) {
         emit(VMOVD_A, src, dest);
      }

      void emit_vmovq(general_register64 src, xmm_register dest) {
         emit(VMOVQ_A, src, dest);
      }

      template<typename T>
      void emit_vpextrb(uint8_t offset, xmm_register src, T dest) {
         emit(VPEXTRB, imm8{offset}, dest, src);
      }

      template<typename T>
      void emit_vpextrw(uint8_t offset, xmm_register src, T dest) {
         emit(VPEXTRW, imm8{offset}, dest, src);
      }

      template<typename T>
      void emit_vpextrd(uint8_t offset, xmm_register src, T dest) {
         emit(VPEXTRD, imm8{offset}, dest, src);
      }

      template<typename T>
      void emit_vpextrq(uint8_t offset, xmm_register src, T dest) {
         emit(VPEXTRQ, imm8{offset}, dest, src);
      }

      void emit_const_zero(xmm_register reg) {
         emit(VPXOR, reg, reg, reg);
      }

      void emit_const_ones(xmm_register reg) {
         emit(VPCMPEQB, reg, reg, reg);
      }

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
      // When adding jcc codes, verify that the rel8/rel32 versions are 7x and 0F 8x
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
      static constexpr auto ROR_cl = IA32_WX(0xd3)/1;
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
      //static constexpr auto XOR_B = IA32_WX_STORE(0x31);
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
         default: unimplemented();
         }
         __builtin_unreachable();
      }

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

      void emit_REX_prefix(bool W, sib_memory_ref mem, int reg)
      {
         emit_REX_prefix(W, reg & 8, mem.reg1 & 8, mem.reg2 & 8);
      }

      template <typename T, int Sz>
      void emit_REX_prefix(bool W, sized_memory_ref<T, Sz> mem, int reg)
      {
         emit_REX_prefix(W, mem.value, reg);
      }

      void emit_VEX_prefix(bool R, bool X, bool B, VEX_mmmm mmmm, bool W, int vvvv, bool L, VEX_pp pp) {
         if(X || B || (mmmm != mmmm_0F) || W) {
            emit_bytes(0xc4, (!R << 7) | (!X << 6) | (!B << 5) | mmmm, (W << 7) | ((vvvv ^ 0xF) << 3) | (L << 2) | pp);
         } else {
            emit_bytes(0xc5, (!R << 7) | ((vvvv ^ 0xF) << 3) | (L << 2) | pp);
         }
      }

      void emit_modrm_sib_disp(sib_memory_ref mem, int reg) {
         if (mem.reg1 == rsp || mem.reg2 == rbp) unimplemented();
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

      template<int W, int N>
      void emit(IA32_t<W, N> opcode) {
         static_assert(W == 0 || W == 1);
         emit_REX_prefix(W, false, false, false);
         for(int i = 0; i < N; ++i) {
            emit_bytes(opcode.opcode[i]);
         }
      }
      template<int W, int N, typename RM>
      void emit(IA32_t<W, N> opcode, RM r_m) {
         emit(opcode, r_m, 0);
      }
      template<int W, int N, typename RM, typename Reg>
      void emit(IA32_t<W, N> opcode, RM r_m, Reg reg) {
         emit_REX_prefix(get_operand_size<W>(r_m, reg), r_m, reg);
         for(int i = 0; i < N; ++i) {
            emit_bytes(opcode.opcode[i]);
         }
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

      auto fixed_size_instr(std::size_t expected_bytes) {
         return scope_guard{[this, expected_code=code+expected_bytes](){
#ifdef EOS_VM_VALIDATE_JIT_SIZE
            assert(code == expected_code);
#endif
            ignore_unused_variable_warning(code, expected_code);
         }};
      }
      auto variable_size_instr(std::size_t min, std::size_t max) {
         return scope_guard{[this, min_code=code+min,max_code=code+max](){
#ifdef EOS_VM_VALIDATE_JIT_SIZE
            assert(min_code <= code && code <= max_code);
#endif
            ignore_unused_variable_warning(code, min_code, max_code);
         }};
      }
      auto softfloat_instr(std::size_t hard_expected, std::size_t soft_expected, std::size_t softbt_expected) {
         return fixed_size_instr(use_softfloat?(Context::async_backtrace()?softbt_expected:soft_expected):hard_expected);
      }
      auto simd_instr() {
         // SIMD instructions use at least 2-bytes, so 64 gives an
         // amplification factor of 32, which is safely below other limits.
         return variable_size_instr(0, 64);
      }

      struct function_parameters {
         function_parameters() = default;
         function_parameters(const func_type* ft) {
            uint32_t current_offset = 16;
            offsets.resize(ft->param_types.size());
            for(uint32_t i = 0; i < ft->param_types.size(); ++i) {
               if(current_offset > 0x7fffffffu) {
                  unimplemented(); // cannot represent the offset as a 32-bit immediate
               }
               offsets[offsets.size() - i - 1] = current_offset;
               if(ft->param_types[ft->param_types.size() - i - 1] == types::v128) {
                  current_offset += 16;
               } else {
                  current_offset += 8;
               }
            }
         }
         int32_t get_frame_offset(uint32_t localidx) const {
            return offsets[localidx];
         }
         std::vector<uint32_t> offsets;
      };

      struct function_locals {
         function_locals() = default;
         function_locals(const std::vector<local_entry>& params) {
            uint32_t offset = 0;
            int32_t frame_offset = 0;
            for(uint32_t i = 0; i < params.size(); ++i) {
               uint8_t size = params[i].type == types::v128? 16 : 8;
               offset += params[i].count;
               if(-0x80000000ll + static_cast<int64_t>(size) * static_cast<int64_t>(params[i].count) > frame_offset) {
                  unimplemented();
               }
               frame_offset -= size * params[i].count;
               groups.push_back({offset, frame_offset, size});
            }
         }
         struct entry {
            uint32_t end;
            int32_t end_offset;
            uint8_t elem_size;
         };
         int32_t get_frame_offset(uint32_t paramidx) const {
            auto pos = std::partition_point(groups.begin(), groups.end(), [=](const auto& e){
               return paramidx >= e.end;
            });
            assert(pos != groups.end());
            return (pos->end_offset + (pos->end - paramidx - 1) * pos->elem_size);
         }
         std::vector<entry> groups;
      };

      int32_t get_frame_offset(uint32_t local_idx) const {
         if (local_idx < _ft->param_types.size()) {
            return _params.get_frame_offset(local_idx);
         } else {
            return _locals.get_frame_offset(local_idx - _ft->param_types.size());
         }
      }

      module& _mod;
      growable_allocator& _allocator;
      void * _code_segment_base;
      const func_type* _ft;
      function_parameters _params;
      function_locals _locals;
      unsigned char * _code_start;
      unsigned char * _code_end;
      unsigned char * code;
      std::vector<std::variant<std::vector<void*>, void*>> _function_relocations;
      void* fpe_handler;
      void* call_indirect_handler;
      void* type_error_handler;
      void* stack_overflow_handler;
      void* memory_handler;
      uint64_t _local_count;
      uint32_t _table_element_size;

      std::uint32_t stack_usage;
      void* stack_limit_entry = nullptr;

      struct get_local_op {
         disp_memory_ref expr;
      };

      struct i32_const_op {
         uint32_t value;
      };

      struct i64_const_op {
         uint64_t value;
      };

      struct generic_register_push_op {
         general_register64 reg;
      };

      struct xmm_register_push_op {
         xmm_register reg;
      };

      struct condition_op {
         Jcc branchop;
      };

      struct recent_op_t {
         unsigned char* start;
         unsigned char* end;
         std::variant<std::monostate, generic_register_push_op, get_local_op, i32_const_op, i64_const_op, condition_op, xmm_register_push_op> data;
      };

      // This is used to fold common instruction sequences
      recent_op_t recent_ops[2];

      void push_recent_op(unsigned char* start, auto op) {
         // If we're not changing end, it means we're overwriting the last instruction
         // (replacing a push with a more specific op).
         if (recent_ops[1].end != code) {
            recent_ops[0] = recent_ops[1];
         }
         recent_ops[1] = {start, code, op};
      }

      template<typename T>
      std::optional<T> try_pop_recent_op() {
         if (recent_ops[1].end == code) {
            if (auto p = std::get_if<T>(&recent_ops[1].data)) {
               std::optional<T> result = *p;
               code = recent_ops[1].start;
               recent_ops[1] = recent_ops[0];
               recent_ops[0] = {};
               return result;
            }
         }
         return {};
      }

      std::optional<generic_register_push_op> try_undo_push() {
         if (recent_ops[1].end == code) {
            if (auto res = try_pop_recent_op<generic_register_push_op>()) {
               return res;
            }
            if (std::holds_alternative<get_local_op>(recent_ops[1].data) ||
                std::holds_alternative<i32_const_op>(recent_ops[1].data) ||
                std::holds_alternative<i64_const_op>(recent_ops[1].data)) {
               --code;
               recent_ops[1] = {};
               recent_ops[0] = {};
               return generic_register_push_op{rax};
            }
         }
         return {};
      }

      template<typename T, typename U>
      std::optional<std::pair<T, U>> try_pop_recent_op() {
         if (recent_ops[1].end == code && recent_ops[0].end == recent_ops[1].start) {
            if (auto p1 = std::get_if<T>(&recent_ops[0].data)) {
               if (auto p2 = std::get_if<U>(&recent_ops[1].data)) {
                  std::optional<std::pair<T, U>> result(*p1, *p2);
                  code = recent_ops[0].start;
                  recent_ops[0] = {};
                  recent_ops[1] = {};
                  return result;
               }
            }
         }
      }

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

      // check_call_depth and check_stack_limit are incompatible
      void emit_check_call_depth() {
         if constexpr (!StackLimitIsBytes)
         {
            auto icount = fixed_size_instr(8);
            emit(DECD, ebx);
            fix_branch(emit_branchcc32(JZ), stack_overflow_handler);
         }
      }
      void emit_check_call_depth_end() {
         if constexpr (!StackLimitIsBytes)
         {
            auto icount = fixed_size_instr(2);
            emit(INCD, ebx);
         }
      }

      void emit_check_stack_limit()
      {
         if constexpr (StackLimitIsBytes)
         {
            // TODO: compress this based on max_func_local_bytes
            emit(IA32(0x81)/5, ebx);
            stack_limit_entry = code;
            emit_bytes(0, 0, 0, 0);
            // save location
            fix_branch(emit_branchcc32(JB), stack_overflow_handler);
         }
      }

      void emit_check_stack_limit_end()
      {
         if constexpr (StackLimitIsBytes)
         {
            emit_add(static_cast<std::uint32_t>(stack_usage), ebx);
         }
      }

      static void unimplemented() { EOS_VM_ASSERT(false, wasm_parse_exception, "Sorry, not implemented."); }

      static constexpr bool is_simple_multipop(uint32_t count, uint8_t rt) {
         switch(rt) {
         case types::pseudo:
            return count == 0;
         case types::i32: case types::i64: case types::f32: case types::f64:
            return count == 1;
         case types::v128:
            return count == 2;
         default:
            return false;
         }
      }

      // clobbers %rax or %xmm0 if rt is not void
      // count includes the return value
      void emit_multipop(uint32_t count, uint8_t rt) {
         auto icount = variable_size_instr(0, 17);
         if(!is_simple_multipop(count, rt)) {
            if (rt == types::v128) {
               emit_vmovups(*rsp, xmm0);
               // Leave room to write the result at the bottom of the stack
               count -= 2;
            } else if (rt != types::pseudo) {
               emit_mov(*rsp, rax);
            }
            if(count & 0x70000000) {
               // This code is probably unreachable.
               // int3
               emit_bytes(0xCC);
               return;
            }
            emit_add(count * 8, rsp);
            if (rt == types::v128) {
               emit_vmovups(xmm0, *rsp);
            } else if (rt != types::pseudo) {
               emit_push(rax);
            }
         }
      }

      // \pre the result (if any) is in %rax or %xmm0
      // \post the result (if any) is at the top of the stack
      void emit_multipop(const func_type& ft) {
         auto icount = variable_size_instr(0, 12);
         uint32_t total_size = 0;
         for(uint32_t i = 0; i < ft.param_types.size(); ++i) {
            if(ft.param_types[i] == v128) {
               total_size += 16;
            } else {
               total_size += 8;
            }
            if(total_size > 0x7fffffffu) {
               unimplemented();
            }
         }
         if (ft.return_count && ft.return_type == types::v128) {
            if (total_size != 16)
            {
               emit_add(total_size - 16, rsp);
            }
            emit_vmovups(xmm0, *rsp);
         } else {
            if(total_size != 0) {
               emit_add(total_size, rsp);
            }
            if (ft.return_count != 0) {
               emit_push(rax);
            }
         }
      }

      sib_memory_ref emit_pop_address(uint32_t offset, general_register64 out, general_register32 tmpreg) {
         emit_pop(out);
         if (offset & 0x80000000u) {
            emit_mov(offset, tmpreg);
            emit_add(static_cast<general_register64>(tmpreg), out);
            return *(out + rsi + 0);
         } else {
            return *(out + rsi + offset);
         }
      }

      // reg should be rax, eax, ax, or al
      template<class I, class R>
      void emit_load_impl2(uint32_t offset, I instr, R reg) {
         auto addr = emit_pop_address(offset, rax, ecx);
         emit(instr, addr, reg);
         emit_push(rax);
      }

      // rax holds the value to be stored
      template<class I, class R>
      void emit_store_impl2(uint32_t offset, I instr, R reg) {
         emit_pop(rax);
         auto addr = emit_pop_address(offset, rcx, edx);
         emit(instr, addr, reg);
      }

      // pops an i32 wasm address off the stack
      // adds offset and converts the result to
      // a native address.  The result is in %rax.
      void emit_pop_address(uint32_t offset) {
         // pop %rax
         emit_bytes(0x58);
         if (offset & 0x80000000) {
            // mov $offset, %ecx
            emit_bytes(0xb9);
            emit_operand32(offset);
            // add %rcx, %rax
            emit_bytes(0x48, 0x01, 0xc8);
         } else if (offset != 0) {
            // add offset, %rax
            emit_bytes(0x48, 0x05);
            emit_operand32(offset);
         }
         // add %rsi, %rax
         emit_bytes(0x48, 0x01, 0xf0);
      }

      void emit_i32_relop(Jcc opcode) {
         emit_pop(rax);
         emit_pop(rcx);
         emit(XOR_A, edx, edx);
         emit(CMP, eax, ecx);
         auto start = code;
         emit_setcc(opcode, dl);
         emit_push(rdx);
         push_recent_op(start, condition_op{opcode});
      }

      template<class... T>
      void emit_i64_relop(uint8_t opcode) {
         COUNT_INSTR();
         // popq %rax
         emit_bytes(0x58);
         // popq %rcx
         emit_bytes(0x59);
         // xorq %rdx, %rdx
         emit_bytes(0x48, 0x31, 0xd2);
         // cmpq %rax, %rcx
         emit_bytes(0x48, 0x39, 0xc1);
         // SETcc %dl
         emit_bytes(0x0f, opcode, 0xc2);
         // pushq %rdx
         emit_bytes(0x52);
      }

      template<typename Op>
      void emit_v128_loadop(Op op, uint32_t /*align*/, uint32_t offset) {
         auto icount = simd_instr();
         auto expr = emit_pop_address(offset, rax, ecx);
         emit(op, expr, xmm0);
         emit_push_v128(xmm0);
      }

      template<typename Op>
      void emit_v128_load_laneop(Op op, uint32_t /*align*/, uint32_t offset, uint8_t lane) {
         auto icount = simd_instr();
         emit_pop_v128(xmm0);
         emit_pop_address(offset);
         emit(op, imm8{lane}, *rax, xmm0, xmm0);
         emit_sub(16, rsp);
         emit_vmovdqu(xmm0, *rsp);
      }

      template<typename Op>
      void emit_v128_store_laneop(Op op, uint32_t /*align*/, uint32_t offset, uint8_t lane) {
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         emit_pop_address(offset);
         emit(op, imm8{lane}, *rax, xmm0);
      }

      template<typename Op>
      void emit_v128_extract_laneop(Op op, uint8_t l) {
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         emit(op, imm8{l}, rax, xmm0);
         emit_push(rax);
      }

      template<typename Op>
      void emit_v128_replace_laneop(Op op, uint8_t l) {
         auto icount = simd_instr();
         emit_pop(rax);
         emit_vmovdqu(*rsp, xmm0);
         emit(op, imm8{l}, rax, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      template<typename Op>
      void emit_v128_splatop(Op op) {
         emit(op, *rsp, xmm0);
         emit_sub(8, rsp);
         emit_vmovdqu(xmm0, *rsp);
      }

      template<typename Op>
      void emit_v128_irelop_subsat(Op opcode, bool switch_params, bool flip_result) {
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         if(switch_params) {
            emit_vmovdqu(*rsp, xmm1);
            // vpsubusb %xmm0, %xmm1, %xmm0
            emit(opcode, xmm0, xmm1, xmm0);
         } else {
            // vpsubusb (%rsp), %xmm0, %xmm0
            emit(opcode, *rsp, xmm0, xmm0);
         }
         emit_const_zero(xmm1);
         // vpcmpeqb %xmm1, %xmm0, %xmm0
         emit(VPCMPEQB, xmm1, xmm0, xmm0);
         if(!flip_result) {
            // vpcmpeqb %xmm1, %xmm0, %xmm0
            emit(VPCMPEQB, xmm1, xmm0, xmm0);
         }
         emit_vmovdqu(xmm0, *rsp);
      }

      // !(op(a,b) == b)
      // max -> gt
      // min -> lt
      template<typename Op, typename Eq>
      void emit_v128_irelop_minmax(Op opcode, Eq eq, bool flip_result) {
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         // vp[min/max]us[b/w/d/q] (%rsp), xmm0, xmm1
         emit(opcode, *rsp, xmm0, xmm1);
         emit(eq, xmm0, xmm1, xmm0);
         if(!flip_result) {
            emit_const_zero(xmm1);
            emit(VPCMPEQB, xmm1, xmm0, xmm0);
         }
         emit_vmovdqu(xmm0, *rsp);
      }

      // Note: for commutative functions switch_params=true is more efficient
      template<typename Op>
      void emit_v128_irelop_cmp(Op opcode, bool switch_params, bool flip_result) {
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         if(!switch_params) {
            emit_vmovdqu(*rsp, xmm1);
            // OP %xmm0, %xmm1, %xmm0
            emit(opcode, xmm0, xmm1, xmm0);
         } else {
            // OP *rsp, %xmm0, %xmm0
            emit(opcode, *rsp, xmm0, xmm0);
         }
         if(flip_result) {
            emit_const_ones(xmm1);
            emit(VPXOR, xmm1, xmm0, xmm0);
         }
         emit_vmovdqu(xmm0, *rsp);
      }

      template<typename Op>
      void emit_v128_shiftop(Op opcode, uint8_t mask) {
         auto icount = simd_instr();
         emit_pop(rax);
         // and $mask, %eax
         emit_bytes(0x83, 0xe0, mask);
         emit_vmovdqu(*rsp, xmm0);
         emit_vmovd(eax, xmm1);
         emit(opcode, xmm1, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      template<typename Op>
      void emit_v128_unop(Op opcode) {
         auto icount = simd_instr();
         emit(opcode, *rsp, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      template<typename Op>
      void emit_v128_binop(Op opcode) {
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         emit_vmovdqu(*rsp, xmm1);
         emit(opcode, xmm0, xmm1, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      template<typename Op>
      void emit_v128_binop_r(Op opcode) {
         auto icount = simd_instr();
         emit_vmovdqu(*rsp, xmm0);
         emit_add(16, rsp);
         emit(opcode, *rsp, xmm0, xmm0);
         emit_vmovdqu(xmm0, *rsp);
      }

      template<typename Op>
      void emit_v128_binop_commutative(Op opcode) {
         emit_v128_binop_r(opcode);
      }

      template<typename T, typename U>
      void emit_softfloat_unop(T(*softfloatfun)(U)) {
         auto extra = emit_setup_backtrace();
         // pushq %rdi
         emit_bytes(0x57);
         // pushq %rsi
         emit_bytes(0x56);
         if constexpr(sizeof(U) == 4) {
            // movq 16(%rsp), %edi
            emit_bytes(0x8b, 0x7c, 0x24, 0x10 + extra);
         } else {
            // movq 16(%rsp), %rdi
            emit_bytes(0x48, 0x8b, 0x7c, 0x24, 0x10 + extra);
         }
         emit_align_stack();
         // movabsq $softfloatfun, %rax
         emit_bytes(0x48, 0xb8);
         emit_operand_ptr(softfloatfun);
         // callq *%rax
         emit_bytes(0xff, 0xd0);
         emit_restore_stack();
         // popq %rsi
         emit_bytes(0x5e);
         // popq %rdi
         emit_bytes(0x5f);
         emit_restore_backtrace();
         if constexpr(sizeof(T) == 4) {
            static_assert(sizeof(U) == 4, "Can only push 4-byte item if the upper 4 bytes are already 0");
            // movq %eax, (%rsp)
            emit_bytes(0x89, 0x04, 0x24);
         } else {
            // movq %rax, (%rsp)
            emit_bytes(0x48, 0x89, 0x04, 0x24);
         }
      }

      void emit_f32_binop_softfloat(float32_t (*softfloatfun)(float32_t, float32_t)) {
         auto extra = emit_setup_backtrace();
         // pushq %rdi
         emit_bytes(0x57);
         // pushq %rsi
         emit_bytes(0x56);
         // movq 16(%rsp), %esi
         emit_bytes(0x8b, 0x74, 0x24, 0x10 + extra);
         // movq 24(%rsp), %edi
         emit_bytes(0x8b, 0x7c, 0x24, 0x18 + extra);
         emit_align_stack();
         // movabsq $softfloatfun, %rax
         emit_bytes(0x48, 0xb8);
         emit_operand_ptr(softfloatfun);
         // callq *%rax
         emit_bytes(0xff, 0xd0);
         emit_restore_stack();
         // popq %rsi
         emit_bytes(0x5e);
         // popq %rdi
         emit_bytes(0x5f);
         emit_restore_backtrace_basic();
         // addq $8, %rsp
         emit_bytes(0x48, 0x83, 0xc4, 0x08 + extra);
         // movq %eax, (%rsp)
         emit_bytes(0x89, 0x04, 0x24);
      }

      void emit_f64_binop_softfloat(float64_t (*softfloatfun)(float64_t, float64_t)) {
         auto extra = emit_setup_backtrace();
         // pushq %rdi
         emit_bytes(0x57);
         // pushq %rsi
         emit_bytes(0x56);
         // movq 16(%rsp), %rsi
         emit_bytes(0x48, 0x8b, 0x74, 0x24, 0x10 + extra);
         // movq 24(%rsp), %rdi
         emit_bytes(0x48, 0x8b, 0x7c, 0x24, 0x18 + extra);
         emit_align_stack();
         // movabsq $softfloatfun, %rax
         emit_bytes(0x48, 0xb8);
         emit_operand_ptr(softfloatfun);
         // callq *%rax
         emit_bytes(0xff, 0xd0);
         emit_restore_stack();
         // popq %rsi
         emit_bytes(0x5e);
         // popq %rdi
         emit_bytes(0x5f);
         emit_restore_backtrace_basic();
         // addq $8, %rsp
         emit_bytes(0x48, 0x83, 0xc4, 0x08 + extra);
         // movq %rax, (%rsp)
         emit_bytes(0x48, 0x89, 0x04, 0x24);
      }

      void emit_f32_relop(uint8_t opcode, uint64_t (*softfloatfun)(float32_t, float32_t), bool switch_params, bool flip_result) {
         COUNT_INSTR();
         if constexpr (use_softfloat) {
            auto extra = emit_setup_backtrace();
            // pushq %rdi
            emit_bytes(0x57);
            // pushq %rsi
            emit_bytes(0x56);
            if(switch_params) {
               // movq 24(%rsp), %esi
               emit_bytes(0x8b, 0x74, 0x24, 0x18 + extra);
               // movq 16(%rsp), %edi
               emit_bytes(0x8b, 0x7c, 0x24, 0x10 + extra);
            } else {
               // movq 16(%rsp), %esi
               emit_bytes(0x8b, 0x74, 0x24, 0x10 + extra);
               // movq 24(%rsp), %edi
               emit_bytes(0x8b, 0x7c, 0x24, 0x18 + extra);
            }
            emit_align_stack();
            // movabsq $softfloatfun, %rax
            emit_bytes(0x48, 0xb8);
            emit_operand_ptr(softfloatfun);
            // callq *%rax
            emit_bytes(0xff, 0xd0);
            emit_restore_stack();
            // popq %rsi
            emit_bytes(0x5e);
            // popq %rdi
            emit_bytes(0x5f);
            emit_restore_backtrace_basic();
            if (flip_result) {
               // xor $0x1, %al
               emit_bytes(0x34, 0x01);
            }
            // addq $8, %rsp
            emit_bytes(0x48, 0x83, 0xc4, 0x08 + extra);
            // movq %rax, (%rsp)
            emit_bytes(0x48, 0x89, 0x04, 0x24);
         } else {
            // ucomiss+seta/setae is shorter but can't handle eq/ne
            if(switch_params) {
               // movss (%rsp), %xmm0
               emit_bytes(0xf3, 0x0f, 0x10, 0x04, 0x24);
               // cmpCCss 8(%rsp), %xmm0
               emit_bytes(0xf3, 0x0f, 0xc2, 0x44, 0x24, 0x08, opcode);
            } else {
               // movss 8(%rsp), %xmm0
               emit_bytes(0xf3, 0x0f, 0x10, 0x44, 0x24, 0x08);
               // cmpCCss (%rsp), %xmm0
               emit_bytes(0xf3, 0x0f, 0xc2, 0x04, 0x24, opcode);
            }               
            // movd %xmm0, %eax
            emit_bytes(0x66, 0x0f, 0x7e, 0xc0);
            if (!flip_result) {
               // andl $1, %eax
               emit_bytes(0x83, 0xe0, 0x01);
            } else {
               // incl %eax {0xffffffff, 0} -> {0, 1}
               emit_bytes(0xff, 0xc0);
            }
            // leaq 16(%rsp), %rsp
            emit_bytes(0x48, 0x8d, 0x64, 0x24, 0x10);
            // pushq %rax
            emit_bytes(0x50);
         }
      }

      void emit_f64_relop(uint8_t opcode, uint64_t (*softfloatfun)(float64_t, float64_t), bool switch_params, bool flip_result) {
         COUNT_INSTR();
         if constexpr (use_softfloat) {
            auto extra = emit_setup_backtrace();
            // pushq %rdi
            emit_bytes(0x57);
            // pushq %rsi
            emit_bytes(0x56);
            if(switch_params) {
               // movq 24(%rsp), %rsi
               emit_bytes(0x48, 0x8b, 0x74, 0x24, 0x18 + extra);
               // movq 16(%rsp), %rdi
               emit_bytes(0x48, 0x8b, 0x7c, 0x24, 0x10 + extra);
            } else {
               // movq 16(%rsp), %rsi
               emit_bytes(0x48, 0x8b, 0x74, 0x24, 0x10 + extra);
               // movq 24(%rsp), %rdi
               emit_bytes(0x48, 0x8b, 0x7c, 0x24, 0x18 + extra);
            }
            emit_align_stack();
            // movabsq $softfloatfun, %rax
            emit_bytes(0x48, 0xb8);
            emit_operand_ptr(softfloatfun);
            // callq *%rax
            emit_bytes(0xff, 0xd0);
            emit_restore_stack();
            // popq %rsi
            emit_bytes(0x5e);
            // popq %rdi
            emit_bytes(0x5f);
            emit_restore_backtrace_basic();
            if (flip_result) {
               // xor $0x1, %al
               emit_bytes(0x34, 0x01);
            }
            // addq $8, %rsp
            emit_bytes(0x48, 0x83, 0xc4, 0x08 + extra);
            // movq %rax, (%rsp)
            emit_bytes(0x48, 0x89, 0x04, 0x24);
         } else {
            // ucomisd+seta/setae is shorter but can't handle eq/ne
            if(switch_params) {
               // movsd (%rsp), %xmm0
               emit_bytes(0xf2, 0x0f, 0x10, 0x04, 0x24);
               // cmpCCsd 8(%rsp), %xmm0
               emit_bytes(0xf2, 0x0f, 0xc2, 0x44, 0x24, 0x08, opcode);
            } else {
               // movsd 8(%rsp), %xmm0
               emit_bytes(0xf2, 0x0f, 0x10, 0x44, 0x24, 0x08);
               // cmpCCsd (%rsp), %xmm0
               emit_bytes(0xf2, 0x0f, 0xc2, 0x04, 0x24, opcode);
            }               
            // movd %xmm0, %eax
            emit_bytes(0x66, 0x0f, 0x7e, 0xc0);
            if (!flip_result) {
               // andl $1, eax
               emit_bytes(0x83, 0xe0, 0x01);
            } else {
               // incl %eax {0xffffffff, 0} -> {0, 1}
               emit_bytes(0xff, 0xc0);
            }
            // leaq 16(%rsp), %rsp
            emit_bytes(0x48, 0x8d, 0x64, 0x24, 0x10);
            // pushq %rax
            emit_bytes(0x50);
         }
      }

      template<class... T>
      void emit_i32_binop(T... op) {
         // popq %rcx
         emit_bytes(0x59);
         // popq %rax
         emit_bytes(0x58);
         // OP %eax, %ecx
         emit_bytes(static_cast<uint8_t>(op)...);
         // pushq %rax
         // emit_bytes(0x50);
      }

      template<class... T>
      void emit_i64_binop(T... op) {
         // popq %rcx
         emit_bytes(0x59);
         // popq %rax
         emit_bytes(0x58);
         // OP %eax, %ecx
         emit_bytes(static_cast<uint8_t>(op)...);
      }

      void emit_f32_binop(uint8_t op, float32_t (*softfloatfun)(float32_t, float32_t)) {
         COUNT_INSTR();
         if constexpr (use_softfloat) {
            return emit_f32_binop_softfloat(softfloatfun);
         }
         // movss 8(%rsp), %xmm0
         emit_bytes(0xf3, 0x0f, 0x10, 0x44, 0x24, 0x08);
         // OPss (%rsp), %xmm0
         emit_bytes(0xf3, 0x0f, op, 0x04, 0x24);
         // leaq 8(%rsp), %rsp
         emit_bytes(0x48, 0x8d, 0x64, 0x24, 0x08);
         // movss %xmm0, (%rsp)
         emit_bytes(0xf3, 0x0f, 0x11, 0x04, 0x24);
      }

      void emit_f64_binop(uint8_t op, float64_t (*softfloatfun)(float64_t, float64_t)) {
         COUNT_INSTR();
         if constexpr (use_softfloat) {
            return emit_f64_binop_softfloat(softfloatfun);
         }
         // movsd 8(%rsp), %xmm0
         emit_bytes(0xf2, 0x0f, 0x10, 0x44, 0x24, 0x08);
         // OPsd (%rsp), %xmm0
         emit_bytes(0xf2, 0x0f, op, 0x04, 0x24);
         // leaq 8(%rsp), %rsp
         emit_bytes(0x48, 0x8d, 0x64, 0x24, 0x08);
         // movsd %xmm0, (%rsp)
         emit_bytes(0xf2, 0x0f, 0x11, 0x04, 0x24);
      }

      // Beware: This pushes and pops mxcsr around the user op.  Remember to adjust access to %rsp in the caller.
      // Note uses %rcx after the user instruction
      template<class... T>
      void emit_f2i(T... op) {
         // mov 0x0x1f80, %eax // round-to-even/all exceptions masked/no exceptions set
         emit_bytes(0xb8, 0x80, 0x1f, 0x00, 0x00);
         // push %rax
         emit_bytes(0x50);
         // ldmxcsr (%rsp)
         emit_bytes(0x0f, 0xae, 0x14, 0x24);
         // user op
         emit_bytes(op...);
         // stmxcsr (%rsp)
         emit_bytes(0x0f, 0xae, 0x1c, 0x24);
         // pop %rcx
         emit_bytes(0x59);
         // test %cl, 0x1 // invalid
         emit_bytes(0xf6, 0xc1, 0x01);
         // jnz FP_ERROR_HANDLER
         emit_bytes(0x0f, 0x85);
         fix_branch(emit_branch_target32(), fpe_handler);
      }

      void emit_v128_unop_softfloat(v128_t (*softfloatfun)(v128_t)) {
         int32_t extra = emit_setup_backtrace();
         emit_push(rdi);
         emit_push(rsi);
         emit_mov(*(rsp + (16 + extra)), rdi);
         emit_mov(*(rsp + (24 + extra)), rsi);
         emit_align_stack(rax);
         // movabsq $softfloatfun, %rax
         emit_bytes(0x48, 0xb8);
         emit_operand_ptr(softfloatfun);
         // callq *%rax
         emit_bytes(0xff, 0xd0);
         emit_restore_stack();
         emit_pop(rsi);
         emit_pop(rdi);
         emit_restore_backtrace(); // FIXME: clobbers rdx
         emit_mov(rax, *rsp);
         emit_mov(rdx, *(rsp + 8));
      }

      void emit_v128_binop_softfloat(v128_t (*softfloatfun)(v128_t, v128_t)) {
         int32_t extra = emit_setup_backtrace();
         emit_push(rdi);
         emit_push(rsi);
         emit_mov(*(rsp + (16 + extra)), rdx);
         emit_mov(*(rsp + (24 + extra)), rcx);
         emit_mov(*(rsp + (32 + extra)), rdi);
         emit_mov(*(rsp + (40 + extra)), rsi);
         emit_align_stack(rax);
         // movabsq $softfloatfun, %rax
         emit_bytes(0x48, 0xb8);
         emit_operand_ptr(softfloatfun);
         // callq *%rax
         emit_bytes(0xff, 0xd0);
         emit_restore_stack();
         emit_pop(rsi);
         emit_pop(rdi);
         emit_restore_backtrace_basic(); // FIXME: clobbers rdx
         emit_add(16 + extra, rsp);
         emit_mov(rax, *rsp);
         emit_mov(rdx, *(rsp + 8));
      }

      void* emit_error_handler(void (*handler)()) {
         void* result = code;
         // andq $-16, %rsp;
         emit_bytes(0x48, 0x83, 0xe4, 0xf0);
         // movabsq &on_unreachable, %rax
         emit_bytes(0x48, 0xb8);
         emit_operand_ptr(handler);
         // callq *%rax
         emit_bytes(0xff, 0xd0);
         return result;
      }

      void emit_align_stack() {
         // mov %rsp, %rcx; andq $-16, %rsp; push rcx; push %rcx
         emit_bytes(0x48, 0x89, 0xe1);
         emit_bytes(0x48, 0x83, 0xe4, 0xf0);
         emit_bytes(0x51);
         emit_bytes(0x51);
      }
      void emit_align_stack(general_register64 temp) {
         // mov %rsp, %temp; andq $-16, %rsp; push temp; push %temp
         if(temp & 8) unimplemented();
         emit_bytes(0x48, 0x89, 0xe0 | temp);
         emit_bytes(0x48, 0x83, 0xe4, 0xf0);
         emit_bytes(0x50 | temp);
         emit_bytes(0x50 | temp);
      }

      void emit_restore_stack() {
         // mov (%rsp), %rsp
         emit_bytes(0x48, 0x8b, 0x24, 0x24);
      }

      void emit_host_call(uint32_t funcnum) {
         uint32_t extra = 0;
         if constexpr (Context::async_backtrace()) {
            // pushq %rbp
            emit_bytes(0x55);
            // movq %rsp, (%rdi)
            emit_bytes(0x48, 0x89, 0x27);
            extra = 8;
         }
         // mov $funcnum, %edx
         emit_bytes(0xba);
         emit_operand32(funcnum);
         // pushq %rdi
         emit_bytes(0x57);
         // pushq %rsi
         emit_bytes(0x56);
         // lea 24(%rsp), %rsi
         emit_bytes(0x48, 0x8d, 0x74, 0x24, 0x18 + extra);
         emit_align_stack();
         emit_mov(ebx, ecx);
         // movabsq $call_host_function, %rax
         emit_bytes(0x48, 0xb8);
         emit_operand_ptr(&call_host_function);
         // callq *%rax
         emit_bytes(0xff, 0xd0);
         emit_restore_stack();
         // popq %rsi
         emit_bytes(0x5e);
         // popq %rdi
         emit_bytes(0x5f);
         if constexpr (Context::async_backtrace()) {
            emit_restore_backtrace_basic();
            // popq %rbp
            emit_bytes(0x5d);
         }
         // retq
         emit_bytes(0xc3);
      }

      // Needs to run before saving %rdi.  Returns the number of bytes pushed onto the stack.
      uint32_t emit_setup_backtrace() {
         if constexpr (Context::async_backtrace()) {
            // callq next
            emit_bytes(0xe8);
            emit_operand32(0);
            // next:
            // pushq %rbp
            emit_bytes(0x55);
            // movq %rsp, (%rdi)
            emit_bytes(0x48, 0x89, 0x27);
            return 16;
         } else {
            return 0;
         }
      }
      // Does not adjust the stack pointer.  Use this if the
      // stack pointer adjustment is combined with another instruction.
      void emit_restore_backtrace_basic() {
         if constexpr (Context::async_backtrace()) {
            // xorl %edx, %edx
            emit_bytes(0x31, 0xd2);
            // movq %rdx, (%rdi)
            emit_bytes(0x48, 0x89, 0x17);
         }
      }
      void emit_restore_backtrace() {
         if constexpr (Context::async_backtrace()) {
            emit_restore_backtrace_basic();
            // addq $16, %rsp
            emit_bytes(0x48, 0x83, 0xc4, 0x10);
         }
      }

      bool is_host_function(uint32_t funcnum) { return funcnum < _mod.get_imported_functions_size(); }

      static native_value call_host_function(Context* context /*rdi*/, native_value* stack /*rsi*/, uint32_t idx /*edx*/, uint32_t remaining_stack) {
         // It's currently unsafe to throw through a jit frame, because we don't set up
         // the exception tables for them.
         native_value result;
         vm::longjmp_on_exception([&]() {
            auto saved = context->_remaining_call_depth;
            context->_remaining_call_depth = remaining_stack;
            scope_guard g{[&](){ context->_remaining_call_depth = saved; }};
            result = context->call_host_function(stack, idx);
         });
         return result;
      }

      static int32_t current_memory(Context* context /*rdi*/) {
         return context->current_linear_memory();
      }

      static int32_t grow_memory(Context* context /*rdi*/, int32_t pages) {
         return context->grow_linear_memory(pages);
      }

      static void init_memory(Context* context /*rdi*/, uint32_t x /*esi*/, uint32_t d /*edx*/, uint32_t s /*ecx*/, uint32_t n /*r8d*/) {
         context->init_linear_memory(x, d, s, n);
      }

      static void drop_data(Context* context /*rdi*/, uint32_t x /*esi*/) {
         context->drop_data(x);
      }

      static void init_table(Context* context /*rdi*/, uint32_t x /*esi*/, uint32_t d /*edx*/, uint32_t s /*ecx*/, uint32_t n /*r8d*/) {
         context->init_table(x, d, s, n);
      }

      static void drop_elem(Context* context /*rdi*/, uint32_t x /*esi*/) {
         context->drop_elem(x);
      }

      static void on_memory_error() { throw_<wasm_memory_exception>("wasm memory out-of-bounds"); }

      static void on_unreachable() { vm::throw_<wasm_interpreter_exception>( "unreachable" ); }
      static void on_fp_error() { vm::throw_<wasm_interpreter_exception>( "floating point error" ); }
      static void on_call_indirect_error() { vm::throw_<wasm_interpreter_exception>( "call_indirect out of range" ); }
      static void on_type_error() { vm::throw_<wasm_interpreter_exception>( "call_indirect incorrect function type" ); }
      static void on_stack_overflow() { vm::throw_<wasm_interpreter_exception>( "stack overflow" ); }
   };
   
}}
