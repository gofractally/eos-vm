#pragma once
#include <cstdint>

namespace eosio { namespace vm {
   enum constants {
      magic   = 0x6D736100,
      version = 0x1,
      magic_size   = sizeof(uint32_t),
      version_size = sizeof(uint32_t),
      id_size      = sizeof(uint8_t),
      varuint32_size = 5,
      max_call_depth        = 250,
      // initial_stack_size is used for reserving initial memory for operand stack.
      // For JIT, operand stack is only used in calling host function calls, where
      // number of elements required can never be big.
      // For Interpreter, performance is not a concern.
      // Intentionally set to a small number.
      initial_stack_size    = 8,
      initial_module_size   = 1 * 1024 * 1024,
      max_memory            = 4ull << 31,
      max_useable_memory    = (1ull << 32), //4GiB
      page_size             = 64ull * 1024, //64kb
      max_pages             = (max_useable_memory/page_size)
   };
}} // namespace eosio::vm
