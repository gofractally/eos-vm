/**
 *  @file
 *  @copyright defined in eos/LICENSE.txt
 */
#include <eosio/vm/allocator.hpp>

#include "utils.hpp"

#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

eosio::vm::wasm_allocator wa;

template class eosio::vm::backend<eosio::vm::standalone_function_t, eosio::vm::interpreter>;
#ifdef __x86_64__
template class eosio::vm::backend<eosio::vm::standalone_function_t, eosio::vm::jit>;
#endif
