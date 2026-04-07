// Unity build batch 2: Mutable globals and func local bytes tests

#include <eosio/vm/backend.hpp>
#include "utils.hpp"
#include <catch2/catch.hpp>

namespace unity_max_mutable_globals_tests {
#include "max_mutable_globals_tests.cpp"
}

namespace unity_max_func_local_bytes_tests {
#include "max_func_local_bytes_tests.cpp"
}
