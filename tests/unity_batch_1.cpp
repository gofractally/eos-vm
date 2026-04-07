// Unity build batch 1: Host functions and guarded pointer tests
// All common headers included at global scope first, then test files
// wrapped in unique namespaces to avoid anonymous-namespace collisions.

#include <algorithm>
#include <vector>
#include <iterator>
#include <cstdlib>
#include <fstream>
#include <string>
#include <catch2/catch.hpp>
#include <eosio/vm/backend.hpp>
#include "wasm_config.hpp"
#include "utils.hpp"
#include <eosio/vm/guarded_ptr.hpp>

namespace unity_host_functions_tests {
#include "host_functions_tests.cpp"
}

namespace unity_guarded_ptr_tests {
#include "guarded_ptr_tests.cpp"
}
