// Unity build batch 5: Code/memory/stack limit tests

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

namespace unity_allow_code_after_function_end_tests {
#include "allow_code_after_function_end_tests.cpp"
}

namespace unity_max_linear_memory_init_tests {
#include "max_linear_memory_init_tests.cpp"
}

namespace unity_max_memory_offset_tests {
#include "max_memory_offset_tests.cpp"
}

namespace unity_allow_u32_limits_flags_tests {
#include "allow_u32_limits_flags_tests.cpp"
}

namespace unity_max_stack_bytes_tests {
#include "max_stack_bytes_tests.cpp"
}

namespace unity_call_return_stack_bytes_tests {
#include "call_return_stack_bytes_tests.cpp"
}

namespace unity_globals_memory_tests {
#include "globals_memory_tests.cpp"
}

namespace unity_call_indirect_tests {
#include "call_indirect_tests.cpp"
}
