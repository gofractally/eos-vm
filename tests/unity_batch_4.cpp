// Unity build batch 4: Allocator, section elements, implementation limits, and backend tests

#include <algorithm>
#include <vector>
#include <iterator>
#include <cstdlib>
#include <fstream>
#include <string>
#include <catch2/catch.hpp>
#include <eosio/vm/backend.hpp>
#include <eosio/vm/allocator.hpp>
#include "wasm_config.hpp"
#include "utils.hpp"

namespace unity_allocator_tests {
#include "allocator_tests.cpp"
}

namespace unity_max_section_elements_tests {
#include "max_section_elements_tests.cpp"
}

namespace unity_implementation_limits_tests {
#include "implementation_limits_tests.cpp"
}

namespace unity_backend_tests {
#include "backend_tests.cpp"
}
