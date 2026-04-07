// Unity build batch 3: Variant, varint, vector, and preconditions tests

#include <algorithm>
#include <vector>
#include <iterator>
#include <cstdlib>
#include <fstream>
#include <string>
#include <limits>
#include <exception>
#include <catch2/catch.hpp>
#include <eosio/vm/backend.hpp>
#include <eosio/vm/host_function.hpp>
#include <eosio/vm/variant.hpp>
#include <eosio/vm/leb128.hpp>
#include <eosio/vm/types.hpp>
#include <eosio/vm/vector.hpp>
#include "utils.hpp"

namespace unity_variant_tests {
#include "variant_tests.cpp"
}

namespace unity_varint_tests {
#include "varint_tests.cpp"
}

namespace unity_vector_tests {
#include "vector_tests.cpp"
}

namespace unity_preconditions_tests {
#include "preconditions_tests.cpp"
}
