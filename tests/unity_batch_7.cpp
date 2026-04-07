// Unity build batch 7: Remaining limit tests, instantiation, null, and watchdog

#include <iostream>
#include <list>
#include <atomic>
#include <chrono>
#include <catch2/catch.hpp>
#include <eosio/vm/backend.hpp>
#include <eosio/vm/watchdog.hpp>
#include "utils.hpp"

namespace unity_instantiation_tests {
#include "instantiation_tests.cpp"
}

namespace unity_max_nested_structures_tests {
#include "max_nested_structures_tests.cpp"
}

namespace unity_max_local_sets_tests {
#include "max_local_sets_tests.cpp"
}

namespace unity_max_code_bytes_tests {
#include "max_code_bytes_tests.cpp"
}

namespace unity_max_br_table_elements_tests {
#include "max_br_table_elements_tests.cpp"
}

namespace unity_allow_invalid_empty_local_set_tests {
#include "allow_invalid_empty_local_set_tests.cpp"
}

namespace unity_forbid_export_mutable_globals_tests {
#include "forbid_export_mutable_globals_tests.cpp"
}

namespace unity_max_table_elements_tests {
#include "max_table_elements_tests.cpp"
}

namespace unity_null_tests {
#include "null_tests.cpp"
}

namespace unity_watchdog_tests {
#include "watchdog_tests.cpp"
}
