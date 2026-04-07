// Unity build batch 6: Signals, reentry, stack, pages, and structure tests

#include <algorithm>
#include <vector>
#include <iterator>
#include <cstdlib>
#include <fstream>
#include <string>
#include <iostream>
#include <chrono>
#include <csignal>
#include <thread>
#include <catch2/catch.hpp>
#include <eosio/vm/backend.hpp>
#include <eosio/vm/signals.hpp>
#include <eosio/vm/watchdog.hpp>
#include "utils.hpp"

namespace unity_signals_tests {
#include "signals_tests.cpp"
}

namespace unity_reentry_tests {
#include "reentry_tests.cpp"
}

namespace unity_stack_restriction_tests {
#include "stack_restriction_tests.cpp"
}

namespace unity_max_pages_tests {
#include "max_pages_tests.cpp"
}

namespace unity_eosio_max_nested_structures_tests {
#include "eosio_max_nested_structures_tests.cpp"
}

namespace unity_allow_zero_blocktype_tests {
#include "allow_zero_blocktype_tests.cpp"
}
