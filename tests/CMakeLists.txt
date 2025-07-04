# #############
# Defines unit tests, spec tests and fuzz tests for eos-vm.
# ############
option(ENABLE_TESTS "enable building of unit tests, spec tests." ON)
cmake_dependent_option(ENABLE_FUZZ_TESTS "enable fuzz testing" OFF "ENABLE_TESTS" ON)

configure_file(wasm_config.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/wasm_config.hpp)
include_directories(${CMAKE_SOURCE_DIR}/external/Catch2/single_include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
#add_subdirectory(memory_tests)
if (ENABLE_FUZZ_TESTS)
   add_subdirectory(fuzz)
endif()

enable_testing()

# ##############
# define spec tests
# ##############

set( WASM_SPEC_TESTS_LOCATION "" CACHE STRING "The location of the webassembly spec tests." )
find_program(WAST2JSON wast2json)

# These tests require manual maintenance
#set(SPEC_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/spec/e_memory_tests.cpp)
set(SPEC_TESTS)

macro( add_spec_test_impl name source)
   list( APPEND SPEC_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/spec/${name}_tests.cpp )
   if ( WAST2JSON )
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/spec/${name}.json
         COMMAND ${WAST2JSON} --no-check ${source}  -o ${name}.json && mv ${name}.json ${CMAKE_CURRENT_BINARY_DIR}/spec/${name}.json
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/wasms
         DEPENDS ${source}
      )
      add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/spec/${name}_tests.cpp
         COMMAND spec_test_generator ${CMAKE_CURRENT_BINARY_DIR}/spec/${name}.json > ${CMAKE_CURRENT_SOURCE_DIR}/spec/${name}_tests.cpp
         DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/spec/${name}.json spec_test_generator)
      message("Generating: ${name}_tests.cpp")
   else()
      message("Using existing: ${name}_tests.cpp")
   endif()
endmacro()

macro( add_spec_test name )
   if ( WASM_SPEC_TESTS_LOCATION )
      add_spec_test_impl(${name} ${WASM_SPEC_TESTS_LOCATION}/${name}.wast)
   else()
      list( APPEND SPEC_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/spec/${name}_tests.cpp )
      message("Using existing: ${name}_tests.cpp")
   endif()
endmacro()

if (ENABLE_SPEC_TESTS)

foreach(testcase address align binary binary-leb128 block br br_if br_table break-drop call_indirect
                 call const conversions custom data elem
                 endianness f32 f32_bitwise f32_cmp f64 f64_bitwise f64_cmp
                 fac float_exprs float_literals float_memory float_misc forward
                 func func_ptrs globals i32 i64 if int_exprs int_literals labels left-to-right
                 load local_get local_set local_tee loop memory
                 bulk memory_init memory_fill memory_copy table_init table_copy
                 memory_redundancy memory_grow memory_size memory_trap names nop return
                 select stack start store switch traps type typecheck
                 unreachable unreached-invalid unwind
                 utf8-custom-section-id utf8-import-field utf8-import-module
                 )
   add_spec_test(${testcase})
endforeach()

add_spec_test_impl(e_block ${CMAKE_CURRENT_SOURCE_DIR}/spec/extra/block.wast)
add_spec_test_impl(e_function ${CMAKE_CURRENT_SOURCE_DIR}/spec/extra/function.wast)
add_spec_test_impl(e_globals ${CMAKE_CURRENT_SOURCE_DIR}/spec/extra/globals.wast)
add_spec_test_impl(e_locals ${CMAKE_CURRENT_SOURCE_DIR}/spec/extra/locals.wast)
#wast2json chokes on invalid binary modules
#add_spec_test_impl(e_memory ${CMAKE_CURRENT_SOURCE_DIR}/spec/extra/memory.wast)
add_spec_test_impl(e_module ${CMAKE_CURRENT_SOURCE_DIR}/spec/extra/module.wast)
add_spec_test_impl(e_table ${CMAKE_CURRENT_SOURCE_DIR}/spec/extra/table.wast)
add_spec_test_impl(e_simd ${CMAKE_CURRENT_SOURCE_DIR}/spec/extra/simd.wast)

add_executable( eos_vm_spec_tests main.cpp ${SPEC_TESTS} )
target_link_libraries( eos_vm_spec_tests eos-vm Catch2::Catch2 )
target_include_directories( eos_vm_spec_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )
target_include_directories( eos_vm_spec_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include )
target_include_directories( eos_vm_spec_tests PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/spec )

target_compile_definitions(eos_vm_spec_tests PUBLIC -DCATCH_CONFIG_NO_POSIX_SIGNALS)

catch_discover_tests( eos_vm_spec_tests )

set(SPEC_TESTS)
foreach(testcase
    simd_address simd_align simd_bit_shift simd_bitwise simd_boolean simd_const
    simd_conversions
    simd_f32x4 simd_f32x4_arith simd_f32x4_cmp simd_f32x4_pmin_pmax simd_f32x4_rounding
    simd_f64x2 simd_f64x2_arith simd_f64x2_cmp simd_f64x2_pmin_pmax simd_f64x2_rounding
    simd_i8x16_arith simd_i8x16_arith2 simd_i8x16_cmp simd_i8x16_sat_arith
    simd_i16x8_arith simd_i16x8_arith2 simd_i16x8_cmp simd_i16x8_extadd_pairwise_i8x16
    simd_i16x8_extmul_i8x16 simd_i16x8_q15mulr_sat_s simd_i16x8_sat_arith
    simd_i32x4_arith simd_i32x4_arith2 simd_i32x4_cmp simd_i32x4_dot_i16x8
    simd_i32x4_extadd_pairwise_i16x8 simd_i32x4_extmul_i16x8
    simd_i32x4_trunc_sat_f32x4 simd_i32x4_trunc_sat_f64x2
    simd_i64x2_arith simd_i64x2_arith2 simd_i64x2_cmp simd_i64x2_extmul_i32x4
    simd_int_to_int_extend simd_lane simd_load
    simd_load8_lane simd_load16_lane simd_load32_lane simd_load64_lane
    simd_load_extend simd_load_splat simd_load_zero
    simd_splat simd_store
    simd_store8_lane simd_store16_lane simd_store32_lane simd_store64_lane
    )
   add_spec_test(${testcase})
endforeach()
add_executable( eos_vm_simd_tests main.cpp ${SPEC_TESTS} )
target_link_libraries( eos_vm_simd_tests eos-vm Catch2::Catch2 )
target_include_directories( eos_vm_simd_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )
target_include_directories( eos_vm_simd_tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include )
target_include_directories( eos_vm_simd_tests PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/spec )
target_compile_definitions(eos_vm_simd_tests PUBLIC -DCATCH_CONFIG_NO_POSIX_SIGNALS)
catch_discover_tests( eos_vm_simd_tests )

endif(ENABLE_SPEC_TESTS)

add_executable(unit_tests main.cpp
                          allocator_tests.cpp
                          eosio_max_nested_structures_tests.cpp
                          guarded_ptr_tests.cpp
                          varint_tests.cpp
                          variant_tests.cpp
                          host_functions_tests.cpp
                          preconditions_tests.cpp
                          allow_code_after_function_end_tests.cpp
                          allow_invalid_empty_local_set_tests.cpp
                          allow_u32_limits_flags_tests.cpp
                          allow_zero_blocktype_tests.cpp
                          forbid_export_mutable_globals_tests.cpp
                          max_br_table_elements_tests.cpp
                          max_code_bytes_tests.cpp
                          max_func_local_bytes_tests.cpp
                          max_linear_memory_init_tests.cpp
                          max_local_sets_tests.cpp
                          max_memory_offset_tests.cpp
                          max_mutable_globals_tests.cpp
                          max_nested_structures_tests.cpp
                          max_pages_tests.cpp
                          max_section_elements_tests.cpp
                          max_stack_bytes_tests.cpp
                          max_table_elements_tests.cpp
                          null_tests.cpp
                          reentry_tests.cpp
                          signals_tests.cpp
                          stack_restriction_tests.cpp
                          watchdog_tests.cpp
                          implementation_limits_tests.cpp
                          instantiation_tests.cpp
                          backend_tests.cpp
                          vector_tests.cpp)

target_link_libraries(unit_tests eos-vm Catch2::Catch2)

# ##############
# disable catch2 from installing it's own signal handlers
# ##############
target_compile_definitions(unit_tests PUBLIC -DCATCH_CONFIG_NO_POSIX_SIGNALS)

add_executable( spec_test_generator ${CMAKE_CURRENT_SOURCE_DIR}/spec_test_generator/spec_test_generator.cpp )

catch_discover_tests( unit_tests )
