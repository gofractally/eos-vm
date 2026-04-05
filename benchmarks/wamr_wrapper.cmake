# Wrapper CMakeLists.txt for WAMR — built in isolated subdirectory scope
# to prevent WAMR's cmake scripts from polluting global compile options.
cmake_minimum_required(VERSION 3.12)

set(wamr_SOURCE_DIR "@wamr_SOURCE_DIR@")

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
   set(WAMR_BUILD_PLATFORM "darwin")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
   set(WAMR_BUILD_PLATFORM "linux")
endif()

set(WAMR_BUILD_INTERP 1)
set(WAMR_BUILD_FAST_INTERP 1)
set(WAMR_BUILD_AOT 0)
set(WAMR_BUILD_JIT 0)
set(WAMR_BUILD_LIBC_BUILTIN 0)
set(WAMR_BUILD_LIBC_WASI 0)
set(WAMR_BUILD_TAIL_CALL 0)
set(WAMR_BUILD_SIMD 0)
set(WAMR_BUILD_MINI_LOADER 0)
# WAMR expects "AARCH64" not "arm64" (which incorrectly matches ARM.* pattern)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
   set(WAMR_BUILD_TARGET "AARCH64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
   set(WAMR_BUILD_TARGET "X86_64")
else()
   set(WAMR_BUILD_TARGET "${CMAKE_SYSTEM_PROCESSOR}")
endif()
set(WAMR_ROOT_DIR "${wamr_SOURCE_DIR}")

include(${wamr_SOURCE_DIR}/build-scripts/runtime_lib.cmake)

add_library(vmlib STATIC ${WAMR_RUNTIME_LIB_SOURCE})
target_include_directories(vmlib PUBLIC
   "${wamr_SOURCE_DIR}/core/iwasm/include"
   "${wamr_SOURCE_DIR}/core/shared/utils"
   "${wamr_SOURCE_DIR}/core/shared/platform/include"
)
target_compile_options(vmlib PRIVATE -w)
