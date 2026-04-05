# ##################################################################################################
# Comparative benchmarks: eos-vm vs other WASM runtimes
# Each runtime is independently optional; the benchmark adapts to what's available.
# ##################################################################################################

include(FetchContent)

set(BENCH_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/bench_compare.cpp)
set(BENCH_LIBS eos-vm)
set(BENCH_DEFS "")

# ##################################################################################################
# wasm3 — Pure C interpreter, trivial to build
# ##################################################################################################
option(ENABLE_BENCH_WASM3 "Fetch and benchmark wasm3 interpreter" OFF)
if(ENABLE_BENCH_WASM3)
   FetchContent_Declare(
      wasm3
      GIT_REPOSITORY https://github.com/wasm3/wasm3.git
      GIT_TAG        main
      GIT_SHALLOW    TRUE
   )
   # wasm3 doesn't have a top-level CMakeLists.txt that builds a library.
   # We compile the source files directly.
   FetchContent_GetProperties(wasm3)
   if(NOT wasm3_POPULATED)
      FetchContent_Populate(wasm3)
   endif()
   # Check if this is the real wasm3 repo
   if(EXISTS "${wasm3_SOURCE_DIR}/source/wasm3.h")
      file(GLOB WASM3_SOURCES "${wasm3_SOURCE_DIR}/source/*.c")
      add_library(wasm3 STATIC ${WASM3_SOURCES})
      target_include_directories(wasm3 PUBLIC "${wasm3_SOURCE_DIR}/source")
      target_compile_options(wasm3 PRIVATE -w)
      list(APPEND BENCH_LIBS wasm3)
      list(APPEND BENCH_DEFS BENCH_HAS_WASM3)
      message(STATUS "wasm3 benchmark: ENABLED")
   else()
      message(WARNING "wasm3 source not found at expected location, disabling")
   endif()
endif()

# ##################################################################################################
# WAMR (wasm-micro-runtime) — C runtime with interpreter + AOT + JIT
# ##################################################################################################
option(ENABLE_BENCH_WAMR "Fetch and benchmark WAMR" OFF)
if(ENABLE_BENCH_WAMR)
   FetchContent_Declare(
      wamr
      GIT_REPOSITORY https://github.com/bytecodealliance/wasm-micro-runtime.git
      GIT_TAG        main
      GIT_SHALLOW    TRUE
   )
   FetchContent_GetProperties(wamr)
   if(NOT wamr_POPULATED)
      FetchContent_Populate(wamr)
   endif()
   if(EXISTS "${wamr_SOURCE_DIR}/build-scripts/runtime_lib.cmake")
      set(WAMR_BUILD_PLATFORM "darwin" CACHE STRING "" FORCE)
      set(WAMR_BUILD_INTERP 1)
      set(WAMR_BUILD_FAST_INTERP 1)
      set(WAMR_BUILD_AOT 0)
      set(WAMR_BUILD_JIT 0)
      set(WAMR_BUILD_LIBC_BUILTIN 0)
      set(WAMR_BUILD_LIBC_WASI 0)
      set(WAMR_BUILD_TAIL_CALL 0)
      set(WAMR_BUILD_SIMD 0)
      set(WAMR_BUILD_MINI_LOADER 0)
      set(WAMR_BUILD_TARGET "${CMAKE_SYSTEM_PROCESSOR}")
      set(WAMR_ROOT_DIR "${wamr_SOURCE_DIR}")
      include(${wamr_SOURCE_DIR}/build-scripts/runtime_lib.cmake)
      add_library(vmlib STATIC ${WAMR_RUNTIME_LIB_SOURCE})
      target_include_directories(vmlib PUBLIC
         "${wamr_SOURCE_DIR}/core/iwasm/include"
         "${wamr_SOURCE_DIR}/core/shared/utils"
         "${wamr_SOURCE_DIR}/core/shared/platform/include"
      )
      target_compile_options(vmlib PRIVATE -w)
      list(APPEND BENCH_LIBS vmlib)
      list(APPEND BENCH_DEFS BENCH_HAS_WAMR)
      message(STATUS "WAMR benchmark: ENABLED")
   else()
      message(WARNING "WAMR build scripts not found, disabling")
   endif()
endif()

# ##################################################################################################
# wasmtime — Rust-based JIT, uses prebuilt C API release
# ##################################################################################################
option(ENABLE_BENCH_WASMTIME "Download prebuilt wasmtime C API for benchmarking" OFF)
if(ENABLE_BENCH_WASMTIME)
   # Determine platform for prebuilt download
   set(WASMTIME_VERSION "v30.0.0")
   if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
         set(WASMTIME_ARCH "aarch64-macos")
      else()
         set(WASMTIME_ARCH "x86_64-macos")
      endif()
   elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
         set(WASMTIME_ARCH "aarch64-linux")
      else()
         set(WASMTIME_ARCH "x86_64-linux")
      endif()
   else()
      message(WARNING "Unsupported platform for wasmtime prebuilt, disabling")
      set(ENABLE_BENCH_WASMTIME OFF)
   endif()

   if(ENABLE_BENCH_WASMTIME)
      set(WASMTIME_TARBALL "wasmtime-${WASMTIME_VERSION}-${WASMTIME_ARCH}-c-api.tar.xz")
      set(WASMTIME_URL "https://github.com/bytecodealliance/wasmtime/releases/download/${WASMTIME_VERSION}/${WASMTIME_TARBALL}")
      set(WASMTIME_DIR "${CMAKE_CURRENT_BINARY_DIR}/wasmtime")

      if(NOT EXISTS "${WASMTIME_DIR}/lib")
         message(STATUS "Downloading wasmtime ${WASMTIME_VERSION} for ${WASMTIME_ARCH}...")
         file(DOWNLOAD "${WASMTIME_URL}" "${CMAKE_CURRENT_BINARY_DIR}/${WASMTIME_TARBALL}"
              STATUS WASMTIME_DL_STATUS SHOW_PROGRESS)
         list(GET WASMTIME_DL_STATUS 0 WASMTIME_DL_RC)
         if(WASMTIME_DL_RC EQUAL 0)
            file(ARCHIVE_EXTRACT INPUT "${CMAKE_CURRENT_BINARY_DIR}/${WASMTIME_TARBALL}"
                 DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
            # The archive extracts to a directory with the version in the name
            file(GLOB WASMTIME_EXTRACTED "${CMAKE_CURRENT_BINARY_DIR}/wasmtime-${WASMTIME_VERSION}-${WASMTIME_ARCH}-c-api")
            if(WASMTIME_EXTRACTED)
               file(RENAME "${WASMTIME_EXTRACTED}" "${WASMTIME_DIR}")
            endif()
         else()
            message(WARNING "Failed to download wasmtime: ${WASMTIME_DL_STATUS}")
            set(ENABLE_BENCH_WASMTIME OFF)
         endif()
      endif()

      if(ENABLE_BENCH_WASMTIME AND EXISTS "${WASMTIME_DIR}/lib")
         add_library(wasmtime STATIC IMPORTED)
         set_target_properties(wasmtime PROPERTIES
            IMPORTED_LOCATION "${WASMTIME_DIR}/lib/libwasmtime.a"
            INTERFACE_INCLUDE_DIRECTORIES "${WASMTIME_DIR}/include"
         )
         list(APPEND BENCH_LIBS wasmtime)
         list(APPEND BENCH_DEFS BENCH_HAS_WASMTIME)
         message(STATUS "wasmtime benchmark: ENABLED (${WASMTIME_VERSION})")
      endif()
   endif()
endif()

# ##################################################################################################
# Wasmer — Rust-based runtime, uses prebuilt C API release
# ##################################################################################################
option(ENABLE_BENCH_WASMER "Download prebuilt wasmer C API for benchmarking" OFF)
if(ENABLE_BENCH_WASMER)
   set(WASMER_VERSION "v5.0.4")
   if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
      if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
         set(WASMER_TARBALL "wasmer-darwin-arm64.tar.gz")
      else()
         set(WASMER_TARBALL "wasmer-darwin-amd64.tar.gz")
      endif()
   elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
         set(WASMER_TARBALL "wasmer-linux-aarch64.tar.gz")
      else()
         set(WASMER_TARBALL "wasmer-linux-amd64.tar.gz")
      endif()
   else()
      message(WARNING "Unsupported platform for wasmer prebuilt, disabling")
      set(ENABLE_BENCH_WASMER OFF)
   endif()

   if(ENABLE_BENCH_WASMER)
      set(WASMER_URL "https://github.com/wasmerio/wasmer/releases/download/${WASMER_VERSION}/${WASMER_TARBALL}")
      set(WASMER_DIR "${CMAKE_CURRENT_BINARY_DIR}/wasmer")

      if(NOT EXISTS "${WASMER_DIR}/lib")
         message(STATUS "Downloading wasmer ${WASMER_VERSION}...")
         file(DOWNLOAD "${WASMER_URL}" "${CMAKE_CURRENT_BINARY_DIR}/${WASMER_TARBALL}"
              STATUS WASMER_DL_STATUS SHOW_PROGRESS)
         list(GET WASMER_DL_STATUS 0 WASMER_DL_RC)
         if(WASMER_DL_RC EQUAL 0)
            file(MAKE_DIRECTORY "${WASMER_DIR}")
            file(ARCHIVE_EXTRACT INPUT "${CMAKE_CURRENT_BINARY_DIR}/${WASMER_TARBALL}"
                 DESTINATION "${WASMER_DIR}")
         else()
            message(WARNING "Failed to download wasmer: ${WASMER_DL_STATUS}")
            set(ENABLE_BENCH_WASMER OFF)
         endif()
      endif()

      if(ENABLE_BENCH_WASMER AND EXISTS "${WASMER_DIR}/lib")
         add_library(wasmer STATIC IMPORTED)
         # Find the actual library file
         find_library(WASMER_LIB NAMES wasmer libwasmer PATHS "${WASMER_DIR}/lib" NO_DEFAULT_PATH)
         if(WASMER_LIB)
            set_target_properties(wasmer PROPERTIES
               IMPORTED_LOCATION "${WASMER_LIB}"
               INTERFACE_INCLUDE_DIRECTORIES "${WASMER_DIR}/include"
            )
            list(APPEND BENCH_LIBS wasmer)
            list(APPEND BENCH_DEFS BENCH_HAS_WASMER)
            message(STATUS "wasmer benchmark: ENABLED (${WASMER_VERSION})")
         else()
            message(WARNING "wasmer library not found in ${WASMER_DIR}/lib")
         endif()
      endif()
   endif()
endif()

# ##################################################################################################
# Build the benchmark executable
# ##################################################################################################
add_executable(bench-compare ${BENCH_SOURCES})
target_link_libraries(bench-compare ${BENCH_LIBS})
target_compile_definitions(bench-compare PRIVATE ${BENCH_DEFS})
