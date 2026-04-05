# ##################################################################################################
# Comparative benchmarks: eos-vm vs other WASM runtimes
# Each runtime is independently optional; the benchmark adapts to what's available.
# ##################################################################################################

include(FetchContent)

set(BENCH_SOURCES
   ${CMAKE_CURRENT_SOURCE_DIR}/bench_compare.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/bench_wasmtime.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/bench_wasmer.cpp
)
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
      # Build WAMR in an isolated subdirectory to prevent its cmake scripts
      # from polluting global compile options (-marm, -m32, etc.)
      configure_file(
         "${CMAKE_CURRENT_SOURCE_DIR}/wamr_wrapper.cmake"
         "${CMAKE_CURRENT_BINARY_DIR}/wamr_build/CMakeLists.txt"
         @ONLY
      )
      add_subdirectory("${CMAKE_CURRENT_BINARY_DIR}/wamr_build" "${CMAKE_CURRENT_BINARY_DIR}/wamr_build")
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
         # Use static library directly to avoid dylib runtime dependency
         if(EXISTS "${WASMER_DIR}/lib/libwasmer.a")
            set(WASMER_LIB "${WASMER_DIR}/lib/libwasmer.a")
         else()
            find_library(WASMER_LIB NAMES wasmer libwasmer PATHS "${WASMER_DIR}/lib" NO_DEFAULT_PATH)
         endif()
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
# Validate: wasmtime and wasmer both export wasm-c-api symbols and cannot be linked together
# ##################################################################################################
# WAMR, wasmtime, and wasmer all export wasm-c-api symbols (wasm_engine_new, etc.)
# Only one of these three can be linked at a time.
set(_wasm_c_api_count 0)
if(ENABLE_BENCH_WAMR)
   math(EXPR _wasm_c_api_count "${_wasm_c_api_count} + 1")
endif()
if(ENABLE_BENCH_WASMTIME)
   math(EXPR _wasm_c_api_count "${_wasm_c_api_count} + 1")
endif()
if(ENABLE_BENCH_WASMER)
   math(EXPR _wasm_c_api_count "${_wasm_c_api_count} + 1")
endif()
if(_wasm_c_api_count GREATER 1)
   message(FATAL_ERROR "Cannot enable more than one of WAMR, wasmtime, or wasmer — they export conflicting wasm-c-api symbols. Enable only one at a time. wasm3 can be combined with any single one.")
endif()

# ##################################################################################################
# Compute benchmarks — standalone WASM modules (SHA-256, ECDSA) compiled from C
# ##################################################################################################
option(ENABLE_BENCH_COMPUTE "Build compute-heavy WASM benchmarks (SHA-256, ECDSA)" ON)
if(ENABLE_BENCH_COMPUTE)
   # Find wasi-sdk
   if(NOT WASI_SDK_PREFIX)
      # Check common locations
      foreach(_candidate
         "$ENV{WASI_SDK_PREFIX}"
         "${CMAKE_CURRENT_BINARY_DIR}/../../build/wasi-sdk-prefix/src/wasi-sdk"
         "/opt/wasi-sdk"
      )
         if(EXISTS "${_candidate}/bin/clang")
            set(WASI_SDK_PREFIX "${_candidate}")
            break()
         endif()
      endforeach()
   endif()

   if(NOT WASI_SDK_PREFIX OR NOT EXISTS "${WASI_SDK_PREFIX}/bin/clang")
      message(WARNING "wasi-sdk not found. Set WASI_SDK_PREFIX to enable compute benchmarks.\n"
                      "  cmake .. -DWASI_SDK_PREFIX=/path/to/wasi-sdk")
      set(ENABLE_BENCH_COMPUTE OFF)
   else()
      message(STATUS "wasi-sdk found: ${WASI_SDK_PREFIX}")
   endif()
endif()

if(ENABLE_BENCH_COMPUTE)
   set(WASM_CC "${WASI_SDK_PREFIX}/bin/clang")
   set(WASM_CFLAGS --target=wasm32-unknown-unknown -O3 -nostdlib
                   -Wl,--no-entry -Wl,--export-dynamic -Wl,--allow-undefined
                   -mbulk-memory -msign-ext)

   # --- SHA-256 WASM module ---
   set(SHA256_WASM "${CMAKE_CURRENT_BINARY_DIR}/bench_sha256.wasm")
   add_custom_command(
      OUTPUT "${SHA256_WASM}"
      COMMAND "${WASM_CC}" ${WASM_CFLAGS}
              -o "${SHA256_WASM}"
              "${CMAKE_CURRENT_SOURCE_DIR}/compute/bench_sha256.c"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/compute/bench_sha256.c"
      COMMENT "Compiling bench_sha256.c -> WASM"
   )

   # --- micro-ecc (ECDSA library) ---
   FetchContent_Declare(
      micro_ecc
      GIT_REPOSITORY https://github.com/kmackay/micro-ecc.git
      GIT_TAG        master
      GIT_SHALLOW    TRUE
   )
   FetchContent_GetProperties(micro_ecc)
   if(NOT micro_ecc_POPULATED)
      FetchContent_Populate(micro_ecc)
   endif()

   # --- ECDSA WASM module ---
   set(ECDSA_WASM "${CMAKE_CURRENT_BINARY_DIR}/bench_ecdsa.wasm")
   add_custom_command(
      OUTPUT "${ECDSA_WASM}"
      COMMAND "${WASM_CC}" ${WASM_CFLAGS}
              -DuECC_OPTIMIZATION_LEVEL=3
              -DuECC_SUPPORTS_secp160r1=0
              -DuECC_SUPPORTS_secp192r1=0
              -DuECC_SUPPORTS_secp224r1=0
              -DuECC_SUPPORTS_secp256r1=0
              -DuECC_SUPPORTS_secp256k1=1
              -DuECC_SUPPORT_COMPRESSED_POINT=0
              -I"${micro_ecc_SOURCE_DIR}"
              -o "${ECDSA_WASM}"
              "${CMAKE_CURRENT_SOURCE_DIR}/compute/bench_ecdsa.c"
              "${micro_ecc_SOURCE_DIR}/uECC.c"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/compute/bench_ecdsa.c"
              "${micro_ecc_SOURCE_DIR}/uECC.c"
      COMMENT "Compiling bench_ecdsa.c + micro-ecc -> WASM"
   )

   add_custom_target(compute-wasm ALL DEPENDS "${SHA256_WASM}" "${ECDSA_WASM}")
   list(APPEND BENCH_DEFS BENCH_HAS_COMPUTE)
   message(STATUS "Compute benchmarks: ENABLED (SHA-256, ECDSA secp256k1)")
endif()

# ##################################################################################################
# Build the benchmark executable
# ##################################################################################################
add_executable(bench-compare ${BENCH_SOURCES})
target_link_libraries(bench-compare ${BENCH_LIBS})
target_compile_definitions(bench-compare PRIVATE ${BENCH_DEFS})
if(ENABLE_BENCH_COMPUTE)
   add_dependencies(bench-compare compute-wasm)
   target_compile_definitions(bench-compare PRIVATE
      BENCH_SHA256_WASM="${SHA256_WASM}"
      BENCH_ECDSA_WASM="${ECDSA_WASM}"
   )
endif()
