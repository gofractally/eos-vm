# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EOS VM is a high-performance, deterministic WebAssembly engine designed for blockchain use (Antelope/psibase). It's a C++20 header-only library (except for the softfloat dependency). It provides three backends: **interpreter** (default), **JIT** (x86_64 only), and **null_backend** (validation only).

## Build Commands

```bash
# Basic build
mkdir build && cd build
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release
ninja

# Build with tests
cmake -G Ninja .. -DENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Release
ninja

# Run all tests
ctest -j$(nproc)

# Run a single test (Catch2-based)
./tests/unit_tests "[allocator]"          # by tag
./tests/unit_tests "test name substring"  # by name
./tests/eos_vm_spec_tests "[i32]"         # spec tests

# Build with benchmarks
cmake -G Ninja .. -DENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
  -DWASI_SDK_PREFIX=/path/to/wasi-sdk
ninja bench-compare
./benchmarks/bench-compare

# Benchmark with competitor runtimes (only one of WAMR/wasmtime/wasmer at a time)
cmake -G Ninja .. -DENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_BENCH_WASM3=ON -DENABLE_BENCH_WASMTIME=ON \
  -DWASI_SDK_PREFIX=/path/to/wasi-sdk
```

### Key CMake Options

| Option | Default | Purpose |
|--------|---------|---------|
| `ENABLE_SOFTFLOAT` | ON | Deterministic software floating point |
| `ENABLE_TESTS` | OFF | Unit tests and spec tests |
| `ENABLE_SPEC_TESTS` | ON (if tests) | WebAssembly spec compliance tests |
| `ENABLE_FUZZ_TESTS` | OFF | Fuzz testing |
| `ENABLE_TOOLS` | ON | CLI tools (interp, bench-interp, hello-driver) |
| `ENABLE_BENCHMARKS` | OFF | Comparative benchmark suite |
| `FULL_DEBUG_BUILD` | OFF | Stack dumping and instruction tracing |

## Architecture

### Core Components (all in `include/eosio/vm/`)

- **`backend.hpp`** — Main entry point. Templated on host function class and backend type (`interpreter`, `jit`, `null_backend`). Owns module parsing, instantiation, and execution.
- **`parser.hpp`** — Binary WASM parser (~2700 lines). Parses all standard sections into the `module` struct defined in `types.hpp`.
- **`execution_context.hpp`** — Manages operand stack, call stack, linear memory, globals, and control flow during execution.
- **`interpret_visitor.hpp`** — Interpreter implementation (~3000 lines). Visitor pattern handler for every WASM opcode.
- **`x86_64.hpp`** — JIT code generator for x86_64 (~207KB). Emits native machine code from parsed WASM.
- **`bitcode_writer.hpp`** — Converts parsed WASM into an intermediate bitcode representation used by the interpreter.

### Host Function Integration

- **`host_function.hpp`** — Type-safe binding of native C++ functions as WASM imports via template metaprogramming. Supports C-style functions, static methods, and class methods.
- **`type_converter.hpp`** — Automatic conversion between WASM types and native C++ types.
- See `tools/hello_driver.cpp` for a working integration example.

### Memory & Safety

- **`allocator.hpp`** — Multiple allocator strategies: `bounded_allocator`, `contiguous_allocator`, `jit_allocator`, `growable_allocator`. Uses OS guard pages for memory sandboxing.
- **`guarded_ptr.hpp`** — Pointer validation via guard pages (no per-access runtime checks).
- **`softfloat.hpp`** — Deterministic IEEE-754 wrapper over berkeley-softfloat-3 for cross-platform reproducibility.
- **`watchdog.hpp`** — Time-bounded execution via signal-based timer (no per-instruction overhead).

### Key Design Patterns

- **Static dispatch visitor**: `variant.hpp` implements a discriminating union with compile-time dispatch (no vtable overhead). Used for opcodes and stack elements.
- **Non-owning data structures**: Core types (fast vector, stack, variant) don't own memory — allocators manage lifetimes matching the WASM module lifetime. No copies, no destructors on scope exit.
- **Guard paging over bounds checks**: Memory safety enforced by OS page protection rather than runtime bounds checking.
- **No unbounded recursion or loops**: All parsing and execution paths are tightly bounded.

### Test Structure

- **Unit tests** (`tests/`): Catch2-based. Cover allocators, host functions, parsing, implementation limits, signals, watchdog, reentry.
- **Spec tests** (`tests/spec/`): Auto-generated from WebAssembly spec `.wast` files via `spec_test_generator`. Requires `wast2json` tool to regenerate.
- **SIMD tests**: Separate executable (`eos_vm_simd_tests`) for v128/SIMD spec compliance.
- **Benchmarks** (`benchmarks/`): Comparative suite measuring host-call overhead and pure compute (SHA-256, ECDSA) against wasm3, WAMR, wasmtime, and wasmer.

### External Dependencies

- **softfloat** (`external/`): Berkeley SoftFloat-3 (AntelopeIO fork) — deterministic float ops.
- **Catch2** (`external/Catch2`): Test framework (single-header, only needed with `ENABLE_TESTS`).
