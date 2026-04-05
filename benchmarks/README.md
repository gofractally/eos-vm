# EOS-VM Comparative Benchmarks

Measures host function call overhead across WASM runtimes. This is the dominant cost in blockchain-style workloads (like psibase) where WASM code frequently calls back into native host functions.

## Quick Start

```bash
# Build with eos-vm only (no competitors)
mkdir build && cd build
cmake -G Ninja .. -DENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
ninja bench-compare
./benchmarks/bench-compare
```

## Enabling Competitor Runtimes

Each competitor is independently enabled. They are fetched/downloaded automatically at configure time.

```bash
cmake -G Ninja .. \
  -DENABLE_BENCHMARKS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_BENCH_WASM3=ON \
  -DENABLE_BENCH_WASMTIME=ON
```

### Available options

| Option | Runtime | Type | Notes |
|--------|---------|------|-------|
| `ENABLE_BENCH_WASM3` | [wasm3](https://github.com/wasm3/wasm3) | C interpreter | Fetched via git, compiled from source |
| `ENABLE_BENCH_WAMR` | [WAMR](https://github.com/bytecodealliance/wasm-micro-runtime) | C interp+AOT | Fetched via git, compiled from source |
| `ENABLE_BENCH_WASMTIME` | [wasmtime](https://github.com/bytecodealliance/wasmtime) | Rust JIT | Prebuilt C API downloaded |
| `ENABLE_BENCH_WASMER` | [wasmer](https://github.com/wasmerio/wasmer) | Rust JIT | Prebuilt C API downloaded |

### Compatibility

WAMR, wasmtime, and wasmer all export standard [wasm-c-api](https://github.com/WebAssembly/wasm-c-api) symbols, so **only one of these three** can be enabled at a time. wasm3 uses its own API and can be combined with any single one.

Valid combinations:
- `WASM3` alone
- `WASM3 + WASMTIME`
- `WASM3 + WAMR`
- `WASM3 + WASMER`
- `WASMTIME` alone
- `WAMR` alone
- `WASMER` alone

To compare all runtimes, run the benchmark multiple times with different configurations.

## Running All Comparisons

```bash
# From the repo root:
SRC=$(pwd)

# --- wasm3 + wasmtime ---
mkdir -p /tmp/bench-wt && cd /tmp/bench-wt
cmake -G Ninja "$SRC" -DENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_BENCH_WASM3=ON -DENABLE_BENCH_WASMTIME=ON
ninja bench-compare && ./benchmarks/bench-compare

# --- wasm3 + WAMR ---
mkdir -p /tmp/bench-wamr && cd /tmp/bench-wamr
cmake -G Ninja "$SRC" -DENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_BENCH_WASM3=ON -DENABLE_BENCH_WAMR=ON
ninja bench-compare && ./benchmarks/bench-compare

# --- wasm3 + wasmer ---
mkdir -p /tmp/bench-ws && cd /tmp/bench-ws
cmake -G Ninja "$SRC" -DENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_BENCH_WASM3=ON -DENABLE_BENCH_WASMER=ON
ninja bench-compare && ./benchmarks/bench-compare
```

## What It Measures

Six benchmark functions, each looping 10M iterations:

| Test | Host calls/iter | Pattern |
|------|----------------|---------|
| `identity` | 1 | `(i32) -> i32` — pure call overhead |
| `accumulate` | 1 | `(i64) -> void` — fire-and-forget |
| `mix` | 1 | `(i64, i64) -> i64` — two-arg hash |
| `mem_op` | 1 | `(i32, i32) -> void` — simulates `(ptr, len)` span pattern |
| `multi` | 1 | `(i64, i64, i32, i32) -> i64` — 4-arg call |
| `mixed` | 5 | All of the above per iteration — most realistic |

The **mixed** test is most representative of real blockchain workloads where each transaction involves many diverse host calls.

## Platform Support

- **macOS** (x86_64, arm64): All runtimes supported
- **Linux** (x86_64, aarch64): All runtimes supported

On x86_64, eos-vm results include both interpreter and JIT columns.
