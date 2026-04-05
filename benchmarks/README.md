# EOS-VM Comparative Benchmarks

Compares WASM runtime performance across two dimensions:

1. **Host-call overhead** — the dominant cost in blockchain-style workloads where WASM frequently calls back into native host functions
2. **Pure computation** — CPU-bound workloads like cryptographic operations (SHA-256, ECDSA) running entirely in WASM

## Quick Start

```bash
# Build with eos-vm only (no competitors)
mkdir build && cd build
cmake -G Ninja .. -DENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
  -DWASI_SDK_PREFIX=/path/to/wasi-sdk
ninja bench-compare
./benchmarks/bench-compare
```

The `WASI_SDK_PREFIX` is needed for compile-to-WASM compute benchmarks. If omitted, only host-call benchmarks run.

## Enabling Competitor Runtimes

Each competitor is independently enabled. They are fetched/downloaded automatically at configure time.

```bash
cmake -G Ninja .. \
  -DENABLE_BENCHMARKS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_BENCH_WASM3=ON \
  -DENABLE_BENCH_WASMTIME=ON \
  -DWASI_SDK_PREFIX=/path/to/wasi-sdk
```

### Available options

| Option | Runtime | Type | Notes |
|--------|---------|------|-------|
| `ENABLE_BENCH_WASM3` | [wasm3](https://github.com/wasm3/wasm3) | C interpreter | Fetched via git, compiled from source |
| `ENABLE_BENCH_WAMR` | [WAMR](https://github.com/bytecodealliance/wasm-micro-runtime) | C interp+AOT | Fetched via git, compiled from source |
| `ENABLE_BENCH_WASMTIME` | [wasmtime](https://github.com/bytecodealliance/wasmtime) | Rust JIT | Prebuilt C API downloaded |
| `ENABLE_BENCH_WASMER` | [wasmer](https://github.com/wasmerio/wasmer) | Rust JIT | Prebuilt C API downloaded |
| `ENABLE_BENCH_COMPUTE` | — | — | Compile SHA-256 & ECDSA WASM modules (default ON, requires wasi-sdk) |

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
WASI=/path/to/wasi-sdk

# --- wasm3 + wasmtime ---
mkdir -p /tmp/bench-wt && cd /tmp/bench-wt
cmake -G Ninja "$SRC" -DENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_BENCH_WASM3=ON -DENABLE_BENCH_WASMTIME=ON -DWASI_SDK_PREFIX="$WASI"
ninja bench-compare && ./benchmarks/bench-compare

# --- wasm3 + WAMR ---
mkdir -p /tmp/bench-wamr && cd /tmp/bench-wamr
cmake -G Ninja "$SRC" -DENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_BENCH_WASM3=ON -DENABLE_BENCH_WAMR=ON -DWASI_SDK_PREFIX="$WASI"
ninja bench-compare && ./benchmarks/bench-compare

# --- wasm3 + wasmer ---
mkdir -p /tmp/bench-ws && cd /tmp/bench-ws
cmake -G Ninja "$SRC" -DENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_BENCH_WASM3=ON -DENABLE_BENCH_WASMER=ON -DWASI_SDK_PREFIX="$WASI"
ninja bench-compare && ./benchmarks/bench-compare
```

## What It Measures

### Host-call benchmarks

Six functions, each looping 10M iterations:

| Test | Host calls/iter | Pattern |
|------|----------------|---------|
| `identity` | 1 | `(i32) -> i32` — pure call overhead |
| `accumulate` | 1 | `(i64) -> void` — fire-and-forget |
| `mix` | 1 | `(i64, i64) -> i64` — two-arg hash |
| `mem_op` | 1 | `(i32, i32) -> void` — simulates `(ptr, len)` span pattern |
| `multi` | 1 | `(i64, i64, i32, i32) -> i64` — 4-arg call |
| `mixed` | 5 | All of the above per iteration — most realistic |

The **mixed** test is most representative of real blockchain workloads where each transaction involves many diverse host calls.

### Compute benchmarks

Pure WASM computation with zero host calls. Compiled from C using wasi-sdk.

| Test | Iterations | What it does |
|------|-----------|--------------|
| SHA-256 | 100K | Hash a 64-byte buffer, chain output as next input |
| ECDSA verify | 100 | Verify a secp256k1 signature (uses [micro-ecc](https://github.com/kmackay/micro-ecc)) |
| ECDSA sign | 100 | Deterministic ECDSA signing (RFC 6979) on secp256k1 |

These represent real psibase workloads — transaction hashing and signature verification happen in WASM.

## Platform Support

- **macOS** (x86_64, arm64): All runtimes supported
- **Linux** (x86_64, aarch64): All runtimes supported

On x86_64, eos-vm results include both interpreter and JIT columns.
