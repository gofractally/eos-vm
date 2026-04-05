#pragma once
#include <cstdint>
#include <vector>

// Shared host functions used by all runtime benchmarks
int32_t bench_host_identity(int32_t x);
void bench_host_accumulate(int64_t val);
int64_t bench_host_mix(int64_t a, int64_t b);
void bench_host_mem_op(int32_t ptr, int32_t len);
int64_t bench_host_multi(int64_t a, int64_t b, int32_t c, int32_t d);

// Build the benchmark WASM module
std::vector<uint8_t> build_bench_wasm();

// Runner function type: (wasm_bytes, func_name, iteration_count) -> milliseconds
using bench_runner_t = double (*)(const std::vector<uint8_t>&, const char*, uint32_t);

// Optional runners provided by separate TUs (host-call benchmarks)
#ifdef BENCH_HAS_WASMTIME
double run_wasmtime(const std::vector<uint8_t>& wasm, const char* func, uint32_t n);
#endif

#ifdef BENCH_HAS_WASMER
double run_wasmer(const std::vector<uint8_t>& wasm, const char* func, uint32_t n);
#endif

// Compute benchmark runners (zero-import WASM modules)
#ifdef BENCH_HAS_WASMTIME
double run_wasmtime_compute(const std::vector<uint8_t>& wasm, const char* func, uint32_t n);
#endif

#ifdef BENCH_HAS_WASMER
double run_wasmer_compute(const std::vector<uint8_t>& wasm, const char* func, uint32_t n);
#endif
