# Benchmark Results — Apple M4 Max (AArch64)

Date: 2026-04-05
Branch: arm64-jit (commit 79e0444)
Build: cmake -DCMAKE_BUILD_TYPE=Release
Runtimes: eos-vm (interpreter + JIT), wasm3 (main), WAMR 2.4.3 (fast interp), wasmtime v30.0.0, wasmer v5.0.4

## Host-call overhead (10M iterations)

Raw times (ms):

| Test             | eos-vm interp | eos-vm JIT | wasm3  | WAMR   | wasmtime | wasmer  |
|------------------|---------------|------------|--------|--------|----------|---------|
| identity         |         263.5 |       50.9 |   61.2 |  142.7 |    207.7 |   644.9 |
| accumulate       |         202.2 |       47.7 |   38.3 |  129.8 |    153.0 |   441.6 |
| mix              |         263.9 |       71.8 |   63.1 |  156.1 |    258.3 |   608.1 |
| mem_op           |         249.4 |       52.4 |   41.5 |  139.3 |    186.3 |   379.8 |
| multi (4 args)   |         329.9 |       64.7 |   49.9 |  172.4 |    324.7 |   591.8 |
| mixed (5 calls)  |         946.0 |      261.5 |  194.3 |  724.4 |   1236.0 |  2658.0 |

Relative to fastest:

| Test             | eos-vm interp | eos-vm JIT | wasm3 | WAMR | wasmtime | wasmer |
|------------------|---------------|------------|-------|------|----------|--------|
| identity         |          5.2x |       1.0x |  1.2x | 2.8x |     4.1x |  12.7x |
| accumulate       |          5.3x |       1.2x |  1.0x | 3.4x |     4.0x |  11.5x |
| mix              |          4.2x |       1.1x |  1.0x | 2.5x |     4.1x |   9.6x |
| mem_op           |          6.0x |       1.3x |  1.0x | 3.4x |     4.5x |   9.2x |
| multi (4 args)   |          6.6x |       1.3x |  1.0x | 3.5x |     6.5x |  11.9x |
| mixed (5 calls)  |          4.9x |       1.3x |  1.0x | 3.7x |     6.4x |  13.7x |

## Pure computation (no host calls)

Raw times (ms):

| Test           | eos-vm interp | eos-vm JIT | wasm3  | WAMR   | wasmtime | wasmer |
|----------------|---------------|------------|--------|--------|----------|--------|
| SHA-256 (100K) |        4554.0 |      332.6 |  455.9 |  606.2 |     57.0 |   56.4 |
| ECDSA verify   |        3637.4 |      318.4 |  444.2 |  516.6 |     52.8 |   46.1 |
| ECDSA sign     |        3671.4 |      316.0 |  450.7 |  527.8 |     52.2 |   45.5 |

Relative to fastest:

| Test           | eos-vm interp | eos-vm JIT | wasm3 | WAMR  | wasmtime | wasmer |
|----------------|---------------|------------|-------|-------|----------|--------|
| SHA-256 (100K) |         80.7x |       5.9x |  8.1x | 10.7x |     1.0x |   1.0x |
| ECDSA verify   |         81.1x |       6.9x |  9.6x | 11.2x |     1.1x |   1.0x |
| ECDSA sign     |         82.9x |       7.0x |  9.9x | 11.6x |     1.1x |   1.0x |

## Analysis

- **Host calls**: eos-vm JIT and wasm3 dominate. wasm3 edges out JIT by ~20-30% due to
  minimal trampoline overhead (threaded-code interpreter reads args directly from value stack).
- **Pure compute**: wasmtime and wasmer (optimizing JIT compilers with Cranelift/LLVM backends)
  are ~6-7x faster than eos-vm JIT, which uses softfloat and a single-pass code generator.
- **eos-vm JIT sweet spot**: fastest host-call transition of any JIT runtime. For host-call-heavy
  workloads (like psibase), it outperforms even wasmtime/wasmer despite slower raw computation.
