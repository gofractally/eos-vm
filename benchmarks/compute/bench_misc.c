// Diverse compute benchmarks for jit2 optimization testing.
// Each function takes an iteration count and returns a result for verification.

#include <stdint.h>

#define EXPORT __attribute__((visibility("default")))

// 1. Integer loop: iterative fibonacci (tests loop overhead, branch prediction)
EXPORT int64_t bench_fib(int32_t n) {
   int64_t a = 0, b = 1;
   for (int32_t i = 0; i < n; i++) {
      int64_t t = a + b;
      a = b;
      b = t;
   }
   return a;
}

// 2. Memory-heavy: bubble sort a small array (tests load/store patterns)
EXPORT int64_t bench_sort(int32_t iterations) {
   int32_t arr[64];
   int64_t checksum = 0;
   for (int32_t iter = 0; iter < iterations; iter++) {
      // Initialize array with pseudo-random values
      for (int i = 0; i < 64; i++)
         arr[i] = (i * 2654435761u + iter) & 0xFFFF;
      // Bubble sort
      for (int i = 0; i < 63; i++)
         for (int j = 0; j < 63 - i; j++)
            if (arr[j] > arr[j + 1]) {
               int32_t tmp = arr[j];
               arr[j] = arr[j + 1];
               arr[j + 1] = tmp;
            }
      checksum += arr[0] + arr[63];
   }
   return checksum;
}

// 3. Bitwise-heavy: CRC32 (tests shift/xor patterns, common in blockchain)
static uint32_t crc32_table[256];
static void crc32_init(void) {
   for (uint32_t i = 0; i < 256; i++) {
      uint32_t c = i;
      for (int j = 0; j < 8; j++)
         c = (c >> 1) ^ (0xEDB88320 & (-(c & 1)));
      crc32_table[i] = c;
   }
}

EXPORT int64_t bench_crc32(int32_t iterations) {
   crc32_init();
   uint32_t crc = 0xFFFFFFFF;
   // CRC32 over a synthetic data stream
   for (int32_t iter = 0; iter < iterations; iter++) {
      for (int i = 0; i < 256; i++) {
         uint8_t byte = (uint8_t)((iter * 251 + i * 37) & 0xFF);
         crc = (crc >> 8) ^ crc32_table[(crc ^ byte) & 0xFF];
      }
   }
   return (int64_t)(crc ^ 0xFFFFFFFF);
}

// 4. Mixed integer: matrix multiply 8x8 (tests nested loops, register pressure)
//    Scalar version for baseline comparison.
EXPORT int64_t bench_matmul(int32_t iterations) {
   int32_t A[64], B[64], C[64];
   int64_t checksum = 0;
   for (int32_t iter = 0; iter < iterations; iter++) {
      for (int i = 0; i < 64; i++) {
         A[i] = (i * 7 + iter) & 0xFF;
         B[i] = (i * 13 + iter) & 0xFF;
      }
      for (int i = 0; i < 8; i++)
         for (int j = 0; j < 8; j++) {
            int32_t sum = 0;
            for (int k = 0; k < 8; k++)
               sum += A[i * 8 + k] * B[k * 8 + j];
            C[i * 8 + j] = sum;
         }
      checksum += C[0] + C[63];
   }
   return checksum;
}

// 5. Floating point: Mandelbrot set iteration count (tests f64 mul/add/cmp)
EXPORT int64_t bench_mandelbrot(int32_t iterations) {
   int64_t total = 0;
   double step = 3.0 / (double)iterations;
   for (int32_t iy = 0; iy < iterations; iy++) {
      double ci = -1.5 + (double)iy * step;
      for (int32_t ix = 0; ix < iterations; ix++) {
         double cr = -2.0 + (double)ix * step;
         double zr = 0.0, zi = 0.0;
         int32_t count = 0;
         for (count = 0; count < 50; count++) {
            double zr2 = zr * zr;
            double zi2 = zi * zi;
            if (zr2 + zi2 > 4.0) break;
            zi = 2.0 * zr * zi + ci;
            zr = zr2 - zi2 + cr;
         }
         total += count;
      }
   }
   return total;
}

// 6. Float math: Nbody gravitational simulation (tests f64 sqrt, mul, add, div)
EXPORT int64_t bench_nbody(int32_t iterations) {
   // 8 bodies with position (x,y,z) and velocity (vx,vy,vz) and mass
   double x[8], y[8], z[8], vx[8], vy[8], vz[8], mass[8];
   // Initialize with deterministic values
   for (int i = 0; i < 8; i++) {
      x[i]  = (double)(i * 17 % 11) - 5.0;
      y[i]  = (double)(i * 31 % 13) - 6.0;
      z[i]  = (double)(i * 7 % 9) - 4.0;
      vx[i] = 0.0; vy[i] = 0.0; vz[i] = 0.0;
      mass[i] = 1.0 + (double)(i % 3);
   }
   double dt = 0.01;
   for (int32_t iter = 0; iter < iterations; iter++) {
      // Compute forces (all pairs)
      for (int i = 0; i < 8; i++) {
         for (int j = i + 1; j < 8; j++) {
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double dz = z[j] - z[i];
            double dist2 = dx*dx + dy*dy + dz*dz + 0.01; // softening
            // Fast inverse sqrt approximation: 1/sqrt(dist2) via Newton
            double inv = 1.0 / dist2; // use 1/r^2 instead of sqrt for speed
            double f = inv * dt;
            double fx = dx * f, fy = dy * f, fz = dz * f;
            vx[i] += fx * mass[j]; vy[i] += fy * mass[j]; vz[i] += fz * mass[j];
            vx[j] -= fx * mass[i]; vy[j] -= fy * mass[i]; vz[j] -= fz * mass[i];
         }
      }
      // Integrate positions
      for (int i = 0; i < 8; i++) {
         x[i] += vx[i] * dt;
         y[i] += vy[i] * dt;
         z[i] += vz[i] * dt;
      }
   }
   // Return checksum as fixed-point
   double sum = 0.0;
   for (int i = 0; i < 8; i++)
      sum += x[i] + y[i] + z[i];
   return (int64_t)(sum * 1000000.0);
}

// 7. Float32: Dot product of large f32 arrays (tests f32 mul+add throughput)
EXPORT int64_t bench_fdot(int32_t iterations) {
   float A[256], B[256];
   for (int i = 0; i < 256; i++) {
      A[i] = (float)(i * 7 + 1) * 0.001f;
      B[i] = (float)(i * 13 + 3) * 0.001f;
   }
   float total = 0.0f;
   for (int32_t iter = 0; iter < iterations; iter++) {
      float dot = 0.0f;
      for (int i = 0; i < 256; i++)
         dot += A[i] * B[i];
      total += dot;
      // Perturb slightly to prevent optimization
      A[iter & 255] += 0.0001f;
   }
   return (int64_t)(total * 1000.0f);
}

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

// 5. SIMD matrix multiply 8x8 using v128 intrinsics
//    Each row of A is loaded as two v128 (4 i32 each).
//    B is transposed so columns become contiguous rows.
//    Dot product via i32x4 multiply + horizontal add.
EXPORT int64_t bench_matmul_simd(int32_t iterations) {
#ifdef __wasm_simd128__
   int32_t A[64], B[64], BT[64], C[64];
   int64_t checksum = 0;
   for (int32_t iter = 0; iter < iterations; iter++) {
      for (int i = 0; i < 64; i++) {
         A[i] = (i * 7 + iter) & 0xFF;
         B[i] = (i * 13 + iter) & 0xFF;
      }
      // Transpose B so column j becomes row j in BT
      for (int i = 0; i < 8; i++)
         for (int j = 0; j < 8; j++)
            BT[j * 8 + i] = B[i * 8 + j];
      // SIMD matmul: dot product of A rows with BT rows
      for (int i = 0; i < 8; i++) {
         v128_t a_lo = wasm_v128_load(&A[i * 8]);
         v128_t a_hi = wasm_v128_load(&A[i * 8 + 4]);
         for (int j = 0; j < 8; j++) {
            v128_t b_lo = wasm_v128_load(&BT[j * 8]);
            v128_t b_hi = wasm_v128_load(&BT[j * 8 + 4]);
            // Multiply and accumulate: sum of element-wise products
            v128_t prod_lo = wasm_i32x4_mul(a_lo, b_lo);
            v128_t prod_hi = wasm_i32x4_mul(a_hi, b_hi);
            v128_t sum = wasm_i32x4_add(prod_lo, prod_hi);
            // Horizontal sum of 4 lanes
            // sum = [s0, s1, s2, s3]
            // shuffle to get [s2, s3, s0, s1] and add
            v128_t shuf1 = wasm_i32x4_shuffle(sum, sum, 2, 3, 0, 1);
            v128_t sum2 = wasm_i32x4_add(sum, shuf1);
            // shuffle to get [s1, s0, s3, s2] and add
            v128_t shuf2 = wasm_i32x4_shuffle(sum2, sum2, 1, 0, 3, 2);
            v128_t sum4 = wasm_i32x4_add(sum2, shuf2);
            C[i * 8 + j] = wasm_i32x4_extract_lane(sum4, 0);
         }
      }
      checksum += C[0] + C[63];
   }
   return checksum;
#else
   return bench_matmul(iterations);
#endif
}
