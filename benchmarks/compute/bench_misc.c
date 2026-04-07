// Diverse compute benchmarks for jit2 optimization testing.
// Each function takes an iteration count and returns a result for verification.

#include <stdint.h>

// 1. Integer loop: iterative fibonacci (tests loop overhead, branch prediction)
int64_t bench_fib(int32_t n) {
   int64_t a = 0, b = 1;
   for (int32_t i = 0; i < n; i++) {
      int64_t t = a + b;
      a = b;
      b = t;
   }
   return a;
}

// 2. Memory-heavy: bubble sort a small array (tests load/store patterns)
int64_t bench_sort(int32_t iterations) {
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

int64_t bench_crc32(int32_t iterations) {
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
int64_t bench_matmul(int32_t iterations) {
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
