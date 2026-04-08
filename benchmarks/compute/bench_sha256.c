// Standalone SHA-256 benchmark for WASM
// Self-contained implementation — no libc, no imports, no WASI.
// Exports: bench_sha256(iterations) -> i64 (returns final hash word as checksum)

#include <stdint.h>
#include <stddef.h>

// ============================================================================
// SHA-256 implementation (FIPS 180-4)
// ============================================================================

static const uint32_t K[64] = {
   0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
   0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
   0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
   0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
   0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
   0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
   0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
   0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
   0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
   0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
   0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
   0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
   0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
   0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
   0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
   0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

typedef struct {
   uint32_t state[8];
   uint8_t  buf[64];
   uint64_t total;
} sha256_ctx;

static void sha256_transform(sha256_ctx* ctx) {
   uint32_t W[64];
   for (int i = 0; i < 16; i++) {
      W[i] = ((uint32_t)ctx->buf[i * 4] << 24) |
             ((uint32_t)ctx->buf[i * 4 + 1] << 16) |
             ((uint32_t)ctx->buf[i * 4 + 2] << 8) |
             ((uint32_t)ctx->buf[i * 4 + 3]);
   }
   for (int i = 16; i < 64; i++)
      W[i] = SIG1(W[i - 2]) + W[i - 7] + SIG0(W[i - 15]) + W[i - 16];

   uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
   uint32_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];

   for (int i = 0; i < 64; i++) {
      uint32_t t1 = h + EP1(e) + CH(e, f, g) + K[i] + W[i];
      uint32_t t2 = EP0(a) + MAJ(a, b, c);
      h = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
   }

   ctx->state[0] += a; ctx->state[1] += b;
   ctx->state[2] += c; ctx->state[3] += d;
   ctx->state[4] += e; ctx->state[5] += f;
   ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(sha256_ctx* ctx) {
   ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85;
   ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
   ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c;
   ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
   ctx->total = 0;
}

static void sha256_update(sha256_ctx* ctx, const uint8_t* data, size_t len) {
   for (size_t i = 0; i < len; i++) {
      ctx->buf[ctx->total & 63] = data[i];
      ctx->total++;
      if ((ctx->total & 63) == 0)
         sha256_transform(ctx);
   }
}

static void sha256_final(sha256_ctx* ctx, uint8_t out[32]) {
   uint64_t bits = ctx->total * 8;
   uint8_t pad = 0x80;
   sha256_update(ctx, &pad, 1);
   pad = 0;
   while ((ctx->total & 63) != 56)
      sha256_update(ctx, &pad, 1);
   for (int i = 7; i >= 0; i--) {
      uint8_t b = (uint8_t)(bits >> (i * 8));
      sha256_update(ctx, &b, 1);
   }
   for (int i = 0; i < 8; i++) {
      out[i * 4]     = (uint8_t)(ctx->state[i] >> 24);
      out[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
      out[i * 4 + 2] = (uint8_t)(ctx->state[i] >> 8);
      out[i * 4 + 3] = (uint8_t)(ctx->state[i]);
   }
}

// ============================================================================
// SIMD SHA-256: explicitly vectorized message schedule
// ============================================================================

#ifdef __wasm_simd128__
#include <wasm_simd128.h>

// SIMD rotate right for 4 x i32
static inline v128_t v_rotr(v128_t x, int n) {
   return wasm_v128_or(wasm_u32x4_shr(x, n), wasm_i32x4_shl(x, 32 - n));
}

// SIMD sigma0: ROTR(x,7) ^ ROTR(x,18) ^ SHR(x,3)  — for 4 words at once
static inline v128_t v_sig0(v128_t x) {
   return wasm_v128_xor(wasm_v128_xor(v_rotr(x, 7), v_rotr(x, 18)),
                        wasm_u32x4_shr(x, 3));
}

// SIMD sigma1: ROTR(x,17) ^ ROTR(x,19) ^ SHR(x,10) — for 4 words at once
static inline v128_t v_sig1(v128_t x) {
   return wasm_v128_xor(wasm_v128_xor(v_rotr(x, 17), v_rotr(x, 19)),
                        wasm_u32x4_shr(x, 10));
}

// Byte-swap 4 x i32 from little-endian to big-endian using SIMD shuffle
static inline v128_t v_bswap32(v128_t x) {
   return wasm_i8x16_swizzle(x,
      wasm_i8x16_const(3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12));
}

static void sha256_transform_simd(sha256_ctx* ctx) {
   uint32_t W[64];

   // Load and byte-swap 16 input words using SIMD (4 at a time)
   v128_t w0 = v_bswap32(wasm_v128_load(&ctx->buf[0]));
   v128_t w1 = v_bswap32(wasm_v128_load(&ctx->buf[16]));
   v128_t w2 = v_bswap32(wasm_v128_load(&ctx->buf[32]));
   v128_t w3 = v_bswap32(wasm_v128_load(&ctx->buf[48]));
   wasm_v128_store(&W[0],  w0);
   wasm_v128_store(&W[4],  w1);
   wasm_v128_store(&W[8],  w2);
   wasm_v128_store(&W[12], w3);

   // Message schedule: W[t] = sig1(W[t-2]) + W[t-7] + sig0(W[t-15]) + W[t-16]
   // Process 4 at a time where dependencies allow.
   // W[16..19] depend on W[14..17] (sig1) — W[16,17] can use W[14,15] but
   // W[18,19] depend on W[16,17]. So we do 2 at a time for sig1.
   for (int i = 16; i < 64; i += 4) {
      // sig0 of W[i-15..i-12] — 4 independent values
      v128_t s0 = v_sig0(wasm_v128_load(&W[i - 15]));
      // W[i-16..i-13] + W[i-7..i-4]
      v128_t base = wasm_i32x4_add(wasm_v128_load(&W[i - 16]),
                                    wasm_v128_load(&W[i - 7]));
      v128_t partial = wasm_i32x4_add(base, s0);

      // sig1 depends on W[i-2], which for i+2,i+3 is W[i],W[i+1] — just computed.
      // Do first two scalarly, then SIMD the pair if possible.
      uint32_t p0 = wasm_i32x4_extract_lane(partial, 0);
      uint32_t p1 = wasm_i32x4_extract_lane(partial, 1);
      uint32_t p2 = wasm_i32x4_extract_lane(partial, 2);
      uint32_t p3 = wasm_i32x4_extract_lane(partial, 3);

      W[i]   = p0 + SIG1(W[i - 2]);
      W[i+1] = p1 + SIG1(W[i - 1]);
      W[i+2] = p2 + SIG1(W[i]);
      W[i+3] = p3 + SIG1(W[i+1]);
   }

   // Compression rounds — inherently sequential
   uint32_t a = ctx->state[0], b = ctx->state[1], c = ctx->state[2], d = ctx->state[3];
   uint32_t e = ctx->state[4], f = ctx->state[5], g = ctx->state[6], h = ctx->state[7];

   for (int i = 0; i < 64; i++) {
      uint32_t t1 = h + EP1(e) + CH(e, f, g) + K[i] + W[i];
      uint32_t t2 = EP0(a) + MAJ(a, b, c);
      h = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
   }

   ctx->state[0] += a; ctx->state[1] += b;
   ctx->state[2] += c; ctx->state[3] += d;
   ctx->state[4] += e; ctx->state[5] += f;
   ctx->state[6] += g; ctx->state[7] += h;
}
#endif // __wasm_simd128__

// ============================================================================
// Benchmark entry points
// ============================================================================

#ifdef __wasm__
#define WASM_EXPORT(name) __attribute__((export_name(name)))
#else
#define WASM_EXPORT(name)
#endif

static int64_t sha256_bench_common(int32_t iterations, void (*transform)(sha256_ctx*)) {
   uint8_t data[64];
   for (int i = 0; i < 64; i++)
      data[i] = (uint8_t)i;

   sha256_ctx ctx;
   uint8_t hash[32];

   for (int32_t iter = 0; iter < iterations; iter++) {
      sha256_init(&ctx);
      // Inline the 64-byte update — write directly to buf, call transform
      for (int i = 0; i < 64; i++)
         ctx.buf[i] = data[i];
      ctx.total = 64;
      transform(&ctx);
      // Finalize: pad + length
      sha256_init(&ctx);  // reuse for simplicity
      sha256_update(&ctx, data, 64);
      sha256_final(&ctx, hash);
      for (int i = 0; i < 32; i++)
         data[i] = hash[i];
   }

   return (int64_t)(
      ((uint64_t)hash[0] << 56) | ((uint64_t)hash[1] << 48) |
      ((uint64_t)hash[2] << 40) | ((uint64_t)hash[3] << 32) |
      ((uint64_t)hash[4] << 24) | ((uint64_t)hash[5] << 16) |
      ((uint64_t)hash[6] << 8)  | ((uint64_t)hash[7])
   );
}

// Original scalar benchmark
WASM_EXPORT("bench_sha256")
int64_t bench_sha256(int32_t iterations) {
   uint8_t data[64];
   for (int i = 0; i < 64; i++)
      data[i] = (uint8_t)i;

   sha256_ctx ctx;
   uint8_t hash[32];

   for (int32_t iter = 0; iter < iterations; iter++) {
      sha256_init(&ctx);
      sha256_update(&ctx, data, 64);
      sha256_final(&ctx, hash);
      for (int i = 0; i < 32; i++)
         data[i] = hash[i];
   }

   return (int64_t)(
      ((uint64_t)hash[0] << 56) | ((uint64_t)hash[1] << 48) |
      ((uint64_t)hash[2] << 40) | ((uint64_t)hash[3] << 32) |
      ((uint64_t)hash[4] << 24) | ((uint64_t)hash[5] << 16) |
      ((uint64_t)hash[6] << 8)  | ((uint64_t)hash[7])
   );
}

// Explicit SIMD benchmark — vectorized message schedule + byte-swap
WASM_EXPORT("bench_sha256_simd")
int64_t bench_sha256_simd(int32_t iterations) {
#ifdef __wasm_simd128__
   uint8_t data[64];
   for (int i = 0; i < 64; i++)
      data[i] = (uint8_t)i;

   sha256_ctx ctx;
   uint8_t hash[32];

   for (int32_t iter = 0; iter < iterations; iter++) {
      sha256_init(&ctx);
      for (int i = 0; i < 64; i++)
         ctx.buf[i] = data[i];
      ctx.total = 64;
      sha256_transform_simd(&ctx);
      // Finalize with padding
      ctx.buf[0] = 0x80;
      for (int i = 1; i < 56; i++) ctx.buf[i] = 0;
      // Length in bits = 512 = 0x200, big-endian in last 8 bytes
      for (int i = 56; i < 62; i++) ctx.buf[i] = 0;
      ctx.buf[62] = 0x02;
      ctx.buf[63] = 0x00;
      sha256_transform_simd(&ctx);
      // Extract hash
      for (int i = 0; i < 8; i++) {
         hash[i * 4]     = (uint8_t)(ctx.state[i] >> 24);
         hash[i * 4 + 1] = (uint8_t)(ctx.state[i] >> 16);
         hash[i * 4 + 2] = (uint8_t)(ctx.state[i] >> 8);
         hash[i * 4 + 3] = (uint8_t)(ctx.state[i]);
      }
      for (int i = 0; i < 32; i++)
         data[i] = hash[i];
   }

   return (int64_t)(
      ((uint64_t)hash[0] << 56) | ((uint64_t)hash[1] << 48) |
      ((uint64_t)hash[2] << 40) | ((uint64_t)hash[3] << 32) |
      ((uint64_t)hash[4] << 24) | ((uint64_t)hash[5] << 16) |
      ((uint64_t)hash[6] << 8)  | ((uint64_t)hash[7])
   );
#else
   return bench_sha256(iterations);
#endif
}
