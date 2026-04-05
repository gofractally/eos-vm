// Standalone ECDSA benchmark for WASM
// Uses micro-ecc for EC operations and a built-in SHA-256 for deterministic signing.
// Self-contained — no libc, no imports, no WASI.
//
// Exports:
//   bench_ecdsa_verify(iterations) -> i64
//   bench_ecdsa_sign(iterations) -> i64

#include <stdint.h>
#include <stddef.h>
#include "uECC.h"

// ============================================================================
// Minimal SHA-256 (needed for deterministic signing via uECC_HashContext)
// ============================================================================

static const uint32_t K256[64] = {
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

typedef struct {
   uint32_t state[8];
   uint8_t  buf[64];
   uint64_t total;
} sha256_ctx;

static void sha256_transform(sha256_ctx* ctx) {
   uint32_t W[64];
   for (int i = 0; i < 16; i++)
      W[i] = ((uint32_t)ctx->buf[i*4] << 24) | ((uint32_t)ctx->buf[i*4+1] << 16) |
             ((uint32_t)ctx->buf[i*4+2] << 8) | ((uint32_t)ctx->buf[i*4+3]);
   for (int i = 16; i < 64; i++) {
      uint32_t s0 = ROTR(W[i-15], 7) ^ ROTR(W[i-15], 18) ^ (W[i-15] >> 3);
      uint32_t s1 = ROTR(W[i-2], 17) ^ ROTR(W[i-2], 19) ^ (W[i-2] >> 10);
      W[i] = W[i-16] + s0 + W[i-7] + s1;
   }
   uint32_t a=ctx->state[0], b=ctx->state[1], c=ctx->state[2], d=ctx->state[3];
   uint32_t e=ctx->state[4], f=ctx->state[5], g=ctx->state[6], h=ctx->state[7];
   for (int i = 0; i < 64; i++) {
      uint32_t S1 = ROTR(e,6) ^ ROTR(e,11) ^ ROTR(e,25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = h + S1 + ch + K256[i] + W[i];
      uint32_t S0 = ROTR(a,2) ^ ROTR(a,13) ^ ROTR(a,22);
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = S0 + maj;
      h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
   }
   ctx->state[0]+=a; ctx->state[1]+=b; ctx->state[2]+=c; ctx->state[3]+=d;
   ctx->state[4]+=e; ctx->state[5]+=f; ctx->state[6]+=g; ctx->state[7]+=h;
}

static void sha256_init(sha256_ctx* ctx) {
   ctx->state[0]=0x6a09e667; ctx->state[1]=0xbb67ae85;
   ctx->state[2]=0x3c6ef372; ctx->state[3]=0xa54ff53a;
   ctx->state[4]=0x510e527f; ctx->state[5]=0x9b05688c;
   ctx->state[6]=0x1f83d9ab; ctx->state[7]=0x5be0cd19;
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
      out[i*4]   = (uint8_t)(ctx->state[i] >> 24);
      out[i*4+1] = (uint8_t)(ctx->state[i] >> 16);
      out[i*4+2] = (uint8_t)(ctx->state[i] >> 8);
      out[i*4+3] = (uint8_t)(ctx->state[i]);
   }
}

// ============================================================================
// uECC_HashContext adapter for SHA-256 (needed by uECC_sign_deterministic)
// ============================================================================

typedef struct {
   uECC_HashContext uECC;
   sha256_ctx       sha;
} SHA256_HashContext;

static void hmac_init(const uECC_HashContext* base) {
   SHA256_HashContext* ctx = (SHA256_HashContext*)base;
   sha256_init(&ctx->sha);
}

static void hmac_update(const uECC_HashContext* base, const uint8_t* msg, unsigned len) {
   SHA256_HashContext* ctx = (SHA256_HashContext*)base;
   sha256_update(&ctx->sha, msg, len);
}

static void hmac_finish(const uECC_HashContext* base, uint8_t* out) {
   SHA256_HashContext* ctx = (SHA256_HashContext*)base;
   sha256_final(&ctx->sha, out);
}

// ============================================================================
// Pre-computed test vector (secp256k1)
//
// We generate a keypair and signature once at startup using deterministic
// signing, then the benchmark loops verify/sign with those fixed values.
// ============================================================================

// Hard-coded private key (32 bytes) — just a test value, not secret
static const uint8_t test_privkey[32] = {
   0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
   0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
   0x0f, 0x1e, 0x2d, 0x3c, 0x4b, 0x5a, 0x69, 0x78,
   0x87, 0x96, 0xa5, 0xb4, 0xc3, 0xd2, 0xe1, 0xf0,
};

// Message hash to sign/verify (SHA-256 of "benchmark test message")
static const uint8_t test_hash[32] = {
   0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0x07, 0x18,
   0x29, 0x3a, 0x4b, 0x5c, 0x6d, 0x7e, 0x8f, 0x90,
   0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76, 0x87,
   0x98, 0xa9, 0xba, 0xcb, 0xdc, 0xed, 0xfe, 0x0f,
};

static uint8_t  g_pubkey[64];    // uncompressed public key (x,y)
static uint8_t  g_signature[64]; // pre-computed signature
static int      g_initialized = 0;

static void ensure_initialized(void) {
   if (g_initialized) return;

   uECC_Curve curve = uECC_secp256k1();

   // Derive public key from private key
   uECC_compute_public_key(test_privkey, g_pubkey, curve);

   // Generate deterministic signature
   uint8_t tmp[32 + 32 + 64]; // result_size*2 + block_size
   SHA256_HashContext hash_ctx = {
      {&hmac_init, &hmac_update, &hmac_finish, 64, 32, tmp},
      {{0}, {0}, 0}
   };
   uECC_sign_deterministic(test_privkey, test_hash, 32, &hash_ctx.uECC,
                           g_signature, curve);

   g_initialized = 1;
}

// ============================================================================
// Benchmark entry points
// ============================================================================

#ifdef __wasm__
#define WASM_EXPORT(name) __attribute__((export_name(name)))
#else
#define WASM_EXPORT(name)
#endif

WASM_EXPORT("bench_ecdsa_verify")
int64_t bench_ecdsa_verify(int32_t iterations) {
   ensure_initialized();
   uECC_Curve curve = uECC_secp256k1();

   int64_t valid_count = 0;
   for (int32_t i = 0; i < iterations; i++) {
      valid_count += uECC_verify(g_pubkey, test_hash, 32, g_signature, curve);
   }
   return valid_count;
}

WASM_EXPORT("bench_ecdsa_sign")
int64_t bench_ecdsa_sign(int32_t iterations) {
   ensure_initialized();
   uECC_Curve curve = uECC_secp256k1();

   uint8_t sig[64];
   uint8_t tmp[32 + 32 + 64];
   SHA256_HashContext hash_ctx = {
      {&hmac_init, &hmac_update, &hmac_finish, 64, 32, tmp},
      {{0}, {0}, 0}
   };

   int64_t ok_count = 0;
   for (int32_t i = 0; i < iterations; i++) {
      ok_count += uECC_sign_deterministic(test_privkey, test_hash, 32,
                                          &hash_ctx.uECC, sig, curve);
   }

   // Return count + last sig byte to prevent optimization
   return ok_count + (int64_t)sig[0];
}
