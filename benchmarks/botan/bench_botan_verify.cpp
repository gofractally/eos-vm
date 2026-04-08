// Botan ECDSA verify benchmark — standalone wasm
// Uses pre-computed key + signature (no RNG, no WASI needed for verify path)
#include <botan/x509_key.h>
#include <botan/pubkey.h>
#include <cstdint>
#include <cstdio>
#include <span>

#include "ecdsa_test_data.h"

#ifdef __wasm__
#define WASM_EXPORT(name) extern "C" __attribute__((export_name(name)))
#else
#define WASM_EXPORT(name) extern "C"
#endif

static Botan::PK_Verifier* g_verifier = nullptr;

static void ensure_init() {
    if (g_verifier) return;
    auto pubkey = Botan::X509::load_key(
        std::span{_tmp_test_ec_pub_der, _tmp_test_ec_pub_der_len});
    g_verifier = new Botan::PK_Verifier(*pubkey, "Raw", Botan::Signature_Format::Standard);
}

WASM_EXPORT("bench_verify")
int64_t bench_verify(int32_t iterations) {
    ensure_init();
    int64_t valid = 0;
    std::span<const uint8_t> msg{_tmp_test_msg_bin, _tmp_test_msg_bin_len};
    std::span<const uint8_t> sig{_tmp_test_sig_der, _tmp_test_sig_der_len};
    for (int32_t i = 0; i < iterations; i++) {
        valid += g_verifier->verify_message(msg, sig) ? 1 : 0;
    }
    return valid;
}
