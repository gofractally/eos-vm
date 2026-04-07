#pragma once

namespace eosio { namespace vm {

// create constexpr flags for whether the backend should obey alignment hints
#ifdef EOS_VM_ALIGN_MEMORY_OPS
   inline constexpr bool should_align_memory_ops = true;
#else
   inline constexpr bool should_align_memory_ops = false;
#endif


#ifdef EOS_VM_SOFTFLOAT
   inline constexpr bool use_softfloat = true;
#else
   inline constexpr bool use_softfloat = false;
#endif

// When EOS_VM_NATIVE_FP is set, the JIT uses native FP instructions with
// hardware NaN canonicalization (FPCR.DN=1 on aarch64) instead of softfloat
// calls. Softfloat is still linked for interpreter SIMD float operations.
#ifdef EOS_VM_NATIVE_FP
   inline constexpr bool use_native_fp = true;
#else
   inline constexpr bool use_native_fp = false;
#endif

#ifdef EOS_VM_FULL_DEBUG
   inline constexpr bool eos_vm_debug = true;
#else
   inline constexpr bool eos_vm_debug = false;
#endif

#ifdef __x86_64__
   inline constexpr bool eos_vm_amd64 = true;
#else
   inline constexpr bool eos_vm_amd64 = false;
#endif

#ifdef __aarch64__
   inline constexpr bool eos_vm_arm64 = true;
#else
   inline constexpr bool eos_vm_arm64 = false;
#endif

}} // namespace eosio::vm
