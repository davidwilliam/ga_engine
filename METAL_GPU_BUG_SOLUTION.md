# Metal GPU NTT Bug - Solution in Progress

**Date**: 2025-11-06 (Updated)
**Status**: ✅ **FIXED** - Both bugs resolved, Metal NTT working perfectly!

---

## Expert Analysis (CORRECT!)

An expert correctly identified TWO distinct issues:

### Issue #1: **Global Synchronization Bug** (PRIMARY)

**Root Cause**: `threadgroup_barrier(mem_flags::mem_device)` only synchronizes threads WITHIN a threadgroup, NOT across different threadgroups!

**Problem**: Our single-dispatch NTT kernel ran all stages in one dispatch. Different threadgroups could be at different NTT stages simultaneously, reading stale values that other groups hadn't written yet.

**Fix**: ✅ **IMPLEMENTED** - Stage-per-dispatch approach:
- 1 dispatch for bit-reversal permutation
- 1 dispatch per butterfly stage (log2(n) dispatches)
- Each dispatch completion provides implicit global barrier

**Status**: Partially working - position 0 now matches CPU output! This confirms the synchronization fix helped.

### Issue #2: **128-bit Modular Multiplication** (SECONDARY)

**Root Cause**: Our `mul_mod` implementation uses `% q` on partial 64-bit results, which loses carry information from the 128-bit product.

**Problem**: The expression `(hi % q) * (2^64 % q) + (lo % q)` is NOT equivalent to `(hi * 2^64 + lo) % q` because intermediate `%` operations don't preserve carries.

**Fix**: ✅ **IMPLEMENTED** - Montgomery multiplication:
- No 128-bit division needed
- No `%` on partial products
- Exact arithmetic using only 64×64→128 multiplications

---

## Implementation Progress

### ✅ Completed: Stage-Per-Dispatch NTT

**New Kernels** (`ntt.metal`):
```metal
kernel void ntt_bit_reverse(...)           // Separate bit-reversal pass
kernel void ntt_forward_stage(...)         // Single stage butterfly
kernel void ntt_inverse_stage(...)         // Single stage inverse butterfly
kernel void ntt_inverse_final_scale(...)   // Final scaling and bit-reverse
```

**Updated Rust Code** (`ntt.rs`):
```rust
pub fn forward(&self, coeffs: &mut [u64]) -> Result<(), String> {
    // 1. Bit-reversal (1 dispatch)
    execute_kernel("ntt_bit_reverse", ...);

    // 2. Each stage separately (log2(n) dispatches)
    for stage in 0..log_n {
        execute_kernel("ntt_forward_stage", stage, ...);
        // Implicit global barrier here!
    }
}
```

**Result**: Position 0 now matches! Synchronization bug is fixed. Remaining errors are from `mul_mod`.

### ✅ Completed: Montgomery Multiplication

**Added mont_mul Kernel** (`ntt.metal:29`):
```metal
inline ulong mont_mul(ulong a, ulong b, ulong q, ulong q_inv) {
    ulong t_lo = a * b;
    ulong t_hi = mulhi(a, b);
    ulong m = t_lo * q_inv;
    ulong mq_lo = m * q;
    ulong mq_hi = mulhi(m, q);
    ulong carry = (t_lo > ~mq_lo) ? 1UL : 0UL;
    ulong sum_hi = t_hi + mq_hi + carry;
    return (sum_hi >= q) ? (sum_hi - q) : sum_hi;
}
```

**Implementation Complete**:
1. ✅ Montgomery multiplication kernel (done)
2. ✅ Precompute `q_inv = -q^{-1} mod 2^64` on CPU
3. ✅ Convert twiddles to Montgomery domain: `twiddle_M = twiddle * R mod q`
4. ✅ Convert input coefficients to Montgomery domain (CPU side)
5. ✅ Update kernels to use `mont_mul` instead of `mul_mod`
6. ✅ Convert output back from Montgomery domain (CPU side)

**Montgomery Domain**:
- R = 2^64
- All operations stay in Montgomery space
- Input: x → x*R mod q
- Multiply: (a*R) * (b*R) * R^{-1} = (a*b)*R mod q (stays in domain)
- Output: (x*R) * 1 * R^{-1} = x mod q (leave domain)

---

## Test Results

### Before Fix
```
❌ Forward NTT: 1024 differences found
Position 0: CPU=6466014720, Metal=2623077945, diff=3842936775
Max diff = 1152921501977374778 (near modulus!)
```

### After Stage-Per-Dispatch Fix
```
Position 0: ✅ MATCH! (0 difference)
Position 1: diff=3228577135096817 (still large)
❌ Forward NTT: 1022 differences found
Max diff = 1088235719584280143
```

**Analysis**: Synchronization bug is FIXED (position 0 matches). Remaining ~1022 errors are from buggy `mul_mod`.

### ✅ After Montgomery Multiplication Fix
```
Testing Metal NTT Correctness vs CPU
═══════════════════════════════════════

Parameters: N=1024, q=1152921504606748673

Test 1: Forward NTT
✅ Forward NTT: PERFECT MATCH!

Test 2: Inverse NTT
CPU roundtrip max error: 0
Metal roundtrip max error: 0

✅ ALL TESTS PASSED - Metal NTT is CORRECT!
```

**Result**: Both bugs are now FIXED! Metal GPU NTT produces identical results to CPU.

---

## ✅ Implementation Complete

All Montgomery precomputation, domain conversion, and kernel updates are now complete:

### CPU-Side Implementation (`ntt.rs`)

1. **Montgomery precomputation functions**:
   - `compute_q_inv(q: u64) -> u64` - Extended Euclidean algorithm
   - `compute_r_squared_mod_q(q: u64) -> u64` - Computes R^2 mod q
   - `to_montgomery(x, r_squared, q, q_inv) -> u64` - Domain conversion
   - `mont_mul_cpu(a, b, q, q_inv) -> u64` - CPU Montgomery multiplication

2. **Updated MetalNttContext**:
   - Stores `q_inv`, `r_squared`
   - Stores `omega_powers_montgomery`, `omega_inv_powers_montgomery`
   - Stores `n_inv_montgomery` for final scaling

3. **Domain conversion in forward/inverse**:
   - Input converted to Montgomery domain on CPU before GPU dispatch
   - Output converted back from Montgomery domain after GPU returns
   - All GPU operations stay in Montgomery space

### GPU-Side Implementation (`ntt.metal`)

1. **Updated kernels to use Montgomery multiplication**:
   - `ntt_forward_stage`: Uses `mont_mul(coeffs[j], omega_M, q, q_inv)`
   - `ntt_inverse_stage`: Uses `mont_mul(sub_mod(u, v, q), omega_inv, q, q_inv)`
   - `ntt_inverse_final_scale`: Uses `mont_mul(coeffs[...], n_inv, q, q_inv)`

2. **All kernels now accept `q_inv` parameter** via buffer(5)

### Testing Results

✅ **Metal NTT correctness test passes with ZERO differences**:
- Forward NTT: Perfect match
- Inverse NTT: Perfect match
- Roundtrip test: Max error = 0

## Next Steps - Performance & Integration

---

## References

### Montgomery Arithmetic
- **Paper**: P. Montgomery, "Modular Multiplication Without Trial Division" (1985)
- **Key Idea**: Work in domain x*R mod q where R=2^k, multiplication becomes shift
- **GPU Usage**: Standard technique for modular arithmetic on GPUs without native 128-bit types

### Metal Synchronization
- **Metal Docs**: [Metal Shader Language Specification 3.1](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- **Section 6.8**: "Threadgroup barrier functions only synchronize threads in the same threadgroup"
- **Best Practice**: One kernel dispatch per dependency stage for global synchronization

---

## Expert's Original Suggestions (Implemented/Planned)

| Suggestion | Status | Notes |
|------------|--------|-------|
| Stage-per-dispatch | ✅ Done | Position 0 matches! Sync bug fixed! |
| Montgomery multiplication | ✅ Done | Kernel + CPU implementation complete |
| `q_inv` precomputation | ✅ Done | Extended Euclidean on CPU |
| Twiddle domain conversion | ✅ Done | Multiply by R^2 mod q |
| NTT correctness test | ✅ Done | Zero differences, perfect match! |
| Tile to threadgroup memory | ⏸️ Later | For performance after correctness |

---

## Timeline

- **2025-11-06 12:00**: Bug discovered, multiple `mul_mod` attempts failed
- **2025-11-06 16:00**: Expert analysis received - identified both bugs!
- **2025-11-06 17:00**: Stage-per-dispatch implemented ✅
- **2025-11-06 18:00**: Montgomery kernel added, CPU precomputation in progress 🔄
- **2025-11-06 20:00**: Montgomery implementation complete ✅
- **2025-11-06 20:15**: All tests passing - Metal NTT FIXED! 🎉

---

## Success Criteria

- [x] `test_metal_ntt_correctness`: Zero differences between Metal and CPU ✅
- [ ] `test_metal_keygen`: Decrypt error < 0.01 (currently ~7M) - Next to test
- [ ] N=8192 key generation: < 10 seconds on Metal GPU
- [ ] All 248 V3 tests pass with Metal backend

**Current Status**: ✅ **100% NTT correctness achieved!** Next: Integration testing with key generation

---

## Lessons Learned

1. **GPU Global Barriers Don't Exist Within Kernels**: Must use multiple dispatches for stages that depend on each other writing/reading shared memory.

2. **Modular Arithmetic on 64-bit Hardware**: Cannot naively split 128-bit operations with `%` on parts - use Montgomery multiplication or proper 128-bit long division.

3. **Test Small First**: Isolate arithmetic bugs (mul_mod test) before testing complex algorithms (full NTT).

4. **Trust Expert Analysis**: The expert immediately identified both root causes - global sync and Montgomery arithmetic.

---

**Status**: ✅ **COMPLETE** - Metal GPU NTT is now working perfectly!

**Updated By**: Claude (AI Assistant) with expert guidance
**Last Update**: 2025-11-06 20:15

---

## Summary

The Metal GPU NTT bug is now **completely fixed**! Both root causes identified by the expert have been resolved:

1. ✅ **Global Synchronization**: Stage-per-dispatch pattern ensures proper ordering
2. ✅ **Montgomery Multiplication**: Correct 128-bit modular arithmetic without division

The implementation now passes all correctness tests with **zero differences** compared to the CPU version. The next step is integration testing with the full V3 FHE key generation pipeline.
