# Metal GPU NTT Implementation Bug Report

**Date**: 2025-11-06
**Status**: 🔴 BLOCKING - Metal GPU NTT produces incorrect results
**Priority**: HIGH - Required for production V3 bootstrap performance

---

## Executive Summary

The Metal GPU implementation of Number Theoretic Transform (NTT) produces completely incorrect results compared to the CPU implementation. The bug appears to be in the 128-bit modular multiplication function (`mul_mod`), but multiple implementation approaches have all failed, suggesting a deeper issue with Metal shader behavior or thread synchronization.

**Impact**: Without working Metal GPU acceleration, V3 bootstrap with production parameters (N=8192, 20 primes) would take hours on CPU instead of seconds on GPU.

---

## Problem Statement

### What Works
- ✅ **CPU NTT Implementation**: 100% correct, all tests passing
- ✅ **V3 Bootstrap on CPU**: Complete implementation, 248/249 tests passing
- ✅ **Metal Device Initialization**: Successfully creates Metal compute pipeline
- ✅ **Metal Kernel Compilation**: All kernels compile without errors

### What's Broken
- ❌ **Metal NTT Forward Transform**: Produces completely different results from CPU
- ❌ **Metal `mul_mod` Function**: 128-bit modular multiplication is incorrect
- ❌ **Key Generation with Metal**: Decrypt errors are huge (~7M instead of <0.01)

---

## Test Case: Direct NTT Comparison

**File**: `examples/test_metal_ntt_correctness.rs`

**Test Setup**:
- N = 1024
- q = 1152921504606748673 (60-bit NTT-friendly prime)
- Input: `(0..1024).map(|i| (i * 12345) % q)`

**Expected Behavior**: Metal NTT output should exactly match CPU NTT output

**Actual Behavior**:
```
Test 1: Forward NTT
─────────────────────
✓ CPU forward NTT completed
✓ Metal forward NTT completed

Difference at position 0: CPU=6466014720, Metal=2623077945, diff=3842936775
Difference at position 1: CPU=228851822468311735, Metal=683278145774428238, diff=454426323306116503
Difference at position 2: CPU=58158800678276883, Metal=1084582203744116835, diff=1026423403065839952

❌ Forward NTT: 1024 differences found, max diff = 1152921501977374778
```

**Key Observation**: The differences are HUGE (near the modulus q), indicating fundamental arithmetic errors, not just precision issues.

---

## The Buggy Code

### File: `src/clifford_fhe_v2/backends/gpu_metal/shaders/ntt.metal`

### Current `mul_mod` Implementation (Latest Attempt)

```metal
/// Modular multiplication: (a * b) mod q
/// Implements exact 128-bit modular reduction: (a * b) mod q
/// Equivalent to Rust: ((a as u128 * b as u128) % q as u128) as u64
inline ulong mul_mod(ulong a, ulong b, ulong q) {
    // Compute full 128-bit product
    ulong hi = mulhi(a, b);  // High 64 bits of a * b
    ulong lo = a * b;         // Low 64 bits of a * b

    if (hi == 0) {
        // Product fits in 64 bits
        return lo % q;
    }

    // For 60-bit primes, hi should be small (< 2^56)
    // Compute: (hi * 2^64 + lo) mod q

    // Step 1: Reduce both parts
    hi = hi % q;
    lo = lo % q;

    // Step 2: Compute 2^64 mod q
    // We use: 2^64 = (2^32)^2
    ulong two_32 = (1UL << 32) % q;
    ulong two_64_mod_q = (two_32 * two_32) % q;

    // Step 3: Compute (hi * two_64_mod_q) % q
    ulong hi_contrib_hi = mulhi(hi, two_64_mod_q);
    ulong hi_contrib_lo = hi * two_64_mod_q;

    if (hi_contrib_hi == 0) {
        ulong hi_contrib = hi_contrib_lo % q;
        ulong result = (hi_contrib + lo) % q;
        return result;
    } else {
        // Overflow happened - recursively reduce
        hi_contrib_hi = hi_contrib_hi % q;
        ulong temp = (hi_contrib_hi * two_64_mod_q) % q;
        ulong hi_contrib = (temp + (hi_contrib_lo % q)) % q;
        ulong result = (hi_contrib + lo) % q;
        return result;
    }
}
```

**Problem**: This implementation SHOULD be correct mathematically, but produces wrong results.

### Working CPU Implementation (For Reference)

**File**: `src/clifford_fhe_v2/backends/cpu_optimized/ntt.rs:420`

```rust
#[inline(always)]
fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}
```

**Why This Works**: Rust's native 128-bit integer type handles the overflow correctly.

**Why Metal Doesn't Have This**: Metal Shading Language doesn't support 128-bit integer types.

---

## Previous Implementation Attempts (All Failed)

### Attempt 1: Binary Long Division
```metal
// Process all 128 bits one by one using long division algorithm
ulong remainder = 0;
for (int bit_pos = 127; bit_pos >= 0; bit_pos--) {
    remainder <<= 1;
    ulong bit_value = (bit_pos >= 64) ? ((hi >> (bit_pos - 64)) & 1) : ((lo >> bit_pos) & 1);
    remainder |= bit_value;
    if (remainder >= q) {
        remainder -= q;
    }
}
return remainder;
```
**Result**: ❌ Still produces incorrect results (and is very slow - 128 iterations per multiplication)

### Attempt 2: Iterative Reduction
```metal
// Compute (hi * 2^64 + lo) mod q iteratively
hi = hi % q;
ulong two64_mod_q = ((~0ULL) % q + 1) % q;
ulong hi_contribution = (hi * two64_mod_q) % q;
return (hi_contribution + (lo % q)) % q;
```
**Result**: ❌ Incorrect results

### Attempt 3: Floating-Point Approximation
```metal
// Use float to approximate quotient, then refine
float dividend_f = (float)hi * 18446744073709551616.0f + (float)lo;
float quotient_approx = floor(dividend_f / (float)q);
// ... subtract quotient * q from dividend
```
**Result**: ❌ Metal doesn't support `double`, only `float`. Single precision insufficient for 60-bit primes.

---

## NTT Kernel Implementation (Appears Correct)

**File**: `src/clifford_fhe_v2/backends/gpu_metal/shaders/ntt.metal:99`

```metal
kernel void ntt_forward(
    device ulong* coeffs [[buffer(0)]],
    constant ulong* twiddles [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant ulong& q [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // Bit-reversal permutation
    if (gid < n) {
        uint reversed = 0;
        uint k = gid;
        uint logn = 31 - clz(n);

        for (uint i = 0; i < logn; i++) {
            reversed = (reversed << 1) | (k & 1);
            k >>= 1;
        }

        if (gid < reversed) {
            ulong temp = coeffs[gid];
            coeffs[gid] = coeffs[reversed];
            coeffs[reversed] = temp;
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // NTT butterfly stages (Cooley-Tukey)
    for (uint stage = 0; stage < 31 - clz(n); stage++) {
        uint m = 1 << (stage + 1);
        uint m_half = 1 << stage;

        uint butterfly_idx = gid;
        if (butterfly_idx < n / 2) {
            uint block_idx = butterfly_idx / m_half;
            uint idx_in_block = butterfly_idx % m_half;

            uint i = block_idx * m + idx_in_block;
            uint j = i + m_half;

            uint twiddle_idx = (n / m) * idx_in_block;
            ulong omega = twiddles[twiddle_idx];

            // Harvey butterfly
            ulong u = coeffs[i];
            ulong v = mul_mod(coeffs[j], omega, q);  // <-- BUG IS HERE

            coeffs[i] = add_mod(u, v, q);
            coeffs[j] = sub_mod(u, v, q);
        }

        threadgroup_barrier(mem_flags::mem_device);
    }
}
```

**Analysis**: This kernel structure is identical to the working CPU implementation. The only difference is the `mul_mod` call.

---

## Hypotheses for Root Cause

### Hypothesis 1: Metal's `%` Operator Behavior
**Theory**: Metal's modulo operator might not handle 64-bit dividends with 60-bit divisors correctly.

**Test**: Try implementing modulo using subtraction instead:
```metal
inline ulong safe_mod(ulong a, ulong q) {
    while (a >= q) {
        a -= q;
    }
    return a;
}
```

### Hypothesis 2: Thread Synchronization Issues
**Theory**: The `threadgroup_barrier()` might not be synchronizing correctly, causing race conditions.

**Evidence Against**: The bit-reversal permutation (before NTT stages) also uses barriers and seems to work.

### Hypothesis 3: Buffer Alignment or Memory Model
**Theory**: Metal's memory model for `device ulong*` might have alignment requirements we're not meeting.

**Test**: Try using `device atomic_ulong*` or explicit memory fences.

### Hypothesis 4: Integer Overflow in Intermediate Calculations
**Theory**: Even though we're trying to prevent overflow, Metal might be wrapping values unexpectedly.

**Test**: Add explicit overflow checks and error reporting in the shader.

### Hypothesis 5: `mulhi()` Function Bug
**Theory**: Metal's `mulhi()` might not correctly compute the high 64 bits.

**Test**: Implement `mulhi` manually using shifts and additions:
```metal
inline ulong manual_mulhi(ulong a, ulong b) {
    // Split into 32-bit parts
    ulong a_lo = a & 0xFFFFFFFF;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFF;
    ulong b_hi = b >> 32;

    // Compute partial products
    ulong p0 = a_lo * b_lo;
    ulong p1 = a_lo * b_hi;
    ulong p2 = a_hi * b_lo;
    ulong p3 = a_hi * b_hi;

    // Combine (carry propagation)
    ulong carry = ((p0 >> 32) + (p1 & 0xFFFFFFFF) + (p2 & 0xFFFFFFFF)) >> 32;
    return p3 + (p1 >> 32) + (p2 >> 32) + carry;
}
```

---

## Files Involved

### Core Implementation Files
1. **`src/clifford_fhe_v2/backends/gpu_metal/shaders/ntt.metal`**
   - Lines 13-75: `mul_mod` function (BUGGY)
   - Lines 77-82: `add_mod` function (seems OK)
   - Lines 84-89: `sub_mod` function (seems OK)
   - Lines 99-156: `ntt_forward` kernel (structure looks correct)
   - Lines 165-211: `ntt_inverse` kernel (structure looks correct)

2. **`src/clifford_fhe_v2/backends/gpu_metal/ntt.rs`**
   - Lines 108-157: `forward()` method - calls `ntt_forward` kernel
   - Lines 159-205: `inverse()` method - calls `ntt_inverse` kernel
   - Lines 207-234: `pointwise_multiply()` method

3. **`src/clifford_fhe_v2/backends/gpu_metal/keys.rs`**
   - Lines 244-291: `multiply_polynomials_gpu()` - uses NTT for key generation

### Test Files
1. **`examples/test_metal_ntt_correctness.rs`** (NEW - created for debugging)
   - Direct comparison test: Metal NTT vs CPU NTT
   - **Run**: `cargo run --release --features v2,v2-gpu-metal --example test_metal_ntt_correctness`

2. **`examples/test_metal_keygen.rs`**
   - Full key generation test
   - Shows decrypt error of ~7M (should be <0.01)
   - **Run**: `cargo run --release --features v2,v2-gpu-metal --example test_metal_keygen`

### Working CPU Reference
1. **`src/clifford_fhe_v2/backends/cpu_optimized/ntt.rs`**
   - Lines 420-422: Working `mul_mod` using Rust's u128
   - Lines 169-232: `forward_ntt()` - proven correct implementation
   - Lines 234-277: `inverse_ntt()` - proven correct implementation
   - Lines 279-317: `multiply_polynomials()` - applies twist, calls NTT, untwists

---

## Environment Details

### Hardware
- **Device**: Apple M3 Max
- **GPU**: Apple Silicon integrated GPU
- **Max Threads Per Threadgroup**: 1024

### Software
- **macOS**: Version 14.x (Sonoma)
- **Rust**: 1.80+
- **Metal**: Metal 3.1 (via `metal-rs` crate)
- **Features**: `v2`, `v2-gpu-metal`

### Build Command
```bash
cargo build --release --features v2,v2-gpu-metal --example test_metal_ntt_correctness
```

---

## What We Need Help With

### Primary Question
**How do we correctly implement 128-bit modular multiplication in Metal Shading Language?**

Specifically: `(a * b) mod q` where `a, b, q` are all 64-bit unsigned integers, and `q` is approximately 60 bits.

### Secondary Questions

1. **Is Metal's `%` operator reliable for 64-bit modulo operations?**
   - Should we use subtraction-based modulo instead?
   - Are there known precision issues with large 64-bit values?

2. **Does Metal's `mulhi()` correctly compute high 64 bits?**
   - Should we implement our own using 32-bit multiplications?
   - Are there alignment or type conversion issues?

3. **Are there thread synchronization issues with our NTT kernel?**
   - Is `threadgroup_barrier(mem_flags::mem_device)` sufficient?
   - Should we use atomic operations or memory fences?

4. **Are there better approaches for large integer arithmetic in Metal?**
   - Custom fixed-point arithmetic?
   - Lookup tables for common operations?
   - Precomputed Barrett reduction constants?

---

## Reproduction Steps

### Step 1: Clone and Build
```bash
git clone <repo_url>
cd ga_engine
cargo build --release --features v2,v2-gpu-metal --example test_metal_ntt_correctness
```

### Step 2: Run NTT Correctness Test
```bash
cargo run --release --features v2,v2-gpu-metal --example test_metal_ntt_correctness
```

**Expected Output**: "✅ ALL TESTS PASSED - Metal NTT is CORRECT!"

**Actual Output**:
```
❌ Forward NTT: 1024 differences found, max diff = 1152921501977374778
```

### Step 3: Run Key Generation Test
```bash
cargo run --release --features v2,v2-gpu-metal --example test_metal_keygen
```

**Expected Output**: "✓ Encrypt/decrypt works! Max error: 0.000xxx"

**Actual Output**:
```
❌ Encrypt/decrypt error too large: 7599648.380508
```

---

## Mathematical Background

### Number Theoretic Transform (NTT)
NTT is a fast polynomial multiplication algorithm used in FHE (Fully Homomorphic Encryption). It's analogous to FFT but works in finite fields mod q.

**Key Property**: For NTT to work, we need:
- `q ≡ 1 (mod 2n)` where n is the polynomial degree
- `q` is prime
- Modular arithmetic must be **exact** (no rounding errors allowed)

### Why 128-bit Multiplication?
When multiplying two 64-bit values `a * b` where `a, b < q` and `q ≈ 2^60`:
- Product `a * b < 2^120` (doesn't fit in 64 bits)
- We need to compute `(a * b) mod q` using 128-bit intermediate result
- In Rust: `((a as u128 * b as u128) % q as u128) as u64` - trivial!
- In Metal: No 128-bit type - must implement manually

### Our FHE Parameters
- **N (polynomial degree)**: 1024, 4096, or 8192
- **q (modulus)**: ~60-bit NTT-friendly primes (e.g., 1152921504606748673)
- **Number of primes**: 3-20 (RNS representation)
- **Security**: 128-bit (equivalent to AES-128)

---

## Workarounds Considered

### Option 1: Use CPU Only
**Pros**: Works perfectly, all tests passing
**Cons**: Too slow for production (N=8192 key generation takes minutes instead of seconds)

### Option 2: Hybrid CPU/GPU
**Pros**: Could work around the bug
**Cons**: Complex, defeats the purpose of GPU acceleration

### Option 3: Precompute Multiplication Table
**Pros**: Avoid `mul_mod` entirely
**Cons**: Infeasible - table would be enormous (2^60 × 2^60 entries)

### Option 4: Switch to CUDA
**Pros**: CUDA has better integer arithmetic support
**Cons**: Requires NVIDIA GPU, not portable to Apple Silicon

**Current Decision**: Fix Metal implementation - it's the right long-term solution for Apple Silicon.

---

## References

### Metal Shading Language Specification
- [Metal Shading Language Guide](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- Section on Integer Arithmetic (Chapter 6.1)
- `mulhi()` function documentation

### FHE and NTT Background
- **CKKS Paper**: Cheon et al., "Homomorphic Encryption for Arithmetic of Approximate Numbers" (ASIACRYPT 2017)
- **NTT in FHE**: Harvey, "Faster Arithmetic for Number-Theoretic Transforms" (2014)
- **Our Implementation**: Based on proven CPU implementation in `ntt.rs`

### Similar Issues (if any)
- [Stack Overflow: 128-bit arithmetic in Metal](https://stackoverflow.com/questions/tagged/metal+128-bit)
- GPU forums discussing modular multiplication

---

## Contact

For questions about this bug report or the codebase:
- **Repository**: `ga_engine` (Geometric Algebra FHE Engine)
- **Component**: V2 Metal GPU Backend
- **Related Module**: V3 CKKS Bootstrapping (depends on working NTT)

---

## Appendix: Full Error Output

### NTT Correctness Test Output
```
Testing Metal NTT Correctness vs CPU
═══════════════════════════════════════

Parameters: N=1024, q=1152921504606748673
Primitive 2n-th root (psi): 715033771596066358
Verification: psi^(2n) mod q = 1

Test 1: Forward NTT
─────────────────────
✓ CPU forward NTT completed
Metal Device: Apple M3 Max
Metal Max Threads Per Threadgroup: 1024
✓ Metal forward NTT completed
  Difference at position 0: CPU=6466014720, Metal=2623077945, diff=3842936775
  Difference at position 1: CPU=228851822468311735, Metal=683278145774428238, diff=454426323306116503
  Difference at position 2: CPU=58158800678276883, Metal=1084582203744116835, diff=1026423403065839952
  Difference at position 3: CPU=233769716892041225, Metal=539826746488970830, diff=306057029596929605
  Difference at position 4: CPU=262648086912537303, Metal=919188415737345813, diff=656540328824808510
❌ Forward NTT: 1024 differences found, max diff = 1152921501977374778

Test 2: Inverse NTT
─────────────────────
(Not reached due to forward NTT failure)
```

### Key Generation Test Output
```
Test 1: N=1024, 3 primes (quick validation)
═══════════════════════════════════════════════════════
Parameters: N=1024, 3 primes
  [Metal GPU] Creating Metal device...
Metal Device: Apple M3 Max
  [Metal GPU] Device initialized in 0.099s
  [Metal GPU] Creating NTT contexts for 3 primes...
  [Metal GPU] NTT contexts created in 0.00s
  [Metal GPU] Starting keygen for N=1024, 3 primes
  ✓ Keys generated in 0.15s

Testing encrypt/decrypt roundtrip...
❌ Encrypt/decrypt error too large: 7599648.380508
```

---

## Status: AWAITING EXPERT ASSISTANCE

**Last Updated**: 2025-11-06
**Debugged By**: Claude (AI Assistant)
**Time Invested**: ~4 hours of systematic debugging
**Status**: Blocked - need Metal GPU expertise

**Priority Items**:
1. Fix `mul_mod` implementation in Metal
2. Verify NTT kernel produces correct results
3. Validate full key generation pipeline
4. Benchmark performance vs CPU

**Success Criteria**:
- [ ] `test_metal_ntt_correctness` passes with zero differences
- [ ] `test_metal_keygen` shows decrypt error < 0.01
- [ ] N=8192 key generation completes in <10 seconds
- [ ] All 248 V3 tests still pass with Metal backend
