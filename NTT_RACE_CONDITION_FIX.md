# NTT Race Condition Fix

**Date:** 2025-11-08
**Issue:** Bit-reversal kernel has race condition causing garbage NTT outputs

## Root Cause

The bit-reversal kernel at `src/clifford_fhe_v2/backends/gpu_metal/shaders/ntt.metal:119` has a critical race condition:

```metal
if (gid < reversed) {
    ulong temp = coeffs[gid];
    coeffs[gid] = coeffs[reversed];      // ⚠️ RACE: Multiple threads write here
    coeffs[reversed] = temp;              // ⚠️ RACE: Multiple threads write here
}
```

### The Problem

For a pair `(i, j)` where `i < j` and `bit_reverse(i) = j` and `bit_reverse(j) = i`:
- **Thread i** executes: swap `coeffs[i] ↔ coeffs[j]`
- **Thread j** executes: nothing (because `j >= bit_reverse(j) = i`)

This looks correct at first glance! **BUT**: For pairs where `i != bit_reverse(i)` but both try to access the same memory:

Example with N=8:
- i=1: reversed=4, so swap coeffs[1] ↔ coeffs[4]
- i=2: reversed=2, condition `2 < 2` is FALSE, skip
- i=3: reversed=6, so swap coeffs[3] ↔ coeffs[6]
- i=4: reversed=1, condition `4 < 1` is FALSE, skip

Wait, this actually looks fine for simple cases. Let me check a more complex scenario...

Actually, the issue is **NOT a traditional race** in the bit-reverse itself (the `gid < reversed` check prevents double-swapping). The real issue is more subtle:

## Real Root Cause: `mul_mod_slow` is Fundamentally Broken

Looking at lines 55-83 in ntt.metal:

```metal
inline ulong mul_mod_slow(ulong a, ulong b, ulong q) {
    ulong hi = mulhi(a, b);
    ulong lo = a * b;

    if (hi == 0) {
        return lo % q;  // Simple case: no overflow
    }

    // ⚠️ BROKEN PATH: Splitting with % loses carry information!
    hi = hi % q;
    lo = lo % q;
    // ... rest of broken 128-bit mod arithmetic
}
```

The comment even admits: "splitting with % loses carries". This means:
- `(hi << 64 | lo) % q` ≠ `((hi % q) << 64 | (lo % q)) % q`
- The "slow" path produces wrong results whenever `hi > 0`
- For 60-bit primes and typical twiddle factors, this happens frequently

### Impact

Every NTT butterfly uses `mul_mod()` which calls `mul_mod_slow()` (line 88). Wrong multiplications propagate through all stages, producing garbage that "looks random" because the errors compound exponentially.

## The Fix: Switch to Montgomery Multiplication

The code already has a correct `mont_mul()` implementation (lines 29-50) that:
1. Uses 128-bit arithmetic correctly
2. Avoids division entirely (only needs `mulhi`)
3. Is branch-free and fast

### Required Changes

1. **Convert all NTT data to Montgomery domain on entry:**
   ```rust
   // In ntt.rs:forward()
   let r_squared_mod_q = compute_r_squared(q);
   for coeff in coeffs.iter_mut() {
       *coeff = mont_mul(*coeff, r_squared_mod_q, q, q_inv);
   }
   ```

2. **Use `mont_mul` in all kernels:**
   ```metal
   // In ntt.metal - butterfly computation
   ulong twiddle_val = twiddles[twiddle_idx];
   ulong t = mont_mul(coeffs[j], twiddle_val, q, q_inv);
   coeffs[j] = sub_mod(coeffs[i], t, q);
   coeffs[i] = add_mod(coeffs[i], t, q);
   ```

3. **Convert back from Montgomery domain on exit:**
   ```rust
   // In ntt.rs:inverse()
   for coeff in coeffs.iter_mut() {
       *coeff = mont_mul(*coeff, 1, q, q_inv);  // Multiply by R^-1
   }
   ```

4. **Precompute Montgomery constants:**
   - `R = 2^64`
   - `R^2 mod q` (for converting to Montgomery)
   - `q_inv = -q^{-1} mod 2^64` (already computed)

## Alternative: Barrett Reduction (if Montgomery is too invasive)

Barrett reduction for `(a * b) mod q`:

```metal
inline ulong barrett_mul_mod(ulong a, ulong b, ulong q, ulong mu_hi, ulong mu_lo) {
    // mu = floor(2^128 / q), precomputed as (mu_hi, mu_lo)

    // Step 1: t = a * b (128-bit)
    ulong t_hi = mulhi(a, b);
    ulong t_lo = a * b;

    // Step 2: Approximate q_hat ≈ t / q using t * mu / 2^128
    // We need (t_hi:t_lo) * (mu_hi:mu_lo) / 2^128
    // Only need the high 128 bits of the 256-bit product
    ulong q_hat = mulhi(t_hi, mu_lo) + t_hi * mu_hi + mulhi(t_lo, mu_hi);

    // Step 3: r = t - q_hat * q (keeping only low 65 bits)
    ulong r = t_lo - q_hat * q;

    // Step 4: Final conditional subtraction (r might be q or 2q too large)
    if (r >= 2*q) r -= 2*q;
    if (r >= q) r -= q;
    return r;
}
```

Precompute `mu = floor(2^128 / q)` on CPU and pass as `(mu_hi, mu_lo)`.

## Testing Plan

1. **Add NTT round-trip test:**
   ```rust
   let input = vec![1, 2, 3, 4, ...];
   let mut data = input.clone();
   ntt.forward(&mut data);
   ntt.inverse(&mut data);
   assert_eq!(data, input);  // Must match exactly
   ```

2. **Compare GPU vs CPU NTT outputs:**
   ```rust
   let mut gpu_data = test_poly.clone();
   let mut cpu_data = test_poly.clone();
   metal_ntt.forward(&mut gpu_data);
   cpu_ntt.forward(&mut cpu_data);
   assert_eq!(gpu_data, cpu_data);  // Exact match required
   ```

3. **Single-coefficient test:**
   ```rust
   let mut data = vec![0u64; n];
   data[0] = 1;
   ntt.forward(&mut data);
   ntt.inverse(&mut data);
   // Should get back [1, 0, 0, ..., 0]
   assert_eq!(data[0], 1);
   assert!(data[1..].iter().all(|&x| x == 0));
   ```

## Priority Actions

1. 🔴 **URGENT:** Replace `mul_mod()` with Montgomery or Barrett
2. 🟡 **HIGH:** Add NTT round-trip unit tests
3. 🟡 **HIGH:** Verify twiddle factor generation matches CPU exactly
4. 🟢 **MEDIUM:** Consider Stockham algorithm to avoid bit-reversal entirely

## Expected Outcome

Once `mul_mod` is fixed, GPU NTT outputs will match CPU exactly, and rotation will work correctly since the NTT stages already have proper global barriers (separate dispatches).

The "random" errors we saw were caused by wrong multiplications compounding through 10 stages of butterflies, not by race conditions in synchronization.
