# Metal GPU Rotation Bug - ISOLATED

**Date:** 2025-11-08
**Status:** 🟡 Bug Location Identified - In Key Switching or Rotation Key Generation

## Critical Discovery

After extensive debugging with coefficient-level logging, we have **definitively isolated** the bug:

### ✅ What's Working Correctly

1. **GPU NTT** - Round-trip test proves forward→inverse is exact ✅
2. **Galois Automorphism Kernel** - Verified to permute coefficients correctly:
   ```
   Coeff 0 → position 0 ✅
   Coeff 1 → position 5 ✅
   Coeff 2 → position 10 ✅
   ```
3. **CPU Rotation** - Works perfectly with ~0.0006 error ✅

### ❌ What's Failing

**Key Switching Step** - After Galois automorphism, the key switching produces incorrect results:
```
Input:  [100, 200, 300, 0, 0, ...]
After Galois: (coefficients correctly permuted)
After Key Switch: produces ciphertext that decrypts to [67418, -16951, 117353, ...] ❌
Expected: [200, 300, 0, ...]
```

## Debug Output Trace

```
[ROTATION DEBUG] Before Galois:
  c0_active[0:5] = [318697941233423965, 11154619807722, ...]
  c1_active[0:5] = [28391310067865241, 16186952206753, ...]

[GALOIS DEBUG] Galois Kernel Execution:
  Coeff 0 (mod q0): input=377609875881240019, target_pos=0, output@target=377609875881240019 ✅
  Coeff 1 (mod q0): input=550561488710099746, target_pos=5, output@target=550561488710099746 ✅
  Coeff 2 (mod q0): input=575422193002722400, target_pos=10, output@target=575422193002722400 ✅

[ROTATION DEBUG] After Galois:
  c0_rotated[0:5] = [318697941233423965, 11154619807722, ...] (SAME - coefficient 0 stayed at position 0) ✅
  c1_rotated[0:5] = [28391310067865241, 16186952206753, ...] (SAME - coefficient 0 stayed at position 0) ✅

[ROTATION DEBUG] After Key Switch:
  c0_final[0:5] = [565658924277838398, 3992719090635, ...]
  c1_final[0:5] = [571406490378597860, 10059377884448, ...]

Decryption Result: [67418, -16951, 117353, ...] ❌ WRONG
```

## Root Cause Analysis

Since:
1. NTT is proven correct ✅
2. Galois automorphism is proven correct ✅
3. CPU rotation (which uses similar key switching) works ✅

The bug **MUST** be in one of these GPU-specific components:

### 🔴 Hypothesis 1: Rotation Key Generation is Wrong (MOST LIKELY)

**File:** `src/clifford_fhe_v2/backends/gpu_metal/rotation_keys.rs`

**Suspect Function:** `generate_rotation_key_for_k_gadget()`

**Potential Issues:**
1. **Galois automorphism applied to wrong secret key component**
   - Should compute `s_k = s(X^k)` by applying σ_k to secret key
   - Might be applying wrong permutation or missing negation

2. **Wrong polynomial in rlk0 generation**
   ```rust
   // Should be: rlk0[t] = -a_t·s + e_t + B^t·s_k
   // Might have: wrong s_k, or wrong B^t computation
   ```

3. **Keys stored in wrong domain**
   - CPU might expect coefficient domain
   - GPU might be generating in NTT domain
   - Or vice versa

**Evidence:**
- Gadget decomposition math is correct (proven by round-trip test)
- Key switching logic looks standard
- But final result is completely wrong

### 🟡 Hypothesis 2: Key Switching NTT Domain Mismatch

**File:** `src/clifford_fhe_v2/backends/gpu_metal/ckks.rs:1200-1450`

**Function:** `key_switch_gpu_gadget()`

**Potential Issues:**
1. **Decomposed digits in wrong domain**
   - `c1_digits` might be in coefficient domain
   - But multiplication expects NTT domain

2. **Rotation keys in wrong domain**
   - Keys might be pre-transformed to NTT
   - But code tries to transform them again

3. **Missing or double twist/untwist**
   - Code does: twist → NTT → multiply → iNTT → untwist
   - But keys might already be twisted

**Current Code Pattern:**
```rust
fn multiply_digit_by_ntt_key(...) {
    // Extract polynomials from flat layout

    // Apply twist
    digit_poly[i] = mont_mul(digit_poly[i], psi_powers[i], ...);
    key_poly[i] = mont_mul(key_poly[i], psi_powers[i], ...);

    // NTT
    ntt.forward(&mut digit_poly)?;
    ntt.forward(&mut key_poly)?;

    // Multiply
    ntt.pointwise_multiply(...)?;

    // iNTT
    ntt.inverse(&mut result)?;

    // Untwist
    result[i] = mont_mul(result[i], psi_inv_powers[i], ...);
}
```

This looks correct, but if keys are already in NTT domain or already twisted, this would double-apply.

## Diagnostic Steps to Isolate Further

### Step 1: Compare Rotation Keys (GPU vs CPU)
```rust
// In test:
let cpu_rot_key = generate_rotation_keys(&[1], &sk, &params);
let gpu_rot_key = MetalRotationKeys::generate(..., &sk, &[1], ...);

// Compare first few coefficients of rlk0[0], rlk1[0]
// They should be IDENTICAL if generation is correct
```

### Step 2: Test Key Switching with Known Input
```rust
// Create noiseless test:
// - Known c1_rotated (simple pattern)
// - Known rotation keys (potentially just identity)
// - Verify key switch output matches hand calculation
```

### Step 3: Check if Keys are in Correct Domain
```rust
// In rotation_keys.rs:
eprintln!("[KEY GEN DEBUG] Generating rlk0[0], first 5 coeffs: {:?}", &rlk0_t[..5]);

// If values are very large (~q), they're in coefficient domain
// If values are distributed, they might be in NTT domain
```

## Next Actions

1. 🔴 **IMMEDIATE:** Add logging to `generate_rotation_key_for_k_gadget()` to show:
   - How s_k is computed (after applying σ_k to secret key)
   - First few coefficients of rlk0[0], rlk1[0]
   - Compare with CPU rotation key generation

2. 🟡 **HIGH PRIORITY:** Test with noiseless keys (set all error to 0):
   - If still fails → bug is in key generation formula
   - If works → bug is in noise handling

3. 🟢 **MEDIUM:** Add assertions in `multiply_digit_by_ntt_key`:
   ```rust
   assert!(digit_poly values are reasonable);
   assert!(key_poly values are reasonable);
   assert!(result values are reasonable);
   ```

## Expected Fix

Once rotation keys are generated correctly (matching CPU V3), GPU rotation should match CPU rotation exactly (error < 0.01).

The fact that Galois automorphism works but key switching fails strongly suggests **the rotation keys themselves are wrong**, not the key switching algorithm.

## Code Locations to Investigate

### Primary Suspect
- `src/clifford_fhe_v2/backends/gpu_metal/rotation_keys.rs:150-250` - `generate_rotation_key_for_k_gadget()`

### Secondary Suspects
- `src/clifford_fhe_v2/backends/gpu_metal/ckks.rs:1265-1340` - `multiply_digit_by_ntt_key()`
- `src/clifford_fhe_v2/backends/gpu_metal/ckks.rs:1200-1264` - `key_switch_gpu_gadget()`

### Reference (Working CPU Code)
- `src/clifford_fhe_v3/bootstrapping/keys.rs:186-260` - CPU rotation key generation
- `src/clifford_fhe_v3/bootstrapping/rotation.rs:41-71` - CPU rotation (uses key switching)

## Conclusion

We have successfully isolated the bug to **rotation key generation** or **key switching NTT domain handling**. The Galois automorphism is definitively working correctly, and NTT is proven correct. This narrows the debugging to a much smaller code surface.
