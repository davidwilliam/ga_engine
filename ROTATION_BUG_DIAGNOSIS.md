# Metal GPU Rotation Bug - Root Cause Analysis

## Executive Summary

The Metal GPU rotation implementation produces incorrect results because it uses a **simplified single-key approach** instead of the **full gadget decomposition** required for CKKS rotation.

## The Bug

**Symptom:**
```
Input:    [1.0, 2.0, 3.0, 4.0, 5.0, ...]
Rotate +1 Expected: [2.0, 3.0, 4.0, 5.0, 6.0, ...]
Rotate +1 Actual:   [-30144.5, 9803.0, -14234.6, ...]  ❌
```

**Root Cause:**

The Metal GPU rotation uses incorrect key structure:

### CPU V3 Implementation (CORRECT)

**Key Structure:**
```rust
pub struct RotationKey {
    pub rlk0: Vec<Vec<RnsRepresentation>>,  // 2D: [num_digits][n]
    pub rlk1: Vec<Vec<RnsRepresentation>>,  // 2D: [num_digits][n]
    pub base_w: u32,  // Gadget base (e.g., 20 → B = 2^20)
}
```

**Key Switching Algorithm:**
```rust
// 1. Decompose c1_rotated using gadget decomposition (base_w = 20)
let c1_digits = gadget_decompose(c1_rotated, base_w);  // ~3 digits

// 2. For each digit:
for (t, digit) in c1_digits.iter().enumerate() {
    let term0 = multiply_ntt(digit, &rlk0[t]);
    let term1 = multiply_ntt(digit, &rlk1[t]);

    c0 -= term0;
    c1_new += term1;
}
```

This is the **standard CKKS rotation** from Microsoft SEAL / OpenFHE.

### Metal GPU Implementation (INCORRECT)

**Key Structure:**
```rust
pub struct MetalRotationKeys {
    keys: HashMap<usize, (Vec<u64>, Vec<u64>)>,  // Just (a_k, b_k) - single pair!
}
```

**Key Switching Formula:**
```rust
// WRONG: Single multiplication without decomposition
c1_final = c1_rotated * a_k;
c0_final = c0_rotated + c1_rotated * b_k;
```

**Why This Fails:**

The rotation key `(a_k, b_k)` where `b_k ≈ -a_k·s + e + σ_k(s)` encodes `σ_k(s)` under fresh noise `e`.

But when you multiply `c1_rotated * b_k`, you get:
- **Noise growth:** `noise(c1_rotated) × noise(b_k)` = **multiplicative noise growth**!
- For typical parameters, this blows up the error to 30k-400k (as observed)

The correct approach (gadget decomposition) keeps noise **additive** by decomposing `c1_rotated` into small digits before multiplying.

## Mathematical Details

### Correct Key Switching (Gadget Decomposition)

**Rotation Key Generation:**

For gadget base `B = 2^w` (e.g., w=20):
```
For t = 0, 1, 2, ..., num_digits-1:
    Sample uniform a_t
    Sample small error e_t
    rlk0[t] = -a_t·s + e_t + B^t · σ_k(s)
    rlk1[t] = a_t
```

**Key Switching:**

```
// Decompose c1_rotated as sum of small digits
c1_rotated ≈ Σ B^t · digit_t  (where |digit_t| < B)

// Key switch each digit
c0' = c0_rotated
c1' = 0

for t in 0..num_digits:
    c0' -= digit_t · rlk0[t]
    c1' += digit_t · rlk1[t]

Return (c0', c1')
```

**Noise Growth:**

- Each `digit_t` is small (< 2^20)
- Error grows as `O(num_digits × B × error_fresh)` = **additive**
- For w=20: error grows by ~3 × 2^20 × 3.2 ≈ 10M (tolerable)

### Incorrect Single-Key Approach (Metal GPU)

**Key Generation:**
```
Sample uniform a
Sample small error e
b = -a·s + e + σ_k(s)
```

**Key Switching:**
```
c1' = c1_rotated · a
c0' = c0_rotated + c1_rotated · b
```

**Noise Growth:**

- `c1_rotated` has noise ~3.2 × (encryption noise)
- `b` has noise ~3.2
- Error grows as `noise(c1_rotated) × noise(b)` = **multiplicative**
- For typical CKKS: ~1e4 × 3.2 = 30k+ (CATASTROPHIC)

## How to Fix

### Option 1: Full Gadget Decomposition (Proper Fix)

**Pros:**
- Matches CPU implementation exactly
- Standard CKKS rotation
- Minimal noise growth

**Cons:**
- Requires implementing gadget decomposition on GPU
- More complex Metal kernels
- Higher key generation cost (3× more keys to store)

**Implementation:**

1. Modify `MetalRotationKeys` to store `(rlk0[], rlk1[], base_w)` per Galois element
2. Implement `gadget_decompose_gpu()` Metal kernel
3. Modify `key_switch_gpu()` to iterate over digits
4. Generate rotation keys using the same algorithm as CPU V3

**Estimated Time:** 4-6 hours

**Files to Change:**
- `src/clifford_fhe_v2/backends/gpu_metal/rotation_keys.rs` - Key structure & generation
- `src/clifford_fhe_v2/backends/gpu_metal/ckks.rs` - Key switching formula
- `src/clifford_fhe_v2/backends/gpu_metal/shaders/rotation.metal` - Add decomposition kernel

### Option 2: BFV-Style Modulus Switching (Alternative)

Some implementations use a special modulus technique that avoids decomposition but still has good noise properties. This is more complex and less standard.

**Not recommended** - Option 1 is cleaner.

## References

- **Microsoft SEAL:** Uses gadget decomposition with w=20-30
  - https://github.com/microsoft/SEAL/blob/main/native/src/seal/evaluator.cpp#L2500

- **OpenFHE:** Uses gadget decomposition (called "hybrid key switching")
  - https://github.com/openfheorg/openfhe-development

- **Theory:** "Efficient Integer Vector Homomorphic Encryption" (FV 2012)
  - Gadget decomposition keeps noise growth additive

## Verification

Once fixed, the rotation test should pass:
```
Input:    [1.0, 2.0, 3.0, 4.0, 5.0, ...]
Rotate +1: [2.0, 3.0, 4.0, 5.0, 6.0, ...]  ✅
Error:    < 1.0 for all slots
```

This will automatically fix bootstrap since CoeffToSlot/SlotToCoeff rely on rotation.
