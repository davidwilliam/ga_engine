# Metal GPU Rotation - Final Debug Status

**Date:** 2025-11-08
**Status:** 🟡 NTT Verified Correct - Bug Isolated to Rotation Logic

## Executive Summary

After comprehensive testing, we have conclusively proven:

✅ **GPU NTT is working perfectly** - Forward→Inverse round-trip returns exact input
✅ **CPU Rotation is working perfectly** - Produces correct results with ~0.001 error
✅ **Montgomery multiplication is correct** - All active kernels use `mont_mul`
✅ **Galois element computation is correct** - Both CPU and GPU use g=5 for step=+1
❌ **GPU Rotation fails** - Produces errors of 50k-120k

## Test Results Summary

### 1. NTT Round-Trip Test ✅ PASS
```
Input:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, ...]
Forward NTT → Inverse NTT
Output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, ...]  ← EXACT MATCH
```

**Conclusion:** GPU NTT implementation is mathematically correct.

### 2. CPU vs GPU Rotation Test
```
Input: [100, 200, 300, 0, 0, ...]
Expected after rotate +1: [200, 300, 0, 0, 0, ...]

CPU Result: [200.0, 300.0, -0.0, 0.0, -0.0]  ✅ PASS
  Errors: slot0=4.01e-4, slot1=2.05e-4

GPU Result: [57187.0, 122929.0, -76446.7, -106506.8, 4590.5]  ❌ FAIL
  Errors: slot0=5.70e4, slot1=1.23e5
```

**Conclusion:** Bug is specifically in Metal GPU rotation, not in shared CPU code.

### 3. Galois Automorphism Test (CPU) ✅ PASS
```
Original: [100, 200, 300]
After automorphism with g=5: [200, 300, 0]
```

**Conclusion:** CPU Galois automorphism works correctly.

## Root Cause Analysis

### Eliminated Hypotheses

1. ❌ **NTT Race Conditions** - ELIMINATED
   - Expert suggested threadgroup barrier issues
   - But code already uses **separate dispatches** for each stage
   - NTT round-trip test proves NTT is correct

2. ❌ **Broken `mul_mod`** - ELIMINATED
   - All active kernels (`ntt_forward_stage`, `ntt_inverse_stage`, `ntt_pointwise_multiply`) use `mont_mul` correctly
   - NTT round-trip test confirms Montgomery multiplication works

3. ❌ **Gadget Decomposition Math** - ELIMINATED
   - Exact floor division fix is in place
   - Decomposition round-trip test passes
   - CPU uses same decomposition and works

4. ❌ **Galois Element Computation** - ELIMINATED
   - Both CPU and GPU compute g=5 for step=+1
   - Formula matches: `g = 5^step mod 2N`

### Active Hypotheses

#### 🔴 Hypothesis 1: Flat RNS Layout Bug in Rotation (MOST LIKELY)

**Evidence:**
- NTT works with flat layout (proven by round-trip test)
- Rotation fails with flat layout
- **Different code paths** handle flat layout extraction:
  - Rotation extracts "active primes" from ciphertext (lines 1121-1132 in ckks.rs)
  - Rotation extracts "active primes" from rotation keys (lines 1094-1110 in ckks.rs)
  - These use different stride calculations

**Suspect Code (ckks.rs:1121-1132):**
```rust
// Extract active primes from ciphertext
let ct_stride = self.c0.len() / n;  // ← May be wrong after rescaling
let mut c0_active = vec![0u64; n * num_primes_active];

for coeff_idx in 0..n {
    for prime_idx in 0..num_primes_active {
        c0_active[coeff_idx * num_primes_active + prime_idx] =
            self.c0[coeff_idx * ct_stride + prime_idx];  // ← Indexing mismatch?
    }
}
```

**Problem:** `ct_stride` might not equal `num_primes_active` after rescaling, causing wrong values to be extracted.

#### 🟡 Hypothesis 2: Galois Automorphism Kernel Applies Wrong Permutation

**Evidence:**
- Slot mapping test shows spikes appear at wrong positions
- But Galois map computation `compute_galois_map(n, k)` looks correct

**Suspect:** The **GPU kernel** might apply the map incorrectly to the flat RNS layout.

**Test Needed:** Manually verify GPU Galois kernel output matches CPU for same flat-layout input.

#### 🟡 Hypothesis 3: Key Switching Bug

**Evidence:**
- Gadget decomposition math is correct (proven by round-trip test)
- But key switching accumulation might have bugs

**Suspect Code (ckks.rs:1139):**
```rust
let (c0_final, c1_final) = self.key_switch_gpu_gadget(
    &c0_rotated, &c1_rotated, &rlk0, &rlk1, moduli, base_w, ctx
)?;
```

**Potential Issues:**
- NTT domain mismatch between decomposed digits and rotation keys
- Wrong twist/untwist application
- Accumulation errors

## Next Debugging Steps

### Step 1: Add Flat Layout Assertions 🔴 HIGH PRIORITY
```rust
// In rotate_by_steps(), before extraction:
assert_eq!(self.c0.len(), n * self.num_primes,
    "Ciphertext c0 must have full flat layout");
assert_eq!(ct_stride, self.num_primes,
    "Stride must equal total num_primes for flat layout");
```

### Step 2: Print Intermediate Values
Add logging to `rotate_by_steps()`:
```rust
eprintln!("[ROTATION DEBUG] n={}, num_primes_active={}, ct_stride={}",
    n, num_primes_active, ct_stride);
eprintln!("[ROTATION DEBUG] c0.len()={}, c1.len()={}",
    self.c0.len(), self.c1.len());
eprintln!("[ROTATION DEBUG] First 5 c0_active values: {:?}",
    &c0_active[..5.min(c0_active.len())]);
```

### Step 3: Test Galois Kernel Directly
Create minimal test:
```rust
// Apply GPU Galois kernel to known input
// Compare byte-for-byte with CPU apply_galois_automorphism
```

### Step 4: Test Key Switching in Isolation
```rust
// Decompose a known c1_rotated
// Apply key switching
// Verify result matches CPU V3 key_switch
```

## Code Locations

### Working Components
- `src/clifford_fhe_v2/backends/gpu_metal/shaders/ntt.metal` - NTT kernels ✅
- `src/clifford_fhe_v2/backends/gpu_metal/ntt.rs` - NTT host code ✅
- `src/clifford_fhe_v3/bootstrapping/rotation.rs` - CPU rotation ✅
- `src/clifford_fhe_v3/bootstrapping/keys.rs` - CPU Galois automorphism ✅

### Suspect Components
- `src/clifford_fhe_v2/backends/gpu_metal/ckks.rs:1072-1149` - `rotate_by_steps()` 🔴
  - Lines 1121-1132: Active prime extraction from ciphertext
  - Lines 1094-1110: Active prime extraction from rotation keys
  - Line 1139: Key switching call
- `src/clifford_fhe_v2/backends/gpu_metal/ckks.rs:1200-1450` - `key_switch_gpu_gadget()` 🟡
- `src/clifford_fhe_v2/backends/gpu_metal/shaders/rotation.metal:45-87` - Galois kernel 🟡

## Expert Guidance Summary

The expert's key points were:
1. ✅ **Global barriers** - Verified: code uses separate dispatches
2. ✅ **Barrett/Montgomery reduction** - Verified: using `mont_mul` correctly
3. 🔴 **Flat RNS layout consistency** - NOT YET VERIFIED
4. 🔴 **On-GPU fingerprint test** - Created NTT round-trip test ✅

**Quote:** "Lock the RNS flat layout once and test it on-GPU"

This suggests the flat layout indexing is the root cause, which aligns with Hypothesis 1.

## Immediate Action Plan

1. **Add stride assertions** to catch layout mismatches
2. **Print c0/c1 lengths and strides** in rotate_by_steps
3. **Test Galois kernel** in isolation with known input/output
4. **If still failing:** Add coefficient-level logging to key_switch_gpu_gadget

## Expected Resolution

Once flat RNS layout indexing is fixed, GPU rotation should match CPU rotation exactly (error < 0.01).

The fact that NTT works perfectly but rotation fails strongly suggests the bug is in how rotation **extracts and reassembles** the flat layout, not in the core mathematical operations.
