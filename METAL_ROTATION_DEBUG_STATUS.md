# Metal GPU Rotation Debug Status

**Date:** 2025-11-08
**Status:** 🔴 Bug Identified - Debugging in Progress

## Summary

Metal GPU rotation is failing while CPU V3 rotation works perfectly on identical inputs. Both use the same Galois element (g=5 for rotation step +1) and gadget decomposition, confirming the bug is specifically in the Metal GPU implementation.

## Test Results

### CPU vs GPU Comparison Test
```
Input: [100.0, 200.0, 300.0, 0.0, 0.0, ...]
Expected after rotate +1: [200.0, 300.0, 0.0, 0.0, 0.0, ...]

CPU V3 Result:  [200.0, 300.0, -0.0, -0.0, -0.0]  ✅ PASS
  Errors: slot0=1.89e-3, slot1=5.15e-4

GPU Metal Result: [-63890.1, -6346.3, 48834.9, -40333.8, 69688.0]  ❌ FAIL
  Errors: slot0=6.41e4, slot1=6.65e3
```

### Slot Mapping Test (Spike Positioning)
When encrypting a message with a spike at one position and rotating:
- Input slot 0 → Found at slot 10 (expected 511)
- Input slot 1 → Found at slot 13 (expected 0)
- Input slot 2 → Found at slot 31 (expected 1)

**Pattern:** Spikes appear at seemingly random positions, suggesting either:
1. Galois automorphism permutation is incorrect
2. Slot-to-coefficient mapping issue
3. NTT domain mismatch

### Decomposition Round-Trip Test
✅ **PASS** - Gadget decomposition with exact floor division mathematically correct.

## Implementation Details

### What's Working
1. ✅ **Gadget decomposition structure** - 31 digits with base_w=30
2. ✅ **Exact floor division fix** - `div_floor()` instead of bit-shift for negative BigInt
3. ✅ **Decomposition round-trip** - Values reconstruct correctly
4. ✅ **CPU V3 rotation** - Perfect accuracy (~0.002 error)
5. ✅ **Galois element computation** - Both CPU and GPU use g=5 for step=+1

### What's Failing
1. ❌ **GPU Metal rotation** - All slots have massive errors (3k-270k)
2. ❌ **Slot positioning** - Spikes appear at wrong locations after rotation

## Confirmed Facts

### Galois Element Computation (CORRECT ✅)
Both CPU and GPU compute the same Galois element:
```rust
// For rotation step +1 with N=1024:
k_normalized = 1
g = 5^1 mod 2048 = 5
```

### Galois Automorphism Logic (VERIFIED ✅)
Both implementations use the same permutation formula:
```rust
for i in 0..n {
    new_idx = (g * i) % 2N
    if new_idx < n:
        result[new_idx] = poly[i]        // Positive
    else:
        result[new_idx - n] = -poly[i]   // Negative (wrap around)
}
```

Metal shader implements this correctly:
```metal
uint target_idx = galois_map[gid];  // Precomputed: (g * gid) % 2N
output[target_idx * num_primes + prime_idx] = input[gid * num_primes + prime_idx];
```

### Key Switching with Gadget Decomposition

**CPU V3 approach:**
```rust
1. Apply Galois automorphism to c0, c1 (coefficient domain)
2. Gadget decompose c1_rotated
3. Key switch: accumulate over digits
```

**GPU Metal approach:**
```rust
1. Apply Galois automorphism to c0, c1 (coefficient domain)
2. Gadget decompose c1_rotated using exact floor division
3. For each digit t:
   - Transform to NTT domain (twist → NTT)
   - Pointwise multiply with rlk0[t], rlk1[t]
   - Transform back (iNTT → untwist)
4. Accumulate results
```

## Hypotheses for Bug Location

### Hypothesis 1: NTT Domain Mismatch ⚠️ LIKELY
The Metal implementation multiplies in NTT domain using:
```rust
twist → NTT → multiply → iNTT → untwist
```

But rotation keys might be stored in a different convention or domain.

**Evidence:**
- Earlier experiments with storing keys in NTT domain vs coefficient domain produced similar errors
- The pattern suggests coefficient permutation rather than complete randomness

### Hypothesis 2: Flat RNS Layout Indexing Bug 🔍 POSSIBLE
The flat layout uses: `idx = coeff * num_primes + prime_idx`

Potential issues:
- Mismatch between ciphertext stride and rotation key stride after rescaling
- Incorrect extraction of active primes from full-sized rotation keys

**Evidence:**
- Code has stride handling logic (lines 1095, 1121-1132 in ckks.rs)
- Different components may have different strides

### Hypothesis 3: Sign Handling in Gadget Decomposition 🔍 POSSIBLE
Centered residues in gadget decomposition need careful sign handling:
```rust
if digit_t > half_base {
    digit_t -= &base_big;  // Center to (-B/2, B/2]
}
```

Then conversion to RNS unsigned:
```rust
let digit_unsigned = if d < &BigInt::zero() {
    let x = (-d).mod_floor(&BigInt::from(q)).to_u64().unwrap_or(0);
    if x == 0 { 0 } else { q - x }
}
```

**Evidence:**
- Exact floor division fix improved Slot 0 from ~85k to ~25 error in earlier tests
- Current tests show all slots failing, suggesting regression or parameter change

## Next Steps

### Immediate Actions
1. 🔍 **Add coefficient-level logging** to key switching
   - Show input/output for first few coefficients
   - Verify NTT forward/inverse round-trips correctly
   - Check if twist/untwist is being double-applied

2. 🔍 **Verify rotation key generation**
   - Print first few coefficients of rlk0[0], rlk1[0]
   - Compare CPU vs GPU rotation key values
   - Check if keys are in correct domain (coefficient vs NTT)

3. 🔍 **Test noiseless single-digit case**
   - Set all noise to zero
   - Use only first digit (t=0)
   - Use single prime (simplest RNS)
   - This isolates whether bug is in decomposition or accumulation

### Investigation Plan
1. Create minimal test with N=16, single prime, single digit
2. Print all intermediate values (Galois map, decomposed digits, key switch terms)
3. Manually verify against CPU V3 implementation
4. Once minimal case works, scale up incrementally

## Files Modified

### Core Implementation
- `src/clifford_fhe_v2/backends/gpu_metal/ckks.rs` - Key switching with gadget decomposition
- `src/clifford_fhe_v2/backends/gpu_metal/rotation_keys.rs` - Rotation key generation
- `src/clifford_fhe_v2/backends/gpu_metal/rotation.rs` - Galois element computation
- `src/clifford_fhe_v2/backends/gpu_metal/shaders/rotation.metal` - Galois automorphism kernel

### Test Files
- `examples/test_rotation_debug.rs` - Coefficient-level debugging
- `examples/test_rotation_mapping.rs` - Slot mapping analysis
- `examples/test_cpu_vs_gpu_rotation.rs` - CPU vs GPU comparison ✅
- `examples/test_decompose_roundtrip.rs` - Decomposition verification ✅

## References

### Expert Guidance (Previous Session)
1. **Root cause:** Simplified single-key approach causes multiplicative noise growth
   - **Fix:** Implemented full gadget decomposition with multiple digits ✅

2. **Bit-shift bug:** Using `>> base_w` on negative `BigInt` gives wrong quotient
   - **Fix:** Use exact floor division `div_floor(&base_big)` ✅

3. **Recommended next steps:**
   - Add detailed logging to see which coefficients are correct ← **IN PROGRESS**
   - Verify Galois automorphism rotates correct coefficient indices
   - Check CKKS slot-to-coefficient mapping
   - Create noiseless single-digit test for isolation

### Key Equations

**Gadget Decomposition:**
```
c1 ≈ Σ_{t=0}^{d-1} B^t · digit_t  where |digit_t| < B/2
```

**Rotation Key:**
```
rlk0[t] = -a_t · s + e_t + B^t · s_k  (where s_k = s(X^k))
rlk1[t] = a_t
```

**Key Switching:**
```
c0_new = c0_rotated - Σ_t digit_t · rlk0[t]
c1_new = Σ_t digit_t · rlk1[t]
```

## Current Blocker

The gap between CPU (perfect) and GPU (failing) with identical parameters and Galois elements suggests a **fundamental implementation difference** in either:
1. How polynomials are represented in memory (flat RNS layout)
2. How NTT domain transformations are applied
3. How gadget decomposition digits are accumulated

Need to add detailed logging and create minimal reproducible case to isolate the exact divergence point.
