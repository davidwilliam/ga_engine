# V3 Test Fixes - Progress Report

**Date:** 2025-11-05
**Status:** BLOCKED on scale management approach clarification

---

## Summary

✅ **Successfully fixed:** All V3 compilation errors (12 errors resolved)
⚠️ **Blocked:** Implementation of expert-recommended scale=1.0 fix produces zeros
📊 **Test Status:** 172/184 passing, 9 failing, 3 ignored

---

## What We Accomplished

### 1. Fixed All V3 Compilation Errors ✅
- Fixed import errors (`encode_vec`, `decode_vec` → proper V2 CKKS API)
- Fixed type naming (`CKKSContext` → `CkksContext`)
- Fixed Plaintext type mismatches (proper encoding/decoding flow)
- Fixed RNS level matching in diagonal_mult

### 2. Identified Root Cause of 6 Test Failures ✅
- **Critical Bug:** V2's `multiply_plain` lacks scale management
- **Issue:** `result_scale = ct.scale × pt.scale` causes scale explosion (2^40 × 2^40 = 2^80)
- **Impact:** Decoded values become ≈ 10^-5 instead of expected integers
- **Affected Tests:**
  - `test_diagonal_mult_simple`
  - `test_multiply_by_constant`
  - `test_eval_sine_polynomial_simple`
  - (These cascade to 3 more tests)

### 3. Created Comprehensive Technical Documentation ✅
- **V3_TEST_FAILURES_ANALYSIS.md:** Complete analysis of all 9 failures
- **SCALE_1_0_ISSUE.md:** Specific blocker on scale=1.0 approach

---

## Current Blocker

### Scale Management Implementation

Following expert guidance to use `scale=1.0` for plaintext multipliers results in **all zeros** instead of expected values.

**What Happens:**
```rust
// Encrypt with scale=2^40
let ct = encrypt([1.0, 2.0, 3.0, 4.0], scale=2^40);

// Multiply by plaintext with scale=1.0
let pt = encode([2.0, 2.0, 2.0, 2.0], scale=1.0);
let result = ct.multiply_plain(&pt);

// Decrypt
let values = decrypt(result);
// EXPECTED: [2.0, 4.0, 6.0, 8.0]
// ACTUAL: [0.0, 0.0, 0.0, 0.0]  ❌
```

**Why It Fails:**
- With `scale=1.0`, coefficients = `round(2.0 × 1.0) = 2`
- Small integers (2, 3, 4) get lost after:
  - Canonical embedding (DFT/IDFT transformations)
  - Modular reduction with large primes (≈ 2^60)
  - NTT-based polynomial multiplication

**Attempted Solutions:**
1. ❌ Literal scale=1.0 → zeros
2. ❌ Manual coefficient division in RNS → too complex
3. ❌ Accept scale explosion → values ≈ 10^-5

---

## Remaining Failures (Brief)

### Other 3 Test Failures (Not Yet Addressed)

4. **test_mod_raise_preserves_plaintext**
   - Issue: RNS moduli mismatch after mod_raise
   - Fix: Need `mod_switch_down_to_level()` implementation

5. **test_rotation_small**
   - Issue: Rotation produces wrong first element
   - Fix: Debug Galois automorphism or key-switching

6. **test_component_extraction / test_extract_and_reassemble**
   - Issue: Slot masking not working
   - Fix: Likely fixed once rotation is correct

7. **test_needs_bootstrap_heuristic**
   - Issue: Heuristic logic inverted
   - Fix: Simple - compute `levels_left` correctly

8. **test_sin_coeffs_precomputed**
   - Issue: Test params have too few levels
   - Fix: Trivial - use different params or lower requirement

---

## What We Need from Expert

### Critical Question: Scale Management Approach

**Option A: Literal scale=1.0** (as recommended)
- **Q:** Why does this produce zeros? Is there an encoding/decoding adjustment needed?
- **Q:** Should we use a small power of 2 instead (e.g., 2^20)?

**Option B: Implement rescale_to_next first**
- Would this be simpler than making scale=1.0 work?
- Can provide skeleton implementation if preferred

**Option C: Alternative workaround**
- Is there a different approach that avoids both scale explosion and precision loss?

### Specific Technical Questions

1. **CKKS Canonical Embedding:**
   - Does encoding values with scale=1.0 preserve enough precision after DFT/IDFT?
   - Is there a minimum practical scale threshold?

2. **Middle-ground Scale:**
   - Would `pt.scale = sqrt(ct.scale)` work better?
   - Example: ct.scale=2^40, pt.scale=2^20 → result.scale=2^60

3. **Coefficient Domain:**
   - Should we be working in coefficient form differently?
   - Is the NTT multiplication the issue, or the encoding?

---

## Files Modified

### Successfully Modified:
1. **src/clifford_fhe_v2/backends/cpu_optimized/ckks.rs**
   - Fixed `multiply_polys_ntt` (removed incorrect i64/u64 casts)
   - Modified `multiply_plain` (reverted to original after scale=1.0 failed)

2. **src/clifford_fhe_v3/bootstrapping/diagonal_mult.rs**
   - Updated to use scale=1.0 (line 79)
   - Has level-matching logic (lines 81-95)

### Ready to Modify (once approach clarified):
3. **src/clifford_fhe_v3/bootstrapping/eval_mod.rs**
4. **src/clifford_fhe_v3/bootstrapping/mod_raise.rs**
5. **src/clifford_fhe_v3/bootstrapping/rotation.rs**
6. **src/clifford_fhe_v3/batched/bootstrap.rs**
7. **src/clifford_fhe_v3/bootstrapping/bootstrap_context.rs**

---

## Test Commands

### Reproduce scale=1.0 issue:
```bash
cargo test --lib --features f64,nd,v2 --no-default-features test_multiply_plain_simple
```

### Test diagonal_mult (depends on multiply_plain fix):
```bash
cargo test --lib --features f64,nd,v2,v3 --no-default-features test_diagonal_mult_simple
```

### Run all V3 tests:
```bash
cargo test --lib --features f64,nd,v2,v3 --no-default-features clifford_fhe_v3
```

---

## Next Steps (Once Unblocked)

1. ✅ Implement confirmed scale management approach
2. ⏳ Test multiply_plain with new approach
3. ⏳ Update all V3 callers (diagonal_mult, eval_mod)
4. ⏳ Implement `mod_switch_down_to_level` for mod_raise
5. ⏳ Debug and fix rotation implementation
6. ⏳ Fix heuristic logic and test params (trivial)
7. ⏳ Validate all 9 tests pass

**Estimated Time:** 1-2 hours once scale approach is clarified

---

## Code Snippets for Expert Review

### Current multiply_plain (reverted to original):
```rust
pub fn multiply_plain(&self, pt: &Plaintext, ckks_ctx: &CkksContext) -> Self {
    assert_eq!(self.n, pt.n, "Dimensions must match");
    assert_eq!(self.level, pt.level, "Levels must match");

    let moduli: Vec<u64> = ckks_ctx.params.moduli[..=self.level].to_vec();

    let new_c0 = ckks_ctx.multiply_polys_ntt(&self.c0, &pt.coeffs, &moduli);
    let new_c1 = ckks_ctx.multiply_polys_ntt(&self.c1, &pt.coeffs, &moduli);

    let new_scale = self.scale * pt.scale;  // Scale explosion here

    Self::new(new_c0, new_c1, self.level, new_scale)
}
```

### Current diagonal_mult caller:
```rust
pub fn diagonal_mult(...) -> Result<Ciphertext, String> {
    // Use scale=1.0 (produces zeros)
    let mut pt_diagonal = Plaintext::encode(diagonal, 1.0, params);

    // Level matching (works correctly)
    if ct.level < params.moduli.len() - 1 {
        for coeff in &mut pt_diagonal.coeffs {
            coeff.values.truncate(ct.level + 1);
            coeff.moduli = params.moduli[..=ct.level].to_vec();
        }
        pt_diagonal.level = ct.level;
    }

    let ckks_ctx = CkksContext::new(params.clone());
    Ok(ct.multiply_plain(&pt_diagonal, &ckks_ctx))
}
```

---

## Additional Context

- **Environment:** macOS (M1), Rust stable
- **Project:** Clifford Algebra-based FHE with CKKS
- **V2:** Production-ready for basic ops (172 tests passing)
- **V3:** Advanced features (bootstrapping, rotation, SIMD batching)
- **Key Constraint:** No rescale_to_next operation in V2

---

**Recommendation:** Please review SCALE_1_0_ISSUE.md and advise on the scale management approach. Once clarified, the remaining fixes are straightforward and can be completed quickly.

---

**End of Progress Report**
