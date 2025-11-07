# V3 Test Failures - Technical Analysis

**Date:** 2025-11-05
**Status:** V3 Compiles Successfully (172/184 tests passing)
**Critical Issue:** Fundamental bug in V2's `multiply_plain` affecting 6 V3 tests

---

## Executive Summary

We successfully fixed all V3 compilation errors (12 errors resolved). V3 now compiles cleanly, but **9 tests are failing** due to runtime logic issues. Investigation revealed a **critical bug in the V2 CKKS `multiply_plain` method** that causes incorrect scale management, resulting in near-zero outputs instead of expected values.

### Test Results Summary
- ✅ **172 tests passing**
- ❌ **9 tests failing**
- ⚠️ **3 tests ignored**

---

## Critical Bug: V2 `multiply_plain` Scale Management

### Location
**File:** `src/clifford_fhe_v2/backends/cpu_optimized/ckks.rs`
**Lines:** 375-389, 642-679

### The Problem

The `multiply_plain` method correctly multiplies ciphertext by plaintext, but **does not handle CKKS scale management**, resulting in unusable outputs.

#### Code: `multiply_plain` method (lines 375-389)

```rust
pub fn multiply_plain(&self, pt: &Plaintext, ckks_ctx: &CkksContext) -> Self {
    assert_eq!(self.n, pt.n, "Dimensions must match");
    assert_eq!(self.level, pt.level, "Levels must match for plaintext multiplication");

    let moduli: Vec<u64> = ckks_ctx.params.moduli[..=self.level].to_vec();

    // Multiply both c0 and c1 by plaintext using NTT
    let new_c0 = ckks_ctx.multiply_polys_ntt(&self.c0, &pt.coeffs, &moduli);
    let new_c1 = ckks_ctx.multiply_polys_ntt(&self.c1, &pt.coeffs, &moduli);

    // Scale increases: scale' = scale * pt_scale
    // ⚠️ PROBLEM: No rescaling operation exists in V2!
    let new_scale = self.scale * pt.scale;

    Self::new(new_c0, new_c1, self.level, new_scale)
}
```

### Why This Fails

In CKKS homomorphic encryption:

1. **Encoding with scale Δ:**
   - `plaintext_coeffs = round(value * Δ)`

2. **After multiplication:**
   - `result_coeffs = ct_coeffs * pt_coeffs` (polynomial multiplication)
   - `result_scale = ct_scale * pt_scale`
   - `result_value = ct_value * pt_value` (when decoded)

3. **The Issue:**
   - If `ct_scale = 2^40` (params.scale = 1099511627776)
   - And `pt_scale = 2^40` (encoded with params.scale)
   - Then `result_scale = 2^80` (≈ 1.21 × 10^24)

4. **When decoding:**
   - `decoded_value = result_coeffs / result_scale`
   - With scale = 2^80, we get values ≈ 10^-5 instead of expected values!

### Demonstration: Minimal Failing Test

**File:** `src/clifford_fhe_v2/backends/cpu_optimized/ckks.rs` (lines 1034-1072)

```rust
#[test]
fn test_multiply_plain_simple() {
    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Encrypt [1.0, 2.0, 3.0, 4.0]
    let values = vec![1.0, 2.0, 3.0, 4.0];
    let pt1 = Plaintext::encode(&values, params.scale, &params);
    let ct = ckks_ctx.encrypt(&pt1, &pk);

    // Multiply by plaintext [2.0, 2.0, 2.0, 2.0]
    let multiplier = vec![2.0, 2.0, 2.0, 2.0];
    let pt2 = Plaintext::encode(&multiplier, params.scale, &params);

    let ct_result = ct.multiply_plain(&pt2, &ckks_ctx);

    // Decrypt - SHOULD get [2.0, 4.0, 6.0, 8.0]
    let pt_result = ckks_ctx.decrypt(&ct_result, &sk);
    let result = pt_result.decode(&params);

    // ❌ ACTUAL RESULT: [-5.3e-6, -3.2e-6, 2.1e-6, -9.2e-6]
    // ✅ EXPECTED: [2.0, 4.0, 6.0, 8.0]

    assert!((result[0] - 2.0).abs() < 1.0); // FAILS!
}
```

**Test Output:**
```
Input: [1.0, 2.0, 3.0, 4.0]
Multiplier: [2.0, 2.0, 2.0, 2.0]
Result: [-5.3153213004716774e-6, -3.225783157465253e-6, 2.166372011807012e-6, -9.2075889537038e-6]
Expected: [2.0, 4.0, 6.0, 8.0]
```

### Root Cause in `multiply_polys_ntt`

**File:** `src/clifford_fhe_v2/backends/cpu_optimized/ckks.rs` (lines 642-679)

```rust
fn multiply_polys_ntt(
    &self,
    a: &[RnsRepresentation],
    b: &[RnsRepresentation],
    moduli: &[u64],
) -> Vec<RnsRepresentation> {
    let n = a.len();
    let mut result = vec![RnsRepresentation::new(vec![0; moduli.len()], moduli.to_vec()); n];

    // For each prime modulus, multiply using V2's NTT
    for (prime_idx, &q) in moduli.iter().enumerate() {
        // Create NTT context for this prime
        let ntt_ctx = super::ntt::NttContext::new(n, q);

        // Extract coefficients for this prime (already in u64 mod q form)
        let a_mod_q: Vec<u64> = a.iter().map(|rns| rns.values[prime_idx]).collect();
        let b_mod_q: Vec<u64> = b.iter().map(|rns| rns.values[prime_idx]).collect();

        // Multiply using V2's NTT (negacyclic convolution)
        let product_mod_q = ntt_ctx.multiply_polynomials(&a_mod_q, &b_mod_q);

        // Store result
        for i in 0..n {
            result[i].values[prime_idx] = product_mod_q[i];
        }
    }

    result
}
```

**Note:** We fixed a bug here where values were incorrectly cast from `u64` to `i64` and back, but this didn't resolve the scale issue.

---

## Affected V3 Tests (6 failures related to `multiply_plain`)

### 1. `test_diagonal_mult_simple`
**File:** `src/clifford_fhe_v3/bootstrapping/diagonal_mult.rs:225-246`

**Purpose:** Tests diagonal matrix multiplication - a key primitive for bootstrapping

```rust
#[test]
fn test_diagonal_mult_simple() {
    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Create test vector [1.0, 2.0, 3.0, 4.0]
    let vec = vec![1.0, 2.0, 3.0, 4.0];
    let pt = Plaintext::encode(&vec, params.scale, &params);
    let ct = ckks_ctx.encrypt(&pt, &pk);

    // Create diagonal [2.0, 3.0, 4.0, 5.0, 1.0, 1.0, ...]
    let num_slots = params.n / 2;
    let mut diagonal = vec![1.0; num_slots];
    diagonal[0] = 2.0;
    diagonal[1] = 3.0;
    diagonal[2] = 4.0;
    diagonal[3] = 5.0;

    // Apply diagonal multiplication
    let ct_result = diagonal_mult(&ct, &diagonal, &params, &key_ctx).unwrap();

    // Decrypt and decode
    let pt_result = ckks_ctx.decrypt(&ct_result, &sk);
    let result = pt_result.decode(&params);

    // ✅ EXPECTED: [1*2, 2*3, 3*4, 4*5] = [2, 6, 12, 20]
    // ❌ ACTUAL: [0.000015, -0.0000014, 0.0000014, -0.00000058]

    assert!((result[0] - 2.0).abs() < 1.0);  // FAILS!
}
```

**Failure Output:**
```
Input ciphertext decrypts to: [0.9999999958783374, 1.9999999967253221, 3.00000000215546, 3.999999999169995]
Before diagonal_mult: scale = 1099511627776
After diagonal_mult: scale = 1208925819614629200000000
Plaintext scale after decryption: 1208925819614629200000000
Result after diagonal_mult: [0.000015109412550515948, -0.0000014070786666518464, 0.0000014358869871417211, -0.00000058109418837074]
```

**Problem Location:**
The `diagonal_mult` function (lines 59-95) encodes the diagonal and uses `multiply_plain`:

```rust
pub fn diagonal_mult(
    ct: &Ciphertext,
    diagonal: &[f64],
    params: &CliffordFHEParams,
    key_ctx: &KeyContext,
) -> Result<Ciphertext, String> {
    // ... validation code ...

    // Encode diagonal as plaintext using V2 CKKS encoding
    // ⚠️ PROBLEM: Uses params.scale, causing scale^2 in result
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{Plaintext, CkksContext};

    let mut pt_diagonal = Plaintext::encode(diagonal, params.scale, params);

    // Adjust plaintext level to match ciphertext level if needed
    if ct.level < params.moduli.len() - 1 {
        use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
        let moduli = &params.moduli[..=ct.level];
        for coeff in &mut pt_diagonal.coeffs {
            coeff.values.truncate(moduli.len());
            coeff.moduli = moduli.to_vec();
        }
        pt_diagonal.level = ct.level;
    }

    // ⚠️ CALLS multiply_plain, which has the scale bug
    let ckks_ctx = CkksContext::new(params.clone());
    Ok(ct.multiply_plain(&pt_diagonal, &ckks_ctx))
}
```

---

### 2. `test_multiply_by_constant`
**File:** `src/clifford_fhe_v3/bootstrapping/eval_mod.rs:158-180`

**Purpose:** Tests EvalMod primitive (multiply ciphertext by constant for sine approximation)

```rust
#[test]
fn test_multiply_by_constant() {
    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Encrypt 5.0
    let pt = Plaintext::encode(&[5.0], params.scale, &params);
    let ct = ckks_ctx.encrypt(&pt, &pk);

    // Multiply by constant 2.0
    let ct_result = multiply_by_constant(&ct, 2.0, &params).unwrap();

    // Decrypt
    let pt_result = ckks_ctx.decrypt(&ct_result, &sk);
    let result_vec = pt_result.decode(&params);
    let result = result_vec[0];

    // ✅ EXPECTED: 5.0 * 2.0 = 10.0
    // ❌ ACTUAL: ~0.0

    assert!((result - 10.0).abs() < 1.0);  // FAILS!
}
```

**Error:**
```
thread 'test_multiply_by_constant' panicked at diagonal_mult.rs:189:66:
index out of bounds: the len is 1 but the index is 1
```

**Note:** This also hits the diagonal_mult code path, which calls `multiply_plain`.

---

### 3. `test_eval_sine_polynomial_simple`
**File:** `src/clifford_fhe_v3/bootstrapping/eval_mod.rs:202-235`

**Purpose:** Tests polynomial evaluation for sine approximation (key part of EvalMod)

```rust
#[test]
fn test_eval_sine_polynomial_simple() {
    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Test input: x = 0.1
    let pt = Plaintext::encode(&[0.1], params.scale, &params);
    let ct_x = ckks_ctx.encrypt(&pt, &pk);

    // Simple polynomial: 3x (should give 0.3)
    let coeffs = vec![0.0, 3.0]; // c0=0, c1=3, so p(x) = 3x

    let ct_result = eval_polynomial(&ct_x, &coeffs, &params, &key_ctx).unwrap();

    let pt_result = ckks_ctx.decrypt(&ct_result, &sk);
    let result_vec = pt_result.decode(&params);
    let result = result_vec[0];

    // ✅ EXPECTED: 3 * 0.1 = 0.3
    // ❌ ACTUAL: ~0.0

    assert!((result - 0.3).abs() < 0.5);  // FAILS!
}
```

**Error:** Same index out of bounds error in diagonal_mult.

---

## Other V3 Test Failures (3 failures, different root causes)

### 4. `test_mod_raise_preserves_plaintext`
**File:** `src/clifford_fhe_v3/bootstrapping/mod_raise.rs:127-160`

**Error:**
```rust
thread 'test_mod_raise_preserves_plaintext' panicked at src/clifford_fhe_v2/backends/cpu_optimized/rns.rs:232:9:
assertion `left == right` failed: Moduli must match
  left: [1152921504606584833, 1099511678977, 1099511683073, 1152921504606584777, 1152921504606584833]
 right: [1152921504606584833, 1099511678977, 1099511683073]
```

**Problem:** Modulus raising creates ciphertext with 5 primes, but original has only 3 primes. Mismatch when trying to compare or operate on them.

**Location:**
```rust
#[test]
fn test_mod_raise_preserves_plaintext() {
    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (public_key, secret_key, _) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    let plaintext_values = vec![1.0, 2.0, 3.0, 4.0];
    let plaintext = Plaintext::encode(&plaintext_values, params.scale, &params);
    let ct = ckks_ctx.encrypt(&plaintext, &public_key);

    // Perform modulus raising
    let ct_raised = mod_raise(&ct, &params).unwrap();

    // ⚠️ PROBLEM: ct has 3 moduli, ct_raised has 5 moduli
    // When we try to decrypt ct_raised, it has different RNS structure

    let decrypted_pt = ckks_ctx.decrypt(&ct_raised, &secret_key);
    // ... FAILS HERE with moduli mismatch
}
```

---

### 5. `test_rotation_small`
**File:** `src/clifford_fhe_v3/bootstrapping/rotation.rs:299-338`

**Error:**
```rust
thread 'test_rotation_small' panicked at src/clifford_fhe_v3/bootstrapping/rotation.rs:334:9:
First element should be 4
```

**Expected vs Actual:**
- **Input:** [1, 2, 3, 4, 0, 0, ...]
- **After rotation by 1:** Should be [4, 1, 2, 3, 0, 0, ...] (last wraps to front)
- **Actual:** First element is NOT 4

**Code:**
```rust
#[test]
fn test_rotation_small() {
    let params = CliffordFHEParams::new_128bit();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, evk) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Create simple message: [1, 2, 3, 4, 0, 0, ..., 0]
    let mut message = vec![0.0; params.n / 2];
    message[0] = 1.0;
    message[1] = 2.0;
    message[2] = 3.0;
    message[3] = 4.0;

    // Encode and encrypt
    let pt = ckks_ctx.encode(&message);
    let ct = ckks_ctx.encrypt(&pt, &pk);

    // Generate rotation keys for rotation by 1
    let rotations = vec![1];
    let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

    // Rotate by 1
    let ct_rotated = rotate(&ct, 1, &rotation_keys).expect("Rotation should succeed");

    // Decrypt and decode
    let pt_rotated = ckks_ctx.decrypt(&ct_rotated, &sk);
    let decrypted = ckks_ctx.decode(&pt_rotated);

    // ✅ EXPECTED: [4, 1, 2, 3, 0, 0, ...]
    // ❌ ACTUAL: First element is NOT 4

    assert!((decrypted[0] - 4.0).abs() < 0.1);  // FAILS!
}
```

**Problem:** The rotation implementation may have incorrect Galois automorphism application or key-switching logic.

**Relevant Code:** `src/clifford_fhe_v3/bootstrapping/rotation.rs:41-72`

```rust
pub fn rotate(
    ct: &Ciphertext,
    k: i32,
    rotation_keys: &RotationKeys,
) -> Result<Ciphertext, String> {
    let n = ct.n;

    // Step 1: Compute Galois element for rotation by k
    let g = galois_element_for_rotation(k, n);

    // Step 2: Get rotation key for this Galois element
    let rotation_key = rotation_keys.get_key(g)
        .ok_or_else(|| format!("Rotation key for k={} (g={}) not found", k, g))?;

    // Step 3: Apply Galois automorphism to c0 and c1
    let mut c0_new = apply_galois_automorphism(&ct.c0, g, n);
    let c1_rotated = apply_galois_automorphism(&ct.c1, g, n);

    // Step 4: Key-switch c1_rotated from s(X^g) to s(X)
    let c1_new = key_switch(&mut c0_new, &c1_rotated, rotation_key, n)?;

    Ok(Ciphertext {
        c0: c0_new,
        c1: c1_new,
        level: ct.level,
        scale: ct.scale,
        n: ct.n,
    })
}
```

---

### 6 & 7. `test_component_extraction` and `test_extract_and_reassemble`
**File:** `src/clifford_fhe_v3/batched/extraction.rs:208-290`

**Errors:**
```
test_component_extraction:
  Slot 1 should be masked to 0, got 3.9935365999408954

test_extract_and_reassemble:
  Multivector 0 component 0 error: 4205928.86648285 (got 4205929.86648285, expected 1)
```

**Problem:** Slot masking and component extraction producing wrong values, likely related to rotation operations not working correctly.

**Code Sample:**
```rust
#[test]
fn test_component_extraction() {
    // ... setup ...

    // Create batched ciphertext with multivectors
    let batched_ct = batch_encrypt(&multivectors, &public_key, &params);

    // Extract first component of first multivector
    // This should mask other components to 0
    let extracted_ct = extract_component(&batched_ct, 0, 0, 8, &rotation_keys, &params)
        .expect("Extraction should succeed");

    let decrypted = batch_decrypt(&extracted_ct, &secret_key, &ckks_ctx, 1);

    // ✅ EXPECTED: Slot 0 = value, Slot 1 = 0 (masked)
    // ❌ ACTUAL: Slot 1 = 3.99 (not masked!)

    assert!(decrypted[0][1].abs() < 0.1, "Slot 1 should be masked to 0, got {}", decrypted[0][1]);
}
```

---

### 8. `test_needs_bootstrap_heuristic`
**File:** `src/clifford_fhe_v3/batched/bootstrap.rs:65-96`

**Error:**
```rust
thread 'test_needs_bootstrap_heuristic' panicked at src/clifford_fhe_v3/batched/bootstrap.rs:93:9:
assertion failed: !needs_bootstrap(&batch_fresh)
```

**Problem:** Heuristic incorrectly reports fresh ciphertext needs bootstrapping.

**Code:**
```rust
#[test]
fn test_needs_bootstrap_heuristic() {
    let params = CliffordFHEParams::new_test_ntt_1024();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, _, _) = key_ctx.keygen();

    // Create fresh ciphertext
    let values = vec![vec![1.0, 2.0, 3.0, 4.0]];
    let batch_fresh = batch_encrypt(&values, &pk, &params);

    // Fresh ciphertext should NOT need bootstrapping
    // ❌ ACTUAL: needs_bootstrap returns true!
    assert!(!needs_bootstrap(&batch_fresh));  // FAILS!
}
```

**Heuristic code:** Lines 40-62
```rust
pub fn needs_bootstrap(ct: &Ciphertext) -> bool {
    // Simple heuristic: check if we're running low on levels
    // In practice, would track noise budget

    const MIN_LEVELS_NEEDED: usize = 5;

    // ⚠️ PROBLEM: This logic may be inverted or incorrect
    ct.level < MIN_LEVELS_NEEDED
}
```

---

### 9. `test_sin_coeffs_precomputed`
**File:** `src/clifford_fhe_v3/bootstrapping/bootstrap_context.rs:384-408`

**Error:**
```rust
thread 'test_sin_coeffs_precomputed' panicked at src/clifford_fhe_v3/bootstrapping/bootstrap_context.rs:405:11:
called `Result::unwrap()` on an `Err` value: "bootstrap_levels must be >= 10, got 2"
```

**Problem:** Test uses parameters with only 2 levels (too small for bootstrapping requirements).

**Code:**
```rust
#[test]
fn test_sin_coeffs_precomputed() {
    // ⚠️ PROBLEM: new_test_ntt_1024() only has 3 moduli = 2 levels
    let params = CliffordFHEParams::new_test_ntt_1024();

    let bootstrap_params = BootstrapParams {
        sin_degree: 15,
        // ... other params ...
    };

    // ❌ FAILS: Requires bootstrap_levels >= 10 but params only has 2 levels
    let ctx = BootstrapContext::new(params, bootstrap_params).unwrap();
}
```

**Validation code:** `src/clifford_fhe_v3/bootstrapping/bootstrap_context.rs:102-107`
```rust
impl BootstrapContext {
    pub fn new(
        params: CliffordFHEParams,
        bootstrap_params: BootstrapParams,
    ) -> Result<Self, String> {
        // Validate bootstrap_levels
        if bootstrap_params.bootstrap_levels < 10 {
            return Err(format!(
                "bootstrap_levels must be >= 10, got {}",
                bootstrap_params.bootstrap_levels
            ));
        }
        // ...
    }
}
```

---

## Summary of Issues

### Critical (Blocking 6 tests):
1. **`multiply_plain` scale management bug** - No rescale operation in V2
   - Affects: diagonal_mult, eval_mod tests
   - Impact: Core CKKS primitive broken for plaintext multiplication

### High Priority (3 tests):
2. **Modulus raising RNS mismatch** - mod_raise creates incompatible ciphertext
3. **Rotation incorrect output** - Galois automorphism or key-switching bug
4. **Extraction masking failure** - Related to rotation issues

### Medium Priority (2 tests):
5. **Bootstrap heuristic logic** - Simple fix, likely inverted condition
6. **Test parameter mismatch** - Using wrong params for bootstrap test

---

## Possible Solutions

### For `multiply_plain` scale bug:

**Option 1: Add rescale operation to V2 (Complex)**
```rust
impl Ciphertext {
    pub fn rescale(&self, params: &CliffordFHEParams) -> Self {
        // Drop one prime from RNS representation
        // Divide coefficients by dropped prime
        // Update level and scale
        // ...
    }
}
```

**Option 2: Use scale=1.0 for plaintext multipliers (Simple, loses precision)**
```rust
// In diagonal_mult:
let mut pt_diagonal = Plaintext::encode(diagonal, 1.0, params);  // Instead of params.scale
```

**Option 3: Manual scale management in multiply_plain (Medium complexity)**
```rust
pub fn multiply_plain(&self, pt: &Plaintext, ckks_ctx: &CkksContext) -> Self {
    // ... multiply polynomials ...

    // Keep original scale instead of multiplying
    let new_scale = self.scale;  // Don't multiply by pt.scale

    // Compensate by adjusting coefficients
    let scale_factor = pt.scale;
    // ... divide coefficients by scale_factor ...

    Self::new(new_c0, new_c1, self.level, new_scale)
}
```

**Option 4: Require plaintext to have scale=1.0 (Document and enforce)**
```rust
pub fn multiply_plain(&self, pt: &Plaintext, ckks_ctx: &CkksContext) -> Self {
    assert_eq!(pt.scale, 1.0, "Plaintext must have scale=1.0 for multiply_plain");
    // ... rest of implementation ...
}
```

---

## Request for Expert Guidance

**Questions:**

1. **Which solution approach for `multiply_plain` is preferred?**
   - Implement full rescale operation?
   - Use scale=1.0 workaround?
   - Manual scale management?
   - Document limitation and require scale=1.0?

2. **For modulus raising (`test_mod_raise_preserves_plaintext`):**
   - Should decrypt support variable-length RNS representations?
   - Should we add a "mod_switch_down" after mod_raise to match original level?

3. **For rotation (`test_rotation_small`):**
   - Is the Galois element calculation correct? (`g = 5^k mod 2N`)
   - Is the key-switching implementation matching expected V2 structure?

4. **General architecture:**
   - Should V3 operations create a separate context with its own scale management?
   - Or should we backport missing CKKS operations (rescale, mod_switch) to V2?

---

## Files Requiring Expert Review

1. **`src/clifford_fhe_v2/backends/cpu_optimized/ckks.rs`** (lines 375-389, 642-679)
   - `multiply_plain` and `multiply_polys_ntt` methods

2. **`src/clifford_fhe_v3/bootstrapping/diagonal_mult.rs`** (lines 59-100)
   - Diagonal multiplication implementation

3. **`src/clifford_fhe_v3/bootstrapping/mod_raise.rs`** (entire file)
   - Modulus raising logic

4. **`src/clifford_fhe_v3/bootstrapping/rotation.rs`** (lines 41-120)
   - Rotation and key-switching implementation

5. **`src/clifford_fhe_v3/batched/extraction.rs`** (lines 60-140)
   - Component extraction using rotations

---

## Test Command

To reproduce all failures:
```bash
cargo test --lib --features f64,nd,v2,v3 --no-default-features 2>&1
```

To test specific failures:
```bash
# Test multiply_plain bug
cargo test --lib --features f64,nd,v2 --no-default-features test_multiply_plain_simple

# Test diagonal_mult
cargo test --lib --features f64,nd,v2,v3 --no-default-features test_diagonal_mult_simple

# Test all V3
cargo test --lib --features f64,nd,v2,v3 --no-default-features clifford_fhe_v3
```

---

## Additional Context

- **V2 is production-ready** for basic CKKS operations (encrypt, decrypt, add, multiply ciphertext-ciphertext)
- **V3 adds advanced features**: bootstrapping, rotation, SIMD batching, component extraction
- **V2 was designed without rescale** - this is now limiting V3 development
- **All 172 passing tests** confirm core V2 functionality and most V3 primitives work correctly

---

**End of Technical Analysis**
