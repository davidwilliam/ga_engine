# CKKS SIMD Slot Operations for Clifford-FHE

**Date**: November 1, 2025
**Approach**: Option A - Proper SIMD Implementation
**Goal**: Production-quality, respected implementation

---

## Background: CKKS SIMD Slots

### What Are SIMD Slots?

In CKKS, the polynomial ring R = Z[x]/(Φ_M(x)) can be viewed as having N/2 "slots" where:
- Each slot holds a complex number
- Operations on ciphertext act on ALL slots in parallel (SIMD = Single Instruction Multiple Data)
- This is why CKKS is efficient for batch operations

### Mathematical Foundation

**Polynomial Ring Decomposition**:
```
R ≅ C[x]/(Φ_M(x)) ≅ C^(N/2)  (via Chinese Remainder Theorem)
```

Where:
- Φ_M(x) is the M-th cyclotomic polynomial
- M = 2N (for CKKS with dimension N)
- The ring decomposes into N/2 complex "slots"

**Key Insight**: This is NOT the same as coefficient packing!

---

## SIMD Slot Encoding

### Standard CKKS Encoding

**Goal**: Encode complex vector [z₀, z₁, ..., z_{N/2-1}] into polynomial p(x)

**Method**: Use inverse FFT-like transform
```
p(x) = Σᵢ aᵢ·xⁱ  where coefficients computed via:

aᵢ = Σⱼ zⱼ · ωᴹ^(i·(2j+1))

where ωᴹ = e^(2πi/M) is the M-th root of unity
```

**Decoding**: Apply forward transform to get slots back

### For Clifford-FHE

**Our requirement**: Encode 8-component multivector [a₀, a₁, ..., a₇]

**Strategy**:
1. Treat each component as a real number (not complex)
2. Encode into first 8 slots: [a₀, a₁, ..., a₇, 0, 0, ...]
3. Use SIMD operations to manipulate slots
4. Decode to get result multivector

**Advantages**:
- Can use up to N/2 slots (for N=8192, that's 4096 slots!)
- Could batch multiple multivectors
- Proper CKKS slot operations work correctly

---

## Galois Automorphisms for Rotation

### What Are They?

A Galois automorphism is a ring homomorphism that permutes slots:
```
σₖ: R → R
σₖ(x) = x^k  where k is odd and gcd(k, M) = 1
```

**Effect on slots**:
- Rotates slots by specific amounts
- Different k values give different rotation patterns

### Standard CKKS Rotations

For ring dimension N, common automorphisms:
```
k = 5:  Rotates slots left by 1
k = 5²: Rotates slots left by 2
k = 5⁻¹: Rotates slots right by 1
...
```

**General pattern**: k = 5^r rotates by r positions (for power-of-2 N)

### Rotation Keys

For each rotation amount r, we need a rotation key:
```
rotkey_r = Enc(s(x^(5^r)))  using public key
```

This allows converting:
```
ct = (c₀, c₁) encrypting m(x)
↓ apply automorphism
ct' = (c₀(x^(5^r)), c₁(x^(5^r))) encrypting m(x^(5^r))
↓ use rotation key to fix encryption
ct_rotated encrypting m(x^(5^r))  [slots rotated by r]
```

---

## Implementation Plan

### Phase 1: Slot Encoding (New Module)

**File**: `src/clifford_fhe/slot_encoding.rs`

**Functions**:
```rust
/// Encode multivector into SIMD slots
pub fn encode_multivector_slots(
    mv: &[f64; 8],
    scale: f64,
    n: usize,
) -> Vec<Complex<f64>>

/// Decode SIMD slots back to multivector
pub fn decode_multivector_slots(
    slots: &[Complex<f64>],
) -> [f64; 8]

/// Compute FFT-like transform for slot → coefficient
fn slots_to_coefficients(
    slots: &[Complex<f64>],
    n: usize,
) -> Vec<i64>

/// Compute inverse transform for coefficient → slot
fn coefficients_to_slots(
    coeffs: &[i64],
    scale: f64,
    n: usize,
) -> Vec<Complex<f64>>
```

**Mathematics**:
```
M = 2N
ωᴹ = e^(2πi/M)  (M-th root of unity)

Slots → Coefficients:
  a[i] = Σⱼ slot[j] · ωᴹ^(i·(2j+1))

Coefficients → Slots:
  slot[j] = (1/N) · Σᵢ a[i] · ωᴹ^(-i·(2j+1))
```

### Phase 2: Galois Automorphisms

**File**: `src/clifford_fhe/automorphisms.rs`

**Functions**:
```rust
/// Apply Galois automorphism to polynomial
pub fn apply_automorphism(
    poly: &[i64],
    k: usize,  // k must be odd, gcd(k, 2N) = 1
    n: usize,
) -> Vec<i64>

/// Get automorphism index for rotation by r slots
pub fn rotation_to_automorphism(r: isize, n: usize) -> usize

/// Check if k is valid automorphism index
pub fn is_valid_automorphism(k: usize, n: usize) -> bool
```

**Implementation**:
```rust
fn apply_automorphism(poly: &[i64], k: usize, n: usize) -> Vec<i64> {
    let mut result = vec![0i64; n];

    for i in 0..n {
        let new_idx = (i * k) % (2 * n);

        if new_idx < n {
            result[new_idx] = poly[i];
        } else {
            // Negacyclic: x^N = -1
            result[new_idx % n] = -poly[i];
        }
    }

    result
}
```

**Key mapping** (for power-of-2 N):
- Rotate left by r: k = 5^r mod M
- Rotate right by r: k = 5^(-r) mod M

### Phase 3: Update Rotation Keys

**File**: `src/clifford_fhe/keys.rs` (update)

**Changes**:
```rust
pub struct RotationKey {
    /// Maps automorphism index k to rotation key
    pub keys: HashMap<usize, (Vec<i64>, Vec<i64>)>,
    pub n: usize,
}

pub fn generate_rotation_keys_simd(
    sk: &SecretKey,
    rotation_amounts: &[isize],  // Can be negative (right rotation)
    params: &CliffordFHEParams,
) -> RotationKey {
    let mut keys = HashMap::new();

    for &r in rotation_amounts {
        let k = rotation_to_automorphism(r, params.n);

        // Compute s(x^k)
        let s_automorphed = apply_automorphism(&sk.coeffs, k, params.n);

        // Generate key encrypting s(x^k)
        let rot_key = generate_key_for_automorphism(&s_automorphed, sk, params);

        keys.insert(k, rot_key);
    }

    RotationKey { keys, n: params.n }
}
```

### Phase 4: Update CKKS Rotation

**File**: `src/clifford_fhe/ckks.rs` (update)

**Changes**:
```rust
pub fn rotate_slots(
    ct: &Ciphertext,
    rotation_amount: isize,  // positive = left, negative = right
    rotk: &RotationKey,
    params: &CliffordFHEParams,
) -> Ciphertext {
    let n = ct.n;
    let k = rotation_to_automorphism(rotation_amount, n);

    // Apply automorphism to both components
    let c0_auto = apply_automorphism(&ct.c0, k, n);
    let c1_auto = apply_automorphism(&ct.c1, k, n);

    // Get rotation key for this automorphism
    let (rotkey_0, rotkey_1) = rotk.keys.get(&k)
        .expect("Rotation key not found");

    // Apply key switching
    let c0_new = add_poly(&c0_auto,
        &mul_poly(&c1_auto, rotkey_0, params), params);
    let c1_new = mul_poly(&c1_auto, rotkey_1, params);

    Ciphertext::new(c0_new, c1_new, ct.level, ct.scale)
}
```

### Phase 5: Slot-Based Component Extraction

**File**: `src/clifford_fhe/slot_operations.rs` (new)

**Functions**:
```rust
/// Extract slot i from ciphertext
pub fn extract_slot(
    ct: &Ciphertext,
    slot_index: usize,
    rotk: &RotationKey,
    params: &CliffordFHEParams,
) -> Ciphertext {
    // Strategy:
    // 1. Rotate so slot i is at position 0
    // 2. Multiply by mask [1, 0, 0, ...]
    // 3. Rotate back

    let n = params.n;

    // Rotate to position 0
    let ct_rotated = if slot_index > 0 {
        rotate_slots(ct, -(slot_index as isize), rotk, params)
    } else {
        ct.clone()
    };

    // Create mask polynomial for slot 0
    let mask_slots = {
        let mut s = vec![Complex::new(0.0, 0.0); n/2];
        s[0] = Complex::new(1.0, 0.0);
        s
    };
    let mask_poly = slots_to_coefficients(&mask_slots, n);
    let mask_pt = Plaintext::new(mask_poly, params.scale);

    // Multiply by mask (zeros all slots except 0)
    let ct_masked = multiply_by_plaintext(&ct_rotated, &mask_pt, params);

    // Rotate back
    if slot_index > 0 {
        rotate_slots(&ct_masked, slot_index as isize, rotk, params)
    } else {
        ct_masked
    }
}

/// Place value at specific slot
pub fn place_at_slot(
    ct: &Ciphertext,
    slot_index: usize,
    rotk: &RotationKey,
    params: &CliffordFHEParams,
) -> Ciphertext {
    // Extract slot 0, rotate to target position
    let ct_at_zero = extract_slot(ct, 0, rotk, params);
    if slot_index > 0 {
        rotate_slots(&ct_at_zero, slot_index as isize, rotk, params)
    } else {
        ct_at_zero
    }
}
```

### Phase 6: Update Geometric Product

**File**: `src/clifford_fhe/geometric_product.rs` (rewrite)

**Strategy**:
```rust
pub fn geometric_product_homomorphic_simd(
    ct_a: &Ciphertext,
    ct_b: &Ciphertext,
    evk: &EvaluationKey,
    rotk: &RotationKey,
    params: &CliffordFHEParams,
) -> Ciphertext {
    let structure = StructureConstants::new_cl30();

    // Build result by accumulating into each slot
    let mut result: Option<Ciphertext> = None;

    for target_slot in 0..8 {
        let products = structure.get_products_for(target_slot);

        for &(coeff, _target, src_a, src_b) in products {
            // Extract slots from operands
            let ct_a_i = extract_slot(ct_a, src_a, rotk, params);
            let ct_b_j = extract_slot(ct_b, src_b, rotk, params);

            // Both values now in slot 0, multiply them
            let ct_product = multiply(&ct_a_i, &ct_b_j, evk, params);

            // Move result to target slot
            let ct_at_target = if target_slot > 0 {
                rotate_slots(&ct_product, target_slot as isize, rotk, params)
            } else {
                ct_product
            };

            // Apply coefficient
            let ct_scaled = if coeff == -1 {
                negate(&ct_at_target, params)
            } else {
                ct_at_target
            };

            // Accumulate
            result = Some(match result {
                None => ct_scaled,
                Some(acc) => add(&acc, &ct_scaled, params),
            });
        }
    }

    result.unwrap()
}
```

---

## Implementation Order

### Day 1: Foundation
1. ✅ Create `slot_encoding.rs`
2. ✅ Implement FFT-like transforms (slots ↔ coefficients)
3. ✅ Test encoding/decoding roundtrip
4. ✅ Create `automorphisms.rs`
5. ✅ Implement Galois automorphism application
6. ✅ Test automorphism correctness

### Day 2: Integration
1. ✅ Update `keys.rs` for SIMD rotation keys
2. ✅ Update `ckks.rs` for slot rotation
3. ✅ Test rotation on encrypted data
4. ✅ Create `slot_operations.rs`
5. ✅ Implement extract_slot and place_at_slot
6. ✅ Test slot extraction

### Day 3: Geometric Product
1. ✅ Rewrite `geometric_product.rs` for SIMD
2. ✅ Test with (1+2e₁)⊗(3+4e₂)
3. ✅ Fix bugs
4. ✅ Test with more complex cases
5. ✅ Document API

### Day 4: Testing & Optimization
1. ✅ Comprehensive testing
2. ✅ Benchmark performance
3. ✅ Compare with separate ciphertext approach
4. ✅ Write examples
5. ✅ Update documentation

---

## Mathematical Correctness Checks

### 1. Slot Encoding
```rust
#[test]
fn test_slot_encoding_roundtrip() {
    let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let slots = encode_multivector_slots(&mv, scale, n);
    let coeffs = slots_to_coefficients(&slots, n);
    let slots_back = coefficients_to_slots(&coeffs, scale, n);
    let mv_back = decode_multivector_slots(&slots_back);

    // Should be identical
    assert_close(&mv, &mv_back);
}
```

### 2. Automorphism Properties
```rust
#[test]
fn test_automorphism_composition() {
    // σₖ₁ ∘ σₖ₂ = σ_{k₁·k₂}
    let poly = random_poly(n);
    let result1 = apply_automorphism(
        &apply_automorphism(&poly, k1, n), k2, n);
    let result2 = apply_automorphism(&poly, (k1*k2) % (2*n), n);
    assert_eq!(result1, result2);
}
```

### 3. Rotation Correctness
```rust
#[test]
fn test_slot_rotation() {
    let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let ct = encrypt_multivector(&mv, pk, params);

    // Rotate left by 2
    let ct_rot = rotate_slots(&ct, 2, rotk, params);
    let mv_rot = decrypt_multivector(&ct_rot, sk, params);

    // Should be [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0]
    assert_close(&mv_rot, &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0]);
}
```

---

## References

### Papers
1. "Homomorphic Encryption for Arithmetic of Approximate Numbers" (CKKS original)
   - Section 3.3: SIMD packing
   - Section 4.2: Rotations via automorphisms

2. "Improved Bootstrapping for Approximate HE"
   - Advanced slot operations
   - Efficient rotation key generation

### Implementations
1. **SEAL** (Microsoft): `seal/util/polyarithsmallmod.cpp`
   - Reference for automorphism implementation
   - Galois key generation

2. **HElib** (IBM): `src/permutations.cpp`
   - Slot permutation strategies
   - FFT-based encoding

3. **PALISADE**: `src/pke/lib/scheme/ckks`
   - CKKS slot operations
   - Comprehensive examples

---

## Success Criteria

### Must Have (Phase 2 Complete)
- ✅ Slot encoding/decoding works correctly
- ✅ Rotation via automorphisms is correct
- ✅ Geometric product produces correct results
- ✅ Error < 1.0 for test cases
- ✅ All tests pass

### Should Have
- ✅ Rotation keys generation optimized
- ✅ Clear API documentation
- ✅ Multiple test cases
- ✅ Comparison with theory

### Nice to Have
- 🎯 Performance benchmarks
- 🎯 Batch processing support (multiple MVs)
- 🎯 Optimized FFT implementation
- 🎯 Paper-quality documentation

---

**Status**: Design complete, ready for implementation
**Timeline**: 3-4 days for full implementation
**Confidence**: High - this is the standard CKKS approach

---

**Next**: Start with `slot_encoding.rs` implementation
