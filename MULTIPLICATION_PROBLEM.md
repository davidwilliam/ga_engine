# CKKS Canonical Embedding: Polynomial Multiplication Issue

## Executive Summary

Following the expert's guidance from the previous consultation, we successfully **fixed the slot leakage issue** by implementing the correct inverse canonical embedding formula with 1/N normalization and single-loop conjugate handling. Slots 8-31 now have values ~10^-6 (floating-point noise) instead of ~0.09 ✅.

However, we discovered a **fundamental issue**: polynomial multiplication does not produce correct slot-wise products, even in plaintext space (no encryption). This suggests either:
1. A misunderstanding of how orbit-order indexing interacts with conjugate symmetry for real-valued slots, OR
2. An implementation bug we haven't identified

## What We Fixed (Based on Expert Advice)

### 1. Corrected Encoder Formula

**Before** (WRONG):
```rust
// Created "extended" array, tried to index conjugate slots
let mut extended = vec![Complex::new(0.0, 0.0); n];
for t in 0..num_slots {
    extended[t] = slots[t];
}
for t in 1..num_slots {
    extended[n - t] = slots[t].conj();  // WRONG: guessing conjugate position
}
// ... then summed over both loops with 2/N normalization
coeffs[j] = sum.re * 2.0 / (n as f64);  // WRONG: 2/N normalization
```

**After** (CORRECT per expert):
```rust
// Single loop with analytical conjugate handling
for j in 0..n {
    let mut sum = Complex::new(0.0, 0.0);

    for t in 0..num_slots {
        let angle = 2.0 * PI * (e[t] as f64) * (j as f64) / (m as f64);
        let w = Complex::new(angle.cos(), -angle.sin());  // exp(-i*angle)

        // Add both z[t] * w and conj(z[t]) * conj(w)
        sum += slots[t] * w + slots[t].conj() * w.conj();
    }

    coeffs_float[j] = sum.re / (n as f64);  // CORRECT: 1/N normalization
}
```

File: [canonical_embedding.rs:82-126](src/clifford_fhe/canonical_embedding.rs#L82-L126)

### 2. Corrected Decoder Formula

**Implementation** (matches expert specification):
```rust
for t in 0..num_slots {
    let mut sum = Complex::new(0.0, 0.0);
    for j in 0..n {
        let angle = 2.0 * PI * (e[t] as f64) * (j as f64) / (m as f64);
        let w = Complex::new(angle.cos(), angle.sin());  // exp(+i*angle)
        sum += coeffs_float[j] * w;
    }
    slots[t] = sum;
}
```

File: [canonical_embedding.rs:141-170](src/clifford_fhe/canonical_embedding.rs#L141-L170)

### 3. Verification: Slot Leakage Fixed ✅

**Test**: Encode `[1.0, 2.0, 0, 0, 0, 0, 0, 0]` and check all 32 slot values.

**Results**:
```
Slot[0]:  0.999995 + -0.000000i  ✅
Slot[1]:  1.999997 + -0.000000i  ✅
Slot[2]: -0.000000 + -0.000000i  ✅
...
Slot[7]:  0.000000 + 0.000000i   ✅
Slot[8]: -0.000003 + -0.000000i  ✅ (was 0.093750 + -0.261068i before!)
...
Slot[31]: -0.000000 + 0.000000i  ✅ (was 0.093750 + 0.002352i before!)
```

**Conclusion**: Slots 8-31 now have magnitudes < 0.00001 (just floating-point noise). The 3/32 DC bias is **completely eliminated** ✅.

Test file: [test_canonical_all_slots.rs](examples/test_canonical_all_slots.rs)

## Current Problem: Polynomial Multiplication Doesn't Work

### Test Case (Plaintext Polynomial Multiplication)

We perform the following **without encryption** to isolate the encoding issue:

1. Encode `mv_a = [2.0, 0, 0, 0, 0, 0, 0, 0]` → polynomial `p_a(x)`
2. Encode `mv_b = [3.0, 0, 0, 0, 0, 0, 0, 0]` → polynomial `p_b(x)`
3. Multiply polynomials: `p_c(x) = p_a(x) × p_b(x) mod (x^N + 1, q)`
4. Decode `p_c(x)` → should give `[6.0, 0, 0, 0, 0, 0, 0, 0]`

### Expected CKKS Property

In standard CKKS, canonical embedding should satisfy:
```
decode(poly_mult(encode(a), encode(b))) = a ⊙ b
```
where ⊙ is element-wise (slot-wise) multiplication.

This is a **fundamental property** of CKKS that enables homomorphic multiplication.

### Actual Results

**Test output**:
```
Encoded a: first 8 coeffs = [65536, 65457, 65220, 64827, 64277, 63572, 62714, 61705]
Encoded b: first 8 coeffs = [98304, 98186, 97831, 97240, 96415, 95358, 94071, 92558]

Product: first 8 coeffs = [206158430592, 205910103735, 205165721401, ...]

Trying different decode scales:
  With scale^2 = 1.10e12: [-13.868, -3.554, ...]
  With scale   = 1.05e6:  [-14541381.355, -3726526.803, ...]
  With scale   = 1:       [-15247743495633.307, ...]

Expected: [6.0, 0, 0, 0, 0, 0, 0, 0]
```

**All decode scales fail!** Even scale^2 gives `-13.868` instead of `6.0`.

Test file: [test_plaintext_multiply.rs](examples/test_plaintext_multiply.rs)

### Why This is Concerning

1. **No encryption involved** - This is pure encode → poly_mult → decode
2. **Rotations work perfectly** - Same encoding/decoding formulas work for automorphisms
3. **Roundtrip works perfectly** - encode → decode gives original values (error < 10^-5)
4. **Encoder/decoder match expert's spec exactly** - We've implemented their formulas verbatim

This suggests the issue is not with the encoder/decoder formulas themselves, but with how they interact with polynomial multiplication when using:
- Orbit-order indexing (`e[t] = 5^t mod M`)
- Conjugate symmetry for real-valued slots
- Real polynomial coefficients

## Implementation Details

### Our Parameters

```rust
N = 64              // Ring dimension
M = 2N = 128        // Cyclotomic order
L = N/2 = 32        // Number of slots
g = 5               // Generator
scale = 2^20        // 1048576
q = 1099511627689   // 40-bit prime (≈ 1.1 × 10^12)
```

### Orbit Order

```
e[0]  = 1
e[1]  = 5
e[2]  = 25
e[3]  = 61   (= 125 mod 128)
e[4]  = 49   (= 305 mod 128)
e[5]  = 53   (= 245 mod 128)
e[6]  = 9    (= 265 mod 128)
e[7]  = 45   (= 45 mod 128)
...
e[31] = 127
```

### Polynomial Multiplication Implementation

```rust
fn polynomial_multiply_mod(a: &[i64], b: &[i64], q: i64, n: usize) -> Vec<i64> {
    let mut result = vec![0i64; n];

    for i in 0..n {
        for j in 0..n {
            let idx = i + j;
            if idx < n {
                result[idx] = (result[idx] + a[i] * b[j]) % q;
            } else {
                // x^n = -1 reduction (negacyclic)
                let wrapped_idx = idx % n;
                result[wrapped_idx] = (result[wrapped_idx] - a[i] * b[j]) % q;
            }
        }
    }

    result.iter().map(|&x| ((x % q) + q) % q).collect()
}
```

This is standard negacyclic polynomial multiplication mod (x^N + 1).

### What Works Correctly ✅

To help diagnose, here's what IS working:

1. **Encode/Decode Roundtrip**:
   ```
   Input:   [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
   Decoded: [1.000004, 2.000001, 2.999997, 3.999999, ...]
   Max error: 4.54e-6  ✅
   ```

2. **Rotation via Automorphism**:
   ```
   Input:  [1, 2, 3, 4, 5, 6, 7, 8]
   σ_5:    [2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 0.00]  ✅ (left rotate)
   σ_77:   [8.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00]  ✅ (right rotate)
   Max error: 4.54e-6
   ```

3. **Slot Leakage Eliminated**:
   - Slots 8-31 have magnitudes < 10^-5 ✅

4. **Homomorphic Encryption/Decryption**:
   - Encrypt → Decrypt recovers original values correctly ✅

## Diagnostic Data

### Coefficient Analysis

For `[2, 0, 0, ...]` encoded with scale = 2^20:

```
Coeffs: [65536, 65457, 65220, 64827, 64277, 63572, 62714, 61705, ...]
```

Observations:
- First coefficient ≈ scale × 2 / 32 = 1048576 × 2 / 32 = 65536 ✅
- Coefficients are decreasing
- All coefficients are positive (as expected for real-valued encoding)

For `[3, 0, 0, ...]` encoded with scale = 2^20:

```
Coeffs: [98304, 98186, 97831, 97240, 96415, 95358, 94071, 92558, ...]
```

- First coefficient ≈ scale × 3 / 32 = 1048576 × 3 / 32 = 98304 ✅

After polynomial multiplication:

```
Product: [206158430592, 205910103735, 205165721401, ...]
```

Expected if slot-wise: `6 × scale^2 / 32 ≈ 6 × 2^40 / 32 = 6 × 2^35 = 206158430208`

**Observation**: First coefficient is very close! `206158430592 ≈ 206158430208`

But when decoded with scale^2, we get `-13.868` instead of `6.0`. This suggests the **DFT is not inverting correctly** for the product polynomial.

### Conjugate Symmetry Check

For encoded `[2, 0, ...]`:

If we decode ALL 32 slots (not just first 8), do we see conjugate symmetry?

```
Slot[0]:  1.999990 + 0.000000i
Slot[1]:  0.000000 + 0.000000i
...
Slot[31]: 0.000000 + 0.000000i
```

Wait - slot[0] shows ~2, which is correct! Let me verify this...

Actually, we're encoding 8-component multivectors into 32 slots. The first slot corresponds to e[0] = 1, which is exponent 1 in the orbit order.

## Questions for the Expert

### Question 1: Does Orbit-Order Preserve Multiplication Property?

In standard CKKS with **natural ordering** (slots indexed by 2k+1), polynomial multiplication gives slot-wise multiplication:

```
p(x) = IDFT(slots)
q(x) = IDFT(slots')
p(x) × q(x) = IDFT(slots ⊙ slots')
```

**Question**: Does this property still hold with **orbit-order indexing** where slots are indexed by `e[t] = g^t mod M`?

Or does orbit-order require a different approach to multiplication?

### Question 2: Real-Valued Slots with Conjugate Symmetry

For real-valued slots, we use conjugate symmetry in the encoding:

```rust
sum += slots[t] * w + slots[t].conj() * w.conj();
```

When we multiply two polynomials that were encoded this way:
- Both have real coefficients
- Both satisfy the conjugate symmetry condition
- Product also has real coefficients

**Question**: Should the product polynomial, when decoded, give the element-wise product of the original slot vectors?

Or is there an additional step needed for real-valued encodings?

### Question 3: Diagnostic Test Request

We observe:
- Roundtrip encode/decode: ✅ Works (error < 10^-5)
- Rotation via automorphism: ✅ Works (error < 10^-5)
- Slot leakage: ✅ Fixed (slots 8-31 ~zero)
- Polynomial multiplication: ❌ Fails (gives -13.868 instead of 6.0)

**Question**: What diagnostic test can we run to identify where the multiplication property breaks down?

Options we've considered:
1. Test with complex-valued slots (no conjugate symmetry)
2. Test with all 32 slots filled (not just 8)
3. Check if the issue is specific to real-valued encodings
4. Verify DFT/IDFT pair is truly adjoint for our implementation

### Question 4: Is Our Implementation Correct?

We've implemented exactly the formulas provided:

**Encoder** (matches expert's pseudocode):
```rust
for j in 0..N {
    let mut sum = Complex::new(0.0, 0.0);
    for t in 0..L {
        let ang = 2.0 * PI * (e[t] as f64) * (j as f64) / (M as f64);
        let w = Complex::new(ang.cos(), -ang.sin());
        sum += slots[t] * w + slots[t].conj() * w.conj();
    }
    coeffs[j] = sum.re / (N as f64);
}
```

**Decoder** (matches expert's pseudocode):
```rust
for t in 0..L {
    let mut y = Complex::new(0.0, 0.0);
    for j in 0..N {
        let ang = 2.0 * PI * (e[t] as f64) * (j as f64) / (M as f64);
        let w = Complex::new(ang.cos(), ang.sin());
        y += coeffs[j] * w;
    }
    slots[t] = y;
}
```

**Question**: Is there anything wrong with this implementation that would break the multiplication property?

### Question 5: Scale Management After Multiplication

After multiplying two polynomials encoded with scale `s`:
- Product coefficients are ~`s^2` larger
- We try decoding with scale `s^2`

**Question**: Is this the correct scale management for orbit-order CKKS?

Or should we:
- Divide the coefficients by `s` before decoding?
- Use a different normalization factor?
- Perform rescaling in a specific way?

## Test Files for Expert Review

### Minimal Reproducible Tests

1. **test_encode_decode_only.rs** - Roundtrip test (PASSES)
   - Verifies encode/decode are inverses
   - Max error: 4.54e-6 ✅

2. **test_canonical_all_slots.rs** - Slot leakage test (PASSES)
   - Verifies slots 8-31 are ~zero
   - Shows the fix worked ✅

3. **test_plaintext_multiply.rs** - Polynomial multiplication test (FAILS)
   - No encryption - pure plaintext operation
   - Multiplies `[2] × [3]`, expects `[6]`, gets `[-13.868]` ❌
   - **This is the key failing test**

4. **sanity_checks_orbit_order.rs** - Complete sanity check suite (PASSES)
   - All 5 checks pass
   - Rotations work correctly ✅

### Source Code

Main implementation file:
- **canonical_embedding.rs** - Encoder and decoder implementation
  - Lines 82-126: `canonical_embed_encode()`
  - Lines 141-170: `canonical_embed_decode()`
  - Lines 173-211: Wrapper functions for multivectors

## What We Need

1. **Root cause**: Why does polynomial multiplication not give slot-wise products?

2. **Mathematical clarification**: Does orbit-order + conjugate symmetry preserve the multiplication property?

3. **Fix or workaround**: How to perform homomorphic multiplication correctly with orbit-order CKKS?

## Appendix: Comparison with Natural Ordering

In natural ordering, slot `k` corresponds to root `ζ_M^{2k+1}`:
- Slot 0 → ζ^1
- Slot 1 → ζ^3
- Slot 2 → ζ^5
- ...

In orbit ordering, slot `t` corresponds to root `ζ_M^{e[t]}` where `e[t] = 5^t mod M`:
- Slot 0 → ζ^1
- Slot 1 → ζ^5
- Slot 2 → ζ^25
- Slot 3 → ζ^61
- ...

**Question**: Could this reordering break the FFT convolution theorem that underpins CKKS multiplication?

In standard FFT:
```
FFT(a × b) = FFT(a) ⊙ FFT(b)
```

Does this hold when FFT uses non-consecutive roots in orbit order?

---

Thank you for your continued expertise! The orbit-order rotation fix was perfect. We're so close to having fully working Clifford-FHE—just need to understand why multiplication doesn't work with the corrected encoding.
