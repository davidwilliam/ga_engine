# CKKS Canonical Embedding: Slot Leakage Problem

## Executive Summary

We successfully implemented orbit-order indexing for CKKS slot rotations (all automorphism tests passing ✅). However, when testing homomorphic multiplication, we discovered a **critical slot leakage issue**: encoding 8 multivector components into a 32-slot CKKS scheme causes non-zero values to appear in unused slots 8-31, leading to catastrophic errors in multiplication (~10^6 error magnitude).

## Background: What We've Accomplished

### Phase 1: Fixed CKKS Slot Rotations ✅

Following expert guidance from the previous consultation, we successfully implemented **orbit-order indexing**:

1. **Implementation** ([canonical_embedding.rs:41-68](src/clifford_fhe/canonical_embedding.rs#L41-L68)):
   ```rust
   fn orbit_order(n: usize, g: usize) -> Vec<usize> {
       let m = 2 * n;
       let num_slots = n / 2;
       let mut e = vec![0usize; num_slots];
       let mut cur = 1usize;
       for t in 0..num_slots {
           e[t] = cur;
           cur = (cur * g) % m;
       }
       e
   }
   ```

2. **Evaluation Points**: Changed from natural ordering `ζ_M^{2k+1}` to orbit ordering `ζ_M^{e[t]}` where `e[t] = 5^t mod M`

3. **Test Results** (ALL PASSING ✅):
   - Generator order verified: `g^{N/2} ≡ 1 (mod M)`
   - Orbit properties confirmed: 32 distinct odd exponents
   - **σ_5 produces LEFT rotation by 1** (error < 1e-5) ✅
   - **σ_77 produces RIGHT rotation by 1** (error < 1e-5) ✅
   - Conjugate orbits disjoint ✅

### Phase 2: Updated Geometric Product Implementation

1. **Updated rotation key generation** ([keys.rs:135-143](src/clifford_fhe/keys.rs#L135-L143)):
   - Now generates keys for all rotations `-(N-1)` to `(N-1)`
   - For N=32: generates rotation keys from -31 to +31

2. **Updated component selectors** ([operations.rs:47-58](src/clifford_fhe/operations.rs#L47-L58)):
   - Changed from naive coefficient setting to proper canonical encoding
   - Now encodes selector multivectors using `encode_multivector_canonical`

3. **Status**: Geometric product compiles and runs without errors, but produces wrong results due to slot leakage (see below)

## Current Problem: Slot Leakage in Canonical Embedding

### Experimental Observation

**Setup**: Clifford-FHE with parameters:
- Ring dimension: `N = 64` (polynomial degree)
- Cyclotomic order: `M = 2N = 128`
- Number of slots: `N/2 = 32`
- Multivector dimension: 8 (for Cl(3,0))
- Generator: `g = 5`

**Test**: Encode multivector `[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` using canonical embedding.

**Expected**:
- Slots 0-7 contain multivector values: `[1.0, 2.0, 0, 0, 0, 0, 0, 0]`
- Slots 8-31 contain zeros: `[0, 0, ..., 0]` (24 zeros)

**Actual** (diagnostic output):
```
Slot[0]:  1.000000 + 0.000000i  ✅
Slot[1]:  2.000000 + 0.000000i  ✅
Slot[2]:  0.000000 + 0.000000i  ✅
Slot[3]:  0.000000 + 0.000000i  ✅
Slot[4]:  0.000000 + 0.000000i  ✅
Slot[5]:  0.000000 + 0.000000i  ✅
Slot[6]:  0.000000 + 0.000000i  ✅
Slot[7]:  0.000000 + 0.000000i  ✅
Slot[8]:  0.093750 + -0.261068i  ❌ NON-ZERO!
Slot[9]:  0.093750 + -0.210173i  ❌
Slot[10]: 0.093750 + -0.178713i  ❌
Slot[11]: 0.093750 + -0.156250i  ❌
...
Slot[31]: 0.093750 + -0.004648i  ❌
```

**Impact on Multiplication**:

Test: `[2, 0, 0, ...] × [3, 0, 0, ...] = [6, 0, 0, ...]`

```
Expected: [6.0, 0, 0, 0, 0, 0, 0, 0]
Got:      [-636153.612, 1025126.663, 1270343.927, -1199055.164, ...]
Error:    6.4×10^5  ❌ CATASTROPHIC
```

The slot leakage causes element-wise multiplication in slot space to produce completely wrong results.

### Implementation Details

Our `encode_multivector_canonical` function ([canonical_embedding.rs:190-200](src/clifford_fhe/canonical_embedding.rs#L190-L200)):

```rust
pub fn encode_multivector_canonical(mv: &[f64; 8], scale: f64, n: usize) -> Vec<i64> {
    assert!(n >= 16);
    let num_slots = n / 2;  // num_slots = 32

    let mut slots = vec![Complex::new(0.0, 0.0); num_slots];
    for i in 0..8 {
        slots[i] = Complex::new(mv[i], 0.0);  // Fill first 8 slots
    }
    // slots[8..31] remain as Complex::new(0.0, 0.0)

    canonical_embed_encode(&slots, scale, n)
}
```

The `canonical_embed_encode` function ([canonical_embedding.rs:82-134](src/clifford_fhe/canonical_embedding.rs#L82-L134)):

```rust
pub fn canonical_embed_encode(slots: &[Complex<f64>], scale: f64, n: usize) -> Vec<i64> {
    let num_slots = n / 2;  // 32
    let m = 2 * n;          // M = 128
    let g = 5;

    let e = orbit_order(n, g);  // e = [1, 5, 25, 61, 49, 53, 9, 45, ...]

    // Step 1: Extend to full N slots with conjugate symmetry
    let mut extended = vec![Complex::new(0.0, 0.0); n];  // Size 64

    for t in 0..num_slots {  // t = 0..31
        extended[t] = slots[t];
    }

    // Conjugate symmetry: extended[N/2 + t] = conj(slots[t])
    for t in 1..num_slots {  // t = 1..31
        extended[n - t] = slots[t].conj();  // extended[64-t] = conj(slots[t])
    }

    // Step 2: Compute coefficients via inverse DFT at orbit-ordered roots
    let mut coeffs_float = vec![0.0; n];

    for j in 0..n {
        let mut sum = Complex::new(0.0, 0.0);

        // Sum over first orbit
        for t in 0..num_slots {
            let exponent = -((e[t] * j) as i64);
            let angle = 2.0 * PI * (exponent as f64) / (m as f64);
            let root = Complex::new(angle.cos(), angle.sin());
            sum += extended[t] * root;
        }

        // Sum over conjugate orbit
        for t in 1..num_slots {
            let exponent = -((e[t] * j) as i64);
            let angle = -2.0 * PI * (exponent as f64) / (m as f64);
            let root = Complex::new(angle.cos(), angle.sin());
            sum += extended[n - t] * root;
        }

        coeffs_float[j] = sum.re * 2.0 / (n as f64);
    }

    // Scale and round to integers
    coeffs_float.iter().map(|&x| (x * scale).round() as i64).collect()
}
```

### Hypotheses for Why Slots 8-31 Are Non-Zero

**Hypothesis 1: Conjugate Symmetry Mapping Error**
- The line `extended[n - t] = slots[t].conj()` at position 106 maps slot `t` to position `n - t` in the extended array
- For orbit-order indexing, is this the correct conjugate pair?
- Should we be using a different mapping based on the orbit structure?
- In orbit-order, if slot `t` corresponds to root `ζ_M^{e[t]}`, its conjugate should be at root `ζ_M^{-e[t]} = ζ_M^{M - e[t]}`
- Does this correspond to position `n - t` in our extended array?

**Hypothesis 2: IDFT Formula Issue**
- We compute two sums: one over the first orbit (lines 115-121) and one over the conjugate orbit (lines 124-130)
- Are we correctly implementing the inverse DFT for orbit-ordered evaluation points?
- The normalization factor is `2.0 / n` — is this correct for orbit-order indexing?

**Hypothesis 3: Zero-Padding Interaction**
- When we set `slots[8..31] = 0.0 + 0.0i` and then apply conjugate symmetry, do these zeros create non-zero values after IDFT?
- Should we be handling unused slots differently?

**Hypothesis 4: Fundamental Mathematical Constraint**
- Is it mathematically possible to have slots 0-7 contain arbitrary values while slots 8-31 are exactly zero in CKKS with orbit-order indexing?
- Or does the IDFT necessarily "spread" information across all available slots?
- If this is a fundamental constraint, how should we handle it?

## What Works Correctly

To help diagnose, here's what IS working:

1. **Roundtrip for specific patterns**: When we encode and immediately decode the same multivector, we get the correct values back in slots 0-7 (with small rounding error)

2. **Rotation**: σ_5 and σ_77 correctly rotate slots as expected (tested with full 8-element vectors)

3. **The automorphism implementation**: Our `apply_automorphism` function correctly computes `x → x^k mod (x^N + 1)`

4. **Encryption/Decryption**: Basic CKKS encrypt/decrypt works correctly with canonical embedding

## Questions for the Expert

### Question 1: Conjugate Symmetry in Orbit-Order Indexing

In orbit-order indexing where slot `t` corresponds to evaluation point `ζ_M^{e[t]}` with `e[t] = g^t mod M`:

**a)** What is the correct way to identify the conjugate pair for slot `t`?
   - Is it at some position `t'` where `e[t'] = M - e[t]`?
   - How do we find `t'` given `t`?
   - Or is the conjugate relationship different in orbit-order indexing?

**b)** In our implementation, we use:
```rust
extended[n - t] = slots[t].conj();  // for t in 1..num_slots
```
Is this the correct mapping for orbit-order indexing, or should we compute the conjugate position differently?

**c)** For the specific case of `g=5`, `M=128`, `N=64`:
   - What is the explicit mapping between slot index `t` and its conjugate slot index?
   - Can you provide a few example pairs (e.g., slot 1 ↔ slot ?, slot 8 ↔ slot ?, etc.)?

### Question 2: Zero-Padding and Unused Slots

When encoding 8 multivector components into a 32-slot CKKS scheme:

**a)** Is it mathematically possible to have slots 0-7 contain arbitrary real values while slots 8-31 are *exactly* zero after canonical embedding?

**b)** If NOT possible (due to IDFT properties), what is the correct way to handle this situation?
   - Should we accept that slots 8-31 will have non-zero "spillover" values?
   - Should we use a smaller ring dimension (e.g., N=16 with 8 slots)?
   - Should we use a different encoding strategy for partial slot usage?

**c)** If slots 8-31 necessarily contain non-zero values, how does CKKS multiplication work correctly?
   - Will element-wise multiplication in slot space still give correct results for slots 0-7?
   - Or do we need to "mask out" the extra slots somehow?

### Question 3: IDFT Formula Verification

Our inverse DFT formula for orbit-order indexing:

```rust
for j in 0..n {
    let mut sum = Complex::new(0.0, 0.0);

    // First orbit: ζ_M^{e[t]}
    for t in 0..num_slots {
        sum += extended[t] * exp(-2πi * e[t] * j / M);
    }

    // Conjugate orbit: ζ_M^{-e[t]}
    for t in 1..num_slots {
        sum += extended[n - t] * exp(2πi * e[t] * j / M);
    }

    coeffs[j] = sum.re * (2.0 / n);
}
```

**a)** Is this the correct IDFT formula for orbit-order evaluation points?

**b)** Is the normalization factor `2.0 / n` correct?

**c)** Should we be handling the first orbit (t=0) and conjugate orbit (t=0) specially, or is our current approach correct?

### Question 4: Alternative Approaches

**a)** Should we consider using N=16 (8 slots) instead of N=64 (32 slots) to exactly match our multivector dimension?
   - Pros: No unused slots, no leakage issue
   - Cons: Smaller ring dimension might impact security or performance

**b)** Is there a "standard" way in CKKS to use only a subset of available slots without interference?
   - Masking techniques?
   - Different packing strategies?
   - Coefficient-space operations instead of slot-space?

**c)** For geometric algebra FHE specifically, are there better encoding strategies than trying to use CKKS slots directly?

## Diagnostic Information

### Our Parameters
```
Ring dimension:        N = 64
Cyclotomic order:      M = 128
Number of slots:       32
Multivector dimension: 8
Generator:            g = 5
Orbit (first 8):      e = [1, 5, 25, 61, 49, 53, 9, 45]
```

### Test Code
We can provide minimal reproducible test cases for:
- Encoding and checking all 32 slot values
- Multiplication test showing the error
- Rotation tests (which work correctly)

### Full Source Code
Available at: [canonical_embedding.rs](src/clifford_fhe/canonical_embedding.rs)

## What We Need

We need to understand:

1. **Root cause**: Why do slots 8-31 have non-zero values?
2. **Fix**: How to properly handle 8 components in a 32-slot scheme
3. **Validation**: How to verify our implementation is correct

Thank you for your expertise! The orbit-order fix was perfect and rotations work flawlessly. We're very close to having a fully working Clifford-FHE implementation.

---

## Appendix: Complete Slot Dump

For reference, here are all 32 slot values when encoding `[1.0, 2.0, 0, 0, 0, 0, 0, 0]`:

```
Slot[0]:  1.000000 + 0.000000i  (magnitude: 1.000000)  ✅
Slot[1]:  2.000000 + 0.000000i  (magnitude: 2.000000)  ✅
Slot[2]:  0.000000 + 0.000000i  (magnitude: 0.000000)  ✅
Slot[3]:  0.000000 + 0.000000i  (magnitude: 0.000000)  ✅
Slot[4]:  0.000000 + 0.000000i  (magnitude: 0.000000)  ✅
Slot[5]:  0.000000 + 0.000000i  (magnitude: 0.000000)  ✅
Slot[6]:  0.000000 + 0.000000i  (magnitude: 0.000000)  ✅
Slot[7]:  0.000000 + 0.000000i  (magnitude: 0.000000)  ✅
Slot[8]:  0.093750 + -0.261068i (magnitude: 0.277245)  ❌
Slot[9]:  0.093750 + -0.210173i (magnitude: 0.230191)  ❌
Slot[10]: 0.093750 + -0.178713i (magnitude: 0.206512)  ❌
Slot[11]: 0.093750 + -0.156250i (magnitude: 0.187862)  ❌
Slot[12]: 0.093750 + -0.138172i (magnitude: 0.172812)  ❌
Slot[13]: 0.093750 + -0.123195i (magnitude: 0.160436)  ❌
Slot[14]: 0.093750 + -0.110238i (magnitude: 0.150113)  ❌
Slot[15]: 0.093750 + -0.098854i (magnitude: 0.141400)  ❌
Slot[16]: 0.093750 + -0.088716i (magnitude: 0.133980)  ❌
Slot[17]: 0.093750 + -0.079591i (magnitude: 0.127621)  ❌
Slot[18]: 0.093750 + -0.071281i (magnitude: 0.122135)  ❌
Slot[19]: 0.093750 + -0.063653i (magnitude: 0.117392)  ❌
Slot[20]: 0.093750 + -0.056580i (magnitude: 0.113276)  ❌
Slot[21]: 0.093750 + -0.049987i (magnitude: 0.109711)  ❌
Slot[22]: 0.093750 + -0.043779i (magnitude: 0.106619)  ❌
Slot[23]: 0.093750 + -0.037902i (magnitude: 0.103948)  ❌
Slot[24]: 0.093750 + -0.032304i (magnitude: 0.101656)  ❌
Slot[25]: 0.093750 + -0.026936i (magnitude: 0.099703)  ❌
Slot[26]: 0.093750 + -0.021760i (magnitude: 0.098062)  ❌
Slot[27]: 0.093750 + -0.016738i (magnitude: 0.096709)  ❌
Slot[28]: 0.093750 + -0.011848i (magnitude: 0.095626)  ❌
Slot[29]: 0.093750 + -0.007056i (magnitude: 0.094798)  ❌
Slot[30]: 0.093750 + -0.002325i (magnitude: 0.094213)  ❌
Slot[31]: 0.093750 + 0.002352i  (magnitude: 0.093865)  ❌
```

Note the pattern:
- Real parts of slots 8-31 are all `0.09375 = 3/32`
- Imaginary parts decrease in magnitude from slot 8 to slot 31
- This suggests a systematic issue with our IDFT or conjugate symmetry handling
