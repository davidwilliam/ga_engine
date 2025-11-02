# RNS-CKKS Implementation Plan

## Overview

We're implementing **full RNS-CKKS** (Residue Number System CKKS) to enable proper homomorphic multiplication for Clifford-FHE. This will allow geometric algebra operations on encrypted multivectors.

## Why RNS-CKKS?

### The Problem with Single-Modulus
- Need Q > scale² for CKKS to work
- Single i64 limits us to Q < 2^63
- Can't chain multiplications (scale grows exponentially)

### RNS Solution
- Represent coefficients as tuples: `(c mod q₀, c mod q₁, c mod q₂, ...)`
- Effective modulus: Q = q₀ · q₁ · q₂ · ... (can be 2^200+)
- Rescaling drops one prime from the tuple
- **Standard approach used in SEAL, HElib, OpenFHE**

## Architecture

### Current (Single-Modulus)
```
Polynomial coefficient: i64 (single value)
Operations mod Q where Q < 2^63
```

### New (RNS)
```
Polynomial coefficient: Vec<i64> (tuple of residues)
Each residue: c_i = c mod q_i where q_i is a prime
Operations: apply to each residue independently
```

### Example
```rust
// Value: c = 123456789
// Primes: [q₀ = 1009, q₁ = 1013]

// Single-modulus: c = 123456789
// RNS: c = [123456789 mod 1009, 123456789 mod 1013]
//        = [341, 997]
```

## Components

### ✅ Phase 1: RNS Core (DONE)

**File**: `src/clifford_fhe/rns.rs`

Implemented:
- `RnsPolynomial` - polynomial in RNS representation
- `rns_add`, `rns_sub`, `rns_negate` - basic operations
- `rns_multiply` - polynomial multiplication (uses NTT per-prime)
- `rns_rescale` - drop last prime from modulus chain
- `from_coeffs`, `to_coeffs` - CRT conversions
- Tests: conversion, addition all passing ✅

### 🔄 Phase 2: Integration (IN PROGRESS)

**Files to Update**:
1. `params.rs` - Add RNS modulus chain
2. `ckks.rs` - Update `Ciphertext`, `Plaintext` to use RNS
3. `keys.rs` - Update key generation for RNS
4. `ckks.rs` - Rewrite `encrypt`, `decrypt`, `multiply`

### ⏳ Phase 3: Operations (PENDING)

**Files to Update**:
1. `automorphisms.rs` - Apply to RNS polynomials
2. `ckks.rs` - `rotate_slots` with RNS
3. `geometric_product.rs` - Should work unchanged!

### ⏳ Phase 4: Canonical Embedding (MINIMAL CHANGES)

**File**: `canonical_embedding.rs`

**KEY INSIGHT**: Canonical embedding works at the **slot level**, not coefficient level!

```rust
// Current interface (STAYS THE SAME):
pub fn encode_multivector_canonical(
    mv: &[f64; 8],
    scale: f64,
    n: usize
) -> Vec<i64>  // Returns single-modulus coefficients

// New interface (compatible):
pub fn encode_multivector_canonical_rns(
    mv: &[f64; 8],
    scale: f64,
    n: usize,
    primes: &[i64],
) -> RnsPolynomial  // Returns RNS coefficients
```

**Implementation**: Just wrap existing encode and convert to RNS:
```rust
pub fn encode_multivector_canonical_rns(
    mv: &[f64; 8],
    scale: f64,
    n: usize,
    primes: &[i64],
) -> RnsPolynomial {
    // Use existing encoder (THIS DOESN'T CHANGE!)
    let coeffs = encode_multivector_canonical(mv, scale, n);

    // Convert to RNS
    RnsPolynomial::from_coeffs(&coeffs, primes, n, 0)
}
```

**Why this works**:
- Canonical embedding: multivector → slots → polynomial coefficients
- RNS is just a different REPRESENTATION of the same coefficients
- Math is identical, only storage changes!

## Implementation Steps

### Step 1: Update Parameters ⏳
```rust
// params.rs
pub struct CliffordFHEParams {
    pub n: usize,
    pub moduli: Vec<i64>,  // Already have this!
    pub scale: f64,
    // ... existing fields
}

impl CliffordFHEParams {
    pub fn new_rns_mult() -> Self {
        Self {
            n: 1024,
            moduli: vec![
                40, // q₀ - base prime
                40, // q₁ - rescaling prime 1
                40, // q₂ - rescaling prime 2
            ]
            .iter()
            .map(|bits| Self::generate_prime(*bits))
            .collect(),
            scale: 2f64.powi(40), // ≈ 40-bit prime
            // Q = q₀ · q₁ · q₂ ≈ 2^120 >> scale² = 2^80 ✅
        }
    }
}
```

### Step 2: Update Ciphertext/Plaintext ⏳
```rust
// ckks.rs
pub struct Ciphertext {
    pub c0: RnsPolynomial,  // was: Vec<i64>
    pub c1: RnsPolynomial,  // was: Vec<i64>
    pub level: usize,
    pub scale: f64,
    pub n: usize,
}

pub struct Plaintext {
    pub coeffs: RnsPolynomial,  // was: Vec<i64>
    pub scale: f64,
}
```

### Step 3: Update Encryption ⏳
```rust
// ckks.rs
pub fn encrypt(pk: &PublicKey, pt: &Plaintext, params: &CliffordFHEParams) -> Ciphertext {
    // All operations now use rns_add, rns_multiply instead of coefficient-wise
    // Algorithm is IDENTICAL, just different arithmetic
}
```

### Step 4: Update Multiplication ⏳
```rust
// ckks.rs
pub fn multiply(ct1: &Ciphertext, ct2: &Ciphertext, evk: &EvaluationKey, params: &CliffordFHEParams) -> Ciphertext {
    // 1. Multiply: c0d0, c0d1 + c1d0, c1d1 (using rns_multiply)
    // 2. Relinearize (using RNS key-switching)
    // 3. Rescale: rns_rescale() - THIS IS THE KEY FIX!
    // 4. Return ciphertext at level+1
}
```

### Step 5: Canonical Embedding Adapters ⏳
```rust
// canonical_embedding.rs

// Keep ALL existing functions unchanged!
// Add RNS wrappers:

pub fn encode_multivector_canonical_rns(...) -> RnsPolynomial {
    let coeffs = encode_multivector_canonical(...);  // Existing function!
    RnsPolynomial::from_coeffs(&coeffs, primes, n, 0)
}

pub fn decode_multivector_canonical_rns(rns_poly: &RnsPolynomial, ...) -> [f64; 8] {
    let coeffs = rns_poly.to_coeffs(primes);  // Convert from RNS
    decode_multivector_canonical(&coeffs, ...)  // Existing function!
}
```

## Testing Plan

### Test 1: RNS Arithmetic ✅ DONE
```rust
test_rns_conversion() ✅
test_rns_add() ✅
test_mod_inverse() ✅
```

### Test 2: RNS Encrypt/Decrypt ⏳
```rust
// Verify: Enc(m) → Dec → m with RNS
```

### Test 3: RNS Multiply ⏳
```rust
// Verify: Enc(a) * Enc(b) → Dec → a*b with RNS rescaling
```

### Test 4: Canonical Embedding with RNS ⏳
```rust
// Verify: encode_rns → encrypt → decrypt → decode_rns gives original multivector
```

### Test 5: Geometric Product ⏳
```rust
// Verify: End-to-end geometric product on encrypted multivectors
```

## Timeline Estimate

- ✅ **Phase 1** (RNS Core): 2 hours → DONE
- ⏳ **Phase 2** (Integration): 4-6 hours
- ⏳ **Phase 3** (Operations): 2-3 hours
- ⏳ **Phase 4** (Canonical Embedding): 1-2 hours
- ⏳ **Testing & Debugging**: 3-4 hours

**Total**: ~12-17 hours of focused work

## Benefits After Completion

1. **Homomorphic multiplication works** ✅
2. **Can chain operations** (depth 3-5 circuits)
3. **Proper CKKS rescaling** (as described in literature)
4. **Compatible with canonical embedding** (no math changes!)
5. **Foundation for bootstrapping** (future work)
6. **Production-quality implementation**

## Current Status

✅ RNS core implemented and tested
🔄 Starting integration with CKKS infrastructure
⏳ Canonical embedding adapters (minimal changes needed)
⏳ End-to-end testing

## Key Insight

**The canonical embedding math DOES NOT CHANGE!**

RNS is just a different way to store polynomial coefficients. The embedding:
```
Multivector → Slots → Polynomial Coefficients
```
is IDENTICAL. We just store those coefficients in RNS form instead of single i64 values.

This means all the expert-verified formulas, orbit-order indexing, slot rotations, etc. **stay exactly the same**. We're only changing the representation layer, not the mathematics!
