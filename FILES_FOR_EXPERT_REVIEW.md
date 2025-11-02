# Files for Expert Review

## Problem Documentation

**Main document**: [MULTIPLICATION_PROBLEM.md](MULTIPLICATION_PROBLEM.md)

**Summary**: Orbit-order encoder/decoder fixed (slot leakage eliminated ✅), but polynomial multiplication doesn't give slot-wise products, even in plaintext. Getting `-13.868` instead of `6.0` for `[2] × [3]`.

## Core Implementation Files

### 1. canonical_embedding.rs
**Path**: `src/clifford_fhe/canonical_embedding.rs`

**Key functions**:
- `orbit_order()` (lines 55-68) - Computes e[t] = g^t mod M
- `canonical_embed_encode()` (lines 82-126) - **Encoder implementation** (matches expert's formula)
- `canonical_embed_decode()` (lines 141-170) - **Decoder implementation** (matches expert's formula)
- `encode_multivector_canonical()` (lines 173-200) - Wrapper for 8-component multivectors
- `decode_multivector_canonical()` (lines 203-211) - Wrapper for decoding

**Changes made**:
- Fixed encoder to use 1/N normalization (not 2/N)
- Single loop with analytical conjugate handling
- Eliminated "extended array" approach
- Fixed decoder to use positive angle

**Status**: Encoder/decoder match expert specification exactly ✅

### 2. ckks.rs
**Path**: `src/clifford_fhe/ckks.rs`

**Key functions**:
- `multiply()` (lines 252-299) - Homomorphic multiplication with relinearization
- `polynomial_multiply_ntt()` (lines 419-437) - Negacyclic polynomial multiplication

**Relevant**: Shows how polynomial multiplication is performed (x^N = -1 reduction)

### 3. params.rs
**Path**: `src/clifford_fhe/params.rs`

**Key section**:
- `new_test()` (lines 43-58) - Test parameters used in all tests

**Parameters**:
```rust
n: 64                          // N = 64
moduli: [40-bit, 40-bit, 40-bit]  // ~1.1 × 10^12 each
scale: 2^20                    // 1048576
```

## Test Files (Evidence of Problem)

### Failing Test

**test_plaintext_multiply.rs** ❌
**Path**: `examples/test_plaintext_multiply.rs`

**What it does**:
1. Encodes `[2, 0, 0, ...]` and `[3, 0, 0, ...]`
2. Multiplies polynomials (no encryption)
3. Tries to decode with different scales
4. **Expected**: `[6, 0, 0, ...]`
5. **Got**: `[-13.868, -3.554, ...]` with scale^2

**Key observation**: Even in plaintext (no noise), multiplication doesn't work.

**Run**: `cargo run --release --example test_plaintext_multiply`

### Passing Tests

**test_canonical_all_slots.rs** ✅
**Path**: `examples/test_canonical_all_slots.rs`

**What it tests**: Verifies slots 8-31 are ~zero after encoding

**Result**: All slots 8-31 have magnitude < 10^-5 ✅

**Run**: `cargo run --release --example test_canonical_all_slots`

---

**test_encode_decode_only.rs** ✅
**Path**: `examples/test_encode_decode_only.rs`

**What it tests**: Roundtrip encode → decode

**Result**: Max error 4.54e-6 ✅

**Run**: `cargo run --release --example test_encode_decode_only`

---

**sanity_checks_orbit_order.rs** ✅
**Path**: `examples/sanity_checks_orbit_order.rs`

**What it tests**: All 5 sanity checks from previous expert consultation
- Generator order
- Orbit properties
- σ_5 rotates left by 1
- σ_77 rotates right by 1
- Conjugate orbits disjoint

**Result**: All checks PASS ✅

**Run**: `cargo run --release --example sanity_checks_orbit_order`

---

**test_canonical_automorphisms.rs** ✅
**Path**: `examples/test_canonical_automorphisms.rs`

**What it tests**: Verifies k=5, k=25, k=77 produce expected rotations

**Result**: All rotation tests PASS ✅

**Run**: `cargo run --release --example test_canonical_automorphisms`

## Additional Context Files

### automorphisms.rs
**Path**: `src/clifford_fhe/automorphisms.rs`

**Relevant**: Shows how `apply_automorphism()` works (x → x^k mod (x^N+1))

### slot_encoding.rs (OLD - not used)
**Path**: `src/clifford_fhe/slot_encoding.rs`

**Note**: This is the OLD FFT-based encoding with natural ordering. We no longer use this for canonical embedding, but it's in the codebase for reference.

## Quick Summary for Expert

### What Works ✅
1. Orbit-order encoding/decoding (roundtrip error < 10^-5)
2. Slot leakage fixed (slots 8-31 ~zero)
3. Rotations via automorphisms (σ_5, σ_77 work perfectly)
4. All 5 sanity checks pass

### What Doesn't Work ❌
1. Polynomial multiplication doesn't give slot-wise products
2. Test: `encode([2]) × encode([3])` decoded gives `-13.868`, not `6.0`
3. This happens in plaintext (no encryption/noise)

### Key Question
**Does orbit-order canonical embedding with conjugate symmetry preserve the slot-wise multiplication property of CKKS?**

Standard CKKS: `decode(p(x) × q(x)) = decode(p(x)) ⊙ decode(q(x))`

This should hold, but it doesn't in our implementation even though encoder/decoder match the expert's formulas exactly.

## How to Send Files

Please send to the expert:
1. **MULTIPLICATION_PROBLEM.md** - Complete problem description
2. **src/clifford_fhe/canonical_embedding.rs** - Main implementation
3. **examples/test_plaintext_multiply.rs** - Failing test (most important)
4. **examples/sanity_checks_orbit_order.rs** - Passing tests (for context)

Optional (for deeper investigation):
5. **src/clifford_fhe/ckks.rs** - Polynomial multiplication implementation
6. **src/clifford_fhe/params.rs** - Parameters being used

The expert should be able to identify the issue from just the first 4 files.
