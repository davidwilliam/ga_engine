# Metal GPU Integration & Bootstrapping Status

## Part 1: Metal GPU Integration ✅ Architecture Complete

### What Was Accomplished

**Device Reuse Architecture** ✅
- Modified `MetalNttContext` to accept existing `MetalDevice` via Arc
- Added `new_with_device()` constructor
- Implemented NTT context caching in `MetalEncryptionContext`
- One Metal device shared across all NTT contexts

**Files Modified:**
- `src/clifford_fhe_v2/backends/gpu_metal/ntt.rs` - Added `new_with_device()`
- `src/medical_imaging/encrypted_metal.rs` - Arc-wrapped device + caching

### Current Status

**✅ Device reuse working** - No more creating 48 Metal devices!
**⚠️  Correctness issue** - Metal NTT produces large decryption errors (475K instead of 0.000000)

**The Issue:**
Metal NTT operations run without errors, but the decrypted result is wrong. This suggests:
1. Root of unity calculation might be incorrect for Metal
2. Twiddle factor generation might differ from CPU
3. Modular arithmetic in Metal shaders might have overflow
4. Scale/normalization handling might differ

**The Fix:** Debug Metal NTT correctness (separate from medical imaging)
- Compare Metal NTT output vs CPU NTT output on same input
- Check primitive root calculation
- Verify modular arithmetic in Metal shaders

**For Now:** Using CPU NTT (415ms, correct)

---

## Part 2: Bootstrapping for Unlimited Depth

### The Problem

**GNN requires 168 multiplications:**
- Layer 1: 16 multiplications (1→16 neurons)
- Layer 2: 128 multiplications (16×8 = 128 weight×input pairs)
- Layer 3: 24 multiplications (8×3 = 24 weight×input pairs)
- **Total: 168 multiplications**

**Current capacity:**
- `new_test_ntt_1024()`: 3 primes → 1 multiplication
- `new_test_ntt_2048()`: 5 primes → 3 multiplications
- `new_test_ntt_4096()`: 7 primes → 5 multiplications
- `new_128bit()` (N=8192): 9 primes → **7 multiplications** max

**Gap:** Need 168 multiplications, have capacity for 7

### Bootstrapping

**What it is:**
- Homomorphically decrypt then re-encrypt a ciphertext
- Refreshes the noise, allowing unlimited multiplications
- Standard technique in FHE, but complex to implement

**Implementation complexity:**
- 2-4 weeks of careful implementation
- Requires additional cryptographic operations
- Performance overhead: ~1 second per bootstrap
- Not currently in your V2 codebase

### Alternative Solutions

#### Option A: Simplify GNN Architecture

**Reduce multiplication depth:**
```rust
// Instead of full geometric product in every neuron,
// use simpler operations:

// Layer 1: 1 → 4 neurons (not 16)
//   4 multiplications instead of 16

// Layer 2: 4 → 2 neurons (not 16→8)
//   8 multiplications instead of 128

// Layer 3: 2 → 3 neurons (not 8→3)
//   6 multiplications instead of 24

// Total: 18 multiplications (fits in depth-7!)
```

**Pros:**
- Works with existing infrastructure
- No bootstrapping needed
- Still demonstrates encrypted GNN
- Faster inference

**Cons:**
- Lower model capacity
- May need more training to achieve same accuracy

#### Option B: Use Plaintext-Ciphertext Multiplication

**Key insight:** Weights don't need to be encrypted!

```rust
// Instead of:
let weight_enc = encrypt(weight);
let result = encrypted_geometric_product(weight_enc, input_enc);  // Uses 1 level

// Do:
let result = plaintext_ct_multiply(weight, input_enc);  // Uses 0 levels!
```

**Benefit:** Plaintext-ciphertext multiplication doesn't consume depth!

**Status:** Not implemented in your V2 `multiply_ciphertexts`
- Would need to add `multiply_plaintext_ciphertext()` function
- Significantly reduces depth requirements
- Layer 1: 0 depth (plaintext weights)
- Layer 2: 0 depth (plaintext weights)
- Layer 3: 0 depth (plaintext weights)
- **Total depth needed: 0!** (Only additions, which are free)

**This is the correct solution!**

#### Option C: Implement Bootstrapping

**What's needed:**
1. Key-switching algorithm
2. Modulus switching
3. Slot-to-coeff and coeff-to-slot transformations
4. Sin/cos evaluation for homomorphic decryption
5. Careful noise analysis

**Effort:** 2-4 weeks
**Performance:** +1 second per bootstrap operation
**Value:** Unlimited depth

---

## Recommendations

### Immediate: Implement Plaintext-Ciphertext Multiplication ⭐

This is the **standard solution** for encrypted neural networks:

**Algorithm:**
```rust
pub fn multiply_plaintext(
    ct: &Ciphertext,
    pt: &Plaintext,
    key_ctx: &KeyContext,
) -> Ciphertext {
    // Multiply each ciphertext component by plaintext
    // No relinearization needed (stays degree-1)
    // No rescaling needed (scale doesn't change much)

    let c0_new = multiply_polynomials(&ct.c0, &pt.coeffs, key_ctx);
    let c1_new = multiply_polynomials(&ct.c1, &pt.coeffs, key_ctx);

    Ciphertext::new(c0_new, c1_new, ct.level, ct.scale * pt.scale)
}
```

**Implementation effort:** 1-2 days
**Benefit:** Enables full GNN without depth issues
**Standard:** This is how all encrypted neural networks work

### Future: Debug Metal NTT

Once GNN is working with plaintext-ciphertext multiplication:
1. Create isolated Metal NTT test
2. Compare Metal vs CPU outputs
3. Fix correctness issue
4. Integrate into medical imaging

**Expected result:** 12× speedup (415ms → 34ms per geometric product)

### Later: Bootstrapping (If Needed)

Only implement if you need:
- Very deep networks (>7 multiplications)
- Ciphertext-ciphertext multiplication chains
- Specific applications requiring it

For most encrypted ML applications, plaintext-ciphertext multiplication is sufficient.

---

## Summary

**Metal GPU:**
- ✅ Device reuse architecture complete
- ⚠️  Needs correctness debugging
- CPU NTT works fine for now (415ms)

**Bootstrapping:**
- Not needed if you implement plaintext-ciphertext multiplication
- Plaintext-ct multiplication is the standard solution
- Enables full GNN without depth limitations

**Next Step:**
Implement `multiply_plaintext_ciphertext()` in V2 to enable encrypted GNN.
