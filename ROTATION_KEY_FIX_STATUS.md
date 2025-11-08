# Rotation Key Fix - Current Status

## What We Fixed

**File:** `src/clifford_fhe_v2/backends/gpu_metal/rotation_keys.rs:283-297`

**Bug:** Rotation keys had wrong sign for `B^t·s_k` term

**Before:**
```rust
rlk0[t] = -a_t·s + e_t + B^t·s_k   // ❌ Missing negation
```

**After:**
```rust
rlk0[t] = -B^t·s_k + a_t·s + e_t   // ✅ Correct formula
```

## Tests Passing

✅ **test_cpu_vs_gpu_rotation** (N=1024, simple rotation)
- CPU rotation: PASS
- GPU rotation: PASS
- Both produce identical results

✅ **test_v3_full_bootstrap** (N=8192, 41 primes, CPU-only)
- Takes 6-10 minutes (expected for CPU)
- Bootstrap completes correctly
- This uses V3's CPU rotation keys (which were already correct)

## Tests Failing

❌ **test_metal_gpu_bootstrap** (N=1024, 20 primes, Metal GPU)
- Errors: ~193k (same magnitude as before the fix!)
- CoeffToSlot: 48s (very slow, should be <5s)
- SlotToCoeff: 11s (slow)
- Scale explosion: `3.52e13` instead of `~1e12`

## Investigation Findings

### Key Switching Formula Compatibility

Both V2 Metal GPU and V3 CPU use the **SAME** key switching formula:

**V3 CPU** (`rotation.rs:107-114`):
```rust
// The RLK encrypts -B^t·s(X^g), so we SUBTRACT term0
c0[i] = c0[i].sub(&term0[i]);  // SUBTRACT
c1_new[i] = c1_new[i].add(&term1[i]);  // ADD
```

**V2 Metal GPU** (`ckks.rs:1300-1318`):
```rust
// c0_final -= term0 (matches V3 CPU implementation)
c0_final[i] = diff;  // SUBTRACT
c1_final[i] = sum;   // ADD
```

**Conclusion:** The fix we applied is correct for BOTH V2 and V3!

### Possible Causes of test_metal_gpu_bootstrap Failure

1. **Scale Management Bug in V2 Bootstrap**
   - Scale is exploding: `3.52e13` vs expected `1.1e12`
   - This is similar to the V3 scale bug we fixed earlier
   - Possible issue: `encode_diagonal_for_metal` might have wrong scale encoding

2. **Cached Compilation Artifacts**
   - The test might be using old rotation keys from before the fix
   - Running `cargo clean` forces rebuild but hits cmake dependency issues

3. **Separate Bug in V2 Metal GPU Bootstrap**
   - The rotation key fix is correct
   - But there might be another bug in CoeffToSlot/SlotToCoeff

## Next Steps to Debug

### Option 1: Check Scale Management in V2 Bootstrap

The scale explosion (`3.52e13`) suggests the plaintext encoding scale might be wrong in `bootstrap.rs:104`:

```rust
let q_top = moduli[current.level] as f64;
let pt_diag1 = encode_diagonal_for_metal(&diag1, q_top, n, current.level, &moduli)?;
```

This looks correct (using `q_top`), but we should verify the encoding function.

### Option 2: Add Debug Logging

Add logging to track:
- Rotation key values after generation
- Scale at each butterfly level
- Plaintext scale vs ciphertext scale

### Option 3: Compare with V3 CPU Bootstrap

V3 CPU bootstrap works perfectly. We should:
1. Run V3 CPU CoeffToSlot/SlotToCoeff only (no EvalMod)
2. Compare intermediate values with V2 Metal GPU
3. Find where they diverge

## Commands for Testing

### Fast Test (GPU rotation only, N=1024):
```bash
cargo run --release --features v2,v2-gpu-metal,v3 --example test_cpu_vs_gpu_rotation
```
**Expected:** ~5-10 seconds, both CPU and GPU pass

### Full V3 Bootstrap (CPU-only, N=8192):
```bash
cargo run --release --features v2,v2-gpu-metal,v3 --example test_v3_full_bootstrap
```
**Expected:** ~6-10 minutes, completes successfully

### Metal GPU Bootstrap (CURRENTLY FAILING, N=1024):
```bash
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap
```
**Current Status:** ~193k errors, scale explosion

## Hypothesis

The rotation key fix is **correct**. The `test_metal_gpu_bootstrap` failure is likely due to a **separate scale management bug** in V2's `bootstrap.rs`, similar to the scale bug we fixed in V3's CoeffToSlot/SlotToCoeff.

**Evidence:**
- Scale is exploding (3.52e13 instead of 1.1e12)
- Operations are very slow (48s for CoeffToSlot vs expected <5s)
- Error magnitude (~193k) matches the pre-fix rotation errors

**Recommendation:** Investigate `encode_diagonal_for_metal` and the multiply_plain operation in V2 Metal GPU bootstrap to find the scale bug.

---

**Document Created:** 2025-01-08
**Status:** Rotation key fix verified correct, but V2 Metal GPU bootstrap has separate scale bug
