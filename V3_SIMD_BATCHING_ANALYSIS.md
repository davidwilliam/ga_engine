# V3 SIMD Batching Analysis

**Question:** Is V3 fully taking advantage of SIMD batching?

**Short Answer:** **No.** V3 Phase 3 implements single-sample operations only. SIMD batching is planned but not yet implemented.

---

## Current Status

### What V3 Phase 3 Has

✅ **Rotation keys and key-switching** - Working for single ciphertext
✅ **CoeffToSlot/SlotToCoeff** - Working for single ciphertext
✅ **Canonical embedding** - Proper orbit-ordered encoding
✅ **Single-sample operations** - Each ciphertext encrypts one multivector (8 components)

### What V3 Phase 3 Does NOT Have

❌ **SIMD batching** - Not implemented
❌ **Slot packing** - Each ciphertext uses only ~8 of N/2 available slots
❌ **Batch rotation** - Rotations operate on single samples
❌ **Parallel multivector operations** - No batch geometric product

---

## Slot Utilization Analysis

### Current Slot Usage

**CKKS Parameters:**
- Ring dimension: N = 8192 (production target)
- Available slots: N/2 = 4096 slots
- Multivector components: 8 (scalar + 3 vectors + 3 bivectors + trivector)

**Current encoding:**
```
Slot 0: component 0 (scalar)
Slot 1: component 1 (e1)
Slot 2: component 2 (e2)
Slot 3: component 3 (e3)
Slot 4: component 4 (e12)
Slot 5: component 5 (e23)
Slot 6: component 6 (e31)
Slot 7: component 7 (e123)
Slots 8-4095: UNUSED (99.8% wasted!)
```

**Slot utilization:** 8 / 4096 = **0.2%**

### Optimal SIMD Batching

**With slot packing:**
```
Batch size: 4096 / 8 = 512 multivectors per ciphertext

Slot 0:   multivector[0].component[0]
Slot 1:   multivector[0].component[1]
...
Slot 7:   multivector[0].component[7]
Slot 8:   multivector[1].component[0]
Slot 9:   multivector[1].component[1]
...
Slot 15:  multivector[1].component[7]
...
Slot 4088: multivector[511].component[0]
...
Slot 4095: multivector[511].component[7]
```

**Slot utilization:** 4096 / 4096 = **100%**
**Throughput multiplier:** **512×**

---

## Performance Impact

### Current V3 Performance (Projected, Single Sample)

| Operation | Time | Throughput |
|-----------|------|------------|
| Bootstrap (CPU) | 2000ms | 0.5 samples/sec |
| Bootstrap (GPU) | 500ms | 2 samples/sec |
| Rotation | 10ms | 100 operations/sec |
| CoeffToSlot | 150ms | 6.7 transforms/sec |

### With SIMD Batching (Projected, 512× Batch)

| Operation | Time | Throughput | Speedup |
|-----------|------|------------|---------|
| Bootstrap (CPU) | 2000ms | **256 samples/sec** | 512× |
| Bootstrap (GPU) | 500ms | **1024 samples/sec** | 512× |
| Rotation | 10ms | **51,200 ops/sec** | 512× |
| CoeffToSlot | 150ms | **3413 transforms/sec** | 512× |

**Cost per sample:**
- Current: 2000ms/sample (CPU)
- With batching: 3.9ms/sample (CPU), **0.98ms/sample (GPU)**

---

## Why SIMD Batching Isn't Implemented Yet

### Technical Requirements

1. **Slot permutation via rotations** ✅ (Phase 3 complete)
   - Rotation keys working
   - Galois automorphisms correct
   - CoeffToSlot/SlotToCoeff operational

2. **Batch-aware encoding** ❌ (Not implemented)
   - Need to pack multiple multivectors into slots
   - Requires careful component ordering
   - Must maintain Galois automorphism compatibility

3. **Batch geometric product** ❌ (Not implemented)
   - Requires component-wise slot operations
   - Need diagonal matrix multiplication
   - Rotation-based component extraction

4. **Bootstrap slot management** ❌ (Phase 4)
   - EvalMod must work on all slots simultaneously
   - Diagonal matrices for CoeffToSlot/SlotToCoeff
   - Careful noise management across batch

### Implementation Complexity

**Why deferred:**
- Phase 3 focused on correctness for single samples
- SIMD batching adds significant complexity
- Need working bootstrap first (Phase 4)
- Then optimize with batching (Phase 5+)

---

## Implementation Plan for SIMD Batching

### Phase 5: SIMD Batching (After Phase 4 Bootstrap)

**Step 1: Batch Encoding/Decoding**
```rust
pub struct BatchedMultivector {
    /// 512 multivectors packed into CKKS slots
    /// Layout: [mv0.c0, mv0.c1, ..., mv0.c7, mv1.c0, ..., mv511.c7]
    ciphertext: Ciphertext,
    batch_size: usize,  // = 512 for N=8192
}

impl BatchedMultivector {
    /// Encode 512 multivectors into one ciphertext
    pub fn encode_batch(multivectors: &[[f64; 8]; 512]) -> Self;

    /// Decode to 512 multivectors
    pub fn decode_batch(&self) -> Vec<[f64; 8]>;
}
```

**Step 2: Component Extraction via Rotation**
```rust
impl BatchedMultivector {
    /// Extract component i from all 512 multivectors
    /// Uses rotation by i to move component to position 0, 8, 16, ...
    pub fn extract_component(&self, component: usize,
                             rotation_keys: &RotationKeys) -> Ciphertext;

    /// Reassemble from extracted components
    pub fn from_components(components: &[Ciphertext; 8]) -> Self;
}
```

**Step 3: Batch Geometric Product**
```rust
impl GeometricOps {
    /// Geometric product on 512 multivector pairs simultaneously
    /// Result: 512 output multivectors in one ciphertext
    pub fn geometric_product_batched(
        &self,
        a_batch: &BatchedMultivector,
        b_batch: &BatchedMultivector,
        rotation_keys: &RotationKeys,
        evk: &EvaluationKey,
    ) -> BatchedMultivector;
}
```

**Step 4: Batch Bootstrap**
```rust
impl BootstrapContext {
    /// Bootstrap 512 ciphertexts simultaneously
    /// All slots refreshed in parallel
    pub fn bootstrap_batched(
        &self,
        ct_batch: &BatchedMultivector,
    ) -> Result<BatchedMultivector, String>;
}
```

---

## Challenges and Solutions

### Challenge 1: Component Ordering

**Problem:** Geometric product requires component-wise operations, but components are interleaved:
```
[mv0.c0, mv0.c1, ..., mv0.c7, mv1.c0, mv1.c1, ...]
```

**Solution:** Use rotations to extract components:
```rust
// Extract all c0 components: rotate by 0 (no-op)
let c0_all = rotate(batch, 0);  // [mv0.c0, mv0.c1, ..., mv0.c7, mv1.c0, ...]

// Extract all c1 components: rotate by 1
let c1_all = rotate(batch, 1);  // [mv0.c1, mv0.c2, ..., mv0.c7, mv1.c0, mv1.c1, ...]

// Mask to keep only every 8th slot starting at position 0
let c0_only = multiply_by_mask(c0_all, mask_0);  // [mv0.c0, 0, 0, ..., 0, mv1.c0, 0, ...]
```

### Challenge 2: Galois Automorphism Stride

**Problem:** Rotations by k slots in standard CKKS don't directly translate to component extraction with interleaving.

**Solution:** Use stride-aware rotation indices:
```rust
// To extract component i from interleaved batch of stride 8:
let component_rotation = i;  // Rotation by i positions
let extract_mask = create_mask_stride_8(offset: i);
```

### Challenge 3: Bootstrap Noise Accumulation

**Problem:** All 512 samples in batch accumulate noise together. One noisy sample contaminates entire batch.

**Solution:**
1. Monitor noise before batching (reject noisy samples)
2. Use homomorphic comparison to detect outliers
3. Split batch if noise varies significantly

### Challenge 4: Geometric Product Complexity

**Problem:** Geometric product has 64 component operations (8×8). With rotations for extraction, this multiplies.

**Solution:**
1. Precompute rotation patterns
2. Reuse extracted components across operations
3. Use diagonal matrices to combine operations

---

## Estimated Performance with SIMD Batching

### Deep GNN Inference (168 multiplications per sample)

**Current V3 (no batching):**
- CPU: 168 × 2s bootstrap ≈ 336 seconds/sample ❌ (impractical)
- GPU: 168 × 0.5s bootstrap ≈ 84 seconds/sample ❌ (impractical)

**V3 with SIMD batching (512× batch):**
- CPU: (168 × 2s) / 512 ≈ **0.65 seconds/sample** ✅
- GPU: (168 × 0.5s) / 512 ≈ **0.16 seconds/sample** ✅

**Throughput:**
- CPU: 1.5 samples/sec = 5400 samples/hour
- GPU: 6.2 samples/sec = 22,000 samples/hour

---

## Recommendations

### Immediate (Phase 4)
1. ✅ Complete bootstrap without batching
2. ✅ Verify correctness on single samples
3. ✅ Optimize single-sample performance

### Short-term (Phase 5)
1. Implement batch encoding/decoding
2. Component extraction via rotation
3. Batch geometric product
4. Test with small batches (32-64 samples)

### Medium-term (Phase 6)
1. Full 512× batching
2. Optimize rotation patterns
3. GPU-accelerated batch operations
4. End-to-end batch GNN inference

### Long-term (Future)
1. Dynamic batch sizing based on noise
2. Heterogeneous batching (different operations per sample)
3. Pipelined batching (overlap computation with I/O)

---

## Comparison with Literature

### SEAL (Microsoft)
- **Batching support:** ✅ Full SIMD batching via BFV/BGV
- **CKKS batching:** ✅ Slot-level parallelism
- **Usage:** Standard for batch operations

### OpenFHE
- **Batching support:** ✅ Full SIMD support
- **Optimization:** Automatic batch detection
- **Performance:** Near-optimal slot utilization

### HEAAN
- **Batching support:** ✅ Native CKKS batching
- **Rotation-based:** ✅ Same approach we plan
- **Production:** Used in real systems with batching

**Our position:** Currently behind in batching, but architectural foundation (Phase 3 rotations) enables future implementation.

---

## Conclusion

**Current State:**
- V3 Phase 3: ❌ No SIMD batching (0.2% slot utilization)
- Single-sample operations only
- 512× throughput potential unrealized

**Path Forward:**
1. Complete Phase 4 (bootstrap, single-sample)
2. Implement Phase 5 (SIMD batching)
3. Target: 512× throughput increase
4. Enable practical deep GNN inference

**Timeline:**
- Phase 4: 4-6 days
- Phase 5 (batching): 1-2 weeks
- Full optimization: 3-4 weeks

**Critical for:** Deep neural network inference (168+ multiplications) becomes practical only with batching.

---

**Bottom Line:** V3 Phase 3 is NOT using SIMD batching. This is intentional—correctness first, optimization second. SIMD batching is planned for Phase 5 after completing Phase 4 bootstrap.
