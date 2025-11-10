# V4: Packed Slot-Interleaved Layout - Implementation Plan

**Branch**: `v4-packed-layout`
**Status**: Planning → Implementation
**Goal**: Achieve CKKS-equivalent memory footprint while maintaining full Clifford algebra capabilities

---

## Executive Summary

**Problem**: Current V1/V2/V3 use naive component-separate layout (8 ciphertexts per multivector), resulting in 8× memory overhead vs standard CKKS.

**Solution**: Implement slot-interleaved packing where all 8 components of a multivector are stored in a single CKKS ciphertext, with slots arranged as: `[s, e1, e2, e3, e12, e23, e31, I, s, e1, ...]` (repeating pattern).

**Expected Impact**:
- **Memory**: 8× reduction (same as CKKS)
- **Batching**: 8× improvement (512 multivectors vs 64)
- **Per-operation latency**: 2-4× slower (due to rotations + diagonal multiplies)
- **Throughput**: 2-4× improvement (batching wins over latency)
- **Adoption**: Removes main barrier - "same size as CKKS, but with GA"

---

## Architectural Decisions

### 1. V4 Position in Codebase

**Location**: `src/clifford_fhe_v4/`

**Relationship to other versions**:
- V1: Frozen (Paper 1 baseline)
- V2: Multi-backend foundation (CPU/Metal/CUDA)
- V3: Bootstrap using V2 backend (naive layout)
- **V4: Packed layout using V2 backend** (new encoding layer)

**Key principle**: V4 USES V2's GPU backends (NTT, rescaling, key switching), just like V3 does. V4 adds a NEW packing/unpacking layer on top of V2's operations.

### 2. Feature Flag

```toml
v4 = ["v2"]  # V4 requires V2 backend (same as V3)
```

**Build commands**:
```bash
# CPU
cargo build --release --features v2,v4

# Metal GPU
cargo build --release --features v2,v2-gpu-metal,v4

# CUDA GPU
cargo build --release --features v2,v2-gpu-cuda,v4
```

### 3. Data Layout

**Slot arrangement** (N=1024, 512 multivector slots):
```
Slot:    0    1    2    3    4     5     6     7  |  8    9   10   11  ...
Value:  s0   e1₀  e2₀  e3₀  e12₀  e23₀  e31₀  I₀ | s1   e1₁  e2₁  e3₁  ...
```

**Properties**:
- Stride = 8 (distance between same component of different multivectors)
- Component offset: scalar=0, e1=1, e2=2, e3=3, e12=4, e23=5, e31=6, I=7
- Batching capacity: N/8 multivectors per ciphertext (1024/8 = 128, but with CKKS packing: 512)

---

## Implementation Phases

### Phase 1: Core Packing Infrastructure (Week 1)

**Goal**: Establish packed multivector type and basic encode/decode operations.

#### 1.1 Create V4 Directory Structure
```
src/clifford_fhe_v4/
├── mod.rs                  # Feature gate, public API
├── packed_multivector.rs   # PackedMultivector ciphertext type
├── packing.rs              # Encode/decode between naive ↔ packed
├── params.rs               # V4-specific parameters (reuse V2's mostly)
└── tests.rs                # Unit tests
```

#### 1.2 Define `PackedMultivector` Type
```rust
pub struct PackedMultivector {
    /// Single CKKS ciphertext containing all 8 components interleaved
    pub ct: Ciphertext,  // From V2

    /// Number of multivectors batched in this ciphertext
    pub batch_size: usize,

    /// Encryption parameters
    pub n: usize,
    pub num_primes: usize,
    pub level: usize,
    pub scale: f64,
}
```

#### 1.3 Implement Packing/Unpacking
```rust
// Convert from naive (8 separate ciphertexts) to packed (1 interleaved)
pub fn pack_multivector(
    naive: &NaiveMultivector,  // V2/V3 type
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String>

// Convert from packed to naive (for testing/validation)
pub fn unpack_multivector(
    packed: &PackedMultivector,
    ckks_ctx: &CudaCkksContext,
) -> Result<NaiveMultivector, String>
```

**Testing**:
- Round-trip: naive → packed → naive (should be identical)
- Golden compare with V2/V3 operations

---

### Phase 2: Geometric Operations with Diagonal Multiply + Rotation (Week 2-3)

**Goal**: Implement geometric product using the multiplication table as diagonal multiplies + rotations.

#### 2.1 Multiplication Table Analysis

**Clifford algebra Cl(3,0) multiplication table**:
```
        1    e1   e2   e3   e12  e23  e31  I
1       1    e1   e2   e3   e12  e23  e31  I
e1      e1   1    e12 -e31  e2  -I    e3   e23
e2      e2  -e12  1    e23  e1   e3  -I   -e31
e3      e3   e31 -e23  1   -I    e2   e1   e12
e12     e12 -e2   e1   I    1   -e31  e23  e3
e23     e23  I   -e3   e2   e31  1   -e12  e1
e31     e31 -e3   I    e1  -e23  e12  1    e2
I       I   -e23  e31 -e12 -e3  -e1  -e2   1
```

**Each result component** is a linear combination of products:
```
result.scalar = a.scalar * b.scalar + a.e1 * b.e1 + a.e2 * b.e2 + a.e3 * b.e3
                - a.e12 * b.e12 - a.e23 * b.e23 - a.e31 * b.e31 - a.I * b.I

result.e1 = a.scalar * b.e1 + a.e1 * b.scalar - a.e2 * b.e12 + a.e3 * b.e31
            + a.e12 * b.e2 - a.e23 * b.I + a.e31 * b.e3 - a.I * b.e23

... (6 more equations)
```

**Optimization**: Many terms can be grouped. Expect ~12-20 operations per component (not 64).

#### 2.2 Implement Diagonal Multiply Operation
```rust
/// Extract component i, multiply by plaintext mask, zero other components
pub fn diagonal_multiply(
    ct: &PackedMultivector,
    component_idx: usize,  // 0=scalar, 1=e1, ..., 7=I
    plaintext_mask: &[f64],
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String>
```

**How it works**:
- Plaintext mask = `[c0, c1, c2, c3, c4, c5, c6, c7, c0, c1, ...]` (repeating)
- Multiply ciphertext by this plaintext
- Result: component i scaled, others zeroed (if mask is selective)

#### 2.3 Implement Rotation Operation
```rust
/// Rotate slots by k positions (for component alignment)
pub fn rotate_slots(
    ct: &PackedMultivector,
    k: i32,  // Rotation amount
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String>
```

**Uses V2's rotation infrastructure**:
- Galois automorphism keys (already exist in V2)
- `apply_galois_automorphism_gpu()` from V2
- `key_switch_hoisted_gpu()` from V2

#### 2.4 Implement Geometric Product
```rust
pub fn geometric_product_packed(
    a: &PackedMultivector,
    b: &PackedMultivector,
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    // Based on multiplication table
    // Example for scalar component:

    let mut result = zero_ciphertext();

    // a.scalar * b.scalar
    let term = extract_component(a, 0)?;     // diagonal multiply
    let term = extract_component(b, 0)?;     // diagonal multiply
    let term = multiply(term_a, term_b)?;    // homomorphic multiply
    result = add(result, term)?;

    // a.e1 * b.e1 (contributes +1 to scalar)
    let term_a = extract_component(a, 1)?;
    let term_b = extract_component(b, 1)?;
    let term_b = rotate(term_b, 0)?;         // Already aligned
    let term = multiply(term_a, term_b)?;
    result = add(result, term)?;

    // a.e2 * b.e2 (contributes +1 to scalar)
    // ... similar

    // a.e12 * b.e12 (contributes -1 to scalar)
    let term_a = extract_component(a, 4)?;
    let term_b = extract_component(b, 4)?;
    let term = multiply(term_a, term_b)?;
    let term = negate(term)?;                // Sign flip
    result = add(result, term)?;

    // ... continue for all 8 output components

    Ok(result)
}
```

**Optimization opportunities**:
- Fuse operations (extract + multiply in one step)
- Pre-compute plaintext masks
- Batch rotations (rotate once, use multiple times)
- Eliminate identity operations

#### 2.5 Implement Other Geometric Operations
```rust
// All built on top of geometric_product_packed
pub fn reverse_packed(...) -> Result<PackedMultivector, String>
pub fn wedge_product_packed(...) -> Result<PackedMultivector, String>
pub fn inner_product_packed(...) -> Result<PackedMultivector, String>
pub fn rotation_packed(...) -> Result<PackedMultivector, String>
```

**Testing**:
- Golden compare vs V2/V3 naive implementation
- Verify algebraic properties (associativity, etc.)
- Measure performance vs naive layout

---

### Phase 3: Bootstrap Integration (Week 4-5)

**Goal**: Adapt V3's bootstrap to work with packed layout.

#### 3.1 Create Bootstrap Module
```
src/clifford_fhe_v4/bootstrapping/
├── mod.rs
├── coeff_to_slot.rs      # Adapt V3's CoeffToSlot for packed layout
├── slot_to_coeff.rs      # Adapt V3's SlotToCoeff for packed layout
├── eval_mod.rs           # Adapt V3's EvalMod for packed layout
└── keys.rs               # Rotation key generation (reuse V2/V3)
```

#### 3.2 Key Insight: Bootstrap Operates on CKKS Ciphertexts

**V3 bootstrap already works on individual CKKS ciphertexts**. Since packed layout uses ONE ciphertext, bootstrap should "just work" with:
- Different slot interpretation
- Same rotation keys
- Same rescaling logic

**Changes needed**:
- Component extraction during CoeffToSlot (use diagonal multiply)
- Component recombination in SlotToCoeff (use rotations + add)
- EvalMod probably unchanged (operates on slots independently)

#### 3.3 Implement Packed Bootstrap
```rust
pub fn bootstrap_packed(
    ct: &PackedMultivector,
    bootstrap_ctx: &BootstrapContext,
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    // ModRaise (same as V3)
    let ct_raised = mod_raise(ct)?;

    // CoeffToSlot (adapted for packed slots)
    let ct_slots = coeff_to_slot_packed(ct_raised, bootstrap_ctx, ckks_ctx)?;

    // EvalMod (should work as-is on packed slots)
    let ct_mod = eval_mod_packed(ct_slots, bootstrap_ctx, ckks_ctx)?;

    // SlotToCoeff (adapted for packed slots)
    let ct_result = slot_to_coeff_packed(ct_mod, bootstrap_ctx, ckks_ctx)?;

    Ok(ct_result)
}
```

**Testing**:
- Verify noise reduction
- Compare error vs V3 CPU reference
- Measure performance (should be similar to V3, maybe faster due to better memory locality)

---

### Phase 4: Optimization & GPU Tuning (Week 6)

**Goal**: Minimize rotation + diagonal multiply overhead.

#### 4.1 Multiplication Table Pruning
- Analyze which operations are identity/negation (no computation needed)
- Group terms with same rotation offset
- Pre-compute plaintext masks on GPU
- Estimate: reduce from 64 theoretical ops to ~12-16 actual ops

#### 4.2 Rotation Key Caching
- Pre-load all rotation keys to GPU VRAM
- Avoid CPU↔GPU transfers during operations
- Use V2's existing key caching infrastructure

#### 4.3 Kernel Fusion (CUDA/Metal)
- Fuse diagonal multiply + rotation into single kernel
- Fuse multiple rotations (if same offset used multiple times)
- Reduce intermediate memory allocations

#### 4.4 Batching Optimization
- Process 512 multivectors at once (vs 64 in V3)
- Amortize rotation costs over larger batches
- Expected: 8× throughput improvement

---

### Phase 5: Benchmarking & Validation (Week 7)

**Goal**: Comprehensive comparison vs V2/V3 and documentation.

#### 5.1 Correctness Tests
```bash
# Golden compare against V3
cargo test --lib --features v2,v2-gpu-cuda,v4 test_v4_vs_v3_golden_compare

# Test all geometric operations
cargo test --lib --features v2,v4 clifford_fhe_v4::geometric_ops

# Test bootstrap
cargo test --lib --features v2,v4 clifford_fhe_v4::bootstrapping
```

#### 5.2 Performance Benchmarks
```bash
# V4 CUDA GPU
cargo run --release --features v2,v2-gpu-cuda,v4 --example test_v4_packed_performance

# Compare vs V3
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_v3_performance
```

**Metrics to measure**:
- Memory per multivector (should be 8× smaller)
- Geometric product latency (expected: 2-4× slower)
- Batched throughput (expected: 2-4× faster overall)
- Bootstrap time (expected: similar or 10-20% faster)

#### 5.3 Create Examples
```bash
examples/
├── test_v4_packed_basic.rs           # Basic pack/unpack
├── test_v4_geometric_ops.rs          # All geometric operations
├── test_v4_bootstrap.rs              # Full bootstrap
└── test_v4_vs_v3_comparison.rs       # Side-by-side comparison
```

---

## Expected Performance (Conservative Estimates)

### Memory Footprint

| Version | Multivector Size | Batch Size | Total Memory |
|---------|------------------|------------|--------------|
| V3 Naive | 8 ciphertexts | 64 multivectors | 512 ciphertexts |
| **V4 Packed** | **1 ciphertext** | **512 multivectors** | **512 ciphertexts** |

**Same total memory, but 8× more multivectors!**

### Operation Latency (Single Multivector, CUDA GPU)

| Operation | V3 Naive | V4 Packed (Est.) | Overhead |
|-----------|----------|------------------|----------|
| Geometric Product | 5.7ms | 10-15ms | 2-3× |
| Rotation | 11.4ms | 20-30ms | 2-3× |
| Bootstrap | 11.95s | 10-14s | ~same |

### Throughput (Batched, CUDA GPU)

| Metric | V3 Naive | V4 Packed (Est.) | Improvement |
|--------|----------|------------------|-------------|
| Multivectors/ciphertext | 64 | 512 | 8× |
| Ops per second (batched) | 11,228 | ~30,000-40,000 | **3-4×** |

**Key insight**: Even though individual ops are 2-3× slower, batching wins massively.

---

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_pack_unpack_roundtrip() {
        // naive → packed → naive should be identical
    }

    #[test]
    fn test_geometric_product_golden_compare() {
        // V4 packed vs V3 naive should give same result
    }

    #[test]
    fn test_all_geometric_operations() {
        // Test reverse, wedge, inner, rotation, etc.
    }

    #[test]
    fn test_bootstrap_noise_reduction() {
        // Verify bootstrap reduces noise as expected
    }
}
```

### Integration Tests
```rust
#[test]
fn test_v4_encrypted_3d_classification() {
    // Run full neural network with packed layout
    // Compare accuracy vs V3
}

#[test]
fn test_v4_batching() {
    // Process 512 multivectors
    // Verify throughput improvement
}
```

---

## Success Criteria

### Must Have (MVP)
- ✅ Pack/unpack working correctly
- ✅ Geometric product with golden compare vs V3
- ✅ All 7 geometric operations implemented
- ✅ Bootstrap working with <5% error vs V3
- ✅ Memory footprint = 1/8th of V3

### Should Have (Production Ready)
- ✅ Performance: 2-4× slowdown per op (not worse)
- ✅ Throughput: 2-4× improvement with batching
- ✅ CUDA + Metal backends both working
- ✅ Comprehensive tests (>50 tests)
- ✅ Example applications

### Nice to Have (Future Work)
- ⚪ Kernel fusion optimizations
- ⚪ Multiplication table pruning to <12 ops
- ⚪ Automatic batching API
- ⚪ Hybrid packing strategies (2-4 components per ct)

---

## Risk Mitigation

### Risk 1: Rotation Overhead Too High
**Mitigation**:
- Benchmark early (Phase 2.3)
- If >5× slowdown, implement kernel fusion
- Fallback: Hybrid packing (4 components per ct = 2× memory, less rotation)

### Risk 2: Bootstrap Breaks with Packed Layout
**Mitigation**:
- V3 bootstrap operates on CKKS ciphertexts (should be agnostic to slot meaning)
- Test bootstrap FIRST with simple packing (Phase 3.2)
- If issues, might need slot-aware CoeffToSlot/SlotToCoeff

### Risk 3: GPU Memory Limits
**Mitigation**:
- 512 multivector batches = 512 ciphertexts (same as V3's 64 batches)
- No additional memory vs V3
- If issues, reduce batch size

---

## Documentation Updates

After V4 implementation, update:

1. **README.md** - Add V4 to performance tables
2. **ARCHITECTURE.md** - Add V4 section, explain packing strategy
3. **BENCHMARKS.md** - V3 vs V4 comparison
4. **COMMANDS.md** - V4 build/test commands
5. **FEATURE_FLAGS.md** - Add v4 feature
6. **Create V4_PACKED_LAYOUT.md** - Deep dive into implementation

---

## Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Core Infrastructure | Packing/unpacking working, tests passing |
| 2-3 | Geometric Operations | All 7 ops with golden compare |
| 4-5 | Bootstrap | Full bootstrap working |
| 6 | Optimization | Performance tuning |
| 7 | Testing & Docs | Comprehensive benchmarks, updated docs |

**Total: ~7 weeks to production-ready V4**

---

## First Implementation Steps (Next 1-2 Days)

### Step 1: Create V4 Directory Structure
```bash
mkdir -p src/clifford_fhe_v4/bootstrapping
touch src/clifford_fhe_v4/mod.rs
touch src/clifford_fhe_v4/packed_multivector.rs
touch src/clifford_fhe_v4/packing.rs
touch src/clifford_fhe_v4/params.rs
touch src/clifford_fhe_v4/geometric_ops.rs
```

### Step 2: Define Feature Flag in Cargo.toml
```toml
v4 = ["v2"]
```

### Step 3: Create `PackedMultivector` Type
Implement basic struct in `packed_multivector.rs`

### Step 4: Implement Pack/Unpack (Placeholder)
Create stub functions that will be fleshed out

### Step 5: Write First Test
```rust
#[test]
fn test_create_packed_multivector() {
    // Sanity check - can we create the type?
}
```

---

## Questions to Resolve During Implementation

1. **Exact slot layout** - Confirm stride=8 works for all operations
2. **Rotation key set** - Which rotations are needed? (probably k ∈ {1,2,3,4,5,6,7})
3. **Plaintext mask storage** - Pre-compute and cache? Or generate on-the-fly?
4. **Error accumulation** - Does packing affect noise growth significantly?
5. **Batching API** - Automatic batching vs manual batching?

---

## Success Metrics (Post-Implementation)

### Performance Target
- **Latency**: 2-4× slower per operation vs V3 (acceptable)
- **Throughput**: 2-4× faster with batching vs V3 (GOAL)
- **Memory**: 8× reduction vs V3 (CRITICAL)

### Adoption Target
- **Same ciphertext size as CKKS** ✅
- **Same performance as CKKS for scalar ops** ✅
- **Plus vectors + quaternions + rotations** ✅
- **Compelling value proposition for CKKS users** ✅
