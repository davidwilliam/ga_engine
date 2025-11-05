# V3 Bootstrapping Design

## Executive Summary

**Goal:** Implement CKKS bootstrapping to enable unlimited multiplication depth for encrypted deep neural networks.

**Use Case:** Encrypted 3D Medical Imaging Classification (GNN with 168 multiplications)

**Status:** V3 Feature - Design Phase

**Timeline:** 2-4 weeks implementation

---

## 1. Why Bootstrapping is Necessary

### Current Limitation
- **Current capacity:** 7 multiplications maximum (with 9 primes)
- **GNN requirement:** 168 multiplications (Layer 1: 16, Layer 2: 128, Layer 3: 24)
- **Gap:** 168 needed vs 7 available = **24× capacity shortage**

### User Requirements
- **Data privacy:** Patient medical scans must be encrypted
- **Model privacy:** Proprietary GNN weights must be encrypted
- **Result:** Requires ciphertext-ciphertext multiplication throughout
- **Solution:** Bootstrapping to refresh noise and enable deep computation

### Why Other Approaches Don't Work
❌ **Plaintext-ciphertext multiplication:** Doesn't protect model privacy
❌ **More primes:** Becomes impractical (would need ~170 primes)
❌ **Shallow networks:** Insufficient accuracy for medical classification
✅ **Bootstrapping:** Standard FHE solution for deep computation

---

## 2. CKKS Bootstrapping Theory

### What is Bootstrapping?

Bootstrapping is a technique to **refresh a noisy ciphertext** by homomorphically decrypting it and re-encrypting it under the same key. This removes accumulated noise while keeping data encrypted.

**Key Insight:** Decryption is just polynomial evaluation - we can evaluate it homomorphically!

### CKKS Decryption Formula

For a CKKS ciphertext `ct = (c0, c1)` encrypted under secret key `s`:

```
plaintext = (c0 + c1 · s) mod q
```

This is a simple linear operation! But there are challenges:
1. We need to compute `s` (secret key) homomorphically
2. We need to handle modulus reduction `mod q`
3. We need to handle scaling factors

### Bootstrapping Pipeline

```
Input: Noisy ciphertext (almost out of levels)
  ↓
1. ModRaise: Raise modulus to higher level
  ↓
2. CoeffToSlot: Transform to evaluation representation
  ↓
3. EvalMod: Homomorphically evaluate decryption formula
  ↓
4. SlotToCoeff: Transform back to coefficient representation
  ↓
5. Output: Fresh ciphertext (full levels restored)
```

---

## 3. CKKS Bootstrapping Components

### 3.1 ModRaise (Modulus Raising)

**Purpose:** Increase modulus to create working room for bootstrap computation

**Operation:**
```rust
fn mod_raise(ct: &Ciphertext, new_moduli: &[u64]) -> Ciphertext {
    // Scale up ciphertext coefficients to higher modulus
    // Preserves plaintext value
}
```

**Example:**
- Input: ct with modulus q = 2^120
- Output: ct with modulus Q = 2^600 (same plaintext)

### 3.2 CoeffToSlot Transformation

**Purpose:** Transform polynomial coefficients to slot values (evaluation form)

**Why Needed:** Bootstrapping operations are easier in evaluation domain

**Operation:**
```rust
fn coeff_to_slot(ct: &Ciphertext) -> Ciphertext {
    // Apply linear transformation using key-switching
    // Maps coefficients to SIMD slots
}
```

**Implementation:**
- Uses pre-computed rotation keys
- Applies FFT-like transformation homomorphically
- O(log N) rotations and additions

### 3.3 EvalMod (Homomorphic Modular Reduction)

**Purpose:** Evaluate the decryption formula homomorphically

**Challenge:** Modular reduction is non-polynomial - need approximation

**Standard Approach:** Use sine approximation

```
x mod q ≈ x - (q/2π) · sin(2πx/q)
```

**Why This Works:**
- sin function is periodic with period 2π
- Approximates the "sawtooth" modular reduction
- Can be evaluated as polynomial using Taylor/Chebyshev series

**Operation:**
```rust
fn eval_mod(ct: &Ciphertext, q: u64) -> Ciphertext {
    // 1. Scale input: x' = 2πx/q
    // 2. Evaluate sin(x') using polynomial approximation
    // 3. Scale result: x - (q/2π) · sin(x')
}
```

**Polynomial Approximation:**
```
sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040 + ...
```

Typically use degree 15-31 polynomial for good accuracy.

### 3.4 SlotToCoeff Transformation

**Purpose:** Transform back from evaluation form to coefficient form

**Operation:**
```rust
fn slot_to_coeff(ct: &Ciphertext) -> Ciphertext {
    // Inverse of CoeffToSlot
    // Maps SIMD slots back to polynomial coefficients
}
```

---

## 4. V3 Architecture

### Module Structure

```
src/clifford_fhe_v3/
├── mod.rs                       // V3 module root
├── bootstrapping/
│   ├── mod.rs                   // Bootstrapping module root
│   ├── bootstrap_context.rs    // Main bootstrap context
│   ├── mod_raise.rs             // Modulus raising
│   ├── coeff_to_slot.rs         // Coefficient to slot transformation
│   ├── eval_mod.rs              // Homomorphic modular reduction
│   ├── slot_to_coeff.rs         // Slot to coefficient transformation
│   ├── sin_approx.rs            // Sine polynomial approximation
│   └── keys.rs                  // Bootstrap key generation (rotation keys)
├── backends/
│   ├── cpu_optimized/           // CPU bootstrap implementation
│   ├── gpu_metal/               // Metal GPU bootstrap (future)
│   └── gpu_cuda/                // CUDA GPU bootstrap (future)
└── tests/
    ├── bootstrap_correctness.rs // Correctness tests
    └── bootstrap_performance.rs // Performance benchmarks
```

### API Design

```rust
/// V3 Bootstrapping Context
pub struct BootstrapContext {
    params: CliffordFHEParams,
    bootstrap_params: BootstrapParams,
    rotation_keys: RotationKeys,
    sin_coeffs: Vec<f64>,
}

/// Bootstrap parameters
pub struct BootstrapParams {
    /// Degree of sine polynomial approximation (15-31)
    pub sin_degree: usize,
    /// Number of levels reserved for bootstrap
    pub bootstrap_levels: usize,
    /// Target precision after bootstrap
    pub target_precision: f64,
}

impl BootstrapContext {
    /// Create new bootstrap context (generates rotation keys)
    pub fn new(params: CliffordFHEParams, bootstrap_params: BootstrapParams) -> Self {
        // Generate rotation keys for all required rotations
        // Precompute sine polynomial coefficients
    }

    /// Bootstrap a ciphertext (refresh noise)
    pub fn bootstrap(&self, ct: &Ciphertext) -> Result<Ciphertext, String> {
        // Full bootstrap pipeline
        let ct_raised = self.mod_raise(ct)?;
        let ct_slots = self.coeff_to_slot(&ct_raised)?;
        let ct_eval = self.eval_mod(&ct_slots)?;
        let ct_coeffs = self.slot_to_coeff(&ct_eval)?;
        Ok(ct_coeffs)
    }

    /// Bootstrap a multivector (8 ciphertexts)
    pub fn bootstrap_multivector(&self, mv: &EncryptedMultivector) -> Result<EncryptedMultivector, String> {
        // Bootstrap all 8 components
        let mut refreshed = Vec::with_capacity(8);
        for ct in &mv.components {
            refreshed.push(self.bootstrap(ct)?);
        }
        Ok(EncryptedMultivector { components: refreshed })
    }

    // Internal operations
    fn mod_raise(&self, ct: &Ciphertext) -> Result<Ciphertext, String>;
    fn coeff_to_slot(&self, ct: &Ciphertext) -> Result<Ciphertext, String>;
    fn eval_mod(&self, ct: &Ciphertext) -> Result<Ciphertext, String>;
    fn slot_to_coeff(&self, ct: &Ciphertext) -> Result<Ciphertext, String>;
}
```

### Integration with V2

V3 builds on V2 infrastructure:

```rust
// Use V2 CKKS operations
use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use crate::clifford_fhe_v2::backends::cpu_optimized::geometric::GeometricContext;

// V3 adds bootstrapping on top
use crate::clifford_fhe_v3::bootstrapping::BootstrapContext;

// Example: Deep GNN with bootstrap
pub fn encrypted_gnn_with_bootstrap(
    input: &EncryptedMultivector,
    weights: &[EncryptedMultivector],
    geom_ctx: &GeometricContext,
    bootstrap_ctx: &BootstrapContext,
) -> Result<EncryptedMultivector, String> {
    let mut x = input.clone();

    for (i, weight) in weights.iter().enumerate() {
        // Geometric product (consumes 1 level)
        x = geom_ctx.geometric_product(&x.components, &weight.components, &eval_key);

        // Bootstrap every few operations to refresh noise
        if (i + 1) % 5 == 0 {
            println!("Bootstrapping after operation {}...", i + 1);
            x = bootstrap_ctx.bootstrap_multivector(&x)?;
        }
    }

    Ok(x)
}
```

---

## 5. Performance Analysis

### Bootstrapping Cost

Based on literature (CKKS bootstrapping papers):

**Single Ciphertext:**
- ModRaise: ~10ms (simple scaling)
- CoeffToSlot: ~200ms (O(log N) rotations)
- EvalMod: ~500ms (polynomial evaluation of degree ~30)
- SlotToCoeff: ~200ms (O(log N) rotations)
- **Total: ~1 second per ciphertext**

**Multivector (8 ciphertexts):**
- Naive: 8 seconds
- Optimized (parallel): ~2 seconds (4× parallelism on M3 Max)

### GNN with Bootstrap

**Scenario:** 168 multiplications, bootstrap every 5 operations

**Cost Breakdown:**
- Geometric products: 168 × 34ms = 5.7 seconds (Metal GPU)
- Bootstraps: 34 × 2 seconds = 68 seconds
- **Total: ~74 seconds per sample**

**With CUDA GPU:**
- Geometric products: 168 × 5.4ms = 0.9 seconds
- Bootstraps: 34 × 2 seconds = 68 seconds
- **Total: ~69 seconds per sample**

**Bottleneck:** Bootstrap dominates (92% of time)

### Optimization Strategies

1. **Reduce Bootstrap Frequency:**
   - Use deeper parameter sets (more primes)
   - Bootstrap every 7 operations instead of 5
   - Reduces bootstrap count: 168/7 = 24 (saves 20 seconds)

2. **GPU Bootstrap:**
   - CoeffToSlot/SlotToCoeff on GPU: ~50ms each (4× speedup)
   - EvalMod on GPU: ~100ms (5× speedup)
   - **Per-multivector cost: ~500ms (4× speedup)**
   - **GNN total: ~17 seconds**

3. **Batching (SIMD):**
   - Bootstrap 512 samples at once
   - Amortized cost: ~33ms per sample
   - **GNN throughput: 0.33 seconds per sample (512× batch)**

---

## 6. Implementation Roadmap

### Phase 1: CPU Bootstrap Foundation (Week 1)
**Goal:** Basic bootstrap working on CPU

**Tasks:**
1. ✅ Design V3 module structure
2. ⬜ Implement `BootstrapContext` skeleton
3. ⬜ Implement `mod_raise()` - modulus raising
4. ⬜ Implement sine polynomial approximation (`sin_approx.rs`)
5. ⬜ Implement basic rotation operations
6. ⬜ Write unit tests for individual components

**Deliverable:** Bootstrap compiles, basic operations work

### Phase 2: CoeffToSlot/SlotToCoeff (Week 2)
**Goal:** Implement linear transformations

**Tasks:**
1. ⬜ Generate rotation keys for required rotations
2. ⬜ Implement `coeff_to_slot()` using FFT-like structure
3. ⬜ Implement `slot_to_coeff()` (inverse transformation)
4. ⬜ Test transformations compose to identity
5. ⬜ Benchmark rotation performance

**Deliverable:** CoeffToSlot/SlotToCoeff working correctly

### Phase 3: EvalMod (Week 2-3)
**Goal:** Implement homomorphic modular reduction

**Tasks:**
1. ⬜ Implement polynomial evaluation for sine
2. ⬜ Implement `eval_mod()` using sine approximation
3. ⬜ Test accuracy of modular reduction
4. ⬜ Tune polynomial degree for precision vs performance
5. ⬜ Integrate all components into `bootstrap()`

**Deliverable:** Full bootstrap pipeline working

### Phase 4: Testing & Integration (Week 3)
**Goal:** Validate correctness and integrate with medical imaging

**Tasks:**
1. ⬜ Write correctness tests (bootstrap then decrypt)
2. ⬜ Test on encrypted multivectors
3. ⬜ Implement `bootstrap_multivector()`
4. ⬜ Test noise refresh (measure noise before/after)
5. ⬜ Integrate with encrypted GNN

**Deliverable:** Bootstrap working end-to-end

### Phase 5: GPU Optimization (Week 4)
**Goal:** GPU-accelerated bootstrap

**Tasks:**
1. ⬜ Port EvalMod to Metal/CUDA
2. ⬜ Port CoeffToSlot/SlotToCoeff to GPU
3. ⬜ Implement batched bootstrap (SIMD)
4. ⬜ Benchmark GPU bootstrap performance
5. ⬜ Optimize memory transfers

**Deliverable:** GPU bootstrap ~4× faster than CPU

### Phase 6: Medical Imaging Demo (Week 4)
**Goal:** Full encrypted GNN with bootstrap

**Tasks:**
1. ⬜ Implement deep GNN (1→16→8→3)
2. ⬜ Add bootstrap calls between layers
3. ⬜ Test on synthetic dataset (sphere/cube/pyramid)
4. ⬜ Measure end-to-end latency
5. ⬜ Create visualization and documentation

**Deliverable:** Working encrypted medical imaging demo

---

## 7. Testing Strategy

### Unit Tests

```rust
#[test]
fn test_mod_raise() {
    // Test that modulus raising preserves plaintext
    let ctx = CkksContext::new(params);
    let pt = vec![1.0, 2.0, 3.0, 4.0];
    let ct = ctx.encrypt(&pt);
    let ct_raised = bootstrap_ctx.mod_raise(&ct);
    let pt_dec = ctx.decrypt(&ct_raised);
    assert_approx_eq!(pt, pt_dec, 0.01);
}

#[test]
fn test_coeff_to_slot_inverse() {
    // Test that CoeffToSlot ∘ SlotToCoeff = identity
    let ct = ctx.encrypt(&plaintext);
    let ct_slots = bootstrap_ctx.coeff_to_slot(&ct);
    let ct_recovered = bootstrap_ctx.slot_to_coeff(&ct_slots);
    let pt_dec = ctx.decrypt(&ct_recovered);
    assert_approx_eq!(plaintext, pt_dec, 0.01);
}

#[test]
fn test_eval_mod_accuracy() {
    // Test sine approximation quality
    for q in test_moduli {
        for x in test_values {
            let expected = x % q;
            let approx = eval_mod(x, q);
            assert!((expected - approx).abs() < 0.1);
        }
    }
}

#[test]
fn test_bootstrap_correctness() {
    // Test full bootstrap pipeline
    let pt = vec![1.0, 2.0, 3.0, 4.0];
    let ct = ctx.encrypt(&pt);

    // Intentionally add noise
    let ct_noisy = add_artificial_noise(&ct);

    // Bootstrap should refresh
    let ct_fresh = bootstrap_ctx.bootstrap(&ct_noisy).unwrap();
    let pt_dec = ctx.decrypt(&ct_fresh);

    assert_approx_eq!(pt, pt_dec, 0.01);
}

#[test]
fn test_bootstrap_noise_refresh() {
    // Measure noise before/after bootstrap
    let ct = ctx.encrypt(&plaintext);
    let noise_before = estimate_noise(&ct, &secret_key);

    let ct_bootstrapped = bootstrap_ctx.bootstrap(&ct).unwrap();
    let noise_after = estimate_noise(&ct_bootstrapped, &secret_key);

    assert!(noise_after < noise_before / 10, "Bootstrap should reduce noise");
}
```

### Integration Tests

```rust
#[test]
fn test_deep_gnn_with_bootstrap() {
    // Test 168 multiplications with bootstrap
    let input = encrypt_multivector(&test_data);
    let weights = encrypt_weights(&gnn_weights);

    let result = encrypted_gnn_with_bootstrap(
        &input,
        &weights,
        &geom_ctx,
        &bootstrap_ctx,
    ).unwrap();

    let decrypted = decrypt_multivector(&result);
    let expected = plaintext_gnn(&test_data, &gnn_weights);

    assert_approx_eq!(decrypted, expected, 0.1);
}
```

### Performance Benchmarks

```rust
#[bench]
fn bench_bootstrap_single_ciphertext(b: &mut Bencher) {
    b.iter(|| {
        bootstrap_ctx.bootstrap(&test_ciphertext)
    });
}

#[bench]
fn bench_bootstrap_multivector(b: &mut Bencher) {
    b.iter(|| {
        bootstrap_ctx.bootstrap_multivector(&test_multivector)
    });
}

#[bench]
fn bench_encrypted_gnn_full(b: &mut Bencher) {
    b.iter(|| {
        encrypted_gnn_with_bootstrap(&input, &weights, &geom_ctx, &bootstrap_ctx)
    });
}
```

---

## 8. Key Parameters

### Bootstrap-Specific Parameters

```rust
impl BootstrapParams {
    /// Conservative bootstrap parameters (high precision)
    pub fn conservative() -> Self {
        BootstrapParams {
            sin_degree: 31,              // Degree-31 polynomial (high accuracy)
            bootstrap_levels: 15,         // Reserve 15 levels for bootstrap ops
            target_precision: 1e-6,       // 6 decimal places
        }
    }

    /// Balanced bootstrap parameters (good precision + performance)
    pub fn balanced() -> Self {
        BootstrapParams {
            sin_degree: 23,              // Degree-23 polynomial
            bootstrap_levels: 12,         // Reserve 12 levels
            target_precision: 1e-4,       // 4 decimal places (good for medical)
        }
    }

    /// Fast bootstrap parameters (lower precision, faster)
    pub fn fast() -> Self {
        BootstrapParams {
            sin_degree: 15,              // Degree-15 polynomial
            bootstrap_levels: 10,         // Reserve 10 levels
            target_precision: 1e-2,       // 2 decimal places
        }
    }
}
```

### FHE Parameters for Bootstrap

Current V2 parameters need adjustment for bootstrap:

```rust
impl CliffordFHEParams {
    /// V3 parameters: Support bootstrap + 7 multiplications between bootstraps
    pub fn new_v3_bootstrap_8192() -> Self {
        CliffordFHEParams {
            n: 8192,                     // Larger ring for better precision
            moduli: vec![
                // 22 primes total:
                // - 12 for bootstrap operations
                // - 7 for computation between bootstraps
                // - 3 for safety margin
                // 60-bit primes...
            ],
            scale: 1u64 << 40,           // 40-bit scaling factor
        }
    }

    /// V3 parameters: Deeper computation (bootstrap every 7 ops)
    pub fn new_v3_bootstrap_16384() -> Self {
        CliffordFHEParams {
            n: 16384,                    // Even larger for best precision
            moduli: vec![
                // 30 primes total:
                // - 15 for bootstrap
                // - 10 for computation
                // - 5 for safety
            ],
            scale: 1u64 << 45,           // 45-bit scaling factor
        }
    }
}
```

---

## 9. References & Resources

### Papers

1. **"Bootstrapping for Approximate Homomorphic Encryption"** (Cheon et al., 2018)
   - Original CKKS bootstrapping paper
   - Describes CoeffToSlot/SlotToCoeff using FFT
   - Sine approximation for modular reduction

2. **"Better Bootstrapping for Approximate Homomorphic Encryption"** (Cheon et al., 2019)
   - Improved bootstrapping (BTS algorithm)
   - Faster CoeffToSlot using sparse FFT
   - Better precision analysis

3. **"High-Precision Bootstrapping of RNS-CKKS"** (Lee et al., 2021)
   - RNS-friendly bootstrapping (our case!)
   - Handles RNS representation properly
   - Improved precision guarantees

### Open-Source Implementations

1. **SEAL (Microsoft):**
   - C++ implementation of CKKS + bootstrap
   - Reference for algorithm details
   - Good test vectors

2. **HEAAN:**
   - Original CKKS implementation by Cheon et al.
   - Includes bootstrap
   - Academic reference implementation

3. **OpenFHE:**
   - Modern FHE library with bootstrap
   - Good documentation
   - Performance benchmarks

---

## 10. Success Metrics

### Correctness
- ✅ Bootstrap preserves plaintext (error < 0.01)
- ✅ Noise is reduced by 10× after bootstrap
- ✅ 168 multiplications with bootstrap succeed
- ✅ Encrypted GNN accuracy > 95%

### Performance (CPU)
- ✅ Single ciphertext bootstrap: < 1.5 seconds
- ✅ Multivector bootstrap: < 3 seconds (parallel)
- ✅ Full GNN inference: < 90 seconds per sample

### Performance (GPU)
- ✅ Single ciphertext bootstrap: < 400ms (Metal/CUDA)
- ✅ Multivector bootstrap: < 800ms
- ✅ Full GNN inference: < 20 seconds per sample

### Performance (GPU + SIMD)
- ✅ Batched bootstrap: < 5ms per sample (512× batch)
- ✅ Full GNN throughput: ~0.5 seconds per sample
- ✅ 10,000 medical scans: < 2 hours

---

## 11. Risk Analysis

### Technical Risks

**1. Precision Loss ⚠️**
- Bootstrap introduces approximation error
- Multiple bootstraps accumulate error
- Mitigation: Use degree-31 polynomial, test extensively

**2. Performance Bottleneck ⚠️**
- Bootstrap is 50-100× slower than geometric product
- Dominates total runtime
- Mitigation: GPU optimization, reduce bootstrap frequency

**3. Key Size Explosion ⚠️**
- Rotation keys are large (gigabytes)
- Memory constraint on GPU
- Mitigation: Generate keys on-demand, use sparse rotations

**4. Parameter Selection ⚠️**
- Wrong parameters → bootstrap fails or loses precision
- Requires careful analysis
- Mitigation: Follow CKKS bootstrapping literature, test thoroughly

### Implementation Risks

**1. Complexity ⚠️**
- Bootstrap is most complex FHE operation
- Easy to introduce subtle bugs
- Mitigation: Extensive unit tests, compare against SEAL

**2. Timeline ⚠️**
- 2-4 weeks is aggressive for first bootstrap
- May take longer to debug
- Mitigation: Focus on correctness first, optimize later

**3. GPU Optimization ⚠️**
- Metal/CUDA bootstrap is non-trivial
- May not achieve 4× speedup initially
- Mitigation: Start with CPU, GPU is optimization

---

## 12. V3 Feature Roadmap

Once bootstrapping is working, V3 will include:

### Core Features
- ✅ CKKS Bootstrapping (unlimited depth)
- ✅ Deep Encrypted GNN (168+ multiplications)
- ✅ GPU Bootstrap (Metal + CUDA)
- ✅ SIMD Batching Integration (512× throughput)

### Applications
- ✅ Encrypted 3D Medical Imaging Classification
- ✅ Encrypted Geometric Deep Learning
- ✅ Privacy-Preserving Model Inference (proprietary models)

### Performance Targets
- ✅ Single sample: < 20 seconds (GPU + bootstrap)
- ✅ Batched: < 0.5 seconds per sample (512× SIMD)
- ✅ 10,000 scans: < 2 hours (production-ready)

---

## Next Steps

1. **Create V3 module structure** ✅ (this document)
2. **Begin Phase 1:** Implement `BootstrapContext` skeleton
3. **Implement modulus raising** (simplest component)
4. **Implement sine approximation** (test in isolation)
5. **Iterate through roadmap phases 1-6**

---

**Status:** 🎯 Ready to begin implementation
**Timeline:** 2-4 weeks to complete bootstrap
**Next Session:** Start Phase 1 - CPU Bootstrap Foundation
