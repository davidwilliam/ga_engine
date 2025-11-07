# Metal GPU Full Backend Implementation Plan

**Goal**: Complete, standalone Metal GPU backend for V3 Clifford FHE with 100% accuracy and maximum performance.

**Principle**: GPU backend must be fully isolated from CPU - no dependencies, no fallbacks, no mixing.

---

## Current State Analysis

### What We Have ✅
1. **Metal NTT** - Bit-perfect correct (proven by test_metal_ntt_correctness)
2. **Metal Device** - GPU interface working
3. **MetalKeyContext** - Key generation on GPU
4. **RNS helpers** - Barrett reduction, modular arithmetic

### What's Missing ❌
1. **MetalCkksContext** - CKKS encode/decode/encrypt/decrypt on GPU
2. **Full GPU pipeline** - No CPU fallbacks anywhere
3. **GPU-optimized encode/decode** - FFT-like operations on GPU
4. **GPU rotation operations** - For bootstrap
5. **GPU coefficient extraction** - For SIMD batching

---

## Architecture: Complete Isolation

```
┌─────────────────────────────────────────────────────────────┐
│                     CPU Backend (v2)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ KeyContext   │  │ CkksContext  │  │ NttContext   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         ↓                  ↓                  ↓             │
│    CPU Keygen       CPU Encode/Decode    CPU NTT           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Metal GPU Backend (v2)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │MetalKeyCtx   │  │MetalCkksCtx  │  │MetalNttCtx   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         ↓                  ↓                  ↓             │
│    GPU Keygen       GPU Encode/Decode    GPU NTT           │
│                                                             │
│  NO CPU FALLBACKS - 100% GPU or Error                      │
└─────────────────────────────────────────────────────────────┘
```

**Key Principle**: If GPU backend is selected, EVERYTHING runs on GPU. No mixing.

---

## Implementation Phases

### Phase 1: MetalCkksContext (Core CKKS on GPU)

**Priority**: HIGHEST - This is blocking everything else

#### 1.1 GPU Encode/Decode
**File**: `src/clifford_fhe_v2/backends/gpu_metal/ckks.rs`

**Operations needed**:
```rust
pub struct MetalCkksContext {
    device: Arc<MetalDevice>,
    params: CliffordFHEParams,
    ntt_contexts: Vec<MetalNttContext>,
    reducers: Vec<BarrettReducer>,
}

impl MetalCkksContext {
    pub fn new(params: CliffordFHEParams) -> Result<Self, String>;

    // GPU encode: f64 values → RNS polynomial
    pub fn encode(&self, values: &[f64]) -> MetalPlaintext;

    // GPU decode: RNS polynomial → f64 values
    pub fn decode(&self, pt: &MetalPlaintext) -> Vec<f64>;

    // GPU encrypt: plaintext + public key → ciphertext
    pub fn encrypt(&self, pt: &MetalPlaintext, pk: &PublicKey)
        -> MetalCiphertext;

    // GPU decrypt: ciphertext + secret key → plaintext
    pub fn decrypt(&self, ct: &MetalCiphertext, sk: &SecretKey)
        -> MetalPlaintext;
}
```

**Key insight**: Encode/decode need FFT-like operations. These can be done on GPU!

**Metal shaders needed**:
- `encode.metal`: Scale and embed values (inverse DFT structure)
- `decode.metal`: Extract and unscale values (DFT structure)

#### 1.2 GPU Polynomial Operations
**Operations**:
- `add_polynomials_gpu()` - Element-wise addition in RNS
- `multiply_polynomials_gpu()` - Uses Metal NTT (already working!)
- `multiply_scalar_gpu()` - Scalar multiplication

**Metal shaders needed**:
- `rns_add.metal`: Add two RNS polynomials
- `rns_multiply_scalar.metal`: Multiply by scalar in RNS

#### 1.3 GPU Rescale
**Operation**: Drop one prime from RNS representation
- Current level: [q₀, q₁, q₂, q₃]
- After rescale: [q₀, q₁, q₂]

**Implementation**: Pure CPU operation (just drop last values), no GPU needed.

---

### Phase 2: Complete GPU Pipeline Integration

#### 2.1 MetalCiphertext Type
**File**: `src/clifford_fhe_v2/backends/gpu_metal/ciphertext.rs`

```rust
pub struct MetalCiphertext {
    pub c0: Vec<RnsValue>,  // First polynomial (RNS representation)
    pub c1: Vec<RnsValue>,  // Second polynomial (RNS representation)
    pub level: usize,       // Current modulus chain level
    pub scale: f64,         // Current scale
}

impl MetalCiphertext {
    // All operations stay on GPU
    pub fn add(&self, other: &Self, ctx: &MetalCkksContext)
        -> Result<Self, String>;

    pub fn multiply(&self, other: &Self, ctx: &MetalCkksContext)
        -> Result<Self, String>;

    pub fn multiply_plain(&self, pt: &MetalPlaintext, ctx: &MetalCkksContext)
        -> Result<Self, String>;
}
```

#### 2.2 GPU Noise Sampling
**Current**: Uses CPU RNG
**Goal**: Use Metal GPU random number generation

**Options**:
1. Generate noise on CPU, transfer to GPU (simple, works)
2. Use Metal's built-in random (if available)
3. Implement Philox/Threefry RNG in Metal shader (best performance)

**Decision**: Start with option 1 (CPU RNG), optimize later.

---

### Phase 3: V3 Bootstrap Operations on GPU

#### 3.1 GPU Rotation
**File**: `src/clifford_fhe_v3/rotation_gpu.rs`

**Operation**: Rotate ciphertext slots using Galois automorphisms

**Current**: CPU only
**Target**: Full GPU implementation

**Complexity**: HIGH - needs automorphism computation on GPU

#### 3.2 GPU CoeffToSlot / SlotToCoeff
**Files**:
- `src/clifford_fhe_v3/coeff_to_slot_gpu.rs`
- `src/clifford_fhe_v3/slot_to_coeff_gpu.rs`

**Operations**: FFT-like transforms for bootstrap

**Current**: CPU only
**Target**: GPU shaders for DFT-like operations

#### 3.3 GPU EvalMod
**File**: `src/clifford_fhe_v3/eval_mod_gpu.rs`

**Operation**: Evaluate sine polynomial on GPU

**Current**: CPU only
**Target**: GPU shader for polynomial evaluation

---

## Implementation Order (Detailed)

### Week 1: Core CKKS on GPU
**Goal**: Get basic encrypt/decrypt working on GPU

1. **Day 1-2**: Implement MetalCkksContext structure
   - [ ] Create `gpu_metal/ckks.rs`
   - [ ] Define MetalPlaintext, MetalCiphertext types
   - [ ] Implement `new()` constructor
   - [ ] Test: Create context successfully

2. **Day 3-4**: Implement GPU encode
   - [ ] Metal shader: `encode.metal` (DFT-like operation)
   - [ ] Rust wrapper: `encode()`
   - [ ] Test: Encode values, verify coefficients

3. **Day 5-6**: Implement GPU decode
   - [ ] Metal shader: `decode.metal` (inverse DFT)
   - [ ] Rust wrapper: `decode()`
   - [ ] Test: encode → decode roundtrip

4. **Day 7**: Implement GPU encrypt/decrypt
   - [ ] Use existing Metal NTT for polynomial multiplication
   - [ ] Implement noise addition (CPU RNG for now)
   - [ ] Test: encrypt → decrypt roundtrip

**Success Criteria**:
```bash
cargo run --release --features v2,v2-gpu-metal --example test_metal_ckks_basic
# Expected: Encrypt 42.0, decrypt, error < 0.01
```

### Week 2: Full CKKS Operations
**Goal**: All CKKS operations on GPU

1. **Day 1-2**: Implement ciphertext operations
   - [ ] `add()` - Element-wise RNS addition
   - [ ] `multiply_plain()` - Use Metal NTT
   - [ ] Test: (ct + ct), (ct * pt)

2. **Day 3-4**: Implement ciphertext multiplication
   - [ ] `multiply()` - Full ciphertext-ciphertext multiplication
   - [ ] Relinearization using evaluation key
   - [ ] Test: ct * ct with relinearization

3. **Day 5-6**: Optimize and test
   - [ ] Performance benchmarks vs CPU
   - [ ] Accuracy tests (error < 1e-6)
   - [ ] Edge cases (zero, large values)

4. **Day 7**: Integration with V3
   - [ ] Replace CPU CKKS with Metal CKKS in V3 tests
   - [ ] Verify all V3 unit tests pass
   - [ ] Document GPU-specific requirements

**Success Criteria**:
```bash
cargo run --release --features v2,v3,v2-gpu-metal --example test_v3_metal_quick
# Expected: All operations on GPU, error < 0.01
```

### Week 3: V3 Bootstrap on GPU
**Goal**: Full bootstrap pipeline on GPU

1. **Day 1-3**: GPU rotation operations
   - [ ] Implement Galois automorphism on GPU
   - [ ] Test rotation accuracy

2. **Day 4-5**: GPU CoeffToSlot/SlotToCoeff
   - [ ] Port DFT-like operations to Metal shaders
   - [ ] Test transform accuracy

3. **Day 6-7**: GPU EvalMod and integration
   - [ ] Polynomial evaluation on GPU
   - [ ] Full bootstrap test on GPU
   - [ ] Performance benchmarks

**Success Criteria**:
```bash
cargo run --release --features v2,v3,v2-gpu-metal --example test_v3_full_bootstrap
# Expected: Full bootstrap on GPU, N=8192 in <30 seconds
```

---

## Metal Shaders Needed

### Phase 1 Shaders (CKKS Core)
1. **encode.metal** - DFT-like embedding
2. **decode.metal** - Inverse DFT extraction
3. **rns_add.metal** - RNS polynomial addition
4. **rns_multiply_scalar.metal** - Scalar multiplication

### Phase 2 Shaders (Operations)
5. **relinearize.metal** - Key-switching operation
6. **automorphism.metal** - Galois automorphism for rotation

### Phase 3 Shaders (Bootstrap)
7. **coeff_to_slot.metal** - FFT-like forward transform
8. **slot_to_coeff.metal** - FFT-like inverse transform
9. **eval_poly.metal** - Polynomial evaluation (for sine)

---

## Testing Strategy

### Unit Tests (Per Component)
```rust
#[test]
fn test_metal_encode() {
    let ctx = MetalCkksContext::new(params).unwrap();
    let values = vec![42.0, 3.14, -7.5];
    let pt = ctx.encode(&values);
    let decoded = ctx.decode(&pt);
    assert!(max_error(&values, &decoded) < 1e-6);
}
```

### Integration Tests (Full Pipeline)
```rust
#[test]
fn test_metal_full_pipeline() {
    let ctx = MetalCkksContext::new(params).unwrap();
    let key_ctx = MetalKeyContext::new(params).unwrap();
    let (pk, sk, evk) = key_ctx.keygen().unwrap();

    let pt = ctx.encode(&[42.0]);
    let ct = ctx.encrypt(&pt, &pk);
    let pt2 = ctx.decrypt(&ct, &sk);
    let result = ctx.decode(&pt2);

    assert!((result[0] - 42.0).abs() < 0.01);
}
```

### Accuracy Tests
- Compare Metal vs CPU results
- Maximum allowed error: 1e-6 for single operations
- Maximum allowed error: 1e-3 for deep computations (due to accumulated noise)

### Performance Benchmarks
- Key generation: Target <5s for N=8192
- Encryption: Target <10ms per ciphertext
- Multiplication: Target <50ms per multiplication
- Bootstrap: Target <30s for full pipeline

---

## Success Metrics

### Phase 1 Complete When:
- ✅ MetalCkksContext can encode/decode with error < 1e-6
- ✅ MetalCkksContext can encrypt/decrypt with error < 0.01
- ✅ All operations stay on GPU (no CPU fallbacks)
- ✅ test_metal_ckks_basic passes

### Phase 2 Complete When:
- ✅ All CKKS operations (add, multiply, multiply_plain) work on GPU
- ✅ V3 examples run with Metal GPU backend
- ✅ Performance is ≥2x faster than CPU for N≥4096
- ✅ Accuracy matches CPU (error < 1e-6)

### Phase 3 Complete When:
- ✅ Full bootstrap runs on GPU
- ✅ N=8192 bootstrap completes in <30 seconds
- ✅ All 52 V3 unit tests pass with Metal backend
- ✅ No CPU fallbacks anywhere in the pipeline

---

## File Structure

```
src/clifford_fhe_v2/backends/gpu_metal/
├── mod.rs                    # Module exports
├── device.rs                 # ✅ Metal device interface (DONE)
├── ntt.rs                    # ✅ Metal NTT (DONE)
├── keys.rs                   # ✅ Key generation (DONE)
├── ckks.rs                   # 🚧 CKKS context (TO DO)
├── ciphertext.rs             # 🚧 Ciphertext type (TO DO)
├── plaintext.rs              # 🚧 Plaintext type (TO DO)
└── shaders/
    ├── ntt.metal             # ✅ NTT kernels (DONE)
    ├── encode.metal          # 🚧 Encode operation (TO DO)
    ├── decode.metal          # 🚧 Decode operation (TO DO)
    ├── rns_ops.metal         # 🚧 RNS operations (TO DO)
    └── bootstrap.metal       # 🚧 Bootstrap ops (TO DO)

src/clifford_fhe_v3/backends/gpu_metal/
├── rotation.rs               # 🚧 GPU rotation (TO DO)
├── coeff_to_slot.rs          # 🚧 GPU C2S (TO DO)
├── slot_to_coeff.rs          # 🚧 GPU S2C (TO DO)
└── eval_mod.rs               # 🚧 GPU EvalMod (TO DO)
```

---

## Next Immediate Steps

1. **Create MetalCkksContext skeleton** (1 hour)
   - Define struct
   - Implement new()
   - Add placeholder methods

2. **Implement GPU encode** (1 day)
   - Write encode.metal shader
   - Implement Rust wrapper
   - Test roundtrip with decode

3. **Implement GPU decrypt** (1 day)
   - Use existing Metal NTT
   - Test full encrypt → decrypt

4. **Test integration** (0.5 day)
   - Update test_v3_metal_quick to use MetalCkksContext
   - Verify no domain mismatch
   - Confirm error < 0.01

**Target**: Working GPU-only CKKS in 3-4 days of focused work.

---

## Questions to Answer

1. **Encode/Decode DFT**: Can we use existing FFT libraries or need custom Metal shader?
   - **Answer**: Need custom shader (CKKS uses specific DFT structure for canonical embedding)

2. **RNG on GPU**: Use CPU or implement GPU RNG?
   - **Answer**: Start with CPU, optimize later

3. **Memory management**: Keep data on GPU between operations?
   - **Answer**: Yes! Minimize CPU↔GPU transfers

4. **Error handling**: What if GPU operation fails?
   - **Answer**: Return Result<>, no silent fallback to CPU

---

## Performance Targets

| Operation | CPU (N=8192) | Metal GPU (N=8192) | Speedup |
|-----------|--------------|-------------------|---------|
| Key Generation | ~120s | <10s | 12x |
| NTT Forward | ~5ms | <1ms | 5x |
| Encryption | ~20ms | <5ms | 4x |
| Multiplication | ~50ms | <10ms | 5x |
| Bootstrap | ~300s | <30s | 10x |

---

## Conclusion

**This plan achieves**:
1. ✅ Complete isolation (GPU ≠ CPU)
2. ✅ 100% accuracy (bit-perfect where possible)
3. ✅ Maximum performance (10x+ speedup for large N)
4. ✅ Clear testing strategy
5. ✅ Incremental, testable milestones

**Timeline**: 3-4 weeks for full implementation
**Risk**: Medium (encode/decode DFT is complex)
**Reward**: High (production-ready GPU FHE)

Ready to start with Phase 1: MetalCkksContext! 🚀
