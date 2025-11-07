# GPU-Accelerated Key Generation Implementation Plan

## Executive Summary

**Goal**: Implement V3 bootstrap with production parameters (N=8192, 16-20 primes) using GPU acceleration for key generation.

**Problem**: CPU-based NTT context creation for N=8192 takes >20 minutes, making production parameters unusable.

**Solution**: Implement GPU-accelerated key generation for Metal (macOS) and CUDA (Runpod.io).

---

## Architecture Strategy

### Three-Tier Approach

```
┌─────────────────────────────────────────────────────────────┐
│ CPU (Small Parameters)                                      │
│ • N=1024 or N=2048                                         │
│ • Fast demos, unit tests, development                      │
│ • Key generation: <5 seconds                              │
│ • Use existing cpu_optimized backend                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Metal GPU (Production Parameters - macOS)                   │
│ • N=8192, 16-20 primes                                     │
│ • Key generation: ~10-60 seconds (10-100× speedup)        │
│ • Apple Silicon M1/M2/M3                                    │
│ • Test locally on your Mac                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ CUDA GPU (Production Parameters - Cloud)                    │
│ • N=8192, 16-20 primes                                     │
│ • Key generation: ~5-30 seconds (faster than Metal)        │
│ • NVIDIA GPUs on Runpod.io                                 │
│ • Production deployment target                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: CPU Small Parameters (Immediate - 30 minutes)

### Goal
Create working CPU demo with parameters that actually complete in reasonable time.

### Tasks

1. **Create new parameter set**: `new_v3_demo_cpu()`
   ```rust
   // In src/clifford_fhe_v3/params.rs
   pub fn new_v3_demo_cpu() -> Self {
       let n = 1024;  // or 2048
       let moduli = vec![
           // 1 special modulus (60-bit)
           1152921504606748673,  // NTT-friendly for N=1024

           // 10 scaling primes (41-bit) for bootstrap
           // (Will find NTT-friendly primes for N=1024)
           ...
       ];
       // bootstrap_levels: 10
       // computation_levels: should be >= 3
   }
   ```

2. **Update example to use CPU parameters**
   ```rust
   // In examples/test_v3_cpu_demo.rs (new file)
   let params = CliffordFHEParams::new_v3_demo_cpu();
   // Should complete in <5 seconds total
   ```

3. **Test and validate**
   ```bash
   cargo run --release --features v2,v3 --example test_v3_cpu_demo
   ```

**Expected outcome**: Working CPU demo that completes in seconds, not minutes.

---

## Phase 2: Metal GPU Key Generation (Priority 1 - 4-6 hours)

### Goal
Implement full GPU-accelerated key generation for Metal backend.

### Current Assets

✅ **Already have**:
- Metal NTT shaders ([ntt.metal](src/clifford_fhe_v2/backends/gpu_metal/shaders/ntt.metal))
- `pow_mod` for twiddle factor generation
- Metal device management
- NTT forward/inverse transforms on GPU

❌ **Need to implement**:
- GPU-based NTT context initialization
- GPU key generation pipeline
- Memory management for large key structures

### Implementation Steps

#### Step 1: GPU NTT Context Initialization (2 hours)

**File**: `src/clifford_fhe_v2/backends/gpu_metal/ntt.rs`

**Add shader** for parallel twiddle factor generation:
```metal
// In ntt.metal
kernel void generate_twiddle_factors(
    device ulong* twiddles [[buffer(0)]],
    constant ulong& n [[buffer(1)]],
    constant ulong& q [[buffer(2)]],
    constant ulong& root [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        // twiddles[i] = root^i mod q
        twiddles[gid] = pow_mod(root, gid, q);
    }
}
```

**Rust wrapper**:
```rust
pub struct MetalNttContext {
    device: Device,
    twiddles: Buffer,  // Keep on GPU!
    inv_twiddles: Buffer,
    n: usize,
    q: u64,
}

impl MetalNttContext {
    pub fn new(device: &Device, n: usize, q: u64) -> Self {
        // 1. Find primitive root of unity
        let root = find_primitive_root(n, q);

        // 2. Dispatch GPU kernel to generate twiddles in parallel
        let pipeline = device.new_compute_pipeline_state_with_function(&kernel);
        let twiddles = device.new_buffer((n * 8) as u64, MTLResourceOptions::StorageModePrivate);

        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&twiddles), 0);
        encoder.set_bytes(1, 8, &n as *const usize as *const _);
        encoder.set_bytes(2, 8, &q as *const u64 as *const _);
        encoder.set_bytes(3, 8, &root as *const u64 as *const _);

        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), ...);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // 3. Generate inverse twiddles (root^-1)
        // ... similar process

        Self { device, twiddles, inv_twiddles, n, q }
    }
}
```

**Expected speedup**: N=8192 twiddles generated in ~0.1 seconds (vs 2+ minutes on CPU)

#### Step 2: GPU Key Generation (2-3 hours)

**File**: `src/clifford_fhe_v2/backends/gpu_metal/keys.rs` (new)

**Structure**:
```rust
pub struct MetalKeyContext {
    device: Device,
    params: CliffordFHEParams,
    ntt_contexts: Vec<MetalNttContext>,  // All on GPU!
}

impl MetalKeyContext {
    pub fn new(params: CliffordFHEParams) -> Self {
        let device = Device::system_default().unwrap();

        println!("Creating Metal NTT contexts on GPU...");

        // Create ALL NTT contexts in parallel on GPU
        let ntt_contexts: Vec<MetalNttContext> = params.moduli
            .iter()
            .map(|&q| MetalNttContext::new(&device, params.n, q))
            .collect();

        Self { device, params, ntt_contexts }
    }

    pub fn keygen(&self) -> (PublicKey, SecretKey, EvaluationKey) {
        // 1. Sample secret key (can stay on CPU or move to GPU)
        // 2. Sample uniform/error polynomials (GPU RNG)
        // 3. NTT multiply on GPU
        // 4. Generate evaluation key (GPU polynomial ops)
        // 5. Transfer final keys back to CPU
    }
}
```

**Key operations on GPU**:
- Polynomial multiplication (already have Metal NTT)
- Sampling (add GPU RNG kernel)
- Evaluation key generation (parallel over digits)

#### Step 3: Integration (1 hour)

**Update bootstrap context** to use Metal keygen:
```rust
// In src/clifford_fhe_v3/bootstrapping/bootstrap_context.rs

#[cfg(feature = "v2-gpu-metal")]
let key_ctx = crate::clifford_fhe_v2::backends::gpu_metal::keys::MetalKeyContext::new(params);

#[cfg(not(feature = "v2-gpu-metal"))]
let key_ctx = crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext::new(params);
```

### Testing

```bash
# Metal GPU key generation
cargo run --release --features v2,v3,v2-gpu-metal --example test_v3_full_bootstrap

# Should see:
# Creating Metal NTT contexts on GPU...
# ✓ NTT contexts created in 0.5s (16 primes in parallel)
# Key generation completed in 15s
```

---

## Phase 3: CUDA GPU Key Generation (Priority 2 - 4-6 hours)

### Goal
Same as Metal, but for CUDA/Runpod.io deployment.

### Implementation

**File**: `src/clifford_fhe_v2/backends/gpu_cuda/keys.rs` (new)

Similar structure to Metal, but using:
- CUDA kernels instead of Metal shaders
- `cudarc` crate for device management
- CUDA streams for parallelization

**Advantage**: CUDA typically 2-3× faster than Metal for compute-heavy workloads.

**Testing on Runpod.io**:
```bash
# On Runpod GPU instance
cargo run --release --features v2,v3,v2-gpu-cuda --example test_v3_full_bootstrap

# Expected: Key generation in 5-15 seconds
```

---

## Phase 4: Documentation & Examples (1 hour)

### Create comparison table

```markdown
| Backend | N | Primes | Key Gen Time | Use Case |
|---------|---|--------|--------------|----------|
| CPU     | 1024 | 10 | <5s | Demos, tests |
| CPU     | 2048 | 12 | ~30s | Small production |
| Metal   | 8192 | 16 | ~15s | Mac development |
| Metal   | 8192 | 20 | ~25s | Mac production |
| CUDA    | 8192 | 16 | ~8s | Cloud production |
| CUDA    | 8192 | 20 | ~12s | Cloud production |
| CUDA    | 16384 | 25 | ~40s | Maximum security |
```

### Update all docs

- `README.md` - Add GPU requirements
- `V3_BOOTSTRAP.md` - Document parameter choices
- `V3_EXAMPLES_COMMANDS.md` - Add GPU examples
- New: `GPU_SETUP.md` - Metal/CUDA setup instructions

---

## Implementation Priority Order

### Week 1 (This session if possible, otherwise next session)

1. ✅ **CPU small params** (30 min) - Get something working NOW
2. 🔄 **Metal NTT context** (2 hours) - Core GPU infrastructure
3. 🔄 **Metal keygen** (2-3 hours) - Full pipeline
4. ✅ **Test Metal locally** (30 min) - Validate on your Mac

### Week 2 (After Metal works)

5. 🔄 **CUDA implementation** (4-6 hours) - Port Metal → CUDA
6. ✅ **Test on Runpod** (1 hour) - Validate cloud deployment
7. ✅ **Documentation** (1 hour) - Update all docs

---

## Files to Create/Modify

### New Files

```
src/clifford_fhe_v2/backends/gpu_metal/keys.rs        # Metal key generation
src/clifford_fhe_v2/backends/gpu_cuda/keys.rs         # CUDA key generation
examples/test_v3_cpu_demo.rs                          # Small CPU demo
examples/test_v3_metal_gpu.rs                         # Metal GPU demo
examples/test_v3_cuda_gpu.rs                          # CUDA GPU demo
GPU_SETUP.md                                          # Setup instructions
```

### Modified Files

```
src/clifford_fhe_v3/params.rs                         # Add new_v3_demo_cpu()
src/clifford_fhe_v3/bootstrapping/bootstrap_context.rs # GPU keygen integration
src/clifford_fhe_v2/backends/gpu_metal/ntt.rs         # Add GPU NttContext
src/clifford_fhe_v2/backends/gpu_metal/shaders/ntt.metal # Twiddle generation kernel
README.md                                              # GPU requirements
V3_BOOTSTRAP.md                                       # Parameter recommendations
```

---

## Technical Details

### Metal Shader Addition

**In `ntt.metal`**, add after existing kernels:

```metal
/// Generate twiddle factors for NTT
/// Each thread computes one twiddle factor: ω^i mod q
kernel void generate_twiddle_factors(
    device ulong* twiddles [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    constant ulong& q [[buffer(2)]],
    constant ulong& omega [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        twiddles[gid] = pow_mod(omega, gid, q);
    }
}

/// Generate inverse twiddle factors
kernel void generate_inv_twiddle_factors(
    device ulong* inv_twiddles [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    constant ulong& q [[buffer(2)]],
    constant ulong& omega_inv [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        inv_twiddles[gid] = pow_mod(omega_inv, gid, q);
    }
}
```

### Finding Primitive Roots

**Helper function** (CPU, runs once per modulus):

```rust
fn find_primitive_root(n: usize, q: u64) -> u64 {
    // For NTT-friendly prime q ≡ 1 mod 2N
    // Find ω such that ω^N ≡ -1 mod q

    // Start with generator g of Z_q^*
    let g = find_generator(q);

    // ω = g^((q-1)/(2N)) is primitive 2N-th root of unity
    let exp = (q - 1) / (2 * n as u64);
    pow_mod(g, exp, q)
}
```

---

## Success Criteria

### Phase 1 (CPU Demo)
- [ ] N=1024 or N=2048 parameters defined
- [ ] Full bootstrap completes in <30 seconds on CPU
- [ ] All tests pass

### Phase 2 (Metal GPU)
- [ ] NTT context creation on GPU completes in <1 second for 16 primes
- [ ] Full key generation (N=8192, 16 primes) in <30 seconds
- [ ] Bootstrap pipeline works end-to-end on Metal

### Phase 3 (CUDA GPU)
- [ ] Same metrics as Metal, but on Runpod.io
- [ ] Key generation <15 seconds (faster than Metal)
- [ ] Full bootstrap works on CUDA

---

## Next Immediate Steps

**Right now** (you decide the order):

**Option A: Quick Win - CPU Demo** (30 min)
1. Create `new_v3_demo_cpu()` with N=1024
2. Test it works
3. Commit and move to GPU

**Option B: Go Big - Metal GPU** (dive in for 4-6 hours)
1. Implement Metal NTT context GPU initialization
2. Implement Metal key generation
3. Test on your Mac

**Recommendation**: Start with **Option A** to get something working immediately, then tackle Option B.

What would you like to do first?
