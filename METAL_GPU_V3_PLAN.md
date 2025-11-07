# Metal GPU V3 Bootstrap Implementation Plan

## Goal
Implement **real, production-ready** V3 bootstrap on Metal GPU with N=8192 parameters.

## Current Status

### ✅ What We Have
- ✅ Metal GPU device management ([device.rs](src/clifford_fhe_v2/backends/gpu_metal/device.rs))
- ✅ Metal NTT acceleration ([ntt.rs](src/clifford_fhe_v2/backends/gpu_metal/ntt.rs))
- ✅ Metal shaders for NTT and RNS ([shaders/](src/clifford_fhe_v2/backends/gpu_metal/shaders/))
- ✅ V3 bootstrap CPU implementation (complete pipeline)
- ✅ V3 parameters for N=8192 with 20+ primes

### ❌ What We Need
- ❌ GPU-accelerated key generation (currently CPU-only, takes forever)
- ❌ GPU-accelerated rotation key generation
- ❌ GPU bootstrap operations integration
- ❌ V3 Metal demo example

## Implementation Steps

### Phase 1: GPU Key Generation (2 hours)
**Problem**: CPU key generation for N=8192 with 20 primes takes 5-10 minutes
**Solution**: Move NTT operations to GPU

#### Tasks:
1. **Create MetalKeyContext** (new file: `gpu_metal/keys.rs`)
   - Wrap CPU KeyContext but use GPU for NTT operations
   - Generate keys on GPU: forward/inverse NTT on GPU
   - Target: <10 seconds for N=8192, 20 primes

2. **GPU NTT Context Creation**
   - Parallelize twiddle factor computation on GPU
   - Currently done on CPU (slow!)
   - Store twiddle factors in GPU memory

3. **Test GPU key generation**
   - Verify correctness: encrypt/decrypt roundtrip
   - Benchmark: should be 30-100× faster than CPU

### Phase 2: GPU Rotation Keys (1 hour)
**Problem**: Rotation keys require many NTT operations
**Solution**: Use GPU NTT for all rotation key generation

#### Tasks:
1. **Extend MetalKeyContext**
   - Add `generate_rotation_keys_gpu()` method
   - Use GPU NTT for all operations
   - Target: <20 seconds for 16 rotation keys

2. **Test rotation keys**
   - Verify rotation correctness
   - Benchmark vs CPU

### Phase 3: GPU Bootstrap Operations (2 hours)
**Problem**: Bootstrap operations (CoeffToSlot, EvalMod, SlotToCoeff) need GPU
**Solution**: Integrate V3 bootstrap with Metal GPU backend

#### Tasks:
1. **Create MetalBootstrapContext** (new file: `clifford_fhe_v3/metal/bootstrap.rs`)
   - Wraps `BootstrapContext` but uses GPU for all NTT operations
   - Same API, different backend

2. **GPU-accelerated operations**:
   - CoeffToSlot: All rotations on GPU
   - EvalMod: Polynomial multiplication on GPU
   - SlotToCoeff: All rotations on GPU

3. **Test bootstrap end-to-end**
   - Verify correctness with known plaintext
   - Benchmark: target <100ms per bootstrap (vs seconds on CPU)

### Phase 4: V3 Metal Demo (1 hour)
**Problem**: Need a demo to test everything
**Solution**: Create comprehensive demo

#### Tasks:
1. **Create `examples/test_v3_metal_demo.rs`**
   - Similar to CPU demo but uses Metal GPU
   - N=8192, 20 primes (production parameters!)
   - Full bootstrap pipeline

2. **Test and verify**
   - Run on your Mac
   - Verify error is acceptable
   - Benchmark all operations

## Timeline

| Phase | Time | Cumulative |
|-------|------|------------|
| Phase 1: GPU Key Generation | 2h | 2h |
| Phase 2: GPU Rotation Keys | 1h | 3h |
| Phase 3: GPU Bootstrap Ops | 2h | 5h |
| Phase 4: V3 Metal Demo | 1h | 6h |
| **Total** | **6 hours** | - |

## Expected Performance (N=8192, 20 primes)

| Operation | CPU (estimated) | Metal GPU (target) | Speedup |
|-----------|-----------------|-------------------|---------|
| Key Generation | 5-10 minutes | <10 seconds | 30-60× |
| Rotation Keys (16) | 2-5 minutes | <20 seconds | 6-15× |
| CoeffToSlot | ~200ms | <20ms | 10× |
| EvalMod | ~500ms | <50ms | 10× |
| SlotToCoeff | ~200ms | <20ms | 10× |
| **Full Bootstrap** | **~1 second** | **<100ms** | **10×** |

## File Structure

```
src/
├── clifford_fhe_v2/
│   └── backends/
│       └── gpu_metal/
│           ├── device.rs         ✅ (exists)
│           ├── ntt.rs             ✅ (exists)
│           ├── keys.rs            ❌ (need to create)
│           └── shaders/
│               ├── ntt.metal      ✅ (exists)
│               └── rns.metal      ✅ (exists)
└── clifford_fhe_v3/
    ├── metal/                     ❌ (new directory)
    │   ├── mod.rs                 ❌ (new)
    │   └── bootstrap.rs           ❌ (new)
    └── bootstrapping/
        └── ... (existing CPU code)

examples/
└── test_v3_metal_demo.rs          ❌ (need to create)
```

## Success Criteria

✅ **Correctness**:
- Key generation produces valid keys (encrypt/decrypt works)
- Rotation keys work correctly
- Bootstrap preserves plaintext (error < 1%)

✅ **Performance**:
- Key generation: <10 seconds (vs 5-10 minutes CPU)
- Full bootstrap: <100ms (vs ~1 second CPU)
- Production parameters: N=8192, 20 primes

✅ **Real Implementation**:
- No mocking
- No shortcuts
- Production-ready code

---

## Next Step

**Start with Phase 1**: Create `src/clifford_fhe_v2/backends/gpu_metal/keys.rs`

This will implement GPU-accelerated key generation, which is the biggest bottleneck.

**Ready to begin?**
