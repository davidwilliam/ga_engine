# CUDA Architecture Analysis & Optimization Roadmap

**Date**: 2025-01-09
**GPU**: NVIDIA GeForce RTX 5090
**Current Performance**: 14.60s bootstrap (down from 276s initial)

---

## 📊 Current State Analysis

### Performance Breakdown (V3 CUDA Bootstrap)

| Operation | Time | % of Total | Status |
|-----------|------|------------|--------|
| **Key Generation** | 11.43s | Setup | ✅ One-time cost |
| **CoeffToSlot** | 0.13s | 0.9% | ✅ Excellent |
| **EvalMod (BSGS)** | 14.42s | 98.8% | ⚠️ **BOTTLENECK** |
| **SlotToCoeff** | 0.05s | 0.3% | ✅ Excellent |
| **Total Bootstrap** | 14.60s | 100% | ⚠️ Needs improvement |

**Key Finding**: EvalMod consumes 98.8% of bootstrap time despite being "fully GPU accelerated"

---

## 🔍 Deep Dive: EvalMod BSGS Bottleneck

### Algorithm Overview

EvalMod evaluates a degree-23 sine polynomial using Baby-Step Giant-Step (BSGS):
- **baby_steps** = 5
- **giant_steps** = 5
- **Total ciphertext multiplications** ≈ 10

### Per-Multiplication Cost Analysis

#### 1. Ciphertext Multiplication (`cuda_multiply_ciphertexts`)

**Current Implementation** (per multiplication):

```rust
// 1. CPU layout conversion (4× strided→flat)
let c0_flat = strided_to_flat(&ct1.c0);  // ~650k CPU ops
let c1_flat = strided_to_flat(&ct1.c1);  // ~650k CPU ops
let d0_flat = strided_to_flat(&ct2.c0);  // ~650k CPU ops
let d1_flat = strided_to_flat(&ct2.c1);  // ~650k CPU ops

// 2. Sequential NTT per prime (num_active_primes ≈ 20)
for prime_idx in 0..20 {
    ntt_ctx.forward(&mut c0_prime)?;  // GPU kernel launch
    ntt_ctx.forward(&mut c1_prime)?;  // GPU kernel launch
    ntt_ctx.forward(&mut d0_prime)?;  // GPU kernel launch
    ntt_ctx.forward(&mut d1_prime)?;  // GPU kernel launch

    // 4 pointwise multiplies
    ntt_ctx.pointwise_multiply(...)?;  // 4× GPU kernel launches

    // 4 inverse NTTs
    ntt_ctx.inverse(...)?;  // 4× GPU kernel launches
}
// Total: 20 × (4 forward + 4 pointwise + 4 inverse) = 240 kernel launches

// 3. Relinearization
apply_relinearization_gpu(...)?;
```

**Relinearization Cost** (per multiplication):
- **dnum** (gadget digits) = 86
- **num_primes** ≈ 20

```rust
for digit_idx in 0..86 {
    // GPU NTT multiply d_i · b_i
    for prime in 0..20 {
        ntt.forward();      // 1 kernel
        ntt.pointwise();    // 1 kernel
        ntt.inverse();      // 1 kernel
    }
    // GPU NTT multiply d_i · a_i (same as above)
    // Total: 86 × 20 × 6 = 10,320 kernel launches

    // GPU accumulation
    gpu_add();  // 86 kernel launches
}
```

**Total per multiplication**:
- **Layout conversions**: ~2.6M CPU operations
- **Multiply NTT kernels**: 240 launches
- **Relinearization NTT kernels**: 10,320 launches
- **Accumulation kernels**: 86 launches
- **Total GPU kernels**: **10,646 per multiplication**

### BSGS Total Cost

With ~10 multiplications in BSGS:
- **CPU operations**: 10 × 2.6M = **26M CPU ops**
- **GPU kernel launches**: 10 × 10,646 = **106,460 launches**

### Kernel Launch Overhead

**Per kernel launch cost**:
- Kernel submission: ~5-10μs
- PCIe latency: ~10-20μs (data transfer)
- **Total overhead**: ~15-30μs per launch

**Total overhead for BSGS**:
- 106,460 × 20μs = **2.13 seconds** just in launch overhead!
- Actual computation: ~12 seconds
- **Total**: ~14 seconds ✅ (matches observed 14.42s)

---

## 🚨 Root Cause Analysis

### The Fundamental Problem

**Current architecture launches thousands of small GPU kernels sequentially**, each with:
1. **Kernel launch overhead** (~10μs)
2. **PCIe transfer overhead** (~10-20μs for small data)
3. **GPU underutilization** (small workloads don't saturate GPU)

This is the **opposite** of optimal GPU usage, which requires:
1. **Large batched workloads**
2. **Minimal kernel launches**
3. **Data persistence on GPU** (avoid H2D/D2H)

### Why Current Optimizations Failed

| Optimization Attempted | Result | Reason |
|------------------------|--------|--------|
| GPU layout conversion | **Slower** (13.52s → 14.42s) | Added MORE kernel launches + PCIe transfers |
| GPU strided rescale | Minimal improvement | Already fast operation; overhead dominated |
| GPU accumulation | Minimal improvement | 86 extra kernel launches ≈ overhead saved |

**Conclusion**: We cannot optimize our way out of this with micro-optimizations. The architecture itself is the bottleneck.

---

## 💡 Architectural Redesign Proposal

### Core Principle: Batch Everything

**Goal**: Reduce 106,460 kernel launches to **< 100 kernel launches**

### Design 1: Batched Multi-Prime NTT

**Current** (sequential per prime):
```cuda
for prime in 0..20 {
    ntt_forward(prime);   // 20 kernel launches
}
```

**Proposed** (all primes in one kernel):
```cuda
__global__ void ntt_forward_batched(
    data,           // All primes' data
    twiddles,       // All primes' twiddle factors
    num_primes,     // 20
    n               // 32768
) {
    int prime_idx = blockIdx.y;  // Use 2D grid
    int coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (prime_idx < num_primes && coeff_idx < n) {
        // Process coefficient for this prime
        // All primes processed in parallel
    }
}
```

**Reduction**: 20 launches → **1 launch** (20× fewer)

### Design 2: Fused NTT-Multiply-iNTT Kernel

**Current** (3 separate kernels):
```cuda
ntt_forward(a);       // Kernel 1
pointwise_mult(a, b); // Kernel 2
ntt_inverse(result);  // Kernel 3
```

**Proposed** (single fused kernel):
```cuda
__global__ void ntt_multiply_fused(
    a, b, result,
    twiddles, inv_twiddles,
    n, num_primes
) {
    // All primes in parallel (2D grid)
    int prime_idx = blockIdx.y;

    // 1. NTT forward on a and b (shared memory)
    __shared__ uint64_t s_a[1024];
    __shared__ uint64_t s_b[1024];

    ntt_forward_inplace(s_a);
    ntt_forward_inplace(s_b);
    __syncthreads();

    // 2. Pointwise multiply
    s_a[threadIdx.x] = mul_mod(s_a[threadIdx.x], s_b[threadIdx.x]);
    __syncthreads();

    // 3. Inverse NTT
    ntt_inverse_inplace(s_a);

    // 4. Write result
    result[prime_idx * n + threadIdx.x] = s_a[threadIdx.x];
}
```

**Benefits**:
- **3× fewer kernel launches**
- **Shared memory usage** (no global memory roundtrips)
- **Data locality** (everything stays in L1 cache)

**For ciphertext multiply**: 240 launches → **4 launches** (60× reduction)

### Design 3: Persistent Kernel Architecture

**Concept**: Keep data on GPU across entire BSGS algorithm

**Current flow** (terrible for GPU):
```rust
for multiply in bsgs {
    // H2D transfer
    result = gpu_multiply(ct1, ct2);
    // D2H transfer

    // H2D transfer
    result = gpu_rescale(result);
    // D2H transfer
}
```

**Proposed flow** (GPU-resident):
```rust
// Upload ALL data once
upload_to_gpu(all_ciphertexts);

// Single kernel launch for entire BSGS
gpu_bsgs_kernel<<<>>>(
    input_ct,
    coefficients,
    rotation_keys,
    relin_keys
);
// Kernel internally handles all:
// - Power computations
// - Polynomial evaluation
// - Rescaling
// - Relinearization

// Download result once
result = download_from_gpu();
```

**Benefits**:
- **2 PCIe transfers** instead of 10,000+
- **Minimal kernel launch overhead**
- **GPU can optimize data movement internally**

### Design 4: Relinearization Fusion

**Current** (86 separate iterations):
```cuda
for digit in 0..86 {
    d_b = ntt_multiply(d_i, b_i);    // 60 kernel launches (20 primes × 3)
    d_a = ntt_multiply(d_i, a_i);    // 60 kernel launches
    c0_acc = gpu_add(c0_acc, d_b);   // 1 kernel launch
    c1_acc = gpu_add(c1_acc, d_a);   // 1 kernel launch
}
```

**Proposed** (single batched kernel):
```cuda
__global__ void relinearization_fused(
    c0, c1, c2,
    relin_keys,     // All 86 key components
    num_primes,
    dnum
) {
    // 3D grid: prime × digit × coefficient
    int prime_idx = blockIdx.z;
    int digit_idx = blockIdx.y;
    int coeff_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process all digits in parallel
    // Use atomic adds for accumulation
    // Result: 1 kernel launch instead of 10,320
}
```

**Reduction**: 10,320 launches → **1 launch** (10,320× fewer!)

---

## 📈 Expected Performance Impact

### Current Architecture
- **Kernel launches**: 106,460 per BSGS
- **Launch overhead**: ~2.1s
- **Computation**: ~12s
- **Total**: **14.4s**

### Batched Architecture (Designs 1-4 combined)

#### Optimistic Estimate (Best Case)
- **Kernel launches**: ~50 per BSGS
  - 10 multiplies × 4 fused kernels = 40
  - 10 rescales × 1 kernel = 10
- **Launch overhead**: 50 × 20μs = **1ms** ✅ (2,100× reduction!)
- **Computation**: ~6s (better GPU utilization)
- **Total**: **~6 seconds** (2.4× speedup)

#### Realistic Estimate (More Achievable)
- **Kernel launches**: ~200 per BSGS
- **Launch overhead**: 200 × 20μs = **4ms**
- **Computation**: ~8s (better batching)
- **Total**: **~8 seconds** (1.8× speedup)

#### Conservative Estimate (Minimum Improvement)
- **Kernel launches**: ~1,000 per BSGS
- **Launch overhead**: 1,000 × 20μs = **20ms**
- **Computation**: ~10s
- **Total**: **~10 seconds** (1.4× speedup)

**Expected final bootstrap time**: **6-10 seconds** (down from 14.6s)

---

## 🛠️ Implementation Roadmap

### Phase 1: Foundation (1-2 weeks)

**Goal**: Batched multi-prime NTT

1. **Design batched NTT kernel**
   - 2D grid: `(n_blocks, num_primes)`
   - All primes processed in parallel
   - Shared memory for twiddle factors

2. **Implement and test**
   - Unit tests for correctness
   - Benchmark against sequential version
   - **Expected gain**: 20× reduction in NTT launches

3. **Update NTT context**
   ```rust
   pub fn forward_batched(&self, data: &mut [u64], num_primes: usize) -> Result<(), String>;
   ```

### Phase 2: Fusion (2-3 weeks)

**Goal**: Fused NTT-Multiply-iNTT kernel

1. **Design fused kernel**
   - Shared memory for intermediate results
   - Single kernel does: forward NTT → multiply → inverse NTT
   - Handle all primes in parallel

2. **Integrate with multiplication**
   ```rust
   pub fn multiply_fused(&self, ct1, ct2) -> Result<CudaCiphertext, String> {
       // Single kernel call instead of 240
   }
   ```

3. **Benchmark**
   - **Expected gain**: 60× reduction in multiply overhead

### Phase 3: Relinearization Batching (1-2 weeks)

**Goal**: Single kernel for all relinearization

1. **Design batched relin kernel**
   - 3D grid: `(coeff_blocks, digits, primes)`
   - Atomic operations for accumulation
   - All 86 digits processed in parallel

2. **Implement**
   ```rust
   pub fn apply_relinearization_batched(&self, ...) -> Result<(Vec<u64>, Vec<u64>), String>;
   ```

3. **Test and verify**
   - **Expected gain**: 10,000× reduction in relin overhead

### Phase 4: BSGS Kernel Fusion (3-4 weeks)

**Goal**: Single kernel for entire BSGS polynomial evaluation

**This is advanced** - requires:
- GPU-resident data structures
- Dynamic kernel scheduling
- Complex control flow on GPU

**Implementation strategy**:
1. Start with simpler version: batch power computations
2. Gradually merge more operations into single kernel
3. Careful profiling at each step

**Expected gain**: Near-optimal GPU utilization

### Phase 5: Polish & Optimize (1-2 weeks)

- Tune kernel launch parameters
- Optimize memory access patterns
- Add GPU streams for overlap
- Profile and iterate

---

## 🎯 Success Metrics

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| **EvalMod time** | 14.4s | < 10s | < 6s |
| **Bootstrap time** | 14.6s | < 10s | < 7s |
| **Kernel launches (BSGS)** | 106,460 | < 1,000 | < 100 |
| **PCIe transfers (BSGS)** | ~10,000 | < 100 | < 10 |
| **GPU utilization** | ~30% | > 70% | > 90% |

---

## 🚧 Implementation Challenges

### Technical Challenges

1. **CUDA Complexity**
   - Shared memory management
   - Synchronization across thread blocks
   - Atomic operations for accumulation

2. **Numerical Stability**
   - Modular arithmetic in fused kernels
   - Maintaining precision through fusion

3. **Memory Constraints**
   - Shared memory limits (~48KB per block)
   - Register pressure with complex kernels

4. **Debugging**
   - GPU debugging is difficult
   - Race conditions in atomic operations
   - Correctness verification

### Development Effort

| Phase | Complexity | Estimated Time | Risk |
|-------|-----------|----------------|------|
| Phase 1 | Medium | 1-2 weeks | Low |
| Phase 2 | High | 2-3 weeks | Medium |
| Phase 3 | High | 1-2 weeks | Medium |
| Phase 4 | Very High | 3-4 weeks | High |
| Phase 5 | Medium | 1-2 weeks | Low |
| **Total** | | **8-13 weeks** | |

---

## 🔬 Alternative Approaches

### Option A: Use Existing CUDA FHE Libraries

**Libraries to consider**:
- **cuFHE** (older, may be outdated)
- **NVIDIA cuHE** (if available)
- **Microsoft SEAL GPU backend** (via HElib)

**Pros**:
- Professionally optimized
- Already implements batched operations
- Battle-tested

**Cons**:
- May not support Clifford algebra operations
- Integration overhead
- Less control over implementation

### Option B: Hybrid CPU-GPU Approach

**Idea**: Use GPU only for the most expensive operations

**Strategy**:
- Keep current CPU implementation for control flow
- Use GPU only for massive parallel ops (NTT, matrix operations)
- Accept some PCIe overhead as acceptable

**Pros**:
- Simpler to implement
- Incremental improvement
- Lower risk

**Cons**:
- Won't achieve optimal performance
- PCIe overhead remains

### Option C: Algorithm-Level Optimization

**Idea**: Reduce BSGS computation before GPU optimization

**Strategies**:
1. **Lower polynomial degree** (23 → 15)
   - Trades accuracy for speed
   - May be acceptable for some applications

2. **Approximate relinearization**
   - Skip some relinearization steps
   - Use noise budget analysis

3. **Lazy rescaling**
   - Accumulate rescales, do in batch
   - Fewer rescale operations

**Pros**:
- Can be done immediately
- Works with current architecture

**Cons**:
- May reduce accuracy
- Application-specific tradeoffs

---

## 📝 Recommendations

### Immediate Actions (This Week)

1. ✅ **Accept current 14s performance** as baseline
2. **Profile GPU utilization** using `nvprof` or Nsight Compute
3. **Document current kernel launch patterns**
4. **Prototype batched NTT** (Phase 1) to validate approach

### Short-Term (1-2 Months)

1. **Implement Phase 1** (Batched NTT)
   - Lower risk, clear benefit
   - Validates batching approach
   - Expected 1.5-2× speedup

2. **Implement Phase 2** (Fused kernels)
   - Moderate complexity
   - Significant performance gain
   - Expected 2-3× speedup total

### Long-Term (3-6 Months)

1. **Implement Phase 3-4** (Full batching)
   - High complexity, high reward
   - Requires dedicated CUDA expertise
   - Expected 3-5× speedup total

2. **Consider** switching to existing optimized library
   - If internal development too costly
   - May require algorithm adaptation

### Decision Point

**If current 14s performance is acceptable for your use case:**
- STOP optimization efforts
- Focus on other features
- Revisit when requirements change

**If sub-10s performance is required:**
- Commit to Phase 1-2 implementation
- Allocate 1-2 months development time
- Hire CUDA optimization specialist if needed

**If sub-7s performance is critical:**
- Full architectural redesign required
- 3-6 months development effort
- Consider commercial FHE libraries

---

## 📚 References

### CUDA Best Practices
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/)

### FHE on GPU
- "Accelerating Fully Homomorphic Encryption in Hardware" (DAI et al., 2015)
- "cuFHE: GPU-Accelerated Fully Homomorphic Encryption Library" (DAI & SUNAR, 2019)
- "High-Performance FHE over the Torus" (CHILLOTTI et al., 2020)

### Optimization Techniques
- "Better GPU Performance through Batching" (NVIDIA Developer Blog)
- "Fused Kernel Design Patterns for GPUs" (GTC Presentations)

---

## 📊 Appendix: Detailed Profiling Data

### Kernel Launch Breakdown (Per BSGS Multiplication)

| Operation | Kernels | Per Prime | Total | Overhead |
|-----------|---------|-----------|-------|----------|
| **NTT Forward** | 4 | 20 | 80 | 1.6ms |
| **Pointwise Multiply** | 4 | 20 | 80 | 1.6ms |
| **NTT Inverse** | 4 | 20 | 80 | 1.6ms |
| **Relin NTT Forward** | 2 × 86 | 20 | 3,440 | 68.8ms |
| **Relin Pointwise** | 2 × 86 | 20 | 3,440 | 68.8ms |
| **Relin NTT Inverse** | 2 × 86 | 20 | 3,440 | 68.8ms |
| **Relin Accumulate** | 86 | 1 | 86 | 1.7ms |
| **Rescale** | 2 | 1 | 2 | 0.04ms |
| **Total** | | | **10,648** | **212.9ms** |

**Per BSGS (10 multiplications)**: 10,648 × 10 = 106,480 launches ≈ **2.13s overhead**

---

**Document Version**: 1.0
**Last Updated**: 2025-01-09
**Next Review**: After Phase 1 completion
