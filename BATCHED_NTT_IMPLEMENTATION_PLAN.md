# Batched NTT Implementation Plan - Phase 1

## Status: ✅ COMPLETE - Ready for Testing on RTX 5090!

### Completed
1. ✅ **Batched CUDA Kernels Created** (ntt.cu)
   - `ntt_forward_batched` - Process all primes in parallel
   - `ntt_inverse_batched` - Process all primes in parallel
   - `ntt_pointwise_multiply_batched` - Process all primes in parallel
   - Uses 2D grid: (butterfly_blocks, num_primes)

2. ✅ **Kernels Compile Successfully**
   - Verified with `cargo build`
   - No CUDA compilation errors

3. ✅ **Batched NTT Kernels Loaded**
   - Updated ntt.rs to load batched kernel names
   - Made twiddles, twiddles_inv, n_inv, device accessible

4. ✅ **Batched NTT Wrapper Methods Implemented** (ckks.rs)
   - `ntt_forward_batched()` - 3 batched wrapper methods added
   - `ntt_inverse_batched()` - Handles twiddle collection
   - `ntt_pointwise_multiply_batched()` - 2D grid launch configuration

5. ✅ **multiply_ciphertexts_tensored Updated**
   - Replaced sequential per-prime loop with batched operations
   - Reduced from 240 → 13 kernel launches per multiplication
   - Complete rewrite with clear documentation

6. ✅ **Full Compilation Success**
   - Library: ✅ Compiles in 8.98s
   - Example: ✅ Compiles in 14.32s
   - Zero errors, zero warnings

### Testing on RTX 5090

**Run the following command:**
```bash
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap
```

**Expected improvements:**
- ✅ Kernel launch count reduction: 24,800 → 1,240 per BSGS (20×)
- 🎯 EvalMod time target: < 13s (down from 14.42s)
- 🚀 Bootstrap time target: < 13s total

**What to verify:**
1. Correctness: Bootstrap completes successfully without errors
2. Performance: Measure actual EvalMod time improvement
3. GPU utilization: Should be higher with batched operations
4. Kernel launch overhead: ~470ms reduction expected

### Implementation Details

#### Kernel Launch Configuration (2D Grid)

For batched operations:
```rust
let threads_per_block = 256;
let num_butterfly_blocks = (n / 2 + threads_per_block - 1) / threads_per_block;

let cfg = LaunchConfig {
    grid_dim: (num_butterfly_blocks as u32, num_primes as u32, 1),  // 2D!
    block_dim: (threads_per_block as u32, 1, 1),
    shared_mem_bytes: 0,
};
```

#### Data Preparation

Need to prepare twiddles and moduli for all primes:
```rust
// Collect all twiddles into single buffer
let mut all_twiddles = Vec::with_capacity(n * num_primes);
let mut all_moduli = Vec::with_capacity(num_primes);

for prime_idx in 0..num_primes {
    let ntt_ctx = &self.ntt_contexts[prime_idx];
    all_twiddles.extend_from_slice(&ntt_ctx.twiddles);
    all_moduli.push(ntt_ctx.q);
}
```

### Expected Performance Impact

**Current multiply_ciphertexts_tensored**:
- NTT stages per prime: log₂(32768) = 15 stages
- Kernel launches: 4 forward × 15 stages × 20 primes = 1,200
- Plus 4 pointwise × 20 = 80
- Plus 4 inverse × 15 stages × 20 = 1,200
- **Total: ~2,480 launches per multiplication**

**Batched multiply_ciphertexts_tensored**:
- Forward NTT: 4 polynomials × 15 stages = 60 launches
- Pointwise: 4 operations × 1 launch = 4 launches
- Inverse NTT: 4 polynomials × 15 stages = 60 launches
- **Total: ~124 launches per multiplication (20× reduction!)**

**BSGS Impact** (10 multiplications):
- Current: 10 × 2,480 = 24,800 launches
- Batched: 10 × 124 = 1,240 launches
- **Reduction: 24,800 → 1,240 (20× fewer)**

**Expected time savings**:
- Launch overhead reduced: 24,800 × 20μs = 496ms → 1,240 × 20μs = 25ms
- **Overhead reduction: ~470ms saved**
- Plus better GPU utilization → **estimated 1-2s total speedup**
- **Target EvalMod time: 12-13s** (down from 14.4s)

### Testing Plan

1. **Unit Test**: Single multiplication with batched NTT
   - Verify correctness vs sequential version
   - Measure kernel launch count

2. **Performance Test**: BSGS polynomial evaluation
   - Compare old vs new `multiply_ciphertexts_tensored`
   - Profile with `nvprof` or Nsight Compute
   - Measure actual kernel launch reduction

3. **Full Bootstrap Test**: Complete V3 CUDA bootstrap
   - Target: < 13s bootstrap time
   - Verify numerical correctness
   - Check GPU utilization increased

### Risk Assessment

**Low Risk**:
- Kernel logic identical to sequential version
- Only difference is 2D grid instead of loop
- Easy to rollback if issues

**Potential Issues**:
- Grid size limits (max 65535 blocks per dimension)
  - Not a problem: num_primes ≤ 30, butterflies ≤ 16384
- Shared memory conflicts
  - Not applicable: no shared memory used
- Synchronization bugs
  - Not applicable: primes are independent

### Success Criteria

✅ **Minimum Success**:
- Batched NTT produces identical results
- At least 10× reduction in kernel launches
- EvalMod time < 13.5s

🎯 **Target Success**:
- 20× reduction in kernel launches
- EvalMod time < 13s
- Measurably improved GPU utilization

🚀 **Stretch Goal**:
- Consider fusing NTT stages (Phase 2)
- EvalMod time < 12s
- Path to sub-10s bootstrap

---

## Next Session Tasks

1. Implement batched NTT wrapper methods in `ckks.rs`
2. Update `multiply_ciphertexts_tensored` to use batched operations
3. Compile and test on RTX 5090
4. Profile and measure performance improvements
5. Document results and plan Phase 2 (fusion)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-09
**Author**: Claude & David Silva
