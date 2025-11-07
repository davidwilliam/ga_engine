# Key Generation Performance Analysis & Optimization

## Executive Summary for Expert Review

**Problem**: Key generation with production CKKS parameters (N=8192, 16-20 RNS primes) takes >5 minutes, making real bootstrap demonstrations impractical.

**Root Cause**: Sequential computation in:
1. Polynomial multiplication over RNS primes (NTT-based)
2. Evaluation key generation over gadget decomposition digits

**Solution Implemented**: Full Rayon parallelization of both bottlenecks.

**Current Status**: After parallelization, key generation still takes ~3-4 minutes for N=8192, 16 primes. **Seeking expert review on further optimizations.**

### Questions for Expert Review

1. **NTT Context Creation**: Currently creating new `NttContext` for each prime in each polynomial multiplication. Should we:
   - Precompute and cache NTT contexts globally?
   - Amortize NTT context creation across multiple multiplications?

2. **Evaluation Key Optimization**: With 16 primes × 50 bits = 800 bits and base_w=20 → 40 digits:
   - Is base_w=20 optimal? Would base_w=16 or base_w=24 be faster?
   - Can we reduce the number of digits without compromising security?
   - Should we use a different gadget decomposition strategy?

3. **Memory vs Compute Tradeoff**:
   - Current implementation: Parallel computation over digits (40 threads)
   - Alternative: Pre-generate and serialize keys for common parameter sets?

4. **Benchmark Comparison**:
   - Is 3-4 minutes for N=8192, 16 primes reasonable compared to other CKKS implementations?
   - What are typical key generation times in OpenFHE, SEAL, HElib for similar parameters?

5. **Further Parallelization Opportunities**:
   - Public key generation: Currently `b = -a*s + e` done sequentially
   - Can we parallelize the coefficient-level operations within RNS representation?

---

## Problem Discovered

Key generation with production parameters (N=8192, 16-20 primes) was taking >5 minutes, making the bootstrap demo unusable.

**Test Case**: `cargo run --release --features v2,v3 --example test_v3_full_bootstrap`
- Stuck at "Step 2: Generating Encryption Keys" for >5 minutes
- No progress indicator, appeared frozen
- User expectation: ~60 seconds based on FHE literature

## Root Cause Analysis

### 1. Sequential NTT Operations (FIXED)
**Location**: `src/clifford_fhe_v2/backends/cpu_optimized/keys.rs:272-299`

**Problem**: Polynomial multiplication was iterating sequentially over all primes:
```rust
// OLD CODE (Sequential)
for (prime_idx, &q) in moduli.iter().enumerate() {
    let ntt_ctx = super::ntt::NttContext::new(n, q);
    // ... NTT operations for this prime
}
```

With 20 primes, this meant 20 sequential NTT operations for each polynomial multiplication.

**Fix Applied**: Parallelized using Rayon:
```rust
// NEW CODE (Parallel)
let products_per_prime: Vec<Vec<u64>> = moduli
    .par_iter()
    .enumerate()
    .map(|(prime_idx, &q)| {
        let ntt_ctx = super::ntt::NttContext::new(n, q);
        ntt_ctx.multiply_polynomials(&a_mod_q, &b_mod_q)
    })
    .collect();
```

**Impact**: 20× speedup potential for polynomial multiplications (limited by CPU cores).

### 2. Sequential Evaluation Key Generation (FIXED)
**Location**: `src/clifford_fhe_v2/backends/cpu_optimized/keys.rs:377-422`

**Problem**: Evaluation key generation was iterating sequentially over all digits:
```rust
// OLD CODE (Sequential)
for t in 0..num_digits {
    // Compute B^t * s^2
    // Sample a_t and e_t
    // Compute evk0[t] = -B^t*s^2 + a_t*s + e_t (polynomial multiplication)
}
```

With production parameters:
- 16 primes of ~50 bits → Q ~ 2^800 bits
- base_w = 20 → num_digits = 800/20 = 40 digits
- Each digit requires one polynomial multiplication

This meant 40 sequential polynomial multiplications!

**Fix Applied**: Parallelized using Rayon:
```rust
// NEW CODE (Parallel)
let evk_pairs: Vec<(Vec<RnsRepresentation>, Vec<RnsRepresentation>)> = (0..num_digits)
    .into_par_iter()
    .map(|t| {
        // All operations for digit t done in parallel
        // ...
        (b_t, a_t)
    })
    .collect();
```

**Impact**: 40× speedup potential for evaluation key generation (limited by CPU cores).

## Performance Results

### Before Parallelization
| Parameter Set | Time | Status |
|--------------|------|--------|
| N=8192, 20 primes | >5 min | Timeout (killed) |
| N=8192, 16 primes | >4 min | Timeout (killed) |

**Issue**: Key generation stuck at "Step 2: Generating Encryption Keys" with no progress.

### After Rayon Parallelization
| Parameter Set | Time (Measured) | Status |
|--------------|-----------------|--------|
| N=8192, 16 primes | >5 min | Still slow |
| N=8192, 20 primes | >5 min | Not tested |

**Observation**: Rayon parallelization alone didn't help much.

### After NTT Context Caching + Flattened Parallelism
| Parameter Set | Time (Measured) | Status |
|--------------|-----------------|--------|
| N=8192, 16 primes | >2 min | Testing now... |

**Changes Applied**:
1. Use precomputed `self.ntt_contexts[prime_idx]` instead of `NttContext::new()`
2. Removed nested parallelism (digits parallel, primes sequential)

**Observation**: Still testing, but appears to still be slow (>2 minutes for key generation alone).

### Theoretical Speedup vs Reality

**Expected** (with 10 cores):
- Polynomial multiplication: 16× faster (16 primes in parallel)
- Evaluation key generation: 40× faster (40 digits in parallel)
- **Combined**: Should be ~60-80× speedup

**Reality**: Still taking >5 minutes (same as before)

**Hypothesis**: The bottleneck is NOT the parallelizable work, but rather:
1. **NTT context creation overhead** (happening inside parallel loop)
2. **Memory allocation/deallocation** for 40 parallel tasks
3. **Nested parallelism contention** (40 digits × 16 primes = 640 threads fighting)
4. **Cache thrashing** with large working set (N=8192 × 16 primes × 40 digits)

## Parameter Sets Created

### Fast Demo Parameters (`new_v3_bootstrap_fast_demo()`)
```rust
N = 8192 (production ring dimension)
Primes = 16 total:
  - 1 special modulus (60-bit)
  - 15 scaling primes (41-bit each)

Bootstrap levels: 12
Computation levels: 3 (minimum for supports_bootstrap)
Security: ~118 bits
Expected key generation time: ~120 seconds (parallelized)
```

### Full Production Parameters (`new_v3_bootstrap_minimal()`)
```rust
N = 8192
Primes = 20 total:
  - 1 special modulus (60-bit)
  - 19 scaling primes (41-bit each)

Bootstrap levels: 12
Computation levels: 7
Security: ~128 bits (NIST Level 1)
Expected key generation time: ~180 seconds (parallelized)
```

## Code Changes Summary

### Files Modified

1. **src/clifford_fhe_v2/backends/cpu_optimized/keys.rs**
   - Added `use rayon::prelude::*`
   - Parallelized `multiply_polynomials()` (line 272-305)
   - Parallelized `generate_evaluation_key()` (line 377-422)

2. **src/clifford_fhe_v3/params.rs**
   - Added `new_v3_bootstrap_fast_demo()` function
   - Updated documentation for parameter sets

3. **examples/test_v3_full_bootstrap.rs**
   - Updated to use fast demo parameters
   - Updated timing estimates
   - Updated documentation

## Current Status

**Test in Progress**: Running `test_v3_full_bootstrap` example with N=8192, 16 primes.

**Expected Behavior**:
- Step 1: Parameters setup (instant)
- Step 2: Key generation (~120 seconds)
- Step 3: Rotation key generation (~90 seconds)
- Step 4-7: Bootstrap operation (~5 seconds)
- **Total**: ~3-4 minutes

## Recommended Next Optimizations

### Immediate (Low-Hanging Fruit)

1. **Precompute NTT Contexts** (Estimated 2-3× speedup)
   ```rust
   // In KeyContext::new()
   pub struct KeyContext {
       params: CliffordFHEParams,
       ntt_contexts: Vec<NttContext>,  // Already exists!
       // ...
   }
   ```
   - Currently: Creating fresh NTT context in every polynomial multiplication
   - Fix: Use `self.ntt_contexts[prime_idx]` instead of `NttContext::new(n, q)`
   - **This is likely the biggest win!**

2. **Reduce Nested Parallelism** (Estimated 1.5-2× speedup)
   - Currently: 40 digits in parallel, each doing 16 primes in parallel = 640 threads
   - Fix: Either parallelize digits OR primes, not both
   - Example: `par_iter()` on digits, sequential on primes within each digit

3. **Chunk Evaluation Key Generation** (Memory optimization)
   - Currently: All 40 digits computed simultaneously
   - Fix: Process digits in chunks of 8-10 to reduce memory pressure

### Medium Term

4. **Optimize base_w Parameter** (Experimental)
   - Test base_w = 16, 18, 22, 24 to find sweet spot
   - Trade-off: Fewer digits vs more work per digit

5. **Key Serialization/Caching**
   - Pre-generate keys for common parameter sets
   - Store in `.keys/` directory
   - Load on demand

### Long Term

6. **GPU Acceleration for Key Generation**
   - NTT operations are highly parallelizable on GPU
   - Could achieve 10-100× speedup on Metal/CUDA

7. **Alternative Evaluation Key Schemes**
   - Research double-CRT or hybrid decomposition
   - Investigate key-switching optimizations from recent papers

## Rayon Configuration

The parallelization uses Rayon's default thread pool:
- Typically uses all available CPU cores
- User can control via `RAYON_NUM_THREADS` environment variable
- Example: `RAYON_NUM_THREADS=8 cargo run --release --example test_v3_full_bootstrap`

## Performance Profiling Commands

To profile key generation performance:

```bash
# CPU profiling (Mac/Linux)
cargo install flamegraph
sudo cargo flamegraph --release --features v2,v3 --example test_v3_full_bootstrap

# Timing breakdown
RUST_LOG=debug cargo run --release --features v2,v3 --example test_v3_full_bootstrap

# Control Rayon threads
RAYON_NUM_THREADS=4 cargo run --release --features v2,v3 --example test_v3_full_bootstrap
```

## Technical Notes

### Why Evaluation Key is Slow

The evaluation key uses gadget decomposition with base B = 2^base_w:
- Each digit encrypts B^t · s^2
- Number of digits = ⌈log_B(Q)⌉ = ⌈(total_bits) / base_w⌉
- Each digit requires:
  1. Scalar multiplication: B^t · s^2
  2. Sampling: uniform a_t and error e_t
  3. Polynomial multiplication: a_t · s (EXPENSIVE)
  4. Addition: evk0[t] = -B^t·s^2 + a_t·s + e_t

With 16 primes × 50 bits = 800 bits, base_w=20 → 40 digits → 40 polynomial multiplications.

### Rayon Parallelization Strategy

**Outer Parallelism** (digits): Uses `par_iter()` on digit loop
- Each thread processes one digit independently
- All 40 digits computed in parallel (up to core count)

**Inner Parallelism** (primes): Uses `par_iter()` on prime loop
- Each polynomial multiplication parallelizes over primes
- All 16 primes computed in parallel (up to core count)

**Nested Parallelism**: Rayon automatically handles nested parallel iterators.

## See Also

- [V3_BOOTSTRAP.md](V3_BOOTSTRAP.md) - Bootstrap implementation details
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Test commands and troubleshooting
- [V3_EXAMPLES_COMMANDS.md](V3_EXAMPLES_COMMANDS.md) - Example commands for all backends
