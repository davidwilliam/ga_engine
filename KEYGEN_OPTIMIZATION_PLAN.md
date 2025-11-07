# Key Generation Optimization Action Plan

## Current Status (2025-11-06)

### What We've Done ✅
1. **Rayon Parallelization** - Parallelized evaluation key generation over digits
2. **NTT Context Caching** - Fixed critical bug: now using `self.ntt_contexts[prime_idx]` instead of recreating
3. **Flattened Parallelism** - Removed nested par_iter (digits parallel, primes sequential)

### Current Performance 📊
- **N=8192, 16 primes**: Still >3 minutes for key generation alone
- **Rotation keys**: Already BSGS-optimized (26 keys for N=8192, not thousands)
- **Bottleneck**: Evaluation key generation (40 digits × polynomial multiplication)

---

## Expert Recommendations (Priority Order)

### 🔥 CRITICAL - Implement Next

#### 1. Buffer Reuse (Expected: 1.3-2× speedup)
**Problem**: Allocating fresh `Vec<u64>` in hot loops causes allocator overhead + cache thrashing

**Solution**: Thread-local buffers
```rust
use std::cell::RefCell;

thread_local! {
    static KEYGEN_BUFS: RefCell<KeygenBuffers> = RefCell::new(KeygenBuffers::new());
}

struct KeygenBuffers {
    // Preallocated scratch space
    tmp_poly_a: Vec<Vec<u64>>,  // [num_primes][n]
    tmp_poly_b: Vec<Vec<u64>>,  // [num_primes][n]
    tmp_ntt: Vec<Vec<u64>>,     // [num_primes][n]
}

impl KeygenBuffers {
    fn new() -> Self {
        // Allocate once per thread
        Self {
            tmp_poly_a: vec![vec![0u64; 8192]; 20],
            tmp_poly_b: vec![vec![0u64; 8192]; 20],
            tmp_ntt: vec![vec![0u64; 8192]; 20],
        }
    }

    fn resize_for(&mut self, n: usize, num_primes: usize) {
        // Only reallocate if params changed
        if self.tmp_poly_a.len() != num_primes || self.tmp_poly_a[0].len() != n {
            self.tmp_poly_a = vec![vec![0u64; n]; num_primes];
            self.tmp_poly_b = vec![vec![0u64; n]; num_primes];
            self.tmp_ntt = vec![vec![0u64; n]; num_primes];
        }
    }
}
```

**Where to apply**:
- `multiply_polynomials()` - reuse a_mod_q, b_mod_q buffers
- `generate_evaluation_key()` - reuse bt_s2, a_t, e_t buffers
- NTT operations - in-place transforms

**File**: `src/clifford_fhe_v2/backends/cpu_optimized/keys.rs`

---

#### 2. Verify No Hidden Parallelism
**Check**: Search entire codebase for stray `par_iter()` calls

```bash
rg "\.par_iter\(\)" src/
```

Expected: Only ONE location (evaluation key digit loop)

**Action**: Remove any others or gate them behind a feature flag

---

#### 3. Hoist & Cache NTT(secret)
**Problem**: Computing `NTT(s)` and `NTT(s(X^g))` repeatedly

**Solution**: Precompute once in `KeyContext::new()` or first keygen call

```rust
pub struct KeyContext {
    params: CliffordFHEParams,
    ntt_contexts: Vec<NttContext>,
    reducers: Vec<BarrettReducer>,

    // NEW: Cached transformed secrets
    secret_ntt_cache: Option<Vec<Vec<Vec<u64>>>>,  // [rotation][prime][n]
}
```

**Where it helps**: Rotation key generation (if we had more rotations)

**Priority**: Medium (we only have 26 rotations, already fast)

---

### ⚡ HIGH PRIORITY

#### 4. Fast Counter-Mode RNG
**Problem**: `thread_rng()` and Gaussian sampling can have long tails

**Solution**: Use ChaCha20 in counter mode with per-thread streams

```rust
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;

thread_local! {
    static THREAD_RNG: RefCell<ChaCha20Rng> = RefCell::new({
        let seed = /* thread-specific seed */;
        ChaCha20Rng::from_seed(seed)
    });
}
```

**File**: `src/clifford_fhe_v2/backends/cpu_optimized/keys.rs`
- Replace `sample_uniform()` and `sample_error()` implementations

---

#### 5. Tune base_w Parameter
**Current**: base_w = 20 → 40 digits for 16 primes

**Experiment**:
```rust
for base_w in [16, 18, 20, 22, 24] {
    let num_digits = (total_bits + base_w - 1) / base_w;
    println!("base_w={}: {} digits", base_w, num_digits);
    // Run keygen, measure time
}
```

**Expected results**:
- base_w=16: 50 digits (more work)
- base_w=18: 45 digits
- base_w=20: 40 digits (current)
- base_w=22: 37 digits
- base_w=24: 34 digits (fewer digits, but heavier per-digit error)

**Goal**: Find sweet spot that minimizes wall clock while keeping noise acceptable

**File**: `src/clifford_fhe_v2/backends/cpu_optimized/keys.rs:346`

---

### 🎯 MEDIUM PRIORITY

#### 6. Key Serialization & Caching
**Goal**: Turn "3 minute keygen" into "instant load"

**Implementation**:
```rust
// In examples or lib
fn load_or_generate_keys(params: &CliffordFHEParams) -> (PublicKey, SecretKey, EvaluationKey) {
    let cache_path = format!(".keys/n{}_primes{}.bin", params.n, params.moduli.len());

    if let Ok(keys) = load_keys_from_disk(&cache_path) {
        println!("✓ Loaded cached keys from {}", cache_path);
        return keys;
    }

    println!("Generating keys (this will take ~2-3 minutes)...");
    let keys = KeyContext::new(params.clone()).keygen();

    save_keys_to_disk(&keys, &cache_path)?;
    println!("✓ Saved keys to {} for future use", cache_path);

    keys
}
```

**Benefits**:
- First run: 2-3 minutes
- Subsequent runs: <1 second
- Perfect for demos and CI/CD

---

### 📈 PROFILING & VALIDATION

#### 7. Profile Current Code
**Commands**:
```bash
# macOS
cargo install flamegraph
sudo cargo flamegraph --release --features v2,v3 --example test_v3_full_bootstrap

# Linux perf
perf record --call-graph dwarf cargo run --release --features v2,v3 --example test_v3_full_bootstrap
perf report

# Check allocation counts
RUST_BACKTRACE=1 cargo run --release --features v2,v3 --example test_v3_full_bootstrap
```

**What to look for**:
- Top functions should be: NTT butterflies, pointwise mul, gadget decomposition
- Should NOT see: allocator functions, context creation, lock contention

---

#### 8. Sanity Checks
```bash
# Test parallelism effectiveness
RAYON_NUM_THREADS=1 cargo run --release --features v2,v3 --example test_v3_full_bootstrap
# vs
RAYON_NUM_THREADS=8 cargo run --release --features v2,v3 --example test_v3_full_bootstrap

# Expected: 3-5× speedup with 8 cores
# If <2×, nested parallelism or allocs are stealing wins
```

---

## Realistic Performance Targets

### Current (after NTT caching):
- N=8192, 16 primes: ~3 minutes (estimate, still testing)

### After Buffer Reuse + RNG optimization:
- N=8192, 16 primes: ~90-120 seconds

### After base_w tuning:
- N=8192, 16 primes: ~60-90 seconds

### With Key Caching:
- N=8192, 16 primes: <1 second (load from disk)

---

## Implementation Order

1. ✅ NTT context caching (DONE)
2. ✅ Flatten parallelism (DONE)
3. **🔴 Buffer reuse (DO NEXT)** ← Biggest remaining win
4. **🟡 RNG optimization** ← Quick fix, measurable gain
5. **🟡 base_w tuning** ← Experimental, find optimum
6. **🟢 Key caching** ← Makes everything instant after first run
7. **📊 Profile & validate** ← Confirm improvements

---

## Benchmark Comparisons

**Question**: Is 2-3 minutes reasonable for N=8192, 16 primes?

**Answer from Expert**:
> "SEAL/OpenFHE class libs typically land in the tens of seconds to ~2 minutes on a single workstation core pool for N=8192, ~16 scaling primes, full eval+rot keys."

**Our target**: After optimizations, aim for ~60-90 seconds cold, <5 seconds with key caching

---

## Files to Modify

1. **`src/clifford_fhe_v2/backends/cpu_optimized/keys.rs`**
   - Add thread-local buffers
   - Switch to ChaCha20 RNG
   - Add base_w parameter tuning
   - Add in-place NTT operations

2. **`src/clifford_fhe_v3/bootstrapping/bootstrap_context.rs`**
   - Add key caching hooks
   - Progress indicators for keygen steps

3. **`examples/test_v3_full_bootstrap.rs`**
   - Add key caching example
   - Add timing breakdown per step

---

## Next Actions (for Human Review)

Based on expert feedback, recommend:

**Option A: Full Optimization Path** (2-3 hours work)
- Implement buffer reuse
- Optimize RNG
- Tune base_w
- Add key caching
- Target: 60-90s cold, instant warm

**Option B: Accept Current + Add Caching** (30 min work)
- Keep current 2-3 minute keygen
- Add key caching (makes it instant after first run)
- Document as expected for production params

**Option C: Hybrid Approach** (1 hour work)
- Buffer reuse only (biggest win, ~50% speedup)
- Add key caching
- Target: ~90-120s cold, instant warm

**Recommendation**: Option C (hybrid) - best ROI for time invested
