# Repository Cleanup Plan for Paper Submission

## Current State
- **110 example files** (many experimental/failed attempts)
- **2 benchmark files**
- **62 source files** (mix of core GA, Clifford FHE, and experimental optimizations)

## Goal
Clean repository showcasing only:
1. **Core Geometric Algebra** (foundation)
2. **Clifford FHE** (paper contribution)
3. **Geometric Neural Networks** (application)
4. **Paper reproduction** (experiments from paper)

---

## Files to KEEP

### Core Geometric Algebra (Foundation)
**src/**
- ✅ `ga.rs` - Core 3D GA implementation
- ✅ `multivector.rs` - Multivector types
- ✅ `vector.rs` - Vector operations
- ✅ `bivector.rs` - Bivector operations
- ✅ `rotor.rs` - Rotor (rotation) operations
- ✅ `prelude.rs` - Public API
- ✅ `lib.rs` - Library root
- ✅ `nd/mod.rs` - N-dimensional GA module
- ✅ `nd/ga.rs` - Generic GA implementation
- ✅ `nd/multivector.rs` - Generic multivector
- ✅ `nd/gp.rs` - Generic geometric product
- ✅ `nd/types.rs` - Type definitions
- ✅ `ops/mod.rs` - Operations module
- ✅ `ops/projection.rs` - Projection operations
- ✅ `ops/reflection.rs` - Reflection operations
- ✅ `ops/motor.rs` - Motor operations

### Clifford FHE (Paper Contribution)
**src/clifford_fhe/**
- ✅ `mod.rs` - Module root
- ✅ `params.rs` - Parameter sets
- ✅ `ckks_rns.rs` - RNS-CKKS implementation
- ✅ `rns.rs` - Residue Number System
- ✅ `keys_rns.rs` - RNS key generation
- ✅ `geometric_product_rns.rs` - Homomorphic geometric product
- ✅ `geometric_nn.rs` - Geometric neural networks
- ✅ `canonical_embedding.rs` - CKKS canonical embedding
- ✅ `automorphisms.rs` - Galois automorphisms
- ✅ `slot_encoding.rs` - SIMD slot encoding
- ✅ `slot_operations.rs` - Slot operations
- ✅ `rotation_keys.rs` - Specialized rotation keys

### Examples (Paper Reproduction)
**examples/**
- ✅ `clifford_fhe_basic.rs` - Basic encryption/decryption demo
- ✅ `clifford_fhe_geometric_product_v2.rs` - Geometric product demo (final version)
- ✅ `encrypted_3d_classification.rs` - Neural network experiment (IF EXISTS, else create)
- ✅ `benchmark_all_operations.rs` - Benchmark all 7 operations (rename from benchmark_*)

### Benchmarks
**benches/**
- ✅ `clifford_fhe_operations.rs` - Performance benchmarks

### Documentation
- ✅ `README.md` - (Will rewrite)
- ✅ `Cargo.toml` - Project manifest
- ✅ `LICENSE` - MIT license
- ✅ `.gitignore`
- ✅ `paper/` directory - Complete paper LaTeX files

---

## Files to REMOVE

### Experimental Implementations (Failed/Superseded)
**src/**
- ❌ `ckks.rs` - Old single-modulus CKKS (superseded by ckks_rns.rs)
- ❌ `keys.rs` - Old single-modulus keys (superseded by keys_rns.rs)
- ❌ `geometric_product.rs` - Old single-modulus GP (superseded by geometric_product_rns.rs)
- ❌ `operations.rs` - Old operations (superseded by RNS versions)
- ❌ `encoding.rs` - Old encoding (superseded by slot_encoding.rs)
- ❌ `simple_rotation.rs` - Experimental rotation (not used in paper)
- ❌ `clifford_ring.rs` - LWE-based approach (abandoned for CKKS)
- ❌ `clifford_ring_simd.rs` - SIMD LWE (abandoned)
- ❌ `clifford_ring_int.rs` - Integer-only LWE (abandoned)
- ❌ `clifford_lwe.rs` - LWE implementation (abandoned for CKKS)
- ❌ `ntt.rs` - Basic NTT (optimization attempt)
- ❌ `ntt_optimized.rs` - Optimized NTT (not used)
- ❌ `ntt_mont.rs` - Montgomery NTT (not used)
- ❌ `ntt_simd.rs` - SIMD NTT (not used)
- ❌ `ntt_clifford.rs` - NTT for Clifford rings (abandoned)
- ❌ `ntt_clifford_simd.rs` - SIMD NTT Clifford (abandoned)
- ❌ `ntt_clifford_optimized.rs` - Optimized NTT Clifford (abandoned)
- ❌ `montgomery.rs` - Montgomery reduction (optimization attempt, not used)
- ❌ `barrett.rs` - Barrett reduction (optimization attempt, not used)
- ❌ `lazy_reduction.rs` - Lazy reduction (optimization attempt, not used)
- ❌ `fast_rng.rs` - Fast RNG experiment (not used)
- ❌ `shake_rng.rs` - SHAKE RNG (not used)
- ❌ `shake_poly.rs` - SHAKE polynomial (not used)
- ❌ `ga_simd_optimized.rs` - SIMD GA optimization (not in paper)
- ❌ `classical.rs` - Classical algorithms (not related to paper)
- ❌ `transform.rs` - Transform utilities (not used)
- ❌ `numerical_checks/` directory - Numerical validation code (development only)
- ❌ `nd/ga4d_optimized.rs` - 4D optimization (not in paper)
- ❌ `nd/gp_lazy.rs` - Lazy GP (not in paper)
- ❌ `nd/vecn.rs` - N-dimensional vectors (not used)
- ❌ `ops/interpolation.rs` - Interpolation (not in paper)
- ❌ `bin/coverage_summary.rs` - Development utility

### Test/Debug Examples (110 files!)
**examples/** - Remove ALL except kept ones above

Debug/test files to remove:
- ❌ `test_*.rs` (all test files - ~30 files)
- ❌ `diagnose_*.rs` (diagnostic files)
- ❌ `trace_*.rs` (trace/debug files)
- ❌ `sanity_checks_*.rs` (sanity checks)
- ❌ `find_*.rs` (utility scripts)
- ❌ `analyze_*.rs` (analysis scripts)
- ❌ `profile_*.rs` (profiling scripts)

Experimental examples to remove:
- ❌ `clifford_lwe_*.rs` (all LWE experiments - ~15 files)
- ❌ `benchmark_ntt_*.rs` (NTT benchmarks)
- ❌ `benchmark_sparse_*.rs` (sparse benchmarks)
- ❌ `benchmark_shake_*.rs` (SHAKE benchmarks)
- ❌ `benchmark_multiplication_methods.rs`
- ❌ `benchmark_optimized_gp.rs` (superseded)
- ❌ `benchmark_core_operations.rs` (superseded)
- ❌ `clifford_fhe_geometric_product.rs` (v1, superseded by v2)
- ❌ `bivector_rotation.rs` (not in paper)
- ❌ `matrix_multivector_16x16_demo.rs` (not in paper)
- ❌ `rotate_cloud.rs` and `rotate_cloud_opt.rs` (not in paper)

### Transient Files
- ❌ `target/` - Build artifacts (already in .gitignore)
- ❌ `*.pdf` in root (if any)
- ❌ Temporary `.md` files created during development
- ❌ Old documentation files

---

## New Files to CREATE

### Examples for Paper Reproduction
1. **`examples/demo_basic_operations.rs`**
   - Demonstrates all 7 fundamental operations
   - Matches paper Section 4 (Implementation)

2. **`examples/encrypted_3d_classification.rs`**
   - Full neural network experiment
   - Matches paper Section 5 (Experiments)
   - 3D point cloud classification (sphere/cube/pyramid)

3. **`examples/benchmark_paper_results.rs`**
   - Reproduces all timing results from paper
   - Table 1: Operation timings
   - Table 2: Neural network results

### Documentation
1. **`README.md`** - Complete rewrite with:
   - Project overview
   - Paper reproduction instructions
   - Clifford FHE API documentation
   - Example usage
   - Citation information

2. **`REPRODUCIBILITY.md`**
   - Step-by-step instructions to reproduce paper results
   - Expected outputs
   - Hardware requirements
   - Timing benchmarks

3. **`API.md`**
   - Complete Clifford FHE API reference
   - All public functions
   - Parameter sets
   - Usage examples

---

## Directory Structure (After Cleanup)

```
ga_engine/
├── src/
│   ├── lib.rs
│   ├── prelude.rs
│   ├── ga.rs                    # Core 3D GA
│   ├── multivector.rs
│   ├── vector.rs
│   ├── bivector.rs
│   ├── rotor.rs
│   ├── nd/                      # N-dimensional GA
│   │   ├── mod.rs
│   │   ├── ga.rs
│   │   ├── multivector.rs
│   │   ├── gp.rs
│   │   └── types.rs
│   ├── ops/                     # GA operations
│   │   ├── mod.rs
│   │   ├── projection.rs
│   │   ├── reflection.rs
│   │   └── motor.rs
│   └── clifford_fhe/           # Clifford FHE (PAPER)
│       ├── mod.rs
│       ├── params.rs
│       ├── ckks_rns.rs
│       ├── rns.rs
│       ├── keys_rns.rs
│       ├── geometric_product_rns.rs
│       ├── geometric_nn.rs
│       ├── canonical_embedding.rs
│       ├── automorphisms.rs
│       ├── slot_encoding.rs
│       ├── slot_operations.rs
│       └── rotation_keys.rs
├── examples/
│   ├── demo_basic_operations.rs        # All 7 operations
│   ├── clifford_fhe_basic.rs          # Basic enc/dec
│   ├── clifford_fhe_geometric_product_v2.rs  # Geometric product demo
│   ├── encrypted_3d_classification.rs  # Neural network (Paper Sec 5)
│   └── benchmark_paper_results.rs      # Reproduce paper timings
├── benches/
│   └── clifford_fhe_operations.rs
├── paper/
│   ├── journal_article.tex
│   ├── references.bib
│   ├── journal_article.pdf
│   └── *.md (review documents)
├── README.md                   # NEW: Paper-focused
├── REPRODUCIBILITY.md          # NEW: Step-by-step guide
├── API.md                      # NEW: Complete API reference
├── Cargo.toml
├── LICENSE
└── .gitignore
```

---

## Execution Plan

### Phase 1: Remove Experimental Implementations
1. Remove old single-modulus CKKS files from src/clifford_fhe/
2. Remove LWE-based implementations (clifford_ring*.rs, clifford_lwe.rs)
3. Remove NTT optimization attempts (ntt*.rs)
4. Remove Montgomery/Barrett/lazy reduction files
5. Remove RNG experiments (fast_rng, shake_*)
6. Remove SIMD experiments (ga_simd_optimized.rs)
7. Remove numerical_checks directory
8. Remove unused nd/ files (ga4d_optimized, gp_lazy, vecn)
9. Remove unused ops/ files (interpolation)
10. Remove bin/ directory

### Phase 2: Clean Up Examples
1. Remove ALL test_*.rs files (~30 files)
2. Remove ALL clifford_lwe_*.rs files (~15 files)
3. Remove ALL diagnostic files (diagnose_*, trace_*, sanity_*, find_*, analyze_*, profile_*)
4. Remove benchmark_* files except paper-relevant ones
5. Remove v1 clifford_fhe_geometric_product.rs (keep v2)
6. Keep only: clifford_fhe_basic.rs, clifford_fhe_geometric_product_v2.rs

### Phase 3: Create New Examples
1. Create demo_basic_operations.rs (all 7 operations)
2. Create encrypted_3d_classification.rs (neural network experiment)
3. Create benchmark_paper_results.rs (reproduce paper tables)

### Phase 4: Update Documentation
1. Rewrite README.md (paper-focused)
2. Create REPRODUCIBILITY.md (step-by-step guide)
3. Create API.md (complete API reference)
4. Update Cargo.toml description

### Phase 5: Final Verification
1. Run `cargo build --release` - verify compilation
2. Run `cargo test` - verify tests pass
3. Run examples - verify they work
4. Run benchmarks - verify performance
5. Check paper/ directory is complete
6. Verify no transient files remain

---

## Success Criteria

✅ Repository contains ONLY paper-related code
✅ All examples reproduce paper results
✅ README provides clear reproduction instructions
✅ API documentation is complete
✅ Code compiles without warnings
✅ Benchmarks match paper performance claims
✅ Reviewer can easily understand and verify implementation

---

## Estimated File Reduction
- Before: 110 examples + 62 src files = 172 Rust files
- After: ~5 examples + ~32 src files = ~37 Rust files
- **Reduction: ~78%** (135 files removed)

Focus: Quality over quantity, clarity over completeness
