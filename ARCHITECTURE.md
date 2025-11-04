# GA Engine Architecture: V1 vs V2

## Design Philosophy

We maintain **two parallel implementations** of Clifford FHE:
- **V1:** Research prototype, feature-complete, stable (for journal review)
- **V2:** Optimized implementation with NTT/GPU/SIMD (active development)

This follows the **state-of-the-art pattern** used by major FHE libraries:
- **SEAL:** Uses feature flags + versioned namespaces
- **OpenFHE:** Modular backends with trait abstraction
- **HEAAN:** Separate optimization paths with common interfaces

## Directory Structure

```
src/
├── clifford_fhe/           # V1 - STABLE, DO NOT MODIFY
│   ├── mod.rs
│   ├── ckks_rns.rs         # Basic RNS-CKKS
│   ├── rns.rs              # Naive RNS arithmetic
│   ├── geometric_product_rns.rs  # All 7 operations
│   ├── keys_rns.rs
│   ├── params.rs
│   └── ... (other V1 files)
│
├── clifford_fhe_v2/        # V2 - ACTIVE DEVELOPMENT
│   ├── mod.rs
│   ├── core/               # Core abstractions (shared traits)
│   │   ├── mod.rs
│   │   ├── traits.rs       # CliffordFHE trait, Ciphertext trait, etc.
│   │   └── types.rs        # Common types
│   │
│   ├── backends/           # Multiple backends for benchmarking
│   │   ├── mod.rs
│   │   ├── cpu_optimized/  # NTT + SIMD (no GPU)
│   │   │   ├── mod.rs
│   │   │   ├── ntt.rs      # Harvey butterfly NTT
│   │   │   ├── rns.rs      # Optimized RNS arithmetic
│   │   │   └── geometric_product.rs
│   │   │
│   │   ├── gpu_cuda/       # CUDA backend (feature-gated)
│   │   │   ├── mod.rs
│   │   │   ├── kernels.cu  # CUDA kernels
│   │   │   └── geometric_product.rs
│   │   │
│   │   ├── gpu_metal/      # Metal backend (feature-gated)
│   │   │   ├── mod.rs
│   │   │   └── geometric_product.rs
│   │   │
│   │   └── simd_batched/   # SIMD slot packing
│   │       ├── mod.rs
│   │       └── geometric_product.rs
│   │
│   ├── ckks_rns.rs         # V2 encryption/decryption
│   ├── keys_rns.rs         # V2 key generation
│   ├── params.rs           # V2 parameter sets
│   └── rotation_keys.rs    # V2 specialized rotation keys
│
└── lib.rs                  # Feature flags expose V1 or V2
```

## Feature Flags (Cargo.toml)

```toml
[features]
default = ["v1"]  # Default to stable V1

# Version selection (mutually exclusive)
v1 = []           # Stable
v2 = []           # Optmized implementation (development)

# V2 backend selection (only active when v2 is enabled)
v2-cpu-optimized = ["v2"]     # NTT + SIMD, no GPU
v2-gpu-cuda = ["v2", "cudarc"] # CUDA backend
v2-gpu-metal = ["v2", "metal"] # Metal backend (Apple Silicon)
v2-simd-batched = ["v2"]      # SIMD slot packing

# Combined optimization (all techniques)
v2-full = ["v2", "v2-cpu-optimized", "v2-gpu-cuda", "v2-simd-batched"]
```

## Usage Examples

### V1 - Default
```bash
# Use stable V1 for Paper 1 validation
cargo test --features v1
cargo run --example encrypted_3d_classification --features v1
```

### V2 - Optimized
```bash
# CPU-only optimized version
cargo test --features v2-cpu-optimized

# CUDA GPU acceleration
cargo run --example encrypted_3d_classification --features v2-gpu-cuda

# Full optimization stack
cargo bench --features v2-full
```

## Code Example: Trait Abstraction

```rust
// src/clifford_fhe_v2/core/traits.rs
pub trait CliffordFHE {
    type Ciphertext;
    type Plaintext;
    type PublicKey;
    type SecretKey;
    type EvaluationKey;

    fn encrypt(&self, pk: &Self::PublicKey, pt: &Self::Plaintext) -> Self::Ciphertext;
    fn decrypt(&self, sk: &Self::SecretKey, ct: &Self::Ciphertext) -> Self::Plaintext;
    fn geometric_product(
        &self,
        a: &[Self::Ciphertext; 8],
        b: &[Self::Ciphertext; 8],
        evk: &Self::EvaluationKey,
    ) -> [Self::Ciphertext; 8];
}

// V1 implementation
impl CliffordFHE for v1::CliffordFHEV1 { ... }

// V2 CPU backend
impl CliffordFHE for v2::backends::CpuOptimized { ... }

// V2 CUDA backend
impl CliffordFHE for v2::backends::GpuCuda { ... }
```

## Migration Strategy

### Phase 1: Setup (Week 1)
1. Copy `clifford_fhe/` → `clifford_fhe_v1/` (preserve Paper 1)
2. Create `clifford_fhe_v2/` structure
3. Add feature flags to `Cargo.toml`
4. Update `lib.rs` with conditional compilation

### Phase 2: Core Traits (Week 1-2)
1. Define `CliffordFHE` trait and common types
2. Implement trait for V1 (wrapper around existing code)
3. Verify all V1 tests still pass

### Phase 3: V2 Backends (Weeks 2-7)
1. **cpu_optimized:** NTT + Barrett + SIMD (Phase 1 of roadmap)
2. **gpu_cuda:** CUDA kernels (Phase 2 of roadmap)
3. **simd_batched:** Slot packing (Phase 3 of roadmap)

### Phase 4: Examples & Benchmarks (Ongoing)
1. Duplicate examples for V1 and V2
2. Comparative benchmarks (V1 vs V2 backends)
3. Ablation studies (NTT only, GPU only, etc.)

## Benefits of This Approach

✅ **Paper 1 stays stable:** V1 frozen, reviewers can reproduce exactly
✅ **Independent development:** V2 work doesn't break V1
✅ **Easy comparison:** Same test cases, different backends
✅ **Feature flags:** Compile only what you need
✅ **Trait abstraction:** Swap backends without changing application code
✅ **Follows best practices:** Same pattern as SEAL, OpenFHE, Concrete

## Testing Strategy

```bash
# Verify V1 unchanged (Paper 1 validation)
cargo test --features v1 --lib
cargo test --features v1 --test test_geometric_operations

# Test V2 CPU backend
cargo test --features v2-cpu-optimized

# Test V2 CUDA backend (requires GPU)
cargo test --features v2-gpu-cuda

# Comparative benchmark (V1 vs V2 all backends)
cargo bench --features v2-full -- --save-baseline crypto2026
```

## Maintenance Rules

### V1 (`clifford_fhe/` or `clifford_fhe_v1/`)
- ❌ **NO MODIFICATIONS** except critical bug fixes
- ✅ Only touch if Paper 1 reviewers request changes
- ✅ All tests must continue passing
- ✅ Keep consistent with journal article text

### V2 (`clifford_fhe_v2/`)
- ✅ **ACTIVE DEVELOPMENT** 
- ✅ Aggressive optimization, breaking changes OK
- ✅ Benchmarks required for all changes
- ✅ Document performance improvements

## Documentation

- `V1/`: Links to stable sections
- `V2/`: Links to optmized README sections
- Each backend documents optimization techniques
- Benchmarks show speedup vs V1 baseline

---

**Next Steps:** Execute Phase 1 (setup) to create this structure.
