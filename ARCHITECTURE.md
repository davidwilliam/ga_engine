# GA Engine Architecture

## Overview

The GA Engine implements Clifford Homomorphic Encryption with **three parallel implementations**:

- **V1:** Research prototype, stable (for Paper 1)
- **V2:** GPU-accelerated backend (Metal + CUDA)
- **V3:** Bootstrap-optimized implementation using V2 backend

## Directory Structure

```
src/
├── clifford_fhe/              # V1 - STABLE (Paper 1)
│   ├── ckks_rns.rs            # Basic RNS-CKKS
│   ├── geometric_product_rns.rs
│   └── ...
│
├── clifford_fhe_v2/           # V2 - GPU Backend (Metal + CUDA)
│   ├── backends/
│   │   ├── cpu_optimized/     # CPU with NTT optimization
│   │   │   ├── ckks.rs        # Core CKKS operations
│   │   │   ├── ntt.rs         # Harvey butterfly NTT
│   │   │   └── keys.rs        # Key generation
│   │   │
│   │   ├── gpu_metal/         # Apple Metal GPU backend
│   │   │   ├── ckks.rs        # Core CKKS operations
│   │   │   ├── ntt.rs         # GPU NTT transforms
│   │   │   ├── keys.rs        # Key generation (CPU)
│   │   │   ├── rotation.rs    # Rotation operations
│   │   │   ├── rotation_keys.rs   # Rotation key switching
│   │   │   ├── relin_keys.rs  # Relinearization keys
│   │   │   ├── device.rs      # Metal device management
│   │   │   └── shaders/
│   │   │       ├── ntt.metal         # NTT kernels
│   │   │       ├── pointwise.metal   # Element-wise ops
│   │   │       ├── galois.metal      # Galois automorphism
│   │   │       ├── gadget.metal      # Key switching
│   │   │       └── rns_fixed.metal   # GPU rescaling ⭐
│   │   │
│   │   ├── gpu_cuda/          # NVIDIA CUDA GPU backend
│   │   │   ├── ckks.rs        # Core CKKS operations
│   │   │   ├── ntt.rs         # GPU NTT transforms
│   │   │   ├── rotation.rs    # Rotation operations
│   │   │   ├── rotation_keys.rs   # Rotation key switching
│   │   │   ├── relin_keys.rs  # Relinearization keys
│   │   │   ├── device.rs      # CUDA device management
│   │   │   ├── geometric.rs   # Geometric product
│   │   │   └── kernels/
│   │   │       ├── ntt.cu            # NTT kernels
│   │   │       ├── galois.cu         # Galois automorphism
│   │   │       ├── gadget.cu         # Gadget decomposition
│   │   │       └── rns.cu            # RNS operations (rescaling, etc) ⭐
│   │   │
│   │   └── simd_batched/      # SIMD slot packing (experimental)
│   │
│   └── params.rs              # Parameter sets
│
└── clifford_fhe_v3/           # V3 - Bootstrap Optimized
    ├── bootstrapping/         # Bootstrap operations
    │   ├── bootstrap_context.rs   # Bootstrap orchestration
    │   │
    │   ├── coeff_to_slot.rs   # CPU CoeffToSlot transform
    │   ├── slot_to_coeff.rs   # CPU SlotToCoeff transform
    │   ├── eval_mod.rs        # CPU modular reduction
    │   │
    │   ├── cuda_bootstrap.rs      # CUDA bootstrap orchestration
    │   ├── cuda_coeff_to_slot.rs  # CUDA CoeffToSlot transform ⭐
    │   ├── cuda_slot_to_coeff.rs  # CUDA SlotToCoeff transform ⭐
    │   ├── cuda_eval_mod.rs       # CUDA modular reduction ⭐
    │   │
    │   ├── keys.rs            # Rotation key generation
    │   ├── rotation.rs        # Rotation helpers
    │   ├── diagonal_mult.rs   # Diagonal matrix multiplication
    │   ├── mod_raise.rs       # Modulus raising
    │   └── sin_approx.rs      # Sine approximation for EvalMod
    │
    ├── params.rs              # V3-optimized parameters
    └── prime_gen.rs           # Dynamic NTT-friendly prime generation
```

## Feature Flags

Defined in `Cargo.toml`:

```toml
[features]
default = ["f64", "nd", "v1", "lattice-reduction"]

# Version selection
v1 = []                        # Stable research implementation
v2 = []                        # GPU backend (Metal + CUDA)
v3 = ["v2"]                    # Bootstrap optimized (uses V2 backend)

# V2 backends (requires v2)
v2-cpu-optimized = ["v2"]           # CPU with NTT
v2-gpu-metal = ["v2", "metal"]      # Apple Metal GPU
v2-gpu-cuda = ["v2", "cudarc"]      # NVIDIA CUDA GPU
v2-simd-batched = ["v2"]            # SIMD slot packing
```

## V3 Uses V2 Backend

**IMPORTANT**: V3 is NOT backend-agnostic. V3 bootstrap implementations directly use V2's backend infrastructure:

- **CPU Version**: Uses `v2/backends/cpu_optimized/` for NTT, encoding, arithmetic
- **Metal Version**: Uses `v2/backends/gpu_metal/` for GPU-accelerated operations
- **CUDA Version**: Uses `v2/backends/gpu_cuda/` for GPU-accelerated operations

V3 provides bootstrap-specific algorithms (CoeffToSlot, SlotToCoeff, EvalMod) but delegates all low-level operations to V2.

## Bootstrap Implementation

### V3 Bootstrap Architecture

V3 provides **three bootstrap implementations** that all use V2 backends:

1. **CPU Bootstrap** (`coeff_to_slot.rs`, `slot_to_coeff.rs`, `eval_mod.rs`)
   - Backend: `v2/backends/cpu_optimized/`
   - Status: ✅ Working (reference implementation)
   - Use case: Testing and validation

2. **Metal GPU Bootstrap** (`cuda_coeff_to_slot.rs`, etc. - shared naming, uses Metal backend)
   - Backend: `v2/backends/gpu_metal/`
   - Status: ✅ Production stable (~3.6e-3 error)
   - Hardware: Apple Silicon (M1/M2/M3)
   - Performance: ~60s full bootstrap on M3 Max

3. **CUDA GPU Bootstrap** (`cuda_coeff_to_slot.rs`, `cuda_slot_to_coeff.rs`, `cuda_eval_mod.rs`)
   - Backend: `v2/backends/gpu_cuda/`
   - Status: ✅ Working (~11.95s full bootstrap)
   - Hardware: NVIDIA GPUs
   - Performance: ~11.95s full bootstrap with relinearization

**Key Principle**: V3 files named `cuda_*.rs` use whichever V2 GPU backend is enabled via feature flags (`v2-gpu-metal` or `v2-gpu-cuda`). The naming is historical.

### V2 Backend Operations Used by V3

V3 bootstrap calls these V2 operations:

**From `CudaCkksContext` (Metal or CUDA):**
- `encode()` - Encode plaintext values
- `ntt_forward_batched_gpu()` - Forward NTT
- `ntt_inverse_batched_gpu()` - Inverse NTT
- `exact_rescale_gpu()` - Exact rescaling with centered rounding
- `strided_to_flat()` - Layout conversion (strided → flat)
- `flat_to_strided()` - Layout conversion (flat → strided)
- `add_polynomials_gpu()` - Polynomial addition
- `subtract_polynomials_gpu()` - Polynomial subtraction
- `pointwise_multiply_polynomials_gpu_strided()` - Pointwise multiplication

**From rotation/key switching:**
- `apply_galois_automorphism_gpu()` - Galois rotation
- `key_switch_hoisted_gpu()` - Key switching for rotation

All these operations are **100% GPU-resident** - no CPU loops in the hot path.

## Key Components

### RNS (Residue Number System)

All versions use multi-prime RNS representation for efficiency:
- **V1/V2:** Fixed prime sets
- **V3:** Dynamic NTT-friendly prime generation

Example: 30 primes for N=1024 bootstrap
- 1× ~60-bit prime
- 29× ~45-bit primes

### NTT (Number Theoretic Transform)

Fast polynomial multiplication:
- **V1:** Schoolbook multiplication (O(n²))
- **V2 CPU:** Harvey butterfly NTT (O(n log n))
- **V2 Metal GPU:** Metal shader NTT with optimized memory access
- **V2 CUDA GPU:** CUDA kernel NTT with coalesced memory access
- **V3:** Uses V2's NTT implementations (CPU/Metal/CUDA)

### GPU Shaders (Metal)

Located in `src/clifford_fhe_v2/backends/gpu_metal/shaders/`:

| Shader | Purpose | Key Feature |
|--------|---------|-------------|
| `ntt.metal` | Forward/inverse NTT | Optimized for Apple GPU |
| `pointwise.metal` | Element-wise multiplication | Fused ops |
| `galois.metal` | Galois automorphism | Rotation permutation |
| `gadget.metal` | Gadget decomposition | Key switching |
| `rns_fixed.metal` | **Exact rescaling** | Russian peasant mul_mod_128 ⭐ |

### GPU Kernels (CUDA)

Located in `src/clifford_fhe_v2/backends/gpu_cuda/kernels/`:

| Kernel | Purpose | Key Feature |
|--------|---------|-------------|
| `ntt.cu` | Forward/inverse NTT | Coalesced memory access |
| `galois.cu` | Galois automorphism | Rotation permutation |
| `gadget.cu` | Gadget decomposition | Key switching |
| `rns.cu` | **RNS operations** | Rescaling, add/sub, pointwise multiply ⭐ |

### GPU Rescaling

**Metal Backend (`rns_fixed.metal`):**

```metal
// Russian peasant multiplication (no 128-bit overflow)
inline ulong mul_mod_128(ulong a, ulong b, ulong q) {
    ulong result = 0;
    a = a % q;

    while (b > 0) {
        if (b & 1) {
            result = add_mod_lazy(result, a, q);
            if (result >= q) result -= q;
        }
        a = add_mod_lazy(a, a, q);
        if (a >= q) a -= q;
        b >>= 1;
    }

    return result;
}
```

**CUDA Backend (`rns.cu`):**

Uses the same Russian peasant algorithm for `mul_mod_128()` in:
- `rns_exact_rescale` kernel
- `rns_exact_rescale_strided` kernel (works directly on strided layout)
- `rns_pointwise_multiply_strided` kernel (128-bit safe multiplication)

**Validation:**
- Golden compare test: ✅ 0 mismatches (bit-exact with CPU)
- Bootstrap test: ✅ 3.6e-3 error (Metal), ~1e-3 error (CUDA)

## Testing

### Unit Tests
```bash
# V1 tests
cargo test --features v1

# V2 CPU tests
cargo test --features v2,v2-cpu-optimized

# V2 Metal GPU tests (requires Apple Silicon)
cargo test --features v2,v2-gpu-metal

# V2 CUDA GPU tests (requires NVIDIA GPU)
cargo test --features v2,v2-gpu-cuda

# V3 tests
cargo test --features v3
```

### Bootstrap Tests

**Metal GPU (Apple Silicon):**
```bash
# V3 Metal GPU bootstrap
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native
```

**CUDA GPU (NVIDIA):**
```bash
# V3 CUDA GPU bootstrap
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap
```

**CPU Reference:**
```bash
# V3 CPU reference
cargo run --release --features v2,v2-gpu-metal,v3 --example test_v3_metal_bootstrap_correct
```

### Validation Tests
```bash
# GPU rescaling golden compare (Metal)
cargo run --release --features v2,v2-gpu-metal,v3 --example test_rescale_golden_compare

# Layout conversion test (Metal)
cargo run --release --features v2,v2-gpu-metal,v3 --example test_multiply_rescale_layout
```

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive testing instructions.

## Performance

### Apple M3 Max (Metal GPU)

| Version | Total Time | Error | Notes |
|---------|-----------|-------|-------|
| **V3 CPU** | ~70s | 3.6e-3 | Reference implementation |
| **V3 Metal GPU** | ~60s | 3.6e-3 | **100% GPU** ⭐ |

**Per-Operation Breakdown:**

| Operation | Time | Backend |
|-----------|------|---------|
| Key Generation | ~73s | CPU |
| Encryption | ~175ms | GPU |
| CoeffToSlot (9 levels) | ~50s | GPU |
| SlotToCoeff (9 levels) | ~12s | GPU |
| Decryption | ~11ms | GPU |

### NVIDIA GPU (CUDA)

| Version | Total Time | Error | Notes |
|---------|-----------|-------|-------|
| **V3 CUDA GPU** | ~11.95s | ~1e-3 | **100% GPU with relinearization** ⭐ |

**Per-Operation Breakdown:**

| Operation | Time | Backend |
|-----------|------|---------|
| EvalMod | ~11.76s | GPU |
| CoeffToSlot | ~0.15s | GPU |
| SlotToCoeff | ~0.04s | GPU |
| Total Bootstrap | ~11.95s | GPU |

**Parameters:** N=1024, 30 primes (1× 60-bit, 29× 45-bit)

See [BENCHMARKS.md](BENCHMARKS.md) for detailed performance analysis.

## Development Guidelines

### V1 - Initial Point of Reference
- ❌ Frozen as the baseline
- ✅ Only critical bug fixes allowed
- ✅ Must maintain exact reproducibility

### V2 - Active Development
- ✅ Metal GPU optimization
- ✅ CUDA GPU optimization
- ✅ Performance tuning
- ✅ New shader/kernel development

### V3 - Bootstrap Research
- ✅ Parameter optimization
- ✅ Algorithm improvements
- ✅ Uses V2 backend (not independent)
- ⚠️ When adding V3 operations, first ensure V2 backend has the needed GPU functions

## Documentation

Core docs (kept):
- [README.md](README.md) - Project overview
- [ARCHITECTURE.md](ARCHITECTURE.md) - This file
- [BENCHMARKS.md](BENCHMARKS.md) - Performance measurements
- [COMMANDS.md](COMMANDS.md) - Command reference
- [FEATURE_FLAGS.md](FEATURE_FLAGS.md) - Feature flag reference
- [INSTALLATION.md](INSTALLATION.md) - Setup instructions
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Testing procedures

## Key Achievements

### V1
- ✅ First working Clifford FHE implementation
- ✅ All 7 geometric operations
- ✅ Paper 1 published

### V2
- ✅ Metal GPU backend operational
- ✅ CUDA GPU backend operational
- ✅ NTT optimization (10x+ speedup)
- ✅ Rotation operations working
- ✅ Relinearization keys (CUDA)
- ✅ GPU bootstrap (Metal/CUDA)

### V3
- ✅ Bootstrap-optimized parameters
- ✅ Dynamic prime generation
- ✅ CPU reference implementation
- ✅ Metal GPU implementation (~60s)
- ✅ CUDA GPU implementation (~11.95s)
- ✅ Bootstrap with relinearization

## Future Work

1. **Optimization**: Further pipeline improvements, persistent GPU buffers
2. **Batching**: Process multiple ciphertexts in parallel
3. **Geometric Operations**: GPU-accelerated geometric product on ciphertexts
4. **Portability**: Vulkan backend for cross-platform GPU support

## Author

Implementation by David Silva (contact@davidwsilva.com | dsilva@datahubz.com).

## License

See [LICENSE](LICENSE) file in the repository root.
