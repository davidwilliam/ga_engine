# GA Engine Feature Flags

Complete reference for feature flags and build configurations.

## Core Feature Flags

### FHE Version Selection

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `v1` | V1 baseline reference implementation | None |
| `v2` | V2 CPU-optimized backend (NTT + Rayon) | None |
| `v3` | V3 bootstrapping (uses V2 backend) | **Requires `v2`** |

**Important**: V3 is NOT backend-agnostic. V3 bootstrap implementations directly call V2 backend functions (CPU, Metal GPU, or CUDA GPU).

### GPU Backend Selection

| Feature | Description | Platform | Dependencies | Works With |
|---------|-------------|----------|--------------|------------|
| `v2-gpu-metal` | Metal GPU acceleration | macOS (Apple Silicon) | `metal`, `objc` | V2, V3 |
| `v2-gpu-cuda` | CUDA GPU acceleration | Linux/Windows (NVIDIA) | `cudarc` | V2, V3 |

**Important**: GPU backends automatically enable `v2`. They work with both V2 operations and V3 bootstrap.

### Other Backend Selection

| Feature | Description | Platform | Dependencies |
|---------|-------------|----------|--------------|
| `v2-cpu-optimized` | CPU-optimized NTT backend | All platforms | Requires `v2` |
| `v2-simd-batched` | SIMD slot packing (experimental) | All platforms | Requires `v2` |

### Optional Modules

| Feature | Description | Use Case | Dependencies |
|---------|-------------|----------|--------------|
| `lattice-reduction` | Lattice reduction for security analysis | CPU-only cryptanalysis | `nalgebra`, `nalgebra-lapack`, `blas-src` |

## Default Features

By default, the following features are enabled:

```toml
default = ["f64", "nd", "v1", "lattice-reduction"]
```

This provides:
- V1 FHE implementation (stable baseline)
- Lattice reduction for security analysis
- Full functionality on local development machines

## Common Build Patterns

### Local Development (Full Features)

```bash
# Use default features (includes lattice-reduction)
cargo build --release

# Or explicitly:
cargo build --release --features v1,v2,v3,lattice-reduction

# Run all tests
cargo test --release --features v1,v2,v3,lattice-reduction
```

### V3 Bootstrap - CUDA GPU (NVIDIA)

```bash
# Build V3 with CUDA backend (RECOMMENDED for NVIDIA GPUs)
cargo build --release --features v2,v2-gpu-cuda,v3

# Run CUDA GPU bootstrap
cargo run --release --features v2,v2-gpu-cuda,v3 --example test_cuda_bootstrap

# Results: ~11.95s bootstrap (100% GPU with relinearization)
```

### V3 Bootstrap - Metal GPU (Apple Silicon)

```bash
# Build V3 with Metal backend (RECOMMENDED for Apple Silicon)
cargo build --release --features v2,v2-gpu-metal,v3

# Run Metal GPU bootstrap
cargo run --release --features v2,v2-gpu-metal,v3 --example test_metal_gpu_bootstrap_native

# Results: ~60s bootstrap (100% GPU)
```

### V3 Bootstrap - CPU Reference

```bash
# Build V3 with CPU backend
cargo build --release --features v2,v3

# Run CPU bootstrap
cargo run --release --features v2,v3 --example test_v3_full_bootstrap

# Results: ~70s bootstrap
```

### Cloud GPU Instances (CUDA, No Lattice)

```bash
# Build without lattice-reduction to avoid long compile times
cargo build --release --features v2,v2-gpu-cuda,v3

# Run CUDA benchmarks
cargo test --release --features v2,v2-gpu-cuda
```

### V2 CPU Only (No GPU, No Lattice)

```bash
# Build V2 CPU backend only
cargo build --release --features v2 --no-default-features

# Run V2 CPU tests
cargo test --release --features v2 --no-default-features
```

## Feature Flag Dependencies

### Automatic Dependencies

When you enable certain features, they automatically enable their dependencies:

- **`v3`** → automatically enables **`v2`** (V3 uses V2 backend)
- **`v2-gpu-metal`** → automatically enables `v2`, `metal`, `objc`
- **`v2-gpu-cuda`** → automatically enables `v2`, `cudarc`
- **`v2-cpu-optimized`** → automatically enables `v2`
- **`v2-simd-batched`** → automatically enables `v2`
- **`lattice-reduction`** → automatically enables `nalgebra`, `nalgebra-lapack`, `blas-src`

### Dependency Graph

```
v3 ──requires──> v2

v2-gpu-metal ──enables──> v2
v2-gpu-cuda  ──enables──> v2
v2-cpu-optimized ──enables──> v2
v2-simd-batched  ──enables──> v2

v3 + v2-gpu-metal ──> V3 bootstrap using Metal GPU
v3 + v2-gpu-cuda  ──> V3 bootstrap using CUDA GPU
v3 + v2           ──> V3 bootstrap using CPU
```

### Optional Dependencies

These dependencies are only compiled when their corresponding features are enabled:

| Dependency | Feature Required | Purpose |
|------------|------------------|---------|
| `nalgebra` | `lattice-reduction` | Linear algebra for lattice reduction |
| `nalgebra-lapack` | `lattice-reduction` | LAPACK bindings for QR decomposition |
| `blas-src` | `lattice-reduction` | BLAS backend (Accelerate on macOS, netlib elsewhere) |
| `metal` | `v2-gpu-metal` | Metal GPU compute API |
| `objc` | `v2-gpu-metal` | Objective-C runtime for Metal |
| `cudarc` | `v2-gpu-cuda` | CUDA GPU compute API |

## Why Feature Flags?

### Problem: Build Issues on Cloud GPU Instances

When building on cloud GPU instances (RunPod, Lambda Labs, etc.), lattice reduction dependencies can cause issues:

1. **netlib-src**: Compiles BLAS/LAPACK from Fortran source (very slow, 1+ hours on some systems)
2. **Not needed**: Lattice reduction is CPU-only security analysis, not used for FHE GPU operations

### Solution: Optional lattice-reduction Feature

By making lattice reduction optional:

- **Local development**: Include lattice-reduction (default behavior)
- **Cloud GPU builds**: Omit lattice-reduction (fast builds, clean output)
- **FHE operations**: Unaffected (V1/V2/V3 work with or without lattice-reduction)

## Feature Combinations

### Valid Combinations

✅ **Recommended combinations:**
```bash
# V1 only
--features v1

# V2 CPU only
--features v2

# V2 + Metal GPU
--features v2,v2-gpu-metal

# V2 + CUDA GPU
--features v2,v2-gpu-cuda

# V3 with CPU backend
--features v2,v3

# V3 with Metal GPU backend (RECOMMENDED for Apple Silicon)
--features v2,v2-gpu-metal,v3

# V3 with CUDA GPU backend (RECOMMENDED for NVIDIA GPUs)
--features v2,v2-gpu-cuda,v3

# All versions with GPU support
--features v1,v2,v2-gpu-metal,v3
--features v1,v2,v2-gpu-cuda,v3
```

### Invalid Combinations

❌ **These will fail:**
```bash
# V3 without V2 (v3 requires v2)
--features v3  # ERROR: v3 requires v2

# Multiple GPU backends (conflicts)
--features v2-gpu-metal,v2-gpu-cuda  # ERROR: Cannot use both
```

### Understanding V3 + GPU Backend

When you use `--features v2,v2-gpu-cuda,v3`:
- V3 bootstrap functions (`cuda_*.rs`) will use CUDA GPU operations from V2
- All low-level operations (NTT, rescaling, rotation) execute on CUDA GPU
- V3 provides bootstrap algorithms, V2 provides GPU execution

When you use `--features v2,v2-gpu-metal,v3`:
- V3 bootstrap functions (`cuda_*.rs`) will use Metal GPU operations from V2
- All low-level operations (NTT, rescaling, rotation) execute on Metal GPU
- V3 provides bootstrap algorithms, V2 provides GPU execution

**Note**: The V3 files are named `cuda_*.rs` for historical reasons, but they use whichever V2 backend is enabled.

## Test Coverage

| Component | Test Count | Command |
|-----------|------------|---------|
| V1 Unit Tests | 31 | `cargo test --lib --features v1` |
| V2 Unit Tests | 127 | `cargo test --lib --features v2` |
| V3 Unit Tests | 52 | `cargo test --lib --features v2,v3 clifford_fhe_v3` |
| Lattice Reduction | ~60 | `cargo test --lib lattice_reduction --features lattice-reduction` |
| **Total (no lattice)** | **~210** | `cargo test --lib --features v1,v2,v3` |
| **Total (with lattice)** | **~270** | `cargo test --lib --features v1,v2,v3,lattice-reduction` |

## Troubleshooting

### Build stuck at netlib-src compilation

**Problem**: Accidentally included `lattice-reduction` feature on cloud GPU instance

**Solution**:
```bash
cargo clean
cargo build --release --features v2,v2-gpu-cuda,v3
```

### Lattice reduction tests not found

**Problem**: Built without `lattice-reduction` feature

**Solution**:
```bash
cargo test --lib lattice_reduction --features lattice-reduction
```

### V3 examples fail with "feature v2 required"

**Problem**: Tried to use V3 without enabling V2

**Solution**:
```bash
# V3 requires V2 - use both:
cargo build --release --features v2,v3

# Or with GPU:
cargo build --release --features v2,v2-gpu-cuda,v3
```

### undefined reference to BLAS symbols

**Problem**: Enabled `lattice-reduction` but BLAS backend not available

**Solution** (Linux):
```bash
# Install system BLAS (optional, netlib-src will compile from source if not found)
sudo apt-get install libblas-dev liblapack-dev
cargo clean && cargo build --release --features lattice-reduction
```

**Solution** (macOS):
```bash
# Xcode Command Line Tools includes Accelerate framework
xcode-select --install
cargo clean && cargo build --release --features lattice-reduction
```

### CUDA library not found

**Problem**: Built with `v2-gpu-cuda` but CUDA not installed

**Solution**:
```bash
# Verify CUDA installation
nvcc --version

# Set CUDA path
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Rebuild
cargo clean
cargo build --release --features v2,v2-gpu-cuda,v3
```

### Metal not found

**Problem**: Built with `v2-gpu-metal` but Xcode Command Line Tools not installed

**Solution**:
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Verify installation
xcode-select -p

# Rebuild
cargo clean
cargo build --release --features v2,v2-gpu-metal,v3
```

## Performance Impact

### Compile Time

| Configuration | Compile Time (from clean) | Reason |
|---------------|--------------------------|--------|
| `--features v2` | ~2 minutes | No Fortran compilation |
| `--features v2,lattice-reduction` | ~5-10 minutes | netlib-src Fortran compilation (if no system BLAS) |
| `--features v2,v2-gpu-cuda,v3` | ~3-4 minutes | CUDA kernel compilation |
| `--features v2,v2-gpu-metal,v3` | ~2-3 minutes | Metal shader compilation |
| Incremental builds | ~10-30 seconds | Cached dependencies |

### Runtime Performance

**No impact**: The `lattice-reduction` feature only affects what modules are compiled, not runtime performance of FHE operations. V2/V3 FHE operations have identical performance with or without lattice-reduction.

**GPU backend impact**:
- V3 CUDA GPU bootstrap: ~11.95s (5.86× faster than CPU)
- V3 Metal GPU bootstrap: ~60s (1.17× faster than CPU)
- See [BENCHMARKS.md](BENCHMARKS.md) for detailed performance measurements

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture (explains V3 uses V2 backend)
- [COMMANDS.md](COMMANDS.md) - Complete command reference
- [BENCHMARKS.md](BENCHMARKS.md) - Performance measurements
- [INSTALLATION.md](INSTALLATION.md) - Setup guide and system requirements
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Testing procedures