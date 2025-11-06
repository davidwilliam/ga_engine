# Installation Guide

Complete setup instructions for GA Engine across all platforms and backends.

## System Requirements

### Minimum Requirements
- **CPU**: Multi-core x86_64 or ARM64 processor
- **RAM**: 4GB
- **Storage**: 2GB free space
- **OS**: Linux, macOS, or Windows
- **Rust**: 1.75 or newer

### Recommended Configuration
- **CPU**: 8+ cores (Apple M3/M4, AMD Ryzen 9, Intel Core i9)
- **RAM**: 16GB+ (32GB for large datasets)
- **OS**: macOS 14+ (for Metal) or Linux (for CUDA)

### GPU Backend Requirements

**Metal GPU** (Apple Silicon):
- macOS 13.0+
- Apple M1/M2/M3/M4 chip
- Xcode Command Line Tools

**CUDA GPU** (NVIDIA):
- NVIDIA GPU with Compute Capability 7.0+
- CUDA Toolkit 12.0+
- Linux (Ubuntu 22.04+ recommended)

## Installation Steps

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustc --version  # Verify ≥1.75
```

### 2. Clone Repository

```bash
git clone https://github.com/davidwilliamsilva/ga_engine.git
cd ga_engine
```

### 3. Choose Backend

#### Option A: CPU Backend (Recommended for Local Development)
```bash
cargo build --release --features f64,nd,v2 --no-default-features
cargo test --lib --features f64,nd,v2 --no-default-features
```

#### Option B: Metal GPU (Apple Silicon)
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Build with Metal support
cargo build --release --features f64,nd,v2-gpu-metal --no-default-features
cargo test --test test_geometric_operations_metal --features f64,nd,v2-gpu-metal --no-default-features -- --nocapture
```

#### Option C: CUDA GPU (NVIDIA)
```bash
# Install CUDA Toolkit 12.0+
# Visit: https://developer.nvidia.com/cuda-downloads

# Verify CUDA installation
nvcc --version

# Set CUDA environment
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Build with CUDA support
cargo build --release --features f64,nd,v2-gpu-cuda --no-default-features
cargo test --test test_geometric_operations_cuda --features f64,nd,v2-gpu-cuda --no-default-features -- --nocapture
```

#### Option D: V3 with Bootstrapping (Currently Has Compilation Errors)
```bash
# Note: V3 needs fixes before this works
cargo build --release --features f64,nd,v2,v3 --no-default-features
cargo test --lib --features f64,nd,v2,v3 --no-default-features
```

## Verification

### Quick Test
```bash
# V2 CPU (38× speedup)
cargo run --release --features f64,nd,v2 --no-default-features --example encrypted_3d_classification

# Expected output: ~11.42s (V1) → ~300ms (V2 CPU)
```

### Full Test Suite
```bash
# V1 Baseline (67 tests)
cargo test --lib --features f64,nd,v1 --no-default-features

# V2 Optimized (132 tests)
cargo test --lib --features f64,nd,v2 --no-default-features

# V3 Bootstrapping (needs fixes)
cargo test --lib --features f64,nd,v2,v3 --no-default-features

# All versions with lattice reduction (requires working CMake)
cargo test --features v1,v2,lattice-reduction
```

## Troubleshooting

### Metal GPU Issues
**Problem**: "Metal not found" error

**Solution**:
```bash
xcode-select --install
# Restart terminal
```

### CUDA Issues
**Problem**: "CUDA not found" or linking errors

**Solution**:
```bash
# Verify CUDA installation
nvcc --version
ls $CUDA_PATH/lib64/libcudart.so

# Add to ~/.bashrc or ~/.zshrc:
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

### Build Performance
For faster builds:
```bash
# Use more cores
export CARGO_BUILD_JOBS=8

# Or build specific examples only
cargo build --release --features v2 --example encrypted_3d_classification
```

## Next Steps

- See [QUICK_START.md](QUICK_START.md) for running examples
- See [API_REFERENCE.md](API_REFERENCE.md) for API documentation
- See [PERFORMANCE.md](PERFORMANCE.md) for benchmarking

For issues, contact: dsilva@datahubz.com or open GitHub issue.
