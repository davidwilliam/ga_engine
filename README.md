# GAEngine

**GAEngine** is a Rust library and benchmark suite comparing **classical** linear algebra (matrix multiplication) with **Geometric Algebra (GA)** implementations. Our goal is to demonstrate, with verifiable benchmarks, that GA-based kernels can match or exceed classical counterparts under equal operation counts.

## Project Status
- ✅ Core classical `multiply_matrices` (n×n) implemented and tested.  
- ✅ Core GA `geometric_product` (vector×vector) + full multivector `geometric_product_full` implemented and tested.  
- ✅ 3D semantic adapters: `Vec3`, `apply_matrix3`, `Rotor3::rotate`, and `Rotor3::rotate_fast` added with unit tests.  
- ✅ Added SIMD‐4× rotor: `Rotor3::rotate_simd` with unit tests.  
- ✅ Criterion benchmarks comparing:
  - **Classical** 8×8 matrix × 1 000 batch  
  - **GA** full 8-component product × 1 000 batch  
  - **rotate 3D point classical** × 1 000 batch  
  - **rotate 3D point GA (sandwich)** × 1 000 batch  
  - **rotate 3D point GA (fast)** × 1 000 batch  
  - **rotate 3D point GA (SIMD 4×)** × 1 000 batch  
- ✅ Optimized GA kernel using compile-time lookup table (`GP_PAIRS`) with zero runtime bit-twiddling.

## Key Findings
1. **Correctness**  
   - All unit tests (`identity`, `simple` matrices; `scalar_vector`, `vector_vector`; `rotate_z_90_degrees`, `rotate_z_90_degrees_fast`, `rotate_z_90_degrees_simd`) pass.  
2. **Performance**  
   - **Classical** 8×8 × 1 000 batch: **~260 µs**  
   - **GA** full multivector 8D × 1 000 batch: **~45 µs** (≈ 5.8× faster than classical)  
   - **rotate 3D point classical**: **~5.6 µs**  
   - **rotate 3D point GA (sandwich)**: **~93 µs**  
   - **rotate 3D point GA (fast)**: **~7.9 µs** (≈ 1.4× slower than classical)  
   - **rotate 3D point GA (SIMD 4×)**: **~10 µs** (≈ 2.5 ns per rotation, **2.2× faster** than classical)  

These results show that GA can be a drop-in, correct replacement for classical routines, and—when vectorized—can outperform them.

## How to Reproduce
1. **Install Rust** (via `rustup`)  
2. **Clone** this repo and change directory:
   ```bash
   git clone <this-repo>
   cd ga_engine
   ```
3. **Run tests:**
   ```bash
   cargo test
   ```
4. **Run benchmarks:**
   ```bash
   cargo bench
   ```
5. **Check coverage:**
   - Generate coverage JSON and summary in one step:
     ```bash
     make coverage
     ```
   - This runs:
     ```bash
     cargo llvm-cov --json --summary-only --output-path cov.json
     cargo run --bin coverage_summary
     ```
     producing a terminal coverage table.

## Next Steps
- **Micro-optimizations:** unroll, wider SIMD lanes (e.g. 8×), f32-precision kernels.  
- **Batch & parallel:** expose batch APIs and Rayon integration for large workloads.  
- **Killer-app demos:** integrate GA in small neural-net layers and FHE primitives with end-to-end benchmarks.  
- **Publish & document:** release on crates.io, add code samples, and plot performance charts.

*Chronology:*  
- **v0.1.0**: Baseline classical & GA implementations + semantic adapters + full benchmarks + coverage tooling + SIMD-4× rotor.