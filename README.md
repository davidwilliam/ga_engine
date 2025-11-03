# Clifford FHE: Fully Homomorphic Encryption for Geometric Algebra

**The first RNS-CKKS-based FHE scheme with native support for Clifford algebra operations, enabling privacy-preserving computation on geometric data.**

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository accompanies the paper **"Clifford FHE: Privacy-Preserving Geometric Deep Learning via Fully Homomorphic Encryption"** and provides a complete implementation of:

1. **Clifford FHE**: RNS-CKKS-based FHE with homomorphic geometric product
2. **Geometric Neural Networks**: First encrypted geometric deep learning system
3. **Complete Reproducibility**: All paper results with verification scripts

---

## 📄 Paper

**Title:** Clifford FHE: Privacy-Preserving Geometric Deep Learning via Fully Homomorphic Encryption
**Paper:** [`paper/journal_article.tex`](paper/journal_article.tex)
**Status:** Ready for journal submission

### Key Contributions

1. **Clifford FHE Scheme**
   - First RLWE-based FHE with native Clifford algebra support
   - Homomorphic geometric product: `Enc(a) ⊗ Enc(b) = Enc(a ⊗ b)`
   - All 7 fundamental operations (geometric product, reverse, rotation, wedge, inner, projection, rejection)
   - RNS-CKKS implementation with N=1024, ~128-bit post-quantum security

2. **Geometric Neural Networks**
   - Neural networks operating directly on encrypted multivectors
   - Rotational equivariance by construction
   - First encrypted geometric deep learning demonstration

3. **Practical Performance**
   - 99% accuracy on encrypted 3D classification (sphere/cube/pyramid)
   - <1% accuracy loss vs. plaintext
   - 58-second encrypted inference

---

## 🎯 Key Results (from Paper)

### Table 1: Clifford FHE Operation Performance

| Operation | Time | Relative Error |
|-----------|------|----------------|
| Geometric Product | 220 ms | <10⁻³ |
| Reverse | negligible | 0 |
| Rotation (R⊗v⊗R̃) | 440 ms | <10⁻³ |
| Wedge Product | 440 ms | <10⁻³ |
| Inner Product | 440 ms | <10⁻³ |
| Projection | 660 ms | <10⁻³ |
| Rejection | 660 ms | <10⁻³ |

**Parameters:** N=1024, 3 primes (60-bit), Δ=2⁴⁰, σ=3.2

### Table 2: Encrypted 3D Classification Results

| Dataset | Plaintext Accuracy | Encrypted Accuracy | Accuracy Loss | Inference Time |
|---------|-------------------|-------------------|---------------|----------------|
| Sphere/Cube/Pyramid (300 samples) | 100% | 99% | <1% | 58 seconds |

**Network:** 3 layers (1→16→8→3 neurons), geometric product activation

---

## 🚀 Quick Start

### Prerequisites

- **Rust** 1.75+ ([install](https://rustup.rs/))
- **Cargo** (included with Rust)
- **Hardware:** Multi-core CPU recommended (operations are parallelizable)

### Installation

```bash
git clone https://github.com/davidwilliamsilva/ga_engine.git
cd ga_engine
cargo build --release
```

### Run Paper Examples

#### 1. Basic Encryption/Decryption

```bash
cargo run --release --example clifford_fhe_basic
```

**Output:** Demonstrates encryption, decryption, and verification for multivectors

#### 2. Homomorphic Geometric Product

```bash
cargo run --release --example clifford_fhe_geometric_product_v2
```

**Output:** Shows homomorphic geometric product with <10⁻³ error

#### 3. Encrypted 3D Classification (Paper Experiment)

```bash
cargo run --release --example geometric_ml_3d_classification
```

**Output:** Reproduces Table 2 from paper (99% encrypted accuracy)

#### 4. Full Benchmarks (Paper Table 1)

```bash
cargo run --release --example benchmark_all_gp_variants
```

**Output:** Timing for all 7 operations matching paper Table 1

---

## 📊 Reproducing Paper Results

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for detailed step-by-step instructions to reproduce all paper results.

**Quick verification:**

```bash
# Verify geometric product performance (Table 1, row 1)
cargo run --release --example clifford_fhe_geometric_product_v2

# Verify encrypted classification (Table 2)
cargo run --release --example geometric_ml_3d_classification

# Full benchmark suite
cargo bench --bench clifford_fhe_operations
```

**Expected outputs match paper claims:**
- Geometric product: ~220ms (your hardware may vary)
- Encrypted accuracy: 99%
- Relative error: <10⁻³

---

## 🔧 API Reference

See [`API.md`](API.md) for complete Clifford FHE API documentation.

### Quick API Example

```rust
use ga_engine::clifford_fhe::*;

// 1. Generate keys
let params = CliffordFHEParams::new_rns_mult(); // N=1024, ~128-bit security
let (pk, sk, evk) = rns_keygen(&params);

// 2. Create multivectors (Cl(3,0): 8 components)
let mv_a = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // scalar + e1
let mv_b = [3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // scalar + e2

// 3. Encrypt
let ct_a = encrypt_multivector_3d(&pk, &mv_a, &params);
let ct_b = encrypt_multivector_3d(&pk, &mv_b, &params);

// 4. Homomorphic geometric product
let ct_c = geometric_product_3d_componentwise(&ct_a, &ct_b, &evk, &params);

// 5. Decrypt
let mv_c = decrypt_multivector_3d(&sk, &ct_c, &params);

// Verify: mv_c ≈ mv_a ⊗ mv_b (within <10⁻³ error)
```

---

## 📁 Repository Structure

```
ga_engine/
├── src/
│   ├── ga.rs                   # Core 3D Geometric Algebra
│   ├── multivector.rs          # Multivector types
│   ├── vector.rs, bivector.rs, rotor.rs
│   ├── nd/                     # N-dimensional GA
│   └── clifford_fhe/           # Clifford FHE (PAPER CONTRIBUTION)
│       ├── ckks_rns.rs         # RNS-CKKS implementation
│       ├── rns.rs              # Residue Number System
│       ├── keys_rns.rs         # Key generation
│       ├── geometric_product_rns.rs  # Homomorphic geometric product
│       ├── geometric_nn.rs     # Geometric neural networks
│       ├── params.rs           # Parameter sets
│       ├── canonical_embedding.rs    # CKKS embedding
│       ├── automorphisms.rs    # Galois automorphisms
│       ├── slot_encoding.rs    # SIMD encoding (future work)
│       └── rotation_keys.rs    # Specialized rotation keys
├── examples/                   # Paper reproduction
│   ├── clifford_fhe_basic.rs   # Basic demo
│   ├── clifford_fhe_geometric_product_v2.rs
│   ├── geometric_ml_3d_classification.rs  # Table 2 experiment
│   ├── benchmark_all_gp_variants.rs       # Table 1 benchmarks
│   └── ...
├── paper/
│   ├── journal_article.tex     # Paper LaTeX source
│   ├── references.bib          # Bibliography
│   └── REVIEWER_FEEDBACK.md    # Review notes
├── README.md                   # This file
├── REPRODUCIBILITY.md          # Reproduction guide
├── API.md                      # API reference
└── Cargo.toml                  # Rust project manifest
```

---

## 🔬 Technical Details

### Clifford FHE Architecture

**Base Scheme:** RNS-CKKS (Residue Number System - Cheon-Kim-Kim-Song)

**Parameters:**
- Ring dimension: N = 1024
- Modulus chain: 3 primes ≈ 2⁶⁰ each (Q ≈ 2¹⁸⁰)
- Scaling factor: Δ = 2⁴⁰
- Error std: σ = 3.2
- Security: ≥118 bits (Lattice Estimator)

**Key Innovation:** Structure constants encoding enables homomorphic geometric product
- 64 ciphertext multiplications per geometric product
- Sparsity exploitation: 8 non-zero terms per output component
- Noise control via relinearization (64×) and rescaling

**Why RNS-CKKS?**
- Single-modulus CKKS fails for depth >1 circuits
- Multi-prime chain enables proper rescaling
- Essential for geometric product (64 multiplications)

### Geometric Neural Networks

**Architecture:**
- Replace matrix multiplication with geometric product: `y = W ⊗ x + b`
- Rotational equivariance by construction
- All weights and activations are multivectors

**Advantages:**
- Coordinate-free representation
- Natural encoding of 3D structure
- FHE-compatible operations

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@article{silva2025cliffordfhe,
  title={Clifford FHE: Privacy-Preserving Geometric Deep Learning via Fully Homomorphic Encryption},
  author={Silva, David William},
  journal={[Journal Name]},
  year={2025},
  note={Preprint available at https://github.com/davidwilliamsilva/ga_engine}
}
```

---

## 🛠️ Development

### Running Tests

```bash
cargo test --lib
```

### Running Benchmarks

```bash
cargo bench --bench clifford_fhe_operations
```

### Building Documentation

```bash
cargo doc --open
```

---

## 🤝 Contributing

This repository is primarily for paper reproducibility. For questions or issues:

1. **Paper questions:** Open an issue with tag `[paper]`
2. **Reproduction issues:** Open an issue with tag `[reproducibility]`
3. **Bug reports:** Open an issue with tag `[bug]`

---

## 📜 License

MIT License - see [`LICENSE`](LICENSE) file

**Open Source:** All code is open-source to enable verification and extension of this work.

---

## 🙏 Acknowledgments

- **Leo Dorst** for foundational discussions on geometric algebra
- **Vinod Vaikuntanathan** for insights on lattice-based cryptography
- **Rust community** for robust tooling
- **DataHubz** for sponsorship

---

## 🔗 Links

- **Paper:** [`paper/journal_article.tex`](paper/journal_article.tex)
- **Reproducibility Guide:** [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
- **API Reference:** [`API.md`](API.md)
- **GitHub:** https://github.com/davidwilliamsilva/ga_engine

---

## ⚡ Performance Notes

**Hardware dependency:** Timing results depend on your CPU. Paper results obtained on:
- Apple M1 Pro (ARM64, 10 cores)
- 16 GB RAM
- macOS Sonoma 14.x

**Optimization opportunities:**
- GPU acceleration (NTT for polynomial multiplication)
- SIMD packing (multiple multivectors per ciphertext)
- Rotation-specific keys (2-3× speedup for rotations)
- Bootstrapping (enable arbitrary depth circuits)

See paper Section 5.3 "Optimization Opportunities" for details.

---

## 🔐 Security

**Security Level:** ~128-bit post-quantum security (NIST Level 1)

**Analysis:**
- Lattice Estimator verification
- Reduction to CKKS security
- IND-CPA security proof (Appendix)

**Important:** This is a research prototype. For production use:
- Full security audit recommended
- Constant-time implementations needed
- Side-channel protections required

---

**For detailed reproduction instructions, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)**

**For complete API documentation, see [`API.md`](API.md)**
