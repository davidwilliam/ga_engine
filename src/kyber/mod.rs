//! CRYSTALS-Kyber: NIST-Selected Post-Quantum Key Encapsulation Mechanism
//!
//! This module implements CRYSTALS-Kyber with both classical and GA-accelerated
//! batch operations to demonstrate performance advantages in post-quantum cryptography.
//!
//! ## Background
//!
//! Kyber is the NIST-selected post-quantum encryption standard (2022). It's based
//! on the Module Learning With Errors (MLWE) problem over polynomial rings.
//!
//! **Core Operation**: Matrix-vector multiplication A·s where:
//! - A: k×k matrix of polynomials in Rq
//! - s: k-vector of polynomials
//! - Rq = Zq[x]/(x^n + 1) with q=3329, n=256
//!
//! ## Security Levels
//!
//! - **Kyber-512** (k=2): AES-128 equivalent
//! - **Kyber-768** (k=3): AES-192 equivalent
//! - **Kyber-1024** (k=4): AES-256 equivalent
//!
//! ## Our GA Acceleration Strategy
//!
//! Individual operations use small matrices (2×2, 3×3, 4×4) that don't benefit
//! from our 8×8 GA speedups. Instead, we **batch multiple operations**:
//!
//! - Batch 4 Kyber-512 encryptions → single 8×8 operation
//! - Expected speedup: 2.0-2.5× (based on NTRU success: 2.57×)
//!
//! ## References
//!
//! - NIST PQC Standardization (2022): Kyber selected for standardization
//! - Bos et al. (2017): "CRYSTALS - Kyber: a CCA-secure module-lattice-based KEM"
//! - Kyber specification round 3 (2021)
//! - "Efficient Batch Algorithms for Post-Quantum Crystals" (2024):
//!   Achieved 12-30% improvement with batch matrix multiplication
//!
//! ## Our Target
//!
//! Achieve 2× speedup on batched Kyber-512 operations using GA, significantly
//! exceeding the 12-30% improvements shown in recent batch optimization papers.

pub mod polynomial;
pub mod params;
pub mod classical;
pub mod batch;
pub mod ga_batch;

pub use polynomial::KyberPoly;
pub use params::KyberParams;
pub use classical::{kyber_encrypt_single, kyber_keygen};
pub use batch::kyber_encrypt_batch_classical;
