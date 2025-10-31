//! NTRU Polynomial Multiplication: Classical vs Geometric Algebra
//!
//! This module implements NTRU polynomial multiplication using both classical
//! and GA-based approaches to demonstrate performance advantages.
//!
//! ## NTRU Background
//!
//! NTRU (Nth-degree TRUncated polynomial ring) is a lattice-based cryptosystem
//! that operates on polynomials in the ring R = Z[x]/(x^N - 1).
//!
//! Polynomial multiplication in this ring is the core operation and computational
//! bottleneck of NTRU encryption/decryption.
//!
//! ## References
//!
//! - Hoffstein, J., Pipher, J., & Silverman, J. H. (1998). NTRU: A ring-based
//!   public key cryptosystem. In ANTS-III (pp. 267-288).
//! - NIST Post-Quantum Cryptography Standardization (2020).
//! - "Fast polynomial multiplication using matrix multiplication accelerators"
//!   (2024) - Achieves 1.54-3.07× speedup using matrix accelerators.

pub mod polynomial;
pub mod classical;
pub mod ga_based;

pub use polynomial::{Polynomial, NTRUParams};
pub use classical::{
    naive_multiply,
    karatsuba_multiply,
    toeplitz_matrix_multiply,
};
pub use ga_based::{
    ga_multiply_n8,
    ga_multiply_n16,
};
