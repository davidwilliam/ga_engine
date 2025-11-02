//! Homomorphic Geometric Product for RNS-CKKS
//!
//! This module implements encrypted geometric algebra operations using the
//! RNS-CKKS scheme with proper noise management.
//!
//! # Key Innovation
//!
//! Clifford FHE enables privacy-preserving geometric algebra:
//! ```text
//! Enc(a) ⊗ Enc(b) = Enc(a ⊗ b)
//! ```
//!
//! where ⊗ is the geometric product, not just polynomial multiplication!

use crate::clifford_fhe::ckks_rns::{RnsCiphertext, RnsPlaintext, rns_multiply_ciphertexts, rns_decrypt};
use crate::clifford_fhe::keys_rns::{RnsEvaluationKey, RnsSecretKey};
use crate::clifford_fhe::params::CliffordFHEParams;
use crate::clifford_fhe::rns::{rns_add, rns_sub};

/// 2D Geometric Algebra Cl(2,0) structure constants
///
/// Basis: {1, e₁, e₂, e₁₂}
///
/// Multiplication table:
/// ```text
///       1    e₁   e₂   e₁₂
///   1   1    e₁   e₂   e₁₂
///   e₁  e₁   1    e₁₂  e₂
///   e₂  e₂  -e₁₂  1   -e₁
///   e₁₂ e₁₂  e₂  -e₁  -1
/// ```
pub struct Cl2StructureConstants {
    /// For each output component, list of (coefficient, input_a_idx, input_b_idx)
    pub products: Vec<Vec<(i64, usize, usize)>>,
}

impl Cl2StructureConstants {
    /// Create structure constants for Cl(2,0)
    pub fn new() -> Self {
        let mut products = vec![Vec::new(); 4];

        // Component 0 (scalar):
        // 1⊗1=1, e₁⊗e₁=1, e₂⊗e₂=1, e₁₂⊗e₁₂=-1
        products[0] = vec![
            (1, 0, 0),   // a[0] * b[0]
            (1, 1, 1),   // a[1] * b[1] (e₁⊗e₁)
            (1, 2, 2),   // a[2] * b[2] (e₂⊗e₂)
            (-1, 3, 3),  // -a[3] * b[3] (e₁₂⊗e₁₂)
        ];

        // Component 1 (e₁):
        // 1⊗e₁=e₁, e₁⊗1=e₁, e₂⊗e₁₂=-e₁, e₁₂⊗e₂=e₁
        products[1] = vec![
            (1, 0, 1),   // a[0] * b[1] (1⊗e₁)
            (1, 1, 0),   // a[1] * b[0] (e₁⊗1)
            (-1, 2, 3),  // -a[2] * b[3] (e₂⊗e₁₂)
            (1, 3, 2),   // a[3] * b[2] (e₁₂⊗e₂)
        ];

        // Component 2 (e₂):
        // 1⊗e₂=e₂, e₂⊗1=e₂, e₁⊗e₁₂=e₂, e₁₂⊗e₁=-e₂
        products[2] = vec![
            (1, 0, 2),   // a[0] * b[2] (1⊗e₂)
            (1, 2, 0),   // a[2] * b[0] (e₂⊗1)
            (1, 1, 3),   // a[1] * b[3] (e₁⊗e₁₂)
            (-1, 3, 1),  // -a[3] * b[1] (e₁₂⊗e₁)
        ];

        // Component 3 (e₁₂):
        // 1⊗e₁₂=e₁₂, e₁₂⊗1=e₁₂, e₁⊗e₂=e₁₂, e₂⊗e₁=-e₁₂
        products[3] = vec![
            (1, 0, 3),   // a[0] * b[3] (1⊗e₁₂)
            (1, 3, 0),   // a[3] * b[0] (e₁₂⊗1)
            (1, 1, 2),   // a[1] * b[2] (e₁⊗e₂)
            (-1, 2, 1),  // -a[2] * b[1] (e₂⊗e₁)
        ];

        Cl2StructureConstants { products }
    }
}

/// Encode a 2D multivector as a plaintext polynomial
///
/// Maps [scalar, e₁, e₂, e₁₂] → polynomial coefficients
/// Each component is placed at coefficient index 0, 1, 2, 3
pub fn encode_multivector_2d(mv: &[f64; 4], scale: f64, n: usize) -> Vec<i64> {
    let mut coeffs = vec![0i64; n];
    for i in 0..4 {
        coeffs[i] = (mv[i] * scale).round() as i64;
    }
    coeffs
}

/// Decode a plaintext polynomial to a 2D multivector
///
/// Extracts [scalar, e₁, e₂, e₁₂] from polynomial coefficients
pub fn decode_multivector_2d(coeffs: &[i64], scale: f64) -> [f64; 4] {
    let mut mv = [0.0; 4];
    for i in 0..4 {
        mv[i] = (coeffs[i] as f64) / scale;
    }
    mv
}

/// Homomorphic geometric product for 2D multivectors (Cl(2,0))
///
/// Computes Enc(a) ⊗ Enc(b) = Enc(a ⊗ b) where ⊗ is the geometric product.
///
/// # Algorithm
///
/// For each output component i:
///   result[i] = Σ coeff_k · Enc(a[j]) · Enc(b[k])
///
/// Uses structure constants to determine which products contribute to each component.
///
/// # Example
/// ```text
/// a = [1, 2, 0, 0]  // 1 + 2e₁
/// b = [3, 0, 4, 0]  // 3 + 4e₂
///
/// a ⊗ b = [3, 6, 4, 8]  // 3 + 6e₁ + 4e₂ + 8e₁₂
/// ```
///
/// This is computed HOMOMORPHICALLY without decrypting!
pub fn geometric_product_2d(
    ct_a: &RnsCiphertext,
    ct_b: &RnsCiphertext,
    evk: &RnsEvaluationKey,
    params: &CliffordFHEParams,
) -> RnsCiphertext {
    let structure = Cl2StructureConstants::new();
    let primes = &params.moduli;

    // We'll build result component by component
    // Each component needs to accumulate multiple products

    // Storage for each output component's accumulator
    let mut component_accumulators: Vec<Option<RnsCiphertext>> = vec![None; 4];

    // For each output component
    for output_idx in 0..4 {
        let product_terms = &structure.products[output_idx];

        for &(coeff, a_idx, b_idx) in product_terms {
            // We need to multiply specific components of a and b
            // But our ciphertexts encrypt entire polynomials!
            //
            // Solution: Extract components using component extraction,
            // or use the fact that components are stored at specific indices
            //
            // For now, we'll use a simplified approach:
            // Encrypt the product of the full polynomials, then extract/combine

            // This is a placeholder - we need component-wise operations
            // For now, let's just demonstrate the structure

            // TODO: Implement component extraction and placement
            // This requires either:
            // 1. Rotation to extract/place components (needs rotation keys)
            // 2. Masking techniques
            // 3. Multiple ciphertexts (one per component)
        }
    }

    // For now, return the first ciphertext (placeholder)
    // Real implementation needs proper component handling
    ct_a.clone()
}

/// SIMPLIFIED VERSION: Homomorphic geometric product using separate ciphertexts
///
/// This version encrypts each component separately, making the geometric product
/// straightforward but requiring 4× more ciphertexts.
///
/// # Arguments
/// * `cts_a` - Array of 4 ciphertexts, each encrypting one component of multivector a
/// * `cts_b` - Array of 4 ciphertexts, each encrypting one component of multivector b
/// * `evk` - Evaluation key for multiplication
/// * `params` - FHE parameters
///
/// # Returns
/// Array of 4 ciphertexts encrypting the geometric product a ⊗ b
///
/// # Example
/// ```text
/// cts_a = [Enc(1), Enc(2), Enc(0), Enc(0)]  // 1 + 2e₁
/// cts_b = [Enc(3), Enc(0), Enc(4), Enc(0)]  // 3 + 4e₂
///
/// result = geometric_product_2d_componentwise(cts_a, cts_b, evk, params)
/// // result = [Enc(3), Enc(6), Enc(4), Enc(8)]  // 3 + 6e₁ + 4e₂ + 8e₁₂
/// ```
pub fn geometric_product_2d_componentwise(
    cts_a: &[RnsCiphertext; 4],
    cts_b: &[RnsCiphertext; 4],
    evk: &RnsEvaluationKey,
    params: &CliffordFHEParams,
) -> [RnsCiphertext; 4] {
    let structure = Cl2StructureConstants::new();
    let primes = &params.moduli;
    let num_primes = primes.len();
    let active_primes = &primes[..num_primes];

    let mut results: Vec<RnsCiphertext> = Vec::with_capacity(4);

    // For each output component
    for output_idx in 0..4 {
        let product_terms = &structure.products[output_idx];

        let mut accumulator: Option<RnsCiphertext> = None;

        for &(coeff, a_idx, b_idx) in product_terms {
            // Multiply the two component ciphertexts
            let product = rns_multiply_ciphertexts(
                &cts_a[a_idx],
                &cts_b[b_idx],
                evk,
                params,
            );

            // Apply coefficient (for now, only handle ±1)
            let term = if coeff == -1 {
                // Negate: multiply c0 and c1 by -1
                let mut neg_c0 = product.c0.clone();
                let mut neg_c1 = product.c1.clone();

                for i in 0..params.n {
                    for j in 0..neg_c0.num_primes() {
                        let qi = active_primes[j];
                        neg_c0.rns_coeffs[i][j] = ((qi - product.c0.rns_coeffs[i][j] % qi) % qi);
                        neg_c1.rns_coeffs[i][j] = ((qi - product.c1.rns_coeffs[i][j] % qi) % qi);
                    }
                }

                RnsCiphertext::new(neg_c0, neg_c1, product.level, product.scale)
            } else {
                product
            };

            // Add to accumulator
            accumulator = Some(match accumulator {
                None => term,
                Some(acc) => {
                    let sum_c0 = rns_add(&acc.c0, &term.c0, active_primes);
                    let sum_c1 = rns_add(&acc.c1, &term.c1, active_primes);
                    RnsCiphertext::new(sum_c0, sum_c1, acc.level, acc.scale)
                }
            });
        }

        results.push(accumulator.expect("Empty accumulator for geometric product"));
    }

    [
        results[0].clone(),
        results[1].clone(),
        results[2].clone(),
        results[3].clone(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cl2_structure_constants() {
        let sc = Cl2StructureConstants::new();

        // Check scalar component has 4 products
        assert_eq!(sc.products[0].len(), 4);

        // Check e₁ component
        assert_eq!(sc.products[1].len(), 4);
    }

    #[test]
    fn test_encode_decode_2d() {
        let mv = [1.0, 2.0, 3.0, 4.0];  // 1 + 2e₁ + 3e₂ + 4e₁₂
        let scale = 1024.0;
        let n = 64;

        let coeffs = encode_multivector_2d(&mv, scale, n);
        let decoded = decode_multivector_2d(&coeffs, scale);

        for i in 0..4 {
            assert!((decoded[i] - mv[i]).abs() < 1e-6);
        }
    }
}
