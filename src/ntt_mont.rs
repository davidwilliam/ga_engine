//! NTT with Montgomery reduction for Clifford ring polynomials
//!
//! This module combines NTT with Montgomery reduction for maximum performance.
//! Key insight: Keep values in Montgomery form throughout NTT computation!
//!
//! Performance gain: ~2× faster modular operations vs standard NTT

use crate::clifford_ring_int::{CliffordPolynomialInt, CliffordRingElementInt};
use crate::montgomery::MontgomeryContext;
use crate::ntt::NTTContext;

/// Multiply two Clifford polynomials using NTT with Montgomery reduction
///
/// This is the fastest polynomial multiplication we have!
/// - NTT: O(N log N) complexity
/// - Montgomery: ~2× faster modular operations
/// - Combined: ~2× speedup over standard NTT
///
/// Algorithm:
/// 1. Convert polynomial coefficients to Montgomery form
/// 2. Apply component-wise forward NTT (in Montgomery domain)
/// 3. Point-wise geometric products (using Montgomery multiplication)
/// 4. Apply component-wise inverse NTT (still in Montgomery domain)
/// 5. Convert result back from Montgomery form
///
/// The key optimization: Steps 2-4 stay in Montgomery form!
pub fn multiply_ntt_montgomery(
    a: &CliffordPolynomialInt,
    b: &CliffordPolynomialInt,
    ntt: &NTTContext,
    mont: &MontgomeryContext,
) -> CliffordPolynomialInt {
    let n = a.coeffs.len();
    assert_eq!(n, b.coeffs.len(), "Polynomials must have same length");
    assert_eq!(n, ntt.n, "Polynomial length must match NTT context");

    // Step 1: Extract components and convert to Montgomery form
    let mut a_ntt = vec![vec![0i64; n]; 8];
    let mut b_ntt = vec![vec![0i64; n]; 8];

    for component in 0..8 {
        for i in 0..n {
            // Extract coefficient and convert to Montgomery form
            let a_val = a.coeffs[i].coeffs[component];
            let b_val = b.coeffs[i].coeffs[component];

            a_ntt[component][i] = mont.to_montgomery(a_val);
            b_ntt[component][i] = mont.to_montgomery(b_val);
        }

        // Forward NTT (values already in Montgomery form!)
        ntt_forward_montgomery(&mut a_ntt[component], ntt, mont);
        ntt_forward_montgomery(&mut b_ntt[component], ntt, mont);
    }

    // Step 2: Point-wise geometric product in frequency domain
    // All values stay in Montgomery form!
    let mut c_ntt = vec![vec![0i64; n]; 8];

    // Pre-allocate result buffer for in-place operations
    let mut c_elem = CliffordRingElementInt::zero();

    for k in 0..n {
        // Construct multivector at frequency index k (in Montgomery form)
        let a_elem = CliffordRingElementInt::from_multivector([
            a_ntt[0][k], a_ntt[1][k], a_ntt[2][k], a_ntt[3][k],
            a_ntt[4][k], a_ntt[5][k], a_ntt[6][k], a_ntt[7][k],
        ]);

        let b_elem = CliffordRingElementInt::from_multivector([
            b_ntt[0][k], b_ntt[1][k], b_ntt[2][k], b_ntt[3][k],
            b_ntt[4][k], b_ntt[5][k], b_ntt[6][k], b_ntt[7][k],
        ]);

        // Geometric product using Montgomery multiplication
        // Inputs and outputs all in Montgomery form!
        c_elem = a_elem.geometric_product_montgomery(&b_elem, mont);

        // Store result components (still in Montgomery form)
        for component in 0..8 {
            c_ntt[component][k] = c_elem.coeffs[component];
        }
    }

    // Step 3: Inverse NTT for each component (values stay in Montgomery form)
    for component in 0..8 {
        ntt_inverse_montgomery(&mut c_ntt[component], ntt, mont);
    }

    // Step 4: Reconstruct Clifford polynomial and convert FROM Montgomery form
    let mut result_coeffs = Vec::with_capacity(n);
    for i in 0..n {
        // Extract components (in Montgomery form)
        let elem_mont = CliffordRingElementInt::from_multivector([
            c_ntt[0][i], c_ntt[1][i], c_ntt[2][i], c_ntt[3][i],
            c_ntt[4][i], c_ntt[5][i], c_ntt[6][i], c_ntt[7][i],
        ]);

        // Convert from Montgomery form
        let mut elem = elem_mont.clone();
        for j in 0..8 {
            elem.coeffs[j] = mont.from_montgomery(elem_mont.coeffs[j]);
        }

        result_coeffs.push(elem);
    }

    CliffordPolynomialInt::new(result_coeffs)
}

/// Forward NTT using Montgomery multiplication
///
/// Input: a[i] in Montgomery form
/// Output: a[i] in Montgomery form (frequency domain)
///
/// This is identical to standard NTT, but uses Montgomery multiplication!
fn ntt_forward_montgomery(a: &mut [i64], ntt: &NTTContext, mont: &MontgomeryContext) {
    let n = a.len();
    assert_eq!(n, ntt.n);
    let q = mont.q;

    // Bit-reversal permutation
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j >= bit {
            j -= bit;
            bit >>= 1;
        }
        j += bit;

        if i < j {
            a.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly (with Montgomery multiplication!)
    // Match the standard NTT structure from ntt.rs
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = 2 * n / len;

        for start in (0..n).step_by(len) {
            let mut k = 0;
            for j in start..(start + half) {
                let u = a[j];

                // Multiply by twiddle factor in Montgomery domain
                // psi[k] is in normal form, a[j+half] is in Montgomery form
                // We need: v = a[j+half] × psi[k] in Montgomery
                let twiddle_mont = mont.to_montgomery(ntt.psi[k]);
                let v = mont.mul_montgomery(a[j + half], twiddle_mont);

                // Butterfly
                a[j] = mont.add_montgomery(u, v);
                a[j + half] = mont.sub_montgomery(u, v);

                k += step;
            }
        }

        len *= 2;
    }
}

/// Inverse NTT using Montgomery multiplication
///
/// Input: a[i] in Montgomery form (frequency domain)
/// Output: a[i] in Montgomery form (time domain)
fn ntt_inverse_montgomery(a: &mut [i64], ntt: &NTTContext, mont: &MontgomeryContext) {
    let n = a.len();
    assert_eq!(n, ntt.n);
    let q = mont.q;

    // Bit-reversal permutation
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j >= bit {
            j -= bit;
            bit >>= 1;
        }
        j += bit;

        if i < j {
            a.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly with inverse twiddle factors
    // Match the standard NTT structure
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = 2 * n / len;

        for start in (0..n).step_by(len) {
            let mut k = 0;
            for j in start..(start + half) {
                let u = a[j];

                // Multiply by inverse twiddle factor in Montgomery domain
                let twiddle_inv_mont = mont.to_montgomery(ntt.psi_inv[k]);
                let v = mont.mul_montgomery(a[j + half], twiddle_inv_mont);

                // Butterfly
                a[j] = mont.add_montgomery(u, v);
                a[j + half] = mont.sub_montgomery(u, v);

                k += step;
            }
        }

        len *= 2;
    }

    // Normalize by N^(-1) (in Montgomery form!)
    let n_inv_mont = mont.to_montgomery(ntt.n_inv);
    for i in 0..n {
        a[i] = mont.mul_montgomery(a[i], n_inv_mont);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy_reduction::LazyReductionContext;

    #[test]
    fn test_ntt_montgomery_multiply() {
        let ntt = NTTContext::new_clifford_lwe();
        let mont = MontgomeryContext::new_clifford_lwe();
        let lazy = LazyReductionContext::new(3329);

        // Create two simple Clifford polynomials
        // a(x) = 1 + e1·x
        let mut a_coeffs = vec![CliffordRingElementInt::zero(); 32];
        a_coeffs[0] = CliffordRingElementInt::from_multivector([1, 0, 0, 0, 0, 0, 0, 0]); // 1
        a_coeffs[1] = CliffordRingElementInt::from_multivector([0, 1, 0, 0, 0, 0, 0, 0]); // e1
        let a = CliffordPolynomialInt::new(a_coeffs);

        // b(x) = 2 + e2·x
        let mut b_coeffs = vec![CliffordRingElementInt::zero(); 32];
        b_coeffs[0] = CliffordRingElementInt::from_multivector([2, 0, 0, 0, 0, 0, 0, 0]); // 2
        b_coeffs[1] = CliffordRingElementInt::from_multivector([0, 0, 1, 0, 0, 0, 0, 0]); // e2
        let b = CliffordPolynomialInt::new(b_coeffs);

        // NTT-Montgomery multiplication
        let mut result_ntt_mont = multiply_ntt_montgomery(&a, &b, &ntt, &mont);

        // Reference: Standard NTT multiplication
        use crate::ntt_clifford::multiply_ntt;
        let mut result_ntt = multiply_ntt(&a, &b, &ntt, &lazy);

        // Both should have same length after reduction
        result_ntt_mont.reduce_modulo_xn_minus_1_lazy(32, &lazy);
        result_ntt.reduce_modulo_xn_minus_1_lazy(32, &lazy);

        // Compare results
        assert_eq!(result_ntt_mont.coeffs.len(), result_ntt.coeffs.len());

        for i in 0..32 {
            for j in 0..8 {
                assert_eq!(
                    result_ntt_mont.coeffs[i].coeffs[j],
                    result_ntt.coeffs[i].coeffs[j],
                    "Mismatch at coeff[{}].coeffs[{}]: Montgomery={}, Standard={}",
                    i,
                    j,
                    result_ntt_mont.coeffs[i].coeffs[j],
                    result_ntt.coeffs[i].coeffs[j]
                );
            }
        }
    }

    #[test]
    fn test_ntt_montgomery_equivalence() {
        let ntt = NTTContext::new_clifford_lwe();
        let mont = MontgomeryContext::new_clifford_lwe();

        // Test forward+inverse round-trip for scalar values
        let mut data = vec![1, 2, 3, 5, 7, 11, 13, 17, 0, 0, 0, 0, 0, 0, 0, 0,
                            19, 23, 29, 31, 37, 41, 43, 47, 0, 0, 0, 0, 0, 0, 0, 0];

        // Convert to Montgomery form
        let original: Vec<i64> = data.iter().map(|&x| mont.to_montgomery(x)).collect();
        data.copy_from_slice(&original);

        // Forward NTT
        ntt_forward_montgomery(&mut data, &ntt, &mont);

        // Inverse NTT
        ntt_inverse_montgomery(&mut data, &ntt, &mont);

        // Should match original (in Montgomery form)
        for i in 0..32 {
            let recovered = mont.from_montgomery(data[i]);
            let expected = mont.from_montgomery(original[i]);
            assert_eq!(
                recovered, expected,
                "Round-trip failed at index {}: got {}, expected {}",
                i, recovered, expected
            );
        }
    }
}
