//! RNS-CKKS encryption scheme for Clifford algebra
//!
//! This is the RNS (Residue Number System) version of CKKS that enables
//! proper homomorphic multiplication with rescaling.
//!
//! Key differences from single-modulus CKKS:
//! - Coefficients stored as RNS tuples instead of single i64
//! - Rescaling drops a prime from the modulus chain
//! - Supports larger effective moduli (Q = q₀ · q₁ · ... can be 2^200+)

use crate::clifford_fhe::keys::{EvaluationKey, PublicKey, SecretKey};
use crate::clifford_fhe::params::CliffordFHEParams;
use crate::clifford_fhe::rns::{RnsPolynomial, rns_add, rns_multiply as rns_poly_multiply, rns_rescale};

/// Helper function for polynomial multiplication modulo q with negacyclic reduction
fn polynomial_multiply_ntt(a: &[i64], b: &[i64], q: i64, n: usize) -> Vec<i64> {
    // Temporary naive implementation
    // TODO: Use actual NTT for efficiency
    let mut result = vec![0i64; n];

    for i in 0..n {
        for j in 0..n {
            let idx = i + j;
            if idx < n {
                result[idx] = (result[idx] + a[i] * b[j]) % q;
            } else {
                // x^n = -1 reduction
                let wrapped_idx = idx % n;
                result[wrapped_idx] = (result[wrapped_idx] - a[i] * b[j]) % q;
            }
        }
    }

    result.iter().map(|&x| ((x % q) + q) % q).collect()
}

/// RNS-CKKS plaintext (polynomial in RNS representation)
#[derive(Debug, Clone)]
pub struct RnsPlaintext {
    /// Polynomial coefficients in RNS form
    pub coeffs: RnsPolynomial,
    /// Scaling factor used for encoding
    pub scale: f64,
    /// Ring dimension
    pub n: usize,
}

impl RnsPlaintext {
    /// Create plaintext from RNS polynomial
    pub fn new(coeffs: RnsPolynomial, scale: f64) -> Self {
        let n = coeffs.n;
        Self { coeffs, scale, n }
    }

    /// Create plaintext from regular coefficients
    pub fn from_coeffs(coeffs: Vec<i64>, scale: f64, primes: &[i64], level: usize) -> Self {
        let n = coeffs.len();
        let rns_coeffs = RnsPolynomial::from_coeffs(&coeffs, primes, n, level);
        Self::new(rns_coeffs, scale)
    }

    /// Convert to regular coefficients (for compatibility with canonical embedding)
    pub fn to_coeffs(&self, primes: &[i64]) -> Vec<i64> {
        self.coeffs.to_coeffs(primes)
    }
}

/// RNS-CKKS ciphertext
///
/// A ciphertext is a pair (c0, c1) of RNS polynomials in R_q
/// Decryption: m ≈ c0 + c1*s (mod Q) where Q = q₀ · q₁ · ...
#[derive(Debug, Clone)]
pub struct RnsCiphertext {
    /// First component (RNS polynomial)
    pub c0: RnsPolynomial,
    /// Second component (RNS polynomial)
    pub c1: RnsPolynomial,
    /// Current level (determines which primes are active)
    /// Level 0: all primes [q₀, q₁, q₂, ...]
    /// Level 1: dropped last prime [q₀, q₁, ...]
    pub level: usize,
    /// Scaling factor (carries through homomorphic operations)
    pub scale: f64,
    /// Ring dimension
    pub n: usize,
}

impl RnsCiphertext {
    /// Create new RNS ciphertext
    pub fn new(c0: RnsPolynomial, c1: RnsPolynomial, level: usize, scale: f64) -> Self {
        let n = c0.n;
        assert_eq!(c1.n, n, "Ciphertext components must have same length");
        assert_eq!(c0.level, level, "c0 level mismatch");
        assert_eq!(c1.level, level, "c1 level mismatch");
        Self {
            c0,
            c1,
            level,
            scale,
            n,
        }
    }
}

/// Encrypt plaintext using public key (RNS version)
///
/// RNS-CKKS encryption:
/// 1. Sample random r, e0, e1 from error distribution
/// 2. Convert to RNS representation
/// 3. Compute (all in RNS):
///    c0 = b*r + e0 + m
///    c1 = a*r + e1
///
/// # Arguments
/// * `pk` - Public key (should contain RNS polynomials - TODO: update PublicKey)
/// * `pt` - Plaintext in RNS form
/// * `params` - CKKS parameters with modulus chain
pub fn rns_encrypt(pk: &PublicKey, pt: &RnsPlaintext, params: &CliffordFHEParams) -> RnsCiphertext {
    use rand::{thread_rng, Rng};
    use rand_distr::{Distribution, Normal};

    let n = params.n;
    let primes = &params.moduli;
    let num_primes = primes.len();
    let mut rng = thread_rng();

    // Sample ternary random polynomial r ∈ {-1, 0, 1}^n
    let r: Vec<i64> = (0..n)
        .map(|_| {
            let val: f64 = rng.gen();
            if val < 0.33 {
                -1
            } else if val < 0.66 {
                0
            } else {
                1
            }
        })
        .collect();

    // Sample errors e0, e1 from Gaussian distribution
    let normal = Normal::new(0.0, params.error_std).unwrap();
    let e0: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();
    let e1: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();

    // Convert to RNS
    let r_rns = RnsPolynomial::from_coeffs(&r, primes, n, 0);
    let e0_rns = RnsPolynomial::from_coeffs(&e0, primes, n, 0);
    let e1_rns = RnsPolynomial::from_coeffs(&e1, primes, n, 0);

    // TODO: PublicKey needs to be updated to use RNS
    // For now, convert pk.a and pk.b to RNS
    let pk_a_rns = RnsPolynomial::from_coeffs(&pk.a, primes, n, 0);
    let pk_b_rns = RnsPolynomial::from_coeffs(&pk.b, primes, n, 0);

    // Compute b*r using RNS multiplication
    let br = rns_poly_multiply(&pk_b_rns, &r_rns, primes, polynomial_multiply_ntt);

    // Compute a*r using RNS multiplication
    let ar = rns_poly_multiply(&pk_a_rns, &r_rns, primes, polynomial_multiply_ntt);

    // c0 = b*r + e0 + m
    let c0_temp = rns_add(&br, &e0_rns, primes);
    let c0 = rns_add(&c0_temp, &pt.coeffs, primes);

    // c1 = a*r + e1
    let c1 = rns_add(&ar, &e1_rns, primes);

    RnsCiphertext::new(c0, c1, 0, pt.scale)
}

/// Decrypt ciphertext using secret key (RNS version)
///
/// RNS-CKKS decryption:
/// m' = c0 + c1*s (all in RNS, then convert back)
///
/// # Arguments
/// * `sk` - Secret key (should contain RNS polynomial - TODO: update SecretKey)
/// * `ct` - Ciphertext in RNS form
/// * `params` - CKKS parameters
pub fn rns_decrypt(sk: &SecretKey, ct: &RnsCiphertext, params: &CliffordFHEParams) -> RnsPlaintext {
    let n = ct.n;
    let primes = &params.moduli;
    let num_primes = primes.len() - ct.level; // Active primes at this level

    // TODO: SecretKey needs to be updated to use RNS
    // For now, convert sk.coeffs to RNS at the current level
    let active_primes = &primes[..num_primes];
    let sk_rns = RnsPolynomial::from_coeffs(&sk.coeffs, active_primes, n, ct.level);

    // Compute c1*s using RNS multiplication
    let c1s = rns_poly_multiply(&ct.c1, &sk_rns, active_primes, polynomial_multiply_ntt);

    // m' = c0 + c1*s
    let m_prime = rns_add(&ct.c0, &c1s, active_primes);

    RnsPlaintext::new(m_prime, ct.scale)
}

/// Homomorphic addition (RNS version)
///
/// Simply add the RNS polynomials component-wise.
/// Scales must match!
pub fn rns_add_ciphertexts(
    ct1: &RnsCiphertext,
    ct2: &RnsCiphertext,
    params: &CliffordFHEParams,
) -> RnsCiphertext {
    assert_eq!(ct1.level, ct2.level, "Ciphertexts must be at same level");
    assert_eq!(ct1.n, ct2.n, "Ciphertexts must have same dimension");

    // TODO: Handle scale mismatch (need to rescale to match)
    // For now, require same scale
    assert!((ct1.scale - ct2.scale).abs() < 1e-6, "Scales must match (for now)");

    let primes = &params.moduli;
    let num_primes = primes.len() - ct1.level;
    let active_primes = &primes[..num_primes];

    let c0 = rns_add(&ct1.c0, &ct2.c0, active_primes);
    let c1 = rns_add(&ct1.c1, &ct2.c1, active_primes);

    RnsCiphertext::new(c0, c1, ct1.level, ct1.scale)
}

/// Homomorphic multiplication with rescaling (RNS version)
///
/// This is the KEY operation that requires RNS!
///
/// Steps:
/// 1. Multiply polynomials (tensored ciphertext): (c0, c1) ⊗ (d0, d1) = (c0d0, c0d1+c1d0, c1d1)
/// 2. Relinearize: convert degree-2 back to degree-1 using evaluation key
/// 3. **Rescale**: drop the last prime from the modulus chain
///
/// After rescaling:
/// - Level increases by 1 (one fewer prime)
/// - Scale divided by the dropped prime
/// - Coefficients properly normalized
pub fn rns_multiply_ciphertexts(
    ct1: &RnsCiphertext,
    ct2: &RnsCiphertext,
    evk: &EvaluationKey,
    params: &CliffordFHEParams,
) -> RnsCiphertext {
    assert_eq!(ct1.level, ct2.level, "Ciphertexts must be at same level");
    assert_eq!(ct1.n, ct2.n, "Ciphertexts must have same dimension");

    let n = ct1.n;
    let primes = &params.moduli;
    let num_primes = primes.len() - ct1.level;
    let active_primes = &primes[..num_primes];

    // Step 1: Multiply ciphertexts (tensored product)
    // Degree-2 ciphertext: (d0, d1, d2) where m1 * m2 = d0 + d1*s + d2*s²
    let c0d0 = rns_poly_multiply(&ct1.c0, &ct2.c0, active_primes, polynomial_multiply_ntt);
    let c0d1 = rns_poly_multiply(&ct1.c0, &ct2.c1, active_primes, polynomial_multiply_ntt);
    let c1d0 = rns_poly_multiply(&ct1.c1, &ct2.c0, active_primes, polynomial_multiply_ntt);
    let c1d1 = rns_poly_multiply(&ct1.c1, &ct2.c1, active_primes, polynomial_multiply_ntt);

    // d1 = c0*d1 + c1*d0
    let d_mid = rns_add(&c0d1, &c1d0, active_primes);

    // Step 2: Relinearization
    // TODO: EvaluationKey needs to be updated to use RNS
    // For now, use a simplified approach
    let (new_c0, new_c1) = rns_relinearize_degree2(&c0d0, &d_mid, &c1d1, evk, active_primes, n);

    // Step 3: Rescaling - THIS IS THE KEY FIX!
    // Drop the last prime from the modulus chain
    let rescaled_c0 = rns_rescale(&new_c0, active_primes);
    let rescaled_c1 = rns_rescale(&new_c1, active_primes);

    // New scale: (scale1 * scale2) / q_last
    // Since we designed scale ≈ q_last, the new scale ≈ original scale
    let q_last = active_primes[num_primes - 1];
    let new_scale = (ct1.scale * ct2.scale) / (q_last as f64);
    let new_level = ct1.level + 1;

    RnsCiphertext::new(rescaled_c0, rescaled_c1, new_level, new_scale)
}

/// Relinearize degree-2 ciphertext to degree-1 (RNS version)
///
/// Input: (d0, d1, d2) where m = d0 + d1*s + d2*s²
/// Output: (c0, c1) where m ≈ c0 + c1*s
///
/// Uses evaluation key which encrypts s²
fn rns_relinearize_degree2(
    d0: &RnsPolynomial,
    d1: &RnsPolynomial,
    d2: &RnsPolynomial,
    evk: &EvaluationKey,
    primes: &[i64],
    n: usize,
) -> (RnsPolynomial, RnsPolynomial) {
    // TODO: Proper RNS relinearization requires RNS evaluation key
    // For now, use simplified approach: convert to coefficients, relinearize, convert back

    // Convert to regular coefficients
    let d0_coeffs = d0.to_coeffs(primes);
    let d1_coeffs = d1.to_coeffs(primes);
    let d2_coeffs = d2.to_coeffs(primes);

    // Use existing relinearization (this works mod each prime)
    // TODO: This is hacky - should do proper RNS relinearization
    let _q: i64 = primes.iter().product(); // This might overflow, but not used for now
    let (evk0, evk1) = &evk.relin_keys[0];

    // For now, just do simplified relinearization per-prime
    // This is not optimal but will work for testing
    // (polynomial_multiply_ntt is defined at module level)

    // Convert evk to RNS
    let evk0_rns = RnsPolynomial::from_coeffs(evk0, primes, n, d0.level);
    let evk1_rns = RnsPolynomial::from_coeffs(evk1, primes, n, d0.level);

    // d2 * evk0
    let d2_evk0 = rns_poly_multiply(d2, &evk0_rns, primes, polynomial_multiply_ntt);

    // d2 * evk1
    let d2_evk1 = rns_poly_multiply(d2, &evk1_rns, primes, polynomial_multiply_ntt);

    // c0 = d0 + d2*evk0
    let c0 = rns_add(d0, &d2_evk0, primes);

    // c1 = d1 + d2*evk1
    let c1 = rns_add(d1, &d2_evk1, primes);

    (c0, c1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rns_plaintext_conversion() {
        let coeffs = vec![123, 456, 789, -100];
        let scale = 1024.0;
        let primes = vec![1_099_511_627_689, 1_099_511_627_691];

        let pt = RnsPlaintext::from_coeffs(coeffs.clone(), scale, &primes, 0);
        let recovered = pt.to_coeffs(&primes);

        for i in 0..coeffs.len() {
            assert_eq!(coeffs[i], recovered[i]);
        }
    }
}
