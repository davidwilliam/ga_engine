//! Kyber polynomial operations in Rq = Zq[x]/(x^256 + 1)
//!
//! This module provides the core polynomial arithmetic for Kyber.
//! Polynomials have 256 coefficients in the range [0, q) where q=3329.

use super::params::KyberParams;

/// Polynomial in Kyber's ring Rq = Zq[x]/(x^256 + 1)
///
/// Coefficients are stored in standard order: a[0] + a[1]·x + ... + a[255]·x^255
#[derive(Clone, PartialEq, Eq)]
pub struct KyberPoly {
    /// Coefficients in range [0, q)
    pub coeffs: Vec<i32>,
    /// Kyber parameters
    pub params: KyberParams,
}

impl KyberPoly {
    /// Create a new polynomial with given coefficients
    pub fn new(coeffs: Vec<i32>, params: KyberParams) -> Self {
        assert_eq!(
            coeffs.len(),
            params.n,
            "Polynomial must have exactly n={} coefficients",
            params.n
        );
        KyberPoly { coeffs, params }
    }

    /// Create a zero polynomial
    pub fn zero(params: KyberParams) -> Self {
        KyberPoly {
            coeffs: vec![0; params.n],
            params,
        }
    }

    /// Create a polynomial with random small coefficients in [-bound, bound]
    pub fn random_small(params: KyberParams, bound: i32, rng: &mut impl rand::Rng) -> Self {
        let coeffs = (0..params.n)
            .map(|_| rng.gen_range(-bound..=bound))
            .collect();
        KyberPoly { coeffs, params }
    }

    /// Create a polynomial with coefficients sampled from centered binomial distribution
    /// This is the standard Kyber sampling method for secrets and errors
    pub fn sample_cbd(params: KyberParams, eta: i32, rng: &mut impl rand::Rng) -> Self {
        let coeffs = (0..params.n)
            .map(|_| {
                // Sample from binomial distribution B(2η, 1/2) - η
                let mut sum = 0i32;
                for _ in 0..(2 * eta) {
                    if rng.gen_bool(0.5) {
                        sum += 1;
                    }
                }
                sum - eta
            })
            .collect();
        KyberPoly { coeffs, params }
    }

    /// Reduce all coefficients modulo q into range [0, q)
    pub fn reduce(&mut self) {
        for coeff in &mut self.coeffs {
            *coeff = mod_positive(*coeff, self.params.q);
        }
    }

    /// Add two polynomials (coefficient-wise modulo q)
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.params, other.params);
        let coeffs = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(&a, &b)| mod_positive(a + b, self.params.q))
            .collect();
        KyberPoly {
            coeffs,
            params: self.params,
        }
    }

    /// Subtract two polynomials (coefficient-wise modulo q)
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.params, other.params);
        let coeffs = self
            .coeffs
            .iter()
            .zip(other.coeffs.iter())
            .map(|(&a, &b)| mod_positive(a - b, self.params.q))
            .collect();
        KyberPoly {
            coeffs,
            params: self.params,
        }
    }

    /// Multiply polynomial by scalar (coefficient-wise modulo q)
    pub fn scalar_mul(&self, scalar: i32) -> Self {
        let coeffs = self
            .coeffs
            .iter()
            .map(|&c| mod_positive(c * scalar, self.params.q))
            .collect();
        KyberPoly {
            coeffs,
            params: self.params,
        }
    }

    /// Naive polynomial multiplication in Rq (O(n²))
    ///
    /// This is a simplified version for testing. Real Kyber uses NTT.
    /// Computes a·b mod (x^256 + 1) in Zq[x].
    pub fn mul_naive(&self, other: &Self) -> Self {
        assert_eq!(self.params, other.params);
        let n = self.params.n;
        let q = self.params.q;
        let mut result = vec![0i32; n];

        // Polynomial multiplication with reduction mod (x^n + 1)
        for i in 0..n {
            for j in 0..n {
                let idx = i + j;
                if idx < n {
                    // Normal case
                    result[idx] = mod_positive(result[idx] + self.coeffs[i] * other.coeffs[j], q);
                } else {
                    // Reduction: x^n = -1, so x^(n+k) = -x^k
                    let wrap_idx = idx - n;
                    result[wrap_idx] = mod_positive(result[wrap_idx] - self.coeffs[i] * other.coeffs[j], q);
                }
            }
        }

        KyberPoly {
            coeffs: result,
            params: self.params,
        }
    }

    /// Get coefficient at index i
    pub fn get(&self, i: usize) -> i32 {
        self.coeffs[i]
    }

    /// Set coefficient at index i
    pub fn set(&mut self, i: usize, value: i32) {
        self.coeffs[i] = mod_positive(value, self.params.q);
    }

    /// Compute L2 norm (sum of squared coefficients)
    pub fn norm_l2_squared(&self) -> i64 {
        self.coeffs.iter().map(|&c| (c as i64) * (c as i64)).sum()
    }
}

impl std::fmt::Debug for KyberPoly {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KyberPoly(n={}, first_coeffs={:?}...)",
            self.params.n,
            &self.coeffs[..std::cmp::min(8, self.coeffs.len())]
        )
    }
}

/// Reduce integer to range [0, q)
#[inline]
pub fn mod_positive(a: i32, q: i32) -> i32 {
    let mut r = a % q;
    if r < 0 {
        r += q;
    }
    r
}

/// Vector of k polynomials (used for Kyber vectors)
#[derive(Clone)]
pub struct PolyVec {
    pub polys: Vec<KyberPoly>,
    pub params: KyberParams,
}

impl PolyVec {
    /// Create new polynomial vector
    pub fn new(polys: Vec<KyberPoly>, params: KyberParams) -> Self {
        assert_eq!(polys.len(), params.k);
        PolyVec { polys, params }
    }

    /// Create zero polynomial vector
    pub fn zero(params: KyberParams) -> Self {
        let polys = (0..params.k)
            .map(|_| KyberPoly::zero(params))
            .collect();
        PolyVec { polys, params }
    }

    /// Sample polynomial vector with small random coefficients
    pub fn sample_cbd(params: KyberParams, eta: i32, rng: &mut impl rand::Rng) -> Self {
        let polys = (0..params.k)
            .map(|_| KyberPoly::sample_cbd(params, eta, rng))
            .collect();
        PolyVec { polys, params }
    }

    /// Add two polynomial vectors
    pub fn add(&self, other: &Self) -> Self {
        let polys = self
            .polys
            .iter()
            .zip(other.polys.iter())
            .map(|(a, b)| a.add(b))
            .collect();
        PolyVec {
            polys,
            params: self.params,
        }
    }

    /// Reduce all coefficients
    pub fn reduce(&mut self) {
        for poly in &mut self.polys {
            poly.reduce();
        }
    }
}

/// Matrix of k×k polynomials (used for Kyber public key matrix A)
#[derive(Clone)]
pub struct PolyMatrix {
    /// Row-major storage: rows[i] is the i-th row
    pub rows: Vec<PolyVec>,
    pub params: KyberParams,
}

impl PolyMatrix {
    /// Create new polynomial matrix
    pub fn new(rows: Vec<PolyVec>, params: KyberParams) -> Self {
        assert_eq!(rows.len(), params.k);
        PolyMatrix { rows, params }
    }

    /// Create zero polynomial matrix
    pub fn zero(params: KyberParams) -> Self {
        let rows = (0..params.k)
            .map(|_| PolyVec::zero(params))
            .collect();
        PolyMatrix { rows, params }
    }

    /// Generate random matrix (for public key generation)
    pub fn random(params: KyberParams, rng: &mut impl rand::Rng) -> Self {
        let rows = (0..params.k)
            .map(|_| {
                let polys = (0..params.k)
                    .map(|_| KyberPoly::random_small(params, params.q / 2, rng))
                    .collect();
                PolyVec::new(polys, params)
            })
            .collect();
        PolyMatrix { rows, params }
    }

    /// Matrix-vector multiplication: A·s
    /// This is the core operation we want to accelerate with GA
    pub fn mul_vec(&self, vec: &PolyVec) -> PolyVec {
        assert_eq!(self.params.k, vec.polys.len());

        let polys = (0..self.params.k)
            .map(|i| {
                // Compute dot product of row i with vec
                let mut result = KyberPoly::zero(self.params);
                for j in 0..self.params.k {
                    let prod = self.rows[i].polys[j].mul_naive(&vec.polys[j]);
                    result = result.add(&prod);
                }
                result
            })
            .collect();

        PolyVec::new(polys, self.params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_polynomial_creation() {
        let params = KyberParams::KYBER512;
        let coeffs = vec![1; 256];
        let poly = KyberPoly::new(coeffs, params);
        assert_eq!(poly.coeffs.len(), 256);
    }

    #[test]
    fn test_polynomial_addition() {
        let params = KyberParams::KYBER512;
        let a = KyberPoly::new(vec![1; 256], params);
        let b = KyberPoly::new(vec![2; 256], params);
        let c = a.add(&b);
        assert_eq!(c.coeffs[0], 3);
    }

    #[test]
    fn test_polynomial_reduction() {
        let params = KyberParams::KYBER512;
        let coeffs = vec![3330; 256]; // > q
        let mut poly = KyberPoly::new(coeffs, params);
        poly.reduce();
        assert_eq!(poly.coeffs[0], 1); // 3330 mod 3329 = 1
    }

    #[test]
    fn test_mod_positive() {
        assert_eq!(mod_positive(5, 3329), 5);
        assert_eq!(mod_positive(-5, 3329), 3324);
        assert_eq!(mod_positive(3329, 3329), 0);
        assert_eq!(mod_positive(3330, 3329), 1);
    }

    #[test]
    fn test_polynomial_multiplication() {
        let params = KyberParams::KYBER512;

        // Test: (1 + x) * (1 + x) in R_q
        let mut coeffs_a = vec![0; 256];
        coeffs_a[0] = 1; // constant term
        coeffs_a[1] = 1; // x term
        let a = KyberPoly::new(coeffs_a, params);

        let result = a.mul_naive(&a);

        // (1 + x)^2 = 1 + 2x + x^2
        assert_eq!(result.coeffs[0], 1);
        assert_eq!(result.coeffs[1], 2);
        assert_eq!(result.coeffs[2], 1);
    }

    #[test]
    fn test_sample_cbd() {
        let params = KyberParams::KYBER512;
        let mut rng = StdRng::seed_from_u64(42);

        let poly = KyberPoly::sample_cbd(params, 3, &mut rng);

        // Check all coefficients are in expected range [-3, 3]
        for &coeff in &poly.coeffs {
            assert!(coeff >= -3 && coeff <= 3);
        }
    }

    #[test]
    fn test_poly_vec_creation() {
        let params = KyberParams::KYBER512;
        let vec = PolyVec::zero(params);
        assert_eq!(vec.polys.len(), 2); // k=2 for Kyber-512
    }

    #[test]
    fn test_matrix_vec_multiplication() {
        let params = KyberParams::KYBER512;
        let mut rng = StdRng::seed_from_u64(42);

        let matrix = PolyMatrix::random(params, &mut rng);
        let vec = PolyVec::sample_cbd(params, 3, &mut rng);

        let result = matrix.mul_vec(&vec);

        // Result should be a vector of k polynomials
        assert_eq!(result.polys.len(), params.k);
    }
}
