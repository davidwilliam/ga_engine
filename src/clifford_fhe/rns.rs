//! Residue Number System (RNS) for CKKS
//!
//! RNS representation allows working with large moduli by using the Chinese Remainder Theorem.
//! Instead of storing coefficients modulo a single large Q, we store them as tuples of
//! residues modulo smaller primes: (c mod q₀, c mod q₁, ..., c mod qₗ)
//!
//! Key benefits:
//! - Support moduli Q > 2^63 (product of primes)
//! - Efficient rescaling (drop one prime)
//! - Parallelizable operations (per-prime)

/// RNS polynomial: coefficients in residue representation
///
/// Each coefficient is represented as a vector of residues mod each prime in the chain.
/// For example, with primes [q₀, q₁, q₂] and coefficient c:
///   rns_coeffs[i] = [c mod q₀, c mod q₁, c mod q₂]
#[derive(Clone, Debug)]
pub struct RnsPolynomial {
    /// Coefficients in RNS form
    /// Outer vector: polynomial coefficients (length N)
    /// Inner vector: residues for each prime (length L = number of primes at this level)
    pub rns_coeffs: Vec<Vec<i64>>,

    /// Number of coefficients (polynomial degree)
    pub n: usize,

    /// Current level (determines which primes are active)
    /// Level 0: all primes [q₀, q₁, ..., qₗ]
    /// Level 1: dropped last prime [q₀, q₁, ..., qₗ₋₁]
    pub level: usize,
}

impl RnsPolynomial {
    /// Create new RNS polynomial from RNS coefficients
    pub fn new(rns_coeffs: Vec<Vec<i64>>, n: usize, level: usize) -> Self {
        assert_eq!(rns_coeffs.len(), n, "Wrong number of coefficients");
        Self { rns_coeffs, n, level }
    }

    /// Create RNS polynomial from regular coefficients
    ///
    /// Converts coefficients [c₀, c₁, ..., cₙ₋₁] to RNS form:
    /// rns_coeffs[i] = [cᵢ mod q₀, cᵢ mod q₁, ...]
    pub fn from_coeffs(coeffs: &[i64], primes: &[i64], n: usize, level: usize) -> Self {
        assert_eq!(coeffs.len(), n, "Wrong number of coefficients");

        let num_primes = primes.len() - level; // Active primes at this level
        let mut rns_coeffs = vec![vec![0i64; num_primes]; n];

        for i in 0..n {
            for (j, &q) in primes.iter().take(num_primes).enumerate() {
                rns_coeffs[i][j] = ((coeffs[i] % q) + q) % q;
            }
        }

        Self::new(rns_coeffs, n, level)
    }

    /// Convert RNS polynomial back to regular coefficients using CRT
    ///
    /// Uses Chinese Remainder Theorem to reconstruct coefficients from residues.
    /// For two primes q₀, q₁: c = (c₀·Q₀·Q₀⁻¹ + c₁·Q₁·Q₁⁻¹) mod Q
    /// where Q = q₀·q₁, Q₀ = Q/q₀, Q₁ = Q/q₁
    pub fn to_coeffs(&self, primes: &[i64]) -> Vec<i64> {
        let num_primes = self.rns_coeffs[0].len();
        let active_primes = &primes[..num_primes];

        // Compute Q = product of all active primes
        let q_product: i128 = active_primes.iter().map(|&q| q as i128).product();

        let mut coeffs = vec![0i64; self.n];

        for i in 0..self.n {
            let mut c: i128 = 0;

            for (j, &qj) in active_primes.iter().enumerate() {
                // Q_j = Q / q_j
                let q_j = q_product / (qj as i128);

                // Q_j^{-1} mod q_j (using extended Euclidean algorithm)
                let q_j_inv = mod_inverse(q_j, qj as i128);

                // Contribution: c_j * Q_j * Q_j^{-1}
                let contrib = (self.rns_coeffs[i][j] as i128) * q_j % q_product;
                let contrib = contrib * q_j_inv % q_product;

                c = (c + contrib) % q_product;
            }

            // Center around 0: map from [0, Q) to (-Q/2, Q/2]
            if c > q_product / 2 {
                c -= q_product;
            }

            coeffs[i] = c as i64;
        }

        coeffs
    }

    /// Number of active primes at this level
    pub fn num_primes(&self) -> usize {
        self.rns_coeffs[0].len()
    }
}

/// Modular inverse using extended Euclidean algorithm
///
/// Returns a^{-1} mod m such that (a * a^{-1}) ≡ 1 (mod m)
fn mod_inverse(a: i128, m: i128) -> i128 {
    let (mut t, mut new_t) = (0i128, 1i128);
    let (mut r, mut new_r) = (m, a % m);

    while new_r != 0 {
        let quotient = r / new_r;

        let temp = new_t;
        new_t = t - quotient * new_t;
        t = temp;

        let temp = new_r;
        new_r = r - quotient * new_r;
        r = temp;
    }

    if r > 1 {
        panic!("a is not invertible mod m");
    }
    if t < 0 {
        t += m;
    }

    t
}

/// RNS polynomial addition
///
/// (a + b) mod Q = [(a₀ + b₀) mod q₀, (a₁ + b₁) mod q₁, ...]
pub fn rns_add(a: &RnsPolynomial, b: &RnsPolynomial, primes: &[i64]) -> RnsPolynomial {
    assert_eq!(a.n, b.n, "Polynomials must have same length");
    assert_eq!(a.level, b.level, "Polynomials must be at same level");

    let n = a.n;
    let level = a.level;
    let num_primes = a.num_primes();
    let active_primes = &primes[..num_primes];

    let mut rns_coeffs = vec![vec![0i64; num_primes]; n];

    for i in 0..n {
        for (j, &q) in active_primes.iter().enumerate() {
            let sum = a.rns_coeffs[i][j] + b.rns_coeffs[i][j];
            rns_coeffs[i][j] = ((sum % q) + q) % q;
        }
    }

    RnsPolynomial::new(rns_coeffs, n, level)
}

/// RNS polynomial subtraction
///
/// (a - b) mod Q = [(a₀ - b₀) mod q₀, (a₁ - b₁) mod q₁, ...]
pub fn rns_sub(a: &RnsPolynomial, b: &RnsPolynomial, primes: &[i64]) -> RnsPolynomial {
    assert_eq!(a.n, b.n, "Polynomials must have same length");
    assert_eq!(a.level, b.level, "Polynomials must be at same level");

    let n = a.n;
    let level = a.level;
    let num_primes = a.num_primes();
    let active_primes = &primes[..num_primes];

    let mut rns_coeffs = vec![vec![0i64; num_primes]; n];

    for i in 0..n {
        for (j, &q) in active_primes.iter().enumerate() {
            let diff = a.rns_coeffs[i][j] - b.rns_coeffs[i][j];
            rns_coeffs[i][j] = ((diff % q) + q) % q;
        }
    }

    RnsPolynomial::new(rns_coeffs, n, level)
}

/// RNS polynomial negation
///
/// (-a) mod Q = [(-a₀) mod q₀, (-a₁) mod q₁, ...]
pub fn rns_negate(a: &RnsPolynomial, primes: &[i64]) -> RnsPolynomial {
    let n = a.n;
    let level = a.level;
    let num_primes = a.num_primes();
    let active_primes = &primes[..num_primes];

    let mut rns_coeffs = vec![vec![0i64; num_primes]; n];

    for i in 0..n {
        for (j, &q) in active_primes.iter().enumerate() {
            rns_coeffs[i][j] = ((q - a.rns_coeffs[i][j]) % q + q) % q;
        }
    }

    RnsPolynomial::new(rns_coeffs, n, level)
}

/// RNS polynomial multiplication
///
/// (a * b) mod Q = [(a₀ * b₀) mod q₀, (a₁ * b₁) mod q₁, ...]
///
/// Each multiplication is done independently modulo each prime using NTT.
/// This is the key advantage of RNS: parallelizable operations!
pub fn rns_multiply(
    a: &RnsPolynomial,
    b: &RnsPolynomial,
    primes: &[i64],
    ntt_multiply_fn: impl Fn(&[i64], &[i64], i64, usize) -> Vec<i64>,
) -> RnsPolynomial {
    assert_eq!(a.n, b.n, "Polynomials must have same length");
    assert_eq!(a.level, b.level, "Polynomials must be at same level");

    let n = a.n;
    let level = a.level;
    let num_primes = a.num_primes();
    let active_primes = &primes[..num_primes];

    let mut rns_coeffs = vec![vec![0i64; num_primes]; n];

    // Multiply modulo each prime independently
    for (j, &q) in active_primes.iter().enumerate() {
        // Extract coefficients for this prime
        let a_mod_q: Vec<i64> = (0..n).map(|i| a.rns_coeffs[i][j]).collect();
        let b_mod_q: Vec<i64> = (0..n).map(|i| b.rns_coeffs[i][j]).collect();

        // Multiply using NTT
        let c_mod_q = ntt_multiply_fn(&a_mod_q, &b_mod_q, q, n);

        // Store result
        for i in 0..n {
            rns_coeffs[i][j] = c_mod_q[i];
        }
    }

    RnsPolynomial::new(rns_coeffs, n, level)
}

/// RNS rescaling: drop the last prime from the modulus chain
///
/// This is how CKKS rescaling works in RNS:
/// - Before: coefficients mod Q = q₀ · q₁ · ... · qₗ
/// - After: coefficients mod Q' = q₀ · q₁ · ... · qₗ₋₁ (dropped qₗ)
///
/// The rescaling process:
/// 1. Convert RNS to regular coefficients (CRT)
/// 2. Divide by the last prime qₗ (with rounding)
/// 3. Convert back to RNS with one fewer prime
///
/// Returns new polynomial at level+1 with scale divided by qₗ
pub fn rns_rescale(poly: &RnsPolynomial, primes: &[i64]) -> RnsPolynomial {
    let n = poly.n;
    let level = poly.level;
    let num_primes = poly.num_primes();

    assert!(num_primes > 1, "Cannot rescale: only one prime remaining");

    // Get the prime we're dropping (last one)
    let q_last = primes[num_primes - 1];

    // New polynomial will have one fewer prime
    let new_num_primes = num_primes - 1;
    let mut new_rns_coeffs = vec![vec![0i64; new_num_primes]; n];

    // For each coefficient, perform rescaling
    for i in 0..n {
        // Get residue modulo q_last
        let c_mod_qlast = poly.rns_coeffs[i][num_primes - 1];

        // For each remaining prime, compute new residue
        for j in 0..new_num_primes {
            let qj = primes[j];
            let c_mod_qj = poly.rns_coeffs[i][j];

            // Compute (c - c_last) / q_last mod qj
            // where c ≡ c_mod_qj (mod qj) and c ≡ c_mod_qlast (mod q_last)
            //
            // Since we're dropping q_last, we need:
            // c_new = round(c / q_last)
            //
            // In RNS: c_new mod qj = (c mod qj - c mod q_last) * (q_last^{-1} mod qj) mod qj
            //
            // But c mod q_last is in [0, q_last), need to lift to consistent value
            // Use approximation: c_new ≈ (c_mod_qj - c_mod_qlast) / q_last (all mod qj)

            // Compute q_last^{-1} mod qj
            let qlast_inv = mod_inverse(q_last as i128, qj as i128) as i64;

            // Compute (c_mod_qj - c_mod_qlast) mod qj
            let diff = ((c_mod_qj - c_mod_qlast % qj) % qj + qj) % qj;

            // Multiply by q_last^{-1}
            let c_new = (diff * qlast_inv % qj + qj) % qj;

            new_rns_coeffs[i][j] = c_new;
        }
    }

    RnsPolynomial::new(new_rns_coeffs, n, level + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rns_conversion() {
        let primes = vec![1_099_511_627_689, 1_099_511_627_691]; // Two 40-bit primes
        let coeffs = vec![123456789, -987654321, 0, 42];
        let n = 4;

        let rns_poly = RnsPolynomial::from_coeffs(&coeffs, &primes, n, 0);
        let recovered = rns_poly.to_coeffs(&primes);

        for i in 0..n {
            assert_eq!(coeffs[i], recovered[i], "Coefficient {} mismatch", i);
        }
    }

    #[test]
    fn test_rns_add() {
        let primes = vec![1_099_511_627_689, 1_099_511_627_691];
        let a_coeffs = vec![100, 200, 300, 400];
        let b_coeffs = vec![50, 60, 70, 80];
        let n = 4;

        let a = RnsPolynomial::from_coeffs(&a_coeffs, &primes, n, 0);
        let b = RnsPolynomial::from_coeffs(&b_coeffs, &primes, n, 0);

        let c = rns_add(&a, &b, &primes);
        let c_coeffs = c.to_coeffs(&primes);

        for i in 0..n {
            assert_eq!(a_coeffs[i] + b_coeffs[i], c_coeffs[i]);
        }
    }

    #[test]
    fn test_mod_inverse() {
        // Test: 3 * 3^{-1} ≡ 1 (mod 7)
        let inv = mod_inverse(3, 7);
        assert_eq!((3 * inv) % 7, 1);

        // Test with larger values
        let q = 1_099_511_627_689i128;
        let a = 12345678i128;
        let inv = mod_inverse(a, q);
        assert_eq!((a * inv) % q, 1);
    }
}
