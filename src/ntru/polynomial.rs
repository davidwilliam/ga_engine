//! NTRU Polynomial representation and basic operations

use std::fmt;

/// NTRU system parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NTRUParams {
    /// Polynomial degree (ring dimension)
    pub n: usize,
    /// Small modulus (typically 3)
    pub p: i64,
    /// Large modulus (typically a power of 2)
    pub q: i64,
}

impl NTRUParams {
    /// Standard toy parameters for N=8 (not cryptographically secure, for benchmarking only)
    pub const N8_TOY: NTRUParams = NTRUParams {
        n: 8,
        p: 3,
        q: 128,
    };

    /// Standard toy parameters for N=16 (not cryptographically secure, for benchmarking only)
    pub const N16_TOY: NTRUParams = NTRUParams {
        n: 16,
        p: 3,
        q: 256,
    };

    /// Toy parameters for N=32 (for scaling benchmarks)
    pub const N32_TOY: NTRUParams = NTRUParams {
        n: 32,
        p: 3,
        q: 512,
    };

    /// Toy parameters for N=64 (for scaling benchmarks)
    pub const N64_TOY: NTRUParams = NTRUParams {
        n: 64,
        p: 3,
        q: 1024,
    };

    /// Toy parameters for N=128 (for scaling benchmarks)
    pub const N128_TOY: NTRUParams = NTRUParams {
        n: 128,
        p: 3,
        q: 2048,
    };

    /// Toy parameters for N=256 (for scaling benchmarks)
    pub const N256_TOY: NTRUParams = NTRUParams {
        n: 256,
        p: 3,
        q: 4096,
    };

    /// NIST-level parameters (N=509, secure but too large for direct GA mapping)
    #[allow(dead_code)]
    pub const NIST_LEVEL1: NTRUParams = NTRUParams {
        n: 509,
        p: 3,
        q: 2048,
    };

    /// Validate parameters
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.n == 0 {
            return Err("N must be positive");
        }
        if self.p <= 0 {
            return Err("p must be positive");
        }
        if self.q <= 0 {
            return Err("q must be positive");
        }
        if self.q <= self.p {
            return Err("q must be greater than p");
        }
        // Check that p and q are coprime (gcd = 1)
        if gcd(self.p, self.q) != 1 {
            return Err("p and q must be coprime");
        }
        Ok(())
    }
}

/// Polynomial in Z[x]/(x^N - 1)
///
/// Coefficients are stored in order: a[0] + a[1]*x + a[2]*x^2 + ... + a[N-1]*x^(N-1)
#[derive(Clone, PartialEq, Eq)]
pub struct Polynomial {
    /// Coefficients (length = N)
    pub coeffs: Vec<i64>,
    /// System parameters
    pub params: NTRUParams,
}

impl Polynomial {
    /// Create a new polynomial with given coefficients
    pub fn new(coeffs: Vec<i64>, params: NTRUParams) -> Self {
        assert_eq!(
            coeffs.len(),
            params.n,
            "Polynomial must have exactly N={} coefficients",
            params.n
        );
        Polynomial { coeffs, params }
    }

    /// Create a zero polynomial
    pub fn zero(params: NTRUParams) -> Self {
        Polynomial {
            coeffs: vec![0; params.n],
            params,
        }
    }

    /// Create a random polynomial with coefficients in [-bound, bound]
    pub fn random(params: NTRUParams, bound: i64, rng: &mut impl rand::Rng) -> Self {
        let coeffs = (0..params.n)
            .map(|_| rng.gen_range(-bound..=bound))
            .collect();
        Polynomial { coeffs, params }
    }

    /// Create a polynomial with specific number of +1, -1, and 0 coefficients
    /// This is the standard NTRU polynomial form
    pub fn random_ternary(
        params: NTRUParams,
        num_ones: usize,
        num_neg_ones: usize,
        rng: &mut impl rand::Rng,
    ) -> Self {
        use rand::seq::SliceRandom;

        assert!(
            num_ones + num_neg_ones <= params.n,
            "num_ones + num_neg_ones must be <= N"
        );

        let mut coeffs = vec![0i64; params.n];

        // Fill with +1s
        for coeff in coeffs.iter_mut().take(num_ones) {
            *coeff = 1;
        }

        // Fill with -1s
        for coeff in coeffs.iter_mut().skip(num_ones).take(num_neg_ones) {
            *coeff = -1;
        }

        // Shuffle
        coeffs.shuffle(rng);

        Polynomial { coeffs, params }
    }

    /// Reduce coefficients modulo q
    pub fn mod_q(&self) -> Self {
        let coeffs = self
            .coeffs
            .iter()
            .map(|&c| mod_centered(c, self.params.q))
            .collect();
        Polynomial {
            coeffs,
            params: self.params,
        }
    }

    /// Reduce coefficients modulo p
    pub fn mod_p(&self) -> Self {
        let coeffs = self
            .coeffs
            .iter()
            .map(|&c| mod_centered(c, self.params.p))
            .collect();
        Polynomial {
            coeffs,
            params: self.params,
        }
    }

    /// Get coefficient at index i (with bounds checking)
    pub fn get(&self, i: usize) -> i64 {
        self.coeffs[i]
    }

    /// Set coefficient at index i
    pub fn set(&mut self, i: usize, value: i64) {
        self.coeffs[i] = value;
    }

    /// Degree of the polynomial (index of highest non-zero coefficient)
    pub fn degree(&self) -> Option<usize> {
        self.coeffs
            .iter()
            .enumerate()
            .rev()
            .find(|(_, &c)| c != 0)
            .map(|(i, _)| i)
    }

    /// Check if polynomial is zero
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|&c| c == 0)
    }

    /// Norm (sum of absolute values of coefficients)
    pub fn norm_l1(&self) -> i64 {
        self.coeffs.iter().map(|&c| c.abs()).sum()
    }

    /// Squared norm (sum of squares of coefficients)
    pub fn norm_l2_squared(&self) -> i64 {
        self.coeffs.iter().map(|&c| c * c).sum()
    }
}

impl fmt::Debug for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Polynomial(N={}, coeffs={:?})", self.params.n, self.coeffs)
    }
}

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (i, &coeff) in self.coeffs.iter().enumerate() {
            if coeff == 0 {
                continue;
            }

            if !first {
                if coeff > 0 {
                    write!(f, " + ")?;
                } else {
                    write!(f, " - ")?;
                }
            } else if coeff < 0 {
                write!(f, "-")?;
            }

            let abs_coeff = coeff.abs();
            match i {
                0 => write!(f, "{}", abs_coeff)?,
                1 => {
                    if abs_coeff == 1 {
                        write!(f, "x")?;
                    } else {
                        write!(f, "{}x", abs_coeff)?;
                    }
                }
                _ => {
                    if abs_coeff == 1 {
                        write!(f, "x^{}", i)?;
                    } else {
                        write!(f, "{}x^{}", abs_coeff, i)?;
                    }
                }
            }
            first = false;
        }

        if first {
            write!(f, "0")?;
        }

        Ok(())
    }
}

/// Compute centered modulo: result in [-(m-1)/2, m/2]
#[inline]
pub fn mod_centered(a: i64, m: i64) -> i64 {
    let mut r = a % m;
    // Ensure r is in [0, m)
    if r < 0 {
        r += m;
    }
    // Center it: if r > m/2, subtract m
    if r > m / 2 {
        r - m
    } else {
        r
    }
}

/// Greatest common divisor (for parameter validation)
fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntru_params_validation() {
        assert!(NTRUParams::N8_TOY.validate().is_ok());
        assert!(NTRUParams::N16_TOY.validate().is_ok());

        // Test invalid params
        let invalid = NTRUParams { n: 0, p: 3, q: 128 };
        assert!(invalid.validate().is_err());

        let invalid = NTRUParams { n: 8, p: 0, q: 128 };
        assert!(invalid.validate().is_err());

        let invalid = NTRUParams { n: 8, p: 3, q: 0 };
        assert!(invalid.validate().is_err());

        let invalid = NTRUParams { n: 8, p: 128, q: 3 };
        assert!(invalid.validate().is_err());

        // Test non-coprime p and q
        let invalid = NTRUParams { n: 8, p: 2, q: 128 }; // gcd(2, 128) = 2
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_polynomial_creation() {
        let params = NTRUParams::N8_TOY;
        let coeffs = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let poly = Polynomial::new(coeffs.clone(), params);

        assert_eq!(poly.coeffs, coeffs);
        assert_eq!(poly.params, params);
    }

    #[test]
    fn test_polynomial_zero() {
        let params = NTRUParams::N8_TOY;
        let poly = Polynomial::zero(params);

        assert!(poly.is_zero());
        assert_eq!(poly.degree(), None);
    }

    #[test]
    fn test_mod_centered() {
        assert_eq!(mod_centered(5, 3), -1); // 5 mod 3 = 2, centered = -1
        assert_eq!(mod_centered(4, 3), 1);  // 4 mod 3 = 1
        assert_eq!(mod_centered(3, 3), 0);  // 3 mod 3 = 0
        assert_eq!(mod_centered(-5, 3), 1); // -5 mod 3 = 1 (centered)
        assert_eq!(mod_centered(65, 128), -63); // 65 > 64 so 65 - 128 = -63
        assert_eq!(mod_centered(127, 128), -1);
        assert_eq!(mod_centered(-63, 128), -63); // -63 mod 128 = 65, then 65 > 64 so -63
    }

    #[test]
    fn test_polynomial_modulo() {
        let params = NTRUParams::N8_TOY;
        let coeffs = vec![127, 128, 129, -127, -128, -129, 0, 1];
        let poly = Polynomial::new(coeffs, params);

        let poly_mod_q = poly.mod_q();
        // 127 mod 128 = 127, but centered = -1
        // 128 mod 128 = 0
        // 129 mod 128 = 1
        // -127 mod 128 = 1
        // -128 mod 128 = 0
        // -129 mod 128 = -1
        assert_eq!(poly_mod_q.coeffs, vec![-1, 0, 1, 1, 0, -1, 0, 1]);
    }

    #[test]
    fn test_polynomial_degree() {
        let params = NTRUParams::N8_TOY;

        let poly = Polynomial::new(vec![0, 0, 0, 0, 0, 0, 0, 5], params);
        assert_eq!(poly.degree(), Some(7));

        let poly = Polynomial::new(vec![1, 0, 0, 0, 0, 0, 0, 0], params);
        assert_eq!(poly.degree(), Some(0));

        let poly = Polynomial::zero(params);
        assert_eq!(poly.degree(), None);
    }

    #[test]
    fn test_polynomial_norms() {
        let params = NTRUParams::N8_TOY;
        let poly = Polynomial::new(vec![1, -2, 3, -4, 0, 0, 0, 0], params);

        assert_eq!(poly.norm_l1(), 10); // |1| + |-2| + |3| + |-4| = 10
        assert_eq!(poly.norm_l2_squared(), 30); // 1 + 4 + 9 + 16 = 30
    }
}
