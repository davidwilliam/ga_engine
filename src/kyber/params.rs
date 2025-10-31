//! Kyber parameters and constants

/// Kyber security parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KyberParams {
    /// Polynomial ring dimension (always 256 for Kyber)
    pub n: usize,
    /// Modulus q = 3329
    pub q: i32,
    /// Matrix/vector dimension k
    pub k: usize,
    /// Secret key coefficient range parameter η₁
    pub eta1: i32,
    /// Error coefficient range parameter η₂
    pub eta2: i32,
}

impl KyberParams {
    /// Kyber-512 parameters (NIST Level 1: AES-128 equivalent)
    /// - k = 2 (2×2 matrix)
    /// - η₁ = 3, η₂ = 2
    /// - Security: ~100 bits classical, ~140 bits quantum
    pub const KYBER512: KyberParams = KyberParams {
        n: 256,
        q: 3329,
        k: 2,
        eta1: 3,
        eta2: 2,
    };

    /// Kyber-768 parameters (NIST Level 3: AES-192 equivalent)
    /// - k = 3 (3×3 matrix)
    /// - η₁ = 2, η₂ = 2
    /// - Security: ~160 bits classical, ~190 bits quantum
    pub const KYBER768: KyberParams = KyberParams {
        n: 256,
        q: 3329,
        k: 3,
        eta1: 2,
        eta2: 2,
    };

    /// Kyber-1024 parameters (NIST Level 5: AES-256 equivalent)
    /// - k = 4 (4×4 matrix)
    /// - η₁ = 2, η₂ = 2
    /// - Security: ~230 bits classical, ~270 bits quantum
    pub const KYBER1024: KyberParams = KyberParams {
        n: 256,
        q: 3329,
        k: 4,
        eta1: 2,
        eta2: 2,
    };

    /// Validate parameters
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.n != 256 {
            return Err("Kyber requires n=256");
        }
        if self.q != 3329 {
            return Err("Kyber requires q=3329");
        }
        if self.k < 2 || self.k > 4 {
            return Err("Kyber requires k in {2, 3, 4}");
        }
        if self.eta1 < 2 || self.eta1 > 3 {
            return Err("η₁ must be 2 or 3");
        }
        if self.eta2 != 2 {
            return Err("η₂ must be 2");
        }
        Ok(())
    }

    /// Total number of polynomials in a k×k matrix
    pub fn matrix_size(&self) -> usize {
        self.k * self.k
    }

    /// Total number of coefficients in a k×k matrix
    pub fn total_coefficients(&self) -> usize {
        self.k * self.k * self.n
    }
}

/// Kyber algorithm constants
pub mod consts {
    /// Modulus q = 3329 = 2^8 · 13 + 1
    /// Chosen so that q ≡ 1 (mod 2n), enabling efficient NTT
    pub const Q: i32 = 3329;

    /// Polynomial degree n = 256
    /// Kyber operates in ring Rq = Zq[x]/(x^256 + 1)
    pub const N: usize = 256;

    /// Primitive 256-th root of unity modulo q
    /// ζ = 17 is used for NTT operations
    pub const ZETA: i32 = 17;

    /// Montgomery R = 2^16 for Montgomery reduction
    pub const MONT_R: i32 = 1 << 16;

    /// q inverse modulo 2^16 for Montgomery reduction
    /// Used in Barrett and Montgomery reduction algorithms
    pub const QINV: i32 = 62209; // q^(-1) mod 2^16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kyber512_params() {
        let params = KyberParams::KYBER512;
        assert_eq!(params.n, 256);
        assert_eq!(params.q, 3329);
        assert_eq!(params.k, 2);
        assert_eq!(params.eta1, 3);
        assert_eq!(params.eta2, 2);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_kyber768_params() {
        let params = KyberParams::KYBER768;
        assert_eq!(params.k, 3);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_kyber1024_params() {
        let params = KyberParams::KYBER1024;
        assert_eq!(params.k, 4);
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_matrix_size() {
        assert_eq!(KyberParams::KYBER512.matrix_size(), 4); // 2×2
        assert_eq!(KyberParams::KYBER768.matrix_size(), 9); // 3×3
        assert_eq!(KyberParams::KYBER1024.matrix_size(), 16); // 4×4
    }

    #[test]
    fn test_invalid_params() {
        let invalid = KyberParams {
            n: 128, // Wrong
            q: 3329,
            k: 2,
            eta1: 3,
            eta2: 2,
        };
        assert!(invalid.validate().is_err());
    }
}
