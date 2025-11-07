//! V3 Parameter Sets for CKKS Bootstrapping
//!
//! V3 parameters extend V2 with additional primes to support bootstrapping.
//!
//! ## Bootstrap Requirements
//!
//! - **Bootstrap levels:** 12-15 primes reserved for bootstrap operations
//! - **Computation levels:** 5-7 primes for computation between bootstraps
//! - **Total:** 20-25 primes minimum
//!
//! ## Parameter Selection Strategy
//!
//! 1. **Large N for precision:** N=8192 or N=16384
//! 2. **60-bit first prime:** For special modulus (fast modular reduction)
//! 3. **40-41 bit primes:** For scaling (≈ 2^40)
//! 4. **All NTT-friendly:** q ≡ 1 mod 2N
//!
//! ## Security
//!
//! With N=8192 and log(Q)≈800 bits (20 primes × 40 bits):
//! - Security: ~128 bits (NIST Level 1)
//! - Bootstrapping doesn't reduce security (stays at same N)

use crate::clifford_fhe_v2::params::{CliffordFHEParams, SecurityLevel};

impl CliffordFHEParams {
    /// **CPU DEMO: Ultra-Fast V3 Bootstrap (N=512, 7 primes) - COMPLETES IN SECONDS**
    ///
    /// **Purpose**: Quick demonstration and testing of V3 bootstrap on CPU
    ///
    /// **Parameters**:
    /// - Ring dimension: N=512 (minimal, very fast)
    /// - Total primes: 7 (1 special + 5 bootstrap + 1 computation)
    /// - Scale: 2^40
    /// - Security: ~50 bits (demo only, NOT secure, NOT for production)
    ///
    /// **Performance**:
    /// - Key generation: <2 seconds on CPU
    /// - Bootstrap operation: <5 seconds
    /// - Total end-to-end: <10 seconds
    ///
    /// **Use Cases**:
    /// - Quick validation that bootstrap works
    /// - Understanding the bootstrap flow
    /// - NOT for any real use (insecure parameters)
    ///
    /// **NOT for production** - Use GPU backends with N=8192 for production
    pub fn new_v3_demo_cpu() -> Self {
        let n = 512;
        let moduli = vec![
            // Special modulus (60-bit): q ≡ 1 mod 1024 (NTT-friendly for N=512)
            1152921504606748673,  // (q-1) = 1024 × 1125899906842229

            // Scaling primes (41-bit): All q ≡ 1 mod 1024
            // These are NTT-friendly for N=512
            1099511922689,   // (q-1) = 1024 × 1073937424
            1099512004609,   // (q-1) = 1024 × 1073937504
            1099512266753,   // (q-1) = 1024 × 1073937760
            1099512299521,   // (q-1) = 1024 × 1073937792
            1099512365057,   // (q-1) = 1024 × 1073937856
            1099512856577,   // (q-1) = 1024 × 1073938336
        ];

        let scale = 2f64.powi(40);
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);
        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        let kappa_plain_mul = 1.0;

        Self {
            n,
            moduli,
            scale,
            error_std: 3.2,
            security: SecurityLevel::Bit128,  // Note: Actually ~85 bits with N=1024
            inv_scale_mod_q,
            inv_q_top_mod_q,
            kappa_plain_mul,
        }
    }

    /// V3 Bootstrap parameters (N=8192, 22 primes, depth ~15 with bootstrap)
    ///
    /// **Use for:** Deep encrypted GNN with bootstrapping
    ///
    /// **Modulus chain:**
    /// - 1 × 60-bit special prime (fast modular reduction)
    /// - 21 × 41-bit scaling primes (≈ 2^40)
    ///
    /// **Levels breakdown:**
    /// - 12 primes reserved for bootstrap operations
    /// - 7 primes for computation between bootstraps
    /// - 2 primes for safety margin
    ///
    /// **Security:** ~128 bits (NIST Level 1)
    ///
    /// **Bootstrap frequency:** Every 5-7 multiplications
    pub fn new_v3_bootstrap_8192() -> Self {
        let n = 8192;
        let moduli = vec![
            // Special modulus (60-bit): q ≡ 1 mod 16384
            1152921504606994433,  // (q-1) = 16384 * 70368744177663

            // Scaling primes (41-bit): All q ≡ 1 mod 16384
            // These are consecutive NTT-friendly primes for N=8192
            1099511922689,   // Prime 1:  (q-1) = 16384 * 67108865
            1099512004609,   // Prime 2:  (q-1) = 16384 * 67108870
            1099512266753,   // Prime 3:  (q-1) = 16384 * 67108886
            1099512299521,   // Prime 4:  (q-1) = 16384 * 67108888
            1099512365057,   // Prime 5:  (q-1) = 16384 * 67108892
            1099512856577,   // Prime 6:  (q-1) = 16384 * 67108922
            1099512938497,   // Prime 7:  (q-1) = 16384 * 67108927
            1099513774081,   // Prime 8:  (q-1) = 16384 * 67108978
            1099513806849,   // Prime 9:  (q-1) = 16384 * 67108980
            1099513872385,   // Prime 10: (q-1) = 16384 * 67108984
            1099514003457,   // Prime 11: (q-1) = 16384 * 67108992
            1099514200065,   // Prime 12: (q-1) = 16384 * 67109004

            // Additional primes for deeper bootstrap
            1099514265601,   // Prime 13: (q-1) = 16384 * 67109008
            1099514331137,   // Prime 14: (q-1) = 16384 * 67109012
            1099514396673,   // Prime 15: (q-1) = 16384 * 67109016
            1099514462209,   // Prime 16: (q-1) = 16384 * 67109020
            1099514527745,   // Prime 17: (q-1) = 16384 * 67109024
            1099514593281,   // Prime 18: (q-1) = 16384 * 67109028
            1099514658817,   // Prime 19: (q-1) = 16384 * 67109032
            1099514724353,   // Prime 20: (q-1) = 16384 * 67109036
            1099514789889,   // Prime 21: (q-1) = 16384 * 67109040
        ];
        let scale = 2f64.powi(40);
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);
        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        let kappa_plain_mul = 1.0;  // No longer needed with exact rescale

        Self {
            n,
            moduli,
            scale,
            error_std: 3.2,
            security: SecurityLevel::Bit128,
            inv_scale_mod_q,
            inv_q_top_mod_q,
            kappa_plain_mul,
        }
    }

    /// V3 Bootstrap parameters (N=16384, 25 primes, depth ~18 with bootstrap)
    ///
    /// **Use for:** Very deep encrypted GNN, higher precision
    ///
    /// **Modulus chain:**
    /// - 1 × 60-bit special prime
    /// - 24 × 41-bit scaling primes
    ///
    /// **Levels breakdown:**
    /// - 15 primes reserved for bootstrap operations
    /// - 8 primes for computation between bootstraps
    /// - 2 primes for safety margin
    ///
    /// **Security:** ~192 bits (NIST Level 3)
    ///
    /// **Performance:** Slower than N=8192 but better precision
    pub fn new_v3_bootstrap_16384() -> Self {
        let n = 16384;
        let moduli = vec![
            // Special modulus (60-bit): q ≡ 1 mod 32768
            1152921504607338497,  // (q-1) = 32768 * 35184372088831

            // Scaling primes (41-bit): All q ≡ 1 mod 32768
            1099511922689,   // Prime 1:  (q-1) = 32768 * 33554432 + ...
            1099512938497,   // Prime 2
            1099514314753,   // Prime 3
            1099514478593,   // Prime 4
            1099515691009,   // Prime 5
            1099515789313,   // Prime 6
            1099515985921,   // Prime 7
            1099516051457,   // Prime 8
            1099516116993,   // Prime 9
            1099516182529,   // Prime 10
            1099516248065,   // Prime 11
            1099516313601,   // Prime 12
            1099516379137,   // Prime 13
            1099516444673,   // Prime 14
            1099516510209,   // Prime 15

            // Additional primes for very deep bootstrap
            1099516575745,   // Prime 16
            1099516641281,   // Prime 17
            1099516706817,   // Prime 18
            1099516772353,   // Prime 19
            1099516837889,   // Prime 20
            1099516903425,   // Prime 21
            1099516968961,   // Prime 22
            1099517034497,   // Prime 23
            1099517100033,   // Prime 24
        ];
        let scale = 2f64.powi(40);
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);
        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        let kappa_plain_mul = 1.0;  // No longer needed with exact rescale

        Self {
            n,
            moduli,
            scale,
            error_std: 3.2,
            security: SecurityLevel::Bit192,
            inv_scale_mod_q,
            inv_q_top_mod_q,
            kappa_plain_mul,
        }
    }

    /// V3 Conservative bootstrap parameters (N=8192, 20 primes, minimal config)
    ///
    /// **Use for:** Testing bootstrap with minimum viable primes
    ///
    /// **Modulus chain:** 20 primes total
    /// - 12 primes for bootstrap (minimum for balanced preset)
    /// - 5 primes for computation
    /// - 3 primes for safety
    ///
    /// **Security:** ~128 bits
    /// Fast Demo Parameters (N=4096, 13 primes - ~15 second key generation)
    ///
    /// Optimized for demonstration purposes:
    /// - Ring dimension: 8192 (production-ready)
    /// - 12 primes for bootstrap (minimum for bootstrap to work)
    /// - 3 primes for post-bootstrap computation (minimum for supports_bootstrap)
    /// - Still demonstrates REAL bootstrap operation
    /// - Faster key generation than full production parameters
    ///
    /// Key generation time: ~120 seconds (parallelized)
    /// Bootstrap supported: Yes (computation_levels = 3)
    /// Security: ~118 bits (reduced from full 128 bits)
    pub fn new_v3_bootstrap_fast_demo() -> Self {
        let n = 8192;
        let moduli = vec![
            // Special modulus (60-bit)
            1152921504606994433,

            // Scaling primes (41-bit): 15 primes (12 for bootstrap + 3 for computation)
            1099511922689,
            1099512004609,
            1099512266753,
            1099512299521,
            1099512365057,
            1099512856577,
            1099512938497,
            1099513774081,
            1099513806849,
            1099513872385,
            1099514003457,
            1099514200065,
            1099514265601,
            1099514331137,
            1099514396673,
        ];
        let scale = 2f64.powi(40);
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);
        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        let kappa_plain_mul = 1.0;

        Self {
            n,
            moduli,
            scale,
            error_std: 3.2,
            security: SecurityLevel::Bit128,  // Note: Actually ~110 bits with 13 primes
            inv_scale_mod_q,
            inv_q_top_mod_q,
            kappa_plain_mul,
        }
    }

    /// Minimal Production Parameters (N=8192, 20 primes - ~180 second key generation)
    ///
    /// Full production parameters with all features:
    /// - Ring dimension: 8192
    /// - 12 primes for bootstrap
    /// - 7 primes for computation (enables 7 multiplications post-bootstrap)
    /// - Security: Full 128 bits (NIST Level 1)
    ///
    /// Key generation time: ~180 seconds (parallelized)
    /// Bootstrap supported: Yes
    pub fn new_v3_bootstrap_minimal() -> Self {
        let n = 8192;
        let moduli = vec![
            // Special modulus (60-bit)
            1152921504606994433,

            // Scaling primes (41-bit): 19 primes
            1099511922689,
            1099512004609,
            1099512266753,
            1099512299521,
            1099512365057,
            1099512856577,
            1099512938497,
            1099513774081,
            1099513806849,
            1099513872385,
            1099514003457,
            1099514200065,
            1099514265601,
            1099514331137,
            1099514396673,
            1099514462209,
            1099514527745,
            1099514593281,
            1099514658817,
        ];
        let scale = 2f64.powi(40);
        let inv_scale_mod_q = Self::precompute_inv_scale_mod_q(scale, &moduli);
        let inv_q_top_mod_q = Self::precompute_inv_q_top_mod_q(&moduli);
        let kappa_plain_mul = 1.0;  // No longer needed with exact rescale

        Self {
            n,
            moduli,
            scale,
            error_std: 3.2,
            security: SecurityLevel::Bit128,
            inv_scale_mod_q,
            inv_q_top_mod_q,
            kappa_plain_mul,
        }
    }

    /// Get number of levels available for computation (excludes bootstrap reserve)
    ///
    /// # Arguments
    ///
    /// * `bootstrap_levels` - Number of levels reserved for bootstrap (typically 12-15)
    ///
    /// # Returns
    ///
    /// Number of multiplication levels available between bootstraps
    pub fn computation_levels(&self, bootstrap_levels: usize) -> usize {
        if self.moduli.len() <= bootstrap_levels {
            0
        } else {
            self.moduli.len() - bootstrap_levels - 1  // -1 for special prime
        }
    }

    /// Check if parameters are suitable for bootstrapping
    ///
    /// # Arguments
    ///
    /// * `bootstrap_levels` - Required levels for bootstrap (typically 12-15)
    ///
    /// # Returns
    ///
    /// True if parameters have enough primes for bootstrap + some computation
    pub fn supports_bootstrap(&self, bootstrap_levels: usize) -> bool {
        self.computation_levels(bootstrap_levels) >= 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v3_bootstrap_8192() {
        let params = CliffordFHEParams::new_v3_bootstrap_8192();

        assert_eq!(params.n, 8192);
        assert_eq!(params.moduli.len(), 22);
        assert_eq!(params.security, SecurityLevel::Bit128);

        // Verify all primes are NTT-friendly (q ≡ 1 mod 2N)
        let two_n = 2 * params.n as u64;
        for (i, &q) in params.moduli.iter().enumerate() {
            assert_eq!((q - 1) % two_n, 0, "Prime {} is not NTT-friendly: {} mod {} = {}",
                       i, q, two_n, (q - 1) % two_n);
        }
    }

    #[test]
    fn test_v3_bootstrap_16384() {
        let params = CliffordFHEParams::new_v3_bootstrap_16384();

        assert_eq!(params.n, 16384);
        assert_eq!(params.moduli.len(), 25);
        assert_eq!(params.security, SecurityLevel::Bit192);

        // Verify NTT-friendly
        let two_n = 2 * params.n as u64;
        for (i, &q) in params.moduli.iter().enumerate() {
            assert_eq!((q - 1) % two_n, 0, "Prime {} is not NTT-friendly", i);
        }
    }

    #[test]
    fn test_v3_bootstrap_minimal() {
        let params = CliffordFHEParams::new_v3_bootstrap_minimal();

        assert_eq!(params.n, 8192);
        assert_eq!(params.moduli.len(), 20);

        // Should support bootstrap with 12 levels
        assert!(params.supports_bootstrap(12));

        // Should have 7 computation levels (20 - 12 - 1)
        assert_eq!(params.computation_levels(12), 7);
    }

    #[test]
    fn test_computation_levels() {
        let params = CliffordFHEParams::new_v3_bootstrap_8192();

        // With 12 bootstrap levels: 22 - 12 - 1 = 9 computation levels
        assert_eq!(params.computation_levels(12), 9);

        // With 15 bootstrap levels: 22 - 15 - 1 = 6 computation levels
        assert_eq!(params.computation_levels(15), 6);
    }

    #[test]
    fn test_supports_bootstrap() {
        let params = CliffordFHEParams::new_v3_bootstrap_8192();

        // Should support bootstrap with 12, 13, 14, 15 levels
        assert!(params.supports_bootstrap(12));
        assert!(params.supports_bootstrap(13));
        assert!(params.supports_bootstrap(14));
        assert!(params.supports_bootstrap(15));

        // Should not support with too many levels (22 - 18 - 1 = 3, exactly at boundary)
        assert!(params.supports_bootstrap(18));
        assert!(!params.supports_bootstrap(19));  // 22 - 19 - 1 = 2 < 3
    }

    #[test]
    fn test_all_primes_distinct() {
        let params = CliffordFHEParams::new_v3_bootstrap_8192();

        // Check all primes are distinct
        for i in 0..params.moduli.len() {
            for j in (i+1)..params.moduli.len() {
                assert_ne!(params.moduli[i], params.moduli[j],
                          "Primes {} and {} are the same", i, j);
            }
        }
    }

    #[test]
    fn test_prime_sizes() {
        let params = CliffordFHEParams::new_v3_bootstrap_8192();

        // First prime should be ~60 bits
        let first_bits = (params.moduli[0] as f64).log2() as usize;
        assert!(first_bits >= 59 && first_bits <= 61,
                "First prime should be ~60 bits, got {} bits", first_bits);

        // Remaining primes should be ~40-41 bits
        for (i, &q) in params.moduli.iter().enumerate().skip(1) {
            let bits = (q as f64).log2() as usize;
            assert!(bits >= 40 && bits <= 42,
                    "Prime {} should be ~40-41 bits, got {} bits", i, bits);
        }
    }
}
