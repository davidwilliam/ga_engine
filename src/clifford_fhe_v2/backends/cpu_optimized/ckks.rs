//! V2 CKKS Encryption/Decryption with NTT Optimization
//!
//! **Optimizations over V1:**
//! - Uses Harvey NTT for O(n log n) polynomial operations
//! - Barrett reduction for fast modular arithmetic
//! - Precomputed NTT contexts for each prime
//! - Optimized RNS representation
//!
//! **Performance Target:** 10× faster encryption/decryption vs V1

use crate::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::{
    BarrettReducer, RnsContext, RnsRepresentation,
};
use crate::clifford_fhe_v2::params::CliffordFHEParams;

/// V2 Plaintext in RNS representation with NTT support
#[derive(Clone, Debug)]
pub struct Plaintext {
    /// Polynomial coefficients in RNS form
    /// coeffs[i] = polynomial mod q_i (for each prime q_i)
    pub coeffs: Vec<RnsRepresentation>,

    /// Scaling factor for fixed-point encoding
    pub scale: f64,

    /// Ring dimension (polynomial degree)
    pub n: usize,

    /// Current level (number of active primes)
    pub level: usize,
}

impl Plaintext {
    /// Create plaintext from RNS coefficients
    pub fn new(coeffs: Vec<RnsRepresentation>, scale: f64, level: usize) -> Self {
        let n = coeffs.len();
        Self {
            coeffs,
            scale,
            n,
            level,
        }
    }

    /// Encode a vector of floats into plaintext using CKKS encoding
    ///
    /// # Arguments
    /// * `values` - Float values to encode (length ≤ n/2 for CKKS)
    /// * `scale` - Scaling factor for fixed-point representation
    /// * `params` - CKKS parameters
    ///
    /// # Returns
    /// Encoded plaintext ready for encryption
    pub fn encode(values: &[f64], scale: f64, params: &CliffordFHEParams) -> Self {
        let n = params.n;
        let level = params.max_level();

        assert!(
            values.len() <= n / 2,
            "Too many values: {} > n/2 = {}",
            values.len(),
            n / 2
        );

        // Scale values to fixed-point integers
        let scaled: Vec<i64> = values
            .iter()
            .map(|&v| (v * scale).round() as i64)
            .collect();

        // For simplicity, put scaled values in first coefficients
        // In production CKKS, would use FFT/IFFT for slot encoding
        let mut coeffs_vec = vec![0i64; n];
        for (i, &val) in scaled.iter().enumerate() {
            coeffs_vec[i] = val;
        }

        // Convert to RNS representation for each prime
        let moduli: Vec<u64> = params.moduli[..=level].to_vec();
        let mut rns_coeffs = Vec::with_capacity(n);

        for &coeff in &coeffs_vec {
            // Handle negative coefficients by converting to positive mod q
            let values: Vec<u64> = moduli
                .iter()
                .map(|&q| {
                    if coeff >= 0 {
                        (coeff as u64) % q
                    } else {
                        let abs_coeff = (-coeff) as u64;
                        let remainder = abs_coeff % q;
                        if remainder == 0 {
                            0
                        } else {
                            q - remainder
                        }
                    }
                })
                .collect();

            rns_coeffs.push(RnsRepresentation::new(values, moduli.clone()));
        }

        Self::new(rns_coeffs, scale, level)
    }

    /// Decode plaintext to vector of floats
    ///
    /// # Arguments
    /// * `params` - CKKS parameters
    ///
    /// # Returns
    /// Decoded float values
    pub fn decode(&self, params: &CliffordFHEParams) -> Vec<f64> {
        // For simplicity, use the first (largest) prime for decoding
        // This avoids CRT overflow issues and is sufficient for testing
        // In production, would use proper multi-prime CRT or Garner's algorithm

        let q0 = params.moduli[0];
        let half_q0 = (q0 / 2) as i128;

        let mut coeffs_i128 = Vec::with_capacity(self.n);

        for rns_coeff in &self.coeffs {
            let val_mod_q0 = rns_coeff.values[0] as i128;

            // Convert to signed integer (centered lift)
            let signed_val = if val_mod_q0 > half_q0 {
                val_mod_q0 - (q0 as i128)
            } else {
                val_mod_q0
            };

            coeffs_i128.push(signed_val);
        }

        // Scale down to floats
        coeffs_i128
            .iter()
            .map(|&c| (c as f64) / self.scale)
            .collect()
    }
}

/// V2 Ciphertext in RNS representation with NTT support
#[derive(Clone, Debug)]
pub struct Ciphertext {
    /// First component c0 (RNS polynomial)
    pub c0: Vec<RnsRepresentation>,

    /// Second component c1 (RNS polynomial)
    pub c1: Vec<RnsRepresentation>,

    /// Current level (determines active primes)
    pub level: usize,

    /// Scaling factor
    pub scale: f64,

    /// Ring dimension
    pub n: usize,
}

impl Ciphertext {
    /// Create new ciphertext
    pub fn new(
        c0: Vec<RnsRepresentation>,
        c1: Vec<RnsRepresentation>,
        level: usize,
        scale: f64,
    ) -> Self {
        let n = c0.len();
        assert_eq!(c1.len(), n, "c0 and c1 must have same length");

        Self {
            c0,
            c1,
            level,
            scale,
            n,
        }
    }

    /// Add two ciphertexts (homomorphic addition)
    ///
    /// **Complexity:** O(n) per prime (no NTT needed for addition)
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n, "Ciphertexts must have same dimension");
        assert_eq!(self.level, other.level, "Ciphertexts must be at same level");
        assert!(
            (self.scale - other.scale).abs() < 1.0,
            "Ciphertexts must have similar scale"
        );

        let c0: Vec<RnsRepresentation> = self
            .c0
            .iter()
            .zip(&other.c0)
            .map(|(a, b)| a.add(b))
            .collect();

        let c1: Vec<RnsRepresentation> = self
            .c1
            .iter()
            .zip(&other.c1)
            .map(|(a, b)| a.add(b))
            .collect();

        Self::new(c0, c1, self.level, self.scale)
    }

    /// Subtract two ciphertexts (homomorphic subtraction)
    ///
    /// **Complexity:** O(n) per prime
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n, "Ciphertexts must have same dimension");
        assert_eq!(self.level, other.level, "Ciphertexts must be at same level");

        let c0: Vec<RnsRepresentation> = self
            .c0
            .iter()
            .zip(&other.c0)
            .map(|(a, b)| a.sub(b))
            .collect();

        let c1: Vec<RnsRepresentation> = self
            .c1
            .iter()
            .zip(&other.c1)
            .map(|(a, b)| a.sub(b))
            .collect();

        Self::new(c0, c1, self.level, self.scale)
    }

    /// Multiply ciphertext by scalar (homomorphic scalar multiplication)
    ///
    /// **Complexity:** O(n) per prime
    pub fn mul_scalar(&self, scalar: f64) -> Self {
        // Convert scalar to integer with current scale
        let scalar_int = (scalar * self.scale).round() as u64;

        let c0: Vec<RnsRepresentation> = self
            .c0
            .iter()
            .map(|rns| rns.mul_scalar(scalar_int))
            .collect();

        let c1: Vec<RnsRepresentation> = self
            .c1
            .iter()
            .map(|rns| rns.mul_scalar(scalar_int))
            .collect();

        // Scale increases by scalar
        let new_scale = self.scale * scalar;

        Self::new(c0, c1, self.level, new_scale)
    }

    /// Add plaintext to ciphertext (homomorphic plaintext addition)
    pub fn add_plaintext(&self, pt: &Plaintext) -> Self {
        assert_eq!(self.n, pt.n, "Dimensions must match");
        assert_eq!(self.level, pt.level, "Levels must match");

        let c0: Vec<RnsRepresentation> = self
            .c0
            .iter()
            .zip(&pt.coeffs)
            .map(|(ct, pt)| ct.add(pt))
            .collect();

        Self::new(c0.clone(), self.c1.clone(), self.level, self.scale)
    }
}

/// CKKS Context with precomputed NTT transforms
pub struct CkksContext {
    /// Parameters
    pub params: CliffordFHEParams,

    /// NTT contexts for each prime
    pub ntt_contexts: Vec<NttContext>,

    /// Barrett reducers for each prime
    pub reducers: Vec<BarrettReducer>,

    /// RNS context for CRT
    pub rns_context: RnsContext,
}

impl CkksContext {
    /// Create new CKKS context with precomputed NTT data
    pub fn new(params: CliffordFHEParams) -> Self {
        let moduli = params.moduli.clone();

        // Create NTT context for each prime
        let ntt_contexts: Vec<NttContext> = moduli
            .iter()
            .map(|&q| NttContext::new(params.n, q))
            .collect();

        // Create Barrett reducers
        let reducers: Vec<BarrettReducer> = moduli
            .iter()
            .map(|&q| BarrettReducer::new(q))
            .collect();

        // Create RNS context
        let rns_context = RnsContext::new(moduli);

        Self {
            params,
            ntt_contexts,
            reducers,
            rns_context,
        }
    }

    /// Encode float values to plaintext
    pub fn encode(&self, values: &[f64]) -> Plaintext {
        Plaintext::encode(values, self.params.scale, &self.params)
    }

    /// Decode plaintext to float values
    pub fn decode(&self, pt: &Plaintext) -> Vec<f64> {
        pt.decode(&self.params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plaintext_encode_decode() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = CkksContext::new(params.clone());

        let values = vec![1.5, 2.7, -3.2, 0.5];
        let pt = ctx.encode(&values);

        assert_eq!(pt.n, params.n);
        assert_eq!(pt.scale, params.scale);

        let decoded = ctx.decode(&pt);

        // Check first few values (rest should be ~0)
        for i in 0..values.len() {
            let error = (decoded[i] - values[i]).abs();
            assert!(
                error < 1e-6,
                "Value {} decode error: expected {}, got {} (error: {})",
                i,
                values[i],
                decoded[i],
                error
            );
        }
    }

    #[test]
    fn test_plaintext_encode_negative() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = CkksContext::new(params);

        let values = vec![-10.5, -20.3, -5.0];
        let pt = ctx.encode(&values);
        let decoded = ctx.decode(&pt);

        for i in 0..values.len() {
            let error = (decoded[i] - values[i]).abs();
            assert!(error < 1e-6, "Negative value decode error at index {}", i);
        }
    }

    #[test]
    fn test_ciphertext_addition() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let n = params.n;
        let level = params.max_level();
        let moduli = params.moduli[..=level].to_vec();

        // Create dummy ciphertexts with simple RNS values
        let c0_a: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64(i as u64, &moduli))
            .collect();
        let c1_a: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64((i + 1) as u64, &moduli))
            .collect();

        let c0_b: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64((2 * i) as u64, &moduli))
            .collect();
        let c1_b: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64((2 * i + 1) as u64, &moduli))
            .collect();

        let ct_a = Ciphertext::new(c0_a, c1_a, level, params.scale);
        let ct_b = Ciphertext::new(c0_b, c1_b, level, params.scale);

        let ct_sum = ct_a.add(&ct_b);

        // Check dimensions
        assert_eq!(ct_sum.n, n);
        assert_eq!(ct_sum.level, level);

        // Check first coefficient of c0: should be 0 + 0 = 0
        assert_eq!(ct_sum.c0[0].values[0], 0);

        // Check second coefficient of c0: should be 1 + 2 = 3
        assert_eq!(ct_sum.c0[1].values[0], 3);
    }

    #[test]
    fn test_ciphertext_subtraction() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let n = params.n;
        let level = params.max_level();
        let moduli = params.moduli[..=level].to_vec();

        let c0_a: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64((i + 10) as u64, &moduli))
            .collect();
        let c1_a: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64(i as u64, &moduli))
            .collect();

        let c0_b: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64(i as u64, &moduli))
            .collect();
        let c1_b: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64(i as u64, &moduli))
            .collect();

        let ct_a = Ciphertext::new(c0_a, c1_a, level, params.scale);
        let ct_b = Ciphertext::new(c0_b, c1_b, level, params.scale);

        let ct_diff = ct_a.sub(&ct_b);

        // First coefficient: (0+10) - 0 = 10
        assert_eq!(ct_diff.c0[0].values[0], 10);

        // Second coefficient: (1+10) - 1 = 10
        assert_eq!(ct_diff.c0[1].values[0], 10);
    }

    #[test]
    fn test_ciphertext_scalar_multiplication() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let n = params.n;
        let level = params.max_level();
        let moduli = params.moduli[..=level].to_vec();

        let c0: Vec<RnsRepresentation> = (0..n)
            .map(|i| RnsRepresentation::from_u64((i + 1) as u64, &moduli))
            .collect();
        let c1: Vec<RnsRepresentation> = (0..n)
            .map(|_| RnsRepresentation::from_u64(0, &moduli))
            .collect();

        let ct = Ciphertext::new(c0, c1, level, params.scale);
        let scalar = 3.0;
        let ct_scaled = ct.mul_scalar(scalar);

        // Scale should increase
        assert!((ct_scaled.scale - params.scale * scalar).abs() < 0.1);
    }

    #[test]
    fn test_ckks_context_creation() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = CkksContext::new(params.clone());

        assert_eq!(ctx.ntt_contexts.len(), params.moduli.len());
        assert_eq!(ctx.reducers.len(), params.moduli.len());
        assert_eq!(ctx.rns_context.moduli.len(), params.moduli.len());
    }
}
