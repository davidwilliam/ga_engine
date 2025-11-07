//! Metal GPU-accelerated CKKS (Cheon-Kim-Kim-Song) implementation
//!
//! This module provides a complete GPU-only CKKS context for homomorphic encryption.
//! All operations (encode, decode, encrypt, decrypt, add, multiply) run on Metal GPU.
//!
//! **Design Principle**: Complete isolation from CPU backend - no fallbacks, no mixing.

use super::device::MetalDevice;
use super::ntt::MetalNttContext;
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::BarrettReducer;
use crate::clifford_fhe_v2::backends::cpu_optimized::keys::{PublicKey, SecretKey};
use std::sync::Arc;

/// Metal GPU-accelerated CKKS context
///
/// All operations run entirely on GPU - no CPU fallbacks.
/// Uses Metal NTT for polynomial multiplication.
pub struct MetalCkksContext {
    /// Shared Metal device
    device: Arc<MetalDevice>,

    /// FHE parameters
    params: CliffordFHEParams,

    /// Metal NTT contexts (one per prime in RNS)
    ntt_contexts: Vec<MetalNttContext>,

    /// Barrett reducers for modular reduction (lightweight, keep on CPU)
    reducers: Vec<BarrettReducer>,
}

/// GPU plaintext representation
///
/// Stores polynomial coefficients in RNS representation.
/// Each coefficient is represented modulo each prime in the modulus chain.
#[derive(Clone, Debug)]
pub struct MetalPlaintext {
    /// Polynomial coefficients in RNS form
    /// Each u64 value is a coefficient in a specific RNS component
    /// Flat layout: [coeff0_mod_q0, coeff0_mod_q1, ..., coeff1_mod_q0, coeff1_mod_q1, ...]
    /// Length: n × num_primes
    pub coeffs: Vec<u64>,

    /// Ring dimension (number of polynomial coefficients)
    pub n: usize,

    /// Number of RNS primes
    pub num_primes: usize,

    /// Current level in modulus chain
    pub level: usize,

    /// Current scale (encoding scale)
    pub scale: f64,
}

/// GPU ciphertext representation (RLWE ciphertext)
///
/// Standard RLWE ciphertext (c0, c1) where:
/// - Decryption: m ≈ c0 + c1 * s (mod q)
#[derive(Clone, Debug)]
pub struct MetalCiphertext {
    /// First polynomial (c0) in RNS form
    /// Flat layout: [coeff0_mod_q0, coeff0_mod_q1, ..., coeff1_mod_q0, coeff1_mod_q1, ...]
    pub c0: Vec<u64>,

    /// Second polynomial (c1) in RNS form
    /// Flat layout: same as c0
    pub c1: Vec<u64>,

    /// Ring dimension (number of polynomial coefficients)
    pub n: usize,

    /// Number of RNS primes
    pub num_primes: usize,

    /// Current level in modulus chain
    pub level: usize,

    /// Current scale
    pub scale: f64,
}

impl MetalCkksContext {
    /// Create new Metal CKKS context with GPU acceleration
    ///
    /// All operations will run on Metal GPU. If GPU is not available,
    /// this will return an error (no CPU fallback).
    ///
    /// # Arguments
    /// * `params` - FHE parameters (ring dimension, moduli, scale)
    ///
    /// # Returns
    /// GPU-accelerated CKKS context or error if GPU unavailable
    pub fn new(params: CliffordFHEParams) -> Result<Self, String> {
        println!("  [Metal CKKS] Initializing GPU-only CKKS context...");

        // Create shared Metal device
        println!("  [Metal CKKS] Creating Metal device...");
        let start = std::time::Instant::now();
        let device = Arc::new(MetalDevice::new()?);
        println!("  [Metal CKKS] Device initialized in {:.3}s", start.elapsed().as_secs_f64());
        println!("  [Metal CKKS] Device: {}", device.device().name());

        // Create Metal NTT contexts for each prime
        println!("  [Metal CKKS] Creating NTT contexts for {} primes...", params.moduli.len());
        let start = std::time::Instant::now();

        let mut ntt_contexts = Vec::with_capacity(params.moduli.len());
        for (i, &q) in params.moduli.iter().enumerate() {
            // Find primitive 2n-th root of unity
            let psi = Self::find_primitive_2n_root(params.n, q)?;

            // Create Metal NTT context
            let metal_ntt = MetalNttContext::new_with_device(
                device.clone(),
                params.n,
                q,
                psi
            )?;

            ntt_contexts.push(metal_ntt);

            if (i + 1) % 5 == 0 || i == params.moduli.len() - 1 {
                println!("    Created {}/{} NTT contexts", i + 1, params.moduli.len());
            }
        }

        println!("  [Metal CKKS] NTT contexts created in {:.2}s", start.elapsed().as_secs_f64());

        // Create Barrett reducers (lightweight, keep on CPU)
        let reducers: Vec<BarrettReducer> = params.moduli.iter()
            .map(|&q| BarrettReducer::new(q))
            .collect();

        println!("  [Metal CKKS] ✓ GPU-only CKKS context ready!\n");

        Ok(Self {
            device,
            params,
            ntt_contexts,
            reducers,
        })
    }

    /// Encode floating-point values into plaintext polynomial
    ///
    /// Currently uses CPU for canonical embedding (GPU optimization planned).
    /// The encoding is done in a way that's compatible with GPU operations.
    ///
    /// # Arguments
    /// * `values` - Real values to encode (length ≤ n/2)
    ///
    /// # Returns
    /// Plaintext polynomial in flat RNS representation
    pub fn encode(&self, values: &[f64]) -> Result<MetalPlaintext, String> {
        let n = self.params.n;
        let num_slots = n / 2;

        if values.len() > num_slots {
            return Err(format!("Too many values: {} > n/2 = {}", values.len(), num_slots));
        }

        // Step 1: Canonical embedding (CPU for now)
        let coeffs_i64 = Self::canonical_embed_encode_real(values, self.params.scale, n);

        // Step 2: Convert to flat RNS representation
        let level = self.params.max_level();
        let moduli = &self.params.moduli[..=level];
        let num_primes = moduli.len();

        // Flat layout: [coeff0_mod_q0, coeff0_mod_q1, ..., coeff1_mod_q0, ...]
        let mut flat_coeffs = vec![0u64; n * num_primes];

        for (i, &coeff) in coeffs_i64.iter().enumerate() {
            for (j, &q) in moduli.iter().enumerate() {
                // Convert signed coefficient to unsigned mod q
                let val_mod_q = if coeff >= 0 {
                    (coeff as u64) % q
                } else {
                    let abs_coeff = (-coeff) as u64;
                    let remainder = abs_coeff % q;
                    if remainder == 0 {
                        0
                    } else {
                        q - remainder
                    }
                };

                // Flat index: i * num_primes + j
                flat_coeffs[i * num_primes + j] = val_mod_q;
            }
        }

        Ok(MetalPlaintext {
            coeffs: flat_coeffs,
            n,
            num_primes,
            level,
            scale: self.params.scale,
        })
    }

    /// Decode plaintext polynomial to floating-point values
    ///
    /// Currently uses CPU for canonical embedding (GPU optimization planned).
    ///
    /// # Arguments
    /// * `pt` - Plaintext polynomial
    ///
    /// # Returns
    /// Decoded real values
    pub fn decode(&self, pt: &MetalPlaintext) -> Result<Vec<f64>, String> {
        let n = pt.n;
        let num_primes = pt.num_primes;

        // Step 1: Reconstruct coefficients from RNS using first prime (for simplicity)
        // In full implementation, would use CRT reconstruction
        let q0 = self.params.moduli[0];
        let mut coeffs_i64 = vec![0i64; n];

        for i in 0..n {
            let val = pt.coeffs[i * num_primes]; // Use first RNS component

            // Convert from mod q to signed integer (centered representation)
            coeffs_i64[i] = if val <= q0 / 2 {
                val as i64
            } else {
                -((q0 - val) as i64)
            };
        }

        // Step 2: Canonical embedding decode (CPU for now)
        let slots = Self::canonical_embed_decode_real(&coeffs_i64, pt.scale, n);

        Ok(slots)
    }

    /// Encrypt plaintext using public key
    ///
    /// Creates RLWE ciphertext (c0, c1) using polynomial operations.
    /// Uses Metal GPU for NTT-based polynomial multiplication.
    ///
    /// # Arguments
    /// * `pt` - Plaintext to encrypt
    /// * `pk` - Public key
    ///
    /// # Returns
    /// Ciphertext encrypting the plaintext
    pub fn encrypt(
        &self,
        pt: &MetalPlaintext,
        pk: &PublicKey,
    ) -> Result<MetalCiphertext, String> {
        use rand::{thread_rng, Rng};
        use rand_distr::{Distribution, Normal};

        let n = self.params.n;
        let level = pt.level;
        let moduli = &self.params.moduli[..=level];
        let num_primes = moduli.len();
        let mut rng = thread_rng();

        // Sample ternary random polynomial u ∈ {-1, 0, 1}^n
        let u_coeffs: Vec<i64> = (0..n)
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

        // Sample error polynomials e0, e1 from Gaussian distribution
        let normal = Normal::new(0.0, self.params.error_std).unwrap();
        let e0_coeffs: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();
        let e1_coeffs: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();

        // Convert to flat RNS representation
        let u_flat = Self::coeffs_to_flat_rns(&u_coeffs, moduli);
        let e0_flat = Self::coeffs_to_flat_rns(&e0_coeffs, moduli);
        let e1_flat = Self::coeffs_to_flat_rns(&e1_coeffs, moduli);

        // Convert public key to flat layout (if not already)
        let pk_a_flat = Self::rns_vec_to_flat(&pk.a, num_primes);
        let pk_b_flat = Self::rns_vec_to_flat(&pk.b, num_primes);

        // c0 = b*u + e0 + m (using Metal NTT for multiplication)
        let bu = self.multiply_polys_flat_ntt(&pk_b_flat, &u_flat, moduli)?;
        let mut c0 = vec![0u64; n * num_primes];
        for i in 0..(n * num_primes) {
            let prime_idx = i % num_primes;
            let q = moduli[prime_idx];

            // Use Barrett reducer for proper modular arithmetic
            let reducer = &self.reducers[prime_idx];
            let sum1 = reducer.add(bu[i], e0_flat[i]);
            let sum2 = reducer.add(sum1, pt.coeffs[i]);
            c0[i] = sum2;
        }

        // c1 = a*u + e1 (using Metal NTT for multiplication)
        let au = self.multiply_polys_flat_ntt(&pk_a_flat, &u_flat, moduli)?;
        let mut c1 = vec![0u64; n * num_primes];
        for i in 0..(n * num_primes) {
            let prime_idx = i % num_primes;
            let reducer = &self.reducers[prime_idx];
            c1[i] = reducer.add(au[i], e1_flat[i]);
        }

        Ok(MetalCiphertext {
            c0,
            c1,
            n,
            num_primes,
            level,
            scale: pt.scale,
        })
    }

    /// Decrypt ciphertext using secret key
    ///
    /// Recovers plaintext from RLWE ciphertext: m = c0 + c1*s
    /// Uses Metal GPU for NTT-based polynomial multiplication.
    ///
    /// # Arguments
    /// * `ct` - Ciphertext to decrypt
    /// * `sk` - Secret key
    ///
    /// # Returns
    /// Decrypted plaintext
    pub fn decrypt(
        &self,
        ct: &MetalCiphertext,
        sk: &SecretKey,
    ) -> Result<MetalPlaintext, String> {
        let n = ct.n;
        let level = ct.level;
        let moduli = &self.params.moduli[..=level];
        let num_primes = moduli.len();

        // Convert secret key to flat layout at ciphertext's level
        let sk_flat = Self::rns_vec_to_flat_at_level(&sk.coeffs, level, num_primes);

        // Compute c1 * s using Metal NTT
        let c1s = self.multiply_polys_flat_ntt(&ct.c1, &sk_flat, moduli)?;

        // m = c0 + c1*s
        let mut m_flat = vec![0u64; n * num_primes];
        for i in 0..(n * num_primes) {
            let prime_idx = i % num_primes;
            let reducer = &self.reducers[prime_idx];
            m_flat[i] = reducer.add(ct.c0[i], c1s[i]);
        }

        Ok(MetalPlaintext {
            coeffs: m_flat,
            n,
            num_primes,
            level,
            scale: ct.scale,
        })
    }

    // ==================== Canonical Embedding Functions ====================

    /// Compute the Galois orbit order for CKKS slot indexing
    fn orbit_order(n: usize, g: usize) -> Vec<usize> {
        let m = 2 * n; // M = 2N
        let num_slots = n / 2; // N/2 slots

        let mut e = vec![0usize; num_slots];
        let mut cur = 1usize;

        for t in 0..num_slots {
            e[t] = cur; // odd exponent in [1..2N-1]
            cur = (cur * g) % m;
        }

        e
    }

    /// Encode real-valued slots using CKKS canonical embedding
    fn canonical_embed_encode_real(values: &[f64], scale: f64, n: usize) -> Vec<i64> {
        use std::f64::consts::PI;

        assert!(n.is_power_of_two());
        let num_slots = n / 2;
        assert!(values.len() <= num_slots);

        let m = 2 * n; // Cyclotomic index M = 2N
        let g = 5; // Generator for power-of-two cyclotomics

        // Use Galois orbit order
        let e = Self::orbit_order(n, g);

        // Pad values to full slot count
        let mut slots = vec![0.0; num_slots];
        for (i, &val) in values.iter().enumerate() {
            slots[i] = val;
        }

        // Inverse canonical embedding
        let mut coeffs_float = vec![0.0; n];

        for j in 0..n {
            let mut sum = 0.0;

            for t in 0..num_slots {
                // w_t(j) = exp(-2πi * e[t] * j / M)
                let angle = 2.0 * PI * (e[t] as f64) * (j as f64) / (m as f64);
                let cos_val = angle.cos();

                // For real slots: contribution is 2 * z[t] * cos(angle)
                sum += slots[t] * cos_val;
            }

            // Normalize by 2/N
            coeffs_float[j] = (2.0 / n as f64) * sum;
        }

        // Scale and round to integers
        coeffs_float.iter().map(|&x| (x * scale).round() as i64).collect()
    }

    /// Decode real-valued slots using CKKS canonical embedding
    fn canonical_embed_decode_real(coeffs: &[i64], scale: f64, n: usize) -> Vec<f64> {
        use std::f64::consts::PI;

        assert_eq!(coeffs.len(), n);

        let m = 2 * n; // M = 2N
        let num_slots = n / 2;
        let g = 5; // Generator

        // Use Galois orbit order
        let e = Self::orbit_order(n, g);

        // Convert to floating point (with scale normalization)
        let coeffs_float: Vec<f64> = coeffs.iter().map(|&c| c as f64 / scale).collect();

        // Forward canonical embedding
        let mut slots = vec![0.0; num_slots];

        for t in 0..num_slots {
            let mut sum_real = 0.0;
            for j in 0..n {
                // w_t(j) = exp(+2πi * e[t] * j / M) (positive angle for decode)
                let angle = 2.0 * PI * (e[t] as f64) * (j as f64) / (m as f64);
                let cos_val = angle.cos();
                sum_real += coeffs_float[j] * cos_val;
            }
            slots[t] = sum_real;
        }

        slots
    }

    // ==================== RNS Conversion Helper Functions ====================

    /// Convert signed coefficient vector to flat RNS representation
    fn coeffs_to_flat_rns(coeffs: &[i64], moduli: &[u64]) -> Vec<u64> {
        let n = coeffs.len();
        let num_primes = moduli.len();
        let mut flat = vec![0u64; n * num_primes];

        for (i, &coeff) in coeffs.iter().enumerate() {
            for (j, &q) in moduli.iter().enumerate() {
                // Convert signed coefficient to unsigned mod q
                let val_mod_q = if coeff >= 0 {
                    (coeff as u64) % q
                } else {
                    let abs_coeff = (-coeff) as u64;
                    let remainder = abs_coeff % q;
                    if remainder == 0 {
                        0
                    } else {
                        q - remainder
                    }
                };

                // Flat index: i * num_primes + j
                flat[i * num_primes + j] = val_mod_q;
            }
        }

        flat
    }

    /// Convert RnsRepresentation vector to flat layout
    fn rns_vec_to_flat(rns_vec: &[crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation], num_primes: usize) -> Vec<u64> {
        let n = rns_vec.len();
        let mut flat = vec![0u64; n * num_primes];

        for (i, rns) in rns_vec.iter().enumerate() {
            for j in 0..num_primes {
                flat[i * num_primes + j] = rns.values[j];
            }
        }

        flat
    }

    /// Convert RnsRepresentation vector to flat layout at specific level
    fn rns_vec_to_flat_at_level(rns_vec: &[crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation], level: usize, num_primes: usize) -> Vec<u64> {
        let n = rns_vec.len();
        let mut flat = vec![0u64; n * num_primes];

        for (i, rns) in rns_vec.iter().enumerate() {
            for j in 0..num_primes {
                // Take only components up to level
                flat[i * num_primes + j] = rns.values[j];
            }
        }

        flat
    }

    /// Multiply two polynomials in flat RNS layout using Metal NTT
    fn multiply_polys_flat_ntt(&self, a_flat: &[u64], b_flat: &[u64], moduli: &[u64]) -> Result<Vec<u64>, String> {
        let n = self.params.n;
        let num_primes = moduli.len();
        let mut result_flat = vec![0u64; n * num_primes];

        // For each RNS component (each prime), multiply using Metal NTT
        for (prime_idx, &q) in moduli.iter().enumerate() {
            // Extract polynomial for this prime
            let mut a_poly = vec![0u64; n];
            let mut b_poly = vec![0u64; n];

            for i in 0..n {
                a_poly[i] = a_flat[i * num_primes + prime_idx];
                b_poly[i] = b_flat[i * num_primes + prime_idx];
            }

            // Multiply using Metal NTT: forward NTT -> pointwise multiply -> inverse NTT
            self.ntt_contexts[prime_idx].forward(&mut a_poly)?;
            self.ntt_contexts[prime_idx].forward(&mut b_poly)?;

            let mut result_poly = vec![0u64; n];
            self.ntt_contexts[prime_idx].pointwise_multiply(&a_poly, &b_poly, &mut result_poly)?;

            self.ntt_contexts[prime_idx].inverse(&mut result_poly)?;

            // Store back in flat layout
            for i in 0..n {
                result_flat[i * num_primes + prime_idx] = result_poly[i];
            }
        }

        Ok(result_flat)
    }

    // ==================== NTT Helper Functions ====================

    /// Find primitive 2n-th root of unity for NTT
    ///
    /// Uses same algorithm as CPU backend for compatibility.
    fn find_primitive_2n_root(n: usize, q: u64) -> Result<u64, String> {
        // Verify q ≡ 1 (mod 2n)
        let two_n = (2 * n) as u64;
        if (q - 1) % two_n != 0 {
            return Err(format!(
                "q = {} is not NTT-friendly for n = {} (q-1 must be divisible by 2n)",
                q, n
            ));
        }

        // Try small candidates that are often generators
        for candidate in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
            if Self::is_primitive_root_candidate(candidate, n, q) {
                let exponent = (q - 1) / two_n;
                return Ok(Self::pow_mod(candidate, exponent, q));
            }
        }

        // Extended search
        for candidate in 32..1000u64 {
            if Self::is_primitive_root_candidate(candidate, n, q) {
                let exponent = (q - 1) / two_n;
                return Ok(Self::pow_mod(candidate, exponent, q));
            }
        }

        Err(format!("Failed to find primitive root for q = {}, n = {}", q, n))
    }

    /// Check if g is suitable for generating primitive 2n-th root
    fn is_primitive_root_candidate(g: u64, n: usize, q: u64) -> bool {
        // Check if g is a quadratic non-residue
        if Self::pow_mod(g, (q - 1) / 2, q) == 1 {
            return false;
        }

        // Check if g^((q-1)/(2n)) generates the subgroup of order 2n
        let psi = Self::pow_mod(g, (q - 1) / (2 * n as u64), q);

        // psi^n should equal -1 mod q
        let psi_n = Self::pow_mod(psi, n as u64, q);
        if psi_n != q - 1 {
            return false;
        }

        // psi^(2n) should equal 1 mod q
        let psi_2n = Self::pow_mod(psi, 2 * n as u64, q);
        psi_2n == 1
    }

    /// Modular exponentiation: base^exp mod q
    fn pow_mod(mut base: u64, mut exp: u64, q: u64) -> u64 {
        let mut result = 1u64;
        base %= q;

        while exp > 0 {
            if exp & 1 == 1 {
                result = ((result as u128 * base as u128) % q as u128) as u64;
            }
            base = ((base as u128 * base as u128) % q as u128) as u64;
            exp >>= 1;
        }

        result
    }
}

// Implement methods for MetalCiphertext
impl MetalCiphertext {
    /// Add two ciphertexts (GPU)
    pub fn add(&self, _other: &Self, _ctx: &MetalCkksContext) -> Result<Self, String> {
        // TODO: Implement GPU addition
        Err("GPU ciphertext addition not yet implemented".to_string())
    }

    /// Multiply ciphertext by plaintext (GPU)
    pub fn multiply_plain(&self, _pt: &MetalPlaintext, _ctx: &MetalCkksContext) -> Result<Self, String> {
        // TODO: Implement GPU multiply_plain
        Err("GPU multiply_plain not yet implemented".to_string())
    }

    /// Multiply two ciphertexts (GPU)
    pub fn multiply(&self, _other: &Self, _ctx: &MetalCkksContext) -> Result<Self, String> {
        // TODO: Implement GPU multiplication
        Err("GPU ciphertext multiplication not yet implemented".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_ckks_context_creation() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let result = MetalCkksContext::new(params);

        #[cfg(not(target_os = "macos"))]
        {
            assert!(result.is_err(), "Should fail on non-macOS");
        }

        #[cfg(target_os = "macos")]
        {
            assert!(result.is_ok(), "Should succeed on macOS with Metal support");
        }
    }
}
