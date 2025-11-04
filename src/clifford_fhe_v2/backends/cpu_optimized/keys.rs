//! V2 Key Generation with NTT Optimization
//!
//! **Optimizations over V1:**
//! - Uses NTT for fast polynomial multiplication during key generation
//! - Barrett reduction for modular arithmetic
//! - Precomputed NTT contexts for efficiency
//!
//! **Performance Target:** 5× faster key generation vs V1

use crate::clifford_fhe_v2::backends::cpu_optimized::ntt::NttContext;
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::{BarrettReducer, RnsRepresentation};
use crate::clifford_fhe_v2::params::CliffordFHEParams;
use rand::distributions::Distribution;
use rand::{thread_rng, Rng};
use rand_distr::Normal;

/// V2 Secret Key
#[derive(Clone, Debug)]
pub struct SecretKey {
    /// Secret polynomial coefficients in RNS form
    /// Ternary polynomial: coefficients in {-1, 0, 1}
    pub coeffs: Vec<RnsRepresentation>,

    /// Ring dimension
    pub n: usize,

    /// Current level
    pub level: usize,
}

impl SecretKey {
    /// Create new secret key
    pub fn new(coeffs: Vec<RnsRepresentation>, level: usize) -> Self {
        let n = coeffs.len();
        Self { coeffs, n, level }
    }

    /// Sample ternary secret key: coefficients in {-1, 0, 1}
    pub fn sample_ternary(params: &CliffordFHEParams) -> Self {
        let n = params.n;
        let level = params.max_level();
        let moduli: Vec<u64> = params.moduli[..=level].to_vec();

        let mut rng = thread_rng();
        let mut coeffs = Vec::with_capacity(n);

        for _ in 0..n {
            // Sample from {-1, 0, 1} with equal probability
            let val: f64 = rng.gen();
            let ternary = if val < 0.33 {
                -1i64
            } else if val < 0.66 {
                0i64
            } else {
                1i64
            };

            // Convert to RNS representation
            let rns_values: Vec<u64> = moduli
                .iter()
                .map(|&q| {
                    if ternary >= 0 {
                        (ternary as u64) % q
                    } else {
                        // -1 mod q = q - 1
                        q - 1
                    }
                })
                .collect();

            coeffs.push(RnsRepresentation::new(rns_values, moduli.clone()));
        }

        Self::new(coeffs, level)
    }
}

/// V2 Public Key
#[derive(Clone, Debug)]
pub struct PublicKey {
    /// First component (uniform random)
    pub a: Vec<RnsRepresentation>,

    /// Second component: b = -a*s - e
    pub b: Vec<RnsRepresentation>,

    /// Ring dimension
    pub n: usize,

    /// Current level
    pub level: usize,
}

impl PublicKey {
    /// Create new public key
    pub fn new(a: Vec<RnsRepresentation>, b: Vec<RnsRepresentation>, level: usize) -> Self {
        let n = a.len();
        assert_eq!(b.len(), n, "a and b must have same length");

        Self { a, b, n, level }
    }
}

/// V2 Evaluation Key (for relinearization)
#[derive(Clone, Debug)]
pub struct EvaluationKey {
    /// Digit width for gadget decomposition (e.g., w=20 means base B=2^20)
    pub base_w: u32,

    /// evk0[t] encrypts B^t · s^2 (one per digit)
    pub evk0: Vec<Vec<RnsRepresentation>>,

    /// evk1[t] = uniform randomness for evk0[t]
    pub evk1: Vec<Vec<RnsRepresentation>>,

    /// Ring dimension
    pub n: usize,

    /// Current level
    pub level: usize,
}

impl EvaluationKey {
    /// Create new evaluation key
    pub fn new(
        base_w: u32,
        evk0: Vec<Vec<RnsRepresentation>>,
        evk1: Vec<Vec<RnsRepresentation>>,
        level: usize,
    ) -> Self {
        let n = evk0[0].len();

        Self {
            base_w,
            evk0,
            evk1,
            n,
            level,
        }
    }
}

/// V2 Key Context with precomputed NTT transforms
pub struct KeyContext {
    /// Parameters
    pub params: CliffordFHEParams,

    /// NTT contexts for each prime
    pub ntt_contexts: Vec<NttContext>,

    /// Barrett reducers for each prime
    pub reducers: Vec<BarrettReducer>,
}

impl KeyContext {
    /// Create new key context
    pub fn new(params: CliffordFHEParams) -> Self {
        let moduli = params.moduli.clone();

        let ntt_contexts: Vec<NttContext> = moduli
            .iter()
            .map(|&q| NttContext::new(params.n, q))
            .collect();

        let reducers: Vec<BarrettReducer> = moduli
            .iter()
            .map(|&q| BarrettReducer::new(q))
            .collect();

        Self {
            params,
            ntt_contexts,
            reducers,
        }
    }

    /// Generate secret, public, and evaluation keys
    ///
    /// # Returns
    /// (PublicKey, SecretKey, EvaluationKey)
    pub fn keygen(&self) -> (PublicKey, SecretKey, EvaluationKey) {
        let n = self.params.n;
        let level = self.params.max_level();
        let moduli: Vec<u64> = self.params.moduli[..=level].to_vec();

        // 1. Sample ternary secret key s ∈ {-1, 0, 1}^N
        let sk = SecretKey::sample_ternary(&self.params);

        // 2. Sample uniform random polynomial a
        let a = self.sample_uniform(&moduli);

        // 3. Sample error polynomial e from Gaussian distribution
        let e = self.sample_error(&moduli);

        // 4. Compute b = -a*s - e using NTT multiplication
        let a_times_s = self.multiply_polynomials(&a, &sk.coeffs, &moduli);
        let neg_a_times_s = self.negate_polynomial(&a_times_s, &moduli);
        let b = self.subtract_polynomials(&neg_a_times_s, &e);

        let pk = PublicKey::new(a, b, level);

        // 5. Generate evaluation key for relinearization
        let evk = self.generate_evaluation_key(&sk, &moduli);

        (pk, sk, evk)
    }

    /// Sample uniform random polynomial from R_q
    fn sample_uniform(&self, moduli: &[u64]) -> Vec<RnsRepresentation> {
        let mut rng = thread_rng();
        let n = self.params.n;

        (0..n)
            .map(|_| {
                let values: Vec<u64> = moduli.iter().map(|&q| rng.gen_range(0..q)).collect();
                RnsRepresentation::new(values, moduli.to_vec())
            })
            .collect()
    }

    /// Sample error polynomial from Gaussian distribution χ
    fn sample_error(&self, moduli: &[u64]) -> Vec<RnsRepresentation> {
        let n = self.params.n;
        let error_std = self.params.error_std;

        let normal = Normal::new(0.0, error_std).expect("Invalid normal distribution parameters");
        let mut rng = thread_rng();

        (0..n)
            .map(|_| {
                // Sample from Gaussian
                let error_val = normal.sample(&mut rng).round() as i64;

                // Convert to RNS representation
                let values: Vec<u64> = moduli
                    .iter()
                    .map(|&q| {
                        if error_val >= 0 {
                            (error_val as u64) % q
                        } else {
                            let abs_val = (-error_val) as u64;
                            let remainder = abs_val % q;
                            if remainder == 0 {
                                0
                            } else {
                                q - remainder
                            }
                        }
                    })
                    .collect();

                RnsRepresentation::new(values, moduli.to_vec())
            })
            .collect()
    }

    /// Multiply two polynomials using NTT (one prime at a time)
    fn multiply_polynomials(
        &self,
        a: &[RnsRepresentation],
        b: &[RnsRepresentation],
        moduli: &[u64],
    ) -> Vec<RnsRepresentation> {
        let n = a.len();
        assert_eq!(b.len(), n);

        let mut result = vec![RnsRepresentation::new(vec![0; moduli.len()], moduli.to_vec()); n];

        // Multiply for each prime separately using NTT
        for (prime_idx, &q) in moduli.iter().enumerate() {
            // Extract coefficients for this prime
            let a_mod_q: Vec<u64> = a.iter().map(|rns| rns.values[prime_idx]).collect();
            let b_mod_q: Vec<u64> = b.iter().map(|rns| rns.values[prime_idx]).collect();

            // Multiply using NTT
            let product_mod_q = self.ntt_contexts[prime_idx].multiply_polynomials(&a_mod_q, &b_mod_q);

            // Store results
            for (i, &val) in product_mod_q.iter().enumerate() {
                result[i].values[prime_idx] = val;
            }
        }

        result
    }

    /// Negate polynomial (compute -a mod q for each coefficient)
    fn negate_polynomial(
        &self,
        a: &[RnsRepresentation],
        moduli: &[u64],
    ) -> Vec<RnsRepresentation> {
        a.iter()
            .map(|rns| {
                let negated_values: Vec<u64> = rns
                    .values
                    .iter()
                    .zip(moduli)
                    .map(|(&val, &q)| if val == 0 { 0 } else { q - val })
                    .collect();
                RnsRepresentation::new(negated_values, moduli.to_vec())
            })
            .collect()
    }

    /// Subtract two polynomials
    fn subtract_polynomials(
        &self,
        a: &[RnsRepresentation],
        b: &[RnsRepresentation],
    ) -> Vec<RnsRepresentation> {
        a.iter().zip(b).map(|(x, y)| x.sub(y)).collect()
    }

    /// Generate evaluation key for relinearization
    fn generate_evaluation_key(&self, sk: &SecretKey, moduli: &[u64]) -> EvaluationKey {
        let base_w = 20u32; // Use base 2^20 for gadget decomposition
        let n = self.params.n;

        // Compute s^2 using NTT
        let s_squared = self.multiply_polynomials(&sk.coeffs, &sk.coeffs, moduli);

        // Determine number of digits needed
        // For Q ~ 2^141 (60+41+41 bits), we need ceil(141/20) = 8 digits
        let q_bits = 141u32; // Approximate total bits
        let num_digits = ((q_bits + base_w - 1) / base_w) as usize;

        let mut evk0 = Vec::with_capacity(num_digits);
        let mut evk1 = Vec::with_capacity(num_digits);

        for t in 0..num_digits {
            // Compute B^t (clamp shift to avoid overflow)
            let shift_amount = (base_w * t as u32).min(63);
            let base_power = 1u64 << shift_amount;

            // Compute B^t * s^2
            let bt_s2: Vec<RnsRepresentation> = s_squared
                .iter()
                .map(|rns| rns.mul_scalar(base_power))
                .collect();

            // Sample uniform a_t
            let a_t = self.sample_uniform(moduli);

            // Sample error e_t
            let e_t = self.sample_error(moduli);

            // Compute b_t = -a_t*s - e_t + B^t*s^2
            let a_t_times_s = self.multiply_polynomials(&a_t, &sk.coeffs, moduli);
            let neg_a_t_s = self.negate_polynomial(&a_t_times_s, moduli);
            let temp = self.subtract_polynomials(&neg_a_t_s, &e_t);
            let b_t: Vec<RnsRepresentation> = temp
                .iter()
                .zip(&bt_s2)
                .map(|(x, y)| x.add(y))
                .collect();

            evk0.push(b_t);
            evk1.push(a_t);
        }

        EvaluationKey::new(base_w, evk0, evk1, sk.level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_key_sampling() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let sk = SecretKey::sample_ternary(&params);

        assert_eq!(sk.n, params.n);
        assert_eq!(sk.level, params.max_level());
        assert_eq!(sk.coeffs.len(), params.n);

        // Check all coefficients are ternary (in {0, 1, q-1} for each prime)
        for rns in &sk.coeffs {
            for (i, &val) in rns.values.iter().enumerate() {
                let q = rns.moduli[i];
                assert!(
                    val == 0 || val == 1 || val == q - 1,
                    "Non-ternary coefficient: {} (q={})",
                    val,
                    q
                );
            }
        }
    }

    #[test]
    fn test_key_context_creation() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = KeyContext::new(params.clone());

        assert_eq!(ctx.ntt_contexts.len(), params.moduli.len());
        assert_eq!(ctx.reducers.len(), params.moduli.len());
    }

    #[test]
    fn test_keygen() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = KeyContext::new(params.clone());

        let (pk, sk, evk) = ctx.keygen();

        // Check dimensions
        assert_eq!(pk.n, params.n);
        assert_eq!(sk.n, params.n);
        assert_eq!(evk.n, params.n);

        // Check levels
        assert_eq!(pk.level, params.max_level());
        assert_eq!(sk.level, params.max_level());
        assert_eq!(evk.level, params.max_level());

        // Check public key has two components
        assert_eq!(pk.a.len(), params.n);
        assert_eq!(pk.b.len(), params.n);

        // Check evaluation key has correct number of digits
        assert!(evk.evk0.len() > 0);
        assert_eq!(evk.evk0.len(), evk.evk1.len());
    }

    #[test]
    fn test_polynomial_multiplication() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = KeyContext::new(params.clone());
        let moduli: Vec<u64> = params.moduli[..=params.max_level()].to_vec();

        // Create simple polynomials for testing
        let mut a = vec![RnsRepresentation::from_u64(0, &moduli); params.n];
        a[0] = RnsRepresentation::from_u64(1, &moduli);

        let mut b = vec![RnsRepresentation::from_u64(0, &moduli); params.n];
        b[0] = RnsRepresentation::from_u64(1, &moduli);

        // Test that multiplication works (a*b should give some result)
        let result = ctx.multiply_polynomials(&a, &b, &moduli);

        // Just verify we got a result with correct dimensions
        assert_eq!(result.len(), params.n);

        // Test commutativity: a*b should equal b*a
        let result_ba = ctx.multiply_polynomials(&b, &a, &moduli);
        for i in 0..params.n {
            assert_eq!(result[i].values[0], result_ba[i].values[0]);
        }
    }

    #[test]
    fn test_negate_polynomial() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let ctx = KeyContext::new(params.clone());
        let moduli: Vec<u64> = params.moduli[..=params.max_level()].to_vec();

        let a = vec![RnsRepresentation::from_u64(5, &moduli); params.n];
        let neg_a = ctx.negate_polynomial(&a, &moduli);

        // For each prime q: -5 mod q = q - 5
        for (i, rns) in neg_a.iter().enumerate() {
            for (j, &val) in rns.values.iter().enumerate() {
                let q = moduli[j];
                assert_eq!(val, q - 5, "Negation failed at index {}, prime {}", i, j);
            }
        }
    }
}
