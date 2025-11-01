//! Optimized NTT with precomputed bit-reversal and lazy normalization
//!
//! Key optimizations over base NTT:
//! 1. **Precomputed bit-reversed indices** - Computed once, reused for all transforms
//! 2. **Lazy normalization** - Defer N^(-1) multiplication until final step
//! 3. **Batch operations** - Process multiple components efficiently
//!
//! Expected savings: ~0.5-1 µs total

use crate::ntt::NTTContext;

/// Optimized NTT context with precomputed tables
#[derive(Clone, Debug)]
pub struct OptimizedNTTContext {
    pub base: NTTContext,
    pub bit_rev_indices: Vec<usize>,  // Precomputed bit-reversal permutation
}

impl OptimizedNTTContext {
    /// Create optimized NTT context for Clifford-LWE-256
    pub fn new_clifford_lwe() -> Self {
        let base = NTTContext::new_clifford_lwe();
        let bit_rev_indices = Self::precompute_bit_reversal(base.n);

        Self {
            base,
            bit_rev_indices,
        }
    }

    /// Precompute bit-reversal permutation indices
    ///
    /// Instead of computing this in every NTT call, compute once and reuse.
    /// Saves ~N bit operations per NTT transform.
    fn precompute_bit_reversal(n: usize) -> Vec<usize> {
        let mut indices = vec![0usize; n];

        let mut j = 0;
        for i in 0..n {
            indices[i] = j;

            // Compute next j using bit-reversal algorithm
            let mut bit = n >> 1;
            while j >= bit && bit > 0 {
                j -= bit;
                bit >>= 1;
            }
            j += bit;
        }

        indices
    }

    /// Apply precomputed bit-reversal permutation
    ///
    /// Much faster than computing on-the-fly - just a lookup!
    #[inline]
    fn apply_bit_reversal(&self, a: &mut [i64]) {
        let n = a.len();
        for i in 0..n {
            let j = self.bit_rev_indices[i];
            if i < j {
                a.swap(i, j);
            }
        }
    }

    /// Forward NTT with precomputed bit-reversal (faster than base NTT)
    pub fn forward(&self, a: &mut [i64]) {
        let n = self.base.n;
        let q = self.base.q;
        assert_eq!(a.len(), n);

        // Use precomputed bit-reversal
        self.apply_bit_reversal(a);

        // Cooley-Tukey butterfly (same as base)
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = 2 * n / len;

            for start in (0..n).step_by(len) {
                let mut k = 0;
                for j in start..(start + half) {
                    let u = a[j];
                    let v = (a[j + half] * self.base.psi[k]) % q;

                    a[j] = (u + v) % q;
                    a[j + half] = (u - v + q) % q;

                    k += step;
                }
            }

            len *= 2;
        }
    }

    /// Inverse NTT WITHOUT normalization (for lazy normalization)
    ///
    /// Use this when you'll normalize later (e.g., after processing all components).
    /// Saves N multiplications + N modular reductions per call.
    pub fn inverse_no_normalize(&self, a: &mut [i64]) {
        let n = self.base.n;
        let q = self.base.q;
        assert_eq!(a.len(), n);

        // Use precomputed bit-reversal
        self.apply_bit_reversal(a);

        // Inverse butterfly
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let step = 2 * n / len;

            for start in (0..n).step_by(len) {
                let mut k = 0;
                for j in start..(start + half) {
                    let u = a[j];
                    let v = (a[j + half] * self.base.psi_inv[k]) % q;

                    a[j] = (u + v) % q;
                    a[j + half] = (u - v + q) % q;

                    k += step;
                }
            }

            len *= 2;
        }

        // NO normalization here! Caller must normalize later.
    }

    /// Inverse NTT with normalization (standard version)
    pub fn inverse(&self, a: &mut [i64]) {
        self.inverse_no_normalize(a);

        // Normalize by N^(-1)
        let q = self.base.q;
        for i in 0..a.len() {
            a[i] = (a[i] * self.base.n_inv) % q;
        }
    }

    /// Batch normalize multiple components at once
    ///
    /// Instead of normalizing each component separately in inverse(),
    /// normalize all at once. Slightly better cache locality.
    pub fn batch_normalize(&self, components: &mut [Vec<i64>]) {
        let q = self.base.q;
        let n_inv = self.base.n_inv;

        for component in components.iter_mut() {
            for x in component.iter_mut() {
                *x = (*x * n_inv) % q;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bit_reversal_precomputed() {
        let ntt = OptimizedNTTContext::new_clifford_lwe();

        // Verify bit-reversal indices are correct for N=32
        assert_eq!(ntt.bit_rev_indices[0], 0);
        assert_eq!(ntt.bit_rev_indices[1], 16);
        assert_eq!(ntt.bit_rev_indices[2], 8);
        assert_eq!(ntt.bit_rev_indices[16], 1);

        // Test applying bit-reversal
        let mut a = vec![0i64; 32];
        for i in 0..32 {
            a[i] = i as i64;
        }

        let original = a.clone();
        ntt.apply_bit_reversal(&mut a);

        // Check some known bit-reversal pairs
        assert_eq!(a[0], original[0]);   // 0 -> 0
        assert_eq!(a[1], original[16]);  // 1 -> 16 (bit-reversed)
        assert_eq!(a[16], original[1]);  // 16 -> 1
    }

    #[test]
    fn test_optimized_ntt_matches_base() {
        let opt_ntt = OptimizedNTTContext::new_clifford_lwe();
        let base_ntt = NTTContext::new_clifford_lwe();

        // Test data
        let mut a_opt = vec![1i64, 2, 3, 5, 8, 13, 21, 34];
        a_opt.resize(32, 0);
        let mut a_base = a_opt.clone();

        // Forward NTT
        opt_ntt.forward(&mut a_opt);
        base_ntt.forward(&mut a_base);

        assert_eq!(a_opt, a_base, "Optimized forward NTT should match base");

        // Inverse NTT
        opt_ntt.inverse(&mut a_opt);
        base_ntt.inverse(&mut a_base);

        assert_eq!(a_opt, a_base, "Optimized inverse NTT should match base");
    }

    #[test]
    fn test_lazy_normalization() {
        let ntt = OptimizedNTTContext::new_clifford_lwe();

        let mut a = vec![1i64, 2, 3, 5, 8, 13, 21, 34];
        a.resize(32, 0);
        let original = a.clone();

        // Forward then inverse without normalization
        ntt.forward(&mut a);
        ntt.inverse_no_normalize(&mut a);

        // Should be scaled by N (not normalized yet)
        for i in 0..a.len() {
            let expected = (original[i] * 32) % ntt.base.q;
            assert_eq!(a[i], expected, "Without normalization, result should be scaled by N");
        }

        // Now normalize
        for i in 0..a.len() {
            a[i] = (a[i] * ntt.base.n_inv) % ntt.base.q;
        }

        // Should match original
        assert_eq!(a, original, "After normalization, should recover original");
    }
}
