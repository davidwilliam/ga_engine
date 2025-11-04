//! Metal-accelerated NTT (Number Theoretic Transform)
//!
//! Rust wrappers for Metal NTT compute kernels.

use super::device::MetalDevice;
use metal::*;

/// Metal NTT context with precomputed twiddle factors
pub struct MetalNttContext {
    device: MetalDevice,
    pub(crate) n: usize,
    pub(crate) q: u64,
    root: u64,
    twiddles: Vec<u64>,      // Forward twiddle factors
    twiddles_inv: Vec<u64>,  // Inverse twiddle factors
    n_inv: u64,              // Modular inverse of n
}

impl MetalNttContext {
    /// Create new Metal NTT context
    ///
    /// @param n Polynomial degree (must be power of 2)
    /// @param q NTT-friendly prime (q ≡ 1 mod 2n)
    /// @param root Primitive n-th root of unity mod q
    pub fn new(n: usize, q: u64, root: u64) -> Result<Self, String> {
        // Verify n is power of 2
        if n & (n - 1) != 0 {
            return Err(format!("n must be power of 2, got {}", n));
        }

        let device = MetalDevice::new()?;

        // Precompute twiddle factors for forward NTT
        let mut twiddles = vec![0u64; n];
        twiddles[0] = 1;
        for i in 1..n {
            twiddles[i] = Self::mul_mod(twiddles[i - 1], root, q);
        }

        // Precompute inverse twiddle factors
        let root_inv = Self::mod_inverse(root, q)?;
        let mut twiddles_inv = vec![0u64; n];
        twiddles_inv[0] = 1;
        for i in 1..n {
            twiddles_inv[i] = Self::mul_mod(twiddles_inv[i - 1], root_inv, q);
        }

        // Compute modular inverse of n
        let n_inv = Self::mod_inverse(n as u64, q)?;

        Ok(MetalNttContext {
            device,
            n,
            q,
            root,
            twiddles,
            twiddles_inv,
            n_inv,
        })
    }

    /// Forward NTT on GPU
    ///
    /// Transforms coefficients → evaluation points
    pub fn forward(&self, coeffs: &mut [u64]) -> Result<(), String> {
        if coeffs.len() != self.n {
            return Err(format!("Expected {} coefficients, got {}", self.n, coeffs.len()));
        }

        // Create GPU buffers
        let coeffs_buffer = self.device.create_buffer_with_data(coeffs);
        let twiddles_buffer = self.device.create_buffer_with_data(&self.twiddles);

        // Metal requires scalar parameters as buffers
        let n_buffer = self.device.create_buffer_with_u32_data(&[self.n as u32]);
        let q_buffer = self.device.create_buffer_with_data(&[self.q]);

        // Get forward NTT kernel
        let kernel = self.device.get_function("ntt_forward")?;
        let pipeline = self.device.device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {:?}", e))?;

        // Execute kernel
        self.device.execute_kernel(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&coeffs_buffer), 0);
            encoder.set_buffer(1, Some(&twiddles_buffer), 0);
            encoder.set_buffer(2, Some(&n_buffer), 0);
            encoder.set_buffer(3, Some(&q_buffer), 0);

            // Dispatch threads (one per coefficient)
            let threadgroup_size = MTLSize::new(256, 1, 1);
            let threadgroups = MTLSize::new(
                ((self.n + 255) / 256) as u64,
                1,
                1,
            );

            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
            Ok(())
        })?;

        // Read result back to CPU
        let result = self.device.read_buffer(&coeffs_buffer, self.n);
        coeffs.copy_from_slice(&result);

        Ok(())
    }

    /// Inverse NTT on GPU
    ///
    /// Transforms evaluation points → coefficients
    pub fn inverse(&self, evals: &mut [u64]) -> Result<(), String> {
        if evals.len() != self.n {
            return Err(format!("Expected {} evaluation points, got {}", self.n, evals.len()));
        }

        // Create GPU buffers
        let evals_buffer = self.device.create_buffer_with_data(evals);
        let twiddles_inv_buffer = self.device.create_buffer_with_data(&self.twiddles_inv);

        // Metal requires scalar parameters as buffers
        let n_buffer = self.device.create_buffer_with_u32_data(&[self.n as u32]);
        let q_buffer = self.device.create_buffer_with_data(&[self.q]);
        let n_inv_buffer = self.device.create_buffer_with_data(&[self.n_inv]);

        // Get inverse NTT kernel
        let kernel = self.device.get_function("ntt_inverse")?;
        let pipeline = self.device.device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {:?}", e))?;

        // Execute kernel
        self.device.execute_kernel(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&evals_buffer), 0);
            encoder.set_buffer(1, Some(&twiddles_inv_buffer), 0);
            encoder.set_buffer(2, Some(&n_buffer), 0);
            encoder.set_buffer(3, Some(&q_buffer), 0);
            encoder.set_buffer(4, Some(&n_inv_buffer), 0);

            let threadgroup_size = MTLSize::new(256, 1, 1);
            let threadgroups = MTLSize::new(((self.n + 255) / 256) as u64, 1, 1);

            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
            Ok(())
        })?;

        // Read result
        let result = self.device.read_buffer(&evals_buffer, self.n);
        evals.copy_from_slice(&result);

        Ok(())
    }

    /// Pointwise multiplication in NTT domain (Hadamard product)
    ///
    /// Implements polynomial multiplication: c(x) = a(x) * b(x)
    pub fn pointwise_multiply(&self, a: &[u64], b: &[u64], c: &mut [u64]) -> Result<(), String> {
        if a.len() != self.n || b.len() != self.n || c.len() != self.n {
            return Err("All arrays must have length n".to_string());
        }

        // Create GPU buffers
        let a_buffer = self.device.create_buffer_with_data(a);
        let b_buffer = self.device.create_buffer_with_data(b);
        let c_buffer = self.device.create_buffer(self.n);

        // Metal requires scalar parameters as buffers
        let n_buffer = self.device.create_buffer_with_u32_data(&[self.n as u32]);
        let q_buffer = self.device.create_buffer_with_data(&[self.q]);

        // Get kernel
        let kernel = self.device.get_function("ntt_pointwise_multiply")?;
        let pipeline = self.device.device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {:?}", e))?;

        // Execute
        self.device.execute_kernel(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&a_buffer), 0);
            encoder.set_buffer(1, Some(&b_buffer), 0);
            encoder.set_buffer(2, Some(&c_buffer), 0);
            encoder.set_buffer(3, Some(&n_buffer), 0);
            encoder.set_buffer(4, Some(&q_buffer), 0);

            let threadgroup_size = MTLSize::new(256, 1, 1);
            let threadgroups = MTLSize::new(((self.n + 255) / 256) as u64, 1, 1);
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);

            Ok(())
        })?;

        // Read result
        let result = self.device.read_buffer(&c_buffer, self.n);
        c.copy_from_slice(&result);

        Ok(())
    }

    // Helper functions (same as CPU version)

    fn mul_mod(a: u64, b: u64, q: u64) -> u64 {
        ((a as u128 * b as u128) % q as u128) as u64
    }

    fn pow_mod(mut base: u64, mut exp: u64, q: u64) -> u64 {
        let mut result = 1u64;
        base %= q;

        while exp > 0 {
            if exp & 1 == 1 {
                result = Self::mul_mod(result, base, q);
            }
            base = Self::mul_mod(base, base, q);
            exp >>= 1;
        }

        result
    }

    fn mod_inverse(a: u64, q: u64) -> Result<u64, String> {
        // Extended Euclidean algorithm
        let (mut old_r, mut r) = (a as i128, q as i128);
        let (mut old_s, mut s) = (1i128, 0i128);

        while r != 0 {
            let quotient = old_r / r;
            let temp_r = r;
            r = old_r - quotient * r;
            old_r = temp_r;

            let temp_s = s;
            s = old_s - quotient * s;
            old_s = temp_s;
        }

        if old_r != 1 {
            return Err(format!("{} has no modular inverse mod {}", a, q));
        }

        // Ensure positive result
        let result = if old_s < 0 {
            (old_s + q as i128) as u64
        } else {
            old_s as u64
        };

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_ntt_basic() {
        // Use small NTT-friendly prime for testing
        // q = 97 = 1 + 96 = 1 + 32*3 (so q ≡ 1 mod 2n for n=32)
        // primitive 32nd root: compute 3^(96/32) mod 97 = 3^3 mod 97 = 27
        let n = 32;
        let q = 97u64;
        let root = 27u64; // Primitive 32nd root of unity mod 97

        let ctx = MetalNttContext::new(n, q, root);

        if ctx.is_err() {
            println!("Skipping test: Metal not available (requires Apple Silicon)");
            return;
        }

        let ctx = ctx.unwrap();

        // Test: NTT -> INTT should be identity
        let mut coeffs = vec![1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                              17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
        let original = coeffs.clone();

        ctx.forward(&mut coeffs).unwrap();
        ctx.inverse(&mut coeffs).unwrap();

        // Check all coefficients match (mod q)
        for i in 0..n {
            let diff = if coeffs[i] >= original[i] {
                coeffs[i] - original[i]
            } else {
                original[i] - coeffs[i]
            };
            assert!(diff == 0 || diff == q, "NTT round-trip failed at index {}: {} != {}", i, coeffs[i], original[i]);
        }

        println!("Metal NTT round-trip test passed!");
    }
}
