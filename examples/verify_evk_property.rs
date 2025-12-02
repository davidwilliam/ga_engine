//! Verify EVK fundamental property: evk0[t] - evk1[t]·s ≈ -B^t·s²
//!
//! This test verifies that the evaluation key correctly encrypts -B^t·s².
//! If this property doesn't hold, relinearization cannot work.
//!
//! Run with:
//! ```bash
//! cargo run --release --features v2,v2-gpu-cuda --example verify_evk_property
//! ```

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use ga_engine::clifford_fhe_v2::{
    backends::{
        cpu_optimized::keys::KeyContext,
        gpu_cuda::{
            ckks::CudaCkksContext,
            device::CudaDeviceContext,
            relin_keys::CudaRelinKeys,
        },
    },
    params::CliffordFHEParams,
};

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
use std::sync::Arc;

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn secret_key_to_strided(
    sk: &ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::SecretKey,
    num_primes: usize,
) -> Vec<u64> {
    let n = sk.n;
    let mut strided = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        for prime_idx in 0..num_primes {
            strided[coeff_idx * num_primes + prime_idx] = sk.coeffs[coeff_idx].values[prime_idx];
        }
    }
    strided
}

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn strided_to_flat(strided: &[u64], n: usize, num_primes: usize) -> Vec<u64> {
    let mut flat = vec![0u64; n * num_primes];
    for coeff_idx in 0..n {
        for prime_idx in 0..num_primes {
            let strided_idx = coeff_idx * num_primes + prime_idx;
            let flat_idx = prime_idx * n + coeff_idx;
            flat[flat_idx] = strided[strided_idx];
        }
    }
    flat
}

/// Compute polynomial product a*b mod (X^N + 1) in coefficient domain (CPU reference)
#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn negacyclic_multiply_cpu(a: &[u64], b: &[u64], q: u64, n: usize) -> Vec<u64> {
    let mut result = vec![0u64; n];
    for i in 0..n {
        for j in 0..n {
            let k = i + j;
            let prod = ((a[i] as u128 * b[j] as u128) % q as u128) as u64;
            if k < n {
                result[k] = (result[k] + prod) % q;
            } else {
                // Negacyclic: X^N = -1
                let wrap_idx = k - n;
                result[wrap_idx] = if result[wrap_idx] >= prod {
                    result[wrap_idx] - prod
                } else {
                    q - (prod - result[wrap_idx])
                };
            }
        }
    }
    result
}

#[cfg(all(feature = "v2", feature = "v2-gpu-cuda"))]
fn main() -> Result<(), String> {
    println!("╔════════════════════════════════════════════════════════════════════════╗");
    println!("║     VERIFY EVK PROPERTY: evk0[t] - evk1[t]·s ≈ -B^t·s²                 ║");
    println!("╚════════════════════════════════════════════════════════════════════════╝\n");

    let params = CliffordFHEParams::new_test_ntt_1024();
    let n = params.n;
    let num_primes = params.moduli.len();
    let base_bits = 16u32;
    let base_w = 1u64 << base_bits;

    println!("Parameters: N={}, num_primes={}, base_w=2^{}={}\n", n, num_primes, base_bits, base_w);

    // Generate keys
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let sk_strided = secret_key_to_strided(&sk, num_primes);
    let sk_flat = strided_to_flat(&sk_strided, n, num_primes);

    // Create CUDA context
    let device = Arc::new(CudaDeviceContext::new()?);
    let ctx = CudaCkksContext::new(params.clone())?;

    // Generate CUDA relin keys
    let relin_keys = CudaRelinKeys::new_gpu(
        device.clone(),
        params.clone(),
        sk_strided.clone(),
        base_bits as usize,
        ctx.ntt_contexts(),
    )?;

    let relin_key = relin_keys.get_relin_key();
    println!("EVK has {} components\n", relin_key.ks_components.len());

    // Compute s² on CPU for reference
    println!("Computing s² on CPU for reference...");
    let mut s_squared_flat = vec![0u64; n * num_primes];
    for prime_idx in 0..num_primes {
        let q = params.moduli[prime_idx];
        let offset = prime_idx * n;
        let s_prime: Vec<u64> = sk_flat[offset..offset+n].to_vec();
        let s2_prime = negacyclic_multiply_cpu(&s_prime, &s_prime, q, n);
        s_squared_flat[offset..offset+n].copy_from_slice(&s2_prime);
    }
    println!("  s²[0] for prime 0: {}\n", s_squared_flat[0]);

    // Verify property for digit 0
    println!("════════════════════════════════════════════════════════════════════════");
    println!("VERIFY: evk0[0] - evk1[0]·s = -B^0·s² = -s² (digit 0, B^0 = 1)");
    println!("════════════════════════════════════════════════════════════════════════\n");

    let (evk0_0, evk1_0) = &relin_key.ks_components[0];

    println!("  evk0[0] len: {} (expected {})", evk0_0.len(), n * num_primes);
    println!("  evk1[0] len: {} (expected {})", evk1_0.len(), n * num_primes);
    println!("  evk0[0][coeff=0, prime=0]: {}", evk0_0[0]);
    println!("  evk1[0][coeff=0, prime=0]: {}", evk1_0[0]);

    // Compute evk1[0] · s for prime 0
    println!("\n  Computing evk1[0] · s...");
    let q0 = params.moduli[0];
    let evk1_0_prime0: Vec<u64> = evk1_0[0..n].to_vec();
    let s_prime0: Vec<u64> = sk_flat[0..n].to_vec();
    let evk1_s_prime0 = negacyclic_multiply_cpu(&evk1_0_prime0, &s_prime0, q0, n);
    println!("  (evk1[0] · s)[coeff=0, prime=0]: {}", evk1_s_prime0[0]);

    // Compute evk0[0] - evk1[0]·s
    println!("\n  Computing evk0[0] - evk1[0]·s...");
    let mut diff_prime0 = vec![0u64; n];
    for i in 0..n {
        diff_prime0[i] = if evk0_0[i] >= evk1_s_prime0[i] {
            evk0_0[i] - evk1_s_prime0[i]
        } else {
            q0 - (evk1_s_prime0[i] - evk0_0[i])
        };
    }
    println!("  (evk0[0] - evk1[0]·s)[coeff=0, prime=0]: {}", diff_prime0[0]);

    // Expected: -s² mod q0
    let neg_s2_prime0: Vec<u64> = s_squared_flat[0..n].iter()
        .map(|&x| if x == 0 { 0 } else { q0 - x })
        .collect();
    println!("  Expected -s²[coeff=0, prime=0]: {}", neg_s2_prime0[0]);

    // Compare
    let mut match_count = 0;
    let mut diff_count = 0;
    let mut max_error = 0u64;
    for i in 0..n {
        let actual = diff_prime0[i];
        let expected = neg_s2_prime0[i];
        // Allow small error from noise
        let error = if actual >= expected { actual - expected } else { expected - actual };
        if error < 1000000 {  // Allow small noise
            match_count += 1;
        } else {
            diff_count += 1;
            if error > max_error {
                max_error = error;
            }
            if diff_count <= 5 {
                println!("  DIFF at coeff[{}]: actual={}, expected={}, error={}", i, actual, expected, error);
            }
        }
    }

    println!("\n  Summary: {} matches, {} significant differences, max_error={}",
             match_count, diff_count, max_error);

    if diff_count == 0 {
        println!("  ✓ EVK property VERIFIED for digit 0!");
    } else {
        println!("  ✗ EVK property FAILED for digit 0!");
        println!("  This means the EVK is incorrectly generated.");
    }

    // Also verify digit 1: evk0[1] - evk1[1]·s = -B^1·s² = -B·s²
    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("VERIFY: evk0[1] - evk1[1]·s = -B^1·s² = -{}·s² (digit 1)", base_w);
    println!("════════════════════════════════════════════════════════════════════════\n");

    let (evk0_1, evk1_1) = &relin_key.ks_components[1];

    // Compute evk1[1] · s for prime 0
    let evk1_1_prime0: Vec<u64> = evk1_1[0..n].to_vec();
    let evk1_1_s_prime0 = negacyclic_multiply_cpu(&evk1_1_prime0, &s_prime0, q0, n);

    // Compute evk0[1] - evk1[1]·s
    let mut diff1_prime0 = vec![0u64; n];
    for i in 0..n {
        diff1_prime0[i] = if evk0_1[i] >= evk1_1_s_prime0[i] {
            evk0_1[i] - evk1_1_s_prime0[i]
        } else {
            q0 - (evk1_1_s_prime0[i] - evk0_1[i])
        };
    }

    // Expected: -B·s² mod q0
    let neg_b_s2_prime0: Vec<u64> = s_squared_flat[0..n].iter()
        .map(|&x| {
            let bx = ((base_w as u128 * x as u128) % q0 as u128) as u64;
            if bx == 0 { 0 } else { q0 - bx }
        })
        .collect();

    println!("  (evk0[1] - evk1[1]·s)[coeff=0]: {}", diff1_prime0[0]);
    println!("  Expected -B·s²[coeff=0]: {}", neg_b_s2_prime0[0]);

    let mut match_count1 = 0;
    let mut diff_count1 = 0;
    for i in 0..n {
        let error = if diff1_prime0[i] >= neg_b_s2_prime0[i] {
            diff1_prime0[i] - neg_b_s2_prime0[i]
        } else {
            neg_b_s2_prime0[i] - diff1_prime0[i]
        };
        if error < 1000000 {
            match_count1 += 1;
        } else {
            diff_count1 += 1;
        }
    }

    println!("  Summary: {} matches, {} significant differences", match_count1, diff_count1);

    if diff_count1 == 0 {
        println!("  ✓ EVK property VERIFIED for digit 1!");
    } else {
        println!("  ✗ EVK property FAILED for digit 1!");
    }

    println!("\n════════════════════════════════════════════════════════════════════════");
    println!("DONE");
    println!("════════════════════════════════════════════════════════════════════════\n");

    Ok(())
}

#[cfg(not(all(feature = "v2", feature = "v2-gpu-cuda")))]
fn main() {
    println!("This example requires the 'v2' and 'v2-gpu-cuda' features.");
    println!("Run with: cargo run --release --features v2,v2-gpu-cuda --example verify_evk_property");
}
