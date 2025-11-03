// Minimal test with 60-bit primes and NTT-based multiplication

use ga_engine::clifford_fhe::{
    ckks_rns::{rns_encrypt, rns_decrypt, RnsPlaintext},
    keys_rns::rns_keygen,
    params::CliffordFHEParams,
};

fn main() {
    // Use ONLY 1 prime to isolate the issue
    let params = CliffordFHEParams {
        n: 1024,
        moduli: vec![1141392289560813569], // Single 60-bit prime
        scale: 2f64.powi(40),
        error_std: 3.2,
        security: ga_engine::clifford_fhe::params::SecurityLevel::Bit128,
    };

    println!("Testing with SINGLE 60-bit prime:");
    println!("q = {}", params.moduli[0]);
    println!("scale = {}", params.scale);
    println!("n = {}", params.n);

    let (pk, sk, _evk) = rns_keygen(&params);

    // TEST 1: Encrypt ZERO and see if we get small noise back
    println!("\n=== TEST 1: Encrypt zero ===");
    let coeffs_zero = vec![0i64; params.n];
    let pt_zero = RnsPlaintext::from_coeffs(coeffs_zero.clone(), params.scale, &params.moduli, 0);
    let ct_zero = rns_encrypt(&pk, &pt_zero, &params);
    let pt_dec_zero = rns_decrypt(&sk, &ct_zero, &params);

    let q = params.moduli[0];

    // Center the values in (-q/2, q/2]
    let center = |val: i64| {
        if val > q / 2 {
            val - q
        } else {
            val
        }
    };

    let noise_0_raw = pt_dec_zero.coeffs.rns_coeffs[0][0];
    let noise_1_raw = pt_dec_zero.coeffs.rns_coeffs[1][0];
    let noise_2_raw = pt_dec_zero.coeffs.rns_coeffs[2][0];

    println!("Decrypted coeffs[0] = {} (centered: {})", noise_0_raw, center(noise_0_raw));
    println!("Decrypted coeffs[1] = {} (centered: {})", noise_1_raw, center(noise_1_raw));
    println!("Decrypted coeffs[2] = {} (centered: {})", noise_2_raw, center(noise_2_raw));

    let noise_0 = center(noise_0_raw).abs();
    let noise_1 = center(noise_1_raw).abs();
    let noise_2 = center(noise_2_raw).abs();

    println!("Noise in coeff[0] = {} (≈{:.2e})", noise_0, noise_0 as f64);
    println!("Noise in coeff[1] = {} (≈{:.2e})", noise_1, noise_1 as f64);
    println!("Noise in coeff[2] = {} (≈{:.2e})", noise_2, noise_2 as f64);

    // Expected noise should be around error_std * sqrt(n) ≈ 3.2 * 32 ≈ 100
    let expected_noise = params.error_std * (params.n as f64).sqrt();
    println!("Expected noise magnitude: ≈{:.2e}", expected_noise);

    if noise_0 < 10000 && noise_1 < 10000 && noise_2 < 10000 {
        println!("✅ TEST 1 PASSED: Noise is reasonable");
    } else {
        println!("❌ TEST 1 FAILED: Noise is HUGE ({:.2e})", noise_0 as f64);
    }
}
