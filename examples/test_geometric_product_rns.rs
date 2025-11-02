//! Test RNS-CKKS Homomorphic Geometric Product
//!
//! This demonstrates the KEY INNOVATION of Clifford FHE:
//! Performing geometric algebra operations on ENCRYPTED data!
//!
//! Test: (1 + 2e₁) ⊗ (3 + 4e₂) = 3 + 6e₁ + 4e₂ + 8e₁₂

use ga_engine::clifford_fhe::params::CliffordFHEParams;
use ga_engine::clifford_fhe::keys_rns::rns_keygen;
use ga_engine::clifford_fhe::ckks_rns::{RnsPlaintext, rns_encrypt, rns_decrypt};
use ga_engine::clifford_fhe::geometric_product_rns::{
    encode_multivector_2d, decode_multivector_2d, geometric_product_2d_componentwise
};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  Clifford FHE: Homomorphic Geometric Product (RNS-CKKS)      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Setup parameters
    let mut params = CliffordFHEParams::new_rns_mult();
    params.n = 64;
    params.scale = 2f64.powi(40);
    params.error_std = 3.2;
    params.moduli = vec![
        1152921504606851201,  // q0 ≈ 2^60
        1099511628161,        // q1 ≈ 2^40 ≈ Δ
    ];

    let delta = params.scale;
    let primes = &params.moduli;

    println!("Parameters:");
    println!("  N = {}", params.n);
    println!("  Δ = {} = 2^40", delta);
    println!("  Primes: {} (60-bit), {} (40-bit)", primes[0], primes[1]);
    println!();

    // Generate keys
    println!("Generating keys...");
    let (pk, sk, evk) = rns_keygen(&params);
    println!("✓ Keys generated\n");

    // Test case: (1 + 2e₁) ⊗ (3 + 4e₂)
    println!("═══════════════════════════════════════════════════════════════");
    println!("Test: (1 + 2e₁) ⊗ (3 + 4e₂)");
    println!("═══════════════════════════════════════════════════════════════\n");

    let mv_a = [1.0, 2.0, 0.0, 0.0];  // 1 + 2e₁
    let mv_b = [3.0, 0.0, 4.0, 0.0];  // 3 + 4e₂

    println!("Multivector a = [scalar, e₁, e₂, e₁₂]");
    println!("  a = {:?}", mv_a);
    println!("  a = {} + {}e₁", mv_a[0], mv_a[1]);
    println!();

    println!("Multivector b:");
    println!("  b = {:?}", mv_b);
    println!("  b = {} + {}e₂", mv_b[0], mv_b[2]);
    println!();

    // Expected result:
    // (1 + 2e₁) ⊗ (3 + 4e₂)
    // = 1·3 + 1·4e₂ + 2e₁·3 + 2e₁·4e₂
    // = 3 + 4e₂ + 6e₁ + 8e₁₂
    let expected = [3.0, 6.0, 4.0, 8.0];
    println!("Expected result:");
    println!("  a ⊗ b = {:?}", expected);
    println!("  a ⊗ b = {} + {}e₁ + {}e₂ + {}e₁₂",
             expected[0], expected[1], expected[2], expected[3]);
    println!();

    // Encrypt each component separately (componentwise approach)
    println!("Encrypting multivectors (componentwise)...");

    let mut cts_a = Vec::new();
    let mut cts_b = Vec::new();

    for i in 0..4 {
        // Encrypt component i of a
        let mut coeffs_a = vec![0i64; params.n];
        coeffs_a[0] = (mv_a[i] * delta).round() as i64;
        let pt_a = RnsPlaintext::from_coeffs(coeffs_a, delta, primes, 0);
        let ct_a = rns_encrypt(&pk, &pt_a, &params);
        cts_a.push(ct_a);

        // Encrypt component i of b
        let mut coeffs_b = vec![0i64; params.n];
        coeffs_b[0] = (mv_b[i] * delta).round() as i64;
        let pt_b = RnsPlaintext::from_coeffs(coeffs_b, delta, primes, 0);
        let ct_b = rns_encrypt(&pk, &pt_b, &params);
        cts_b.push(ct_b);
    }

    println!("✓ Encrypted 4 components of each multivector\n");

    // Verify encryption
    println!("Verifying encryption...");
    for i in 0..4 {
        let pt_check = rns_decrypt(&sk, &cts_a[i], &params);
        let val = (pt_check.coeffs.rns_coeffs[0][0] as f64) / delta;
        println!("  a[{}]: encrypted {:.3} → decrypts to {:.6}", i, mv_a[i], val);
    }
    println!();

    // HOMOMORPHIC GEOMETRIC PRODUCT!
    println!("Computing homomorphic geometric product...");
    println!("  (This happens on ENCRYPTED data!)");

    let cts_a_array: [_; 4] = [
        cts_a[0].clone(),
        cts_a[1].clone(),
        cts_a[2].clone(),
        cts_a[3].clone(),
    ];

    let cts_b_array: [_; 4] = [
        cts_b[0].clone(),
        cts_b[1].clone(),
        cts_b[2].clone(),
        cts_b[3].clone(),
    ];

    let cts_result = geometric_product_2d_componentwise(
        &cts_a_array,
        &cts_b_array,
        &evk,
        &params,
    );

    println!("✓ Homomorphic geometric product completed\n");

    // Decrypt result
    println!("Decrypting result...");
    let mut result = [0.0; 4];

    for i in 0..4 {
        let pt_result = rns_decrypt(&sk, &cts_result[i], &params);
        result[i] = (pt_result.coeffs.rns_coeffs[0][0] as f64) / cts_result[i].scale;
    }

    println!("  result = {:?}", result);
    println!("  result = {} + {}e₁ + {}e₂ + {}e₁₂\n",
             result[0], result[1], result[2], result[3]);

    // Compare with expected
    println!("═══════════════════════════════════════════════════════════════");
    println!("Results:");
    println!("═══════════════════════════════════════════════════════════════");

    let mut max_error: f64 = 0.0;
    for i in 0..4 {
        let error = (result[i] - expected[i]).abs();
        max_error = max_error.max(error);

        println!("  Component {}: {:.6} (expected {:.6}, error {:.2e})",
                 i, result[i], expected[i], error);
    }

    println!();

    if max_error < 0.01 {
        println!("✅ SUCCESS! Homomorphic geometric product works!");
        println!("   Max error: {:.2e}", max_error);
        println!();
        println!("🎉 You just computed (1 + 2e₁) ⊗ (3 + 4e₂) on ENCRYPTED data!");
        println!("   The server never saw the plaintext values!");
    } else {
        println!("❌ FAILED: Max error too large: {:.6}", max_error);
    }

    println!();
    println!("═══════════════════════════════════════════════════════════════");
    println!("What Just Happened:");
    println!("═══════════════════════════════════════════════════════════════");
    println!();
    println!("1. Encrypted two multivectors: a = 1 + 2e₁, b = 3 + 4e₂");
    println!("2. Computed geometric product a ⊗ b HOMOMORPHICALLY");
    println!("3. Decrypted to get: {} + {}e₁ + {}e₂ + {}e₁₂",
             result[0], result[1], result[2], result[3]);
    println!();
    println!("This enables:");
    println!("  • Privacy-preserving robotics (encrypted poses/transforms)");
    println!("  • Secure physics simulations (encrypted spacetime calcs)");
    println!("  • Confidential computer graphics (encrypted geometry)");
    println!("  • Private machine learning (encrypted features)");
    println!();
}
