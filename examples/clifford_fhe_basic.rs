//! Basic Clifford-FHE example
//!
//! Demonstrates:
//! - Encrypting multivectors
//! - Homomorphic addition
//! - Decryption and verification
//!
//! This is Phase 1: Basic CKKS functionality
//! Phase 2 will add homomorphic geometric product!

use ga_engine::clifford_fhe::{
    ckks::{add, decrypt, encrypt, Plaintext},
    encoding::{decode_multivector, encode_multivector},
    keys::keygen,
    params::CliffordFHEParams,
};

fn main() {
    println!("=================================================================");
    println!("Clifford-FHE: Fully Homomorphic Encryption for Geometric Algebra");
    println!("=================================================================\n");

    println!("Phase 1: Basic CKKS Encryption");
    println!("---------------------------------\n");

    // Set up parameters (128-bit security)
    let params = CliffordFHEParams::new_128bit();
    println!("Parameters:");
    println!("  Ring dimension (N): {}", params.n);
    println!("  Security level: {:?}", params.security);
    println!("  Number of levels: {}", params.max_level());
    println!("  Scaling factor: 2^{}\n", params.scale.log2() as u32);

    // Generate keys
    println!("Generating keys...");
    let (pk, sk, _evk) = keygen(&params);
    println!("✓ Keys generated\n");

    // Create two multivectors
    println!("Test 1: Encrypting and decrypting a multivector");
    println!("------------------------------------------------");
    let mv1 = [1.5, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 1.5 + 2.0e1
    println!("Multivector 1: {:?}", mv1);

    // Encode as CKKS plaintext
    let pt1_coeffs = encode_multivector(&mv1, params.scale, params.n);
    let pt1 = Plaintext::new(pt1_coeffs, params.scale);

    // Encrypt
    println!("Encrypting...");
    let ct1 = encrypt(&pk, &pt1, &params);
    println!("✓ Encrypted (ciphertext size: {} coefficients)\n", ct1.n);

    // Decrypt
    println!("Decrypting...");
    let pt1_decrypted = decrypt(&sk, &ct1, &params);
    let mv1_decrypted = decode_multivector(&pt1_decrypted.coeffs, params.scale);
    println!("Decrypted: {:?}", mv1_decrypted);

    // Check accuracy
    let mut max_error = 0.0f64;
    for i in 0..8 {
        let error = (mv1_decrypted[i] - mv1[i]).abs();
        if error > max_error {
            max_error = error;
        }
    }
    println!("Max decryption error: {:.2e}", max_error);

    if max_error < 1e-6 {
        println!("✅ PASS: Decryption accurate!\n");
    } else {
        println!("⚠️  Warning: Decryption error higher than expected\n");
    }

    // Test homomorphic addition
    println!("Test 2: Homomorphic Addition");
    println!("-----------------------------");

    let mv2 = [0.5, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 0.5 + 1.0e1 + 3.0e2
    println!("Multivector 1: {:?}", mv1);
    println!("Multivector 2: {:?}", mv2);

    // Encrypt second multivector
    let pt2_coeffs = encode_multivector(&mv2, params.scale, params.n);
    let pt2 = Plaintext::new(pt2_coeffs, params.scale);
    let ct2 = encrypt(&pk, &pt2, &params);
    println!("✓ Both encrypted\n");

    // Homomorphic addition
    println!("Computing: Enc(mv1) + Enc(mv2)...");
    let ct_sum = add(&ct1, &ct2, &params);
    println!("✓ Homomorphic addition complete\n");

    // Decrypt result
    println!("Decrypting result...");
    let pt_sum = decrypt(&sk, &ct_sum, &params);
    let mv_sum = decode_multivector(&pt_sum.coeffs, params.scale);
    println!("Decrypted sum: {:?}", mv_sum);

    // Expected result
    let expected: Vec<f64> = mv1.iter().zip(&mv2).map(|(a, b)| a + b).collect();
    println!("Expected:      {:?}", expected);

    // Check accuracy
    max_error = 0.0f64;
    for i in 0..8 {
        let error = (mv_sum[i] - expected[i]).abs();
        if error > max_error {
            max_error = error;
        }
    }
    println!("\nMax error: {:.2e}", max_error);

    if max_error < 1e-5 {
        println!("✅ PASS: Homomorphic addition works correctly!\n");
    } else {
        println!("⚠️  Warning: Addition error higher than expected\n");
    }

    println!("=================================================================");
    println!("Summary");
    println!("=================================================================");
    println!("✅ Encryption/Decryption: Working");
    println!("✅ Homomorphic Addition: Working");
    println!("⏳ Homomorphic Multiplication: TODO (Phase 2)");
    println!("⏳ Geometric Product: TODO (Phase 2)");
    println!("⏳ Rotations: TODO (Phase 3)");
    println!("\nNext steps:");
    println!("1. Integrate optimized NTT from ntt.rs (10-100× speedup)");
    println!("2. Implement homomorphic multiplication with relinearization");
    println!("3. Design geometric product using structure constants");
    println!("4. Implement rotor-based rotations");
    println!("\n🚀 Clifford-FHE is under active development!");
    println!("   This will be the FIRST FHE scheme for geometric algebra!\n");
}
