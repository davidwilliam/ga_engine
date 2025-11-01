//! Clifford-FHE Geometric Product Example
//!
//! This demonstrates Phase 2: Homomorphic Geometric Product
//!
//! **THE KEY INNOVATION** that makes Clifford-FHE unique!

use ga_engine::clifford_fhe::{
    ckks::{decrypt, encrypt, Plaintext},
    encoding::{decode_multivector, encode_multivector},
    geometric_product::geometric_product_homomorphic,
    keys::keygen,
    params::CliffordFHEParams,
};
use ga_engine::ga::geometric_product;

fn main() {
    println!("=================================================================");
    println!("Clifford-FHE Phase 2: Homomorphic Geometric Product");
    println!("=================================================================\n");

    println!("🚀 THE KEY INNOVATION: Computing GP on encrypted multivectors!\n");

    // Set up parameters
    let params = CliffordFHEParams::new_128bit();
    println!("Parameters:");
    println!("  Ring dimension (N): {}", params.n);
    println!("  Security level: {:?}", params.security);
    println!("  Scaling factor: 2^{}\n", params.scale.log2() as u32);

    // Generate keys
    println!("Generating keys (includes evaluation key for multiplication)...");
    let (pk, sk, evk) = keygen(&params);
    println!("✓ Keys generated\n");

    // Test Case 1: Simple geometric product
    println!("Test 1: Geometric Product - (1 + 2e1) ⊗ (3 + 4e2)");
    println!("-------------------------------------------------------");

    let mv_a = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 1 + 2e1
    let mv_b = [3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // 3 + 4e2

    println!("Multivector A: {:?}", &mv_a[..4]);
    println!("Multivector B: {:?}", &mv_b[..4]);

    // Compute expected result using our existing GP
    let expected = geometric_product(&mv_a, &mv_b);

    println!("\nExpected result (plaintext GP):");
    println!("  {:?}", &expected[..4]);
    println!("  (Should be: [3.0, 6.0, 4.0, 0.0, 8.0, 0.0, 0.0, 0.0])");

    // Encrypt both multivectors
    println!("\nEncrypting multivectors...");
    let pt_a_coeffs = encode_multivector(&mv_a, params.scale, params.n);
    let pt_a = Plaintext::new(pt_a_coeffs, params.scale);
    let ct_a = encrypt(&pk, &pt_a, &params);

    let pt_b_coeffs = encode_multivector(&mv_b, params.scale, params.n);
    let pt_b = Plaintext::new(pt_b_coeffs, params.scale);
    let ct_b = encrypt(&pk, &pt_b, &params);
    println!("✓ Both multivectors encrypted\n");

    // HOMOMORPHIC GEOMETRIC PRODUCT!
    println!("🔥 Computing HOMOMORPHIC geometric product...");
    println!("   Enc(a) ⊗ Enc(b) = Enc(a ⊗ b)");

    let ct_result = geometric_product_homomorphic(&ct_a, &ct_b, &evk, &params);

    println!("✓ Homomorphic GP complete!\n");

    // Decrypt result
    println!("Decrypting result...");
    let pt_result = decrypt(&sk, &ct_result, &params);
    let result = decode_multivector(&pt_result.coeffs, params.scale);

    println!("Decrypted result:");
    println!("  {:?}", &result[..4]);

    // Check accuracy
    println!("\nAccuracy check:");
    let mut max_error = 0.0f64;
    for i in 0..8 {
        let error = (result[i] - expected[i]).abs();
        if error > max_error {
            max_error = error;
        }
        if i < 4 && error > 0.1 {
            println!("  Component {}: {:.6} (expected {:.6}, error {:.2e})",
                     i, result[i], expected[i], error);
        }
    }
    println!("Max error: {:.2e}", max_error);

    if max_error < 1e-3 {
        println!("✅ PASS: Homomorphic GP works correctly!\n");
    } else {
        println!("⚠️  Warning: Error higher than expected (this is Phase 2 MVP)\n");
        println!("   NOTE: Current implementation is simplified proof-of-concept.");
        println!("   Full implementation needs proper component extraction.");
    }

    println!("=================================================================");
    println!("Phase 2 Status");
    println!("=================================================================");
    println!("✅ Structure constants: Implemented (all 64 products)");
    println!("🚧 Homomorphic multiplication: Working (basic version)");
    println!("🚧 Geometric product: Proof-of-concept (simplified)");
    println!("⏳ Component extraction: TODO (needed for full GP)");
    println!("⏳ Component packing: TODO (combine 8 results into 1 CT)");
    println!("\nNext steps:");
    println!("1. Implement proper component extraction from ciphertexts");
    println!("2. Implement component packing (8 CTs → 1 CT)");
    println!("3. Optimize: Use polynomial masking instead of extraction");
    println!("4. Test on complex multivectors (all 8 components non-zero)");
    println!("\n🎯 Goal: Full homomorphic GP by end of week!");
    println!("\n💡 This will be FIRST FHE scheme with native geometric algebra support!");
}
