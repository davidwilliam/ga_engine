//! Test basic slot multiplication with canonical embedding
//!
//! Verify that multiplying two ciphertexts gives the expected slot-wise product.

use ga_engine::clifford_fhe::{
    CliffordFHEParams, keygen,
    Plaintext,
};
use ga_engine::clifford_fhe::canonical_embedding::{encode_multivector_canonical, decode_multivector_canonical};
use ga_engine::clifford_fhe::ckks::{encrypt, decrypt, multiply};

fn main() {
    println!("Testing basic slot multiplication with canonical embedding\n");

    let params = CliffordFHEParams::new_test_mult(); // N=1024, scale=2^30
    let (pk, sk, evk) = keygen(&params);

    // Test 1: Simple scalar multiplication
    println!("Test 1: [2, 0, 0, ...] × [3, 0, 0, ...] = [6, 0, 0, ...]");
    let mv_a = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mv_b = [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let coeffs_a = encode_multivector_canonical(&mv_a, params.scale, params.n);
    let pt_a = Plaintext::new(coeffs_a, params.scale);
    let ct_a = encrypt(&pk, &pt_a, &params);

    let coeffs_b = encode_multivector_canonical(&mv_b, params.scale, params.n);
    let pt_b = Plaintext::new(coeffs_b, params.scale);
    let ct_b = encrypt(&pk, &pt_b, &params);

    let ct_result = multiply(&ct_a, &ct_b, &evk, &params);

    println!("  Ciphertext result scale: {:.2e}", ct_result.scale);

    let pt_result = decrypt(&sk, &ct_result, &params);
    println!("  Plaintext result scale: {:.2e}", pt_result.scale);

    // Decode with the plaintext's actual scale!
    let mv_result = decode_multivector_canonical(&pt_result.coeffs, pt_result.scale, params.n);

    println!("Expected: [6.0, 0, 0, 0, 0, 0, 0, 0]");
    println!("Got:      [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
             mv_result[0], mv_result[1], mv_result[2], mv_result[3],
             mv_result[4], mv_result[5], mv_result[6], mv_result[7]);

    let error = (mv_result[0] - 6.0).abs();
    if error < 1.0 {
        println!("✓ PASS\n");
    } else {
        println!("✗ FAIL (error: {:.2e})\n", error);
    }

    // Test 2: Slot-wise multiplication
    println!("Test 2: [1, 2, 0, ...] × [3, 4, 0, ...] = [3, 8, 0, ...]");
    let mv_a = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mv_b = [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let coeffs_a = encode_multivector_canonical(&mv_a, params.scale, params.n);
    let pt_a = Plaintext::new(coeffs_a, params.scale);
    let ct_a = encrypt(&pk, &pt_a, &params);

    let coeffs_b = encode_multivector_canonical(&mv_b, params.scale, params.n);
    let pt_b = Plaintext::new(coeffs_b, params.scale);
    let ct_b = encrypt(&pk, &pt_b, &params);

    let ct_result = multiply(&ct_a, &ct_b, &evk, &params);

    println!("  Ciphertext result scale: {:.2e}", ct_result.scale);

    let pt_result = decrypt(&sk, &ct_result, &params);
    println!("  Plaintext result scale: {:.2e}", pt_result.scale);

    // Decode with the plaintext's actual scale!
    let mv_result = decode_multivector_canonical(&pt_result.coeffs, pt_result.scale, params.n);

    println!("Expected: [3.0, 8.0, 0, 0, 0, 0, 0, 0]");
    println!("Got:      [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
             mv_result[0], mv_result[1], mv_result[2], mv_result[3],
             mv_result[4], mv_result[5], mv_result[6], mv_result[7]);

    let error0 = (mv_result[0] - 3.0).abs();
    let error1 = (mv_result[1] - 8.0).abs();
    if error0 < 1.0 && error1 < 1.0 {
        println!("✓ PASS\n");
    } else {
        println!("✗ FAIL (errors: {:.2e}, {:.2e})\n", error0, error1);
    }
}
