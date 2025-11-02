//! Debug multiplication to see where noise comes from

use ga_engine::clifford_fhe::{CliffordFHEParams, keygen};
use ga_engine::clifford_fhe::canonical_embedding::{encode_multivector_canonical, decode_multivector_canonical};
use ga_engine::clifford_fhe::ckks::{encrypt, decrypt, multiply, Plaintext};

fn main() {
    let params = CliffordFHEParams::new_test_mult();
    let (pk, sk, evk) = keygen(&params);

    println!("Parameters:");
    println!("  N: {}", params.n);
    println!("  Q: {}", params.modulus_at_level(0));
    println!("  scale: {:.2e}", params.scale);
    println!("  scale²: {:.2e}", params.scale * params.scale);
    println!();

    // Test: [2] × [3] = [6]
    let mv_a = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mv_b = [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let coeffs_a = encode_multivector_canonical(&mv_a, params.scale, params.n);
    let pt_a = Plaintext::new(coeffs_a, params.scale);
    let ct_a = encrypt(&pk, &pt_a, &params);

    println!("After encrypting [2]:");
    let pt_test = decrypt(&sk, &ct_a, &params);
    let mv_test = decode_multivector_canonical(&pt_test.coeffs, pt_test.scale, params.n);
    println!("  Decrypt gives: [{:.6}, {:.6}, ...]", mv_test[0], mv_test[1]);
    println!();

    let coeffs_b = encode_multivector_canonical(&mv_b, params.scale, params.n);
    let pt_b = Plaintext::new(coeffs_b, params.scale);
    let ct_b = encrypt(&pk, &pt_b, &params);

    println!("After encrypting [3]:");
    let pt_test = decrypt(&sk, &ct_b, &params);
    let mv_test = decode_multivector_canonical(&pt_test.coeffs, pt_test.scale, params.n);
    println!("  Decrypt gives: [{:.6}, {:.6}, ...]", mv_test[0], mv_test[1]);
    println!();

    println!("Multiplying ciphertexts...");
    let ct_result = multiply(&ct_a, &ct_b, &evk, &params);
    println!("  Result scale: {:.2e}", ct_result.scale);
    println!("  Result level: {}", ct_result.level);
    println!();

    println!("First few coefficients of c0:");
    for i in 0..5 {
        println!("  c0[{}]: {}", i, ct_result.c0[i]);
    }
    println!();

    let pt_result = decrypt(&sk, &ct_result, &params);
    println!("After decryption:");
    println!("  Plaintext scale: {:.2e}", pt_result.scale);
    println!("First few coefficients:");
    for i in 0..5 {
        println!("  pt[{}]: {}", i, pt_result.coeffs[i]);
    }
    println!();

    let mv_result = decode_multivector_canonical(&pt_result.coeffs, pt_result.scale, params.n);

    println!("Final result: [{:.6}, {:.6}, {:.6}, {:.6}, ...]",
             mv_result[0], mv_result[1], mv_result[2], mv_result[3]);
    println!("Expected: [6.000000, 0.000000, 0.000000, 0.000000, ...]");
    println!();

    let error = (mv_result[0] - 6.0).abs();
    println!("Error: {:.2e}", error);

    if error < 1.0 {
        println!("✓ PASS");
    } else {
        println!("✗ FAIL");
    }
}
