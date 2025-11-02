//! Debug test for multiplication with scale checking

use ga_engine::clifford_fhe::{
    CliffordFHEParams, keygen,
    Plaintext,
};
use ga_engine::clifford_fhe::canonical_embedding::{encode_multivector_canonical, decode_multivector_canonical};
use ga_engine::clifford_fhe::ckks::{encrypt, decrypt, multiply};

fn main() {
    let params = CliffordFHEParams::new_test();
    let (pk, sk, evk) = keygen(&params);

    println!("Initial scale: {}", params.scale);

    // Simple test: [2] × [3] = [6]
    let mv_a = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mv_b = [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let coeffs_a = encode_multivector_canonical(&mv_a, params.scale, params.n);
    let pt_a = Plaintext::new(coeffs_a, params.scale);
    println!("Plaintext a scale: {}", pt_a.scale);

    let ct_a = encrypt(&pk, &pt_a, &params);
    println!("Ciphertext a scale: {}, level: {}", ct_a.scale, ct_a.level);

    let coeffs_b = encode_multivector_canonical(&mv_b, params.scale, params.n);
    let pt_b = Plaintext::new(coeffs_b, params.scale);
    let ct_b = encrypt(&pk, &pt_b, &params);
    println!("Ciphertext b scale: {}, level: {}", ct_b.scale, ct_b.level);

    let ct_result = multiply(&ct_a, &ct_b, &evk, &params);
    println!("Result scale: {}, level: {}", ct_result.scale, ct_result.level);
    println!("Expected scale after multiply: {:.2e}", params.scale * params.scale / params.scale);

    let pt_result = decrypt(&sk, &ct_result, &params);
    println!("Decrypted plaintext scale: {}", pt_result.scale);

    // Try decoding with the result's actual scale
    let mv_result = decode_multivector_canonical(&pt_result.coeffs, pt_result.scale, params.n);

    println!("\nExpected: [6.0, 0, 0, ...]");
    println!("Got:      [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
             mv_result[0], mv_result[1], mv_result[2], mv_result[3],
             mv_result[4], mv_result[5], mv_result[6], mv_result[7]);

    let error = (mv_result[0] - 6.0).abs();
    if error < 1.0 {
        println!("\n✓ PASS (error: {:.2e})", error);
    } else {
        println!("\n✗ FAIL (error: {:.2e})", error);
    }
}
