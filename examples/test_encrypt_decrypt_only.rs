//! Test basic encrypt/decrypt without multiplication

use ga_engine::clifford_fhe::{CliffordFHEParams, keygen};
use ga_engine::clifford_fhe::canonical_embedding::{encode_multivector_canonical, decode_multivector_canonical};
use ga_engine::clifford_fhe::ckks::{encrypt, decrypt, Plaintext};

fn main() {
    let params = CliffordFHEParams::new_test_mult();
    let (pk, sk, _evk) = keygen(&params);

    let mv = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let coeffs = encode_multivector_canonical(&mv, params.scale, params.n);
    let pt = Plaintext::new(coeffs, params.scale);

    println!("Original: {:?}", mv);
    println!("Scale: {:.2e}", params.scale);
    println!("N: {}", params.n);
    println!("Q: {}", params.modulus_at_level(0));

    let ct = encrypt(&pk, &pt, &params);
    println!("Ciphertext scale: {:.2e}", ct.scale);

    let pt_result = decrypt(&sk, &ct, &params);
    println!("Plaintext result scale: {:.2e}", pt_result.scale);

    let mv_result = decode_multivector_canonical(&pt_result.coeffs, pt_result.scale, params.n);

    println!("After encrypt/decrypt: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
             mv_result[0], mv_result[1], mv_result[2], mv_result[3],
             mv_result[4], mv_result[5], mv_result[6], mv_result[7]);

    let error = (mv_result[0] - 2.0).abs();
    println!("Error: {:.2e}", error);

    if error < 0.1 {
        println!("✓ PASS");
    } else {
        println!("✗ FAIL");
    }
}
