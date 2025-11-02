//! Test homomorphic multiplication with manual rescaling
//!
//! This manually handles the rescaling step that should happen after multiplication

use ga_engine::clifford_fhe::{
    CliffordFHEParams, keygen,
    Plaintext, Ciphertext,
};
use ga_engine::clifford_fhe::canonical_embedding::{encode_multivector_canonical, canonical_embed_decode_homomorphic_product};
use ga_engine::clifford_fhe::ckks::{encrypt, decrypt, multiply};

fn main() {
    println!("========================================================================");
    println!("Test: Homomorphic Multiplication with Manual Rescaling");
    println!("========================================================================\n");

    let params = CliffordFHEParams::new_test();
    let (pk, sk, evk) = keygen(&params);
    let q = params.modulus_at_level(1); // Level after multiply

    println!("Test: [2, 0, ...] × [3, 0, ...] = [6, 0, ...]");
    println!("------------------------------------------------------------------------\n");

    let mv_a = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mv_b = [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let coeffs_a = encode_multivector_canonical(&mv_a, params.scale, params.n);
    let pt_a = Plaintext::new(coeffs_a, params.scale);
    let ct_a = encrypt(&pk, &pt_a, &params);

    let coeffs_b = encode_multivector_canonical(&mv_b, params.scale, params.n);
    let pt_b = Plaintext::new(coeffs_b, params.scale);
    let ct_b = encrypt(&pk, &pt_b, &params);

    println!("Input scales: a={:.2e}, b={:.2e}", ct_a.scale, ct_b.scale);

    // Homomorphic multiplication
    let ct_result = multiply(&ct_a, &ct_b, &evk, &params);

    println!("Result scale: {:.2e}", ct_result.scale);
    println!("Result level: {}", ct_result.level);

    // Decrypt
    let pt_result = decrypt(&sk, &ct_result, &params);

    println!("\nDecrypted plaintext:");
    println!("  Scale: {:.2e}", pt_result.scale);
    println!("  First 4 coeffs: {:?}", &pt_result.coeffs[0..4]);

    // Decode using canonical_embed_decode_homomorphic_product
    // (decrypt already center-lifted, so don't re-lift)
    let slots = canonical_embed_decode_homomorphic_product(&pt_result.coeffs, params.scale, params.n);
    let mv_result: Vec<f64> = slots.iter().take(8).map(|z| z.re).collect();

    println!("\nResult (using decode_product):");
    println!("  [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
             mv_result[0], mv_result[1], mv_result[2], mv_result[3],
             mv_result[4], mv_result[5], mv_result[6], mv_result[7]);
    println!("\nExpected: [6.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]");

    let error = (mv_result[0] - 6.0).abs();
    let max_other = mv_result[1..].iter().map(|x| x.abs()).fold(0.0, f64::max);

    println!("\nSlot 0 error: {:.2e}", error);
    println!("Max error in other slots: {:.2e}", max_other);

    if error < 1.0 && max_other < 1.0 {
        println!("\n✓ PASS: Homomorphic multiplication works!\n");
    } else {
        println!("\n✗ FAIL: Errors too large\n");
    }

    println!("========================================================================");
}
