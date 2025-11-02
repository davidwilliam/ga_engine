//! Test polynomial multiplication in plaintext (no encryption)
//!
//! This test verifies that the canonical embedding with orbit-order indexing
//! preserves the slot-wise multiplication property of CKKS:
//!   decode(poly_mult(encode(a), encode(b))) = a ⊙ b

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::canonical_embedding::{encode_multivector_canonical, canonical_embed_decode_product};

fn polynomial_multiply_mod(a: &[i64], b: &[i64], q: i64, n: usize) -> Vec<i64> {
    let mut result = vec![0i64; n];

    for i in 0..n {
        for j in 0..n {
            let idx = i + j;
            if idx < n {
                result[idx] = (result[idx] + a[i] * b[j]) % q;
            } else {
                // x^n = -1 reduction (negacyclic)
                let wrapped_idx = idx % n;
                result[wrapped_idx] = (result[wrapped_idx] - a[i] * b[j]) % q;
            }
        }
    }

    result.iter().map(|&x| ((x % q) + q) % q).collect()
}

fn main() {
    let params = CliffordFHEParams::new_test();
    let q = params.modulus_at_level(0);

    println!("========================================================================");
    println!("Test: Plaintext Polynomial Multiplication with Orbit-Order CKKS");
    println!("========================================================================\n");

    println!("Test 1: [2, 0, 0, ...] × [3, 0, 0, ...] = [6, 0, 0, ...]");
    println!("------------------------------------------------------------------------\n");

    // Encode to polynomials
    let mv_a = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mv_b = [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let coeffs_a = encode_multivector_canonical(&mv_a, params.scale, params.n);
    let coeffs_b = encode_multivector_canonical(&mv_b, params.scale, params.n);

    println!("Encoded a: first 8 coeffs = {:?}", &coeffs_a[0..8]);
    println!("Encoded b: first 8 coeffs = {:?}", &coeffs_b[0..8]);

    // Multiply polynomials (negacyclic convolution mod (x^N + 1, q))
    let coeffs_product = polynomial_multiply_mod(&coeffs_a, &coeffs_b, q, params.n);

    println!("\nProduct: first 8 coeffs = {:?}", &coeffs_product[0..8]);
    println!("Expected first coeff ≈ s²×6/N = {:.0}", params.scale * params.scale * 6.0 / params.n as f64);

    // Decode using the CORRECTED method with center-lift and s² normalization
    println!("\nDecoding with center-lift and s² normalization...");
    let slots = canonical_embed_decode_product(&coeffs_product, params.scale, q, params.n);

    // Extract real parts (should be real-valued for real inputs)
    let mv_result: Vec<f64> = slots.iter().take(8).map(|z| z.re).collect();

    println!("\nResult:   [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
             mv_result[0], mv_result[1], mv_result[2], mv_result[3],
             mv_result[4], mv_result[5], mv_result[6], mv_result[7]);
    println!("Expected: [6.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]");

    // Check error
    let error = (mv_result[0] - 6.0).abs();
    let max_other = mv_result[1..].iter().map(|x| x.abs()).fold(0.0, f64::max);

    println!("\nSlot 0 error: {:.2e}", error);
    println!("Max error in other slots: {:.2e}", max_other);

    if error < 1e-3 && max_other < 1e-3 {
        println!("\n✓ PASS: Polynomial multiplication gives correct slot-wise product!");
        println!("  Orbit-order CKKS preserves the multiplication property ✅\n");
    } else {
        println!("\n✗ FAIL: Errors too large");
        println!("  Slot 0: expected 6.0, got {:.6} (error {:.2e})", mv_result[0], error);
        println!("  Check implementation of center-lift and scale normalization\n");
    }

    println!("========================================================================");
    println!("Test 2: [1, 2, 0, ...] × [3, 4, 0, ...] = [3, 8, 0, ...]");
    println!("========================================================================\n");

    let mv_a = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mv_b = [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let coeffs_a = encode_multivector_canonical(&mv_a, params.scale, params.n);
    let coeffs_b = encode_multivector_canonical(&mv_b, params.scale, params.n);
    let coeffs_product = polynomial_multiply_mod(&coeffs_a, &coeffs_b, q, params.n);

    let slots = canonical_embed_decode_product(&coeffs_product, params.scale, q, params.n);
    let mv_result: Vec<f64> = slots.iter().take(8).map(|z| z.re).collect();

    println!("Result:   [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
             mv_result[0], mv_result[1], mv_result[2], mv_result[3],
             mv_result[4], mv_result[5], mv_result[6], mv_result[7]);
    println!("Expected: [3.000000, 8.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]");

    let error0 = (mv_result[0] - 3.0).abs();
    let error1 = (mv_result[1] - 8.0).abs();
    let max_other = mv_result[2..].iter().map(|x| x.abs()).fold(0.0, f64::max);

    println!("\nSlot 0 error: {:.2e}", error0);
    println!("Slot 1 error: {:.2e}", error1);
    println!("Max error in other slots: {:.2e}", max_other);

    if error0 < 1e-3 && error1 < 1e-3 && max_other < 1e-3 {
        println!("\n✓ PASS: Element-wise multiplication works for multiple slots!\n");
    } else {
        println!("\n✗ FAIL: Errors too large\n");
    }

    println!("========================================================================");
}
