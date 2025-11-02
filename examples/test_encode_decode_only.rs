//! Test encode/decode without encryption

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::canonical_embedding::{encode_multivector_canonical, decode_multivector_canonical};

fn main() {
    let params = CliffordFHEParams::new_test();

    println!("Test 1: Simple roundtrip");
    let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let coeffs = encode_multivector_canonical(&mv, params.scale, params.n);
    let mv_decoded = decode_multivector_canonical(&coeffs, params.scale, params.n);

    println!("Input:   {:?}", mv);
    println!("Decoded: [{:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6}]",
             mv_decoded[0], mv_decoded[1], mv_decoded[2], mv_decoded[3],
             mv_decoded[4], mv_decoded[5], mv_decoded[6], mv_decoded[7]);

    let mut max_error = 0.0;
    for i in 0..8 {
        let error = (mv[i] - mv_decoded[i]).abs();
        if error > max_error {
            max_error = error;
        }
    }

    if max_error < 1e-3 {
        println!("✓ Roundtrip PASS (max error: {:.2e})\n", max_error);
    } else {
        println!("✗ Roundtrip FAIL (max error: {:.2e})\n", max_error);
    }
}
