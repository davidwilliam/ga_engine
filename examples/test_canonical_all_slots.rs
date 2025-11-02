//! Diagnostic: Check what values are in ALL slots after canonical encoding

use ga_engine::clifford_fhe::CliffordFHEParams;
use ga_engine::clifford_fhe::canonical_embedding::{encode_multivector_canonical, canonical_embed_decode};

fn main() {
    let params = CliffordFHEParams::new_test();
    let scale = params.scale;
    let n = params.n;
    let num_slots = n / 2; // Should be 32

    println!("N = {}, num_slots = {}\n", n, num_slots);

    // Encode a simple multivector: [1, 2, 0, 0, 0, 0, 0, 0]
    let mv = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let coeffs = encode_multivector_canonical(&mv, scale, n);

    // Convert to all slots using canonical_embed_decode (NOT the old FFT-based decode!)
    let slots = canonical_embed_decode(&coeffs, scale, n);

    println!("Input multivector: {:?}\n", mv);
    println!("All {} slots after encoding:", num_slots);
    for i in 0..num_slots {
        println!("  Slot[{}]: {:.6} + {:.6}i (magnitude: {:.6})",
                 i, slots[i].re, slots[i].im, slots[i].norm());
    }

    println!("\n\nDo we have non-zero values in slots 8-31?");
    let mut has_nonzero = false;
    for i in 8..num_slots {
        if slots[i].norm() > 0.001 {
            println!("  Slot[{}] is non-zero: {:.6}", i, slots[i].norm());
            has_nonzero = true;
        }
    }

    if has_nonzero {
        println!("\n✗ WARNING: Slots 8-31 have non-zero values!");
        println!("   This will cause incorrect multiplication results!");
    } else {
        println!("\n✓ Slots 8-31 are all zero (or very small)");
    }
}
