//! Debug rotation to understand what's going wrong

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;

fn main() {
    println!("=== Rotation Debug ===\n");

    // Use small params
    let params = CliffordFHEParams::new_128bit();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _) = key_ctx.keygen();
    let ckks_ctx = CkksContext::new(params.clone());

    // Simple message
    let mut message = vec![0.0; params.n / 2];
    message[0] = 1.0;
    message[1] = 2.0;
    message[2] = 3.0;

    println!("Original message: {:?}", &message[..5]);

    // Encode, encrypt, decrypt to verify baseline works
    let pt = ckks_ctx.encode(&message);
    let ct = ckks_ctx.encrypt(&pt, &pk);
    let pt_dec = ckks_ctx.decrypt(&ct, &sk);
    let dec = ckks_ctx.decode(&pt_dec);

    println!("Decrypted (no rotation): {:?}", &dec[..5]);

    // Verify encryption/decryption works
    for i in 0..3 {
        assert!((dec[i] - message[i]).abs() < 0.01,
                "Baseline enc/dec failed: expected {}, got {}", message[i], dec[i]);
    }

    println!("\n✓ Baseline encryption/decryption works correctly");

    // Now let's manually inspect what a rotation should do in CKKS
    println!("\n=== CKKS Rotation Theory ===");
    println!("In CKKS, rotation by k moves slot[i] → slot[(i-k) mod N/2]");
    println!("So rotation by 1 should: [1,2,3,0,...] → [2,3,0,0,...]");
    println!("(first element wraps to end, which is all zeros)");
}
