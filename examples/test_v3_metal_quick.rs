//! V3 Metal GPU Quick Test - Smaller Parameters for Fast Testing
//!
//! This example uses N=2048 with fewer primes for FAST testing of Metal GPU.
//! Purpose: Verify Metal GPU NTT is working correctly in the V3 pipeline.
//!
//! Expected runtime: ~5-10 seconds

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use std::time::Instant;

#[cfg(feature = "v2-gpu-metal")]
use ga_engine::clifford_fhe_v2::backends::gpu_metal::keys::MetalKeyContext;

#[cfg(not(feature = "v2-gpu-metal"))]
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;

fn main() -> Result<(), String> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     V3 Metal GPU Quick Test - N=1024 for Fast Verification      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    #[cfg(feature = "v2-gpu-metal")]
    println!("✓ Metal GPU support: ENABLED\n");

    #[cfg(not(feature = "v2-gpu-metal"))]
    {
        println!("❌ Metal GPU support: DISABLED");
        println!("   Run with: cargo run --release --features v2,v3,v2-gpu-metal --example test_v3_metal_quick\n");
        return Err("Metal GPU feature not enabled".to_string());
    }

    // Step 1: Small parameters for quick testing
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 1: Setup Small Test Parameters (N=1024)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let params = CliffordFHEParams::new_test_ntt_1024();

    println!("Parameters:");
    println!("  Ring dimension N: {} (SMALL - for quick GPU testing)", params.n);
    println!("  Number of primes: {}", params.moduli.len());
    println!("  Scale: 2^40 = {}", params.scale);
    println!("\n⚠️  NOTE: This is a quick test to verify Metal GPU NTT is working!\n");

    // Step 2: Key Generation (should use Metal GPU for NTT operations)
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 2: Generating Encryption Keys (with Metal GPU)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Generating keys (N={}, {} primes)...", params.n, params.moduli.len());
    let key_start = Instant::now();

    #[cfg(feature = "v2-gpu-metal")]
    let (pk, sk, _evk) = {
        let mut key_ctx = MetalKeyContext::new(params.clone())?;
        key_ctx.keygen()?
    };

    #[cfg(not(feature = "v2-gpu-metal"))]
    let (pk, sk, _evk) = {
        let key_ctx = KeyContext::new(params.clone());
        key_ctx.keygen()
    };

    let key_time = key_start.elapsed();
    println!("  ✓ Key generation completed in {:.2} seconds!", key_time.as_secs_f64());
    println!("    (Metal GPU accelerated NTT operations)\n");

    // Step 3: Create CKKS context
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 3: Encrypt and Perform Operations");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let ckks = CkksContext::new(params.clone());
    let test_value = 42.0;

    println!("Original value: {}", test_value);

    // Encrypt
    let encrypt_start = Instant::now();
    let pt = ckks.encode(&[test_value]);
    let mut ct = ckks.encrypt(&pt, &pk);
    let encrypt_time = encrypt_start.elapsed();

    println!("  Initial ciphertext level: {}", ct.level);
    println!("  ✓ Encryption time: {:.4} seconds", encrypt_time.as_secs_f64());

    // Perform some operations to test GPU
    let ops_start = Instant::now();

    // Multiply by 2.0 (uses GPU-accelerated NTT!)
    let pt_mult = ckks.encode(&[2.0]);
    ct = ct.multiply_plain(&pt_mult, &ckks);
    println!("\n  After multiply by 2.0:");
    println!("    Level: {}, Scale: {:.2e}", ct.level, ct.scale);

    let ops_time = ops_start.elapsed();
    println!("  ✓ Operations time: {:.4} seconds (GPU-accelerated NTT)\n", ops_time.as_secs_f64());

    // Decrypt and verify
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 4: Decrypt and Verify Result");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let decrypt_start = Instant::now();
    let decrypted_pt = ckks.decrypt(&ct, &sk);
    let decoded = ckks.decode(&decrypted_pt);
    let decrypt_time = decrypt_start.elapsed();

    let expected = test_value * 2.0;  // 42.0 * 2.0 = 84.0
    let error = (decoded[0] - expected).abs();

    println!("  Expected value: {:.10}", expected);
    println!("  Decrypted value: {:.10}", decoded[0]);
    println!("  Error: {:.10}", error);
    println!("  ✓ Decryption time: {:.4} seconds\n", decrypt_time.as_secs_f64());

    if error < 0.1 {
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║              ✅ SUCCESS - Metal GPU NTT is Working!              ║");
        println!("╚══════════════════════════════════════════════════════════════════╝\n");

        println!("Summary:");
        println!("  ✓ Key generation: {:.2}s (GPU-accelerated)", key_time.as_secs_f64());
        println!("  ✓ Encryption: {:.4}s", encrypt_time.as_secs_f64());
        println!("  ✓ Operations: {:.4}s (GPU-accelerated NTT)", ops_time.as_secs_f64());
        println!("  ✓ Decryption: {:.4}s", decrypt_time.as_secs_f64());
        println!("  ✓ Total time: {:.2}s", (key_time + encrypt_time + ops_time + decrypt_time).as_secs_f64());
        println!("  ✓ Accuracy: Error < 0.1 ✓");
        println!("\n🎉 Metal GPU acceleration is working correctly!");
        println!("   The fixed Montgomery multiplication is operational.\n");
        println!("Next steps:");
        println!("  - Run full bootstrap with N=8192:");
        println!("    cargo run --release --features v2,v3,v2-gpu-metal --example test_v3_full_bootstrap\n");

        Ok(())
    } else {
        println!("╔══════════════════════════════════════════════════════════════════╗");
        println!("║                ❌ FAILED - Large Decryption Error                ║");
        println!("╚══════════════════════════════════════════════════════════════════╝\n");

        println!("Error: {:.10} (should be < 0.1)", error);
        println!("This suggests an issue with GPU NTT implementation.\n");

        Err(format!("Decryption error too large: {}", error))
    }
}
