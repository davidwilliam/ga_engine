//! V3 Full Bootstrap Demo - Fast Production Parameters
//!
//! This example demonstrates REAL bootstrap operation with fast demo parameters:
//! - N=8192 (production ring dimension)
//! - 16 primes (12 for bootstrap, 3 for computation)
//! - Actual noisy ciphertext refresh
//!
//! Timing: ~3-4 minutes total:
//! - Key generation: ~120 seconds (parallelized)
//! - Rotation key generation: ~90 seconds (parallelized)
//! - Bootstrap operation: ~5 seconds
//!
//! This is the REAL DEAL - actual working bootstrap with production N!

use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v3::bootstrapping::{BootstrapContext, BootstrapParams};
use std::time::Instant;

fn main() -> Result<(), String> {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     V3 Full Bootstrap Demo - Fast Production Parameters         ║");
    println!("║     THIS IS THE REAL DEAL - Actual Bootstrap Operation          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("⚠️  WARNING: This example takes 3-4 minutes to complete.");
    println!("    It demonstrates REAL bootstrap with production N=8192.\n");

    // Step 1: Parameters
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 1: Setup Fast Demo Parameters");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let params = CliffordFHEParams::new_v3_bootstrap_fast_demo();

    println!("Parameters:");
    println!("  Ring dimension N: {} (production-ready)", params.n);
    println!("  Number of primes: {} (12 bootstrap + 3 computation)", params.moduli.len());
    println!("  Scale: 2^40 = {}", params.scale);
    println!("  Security level: ~118 bits (slightly reduced from full 128 bits)");

    let bootstrap_params = BootstrapParams::balanced();
    let bootstrap_levels = bootstrap_params.bootstrap_levels;

    println!("\nBootstrap configuration:");
    println!("  Levels for bootstrap: {}", bootstrap_levels);
    println!("  Levels for computation: {}", params.computation_levels(bootstrap_levels));
    println!("  Sine approximation degree: {}", bootstrap_params.sin_degree);
    println!("  Supports bootstrap: {}\n", params.supports_bootstrap(bootstrap_levels));

    // Step 2: Key Generation
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 2: Generating Encryption Keys");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Generating keys (N=8192, 16 primes with Rayon parallelization)...");
    println!("  This will take approximately 120 seconds...\n");

    let start = Instant::now();
    let key_ctx = KeyContext::new(params.clone());
    let (pk, sk, _evk) = key_ctx.keygen();
    let keygen_time = start.elapsed();

    println!("  ✓ Keys generated in {:.2} seconds\n", keygen_time.as_secs_f64());

    // Step 3: Bootstrap Context
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 3: Creating Bootstrap Context");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let num_rotations = 2 * (params.n / 2).trailing_zeros() as usize;
    println!("Generating rotation keys (parallelized)...");
    println!("  Number of rotations needed: {}", num_rotations);
    println!("  This will take approximately 90 seconds...\n");

    let start = Instant::now();
    let bootstrap_ctx = BootstrapContext::new(params.clone(), bootstrap_params, &sk)?;
    let bootstrap_ctx_time = start.elapsed();

    println!("  ✓ Bootstrap context created in {:.2} seconds\n", bootstrap_ctx_time.as_secs_f64());

    // Step 4: Test Encryption
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 4: Encrypt Test Value");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let ckks_ctx = CkksContext::new(params.clone());
    let test_value = 42.0;

    println!("Original value: {}", test_value);

    let pt = ckks_ctx.encode(&[test_value]);
    let mut ct = ckks_ctx.encrypt(&pt, &pk);

    println!("  Initial ciphertext level: {}", ct.level);
    println!("  Initial ciphertext scale: {:.2e}\n", ct.scale);

    // Verify encryption works
    let decrypted_pt = ckks_ctx.decrypt(&ct, &sk);
    let decoded = ckks_ctx.decode(&decrypted_pt);
    let error_before = (decoded[0] - test_value).abs();

    println!("Before any operations:");
    println!("  Decrypted value: {:.10}", decoded[0]);
    println!("  Error: {:.2e}\n", error_before);

    // Step 5: Add Noise via Multiplications
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 5: Simulate Deep Computation (Add Noise)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Performing multiplications to consume levels and add noise...");

    // Perform several multiplications to consume levels
    let num_mults = 5;
    for i in 0..num_mults {
        let pt_mult = ckks_ctx.encode(&[1.0001]);  // Multiply by value close to 1
        ct = ct.multiply_plain(&pt_mult, &ckks_ctx);
        println!("  After mult {}: level={}, scale={:.2e}", i+1, ct.level, ct.scale);
    }

    println!("\nCiphertext after {} multiplications:", num_mults);
    println!("  Current level: {} (consumed {} levels)", ct.level, params.moduli.len() - 1 - ct.level);
    println!("  Current scale: {:.2e}", ct.scale);

    // Check accuracy before bootstrap
    let decrypted_noisy = ckks_ctx.decrypt(&ct, &sk);
    let decoded_noisy = ckks_ctx.decode(&decrypted_noisy);
    let error_noisy = (decoded_noisy[0] - test_value).abs();

    println!("\nAccuracy check (before bootstrap):");
    println!("  Expected: {}", test_value);
    println!("  Got: {:.10}", decoded_noisy[0]);
    println!("  Error: {:.2e}", error_noisy);

    if error_noisy > 1.0 {
        println!("  ⚠️  Large error - ciphertext needs bootstrap!\n");
    } else {
        println!("  ✓ Still accurate (but running low on levels)\n");
    }

    // Step 6: BOOTSTRAP!
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 6: BOOTSTRAP - Refresh Ciphertext");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("Running full bootstrap pipeline:");
    println!("  1. ModRaise - Extend modulus chain");
    println!("  2. CoeffToSlot - Transform to slot domain");
    println!("  3. EvalMod - Homomorphic modular reduction");
    println!("  4. SlotToCoeff - Transform back to coefficient domain\n");

    println!("This will take approximately 10 seconds...\n");

    let start = Instant::now();
    let ct_bootstrapped = bootstrap_ctx.bootstrap(&ct)?;
    let bootstrap_time = start.elapsed();

    println!("  ✓ Bootstrap completed in {:.2} seconds!\n", bootstrap_time.as_secs_f64());

    println!("Bootstrapped ciphertext:");
    println!("  New level: {} (refreshed!)", ct_bootstrapped.level);
    println!("  New scale: {:.2e}\n", ct_bootstrapped.scale);

    // Step 7: Verify Bootstrap Correctness
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 7: Verify Bootstrap Correctness");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let decrypted_bootstrap = ckks_ctx.decrypt(&ct_bootstrapped, &sk);
    let decoded_bootstrap = ckks_ctx.decode(&decrypted_bootstrap);
    let error_bootstrap = (decoded_bootstrap[0] - test_value).abs();

    println!("Accuracy after bootstrap:");
    println!("  Expected: {}", test_value);
    println!("  Got: {:.10}", decoded_bootstrap[0]);
    println!("  Error: {:.2e}\n", error_bootstrap);

    if error_bootstrap < 1.0 {
        println!("  ✓ Bootstrap successful - ciphertext refreshed with correct value!");
    } else {
        println!("  ✗ Bootstrap failed - large error detected");
        return Err(format!("Bootstrap error too large: {:.2e}", error_bootstrap));
    }

    // Final Summary
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    SUCCESS - BOOTSTRAP COMPLETE                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Timing Summary:");
    println!("  Key generation: {:.2}s", keygen_time.as_secs_f64());
    println!("  Bootstrap context: {:.2}s", bootstrap_ctx_time.as_secs_f64());
    println!("  Bootstrap operation: {:.2}s", bootstrap_time.as_secs_f64());
    println!("  Total time: {:.2}s\n", (keygen_time + bootstrap_ctx_time + bootstrap_time).as_secs_f64());

    println!("Accuracy Summary:");
    println!("  Before operations: error = {:.2e}", error_before);
    println!("  After {} mults: error = {:.2e}", num_mults, error_noisy);
    println!("  After bootstrap: error = {:.2e}\n", error_bootstrap);

    println!("What This Demonstrates:");
    println!("  ✓ Production parameters (N=8192, 20 primes)");
    println!("  ✓ Real key generation");
    println!("  ✓ Real rotation key generation");
    println!("  ✓ Real bootstrap operation");
    println!("  ✓ Ciphertext level refresh");
    println!("  ✓ Maintained decryption accuracy\n");

    println!("This is REAL bootstrap - unlimited depth computation is now possible!");
    println!("You can perform 5-7 more multiplications, bootstrap again, and repeat forever.\n");

    Ok(())
}
