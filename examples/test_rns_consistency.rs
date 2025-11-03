/// Test RNS consistency with 10 primes vs 2-3 primes
use ga_engine::clifford_fhe::{
    ckks_rns::{rns_decrypt, rns_encrypt, RnsPlaintext},
    keys_rns::rns_keygen,
    params::CliffordFHEParams,
    rns::RnsPolynomial,
};

fn check_crt_consistency(name: &str, poly: &RnsPolynomial, primes: &[i64]) -> bool {
    let num_primes = poly.num_primes();
    if num_primes < 2 {
        println!("  {} - only 1 prime, skipping CRT check", name);
        return true;
    }

    // Check first coefficient for CRT consistency
    let mut all_consistent = true;

    // Try to reconstruct using CRT with first 2 primes
    let r0 = poly.rns_coeffs[0][0] as i128;
    let r1 = poly.rns_coeffs[0][1] as i128;
    let p0 = primes[0] as i128;
    let p1 = primes[1] as i128;

    // Check: r0 mod p1 should equal r1 mod p1
    let r0_mod_p1 = ((r0 % p1) + p1) % p1;
    let r1_mod_p1 = ((r1 % p1) + p1) % p1;

    if r0_mod_p1 != r1_mod_p1 {
        println!("  {} - ❌ INCONSISTENT: coeff[0]", name);
        println!("    r0={} mod p1={} gives {}", r0, p1, r0_mod_p1);
        println!("    r1={} (should match {})", r1, r0_mod_p1);
        all_consistent = false;
    } else {
        println!("  {} - ✓ Consistent (coeff[0] checked with 2 primes)", name);
    }

    // Also check that r1 mod p0 equals r0 mod p0
    let r1_mod_p0 = ((r1 % p0) + p0) % p0;
    let r0_mod_p0 = ((r0 % p0) + p0) % p0;

    if r1_mod_p0 != r0_mod_p0 {
        println!("    (Also: r1 mod p0 doesn't match r0)");
        all_consistent = false;
    }

    all_consistent
}

fn test_with_params(params: &CliffordFHEParams, label: &str) {
    println!("\n{}", "=".repeat(70));
    println!("Testing: {}", label);
    println!("  N={}, {} primes, scale=2^{}",
             params.n, params.moduli.len(), params.scale.log2() as u32);
    println!("{}", "=".repeat(70));

    let (pk, sk, _evk) = rns_keygen(&params);

    // Encode simple value: 1.5
    let value = 1.5;
    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = (value * params.scale).round() as i64;

    println!("\n1. Plaintext encoding:");
    let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);
    check_crt_consistency("pt.coeffs", &pt.coeffs, &params.moduli);

    println!("\n2. After encryption:");
    let ct = rns_encrypt(&pk, &pt, &params);
    check_crt_consistency("ct.c0", &ct.c0, &params.moduli);
    check_crt_consistency("ct.c1", &ct.c1, &params.moduli);

    println!("\n3. After decryption:");
    let pt_dec = rns_decrypt(&sk, &ct, &params);
    let consistent = check_crt_consistency("pt_dec.coeffs", &pt_dec.coeffs, &params.moduli);

    println!("\n4. Decode and verify:");
    let decoded = (pt_dec.coeffs.rns_coeffs[0][0] as f64) / ct.scale;
    println!("  Decoded: {}", decoded);
    println!("  Original: {}", value);
    println!("  Error: {}", (decoded - value).abs());

    if consistent && (decoded - value).abs() < 0.01 {
        println!("\n✅ SUCCESS - CRT consistent and value correct!");
    } else {
        println!("\n❌ FAILED - {} {}",
                 if !consistent { "CRT inconsistent" } else { "" },
                 if (decoded - value).abs() >= 0.01 { "value wrong" } else { "" });
    }
}

fn main() {
    println!("RNS Consistency Test: 2 vs 3 vs 10 primes");

    // Test with 2 primes
    let params_2 = CliffordFHEParams {
        n: 1024,
        moduli: vec![
            1141392289560813569,  // q₀
            1141392289560840193,  // q₁
        ],
        scale: 2f64.powi(40),
        error_std: 3.2,
        security: ga_engine::clifford_fhe::params::SecurityLevel::Bit128,
    };
    test_with_params(&params_2, "2 primes");

    // Test with 3 primes
    let params_3 = CliffordFHEParams {
        n: 1024,
        moduli: vec![
            1141392289560813569,  // q₀
            1141392289560840193,  // q₁
            1141392289560907777,  // q₂
        ],
        scale: 2f64.powi(40),
        error_std: 3.2,
        security: ga_engine::clifford_fhe::params::SecurityLevel::Bit128,
    };
    test_with_params(&params_3, "3 primes");

    // Test with 10 primes
    let params_10 = CliffordFHEParams::new_rns_mult();
    test_with_params(&params_10, "10 primes (new_rns_mult)");
}
