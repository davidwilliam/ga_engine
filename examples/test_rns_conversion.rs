//! Test RNS conversion (CRT) without encryption
//!
//! Verify that converting to/from RNS works correctly

use ga_engine::clifford_fhe::rns::RnsPolynomial;

fn main() {
    println!("Testing RNS Conversion (CRT)\n");

    let primes = vec![
        1_099_511_627_689,  // q₀
        1_099_511_627_691,  // q₁
        1_099_511_627_693,  // q₂
    ];

    let n = 1024;
    let level = 0;

    println!("Primes:");
    for (i, &q) in primes.iter().enumerate() {
        println!("  q{}: {}", i, q);
    }
    println!();

    // Test with small values
    let original = vec![42i64, -17, 0, 100, -200];
    let mut coeffs = vec![0i64; n];
    for (i, &val) in original.iter().enumerate() {
        coeffs[i] = val;
    }

    println!("Original coefficients (first 5): {:?}", &coeffs[..5]);

    // Convert to RNS
    let rns_poly = RnsPolynomial::from_coeffs(&coeffs, &primes, n, level);

    println!("\nRNS representation of coeff[0] = 42:");
    for (j, &q) in primes.iter().enumerate() {
        println!("  42 mod q{}: {}", j, rns_poly.rns_coeffs[0][j]);
    }

    // Convert back
    let recovered = rns_poly.to_coeffs(&primes);

    println!("\nRecovered coefficients (first 5): {:?}", &recovered[..5]);

    // Check errors
    println!("\nErrors:");
    for i in 0..5 {
        let error = (coeffs[i] - recovered[i]).abs();
        println!("  coeff[{}]: original={}, recovered={}, error={}",
                 i, coeffs[i], recovered[i], error);
    }

    // Check if all match
    let all_match = coeffs.iter().zip(&recovered).all(|(a, b)| a == b);

    if all_match {
        println!("\n✓ PASS: All coefficients recovered correctly!");
    } else {
        println!("\n✗ FAIL: Coefficients don't match!");
    }
}
