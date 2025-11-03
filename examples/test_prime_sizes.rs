/// Test encryption/decryption with different prime sizes to find overflow
use ga_engine::clifford_fhe::{
    ckks_rns::{rns_encrypt, RnsPlaintext},
    keys_rns::rns_keygen,
    params::CliffordFHEParams,
    rns::{rns_add, rns_multiply, RnsPolynomial},
};

fn polynomial_multiply_ntt(a: &[i64], b: &[i64], q: i64, n: usize) -> Vec<i64> {
    let mut result = vec![0i128; n];
    let q128 = q as i128;
    for i in 0..n {
        for j in 0..n {
            let idx = i + j;
            let prod = (a[i] as i128) * (b[j] as i128);
            if idx < n {
                result[idx] += prod;
            } else {
                result[idx - n] -= prod;
            }
        }
    }
    result.iter().map(|&x| {
        let r = x % q128;
        if r < 0 { (r + q128) as i64 } else { r as i64 }
    }).collect()
}

fn test_params(name: &str, params: &CliffordFHEParams) {
    println!("\n{}", "=".repeat(70));
    println!("Testing: {}", name);
    println!("N={}, primes={:?}", params.n, params.moduli);
    println!("Prime bits: {:?}", params.moduli.iter().map(|p| ((*p as f64).log2() as u32)).collect::<Vec<_>>());
    println!("{}", "=".repeat(70));

    let (pk, sk, _evk) = rns_keygen(&params);

    let value = 3.0;
    let scaled = (value * params.scale).round() as i64;
    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = scaled;

    let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);
    let ct = rns_encrypt(&pk, &pt, &params);

    // Decrypt manually
    let c1s = rns_multiply(&ct.c1, &sk.coeffs, &params.moduli, polynomial_multiply_ntt);
    let m_prime = rns_add(&ct.c0, &c1s, &params.moduli);

    // Decode with CRT
    let coeffs_crt = m_prime.to_coeffs(&params.moduli);
    let decoded = (coeffs_crt[0] as f64) / ct.scale;

    let error = (decoded - value).abs();
    let relative_error = error / value;

    println!("Value: {} → Decoded: {}", value, decoded);
    println!("Error: {} ({:.2}%)", error, relative_error * 100.0);

    if error < 0.5 {
        println!("✅ PASS");
    } else {
        println!("❌ FAIL");
        println!("  Scaled value: {}", scaled);
        println!("  m'.rns_coeffs[0]: {:?}", &m_prime.rns_coeffs[0]);
        println!("  CRT result: {}", coeffs_crt[0]);
    }
}

fn main() {
    println!("Testing RNS-CKKS with different prime sizes");

    // Test 1: Small primes (7-bit)
    test_params("Small primes (7-bit)", &CliffordFHEParams {
        n: 8,
        moduli: vec![97, 101],
        scale: 16.0,
        error_std: 0.1,
        security: ga_engine::clifford_fhe::params::SecurityLevel::Bit128,
    });

    // Test 2: Medium primes (16-bit)
    test_params("Medium primes (16-bit)", &CliffordFHEParams {
        n: 64,
        moduli: vec![65521, 65519], // Largest 16-bit primes
        scale: 256.0,
        error_std: 3.2,
        security: ga_engine::clifford_fhe::params::SecurityLevel::Bit128,
    });

    // Test 3: Large primes (30-bit)
    test_params("Large primes (30-bit)", &CliffordFHEParams {
        n: 512,
        moduli: vec![1073741789, 1073741783], // ~30-bit primes
        scale: 2f64.powi(20),
        error_std: 3.2,
        security: ga_engine::clifford_fhe::params::SecurityLevel::Bit128,
    });

    // Test 4: Very large primes (40-bit)
    test_params("Very large primes (40-bit)", &CliffordFHEParams {
        n: 1024,
        moduli: vec![1099511627689, 1099511627691], // ~40-bit primes
        scale: 2f64.powi(30),
        error_std: 3.2,
        security: ga_engine::clifford_fhe::params::SecurityLevel::Bit128,
    });

    // Test 5: Production primes (60-bit) - THIS SHOULD FAIL
    test_params("Production primes (60-bit) - EXPECTED TO FAIL", &CliffordFHEParams {
        n: 1024,
        moduli: vec![1141392289560813569, 1141392289560840193],
        scale: 2f64.powi(40),
        error_std: 3.2,
        security: ga_engine::clifford_fhe::params::SecurityLevel::Bit128,
    });

    println!("\n{}", "=".repeat(70));
    println!("CONCLUSION:");
    println!("Find the largest prime size that works, then investigate overflow.");
    println!("{}", "=".repeat(70));
}
