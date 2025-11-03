/// Detailed trace of decryption to find the bug
use ga_engine::clifford_fhe::{
    ckks_rns::{rns_decrypt, rns_encrypt, RnsPlaintext},
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
                let wrapped_idx = idx % n;
                result[wrapped_idx] -= prod;
            }
        }
    }

    result.iter().map(|&x| {
        let r = x % q128;
        if r < 0 {
            (r + q128) as i64
        } else {
            r as i64
        }
    }).collect()
}

fn main() {
    let params = CliffordFHEParams {
        n: 8, // SMALL for manual verification
        moduli: vec![97, 101], // Small primes for hand calculation
        scale: 16.0, // Small scale
        error_std: 0.1, // Minimal noise
        security: ga_engine::clifford_fhe::params::SecurityLevel::Bit128,
    };

    println!("=== DETAILED DECRYPTION TRACE ===");
    println!("N={}, primes={:?}, scale={}", params.n, params.moduli, params.scale);
    println!();

    let (pk, sk, _evk) = rns_keygen(&params);

    // Encrypt value 3.0
    let value = 3.0;
    let scaled = (value * params.scale).round() as i64;
    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = scaled;

    println!("1. PLAINTEXT:");
    println!("   value = {}", value);
    println!("   scaled = {} (value * scale)", scaled);
    println!("   coeffs[0] = {}", coeffs[0]);
    println!();

    let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);
    println!("2. RNS ENCODING:");
    println!("   pt.rns_coeffs[0] = {:?}", pt.coeffs.rns_coeffs[0]);
    println!("   (should be [{} mod 97, {} mod 101] = [{}, {}])",
             scaled, scaled, scaled % 97, scaled % 101);
    println!();

    let ct = rns_encrypt(&pk, &pt, &params);
    println!("3. CIPHERTEXT:");
    println!("   ct.c0.rns_coeffs[0] = {:?}", ct.c0.rns_coeffs[0]);
    println!("   ct.c1.rns_coeffs[0] = {:?}", ct.c1.rns_coeffs[0]);
    println!();

    // Manual decryption step-by-step
    println!("4. MANUAL DECRYPTION:");
    println!("   Formula: m' = c0 + c1*s");
    println!();

    println!("   Secret key (first 4 coeffs):");
    for i in 0..4.min(params.n) {
        println!("     sk.coeffs[{}] = {:?}", i, sk.coeffs.rns_coeffs[i]);
    }
    println!();

    // Compute c1*s manually
    println!("   Computing c1*s:");
    let c1s = rns_multiply(&ct.c1, &sk.coeffs, &params.moduli, polynomial_multiply_ntt);
    println!("     c1s.rns_coeffs[0] = {:?}", c1s.rns_coeffs[0]);
    println!();

    // Compute c0 + c1*s
    println!("   Computing m' = c0 + c1*s:");
    let m_prime = rns_add(&ct.c0, &c1s, &params.moduli);
    println!("     m'.rns_coeffs[0] = {:?}", m_prime.rns_coeffs[0]);
    println!();

    // Try different decoding methods
    println!("5. DECODING:");

    // Method 1: Direct first prime
    let r0 = m_prime.rns_coeffs[0][0];
    let p0 = params.moduli[0];
    let centered0 = if r0 > p0 / 2 { r0 - p0 } else { r0 };
    let decoded1 = (centered0 as f64) / ct.scale;
    println!("   Method 1 (prime 0 only): {} / {} = {}", centered0, ct.scale, decoded1);

    // Method 2: CRT reconstruction
    let coeffs_crt = m_prime.to_coeffs(&params.moduli);
    let decoded2 = (coeffs_crt[0] as f64) / ct.scale;
    println!("   Method 2 (CRT): {} / {} = {}", coeffs_crt[0], ct.scale, decoded2);

    println!();
    println!("6. RESULT:");
    println!("   Original value: {}", value);
    println!("   Decoded (method 1): {}", decoded1);
    println!("   Decoded (method 2): {}", decoded2);
    println!("   Error 1: {}", (decoded1 - value).abs());
    println!("   Error 2: {}", (decoded2 - value).abs());

    if (decoded2 - value).abs() < 0.5 {
        println!("\n✅ SUCCESS!");
    } else {
        println!("\n❌ FAILED - need to debug further");
    }
}
