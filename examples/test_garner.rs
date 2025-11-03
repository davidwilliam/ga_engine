use ga_engine::clifford_fhe::{
    ckks_rns::{rns_encrypt, RnsPlaintext},
    keys_rns::rns_keygen,
    params::CliffordFHEParams,
    rns::{rns_add, rns_multiply},
};

fn polynomial_multiply_ntt(a: &[i64], b: &[i64], q: i64, n: usize) -> Vec<i64> {
    let mut result = vec![0i128; n];
    let q128 = q as i128;
    for i in 0..n {
        for j in 0..n {
            let idx = i + j;
            let prod = (a[i] as i128) * (b[j] as i128);
            if idx < n { result[idx] += prod; } else { result[idx - n] -= prod; }
        }
    }
    result.iter().map(|&x| {
        let r = x % q128;
        if r < 0 { (r + q128) as i64 } else { r as i64 }
    }).collect()
}

fn main() {
    let params = CliffordFHEParams::new_rns_mult();
    println!("Testing Garner's algorithm with {} primes\n", params.moduli.len());

    let (pk, sk, _evk) = rns_keygen(&params);

    let value = 1.5;
    let scaled = (value * params.scale).round() as i64;
    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = scaled;

    let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);
    let ct = rns_encrypt(&pk, &pt, &params);

    let c1s = rns_multiply(&ct.c1, &sk.coeffs, &params.moduli, polynomial_multiply_ntt);
    let m_prime = rns_add(&ct.c0, &c1s, &params.moduli);

    // Method 1: Old to_coeffs (i128 - OVERFLOWS)
    println!("Method 1: to_coeffs() [i128 - will overflow]");
    let coeffs_old = m_prime.to_coeffs(&params.moduli);
    let decoded_old = (coeffs_old[0] as f64) / ct.scale;
    println!("  Decoded: {}, Error: {}\n", decoded_old, (decoded_old - value).abs());

    // Method 2: Garner's algorithm (f64)
    println!("Method 2: to_coeffs_crt_centered() [Garner's algorithm with f64]");
    let coeffs_garner = m_prime.to_coeffs_crt_centered(&params.moduli);
    let decoded_garner = coeffs_garner[0] / ct.scale;
    println!("  Decoded: {}, Error: {}\n", decoded_garner, (decoded_garner - value).abs());

    println!("Expected: {}", value);
    if (decoded_garner - value).abs() < 0.01 {
        println!("✅ SUCCESS with Garner's algorithm!");
    } else {
        println!("❌ Still failing");
    }
}
