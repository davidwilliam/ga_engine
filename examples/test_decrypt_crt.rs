/// Test what value the decrypted RNS actually represents
use ga_engine::clifford_fhe::{
    ckks_rns::{rns_decrypt, rns_encrypt, RnsPlaintext},
    keys_rns::rns_keygen,
    params::CliffordFHEParams,
};

fn crt_reconstruct_2primes(r0: i64, r1: i64, p0: i64, p1: i64) -> f64 {
    // CRT reconstruction for 2 primes
    let r0 = r0 as i128;
    let r1 = r1 as i128;
    let p0 = p0 as i128;
    let p1 = p1 as i128;

    // Extended GCD to find modular inverse
    fn mod_inv(a: i128, m: i128) -> i128 {
        fn extended_gcd(a: i128, b: i128) -> (i128, i128, i128) {
            if a == 0 {
                (b, 0, 1)
            } else {
                let (g, x, y) = extended_gcd(b % a, a);
                (g, y - (b / a) * x, x)
            }
        }
        let (_, x, _) = extended_gcd(a, m);
        ((x % m) + m) % m
    }

    let p0_inv = mod_inv(p0, p1);
    let diff = ((r1 - r0) % p1 + p1) % p1;
    let factor = (diff * p0_inv) % p1;
    let x = r0 + p0 * factor;

    // Center around 0
    let p0p1 = p0 * p1;
    let x_centered = if x > p0p1 / 2 { x - p0p1 } else { x };

    x_centered as f64
}

fn main() {
    let params = CliffordFHEParams {
        n: 1024,
        moduli: vec![
            1141392289560813569,
            1141392289560840193,
        ],
        scale: 2f64.powi(40),
        error_std: 3.2,
        security: ga_engine::clifford_fhe::params::SecurityLevel::Bit128,
    };

    println!("Testing CRT reconstruction of decrypted plaintext\n");

    let (pk, sk, _evk) = rns_keygen(&params);

    // Encode value 1.5
    let value = 1.5;
    let scaled_value = (value * params.scale).round() as i64;
    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = scaled_value;

    println!("Original value: {}", value);
    println!("Scale: {}", params.scale);
    println!("Scaled (encoded): {}", scaled_value);
    println!();

    let pt = RnsPlaintext::from_coeffs(coeffs, params.scale, &params.moduli, 0);
    let ct = rns_encrypt(&pk, &pt, &params);
    let pt_dec = rns_decrypt(&sk, &ct, &params);

    println!("After decrypt, RNS residues:");
    println!("  r0 = {}", pt_dec.coeffs.rns_coeffs[0][0]);
    println!("  r1 = {}", pt_dec.coeffs.rns_coeffs[0][1]);
    println!();

    // Reconstruct using CRT
    let reconstructed = crt_reconstruct_2primes(
        pt_dec.coeffs.rns_coeffs[0][0],
        pt_dec.coeffs.rns_coeffs[0][1],
        params.moduli[0],
        params.moduli[1],
    );

    println!("CRT reconstruction: {}", reconstructed);
    println!("Decoded: {} / {} = {}", reconstructed, ct.scale, reconstructed / ct.scale);
    println!();
    println!("Expected (with noise): ~{} ± small noise", scaled_value);
    println!("Error: {}", (reconstructed - scaled_value as f64).abs());
    println!("Relative error: {:.6}%", 100.0 * (reconstructed - scaled_value as f64).abs() / scaled_value as f64);
}
