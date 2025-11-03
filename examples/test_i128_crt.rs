use ga_engine::clifford_fhe::{
    ckks_rns::{rns_decrypt, rns_encrypt, RnsPlaintext},
    keys_rns::rns_keygen,
    params::CliffordFHEParams,
};

fn main() {
    let params = CliffordFHEParams::new_rns_mult();
    let (pk, sk, _evk) = rns_keygen(&params);

    // Encode simple value: 1.5
    let value = 1.5;
    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = (value * params.scale).round() as i64;

    println!("Value: {}", value);
    println!("Scale: {}", params.scale);
    println!("Encoded coeffs[0]: {}", coeffs[0]);
    println!("Number of primes: {}", params.moduli.len());
    println!("First 3 primes: {}, {}, {}", params.moduli[0], params.moduli[1], params.moduli[2]);

    // Create plaintext
    let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);
    println!("\nAfter RnsPlaintext::from_coeffs:");
    println!("  pt.coeffs.rns_coeffs[0][0] = {}", pt.coeffs.rns_coeffs[0][0]);
    println!("  pt.coeffs.rns_coeffs[0][1] = {}", pt.coeffs.rns_coeffs[0][1]);
    println!("  pt.coeffs.rns_coeffs[0][2] = {}", pt.coeffs.rns_coeffs[0][2]);

    // Encrypt
    let ct = rns_encrypt(&pk, &pt, &params);

    // Decrypt
    let pt_dec = rns_decrypt(&sk, &ct, &params);
    println!("\nAfter rns_decrypt:");
    println!("  pt_dec.coeffs.rns_coeffs[0][0] = {}", pt_dec.coeffs.rns_coeffs[0][0]);
    println!("  pt_dec.coeffs.rns_coeffs[0][1] = {}", pt_dec.coeffs.rns_coeffs[0][1]);
    println!("  pt_dec.coeffs.rns_coeffs[0][2] = {}", pt_dec.coeffs.rns_coeffs[0][2]);
    println!("  ct.scale = {}", ct.scale);

    // Try the new i128 method
    let coeffs_i128 = pt_dec.to_coeffs_i128(&params.moduli);
    let decoded_i128 = (coeffs_i128[0] as f64) / ct.scale;
    println!("\nMethod: i128 CRT");
    println!("  coeffs_i128[0] = {}", coeffs_i128[0]);
    println!("  decoded_i128 = {}", decoded_i128);

    println!("\nOriginal: {}", value);
    println!("Error: {}", (decoded_i128 - value).abs());

    if (decoded_i128 - value).abs() < 0.01 {
        println!("\n✅ SUCCESS!");
    } else {
        println!("\n❌ FAILED!");
    }
}
