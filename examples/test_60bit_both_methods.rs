use ga_engine::clifford_fhe::{
    ckks_rns::{rns_decrypt, rns_encrypt, RnsPlaintext},
    keys_rns::rns_keygen,
    params::CliffordFHEParams,
};

fn main() {
    // Use 60-bit primes
    let params = CliffordFHEParams {
        n: 1024,
        moduli: vec![1141392289560813569, 1141392289560840193], // ~60-bit primes
        scale: 2f64.powi(40),
        error_std: 3.2,
        security: ga_engine::clifford_fhe::params::SecurityLevel::Bit128,
    };

    let (pk, sk, _evk) = rns_keygen(&params);

    // Encode simple value: 1.5
    let value = 1.5;
    let mut coeffs = vec![0i64; params.n];
    coeffs[0] = (value * params.scale).round() as i64;

    println!("Value: {}", value);
    println!("Scale: {}", params.scale);
    println!("Encoded coeffs[0]: {}", coeffs[0]);

    // Create plaintext
    let pt = RnsPlaintext::from_coeffs(coeffs.clone(), params.scale, &params.moduli, 0);

    // Encrypt
    let ct = rns_encrypt(&pk, &pt, &params);

    // Decrypt
    let pt_dec = rns_decrypt(&sk, &ct, &params);

    println!("\nDecrypted residues:");
    println!("  pt_dec.coeffs.rns_coeffs[0][0] = {}", pt_dec.coeffs.rns_coeffs[0][0]);
    println!("  pt_dec.coeffs.rns_coeffs[0][1] = {}", pt_dec.coeffs.rns_coeffs[0][1]);

    println!("\nFor comparison, original encoded:");
    println!("  Original coeffs[0] = {}", coeffs[0]);
    println!("  Noise in first residue: {}", (pt_dec.coeffs.rns_coeffs[0][0] as i128 - coeffs[0] as i128).abs());

    // Method 1: Old to_coeffs() returning i64
    let coeffs_i64 = pt_dec.to_coeffs(&params.moduli);
    let decoded_i64 = (coeffs_i64[0] as f64) / ct.scale;
    println!("\nMethod 1 (old to_coeffs -> i64):");
    println!("  coeffs[0] = {}", coeffs_i64[0]);
    println!("  decoded = {}", decoded_i64);
    println!("  error = {}", (decoded_i64 - value).abs());

    // Method 2: New to_coeffs_i128()
    let coeffs_i128 = pt_dec.to_coeffs_i128(&params.moduli);
    let decoded_i128 = (coeffs_i128[0] as f64) / ct.scale;
    println!("\nMethod 2 (new to_coeffs_i128 -> i128):");
    println!("  coeffs[0] = {}", coeffs_i128[0]);
    println!("  decoded = {}", decoded_i128);
    println!("  error = {}", (decoded_i128 - value).abs());

    println!("\nOriginal: {}", value);

    if (decoded_i128 - value).abs() < 0.01 {
        println!("\n✅ SUCCESS!");
    } else {
        println!("\n❌ FAILED!");
    }
}
