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
    println!("Expected: {}", (value * params.scale).round() as i64);

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
    println!("  ct.scale = {}", ct.scale);

    // Method 1: Direct first residue (WRONG)
    let decoded_wrong = (pt_dec.coeffs.rns_coeffs[0][0] as f64) / ct.scale;
    println!("\nMethod 1 (direct residue): {}", decoded_wrong);

    // Method 2: Full CRT reconstruction
    let coeffs_crt = pt_dec.to_coeffs(&params.moduli);
    let decoded_crt = (coeffs_crt[0] as f64) / ct.scale;
    println!("Method 2 (full CRT): {}", decoded_crt);

    // Method 3: Single prime extraction (RECOMMENDED)
    let coeffs_single = pt_dec.to_coeffs_single_prime(&params.moduli);
    let decoded_single = (coeffs_single[0] as f64) / ct.scale;
    println!("Method 3 (single prime - RECOMMENDED): {}", decoded_single);

    println!("\nOriginal: {}", value);
    println!("Error (method 1): {}", (decoded_wrong - value).abs());
    println!("Error (method 2): {}", (decoded_crt - value).abs());
    println!("Error (method 3): {}", (decoded_single - value).abs());

    if (decoded_single - value).abs() < 0.01 {
        println!("\n✅ SUCCESS!");
    } else {
        println!("\n❌ FAILED!");
    }
}
