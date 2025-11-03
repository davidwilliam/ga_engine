/// Test if key generation produces CRT-consistent keys
use ga_engine::clifford_fhe::{
    keys_rns::rns_keygen,
    params::CliffordFHEParams,
    rns::RnsPolynomial,
};

fn check_crt_consistency(name: &str, poly: &RnsPolynomial, primes: &[i64]) -> bool {
    if poly.num_primes() < 2 {
        return true;
    }

    let r0 = poly.rns_coeffs[0][0] as i128;
    let r1 = poly.rns_coeffs[0][1] as i128;
    let p0 = primes[0] as i128;
    let p1 = primes[1] as i128;

    let r0_mod_p1 = ((r0 % p1) + p1) % p1;
    let r1_mod_p1 = ((r1 % p1) + p1) % p1;

    if r0_mod_p1 != r1_mod_p1 {
        println!("  {} - ❌ INCONSISTENT", name);
        println!("    r0={} mod p1 = {}", r0, r0_mod_p1);
        println!("    r1={}", r1);
        return false;
    }

    println!("  {} - ✓ Consistent", name);
    true
}

fn main() {
    println!("Testing key generation CRT consistency\n");

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

    println!("Generating keys...");
    let (pk, sk, _evk) = rns_keygen(&params);

    println!("\nChecking secret key:");
    check_crt_consistency("sk.coeffs", &sk.coeffs, &params.moduli);

    println!("\nChecking public key:");
    check_crt_consistency("pk.a", &pk.a, &params.moduli);
    check_crt_consistency("pk.b", &pk.b, &params.moduli);

    println!("\nDone!");
}
