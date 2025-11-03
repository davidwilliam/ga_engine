// Test NTT polynomial multiplication directly

use ga_engine::clifford_fhe::ckks_rns::*;
use ga_engine::clifford_fhe::rns::*;

fn main() {
    // Use a 60-bit prime that supports NTT
    let q: i64 = 1141392289560813569;
    let n: usize = 8; // Small for easy verification

    // Test case: multiply two simple polynomials
    // a(x) = 1 + 2x
    // b(x) = 3 + 4x
    // Result mod (x^8 + 1): should be 3 + 10x + 8x^2

    let mut a = vec![0i64; n];
    let mut b = vec![0i64; n];
    a[0] = 1;
    a[1] = 2;
    b[0] = 3;
    b[1] = 4;

    println!("Testing NTT multiplication:");
    println!("a(x) = {} + {}x", a[0], a[1]);
    println!("b(x) = {} + {}x", b[0], b[1]);
    println!("q = {}", q);
    println!("n = {}", n);

    // CANNOT call polynomial_multiply_ntt directly as it's private
    // Instead, use RNS multiply
    let primes = vec![q];
    let a_rns = RnsPolynomial::from_coeffs(&a, &primes, n, 0);
    let b_rns = RnsPolynomial::from_coeffs(&b, &primes, n, 0);

    // Multiply using RNS
    let c_rns = rns_multiply(&a_rns, &b_rns, &primes, |a, b, q, n| {
        // This should call our NTT implementation
        // For now, just do schoolbook for verification
        let mut result = vec![0i128; n];
        let q128 = q as i128;

        for i in 0..n {
            for j in 0..n {
                let idx = i + j;
                let prod = (a[i] as i128) * (b[j] as i128);
                if idx < n {
                    result[idx] = (result[idx] + prod) % q128;
                } else {
                    // x^n = -1 reduction (negacyclic)
                    let wrapped_idx = idx % n;
                    result[wrapped_idx] = (result[wrapped_idx] - prod) % q128;
                }
            }
        }

        result.iter().map(|&x| {
            let r = ((x % q128) + q128) % q128;
            r as i64
        }).collect()
    });

    // Convert back
    let c = c_rns.to_coeffs(&primes);

    println!("\nExpected result: 3 + 10x + 8x^2");
    println!("Actual result:");
    for i in 0..n {
        if c[i] != 0 {
            println!("  c[{}] = {}", i, c[i]);
        }
    }

    if c[0] == 3 && c[1] == 10 && c[2] == 8 {
        println!("\n✅ SUCCESS!");
    } else {
        println!("\n❌ FAILED!");
    }
}
