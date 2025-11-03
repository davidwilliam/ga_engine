/// Test RNS polynomial multiplication
use ga_engine::clifford_fhe::rns::RnsPolynomial;

// Simple polynomial multiplication (schoolbook)
fn polynomial_multiply_simple(a: &[i64], b: &[i64], q: i64, n: usize) -> Vec<i64> {
    let mut result = vec![0i128; n];
    let q128 = q as i128;

    for i in 0..n {
        for j in 0..n {
            let idx = i + j;
            let prod = (a[i] as i128) * (b[j] as i128);
            if idx < n {
                result[idx] += prod;
            } else {
                // x^n = -1 reduction (negacyclic)
                let wrapped_idx = idx % n;
                result[wrapped_idx] -= prod;
            }
        }
    }

    // Reduce modulo q
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
    println!("Testing RNS polynomial multiplication\n");

    let n = 8; // Small for testing
    let primes = vec![1141392289560813569i64, 1141392289560840193i64];

    // Test 1: Multiply by 1 (identity)
    println!("Test 1: Multiply [1, 2, 3, ...] by [1, 0, 0, ...]");
    let a_coeffs = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let b_coeffs = vec![1, 0, 0, 0, 0, 0, 0, 0]; // Identity: just the constant 1

    let a_rns = RnsPolynomial::from_coeffs(&a_coeffs, &primes, n, 0);
    let b_rns = RnsPolynomial::from_coeffs(&b_coeffs, &primes, n, 0);

    println!("Input a:");
    println!("  coeffs: {:?}", a_coeffs);
    println!("  RNS[0]: {:?}", a_rns.rns_coeffs[0]);
    println!("  RNS[1]: {:?}", a_rns.rns_coeffs[1]);

    println!("\nInput b:");
    println!("  coeffs: {:?}", b_coeffs);
    println!("  RNS[0]: {:?}", b_rns.rns_coeffs[0]);

    // Multiply using RNS
    use ga_engine::clifford_fhe::rns::rns_multiply;
    let c_rns = rns_multiply(&a_rns, &b_rns, &primes, polynomial_multiply_simple);

    println!("\nResult c = a * b:");
    println!("  RNS[0]: {:?}", c_rns.rns_coeffs[0]);
    println!("  RNS[1]: {:?}", c_rns.rns_coeffs[1]);

    // Convert back to regular coefficients
    let c_coeffs = c_rns.to_coeffs(&primes);
    println!("  Regular coeffs: {:?}", c_coeffs);
    println!("\nExpected: {:?} (same as a)", a_coeffs);

    if c_coeffs == a_coeffs {
        println!("✅ PASS");
    } else {
        println!("❌ FAIL");
    }

    // Test 2: Multiply [1] by [-1]
    println!("\n{}", "=".repeat(70));
    println!("Test 2: Multiply [1, 0, ...] by [-1, 0, ...]");
    let a2_coeffs = vec![1, 0, 0, 0, 0, 0, 0, 0];
    let b2_coeffs = vec![-1, 0, 0, 0, 0, 0, 0, 0];

    let a2_rns = RnsPolynomial::from_coeffs(&a2_coeffs, &primes, n, 0);
    let b2_rns = RnsPolynomial::from_coeffs(&b2_coeffs, &primes, n, 0);

    println!("Input a: {:?}", a2_coeffs);
    println!("  a RNS[0]: {:?}", a2_rns.rns_coeffs[0]);

    println!("\nInput b: {:?}", b2_coeffs);
    println!("  b RNS[0]: {:?}", b2_rns.rns_coeffs[0]);

    let c2_rns = rns_multiply(&a2_rns, &b2_rns, &primes, polynomial_multiply_simple);

    println!("\nResult c = a * b:");
    println!("  c RNS[0]: {:?}", c2_rns.rns_coeffs[0]);

    let c2_coeffs = c2_rns.to_coeffs(&primes);
    println!("  Regular coeffs: {:?}", c2_coeffs);
    println!("\nExpected: [-1, 0, 0, ...]");

    if c2_coeffs[0] == -1 && c2_coeffs[1..].iter().all(|&x| x == 0) {
        println!("✅ PASS");
    } else {
        println!("❌ FAIL");
    }
}
