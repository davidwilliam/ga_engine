// Simple test to verify NTT works correctly

fn mod_mul_u64(a: u64, b: u64, q: u64) -> u64 {
    let p = (a as u128) * (b as u128);
    (p % (q as u128)) as u64
}

fn mod_pow_u64(mut base: u64, mut exp: u64, q: u64) -> u64 {
    let mut acc = 1u64;
    while exp > 0 {
        if (exp & 1) == 1 { acc = mod_mul_u64(acc, base, q); }
        base = mod_mul_u64(base, base, q);
        exp >>= 1;
    }
    acc
}

fn main() {
    // Test with a 60-bit prime
    let q: i64 = 1141392289560813569;
    let n: usize = 1024;

    println!("Testing NTT parameters:");
    println!("q = {}", q);
    println!("n = {}", n);
    println!("q-1 = {}", q - 1);
    println!("2*n = {}", 2 * n as i64);

    // Check if q-1 is divisible by 2N
    if (q - 1) % (2 * n as i64) == 0 {
        println!("✓ (q-1) is divisible by 2N");
        println!("  (q-1) / 2N = {}", (q - 1) / (2 * n as i64));
    } else {
        println!("✗ (q-1) is NOT divisible by 2N!");
        return;
    }

    // Try to find primitive root
    println!("\nSearching for primitive root...");
    let q_u64 = q as u64;
    let phi = q_u64 - 1;
    let mut odd = phi;
    while odd % 2 == 0 { odd /= 2; }

    for g in 2..100 {
        if mod_pow_u64(g, phi/2, q_u64) == 1 { continue; }
        if odd != 1 && mod_pow_u64(g, phi/odd, q_u64) == 1 { continue; }
        println!("Found primitive root: g = {}", g);

        // Compute psi
        let exp = phi / (2 * n as u64);
        let psi = mod_pow_u64(g, exp, q_u64);
        println!("psi = g^{} mod q = {}", exp, psi);

        // Verify psi^(2N) = 1
        let test1 = mod_pow_u64(psi, 2 * n as u64, q_u64);
        println!("psi^(2N) mod q = {} (should be 1)", test1);

        // Verify psi^N = -1 (mod q) = q-1
        let test2 = mod_pow_u64(psi, n as u64, q_u64);
        println!("psi^N mod q = {} (should be {})", test2, q_u64 - 1);

        if test1 == 1 && test2 == q_u64 - 1 {
            println!("✅ psi is a valid 2N-th root of unity!");
        } else {
            println!("❌ psi is INVALID!");
        }

        break;
    }
}
