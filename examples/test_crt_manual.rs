// Manual test of CRT formula with explicit values

fn mod_inverse(a: i128, m: i128) -> i128 {
    let (mut t, mut new_t) = (0i128, 1i128);
    let (mut r, mut new_r) = (m, a % m);

    while new_r != 0 {
        let quotient = r / new_r;
        let temp = new_t;
        new_t = t - quotient * new_t;
        t = temp;
        let temp = new_r;
        new_r = r - quotient * new_r;
        r = temp;
    }

    if r > 1 {
        panic!("a is not invertible mod m");
    }
    if t < 0 {
        t += m;
    }

    t
}

fn main() {
    // Simple test: reconstruct 1649267441664 from its residues
    let original_value: i128 = 1649267441664;
    let p: i128 = 1141392289560813569;
    let q: i128 = 1141392289560840193;

    // Compute residues
    let a = original_value % p;
    let b = original_value % q;

    println!("Original value: {}", original_value);
    println!("p = {}", p);
    println!("q = {}", q);
    println!("a = {} (residue mod p)", a);
    println!("b = {} (residue mod q)", b);

    // Apply CRT
    let p_inv_mod_q = mod_inverse(p, q);
    println!("\np^{{-1}} mod q = {}", p_inv_mod_q);

    // Verify: p * p_inv ≡ 1 (mod q)
    let check = (p * p_inv_mod_q) % q;
    println!("Verification: (p * p^{{-1}}) mod q = {} (should be 1)", check);

    // CRT formula: x = a + p * ((b - a) * p^{-1} mod q)
    let diff = ((b - a) % q + q) % q;
    let factor = (diff * p_inv_mod_q) % q;
    println!("\ndiff = (b - a) mod q = {}", diff);
    println!("factor = (diff * p_inv) mod q = {}", factor);

    let p_times_factor = p * factor;
    println!("p * factor = {}", p_times_factor);

    let x = a + p_times_factor;
    println!("x = a + p * factor = {}", x);

    // Compute pq
    let pq = p * q;
    println!("\npq = {}", pq);

    // Should x be in [0, pq)?
    if x >= 0 && x < pq {
        println!("✓ x is in [0, pq)");
    } else {
        println!("✗ x is NOT in [0, pq) - something is wrong!");
    }

    // Verify x ≡ a (mod p) and x ≡ b (mod q)
    println!("\nVerification:");
    println!("x mod p = {} (should be {})", x % p, a);
    println!("x mod q = {} (should be {})", x % q, b);

    if x % p == a && x % q == b {
        println!("✅ CRT reconstruction successful!");
        println!("Reconstructed: {}", x);
        println!("Original: {}", original_value);
        println!("Match: {}", x == original_value);
    } else {
        println!("❌ CRT reconstruction FAILED!");
    }
}
