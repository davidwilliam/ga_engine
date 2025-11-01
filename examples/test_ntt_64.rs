// Test if NTT is working correctly for N=64
use ga_engine::ntt::NTTContext;

fn main() {
    let ntt = NTTContext::new_clifford_lwe_512();
    
    println!("Testing NTT for N=64:");
    println!("  ω = {}, ω^(-1) = {}, N^(-1) = {}", ntt.omega, ntt.omega_inv, ntt.n_inv);
    
    // Test with a simple polynomial
    let mut a = vec![0i64; 64];
    a[0] = 1;  // Polynomial 1 + 0x + 0x^2 + ...
    
    let original = a.clone();
    
    // Forward NTT
    ntt.forward(&mut a);
    
    // Inverse NTT
    ntt.inverse(&mut a);
    
    // Check if we got back the original
    let mut correct = true;
    for i in 0..64 {
        if a[i] != original[i] {
            correct = false;
            println!("  Mismatch at index {}: expected {}, got {}", i, original[i], a[i]);
            break;
        }
    }
    
    if correct {
        println!("✓ NTT round-trip test PASSED");
    } else {
        println!("✗ NTT round-trip test FAILED");
    }
}
