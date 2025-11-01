use ga_engine::clifford_ring_int::{CliffordPolynomialInt, CliffordRingElementInt};
use ga_engine::lazy_reduction::LazyReductionContext;
use ga_engine::ntt_optimized::OptimizedNTTContext;
use ga_engine::ntt_clifford_optimized::multiply_ntt_optimized;
use ga_engine::shake_poly::{discrete_poly_shake, error_poly_shake, generate_seed};

fn main() {
    let n = 64;
    let q = 3329i64;
    
    let ntt = OptimizedNTTContext::new_clifford_lwe_512();
    let lazy = LazyReductionContext::new(q);
    
    // Create a simple zero message
    let mut msg_coeffs = Vec::with_capacity(n);
    for _ in 0..n {
        msg_coeffs.push(CliffordRingElementInt::zero());
    }
    let message = CliffordPolynomialInt::new(msg_coeffs);
    
    println!("Message (all zeros):");
    println!("  First coeff: {:?}", message.coeffs[0]);
    
    // Simple keygen
    let seed_s = generate_seed();
    let mut s = discrete_poly_shake(&seed_s, n);
    s.reduce_modulo_xn_minus_1_lazy(n, &lazy);
    
    let seed_a = generate_seed();
    let mut a = discrete_poly_shake(&seed_a, n);
    a.reduce_modulo_xn_minus_1_lazy(n, &lazy);
    
    let seed_e = generate_seed();
    let mut e = error_poly_shake(&seed_e, n, 2);
    e.reduce_modulo_xn_minus_1_lazy(n, &lazy);
    
    let mut b = multiply_ntt_optimized(&a, &s, &ntt, &lazy);
    b.reduce_modulo_xn_minus_1_lazy(n, &lazy);
    b = b.add_lazy_poly(&e);
    
    // Simple encrypt
    let seed_r = generate_seed();
    let mut r = discrete_poly_shake(&seed_r, n);
    r.reduce_modulo_xn_minus_1_lazy(n, &lazy);
    
    let mut u = multiply_ntt_optimized(&a, &r, &ntt, &lazy);
    u.reduce_modulo_xn_minus_1_lazy(n, &lazy);
    
    let mut v = multiply_ntt_optimized(&b, &r, &ntt, &lazy);
    v.reduce_modulo_xn_minus_1_lazy(n, &lazy);
    v = v.add_lazy_poly(&message);
    
    println!("Ciphertext u first coeff: {:?}", u.coeffs[0]);
    println!("Ciphertext v first coeff: {:?}", v.coeffs[0]);
    
    // Simple decrypt
    let mut s_times_u = multiply_ntt_optimized(&s, &u, &ntt, &lazy);
    s_times_u.reduce_modulo_xn_minus_1_lazy(n, &lazy);
    
    let mut result = v.add_lazy_poly(&s_times_u.scalar_mul(-1, q));
    for coeff in &mut result.coeffs {
        *coeff = coeff.finalize_lazy(&lazy);
    }
    
    println!("Decrypted (before rounding) first coeff: {:?}", result.coeffs[0]);
    
    // Round
    let threshold_low = q / 4;
    let threshold_high = 3 * q / 4;
    
    for coeff in &mut result.coeffs {
        for i in 0..8 {
            let val = coeff.coeffs[i];
            coeff.coeffs[i] = if val >= threshold_low && val < threshold_high {
                1
            } else {
                0
            };
        }
    }
    
    println!("Decrypted (after rounding) first coeff: {:?}", result.coeffs[0]);
    println!("Expected (all zeros): {:?}", CliffordRingElementInt::zero());
    
    // Check correctness
    let mut correct = true;
    for i in 0..n {
        if result.coeffs[i] != message.coeffs[i] {
            println!("Mismatch at index {}: expected {:?}, got {:?}", i, message.coeffs[i], result.coeffs[i]);
            correct = false;
            break;
        }
    }
    
    if correct {
        println!("✓ PASS");
    } else {
        println!("✗ FAIL");
    }
}
