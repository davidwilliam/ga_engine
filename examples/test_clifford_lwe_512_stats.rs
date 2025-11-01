use ga_engine::clifford_ring_int::{CliffordPolynomialInt, CliffordRingElementInt};
use ga_engine::lazy_reduction::LazyReductionContext;
use ga_engine::ntt_optimized::OptimizedNTTContext;
use ga_engine::ntt_clifford_optimized::multiply_ntt_optimized;
use ga_engine::shake_poly::{discrete_poly_shake, error_poly_shake, generate_seed};

fn test_once(n: usize, q: i64, ntt: &OptimizedNTTContext, lazy: &LazyReductionContext) -> bool {
    // Create a simple zero message
    let mut msg_coeffs = Vec::with_capacity(n);
    for _ in 0..n {
        msg_coeffs.push(CliffordRingElementInt::zero());
    }
    let message = CliffordPolynomialInt::new(msg_coeffs);
    
    // Simple keygen
    let seed_s = generate_seed();
    let mut s = discrete_poly_shake(&seed_s, n);
    s.reduce_modulo_xn_minus_1_lazy(n, lazy);
    
    let seed_a = generate_seed();
    let mut a = discrete_poly_shake(&seed_a, n);
    a.reduce_modulo_xn_minus_1_lazy(n, lazy);
    
    let seed_e = generate_seed();
    let mut e = error_poly_shake(&seed_e, n, 2);
    e.reduce_modulo_xn_minus_1_lazy(n, lazy);
    
    let mut b = multiply_ntt_optimized(&a, &s, &ntt, lazy);
    b.reduce_modulo_xn_minus_1_lazy(n, lazy);
    b = b.add_lazy_poly(&e);
    
    // Simple encrypt
    let seed_r = generate_seed();
    let mut r = discrete_poly_shake(&seed_r, n);
    r.reduce_modulo_xn_minus_1_lazy(n, lazy);
    
    let mut u = multiply_ntt_optimized(&a, &r, &ntt, lazy);
    u.reduce_modulo_xn_minus_1_lazy(n, lazy);
    
    let mut v = multiply_ntt_optimized(&b, &r, &ntt, lazy);
    v.reduce_modulo_xn_minus_1_lazy(n, lazy);
    v = v.add_lazy_poly(&message);
    
    // Simple decrypt
    let mut s_times_u = multiply_ntt_optimized(&s, &u, &ntt, lazy);
    s_times_u.reduce_modulo_xn_minus_1_lazy(n, lazy);
    
    let mut result = v.add_lazy_poly(&s_times_u.scalar_mul(-1, q));
    for coeff in &mut result.coeffs {
        *coeff = coeff.finalize_lazy(lazy);
    }
    
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
    
    // Check correctness
    for i in 0..n {
        if result.coeffs[i] != message.coeffs[i] {
            return false;
        }
    }
    
    true
}

fn main() {
    let n = 64;
    let q = 3329i64;
    
    let ntt = OptimizedNTTContext::new_clifford_lwe_512();
    let lazy = LazyReductionContext::new(q);
    
    println!("Testing Clifford-LWE-512 correctness (10,000 trials)...");
    
    let mut successes = 0;
    let trials = 10000;
    
    for i in 0..trials {
        if test_once(n, q, &ntt, &lazy) {
            successes += 1;
        }
        
        if (i + 1) % 1000 == 0 {
            println!("  {} / {} trials ({:.2}% success rate so far)", 
                     i + 1, trials, (successes as f64 / (i + 1) as f64) * 100.0);
        }
    }
    
    println!();
    println!("Final results:");
    println!("  Successes: {} / {}", successes, trials);
    println!("  Success rate: {:.2}%", (successes as f64 / trials as f64) * 100.0);
    println!("  Failure rate: {:.2}%", ((trials - successes) as f64 / trials as f64) * 100.0);
    
    if successes == trials {
        println!("✓ PERFECT: 100% success rate!");
    } else if successes >= trials * 99 / 100 {
        println!("✓ EXCELLENT: >99% success rate");
    } else if successes >= trials * 95 / 100 {
        println!("⚠️  ACCEPTABLE: >95% success rate (some decryption errors)");
    } else {
        println!("✗ POOR: <95% success rate (too many errors)");
    }
}
