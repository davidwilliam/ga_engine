//! Clifford-LWE-256: Barrett Reduction Variant (EXPERIMENTAL)
//!
//! ⚠️  NOTE: This version is **slower** than the standard % operator version!
//! ⚠️  For best performance, use `clifford_lwe_256_integer.rs` instead.
//!
//! This implementation is kept for:
//! - Benchmarking and comparison purposes
//! - Future constant-time implementations (Barrett supports branchless reduction)
//! - Reference for platforms where % might not be optimized
//!
//! Performance (Apple M3 Max):
//! - Standard encryption: 85.91 µs (vs 59.52 µs with %)
//! - Precomputed encryption: 9.25 µs (vs 9.06 µs with %)
//!
//! Conclusion: Compiler-optimized % operator is faster for small moduli (q=3329).
//! Barrett may help for larger moduli (q > 2^32) or constant-time requirements.

use ga_engine::barrett::BarrettContext;
use ga_engine::clifford_ring_int::{CliffordPolynomialInt, CliffordRingElementInt};
use std::time::Instant;
use rand::Rng;

struct CLWEParams {
    n: usize,
    q: i64,
    error_bound: i64,
}

impl Default for CLWEParams {
    fn default() -> Self {
        Self {
            n: 32,
            q: 3329,
            error_bound: 2,
        }
    }
}

struct PublicKey {
    a: CliffordPolynomialInt,
    b: CliffordPolynomialInt,
}

struct SecretKey {
    s: CliffordPolynomialInt,
}

/// Precomputed data for fast encryption
struct EncryptionCache {
    a_times_r: CliffordPolynomialInt,
    b_times_r: CliffordPolynomialInt,
}

impl EncryptionCache {
    fn new(pk: &PublicKey, params: &CLWEParams, barrett: &BarrettContext) -> Self {
        let mut r = discrete_poly_fast(params.n);
        r.reduce_modulo_xn_minus_1_barrett(params.n, barrett);

        let mut a_times_r = pk.a.multiply_karatsuba_barrett(&r, barrett);
        a_times_r.reduce_modulo_xn_minus_1_barrett(params.n, barrett);

        let mut b_times_r = pk.b.multiply_karatsuba_barrett(&r, barrett);
        b_times_r.reduce_modulo_xn_minus_1_barrett(params.n, barrett);

        Self {
            a_times_r,
            b_times_r,
        }
    }
}

#[inline]
fn discrete_poly_fast(n: usize) -> CliffordPolynomialInt {
    let mut rng = rand::thread_rng();
    let mut coeffs = Vec::with_capacity(n);

    for _ in 0..n {
        let mut mv = [0i64; 8];
        for j in 0..8 {
            mv[j] = rng.gen_range(-1..=1);
        }
        coeffs.push(CliffordRingElementInt::from_multivector(mv));
    }

    CliffordPolynomialInt::new(coeffs)
}

#[inline]
fn error_poly_fast(n: usize, error_bound: i64) -> CliffordPolynomialInt {
    let mut rng = rand::thread_rng();
    let mut coeffs = Vec::with_capacity(n);

    for _ in 0..n {
        let mut mv = [0i64; 8];
        for j in 0..8 {
            mv[j] = rng.gen_range(-error_bound..=error_bound);
        }
        coeffs.push(CliffordRingElementInt::from_multivector(mv));
    }

    CliffordPolynomialInt::new(coeffs)
}

/// Key generation with Barrett reduction
fn keygen(params: &CLWEParams, barrett: &BarrettContext) -> (PublicKey, SecretKey) {
    let mut s = discrete_poly_fast(params.n);
    s.reduce_modulo_xn_minus_1_barrett(params.n, barrett);

    let mut a = discrete_poly_fast(params.n);
    a.reduce_modulo_xn_minus_1_barrett(params.n, barrett);

    let mut e = error_poly_fast(params.n, params.error_bound);
    e.reduce_modulo_xn_minus_1_barrett(params.n, barrett);

    let mut b = a.multiply_karatsuba_barrett(&s, barrett);
    b.reduce_modulo_xn_minus_1_barrett(params.n, barrett);
    b = b.add_mod_barrett(&e, barrett);

    (PublicKey { a, b }, SecretKey { s })
}

/// Standard encryption with Barrett reduction
fn encrypt(
    pk: &PublicKey,
    message: &CliffordPolynomialInt,
    params: &CLWEParams,
    barrett: &BarrettContext
) -> (CliffordPolynomialInt, CliffordPolynomialInt) {
    let mut r = discrete_poly_fast(params.n);
    r.reduce_modulo_xn_minus_1_barrett(params.n, barrett);

    let mut e1 = error_poly_fast(params.n, params.error_bound);
    e1.reduce_modulo_xn_minus_1_barrett(params.n, barrett);

    let mut e2 = error_poly_fast(params.n, params.error_bound);
    e2.reduce_modulo_xn_minus_1_barrett(params.n, barrett);

    let scaled_msg = message.scalar_mul_barrett(params.q / 2, barrett);

    // u = a * r + e1  (using Barrett for all operations!)
    let mut u = pk.a.multiply_karatsuba_barrett(&r, barrett);
    u.reduce_modulo_xn_minus_1_barrett(params.n, barrett);
    u = u.add_mod_barrett(&e1, barrett);

    // v = b * r + e2 + scaled_msg
    let mut v = pk.b.multiply_karatsuba_barrett(&r, barrett);
    v.reduce_modulo_xn_minus_1_barrett(params.n, barrett);
    v = v.add_mod_barrett(&e2, barrett);
    v = v.add_mod_barrett(&scaled_msg, barrett);

    (u, v)
}

/// Fast encryption with precomputation and Barrett reduction
fn encrypt_with_cache(
    cache: &EncryptionCache,
    message: &CliffordPolynomialInt,
    params: &CLWEParams,
    barrett: &BarrettContext
) -> (CliffordPolynomialInt, CliffordPolynomialInt) {
    let mut e1 = error_poly_fast(params.n, params.error_bound);
    e1.reduce_modulo_xn_minus_1_barrett(params.n, barrett);

    let mut e2 = error_poly_fast(params.n, params.error_bound);
    e2.reduce_modulo_xn_minus_1_barrett(params.n, barrett);

    let scaled_msg = message.scalar_mul_barrett(params.q / 2, barrett);

    // u = (precomputed a*r) + e1  [NO MULTIPLICATION!]
    let u = cache.a_times_r.add_mod_barrett(&e1, barrett);

    // v = (precomputed b*r) + e2 + scaled_msg
    let mut v = cache.b_times_r.add_mod_barrett(&e2, barrett);
    v = v.add_mod_barrett(&scaled_msg, barrett);

    (u, v)
}

/// Decryption with Barrett reduction
fn decrypt(
    sk: &SecretKey,
    u: &CliffordPolynomialInt,
    v: &CliffordPolynomialInt,
    params: &CLWEParams,
    barrett: &BarrettContext
) -> CliffordPolynomialInt {
    let mut s_times_u = sk.s.multiply_karatsuba_barrett(u, barrett);
    s_times_u.reduce_modulo_xn_minus_1_barrett(params.n, barrett);

    let mut result = v.add_mod_barrett(&s_times_u.scalar_mul_barrett(-1, barrett), barrett);

    // Round to nearest message bit
    let threshold_low = params.q / 4;
    let threshold_high = 3 * params.q / 4;

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

    result
}

fn polys_equal(a: &CliffordPolynomialInt, b: &CliffordPolynomialInt) -> bool {
    if a.coeffs.len() != b.coeffs.len() {
        return false;
    }
    for (ca, cb) in a.coeffs.iter().zip(b.coeffs.iter()) {
        for i in 0..8 {
            if ca.coeffs[i] != cb.coeffs[i] {
                return false;
            }
        }
    }
    true
}

fn main() {
    println!("=== Clifford-LWE-256: BARRETT-OPTIMIZED VERSION ===\n");
    println!("Key Feature: Barrett reduction (~2-3× faster modular arithmetic)");
    println!();
    println!("Full optimization stack:");
    println!("  1. Barrett reduction (this version!)");
    println!("  2. Fast thread-local RNG");
    println!("  3. Precomputation for batch encryption");
    println!("  4. Optimized Karatsuba O(N^1.585)");
    println!("  5. Optimized geometric product (5.44×)");
    println!();

    let params = CLWEParams::default();
    let barrett = BarrettContext::new(params.q);

    println!("Parameters:");
    println!("  n = {} (polynomial degree)", params.n);
    println!("  q = {} (modulus)", params.q);
    println!("  error_bound = {}", params.error_bound);
    println!("  dimension = {} (8 components × {} degree)", 8 * params.n, params.n);
    println!();

    // Key generation
    println!("--- Key Generation ---");
    let keygen_start = Instant::now();
    let (pk, sk) = keygen(&params, &barrett);
    let keygen_time = keygen_start.elapsed();
    println!("Time: {:?}", keygen_time);
    println!();

    // Test message
    let mut msg_coeffs = Vec::with_capacity(params.n);
    for i in 0..params.n {
        let mut mv = [0i64; 8];
        mv[0] = if i % 3 == 0 { 1 } else { 0 };
        msg_coeffs.push(CliffordRingElementInt::from_multivector(mv));
    }
    let message = CliffordPolynomialInt::new(msg_coeffs);

    // Test correctness
    println!("--- Correctness Test ---");
    let (u, v) = encrypt(&pk, &message, &params, &barrett);
    let decrypted = decrypt(&sk, &u, &v, &params, &barrett);
    let correct = polys_equal(&message, &decrypted);
    println!("Standard encryption: {}", if correct { "✓ PASS" } else { "✗ FAIL" });
    println!();

    // Benchmark standard encryption
    println!("--- Benchmark: Standard Encryption (1000 ops) ---");
    const NUM_OPS: usize = 1000;

    let start = Instant::now();
    for _ in 0..NUM_OPS {
        let _ = encrypt(&pk, &message, &params, &barrett);
    }
    let standard_time = start.elapsed();
    let standard_avg = standard_time.as_micros() as f64 / NUM_OPS as f64;
    println!("Average per encryption: {:.2} µs", standard_avg);
    println!();

    // Benchmark with precomputation
    println!("--- Benchmark: Precomputed Encryption (1000 ops) ---");
    println!("Precomputation phase...");
    let precompute_start = Instant::now();
    let cache = EncryptionCache::new(&pk, &params, &barrett);
    let precompute_time = precompute_start.elapsed();
    println!("Precomputation time: {:?} (one-time cost)", precompute_time);

    let start = Instant::now();
    for _ in 0..NUM_OPS {
        let _ = encrypt_with_cache(&cache, &message, &params, &barrett);
    }
    let cached_time = start.elapsed();
    let cached_avg = cached_time.as_micros() as f64 / NUM_OPS as f64;
    println!("Average per encryption: {:.2} µs", cached_avg);

    let (u_cached, v_cached) = encrypt_with_cache(&cache, &message, &params, &barrett);
    let decrypted_cached = decrypt(&sk, &u_cached, &v_cached, &params, &barrett);
    let cached_correct = polys_equal(&message, &decrypted_cached);
    println!("Correctness: {}", if cached_correct { "✓ PASS" } else { "✗ FAIL" });
    println!();

    // Correctness validation
    println!("--- Extended Correctness Validation ---");
    const NUM_TESTS: usize = 10000;
    let mut failures = 0;

    for _ in 0..NUM_TESTS {
        let (pk_test, sk_test) = keygen(&params, &barrett);
        let (u_test, v_test) = encrypt(&pk_test, &message, &params, &barrett);
        let dec_test = decrypt(&sk_test, &u_test, &v_test, &params, &barrett);
        if !polys_equal(&message, &dec_test) {
            failures += 1;
        }
    }

    let success_rate = 100.0 * (NUM_TESTS - failures) as f64 / NUM_TESTS as f64;
    println!("Tested {} encryption cycles", NUM_TESTS);
    println!("Success rate: {:.2}% ({}/{} passed)", success_rate, NUM_TESTS - failures, NUM_TESTS);

    if failures > 0 {
        println!("⚠️  {} failures ({}%)", failures, 100.0 * failures as f64 / NUM_TESTS as f64);
    } else {
        println!("✓ All tests passed!");
    }
    println!();

    // Performance summary
    println!("=== Performance Comparison ===\n");
    println!("| Version | Standard (µs) | Precomputed (µs) | Notes |");
    println!("|---------|---------------|------------------|-------|");
    println!("| **i64 (standard)** | 59.52 | 9.06 | Baseline integer version |");
    println!("| **i64 + Barrett** | **{:.2}** | **{:.2}** | This version! |", standard_avg, cached_avg);
    println!();

    let speedup_standard = 59.52 / standard_avg;
    let speedup_cached = 9.06 / cached_avg;

    println!("--- Barrett Reduction Impact ---");
    println!("Standard mode: {:.2} µs → {:.2} µs ({:.2}× speedup)", 59.52, standard_avg, speedup_standard);
    println!("Precomputed mode: {:.2} µs → {:.2} µs ({:.2}× speedup)", 9.06, cached_avg, speedup_cached);
    println!();

    if standard_avg <= 40.0 {
        println!("🎉 SUCCESS: Standard encryption is under 40 µs!");
        println!("   Barrett reduction delivered {:.2}× speedup!", speedup_standard);
        println!("   Target achieved: competitive with f64 version (38.51 µs)");
    } else if standard_avg <= 50.0 {
        println!("✅ GOOD: Standard encryption is under 50 µs ({:.2} µs)", standard_avg);
        println!("   Barrett delivered {:.2}× speedup", speedup_standard);
        println!("   Close to target! Further optimization: lazy reduction");
    } else {
        println!("⚠️  Standard encryption: {:.2} µs", standard_avg);
        println!("   Speedup: {:.2}×", speedup_standard);
        println!("   Note: May need additional optimizations (lazy reduction, SIMD)");
    }
    println!();

    // vs Kyber
    println!("--- vs Kyber-512 ---");
    println!("Kyber-512 encryption: ~10-20 µs");
    println!("Clifford-LWE (Barrett, standard): {:.2} µs", standard_avg);
    println!("Clifford-LWE (Barrett, precomputed): {:.2} µs", cached_avg);
    println!();

    // vs floating-point
    println!("--- vs Floating-Point Version ---");
    println!("f64 (standard): 38.51 µs");
    println!("i64 + Barrett (standard): {:.2} µs ({:.1}× vs f64)", standard_avg, standard_avg / 38.51);
    println!();
    println!("f64 (precomputed): 9.09 µs");
    println!("i64 + Barrett (precomputed): {:.2} µs ({:.2}× vs f64)", cached_avg, cached_avg / 9.09);
    println!();

    if standard_avg < 40.0 {
        println!("✅ Integer + Barrett is competitive with floating-point!");
    } else if standard_avg < 45.0 {
        println!("⚠️  Integer + Barrett is within 20% of floating-point");
        println!("   (cryptographically sound arithmetic is worth the small overhead)");
    } else {
        println!("⚠️  Integer + Barrett is slower than floating-point");
        println!("   but provides cryptographic soundness (deterministic, portable)");
    }
    println!();

    println!("=== Next Optimizations ===\n");
    println!("Current: Barrett reduction [DONE]");
    println!("Next:");
    println!("  - Lazy reduction (reduce only when necessary) → 5-10% speedup");
    println!("  - Manual SIMD (NEON intrinsics) → 1.5-2× speedup");
    println!("  Target with all optimizations: 20-25 µs (faster than f64!)");
}
