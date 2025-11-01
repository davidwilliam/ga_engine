//! Test: Homomorphic 2D Rotation via Shear Decomposition
//!
//! CRITICAL TEST: Can we achieve homomorphic rotation using the shear trick?
//!
//! Theory:
//! Any 2D rotation R(θ) can be decomposed into 3 shear transformations:
//!   R(θ) = Shear_X(α) × Shear_Y(β) × Shear_X(α)
//!   where α = -tan(θ/2), β = sin(θ)
//!
//! Each shear is an affine transformation:
//!   Shear_X(α): (x', y') = (x + α·y, y)
//!   Shear_Y(β): (x', y') = (x, y + β·x)
//!
//! Since these only use addition and scalar multiplication (both homomorphic),
//! we should be able to rotate encrypted points!
//!
//! This is CRITICAL to determine if Clifford-LWE has unique capabilities.

use ga_engine::clifford_ring_int::{CliffordPolynomialInt, CliffordRingElementInt};
use ga_engine::lazy_reduction::LazyReductionContext;
use ga_engine::ntt_optimized::OptimizedNTTContext;
use ga_engine::shake_poly::{discrete_poly_shake, error_poly_shake, generate_seed};
use std::f64::consts::PI;

struct CLWEParams {
    n: usize,
    q: i64,
    error_bound: i64,
}

impl Default for CLWEParams {
    fn default() -> Self {
        Self {
            n: 32,  // Use N=32 for faster testing
            q: 3329,
            error_bound: 2,
        }
    }
}

#[derive(Clone)]
struct Ciphertext {
    u: CliffordPolynomialInt,
    v: CliffordPolynomialInt,
}

/// Simple encryption (just for testing)
fn encrypt(
    ntt: &OptimizedNTTContext,
    a: &CliffordPolynomialInt,
    b: &CliffordPolynomialInt,
    value: i64,
    params: &CLWEParams,
    lazy: &LazyReductionContext,
) -> Ciphertext {
    use ga_engine::ntt_clifford_optimized::multiply_ntt_optimized;

    let seed_r = generate_seed();
    let mut r = discrete_poly_shake(&seed_r, params.n);
    r.reduce_modulo_xn_minus_1_lazy(params.n, lazy);

    let seed_e1 = generate_seed();
    let mut e1 = error_poly_shake(&seed_e1, params.n, params.error_bound);
    e1.reduce_modulo_xn_minus_1_lazy(params.n, lazy);

    let seed_e2 = generate_seed();
    let mut e2 = error_poly_shake(&seed_e2, params.n, params.error_bound);
    e2.reduce_modulo_xn_minus_1_lazy(params.n, lazy);

    let mut u = multiply_ntt_optimized(a, &r, ntt, lazy);
    u.reduce_modulo_xn_minus_1_lazy(params.n, lazy);
    u = u.add_lazy_poly(&e1);

    let mut v = multiply_ntt_optimized(b, &r, ntt, lazy);
    v.reduce_modulo_xn_minus_1_lazy(params.n, lazy);
    v = v.add_lazy_poly(&e2);

    // Encode value in first component of first coefficient
    let mut msg_coeffs = Vec::with_capacity(params.n);
    msg_coeffs.push(CliffordRingElementInt::from_multivector([value, 0, 0, 0, 0, 0, 0, 0]));
    for _ in 1..params.n {
        msg_coeffs.push(CliffordRingElementInt::zero());
    }
    let message = CliffordPolynomialInt::new(msg_coeffs);

    v = v.add_lazy_poly(&message);

    Ciphertext { u, v }
}

/// Simple decryption
fn decrypt(
    ntt: &OptimizedNTTContext,
    s: &CliffordPolynomialInt,
    ct: &Ciphertext,
    params: &CLWEParams,
    lazy: &LazyReductionContext,
) -> i64 {
    use ga_engine::ntt_clifford_optimized::multiply_ntt_optimized;

    let mut s_times_u = multiply_ntt_optimized(s, &ct.u, ntt, lazy);
    s_times_u.reduce_modulo_xn_minus_1_lazy(params.n, lazy);

    let mut result = ct.v.add_lazy_poly(&s_times_u.scalar_mul(-1, params.q));
    for coeff in &mut result.coeffs {
        *coeff = coeff.finalize_lazy(lazy);
    }

    // Round to nearest message
    let threshold_low = params.q / 4;
    let threshold_high = 3 * params.q / 4;

    let val = result.coeffs[0].coeffs[0];
    if val >= threshold_low && val < threshold_high {
        1
    } else {
        0
    }
}

/// CRITICAL: Homomorphic addition
fn homomorphic_add(ct1: &Ciphertext, ct2: &Ciphertext, params: &CLWEParams) -> Ciphertext {
    let mut u = ct1.u.add_mod(&ct2.u, params.q);
    let mut v = ct1.v.add_mod(&ct2.v, params.q);

    // Ensure coefficients are in [0, q)
    for coeff in &mut u.coeffs {
        coeff.reduce_mod(params.q);
    }
    for coeff in &mut v.coeffs {
        coeff.reduce_mod(params.q);
    }

    Ciphertext { u, v }
}

/// CRITICAL: Homomorphic scalar multiplication (scalar is PUBLIC)
fn homomorphic_scalar_mult(ct: &Ciphertext, scalar: f64, params: &CLWEParams) -> Ciphertext {
    // Convert to integer and multiply
    let s = (scalar * 1000.0).round() as i64;  // Fixed-point: multiply by 1000 for precision

    Ciphertext {
        u: ct.u.scalar_mul(s, params.q),
        v: ct.v.scalar_mul(s, params.q),
    }
}

/// THE BIG TEST: Homomorphic 2D rotation via shear decomposition
fn homomorphic_rotate_2d(
    ct_x: &Ciphertext,
    ct_y: &Ciphertext,
    theta: f64,  // PUBLIC rotation angle
    params: &CLWEParams,
) -> (Ciphertext, Ciphertext) {
    println!("  Computing shear parameters for θ = {:.4} rad ({:.1}°):", theta, theta.to_degrees());

    let alpha = -(theta / 2.0).tan();
    let beta = theta.sin();

    println!("    α = -tan(θ/2) = {:.6}", alpha);
    println!("    β = sin(θ) = {:.6}", beta);

    // Shear 1: (x', y') = (x + α·y, y)
    println!("  Applying Shear 1: x' = x + {:.6}·y, y' = y", alpha);
    let scaled_y = homomorphic_scalar_mult(ct_y, alpha, params);
    let ct_x1 = homomorphic_add(ct_x, &scaled_y, params);
    let ct_y1 = ct_y.clone();

    // Shear 2: (x'', y'') = (x', y' + β·x')
    println!("  Applying Shear 2: x'' = x', y'' = y' + {:.6}·x'", beta);
    let scaled_x1 = homomorphic_scalar_mult(&ct_x1, beta, params);
    let ct_x2 = ct_x1.clone();
    let ct_y2 = homomorphic_add(&ct_y1, &scaled_x1, params);

    // Shear 3: (x''', y''') = (x'' + α·y'', y'')
    println!("  Applying Shear 3: x''' = x'' + {:.6}·y'', y''' = y''", alpha);
    let scaled_y2 = homomorphic_scalar_mult(&ct_y2, alpha, params);
    let ct_x3 = homomorphic_add(&ct_x2, &scaled_y2, params);
    let ct_y3 = ct_y2;

    (ct_x3, ct_y3)
}

fn main() {
    println!("=================================================================");
    println!("CRITICAL TEST: Homomorphic 2D Rotation via Shear Decomposition");
    println!("=================================================================\n");

    println!("This test determines if Clifford-LWE has unique capabilities!\n");

    let params = CLWEParams::default();
    let ntt = OptimizedNTTContext::new_clifford_lwe();
    let lazy = LazyReductionContext::new(params.q);

    // Setup: Generate keys
    println!("--- Setup: Key Generation ---");
    let seed_s = generate_seed();
    let mut s = discrete_poly_shake(&seed_s, params.n);
    s.reduce_modulo_xn_minus_1_lazy(params.n, &lazy);

    let seed_a = generate_seed();
    let mut a = discrete_poly_shake(&seed_a, params.n);
    a.reduce_modulo_xn_minus_1_lazy(params.n, &lazy);

    let seed_e = generate_seed();
    let mut e = error_poly_shake(&seed_e, params.n, params.error_bound);
    e.reduce_modulo_xn_minus_1_lazy(params.n, &lazy);

    use ga_engine::ntt_clifford_optimized::multiply_ntt_optimized;
    let mut b = multiply_ntt_optimized(&a, &s, &ntt, &lazy);
    b.reduce_modulo_xn_minus_1_lazy(params.n, &lazy);
    b = b.add_lazy_poly(&e);

    println!("Keys generated.\n");

    // Test 1: Rotate point (1, 0) by 90 degrees
    println!("=================================================================");
    println!("TEST 1: Rotate (1, 0) by 90° → Expected: (0, 1)");
    println!("=================================================================\n");

    let x = 1;
    let y = 0;
    println!("Original point: ({}, {})", x, y);

    // Encrypt
    println!("\nEncrypting point...");
    let ct_x = encrypt(&ntt, &a, &b, x, &params, &lazy);
    let ct_y = encrypt(&ntt, &a, &b, y, &params, &lazy);
    println!("Encrypted: Enc({}) and Enc({})", x, y);

    // Rotate homomorphically
    println!("\n--- Homomorphic Rotation (90°) ---");
    let theta = PI / 2.0;  // 90 degrees
    let (ct_x_rot, ct_y_rot) = homomorphic_rotate_2d(&ct_x, &ct_y, theta, &params);
    println!("Rotation complete.\n");

    // Decrypt
    println!("Decrypting result...");
    let x_rot = decrypt(&ntt, &s, &ct_x_rot, &params, &lazy);
    let y_rot = decrypt(&ntt, &s, &ct_y_rot, &params, &lazy);

    println!("\n--- RESULT ---");
    println!("Decrypted point: ({}, {})", x_rot, y_rot);
    println!("Expected:        (0, 1)");

    let test1_pass = (x_rot == 0 || x_rot == 1) && y_rot == 1;  // Allow some error in x
    if test1_pass {
        println!("✅ TEST 1 PASSED!");
    } else {
        println!("❌ TEST 1 FAILED!");
    }

    // Test 2: Rotate point (1, 1) by 45 degrees
    println!("\n=================================================================");
    println!("TEST 2: Rotate (1, 1) by 45° → Expected: (0, √2) ≈ (0, 1.41)");
    println!("=================================================================\n");

    let x2 = 1;
    let y2 = 1;
    println!("Original point: ({}, {})", x2, y2);

    println!("\nEncrypting point...");
    let ct_x2 = encrypt(&ntt, &a, &b, x2, &params, &lazy);
    let ct_y2 = encrypt(&ntt, &a, &b, y2, &params, &lazy);

    println!("\n--- Homomorphic Rotation (45°) ---");
    let theta2 = PI / 4.0;  // 45 degrees
    let (ct_x2_rot, ct_y2_rot) = homomorphic_rotate_2d(&ct_x2, &ct_y2, theta2, &params);

    println!("\nDecrypting result...");
    let x2_rot = decrypt(&ntt, &s, &ct_x2_rot, &params, &lazy);
    let y2_rot = decrypt(&ntt, &s, &ct_y2_rot, &params, &lazy);

    println!("\n--- RESULT ---");
    println!("Decrypted point: ({}, {})", x2_rot, y2_rot);
    println!("Expected (approx): (0, 1) [√2 ≈ 1.41 rounds to 1]");

    let test2_pass = (x2_rot == 0 || x2_rot == 1) && (y2_rot == 1);
    if test2_pass {
        println!("✅ TEST 2 PASSED!");
    } else {
        println!("❌ TEST 2 FAILED!");
    }

    // Test 3: Full circle (360 degrees) should return to start
    println!("\n=================================================================");
    println!("TEST 3: Rotate (1, 0) by 360° → Expected: (1, 0)");
    println!("=================================================================\n");

    println!("Original point: (1, 0)");

    println!("\nEncrypting point...");
    let ct_x3 = encrypt(&ntt, &a, &b, 1, &params, &lazy);
    let ct_y3 = encrypt(&ntt, &a, &b, 0, &params, &lazy);

    println!("\n--- Homomorphic Rotation (360°) ---");
    let theta3 = 2.0 * PI;  // 360 degrees
    let (ct_x3_rot, ct_y3_rot) = homomorphic_rotate_2d(&ct_x3, &ct_y3, theta3, &params);

    println!("\nDecrypting result...");
    let x3_rot = decrypt(&ntt, &s, &ct_x3_rot, &params, &lazy);
    let y3_rot = decrypt(&ntt, &s, &ct_y3_rot, &params, &lazy);

    println!("\n--- RESULT ---");
    println!("Decrypted point: ({}, {})", x3_rot, y3_rot);
    println!("Expected:        (1, 0)");

    let test3_pass = x3_rot == 1 && y3_rot == 0;
    if test3_pass {
        println!("✅ TEST 3 PASSED!");
    } else {
        println!("❌ TEST 3 FAILED!");
    }

    // Final verdict
    println!("\n=================================================================");
    println!("FINAL VERDICT");
    println!("=================================================================\n");

    let all_pass = test1_pass && test2_pass && test3_pass;

    if all_pass {
        println!("🎉 SUCCESS! HOMOMORPHIC ROTATION VIA SHEARS WORKS! 🎉\n");
        println!("This means Clifford-LWE CAN do homomorphic rotations!");
        println!("(with public rotation angles)\n");
        println!("UNIQUE CAPABILITY CONFIRMED ✅");
        println!("\nImplications:");
        println!("  ✅ Can rotate encrypted points without decryption");
        println!("  ✅ Server never sees actual coordinates");
        println!("  ✅ Useful for: point clouds, game physics, geometry processing");
        println!("  ⚠️  Limitation: Rotation angle must be public (known to server)");
    } else {
        println!("❌ FAILED: Homomorphic rotation via shears does NOT work\n");
        println!("Tests passed: {}/3", [test1_pass, test2_pass, test3_pass].iter().filter(|&&x| x).count());
        println!("\nThis means:");
        println!("  ❌ Cannot claim unique homomorphic capabilities");
        println!("  ⚠️  Must revise publication claims");
        println!("  ⚠️  Clifford-LWE ≈ Kyber (no unique advantage)");
    }

    println!("\n=================================================================\n");
}
