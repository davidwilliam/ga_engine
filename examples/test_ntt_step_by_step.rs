// Comprehensive step-by-step NTT testing
// Break down into smallest testable units

// ============================================================================
// Copy NTT primitives to test them independently
// ============================================================================

#[inline(always)]
fn mod_add_u64(a: u64, b: u64, q: u64) -> u64 {
    let s = a.wrapping_add(b);
    if s >= q { s - q } else { s }
}

#[inline(always)]
fn mod_sub_u64(a: u64, b: u64, q: u64) -> u64 {
    if a >= b { a - b } else { a + q - b }
}

#[inline(always)]
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

fn primitive_root(q: u64) -> u64 {
    let phi = q - 1;
    let mut odd = phi;
    while odd % 2 == 0 { odd /= 2; }

    for g in 2..q {
        if mod_pow_u64(g, phi/2, q) == 1 { continue; }
        if odd != 1 && mod_pow_u64(g, phi/odd, q) == 1 { continue; }
        return g;
    }
    unreachable!("no primitive root found");
}

fn negacyclic_roots(q: u64, n: usize) -> (u64, u64) {
    let g = primitive_root(q);
    let two_n = 2u64 * (n as u64);
    assert_eq!((q - 1) % two_n, 0);
    let exp = (q - 1) / two_n;
    let psi = mod_pow_u64(g, exp, q);
    let omega = mod_mul_u64(psi, psi, q);
    (psi, omega)
}

#[inline(always)]
fn bitrev(mut k: usize, logn: usize) -> usize {
    let mut r = 0usize;
    for _ in 0..logn {
        r = (r << 1) | (k & 1);
        k >>= 1;
    }
    r
}

fn ntt_in_place(a: &mut [u64], q: u64, omega: u64) {
    let n = a.len();
    let logn = n.trailing_zeros() as usize;

    for i in 0..n {
        let j = bitrev(i, logn);
        if j > i { a.swap(i, j); }
    }

    let mut m = 1;
    for _stage in 0..logn {
        let m2 = m << 1;
        let w_m = mod_pow_u64(omega, (n / m2) as u64, q);
        let mut k = 0;
        while k < n {
            let mut w = 1u64;
            for j in 0..m {
                let t = mod_mul_u64(w, a[k + j + m], q);
                let u = a[k + j];
                a[k + j]     = mod_add_u64(u, t, q);
                a[k + j + m] = mod_sub_u64(u, t, q);
                w = mod_mul_u64(w, w_m, q);
            }
            k += m2;
        }
        m = m2;
    }
}

fn intt_in_place(a: &mut [u64], q: u64, omega: u64) {
    let n = a.len();
    // omega_inv = omega^{-1} = omega^{q-2} by Fermat's Little Theorem
    let omega_inv = mod_pow_u64(omega, q - 2, q);
    ntt_in_place(a, q, omega_inv);
    // scale by n^{-1} = n^{q-2} by Fermat
    let n_inv = mod_pow_u64(n as u64, q - 2, q);
    for v in a.iter_mut() {
        *v = mod_mul_u64(*v, n_inv, q);
    }
}

fn negacyclic_ntt(mut a: Vec<u64>, q: u64, psi: u64, omega: u64) -> Vec<u64> {
    let n = a.len();
    let mut pow = 1u64;
    for i in 0..n {
        a[i] = mod_mul_u64(a[i], pow, q);
        pow = mod_mul_u64(pow, psi, q);
    }
    ntt_in_place(&mut a, q, omega);
    a
}

fn negacyclic_intt(mut a: Vec<u64>, q: u64, psi: u64, omega: u64) -> Vec<u64> {
    let n = a.len();
    intt_in_place(&mut a, q, omega);
    let psi_inv = mod_pow_u64(psi, (q - 1) - 1, q);
    let mut pow = 1u64;
    for i in 0..n {
        a[i] = mod_mul_u64(a[i], pow, q);
        pow = mod_mul_u64(pow, psi_inv, q);
    }
    a
}

// ============================================================================
// TESTS
// ============================================================================

fn test_1_modular_arithmetic() {
    println!("TEST 1: Modular arithmetic primitives");
    let q = 17u64;

    // Test addition
    assert_eq!(mod_add_u64(10, 12, q), 5);  // (10 + 12) mod 17 = 5
    println!("  ✓ mod_add: (10 + 12) mod 17 = 5");

    // Test subtraction
    assert_eq!(mod_sub_u64(5, 12, q), 10);  // (5 - 12) mod 17 = 10
    println!("  ✓ mod_sub: (5 - 12) mod 17 = 10");

    // Test multiplication
    assert_eq!(mod_mul_u64(3, 5, q), 15);   // (3 * 5) mod 17 = 15
    println!("  ✓ mod_mul: (3 * 5) mod 17 = 15");

    // Test power
    assert_eq!(mod_pow_u64(2, 4, q), 16);   // 2^4 mod 17 = 16
    println!("  ✓ mod_pow: 2^4 mod 17 = 16");

    println!("  ✅ TEST 1 PASSED\n");
}

fn test_2_primitive_root() {
    println!("TEST 2: Primitive root finding");
    let q = 17u64; // 17 is prime, phi(17) = 16
    let g = primitive_root(q);
    println!("  Found primitive root: g = {}", g);

    // Verify g is primitive: g^(phi/2) != 1
    let half_order = mod_pow_u64(g, (q-1)/2, q);
    assert_ne!(half_order, 1);
    println!("  ✓ g^8 mod 17 = {} ≠ 1", half_order);

    // Verify g^phi = 1
    let full_order = mod_pow_u64(g, q-1, q);
    assert_eq!(full_order, 1);
    println!("  ✓ g^16 mod 17 = 1");

    println!("  ✅ TEST 2 PASSED\n");
}

fn test_3_negacyclic_roots() {
    println!("TEST 3: 2N-th root of unity");
    let q = 17u64; // q-1 = 16 = 2*8, so n can be at most 8
    let n = 8usize;

    let (psi, omega) = negacyclic_roots(q, n);
    println!("  psi = {}, omega = {}", psi, omega);

    // Verify psi^(2n) = 1
    let test1 = mod_pow_u64(psi, 2*n as u64, q);
    assert_eq!(test1, 1);
    println!("  ✓ psi^16 mod 17 = 1");

    // Verify psi^n = -1 (mod q) = q-1
    let test2 = mod_pow_u64(psi, n as u64, q);
    assert_eq!(test2, q - 1);
    println!("  ✓ psi^8 mod 17 = 16 (=-1)");

    // Verify omega = psi^2
    assert_eq!(omega, mod_mul_u64(psi, psi, q));
    println!("  ✓ omega = psi^2");

    // Verify omega^n = 1
    let test3 = mod_pow_u64(omega, n as u64, q);
    assert_eq!(test3, 1);
    println!("  ✓ omega^8 mod 17 = 1");

    println!("  ✅ TEST 3 PASSED\n");
}

fn test_4_bitrev() {
    println!("TEST 4: Bit-reversal permutation");
    let logn = 3; // n = 8

    assert_eq!(bitrev(0, logn), 0); // 000 -> 000
    assert_eq!(bitrev(1, logn), 4); // 001 -> 100
    assert_eq!(bitrev(2, logn), 2); // 010 -> 010
    assert_eq!(bitrev(3, logn), 6); // 011 -> 110
    assert_eq!(bitrev(4, logn), 1); // 100 -> 001
    assert_eq!(bitrev(5, logn), 5); // 101 -> 101
    assert_eq!(bitrev(6, logn), 3); // 110 -> 011
    assert_eq!(bitrev(7, logn), 7); // 111 -> 111

    println!("  ✓ All bit reversals correct for log(n)=3");
    println!("  ✅ TEST 4 PASSED\n");
}

fn test_5_cyclic_ntt() {
    println!("TEST 5: Forward cyclic NTT");
    let q = 17u64;
    let n = 8usize;
    let (_, omega) = negacyclic_roots(q, n);

    // Test with simple input: [1, 0, 0, 0, 0, 0, 0, 0]
    let mut a = vec![1u64, 0, 0, 0, 0, 0, 0, 0];
    println!("  Input: {:?}", a);

    ntt_in_place(&mut a, q, omega);
    println!("  NTT output: {:?}", a);

    // For polynomial f(x) = 1, NTT should give [1, 1, 1, ..., 1]
    // because f(omega^i) = 1 for all i
    for &val in &a {
        assert_eq!(val, 1);
    }
    println!("  ✓ NTT of [1,0,...,0] = [1,1,...,1]");

    println!("  ✅ TEST 5 PASSED\n");
}

fn test_6_inverse_ntt() {
    println!("TEST 6: Inverse cyclic NTT (roundtrip)");
    let q = 17u64;
    let n = 8usize;
    let (_, omega) = negacyclic_roots(q, n);

    // Test roundtrip with various inputs
    let inputs = vec![
        vec![1u64, 0, 0, 0, 0, 0, 0, 0],
        vec![1u64, 2, 3, 4, 5, 6, 7, 8],
        vec![5u64, 5, 5, 5, 5, 5, 5, 5],
    ];

    for original in inputs {
        let mut a = original.clone();
        ntt_in_place(&mut a, q, omega);
        intt_in_place(&mut a, q, omega);

        assert_eq!(a, original);
        println!("  ✓ Roundtrip successful for {:?}", original);
    }

    println!("  ✅ TEST 6 PASSED\n");
}

fn test_7_negacyclic_twist() {
    println!("TEST 7: Negacyclic twist (multiply by psi^i)");
    let q = 17u64;
    let n = 8usize;
    let (psi, _) = negacyclic_roots(q, n);

    let a = vec![1u64, 2, 3, 4, 5, 6, 7, 8];
    println!("  Input: {:?}", a);

    let mut twisted = a.clone();
    let mut pow = 1u64;
    for i in 0..n {
        twisted[i] = mod_mul_u64(a[i], pow, q);
        pow = mod_mul_u64(pow, psi, q);
    }

    println!("  Twisted: {:?}", twisted);
    println!("  ✓ Twist applied");

    // Verify untwist reverses it
    let psi_inv = mod_pow_u64(psi, (q - 1) - 1, q);
    let mut untwisted = twisted.clone();
    pow = 1u64;
    for i in 0..n {
        untwisted[i] = mod_mul_u64(twisted[i], pow, q);
        pow = mod_mul_u64(pow, psi_inv, q);
    }

    assert_eq!(untwisted, a);
    println!("  ✓ Untwist recovers original");

    println!("  ✅ TEST 7 PASSED\n");
}

fn test_9_negacyclic_ntt_roundtrip() {
    println!("TEST 9: Full negacyclic NTT roundtrip");
    let q = 17u64;
    let n = 8usize;
    let (psi, omega) = negacyclic_roots(q, n);

    let inputs = vec![
        vec![1u64, 0, 0, 0, 0, 0, 0, 0],
        vec![1u64, 2, 3, 4, 5, 6, 7, 8],
        vec![5u64, 5, 5, 5, 5, 5, 5, 5],
    ];

    for original in inputs {
        let a_ntt = negacyclic_ntt(original.clone(), q, psi, omega);
        let recovered = negacyclic_intt(a_ntt, q, psi, omega);

        assert_eq!(recovered, original);
        println!("  ✓ Roundtrip successful for {:?}", original);
    }

    println!("  ✅ TEST 9 PASSED\n");
}

fn test_10_negacyclic_multiply_small() {
    println!("TEST 10: Negacyclic polynomial multiplication (n=8)");
    let q = 17u64;
    let n = 8usize;
    let (psi, omega) = negacyclic_roots(q, n);

    // Test: (1 + 2x) * (3 + 4x) mod (x^8 + 1)
    // = 3 + 4x + 6x + 8x^2
    // = 3 + 10x + 8x^2
    let mut a = vec![0u64; n];
    let mut b = vec![0u64; n];
    a[0] = 1; a[1] = 2;
    b[0] = 3; b[1] = 4;

    println!("  a(x) = 1 + 2x");
    println!("  b(x) = 3 + 4x");
    println!("  Computing via NTT...");

    let a_ntt = negacyclic_ntt(a.clone(), q, psi, omega);
    let b_ntt = negacyclic_ntt(b.clone(), q, psi, omega);

    let mut c_ntt = vec![0u64; n];
    for i in 0..n {
        c_ntt[i] = mod_mul_u64(a_ntt[i], b_ntt[i], q);
    }

    let c = negacyclic_intt(c_ntt, q, psi, omega);

    println!("  Result: {:?}", c);
    println!("  Expected: [3, 10, 8, 0, 0, 0, 0, 0]");

    assert_eq!(c[0], 3);
    assert_eq!(c[1], 10);
    assert_eq!(c[2], 8);
    for i in 3..n {
        assert_eq!(c[i], 0);
    }

    println!("  ✓ (1+2x)*(3+4x) = 3+10x+8x^2");
    println!("  ✅ TEST 10 PASSED\n");
}

fn test_11_negacyclic_property() {
    println!("TEST 11: Verify negacyclic property (x^n ≡ -1)");
    let q = 17u64;
    let n = 8usize;
    let (psi, omega) = negacyclic_roots(q, n);

    // Test: x^8 * 1 should give -1 (i.e., q-1)
    // Polynomial x^8 is represented as having coeff 1 at position 8,
    // which wraps to position 0 with sign flip
    let mut a = vec![0u64; n];
    a[0] = 1;  // f(x) = 1

    let mut b = vec![0u64; n];
    b[0] = 0;  // This is tricky - we want x^8, but that's not representable!
    // Instead, let's test x^4 * x^4 = x^8 ≡ -1

    // Actually, let's test: x^7 * x = x^8 ≡ -1
    let mut x = vec![0u64; n];
    x[1] = 1;  // x

    let mut x7 = vec![0u64; n];
    x7[7] = 1;  // x^7

    let x_ntt = negacyclic_ntt(x.clone(), q, psi, omega);
    let x7_ntt = negacyclic_ntt(x7.clone(), q, psi, omega);

    let mut result_ntt = vec![0u64; n];
    for i in 0..n {
        result_ntt[i] = mod_mul_u64(x_ntt[i], x7_ntt[i], q);
    }

    let result = negacyclic_intt(result_ntt, q, psi, omega);

    println!("  x * x^7 = {:?}", result);
    println!("  Expected: [16, 0, 0, 0, 0, 0, 0, 0] (16 = -1 mod 17)");

    assert_eq!(result[0], q - 1);  // -1 mod q = q-1
    for i in 1..n {
        assert_eq!(result[i], 0);
    }

    println!("  ✓ x^8 ≡ -1 (mod x^8+1, q)");
    println!("  ✅ TEST 11 PASSED\n");
}

fn main() {
    println!("{}", "=".repeat(70));
    println!("NTT STEP-BY-STEP VALIDATION");
    println!("{}", "=".repeat(70));
    println!();

    test_1_modular_arithmetic();
    test_2_primitive_root();
    test_3_negacyclic_roots();
    test_4_bitrev();
    test_5_cyclic_ntt();
    test_6_inverse_ntt();
    test_7_negacyclic_twist();
    test_9_negacyclic_ntt_roundtrip();
    test_10_negacyclic_multiply_small();
    test_11_negacyclic_property();

    println!("{}", "=".repeat(70));
    println!("✅ ALL TESTS PASSED!");
    println!("{}", "=".repeat(70));
}
