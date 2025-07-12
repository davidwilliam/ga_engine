use crate::numerical_checks::multivector2::Multivector2;
use rustfft::num_complex::Complex;
use wide::f64x4;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-10;

/// Classical DFT (naive O(N²))
pub fn classical_dft(input: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = input.len();
    (0..n)
        .map(|k| {
            let mut sum = Complex::new(0.0, 0.0);
            for (n_idx, x_n) in input.iter().enumerate() {
                let angle = -2.0 * PI * (k as f64) * (n_idx as f64) / (n as f64);
                let twiddle = Complex::from_polar(1.0, angle);
                sum += x_n * twiddle;
            }
            sum
        })
        .collect()
}

/// Precompute rotors as f64x4
fn precompute_rotors(n: usize) -> Vec<f64x4> {
    let mut rotors = Vec::with_capacity(n * n);
    for k in 0..n {
        for n_idx in 0..n {
            let angle = -2.0 * PI * (k as f64) * (n_idx as f64) / (n as f64);
            rotors.push(f64x4::new([
                angle.cos(), // scalar
                0.0,         // e1
                0.0,         // e2
                angle.sin(), // e12
            ]));
        }
    }
    rotors
}

/// Geometric product using only simple vector ops
#[inline(always)]
fn gp(rotor: f64x4, mv: f64x4) -> f64x4 {
    let r = rotor.to_array();
    let m = mv.to_array();

    f64x4::new([
        r[0] * m[0] + r[1] * m[1] + r[2] * m[2] - r[3] * m[3], // scalar
        r[0] * m[1] + r[1] * m[0] - r[2] * m[3] + r[3] * m[2], // e1
        r[0] * m[2] + r[1] * m[3] + r[2] * m[0] - r[3] * m[1], // e2
        r[0] * m[3] + r[1] * m[2] - r[2] * m[1] + r[3] * m[0], // e12
    ])
}

/// Optimized GA DFT
pub fn ga_dft(input: &[Multivector2]) -> Vec<Multivector2> {
    let n = input.len();
    let rotors = precompute_rotors(n);

    let input_simd: Vec<f64x4> = input.iter().map(|x| f64x4::new(x.data)).collect();

    let mut output = Vec::with_capacity(n);

    for k in 0..n {
        let mut sum = f64x4::splat(0.0);
        for n_idx in 0..n {
            let rotor = rotors[k * n + n_idx];
            let x_n = input_simd[n_idx];
            sum = sum + gp(rotor, x_n);
        }
        output.push(Multivector2 {
            data: sum.to_array(),
        });
    }
    output
}

/// Compare Complex vs GA output
pub fn compare_complex_outputs(reference: &[Complex<f64>], output: &[Multivector2]) -> bool {
    if reference.len() != output.len() {
        return false;
    }
    reference.iter().zip(output.iter()).all(|(a, b)| {
        let re = b.data[0]; // scalar
        let im = b.data[3]; // e12
        (a.re - re).abs() < EPSILON && (a.im - im).abs() < EPSILON
    })
}
