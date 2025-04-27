//! Numerical equivalence check between classical DFT and GA DFT (2D multivectors).

use crate::numerical_checks::multivector2::Multivector2;
use rustfft::num_complex::Complex;
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

/// Optimized GA DFT using rotor recurrence (no trig inside loops)
pub fn ga_dft(input: &[Multivector2]) -> Vec<Multivector2> {
    let n = input.len();
    let mut output = Vec::with_capacity(n);

    for k in 0..n {
        let delta = -2.0 * PI * (k as f64) / (n as f64);
        let cos_delta = delta.cos();
        let sin_delta = delta.sin();

        // Start with angle 0
        let mut cos_theta = 1.0;
        let mut sin_theta = 0.0;

        let mut sum = Multivector2::zero();

        for x_n in input {
            // rotor = cos(θ) + sin(θ) * e12
            let rotor = Multivector2::new(cos_theta, 0.0, 0.0, sin_theta);
            let product = rotor.gp(*x_n);
            sum = sum + product;

            // Recurrence update:
            let new_cos = cos_theta * cos_delta - sin_theta * sin_delta;
            let new_sin = sin_theta * cos_delta + cos_theta * sin_delta;
            cos_theta = new_cos;
            sin_theta = new_sin;
        }

        output.push(sum);
    }

    output
}

/// Compare two sequences (Complex ↔ Multivector2)
pub fn compare_complex_outputs(reference: &[Complex<f64>], output: &[Multivector2]) -> bool {
    if reference.len() != output.len() {
        return false;
    }

    reference.iter().zip(output.iter()).all(|(a, b)| {
        let re = b.data[0]; // scalar part
        let im = b.data[3]; // e12 part
        (a.re - re).abs() < EPSILON && (a.im - im).abs() < EPSILON
    })
}
