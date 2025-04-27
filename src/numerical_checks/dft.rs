// src/numerical_checks/dft.rs

use crate::nd::Multivector;
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

/// GA DFT using 2D multivectors (Multivector<2>)
pub fn ga_dft(input: &[Multivector<2>]) -> Vec<Multivector<2>> {
    let n = input.len();
    (0..n)
        .map(|k| {
            let mut sum = Multivector::<2>::zero();
            for (n_idx, x_n) in input.iter().enumerate() {
                let angle = -2.0 * PI * (k as f64) * (n_idx as f64) / (n as f64);
                // Represent e^(iθ) = cos(θ) + sin(θ) * e12
                let rotor = Multivector::<2>::new(vec![
                    angle.cos(), // scalar part
                    0.0,         // e1
                    0.0,         // e2
                    angle.sin(), // e12
                ]);
                let product = rotor.gp(x_n);
                sum = sum + product;
            }
            sum
        })
        .collect()
}

/// Compare two sequences (complex to complex) elementwise
pub fn compare_complex_outputs(reference: &[Complex<f64>], output: &[Multivector<2>]) -> bool {
    if reference.len() != output.len() {
        return false;
    }

    reference.iter().zip(output.iter()).all(|(a, b)| {
        let re = b.data[0]; // scalar part
        let im = b.data[3]; // bivector e12 part
        (a.re - re).abs() < EPSILON && (a.im - im).abs() < EPSILON
    })
}
