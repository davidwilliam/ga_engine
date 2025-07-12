use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::numerical_checks::dft;
use ga_engine::numerical_checks::multivector2::Multivector2;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;

fn generate_inputs(size: usize) -> Vec<(Vec<Complex<f64>>, Vec<Multivector2>, &'static str)> {
    let mut variants = Vec::new();

    // Linear ramp
    let ramp: Vec<f64> = (0..size).map(|x| x as f64).collect();
    variants.push((
        ramp.iter().map(|&x| Complex::new(x, 0.0)).collect(),
        ramp.iter().map(|&x| Multivector2::new(x, 0.0, 0.0, 0.0)).collect(),
        "ramp",
    ));

    // Sine wave
    let sine: Vec<f64> = (0..size).map(|x| (2.0 * PI * x as f64 / size as f64).sin()).collect();
    variants.push((
        sine.iter().map(|&x| Complex::new(x, 0.0)).collect(),
        sine.iter().map(|&x| Multivector2::new(x, 0.0, 0.0, 0.0)).collect(),
        "sine",
    ));

    // Gaussian blob
    let center = size as f64 / 2.0;
    let gaussian: Vec<f64> = (0..size)
        .map(|x| (-(x as f64 - center).powi(2) / 30.0).exp())
        .collect();
    variants.push((
        gaussian.iter().map(|&x| Complex::new(x, 0.0)).collect(),
        gaussian.iter().map(|&x| Multivector2::new(x, 0.0, 0.0, 0.0)).collect(),
        "gaussian",
    ));

    variants
}

fn benchmark_structured_dft(c: &mut Criterion) {
    let sizes = [4, 8, 9, 12, 16, 24];
    for &size in &sizes {
        for (input_complex, input_ga, tag) in generate_inputs(size) {
            c.bench_function(&format!("classical_dft_{}_{}", tag, size), |b| {
                b.iter(|| dft::classical_dft(black_box(&input_complex)))
            });

            c.bench_function(&format!("ga_dft_{}_{}", tag, size), |b| {
                b.iter(|| dft::ga_dft(black_box(&input_ga)))
            });
        }
    }
}

criterion_group!(benches, benchmark_structured_dft);
criterion_main!(benches);
