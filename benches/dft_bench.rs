use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::nd::Multivector;
use ga_engine::numerical_checks::dft::{classical_dft, ga_dft};
use rustfft::num_complex::Complex;

fn generate_input(size: usize) -> (Vec<Complex<f64>>, Vec<Multivector<2>>) {
    let input_real: Vec<f64> = (0..size).map(|x| x as f64).collect();
    let input_complex: Vec<Complex<f64>> =
        input_real.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let input_ga: Vec<Multivector<2>> = input_real
        .iter()
        .map(|&x| Multivector::<2>::new(vec![x, 0.0, 0.0, 0.0]))
        .collect();
    (input_complex, input_ga)
}

fn benchmark_dfts(c: &mut Criterion) {
    let sizes = [16, 64, 256, 1024];
    for &size in &sizes {
        let (input_complex, input_ga) = generate_input(size);

        c.bench_function(&format!("classical_dft_{}", size), |b| {
            b.iter(|| classical_dft(black_box(&input_complex)))
        });

        c.bench_function(&format!("ga_dft_{}", size), |b| {
            b.iter(|| ga_dft(black_box(&input_ga)))
        });
    }
}

criterion_group!(benches, benchmark_dfts);
criterion_main!(benches);
