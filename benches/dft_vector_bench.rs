use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::numerical_checks::dft;
use ga_engine::numerical_checks::multivector2::Multivector2;
use rustfft::num_complex::Complex;

fn generate_vector_input(size: usize) -> (Vec<Complex<f64>>, Vec<Multivector2>) {
    let input_real: Vec<f64> = (0..size).map(|x| x as f64).collect();
    let input_complex: Vec<Complex<f64>> =
        input_real.iter().map(|&x| Complex::new(x, x)).collect();

    let input_ga: Vec<Multivector2> = input_real
        .iter()
        .map(|&x| Multivector2::new(0.0, x, x * 0.5, 0.0))
        .collect();

    (input_complex, input_ga)
}

fn benchmark_dfts_vector(c: &mut Criterion) {
    let sizes = [16, 64, 256, 1024];
    for &size in &sizes {
        let (input_complex, input_ga) = generate_vector_input(size);

        c.bench_function(&format!("classical_dft_vector_{}", size), |b| {
            b.iter(|| dft::classical_dft(black_box(&input_complex)))
        });

        c.bench_function(&format!("ga_dft_vector_{}", size), |b| {
            b.iter(|| dft::ga_dft(black_box(&input_ga[..])))
        });
    }
}

criterion_group!(vector_benches, benchmark_dfts_vector);
criterion_main!(vector_benches);
