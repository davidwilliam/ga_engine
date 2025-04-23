use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::{classical, ga};

const BATCH_SIZE: usize = 1_000;

/// Benchmark 8×8 matrix multiplication batched.
fn bench_matrix_mult(c: &mut Criterion) {
    let n = black_box(8);
    let a: Vec<f64> = (0..n * n).map(|i| (i % 10) as f64).collect();
    let b = a.clone();

    c.bench_function("matrix 8×8 × 1000 batch", |bencher| {
        bencher.iter(|| {
            let mut res = Vec::with_capacity(n * n);
            for _ in 0..BATCH_SIZE {
                res = classical::multiply_matrices(black_box(&a), black_box(&b), n);
            }
            black_box(res)
        })
    });
}

/// Benchmark full 8-component multivector geometric product batched.
fn bench_geometric_product_full(c: &mut Criterion) {
    let a: [f64; 8] = black_box([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = a;

    c.bench_function("GA full product 8D × 1000 batch", |bencher| {
        bencher.iter(|| {
            let mut out = [0.0; 8];
            for _ in 0..BATCH_SIZE {
                ga::geometric_product_full(black_box(&a), black_box(&b), &mut out);
            }
            black_box(out)
        })
    });
}

criterion_group!(benches, bench_matrix_mult, bench_geometric_product_full);
criterion_main!(benches);
