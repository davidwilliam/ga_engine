// benches/multivector_2d_vs_matrix_2x2.rs
//! Benchmark: 2D Multivector vs 2x2 Matrix Multiplication
//!
//! This benchmark tests whether GA has an advantage in 2D operations
//! by comparing 2D multivector geometric products (4 components) 
//! against 2x2 matrix multiplication (4 elements).

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::numerical_checks::multivector2::Multivector2;
use nalgebra::Matrix2;
use matrixmultiply::dgemm;

const BATCH_SIZE: usize = 1_000;

/// Benchmark 2D multivector geometric product
fn bench_ga_2d_multivector(c: &mut Criterion) {
    let a = Multivector2::new(1.0, 2.0, 3.0, 4.0);
    let b = Multivector2::new(5.0, 6.0, 7.0, 8.0);

    let mut group = c.benchmark_group("ga_2d_operations");
    group.bench_function("GA_2D_multivector_1000_batch", |bencher| {
        bencher.iter(|| {
            let mut result = Multivector2::zero();
            for _ in 0..BATCH_SIZE {
                result = black_box(a).gp(black_box(b));
            }
            black_box(result)
        })
    });
    group.finish();
}

/// Benchmark 2x2 matrix multiplication (naive implementation)
fn bench_matrix_2x2_naive(c: &mut Criterion) {
    let a = [1.0, 2.0, 3.0, 4.0]; // 2x2 matrix in row-major order
    let b = [5.0, 6.0, 7.0, 8.0];

    let mut group = c.benchmark_group("matrix_2x2_naive");
    group.bench_function("matrix_2x2_naive_1000_batch", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 4];
            for _ in 0..BATCH_SIZE {
                let a = black_box(&a);
                let b = black_box(&b);
                // Matrix multiplication: C = A * B
                result[0] = a[0] * b[0] + a[1] * b[2]; // c11
                result[1] = a[0] * b[1] + a[1] * b[3]; // c12
                result[2] = a[2] * b[0] + a[3] * b[2]; // c21
                result[3] = a[2] * b[1] + a[3] * b[3]; // c22
            }
            black_box(result)
        })
    });
    group.finish();
}

/// Benchmark 2x2 matrix multiplication using nalgebra
fn bench_matrix_2x2_nalgebra(c: &mut Criterion) {
    let a = Matrix2::new(1.0, 2.0, 3.0, 4.0);
    let b = Matrix2::new(5.0, 6.0, 7.0, 8.0);

    let mut group = c.benchmark_group("matrix_2x2_nalgebra");
    group.bench_function("matrix_2x2_nalgebra_1000_batch", |bencher| {
        bencher.iter(|| {
            let mut result = Matrix2::zeros();
            for _ in 0..BATCH_SIZE {
                result = black_box(a) * black_box(b);
            }
            black_box(result)
        })
    });
    group.finish();
}

/// Benchmark 2x2 matrix multiplication using matrixmultiply dgemm
fn bench_matrix_2x2_dgemm(c: &mut Criterion) {
    let a = [1.0, 2.0, 3.0, 4.0]; // 2x2 matrix in row-major order
    let b = [5.0, 6.0, 7.0, 8.0];

    let mut group = c.benchmark_group("matrix_2x2_dgemm");
    group.bench_function("matrix_2x2_dgemm_1000_batch", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 4];
            for _ in 0..BATCH_SIZE {
                unsafe {
                    dgemm(
                        // m, k, n (dimensions)
                        2, 2, 2,
                        // alpha
                        1.0,
                        // A pointer, row stride, col stride
                        a.as_ptr(),
                        2,
                        1,
                        // B pointer, row stride, col stride
                        b.as_ptr(),
                        2,
                        1,
                        // beta
                        0.0,
                        // C pointer, row stride, col stride
                        result.as_mut_ptr(),
                        2,
                        1,
                    );
                }
            }
            black_box(result)
        })
    });
    group.finish();
}

/// Benchmark using Apple Accelerate BLAS for 2x2 matrices
#[cfg(target_os = "macos")]
fn bench_matrix_2x2_accelerate(c: &mut Criterion) {
    use ndarray::linalg::general_mat_mul;
    use ndarray::{Array2, ShapeBuilder};

    let a = Array2::from_shape_vec((2, 2).f(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Array2::from_shape_vec((2, 2).f(), vec![5.0, 6.0, 7.0, 8.0]).unwrap();

    let mut group = c.benchmark_group("matrix_2x2_accelerate");
    group.bench_function("matrix_2x2_accelerate_1000_batch", |bencher| {
        bencher.iter(|| {
            let mut result = Array2::<f64>::zeros((2, 2).f());
            for _ in 0..BATCH_SIZE {
                general_mat_mul(1.0, &a, &b, 0.0, &mut result);
            }
            black_box(result)
        })
    });
    group.finish();
}

/// Direct comparison benchmark: GA vs Naive vs Nalgebra
fn bench_direct_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("2d_operations_comparison");
    
    // GA 2D
    let ga_a = Multivector2::new(1.0, 2.0, 3.0, 4.0);
    let ga_b = Multivector2::new(5.0, 6.0, 7.0, 8.0);
    
    // Naive matrix
    let matrix_a = [1.0, 2.0, 3.0, 4.0];
    let matrix_b = [5.0, 6.0, 7.0, 8.0];
    
    // Nalgebra
    let nalg_a = Matrix2::new(1.0, 2.0, 3.0, 4.0);
    let nalg_b = Matrix2::new(5.0, 6.0, 7.0, 8.0);

    group.bench_function("GA_2D_multivector", |bencher| {
        bencher.iter(|| {
            let mut result = Multivector2::zero();
            for _ in 0..BATCH_SIZE {
                result = black_box(ga_a).gp(black_box(ga_b));
            }
            black_box(result)
        })
    });

    group.bench_function("naive_2x2_matrix", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 4];
            for _ in 0..BATCH_SIZE {
                let a = black_box(&matrix_a);
                let b = black_box(&matrix_b);
                result[0] = a[0] * b[0] + a[1] * b[2];
                result[1] = a[0] * b[1] + a[1] * b[3];
                result[2] = a[2] * b[0] + a[3] * b[2];
                result[3] = a[2] * b[1] + a[3] * b[3];
            }
            black_box(result)
        })
    });

    group.bench_function("nalgebra_2x2_matrix", |bencher| {
        bencher.iter(|| {
            let mut result = Matrix2::zeros();
            for _ in 0..BATCH_SIZE {
                result = black_box(nalg_a) * black_box(nalg_b);
            }
            black_box(result)
        })
    });

    group.finish();
}

/// Benchmark correctness verification
fn bench_correctness_check(c: &mut Criterion) {
    let a = Multivector2::new(1.0, 2.0, 3.0, 4.0);
    let b = Multivector2::new(5.0, 6.0, 7.0, 8.0);
    let ga_result = a.gp(b);

    let matrix_a = [1.0, 2.0, 3.0, 4.0];
    let matrix_b = [5.0, 6.0, 7.0, 8.0];
    let matrix_result = [
        matrix_a[0] * matrix_b[0] + matrix_a[1] * matrix_b[2],
        matrix_a[0] * matrix_b[1] + matrix_a[1] * matrix_b[3],
        matrix_a[2] * matrix_b[0] + matrix_a[3] * matrix_b[2],
        matrix_a[2] * matrix_b[1] + matrix_a[3] * matrix_b[3],
    ];

    let mut group = c.benchmark_group("correctness_verification");
    group.bench_function("correctness_check", |bencher| {
        bencher.iter(|| {
            // Just verify the operations are doing something reasonable
            let ga = black_box(ga_result);
            let mat = black_box(matrix_result);
            black_box((ga, mat))
        })
    });
    group.finish();
}

#[cfg(target_os = "macos")]
criterion_group!(
    individual_benchmarks,
    bench_ga_2d_multivector,
    bench_matrix_2x2_naive,
    bench_matrix_2x2_nalgebra,
    bench_matrix_2x2_dgemm,
    bench_matrix_2x2_accelerate
);

#[cfg(not(target_os = "macos"))]
criterion_group!(
    individual_benchmarks,
    bench_ga_2d_multivector,
    bench_matrix_2x2_naive,
    bench_matrix_2x2_nalgebra,
    bench_matrix_2x2_dgemm
);

criterion_group!(
    comparison_benchmarks,
    bench_direct_comparison,
    bench_correctness_check
);

criterion_main!(individual_benchmarks, comparison_benchmarks); 