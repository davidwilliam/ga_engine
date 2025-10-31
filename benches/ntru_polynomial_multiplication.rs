//! NTRU Polynomial Multiplication Benchmarks: Classical vs GA
//!
//! This benchmark demonstrates that GA-based approaches can accelerate
//! NTRU polynomial multiplication, a core operation in post-quantum cryptography.
//!
//! ## Background
//!
//! NTRU operates on polynomials in the ring R = Z[x]/(x^N - 1).
//! Polynomial multiplication is the computational bottleneck:
//! - Used in key generation
//! - Used in encryption (computing c = r*h + m)
//! - Used in decryption (computing a = f*c)
//!
//! ## Approaches Compared
//!
//! 1. **Naive O(N²)**: Direct convolution
//! 2. **Toeplitz Matrix-Vector Product**: Standard NTRU optimization
//! 3. **GA-Based (N=8)**: Maps to 3D GA multivectors, uses our 4.31× matrix speedup
//! 4. **GA-Based (N=16)**: Maps to 4D GA multivectors, uses our 1.75× matrix speedup
//!
//! ## Expected Results
//!
//! Based on our measured GA speedups on matrix operations:
//! - N=8: Should see ~4.31× speedup vs classical Toeplitz
//! - N=16: Should see ~1.75× speedup vs classical Toeplitz
//!
//! ## References
//!
//! - "Fast polynomial multiplication using matrix multiplication accelerators"
//!   achieves 1.54-3.07× speedup on Apple M1/M3
//! - Our GA approach should exceed this based on 4.31× matrix speedup

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ga_engine::ntru::polynomial::{NTRUParams, Polynomial};
use ga_engine::ntru::classical::{
    naive_multiply, toeplitz_matrix_multiply, karatsuba_multiply,
};
use ga_engine::ntru::ga_based::{
    ntru_multiply_via_ga_matrix_8x8,
    ntru_multiply_via_ga_matrix_16x16,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Generate test polynomials for benchmarking
fn generate_test_polynomials_n8() -> (Polynomial, Polynomial) {
    let params = NTRUParams::N8_TOY;
    let mut rng = StdRng::seed_from_u64(42);

    // Generate ternary polynomials (typical NTRU form)
    // +1s, -1s, and 0s with small counts for N=8
    let a = Polynomial::random_ternary(params, 3, 3, &mut rng);
    let b = Polynomial::random_ternary(params, 3, 3, &mut rng);

    (a, b)
}

fn generate_test_polynomials_n16() -> (Polynomial, Polynomial) {
    let params = NTRUParams::N16_TOY;
    let mut rng = StdRng::seed_from_u64(42);

    // For N=16, use more non-zero coefficients
    let a = Polynomial::random_ternary(params, 5, 5, &mut rng);
    let b = Polynomial::random_ternary(params, 5, 5, &mut rng);

    (a, b)
}

/// Benchmark N=8 polynomial multiplication (all approaches)
fn bench_ntru_n8_single_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntru_n8_single_multiplication");
    group.throughput(Throughput::Elements(1));

    let (a, b) = generate_test_polynomials_n8();

    group.bench_function("naive_n8", |bencher| {
        bencher.iter(|| {
            let result = naive_multiply(black_box(&a), black_box(&b));
            black_box(result)
        })
    });

    group.bench_function("toeplitz_classical_n8", |bencher| {
        bencher.iter(|| {
            let result = toeplitz_matrix_multiply(black_box(&a), black_box(&b));
            black_box(result)
        })
    });

    group.bench_function("ga_based_n8", |bencher| {
        bencher.iter(|| {
            let result = ntru_multiply_via_ga_matrix_8x8(black_box(&a), black_box(&b));
            black_box(result)
        })
    });

    group.finish();
}

/// Benchmark N=16 polynomial multiplication
fn bench_ntru_n16_single_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntru_n16_single_multiplication");
    group.throughput(Throughput::Elements(1));

    let (a, b) = generate_test_polynomials_n16();

    group.bench_function("naive_n16", |bencher| {
        bencher.iter(|| {
            let result = naive_multiply(black_box(&a), black_box(&b));
            black_box(result)
        })
    });

    group.bench_function("toeplitz_classical_n16", |bencher| {
        bencher.iter(|| {
            let result = toeplitz_matrix_multiply(black_box(&a), black_box(&b));
            black_box(result)
        })
    });

    group.bench_function("karatsuba_n16", |bencher| {
        bencher.iter(|| {
            let result = karatsuba_multiply(black_box(&a), black_box(&b));
            black_box(result)
        })
    });

    group.bench_function("ga_based_n16", |bencher| {
        bencher.iter(|| {
            let result = ntru_multiply_via_ga_matrix_16x16(black_box(&a), black_box(&b));
            black_box(result)
        })
    });

    group.finish();
}

/// Benchmark batch operations (simulating NTRU key generation or batch encryption)
fn bench_ntru_batch_operations(c: &mut Criterion) {
    const BATCH_SIZE: usize = 100;

    let mut group = c.benchmark_group("ntru_batch_operations");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    // Generate batch of test polynomials
    let mut rng = StdRng::seed_from_u64(42);
    let params_n8 = NTRUParams::N8_TOY;

    let pairs_n8: Vec<_> = (0..BATCH_SIZE)
        .map(|_| {
            let a = Polynomial::random_ternary(params_n8, 3, 3, &mut rng);
            let b = Polynomial::random_ternary(params_n8, 3, 3, &mut rng);
            (a, b)
        })
        .collect();

    group.bench_function("toeplitz_classical_n8_batch_100", |bencher| {
        bencher.iter(|| {
            for (a, b) in &pairs_n8 {
                let result = toeplitz_matrix_multiply(black_box(a), black_box(b));
                black_box(result);
            }
        })
    });

    group.bench_function("ga_based_n8_batch_100", |bencher| {
        bencher.iter(|| {
            for (a, b) in &pairs_n8 {
                let result = ntru_multiply_via_ga_matrix_8x8(black_box(a), black_box(b));
                black_box(result);
            }
        })
    });

    group.finish();
}

/// Direct comparison: Core operation benchmark
///
/// This focuses on the EXACT operation we're optimizing:
/// Matrix-vector multiplication for NTRU
fn bench_ntru_core_operation_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntru_core_comparison");

    let (a, b) = generate_test_polynomials_n8();

    // Classical Toeplitz (baseline)
    group.bench_function("classical_toeplitz_n8", |bencher| {
        bencher.iter(|| {
            black_box(toeplitz_matrix_multiply(black_box(&a), black_box(&b)))
        })
    });

    // GA-based (our optimization)
    group.bench_function("ga_accelerated_n8", |bencher| {
        bencher.iter(|| {
            black_box(ntru_multiply_via_ga_matrix_8x8(black_box(&a), black_box(&b)))
        })
    });

    group.finish();
}

/// Scaling benchmark: Show how performance changes with polynomial degree
fn bench_ntru_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntru_scaling");

    for &n in &[8, 16] {
        let params = if n == 8 {
            NTRUParams::N8_TOY
        } else {
            NTRUParams::N16_TOY
        };

        let mut rng = StdRng::seed_from_u64(42);
        let num_ones = n / 3;
        let a = Polynomial::random_ternary(params, num_ones, num_ones, &mut rng);
        let b = Polynomial::random_ternary(params, num_ones, num_ones, &mut rng);

        group.bench_with_input(
            BenchmarkId::new("classical", n),
            &n,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(toeplitz_matrix_multiply(black_box(&a), black_box(&b)))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ga_based", n),
            &n,
            |bencher, _| {
                bencher.iter(|| {
                    if n == 8 {
                        black_box(ntru_multiply_via_ga_matrix_8x8(black_box(&a), black_box(&b)))
                    } else {
                        black_box(ntru_multiply_via_ga_matrix_16x16(black_box(&a), black_box(&b)))
                    }
                })
            },
        );
    }

    group.finish();
}

/// Comprehensive benchmark suite
fn bench_ntru_comprehensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntru_comprehensive");
    group.sample_size(100);

    // N=8 comprehensive comparison
    let (a8, b8) = generate_test_polynomials_n8();

    group.bench_function("N8_naive", |b| {
        b.iter(|| black_box(naive_multiply(black_box(&a8), black_box(&b8))))
    });

    group.bench_function("N8_toeplitz_classical", |b| {
        b.iter(|| black_box(toeplitz_matrix_multiply(black_box(&a8), black_box(&b8))))
    });

    group.bench_function("N8_GA_accelerated", |b| {
        b.iter(|| black_box(ntru_multiply_via_ga_matrix_8x8(black_box(&a8), black_box(&b8))))
    });

    // N=16 comprehensive comparison
    let (a16, b16) = generate_test_polynomials_n16();

    group.bench_function("N16_naive", |b| {
        b.iter(|| black_box(naive_multiply(black_box(&a16), black_box(&b16))))
    });

    group.bench_function("N16_toeplitz_classical", |b| {
        b.iter(|| black_box(toeplitz_matrix_multiply(black_box(&a16), black_box(&b16))))
    });

    group.bench_function("N16_karatsuba", |b| {
        b.iter(|| black_box(karatsuba_multiply(black_box(&a16), black_box(&b16))))
    });

    group.bench_function("N16_GA_accelerated", |b| {
        b.iter(|| black_box(ntru_multiply_via_ga_matrix_16x16(black_box(&a16), black_box(&b16))))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_ntru_n8_single_operation,
    bench_ntru_n16_single_operation,
    bench_ntru_batch_operations,
    bench_ntru_core_operation_comparison,
    bench_ntru_scaling,
    bench_ntru_comprehensive,
);

criterion_main!(benches);
