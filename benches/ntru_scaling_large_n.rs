//! NTRU Polynomial Multiplication Scaling Benchmarks
//!
//! This benchmark tests how NTRU polynomial multiplication performance scales
//! with increasing N values, from small toy parameters (N=8) to production
//! NIST parameters (N=509).
//!
//! ## Purpose
//!
//! To compare our GA-based approach's performance characteristics with the
//! AMX hardware accelerator paper results:
//! - "Fast polynomial multiplication using matrix multiplication accelerators"
//!   by Gazzoni Filho et al. achieves 1.54-3.07× speedup on Apple M1/M3
//!   for production NTRU parameters (N=509, 677, 821, 701)
//!
//! ## Test Parameters
//!
//! - N=8, 16: GA-accelerated (3D and 4D multivectors)
//! - N=32, 64, 128, 256, 509: Classical methods only (Toeplitz, Karatsuba)
//!
//! ## Expected Results
//!
//! - Small N (8, 16): GA shows significant advantage
//! - Medium N (32, 64): Classical methods may dominate
//! - Large N (128+): Karatsuba or NTT would be optimal
//! - N=509: Direct comparison point with AMX paper

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ga_engine::ntru::polynomial::{NTRUParams, Polynomial};
use ga_engine::ntru::classical::{naive_multiply, toeplitz_matrix_multiply, karatsuba_multiply};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Generate test polynomials for a given parameter set
fn generate_test_polynomials(params: NTRUParams) -> (Polynomial, Polynomial) {
    let mut rng = StdRng::seed_from_u64(42);

    // Use ternary polynomials with ~1/3 non-zero coefficients (typical for NTRU)
    let num_ones = params.n / 6;
    let num_neg_ones = params.n / 6;

    let a = Polynomial::random_ternary(params, num_ones, num_neg_ones, &mut rng);
    let b = Polynomial::random_ternary(params, num_ones, num_neg_ones, &mut rng);

    (a, b)
}

/// Benchmark Toeplitz matrix multiplication across all N values
fn bench_toeplitz_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntru_toeplitz_scaling");

    let params_list = [
        NTRUParams::N8_TOY,
        NTRUParams::N16_TOY,
        NTRUParams::N32_TOY,
        NTRUParams::N64_TOY,
        NTRUParams::N128_TOY,
        NTRUParams::N256_TOY,
        // N=509 takes too long for quick benchmarks, uncomment for full comparison
        // NTRUParams::NIST_LEVEL1,
    ];

    for params in &params_list {
        let (a, b) = generate_test_polynomials(*params);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("toeplitz", params.n),
            &params.n,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(toeplitz_matrix_multiply(black_box(&a), black_box(&b)))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark Karatsuba multiplication for larger N values
fn bench_karatsuba_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntru_karatsuba_scaling");

    // Karatsuba is more efficient for larger N
    let params_list = [
        NTRUParams::N16_TOY,
        NTRUParams::N32_TOY,
        NTRUParams::N64_TOY,
        NTRUParams::N128_TOY,
        NTRUParams::N256_TOY,
        // NTRUParams::NIST_LEVEL1,
    ];

    for params in &params_list {
        let (a, b) = generate_test_polynomials(*params);

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("karatsuba", params.n),
            &params.n,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(karatsuba_multiply(black_box(&a), black_box(&b)))
                })
            },
        );
    }

    group.finish();
}

/// Direct comparison at different scales
fn bench_ntru_comparison_by_size(c: &mut Criterion) {
    // Small N: Compare all methods including GA
    {
        let mut group = c.benchmark_group("ntru_small_n_comparison");

        for &n in &[8, 16] {
            let params = match n {
                8 => NTRUParams::N8_TOY,
                16 => NTRUParams::N16_TOY,
                _ => unreachable!(),
            };

            let (a, b) = generate_test_polynomials(params);

            group.bench_with_input(
                BenchmarkId::new("toeplitz", n),
                &n,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(toeplitz_matrix_multiply(black_box(&a), black_box(&b)))
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("karatsuba", n),
                &n,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(karatsuba_multiply(black_box(&a), black_box(&b)))
                    })
                },
            );

            // GA-based methods (only available for N=8 and N=16)
            if n == 8 {
                use ga_engine::ntru::ga_based::ntru_multiply_via_ga_matrix_8x8;
                group.bench_with_input(
                    BenchmarkId::new("ga_accelerated", n),
                    &n,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(ntru_multiply_via_ga_matrix_8x8(black_box(&a), black_box(&b)))
                        })
                    },
                );
            } else if n == 16 {
                use ga_engine::ntru::ga_based::ntru_multiply_via_ga_matrix_16x16;
                group.bench_with_input(
                    BenchmarkId::new("ga_accelerated", n),
                    &n,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(ntru_multiply_via_ga_matrix_16x16(black_box(&a), black_box(&b)))
                        })
                    },
                );
            }
        }

        group.finish();
    }

    // Medium N: Classical methods + GA for N=32 (to demonstrate GA doesn't scale)
    {
        let mut group = c.benchmark_group("ntru_medium_n_comparison");

        for &n in &[32, 64] {
            let params = match n {
                32 => NTRUParams::N32_TOY,
                64 => NTRUParams::N64_TOY,
                _ => unreachable!(),
            };

            let (a, b) = generate_test_polynomials(params);

            group.bench_with_input(
                BenchmarkId::new("toeplitz", n),
                &n,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(toeplitz_matrix_multiply(black_box(&a), black_box(&b)))
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("karatsuba", n),
                &n,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(karatsuba_multiply(black_box(&a), black_box(&b)))
                    })
                },
            );

            // Include GA using generic implementation to test scaling
            use ga_engine::ntru::ga_based::ga_multiply_generic;
            if n == 32 {
                group.bench_with_input(
                    BenchmarkId::new("ga_generic", n),
                    &n,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(ga_multiply_generic::<5>(black_box(&a), black_box(&b)))
                        })
                    },
                );
            } else if n == 64 {
                group.bench_with_input(
                    BenchmarkId::new("ga_generic", n),
                    &n,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(ga_multiply_generic::<6>(black_box(&a), black_box(&b)))
                        })
                    },
                );
            }
        }

        group.finish();
    }

    // Large N: Classical methods only (longer sample time)
    {
        let mut group = c.benchmark_group("ntru_large_n_comparison");
        group.sample_size(20); // Reduce sample size for large N

        for &n in &[128, 256] {
            let params = match n {
                128 => NTRUParams::N128_TOY,
                256 => NTRUParams::N256_TOY,
                _ => unreachable!(),
            };

            let (a, b) = generate_test_polynomials(params);

            group.bench_with_input(
                BenchmarkId::new("toeplitz", n),
                &n,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(toeplitz_matrix_multiply(black_box(&a), black_box(&b)))
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new("karatsuba", n),
                &n,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(karatsuba_multiply(black_box(&a), black_box(&b)))
                    })
                },
            );

            // Block-based GA for N=128 (using 8×8 block decomposition)
            if n == 128 {
                use ga_engine::ntru::ga_based::ga_multiply_n128_block;
                group.bench_with_input(
                    BenchmarkId::new("ga_block_8x8", n),
                    &n,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(ga_multiply_n128_block(black_box(&a), black_box(&b)))
                        })
                    },
                );
            }
        }

        group.finish();
    }
}

/// Benchmark specifically for N=509 (NIST Level 1 parameter)
/// This is the direct comparison point with the AMX accelerator paper
fn bench_nist_level1_n509(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntru_nist_level1");
    group.sample_size(10); // Very small sample size for N=509

    let params = NTRUParams::NIST_LEVEL1;
    let (a, b) = generate_test_polynomials(params);

    // Naive is O(N²) and will be very slow for N=509
    // Uncomment only if you want to wait a long time
    // group.bench_function("naive_n509", |bencher| {
    //     bencher.iter(|| {
    //         black_box(naive_multiply(black_box(&a), black_box(&b)))
    //     })
    // });

    group.bench_function("toeplitz_n509", |bencher| {
        bencher.iter(|| {
            black_box(toeplitz_matrix_multiply(black_box(&a), black_box(&b)))
        })
    });

    group.bench_function("karatsuba_n509", |bencher| {
        bencher.iter(|| {
            black_box(karatsuba_multiply(black_box(&a), black_box(&b)))
        })
    });

    group.finish();
}

/// Batch operations for realistic workload simulation
fn bench_batch_operations_scaling(c: &mut Criterion) {
    const BATCH_SIZE: usize = 10; // Smaller batch for larger N

    let mut group = c.benchmark_group("ntru_batch_scaling");

    for &n in &[8, 16, 32, 64] {
        let params = match n {
            8 => NTRUParams::N8_TOY,
            16 => NTRUParams::N16_TOY,
            32 => NTRUParams::N32_TOY,
            64 => NTRUParams::N64_TOY,
            _ => unreachable!(),
        };

        let mut rng = StdRng::seed_from_u64(42);
        let num_ones = params.n / 6;

        let pairs: Vec<_> = (0..BATCH_SIZE)
            .map(|_| {
                let a = Polynomial::random_ternary(params, num_ones, num_ones, &mut rng);
                let b = Polynomial::random_ternary(params, num_ones, num_ones, &mut rng);
                (a, b)
            })
            .collect();

        group.throughput(Throughput::Elements(BATCH_SIZE as u64));
        group.bench_with_input(
            BenchmarkId::new("toeplitz_batch", n),
            &n,
            |bencher, _| {
                bencher.iter(|| {
                    for (a, b) in &pairs {
                        black_box(toeplitz_matrix_multiply(black_box(a), black_box(b)));
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("karatsuba_batch", n),
            &n,
            |bencher, _| {
                bencher.iter(|| {
                    for (a, b) in &pairs {
                        black_box(karatsuba_multiply(black_box(a), black_box(b)));
                    }
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_toeplitz_scaling,
    bench_karatsuba_scaling,
    bench_ntru_comparison_by_size,
    bench_nist_level1_n509,
    bench_batch_operations_scaling,
);

criterion_main!(benches);
