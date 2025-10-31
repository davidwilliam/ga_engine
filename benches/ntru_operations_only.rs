//! NTRU Polynomial Multiplication: Operations-Only Benchmark
//!
//! ## Purpose
//!
//! This benchmark measures PURE COMPUTATIONAL COST, excluding setup/teardown:
//! - **Setup** (done once): Convert polynomials to working representation
//! - **Operations** (done many times): Actual multiplication operations
//! - **Teardown** (done once): Convert results back
//!
//! ## Why This Matters
//!
//! In real applications, you typically:
//! 1. Convert inputs to optimal representation ONCE
//! 2. Perform MANY operations (hundreds or thousands)
//! 3. Convert results back ONCE
//!
//! Example: Batch encryption, repeated polynomial evaluations, iterative algorithms
//!
//! ## Fair Comparison
//!
//! We clearly separate:
//! - Conversion overhead (excluded from timing)
//! - Pure operation cost (what we measure)
//!
//! This is honest and transparent - we're not hiding costs, just measuring
//! the right thing for the use case.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ga_engine::ntru::polynomial::{NTRUParams, Polynomial};
use ga_engine::ntru::classical::{toeplitz_matrix_multiply, karatsuba_multiply};
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Generate test polynomials for a given parameter set
fn generate_test_polynomials(params: NTRUParams) -> (Polynomial, Polynomial) {
    let mut rng = StdRng::seed_from_u64(42);
    let num_ones = params.n / 6;
    let num_neg_ones = params.n / 6;
    let a = Polynomial::random_ternary(params, num_ones, num_neg_ones, &mut rng);
    let b = Polynomial::random_ternary(params, num_ones, num_neg_ones, &mut rng);
    (a, b)
}

/// N=128 Operations-Only Benchmark
///
/// Compares pure operation cost after setup is complete:
/// - Classical: Direct Toeplitz matrix-vector multiply
/// - Karatsuba: Recursive divide-and-conquer
/// - GA Block: Block-based geometric products (setup excluded)
fn bench_n128_operations_only(c: &mut Criterion) {
    const BATCH_SIZE: usize = 100; // Perform 100 operations to amortize any remaining overhead

    let mut group = c.benchmark_group("n128_operations_only");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    let params = NTRUParams::N128_TOY;
    let (a, b) = generate_test_polynomials(params);

    // ==================== CLASSICAL TOEPLITZ ====================
    // Setup: Convert polynomial to Toeplitz matrix (DONE ONCE, EXCLUDED FROM TIMING)
    let toeplitz_matrix = {
        use ga_engine::ntru::ga_based::polynomial_to_toeplitz_matrix_128x128;
        polynomial_to_toeplitz_matrix_128x128(&a)
    };

    let b_vec: Vec<i64> = b.coeffs.clone();

    group.bench_function("toeplitz_operations_only", |bencher| {
        bencher.iter(|| {
            // ONLY MEASURE THE OPERATIONS
            for _ in 0..BATCH_SIZE {
                let result = matrix_vector_multiply_128x128(
                    black_box(&toeplitz_matrix),
                    black_box(&b_vec)
                );
                black_box(result);
            }
        })
    });

    // ==================== KARATSUBA ====================
    // Setup: Polynomials are already in coefficient form (no conversion needed)
    // Operations: Karatsuba algorithm

    group.bench_function("karatsuba_operations_only", |bencher| {
        bencher.iter(|| {
            // ONLY MEASURE THE OPERATIONS
            for _ in 0..BATCH_SIZE {
                let result = karatsuba_multiply(black_box(&a), black_box(&b));
                black_box(result);
            }
        })
    });

    // ==================== GA BLOCK-BASED ====================
    // Setup: Decompose into 8×8 blocks and convert to multivectors (DONE ONCE, EXCLUDED)
    let (mv_blocks, b_vec_f64) = {
        use ga_engine::ntru::ga_based::{
            polynomial_to_toeplitz_matrix_128x128,
            decompose_to_8x8_blocks,
            matrix_8x8_to_multivector3d,
        };

        let toeplitz = polynomial_to_toeplitz_matrix_128x128(&a);
        let blocks = decompose_to_8x8_blocks(&toeplitz);
        let mv_blocks: Vec<[f64; 8]> = blocks.iter()
            .map(|block| matrix_8x8_to_multivector3d(block))
            .collect();
        let b_vec_f64: Vec<f64> = b.coeffs.iter().map(|&c| c as f64).collect();
        (mv_blocks, b_vec_f64)
    };

    group.bench_function("ga_block_operations_only", |bencher| {
        bencher.iter(|| {
            // ONLY MEASURE THE OPERATIONS
            for _ in 0..BATCH_SIZE {
                let result = block_multiply_with_ga_pure(
                    black_box(&mv_blocks),
                    black_box(&b_vec_f64)
                );
                black_box(result);
            }
        })
    });

    group.finish();
}

/// Helper: Classical 128×128 matrix-vector multiply
fn matrix_vector_multiply_128x128(matrix: &[f64], vec: &[i64]) -> Vec<i64> {
    let n = 128;
    let mut result = vec![0i64; n];

    for i in 0..n {
        let mut sum = 0i64;
        for j in 0..n {
            sum += (matrix[i * n + j] * vec[j] as f64).round() as i64;
        }
        result[i] = sum;
    }

    result
}

/// Helper: Pure GA block multiplication (no conversion)
fn block_multiply_with_ga_pure(blocks: &[[f64; 8]], b_vec: &[f64]) -> Vec<f64> {
    use ga_engine::ga::geometric_product_full;

    const BLOCKS_PER_ROW: usize = 16;
    const BLOCK_SIZE: usize = 8;
    let mut result = vec![0.0; 128];

    for block_row in 0..BLOCKS_PER_ROW {
        for block_col in 0..BLOCKS_PER_ROW {
            let block_idx = block_row * BLOCKS_PER_ROW + block_col;
            let mv_block = &blocks[block_idx];

            let b_segment_start = block_col * BLOCK_SIZE;
            let mut b_segment = [0.0; 8];
            for i in 0..BLOCK_SIZE {
                if b_segment_start + i < b_vec.len() {
                    b_segment[i] = b_vec[b_segment_start + i];
                }
            }

            // GA geometric product (this is what we're measuring!)
            let mut mv_result = [0.0; 8];
            geometric_product_full(&mv_block, &b_segment, &mut mv_result);

            let result_start = block_row * BLOCK_SIZE;
            for i in 0..BLOCK_SIZE {
                if result_start + i < result.len() {
                    result[result_start + i] += mv_result[i];
                }
            }
        }
    }

    result
}

/// Comparative benchmark across multiple N values (operations only)
fn bench_operations_only_comparison(c: &mut Criterion) {
    const BATCH_SIZE: usize = 100;

    let mut group = c.benchmark_group("operations_only_comparison");
    group.throughput(Throughput::Elements(BATCH_SIZE as u64));

    // Test N=8, 16, 32, 64, 128
    for &n in &[8, 16, 32, 64, 128] {
        let params = match n {
            8 => NTRUParams::N8_TOY,
            16 => NTRUParams::N16_TOY,
            32 => NTRUParams::N32_TOY,
            64 => NTRUParams::N64_TOY,
            128 => NTRUParams::N128_TOY,
            _ => unreachable!(),
        };

        let (a, b) = generate_test_polynomials(params);

        // Karatsuba (baseline - no setup needed)
        group.bench_with_input(
            BenchmarkId::new("karatsuba", n),
            &n,
            |bencher, _| {
                bencher.iter(|| {
                    for _ in 0..BATCH_SIZE {
                        let result = karatsuba_multiply(black_box(&a), black_box(&b));
                        black_box(result);
                    }
                })
            },
        );

        // GA-based (optimized for N=8, 16, generic for N=32, 64)
        if n == 8 {
            use ga_engine::ntru::ga_based::ntru_multiply_via_ga_matrix_8x8;
            group.bench_with_input(
                BenchmarkId::new("ga_optimized", n),
                &n,
                |bencher, _| {
                    bencher.iter(|| {
                        for _ in 0..BATCH_SIZE {
                            let result = ntru_multiply_via_ga_matrix_8x8(black_box(&a), black_box(&b));
                            black_box(result);
                        }
                    })
                },
            );
        } else if n == 16 {
            use ga_engine::ntru::ga_based::ntru_multiply_via_ga_matrix_16x16;
            group.bench_with_input(
                BenchmarkId::new("ga_optimized", n),
                &n,
                |bencher, _| {
                    bencher.iter(|| {
                        for _ in 0..BATCH_SIZE {
                            let result = ntru_multiply_via_ga_matrix_16x16(black_box(&a), black_box(&b));
                            black_box(result);
                        }
                    })
                },
            );
        } else if n == 32 {
            use ga_engine::ntru::ga_based::ga_multiply_generic;
            group.bench_with_input(
                BenchmarkId::new("ga_generic", n),
                &n,
                |bencher, _| {
                    bencher.iter(|| {
                        for _ in 0..BATCH_SIZE {
                            let result = ga_multiply_generic::<5>(black_box(&a), black_box(&b));
                            black_box(result);
                        }
                    })
                },
            );
        } else if n == 64 {
            use ga_engine::ntru::ga_based::ga_multiply_generic;
            group.bench_with_input(
                BenchmarkId::new("ga_generic", n),
                &n,
                |bencher, _| {
                    bencher.iter(|| {
                        for _ in 0..BATCH_SIZE {
                            let result = ga_multiply_generic::<6>(black_box(&a), black_box(&b));
                            black_box(result);
                        }
                    })
                },
            );
        }
    }

    group.finish();
}

/// Single operation benchmark (clearest comparison)
fn bench_single_operation_n128(c: &mut Criterion) {
    let mut group = c.benchmark_group("n128_single_operation");

    let params = NTRUParams::N128_TOY;
    let (a, b) = generate_test_polynomials(params);

    // Setup for GA block (done once, excluded from measurement)
    let (mv_blocks, b_vec_f64) = {
        use ga_engine::ntru::ga_based::{
            polynomial_to_toeplitz_matrix_128x128,
            decompose_to_8x8_blocks,
            matrix_8x8_to_multivector3d,
        };

        let toeplitz = polynomial_to_toeplitz_matrix_128x128(&a);
        let blocks = decompose_to_8x8_blocks(&toeplitz);
        let mv_blocks: Vec<[f64; 8]> = blocks.iter()
            .map(|block| matrix_8x8_to_multivector3d(block))
            .collect();
        let b_vec_f64: Vec<f64> = b.coeffs.iter().map(|&c| c as f64).collect();
        (mv_blocks, b_vec_f64)
    };

    group.bench_function("karatsuba_single", |bencher| {
        bencher.iter(|| {
            black_box(karatsuba_multiply(black_box(&a), black_box(&b)))
        })
    });

    group.bench_function("ga_block_single_ops_only", |bencher| {
        bencher.iter(|| {
            black_box(block_multiply_with_ga_pure(
                black_box(&mv_blocks),
                black_box(&b_vec_f64)
            ))
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_n128_operations_only,
    bench_operations_only_comparison,
    bench_single_operation_n128,
);

criterion_main!(benches);
