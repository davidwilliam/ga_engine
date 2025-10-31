//! Test: Does Block Decomposition Preserve GA Advantages?
//!
//! Question: If GA wins for 8×8, why doesn't it win for 16×16 via 2×2 blocks?
//!
//! Theory:
//! - 16×16 matrix = 2×2 grid of 8×8 blocks
//! - Each 8×8 block: GA is 2.54× faster
//! - Should block approach also be faster?
//!
//! This benchmark tests that hypothesis.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::ntru::polynomial::{NTRUParams, Polynomial};
use ga_engine::ntru::classical::toeplitz_matrix_multiply;
use ga_engine::ntru::ga_based::{ntru_multiply_via_ga_matrix_16x16, matrix_8x8_to_multivector3d};
use ga_engine::ga::geometric_product_full;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn generate_test_polynomials_n16() -> (Polynomial, Polynomial) {
    let params = NTRUParams::N16_TOY;
    let mut rng = StdRng::seed_from_u64(42);
    let a = Polynomial::random_ternary(params, 5, 5, &mut rng);
    let b = Polynomial::random_ternary(params, 5, 5, &mut rng);
    (a, b)
}

/// Direct 16×16 Toeplitz matrix-vector multiply
fn toeplitz_16x16_direct(matrix: &[f64; 256], vec: &[f64; 16]) -> [f64; 16] {
    let mut result = [0.0; 16];
    for i in 0..16 {
        for j in 0..16 {
            result[i] += matrix[i * 16 + j] * vec[j];
        }
    }
    result
}

/// Block-based 16×16 using 2×2 grid of 8×8 GA blocks
fn toeplitz_16x16_block_ga(matrix: &[f64; 256], vec: &[f64; 16]) -> [f64; 16] {
    // Decompose 16×16 matrix into 2×2 grid of 8×8 blocks
    let mut blocks = [[0.0f64; 64]; 4];

    for block_row in 0..2 {
        for block_col in 0..2 {
            let block_idx = block_row * 2 + block_col;
            for i in 0..8 {
                for j in 0..8 {
                    let global_i = block_row * 8 + i;
                    let global_j = block_col * 8 + j;
                    blocks[block_idx][i * 8 + j] = matrix[global_i * 16 + global_j];
                }
            }
        }
    }

    // Convert each 8×8 block to GA multivector
    let mv_blocks: Vec<[f64; 8]> = blocks.iter()
        .map(|block| matrix_8x8_to_multivector3d(block))
        .collect();

    // Decompose vector into 2 segments of 8
    let vec_segments = [[vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7]],
                        [vec[8], vec[9], vec[10], vec[11], vec[12], vec[13], vec[14], vec[15]]];

    let mut result = [0.0; 16];

    // Block matrix-vector multiplication: C = A × v
    // C[0:8]  = A[0,0] × v[0:8]  + A[0,1] × v[8:16]
    // C[8:16] = A[1,0] × v[0:8]  + A[1,1] × v[8:16]

    for block_row in 0..2 {
        for block_col in 0..2 {
            let block_idx = block_row * 2 + block_col;
            let mv_block = &mv_blocks[block_idx];
            let vec_seg = &vec_segments[block_col];

            // GA geometric product
            let mut mv_result = [0.0; 8];
            geometric_product_full(mv_block, vec_seg, &mut mv_result);

            // Accumulate into result
            let result_start = block_row * 8;
            for i in 0..8 {
                result[result_start + i] += mv_result[i];
            }
        }
    }

    result
}

/// Benchmark: Direct vs Block Decomposition
fn bench_direct_vs_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("n16_direct_vs_block");

    let (a, b) = generate_test_polynomials_n16();

    // Setup: Create 16×16 Toeplitz matrix (excluded from timing)
    let toeplitz = {
        let mut matrix = [0.0f64; 256];
        for i in 0..16 {
            for j in 0..16 {
                let idx = (16 + i - j) % 16;
                matrix[i * 16 + j] = a.coeffs[idx] as f64;
            }
        }
        matrix
    };

    let b_vec: [f64; 16] = [
        b.coeffs[0] as f64, b.coeffs[1] as f64, b.coeffs[2] as f64, b.coeffs[3] as f64,
        b.coeffs[4] as f64, b.coeffs[5] as f64, b.coeffs[6] as f64, b.coeffs[7] as f64,
        b.coeffs[8] as f64, b.coeffs[9] as f64, b.coeffs[10] as f64, b.coeffs[11] as f64,
        b.coeffs[12] as f64, b.coeffs[13] as f64, b.coeffs[14] as f64, b.coeffs[15] as f64,
    ];

    // Benchmark: Direct classical 16×16
    group.bench_function("classical_16x16_direct", |bencher| {
        bencher.iter(|| {
            black_box(toeplitz_16x16_direct(black_box(&toeplitz), black_box(&b_vec)))
        })
    });

    // Benchmark: Block-based using 2×2 of 8×8 GA
    group.bench_function("ga_16x16_block_2x2", |bencher| {
        bencher.iter(|| {
            black_box(toeplitz_16x16_block_ga(black_box(&toeplitz), black_box(&b_vec)))
        })
    });

    // Benchmark: Direct GA 16×16 (our current implementation)
    group.bench_function("ga_16x16_direct", |bencher| {
        bencher.iter(|| {
            black_box(ntru_multiply_via_ga_matrix_16x16(black_box(&a), black_box(&b)))
        })
    });

    group.finish();
}

/// Theoretical analysis
fn bench_operations_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("operations_count_analysis");

    // Single 8×8 GA operation (baseline)
    let mv_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mv_b = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    group.bench_function("single_8x8_ga_op", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 8];
            geometric_product_full(black_box(&mv_a), black_box(&mv_b), &mut result);
            black_box(result)
        })
    });

    // Four 8×8 GA operations (what block approach needs)
    group.bench_function("four_8x8_ga_ops", |bencher| {
        bencher.iter(|| {
            for _ in 0..4 {
                let mut result = [0.0; 8];
                geometric_product_full(black_box(&mv_a), black_box(&mv_b), &mut result);
                black_box(result);
            }
        })
    });

    // Eight 8×8 GA operations (actual block multiplication cost)
    group.bench_function("eight_8x8_ga_ops", |bencher| {
        bencher.iter(|| {
            for _ in 0..8 {
                let mut result = [0.0; 8];
                geometric_product_full(black_box(&mv_a), black_box(&mv_b), &mut result);
                black_box(result);
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_direct_vs_block,
    bench_operations_count,
);

criterion_main!(benches);
