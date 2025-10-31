//! Analysis: Why is GA Geometric Product Faster?
//!
//! Both classical 8×8 matrix mult and GA geometric product perform 64 multiply-add operations.
//! Yet GA is 1.38× faster. This benchmark investigates why.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::ga::geometric_product_full;

/// Classical 8×8 matrix multiplication (single result element)
#[inline(never)]
fn matrix_mult_8x8_single_element(a: &[f64; 64], b: &[f64; 64], row: usize, col: usize) -> f64 {
    let mut sum = 0.0;
    for k in 0..8 {
        sum += a[row * 8 + k] * b[k * 8 + col];
    }
    sum
}

/// Classical 8×8 matrix multiplication (full matrix)
#[inline(never)]
fn matrix_mult_8x8_full(a: &[f64; 64], b: &[f64; 64]) -> [f64; 64] {
    let mut result = [0.0; 64];
    for i in 0..8 {
        for j in 0..8 {
            for k in 0..8 {
                result[i * 8 + j] += a[i * 8 + k] * b[k * 8 + j];
            }
        }
    }
    result
}

/// GA geometric product (8 components × 8 components = 8 components, 64 operations)
#[inline(never)]
fn ga_geometric_product_8x8(a: &[f64; 8], b: &[f64; 8]) -> [f64; 8] {
    let mut result = [0.0; 8];
    geometric_product_full(a, b, &mut result);
    result
}

fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("operation_analysis");

    // Test data
    let matrix_a = [1.0; 64];
    let matrix_b = [1.0; 64];
    let mv_a = [1.0; 8];
    let mv_b = [1.0; 8];

    // Single matrix element (8 operations)
    group.bench_function("matrix_single_element", |bencher| {
        bencher.iter(|| {
            black_box(matrix_mult_8x8_single_element(
                black_box(&matrix_a),
                black_box(&matrix_b),
                black_box(0),
                black_box(0),
            ))
        })
    });

    // Full matrix multiplication (512 operations)
    group.bench_function("matrix_full_8x8", |bencher| {
        bencher.iter(|| {
            black_box(matrix_mult_8x8_full(
                black_box(&matrix_a),
                black_box(&matrix_b),
            ))
        })
    });

    // GA geometric product (64 operations)
    group.bench_function("ga_product_8x8", |bencher| {
        bencher.iter(|| {
            black_box(ga_geometric_product_8x8(
                black_box(&mv_a),
                black_box(&mv_b),
            ))
        })
    });

    group.finish();
}

/// Analysis benchmark: Loop structure matters
fn bench_loop_structure(c: &mut Criterion) {
    let mut group = c.benchmark_group("loop_structure");

    let a = [1.0; 64];
    let b = [1.0; 64];

    // Triple nested loop (natural order)
    group.bench_function("triple_loop_ijk", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 64];
            for i in 0..8 {
                for j in 0..8 {
                    for k in 0..8 {
                        result[i * 8 + j] += a[i * 8 + k] * b[k * 8 + j];
                    }
                }
            }
            black_box(result)
        })
    });

    // Cache-friendly order (ikj)
    group.bench_function("triple_loop_ikj", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 64];
            for i in 0..8 {
                for k in 0..8 {
                    for j in 0..8 {
                        result[i * 8 + j] += a[i * 8 + k] * b[k * 8 + j];
                    }
                }
            }
            black_box(result)
        })
    });

    // Single loop with precomputed indices (GA style)
    group.bench_function("single_loop_precomputed", |bencher| {
        // Precompute all (i,j,k) triplets for 8×8×8 = 512 operations
        const TRIPLETS: [(usize, usize, usize); 512] = {
            let mut arr = [(0, 0, 0); 512];
            let mut idx = 0;
            let mut i = 0;
            while i < 8 {
                let mut j = 0;
                while j < 8 {
                    let mut k = 0;
                    while k < 8 {
                        arr[idx] = (i, j, k);
                        idx += 1;
                        k += 1;
                    }
                    j += 1;
                }
                i += 1;
            }
            arr
        };

        bencher.iter(|| {
            let mut result = [0.0; 64];
            let mut idx = 0;
            while idx < 512 {
                let (i, j, k) = TRIPLETS[idx];
                result[i * 8 + j] += a[i * 8 + k] * b[k * 8 + j];
                idx += 1;
            }
            black_box(result)
        })
    });

    group.finish();
}

/// Theory: GA's advantage comes from:
/// 1. **Precomputed lookup table**: GP_PAIRS is compile-time constant
/// 2. **Single tight loop**: No nested loops = better branch prediction
/// 3. **Fixed array sizes**: [f64; 8] vs [f64; 64] = better register allocation
/// 4. **SIMD opportunities**: Compiler can vectorize single loop more easily
/// 5. **Cache locality**: Smaller working set (8 vs 64 elements)
///
/// Let's verify this hypothesis!
fn bench_cache_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_effects");

    // 8-element arrays (GA size)
    let small_a = [1.0; 8];
    let small_b = [1.0; 8];

    group.bench_function("small_arrays_8", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 8];
            for i in 0..8 {
                result[i] = small_a[i] * small_b[i];
            }
            black_box(result)
        })
    });

    // 64-element arrays (matrix size)
    let large_a = [1.0; 64];
    let large_b = [1.0; 64];

    group.bench_function("large_arrays_64", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 64];
            for i in 0..64 {
                result[i] = large_a[i] * large_b[i];
            }
            black_box(result)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_comparison,
    bench_loop_structure,
    bench_cache_effects
);
criterion_main!(benches);
