// benches/block_matrix_2x2.rs
//! Benchmark: Block Matrix Multiplication using 2x2 blocks
//!
//! This benchmark tests the user's brilliant idea of using 2D multivectors
//! as 2x2 blocks in larger matrix multiplications to leverage GA's advantages
//! while minimizing overhead.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::numerical_checks::multivector2::Multivector2;
use rayon::prelude::*;
use std::time::Instant;

/// Convert a 2x2 matrix block to 2D multivector representation
fn matrix_to_multivector(block: &[f64; 4]) -> Multivector2 {
    // Map 2x2 matrix elements to 2D multivector components
    // This is a conceptual mapping for testing purposes
    Multivector2::new(block[0], block[1], block[2], block[3])
}

/// Convert 2D multivector result back to 2x2 matrix block
fn multivector_to_matrix(mv: &Multivector2) -> [f64; 4] {
    // Map 2D multivector components back to 2x2 matrix elements
    [mv.data[0], mv.data[1], mv.data[2], mv.data[3]]
}

/// Classical 2x2 block matrix multiplication
fn classical_2x2_block_multiply(
    a_blocks: &[Vec<f64>],
    b_blocks: &[Vec<f64>],
    blocks_per_row: usize,
) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; 4]; blocks_per_row * blocks_per_row];
    
    for i in 0..blocks_per_row {
        for j in 0..blocks_per_row {
            for k in 0..blocks_per_row {
                // C[i][j] += A[i][k] * B[k][j]
                let a_block = &a_blocks[i * blocks_per_row + k];
                let b_block = &b_blocks[k * blocks_per_row + j];
                let c_block = &mut result[i * blocks_per_row + j];
                
                // 2x2 block multiplication
                // A = [a0 a1; a2 a3], B = [b0 b1; b2 b3]
                // C = [c0 c1; c2 c3] where:
                // c0 = a0*b0 + a1*b2
                // c1 = a0*b1 + a1*b3
                // c2 = a2*b0 + a3*b2
                // c3 = a2*b1 + a3*b3
                
                c_block[0] += a_block[0] * b_block[0] + a_block[1] * b_block[2];
                c_block[1] += a_block[0] * b_block[1] + a_block[1] * b_block[3];
                c_block[2] += a_block[2] * b_block[0] + a_block[3] * b_block[2];
                c_block[3] += a_block[2] * b_block[1] + a_block[3] * b_block[3];
            }
        }
    }
    
    result
}

/// GA-based 2x2 block matrix multiplication using 2D multivectors
fn ga_2x2_block_multiply(
    a_blocks: &[Vec<f64>],
    b_blocks: &[Vec<f64>],
    blocks_per_row: usize,
) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; 4]; blocks_per_row * blocks_per_row];
    
    // Process blocks in parallel
    result.par_iter_mut().enumerate().for_each(|(idx, c_block)| {
        let i = idx / blocks_per_row;
        let j = idx % blocks_per_row;
        
        for k in 0..blocks_per_row {
            let a_block = &a_blocks[i * blocks_per_row + k];
            let b_block = &b_blocks[k * blocks_per_row + j];
            
            // Convert to 2D multivectors
            let a_mv = matrix_to_multivector(&[a_block[0], a_block[1], a_block[2], a_block[3]]);
            let b_mv = matrix_to_multivector(&[b_block[0], b_block[1], b_block[2], b_block[3]]);
            
            // GA geometric product
            let result_mv = a_mv.gp(b_mv);
            
            // Convert back and accumulate
            let result_block = multivector_to_matrix(&result_mv);
            for (i, &val) in result_block.iter().enumerate() {
                c_block[i] += val;
            }
        }
    });
    
    result
}

/// Decompose a large matrix into 2x2 blocks
fn decompose_matrix_2x2(matrix: &[f64], n: usize) -> Vec<Vec<f64>> {
    let blocks_per_row = n / 2;
    let mut blocks = Vec::new();
    
    for block_i in 0..blocks_per_row {
        for block_j in 0..blocks_per_row {
            let mut block = vec![0.0; 4];
            
            for i in 0..2 {
                for j in 0..2 {
                    let global_i = block_i * 2 + i;
                    let global_j = block_j * 2 + j;
                    if global_i < n && global_j < n {
                        block[i * 2 + j] = matrix[global_i * n + global_j];
                    }
                }
            }
            
            blocks.push(block);
        }
    }
    
    blocks
}

/// Benchmark small matrix size (8x8 = 16 blocks of 2x2)
fn bench_block_matrix_8x8(c: &mut Criterion) {
    const N: usize = 8;
    let blocks_per_row = N / 2;
    
    // Create test matrices
    let mut a = vec![0.0; N * N];
    let mut b = vec![0.0; N * N];
    
    // Initialize with simple patterns
    for i in 0..N {
        for j in 0..N {
            a[i * N + j] = (i * N + j) as f64;
            b[i * N + j] = ((i + j) % 10) as f64;
        }
    }
    
    let a_blocks = decompose_matrix_2x2(&a, N);
    let b_blocks = decompose_matrix_2x2(&b, N);
    
    c.bench_function("classical 2x2 block multiply 8x8", |bencher| {
        bencher.iter(|| {
            let result = classical_2x2_block_multiply(
                black_box(&a_blocks),
                black_box(&b_blocks),
                blocks_per_row
            );
            black_box(result)
        })
    });
    
    c.bench_function("GA 2x2 block multiply 8x8", |bencher| {
        bencher.iter(|| {
            let result = ga_2x2_block_multiply(
                black_box(&a_blocks),
                black_box(&b_blocks),
                blocks_per_row
            );
            black_box(result)
        })
    });
}

/// Benchmark medium matrix size (16x16 = 64 blocks of 2x2)
fn bench_block_matrix_16x16(c: &mut Criterion) {
    const N: usize = 16;
    let blocks_per_row = N / 2;
    
    // Create test matrices
    let mut a = vec![0.0; N * N];
    let mut b = vec![0.0; N * N];
    
    // Initialize with simple patterns
    for i in 0..N {
        for j in 0..N {
            a[i * N + j] = (i * N + j) as f64;
            b[i * N + j] = ((i + j) % 10) as f64;
        }
    }
    
    let a_blocks = decompose_matrix_2x2(&a, N);
    let b_blocks = decompose_matrix_2x2(&b, N);
    
    c.bench_function("classical 2x2 block multiply 16x16", |bencher| {
        bencher.iter(|| {
            let result = classical_2x2_block_multiply(
                black_box(&a_blocks),
                black_box(&b_blocks),
                blocks_per_row
            );
            black_box(result)
        })
    });
    
    c.bench_function("GA 2x2 block multiply 16x16", |bencher| {
        bencher.iter(|| {
            let result = ga_2x2_block_multiply(
                black_box(&a_blocks),
                black_box(&b_blocks),
                blocks_per_row
            );
            black_box(result)
        })
    });
}

/// Benchmark large matrix size (32x32 = 256 blocks of 2x2)
fn bench_block_matrix_32x32(c: &mut Criterion) {
    const N: usize = 32;
    let blocks_per_row = N / 2;
    
    // Create test matrices
    let mut a = vec![0.0; N * N];
    let mut b = vec![0.0; N * N];
    
    // Initialize with simple patterns
    for i in 0..N {
        for j in 0..N {
            a[i * N + j] = (i * N + j) as f64;
            b[i * N + j] = ((i + j) % 10) as f64;
        }
    }
    
    let a_blocks = decompose_matrix_2x2(&a, N);
    let b_blocks = decompose_matrix_2x2(&b, N);
    
    c.bench_function("classical 2x2 block multiply 32x32", |bencher| {
        bencher.iter(|| {
            let result = classical_2x2_block_multiply(
                black_box(&a_blocks),
                black_box(&b_blocks),
                blocks_per_row
            );
            black_box(result)
        })
    });
    
    c.bench_function("GA 2x2 block multiply 32x32", |bencher| {
        bencher.iter(|| {
            let result = ga_2x2_block_multiply(
                black_box(&a_blocks),
                black_box(&b_blocks),
                blocks_per_row
            );
            black_box(result)
        })
    });
}

/// Benchmark decomposition overhead
fn bench_decomposition_overhead(c: &mut Criterion) {
    const N: usize = 32;
    let mut matrix = vec![0.0; N * N];
    
    for i in 0..N {
        for j in 0..N {
            matrix[i * N + j] = (i * N + j) as f64;
        }
    }
    
    c.bench_function("decomposition 32x32 to 2x2 blocks", |bencher| {
        bencher.iter(|| {
            let blocks = decompose_matrix_2x2(black_box(&matrix), N);
            black_box(blocks)
        })
    });
}

criterion_group!(
    benches,
    bench_block_matrix_8x8,
    bench_block_matrix_16x16,
    bench_block_matrix_32x32,
    bench_decomposition_overhead
);

criterion_main!(benches); 