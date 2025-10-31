//! Test: Matrix × Vector as Sparse Matrix Multiplication
//!
//! Hypothesis: Matrix × Vector = Matrix × Sparse_Matrix
//! If we represent v as a sparse matrix with v in first column, zeros elsewhere,
//! can we use block GA matrix multiplication to accelerate it?
//!
//! This would let us use the 12.4× matrix mult speedup for polynomial operations!

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::ntru::ga_based::matrix_8x8_to_multivector3d;
use ga_engine::ga::geometric_product_full;

/// ALTERNATIVE: Convert vector directly to multivector (not via sparse matrix)
/// This treats the vector as an 8-component multivector in the column space
fn vector_to_multivector_direct(vec: &[f64; 8]) -> [f64; 8] {
    // Direct mapping: vector components → multivector components
    // This preserves the column structure
    *vec
}

/// Extract vector from result multivector
/// This is the inverse of: vector → sparse matrix → multivector
/// We reconstruct the first column of the sparse matrix representation
fn multivector3d_to_vector(mv: &[f64; 8]) -> [f64; 8] {
    let mut vec = [0.0; 8];

    // Based on the homomorphic mapping, extract vector components
    // These scaling factors should match the inverse of matrix_8x8_to_multivector3d
    vec[0] = mv[0] * 4.0;     // Scalar component
    vec[1] = mv[1] * 10.0;    // e1 vector component
    vec[2] = mv[2] * 10.0;    // e2 vector component
    vec[3] = mv[3] * 10.0;    // e3 vector component
    vec[4] = mv[4] * 4.0;     // e12 bivector component
    vec[5] = mv[5] * 4.0;     // e13 bivector component
    vec[6] = mv[6] * 4.0;     // e23 bivector component
    vec[7] = mv[7] * 10.0;    // e123 pseudoscalar component

    vec
}

/// Debug: Print what a sparse matrix looks like when converted to multivector
fn debug_sparse_matrix_conversion() {
    // Create a simple test vector [1, 2, 3, 4, 5, 6, 7, 8]
    let test_vec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // Convert to sparse matrix
    let mut sparse_matrix = [0.0; 64];
    for i in 0..8 {
        sparse_matrix[i * 8 + 0] = test_vec[i]; // First column only
    }

    println!("\nTest vector: {:?}", test_vec);
    println!("Sparse matrix (8x8, first column = vector):");
    for i in 0..8 {
        print!("[");
        for j in 0..8 {
            print!("{:6.1}", sparse_matrix[i * 8 + j]);
        }
        println!("]");
    }

    // Convert to multivector
    let mv = matrix_8x8_to_multivector3d(&sparse_matrix);
    println!("\nMultivector representation: {:?}", mv);

    // What does this mean?
    // - mv[0] (scalar): (0 + 0 + 0 + 0) / 4 = 0.0
    // - mv[1] (e1): 5 * 0.1 = 0.5
    // - mv[2] (e2): 6 * 0.1 = 0.6
    // - mv[3] (e3): 7 * 0.1 = 0.7
    // - mv[4] (e12): (0 - 0) * 0.25 = 0.0
    // - mv[5] (e13): (0 - 0) * 0.25 = 0.0
    // - mv[6] (e23): (0 - 0) * 0.25 = 0.0
    // - mv[7] (e123): 8 * 0.1 = 0.8
    println!("Expected: scalar=0, e1=0.5, e2=0.6, e3=0.7, e12=0, e13=0, e23=0, e123=0.8");
}

/// Classical matrix-vector multiply
fn matrix_vector_classical(matrix: &[f64], vec: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            result[i] += matrix[i * n + j] * vec[j];
        }
    }
    result
}

/// Matrix × Vector AS Matrix × Sparse Matrix (first column only)
///
/// Key insight: A × v = A × V_sparse where V_sparse has v in column 0, zeros elsewhere
/// Result is also sparse: only column 0 is non-zero
///
/// Question: Can we exploit block structure here?
fn matrix_vector_as_sparse_matrix_mult(
    a_blocks: &[[f64; 8]],  // Pre-converted to multivectors
    vec: &[f64],
    n: usize
) -> Vec<f64> {
    let blocks_per_row = n / 8;

    // Convert vector to "sparse matrix" blocks (only first column non-zero)
    // Actually, we just need the vector segmented into blocks
    let vec_blocks: Vec<[f64; 8]> = (0..blocks_per_row)
        .map(|i| {
            let mut block = [0.0; 8];
            for j in 0..8 {
                if i * 8 + j < vec.len() {
                    block[j] = vec[i * 8 + j];
                }
            }
            block
        })
        .collect();

    // Convert vector segments to multivectors using sparse matrix technique
    let vec_mv: Vec<[f64; 8]> = vec_blocks.iter()
        .map(|block| {
            // Step 1: Convert 8×1 vector block to 8×8 sparse matrix
            // (vector values in first column, zeros elsewhere)
            let mut sparse_matrix = [0.0; 64];
            for i in 0..8 {
                sparse_matrix[i * 8 + 0] = block[i]; // First column only
            }

            // Step 2: Apply homomorphic mapping to multivector
            matrix_8x8_to_multivector3d(&sparse_matrix)
        })
        .collect();

    let mut result = vec![0.0; n];

    // Block matrix-vector multiplication
    // For each row of blocks
    for block_row in 0..blocks_per_row {
        let mut row_sum = [0.0; 8];

        // For each block in this row, multiply with corresponding vector block
        for block_col in 0..blocks_per_row {
            let matrix_block_idx = block_row * blocks_per_row + block_col;
            let matrix_mv = &a_blocks[matrix_block_idx];
            let vec_mv_block = &vec_mv[block_col];

            // GA geometric product
            let mut block_result = [0.0; 8];
            geometric_product_full(matrix_mv, vec_mv_block, &mut block_result);

            // Accumulate
            for i in 0..8 {
                row_sum[i] += block_result[i];
            }
        }

        // Extract vector from result multivector and write to result
        let result_vec = multivector3d_to_vector(&row_sum);
        for i in 0..8 {
            if block_row * 8 + i < n {
                result[block_row * 8 + i] = result_vec[i];
            }
        }
    }

    result
}

/// The CRITICAL question: Does treating vector as sparse matrix preserve structure?
///
/// Matrix multiplication: C = A × B
/// Block decomposition: C[i,j] = Σₖ A[i,k] × B[k,j]
///
/// Matrix-vector as sparse: c = A × v_sparse
/// Block decomposition: c[i,0] = Σₖ A[i,k] × v_sparse[k,0]
///
/// This is EXACTLY matrix-vector multiplication!
/// So we're doing: k² block operations instead of k operations
///
/// But each "block operation" is simpler (vector vs matrix)
/// Does this balance out?

fn bench_128x128_matrix_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_vector_128x128");

    const N: usize = 128;
    const BLOCKS: usize = 16;

    // Setup: Create matrix and vector
    let matrix = vec![1.0; N * N];
    let vector = vec![1.0; N];

    // Setup: Decompose matrix into blocks and convert to multivectors
    let matrix_blocks = {
        let mut blocks = vec![[0.0f64; 64]; BLOCKS * BLOCKS];
        for block_i in 0..BLOCKS {
            for block_j in 0..BLOCKS {
                let block_idx = block_i * BLOCKS + block_j;
                for i in 0..8 {
                    for j in 0..8 {
                        let global_i = block_i * 8 + i;
                        let global_j = block_j * 8 + j;
                        blocks[block_idx][i * 8 + j] = matrix[global_i * N + global_j];
                    }
                }
            }
        }

        // Convert to multivectors
        blocks.iter()
            .map(|block| matrix_8x8_to_multivector3d(block))
            .collect::<Vec<_>>()
    };

    // Benchmark: Classical matrix-vector
    group.bench_function("classical", |bencher| {
        bencher.iter(|| {
            black_box(matrix_vector_classical(
                black_box(&matrix),
                black_box(&vector),
                N
            ))
        })
    });

    // Benchmark: Block GA treating vector as sparse matrix
    group.bench_function("block_ga_sparse", |bencher| {
        bencher.iter(|| {
            black_box(matrix_vector_as_sparse_matrix_mult(
                black_box(&matrix_blocks),
                black_box(&vector),
                N
            ))
        })
    });

    group.finish();
}

/// Analyze the operation count difference
fn bench_operation_count_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("operation_count");

    // Matrix × Matrix: k³ block operations
    // Each block: 8×8 × 8×8 → 8×8

    // Matrix × Vector: k² block operations (only k columns, but each needs k blocks)
    // Each block: 8×8 × 8×1 → 8×1
    //
    // Wait - this is actually the SAME structure!
    // Matrix-vector: For each of k row blocks, do k block operations
    // Total: k × k = k² operations
    //
    // But matrix-matrix: k × k result blocks, each needs k operations
    // Total: k² × k = k³ operations
    //
    // So matrix-vector should be k times fewer operations!

    let mv_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mv_b = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    // Single 8×8 "block" operation
    group.bench_function("single_block_op", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 8];
            geometric_product_full(black_box(&mv_a), black_box(&mv_b), &mut result);
            black_box(result)
        })
    });

    // Matrix × Vector: 16² = 256 block operations
    group.bench_function("256_block_ops_matvec", |bencher| {
        bencher.iter(|| {
            for _ in 0..256 {
                let mut result = [0.0; 8];
                geometric_product_full(black_box(&mv_a), black_box(&mv_b), &mut result);
                black_box(result);
            }
        })
    });

    // Matrix × Matrix: 16³ = 4096 block operations
    group.bench_function("4096_block_ops_matmat", |bencher| {
        bencher.iter(|| {
            for _ in 0..4096 {
                let mut result = [0.0; 8];
                geometric_product_full(black_box(&mv_a), black_box(&mv_b), &mut result);
                black_box(result);
            }
        })
    });

    group.finish();
}

/// Critical test: Does the sparse matrix representation actually work?
fn bench_correctness_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("correctness");

    const N: usize = 16;  // Smaller for verification

    let matrix = vec![2.0; N * N];
    let vector = vec![1.0; N];

    // Classical result
    let classical_result = matrix_vector_classical(&matrix, &vector, N);

    // Block GA result (setup)
    let matrix_blocks = {
        let mut blocks = vec![[0.0f64; 64]; 4];  // 2×2 blocks for N=16
        for block_i in 0..2 {
            for block_j in 0..2 {
                let block_idx = block_i * 2 + block_j;
                for i in 0..8 {
                    for j in 0..8 {
                        let global_i = block_i * 8 + i;
                        let global_j = block_j * 8 + j;
                        blocks[block_idx][i * 8 + j] = matrix[global_i * N + global_j];
                    }
                }
            }
        }
        blocks.iter()
            .map(|block| matrix_8x8_to_multivector3d(block))
            .collect::<Vec<_>>()
    };

    let ga_result = matrix_vector_as_sparse_matrix_mult(&matrix_blocks, &vector, N);

    println!("Classical result (first 8): {:?}", &classical_result[..8]);
    println!("GA result (first 8): {:?}", &ga_result[..8]);

    // Just benchmark to see timings
    group.bench_function("verify_classical", |b| {
        b.iter(|| black_box(matrix_vector_classical(&matrix, &vector, N)))
    });

    group.bench_function("verify_ga", |b| {
        b.iter(|| black_box(matrix_vector_as_sparse_matrix_mult(&matrix_blocks, &vector, N)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_operation_count_analysis,
    bench_128x128_matrix_vector,
    bench_correctness_test,
);

criterion_main!(benches);
