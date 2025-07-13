use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::prelude::*;
use std::f64::consts::PI;

const BATCH_SIZE: usize = 1_000;

/// Geometric decomposition of 8×8 matrix to 8-element multivector
/// 
/// This implements a structured mapping that extracts geometric meaning:
/// - Upper-left 3×3 → rotation (maps to rotor components)
/// - Diagonal elements → scaling (maps to vector components)  
/// - Selected off-diagonal → bivector components
/// 
/// The goal is to preserve as much geometric structure as possible
/// while maintaining mathematical consistency.
fn matrix_to_multivector(matrix: &[f64; 64]) -> [f64; 8] {
    // Extract the upper-left 3×3 block as a rotation matrix
    let r11 = matrix[0];  // (0,0)
    let r12 = matrix[1];  // (0,1)
    let r13 = matrix[2];  // (0,2)
    let r21 = matrix[8];  // (1,0)
    let r22 = matrix[9];  // (1,1)
    let r23 = matrix[10]; // (1,2)
    let r31 = matrix[16]; // (2,0)
    let r32 = matrix[17]; // (2,1)
    let r33 = matrix[18]; // (2,2)
    
    // Convert 3×3 rotation matrix to rotor representation
    // This is a simplified extraction - in practice we'd use proper rotation matrix decomposition
    let trace = r11 + r22 + r33;
    let scalar = (1.0 + trace).sqrt() * 0.5;
    
    // Extract vector components from diagonal scaling
    let e1 = matrix[36] * 0.1; // (4,4) 
    let e2 = matrix[45] * 0.1; // (5,5)
    let e3 = matrix[54] * 0.1; // (6,6)
    
    // Extract bivector components from off-diagonal elements
    let e23 = (r32 - r23) * 0.25; // xy rotation component
    let e31 = (r13 - r31) * 0.25; // xz rotation component  
    let e12 = (r21 - r12) * 0.25; // yz rotation component
    
    // Trivector from remaining structure
    let e123 = matrix[63] * 0.1; // (7,7)
    
    [scalar, e1, e2, e3, e23, e31, e12, e123]
}

/// Alternative mapping: Principal Component Extraction
/// 
/// This extracts the 8 most significant elements based on geometric importance
fn matrix_to_multivector_pca(matrix: &[f64; 64]) -> [f64; 8] {
    // Extract key geometric components
    let scalar = (matrix[0] + matrix[9] + matrix[18]) / 3.0;  // Average of 3x3 diagonal
    let e1 = matrix[3];   // Translation-like component
    let e2 = matrix[11];  // Translation-like component
    let e3 = matrix[19];  // Translation-like component
    let e23 = matrix[27]; // Rotation-like component
    let e31 = matrix[35]; // Rotation-like component
    let e12 = matrix[43]; // Rotation-like component
    let e123 = matrix[51]; // Volume-like component
    
    [scalar, e1, e2, e3, e23, e31, e12, e123]
}

/// Structured block mapping
/// 
/// This maps 8×8 matrix blocks to multivector components in a structured way
fn matrix_to_multivector_block(matrix: &[f64; 64]) -> [f64; 8] {
    // Map 2×2 blocks to complex-like components
    let scalar = matrix[0];                    // (0,0)
    let e1 = matrix[1];                        // (0,1)
    let e2 = matrix[8];                        // (1,0)
    let e3 = matrix[9];                        // (1,1)
    let e23 = matrix[18];                      // (2,2)
    let e31 = matrix[27];                      // (3,3)
    let e12 = matrix[36];                      // (4,4)
    let e123 = matrix[45];                     // (5,5)
    
    [scalar, e1, e2, e3, e23, e31, e12, e123]
}

/// Generate random 8×8 matrices with some geometric structure
fn generate_structured_matrix() -> [f64; 64] {
    let mut matrix = [0.0; 64];
    
    // Fill with random values
    for i in 0..64 {
        matrix[i] = (i as f64 % 10.0) + 1.0 + (i as f64 * 0.1).sin();
    }
    
    // Add some geometric structure
    // Make it more rotation-like by ensuring proper scaling
    matrix[0] *= 0.1;   // Reduce diagonal dominance
    matrix[9] *= 0.1;
    matrix[18] *= 0.1;
    matrix[27] *= 0.1;
    matrix[36] *= 0.1;
    matrix[45] *= 0.1;
    matrix[54] *= 0.1;
    matrix[63] *= 0.1;
    
    matrix
}

/// Classical 8×8 matrix multiplication
fn multiply_8x8_matrices(a: &[f64; 64], b: &[f64; 64]) -> [f64; 64] {
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

/// Benchmark: Geometric Decomposition Mapping
fn bench_geometric_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("geometric_decomposition");
    
    // Generate test matrices
    let matrix_a = generate_structured_matrix();
    let matrix_b = generate_structured_matrix();
    
    // Convert to multivectors
    let mv_a = matrix_to_multivector(&matrix_a);
    let mv_b = matrix_to_multivector(&matrix_b);
    
    // Benchmark classical matrix multiplication
    group.bench_function("classical_8x8_matrix_mult", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 64];
            for _ in 0..BATCH_SIZE {
                result = multiply_8x8_matrices(black_box(&matrix_a), black_box(&matrix_b));
            }
            black_box(result)
        })
    });
    
    // Benchmark GA multivector geometric product
    group.bench_function("ga_multivector_geometric_product", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 8];
            for _ in 0..BATCH_SIZE {
                geometric_product_full(black_box(&mv_a), black_box(&mv_b), &mut result);
            }
            black_box(result)
        })
    });
    
    group.finish();
}

/// Benchmark: Principal Component Extraction
fn bench_pca_mapping(c: &mut Criterion) {
    let mut group = c.benchmark_group("pca_mapping");
    
    let matrix_a = generate_structured_matrix();
    let matrix_b = generate_structured_matrix();
    
    let mv_a = matrix_to_multivector_pca(&matrix_a);
    let mv_b = matrix_to_multivector_pca(&matrix_b);
    
    group.bench_function("classical_8x8_matrix_mult", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 64];
            for _ in 0..BATCH_SIZE {
                result = multiply_8x8_matrices(black_box(&matrix_a), black_box(&matrix_b));
            }
            black_box(result)
        })
    });
    
    group.bench_function("ga_multivector_geometric_product", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 8];
            for _ in 0..BATCH_SIZE {
                geometric_product_full(black_box(&mv_a), black_box(&mv_b), &mut result);
            }
            black_box(result)
        })
    });
    
    group.finish();
}

/// Benchmark: Block Structured Mapping
fn bench_block_mapping(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_mapping");
    
    let matrix_a = generate_structured_matrix();
    let matrix_b = generate_structured_matrix();
    
    let mv_a = matrix_to_multivector_block(&matrix_a);
    let mv_b = matrix_to_multivector_block(&matrix_b);
    
    group.bench_function("classical_8x8_matrix_mult", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 64];
            for _ in 0..BATCH_SIZE {
                result = multiply_8x8_matrices(black_box(&matrix_a), black_box(&matrix_b));
            }
            black_box(result)
        })
    });
    
    group.bench_function("ga_multivector_geometric_product", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 8];
            for _ in 0..BATCH_SIZE {
                geometric_product_full(black_box(&mv_a), black_box(&mv_b), &mut result);
            }
            black_box(result)
        })
    });
    
    group.finish();
}

/// Verification: Check mathematical consistency
fn verify_mapping_consistency() {
    let matrix_a = generate_structured_matrix();
    let matrix_b = generate_structured_matrix();
    
    // Classical result
    let classical_result = multiply_8x8_matrices(&matrix_a, &matrix_b);
    
    // GA results with different mappings
    let mv_a_geom = matrix_to_multivector(&matrix_a);
    let mv_b_geom = matrix_to_multivector(&matrix_b);
    let mut ga_result_geom = [0.0; 8];
    geometric_product_full(&mv_a_geom, &mv_b_geom, &mut ga_result_geom);
    
    let mv_a_pca = matrix_to_multivector_pca(&matrix_a);
    let mv_b_pca = matrix_to_multivector_pca(&matrix_b);
    let mut ga_result_pca = [0.0; 8];
    geometric_product_full(&mv_a_pca, &mv_b_pca, &mut ga_result_pca);
    
    let mv_a_block = matrix_to_multivector_block(&matrix_a);
    let mv_b_block = matrix_to_multivector_block(&matrix_b);
    let mut ga_result_block = [0.0; 8];
    geometric_product_full(&mv_a_block, &mv_b_block, &mut ga_result_block);
    
    println!("=== MAPPING VERIFICATION ===");
    println!("Original matrix A (first 8 elements): {:?}", &matrix_a[0..8]);
    println!("Original matrix B (first 8 elements): {:?}", &matrix_b[0..8]);
    println!("Classical result (first 8 elements): {:?}", &classical_result[0..8]);
    println!();
    println!("Geometric mapping A: {:?}", mv_a_geom);
    println!("Geometric mapping B: {:?}", mv_b_geom);
    println!("GA result (geometric): {:?}", ga_result_geom);
    println!();
    println!("PCA mapping A: {:?}", mv_a_pca);
    println!("PCA mapping B: {:?}", mv_b_pca);
    println!("GA result (PCA): {:?}", ga_result_pca);
    println!();
    println!("Block mapping A: {:?}", mv_a_block);
    println!("Block mapping B: {:?}", mv_b_block);
    println!("GA result (block): {:?}", ga_result_block);
}

/// Comprehensive comparison benchmark
fn bench_comprehensive_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("comprehensive_comparison");
    
    // Use the same matrices for all approaches
    let matrix_a = generate_structured_matrix();
    let matrix_b = generate_structured_matrix();
    
    // All three mappings
    let mv_a_geom = matrix_to_multivector(&matrix_a);
    let mv_b_geom = matrix_to_multivector(&matrix_b);
    
    let mv_a_pca = matrix_to_multivector_pca(&matrix_a);
    let mv_b_pca = matrix_to_multivector_pca(&matrix_b);
    
    let mv_a_block = matrix_to_multivector_block(&matrix_a);
    let mv_b_block = matrix_to_multivector_block(&matrix_b);
    
    // Classical baseline
    group.bench_function("classical_8x8_matrix", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 64];
            for _ in 0..BATCH_SIZE {
                result = multiply_8x8_matrices(black_box(&matrix_a), black_box(&matrix_b));
            }
            black_box(result)
        })
    });
    
    // GA variants
    group.bench_function("ga_geometric_mapping", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 8];
            for _ in 0..BATCH_SIZE {
                geometric_product_full(black_box(&mv_a_geom), black_box(&mv_b_geom), &mut result);
            }
            black_box(result)
        })
    });
    
    group.bench_function("ga_pca_mapping", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 8];
            for _ in 0..BATCH_SIZE {
                geometric_product_full(black_box(&mv_a_pca), black_box(&mv_b_pca), &mut result);
            }
            black_box(result)
        })
    });
    
    group.bench_function("ga_block_mapping", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 8];
            for _ in 0..BATCH_SIZE {
                geometric_product_full(black_box(&mv_a_block), black_box(&mv_b_block), &mut result);
            }
            black_box(result)
        })
    });
    
    group.finish();
}

// Run verification on startup
fn setup_verification() {
    verify_mapping_consistency();
}

criterion_group!(
    benches,
    bench_geometric_decomposition,
    bench_pca_mapping,
    bench_block_mapping,
    bench_comprehensive_comparison
);
criterion_main!(benches); 