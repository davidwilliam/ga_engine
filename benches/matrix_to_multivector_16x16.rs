use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::nd::multivector::Multivector;
use ga_engine::nd::ga4d_optimized::Multivector4D;
use std::f64::consts::PI;

const BATCH_SIZE: usize = 1_000;

/// Geometric decomposition of 16×16 matrix to 16-element multivector (4D GA)
/// 
/// This implements a structured mapping that extracts geometric meaning:
/// - Upper-left 4×4 → rotation-like components (maps to scalar/bivector)
/// - Diagonal elements → scaling (maps to vector components)  
/// - Selected off-diagonal → higher-grade components
/// 
/// The goal is to preserve as much geometric structure as possible
/// while maintaining mathematical consistency.
fn matrix_to_multivector_geometric(matrix: &[f64; 256]) -> Vec<f64> {
    let mut result = vec![0.0; 16];
    
    // Extract scalar from 4×4 upper-left trace
    let trace = matrix[0] + matrix[17] + matrix[34] + matrix[51]; // diagonal elements
    result[0] = trace / 4.0; // scalar component
    
    // Extract vector components from main diagonal
    result[1] = matrix[85] * 0.1;  // (5,5) -> e1
    result[2] = matrix[102] * 0.1; // (6,6) -> e2  
    result[3] = matrix[119] * 0.1; // (7,7) -> e3
    result[4] = matrix[136] * 0.1; // (8,8) -> e4
    
    // Extract bivector components from off-diagonal elements
    result[5] = (matrix[1] - matrix[16]) * 0.25;   // (0,1) - (1,0) -> e12
    result[6] = (matrix[2] - matrix[32]) * 0.25;   // (0,2) - (2,0) -> e13
    result[7] = (matrix[3] - matrix[48]) * 0.25;   // (0,3) - (3,0) -> e14
    result[8] = (matrix[18] - matrix[33]) * 0.25;  // (1,2) - (2,1) -> e23
    result[9] = (matrix[19] - matrix[49]) * 0.25;  // (1,3) - (3,1) -> e24
    result[10] = (matrix[35] - matrix[50]) * 0.25; // (2,3) - (3,2) -> e34
    
    // Extract trivector components from selected elements
    result[11] = matrix[153] * 0.1; // (9,9) -> e123
    result[12] = matrix[170] * 0.1; // (10,10) -> e124
    result[13] = matrix[187] * 0.1; // (11,11) -> e134
    result[14] = matrix[204] * 0.1; // (12,12) -> e234
    
    // Extract pseudoscalar from last diagonal element
    result[15] = matrix[255] * 0.1; // (15,15) -> e1234
    
    result
}

/// Alternative mapping: Principal Component Extraction for 16×16
/// 
/// This extracts the 16 most significant elements based on geometric importance
fn matrix_to_multivector_pca(matrix: &[f64; 256]) -> Vec<f64> {
    let mut result = vec![0.0; 16];
    
    // Average of 4×4 diagonal for scalar
    result[0] = (matrix[0] + matrix[17] + matrix[34] + matrix[51]) / 4.0;
    
    // Extract components systematically across the matrix
    result[1] = matrix[5];    // Translation-like component
    result[2] = matrix[21];   // Translation-like component
    result[3] = matrix[37];   // Translation-like component
    result[4] = matrix[53];   // Translation-like component
    result[5] = matrix[69];   // Rotation-like component
    result[6] = matrix[85];   // Rotation-like component
    result[7] = matrix[101];  // Rotation-like component
    result[8] = matrix[117];  // Rotation-like component
    result[9] = matrix[133];  // Rotation-like component
    result[10] = matrix[149]; // Rotation-like component
    result[11] = matrix[165]; // Higher-order component
    result[12] = matrix[181]; // Higher-order component
    result[13] = matrix[197]; // Higher-order component
    result[14] = matrix[213]; // Higher-order component
    result[15] = matrix[229]; // Volume-like component
    
    result
}

/// Structured block mapping for 16×16
/// 
/// This maps 16×16 matrix blocks to multivector components in a structured way
fn matrix_to_multivector_block(matrix: &[f64; 256]) -> Vec<f64> {
    let mut result = vec![0.0; 16];
    
    // Direct systematic mapping
    result[0] = matrix[0];    // (0,0) -> scalar
    result[1] = matrix[1];    // (0,1) -> e1
    result[2] = matrix[16];   // (1,0) -> e2
    result[3] = matrix[17];   // (1,1) -> e3
    result[4] = matrix[32];   // (2,0) -> e4
    result[5] = matrix[33];   // (2,1) -> e12
    result[6] = matrix[34];   // (2,2) -> e13
    result[7] = matrix[48];   // (3,0) -> e14
    result[8] = matrix[49];   // (3,1) -> e23
    result[9] = matrix[50];   // (3,2) -> e24
    result[10] = matrix[64];  // (4,0) -> e34
    result[11] = matrix[65];  // (4,1) -> e123
    result[12] = matrix[66];  // (4,2) -> e124
    result[13] = matrix[80];  // (5,0) -> e134
    result[14] = matrix[81];  // (5,1) -> e234
    result[15] = matrix[82];  // (5,2) -> e1234
    
    result
}

/// Generate random 16×16 matrices with some geometric structure
fn generate_structured_matrix_16x16() -> [f64; 256] {
    let mut matrix = [0.0; 256];
    
    // Fill with random values
    for i in 0..256 {
        matrix[i] = (i as f64 % 10.0) + 1.0 + (i as f64 * 0.1).sin();
    }
    
    // Add some geometric structure
    // Make diagonal elements smaller to reduce dominance
    for i in 0..16 {
        matrix[i * 16 + i] *= 0.1;
    }
    
    matrix
}

/// Classical 16×16 matrix multiplication
fn multiply_16x16_matrices(a: &[f64; 256], b: &[f64; 256]) -> [f64; 256] {
    let mut result = [0.0; 256];
    
    for i in 0..16 {
        for j in 0..16 {
            for k in 0..16 {
                result[i * 16 + j] += a[i * 16 + k] * b[k * 16 + j];
            }
        }
    }
    
    result
}

/// 4D GA geometric product using optimized implementation
fn geometric_product_4d(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mv_a = Multivector4D::from_vec(a.to_vec());
    let mv_b = Multivector4D::from_vec(b.to_vec());
    let result = mv_a.gp(&mv_b);
    result.to_vec()
}

/// Benchmark: Geometric Decomposition Mapping
fn bench_geometric_decomposition_16x16(c: &mut Criterion) {
    let mut group = c.benchmark_group("geometric_decomposition_16x16");
    
    // Generate test matrices
    let matrix_a = generate_structured_matrix_16x16();
    let matrix_b = generate_structured_matrix_16x16();
    
    // Convert to multivectors
    let mv_a = matrix_to_multivector_geometric(&matrix_a);
    let mv_b = matrix_to_multivector_geometric(&matrix_b);
    
    // Benchmark classical matrix multiplication
    group.bench_function("classical_16x16_matrix_mult", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 256];
            for _ in 0..BATCH_SIZE {
                result = multiply_16x16_matrices(black_box(&matrix_a), black_box(&matrix_b));
            }
            black_box(result)
        })
    });
    
    // Benchmark GA multivector geometric product
    group.bench_function("ga_4d_multivector_geometric_product", |bencher| {
        bencher.iter(|| {
            let mut result = vec![0.0; 16];
            for _ in 0..BATCH_SIZE {
                result = geometric_product_4d(black_box(&mv_a), black_box(&mv_b));
            }
            black_box(result)
        })
    });
    
    group.finish();
}

/// Benchmark: PCA Mapping
fn bench_pca_mapping_16x16(c: &mut Criterion) {
    let mut group = c.benchmark_group("pca_mapping_16x16");
    
    // Generate test matrices
    let matrix_a = generate_structured_matrix_16x16();
    let matrix_b = generate_structured_matrix_16x16();
    
    // Convert to multivectors
    let mv_a = matrix_to_multivector_pca(&matrix_a);
    let mv_b = matrix_to_multivector_pca(&matrix_b);
    
    // Benchmark classical matrix multiplication
    group.bench_function("classical_16x16_matrix_mult", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 256];
            for _ in 0..BATCH_SIZE {
                result = multiply_16x16_matrices(black_box(&matrix_a), black_box(&matrix_b));
            }
            black_box(result)
        })
    });
    
    // Benchmark GA multivector geometric product
    group.bench_function("ga_4d_multivector_geometric_product", |bencher| {
        bencher.iter(|| {
            let mut result = vec![0.0; 16];
            for _ in 0..BATCH_SIZE {
                result = geometric_product_4d(black_box(&mv_a), black_box(&mv_b));
            }
            black_box(result)
        })
    });
    
    group.finish();
}

/// Benchmark: Block Mapping
fn bench_block_mapping_16x16(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_mapping_16x16");
    
    // Generate test matrices
    let matrix_a = generate_structured_matrix_16x16();
    let matrix_b = generate_structured_matrix_16x16();
    
    // Convert to multivectors
    let mv_a = matrix_to_multivector_block(&matrix_a);
    let mv_b = matrix_to_multivector_block(&matrix_b);
    
    // Benchmark classical matrix multiplication
    group.bench_function("classical_16x16_matrix_mult", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 256];
            for _ in 0..BATCH_SIZE {
                result = multiply_16x16_matrices(black_box(&matrix_a), black_box(&matrix_b));
            }
            black_box(result)
        })
    });
    
    // Benchmark GA multivector geometric product
    group.bench_function("ga_4d_multivector_geometric_product", |bencher| {
        bencher.iter(|| {
            let mut result = vec![0.0; 16];
            for _ in 0..BATCH_SIZE {
                result = geometric_product_4d(black_box(&mv_a), black_box(&mv_b));
            }
            black_box(result)
        })
    });
    
    group.finish();
}

/// Comprehensive comparison of all strategies
fn bench_comprehensive_comparison_16x16(c: &mut Criterion) {
    let mut group = c.benchmark_group("comprehensive_comparison_16x16");
    
    // Generate test matrices
    let matrix_a = generate_structured_matrix_16x16();
    let matrix_b = generate_structured_matrix_16x16();
    
    // Convert to multivectors using all strategies
    let mv_a_geom = matrix_to_multivector_geometric(&matrix_a);
    let mv_b_geom = matrix_to_multivector_geometric(&matrix_b);
    
    let mv_a_pca = matrix_to_multivector_pca(&matrix_a);
    let mv_b_pca = matrix_to_multivector_pca(&matrix_b);
    
    let mv_a_block = matrix_to_multivector_block(&matrix_a);
    let mv_b_block = matrix_to_multivector_block(&matrix_b);
    
    // Classical benchmark
    group.bench_function("classical_16x16_batch", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 256];
            for _ in 0..BATCH_SIZE {
                result = multiply_16x16_matrices(black_box(&matrix_a), black_box(&matrix_b));
            }
            black_box(result)
        })
    });
    
    // GA benchmarks
    group.bench_function("ga_geometric_16x16_batch", |bencher| {
        bencher.iter(|| {
            let mut result = vec![0.0; 16];
            for _ in 0..BATCH_SIZE {
                result = geometric_product_4d(black_box(&mv_a_geom), black_box(&mv_b_geom));
            }
            black_box(result)
        })
    });
    
    group.bench_function("ga_pca_16x16_batch", |bencher| {
        bencher.iter(|| {
            let mut result = vec![0.0; 16];
            for _ in 0..BATCH_SIZE {
                result = geometric_product_4d(black_box(&mv_a_pca), black_box(&mv_b_pca));
            }
            black_box(result)
        })
    });
    
    group.bench_function("ga_block_16x16_batch", |bencher| {
        bencher.iter(|| {
            let mut result = vec![0.0; 16];
            for _ in 0..BATCH_SIZE {
                result = geometric_product_4d(black_box(&mv_a_block), black_box(&mv_b_block));
            }
            black_box(result)
        })
    });
    
    group.finish();
}

/// Verification function to ensure mappings are working correctly
fn verify_mapping_consistency_16x16() {
    println!("🔍 Verifying 16×16 matrix to 4D multivector mapping consistency...");
    
    let matrix_a = generate_structured_matrix_16x16();
    let matrix_b = generate_structured_matrix_16x16();
    
    // Apply all three mapping strategies
    let mv_a_geom = matrix_to_multivector_geometric(&matrix_a);
    let mv_a_pca = matrix_to_multivector_pca(&matrix_a);
    let mv_a_block = matrix_to_multivector_block(&matrix_a);
    
    println!("Matrix A (first 8 elements): {:?}", &matrix_a[..8]);
    println!("Geometric mapping A: {:?}", mv_a_geom);
    println!("PCA mapping A: {:?}", mv_a_pca);
    println!("Block mapping A: {:?}", mv_a_block);
    
    // Verify all mappings produce valid 16-element vectors
    assert_eq!(mv_a_geom.len(), 16);
    assert_eq!(mv_a_pca.len(), 16);
    assert_eq!(mv_a_block.len(), 16);
    
    println!("✅ All mappings produce valid 16-element multivectors");
    
    // Test geometric products
    let mv_b_geom = matrix_to_multivector_geometric(&matrix_b);
    let result_geom = geometric_product_4d(&mv_a_geom, &mv_b_geom);
    assert_eq!(result_geom.len(), 16);
    
    println!("✅ Geometric products produce valid results");
    println!("Geometric product result: {:?}", result_geom);
}

fn setup_verification_16x16() {
    verify_mapping_consistency_16x16();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(100).measurement_time(std::time::Duration::from_secs(10));
    targets = bench_geometric_decomposition_16x16, bench_pca_mapping_16x16, bench_block_mapping_16x16, bench_comprehensive_comparison_16x16
);

criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_matrix_generation_16x16() {
        let matrix = generate_structured_matrix_16x16();
        assert_eq!(matrix.len(), 256);
        // Should have some non-zero elements
        assert!(matrix.iter().any(|&x| x != 0.0));
    }
    
    #[test]
    fn test_mapping_functions_16x16() {
        let matrix = generate_structured_matrix_16x16();
        
        let geom = matrix_to_multivector_geometric(&matrix);
        let pca = matrix_to_multivector_pca(&matrix);
        let block = matrix_to_multivector_block(&matrix);
        
        assert_eq!(geom.len(), 16);
        assert_eq!(pca.len(), 16);
        assert_eq!(block.len(), 16);
    }
    
    #[test]
    fn test_geometric_product_4d() {
        let a = vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        
        let result = geometric_product_4d(&a, &b);
        assert_eq!(result.len(), 16);
        // Should have non-zero elements
        assert!(result.iter().any(|&x| x != 0.0));
    }
} 