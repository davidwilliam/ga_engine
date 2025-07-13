use ga_engine::nd::multivector::Multivector;
use ga_engine::nd::ga4d_optimized::Multivector4D;
use std::time::Instant;

/// Geometric decomposition mapping (simplified from benchmark)
fn matrix_to_multivector_geometric(matrix: &[f64; 256]) -> Vec<f64> {
    let mut result = vec![0.0; 16];
    
    // Extract scalar from 4×4 upper-left trace
    let trace = matrix[0] + matrix[17] + matrix[34] + matrix[51];
    result[0] = trace / 4.0;
    
    // Extract vector components from main diagonal
    result[1] = matrix[85] * 0.1;   // (5,5) -> e1
    result[2] = matrix[102] * 0.1;  // (6,6) -> e2  
    result[3] = matrix[119] * 0.1;  // (7,7) -> e3
    result[4] = matrix[136] * 0.1;  // (8,8) -> e4
    
    // Extract bivector components from off-diagonal elements
    result[5] = (matrix[1] - matrix[16]) * 0.25;   // (0,1) - (1,0) -> e12
    result[6] = (matrix[2] - matrix[32]) * 0.25;   // (0,2) - (2,0) -> e13
    result[7] = (matrix[3] - matrix[48]) * 0.25;   // (0,3) - (3,0) -> e14
    result[8] = (matrix[18] - matrix[33]) * 0.25;  // (1,2) - (2,1) -> e23
    result[9] = (matrix[19] - matrix[49]) * 0.25;  // (1,3) - (3,1) -> e24
    result[10] = (matrix[35] - matrix[50]) * 0.25; // (2,3) - (3,2) -> e34
    
    // Extract trivector components
    result[11] = matrix[153] * 0.1; // (9,9) -> e123
    result[12] = matrix[170] * 0.1; // (10,10) -> e124
    result[13] = matrix[187] * 0.1; // (11,11) -> e134
    result[14] = matrix[204] * 0.1; // (12,12) -> e234
    
    // Extract pseudoscalar
    result[15] = matrix[255] * 0.1; // (15,15) -> e1234
    
    result
}

/// Generate a simple test matrix
fn generate_test_matrix() -> [f64; 256] {
    let mut matrix = [0.0; 256];
    
    // Fill with simple pattern
    for i in 0..256 {
        matrix[i] = (i as f64 % 10.0) + 1.0;
    }
    
    // Make diagonal elements smaller
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

fn main() {
    println!("🧮 16×16 Matrix to 4D Multivector Mapping Demo");
    println!("==============================================");
    
    // Generate test matrices
    let matrix_a = generate_test_matrix();
    let matrix_b = generate_test_matrix();
    
    println!("\n📊 Matrix A (first 8 elements): {:?}", &matrix_a[..8]);
    println!("📊 Matrix B (first 8 elements): {:?}", &matrix_b[..8]);
    
    // Convert to multivectors
    let mv_a = matrix_to_multivector_geometric(&matrix_a);
    let mv_b = matrix_to_multivector_geometric(&matrix_b);
    
    println!("\n🔄 Converted multivectors:");
    println!("   Multivector A: {:?}", mv_a);
    println!("   Multivector B: {:?}", mv_b);
    
    // Test single operations
    println!("\n⏱️  Single Operation Performance:");
    
    // Classical matrix multiplication
    let start = Instant::now();
    let classical_result = multiply_16x16_matrices(&matrix_a, &matrix_b);
    let classical_time = start.elapsed();
    
    println!("   Classical 16×16 multiplication: {:?}", classical_time);
    println!("   Result (first 8 elements): {:?}", &classical_result[..8]);
    
    // GA geometric product
    let start = Instant::now();
    let ga_result = geometric_product_4d(&mv_a, &mv_b);
    let ga_time = start.elapsed();
    
    println!("   GA 4D geometric product: {:?}", ga_time);
    println!("   Result: {:?}", ga_result);
    
    // Performance comparison
    let speedup = classical_time.as_nanos() as f64 / ga_time.as_nanos() as f64;
    println!("\n📈 Performance Analysis:");
    println!("   Classical time: {:?}", classical_time);
    println!("   GA time: {:?}", ga_time);
    
    if speedup > 1.0 {
        println!("   ✅ GA is {:.2}× faster than classical", speedup);
    } else {
        println!("   ⚠️  GA is {:.2}× slower than classical", 1.0 / speedup);
    }
    
    // Theoretical analysis
    println!("\n🔬 Theoretical Analysis:");
    println!("   Classical operations: 16³ = 4,096 multiply-add operations");
    println!("   GA operations: 16×16 = 256 multiply-add operations");
    println!("   Theoretical speedup: 4,096 / 256 = 16.0×");
    println!("   Measured speedup: {:.2}×", speedup);
    
    if speedup > 1.0 {
        println!("   Implementation efficiency: {:.1}%", (speedup / 16.0) * 100.0);
    } else {
        println!("   ⚠️  Implementation is slower than classical - needs investigation");
    }
    
    // Memory analysis
    println!("\n💾 Memory Analysis:");
    println!("   Matrix storage: 256 × f64 = 2,048 bytes");
    println!("   Multivector storage: 16 × f64 = 128 bytes");
    println!("   Memory reduction: 16× smaller");
    
    // Test multiple operations
    println!("\n🔄 Batch Performance Test (1000 operations):");
    
    let iterations = 1000;
    
    // Classical batch
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = multiply_16x16_matrices(&matrix_a, &matrix_b);
    }
    let classical_batch_time = start.elapsed();
    
    // GA batch
    let start = Instant::now();
    for _ in 0..iterations {
        let _result = geometric_product_4d(&mv_a, &mv_b);
    }
    let ga_batch_time = start.elapsed();
    
    let batch_speedup = classical_batch_time.as_nanos() as f64 / ga_batch_time.as_nanos() as f64;
    
    println!("   Classical batch ({} ops): {:?}", iterations, classical_batch_time);
    println!("   GA batch ({} ops): {:?}", iterations, ga_batch_time);
    
    if batch_speedup > 1.0 {
        println!("   ✅ GA batch is {:.2}× faster", batch_speedup);
    } else {
        println!("   ⚠️  GA batch is {:.2}× slower", 1.0 / batch_speedup);
    }
    
    println!("\n🏁 Demo Complete!");
} 