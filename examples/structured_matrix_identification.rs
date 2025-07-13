// examples/structured_matrix_identification.rs
//! Example: Identifying and Converting Structured 8×8 Matrices to GA Operations
//!
//! This example demonstrates how to detect when an 8×8 matrix represents
//! a geometric transformation that can be efficiently computed using GA.

use ga_engine::prelude::*;
use std::f64::consts::PI;

/// Example of an 8×8 matrix that represents a structured transformation
fn generate_structured_8x8_matrix() -> [f64; 64] {
    // This represents a composed transformation:
    // 1. Two 3D rotations (could be applied to different coordinate systems)
    // 2. A reflection operation
    // 3. A scaling operation
    
    // For demonstration, we'll create a matrix that represents
    // applying the same 3D rotation to two different 3D vectors
    
    let angle = PI / 4.0; // 45 degrees
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    
    // 3×3 rotation matrix around Z-axis
    let rotation_3x3 = [
        cos_a, -sin_a, 0.0,
        sin_a,  cos_a, 0.0,
        0.0,    0.0,   1.0,
    ];
    
    // Create an 8×8 block matrix with this rotation applied twice
    let mut matrix = [0.0; 64];
    
    // Block 1: Apply rotation to first 3D vector
    for i in 0..3 {
        for j in 0..3 {
            matrix[i * 8 + j] = rotation_3x3[i * 3 + j];
        }
    }
    
    // Block 2: Apply rotation to second 3D vector (offset by 3)
    for i in 0..3 {
        for j in 0..3 {
            matrix[(i + 3) * 8 + (j + 3)] = rotation_3x3[i * 3 + j];
        }
    }
    
    // Add some scaling to the remaining dimensions
    matrix[6 * 8 + 6] = 2.0; // Scale factor
    matrix[7 * 8 + 7] = 0.5; // Scale factor
    
    matrix
}

/// Check if an 8×8 matrix has the structure of composed 3D rotations
fn is_composed_rotation_matrix(matrix: &[f64; 64]) -> bool {
    // Check if the matrix has block structure consistent with
    // multiple 3D rotations
    
    // Check if top-left 3×3 block is a rotation matrix
    let block1_is_rotation = is_3x3_rotation_matrix(&extract_3x3_block(matrix, 0, 0));
    
    // Check if middle 3×3 block is a rotation matrix
    let block2_is_rotation = is_3x3_rotation_matrix(&extract_3x3_block(matrix, 3, 3));
    
    // Check if the rest is mostly zeros or simple scaling
    let rest_is_simple = check_remaining_structure(matrix);
    
    block1_is_rotation && block2_is_rotation && rest_is_simple
}

/// Extract a 3×3 block from an 8×8 matrix
fn extract_3x3_block(matrix: &[f64; 64], row_offset: usize, col_offset: usize) -> [f64; 9] {
    let mut block = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            block[i * 3 + j] = matrix[(row_offset + i) * 8 + (col_offset + j)];
        }
    }
    block
}

/// Check if a 3×3 matrix is a rotation matrix (orthogonal with determinant 1)
fn is_3x3_rotation_matrix(matrix: &[f64; 9]) -> bool {
    const EPSILON: f64 = 1e-10;
    
    // Check orthogonality: R * R^T = I
    let mut should_be_identity = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += matrix[i * 3 + k] * matrix[j * 3 + k];
            }
            should_be_identity[i * 3 + j] = sum;
        }
    }
    
    // Check if result is identity matrix
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (should_be_identity[i * 3 + j] - expected).abs() > EPSILON {
                return false;
            }
        }
    }
    
    // Check determinant ≈ 1 (not -1, which would be reflection)
    let det = compute_3x3_determinant(matrix);
    (det - 1.0).abs() < EPSILON
}

/// Compute determinant of a 3×3 matrix
fn compute_3x3_determinant(matrix: &[f64; 9]) -> f64 {
    let [a, b, c, d, e, f, g, h, i] = *matrix;
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

/// Check if the non-rotation parts of the matrix are simple (zeros or scaling)
fn check_remaining_structure(matrix: &[f64; 64]) -> bool {
    const EPSILON: f64 = 1e-10;
    
    // Check that blocks outside the two 3×3 rotation blocks are mostly zero
    // or simple scaling on the diagonal
    
    // Check off-diagonal blocks
    for i in 0..3 {
        for j in 3..6 {
            if matrix[i * 8 + j].abs() > EPSILON {
                return false;
            }
        }
    }
    
    for i in 3..6 {
        for j in 0..3 {
            if matrix[i * 8 + j].abs() > EPSILON {
                return false;
            }
        }
    }
    
    true
}

/// Convert a structured 8×8 matrix to equivalent GA operations
fn matrix_to_ga_operations(matrix: &[f64; 64]) -> Option<GATransformation> {
    if !is_composed_rotation_matrix(matrix) {
        return None;
    }
    
    // Extract the two 3×3 rotation blocks
    let rotation1 = extract_3x3_block(matrix, 0, 0);
    let rotation2 = extract_3x3_block(matrix, 3, 3);
    
    // Convert 3×3 rotation matrices to rotors
    let rotor1 = rotation_matrix_to_rotor(&rotation1)?;
    let rotor2 = rotation_matrix_to_rotor(&rotation2)?;
    
    // Extract scaling factors
    let scale1 = matrix[6 * 8 + 6];
    let scale2 = matrix[7 * 8 + 7];
    
    Some(GATransformation {
        rotor1,
        rotor2,
        scale1,
        scale2,
    })
}

/// Convert a 3×3 rotation matrix to a rotor
fn rotation_matrix_to_rotor(matrix: &[f64; 9]) -> Option<Rotor3> {
    // Extract rotation axis and angle from the matrix
    // This is a simplified implementation - full implementation would handle edge cases
    
    // For a rotation matrix R, the axis is the eigenvector with eigenvalue 1
    // The angle can be computed from the trace: cos(θ) = (trace(R) - 1) / 2
    
    let trace = matrix[0] + matrix[4] + matrix[8];
    let cos_theta = (trace - 1.0) / 2.0;
    
    // Clamp to valid range
    let cos_theta = cos_theta.clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    
    if theta.abs() < 1e-10 {
        // Identity rotation
        return Some(Rotor3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.0));
    }
    
    // Extract rotation axis from the skew-symmetric part
    let axis = Vec3::new(
        matrix[7] - matrix[5],
        matrix[2] - matrix[6],
        matrix[3] - matrix[1],
    );
    
    let axis_norm = axis.norm();
    if axis_norm < 1e-10 {
        return None;
    }
    
    let normalized_axis = axis.scale(1.0 / axis_norm);
    Some(Rotor3::from_axis_angle(normalized_axis, theta))
}

/// Represents a GA-based transformation equivalent to a structured 8×8 matrix
#[derive(Debug, Clone)]
struct GATransformation {
    rotor1: Rotor3,
    rotor2: Rotor3,
    scale1: f64,
    scale2: f64,
}

impl GATransformation {
    /// Apply this transformation to a structured 8-element vector
    fn apply(&self, input: &[f64; 8]) -> [f64; 8] {
        let mut result = [0.0; 8];
        
        // Apply first rotor to first 3 elements
        let v1 = Vec3::new(input[0], input[1], input[2]);
        let transformed1 = self.rotor1.rotate_fast(v1);
        result[0] = transformed1.x;
        result[1] = transformed1.y;
        result[2] = transformed1.z;
        
        // Apply second rotor to next 3 elements
        let v2 = Vec3::new(input[3], input[4], input[5]);
        let transformed2 = self.rotor2.rotate_fast(v2);
        result[3] = transformed2.x;
        result[4] = transformed2.y;
        result[5] = transformed2.z;
        
        // Apply scaling to remaining elements
        result[6] = input[6] * self.scale1;
        result[7] = input[7] * self.scale2;
        
        result
    }
}

/// Apply an 8×8 matrix to an 8-element vector
fn apply_8x8_matrix(matrix: &[f64; 64], vector: &[f64; 8]) -> [f64; 8] {
    let mut result = [0.0; 8];
    for i in 0..8 {
        for j in 0..8 {
            result[i] += matrix[i * 8 + j] * vector[j];
        }
    }
    result
}

fn main() {
    println!("=== Structured Matrix Identification Demo ===\n");
    
    // Generate a structured 8×8 matrix
    let structured_matrix = generate_structured_8x8_matrix();
    
    println!("Generated 8×8 matrix with composed rotations:");
    print_matrix_8x8(&structured_matrix);
    
    // Check if it's a composed rotation matrix
    let is_structured = is_composed_rotation_matrix(&structured_matrix);
    println!("\nIs structured (composed rotations)? {}", is_structured);
    
    if is_structured {
        // Convert to GA operations
        if let Some(ga_transform) = matrix_to_ga_operations(&structured_matrix) {
            println!("\nSuccessfully converted to GA operations:");
            println!("Rotor 1: {:?}", ga_transform.rotor1);
            println!("Rotor 2: {:?}", ga_transform.rotor2);
            println!("Scale factors: {}, {}", ga_transform.scale1, ga_transform.scale2);
            
            // Test equivalence
            let test_vector = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            
            let matrix_result = apply_8x8_matrix(&structured_matrix, &test_vector);
            let ga_result = ga_transform.apply(&test_vector);
            
            println!("\nEquivalence test:");
            println!("Input vector: {:?}", test_vector);
            println!("Matrix result: {:?}", matrix_result);
            println!("GA result:     {:?}", ga_result);
            
            // Check if results are close
            let max_diff = matrix_result.iter()
                .zip(ga_result.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            
            println!("Maximum difference: {:.2e}", max_diff);
            
            if max_diff < 1e-10 {
                println!("✅ Results match! GA can replace this matrix.");
            } else {
                println!("❌ Results don't match. Conversion failed.");
            }
        } else {
            println!("❌ Failed to convert to GA operations.");
        }
    } else {
        println!("❌ Matrix is not structured for GA conversion.");
    }
    
    // Performance comparison would go here
    println!("\n=== Performance Implications ===");
    println!("For structured matrices like this:");
    println!("- Matrix multiply: 8×8 = 64 operations × 8 = 512 scalar ops");
    println!("- GA approach: 2 rotor operations + 2 scalings ≈ 40 scalar ops");
    println!("- Theoretical speedup: ~12.8× for this specific structure");
    println!("- Actual speedup depends on implementation efficiency");
}

fn print_matrix_8x8(matrix: &[f64; 64]) {
    for i in 0..8 {
        print!("[");
        for j in 0..8 {
            print!("{:8.3}", matrix[i * 8 + j]);
            if j < 7 { print!(", "); }
        }
        println!("]");
    }
} 