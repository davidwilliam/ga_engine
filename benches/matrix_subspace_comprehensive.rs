// benches/matrix_subspace_comprehensive.rs
//! Comprehensive benchmark demonstrating GA performance across matrix types
//!
//! This benchmark provides concrete numbers for presentation:
//! 1. Arbitrary 8×8 matrices (current benchmark)
//! 2. Structured geometric matrices (composed rotations)
//! 3. Optimal geometric matrices (pure rotations)
//! 4. Real-world scenarios (graphics, robotics, physics)

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::prelude::*;
use std::f64::consts::PI;

const BATCH_SIZE: usize = 1_000;

// ============================================================================
// Level 1: Arbitrary Matrices (Current Benchmark Performance)
// ============================================================================

/// Benchmark arbitrary 8×8 matrix multiplication vs GA
fn bench_arbitrary_matrices(c: &mut Criterion) {
    let mut group = c.benchmark_group("1_arbitrary_matrices");
    
    // Arbitrary 8×8 matrices
    let a: Vec<f64> = (0..64).map(|i| (i % 10) as f64 + 1.0).collect();
    let b = a.clone();
    
    // GA representation
    let ga_a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let ga_b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    group.bench_function("classical_8x8_matrix", |bencher| {
        bencher.iter(|| {
            let mut result = Vec::with_capacity(64);
            for _ in 0..BATCH_SIZE {
                result = multiply_matrices(black_box(&a), black_box(&b), 8);
            }
            black_box(result)
        })
    });

    group.bench_function("ga_geometric_product", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 8];
            for _ in 0..BATCH_SIZE {
                geometric_product_full(black_box(&ga_a), black_box(&ga_b), &mut result);
            }
            black_box(result)
        })
    });

    group.finish();
}

// ============================================================================
// Level 2: Structured Matrices (Composed Geometric Transformations)
// ============================================================================

/// Generate a structured 8×8 matrix representing composed 3D transformations
fn generate_structured_matrix(angle1: f64, angle2: f64, scale1: f64, scale2: f64) -> [f64; 64] {
    let cos1 = angle1.cos();
    let sin1 = angle1.sin();
    let cos2 = angle2.cos();
    let sin2 = angle2.sin();
    
    let mut matrix = [0.0; 64];
    
    // First 3×3 rotation block (Z-axis rotation)
    matrix[0 * 8 + 0] = cos1;  matrix[0 * 8 + 1] = -sin1; matrix[0 * 8 + 2] = 0.0;
    matrix[1 * 8 + 0] = sin1;  matrix[1 * 8 + 1] = cos1;  matrix[1 * 8 + 2] = 0.0;
    matrix[2 * 8 + 0] = 0.0;   matrix[2 * 8 + 1] = 0.0;   matrix[2 * 8 + 2] = 1.0;
    
    // Second 3×3 rotation block (Y-axis rotation)
    matrix[3 * 8 + 3] = cos2;  matrix[3 * 8 + 4] = 0.0;   matrix[3 * 8 + 5] = sin2;
    matrix[4 * 8 + 3] = 0.0;   matrix[4 * 8 + 4] = 1.0;   matrix[4 * 8 + 5] = 0.0;
    matrix[5 * 8 + 3] = -sin2; matrix[5 * 8 + 4] = 0.0;   matrix[5 * 8 + 5] = cos2;
    
    // Scaling components
    matrix[6 * 8 + 6] = scale1;
    matrix[7 * 8 + 7] = scale2;
    
    matrix
}

/// Convert structured matrix to GA operations
fn structured_matrix_to_ga(angle1: f64, angle2: f64, scale1: f64, scale2: f64) -> StructuredGATransform {
    let rotor1 = Rotor3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), angle1);
    let rotor2 = Rotor3::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), angle2);
    
    StructuredGATransform {
        rotor1,
        rotor2,
        scale1,
        scale2,
    }
}

#[derive(Clone)]
struct StructuredGATransform {
    rotor1: Rotor3,
    rotor2: Rotor3,
    scale1: f64,
    scale2: f64,
}

impl StructuredGATransform {
    fn apply(&self, input: &[f64; 8]) -> [f64; 8] {
        let mut result = [0.0; 8];
        
        // Apply first rotor
        let v1 = Vec3::new(input[0], input[1], input[2]);
        let t1 = self.rotor1.rotate_fast(v1);
        result[0] = t1.x; result[1] = t1.y; result[2] = t1.z;
        
        // Apply second rotor
        let v2 = Vec3::new(input[3], input[4], input[5]);
        let t2 = self.rotor2.rotate_fast(v2);
        result[3] = t2.x; result[4] = t2.y; result[5] = t2.z;
        
        // Apply scaling
        result[6] = input[6] * self.scale1;
        result[7] = input[7] * self.scale2;
        
        result
    }
}

fn apply_8x8_matrix(matrix: &[f64; 64], vector: &[f64; 8]) -> [f64; 8] {
    let mut result = [0.0; 8];
    for i in 0..8 {
        for j in 0..8 {
            result[i] += matrix[i * 8 + j] * vector[j];
        }
    }
    result
}

/// Benchmark structured matrices vs GA operations
fn bench_structured_matrices(c: &mut Criterion) {
    let mut group = c.benchmark_group("2_structured_matrices");
    
    // Test parameters
    let angle1 = PI / 4.0;  // 45 degrees
    let angle2 = PI / 6.0;  // 30 degrees
    let scale1 = 2.0;
    let scale2 = 0.5;
    
    let structured_matrix = generate_structured_matrix(angle1, angle2, scale1, scale2);
    let ga_transform = structured_matrix_to_ga(angle1, angle2, scale1, scale2);
    let test_vector = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    group.bench_function("structured_matrix_multiply", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 8];
            for _ in 0..BATCH_SIZE {
                result = apply_8x8_matrix(black_box(&structured_matrix), black_box(&test_vector));
            }
            black_box(result)
        })
    });

    group.bench_function("structured_ga_operations", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 8];
            for _ in 0..BATCH_SIZE {
                result = ga_transform.apply(black_box(&test_vector));
            }
            black_box(result)
        })
    });

    group.finish();
}

// ============================================================================
// Level 3: Optimal Matrices (Pure Geometric Operations)
// ============================================================================

/// Pure rotation matrix that can be optimally represented in GA
fn generate_pure_rotation_matrix(axis: Vec3, angle: f64) -> [f64; 64] {
    let mut matrix = [0.0; 64];
    
    // Create rotation matrix for the given axis and angle
    let rotor = Rotor3::from_axis_angle(axis, angle);
    
    // Convert to equivalent 8×8 representation by applying to basis vectors
    let basis_vectors = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ];
    
    for i in 0..8 {
        for j in 0..8 {
            if j < 3 {
                // Apply rotation to first 3 components
                let v = Vec3::new(basis_vectors[j][0], basis_vectors[j][1], basis_vectors[j][2]);
                let rotated = rotor.rotate_fast(v);
                matrix[i * 8 + j] = match i {
                    0 => rotated.x,
                    1 => rotated.y,
                    2 => rotated.z,
                    _ => 0.0,
                };
            } else if i == j {
                // Identity for remaining components
                matrix[i * 8 + j] = 1.0;
            }
        }
    }
    
    matrix
}

/// Benchmark optimal matrices vs GA operations
fn bench_optimal_matrices(c: &mut Criterion) {
    let mut group = c.benchmark_group("3_optimal_matrices");
    
    let axis = Vec3::new(0.0, 0.0, 1.0);
    let angle = PI / 3.0;  // 60 degrees
    
    let optimal_matrix = generate_pure_rotation_matrix(axis, angle);
    let rotor = Rotor3::from_axis_angle(axis, angle);
    let test_vector = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    group.bench_function("optimal_matrix_multiply", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 8];
            for _ in 0..BATCH_SIZE {
                result = apply_8x8_matrix(black_box(&optimal_matrix), black_box(&test_vector));
            }
            black_box(result)
        })
    });

    group.bench_function("optimal_ga_rotor", |bencher| {
        bencher.iter(|| {
            let mut result = [0.0; 8];
            for _ in 0..BATCH_SIZE {
                let v = Vec3::new(test_vector[0], test_vector[1], test_vector[2]);
                let rotated = rotor.rotate_fast(black_box(v));
                result[0] = rotated.x;
                result[1] = rotated.y;
                result[2] = rotated.z;
                result[3] = test_vector[3];
                result[4] = test_vector[4];
                result[5] = test_vector[5];
                result[6] = test_vector[6];
                result[7] = test_vector[7];
            }
            black_box(result)
        })
    });

    group.finish();
}

// ============================================================================
// Real-World Scenarios
// ============================================================================

/// Computer Graphics: Batch transform pipeline
fn bench_graphics_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("4_graphics_pipeline");
    
    // Simulate transforming 100 3D points with rotation + scaling
    let points: Vec<Vec3> = (0..100)
        .map(|i| Vec3::new(i as f64, (i * 2) as f64, (i * 3) as f64))
        .collect();
    
    let rotation_matrix = [
        0.707, -0.707, 0.0, 0.0,
        0.707,  0.707, 0.0, 0.0,
        0.0,    0.0,   1.0, 0.0,
        0.0,    0.0,   0.0, 1.0,
    ];
    
    let rotor = Rotor3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), PI / 4.0);

    group.bench_function("graphics_matrix_pipeline", |bencher| {
        bencher.iter(|| {
            let mut transformed = Vec::with_capacity(100);
            for point in &points {
                // Apply 4×4 matrix transformation
                let x = rotation_matrix[0] * point.x + rotation_matrix[1] * point.y + rotation_matrix[2] * point.z;
                let y = rotation_matrix[4] * point.x + rotation_matrix[5] * point.y + rotation_matrix[6] * point.z;
                let z = rotation_matrix[8] * point.x + rotation_matrix[9] * point.y + rotation_matrix[10] * point.z;
                transformed.push(Vec3::new(x, y, z));
            }
            black_box(transformed)
        })
    });

    group.bench_function("graphics_ga_pipeline", |bencher| {
        bencher.iter(|| {
            let mut transformed = Vec::with_capacity(100);
            for point in &points {
                let rotated = rotor.rotate_fast(black_box(*point));
                transformed.push(rotated);
            }
            black_box(transformed)
        })
    });

    group.finish();
}

/// Robotics: Forward kinematics chain
fn bench_robotics_kinematics(c: &mut Criterion) {
    let mut group = c.benchmark_group("5_robotics_kinematics");
    
    // 6-DOF robot arm: 6 joint angles
    let joint_angles = [PI/6.0, PI/4.0, PI/3.0, PI/2.0, PI/5.0, PI/7.0];
    
    // Precompute transformation matrices for each joint
    let joint_matrices: Vec<[f64; 16]> = joint_angles.iter().map(|&angle| {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        [
            cos_a, -sin_a, 0.0, 0.0,
            sin_a,  cos_a, 0.0, 0.0,
            0.0,    0.0,   1.0, 1.0,  // 1 unit translation
            0.0,    0.0,   0.0, 1.0,
        ]
    }).collect();
    
    // Precompute rotors for each joint
    let joint_rotors: Vec<Rotor3> = joint_angles.iter().map(|&angle| {
        Rotor3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), angle)
    }).collect();

    group.bench_function("robotics_matrix_forward_kinematics", |bencher| {
        bencher.iter(|| {
            // Chain multiply 6 transformation matrices
            let mut result = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
            for matrix in &joint_matrices {
                result = multiply_4x4_matrices(&result, matrix);
            }
            black_box(result)
        })
    });

    group.bench_function("robotics_ga_forward_kinematics", |bencher| {
        bencher.iter(|| {
            // Chain multiply 6 rotors
            let mut result = Rotor3::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), 0.0);
            for rotor in &joint_rotors {
                result = result.mul(rotor);
            }
            black_box(result)
        })
    });

    group.finish();
}

/// Helper function for 4×4 matrix multiplication
fn multiply_4x4_matrices(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut result = [0.0; 16];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
            }
        }
    }
    result
}

criterion_group!(
    comprehensive_benches,
    bench_arbitrary_matrices,
    bench_structured_matrices,
    bench_optimal_matrices,
    bench_graphics_pipeline,
    bench_robotics_kinematics
);
criterion_main!(comprehensive_benches); 