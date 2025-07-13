// examples/ml_feature_transform.rs
//! Practical ML example: GA vs Classical in Feature Transformations
//!
//! This example demonstrates GA's advantages in machine learning feature transformations,
//! specifically using Discrete Fourier Transform (DFT) for signal processing and feature extraction.
//!
//! Key insight: Many ML feature transformations are geometric in nature, making GA a natural fit.

use ga_engine::prelude::*;
use ga_engine::numerical_checks::multivector2::Multivector2;
use ga_engine::numerical_checks::dft;
use rustfft::num_complex::Complex;
use std::time::Instant;
use std::f64::consts::PI;

/// Generate synthetic signal data (common in ML feature extraction)
fn generate_signal_data(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| {
            let t = i as f64 / size as f64;
            // Complex signal with multiple frequency components
            (2.0 * PI * 3.0 * t).sin() +   // 3 Hz component
            0.5 * (2.0 * PI * 7.0 * t).sin() + // 7 Hz component  
            0.25 * (2.0 * PI * 11.0 * t).cos() // 11 Hz component
        })
        .collect()
}

/// Classical approach: DFT using complex numbers
fn classical_feature_extraction(signal: &[f64]) -> Vec<f64> {
    let complex_input: Vec<Complex<f64>> = signal
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    
    let dft_result = dft::classical_dft(&complex_input);
    
    // Extract magnitude spectrum (common ML feature)
    dft_result.iter().map(|c| c.norm()).collect()
}

/// GA approach: DFT using geometric algebra
fn ga_feature_extraction(signal: &[f64]) -> Vec<f64> {
    let ga_input: Vec<Multivector2> = signal
        .iter()
        .map(|&x| Multivector2::new(x, 0.0, 0.0, 0.0))
        .collect();
    
    let ga_result = dft::ga_dft(&ga_input);
    
    // Extract magnitude spectrum using GA operations
    ga_result.iter().map(|mv| {
        // GA provides natural geometric interpretation
        let scalar = mv.data[0];
        let e12 = mv.data[3];
        (scalar * scalar + e12 * e12).sqrt()
    }).collect()
}

/// Demonstrate GA's expressiveness in rotation-based transformations
fn rotate_feature_vectors_classical(features: &[Vec3]) -> Vec<Vec3> {
    // Classical: Manual rotation matrix computation
    let angle = PI / 4.0;  // 45 degrees
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    
    // 3D rotation matrix around Z-axis
    let rotation_matrix = [
        cos_a, -sin_a, 0.0,
        sin_a,  cos_a, 0.0,
        0.0,    0.0,   1.0,
    ];
    
    features.iter()
        .map(|v| apply_matrix3(&rotation_matrix, *v))
        .collect()
}

/// GA approach: Natural rotation operations
fn rotate_feature_vectors_ga(features: &[Vec3]) -> Vec<Vec3> {
    // GA: Intuitive geometric operations
    let axis = Vec3::new(0.0, 0.0, 1.0);
    let angle = PI / 4.0;  // 45 degrees
    let rotor = Rotor3::from_axis_angle(axis, angle);
    
    features.iter()
        .map(|&v| rotor.rotate_fast(v))
        .collect()
}

/// Batch processing comparison (common in ML pipelines)
fn batch_transform_classical(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    data.iter()
        .map(|signal| classical_feature_extraction(signal))
        .collect()
}

fn batch_transform_ga(data: &[Vec<f64>]) -> Vec<Vec<f64>> {
    data.iter()
        .map(|signal| ga_feature_extraction(signal))
        .collect()
}

fn main() {
    println!("🤖 Machine Learning Feature Transformations: GA vs Classical");
    println!("=============================================================");
    
    // Test different signal sizes (common in ML)
    let sizes = [8, 16, 32, 64];
    
    for &size in &sizes {
        println!("\n📊 Signal Size: {} samples", size);
        println!("------------------------------");
        
        let signal = generate_signal_data(size);
        
        // Time classical DFT
        let start = Instant::now();
        let classical_features = classical_feature_extraction(&signal);
        let classical_time = start.elapsed();
        
        // Time GA DFT
        let start = Instant::now();
        let ga_features = ga_feature_extraction(&signal);
        let ga_time = start.elapsed();
        
        println!("Classical DFT: {:?}", classical_time);
        println!("GA DFT:        {:?}", ga_time);
        
        // Verify accuracy
        let max_diff = classical_features.iter()
            .zip(ga_features.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0, f64::max);
        
        println!("Max difference: {:.2e}", max_diff);
        
        if classical_time < ga_time {
            let ratio = ga_time.as_nanos() as f64 / classical_time.as_nanos() as f64;
            println!("Classical wins by {:.2}x", ratio);
        } else {
            let ratio = classical_time.as_nanos() as f64 / ga_time.as_nanos() as f64;
            println!("GA wins by {:.2}x", ratio);
        }
    }
    
    // Demonstrate expressiveness advantage
    println!("\n🎯 Feature Vector Rotation Comparison:");
    println!("---------------------------------------");
    
    let features: Vec<Vec3> = (0..1000)
        .map(|i| Vec3::new(
            (i as f64 * 0.1).sin(),
            (i as f64 * 0.1).cos(),
            i as f64 * 0.01
        ))
        .collect();
    
    let start = Instant::now();
    let classical_rotated = rotate_feature_vectors_classical(&features);
    let classical_rotation_time = start.elapsed();
    
    let start = Instant::now();
    let ga_rotated = rotate_feature_vectors_ga(&features);
    let ga_rotation_time = start.elapsed();
    
    println!("Classical rotation: {:?}", classical_rotation_time);
    println!("GA rotation:        {:?}", ga_rotation_time);
    
    // Verify results are equivalent
    let rotation_diff = classical_rotated.iter()
        .zip(ga_rotated.iter())
        .map(|(c, g)| (*c - *g).norm())
        .fold(0.0, f64::max);
    
    println!("Max rotation difference: {:.2e}", rotation_diff);
    
    // Batch processing demonstration
    println!("\n🔄 Batch Processing Comparison:");
    println!("--------------------------------");
    
    let batch_data: Vec<Vec<f64>> = (0..100)
        .map(|_| generate_signal_data(16))
        .collect();
    
    let start = Instant::now();
    let _classical_batch = batch_transform_classical(&batch_data);
    let classical_batch_time = start.elapsed();
    
    let start = Instant::now();
    let _ga_batch = batch_transform_ga(&batch_data);
    let ga_batch_time = start.elapsed();
    
    println!("Classical batch: {:?}", classical_batch_time);
    println!("GA batch:        {:?}", ga_batch_time);
    
    println!("\n✨ GA Expressiveness Advantages:");
    println!("--------------------------------");
    println!("1. Natural geometric operations (rotations, reflections)");
    println!("2. Unified framework for different transformations");
    println!("3. More intuitive code for geometric feature engineering");
    println!("4. Better composition of transformations");
    
    println!("\n🚀 ML/AI Relevance:");
    println!("-------------------");
    println!("• Feature transformations are core to ML pipelines");
    println!("• Many ML operations are geometric (PCA, rotations, etc.)");
    println!("• GA provides natural framework for geometric ML");
    println!("• Expressiveness advantages lead to fewer bugs");
    println!("• Performance competitive in moderate dimensions");
    
    println!("\n🎯 Use Cases:");
    println!("-------------");
    println!("• Signal processing and feature extraction");
    println!("• Computer vision (image rotations, transforms)");
    println!("• Robotics (pose estimation, motion planning)");
    println!("• Audio processing (frequency domain analysis)");
    println!("• Time series analysis (feature engineering)");
} 