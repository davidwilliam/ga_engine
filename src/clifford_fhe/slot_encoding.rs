//! SIMD Slot Encoding for CKKS
//!
//! This module implements proper CKKS SIMD slot encoding using FFT-like transforms.
//! Unlike simple coefficient packing, this uses the Chinese Remainder Theorem to
//! decompose the polynomial ring into "slots" that can be manipulated independently.
//!
//! # Mathematical Foundation
//!
//! The polynomial ring R = Z[x]/(Φ_M(x)) can be viewed as having N/2 complex slots:
//! ```text
//! R ≅ C^(N/2)  via CRT
//! ```
//!
//! Where M = 2N and Φ_M is the M-th cyclotomic polynomial.
//!
//! # Encoding Process
//!
//! To encode vector [z₀, z₁, ..., z_{N/2-1}] into polynomial p(x):
//! ```text
//! p(x) = Σᵢ aᵢ·xⁱ
//! where aᵢ = Σⱼ zⱼ · ωᴹ^(i·(2j+1))
//! and ωᴹ = e^(2πi/M)
//! ```

use rustfft::num_complex::Complex;
use std::f64::consts::PI;

/// Encode multivector into SIMD slots
///
/// Takes an 8-component multivector and encodes it into the first 8 slots
/// of a CKKS ciphertext. Remaining slots are set to zero.
///
/// # Arguments
/// * `mv` - Multivector components [scalar, e1, e2, e3, e12, e13, e23, e123]
/// * `scale` - Scaling factor for fixed-point encoding
/// * `n` - Ring dimension (must be power of 2, typically 4096-32768)
///
/// # Returns
/// Vector of scaled polynomial coefficients ready for CKKS encryption
pub fn encode_multivector_slots(mv: &[f64; 8], scale: f64, n: usize) -> Vec<i64> {
    assert!(n.is_power_of_two(), "Ring dimension must be power of 2");
    assert!(n >= 16, "Ring dimension too small for 8 slots");

    // Create complex slot vector (first 8 slots = multivector, rest = 0)
    let num_slots = n / 2;
    let mut slots = vec![Complex::new(0.0, 0.0); num_slots];

    for i in 0..8 {
        slots[i] = Complex::new(mv[i], 0.0); // Real values only
    }

    // Convert slots to polynomial coefficients via inverse FFT-like transform
    slots_to_coefficients(&slots, scale, n)
}

/// Decode SIMD slots back to multivector
///
/// Extracts the first 8 slots from a polynomial and returns them as a multivector.
///
/// # Arguments
/// * `coeffs` - Polynomial coefficients
/// * `scale` - Scaling factor used during encoding
/// * `n` - Ring dimension
///
/// # Returns
/// Multivector with 8 components
pub fn decode_multivector_slots(coeffs: &[i64], scale: f64, n: usize) -> [f64; 8] {
    assert_eq!(coeffs.len(), n, "Coefficient vector must have length n");

    // Convert coefficients to slots via forward FFT-like transform
    let slots = coefficients_to_slots(coeffs, scale, n);

    // Extract first 8 slots (real parts only)
    let mut mv = [0.0f64; 8];
    for i in 0..8 {
        mv[i] = slots[i].re; // Take real part
    }

    mv
}

/// Convert SIMD slots to polynomial coefficients
///
/// Implements the encoding map from C^(N/2) to R using roots of unity.
///
/// # Mathematical Formula
/// ```text
/// aᵢ = Σⱼ slot[j] · ωᴹ^(i·(2j+1)) · scale
/// where ωᴹ = e^(2πi/M), M = 2N
/// ```
pub fn slots_to_coefficients(slots: &[Complex<f64>], scale: f64, n: usize) -> Vec<i64> {
    let num_slots = n / 2;
    assert_eq!(
        slots.len(),
        num_slots,
        "Must have N/2 slots for ring dimension N"
    );

    let m = 2 * n; // Cyclotomic index
    let mut coeffs = vec![0i64; n];

    // Compute root of unity: ωᴹ = e^(2πi/M)
    let omega_m = |k: usize| -> Complex<f64> {
        let angle = 2.0 * PI * (k as f64) / (m as f64);
        Complex::new(angle.cos(), angle.sin())
    };

    // For each coefficient position i
    for i in 0..n {
        let mut sum = Complex::new(0.0, 0.0);

        // Sum over all slots j
        for j in 0..num_slots {
            // Compute exponent: i * (2j + 1)
            let exponent = (i * (2 * j + 1)) % m;

            // Add: slot[j] * ω^exponent
            sum += slots[j] * omega_m(exponent);
        }

        // The sum needs to be multiplied by 2 for proper normalization
        // This comes from the CKKS encoding being C^(N/2) -> R (not C^N -> R)
        sum *= 2.0;

        // Scale and round to integer
        coeffs[i] = (sum.re * scale).round() as i64;

        // Note: We ignore imaginary part since we're encoding real multivectors
        // In full CKKS, would need to handle complex values properly
    }

    coeffs
}

/// Convert polynomial coefficients to SIMD slots
///
/// Implements the decoding map from R to C^(N/2) using roots of unity.
///
/// # Mathematical Formula
/// ```text
/// slot[j] = (1/N) · Σᵢ a[i] · ωᴹ^(-i·(2j+1)) / scale
/// where ωᴹ = e^(2πi/M), M = 2N
/// ```
pub fn coefficients_to_slots(coeffs: &[i64], scale: f64, n: usize) -> Vec<Complex<f64>> {
    assert_eq!(coeffs.len(), n, "Must have N coefficients");

    let num_slots = n / 2;
    let m = 2 * n;
    let mut slots = vec![Complex::new(0.0, 0.0); num_slots];

    // Compute root of unity: ωᴹ = e^(2πi/M)
    let omega_m = |k: isize| -> Complex<f64> {
        let angle = 2.0 * PI * (k as f64) / (m as f64);
        Complex::new(angle.cos(), angle.sin())
    };

    // For each slot position j
    for j in 0..num_slots {
        let mut sum = Complex::new(0.0, 0.0);

        // Sum over all coefficients i
        for i in 0..n {
            // Compute exponent: -i * (2j + 1)
            let exponent = -((i * (2 * j + 1)) as isize);

            // Add: a[i] * ω^exponent
            sum += (coeffs[i] as f64) * omega_m(exponent);
        }

        // Normalize and unscale (divide by 2N since we multiplied by 2 in encoding)
        slots[j] = sum / (2.0 * n as f64) / scale;
    }

    slots
}

/// Create plaintext with value in specific slot
///
/// Useful for masking operations: creates a polynomial that has value 1.0
/// in one slot and 0.0 in all others.
pub fn create_slot_mask(slot_index: usize, scale: f64, n: usize) -> Vec<i64> {
    let num_slots = n / 2;
    assert!(
        slot_index < num_slots,
        "Slot index {} out of range [0, {})",
        slot_index,
        num_slots
    );

    let mut slots = vec![Complex::new(0.0, 0.0); num_slots];
    slots[slot_index] = Complex::new(1.0, 0.0);

    slots_to_coefficients(&slots, scale, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_roundtrip() {
        let n = 64; // Small for testing
        let scale = 2f64.powi(20);

        // Test multivector
        let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Encode
        let coeffs = encode_multivector_slots(&mv, scale, n);
        assert_eq!(coeffs.len(), n);

        // Decode
        let mv_decoded = decode_multivector_slots(&coeffs, scale, n);

        // Check roundtrip accuracy
        for i in 0..8 {
            let error = (mv_decoded[i] - mv[i]).abs();
            assert!(
                error < 1e-6,
                "Component {} error too large: {} (got {}, expected {})",
                i,
                error,
                mv_decoded[i],
                mv[i]
            );
        }
    }

    #[test]
    fn test_slot_mask() {
        let n = 64;
        let scale = 2f64.powi(20);

        // Create mask for slot 3
        let mask_coeffs = create_slot_mask(3, scale, n);

        // Decode to verify
        let slots = coefficients_to_slots(&mask_coeffs, scale, n);

        // Slot 3 should be 1.0, others should be ~0.0
        for i in 0..8 {
            let expected = if i == 3 { 1.0 } else { 0.0 };
            let error = (slots[i].re - expected).abs();
            assert!(
                error < 1e-6,
                "Slot {} error: {} (got {}, expected {})",
                i,
                error,
                slots[i].re,
                expected
            );
        }
    }

    #[test]
    fn test_slots_to_coefficients_to_slots() {
        let n = 64;
        let scale = 2f64.powi(20);
        let num_slots = n / 2;

        // Create random slots
        let mut original_slots = vec![Complex::new(0.0, 0.0); num_slots];
        for i in 0..8 {
            original_slots[i] = Complex::new(i as f64 + 1.0, 0.0);
        }

        // slots → coefficients → slots
        let coeffs = slots_to_coefficients(&original_slots, scale, n);
        let recovered_slots = coefficients_to_slots(&coeffs, scale, n);

        // Check roundtrip
        for i in 0..8 {
            let error = (recovered_slots[i].re - original_slots[i].re).abs();
            assert!(
                error < 1e-6,
                "Slot {} error: {} (got {}, expected {})",
                i,
                error,
                recovered_slots[i].re,
                original_slots[i].re
            );
        }
    }

    #[test]
    fn test_zero_multivector() {
        let n = 64;
        let scale = 2f64.powi(20);

        let mv = [0.0; 8];
        let coeffs = encode_multivector_slots(&mv, scale, n);
        let mv_decoded = decode_multivector_slots(&coeffs, scale, n);

        for i in 0..8 {
            assert!(mv_decoded[i].abs() < 1e-6);
        }
    }

    #[test]
    fn test_large_values() {
        let n = 64;
        let scale = 2f64.powi(20);

        // Test with large values
        let mv = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0];

        let coeffs = encode_multivector_slots(&mv, scale, n);
        let mv_decoded = decode_multivector_slots(&coeffs, scale, n);

        for i in 0..8 {
            let error = (mv_decoded[i] - mv[i]).abs();
            assert!(
                error < 1e-3, // Slightly relaxed for large values
                "Component {} error: {} (got {}, expected {})",
                i,
                error,
                mv_decoded[i],
                mv[i]
            );
        }
    }
}
