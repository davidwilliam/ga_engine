//! Encoding/decoding between Clifford algebra multivectors and CKKS polynomials
//!
//! # Core Idea
//!
//! A multivector in Cl(3,0) has 8 components: [s, e1, e2, e3, e12, e13, e23, e123]
//!
//! We encode this as a polynomial in R = Z[x]/(x^N + 1):
//! ```text
//! m(x) = s + e1·x + e2·x² + e3·x³ + e12·x⁴ + e13·x⁵ + e23·x⁶ + e123·x⁷
//! ```
//!
//! This allows CKKS to operate on the polynomial representation,
//! while we maintain geometric algebra semantics.

/// Encode a multivector as polynomial coefficients for CKKS
///
/// # Arguments
/// * `mv` - Multivector with 8 floating-point components
/// * `scale` - CKKS scaling factor (determines fixed-point precision)
/// * `n` - Polynomial degree (must be ≥ 8)
///
/// # Returns
/// Vector of scaled integer coefficients ready for CKKS encryption
///
/// # Example
/// ```rust,ignore
/// let mv = [1.5, 2.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
/// let scale = 2f64.powi(40);
/// let poly = encode_multivector(&mv, scale, 8192);
/// // poly[0] = round(1.5 * 2^40)
/// // poly[1] = round(2.3 * 2^40)
/// // ... rest are zero-padded to length 8192
/// ```
pub fn encode_multivector(mv: &[f64; 8], scale: f64, n: usize) -> Vec<i64> {
    assert!(n >= 8, "Polynomial degree must be at least 8 for Cl(3,0)");

    let mut poly = vec![0i64; n];

    // Encode each multivector component as a polynomial coefficient
    // Component 0 (scalar) → coefficient of x^0
    // Component 1 (e1) → coefficient of x^1
    // ... and so on
    for i in 0..8 {
        poly[i] = (mv[i] * scale).round() as i64;
    }

    // Remaining coefficients are 0 (padding)
    poly
}

/// Decode polynomial coefficients back to multivector
///
/// # Arguments
/// * `poly` - Polynomial coefficients (first 8 are multivector components)
/// * `scale` - CKKS scaling factor used during encoding
///
/// # Returns
/// Multivector with 8 floating-point components
///
/// # Example
/// ```rust,ignore
/// let poly = vec![1649267441664i64, ...]; // Scaled coefficients
/// let scale = 2f64.powi(40);
/// let mv = decode_multivector(&poly, scale);
/// // mv ≈ [1.5, 2.3, 0.0, ...]
/// ```
pub fn decode_multivector(poly: &[i64], scale: f64) -> [f64; 8] {
    assert!(
        poly.len() >= 8,
        "Polynomial must have at least 8 coefficients"
    );

    let mut mv = [0f64; 8];
    for i in 0..8 {
        mv[i] = poly[i] as f64 / scale;
    }
    mv
}

/// Encode multivector for integer-based CKKS (alternative approach)
///
/// Instead of floating-point, use CliffordRingElementInt directly.
/// This is useful if we want to avoid floating-point rounding entirely.
pub fn encode_multivector_int(mv: &[i64; 8], n: usize) -> Vec<i64> {
    assert!(n >= 8, "Polynomial degree must be at least 8 for Cl(3,0)");

    let mut poly = vec![0i64; n];
    for i in 0..8 {
        poly[i] = mv[i];
    }
    poly
}

/// Decode integer polynomial to multivector
pub fn decode_multivector_int(poly: &[i64]) -> [i64; 8] {
    assert!(
        poly.len() >= 8,
        "Polynomial must have at least 8 coefficients"
    );

    let mut mv = [0i64; 8];
    for i in 0..8 {
        mv[i] = poly[i];
    }
    mv
}

/// SIMD packing: Encode multiple multivectors in one polynomial
///
/// CKKS supports SIMD-style packing where a single polynomial can encode
/// many values using different slots (via NTT structure).
///
/// For N=8192, we can pack up to N/2 = 4096 complex values.
/// Each multivector needs 8 slots, so we can pack 4096/8 = 512 multivectors!
///
/// # Future optimization
pub fn encode_multivector_batch(mvs: &[[f64; 8]], scale: f64, n: usize) -> Vec<i64> {
    let num_mvs = mvs.len();
    let slots_per_mv = 8;
    let max_mvs = n / slots_per_mv;

    assert!(
        num_mvs <= max_mvs,
        "Can pack at most {} multivectors with N={}",
        max_mvs,
        n
    );

    let mut poly = vec![0i64; n];

    for (mv_idx, mv) in mvs.iter().enumerate() {
        let base_idx = mv_idx * slots_per_mv;
        for i in 0..8 {
            poly[base_idx + i] = (mv[i] * scale).round() as i64;
        }
    }

    poly
}

/// Decode batch of multivectors from SIMD-packed polynomial
pub fn decode_multivector_batch(poly: &[i64], scale: f64, num_mvs: usize) -> Vec<[f64; 8]> {
    let slots_per_mv = 8;
    let mut mvs = Vec::with_capacity(num_mvs);

    for mv_idx in 0..num_mvs {
        let base_idx = mv_idx * slots_per_mv;
        let mut mv = [0f64; 8];
        for i in 0..8 {
            if base_idx + i < poly.len() {
                mv[i] = poly[base_idx + i] as f64 / scale;
            }
        }
        mvs.push(mv);
    }

    mvs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let mv = [1.5, 2.3, -0.7, 3.14, 0.0, 1.0, -2.0, 0.5];
        let scale = 2f64.powi(40);
        let n = 8192;

        let poly = encode_multivector(&mv, scale, n);
        let decoded = decode_multivector(&poly, scale);

        // Check all components match (with floating-point tolerance)
        for i in 0..8 {
            assert!(
                (decoded[i] - mv[i]).abs() < 1e-9,
                "Component {} mismatch: {} vs {}",
                i,
                decoded[i],
                mv[i]
            );
        }
    }

    #[test]
    fn test_encode_multivector_structure() {
        let mv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let scale = 1000.0; // Use small scale for easy verification
        let n = 16;

        let poly = encode_multivector(&mv, scale, n);

        // Check first 8 coefficients
        assert_eq!(poly[0], 1000); // scalar
        assert_eq!(poly[1], 2000); // e1
        assert_eq!(poly[2], 3000); // e2
        assert_eq!(poly[7], 8000); // e123

        // Check padding
        for i in 8..n {
            assert_eq!(poly[i], 0);
        }
    }

    #[test]
    fn test_batch_encoding() {
        let mvs = vec![
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let scale = 2f64.powi(40);
        let n = 8192;

        let poly = encode_multivector_batch(&mvs, scale, n);
        let decoded = decode_multivector_batch(&poly, scale, 3);

        assert_eq!(decoded.len(), 3);
        for (i, mv) in decoded.iter().enumerate() {
            for j in 0..8 {
                let expected = mvs[i][j];
                assert!(
                    (mv[j] - expected).abs() < 1e-9,
                    "MV {} component {} mismatch",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_integer_encoding() {
        let mv = [100, 200, -50, 0, 150, -100, 75, 0];
        let n = 16;

        let poly = encode_multivector_int(&mv, n);
        let decoded = decode_multivector_int(&poly);

        assert_eq!(mv, decoded);
    }
}
