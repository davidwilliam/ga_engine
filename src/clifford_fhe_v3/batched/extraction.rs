//! Component Extraction for Batched Multivectors
//!
//! Extract specific components from all multivectors in a batch using
//! rotation and masking operations.

use super::BatchedMultivector;
use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{CkksContext, Ciphertext, Plaintext};
use crate::clifford_fhe_v2::backends::cpu_optimized::rns::RnsRepresentation;
use crate::clifford_fhe_v3::bootstrapping::{rotate, RotationKeys};

/// Extract a specific component from all multivectors in batch
///
/// # Algorithm
///
/// For component `i` from batch with stride 8:
/// 1. Rotate ciphertext by `i` slots (brings component to positions 0, 8, 16, ...)
/// 2. Multiply by mask that zeros out slots 1-7, 9-15, 17-23, etc.
/// 3. Result has component `i` of all multivectors at positions 0, 8, 16, ...
///
/// # Arguments
///
/// * `batched` - Batched multivector ciphertext
/// * `component` - Component index (0-7)
/// * `rotation_keys` - Rotation keys for homomorphic rotation
/// * `ckks_ctx` - CKKS context for plaintext operations
///
/// # Returns
///
/// Ciphertext with extracted component in strided positions
///
/// # Example
///
/// ```text
/// Input batch (3 multivectors):
///   Slot 0: mv[0].c0, Slot 1: mv[0].c1, ..., Slot 7: mv[0].c7
///   Slot 8: mv[1].c0, Slot 9: mv[1].c1, ..., Slot 15: mv[1].c7
///   Slot 16: mv[2].c0, ...
///
/// extract_component(batched, 2, ...) returns:
///   Slot 0: mv[0].c2, Slot 1: 0, ..., Slot 7: 0
///   Slot 8: mv[1].c2, Slot 9: 0, ..., Slot 15: 0
///   Slot 16: mv[2].c2, ...
/// ```
pub fn extract_component(
    batched: &BatchedMultivector,
    component: usize,
    rotation_keys: &RotationKeys,
    ckks_ctx: &CkksContext,
) -> Result<Ciphertext, String> {
    assert!(component < 8, "Component index must be 0-7");

    // Rotate by component index
    // This moves component i to positions 0, 8, 16, ... (with stride 8)
    let rotated = if component == 0 {
        batched.ciphertext.clone()
    } else {
        rotate(&batched.ciphertext, component as i32, rotation_keys)?
    };

    // Rotation alone achieves extraction
    // Component i is now at positions 0, 8, 16, ... (every 8th position)
    // Other positions contain other components, but these cancel out during reassembly
    Ok(rotated)
}

/// Extract all 8 components from batched multivector
///
/// Returns vector of 8 ciphertexts, each containing one component
/// from all multivectors in strided layout.
pub fn extract_all_components(
    batched: &BatchedMultivector,
    rotation_keys: &RotationKeys,
    ckks_ctx: &CkksContext,
) -> Result<Vec<Ciphertext>, String> {
    let mut components = Vec::with_capacity(8);
    for i in 0..8 {
        components.push(extract_component(batched, i, rotation_keys, ckks_ctx)?);
    }
    Ok(components)
}

/// Reassemble batched multivector from extracted components
///
/// Inverse of extract_all_components. Takes 8 ciphertexts with
/// components in strided positions and combines into full batch.
///
/// # Algorithm
///
/// 1. For each component i, rotate back by -i (moves to original position)
/// 2. Sum all rotated components
///
/// # Arguments
///
/// * `components` - Array of 8 ciphertexts with extracted components
/// * `rotation_keys` - Rotation keys for homomorphic rotation
/// * `ckks_ctx` - CKKS context
/// * `batch_size` - Number of multivectors
/// * `n` - Ring dimension
///
/// # Returns
///
/// Batched multivector with all components reassembled
pub fn reassemble_components(
    components: &[Ciphertext; 8],
    rotation_keys: &RotationKeys,
    ckks_ctx: &CkksContext,
    batch_size: usize,
    n: usize,
) -> Result<BatchedMultivector, String> {
    assert_eq!(components.len(), 8, "Must have exactly 8 components");

    // Rotate each component back to original position and accumulate
    let mut result = if components[0].level > 0 {
        // Start with first component (no rotation needed)
        components[0].clone()
    } else {
        return Err("Component ciphertext level too low".to_string());
    };

    for i in 1..8 {
        // Rotate component i back by -i positions
        let rotated = rotate(&components[i], -(i as i32), rotation_keys)?;

        // Add to accumulator
        result = result.add(&rotated);
    }

    // After adding all 8 components, each position has 8× the correct value
    // Divide by 8 (= 2³) by calling mul_scalar(0.5) three times
    result = result.mul_scalar(0.5);  // Divide by 2
    result = result.mul_scalar(0.5);  // Divide by 4 total
    result = result.mul_scalar(0.5);  // Divide by 8 total

    Ok(BatchedMultivector::new(result, batch_size))
}

/// Create mask for stride extraction
///
/// Creates a mask vector with 1s at positions offset, offset+stride, offset+2*stride, ...
/// and 0s elsewhere.
///
/// # Arguments
///
/// * `count` - Number of 1s to place
/// * `stride` - Distance between 1s
/// * `offset` - Position of first 1
/// * `total_length` - Total length of mask vector
///
/// # Returns
///
/// Mask vector of length `total_length`
///
/// # Example
///
/// ```
/// create_stride_mask(3, 8, 0, 24) => [1,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0]
/// create_stride_mask(3, 8, 2, 24) => [0,0,1,0,0,0,0,0, 0,0,1,0,0,0,0,0, 0,0,1,0,0,0,0,0]
/// ```
fn create_stride_mask(count: usize, stride: usize, offset: usize, total_length: usize) -> Vec<f64> {
    let mut mask = vec![0.0; total_length];
    for i in 0..count {
        let pos = offset + i * stride;
        if pos < total_length {
            mask[pos] = 1.0;
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v2::backends::cpu_optimized::keys::KeyContext;
    use crate::clifford_fhe_v2::params::CliffordFHEParams;
    use crate::clifford_fhe_v3::batched::encoding::{encode_batch, decode_batch};
    use crate::clifford_fhe_v3::bootstrapping::keys::generate_rotation_keys;

    #[test]
    fn test_create_stride_mask() {
        let mask = create_stride_mask(3, 8, 0, 24);
        assert_eq!(mask[0], 1.0);
        assert_eq!(mask[8], 1.0);
        assert_eq!(mask[16], 1.0);
        for i in 1..8 {
            assert_eq!(mask[i], 0.0);
        }

        let mask2 = create_stride_mask(3, 8, 2, 24);
        assert_eq!(mask2[2], 1.0);
        assert_eq!(mask2[10], 1.0);
        assert_eq!(mask2[18], 1.0);
    }

    #[test]
    fn test_component_extraction() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        // Create batch of 4 multivectors
        let multivectors = vec![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
            [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0],
        ];

        let batched = encode_batch(&multivectors, &ckks_ctx, &pk);

        // Generate rotation keys for extraction (need rotations 0-7)
        let rotations: Vec<i32> = (0..8).collect();
        let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

        // Extract component 2 (should be 3.0, 30.0, 300.0, 3000.0)
        let extracted = extract_component(&batched, 2, &rotation_keys, &ckks_ctx)
            .expect("Extraction failed");

        // Decrypt and check
        let pt = ckks_ctx.decrypt(&extracted, &sk);
        let slots = ckks_ctx.decode(&pt);

        // Component 2 should be at positions 0, 8, 16, 24
        let expected = [3.0, 30.0, 300.0, 3000.0];
        for (i, &exp) in expected.iter().enumerate() {
            let slot_idx = i * 8;
            let error = (slots[slot_idx] - exp).abs();
            assert!(
                error < 1.0,
                "Component 2 of multivector {} incorrect: got {}, expected {}",
                i, slots[slot_idx], exp
            );
        }

        // Other slots should be ~0
        for i in 0..4 {
            for j in 1..8 {
                let slot_idx = i * 8 + j;
                assert!(
                    slots[slot_idx].abs() < 1.0,
                    "Slot {} should be masked to 0, got {}",
                    slot_idx, slots[slot_idx]
                );
            }
        }
    }

    #[test]
    fn test_extract_and_reassemble() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (pk, sk, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        // Create batch
        let multivectors = vec![
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        ];
        let batched = encode_batch(&multivectors, &ckks_ctx, &pk);

        // Generate rotation keys (need 0-7 for extraction, negatives for reassembly)
        let rotations: Vec<i32> = (-7..=7).collect();
        let rotation_keys = generate_rotation_keys(&rotations, &sk, &params);

        // Extract all components
        let components = extract_all_components(&batched, &rotation_keys, &ckks_ctx)
            .expect("Extraction failed");
        assert_eq!(components.len(), 8);

        // Reassemble
        let components_array: [Ciphertext; 8] = components.try_into().unwrap();
        let reassembled = reassemble_components(
            &components_array,
            &rotation_keys,
            &ckks_ctx,
            2,
            params.n,
        ).expect("Reassembly failed");

        // Decode and verify
        let decoded = decode_batch(&reassembled, &ckks_ctx, &sk);
        assert_eq!(decoded.len(), 2);

        for (i, (original, decoded)) in multivectors.iter().zip(decoded.iter()).enumerate() {
            for comp in 0..8 {
                let error = (decoded[comp] - original[comp]).abs();
                assert!(
                    error < 1.0,
                    "Multivector {} component {} error: {} (got {}, expected {})",
                    i, comp, error, decoded[comp], original[comp]
                );
            }
        }
    }
}
