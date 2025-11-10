/// Packing and Unpacking Operations
///
/// Convert between V2/V3 naive component-separate layout and V4 packed slot-interleaved layout.

use super::packed_multivector::PackedMultivector;

#[cfg(feature = "v2-gpu-cuda")]
use crate::clifford_fhe_v2::backends::gpu_cuda::{
    ckks::{CudaCiphertext as Ciphertext, CudaCkksContext},
    rotation_keys::CudaRotationKeys as RotationKeys,
};

#[cfg(all(feature = "v2-gpu-metal", not(feature = "v2-gpu-cuda")))]
use crate::clifford_fhe_v2::backends::gpu_metal::{
    ckks::{MetalCiphertext as Ciphertext, MetalCkksContext as CudaCkksContext},
    rotation_keys::MetalRotationKeys as RotationKeys,
};

#[cfg(all(feature = "v2-cpu-optimized", not(feature = "v2-gpu-cuda"), not(feature = "v2-gpu-metal")))]
use crate::clifford_fhe_v2::backends::cpu_optimized::{
    ckks::{Ciphertext, CpuCkksContext as CudaCkksContext},
    // Note: CPU backend may not have rotation keys implemented yet
};

/// Pack multiple component-separate ciphertexts into a single packed ciphertext
///
/// Input: 8 ciphertexts [ct_s, ct_e1, ct_e2, ct_e3, ct_e12, ct_e23, ct_e31, ct_I]
///        Each encrypts batch_size scalar values
///
/// Output: Single ciphertext with interleaved slots
///         [s₀, e1₀, e2₀, e3₀, e12₀, e23₀, e31₀, I₀, s₁, e1₁, ...]
///
/// Algorithm:
/// 1. For each component i:
///    - Rotate ct[i] left by i positions
/// 2. Sum all rotated ciphertexts
///
/// This interleaves the components by rotating each one into its designated position
/// within each 8-slot group.
#[cfg(any(feature = "v2-gpu-cuda", feature = "v2-gpu-metal"))]
pub fn pack_multivector(
    components: &[Ciphertext; 8],
    batch_size: usize,
    rot_keys: &RotationKeys,
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    // Get parameters from the first ciphertext (they should all match)
    let n = components[0].n;
    let num_primes = components[0].num_primes;
    let level = components[0].level;
    let scale = components[0].scale;

    // Verify all components are compatible
    for (i, ct) in components.iter().enumerate() {
        if ct.n != n {
            return Err(format!("Component {} has mismatched n", i));
        }
        if ct.num_primes != num_primes {
            return Err(format!("Component {} has mismatched num_primes", i));
        }
        if ct.level != level {
            return Err(format!("Component {} has mismatched level", i));
        }
        if (ct.scale - scale).abs() > 1e-6 {
            return Err(format!("Component {} has mismatched scale", i));
        }
    }

    // Verify batch size is valid
    if batch_size * 8 > n / 2 {
        return Err(format!(
            "Batch size {} × 8 components = {} exceeds n/2 = {}",
            batch_size, batch_size * 8, n / 2
        ));
    }

    // Start with component 0 (scalar) - no rotation needed
    let mut packed_ct = components[0].clone();

    // Rotate and accumulate remaining components
    for i in 1..8 {
        // Rotate component i left by i positions
        let rotated = components[i].rotate_by_steps(i as i32, rot_keys, ckks_ctx)?;

        // Add to accumulator
        packed_ct = packed_ct.add(&rotated, ckks_ctx)?;
    }

    Ok(PackedMultivector::new(
        packed_ct,
        batch_size,
        n,
        num_primes,
        level,
        scale,
    ))
}

/// CPU-optimized version (placeholder - rotation may not be available)
#[cfg(all(feature = "v2-cpu-optimized", not(feature = "v2-gpu-cuda"), not(feature = "v2-gpu-metal")))]
pub fn pack_multivector(
    components: &[Ciphertext; 8],
    batch_size: usize,
    ckks_ctx: &CudaCkksContext,
) -> Result<PackedMultivector, String> {
    Err("Packing not yet implemented for CPU backend - requires rotation keys".to_string())
}

/// Unpack a packed ciphertext into 8 component-separate ciphertexts
///
/// Input: Single packed ciphertext with interleaved slots
///        [s₀, e1₀, e2₀, e3₀, e12₀, e23₀, e31₀, I₀, s₁, e1₁, ...]
///
/// Output: 8 ciphertexts [ct_s, ct_e1, ct_e2, ct_e3, ct_e12, ct_e23, ct_e31, ct_I]
///
/// Algorithm:
/// 1. For each component i:
///    - Rotate packed_ct right by i positions (to align component i to position 0)
///    - Multiply by mask plaintext that zeros out positions not divisible by 8
///
/// TODO: Implement masking (for now, just rotations without masking)
#[cfg(any(feature = "v2-gpu-cuda", feature = "v2-gpu-metal"))]
pub fn unpack_multivector(
    packed: &PackedMultivector,
    rot_keys: &RotationKeys,
    ckks_ctx: &CudaCkksContext,
) -> Result<[Ciphertext; 8], String> {
    let mut components = Vec::with_capacity(8);

    for i in 0..8 {
        // Rotate right by i positions (equivalent to left by -i)
        // This brings component i to the 0th position of each 8-slot group
        let rotated = if i == 0 {
            // No rotation needed for component 0
            packed.ct.clone()
        } else {
            packed.ct.rotate_by_steps(-(i as i32), rot_keys, ckks_ctx)?
        };

        // TODO: Apply mask to zero out positions 1-7, 9-15, 17-23, etc.
        // For now, store the rotated ciphertext (will decrypt to have all components)
        components.push(rotated);
    }

    Ok([
        components[0].clone(),
        components[1].clone(),
        components[2].clone(),
        components[3].clone(),
        components[4].clone(),
        components[5].clone(),
        components[6].clone(),
        components[7].clone(),
    ])
}

/// CPU version (placeholder)
#[cfg(all(feature = "v2-cpu-optimized", not(feature = "v2-gpu-cuda"), not(feature = "v2-gpu-metal")))]
pub fn unpack_multivector(
    packed: &PackedMultivector,
    ckks_ctx: &CudaCkksContext,
) -> Result<[Ciphertext; 8], String> {
    Err("Unpacking not yet implemented for CPU backend - requires rotation keys".to_string())
}

/// Extract a single component from a packed ciphertext
///
/// This is more efficient than unpacking all 8 components when you only need one.
///
/// Algorithm:
/// 1. Rotate packed_ct right by component_idx positions
/// 2. Multiply by mask plaintext (TODO: masking not yet implemented)
///
#[cfg(any(feature = "v2-gpu-cuda", feature = "v2-gpu-metal"))]
pub fn extract_component(
    packed: &PackedMultivector,
    component_idx: usize,
    rot_keys: &RotationKeys,
    ckks_ctx: &CudaCkksContext,
) -> Result<Ciphertext, String> {
    if component_idx >= 8 {
        return Err(format!("Component index {} out of range [0,8)", component_idx));
    }

    // Rotate right by component_idx positions
    let rotated = if component_idx == 0 {
        packed.ct.clone()
    } else {
        packed.ct.rotate_by_steps(-(component_idx as i32), rot_keys, ckks_ctx)?
    };

    // TODO: Apply mask to zero out positions 1-7, 9-15, etc.
    Ok(rotated)
}

/// CPU version (placeholder)
#[cfg(all(feature = "v2-cpu-optimized", not(feature = "v2-gpu-cuda"), not(feature = "v2-gpu-metal")))]
pub fn extract_component(
    packed: &PackedMultivector,
    component_idx: usize,
    ckks_ctx: &CudaCkksContext,
) -> Result<Ciphertext, String> {
    Err("Component extraction not yet implemented for CPU backend".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Note: Actual tests will require V2 CKKS context initialization
    // For now, just test the API structure
    
    #[test]
    fn test_api_structure() {
        // This test just verifies the API compiles
        // Real tests will be added once packing is implemented
    }
}
