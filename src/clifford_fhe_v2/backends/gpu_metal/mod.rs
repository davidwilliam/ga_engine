//! Metal GPU Backend for Clifford FHE V2 (Apple Silicon)
//!
//! **Target:** 30-50× speedup vs V1 baseline
//!
//! **Optimizations:**
//! - Metal compute shaders for NTT
//! - Unified memory architecture (M1/M2/M3 advantage)
//! - Neural Engine integration (experimental)
//!
//! **Requirements:**
//! - Apple Silicon Mac (M1/M2/M3)
//! - `metal` Rust crate
//!
//! **Status:** Phase 2 implementation (V2 optimization roadmap)

#[cfg(feature = "v2-gpu-metal")]
use crate::clifford_fhe_v2::core::{BackendCapabilities, BackendInfo};

#[cfg(feature = "v2-gpu-metal")]
pub struct GpuMetalBackend;

#[cfg(feature = "v2-gpu-metal")]
impl BackendInfo for GpuMetalBackend {
    fn capabilities() -> BackendCapabilities {
        BackendCapabilities {
            has_ntt_optimization: true,
            has_gpu_acceleration: true,
            has_simd_batching: false,
            has_rotation_keys: false,
        }
    }

    fn max_polynomial_degree() -> usize {
        16384  // Metal supports up to N=16384
    }

    fn recommended_params() -> Vec<String> {
        vec![
            "N=1024, primes=3 (Apple Silicon optimal)".to_string(),
            "N=2048, primes=5 (balanced)".to_string(),
        ]
    }
}

#[cfg(not(feature = "v2-gpu-metal"))]
pub struct GpuMetalBackend;

#[cfg(not(feature = "v2-gpu-metal"))]
impl GpuMetalBackend {
    pub fn not_available() -> ! {
        panic!("Metal backend not compiled. Enable with: --features v2-gpu-metal");
    }
}
