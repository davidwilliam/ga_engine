# Metal GPU Rotation Implementation Design

## Overview

Implement homomorphic rotation (Galois automorphisms) on Metal GPU to unlock full GPU-accelerated bootstrap for V3.

**Current Status:** Rotation for Metal ciphertexts **DOES NOT EXIST**
**Blocker:** Bootstrap operations (CoeffToSlot/SlotToCoeff) convert GPU→CPU→GPU
**Goal:** 12× speedup by keeping everything on GPU

---

## Background: How Rotation Works in CKKS

### Mathematical Foundation

CKKS operates in the ring **R = ℤ[X]/(X^N + 1)** where N is a power of 2.

**Galois Automorphism:**
- For each k ∈ ℤ*_{2N} (units mod 2N), we have an automorphism σ_k: X → X^k
- This permutes the slots in a specific pattern
- For power-of-two N, the group is isomorphic to ℤ/2 × ℤ/(N/2)

**Slot Rotation:**
- **Column rotation by r steps:** Uses σ_k where k = 5^r (mod 2N)
- **Row rotation (conjugation):** Uses σ_k where k = 2N-1 (conjugates complex slots)

**Key Property:**
σ_k(c₀ + c₁·s) ≠ σ_k(c₀) + σ_k(c₁)·s  (automorphism doesn't commute with secret key!)

**Solution: Key Switching**
After applying σ_k to ciphertext, we need rotation keys to "fix" the relationship with s.

---

## CPU Implementation Analysis

### Existing CPU Rotation (DOES NOT EXIST)

Searching the codebase shows **NO rotation implementation** in V2:
- `has_rotation_keys: false` in GPU backend capabilities
- No `rotate_by_steps` or `rotate_slots` functions found
- CoeffToSlot/SlotToCoeff use rotation but only exist for V3 (CPU-only)

### V3 Bootstrap Uses Rotation

From [V3_METAL_GPU_ASSESSMENT.md](V3_METAL_GPU_ASSESSMENT.md):
```
CoeffToSlot: Needs 24 rotations (log₂ N + 1 for N=1024)
SlotToCoeff: Needs 24 rotations
```

**Current V3 behavior:**
```rust
// v3/bootstrapping/coeff_to_slot.rs (lines ~150-180)
for i in 0..logn+1 {
    current = current.rotate_by_steps(rotation_step, rot_keys);
    // ↑ This ONLY works on CPU ciphertexts!
}
```

---

## Metal GPU Rotation Design

### Phase 1: Core Infrastructure

#### 1.1 Metal Shader for Galois Automorphism

**File:** `src/clifford_fhe_v2/backends/gpu_metal/shaders/rotation.metal`

```metal
//! Metal Compute Shaders for Galois Automorphisms (Homomorphic Rotation)
//!
//! Implements permutation X → X^k for CKKS slot rotations.

#include <metal_stdlib>
using namespace metal;

/// Apply Galois automorphism: polynomial[i] → polynomial[galois_map[i]]
///
/// For σ_k: X → X^k in ring R = Z[X]/(X^N + 1):
/// - Input:  f(X) = Σ fᵢ X^i
/// - Output: f(X^k) = Σ fᵢ X^(i·k mod 2N) with sign correction
///
/// The galois_map precomputes the permutation and sign flips.
///
/// @param input Input polynomial coefficients (N coefficients × num_primes RNS)
/// @param output Output polynomial after automorphism
/// @param galois_map Precomputed permutation map [N]
/// @param galois_signs Sign correction (+1 or -1) [N]
/// @param n Polynomial degree
/// @param num_primes Number of RNS components
/// @param moduli Array of RNS moduli [num_primes]
kernel void apply_galois_automorphism(
    device const ulong* input [[buffer(0)]],
    device ulong* output [[buffer(1)]],
    constant uint* galois_map [[buffer(2)]],
    constant int* galois_signs [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& num_primes [[buffer(5)]],
    constant ulong* moduli [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    // Each thread handles one coefficient across all RNS components
    if (gid < n) {
        uint target_idx = galois_map[gid];  // Where this coefficient goes
        int sign = galois_signs[gid];        // Sign flip (+1 or -1)

        // Process all RNS components for this coefficient
        for (uint prime_idx = 0; prime_idx < num_primes; prime_idx++) {
            ulong q = moduli[prime_idx];

            // Flat layout index: coeff_idx * num_primes + prime_idx
            uint in_idx = gid * num_primes + prime_idx;
            uint out_idx = target_idx * num_primes + prime_idx;

            ulong val = input[in_idx];

            // Apply sign correction
            if (sign < 0 && val != 0) {
                val = q - val;  // Negate: -x = q - x (mod q)
            }

            output[out_idx] = val;
        }
    }
}
```

**Key Design Decisions:**
- **Precomputed galois_map:** CPU computes permutation once, GPU just applies it
- **Flat RNS layout:** Same as existing Metal CKKS (coeff-major ordering)
- **Fully parallel:** Each thread handles one coefficient, no synchronization needed
- **Sign correction:** Handles X^(i·k) > N case where we get -X^(i·k mod N)

#### 1.2 Galois Map Precomputation (CPU)

**File:** `src/clifford_fhe_v2/backends/gpu_metal/rotation.rs`

```rust
/// Precompute Galois automorphism map for X → X^k in R = Z[X]/(X^N + 1)
///
/// Returns (permutation_map, sign_map) where:
/// - permutation_map[i] = target coefficient index
/// - sign_map[i] = +1 or -1 (sign correction)
pub fn compute_galois_map(n: usize, k: usize) -> (Vec<u32>, Vec<i32>) {
    let two_n = 2 * n;
    let mut perm = vec![0u32; n];
    let mut signs = vec![1i32; n];

    for i in 0..n {
        // Compute i·k mod 2N
        let ik = ((i * k) % two_n) as usize;

        if ik < n {
            // X^(i·k) stays positive
            perm[i] = ik as u32;
            signs[i] = 1;
        } else {
            // X^(i·k) = -X^(i·k - N) in quotient ring (since X^N = -1)
            perm[i] = (ik - n) as u32;
            signs[i] = -1;
        }
    }

    (perm, signs)
}

/// Convert rotation step to Galois element k
///
/// For power-of-two cyclotomics, k = 5^step (mod 2N)
pub fn rotation_step_to_galois_element(step: i32, n: usize) -> usize {
    let two_n = 2 * n;
    let g = 5usize;  // Generator for power-of-two cyclotomics

    if step >= 0 {
        // Positive rotation: k = 5^step mod 2N
        pow_mod(g, step as usize, two_n)
    } else {
        // Negative rotation: k = 5^(-step) = 5^(φ(2N) + step) mod 2N
        // φ(2N) = N for power-of-two N
        let phi = n;
        let adjusted_step = (phi as i32 + step) as usize;
        pow_mod(g, adjusted_step, two_n)
    }
}

fn pow_mod(base: usize, exp: usize, modulus: usize) -> usize {
    let mut result = 1;
    let mut b = base % modulus;
    let mut e = exp;

    while e > 0 {
        if e & 1 == 1 {
            result = (result * b) % modulus;
        }
        b = (b * b) % modulus;
        e >>= 1;
    }

    result
}
```

#### 1.3 Metal Rotation Keys Structure

**File:** `src/clifford_fhe_v2/backends/gpu_metal/rotation_keys.rs`

```rust
use super::device::MetalDevice;
use std::sync::Arc;
use std::collections::HashMap;

/// Metal GPU Rotation Keys
///
/// Stores key-switching keys for each rotation step.
/// Keys are stored in Metal buffers for GPU access.
pub struct MetalRotationKeys {
    /// Metal device (shared)
    device: Arc<MetalDevice>,

    /// Rotation keys for each Galois element k
    /// Maps k → (a_k, b_k) where each is Vec<u64> in flat RNS layout
    keys: HashMap<usize, (Vec<u64>, Vec<u64>)>,

    /// Ring dimension
    n: usize,

    /// Number of RNS primes
    num_primes: usize,

    /// Level these keys were generated for
    level: usize,
}

impl MetalRotationKeys {
    /// Generate rotation keys for a set of rotation steps
    ///
    /// Uses same gadget decomposition as evaluation keys (base 2^20).
    pub fn generate(
        device: Arc<MetalDevice>,
        sk: &SecretKey,
        rotation_steps: &[i32],
        params: &CliffordFHEParams,
        ntt_contexts: &[MetalNttContext],
    ) -> Result<Self, String> {
        let n = params.n;
        let level = sk.level;
        let moduli = &params.moduli[..=level];
        let num_primes = moduli.len();

        let mut keys = HashMap::new();

        for &step in rotation_steps {
            // Convert rotation step to Galois element k
            let k = rotation_step_to_galois_element(step, n);

            // Skip if already generated
            if keys.contains_key(&k) {
                continue;
            }

            // Generate rotation key for σ_k
            let (a_k, b_k) = Self::generate_rotation_key_for_k(
                k, sk, moduli, n, ntt_contexts
            )?;

            keys.insert(k, (a_k, b_k));
        }

        Ok(Self {
            device,
            keys,
            n,
            num_primes,
            level,
        })
    }

    /// Generate rotation key for specific Galois element k
    ///
    /// Key switching key: (a, b) where b ≈ -a·s + e + σ_k(s)
    fn generate_rotation_key_for_k(
        k: usize,
        sk: &SecretKey,
        moduli: &[u64],
        n: usize,
        ntt_contexts: &[MetalNttContext],
    ) -> Result<(Vec<u64>, Vec<u64>), String> {
        use rand::{thread_rng, Rng};
        use rand_distr::{Distribution, Normal};

        // Apply Galois automorphism to secret key: s_k = σ_k(s)
        let (galois_map, galois_signs) = compute_galois_map(n, k);
        let s_k = Self::apply_galois_to_secret_key(sk, &galois_map, &galois_signs, moduli);

        // Sample uniform random a
        let mut rng = thread_rng();
        let a: Vec<u64> = (0..(n * moduli.len()))
            .map(|i| {
                let prime_idx = i % moduli.len();
                rng.gen_range(0..moduli[prime_idx])
            })
            .collect();

        // Sample error e from Gaussian
        let normal = Normal::new(0.0, 3.2).unwrap();
        let e: Vec<i64> = (0..n).map(|_| normal.sample(&mut rng).round() as i64).collect();
        let e_flat = Self::coeffs_to_flat_rns(&e, moduli);

        // Compute b = -a·s + e + s_k using NTT multiplication on GPU
        let as_product = Self::multiply_polys_ntt(&a, &Self::sk_to_flat(sk, moduli), moduli, ntt_contexts)?;

        // b = -a·s + e + s_k (all in flat RNS layout)
        let mut b = vec![0u64; n * moduli.len()];
        for i in 0..(n * moduli.len()) {
            let prime_idx = i % moduli.len();
            let q = moduli[prime_idx];

            // -a·s
            let neg_as = if as_product[i] == 0 { 0 } else { q - as_product[i] };

            // -a·s + e
            let temp = (neg_as as u128 + e_flat[i] as u128) % q as u128;

            // -a·s + e + s_k
            b[i] = ((temp + s_k[i] as u128) % q as u128) as u64;
        }

        Ok((a, b))
    }

    /// Get rotation key for a rotation step
    pub fn get_key_for_step(&self, step: i32) -> Option<&(Vec<u64>, Vec<u64>)> {
        let k = rotation_step_to_galois_element(step, self.n);
        self.keys.get(&k)
    }

    // Helper functions (apply_galois_to_secret_key, coeffs_to_flat_rns, etc.)
    // ... (implementation details)
}
```

### Phase 2: Metal GPU Rotation Operation

**File:** `src/clifford_fhe_v2/backends/gpu_metal/ckks.rs` (extend MetalCiphertext)

```rust
impl MetalCiphertext {
    /// Rotate ciphertext slots by r steps using Metal GPU
    ///
    /// Uses Galois automorphism σ_k where k = 5^r (mod 2N).
    /// Requires rotation keys for the specified step.
    ///
    /// # Algorithm
    /// 1. Apply σ_k to c₀ and c₁ (GPU kernel)
    /// 2. Key switch c₁ using rotation key (GPU NTT)
    /// 3. Return rotated ciphertext
    pub fn rotate_by_steps(
        &self,
        step: i32,
        rot_keys: &MetalRotationKeys,
        ctx: &MetalCkksContext,
    ) -> Result<Self, String> {
        let n = self.n;
        let num_primes = self.num_primes;
        let moduli = &ctx.params.moduli[..=self.level];

        // Get rotation key
        let (a_k, b_k) = rot_keys.get_key_for_step(step)
            .ok_or_else(|| format!("Rotation key for step {} not found", step))?;

        // Convert step to Galois element
        let k = rotation_step_to_galois_element(step, n);

        // Precompute Galois map
        let (galois_map, galois_signs) = compute_galois_map(n, k);

        // Apply Galois automorphism to c₀ and c₁ (GPU)
        let c0_rotated = self.apply_galois_gpu(&self.c0, &galois_map, &galois_signs, moduli, ctx)?;
        let c1_rotated = self.apply_galois_gpu(&self.c1, &galois_map, &galois_signs, moduli, ctx)?;

        // Key switch c₁_rotated (GPU NTT multiplication)
        let c1_switched = self.key_switch_gpu(&c1_rotated, a_k, b_k, moduli, ctx)?;

        Ok(Self {
            c0: c0_rotated,
            c1: c1_switched,
            n,
            num_primes,
            level: self.level,
            scale: self.scale,
        })
    }

    /// Apply Galois automorphism using Metal GPU kernel
    fn apply_galois_gpu(
        &self,
        poly: &[u64],
        galois_map: &[u32],
        galois_signs: &[i32],
        moduli: &[u64],
        ctx: &MetalCkksContext,
    ) -> Result<Vec<u64>, String> {
        // Create Metal buffers
        let device = ctx.device.device();
        let input_buffer = ctx.device.create_buffer_with_data(poly);
        let output_buffer = ctx.device.create_buffer(poly.len() * 8);
        let map_buffer = ctx.device.create_buffer_with_data(galois_map);
        let signs_buffer = ctx.device.create_buffer_with_data(galois_signs);
        let moduli_buffer = ctx.device.create_buffer_with_data(moduli);

        // Dispatch GPU kernel
        let pipeline = ctx.device.get_rotation_pipeline()?;
        let command_buffer = ctx.device.new_command_buffer();
        let encoder = command_buffer.compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, &input_buffer);
        encoder.set_buffer(1, &output_buffer);
        encoder.set_buffer(2, &map_buffer);
        encoder.set_buffer(3, &signs_buffer);
        encoder.set_u32(4, self.n as u32);
        encoder.set_u32(5, self.num_primes as u32);
        encoder.set_buffer(6, &moduli_buffer);

        // Dispatch threads: one per coefficient
        let thread_group_size = 256;
        let thread_groups = (self.n + thread_group_size - 1) / thread_group_size;
        encoder.dispatch_threads(self.n, thread_group_size);

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read back result
        Ok(ctx.device.read_buffer(&output_buffer, poly.len()))
    }

    /// Key switch after rotation (multiply by rotation key)
    fn key_switch_gpu(
        &self,
        c1_rotated: &[u64],
        a_k: &[u64],
        b_k: &[u64],
        moduli: &[u64],
        ctx: &MetalCkksContext,
    ) -> Result<Vec<u64>, String> {
        // c1_new = c1_rotated · a_k + b_k (using NTT multiplication)
        let product = ctx.multiply_polys_flat_ntt_negacyclic(c1_rotated, a_k, moduli)?;

        // Add b_k
        let mut result = vec![0u64; product.len()];
        for i in 0..product.len() {
            let prime_idx = i % moduli.len();
            let q = moduli[prime_idx];
            result[i] = ((product[i] as u128 + b_k[i] as u128) % q as u128) as u64;
        }

        Ok(result)
    }
}
```

---

## Implementation Plan

### Week 1-2: Core Rotation Infrastructure
- [ ] Create `rotation.metal` shader with Galois automorphism kernel
- [ ] Implement `rotation.rs` with galois_map precomputation
- [ ] Add rotation pipeline to `MetalDevice`
- [ ] Unit tests for Galois map correctness

### Week 2-3: Rotation Keys
- [ ] Implement `MetalRotationKeys` structure
- [ ] Add rotation key generation to `MetalKeyContext`
- [ ] Gadget decomposition for key switching
- [ ] Tests: verify rotation keys decrypt correctly

### Week 3-4: Rotation Operation
- [ ] Extend `MetalCiphertext` with `rotate_by_steps`
- [ ] Implement `apply_galois_gpu` using Metal kernel
- [ ] Implement `key_switch_gpu` using NTT multiplication
- [ ] Integration tests: encrypt → rotate → decrypt

### Week 4-5: CoeffToSlot/SlotToCoeff GPU Port
- [ ] Port V3 `coeff_to_slot.rs` to use Metal ciphertexts
- [ ] Port V3 `slot_to_coeff.rs` to use Metal ciphertexts
- [ ] Ensure all 24 rotations run on GPU (no CPU conversion!)
- [ ] Correctness tests against CPU bootstrap

### Week 5-6: EvalMod GPU Port
- [ ] Port V3 `eval_mod.rs` to use Metal operations
- [ ] GPU `multiply_plain` already exists - verify compatibility
- [ ] Add GPU `add` operation if missing
- [ ] End-to-end bootstrap test (GPU-only)

### Week 6-7: V3 Integration
- [ ] Create V3 Metal backend: `src/clifford_fhe_v3/backends/gpu_metal/`
- [ ] Bootstrap pipeline using Metal operations
- [ ] SIMD batching integration (512 samples in parallel)
- [ ] Performance benchmarking

### Week 7-8: Testing & Optimization
- [ ] Comprehensive correctness tests
- [ ] Performance profiling (Metal Instruments)
- [ ] Optimize GPU kernel occupancy
- [ ] Documentation and examples

---

## Performance Targets

**Current (CPU):**
- Bootstrap: 360s per batch (N=1024)
- With SIMD: 0.7s per sample (512 samples)

**Projected (Metal GPU):**
- Bootstrap: 30s per batch (12× faster)
- With SIMD: 0.06s per sample (60ms)
- Rotation: <1ms per rotation (vs ~15ms CPU)

**Breakdown:**
- NTT operations: ~10× speedup (already validated in V2)
- Rotation: ~15× speedup (GPU parallelism + zero-copy buffers)
- CoeffToSlot/SlotToCoeff: ~12× speedup (all 24 rotations on GPU)

---

## Testing Strategy

### Unit Tests
1. **Galois Map:** Verify permutation correctness for various k
2. **Rotation Keys:** Ensure σ_k(s) is correctly embedded
3. **Single Rotation:** encrypt(m) → rotate(1) → decrypt = rotated_m
4. **Conjugation:** rotate(-1) should conjugate complex slots

### Integration Tests
1. **CoeffToSlot:** Match CPU implementation exactly
2. **SlotToCoeff:** Match CPU implementation exactly
3. **Full Bootstrap:** GPU bootstrap = CPU bootstrap (within error threshold)

### Performance Tests
1. **Rotation Benchmark:** Measure single rotation time
2. **Bootstrap Benchmark:** Full pipeline timing
3. **SIMD Batching:** Verify 512× throughput

---

## Risk Mitigation

**Risk 1: Rotation Key Size**
- 24 rotations × (a, b) × N × num_primes × 8 bytes = ~10 MB for N=1024, 41 primes
- **Mitigation:** Store on GPU once, reuse across batched operations

**Risk 2: Key Switching Precision**
- Error growth from gadget decomposition
- **Mitigation:** Use base 2^20 (same as evaluation keys), validate noise budget

**Risk 3: GPU Memory Constraints**
- SIMD batching 512 samples × 10 MB keys = 5 GB
- **Mitigation:** Unified memory architecture on Apple Silicon (zero-copy), batch size tuning

---

## Success Criteria

✅ **Phase 1 Complete:** Single rotation works on Metal GPU ciphertext
✅ **Phase 2 Complete:** CoeffToSlot/SlotToCoeff run entirely on GPU
✅ **Phase 3 Complete:** Full V3 bootstrap runs on GPU (no CPU conversion)
✅ **Phase 4 Complete:** 12× speedup validated in benchmarks
✅ **Phase 5 Complete:** SIMD batching achieves <100ms per sample

---

## Next Steps

**Immediate (Week 1):**
1. Create `rotation.metal` shader
2. Implement `compute_galois_map` in `rotation.rs`
3. Add rotation pipeline to `MetalDevice`
4. Write unit tests for Galois automorphism

**After Phase 1:**
- Generate rotation keys on GPU
- Implement `rotate_by_steps` for `MetalCiphertext`
- Validate against test vectors

**Final Goal:**
- Full Metal GPU V3 bootstrap: <30s per batch
- SIMD batching: 60ms per sample (512× throughput)
- Zero CPU conversion overhead
