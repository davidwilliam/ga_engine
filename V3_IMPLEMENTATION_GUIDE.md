# V3 Bootstrapping Implementation Guide

## Quick Start

This guide provides concrete code templates for implementing CKKS bootstrapping in V3.

---

## Phase 1: CPU Bootstrap Foundation

### Step 1: Create V3 Module Structure

```bash
mkdir -p src/clifford_fhe_v3
mkdir -p src/clifford_fhe_v3/bootstrapping
mkdir -p src/clifford_fhe_v3/backends/cpu_optimized
mkdir -p src/clifford_fhe_v3/tests
```

### Step 2: Module Files

**`src/clifford_fhe_v3/mod.rs`:**
```rust
//! Clifford FHE V3 - Bootstrapping and Deep Computation
//!
//! V3 adds CKKS bootstrapping to enable unlimited multiplication depth.

pub mod bootstrapping;

#[cfg(test)]
mod tests;
```

**`src/clifford_fhe_v3/bootstrapping/mod.rs`:**
```rust
//! CKKS Bootstrapping Module
//!
//! Implements homomorphic noise refresh to enable deep computation.

mod bootstrap_context;
mod mod_raise;
mod coeff_to_slot;
mod eval_mod;
mod slot_to_coeff;
mod sin_approx;
mod keys;

pub use bootstrap_context::{BootstrapContext, BootstrapParams};
```

### Step 3: Bootstrap Context Skeleton

**`src/clifford_fhe_v3/bootstrapping/bootstrap_context.rs`:**
```rust
//! Bootstrap Context - Main bootstrap API

use crate::clifford_fhe_v2::params::CliffordFHEParams;
use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::{Ciphertext, SecretKey};

/// Bootstrap parameters
#[derive(Clone, Debug)]
pub struct BootstrapParams {
    /// Degree of sine polynomial approximation (15-31)
    pub sin_degree: usize,
    /// Number of levels reserved for bootstrap
    pub bootstrap_levels: usize,
    /// Target precision after bootstrap
    pub target_precision: f64,
}

impl BootstrapParams {
    /// Balanced bootstrap parameters (recommended)
    pub fn balanced() -> Self {
        BootstrapParams {
            sin_degree: 23,
            bootstrap_levels: 12,
            target_precision: 1e-4,
        }
    }

    /// Conservative bootstrap parameters (high precision)
    pub fn conservative() -> Self {
        BootstrapParams {
            sin_degree: 31,
            bootstrap_levels: 15,
            target_precision: 1e-6,
        }
    }

    /// Fast bootstrap parameters (lower precision)
    pub fn fast() -> Self {
        BootstrapParams {
            sin_degree: 15,
            bootstrap_levels: 10,
            target_precision: 1e-2,
        }
    }
}

/// Bootstrap context for CKKS bootstrapping
pub struct BootstrapContext {
    params: CliffordFHEParams,
    bootstrap_params: BootstrapParams,
    // TODO: Add rotation keys
    // TODO: Add sine polynomial coefficients
}

impl BootstrapContext {
    /// Create new bootstrap context
    pub fn new(
        params: CliffordFHEParams,
        bootstrap_params: BootstrapParams,
        secret_key: &SecretKey,
    ) -> Result<Self, String> {
        // TODO: Generate rotation keys
        // TODO: Precompute sine polynomial coefficients

        println!("Creating bootstrap context:");
        println!("  Sine degree: {}", bootstrap_params.sin_degree);
        println!("  Bootstrap levels: {}", bootstrap_params.bootstrap_levels);
        println!("  Target precision: {}", bootstrap_params.target_precision);

        Ok(BootstrapContext {
            params,
            bootstrap_params,
        })
    }

    /// Bootstrap a ciphertext (refresh noise)
    pub fn bootstrap(&self, ct: &Ciphertext) -> Result<Ciphertext, String> {
        println!("Starting bootstrap pipeline...");

        // Step 1: ModRaise - raise modulus to higher level
        println!("  [1/4] ModRaise...");
        let ct_raised = self.mod_raise(ct)?;

        // Step 2: CoeffToSlot - transform to evaluation form
        println!("  [2/4] CoeffToSlot...");
        let ct_slots = self.coeff_to_slot(&ct_raised)?;

        // Step 3: EvalMod - homomorphically evaluate modular reduction
        println!("  [3/4] EvalMod...");
        let ct_eval = self.eval_mod(&ct_slots)?;

        // Step 4: SlotToCoeff - transform back to coefficient form
        println!("  [4/4] SlotToCoeff...");
        let ct_coeffs = self.slot_to_coeff(&ct_eval)?;

        println!("  Bootstrap complete!");

        Ok(ct_coeffs)
    }

    // Internal operations (to be implemented)
    fn mod_raise(&self, ct: &Ciphertext) -> Result<Ciphertext, String> {
        Err("ModRaise not yet implemented".to_string())
    }

    fn coeff_to_slot(&self, ct: &Ciphertext) -> Result<Ciphertext, String> {
        Err("CoeffToSlot not yet implemented".to_string())
    }

    fn eval_mod(&self, ct: &Ciphertext) -> Result<Ciphertext, String> {
        Err("EvalMod not yet implemented".to_string())
    }

    fn slot_to_coeff(&self, ct: &Ciphertext) -> Result<Ciphertext, String> {
        Err("SlotToCoeff not yet implemented".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_context_creation() {
        use crate::clifford_fhe_v2::params::CliffordFHEParams;
        use crate::clifford_fhe_v2::backends::cpu_optimized::key_generation::KeyContext;

        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (_, secret_key, _) = key_ctx.keygen();

        let bootstrap_params = BootstrapParams::balanced();
        let bootstrap_ctx = BootstrapContext::new(params, bootstrap_params, &secret_key);

        assert!(bootstrap_ctx.is_ok(), "Bootstrap context creation failed");
    }
}
```

### Step 4: Modulus Raising (Simplest Component)

**`src/clifford_fhe_v3/bootstrapping/mod_raise.rs`:**
```rust
//! Modulus Raising
//!
//! Raises ciphertext to higher modulus level to create working room for bootstrap.

use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::Ciphertext;
use crate::clifford_fhe_v2::rns::RnsRepresentation;

/// Raise ciphertext modulus to higher level
///
/// This scales up the ciphertext coefficients to work with a larger modulus chain,
/// preserving the plaintext value.
pub fn mod_raise(
    ct: &Ciphertext,
    target_moduli: &[u64],
) -> Result<Ciphertext, String> {
    // Current moduli
    let current_moduli = &ct.c0[0].moduli;
    let n = ct.c0.len();

    if target_moduli.len() <= current_moduli.len() {
        return Err(format!(
            "Target moduli count ({}) must be larger than current ({})",
            target_moduli.len(),
            current_moduli.len()
        ));
    }

    // Scale c0 to higher modulus
    let mut c0_raised = Vec::with_capacity(n);
    for rns in &ct.c0 {
        c0_raised.push(scale_rns_to_higher_modulus(rns, current_moduli, target_moduli));
    }

    // Scale c1 to higher modulus
    let mut c1_raised = Vec::with_capacity(n);
    for rns in &ct.c1 {
        c1_raised.push(scale_rns_to_higher_modulus(rns, current_moduli, target_moduli));
    }

    Ok(Ciphertext {
        c0: c0_raised,
        c1: c1_raised,
        level: ct.level,
        scale: ct.scale,
    })
}

/// Scale a single RNS representation to higher modulus
fn scale_rns_to_higher_modulus(
    rns: &RnsRepresentation,
    old_moduli: &[u64],
    new_moduli: &[u64],
) -> RnsRepresentation {
    // Use Chinese Remainder Theorem (CRT) to reconstruct value
    // Then re-represent in new moduli basis

    let mut new_residues = Vec::with_capacity(new_moduli.len());

    // Reconstruct integer using CRT
    let value = crt_reconstruct(&rns.residues, old_moduli);

    // Re-represent in new moduli
    for &q in new_moduli {
        new_residues.push(value % q);
    }

    RnsRepresentation::new(new_residues, new_moduli.to_vec())
}

/// Reconstruct integer from RNS representation using CRT
///
/// For small values only (won't work for full-size ciphertext coefficients).
/// Need more sophisticated approach for production.
fn crt_reconstruct(residues: &[u64], moduli: &[u64]) -> u64 {
    // Simplified CRT for demonstration
    // TODO: Implement proper multi-precision CRT reconstruction

    // For now, just return first residue as placeholder
    residues[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clifford_fhe_v2::params::CliffordFHEParams;
    use crate::clifford_fhe_v2::backends::cpu_optimized::key_generation::KeyContext;
    use crate::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;

    #[test]
    fn test_mod_raise_preserves_plaintext() {
        let params = CliffordFHEParams::new_test_ntt_1024();
        let key_ctx = KeyContext::new(params.clone());
        let (public_key, secret_key, _) = key_ctx.keygen();
        let ckks_ctx = CkksContext::new(params.clone());

        // Encrypt a plaintext
        let plaintext = vec![1.0, 2.0, 3.0, 4.0];
        let ct = ckks_ctx.encrypt(&plaintext, &public_key);

        // Define higher modulus chain (add more primes)
        let target_moduli = vec![
            params.moduli[0],
            params.moduli[1],
            params.moduli[2],
            1152921504606584833,  // Additional prime
            1152921504606584777,  // Additional prime
        ];

        // Raise modulus
        let ct_raised = mod_raise(&ct, &target_moduli).unwrap();

        // Decrypt - should get same plaintext
        let decrypted = ckks_ctx.decrypt(&ct_raised, &secret_key);

        // Check accuracy
        for i in 0..plaintext.len() {
            let error = (plaintext[i] - decrypted[i]).abs();
            assert!(error < 0.01, "ModRaise changed plaintext");
        }
    }
}
```

### Step 5: Sine Approximation

**`src/clifford_fhe_v3/bootstrapping/sin_approx.rs`:**
```rust
//! Sine Polynomial Approximation
//!
//! Computes Chebyshev/Taylor polynomial coefficients for sine function.

use std::f64::consts::PI;

/// Compute Chebyshev polynomial coefficients for sin(x) on [-π, π]
///
/// Returns coefficients [c0, c1, c2, ..., c_degree]
pub fn chebyshev_sin_coeffs(degree: usize) -> Vec<f64> {
    assert!(degree >= 5, "Need at least degree 5 for reasonable accuracy");
    assert!(degree % 2 == 1, "Sine is odd function, use odd degree");

    // For now, use Taylor series coefficients
    // TODO: Implement proper Chebyshev approximation (better than Taylor)
    taylor_sin_coeffs(degree)
}

/// Compute Taylor series coefficients for sin(x)
///
/// sin(x) = x - x³/3! + x⁵/5! - x⁷/7! + ...
pub fn taylor_sin_coeffs(degree: usize) -> Vec<f64> {
    let mut coeffs = vec![0.0; degree + 1];

    // sin(x) has only odd powers
    for k in 0..=(degree / 2) {
        let power = 2 * k + 1;
        if power <= degree {
            // Coefficient for x^power is (-1)^k / (2k+1)!
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            let factorial = factorial(power);
            coeffs[power] = sign / factorial;
        }
    }

    coeffs
}

/// Compute factorial
fn factorial(n: usize) -> f64 {
    if n <= 1 {
        1.0
    } else {
        (2..=n).fold(1.0, |acc, x| acc * x as f64)
    }
}

/// Evaluate polynomial with given coefficients
///
/// p(x) = c0 + c1*x + c2*x² + ... + cn*x^n
pub fn eval_polynomial(coeffs: &[f64], x: f64) -> f64 {
    // Use Horner's method for numerical stability
    let mut result = 0.0;
    for &coeff in coeffs.iter().rev() {
        result = result * x + coeff;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_taylor_sin_accuracy() {
        let coeffs = taylor_sin_coeffs(31);

        // Test on [-π, π]
        let test_points = vec![0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI];

        for x in test_points {
            let approx = eval_polynomial(&coeffs, x);
            let exact = x.sin();
            let error = (approx - exact).abs();

            println!("sin({:.4}) = {:.6} (approx: {:.6}, error: {:.6})",
                     x, exact, approx, error);

            assert!(error < 1e-6, "Taylor approximation error too large");
        }
    }

    #[test]
    fn test_chebyshev_sin_coeffs() {
        let coeffs = chebyshev_sin_coeffs(15);
        assert_eq!(coeffs.len(), 16);

        // Sine has no even powers
        for k in 0..coeffs.len() {
            if k % 2 == 0 && k > 0 {
                assert_eq!(coeffs[k], 0.0, "Even coefficient should be zero");
            }
        }
    }
}
```

### Step 6: First Test Example

**`examples/test_v3_bootstrap_skeleton.rs`:**
```rust
//! Test V3 Bootstrap Skeleton
//!
//! Verifies that V3 module structure compiles and basic operations work.

use ga_engine::clifford_fhe_v2::params::CliffordFHEParams;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::key_generation::KeyContext;
use ga_engine::clifford_fhe_v2::backends::cpu_optimized::ckks::CkksContext;
use ga_engine::clifford_fhe_v3::bootstrapping::{BootstrapContext, BootstrapParams};

fn main() {
    println!("=== V3 Bootstrap Skeleton Test ===\n");

    // 1. Setup parameters
    println!("Phase 1: Setting up parameters...");
    let params = CliffordFHEParams::new_test_ntt_1024();
    println!("  N = {}", params.n);
    println!("  {} primes", params.moduli.len());

    // 2. Generate keys
    println!("\nPhase 2: Generating keys...");
    let key_ctx = KeyContext::new(params.clone());
    let (public_key, secret_key, _) = key_ctx.keygen();
    println!("  ✓ Keys generated");

    // 3. Create bootstrap context
    println!("\nPhase 3: Creating bootstrap context...");
    let bootstrap_params = BootstrapParams::balanced();
    let bootstrap_ctx = BootstrapContext::new(
        params.clone(),
        bootstrap_params,
        &secret_key,
    ).unwrap();
    println!("  ✓ Bootstrap context created");

    // 4. Test encryption/decryption (without bootstrap)
    println!("\nPhase 4: Testing basic encryption...");
    let ckks_ctx = CkksContext::new(params.clone());
    let plaintext = vec![1.0, 2.0, 3.0, 4.0];
    let ct = ckks_ctx.encrypt(&plaintext, &public_key);
    let decrypted = ckks_ctx.decrypt(&ct, &secret_key);

    println!("  Original: {:?}", plaintext);
    println!("  Decrypted: {:?}", decrypted);

    let mut max_error = 0.0;
    for i in 0..plaintext.len() {
        let error = (plaintext[i] - decrypted[i]).abs();
        max_error = max_error.max(error);
    }
    println!("  Max error: {:.6}", max_error);

    if max_error < 0.01 {
        println!("  ✓ Encryption working correctly");
    }

    // 5. Test bootstrap (will fail until implemented)
    println!("\nPhase 5: Testing bootstrap...");
    match bootstrap_ctx.bootstrap(&ct) {
        Ok(_) => {
            println!("  ✓ Bootstrap succeeded!");
        }
        Err(e) => {
            println!("  ⚠️  Bootstrap not yet implemented: {}", e);
            println!("  (This is expected - bootstrap components not implemented yet)");
        }
    }

    println!("\n=== V3 Bootstrap Skeleton Test Complete ===");
    println!("Status: Module structure working, ready for component implementation");
}
```

---

## Building and Testing

### Add V3 to lib.rs

**`src/lib.rs`:**
```rust
pub mod clifford_fhe_v1;
pub mod clifford_fhe_v2;
pub mod clifford_fhe_v3;  // Add V3 module
pub mod medical_imaging;
pub mod params;
pub mod rns;
```

### Build Commands

```bash
# Build V3 module
cargo build --release

# Run skeleton test
cargo run --release --example test_v3_bootstrap_skeleton

# Run unit tests
cargo test --lib clifford_fhe_v3

# Run specific test
cargo test --lib clifford_fhe_v3::bootstrapping::sin_approx::tests::test_taylor_sin_accuracy
```

---

## Implementation Checklist

### Phase 1: Foundation ✅
- [x] Create module structure
- [x] Implement `BootstrapContext` skeleton
- [x] Implement `BootstrapParams`
- [ ] Implement `mod_raise()` (in progress)
- [ ] Implement `sin_approx.rs` (in progress)
- [ ] Write unit tests
- [ ] Create example demonstrating skeleton

### Phase 2: CoeffToSlot/SlotToCoeff (Next)
- [ ] Implement rotation key generation
- [ ] Implement basic rotation operations
- [ ] Implement CoeffToSlot transformation
- [ ] Implement SlotToCoeff transformation
- [ ] Test that transformations compose to identity

### Phase 3: EvalMod (Core Challenge)
- [ ] Integrate sine approximation
- [ ] Implement homomorphic polynomial evaluation
- [ ] Implement EvalMod using sine
- [ ] Test accuracy of modular reduction
- [ ] Tune polynomial degree

### Phase 4: Integration
- [ ] Complete `bootstrap()` pipeline
- [ ] Test on encrypted multivectors
- [ ] Implement `bootstrap_multivector()`
- [ ] Test noise refresh
- [ ] Create benchmarks

---

## Key Design Decisions

### 1. CRT Reconstruction for ModRaise
**Challenge:** Need to convert between RNS bases with different moduli

**Options:**
- **Option A (Simple):** Use multi-precision integers (num-bigint crate)
- **Option B (Fast):** Implement fast basis extension (Bajard et al.)
- **Recommendation:** Start with Option A, optimize to Option B later

### 2. Sine Approximation Method
**Challenge:** Need good approximation of sin(x) as polynomial

**Options:**
- **Option A (Simple):** Taylor series (easy to implement)
- **Option B (Better):** Chebyshev polynomials (better accuracy)
- **Option C (Best):** Remez algorithm (optimal minimax)
- **Recommendation:** Start with Taylor (Option A), move to Chebyshev (Option B)

### 3. Rotation Strategy
**Challenge:** CoeffToSlot needs O(log N) rotations

**Options:**
- **Option A (Standard):** Baby-step giant-step (BSGS) algorithm
- **Option B (Sparse):** Sparse FFT (fewer rotations)
- **Recommendation:** Start with standard BSGS (Option A)

---

## Next Session

Ready to start implementing:

1. **Complete `mod_raise.rs`** with proper CRT reconstruction
2. **Complete `sin_approx.rs`** with test coverage
3. **Create rotation key generation** in `keys.rs`
4. **Begin `coeff_to_slot.rs`** implementation

**Goal:** Have working ModRaise and sine approximation by end of next session.
