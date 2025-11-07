# Metal NTT Montgomery Multiplication Bug Report

## ✅ RESOLVED

**Root Cause**: Double Montgomery conversion in NTT pipeline
**Solution**: Keep values in Montgomery domain between forward() and inverse()
**Status**: FIXED - All tests passing with perfect accuracy

See [METAL_GPU_CKKS_COMPLETE.md](METAL_GPU_CKKS_COMPLETE.md) for full resolution details.

---

## Executive Summary (Historical)

The Metal GPU NTT implementation had a critical bug in polynomial multiplication that produced incorrect results with a large constant scaling factor (~2^56). Basic NTT forward/inverse transformations worked perfectly, but when used for polynomial multiplication (forward → pointwise multiply → inverse), the results were wrong by a consistent multiplicative factor.

**Impact**: This bug blocked the complete Metal GPU CKKS implementation, preventing homomorphic encryption operations from working correctly.

**Resolution**: Fixed by removing unnecessary Montgomery domain conversions.

---

## Problem Description

### What Works ✅
- **Forward NTT**: Correctly transforms polynomial coefficients to NTT domain (Montgomery representation)
- **Inverse NTT**: Correctly transforms NTT domain back to polynomial coefficients
- **NTT Roundtrip**: `inverse(forward(x)) == x` passes with PERFECT match (zero errors)

### What Fails ❌
- **Polynomial Multiplication via NTT**: Produces results with incorrect scaling
  - Expected: `(1 + X) * (2 + X) = 2 + 3X + X^2`
  - Got: `141863379672171008 + 212795069508256512·X + 70931689836085504·X^2`
  - **Observation**: The ratios are correct (2:3:1), but absolute values are scaled by ~70931689836085504

---

## Test Case - Polynomial Multiplication

### Setup
```rust
// Parameters
let n = 1024;
let q = 1152921504606584833u64;  // 60-bit NTT-friendly prime
let psi = 693807653563943717u64; // Primitive 2048-th root of unity

// Polynomials
let a = [1, 1, 0, 0, ...];  // 1 + X
let b = [2, 1, 0, 0, ...];  // 2 + X

// Expected result
// (1 + X) * (2 + X) = 2 + 3X + X^2
```

### Algorithm
```rust
// 1. Forward NTT
ntt_ctx.forward(&mut a);  // a → NTT domain (Montgomery)
ntt_ctx.forward(&mut b);  // b → NTT domain (Montgomery)

// 2. Pointwise multiplication
ntt_ctx.pointwise_multiply(&a, &b, &mut result);

// 3. Inverse NTT
ntt_ctx.inverse(&mut result);  // result → coefficient domain
```

### Detailed Trace

```
Before forward NTT:
  a[0..3]: [1, 1, 0]
  b[0..3]: [2, 1, 0]

After forward NTT (in Montgomery domain):
  a_ntt[0..3]: [2, 643841514320604599, 100696515027518316]
  b_ntt[0..3]: [3, 643841514320604600, 100696515027518317]
  ✅ First value is sum of coefficients (correct!)

After pointwise multiply:
  result_ntt[0..3]: [425590139016513024, 497848329455151370, 267758538687779512]

After inverse NTT (back to normal domain):
  result[0..3]: [141863379672171008, 212795069508256512, 70931689836085504]
  ❌ Should be [2, 3, 1]
```

### Scaling Factor Analysis

```
Actual values / Expected values:
  141863379672171008 / 2 = 70931689836085504
  212795069508256512 / 3 = 70931689836085504
  70931689836085504  / 1 = 70931689836085504

Consistent scaling factor: 70931689836085504 ≈ 2^56.00009
```

The ratios are **exactly correct**, but there's an unwanted scaling factor.

---

## Code - Metal NTT Implementation

### 1. Montgomery Multiplication (Metal Shader)

**File**: `src/clifford_fhe_v2/backends/gpu_metal/shaders/ntt.metal`

```metal
/// Montgomery multiplication: (a * b) * R^{-1} mod q
///
/// Given inputs in Montgomery domain (a*R, b*R), computes:
///   (a*R) * (b*R) * R^{-1} = a*b*R (stays in Montgomery domain)
///
/// Algorithm (FIPS 186-4):
/// 1. Compute t = a * b (128-bit result)
/// 2. Compute m = (t mod R) * q_inv mod R
/// 3. Compute u = (t + m*q) / R
/// 4. If u >= q, return u - q, else return u
///
/// Where:
/// - R = 2^64 (Montgomery radix)
/// - q_inv = -q^{-1} mod R (precomputed)
inline ulong mont_mul(ulong a, ulong b, ulong q, ulong q_inv) {
    // Step 1: Compute t = a * b (as 128-bit value)
    ulong t_lo = a * b;          // Low 64 bits
    ulong t_hi = mulhi(a, b);    // High 64 bits

    // Step 2: Compute m = (t_lo * q_inv) mod 2^64
    // Since we're working mod 2^64, this is just multiplication
    ulong m = t_lo * q_inv;

    // Step 3: Compute mq = m * q (as 128-bit value)
    ulong mq_lo = m * q;
    ulong mq_hi = mulhi(m, q);

    // Step 4: Compute (t + mq) / 2^64
    // Add low parts to check for carry
    ulong carry = (t_lo > ~mq_lo) ? 1UL : 0UL;

    // High part is the result after division by 2^64
    ulong sum_hi = t_hi + mq_hi + carry;

    // Step 5: Final reduction
    if (sum_hi >= q) {
        sum_hi -= q;
    }

    return sum_hi;
}
```

### 2. Pointwise Multiplication Kernel

**File**: `src/clifford_fhe_v2/backends/gpu_metal/shaders/ntt.metal`

```metal
/// Pointwise modular multiplication in NTT domain (using Montgomery multiplication)
///
/// IMPORTANT: Input values a[] and b[] must be in Montgomery domain!
/// Output c[] is also in Montgomery domain.
kernel void ntt_pointwise_multiply(
    device const ulong* a [[buffer(0)]],
    device const ulong* b [[buffer(1)]],
    device ulong* c [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant ulong& q [[buffer(4)]],
    constant ulong& q_inv [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        // Montgomery multiplication: (a*R) * (b*R) * R^{-1} = (a*b)*R
        c[gid] = mont_mul(a[gid], b[gid], q, q_inv);
    }
}
```

### 3. Forward NTT (Rust)

**File**: `src/clifford_fhe_v2/backends/gpu_metal/ntt.rs`

```rust
pub fn forward(&self, coeffs: &mut [u64]) -> Result<(), String> {
    if coeffs.len() != self.n {
        return Err(format!("Expected {} coefficients, got {}", self.n, coeffs.len()));
    }

    // Convert input to Montgomery domain (CPU)
    let mut coeffs_montgomery: Vec<u64> = coeffs.iter()
        .map(|&c| Self::to_montgomery(c, self.r_squared, self.q, self.q_inv))
        .collect();

    let log_n = (self.n as f64).log2() as u32;

    // Create GPU buffers
    let coeffs_buffer = self.device.create_buffer_with_data(&coeffs_montgomery);
    let omega_powers_buffer = self.device.create_buffer_with_data(&self.omega_powers_montgomery);
    let n_buffer = self.device.create_buffer_with_u32_data(&[self.n as u32]);
    let q_buffer = self.device.create_buffer_with_data(&[self.q]);
    let q_inv_buffer = self.device.create_buffer_with_data(&[self.q_inv]);

    let threadgroup_size = MTLSize::new(256, 1, 1);

    // Execute butterfly stages (GPU)
    let kernel = self.device.get_function("ntt_forward_stage")?;
    let pipeline = self.device.device()
        .new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| format!("Failed to create forward stage pipeline: {:?}", e))?;

    for stage in 0..log_n {
        let stage_buffer = self.device.create_buffer_with_u32_data(&[stage]);

        self.device.execute_kernel(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&coeffs_buffer), 0);
            encoder.set_buffer(1, Some(&omega_powers_buffer), 0);
            encoder.set_buffer(2, Some(&n_buffer), 0);
            encoder.set_buffer(3, Some(&q_buffer), 0);
            encoder.set_buffer(4, Some(&stage_buffer), 0);
            encoder.set_buffer(5, Some(&q_inv_buffer), 0);

            let butterfly_threads = ((self.n / 2 + 255) / 256) as u64;
            let butterfly_threadgroups = MTLSize::new(butterfly_threads, 1, 1);
            encoder.dispatch_thread_groups(butterfly_threadgroups, threadgroup_size);
            Ok(())
        })?;
        // Implicit global barrier between stages
    }

    // Read result back (still in Montgomery domain)
    let result_montgomery = self.device.read_buffer(&coeffs_buffer, self.n);

    // Convert from Montgomery domain back to normal domain (CPU)
    for i in 0..self.n {
        coeffs[i] = Self::mont_mul_cpu(result_montgomery[i], 1, self.q, self.q_inv);
    }

    Ok(())
}
```

**Helper - Convert to Montgomery domain**:
```rust
fn to_montgomery(x: u64, r_squared: u64, q: u64, q_inv: u64) -> u64 {
    // To convert x to Montgomery: mont_mul(x, R^2 mod q, q, q_inv)
    // This gives: x * R^2 * R^{-1} = x * R
    Self::mont_mul_cpu(x, r_squared, q, q_inv)
}
```

### 4. Inverse NTT (Rust)

**File**: `src/clifford_fhe_v2/backends/gpu_metal/ntt.rs`

```rust
pub fn inverse(&self, evals: &mut [u64]) -> Result<(), String> {
    if evals.len() != self.n {
        return Err(format!("Expected {} evaluation points, got {}", self.n, evals.len()));
    }

    // Convert input to Montgomery domain (CPU)
    let mut evals_montgomery: Vec<u64> = evals.iter()
        .map(|&e| Self::to_montgomery(e, self.r_squared, self.q, self.q_inv))
        .collect();

    let log_n = (self.n as f64).log2() as u32;

    // Create GPU buffers
    let evals_buffer = self.device.create_buffer_with_data(&evals_montgomery);
    let omega_inv_powers_buffer = self.device.create_buffer_with_data(&self.omega_inv_powers_montgomery);
    let n_buffer = self.device.create_buffer_with_u32_data(&[self.n as u32]);
    let q_buffer = self.device.create_buffer_with_data(&[self.q]);
    let n_inv_buffer = self.device.create_buffer_with_data(&[self.n_inv_montgomery]);
    let q_inv_buffer = self.device.create_buffer_with_data(&[self.q_inv]);

    let threadgroup_size = MTLSize::new(256, 1, 1);

    // Execute inverse butterfly stages (GPU, reverse order)
    let kernel = self.device.get_function("ntt_inverse_stage")?;
    let pipeline = self.device.device()
        .new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| format!("Failed to create inverse stage pipeline: {:?}", e))?;

    for stage in (0..log_n).rev() {
        let stage_buffer = self.device.create_buffer_with_u32_data(&[stage]);

        self.device.execute_kernel(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&evals_buffer), 0);
            encoder.set_buffer(1, Some(&omega_inv_powers_buffer), 0);
            encoder.set_buffer(2, Some(&n_buffer), 0);
            encoder.set_buffer(3, Some(&q_buffer), 0);
            encoder.set_buffer(4, Some(&stage_buffer), 0);
            encoder.set_buffer(5, Some(&q_inv_buffer), 0);

            let butterfly_threads = ((self.n / 2 + 255) / 256) as u64;
            let butterfly_threadgroups = MTLSize::new(butterfly_threads, 1, 1);
            encoder.dispatch_thread_groups(butterfly_threadgroups, threadgroup_size);
            Ok(())
        })?;
        // Implicit global barrier between stages
    }

    // Bit-reversal and scaling by n^{-1} (GPU)
    {
        let kernel = self.device.get_function("ntt_inverse_final_scale")?;
        let pipeline = self.device.device()
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create final scale pipeline: {:?}", e))?;

        self.device.execute_kernel(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&evals_buffer), 0);
            encoder.set_buffer(1, Some(&n_buffer), 0);
            encoder.set_buffer(2, Some(&q_buffer), 0);
            encoder.set_buffer(3, Some(&n_inv_buffer), 0);
            encoder.set_buffer(4, Some(&q_inv_buffer), 0);

            let threadgroups = MTLSize::new(((self.n + 255) / 256) as u64, 1, 1);
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
            Ok(())
        })?;
    }

    // Read result back (still in Montgomery domain)
    let result_montgomery = self.device.read_buffer(&evals_buffer, self.n);

    // Convert from Montgomery domain to normal domain (CPU)
    for i in 0..self.n {
        evals[i] = Self::mont_mul_cpu(result_montgomery[i], 1, self.q, self.q_inv);
    }

    Ok(())
}
```

### 5. Inverse NTT Final Scale Kernel

**File**: `src/clifford_fhe_v2/backends/gpu_metal/shaders/ntt.metal`

```metal
/// Final step of inverse NTT: bit-reversal + scaling by n^{-1}
///
/// After inverse butterfly stages, coefficients are:
/// 1. In bit-reversed order
/// 2. Scaled by n (need to multiply by n^{-1})
/// 3. In Montgomery domain
kernel void ntt_inverse_final_scale(
    device ulong* coeffs [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    constant ulong& q [[buffer(2)]],
    constant ulong& n_inv [[buffer(3)]],  // n^{-1} in Montgomery domain
    constant ulong& q_inv [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        // Bit-reverse index
        uint rev = bit_reverse(gid, n);

        // Scale by n^{-1} using Montgomery multiplication
        // coeffs[rev] is in Montgomery domain
        // n_inv is n^{-1} * R (Montgomery form)
        // Result: (coeffs * R) * (n^{-1} * R) * R^{-1} = coeffs * n^{-1} * R
        coeffs[gid] = mont_mul(coeffs[rev], n_inv, q, q_inv);
    }
}
```

### 6. Montgomery Parameters Computation

**File**: `src/clifford_fhe_v2/backends/gpu_metal/ntt.rs`

```rust
impl MetalNttContext {
    pub fn new_with_device(
        device: std::sync::Arc<MetalDevice>,
        n: usize,
        q: u64,
        psi: u64,
    ) -> Result<Self, String> {
        // ... validation code ...

        // Compute Montgomery multiplication parameters
        println!("    [NTT] Computing q_inv...");
        let q_inv = Self::compute_q_inv(q);

        println!("    [NTT] Computing R^2 mod q...");
        let r_squared = Self::compute_r_squared(q);

        // ... rest of initialization ...

        Ok(Self {
            device,
            n,
            q,
            psi,
            psi_powers,
            psi_inv_powers,
            omega_powers,
            omega_inv_powers,
            n_inv,
            n_inv_montgomery,
            q_inv,
            r_squared,
            omega_powers_montgomery,
            omega_inv_powers_montgomery,
        })
    }

    /// Compute -q^{-1} mod 2^64 using Newton's method (Hensel lifting)
    fn compute_q_inv(q: u64) -> u64 {
        assert!(q & 1 == 1, "q must be odd for Montgomery multiplication");

        // Start with 3-bit approximation
        let mut q_inv = q;

        // Newton iteration: x_{i+1} = x_i * (2 - q * x_i)
        // Doubles precision each iteration
        for _ in 0..5 {
            q_inv = q_inv.wrapping_mul(2u64.wrapping_sub(q.wrapping_mul(q_inv)));
        }

        // Return -q^{-1} mod 2^64
        q_inv.wrapping_neg()
    }

    /// Compute R^2 mod q where R = 2^64
    ///
    /// Uses repeated squaring and reduction to avoid overflow
    fn compute_r_squared(q: u64) -> u64 {
        // We want to compute 2^128 mod q
        // Strategy: 2^128 = (2^64)^2 mod q

        // First compute 2^64 mod q
        let r_mod_q = if q > (1u64 << 63) {
            // For large q close to 2^64, use direct computation
            // 2^64 mod q = 2^64 - q (since q < 2^64)
            (1u128 << 64) % (q as u128)
        } else {
            // For smaller q, 2^64 mod q can overflow u64, use u128
            ((1u128 << 64) % (q as u128)) as u64
        } as u64;

        // Now compute (2^64 mod q)^2 mod q = 2^128 mod q
        let r_squared_wide = (r_mod_q as u128) * (r_mod_q as u128);
        (r_squared_wide % (q as u128)) as u64
    }
}
```

---

## Hypothesis

The bug likely stems from one of these areas:

1. **Double Montgomery conversion**: The values might be getting converted to/from Montgomery domain incorrectly, causing an extra factor of R or R^{-1}

2. **R^2 calculation error**: The `compute_r_squared()` function might be computing the wrong value for large 60-bit primes

3. **Scaling in inverse NTT**: The final scaling by n^{-1} might be interacting incorrectly with Montgomery domain

4. **Pointwise multiply domain mismatch**: Though inputs are supposed to be in Montgomery domain, perhaps there's a mismatch in expectations

---

## Verification Tests

### Test 1: NTT Roundtrip (✅ PASSES)

```bash
cargo run --release --features v2,v2-gpu-metal --example test_metal_ntt_correctness
```

**Result**: PERFECT MATCH (zero errors)
- Tests: `inverse(forward(x)) == x`
- This confirms forward and inverse NTT are correct in isolation

### Test 2: Polynomial Multiplication (❌ FAILS)

```bash
cargo run --release --features v2,v2-gpu-metal --example test_metal_ntt_poly_mult
```

**Result**: Wrong by constant factor ~2^56
- Tests: `(1 + X) * (2 + X) = 2 + 3X + X^2`
- Got correct ratios but wrong absolute values

### Test 3: Full CKKS Pipeline (❌ FAILS due to Test 2)

```bash
cargo run --release --features v2,v2-gpu-metal --example test_metal_ckks_simple
```

**Result**: Huge decryption errors (~10^7) because polynomial multiplication is broken

---

## Request for Expert Analysis

**Questions**:

1. Is the Montgomery multiplication algorithm (`mont_mul`) implemented correctly?
2. Is `compute_r_squared()` calculating the correct value for 60-bit primes?
3. Should pointwise multiplication in NTT domain use Montgomery multiplication, or regular multiplication?
4. Is there a missing or extra scaling factor somewhere in the forward/inverse/multiply pipeline?
5. Why does the roundtrip `inverse(forward(x))` work perfectly, but `inverse(pointwise_mul(forward(a), forward(b)))` has a scaling issue?

**Debugging suggestions needed**:
- How to verify R^2 mod q is computed correctly?
- What intermediate values should we check to isolate the bug?
- Is there a reference implementation we can compare against?

---

## Environment

- **Hardware**: Apple M3 Max
- **Language**: Rust + Metal Shading Language
- **NTT Parameters**:
  - N = 1024 (power of 2)
  - q = 1152921504606584833 (60-bit NTT-friendly prime, q ≡ 1 mod 2048)
  - psi = 693807653563943717 (primitive 2048-th root of unity)
  - R = 2^64 (Montgomery radix)

---

## Files to Review

1. **Metal Shader**: `src/clifford_fhe_v2/backends/gpu_metal/shaders/ntt.metal`
   - `mont_mul()` - Montgomery multiplication
   - `ntt_pointwise_multiply` kernel
   - `ntt_inverse_final_scale` kernel

2. **Rust NTT Implementation**: `src/clifford_fhe_v2/backends/gpu_metal/ntt.rs`
   - `forward()` - Forward NTT with Montgomery conversion
   - `inverse()` - Inverse NTT with Montgomery conversion
   - `pointwise_multiply()` - Hadamard product in NTT domain
   - `compute_q_inv()` - Compute -q^{-1} mod 2^64
   - `compute_r_squared()` - Compute R^2 mod q

3. **Test Cases**:
   - `examples/test_metal_ntt_correctness.rs` - Roundtrip test (PASSES)
   - `examples/test_metal_ntt_poly_mult.rs` - Polynomial multiplication test (FAILS)
   - `examples/test_metal_ckks_simple.rs` - Full CKKS pipeline test (FAILS)

---

## Expected Behavior

For polynomial multiplication `c(x) = a(x) * b(x)` via NTT:

```
Input:  a = [a₀, a₁, ..., aₙ₋₁]
        b = [b₀, b₁, ..., bₙ₋₁]

1. ā = forward(a)      // Transform to Montgomery NTT domain
2. b̄ = forward(b)      // Transform to Montgomery NTT domain
3. c̄ = ā ⊙ b̄          // Pointwise multiply (using mont_mul)
4. c = inverse(c̄)     // Transform back to coefficient domain

Output: c = [c₀, c₁, ..., cₙ₋₁] where c(x) = a(x) * b(x) mod (x^n + 1)
```

**Currently**: Step 4 produces values that are correct up to a scaling factor of ~2^56.

---

## Additional Notes

- The CPU NTT implementation works correctly and could be used as a reference
- The basic Montgomery multiplication `mont_mul` has been tested in isolation and appears correct
- The NTT butterfly operations use the same `mont_mul` and work correctly
- The bug appears only when combining forward + pointwise_multiply + inverse

Thank you for investigating this issue!
