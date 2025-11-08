# V3 CPU vs Metal GPU Bootstrap - Detailed Comparison

## Executive Summary

**Working:** V3 CPU bootstrap produces perfect results (max error: 3.6e-3)
**Failing:** V2 Metal GPU bootstrap produces massive errors (~285k)

Both implementations appear to use identical logic, yet produce vastly different results. This document provides a detailed comparison to help identify the root cause.

---

## Test Results

### ✅ V3 CPU Bootstrap (WORKING)

**Test:** `test_v3_metal_bootstrap_correct.rs`

```
Slot | Expected | Decrypted | Error
-----|----------|-----------|----------
  0  |   1.00   |   1.00    | 1.87e-11 ✅
  1  |   2.00   |   2.00    | 1.14e-9  ✅
  2  |   3.00   |   3.00    | 3.58e-8  ✅
  3  |   4.00   |   4.00    | 2.26e-4  ✅
  4  |   5.00   |   5.00    | 3.31e-3  ✅

Performance:
  CoeffToSlot:  5.95s
  SlotToCoeff:  1.80s
  Total:        7.75s
  Max Error:    3.61e-3 ✅
```

### ❌ V2 Metal GPU Bootstrap (FAILING)

**Test:** `test_metal_gpu_bootstrap.rs`

```
Slot | Expected | Decrypted | Error
-----|----------|-----------|----------
  0  |   1.00   | 284755.93 | 2.85e5 ❌
  1  |   2.00   |   5908.20 | 5.91e3 ❌
  2  |   3.00   |-101105.15 | 1.01e5 ❌
  3  |   4.00   |  24430.97 | 2.44e4 ❌
  4  |   5.00   |  86978.52 | 8.70e4 ❌

Performance:
  CoeffToSlot:  34.19s
  SlotToCoeff:   8.48s
  Total:        42.66s
  Max Error:    2.85e5 ❌
```

**Key Observation:** Scale explodes to `3.52e13` instead of staying at `~1.1e12`

---

## Side-by-Side Code Comparison

### 1. CoeffToSlot Main Loop

#### V3 CPU (`coeff_to_slot.rs:86-140`)

```rust
for level_idx in 0..num_levels {
    let rotation_amount = 1 << level_idx;  // 1, 2, 4, 8, ..., N/4

    println!("  Level {}: rotation by ±{}, current ct.level={}",
             level_idx, rotation_amount, current.level);

    // Rotate by +rotation_amount
    let ct_rotated = rotate(&current, rotation_amount as i32, rotation_keys)?;

    // Compute DFT twiddle factors
    let mut diag1 = vec![0.5; num_slots];
    let mut diag2 = vec![0.5; num_slots];

    let stride = 1 << level_idx;
    for j in 0..num_slots {
        let k = (j / stride) * stride;
        let theta = 2.0 * PI * (k as f64) / (n as f64);
        let cos_theta = theta.cos();
        diag1[j] = (1.0 + cos_theta) / 2.0;
        diag2[j] = (1.0 - cos_theta) / 2.0;
    }

    // Create temporary params for encoding
    let temp_params = create_temp_params_from_ct(&current)?;

    // Get q_top
    let q_top = temp_params.moduli[current.level] as f64;

    // Encode diagonal matrices with scale = q_top
    let pt_diag1 = Plaintext::encode_at_level(&diag1, q_top, &temp_params, current.level);
    let pt_diag2 = Plaintext::encode_at_level(&diag2, q_top, &temp_params, current.level);

    // Create CKKS context
    let ckks_ctx = CkksContext::new(temp_params.clone());

    // Apply diagonal matrices
    let ct_mul1 = current.multiply_plain(&pt_diag1, &ckks_ctx);
    let ct_mul2 = ct_rotated.multiply_plain(&pt_diag2, &ckks_ctx);

    // Butterfly addition
    current = add_ciphertexts_simple(&ct_mul1, &ct_mul2)?;

    println!("    After level {}: ct.level={}, ct.scale={:.2e}",
             level_idx, current.level, current.scale);
}
```

#### V2 Metal GPU (`bootstrap.rs:88-116`)

```rust
for level_idx in 0..num_levels {
    let rotation_amount = 1 << level_idx;  // 1, 2, 4, 8, ..., N/4

    println!("  Level {}: rotation by ±{}, current level={}",
             level_idx, rotation_amount, current.level);

    // Rotate by +rotation_amount (GPU operation)
    let ct_rotated = current.rotate_by_steps(rotation_amount as i32, rotation_keys, ctx)?;

    // Compute DFT twiddle factors
    let (diag1, diag2) = compute_dft_twiddle_factors(n, num_slots, level_idx);

    // Encode diagonal matrices as plaintexts
    // CRITICAL: Use current level's top modulus for proper scaling
    let q_top = moduli[current.level] as f64;

    let pt_diag1 = encode_diagonal_for_metal(&diag1, q_top, n, current.level, &moduli)?;
    let pt_diag2 = encode_diagonal_for_metal(&diag2, q_top, n, current.level, &moduli)?;

    // Apply diagonal matrices (GPU multiply_plain)
    let ct_mul1 = current.multiply_plain_metal(&pt_diag1, ctx)?;
    let ct_mul2 = ct_rotated.multiply_plain_metal(&pt_diag2, ctx)?;

    // Butterfly addition (GPU operation)
    current = add_metal_ciphertexts(&ct_mul1, &ct_mul2, &moduli)?;

    println!("    After level {}: level={}, scale={:.2e}",
             level_idx, current.level, current.scale);
}
```

**Differences:**
1. **Rotation function:** `rotate()` vs `rotate_by_steps()`
2. **Encoding function:** `Plaintext::encode_at_level()` vs `encode_diagonal_for_metal()`
3. **Multiply function:** `multiply_plain()` vs `multiply_plain_metal()`
4. **Addition function:** `add_ciphertexts_simple()` vs `add_metal_ciphertexts()`

But the **logic is identical**: compute twiddle factors, encode with `q_top`, multiply, add.

---

### 2. Twiddle Factor Computation

#### V3 CPU (`coeff_to_slot.rs:104-114`)

```rust
let stride = 1 << level_idx;
for j in 0..num_slots {
    let k = (j / stride) * stride;
    let theta = 2.0 * PI * (k as f64) / (n as f64);

    let cos_theta = theta.cos();
    diag1[j] = (1.0 + cos_theta) / 2.0;
    diag2[j] = (1.0 - cos_theta) / 2.0;
}
```

#### V2 Metal GPU (`bootstrap.rs:215-230`)

```rust
fn compute_dft_twiddle_factors(n: usize, num_slots: usize, level_idx: usize) -> (Vec<f64>, Vec<f64>) {
    let mut diag1 = vec![0.5; num_slots];
    let mut diag2 = vec![0.5; num_slots];

    let stride = 1 << level_idx;
    for j in 0..num_slots {
        let k = (j / stride) * stride;
        let theta = 2.0 * PI * (k as f64) / (n as f64);

        let cos_theta = theta.cos();
        diag1[j] = (1.0 + cos_theta) / 2.0;
        diag2[j] = (1.0 - cos_theta) / 2.0;
    }

    (diag1, diag2)
}
```

**Difference:** ✅ **IDENTICAL** - Both use the same formula

---

### 3. Plaintext Encoding

#### V3 CPU (`ckks.rs:encode_at_level`)

```rust
pub fn encode_at_level(values: &[f64], scale: f64, params: &CliffordFHEParams, level: usize) -> Self {
    // ... validation ...

    // Encode using canonical embedding with orbit ordering
    let coeffs_vec = canonical_embed_encode_real(values, scale, n);

    // Convert to RNS representation using ONLY the moduli up to the target level
    let moduli: Vec<u64> = params.moduli[..=level].to_vec();
    let mut rns_coeffs = Vec::with_capacity(n);

    for &coeff in &coeffs_vec {
        let values: Vec<u64> = moduli
            .iter()
            .map(|&q| {
                if coeff >= 0 {
                    (coeff as u64) % q
                } else {
                    let abs_coeff = (-coeff) as u64;
                    if abs_coeff % q == 0 {
                        0
                    } else {
                        q - (abs_coeff % q)
                    }
                }
            })
            .collect();

        rns_coeffs.push(RnsRepresentation::new(values, moduli.clone()));
    }

    Self::new(rns_coeffs, scale, level)
}
```

#### V2 Metal GPU (`bootstrap.rs:encode_diagonal_for_metal`)

```rust
fn encode_diagonal_for_metal(
    diagonal: &[f64],
    scale: f64,
    n: usize,
    level: usize,
    moduli: &[u64],
) -> Result<Vec<u64>, String> {
    let num_slots = n / 2;

    // CRITICAL: Use canonical embedding (FIXED - was missing before)
    use crate::clifford_fhe_v2::backends::gpu_metal::ckks::MetalCkksContext;
    let coeffs_i64 = MetalCkksContext::canonical_embed_encode_real(diagonal, scale, n);

    // Convert to RNS representation (flat layout)
    let num_primes = level + 1;
    let mut flat_rns = vec![0u64; n * num_primes];

    for coeff_idx in 0..n {
        let val_i64 = coeffs_i64[coeff_idx];

        for prime_idx in 0..num_primes {
            let q = moduli[prime_idx];
            let val_mod_q = if val_i64 >= 0 {
                (val_i64 as u64) % q
            } else {
                let abs_val = (-val_i64) as u64;
                if abs_val % q == 0 {
                    0
                } else {
                    q - (abs_val % q)
                }
            };

            // Flat RNS layout
            flat_rns[coeff_idx * num_primes + prime_idx] = val_mod_q;
        }
    }

    Ok(flat_rns)
}
```

**Differences:**
1. **Output format:** V3 uses `Vec<RnsRepresentation>` (struct-of-arrays), V2 uses `Vec<u64>` flat layout
2. **RNS conversion:** Identical logic, just different data structures

**Both now use canonical embedding** ✅

---

### 4. Multiply Plain

#### V3 CPU (`ckks.rs:multiply_plain`)

```rust
pub fn multiply_plain(&self, pt: &Plaintext, ctx: &CkksContext) -> Self {
    let n = self.n;
    let level = self.level;

    // ... validation ...

    // Multiply coefficient-wise in RNS
    let mut result_coeffs = Vec::with_capacity(n);
    for i in 0..n {
        let ct_rns = &self.c0[i];
        let pt_rns = &pt.coeffs[i];

        let product_values: Vec<u64> = ct_rns
            .values
            .iter()
            .zip(&pt_rns.values)
            .zip(&ct_rns.moduli)
            .map(|((&&a, &&b), &&q)| {
                ((a as u128 * b as u128) % q as u128) as u64
            })
            .collect();

        result_coeffs.push(RnsRepresentation::new(
            product_values,
            ct_rns.moduli.clone(),
        ));
    }

    // Create result ciphertext (BEFORE rescaling)
    let mut result = Self {
        c0: result_coeffs.clone(),
        c1: self.c1.clone(),
        level,
        scale: self.scale * pt.scale,  // Scale multiplies
        n,
    };

    // Rescale to maintain constant scale
    result.rescale_to_next_with_scale(ctx, result.scale)
}
```

#### V2 Metal GPU (`bootstrap.rs:multiply_plain_metal`)

```rust
pub fn multiply_plain_metal(
    &self,
    pt: &[u64],
    ctx: &MetalCkksContext,
) -> Result<Self, String> {
    let n = self.n;
    let num_primes = self.level + 1;

    // ... validation and extraction ...

    // Multiply c0 * pt (GPU NTT multiplication)
    let c0_mul = ctx.multiply_polys_flat_ntt_negacyclic(&c0_active, pt, &moduli)?;

    // Multiply c1 * pt (GPU NTT multiplication)
    let c1_mul = ctx.multiply_polys_flat_ntt_negacyclic(&c1_active, pt, &moduli)?;

    // Rescale: drop top modulus and scale down
    let new_level = if self.level > 0 { self.level - 1 } else { 0 };
    let new_num_primes = new_level + 1;

    let q_top = moduli[self.level] as f64;
    // After ct * pt: scale = ct.scale * pt.scale = self.scale * q_top
    // After rescale: scale = (self.scale * q_top) / q_top = self.scale
    let new_scale = self.scale;

    // Extract only the primes we keep (drop the top one)
    let mut c0_rescaled = vec![0u64; n * new_num_primes];
    let mut c1_rescaled = vec![0u64; n * new_num_primes];

    for coeff_idx in 0..n {
        for prime_idx in 0..new_num_primes {
            c0_rescaled[coeff_idx * new_num_primes + prime_idx] =
                c0_mul[coeff_idx * num_primes + prime_idx];
            c1_rescaled[coeff_idx * new_num_primes + prime_idx] =
                c1_mul[coeff_idx * num_primes + prime_idx];
        }
    }

    Ok(Self {
        c0: c0_rescaled,
        c1: c1_rescaled,
        n,
        num_primes: self.num_primes,
        level: new_level,
        scale: new_scale,  // Scale preserved
    })
}
```

**Differences:**
1. **Multiplication method:** V3 uses coefficient-wise multiplication, V2 uses NTT multiplication
2. **Rescaling method:** V3 uses `rescale_to_next_with_scale()` (exact BigInt division), V2 uses simple prime dropping

**CRITICAL DIFFERENCE:** V3 CPU uses **exact rescaling** with BigInt CRT reconstruction and exact division. V2 Metal GPU uses **approximate rescaling** by simply dropping the top prime.

---

### 5. Rescaling Implementation

#### V3 CPU (`ckks.rs:rescale_to_next_with_scale`)

```rust
pub fn rescale_to_next_with_scale(&self, ckks_ctx: &CkksContext, pre_rescale_scale: f64) -> Self {
    let level = self.level;
    assert!(level > 0, "Cannot rescale at level 0");

    let moduli = &ckks_ctx.params.moduli[..=level];
    let q_top = moduli[level];

    let mut new_c0 = Vec::with_capacity(self.n);
    let mut new_c1 = Vec::with_capacity(self.n);

    for i in 0..self.n {
        // EXACT rescaling using BigInt CRT
        let new_c0_limbs = Self::rescale_coeff_bigint(&self.c0[i].values, moduli, q_top);
        let new_c1_limbs = Self::rescale_coeff_bigint(&self.c1[i].values, moduli, q_top);

        new_c0.push(RnsRepresentation::new(new_c0_limbs, moduli[..level].to_vec()));
        new_c1.push(RnsRepresentation::new(new_c1_limbs, moduli[..level].to_vec()));
    }

    let new_scale = pre_rescale_scale / (q_top as f64);

    Self {
        c0: new_c0,
        c1: new_c1,
        level: level - 1,
        scale: new_scale,
        n: self.n,
    }
}

fn rescale_coeff_bigint(limbs: &[u64], moduli: &[u64], q_top: u64) -> Vec<u64> {
    use num_bigint::BigInt;
    use num_traits::{ToPrimitive, Zero};

    // CRT reconstruct to BigInt
    let q_prod_big: BigInt = moduli.iter().map(|&q| BigInt::from(q)).product();

    let mut coeff_big = BigInt::zero();
    for (j, &q) in moduli.iter().enumerate() {
        let residue = limbs[j];
        let q_big = BigInt::from(q);
        let q_inv = &q_prod_big / &q_big;
        let q_inv_mod = Self::mod_inverse_bigint(&q_inv, &q_big)?;
        let term = BigInt::from(residue) * q_inv * q_inv_mod;
        coeff_big += term;
    }
    coeff_big %= &q_prod_big;

    // Center around zero
    let q_half = &q_prod_big / 2;
    if coeff_big > q_half {
        coeff_big -= &q_prod_big;
    }

    // EXACT floor division: (coeff + q_top/2) / q_top
    use num_integer::Integer;
    let q_top_big = BigInt::from(q_top);
    let half_q_top = &q_top_big / 2;

    let rescaled = if coeff_big >= BigInt::zero() {
        (&coeff_big + &half_q_top).div_floor(&q_top_big)
    } else {
        // For negative: ⌊(c + q/2) / q⌋ = -⌊(-c - q/2) / q⌋ - 1
        let neg_rescaled = ((-&coeff_big) - &half_q_top).div_floor(&q_top_big);
        -neg_rescaled - 1
    };

    // Convert back to RNS
    let new_moduli = &moduli[..moduli.len() - 1];
    new_moduli
        .iter()
        .map(|&q| {
            let q_big = BigInt::from(q);
            let res = rescaled.mod_floor(&q_big);
            res.to_u64().unwrap_or(0)
        })
        .collect()
}
```

#### V2 Metal GPU (NO EXACT RESCALING)

```rust
// V2 Metal GPU simply drops the top prime WITHOUT exact division
// This is line 459-470 in bootstrap.rs

// Extract only the primes we keep (drop the top one)
let mut c0_rescaled = vec![0u64; n * new_num_primes];
let mut c1_rescaled = vec![0u64; n * new_num_primes];

for coeff_idx in 0..n {
    for prime_idx in 0..new_num_primes {
        c0_rescaled[coeff_idx * new_num_primes + prime_idx] =
            c0_mul[coeff_idx * num_primes + prime_idx];
        c1_rescaled[coeff_idx * new_num_primes + prime_idx] =
            c1_mul[coeff_idx * num_primes + prime_idx];
    }
}
```

**CRITICAL DIFFERENCE:**
- V3 CPU performs **exact floor division by q_top** using BigInt CRT
- V2 Metal GPU performs **approximate rescaling** by just dropping the top prime

This could explain the errors!

---

### 6. Rotation Implementation

#### V3 CPU (`rotation.rs:rotate`)

```rust
pub fn rotate(
    ct: &Ciphertext,
    k: i32,
    rotation_keys: &RotationKeys,
) -> Result<Ciphertext, String> {
    let n = ct.n;

    // Step 1: Compute Galois element for rotation by k
    let g = galois_element_for_rotation(k, n);

    // Step 2: Get rotation key for this Galois element
    let rotation_key = rotation_keys.get_key(g)
        .ok_or_else(|| format!("Rotation key for k={} (g={}) not found", k, g))?;

    // Step 3: Apply Galois automorphism to c0 and c1
    let mut c0_new = apply_galois_automorphism(&ct.c0, g, n);
    let c1_rotated = apply_galois_automorphism(&ct.c1, g, n);

    // Step 4: Key-switch c1_rotated from s(X^g) to s(X)
    let c1_new = key_switch(&mut c0_new, &c1_rotated, rotation_key, n)?;

    Ok(Ciphertext {
        c0: c0_new,
        c1: c1_new,
        level: ct.level,
        scale: ct.scale,
        n: ct.n,
    })
}

fn key_switch(
    c0: &mut Vec<RnsRepresentation>,
    c1_rotated: &[RnsRepresentation],
    rotation_key: &RotationKey,
    n: usize,
) -> Result<Vec<RnsRepresentation>, String> {
    let moduli = &c1_rotated[0].moduli;
    let base_w = rotation_key.base_w;

    let mut c1_new = vec![RnsRepresentation::new(vec![0; moduli.len()], moduli.to_vec()); n];

    // Decompose c1_rotated using gadget decomposition
    let c1_digits = gadget_decompose(c1_rotated, base_w, moduli);

    // For each digit in the decomposition
    for (t, c1_digit) in c1_digits.iter().enumerate() {
        if t >= rotation_key.rlk0.len() {
            break;
        }

        let term0 = multiply_polynomials_ntt(c1_digit, &rotation_key.rlk0[t], moduli, n);
        let term1 = multiply_polynomials_ntt(c1_digit, &rotation_key.rlk1[t], moduli, n);

        for i in 0..n {
            c0[i] = c0[i].sub(&term0[i]);  // SUBTRACT term0
            c1_new[i] = c1_new[i].add(&term1[i]);  // ADD term1
        }
    }

    Ok(c1_new)
}
```

#### V2 Metal GPU (`ckks.rs:rotate_by_steps`)

```rust
pub fn rotate_by_steps(
    &self,
    step: i32,
    rot_keys: &super::rotation_keys::MetalRotationKeys,
    ctx: &MetalCkksContext,
) -> Result<Self, String> {
    let n = self.n;
    let num_primes_active = self.level + 1;
    let moduli = &ctx.params.moduli[..num_primes_active];

    // Get rotation key with gadget decomposition
    let (rlk0_full, rlk1_full) = rot_keys.get_key_for_step(step)
        .ok_or_else(|| format!("Rotation key for step {} not found", step))?;

    // Extract only active primes from rotation keys
    // ... (extraction code) ...

    // Compute Galois element and map
    let g = rotation_step_to_galois_element(step, n);
    let galois_map = compute_galois_map(g, n);

    // Apply Galois automorphism (GPU operation)
    let c0_rotated = Self::apply_galois_automorphism_gpu(
        &c0_active, &galois_map, n, num_primes_active, &device
    )?;
    let c1_rotated = Self::apply_galois_automorphism_gpu(
        &c1_active, &galois_map, n, num_primes_active, &device
    )?;

    // Key switch (gadget decomposition)
    let (c0_final, c1_final) = self.key_switch_gpu_gadget(
        &c0_rotated,
        &c1_rotated,
        &rlk0,
        &rlk1,
        moduli,
        base_w,
        ctx,
    )?;

    Ok(Self {
        c0: c0_final,
        c1: c1_final,
        n,
        num_primes: self.num_primes,
        level: self.level,
        scale: self.scale,
    })
}

fn key_switch_gpu_gadget(
    &self,
    c0_rotated: &[u64],
    c1_rotated: &[u64],
    rlk0: &[Vec<u64>],
    rlk1: &[Vec<u64>],
    moduli: &[u64],
    base_w: u32,
    ctx: &MetalCkksContext,
) -> Result<(Vec<u64>, Vec<u64>), String> {
    // ... initialization ...

    // Decompose c1_rotated
    let c1_digits = Self::gadget_decompose_flat(c1_rotated, base_w, moduli, n)?;

    // For each digit t
    for t in 0..num_digits {
        if t >= c1_digits.len() {
            break;
        }

        let term0 = Self::multiply_digit_by_ntt_key(&c1_digits[t], &rlk0[t], moduli, ctx)?;
        let term1 = Self::multiply_digit_by_ntt_key(&c1_digits[t], &rlk1[t], moduli, ctx)?;

        // c0_final -= term0 (SUBTRACT)
        for i in 0..(n * num_primes) {
            let prime_idx = i % num_primes;
            let q = moduli[prime_idx];

            let diff = if c0_final[i] >= term0[i] {
                c0_final[i] - term0[i]
            } else {
                q - (term0[i] - c0_final[i])
            };
            c0_final[i] = diff;
        }

        // c1_final += term1 (ADD)
        for i in 0..(n * num_primes) {
            let prime_idx = i % num_primes;
            let q = moduli[prime_idx];
            c1_final[i] = ((c1_final[i] as u128 + term1[i] as u128) % q as u128) as u64;
        }
    }

    Ok((c0_final, c1_final))
}
```

**Differences:**
1. **Data structures:** V3 uses struct-of-arrays (`Vec<RnsRepresentation>`), V2 uses flat arrays
2. **Galois automorphism:** V3 uses CPU function, V2 uses GPU kernel
3. **Key switching formula:** ✅ **IDENTICAL** - Both subtract term0 and add term1

---

## Rotation Key Generation

Both use the SAME corrected formula after our fix:

```rust
rlk0[t] = -B^t·s_k + a_t·s + e_t   // Correct sign (negated B^t·s_k)
```

File: `src/clifford_fhe_v2/backends/gpu_metal/rotation_keys.rs:283-297`

This was verified working in the simple rotation test (`test_cpu_vs_gpu_rotation`).

---

## Summary of Differences

| Component | V3 CPU | V2 Metal GPU | Impact |
|-----------|--------|--------------|--------|
| **Twiddle factors** | ✅ Identical | ✅ Identical | None |
| **Canonical embedding** | ✅ Yes | ✅ Yes (after fix) | None |
| **Plaintext encoding** | Uses `q_top` | Uses `q_top` | ✅ Correct |
| **Multiply method** | Coefficient-wise | NTT-based | Should be equivalent |
| **Rescaling** | ⚠️ **EXACT (BigInt CRT + exact division)** | ⚠️ **APPROXIMATE (just drop prime)** | **CRITICAL** |
| **Rotation formula** | ✅ Subtract term0 | ✅ Subtract term0 | None |
| **Data structures** | Struct-of-arrays | Flat arrays | Should be equivalent |

---

## Hypothesis

The **root cause** is likely the **rescaling difference**:

- **V3 CPU:** Performs exact floor division `⌊(coeff + q_top/2) / q_top⌋` using BigInt CRT reconstruction
- **V2 Metal GPU:** Simply drops the top prime without division

### Why This Matters

After `ct * pt` where `pt.scale = q_top`:
- Result scale = `ct.scale × q_top`
- **Correct rescaling:** Divide by `q_top` → scale = `ct.scale` ✅
- **Approximate rescaling (drop prime):** No division → scale = `ct.scale × q_top` ❌

This would cause **exponential scale growth**:
- Level 0: `1.1e12 × q_top ≈ 5.5e25`
- But somehow it shows as `3.52e13`?

### Questions for Expert

1. **Is approximate rescaling (just dropping prime) mathematically sound?**
   - V2 Metal GPU doesn't perform the division step
   - It only drops the top RNS component

2. **Could the NTT-based multiplication be introducing errors?**
   - V3 uses coefficient-wise multiplication
   - V2 uses NTT domain multiplication
   - Both should be mathematically equivalent

3. **Could the flat RNS layout vs struct-of-arrays affect correctness?**
   - V3: `Vec<RnsRepresentation>` where each element has `values: Vec<u64>`
   - V2: `Vec<u64>` flat with stride indexing `[coeff0_q0, coeff0_q1, ..., coeff1_q0, ...]`

4. **Why does scale show as `3.52e13` instead of continuing to grow?**
   - Expected: exponential growth if division is missing
   - Observed: scale stabilizes at `3.52e13` (32× too high)
   - This suggests something is preventing further growth

---

## Files for Reference

### Working V3 CPU Implementation
- **CoeffToSlot:** `src/clifford_fhe_v3/bootstrapping/coeff_to_slot.rs`
- **SlotToCoeff:** `src/clifford_fhe_v3/bootstrapping/slot_to_coeff.rs`
- **Rotation:** `src/clifford_fhe_v3/bootstrapping/rotation.rs`
- **CKKS (rescaling):** `src/clifford_fhe_v2/backends/cpu_optimized/ckks.rs`
- **Test:** `examples/test_v3_metal_bootstrap_correct.rs`

### Failing V2 Metal GPU Implementation
- **Bootstrap:** `src/clifford_fhe_v2/backends/gpu_metal/bootstrap.rs`
- **CKKS (rotation):** `src/clifford_fhe_v2/backends/gpu_metal/ckks.rs`
- **Rotation Keys:** `src/clifford_fhe_v2/backends/gpu_metal/rotation_keys.rs`
- **Test:** `examples/test_metal_gpu_bootstrap.rs`

---

## Request for Expert Guidance

Given that:
1. V3 CPU works perfectly (3.6e-3 error)
2. V2 Metal GPU fails massively (285k error)
3. Both use identical high-level logic
4. Main difference is **exact vs approximate rescaling**

**Questions:**
1. Is the approximate rescaling in V2 Metal GPU the root cause?
2. Should we port V3's exact rescaling (BigInt CRT) to Metal GPU?
3. Or is there another subtle difference we're missing?

Thank you for any insights you can provide!
