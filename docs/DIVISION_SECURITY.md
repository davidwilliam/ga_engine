# Security Analysis: Homomorphic Division Implementation

## Summary

This document provides a security analysis of the Newton-Raphson homomorphic division implementation. The analysis concludes that the implementation preserves CKKS semantic security under the Ring Learning With Errors (RLWE) assumption.

**Key Properties:**
1. All operations are performed homomorphically without intermediate decryption
2. Control flow is independent of encrypted values (constant-time execution)
3. Public parameters (initial guess, iteration count) do not compromise security
4. Noise growth is bounded and predictable
5. Trivial encryption of constants follows established FHE conventions

## 1. Algorithm Overview

The implementation computes `a/b` on encrypted inputs using Newton-Raphson iteration:

```
x_{n+1} = x_n × (2 - b × x_n)
```

Starting from initial approximation `x_0 ≈ 1/b`, this converges quadratically to `1/b`. The final result is computed as `a × (1/b)`.

## 2. Security Model

### 2.1 Threat Model

We consider a passive adversary with access to:
- Public parameters (ring dimension N, moduli chain, scale)
- Public algorithm parameters (initial guess, iteration count)
- Ciphertexts (encrypted inputs, intermediate values, encrypted output)
- Public keys and evaluation keys

The adversary does not have access to:
- Secret key
- Plaintext values

### 2.2 Security Goal

**IND-CPA Security**: The adversary cannot distinguish between encryptions of two chosen plaintexts with non-negligible advantage.

## 3. Operation-by-Operation Analysis

### 3.1 Initial Guess Encryption

```rust
let pt_guess = Plaintext::encode(&guess_vec, ct.scale, params);
let ct_xn = ckks_ctx.encrypt(&pt_guess, pk);
```

**Security**: Standard CKKS encryption with fresh randomness. The initial guess is a public algorithm parameter, analogous to polynomial coefficients in function approximation schemes [1, 2].

### 3.2 Ciphertext-Ciphertext Multiplication

```rust
let ct_axn = multiply_ciphertexts(ct, &ct_xn, evk, key_ctx);
```

**Security**: Standard CKKS multiplication with relinearization. This operation preserves IND-CPA security under the RLWE assumption [3].

### 3.3 Trivial Encryption of Constant

```rust
let c0_two: Vec<RnsRepresentation> = pt_two.coeffs.clone();
let c1_zero: Vec<RnsRepresentation> = /* zeros */;
let ct_two = Ciphertext::new(c0_two, c1_zero, ct_axn.level, ct_axn.scale);
```

**Analysis**: This creates a trivial ciphertext `(m, 0)` for the public constant 2.

Trivial encryption of public constants is standard practice in FHE implementations:
- SEAL uses trivial encryption for plaintext-ciphertext operations [4]
- HElib employs the same technique for constant addition [5]
- The CKKS specification explicitly supports plaintext addition [3]

**Security Argument**: After the subtraction `ct_two - ct_axn`, the result has the form:
```
(c0_two - c0_axn, 0 - c1_axn) = (c0_two - c0_axn, -c1_axn)
```

The non-zero `c1` component from `ct_axn` ensures the result is a proper ciphertext. The public constant does not compromise the security of the encrypted operand.

### 3.4 Ciphertext Subtraction

```rust
let ct_two_minus_axn = ct_two.sub(&ct_axn);
```

**Security**: Ciphertext addition/subtraction preserves IND-CPA security [3].

### 3.5 Iteration Structure

```rust
for _ in 0..iterations {
    // Fixed number of iterations
}
```

**Security**: The iteration count is a public parameter, independent of encrypted values. This ensures constant-time execution with respect to the input, preventing timing side-channels.

## 4. Public Parameters

The following parameters are public and do not compromise semantic security:

| Parameter | Justification |
|-----------|---------------|
| Initial guess | Algorithm configuration; analogous to polynomial coefficients in approximation methods |
| Iteration count | Determines precision-depth tradeoff; user-selected prior to computation |
| Constant 2 | Part of Newton-Raphson formula; public by definition |
| Ring dimension, moduli, scale | Standard CKKS public parameters |

## 5. Information Leakage Analysis

### 5.1 Timing

The implementation executes a fixed number of operations regardless of input values. No early termination or adaptive iteration based on encrypted data.

### 5.2 Memory Access Patterns

All memory accesses are determined by public parameters (polynomial degree, number of primes), not by encrypted values.

### 5.3 Control Flow

No conditional branches depend on encrypted data. The algorithm structure is entirely determined by public parameters.

## 6. Noise Growth

### 6.1 Per-Iteration Analysis

Each iteration performs:
- Two ciphertext multiplications (noise approximately doubles per multiplication)
- One ciphertext subtraction (negligible noise increase)

### 6.2 Cumulative Growth

For `k` iterations, noise grows approximately as `O(2^{2k})`:

| Iterations | Noise Factor | Typical Noise Level |
|------------|--------------|---------------------|
| 1 | ~4σ | ~13 |
| 2 | ~16σ | ~51 |
| 3 | ~64σ | ~205 |
| 4 | ~256σ | ~819 |

With a 40-60 bit noise budget (moduli of 2^40 to 2^60), these noise levels remain well within acceptable bounds.

## 7. Security Reduction

**Claim**: If CKKS is IND-CPA secure under the RLWE assumption, then the Newton-Raphson division algorithm is IND-CPA secure.

**Argument**:

1. The algorithm consists solely of:
   - CKKS encryption (IND-CPA secure)
   - Ciphertext-ciphertext multiplication (IND-CPA preserving)
   - Ciphertext addition/subtraction (IND-CPA preserving)
   - Addition of public constants via trivial encryption

2. The composition of IND-CPA preserving operations remains IND-CPA secure.

3. Control flow is independent of encrypted values, preventing adaptive attacks.

4. Therefore, the algorithm inherits the IND-CPA security of CKKS.

## 8. Comparison with Alternative Approaches

### 8.1 Binary Circuit Division

Binary long division requires:
- Comparison operations (data-dependent)
- Conditional subtraction (data-dependent)

Secure implementation requires oblivious selection (multiplexers) for all branches, adding complexity and overhead.

### 8.2 Newton-Raphson Approach

- No data-dependent operations
- Fixed computation structure
- Simpler security analysis
- Lower multiplicative depth

## 9. Recommendations

1. **Documentation**: Clearly state that initial guess and iteration count are public parameters
2. **Parameter Selection**: Provide guidance on choosing initial guess based on known input ranges
3. **Noise Budget**: Include noise growth estimates for different iteration counts
4. **Validation**: For high-assurance applications, consider formal verification of constant-time properties

## 10. References

[1] Cheon, J.H., Han, K., Kim, A., Kim, M., Song, Y. (2018). "Bootstrapping for Approximate Homomorphic Encryption." EUROCRYPT 2018.

[2] Lee, J.W., Lee, E., Lee, Y., Kim, Y.S., No, J.S. (2020). "High-Precision Bootstrapping of RNS-CKKS Homomorphic Encryption Using Optimal Minimax Polynomial Approximation and Inverse Sine Function." EUROCRYPT 2021.

[3] Cheon, J.H., Kim, A., Kim, M., Song, Y. (2017). "Homomorphic Encryption for Arithmetic of Approximate Numbers." ASIACRYPT 2017.

[4] Microsoft SEAL Documentation. "Plaintext Operations." https://github.com/microsoft/SEAL

[5] HElib Documentation. "Adding Constants to Ciphertexts." https://github.com/homenc/HElib

[6] Kim, A., Song, Y., Kim, M., Lee, K., Cheon, J.H. (2018). "Logistic Regression Model Training based on the Approximate Homomorphic Encryption." BMC Medical Genomics.

## Appendix: Code References

| Component | Location |
|-----------|----------|
| CPU implementation | `src/clifford_fhe_v2/inversion.rs` |
| Metal GPU implementation | `src/clifford_fhe_v2/backends/gpu_metal/inversion.rs` |
| Trivial ciphertext creation | Lines 107-117 (CPU), Lines 143-146 (GPU) |
| Iteration loop | Lines 103-124 (CPU), Lines 129-162 (GPU) |
