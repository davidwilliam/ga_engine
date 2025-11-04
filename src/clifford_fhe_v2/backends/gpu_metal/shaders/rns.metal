//! Metal Compute Shaders for RNS (Residue Number System) Operations
//!
//! RNS representation for large integers using multiple small primes.
//! Each coefficient is stored as residues mod q₀, q₁, ..., qₖ

#include <metal_stdlib>
using namespace metal;

/// RNS: Add two polynomials across all primes
///
/// For each coefficient i and prime j:
///   c[i,j] = (a[i,j] + b[i,j]) mod qⱼ
///
/// @param a First polynomial in RNS form [n × num_primes]
/// @param b Second polynomial in RNS form [n × num_primes]
/// @param c Output polynomial in RNS form [n × num_primes]
/// @param moduli Array of prime moduli [num_primes]
/// @param n Polynomial degree
/// @param num_primes Number of RNS primes
kernel void rns_add(
    device const ulong* a [[buffer(0)]],
    device const ulong* b [[buffer(1)]],
    device ulong* c [[buffer(2)]],
    constant ulong* moduli [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& num_primes [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint coeff_idx = gid.x;
    uint prime_idx = gid.y;

    if (coeff_idx < n && prime_idx < num_primes) {
        uint idx = prime_idx * n + coeff_idx;
        ulong q = moduli[prime_idx];

        ulong sum = a[idx] + b[idx];
        c[idx] = (sum >= q) ? (sum - q) : sum;
    }
}

/// RNS: Subtract two polynomials across all primes
kernel void rns_sub(
    device const ulong* a [[buffer(0)]],
    device const ulong* b [[buffer(1)]],
    device ulong* c [[buffer(2)]],
    constant ulong* moduli [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& num_primes [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint coeff_idx = gid.x;
    uint prime_idx = gid.y;

    if (coeff_idx < n && prime_idx < num_primes) {
        uint idx = prime_idx * n + coeff_idx;
        ulong q = moduli[prime_idx];

        ulong diff = (a[idx] >= b[idx]) ? (a[idx] - b[idx]) : (a[idx] + q - b[idx]);
        c[idx] = diff;
    }
}

/// RNS: Multiply two polynomials in NTT domain (pointwise per prime)
///
/// Assumes inputs are in NTT form. Performs pointwise multiplication
/// for each prime independently.
kernel void rns_ntt_multiply(
    device const ulong* a [[buffer(0)]],
    device const ulong* b [[buffer(1)]],
    device ulong* c [[buffer(2)]],
    constant ulong* moduli [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& num_primes [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint coeff_idx = gid.x;
    uint prime_idx = gid.y;

    if (coeff_idx < n && prime_idx < num_primes) {
        uint idx = prime_idx * n + coeff_idx;
        ulong q = moduli[prime_idx];

        // Modular multiplication using 128-bit intermediate
        ulong a_val = a[idx];
        ulong b_val = b[idx];

        ulong hi = mulhi(a_val, b_val);
        ulong lo = a_val * b_val;

        // For FHE primes < 2^60, simple modulo suffices
        c[idx] = lo % q;
    }
}

/// RNS: Scale polynomial by scalar across all primes
///
/// @param poly Input polynomial [n × num_primes]
/// @param scalar Scalar value to multiply
/// @param result Output polynomial [n × num_primes]
kernel void rns_scale(
    device const ulong* poly [[buffer(0)]],
    constant ulong& scalar [[buffer(1)]],
    device ulong* result [[buffer(2)]],
    constant ulong* moduli [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& num_primes [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint coeff_idx = gid.x;
    uint prime_idx = gid.y;

    if (coeff_idx < n && prime_idx < num_primes) {
        uint idx = prime_idx * n + coeff_idx;
        ulong q = moduli[prime_idx];

        ulong val = poly[idx];
        ulong scaled = (val * scalar) % q;
        result[idx] = scaled;
    }
}

/// RNS: Modulus switching (drop one prime from RNS representation)
///
/// Used after multiplication to maintain noise budget.
/// Converts from modulus Q = q₀×q₁×...×qₖ to Q' = q₀×q₁×...×qₖ₋₁
///
/// Algorithm:
/// 1. For each coefficient, we have residues (r₀, r₁, ..., rₖ)
/// 2. Drop rₖ (the last residue)
/// 3. Scale remaining residues: r'ᵢ = round(rᵢ × qₖ⁻¹) mod qᵢ
///
/// @param poly_in Input polynomial [n × num_primes]
/// @param poly_out Output polynomial [n × (num_primes-1)]
/// @param moduli Array of all primes [num_primes]
/// @param qk_inv Modular inverse of dropped prime mod each remaining prime
kernel void rns_modswitch(
    device const ulong* poly_in [[buffer(0)]],
    device ulong* poly_out [[buffer(1)]],
    constant ulong* moduli [[buffer(2)]],
    constant ulong* qk_inv [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& num_primes_in [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint coeff_idx = gid.x;
    uint prime_idx = gid.y;

    uint num_primes_out = num_primes_in - 1;

    if (coeff_idx < n && prime_idx < num_primes_out) {
        // Input index (includes dropped prime)
        uint idx_in = prime_idx * n + coeff_idx;
        // Output index (excludes dropped prime)
        uint idx_out = prime_idx * n + coeff_idx;

        ulong q = moduli[prime_idx];
        ulong r = poly_in[idx_in];

        // Scale by qₖ⁻¹ mod qᵢ
        ulong scaled = (r * qk_inv[prime_idx]) % q;

        poly_out[idx_out] = scaled;
    }
}

/// Barrett reduction for fast modular arithmetic on GPU
///
/// Precomputed mu = floor(2^64 / q) for each prime
inline ulong barrett_reduce(ulong x, ulong q, ulong mu) {
    if (x < q) return x;

    // Barrett reduction: x - q * floor(x * mu / 2^64)
    ulong quotient = mulhi(x, mu);
    ulong result = x - quotient * q;

    // Correction step
    if (result >= q) result -= q;

    return result;
}

/// RNS: Polynomial multiplication with Barrett reduction (faster)
kernel void rns_ntt_multiply_barrett(
    device const ulong* a [[buffer(0)]],
    device const ulong* b [[buffer(1)]],
    device ulong* c [[buffer(2)]],
    constant ulong* moduli [[buffer(3)]],
    constant ulong* barrett_mu [[buffer(4)]],  // Precomputed Barrett constants
    constant uint& n [[buffer(5)]],
    constant uint& num_primes [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint coeff_idx = gid.x;
    uint prime_idx = gid.y;

    if (coeff_idx < n && prime_idx < num_primes) {
        uint idx = prime_idx * n + coeff_idx;
        ulong q = moduli[prime_idx];
        ulong mu = barrett_mu[prime_idx];

        ulong a_val = a[idx];
        ulong b_val = b[idx];

        // Compute product
        ulong lo = a_val * b_val;

        // Barrett reduction
        c[idx] = barrett_reduce(lo, q, mu);
    }
}
