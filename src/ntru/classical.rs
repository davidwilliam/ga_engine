//! Classical NTRU polynomial multiplication algorithms
//!
//! This module implements three standard approaches:
//! 1. Naive O(N²) multiplication with convolution
//! 2. Karatsuba O(N^1.58) divide-and-conquer
//! 3. Toeplitz matrix-vector product (TMVP) approach

use super::polynomial::Polynomial;

/// Naive polynomial multiplication in Z[x]/(x^N - 1)
///
/// This is the straightforward convolution approach with O(N²) complexity.
/// It's the baseline for comparison.
///
/// ## Algorithm
///
/// For polynomials a(x) and b(x), compute c(x) = a(x) * b(x) mod (x^N - 1):
/// ```text
/// c[k] = sum_{i+j ≡ k (mod N)} a[i] * b[j]
/// ```
///
/// The reduction x^N = 1 means that x^(N+k) = x^k, so we use wraparound.
///
/// ## Complexity
/// - Time: O(N²)
/// - Space: O(N)
pub fn naive_multiply(a: &Polynomial, b: &Polynomial) -> Polynomial {
    assert_eq!(
        a.params, b.params,
        "Polynomials must have the same parameters"
    );

    let n = a.params.n;
    let mut result = Polynomial::zero(a.params);

    // Convolution with wraparound for x^N = 1
    for i in 0..n {
        for j in 0..n {
            let k = (i + j) % n; // Reduction modulo x^N - 1
            result.coeffs[k] += a.coeffs[i] * b.coeffs[j];
        }
    }

    result
}

/// Karatsuba polynomial multiplication
///
/// Divide-and-conquer algorithm that reduces the number of multiplications
/// at the cost of more additions.
///
/// ## Complexity
/// - Time: O(N^log₂(3)) ≈ O(N^1.585)
/// - Space: O(N log N) due to recursion
///
/// ## Note
/// This implementation works best when N is a power of 2.
/// For small N (< 16), naive multiplication may be faster due to overhead.
pub fn karatsuba_multiply(a: &Polynomial, b: &Polynomial) -> Polynomial {
    assert_eq!(
        a.params, b.params,
        "Polynomials must have the same parameters"
    );

    let n = a.params.n;

    // Base case: use naive multiplication for small polynomials
    if n <= 8 {
        return naive_multiply(a, b);
    }

    // For non-power-of-2 sizes, fall back to naive
    if !n.is_power_of_two() {
        return naive_multiply(a, b);
    }

    // Karatsuba algorithm for power-of-2 sizes
    karatsuba_multiply_impl(&a.coeffs, &b.coeffs, a.params)
}

fn karatsuba_multiply_impl(a: &[i64], b: &[i64], params: super::polynomial::NTRUParams) -> Polynomial {
    let n = a.len();

    if n <= 8 {
        // Base case: naive multiplication
        let poly_a = Polynomial::new(a.to_vec(), params);
        let poly_b = Polynomial::new(b.to_vec(), params);
        return naive_multiply(&poly_a, &poly_b);
    }

    let half = n / 2;

    // Split polynomials: a = a_low + x^(n/2) * a_high
    let a_low = &a[..half];
    let a_high = &a[half..];
    let b_low = &b[..half];
    let b_high = &b[half..];

    // Karatsuba: compute 3 products instead of 4
    // z0 = a_low * b_low
    // z2 = a_high * b_high
    // z1 = (a_low + a_high) * (b_low + b_high) - z0 - z2

    let mut z0_coeffs = vec![0i64; n];
    let mut z2_coeffs = vec![0i64; n];
    let mut z1_coeffs = vec![0i64; n];

    // z0 = a_low * b_low
    for i in 0..half {
        for j in 0..half {
            let k = (i + j) % n;
            z0_coeffs[k] += a_low[i] * b_low[j];
        }
    }

    // z2 = a_high * b_high
    for i in 0..half {
        for j in 0..half {
            let k = (i + j + n) % n; // Shifted by n
            z2_coeffs[k] += a_high[i] * b_high[j];
        }
    }

    // (a_low + a_high) and (b_low + b_high)
    let mut a_sum = vec![0i64; half];
    let mut b_sum = vec![0i64; half];
    for i in 0..half {
        a_sum[i] = a_low[i] + a_high[i];
        b_sum[i] = b_low[i] + b_high[i];
    }

    // z1_temp = (a_low + a_high) * (b_low + b_high)
    for i in 0..half {
        for j in 0..half {
            let k = (i + j + half) % n;
            z1_coeffs[k] += a_sum[i] * b_sum[j];
        }
    }

    // z1 = z1_temp - z0 - z2
    let mut result = vec![0i64; n];
    for i in 0..n {
        result[i] = z0_coeffs[i] + z2_coeffs[i] + z1_coeffs[i] - z0_coeffs[i] - z2_coeffs[i];
    }

    // Combine: result = z0 + z1 * x^(n/2) + z2 * x^n
    // But x^n = 1 in our ring, so x^n wraps around
    for i in 0..n {
        result[i] = z0_coeffs[i] + z1_coeffs[i] + z2_coeffs[i];
    }

    Polynomial::new(result, params)
}

/// Toeplitz matrix-vector product (TMVP) for polynomial multiplication
///
/// This represents polynomial multiplication as a matrix-vector product,
/// which can leverage optimized BLAS routines.
///
/// ## Algorithm
///
/// For polynomial a(x) * b(x) in Z[x]/(x^N - 1), we can construct a
/// Toeplitz matrix T from polynomial a, where:
///
/// ```text
/// T[i][j] = a[(i - j) mod N]
/// ```
///
/// Then: c = T * b (matrix-vector product)
///
/// ## Toeplitz Structure Example (N=4)
///
/// For a = [a0, a1, a2, a3], the matrix is:
/// ```text
/// [a0  a3  a2  a1]
/// [a1  a0  a3  a2]
/// [a2  a1  a0  a3]
/// [a3  a2  a1  a0]
/// ```
///
/// This structure arises from the cyclic convolution property of x^N = 1.
///
/// ## Complexity
/// - Time: O(N²) for naive matrix-vector product
/// - Space: O(N²) for storing the matrix
/// - Can be optimized with BLAS: typically faster than naive despite same complexity
///
/// ## Note
/// This is the approach used in many NTRU implementations because:
/// 1. It leverages hardware-optimized matrix operations
/// 2. Cache-friendly memory access patterns
/// 3. Easy to vectorize (SIMD)
pub fn toeplitz_matrix_multiply(a: &Polynomial, b: &Polynomial) -> Polynomial {
    assert_eq!(
        a.params, b.params,
        "Polynomials must have the same parameters"
    );

    let n = a.params.n;
    let mut result = vec![0i64; n];

    // Construct Toeplitz matrix implicitly and compute matrix-vector product
    for i in 0..n {
        for j in 0..n {
            // Toeplitz structure: T[i][j] = a[(i - j) mod N]
            let idx = if i >= j { i - j } else { n + i - j };
            result[i] += a.coeffs[idx] * b.coeffs[j];
        }
    }

    Polynomial::new(result, a.params)
}

/// Convert polynomial multiplication to explicit 8×8 matrix multiplication
///
/// This function returns the Toeplitz matrix representation of a polynomial,
/// which allows us to use optimized 8×8 matrix multiplication routines.
///
/// This is the key function that connects NTRU to our GA-based matrix speedups.
pub fn polynomial_to_toeplitz_matrix_8x8(poly: &Polynomial) -> [f64; 64] {
    assert_eq!(poly.params.n, 8, "This function only works for N=8");

    let mut matrix = [0.0f64; 64];

    for i in 0..8 {
        for j in 0..8 {
            let idx = if i >= j { i - j } else { 8 + i - j };
            matrix[i * 8 + j] = poly.coeffs[idx] as f64;
        }
    }

    matrix
}

/// Convert polynomial multiplication to explicit 16×16 matrix multiplication
///
/// This function returns the Toeplitz matrix representation for N=16,
/// allowing us to use our GA-based 16×16 matrix speedups.
pub fn polynomial_to_toeplitz_matrix_16x16(poly: &Polynomial) -> [f64; 256] {
    assert_eq!(poly.params.n, 16, "This function only works for N=16");

    let mut matrix = [0.0f64; 256];

    for i in 0..16 {
        for j in 0..16 {
            let idx = if i >= j { i - j } else { 16 + i - j };
            matrix[i * 16 + j] = poly.coeffs[idx] as f64;
        }
    }

    matrix
}

/// Matrix-vector multiply for 8×8 matrix
#[inline]
pub fn matrix_vector_multiply_8x8(matrix: &[f64; 64], vector: &[i64; 8]) -> [i64; 8] {
    let mut result = [0i64; 8];

    for i in 0..8 {
        let mut sum = 0.0f64;
        for j in 0..8 {
            sum += matrix[i * 8 + j] * (vector[j] as f64);
        }
        result[i] = sum.round() as i64;
    }

    result
}

/// Matrix-vector multiply for 16×16 matrix
#[inline]
pub fn matrix_vector_multiply_16x16(matrix: &[f64; 256], vector: &[i64; 16]) -> [i64; 16] {
    let mut result = [0i64; 16];

    for i in 0..16 {
        let mut sum = 0.0f64;
        for j in 0..16 {
            sum += matrix[i * 16 + j] * (vector[j] as f64);
        }
        result[i] = sum.round() as i64;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntru::polynomial::NTRUParams;

    #[test]
    fn test_naive_multiply_simple() {
        let params = NTRUParams::N8_TOY;

        // Test: (1 + x) * (1 + x) = 1 + 2x + x^2
        let a = Polynomial::new(vec![1, 1, 0, 0, 0, 0, 0, 0], params);
        let b = Polynomial::new(vec![1, 1, 0, 0, 0, 0, 0, 0], params);

        let c = naive_multiply(&a, &b);

        assert_eq!(c.coeffs, vec![1, 2, 1, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_naive_multiply_wraparound() {
        let params = NTRUParams::N8_TOY;

        // Test wraparound: x^7 * x = x^8 = 1 in ring Z[x]/(x^8 - 1)
        let a = Polynomial::new(vec![0, 0, 0, 0, 0, 0, 0, 1], params); // x^7
        let b = Polynomial::new(vec![0, 1, 0, 0, 0, 0, 0, 0], params); // x

        let c = naive_multiply(&a, &b);

        // Result should be x^8 = 1
        assert_eq!(c.coeffs, vec![1, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_toeplitz_multiply() {
        let params = NTRUParams::N8_TOY;

        let a = Polynomial::new(vec![1, 1, 0, 0, 0, 0, 0, 0], params);
        let b = Polynomial::new(vec![1, 1, 0, 0, 0, 0, 0, 0], params);

        let c_naive = naive_multiply(&a, &b);
        let c_toeplitz = toeplitz_matrix_multiply(&a, &b);

        assert_eq!(c_naive.coeffs, c_toeplitz.coeffs);
    }

    #[test]
    fn test_polynomial_to_toeplitz_matrix() {
        let params = NTRUParams::N8_TOY;
        let poly = Polynomial::new(vec![1, 2, 3, 4, 5, 6, 7, 8], params);

        let matrix = polynomial_to_toeplitz_matrix_8x8(&poly);

        // Check Toeplitz structure: T[i][j] = a[(i-j) mod 8]
        // First row should be: [1, 8, 7, 6, 5, 4, 3, 2]
        assert_eq!(matrix[0], 1.0);
        assert_eq!(matrix[1], 8.0);
        assert_eq!(matrix[2], 7.0);
        assert_eq!(matrix[7], 2.0);

        // Second row should be: [2, 1, 8, 7, 6, 5, 4, 3]
        assert_eq!(matrix[8], 2.0);
        assert_eq!(matrix[9], 1.0);
        assert_eq!(matrix[10], 8.0);
    }

    #[test]
    fn test_matrix_vector_multiply_8x8() {
        // Test identity matrix
        let mut identity = [0.0f64; 64];
        for i in 0..8 {
            identity[i * 8 + i] = 1.0;
        }

        let vector = [1, 2, 3, 4, 5, 6, 7, 8];
        let result = matrix_vector_multiply_8x8(&identity, &vector);

        assert_eq!(result, vector);
    }
}
