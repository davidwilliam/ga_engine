//! Sine Polynomial Approximation
//!
//! Computes Chebyshev/Taylor polynomial coefficients for sine function.
//! Used in EvalMod to approximate modular reduction: x mod q ≈ x - (q/2π) · sin(2πx/q)

use std::f64::consts::PI;

/// Compute Chebyshev polynomial coefficients for sin(x) on [-π, π]
///
/// Returns coefficients [c0, c1, c2, ..., c_degree]
///
/// # Arguments
///
/// * `degree` - Degree of polynomial (must be odd, >= 5)
///
/// # Returns
///
/// Vector of coefficients where result[i] is coefficient for x^i
///
/// # Example
///
/// ```
/// use ga_engine::clifford_fhe_v3::bootstrapping::chebyshev_sin_coeffs;
///
/// let coeffs = chebyshev_sin_coeffs(15);
/// assert_eq!(coeffs.len(), 16);
/// ```
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
///
/// # Arguments
///
/// * `degree` - Maximum degree of polynomial
///
/// # Returns
///
/// Vector of coefficients where result[i] is coefficient for x^i
///
/// # Example
///
/// ```
/// use ga_engine::clifford_fhe_v3::bootstrapping::taylor_sin_coeffs;
///
/// let coeffs = taylor_sin_coeffs(7);
/// // sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040
/// assert!((coeffs[1] - 1.0).abs() < 1e-10);  // x term
/// assert!((coeffs[3] + 1.0/6.0).abs() < 1e-10);  // -x³/6 term
/// ```
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
///
/// # Arguments
///
/// * `n` - Input value
///
/// # Returns
///
/// n! as f64
///
/// # Example
///
/// ```
/// # use ga_engine::clifford_fhe_v3::bootstrapping::sin_approx::factorial;
/// assert_eq!(factorial(5), 120.0);
/// ```
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
///
/// Uses Horner's method for numerical stability.
///
/// # Arguments
///
/// * `coeffs` - Polynomial coefficients [c0, c1, ..., cn]
/// * `x` - Point at which to evaluate
///
/// # Returns
///
/// Value of polynomial at x
///
/// # Example
///
/// ```
/// use ga_engine::clifford_fhe_v3::bootstrapping::eval_polynomial;
///
/// // Evaluate p(x) = 1 + 2x + 3x²
/// let coeffs = vec![1.0, 2.0, 3.0];
/// let result = eval_polynomial(&coeffs, 2.0);
/// assert_eq!(result, 1.0 + 2.0*2.0 + 3.0*4.0);  // 17.0
/// ```
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

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(1), 1.0);
        assert_eq!(factorial(5), 120.0);
        assert_eq!(factorial(7), 5040.0);
    }

    #[test]
    fn test_taylor_sin_coeffs() {
        let coeffs = taylor_sin_coeffs(7);
        assert_eq!(coeffs.len(), 8);

        // Check specific coefficients
        assert!((coeffs[0] - 0.0).abs() < 1e-10, "Constant term should be 0");
        assert!((coeffs[1] - 1.0).abs() < 1e-10, "x term should be 1");
        assert!((coeffs[2] - 0.0).abs() < 1e-10, "x² term should be 0");
        assert!((coeffs[3] + 1.0/6.0).abs() < 1e-10, "x³ term should be -1/6");
        assert!((coeffs[4] - 0.0).abs() < 1e-10, "x⁴ term should be 0");
        assert!((coeffs[5] - 1.0/120.0).abs() < 1e-10, "x⁵ term should be 1/120");
    }

    #[test]
    fn test_eval_polynomial() {
        // Test p(x) = 1 + 2x + 3x²
        let coeffs = vec![1.0, 2.0, 3.0];
        let result = eval_polynomial(&coeffs, 2.0);
        assert_eq!(result, 17.0);  // 1 + 4 + 12

        // Test p(x) = 0
        let coeffs = vec![0.0, 0.0, 0.0];
        let result = eval_polynomial(&coeffs, 5.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_taylor_sin_accuracy() {
        let coeffs = taylor_sin_coeffs(31);

        // Test on [-π, π]
        let test_points = vec![
            0.0,
            PI / 4.0,
            PI / 2.0,
            3.0 * PI / 4.0,
            PI,
            -PI / 4.0,
            -PI / 2.0,
        ];

        for x in test_points {
            let approx = eval_polynomial(&coeffs, x);
            let exact = x.sin();
            let error = (approx - exact).abs();

            println!("sin({:.4}) = {:.6} (approx: {:.6}, error: {:.6})",
                     x, exact, approx, error);

            assert!(error < 1e-6, "Taylor approximation error too large: {} at x={}", error, x);
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

    #[test]
    #[should_panic(expected = "Need at least degree 5")]
    fn test_chebyshev_degree_too_small() {
        chebyshev_sin_coeffs(3);
    }

    #[test]
    #[should_panic(expected = "Sine is odd function")]
    fn test_chebyshev_even_degree() {
        chebyshev_sin_coeffs(8);
    }
}
