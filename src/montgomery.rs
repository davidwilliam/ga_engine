//! Montgomery reduction for modular arithmetic
//!
//! Montgomery reduction replaces expensive division with multiplication + shifts.
//! Used by Kyber-512 for q=3329.
//!
//! Theory:
//! - Work in Montgomery form: x̄ = xR mod q (where R = 2^16)
//! - Multiplication: x̄ × ȳ → MontgomeryReduce(x̄ × ȳ) = x̄ȳ/R = x̄y̅
//! - Benefit: Division by R is a right shift (>>16), avoiding expensive % operator

/// Montgomery reduction context for q=3329
///
/// Precomputes constants for efficient Montgomery reduction:
/// - R = 2^16 = 65536 (Montgomery constant)
/// - R^2 mod q (for converting to Montgomery form)
/// - q_inv = -q^(-1) mod R (for Montgomery reduction)
#[derive(Debug, Clone)]
pub struct MontgomeryContext {
    pub q: i64,
    pub r: i64,           // R = 2^16 = 65536
    pub r2_mod_q: i64,    // R^2 mod q (for to_montgomery)
    pub q_inv: i64,       // -q^(-1) mod R (for reduction)
}

impl MontgomeryContext {
    /// Create Montgomery context for Clifford-LWE (q=3329, same as Kyber)
    pub fn new_clifford_lwe() -> Self {
        Self::new(3329)
    }

    /// Create Montgomery context for given modulus q
    ///
    /// Computes:
    /// - R = 2^16
    /// - R^2 mod q
    /// - q' = -q^(-1) mod R
    pub fn new(q: i64) -> Self {
        let r = 65536i64; // 2^16

        // Compute R^2 mod q
        let r2_mod_q = ((r as i128 * r as i128) % q as i128) as i64;

        // Compute q_inv = -q^(-1) mod R using extended Euclidean algorithm
        let q_inv = Self::compute_q_inv(q, r);

        Self {
            q,
            r,
            r2_mod_q,
            q_inv,
        }
    }

    /// Compute q' = -q^(-1) mod R using extended Euclidean algorithm
    ///
    /// We need q' such that: q × q' ≡ -1 (mod R)
    /// Equivalently: q × q' + R × k = -1 for some k
    fn compute_q_inv(q: i64, r: i64) -> i64 {
        // Extended Euclidean algorithm to find q^(-1) mod R
        let mut t = 0i64;
        let mut new_t = 1i64;
        let mut rem = r;
        let mut new_rem = q;

        while new_rem != 0 {
            let quotient = rem / new_rem;
            let tmp_t = new_t;
            new_t = t - quotient * new_t;
            t = tmp_t;

            let tmp_r = new_rem;
            new_rem = rem - quotient * new_rem;
            rem = tmp_r;
        }

        if rem > 1 {
            panic!("q={} is not invertible mod R={}", q, r);
        }

        // t is now q^(-1) mod R, we need -q^(-1) mod R
        let q_inv_positive = if t < 0 { t + r } else { t };
        let q_inv_negative = (-q_inv_positive).rem_euclid(r);

        q_inv_negative
    }

    /// Convert value to Montgomery form: x → xR mod q
    ///
    /// Uses: to_montgomery(x) = MontgomeryReduce(x × R^2 mod q)
    #[inline]
    pub fn to_montgomery(&self, x: i64) -> i64 {
        // Normalize x to [0, q)
        let x_normalized = ((x % self.q) + self.q) % self.q;

        // Compute x × R^2 mod q, then Montgomery reduce
        let product = (x_normalized as i128 * self.r2_mod_q as i128) % self.q as i128;
        self.montgomery_reduce(product as i64)
    }

    /// Convert from Montgomery form: x̄ → x
    ///
    /// Uses: from_montgomery(x̄) = MontgomeryReduce(x̄)
    #[inline]
    pub fn from_montgomery(&self, x_bar: i64) -> i64 {
        self.montgomery_reduce(x_bar)
    }

    /// Montgomery reduction: T → T/R mod q
    ///
    /// Given T (possibly large), compute T/R mod q efficiently.
    ///
    /// Algorithm (from Kyber reference implementation):
    /// 1. m = (T × q') mod R    [low 16 bits of T × q']
    /// 2. t = (T + m × q) / R   [exact division, R divides (T + m×q)]
    /// 3. if t >= q: t -= q
    /// 4. return t
    ///
    /// Why it works:
    /// - T + m×q ≡ T + (T×q')×q ≡ T(1 + q×q') ≡ T(1 - 1) ≡ 0 (mod R)
    /// - So (T + m×q)/R is exact (no remainder)
    /// - (T + m×q)/R ≡ T/R (mod q) because m×q ≡ 0 (mod q)
    #[inline]
    pub fn montgomery_reduce(&self, t: i64) -> i64 {
        // Compute m = (T × q') mod R
        // Since R = 2^16, "mod R" means taking low 16 bits
        let m = ((t as i128 * self.q_inv as i128) & 0xFFFF) as i64;

        // Compute (T + m × q) / R
        // Division by R = 2^16 is a right shift by 16 bits
        let t_shifted = ((t as i128 + m as i128 * self.q as i128) >> 16) as i64;

        // Conditional subtraction (constant-time friendly)
        // If result >= q, subtract q
        if t_shifted >= self.q {
            t_shifted - self.q
        } else if t_shifted < 0 {
            // Handle negative results (can occur with lazy reduction)
            t_shifted + self.q
        } else {
            t_shifted
        }
    }

    /// Multiply two values in Montgomery form: x̄ × ȳ → x̄y̅
    ///
    /// This is the core operation that benefits from Montgomery reduction!
    #[inline]
    pub fn mul_montgomery(&self, x_bar: i64, y_bar: i64) -> i64 {
        let product = x_bar as i128 * y_bar as i128;
        self.montgomery_reduce(product as i64)
    }

    /// Add two values in Montgomery form: x̄ + ȳ → x̄+ȳ (mod q)
    #[inline]
    pub fn add_montgomery(&self, x_bar: i64, y_bar: i64) -> i64 {
        let sum = x_bar + y_bar;
        if sum >= self.q {
            sum - self.q
        } else {
            sum
        }
    }

    /// Subtract in Montgomery form: x̄ - ȳ → x̄-ȳ (mod q)
    #[inline]
    pub fn sub_montgomery(&self, x_bar: i64, y_bar: i64) -> i64 {
        let diff = x_bar - y_bar;
        if diff < 0 {
            diff + self.q
        } else {
            diff
        }
    }

    /// Scalar multiplication in Montgomery form: scalar × x̄
    ///
    /// Note: If scalar is not in Montgomery form, use:
    /// scalar_mul(s, x̄) = to_montgomery(s) × x̄ (in Montgomery)
    #[inline]
    pub fn scalar_mul_montgomery(&self, scalar_bar: i64, x_bar: i64) -> i64 {
        self.mul_montgomery(scalar_bar, x_bar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_montgomery_context_creation() {
        let ctx = MontgomeryContext::new_clifford_lwe();

        assert_eq!(ctx.q, 3329);
        assert_eq!(ctx.r, 65536);

        // Verify q_inv: q × q_inv ≡ -1 (mod R)
        let product = (ctx.q as i128 * ctx.q_inv as i128) % ctx.r as i128;
        let expected = (-1i128).rem_euclid(ctx.r as i128);
        assert_eq!(product, expected, "q × q_inv should be -1 mod R");
    }

    #[test]
    fn test_to_from_montgomery() {
        let ctx = MontgomeryContext::new_clifford_lwe();

        // Test round-trip: x → to_montgomery → from_montgomery → x
        for x in [0, 1, 42, 1000, 3328, -1, -100] {
            let x_normalized = ((x % ctx.q) + ctx.q) % ctx.q;
            let x_bar = ctx.to_montgomery(x);
            let x_recovered = ctx.from_montgomery(x_bar);

            assert_eq!(
                x_recovered, x_normalized,
                "Round-trip failed for x={}: got {}, expected {}",
                x, x_recovered, x_normalized
            );
        }
    }

    #[test]
    fn test_montgomery_multiplication() {
        let ctx = MontgomeryContext::new_clifford_lwe();

        // Test: (a × b) mod q == from_montgomery(to_montgomery(a) × to_montgomery(b))
        let test_cases = [
            (5, 7, (5 * 7) % 3329),
            (100, 200, (100 * 200) % 3329),
            (1000, 2000, (1000 * 2000) % 3329),
            (3328, 3328, ((3328 as i64 * 3328) % 3329)),
        ];

        for (a, b, expected) in test_cases {
            let a_bar = ctx.to_montgomery(a);
            let b_bar = ctx.to_montgomery(b);
            let product_bar = ctx.mul_montgomery(a_bar, b_bar);
            let product = ctx.from_montgomery(product_bar);

            assert_eq!(
                product, expected,
                "Montgomery multiplication failed: {} × {} mod {} = {}, expected {}",
                a, b, ctx.q, product, expected
            );
        }
    }

    #[test]
    fn test_montgomery_addition() {
        let ctx = MontgomeryContext::new_clifford_lwe();

        let test_cases = [
            (5, 7, 12),
            (3300, 100, 71), // Wraparound: (3300 + 100) mod 3329 = 71
            (1000, 2000, 3000),
        ];

        for (a, b, expected) in test_cases {
            let a_bar = ctx.to_montgomery(a);
            let b_bar = ctx.to_montgomery(b);
            let sum_bar = ctx.add_montgomery(a_bar, b_bar);
            let sum = ctx.from_montgomery(sum_bar);

            assert_eq!(
                sum, expected,
                "Montgomery addition failed: {} + {} mod {} = {}, expected {}",
                a, b, ctx.q, sum, expected
            );
        }
    }

    #[test]
    fn test_montgomery_subtraction() {
        let ctx = MontgomeryContext::new_clifford_lwe();

        let test_cases = [
            (10, 5, 5),
            (5, 10, 3324), // Negative wraparound: (5 - 10) mod 3329 = 3324
            (3329, 1, 3328),
        ];

        for (a, b, expected) in test_cases {
            let a_bar = ctx.to_montgomery(a);
            let b_bar = ctx.to_montgomery(b);
            let diff_bar = ctx.sub_montgomery(a_bar, b_bar);
            let diff = ctx.from_montgomery(diff_bar);

            assert_eq!(
                diff, expected,
                "Montgomery subtraction failed: {} - {} mod {} = {}, expected {}",
                a, b, ctx.q, diff, expected
            );
        }
    }

    #[test]
    fn test_montgomery_reduction_correctness() {
        let ctx = MontgomeryContext::new_clifford_lwe();

        // MontgomeryReduce(T) should equal T/R mod q
        // Test with known values
        let r = ctx.r;
        let q = ctx.q;

        // Example: T = 100 × R
        // MontgomeryReduce(100R) = 100R/R mod q = 100 mod q = 100
        let t = 100 * r;
        let result = ctx.montgomery_reduce(t);
        assert_eq!(result, 100 % q);

        // Example: T = q × R
        // MontgomeryReduce(qR) = qR/R mod q = q mod q = 0
        let t = q * r;
        let result = ctx.montgomery_reduce(t);
        assert_eq!(result, 0);
    }
}
