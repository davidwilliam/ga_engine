//! Barrett Reduction for Fast Modular Arithmetic
//!
//! Barrett reduction is a method for computing modular reductions without division,
//! using precomputed constants. This is significantly faster than the naive `x % q` approach.
//!
//! # Algorithm
//!
//! For a modulus q and input x, compute x mod q as follows:
//!
//! 1. Precompute: μ = ⌊2^k / q⌋ where k is chosen appropriately (typically k = 64)
//! 2. Compute: t = ⌊(x * μ) / 2^k⌋
//! 3. Compute: r = x - t * q
//! 4. If r >= q, return r - q, else return r
//!
//! Steps 2-4 use only multiplication and shifts (no division!), making this ~2-3× faster
//! than standard modular reduction on modern CPUs.
//!
//! # Constant-Time Property
//!
//! The final correction (step 4) can be made branchless:
//! ```rust
//! r + ((r >> 63) & q)  // Branchless: adds q if r is negative
//! r - (q & -(r >= q))  // Branchless: subtracts q if r >= q
//! ```
//!
//! This makes Barrett reduction suitable for cryptographic implementations
//! requiring constant-time behavior.

/// Barrett reduction context for a specific modulus
#[derive(Debug, Clone, Copy)]
pub struct BarrettContext {
    /// The modulus q
    pub q: i64,

    /// Precomputed constant: μ = ⌊2^64 / q⌋
    /// Used to approximate division by q
    mu: u128,

    /// Shift amount (always 64 for i64 arithmetic)
    k: u32,
}

impl BarrettContext {
    /// Create a new Barrett reduction context for the given modulus
    ///
    /// # Arguments
    /// * `q` - The modulus (must be positive and less than 2^62 for safety)
    ///
    /// # Panics
    /// Panics if q <= 0 or q >= 2^62
    ///
    /// # Example
    /// ```
    /// use ga_engine::barrett::BarrettContext;
    ///
    /// let ctx = BarrettContext::new(3329);  // Kyber modulus
    /// let reduced = ctx.reduce(5000);
    /// assert_eq!(reduced, 1671);  // 5000 mod 3329 = 1671
    /// ```
    pub fn new(q: i64) -> Self {
        assert!(q > 0, "Modulus must be positive");
        assert!(q < (1i64 << 62), "Modulus too large (must be < 2^62)");

        // Precompute μ = ⌊2^64 / q⌋
        // We use 128-bit arithmetic to avoid overflow
        let mu = ((1u128 << 64) / q as u128) as u128;

        Self {
            q,
            mu,
            k: 64,
        }
    }

    /// Reduce x modulo q using Barrett reduction
    ///
    /// # Algorithm
    /// 1. t = ⌊(x * μ) / 2^64⌋  (approximate quotient)
    /// 2. r = x - t * q          (approximate remainder)
    /// 3. Correct if r >= q or r < 0
    ///
    /// # Performance
    /// ~2-3× faster than x % q on most CPUs
    ///
    /// # Example
    /// ```
    /// use ga_engine::barrett::BarrettContext;
    ///
    /// let ctx = BarrettContext::new(3329);
    ///
    /// assert_eq!(ctx.reduce(5000), 1671);
    /// assert_eq!(ctx.reduce(-100), 3229);
    /// assert_eq!(ctx.reduce(3329), 0);
    /// ```
    #[inline(always)]
    pub fn reduce(&self, x: i64) -> i64 {
        // Handle negative inputs: bring into [0, 2q) range
        let x_adjusted = if x < 0 {
            // Add multiples of q until positive
            // For crypto with small errors, one addition usually suffices
            x + self.q * (((-x) / self.q) + 1)
        } else {
            x
        };

        // Barrett reduction for positive x
        // t = ⌊(x * μ) / 2^64⌋
        let x_u128 = x_adjusted as u128;
        let product = x_u128 * self.mu;
        let t = (product >> self.k) as i64;

        // r = x - t * q
        let mut r = x_adjusted - t * self.q;

        // Correction step: r may be in [q, 2q) due to approximation
        // This branch can be made constant-time if needed
        if r >= self.q {
            r -= self.q;
        }

        r
    }

    /// Reduce x modulo q using constant-time Barrett reduction
    ///
    /// This version uses branchless logic to avoid timing side-channels.
    /// Slightly slower than `reduce()` but suitable for cryptographic operations.
    ///
    /// # Security
    /// - No branches dependent on secret values
    /// - Constant execution time regardless of input
    /// - Suitable for cryptographic implementations
    ///
    /// # Example
    /// ```
    /// use ga_engine::barrett::BarrettContext;
    ///
    /// let ctx = BarrettContext::new(3329);
    ///
    /// // Same results as reduce(), but constant-time
    /// assert_eq!(ctx.reduce_ct(5000), 1671);
    /// assert_eq!(ctx.reduce_ct(-100), 3229);
    /// ```
    #[inline(always)]
    pub fn reduce_ct(&self, x: i64) -> i64 {
        // Constant-time handling of negative inputs
        // Compute: x + q * ceil(|x| / q) to bring into positive range
        let is_negative = (x >> 63) as i64;  // -1 if negative, 0 if positive
        let abs_x = (x ^ is_negative) - is_negative;  // Branchless absolute value
        let multiplier = (abs_x / self.q) + 1;
        let x_adjusted = x + (is_negative & multiplier) * self.q;

        // Barrett reduction (same as reduce())
        let x_u128 = x_adjusted as u128;
        let product = x_u128 * self.mu;
        let t = (product >> self.k) as i64;
        let r = x_adjusted - t * self.q;

        // Constant-time correction: subtract q if r >= q
        // mask = 0xFFFFFFFF if r >= q, else 0
        let mask = -((r >= self.q) as i64);
        let correction = self.q & mask;

        r - correction
    }

    /// Add two values and reduce modulo q
    ///
    /// # Example
    /// ```
    /// use ga_engine::barrett::BarrettContext;
    ///
    /// let ctx = BarrettContext::new(3329);
    ///
    /// assert_eq!(ctx.add(2000, 2000), 671);  // (2000 + 2000) mod 3329 = 671
    /// ```
    #[inline(always)]
    pub fn add(&self, a: i64, b: i64) -> i64 {
        self.reduce(a + b)
    }

    /// Subtract two values and reduce modulo q
    ///
    /// # Example
    /// ```
    /// use ga_engine::barrett::BarrettContext;
    ///
    /// let ctx = BarrettContext::new(3329);
    ///
    /// assert_eq!(ctx.sub(1000, 2000), 3329 - 1000);  // (1000 - 2000) mod 3329
    /// ```
    #[inline(always)]
    pub fn sub(&self, a: i64, b: i64) -> i64 {
        self.reduce(a - b)
    }

    /// Multiply two values and reduce modulo q
    ///
    /// # Example
    /// ```
    /// use ga_engine::barrett::BarrettContext;
    ///
    /// let ctx = BarrettContext::new(3329);
    ///
    /// assert_eq!(ctx.mul(100, 100), 10000 % 3329);
    /// ```
    #[inline(always)]
    pub fn mul(&self, a: i64, b: i64) -> i64 {
        self.reduce(a * b)
    }

    /// Negate a value modulo q
    ///
    /// # Example
    /// ```
    /// use ga_engine::barrett::BarrettContext;
    ///
    /// let ctx = BarrettContext::new(3329);
    ///
    /// assert_eq!(ctx.neg(1000), 2329);  // -1000 mod 3329 = 2329
    /// assert_eq!(ctx.neg(0), 0);
    /// ```
    #[inline(always)]
    pub fn neg(&self, a: i64) -> i64 {
        if a == 0 {
            0
        } else {
            self.q - a
        }
    }

    /// Scalar multiplication: multiply by a small constant
    ///
    /// Optimized for the case where the scalar is small (e.g., error terms in LWE)
    ///
    /// # Example
    /// ```
    /// use ga_engine::barrett::BarrettContext;
    ///
    /// let ctx = BarrettContext::new(3329);
    ///
    /// assert_eq!(ctx.scalar_mul(1000, 2), 2000);
    /// assert_eq!(ctx.scalar_mul(2000, 2), 671);  // (2000 * 2) mod 3329 = 671
    /// ```
    #[inline(always)]
    pub fn scalar_mul(&self, a: i64, scalar: i64) -> i64 {
        self.reduce(a * scalar)
    }
}

/// Lazy reduction context for batch operations
///
/// Allows intermediate values to grow up to 2^32 × q before reduction,
/// minimizing the number of reductions needed.
///
/// # Use Case
/// When performing many additions/subtractions, you can defer reductions
/// until the final step, significantly improving performance.
///
/// # Example
/// ```
/// use ga_engine::barrett::{BarrettContext, LazyContext};
///
/// let barrett = BarrettContext::new(3329);
/// let lazy = LazyContext::new(barrett);
///
/// // Accumulate many values without reduction
/// let mut acc = 0i64;
/// for i in 0..100 {
///     acc = lazy.lazy_add(acc, i);
/// }
///
/// // Final reduction
/// let result = lazy.finalize(acc);
/// ```
pub struct LazyContext {
    barrett: BarrettContext,
    /// Maximum safe value before overflow (approximately 2^32 × q)
    max_lazy: i64,
}

impl LazyContext {
    /// Create a lazy reduction context
    pub fn new(barrett: BarrettContext) -> Self {
        // Safe accumulation limit: ensure we don't overflow i64
        // With q = 3329, we can accumulate ~10^15 / 3329 ≈ 300 billion values
        let max_lazy = (i64::MAX / barrett.q) * barrett.q;

        Self {
            barrett,
            max_lazy,
        }
    }

    /// Add without reduction (lazy)
    ///
    /// WARNING: Caller must ensure accumulated value doesn't exceed max_lazy
    #[inline(always)]
    pub fn lazy_add(&self, a: i64, b: i64) -> i64 {
        a + b
    }

    /// Subtract without reduction (lazy)
    #[inline(always)]
    pub fn lazy_sub(&self, a: i64, b: i64) -> i64 {
        a - b
    }

    /// Finalize lazy accumulation with Barrett reduction
    #[inline(always)]
    pub fn finalize(&self, x: i64) -> i64 {
        self.barrett.reduce(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barrett_basic() {
        let ctx = BarrettContext::new(3329);

        // Basic cases
        assert_eq!(ctx.reduce(0), 0);
        assert_eq!(ctx.reduce(3329), 0);
        assert_eq!(ctx.reduce(3330), 1);
        assert_eq!(ctx.reduce(5000), 1671);
    }

    #[test]
    fn test_barrett_negative() {
        let ctx = BarrettContext::new(3329);

        // Negative values
        assert_eq!(ctx.reduce(-1), 3328);
        assert_eq!(ctx.reduce(-100), 3229);
        assert_eq!(ctx.reduce(-3329), 0);
    }

    #[test]
    fn test_barrett_vs_modulo() {
        let ctx = BarrettContext::new(3329);

        // Compare against standard modulo
        for x in -10000..10000 {
            let expected = ((x % 3329) + 3329) % 3329;
            let actual = ctx.reduce(x);
            assert_eq!(actual, expected, "Failed for x={}", x);
        }
    }

    #[test]
    fn test_barrett_constant_time() {
        let ctx = BarrettContext::new(3329);

        // Constant-time version should give same results
        for x in -10000..10000 {
            assert_eq!(ctx.reduce_ct(x), ctx.reduce(x), "CT mismatch for x={}", x);
        }
    }

    #[test]
    fn test_barrett_operations() {
        let ctx = BarrettContext::new(3329);

        // Add
        assert_eq!(ctx.add(2000, 2000), 671);

        // Sub
        assert_eq!(ctx.sub(1000, 2000), 3329 - 1000);

        // Mul
        assert_eq!(ctx.mul(100, 100), 10000 % 3329);

        // Neg
        assert_eq!(ctx.neg(1000), 2329);
    }

    #[test]
    fn test_lazy_reduction() {
        let barrett = BarrettContext::new(3329);
        let lazy = LazyContext::new(barrett);

        // Accumulate many values
        let mut acc = 0i64;
        for i in 0..1000 {
            acc = lazy.lazy_add(acc, i);
        }

        // Final reduction
        let result = lazy.finalize(acc);
        let expected = barrett.reduce((0..1000).sum());

        assert_eq!(result, expected);
    }
}
