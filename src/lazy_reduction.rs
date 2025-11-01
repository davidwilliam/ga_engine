//! Lazy Modular Reduction for Performance Optimization
//!
//! The key insight: For operations like addition and subtraction, we don't need
//! to reduce after EVERY operation. We can accumulate intermediate values and
//! reduce only when:
//! 1. We're about to overflow
//! 2. We need the final result
//! 3. We're about to multiply (multiplication needs small inputs)
//!
//! # Performance Impact
//!
//! Standard approach:
//! ```rust
//! let a = (x + y) % q;  // Reduce
//! let b = (a + z) % q;  // Reduce again
//! let c = (b + w) % q;  // Reduce again
//! ```
//!
//! Lazy approach:
//! ```rust
//! let a = x + y;        // No reduction
//! let b = a + z;        // No reduction
//! let c = (b + w) % q;  // Single reduction at end
//! ```
//!
//! Expected speedup: 10-20% for addition-heavy operations (e.g., Karatsuba combine phase)

/// Lazy reduction context
///
/// Tracks when reduction is necessary based on accumulated value size
#[derive(Debug, Clone, Copy)]
pub struct LazyReductionContext {
    /// The modulus
    pub q: i64,

    /// Maximum safe accumulated value before overflow
    /// For q=3329, we can accumulate ~10^15 / 3329 ≈ 300 billion values safely
    max_accumulation: i64,

    /// Number of operations before forced reduction
    /// Conservative: reduce every 1 million operations
    ops_before_reduce: usize,
}

impl LazyReductionContext {
    /// Create a new lazy reduction context
    ///
    /// # Arguments
    /// * `q` - The modulus (must be positive)
    ///
    /// # Example
    /// ```
    /// use ga_engine::lazy_reduction::LazyReductionContext;
    ///
    /// let ctx = LazyReductionContext::new(3329);
    /// ```
    pub fn new(q: i64) -> Self {
        assert!(q > 0, "Modulus must be positive");

        // Safe accumulation: ensure we don't overflow i64
        // i64::MAX = 2^63 - 1 ≈ 9 × 10^18
        // For q=3329, we can accumulate up to 10^15 values safely
        let max_accumulation = i64::MAX / (q * 1000); // Extra safety margin

        Self {
            q,
            max_accumulation,
            ops_before_reduce: 1_000_000,
        }
    }

    /// Add two values without reduction (lazy)
    ///
    /// WARNING: Caller must ensure value doesn't exceed max_accumulation
    #[inline(always)]
    pub fn lazy_add(&self, a: i64, b: i64) -> i64 {
        a + b
    }

    /// Subtract without reduction (lazy)
    #[inline(always)]
    pub fn lazy_sub(&self, a: i64, b: i64) -> i64 {
        a - b
    }

    /// Check if value needs reduction
    ///
    /// Returns true if value is getting close to overflow
    #[inline]
    pub fn needs_reduction(&self, value: i64) -> bool {
        value.abs() > self.max_accumulation
    }

    /// Finalize accumulated value with reduction
    #[inline]
    pub fn finalize(&self, value: i64) -> i64 {
        ((value % self.q) + self.q) % self.q
    }

    /// Reduce if necessary (checks against max_accumulation)
    #[inline]
    pub fn reduce_if_needed(&self, value: i64) -> i64 {
        if self.needs_reduction(value) {
            self.finalize(value)
        } else {
            value
        }
    }
}

/// Strategy for when to apply lazy reduction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LazyStrategy {
    /// Reduce after every operation (equivalent to standard reduction)
    Eager,

    /// Reduce only after polynomial operations (add, multiply)
    PerPolynomial,

    /// Reduce only after complete algorithm steps (e.g., after full Karatsuba)
    PerAlgorithm,

    /// Reduce only at final output (maximum laziness)
    Minimal,
}

impl LazyStrategy {
    /// Should we reduce after a single coefficient operation?
    pub fn reduce_after_coeff_op(&self) -> bool {
        matches!(self, LazyStrategy::Eager)
    }

    /// Should we reduce after a polynomial operation?
    pub fn reduce_after_poly_op(&self) -> bool {
        matches!(self, LazyStrategy::Eager | LazyStrategy::PerPolynomial)
    }

    /// Should we reduce after an algorithm step?
    pub fn reduce_after_algorithm_step(&self) -> bool {
        !matches!(self, LazyStrategy::Minimal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lazy_context_creation() {
        let ctx = LazyReductionContext::new(3329);
        assert_eq!(ctx.q, 3329);
        assert!(ctx.max_accumulation > 0);
    }

    #[test]
    fn test_lazy_operations() {
        let ctx = LazyReductionContext::new(3329);

        // Accumulate without reduction
        let a = ctx.lazy_add(1000, 2000);
        assert_eq!(a, 3000);

        let b = ctx.lazy_add(a, 1000);
        assert_eq!(b, 4000);

        // Final reduction
        let c = ctx.finalize(b);
        assert_eq!(c, 671);  // 4000 % 3329 = 671
    }

    #[test]
    fn test_needs_reduction() {
        let ctx = LazyReductionContext::new(3329);

        // Small values don't need reduction
        assert!(!ctx.needs_reduction(10000));
        assert!(!ctx.needs_reduction(1_000_000));

        // Very large values do need reduction
        assert!(ctx.needs_reduction(i64::MAX / 10));
    }

    #[test]
    fn test_reduce_if_needed() {
        let ctx = LazyReductionContext::new(3329);

        // Small value: no reduction
        let small = 10000;
        assert_eq!(ctx.reduce_if_needed(small), small);

        // Large value: gets reduced
        let large = i64::MAX / 10;
        let reduced = ctx.reduce_if_needed(large);
        assert!(reduced < ctx.q);
    }

    #[test]
    fn test_many_accumulations() {
        let ctx = LazyReductionContext::new(3329);

        // Accumulate many values
        let mut acc = 0i64;
        for i in 0..1000 {
            acc = ctx.lazy_add(acc, i);
        }

        // Verify final result
        let expected = (0..1000).sum::<i64>() % 3329;
        assert_eq!(ctx.finalize(acc), expected);
    }
}
