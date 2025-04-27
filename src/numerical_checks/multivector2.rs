//! Specialized fast 2D multivector for numerical experiments (DFT, FFT).
//!
//! Basis: [scalar, e1, e2, e12]
//! Components indexed by:
//! - 0 = scalar
//! - 1 = e1
//! - 2 = e2
//! - 3 = e12 (bivector)

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Multivector2 {
    pub data: [f64; 4],
}

impl Multivector2 {
    /// Create a new multivector from its four components.
    #[inline(always)]
    pub fn new(scalar: f64, e1: f64, e2: f64, e12: f64) -> Self {
        Self {
            data: [scalar, e1, e2, e12],
        }
    }

    /// Zero multivector.
    #[inline(always)]
    pub fn zero() -> Self {
        Self { data: [0.0; 4] }
    }

    /// Geometric product: self * other.
    #[inline(always)]
    pub fn gp(self, other: Self) -> Self {
        let a = self.data;
        let b = other.data;

        Self {
            data: [
                a[0] * b[0] + a[1] * b[1] + a[2] * b[2] - a[3] * b[3], // scalar part
                a[0] * b[1] + a[1] * b[0] - a[2] * b[3] + a[3] * b[2], // e1
                a[0] * b[2] + a[1] * b[3] + a[2] * b[0] - a[3] * b[1], // e2
                a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0], // e12
            ],
        }
    }
}

// Arithmetic operators
use std::ops::{Add, Mul, Sub};

impl Add for Multivector2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = [0.0; 4];
        for (i, item) in out.iter_mut().enumerate() {
            *item = self.data[i] + rhs.data[i];
        }
        Self { data: out }
    }
}

impl Sub for Multivector2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = [0.0; 4];
        for (i, item) in out.iter_mut().enumerate() {
            *item = self.data[i] - rhs.data[i];
        }
        Self { data: out }
    }
}

impl Mul<f64> for Multivector2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self::Output {
        let mut out = [0.0; 4];
        for (i, item) in out.iter_mut().enumerate() {
            *item = self.data[i] * rhs;
        }
        Self { data: out }
    }
}
