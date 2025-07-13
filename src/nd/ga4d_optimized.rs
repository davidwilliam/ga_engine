//! Optimized 4D Geometric Algebra operations using compile-time lookup tables.
//!
//! This implementation uses a compile-time-generated lookup table similar to the 3D version
//! to perform the full 16×16 multivector geometric product efficiently.

use crate::nd::types::Scalar;

/// Bitmask for each 4D GA basis blade: [1, e1, e2, e3, e4, e12, e13, e14, e23, e24, e34, e123, e124, e134, e234, e1234]
const BLADE_MASKS_4D: [u8; 16] = [
    0b0000, // 1 (scalar)
    0b0001, // e1
    0b0010, // e2
    0b0011, // e12
    0b0100, // e3
    0b0101, // e13
    0b0110, // e23
    0b0111, // e123
    0b1000, // e4
    0b1001, // e14
    0b1010, // e24
    0b1011, // e124
    0b1100, // e34
    0b1101, // e134
    0b1110, // e234
    0b1111, // e1234
];

/// Mapping from bitmask back to blade index in the multivector array.
const MASK2INDEX_4D: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Compute the (sign, index) pair for blade i × blade j in a const context for 4D GA.
const fn sign_and_index_4d(i: usize, j: usize) -> (Scalar, usize) {
    let mi = BLADE_MASKS_4D[i];
    let mj = BLADE_MASKS_4D[j];
    let k_mask = mi ^ mj;
    let k = MASK2INDEX_4D[k_mask as usize];

    // Count bit swaps to determine sign
    let mut sgn = 1i32;
    let mut bit = 0;
    while bit < 4 {
        if ((mi >> bit) & 1) == 1 {
            let mut lower = mj & ((1 << bit) - 1);
            let mut cnt = 0u8;
            while lower != 0 {
                cnt = cnt.wrapping_add(lower & 1);
                lower >>= 1;
            }
            if (cnt & 1) == 1 {
                sgn = -sgn;
            }
        }
        bit += 1;
    }
    (sgn as Scalar, k)
}

/// Build the full table of blade-pair products at compile time for 4D GA.
const fn make_gp_pairs_4d() -> [(usize, usize, Scalar, usize); 256] {
    let mut table = [(0, 0, 0.0, 0); 256];
    let mut idx = 0;
    while idx < 256 {
        let i = idx / 16;
        let j = idx % 16;
        let (sign, k) = sign_and_index_4d(i, j);
        table[idx] = (i, j, sign, k);
        idx += 1;
    }
    table
}

/// Lookup table of all 16×16 blade-pair products for 4D GA: (i, j, sign, k).
const GP_PAIRS_4D: [(usize, usize, Scalar, usize); 256] = make_gp_pairs_4d();

/// Optimized 4D multivector type with compile-time lookup table.
#[derive(Debug, Clone, PartialEq)]
pub struct Multivector4D {
    /// 16 components: [scalar, e1, e2, e12, e3, e13, e23, e123, e4, e14, e24, e124, e34, e134, e234, e1234]
    pub data: [Scalar; 16],
}

impl Multivector4D {
    /// Construct from a 16-element array.
    pub fn new(data: [Scalar; 16]) -> Self {
        Self { data }
    }

    /// Construct from a Vec (must be length 16).
    pub fn from_vec(data: Vec<Scalar>) -> Self {
        assert_eq!(data.len(), 16, "4D multivector requires 16 components");
        let mut arr = [0.0; 16];
        arr.copy_from_slice(&data);
        Self { data: arr }
    }

    /// The zero multivector (all components zero).
    pub fn zero() -> Self {
        Self { data: [0.0; 16] }
    }

    /// Fast geometric product using compile-time lookup table.
    #[inline(always)]
    pub fn gp(&self, other: &Self) -> Self {
        let mut out = [0.0; 16];

        // Single pass over all precomputed blade products
        let mut idx = 0;
        while idx < 256 {
            let (i, j, sign, k) = GP_PAIRS_4D[idx];
            out[k] += sign * self.data[i] * other.data[j];
            idx += 1;
        }

        Self { data: out }
    }

    /// Convert to Vec for compatibility with generic implementation.
    pub fn to_vec(&self) -> Vec<Scalar> {
        self.data.to_vec()
    }
}

impl Default for Multivector4D {
    fn default() -> Self {
        Self::zero()
    }
}

impl std::ops::Add for Multivector4D {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut result = [0.0; 16];
        for i in 0..16 {
            result[i] = self.data[i] + rhs.data[i];
        }
        Self { data: result }
    }
}

impl std::ops::Sub for Multivector4D {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut result = [0.0; 16];
        for i in 0..16 {
            result[i] = self.data[i] - rhs.data[i];
        }
        Self { data: result }
    }
}

impl std::ops::Mul<Scalar> for Multivector4D {
    type Output = Self;
    fn mul(self, s: Scalar) -> Self {
        let mut result = [0.0; 16];
        for i in 0..16 {
            result[i] = self.data[i] * s;
        }
        Self { data: result }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multivector4d_creation() {
        let mv = Multivector4D::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
        assert_eq!(mv.data[0], 1.0);  // scalar
        assert_eq!(mv.data[1], 2.0);  // e1
        assert_eq!(mv.data[15], 16.0); // e1234
    }

    #[test]
    fn test_geometric_product_4d() {
        let a = Multivector4D::new([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let b = Multivector4D::new([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        
        let result = a.gp(&b);
        
        // (1 + e1) * (1 + e2) = 1 + e1 + e2 + e1*e2 = 1 + e1 + e2 + e12
        assert_eq!(result.data[0], 1.0);  // scalar
        assert_eq!(result.data[1], 1.0);  // e1
        assert_eq!(result.data[2], 1.0);  // e2
        assert_eq!(result.data[3], 1.0);  // e12
    }

    #[test]
    fn test_zero_multivector() {
        let zero = Multivector4D::zero();
        assert!(zero.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_addition() {
        let a = Multivector4D::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
        let b = Multivector4D::new([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        
        let result = a + b;
        assert_eq!(result.data[0], 2.0);
        assert_eq!(result.data[1], 3.0);
        assert_eq!(result.data[15], 17.0);
    }
} 