// src/nd/gp_lazy.rs
//! On-the-fly geometric product for N-dimensional GA without full table.

use crate::nd::types::Scalar;

/// Compute the sign (\u00b11) and resulting blade index for blades `i * j`.
/// Blades are bitmasks 0..2^N. Result blade is `i ^ j`.
/// Sign is determined by counting swaps.
#[inline(always)]
pub fn gp_blades(i: usize, j: usize) -> (Scalar, usize) {
    let mut sign = 1.0;
    let mut mi = i;
    for b in 0..usize::BITS {
        if (mi & 1) != 0 {
            let mj = j >> (b + 1);
            if mj.count_ones() & 1 != 0 {
                sign = -sign;
            }
        }
        mi >>= 1;
        if mi == 0 {
            break;
        }
    }
    (sign, i ^ j)
}
