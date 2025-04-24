//! Geometric‐product table dispatcher for N dimensions.
//!
//! Re-exports the `make_gp_table` const‐fn so that any N-dimensional
//! multivector can grab its own lookup table.

use crate::nd::types::Scalar;

// Build a runtime GP‐table once, for a given dimension.
pub fn make_gp_table(n: usize) -> Vec<(Scalar, usize)> {
    let m = 1 << n;
    let mut table = Vec::with_capacity(m * m);
    for i in 0..m {
        for j in 0..m {
            table.push(sign_and_index(i, j, n));
        }
    }
    table
}

/// Count out the sign and compute the output blade index.
fn sign_and_index(i: usize, j: usize, n: usize) -> (Scalar, usize) {
    let mi = i;
    let mj = j;
    let k  = mi ^ mj;
    let mut sgn = 1i32;
    for bit in 0..n {
        if ((mi >> bit) & 1) != 0 {
            let mut lower = mj & ((1 << bit) - 1);
            let mut cnt = 0usize;
            while lower != 0 {
                cnt    += lower & 1;
                lower >>= 1;
            }
            if (cnt & 1) != 0 {
                sgn = -sgn;
            }
        }
    }
    let sign: Scalar = if sgn > 0 { 1.0 } else { -1.0 };
    (sign, k)
}
