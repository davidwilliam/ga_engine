//! Semantic adapters and canonical use-cases: 3D point rotation via classical matrix vs. GA rotor

/// 3D vector type
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    /// Create a new Vec3
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// Apply a 3×3 matrix (row-major) to a Vec3
pub fn apply_matrix3(m: &[f64; 9], v: &Vec3) -> Vec3 {
    let x = m[0] * v.x + m[1] * v.y + m[2] * v.z;
    let y = m[3] * v.x + m[4] * v.y + m[5] * v.z;
    let z = m[6] * v.x + m[7] * v.y + m[8] * v.z;
    Vec3::new(x, y, z)
}

use crate::ga::geometric_product_full;

/// GA rotor for 3D rotations
pub struct Rotor3 {
    mv: [f64; 8], // multivector: [scalar, e1, e2, e3, e23, e31, e12, e123]
}

impl Rotor3 {
    /// Construct a rotor from an axis (unit Vec3) and angle (radians)
    pub fn from_axis_angle(axis: &Vec3, angle: f64) -> Self {
        let half = angle * 0.5;
        let (s, c) = (half.sin(), half.cos());
        // Bivector components: e23, e31, e12
        let b23 = -axis.x * s;
        let b31 = -axis.y * s;
        let b12 = -axis.z * s;

        let mut mv = [0.0; 8];
        mv[0] = c;
        mv[4] = b23;
        mv[5] = b31;
        mv[6] = b12;
        Self { mv }
    }

    /// Reverse (conjugate) of the rotor: invert bivector parts
    pub fn conjugate(&self) -> Self {
        let mut mv_conj = self.mv;
        mv_conj[4] = -mv_conj[4];
        mv_conj[5] = -mv_conj[5];
        mv_conj[6] = -mv_conj[6];
        Self { mv: mv_conj }
    }

    /// Rotate a Vec3 using sandwich: r * v * r⁻¹
    pub fn rotate(&self, v: &Vec3) -> Vec3 {
        // Promote v to a pure-vector multivector
        let mut v_mv = [0.0; 8];
        v_mv[1] = v.x;
        v_mv[2] = v.y;
        v_mv[3] = v.z;

        // r * v
        let mut tmp = [0.0; 8];
        geometric_product_full(&self.mv, &v_mv, &mut tmp);

        // (r * v) * r⁻¹
        let r_conj = self.conjugate();
        let mut res_mv = [0.0; 8];
        geometric_product_full(&tmp, &r_conj.mv, &mut res_mv);

        // Extract vector part
        Vec3::new(res_mv[1], res_mv[2], res_mv[3])
    }
}

