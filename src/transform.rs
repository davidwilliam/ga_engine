//! Semantic adapters and canonical use-cases: 3D point rotation via classical matrix vs. GA rotor

use crate::ga::geometric_product_full;
use wide::f64x4;

/// 3D vector type
#[derive(Copy, Clone, Debug, PartialEq)]
/// A simple 3D point or vector
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

    /// Vector norm (length)
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Return a normalized copy of this vector
    pub fn normalized(&self) -> Self {
        let n = self.norm();
        Vec3::new(self.x / n, self.y / n, self.z / n)
    }

    /// Cross product
    pub fn cross(&self, other: &Vec3) -> Self {
        Vec3::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Multiply by scalar
    pub fn mul_scalar(&self, s: f64) -> Self {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: Vec3) -> Vec3 {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Add<&Vec3> for Vec3 {
    type Output = Vec3;
    fn add(self, rhs: &Vec3) -> Vec3 {
        Vec3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

/// Apply a 3×3 matrix (row-major) to a Vec3
pub fn apply_matrix3(m: &[f64; 9], v: &Vec3) -> Vec3 {
    Vec3::new(
        m[0] * v.x + m[1] * v.y + m[2] * v.z,
        m[3] * v.x + m[4] * v.y + m[5] * v.z,
        m[6] * v.x + m[7] * v.y + m[8] * v.z,
    )
}

/// GA rotor for 3D rotations
pub struct Rotor3 {
    mv: [f64; 8],   // multivector components
    axis: Vec3,     // unit rotation axis
    w: f64,         // scalar part (cos half-angle)
    s: f64,         // sin half-angle
}

impl Rotor3 {
    /// Construct a rotor from an axis (Vec3) and angle (radians).
    pub fn from_axis_angle(axis: &Vec3, angle: f64) -> Self {
        let half = angle * 0.5;
        let w = half.cos();
        let s = half.sin();
        let axis_norm = axis.normalized();
        // Initialize GA multivector: scalar + bivector parts
        let mut mv = [0.0; 8];
        mv[0] = w;
        mv[4] = -axis_norm.x * s; // e23
        mv[5] = -axis_norm.y * s; // e31
        mv[6] = -axis_norm.z * s; // e12
        Self { mv, axis: axis_norm, w, s }
    }

    /// Rotate a Vec3 using the GA sandwich product: r * v * r⁻¹.
    pub fn rotate(&self, v: &Vec3) -> Vec3 {
        // Promote to multivector
        let mut v_mv = [0.0; 8];
        v_mv[1] = v.x;
        v_mv[2] = v.y;
        v_mv[3] = v.z;

        // r * v
        let mut tmp = [0.0; 8];
        geometric_product_full(&self.mv, &v_mv, &mut tmp);

        // r_conj = r⁻¹
        let mut mv_conj = self.mv;
        mv_conj[4] = -mv_conj[4];
        mv_conj[5] = -mv_conj[5];
        mv_conj[6] = -mv_conj[6];

        // (r * v) * r⁻¹
        let mut res_mv = [0.0; 8];
        geometric_product_full(&tmp, &mv_conj, &mut res_mv);

        Vec3::new(res_mv[1], res_mv[2], res_mv[3])
    }

    /// Fast rotate using chained FMA operations
    #[inline(always)]
    pub fn rotate_fast(&self, v: &Vec3) -> Vec3 {
        let ax = self.axis.x;
        let ay = self.axis.y;
        let az = self.axis.z;
        let vx = v.x;
        let vy = v.y;
        let vz = v.z;
        // t = axis × v
        let tx = ay * vz - az * vy;
        let ty = az * vx - ax * vz;
        let tz = ax * vy - ay * vx;
        // u = axis × t
        let ux = ay * tz - az * ty;
        let uy = az * tx - ax * tz;
        let uz = ax * ty - ay * tx;
        let k1 = 2.0 * self.w * self.s;
        let k2 = 2.0 * self.s * self.s;
        // FMA chains
        let x = k2.mul_add(ux, k1.mul_add(tx, vx));
        let y = k2.mul_add(uy, k1.mul_add(ty, vy));
        let z = k2.mul_add(uz, k1.mul_add(tz, vz));
        Vec3::new(x, y, z)
    }

    /// SIMD rotate 4 Vec3s in parallel using wide::f64x4
    #[inline(always)]
    pub fn rotate_simd(&self, vs: &[Vec3; 4]) -> [Vec3; 4] {
        // Broadcast axis components into lanes
        let ax = f64x4::splat(self.axis.x);
        let ay = f64x4::splat(self.axis.y);
        let az = f64x4::splat(self.axis.z);
        // Load input Vec3s into SIMD lanes
        let vx = f64x4::from([vs[0].x, vs[1].x, vs[2].x, vs[3].x]);
        let vy = f64x4::from([vs[0].y, vs[1].y, vs[2].y, vs[3].y]);
        let vz = f64x4::from([vs[0].z, vs[1].z, vs[2].z, vs[3].z]);
        // Compute t = axis × v
        let tx = ay * vz - az * vy;
        let ty = az * vx - ax * vz;
        let tz = ax * vy - ay * vx;
        // Compute u = axis × t
        let ux = ay * tz - az * ty;
        let uy = az * tx - ax * tz;
        let uz = ax * ty - ay * tx;
        // Broadcast scalar coefficients
        let k1 = f64x4::splat(2.0 * self.w * self.s);
        let k2 = f64x4::splat(2.0 * self.s * self.s);
        // Chain FMAs: result lanes
        let rx = k2.mul_add(ux, k1.mul_add(tx, vx));
        let ry = k2.mul_add(uy, k1.mul_add(ty, vy));
        let rz = k2.mul_add(uz, k1.mul_add(tz, vz));
        // Extract back to scalar Vec3s
        let arrx = rx.to_array();
        let arry = ry.to_array();
        let arrz = rz.to_array();

        [
            Vec3::new(arrx[0], arry[0], arrz[0]),
            Vec3::new(arrx[1], arry[1], arrz[1]),
            Vec3::new(arrx[2], arry[2], arrz[2]),
            Vec3::new(arrx[3], arry[3], arrz[3]),
        ]
    }

    /// SIMD rotate 8 Vec3s in parallel by doing two 4-lane SIMD passes
    #[inline(always)]
    pub fn rotate_simd8(&self, vs: &[Vec3; 8]) -> [Vec3; 8] {
        // First 4 lanes
        let chunk0 = [vs[0], vs[1], vs[2], vs[3]];
        let r0 = self.rotate_simd(&chunk0);

        // Next 4 lanes
        let chunk1 = [vs[4], vs[5], vs[6], vs[7]];
        let r1 = self.rotate_simd(&chunk1);

        // Stitch results back together
        [
            r0[0], r0[1], r0[2], r0[3],
            r1[0], r1[1], r1[2], r1[3],
        ]
    }
}