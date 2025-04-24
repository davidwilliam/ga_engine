#![doc = include_str!("../README.md")]

pub mod classical;
pub mod vector;
pub mod transform;
pub mod rotor;
pub mod bivector;
pub mod multivector;
pub mod ga;

pub use classical::multiply_matrices;
pub use transform::apply_matrix3;
pub use vector::Vec3;
pub use bivector::Bivector3;
pub use rotor::Rotor3;
pub use multivector::Multivector3;
