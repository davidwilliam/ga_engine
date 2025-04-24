#![doc = include_str!("../README.md")]

pub mod classical;
pub mod vector;
pub mod bivector;
pub mod multivector;
pub mod rotor;
pub mod transform;
pub mod ga;
pub mod prelude;

pub use vector::{Vec3, Rounded};
pub use bivector::Bivector3;
pub use multivector::Multivector3;
pub use rotor::Rotor3;
pub use transform::apply_matrix3;
pub use classical::multiply_matrices;
pub use ga::{geometric_product, geometric_product_full};
