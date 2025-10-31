pub mod ga;
pub mod ga4d_optimized;
pub mod gp;
pub mod multivector;
pub mod types;
pub mod vecn;

// Re-exports for easy import:
pub use multivector::Multivector;
pub use vecn::VecN;
pub use ga4d_optimized::Multivector4D;
