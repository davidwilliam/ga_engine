use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::vector::Vec3;
use ga_engine::Vec3Reflect;
use ga_engine::bivector::Bivector3;
use ga_engine::ops::reflection::reflect_ga;
use ga_engine::transform::apply_matrix3;

/// Create a reflection matrix across the plane normal to (1, 0, 0)
fn reflection_matrix() -> [f64; 9] {
    // Householder reflection matrix: I - 2nn^T for unit vector n = (1, 0, 0)
    [
        -1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0,
    ]
}

/// Apply 100 chained classical reflections using a matrix
fn classical_reflection_chain(v: Vec3) -> Vec3 {
    let m = reflection_matrix();
    let mut result = v;
    for _ in 0..100 {
        result = apply_matrix3(&m, result);
    }
    result
}

/// Apply 100 chained scalar-formula reflections
fn scalar_reflection_chain(v: Vec3) -> Vec3 {
    let n = Vec3::new(1.0, 0.0, 0.0); // plane normal
    let mut result = v;
    for _ in 0..100 {
        result = result.reflect_in_plane(n);
    }
    result
}

/// Apply 100 chained GA-based reflections
fn ga_reflection_chain(v: Vec3) -> Vec3 {
    let n = Vec3::new(1.0, 0.0, 0.0);
    let b = Bivector3::from_wedge(n, Vec3::new(0.0, 1.0, 0.0)); // orthogonal plane: xy
    let mut result = v;
    for _ in 0..100 {
        result = reflect_ga(result, b);
    }
    result
}

fn bench_reflection_chain(c: &mut Criterion) {
    let v = Vec3::new(2.0, 3.0, 4.0);

    c.bench_function("classical_matrix_100_reflections", |b| {
        b.iter(|| classical_reflection_chain(black_box(v)))
    });

    c.bench_function("scalar_formula_100_reflections", |b| {
        b.iter(|| scalar_reflection_chain(black_box(v)))
    });

    c.bench_function("ga_100_reflections", |b| {
        b.iter(|| ga_reflection_chain(black_box(v)))
    });
}

criterion_group!(benches, bench_reflection_chain);
criterion_main!(benches);
