// benches/motor_transform.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ga_engine::{Rotor3, Vec3, Motor3};

/// Generate a test motor: 45° rotation around Z + 1 unit translation in X
fn generate_motor() -> Motor3 {
    let axis = Vec3::new(0.0, 0.0, 1.0);
    let angle = std::f64::consts::FRAC_PI_4;
    let rotor = Rotor3::from_axis_angle(axis, angle);
    Motor3::new(rotor, Vec3::new(1.0, 0.0, 0.0))
}

/// Generate a 4×4 matrix representing the same transformation
fn generate_matrix() -> [[f64; 4]; 4] {
    let cos = std::f64::consts::FRAC_1_SQRT_2;
    let sin = std::f64::consts::FRAC_1_SQRT_2;
    [
        [ cos, -sin, 0.0, 1.0],
        [ sin,  cos, 0.0, 0.0],
        [ 0.0,  0.0, 1.0, 0.0],
        [ 0.0,  0.0, 0.0, 1.0],
    ]
}

/// Apply 100 chained matrix transforms
fn classical_chain(v: Vec3) -> Vec3 {
    let m = generate_matrix();
    let mut result = v;
    for _ in 0..100 {
        let x = result.x;
        let y = result.y;
        let z = result.z;
        result = Vec3::new(
            m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
            m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
            m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3],
        );
    }
    result
}

/// Apply 100 chained motor transforms
fn ga_chain(v: Vec3) -> Vec3 {
    let m = generate_motor();
    let mut result = v;
    for _ in 0..100 {
        result = m.transform_point(result);
    }
    result
}

fn bench_motor_vs_matrix(c: &mut Criterion) {
    let v = Vec3::new(1.0, 0.0, 0.0);

    c.bench_function("classical_100_matrix_transforms", |b| {
        b.iter(|| classical_chain(black_box(v)))
    });

    c.bench_function("ga_100_motor_transforms", |b| {
        b.iter(|| ga_chain(black_box(v)))
    });
}

criterion_group!(benches, bench_motor_vs_matrix);
criterion_main!(benches);
