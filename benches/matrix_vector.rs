use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra::{Matrix2, Vector2};
use ga_engine::{multivector::Multivector3, vector::Vec3};

fn chained_matrix_vector(c: &mut Criterion) {
    // Compose two 2D rotations: 45 deg then 30 deg
    let theta1 = std::f64::consts::FRAC_PI_4; // 45 degrees
    let theta2 = std::f64::consts::PI / 6.0;   // 30 degrees

    let rot1 = Matrix2::new(theta1.cos(), -theta1.sin(), theta1.sin(), theta1.cos());
    let rot2 = Matrix2::new(theta2.cos(), -theta2.sin(), theta2.sin(), theta2.cos());

    let composed = rot2 * rot1;
    let v = Vector2::new(3.0, 4.0);

    c.bench_function("chained_matrix_vector", |b| {
        b.iter(|| {
            let r = composed * v;
            black_box(r)
        })
    });
}

fn chained_rotor_vector(c: &mut Criterion) {
    // Use GA rotors: rotor = cos(theta/2) + sin(theta/2)*e12
    let theta1 = std::f64::consts::FRAC_PI_4; // 45 degrees
    let theta2 = std::f64::consts::PI / 6.0;   // 30 degrees

    let r1 = Multivector3 {
        scalar: (theta1 / 2.0).cos(),
        vector: Vec3::default(),
        bivector: ga_engine::bivector::Bivector3::new((theta1 / 2.0).sin(), 0.0, 0.0),
        pseudo: 0.0,
    };

    let r2 = Multivector3 {
        scalar: (theta2 / 2.0).cos(),
        vector: Vec3::default(),
        bivector: ga_engine::bivector::Bivector3::new((theta2 / 2.0).sin(), 0.0, 0.0),
        pseudo: 0.0,
    };

    let rotor = r2.gp(&r1); // compose rotors
    let rotor_rev = rotor.reverse();
    let x = Multivector3::from_vector(Vec3::new(3.0, 4.0, 0.0));

    c.bench_function("chained_rotor_vector", |b| {
        b.iter(|| {
            let tmp = rotor.gp(&x);
            let rotated = tmp.gp(&rotor_rev);
            black_box(rotated)
        })
    });
}

criterion_group!(benches, chained_matrix_vector, chained_rotor_vector);
criterion_main!(benches);
