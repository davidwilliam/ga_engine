use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::f64::consts::PI;
use wide::f64x4;

#[derive(Clone, Copy)]
struct Vec2 {
    x: f64,
    y: f64,
}

fn rotate_matrix(v: Vec2, angle: f64) -> Vec2 {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    Vec2 {
        x: v.x * cos_a - v.y * sin_a,
        y: v.x * sin_a + v.y * cos_a,
    }
}

fn rotate_rotor(v: Vec2, angle: f64) -> Vec2 {
    // Represent vector as multivector: (0, x, y, 0)
    let mv = f64x4::new([0.0, v.x, v.y, 0.0]);
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let rotor = f64x4::new([cos_a, 0.0, 0.0, sin_a]);
    let rotor_inv = f64x4::new([cos_a, 0.0, 0.0, -sin_a]);

    // Geometric product: r * v * ~r (simplified for 2D rotation)
    let temp = gp(rotor, mv);
    let result = gp(temp, rotor_inv);
    let r = result.to_array();
    Vec2 { x: r[1], y: r[2] }
}

#[inline(always)]
fn gp(a: f64x4, b: f64x4) -> f64x4 {
    let a = a.to_array();
    let b = b.to_array();
    f64x4::new([
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] - a[2] * b[3] + a[3] * b[2],
        a[0] * b[2] + a[1] * b[3] + a[2] * b[0] - a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ])
}

fn benchmark_rotation(c: &mut Criterion) {
    let input: Vec<Vec2> = (0..1000)
        .map(|i| Vec2 {
            x: i as f64 * 0.01,
            y: (i as f64 * 0.01).sin(),
        })
        .collect();
    let angle = PI / 4.0; // 45 degrees

    c.bench_function("matrix_rotate_2d", |b| {
        b.iter(|| {
            let rotated: Vec<Vec2> = input
                .iter()
                .map(|&v| rotate_matrix(black_box(v), black_box(angle)))
                .collect();
            black_box(rotated);
        })
    });

    c.bench_function("rotor_rotate_2d", |b| {
        b.iter(|| {
            let rotated: Vec<Vec2> = input
                .iter()
                .map(|&v| rotate_rotor(black_box(v), black_box(angle)))
                .collect();
            black_box(rotated);
        })
    });
}

criterion_group!(benches, benchmark_rotation);
criterion_main!(benches);
