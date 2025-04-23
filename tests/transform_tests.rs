use ga_engine::transform::{Vec3, apply_matrix3, Rotor3};

const EPS: f64 = 1e-12;

#[test]
fn rotate_z_90_degrees() {
    // Original vector
    let v = Vec3::new(1.0, 0.0, 0.0);
    // Classical rotation matrix for +90° about Z
    let m: [f64; 9] = [
        0.0, -1.0, 0.0,
        1.0,  0.0, 0.0,
        0.0,  0.0, 1.0,
    ];
    let v_mat = apply_matrix3(&m, &v);

    // GA rotor for +90° about Z
    let rotor = Rotor3::from_axis_angle(&Vec3::new(0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2);
    let v_ga = rotor.rotate(&v);

    assert!((v_mat.x - v_ga.x).abs() < EPS, "x mismatch: {} vs {}", v_mat.x, v_ga.x);
    assert!((v_mat.y - v_ga.y).abs() < EPS, "y mismatch: {} vs {}", v_mat.y, v_ga.y);
    assert!((v_mat.z - v_ga.z).abs() < EPS, "z mismatch: {} vs {}", v_mat.z, v_ga.z);
}

#[test]
fn rotate_z_90_degrees_fast() {
    // Original vector
    let v = Vec3::new(1.0, 0.0, 0.0);
    // Classical rotation matrix for +90° about Z
    let m: [f64; 9] = [
        0.0, -1.0, 0.0,
        1.0,  0.0, 0.0,
        0.0,  0.0, 1.0,
    ];
    let v_mat = apply_matrix3(&m, &v);

    // GA rotor for +90° about Z
    let rotor = Rotor3::from_axis_angle(&Vec3::new(0.0, 0.0, 1.0), std::f64::consts::FRAC_PI_2);
    let v_fast = rotor.rotate_fast(&v);

    assert!((v_mat.x - v_fast.x).abs() < EPS, "x mismatch fast: {} vs {}", v_mat.x, v_fast.x);
    assert!((v_mat.y - v_fast.y).abs() < EPS, "y mismatch fast: {} vs {}", v_mat.y, v_fast.y);
    assert!((v_mat.z - v_fast.z).abs() < EPS, "z mismatch fast: {} vs {}", v_mat.z, v_fast.z);
}
