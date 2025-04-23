use ga_engine::transform::{Vec3, apply_matrix3, Rotor3};

#[test]
fn rotate_z_90_degrees() {
    let v = Vec3::new(1.0, 0.0, 0.0);
    // Classical matrix for 90° about Z
    let m: [f64;9] = [
         0.0, -1.0, 0.0,
         1.0,  0.0, 0.0,
         0.0,  0.0, 1.0,
    ];
    let v_mat = apply_matrix3(&m, &v);

    let rotor = Rotor3::from_axis_angle(&Vec3::new(0.0,0.0,1.0), std::f64::consts::FRAC_PI_2);
    let v_ga = rotor.rotate(&v);

    assert!((v_mat.x - v_ga.x).abs() < 1e-12);
    assert!((v_mat.y - v_ga.y).abs() < 1e-12);
    assert!((v_mat.z - v_ga.z).abs() < 1e-12);
}
