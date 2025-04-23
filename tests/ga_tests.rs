use ga_engine::ga::geometric_product;

#[test]
fn scalar_vector() {
    let scalar = vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let vector = vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0];
    let result = geometric_product(&scalar, &vector);
    let expected: Vec<f64> = vector.iter().map(|x| 2.0 * x).collect();
    assert_eq!(result, expected);
}

#[test]
fn vector_vector() {
    let v1 = vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0];
    let v2 = vec![0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0];
    let result = geometric_product(&v1, &v2);
    // Dot product: 1*4 + 2*5 + 3*6 = 32
    assert_eq!(result[0], 32.0);
    // Bivector parts (e23, e31, e12)
    assert_eq!(result[4], v1[2] * v2[3] - v1[3] * v2[2]);
    assert_eq!(result[5], v1[3] * v2[1] - v1[1] * v2[3]);
    assert_eq!(result[6], v1[1] * v2[2] - v1[2] * v2[1]);
}
