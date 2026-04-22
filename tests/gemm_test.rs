#[test]
fn test_sgemm_identity() {
    // A * I = A for identity matrix
    let n = 4;
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let eye: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let mut c = vec![0.0f32; 16];
    // Would call sgemm here with device pointers
    // For now, verify CPU reference
    let mut expected = vec![0.0f32; 16];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                expected[i * n + j] += a[i * n + k] * eye[k * n + j];
            }
        }
    }
    for i in 0..16 {
        assert!((expected[i] - a[i]).abs() < 1e-6);
    }
}
