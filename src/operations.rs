use crate::{Handle, Result, types::Operation};

extern "C" {
    fn rocblas_sgemm(handle: u64, transa: u8, transb: u8,
                     m: i64, n: i64, k: i64, alpha: *const f32,
                     da: *const f32, lda: i64, db: *const f32, ldb: i64,
                     beta: *const f32, dc: *mut f32, ldc: i64) -> i32;
    fn rocblas_dgemm(handle: u64, transa: u8, transb: u8,
                     m: i64, n: i64, k: i64, alpha: *const f64,
                     da: *const f64, lda: i64, db: *const f64, ldb: i64,
                     beta: *const f64, dc: *mut f64, ldc: i64) -> i32;
    fn rocblas_sgemv(handle: u64, trans: u8, m: i64, n: i64,
                     alpha: *const f32, da: *const f32, lda: i64,
                     dx: *const f32, incx: i64, beta: *const f32,
                     dy: *mut f32, incy: i64) -> i32;
    fn rocblas_sdot(handle: u64, n: i64, x: *const f32, incx: i64,
                    y: *const f32, incy: i64, result: *mut f32) -> i32;
    fn rocblas_saxpy(handle: u64, n: i64, alpha: *const f32,
                     x: *const f32, incx: i64, y: *mut f32, incy: i64) -> i32;
    fn rocblas_sscal(handle: u64, n: i64, alpha: *const f32,
                     x: *mut f32, incx: i64) -> i32;
}

/// Level 3 BLAS: C = alpha * op(A) * op(B) + beta * C (single precision)
pub unsafe fn sgemm(
    handle: &Handle,
    transa: Operation, transb: Operation,
    m: u64, n: u64, k: u64,
    alpha: f32, da: *const f32, lda: u64,
    db: *const f32, ldb: u64, beta: f32,
    dc: *mut f32, ldc: u64,
) -> Result<()> {
    let status = rocblas_sgemm(
        handle.raw(), transa.to_char() as u8, transb.to_char() as u8,
        m as i64, n as i64, k as i64,
        &alpha as *const f32, da, lda as i64,
        db, ldb as i64, &beta as *const f32,
        dc, ldc as i64,
    );
    if status != 0 { return Err(crate::RocblasError::from(status)); }
    Ok(())
}

/// Level 2 BLAS: y = alpha * op(A) * x + beta * y (single precision)
pub unsafe fn sgemv(
    handle: &Handle, trans: Operation,
    m: u64, n: u64, alpha: f32,
    da: *const f32, lda: u64,
    dx: *const f32, incx: u64, beta: f32,
    dy: *mut f32, incy: u64,
) -> Result<()> {
    let status = rocblas_sgemv(
        handle.raw(), trans.to_char() as u8, m as i64, n as i64,
        &alpha as *const f32, da, lda as i64,
        dx, incx as i64, &beta as *const f32,
        dy, incy as i64,
    );
    if status != 0 { return Err(crate::RocblasError::from(status)); }
    Ok(())
}

/// Level 1 BLAS: dot product
pub unsafe fn sdot(
    handle: &Handle, n: u64,
    x: *const f32, incx: u64,
    y: *const f32, incy: u64,
    result: *mut f32,
) -> Result<()> {
    let status = rocblas_sdot(handle.raw(), n as i64, x, incx as i64, y, incy as i64, result);
    if status != 0 { return Err(crate::RocblasError::from(status)); }
    Ok(())
}

/// Level 1 BLAS: y = alpha*x + y
pub unsafe fn saxpy(
    handle: &Handle, n: u64, alpha: f32,
    x: *const f32, incx: u64,
    y: *mut f32, incy: u64,
) -> Result<()> {
    let status = rocblas_saxpy(handle.raw(), n as i64, &alpha as *const f32, x, incx as i64, y, incy as i64);
    if status != 0 { return Err(crate::RocblasError::from(status)); }
    Ok(())
}

/// Level 1 BLAS: x = alpha * x
pub unsafe fn sscal(
    handle: &Handle, n: u64, alpha: f32,
    x: *mut f32, incx: u64,
) -> Result<()> {
    let status = rocblas_sscal(handle.raw(), n as i64, &alpha as *const f32, x, incx as i64);
    if status != 0 { return Err(crate::RocblasError::from(status)); }
    Ok(())
}
