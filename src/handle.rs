use crate::{RocblasError, Result};

extern "C" {
    fn rocblas_create_handle(handle: *mut u64) -> i32;
    fn rocblas_destroy_handle(handle: u64) -> i32;
    fn rocblas_set_stream(handle: u64, stream_id: u64) -> i32;
    fn rocblas_get_stream(handle: u64, stream_id: *mut u64) -> i32;
    fn rocblas_set_math_mode(handle: u64, mode: i32) -> i32;
    fn rocblas_get_version_string(buf: *mut u8, len: u64) -> i32;
}

/// A rocBLAS handle for managing GPU operations.
///
/// Creating a handle initializes internal state and device memory.
/// Drop automatically destroys the handle.
pub struct Handle {
    raw: u64,
}

impl Handle {
    /// Create a new rocBLAS handle.
    pub fn new() -> Result<Self> {
        let mut raw: u64 = 0;
        let status = unsafe { rocblas_create_handle(&mut raw) };
        if status != 0 {
            return Err(RocblasError::from(status));
        }
        Ok(Self { raw })
    }

    /// Get the underlying handle value.
    pub fn raw(&self) -> u64 {
        self.raw
    }

    /// Set the HIP stream for this handle.
    pub fn set_stream(&self, stream: u64) -> Result<()> {
        let status = unsafe { rocblas_set_stream(self.raw, stream) };
        if status != 0 { return Err(RocblasError::from(status)); }
        Ok(())
    }

    /// Get rocBLAS version string.
    pub fn version() -> Result<String> {
        let mut buf = [0u8; 64];
        let status = unsafe { rocblas_get_version_string(buf.as_mut_ptr(), 64) };
        if status != 0 { return Err(RocblasError::from(status)); }
        Ok(String::from_utf8_lossy(&buf).trim_end_matches(char::from(0)).to_string())
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe { rocblas_destroy_handle(self.raw); }
    }
}

unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}
