use thiserror::Error;

#[derive(Error, Debug)]
pub enum RocblasError {
    #[error("rocBLAS status: {0}")]
    StatusError(i32),
    #[error("Invalid dimension: {0}")]
    InvalidDimension(String),
    #[error("Memory allocation failed: {0}")]
    AllocFailed(String),
    #[error("Handle not initialized")]
    NotInitialized,
    #[error("Invalid value: {0}")]
    InvalidValue(String),
}

impl From<i32> for RocblasError {
    fn from(status: i32) -> Self {
        match status {
            0 => RocblasError::StatusError(0), // success, shouldn't happen
            1 => RocblasError::NotInitialized,
            2 => RocblasError::AllocFailed("internal".into()),
            3 => RocblasError::InvalidValue("bad parameter".into()),
            _ => RocblasError::StatusError(status),
        }
    }
}
