//! rocblas-lite: Safe Rust bindings for AMD rocBLAS
//!
//! Provides Level 1/2/3 BLAS operations with a safe Rust API.
//!
//! # Example
//! ```no_run
//! use rocblas_lite::{Handle, Operation};
//!
//! let handle = Handle::new().unwrap();
//! let m = 1024u64;
//! let n = 1024u64;
//! let k = 1024u64;
//! // sgemm would be called here with proper GPU pointers
//! ```

pub mod error;
pub mod handle;
pub mod operations;
pub mod types;

pub use error::RocblasError;
pub use handle::Handle;
pub use types::{DataType, Operation, Side, Uplo, Diag, FillMode};

/// Result type for rocBLAS operations
pub type Result<T> = std::result::Result<T, RocblasError>;
