/// Data type for BLAS operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    F32,
    F64,
    F16,
    BF16,
    I8,
}

/// Matrix operation type
#[derive(Debug, Clone, Copy)]
pub enum Operation {
    None,  // No transpose
    Transpose,
    ConjugateTranspose,
}

impl Operation {
    pub fn to_char(self) -> char {
        match self {
            Self::None => 'N',
            Self::Transpose => 'T',
            Self::ConjugateTranspose => 'C',
        }
    }
}

/// Matrix side for symmetric operations
#[derive(Debug, Clone, Copy)]
pub enum Side { Left, Right }

/// Triangle fill mode
#[derive(Debug, Clone, Copy)]
pub enum Uplo { Upper, Lower }

/// Diagonal type
#[derive(Debug, Clone, Copy)]
pub enum Diag { NonUnit, Unit }

/// Fill mode for packed storage
#[derive(Debug, Clone, Copy)]
pub enum FillMode { Upper, Lower, Full }
