# rocblas-lite

Lightweight, safe Rust bindings for AMD rocBLAS — the GPU-accelerated BLAS library for ROCm.

## Why rocblas-lite?

- **Safe API**: No raw pointers in user code — rocblas-lite wraps unsafe FFI calls
- **Zero-cost abstraction**: Minimal overhead over direct rocBLAS calls
- **Rust ergonomics**: Builder patterns, error types, and RAII handles
- **MI300X optimized**: Targets CDNA3 matrix core instructions

## Supported Operations

| Level | Operations | Status |
|-------|-----------|--------|
| Level 1 | dot, axpy, scal, nrm2, asum, iamax | ✅ Partial |
| Level 2 | gemv, ger, symv, trsv | ✅ Partial |
| Level 3 | gemm, symm, syrk, trmm, trsm | ✅ Partial |

## Usage

```rust
use rocblas_lite::Handle;

fn main() -> rocblas_lite::Result<()> {
    let handle = Handle::new()?;
    println!("rocBLAS version: {}", Handle::version()?);
    // ... GPU operations
    Ok(())
}
```

## Requirements

- ROCm 6.0+
- `rocBLAS` package installed (`apt install rocblas`)
- Rust 1.70+

## Related Projects

Part of the [ROCm Developer Toolkit](https://github.com/huang-poi) ecosystem:
- [rocprobe](https://github.com/huang-poi/rocprobe) — GPU profiler CLI
- [hip-graph-capture](https://github.com/huang-poi/hip-graph-capture) — HIP graph capture
- [mi300x-bench](https://github.com/huang-poi/mi300x-bench) — MI300X benchmarks
- [hip-kernel-lab](https://github.com/huang-poi/hip-kernel-lab) — Kernel examples
- [rocm-devbox](https://github.com/huang-poi/rocm-devbox) — Dev environment configs

## License

MIT
