fn main() {
    // Try to find rocBLAS via pkg-config
    match pkg_config::probe_library("rocblas") {
        Ok(_) => println!("cargo:rustc-link-lib=rocblas"),
        Err(_) => {
            // Fallback: link directly
            println!("cargo:rustc-link-lib=rocblas");
            if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
                println!("cargo:rustc-link-search=native={rocm_path}/lib");
            } else {
                println!("cargo:rustc-link-search=native=/opt/rocm/lib");
            }
        }
    }
}
