fn main() {
    // Platform-specific BLAS/LAPACK linking
    #[cfg(target_os = "linux")]
    {
        // Link against system OpenBLAS on Linux
        println!("cargo:rustc-link-lib=openblas");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/openblas-pthread");
    }

    #[cfg(target_os = "macos")]
    {
        // macOS uses Accelerate framework - handled by .cargo/config.toml
        // No explicit linking needed here

        // Compile MPSGraph Objective-C wrapper if mpsgraph feature is enabled
        #[cfg(feature = "mpsgraph")]
        {
            cc::Build::new()
                .file("src/gpu/backends/mpsgraph_wrapper.m")
                .flag("-fobjc-arc") // Enable ARC for automatic memory management
                .flag("-Wno-deprecated-declarations")
                .compile("mpsgraph_wrapper");

            // Link Metal frameworks
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");

            // Rerun if wrapper source changes
            println!("cargo:rerun-if-changed=src/gpu/backends/mpsgraph_wrapper.m");
        }
    }
}
