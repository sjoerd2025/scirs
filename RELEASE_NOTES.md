# Release Notes - SciRS2

## 🚀 v0.1.0-rc.4 (2025-12-22) - Release Candidate 4

### 🎯 Python Integration & Performance Enhancements

This is the third Release Candidate for SciRS2 v0.1.0, introducing Python bindings, expanding SIMD optimizations, and enhancing performance across the ecosystem.

**🌟 Highlights**:
- 🐍 **Python Bindings (scirs2-python)** - Complete PyO3 integration for Python interoperability
- 🔢 **scirs2-numpy** - Enhanced ndarray 0.17 compatibility layer
- ⚡ **SIMD Expansion (Phases 25-80)** - Transcendental functions & normalization library
- 🚀 **BLAS/LAPACK Optimizations** - Platform-specific performance tuning
- 🎮 **CUDA Improvements** - Enhanced GPU acceleration capabilities
- ✅ **11,400+ Tests Passing** - Maintained production-ready quality

#### ✨ Major Features & Improvements:

**Python Integration (scirs2-python)**:
- ✅ Complete PyO3 integration for native Python bindings
  - Native bindings for all major SciRS2 modules
  - Zero-copy data sharing between Rust and Python
  - Type-safe API with Python's dynamic typing
  - PyPI packaging infrastructure ready for distribution
- ✅ scirs2-numpy: Enhanced compatibility layer for ndarray 0.17
  - Seamless rust-numpy integration
  - Type conversions and zero-copy views
  - Full NumPy array protocol support

**Advanced SIMD Library (Phases 25-80)**:
- ✅ **Phase 25**: SIMD-accelerated integer exponentiation
  - Fast integer power operations (15-25x speedup)
  - Optimized for common exponent values
- ✅ **Phases 75-80**: Complete SIMD transcendental & normalization library
  - SIMD-accelerated transcendental functions (exp, log, sin, cos, tan)
  - Batch normalization with SIMD vectorization
  - Layer normalization optimizations
  - Bandwidth-saturated implementations (10-30x speedup)
  - TLB-optimized memory access patterns
  - Cache-line aware processing for maximum throughput

**BLAS/LAPACK Optimizations**:
- ✅ Enhanced linear algebra performance with platform-specific tuning
  - macOS Accelerate framework optimizations
  - Linux OpenBLAS optimizations
- ✅ Improved batch matrix operations
- ✅ Reduced memory allocations in hot paths
- ✅ Better numerical stability algorithms

**CUDA Backend Improvements**:
- ✅ Kernel performance improvements (2-5x for specific workloads)
- ✅ Enhanced GPU memory management
- ✅ Better multi-GPU support
- ✅ Improved asynchronous execution patterns

**Testing & Quality**:
- ✅ Refactored tests for consistency and maintainability
- ✅ Enhanced benchmark implementations (softmax and others)
- ✅ Improved test coverage and reliability
- ✅ Linter fixes and code cleanup across workspace

#### 🔧 Technical Details:

**Performance Improvements**:
- SIMD transcendental functions: 10-30x speedup over scalar implementations
- Integer exponentiation: 15-25x speedup with SIMD vectorization
- BLAS/LAPACK operations: Platform-specific optimizations
- CUDA kernels: 2-5x improvement in specific workloads

**Build & Test Status**:
- ✅ All 11,407 tests passing (174 skipped)
- ✅ Zero compilation warnings across entire workspace
- ✅ Full clippy compliance maintained
- ✅ Verified on macOS (M3), Linux (x86_64), and Linux + CUDA

#### 📋 Changes Summary:
- **Added**: scirs2-python with PyO3 bindings, scirs2-numpy compatibility layer
- **Enhanced**: SIMD library (Phases 25-80), BLAS/LAPACK optimizations, CUDA backend
- **Improved**: Test consistency, benchmark implementations, code quality


#### 📦 Installation:
```toml
[dependencies]
scirs2-core = "0.1.0-rc.4"
scirs2 = "0.1.0-rc.4"
# Python bindings
scirs2-python = "0.1.0-rc.4"  # PyO3 integration
```

#### 🎯 Migration from rc.2:
- **No breaking API changes** - Fully backward compatible
- **Python bindings** - Available through scirs2-python package
- **Enhanced SIMD** - Automatic usage with `simd` feature flag
- **BLAS/LAPACK** - Automatic platform-specific optimizations
- See [CHANGELOG.md](CHANGELOG.md) for complete details

#### 📊 Release Statistics:
- **Python Integration**: scirs2-python with complete PyO3 bindings
- **SIMD Phases**: 25-80 completed (transcendental & normalization)
- **Test Status**: 11,407 tests passing (174 skipped)
- **Code Quality**: Zero warnings, full clippy compliance
- **Platform Support**: macOS, Linux, Windows, Linux + CUDA verified

---

## 🚀 v0.1.0-rc.2 (2025-10-19) - Release Candidate 2

### 🎯 Performance & Quality Refinements

This is the second Release Candidate for SciRS2 v0.1.0, focusing on performance optimizations, code quality improvements, and completing deferred high-priority features from RC.1.

**🌟 Highlights**:
- ⚡ **SIMD-Accelerated ODE Solvers** - High-performance RK4 and RK45 implementations
- ✨ **Zero Warnings Build** - Full clippy compliance across all 23 crates
- 📚 **80+ Examples Restored** - Complete example coverage in scirs2-core
- 📝 **107 Files Updated** - Comprehensive version consistency across entire workspace
- ✅ **9,303 Tests Passing** - Production-ready quality assurance

#### ✨ Major Features & Improvements:

**SIMD-Accelerated ODE Solvers (scirs2-integrate)**:
- ✅ Implemented high-performance SIMD methods:
  - `simd_rk4_method`: 4th-order Runge-Kutta with SIMD acceleration
  - `simd_rk45_method`: Adaptive RK45 with full Dormand-Prince SIMD implementation
- ✅ Uses `scirs2-core::simd_ops::SimdUnifiedOps` trait for portable SIMD operations
- ✅ Feature-gated fallbacks for non-SIMD builds
- ✅ Updated examples to demonstrate SIMD performance benefits

**Code Quality Improvements**:
- ✅ Fixed all clippy warnings across the workspace - **zero warnings build**
- ✅ Improved code clarity with `.is_multiple_of()` for even/odd checks
- ✅ Cleaned up unnecessary type casts and unused variables
- ✅ Enhanced pattern matching and iterator usage

**Examples & Documentation**:
- ✅ Restored 80 gutted examples in scirs2-core that were stripped in previous refactoring
- ✅ Fixed examples in scirs2-series:
  - `financial_analysis_demo.rs`: Updated for current function-based API
  - `series_comprehensive_analysis.rs`: Fixed ARIMA forecast signature
- ✅ Enhanced inline documentation with performance notes
- ✅ Added feature requirement clarifications for SIMD functionality

**Comprehensive Documentation Update**:
- ✅ Updated all 24 subcrate README.md files with rc.2 version references
  - Fixed "Beta 4" → "rc.2" in 6 crates
  - Updated version badges across all READMEs
  - Ensured installation examples reflect rc.2
- ✅ Updated all 22 subcrate TODO.md files with rc.2 versions and dates
- ✅ Updated all 24 subcrate lib.rs files with rc.2 installation examples
- ✅ Updated tutorial files in scirs2-datasets and scirs2-metrics
- ✅ Achieved 100% version consistency across 107 files

#### 🔧 Technical Details:

**SIMD Implementation**:
- Vector operations: `F::simd_add`, `F::simd_sub`, `F::simd_scalar_mul`
- Error estimation: `SimdOdeOps::simd_norm_inf`
- Initial step estimation: Fixed 6-parameter signature for all ODE methods
- Corrected `ODEOptions` field names (`atol`/`rtol` instead of `abs_tol`/`rel_tol`)

**Build & Test Status**:
- ✅ All 9,303 tests passing (334 skipped)
- ✅ Zero compilation warnings across entire workspace
- ✅ Full clippy compliance
- ✅ Verified on macOS (M3), Linux (x86_64), and Linux + CUDA

#### 📋 Changes Summary:
- **Added**: SIMD-accelerated ODE solvers
- **Fixed**: All clippy warnings, example API compatibility issues
- **Improved**: Documentation clarity and performance notes
- **Restored**: 80+ examples in scirs2-core
- **Updated**: 107 files across the entire workspace for version consistency
  - 7 core documentation files
  - 24 subcrate lib.rs files
  - 24 subcrate README.md files
  - 22 subcrate TODO.md files
  - 2 tutorial files
  - 1 Cargo.lock with 111 updated packages

#### 📦 Installation:
```toml
[dependencies]
scirs2-core = "0.1.0-rc.2"
scirs2 = "0.1.0-rc.2"
```

#### 🎯 Migration from rc.1:
- **No breaking API changes** - Fully backward compatible
- **Examples updated** - All examples now use current APIs
- **SIMD ODE methods** - Available with `simd` feature flag in scirs2-integrate
- **Documentation** - All 107 files updated for version consistency
- See [CHANGELOG.md](CHANGELOG.md) for complete details

#### 📊 Release Statistics:
- **Files Updated**: 107 files across entire workspace
- **Version References**: 100+ rc.2 references verified
- **Test Status**: 9,303 tests passing (334 skipped)
- **Code Quality**: Zero warnings, full clippy compliance
- **Platform Support**: macOS (M3), Linux (x86_64), Linux + CUDA verified

---

## 🚀 v0.1.0-rc.1 (2025-10-03) - Release Candidate 1

### 🎯 First Release Candidate - Production Readiness Verification

This is the first Release Candidate for SciRS2 v0.1.0, focusing on comprehensive platform testing and final preparation for stable release.

#### 🖥️ Platform Compatibility Testing:
- **macOS Support**: Verified on Apple M3 (ARM64), macOS 15.6.1, 24GB RAM
  - ✅ All 9,800+ tests passing with `cargo nextest run --nff --all-features`
  - ✅ Build succeeds with zero warnings
- **Linux Support**: Verified on x86_64 with required system libraries
  - ✅ All 9,800+ tests passing
  - ✅ Complete functionality with proper environment configuration
- **Linux + CUDA**: Verified on x86_64 with NVIDIA GPU
  - ✅ All 9,800+ tests passing
  - ✅ GPU acceleration features working correctly
- **Windows Support**: Partial support on Windows 11 Pro x86_64
  - ✅ `cargo build` succeeds
  - ⚠️ Some crate tests fail (full compatibility planned for v0.2.0)

#### 📝 Documentation Updates:
- **Platform Compatibility Guide**: Added comprehensive platform testing information to README
- **Version Consistency**: Updated all version references from beta.4 to rc.1
- **Release Documentation**: Updated lib.rs documentation across all 23 crates
- **Roadmap Clarification**: Next release is v0.1.0 (stable), not beta.5

#### 🔧 Release Preparation:
- **Version Bumps**: All crates updated to 0.1.0-rc.1
- **Dependency Updates**: Workspace dependencies aligned to rc.1
- **Date Updates**: Release dates updated to October 03, 2025
- **Build Verification**: All modules compile cleanly with zero warnings

#### 🏗️ Build System Improvements:
- **System Library Priority**: Removed `openblas-src` from workspace dependencies
  - macOS: Uses Accelerate framework (system BLAS, no source build required)
  - Linux: Uses system OpenBLAS (openblas-system feature, no source build required)
  - Windows: Uses system BLAS libraries where available
- **Faster Builds**: Eliminated source compilation of BLAS libraries, significantly reducing build times
- **Platform Optimization**: Leverages platform-specific optimized BLAS implementations
- **Simplified Dependencies**: Removed `intel-mkl-src` to avoid conflicts with system BLAS

#### 🎯 Code Quality & Architecture Overhaul:
- **100% POLICY Compliance**: All 23 crates now fully comply with SciRS2 POLICY
  - Only `scirs2-core` has direct external dependencies
  - All other crates use `scirs2_core::` abstractions
  - ~600+ import statements updated to unified architecture
  - ~100+ external dependencies centralized in scirs2-core
- **Zero Warnings Achievement**: Complete workspace builds with zero warnings
  - Fixed parallel execution errors (Sync trait bounds, Rayon patterns)
  - Added 100+ documentation comments for GPU and memory management
  - Resolved POLICY violations and conditional compilation issues
  - All 23 crates (lib, tests, examples) compile cleanly
- **Code Cleanup**: Significant TODO/FIXME reduction
  - Removed obsolete TODO markers and implemented pending features
  - Refactored complex code sections for better maintainability
  - Enhanced error handling and validation across modules
- **Massive Scale**: 3,016 files modified in comprehensive RC.1 transformation
  - 18,081 insertions across all changes
  - 15,487 deletions for cleanup and refactoring
  - Major commits: POLICY migration (470 files), zero warnings fixes, TODO/FIXME cleanup

#### 📋 Technical Details:
- **Release Date**: October 03, 2025
- **Test Coverage**: 9,800+ tests passing on supported platforms
- **Breaking Changes**: None (API compatible with beta.4)
- **Next Release**: v0.1.0 (stable) - Q4 2026

#### 📦 Installation:
```toml
[dependencies]
scirs2-core = "0.1.0-rc.1"
scirs2 = "0.1.0-rc.1"
```

#### 🎉 Release Candidate Status:
This RC.1 release marks the transition from beta to release candidate status. After community testing and feedback, the next major release will be v0.1.0 stable with long-term API stability guarantees.

---

## 🚀 v0.1.0-beta.4 (2025-10-01) - Release Stabilization

### 🎯 Focus on Stability and Documentation

This release concentrates on stabilizing the ecosystem architecture established in beta.3 and preparing for production deployment.

#### 📝 Documentation and Release Process:
- **Version Management**: Comprehensive version updates across all documentation files
- **Release Workflow**: Streamlined process for version updates and release preparation
- **Documentation Consistency**: Ensured all references point to current beta.4 release
- **API Documentation**: Maintained up-to-date examples and usage guides

#### 🔧 Stability Improvements:
- **Ecosystem Refinement**: Continued implementation of SciRS2 POLICY abstractions
- **Build Verification**: All modules compile cleanly with zero warnings
- **Test Coverage**: Maintained 9,800+ passing tests across all modules
- **Platform Support**: Verified compatibility on Linux, macOS, and Windows

#### 🏗️ Architectural Consistency:
- **Policy Compliance**: Ongoing migration to scirs2-core abstractions
- **API Stability**: No breaking changes from beta.3
- **Performance Validation**: Verified benchmark results across platforms
- **SciPy Compatibility**: Maintained compatibility while improving Rust idioms

#### 📋 Technical Details:
- **Release Date**: October 01, 2025
- **Breaking Changes**: None (API compatible with beta.3)
- **New Features**: Documentation and process improvements
- **Deprecated**: None

#### 📦 Installation:
```toml
[dependencies]
scirs2-core = "0.1.0-beta.4"
scirs2 = "0.1.0-beta.4"
```

---

## 🚀 v0.1.0-beta.4 (2025-09-29) - SciRS2 POLICY & Major Modernization

### 🎯 SciRS2 Ecosystem Architecture Implementation

This release establishes the foundational architecture for the SciRS2 ecosystem with comprehensive policy framework and major dependency modernization.

#### 🏗️ SciRS2 POLICY Framework:
- **Layered Abstraction Architecture**: Established core-only external dependency policy
- **Ecosystem Consistency**: All non-core crates must use scirs2-core abstractions
- **Policy Documentation**: Complete [SciRS2 Ecosystem Policy](SCIRS2_POLICY.md) with guidelines and enforcement strategies
- **Migration Strategy**: Phased approach for systematic refactoring to scirs2-core abstractions

#### 🔧 Major Dependency Updates:
- **Comprehensive Modernization**: Updated all dependencies to latest available versions
- **156 Files Changed**: Extensive updates across the entire workspace
- **Enhanced Performance**: Improved SIMD operations, numerical algorithms, and spatial computations
- **Advanced Random Generation**: Enhanced ecosystem integration with cutting-edge MCMC and neural sampling

#### 🖥️ GPU and Platform Enhancements:
- **CUDA/Linux Optimization**: Significant improvements to CUDA backend for Linux platforms
- **WebGPU Backend**: Major enhancements (333+ lines) for better cross-platform GPU support
- **Memory-Mapped Operations**: Advanced chunking, zero-copy serialization, and large dataset handling
- **Sparse Matrix GPU**: Enhanced GPU operation support for sparse matrix computations

#### 📊 Advanced Feature Stabilization:
- **Real-time Processing**: Improved streaming capabilities in scirs2-io with better error handling
- **Distributed Computing**: Enhanced distributed processing in scirs2-transform with fault tolerance
- **Performance Validation**: Comprehensive SIMD performance validation with automated benchmarking
- **High-Dimensional Interpolation**: Enhanced parallel algorithms for advanced interpolation

#### 📋 Technical Details:
- **Policy Effective Date**: v0.1.0-beta.4
- **Benefits**: Consistent APIs, centralized version control, type safety, maintainability
- **Migration Plan**: Automated tooling for transitioning to scirs2-core abstractions
- **Documentation**: Complete usage guidelines and implementation examples

#### 🎯 Impact:
- **Ecosystem Consistency**: Foundation for unified SciRS2 development patterns
- **Performance**: Better optimization opportunities through centralized abstractions
- **Maintainability**: Simplified dependency management and version control
- **Future-Proofing**: Scalable architecture for continued ecosystem growth

#### 📦 Installation:
```toml
[dependencies]
scirs2-core = "0.1.0-beta.4"
scirs2 = "0.1.0-beta.4"
```

---

## 🔧 v0.1.0-beta.2 (2025-09-16) - Critical Compilation Fixes

### 🚨 Emergency Release - Fixes crates.io Compilation Errors

This is a critical hotfix release that resolves all compilation errors present in v0.1.0-beta.2 when downloaded from crates.io.

#### 🔥 Fixed Compilation Errors (100% resolved):
- **Variable Name Inconsistencies**: Fixed `chunk_size`/`chunksize`, `op_name`/`opname`, `target_unit`/`targetunit` mismatches
- **Undefined Variable References**: Fixed unresolved variables in batch conversion functions (`sequential`, `simd`, `parallel`, `simd_parallel`)
- **Pattern Match Errors**: Corrected pattern matching in SIMD conversion loops (`for (0, &val)` → `for (i, &val)` with proper indexing)
- **Function Name Conflicts**: Resolved duplicate `center()` function definitions by renaming constructor to `centered()`
- **Type Field References**: Fixed `type_info` field reference consistency

#### 📋 Technical Details:
- **Files Modified**: `batch_conversions.rs`, `types.rs`, `dynamic_dispatch.rs`
- **Total Errors Fixed**: 20+ compilation errors reduced to zero
- **Build Verification**: ✅ `cargo build` successful
- **Lint Check**: ✅ `cargo clippy` with zero warnings
- **Publication Ready**: ✅ `cargo publish --dry-run` successful

---

## 🎉 v0.1.0-beta.1 - First Beta Release!

We are excited to announce the first beta release of SciRS2, a comprehensive scientific computing and AI/ML infrastructure in Rust. After months of development, we've reached a significant milestone with over 2 million lines of code and 9,800+ tests.

## ✨ Key Features

### Scientific Computing Core
- **Linear Algebra**: Complete matrix operations, decompositions, eigensolvers
- **Statistics**: Full distribution suite, descriptive statistics, hypothesis tests
- **Optimization**: Unconstrained/constrained optimization, root finding, least squares
- **Integration**: Numerical integration, ODE solvers, boundary value problems
- **FFT**: High-performance Fast Fourier Transform with GPU acceleration
- **Signal Processing**: Filtering, spectral analysis, wavelet transforms
- **Special Functions**: Bessel, gamma, elliptic, and more mathematical functions

### AI/ML Infrastructure  
- **Automatic Differentiation**: Reverse-mode autodiff engine for gradient computation
- **Neural Networks**: Layer abstractions, optimizers, and model building blocks
- **Computer Vision**: Image processing, feature detection, streaming operations
- **Time Series**: Analysis and forecasting tools
- **Metrics**: Comprehensive ML evaluation metrics

### Performance Features
- **SIMD Optimization**: Hardware-accelerated operations using unified SIMD ops
- **GPU Support**: CUDA acceleration with automatic CPU fallback
- **Parallel Processing**: Multi-core computation for intensive operations
- **Memory Efficiency**: Optimized memory management for large datasets

## 📊 Project Statistics

- **Total Lines of Code**: 2,000,000+
- **Number of Modules**: 24 specialized crates
- **Test Coverage**: 9,300+ tests
- **Passing Tests**: All regular tests passing ✅
- **Ignored Tests**: ~600 (benchmarks and hardware-specific)

## 🔧 Recent Improvements

### GPU Test Adaptations
Fixed 12 GPU-dependent tests in the FFT module:
- CUDA initialization tests now gracefully handle missing hardware
- GPU kernel tests provide mock implementations for CPU-only systems
- Multi-GPU tests adapt to available hardware configurations
- Specialized hardware tests (FPGA/ASIC) use appropriate fallbacks

### Test Infrastructure
- Added proper test helpers for graph context management
- Improved test organization and categorization
- Clear separation between functional tests and benchmarks

## ⚠️ Known Limitations

### Autograd Module
- Some gradient shape propagation limitations in complex operations (Issue #1)
- Graph context requirements for certain stability tests
- Workarounds and helper functions are provided

### Unimplemented Features
These features are planned for upcoming releases:
- Cholesky decomposition (0.2.0)
- Thin Plate Spline solver (0.2.0)
- Additional advanced linear algebra decompositions

### Performance Tests
- 404 benchmark tests are ignored by default to optimize CI build times
- Run `cargo test -- --ignored` for full benchmark suite

## 📦 Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2 = "0.1.0-rc.1"  # Use latest version

# Or select specific modules:
scirs2-linalg = "0.1.0-rc.1"
scirs2-stats = "0.1.0-rc.1"
scirs2-autograd = "0.1.0-rc.1"
```

## 🚀 Quick Start

```rust
use scirs2::prelude::*;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create and manipulate matrices
    let a = Array2::eye(3);
    let (u, s, vt) = scirs2::linalg::decomposition::svd(&a)?;
    
    // Statistical computations
    let normal = scirs2::stats::distributions::normal::Normal::new(0.0, 1.0)?;
    let samples = normal.random_sample(1000, None)?;
    
    Ok(())
}
```

## 🔄 Migration from SciPy

SciRS2 provides SciPy-compatible APIs where reasonable, making migration straightforward:

| SciPy | SciRS2 |
|-------|--------|
| `scipy.linalg.svd()` | `scirs2::linalg::decomposition::svd()` |
| `scipy.stats.norm()` | `scirs2::stats::distributions::normal::Normal` |
| `scipy.optimize.minimize()` | `scirs2::optimize::minimize()` |
| `scipy.fft.fft()` | `scirs2::fft::fft()` |

## 🐛 Bug Reports and Feedback

This is a beta release, and we welcome your feedback! Please report issues at:
https://github.com/cool-japan/scirs/issues

## 🔮 What's Next

### Version 0.2.0 (Planned)
- Complete Cholesky decomposition implementation
- Thin Plate Spline solver
- Enhanced GPU backend support
- Performance optimizations based on user feedback

### Long-term Roadmap
- Python bindings for easier migration
- WebAssembly support for browser deployment
- Extended hardware support (ARM, RISC-V)
- Domain-specific extensions (finance, bioinformatics)

## 🙏 Acknowledgments

Thank you to all contributors who made this release possible. Special thanks to:
- The Rust scientific computing community
- Early adopters and testers
- The SciPy and NumPy teams for inspiration

## 📜 License

Dual-licensed under MIT and Apache 2.0.

---

**Note**: This is a beta release. While core functionality is stable and well-tested, some features are still under development. Production use should be carefully evaluated based on your specific requirements.

For detailed documentation, visit: https://docs.rs/scirs2/0.1.0-rc.4/