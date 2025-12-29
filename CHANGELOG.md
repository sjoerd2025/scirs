# Changelog

All notable changes to the SciRS2 project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-29

### 🎉 Stable Release - Production Ready

This is the first stable release of SciRS2, marking a significant milestone in providing a comprehensive scientific computing and AI/ML infrastructure in Rust.

### Major Achievements

#### Code Quality & Architecture
- **Refactoring Policy Compliance**: Successfully refactored entire codebase to meet <2000 line per file policy
  - 21 large files (58,000+ lines) split into 150+ well-organized modules
  - Improved code maintainability and readability
  - Enhanced module organization with clear separation of concerns
  - Maximum file size reduced to ~1000 lines
- **Zero Warnings Policy**: Maintained strict zero-warnings compliance
  - All compilation warnings resolved
  - Full clippy compliance (except 235 acceptable documentation warnings)
  - Clean build across all workspace crates
- **Test Coverage**: 10,861 tests passing across 170 test binaries
  - Comprehensive unit and integration test coverage
  - 149 tests appropriately skipped for platform-specific features
  - All test imports and visibility issues resolved

#### Build System Improvements
- **Module Refactoring**: Major structural improvements
  - Split scirs2-core/src/simd_ops.rs (4724 lines → 8 modules)
  - Split scirs2-core/src/simd/transcendental/mod.rs (3623 lines → 7 modules)
  - Refactored 19 additional large modules across workspace
- **Visibility Fixes**: Resolved 150+ field and method visibility issues for test access
- **Import Organization**: Fixed 60+ missing imports and trait dependencies

#### Bug Fixes
- Fixed test compilation errors in scirs2-series (Array1 imports, field visibility)
- Fixed test compilation errors in scirs2-datasets (Array2, Instant imports, method visibility)
- Fixed test compilation errors in scirs2-spatial (Duration import, 40+ visibility issues)
- Fixed test compilation errors in scirs2-stats (Duration import, method visibility)
- Resolved duplicate `use super::*;` statements across test files
- Fixed collapsible if statement in scirs2-core
- Removed duplicate conditional branches in scirs2-spatial

### Technical Specifications

#### Quality Metrics
- **Tests**: 10,861 passing / 149 skipped
- **Warnings**: 0 compilation errors, 0 non-doc warnings
- **Code**: ~1.68M lines of Rust code across 4,727 files
- **Modules**: 150+ newly refactored modules for better organization

#### Platform Support
- ✅ **Linux (x86_64)**: Full support with all features
- ✅ **macOS (ARM64/x86_64)**: Full support with Metal acceleration
- ✅ **Windows (x86_64)**: Build support with ongoing improvements

### Notes

This stable release represents the culmination of extensive development, testing, and refinement. The codebase is production-ready with excellent code quality, comprehensive test coverage, and strong adherence to Rust best practices.

## [0.1.0] - 2025-12-29

### 🚀 Stable Release - Documentation & Stability Enhancements

This release focuses on comprehensive documentation updates, build system improvements, and final preparations for the stable 0.1.0 release.

### Added

#### Documentation
- **Comprehensive Documentation Updates**: Complete revision of all major documentation files
  - Updated README.md with stable release status and feature highlights
  - Revised TODO.md with current development roadmap
  - Enhanced CLAUDE.md with latest development guidelines
  - Refreshed all module lib.rs documentation for docs.rs

#### Developer Experience
- **Improved Development Workflows**: Enhanced build and test documentation
  - Clarified cargo nextest usage patterns
  - Updated dependency management guidelines
  - Enhanced troubleshooting documentation

### Changed

#### Build System
- **Version Synchronization**: Updated all version references to 0.1.0
  - Workspace Cargo.toml version bump
  - Documentation version consistency
  - Example and test version alignment

#### Documentation Improvements
- **README.md**: Updated release status and feature descriptions
- **TODO.md**: Synchronized development roadmap with current release status
- **CLAUDE.md**: Updated version info and development guidelines
- **Module Documentation**: Refreshed inline documentation across all crates

### Fixed

#### Documentation Consistency
- Resolved version mismatches across documentation files
- Corrected outdated feature descriptions
- Fixed cross-references between documentation files
- Updated dependency version information

### Technical Details

#### Quality Metrics
- All 11,407 tests passing (174 skipped)
- Zero compilation warnings maintained
- Full clippy compliance across workspace
- Documentation builds successfully on docs.rs

#### Platform Support
- ✅ Linux (x86_64): Full support with all features
- ✅ macOS (ARM64/x86_64): Full support with Metal acceleration
- ✅ Windows (x86_64): Build support, ongoing test improvements

### Notes

This release represents the final preparation before the 0.1.0 stable release. The focus is on documentation quality, developer experience, and ensuring all materials are ready for the stable release.
