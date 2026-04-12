# scirs2 Meta-Crate TODO

## Status: v0.3.3 Released (March 17, 2026)

## Purpose

The `scirs2` meta-crate is the all-in-one convenience entry-point for the SciRS2 ecosystem. It re-exports all sub-crates via Cargo feature flags, exposing them as unified top-level modules. Users who want a single dependency add `scirs2 = "0.3.3"` instead of listing each sub-crate individually.

---

## v0.3.3 Completed

- [x] Feature-gated re-exports for all 23 sub-crates
- [x] `standard`, `ai`, `experimental`, `full` feature groups
- [x] `oxifft` feature for high-performance pure-Rust FFT via OxiFFT
- [x] `prelude` module re-exporting common types across the ecosystem
- [x] Workspace-unified version (`version.workspace = true`)
- [x] Pure Rust by default (no C/Fortran transitive dependencies in default features)
- [x] README updated with all feature flags and quick-start examples

---

## v0.4.2 Completed (2026-04-12)

- [x] `cuda` feature group: gates GPU-accelerated paths in sub-crates when CUDA kernels land
- [x] `rocm` feature group: AMD GPU acceleration
- [x] `distributed` feature: enables cluster/MPI abstractions via scirs2-core/array_protocol_distributed
- [x] `benchmarks` feature: expose benchmark helpers from scirs2-datasets
- [x] Expand `prelude` to cover more commonly-used types from new v0.4.2 additions
  - [x] `GpuBuffer`, `GpuDataType` from scirs2-core (always available)
  - [x] `DefragPlanner` from scirs2-core::memory::defrag (memory_management feature)
  - [x] `Precision`, `HMatrixKernel` from scirs2-linalg (linalg feature)
  - [x] `CmaEs`, `CmaEsConfig`, `CmaEsResult` from scirs2-optimize::global (optimize feature)
  - [x] `Ball`, `ball_sin`, `ball_cos` from scirs2-special::validated (special feature)
- [x] Auto-generated feature matrix in documentation (docs.rs all-features already set)
- [x] `jit` feature: gates JAX-style functional transformation framework (implies autograd)
- [x] `vmap`/`pmap` re-exports from scirs2-autograd::transforms under `scirs2::transforms`
- [x] `mobile` feature group: iOS Metal + Android NNAPI gated paths (placeholder, future)
- [x] Structured `scirs2::nn` re-export namespace for neural architecture types
- [x] `symbolic` feature: planned scirs2-symbolic crate integration (placeholder, future)
- [ ] Re-export `scirs2_wasm` module under `wasm` feature for WASM builds

---

## v0.5.0 Planned

- [ ] Wire `cuda`/`rocm`/`mobile` features to actual GPU kernel dispatch once platform crates land
- [ ] Wire `symbolic` to scirs2-symbolic once that crate is created
- [ ] Add `wasm` feature re-exporting scirs2-core WASM backend

---

## Ongoing Maintenance

- [ ] Keep feature list in README.md in sync with Cargo.toml on every release
- [ ] Verify that `cargo doc --all-features` renders correctly on docs.rs after each release
- [ ] Confirm `cargo check --no-default-features` and each individual feature flag compile cleanly
- [ ] Add compile-fail tests for feature-gated paths where API changes land
