//! # ⚠️ MOVED: scirs2-optim → OptiRS
//!
//! This crate has been **moved to an independent project**.
//!
//! ## Migration
//!
//! **Old:**
//! ```toml
//! [dependencies]
//! scirs2-optim = "<removed>"  # ❌ No longer available
//! ```
//!
//! **New:**
//! ```toml
//! [dependencies]
//! optirs = "0.1.0"  # ✅ Use this instead
//! ```
//!
//! ## New Location
//!
//! - **Repository**: <https://github.com/cool-japan/optirs>
//! - **Documentation**: Available in the OptiRS repository
//!
//! ## API Compatibility
//!
//! The OptiRS API is designed to be largely compatible with the previous scirs2-optim API:
//!
//! ```rust,ignore
//! // Before (scirs2-optim)
//! use scirs2_optim::optimizers::Adam;
//!
//! // After (OptiRS)
//! use optirs::optimizers::Adam;
//! // or
//! use optirs::prelude::*;
//! ```

#![deprecated(note = "This crate has been moved to the independent OptiRS project. Use 'optirs' crate instead. See: https://github.com/cool-japan/optirs")]

#[deprecated(
    note = "This crate has been moved to OptiRS. Use 'optirs' crate instead."
)]
pub fn migration_notice() {
    eprintln!("⚠️  scirs2-optim has moved to OptiRS!");
    eprintln!("📦 Add to Cargo.toml: optirs = \"0.1.0\"");
    eprintln!("🔗 Repository: https://github.com/cool-japan/optirs");
}