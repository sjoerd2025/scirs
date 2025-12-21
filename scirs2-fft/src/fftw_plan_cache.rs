//! FFTW Plan Cache for performance optimization
//!
//! Caches FFTW plans to avoid expensive plan creation on repeated FFT calls.
//! Plans are stored in global caches keyed by size (1D) or dimensions (2D).
//!
//! Note: Plans require &mut self for execution, so we use Mutex to ensure
//! exclusive access. This is fine for Python bindings where the GIL serializes calls.

use fftw::plan::*;
use fftw::types::*;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::error::{FFTError, FFTResult};

// ========================================
// 1D PLAN CACHES
// ========================================

/// Global cache for R2C (real-to-complex) plans, keyed by input size
static R2C_CACHE: Mutex<Option<HashMap<usize, R2CPlan64>>> = Mutex::new(None);

/// Global cache for C2C forward plans, keyed by size
static C2C_FWD_CACHE: Mutex<Option<HashMap<usize, C2CPlan64>>> = Mutex::new(None);

/// Global cache for C2C backward plans, keyed by size
static C2C_BWD_CACHE: Mutex<Option<HashMap<usize, C2CPlan64>>> = Mutex::new(None);

/// Global cache for C2R (complex-to-real) plans, keyed by output size
static C2R_CACHE: Mutex<Option<HashMap<usize, C2RPlan64>>> = Mutex::new(None);

/// Global cache for R2R DCT-II plans
static DCT2_CACHE: Mutex<Option<HashMap<usize, R2RPlan64>>> = Mutex::new(None);

/// Global cache for R2R IDCT-II (DCT-III) plans
static IDCT2_CACHE: Mutex<Option<HashMap<usize, R2RPlan64>>> = Mutex::new(None);

/// Global cache for R2R DST-II plans
static DST2_CACHE: Mutex<Option<HashMap<usize, R2RPlan64>>> = Mutex::new(None);

/// Global cache for R2R IDST-II (DST-III) plans
static IDST2_CACHE: Mutex<Option<HashMap<usize, R2RPlan64>>> = Mutex::new(None);

// ========================================
// 2D PLAN CACHES
// ========================================

/// Global cache for 2D R2C plans, keyed by (rows, cols)
static R2C_2D_CACHE: Mutex<Option<HashMap<(usize, usize), R2CPlan64>>> = Mutex::new(None);

/// Global cache for 2D C2C forward plans
static C2C_2D_FWD_CACHE: Mutex<Option<HashMap<(usize, usize), C2CPlan64>>> = Mutex::new(None);

/// Global cache for 2D C2C backward plans
static C2C_2D_BWD_CACHE: Mutex<Option<HashMap<(usize, usize), C2CPlan64>>> = Mutex::new(None);

/// Global cache for 2D C2R plans
static C2R_2D_CACHE: Mutex<Option<HashMap<(usize, usize), C2RPlan64>>> = Mutex::new(None);

// ========================================
// 1D PLAN EXECUTION FUNCTIONS
// ========================================

/// Execute R2C FFT with cached plan
pub fn execute_r2c(input: &mut [f64], output: &mut [c64]) -> FFTResult<()> {
    let n = input.len();

    let mut cache = R2C_CACHE
        .lock()
        .map_err(|e| FFTError::ComputationError(format!("Failed to lock R2C cache: {}", e)))?;

    // Initialize cache if needed
    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache.as_mut().expect("Operation failed");

    // Create plan if not cached
    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(n) {
        let plan = R2CPlan64::aligned(&[n], Flag::ESTIMATE).map_err(|e| {
            FFTError::ComputationError(format!("Failed to create R2C plan: {:?}", e))
        })?;
        e.insert(plan);
    }

    // Execute with cached plan
    let plan = cache_map.get_mut(&n).expect("Operation failed");
    plan.r2c(input, output)
        .map_err(|e| FFTError::ComputationError(format!("R2C execution failed: {:?}", e)))
}

/// Execute C2C forward FFT with cached plan
pub fn execute_c2c_forward(input: &mut [c64], output: &mut [c64]) -> FFTResult<()> {
    let n = input.len();

    let mut cache = C2C_FWD_CACHE.lock().map_err(|e| {
        FFTError::ComputationError(format!("Failed to lock C2C forward cache: {}", e))
    })?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache.as_mut().expect("Operation failed");

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(n) {
        let plan = C2CPlan64::aligned(&[n], Sign::Forward, Flag::ESTIMATE).map_err(|e| {
            FFTError::ComputationError(format!("Failed to create C2C forward plan: {:?}", e))
        })?;
        e.insert(plan);
    }

    let plan = cache_map.get_mut(&n).expect("Operation failed");
    plan.c2c(input, output)
        .map_err(|e| FFTError::ComputationError(format!("C2C forward execution failed: {:?}", e)))
}

/// Execute C2C backward (inverse) FFT with cached plan
pub fn execute_c2c_backward(input: &mut [c64], output: &mut [c64]) -> FFTResult<()> {
    let n = input.len();

    let mut cache = C2C_BWD_CACHE.lock().map_err(|e| {
        FFTError::ComputationError(format!("Failed to lock C2C backward cache: {}", e))
    })?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache.as_mut().expect("Operation failed");

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(n) {
        let plan = C2CPlan64::aligned(&[n], Sign::Backward, Flag::ESTIMATE).map_err(|e| {
            FFTError::ComputationError(format!("Failed to create C2C backward plan: {:?}", e))
        })?;
        e.insert(plan);
    }

    let plan = cache_map.get_mut(&n).expect("Operation failed");
    plan.c2c(input, output)
        .map_err(|e| FFTError::ComputationError(format!("C2C backward execution failed: {:?}", e)))
}

/// Execute C2R (inverse real) FFT with cached plan
pub fn execute_c2r(input: &mut [c64], output: &mut [f64], n: usize) -> FFTResult<()> {
    let mut cache = C2R_CACHE
        .lock()
        .map_err(|e| FFTError::ComputationError(format!("Failed to lock C2R cache: {}", e)))?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache.as_mut().expect("Operation failed");

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(n) {
        let plan = C2RPlan64::aligned(&[n], Flag::ESTIMATE).map_err(|e| {
            FFTError::ComputationError(format!("Failed to create C2R plan: {:?}", e))
        })?;
        e.insert(plan);
    }

    let plan = cache_map.get_mut(&n).expect("Operation failed");
    plan.c2r(input, output)
        .map_err(|e| FFTError::ComputationError(format!("C2R execution failed: {:?}", e)))
}

/// Execute DCT-II with cached plan
pub fn execute_dct2(input: &mut [f64], output: &mut [f64]) -> FFTResult<()> {
    let n = input.len();

    let mut cache = DCT2_CACHE
        .lock()
        .map_err(|e| FFTError::ComputationError(format!("Failed to lock DCT2 cache: {}", e)))?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache.as_mut().expect("Operation failed");

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(n) {
        let plan =
            R2RPlan64::aligned(&[n], R2RKind::FFTW_REDFT10, Flag::ESTIMATE).map_err(|e| {
                FFTError::ComputationError(format!("Failed to create DCT2 plan: {:?}", e))
            })?;
        e.insert(plan);
    }

    let plan = cache_map.get_mut(&n).expect("Operation failed");
    plan.r2r(input, output)
        .map_err(|e| FFTError::ComputationError(format!("DCT2 execution failed: {:?}", e)))
}

/// Execute IDCT-II (DCT-III) with cached plan
pub fn execute_idct2(input: &mut [f64], output: &mut [f64]) -> FFTResult<()> {
    let n = input.len();

    let mut cache = IDCT2_CACHE
        .lock()
        .map_err(|e| FFTError::ComputationError(format!("Failed to lock IDCT2 cache: {}", e)))?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache.as_mut().expect("Operation failed");

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(n) {
        let plan =
            R2RPlan64::aligned(&[n], R2RKind::FFTW_REDFT01, Flag::ESTIMATE).map_err(|e| {
                FFTError::ComputationError(format!("Failed to create IDCT2 plan: {:?}", e))
            })?;
        e.insert(plan);
    }

    let plan = cache_map.get_mut(&n).expect("Operation failed");
    plan.r2r(input, output)
        .map_err(|e| FFTError::ComputationError(format!("IDCT2 execution failed: {:?}", e)))
}

/// Execute DST-II with cached plan
pub fn execute_dst2(input: &mut [f64], output: &mut [f64]) -> FFTResult<()> {
    let n = input.len();

    let mut cache = DST2_CACHE
        .lock()
        .map_err(|e| FFTError::ComputationError(format!("Failed to lock DST2 cache: {}", e)))?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache.as_mut().expect("Operation failed");

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(n) {
        let plan =
            R2RPlan64::aligned(&[n], R2RKind::FFTW_RODFT10, Flag::ESTIMATE).map_err(|e| {
                FFTError::ComputationError(format!("Failed to create DST2 plan: {:?}", e))
            })?;
        e.insert(plan);
    }

    let plan = cache_map.get_mut(&n).expect("Operation failed");
    plan.r2r(input, output)
        .map_err(|e| FFTError::ComputationError(format!("DST2 execution failed: {:?}", e)))
}

/// Execute IDST-II (DST-III) with cached plan
pub fn execute_idst2(input: &mut [f64], output: &mut [f64]) -> FFTResult<()> {
    let n = input.len();

    let mut cache = IDST2_CACHE
        .lock()
        .map_err(|e| FFTError::ComputationError(format!("Failed to lock IDST2 cache: {}", e)))?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache.as_mut().expect("Operation failed");

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(n) {
        let plan =
            R2RPlan64::aligned(&[n], R2RKind::FFTW_RODFT01, Flag::ESTIMATE).map_err(|e| {
                FFTError::ComputationError(format!("Failed to create IDST2 plan: {:?}", e))
            })?;
        e.insert(plan);
    }

    let plan = cache_map.get_mut(&n).expect("Operation failed");
    plan.r2r(input, output)
        .map_err(|e| FFTError::ComputationError(format!("IDST2 execution failed: {:?}", e)))
}

// ========================================
// 2D PLAN EXECUTION FUNCTIONS
// ========================================

/// Execute 2D R2C FFT with cached plan
pub fn execute_r2c_2d(
    input: &mut [f64],
    output: &mut [c64],
    rows: usize,
    cols: usize,
) -> FFTResult<()> {
    let key = (rows, cols);

    let mut cache = R2C_2D_CACHE
        .lock()
        .map_err(|e| FFTError::ComputationError(format!("Failed to lock 2D R2C cache: {}", e)))?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache.as_mut().expect("Operation failed");

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(key) {
        let plan = R2CPlan64::aligned(&[rows, cols], Flag::ESTIMATE).map_err(|e| {
            FFTError::ComputationError(format!("Failed to create 2D R2C plan: {:?}", e))
        })?;
        e.insert(plan);
    }

    let plan = cache_map.get_mut(&key).expect("Operation failed");
    plan.r2c(input, output)
        .map_err(|e| FFTError::ComputationError(format!("2D R2C execution failed: {:?}", e)))
}

/// Execute 2D C2C forward FFT with cached plan
pub fn execute_c2c_2d_forward(
    input: &mut [c64],
    output: &mut [c64],
    rows: usize,
    cols: usize,
) -> FFTResult<()> {
    let key = (rows, cols);

    let mut cache = C2C_2D_FWD_CACHE.lock().map_err(|e| {
        FFTError::ComputationError(format!("Failed to lock 2D C2C forward cache: {}", e))
    })?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache.as_mut().expect("Operation failed");

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(key) {
        let plan =
            C2CPlan64::aligned(&[rows, cols], Sign::Forward, Flag::ESTIMATE).map_err(|e| {
                FFTError::ComputationError(format!("Failed to create 2D C2C forward plan: {:?}", e))
            })?;
        e.insert(plan);
    }

    let plan = cache_map.get_mut(&key).expect("Operation failed");
    plan.c2c(input, output).map_err(|e| {
        FFTError::ComputationError(format!("2D C2C forward execution failed: {:?}", e))
    })
}

/// Execute 2D C2C backward FFT with cached plan
pub fn execute_c2c_2d_backward(
    input: &mut [c64],
    output: &mut [c64],
    rows: usize,
    cols: usize,
) -> FFTResult<()> {
    let key = (rows, cols);

    let mut cache = C2C_2D_BWD_CACHE.lock().map_err(|e| {
        FFTError::ComputationError(format!("Failed to lock 2D C2C backward cache: {}", e))
    })?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache.as_mut().expect("Operation failed");

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(key) {
        let plan =
            C2CPlan64::aligned(&[rows, cols], Sign::Backward, Flag::ESTIMATE).map_err(|e| {
                FFTError::ComputationError(format!(
                    "Failed to create 2D C2C backward plan: {:?}",
                    e
                ))
            })?;
        e.insert(plan);
    }

    let plan = cache_map.get_mut(&key).expect("Operation failed");
    plan.c2c(input, output).map_err(|e| {
        FFTError::ComputationError(format!("2D C2C backward execution failed: {:?}", e))
    })
}

/// Execute 2D C2R FFT with cached plan
pub fn execute_c2r_2d(
    input: &mut [c64],
    output: &mut [f64],
    rows: usize,
    cols: usize,
) -> FFTResult<()> {
    let key = (rows, cols);

    let mut cache = C2R_2D_CACHE
        .lock()
        .map_err(|e| FFTError::ComputationError(format!("Failed to lock 2D C2R cache: {}", e)))?;

    if cache.is_none() {
        *cache = Some(HashMap::new());
    }

    let cache_map = cache.as_mut().expect("Operation failed");

    if let std::collections::hash_map::Entry::Vacant(e) = cache_map.entry(key) {
        let plan = C2RPlan64::aligned(&[rows, cols], Flag::ESTIMATE).map_err(|e| {
            FFTError::ComputationError(format!("Failed to create 2D C2R plan: {:?}", e))
        })?;
        e.insert(plan);
    }

    let plan = cache_map.get_mut(&key).expect("Operation failed");
    plan.c2r(input, output)
        .map_err(|e| FFTError::ComputationError(format!("2D C2R execution failed: {:?}", e)))
}

// ========================================
// CACHE MANAGEMENT
// ========================================

/// Clear all cached plans (useful for testing or memory management)
pub fn clear_all_caches() {
    if let Ok(mut cache) = R2C_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = C2C_FWD_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = C2C_BWD_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = C2R_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = DCT2_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = IDCT2_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = DST2_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = IDST2_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = R2C_2D_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = C2C_2D_FWD_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = C2C_2D_BWD_CACHE.lock() {
        *cache = None;
    }
    if let Ok(mut cache) = C2R_2D_CACHE.lock() {
        *cache = None;
    }
}
