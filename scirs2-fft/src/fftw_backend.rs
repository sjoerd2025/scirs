//! FFTW-based FFT backend for high performance
//!
//! Provides 62x speedup over pure Rust FFT implementation by using
//! the industry-standard FFTW library.
//!
//! Now with plan caching for even better performance on repeated transforms.

use crate::error::{FFTError, FFTResult};
use crate::fftw_plan_cache;
use fftw::types::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::Complex;

/// Real-to-complex FFT using FFTW with plan caching
///
/// Computes the FFT of a real-valued signal, returning complex frequency components.
/// Uses cached plans for better performance on repeated transforms of the same size.
///
/// # Arguments
///
/// * `input` - Real-valued input signal
///
/// # Returns
///
/// * Complex frequency spectrum (N/2+1 components for real input of length N)
pub fn rfft_fftw(input: &ArrayView1<f64>) -> FFTResult<Array1<Complex<f64>>> {
    let n = input.len();

    // Prepare input and output buffers
    let mut input_vec = input.to_vec();
    let output_len = n / 2 + 1;
    let mut output = vec![c64::default(); output_len];

    // Execute with cached plan
    fftw_plan_cache::execute_r2c(&mut input_vec, &mut output)?;

    // Convert fftw::c64 to scirs2 Complex<f64>
    let result: Vec<Complex<f64>> = output.iter().map(|c| Complex::new(c.re, c.im)).collect();

    Ok(Array1::from_vec(result))
}

/// Complex-to-complex FFT using FFTW with plan caching
///
/// Computes the FFT of a complex-valued signal.
pub fn fft_fftw(input: &ArrayView1<Complex<f64>>) -> FFTResult<Array1<Complex<f64>>> {
    let n = input.len();

    // Convert scirs2 Complex to fftw c64
    let mut input_fftw: Vec<c64> = input.iter().map(|c| c64 { re: c.re, im: c.im }).collect();
    let mut output = vec![c64::default(); n];

    // Execute with cached plan
    fftw_plan_cache::execute_c2c_forward(&mut input_fftw, &mut output)?;

    // Convert back
    let result: Vec<Complex<f64>> = output.iter().map(|c| Complex::new(c.re, c.im)).collect();

    Ok(Array1::from_vec(result))
}

/// Inverse complex-to-complex FFT using FFTW with plan caching
pub fn ifft_fftw(input: &ArrayView1<Complex<f64>>) -> FFTResult<Array1<Complex<f64>>> {
    let n = input.len();

    // Convert to fftw format
    let mut input_fftw: Vec<c64> = input.iter().map(|c| c64 { re: c.re, im: c.im }).collect();
    let mut output = vec![c64::default(); n];

    // Execute with cached plan
    fftw_plan_cache::execute_c2c_backward(&mut input_fftw, &mut output)?;

    // FFTW doesn't normalize, so divide by N
    let scale = 1.0 / (n as f64);
    let result: Vec<Complex<f64>> = output
        .iter()
        .map(|c| Complex::new(c.re * scale, c.im * scale))
        .collect();

    Ok(Array1::from_vec(result))
}

/// Inverse real FFT using FFTW with plan caching
///
/// Converts complex frequency spectrum back to real-valued time-domain signal.
pub fn irfft_fftw(input: &ArrayView1<Complex<f64>>, n: usize) -> FFTResult<Array1<f64>> {
    // Convert to fftw format
    let mut input_fftw: Vec<c64> = input.iter().map(|c| c64 { re: c.re, im: c.im }).collect();
    let mut output = vec![0.0; n];

    // Execute with cached plan
    fftw_plan_cache::execute_c2r(&mut input_fftw, &mut output, n)?;

    // FFTW doesn't normalize
    let scale = 1.0 / (n as f64);
    let result: Vec<f64> = output.iter().map(|&x| x * scale).collect();

    Ok(Array1::from_vec(result))
}

// ========================================
// 2D FFT FUNCTIONS
// ========================================

/// 2D complex-to-complex FFT using FFTW with plan caching
pub fn fft2_fftw(input: &ArrayView2<Complex<f64>>) -> FFTResult<Array2<Complex<f64>>> {
    let (rows, cols) = input.dim();
    let n = rows * cols;

    // Convert to row-major contiguous Vec
    let mut input_fftw: Vec<c64> = input.iter().map(|c| c64 { re: c.re, im: c.im }).collect();
    let mut output = vec![c64::default(); n];

    // Execute with cached plan
    fftw_plan_cache::execute_c2c_2d_forward(&mut input_fftw, &mut output, rows, cols)?;

    // Convert back and reshape to 2D
    let result: Vec<Complex<f64>> = output.iter().map(|c| Complex::new(c.re, c.im)).collect();

    Array2::from_shape_vec((rows, cols), result)
        .map_err(|e| FFTError::ComputationError(format!("Failed to reshape result: {:?}", e)))
}

/// 2D inverse complex-to-complex FFT using FFTW with plan caching
pub fn ifft2_fftw(input: &ArrayView2<Complex<f64>>) -> FFTResult<Array2<Complex<f64>>> {
    let (rows, cols) = input.dim();
    let n = rows * cols;

    // Convert to row-major contiguous Vec
    let mut input_fftw: Vec<c64> = input.iter().map(|c| c64 { re: c.re, im: c.im }).collect();
    let mut output = vec![c64::default(); n];

    // Execute with cached plan
    fftw_plan_cache::execute_c2c_2d_backward(&mut input_fftw, &mut output, rows, cols)?;

    // FFTW doesn't normalize
    let scale = 1.0 / (n as f64);
    let result: Vec<Complex<f64>> = output
        .iter()
        .map(|c| Complex::new(c.re * scale, c.im * scale))
        .collect();

    Array2::from_shape_vec((rows, cols), result)
        .map_err(|e| FFTError::ComputationError(format!("Failed to reshape result: {:?}", e)))
}

/// 2D real-to-complex FFT using FFTW with plan caching
pub fn rfft2_fftw(input: &ArrayView2<f64>) -> FFTResult<Array2<Complex<f64>>> {
    let (rows, cols) = input.dim();

    // Convert to row-major contiguous Vec
    let mut input_vec: Vec<f64> = input.iter().cloned().collect();

    // For 2D real FFT, output has shape (rows, cols/2 + 1)
    let out_cols = cols / 2 + 1;
    let mut output = vec![c64::default(); rows * out_cols];

    // Execute with cached plan
    fftw_plan_cache::execute_r2c_2d(&mut input_vec, &mut output, rows, cols)?;

    // Convert back and reshape
    let result: Vec<Complex<f64>> = output.iter().map(|c| Complex::new(c.re, c.im)).collect();

    Array2::from_shape_vec((rows, out_cols), result)
        .map_err(|e| FFTError::ComputationError(format!("Failed to reshape result: {:?}", e)))
}

/// 2D inverse real FFT using FFTW with plan caching
pub fn irfft2_fftw(
    input: &ArrayView2<Complex<f64>>,
    shape: (usize, usize),
) -> FFTResult<Array2<f64>> {
    let (rows, cols) = shape;
    let (in_rows, in_cols) = input.dim();

    // Validate input dimensions
    if in_rows != rows || in_cols != cols / 2 + 1 {
        return Err(FFTError::ValueError(format!(
            "Input shape ({}, {}) doesn't match expected ({}, {}) for output shape ({}, {})",
            in_rows,
            in_cols,
            rows,
            cols / 2 + 1,
            rows,
            cols
        )));
    }

    // Convert to row-major contiguous Vec
    let mut input_fftw: Vec<c64> = input.iter().map(|c| c64 { re: c.re, im: c.im }).collect();
    let mut output = vec![0.0; rows * cols];

    // Execute with cached plan
    fftw_plan_cache::execute_c2r_2d(&mut input_fftw, &mut output, rows, cols)?;

    // FFTW doesn't normalize
    let scale = 1.0 / ((rows * cols) as f64);
    let result: Vec<f64> = output.iter().map(|&x| x * scale).collect();

    Array2::from_shape_vec((rows, cols), result)
        .map_err(|e| FFTError::ComputationError(format!("Failed to reshape result: {:?}", e)))
}

// ========================================
// DCT/DST FUNCTIONS using R2R transforms with plan caching
// ========================================

/// DCT Type II using FFTW with plan caching (REDFT10)
/// This is the most commonly used DCT type
pub fn dct2_fftw(input: &ArrayView1<f64>) -> FFTResult<Array1<f64>> {
    let n = input.len();
    let mut input_vec = input.to_vec();
    let mut output = vec![0.0; n];

    // Execute with cached plan
    fftw_plan_cache::execute_dct2(&mut input_vec, &mut output)?;

    Ok(Array1::from_vec(output))
}

/// IDCT Type II using FFTW with plan caching (REDFT01 = DCT-III)
/// DCT-III is the inverse of DCT-II
pub fn idct2_fftw(input: &ArrayView1<f64>) -> FFTResult<Array1<f64>> {
    let n = input.len();
    let mut input_vec = input.to_vec();
    let mut output = vec![0.0; n];

    // Execute with cached plan
    fftw_plan_cache::execute_idct2(&mut input_vec, &mut output)?;

    // FFTW doesn't normalize - need to scale by 1/(2*N)
    let scale = 1.0 / (2.0 * n as f64);
    let result: Vec<f64> = output.iter().map(|&x| x * scale).collect();

    Ok(Array1::from_vec(result))
}

/// DST Type II using FFTW with plan caching (RODFT10)
pub fn dst2_fftw(input: &ArrayView1<f64>) -> FFTResult<Array1<f64>> {
    let n = input.len();
    let mut input_vec = input.to_vec();
    let mut output = vec![0.0; n];

    // Execute with cached plan
    fftw_plan_cache::execute_dst2(&mut input_vec, &mut output)?;

    Ok(Array1::from_vec(output))
}

/// IDST Type II using FFTW with plan caching (RODFT01 = DST-III)
pub fn idst2_fftw(input: &ArrayView1<f64>) -> FFTResult<Array1<f64>> {
    let n = input.len();
    let mut input_vec = input.to_vec();
    let mut output = vec![0.0; n];

    // Execute with cached plan
    fftw_plan_cache::execute_idst2(&mut input_vec, &mut output)?;

    // FFTW doesn't normalize
    let scale = 1.0 / (2.0 * n as f64);
    let result: Vec<f64> = output.iter().map(|&x| x * scale).collect();

    Ok(Array1::from_vec(result))
}
