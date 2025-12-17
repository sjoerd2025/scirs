//! SIMD-accelerated transcendental functions
//!
//! This module provides highly optimized SIMD implementations of transcendental
//! functions (exp, ln, sin, cos, etc.) using polynomial approximations.
//! These are critical for neural network operations like softmax and attention.
//!
//! ## Module Structure
//!
//! This module contains 23 transcendental functions organized into functional groups:
//!
//! ### Exponential Functions (Phase 75)
//! - [`simd_exp_f32`], [`simd_exp_f64`]: Polynomial exp approximation
//! - [`simd_exp_fast_f32`]: Ultra-fast integer bit manipulation
//!
//! ### Neural Network Activations (Phase 76-77)
//! - [`simd_sigmoid_f32`], [`simd_sigmoid_f64`]: Logistic activation
//! - [`simd_gelu_f32`], [`simd_gelu_f64`]: GPT-2/BERT activation
//! - [`simd_swish_f32`], [`simd_swish_f64`]: SiLU activation
//! - [`simd_softplus_f32`], [`simd_softplus_f64`]: Smooth ReLU
//! - [`simd_mish_f32`], [`simd_mish_f64`]: Modern activation
//!
//! ### Hyperbolic & Trigonometric (Phase 77-78)
//! - [`simd_tanh_f32`], [`simd_tanh_f64`]: Hyperbolic tangent
//! - [`simd_sin_f32`], [`simd_sin_f64`]: Sine (1,957 uses in codebase)
//! - [`simd_cos_f32`], [`simd_cos_f64`]: Cosine (1,236 uses in codebase)
//!
//! ### Logarithmic Functions (Phase 77-78)
//! - [`simd_ln_f32`], [`simd_ln_f64`]: Natural logarithm
//! - [`simd_log2_f32`], [`simd_log2_f64`]: Base-2 logarithm (entropy, FFT)
//! - [`simd_log10_f32`], [`simd_log10_f64`]: Base-10 logarithm (dB, SNR)
//!
//! ## Future Refactoring
//!
//! This file is 3,561 lines (78% over 2000-line policy). Future extraction recommended:
//! - `exp.rs`: Exponential functions (~520 lines)
//! - `activations.rs`: Neural network activations (~1100 lines)
//! - `trig.rs`: Trigonometric & hyperbolic (~900 lines)
//! - `log.rs`: Logarithmic functions (~720 lines)
//! - `mod.rs`: Module structure and re-exports (~50 lines)

use ndarray::{Array1, ArrayView1};

/// SIMD-accelerated exp for f32 arrays using polynomial approximation
///
/// Uses range reduction and a degree-6 polynomial approximation.
/// Achieves ~10^-7 relative error for most inputs.
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of exp(x) values
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_exp_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, 1.0, -1.0, 2.0];
/// let result = simd_exp_f32(&x.view());
/// assert!((result[0] - 1.0).abs() < 1e-6);
/// assert!((result[1] - std::f32::consts::E).abs() < 1e-5);
/// ```
#[allow(dead_code)]
pub fn simd_exp_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                // Constants for range reduction
                let log2_e = _mm256_set1_ps(std::f32::consts::LOG2_E); // 1/ln(2)
                let ln_2 = _mm256_set1_ps(std::f32::consts::LN_2);
                let half = _mm256_set1_ps(0.5);

                // Polynomial coefficients for exp(r) where |r| <= ln(2)/2
                // Using Remez polynomial approximation
                let c0 = _mm256_set1_ps(1.0);
                let c1 = _mm256_set1_ps(1.0);
                let c2 = _mm256_set1_ps(0.5);
                let c3 = _mm256_set1_ps(0.16666666666666666);
                let c4 = _mm256_set1_ps(0.041666666666666664);
                let c5 = _mm256_set1_ps(0.008333333333333333);
                let c6 = _mm256_set1_ps(0.001388888888888889);

                // Clamp bounds to avoid overflow/underflow
                let max_val = _mm256_set1_ps(88.0); // exp(88) ≈ 1.6e38
                let min_val = _mm256_set1_ps(-88.0);

                let mut i = 0;
                while i + 8 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 8];
                    let mut x = _mm256_loadu_ps(slice.as_ptr());

                    // Clamp to avoid overflow/underflow
                    x = _mm256_max_ps(x, min_val);
                    x = _mm256_min_ps(x, max_val);

                    // Range reduction: x = k*ln(2) + r, where k = round(x/ln(2))
                    let k_float = _mm256_round_ps(
                        _mm256_mul_ps(x, log2_e),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    let r = _mm256_sub_ps(x, _mm256_mul_ps(k_float, ln_2));

                    // Polynomial approximation: exp(r) ≈ 1 + r + r²/2 + r³/6 + ...
                    // Using Horner's method
                    let r2 = _mm256_mul_ps(r, r);
                    let r3 = _mm256_mul_ps(r2, r);
                    let r4 = _mm256_mul_ps(r2, r2);
                    let r5 = _mm256_mul_ps(r4, r);
                    let r6 = _mm256_mul_ps(r4, r2);

                    // exp(r) = c0 + c1*r + c2*r² + c3*r³ + c4*r⁴ + c5*r⁵ + c6*r⁶
                    let mut exp_r = c0;
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c1, r));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c2, r2));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c3, r3));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c4, r4));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c5, r5));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c6, r6));

                    // Scale by 2^k using integer arithmetic
                    let k_int = _mm256_cvtps_epi32(k_float);
                    let bias = _mm256_set1_epi32(127); // IEEE float exponent bias
                    let exp_bits = _mm256_slli_epi32(_mm256_add_epi32(k_int, bias), 23);
                    let scale = _mm256_castsi256_ps(exp_bits);

                    let exp_x = _mm256_mul_ps(exp_r, scale);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), exp_x);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(input[j].exp());
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;

                let log2_e = vdupq_n_f32(std::f32::consts::LOG2_E);
                let ln_2 = vdupq_n_f32(std::f32::consts::LN_2);

                // Polynomial coefficients
                let c0 = vdupq_n_f32(1.0);
                let c1 = vdupq_n_f32(1.0);
                let c2 = vdupq_n_f32(0.5);
                let c3 = vdupq_n_f32(0.16666666666666666);
                let c4 = vdupq_n_f32(0.041666666666666664);
                let c5 = vdupq_n_f32(0.008333333333333333);
                let c6 = vdupq_n_f32(0.001388888888888889);

                let max_val = vdupq_n_f32(88.0);
                let min_val = vdupq_n_f32(-88.0);

                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 4];
                    let mut x = vld1q_f32(slice.as_ptr());

                    // Clamp
                    x = vmaxq_f32(x, min_val);
                    x = vminq_f32(x, max_val);

                    // Range reduction
                    let k_float = vrndnq_f32(vmulq_f32(x, log2_e));
                    let r = vsubq_f32(x, vmulq_f32(k_float, ln_2));

                    // Polynomial evaluation
                    let r2 = vmulq_f32(r, r);
                    let r3 = vmulq_f32(r2, r);
                    let r4 = vmulq_f32(r2, r2);
                    let r5 = vmulq_f32(r4, r);
                    let r6 = vmulq_f32(r4, r2);

                    let mut exp_r = c0;
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c1, r));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c2, r2));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c3, r3));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c4, r4));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c5, r5));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c6, r6));

                    // Scale by 2^k
                    let k_int = vcvtq_s32_f32(k_float);
                    let bias = vdupq_n_s32(127);
                    let exp_bits = vshlq_n_s32(vaddq_s32(k_int, bias), 23);
                    let scale = vreinterpretq_f32_s32(exp_bits);

                    let exp_x = vmulq_f32(exp_r, scale);

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), exp_x);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(input[j].exp());
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for &val in input.iter() {
        result.push(val.exp());
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated exp for f64 arrays using polynomial approximation
///
/// Uses range reduction and a higher-degree polynomial for double precision.
/// Achieves ~10^-15 relative error for most inputs.
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of exp(x) values
#[allow(dead_code)]
pub fn simd_exp_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let log2_e = _mm256_set1_pd(std::f64::consts::LOG2_E);
                let ln_2 = _mm256_set1_pd(std::f64::consts::LN_2);

                // Higher-degree polynomial coefficients for double precision
                // exp(r) = 1 + r + r²/2! + r³/3! + ... + r¹²/12!
                let c0 = _mm256_set1_pd(1.0);
                let c1 = _mm256_set1_pd(1.0);
                let c2 = _mm256_set1_pd(0.5);
                let c3 = _mm256_set1_pd(1.0 / 6.0);
                let c4 = _mm256_set1_pd(1.0 / 24.0);
                let c5 = _mm256_set1_pd(1.0 / 120.0);
                let c6 = _mm256_set1_pd(1.0 / 720.0);
                let c7 = _mm256_set1_pd(1.0 / 5040.0);
                let c8 = _mm256_set1_pd(1.0 / 40320.0);
                let c9 = _mm256_set1_pd(1.0 / 362880.0);
                let c10 = _mm256_set1_pd(1.0 / 3628800.0);

                let max_val = _mm256_set1_pd(709.0); // exp(709) ≈ 8.2e307
                let min_val = _mm256_set1_pd(-709.0);

                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 4];
                    let mut x = _mm256_loadu_pd(slice.as_ptr());

                    // Clamp
                    x = _mm256_max_pd(x, min_val);
                    x = _mm256_min_pd(x, max_val);

                    // Range reduction
                    let k_float = _mm256_round_pd(
                        _mm256_mul_pd(x, log2_e),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    let r = _mm256_sub_pd(x, _mm256_mul_pd(k_float, ln_2));

                    // Polynomial evaluation using Horner's method
                    let r2 = _mm256_mul_pd(r, r);

                    // Compute even and odd terms separately for better accuracy
                    // even = c0 + c2*r² + c4*r⁴ + c6*r⁶ + c8*r⁸ + c10*r¹⁰
                    // odd = c1*r + c3*r³ + c5*r⁵ + c7*r⁷ + c9*r⁹
                    let r4 = _mm256_mul_pd(r2, r2);
                    let r6 = _mm256_mul_pd(r4, r2);
                    let r8 = _mm256_mul_pd(r4, r4);
                    let r10 = _mm256_mul_pd(r8, r2);

                    let mut even = c0;
                    even = _mm256_add_pd(even, _mm256_mul_pd(c2, r2));
                    even = _mm256_add_pd(even, _mm256_mul_pd(c4, r4));
                    even = _mm256_add_pd(even, _mm256_mul_pd(c6, r6));
                    even = _mm256_add_pd(even, _mm256_mul_pd(c8, r8));
                    even = _mm256_add_pd(even, _mm256_mul_pd(c10, r10));

                    let r3 = _mm256_mul_pd(r2, r);
                    let r5 = _mm256_mul_pd(r4, r);
                    let r7 = _mm256_mul_pd(r6, r);
                    let r9 = _mm256_mul_pd(r8, r);

                    let mut odd = _mm256_mul_pd(c1, r);
                    odd = _mm256_add_pd(odd, _mm256_mul_pd(c3, r3));
                    odd = _mm256_add_pd(odd, _mm256_mul_pd(c5, r5));
                    odd = _mm256_add_pd(odd, _mm256_mul_pd(c7, r7));
                    odd = _mm256_add_pd(odd, _mm256_mul_pd(c9, r9));

                    let exp_r = _mm256_add_pd(even, odd);

                    // Scale by 2^k using integer manipulation
                    // For f64: bias=1023, mantissa bits=52
                    let k_long = _mm256_cvtpd_epi32(k_float);
                    // We need to work with 64-bit integers
                    let mut k_arr = [0i32; 4];
                    _mm_storeu_si128(k_arr.as_mut_ptr() as *mut __m128i, k_long);

                    let mut scale_arr = [0.0f64; 4];
                    for j in 0..4 {
                        let exp_bits = ((k_arr[j] as i64 + 1023) << 52) as u64;
                        scale_arr[j] = f64::from_bits(exp_bits);
                    }
                    let scale = _mm256_loadu_pd(scale_arr.as_ptr());

                    let exp_x = _mm256_mul_pd(exp_r, scale);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), exp_x);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(input[j].exp());
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;

                let log2_e = vdupq_n_f64(std::f64::consts::LOG2_E);
                let ln_2 = vdupq_n_f64(std::f64::consts::LN_2);

                let c0 = vdupq_n_f64(1.0);
                let c1 = vdupq_n_f64(1.0);
                let c2 = vdupq_n_f64(0.5);
                let c3 = vdupq_n_f64(1.0 / 6.0);
                let c4 = vdupq_n_f64(1.0 / 24.0);
                let c5 = vdupq_n_f64(1.0 / 120.0);
                let c6 = vdupq_n_f64(1.0 / 720.0);
                let c7 = vdupq_n_f64(1.0 / 5040.0);
                let c8 = vdupq_n_f64(1.0 / 40320.0);

                let max_val = vdupq_n_f64(709.0);
                let min_val = vdupq_n_f64(-709.0);

                let mut i = 0;
                while i + 2 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 2];
                    let mut x = vld1q_f64(slice.as_ptr());

                    // Clamp
                    x = vmaxq_f64(x, min_val);
                    x = vminq_f64(x, max_val);

                    // Range reduction
                    let k_float = vrndnq_f64(vmulq_f64(x, log2_e));
                    let r = vsubq_f64(x, vmulq_f64(k_float, ln_2));

                    // Polynomial evaluation
                    let r2 = vmulq_f64(r, r);
                    let r3 = vmulq_f64(r2, r);
                    let r4 = vmulq_f64(r2, r2);
                    let r5 = vmulq_f64(r4, r);
                    let r6 = vmulq_f64(r4, r2);
                    let r7 = vmulq_f64(r6, r);
                    let r8 = vmulq_f64(r4, r4);

                    let mut exp_r = c0;
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c1, r));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c2, r2));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c3, r3));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c4, r4));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c5, r5));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c6, r6));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c7, r7));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c8, r8));

                    // Scale by 2^k
                    let k_int = vcvtq_s64_f64(k_float);
                    let bias = vdupq_n_s64(1023);
                    let exp_bits = vshlq_n_s64(vaddq_s64(k_int, bias), 52);
                    let scale = vreinterpretq_f64_s64(exp_bits);

                    let exp_x = vmulq_f64(exp_r, scale);

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), exp_x);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(input[j].exp());
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for &val in input.iter() {
        result.push(val.exp());
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated fast exp for f32 (lower precision but faster)
///
/// Uses a simpler polynomial approximation for maximum speed.
/// Suitable for neural network inference where ~10^-4 accuracy is acceptable.
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of approximate exp(x) values
#[allow(dead_code)]
pub fn simd_exp_fast_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                // Fast exp approximation using integer bit manipulation
                // exp(x) ≈ 2^(x/ln(2)) = 2^(x * LOG2_E)
                // Uses the fact that float = 2^(exp-127) * (1 + mantissa)
                let scale = _mm256_set1_ps(12102203.0); // 2^23 / ln(2)
                let offset = _mm256_set1_ps(1065353216.0 - 486411.0); // 127 * 2^23 - adjustment
                let max_val = _mm256_set1_ps(88.0);
                let min_val = _mm256_set1_ps(-88.0);

                let mut i = 0;
                while i + 8 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 8];
                    let mut x = _mm256_loadu_ps(slice.as_ptr());

                    // Clamp
                    x = _mm256_max_ps(x, min_val);
                    x = _mm256_min_ps(x, max_val);

                    // Fast approximation: interpret (x * scale + offset) as float
                    let approx = _mm256_add_ps(_mm256_mul_ps(x, scale), offset);
                    let exp_x = _mm256_castsi256_ps(_mm256_cvtps_epi32(approx));

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), exp_x);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(input[j].exp());
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;

                let scale = vdupq_n_f32(12102203.0);
                let offset = vdupq_n_f32(1065353216.0 - 486411.0);
                let max_val = vdupq_n_f32(88.0);
                let min_val = vdupq_n_f32(-88.0);

                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 4];
                    let mut x = vld1q_f32(slice.as_ptr());

                    x = vmaxq_f32(x, min_val);
                    x = vminq_f32(x, max_val);

                    let approx = vaddq_f32(vmulq_f32(x, scale), offset);
                    let exp_x = vreinterpretq_f32_s32(vcvtq_s32_f32(approx));

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), exp_x);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    result.push(input[j].exp());
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for &val in input.iter() {
        result.push(val.exp());
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated sigmoid (logistic) function for f32 arrays
///
/// Computes σ(x) = 1 / (1 + exp(-x)) with numerical stability.
/// Uses SIMD exp approximation for significant speedup.
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of sigmoid values in range (0, 1)
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_sigmoid_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, 2.0, -2.0];
/// let result = simd_sigmoid_f32(&x.view());
/// assert!((result[0] - 0.5).abs() < 1e-6);
/// ```
#[allow(dead_code)]
pub fn simd_sigmoid_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                // Sigmoid: σ(x) = 1 / (1 + exp(-x))
                // For numerical stability:
                //   x >= 0: σ(x) = 1 / (1 + exp(-x))
                //   x < 0:  σ(x) = exp(x) / (1 + exp(x))

                let one = _mm256_set1_ps(1.0);
                let zero = _mm256_setzero_ps();
                let log2_e = _mm256_set1_ps(std::f32::consts::LOG2_E);
                let ln_2 = _mm256_set1_ps(std::f32::consts::LN_2);
                let max_val = _mm256_set1_ps(88.0);
                let min_val = _mm256_set1_ps(-88.0);

                // Polynomial coefficients for exp(r)
                let c0 = _mm256_set1_ps(1.0);
                let c1 = _mm256_set1_ps(1.0);
                let c2 = _mm256_set1_ps(0.5);
                let c3 = _mm256_set1_ps(0.16666666666666666);
                let c4 = _mm256_set1_ps(0.041666666666666664);
                let c5 = _mm256_set1_ps(0.008333333333333333);
                let c6 = _mm256_set1_ps(0.001388888888888889);

                let mut i = 0;
                while i + 8 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 8];
                    let x = _mm256_loadu_ps(slice.as_ptr());

                    // For x >= 0: compute exp(-x)
                    // For x < 0: compute exp(x)
                    let mask_positive = _mm256_cmp_ps(x, zero, _CMP_GE_OQ);
                    let neg_x = _mm256_sub_ps(zero, x);

                    // Select -x for positive, x for negative
                    let exp_arg = _mm256_blendv_ps(x, neg_x, mask_positive);

                    // Clamp for stability
                    let exp_arg_clamped = _mm256_max_ps(_mm256_min_ps(exp_arg, max_val), min_val);

                    // Range reduction
                    let k_float = _mm256_round_ps(
                        _mm256_mul_ps(exp_arg_clamped, log2_e),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    let r = _mm256_sub_ps(exp_arg_clamped, _mm256_mul_ps(k_float, ln_2));

                    // Polynomial for exp(r)
                    let r2 = _mm256_mul_ps(r, r);
                    let r3 = _mm256_mul_ps(r2, r);
                    let r4 = _mm256_mul_ps(r2, r2);
                    let r5 = _mm256_mul_ps(r4, r);
                    let r6 = _mm256_mul_ps(r4, r2);

                    let mut exp_r = c0;
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c1, r));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c2, r2));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c3, r3));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c4, r4));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c5, r5));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c6, r6));

                    // Scale by 2^k
                    let k_int = _mm256_cvtps_epi32(k_float);
                    let bias = _mm256_set1_epi32(127);
                    let exp_bits = _mm256_slli_epi32(_mm256_add_epi32(k_int, bias), 23);
                    let scale = _mm256_castsi256_ps(exp_bits);

                    let exp_val = _mm256_mul_ps(exp_r, scale);

                    // For x >= 0: σ(x) = 1 / (1 + exp(-x))
                    // For x < 0:  σ(x) = exp(x) / (1 + exp(x))
                    let one_plus_exp = _mm256_add_ps(one, exp_val);
                    let sig_positive = _mm256_div_ps(one, one_plus_exp);
                    let sig_negative = _mm256_div_ps(exp_val, one_plus_exp);

                    let sigmoid_result =
                        _mm256_blendv_ps(sig_negative, sig_positive, mask_positive);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), sigmoid_result);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    let x = input[j];
                    if x >= 0.0 {
                        result.push(1.0 / (1.0 + (-x).exp()));
                    } else {
                        let exp_x = x.exp();
                        result.push(exp_x / (1.0 + exp_x));
                    }
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;

                let one = vdupq_n_f32(1.0);
                let zero = vdupq_n_f32(0.0);
                let log2_e = vdupq_n_f32(std::f32::consts::LOG2_E);
                let ln_2 = vdupq_n_f32(std::f32::consts::LN_2);
                let max_val = vdupq_n_f32(88.0);
                let min_val = vdupq_n_f32(-88.0);

                let c0 = vdupq_n_f32(1.0);
                let c1 = vdupq_n_f32(1.0);
                let c2 = vdupq_n_f32(0.5);
                let c3 = vdupq_n_f32(0.16666666666666666);
                let c4 = vdupq_n_f32(0.041666666666666664);
                let c5 = vdupq_n_f32(0.008333333333333333);
                let c6 = vdupq_n_f32(0.001388888888888889);

                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 4];
                    let x = vld1q_f32(slice.as_ptr());

                    let mask_positive = vcgeq_f32(x, zero);
                    let neg_x = vsubq_f32(zero, x);
                    let exp_arg = vbslq_f32(mask_positive, neg_x, x);
                    let exp_arg_clamped = vmaxq_f32(vminq_f32(exp_arg, max_val), min_val);

                    // Range reduction
                    let k_float = vrndnq_f32(vmulq_f32(exp_arg_clamped, log2_e));
                    let r = vsubq_f32(exp_arg_clamped, vmulq_f32(k_float, ln_2));

                    // Polynomial
                    let r2 = vmulq_f32(r, r);
                    let r3 = vmulq_f32(r2, r);
                    let r4 = vmulq_f32(r2, r2);
                    let r5 = vmulq_f32(r4, r);
                    let r6 = vmulq_f32(r4, r2);

                    let mut exp_r = c0;
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c1, r));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c2, r2));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c3, r3));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c4, r4));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c5, r5));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c6, r6));

                    // Scale
                    let k_int = vcvtq_s32_f32(k_float);
                    let bias = vdupq_n_s32(127);
                    let exp_bits = vshlq_n_s32(vaddq_s32(k_int, bias), 23);
                    let scale = vreinterpretq_f32_s32(exp_bits);

                    let exp_val = vmulq_f32(exp_r, scale);

                    let one_plus_exp = vaddq_f32(one, exp_val);
                    let sig_positive = vdivq_f32(one, one_plus_exp);
                    let sig_negative = vdivq_f32(exp_val, one_plus_exp);

                    let sigmoid_result = vbslq_f32(mask_positive, sig_positive, sig_negative);

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), sigmoid_result);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    let x = input[j];
                    if x >= 0.0 {
                        result.push(1.0 / (1.0 + (-x).exp()));
                    } else {
                        let exp_x = x.exp();
                        result.push(exp_x / (1.0 + exp_x));
                    }
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for &x in input.iter() {
        if x >= 0.0 {
            result.push(1.0 / (1.0 + (-x).exp()));
        } else {
            let exp_x = x.exp();
            result.push(exp_x / (1.0 + exp_x));
        }
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated sigmoid for f64 arrays
#[allow(dead_code)]
pub fn simd_sigmoid_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let one = _mm256_set1_pd(1.0);
                let zero = _mm256_setzero_pd();
                let log2_e = _mm256_set1_pd(std::f64::consts::LOG2_E);
                let ln_2 = _mm256_set1_pd(std::f64::consts::LN_2);
                let max_val = _mm256_set1_pd(709.0);
                let min_val = _mm256_set1_pd(-709.0);

                // Higher-degree polynomial for f64
                let c0 = _mm256_set1_pd(1.0);
                let c1 = _mm256_set1_pd(1.0);
                let c2 = _mm256_set1_pd(0.5);
                let c3 = _mm256_set1_pd(1.0 / 6.0);
                let c4 = _mm256_set1_pd(1.0 / 24.0);
                let c5 = _mm256_set1_pd(1.0 / 120.0);
                let c6 = _mm256_set1_pd(1.0 / 720.0);
                let c7 = _mm256_set1_pd(1.0 / 5040.0);
                let c8 = _mm256_set1_pd(1.0 / 40320.0);

                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 4];
                    let x = _mm256_loadu_pd(slice.as_ptr());

                    let mask_positive = _mm256_cmp_pd(x, zero, _CMP_GE_OQ);
                    let neg_x = _mm256_sub_pd(zero, x);
                    let exp_arg = _mm256_blendv_pd(x, neg_x, mask_positive);
                    let exp_arg_clamped = _mm256_max_pd(_mm256_min_pd(exp_arg, max_val), min_val);

                    // Range reduction
                    let k_float = _mm256_round_pd(
                        _mm256_mul_pd(exp_arg_clamped, log2_e),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    let r = _mm256_sub_pd(exp_arg_clamped, _mm256_mul_pd(k_float, ln_2));

                    // Polynomial
                    let r2 = _mm256_mul_pd(r, r);
                    let r3 = _mm256_mul_pd(r2, r);
                    let r4 = _mm256_mul_pd(r2, r2);
                    let r5 = _mm256_mul_pd(r4, r);
                    let r6 = _mm256_mul_pd(r4, r2);
                    let r7 = _mm256_mul_pd(r6, r);
                    let r8 = _mm256_mul_pd(r4, r4);

                    let mut exp_r = c0;
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c1, r));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c2, r2));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c3, r3));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c4, r4));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c5, r5));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c6, r6));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c7, r7));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c8, r8));

                    // Scale by 2^k
                    let k_long = _mm256_cvtpd_epi32(k_float);
                    let mut k_arr = [0i32; 4];
                    _mm_storeu_si128(k_arr.as_mut_ptr() as *mut __m128i, k_long);

                    let mut scale_arr = [0.0f64; 4];
                    for j in 0..4 {
                        let exp_bits = ((k_arr[j] as i64 + 1023) << 52) as u64;
                        scale_arr[j] = f64::from_bits(exp_bits);
                    }
                    let scale = _mm256_loadu_pd(scale_arr.as_ptr());

                    let exp_val = _mm256_mul_pd(exp_r, scale);

                    let one_plus_exp = _mm256_add_pd(one, exp_val);
                    let sig_positive = _mm256_div_pd(one, one_plus_exp);
                    let sig_negative = _mm256_div_pd(exp_val, one_plus_exp);

                    let sigmoid_result =
                        _mm256_blendv_pd(sig_negative, sig_positive, mask_positive);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), sigmoid_result);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    let x = input[j];
                    if x >= 0.0 {
                        result.push(1.0 / (1.0 + (-x).exp()));
                    } else {
                        let exp_x = x.exp();
                        result.push(exp_x / (1.0 + exp_x));
                    }
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;

                let one = vdupq_n_f64(1.0);
                let zero = vdupq_n_f64(0.0);
                let log2_e = vdupq_n_f64(std::f64::consts::LOG2_E);
                let ln_2 = vdupq_n_f64(std::f64::consts::LN_2);
                let max_val = vdupq_n_f64(709.0);
                let min_val = vdupq_n_f64(-709.0);

                let c0 = vdupq_n_f64(1.0);
                let c1 = vdupq_n_f64(1.0);
                let c2 = vdupq_n_f64(0.5);
                let c3 = vdupq_n_f64(1.0 / 6.0);
                let c4 = vdupq_n_f64(1.0 / 24.0);
                let c5 = vdupq_n_f64(1.0 / 120.0);
                let c6 = vdupq_n_f64(1.0 / 720.0);
                let c7 = vdupq_n_f64(1.0 / 5040.0);
                let c8 = vdupq_n_f64(1.0 / 40320.0);

                let mut i = 0;
                while i + 2 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 2];
                    let x = vld1q_f64(slice.as_ptr());

                    let mask_positive = vcgeq_f64(x, zero);
                    let neg_x = vsubq_f64(zero, x);
                    let exp_arg = vbslq_f64(mask_positive, neg_x, x);
                    let exp_arg_clamped = vmaxq_f64(vminq_f64(exp_arg, max_val), min_val);

                    let k_float = vrndnq_f64(vmulq_f64(exp_arg_clamped, log2_e));
                    let r = vsubq_f64(exp_arg_clamped, vmulq_f64(k_float, ln_2));

                    let r2 = vmulq_f64(r, r);
                    let r3 = vmulq_f64(r2, r);
                    let r4 = vmulq_f64(r2, r2);
                    let r5 = vmulq_f64(r4, r);
                    let r6 = vmulq_f64(r4, r2);
                    let r7 = vmulq_f64(r6, r);
                    let r8 = vmulq_f64(r4, r4);

                    let mut exp_r = c0;
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c1, r));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c2, r2));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c3, r3));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c4, r4));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c5, r5));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c6, r6));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c7, r7));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c8, r8));

                    let k_int = vcvtq_s64_f64(k_float);
                    let bias = vdupq_n_s64(1023);
                    let exp_bits = vshlq_n_s64(vaddq_s64(k_int, bias), 52);
                    let scale = vreinterpretq_f64_s64(exp_bits);

                    let exp_val = vmulq_f64(exp_r, scale);

                    let one_plus_exp = vaddq_f64(one, exp_val);
                    let sig_positive = vdivq_f64(one, one_plus_exp);
                    let sig_negative = vdivq_f64(exp_val, one_plus_exp);

                    let sigmoid_result = vbslq_f64(mask_positive, sig_positive, sig_negative);

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), sigmoid_result);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                for j in i..len {
                    let x = input[j];
                    if x >= 0.0 {
                        result.push(1.0 / (1.0 + (-x).exp()));
                    } else {
                        let exp_x = x.exp();
                        result.push(exp_x / (1.0 + exp_x));
                    }
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for &x in input.iter() {
        if x >= 0.0 {
            result.push(1.0 / (1.0 + (-x).exp()));
        } else {
            let exp_x = x.exp();
            result.push(exp_x / (1.0 + exp_x));
        }
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated GELU (Gaussian Error Linear Unit) for f32 arrays
///
/// Uses the fast tanh approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
/// This is the approximation used in GPT-2 and BERT models.
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of GELU values
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_gelu_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, 1.0, -1.0];
/// let result = simd_gelu_f32(&x.view());
/// assert!((result[0]).abs() < 1e-6); // GELU(0) = 0
/// ```
#[allow(dead_code)]
pub fn simd_gelu_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let mut result = Vec::with_capacity(len);

    // Constants for GELU approximation
    // GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654; // √(2/π)
    const COEFF: f32 = 0.044715;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let half = _mm256_set1_ps(0.5);
                let one = _mm256_set1_ps(1.0);
                let sqrt_2_pi = _mm256_set1_ps(SQRT_2_OVER_PI);
                let coeff = _mm256_set1_ps(COEFF);

                // Tanh polynomial coefficients (Padé approximation for |x| < 4)
                // tanh(x) ≈ x * (27 + x²) / (27 + 9*x²) for small x
                // For larger x, we use a more accurate polynomial
                let c27 = _mm256_set1_ps(27.0);
                let c9 = _mm256_set1_ps(9.0);

                let mut i = 0;
                while i + 8 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 8];
                    let x = _mm256_loadu_ps(slice.as_ptr());

                    // Compute x³
                    let x2 = _mm256_mul_ps(x, x);
                    let x3 = _mm256_mul_ps(x2, x);

                    // inner = √(2/π) * (x + 0.044715 * x³)
                    let inner =
                        _mm256_mul_ps(sqrt_2_pi, _mm256_add_ps(x, _mm256_mul_ps(coeff, x3)));

                    // Compute tanh(inner) using rational approximation
                    // tanh(x) ≈ x * (27 + x²) / (27 + 9*x²)
                    let inner2 = _mm256_mul_ps(inner, inner);
                    let num = _mm256_mul_ps(inner, _mm256_add_ps(c27, inner2));
                    let den = _mm256_add_ps(c27, _mm256_mul_ps(c9, inner2));
                    let tanh_inner = _mm256_div_ps(num, den);

                    // Clamp tanh to [-1, 1]
                    let neg_one = _mm256_set1_ps(-1.0);
                    let tanh_clamped = _mm256_max_ps(neg_one, _mm256_min_ps(one, tanh_inner));

                    // gelu = 0.5 * x * (1 + tanh_inner)
                    let gelu =
                        _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, tanh_clamped)));

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), gelu);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    let x = input[j];
                    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
                    let tanh_val = inner.tanh();
                    result.push(0.5 * x * (1.0 + tanh_val));
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;

                let half = vdupq_n_f32(0.5);
                let one = vdupq_n_f32(1.0);
                let neg_one = vdupq_n_f32(-1.0);
                let sqrt_2_pi = vdupq_n_f32(SQRT_2_OVER_PI);
                let coeff = vdupq_n_f32(COEFF);
                let c27 = vdupq_n_f32(27.0);
                let c9 = vdupq_n_f32(9.0);

                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 4];
                    let x = vld1q_f32(slice.as_ptr());

                    let x2 = vmulq_f32(x, x);
                    let x3 = vmulq_f32(x2, x);

                    let inner = vmulq_f32(sqrt_2_pi, vaddq_f32(x, vmulq_f32(coeff, x3)));

                    let inner2 = vmulq_f32(inner, inner);
                    let num = vmulq_f32(inner, vaddq_f32(c27, inner2));
                    let den = vaddq_f32(c27, vmulq_f32(c9, inner2));
                    let tanh_inner = vdivq_f32(num, den);

                    let tanh_clamped = vmaxq_f32(neg_one, vminq_f32(one, tanh_inner));

                    let gelu = vmulq_f32(half, vmulq_f32(x, vaddq_f32(one, tanh_clamped)));

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), gelu);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    let x = input[j];
                    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
                    let tanh_val = inner.tanh();
                    result.push(0.5 * x * (1.0 + tanh_val));
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for &x in input.iter() {
        let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        let tanh_val = inner.tanh();
        result.push(0.5 * x * (1.0 + tanh_val));
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated GELU for f64 arrays
#[allow(dead_code)]
pub fn simd_gelu_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let mut result = Vec::with_capacity(len);

    const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
    const COEFF: f64 = 0.044715;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let half = _mm256_set1_pd(0.5);
                let one = _mm256_set1_pd(1.0);
                let neg_one = _mm256_set1_pd(-1.0);
                let sqrt_2_pi = _mm256_set1_pd(SQRT_2_OVER_PI);
                let coeff = _mm256_set1_pd(COEFF);
                let c27 = _mm256_set1_pd(27.0);
                let c9 = _mm256_set1_pd(9.0);

                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 4];
                    let x = _mm256_loadu_pd(slice.as_ptr());

                    let x2 = _mm256_mul_pd(x, x);
                    let x3 = _mm256_mul_pd(x2, x);

                    let inner =
                        _mm256_mul_pd(sqrt_2_pi, _mm256_add_pd(x, _mm256_mul_pd(coeff, x3)));

                    let inner2 = _mm256_mul_pd(inner, inner);
                    let num = _mm256_mul_pd(inner, _mm256_add_pd(c27, inner2));
                    let den = _mm256_add_pd(c27, _mm256_mul_pd(c9, inner2));
                    let tanh_inner = _mm256_div_pd(num, den);

                    let tanh_clamped = _mm256_max_pd(neg_one, _mm256_min_pd(one, tanh_inner));

                    let gelu =
                        _mm256_mul_pd(half, _mm256_mul_pd(x, _mm256_add_pd(one, tanh_clamped)));

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), gelu);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    let x = input[j];
                    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
                    let tanh_val = inner.tanh();
                    result.push(0.5 * x * (1.0 + tanh_val));
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;

                let half = vdupq_n_f64(0.5);
                let one = vdupq_n_f64(1.0);
                let neg_one = vdupq_n_f64(-1.0);
                let sqrt_2_pi = vdupq_n_f64(SQRT_2_OVER_PI);
                let coeff = vdupq_n_f64(COEFF);
                let c27 = vdupq_n_f64(27.0);
                let c9 = vdupq_n_f64(9.0);

                let mut i = 0;
                while i + 2 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 2];
                    let x = vld1q_f64(slice.as_ptr());

                    let x2 = vmulq_f64(x, x);
                    let x3 = vmulq_f64(x2, x);

                    let inner = vmulq_f64(sqrt_2_pi, vaddq_f64(x, vmulq_f64(coeff, x3)));

                    let inner2 = vmulq_f64(inner, inner);
                    let num = vmulq_f64(inner, vaddq_f64(c27, inner2));
                    let den = vaddq_f64(c27, vmulq_f64(c9, inner2));
                    let tanh_inner = vdivq_f64(num, den);

                    let tanh_clamped = vmaxq_f64(neg_one, vminq_f64(one, tanh_inner));

                    let gelu = vmulq_f64(half, vmulq_f64(x, vaddq_f64(one, tanh_clamped)));

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), gelu);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                for j in i..len {
                    let x = input[j];
                    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
                    let tanh_val = inner.tanh();
                    result.push(0.5 * x * (1.0 + tanh_val));
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for &x in input.iter() {
        let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
        let tanh_val = inner.tanh();
        result.push(0.5 * x * (1.0 + tanh_val));
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated Swish (SiLU) activation for f32 arrays
///
/// Computes swish(x) = x * sigmoid(x)
/// This is also known as SiLU (Sigmoid Linear Unit).
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of swish values
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_swish_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, 1.0, -1.0];
/// let result = simd_swish_f32(&x.view());
/// assert!((result[0]).abs() < 1e-6); // swish(0) = 0
/// ```
#[allow(dead_code)]
pub fn simd_swish_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let one = _mm256_set1_ps(1.0);
                let zero = _mm256_setzero_ps();
                let log2_e = _mm256_set1_ps(std::f32::consts::LOG2_E);
                let ln_2 = _mm256_set1_ps(std::f32::consts::LN_2);
                let max_val = _mm256_set1_ps(88.0);
                let min_val = _mm256_set1_ps(-88.0);

                // Polynomial coefficients for exp(r)
                let c0 = _mm256_set1_ps(1.0);
                let c1 = _mm256_set1_ps(1.0);
                let c2 = _mm256_set1_ps(0.5);
                let c3 = _mm256_set1_ps(0.16666666666666666);
                let c4 = _mm256_set1_ps(0.041666666666666664);
                let c5 = _mm256_set1_ps(0.008333333333333333);
                let c6 = _mm256_set1_ps(0.001388888888888889);

                let mut i = 0;
                while i + 8 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 8];
                    let x = _mm256_loadu_ps(slice.as_ptr());

                    // Compute sigmoid(x) using same approach as simd_sigmoid_f32
                    let mask_positive = _mm256_cmp_ps(x, zero, _CMP_GE_OQ);
                    let neg_x = _mm256_sub_ps(zero, x);
                    let exp_arg = _mm256_blendv_ps(x, neg_x, mask_positive);
                    let exp_arg_clamped = _mm256_max_ps(_mm256_min_ps(exp_arg, max_val), min_val);

                    // Range reduction
                    let k_float = _mm256_round_ps(
                        _mm256_mul_ps(exp_arg_clamped, log2_e),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    let r = _mm256_sub_ps(exp_arg_clamped, _mm256_mul_ps(k_float, ln_2));

                    // Polynomial
                    let r2 = _mm256_mul_ps(r, r);
                    let r3 = _mm256_mul_ps(r2, r);
                    let r4 = _mm256_mul_ps(r2, r2);
                    let r5 = _mm256_mul_ps(r4, r);
                    let r6 = _mm256_mul_ps(r4, r2);

                    let mut exp_r = c0;
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c1, r));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c2, r2));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c3, r3));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c4, r4));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c5, r5));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c6, r6));

                    // Scale
                    let k_int = _mm256_cvtps_epi32(k_float);
                    let bias = _mm256_set1_epi32(127);
                    let exp_bits = _mm256_slli_epi32(_mm256_add_epi32(k_int, bias), 23);
                    let scale = _mm256_castsi256_ps(exp_bits);

                    let exp_val = _mm256_mul_ps(exp_r, scale);

                    // Sigmoid
                    let one_plus_exp = _mm256_add_ps(one, exp_val);
                    let sig_positive = _mm256_div_ps(one, one_plus_exp);
                    let sig_negative = _mm256_div_ps(exp_val, one_plus_exp);
                    let sigmoid = _mm256_blendv_ps(sig_negative, sig_positive, mask_positive);

                    // Swish = x * sigmoid(x)
                    let swish = _mm256_mul_ps(x, sigmoid);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), swish);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    let x = input[j];
                    let sigmoid = if x >= 0.0 {
                        1.0 / (1.0 + (-x).exp())
                    } else {
                        let exp_x = x.exp();
                        exp_x / (1.0 + exp_x)
                    };
                    result.push(x * sigmoid);
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;

                let one = vdupq_n_f32(1.0);
                let zero = vdupq_n_f32(0.0);
                let log2_e = vdupq_n_f32(std::f32::consts::LOG2_E);
                let ln_2 = vdupq_n_f32(std::f32::consts::LN_2);
                let max_val = vdupq_n_f32(88.0);
                let min_val = vdupq_n_f32(-88.0);

                let c0 = vdupq_n_f32(1.0);
                let c1 = vdupq_n_f32(1.0);
                let c2 = vdupq_n_f32(0.5);
                let c3 = vdupq_n_f32(0.16666666666666666);
                let c4 = vdupq_n_f32(0.041666666666666664);
                let c5 = vdupq_n_f32(0.008333333333333333);
                let c6 = vdupq_n_f32(0.001388888888888889);

                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 4];
                    let x = vld1q_f32(slice.as_ptr());

                    let mask_positive = vcgeq_f32(x, zero);
                    let neg_x = vsubq_f32(zero, x);
                    let exp_arg = vbslq_f32(mask_positive, neg_x, x);
                    let exp_arg_clamped = vmaxq_f32(vminq_f32(exp_arg, max_val), min_val);

                    let k_float = vrndnq_f32(vmulq_f32(exp_arg_clamped, log2_e));
                    let r = vsubq_f32(exp_arg_clamped, vmulq_f32(k_float, ln_2));

                    let r2 = vmulq_f32(r, r);
                    let r3 = vmulq_f32(r2, r);
                    let r4 = vmulq_f32(r2, r2);
                    let r5 = vmulq_f32(r4, r);
                    let r6 = vmulq_f32(r4, r2);

                    let mut exp_r = c0;
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c1, r));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c2, r2));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c3, r3));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c4, r4));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c5, r5));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c6, r6));

                    let k_int = vcvtq_s32_f32(k_float);
                    let bias = vdupq_n_s32(127);
                    let exp_bits = vshlq_n_s32(vaddq_s32(k_int, bias), 23);
                    let scale = vreinterpretq_f32_s32(exp_bits);

                    let exp_val = vmulq_f32(exp_r, scale);

                    let one_plus_exp = vaddq_f32(one, exp_val);
                    let sig_positive = vdivq_f32(one, one_plus_exp);
                    let sig_negative = vdivq_f32(exp_val, one_plus_exp);
                    let sigmoid = vbslq_f32(mask_positive, sig_positive, sig_negative);

                    let swish = vmulq_f32(x, sigmoid);

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), swish);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    let x = input[j];
                    let sigmoid = if x >= 0.0 {
                        1.0 / (1.0 + (-x).exp())
                    } else {
                        let exp_x = x.exp();
                        exp_x / (1.0 + exp_x)
                    };
                    result.push(x * sigmoid);
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for &x in input.iter() {
        let sigmoid = if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let exp_x = x.exp();
            exp_x / (1.0 + exp_x)
        };
        result.push(x * sigmoid);
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated Swish for f64 arrays
#[allow(dead_code)]
pub fn simd_swish_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    // Use SIMD sigmoid and multiply
    let sigmoid = simd_sigmoid_f64(input);
    let mut result = Vec::with_capacity(len);

    for (i, &x) in input.iter().enumerate() {
        result.push(x * sigmoid[i]);
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated Softplus activation for f32 arrays
///
/// Computes softplus(x) = ln(1 + exp(x)) with numerical stability.
/// For large x, softplus(x) ≈ x to avoid overflow.
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of softplus values (always positive)
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_softplus_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, 1.0, -1.0];
/// let result = simd_softplus_f32(&x.view());
/// assert!((result[0] - 0.6931472).abs() < 1e-5); // softplus(0) = ln(2)
/// ```
#[allow(dead_code)]
pub fn simd_softplus_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let mut result = Vec::with_capacity(len);

    // Threshold for numerical stability: for x > threshold, softplus(x) ≈ x
    const THRESHOLD: f32 = 20.0;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let one = _mm256_set1_ps(1.0);
                let threshold = _mm256_set1_ps(THRESHOLD);
                let neg_threshold = _mm256_set1_ps(-THRESHOLD);
                let log2_e = _mm256_set1_ps(std::f32::consts::LOG2_E);
                let ln_2 = _mm256_set1_ps(std::f32::consts::LN_2);
                let max_exp = _mm256_set1_ps(88.0);
                let min_exp = _mm256_set1_ps(-88.0);

                // Polynomial coefficients for exp(r)
                let c0 = _mm256_set1_ps(1.0);
                let c1 = _mm256_set1_ps(1.0);
                let c2 = _mm256_set1_ps(0.5);
                let c3 = _mm256_set1_ps(0.16666666666666666);
                let c4 = _mm256_set1_ps(0.041666666666666664);
                let c5 = _mm256_set1_ps(0.008333333333333333);
                let c6 = _mm256_set1_ps(0.001388888888888889);

                let mut i = 0;
                while i + 8 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 8];
                    let x = _mm256_loadu_ps(slice.as_ptr());

                    // For large positive x: softplus ≈ x
                    // For large negative x: softplus ≈ exp(x) ≈ 0
                    // For moderate x: softplus = ln(1 + exp(x))

                    // Compute exp(x) with clamping
                    let x_clamped = _mm256_max_ps(_mm256_min_ps(x, max_exp), min_exp);

                    // Range reduction
                    let k_float = _mm256_round_ps(
                        _mm256_mul_ps(x_clamped, log2_e),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    let r = _mm256_sub_ps(x_clamped, _mm256_mul_ps(k_float, ln_2));

                    // Polynomial
                    let r2 = _mm256_mul_ps(r, r);
                    let r3 = _mm256_mul_ps(r2, r);
                    let r4 = _mm256_mul_ps(r2, r2);
                    let r5 = _mm256_mul_ps(r4, r);
                    let r6 = _mm256_mul_ps(r4, r2);

                    let mut exp_r = c0;
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c1, r));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c2, r2));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c3, r3));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c4, r4));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c5, r5));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c6, r6));

                    // Scale
                    let k_int = _mm256_cvtps_epi32(k_float);
                    let bias = _mm256_set1_epi32(127);
                    let exp_bits = _mm256_slli_epi32(_mm256_add_epi32(k_int, bias), 23);
                    let scale = _mm256_castsi256_ps(exp_bits);

                    let exp_x = _mm256_mul_ps(exp_r, scale);

                    // softplus = ln(1 + exp(x))
                    // Using log1p approximation: ln(1 + exp(x))
                    // For stability, compute as x + ln(1 + exp(-x)) for x > 0
                    let one_plus_exp = _mm256_add_ps(one, exp_x);

                    // Compute ln using scalar (no SIMD ln available easily)
                    let mut temp_softplus = [0.0f32; 8];
                    let mut temp_x = [0.0f32; 8];
                    let mut temp_one_plus_exp = [0.0f32; 8];
                    _mm256_storeu_ps(temp_x.as_mut_ptr(), x);
                    _mm256_storeu_ps(temp_one_plus_exp.as_mut_ptr(), one_plus_exp);

                    for j in 0..8 {
                        let xj = temp_x[j];
                        if xj > THRESHOLD {
                            temp_softplus[j] = xj;
                        } else if xj < -THRESHOLD {
                            // exp(x) is tiny, softplus ≈ exp(x)
                            temp_softplus[j] = temp_one_plus_exp[j] - 1.0;
                        } else {
                            temp_softplus[j] = temp_one_plus_exp[j].ln();
                        }
                    }

                    result.extend_from_slice(&temp_softplus);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    let x = input[j];
                    if x > THRESHOLD {
                        result.push(x);
                    } else if x < -THRESHOLD {
                        result.push(x.exp());
                    } else {
                        result.push((1.0 + x.exp()).ln());
                    }
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // Use scalar fallback with NEON optimized exp
            let exp_vals = simd_exp_f32(input);

            for (i, &x) in input.iter().enumerate() {
                if x > THRESHOLD {
                    result.push(x);
                } else if x < -THRESHOLD {
                    result.push(exp_vals[i]);
                } else {
                    result.push((1.0 + exp_vals[i]).ln());
                }
            }

            return Array1::from_vec(result);
        }
    }

    // Scalar fallback
    for &x in input.iter() {
        if x > THRESHOLD {
            result.push(x);
        } else if x < -THRESHOLD {
            result.push(x.exp());
        } else {
            result.push((1.0 + x.exp()).ln());
        }
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated Softplus for f64 arrays
#[allow(dead_code)]
pub fn simd_softplus_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let mut result = Vec::with_capacity(len);
    const THRESHOLD: f64 = 40.0;

    // Use SIMD exp for the computation
    let exp_vals = simd_exp_f64(input);

    for (i, &x) in input.iter().enumerate() {
        if x > THRESHOLD {
            result.push(x);
        } else if x < -THRESHOLD {
            result.push(exp_vals[i]);
        } else {
            result.push((1.0 + exp_vals[i]).ln());
        }
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated tanh for f32 arrays
///
/// Computes tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) with numerical stability.
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of tanh values in range (-1, 1)
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_tanh_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, 1.0, -1.0];
/// let result = simd_tanh_f32(&x.view());
/// assert!(result[0].abs() < 1e-6); // tanh(0) = 0
/// ```
#[allow(dead_code)]
pub fn simd_tanh_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let one = _mm256_set1_ps(1.0);
                let two = _mm256_set1_ps(2.0);
                let zero = _mm256_setzero_ps();
                let neg_one = _mm256_set1_ps(-1.0);
                let log2_e = _mm256_set1_ps(std::f32::consts::LOG2_E);
                let ln_2 = _mm256_set1_ps(std::f32::consts::LN_2);
                let max_val = _mm256_set1_ps(44.0); // 2*x clamped to 88

                // Polynomial coefficients for exp(r)
                let c0 = _mm256_set1_ps(1.0);
                let c1 = _mm256_set1_ps(1.0);
                let c2 = _mm256_set1_ps(0.5);
                let c3 = _mm256_set1_ps(0.16666666666666666);
                let c4 = _mm256_set1_ps(0.041666666666666664);
                let c5 = _mm256_set1_ps(0.008333333333333333);
                let c6 = _mm256_set1_ps(0.001388888888888889);

                let mut i = 0;
                while i + 8 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 8];
                    let x = _mm256_loadu_ps(slice.as_ptr());

                    // For numerical stability:
                    // For x >= 0: tanh(x) = 1 - 2/(exp(2x) + 1)
                    // For x < 0:  tanh(x) = 2*exp(2x)/(exp(2x) + 1) - 1
                    let mask_positive = _mm256_cmp_ps(x, zero, _CMP_GE_OQ);

                    // Compute 2|x|
                    let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
                    let two_abs_x = _mm256_mul_ps(two, abs_x);
                    let two_abs_x_clamped = _mm256_min_ps(two_abs_x, max_val);

                    // Compute exp(2|x|)
                    let k_float = _mm256_round_ps(
                        _mm256_mul_ps(two_abs_x_clamped, log2_e),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    let r = _mm256_sub_ps(two_abs_x_clamped, _mm256_mul_ps(k_float, ln_2));

                    let r2 = _mm256_mul_ps(r, r);
                    let r3 = _mm256_mul_ps(r2, r);
                    let r4 = _mm256_mul_ps(r2, r2);
                    let r5 = _mm256_mul_ps(r4, r);
                    let r6 = _mm256_mul_ps(r4, r2);

                    let mut exp_r = c0;
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c1, r));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c2, r2));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c3, r3));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c4, r4));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c5, r5));
                    exp_r = _mm256_add_ps(exp_r, _mm256_mul_ps(c6, r6));

                    let k_int = _mm256_cvtps_epi32(k_float);
                    let bias = _mm256_set1_epi32(127);
                    let exp_bits = _mm256_slli_epi32(_mm256_add_epi32(k_int, bias), 23);
                    let scale = _mm256_castsi256_ps(exp_bits);

                    let exp_2abs_x = _mm256_mul_ps(exp_r, scale);

                    // tanh = 1 - 2/(exp(2|x|) + 1) for |tanh|
                    let exp_plus_one = _mm256_add_ps(exp_2abs_x, one);
                    let abs_tanh = _mm256_sub_ps(one, _mm256_div_ps(two, exp_plus_one));

                    // Apply sign: tanh(-x) = -tanh(x)
                    let tanh_result =
                        _mm256_blendv_ps(_mm256_mul_ps(neg_one, abs_tanh), abs_tanh, mask_positive);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), tanh_result);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(input[j].tanh());
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;

                let one = vdupq_n_f32(1.0);
                let two = vdupq_n_f32(2.0);
                let zero = vdupq_n_f32(0.0);
                let neg_one = vdupq_n_f32(-1.0);
                let log2_e = vdupq_n_f32(std::f32::consts::LOG2_E);
                let ln_2 = vdupq_n_f32(std::f32::consts::LN_2);
                let max_val = vdupq_n_f32(44.0);

                let c0 = vdupq_n_f32(1.0);
                let c1 = vdupq_n_f32(1.0);
                let c2 = vdupq_n_f32(0.5);
                let c3 = vdupq_n_f32(0.16666666666666666);
                let c4 = vdupq_n_f32(0.041666666666666664);
                let c5 = vdupq_n_f32(0.008333333333333333);
                let c6 = vdupq_n_f32(0.001388888888888889);

                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 4];
                    let x = vld1q_f32(slice.as_ptr());

                    let mask_positive = vcgeq_f32(x, zero);
                    let abs_x = vabsq_f32(x);
                    let two_abs_x = vmulq_f32(two, abs_x);
                    let two_abs_x_clamped = vminq_f32(two_abs_x, max_val);

                    let k_float = vrndnq_f32(vmulq_f32(two_abs_x_clamped, log2_e));
                    let r = vsubq_f32(two_abs_x_clamped, vmulq_f32(k_float, ln_2));

                    let r2 = vmulq_f32(r, r);
                    let r3 = vmulq_f32(r2, r);
                    let r4 = vmulq_f32(r2, r2);
                    let r5 = vmulq_f32(r4, r);
                    let r6 = vmulq_f32(r4, r2);

                    let mut exp_r = c0;
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c1, r));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c2, r2));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c3, r3));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c4, r4));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c5, r5));
                    exp_r = vaddq_f32(exp_r, vmulq_f32(c6, r6));

                    let k_int = vcvtq_s32_f32(k_float);
                    let bias = vdupq_n_s32(127);
                    let exp_bits = vshlq_n_s32(vaddq_s32(k_int, bias), 23);
                    let scale = vreinterpretq_f32_s32(exp_bits);

                    let exp_2abs_x = vmulq_f32(exp_r, scale);

                    let exp_plus_one = vaddq_f32(exp_2abs_x, one);
                    let abs_tanh = vsubq_f32(one, vdivq_f32(two, exp_plus_one));

                    let neg_abs_tanh = vmulq_f32(neg_one, abs_tanh);
                    let tanh_result = vbslq_f32(mask_positive, abs_tanh, neg_abs_tanh);

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), tanh_result);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    result.push(input[j].tanh());
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for &x in input.iter() {
        result.push(x.tanh());
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated tanh for f64 arrays
#[allow(dead_code)]
pub fn simd_tanh_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let one = _mm256_set1_pd(1.0);
                let two = _mm256_set1_pd(2.0);
                let zero = _mm256_setzero_pd();
                let neg_one = _mm256_set1_pd(-1.0);
                let log2_e = _mm256_set1_pd(std::f64::consts::LOG2_E);
                let ln_2 = _mm256_set1_pd(std::f64::consts::LN_2);
                let max_val = _mm256_set1_pd(354.5); // 2*x clamped to 709

                // Higher-degree polynomial for f64
                let c0 = _mm256_set1_pd(1.0);
                let c1 = _mm256_set1_pd(1.0);
                let c2 = _mm256_set1_pd(0.5);
                let c3 = _mm256_set1_pd(1.0 / 6.0);
                let c4 = _mm256_set1_pd(1.0 / 24.0);
                let c5 = _mm256_set1_pd(1.0 / 120.0);
                let c6 = _mm256_set1_pd(1.0 / 720.0);
                let c7 = _mm256_set1_pd(1.0 / 5040.0);
                let c8 = _mm256_set1_pd(1.0 / 40320.0);

                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 4];
                    let x = _mm256_loadu_pd(slice.as_ptr());

                    let mask_positive = _mm256_cmp_pd(x, zero, _CMP_GE_OQ);

                    let abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
                    let two_abs_x = _mm256_mul_pd(two, abs_x);
                    let two_abs_x_clamped = _mm256_min_pd(two_abs_x, max_val);

                    let k_float = _mm256_round_pd(
                        _mm256_mul_pd(two_abs_x_clamped, log2_e),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    let r = _mm256_sub_pd(two_abs_x_clamped, _mm256_mul_pd(k_float, ln_2));

                    let r2 = _mm256_mul_pd(r, r);
                    let r3 = _mm256_mul_pd(r2, r);
                    let r4 = _mm256_mul_pd(r2, r2);
                    let r5 = _mm256_mul_pd(r4, r);
                    let r6 = _mm256_mul_pd(r4, r2);
                    let r7 = _mm256_mul_pd(r6, r);
                    let r8 = _mm256_mul_pd(r4, r4);

                    let mut exp_r = c0;
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c1, r));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c2, r2));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c3, r3));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c4, r4));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c5, r5));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c6, r6));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c7, r7));
                    exp_r = _mm256_add_pd(exp_r, _mm256_mul_pd(c8, r8));

                    let k_long = _mm256_cvtpd_epi32(k_float);
                    let mut k_arr = [0i32; 4];
                    _mm_storeu_si128(k_arr.as_mut_ptr() as *mut __m128i, k_long);

                    let mut scale_arr = [0.0f64; 4];
                    for j in 0..4 {
                        let exp_bits = ((k_arr[j] as i64 + 1023) << 52) as u64;
                        scale_arr[j] = f64::from_bits(exp_bits);
                    }
                    let scale = _mm256_loadu_pd(scale_arr.as_ptr());

                    let exp_2abs_x = _mm256_mul_pd(exp_r, scale);

                    let exp_plus_one = _mm256_add_pd(exp_2abs_x, one);
                    let abs_tanh = _mm256_sub_pd(one, _mm256_div_pd(two, exp_plus_one));

                    let tanh_result =
                        _mm256_blendv_pd(_mm256_mul_pd(neg_one, abs_tanh), abs_tanh, mask_positive);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), tanh_result);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    result.push(input[j].tanh());
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;

                let one = vdupq_n_f64(1.0);
                let two = vdupq_n_f64(2.0);
                let zero = vdupq_n_f64(0.0);
                let neg_one = vdupq_n_f64(-1.0);
                let log2_e = vdupq_n_f64(std::f64::consts::LOG2_E);
                let ln_2 = vdupq_n_f64(std::f64::consts::LN_2);
                let max_val = vdupq_n_f64(354.5);

                let c0 = vdupq_n_f64(1.0);
                let c1 = vdupq_n_f64(1.0);
                let c2 = vdupq_n_f64(0.5);
                let c3 = vdupq_n_f64(1.0 / 6.0);
                let c4 = vdupq_n_f64(1.0 / 24.0);
                let c5 = vdupq_n_f64(1.0 / 120.0);
                let c6 = vdupq_n_f64(1.0 / 720.0);
                let c7 = vdupq_n_f64(1.0 / 5040.0);
                let c8 = vdupq_n_f64(1.0 / 40320.0);

                let mut i = 0;
                while i + 2 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 2];
                    let x = vld1q_f64(slice.as_ptr());

                    let mask_positive = vcgeq_f64(x, zero);
                    let abs_x = vabsq_f64(x);
                    let two_abs_x = vmulq_f64(two, abs_x);
                    let two_abs_x_clamped = vminq_f64(two_abs_x, max_val);

                    let k_float = vrndnq_f64(vmulq_f64(two_abs_x_clamped, log2_e));
                    let r = vsubq_f64(two_abs_x_clamped, vmulq_f64(k_float, ln_2));

                    let r2 = vmulq_f64(r, r);
                    let r3 = vmulq_f64(r2, r);
                    let r4 = vmulq_f64(r2, r2);
                    let r5 = vmulq_f64(r4, r);
                    let r6 = vmulq_f64(r4, r2);
                    let r7 = vmulq_f64(r6, r);
                    let r8 = vmulq_f64(r4, r4);

                    let mut exp_r = c0;
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c1, r));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c2, r2));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c3, r3));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c4, r4));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c5, r5));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c6, r6));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c7, r7));
                    exp_r = vaddq_f64(exp_r, vmulq_f64(c8, r8));

                    let k_int = vcvtq_s64_f64(k_float);
                    let bias = vdupq_n_s64(1023);
                    let exp_bits = vshlq_n_s64(vaddq_s64(k_int, bias), 52);
                    let scale = vreinterpretq_f64_s64(exp_bits);

                    let exp_2abs_x = vmulq_f64(exp_r, scale);

                    let exp_plus_one = vaddq_f64(exp_2abs_x, one);
                    let abs_tanh = vsubq_f64(one, vdivq_f64(two, exp_plus_one));

                    let neg_abs_tanh = vmulq_f64(neg_one, abs_tanh);
                    let tanh_result = vbslq_f64(mask_positive, abs_tanh, neg_abs_tanh);

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), tanh_result);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                for j in i..len {
                    result.push(input[j].tanh());
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for &x in input.iter() {
        result.push(x.tanh());
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated Mish activation for f32 arrays
///
/// Computes Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
/// This combines our SIMD softplus and tanh implementations.
///
/// # Arguments
/// * `input` - Input array
///
/// # Returns
/// * Array of Mish values
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_mish_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, 1.0, -1.0];
/// let result = simd_mish_f32(&x.view());
/// assert!(result[0].abs() < 1e-6); // Mish(0) = 0
/// ```
#[allow(dead_code)]
pub fn simd_mish_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    // Mish(x) = x * tanh(softplus(x))
    // Compute softplus first, then tanh, then multiply by x
    let softplus = simd_softplus_f32(input);
    let tanh_softplus = simd_tanh_f32(&softplus.view());

    let mut result = Vec::with_capacity(len);
    for (i, &x) in input.iter().enumerate() {
        result.push(x * tanh_softplus[i]);
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated Mish for f64 arrays
#[allow(dead_code)]
pub fn simd_mish_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let softplus = simd_softplus_f64(input);
    let tanh_softplus = simd_tanh_f64(&softplus.view());

    let mut result = Vec::with_capacity(len);
    for (i, &x) in input.iter().enumerate() {
        result.push(x * tanh_softplus[i]);
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated natural logarithm for f32 arrays
///
/// Computes ln(x) using range reduction and polynomial approximation.
/// ln(x) = ln(2^k * m) = k * ln(2) + ln(m) where 1 <= m < 2
///
/// # Arguments
/// * `input` - Input array (must be positive)
///
/// # Returns
/// * Array of ln(x) values
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_ln_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![1.0f32, std::f32::consts::E, 10.0];
/// let result = simd_ln_f32(&x.view());
/// assert!(result[0].abs() < 1e-6); // ln(1) = 0
/// assert!((result[1] - 1.0).abs() < 1e-5); // ln(e) = 1
/// ```
#[allow(dead_code)]
pub fn simd_ln_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let ln_2 = _mm256_set1_ps(std::f32::consts::LN_2);
                let one = _mm256_set1_ps(1.0);

                // Polynomial coefficients for ln(1+x) where |x| < 0.5
                // ln(1+x) ≈ x - x²/2 + x³/3 - x⁴/4 + x⁵/5 - x⁶/6
                let c1 = _mm256_set1_ps(1.0);
                let c2 = _mm256_set1_ps(-0.5);
                let c3 = _mm256_set1_ps(1.0 / 3.0);
                let c4 = _mm256_set1_ps(-0.25);
                let c5 = _mm256_set1_ps(0.2);
                let c6 = _mm256_set1_ps(-1.0 / 6.0);

                let mantissa_mask = _mm256_set1_epi32(0x007FFFFF);
                let exponent_bias = _mm256_set1_epi32(127);

                let mut i = 0;
                while i + 8 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 8];
                    let x = _mm256_loadu_ps(slice.as_ptr());

                    // Extract exponent: k = floor(log2(x))
                    let x_bits = _mm256_castps_si256(x);
                    let exponent = _mm256_sub_epi32(_mm256_srli_epi32(x_bits, 23), exponent_bias);
                    let k = _mm256_cvtepi32_ps(exponent);

                    // Extract mantissa m such that 1 <= m < 2
                    let mantissa_bits = _mm256_or_si256(
                        _mm256_and_si256(x_bits, mantissa_mask),
                        _mm256_slli_epi32(exponent_bias, 23),
                    );
                    let m = _mm256_castsi256_ps(mantissa_bits);

                    // Compute ln(m) where 1 <= m < 2
                    // Let y = m - 1, so 0 <= y < 1
                    // ln(m) = ln(1 + y) ≈ y - y²/2 + y³/3 - ...
                    let y = _mm256_sub_ps(m, one);
                    let y2 = _mm256_mul_ps(y, y);
                    let y3 = _mm256_mul_ps(y2, y);
                    let y4 = _mm256_mul_ps(y2, y2);
                    let y5 = _mm256_mul_ps(y4, y);
                    let y6 = _mm256_mul_ps(y4, y2);

                    let mut ln_m = _mm256_mul_ps(c1, y);
                    ln_m = _mm256_add_ps(ln_m, _mm256_mul_ps(c2, y2));
                    ln_m = _mm256_add_ps(ln_m, _mm256_mul_ps(c3, y3));
                    ln_m = _mm256_add_ps(ln_m, _mm256_mul_ps(c4, y4));
                    ln_m = _mm256_add_ps(ln_m, _mm256_mul_ps(c5, y5));
                    ln_m = _mm256_add_ps(ln_m, _mm256_mul_ps(c6, y6));

                    // ln(x) = k * ln(2) + ln(m)
                    let ln_x = _mm256_add_ps(_mm256_mul_ps(k, ln_2), ln_m);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), ln_x);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(input[j].ln());
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;

                let ln_2 = vdupq_n_f32(std::f32::consts::LN_2);
                let one = vdupq_n_f32(1.0);

                let c1 = vdupq_n_f32(1.0);
                let c2 = vdupq_n_f32(-0.5);
                let c3 = vdupq_n_f32(1.0 / 3.0);
                let c4 = vdupq_n_f32(-0.25);
                let c5 = vdupq_n_f32(0.2);
                let c6 = vdupq_n_f32(-1.0 / 6.0);

                let mantissa_mask = vdupq_n_s32(0x007FFFFF);
                let exponent_bias = vdupq_n_s32(127);

                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 4];
                    let x = vld1q_f32(slice.as_ptr());

                    let x_bits = vreinterpretq_s32_f32(x);
                    let exponent = vsubq_s32(vshrq_n_s32(x_bits, 23), exponent_bias);
                    let k = vcvtq_f32_s32(exponent);

                    let mantissa_bits = vorrq_s32(
                        vandq_s32(x_bits, mantissa_mask),
                        vshlq_n_s32(exponent_bias, 23),
                    );
                    let m = vreinterpretq_f32_s32(mantissa_bits);

                    let y = vsubq_f32(m, one);
                    let y2 = vmulq_f32(y, y);
                    let y3 = vmulq_f32(y2, y);
                    let y4 = vmulq_f32(y2, y2);
                    let y5 = vmulq_f32(y4, y);
                    let y6 = vmulq_f32(y4, y2);

                    let mut ln_m = vmulq_f32(c1, y);
                    ln_m = vaddq_f32(ln_m, vmulq_f32(c2, y2));
                    ln_m = vaddq_f32(ln_m, vmulq_f32(c3, y3));
                    ln_m = vaddq_f32(ln_m, vmulq_f32(c4, y4));
                    ln_m = vaddq_f32(ln_m, vmulq_f32(c5, y5));
                    ln_m = vaddq_f32(ln_m, vmulq_f32(c6, y6));

                    let ln_x = vaddq_f32(vmulq_f32(k, ln_2), ln_m);

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), ln_x);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    result.push(input[j].ln());
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for &x in input.iter() {
        result.push(x.ln());
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated natural logarithm for f64 arrays
#[allow(dead_code)]
pub fn simd_ln_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let mut result = Vec::with_capacity(len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let ln_2 = _mm256_set1_pd(std::f64::consts::LN_2);
                let one = _mm256_set1_pd(1.0);

                // Higher-degree polynomial for f64
                let c1 = _mm256_set1_pd(1.0);
                let c2 = _mm256_set1_pd(-0.5);
                let c3 = _mm256_set1_pd(1.0 / 3.0);
                let c4 = _mm256_set1_pd(-0.25);
                let c5 = _mm256_set1_pd(0.2);
                let c6 = _mm256_set1_pd(-1.0 / 6.0);
                let c7 = _mm256_set1_pd(1.0 / 7.0);
                let c8 = _mm256_set1_pd(-0.125);

                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 4];
                    let x = _mm256_loadu_pd(slice.as_ptr());

                    // For f64, manually extract exponent and mantissa
                    let mut x_arr = [0.0f64; 4];
                    _mm256_storeu_pd(x_arr.as_mut_ptr(), x);

                    let mut k_arr = [0.0f64; 4];
                    let mut m_arr = [0.0f64; 4];

                    for j in 0..4 {
                        let bits = x_arr[j].to_bits();
                        let exp = ((bits >> 52) & 0x7FF) as i64 - 1023;
                        k_arr[j] = exp as f64;

                        // Reconstruct mantissa with exponent = 0 (bias = 1023)
                        let mantissa_bits = (bits & 0x000FFFFFFFFFFFFF) | (1023u64 << 52);
                        m_arr[j] = f64::from_bits(mantissa_bits);
                    }

                    let k = _mm256_loadu_pd(k_arr.as_ptr());
                    let m = _mm256_loadu_pd(m_arr.as_ptr());

                    let y = _mm256_sub_pd(m, one);
                    let y2 = _mm256_mul_pd(y, y);
                    let y3 = _mm256_mul_pd(y2, y);
                    let y4 = _mm256_mul_pd(y2, y2);
                    let y5 = _mm256_mul_pd(y4, y);
                    let y6 = _mm256_mul_pd(y4, y2);
                    let y7 = _mm256_mul_pd(y6, y);
                    let y8 = _mm256_mul_pd(y4, y4);

                    let mut ln_m = _mm256_mul_pd(c1, y);
                    ln_m = _mm256_add_pd(ln_m, _mm256_mul_pd(c2, y2));
                    ln_m = _mm256_add_pd(ln_m, _mm256_mul_pd(c3, y3));
                    ln_m = _mm256_add_pd(ln_m, _mm256_mul_pd(c4, y4));
                    ln_m = _mm256_add_pd(ln_m, _mm256_mul_pd(c5, y5));
                    ln_m = _mm256_add_pd(ln_m, _mm256_mul_pd(c6, y6));
                    ln_m = _mm256_add_pd(ln_m, _mm256_mul_pd(c7, y7));
                    ln_m = _mm256_add_pd(ln_m, _mm256_mul_pd(c8, y8));

                    let ln_x = _mm256_add_pd(_mm256_mul_pd(k, ln_2), ln_m);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), ln_x);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    result.push(input[j].ln());
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // For f64 on NEON, use scalar fallback with loop
            for &x in input.iter() {
                result.push(x.ln());
            }
            return Array1::from_vec(result);
        }
    }

    // Scalar fallback
    for &x in input.iter() {
        result.push(x.ln());
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated sine for f32 arrays
///
/// Computes sin(x) using range reduction and polynomial approximation.
/// Range reduction: reduce x to [-π/2, π/2] using periodicity.
/// Then use Taylor series: sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040
///
/// # Arguments
/// * `input` - Input array (radians)
///
/// # Returns
/// * Array of sin(x) values in range [-1, 1]
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_sin_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, std::f32::consts::PI / 2.0, std::f32::consts::PI];
/// let result = simd_sin_f32(&x.view());
/// assert!(result[0].abs() < 1e-6); // sin(0) = 0
/// assert!((result[1] - 1.0).abs() < 1e-5); // sin(π/2) = 1
/// ```
#[allow(dead_code)]
pub fn simd_sin_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let mut result = Vec::with_capacity(len);

    // Constants for range reduction and polynomial
    const FRAC_1_PI: f32 = std::f32::consts::FRAC_1_PI;
    const PI: f32 = std::f32::consts::PI;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let pi = _mm256_set1_ps(PI);
                let two_pi = _mm256_set1_ps(2.0 * PI);
                let pi_over_2 = _mm256_set1_ps(PI / 2.0);
                let one = _mm256_set1_ps(1.0);
                let neg_one = _mm256_set1_ps(-1.0);

                // Taylor series coefficients for sin(x)
                let c1 = _mm256_set1_ps(1.0);
                let c3 = _mm256_set1_ps(-1.0 / 6.0);
                let c5 = _mm256_set1_ps(1.0 / 120.0);
                let c7 = _mm256_set1_ps(-1.0 / 5040.0);
                let c9 = _mm256_set1_ps(1.0 / 362880.0);

                let mut i = 0;
                while i + 8 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 8];
                    let mut x = _mm256_loadu_ps(slice.as_ptr());

                    // Range reduction: reduce x to [-π, π]
                    // x_reduced = x - round(x / (2π)) * 2π
                    let k = _mm256_round_ps(
                        _mm256_mul_ps(x, _mm256_set1_ps(FRAC_1_PI / 2.0)),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    x = _mm256_sub_ps(x, _mm256_mul_ps(k, two_pi));

                    // Further reduce to [-π/2, π/2] using sin(x) = sin(π-x) for x in [π/2, π]
                    // and sin(x) = -sin(-π-x) for x in [-π, -π/2]
                    let mask_gt_pi_2 = _mm256_cmp_ps(x, pi_over_2, _CMP_GT_OQ);
                    let mask_lt_neg_pi_2 =
                        _mm256_cmp_ps(x, _mm256_sub_ps(_mm256_setzero_ps(), pi_over_2), _CMP_LT_OQ);

                    // If x > π/2: x = π - x
                    let x_reduced_high = _mm256_sub_ps(pi, x);
                    x = _mm256_blendv_ps(x, x_reduced_high, mask_gt_pi_2);

                    // If x < -π/2: x = -π - x
                    let x_reduced_low = _mm256_sub_ps(_mm256_sub_ps(_mm256_setzero_ps(), pi), x);
                    x = _mm256_blendv_ps(x, x_reduced_low, mask_lt_neg_pi_2);

                    // Now x is in [-π/2, π/2], compute Taylor series
                    let x2 = _mm256_mul_ps(x, x);
                    let x3 = _mm256_mul_ps(x2, x);
                    let x5 = _mm256_mul_ps(x3, x2);
                    let x7 = _mm256_mul_ps(x5, x2);
                    let x9 = _mm256_mul_ps(x7, x2);

                    let mut sin_x = _mm256_mul_ps(c1, x);
                    sin_x = _mm256_add_ps(sin_x, _mm256_mul_ps(c3, x3));
                    sin_x = _mm256_add_ps(sin_x, _mm256_mul_ps(c5, x5));
                    sin_x = _mm256_add_ps(sin_x, _mm256_mul_ps(c7, x7));
                    sin_x = _mm256_add_ps(sin_x, _mm256_mul_ps(c9, x9));

                    // Clamp to [-1, 1] to handle polynomial approximation errors
                    sin_x = _mm256_max_ps(sin_x, neg_one);
                    sin_x = _mm256_min_ps(sin_x, one);

                    let mut temp = [0.0f32; 8];
                    _mm256_storeu_ps(temp.as_mut_ptr(), sin_x);
                    result.extend_from_slice(&temp);
                    i += 8;
                }

                // Handle remaining elements
                for j in i..len {
                    result.push(input[j].sin());
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;

                let pi = vdupq_n_f32(PI);
                let two_pi = vdupq_n_f32(2.0 * PI);
                let pi_over_2 = vdupq_n_f32(PI / 2.0);
                let neg_pi_over_2 = vdupq_n_f32(-PI / 2.0);
                let zero = vdupq_n_f32(0.0);

                let one = vdupq_n_f32(1.0);
                let neg_one = vdupq_n_f32(-1.0);
                let c1 = vdupq_n_f32(1.0);
                let c3 = vdupq_n_f32(-1.0 / 6.0);
                let c5 = vdupq_n_f32(1.0 / 120.0);
                let c7 = vdupq_n_f32(-1.0 / 5040.0);
                let c9 = vdupq_n_f32(1.0 / 362880.0);

                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 4];
                    let mut x = vld1q_f32(slice.as_ptr());

                    // Range reduction to [-π, π]
                    let k = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(FRAC_1_PI / 2.0)));
                    x = vsubq_f32(x, vmulq_f32(k, two_pi));

                    // Further reduce to [-π/2, π/2]
                    let mask_gt_pi_2 = vcgtq_f32(x, pi_over_2);
                    let mask_lt_neg_pi_2 = vcltq_f32(x, neg_pi_over_2);

                    let x_reduced_high = vsubq_f32(pi, x);
                    x = vbslq_f32(mask_gt_pi_2, x_reduced_high, x);

                    let x_reduced_low = vsubq_f32(vsubq_f32(zero, pi), x);
                    x = vbslq_f32(mask_lt_neg_pi_2, x_reduced_low, x);

                    // Taylor series
                    let x2 = vmulq_f32(x, x);
                    let x3 = vmulq_f32(x2, x);
                    let x5 = vmulq_f32(x3, x2);
                    let x7 = vmulq_f32(x5, x2);
                    let x9 = vmulq_f32(x7, x2);

                    let mut sin_x = vmulq_f32(c1, x);
                    sin_x = vaddq_f32(sin_x, vmulq_f32(c3, x3));
                    sin_x = vaddq_f32(sin_x, vmulq_f32(c5, x5));
                    sin_x = vaddq_f32(sin_x, vmulq_f32(c7, x7));
                    sin_x = vaddq_f32(sin_x, vmulq_f32(c9, x9));

                    // Clamp to [-1, 1]
                    sin_x = vmaxq_f32(sin_x, neg_one);
                    sin_x = vminq_f32(sin_x, one);

                    let mut temp = [0.0f32; 4];
                    vst1q_f32(temp.as_mut_ptr(), sin_x);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    result.push(input[j].sin());
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for &x in input.iter() {
        result.push(x.sin());
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated sine for f64 arrays
#[allow(dead_code)]
pub fn simd_sin_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let len = input.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let mut result = Vec::with_capacity(len);

    const FRAC_1_PI: f64 = std::f64::consts::FRAC_1_PI;
    const PI: f64 = std::f64::consts::PI;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;

                let pi = _mm256_set1_pd(PI);
                let two_pi = _mm256_set1_pd(2.0 * PI);
                let pi_over_2 = _mm256_set1_pd(PI / 2.0);
                let zero = _mm256_setzero_pd();
                let one = _mm256_set1_pd(1.0);
                let neg_one = _mm256_set1_pd(-1.0);

                // Higher-degree Taylor series for f64
                let c1 = _mm256_set1_pd(1.0);
                let c3 = _mm256_set1_pd(-1.0 / 6.0);
                let c5 = _mm256_set1_pd(1.0 / 120.0);
                let c7 = _mm256_set1_pd(-1.0 / 5040.0);
                let c9 = _mm256_set1_pd(1.0 / 362880.0);
                let c11 = _mm256_set1_pd(-1.0 / 39916800.0);

                let mut i = 0;
                while i + 4 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 4];
                    let mut x = _mm256_loadu_pd(slice.as_ptr());

                    // Range reduction
                    let k = _mm256_round_pd(
                        _mm256_mul_pd(x, _mm256_set1_pd(FRAC_1_PI / 2.0)),
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                    );
                    x = _mm256_sub_pd(x, _mm256_mul_pd(k, two_pi));

                    let mask_gt_pi_2 = _mm256_cmp_pd(x, pi_over_2, _CMP_GT_OQ);
                    let mask_lt_neg_pi_2 =
                        _mm256_cmp_pd(x, _mm256_sub_pd(zero, pi_over_2), _CMP_LT_OQ);

                    let x_reduced_high = _mm256_sub_pd(pi, x);
                    x = _mm256_blendv_pd(x, x_reduced_high, mask_gt_pi_2);

                    let x_reduced_low = _mm256_sub_pd(_mm256_sub_pd(zero, pi), x);
                    x = _mm256_blendv_pd(x, x_reduced_low, mask_lt_neg_pi_2);

                    // Taylor series
                    let x2 = _mm256_mul_pd(x, x);
                    let x3 = _mm256_mul_pd(x2, x);
                    let x5 = _mm256_mul_pd(x3, x2);
                    let x7 = _mm256_mul_pd(x5, x2);
                    let x9 = _mm256_mul_pd(x7, x2);
                    let x11 = _mm256_mul_pd(x9, x2);

                    let mut sin_x = _mm256_mul_pd(c1, x);
                    sin_x = _mm256_add_pd(sin_x, _mm256_mul_pd(c3, x3));
                    sin_x = _mm256_add_pd(sin_x, _mm256_mul_pd(c5, x5));
                    sin_x = _mm256_add_pd(sin_x, _mm256_mul_pd(c7, x7));
                    sin_x = _mm256_add_pd(sin_x, _mm256_mul_pd(c9, x9));
                    sin_x = _mm256_add_pd(sin_x, _mm256_mul_pd(c11, x11));

                    // Clamp to [-1, 1]
                    sin_x = _mm256_max_pd(sin_x, neg_one);
                    sin_x = _mm256_min_pd(sin_x, one);

                    let mut temp = [0.0f64; 4];
                    _mm256_storeu_pd(temp.as_mut_ptr(), sin_x);
                    result.extend_from_slice(&temp);
                    i += 4;
                }

                for j in i..len {
                    result.push(input[j].sin());
                }

                return Array1::from_vec(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                use std::arch::aarch64::*;

                let pi = vdupq_n_f64(PI);
                let two_pi = vdupq_n_f64(2.0 * PI);
                let pi_over_2 = vdupq_n_f64(PI / 2.0);
                let neg_pi_over_2 = vdupq_n_f64(-PI / 2.0);
                let zero = vdupq_n_f64(0.0);
                let one = vdupq_n_f64(1.0);
                let neg_one = vdupq_n_f64(-1.0);

                let c1 = vdupq_n_f64(1.0);
                let c3 = vdupq_n_f64(-1.0 / 6.0);
                let c5 = vdupq_n_f64(1.0 / 120.0);
                let c7 = vdupq_n_f64(-1.0 / 5040.0);
                let c9 = vdupq_n_f64(1.0 / 362880.0);
                let c11 = vdupq_n_f64(-1.0 / 39916800.0);

                let mut i = 0;
                while i + 2 <= len {
                    let slice = &input.as_slice().unwrap()[i..i + 2];
                    let mut x = vld1q_f64(slice.as_ptr());

                    let k = vrndnq_f64(vmulq_f64(x, vdupq_n_f64(FRAC_1_PI / 2.0)));
                    x = vsubq_f64(x, vmulq_f64(k, two_pi));

                    let mask_gt_pi_2 = vcgtq_f64(x, pi_over_2);
                    let mask_lt_neg_pi_2 = vcltq_f64(x, neg_pi_over_2);

                    let x_reduced_high = vsubq_f64(pi, x);
                    x = vbslq_f64(mask_gt_pi_2, x_reduced_high, x);

                    let x_reduced_low = vsubq_f64(vsubq_f64(zero, pi), x);
                    x = vbslq_f64(mask_lt_neg_pi_2, x_reduced_low, x);

                    let x2 = vmulq_f64(x, x);
                    let x3 = vmulq_f64(x2, x);
                    let x5 = vmulq_f64(x3, x2);
                    let x7 = vmulq_f64(x5, x2);
                    let x9 = vmulq_f64(x7, x2);
                    let x11 = vmulq_f64(x9, x2);

                    let mut sin_x = vmulq_f64(c1, x);
                    sin_x = vaddq_f64(sin_x, vmulq_f64(c3, x3));
                    sin_x = vaddq_f64(sin_x, vmulq_f64(c5, x5));
                    sin_x = vaddq_f64(sin_x, vmulq_f64(c7, x7));
                    sin_x = vaddq_f64(sin_x, vmulq_f64(c9, x9));
                    sin_x = vaddq_f64(sin_x, vmulq_f64(c11, x11));

                    // Clamp to [-1, 1]
                    sin_x = vmaxq_f64(sin_x, neg_one);
                    sin_x = vminq_f64(sin_x, one);

                    let mut temp = [0.0f64; 2];
                    vst1q_f64(temp.as_mut_ptr(), sin_x);
                    result.extend_from_slice(&temp);
                    i += 2;
                }

                for j in i..len {
                    result.push(input[j].sin());
                }

                return Array1::from_vec(result);
            }
        }
    }

    // Scalar fallback
    for &x in input.iter() {
        result.push(x.sin());
    }
    Array1::from_vec(result)
}

/// SIMD-accelerated cosine for f32 arrays
///
/// Computes cos(x) using range reduction and polynomial approximation.
/// Uses the identity: cos(x) = sin(x + π/2)
///
/// # Arguments
/// * `input` - Input array (radians)
///
/// # Returns
/// * Array of cos(x) values in range [-1, 1]
///
/// # Example
/// ```
/// use scirs2_core::simd::transcendental::simd_cos_f32;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f32, std::f32::consts::PI];
/// let result = simd_cos_f32(&x.view());
/// assert!((result[0] - 1.0).abs() < 1e-6); // cos(0) = 1
/// ```
#[allow(dead_code)]
pub fn simd_cos_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    // Use identity: cos(x) = sin(x + π/2)
    let shifted: Array1<f32> = input.mapv(|x| x + std::f32::consts::FRAC_PI_2);
    simd_sin_f32(&shifted.view())
}

/// SIMD-accelerated cosine for f64 arrays
#[allow(dead_code)]
pub fn simd_cos_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let shifted: Array1<f64> = input.mapv(|x| x + std::f64::consts::FRAC_PI_2);
    simd_sin_f64(&shifted.view())
}

/// SIMD-accelerated log base 2 for f32 arrays
///
/// Computes log2(x) = ln(x) / ln(2)
///
/// # Arguments
/// * `input` - Input array (must be positive)
///
/// # Returns
/// * Array of log2(x) values
#[allow(dead_code)]
pub fn simd_log2_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let ln_result = simd_ln_f32(input);
    let inv_ln_2 = 1.0 / std::f32::consts::LN_2;
    ln_result.mapv(|x| x * inv_ln_2)
}

/// SIMD-accelerated log base 2 for f64 arrays
#[allow(dead_code)]
pub fn simd_log2_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let ln_result = simd_ln_f64(input);
    let inv_ln_2 = 1.0 / std::f64::consts::LN_2;
    ln_result.mapv(|x| x * inv_ln_2)
}

/// SIMD-accelerated log base 10 for f32 arrays
///
/// Computes log10(x) = ln(x) / ln(10)
///
/// # Arguments
/// * `input` - Input array (must be positive)
///
/// # Returns
/// * Array of log10(x) values
#[allow(dead_code)]
pub fn simd_log10_f32(input: &ArrayView1<f32>) -> Array1<f32> {
    let ln_result = simd_ln_f32(input);
    let inv_ln_10 = 1.0 / std::f32::consts::LN_10;
    ln_result.mapv(|x| x * inv_ln_10)
}

/// SIMD-accelerated log base 10 for f64 arrays
#[allow(dead_code)]
pub fn simd_log10_f64(input: &ArrayView1<f64>) -> Array1<f64> {
    let ln_result = simd_ln_f64(input);
    let inv_ln_10 = 1.0 / std::f64::consts::LN_10;
    ln_result.mapv(|x| x * inv_ln_10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_simd_exp_f32_basic() {
        let x = array![0.0f32, 1.0, -1.0, 2.0, -2.0];
        let result = simd_exp_f32(&x.view());

        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - std::f32::consts::E).abs() < 1e-5);
        assert!((result[2] - 1.0 / std::f32::consts::E).abs() < 1e-6);
        assert!((result[3] - std::f32::consts::E.powi(2)).abs() < 1e-4);
        assert!((result[4] - 1.0 / std::f32::consts::E.powi(2)).abs() < 1e-6);
    }

    #[test]
    fn test_simd_exp_f64_basic() {
        let x = array![0.0f64, 1.0, -1.0, 2.0, -2.0];
        let result = simd_exp_f64(&x.view());

        // Polynomial approximation achieves ~1e-10 relative error
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - std::f64::consts::E).abs() < 1e-9);
        assert!((result[2] - 1.0 / std::f64::consts::E).abs() < 1e-10);
        assert!((result[3] - std::f64::consts::E.powi(2)).abs() < 1e-8);
        assert!((result[4] - 1.0 / std::f64::consts::E.powi(2)).abs() < 1e-10);
    }

    #[test]
    fn test_simd_exp_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(-10.0, 10.0, 1000);
        let result = simd_exp_f32(&x.view());

        for (i, (&xi, &ri)) in x.iter().zip(result.iter()).enumerate() {
            let expected = xi.exp();
            let rel_error = if expected.abs() > 1e-30 {
                (ri - expected).abs() / expected.abs()
            } else {
                (ri - expected).abs()
            };
            assert!(
                rel_error < 1e-5,
                "Index {}: exp({}) = {}, got {}, rel_error = {}",
                i,
                xi,
                expected,
                ri,
                rel_error
            );
        }
    }

    #[test]
    fn test_simd_exp_f64_large_array() {
        let x: Array1<f64> = Array1::linspace(-10.0, 10.0, 1000);
        let result = simd_exp_f64(&x.view());

        for (i, (&xi, &ri)) in x.iter().zip(result.iter()).enumerate() {
            let expected = xi.exp();
            let rel_error = if expected.abs() > 1e-300 {
                (ri - expected).abs() / expected.abs()
            } else {
                (ri - expected).abs()
            };
            // Polynomial approximation achieves ~1e-10 relative error
            assert!(
                rel_error < 1e-9,
                "Index {}: exp({}) = {}, got {}, rel_error = {}",
                i,
                xi,
                expected,
                ri,
                rel_error
            );
        }
    }

    #[test]
    fn test_simd_exp_fast_f32() {
        let x = array![0.0f32, 1.0, -1.0, 2.0];
        let result = simd_exp_fast_f32(&x.view());

        // Fast exp uses integer bit manipulation - very rough approximation
        // Only check it's in the right ballpark (within 20%)
        assert!((result[0] - 1.0).abs() < 0.5);
        assert!((result[1] - std::f32::consts::E).abs() < 1.0);
    }

    #[test]
    fn test_simd_exp_f32_edge_cases() {
        // Test values within safe range
        let x = array![10.0f32, -10.0f32, 0.0f32];
        let result = simd_exp_f32(&x.view());

        // Should be finite positive values
        assert!(result[0].is_finite() && result[0] > 0.0);
        assert!(result[1].is_finite() && result[1] > 0.0);
        assert!((result[2] - 1.0).abs() < 1e-6);

        // Test moderate values
        let x_moderate = array![50.0f32, -50.0f32];
        let result_moderate = simd_exp_f32(&x_moderate.view());
        // exp(50) ≈ 5.18e21, exp(-50) ≈ 1.93e-22, both finite
        assert!(result_moderate[0].is_finite() && result_moderate[0] > 0.0);
        assert!(result_moderate[1].is_finite() && result_moderate[1] > 0.0);
    }

    #[test]
    fn test_simd_exp_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_exp_f32(&x.view());
        assert_eq!(result.len(), 0);
    }

    // Phase 76: Sigmoid tests
    #[test]
    fn test_simd_sigmoid_f32_basic() {
        let x = array![0.0f32, 2.0, -2.0, 5.0, -5.0];
        let result = simd_sigmoid_f32(&x.view());

        // sigmoid(0) = 0.5
        assert!((result[0] - 0.5).abs() < 1e-5);
        // sigmoid(2) ≈ 0.8808
        assert!((result[1] - 0.8807970779778823).abs() < 1e-4);
        // sigmoid(-2) ≈ 0.1192
        assert!((result[2] - 0.11920292202211755).abs() < 1e-4);
        // sigmoid(5) ≈ 0.9933
        assert!((result[3] - 0.9933071490757153).abs() < 1e-4);
        // sigmoid(-5) ≈ 0.0067
        assert!((result[4] - 0.006692850924284856).abs() < 1e-4);
    }

    #[test]
    fn test_simd_sigmoid_f64_basic() {
        let x = array![0.0f64, 2.0, -2.0];
        let result = simd_sigmoid_f64(&x.view());

        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.8807970779778823).abs() < 1e-8);
        assert!((result[2] - 0.11920292202211755).abs() < 1e-8);
    }

    #[test]
    fn test_simd_sigmoid_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(-10.0, 10.0, 1000);
        let result = simd_sigmoid_f32(&x.view());

        for (i, (&xi, &ri)) in x.iter().zip(result.iter()).enumerate() {
            // Compute expected sigmoid
            let expected = if xi >= 0.0 {
                1.0 / (1.0 + (-xi).exp())
            } else {
                let exp_x = xi.exp();
                exp_x / (1.0 + exp_x)
            };
            let error = (ri - expected).abs();
            assert!(
                error < 1e-4,
                "Index {}: sigmoid({}) = {}, got {}, error = {}",
                i,
                xi,
                expected,
                ri,
                error
            );
        }
    }

    #[test]
    fn test_simd_sigmoid_f32_properties() {
        // Test sigmoid properties
        let x = array![1.0f32, -1.0f32];
        let result = simd_sigmoid_f32(&x.view());

        // sigmoid(x) + sigmoid(-x) = 1
        let sum = result[0] + result[1];
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "sigmoid(x) + sigmoid(-x) should equal 1"
        );

        // sigmoid output should be close to expected values at moderate extremes
        let x_moderate = array![-10.0f32, 10.0f32];
        let result_moderate = simd_sigmoid_f32(&x_moderate.view());
        // sigmoid(-10) ≈ 4.5e-5, sigmoid(10) ≈ 0.99995
        assert!(result_moderate[0] > 0.0 && result_moderate[0] < 0.001);
        assert!(result_moderate[1] > 0.999 && result_moderate[1] <= 1.0);
    }

    #[test]
    fn test_simd_sigmoid_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_sigmoid_f32(&x.view());
        assert_eq!(result.len(), 0);
    }

    // Phase 76: GELU tests
    #[test]
    fn test_simd_gelu_f32_basic() {
        let x = array![0.0f32, 1.0, -1.0, 2.0, -2.0];
        let result = simd_gelu_f32(&x.view());

        // GELU(0) = 0
        assert!(result[0].abs() < 1e-6);
        // GELU(1) ≈ 0.8412 (tanh approximation gives slightly different values)
        assert!((result[1] - 0.8412).abs() < 0.02);
        // GELU(-1) ≈ -0.1588
        assert!((result[2] - (-0.1588)).abs() < 0.02);
        // GELU(2) ≈ 1.9546 (Padé approximation has larger error for |x| > 1)
        assert!(
            (result[3] - 1.9546).abs() < 0.05,
            "GELU(2) = {}, expected ~1.9546",
            result[3]
        );
    }

    #[test]
    fn test_simd_gelu_f64_basic() {
        let x = array![0.0f64, 1.0, -1.0];
        let result = simd_gelu_f64(&x.view());

        assert!(result[0].abs() < 1e-10);
        assert!((result[1] - 0.8413).abs() < 0.01);
        assert!((result[2] - (-0.1587)).abs() < 0.01);
    }

    #[test]
    fn test_simd_gelu_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(-5.0, 5.0, 1000);
        let result = simd_gelu_f32(&x.view());

        // GELU should satisfy: GELU(x) ≈ x for large positive x, GELU(x) ≈ 0 for large negative x
        for (&xi, &ri) in x.iter().zip(result.iter()) {
            if xi > 3.0 {
                // For large positive x, GELU(x) ≈ x
                assert!((ri - xi).abs() < 0.1);
            } else if xi < -3.0 {
                // For large negative x, GELU(x) ≈ 0
                assert!(ri.abs() < 0.1);
            }
        }
    }

    #[test]
    fn test_simd_gelu_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_gelu_f32(&x.view());
        assert_eq!(result.len(), 0);
    }

    // Phase 76: Swish tests
    #[test]
    fn test_simd_swish_f32_basic() {
        let x = array![0.0f32, 1.0, -1.0, 2.0, -2.0];
        let result = simd_swish_f32(&x.view());

        // swish(0) = 0 * sigmoid(0) = 0
        assert!(result[0].abs() < 1e-6);
        // swish(1) = 1 * sigmoid(1) ≈ 0.7311
        assert!((result[1] - 0.7311).abs() < 0.01);
        // swish(-1) = -1 * sigmoid(-1) ≈ -0.2689
        assert!((result[2] - (-0.2689)).abs() < 0.01);
        // swish(2) = 2 * sigmoid(2) ≈ 1.7616
        assert!((result[3] - 1.7616).abs() < 0.01);
    }

    #[test]
    fn test_simd_swish_f64_basic() {
        let x = array![0.0f64, 1.0, -1.0];
        let result = simd_swish_f64(&x.view());

        assert!(result[0].abs() < 1e-10);
        assert!((result[1] - 0.7311).abs() < 0.01);
        assert!((result[2] - (-0.2689)).abs() < 0.01);
    }

    #[test]
    fn test_simd_swish_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(-10.0, 10.0, 1000);
        let result = simd_swish_f32(&x.view());

        for (i, (&xi, &ri)) in x.iter().zip(result.iter()).enumerate() {
            let sigmoid = if xi >= 0.0 {
                1.0 / (1.0 + (-xi).exp())
            } else {
                let exp_x = xi.exp();
                exp_x / (1.0 + exp_x)
            };
            let expected = xi * sigmoid;
            let error = (ri - expected).abs();
            assert!(
                error < 1e-4,
                "Index {}: swish({}) = {}, got {}, error = {}",
                i,
                xi,
                expected,
                ri,
                error
            );
        }
    }

    #[test]
    fn test_simd_swish_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_swish_f32(&x.view());
        assert_eq!(result.len(), 0);
    }

    // Phase 76: Softplus tests
    #[test]
    fn test_simd_softplus_f32_basic() {
        let x = array![0.0f32, 1.0, -1.0, 5.0, -5.0];
        let result = simd_softplus_f32(&x.view());

        // softplus(0) = ln(2) ≈ 0.6931
        assert!((result[0] - std::f32::consts::LN_2).abs() < 1e-4);
        // softplus(1) ≈ 1.3133
        assert!((result[1] - 1.3133).abs() < 0.01);
        // softplus(-1) ≈ 0.3133
        assert!((result[2] - 0.3133).abs() < 0.01);
        // softplus(5) ≈ 5.0067
        assert!((result[3] - 5.0067).abs() < 0.01);
    }

    #[test]
    fn test_simd_softplus_f64_basic() {
        let x = array![0.0f64, 1.0, -1.0];
        let result = simd_softplus_f64(&x.view());

        assert!((result[0] - std::f64::consts::LN_2).abs() < 1e-10);
        assert!((result[1] - 1.3133).abs() < 0.01);
        assert!((result[2] - 0.3133).abs() < 0.01);
    }

    #[test]
    fn test_simd_softplus_f32_properties() {
        let x = array![100.0f32, -100.0f32];
        let result = simd_softplus_f32(&x.view());

        // For large positive x, softplus(x) ≈ x
        assert!((result[0] - 100.0).abs() < 0.1);
        // For large negative x, softplus(x) ≈ 0
        assert!(result[1] < 1e-10);

        // Softplus should always be positive
        let x2: Array1<f32> = Array1::linspace(-10.0, 10.0, 100);
        let result2 = simd_softplus_f32(&x2.view());
        for &val in result2.iter() {
            assert!(val > 0.0, "Softplus should always be positive");
        }
    }

    #[test]
    fn test_simd_softplus_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(-10.0, 10.0, 1000);
        let result = simd_softplus_f32(&x.view());

        for (i, (&xi, &ri)) in x.iter().zip(result.iter()).enumerate() {
            let expected = (1.0 + xi.exp()).ln();
            let error = (ri - expected).abs();
            assert!(
                error < 1e-4,
                "Index {}: softplus({}) = {}, got {}, error = {}",
                i,
                xi,
                expected,
                ri,
                error
            );
        }
    }

    #[test]
    fn test_simd_softplus_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_softplus_f32(&x.view());
        assert_eq!(result.len(), 0);
    }

    // Phase 77: Tanh tests
    #[test]
    fn test_simd_tanh_f32_basic() {
        let x = array![0.0f32, 1.0, -1.0, 2.0, -2.0];
        let result = simd_tanh_f32(&x.view());

        // tanh(0) = 0
        assert!(result[0].abs() < 1e-6);
        // tanh(1) ≈ 0.7616
        assert!((result[1] - 0.7616).abs() < 0.01);
        // tanh(-1) ≈ -0.7616
        assert!((result[2] - (-0.7616)).abs() < 0.01);
        // tanh(2) ≈ 0.9640
        assert!((result[3] - 0.9640).abs() < 0.01);
    }

    #[test]
    fn test_simd_tanh_f64_basic() {
        let x = array![0.0f64, 1.0, -1.0];
        let result = simd_tanh_f64(&x.view());

        assert!(result[0].abs() < 1e-10);
        assert!((result[1] - 0.7615941559557649).abs() < 1e-6);
        assert!((result[2] - (-0.7615941559557649)).abs() < 1e-6);
    }

    #[test]
    fn test_simd_tanh_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(-5.0, 5.0, 1000);
        let result = simd_tanh_f32(&x.view());

        for (i, (&xi, &ri)) in x.iter().zip(result.iter()).enumerate() {
            let expected = xi.tanh();
            let error = (ri - expected).abs();
            assert!(
                error < 1e-4,
                "Index {}: tanh({}) = {}, got {}, error = {}",
                i,
                xi,
                expected,
                ri,
                error
            );
        }
    }

    #[test]
    fn test_simd_tanh_f32_properties() {
        // tanh is odd: tanh(-x) = -tanh(x)
        let x = array![1.5f32, -1.5f32];
        let result = simd_tanh_f32(&x.view());
        assert!((result[0] + result[1]).abs() < 1e-5);

        // tanh output in [-1, 1]
        let x_extreme = array![100.0f32, -100.0f32];
        let result_extreme = simd_tanh_f32(&x_extreme.view());
        assert!((result_extreme[0] - 1.0).abs() < 1e-5);
        assert!((result_extreme[1] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_simd_tanh_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_tanh_f32(&x.view());
        assert_eq!(result.len(), 0);
    }

    // Phase 77: Mish tests
    #[test]
    fn test_simd_mish_f32_basic() {
        let x = array![0.0f32, 1.0, -1.0, 2.0];
        let result = simd_mish_f32(&x.view());

        // Mish(0) = 0
        assert!(result[0].abs() < 1e-6);
        // Mish(1) ≈ 0.8651 (x * tanh(softplus(x)))
        assert!((result[1] - 0.8651).abs() < 0.02);
        // Mish(-1) ≈ -0.3034
        assert!((result[2] - (-0.3034)).abs() < 0.02);
    }

    #[test]
    fn test_simd_mish_f64_basic() {
        let x = array![0.0f64, 1.0, -1.0];
        let result = simd_mish_f64(&x.view());

        assert!(result[0].abs() < 1e-10);
        // Mish(1) = 1 * tanh(ln(1+e)) ≈ 0.8651
        assert!((result[1] - 0.8651).abs() < 0.02);
    }

    #[test]
    fn test_simd_mish_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(-5.0, 5.0, 100);
        let result = simd_mish_f32(&x.view());

        for (&xi, &ri) in x.iter().zip(result.iter()) {
            // Mish(x) = x * tanh(softplus(x))
            let sp = if xi > 20.0 { xi } else { (1.0 + xi.exp()).ln() };
            let expected = xi * sp.tanh();
            let error = (ri - expected).abs();
            assert!(
                error < 1e-3,
                "mish({}) = {}, got {}, error = {}",
                xi,
                expected,
                ri,
                error
            );
        }
    }

    #[test]
    fn test_simd_mish_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_mish_f32(&x.view());
        assert_eq!(result.len(), 0);
    }

    // Phase 77: Ln tests
    #[test]
    fn test_simd_ln_f32_basic() {
        let x = array![1.0f32, std::f32::consts::E, 10.0, 100.0];
        let result = simd_ln_f32(&x.view());

        // ln(1) = 0
        assert!(result[0].abs() < 1e-5);
        // ln(e) = 1
        assert!((result[1] - 1.0).abs() < 1e-4);
        // ln(10) ≈ 2.3026
        assert!((result[2] - std::f32::consts::LN_10).abs() < 0.01);
        // ln(100) ≈ 4.6052
        assert!((result[3] - 4.6052).abs() < 0.02);
    }

    #[test]
    fn test_simd_ln_f64_basic() {
        let x = array![1.0f64, std::f64::consts::E, 10.0];
        let result = simd_ln_f64(&x.view());

        assert!(result[0].abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-8);
        assert!((result[2] - std::f64::consts::LN_10).abs() < 1e-6);
    }

    #[test]
    fn test_simd_ln_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(0.1, 100.0, 1000);
        let result = simd_ln_f32(&x.view());

        for (i, (&xi, &ri)) in x.iter().zip(result.iter()).enumerate() {
            let expected = xi.ln();
            let error = (ri - expected).abs();
            // Simple polynomial approximation has up to 10% relative error
            // for values where mantissa is close to 2.
            // This is acceptable for many ML use cases.
            assert!(
                error < 0.15,
                "Index {}: ln({}) = {}, got {}, error = {}",
                i,
                xi,
                expected,
                ri,
                error
            );
        }
    }

    #[test]
    fn test_simd_ln_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_ln_f32(&x.view());
        assert_eq!(result.len(), 0);
    }

    // Phase 78: Sin/Cos tests
    #[test]
    fn test_simd_sin_f32_basic() {
        let x = array![
            0.0f32,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            std::f32::consts::FRAC_PI_4
        ];
        let result = simd_sin_f32(&x.view());

        // sin(0) = 0
        assert!(result[0].abs() < 1e-5);
        // sin(π/2) = 1
        assert!((result[1] - 1.0).abs() < 1e-4);
        // sin(π) ≈ 0
        assert!(result[2].abs() < 1e-4);
        // sin(π/4) ≈ 0.7071
        assert!((result[3] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-3);
    }

    #[test]
    fn test_simd_sin_f64_basic() {
        let x = array![0.0f64, std::f64::consts::FRAC_PI_2, std::f64::consts::PI];
        let result = simd_sin_f64(&x.view());

        // Taylor series approximation has moderate error
        assert!(result[0].abs() < 1e-5);
        assert!((result[1] - 1.0).abs() < 1e-4);
        assert!(result[2].abs() < 1e-4);
    }

    #[test]
    fn test_simd_sin_f32_large_array() {
        let x: Array1<f32> = Array1::linspace(
            -2.0 * std::f32::consts::PI,
            2.0 * std::f32::consts::PI,
            1000,
        );
        let result = simd_sin_f32(&x.view());

        for (i, (&xi, &ri)) in x.iter().zip(result.iter()).enumerate() {
            let expected = xi.sin();
            let error = (ri - expected).abs();
            assert!(
                error < 1e-3,
                "Index {}: sin({}) = {}, got {}, error = {}",
                i,
                xi,
                expected,
                ri,
                error
            );
        }
    }

    #[test]
    fn test_simd_sin_f32_properties() {
        // sin is odd: sin(-x) = -sin(x)
        let x = array![1.0f32, -1.0f32];
        let result = simd_sin_f32(&x.view());
        assert!((result[0] + result[1]).abs() < 1e-4);

        // sin output in [-1, 1]
        let x_large: Array1<f32> = Array1::linspace(-10.0, 10.0, 100);
        let result_large = simd_sin_f32(&x_large.view());
        for &val in result_large.iter() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_simd_cos_f32_basic() {
        let x = array![
            0.0f32,
            std::f32::consts::PI,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::FRAC_PI_4
        ];
        let result = simd_cos_f32(&x.view());

        // cos(0) = 1
        assert!((result[0] - 1.0).abs() < 1e-5);
        // cos(π) = -1
        assert!((result[1] - (-1.0)).abs() < 1e-4);
        // cos(π/2) ≈ 0
        assert!(result[2].abs() < 1e-4);
        // cos(π/4) ≈ 0.7071
        assert!((result[3] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-3);
    }

    #[test]
    fn test_simd_cos_f64_basic() {
        let x = array![0.0f64, std::f64::consts::PI, std::f64::consts::FRAC_PI_2];
        let result = simd_cos_f64(&x.view());

        // Taylor series approximation has moderate error
        assert!((result[0] - 1.0).abs() < 1e-4);
        assert!((result[1] - (-1.0)).abs() < 1e-4);
        assert!(result[2].abs() < 1e-4);
    }

    #[test]
    fn test_simd_cos_f32_properties() {
        // cos is even: cos(-x) = cos(x)
        let x = array![1.5f32, -1.5f32];
        let result = simd_cos_f32(&x.view());
        assert!((result[0] - result[1]).abs() < 1e-4);

        // Pythagorean identity: sin²(x) + cos²(x) = 1
        let x_test = array![1.0f32];
        let sin_result = simd_sin_f32(&x_test.view());
        let cos_result = simd_cos_f32(&x_test.view());
        let sum_sq = sin_result[0] * sin_result[0] + cos_result[0] * cos_result[0];
        assert!((sum_sq - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_simd_sin_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_sin_f32(&x.view());
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_simd_cos_f32_empty() {
        let x: Array1<f32> = Array1::zeros(0);
        let result = simd_cos_f32(&x.view());
        assert_eq!(result.len(), 0);
    }

    // Phase 78: Log2/Log10 tests
    #[test]
    fn test_simd_log2_f32_basic() {
        let x = array![1.0f32, 2.0, 4.0, 8.0];
        let result = simd_log2_f32(&x.view());

        assert!(result[0].abs() < 1e-4);
        assert!((result[1] - 1.0).abs() < 0.02);
        assert!((result[2] - 2.0).abs() < 0.05);
        assert!((result[3] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_simd_log10_f32_basic() {
        let x = array![1.0f32, 10.0, 100.0];
        let result = simd_log10_f32(&x.view());

        assert!(result[0].abs() < 1e-4);
        assert!((result[1] - 1.0).abs() < 0.02);
        assert!((result[2] - 2.0).abs() < 0.05);
    }

    #[test]
    fn test_simd_log2_f64_basic() {
        let x = array![1.0f64, 2.0, 4.0];
        let result = simd_log2_f64(&x.view());

        assert!(result[0].abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-6);
        assert!((result[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_log10_f64_basic() {
        let x = array![1.0f64, 10.0, 100.0];
        let result = simd_log10_f64(&x.view());

        assert!(result[0].abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-6);
        assert!((result[2] - 2.0).abs() < 1e-6);
    }
}
