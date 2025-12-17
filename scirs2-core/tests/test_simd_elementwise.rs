//! Comprehensive tests for SIMD-accelerated element-wise operations
//!
//! Tests abs_simd, sign_simd, sqrt_simd, exp_simd, ln_simd, sin_simd, cos_simd, tan_simd,
//! sinh_simd, cosh_simd, tanh_simd, powf_simd, pow_simd, powi_simd, gamma_simd, floor_simd,
//! ceil_simd, round_simd, fract_simd, recip_simd, atan_simd, asin_simd, acos_simd, atan2_simd,
//! log10_simd, log2_simd, clamp_simd, and sign_simd functions with:
//! - Basic correctness (f32/f64)
//! - Edge cases (empty, zero, negative, NaN, infinity)
//! - Large arrays (SIMD path)
//! - SIMD vs scalar equivalence
//! - Numerical properties
//! - Inverse function properties (exp/ln)
//! - Trigonometric identities (sin/cos/tan)
//! - Hyperbolic identities (sinh/cosh/tanh)
//! - Power function properties (powf/pow)

use scirs2_core::ndarray::{array, Array1};
use scirs2_core::ndarray_ext::elementwise::{
    abs_simd, acos_simd, asin_simd, atan2_simd, atan_simd, ceil_simd, clamp_simd, cos_simd,
    cosh_simd, elu_simd, exp_simd, floor_simd, fract_simd, gamma_simd, gelu_simd, hardsigmoid_simd,
    hardswish_simd, leaky_relu_simd, ln_simd, log10_simd, log2_simd, mish_simd, pow_simd,
    powf_simd, powi_simd, prelu_simd, recip_simd, round_simd, selu_simd, sigmoid_simd, sign_simd,
    sin_simd, sinc_simd, sinh_simd, softplus_simd, sqrt_simd, swish_simd, tan_simd, tanh_simd,
};
#[cfg(feature = "random")]
use scirs2_core::random::{thread_rng, Distribution, Uniform};

// ============================================================================
// Abs SIMD Tests
// ============================================================================

#[test]
fn test_abs_simd_f64_basic() {
    let x = array![-3.0, -1.5, 0.0, 1.5, 3.0];
    let result = abs_simd(&x.view());

    assert_eq!(result[0], 3.0);
    assert_eq!(result[1], 1.5);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 1.5);
    assert_eq!(result[4], 3.0);
}

#[test]
fn test_abs_simd_f32_basic() {
    let x = array![-3.0f32, -1.5, 0.0, 1.5, 3.0];
    let result = abs_simd(&x.view());

    assert_eq!(result[0], 3.0);
    assert_eq!(result[1], 1.5);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 1.5);
    assert_eq!(result[4], 3.0);
}

#[test]
fn test_abs_simd_f64_empty() {
    let x: Array1<f64> = array![];
    let result = abs_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_abs_simd_f32_empty() {
    let x: Array1<f32> = array![];
    let result = abs_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_abs_simd_f64_all_positive() {
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = abs_simd(&x.view());

    // All positive values should be unchanged
    for i in 0..x.len() {
        assert_eq!(result[i], x[i]);
    }
}

#[test]
fn test_abs_simd_f32_all_positive() {
    let x = array![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let result = abs_simd(&x.view());

    for i in 0..x.len() {
        assert_eq!(result[i], x[i]);
    }
}

#[test]
fn test_abs_simd_f64_all_negative() {
    let x = array![-1.0, -2.0, -3.0, -4.0, -5.0];
    let result = abs_simd(&x.view());

    // All negative values should become positive
    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], 3.0);
    assert_eq!(result[3], 4.0);
    assert_eq!(result[4], 5.0);
}

#[test]
fn test_abs_simd_f32_all_negative() {
    let x = array![-1.0f32, -2.0, -3.0, -4.0, -5.0];
    let result = abs_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], 3.0);
    assert_eq!(result[3], 4.0);
    assert_eq!(result[4], 5.0);
}

#[test]
fn test_abs_simd_f64_zero() {
    let x = array![0.0, -0.0, 0.0];
    let result = abs_simd(&x.view());

    // Both +0 and -0 should become +0
    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_abs_simd_f32_zero() {
    let x = array![0.0f32, -0.0, 0.0];
    let result = abs_simd(&x.view());

    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_abs_simd_f64_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY];
    let result = abs_simd(&x.view());

    assert_eq!(result[0], f64::INFINITY);
    assert_eq!(result[1], f64::INFINITY); // -∞ becomes +∞
}

#[test]
fn test_abs_simd_f32_infinity() {
    let x = array![f32::INFINITY, f32::NEG_INFINITY];
    let result = abs_simd(&x.view());

    assert_eq!(result[0], f32::INFINITY);
    assert_eq!(result[1], f32::INFINITY);
}

#[test]
fn test_abs_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = abs_simd(&x.view());

    // abs(NaN) should be NaN
    assert!(result[0].is_nan());
}

#[test]
fn test_abs_simd_f32_nan() {
    let x = array![f32::NAN];
    let result = abs_simd(&x.view());

    assert!(result[0].is_nan());
}

#[test]
fn test_abs_simd_f64_large_array() {
    // Test SIMD path with large array
    let data: Vec<f64> = (0..10000)
        .map(|i| ((i as f64) * 0.1).sin() * 100.0)
        .collect();
    let x = Array1::from_vec(data.clone());
    let result = abs_simd(&x.view());

    // Verify all values are non-negative
    for &val in result.iter() {
        assert!(val >= 0.0);
    }

    // Verify correctness
    for i in 0..data.len() {
        assert_eq!(result[i], data[i].abs());
    }
}

#[test]
fn test_abs_simd_f32_large_array() {
    let data: Vec<f32> = (0..10000)
        .map(|i| ((i as f32) * 0.1).sin() * 100.0)
        .collect();
    let x = Array1::from_vec(data.clone());
    let result = abs_simd(&x.view());

    for &val in result.iter() {
        assert!(val >= 0.0);
    }

    for i in 0..data.len() {
        assert_eq!(result[i], data[i].abs());
    }
}

#[cfg(feature = "random")]
#[test]
fn test_abs_simd_equivalence_random() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).unwrap();
    let data: Vec<f64> = (0..5000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from_vec(data.clone());

    // SIMD result
    let simd_result = abs_simd(&x.view());

    // Scalar result
    let expected: Vec<f64> = data.iter().map(|&x| x.abs()).collect();

    // Compare
    for i in 0..data.len() {
        let diff = (simd_result[i] - expected[i]).abs();
        assert!(
            diff < 1e-10,
            "Mismatch at index {}: SIMD={}, expected={}",
            i,
            simd_result[i],
            expected[i]
        );
    }
}

// ============================================================================
// Sqrt SIMD Tests
// ============================================================================

#[test]
fn test_sqrt_simd_f64_basic() {
    let x = array![1.0, 4.0, 9.0, 16.0, 25.0];
    let result = sqrt_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], 3.0);
    assert_eq!(result[3], 4.0);
    assert_eq!(result[4], 5.0);
}

#[test]
fn test_sqrt_simd_f32_basic() {
    let x = array![1.0f32, 4.0, 9.0, 16.0, 25.0];
    let result = sqrt_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], 3.0);
    assert_eq!(result[3], 4.0);
    assert_eq!(result[4], 5.0);
}

#[test]
fn test_sqrt_simd_f64_empty() {
    let x: Array1<f64> = array![];
    let result = sqrt_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_sqrt_simd_f32_empty() {
    let x: Array1<f32> = array![];
    let result = sqrt_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_sqrt_simd_f64_zero() {
    let x = array![0.0];
    let result = sqrt_simd(&x.view());

    assert_eq!(result[0], 0.0);
}

#[test]
fn test_sqrt_simd_f32_zero() {
    let x = array![0.0f32];
    let result = sqrt_simd(&x.view());

    assert_eq!(result[0], 0.0);
}

#[test]
fn test_sqrt_simd_f64_negative() {
    let x = array![-1.0];
    let result = sqrt_simd(&x.view());

    // sqrt of negative should be NaN
    let val: f64 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_sqrt_simd_f32_negative() {
    let x = array![-1.0f32];
    let result = sqrt_simd(&x.view());

    let val: f32 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_sqrt_simd_f64_infinity() {
    let x = array![f64::INFINITY];
    let result = sqrt_simd(&x.view());

    assert_eq!(result[0], f64::INFINITY);
}

#[test]
fn test_sqrt_simd_f32_infinity() {
    let x = array![f32::INFINITY];
    let result = sqrt_simd(&x.view());

    assert_eq!(result[0], f32::INFINITY);
}

#[test]
fn test_sqrt_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = sqrt_simd(&x.view());

    // sqrt(NaN) should be NaN
    let val: f64 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_sqrt_simd_f32_nan() {
    let x = array![f32::NAN];
    let result = sqrt_simd(&x.view());

    assert!(result[0].is_nan());
}

#[test]
fn test_sqrt_simd_f64_small_values() {
    let x = array![0.01, 0.04, 0.09, 0.16, 0.25];
    let result = sqrt_simd(&x.view());

    let diff0: f64 = result[0] - 0.1;
    assert!(diff0.abs() < 1e-10);
    let diff1: f64 = result[1] - 0.2;
    assert!(diff1.abs() < 1e-10);
    let diff2: f64 = result[2] - 0.3;
    assert!(diff2.abs() < 1e-10);
    let diff3: f64 = result[3] - 0.4;
    assert!(diff3.abs() < 1e-10);
    let diff4: f64 = result[4] - 0.5;
    assert!(diff4.abs() < 1e-10);
}

#[test]
fn test_sqrt_simd_f32_small_values() {
    let x = array![0.01f32, 0.04, 0.09, 0.16, 0.25];
    let result = sqrt_simd(&x.view());

    assert!((result[0] - 0.1).abs() < 1e-6);
    assert!((result[1] - 0.2).abs() < 1e-6);
    assert!((result[2] - 0.3).abs() < 1e-6);
    assert!((result[3] - 0.4).abs() < 1e-6);
    assert!((result[4] - 0.5).abs() < 1e-6);
}

#[test]
fn test_sqrt_simd_f64_large_array() {
    // Test SIMD path with large array
    let data: Vec<f64> = (0..10000).map(|i| i as f64).collect();
    let x = Array1::from_vec(data.clone());
    let result = sqrt_simd(&x.view());

    // Verify correctness
    for i in 0..data.len() {
        let expected = data[i].sqrt();
        let diff = (result[i] - expected).abs();
        assert!(
            diff < 1e-10,
            "Mismatch at index {}: SIMD={}, expected={}",
            i,
            result[i],
            expected
        );
    }
}

#[test]
fn test_sqrt_simd_f32_large_array() {
    let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();
    let x = Array1::from_vec(data.clone());
    let result = sqrt_simd(&x.view());

    for i in 0..data.len() {
        let expected = data[i].sqrt();
        let diff = (result[i] - expected).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: SIMD={}, expected={}",
            i,
            result[i],
            expected
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_sqrt_simd_equivalence_random() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(0.0, 100.0).unwrap(); // Only positive values
    let data: Vec<f64> = (0..5000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from_vec(data.clone());

    // SIMD result
    let simd_result = sqrt_simd(&x.view());

    // Scalar result
    let expected: Vec<f64> = data.iter().map(|&x| x.sqrt()).collect();

    // Compare
    for i in 0..data.len() {
        let diff = (simd_result[i] - expected[i]).abs();
        assert!(
            diff < 1e-10,
            "Mismatch at index {}: SIMD={}, expected={}",
            i,
            simd_result[i],
            expected[i]
        );
    }
}

// ============================================================================
// Property Tests
// ============================================================================

#[test]
fn test_abs_simd_idempotent() {
    // abs(abs(x)) = abs(x)
    let x = array![-3.0, -1.0, 0.0, 1.0, 3.0];
    let abs_once = abs_simd(&x.view());
    let abs_twice = abs_simd(&abs_once.view());

    for i in 0..x.len() {
        assert_eq!(abs_once[i], abs_twice[i]);
    }
}

#[cfg(feature = "random")]
#[test]
fn test_abs_simd_non_negative() {
    // abs(x) >= 0 for all x
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).unwrap();
    let data: Vec<f64> = (0..1000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from_vec(data);
    let result = abs_simd(&x.view());

    for &val in result.iter() {
        assert!(val >= 0.0 || val.is_nan());
    }
}

#[test]
fn test_sqrt_abs_non_negative() {
    // sqrt(abs(x)) is always defined
    let x = array![-9.0, -4.0, 0.0, 4.0, 9.0];
    let abs_x = abs_simd(&x.view());
    let sqrt_abs_x = sqrt_simd(&abs_x.view());

    // All values should be non-NaN
    for &val in sqrt_abs_x.iter() {
        let v: f64 = val;
        assert!(!v.is_nan());
    }

    // Check specific values
    assert_eq!(sqrt_abs_x[0], 3.0); // sqrt(abs(-9)) = sqrt(9) = 3
    assert_eq!(sqrt_abs_x[1], 2.0); // sqrt(abs(-4)) = sqrt(4) = 2
    assert_eq!(sqrt_abs_x[2], 0.0); // sqrt(abs(0)) = sqrt(0) = 0
    assert_eq!(sqrt_abs_x[3], 2.0); // sqrt(abs(4)) = sqrt(4) = 2
    assert_eq!(sqrt_abs_x[4], 3.0); // sqrt(abs(9)) = sqrt(9) = 3
}

// ============================================================================
// Exp SIMD Tests
// ============================================================================

#[test]
fn test_exp_simd_f64_basic() {
    let x = array![0.0, 1.0, 2.0];
    let result = exp_simd(&x.view());

    let diff0: f64 = result[0] - 1.0;
    assert!(diff0.abs() < 1e-10);
    let diff1: f64 = result[1] - std::f64::consts::E;
    assert!(diff1.abs() < 1e-9);
    let diff2: f64 = result[2] - std::f64::consts::E.powi(2);
    assert!(diff2.abs() < 1e-9);
}

#[test]
fn test_exp_simd_f32_basic() {
    let x = array![0.0f32, 1.0, 2.0];
    let result = exp_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - std::f32::consts::E).abs() < 1e-6);
    assert!((result[2] - std::f32::consts::E.powi(2)).abs() < 1e-6);
}

#[test]
fn test_exp_simd_f64_empty() {
    let x: Array1<f64> = array![];
    let result = exp_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_exp_simd_f32_empty() {
    let x: Array1<f32> = array![];
    let result = exp_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_exp_simd_f64_zero() {
    let x = array![0.0];
    let result = exp_simd(&x.view());

    let diff: f64 = result[0] - 1.0;
    assert!(diff.abs() < 1e-10);
}

#[test]
fn test_exp_simd_f32_zero() {
    let x = array![0.0f32];
    let result = exp_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_exp_simd_f64_negative() {
    let x = array![-1.0, -2.0, -5.0];
    let result = exp_simd(&x.view());

    let diff0: f64 = result[0] - 0.367879441;
    assert!(diff0.abs() < 1e-9);
    let diff1: f64 = result[1] - 0.135335283;
    assert!(diff1.abs() < 1e-9);
    let diff2: f64 = result[2] - 0.006737947;
    assert!(diff2.abs() < 1e-9);
}

#[test]
fn test_exp_simd_f32_negative() {
    let x = array![-1.0f32, -2.0, -5.0];
    let result = exp_simd(&x.view());

    assert!((result[0] - 0.367_879_45).abs() < 1e-6);
    assert!((result[1] - 0.135_335_28).abs() < 1e-6);
    assert!((result[2] - 0.006737947).abs() < 1e-6);
}

#[test]
fn test_exp_simd_f64_large_positive() {
    let x = array![10.0, 20.0];
    let result = exp_simd(&x.view());

    let diff: f64 = result[0] - 22026.465794806718;
    assert!(diff.abs() < 1e-9);
    assert!(result[1] > 1e8); // exp(20) is very large
}

#[test]
fn test_exp_simd_f32_large_positive() {
    let x = array![10.0f32];
    let result = exp_simd(&x.view());

    assert!((result[0] - 22_026.465).abs() < 1e-2);
}

#[test]
fn test_exp_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = exp_simd(&x.view());

    let val: f64 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_exp_simd_f32_nan() {
    let x = array![f32::NAN];
    let result = exp_simd(&x.view());

    let val: f32 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_exp_simd_f64_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY];
    let result = exp_simd(&x.view());

    assert_eq!(result[0], f64::INFINITY);
    assert_eq!(result[1], 0.0); // exp(-∞) = 0
}

#[test]
fn test_exp_simd_f32_infinity() {
    let x = array![f32::INFINITY, f32::NEG_INFINITY];
    let result = exp_simd(&x.view());

    assert_eq!(result[0], f32::INFINITY);
    assert_eq!(result[1], 0.0);
}

#[test]
fn test_exp_simd_f64_large_array() {
    let data: Vec<f64> = (0..10000).map(|i| ((i as f64) * 0.001) - 5.0).collect();
    let x = Array1::from_vec(data.clone());
    let result = exp_simd(&x.view());

    // Verify correctness
    for i in 0..data.len() {
        let expected = data[i].exp();
        let diff = (result[i] - expected).abs();
        let rel_error = if expected.abs() > 1e-10 {
            diff / expected.abs()
        } else {
            diff
        };
        assert!(
            rel_error < 1e-10,
            "Mismatch at index {}: SIMD={}, expected={}",
            i,
            result[i],
            expected
        );
    }
}

#[test]
fn test_exp_simd_f32_large_array() {
    let data: Vec<f32> = (0..10000).map(|i| ((i as f32) * 0.001) - 5.0).collect();
    let x = Array1::from_vec(data.clone());
    let result = exp_simd(&x.view());

    for i in 0..data.len() {
        let expected = data[i].exp();
        let diff = (result[i] - expected).abs();
        let rel_error = if expected.abs() > 1e-6 {
            diff / expected.abs()
        } else {
            diff
        };
        assert!(rel_error < 1e-5, "Mismatch at index {}", i);
    }
}

// ============================================================================
// Ln (Natural Logarithm) SIMD Tests
// ============================================================================

#[test]
fn test_ln_simd_f64_basic() {
    let x = array![1.0, std::f64::consts::E, std::f64::consts::E.powi(2)];
    let result = ln_simd(&x.view());

    let diff0: f64 = result[0] - 0.0;
    assert!(diff0.abs() < 1e-10);
    let diff1: f64 = result[1] - 1.0;
    assert!(diff1.abs() < 1e-9);
    let diff2: f64 = result[2] - 2.0;
    assert!(diff2.abs() < 1e-9);
}

#[test]
fn test_ln_simd_f32_basic() {
    let x = array![1.0f32, std::f32::consts::E, std::f32::consts::E.powi(2)];
    let result = ln_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!((result[2] - 2.0).abs() < 1e-6);
}

#[test]
fn test_ln_simd_f64_empty() {
    let x: Array1<f64> = array![];
    let result = ln_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_ln_simd_f32_empty() {
    let x: Array1<f32> = array![];
    let result = ln_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_ln_simd_f64_one() {
    let x = array![1.0];
    let result = ln_simd(&x.view());

    let diff: f64 = result[0] - 0.0;
    assert!(diff.abs() < 1e-10);
}

#[test]
fn test_ln_simd_f32_one() {
    let x = array![1.0f32];
    let result = ln_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_ln_simd_f64_zero() {
    let x = array![0.0];
    let result = ln_simd(&x.view());

    assert_eq!(result[0], f64::NEG_INFINITY);
}

#[test]
fn test_ln_simd_f32_zero() {
    let x = array![0.0f32];
    let result = ln_simd(&x.view());

    assert_eq!(result[0], f32::NEG_INFINITY);
}

#[test]
fn test_ln_simd_f64_negative() {
    let x = array![-1.0];
    let result = ln_simd(&x.view());

    let val: f64 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_ln_simd_f32_negative() {
    let x = array![-1.0f32];
    let result = ln_simd(&x.view());

    let val: f32 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_ln_simd_f64_infinity() {
    let x = array![f64::INFINITY];
    let result = ln_simd(&x.view());

    assert_eq!(result[0], f64::INFINITY);
}

#[test]
fn test_ln_simd_f32_infinity() {
    let x = array![f32::INFINITY];
    let result = ln_simd(&x.view());

    assert_eq!(result[0], f32::INFINITY);
}

#[test]
fn test_ln_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = ln_simd(&x.view());

    let val: f64 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_ln_simd_f32_nan() {
    let x = array![f32::NAN];
    let result = ln_simd(&x.view());

    let val: f32 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_ln_simd_f64_small_values() {
    let x = array![0.1, 0.01, 0.001];
    let result = ln_simd(&x.view());

    let diff0: f64 = result[0] - (-std::f64::consts::LN_10);
    assert!(diff0.abs() < 1e-9);
    let diff1: f64 = result[1] - (-2.0 * std::f64::consts::LN_10);
    assert!(diff1.abs() < 1e-9);
    let diff2: f64 = result[2] - (-3.0 * std::f64::consts::LN_10);
    assert!(diff2.abs() < 1e-9);
}

#[test]
fn test_ln_simd_f32_small_values() {
    let x = array![0.1f32, 0.01, 0.001];
    let result = ln_simd(&x.view());

    assert!((result[0] - (-std::f32::consts::LN_10)).abs() < 1e-6);
    assert!((result[1] - (-2.0 * std::f32::consts::LN_10)).abs() < 1e-6);
    assert!((result[2] - (-3.0 * std::f32::consts::LN_10)).abs() < 1e-6);
}

#[test]
fn test_ln_simd_f64_large_array() {
    let data: Vec<f64> = (1..=10000).map(|i| (i as f64) * 0.1).collect();
    let x = Array1::from_vec(data.clone());
    let result = ln_simd(&x.view());

    for i in 0..data.len() {
        let expected = data[i].ln();
        let diff = (result[i] - expected).abs();
        assert!(
            diff < 1e-10,
            "Mismatch at index {}: SIMD={}, expected={}",
            i,
            result[i],
            expected
        );
    }
}

#[test]
fn test_ln_simd_f32_large_array() {
    let data: Vec<f32> = (1..=10000).map(|i| (i as f32) * 0.1).collect();
    let x = Array1::from_vec(data.clone());
    let result = ln_simd(&x.view());

    for i in 0..data.len() {
        let expected = data[i].ln();
        let diff = (result[i] - expected).abs();
        assert!(diff < 1e-5, "Mismatch at index {}", i);
    }
}

// ============================================================================
// Combined Property Tests
// ============================================================================

#[test]
fn test_exp_ln_inverse() {
    // exp(ln(x)) = x for positive x
    let x = array![1.0, 2.0, 5.0, 10.0, 100.0];
    let ln_x = ln_simd(&x.view());
    let exp_ln_x = exp_simd(&ln_x.view());

    for i in 0..x.len() {
        let diff: f64 = exp_ln_x[i] - x[i];
        let abs_diff = diff.abs();
        let rel_error = abs_diff / x[i];
        assert!(rel_error < 1e-10, "exp(ln(x)) != x at index {}", i);
    }
}

#[test]
fn test_ln_exp_inverse() {
    // ln(exp(x)) = x
    let x = array![-5.0, -1.0, 0.0, 1.0, 5.0];
    let exp_x = exp_simd(&x.view());
    let ln_exp_x = ln_simd(&exp_x.view());

    for i in 0..x.len() {
        let diff: f64 = ln_exp_x[i] - x[i];
        let abs_diff = diff.abs();
        assert!(abs_diff < 1e-10, "ln(exp(x)) != x at index {}", i);
    }
}

// ============================================================================
// Sin SIMD Tests
// ============================================================================

#[test]
fn test_sin_simd_f64_basic() {
    use std::f64::consts::PI;
    let x = array![0.0, PI / 6.0, PI / 4.0, PI / 2.0, PI];
    let result = sin_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-10);
    assert!((result[1] - 0.5).abs() < 1e-10);
    assert!((result[2] - (2.0_f64.sqrt() / 2.0)).abs() < 1e-10);
    assert!((result[3] - 1.0).abs() < 1e-10);
    assert!((result[4] - 0.0).abs() < 1e-10);
}

#[test]
fn test_sin_simd_f32_basic() {
    use std::f32::consts::PI;
    let x = array![0.0f32, PI / 6.0, PI / 4.0, PI / 2.0, PI];
    let result = sin_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 0.5).abs() < 1e-6);
    assert!((result[2] - (2.0_f32.sqrt() / 2.0)).abs() < 1e-6);
    assert!((result[3] - 1.0).abs() < 1e-6);
    assert!((result[4] - 0.0).abs() < 1e-6);
}

#[test]
fn test_sin_simd_f64_empty() {
    let x: Array1<f64> = array![];
    let result = sin_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_sin_simd_f64_zero() {
    let x = array![0.0];
    let result = sin_simd(&x.view());
    assert_eq!(result[0], 0.0);
}

#[test]
fn test_sin_simd_f64_negative() {
    use std::f64::consts::PI;
    let x = array![-PI / 2.0, -PI / 6.0];
    let result = sin_simd(&x.view());

    assert!((result[0] - (-1.0)).abs() < 1e-10);
    assert!((result[1] - (-0.5)).abs() < 1e-10);
}

#[test]
fn test_sin_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = sin_simd(&x.view());
    assert!(result[0].is_nan());
}

#[test]
fn test_sin_simd_f64_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY];
    let result = sin_simd(&x.view());
    assert!(result[0].is_nan());
    assert!(result[1].is_nan());
}

#[test]
fn test_sin_simd_f64_large_array() {
    use std::f64::consts::PI;
    let x: Array1<f64> = Array1::linspace(0.0, 2.0 * PI, 10000);
    let result = sin_simd(&x.view());

    // Check periodicity
    assert!((result[0] - 0.0).abs() < 1e-10);
    assert!((result[result.len() - 1] - 0.0).abs() < 1e-9);
}

#[test]
fn test_sin_simd_f32_large_array() {
    use std::f32::consts::PI;
    let x: Array1<f32> = Array1::linspace(0.0, 2.0 * PI, 10000);
    let result = sin_simd(&x.view());

    // Check range: sin(x) ∈ [-1, 1]
    for &val in result.iter() {
        assert!((-1.0..=1.0).contains(&val));
    }
}

#[cfg(feature = "random")]
#[test]
fn test_sin_simd_f64_range() {
    // sin(x) should always be in [-1, 1]
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).unwrap();
    let x: Array1<f64> = Array1::from_vec((0..1000).map(|_| uniform.sample(&mut rng)).collect());
    let result = sin_simd(&x.view());

    for &val in result.iter() {
        assert!(
            (-1.0..=1.0).contains(&val),
            "sin value {} out of range",
            val
        );
    }
}

// ============================================================================
// Cos SIMD Tests
// ============================================================================

#[test]
fn test_cos_simd_f64_basic() {
    use std::f64::consts::PI;
    let x = array![0.0, PI / 3.0, PI / 2.0, PI];
    let result = cos_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-10);
    assert!((result[1] - 0.5).abs() < 1e-10);
    assert!((result[2] - 0.0).abs() < 1e-10);
    assert!((result[3] - (-1.0)).abs() < 1e-10);
}

#[test]
fn test_cos_simd_f32_basic() {
    use std::f32::consts::PI;
    let x = array![0.0f32, PI / 3.0, PI / 2.0, PI];
    let result = cos_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 0.5).abs() < 1e-6);
    assert!((result[2] - 0.0).abs() < 1e-6);
    assert!((result[3] - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_cos_simd_f64_empty() {
    let x: Array1<f64> = array![];
    let result = cos_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_cos_simd_f64_zero() {
    let x = array![0.0];
    let result = cos_simd(&x.view());
    assert_eq!(result[0], 1.0);
}

#[test]
fn test_cos_simd_f64_negative() {
    use std::f64::consts::PI;
    let x = array![-PI, -PI / 3.0];
    let result = cos_simd(&x.view());

    assert!((result[0] - (-1.0)).abs() < 1e-10);
    assert!((result[1] - 0.5).abs() < 1e-10);
}

#[test]
fn test_cos_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = cos_simd(&x.view());
    assert!(result[0].is_nan());
}

#[test]
fn test_cos_simd_f64_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY];
    let result = cos_simd(&x.view());
    assert!(result[0].is_nan());
    assert!(result[1].is_nan());
}

#[test]
fn test_cos_simd_f64_large_array() {
    use std::f64::consts::PI;
    let x: Array1<f64> = Array1::linspace(0.0, 2.0 * PI, 10000);
    let result = cos_simd(&x.view());

    // Check periodicity
    assert!((result[0] - 1.0).abs() < 1e-10);
    assert!((result[result.len() - 1] - 1.0).abs() < 1e-9);
}

#[test]
fn test_cos_simd_f32_large_array() {
    use std::f32::consts::PI;
    let x: Array1<f32> = Array1::linspace(0.0, 2.0 * PI, 10000);
    let result = cos_simd(&x.view());

    // Check range: cos(x) ∈ [-1, 1]
    for &val in result.iter() {
        assert!((-1.0..=1.0).contains(&val));
    }
}

#[cfg(feature = "random")]
#[test]
fn test_cos_simd_f64_range() {
    // cos(x) should always be in [-1, 1]
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).unwrap();
    let x: Array1<f64> = Array1::from_vec((0..1000).map(|_| uniform.sample(&mut rng)).collect());
    let result = cos_simd(&x.view());

    for &val in result.iter() {
        assert!(
            (-1.0..=1.0).contains(&val),
            "cos value {} out of range",
            val
        );
    }
}

// ============================================================================
// Tan SIMD Tests
// ============================================================================

#[test]
fn test_tan_simd_f64_basic() {
    use std::f64::consts::PI;
    let x = array![0.0, PI / 4.0, PI / 6.0];
    let result = tan_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-10);
    assert!((result[1] - 1.0).abs() < 1e-10);
    assert!((result[2] - (1.0 / 3.0_f64.sqrt())).abs() < 1e-10);
}

#[test]
fn test_tan_simd_f32_basic() {
    use std::f32::consts::PI;
    let x = array![0.0f32, PI / 4.0, PI / 6.0];
    let result = tan_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!((result[2] - (1.0 / 3.0_f32.sqrt())).abs() < 1e-6);
}

#[test]
fn test_tan_simd_f64_empty() {
    let x: Array1<f64> = array![];
    let result = tan_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_tan_simd_f64_zero() {
    let x = array![0.0];
    let result = tan_simd(&x.view());
    assert_eq!(result[0], 0.0);
}

#[test]
fn test_tan_simd_f64_negative() {
    use std::f64::consts::PI;
    let x = array![-PI / 4.0, -PI / 6.0];
    let result = tan_simd(&x.view());

    assert!((result[0] - (-1.0)).abs() < 1e-10);
    assert!((result[1] - (-1.0 / 3.0_f64.sqrt())).abs() < 1e-10);
}

#[test]
fn test_tan_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = tan_simd(&x.view());
    assert!(result[0].is_nan());
}

#[test]
fn test_tan_simd_f64_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY];
    let result = tan_simd(&x.view());
    assert!(result[0].is_nan());
    assert!(result[1].is_nan());
}

#[test]
fn test_tan_simd_f64_singularity() {
    use std::f64::consts::PI;
    // tan has singularities at π/2, 3π/2, etc.
    let x = array![PI / 2.0, 3.0 * PI / 2.0];
    let result = tan_simd(&x.view());

    // Should return very large values or infinity
    assert!(result[0].abs() > 1e10 || result[0].is_infinite());
    assert!(result[1].abs() > 1e10 || result[1].is_infinite());
}

#[test]
fn test_tan_simd_f64_large_array() {
    use std::f64::consts::PI;
    let x: Array1<f64> = Array1::linspace(-PI / 3.0, PI / 3.0, 10000);
    let result = tan_simd(&x.view());

    // Check monotonicity in (-π/2, π/2)
    for i in 1..result.len() {
        assert!(
            result[i] >= result[i - 1],
            "tan not monotonic at index {}",
            i
        );
    }
}

#[test]
fn test_tan_simd_f32_large_array() {
    use std::f32::consts::PI;
    let x: Array1<f32> = Array1::linspace(-PI / 3.0, PI / 3.0, 10000);
    let result = tan_simd(&x.view());

    // All values should be finite in this range
    for &val in result.iter() {
        assert!(val.is_finite(), "tan value {} not finite", val);
    }
}

// ============================================================================
// Trigonometric Identity Tests
// ============================================================================

#[cfg(feature = "random")]
#[test]
fn test_sin_cos_pythagorean_identity() {
    // sin²(x) + cos²(x) = 1
    let mut rng = thread_rng();
    let uniform = Uniform::new(-10.0, 10.0).unwrap();
    let x: Array1<f64> = Array1::from_vec((0..1000).map(|_| uniform.sample(&mut rng)).collect());

    let sin_x = sin_simd(&x.view());
    let cos_x = cos_simd(&x.view());

    for i in 0..x.len() {
        let sum = sin_x[i].powi(2) + cos_x[i].powi(2);
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Pythagorean identity failed at index {}: sin²={}, cos²={}, sum={}",
            i,
            sin_x[i].powi(2),
            cos_x[i].powi(2),
            sum
        );
    }
}

#[test]
fn test_tan_sin_cos_identity() {
    // tan(x) = sin(x) / cos(x)
    use std::f64::consts::PI;
    let x: Array1<f64> = Array1::linspace(-PI / 3.0, PI / 3.0, 100);

    let sin_x = sin_simd(&x.view());
    let cos_x = cos_simd(&x.view());
    let tan_x = tan_simd(&x.view());

    for i in 0..x.len() {
        if cos_x[i].abs() > 1e-10 {
            let expected = sin_x[i] / cos_x[i];
            assert!(
                (tan_x[i] - expected).abs() < 1e-10,
                "tan(x) != sin(x)/cos(x) at index {}: tan={}, sin/cos={}",
                i,
                tan_x[i],
                expected
            );
        }
    }
}

#[cfg(feature = "random")]
#[test]
fn test_sin_negative_angle() {
    // sin(-x) = -sin(x)
    let mut rng = thread_rng();
    let uniform = Uniform::new(-10.0, 10.0).unwrap();
    let x: Array1<f64> = Array1::from_vec((0..100).map(|_| uniform.sample(&mut rng)).collect());

    let sin_x = sin_simd(&x.view());
    let neg_x = x.mapv(|v| -v);
    let sin_neg_x = sin_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!(
            (sin_neg_x[i] - (-sin_x[i])).abs() < 1e-10,
            "sin(-x) != -sin(x) at index {}",
            i
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_cos_negative_angle() {
    // cos(-x) = cos(x)
    let mut rng = thread_rng();
    let uniform = Uniform::new(-10.0, 10.0).unwrap();
    let x: Array1<f64> = Array1::from_vec((0..100).map(|_| uniform.sample(&mut rng)).collect());

    let cos_x = cos_simd(&x.view());
    let neg_x = x.mapv(|v| -v);
    let cos_neg_x = cos_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!(
            (cos_neg_x[i] - cos_x[i]).abs() < 1e-10,
            "cos(-x) != cos(x) at index {}",
            i
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_tan_negative_angle() {
    // tan(-x) = -tan(x)
    let mut rng = thread_rng();
    let uniform = Uniform::new(-1.0, 1.0).unwrap(); // Avoid singularities
    let x: Array1<f64> = Array1::from_vec((0..100).map(|_| uniform.sample(&mut rng)).collect());

    let tan_x = tan_simd(&x.view());
    let neg_x = x.mapv(|v| -v);
    let tan_neg_x = tan_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!(
            (tan_neg_x[i] - (-tan_x[i])).abs() < 1e-10,
            "tan(-x) != -tan(x) at index {}",
            i
        );
    }
}

// ============================================================================
// Powf SIMD Tests (scalar exponent)
// ============================================================================

#[test]
fn test_powf_simd_f64_basic() {
    let base = array![2.0, 3.0, 4.0, 5.0];
    let result = powf_simd(&base.view(), 2.0);

    assert_eq!(result[0], 4.0);
    assert_eq!(result[1], 9.0);
    assert_eq!(result[2], 16.0);
    assert_eq!(result[3], 25.0);
}

#[test]
fn test_powf_simd_f32_basic() {
    let base = array![2.0f32, 3.0, 4.0, 5.0];
    let result = powf_simd(&base.view(), 2.0);

    assert_eq!(result[0], 4.0);
    assert_eq!(result[1], 9.0);
    assert_eq!(result[2], 16.0);
    assert_eq!(result[3], 25.0);
}

#[test]
fn test_powf_simd_f64_empty() {
    let base: Array1<f64> = array![];
    let result = powf_simd(&base.view(), 2.0);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_powf_simd_f64_exp_zero() {
    // x^0 = 1 for any x
    let base = array![2.0, 3.0, 0.0, -5.0];
    let result = powf_simd(&base.view(), 0.0);

    for &val in result.iter() {
        assert_eq!(val, 1.0);
    }
}

#[test]
fn test_powf_simd_f64_exp_one() {
    // x^1 = x
    let base = array![2.0, 3.0, 4.0, 5.0];
    let result = powf_simd(&base.view(), 1.0);

    for i in 0..base.len() {
        assert_eq!(result[i], base[i]);
    }
}

#[test]
fn test_powf_simd_f64_negative_base() {
    let base = array![-2.0, -3.0];
    let result = powf_simd(&base.view(), 3.0); // Odd integer exponent

    assert_eq!(result[0], -8.0);
    assert_eq!(result[1], -27.0);
}

#[test]
fn test_powf_simd_f64_negative_base_non_integer() {
    let base = array![-2.0];
    let result = powf_simd(&base.view(), 2.5); // Non-integer exponent

    // Negative base with non-integer exponent should return NaN
    let val: f64 = result[0];
    assert!(val.is_nan());
}

#[test]
fn test_powf_simd_f64_zero_base_negative_exp() {
    let base = array![0.0];
    let result = powf_simd(&base.view(), -1.0);

    // 0^negative = infinity
    let val: f64 = result[0];
    assert!(val.is_infinite() && val.is_sign_positive());
}

#[test]
fn test_powf_simd_f64_nan() {
    let base = array![f64::NAN];
    let result = powf_simd(&base.view(), 2.0);
    assert!(result[0].is_nan());
}

#[test]
fn test_powf_simd_f64_large_array() {
    let base: Array1<f64> = Array1::linspace(1.0, 100.0, 10000);
    let result = powf_simd(&base.view(), 2.0);

    // Check some values
    assert!((result[0] - 1.0).abs() < 1e-10);
    assert!((result[result.len() - 1] - 10000.0).abs() < 1e-8);
}

#[test]
fn test_powf_simd_f32_large_array() {
    let base: Array1<f32> = Array1::linspace(1.0, 100.0, 10000);
    let result = powf_simd(&base.view(), 2.0);

    // All values should be finite and positive
    for &val in result.iter() {
        assert!(val.is_finite() && val >= 0.0);
    }
}

// ============================================================================
// Pow SIMD Tests (array exponent)
// ============================================================================

#[test]
fn test_pow_simd_f64_basic() {
    let base = array![2.0, 3.0, 4.0, 5.0];
    let exp = array![2.0, 3.0, 2.0, 1.0];
    let result = pow_simd(&base.view(), &exp.view());

    assert_eq!(result[0], 4.0);
    assert_eq!(result[1], 27.0);
    assert_eq!(result[2], 16.0);
    assert_eq!(result[3], 5.0);
}

#[test]
fn test_pow_simd_f32_basic() {
    let base = array![2.0f32, 3.0, 4.0, 5.0];
    let exp = array![2.0f32, 3.0, 2.0, 1.0];
    let result = pow_simd(&base.view(), &exp.view());

    assert_eq!(result[0], 4.0);
    assert_eq!(result[1], 27.0);
    assert_eq!(result[2], 16.0);
    assert_eq!(result[3], 5.0);
}

#[test]
fn test_pow_simd_f64_empty() {
    let base: Array1<f64> = array![];
    let exp: Array1<f64> = array![];
    let result = pow_simd(&base.view(), &exp.view());
    assert_eq!(result.len(), 0);
}

#[test]
#[should_panic(expected = "Base and exponent arrays must have the same length")]
fn test_pow_simd_f64_mismatched_lengths() {
    let base = array![2.0, 3.0, 4.0];
    let exp = array![2.0, 3.0];
    let _ = pow_simd(&base.view(), &exp.view());
}

#[test]
fn test_pow_simd_f64_exp_zero() {
    // x^0 = 1 for any x
    let base = array![2.0, 3.0, 0.0, -5.0];
    let exp = array![0.0, 0.0, 0.0, 0.0];
    let result = pow_simd(&base.view(), &exp.view());

    for &val in result.iter() {
        assert_eq!(val, 1.0);
    }
}

#[test]
fn test_pow_simd_f64_exp_one() {
    // x^1 = x
    let base = array![2.0, 3.0, 4.0, 5.0];
    let exp = array![1.0, 1.0, 1.0, 1.0];
    let result = pow_simd(&base.view(), &exp.view());

    for i in 0..base.len() {
        assert_eq!(result[i], base[i]);
    }
}

#[test]
fn test_pow_simd_f64_mixed_exponents() {
    let base = array![2.0, 3.0, 4.0];
    let exp = array![0.0, 1.0, 2.0];
    let result = pow_simd(&base.view(), &exp.view());

    assert_eq!(result[0], 1.0); // 2^0 = 1
    assert_eq!(result[1], 3.0); // 3^1 = 3
    assert_eq!(result[2], 16.0); // 4^2 = 16
}

#[test]
fn test_pow_simd_f64_negative_bases() {
    let base = array![-2.0, -3.0];
    let exp = array![3.0, 2.0];
    let result = pow_simd(&base.view(), &exp.view());

    assert_eq!(result[0], -8.0); // (-2)^3 = -8
    assert_eq!(result[1], 9.0); // (-3)^2 = 9
}

#[test]
fn test_pow_simd_f64_nan_propagation() {
    let base = array![f64::NAN, 2.0];
    let exp = array![2.0, f64::NAN];
    let result = pow_simd(&base.view(), &exp.view());

    assert!(result[0].is_nan());
    assert!(result[1].is_nan());
}

#[test]
fn test_pow_simd_f64_large_array() {
    let base: Array1<f64> = Array1::linspace(1.0, 10.0, 10000);
    let exp: Array1<f64> = Array1::from_elem(10000, 2.0);
    let result = pow_simd(&base.view(), &exp.view());

    // Check first and last values
    assert!((result[0] - 1.0).abs() < 1e-10);
    assert!((result[result.len() - 1] - 100.0).abs() < 1e-8);
}

#[test]
fn test_pow_simd_f32_large_array() {
    let base: Array1<f32> = Array1::linspace(1.0, 10.0, 10000);
    let exp: Array1<f32> = Array1::from_elem(10000, 2.0);
    let result = pow_simd(&base.view(), &exp.view());

    // All values should be finite and positive
    for &val in result.iter() {
        assert!(val.is_finite() && val >= 0.0);
    }
}

// ============================================================================
// Power Function Property Tests
// ============================================================================

#[test]
fn test_pow_exp_log_identity() {
    // log(x^y) = y * log(x)
    let base = array![2.0, 3.0, 5.0, 10.0];
    let exp = array![2.0, 3.0, 2.0, 2.0];

    let pow_result = pow_simd(&base.view(), &exp.view());
    let log_pow = ln_simd(&pow_result.view());

    let log_base = ln_simd(&base.view());
    let exp_log = exp
        .iter()
        .zip(log_base.iter())
        .map(|(&e, &l)| e * l)
        .collect::<Array1<_>>();

    for i in 0..base.len() {
        let diff: f64 = log_pow[i] - exp_log[i];
        assert!(diff.abs() < 1e-10, "log(x^y) != y*log(x) at index {}", i);
    }
}

#[test]
fn test_powf_square_sqrt_inverse() {
    // sqrt(x^2) = |x| for x >= 0
    let base = array![2.0, 3.0, 4.0, 5.0];
    let squared = powf_simd(&base.view(), 2.0);
    let sqrt_result = sqrt_simd(&squared.view());

    for i in 0..base.len() {
        let diff: f64 = sqrt_result[i] - base[i];
        assert!(diff.abs() < 1e-10, "sqrt(x^2) != x at index {}", i);
    }
}

// =============================================================================
// Hyperbolic Function Tests (Phase 17)
// =============================================================================

// Basic correctness tests
#[test]
fn test_sinh_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0];
    let result = sinh_simd(&x.view());

    // sinh(0) = 0
    assert!((result[0] - 0.0).abs() < 1e-10);
    // sinh(1) ≈ 1.175201194
    assert!((result[1] - 1.1752011936438014).abs() < 1e-10);
    // sinh(-1) ≈ -1.175201194
    assert!((result[2] + 1.1752011936438014).abs() < 1e-10);
    // sinh(2) ≈ 3.626860408
    assert!((result[3] - 3.626860407847019).abs() < 1e-10);
}

#[test]
fn test_sinh_simd_f32_basic() {
    let x = array![0.0f32, 1.0, -1.0, 2.0];
    let result = sinh_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.1752012).abs() < 1e-6);
    assert!((result[2] + 1.1752012).abs() < 1e-6);
    assert!((result[3] - 3.6268604).abs() < 1e-6);
}

#[test]
fn test_cosh_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0];
    let result = cosh_simd(&x.view());

    // cosh(0) = 1
    assert!((result[0] - 1.0).abs() < 1e-10);
    // cosh(1) ≈ 1.543080635
    assert!((result[1] - 1.5430806348152437).abs() < 1e-10);
    // cosh(-1) ≈ 1.543080635 (symmetric)
    assert!((result[2] - 1.5430806348152437).abs() < 1e-10);
    // cosh(2) ≈ 3.762195691
    assert!((result[3] - 3.7621956910836314).abs() < 1e-10);
}

#[test]
fn test_cosh_simd_f32_basic() {
    let x = array![0.0f32, 1.0, -1.0, 2.0];
    let result = cosh_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 1.5430807).abs() < 1e-6);
    assert!((result[2] - 1.5430807).abs() < 1e-6);
    assert!((result[3] - 3.7621956).abs() < 1e-6);
}

#[test]
fn test_tanh_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0, 10.0];
    let result = tanh_simd(&x.view());

    // tanh(0) = 0
    assert!((result[0] - 0.0).abs() < 1e-10);
    // tanh(1) ≈ 0.761594156
    assert!((result[1] - 0.7615941559557649).abs() < 1e-10);
    // tanh(-1) ≈ -0.761594156
    assert!((result[2] + 0.7615941559557649).abs() < 1e-10);
    // tanh(2) ≈ 0.964027580
    assert!((result[3] - 0.9640275800758169).abs() < 1e-10);
    // tanh(10) ≈ 1.0 (asymptotic)
    assert!((result[4] - 1.0).abs() < 1e-8);
}

#[test]
fn test_tanh_simd_f32_basic() {
    let x = array![0.0f32, 1.0, -1.0, 2.0, 10.0];
    let result = tanh_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 0.7615942).abs() < 1e-6);
    assert!((result[2] + 0.7615942).abs() < 1e-6);
    assert!((result[3] - 0.9640276).abs() < 1e-6);
    assert!((result[4] - 1.0).abs() < 1e-6);
}

// Empty array tests
#[test]
fn test_sinh_simd_empty() {
    let x: Array1<f64> = array![];
    let result = sinh_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_cosh_simd_empty() {
    let x: Array1<f64> = array![];
    let result = cosh_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_tanh_simd_empty() {
    let x: Array1<f64> = array![];
    let result = tanh_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// Zero tests
#[test]
fn test_sinh_zero() {
    let x = array![0.0];
    let result = sinh_simd(&x.view());
    assert_eq!(result[0], 0.0);
}

#[test]
fn test_cosh_zero() {
    let x = array![0.0];
    let result = cosh_simd(&x.view());
    assert_eq!(result[0], 1.0);
}

#[test]
fn test_tanh_zero() {
    let x = array![0.0];
    let result = tanh_simd(&x.view());
    assert_eq!(result[0], 0.0);
}

// NaN handling
#[test]
fn test_sinh_nan() {
    let x = array![f64::NAN];
    let result = sinh_simd(&x.view());
    assert!(result[0].is_nan());
}

#[test]
fn test_cosh_nan() {
    let x = array![f64::NAN];
    let result = cosh_simd(&x.view());
    assert!(result[0].is_nan());
}

#[test]
fn test_tanh_nan() {
    let x = array![f64::NAN];
    let result = tanh_simd(&x.view());
    assert!(result[0].is_nan());
}

// Hyperbolic identity: cosh²(x) - sinh²(x) = 1
#[test]
fn test_hyperbolic_identity_cosh_squared_minus_sinh_squared() {
    let x = array![0.0_f64, 0.5, 1.0, 1.5, 2.0];
    let sinh_result = sinh_simd(&x.view());
    let cosh_result = cosh_simd(&x.view());

    for i in 0..x.len() {
        let identity = cosh_result[i] * cosh_result[i] - sinh_result[i] * sinh_result[i];
        assert!(
            (identity - 1.0).abs() < 1e-10,
            "cosh²(x) - sinh²(x) != 1 at index {}",
            i
        );
    }
}

// Hyperbolic identity: tanh(x) = sinh(x) / cosh(x)
#[test]
fn test_hyperbolic_identity_tanh_equals_sinh_over_cosh() {
    let x = array![0.0_f64, 0.5, 1.0, 1.5, 2.0];
    let sinh_result = sinh_simd(&x.view());
    let cosh_result = cosh_simd(&x.view());
    let tanh_result = tanh_simd(&x.view());

    for i in 0..x.len() {
        let tanh_from_ratio = sinh_result[i] / cosh_result[i];
        assert!(
            (tanh_result[i] - tanh_from_ratio).abs() < 1e-10,
            "tanh(x) != sinh(x)/cosh(x) at index {}",
            i
        );
    }
}

// Anti-symmetry: sinh(-x) = -sinh(x)
#[test]
fn test_sinh_anti_symmetry() {
    let x = array![0.5_f64, 1.0, 1.5, 2.0];
    let sinh_pos = sinh_simd(&x.view());
    let x_neg = array![-0.5_f64, -1.0, -1.5, -2.0];
    let sinh_neg = sinh_simd(&x_neg.view());

    for i in 0..x.len() {
        assert!(
            (sinh_neg[i] + sinh_pos[i]).abs() < 1e-10,
            "sinh(-x) != -sinh(x) at index {}",
            i
        );
    }
}

// Anti-symmetry: tanh(-x) = -tanh(x)
#[test]
fn test_tanh_anti_symmetry() {
    let x = array![0.5_f64, 1.0, 1.5, 2.0];
    let tanh_pos = tanh_simd(&x.view());
    let x_neg = array![-0.5_f64, -1.0, -1.5, -2.0];
    let tanh_neg = tanh_simd(&x_neg.view());

    for i in 0..x.len() {
        assert!(
            (tanh_neg[i] + tanh_pos[i]).abs() < 1e-10,
            "tanh(-x) != -tanh(x) at index {}",
            i
        );
    }
}

// Symmetry: cosh(-x) = cosh(x)
#[test]
fn test_cosh_symmetry() {
    let x = array![0.5_f64, 1.0, 1.5, 2.0];
    let cosh_pos = cosh_simd(&x.view());
    let x_neg = array![-0.5_f64, -1.0, -1.5, -2.0];
    let cosh_neg = cosh_simd(&x_neg.view());

    for i in 0..x.len() {
        assert!(
            (cosh_neg[i] - cosh_pos[i]).abs() < 1e-10,
            "cosh(-x) != cosh(x) at index {}",
            i
        );
    }
}

// Asymptotic behavior: tanh(x) → ±1 as x → ±∞
#[test]
fn test_tanh_asymptotic_behavior() {
    let x_large_pos = array![10.0_f64, 20.0, 50.0];
    let result_pos = tanh_simd(&x_large_pos.view());

    for i in 0..x_large_pos.len() {
        assert!(
            (result_pos[i] - 1.0).abs() < 1e-8,
            "tanh(large_positive) should approach 1"
        );
    }

    let x_large_neg = array![-10.0_f64, -20.0, -50.0];
    let result_neg = tanh_simd(&x_large_neg.view());

    for i in 0..x_large_neg.len() {
        assert!(
            (result_neg[i] + 1.0).abs() < 1e-8,
            "tanh(large_negative) should approach -1"
        );
    }
}

// Range constraint: cosh(x) >= 1 always
#[test]
fn test_cosh_range_constraint() {
    let x = array![-10.0_f64, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0];
    let result = cosh_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i] >= 1.0,
            "cosh(x) must be >= 1, but got {} at index {}",
            result[i],
            i
        );
    }
}

// Range constraint: |tanh(x)| < 1 always
#[test]
fn test_tanh_range_constraint() {
    let x = array![-10.0_f64, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0];
    let result = tanh_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i].abs() <= 1.0,
            "|tanh(x)| must be <= 1, but got {} at index {}",
            result[i],
            i
        );
    }
}

// Derivative identity: d/dx tanh(x) = 1 - tanh²(x) = sech²(x)
#[test]
fn test_tanh_derivative_identity() {
    let x = array![0.0_f64, 0.5, 1.0, 1.5];
    let tanh_result = tanh_simd(&x.view());

    for i in 0..x.len() {
        // d/dx tanh(x) = 1 - tanh²(x)
        let derivative = 1.0 - tanh_result[i] * tanh_result[i];

        // Also equals sech²(x) = 1/cosh²(x)
        let cosh_val = x[i].cosh();
        let sech_squared = 1.0 / (cosh_val * cosh_val);

        assert!(
            (derivative - sech_squared).abs() < 1e-10,
            "1 - tanh²(x) != sech²(x) at index {}",
            i
        );
    }
}

// Relation with exponential: sinh(x) = (e^x - e^(-x))/2
#[test]
fn test_sinh_exponential_definition() {
    let x = array![0.0_f64, 0.5, 1.0, 2.0];
    let sinh_result = sinh_simd(&x.view());

    for i in 0..x.len() {
        let sinh_from_exp = (x[i].exp() - (-x[i]).exp()) / 2.0;
        assert!(
            (sinh_result[i] - sinh_from_exp).abs() < 1e-10,
            "sinh(x) != (e^x - e^(-x))/2 at index {}",
            i
        );
    }
}

// Relation with exponential: cosh(x) = (e^x + e^(-x))/2
#[test]
fn test_cosh_exponential_definition() {
    let x = array![0.0_f64, 0.5, 1.0, 2.0];
    let cosh_result = cosh_simd(&x.view());

    for i in 0..x.len() {
        let cosh_from_exp = (x[i].exp() + (-x[i]).exp()) / 2.0;
        assert!(
            (cosh_result[i] - cosh_from_exp).abs() < 1e-10,
            "cosh(x) != (e^x + e^(-x))/2 at index {}",
            i
        );
    }
}

// ============================================================================
// Floor SIMD Tests
// ============================================================================

#[test]
fn test_floor_simd_f64_basic() {
    let x = array![1.2_f64, 2.7, -1.3, -2.9, 3.0, 0.1, -0.9];
    let result = floor_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], -2.0);
    assert_eq!(result[3], -3.0);
    assert_eq!(result[4], 3.0);
    assert_eq!(result[5], 0.0);
    assert_eq!(result[6], -1.0);
}

#[test]
fn test_floor_simd_f32_basic() {
    let x = array![1.2_f32, 2.7, -1.3, -2.9, 3.0, 0.1, -0.9];
    let result = floor_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 2.0);
    assert_eq!(result[2], -2.0);
    assert_eq!(result[3], -3.0);
    assert_eq!(result[4], 3.0);
    assert_eq!(result[5], 0.0);
    assert_eq!(result[6], -1.0);
}

#[test]
fn test_floor_simd_empty() {
    let x: Array1<f64> = array![];
    let result = floor_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_floor_simd_integers() {
    let x = array![1.0_f64, 2.0, 3.0, -1.0, -2.0];
    let result = floor_simd(&x.view());

    for i in 0..x.len() {
        assert_eq!(result[i], x[i], "floor(integer) should equal integer");
    }
}

#[test]
fn test_floor_simd_nan_propagation() {
    let x = array![1.5_f64, f64::NAN, 2.5];
    let result = floor_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert!(result[1].is_nan());
    assert_eq!(result[2], 2.0);
}

// Mathematical property: floor(x) <= x
#[test]
fn test_floor_property_less_than_or_equal() {
    let x = array![1.1_f64, 2.9, -0.5, -1.7, 3.5, 0.01];
    let result = floor_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i] <= x[i],
            "floor(x) should be <= x, but floor({}) = {} > {}",
            x[i],
            result[i],
            x[i]
        );
    }
}

// Relationship: floor(-x) = -ceil(x)
#[test]
fn test_floor_ceil_relationship() {
    let x = array![1.2_f64, 2.7, 3.5, 0.1, 0.9];
    let neg_x = x.mapv(|v| -v);

    let floor_neg_x = floor_simd(&neg_x.view());
    let ceil_x = ceil_simd(&x.view());
    let neg_ceil_x = ceil_x.mapv(|v| -v);

    for i in 0..x.len() {
        assert_eq!(
            floor_neg_x[i], neg_ceil_x[i],
            "floor(-x) should equal -ceil(x)"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_floor_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).unwrap();
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let result = floor_simd(&x.view());

    // Verify correctness for first 100 elements
    for i in 0..100.min(data.len()) {
        assert_eq!(result[i], data[i].floor());
    }
}

// ============================================================================
// Ceil SIMD Tests
// ============================================================================

#[test]
fn test_ceil_simd_f64_basic() {
    let x = array![1.2_f64, 2.7, -1.3, -2.9, 3.0, 0.1, -0.9];
    let result = ceil_simd(&x.view());

    assert_eq!(result[0], 2.0);
    assert_eq!(result[1], 3.0);
    assert_eq!(result[2], -1.0);
    assert_eq!(result[3], -2.0);
    assert_eq!(result[4], 3.0);
    assert_eq!(result[5], 1.0);
    assert_eq!(result[6], 0.0);
}

#[test]
fn test_ceil_simd_f32_basic() {
    let x = array![1.2_f32, 2.7, -1.3, -2.9, 3.0, 0.1, -0.9];
    let result = ceil_simd(&x.view());

    assert_eq!(result[0], 2.0);
    assert_eq!(result[1], 3.0);
    assert_eq!(result[2], -1.0);
    assert_eq!(result[3], -2.0);
    assert_eq!(result[4], 3.0);
    assert_eq!(result[5], 1.0);
    assert_eq!(result[6], 0.0);
}

#[test]
fn test_ceil_simd_empty() {
    let x: Array1<f64> = array![];
    let result = ceil_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_ceil_simd_integers() {
    let x = array![1.0_f64, 2.0, 3.0, -1.0, -2.0];
    let result = ceil_simd(&x.view());

    for i in 0..x.len() {
        assert_eq!(result[i], x[i], "ceil(integer) should equal integer");
    }
}

#[test]
fn test_ceil_simd_nan_propagation() {
    let x = array![1.5_f64, f64::NAN, 2.5];
    let result = ceil_simd(&x.view());

    assert_eq!(result[0], 2.0);
    assert!(result[1].is_nan());
    assert_eq!(result[2], 3.0);
}

// Mathematical property: ceil(x) >= x
#[test]
fn test_ceil_property_greater_than_or_equal() {
    let x = array![1.1_f64, 2.9, -0.5, -1.7, 3.5, 0.01];
    let result = ceil_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i] >= x[i],
            "ceil(x) should be >= x, but ceil({}) = {} < {}",
            x[i],
            result[i],
            x[i]
        );
    }
}

// Relationship: ceil(-x) = -floor(x)
#[test]
fn test_ceil_floor_relationship() {
    let x = array![1.2_f64, 2.7, 3.5, 0.1, 0.9];
    let neg_x = x.mapv(|v| -v);

    let ceil_neg_x = ceil_simd(&neg_x.view());
    let floor_x = floor_simd(&x.view());
    let neg_floor_x = floor_x.mapv(|v| -v);

    for i in 0..x.len() {
        assert_eq!(
            ceil_neg_x[i], neg_floor_x[i],
            "ceil(-x) should equal -floor(x)"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_ceil_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).unwrap();
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let result = ceil_simd(&x.view());

    // Verify correctness for first 100 elements
    for i in 0..100.min(data.len()) {
        assert_eq!(result[i], data[i].ceil());
    }
}

// ============================================================================
// Round SIMD Tests
// ============================================================================

#[test]
fn test_round_simd_f64_basic() {
    let x = array![1.2_f64, 2.7, -1.3, -2.9, 3.0, 0.4, -0.6];
    let result = round_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 3.0);
    assert_eq!(result[2], -1.0);
    assert_eq!(result[3], -3.0);
    assert_eq!(result[4], 3.0);
    assert_eq!(result[5], 0.0);
    assert_eq!(result[6], -1.0);
}

#[test]
fn test_round_simd_f32_basic() {
    let x = array![1.2_f32, 2.7, -1.3, -2.9, 3.0, 0.4, -0.6];
    let result = round_simd(&x.view());

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 3.0);
    assert_eq!(result[2], -1.0);
    assert_eq!(result[3], -3.0);
    assert_eq!(result[4], 3.0);
    assert_eq!(result[5], 0.0);
    assert_eq!(result[6], -1.0);
}

#[test]
fn test_round_simd_empty() {
    let x: Array1<f64> = array![];
    let result = round_simd(&x.view());
    assert_eq!(result.len(), 0);
}

#[test]
fn test_round_simd_integers() {
    let x = array![1.0_f64, 2.0, 3.0, -1.0, -2.0];
    let result = round_simd(&x.view());

    for i in 0..x.len() {
        assert_eq!(result[i], x[i], "round(integer) should equal integer");
    }
}

// Round half away from zero (Rust default behavior)
#[test]
fn test_round_simd_half_ties() {
    let x = array![0.5_f64, 1.5, 2.5, 3.5, 4.5, -0.5, -1.5, -2.5];
    let result = round_simd(&x.view());

    // Rust rounds half away from zero
    assert_eq!(result[0], 1.0); // 0.5 -> 1.0
    assert_eq!(result[1], 2.0); // 1.5 -> 2.0
    assert_eq!(result[2], 3.0); // 2.5 -> 3.0
    assert_eq!(result[3], 4.0); // 3.5 -> 4.0
    assert_eq!(result[4], 5.0); // 4.5 -> 5.0
    assert_eq!(result[5], -1.0); // -0.5 -> -1.0
    assert_eq!(result[6], -2.0); // -1.5 -> -2.0
    assert_eq!(result[7], -3.0); // -2.5 -> -3.0
}

#[test]
fn test_round_simd_nan_propagation() {
    let x = array![1.5_f64, f64::NAN, 2.5];
    let result = round_simd(&x.view());

    assert_eq!(result[0], 2.0);
    assert!(result[1].is_nan());
    assert_eq!(result[2], 3.0);
}

// Mathematical property: |round(x) - x| <= 0.5
#[test]
fn test_round_property_distance() {
    let x = array![1.1_f64, 2.9, 3.5, -0.7, -1.5, 0.01, 4.99];
    let result = round_simd(&x.view());

    for i in 0..x.len() {
        let distance = (result[i] - x[i]).abs();
        assert!(
            distance <= 0.5,
            "|round({}) - {}| = {} should be <= 0.5",
            x[i],
            x[i],
            distance
        );
    }
}

// Symmetry: round(-x) = -round(x)
#[test]
fn test_round_symmetry() {
    let x = array![1.2_f64, 2.7, 3.3, 0.9];
    let neg_x = x.mapv(|v| -v);

    let round_x = round_simd(&x.view());
    let round_neg_x = round_simd(&neg_x.view());

    for i in 0..x.len() {
        assert_eq!(
            round_neg_x[i], -round_x[i],
            "round(-x) should equal -round(x)"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_round_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).unwrap();
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let result = round_simd(&x.view());

    // Verify correctness for first 100 elements
    for i in 0..100.min(data.len()) {
        assert_eq!(result[i], data[i].round());
    }
}

// ============================================================================
// Cross-function Rounding Tests
// ============================================================================

// Property: floor(x) <= round(x) <= ceil(x)
#[test]
fn test_rounding_ordering() {
    let x = array![1.1_f64, 2.9, 3.5, -0.7, -1.5];

    let floor_result = floor_simd(&x.view());
    let round_result = round_simd(&x.view());
    let ceil_result = ceil_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            floor_result[i] <= round_result[i] && round_result[i] <= ceil_result[i],
            "floor({}) <= round({}) <= ceil({}) not satisfied",
            x[i],
            x[i],
            x[i]
        );
    }
}

// Property: ceil(x) - floor(x) is 0 or 1
#[test]
fn test_ceil_minus_floor() {
    let x = array![1.1_f64, 2.0, 3.5, -0.7, -1.0];

    let floor_result = floor_simd(&x.view());
    let ceil_result = ceil_simd(&x.view());

    for i in 0..x.len() {
        let diff = ceil_result[i] - floor_result[i];
        assert!(
            diff == 0.0 || diff == 1.0,
            "ceil(x) - floor(x) should be 0 or 1, got {} for x={}",
            diff,
            x[i]
        );
    }
}

// ============================================================================
// Atan SIMD Tests
// ============================================================================

#[test]
fn test_atan_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY];
    let result = atan_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-10);
    assert!((result[1] - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
    assert!((result[2] + std::f64::consts::FRAC_PI_4).abs() < 1e-10);
    assert!((result[3] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    assert!((result[4] + std::f64::consts::FRAC_PI_2).abs() < 1e-10);
}

#[test]
fn test_atan_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, -1.0];
    let result = atan_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - std::f32::consts::FRAC_PI_4).abs() < 1e-6);
    assert!((result[2] + std::f32::consts::FRAC_PI_4).abs() < 1e-6);
}

#[test]
fn test_atan_simd_empty() {
    let x: Array1<f64> = array![];
    let result = atan_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// Anti-symmetry: atan(-x) = -atan(x)
#[test]
fn test_atan_anti_symmetry() {
    let x = array![0.5_f64, 1.0, 2.0, 3.0];
    let neg_x = x.mapv(|v| -v);

    let atan_x = atan_simd(&x.view());
    let atan_neg_x = atan_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!(
            (atan_neg_x[i] + atan_x[i]).abs() < 1e-10,
            "atan(-x) should equal -atan(x)"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_atan_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-10.0, 10.0).unwrap();
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let result = atan_simd(&x.view());

    for i in 0..100.min(data.len()) {
        assert_eq!(result[i], data[i].atan());
    }
}

// ============================================================================
// Asin SIMD Tests
// ============================================================================

#[test]
fn test_asin_simd_f64_basic() {
    let x = array![0.0_f64, 0.5, 1.0, -1.0];
    let result = asin_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-10);
    assert!((result[1] - std::f64::consts::FRAC_PI_6).abs() < 1e-10);
    assert!((result[2] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    assert!((result[3] + std::f64::consts::FRAC_PI_2).abs() < 1e-10);
}

#[test]
fn test_asin_simd_f32_basic() {
    let x = array![0.0_f32, 0.5, 1.0, -1.0];
    let result = asin_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - std::f32::consts::FRAC_PI_6).abs() < 1e-6);
    assert!((result[2] - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    assert!((result[3] + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
}

#[test]
fn test_asin_simd_empty() {
    let x: Array1<f64> = array![];
    let result = asin_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// Out of domain returns NaN
#[test]
fn test_asin_out_of_domain() {
    let x = array![0.5_f64, 1.5, -1.5, 2.0];
    let result = asin_simd(&x.view());

    assert!(!result[0].is_nan());
    assert!(result[1].is_nan());
    assert!(result[2].is_nan());
    assert!(result[3].is_nan());
}

// Anti-symmetry: asin(-x) = -asin(x)
#[test]
fn test_asin_anti_symmetry() {
    let x = array![0.0_f64, 0.5, 0.7, 1.0];
    let neg_x = x.mapv(|v| -v);

    let asin_x = asin_simd(&x.view());
    let asin_neg_x = asin_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!(
            (asin_neg_x[i] + asin_x[i]).abs() < 1e-10,
            "asin(-x) should equal -asin(x)"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_asin_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-1.0, 1.0).unwrap();
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let result = asin_simd(&x.view());

    for i in 0..100.min(data.len()) {
        assert_eq!(result[i], data[i].asin());
    }
}

// ============================================================================
// Acos SIMD Tests
// ============================================================================

#[test]
fn test_acos_simd_f64_basic() {
    let x = array![1.0_f64, 0.5, 0.0, -1.0];
    let result = acos_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-10);
    assert!((result[1] - std::f64::consts::FRAC_PI_3).abs() < 1e-10);
    assert!((result[2] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    assert!((result[3] - std::f64::consts::PI).abs() < 1e-10);
}

#[test]
fn test_acos_simd_f32_basic() {
    let x = array![1.0_f32, 0.5, 0.0, -1.0];
    let result = acos_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - std::f32::consts::FRAC_PI_3).abs() < 1e-6);
    assert!((result[2] - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    assert!((result[3] - std::f32::consts::PI).abs() < 1e-6);
}

#[test]
fn test_acos_simd_empty() {
    let x: Array1<f64> = array![];
    let result = acos_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// Out of domain returns NaN
#[test]
fn test_acos_out_of_domain() {
    let x = array![0.5_f64, 1.5, -1.5, 2.0];
    let result = acos_simd(&x.view());

    assert!(!result[0].is_nan());
    assert!(result[1].is_nan());
    assert!(result[2].is_nan());
    assert!(result[3].is_nan());
}

// Identity: acos(x) + asin(x) = π/2
#[test]
fn test_acos_asin_identity() {
    let x = array![0.0_f64, 0.25, 0.5, 0.75, 1.0];
    let acos_result = acos_simd(&x.view());
    let asin_result = asin_simd(&x.view());

    for i in 0..x.len() {
        let sum = acos_result[i] + asin_result[i];
        assert!(
            (sum - std::f64::consts::FRAC_PI_2).abs() < 1e-10,
            "acos(x) + asin(x) should equal π/2"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_acos_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-1.0, 1.0).unwrap();
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let result = acos_simd(&x.view());

    for i in 0..100.min(data.len()) {
        assert_eq!(result[i], data[i].acos());
    }
}

// ============================================================================
// Atan2 SIMD Tests
// ============================================================================

#[test]
fn test_atan2_simd_f64_basic() {
    let y = array![1.0_f64, 1.0, -1.0, -1.0];
    let x = array![1.0_f64, -1.0, -1.0, 1.0];
    let result = atan2_simd(&y.view(), &x.view());

    // Quadrant I: (1, 1) -> π/4
    assert!((result[0] - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
    // Quadrant II: (1, -1) -> 3π/4
    assert!((result[1] - 3.0 * std::f64::consts::FRAC_PI_4).abs() < 1e-10);
    // Quadrant III: (-1, -1) -> -3π/4
    assert!((result[2] + 3.0 * std::f64::consts::FRAC_PI_4).abs() < 1e-10);
    // Quadrant IV: (-1, 1) -> -π/4
    assert!((result[3] + std::f64::consts::FRAC_PI_4).abs() < 1e-10);
}

#[test]
fn test_atan2_simd_f32_basic() {
    let y = array![1.0_f32, 0.0, -1.0];
    let x = array![0.0_f32, 1.0, 0.0];
    let result = atan2_simd(&y.view(), &x.view());

    assert!((result[0] - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    assert!((result[1] - 0.0).abs() < 1e-6);
    assert!((result[2] + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
}

#[test]
fn test_atan2_simd_empty() {
    let y: Array1<f64> = array![];
    let x: Array1<f64> = array![];
    let result = atan2_simd(&y.view(), &x.view());
    assert_eq!(result.len(), 0);
}

// Test all four quadrants
#[test]
fn test_atan2_quadrants() {
    // Quadrant I: x > 0, y > 0
    let y1 = array![1.0_f64];
    let x1 = array![1.0_f64];
    let result1 = atan2_simd(&y1.view(), &x1.view());
    assert!(result1[0] > 0.0 && result1[0] < std::f64::consts::FRAC_PI_2);

    // Quadrant II: x < 0, y > 0
    let y2 = array![1.0_f64];
    let x2 = array![-1.0_f64];
    let result2 = atan2_simd(&y2.view(), &x2.view());
    assert!(result2[0] > std::f64::consts::FRAC_PI_2 && result2[0] < std::f64::consts::PI);

    // Quadrant III: x < 0, y < 0
    let y3 = array![-1.0_f64];
    let x3 = array![-1.0_f64];
    let result3 = atan2_simd(&y3.view(), &x3.view());
    assert!(result3[0] < -std::f64::consts::FRAC_PI_2 && result3[0] > -std::f64::consts::PI);

    // Quadrant IV: x > 0, y < 0
    let y4 = array![-1.0_f64];
    let x4 = array![1.0_f64];
    let result4 = atan2_simd(&y4.view(), &x4.view());
    assert!(result4[0] < 0.0 && result4[0] > -std::f64::consts::FRAC_PI_2);
}

// Anti-symmetry: atan2(-y, x) = -atan2(y, x)
#[test]
fn test_atan2_anti_symmetry() {
    let y = array![0.5_f64, 1.0, 2.0];
    let x = array![1.0_f64, 1.0, 1.0];
    let neg_y = y.mapv(|v| -v);

    let atan2_y_x = atan2_simd(&y.view(), &x.view());
    let atan2_neg_y_x = atan2_simd(&neg_y.view(), &x.view());

    for i in 0..y.len() {
        assert!(
            (atan2_neg_y_x[i] + atan2_y_x[i]).abs() < 1e-10,
            "atan2(-y, x) should equal -atan2(y, x)"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_atan2_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-10.0, 10.0).unwrap();
    let y_data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x_data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let y = Array1::from(y_data.clone());
    let x = Array1::from(x_data.clone());

    let result = atan2_simd(&y.view(), &x.view());

    for i in 0..100.min(y_data.len()) {
        assert_eq!(result[i], y_data[i].atan2(x_data[i]));
    }
}

// ============================================================================
// Inverse Function Property Tests
// ============================================================================

// Property: sin(asin(x)) = x for x ∈ [-1, 1]
#[test]
fn test_sin_asin_inverse() {
    let x = array![0.0_f64, 0.25, 0.5, 0.75, 1.0, -0.5, -1.0];
    let asin_result = asin_simd(&x.view());
    let sin_result = sin_simd(&asin_result.view());

    for i in 0..x.len() {
        assert!(
            (sin_result[i] - x[i]).abs() < 1e-10,
            "sin(asin(x)) should equal x"
        );
    }
}

// Property: cos(acos(x)) = x for x ∈ [-1, 1]
#[test]
fn test_cos_acos_inverse() {
    let x = array![1.0_f64, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0];
    let acos_result = acos_simd(&x.view());
    let cos_result = cos_simd(&acos_result.view());

    for i in 0..x.len() {
        assert!(
            (cos_result[i] - x[i]).abs() < 1e-10,
            "cos(acos(x)) should equal x"
        );
    }
}

// Property: tan(atan(x)) = x for all x
#[test]
fn test_tan_atan_inverse() {
    let x = array![0.0_f64, 0.5, 1.0, 2.0, -1.0, -2.0];
    let atan_result = atan_simd(&x.view());
    let tan_result = tan_simd(&atan_result.view());

    for i in 0..x.len() {
        assert!(
            (tan_result[i] - x[i]).abs() < 1e-10,
            "tan(atan(x)) should equal x"
        );
    }
}

// ============================================================================
// Log10 SIMD Tests
// ============================================================================

#[test]
fn test_log10_simd_f64_basic() {
    let x = array![1.0_f64, 10.0, 100.0, 1000.0];
    let result = log10_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-10);
    assert!((result[1] - 1.0).abs() < 1e-10);
    assert!((result[2] - 2.0).abs() < 1e-10);
    assert!((result[3] - 3.0).abs() < 1e-10);
}

#[test]
fn test_log10_simd_f32_basic() {
    let x = array![1.0_f32, 10.0, 100.0, 1000.0];
    let result = log10_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!((result[2] - 2.0).abs() < 1e-6);
    assert!((result[3] - 3.0).abs() < 1e-6);
}

#[test]
fn test_log10_simd_empty() {
    let x: Array1<f64> = array![];
    let result = log10_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// Domain constraint: log10(negative) = NaN
#[test]
fn test_log10_negative_returns_nan() {
    let x = array![1.0_f64, -1.0, 10.0, -10.0];
    let result = log10_simd(&x.view());

    assert!(!result[0].is_nan());
    assert!(result[1].is_nan());
    assert!(!result[2].is_nan());
    assert!(result[3].is_nan());
}

// Domain constraint: log10(0) = -∞
#[test]
fn test_log10_zero_returns_neg_inf() {
    let x = array![0.0_f64, 1.0];
    let result = log10_simd(&x.view());

    assert!(result[0].is_infinite() && result[0].is_sign_negative());
    assert!((result[1] - 0.0).abs() < 1e-10);
}

// Logarithm property: log10(a * b) = log10(a) + log10(b)
#[test]
fn test_log10_product_property() {
    let a = array![2.0_f64, 5.0, 10.0];
    let b = array![5.0_f64, 2.0, 10.0];

    let product = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .collect::<Vec<_>>();
    let product_array = Array1::from(product);

    let log_a = log10_simd(&a.view());
    let log_b = log10_simd(&b.view());
    let log_product = log10_simd(&product_array.view());

    for i in 0..a.len() {
        assert!(
            (log_product[i] - (log_a[i] + log_b[i])).abs() < 1e-10,
            "log10(a*b) should equal log10(a) + log10(b)"
        );
    }
}

// Logarithm property: log10(a / b) = log10(a) - log10(b)
#[test]
fn test_log10_quotient_property() {
    let a = array![100.0_f64, 1000.0, 10000.0];
    let b = array![10.0_f64, 100.0, 1000.0];

    let quotient = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x / y)
        .collect::<Vec<_>>();
    let quotient_array = Array1::from(quotient);

    let log_a = log10_simd(&a.view());
    let log_b = log10_simd(&b.view());
    let log_quotient = log10_simd(&quotient_array.view());

    for i in 0..a.len() {
        assert!(
            (log_quotient[i] - (log_a[i] - log_b[i])).abs() < 1e-10,
            "log10(a/b) should equal log10(a) - log10(b)"
        );
    }
}

// Logarithm property: log10(a^n) = n * log10(a)
#[test]
fn test_log10_power_property() {
    let a = array![2.0_f64, 5.0, 10.0];
    let n = 3.0;

    let power = a.mapv(|x| x.powf(n));

    let log_a = log10_simd(&a.view());
    let log_power = log10_simd(&power.view());

    for i in 0..a.len() {
        assert!(
            (log_power[i] - n * log_a[i]).abs() < 1e-10,
            "log10(a^n) should equal n * log10(a)"
        );
    }
}

// Change of base formula: log10(x) = ln(x) / ln(10)
#[test]
fn test_log10_change_of_base() {
    let x = array![1.0_f64, 2.0, 5.0, 10.0, 50.0, 100.0];
    let log10_result = log10_simd(&x.view());
    let ln_result = ln_simd(&x.view());
    let ln10 = 10.0_f64.ln();

    for i in 0..x.len() {
        assert!(
            (log10_result[i] - ln_result[i] / ln10).abs() < 1e-10,
            "log10(x) should equal ln(x) / ln(10)"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_log10_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(1.0, 1000.0).unwrap();
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let result = log10_simd(&x.view());

    for i in 0..100.min(data.len()) {
        let expected = data[i].log10();
        let diff = (result[i] - expected).abs();
        assert!(
            diff < 1e-14,
            "log10 mismatch at {}: {} vs {} (diff: {})",
            i,
            result[i],
            expected,
            diff
        );
    }
}

// ============================================================================
// Log2 SIMD Tests
// ============================================================================

#[test]
fn test_log2_simd_f64_basic() {
    let x = array![1.0_f64, 2.0, 4.0, 8.0, 16.0];
    let result = log2_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-10);
    assert!((result[1] - 1.0).abs() < 1e-10);
    assert!((result[2] - 2.0).abs() < 1e-10);
    assert!((result[3] - 3.0).abs() < 1e-10);
    assert!((result[4] - 4.0).abs() < 1e-10);
}

#[test]
fn test_log2_simd_f32_basic() {
    let x = array![1.0_f32, 2.0, 4.0, 8.0, 16.0];
    let result = log2_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!((result[2] - 2.0).abs() < 1e-6);
    assert!((result[3] - 3.0).abs() < 1e-6);
    assert!((result[4] - 4.0).abs() < 1e-6);
}

#[test]
fn test_log2_simd_empty() {
    let x: Array1<f64> = array![];
    let result = log2_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// Domain constraint: log2(negative) = NaN
#[test]
fn test_log2_negative_returns_nan() {
    let x = array![1.0_f64, -1.0, 2.0, -2.0];
    let result = log2_simd(&x.view());

    assert!(!result[0].is_nan());
    assert!(result[1].is_nan());
    assert!(!result[2].is_nan());
    assert!(result[3].is_nan());
}

// Domain constraint: log2(0) = -∞
#[test]
fn test_log2_zero_returns_neg_inf() {
    let x = array![0.0_f64, 1.0];
    let result = log2_simd(&x.view());

    assert!(result[0].is_infinite() && result[0].is_sign_negative());
    assert!((result[1] - 0.0).abs() < 1e-10);
}

// Logarithm property: log2(a * b) = log2(a) + log2(b)
#[test]
fn test_log2_product_property() {
    let a = array![2.0_f64, 4.0, 8.0];
    let b = array![4.0_f64, 2.0, 8.0];

    let product = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .collect::<Vec<_>>();
    let product_array = Array1::from(product);

    let log_a = log2_simd(&a.view());
    let log_b = log2_simd(&b.view());
    let log_product = log2_simd(&product_array.view());

    for i in 0..a.len() {
        assert!(
            (log_product[i] - (log_a[i] + log_b[i])).abs() < 1e-10,
            "log2(a*b) should equal log2(a) + log2(b)"
        );
    }
}

// Logarithm property: log2(a / b) = log2(a) - log2(b)
#[test]
fn test_log2_quotient_property() {
    let a = array![16.0_f64, 64.0, 256.0];
    let b = array![2.0_f64, 4.0, 8.0];

    let quotient = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x / y)
        .collect::<Vec<_>>();
    let quotient_array = Array1::from(quotient);

    let log_a = log2_simd(&a.view());
    let log_b = log2_simd(&b.view());
    let log_quotient = log2_simd(&quotient_array.view());

    for i in 0..a.len() {
        assert!(
            (log_quotient[i] - (log_a[i] - log_b[i])).abs() < 1e-10,
            "log2(a/b) should equal log2(a) - log2(b)"
        );
    }
}

// Logarithm property: log2(a^n) = n * log2(a)
#[test]
fn test_log2_power_property() {
    let a = array![2.0_f64, 3.0, 5.0];
    let n = 4.0;

    let power = a.mapv(|x| x.powf(n));

    let log_a = log2_simd(&a.view());
    let log_power = log2_simd(&power.view());

    for i in 0..a.len() {
        assert!(
            (log_power[i] - n * log_a[i]).abs() < 1e-10,
            "log2(a^n) should equal n * log2(a)"
        );
    }
}

// Change of base formula: log2(x) = ln(x) / ln(2)
#[test]
fn test_log2_change_of_base() {
    let x = array![1.0_f64, 2.0, 4.0, 8.0, 16.0, 32.0];
    let log2_result = log2_simd(&x.view());
    let ln_result = ln_simd(&x.view());
    let ln2 = 2.0_f64.ln();

    for i in 0..x.len() {
        assert!(
            (log2_result[i] - ln_result[i] / ln2).abs() < 1e-10,
            "log2(x) should equal ln(x) / ln(2)"
        );
    }
}

#[cfg(feature = "random")]
#[test]
fn test_log2_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(1.0, 1024.0).unwrap();
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let result = log2_simd(&x.view());

    for i in 0..100.min(data.len()) {
        let expected = data[i].log2();
        let diff = (result[i] - expected).abs();
        assert!(
            diff < 1e-14,
            "log2 mismatch at {}: {} vs {} (diff: {})",
            i,
            result[i],
            expected,
            diff
        );
    }
}

// ============================================================================
// Clamp SIMD Tests
// ============================================================================

#[test]
fn test_clamp_simd_f64_basic() {
    let x = array![-1.0_f64, 0.5, 1.5, 2.5, 3.5];
    let result = clamp_simd(&x.view(), 0.0, 2.0);

    assert_eq!(result[0], 0.0); // Below min
    assert_eq!(result[1], 0.5); // Within range
    assert_eq!(result[2], 1.5); // Within range
    assert_eq!(result[3], 2.0); // Above max
    assert_eq!(result[4], 2.0); // Above max
}

#[test]
fn test_clamp_simd_f32_basic() {
    let x = array![-1.0_f32, 0.5, 1.5, 2.5, 3.5];
    let result = clamp_simd(&x.view(), 0.0, 2.0);

    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 0.5);
    assert_eq!(result[2], 1.5);
    assert_eq!(result[3], 2.0);
    assert_eq!(result[4], 2.0);
}

#[test]
fn test_clamp_simd_empty() {
    let x: Array1<f64> = array![];
    let result = clamp_simd(&x.view(), 0.0, 1.0);
    assert_eq!(result.len(), 0);
}

// Boundary conditions: values at min/max
#[test]
fn test_clamp_boundary_values() {
    let x = array![0.0_f64, 1.0, 2.0];
    let result = clamp_simd(&x.view(), 0.0, 2.0);

    assert_eq!(result[0], 0.0); // At min
    assert_eq!(result[1], 1.0); // Within
    assert_eq!(result[2], 2.0); // At max
}

// All values below min
#[test]
fn test_clamp_all_below_min() {
    let x = array![-5.0_f64, -3.0, -1.0];
    let result = clamp_simd(&x.view(), 0.0, 10.0);

    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

// All values above max
#[test]
fn test_clamp_all_above_max() {
    let x = array![11.0_f64, 15.0, 20.0];
    let result = clamp_simd(&x.view(), 0.0, 10.0);

    for &val in result.iter() {
        assert_eq!(val, 10.0);
    }
}

// All values within range (should remain unchanged)
#[test]
fn test_clamp_all_within_range() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let result = clamp_simd(&x.view(), 0.0, 10.0);

    for i in 0..x.len() {
        assert_eq!(result[i], x[i]);
    }
}

// Idempotency: clamp(clamp(x)) = clamp(x)
#[test]
fn test_clamp_idempotent() {
    let x = array![-5.0_f64, 0.5, 5.0, 15.0];
    let result1 = clamp_simd(&x.view(), 0.0, 10.0);
    let result2 = clamp_simd(&result1.view(), 0.0, 10.0);

    for i in 0..x.len() {
        assert_eq!(result1[i], result2[i], "Clamp should be idempotent");
    }
}

// Monotonicity: if x1 <= x2, then clamp(x1) <= clamp(x2)
#[test]
fn test_clamp_preserves_monotonicity() {
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let result = clamp_simd(&x.view(), 0.0, 2.0);

    for i in 1..x.len() {
        assert!(
            result[i - 1] <= result[i],
            "Clamp should preserve monotonicity"
        );
    }
}

// Edge case: min == max (all values should equal this value)
#[test]
fn test_clamp_min_equals_max() {
    let x = array![-1.0_f64, 0.5, 1.5, 2.5];
    let result = clamp_simd(&x.view(), 1.0, 1.0);

    for &val in result.iter() {
        assert_eq!(val, 1.0);
    }
}

// Range bounds test
#[test]
fn test_clamp_result_in_bounds() {
    let x = array![-10.0_f64, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0];
    let min = -5.0;
    let max = 10.0;
    let result = clamp_simd(&x.view(), min, max);

    for &val in result.iter() {
        assert!(
            val >= min && val <= max,
            "Clamped value {} should be in [{}, {}]",
            val,
            min,
            max
        );
    }
}

// Property: clamp(x, min, max) = max(min, min(x, max))
#[test]
fn test_clamp_mathematical_definition() {
    let x = array![-5.0_f64, 0.0, 2.5, 7.0, 12.0];
    let min = 0.0;
    let max = 10.0;

    let result = clamp_simd(&x.view(), min, max);
    let expected = x.mapv(|val| val.max(min).min(max));

    for i in 0..x.len() {
        assert_eq!(result[i], expected[i]);
    }
}

// NaN handling
#[test]
fn test_clamp_nan_propagation() {
    let x = array![1.0_f64, f64::NAN, 3.0];
    let result = clamp_simd(&x.view(), 0.0, 5.0);

    assert_eq!(result[0], 1.0);
    assert!(result[1].is_nan());
    assert_eq!(result[2], 3.0);
}

// Negative range
#[test]
fn test_clamp_negative_range() {
    let x = array![-10.0_f64, -5.0, 0.0, 5.0];
    let result = clamp_simd(&x.view(), -7.0, -2.0);

    assert_eq!(result[0], -7.0); // Below min
    assert_eq!(result[1], -5.0); // Within
    assert_eq!(result[2], -2.0); // Above max
    assert_eq!(result[3], -2.0); // Above max
}

// Large array (SIMD path)
#[cfg(feature = "random")]
#[test]
fn test_clamp_simd_large_array() {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-100.0, 100.0).unwrap();
    let data: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
    let x = Array1::from(data.clone());

    let min = -50.0;
    let max = 50.0;
    let result = clamp_simd(&x.view(), min, max);

    for i in 0..100.min(data.len()) {
        let expected = data[i].clamp(min, max);
        assert_eq!(result[i], expected);
        assert!(result[i] >= min && result[i] <= max);
    }
}

// Pixel normalization use case (0.0 - 1.0)
#[test]
fn test_clamp_pixel_normalization() {
    let pixels = array![-0.5_f64, 0.0, 0.3, 0.7, 1.0, 1.5];
    let result = clamp_simd(&pixels.view(), 0.0, 1.0);

    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 0.3);
    assert_eq!(result[3], 0.7);
    assert_eq!(result[4], 1.0);
    assert_eq!(result[5], 1.0);
}

// Byte range normalization (0 - 255)
#[test]
fn test_clamp_byte_range() {
    let values = array![-50.0_f32, 0.0, 127.5, 255.0, 300.0];
    let result = clamp_simd(&values.view(), 0.0, 255.0);

    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 127.5);
    assert_eq!(result[3], 255.0);
    assert_eq!(result[4], 255.0);
}

// ============================================================================
// Sign SIMD Tests
// ============================================================================

// Basic correctness f64
#[test]
fn test_sign_simd_f64_basic() {
    let x = array![-3.0, -1.5, 0.0, 1.5, 3.0];
    let result = sign_simd(&x.view());

    assert_eq!(result[0], -1.0);
    assert_eq!(result[1], -1.0);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 1.0);
    assert_eq!(result[4], 1.0);
}

// Basic correctness f32
#[test]
fn test_sign_simd_f32_basic() {
    let x = array![-3.0_f32, -1.5, 0.0, 1.5, 3.0];
    let result = sign_simd(&x.view());

    assert_eq!(result[0], -1.0);
    assert_eq!(result[1], -1.0);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 1.0);
    assert_eq!(result[4], 1.0);
}

// Empty array
#[test]
fn test_sign_simd_empty() {
    let x: Array1<f64> = Array1::zeros(0);
    let result = sign_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// All positive values
#[test]
fn test_sign_simd_all_positive() {
    let x = array![0.1_f64, 1.0, 5.0, 100.0, 1e6];
    let result = sign_simd(&x.view());

    for &val in result.iter() {
        assert_eq!(val, 1.0);
    }
}

// All negative values
#[test]
fn test_sign_simd_all_negative() {
    let x = array![-0.1_f64, -1.0, -5.0, -100.0, -1e6];
    let result = sign_simd(&x.view());

    for &val in result.iter() {
        assert_eq!(val, -1.0);
    }
}

// All zeros
#[test]
fn test_sign_simd_all_zeros() {
    let x = array![0.0_f64, 0.0, 0.0, 0.0, 0.0];
    let result = sign_simd(&x.view());

    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

// Mixed values
#[test]
fn test_sign_simd_mixed() {
    let x = array![-100.0_f64, -0.001, 0.0, 0.001, 100.0];
    let result = sign_simd(&x.view());

    assert_eq!(result[0], -1.0);
    assert_eq!(result[1], -1.0);
    assert_eq!(result[2], 0.0);
    assert_eq!(result[3], 1.0);
    assert_eq!(result[4], 1.0);
}

// Property: sign(-x) = -sign(x) (odd function)
#[test]
fn test_sign_odd_function() {
    let x = array![-5.0_f64, -2.0, -1.0, 1.0, 2.0, 5.0];
    let neg_x = -&x;

    let sign_x = sign_simd(&x.view());
    let sign_neg_x = sign_simd(&neg_x.view());

    for i in 0..x.len() {
        assert_eq!(sign_neg_x[i], -sign_x[i]);
    }
}

// Property: sign(x) * |x| = x
#[test]
fn test_sign_times_abs_equals_x() {
    let x = array![-5.0_f64, -2.0, 0.0, 2.0, 5.0];
    let signs = sign_simd(&x.view());
    let abs_values = x.mapv(|v| v.abs());
    let reconstructed = signs * abs_values;

    for i in 0..x.len() {
        assert!((reconstructed[i] - x[i]).abs() < 1e-10);
    }
}

// Property: sign(x) * sign(y) = sign(x * y)
#[test]
fn test_sign_product_rule() {
    let x = array![-3.0_f64, -1.0, 2.0, 5.0];
    let y = array![2.0_f64, -4.0, -1.0, 3.0];

    let sign_x = sign_simd(&x.view());
    let sign_y = sign_simd(&y.view());
    let product = &x * &y;
    let sign_product = sign_simd(&product.view());

    let sign_x_times_sign_y = &sign_x * &sign_y;

    for i in 0..x.len() {
        assert_eq!(sign_x_times_sign_y[i], sign_product[i]);
    }
}

// Property: |sign(x)| ≤ 1 for all x
#[test]
fn test_sign_bounded() {
    let x = array![-1e10_f64, -100.0, -0.001, 0.0, 0.001, 100.0, 1e10];
    let result = sign_simd(&x.view());

    for &val in result.iter() {
        assert!(val.abs() <= 1.0);
    }
}

// NaN handling (NaN comparisons are false, so sign(NaN) = 0.0)
#[test]
fn test_sign_nan_handling() {
    let x = array![f64::NAN, 1.0, -1.0];
    let result = sign_simd(&x.view());

    assert_eq!(result[0], 0.0); // NaN → 0.0 (neither > 0 nor < 0)
    assert_eq!(result[1], 1.0);
    assert_eq!(result[2], -1.0);
}

// Infinity handling
#[test]
fn test_sign_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY, 0.0];
    let result = sign_simd(&x.view());

    assert_eq!(result[0], 1.0); // +inf → +1
    assert_eq!(result[1], -1.0); // -inf → -1
    assert_eq!(result[2], 0.0); // 0 → 0
}

// Large array SIMD path
#[test]
fn test_sign_large_array() {
    let size = 10_000;
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        let val = (i as f64) - (size as f64 / 2.0);
        data.push(val);
    }
    let x = Array1::from_vec(data);
    let result = sign_simd(&x.view());

    for i in 0..size {
        let expected = if x[i] > 0.0 {
            1.0
        } else if x[i] < 0.0 {
            -1.0
        } else {
            0.0
        };
        assert_eq!(result[i], expected);
    }
}

// Very small values (near zero)
#[test]
fn test_sign_very_small_values() {
    let x = array![-1e-300_f64, -1e-100, 1e-100, 1e-300];
    let result = sign_simd(&x.view());

    assert_eq!(result[0], -1.0); // very small negative → -1
    assert_eq!(result[1], -1.0); // small negative → -1
    assert_eq!(result[2], 1.0); // small positive → +1
    assert_eq!(result[3], 1.0); // very small positive → +1
}

// Application: gradient direction
#[test]
fn test_sign_gradient_direction() {
    let gradients = array![-0.5_f32, -0.1, 0.0, 0.1, 0.5];
    let directions = sign_simd(&gradients.view());

    // In gradient descent, we move opposite to the gradient
    let descent_directions = -directions;

    assert_eq!(descent_directions[0], 1.0); // negative gradient → move right
    assert_eq!(descent_directions[1], 1.0);
    assert_eq!(descent_directions[2], 0.0); // zero gradient → no movement
    assert_eq!(descent_directions[3], -1.0); // positive gradient → move left
    assert_eq!(descent_directions[4], -1.0);
}

// ============================================================================
// Fract SIMD Tests
// ============================================================================

// Basic correctness f64
#[test]
fn test_fract_simd_f64_basic() {
    let x = array![1.5_f64, 2.7, -1.3, 0.0, 3.0];
    let result = fract_simd(&x.view());

    assert!((result[0] - 0.5).abs() < 1e-10); // 1.5.fract() = 0.5
    assert!((result[1] - 0.7).abs() < 1e-10); // 2.7.fract() = 0.7
    assert!((result[2] - (-0.3)).abs() < 1e-10); // -1.3.fract() = -0.3 (signed!)
    assert_eq!(result[3], 0.0); // 0.0.fract() = 0.0
    assert_eq!(result[4], 0.0); // 3.0.fract() = 0.0
}

// Basic correctness f32
#[test]
fn test_fract_simd_f32_basic() {
    let x = array![1.5_f32, 2.7, -1.3, 0.0, 3.0];
    let result = fract_simd(&x.view());

    assert!((result[0] - 0.5_f32).abs() < 1e-6);
    assert!((result[1] - 0.7_f32).abs() < 1e-6);
    assert!((result[2] - (-0.3_f32)).abs() < 1e-6);
    assert_eq!(result[3], 0.0);
    assert_eq!(result[4], 0.0);
}

// Empty array
#[test]
fn test_fract_simd_empty() {
    let x: Array1<f64> = Array1::zeros(0);
    let result = fract_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// All positive integers
#[test]
fn test_fract_simd_positive_integers() {
    let x = array![1.0_f64, 2.0, 3.0, 10.0, 100.0];
    let result = fract_simd(&x.view());

    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

// All negative integers
#[test]
fn test_fract_simd_negative_integers() {
    let x = array![-1.0_f64, -2.0, -3.0, -10.0, -100.0];
    let result = fract_simd(&x.view());

    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

// All positive non-integers
#[test]
fn test_fract_simd_positive_fractions() {
    let x = array![0.1_f64, 0.5, 0.9, 1.1, 1.5];
    let result = fract_simd(&x.view());

    assert!((result[0] - 0.1).abs() < 1e-10);
    assert!((result[1] - 0.5).abs() < 1e-10);
    assert!((result[2] - 0.9).abs() < 1e-10);
    assert!((result[3] - 0.1).abs() < 1e-10);
    assert!((result[4] - 0.5).abs() < 1e-10);
}

// All negative non-integers
#[test]
fn test_fract_simd_negative_fractions() {
    let x = array![-0.1_f64, -0.5, -0.9, -1.1, -1.5];
    let result = fract_simd(&x.view());

    assert!((result[0] - (-0.1)).abs() < 1e-10); // -0.1.fract() = -0.1
    assert!((result[1] - (-0.5)).abs() < 1e-10); // -0.5.fract() = -0.5
    assert!((result[2] - (-0.9)).abs() < 1e-10); // -0.9.fract() = -0.9
    assert!((result[3] - (-0.1)).abs() < 1e-10); // -1.1.fract() = -0.1
    assert!((result[4] - (-0.5)).abs() < 1e-10); // -1.5.fract() = -0.5
}

// Property: fract(x) + trunc(x) = x
#[test]
fn test_fract_plus_trunc_equals_x() {
    let x = array![5.25_f64, -2.75, 0.5, 10.0, -7.3];
    let fract_parts = fract_simd(&x.view());
    let trunc_parts = x.mapv(|v| v.trunc());
    let reconstructed = &fract_parts + &trunc_parts;

    for i in 0..x.len() {
        assert!((reconstructed[i] - x[i]).abs() < 1e-10);
    }
}

// Property: fract(x + n) = fract(x) for integer n (same sign)
#[test]
fn test_fract_periodic_property() {
    let x = array![0.3_f64, 0.7, 0.5, 1.2, 2.9];
    let n = 5.0;

    let fract_x = fract_simd(&x.view());
    let x_plus_n = &x + n;
    let fract_x_plus_n = fract_simd(&x_plus_n.view());

    for i in 0..x.len() {
        assert!((fract_x[i] - fract_x_plus_n[i]).abs() < 1e-10);
    }
}

// Property: -1 < fract(x) < 1 for all finite x
#[test]
fn test_fract_range_property() {
    let x = array![-100.5_f64, -10.3, -0.1, 0.0, 0.1, 10.3, 100.5];
    let result = fract_simd(&x.view());

    for &val in result.iter() {
        assert!(val > -1.0 && val < 1.0);
    }
}

// Property: fract(x) = 0 if and only if x is an integer
#[test]
fn test_fract_zero_iff_integer() {
    let integers = array![-5.0_f64, -1.0, 0.0, 1.0, 5.0];
    let result = fract_simd(&integers.view());

    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

// Property: fract(-x) = -fract(x) (odd function)
#[test]
fn test_fract_odd_function() {
    let x = array![0.3_f64, 0.7, 1.5, 2.9];
    let neg_x = -&x;

    let fract_x = fract_simd(&x.view());
    let fract_neg_x = fract_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!((fract_neg_x[i] + fract_x[i]).abs() < 1e-10);
    }
}

// NaN handling
#[test]
fn test_fract_nan_handling() {
    let x = array![f64::NAN, 1.5, -1.5];
    let result = fract_simd(&x.view());

    assert!(result[0].is_nan());
    assert!((result[1] - 0.5).abs() < 1e-10);
    assert!((result[2] - (-0.5)).abs() < 1e-10);
}

// Infinity handling
#[test]
fn test_fract_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY, 0.0];
    let result = fract_simd(&x.view());

    assert!(result[0].is_nan()); // fract(+inf) = NaN
    assert!(result[1].is_nan()); // fract(-inf) = NaN
    assert_eq!(result[2], 0.0);
}

// Large array SIMD path
#[test]
fn test_fract_large_array() {
    let size = 10_000;
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        let val = ((i as f64) - 5000.0) * 0.137; // Mix of positive and negative
        data.push(val);
    }
    let x = Array1::from_vec(data);
    let result = fract_simd(&x.view());

    for i in 0..size {
        let expected = x[i].fract();
        assert!((result[i] - expected).abs() < 1e-10);
        assert!(result[i] > -1.0 && result[i] < 1.0);
    }
}

// Very large values
#[test]
fn test_fract_very_large_values() {
    let x = array![1e10_f64 + 0.5, -1e10 - 0.5];
    let result = fract_simd(&x.view());

    assert!((result[0] - 0.5).abs() < 1e-6);
    assert!((result[1] - (-0.5)).abs() < 1e-6);
}

// Signed fractional parts
#[test]
fn test_fract_signed_values() {
    let x = array![-2.5_f32, -1.5, -0.5, 0.5, 1.5, 2.5];
    let result = fract_simd(&x.view());

    assert!((result[0] - (-0.5_f32)).abs() < 1e-6);
    assert!((result[1] - (-0.5_f32)).abs() < 1e-6);
    assert!((result[2] - (-0.5_f32)).abs() < 1e-6);
    assert!((result[3] - 0.5_f32).abs() < 1e-6);
    assert!((result[4] - 0.5_f32).abs() < 1e-6);
    assert!((result[5] - 0.5_f32).abs() < 1e-6);
}

// ============================================================================
// recip_simd Tests (Phase 24)
// ============================================================================

// Basic correctness f32
#[test]
fn test_recip_simd_f32_basic() {
    let x = array![2.0_f32, 4.0, 0.5, 1.0, -2.0];
    let result = recip_simd(&x.view());

    assert!((result[0] - 0.5_f32).abs() < 1e-6);
    assert!((result[1] - 0.25_f32).abs() < 1e-6);
    assert!((result[2] - 2.0_f32).abs() < 1e-6);
    assert!((result[3] - 1.0_f32).abs() < 1e-6);
    assert!((result[4] - (-0.5_f32)).abs() < 1e-6);
}

// Basic correctness f64
#[test]
fn test_recip_simd_f64_basic() {
    let x = array![2.0_f64, 4.0, 0.5, 1.0, -2.0];
    let result = recip_simd(&x.view());

    assert!((result[0] - 0.5).abs() < 1e-10);
    assert!((result[1] - 0.25).abs() < 1e-10);
    assert!((result[2] - 2.0).abs() < 1e-10);
    assert!((result[3] - 1.0).abs() < 1e-10);
    assert!((result[4] - (-0.5)).abs() < 1e-10);
}

// Empty array
#[test]
fn test_recip_simd_empty() {
    let x: Array1<f64> = array![];
    let result = recip_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// All positive values
#[test]
fn test_recip_simd_all_positive() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let result = recip_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-10);
    assert!((result[1] - 0.5).abs() < 1e-10);
    assert!((result[2] - (1.0 / 3.0)).abs() < 1e-10);
    assert!((result[3] - 0.25).abs() < 1e-10);
    assert!((result[4] - 0.2).abs() < 1e-10);
}

// All negative values
#[test]
fn test_recip_simd_all_negative() {
    let x = array![-1.0_f64, -2.0, -4.0, -5.0];
    let result = recip_simd(&x.view());

    assert!((result[0] - (-1.0)).abs() < 1e-10);
    assert!((result[1] - (-0.5)).abs() < 1e-10);
    assert!((result[2] - (-0.25)).abs() < 1e-10);
    assert!((result[3] - (-0.2)).abs() < 1e-10);
}

// Mixed positive and negative
#[test]
fn test_recip_simd_mixed() {
    let x = array![-4.0_f64, -2.0, -1.0, 1.0, 2.0, 4.0];
    let result = recip_simd(&x.view());

    assert!((result[0] - (-0.25)).abs() < 1e-10);
    assert!((result[1] - (-0.5)).abs() < 1e-10);
    assert!((result[2] - (-1.0)).abs() < 1e-10);
    assert!((result[3] - 1.0).abs() < 1e-10);
    assert!((result[4] - 0.5).abs() < 1e-10);
    assert!((result[5] - 0.25).abs() < 1e-10);
}

// Property: recip(recip(x)) = x (involutive property)
#[test]
fn test_recip_involutive() {
    let x = array![2.0_f64, 3.0, 5.0, 7.0, 11.0];
    let recip_x = recip_simd(&x.view());
    let recip_recip_x = recip_simd(&recip_x.view());

    for i in 0..x.len() {
        assert!((recip_recip_x[i] - x[i]).abs() < 1e-10);
    }
}

// Property: recip(x * y) = recip(x) * recip(y)
#[test]
fn test_recip_multiplicative() {
    let x = array![2.0_f64, 3.0, 5.0];
    let y = array![4.0_f64, 6.0, 10.0];
    let xy = &x * &y;

    let recip_xy = recip_simd(&xy.view());
    let recip_x = recip_simd(&x.view());
    let recip_y = recip_simd(&y.view());
    let product = &recip_x * &recip_y;

    for i in 0..x.len() {
        assert!((recip_xy[i] - product[i]).abs() < 1e-10);
    }
}

// Property: recip(x/y) = y/x
#[test]
fn test_recip_division_inversion() {
    let x = array![8.0_f64, 12.0, 20.0];
    let y = array![2.0_f64, 3.0, 5.0];
    let x_div_y = &x / &y;

    let recip_result = recip_simd(&x_div_y.view());
    let expected = &y / &x;

    for i in 0..x.len() {
        assert!((recip_result[i] - expected[i]).abs() < 1e-10);
    }
}

// Property: recip(1) = 1 (identity element)
#[test]
fn test_recip_identity() {
    let x = array![1.0_f64, 1.0, 1.0, 1.0];
    let result = recip_simd(&x.view());

    for &val in result.iter() {
        assert!((val - 1.0).abs() < 1e-10);
    }
}

// Property: recip(-x) = -recip(x) (odd function)
#[test]
fn test_recip_odd_function() {
    let x = array![2.0_f64, 3.0, 5.0, 7.0];
    let neg_x = -&x;

    let recip_x = recip_simd(&x.view());
    let recip_neg_x = recip_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!((recip_neg_x[i] + recip_x[i]).abs() < 1e-10);
    }
}

// Property: recip(x^n) = (recip(x))^n
#[test]
fn test_recip_power_property() {
    let x = array![2.0_f64, 3.0, 4.0];
    let n = 3.0;

    let x_pow_n = x.mapv(|v| v.powf(n));
    let recip_x_pow_n = recip_simd(&x_pow_n.view());

    let recip_x = recip_simd(&x.view());
    let recip_x_pow_n_expected = recip_x.mapv(|v| v.powf(n));

    for i in 0..x.len() {
        assert!((recip_x_pow_n[i] - recip_x_pow_n_expected[i]).abs() < 1e-10);
    }
}

// Zero handling (returns infinity)
#[test]
fn test_recip_zero() {
    let x = array![0.0_f64];
    let result = recip_simd(&x.view());
    assert!(result[0].is_infinite() && result[0].is_sign_positive());

    // Test negative zero
    let x = array![-0.0_f64];
    let result = recip_simd(&x.view());
    assert!(result[0].is_infinite() && result[0].is_sign_negative());
}

// NaN propagation
#[test]
fn test_recip_nan_propagation() {
    let x = array![f64::NAN, 2.0, 3.0];
    let result = recip_simd(&x.view());

    assert!(result[0].is_nan());
    assert!((result[1] - 0.5).abs() < 1e-10);
    assert!((result[2] - (1.0 / 3.0)).abs() < 1e-10);
}

// Infinity handling
#[test]
fn test_recip_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY, 2.0];
    let result = recip_simd(&x.view());

    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], -0.0);
    assert!((result[2] - 0.5).abs() < 1e-10);
}

// Large array (triggers SIMD path)
#[test]
fn test_recip_large_array() {
    let size = 10_000;
    let data: Array1<f64> = Array1::from_vec((1..=size).map(|i| i as f64).collect());

    let result = recip_simd(&data.view());

    assert_eq!(result.len(), size);
    // Check a few values
    assert!((result[0] - 1.0).abs() < 1e-10);
    assert!((result[1] - 0.5).abs() < 1e-10);
    assert!((result[99] - 0.01).abs() < 1e-10);
    assert!((result[999] - 0.001).abs() < 1e-10);
}

// Very small values (near zero)
#[test]
fn test_recip_very_small_values() {
    let x = array![1e-10_f64, 1e-20, 1e-30];
    let result = recip_simd(&x.view());

    assert!((result[0] - 1e10).abs() < 1e-10);
    assert!((result[1] - 1e20).abs() < 1e10);
    assert!((result[2] - 1e30).abs() < 1e20);
}

// Very large values
#[test]
fn test_recip_very_large_values() {
    let x = array![1e10_f64, 1e20, 1e30];
    let result = recip_simd(&x.view());

    assert!((result[0] - 1e-10).abs() < 1e-20);
    assert!((result[1] - 1e-20).abs() < 1e-30);
    // Note: 1e30 reciprocal is too small for standard comparison
    assert!(result[2] < 1e-29 && result[2] > 0.0);
}

// Application: Inverse square law (physics)
#[test]
fn test_recip_inverse_square_law() {
    // F = k / r^2 for various distances
    let distances = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let r_squared = distances.mapv(|r| r * r);
    let forces = recip_simd(&r_squared.view()); // Proportional to force

    // Verify inverse square relationship
    assert!((forces[0] - 1.0).abs() < 1e-10); // 1/1^2 = 1
    assert!((forces[1] - 0.25).abs() < 1e-10); // 1/2^2 = 0.25
    assert!((forces[2] - (1.0 / 9.0)).abs() < 1e-10); // 1/3^2 = 1/9
    assert!((forces[3] - 0.0625).abs() < 1e-10); // 1/4^2 = 1/16
    assert!((forces[4] - 0.04).abs() < 1e-10); // 1/5^2 = 1/25
}

// ============================================================================
// powi_simd Tests (Phase 25)
// ============================================================================

// Basic correctness f32 - square
#[test]
fn test_powi_simd_f32_square() {
    let x = array![2.0_f32, 3.0, 4.0, 5.0];
    let result = powi_simd(&x.view(), 2);

    assert!((result[0] - 4.0_f32).abs() < 1e-6);
    assert!((result[1] - 9.0_f32).abs() < 1e-6);
    assert!((result[2] - 16.0_f32).abs() < 1e-6);
    assert!((result[3] - 25.0_f32).abs() < 1e-6);
}

// Basic correctness f64 - cube
#[test]
fn test_powi_simd_f64_cube() {
    let x = array![2.0_f64, 3.0, 4.0, 5.0];
    let result = powi_simd(&x.view(), 3);

    assert!((result[0] - 8.0).abs() < 1e-10);
    assert!((result[1] - 27.0).abs() < 1e-10);
    assert!((result[2] - 64.0).abs() < 1e-10);
    assert!((result[3] - 125.0).abs() < 1e-10);
}

// Empty array
#[test]
fn test_powi_simd_empty() {
    let x: Array1<f64> = array![];
    let result = powi_simd(&x.view(), 2);
    assert_eq!(result.len(), 0);
}

// Exponent zero (should return all ones)
#[test]
fn test_powi_simd_exp_zero() {
    let x = array![2.0_f64, 3.0, 4.0, 5.0, -2.0];
    let result = powi_simd(&x.view(), 0);

    for &val in result.iter() {
        assert!((val - 1.0).abs() < 1e-10);
    }
}

// Exponent one (should return input)
#[test]
fn test_powi_simd_exp_one() {
    let x = array![2.0_f64, 3.0, 4.0, 5.0, -2.0];
    let result = powi_simd(&x.view(), 1);

    for i in 0..x.len() {
        assert!((result[i] - x[i]).abs() < 1e-10);
    }
}

// Negative exponent (reciprocal powers)
#[test]
fn test_powi_simd_negative_exp() {
    let x = array![2.0_f64, 4.0, 5.0];
    let result = powi_simd(&x.view(), -2);

    assert!((result[0] - 0.25).abs() < 1e-10); // 2^(-2) = 0.25
    assert!((result[1] - 0.0625).abs() < 1e-10); // 4^(-2) = 0.0625
    assert!((result[2] - 0.04).abs() < 1e-10); // 5^(-2) = 0.04
}

// Negative bases with even exponent
#[test]
fn test_powi_simd_negative_base_even() {
    let x = array![-2.0_f64, -3.0, -4.0];
    let result = powi_simd(&x.view(), 2);

    assert!((result[0] - 4.0).abs() < 1e-10); // (-2)^2 = 4
    assert!((result[1] - 9.0).abs() < 1e-10); // (-3)^2 = 9
    assert!((result[2] - 16.0).abs() < 1e-10); // (-4)^2 = 16
}

// Negative bases with odd exponent
#[test]
fn test_powi_simd_negative_base_odd() {
    let x = array![-2.0_f64, -3.0, -4.0];
    let result = powi_simd(&x.view(), 3);

    assert!((result[0] - (-8.0)).abs() < 1e-10); // (-2)^3 = -8
    assert!((result[1] - (-27.0)).abs() < 1e-10); // (-3)^3 = -27
    assert!((result[2] - (-64.0)).abs() < 1e-10); // (-4)^3 = -64
}

// Property: powi(x, n+m) = powi(x, n) * powi(x, m)
#[test]
fn test_powi_exponent_addition() {
    let x = array![2.0_f64, 3.0, 5.0];
    let n = 3;
    let m = 2;

    let powi_n_plus_m = powi_simd(&x.view(), n + m);
    let powi_n = powi_simd(&x.view(), n);
    let powi_m = powi_simd(&x.view(), m);
    let product = &powi_n * &powi_m;

    for i in 0..x.len() {
        assert!((powi_n_plus_m[i] - product[i]).abs() < 1e-10);
    }
}

// Property: powi(x*y, n) = powi(x, n) * powi(y, n)
#[test]
fn test_powi_distributive() {
    let x = array![2.0_f64, 3.0, 5.0];
    let y = array![4.0_f64, 2.0, 3.0];
    let n = 3;

    let xy = &x * &y;
    let powi_xy = powi_simd(&xy.view(), n);

    let powi_x = powi_simd(&x.view(), n);
    let powi_y = powi_simd(&y.view(), n);
    let product = &powi_x * &powi_y;

    for i in 0..x.len() {
        assert!((powi_xy[i] - product[i]).abs() < 1e-10);
    }
}

// Property: powi(x, n*m) = powi(powi(x, n), m)
#[test]
fn test_powi_exponent_multiplication() {
    let x = array![2.0_f64, 3.0, 4.0];
    let n = 2;
    let m = 3;

    let powi_nm = powi_simd(&x.view(), n * m);
    let powi_n = powi_simd(&x.view(), n);
    let powi_powi = powi_simd(&powi_n.view(), m);

    for i in 0..x.len() {
        assert!((powi_nm[i] - powi_powi[i]).abs() < 1e-10);
    }
}

// Property: powi(x, -n) = 1 / powi(x, n)
#[test]
fn test_powi_negative_exp_reciprocal() {
    let x = array![2.0_f64, 3.0, 4.0, 5.0];
    let n = 3;

    let powi_neg_n = powi_simd(&x.view(), -n);
    let powi_n = powi_simd(&x.view(), n);
    let recip_powi_n = recip_simd(&powi_n.view());

    for i in 0..x.len() {
        assert!((powi_neg_n[i] - recip_powi_n[i]).abs() < 1e-10);
    }
}

// Zero base with positive exponent
#[test]
fn test_powi_zero_base_positive_exp() {
    let x = array![0.0_f64, 0.0, 0.0];
    let result = powi_simd(&x.view(), 3);

    for &val in result.iter() {
        assert_eq!(val, 0.0);
    }
}

// Zero base with zero exponent (0^0 = 1 convention)
#[test]
fn test_powi_zero_base_zero_exp() {
    let x = array![0.0_f64];
    let result = powi_simd(&x.view(), 0);

    assert_eq!(result[0], 1.0);
}

// NaN propagation
#[test]
fn test_powi_nan_propagation() {
    let x = array![f64::NAN, 2.0, 3.0];
    let result = powi_simd(&x.view(), 2);

    assert!(result[0].is_nan());
    assert!((result[1] - 4.0).abs() < 1e-10);
    assert!((result[2] - 9.0).abs() < 1e-10);
}

// Infinity handling
#[test]
fn test_powi_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY, 2.0];
    let result_pos = powi_simd(&x.view(), 2);
    let result_neg = powi_simd(&x.view(), -2);

    // Positive exponent
    assert!(result_pos[0].is_infinite() && result_pos[0].is_sign_positive());
    assert!(result_pos[1].is_infinite() && result_pos[1].is_sign_positive()); // (-∞)^2 = ∞

    // Negative exponent
    assert_eq!(result_neg[0], 0.0); // ∞^(-2) = 0
    assert_eq!(result_neg[1], 0.0); // (-∞)^(-2) = 0
}

// Large array (triggers SIMD path)
#[test]
fn test_powi_large_array() {
    let size = 10_000;
    let data: Array1<f64> = Array1::from_vec((1..=size).map(|i| i as f64).collect());

    let result = powi_simd(&data.view(), 2);

    assert_eq!(result.len(), size);
    // Check a few values
    assert!((result[0] - 1.0).abs() < 1e-10); // 1^2 = 1
    assert!((result[1] - 4.0).abs() < 1e-10); // 2^2 = 4
    assert!((result[99] - 10000.0).abs() < 1e-10); // 100^2 = 10000
    assert!((result[999] - 1000000.0).abs() < 1e-10); // 1000^2 = 1000000
}

// High exponent
#[test]
fn test_powi_high_exponent() {
    let x = array![2.0_f64, 3.0];
    let result = powi_simd(&x.view(), 10);

    assert!((result[0] - 1024.0).abs() < 1e-10); // 2^10 = 1024
    assert!((result[1] - 59049.0).abs() < 1e-10); // 3^10 = 59049
}

// Fractional bases
#[test]
fn test_powi_fractional_bases() {
    let x = array![0.5_f64, 0.1, 0.25];
    let result = powi_simd(&x.view(), 2);

    assert!((result[0] - 0.25).abs() < 1e-10); // 0.5^2 = 0.25
    assert!((result[1] - 0.01).abs() < 1e-10); // 0.1^2 = 0.01
    assert!((result[2] - 0.0625).abs() < 1e-10); // 0.25^2 = 0.0625
}

// Application: Variance calculation (second moment)
#[test]
fn test_powi_variance_application() {
    // Computing (x - mean)^2 for variance
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let mean = 3.0;
    let deviations = x.mapv(|v| v - mean);
    let squared_deviations = powi_simd(&deviations.view(), 2);

    assert!((squared_deviations[0] - 4.0).abs() < 1e-10); // (1-3)^2 = 4
    assert!((squared_deviations[1] - 1.0).abs() < 1e-10); // (2-3)^2 = 1
    assert!((squared_deviations[2] - 0.0).abs() < 1e-10); // (3-3)^2 = 0
    assert!((squared_deviations[3] - 1.0).abs() < 1e-10); // (4-3)^2 = 1
    assert!((squared_deviations[4] - 4.0).abs() < 1e-10); // (5-3)^2 = 4
}

// ============================================================================
// gamma_simd Tests (Phase 26)
// ============================================================================

// Basic correctness f32
#[test]
fn test_gamma_simd_f32_basic() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let result = gamma_simd(&x.view());

    // Γ(1) = 1, Γ(2) = 1, Γ(3) = 2, Γ(4) = 6, Γ(5) = 24
    // f32 has lower precision, so we use looser tolerances
    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[1] - 1.0).abs() < 1e-5);
    assert!((result[2] - 2.0).abs() < 1e-4);
    assert!((result[3] - 6.0).abs() < 1e-3);
    assert!((result[4] - 24.0).abs() < 0.01);
}

// Basic correctness f64
#[test]
fn test_gamma_simd_f64_basic() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let result = gamma_simd(&x.view());

    // Γ(1) = 1, Γ(2) = 1, Γ(3) = 2, Γ(4) = 6, Γ(5) = 24
    assert!((result[0] - 1.0).abs() < 1e-14);
    assert!((result[1] - 1.0).abs() < 1e-14);
    assert!((result[2] - 2.0).abs() < 1e-14);
    assert!((result[3] - 6.0).abs() < 1e-13);
    assert!((result[4] - 24.0).abs() < 1e-12);
}

// Empty array
#[test]
fn test_gamma_simd_empty() {
    let x: Array1<f64> = array![];
    let result = gamma_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// Half-integer value: Γ(1/2) = √π
#[test]
fn test_gamma_half() {
    let x = array![0.5_f64];
    let result = gamma_simd(&x.view());
    let expected = std::f64::consts::PI.sqrt();
    assert!(
        (result[0] - expected).abs() < 1e-14,
        "Γ(1/2) should be √π, got {}, expected {}",
        result[0],
        expected
    );
}

// Half-integer values: Γ(3/2) = √π/2, Γ(5/2) = 3√π/4
#[test]
fn test_gamma_half_integers() {
    let x = array![0.5_f64, 1.5, 2.5];
    let result = gamma_simd(&x.view());

    let sqrt_pi = std::f64::consts::PI.sqrt();
    assert!((result[0] - sqrt_pi).abs() < 1e-14); // Γ(1/2) = √π
    assert!((result[1] - sqrt_pi / 2.0).abs() < 1e-14); // Γ(3/2) = √π/2
    assert!((result[2] - 3.0 * sqrt_pi / 4.0).abs() < 1e-13); // Γ(5/2) = 3√π/4
}

// Functional equation: Γ(z+1) = z·Γ(z)
#[test]
fn test_gamma_functional_equation() {
    let z = array![2.5_f64, 3.5, 4.5];
    let gamma_z = gamma_simd(&z.view());
    let z_plus_1 = z.mapv(|v| v + 1.0);
    let gamma_z_plus_1 = gamma_simd(&z_plus_1.view());

    for i in 0..z.len() {
        let expected = z[i] * gamma_z[i];
        let relative_error = ((gamma_z_plus_1[i] - expected) / expected).abs();
        assert!(
            relative_error < 1e-14,
            "Functional equation failed: Γ({}) = {}, but {}·Γ({}) = {}",
            z[i] + 1.0,
            gamma_z_plus_1[i],
            z[i],
            z[i],
            expected
        );
    }
}

// Negative values with reflection formula
#[test]
fn test_gamma_negative_values() {
    let x = array![-0.5_f64, -1.5, -2.5];
    let result = gamma_simd(&x.view());

    // These values use the reflection formula: Γ(z)Γ(1-z) = π/sin(πz)
    // For z = -0.5: Γ(-0.5) = -2√π
    let expected_minus_half = -2.0 * std::f64::consts::PI.sqrt();
    assert!((result[0] - expected_minus_half).abs() < 1e-13);

    // All should be finite
    assert!(result[1].is_finite());
    assert!(result[2].is_finite());
}

// Negative integers return infinity (poles)
#[test]
fn test_gamma_negative_integers() {
    let x = array![-1.0_f64, -2.0, -3.0];
    let result = gamma_simd(&x.view());

    // Gamma has poles at negative integers
    // The reflection formula involves sin(πx) which is ~0 for integers
    // Due to numerical precision, we expect very large values or infinity
    for &val in result.iter() {
        assert!(
            val.abs() > 1e10 || val.is_infinite(),
            "Gamma at negative integers should be very large or infinite"
        );
    }
}

// NaN propagation
#[test]
fn test_gamma_nan() {
    let x = array![f64::NAN, 1.0, 2.0];
    let result = gamma_simd(&x.view());

    assert!(result[0].is_nan());
    assert!(result[1].is_finite());
    assert!(result[2].is_finite());
}

// Large positive values
#[test]
fn test_gamma_large_positive() {
    let x = array![10.0_f64, 15.0, 20.0];
    let result = gamma_simd(&x.view());

    // Γ(10) = 9! = 362880
    assert!((result[0] - 362880.0).abs() < 1.0);

    // All should be large and finite (or infinity for very large inputs)
    assert!(result[1] > 1e10);
    assert!(result[2] > 1e15);
}

// Small positive values (< 1)
#[test]
fn test_gamma_small_positive() {
    let x = array![0.1_f64, 0.2, 0.3];
    let result = gamma_simd(&x.view());

    // For small x, Γ(x) ≈ 1/x (dominant term)
    // Γ(0.1) ≈ 9.51, Γ(0.2) ≈ 4.59, Γ(0.3) ≈ 2.99
    assert!((result[0] - 9.51350769866873).abs() < 0.01);
    assert!((result[1] - 4.590843711998803).abs() < 0.01);
    assert!((result[2] - 2.9915689876875904).abs() < 0.01);
}

// Large array (triggers SIMD path)
#[test]
fn test_gamma_large_array() {
    let size = 10_000;
    let data: Array1<f64> =
        Array1::from_vec((0..size).map(|i| 1.0 + (i as f64) * 0.0001).collect());

    let result = gamma_simd(&data.view());

    assert_eq!(result.len(), size);
    // Γ(1) = 1
    assert!((result[0] - 1.0).abs() < 1e-13);
    // Values should increase monotonically for x > 1.46
    for i in 5000..5010 {
        assert!(result[i] < result[i + 1]);
    }
}

// Comparison with known values
#[test]
fn test_gamma_known_values() {
    let x = array![
        1.0_f64, // Γ(1) = 1
        2.0,     // Γ(2) = 1
        6.0,     // Γ(6) = 120
        0.5,     // Γ(1/2) = √π ≈ 1.772453850905516
    ];
    let result = gamma_simd(&x.view());
    let expected = array![1.0, 1.0, 120.0, std::f64::consts::PI.sqrt()];

    for i in 0..x.len() {
        let relative_error = ((result[i] - expected[i]) / expected[i]).abs();
        assert!(
            relative_error < 1e-13,
            "Γ({}) = {}, expected {}, relative error = {}",
            x[i],
            result[i],
            expected[i],
            relative_error
        );
    }
}

// Statistical application: Beta function B(α,β) = Γ(α)Γ(β)/Γ(α+β)
#[test]
fn test_gamma_beta_function() {
    let alpha = 2.0_f64;
    let beta = 3.0_f64;

    let params = array![alpha, beta, alpha + beta];
    let gammas = gamma_simd(&params.view());

    let beta_function = (gammas[0] * gammas[1]) / gammas[2];
    let expected = 1.0 / 12.0; // B(2,3) = Γ(2)Γ(3)/Γ(5) = 1·2/24 = 1/12

    assert!((beta_function - expected).abs() < 1e-14);
}

// Ratio property: Γ(n+1)/Γ(n) = n
#[test]
fn test_gamma_ratio_property() {
    let n = array![5.0_f64, 7.5, 10.0];
    let n_plus_1 = n.mapv(|v| v + 1.0);

    let gamma_n = gamma_simd(&n.view());
    let gamma_n_plus_1 = gamma_simd(&n_plus_1.view());

    for i in 0..n.len() {
        let ratio = gamma_n_plus_1[i] / gamma_n[i];
        let relative_error = ((ratio - n[i]) / n[i]).abs();
        assert!(
            relative_error < 1e-14,
            "Ratio property failed: Γ({})/Γ({}) = {}, expected {}",
            n[i] + 1.0,
            n[i],
            ratio,
            n[i]
        );
    }
}

// Symmetry around 2: Γ(2+x) and Γ(2-x) relationship
#[test]
fn test_gamma_symmetry() {
    let x = array![0.3_f64, 0.5, 0.7];
    let two_plus_x = x.mapv(|v| 2.0 + v);
    let two_minus_x = x.mapv(|v| 2.0 - v);

    let gamma_plus = gamma_simd(&two_plus_x.view());
    let gamma_minus = gamma_simd(&two_minus_x.view());

    // Both should be finite and positive
    for i in 0..x.len() {
        assert!(gamma_plus[i] > 0.0 && gamma_plus[i].is_finite());
        assert!(gamma_minus[i] > 0.0 && gamma_minus[i].is_finite());
    }
}

// Values near 1
#[test]
fn test_gamma_near_one() {
    let x = array![0.9_f64, 0.95, 1.0, 1.05, 1.1];
    let result = gamma_simd(&x.view());

    // Γ(1) = 1, and gamma is continuous, so values near 1 should be close to 1
    assert!((result[2] - 1.0).abs() < 1e-14); // Γ(1) = 1 exactly
    assert!((result[0] - 1.0).abs() < 0.1); // Γ(0.9) ≈ 1.068
    assert!((result[4] - 1.0).abs() < 0.1); // Γ(1.1) ≈ 0.951
}

// Duplication formula test: Γ(2z) related to Γ(z) and Γ(z+1/2)
#[test]
fn test_gamma_duplication_formula() {
    let z = 2.0_f64;
    let params = array![z, z + 0.5, 2.0 * z];
    let gammas = gamma_simd(&params.view());

    // Legendre duplication formula: Γ(2z) = (2^(2z-1))/√π · Γ(z)Γ(z+1/2)
    let two_power = 2.0_f64.powf(2.0 * z - 1.0);
    let sqrt_pi = std::f64::consts::PI.sqrt();
    let expected = (two_power / sqrt_pi) * gammas[0] * gammas[1];

    let relative_error = ((gammas[2] - expected) / expected).abs();
    assert!(
        relative_error < 1e-12,
        "Duplication formula failed: got {}, expected {}",
        gammas[2],
        expected
    );
}

// Monotonicity for x > 1.46 (local minimum at x ≈ 1.46)
#[test]
fn test_gamma_monotonicity() {
    let x = array![2.0_f64, 3.0, 4.0, 5.0, 6.0];
    let result = gamma_simd(&x.view());

    // For x > 1.46, gamma is strictly increasing
    for i in 0..(x.len() - 1) {
        assert!(
            result[i] < result[i + 1],
            "Γ should be monotonically increasing for x > 1.46"
        );
    }
}

// ============================================================================
// Exp2 SIMD Tests (Phase 42)
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::exp2_simd;

#[test]
fn test_exp2_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, 2.0, 3.0, -1.0];
    let result = exp2_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-14);
    assert!((result[1] - 2.0).abs() < 1e-14);
    assert!((result[2] - 4.0).abs() < 1e-14);
    assert!((result[3] - 8.0).abs() < 1e-14);
    assert!((result[4] - 0.5).abs() < 1e-14);
}

#[test]
fn test_exp2_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, 2.0, 3.0, -1.0];
    let result = exp2_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 2.0).abs() < 1e-6);
    assert!((result[2] - 4.0).abs() < 1e-6);
    assert!((result[3] - 8.0).abs() < 1e-6);
    assert!((result[4] - 0.5).abs() < 1e-6);
}

#[test]
fn test_exp2_simd_empty() {
    let x: Array1<f64> = array![];
    let result = exp2_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// ============================================================================
// Cbrt SIMD Tests (Phase 42)
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::cbrt_simd;

#[test]
fn test_cbrt_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, 8.0, 27.0, -8.0, -27.0];
    let result = cbrt_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-14);
    assert!((result[1] - 1.0).abs() < 1e-14);
    assert!((result[2] - 2.0).abs() < 1e-14);
    assert!((result[3] - 3.0).abs() < 1e-14);
    assert!((result[4] - (-2.0)).abs() < 1e-14);
    assert!((result[5] - (-3.0)).abs() < 1e-14);
}

#[test]
fn test_cbrt_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, 8.0, 27.0, -8.0];
    let result = cbrt_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!((result[2] - 2.0).abs() < 1e-6);
    assert!((result[3] - 3.0).abs() < 1e-6);
    assert!((result[4] - (-2.0)).abs() < 1e-6);
}

#[test]
fn test_cbrt_simd_inverse_cube() {
    // cbrt(x^3) should equal x
    let x = array![-5.0_f64, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
    let cubed = x.mapv(|v| v * v * v);
    let result = cbrt_simd(&cubed.view());

    for i in 0..x.len() {
        assert!(
            (result[i] - x[i]).abs() < 1e-14,
            "cbrt(x^3) should equal x at index {}: got {}, expected {}",
            i,
            result[i],
            x[i]
        );
    }
}

// ============================================================================
// Ln_1p SIMD Tests (Phase 42)
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::ln_1p_simd;

#[test]
fn test_ln_1p_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, std::f64::consts::E - 1.0];
    let result = ln_1p_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-14); // ln(1+0) = ln(1) = 0
    assert!((result[1] - 2.0_f64.ln()).abs() < 1e-14); // ln(1+1) = ln(2)
    assert!((result[2] - 1.0).abs() < 1e-14); // ln(e) = 1
}

#[test]
fn test_ln_1p_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, std::f32::consts::E - 1.0];
    let result = ln_1p_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 2.0_f32.ln()).abs() < 1e-6);
    assert!((result[2] - 1.0).abs() < 1e-5);
}

#[test]
fn test_ln_1p_simd_small_values() {
    // Test numerical stability for small values
    let x = array![1e-15_f64, 1e-10, 1e-5];
    let result = ln_1p_simd(&x.view());

    // For small x, ln(1+x) ≈ x
    assert!((result[0] - x[0]).abs() < 1e-28);
    assert!((result[1] - x[1]).abs() < 1e-20);
    assert!((result[2] - x[2]).abs() < 1e-10);
}

// ============================================================================
// Exp_m1 SIMD Tests (Phase 42)
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::exp_m1_simd;

#[test]
fn test_exp_m1_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, 2.0];
    let result = exp_m1_simd(&x.view());

    let e: f64 = std::f64::consts::E;
    assert!((result[0] - 0.0).abs() < 1e-14); // exp(0) - 1 = 0
    assert!((result[1] - (e - 1.0)).abs() < 1e-14); // exp(1) - 1 = e - 1
    assert!((result[2] - (e * e - 1.0)).abs() < 1e-14); // exp(2) - 1 = e^2 - 1
}

#[test]
fn test_exp_m1_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, 2.0];
    let result = exp_m1_simd(&x.view());

    let e: f32 = std::f32::consts::E;
    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - (e - 1.0)).abs() < 1e-6);
    assert!((result[2] - (e * e - 1.0)).abs() < 1e-5);
}

#[test]
fn test_exp_m1_simd_small_values() {
    // Test numerical stability for small values
    let x = array![1e-15_f64, 1e-10, 1e-5];
    let result = exp_m1_simd(&x.view());

    // For small x, exp(x)-1 ≈ x
    assert!((result[0] - x[0]).abs() < 1e-28);
    assert!((result[1] - x[1]).abs() < 1e-20);
    assert!((result[2] - x[2]).abs() < 1e-10);
}

#[test]
fn test_exp_m1_ln_1p_inverse() {
    // exp_m1 and ln_1p are inverse functions
    let x = array![-0.5_f64, 0.0, 0.5, 1.0, 2.0];
    let exp_m1_result = exp_m1_simd(&x.view());
    let back = ln_1p_simd(&exp_m1_result.view());

    for i in 0..x.len() {
        assert!(
            (back[i] - x[i]).abs() < 1e-14,
            "ln_1p(exp_m1(x)) should equal x at index {}: got {}, expected {}",
            i,
            back[i],
            x[i]
        );
    }
}

// ============================================================================
// To_radians SIMD Tests (Phase 42b)
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::to_radians_simd;

#[test]
fn test_to_radians_simd_f64_basic() {
    let degrees = array![0.0_f64, 90.0, 180.0, 270.0, 360.0];
    let result = to_radians_simd(&degrees.view());

    let pi = std::f64::consts::PI;
    assert!((result[0] - 0.0).abs() < 1e-14);
    assert!((result[1] - pi / 2.0).abs() < 1e-14);
    assert!((result[2] - pi).abs() < 1e-14);
    assert!((result[3] - 3.0 * pi / 2.0).abs() < 1e-14);
    assert!((result[4] - 2.0 * pi).abs() < 1e-13);
}

#[test]
fn test_to_radians_simd_f32_basic() {
    let degrees = array![0.0_f32, 90.0, 180.0, 270.0, 360.0];
    let result = to_radians_simd(&degrees.view());

    let pi = std::f32::consts::PI;
    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - pi / 2.0).abs() < 1e-6);
    assert!((result[2] - pi).abs() < 1e-6);
    assert!((result[3] - 3.0 * pi / 2.0).abs() < 1e-5);
    assert!((result[4] - 2.0 * pi).abs() < 1e-5);
}

#[test]
fn test_to_radians_simd_negative() {
    let degrees = array![-90.0_f64, -180.0, -45.0];
    let result = to_radians_simd(&degrees.view());

    let pi = std::f64::consts::PI;
    assert!((result[0] - (-pi / 2.0)).abs() < 1e-14);
    assert!((result[1] - (-pi)).abs() < 1e-14);
    assert!((result[2] - (-pi / 4.0)).abs() < 1e-14);
}

#[test]
fn test_to_radians_simd_empty() {
    let x: Array1<f64> = array![];
    let result = to_radians_simd(&x.view());
    assert_eq!(result.len(), 0);
}

// ============================================================================
// To_degrees SIMD Tests (Phase 42b)
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::to_degrees_simd;

#[test]
fn test_to_degrees_simd_f64_basic() {
    let pi = std::f64::consts::PI;
    let radians = array![0.0_f64, pi / 2.0, pi, 3.0 * pi / 2.0, 2.0 * pi];
    let result = to_degrees_simd(&radians.view());

    assert!((result[0] - 0.0).abs() < 1e-14);
    assert!((result[1] - 90.0).abs() < 1e-13);
    assert!((result[2] - 180.0).abs() < 1e-13);
    assert!((result[3] - 270.0).abs() < 1e-12);
    assert!((result[4] - 360.0).abs() < 1e-12);
}

#[test]
fn test_to_degrees_simd_f32_basic() {
    let pi = std::f32::consts::PI;
    let radians = array![0.0_f32, pi / 2.0, pi, 3.0 * pi / 2.0, 2.0 * pi];
    let result = to_degrees_simd(&radians.view());

    assert!((result[0] - 0.0).abs() < 1e-5);
    assert!((result[1] - 90.0).abs() < 1e-4);
    assert!((result[2] - 180.0).abs() < 1e-4);
    assert!((result[3] - 270.0).abs() < 1e-4);
    assert!((result[4] - 360.0).abs() < 1e-3);
}

#[test]
fn test_to_degrees_simd_negative() {
    let pi = std::f64::consts::PI;
    let radians = array![-pi / 2.0, -pi, -pi / 4.0];
    let result = to_degrees_simd(&radians.view());

    assert!((result[0] - (-90.0)).abs() < 1e-13);
    assert!((result[1] - (-180.0)).abs() < 1e-13);
    assert!((result[2] - (-45.0)).abs() < 1e-13);
}

#[test]
fn test_to_radians_to_degrees_roundtrip() {
    // Converting degrees -> radians -> degrees should be identity
    let degrees = array![0.0_f64, 45.0, 90.0, 135.0, 180.0, 270.0, 360.0];
    let radians = to_radians_simd(&degrees.view());
    let back = to_degrees_simd(&radians.view());

    for i in 0..degrees.len() {
        assert!(
            (back[i] - degrees[i]).abs() < 1e-12,
            "to_degrees(to_radians(x)) should equal x at index {}: got {}, expected {}",
            i,
            back[i],
            degrees[i]
        );
    }
}

#[test]
fn test_to_degrees_to_radians_roundtrip() {
    // Converting radians -> degrees -> radians should be identity
    let pi = std::f64::consts::PI;
    let radians = array![0.0_f64, pi / 4.0, pi / 2.0, pi, 3.0 * pi / 2.0, 2.0 * pi];
    let degrees = to_degrees_simd(&radians.view());
    let back = to_radians_simd(&degrees.view());

    for i in 0..radians.len() {
        assert!(
            (back[i] - radians[i]).abs() < 1e-14,
            "to_radians(to_degrees(x)) should equal x at index {}: got {}, expected {}",
            i,
            back[i],
            radians[i]
        );
    }
}

// ============ Digamma (ψ) Tests ============

use scirs2_core::ndarray_ext::elementwise::digamma_simd;

/// Euler-Mascheroni constant γ ≈ 0.5772156649
const EULER_MASCHERONI: f64 = 0.5772156649015329;

#[test]
fn test_digamma_simd_f64_at_one() {
    // ψ(1) = -γ (Euler-Mascheroni constant)
    let x = array![1.0_f64];
    let result = digamma_simd(&x.view());

    assert!(
        (result[0] - (-EULER_MASCHERONI)).abs() < 1e-10,
        "ψ(1) should equal -γ: got {}, expected {}",
        result[0],
        -EULER_MASCHERONI
    );
}

#[test]
fn test_digamma_simd_f64_positive_integers() {
    // ψ(n) = -γ + H_(n-1) where H_k = 1 + 1/2 + 1/3 + ... + 1/k (harmonic number)
    // ψ(1) = -γ
    // ψ(2) = -γ + 1
    // ψ(3) = -γ + 1 + 1/2
    // ψ(4) = -γ + 1 + 1/2 + 1/3
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let result = digamma_simd(&x.view());

    let expected = [
        -EULER_MASCHERONI,
        -EULER_MASCHERONI + 1.0,
        -EULER_MASCHERONI + 1.0 + 0.5,
        -EULER_MASCHERONI + 1.0 + 0.5 + 1.0 / 3.0,
        -EULER_MASCHERONI + 1.0 + 0.5 + 1.0 / 3.0 + 0.25,
    ];

    for i in 0..x.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-10,
            "ψ({}) should be {}: got {}",
            i + 1,
            expected[i],
            result[i]
        );
    }
}

#[test]
fn test_digamma_simd_f64_recurrence_relation() {
    // ψ(x+1) = ψ(x) + 1/x
    let x_vals = array![1.5_f64, 2.5, 3.5, 4.5];
    let x_plus_one = array![2.5_f64, 3.5, 4.5, 5.5];

    let psi_x = digamma_simd(&x_vals.view());
    let psi_x_plus_one = digamma_simd(&x_plus_one.view());

    for i in 0..x_vals.len() {
        let expected = psi_x[i] + 1.0 / x_vals[i];
        assert!(
            (psi_x_plus_one[i] - expected).abs() < 1e-12,
            "ψ({}) should equal ψ({}) + 1/{}: got {}, expected {}",
            x_plus_one[i],
            x_vals[i],
            x_vals[i],
            psi_x_plus_one[i],
            expected
        );
    }
}

#[test]
fn test_digamma_simd_f64_half_integer() {
    // ψ(1/2) = -γ - 2ln(2) ≈ -1.9635
    let x = array![0.5_f64];
    let result = digamma_simd(&x.view());

    let expected = -EULER_MASCHERONI - 2.0 * std::f64::consts::LN_2;
    assert!(
        (result[0] - expected).abs() < 1e-10,
        "ψ(1/2) should be -γ - 2ln(2): got {}, expected {}",
        result[0],
        expected
    );
}

#[test]
fn test_digamma_simd_f64_large_values() {
    // For large x, ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + ...
    // At x = 100, the approximation should be very accurate
    let x = array![100.0_f64, 1000.0, 10000.0];
    let result = digamma_simd(&x.view());

    for i in 0..x.len() {
        let xi = x[i];
        // Leading term approximation
        let approx = xi.ln() - 0.5 / xi - 1.0 / (12.0 * xi * xi);
        assert!(
            (result[i] - approx).abs() < 1e-6,
            "ψ({}) should be approximately ln(x) - 1/(2x): got {}, expected {}",
            xi,
            result[i],
            approx
        );
    }
}

#[test]
fn test_digamma_simd_f64_nan_and_poles() {
    // ψ at non-positive integers should be NaN (poles)
    let x = array![0.0_f64, -1.0, -2.0, -3.0];
    let result = digamma_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i].is_nan(),
            "ψ({}) should be NaN (pole): got {}",
            x[i],
            result[i]
        );
    }
}

#[test]
fn test_digamma_simd_f64_negative_non_integers() {
    // For negative non-integers, use reflection: ψ(1-x) - ψ(x) = π·cot(πx)
    // This means: ψ(x) = ψ(1-x) - π·cot(πx) for x < 0
    let x = array![-0.5_f64, -1.5, -2.5];
    let result = digamma_simd(&x.view());

    // All results should be finite (not NaN)
    for i in 0..x.len() {
        assert!(
            result[i].is_finite(),
            "ψ({}) should be finite: got {}",
            x[i],
            result[i]
        );
    }

    // ψ(-0.5) using reflection: ψ(1.5) - π·cot(-π/2) = ψ(1.5) - 0
    let psi_1_5 = digamma_simd(&array![1.5_f64].view())[0];
    assert!(
        (result[0] - psi_1_5).abs() < 1e-10,
        "ψ(-0.5) should equal ψ(1.5): got {}, expected {}",
        result[0],
        psi_1_5
    );
}

#[test]
fn test_digamma_simd_f32_basic() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0];
    let result = digamma_simd(&x.view());

    let gamma32 = EULER_MASCHERONI as f32;
    let expected = [
        -gamma32,
        -gamma32 + 1.0,
        -gamma32 + 1.0 + 0.5,
        -gamma32 + 1.0 + 0.5 + 1.0 / 3.0,
    ];

    for i in 0..x.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-5,
            "ψ({}) f32 should be {}: got {}",
            i + 1,
            expected[i],
            result[i]
        );
    }
}

#[test]
fn test_digamma_simd_f32_recurrence() {
    let x_vals = array![2.0_f32, 3.0, 4.0];
    let x_plus_one = array![3.0_f32, 4.0, 5.0];

    let psi_x = digamma_simd(&x_vals.view());
    let psi_x_plus_one = digamma_simd(&x_plus_one.view());

    for i in 0..x_vals.len() {
        let expected = psi_x[i] + 1.0 / x_vals[i];
        assert!(
            (psi_x_plus_one[i] - expected).abs() < 1e-5,
            "ψ({}) f32 should equal ψ({}) + 1/{}: got {}, expected {}",
            x_plus_one[i],
            x_vals[i],
            x_vals[i],
            psi_x_plus_one[i],
            expected
        );
    }
}

#[test]
fn test_digamma_simd_empty() {
    let empty: Array1<f64> = Array1::zeros(0);
    let result = digamma_simd(&empty.view());
    assert!(result.is_empty());
}

#[test]
fn test_digamma_simd_large_array() {
    // Test with a larger array to exercise SIMD paths
    let n = 1000;
    let x: Array1<f64> = Array1::from_iter((1..=n).map(|i| i as f64));
    let result = digamma_simd(&x.view());

    // Verify first and last few values
    assert!((result[0] - (-EULER_MASCHERONI)).abs() < 1e-10);

    // ψ(1000) ≈ ln(1000) - 1/(2*1000) ≈ 6.906
    let approx_last = 1000.0_f64.ln() - 0.5 / 1000.0;
    assert!((result[999] - approx_last).abs() < 1e-3);
}

#[test]
fn test_digamma_simd_infinity() {
    // ψ(+∞) = +∞
    let x = array![f64::INFINITY];
    let result = digamma_simd(&x.view());
    assert!(result[0].is_infinite() && result[0] > 0.0);

    // ψ(-∞) = NaN
    let x_neg = array![f64::NEG_INFINITY];
    let result_neg = digamma_simd(&x_neg.view());
    assert!(result_neg[0].is_nan());
}

#[test]
fn test_digamma_simd_nan_input() {
    let x = array![f64::NAN];
    let result = digamma_simd(&x.view());
    assert!(result[0].is_nan());
}

// ============ Trigamma (ψ') Tests ============

use scirs2_core::ndarray_ext::elementwise::trigamma_simd;

/// π²/6 ≈ 1.6449340668 (Basel problem - ψ'(1))
const PI_SQUARED_OVER_6: f64 = std::f64::consts::PI * std::f64::consts::PI / 6.0;

#[test]
fn test_trigamma_simd_f64_at_one() {
    // ψ'(1) = π²/6 (Basel problem)
    let x = array![1.0_f64];
    let result = trigamma_simd(&x.view());

    assert!(
        (result[0] - PI_SQUARED_OVER_6).abs() < 1e-10,
        "ψ'(1) should equal π²/6: got {}, expected {}",
        result[0],
        PI_SQUARED_OVER_6
    );
}

#[test]
fn test_trigamma_simd_f64_positive_integers() {
    // ψ'(n) = π²/6 - Σ(k=1 to n-1) 1/k² for positive integers
    // ψ'(1) = π²/6
    // ψ'(2) = π²/6 - 1
    // ψ'(3) = π²/6 - 1 - 1/4
    // ψ'(4) = π²/6 - 1 - 1/4 - 1/9
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let result = trigamma_simd(&x.view());

    let expected = [
        PI_SQUARED_OVER_6,
        PI_SQUARED_OVER_6 - 1.0,
        PI_SQUARED_OVER_6 - 1.0 - 0.25,
        PI_SQUARED_OVER_6 - 1.0 - 0.25 - 1.0 / 9.0,
        PI_SQUARED_OVER_6 - 1.0 - 0.25 - 1.0 / 9.0 - 0.0625,
    ];

    for i in 0..x.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-10,
            "ψ'({}) should be {}: got {}",
            i + 1,
            expected[i],
            result[i]
        );
    }
}

#[test]
fn test_trigamma_simd_f64_recurrence_relation() {
    // ψ'(x+1) = ψ'(x) - 1/x²
    let x_vals = array![1.5_f64, 2.5, 3.5, 4.5];
    let x_plus_one = array![2.5_f64, 3.5, 4.5, 5.5];

    let psi1_x = trigamma_simd(&x_vals.view());
    let psi1_x_plus_one = trigamma_simd(&x_plus_one.view());

    for i in 0..x_vals.len() {
        let expected = psi1_x[i] - 1.0 / (x_vals[i] * x_vals[i]);
        assert!(
            (psi1_x_plus_one[i] - expected).abs() < 1e-12,
            "ψ'({}) should equal ψ'({}) - 1/{}²: got {}, expected {}",
            x_plus_one[i],
            x_vals[i],
            x_vals[i],
            psi1_x_plus_one[i],
            expected
        );
    }
}

#[test]
fn test_trigamma_simd_f64_half_integer() {
    // ψ'(1/2) = π²/2 = 3 * π²/6 (from reflection and special value)
    let x = array![0.5_f64];
    let result = trigamma_simd(&x.view());

    // ψ'(1/2) ≈ 4.9348 (π²/2)
    let expected = std::f64::consts::PI * std::f64::consts::PI / 2.0;
    assert!(
        (result[0] - expected).abs() < 1e-9,
        "ψ'(1/2) should be π²/2: got {}, expected {}",
        result[0],
        expected
    );
}

#[test]
fn test_trigamma_simd_f64_large_values() {
    // For large x, ψ'(x) ≈ 1/x + 1/(2x²)
    let x = array![100.0_f64, 1000.0, 10000.0];
    let result = trigamma_simd(&x.view());

    for i in 0..x.len() {
        let xi = x[i];
        // Leading term approximation
        let approx = 1.0 / xi + 1.0 / (2.0 * xi * xi);
        assert!(
            (result[i] - approx).abs() < 1e-6,
            "ψ'({}) should be approximately 1/x + 1/(2x²): got {}, expected {}",
            xi,
            result[i],
            approx
        );
    }
}

#[test]
fn test_trigamma_simd_f64_nan_and_poles() {
    // ψ' at non-positive integers should be NaN (poles)
    let x = array![0.0_f64, -1.0, -2.0, -3.0];
    let result = trigamma_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i].is_nan(),
            "ψ'({}) should be NaN (pole): got {}",
            x[i],
            result[i]
        );
    }
}

#[test]
fn test_trigamma_simd_f64_negative_non_integers() {
    // For negative non-integers, use reflection: ψ'(1-x) + ψ'(x) = π²/sin²(πx)
    let x = array![-0.5_f64, -1.5, -2.5];
    let result = trigamma_simd(&x.view());

    // All results should be finite (not NaN)
    for i in 0..x.len() {
        assert!(
            result[i].is_finite(),
            "ψ'({}) should be finite: got {}",
            x[i],
            result[i]
        );
    }

    // ψ'(-0.5) using reflection: π²/sin²(-π/2) - ψ'(1.5) = π² - ψ'(1.5)
    let psi1_1_5 = trigamma_simd(&array![1.5_f64].view())[0];
    let pi_sq = std::f64::consts::PI * std::f64::consts::PI;
    let sin_neg_half_pi = (-std::f64::consts::PI * 0.5).sin();
    let expected = pi_sq / (sin_neg_half_pi * sin_neg_half_pi) - psi1_1_5;
    assert!(
        (result[0] - expected).abs() < 1e-9,
        "ψ'(-0.5) should follow reflection formula: got {}, expected {}",
        result[0],
        expected
    );
}

#[test]
fn test_trigamma_simd_f32_basic() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0];
    let result = trigamma_simd(&x.view());

    let pi_sq_6_f32 = std::f32::consts::PI * std::f32::consts::PI / 6.0;
    let expected = [
        pi_sq_6_f32,
        pi_sq_6_f32 - 1.0,
        pi_sq_6_f32 - 1.0 - 0.25,
        pi_sq_6_f32 - 1.0 - 0.25 - 1.0 / 9.0,
    ];

    for i in 0..x.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-5,
            "ψ'({}) f32 should be {}: got {}",
            i + 1,
            expected[i],
            result[i]
        );
    }
}

#[test]
fn test_trigamma_simd_f32_recurrence() {
    let x_vals = array![2.0_f32, 3.0, 4.0];
    let x_plus_one = array![3.0_f32, 4.0, 5.0];

    let psi1_x = trigamma_simd(&x_vals.view());
    let psi1_x_plus_one = trigamma_simd(&x_plus_one.view());

    for i in 0..x_vals.len() {
        let expected = psi1_x[i] - 1.0 / (x_vals[i] * x_vals[i]);
        assert!(
            (psi1_x_plus_one[i] - expected).abs() < 1e-5,
            "ψ'({}) f32 should equal ψ'({}) - 1/{}²: got {}, expected {}",
            x_plus_one[i],
            x_vals[i],
            x_vals[i],
            psi1_x_plus_one[i],
            expected
        );
    }
}

#[test]
fn test_trigamma_simd_empty() {
    let empty: Array1<f64> = Array1::zeros(0);
    let result = trigamma_simd(&empty.view());
    assert!(result.is_empty());
}

#[test]
fn test_trigamma_simd_large_array() {
    // Test with a larger array to exercise SIMD paths
    let n = 1000;
    let x: Array1<f64> = Array1::from_iter((1..=n).map(|i| i as f64));
    let result = trigamma_simd(&x.view());

    // Verify first value: ψ'(1) = π²/6
    assert!((result[0] - PI_SQUARED_OVER_6).abs() < 1e-10);

    // ψ'(1000) ≈ 1/1000 + 1/(2*1000²) ≈ 0.001
    let approx_last = 1.0 / 1000.0 + 0.5 / (1000.0 * 1000.0);
    assert!((result[999] - approx_last).abs() < 1e-6);
}

#[test]
fn test_trigamma_simd_infinity() {
    // ψ'(+∞) = 0
    let x = array![f64::INFINITY];
    let result = trigamma_simd(&x.view());
    assert!(
        (result[0] - 0.0).abs() < 1e-10,
        "ψ'(+∞) should be 0: got {}",
        result[0]
    );

    // ψ'(-∞) = NaN
    let x_neg = array![f64::NEG_INFINITY];
    let result_neg = trigamma_simd(&x_neg.view());
    assert!(result_neg[0].is_nan());
}

#[test]
fn test_trigamma_simd_nan_input() {
    let x = array![f64::NAN];
    let result = trigamma_simd(&x.view());
    assert!(result[0].is_nan());
}

#[test]
fn test_trigamma_simd_positivity() {
    // ψ'(x) should always be positive for x > 0
    let x = array![0.1_f64, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0];
    let result = trigamma_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i] > 0.0,
            "ψ'({}) should be positive: got {}",
            x[i],
            result[i]
        );
    }
}

#[test]
fn test_trigamma_simd_monotonically_decreasing() {
    // ψ'(x) is strictly decreasing for x > 0
    let x = array![0.5_f64, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];
    let result = trigamma_simd(&x.view());

    for i in 1..x.len() {
        assert!(
            result[i] < result[i - 1],
            "ψ'({}) should be less than ψ'({}): got {} >= {}",
            x[i],
            x[i - 1],
            result[i],
            result[i - 1]
        );
    }
}

// ============ Log-Gamma (ln Γ) Tests ============

use scirs2_core::ndarray_ext::elementwise::ln_gamma_simd;

/// ln(√π) ≈ 0.5724 - value of ln(Γ(1/2))
const LN_SQRT_PI: f64 = 0.5723649429247001;

#[test]
fn test_ln_gamma_simd_f64_at_one_and_two() {
    // ln(Γ(1)) = ln(0!) = ln(1) = 0
    // ln(Γ(2)) = ln(1!) = ln(1) = 0
    let x = array![1.0_f64, 2.0];
    let result = ln_gamma_simd(&x.view());

    assert!(
        result[0].abs() < 1e-12,
        "ln(Γ(1)) should be 0: got {}",
        result[0]
    );
    assert!(
        result[1].abs() < 1e-12,
        "ln(Γ(2)) should be 0: got {}",
        result[1]
    );
}

#[test]
fn test_ln_gamma_simd_f64_factorials() {
    // ln(Γ(n)) = ln((n-1)!) for positive integers
    // ln(Γ(3)) = ln(2!) = ln(2) ≈ 0.693
    // ln(Γ(4)) = ln(3!) = ln(6) ≈ 1.792
    // ln(Γ(5)) = ln(4!) = ln(24) ≈ 3.178
    // ln(Γ(6)) = ln(5!) = ln(120) ≈ 4.787
    let x = array![3.0_f64, 4.0, 5.0, 6.0];
    let result = ln_gamma_simd(&x.view());

    let expected = [
        2.0_f64.ln(),   // ln(2!)
        6.0_f64.ln(),   // ln(3!)
        24.0_f64.ln(),  // ln(4!)
        120.0_f64.ln(), // ln(5!)
    ];

    for i in 0..x.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-10,
            "ln(Γ({})) should be ln(({}!)): got {}, expected {}",
            x[i] as i32,
            x[i] as i32 - 1,
            result[i],
            expected[i]
        );
    }
}

#[test]
fn test_ln_gamma_simd_f64_half_integer() {
    // ln(Γ(1/2)) = ln(√π) ≈ 0.5724
    let x = array![0.5_f64];
    let result = ln_gamma_simd(&x.view());

    assert!(
        (result[0] - LN_SQRT_PI).abs() < 1e-10,
        "ln(Γ(1/2)) should be ln(√π): got {}, expected {}",
        result[0],
        LN_SQRT_PI
    );
}

#[test]
fn test_ln_gamma_simd_f64_recurrence_relation() {
    // Γ(x+1) = x·Γ(x), so ln(Γ(x+1)) = ln(x) + ln(Γ(x))
    let x_vals = array![1.5_f64, 2.5, 3.5, 4.5];
    let x_plus_one = array![2.5_f64, 3.5, 4.5, 5.5];

    let lng_x = ln_gamma_simd(&x_vals.view());
    let lng_x_plus_one = ln_gamma_simd(&x_plus_one.view());

    for i in 0..x_vals.len() {
        let expected = x_vals[i].ln() + lng_x[i];
        assert!(
            (lng_x_plus_one[i] - expected).abs() < 1e-10,
            "ln(Γ({})) should equal ln({}) + ln(Γ({})): got {}, expected {}",
            x_plus_one[i],
            x_vals[i],
            x_vals[i],
            lng_x_plus_one[i],
            expected
        );
    }
}

#[test]
fn test_ln_gamma_simd_f64_large_values() {
    // For large x: ln(Γ(x)) ≈ (x-0.5)·ln(x) - x + ln(√(2π))
    let ln_sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt().ln();
    let x = array![100.0_f64, 1000.0];
    let result = ln_gamma_simd(&x.view());

    for i in 0..x.len() {
        let xi = x[i];
        // Stirling's approximation (basic form without correction terms)
        let approx = (xi - 0.5) * xi.ln() - xi + ln_sqrt_2pi;
        // For large x, the basic Stirling approximation error is O(1/x)
        // With correction term +1/(12x), error is O(1/x³)
        let approx_with_correction = approx + 1.0 / (12.0 * xi);
        let rel_error = ((result[i] - approx_with_correction) / result[i]).abs();
        assert!(
            rel_error < 1e-5,
            "ln(Γ({})) should follow Stirling's approximation: got {}, approx {}",
            xi,
            result[i],
            approx_with_correction
        );
    }
}

#[test]
fn test_ln_gamma_simd_f64_poles() {
    // ln(Γ) at non-positive integers should be +∞ (Γ has poles)
    let x = array![0.0_f64, -1.0, -2.0, -3.0];
    let result = ln_gamma_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i].is_infinite() && result[i] > 0.0,
            "ln(Γ({})) should be +∞ (pole): got {}",
            x[i],
            result[i]
        );
    }
}

#[test]
fn test_ln_gamma_simd_f64_negative_non_integers() {
    // For negative non-integers, ln(Γ(x)) is finite
    let x = array![-0.5_f64, -1.5, -2.5];
    let result = ln_gamma_simd(&x.view());

    // All results should be finite (not NaN, not ±∞)
    for i in 0..x.len() {
        assert!(
            result[i].is_finite(),
            "ln(Γ({})) should be finite: got {}",
            x[i],
            result[i]
        );
    }

    // Verify using reflection: ln(Γ(x)) + ln(Γ(1-x)) = ln(π/sin(πx))
    // For x = -0.5: ln(Γ(-0.5)) + ln(Γ(1.5)) = ln(π/sin(-π/2)) = ln(π)
    let lng_1_5 = ln_gamma_simd(&array![1.5_f64].view())[0];
    let expected_sum = std::f64::consts::PI.ln(); // ln(π/|sin(-π/2)|) = ln(π/1) = ln(π)
    assert!(
        (result[0] + lng_1_5 - expected_sum).abs() < 1e-9,
        "ln(Γ(-0.5)) + ln(Γ(1.5)) should equal ln(π): got {}, expected {}",
        result[0] + lng_1_5,
        expected_sum
    );
}

#[test]
fn test_ln_gamma_simd_f32_basic() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let result = ln_gamma_simd(&x.view());

    let expected = [
        0.0_f32,       // ln(Γ(1)) = 0
        0.0,           // ln(Γ(2)) = 0
        2.0_f32.ln(),  // ln(2!)
        6.0_f32.ln(),  // ln(3!)
        24.0_f32.ln(), // ln(4!)
    ];

    for i in 0..x.len() {
        assert!(
            (result[i] - expected[i]).abs() < 1e-5,
            "ln(Γ({})) f32 should be {}: got {}",
            x[i] as i32,
            expected[i],
            result[i]
        );
    }
}

#[test]
fn test_ln_gamma_simd_empty() {
    let empty: Array1<f64> = Array1::zeros(0);
    let result = ln_gamma_simd(&empty.view());
    assert!(result.is_empty());
}

#[test]
fn test_ln_gamma_simd_large_array() {
    // Test with a larger array to exercise SIMD paths
    let n = 1000;
    let x: Array1<f64> = Array1::from_iter((1..=n).map(|i| i as f64));
    let result = ln_gamma_simd(&x.view());

    // Verify first few values
    assert!(result[0].abs() < 1e-12); // ln(Γ(1)) = 0
    assert!(result[1].abs() < 1e-12); // ln(Γ(2)) = 0
    assert!((result[2] - 2.0_f64.ln()).abs() < 1e-10); // ln(Γ(3)) = ln(2!)

    // Verify last value using Stirling
    let ln_sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt().ln();
    let x_last = 1000.0_f64;
    let stirling = (x_last - 0.5) * x_last.ln() - x_last + ln_sqrt_2pi;
    assert!((result[999] - stirling).abs() / result[999] < 1e-6);
}

#[test]
fn test_ln_gamma_simd_infinity() {
    // ln(Γ(+∞)) = +∞
    let x = array![f64::INFINITY];
    let result = ln_gamma_simd(&x.view());
    assert!(result[0].is_infinite() && result[0] > 0.0);

    // ln(Γ(-∞)) = NaN
    let x_neg = array![f64::NEG_INFINITY];
    let result_neg = ln_gamma_simd(&x_neg.view());
    assert!(result_neg[0].is_nan());
}

#[test]
fn test_ln_gamma_simd_nan_input() {
    let x = array![f64::NAN];
    let result = ln_gamma_simd(&x.view());
    assert!(result[0].is_nan());
}

#[test]
fn test_ln_gamma_simd_consistency_with_gamma() {
    // For moderate values, ln(Γ(x)) should equal ln(gamma(x))
    // (avoiding overflow range)
    let x = array![2.0_f64, 3.0, 4.0, 5.0, 10.0];
    let ln_gamma_result = ln_gamma_simd(&x.view());

    // Compute gamma and take ln manually
    use scirs2_core::ndarray_ext::elementwise::gamma_simd;
    let gamma_result = gamma_simd(&x.view());

    for i in 0..x.len() {
        let expected = gamma_result[i].ln();
        assert!(
            (ln_gamma_result[i] - expected).abs() < 1e-10,
            "ln_gamma({}) should equal ln(gamma({})): got {}, expected {}",
            x[i],
            x[i],
            ln_gamma_result[i],
            expected
        );
    }
}

#[test]
fn test_ln_gamma_simd_stirling_asymptotic() {
    // Verify Stirling's approximation becomes more accurate for larger x
    // ln(Γ(x)) ≈ (x-0.5)·ln(x) - x + ln(√(2π)) + 1/(12x) - 1/(360x³) + ...
    let ln_sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt().ln();
    let x = array![10.0_f64, 50.0, 100.0, 500.0, 1000.0];
    let result = ln_gamma_simd(&x.view());

    for i in 0..x.len() {
        let xi = x[i];
        // Stirling with first correction term
        let stirling = (xi - 0.5) * xi.ln() - xi + ln_sqrt_2pi + 1.0 / (12.0 * xi);
        let rel_error = ((result[i] - stirling) / result[i]).abs();
        // Error should decrease with increasing x
        assert!(
            rel_error < 1e-4 / (xi / 10.0),
            "ln(Γ({})) Stirling error too large: got {}, Stirling {}",
            xi,
            result[i],
            stirling
        );
    }
}

// =============================================================================
// Phase 45: Error Function Tests
// =============================================================================

use scirs2_core::ndarray_ext::elementwise::{erf_simd, erfc_simd};

/// Test erf(0) = 0
#[test]
fn test_erf_simd_f64_at_zero() {
    let x = array![0.0_f64];
    let result = erf_simd(&x.view());
    assert!(
        result[0].abs() < 1e-15,
        "erf(0) should be 0, got {}",
        result[0]
    );
}

/// Test erf(1) ≈ 0.8427007929497148
#[test]
fn test_erf_simd_f64_at_one() {
    let x = array![1.0_f64];
    let result = erf_simd(&x.view());
    let expected = 0.8427007929497148_f64;
    // A&S approximation has max error ~1.5e-7
    assert!(
        (result[0] - expected).abs() < 2e-6,
        "erf(1) should be {}, got {}",
        expected,
        result[0]
    );
}

/// Test erf is an odd function: erf(-x) = -erf(x)
#[test]
fn test_erf_simd_f64_odd_function() {
    let x_pos = array![0.5_f64, 1.0, 1.5, 2.0, 3.0];
    let x_neg = array![-0.5_f64, -1.0, -1.5, -2.0, -3.0];
    let result_pos = erf_simd(&x_pos.view());
    let result_neg = erf_simd(&x_neg.view());

    for i in 0..5 {
        assert!(
            (result_pos[i] + result_neg[i]).abs() < 1e-14,
            "erf({}) should equal -erf({}): {} vs {}",
            x_pos[i],
            x_neg[i],
            result_pos[i],
            result_neg[i]
        );
    }
}

/// Test erf approaches 1 for large positive x
#[test]
fn test_erf_simd_f64_large_positive() {
    let x = array![3.0_f64, 4.0, 5.0, 6.0];
    let result = erf_simd(&x.view());

    // erf(3) ≈ 0.99998, erf(4+) is very close to 1
    for i in 0..4 {
        assert!(
            (result[i] - 1.0).abs() < 1e-4,
            "erf({}) should be close to 1, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test erf approaches -1 for large negative x
#[test]
fn test_erf_simd_f64_large_negative() {
    let x = array![-3.0_f64, -4.0, -5.0, -6.0];
    let result = erf_simd(&x.view());

    // erf(-3) ≈ -0.99998, erf(-4-) is very close to -1
    for i in 0..4 {
        assert!(
            (result[i] + 1.0).abs() < 1e-4,
            "erf({}) should be close to -1, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test known values of erf
#[test]
fn test_erf_simd_f64_known_values() {
    // Known values from reference tables
    let x = array![0.5_f64, 1.0, 1.5, 2.0];
    let expected = [
        0.5204998778130465_f64, // erf(0.5)
        0.8427007929497148_f64, // erf(1.0)
        0.9661051464753108_f64, // erf(1.5)
        0.9953222650189527_f64, // erf(2.0)
    ];
    let result = erf_simd(&x.view());

    // A&S approximation has max error ~1.5e-7
    for i in 0..4 {
        assert!(
            (result[i] - expected[i]).abs() < 2e-6,
            "erf({}) should be {}, got {}",
            x[i],
            expected[i],
            result[i]
        );
    }
}

/// Test erf for very small x (linear regime)
#[test]
fn test_erf_simd_f64_small_values() {
    // For very small x, erf(x) ≈ 2x/√π
    let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
    let x = array![1e-10_f64, 1e-8, 1e-6];
    let result = erf_simd(&x.view());

    for i in 0..3 {
        let expected = x[i] * two_over_sqrt_pi;
        assert!(
            (result[i] - expected).abs() / expected.abs() < 1e-10,
            "erf({}) should be approximately {}, got {}",
            x[i],
            expected,
            result[i]
        );
    }
}

/// Test erf(infinity) = 1
#[test]
fn test_erf_simd_f64_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY];
    let result = erf_simd(&x.view());

    assert!(
        (result[0] - 1.0).abs() < 1e-10,
        "erf(∞) should be 1, got {}",
        result[0]
    );
    assert!(
        (result[1] + 1.0).abs() < 1e-10,
        "erf(-∞) should be -1, got {}",
        result[1]
    );
}

/// Test erf(NaN) = NaN
#[test]
fn test_erf_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = erf_simd(&x.view());
    assert!(result[0].is_nan(), "erf(NaN) should be NaN");
}

/// Test erf f32 version
#[test]
fn test_erf_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, 2.0, -1.0];
    let result = erf_simd(&x.view());

    assert!(
        result[0].abs() < 1e-7,
        "erf(0) should be 0, got {}",
        result[0]
    );
    assert!(
        (result[1] - 0.8427008_f32).abs() < 1e-5,
        "erf(1) should be ~0.8427, got {}",
        result[1]
    );
    assert!(
        (result[3] + 0.8427008_f32).abs() < 1e-5,
        "erf(-1) should be ~-0.8427, got {}",
        result[3]
    );
}

/// Test empty input for erf
#[test]
fn test_erf_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = erf_simd(&x.view());
    assert!(result.is_empty(), "erf of empty array should be empty");
}

/// Test large array for erf (performance)
#[test]
fn test_erf_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-3.0, 3.0, n);
    let result = erf_simd(&x.view());

    assert_eq!(result.len(), n);
    // Check bounds: -1 <= erf(x) <= 1
    for val in result.iter() {
        assert!(
            *val >= -1.0 - 1e-10 && *val <= 1.0 + 1e-10,
            "erf should be in [-1, 1], got {}",
            val
        );
    }
}

// =============================================================================
// Complementary Error Function (erfc) Tests
// =============================================================================

/// Test erfc(0) = 1
#[test]
fn test_erfc_simd_f64_at_zero() {
    let x = array![0.0_f64];
    let result = erfc_simd(&x.view());
    assert!(
        (result[0] - 1.0).abs() < 1e-15,
        "erfc(0) should be 1, got {}",
        result[0]
    );
}

/// Test erfc(1) ≈ 0.1572992070502852
#[test]
fn test_erfc_simd_f64_at_one() {
    let x = array![1.0_f64];
    let result = erfc_simd(&x.view());
    let expected = 0.1572992070502852_f64;
    // A&S approximation has max error ~1.5e-7
    assert!(
        (result[0] - expected).abs() < 2e-6,
        "erfc(1) should be {}, got {}",
        expected,
        result[0]
    );
}

/// Test erfc(-x) = 2 - erfc(x) (reflection property)
#[test]
fn test_erfc_simd_f64_reflection() {
    let x_pos = array![0.5_f64, 1.0, 1.5, 2.0];
    let x_neg = array![-0.5_f64, -1.0, -1.5, -2.0];
    let result_pos = erfc_simd(&x_pos.view());
    let result_neg = erfc_simd(&x_neg.view());

    for i in 0..4 {
        assert!(
            (result_neg[i] - (2.0 - result_pos[i])).abs() < 1e-14,
            "erfc({}) should equal 2 - erfc({}): {} vs {}",
            x_neg[i],
            x_pos[i],
            result_neg[i],
            2.0 - result_pos[i]
        );
    }
}

/// Test erfc approaches 0 for large positive x
#[test]
fn test_erfc_simd_f64_large_positive() {
    let x = array![3.0_f64, 4.0, 5.0, 6.0];
    let result = erfc_simd(&x.view());

    // erfc(3) ≈ 2.2e-5, erfc(4+) is very close to 0
    for i in 0..4 {
        assert!(
            result[i] < 1e-4,
            "erfc({}) should be close to 0, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test erfc approaches 2 for large negative x
#[test]
fn test_erfc_simd_f64_large_negative() {
    let x = array![-3.0_f64, -4.0, -5.0, -6.0];
    let result = erfc_simd(&x.view());

    // erfc(-3) ≈ 1.99998, erfc(-4-) is very close to 2
    for i in 0..4 {
        assert!(
            (result[i] - 2.0).abs() < 1e-4,
            "erfc({}) should be close to 2, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test consistency: erfc(x) = 1 - erf(x)
#[test]
fn test_erfc_simd_f64_consistency_with_erf() {
    let x = array![0.0_f64, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0];
    let erf_result = erf_simd(&x.view());
    let erfc_result = erfc_simd(&x.view());

    for i in 0..x.len() {
        let expected = 1.0 - erf_result[i];
        assert!(
            (erfc_result[i] - expected).abs() < 1e-14,
            "erfc({}) should equal 1 - erf({}): got {} vs {}",
            x[i],
            x[i],
            erfc_result[i],
            expected
        );
    }
}

/// Test erfc(infinity) = 0, erfc(-infinity) = 2
#[test]
fn test_erfc_simd_f64_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY];
    let result = erfc_simd(&x.view());

    assert!(
        result[0].abs() < 1e-10,
        "erfc(∞) should be 0, got {}",
        result[0]
    );
    assert!(
        (result[1] - 2.0).abs() < 1e-10,
        "erfc(-∞) should be 2, got {}",
        result[1]
    );
}

/// Test erfc(NaN) = NaN
#[test]
fn test_erfc_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = erfc_simd(&x.view());
    assert!(result[0].is_nan(), "erfc(NaN) should be NaN");
}

/// Test erfc f32 version
#[test]
fn test_erfc_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, 2.0];
    let result = erfc_simd(&x.view());

    assert!(
        (result[0] - 1.0).abs() < 1e-7,
        "erfc(0) should be 1, got {}",
        result[0]
    );
    assert!(
        (result[1] - 0.157299_f32).abs() < 1e-5,
        "erfc(1) should be ~0.157, got {}",
        result[1]
    );
}

/// Test empty input for erfc
#[test]
fn test_erfc_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = erfc_simd(&x.view());
    assert!(result.is_empty(), "erfc of empty array should be empty");
}

/// Test normal distribution CDF using erf
/// Φ(x) = 0.5 * (1 + erf(x/√2))
#[test]
fn test_erf_simd_normal_cdf() {
    let sqrt_2 = std::f64::consts::SQRT_2;
    let x = array![0.0_f64, 1.0, -1.0, 2.0, -2.0];
    let x_scaled: Array1<f64> = x.mapv(|v| v / sqrt_2);
    let erf_result = erf_simd(&x_scaled.view());
    let cdf: Array1<f64> = erf_result.mapv(|v| 0.5 * (1.0 + v));

    // Known CDF values for standard normal
    let expected_cdf = [
        0.5_f64,              // Φ(0) = 0.5
        0.8413447460685429,   // Φ(1) ≈ 0.8413
        0.15865525393145707,  // Φ(-1) ≈ 0.1587
        0.9772498680518208,   // Φ(2) ≈ 0.9772
        0.022750131948179195, // Φ(-2) ≈ 0.0228
    ];

    for i in 0..5 {
        assert!(
            (cdf[i] - expected_cdf[i]).abs() < 1e-6,
            "Φ({}) should be {}, got {}",
            x[i],
            expected_cdf[i],
            cdf[i]
        );
    }
}

// =============================================================================
// Phase 46: Inverse Error Function Tests
// =============================================================================

use scirs2_core::ndarray_ext::elementwise::{erfcinv_simd, erfinv_simd};

/// Test erfinv(0) = 0
#[test]
fn test_erfinv_simd_f64_at_zero() {
    let y = array![0.0_f64];
    let result = erfinv_simd(&y.view());
    assert!(
        result[0].abs() < 1e-15,
        "erfinv(0) should be 0, got {}",
        result[0]
    );
}

/// Test erfinv is the inverse of erf: erf(erfinv(y)) = y
#[test]
fn test_erfinv_simd_f64_inverse_property() {
    let y = array![0.0_f64, 0.1, 0.3, 0.5, 0.7, 0.9, -0.3, -0.7];
    let x = erfinv_simd(&y.view());
    let y_back = erf_simd(&x.view());

    for i in 0..y.len() {
        assert!(
            (y[i] - y_back[i]).abs() < 1e-10,
            "erf(erfinv({})) should be {}, got {}",
            y[i],
            y[i],
            y_back[i]
        );
    }
}

/// Test erfinv is odd: erfinv(-y) = -erfinv(y)
#[test]
fn test_erfinv_simd_f64_odd_function() {
    let y_pos = array![0.1_f64, 0.3, 0.5, 0.7, 0.9];
    let y_neg = array![-0.1_f64, -0.3, -0.5, -0.7, -0.9];
    let result_pos = erfinv_simd(&y_pos.view());
    let result_neg = erfinv_simd(&y_neg.view());

    for i in 0..5 {
        assert!(
            (result_pos[i] + result_neg[i]).abs() < 1e-12,
            "erfinv({}) should equal -erfinv({}): {} vs {}",
            y_pos[i],
            y_neg[i],
            result_pos[i],
            result_neg[i]
        );
    }
}

/// Test erfinv at boundaries: erfinv(1) = ∞, erfinv(-1) = -∞
#[test]
fn test_erfinv_simd_f64_boundaries() {
    let y = array![1.0_f64, -1.0];
    let result = erfinv_simd(&y.view());

    assert!(
        result[0].is_infinite() && result[0] > 0.0,
        "erfinv(1) should be ∞, got {}",
        result[0]
    );
    assert!(
        result[1].is_infinite() && result[1] < 0.0,
        "erfinv(-1) should be -∞, got {}",
        result[1]
    );
}

/// Test erfinv outside domain returns NaN
#[test]
fn test_erfinv_simd_f64_outside_domain() {
    let y = array![1.5_f64, -1.5, 2.0, -3.0];
    let result = erfinv_simd(&y.view());

    for i in 0..4 {
        assert!(
            result[i].is_nan(),
            "erfinv({}) should be NaN, got {}",
            y[i],
            result[i]
        );
    }
}

/// Test erfinv(NaN) = NaN
#[test]
fn test_erfinv_simd_f64_nan() {
    let y = array![f64::NAN];
    let result = erfinv_simd(&y.view());
    assert!(result[0].is_nan(), "erfinv(NaN) should be NaN");
}

/// Test erfinv f32 version
#[test]
fn test_erfinv_simd_f32_basic() {
    let y = array![0.0_f32, 0.5, -0.5];
    let result = erfinv_simd(&y.view());

    assert!(
        result[0].abs() < 1e-7,
        "erfinv(0) should be 0, got {}",
        result[0]
    );
    // Verify inverse property for f32
    let y_back = erf_simd(&result.view());
    for i in 0..3 {
        assert!(
            (y[i] - y_back[i]).abs() < 1e-5,
            "erf(erfinv({})) should be {}, got {}",
            y[i],
            y[i],
            y_back[i]
        );
    }
}

/// Test empty input for erfinv
#[test]
fn test_erfinv_simd_empty() {
    let y = array![] as Array1<f64>;
    let result = erfinv_simd(&y.view());
    assert!(result.is_empty(), "erfinv of empty array should be empty");
}

/// Test large array for erfinv (performance)
#[test]
fn test_erfinv_simd_large_array() {
    let n = 10000;
    let y: Array1<f64> = Array1::linspace(-0.99, 0.99, n);
    let result = erfinv_simd(&y.view());

    assert_eq!(result.len(), n);
    // Verify inverse property
    let y_back = erf_simd(&result.view());
    for i in 0..n {
        assert!(
            (y[i] - y_back[i]).abs() < 1e-8,
            "erf(erfinv({})) roundtrip error too large: got {}",
            y[i],
            y_back[i]
        );
    }
}

// =============================================================================
// Inverse Complementary Error Function (erfcinv) Tests
// =============================================================================

/// Test erfcinv(1) = 0
#[test]
fn test_erfcinv_simd_f64_at_one() {
    let y = array![1.0_f64];
    let result = erfcinv_simd(&y.view());
    assert!(
        result[0].abs() < 1e-15,
        "erfcinv(1) should be 0, got {}",
        result[0]
    );
}

/// Test erfcinv is the inverse of erfc: erfc(erfcinv(y)) = y
#[test]
fn test_erfcinv_simd_f64_inverse_property() {
    let y = array![0.1_f64, 0.5, 1.0, 1.5, 1.9];
    let x = erfcinv_simd(&y.view());
    let y_back = erfc_simd(&x.view());

    for i in 0..y.len() {
        assert!(
            (y[i] - y_back[i]).abs() < 1e-10,
            "erfc(erfcinv({})) should be {}, got {}",
            y[i],
            y[i],
            y_back[i]
        );
    }
}

/// Test erfcinv at boundaries: erfcinv(0) = ∞, erfcinv(2) = -∞
#[test]
fn test_erfcinv_simd_f64_boundaries() {
    let y = array![0.0_f64, 2.0];
    let result = erfcinv_simd(&y.view());

    assert!(
        result[0].is_infinite() && result[0] > 0.0,
        "erfcinv(0) should be ∞, got {}",
        result[0]
    );
    assert!(
        result[1].is_infinite() && result[1] < 0.0,
        "erfcinv(2) should be -∞, got {}",
        result[1]
    );
}

/// Test erfcinv outside domain returns NaN
#[test]
fn test_erfcinv_simd_f64_outside_domain() {
    let y = array![-0.5_f64, 2.5, 3.0, -1.0];
    let result = erfcinv_simd(&y.view());

    for i in 0..4 {
        assert!(
            result[i].is_nan(),
            "erfcinv({}) should be NaN, got {}",
            y[i],
            result[i]
        );
    }
}

/// Test erfcinv(NaN) = NaN
#[test]
fn test_erfcinv_simd_f64_nan() {
    let y = array![f64::NAN];
    let result = erfcinv_simd(&y.view());
    assert!(result[0].is_nan(), "erfcinv(NaN) should be NaN");
}

/// Test erfcinv f32 version
#[test]
fn test_erfcinv_simd_f32_basic() {
    let y = array![1.0_f32, 0.5, 1.5];
    let result = erfcinv_simd(&y.view());

    assert!(
        result[0].abs() < 1e-6,
        "erfcinv(1) should be 0, got {}",
        result[0]
    );
    // Verify inverse property for f32
    let y_back = erfc_simd(&result.view());
    for i in 0..3 {
        assert!(
            (y[i] - y_back[i]).abs() < 1e-5,
            "erfc(erfcinv({})) should be {}, got {}",
            y[i],
            y[i],
            y_back[i]
        );
    }
}

/// Test empty input for erfcinv
#[test]
fn test_erfcinv_simd_empty() {
    let y = array![] as Array1<f64>;
    let result = erfcinv_simd(&y.view());
    assert!(result.is_empty(), "erfcinv of empty array should be empty");
}

/// Test consistency: erfcinv(y) = erfinv(1 - y)
#[test]
fn test_erfcinv_simd_f64_consistency_with_erfinv() {
    let y = array![0.5_f64, 1.0, 1.5];
    let erfcinv_result = erfcinv_simd(&y.view());
    let y_minus_one: Array1<f64> = y.mapv(|v| 1.0 - v);
    let erfinv_result = erfinv_simd(&y_minus_one.view());

    for i in 0..y.len() {
        assert!(
            (erfcinv_result[i] - erfinv_result[i]).abs() < 1e-10,
            "erfcinv({}) should equal erfinv({}): got {} vs {}",
            y[i],
            1.0 - y[i],
            erfcinv_result[i],
            erfinv_result[i]
        );
    }
}

/// Test inverse normal CDF (probit function) using erfinv
/// Φ⁻¹(p) = √2 * erfinv(2p - 1)
#[test]
fn test_erfinv_simd_inverse_normal_cdf() {
    let sqrt_2 = std::f64::consts::SQRT_2;

    // Known quantiles of standard normal
    let p = array![0.5_f64, 0.84134, 0.97725]; // Φ(0), Φ(1), Φ(2) approximately
    let expected_x = [0.0_f64, 1.0, 2.0];

    let two_p_minus_one: Array1<f64> = p.mapv(|v| 2.0 * v - 1.0);
    let erfinv_result = erfinv_simd(&two_p_minus_one.view());
    let probit: Array1<f64> = erfinv_result.mapv(|v| sqrt_2 * v);

    for i in 0..3 {
        assert!(
            (probit[i] - expected_x[i]).abs() < 0.01,
            "Φ⁻¹({}) should be approximately {}, got {}",
            p[i],
            expected_x[i],
            probit[i]
        );
    }
}

// =============================================================================
// Sigmoid (Logistic) Function Tests
// =============================================================================

/// Test sigmoid(0) = 0.5
#[test]
fn test_sigmoid_simd_f64_at_zero() {
    let x = array![0.0_f64];
    let result = sigmoid_simd(&x.view());
    assert!(
        (result[0] - 0.5).abs() < 1e-15,
        "sigmoid(0) should be 0.5, got {}",
        result[0]
    );
}

/// Test sigmoid basic values
#[test]
fn test_sigmoid_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0, -2.0];
    let result = sigmoid_simd(&x.view());

    // sigmoid(0) = 0.5
    assert!(
        (result[0] - 0.5).abs() < 1e-15,
        "sigmoid(0) should be 0.5, got {}",
        result[0]
    );

    // sigmoid(1) ≈ 0.7310585786
    let expected_1 = 1.0 / (1.0 + (-1.0_f64).exp());
    assert!(
        (result[1] - expected_1).abs() < 1e-10,
        "sigmoid(1) should be {}, got {}",
        expected_1,
        result[1]
    );

    // sigmoid(-1) = 1 - sigmoid(1)
    assert!(
        (result[2] - (1.0 - expected_1)).abs() < 1e-10,
        "sigmoid(-1) should be {}, got {}",
        1.0 - expected_1,
        result[2]
    );

    // Verify σ(-x) = 1 - σ(x) property
    assert!(
        (result[1] + result[2] - 1.0).abs() < 1e-10,
        "sigmoid(1) + sigmoid(-1) should be 1, got {}",
        result[1] + result[2]
    );
    assert!(
        (result[3] + result[4] - 1.0).abs() < 1e-10,
        "sigmoid(2) + sigmoid(-2) should be 1, got {}",
        result[3] + result[4]
    );
}

/// Test sigmoid range is (0, 1) for moderate values
#[test]
fn test_sigmoid_simd_f64_range() {
    // For moderate values, sigmoid is strictly in (0, 1)
    let x = array![-10.0_f64, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0];
    let result = sigmoid_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i] > 0.0 && result[i] < 1.0,
            "sigmoid({}) should be in (0,1), got {}",
            x[i],
            result[i]
        );
    }

    // For extreme values, check bounds (may round to 0 or 1)
    let x_extreme = array![-100.0_f64, 100.0];
    let result_extreme = sigmoid_simd(&x_extreme.view());
    for i in 0..x_extreme.len() {
        assert!(
            result_extreme[i] >= 0.0 && result_extreme[i] <= 1.0,
            "sigmoid({}) should be in [0,1], got {}",
            x_extreme[i],
            result_extreme[i]
        );
    }
}

/// Test sigmoid symmetry: σ(-x) = 1 - σ(x)
#[test]
fn test_sigmoid_simd_f64_symmetry() {
    let x_pos = array![0.5_f64, 1.0, 2.0, 5.0, 10.0];
    let x_neg = array![-0.5_f64, -1.0, -2.0, -5.0, -10.0];
    let result_pos = sigmoid_simd(&x_pos.view());
    let result_neg = sigmoid_simd(&x_neg.view());

    for i in 0..x_pos.len() {
        assert!(
            (result_pos[i] + result_neg[i] - 1.0).abs() < 1e-10,
            "sigmoid({}) + sigmoid({}) should be 1, got {} + {} = {}",
            x_pos[i],
            x_neg[i],
            result_pos[i],
            result_neg[i],
            result_pos[i] + result_neg[i]
        );
    }
}

/// Test sigmoid numerical stability for large positive values
#[test]
fn test_sigmoid_simd_f64_large_positive() {
    let x = array![50.0_f64, 100.0, 500.0, 700.0];
    let result = sigmoid_simd(&x.view());

    for i in 0..x.len() {
        // Should be very close to 1 but not exactly 1
        assert!(
            result[i] > 0.999,
            "sigmoid({}) should be close to 1, got {}",
            x[i],
            result[i]
        );
        assert!(
            result[i] <= 1.0,
            "sigmoid({}) should be <= 1, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test sigmoid numerical stability for large negative values
#[test]
fn test_sigmoid_simd_f64_large_negative() {
    let x = array![-50.0_f64, -100.0, -500.0, -700.0];
    let result = sigmoid_simd(&x.view());

    for i in 0..x.len() {
        // Should be very close to 0 but not exactly 0
        assert!(
            result[i] < 0.001,
            "sigmoid({}) should be close to 0, got {}",
            x[i],
            result[i]
        );
        assert!(
            result[i] >= 0.0,
            "sigmoid({}) should be >= 0, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test sigmoid with infinity
#[test]
fn test_sigmoid_simd_f64_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY];
    let result = sigmoid_simd(&x.view());

    // sigmoid(+∞) = 1
    assert!(
        (result[0] - 1.0).abs() < 1e-10,
        "sigmoid(+∞) should be 1, got {}",
        result[0]
    );

    // sigmoid(-∞) = 0
    assert!(
        result[1].abs() < 1e-10,
        "sigmoid(-∞) should be 0, got {}",
        result[1]
    );
}

/// Test sigmoid with NaN
#[test]
fn test_sigmoid_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = sigmoid_simd(&x.view());

    assert!(result[0].is_nan(), "sigmoid(NaN) should be NaN");
}

/// Test sigmoid with f32
#[test]
fn test_sigmoid_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, -1.0, 5.0, -5.0];
    let result = sigmoid_simd(&x.view());

    // sigmoid(0) = 0.5
    assert!(
        (result[0] - 0.5).abs() < 1e-6,
        "sigmoid(0) should be 0.5, got {}",
        result[0]
    );

    // Verify symmetry
    assert!(
        (result[1] + result[2] - 1.0).abs() < 1e-5,
        "sigmoid(1) + sigmoid(-1) should be 1, got {}",
        result[1] + result[2]
    );
    assert!(
        (result[3] + result[4] - 1.0).abs() < 1e-5,
        "sigmoid(5) + sigmoid(-5) should be 1, got {}",
        result[3] + result[4]
    );
}

/// Test sigmoid empty array
#[test]
fn test_sigmoid_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = sigmoid_simd(&x.view());
    assert!(result.is_empty(), "sigmoid of empty array should be empty");
}

/// Test sigmoid large array (SIMD path)
#[test]
fn test_sigmoid_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-10.0, 10.0, n);
    let result = sigmoid_simd(&x.view());

    assert_eq!(result.len(), n);

    // Verify monotonically increasing
    for i in 1..n {
        assert!(
            result[i] >= result[i - 1],
            "sigmoid should be monotonically increasing: sigmoid({}) = {} < sigmoid({}) = {}",
            x[i - 1],
            result[i - 1],
            x[i],
            result[i]
        );
    }

    // Verify range (allow boundary values for extreme inputs)
    for i in 0..n {
        assert!(
            result[i] >= 0.0 && result[i] <= 1.0,
            "sigmoid({}) out of range: {}",
            x[i],
            result[i]
        );
    }

    // Verify symmetry using explicitly symmetric values
    let symmetric_x = array![-5.0_f64, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0];
    let symmetric_result = sigmoid_simd(&symmetric_x.view());
    // Check pairs: sigmoid(-x) + sigmoid(x) = 1
    assert!((symmetric_result[0] + symmetric_result[8] - 1.0).abs() < 1e-10); // -5, 5
    assert!((symmetric_result[1] + symmetric_result[7] - 1.0).abs() < 1e-10); // -2, 2
    assert!((symmetric_result[2] + symmetric_result[6] - 1.0).abs() < 1e-10); // -1, 1
    assert!((symmetric_result[3] + symmetric_result[5] - 1.0).abs() < 1e-10); // -0.5, 0.5
    assert!((symmetric_result[4] - 0.5).abs() < 1e-10); // 0
}

/// Test sigmoid derivative property: σ'(x) = σ(x)(1 - σ(x))
/// Using numerical differentiation to verify
#[test]
fn test_sigmoid_simd_derivative_property() {
    let h = 1e-8_f64;
    let x_vals = [0.0_f64, 0.5, 1.0, -1.0, 2.0];

    for &x in &x_vals {
        let x_arr = array![x];
        let x_plus_h = array![x + h];
        let x_minus_h = array![x - h];

        let s = sigmoid_simd(&x_arr.view())[0];
        let s_plus = sigmoid_simd(&x_plus_h.view())[0];
        let s_minus = sigmoid_simd(&x_minus_h.view())[0];

        // Numerical derivative
        let numerical_deriv = (s_plus - s_minus) / (2.0 * h);

        // Analytical derivative: σ(x)(1 - σ(x))
        let analytical_deriv = s * (1.0 - s);

        assert!(
            (numerical_deriv - analytical_deriv).abs() < 1e-5,
            "Derivative at x={}: numerical {} vs analytical {}",
            x,
            numerical_deriv,
            analytical_deriv
        );
    }
}

/// Test logistic regression example: probability = sigmoid(weights · features + bias)
#[test]
fn test_sigmoid_simd_logistic_regression() {
    // Typical logistic regression scenario
    let logits = array![0.0_f64, -2.0, 2.0, -5.0, 5.0];
    let probs = sigmoid_simd(&logits.view());

    // Verify probabilities sum-related properties
    // For balanced classes (logit=0), probability should be 0.5
    assert!((probs[0] - 0.5).abs() < 1e-10);

    // Negative logits -> probability < 0.5
    assert!(probs[1] < 0.5);
    assert!(probs[3] < 0.5);

    // Positive logits -> probability > 0.5
    assert!(probs[2] > 0.5);
    assert!(probs[4] > 0.5);
}

// =============================================================================
// GELU (Gaussian Error Linear Unit) Tests
// =============================================================================

/// Test GELU(0) = 0
#[test]
fn test_gelu_simd_f64_at_zero() {
    let x = array![0.0_f64];
    let result = gelu_simd(&x.view());
    assert!(
        result[0].abs() < 1e-15,
        "GELU(0) should be 0, got {}",
        result[0]
    );
}

/// Test GELU basic values
#[test]
fn test_gelu_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0, -2.0];
    let result = gelu_simd(&x.view());

    // GELU(0) = 0
    assert!(
        result[0].abs() < 1e-15,
        "GELU(0) should be 0, got {}",
        result[0]
    );

    // GELU(1) ≈ 0.8413 (1 * Φ(1) where Φ(1) ≈ 0.8413)
    // More precisely: 1 * 0.5 * (1 + erf(1/√2)) ≈ 0.8413447460685429
    let expected_1 = 0.8413447460685429_f64;
    assert!(
        (result[1] - expected_1).abs() < 1e-6,
        "GELU(1) should be approximately {}, got {}",
        expected_1,
        result[1]
    );

    // GELU(-1) ≈ -0.1587 (-1 * Φ(-1) where Φ(-1) ≈ 0.1587)
    let expected_neg1 = -0.15865525393145707_f64;
    assert!(
        (result[2] - expected_neg1).abs() < 1e-6,
        "GELU(-1) should be approximately {}, got {}",
        expected_neg1,
        result[2]
    );

    // GELU(2) > GELU(1) (monotonically increasing for positive x)
    assert!(
        result[3] > result[1],
        "GELU should be increasing: GELU(2)={} should be > GELU(1)={}",
        result[3],
        result[1]
    );
}

/// Test GELU asymptotic behavior: GELU(x) ≈ x for large positive x
#[test]
fn test_gelu_simd_f64_large_positive() {
    let x = array![5.0_f64, 10.0, 20.0];
    let result = gelu_simd(&x.view());

    for i in 0..x.len() {
        // For large positive x, GELU(x) ≈ x because Φ(x) ≈ 1
        assert!(
            (result[i] - x[i]).abs() < 0.01,
            "GELU({}) should be approximately {}, got {}",
            x[i],
            x[i],
            result[i]
        );
    }
}

/// Test GELU asymptotic behavior: GELU(x) ≈ 0 for large negative x
#[test]
fn test_gelu_simd_f64_large_negative() {
    let x = array![-5.0_f64, -10.0, -20.0];
    let result = gelu_simd(&x.view());

    for i in 0..x.len() {
        // For large negative x, GELU(x) ≈ 0 because Φ(x) ≈ 0
        assert!(
            result[i].abs() < 0.01,
            "GELU({}) should be approximately 0, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test GELU is smooth (no discontinuities)
#[test]
fn test_gelu_simd_f64_smoothness() {
    // Check that GELU is continuous around 0
    let eps = 1e-6_f64;
    let x = array![-eps, 0.0, eps];
    let result = gelu_simd(&x.view());

    // Values should be close to each other (smooth transition)
    assert!(
        (result[1] - result[0]).abs() < 0.001,
        "GELU should be smooth at 0: GELU({})={}, GELU(0)={}",
        -eps,
        result[0],
        result[1]
    );
    assert!(
        (result[2] - result[1]).abs() < 0.001,
        "GELU should be smooth at 0: GELU(0)={}, GELU({})={}",
        result[1],
        eps,
        result[2]
    );
}

/// Test GELU monotonicity: GELU is monotonically increasing for x > 0
#[test]
fn test_gelu_simd_f64_monotonicity() {
    let n = 100;
    let x: Array1<f64> = Array1::linspace(0.0, 10.0, n);
    let result = gelu_simd(&x.view());

    for i in 1..n {
        assert!(
            result[i] >= result[i - 1],
            "GELU should be monotonically increasing for positive x: GELU({})={} < GELU({})={}",
            x[i - 1],
            result[i - 1],
            x[i],
            result[i]
        );
    }
}

/// Test GELU with NaN
#[test]
fn test_gelu_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = gelu_simd(&x.view());

    assert!(result[0].is_nan(), "GELU(NaN) should be NaN");
}

/// Test GELU with f32
#[test]
fn test_gelu_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, -1.0, 2.0, -2.0];
    let result = gelu_simd(&x.view());

    // GELU(0) = 0
    assert!(
        result[0].abs() < 1e-6,
        "GELU(0) should be 0, got {}",
        result[0]
    );

    // GELU(1) > 0
    assert!(result[1] > 0.0, "GELU(1) should be positive");

    // GELU(-1) < 0
    assert!(result[2] < 0.0, "GELU(-1) should be negative");
}

/// Test GELU empty array
#[test]
fn test_gelu_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = gelu_simd(&x.view());
    assert!(result.is_empty(), "GELU of empty array should be empty");
}

/// Test GELU large array (SIMD path)
#[test]
fn test_gelu_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let result = gelu_simd(&x.view());

    assert_eq!(result.len(), n);

    // Verify basic properties
    for i in 0..n {
        // GELU should be between x and 0 for negative x
        if x[i] < 0.0 {
            assert!(
                result[i] >= x[i] && result[i] <= 0.0,
                "For x={}, GELU={} should be between {} and 0",
                x[i],
                result[i],
                x[i]
            );
        }
        // GELU should be between 0 and x for positive x
        if x[i] > 0.0 {
            assert!(
                result[i] >= 0.0 && result[i] <= x[i],
                "For x={}, GELU={} should be between 0 and {}",
                x[i],
                result[i],
                x[i]
            );
        }
    }
}

/// Test GELU vs ReLU relationship: GELU is a smooth approximation of ReLU
#[test]
fn test_gelu_simd_vs_relu() {
    // ReLU(x) = max(0, x)
    // GELU is a smooth version of ReLU
    let x = array![-3.0_f64, -1.0, 0.0, 1.0, 3.0];
    let gelu_result = gelu_simd(&x.view());

    // For large positive x, GELU ≈ x (like ReLU)
    assert!(
        (gelu_result[4] - x[4]).abs() < 0.01,
        "GELU(3) should be close to 3"
    );

    // For large negative x, GELU ≈ 0 (like ReLU)
    assert!(gelu_result[0].abs() < 0.01, "GELU(-3) should be close to 0");

    // At 0, both are 0
    assert!(gelu_result[2].abs() < 1e-10, "GELU(0) should be 0");
}

/// Test GELU in Transformer-like computation
#[test]
fn test_gelu_simd_transformer_use_case() {
    // Simulate feed-forward layer in Transformer: GELU(W*x + b)
    let hidden = array![0.5_f64, 1.2, -0.8, 2.0, -1.5];
    let activated = gelu_simd(&hidden.view());

    // All results should be finite
    for i in 0..hidden.len() {
        assert!(
            activated[i].is_finite(),
            "GELU output should be finite for input {}",
            hidden[i]
        );
    }

    // Positive values should remain positive (scaled down)
    assert!(activated[0] > 0.0);
    assert!(activated[1] > 0.0);
    assert!(activated[3] > 0.0);
}

// ============================================================================
// Swish (SiLU) SIMD Tests
// ============================================================================

/// Test Swish(0) = 0
#[test]
fn test_swish_simd_f64_at_zero() {
    let x = array![0.0_f64];
    let result = swish_simd(&x.view());
    assert!(
        result[0].abs() < 1e-15,
        "Swish(0) should be 0, got {}",
        result[0]
    );
}

/// Test Swish basic values
#[test]
fn test_swish_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0, -2.0];
    let result = swish_simd(&x.view());

    // Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    // Swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    assert!(result[0].abs() < 1e-10, "Swish(0) should be 0");

    // Swish(1) = 1 * sigmoid(1) = 1 / (1 + e^-1) ≈ 0.7310585786
    let expected_swish_1 = 1.0 / (1.0 + (-1.0_f64).exp());
    assert!(
        (result[1] - expected_swish_1).abs() < 1e-10,
        "Swish(1) should be approximately {}, got {}",
        expected_swish_1,
        result[1]
    );

    // Swish(-1) = -1 * sigmoid(-1) = -1 / (1 + e) ≈ -0.2689414214
    let expected_swish_neg1 = -1.0 / (1.0 + 1.0_f64.exp());
    assert!(
        (result[2] - expected_swish_neg1).abs() < 1e-10,
        "Swish(-1) should be approximately {}, got {}",
        expected_swish_neg1,
        result[2]
    );

    // Swish(2) = 2 * sigmoid(2) ≈ 1.7615942
    let expected_swish_2 = 2.0 / (1.0 + (-2.0_f64).exp());
    assert!(
        (result[3] - expected_swish_2).abs() < 1e-10,
        "Swish(2) should be approximately {}, got {}",
        expected_swish_2,
        result[3]
    );

    // Swish(-2) = -2 * sigmoid(-2) ≈ -0.2384058
    let expected_swish_neg2 = -2.0 / (1.0 + 2.0_f64.exp());
    assert!(
        (result[4] - expected_swish_neg2).abs() < 1e-10,
        "Swish(-2) should be approximately {}, got {}",
        expected_swish_neg2,
        result[4]
    );
}

/// Test Swish asymptotic behavior: Swish(x) ≈ x for large positive x
#[test]
fn test_swish_simd_f64_large_positive() {
    let x = array![5.0_f64, 10.0, 20.0];
    let result = swish_simd(&x.view());

    // For large x, sigmoid(x) ≈ 1, so Swish(x) ≈ x
    for i in 0..x.len() {
        // The relative error decreases as x increases
        let relative_error = (result[i] - x[i]).abs() / x[i];
        assert!(
            relative_error < 0.01,
            "Swish({}) ≈ {}, relative error should be small: {}",
            x[i],
            result[i],
            relative_error
        );
    }
}

/// Test Swish asymptotic behavior: Swish(x) ≈ 0 for large negative x
#[test]
fn test_swish_simd_f64_large_negative() {
    let x = array![-5.0_f64, -10.0, -20.0];
    let result = swish_simd(&x.view());

    // For large negative x, sigmoid(x) ≈ 0, so Swish(x) ≈ 0
    for i in 0..x.len() {
        assert!(
            result[i].abs() < 0.1,
            "Swish({}) should be approximately 0, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test Swish is smooth (no discontinuities)
#[test]
fn test_swish_simd_f64_smoothness() {
    // Check that Swish is continuous around 0
    let eps = 1e-6_f64;
    let x = array![-eps, 0.0, eps];
    let result = swish_simd(&x.view());

    // Values should be close to each other (continuous)
    let diff_left = (result[0] - result[1]).abs();
    let diff_right = (result[2] - result[1]).abs();

    assert!(
        diff_left < 1e-5,
        "Swish should be continuous at 0 from the left"
    );
    assert!(
        diff_right < 1e-5,
        "Swish should be continuous at 0 from the right"
    );
}

/// Test Swish non-monotonicity: Swish has a global minimum around x ≈ -1.278
#[test]
fn test_swish_simd_f64_global_minimum() {
    // Swish has a global minimum at approximately x = -1.278
    // Minimum value ≈ -0.2784
    let x = array![-1.278_f64, -1.5, -1.0, -2.0];
    let result = swish_simd(&x.view());

    // The minimum should be around -0.278
    let min_value = result[0];
    assert!(
        min_value < -0.27 && min_value > -0.29,
        "Swish global minimum should be around -0.278, got {}",
        min_value
    );

    // Values at other points should be >= minimum (within numerical tolerance)
    for i in 1..result.len() {
        assert!(
            result[i] >= min_value - 0.01,
            "Swish({}) = {} should be >= minimum {}",
            x[i],
            result[i],
            min_value
        );
    }
}

/// Test Swish with NaN
#[test]
fn test_swish_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = swish_simd(&x.view());

    assert!(result[0].is_nan(), "Swish(NaN) should be NaN");
}

/// Test Swish with f32
#[test]
fn test_swish_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, -1.0, 2.0, -2.0];
    let result = swish_simd(&x.view());

    // Swish(0) = 0
    assert!(result[0].abs() < 1e-6, "Swish(0) should be 0");

    // Swish(1) ≈ 0.7311
    let expected_swish_1 = 1.0_f32 / (1.0 + (-1.0_f32).exp());
    assert!(
        (result[1] - expected_swish_1).abs() < 1e-5,
        "Swish(1) should be approximately {}",
        expected_swish_1
    );

    // Swish(-1) ≈ -0.2689
    let expected_swish_neg1 = -1.0_f32 / (1.0 + 1.0_f32.exp());
    assert!(
        (result[2] - expected_swish_neg1).abs() < 1e-5,
        "Swish(-1) should be approximately {}",
        expected_swish_neg1
    );
}

/// Test Swish empty array
#[test]
fn test_swish_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = swish_simd(&x.view());
    assert!(result.is_empty(), "Swish of empty array should be empty");
}

/// Test Swish large array (SIMD path)
#[test]
fn test_swish_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let result = swish_simd(&x.view());

    // Check length
    assert_eq!(result.len(), n, "Result should have same length as input");

    // Check symmetry property: Swish(-x) = -x * sigmoid(-x)
    // Note: Swish is NOT point-symmetric, but has a specific relationship
    let mid = n / 2;
    for i in 0..100 {
        let pos_idx = mid + i;
        let neg_idx = mid - i;
        if neg_idx > 0 && pos_idx < n {
            let x_pos = x[pos_idx];
            let x_neg = x[neg_idx];
            // Verify both values are computed correctly
            let expected_pos = x_pos / (1.0 + (-x_pos).exp());
            let expected_neg = x_neg / (1.0 + (-x_neg).exp());
            assert!(
                (result[pos_idx] - expected_pos).abs() < 1e-10,
                "Swish({}) should be {}",
                x_pos,
                expected_pos
            );
            assert!(
                (result[neg_idx] - expected_neg).abs() < 1e-10,
                "Swish({}) should be {}",
                x_neg,
                expected_neg
            );
        }
    }
}

/// Test Swish relation to sigmoid: Swish(x) = x * sigmoid(x)
#[test]
fn test_swish_simd_sigmoid_relation() {
    let x = array![-2.0_f64, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let swish_result = swish_simd(&x.view());
    let sigmoid_result = sigmoid_simd(&x.view());

    for i in 0..x.len() {
        let expected = x[i] * sigmoid_result[i];
        assert!(
            (swish_result[i] - expected).abs() < 1e-10,
            "Swish({}) should equal {} * sigmoid({}) = {}",
            x[i],
            x[i],
            x[i],
            expected
        );
    }
}

/// Test Swish in EfficientNet-like computation
#[test]
fn test_swish_simd_efficientnet_use_case() {
    // Simulate activation in EfficientNet: Swish applied after convolution
    let features = array![0.5_f64, 1.2, -0.8, 2.0, -1.5, 0.0, 3.0, -3.0];
    let activated = swish_simd(&features.view());

    // All results should be finite
    for i in 0..features.len() {
        assert!(
            activated[i].is_finite(),
            "Swish output should be finite for input {}",
            features[i]
        );
    }

    // Positive values should produce positive outputs
    assert!(activated[0] > 0.0);
    assert!(activated[1] > 0.0);
    assert!(activated[3] > 0.0);
    assert!(activated[6] > 0.0);

    // Zero should produce zero
    assert!(activated[5].abs() < 1e-10);

    // Negative values should produce small negative or positive outputs
    // (depending on how negative they are)
    for i in 0..features.len() {
        // All outputs should be >= global minimum ≈ -0.278
        assert!(
            activated[i] > -0.3,
            "Swish output should be >= -0.278, got {}",
            activated[i]
        );
    }
}

/// Test Swish vs ReLU: Swish is smoother than ReLU
#[test]
fn test_swish_simd_vs_relu() {
    let x = array![-3.0_f64, -1.0, 0.0, 1.0, 3.0];
    let swish_result = swish_simd(&x.view());

    // ReLU(x) = max(0, x)
    // Unlike ReLU which is 0 for all x < 0, Swish has a small negative region
    // and smoothly transitions to positive

    // For x < 0, Swish is small but not exactly 0
    assert!(swish_result[0] < 0.0 && swish_result[0] > -0.3);
    assert!(swish_result[1] < 0.0 && swish_result[1] > -0.3);

    // At x = 0, both Swish and ReLU are 0
    assert!(swish_result[2].abs() < 1e-10);

    // For x > 0, Swish is close to x (slightly less)
    assert!(swish_result[3] > 0.0 && swish_result[3] < 1.0);
    assert!(swish_result[4] > 0.0 && swish_result[4] < 3.0);
}

/// Test Swish derivative approximation
#[test]
fn test_swish_simd_derivative() {
    // Swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    // = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    // = swish(x) / x + sigmoid(x) * (1 - sigmoid(x)) * x  (for x != 0)

    // Test numerical derivative at a few points
    let eps = 1e-6_f64;
    let test_points = array![0.5_f64, 1.0, 2.0, -0.5, -1.0];

    for i in 0..test_points.len() {
        let x = test_points[i];
        let x_plus = array![x + eps];
        let x_minus = array![x - eps];

        let swish_plus = swish_simd(&x_plus.view())[0];
        let swish_minus = swish_simd(&x_minus.view())[0];

        let numerical_derivative = (swish_plus - swish_minus) / (2.0 * eps);

        // Analytical derivative: σ(x) + x * σ(x) * (1 - σ(x)) = σ(x) * (1 + x * (1 - σ(x)))
        let sigmoid_x = 1.0 / (1.0 + (-x).exp());
        let analytical_derivative = sigmoid_x * (1.0 + x * (1.0 - sigmoid_x));

        assert!(
            (numerical_derivative - analytical_derivative).abs() < 1e-4,
            "Swish derivative at {} should be approximately {}, numerical: {}",
            x,
            analytical_derivative,
            numerical_derivative
        );
    }
}

// ============================================================================
// Softplus SIMD Tests
// ============================================================================

/// Test Softplus(0) = ln(2) ≈ 0.693
#[test]
fn test_softplus_simd_f64_at_zero() {
    let x = array![0.0_f64];
    let result = softplus_simd(&x.view());
    let expected = (2.0_f64).ln(); // ln(1 + exp(0)) = ln(2)
    assert!(
        (result[0] - expected).abs() < 1e-15,
        "Softplus(0) should be ln(2) ≈ {}, got {}",
        expected,
        result[0]
    );
}

/// Test Softplus basic values
#[test]
fn test_softplus_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0, -2.0];
    let result = softplus_simd(&x.view());

    // Softplus(x) = ln(1 + exp(x))
    // Softplus(0) = ln(2)
    let expected_0 = (2.0_f64).ln();
    assert!(
        (result[0] - expected_0).abs() < 1e-10,
        "Softplus(0) should be ln(2)"
    );

    // Softplus(1) = ln(1 + e) ≈ 1.3133
    let expected_1 = (1.0_f64 + 1.0_f64.exp()).ln();
    assert!(
        (result[1] - expected_1).abs() < 1e-10,
        "Softplus(1) should be approximately {}, got {}",
        expected_1,
        result[1]
    );

    // Softplus(-1) = ln(1 + 1/e) ≈ 0.3133
    let expected_neg1 = (1.0_f64 + (-1.0_f64).exp()).ln();
    assert!(
        (result[2] - expected_neg1).abs() < 1e-10,
        "Softplus(-1) should be approximately {}, got {}",
        expected_neg1,
        result[2]
    );

    // Softplus(2) = ln(1 + e²) ≈ 2.1269
    let expected_2 = (1.0_f64 + 2.0_f64.exp()).ln();
    assert!(
        (result[3] - expected_2).abs() < 1e-10,
        "Softplus(2) should be approximately {}, got {}",
        expected_2,
        result[3]
    );

    // Softplus(-2) = ln(1 + e^-2) ≈ 0.1269
    let expected_neg2 = (1.0_f64 + (-2.0_f64).exp()).ln();
    assert!(
        (result[4] - expected_neg2).abs() < 1e-10,
        "Softplus(-2) should be approximately {}, got {}",
        expected_neg2,
        result[4]
    );
}

/// Test Softplus asymptotic behavior: Softplus(x) ≈ x for large positive x
#[test]
fn test_softplus_simd_f64_large_positive() {
    let x = array![10.0_f64, 20.0, 50.0];
    let result = softplus_simd(&x.view());

    // For large x, exp(x) >> 1, so ln(1 + exp(x)) ≈ ln(exp(x)) = x
    for i in 0..x.len() {
        let relative_error = (result[i] - x[i]).abs() / x[i];
        assert!(
            relative_error < 1e-4,
            "Softplus({}) ≈ {}, relative error should be small: {}",
            x[i],
            result[i],
            relative_error
        );
    }
}

/// Test Softplus asymptotic behavior: Softplus(x) ≈ exp(x) for large negative x
#[test]
fn test_softplus_simd_f64_large_negative() {
    let x = array![-10.0_f64, -20.0, -50.0];
    let result = softplus_simd(&x.view());

    // For large negative x, ln(1 + exp(x)) ≈ exp(x) ≈ 0
    for i in 0..x.len() {
        let expected = x[i].exp();
        // Use relative error for small positive values
        let relative_error = if expected > 1e-15 {
            (result[i] - expected).abs() / expected
        } else {
            result[i].abs()
        };
        assert!(
            relative_error < 0.01, // 1% relative error tolerance
            "Softplus({}) should be approximately exp({}) = {}, got {}, relative error: {}",
            x[i],
            x[i],
            expected,
            result[i],
            relative_error
        );
    }
}

/// Test Softplus is always positive
#[test]
fn test_softplus_simd_f64_always_positive() {
    let x = array![-100.0_f64, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0];
    let result = softplus_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            result[i] > 0.0,
            "Softplus({}) should be positive, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test Softplus is monotonically increasing
#[test]
fn test_softplus_simd_f64_monotonicity() {
    let n = 100;
    let x: Array1<f64> = Array1::linspace(-10.0, 10.0, n);
    let result = softplus_simd(&x.view());

    for i in 1..n {
        assert!(
            result[i] > result[i - 1],
            "Softplus should be monotonically increasing: softplus({}) = {} <= softplus({}) = {}",
            x[i - 1],
            result[i - 1],
            x[i],
            result[i]
        );
    }
}

/// Test Softplus is smooth (no discontinuities)
#[test]
fn test_softplus_simd_f64_smoothness() {
    let eps = 1e-6_f64;
    let x = array![-eps, 0.0, eps];
    let result = softplus_simd(&x.view());

    // Values should be close to each other (continuous)
    let diff_left = (result[0] - result[1]).abs();
    let diff_right = (result[2] - result[1]).abs();

    assert!(
        diff_left < 1e-5,
        "Softplus should be continuous at 0 from the left"
    );
    assert!(
        diff_right < 1e-5,
        "Softplus should be continuous at 0 from the right"
    );
}

/// Test Softplus with NaN
#[test]
fn test_softplus_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = softplus_simd(&x.view());

    assert!(result[0].is_nan(), "Softplus(NaN) should be NaN");
}

/// Test Softplus with f32
#[test]
fn test_softplus_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, -1.0, 2.0, -2.0];
    let result = softplus_simd(&x.view());

    // Softplus(0) = ln(2)
    let expected_0 = (2.0_f32).ln();
    assert!(
        (result[0] - expected_0).abs() < 1e-6,
        "Softplus(0) should be ln(2)"
    );

    // Softplus(1) = ln(1 + e)
    let expected_1 = (1.0_f32 + 1.0_f32.exp()).ln();
    assert!(
        (result[1] - expected_1).abs() < 1e-5,
        "Softplus(1) should be approximately {}",
        expected_1
    );

    // Softplus(-1) = ln(1 + 1/e)
    let expected_neg1 = (1.0_f32 + (-1.0_f32).exp()).ln();
    assert!(
        (result[2] - expected_neg1).abs() < 1e-5,
        "Softplus(-1) should be approximately {}",
        expected_neg1
    );
}

/// Test Softplus empty array
#[test]
fn test_softplus_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = softplus_simd(&x.view());
    assert!(result.is_empty(), "Softplus of empty array should be empty");
}

/// Test Softplus large array (SIMD path)
#[test]
fn test_softplus_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-10.0, 10.0, n);
    let result = softplus_simd(&x.view());

    // Check length
    assert_eq!(result.len(), n, "Result should have same length as input");

    // Verify all values are positive
    for i in 0..n {
        assert!(
            result[i] > 0.0,
            "Softplus({}) should be positive, got {}",
            x[i],
            result[i]
        );
    }

    // Verify monotonicity in samples
    for i in (0..n).step_by(100) {
        if i > 0 {
            assert!(
                result[i] >= result[i - 100],
                "Softplus should be monotonically increasing"
            );
        }
    }
}

/// Test Softplus derivative relation: softplus'(x) = sigmoid(x)
#[test]
fn test_softplus_simd_derivative() {
    // softplus'(x) = exp(x) / (1 + exp(x)) = 1 / (1 + exp(-x)) = sigmoid(x)
    let eps = 1e-6_f64;
    let test_points = array![0.0_f64, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0];

    for i in 0..test_points.len() {
        let x = test_points[i];
        let x_plus = array![x + eps];
        let x_minus = array![x - eps];

        let softplus_plus = softplus_simd(&x_plus.view())[0];
        let softplus_minus = softplus_simd(&x_minus.view())[0];

        let numerical_derivative = (softplus_plus - softplus_minus) / (2.0 * eps);

        // Analytical derivative: sigmoid(x)
        let analytical_derivative = 1.0 / (1.0 + (-x).exp());

        assert!(
            (numerical_derivative - analytical_derivative).abs() < 1e-4,
            "Softplus derivative at {} should be sigmoid({}) = {}, numerical: {}",
            x,
            x,
            analytical_derivative,
            numerical_derivative
        );
    }
}

/// Test Softplus in probabilistic model use case
#[test]
fn test_softplus_simd_probabilistic_model() {
    // Softplus is often used to ensure positive parameters (e.g., variance)
    // log_var = raw_output
    // var = softplus(log_var)  (ensures var > 0)
    let raw_outputs = array![-5.0_f64, -2.0, 0.0, 2.0, 5.0];
    let variances = softplus_simd(&raw_outputs.view());

    // All variances should be positive
    for i in 0..raw_outputs.len() {
        assert!(
            variances[i] > 0.0,
            "Variance should be positive, got {}",
            variances[i]
        );
        assert!(
            variances[i].is_finite(),
            "Variance should be finite, got {}",
            variances[i]
        );
    }

    // Larger raw outputs should give larger variances
    for i in 1..raw_outputs.len() {
        assert!(
            variances[i] > variances[i - 1],
            "Variance should increase with raw output"
        );
    }
}

/// Test Softplus vs ReLU approximation
#[test]
fn test_softplus_simd_vs_relu() {
    // Softplus is a smooth approximation of ReLU
    // For large positive x, Softplus(x) ≈ x ≈ ReLU(x)
    // For large negative x, Softplus(x) ≈ 0 ≈ ReLU(x)
    // But Softplus is always > 0, while ReLU(x) = 0 for x < 0
    // Note: Softplus(x) = x + ln(1 + exp(-x)) > x for all x

    let x = array![-3.0_f64, -1.0, 0.0, 1.0, 3.0];
    let softplus_result = softplus_simd(&x.view());

    // At x = 0, Softplus(0) = ln(2) ≠ 0 = ReLU(0)
    assert!(softplus_result[2] > 0.0, "Softplus(0) = ln(2) > 0");

    // For x > 0, Softplus(x) > x (by ln(1 + exp(-x)))
    // but approaches x as x increases
    assert!(softplus_result[3] > x[3], "Softplus(1) > 1");
    assert!(softplus_result[4] > x[4], "Softplus(3) > 3");

    // The difference Softplus(x) - x = ln(1 + exp(-x)) decreases as x increases
    let diff_1 = softplus_result[3] - x[3]; // ln(1 + exp(-1)) ≈ 0.313
    let diff_3 = softplus_result[4] - x[4]; // ln(1 + exp(-3)) ≈ 0.049
    assert!(
        diff_3 < diff_1,
        "Softplus - x should decrease as x increases"
    );

    // For large positive x, Softplus(x) approaches x
    let large_x = array![10.0_f64];
    let large_result = softplus_simd(&large_x.view())[0];
    assert!((large_result - 10.0).abs() < 0.001);
}

// ============================================================================
// Mish SIMD Tests
// ============================================================================

/// Test Mish(0) = 0
#[test]
fn test_mish_simd_f64_at_zero() {
    let x = array![0.0_f64];
    let result = mish_simd(&x.view());
    // Mish(0) = 0 * tanh(softplus(0)) = 0 * tanh(ln(2)) = 0
    assert!(
        result[0].abs() < 1e-15,
        "Mish(0) should be 0, got {}",
        result[0]
    );
}

/// Test Mish basic values
#[test]
fn test_mish_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0, -2.0];
    let result = mish_simd(&x.view());

    // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    // Mish(0) = 0
    assert!(result[0].abs() < 1e-10, "Mish(0) should be 0");

    // Mish(1) = 1 * tanh(ln(1 + e)) ≈ 0.8651
    let softplus_1 = (1.0_f64 + 1.0_f64.exp()).ln();
    let expected_mish_1 = 1.0 * softplus_1.tanh();
    assert!(
        (result[1] - expected_mish_1).abs() < 1e-10,
        "Mish(1) should be approximately {}, got {}",
        expected_mish_1,
        result[1]
    );

    // Mish(-1) = -1 * tanh(ln(1 + 1/e)) ≈ -0.3034
    let softplus_neg1 = (1.0_f64 + (-1.0_f64).exp()).ln();
    let expected_mish_neg1 = -softplus_neg1.tanh();
    assert!(
        (result[2] - expected_mish_neg1).abs() < 1e-10,
        "Mish(-1) should be approximately {}, got {}",
        expected_mish_neg1,
        result[2]
    );

    // Mish(2) = 2 * tanh(ln(1 + e²)) ≈ 1.9439
    let softplus_2 = (1.0_f64 + 2.0_f64.exp()).ln();
    let expected_mish_2 = 2.0 * softplus_2.tanh();
    assert!(
        (result[3] - expected_mish_2).abs() < 1e-10,
        "Mish(2) should be approximately {}, got {}",
        expected_mish_2,
        result[3]
    );
}

/// Test Mish asymptotic behavior: Mish(x) ≈ x for large positive x
#[test]
fn test_mish_simd_f64_large_positive() {
    let x = array![5.0_f64, 10.0, 20.0];
    let result = mish_simd(&x.view());

    // For large x, softplus(x) ≈ x, tanh(x) ≈ 1, so Mish(x) ≈ x * 1 = x
    for i in 0..x.len() {
        let relative_error = (result[i] - x[i]).abs() / x[i];
        assert!(
            relative_error < 0.01,
            "Mish({}) ≈ {}, relative error should be small: {}",
            x[i],
            result[i],
            relative_error
        );
    }
}

/// Test Mish asymptotic behavior: Mish(x) ≈ 0 for large negative x
#[test]
fn test_mish_simd_f64_large_negative() {
    let x = array![-5.0_f64, -10.0, -20.0];
    let result = mish_simd(&x.view());

    // For large negative x, softplus(x) ≈ 0, tanh(0) = 0, so Mish(x) ≈ 0
    for i in 0..x.len() {
        assert!(
            result[i].abs() < 0.1,
            "Mish({}) should be approximately 0, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test Mish has a global minimum around x ≈ -1.2 with value ≈ -0.31
#[test]
fn test_mish_simd_f64_global_minimum() {
    // Mish has a global minimum around x ≈ -1.2
    let x = array![-1.2_f64, -1.5, -1.0, -2.0];
    let result = mish_simd(&x.view());

    // The minimum should be around -0.31
    let min_value = result[0];
    assert!(
        min_value < -0.28 && min_value > -0.35,
        "Mish global minimum should be around -0.31, got {}",
        min_value
    );
}

/// Test Mish is smooth (no discontinuities)
#[test]
fn test_mish_simd_f64_smoothness() {
    let eps = 1e-6_f64;
    let x = array![-eps, 0.0, eps];
    let result = mish_simd(&x.view());

    // Values should be close to each other (continuous)
    let diff_left = (result[0] - result[1]).abs();
    let diff_right = (result[2] - result[1]).abs();

    assert!(
        diff_left < 1e-5,
        "Mish should be continuous at 0 from the left"
    );
    assert!(
        diff_right < 1e-5,
        "Mish should be continuous at 0 from the right"
    );
}

/// Test Mish with NaN
#[test]
fn test_mish_simd_f64_nan() {
    let x = array![f64::NAN];
    let result = mish_simd(&x.view());

    assert!(result[0].is_nan(), "Mish(NaN) should be NaN");
}

/// Test Mish with f32
#[test]
fn test_mish_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, -1.0, 2.0, -2.0];
    let result = mish_simd(&x.view());

    // Mish(0) = 0
    assert!(result[0].abs() < 1e-6, "Mish(0) should be 0");

    // Mish(1) ≈ 0.8651
    let softplus_1 = (1.0_f32 + 1.0_f32.exp()).ln();
    let expected_mish_1 = 1.0_f32 * softplus_1.tanh();
    assert!(
        (result[1] - expected_mish_1).abs() < 1e-5,
        "Mish(1) should be approximately {}",
        expected_mish_1
    );
}

/// Test Mish empty array
#[test]
fn test_mish_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = mish_simd(&x.view());
    assert!(result.is_empty(), "Mish of empty array should be empty");
}

/// Test Mish large array (SIMD path)
#[test]
fn test_mish_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let result = mish_simd(&x.view());

    // Check length
    assert_eq!(result.len(), n, "Result should have same length as input");

    // Verify all values are computed correctly for a sample
    for i in (0..n).step_by(500) {
        let xi = x[i];
        let softplus_xi = (1.0_f64 + xi.exp()).ln();
        let expected = xi * softplus_xi.tanh();
        assert!(
            (result[i] - expected).abs() < 1e-8,
            "Mish({}) should be {}, got {}",
            xi,
            expected,
            result[i]
        );
    }
}

/// Test Mish relation to Swish: both are self-gating activations
#[test]
fn test_mish_simd_vs_swish() {
    // Both Mish and Swish are self-gating: x * f(x)
    // Mish: x * tanh(softplus(x))
    // Swish: x * sigmoid(x)
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let mish_result = mish_simd(&x.view());
    let swish_result = swish_simd(&x.view());

    // Both should be 0 at x = 0
    assert!(mish_result[2].abs() < 1e-10);
    assert!(swish_result[2].abs() < 1e-10);

    // For positive x, Mish > Swish (tanh(softplus(x)) > sigmoid(x) for x > 0)
    assert!(mish_result[3] > swish_result[3]);
    assert!(mish_result[4] > swish_result[4]);

    // Both approach x for large positive x
    let large_x = array![10.0_f64];
    let mish_large = mish_simd(&large_x.view())[0];
    let swish_large = swish_simd(&large_x.view())[0];
    assert!((mish_large - 10.0).abs() < 0.01);
    assert!((swish_large - 10.0).abs() < 0.01);
}

/// Test Mish in YOLOv4-like computation
#[test]
fn test_mish_simd_yolov4_use_case() {
    // Simulate activation in YOLOv4: Mish applied after convolution
    let features = array![0.5_f64, 1.2, -0.8, 2.0, -1.5, 0.0, 3.0, -3.0];
    let activated = mish_simd(&features.view());

    // All results should be finite
    for i in 0..features.len() {
        assert!(
            activated[i].is_finite(),
            "Mish output should be finite for input {}",
            features[i]
        );
    }

    // Zero should produce zero
    assert!(activated[5].abs() < 1e-10);

    // Positive values should produce positive outputs
    assert!(activated[0] > 0.0);
    assert!(activated[1] > 0.0);
    assert!(activated[3] > 0.0);
    assert!(activated[6] > 0.0);

    // All outputs should be >= global minimum ≈ -0.31
    for i in 0..features.len() {
        assert!(
            activated[i] > -0.35,
            "Mish output should be >= -0.31, got {}",
            activated[i]
        );
    }
}

// ============================================================================
// ELU SIMD Tests
// ============================================================================

/// Test ELU with f64 - basic behavior
#[test]
fn test_elu_simd_f64_basic() {
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let alpha = 1.0;
    let result = elu_simd(&x.view(), alpha);

    // ELU(x) = x for x >= 0
    assert!((result[2] - 0.0).abs() < 1e-10, "ELU(0) should be 0");
    assert!((result[3] - 1.0).abs() < 1e-10, "ELU(1) should be 1");
    assert!((result[4] - 2.0).abs() < 1e-10, "ELU(2) should be 2");

    // ELU(x) = α * (exp(x) - 1) for x < 0
    let expected_neg1 = alpha * ((-1.0_f64).exp() - 1.0); // ≈ -0.6321
    let expected_neg2 = alpha * ((-2.0_f64).exp() - 1.0); // ≈ -0.8647
    assert!(
        (result[1] - expected_neg1).abs() < 1e-10,
        "ELU(-1) should be {}, got {}",
        expected_neg1,
        result[1]
    );
    assert!(
        (result[0] - expected_neg2).abs() < 1e-10,
        "ELU(-2) should be {}, got {}",
        expected_neg2,
        result[0]
    );
}

/// Test ELU with f32
#[test]
fn test_elu_simd_f32_basic() {
    let x = array![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
    let alpha = 1.0_f32;
    let result = elu_simd(&x.view(), alpha);

    // ELU(x) = x for x >= 0
    assert!((result[2] - 0.0).abs() < 1e-6, "ELU(0) should be 0");
    assert!((result[3] - 1.0).abs() < 1e-6, "ELU(1) should be 1");

    // ELU(x) = α * (exp(x) - 1) for x < 0
    let expected_neg1 = alpha * ((-1.0_f32).exp() - 1.0);
    assert!(
        (result[1] - expected_neg1).abs() < 1e-5,
        "ELU(-1) should be approximately {}",
        expected_neg1
    );
}

/// Test ELU empty array
#[test]
fn test_elu_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = elu_simd(&x.view(), 1.0);
    assert!(result.is_empty(), "ELU of empty array should be empty");
}

/// Test ELU with different alpha values
#[test]
fn test_elu_simd_different_alpha() {
    let x = array![-1.0_f64, -0.5, 0.5, 1.0];

    // Alpha = 0.5
    let alpha_half = 0.5;
    let result_half = elu_simd(&x.view(), alpha_half);
    let expected_neg1_half = alpha_half * ((-1.0_f64).exp() - 1.0);
    assert!(
        (result_half[0] - expected_neg1_half).abs() < 1e-10,
        "ELU(-1, α=0.5) should be {}, got {}",
        expected_neg1_half,
        result_half[0]
    );

    // Alpha = 2.0
    let alpha_double = 2.0;
    let result_double = elu_simd(&x.view(), alpha_double);
    let expected_neg1_double = alpha_double * ((-1.0_f64).exp() - 1.0);
    assert!(
        (result_double[0] - expected_neg1_double).abs() < 1e-10,
        "ELU(-1, α=2.0) should be {}, got {}",
        expected_neg1_double,
        result_double[0]
    );

    // Positive values should be unchanged regardless of alpha
    assert!((result_half[2] - 0.5).abs() < 1e-10);
    assert!((result_double[2] - 0.5).abs() < 1e-10);
    assert!((result_half[3] - 1.0).abs() < 1e-10);
    assert!((result_double[3] - 1.0).abs() < 1e-10);
}

/// Test ELU large array (SIMD path)
#[test]
fn test_elu_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let alpha = 1.0;
    let result = elu_simd(&x.view(), alpha);

    // Check length
    assert_eq!(result.len(), n, "Result should have same length as input");

    // Verify all values are computed correctly for a sample
    for i in (0..n).step_by(500) {
        let xi = x[i];
        let expected = if xi >= 0.0 {
            xi
        } else {
            alpha * (xi.exp() - 1.0)
        };
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "ELU({}) should be {}, got {}",
            xi,
            expected,
            result[i]
        );
    }
}

/// Test ELU NaN handling
#[test]
fn test_elu_simd_nan() {
    let x = array![f64::NAN, 1.0, f64::NAN, -1.0];
    let result = elu_simd(&x.view(), 1.0);

    assert!(result[0].is_nan(), "ELU(NaN) should be NaN");
    assert!(result[2].is_nan(), "ELU(NaN) should be NaN");
    assert!((result[1] - 1.0).abs() < 1e-10);
}

/// Test ELU infinity handling
#[test]
fn test_elu_simd_infinity() {
    let x = array![f64::INFINITY, f64::NEG_INFINITY, 0.0];
    let result = elu_simd(&x.view(), 1.0);

    // ELU(+inf) = +inf
    assert!(
        result[0].is_infinite() && result[0] > 0.0,
        "ELU(+inf) should be +inf"
    );
    // ELU(-inf) = α * (exp(-inf) - 1) = α * (0 - 1) = -α
    assert!(
        (result[1] - (-1.0)).abs() < 1e-10,
        "ELU(-inf) should be -α = -1.0, got {}",
        result[1]
    );
}

/// Test ELU vs ReLU: ELU is smoother and has non-zero gradient for x < 0
#[test]
fn test_elu_simd_vs_relu() {
    let x = array![-2.0_f64, -1.0, -0.5, 0.0, 0.5, 1.0];
    let alpha = 1.0;
    let elu_result = elu_simd(&x.view(), alpha);

    // For x >= 0, ELU(x) = x (same as ReLU)
    assert!((elu_result[3] - 0.0).abs() < 1e-10); // x = 0
    assert!((elu_result[4] - 0.5).abs() < 1e-10); // x = 0.5
    assert!((elu_result[5] - 1.0).abs() < 1e-10); // x = 1

    // For x < 0, ELU(x) = α * (exp(x) - 1) ∈ (-α, 0)
    // ReLU(x) = 0 for x < 0
    // ELU is strictly greater than -α for finite x
    assert!(elu_result[0] > -alpha, "ELU(-2) should be > -α");
    assert!(elu_result[1] > -alpha, "ELU(-1) should be > -α");
    assert!(elu_result[2] > -alpha, "ELU(-0.5) should be > -α");

    // ELU is negative for x < 0
    assert!(elu_result[0] < 0.0, "ELU(-2) should be < 0");
    assert!(elu_result[1] < 0.0, "ELU(-1) should be < 0");
    assert!(elu_result[2] < 0.0, "ELU(-0.5) should be < 0");
}

/// Test ELU continuity at x = 0
#[test]
fn test_elu_simd_continuity_at_zero() {
    // Test that ELU approaches 0 from both sides
    let alpha = 1.0_f64;
    let eps = 1e-8;
    let x = array![-eps, 0.0, eps];
    let result = elu_simd(&x.view(), alpha);

    // All values should be close to 0
    assert!(
        result[0].abs() < 1e-7,
        "ELU(-eps) should be close to 0, got {}",
        result[0]
    );
    assert!(
        result[1].abs() < 1e-10,
        "ELU(0) should be 0, got {}",
        result[1]
    );
    assert!(
        result[2].abs() < 1e-7,
        "ELU(eps) should be close to 0, got {}",
        result[2]
    );

    // Check derivative continuity: derivative at 0- = α, derivative at 0+ = 1
    // With α = 1, derivative is continuous
}

/// Test ELU asymptotic behavior
#[test]
fn test_elu_simd_asymptotic() {
    let alpha = 1.0_f64;

    // For large positive x, ELU(x) = x
    let large_positive = array![10.0_f64, 50.0, 100.0];
    let result_pos = elu_simd(&large_positive.view(), alpha);
    assert!((result_pos[0] - 10.0).abs() < 1e-10);
    assert!((result_pos[1] - 50.0).abs() < 1e-10);
    assert!((result_pos[2] - 100.0).abs() < 1e-10);

    // For large negative x, ELU(x) → -α
    let large_negative = array![-10.0_f64, -50.0, -100.0];
    let result_neg = elu_simd(&large_negative.view(), alpha);
    // exp(-10) ≈ 4.5e-5, so ELU(-10) ≈ -1 + 4.5e-5 ≈ -0.99995
    assert!(
        (result_neg[0] - (-alpha)).abs() < 0.001,
        "ELU(-10) should be approximately -α, got {}",
        result_neg[0]
    );
    assert!(
        (result_neg[1] - (-alpha)).abs() < 1e-10,
        "ELU(-50) should be approximately -α, got {}",
        result_neg[1]
    );
    assert!(
        (result_neg[2] - (-alpha)).abs() < 1e-10,
        "ELU(-100) should be approximately -α, got {}",
        result_neg[2]
    );
}

/// Test ELU in neural network use case
#[test]
fn test_elu_simd_neural_network_use_case() {
    // Simulate activation in a hidden layer
    let features = array![0.5_f64, 1.2, -0.8, 2.0, -1.5, 0.0, 3.0, -3.0];
    let alpha = 1.0;
    let activated = elu_simd(&features.view(), alpha);

    // All results should be finite
    for i in 0..features.len() {
        assert!(
            activated[i].is_finite(),
            "ELU output should be finite for input {}",
            features[i]
        );
    }

    // Zero should produce zero
    assert!(activated[5].abs() < 1e-10);

    // Positive values should be unchanged
    assert!((activated[0] - 0.5).abs() < 1e-10);
    assert!((activated[1] - 1.2).abs() < 1e-10);
    assert!((activated[3] - 2.0).abs() < 1e-10);
    assert!((activated[6] - 3.0).abs() < 1e-10);

    // Negative values should be in range (-α, 0)
    assert!(activated[2] > -alpha && activated[2] < 0.0);
    assert!(activated[4] > -alpha && activated[4] < 0.0);
    assert!(activated[7] > -alpha && activated[7] < 0.0);
}

// ============================================================================
// Leaky ReLU / PReLU SIMD Tests
// ============================================================================

/// Test Leaky ReLU with f64 - basic behavior
#[test]
fn test_leaky_relu_simd_f64_basic() {
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let alpha = 0.01;
    let result = leaky_relu_simd(&x.view(), alpha);

    // Leaky ReLU(x) = x for x >= 0
    assert!((result[2] - 0.0).abs() < 1e-10, "Leaky ReLU(0) should be 0");
    assert!((result[3] - 1.0).abs() < 1e-10, "Leaky ReLU(1) should be 1");
    assert!((result[4] - 2.0).abs() < 1e-10, "Leaky ReLU(2) should be 2");

    // Leaky ReLU(x) = alpha * x for x < 0
    assert!(
        (result[1] - (-0.01)).abs() < 1e-10,
        "Leaky ReLU(-1) should be -0.01, got {}",
        result[1]
    );
    assert!(
        (result[0] - (-0.02)).abs() < 1e-10,
        "Leaky ReLU(-2) should be -0.02, got {}",
        result[0]
    );
}

/// Test Leaky ReLU with f32
#[test]
fn test_leaky_relu_simd_f32_basic() {
    let x = array![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
    let alpha = 0.1_f32;
    let result = leaky_relu_simd(&x.view(), alpha);

    // Leaky ReLU(x) = x for x >= 0
    assert!((result[2] - 0.0).abs() < 1e-6, "Leaky ReLU(0) should be 0");
    assert!((result[3] - 1.0).abs() < 1e-6, "Leaky ReLU(1) should be 1");

    // Leaky ReLU(x) = alpha * x for x < 0
    assert!(
        (result[1] - (-0.1)).abs() < 1e-6,
        "Leaky ReLU(-1) should be -0.1"
    );
}

/// Test Leaky ReLU empty array
#[test]
fn test_leaky_relu_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = leaky_relu_simd(&x.view(), 0.01);
    assert!(
        result.is_empty(),
        "Leaky ReLU of empty array should be empty"
    );
}

/// Test PReLU alias works correctly
#[test]
fn test_prelu_simd_alias() {
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let alpha = 0.25; // Typical learned PReLU value

    let leaky_result = leaky_relu_simd(&x.view(), alpha);
    let prelu_result = prelu_simd(&x.view(), alpha);

    // PReLU should produce identical results to Leaky ReLU
    for i in 0..x.len() {
        assert!(
            (leaky_result[i] - prelu_result[i]).abs() < 1e-15,
            "PReLU and Leaky ReLU should be identical"
        );
    }
}

/// Test Leaky ReLU with different alpha values
#[test]
fn test_leaky_relu_simd_different_alpha() {
    let x = array![-1.0_f64, -0.5, 0.5, 1.0];

    // Standard Leaky ReLU: alpha = 0.01
    let alpha_001 = 0.01;
    let result_001 = leaky_relu_simd(&x.view(), alpha_001);
    assert!((result_001[0] - (-0.01)).abs() < 1e-10);

    // More aggressive: alpha = 0.3
    let alpha_03 = 0.3;
    let result_03 = leaky_relu_simd(&x.view(), alpha_03);
    assert!((result_03[0] - (-0.3)).abs() < 1e-10);

    // Positive values unchanged regardless of alpha
    assert!((result_001[2] - 0.5).abs() < 1e-10);
    assert!((result_03[2] - 0.5).abs() < 1e-10);
}

/// Test Leaky ReLU large array (SIMD path)
#[test]
fn test_leaky_relu_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let alpha = 0.01;
    let result = leaky_relu_simd(&x.view(), alpha);

    // Check length
    assert_eq!(result.len(), n, "Result should have same length as input");

    // Verify all values are computed correctly for a sample
    for i in (0..n).step_by(500) {
        let xi = x[i];
        let expected = if xi >= 0.0 { xi } else { alpha * xi };
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "Leaky ReLU({}) should be {}, got {}",
            xi,
            expected,
            result[i]
        );
    }
}

/// Test Leaky ReLU NaN handling
#[test]
fn test_leaky_relu_simd_nan() {
    let x = array![f64::NAN, 1.0, f64::NAN, -1.0];
    let result = leaky_relu_simd(&x.view(), 0.01);

    assert!(result[0].is_nan(), "Leaky ReLU(NaN) should be NaN");
    assert!(result[2].is_nan(), "Leaky ReLU(NaN) should be NaN");
    assert!((result[1] - 1.0).abs() < 1e-10);
}

/// Test Leaky ReLU vs ReLU: Leaky ReLU doesn't "die"
#[test]
fn test_leaky_relu_simd_vs_relu() {
    let x = array![-2.0_f64, -1.0, -0.5, 0.0, 0.5, 1.0];
    let alpha = 0.01;
    let result = leaky_relu_simd(&x.view(), alpha);

    // For x >= 0, Leaky ReLU(x) = x (same as ReLU)
    assert!((result[3] - 0.0).abs() < 1e-10); // x = 0
    assert!((result[4] - 0.5).abs() < 1e-10); // x = 0.5
    assert!((result[5] - 1.0).abs() < 1e-10); // x = 1

    // For x < 0, Leaky ReLU(x) = alpha * x (unlike ReLU which outputs 0)
    // This ensures gradient flow for negative inputs
    assert!(
        result[0] < 0.0 && result[0] > -0.1,
        "Leaky ReLU(-2) should be small negative"
    );
    assert!(
        result[1] < 0.0 && result[1] > -0.1,
        "Leaky ReLU(-1) should be small negative"
    );
    assert!(
        result[2] < 0.0 && result[2] > -0.1,
        "Leaky ReLU(-0.5) should be small negative"
    );
}

/// Test Leaky ReLU continuity at x = 0
#[test]
fn test_leaky_relu_simd_continuity_at_zero() {
    let alpha = 0.01_f64;
    let eps = 1e-10;
    let x = array![-eps, 0.0, eps];
    let result = leaky_relu_simd(&x.view(), alpha);

    // All values should be very close to 0
    assert!(
        result[0].abs() < 1e-9,
        "Leaky ReLU(-eps) should be close to 0"
    );
    assert!(result[1].abs() < 1e-15, "Leaky ReLU(0) should be 0");
    assert!(
        result[2].abs() < 1e-9,
        "Leaky ReLU(eps) should be close to 0"
    );
}

/// Test Leaky ReLU in neural network use case
#[test]
fn test_leaky_relu_simd_neural_network_use_case() {
    // Simulate activations in a hidden layer
    let features = array![0.5_f64, 1.2, -0.8, 2.0, -1.5, 0.0, 3.0, -3.0];
    let alpha = 0.01;
    let activated = leaky_relu_simd(&features.view(), alpha);

    // All results should be finite
    for i in 0..features.len() {
        assert!(
            activated[i].is_finite(),
            "Leaky ReLU output should be finite for input {}",
            features[i]
        );
    }

    // Zero should produce zero
    assert!(activated[5].abs() < 1e-10);

    // Positive values should be unchanged
    assert!((activated[0] - 0.5).abs() < 1e-10);
    assert!((activated[1] - 1.2).abs() < 1e-10);
    assert!((activated[3] - 2.0).abs() < 1e-10);
    assert!((activated[6] - 3.0).abs() < 1e-10);

    // Negative values should be scaled by alpha
    assert!((activated[2] - (-0.008)).abs() < 1e-10); // -0.8 * 0.01
    assert!((activated[4] - (-0.015)).abs() < 1e-10); // -1.5 * 0.01
    assert!((activated[7] - (-0.03)).abs() < 1e-10); // -3.0 * 0.01
}

// ============================================================================
// SELU SIMD Tests
// ============================================================================

// SELU constants for reference in tests
const SELU_ALPHA: f64 = 1.6732632423543772;
const SELU_LAMBDA: f64 = 1.0507009873554805;

/// Test SELU with f64 - basic behavior
#[test]
fn test_selu_simd_f64_basic() {
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let result = selu_simd(&x.view());

    // SELU(x) = λ * x for x > 0
    assert!(
        (result[3] - SELU_LAMBDA * 1.0).abs() < 1e-10,
        "SELU(1) should be λ * 1 = {}, got {}",
        SELU_LAMBDA,
        result[3]
    );
    assert!(
        (result[4] - SELU_LAMBDA * 2.0).abs() < 1e-10,
        "SELU(2) should be λ * 2 = {}, got {}",
        SELU_LAMBDA * 2.0,
        result[4]
    );

    // SELU(0) = 0 (boundary case: 0 is treated as x <= 0, so λ * α * (exp(0) - 1) = 0)
    assert!(
        result[2].abs() < 1e-10,
        "SELU(0) should be 0, got {}",
        result[2]
    );

    // SELU(x) = λ * α * (exp(x) - 1) for x <= 0
    let expected_neg1 = SELU_LAMBDA * SELU_ALPHA * ((-1.0_f64).exp() - 1.0);
    let expected_neg2 = SELU_LAMBDA * SELU_ALPHA * ((-2.0_f64).exp() - 1.0);
    assert!(
        (result[1] - expected_neg1).abs() < 1e-10,
        "SELU(-1) should be {}, got {}",
        expected_neg1,
        result[1]
    );
    assert!(
        (result[0] - expected_neg2).abs() < 1e-10,
        "SELU(-2) should be {}, got {}",
        expected_neg2,
        result[0]
    );
}

/// Test SELU with f32
#[test]
fn test_selu_simd_f32_basic() {
    let x = array![-1.0_f32, 0.0, 1.0, 2.0];
    let result = selu_simd(&x.view());

    // SELU(1) = λ * 1 ≈ 1.0507
    assert!(
        (result[2] - 1.050_701_f32).abs() < 1e-5,
        "SELU(1) should be approximately 1.0507"
    );

    // SELU(0) = 0
    assert!(result[1].abs() < 1e-6, "SELU(0) should be 0");

    // SELU(-1) should be negative
    assert!(result[0] < 0.0, "SELU(-1) should be negative");
}

/// Test SELU empty array
#[test]
fn test_selu_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = selu_simd(&x.view());
    assert!(result.is_empty(), "SELU of empty array should be empty");
}

/// Test SELU large array (SIMD path)
#[test]
fn test_selu_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let result = selu_simd(&x.view());

    // Check length
    assert_eq!(result.len(), n, "Result should have same length as input");

    // Verify all values are computed correctly for a sample
    for i in (0..n).step_by(500) {
        let xi = x[i];
        let expected = if xi > 0.0 {
            SELU_LAMBDA * xi
        } else {
            SELU_LAMBDA * SELU_ALPHA * (xi.exp() - 1.0)
        };
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "SELU({}) should be {}, got {}",
            xi,
            expected,
            result[i]
        );
    }
}

/// Test SELU NaN handling
#[test]
fn test_selu_simd_nan() {
    let x = array![f64::NAN, 1.0, f64::NAN, -1.0];
    let result = selu_simd(&x.view());

    assert!(result[0].is_nan(), "SELU(NaN) should be NaN");
    assert!(result[2].is_nan(), "SELU(NaN) should be NaN");
    assert!((result[1] - SELU_LAMBDA).abs() < 1e-10);
}

/// Test SELU vs ELU: SELU is scaled ELU with specific constants
#[test]
fn test_selu_simd_vs_elu() {
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let selu_result = selu_simd(&x.view());
    let elu_result = elu_simd(&x.view(), SELU_ALPHA);

    // SELU(x) = λ * ELU(x, α) for the same α
    for i in 0..x.len() {
        let expected = SELU_LAMBDA * elu_result[i];
        assert!(
            (selu_result[i] - expected).abs() < 1e-10,
            "SELU should be λ * ELU at index {}",
            i
        );
    }
}

/// Test SELU scaling property
#[test]
fn test_selu_simd_scaling() {
    // For positive x, SELU(x) = λ * x
    // So SELU(x) / x = λ for all x > 0
    let x = array![0.5_f64, 1.0, 2.0, 5.0, 10.0];
    let result = selu_simd(&x.view());

    for i in 0..x.len() {
        let ratio = result[i] / x[i];
        assert!(
            (ratio - SELU_LAMBDA).abs() < 1e-10,
            "SELU(x)/x should be λ for positive x"
        );
    }
}

/// Test SELU asymptotic behavior
#[test]
fn test_selu_simd_asymptotic() {
    // For large positive x, SELU(x) = λ * x
    let large_positive = array![10.0_f64, 50.0, 100.0];
    let result_pos = selu_simd(&large_positive.view());
    assert!((result_pos[0] - SELU_LAMBDA * 10.0).abs() < 1e-10);
    assert!((result_pos[1] - SELU_LAMBDA * 50.0).abs() < 1e-10);
    assert!((result_pos[2] - SELU_LAMBDA * 100.0).abs() < 1e-10);

    // For large negative x, SELU(x) → -λ * α ≈ -1.7581
    let large_negative = array![-10.0_f64, -50.0, -100.0];
    let result_neg = selu_simd(&large_negative.view());
    let asymptote = -SELU_LAMBDA * SELU_ALPHA;
    assert!(
        (result_neg[0] - asymptote).abs() < 0.001,
        "SELU(-10) should approach -λα"
    );
    assert!(
        (result_neg[1] - asymptote).abs() < 1e-10,
        "SELU(-50) should be very close to -λα"
    );
    assert!(
        (result_neg[2] - asymptote).abs() < 1e-10,
        "SELU(-100) should be very close to -λα"
    );
}

/// Test SELU self-normalizing property (informal check)
#[test]
fn test_selu_simd_self_normalizing_property() {
    // With properly initialized weights, SELU maintains mean ≈ 0, variance ≈ 1
    // This is a simplified test showing the basic transformation properties

    // Create input with approximate mean=0, var=1
    let n = 1000;
    let x: Array1<f64> = Array1::linspace(-3.0, 3.0, n);

    let result = selu_simd(&x.view());

    // All results should be finite
    for i in 0..n {
        assert!(result[i].is_finite(), "SELU output should be finite");
    }

    // Outputs should be bounded below by -λα ≈ -1.7581
    let lower_bound = -SELU_LAMBDA * SELU_ALPHA;
    for i in 0..n {
        assert!(
            result[i] >= lower_bound - 0.001,
            "SELU output should be >= -λα"
        );
    }
}

/// Test SELU in self-normalizing neural network use case
#[test]
fn test_selu_simd_snn_use_case() {
    // Simulate activation in a Self-Normalizing Neural Network
    let features = array![0.5_f64, 1.2, -0.8, 2.0, -1.5, 0.0, 3.0, -3.0];
    let activated = selu_simd(&features.view());

    // All results should be finite
    for i in 0..features.len() {
        assert!(
            activated[i].is_finite(),
            "SELU output should be finite for input {}",
            features[i]
        );
    }

    // Zero should produce zero
    assert!(activated[5].abs() < 1e-10);

    // Positive values should be scaled by λ
    assert!((activated[0] - SELU_LAMBDA * 0.5).abs() < 1e-10);
    assert!((activated[1] - SELU_LAMBDA * 1.2).abs() < 1e-10);
    assert!((activated[3] - SELU_LAMBDA * 2.0).abs() < 1e-10);
    assert!((activated[6] - SELU_LAMBDA * 3.0).abs() < 1e-10);

    // Negative values should be in range (-λα, 0)
    let lower_bound = -SELU_LAMBDA * SELU_ALPHA;
    assert!(activated[2] > lower_bound && activated[2] < 0.0);
    assert!(activated[4] > lower_bound && activated[4] < 0.0);
    assert!(activated[7] > lower_bound && activated[7] < 0.0);
}

// ============================================================================
// Hardsigmoid SIMD Tests
// ============================================================================

/// Test Hardsigmoid with f64 - basic behavior
#[test]
fn test_hardsigmoid_simd_f64_basic() {
    let x = array![-4.0_f64, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0];
    let result = hardsigmoid_simd(&x.view());

    // Hardsigmoid(x) = 0 for x <= -3
    assert!(result[0].abs() < 1e-10, "Hardsigmoid(-4) should be 0");
    assert!(result[1].abs() < 1e-10, "Hardsigmoid(-3) should be 0");

    // Hardsigmoid(x) = (x + 3) / 6 for -3 < x < 3
    assert!(
        (result[2] - 0.25).abs() < 1e-10,
        "Hardsigmoid(-1.5) should be 0.25"
    );
    assert!(
        (result[3] - 0.5).abs() < 1e-10,
        "Hardsigmoid(0) should be 0.5"
    );
    assert!(
        (result[4] - 0.75).abs() < 1e-10,
        "Hardsigmoid(1.5) should be 0.75"
    );

    // Hardsigmoid(x) = 1 for x >= 3
    assert!(
        (result[5] - 1.0).abs() < 1e-10,
        "Hardsigmoid(3) should be 1"
    );
    assert!(
        (result[6] - 1.0).abs() < 1e-10,
        "Hardsigmoid(4) should be 1"
    );
}

/// Test Hardsigmoid with f32
#[test]
fn test_hardsigmoid_simd_f32_basic() {
    let x = array![-4.0_f32, 0.0, 4.0];
    let result = hardsigmoid_simd(&x.view());

    assert!(result[0].abs() < 1e-6, "Hardsigmoid(-4) should be 0");
    assert!(
        (result[1] - 0.5).abs() < 1e-6,
        "Hardsigmoid(0) should be 0.5"
    );
    assert!((result[2] - 1.0).abs() < 1e-6, "Hardsigmoid(4) should be 1");
}

/// Test Hardsigmoid empty array
#[test]
fn test_hardsigmoid_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = hardsigmoid_simd(&x.view());
    assert!(
        result.is_empty(),
        "Hardsigmoid of empty array should be empty"
    );
}

/// Test Hardsigmoid large array (SIMD path)
#[test]
fn test_hardsigmoid_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let result = hardsigmoid_simd(&x.view());

    assert_eq!(result.len(), n);

    for i in (0..n).step_by(500) {
        let xi = x[i];
        let expected = if xi <= -3.0 {
            0.0
        } else if xi >= 3.0 {
            1.0
        } else {
            (xi + 3.0) / 6.0
        };
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "Hardsigmoid({}) should be {}, got {}",
            xi,
            expected,
            result[i]
        );
    }
}

/// Test Hardsigmoid output range [0, 1]
#[test]
fn test_hardsigmoid_simd_output_range() {
    let x: Array1<f64> = Array1::linspace(-10.0, 10.0, 1000);
    let result = hardsigmoid_simd(&x.view());

    for i in 0..result.len() {
        assert!(
            result[i] >= 0.0 && result[i] <= 1.0,
            "Hardsigmoid output should be in [0, 1], got {}",
            result[i]
        );
    }
}

/// Test Hardsigmoid vs Sigmoid approximation
#[test]
fn test_hardsigmoid_simd_vs_sigmoid() {
    // Hardsigmoid approximates sigmoid in the linear region
    let x = array![0.0_f64];
    let hardsigmoid_result = hardsigmoid_simd(&x.view());
    let sigmoid_result = sigmoid_simd(&x.view());

    // Both should be 0.5 at x = 0
    assert!(
        (hardsigmoid_result[0] - 0.5).abs() < 1e-10,
        "Hardsigmoid(0) should be 0.5"
    );
    assert!(
        (sigmoid_result[0] - 0.5).abs() < 1e-10,
        "Sigmoid(0) should be 0.5"
    );
}

/// Test Hardsigmoid NaN handling
#[test]
fn test_hardsigmoid_simd_nan() {
    let x = array![f64::NAN, 0.0, f64::NAN];
    let result = hardsigmoid_simd(&x.view());

    assert!(result[0].is_nan(), "Hardsigmoid(NaN) should be NaN");
    assert!(result[2].is_nan(), "Hardsigmoid(NaN) should be NaN");
    assert!((result[1] - 0.5).abs() < 1e-10);
}

// ============================================================================
// Hardswish SIMD Tests
// ============================================================================

/// Test Hardswish with f64 - basic behavior
#[test]
fn test_hardswish_simd_f64_basic() {
    let x = array![-4.0_f64, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0];
    let result = hardswish_simd(&x.view());

    // Hardswish(x) = 0 for x <= -3
    assert!(result[0].abs() < 1e-10, "Hardswish(-4) should be 0");
    assert!(result[1].abs() < 1e-10, "Hardswish(-3) should be 0");

    // Hardswish(x) = x * (x + 3) / 6 for -3 < x < 3
    let expected_neg15 = -1.5 * (-1.5 + 3.0) / 6.0; // = -1.5 * 1.5 / 6 = -0.375
    assert!(
        (result[2] - expected_neg15).abs() < 1e-10,
        "Hardswish(-1.5) should be {}, got {}",
        expected_neg15,
        result[2]
    );
    assert!(result[3].abs() < 1e-10, "Hardswish(0) should be 0");
    let expected_15 = 1.5 * (1.5 + 3.0) / 6.0; // = 1.5 * 4.5 / 6 = 1.125
    assert!(
        (result[4] - expected_15).abs() < 1e-10,
        "Hardswish(1.5) should be {}, got {}",
        expected_15,
        result[4]
    );

    // Hardswish(x) = x for x >= 3
    assert!((result[5] - 3.0).abs() < 1e-10, "Hardswish(3) should be 3");
    assert!((result[6] - 4.0).abs() < 1e-10, "Hardswish(4) should be 4");
}

/// Test Hardswish with f32
#[test]
fn test_hardswish_simd_f32_basic() {
    let x = array![-4.0_f32, 0.0, 4.0];
    let result = hardswish_simd(&x.view());

    assert!(result[0].abs() < 1e-6, "Hardswish(-4) should be 0");
    assert!(result[1].abs() < 1e-6, "Hardswish(0) should be 0");
    assert!((result[2] - 4.0).abs() < 1e-6, "Hardswish(4) should be 4");
}

/// Test Hardswish empty array
#[test]
fn test_hardswish_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = hardswish_simd(&x.view());
    assert!(
        result.is_empty(),
        "Hardswish of empty array should be empty"
    );
}

/// Test Hardswish large array (SIMD path)
#[test]
fn test_hardswish_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let result = hardswish_simd(&x.view());

    assert_eq!(result.len(), n);

    for i in (0..n).step_by(500) {
        let xi = x[i];
        let expected = if xi <= -3.0 {
            0.0
        } else if xi >= 3.0 {
            xi
        } else {
            xi * (xi + 3.0) / 6.0
        };
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "Hardswish({}) should be {}, got {}",
            xi,
            expected,
            result[i]
        );
    }
}

/// Test Hardswish relation to Hardsigmoid
#[test]
fn test_hardswish_simd_vs_hardsigmoid() {
    // Hardswish(x) = x * hardsigmoid(x)
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let hardswish_result = hardswish_simd(&x.view());
    let hardsigmoid_result = hardsigmoid_simd(&x.view());

    for i in 0..x.len() {
        let expected = x[i] * hardsigmoid_result[i];
        assert!(
            (hardswish_result[i] - expected).abs() < 1e-10,
            "Hardswish should be x * hardsigmoid(x)"
        );
    }
}

/// Test Hardswish vs Swish approximation
#[test]
fn test_hardswish_simd_vs_swish() {
    // Hardswish approximates Swish but is faster (no exp)
    let x = array![0.0_f64, 1.0, 2.0];
    let hardswish_result = hardswish_simd(&x.view());
    let swish_result = swish_simd(&x.view());

    // At x = 0, both should be 0
    assert!(hardswish_result[0].abs() < 1e-10);
    assert!(swish_result[0].abs() < 1e-10);

    // At x = 1 and 2, they should be close but not identical
    // Swish(1) ≈ 0.731, Hardswish(1) = 1 * 4/6 ≈ 0.667
    // Swish(2) ≈ 1.762, Hardswish(2) = 2 * 5/6 ≈ 1.667
    assert!(hardswish_result[1] > 0.5 && hardswish_result[1] < 1.0);
    assert!(hardswish_result[2] > 1.5 && hardswish_result[2] < 2.0);
}

/// Test Hardswish NaN handling
#[test]
fn test_hardswish_simd_nan() {
    let x = array![f64::NAN, 0.0, f64::NAN];
    let result = hardswish_simd(&x.view());

    assert!(result[0].is_nan(), "Hardswish(NaN) should be NaN");
    assert!(result[2].is_nan(), "Hardswish(NaN) should be NaN");
    assert!(result[1].abs() < 1e-10);
}

/// Test Hardswish in MobileNetV3 use case
#[test]
fn test_hardswish_simd_mobilenetv3_use_case() {
    // Simulate activations in a MobileNetV3 layer
    let features = array![0.5_f64, 1.2, -0.8, 2.0, -1.5, 0.0, 3.5, -3.5];
    let activated = hardswish_simd(&features.view());

    // All results should be finite
    for i in 0..features.len() {
        assert!(
            activated[i].is_finite(),
            "Hardswish output should be finite"
        );
    }

    // Zero should produce zero
    assert!(activated[5].abs() < 1e-10);

    // Large positive should be identity
    assert!((activated[6] - 3.5).abs() < 1e-10);

    // Large negative should be zero
    assert!(activated[7].abs() < 1e-10);
}

// ============================================================================
// Sinc SIMD Tests
// ============================================================================

/// Test Sinc with f64 - basic behavior
#[test]
fn test_sinc_simd_f64_basic() {
    let x = array![0.0_f64, 0.5, 1.0, 2.0, 3.0, -1.0, -2.0];
    let result = sinc_simd(&x.view());

    // sinc(0) = 1
    assert!(
        (result[0] - 1.0).abs() < 1e-10,
        "sinc(0) should be 1, got {}",
        result[0]
    );

    // sinc(0.5) = sin(π/2) / (π/2) = 1 / (π/2) ≈ 0.6366
    let expected_05 = (std::f64::consts::FRAC_PI_2).sin() / std::f64::consts::FRAC_PI_2;
    assert!(
        (result[1] - expected_05).abs() < 1e-10,
        "sinc(0.5) should be {}, got {}",
        expected_05,
        result[1]
    );

    // sinc(n) = 0 for all non-zero integers n
    assert!(
        result[2].abs() < 1e-10,
        "sinc(1) should be 0, got {}",
        result[2]
    );
    assert!(
        result[3].abs() < 1e-10,
        "sinc(2) should be 0, got {}",
        result[3]
    );
    assert!(
        result[4].abs() < 1e-10,
        "sinc(3) should be 0, got {}",
        result[4]
    );

    // sinc is an even function: sinc(-x) = sinc(x)
    assert!(
        result[5].abs() < 1e-10,
        "sinc(-1) should be 0, got {}",
        result[5]
    );
    assert!(
        result[6].abs() < 1e-10,
        "sinc(-2) should be 0, got {}",
        result[6]
    );
}

/// Test Sinc with f32
#[test]
fn test_sinc_simd_f32_basic() {
    let x = array![0.0_f32, 0.5, 1.0, -1.0];
    let result = sinc_simd(&x.view());

    // sinc(0) = 1
    assert!((result[0] - 1.0).abs() < 1e-6, "sinc(0) should be 1");

    // sinc(1) = 0
    assert!(result[2].abs() < 1e-6, "sinc(1) should be 0");

    // sinc(-1) = 0
    assert!(result[3].abs() < 1e-6, "sinc(-1) should be 0");
}

/// Test Sinc empty array
#[test]
fn test_sinc_simd_empty() {
    let x = array![] as Array1<f64>;
    let result = sinc_simd(&x.view());
    assert!(result.is_empty(), "sinc of empty array should be empty");
}

/// Test Sinc even function property
#[test]
fn test_sinc_simd_even_function() {
    // sinc(-x) = sinc(x) for all x
    let x = array![-3.0_f64, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 3.0];
    let result = sinc_simd(&x.view());

    // Verify symmetry
    assert!(
        (result[0] - result[8]).abs() < 1e-10,
        "sinc(-3) should equal sinc(3)"
    );
    assert!(
        (result[1] - result[7]).abs() < 1e-10,
        "sinc(-2.5) should equal sinc(2.5)"
    );
    assert!(
        (result[2] - result[6]).abs() < 1e-10,
        "sinc(-1.5) should equal sinc(1.5)"
    );
    assert!(
        (result[3] - result[5]).abs() < 1e-10,
        "sinc(-0.5) should equal sinc(0.5)"
    );
}

/// Test Sinc large array (SIMD path)
#[test]
fn test_sinc_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-5.0, 5.0, n);
    let result = sinc_simd(&x.view());

    assert_eq!(result.len(), n);

    // All values should be in range [-0.22, 1.0] (sinc is bounded)
    for i in 0..result.len() {
        assert!(
            result[i] >= -0.25 && result[i] <= 1.0,
            "sinc output should be bounded, got {} at index {}",
            result[i],
            i
        );
    }

    // Maximum is at x = 0
    let mid = n / 2;
    assert!(
        (result[mid] - 1.0).abs() < 0.01,
        "sinc(0) should be approximately 1"
    );
}

/// Test Sinc zeros at integers
#[test]
fn test_sinc_simd_zeros_at_integers() {
    // sinc(n) = 0 for all non-zero integers n
    let x = array![-5.0_f64, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let result = sinc_simd(&x.view());

    for i in 0..result.len() {
        assert!(
            result[i].abs() < 1e-10,
            "sinc({}) should be 0, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test Sinc NaN handling
#[test]
fn test_sinc_simd_nan() {
    let x = array![f64::NAN, 0.0, f64::NAN, 1.0];
    let result = sinc_simd(&x.view());

    assert!(result[0].is_nan(), "sinc(NaN) should be NaN");
    assert!(result[2].is_nan(), "sinc(NaN) should be NaN");
    assert!((result[1] - 1.0).abs() < 1e-10);
    assert!(result[3].abs() < 1e-10);
}

/// Test Sinc numerical stability near zero
#[test]
fn test_sinc_simd_near_zero() {
    // Test that we handle very small x values correctly (avoid 0/0)
    let x = array![1e-10_f64, 1e-15_f64, 1e-20_f64, -1e-10, -1e-15, -1e-20];
    let result = sinc_simd(&x.view());

    // All should be very close to 1
    for i in 0..result.len() {
        assert!(
            (result[i] - 1.0).abs() < 1e-8,
            "sinc({}) should be approximately 1, got {}",
            x[i],
            result[i]
        );
    }
}

/// Test Sinc for signal processing use case
#[test]
fn test_sinc_simd_signal_processing_use_case() {
    // In ideal low-pass filter design, sinc is the impulse response
    // Test the typical values used in reconstruction
    let n = 21;
    let x: Array1<f64> = Array1::linspace(-10.0, 10.0, n);
    let result = sinc_simd(&x.view());

    // All results should be finite
    for i in 0..n {
        assert!(
            result[i].is_finite(),
            "sinc output should be finite for input {}",
            x[i]
        );
    }

    // Center should be 1
    assert!(
        (result[10] - 1.0).abs() < 1e-10,
        "sinc(0) should be 1 in the center"
    );

    // Integer points should be 0 (except center)
    for i in [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    ] {
        if (x[i] - x[i].round()).abs() < 1e-10 {
            assert!(
                result[i].abs() < 1e-10,
                "sinc({}) should be 0, got {}",
                x[i],
                result[i]
            );
        }
    }
}

// =============================================================================
// Log-Softmax Tests (Phase 58)
// =============================================================================

use scirs2_core::ndarray_ext::elementwise::log_softmax_simd;

/// Test Log-Softmax basic f64
#[test]
fn test_log_softmax_simd_basic_f64() {
    let x = array![1.0_f64, 2.0, 3.0];
    let result = log_softmax_simd(&x.view());

    // exp(log_softmax) should sum to 1
    let sum_exp: f64 = result.mapv(|v| v.exp()).sum();
    assert!(
        (sum_exp - 1.0).abs() < 1e-10,
        "exp(log_softmax) should sum to 1, got {}",
        sum_exp
    );

    // All outputs should be <= 0 (log of probabilities <= 1)
    for (i, &v) in result.iter().enumerate() {
        assert!(v <= 0.0, "log_softmax[{}] = {} should be <= 0", i, v);
    }
}

/// Test Log-Softmax basic f32
#[test]
fn test_log_softmax_simd_basic_f32() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0];
    let result = log_softmax_simd(&x.view());

    // exp(log_softmax) should sum to 1
    let sum_exp: f32 = result.mapv(|v| v.exp()).sum();
    assert!(
        (sum_exp - 1.0).abs() < 1e-5,
        "exp(log_softmax) should sum to 1, got {}",
        sum_exp
    );

    // Largest input should have log_softmax closest to 0
    let max_idx = 3; // x[3] = 4.0 is largest
    for i in 0..result.len() {
        if i != max_idx {
            assert!(
                result[i] < result[max_idx],
                "Largest input should have largest log_softmax output"
            );
        }
    }
}

/// Test Log-Softmax empty array
#[test]
fn test_log_softmax_simd_empty() {
    let x: Array1<f64> = array![];
    let result = log_softmax_simd(&x.view());
    assert_eq!(result.len(), 0, "Empty input should give empty output");
}

/// Test Log-Softmax large array (SIMD path)
#[test]
fn test_log_softmax_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-10.0, 10.0, n);
    let result = log_softmax_simd(&x.view());

    assert_eq!(result.len(), n, "Output should have same length as input");

    // exp(log_softmax) should sum to 1
    let sum_exp: f64 = result.mapv(|v| v.exp()).sum();
    assert!(
        (sum_exp - 1.0).abs() < 1e-8,
        "exp(log_softmax) should sum to 1, got {}",
        sum_exp
    );
}

/// Test Log-Softmax shift invariance
#[test]
fn test_log_softmax_simd_shift_invariance() {
    // log_softmax(x + c) = log_softmax(x) for any constant c
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let x_shifted = &x + 100.0;

    let result1 = log_softmax_simd(&x.view());
    let result2 = log_softmax_simd(&x_shifted.view());

    for i in 0..x.len() {
        assert!(
            (result1[i] - result2[i]).abs() < 1e-10,
            "log_softmax should be shift-invariant, but got {} vs {} at index {}",
            result1[i],
            result2[i],
            i
        );
    }
}

/// Test Log-Softmax equals log(softmax)
#[test]
fn test_log_softmax_simd_equals_log_softmax() {
    use scirs2_core::simd_ops::SimdUnifiedOps;

    let x = array![0.5_f64, 1.5, 2.5, -0.5, -1.5];
    let log_sm = log_softmax_simd(&x.view());

    // Compute softmax and take log
    let softmax = f64::simd_softmax(&x.view());
    let log_of_softmax = softmax.mapv(|v| v.ln());

    for i in 0..x.len() {
        assert!(
            (log_sm[i] - log_of_softmax[i]).abs() < 1e-10,
            "log_softmax should equal log(softmax), got {} vs {} at index {}",
            log_sm[i],
            log_of_softmax[i],
            i
        );
    }
}

/// Test Log-Softmax NaN handling
#[test]
fn test_log_softmax_simd_nan() {
    let x = array![1.0_f64, f64::NAN, 3.0];
    let result = log_softmax_simd(&x.view());

    // When there's a NaN in input, the log_sum_exp will be NaN,
    // so all outputs will be NaN
    for &v in result.iter() {
        assert!(v.is_nan(), "log_softmax with NaN input should produce NaN");
    }
}

/// Test Log-Softmax numerical stability for large values
#[test]
fn test_log_softmax_simd_large_values() {
    // Large positive values that would overflow in naive softmax
    let x = array![1000.0_f64, 1001.0, 1002.0];
    let result = log_softmax_simd(&x.view());

    // Should still be valid (not NaN or -Inf)
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "log_softmax should be stable for large values, got {} at index {}",
            v,
            i
        );
    }

    // exp(log_softmax) should sum to 1
    let sum_exp: f64 = result.mapv(|v| v.exp()).sum();
    assert!(
        (sum_exp - 1.0).abs() < 1e-10,
        "exp(log_softmax) should sum to 1 even for large inputs, got {}",
        sum_exp
    );
}

/// Test Log-Softmax numerical stability for large negative values
#[test]
fn test_log_softmax_simd_large_negative() {
    // Large negative values
    let x = array![-1000.0_f64, -999.0, -998.0];
    let result = log_softmax_simd(&x.view());

    // Should still be valid
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "log_softmax should be stable for large negative values, got {} at index {}",
            v,
            i
        );
    }

    // exp(log_softmax) should sum to 1
    let sum_exp: f64 = result.mapv(|v| v.exp()).sum();
    assert!(
        (sum_exp - 1.0).abs() < 1e-10,
        "exp(log_softmax) should sum to 1 even for large negative inputs, got {}",
        sum_exp
    );
}

/// Test Log-Softmax for cross-entropy loss use case
#[test]
fn test_log_softmax_simd_cross_entropy_use_case() {
    // Simulate classification with 10 classes
    let logits = array![1.0_f64, 0.5, 2.5, -1.0, 0.0, 3.0, 1.5, -0.5, 0.25, 0.75];
    let log_probs = log_softmax_simd(&logits.view());

    // One-hot target (class 5 is correct)
    let target = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

    // Cross-entropy loss = -sum(target * log_probs)
    let cross_entropy: f64 = -(0..10).map(|i| target[i] * log_probs[i]).sum::<f64>();

    // Loss should be positive (since log_probs are negative)
    assert!(
        cross_entropy > 0.0,
        "Cross-entropy loss should be positive, got {}",
        cross_entropy
    );

    // Loss should equal -log_probs[5] (the correct class)
    assert!(
        (cross_entropy - (-log_probs[5])).abs() < 1e-10,
        "Cross-entropy should equal -log_probs[correct_class], got {} vs {}",
        cross_entropy,
        -log_probs[5]
    );
}

/// Test Log-Softmax for Transformer use case
#[test]
fn test_log_softmax_simd_transformer_use_case() {
    // Simulate vocabulary logits in a language model (vocab size = 50)
    let n = 50;
    let mut logits = Array1::zeros(n);
    for i in 0..n {
        logits[i] = (i as f64 - 25.0) * 0.1; // Some variation
    }

    let log_probs = log_softmax_simd(&logits.view());

    // All log probabilities should be finite
    for (i, &lp) in log_probs.iter().enumerate() {
        assert!(
            lp.is_finite(),
            "Log probability at index {} should be finite, got {}",
            i,
            lp
        );
        assert!(lp <= 0.0, "Log probability should be <= 0, got {}", lp);
    }

    // Perplexity = exp(-mean(log_probs)) should be reasonable
    let mean_log_prob = log_probs.mean().unwrap();
    let perplexity = (-mean_log_prob).exp();
    assert!(
        perplexity > 0.0 && perplexity < f64::INFINITY,
        "Perplexity should be finite positive, got {}",
        perplexity
    );
}

// =============================================================================
// Inverse Hyperbolic Function Tests (Phase 59)
// =============================================================================

use scirs2_core::ndarray_ext::elementwise::{acosh_simd, asinh_simd, atanh_simd};

// -----------------------------------------------------------------------------
// asinh (Inverse Hyperbolic Sine) Tests
// -----------------------------------------------------------------------------

/// Test asinh basic f64
#[test]
fn test_asinh_simd_basic_f64() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0, -2.0];
    let result = asinh_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-10, "asinh(0) should be 0");
    assert!(
        (result[1] - 0.881373587).abs() < 1e-6,
        "asinh(1) should be ~0.881"
    );
    assert!(
        (result[1] + result[2]).abs() < 1e-10,
        "asinh should be odd function"
    );
}

/// Test asinh basic f32
#[test]
fn test_asinh_simd_basic_f32() {
    let x = array![0.0_f32, 1.0, -1.0, 5.0];
    let result = asinh_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-5, "asinh(0) should be 0");
    assert!(
        (result[1] - 0.881_373_6).abs() < 1e-4,
        "asinh(1) should be ~0.881"
    );
}

/// Test asinh empty array
#[test]
fn test_asinh_simd_empty() {
    let x: Array1<f64> = array![];
    let result = asinh_simd(&x.view());
    assert_eq!(result.len(), 0, "Empty input should give empty output");
}

/// Test asinh large array (SIMD path)
#[test]
fn test_asinh_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-100.0, 100.0, n);
    let result = asinh_simd(&x.view());

    assert_eq!(result.len(), n);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "asinh should be finite at index {}", i);
    }
}

/// Test asinh odd function property
#[test]
fn test_asinh_simd_odd_function() {
    let x = array![0.5_f64, 1.5, 2.5, 10.0, 100.0];
    let neg_x = x.mapv(|v| -v);

    let result_pos = asinh_simd(&x.view());
    let result_neg = asinh_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!(
            (result_pos[i] + result_neg[i]).abs() < 1e-10,
            "asinh(-x) should equal -asinh(x) at index {}",
            i
        );
    }
}

/// Test asinh inverse of sinh
#[test]
fn test_asinh_simd_inverse_of_sinh() {
    let x = array![0.0_f64, 0.5, 1.0, 2.0, 5.0];
    let sinh_x = x.mapv(|v| v.sinh());
    let asinh_sinh_x = asinh_simd(&sinh_x.view());

    for i in 0..x.len() {
        assert!(
            (asinh_sinh_x[i] - x[i]).abs() < 1e-10,
            "asinh(sinh(x)) should equal x, got {} vs {} at index {}",
            asinh_sinh_x[i],
            x[i],
            i
        );
    }
}

/// Test asinh NaN handling
#[test]
fn test_asinh_simd_nan() {
    let x = array![f64::NAN, 1.0, f64::NAN];
    let result = asinh_simd(&x.view());

    assert!(result[0].is_nan(), "asinh(NaN) should be NaN");
    assert!(result[2].is_nan(), "asinh(NaN) should be NaN");
    assert!(result[1].is_finite(), "asinh(1) should be finite");
}

// -----------------------------------------------------------------------------
// acosh (Inverse Hyperbolic Cosine) Tests
// -----------------------------------------------------------------------------

/// Test acosh basic f64
#[test]
fn test_acosh_simd_basic_f64() {
    let x = array![1.0_f64, 2.0, 3.0, 10.0];
    let result = acosh_simd(&x.view());

    assert!(
        (result[0] - 0.0).abs() < 1e-10,
        "acosh(1) should be 0, got {}",
        result[0]
    );
    assert!(
        (result[1] - 1.31695790).abs() < 1e-6,
        "acosh(2) should be ~1.317"
    );
}

/// Test acosh basic f32
#[test]
fn test_acosh_simd_basic_f32() {
    let x = array![1.0_f32, 2.0, 5.0];
    let result = acosh_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-5, "acosh(1) should be 0");
    assert!(
        (result[1] - 1.316_958).abs() < 1e-4,
        "acosh(2) should be ~1.317"
    );
}

/// Test acosh empty array
#[test]
fn test_acosh_simd_empty() {
    let x: Array1<f64> = array![];
    let result = acosh_simd(&x.view());
    assert_eq!(result.len(), 0, "Empty input should give empty output");
}

/// Test acosh large array (SIMD path)
#[test]
fn test_acosh_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(1.0, 1000.0, n);
    let result = acosh_simd(&x.view());

    assert_eq!(result.len(), n);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "acosh should be finite at index {}", i);
        assert!(
            v >= 0.0,
            "acosh should be non-negative, got {} at index {}",
            v,
            i
        );
    }
}

/// Test acosh out of domain (x < 1)
#[test]
fn test_acosh_simd_out_of_domain() {
    let x = array![0.0_f64, 0.5, 0.99, -1.0];
    let result = acosh_simd(&x.view());

    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_nan(),
            "acosh({}) should be NaN for x < 1, got {}",
            x[i],
            v
        );
    }
}

/// Test acosh inverse of cosh
#[test]
fn test_acosh_simd_inverse_of_cosh() {
    let x = array![0.0_f64, 0.5, 1.0, 2.0, 5.0];
    let cosh_x = x.mapv(|v| v.cosh());
    let acosh_cosh_x = acosh_simd(&cosh_x.view());

    for i in 0..x.len() {
        // Note: acosh(cosh(x)) = |x| since cosh is even
        assert!(
            (acosh_cosh_x[i] - x[i].abs()).abs() < 1e-10,
            "acosh(cosh(x)) should equal |x|, got {} vs {} at index {}",
            acosh_cosh_x[i],
            x[i].abs(),
            i
        );
    }
}

/// Test acosh NaN handling
#[test]
fn test_acosh_simd_nan() {
    let x = array![f64::NAN, 2.0, f64::NAN];
    let result = acosh_simd(&x.view());

    assert!(result[0].is_nan(), "acosh(NaN) should be NaN");
    assert!(result[2].is_nan(), "acosh(NaN) should be NaN");
    assert!(result[1].is_finite(), "acosh(2) should be finite");
}

// -----------------------------------------------------------------------------
// atanh (Inverse Hyperbolic Tangent) Tests
// -----------------------------------------------------------------------------

/// Test atanh basic f64
#[test]
fn test_atanh_simd_basic_f64() {
    let x = array![0.0_f64, 0.5, -0.5, 0.9];
    let result = atanh_simd(&x.view());

    assert!(
        (result[0] - 0.0).abs() < 1e-10,
        "atanh(0) should be 0, got {}",
        result[0]
    );
    assert!(
        (result[1] - 0.54930614).abs() < 1e-6,
        "atanh(0.5) should be ~0.549"
    );
    assert!(
        (result[1] + result[2]).abs() < 1e-10,
        "atanh should be odd function"
    );
}

/// Test atanh basic f32
#[test]
fn test_atanh_simd_basic_f32() {
    let x = array![0.0_f32, 0.5, -0.5];
    let result = atanh_simd(&x.view());

    assert!((result[0] - 0.0).abs() < 1e-5, "atanh(0) should be 0");
    assert!(
        (result[1] - 0.54930614).abs() < 1e-4,
        "atanh(0.5) should be ~0.549"
    );
}

/// Test atanh empty array
#[test]
fn test_atanh_simd_empty() {
    let x: Array1<f64> = array![];
    let result = atanh_simd(&x.view());
    assert_eq!(result.len(), 0, "Empty input should give empty output");
}

/// Test atanh large array (SIMD path)
#[test]
fn test_atanh_simd_large_array() {
    let n = 10000;
    let x: Array1<f64> = Array1::linspace(-0.99, 0.99, n);
    let result = atanh_simd(&x.view());

    assert_eq!(result.len(), n);
    for (i, &v) in result.iter().enumerate() {
        assert!(v.is_finite(), "atanh should be finite at index {}", i);
    }
}

/// Test atanh odd function property
#[test]
fn test_atanh_simd_odd_function() {
    let x = array![0.1_f64, 0.3, 0.5, 0.7, 0.9];
    let neg_x = x.mapv(|v| -v);

    let result_pos = atanh_simd(&x.view());
    let result_neg = atanh_simd(&neg_x.view());

    for i in 0..x.len() {
        assert!(
            (result_pos[i] + result_neg[i]).abs() < 1e-10,
            "atanh(-x) should equal -atanh(x) at index {}",
            i
        );
    }
}

/// Test atanh boundaries (±1 → ±∞)
#[test]
fn test_atanh_simd_boundaries() {
    let x = array![1.0_f64, -1.0];
    let result = atanh_simd(&x.view());

    assert!(
        result[0].is_infinite() && result[0] > 0.0,
        "atanh(1) should be +∞"
    );
    assert!(
        result[1].is_infinite() && result[1] < 0.0,
        "atanh(-1) should be -∞"
    );
}

/// Test atanh out of domain (|x| > 1)
#[test]
fn test_atanh_simd_out_of_domain() {
    let x = array![1.5_f64, -1.5, 10.0];
    let result = atanh_simd(&x.view());

    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_nan(),
            "atanh({}) should be NaN for |x| > 1, got {}",
            x[i],
            v
        );
    }
}

/// Test atanh inverse of tanh
#[test]
fn test_atanh_simd_inverse_of_tanh() {
    let x = array![0.0_f64, 0.5, 1.0, 2.0, -0.5];
    let tanh_x = x.mapv(|v| v.tanh());
    let atanh_tanh_x = atanh_simd(&tanh_x.view());

    for i in 0..x.len() {
        assert!(
            (atanh_tanh_x[i] - x[i]).abs() < 1e-10,
            "atanh(tanh(x)) should equal x, got {} vs {} at index {}",
            atanh_tanh_x[i],
            x[i],
            i
        );
    }
}

/// Test atanh NaN handling
#[test]
fn test_atanh_simd_nan() {
    let x = array![f64::NAN, 0.5, f64::NAN];
    let result = atanh_simd(&x.view());

    assert!(result[0].is_nan(), "atanh(NaN) should be NaN");
    assert!(result[2].is_nan(), "atanh(NaN) should be NaN");
    assert!(result[1].is_finite(), "atanh(0.5) should be finite");
}

/// Test atanh Fisher's z-transformation use case
#[test]
fn test_atanh_simd_fisher_z_use_case() {
    // Fisher's z-transformation: z = atanh(r) where r is correlation coefficient
    // This is used to make correlation coefficient sampling distributions more normal
    let correlations = array![0.0_f64, 0.3, 0.5, 0.7, 0.9, -0.5];
    let fisher_z = atanh_simd(&correlations.view());

    // z values should be finite for valid correlations
    for (i, &z) in fisher_z.iter().enumerate() {
        assert!(
            z.is_finite(),
            "Fisher z for correlation {} should be finite, got {}",
            correlations[i],
            z
        );
    }

    // z should preserve order (atanh is monotonically increasing)
    for i in 0..correlations.len() - 1 {
        if correlations[i] < correlations[i + 1] {
            assert!(
                fisher_z[i] < fisher_z[i + 1],
                "atanh should be monotonically increasing"
            );
        }
    }
}

// =============================================================================
// Beta Function Tests (Phase 60)
// =============================================================================

use scirs2_core::ndarray_ext::elementwise::{beta_simd, ln_beta_simd};

// -----------------------------------------------------------------------------
// beta (Beta Function) Tests
// -----------------------------------------------------------------------------

/// Test beta basic f64
#[test]
fn test_beta_simd_basic_f64() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![1.0_f64, 2.0, 3.0];
    let result = beta_simd(&a.view(), &b.view());

    // B(1, 1) = 1
    assert!(
        (result[0] - 1.0).abs() < 1e-10,
        "B(1, 1) should be 1, got {}",
        result[0]
    );

    // B(2, 2) = Γ(2)Γ(2)/Γ(4) = 1*1/6 = 1/6
    assert!(
        (result[1] - 1.0 / 6.0).abs() < 1e-10,
        "B(2, 2) should be 1/6, got {}",
        result[1]
    );

    // B(3, 3) = Γ(3)Γ(3)/Γ(6) = 2*2/120 = 1/30
    assert!(
        (result[2] - 1.0 / 30.0).abs() < 1e-10,
        "B(3, 3) should be 1/30, got {}",
        result[2]
    );
}

/// Test beta basic f32
#[test]
fn test_beta_simd_basic_f32() {
    let a = array![1.0_f32, 2.0, 3.0];
    let b = array![1.0_f32, 2.0, 3.0];
    let result = beta_simd(&a.view(), &b.view());

    assert!(
        (result[0] - 1.0).abs() < 1e-5,
        "B(1, 1) should be 1, got {}",
        result[0]
    );
    assert!(
        (result[1] - 1.0 / 6.0).abs() < 1e-5,
        "B(2, 2) should be 1/6, got {}",
        result[1]
    );
}

/// Test beta empty array
#[test]
fn test_beta_simd_empty() {
    let a: Array1<f64> = array![];
    let b: Array1<f64> = array![];
    let result = beta_simd(&a.view(), &b.view());
    assert_eq!(result.len(), 0, "Empty input should give empty output");
}

/// Test beta large array (SIMD path)
#[test]
fn test_beta_simd_large_array() {
    let n = 1000;
    let a: Array1<f64> = Array1::linspace(1.0, 5.0, n);
    let b: Array1<f64> = Array1::linspace(1.0, 5.0, n);
    let result = beta_simd(&a.view(), &b.view());

    assert_eq!(result.len(), n);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite() && v > 0.0,
            "Beta should be finite positive at index {}, got {}",
            i,
            v
        );
    }
}

/// Test beta symmetry B(a, b) = B(b, a)
#[test]
fn test_beta_simd_symmetry() {
    let a = array![1.5_f64, 2.5, 3.5, 0.5];
    let b = array![2.0_f64, 1.0, 4.0, 3.0];

    let result1 = beta_simd(&a.view(), &b.view());
    let result2 = beta_simd(&b.view(), &a.view());

    for i in 0..a.len() {
        assert!(
            (result1[i] - result2[i]).abs() < 1e-10,
            "B(a, b) should equal B(b, a) at index {}",
            i
        );
    }
}

/// Test beta special value B(0.5, 0.5) = π
#[test]
fn test_beta_simd_half_half() {
    let a = array![0.5_f64];
    let b = array![0.5_f64];
    let result = beta_simd(&a.view(), &b.view());

    assert!(
        (result[0] - std::f64::consts::PI).abs() < 1e-8,
        "B(0.5, 0.5) should be π, got {}",
        result[0]
    );
}

/// Test beta B(a, 1) = 1/a
#[test]
fn test_beta_simd_a_one() {
    let a = array![2.0_f64, 3.0, 4.0, 5.0];
    let b = array![1.0_f64, 1.0, 1.0, 1.0];
    let result = beta_simd(&a.view(), &b.view());

    for i in 0..a.len() {
        let expected = 1.0 / a[i];
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "B({}, 1) should be 1/{} = {}, got {}",
            a[i],
            a[i],
            expected,
            result[i]
        );
    }
}

/// Test beta for Beta distribution use case
#[test]
fn test_beta_simd_distribution_use_case() {
    // Beta distribution parameters
    let alpha = array![2.0_f64, 5.0, 1.0, 0.5];
    let beta_param = array![5.0_f64, 2.0, 1.0, 0.5];
    let beta_val = beta_simd(&alpha.view(), &beta_param.view());

    // All beta values should be positive for valid parameters
    for (i, &b) in beta_val.iter().enumerate() {
        assert!(
            b > 0.0 && b.is_finite(),
            "Beta function should be positive for valid params, got {} at index {}",
            b,
            i
        );
    }
}

// -----------------------------------------------------------------------------
// ln_beta (Log-Beta Function) Tests
// -----------------------------------------------------------------------------

/// Test ln_beta basic f64
#[test]
fn test_ln_beta_simd_basic_f64() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![1.0_f64, 2.0, 3.0];
    let result = ln_beta_simd(&a.view(), &b.view());

    // ln(B(1, 1)) = ln(1) = 0
    assert!(
        result[0].abs() < 1e-10,
        "ln(B(1, 1)) should be 0, got {}",
        result[0]
    );

    // ln(B(2, 2)) = ln(1/6)
    let expected_ln_b22 = (1.0_f64 / 6.0).ln();
    assert!(
        (result[1] - expected_ln_b22).abs() < 1e-10,
        "ln(B(2, 2)) should be ln(1/6), got {}",
        result[1]
    );
}

/// Test ln_beta basic f32
#[test]
fn test_ln_beta_simd_basic_f32() {
    let a = array![1.0_f32, 2.0];
    let b = array![1.0_f32, 2.0];
    let result = ln_beta_simd(&a.view(), &b.view());

    assert!(
        result[0].abs() < 1e-5,
        "ln(B(1, 1)) should be 0, got {}",
        result[0]
    );
}

/// Test ln_beta empty array
#[test]
fn test_ln_beta_simd_empty() {
    let a: Array1<f64> = array![];
    let b: Array1<f64> = array![];
    let result = ln_beta_simd(&a.view(), &b.view());
    assert_eq!(result.len(), 0, "Empty input should give empty output");
}

/// Test ln_beta large array (SIMD path)
#[test]
fn test_ln_beta_simd_large_array() {
    let n = 1000;
    let a: Array1<f64> = Array1::linspace(1.0, 10.0, n);
    let b: Array1<f64> = Array1::linspace(1.0, 10.0, n);
    let result = ln_beta_simd(&a.view(), &b.view());

    assert_eq!(result.len(), n);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "ln(Beta) should be finite at index {}, got {}",
            i,
            v
        );
    }
}

/// Test ln_beta symmetry
#[test]
fn test_ln_beta_simd_symmetry() {
    let a = array![1.5_f64, 2.5, 3.5];
    let b = array![2.0_f64, 1.0, 4.0];

    let result1 = ln_beta_simd(&a.view(), &b.view());
    let result2 = ln_beta_simd(&b.view(), &a.view());

    for i in 0..a.len() {
        assert!(
            (result1[i] - result2[i]).abs() < 1e-10,
            "ln(B(a, b)) should equal ln(B(b, a)) at index {}",
            i
        );
    }
}

/// Test ln_beta consistency with beta: exp(ln_beta) = beta
#[test]
fn test_ln_beta_simd_consistency() {
    let a = array![1.0_f64, 2.0, 3.0, 0.5, 1.5];
    let b = array![1.0_f64, 2.0, 3.0, 0.5, 2.5];

    let ln_beta_val = ln_beta_simd(&a.view(), &b.view());
    let beta_val = beta_simd(&a.view(), &b.view());

    for i in 0..a.len() {
        let exp_ln_beta = ln_beta_val[i].exp();
        assert!(
            (exp_ln_beta - beta_val[i]).abs() < 1e-10,
            "exp(ln(B(a,b))) should equal B(a,b), got {} vs {} at index {}",
            exp_ln_beta,
            beta_val[i],
            i
        );
    }
}

/// Test ln_beta numerical stability for large arguments
#[test]
fn test_ln_beta_simd_large_args() {
    // For large arguments, beta would overflow but ln_beta should be stable
    let a = array![100.0_f64, 200.0, 500.0];
    let b = array![100.0_f64, 200.0, 500.0];
    let result = ln_beta_simd(&a.view(), &b.view());

    // Results should be finite (very negative for large symmetric args)
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "ln(Beta) should be finite for large args at index {}, got {}",
            i,
            v
        );
        assert!(
            v < 0.0,
            "ln(Beta) should be negative for large args, got {}",
            v
        );
    }
}

/// Test ln_beta for Bayesian inference use case
#[test]
fn test_ln_beta_simd_bayesian_use_case() {
    // Common Beta distribution parameters for A/B testing
    // Prior: Beta(1, 1) = uniform
    // Posterior after observations: Beta(alpha + successes, beta + failures)
    let alpha = array![1.0_f64, 10.0, 50.0, 100.0]; // Prior + successes
    let beta_param = array![1.0_f64, 10.0, 50.0, 100.0]; // Prior + failures

    let ln_beta_val = ln_beta_simd(&alpha.view(), &beta_param.view());

    // All log-beta values should be finite
    for (i, &lb) in ln_beta_val.iter().enumerate() {
        assert!(
            lb.is_finite(),
            "ln(Beta) should be finite for Bayesian params at index {}",
            i
        );
    }

    // ln(Beta) should decrease as parameters increase (normalizing constant gets smaller)
    for i in 0..ln_beta_val.len() - 1 {
        assert!(
            ln_beta_val[i] > ln_beta_val[i + 1],
            "ln(Beta) should decrease as params increase"
        );
    }
}

// ============================================================================
// Phase 61: SIMD lerp (linear interpolation) tests
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::lerp_simd;
use scirs2_core::ndarray_ext::elementwise::smoothstep_simd;

/// Test lerp basic f32 correctness
#[test]
fn test_lerp_simd_f32_basic() {
    let a = array![0.0_f32, 10.0, -5.0, 100.0];
    let b = array![10.0_f32, 20.0, 5.0, 200.0];

    // t = 0: should return a
    let result = lerp_simd(&a.view(), &b.view(), 0.0_f32);
    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 10.0).abs() < 1e-6);
    assert!((result[2] - (-5.0)).abs() < 1e-6);
    assert!((result[3] - 100.0).abs() < 1e-6);

    // t = 1: should return b
    let result = lerp_simd(&a.view(), &b.view(), 1.0_f32);
    assert!((result[0] - 10.0).abs() < 1e-6);
    assert!((result[1] - 20.0).abs() < 1e-6);
    assert!((result[2] - 5.0).abs() < 1e-6);
    assert!((result[3] - 200.0).abs() < 1e-6);

    // t = 0.5: should return midpoint
    let result = lerp_simd(&a.view(), &b.view(), 0.5_f32);
    assert!((result[0] - 5.0).abs() < 1e-6);
    assert!((result[1] - 15.0).abs() < 1e-6);
    assert!((result[2] - 0.0).abs() < 1e-6);
    assert!((result[3] - 150.0).abs() < 1e-6);
}

/// Test lerp basic f64 correctness
#[test]
fn test_lerp_simd_f64_basic() {
    let a = array![0.0_f64, 100.0, -50.0];
    let b = array![100.0_f64, 200.0, 50.0];

    // t = 0.25
    let result = lerp_simd(&a.view(), &b.view(), 0.25_f64);
    assert!((result[0] - 25.0).abs() < 1e-14);
    assert!((result[1] - 125.0).abs() < 1e-14);
    assert!((result[2] - (-25.0)).abs() < 1e-14);

    // t = 0.75
    let result = lerp_simd(&a.view(), &b.view(), 0.75_f64);
    assert!((result[0] - 75.0).abs() < 1e-14);
    assert!((result[1] - 175.0).abs() < 1e-14);
    assert!((result[2] - 25.0).abs() < 1e-14);
}

/// Test lerp empty array
#[test]
fn test_lerp_simd_empty() {
    let a: Array1<f64> = array![];
    let b: Array1<f64> = array![];
    let result = lerp_simd(&a.view(), &b.view(), 0.5_f64);
    assert_eq!(result.len(), 0);
}

/// Test lerp large array (SIMD path)
#[test]
fn test_lerp_simd_large_array() {
    let n = 10000;
    let a = Array1::from_vec((0..n).map(|i| i as f64).collect());
    let b = Array1::from_vec((0..n).map(|i| (i * 2) as f64).collect());

    let result = lerp_simd(&a.view(), &b.view(), 0.5_f64);
    assert_eq!(result.len(), n);

    // Check a few values: lerp(i, 2*i, 0.5) = i + 0.5*(2i - i) = i + 0.5*i = 1.5*i
    for i in [0, 100, 1000, 5000, 9999] {
        let expected = 1.5 * i as f64;
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "lerp[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test lerp extrapolation (t outside [0, 1])
#[test]
fn test_lerp_simd_extrapolation() {
    let a = array![0.0_f64, 0.0];
    let b = array![10.0_f64, 20.0];

    // t = -1: extrapolate backward
    let result = lerp_simd(&a.view(), &b.view(), -1.0_f64);
    assert!((result[0] - (-10.0)).abs() < 1e-14);
    assert!((result[1] - (-20.0)).abs() < 1e-14);

    // t = 2: extrapolate forward
    let result = lerp_simd(&a.view(), &b.view(), 2.0_f64);
    assert!((result[0] - 20.0).abs() < 1e-14);
    assert!((result[1] - 40.0).abs() < 1e-14);
}

/// Test lerp property: lerp(a, b, t) = a + t*(b-a)
#[test]
fn test_lerp_simd_formula_verification() {
    let a = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let b = array![11.0_f64, 12.0, 13.0, 14.0, 15.0];
    let t = 0.3_f64;

    let result = lerp_simd(&a.view(), &b.view(), t);

    for i in 0..a.len() {
        let expected = a[i] + t * (b[i] - a[i]);
        assert!(
            (result[i] - expected).abs() < 1e-14,
            "Formula verification failed at index {}",
            i
        );
    }
}

/// Test lerp with same values (a == b)
#[test]
fn test_lerp_simd_same_values() {
    let a = array![5.0_f64, 5.0, 5.0];
    let b = array![5.0_f64, 5.0, 5.0];

    for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let result = lerp_simd(&a.view(), &b.view(), t);
        for i in 0..a.len() {
            assert!(
                (result[i] - 5.0).abs() < 1e-14,
                "lerp of identical values should be the same value"
            );
        }
    }
}

/// Test lerp animation blending use case
#[test]
fn test_lerp_simd_animation_blending() {
    // Simulating position interpolation between keyframes
    let keyframe_0 = array![0.0_f64, 0.0, 0.0]; // Position at t=0
    let keyframe_1 = array![10.0_f64, 5.0, 2.0]; // Position at t=1

    // Interpolate at various animation times
    for time in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
        let position = lerp_simd(&keyframe_0.view(), &keyframe_1.view(), time);
        for i in 0..3 {
            let expected = keyframe_0[i] + time * (keyframe_1[i] - keyframe_0[i]);
            assert!(
                (position[i] - expected).abs() < 1e-14,
                "Animation interpolation at t={} failed",
                time
            );
        }
    }
}

// ============================================================================
// Phase 61: SIMD smoothstep tests
// ============================================================================

/// Test smoothstep basic f32 correctness
#[test]
fn test_smoothstep_simd_f32_basic() {
    let x = array![0.0_f32, 0.25, 0.5, 0.75, 1.0];

    let result = smoothstep_simd(0.0_f32, 1.0_f32, &x.view());

    // smoothstep(0) = 0
    assert!((result[0] - 0.0).abs() < 1e-6);
    // smoothstep(0.5) = 0.5 (symmetric about midpoint)
    assert!((result[2] - 0.5).abs() < 1e-6);
    // smoothstep(1) = 1
    assert!((result[4] - 1.0).abs() < 1e-6);

    // Values should be monotonically increasing
    for i in 0..result.len() - 1 {
        assert!(result[i] <= result[i + 1], "smoothstep should be monotonic");
    }
}

/// Test smoothstep basic f64 correctness
#[test]
fn test_smoothstep_simd_f64_basic() {
    let x = array![0.0_f64, 0.25, 0.5, 0.75, 1.0];

    let result = smoothstep_simd(0.0_f64, 1.0_f64, &x.view());

    // Verify the Hermite polynomial: 3t² - 2t³
    for i in 0..x.len() {
        let t = x[i];
        let expected = t * t * (3.0 - 2.0 * t);
        assert!(
            (result[i] - expected).abs() < 1e-14,
            "smoothstep[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test smoothstep empty array
#[test]
fn test_smoothstep_simd_empty() {
    let x: Array1<f64> = array![];
    let result = smoothstep_simd(0.0_f64, 1.0_f64, &x.view());
    assert_eq!(result.len(), 0);
}

/// Test smoothstep clamping at edges
#[test]
fn test_smoothstep_simd_clamping() {
    let x = array![-1.0_f64, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];

    let result = smoothstep_simd(0.0_f64, 1.0_f64, &x.view());

    // Values below edge0 should be 0
    assert!((result[0] - 0.0).abs() < 1e-14); // x = -1
    assert!((result[1] - 0.0).abs() < 1e-14); // x = -0.5
    assert!((result[2] - 0.0).abs() < 1e-14); // x = 0

    // Values above edge1 should be 1
    assert!((result[4] - 1.0).abs() < 1e-14); // x = 1
    assert!((result[5] - 1.0).abs() < 1e-14); // x = 1.5
    assert!((result[6] - 1.0).abs() < 1e-14); // x = 2
}

/// Test smoothstep with custom edges
#[test]
fn test_smoothstep_simd_custom_edges() {
    let x = array![0.0_f64, 2.5, 5.0, 7.5, 10.0];

    // Edges at [0, 10]
    let result = smoothstep_simd(0.0_f64, 10.0_f64, &x.view());

    // x=0: t=0, smoothstep=0
    assert!((result[0] - 0.0).abs() < 1e-14);
    // x=5: t=0.5, smoothstep=0.5
    assert!((result[2] - 0.5).abs() < 1e-14);
    // x=10: t=1, smoothstep=1
    assert!((result[4] - 1.0).abs() < 1e-14);
}

/// Test smoothstep large array (SIMD path)
#[test]
fn test_smoothstep_simd_large_array() {
    let n = 10000;
    let x = Array1::linspace(0.0, 1.0, n);

    let result = smoothstep_simd(0.0_f64, 1.0_f64, &x.view());
    assert_eq!(result.len(), n);

    // Verify first, middle, and last values
    assert!((result[0] - 0.0).abs() < 1e-14);
    let mid = n / 2;
    let t_mid = x[mid];
    let expected_mid = t_mid * t_mid * (3.0 - 2.0 * t_mid);
    assert!((result[mid] - expected_mid).abs() < 1e-10);
    assert!((result[n - 1] - 1.0).abs() < 1e-14);
}

/// Test smoothstep property: derivative at edges is zero
#[test]
fn test_smoothstep_simd_derivative_at_edges() {
    // The derivative of 3t² - 2t³ is 6t - 6t² = 6t(1-t)
    // At t=0: derivative = 0, at t=1: derivative = 0

    // Approximate derivative at boundaries with very small epsilon
    let eps = 1e-8_f64;
    let x_near_0 = array![0.0, eps];
    let x_near_1 = array![1.0 - eps, 1.0];

    let result_0 = smoothstep_simd(0.0_f64, 1.0_f64, &x_near_0.view());
    let result_1 = smoothstep_simd(0.0_f64, 1.0_f64, &x_near_1.view());

    // Derivative near 0 should be very small
    let deriv_near_0 = (result_0[1] - result_0[0]) / eps;
    assert!(
        deriv_near_0.abs() < 1e-5,
        "Derivative at edge0 should be near 0, got {}",
        deriv_near_0
    );

    // Derivative near 1 should be very small
    let deriv_near_1 = (result_1[1] - result_1[0]) / eps;
    assert!(
        deriv_near_1.abs() < 1e-5,
        "Derivative at edge1 should be near 0, got {}",
        deriv_near_1
    );
}

/// Test smoothstep with reversed edges (edge0 > edge1)
#[test]
fn test_smoothstep_simd_reversed_edges() {
    let x = array![0.0_f64, 0.5, 1.0];

    // Reversed: edge0=1, edge1=0
    let result = smoothstep_simd(1.0_f64, 0.0_f64, &x.view());

    // With reversed edges, x=0 maps to t=(0-1)/(0-1)=1, x=1 maps to t=0
    // smoothstep(1.0, 0.0, 0) should be 1 (since x < edge1)
    // smoothstep(1.0, 0.0, 1) should be 0 (since x > edge0)
    assert!((result[0] - 1.0).abs() < 1e-14); // x=0 gives smoothstep=1
    assert!((result[2] - 0.0).abs() < 1e-14); // x=1 gives smoothstep=0
}

/// Test smoothstep shader use case: soft shadow transition
#[test]
fn test_smoothstep_simd_shadow_transition() {
    // Simulating soft shadow from light source
    // Shadow starts at distance 2.0, fully dark at 5.0
    let distances = array![0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let shadow_start = 2.0_f64;
    let shadow_end = 5.0_f64;

    let shadow_factor = smoothstep_simd(shadow_start, shadow_end, &distances.view());

    // Before shadow_start: no shadow (0)
    assert!((shadow_factor[0] - 0.0).abs() < 1e-14);
    assert!((shadow_factor[1] - 0.0).abs() < 1e-14);
    assert!((shadow_factor[2] - 0.0).abs() < 1e-14);

    // After shadow_end: full shadow (1)
    assert!((shadow_factor[5] - 1.0).abs() < 1e-14);
    assert!((shadow_factor[6] - 1.0).abs() < 1e-14);
    assert!((shadow_factor[7] - 1.0).abs() < 1e-14);

    // Middle region: smooth transition
    assert!(shadow_factor[3] > 0.0 && shadow_factor[3] < 1.0);
    assert!(shadow_factor[4] > 0.0 && shadow_factor[4] < 1.0);
}

// ============================================================================
// Phase 62: SIMD hypot (hypotenuse) tests
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::copysign_simd;
use scirs2_core::ndarray_ext::elementwise::hypot_simd;

/// Test hypot basic f32 correctness with Pythagorean triples
#[test]
fn test_hypot_simd_f32_basic() {
    // Well-known Pythagorean triples
    let x = array![3.0_f32, 5.0, 8.0, 7.0];
    let y = array![4.0_f32, 12.0, 15.0, 24.0];

    let result = hypot_simd(&x.view(), &y.view());

    assert!((result[0] - 5.0).abs() < 1e-5); // 3-4-5
    assert!((result[1] - 13.0).abs() < 1e-5); // 5-12-13
    assert!((result[2] - 17.0).abs() < 1e-5); // 8-15-17
    assert!((result[3] - 25.0).abs() < 1e-5); // 7-24-25
}

/// Test hypot basic f64 correctness
#[test]
fn test_hypot_simd_f64_basic() {
    let x = array![3.0_f64, 5.0, 8.0, 7.0];
    let y = array![4.0_f64, 12.0, 15.0, 24.0];

    let result = hypot_simd(&x.view(), &y.view());

    assert!((result[0] - 5.0).abs() < 1e-14);
    assert!((result[1] - 13.0).abs() < 1e-14);
    assert!((result[2] - 17.0).abs() < 1e-14);
    assert!((result[3] - 25.0).abs() < 1e-14);
}

/// Test hypot empty array
#[test]
fn test_hypot_simd_empty() {
    let x: Array1<f64> = array![];
    let y: Array1<f64> = array![];
    let result = hypot_simd(&x.view(), &y.view());
    assert_eq!(result.len(), 0);
}

/// Test hypot large array (SIMD path)
#[test]
fn test_hypot_simd_large_array() {
    let n = 10000;
    // Generate values that form 3-4-5 scaled triangles
    let x = Array1::from_vec((0..n).map(|i| 3.0 * i as f64).collect());
    let y = Array1::from_vec((0..n).map(|i| 4.0 * i as f64).collect());

    let result = hypot_simd(&x.view(), &y.view());
    assert_eq!(result.len(), n);

    // Check: hypot(3i, 4i) = 5i
    for i in [0, 100, 1000, 5000, 9999] {
        let expected = 5.0 * i as f64;
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "hypot[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test hypot with zeros
#[test]
fn test_hypot_simd_zeros() {
    let x = array![0.0_f64, 5.0, 0.0];
    let y = array![3.0_f64, 0.0, 0.0];

    let result = hypot_simd(&x.view(), &y.view());

    assert!((result[0] - 3.0).abs() < 1e-14); // hypot(0, 3) = 3
    assert!((result[1] - 5.0).abs() < 1e-14); // hypot(5, 0) = 5
    assert!((result[2] - 0.0).abs() < 1e-14); // hypot(0, 0) = 0
}

/// Test hypot with negative values
#[test]
fn test_hypot_simd_negative_values() {
    let x = array![-3.0_f64, -5.0, 3.0, -3.0];
    let y = array![4.0_f64, -12.0, -4.0, -4.0];

    let result = hypot_simd(&x.view(), &y.view());

    // hypot should always return positive values regardless of input signs
    assert!((result[0] - 5.0).abs() < 1e-14);
    assert!((result[1] - 13.0).abs() < 1e-14);
    assert!((result[2] - 5.0).abs() < 1e-14);
    assert!((result[3] - 5.0).abs() < 1e-14);
}

/// Test hypot overflow protection
#[test]
fn test_hypot_simd_overflow_protection() {
    // Large values that would overflow with naive x*x + y*y
    let large = 1e150_f64;
    let x = array![large, large];
    let y = array![large, large];

    let result = hypot_simd(&x.view(), &y.view());

    // Result should be finite, approximately large * sqrt(2)
    assert!(result[0].is_finite(), "hypot should handle large values");
    let expected = large * std::f64::consts::SQRT_2;
    let relative_error = ((result[0] - expected) / expected).abs();
    assert!(
        relative_error < 1e-10,
        "hypot({}, {}) = {}, expected {}",
        large,
        large,
        result[0],
        expected
    );
}

/// Test hypot underflow protection
#[test]
fn test_hypot_simd_underflow_protection() {
    // Small values that would underflow with naive x*x + y*y
    let small = 1e-200_f64;
    let x = array![small, small];
    let y = array![small, small];

    let result = hypot_simd(&x.view(), &y.view());

    // Result should be finite and non-zero
    assert!(result[0].is_finite(), "hypot should handle small values");
    assert!(result[0] > 0.0, "hypot should not underflow to zero");
    let expected = small * std::f64::consts::SQRT_2;
    let relative_error = ((result[0] - expected) / expected).abs();
    assert!(
        relative_error < 1e-10,
        "hypot({}, {}) = {}, expected {}",
        small,
        small,
        result[0],
        expected
    );
}

/// Test hypot property: hypot(x, 0) = |x|
#[test]
fn test_hypot_simd_identity_property() {
    let x = array![5.0_f64, -5.0, 0.0, std::f64::consts::PI];
    let zeros = Array1::zeros(4);

    let result = hypot_simd(&x.view(), &zeros.view());

    for i in 0..x.len() {
        assert!(
            (result[i] - x[i].abs()).abs() < 1e-14,
            "hypot(x, 0) should equal |x|"
        );
    }
}

/// Test hypot distance calculation use case
#[test]
fn test_hypot_simd_distance_calculation() {
    // Calculate distances from origin to various 2D points
    let points_x = array![1.0_f64, 1.0, 2.0, 3.0];
    let points_y = array![0.0_f64, 1.0, 2.0, 4.0];

    let distances = hypot_simd(&points_x.view(), &points_y.view());

    assert!((distances[0] - 1.0).abs() < 1e-14); // Point (1,0)
    assert!((distances[1] - std::f64::consts::SQRT_2).abs() < 1e-14); // Point (1,1)
    assert!((distances[2] - (2.0 * std::f64::consts::SQRT_2)).abs() < 1e-14); // Point (2,2)
    assert!((distances[3] - 5.0).abs() < 1e-14); // Point (3,4) -> 3-4-5 triangle
}

// ============================================================================
// Phase 62: SIMD copysign tests
// ============================================================================

/// Test copysign basic f32 correctness
#[test]
fn test_copysign_simd_f32_basic() {
    let x = array![1.0_f32, -2.0, 3.0, -4.0];
    let y = array![-1.0_f32, 1.0, 1.0, -1.0];

    let result = copysign_simd(&x.view(), &y.view());

    assert!((result[0] - (-1.0)).abs() < 1e-6); // |1| with sign of -1 = -1
    assert!((result[1] - 2.0).abs() < 1e-6); // |-2| with sign of 1 = 2
    assert!((result[2] - 3.0).abs() < 1e-6); // |3| with sign of 1 = 3
    assert!((result[3] - (-4.0)).abs() < 1e-6); // |-4| with sign of -1 = -4
}

/// Test copysign basic f64 correctness
#[test]
fn test_copysign_simd_f64_basic() {
    let x = array![1.0_f64, -2.0, 3.0, -4.0];
    let y = array![-1.0_f64, 1.0, 1.0, -1.0];

    let result = copysign_simd(&x.view(), &y.view());

    assert!((result[0] - (-1.0)).abs() < 1e-14);
    assert!((result[1] - 2.0).abs() < 1e-14);
    assert!((result[2] - 3.0).abs() < 1e-14);
    assert!((result[3] - (-4.0)).abs() < 1e-14);
}

/// Test copysign empty array
#[test]
fn test_copysign_simd_empty() {
    let x: Array1<f64> = array![];
    let y: Array1<f64> = array![];
    let result = copysign_simd(&x.view(), &y.view());
    assert_eq!(result.len(), 0);
}

/// Test copysign large array (SIMD path)
#[test]
fn test_copysign_simd_large_array() {
    let n = 10000;
    let x = Array1::from_vec((0..n).map(|i| i as f64 + 1.0).collect());
    let y = Array1::from_vec(
        (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect(),
    );

    let result = copysign_simd(&x.view(), &y.view());
    assert_eq!(result.len(), n);

    // Check: even indices positive, odd indices negative
    for i in [0, 101, 1000, 5001, 9998] {
        if i % 2 == 0 {
            assert!(result[i] > 0.0, "Even index {} should be positive", i);
        } else {
            assert!(result[i] < 0.0, "Odd index {} should be negative", i);
        }
        assert!(
            (result[i].abs() - (i as f64 + 1.0)).abs() < 1e-10,
            "Magnitude should be preserved"
        );
    }
}

/// Test copysign with zeros
#[test]
fn test_copysign_simd_zeros() {
    let x = array![5.0_f64, 0.0, -3.0];
    let y = array![0.0_f64, -1.0, 0.0];

    let result = copysign_simd(&x.view(), &y.view());

    // Zero has positive sign in IEEE 754
    assert!((result[0] - 5.0).abs() < 1e-14); // 5 with sign of +0 = 5
                                              // copysign(0, -1) should be -0
    assert!(result[1] == 0.0 || result[1] == -0.0);
    assert!((result[2] - 3.0).abs() < 1e-14); // -3 with sign of +0 = 3
}

/// Test copysign property: |copysign(x, y)| = |x|
#[test]
fn test_copysign_simd_magnitude_preservation() {
    let x = array![1.5_f64, -2.5, 3.5, -4.5];
    let y = array![-1.0_f64, -1.0, 1.0, 1.0];

    let result = copysign_simd(&x.view(), &y.view());

    for i in 0..x.len() {
        assert!(
            (result[i].abs() - x[i].abs()).abs() < 1e-14,
            "copysign should preserve magnitude"
        );
    }
}

/// Test copysign property: sign(copysign(x, y)) = sign(y)
#[test]
fn test_copysign_simd_sign_transfer() {
    let x = array![1.0_f64, -2.0, 3.0, -4.0, 5.0];
    let y = array![-10.0_f64, 20.0, -30.0, 40.0, -50.0];

    let result = copysign_simd(&x.view(), &y.view());

    for i in 0..x.len() {
        let result_sign = if result[i] >= 0.0 { 1 } else { -1 };
        let y_sign = if y[i] >= 0.0 { 1 } else { -1 };
        assert_eq!(result_sign, y_sign, "Sign should match y at index {}", i);
    }
}

/// Test copysign use case: implementing absolute value
#[test]
fn test_copysign_simd_abs_implementation() {
    let x = array![-5.0_f64, 3.0, -2.0, 0.0, -0.0];
    let ones = array![1.0_f64, 1.0, 1.0, 1.0, 1.0];

    let result = copysign_simd(&x.view(), &ones.view());

    // copysign(x, 1) = |x|
    for i in 0..x.len() {
        assert!(
            (result[i] - x[i].abs()).abs() < 1e-14,
            "copysign(x, 1) should equal |x|"
        );
    }
}

/// Test copysign use case: implementing negation
#[test]
fn test_copysign_simd_negation_implementation() {
    let x = array![5.0_f64, -3.0, 2.0];
    let neg_ones = array![-1.0_f64, -1.0, -1.0];

    let result = copysign_simd(&x.view(), &neg_ones.view());

    // copysign(x, -1) = -|x|
    for i in 0..x.len() {
        assert!(
            (result[i] - (-x[i].abs())).abs() < 1e-14,
            "copysign(x, -1) should equal -|x|"
        );
    }
}

// ============================================================================
// Phase 63: SIMD smootherstep (Ken Perlin's improved smoothstep) tests
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::smootherstep_simd;

/// Test smootherstep basic f32 correctness
#[test]
fn test_smootherstep_simd_f32_basic() {
    let x = array![0.0_f32, 0.25, 0.5, 0.75, 1.0];

    let result = smootherstep_simd(0.0_f32, 1.0_f32, &x.view());

    // smootherstep(0) = 0
    assert!((result[0] - 0.0).abs() < 1e-6);
    // smootherstep(0.5) = 0.5 (symmetric about midpoint)
    assert!((result[2] - 0.5).abs() < 1e-6);
    // smootherstep(1) = 1
    assert!((result[4] - 1.0).abs() < 1e-6);

    // Values should be monotonically increasing
    for i in 0..result.len() - 1 {
        assert!(
            result[i] <= result[i + 1],
            "smootherstep should be monotonic"
        );
    }
}

/// Test smootherstep basic f64 correctness
#[test]
fn test_smootherstep_simd_f64_basic() {
    let x = array![0.0_f64, 0.25, 0.5, 0.75, 1.0];

    let result = smootherstep_simd(0.0_f64, 1.0_f64, &x.view());

    // Verify the Perlin polynomial: 6t⁵ - 15t⁴ + 10t³
    for i in 0..x.len() {
        let t = x[i];
        let t3 = t * t * t;
        let expected = t3 * (t * (t * 6.0 - 15.0) + 10.0);
        assert!(
            (result[i] - expected).abs() < 1e-14,
            "smootherstep[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test smootherstep empty array
#[test]
fn test_smootherstep_simd_empty() {
    let x: Array1<f64> = array![];
    let result = smootherstep_simd(0.0_f64, 1.0_f64, &x.view());
    assert_eq!(result.len(), 0);
}

/// Test smootherstep clamping at edges
#[test]
fn test_smootherstep_simd_clamping() {
    let x = array![-1.0_f64, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];

    let result = smootherstep_simd(0.0_f64, 1.0_f64, &x.view());

    // Values below edge0 should be 0
    assert!((result[0] - 0.0).abs() < 1e-14); // x = -1
    assert!((result[1] - 0.0).abs() < 1e-14); // x = -0.5
    assert!((result[2] - 0.0).abs() < 1e-14); // x = 0

    // Values above edge1 should be 1
    assert!((result[4] - 1.0).abs() < 1e-14); // x = 1
    assert!((result[5] - 1.0).abs() < 1e-14); // x = 1.5
    assert!((result[6] - 1.0).abs() < 1e-14); // x = 2
}

/// Test smootherstep large array (SIMD path)
#[test]
fn test_smootherstep_simd_large_array() {
    let n = 10000;
    let x = Array1::linspace(0.0, 1.0, n);

    let result = smootherstep_simd(0.0_f64, 1.0_f64, &x.view());
    assert_eq!(result.len(), n);

    // Verify first, middle, and last values
    assert!((result[0] - 0.0).abs() < 1e-14);
    let mid = n / 2;
    let t_mid = x[mid];
    let t3 = t_mid * t_mid * t_mid;
    let expected_mid = t3 * (t_mid * (t_mid * 6.0 - 15.0) + 10.0);
    assert!((result[mid] - expected_mid).abs() < 1e-10);
    assert!((result[n - 1] - 1.0).abs() < 1e-14);
}

/// Test smootherstep property: first derivative at edges is zero
#[test]
fn test_smootherstep_simd_first_derivative_at_edges() {
    // The first derivative of 6t⁵ - 15t⁴ + 10t³ is 30t⁴ - 60t³ + 30t² = 30t²(t-1)²
    // At t=0 and t=1, the derivative is 0

    let eps = 1e-8_f64;
    let x_near_0 = array![0.0, eps];
    let x_near_1 = array![1.0 - eps, 1.0];

    let result_0 = smootherstep_simd(0.0_f64, 1.0_f64, &x_near_0.view());
    let result_1 = smootherstep_simd(0.0_f64, 1.0_f64, &x_near_1.view());

    // Derivative near 0 should be very small
    let deriv_near_0 = (result_0[1] - result_0[0]) / eps;
    assert!(
        deriv_near_0.abs() < 1e-5,
        "First derivative at edge0 should be near 0, got {}",
        deriv_near_0
    );

    // Derivative near 1 should be very small
    let deriv_near_1 = (result_1[1] - result_1[0]) / eps;
    assert!(
        deriv_near_1.abs() < 1e-5,
        "First derivative at edge1 should be near 0, got {}",
        deriv_near_1
    );
}

/// Test smootherstep property: second derivative at edges is zero
#[test]
fn test_smootherstep_simd_second_derivative_at_edges() {
    // The second derivative is 120t³ - 180t² + 60t = 60t(2t-1)(t-1)
    // At t=0 and t=1, the second derivative is 0

    let eps = 1e-6_f64;
    let x = array![0.0, eps, 2.0 * eps];

    let result = smootherstep_simd(0.0_f64, 1.0_f64, &x.view());

    // Approximate second derivative at t=0 using central difference
    let d1 = (result[1] - result[0]) / eps;
    let d2 = (result[2] - result[1]) / eps;
    let second_deriv = (d2 - d1) / eps;

    assert!(
        second_deriv.abs() < 1e-2,
        "Second derivative at edge0 should be near 0, got {}",
        second_deriv
    );
}

/// Test smootherstep comparison with smoothstep (should be smoother)
#[test]
fn test_smootherstep_vs_smoothstep() {
    let x = array![0.0_f64, 0.25, 0.5, 0.75, 1.0];

    let result_smoother = smootherstep_simd(0.0_f64, 1.0_f64, &x.view());
    let result_smooth = smoothstep_simd(0.0_f64, 1.0_f64, &x.view());

    // Both should agree at 0, 0.5, and 1
    assert!((result_smoother[0] - result_smooth[0]).abs() < 1e-14);
    assert!((result_smoother[2] - result_smooth[2]).abs() < 1e-14);
    assert!((result_smoother[4] - result_smooth[4]).abs() < 1e-14);

    // At t=0.25 and t=0.75, smootherstep differs from smoothstep
    // smoothstep(0.25) = 3(0.0625) - 2(0.015625) = 0.1875 - 0.03125 = 0.15625
    // smootherstep(0.25) = 6(0.000976...) - 15(0.00390625) + 10(0.015625)
    //                    = 0.005859... - 0.058594... + 0.15625 ≈ 0.103516
    // They should be different
    assert!(
        (result_smoother[1] - result_smooth[1]).abs() > 0.01,
        "smootherstep and smoothstep should differ at t=0.25"
    );
}

/// Test smootherstep Perlin noise use case
#[test]
fn test_smootherstep_simd_perlin_noise_use_case() {
    // In Perlin noise, smootherstep is used for gradient interpolation
    // This tests the fade function behavior that Perlin noise expects

    let t_values = array![0.0_f64, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    let fade_result = smootherstep_simd(0.0_f64, 1.0_f64, &t_values.view());

    // All values should be in [0, 1]
    for (i, &v) in fade_result.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&v),
            "Perlin fade at t={} should be in [0,1], got {}",
            t_values[i],
            v
        );
    }

    // Should be strictly monotonically increasing (except at endpoints)
    for i in 1..fade_result.len() {
        assert!(
            fade_result[i] >= fade_result[i - 1],
            "Perlin fade should be monotonic"
        );
    }

    // The curve should be more "S-shaped" than linear
    // At t=0.5, smootherstep should equal exactly 0.5 (symmetry)
    assert!((fade_result[5] - 0.5).abs() < 1e-14);
}

// ============================================================================
// Phase 64-65: SIMD logaddexp and logit tests
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::logaddexp_simd;
use scirs2_core::ndarray_ext::elementwise::logit_simd;

/// Test logaddexp basic f32 correctness
#[test]
fn test_logaddexp_simd_f32_basic() {
    let a = array![0.0_f32, 1.0, 2.0, -1.0];
    let b = array![0.0_f32, 1.0, 2.0, 1.0];

    let result = logaddexp_simd(&a.view(), &b.view());

    // log(exp(0) + exp(0)) = log(2)
    assert!((result[0] - 2.0_f32.ln()).abs() < 1e-5);
    // log(exp(1) + exp(1)) = log(2) + 1
    assert!((result[1] - (2.0_f32.ln() + 1.0)).abs() < 1e-5);
    // log(exp(2) + exp(2)) = log(2) + 2
    assert!((result[2] - (2.0_f32.ln() + 2.0)).abs() < 1e-5);
}

/// Test logaddexp basic f64 correctness
#[test]
fn test_logaddexp_simd_f64_basic() {
    let a = array![0.0_f64, 1.0, 2.0, -1.0];
    let b = array![0.0_f64, 1.0, 2.0, 1.0];

    let result = logaddexp_simd(&a.view(), &b.view());

    // log(exp(0) + exp(0)) = log(2)
    assert!((result[0] - 2.0_f64.ln()).abs() < 1e-14);
    // log(exp(1) + exp(1)) = log(2) + 1
    assert!((result[1] - (2.0_f64.ln() + 1.0)).abs() < 1e-14);
    // log(exp(2) + exp(2)) = log(2) + 2
    assert!((result[2] - (2.0_f64.ln() + 2.0)).abs() < 1e-14);
}

/// Test logaddexp empty array
#[test]
fn test_logaddexp_simd_empty() {
    let a: Array1<f64> = array![];
    let b: Array1<f64> = array![];
    let result = logaddexp_simd(&a.view(), &b.view());
    assert_eq!(result.len(), 0);
}

/// Test logaddexp large array (SIMD path)
#[test]
fn test_logaddexp_simd_large_array() {
    let n = 10000;
    // Use smaller values to avoid exp() overflow in the naive verification
    let a = Array1::from_vec((0..n).map(|i| i as f64 * 0.01).collect());
    let b = Array1::from_vec((0..n).map(|i| i as f64 * 0.01 + 0.5).collect());

    let result = logaddexp_simd(&a.view(), &b.view());
    assert_eq!(result.len(), n);

    // Check a few values (keeping values small enough for naive exp())
    for i in [0, 100, 500, 1000] {
        let expected = (a[i].exp() + b[i].exp()).ln();
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "logaddexp[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }

    // For large indices, verify monotonicity and finiteness
    assert!(result[9999].is_finite());
    assert!(result[9999] > result[5000]);
}

/// Test logaddexp numerical stability with large values
#[test]
fn test_logaddexp_simd_large_values() {
    // Large positive values that would overflow with naive exp()
    let a = array![700.0_f64, 700.0, -700.0];
    let b = array![700.0_f64, 500.0, -500.0];

    let result = logaddexp_simd(&a.view(), &b.view());

    // All results should be finite
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "logaddexp should handle large values, got {} at index {}",
            v,
            i
        );
    }

    // log(exp(700) + exp(700)) = 700 + log(2)
    assert!((result[0] - (700.0 + 2.0_f64.ln())).abs() < 1e-10);
    // log(exp(700) + exp(500)) ≈ 700 (dominated by larger term)
    assert!((result[1] - 700.0).abs() < 1e-10);
}

/// Test logaddexp property: logaddexp(a, a) = a + log(2)
#[test]
fn test_logaddexp_simd_equal_inputs() {
    let a = array![0.0_f64, 1.0, -1.0, 10.0, -10.0];

    let result = logaddexp_simd(&a.view(), &a.view());

    let ln2 = 2.0_f64.ln();
    for i in 0..a.len() {
        let expected = a[i] + ln2;
        assert!(
            (result[i] - expected).abs() < 1e-14,
            "logaddexp(a, a) should be a + log(2)"
        );
    }
}

/// Test logaddexp commutativity: logaddexp(a, b) = logaddexp(b, a)
#[test]
fn test_logaddexp_simd_commutativity() {
    let a = array![1.0_f64, 2.0, -3.0, 4.0];
    let b = array![5.0_f64, -2.0, 3.0, -4.0];

    let result_ab = logaddexp_simd(&a.view(), &b.view());
    let result_ba = logaddexp_simd(&b.view(), &a.view());

    for i in 0..a.len() {
        assert!(
            (result_ab[i] - result_ba[i]).abs() < 1e-14,
            "logaddexp should be commutative"
        );
    }
}

/// Test logaddexp log-probability use case
#[test]
fn test_logaddexp_simd_log_probability() {
    // In log-probability space, logaddexp combines probabilities
    // log(P(A or B)) = logaddexp(log(P(A)), log(P(B))) when A, B are mutually exclusive

    let log_p_a = array![-1.0_f64, -2.0, -3.0]; // P(A) = exp(-1), exp(-2), exp(-3)
    let log_p_b = array![-1.0_f64, -2.0, -3.0]; // P(B) = exp(-1), exp(-2), exp(-3)

    let log_p_union = logaddexp_simd(&log_p_a.view(), &log_p_b.view());

    // P(A or B) = 2 * P(A) when P(A) = P(B), so log(P) = log(2) + log(P(A))
    let ln2 = 2.0_f64.ln();
    for i in 0..log_p_a.len() {
        let expected = log_p_a[i] + ln2;
        assert!(
            (log_p_union[i] - expected).abs() < 1e-14,
            "Log-probability combination failed"
        );
    }
}

// ============================================================================
// Logit tests
// ============================================================================

/// Test logit basic f32 correctness
#[test]
fn test_logit_simd_f32_basic() {
    let p = array![0.5_f32, 0.1, 0.9, 0.01, 0.99];

    let result = logit_simd(&p.view());

    // logit(0.5) = log(1) = 0
    assert!((result[0] - 0.0).abs() < 1e-5);
    // logit(0.1) = log(0.1/0.9)
    assert!((result[1] - (0.1_f32 / 0.9).ln()).abs() < 1e-5);
    // logit(0.9) = log(0.9/0.1)
    assert!((result[2] - (0.9_f32 / 0.1).ln()).abs() < 1e-5);
}

/// Test logit basic f64 correctness
#[test]
fn test_logit_simd_f64_basic() {
    let p = array![0.5_f64, 0.1, 0.9, 0.01, 0.99];

    let result = logit_simd(&p.view());

    // logit(0.5) = log(1) = 0
    assert!((result[0] - 0.0).abs() < 1e-14);
    // logit(0.1) = log(0.1/0.9)
    assert!((result[1] - (0.1_f64 / 0.9).ln()).abs() < 1e-14);
    // logit(0.9) = log(0.9/0.1)
    assert!((result[2] - (0.9_f64 / 0.1).ln()).abs() < 1e-14);
}

/// Test logit empty array
#[test]
fn test_logit_simd_empty() {
    let p: Array1<f64> = array![];
    let result = logit_simd(&p.view());
    assert_eq!(result.len(), 0);
}

/// Test logit large array (SIMD path)
#[test]
fn test_logit_simd_large_array() {
    let n = 10000;
    // Generate probabilities in (0.01, 0.99) range
    let p = Array1::from_vec(
        (0..n)
            .map(|i| 0.01 + 0.98 * (i as f64 / n as f64))
            .collect(),
    );

    let result = logit_simd(&p.view());
    assert_eq!(result.len(), n);

    // Check a few values
    for i in [0, 100, 1000, 5000, 9999] {
        let expected = (p[i] / (1.0 - p[i])).ln();
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "logit[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test logit edge cases
#[test]
fn test_logit_simd_edge_cases() {
    let p = array![0.0_f64, 1.0];

    let result = logit_simd(&p.view());

    // logit(0) = -∞
    assert!(result[0].is_infinite() && result[0] < 0.0);
    // logit(1) = +∞
    assert!(result[1].is_infinite() && result[1] > 0.0);
}

/// Test logit symmetry: logit(p) = -logit(1-p)
#[test]
fn test_logit_simd_symmetry() {
    let p = array![0.1_f64, 0.2, 0.3, 0.4];
    let one_minus_p = p.mapv(|x| 1.0 - x);

    let result_p = logit_simd(&p.view());
    let result_1mp = logit_simd(&one_minus_p.view());

    for i in 0..p.len() {
        assert!(
            (result_p[i] + result_1mp[i]).abs() < 1e-14,
            "logit(p) + logit(1-p) should be 0"
        );
    }
}

/// Test logit as inverse of sigmoid
#[test]
fn test_logit_simd_inverse_of_sigmoid() {
    use scirs2_core::ndarray_ext::elementwise::sigmoid_simd;

    let p = array![0.1_f64, 0.3, 0.5, 0.7, 0.9];

    // logit(p) then sigmoid should return p
    let logit_p = logit_simd(&p.view());
    let sigmoid_logit_p = sigmoid_simd(&logit_p.view());

    for i in 0..p.len() {
        assert!(
            (sigmoid_logit_p[i] - p[i]).abs() < 1e-14,
            "sigmoid(logit(p)) should equal p"
        );
    }
}

/// Test logit logistic regression use case
#[test]
fn test_logit_simd_logistic_regression() {
    // In logistic regression, we model log-odds = β₀ + β₁x₁ + ...
    // Converting predicted probabilities to log-odds should give linear values

    // Simulating probabilities from a logistic model with linear log-odds
    let true_log_odds = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let probabilities = true_log_odds.mapv(|lo| 1.0 / (1.0 + (-lo).exp()));

    let recovered_log_odds = logit_simd(&probabilities.view());

    for i in 0..true_log_odds.len() {
        assert!(
            (recovered_log_odds[i] - true_log_odds[i]).abs() < 1e-14,
            "Logit should recover log-odds from probabilities"
        );
    }
}

// ============================================================================
// Phase 66-67: SIMD square, rsqrt, sincos tests
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::rsqrt_simd;
use scirs2_core::ndarray_ext::elementwise::sincos_simd;
use scirs2_core::ndarray_ext::elementwise::square_simd;
use std::f64::consts::PI;

// ============================================================================
// Square tests
// ============================================================================

/// Test square basic f32 correctness
#[test]
fn test_square_simd_f32_basic() {
    let x = array![1.0_f32, 2.0, 3.0, -4.0, 0.5];

    let result = square_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-6); // 1² = 1
    assert!((result[1] - 4.0).abs() < 1e-6); // 2² = 4
    assert!((result[2] - 9.0).abs() < 1e-6); // 3² = 9
    assert!((result[3] - 16.0).abs() < 1e-6); // (-4)² = 16
    assert!((result[4] - 0.25).abs() < 1e-6); // 0.5² = 0.25
}

/// Test square basic f64 correctness
#[test]
fn test_square_simd_f64_basic() {
    let x = array![1.0_f64, 2.0, 3.0, -4.0, 0.5];

    let result = square_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-14); // 1² = 1
    assert!((result[1] - 4.0).abs() < 1e-14); // 2² = 4
    assert!((result[2] - 9.0).abs() < 1e-14); // 3² = 9
    assert!((result[3] - 16.0).abs() < 1e-14); // (-4)² = 16
    assert!((result[4] - 0.25).abs() < 1e-14); // 0.5² = 0.25
}

/// Test square empty array
#[test]
fn test_square_simd_empty() {
    let x: Array1<f64> = array![];
    let result = square_simd(&x.view());
    assert_eq!(result.len(), 0);
}

/// Test square large array (SIMD path)
#[test]
fn test_square_simd_large_array() {
    let n = 10000;
    let x = Array1::from_vec((0..n).map(|i| (i as f64 - 5000.0) * 0.01).collect());

    let result = square_simd(&x.view());
    assert_eq!(result.len(), n);

    // Check a few values
    for i in [0, 100, 5000, 9999] {
        let expected = x[i] * x[i];
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "square[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test square special values
#[test]
fn test_square_simd_special_values() {
    let x = array![0.0_f64, -0.0, f64::INFINITY, f64::NEG_INFINITY];

    let result = square_simd(&x.view());

    // 0² = 0
    assert_eq!(result[0], 0.0);
    // (-0)² = 0
    assert_eq!(result[1], 0.0);
    // ∞² = ∞
    assert!(result[2].is_infinite() && result[2] > 0.0);
    // (-∞)² = ∞
    assert!(result[3].is_infinite() && result[3] > 0.0);
}

/// Test square always non-negative
#[test]
fn test_square_simd_non_negative() {
    let x = array![-10.0_f64, -5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 10.0];

    let result = square_simd(&x.view());

    for (i, &v) in result.iter().enumerate() {
        assert!(
            v >= 0.0,
            "square should always be non-negative, got {} at index {}",
            v,
            i
        );
    }
}

/// Test square vs pow comparison
#[test]
fn test_square_simd_vs_pow() {
    let x = array![1.5_f64, 2.5, 3.5, -4.5, 0.5];

    let result = square_simd(&x.view());
    let pow_result = x.mapv(|v| v.powf(2.0));

    for i in 0..x.len() {
        assert!(
            (result[i] - pow_result[i]).abs() < 1e-14,
            "square should match pow(x, 2)"
        );
    }
}

/// Test square MSE use case
#[test]
fn test_square_simd_mse_use_case() {
    // Computing Mean Squared Error: sum((y - y_hat)²) / n
    let y_true = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let y_pred = array![1.1_f64, 2.2, 2.9, 4.1, 4.9];

    let errors = &y_true - &y_pred;
    let squared_errors = square_simd(&errors.view());
    let mse = squared_errors.sum() / y_true.len() as f64;

    // Manual calculation: (0.1² + 0.2² + 0.1² + 0.1² + 0.1²) / 5 = 0.08 / 5 = 0.016
    let expected_mse = (0.01 + 0.04 + 0.01 + 0.01 + 0.01) / 5.0;
    assert!((mse - expected_mse).abs() < 1e-10, "MSE calculation failed");
}

// ============================================================================
// Rsqrt tests
// ============================================================================

/// Test rsqrt basic f32 correctness
#[test]
fn test_rsqrt_simd_f32_basic() {
    let x = array![1.0_f32, 4.0, 9.0, 16.0, 25.0];

    let result = rsqrt_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-6); // 1/sqrt(1) = 1
    assert!((result[1] - 0.5).abs() < 1e-6); // 1/sqrt(4) = 0.5
    assert!((result[2] - 1.0 / 3.0).abs() < 1e-6); // 1/sqrt(9) = 1/3
    assert!((result[3] - 0.25).abs() < 1e-6); // 1/sqrt(16) = 0.25
    assert!((result[4] - 0.2).abs() < 1e-6); // 1/sqrt(25) = 0.2
}

/// Test rsqrt basic f64 correctness
#[test]
fn test_rsqrt_simd_f64_basic() {
    let x = array![1.0_f64, 4.0, 9.0, 16.0, 25.0];

    let result = rsqrt_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-14); // 1/sqrt(1) = 1
    assert!((result[1] - 0.5).abs() < 1e-14); // 1/sqrt(4) = 0.5
    assert!((result[2] - 1.0 / 3.0).abs() < 1e-14); // 1/sqrt(9) = 1/3
    assert!((result[3] - 0.25).abs() < 1e-14); // 1/sqrt(16) = 0.25
    assert!((result[4] - 0.2).abs() < 1e-14); // 1/sqrt(25) = 0.2
}

/// Test rsqrt empty array
#[test]
fn test_rsqrt_simd_empty() {
    let x: Array1<f64> = array![];
    let result = rsqrt_simd(&x.view());
    assert_eq!(result.len(), 0);
}

/// Test rsqrt large array (SIMD path)
#[test]
fn test_rsqrt_simd_large_array() {
    let n = 10000;
    let x = Array1::from_vec((1..=n).map(|i| i as f64).collect());

    let result = rsqrt_simd(&x.view());
    assert_eq!(result.len(), n);

    // Check a few values
    for i in [0, 99, 999, 9999] {
        let expected = 1.0 / x[i].sqrt();
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "rsqrt[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test rsqrt special values
#[test]
fn test_rsqrt_simd_special_values() {
    let x = array![0.0_f64, -1.0, f64::INFINITY];

    let result = rsqrt_simd(&x.view());

    // 1/sqrt(0) = ∞
    assert!(result[0].is_infinite() && result[0] > 0.0);
    // 1/sqrt(-1) = NaN
    assert!(result[1].is_nan());
    // 1/sqrt(∞) = 0
    assert_eq!(result[2], 0.0);
}

/// Test rsqrt identity: x * rsqrt(x)² = 1
#[test]
fn test_rsqrt_simd_identity() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];

    let rsqrt_x = rsqrt_simd(&x.view());

    for i in 0..x.len() {
        let product = x[i] * rsqrt_x[i] * rsqrt_x[i];
        assert!(
            (product - 1.0).abs() < 1e-14,
            "x * rsqrt(x)² should be 1, got {}",
            product
        );
    }
}

/// Test rsqrt vs 1/sqrt comparison
#[test]
fn test_rsqrt_simd_vs_one_over_sqrt() {
    let x = array![0.5_f64, 1.5, 2.5, 3.5, 4.5];

    let result = rsqrt_simd(&x.view());
    let manual = x.mapv(|v| 1.0 / v.sqrt());

    for i in 0..x.len() {
        assert!(
            (result[i] - manual[i]).abs() < 1e-14,
            "rsqrt should match 1/sqrt"
        );
    }
}

/// Test rsqrt vector normalization use case
#[test]
fn test_rsqrt_simd_vector_normalization() {
    // Vector normalization: v_normalized = v * rsqrt(dot(v, v))
    // This is the primary use case in graphics

    // 3D vectors
    let vx = array![3.0_f64, 1.0, 0.0];
    let vy = array![4.0_f64, 1.0, 1.0];
    let vz = array![0.0_f64, 1.0, 0.0];

    // Compute squared magnitudes
    let mag_squared = &vx * &vx + &vy * &vy + &vz * &vz;
    // Get inverse magnitudes
    let inv_mag = rsqrt_simd(&mag_squared.view());

    // Normalize
    let nx = &vx * &inv_mag;
    let ny = &vy * &inv_mag;
    let nz = &vz * &inv_mag;

    // Check normalized vector magnitudes are 1
    for i in 0..vx.len() {
        let norm = (nx[i] * nx[i] + ny[i] * ny[i] + nz[i] * nz[i]).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-14,
            "Normalized vector magnitude should be 1, got {}",
            norm
        );
    }
}

/// Test rsqrt quaternion normalization use case
#[test]
fn test_rsqrt_simd_quaternion_normalization() {
    // Quaternion normalization: q / |q|
    // Given q = (w, x, y, z), |q|² = w² + x² + y² + z²

    let w = array![1.0_f64, 0.5, 0.707];
    let x = array![0.0_f64, 0.5, 0.0];
    let y = array![0.0_f64, 0.5, 0.707];
    let z = array![0.0_f64, 0.5, 0.0];

    let mag_squared = &w * &w + &x * &x + &y * &y + &z * &z;
    let inv_mag = rsqrt_simd(&mag_squared.view());

    let nw = &w * &inv_mag;
    let nx = &x * &inv_mag;
    let ny = &y * &inv_mag;
    let nz = &z * &inv_mag;

    // Check normalized quaternion magnitudes are 1
    for i in 0..w.len() {
        let norm_sq = nw[i] * nw[i] + nx[i] * nx[i] + ny[i] * ny[i] + nz[i] * nz[i];
        assert!(
            (norm_sq - 1.0).abs() < 1e-14,
            "Normalized quaternion magnitude² should be 1, got {}",
            norm_sq
        );
    }
}

// ============================================================================
// Sincos tests
// ============================================================================

/// Test sincos basic f32 correctness
#[test]
fn test_sincos_simd_f32_basic() {
    let x = array![
        0.0_f32,
        PI as f32 / 6.0,
        PI as f32 / 4.0,
        PI as f32 / 2.0,
        PI as f32
    ];

    let (sin_result, cos_result) = sincos_simd(&x.view());

    // sin(0) = 0, cos(0) = 1
    assert!(sin_result[0].abs() < 1e-6);
    assert!((cos_result[0] - 1.0).abs() < 1e-6);
    // sin(π/6) = 0.5, cos(π/6) = √3/2
    assert!((sin_result[1] - 0.5).abs() < 1e-5);
    // sin(π/4) = cos(π/4) = √2/2
    let sqrt2_2 = std::f32::consts::FRAC_1_SQRT_2;
    assert!((sin_result[2] - sqrt2_2).abs() < 1e-5);
    assert!((cos_result[2] - sqrt2_2).abs() < 1e-5);
    // sin(π/2) = 1, cos(π/2) = 0
    assert!((sin_result[3] - 1.0).abs() < 1e-5);
    assert!(cos_result[3].abs() < 1e-5);
}

/// Test sincos basic f64 correctness
#[test]
fn test_sincos_simd_f64_basic() {
    let x = array![0.0_f64, PI / 6.0, PI / 4.0, PI / 2.0, PI];

    let (sin_result, cos_result) = sincos_simd(&x.view());

    // sin(0) = 0, cos(0) = 1
    assert!(sin_result[0].abs() < 1e-14);
    assert!((cos_result[0] - 1.0).abs() < 1e-14);
    // sin(π/6) = 0.5
    assert!((sin_result[1] - 0.5).abs() < 1e-14);
    // sin(π/4) = cos(π/4) = √2/2
    let sqrt2_2 = std::f64::consts::FRAC_1_SQRT_2;
    assert!((sin_result[2] - sqrt2_2).abs() < 1e-14);
    assert!((cos_result[2] - sqrt2_2).abs() < 1e-14);
    // sin(π/2) = 1, cos(π/2) ≈ 0
    assert!((sin_result[3] - 1.0).abs() < 1e-14);
    assert!(cos_result[3].abs() < 1e-14);
    // sin(π) ≈ 0, cos(π) = -1
    assert!(sin_result[4].abs() < 1e-14);
    assert!((cos_result[4] + 1.0).abs() < 1e-14);
}

/// Test sincos empty array
#[test]
fn test_sincos_simd_empty() {
    let x: Array1<f64> = array![];
    let (sin_result, cos_result) = sincos_simd(&x.view());
    assert_eq!(sin_result.len(), 0);
    assert_eq!(cos_result.len(), 0);
}

/// Test sincos large array (SIMD path)
#[test]
fn test_sincos_simd_large_array() {
    let n = 10000;
    let x = Array1::from_vec((0..n).map(|i| i as f64 * 0.01).collect());

    let (sin_result, cos_result) = sincos_simd(&x.view());
    assert_eq!(sin_result.len(), n);
    assert_eq!(cos_result.len(), n);

    // Check a few values
    for i in [0, 100, 1000, 5000, 9999] {
        let expected_sin = x[i].sin();
        let expected_cos = x[i].cos();
        assert!(
            (sin_result[i] - expected_sin).abs() < 1e-10,
            "sin[{}] = {}, expected {}",
            i,
            sin_result[i],
            expected_sin
        );
        assert!(
            (cos_result[i] - expected_cos).abs() < 1e-10,
            "cos[{}] = {}, expected {}",
            i,
            cos_result[i],
            expected_cos
        );
    }
}

/// Test sincos Pythagorean identity: sin²(x) + cos²(x) = 1
#[test]
fn test_sincos_simd_pythagorean_identity() {
    let x = array![0.0_f64, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, PI, 2.0 * PI];

    let (sin_result, cos_result) = sincos_simd(&x.view());

    for i in 0..x.len() {
        let sum_sq = sin_result[i] * sin_result[i] + cos_result[i] * cos_result[i];
        assert!(
            (sum_sq - 1.0).abs() < 1e-14,
            "sin²(x) + cos²(x) should be 1, got {} at x={}",
            sum_sq,
            x[i]
        );
    }
}

/// Test sincos consistency with separate sin/cos
#[test]
fn test_sincos_simd_consistency() {
    use scirs2_core::ndarray_ext::elementwise::{cos_simd, sin_simd};

    let x = array![0.1_f64, 0.5, 1.0, 2.0, 3.0];

    let (sin_result, cos_result) = sincos_simd(&x.view());
    let sin_separate = sin_simd(&x.view());
    let cos_separate = cos_simd(&x.view());

    for i in 0..x.len() {
        assert!(
            (sin_result[i] - sin_separate[i]).abs() < 1e-14,
            "sincos sin should match sin_simd"
        );
        assert!(
            (cos_result[i] - cos_separate[i]).abs() < 1e-14,
            "sincos cos should match cos_simd"
        );
    }
}

/// Test sincos rotation matrix use case
#[test]
fn test_sincos_simd_rotation_matrix() {
    // 2D rotation matrix: [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
    // Rotating point (1, 0) by various angles

    let angles = array![0.0_f64, PI / 4.0, PI / 2.0, PI, 3.0 * PI / 2.0];
    let (sin_theta, cos_theta) = sincos_simd(&angles.view());

    // Point (1, 0) rotated by θ should be at (cos(θ), sin(θ))
    let px = 1.0_f64;
    let py = 0.0_f64;

    for i in 0..angles.len() {
        let rx = cos_theta[i] * px - sin_theta[i] * py;
        let ry = sin_theta[i] * px + cos_theta[i] * py;

        // After rotation, point should be at (cos(θ), sin(θ))
        assert!(
            (rx - cos_theta[i]).abs() < 1e-14,
            "Rotated x should be cos(θ)"
        );
        assert!(
            (ry - sin_theta[i]).abs() < 1e-14,
            "Rotated y should be sin(θ)"
        );

        // Distance from origin should remain 1
        let dist = (rx * rx + ry * ry).sqrt();
        assert!(
            (dist - 1.0).abs() < 1e-14,
            "Rotation should preserve distance from origin"
        );
    }
}

/// Test sincos complex exponential: e^(iθ) = cos(θ) + i*sin(θ)
#[test]
fn test_sincos_simd_euler_formula() {
    // Euler's formula: e^(iθ) = cos(θ) + i*sin(θ)
    // |e^(iθ)| = sqrt(cos²(θ) + sin²(θ)) = 1

    let theta = array![0.0_f64, PI / 4.0, PI / 2.0, PI, 2.0 * PI];
    let (sin_theta, cos_theta) = sincos_simd(&theta.view());

    for i in 0..theta.len() {
        // Magnitude of complex exponential
        let magnitude = (cos_theta[i] * cos_theta[i] + sin_theta[i] * sin_theta[i]).sqrt();
        assert!(
            (magnitude - 1.0).abs() < 1e-14,
            "Complex exponential magnitude should be 1"
        );
    }
}

/// Test sincos wave simulation use case
#[test]
fn test_sincos_simd_wave_simulation() {
    // Simulating a wave: y(t) = A * sin(ωt + φ)
    // Velocity: v(t) = A * ω * cos(ωt + φ)
    // Both need sin and cos

    let amplitude = 2.0_f64;
    let omega = 3.0_f64;
    let phase = PI / 4.0;

    let t = array![0.0_f64, 0.1, 0.2, 0.3, 0.4, 0.5];
    let omega_t_plus_phi = t.mapv(|ti| omega * ti + phase);

    let (sin_result, cos_result) = sincos_simd(&omega_t_plus_phi.view());

    let position = sin_result.mapv(|s| amplitude * s);
    let velocity = cos_result.mapv(|c| amplitude * omega * c);

    // At t=0: position = A*sin(φ), velocity = A*ω*cos(φ)
    let expected_pos_0 = amplitude * (omega * 0.0 + phase).sin();
    let expected_vel_0 = amplitude * omega * (omega * 0.0 + phase).cos();

    assert!((position[0] - expected_pos_0).abs() < 1e-14);
    assert!((velocity[0] - expected_vel_0).abs() < 1e-14);
}

/// Test sincos Fourier transform building block
#[test]
fn test_sincos_simd_fourier_basis() {
    // DFT basis functions: e^(-2πi*k*n/N) = cos(2πkn/N) - i*sin(2πkn/N)
    // For N=8, k=1, compute basis function at each n

    let n = 8;
    let k = 1.0_f64;
    let two_pi_k_over_n = 2.0 * PI * k / (n as f64);

    let indices: Array1<f64> = Array1::from_vec((0..n).map(|i| i as f64).collect());
    let angles = indices.mapv(|i| two_pi_k_over_n * i);

    let (sin_result, cos_result) = sincos_simd(&angles.view());

    // Verify orthogonality property: sum over n should be 0 for k != 0
    let real_sum: f64 = cos_result.iter().sum();
    let imag_sum: f64 = sin_result.iter().sum();

    assert!(
        real_sum.abs() < 1e-14,
        "Sum of cos(2πk*n/N) over n should be 0 for k != 0"
    );
    assert!(
        imag_sum.abs() < 1e-14,
        "Sum of sin(2πk*n/N) over n should be 0 for k != 0"
    );
}

// ============================================================================
// Phase 68: SIMD expm1 and log1p tests
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::expm1_simd;
use scirs2_core::ndarray_ext::elementwise::log1p_simd;

// ============================================================================
// expm1 tests
// ============================================================================

/// Test expm1 basic f32 correctness
#[test]
fn test_expm1_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, -1.0, 2.0];

    let result = expm1_simd(&x.view());

    // exp(0) - 1 = 0
    assert!((result[0] - 0.0).abs() < 1e-6);
    // exp(1) - 1 ≈ 1.718
    assert!((result[1] - (1.0_f32.exp() - 1.0)).abs() < 1e-5);
    // exp(-1) - 1 ≈ -0.632
    assert!((result[2] - ((-1.0_f32).exp() - 1.0)).abs() < 1e-5);
}

/// Test expm1 basic f64 correctness
#[test]
fn test_expm1_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -1.0, 2.0];

    let result = expm1_simd(&x.view());

    // exp(0) - 1 = 0
    assert!((result[0] - 0.0).abs() < 1e-14);
    // exp(1) - 1 ≈ 1.718
    assert!((result[1] - (1.0_f64.exp() - 1.0)).abs() < 1e-14);
    // exp(-1) - 1 ≈ -0.632
    assert!((result[2] - ((-1.0_f64).exp() - 1.0)).abs() < 1e-14);
}

/// Test expm1 empty array
#[test]
fn test_expm1_simd_empty() {
    let x: Array1<f64> = array![];
    let result = expm1_simd(&x.view());
    assert_eq!(result.len(), 0);
}

/// Test expm1 large array (SIMD path)
#[test]
fn test_expm1_simd_large_array() {
    let n = 10000;
    let x = Array1::from_vec((0..n).map(|i| (i as f64 - 5000.0) * 0.001).collect());

    let result = expm1_simd(&x.view());
    assert_eq!(result.len(), n);

    // Check a few values
    for i in [0, 100, 5000, 9999] {
        let expected = x[i].exp_m1();
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "expm1[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test expm1 numerical stability for small values
#[test]
fn test_expm1_simd_small_values() {
    // For very small x, exp(x) - 1 ≈ x (first term of Taylor series)
    // Direct calculation exp(x) - 1 would lose precision here

    let x = array![1e-15_f64, 1e-14, 1e-13, 1e-12, 1e-10];

    let result = expm1_simd(&x.view());

    // For very small x, expm1(x) ≈ x with high precision
    for i in 0..x.len() {
        // The relative error should be very small
        let rel_error = if x[i].abs() > 1e-20 {
            (result[i] - x[i]).abs() / x[i].abs()
        } else {
            (result[i] - x[i]).abs()
        };
        assert!(
            rel_error < 1e-10,
            "expm1 should be numerically stable for small x, relative error = {} at x = {}",
            rel_error,
            x[i]
        );
    }
}

/// Test expm1 vs naive exp(x) - 1 comparison for small values
#[test]
fn test_expm1_simd_vs_naive_small() {
    // This demonstrates the numerical advantage of expm1 over exp(x) - 1

    let x = array![1e-16_f64];
    let result = expm1_simd(&x.view());

    // Naive calculation would give 0 due to floating point precision loss
    let naive = x[0].exp() - 1.0;

    // expm1 should preserve precision
    // For x = 1e-16, expm1(x) should be approximately 1e-16
    // while naive exp(x) - 1 = 0 (precision loss)
    assert!(
        (result[0] - x[0]).abs() < 1e-30,
        "expm1 should preserve precision for tiny values"
    );
    // The naive calculation has lost precision
    assert_eq!(naive, 0.0, "naive exp(x)-1 loses precision for tiny x");
}

/// Test expm1 identity: expm1(-x) = -expm1(x) / (1 + expm1(x))
#[test]
fn test_expm1_simd_identity() {
    let x = array![0.5_f64, 1.0, 1.5, 2.0];

    let result_pos = expm1_simd(&x.view());
    let neg_x = x.mapv(|v| -v);
    let result_neg = expm1_simd(&neg_x.view());

    // expm1(-x) = -expm1(x) / (1 + expm1(x)) = -expm1(x) / exp(x)
    for i in 0..x.len() {
        let expected_neg = -result_pos[i] / (1.0 + result_pos[i]);
        assert!(
            (result_neg[i] - expected_neg).abs() < 1e-14,
            "expm1(-x) identity failed at x = {}",
            x[i]
        );
    }
}

/// Test expm1 financial compound interest use case
#[test]
fn test_expm1_simd_compound_interest() {
    // Continuous compound interest: A = P * e^(rt)
    // Interest earned = P * (e^(rt) - 1) = P * expm1(rt)
    // For small rates, this is critical for accuracy

    let principal = 10000.0_f64;
    let rate = 0.0001_f64; // 0.01% daily rate
    let time = 1.0_f64; // 1 day

    let rt = array![rate * time];
    let growth_factor = expm1_simd(&rt.view());
    let interest = principal * growth_factor[0];

    // For small rates: interest ≈ P * r * t (simple interest approximation)
    let simple_interest = principal * rate * time;

    // The difference between compound and simple interest is very small:
    // compound_interest - simple_interest ≈ P * r² * t² / 2 = 10000 * 1e-8 / 2 = 5e-5
    // So the interest should be very close to simple interest
    let diff = (interest - simple_interest).abs();
    assert!(
        diff < 1e-3,
        "expm1 preserves precision for small compound interest calculations, diff = {}",
        diff
    );

    // Also verify the compound interest is slightly higher than simple interest
    // (due to compounding effect)
    assert!(
        interest >= simple_interest,
        "Compound interest should be >= simple interest"
    );
}

// ============================================================================
// log1p tests
// ============================================================================

/// Test log1p basic f32 correctness
#[test]
fn test_log1p_simd_f32_basic() {
    let x = array![0.0_f32, 1.0, -0.5, 3.0];

    let result = log1p_simd(&x.view());

    // ln(1 + 0) = 0
    assert!((result[0] - 0.0).abs() < 1e-6);
    // ln(1 + 1) = ln(2)
    assert!((result[1] - 2.0_f32.ln()).abs() < 1e-6);
    // ln(1 + (-0.5)) = ln(0.5)
    assert!((result[2] - 0.5_f32.ln()).abs() < 1e-6);
}

/// Test log1p basic f64 correctness
#[test]
fn test_log1p_simd_f64_basic() {
    let x = array![0.0_f64, 1.0, -0.5, 3.0];

    let result = log1p_simd(&x.view());

    // ln(1 + 0) = 0
    assert!((result[0] - 0.0).abs() < 1e-14);
    // ln(1 + 1) = ln(2)
    assert!((result[1] - 2.0_f64.ln()).abs() < 1e-14);
    // ln(1 + (-0.5)) = ln(0.5)
    assert!((result[2] - 0.5_f64.ln()).abs() < 1e-14);
}

/// Test log1p empty array
#[test]
fn test_log1p_simd_empty() {
    let x: Array1<f64> = array![];
    let result = log1p_simd(&x.view());
    assert_eq!(result.len(), 0);
}

/// Test log1p large array (SIMD path)
#[test]
fn test_log1p_simd_large_array() {
    let n = 10000;
    // Use values in (-0.9, 10) to avoid ln of negative numbers
    let x = Array1::from_vec(
        (0..n)
            .map(|i| -0.9 + (i as f64 * 11.0 / n as f64))
            .collect(),
    );

    let result = log1p_simd(&x.view());
    assert_eq!(result.len(), n);

    // Check a few values
    for i in [0, 100, 5000, 9999] {
        let expected = x[i].ln_1p();
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "log1p[{}] = {}, expected {}",
            i,
            result[i],
            expected
        );
    }
}

/// Test log1p numerical stability for small values
#[test]
fn test_log1p_simd_small_values() {
    // For very small x, ln(1 + x) ≈ x (first term of Taylor series)
    // Direct calculation ln(1 + x) would lose precision here

    let x = array![1e-15_f64, 1e-14, 1e-13, 1e-12, 1e-10];

    let result = log1p_simd(&x.view());

    // For very small x, log1p(x) ≈ x with high precision
    for i in 0..x.len() {
        let rel_error = if x[i].abs() > 1e-20 {
            (result[i] - x[i]).abs() / x[i].abs()
        } else {
            (result[i] - x[i]).abs()
        };
        assert!(
            rel_error < 1e-10,
            "log1p should be numerically stable for small x, relative error = {} at x = {}",
            rel_error,
            x[i]
        );
    }
}

/// Test log1p vs naive ln(1 + x) comparison for small values
#[test]
fn test_log1p_simd_vs_naive_small() {
    // This demonstrates the numerical advantage of log1p over ln(1 + x)

    let x = array![1e-16_f64];
    let result = log1p_simd(&x.view());

    // Naive calculation would give 0 due to floating point precision loss
    let naive = (1.0 + x[0]).ln();

    // log1p should preserve precision
    // For x = 1e-16, log1p(x) should be approximately 1e-16
    // while naive ln(1 + x) = 0 (precision loss because 1 + 1e-16 = 1 in f64)
    assert!(
        (result[0] - x[0]).abs() < 1e-30,
        "log1p should preserve precision for tiny values"
    );
    assert_eq!(naive, 0.0, "naive ln(1+x) loses precision for tiny x");
}

/// Test log1p edge cases
#[test]
fn test_log1p_simd_edge_cases() {
    let x = array![-1.0_f64, -2.0];

    let result = log1p_simd(&x.view());

    // ln(1 + (-1)) = ln(0) = -∞
    assert!(result[0].is_infinite() && result[0] < 0.0);
    // ln(1 + (-2)) = ln(-1) = NaN
    assert!(result[1].is_nan());
}

/// Test log1p identity: log1p(x) + log1p(y) = log1p(x + y + xy)
#[test]
fn test_log1p_simd_addition_identity() {
    // log(a) + log(b) = log(ab)
    // log1p(x) + log1p(y) = ln(1+x) + ln(1+y) = ln((1+x)(1+y)) = ln(1 + x + y + xy)
    // = log1p(x + y + xy)

    let x = array![0.1_f64, 0.2, 0.3, 0.5];
    let y = array![0.2_f64, 0.3, 0.1, 0.5];

    let log1p_x = log1p_simd(&x.view());
    let log1p_y = log1p_simd(&y.view());

    let combined = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| xi + yi + xi * yi)
        .collect::<Vec<_>>();
    let combined_arr = Array1::from_vec(combined);
    let log1p_combined = log1p_simd(&combined_arr.view());

    for i in 0..x.len() {
        let sum = log1p_x[i] + log1p_y[i];
        assert!(
            (sum - log1p_combined[i]).abs() < 1e-14,
            "log1p addition identity failed at i = {}",
            i
        );
    }
}

/// Test log1p inverse relationship with expm1
#[test]
fn test_log1p_simd_inverse_of_expm1() {
    // log1p(expm1(x)) = x for any x
    // expm1(log1p(x)) = x for x > -1

    let x = array![0.1_f64, 0.5, 1.0, 2.0, -0.5];

    let expm1_x = expm1_simd(&x.view());
    let log1p_expm1_x = log1p_simd(&expm1_x.view());

    for i in 0..x.len() {
        assert!(
            (log1p_expm1_x[i] - x[i]).abs() < 1e-14,
            "log1p(expm1(x)) should equal x"
        );
    }

    // Also test expm1(log1p(x)) = x
    let log1p_x = log1p_simd(&x.view());
    let expm1_log1p_x = expm1_simd(&log1p_x.view());

    for i in 0..x.len() {
        assert!(
            (expm1_log1p_x[i] - x[i]).abs() < 1e-14,
            "expm1(log1p(x)) should equal x"
        );
    }
}

/// Test log1p binary cross-entropy use case
#[test]
fn test_log1p_simd_cross_entropy() {
    // Binary cross-entropy: -y*log(p) - (1-y)*log(1-p)
    // When p is very close to 0 or 1, we need log1p for stability
    // log(1-p) = log1p(-p)

    let p = array![0.99999_f64, 0.9999, 0.999]; // p close to 1
    let neg_p = p.mapv(|v| -v);

    let log_1_minus_p = log1p_simd(&neg_p.view());

    // For p close to 1, log(1-p) should be a large negative number
    for (i, &v) in log_1_minus_p.iter().enumerate() {
        assert!(
            v < 0.0,
            "log(1-p) should be negative for p close to 1, got {} at index {}",
            v,
            i
        );
        assert!(v.is_finite(), "log(1-p) should be finite for p < 1");
    }
}

// ============================================================================
// Phase 69: SIMD clip, cumsum, cumprod, diff tests
// ============================================================================

use scirs2_core::ndarray_ext::elementwise::clip_simd;
use scirs2_core::ndarray_ext::elementwise::cumprod_simd;
use scirs2_core::ndarray_ext::elementwise::cumsum_simd;
use scirs2_core::ndarray_ext::elementwise::diff_simd;

// ============================================================================
// clip tests
// ============================================================================

/// Test clip basic f32 correctness
#[test]
fn test_clip_simd_f32_basic() {
    let x = array![-2.0_f32, -1.0, 0.0, 1.0, 2.0, 3.0];

    let result = clip_simd(&x.view(), 0.0, 1.0);

    assert!((result[0] - 0.0).abs() < 1e-6); // -2 clipped to 0
    assert!((result[1] - 0.0).abs() < 1e-6); // -1 clipped to 0
    assert!((result[2] - 0.0).abs() < 1e-6); // 0 unchanged
    assert!((result[3] - 1.0).abs() < 1e-6); // 1 unchanged
    assert!((result[4] - 1.0).abs() < 1e-6); // 2 clipped to 1
    assert!((result[5] - 1.0).abs() < 1e-6); // 3 clipped to 1
}

/// Test clip basic f64 correctness
#[test]
fn test_clip_simd_f64_basic() {
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0, 3.0];

    let result = clip_simd(&x.view(), 0.0, 1.0);

    assert!((result[0] - 0.0).abs() < 1e-14); // -2 clipped to 0
    assert!((result[1] - 0.0).abs() < 1e-14); // -1 clipped to 0
    assert!((result[2] - 0.0).abs() < 1e-14); // 0 unchanged
    assert!((result[3] - 1.0).abs() < 1e-14); // 1 unchanged
    assert!((result[4] - 1.0).abs() < 1e-14); // 2 clipped to 1
    assert!((result[5] - 1.0).abs() < 1e-14); // 3 clipped to 1
}

/// Test clip empty array
#[test]
fn test_clip_simd_empty() {
    let x: Array1<f64> = array![];
    let result = clip_simd(&x.view(), 0.0, 1.0);
    assert_eq!(result.len(), 0);
}

/// Test clip large array (SIMD path)
#[test]
fn test_clip_simd_large_array() {
    let n = 10000;
    let x = Array1::from_vec((0..n).map(|i| (i as f64 - 5000.0) * 0.001).collect());

    let result = clip_simd(&x.view(), -1.0, 1.0);
    assert_eq!(result.len(), n);

    // All values should be in [-1, 1]
    for &v in result.iter() {
        assert!((-1.0..=1.0).contains(&v), "Value {} out of clip range", v);
    }
}

/// Test clip gradient clipping use case
#[test]
fn test_clip_simd_gradient_clipping() {
    // In neural networks, gradients are often clipped to prevent exploding gradients
    let gradients = array![100.0_f64, -50.0, 0.5, -0.1, 200.0];

    let clipped = clip_simd(&gradients.view(), -1.0, 1.0);

    assert!((clipped[0] - 1.0).abs() < 1e-14); // 100 clipped to 1
    assert!((clipped[1] - (-1.0)).abs() < 1e-14); // -50 clipped to -1
    assert!((clipped[2] - 0.5).abs() < 1e-14); // 0.5 unchanged
    assert!((clipped[3] - (-0.1)).abs() < 1e-14); // -0.1 unchanged
    assert!((clipped[4] - 1.0).abs() < 1e-14); // 200 clipped to 1
}

// ============================================================================
// cumsum tests
// ============================================================================

/// Test cumsum basic f32 correctness
#[test]
fn test_cumsum_simd_f32_basic() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0, 5.0];

    let result = cumsum_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-5); // 1
    assert!((result[1] - 3.0).abs() < 1e-5); // 1+2
    assert!((result[2] - 6.0).abs() < 1e-5); // 1+2+3
    assert!((result[3] - 10.0).abs() < 1e-5); // 1+2+3+4
    assert!((result[4] - 15.0).abs() < 1e-5); // 1+2+3+4+5
}

/// Test cumsum basic f64 correctness
#[test]
fn test_cumsum_simd_f64_basic() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];

    let result = cumsum_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-14); // 1
    assert!((result[1] - 3.0).abs() < 1e-14); // 1+2
    assert!((result[2] - 6.0).abs() < 1e-14); // 1+2+3
    assert!((result[3] - 10.0).abs() < 1e-14); // 1+2+3+4
    assert!((result[4] - 15.0).abs() < 1e-14); // 1+2+3+4+5
}

/// Test cumsum empty array
#[test]
fn test_cumsum_simd_empty() {
    let x: Array1<f64> = array![];
    let result = cumsum_simd(&x.view());
    assert_eq!(result.len(), 0);
}

/// Test cumsum large array (SIMD path)
#[test]
fn test_cumsum_simd_large_array() {
    let n = 10000;
    let x = Array1::from_vec((1..=n).map(|i| i as f64).collect());

    let result = cumsum_simd(&x.view());
    assert_eq!(result.len(), n);

    // Last element should be n*(n+1)/2 = 50005000
    let expected_last = (n * (n + 1) / 2) as f64;
    assert!(
        (result[n - 1] - expected_last).abs() < 1e-6,
        "cumsum last element should be {}, got {}",
        expected_last,
        result[n - 1]
    );
}

/// Test cumsum CDF computation use case
#[test]
fn test_cumsum_simd_cdf() {
    // Computing CDF from PDF
    let pdf = array![0.1_f64, 0.2, 0.3, 0.25, 0.15]; // Should sum to 1

    let cdf = cumsum_simd(&pdf.view());

    assert!((cdf[0] - 0.1).abs() < 1e-14);
    assert!((cdf[1] - 0.3).abs() < 1e-14);
    assert!((cdf[2] - 0.6).abs() < 1e-14);
    assert!((cdf[3] - 0.85).abs() < 1e-14);
    assert!((cdf[4] - 1.0).abs() < 1e-14); // CDF should end at 1
}

// ============================================================================
// cumprod tests
// ============================================================================

/// Test cumprod basic f32 correctness
#[test]
fn test_cumprod_simd_f32_basic() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0];

    let result = cumprod_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-5); // 1
    assert!((result[1] - 2.0).abs() < 1e-5); // 1*2
    assert!((result[2] - 6.0).abs() < 1e-5); // 1*2*3
    assert!((result[3] - 24.0).abs() < 1e-5); // 1*2*3*4 = 4!
}

/// Test cumprod basic f64 correctness
#[test]
fn test_cumprod_simd_f64_basic() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];

    let result = cumprod_simd(&x.view());

    assert!((result[0] - 1.0).abs() < 1e-14); // 1
    assert!((result[1] - 2.0).abs() < 1e-14); // 1*2
    assert!((result[2] - 6.0).abs() < 1e-14); // 1*2*3
    assert!((result[3] - 24.0).abs() < 1e-14); // 1*2*3*4 = 4!
    assert!((result[4] - 120.0).abs() < 1e-14); // 1*2*3*4*5 = 5!
}

/// Test cumprod empty array
#[test]
fn test_cumprod_simd_empty() {
    let x: Array1<f64> = array![];
    let result = cumprod_simd(&x.view());
    assert_eq!(result.len(), 0);
}

/// Test cumprod factorial computation
#[test]
fn test_cumprod_simd_factorial() {
    // Computing factorials: cumprod([1,2,3,...,n]) = [1!, 2!, 3!, ..., n!]
    let n = 10;
    let seq = Array1::from_vec((1..=n).map(|i| i as f64).collect());

    let factorials = cumprod_simd(&seq.view());

    // Known factorials
    let expected = [
        1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0,
    ];

    for i in 0..n {
        assert!(
            (factorials[i] - expected[i]).abs() < 1e-10,
            "{}! should be {}, got {}",
            i + 1,
            expected[i],
            factorials[i]
        );
    }
}

/// Test cumprod survival probability use case
#[test]
fn test_cumprod_simd_survival() {
    // Computing survival probability from hazard rates
    // Survival at time t = product of (1 - hazard_i) for i < t
    let survival_probs = array![0.99_f64, 0.98, 0.97, 0.96, 0.95];

    let cumulative_survival = cumprod_simd(&survival_probs.view());

    // Cumulative survival should be decreasing
    for i in 1..survival_probs.len() {
        assert!(
            cumulative_survival[i] < cumulative_survival[i - 1],
            "Cumulative survival should be decreasing"
        );
    }

    // Last value should be product of all survival probabilities
    let expected = 0.99 * 0.98 * 0.97 * 0.96 * 0.95;
    assert!(
        (cumulative_survival[4] - expected).abs() < 1e-14,
        "Final survival should be {}",
        expected
    );
}

// ============================================================================
// diff tests
// ============================================================================

/// Test diff basic f32 correctness
#[test]
fn test_diff_simd_f32_basic() {
    let x = array![1.0_f32, 3.0, 6.0, 10.0, 15.0];

    let result = diff_simd(&x.view());
    assert_eq!(result.len(), 4); // n-1 elements

    assert!((result[0] - 2.0).abs() < 1e-5); // 3-1
    assert!((result[1] - 3.0).abs() < 1e-5); // 6-3
    assert!((result[2] - 4.0).abs() < 1e-5); // 10-6
    assert!((result[3] - 5.0).abs() < 1e-5); // 15-10
}

/// Test diff basic f64 correctness
#[test]
fn test_diff_simd_f64_basic() {
    let x = array![1.0_f64, 3.0, 6.0, 10.0, 15.0];

    let result = diff_simd(&x.view());
    assert_eq!(result.len(), 4); // n-1 elements

    assert!((result[0] - 2.0).abs() < 1e-14); // 3-1
    assert!((result[1] - 3.0).abs() < 1e-14); // 6-3
    assert!((result[2] - 4.0).abs() < 1e-14); // 10-6
    assert!((result[3] - 5.0).abs() < 1e-14); // 15-10
}

/// Test diff empty and single element arrays
#[test]
fn test_diff_simd_edge_cases() {
    let x_empty: Array1<f64> = array![];
    let result_empty = diff_simd(&x_empty.view());
    assert_eq!(result_empty.len(), 0);

    let x_single = array![1.0_f64];
    let result_single = diff_simd(&x_single.view());
    assert_eq!(result_single.len(), 0); // Need at least 2 elements
}

/// Test diff large array (SIMD path)
#[test]
fn test_diff_simd_large_array() {
    let n = 10000;
    // Create quadratic sequence: x[i] = i^2
    let x = Array1::from_vec((0..n).map(|i| (i as f64).powi(2)).collect());

    let result = diff_simd(&x.view());
    assert_eq!(result.len(), n - 1);

    // diff of x^2 gives 2x+1 = 1, 3, 5, 7, ... (odd numbers)
    for i in 0..result.len() {
        let expected = (2 * i + 1) as f64;
        assert!(
            (result[i] - expected).abs() < 1e-10,
            "diff[{}] should be {}, got {}",
            i,
            expected,
            result[i]
        );
    }
}

/// Test diff constant array gives zeros
#[test]
fn test_diff_simd_constant() {
    let x = array![5.0_f64, 5.0, 5.0, 5.0, 5.0];

    let result = diff_simd(&x.view());

    for &v in result.iter() {
        assert!((v - 0.0).abs() < 1e-14, "diff of constant should be 0");
    }
}

/// Test diff numerical differentiation use case
#[test]
fn test_diff_simd_numerical_derivative() {
    // Approximate derivative of sin(x) should be cos(x)
    let n = 100;
    let h = 0.01_f64;
    let x = Array1::from_vec((0..n).map(|i| i as f64 * h).collect());
    let sin_x = x.mapv(|xi| xi.sin());

    let diff_sin = diff_simd(&sin_x.view());

    // diff[i] / h ≈ cos(x[i])
    for i in 0..diff_sin.len() {
        let numerical_derivative = diff_sin[i] / h;
        let analytical_derivative = x[i].cos();
        // Should be close (within O(h) error)
        assert!(
            (numerical_derivative - analytical_derivative).abs() < 0.02,
            "Numerical derivative should approximate cos(x)"
        );
    }
}

/// Test diff and cumsum are inverse operations
#[test]
fn test_diff_simd_inverse_of_cumsum() {
    // For a sequence starting from 0: diff(cumsum(x)) = x (except for first element)
    // cumsum(diff(x)) = x - x[0] (offset by first element)

    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];

    let cumsum_x = cumsum_simd(&x.view());
    let diff_cumsum_x = diff_simd(&cumsum_x.view());

    // diff(cumsum(x))[i] should equal x[i+1]
    for i in 0..diff_cumsum_x.len() {
        assert!(
            (diff_cumsum_x[i] - x[i + 1]).abs() < 1e-14,
            "diff(cumsum(x))[{}] should equal x[{}]",
            i,
            i + 1
        );
    }
}

// =============================================================================
// Phase 70: Statistical Functions Tests
// =============================================================================

use scirs2_core::ndarray_ext::elementwise::{
    add_simd, argmax_simd, argmin_simd, cosine_similarity_simd, distance_chebyshev_simd,
    distance_cosine_simd, distance_euclidean_simd, distance_manhattan_simd, div_simd, dot_simd,
    fma_simd, log_sum_exp_simd, max_element_simd, max_simd, mean_simd, min_element_simd, min_simd,
    mul_simd, norm_l1_simd, norm_linf_simd, norm_simd, normalize_simd, relu_simd, scalar_mul_simd,
    softmax_simd, standardize_simd, std_simd, sub_simd, sum_cubes_simd, sum_simd, sum_squares_simd,
    trunc_simd, variance_simd, weighted_mean_simd, weighted_sum_simd,
};

/// Test variance_simd with known values
#[test]
fn test_variance_simd_basic() {
    // Sample variance (Bessel's correction) of [2,4,4,4,5,5,7,9]
    // mean = 5, sum of squared deviations = 32, n = 8
    // sample variance = 32/(8-1) = 32/7 ≈ 4.571
    let x = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let var = variance_simd::<f64>(&x.view());
    let expected = 32.0 / 7.0;
    assert!(
        (var - expected).abs() < 1e-10,
        "Variance should be {}, got {}",
        expected,
        var
    );
}

/// Test variance_simd f32
#[test]
fn test_variance_simd_f32() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let var = variance_simd::<f32>(&x.view());
    // Sample variance: Mean = 3, sum of squared deviations = 10
    // sample variance = 10/(5-1) = 2.5
    let expected = 10.0 / 4.0;
    assert!(
        (var - expected).abs() < 1e-5,
        "Variance should be {}, got {}",
        expected,
        var
    );
}

/// Test variance_simd empty array
#[test]
fn test_variance_simd_empty() {
    let x: Array1<f64> = Array1::zeros(0);
    let var = variance_simd::<f64>(&x.view());
    assert!(var.abs() < 1e-14, "Empty array variance should be 0");
}

/// Test std_simd with known values
#[test]
fn test_std_simd_basic() {
    // Sample std = sqrt(sample variance) = sqrt(32/7) ≈ 2.138
    let x = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let s = std_simd::<f64>(&x.view());
    let expected = (32.0_f64 / 7.0).sqrt();
    assert!(
        (s - expected).abs() < 1e-10,
        "Std should be {}, got {}",
        expected,
        s
    );
}

/// Test std_simd f32
#[test]
fn test_std_simd_f32() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let s = std_simd::<f32>(&x.view());
    // Sample std = sqrt(10/4) = sqrt(2.5) ≈ 1.581
    let expected = (10.0_f32 / 4.0).sqrt();
    assert!(
        (s - expected).abs() < 1e-5,
        "Std should be {}, got {}",
        expected,
        s
    );
}

/// Test sum_simd basic
#[test]
fn test_sum_simd_basic() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let s = sum_simd::<f64>(&x.view());
    assert!((s - 15.0).abs() < 1e-14, "Sum should be 15.0, got {}", s);
}

/// Test sum_simd f32
#[test]
fn test_sum_simd_f32() {
    let x = array![10.0_f32, 20.0, 30.0];
    let s = sum_simd::<f32>(&x.view());
    assert!((s - 60.0).abs() < 1e-5, "Sum should be 60.0, got {}", s);
}

/// Test sum_simd large array
#[test]
fn test_sum_simd_large() {
    let n = 10000;
    let x = Array1::from_vec((1..=n).map(|i| i as f64).collect());
    let s = sum_simd::<f64>(&x.view());
    // Sum of 1 to n = n*(n+1)/2
    let expected = (n * (n + 1) / 2) as f64;
    assert!(
        (s - expected).abs() < 1e-8,
        "Sum should be {}, got {}",
        expected,
        s
    );
}

/// Test mean_simd basic
#[test]
fn test_mean_simd_basic() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let m = mean_simd::<f64>(&x.view());
    assert!((m - 3.0).abs() < 1e-14, "Mean should be 3.0, got {}", m);
}

/// Test mean_simd f32
#[test]
fn test_mean_simd_f32() {
    let x = array![2.0_f32, 4.0, 6.0, 8.0];
    let m = mean_simd::<f32>(&x.view());
    assert!((m - 5.0).abs() < 1e-5, "Mean should be 5.0, got {}", m);
}

/// Test weighted_sum_simd basic
#[test]
fn test_weighted_sum_simd_basic() {
    let values = array![10.0_f64, 20.0, 30.0];
    let weights = array![0.2_f64, 0.3, 0.5];
    let ws = weighted_sum_simd::<f64>(&values.view(), &weights.view());
    // 10*0.2 + 20*0.3 + 30*0.5 = 2 + 6 + 15 = 23
    assert!(
        (ws - 23.0).abs() < 1e-10,
        "Weighted sum should be 23.0, got {}",
        ws
    );
}

/// Test weighted_sum_simd f32
#[test]
fn test_weighted_sum_simd_f32() {
    let values = array![1.0_f32, 2.0, 3.0];
    let weights = array![1.0_f32, 1.0, 1.0];
    let ws = weighted_sum_simd::<f32>(&values.view(), &weights.view());
    assert!(
        (ws - 6.0).abs() < 1e-5,
        "Weighted sum should be 6.0, got {}",
        ws
    );
}

/// Test weighted_mean_simd basic
#[test]
fn test_weighted_mean_simd_basic() {
    let values = array![10.0_f64, 20.0, 30.0];
    let weights = array![1.0_f64, 2.0, 2.0];
    let wm = weighted_mean_simd::<f64>(&values.view(), &weights.view());
    // (10*1 + 20*2 + 30*2) / (1+2+2) = 110/5 = 22
    assert!(
        (wm - 22.0).abs() < 1e-10,
        "Weighted mean should be 22.0, got {}",
        wm
    );
}

/// Test weighted_mean_simd f32
#[test]
fn test_weighted_mean_simd_f32() {
    let values = array![100.0_f32, 200.0];
    let weights = array![3.0_f32, 1.0];
    let wm = weighted_mean_simd::<f32>(&values.view(), &weights.view());
    // (100*3 + 200*1) / (3+1) = 500/4 = 125
    assert!(
        (wm - 125.0).abs() < 1e-4,
        "Weighted mean should be 125.0, got {}",
        wm
    );
}

// =============================================================================
// Phase 71: Reduction Functions Tests
// =============================================================================

/// Test max_element_simd basic
#[test]
fn test_max_element_simd_basic() {
    let x = array![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let max = max_element_simd::<f64>(&x.view());
    assert!((max - 9.0).abs() < 1e-14, "Max should be 9.0, got {}", max);
}

/// Test max_element_simd f32
#[test]
fn test_max_element_simd_f32() {
    let x = array![-5.0_f32, -3.0, -1.0, -7.0];
    let max = max_element_simd::<f32>(&x.view());
    assert!(
        (max - (-1.0)).abs() < 1e-5,
        "Max should be -1.0, got {}",
        max
    );
}

/// Test min_element_simd basic
#[test]
fn test_min_element_simd_basic() {
    let x = array![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let min = min_element_simd::<f64>(&x.view());
    assert!((min - 1.0).abs() < 1e-14, "Min should be 1.0, got {}", min);
}

/// Test min_element_simd f32
#[test]
fn test_min_element_simd_f32() {
    let x = array![5.0_f32, 3.0, 7.0, 2.0, 8.0];
    let min = min_element_simd::<f32>(&x.view());
    assert!((min - 2.0).abs() < 1e-5, "Min should be 2.0, got {}", min);
}

/// Test argmax_simd basic
#[test]
fn test_argmax_simd_basic() {
    let x = array![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let idx = argmax_simd::<f64>(&x.view());
    assert_eq!(idx, Some(5), "Argmax should be 5 (x[5] = 9.0)");
}

/// Test argmax_simd f32
#[test]
fn test_argmax_simd_f32() {
    let x = array![1.0_f32, 2.0, 5.0, 3.0];
    let idx = argmax_simd::<f32>(&x.view());
    assert_eq!(idx, Some(2), "Argmax should be 2 (x[2] = 5.0)");
}

/// Test argmax_simd empty
#[test]
fn test_argmax_simd_empty() {
    let x: Array1<f64> = Array1::zeros(0);
    let idx = argmax_simd::<f64>(&x.view());
    assert_eq!(idx, None, "Empty array argmax should be None");
}

/// Test argmin_simd basic
#[test]
fn test_argmin_simd_basic() {
    let x = array![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let idx = argmin_simd::<f64>(&x.view());
    assert_eq!(
        idx,
        Some(1),
        "Argmin should be 1 (x[1] = 1.0, first occurrence)"
    );
}

/// Test argmin_simd f32
#[test]
fn test_argmin_simd_f32() {
    let x = array![4.0_f32, 2.0, 5.0, 1.0, 3.0];
    let idx = argmin_simd::<f32>(&x.view());
    assert_eq!(idx, Some(3), "Argmin should be 3 (x[3] = 1.0)");
}

/// Test sum_squares_simd basic
#[test]
fn test_sum_squares_simd_basic() {
    let x = array![1.0_f64, 2.0, 3.0, 4.0];
    let ss = sum_squares_simd::<f64>(&x.view());
    // 1 + 4 + 9 + 16 = 30
    assert!(
        (ss - 30.0).abs() < 1e-14,
        "Sum of squares should be 30, got {}",
        ss
    );
}

/// Test sum_squares_simd f32
#[test]
fn test_sum_squares_simd_f32() {
    let x = array![3.0_f32, 4.0];
    let ss = sum_squares_simd::<f32>(&x.view());
    // 9 + 16 = 25
    assert!(
        (ss - 25.0).abs() < 1e-5,
        "Sum of squares should be 25, got {}",
        ss
    );
}

/// Test sum_cubes_simd basic
#[test]
fn test_sum_cubes_simd_basic() {
    let x = array![1.0_f64, 2.0, 3.0];
    let sc = sum_cubes_simd::<f64>(&x.view());
    // 1 + 8 + 27 = 36
    assert!(
        (sc - 36.0).abs() < 1e-14,
        "Sum of cubes should be 36, got {}",
        sc
    );
}

/// Test sum_cubes_simd f32
#[test]
fn test_sum_cubes_simd_f32() {
    let x = array![2.0_f32, 3.0];
    let sc = sum_cubes_simd::<f32>(&x.view());
    // 8 + 27 = 35
    assert!(
        (sc - 35.0).abs() < 1e-4,
        "Sum of cubes should be 35, got {}",
        sc
    );
}

/// Test log_sum_exp_simd basic
#[test]
fn test_log_sum_exp_simd_basic() {
    let x = array![1.0_f64, 2.0, 3.0];
    let lse = log_sum_exp_simd::<f64>(&x.view());
    // log(e^1 + e^2 + e^3)
    let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
    assert!(
        (lse - expected).abs() < 1e-10,
        "Log-sum-exp should be {}, got {}",
        expected,
        lse
    );
}

/// Test log_sum_exp_simd numerical stability
#[test]
fn test_log_sum_exp_simd_stability() {
    // Large values that would overflow naive exp
    let x = array![1000.0_f64, 1001.0, 1002.0];
    let lse = log_sum_exp_simd::<f64>(&x.view());
    // Should be approximately 1002 + log(1 + e^-1 + e^-2)
    assert!(
        lse > 1000.0 && lse < 1003.0,
        "Log-sum-exp should handle large values stably, got {}",
        lse
    );
}

/// Test log_sum_exp_simd f32
#[test]
fn test_log_sum_exp_simd_f32() {
    let x = array![0.0_f32, 1.0, 2.0];
    let lse = log_sum_exp_simd::<f32>(&x.view());
    let expected = (1.0_f32 + 1.0_f32.exp() + 2.0_f32.exp()).ln();
    assert!(
        (lse - expected).abs() < 1e-5,
        "Log-sum-exp should be {}, got {}",
        expected,
        lse
    );
}

// =============================================================================
// Phase 72: Norm and Distance Functions Tests
// =============================================================================

/// Test norm_simd (L2) basic
#[test]
fn test_norm_simd_basic() {
    let x = array![3.0_f64, 4.0];
    let n = norm_simd::<f64>(&x.view());
    assert!((n - 5.0).abs() < 1e-14, "L2 norm should be 5.0, got {}", n);
}

/// Test norm_simd f32
#[test]
fn test_norm_simd_f32() {
    let x = array![1.0_f32, 2.0, 2.0];
    let n = norm_simd::<f32>(&x.view());
    // sqrt(1 + 4 + 4) = 3
    assert!((n - 3.0).abs() < 1e-5, "L2 norm should be 3.0, got {}", n);
}

/// Test norm_l1_simd basic
#[test]
fn test_norm_l1_simd_basic() {
    let x = array![1.0_f64, -2.0, 3.0, -4.0];
    let n = norm_l1_simd::<f64>(&x.view());
    // |1| + |-2| + |3| + |-4| = 10
    assert!(
        (n - 10.0).abs() < 1e-14,
        "L1 norm should be 10.0, got {}",
        n
    );
}

/// Test norm_l1_simd f32
#[test]
fn test_norm_l1_simd_f32() {
    let x = array![-3.0_f32, 4.0];
    let n = norm_l1_simd::<f32>(&x.view());
    assert!((n - 7.0).abs() < 1e-5, "L1 norm should be 7.0, got {}", n);
}

/// Test norm_linf_simd basic
#[test]
fn test_norm_linf_simd_basic() {
    let x = array![1.0_f64, -5.0, 3.0, -2.0];
    let n = norm_linf_simd::<f64>(&x.view());
    // max(|1|, |-5|, |3|, |-2|) = 5
    assert!(
        (n - 5.0).abs() < 1e-14,
        "L-inf norm should be 5.0, got {}",
        n
    );
}

/// Test norm_linf_simd f32
#[test]
fn test_norm_linf_simd_f32() {
    let x = array![2.0_f32, -7.0, 4.0];
    let n = norm_linf_simd::<f32>(&x.view());
    assert!(
        (n - 7.0).abs() < 1e-5,
        "L-inf norm should be 7.0, got {}",
        n
    );
}

/// Test distance_euclidean_simd basic
#[test]
fn test_distance_euclidean_simd_basic() {
    let a = array![0.0_f64, 0.0];
    let b = array![3.0_f64, 4.0];
    let d = distance_euclidean_simd::<f64>(&a.view(), &b.view());
    assert!(
        (d - 5.0).abs() < 1e-14,
        "Euclidean distance should be 5.0, got {}",
        d
    );
}

/// Test distance_euclidean_simd f32
#[test]
fn test_distance_euclidean_simd_f32() {
    let a = array![1.0_f32, 1.0, 1.0];
    let b = array![2.0_f32, 2.0, 2.0];
    let d = distance_euclidean_simd::<f32>(&a.view(), &b.view());
    // sqrt(3) ≈ 1.732
    assert!(
        (d - 3.0_f32.sqrt()).abs() < 1e-5,
        "Euclidean distance should be sqrt(3), got {}",
        d
    );
}

/// Test distance_manhattan_simd basic
#[test]
fn test_distance_manhattan_simd_basic() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![4.0_f64, 0.0, 3.0];
    let d = distance_manhattan_simd::<f64>(&a.view(), &b.view());
    // |1-4| + |2-0| + |3-3| = 3 + 2 + 0 = 5
    assert!(
        (d - 5.0).abs() < 1e-14,
        "Manhattan distance should be 5.0, got {}",
        d
    );
}

/// Test distance_manhattan_simd f32
#[test]
fn test_distance_manhattan_simd_f32() {
    let a = array![0.0_f32, 0.0];
    let b = array![3.0_f32, 4.0];
    let d = distance_manhattan_simd::<f32>(&a.view(), &b.view());
    assert!(
        (d - 7.0).abs() < 1e-5,
        "Manhattan distance should be 7.0, got {}",
        d
    );
}

/// Test distance_chebyshev_simd basic
#[test]
fn test_distance_chebyshev_simd_basic() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![4.0_f64, 0.0, 3.0];
    let d = distance_chebyshev_simd::<f64>(&a.view(), &b.view());
    // max(|1-4|, |2-0|, |3-3|) = max(3, 2, 0) = 3
    assert!(
        (d - 3.0).abs() < 1e-14,
        "Chebyshev distance should be 3.0, got {}",
        d
    );
}

/// Test distance_chebyshev_simd f32
#[test]
fn test_distance_chebyshev_simd_f32() {
    let a = array![0.0_f32, 0.0];
    let b = array![3.0_f32, 4.0];
    let d = distance_chebyshev_simd::<f32>(&a.view(), &b.view());
    assert!(
        (d - 4.0).abs() < 1e-5,
        "Chebyshev distance should be 4.0, got {}",
        d
    );
}

/// Test distance_cosine_simd orthogonal
#[test]
fn test_distance_cosine_simd_orthogonal() {
    let a = array![1.0_f64, 0.0, 0.0];
    let b = array![0.0_f64, 1.0, 0.0];
    let d = distance_cosine_simd::<f64>(&a.view(), &b.view());
    // Orthogonal vectors: cosine similarity = 0, distance = 1
    assert!(
        (d - 1.0).abs() < 1e-10,
        "Cosine distance for orthogonal vectors should be 1.0, got {}",
        d
    );
}

/// Test distance_cosine_simd parallel
#[test]
fn test_distance_cosine_simd_parallel() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![2.0_f64, 4.0, 6.0];
    let d = distance_cosine_simd::<f64>(&a.view(), &b.view());
    // Parallel vectors: cosine similarity = 1, distance = 0
    assert!(
        (d - 0.0).abs() < 1e-10,
        "Cosine distance for parallel vectors should be 0.0, got {}",
        d
    );
}

/// Test cosine_similarity_simd parallel
#[test]
fn test_cosine_similarity_simd_parallel() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![2.0_f64, 4.0, 6.0];
    let sim = cosine_similarity_simd::<f64>(&a.view(), &b.view());
    assert!(
        (sim - 1.0).abs() < 1e-10,
        "Cosine similarity for parallel vectors should be 1.0, got {}",
        sim
    );
}

/// Test cosine_similarity_simd orthogonal
#[test]
fn test_cosine_similarity_simd_orthogonal() {
    let a = array![1.0_f64, 0.0];
    let b = array![0.0_f64, 1.0];
    let sim = cosine_similarity_simd::<f64>(&a.view(), &b.view());
    assert!(
        (sim - 0.0).abs() < 1e-10,
        "Cosine similarity for orthogonal vectors should be 0.0, got {}",
        sim
    );
}

/// Test cosine_similarity_simd f32
#[test]
fn test_cosine_similarity_simd_f32() {
    let a = array![1.0_f32, 1.0];
    let b = array![1.0_f32, 0.0];
    let sim = cosine_similarity_simd::<f32>(&a.view(), &b.view());
    // cos(45°) = 1/sqrt(2) ≈ 0.707
    let expected = 1.0 / 2.0_f32.sqrt();
    assert!(
        (sim - expected).abs() < 1e-5,
        "Cosine similarity should be {}, got {}",
        expected,
        sim
    );
}

// =============================================================================
// Phase 73: Binary Operations Tests
// =============================================================================

/// Test add_simd basic
#[test]
fn test_add_simd_basic() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![4.0_f64, 5.0, 6.0];
    let c = add_simd::<f64>(&a.view(), &b.view());
    assert!((c[0] - 5.0).abs() < 1e-14);
    assert!((c[1] - 7.0).abs() < 1e-14);
    assert!((c[2] - 9.0).abs() < 1e-14);
}

/// Test add_simd f32
#[test]
fn test_add_simd_f32() {
    let a = array![1.5_f32, 2.5];
    let b = array![3.5_f32, 4.5];
    let c = add_simd::<f32>(&a.view(), &b.view());
    assert!((c[0] - 5.0).abs() < 1e-5);
    assert!((c[1] - 7.0).abs() < 1e-5);
}

/// Test sub_simd basic
#[test]
fn test_sub_simd_basic() {
    let a = array![5.0_f64, 7.0, 9.0];
    let b = array![1.0_f64, 2.0, 3.0];
    let c = sub_simd::<f64>(&a.view(), &b.view());
    assert!((c[0] - 4.0).abs() < 1e-14);
    assert!((c[1] - 5.0).abs() < 1e-14);
    assert!((c[2] - 6.0).abs() < 1e-14);
}

/// Test sub_simd f32
#[test]
fn test_sub_simd_f32() {
    let a = array![10.0_f32, 20.0];
    let b = array![3.0_f32, 5.0];
    let c = sub_simd::<f32>(&a.view(), &b.view());
    assert!((c[0] - 7.0).abs() < 1e-5);
    assert!((c[1] - 15.0).abs() < 1e-5);
}

/// Test mul_simd basic
#[test]
fn test_mul_simd_basic() {
    let a = array![2.0_f64, 3.0, 4.0];
    let b = array![5.0_f64, 6.0, 7.0];
    let c = mul_simd::<f64>(&a.view(), &b.view());
    assert!((c[0] - 10.0).abs() < 1e-14);
    assert!((c[1] - 18.0).abs() < 1e-14);
    assert!((c[2] - 28.0).abs() < 1e-14);
}

/// Test mul_simd f32
#[test]
fn test_mul_simd_f32() {
    let a = array![2.0_f32, 0.5];
    let b = array![4.0_f32, 8.0];
    let c = mul_simd::<f32>(&a.view(), &b.view());
    assert!((c[0] - 8.0).abs() < 1e-5);
    assert!((c[1] - 4.0).abs() < 1e-5);
}

/// Test div_simd basic
#[test]
fn test_div_simd_basic() {
    let a = array![10.0_f64, 20.0, 30.0];
    let b = array![2.0_f64, 4.0, 5.0];
    let c = div_simd::<f64>(&a.view(), &b.view());
    assert!((c[0] - 5.0).abs() < 1e-14);
    assert!((c[1] - 5.0).abs() < 1e-14);
    assert!((c[2] - 6.0).abs() < 1e-14);
}

/// Test div_simd f32
#[test]
fn test_div_simd_f32() {
    let a = array![9.0_f32, 16.0];
    let b = array![3.0_f32, 4.0];
    let c = div_simd::<f32>(&a.view(), &b.view());
    assert!((c[0] - 3.0).abs() < 1e-5);
    assert!((c[1] - 4.0).abs() < 1e-5);
}

/// Test max_simd basic
#[test]
fn test_max_simd_basic() {
    let a = array![1.0_f64, 5.0, 3.0];
    let b = array![4.0_f64, 2.0, 6.0];
    let c = max_simd::<f64>(&a.view(), &b.view());
    assert!((c[0] - 4.0).abs() < 1e-14);
    assert!((c[1] - 5.0).abs() < 1e-14);
    assert!((c[2] - 6.0).abs() < 1e-14);
}

/// Test max_simd f32
#[test]
fn test_max_simd_f32() {
    let a = array![-1.0_f32, 0.0, 1.0];
    let b = array![0.0_f32, 0.0, 0.0];
    let c = max_simd::<f32>(&a.view(), &b.view());
    assert!((c[0] - 0.0).abs() < 1e-5);
    assert!((c[1] - 0.0).abs() < 1e-5);
    assert!((c[2] - 1.0).abs() < 1e-5);
}

/// Test min_simd basic
#[test]
fn test_min_simd_basic() {
    let a = array![1.0_f64, 5.0, 3.0];
    let b = array![4.0_f64, 2.0, 6.0];
    let c = min_simd::<f64>(&a.view(), &b.view());
    assert!((c[0] - 1.0).abs() < 1e-14);
    assert!((c[1] - 2.0).abs() < 1e-14);
    assert!((c[2] - 3.0).abs() < 1e-14);
}

/// Test min_simd f32
#[test]
fn test_min_simd_f32() {
    let a = array![10.0_f32, 5.0];
    let b = array![7.0_f32, 8.0];
    let c = min_simd::<f32>(&a.view(), &b.view());
    assert!((c[0] - 7.0).abs() < 1e-5);
    assert!((c[1] - 5.0).abs() < 1e-5);
}

/// Test scalar_mul_simd basic
#[test]
fn test_scalar_mul_simd_basic() {
    let a = array![1.0_f64, 2.0, 3.0];
    let c = scalar_mul_simd::<f64>(&a.view(), 2.5);
    assert!((c[0] - 2.5).abs() < 1e-14);
    assert!((c[1] - 5.0).abs() < 1e-14);
    assert!((c[2] - 7.5).abs() < 1e-14);
}

/// Test scalar_mul_simd f32
#[test]
fn test_scalar_mul_simd_f32() {
    let a = array![2.0_f32, 4.0];
    let c = scalar_mul_simd::<f32>(&a.view(), 0.5);
    assert!((c[0] - 1.0).abs() < 1e-5);
    assert!((c[1] - 2.0).abs() < 1e-5);
}

/// Test fma_simd basic
#[test]
fn test_fma_simd_basic() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![4.0_f64, 5.0, 6.0];
    let c = array![7.0_f64, 8.0, 9.0];
    let result = fma_simd::<f64>(&a.view(), &b.view(), &c.view());
    // 1*4+7=11, 2*5+8=18, 3*6+9=27
    assert!((result[0] - 11.0).abs() < 1e-14);
    assert!((result[1] - 18.0).abs() < 1e-14);
    assert!((result[2] - 27.0).abs() < 1e-14);
}

/// Test fma_simd f32
#[test]
fn test_fma_simd_f32() {
    let a = array![2.0_f32, 3.0];
    let b = array![4.0_f32, 5.0];
    let c = array![1.0_f32, 2.0];
    let result = fma_simd::<f32>(&a.view(), &b.view(), &c.view());
    // 2*4+1=9, 3*5+2=17
    assert!((result[0] - 9.0).abs() < 1e-5);
    assert!((result[1] - 17.0).abs() < 1e-5);
}

/// Test dot_simd basic
#[test]
fn test_dot_simd_basic() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![4.0_f64, 5.0, 6.0];
    let d = dot_simd::<f64>(&a.view(), &b.view());
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert!(
        (d - 32.0).abs() < 1e-14,
        "Dot product should be 32.0, got {}",
        d
    );
}

/// Test dot_simd f32
#[test]
fn test_dot_simd_f32() {
    let a = array![1.0_f32, 0.0, 1.0];
    let b = array![2.0_f32, 3.0, 4.0];
    let d = dot_simd::<f32>(&a.view(), &b.view());
    // 1*2 + 0*3 + 1*4 = 6
    assert!(
        (d - 6.0).abs() < 1e-5,
        "Dot product should be 6.0, got {}",
        d
    );
}

/// Test dot_simd orthogonal
#[test]
fn test_dot_simd_orthogonal() {
    let a = array![1.0_f64, 0.0];
    let b = array![0.0_f64, 1.0];
    let d = dot_simd::<f64>(&a.view(), &b.view());
    assert!(
        (d - 0.0).abs() < 1e-14,
        "Dot product of orthogonal vectors should be 0.0"
    );
}

// =============================================================================
// Phase 74: Normalization and Activation Functions Tests
// =============================================================================

/// Test relu_simd basic
#[test]
fn test_relu_simd_basic() {
    let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let result = relu_simd::<f64>(&x.view());
    assert!((result[0] - 0.0).abs() < 1e-14);
    assert!((result[1] - 0.0).abs() < 1e-14);
    assert!((result[2] - 0.0).abs() < 1e-14);
    assert!((result[3] - 1.0).abs() < 1e-14);
    assert!((result[4] - 2.0).abs() < 1e-14);
}

/// Test relu_simd f32
#[test]
fn test_relu_simd_f32() {
    let x = array![-5.0_f32, 0.0, 5.0];
    let result = relu_simd::<f32>(&x.view());
    assert!((result[0] - 0.0).abs() < 1e-5);
    assert!((result[1] - 0.0).abs() < 1e-5);
    assert!((result[2] - 5.0).abs() < 1e-5);
}

/// Test relu_simd all negative
#[test]
fn test_relu_simd_all_negative() {
    let x = array![-3.0_f64, -2.0, -1.0];
    let result = relu_simd::<f64>(&x.view());
    for val in result.iter() {
        assert!(*val == 0.0, "ReLU of negative should be 0");
    }
}

/// Test normalize_simd basic
#[test]
fn test_normalize_simd_basic() {
    let x = array![3.0_f64, 4.0];
    let result = normalize_simd::<f64>(&x.view());
    // ||result|| = 1
    let norm = (result[0] * result[0] + result[1] * result[1]).sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-10,
        "Normalized vector should have unit length"
    );
    // x/5 = [0.6, 0.8]
    assert!((result[0] - 0.6).abs() < 1e-10);
    assert!((result[1] - 0.8).abs() < 1e-10);
}

/// Test normalize_simd f32
#[test]
fn test_normalize_simd_f32() {
    let x = array![1.0_f32, 1.0, 1.0, 1.0];
    let result = normalize_simd::<f32>(&x.view());
    // ||result|| = 1
    let norm: f32 = result.iter().map(|&v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "Normalized vector should have unit length"
    );
}

/// Test normalize_simd preserves direction
#[test]
fn test_normalize_simd_direction() {
    let x = array![2.0_f64, 4.0, 6.0];
    let result = normalize_simd::<f64>(&x.view());
    // All elements should have same ratio
    let ratio1 = result[1] / result[0];
    let ratio2 = result[2] / result[0];
    assert!(
        (ratio1 - 2.0).abs() < 1e-10,
        "Direction should be preserved"
    );
    assert!(
        (ratio2 - 3.0).abs() < 1e-10,
        "Direction should be preserved"
    );
}

/// Test standardize_simd basic
#[test]
fn test_standardize_simd_basic() {
    let x = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let result = standardize_simd::<f64>(&x.view());
    // Mean should be ~0
    let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
    assert!(
        mean.abs() < 1e-10,
        "Standardized mean should be 0, got {}",
        mean
    );
}

/// Test standardize_simd f32
#[test]
fn test_standardize_simd_f32() {
    let x = array![1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let result = standardize_simd::<f32>(&x.view());
    let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
    assert!(
        mean.abs() < 1e-4,
        "Standardized mean should be 0, got {}",
        mean
    );
}

/// Test softmax_simd sums to one
#[test]
fn test_softmax_simd_sums_to_one() {
    let x = array![1.0_f64, 2.0, 3.0];
    let result = softmax_simd::<f64>(&x.view());
    let sum: f64 = result.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "Softmax should sum to 1.0, got {}",
        sum
    );
}

/// Test softmax_simd ordering preserved
#[test]
fn test_softmax_simd_ordering() {
    let x = array![1.0_f64, 2.0, 3.0];
    let result = softmax_simd::<f64>(&x.view());
    assert!(
        result[2] > result[1],
        "Higher logit should have higher probability"
    );
    assert!(
        result[1] > result[0],
        "Higher logit should have higher probability"
    );
}

/// Test softmax_simd f32
#[test]
fn test_softmax_simd_f32() {
    let x = array![0.0_f32, 0.0, 0.0];
    let result = softmax_simd::<f32>(&x.view());
    // Equal inputs should give equal probabilities
    let expected = 1.0_f32 / 3.0;
    for val in result.iter() {
        assert!(
            (val - expected).abs() < 1e-5,
            "Equal inputs should give equal probabilities"
        );
    }
}

/// Test softmax_simd numerical stability
#[test]
fn test_softmax_simd_stability() {
    let x = array![1000.0_f64, 1001.0, 1002.0];
    let result = softmax_simd::<f64>(&x.view());
    let sum: f64 = result.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-10,
        "Softmax should handle large values stably"
    );
    assert!(
        result.iter().all(|&v| v.is_finite()),
        "All softmax values should be finite"
    );
}

/// Test trunc_simd basic
#[test]
fn test_trunc_simd_basic() {
    let x = array![2.7_f64, -2.7, 0.9, -0.9];
    let result = trunc_simd::<f64>(&x.view());
    assert!((result[0] - 2.0).abs() < 1e-14);
    assert!((result[1] - (-2.0)).abs() < 1e-14);
    assert!((result[2] - 0.0).abs() < 1e-14);
    assert!((result[3] - 0.0).abs() < 1e-14);
}

/// Test trunc_simd f32
#[test]
fn test_trunc_simd_f32() {
    let x = array![3.9_f32, -3.9];
    let result = trunc_simd::<f32>(&x.view());
    assert!((result[0] - 3.0).abs() < 1e-5);
    assert!((result[1] - (-3.0)).abs() < 1e-5);
}

/// Test trunc_simd integers unchanged
#[test]
fn test_trunc_simd_integers() {
    let x = array![1.0_f64, 2.0, -3.0, 0.0];
    let result = trunc_simd::<f64>(&x.view());
    for i in 0..x.len() {
        assert!(
            (result[i] - x[i]).abs() < 1e-14,
            "Integer values should be unchanged"
        );
    }
}

// =============================================================================
// Integration and Edge Case Tests
// =============================================================================

/// Test large array operations
#[test]
fn test_large_array_operations() {
    let n = 10000;
    let x: Array1<f64> = Array1::from_vec((0..n).map(|i| i as f64).collect());
    let y: Array1<f64> = Array1::from_vec((0..n).map(|i| (i * 2) as f64).collect());

    // Test add
    let sum = add_simd::<f64>(&x.view(), &y.view());
    assert!((sum[0] - 0.0).abs() < 1e-10);
    assert!((sum[n - 1] - (3.0 * (n - 1) as f64)).abs() < 1e-10);

    // Test dot
    let d = dot_simd::<f64>(&x.view(), &y.view());
    // sum(i * 2i) = 2 * sum(i^2) = 2 * n*(n-1)*(2n-1)/6
    let expected = 2.0 * (n as f64) * ((n - 1) as f64) * ((2 * n - 1) as f64) / 6.0;
    assert!((d - expected).abs() / expected < 1e-10, "Large dot product");
}

/// Test empty array handling
#[test]
fn test_empty_array_handling() {
    let empty: Array1<f64> = Array1::zeros(0);
    let non_empty = array![1.0_f64, 2.0, 3.0];

    assert!(sum_simd::<f64>(&empty.view()).abs() < 1e-14);
    assert!(mean_simd::<f64>(&empty.view()).abs() < 1e-14);
    assert_eq!(add_simd::<f64>(&empty.view(), &non_empty.view()).len(), 0);
    assert_eq!(relu_simd::<f64>(&empty.view()).len(), 0);
    assert_eq!(softmax_simd::<f64>(&empty.view()).len(), 0);
}

/// Test numerical properties of distance metrics
#[test]
fn test_distance_metric_properties() {
    let a = array![1.0_f64, 2.0, 3.0];
    let b = array![4.0_f64, 5.0, 6.0];
    let c = array![7.0_f64, 8.0, 9.0];

    // Identity: d(a, a) = 0
    assert!(distance_euclidean_simd::<f64>(&a.view(), &a.view()).abs() < 1e-14);
    assert!(distance_manhattan_simd::<f64>(&a.view(), &a.view()).abs() < 1e-14);
    assert!(distance_chebyshev_simd::<f64>(&a.view(), &a.view()).abs() < 1e-14);

    // Symmetry: d(a, b) = d(b, a)
    let d_ab = distance_euclidean_simd::<f64>(&a.view(), &b.view());
    let d_ba = distance_euclidean_simd::<f64>(&b.view(), &a.view());
    assert!((d_ab - d_ba).abs() < 1e-14);

    // Triangle inequality: d(a, c) <= d(a, b) + d(b, c)
    let d_ac = distance_euclidean_simd::<f64>(&a.view(), &c.view());
    let d_bc = distance_euclidean_simd::<f64>(&b.view(), &c.view());
    assert!(d_ac <= d_ab + d_bc + 1e-10);
}

/// Test statistical consistency
#[test]
fn test_statistical_consistency() {
    let x = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

    let mean = mean_simd::<f64>(&x.view());
    let var = variance_simd::<f64>(&x.view());
    let std = std_simd::<f64>(&x.view());

    // std = sqrt(var)
    assert!((std - var.sqrt()).abs() < 1e-10);

    // Sample variance = sum((x - mean)^2) / (n-1) (Bessel's correction)
    let n = x.len() as f64;
    let manual_var: f64 = x.iter().map(|&xi| (xi - mean).powi(2)).sum::<f64>() / (n - 1.0);
    assert!((var - manual_var).abs() < 1e-10);
}
