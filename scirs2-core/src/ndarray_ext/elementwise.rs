//! SIMD-accelerated element-wise mathematical operations
//!
//! This module provides high-performance implementations of common element-wise
//! mathematical functions that are fundamental for scientific computing, numerical
//! analysis, and data processing.
//!
//! # Operations
//!
//! ## Basic Operations
//! - **Absolute Value** (`abs_simd`): Computes |x| for each element
//! - **Sign** (`sign_simd`): Computes sign(x) = +1, 0, or -1 for each element
//! - **Square Root** (`sqrt_simd`): Computes √x for each element
//!
//! ## Exponential and Logarithmic
//! - **Exponential** (`exp_simd`): Computes e^x for each element
//! - **Natural Logarithm** (`ln_simd`): Computes ln(x) for each element
//!
//! ## Trigonometric Functions
//! - **Sine** (`sin_simd`): Computes sin(x) for each element
//! - **Cosine** (`cos_simd`): Computes cos(x) for each element
//! - **Tangent** (`tan_simd`): Computes tan(x) for each element
//!
//! ## Hyperbolic Functions
//! - **Hyperbolic Sine** (`sinh_simd`): Computes sinh(x) for each element
//! - **Hyperbolic Cosine** (`cosh_simd`): Computes cosh(x) for each element
//! - **Hyperbolic Tangent** (`tanh_simd`): Computes tanh(x) for each element
//!
//! ## Power Functions
//! - **Power (scalar)** (`powf_simd`): Computes base^exp with scalar exponent
//! - **Power (array)** (`pow_simd`): Computes `base[i]^exp[i]` element-wise
//!
//! ## Rounding Functions
//! - **Floor** (`floor_simd`): Rounds down to largest integer <= x
//! - **Ceiling** (`ceil_simd`): Rounds up to smallest integer >= x
//! - **Round** (`round_simd`): Rounds to nearest integer (ties to even)
//! - **Fractional Part** (`fract_simd`): Returns fractional part (x - floor(x))
//!
//! ## Inverse Trigonometric Functions
//! - **Arctangent** (`atan_simd`): Computes atan(x) for each element
//! - **Arcsine** (`asin_simd`): Computes asin(x) for each element
//! - **Arccosine** (`acos_simd`): Computes acos(x) for each element
//! - **Two-argument arctangent** (`atan2_simd`): Computes atan2(y, x) element-wise
//!
//! ## Logarithm Variants
//! - **Base-10 Logarithm** (`log10_simd`): Computes log₁₀(x) for each element
//! - **Base-2 Logarithm** (`log2_simd`): Computes log₂(x) for each element
//!
//! ## Clamping Operations
//! - **Clamp** (`clamp_simd`): Constrains each element to [min, max] range
//!
//! # Performance
//!
//! All operations automatically use SIMD acceleration when:
//! - Platform supports AVX2 (x86_64) or NEON (ARM)
//! - Array size is large enough to benefit from vectorization
//! - Array memory layout is contiguous
//!
//! Falls back to scalar implementations for small arrays or unsupported platforms.
//!
//! # Examples
//!
//! ```
//! use scirs2_core::ndarray::array;
//! use scirs2_core::ndarray_ext::elementwise::{abs_simd, sign_simd, sqrt_simd, sinh_simd, tanh_simd, floor_simd, atan2_simd, log10_simd, clamp_simd};
//!
//! // Absolute value
//! let x = array![-3.0, -1.0, 0.0, 1.0, 3.0];
//! let abs_x = abs_simd(&x.view());
//! // Result: [3.0, 1.0, 0.0, 1.0, 3.0]
//!
//! // Sign function (signum)
//! let signs = sign_simd(&x.view());
//! // Result: [-1.0, -1.0, 0.0, 1.0, 1.0]
//!
//! // Square root
//! let y = array![1.0, 4.0, 9.0, 16.0, 25.0];
//! let sqrt_y = sqrt_simd(&y.view());
//! // Result: [1.0, 2.0, 3.0, 4.0, 5.0]
//!
//! // Hyperbolic tangent (neural network activation)
//! let z = array![-1.0, 0.0, 1.0];
//! let tanh_z = tanh_simd(&z.view());
//! // Result: [-0.762, 0.0, 0.762]
//!
//! // Floor (rounding down)
//! let w = array![1.2, 2.7, -1.3, -2.9];
//! let floor_w = floor_simd(&w.view());
//! // Result: [1.0, 2.0, -2.0, -3.0]
//!
//! // Arctangent 2 (angle from coordinates)
//! let y_coords = array![1.0, 1.0, -1.0, -1.0];
//! let x_coords = array![1.0, -1.0, -1.0, 1.0];
//! let angles = atan2_simd(&y_coords.view(), &x_coords.view());
//! // Result: [π/4, 3π/4, -3π/4, -π/4]
//!
//! // Base-10 logarithm (decibels, pH scale)
//! let powers = array![1.0, 10.0, 100.0, 1000.0];
//! let log10_powers = log10_simd(&powers.view());
//! // Result: [0.0, 1.0, 2.0, 3.0]
//!
//! // Clamp values to range (pixel normalization)
//! let pixels = array![-0.5, 0.3, 0.7, 1.2, 1.8];
//! let normalized = clamp_simd(&pixels.view(), 0.0, 1.0);
//! // Result: [0.0, 0.3, 0.7, 1.0, 1.0]
//! ```

use crate::numeric::Float;
use crate::simd_ops::{AutoOptimizer, SimdUnifiedOps};
use ::ndarray::{Array1, ArrayView1};

/// Compute the absolute value of each element (SIMD-accelerated).
///
/// Computes |x| for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with absolute values.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD unavailable
/// - **Speedup**: 2-4x for large f32 arrays on AVX2 systems
///
/// # Mathematical Definition
///
/// ```text
/// abs(x) = |x| = {
///     x   if x >= 0
///     -x  if x < 0
/// }
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::abs_simd;
///
/// let x = array![-3.0, -1.5, 0.0, 1.5, 3.0];
/// let result = abs_simd(&x.view());
///
/// assert_eq!(result[0], 3.0);
/// assert_eq!(result[1], 1.5);
/// assert_eq!(result[2], 0.0);
/// assert_eq!(result[3], 1.5);
/// assert_eq!(result[4], 3.0);
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: Returns zero
/// - **NaN**: Returns NaN (preserves NaN)
/// - **Infinity**: Returns positive infinity
///
/// # Applications
///
/// - **Statistics**: Absolute deviation, MAE (Mean Absolute Error)
/// - **Signal Processing**: Envelope detection, magnitude calculation
/// - **Optimization**: L1 norm computation, absolute loss functions
/// - **Numerical Analysis**: Error analysis, residual computation
/// - **Image Processing**: Gradient magnitude, difference images
pub fn abs_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_abs(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.abs())
}

/// Compute the sign (signum) of each element (SIMD-accelerated).
///
/// Returns +1.0 for positive values, -1.0 for negative values, and 0.0 for zero.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the sign of the input.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD unavailable
/// - **Speedup**: 2-4x for large arrays on AVX2/NEON systems
///
/// # Mathematical Definition
///
/// ```text
/// sign(x) = { +1.0  if x > 0
///           {  0.0  if x = 0
///           { -1.0  if x < 0
/// ```
///
/// # Properties
///
/// - sign(-x) = -sign(x) (odd function)
/// - sign(0) = 0
/// - sign(x) * |x| = x
/// - sign(x) * sign(y) = sign(x * y)
/// - |sign(x)| ≤ 1 for all x
///
/// # Applications
///
/// - **Numerical Analysis**: Gradient descent direction, optimization algorithms
/// - **Signal Processing**: Phase detection, zero-crossing analysis
/// - **Physics Simulations**: Force direction, velocity direction
/// - **Machine Learning**: Feature engineering, binary classification features
/// - **Control Systems**: Error sign for PID controllers
/// - **Game Development**: Direction vectors, collision normals
/// - **Statistics**: Wilcoxon signed-rank test, sign test
/// - **Finance**: Market trend indicators (bull/bear)
/// - **Geometry**: Surface normal orientation, vector direction
/// - **Image Processing**: Edge direction, gradient orientation
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sign_simd;
///
/// let x = array![-3.0_f64, -1.5, 0.0, 1.5, 3.0];
/// let result = sign_simd(&x.view());
/// assert_eq!(result[0], -1.0_f64); // negative → -1
/// assert_eq!(result[1], -1.0_f64); // negative → -1
/// assert_eq!(result[2],  0.0_f64); // zero → 0
/// assert_eq!(result[3],  1.0_f64); // positive → +1
/// assert_eq!(result[4],  1.0_f64); // positive → +1
///
/// // Property: sign(x) * |x| = x
/// let values = array![-5.0_f64, -2.0, 0.0, 2.0, 5.0];
/// let signs = sign_simd(&values.view());
/// let abs_values = values.mapv(|x: f64| x.abs());
/// let reconstructed = signs * abs_values;
/// for i in 0..values.len() {
///     assert!((reconstructed[i] - values[i]).abs() < 1e-10);
/// }
/// ```
///
/// # See Also
///
/// - [`abs_simd`]: Magnitude without sign
/// - [`clamp_simd`]: Constrain values to a range
pub fn sign_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_sign(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| {
        if val > F::zero() {
            F::one()
        } else if val < F::zero() {
            -F::one()
        } else {
            F::zero()
        }
    })
}

/// Compute the square root of each element (SIMD-accelerated).
///
/// Computes √x for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array (must contain non-negative values)
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with square roots.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD unavailable
/// - **Speedup**: 2-4x for large f32 arrays on AVX2 systems
///
/// # Mathematical Definition
///
/// ```text
/// sqrt(x) = √x = y such that y² = x
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sqrt_simd;
///
/// let x = array![1.0_f64, 4.0, 9.0, 16.0, 25.0];
/// let result = sqrt_simd(&x.view());
///
/// assert_eq!(result[0], 1.0);
/// assert_eq!(result[1], 2.0);
/// assert_eq!(result[2], 3.0);
/// assert_eq!(result[3], 4.0);
/// assert_eq!(result[4], 5.0);
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: Returns zero
/// - **Negative numbers**: Returns NaN (undefined in real numbers)
/// - **NaN**: Returns NaN (preserves NaN)
/// - **Positive infinity**: Returns positive infinity
///
/// # Applications
///
/// - **Statistics**: Standard deviation, RMS (Root Mean Square)
/// - **Geometry**: Distance calculations, Euclidean metrics
/// - **Physics**: Velocity from kinetic energy, magnitude calculations
/// - **Machine Learning**: Gradient scaling, learning rate schedules
/// - **Image Processing**: Gaussian blur, distance transforms
///
/// # Note on Negative Values
///
/// Square root of negative numbers is undefined in real arithmetic.
/// The function will return NaN for negative inputs, following IEEE 754 standard.
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sqrt_simd;
///
/// let x = array![-1.0_f64];
/// let result = sqrt_simd(&x.view());
/// assert!(result[0].is_nan());
/// ```
pub fn sqrt_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_sqrt(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.sqrt())
}

/// Compute the exponential (e^x) of each element (SIMD-accelerated).
///
/// Computes e^x for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with exponential values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// exp(x) = e^x where e ≈ 2.71828...
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::exp_simd;
///
/// let x = array![0.0_f64, 1.0, 2.0];
/// let result = exp_simd(&x.view());
///
/// assert!((result[0] - 1.0).abs() < 1e-10);
/// assert!((result[1] - 2.718281828).abs() < 1e-9);
/// assert!((result[2] - 7.389056099).abs() < 1e-9);
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: exp(0) = 1
/// - **Large positive**: May overflow to infinity
/// - **Large negative**: Approaches zero
/// - **NaN**: Returns NaN (preserves NaN)
///
/// # Applications
///
/// - **Machine Learning**: Softmax, sigmoid activation
/// - **Optimization**: Exponential decay, learning rate schedules
/// - **Probability**: Exponential distribution, Gaussian PDF
/// - **Neural Networks**: Attention mechanisms, transformer models
/// - **Reinforcement Learning**: Policy gradients, Q-learning
pub fn exp_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_exp(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.exp())
}

/// Compute the natural logarithm (ln(x)) of each element (SIMD-accelerated).
///
/// Computes ln(x) for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array (must contain positive values)
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with natural logarithm values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// ln(x) = log_e(x) = y such that e^y = x
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::ln_simd;
///
/// let x = array![1.0_f64, 2.718281828, 7.389056099];
/// let result = ln_simd(&x.view());
///
/// assert!((result[0] - 0.0).abs() < 1e-10);
/// assert!((result[1] - 1.0).abs() < 1e-9);
/// assert!((result[2] - 2.0).abs() < 1e-9);
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **One**: ln(1) = 0
/// - **Zero**: Returns negative infinity
/// - **Negative numbers**: Returns NaN (undefined in reals)
/// - **NaN**: Returns NaN (preserves NaN)
/// - **Positive infinity**: Returns positive infinity
///
/// # Applications
///
/// - **Machine Learning**: Log-likelihood, cross-entropy loss
/// - **Statistics**: Log-normal distribution, Shannon entropy
/// - **Optimization**: Log-barrier methods, logarithmic objectives
/// - **Automatic Differentiation**: Logarithmic derivatives
/// - **Information Theory**: Mutual information, KL divergence
///
/// # Note on Negative Values
///
/// Natural logarithm of negative numbers is undefined in real arithmetic.
/// The function will return NaN for negative inputs, following IEEE 754 standard.
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::ln_simd;
///
/// let x = array![-1.0_f64];
/// let result = ln_simd(&x.view());
/// assert!(result[0].is_nan());
/// ```
pub fn ln_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_ln(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.ln())
}

/// Compute the sine of each element (SIMD-accelerated).
///
/// Computes sin(x) for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array (angles in radians)
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with sine values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// sin(x) = opposite/hypotenuse in a right triangle
/// Periodic with period 2π: sin(x + 2π) = sin(x)
/// Range: [-1, 1]
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sin_simd;
/// use std::f64::consts::PI;
///
/// let x = array![0.0_f64, PI/6.0, PI/4.0, PI/2.0, PI];
/// let result = sin_simd(&x.view());
///
/// // sin(0) = 0, sin(π/2) = 1, sin(π) ≈ 0
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: sin(0) = 0
/// - **NaN**: Returns NaN (preserves NaN)
/// - **Infinity**: Returns NaN (undefined for infinity)
///
/// # Applications
///
/// - **Signal Processing**: Wave generation, modulation, Fourier analysis
/// - **Computer Graphics**: Rotation matrices, smooth animations
/// - **Physics Simulations**: Oscillations, wave propagation
/// - **Audio**: Synthesizers, tone generation
/// - **Robotics**: Trajectory planning, inverse kinematics
pub fn sin_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_sin(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.sin())
}

/// Compute the cosine of each element (SIMD-accelerated).
///
/// Computes cos(x) for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array (angles in radians)
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with cosine values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// cos(x) = adjacent/hypotenuse in a right triangle
/// Periodic with period 2π: cos(x + 2π) = cos(x)
/// Range: [-1, 1]
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::cos_simd;
/// use std::f64::consts::PI;
///
/// let x = array![0.0_f64, PI/3.0, PI/2.0, PI];
/// let result = cos_simd(&x.view());
///
/// // cos(0) = 1, cos(π/2) ≈ 0, cos(π) = -1
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: cos(0) = 1
/// - **NaN**: Returns NaN (preserves NaN)
/// - **Infinity**: Returns NaN (undefined for infinity)
///
/// # Applications
///
/// - **Computer Vision**: Rotation matrices, coordinate transformations
/// - **Signal Processing**: Fourier transforms, filtering
/// - **Path Planning**: Dubins curves, trajectory optimization
/// - **Geometry**: Circle parametrization, spherical coordinates
/// - **Neural Networks**: Positional encoding (transformers)
pub fn cos_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_cos(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.cos())
}

/// Compute the tangent of each element (SIMD-accelerated).
///
/// Computes tan(x) for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array (angles in radians)
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with tangent values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// tan(x) = sin(x) / cos(x)
/// Periodic with period π: tan(x + π) = tan(x)
/// Range: (-∞, ∞)
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::tan_simd;
/// use std::f64::consts::PI;
///
/// let x = array![0.0_f64, PI/4.0, PI/6.0];
/// let result = tan_simd(&x.view());
///
/// // tan(0) = 0, tan(π/4) = 1
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: tan(0) = 0
/// - **π/2, 3π/2, ...**: Returns ±infinity (undefined at cos(x)=0)
/// - **NaN**: Returns NaN (preserves NaN)
/// - **Infinity**: Returns NaN (undefined for infinity)
///
/// # Applications
///
/// - **Computer Graphics**: Perspective projection, field of view
/// - **Navigation**: Bearing calculations, angle determination
/// - **Physics**: Slope calculations, inclined planes
/// - **Image Processing**: Gradient direction, edge angles
/// - **Surveying**: Distance and height measurements
///
/// # Note on Singularities
///
/// Tangent has singularities at x = π/2 + nπ where n is any integer.
/// At these points, cos(x) = 0 and tan(x) approaches ±infinity.
pub fn tan_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_tan(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.tan())
}

/// Compute the power of each element with a scalar exponent (SIMD-accelerated).
///
/// Computes base^exp for each element in the base array.
///
/// # Arguments
///
/// * `base` - Input 1D array of base values
/// * `exp` - Scalar exponent value
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with power values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// powf(base, exp) = base^exp
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::powf_simd;
///
/// let base = array![2.0_f64, 3.0, 4.0, 5.0];
/// let result = powf_simd(&base.view(), 2.0);
///
/// // [4.0, 9.0, 16.0, 25.0]
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **x^0**: Returns 1 for any x (including 0^0 = 1 by convention)
/// - **x^1**: Returns x
/// - **0^negative**: Returns infinity
/// - **negative^non-integer**: Returns NaN
/// - **NaN**: Returns NaN (preserves NaN)
///
/// # Applications
///
/// - **Statistics**: Variance, standard deviation, moment calculations
/// - **Machine Learning**: Polynomial features, L2 regularization
/// - **Signal Processing**: Power spectral density, energy calculations
/// - **Physics**: Kinetic energy, gravitational potential
/// - **Finance**: Compound interest, exponential growth models
pub fn powf_simd<F>(base: &ArrayView1<F>, exp: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if base.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(base.len()) {
        return F::simd_powf(base, exp);
    }

    // Scalar fallback for small arrays
    base.mapv(|val| val.powf(exp))
}

/// Compute the element-wise power with array exponents (SIMD-accelerated).
///
/// Computes `base[i]^exp[i]` for each pair of elements.
///
/// # Arguments
///
/// * `base` - Input 1D array of base values
/// * `exp` - Input 1D array of exponent values (must match base length)
///
/// # Returns
///
/// `Array1<F>` with the same length as inputs, with power values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// pow(base, exp)[i] = base[i]^exp[i]
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::pow_simd;
///
/// let base = array![2.0_f64, 3.0, 4.0, 5.0];
/// let exp = array![2.0, 3.0, 2.0, 1.0];
/// let result = pow_simd(&base.view(), &exp.view());
///
/// // [4.0, 27.0, 16.0, 5.0]
/// ```
///
/// # Edge Cases
///
/// - **Empty arrays**: Returns empty array
/// - **Mismatched lengths**: Panics (arrays must have same length)
/// - **x^0**: Returns 1 for any x
/// - **0^0**: Returns 1 by convention
/// - **0^negative**: Returns infinity
/// - **negative^non-integer**: Returns NaN
/// - **NaN**: Returns NaN (preserves NaN)
///
/// # Applications
///
/// - **Machine Learning**: Custom activation functions, attention mechanisms
/// - **Statistics**: Generalized power transformations
/// - **Optimization**: Power-law scaling, Pareto distributions
/// - **Signal Processing**: Non-linear transformations, compression
/// - **Physics**: Variable exponent models, fractal dimensions
pub fn pow_simd<F>(base: &ArrayView1<F>, exp: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    assert_eq!(
        base.len(),
        exp.len(),
        "Base and exponent arrays must have the same length"
    );

    if base.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(base.len()) {
        return F::simd_pow(base, exp);
    }

    // Scalar fallback for small arrays
    base.iter()
        .zip(exp.iter())
        .map(|(&b, &e)| b.powf(e))
        .collect::<Vec<_>>()
        .into()
}

/// Compute the hyperbolic sine of each element (SIMD-accelerated).
///
/// Computes sinh(x) for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with hyperbolic sine values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// sinh(x) = (e^x - e^(-x)) / 2
/// Range: (-∞, ∞)
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sinh_simd;
///
/// let x = array![0.0_f64, 1.0, -1.0];
/// let result = sinh_simd(&x.view());
///
/// // sinh(0) = 0, sinh(1) ≈ 1.175, sinh(-1) ≈ -1.175
/// assert!((result[0] - 0.0).abs() < 1e-10);
/// assert!((result[1] - 1.1752011936).abs() < 1e-9);
/// assert!((result[2] + 1.1752011936).abs() < 1e-9);
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: sinh(0) = 0
/// - **Large positive**: May overflow to infinity
/// - **Large negative**: May overflow to negative infinity
/// - **NaN**: Returns NaN (preserves NaN)
///
/// # Applications
///
/// - **Neural Networks**: Tanh activation (related via tanh = sinh/cosh)
/// - **Numerical Integration**: Tanh-sinh quadrature
/// - **Physics**: Catenary curves, special relativity
/// - **Engineering**: Transmission line theory, heat transfer
/// - **Mathematics**: Hyperbolic geometry, complex analysis
pub fn sinh_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_sinh(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.sinh())
}

/// Compute the hyperbolic cosine of each element (SIMD-accelerated).
///
/// Computes cosh(x) for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with hyperbolic cosine values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// cosh(x) = (e^x + e^(-x)) / 2
/// Range: [1, ∞) (always >= 1)
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::cosh_simd;
///
/// let x = array![0.0_f64, 1.0, -1.0];
/// let result = cosh_simd(&x.view());
///
/// // cosh(0) = 1, cosh(1) ≈ 1.543, cosh(-1) ≈ 1.543
/// assert!((result[0] - 1.0).abs() < 1e-10);
/// assert!((result[1] - 1.5430806348).abs() < 1e-9);
/// assert!((result[2] - 1.5430806348).abs() < 1e-9);
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: cosh(0) = 1
/// - **Symmetric**: cosh(-x) = cosh(x)
/// - **Large values**: May overflow to infinity
/// - **NaN**: Returns NaN (preserves NaN)
///
/// # Applications
///
/// - **Neural Networks**: Activation functions, normalization
/// - **Physics**: Wave propagation, special relativity
/// - **Engineering**: Cable suspension, arch design
/// - **Mathematics**: Hyperbolic identities (cosh² - sinh² = 1)
/// - **Numerical Methods**: Stability analysis
pub fn cosh_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_cosh(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.cosh())
}

/// Compute the hyperbolic tangent of each element (SIMD-accelerated).
///
/// Computes tanh(x) for each element in the array.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, with hyperbolic tangent values.
///
/// # Performance
///
/// - **Auto-vectorization**: Compiler optimizations provide excellent performance
/// - **Speedup**: 2-4x on large arrays via auto-vectorization
///
/// # Mathematical Definition
///
/// ```text
/// tanh(x) = sinh(x) / cosh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
/// Range: (-1, 1)
/// ```
///
/// # Examples
///
/// ```ignore
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::tanh_simd;
///
/// let x = array![0.0_f64, 1.0, -1.0, 10.0];
/// let result = tanh_simd(&x.view());
///
/// // tanh(0) = 0, tanh(1) ≈ 0.762, tanh(-1) ≈ -0.762, tanh(∞) → 1
/// assert!((result[0] - 0.0_f64).abs() < 1e-10);
/// assert!((result[1] - 0.7615941559_f64).abs() < 1e-9);
/// assert!((result[2] + 0.7615941559_f64).abs() < 1e-9);
/// assert!((result[3] - 1.0_f64).abs() < 1e-9); // tanh(10) ≈ 1
/// ```
///
/// # Edge Cases
///
/// - **Empty array**: Returns empty array
/// - **Zero**: tanh(0) = 0
/// - **Asymptotic**: tanh(x) → ±1 as x → ±∞
/// - **Anti-symmetric**: tanh(-x) = -tanh(x)
/// - **NaN**: Returns NaN (preserves NaN)
///
/// # Applications
///
/// - **Neural Networks**: Tanh activation function (classic activation)
/// - **Machine Learning**: Gradient clipping, normalization layers
/// - **Reinforcement Learning**: Policy networks, value functions
/// - **Signal Processing**: Soft limiting, saturation
/// - **Optimization**: Smooth approximation to sign function
/// - **Physics**: Relativistic velocity addition
///
/// # Note on Neural Networks
///
/// Tanh was historically the most popular activation function before ReLU.
/// It's still widely used in RNNs, LSTMs, and GRUs for gating mechanisms.
/// Gradient: d/dx tanh(x) = 1 - tanh²(x) = sech²(x)
pub fn tanh_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_tanh(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.tanh())
}

/// Compute the floor (round down) of each element (SIMD-accelerated).
///
/// Computes the largest integer less than or equal to x for each element.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is rounded down
/// to the nearest integer.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Mathematical Properties
///
/// - For any x: floor(x) <= x
/// - floor(x) is the largest integer <= x
/// - floor(-x) = -ceil(x)
/// - floor(x) = x if x is already an integer
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::floor_simd;
///
/// let x = array![1.2, 2.7, -1.3, -2.9, 3.0];
/// let result = floor_simd(&x.view());
/// // Result: [1.0, 2.0, -2.0, -3.0, 3.0]
/// ```
///
/// # Applications
///
/// - **Binning**: Discretizing continuous values into bins
/// - **Indexing**: Converting continuous coordinates to discrete indices
/// - **Quantization**: Reducing precision for data compression
/// - **Digital Signal Processing**: Sample rate conversion, downsampling
/// - **Computer Graphics**: Pixel coordinate calculations
/// - **Financial**: Rounding down monetary amounts
///
/// # See Also
///
/// - [`ceil_simd`]: Round up to smallest integer >= x
/// - [`round_simd`]: Round to nearest integer
pub fn floor_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_floor(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.floor())
}

/// Compute the ceiling (round up) of each element (SIMD-accelerated).
///
/// Computes the smallest integer greater than or equal to x for each element.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is rounded up
/// to the nearest integer.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Mathematical Properties
///
/// - For any x: ceil(x) >= x
/// - ceil(x) is the smallest integer >= x
/// - ceil(-x) = -floor(x)
/// - ceil(x) = x if x is already an integer
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::ceil_simd;
///
/// let x = array![1.2, 2.7, -1.3, -2.9, 3.0];
/// let result = ceil_simd(&x.view());
/// // Result: [2.0, 3.0, -1.0, -2.0, 3.0]
/// ```
///
/// # Applications
///
/// - **Memory Allocation**: Rounding up buffer sizes
/// - **Pagination**: Calculating number of pages needed
/// - **Resource Planning**: Determining minimum resources required
/// - **Digital Signal Processing**: Sample rate conversion, upsampling
/// - **Computer Graphics**: Bounding box calculations
/// - **Financial**: Rounding up monetary amounts (conservative estimation)
///
/// # See Also
///
/// - [`floor_simd`]: Round down to largest integer <= x
/// - [`round_simd`]: Round to nearest integer
pub fn ceil_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_ceil(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.ceil())
}

/// Compute the rounding to nearest integer of each element (SIMD-accelerated).
///
/// Rounds each element to the nearest integer, with ties (x.5) rounding away
/// from zero (standard rounding).
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is rounded to
/// the nearest integer.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Rounding Behavior
///
/// This function uses "round half away from zero" (standard rounding):
/// - 0.5 rounds to 1.0
/// - 1.5 rounds to 2.0
/// - 2.5 rounds to 3.0
/// - -0.5 rounds to -1.0
/// - -1.5 rounds to -2.0
///
/// # Mathematical Properties
///
/// - |round(x) - x| <= 0.5 for all x
/// - round(x) = x if x is already an integer
/// - round(-x) = -round(x)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::round_simd;
///
/// let x = array![1.2, 2.7, -1.3, -2.9, 3.0, 2.5];
/// let result = round_simd(&x.view());
/// // Result: [1.0, 3.0, -1.0, -3.0, 3.0, 3.0]
/// //          note: 2.5 rounds to 3.0 (away from zero)
/// ```
///
/// # Applications
///
/// - **Data Visualization**: Rounding values for display
/// - **Statistics**: Rounding statistical summaries
/// - **Financial**: General-purpose monetary rounding
/// - **Machine Learning**: Quantization for model compression
/// - **Image Processing**: Pixel value normalization
/// - **Scientific Computing**: Reducing numerical noise
///
/// # See Also
///
/// - [`floor_simd`]: Round down to largest integer <= x
/// - [`ceil_simd`]: Round up to smallest integer >= x
pub fn round_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_round(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.round())
}

/// Compute the fractional part of each element (SIMD-accelerated).
///
/// Returns the signed fractional component of each value (x - trunc(x)).
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the signed fractional part
/// in the range (-1, 1).
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD unavailable
/// - **Speedup**: 2-4x for large arrays on AVX2/NEON systems
///
/// # Mathematical Definition
///
/// ```text
/// fract(x) = x - trunc(x)
/// ```
///
/// For positive x: same as x - floor(x)
/// For negative x: preserves sign (e.g., fract(-1.5) = -0.5)
///
/// # Properties
///
/// - -1 < fract(x) < 1 for all finite x
/// - fract(x + n) = fract(x) for any integer n (periodic with period 1)
/// - fract(x) = 0 if and only if x is an integer
/// - fract(x) + trunc(x) = x
/// - fract(-x) = -fract(x) (odd function)
///
/// # Applications
///
/// - **Computer Graphics**: Texture coordinate wrapping, repeating patterns
/// - **Animation**: Cyclic motion, looping animations
/// - **Signal Processing**: Modulo 1 operations, phase wrapping
/// - **Numerical Methods**: Fractional part extraction, decimal decomposition
/// - **Game Development**: Tile-based rendering, repeating textures
/// - **Scientific Computing**: Periodic boundary conditions, modulo arithmetic
/// - **Audio Processing**: Phase accumulation, waveform generation
/// - **Time Calculations**: Extracting fractional seconds, subsecond precision
/// - **Cryptography**: Linear congruential generators, pseudo-random sequences
/// - **Financial**: Fractional share calculations, interest accrual
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::fract_simd;
///
/// let x = array![1.5_f64, 2.7, -1.3, 0.0, 3.0];
/// let result = fract_simd(&x.view());
/// assert!((result[0] - 0.5_f64).abs() < 1e-10);    // 1.5 - trunc(1.5) = 0.5
/// assert!((result[1] - 0.7_f64).abs() < 1e-10);    // 2.7 - trunc(2.7) = 0.7
/// assert!((result[2] - (-0.3_f64)).abs() < 1e-10); // -1.3 - trunc(-1.3) = -0.3
/// assert_eq!(result[3], 0.0_f64);                   // 0.0 - trunc(0.0) = 0.0
/// assert_eq!(result[4], 0.0_f64);                   // 3.0 - trunc(3.0) = 0.0
///
/// // Property: fract(x) + trunc(x) = x
/// let values = array![5.25_f64, -2.75, 0.5, 10.0];
/// let fract_parts = fract_simd(&values.view());
/// let trunc_parts = values.mapv(|v: f64| v.trunc());
/// let reconstructed = &fract_parts + &trunc_parts;
/// for i in 0..values.len() {
///     assert!((reconstructed[i] - values[i]).abs() < 1e-10);
/// }
/// ```
///
/// # See Also
///
/// - [`floor_simd`]: Round down to integer
/// - [`round_simd`]: Round to nearest integer
pub fn fract_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_fract(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.fract())
}

/// Compute the reciprocal (multiplicative inverse) of each element (SIMD-accelerated).
///
/// Returns 1/x for each element in the array. For zero values, the result is infinity.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the reciprocal (1/x).
/// Zero elements yield infinity values.
///
/// # Mathematical Definition
///
/// ```text
/// recip(x) = 1/x for x ≠ 0
/// recip(0) = ∞
/// ```
///
/// # Properties
///
/// - recip(recip(x)) = x (involutive property)
/// - recip(x * y) = recip(x) * recip(y) (multiplicative property)
/// - recip(x/y) = y/x (division inversion)
/// - recip(1) = 1 (identity element)
/// - recip(-x) = -recip(x) (odd function)
/// - recip(x^n) = (recip(x))^n (power property)
///
/// # Applications
///
/// ## Numerical Analysis
/// - Matrix inversions: Computing A^(-1) in linear systems
/// - Newton's method: Division-free iterations via reciprocal approximations
/// - Condition number estimation: ||A|| * ||A^(-1)||
/// - Iterative refinement: Improving solution accuracy in linear solvers
///
/// ## Computer Graphics
/// - Projection transformations: Converting world coordinates to screen space
/// - Lighting calculations: Inverse square law for light attenuation
/// - Texture mapping: Computing texture coordinate gradients
/// - Depth buffer operations: Converting depth to 1/z for perspective
///
/// ## Physics Simulations
/// - Inverse square laws: Gravitational force (F = G * m1 * m2 / r^2)
/// - Wave propagation: Intensity attenuation (I ∝ 1/r^2)
/// - Electromagnetic fields: Coulomb's law (F = k * q1 * q2 / r^2)
/// - Fluid dynamics: Pressure gradient computations
///
/// ## Signal Processing
/// - Filter design: Converting frequency response to impulse response
/// - Normalization: Scaling signals by reciprocal of maximum amplitude
/// - Frequency domain analysis: Converting period to frequency (f = 1/T)
/// - Deconvolution: Inverse filtering for signal restoration
///
/// ## Machine Learning
/// - Normalization layers: Scaling features by reciprocal of variance
/// - Activation functions: Sigmoid (1 / (1 + exp(-x)))
/// - Loss functions: Reciprocal of predictions in certain regression tasks
/// - Learning rate schedules: Inverse decay (lr = initial_lr / (1 + decay * t))
///
/// ## Financial Modeling
/// - Discount rates: Present value calculations (PV = FV * recip(1 + r)^n)
/// - Interest rate conversions: Converting rates to factors
/// - Portfolio optimization: Inverse volatility weighting
/// - Option pricing: Black-Scholes model components
///
/// # Edge Cases
///
/// - recip(0.0) = f64::INFINITY
/// - recip(-0.0) = f64::NEG_INFINITY
/// - recip(INFINITY) = 0.0
/// - recip(NEG_INFINITY) = -0.0
/// - recip(NaN) = NaN
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::recip_simd;
///
/// let x = array![2.0_f64, 4.0, 0.5, 1.0];
/// let result = recip_simd(&x.view());
///
/// assert!((result[0] - 0.5_f64).abs() < 1e-10);    // recip(2.0) = 0.5
/// assert!((result[1] - 0.25_f64).abs() < 1e-10);   // recip(4.0) = 0.25
/// assert!((result[2] - 2.0_f64).abs() < 1e-10);    // recip(0.5) = 2.0
/// assert!((result[3] - 1.0_f64).abs() < 1e-10);    // recip(1.0) = 1.0
/// ```
///
/// # Performance
///
/// This function uses SIMD acceleration for arrays larger than 1000 elements,
/// providing significant speedups through vectorized reciprocal approximations.
/// For smaller arrays, a scalar fallback is used to avoid SIMD overhead.
///
/// # Panics
///
/// This function does not panic. Division by zero results in infinity values
/// as per IEEE 754 floating-point semantics.
pub fn recip_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_recip(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.recip())
}

/// Compute integer power (base^n) for each element (SIMD-accelerated).
///
/// Returns base^n for each element in the array, where n is an integer exponent.
/// More efficient than powf for integer exponents.
///
/// # Arguments
///
/// * `base` - Input 1D array
/// * `n` - Integer exponent
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is base^n.
///
/// # Mathematical Definition
///
/// ```text
/// powi(x, n) = x^n for integer n
/// ```
///
/// # Properties
///
/// - powi(x, 0) = 1 for all x ≠ 0
/// - powi(x, 1) = x
/// - powi(x, -n) = 1 / powi(x, n)
/// - powi(x, n+m) = powi(x, n) * powi(x, m) (exponent addition)
/// - powi(x*y, n) = powi(x, n) * powi(y, n) (distributive over multiplication)
/// - powi(x, n*m) = powi(powi(x, n), m) (exponent multiplication)
///
/// # Applications
///
/// ## Statistics
/// - Chi-square distributions: Σ(x_i - μ)^2
/// - Polynomial distributions: Computing moments
/// - Variance calculations: E\[X^2\] - (E\[X\])^2
/// - Higher-order moments: Skewness (3rd moment), kurtosis (4th moment)
///
/// ## Linear Algebra
/// - Matrix powers: A^n for matrix operations
/// - Eigenvalue problems: Computing characteristic polynomials
/// - Norm calculations: ||x||_p = (Σ|x_i|^p)^(1/p)
/// - Gram matrices: X^T X operations
///
/// ## Signal Processing
/// - Polynomial filters: Computing filter responses
/// - Power spectral density: |X(f)|^2
/// - Autocorrelation: r(k) = Σ x(n) * x(n-k)
/// - Window functions: Raised cosine windows with integer powers
///
/// ## Machine Learning
/// - Polynomial features: x^2, x^3 for feature engineering
/// - Loss functions: L2 loss with (y - ŷ)^2
/// - Regularization: Ridge regression with ||w||^2
/// - Gradient descent: Computing squared gradients
///
/// ## Numerical Analysis
/// - Taylor series: Computing polynomial approximations
/// - Newton's method: f(x) and f'(x) evaluations
/// - Polynomial interpolation: Lagrange basis functions
/// - Finite differences: Computing higher-order derivatives
///
/// ## Physics Simulations
/// - Inverse square laws: 1/r^2 for gravity and electromagnetism
/// - Kinetic energy: (1/2)mv^2
/// - Potential energy: Polynomial potentials
/// - Power laws: Scaling relationships
///
/// # Edge Cases
///
/// - powi(0, n) = 0 for n > 0
/// - powi(0, 0) = 1 (mathematical convention)
/// - powi(x, 0) = 1 for all x
/// - powi(∞, n) = ∞ for n > 0, 0 for n < 0
/// - powi(NaN, n) = NaN
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::powi_simd;
///
/// let x = array![2.0_f64, 3.0, 4.0, 5.0];
///
/// // Square
/// let squared = powi_simd(&x.view(), 2);
/// assert!((squared[0] - 4.0_f64).abs() < 1e-10);    // 2^2 = 4
/// assert!((squared[1] - 9.0_f64).abs() < 1e-10);    // 3^2 = 9
/// assert!((squared[2] - 16.0_f64).abs() < 1e-10);   // 4^2 = 16
/// assert!((squared[3] - 25.0_f64).abs() < 1e-10);   // 5^2 = 25
///
/// // Cube
/// let cubed = powi_simd(&x.view(), 3);
/// assert!((cubed[0] - 8.0_f64).abs() < 1e-10);      // 2^3 = 8
/// assert!((cubed[1] - 27.0_f64).abs() < 1e-10);     // 3^3 = 27
///
/// // Negative exponent (reciprocal)
/// let inv_squared = powi_simd(&x.view(), -2);
/// assert!((inv_squared[0] - 0.25_f64).abs() < 1e-10);   // 2^(-2) = 0.25
/// assert!((inv_squared[1] - (1.0_f64/9.0)).abs() < 1e-10); // 3^(-2) = 1/9
/// ```
///
/// # Performance
///
/// This function uses SIMD acceleration for arrays larger than 1000 elements.
/// Integer power is computed using efficient exponentiation by squaring,
/// providing better performance than powf for integer exponents.
///
/// For small arrays, a scalar fallback is used to avoid SIMD overhead.
///
/// # Panics
///
/// This function does not panic. Edge cases like 0^0 return 1 as per
/// mathematical convention and IEEE 754 semantics.
pub fn powi_simd<F>(base: &ArrayView1<F>, n: i32) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if base.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(base.len()) {
        return F::simd_powi(base, n);
    }

    // Scalar fallback for small arrays
    base.mapv(|val| val.powi(n))
}

/// Compute the gamma function Γ(x) for each element (SIMD-accelerated).
///
/// Evaluates the gamma function using the Lanczos approximation, providing high accuracy
/// across the entire real domain. The gamma function is a generalization of the factorial
/// function to non-integer values.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is Γ(x).
///
/// # Mathematical Definition
///
/// The gamma function is defined by Euler's integral:
/// ```text
/// Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt,    for Re(z) > 0
/// ```
///
/// For other values, it is defined by analytic continuation using the functional equation:
/// ```text
/// Γ(z+1) = z·Γ(z)
/// ```
///
/// # Key Properties
///
/// **Fundamental Properties**:
/// - Γ(1) = 1
/// - Γ(n+1) = n! for positive integers n
/// - Γ(1/2) = √π
/// - Γ(z+1) = z·Γ(z) (functional equation)
/// - Γ(z)·Γ(1-z) = π/sin(πz) (reflection formula)
///
/// **Special Values**:
/// - Γ(0) = ∞ (pole)
/// - Γ(-n) = ∞ for negative integers (poles)
/// - Γ(n+1/2) = (2n-1)!!·√π/2ⁿ for non-negative integers n
///
/// # Applications
///
/// ## 1. Statistical Distributions
/// - **Gamma Distribution**: PDF = (x^(α-1) e^(-x/β)) / (β^α Γ(α))
/// - **Beta Distribution**: B(α,β) = Γ(α)Γ(β)/Γ(α+β)
/// - **Chi-Square Distribution**: Uses Γ(k/2)
/// - **Student's t-Distribution**: Normalization involves Γ((ν+1)/2) and Γ(ν/2)
///
/// ## 2. Special Functions
/// - **Incomplete Gamma**: γ(s,x), Γ(s,x)
/// - **Beta Function**: B(α,β) = Γ(α)Γ(β)/Γ(α+β)
/// - **Bessel Functions**: Many representations involve gamma
/// - **Hypergeometric Functions**: Coefficients use gamma ratios
///
/// ## 3. Combinatorics & Number Theory
/// - **Binomial Coefficients**: C(n,k) = Γ(n+1)/(Γ(k+1)Γ(n-k+1))
/// - **Stirling Numbers**: Involve gamma function ratios
/// - **Riemann Zeta Function**: Functional equation uses Γ(s/2)
///
/// ## 4. Bayesian Inference
/// - **Conjugate Priors**: Gamma and Beta distributions
/// - **Dirichlet Distribution**: Normalization uses gamma products
/// - **Variational Inference**: Optimization involves gamma and digamma
///
/// ## 5. Physics & Engineering
/// - **Quantum Mechanics**: Angular momentum eigenstates
/// - **String Theory**: Veneziano amplitude
/// - **Statistical Mechanics**: Partition functions
/// - **Signal Processing**: Window functions
///
/// ## 6. Numerical Analysis
/// - **Integration**: Change of variables
/// - **Asymptotic Analysis**: Stirling's approximation
/// - **Interpolation**: Gamma function interpolates factorial
///
/// # Implementation
///
/// Uses the **Lanczos approximation** with g=7 and 9 coefficients, providing
/// ~15 decimal digits of accuracy for x ∈ [0.5, 100]. For x < 0.5, uses the
/// reflection formula to leverage the accurate computation of Γ(1-x).
///
/// **Algorithm**:
/// ```text
/// Γ(z+1) = √(2π) * (z + g + 1/2)^(z + 1/2) * e^(-(z + g + 1/2)) * A_g(z)
///
/// where A_g(z) = c₀ + c₁/(z+1) + c₂/(z+2) + ... + c₈/(z+9)
/// ```
///
/// **Coefficients** (from Boost C++ Library):
/// - Optimized using Remez exchange algorithm
/// - Minimax criterion for optimal approximation
/// - Provides uniform accuracy across domain
///
/// # Edge Cases
///
/// - `gamma(NaN)` → NaN
/// - `gamma(0)` → ∞
/// - `gamma(-n)` → ∞ for negative integers (poles)
/// - `gamma(x)` → ∞ as x → ∞
/// - `gamma(x)` → 0 as x → -∞ (alternating sign)
///
/// # Performance Characteristics
///
/// - **Small arrays** (< 1000 elements): Scalar implementation
/// - **Large arrays** (≥ 1000 elements): SIMD-accelerated via compiler auto-vectorization
/// - **Time complexity**: O(n) where n is array length
/// - **Space complexity**: O(n) for output array
/// - **Accuracy**: ~15 decimal digits for typical inputs
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::gamma_simd;
///
/// // Factorials: Γ(n+1) = n!
/// let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let gamma_x = gamma_simd(&x.view());
/// // gamma_x ≈ [1, 1, 2, 6, 24] (0!, 1!, 2!, 3!, 4!)
///
/// // Half-integer values
/// let x_half = array![0.5];
/// let gamma_half = gamma_simd(&x_half.view());
/// // gamma_half[0] ≈ 1.772453850905516 (√π)
///
/// // Statistical applications
/// use scirs2_core::numeric::Float;
/// let alpha = array![2.0, 3.0, 5.0];
/// let normalization = gamma_simd(&alpha.view());
/// // Use in gamma distribution PDF: x^(α-1) e^(-x/β) / (β^α Γ(α))
/// ```
///
/// # Mathematical References
///
/// - Abramowitz & Stegun, "Handbook of Mathematical Functions", §6.1
/// - Lanczos, "A Precision Approximation of the Gamma Function" (1964)
/// - Press et al., "Numerical Recipes", §6.1
/// - Boost C++ Libraries: Math Toolkit Documentation
///
/// # See Also
///
/// - [`powi_simd`] - Integer power (related operation)
/// - [`exp_simd`], [`ln_simd`] - Component operations
/// - Future: `digamma_simd` - Logarithmic derivative of gamma
pub fn gamma_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_gamma(x);
    }

    // Scalar fallback for small arrays - use the same Lanczos implementation
    F::simd_gamma(x)
}

/// Compute the arctangent of each element (SIMD-accelerated).
///
/// Computes atan(x) for each element, returning values in the range (-π/2, π/2).
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the arctangent
/// in radians.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Mathematical Properties
///
/// - Range: (-π/2, π/2) for all finite x
/// - atan(0) = 0
/// - atan(-x) = -atan(x) (odd function)
/// - atan(∞) = π/2, atan(-∞) = -π/2
/// - atan(tan(x)) = x for x ∈ (-π/2, π/2)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::atan_simd;
///
/// let x = array![0.0_f64, 1.0, -1.0, f64::INFINITY];
/// let result = atan_simd(&x.view());
/// // Result: [0.0, π/4, -π/4, π/2]
/// ```
///
/// # Applications
///
/// - **Geometry**: Calculating angles from slopes
/// - **Robotics**: Inverse kinematics for joint angles
/// - **Computer Vision**: Feature detection, edge orientation
/// - **Signal Processing**: Phase unwrapping, frequency analysis
/// - **Physics**: Trajectory analysis, projectile motion
///
/// # See Also
///
/// - [`asin_simd`]: Arcsine function
/// - [`acos_simd`]: Arccosine function
/// - [`atan2_simd`]: Two-argument arctangent for full angle range
pub fn atan_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_atan(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.atan())
}

/// Compute the arcsine of each element (SIMD-accelerated).
///
/// Computes asin(x) for each element, returning values in the range [-π/2, π/2].
///
/// # Arguments
///
/// * `x` - Input 1D array with values in [-1, 1]
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the arcsine
/// in radians. Returns NaN for |x| > 1.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Mathematical Properties
///
/// - Domain: [-1, 1]
/// - Range: [-π/2, π/2]
/// - asin(0) = 0
/// - asin(-x) = -asin(x) (odd function)
/// - asin(1) = π/2, asin(-1) = -π/2
/// - asin(sin(x)) = x for x ∈ [-π/2, π/2]
/// - Returns NaN for |x| > 1
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::asin_simd;
///
/// let x = array![0.0_f64, 0.5, 1.0, -1.0];
/// let result = asin_simd(&x.view());
/// // Result: [0.0, π/6, π/2, -π/2]
/// ```
///
/// # Applications
///
/// - **Navigation**: Great circle distance calculations
/// - **Astronomy**: Celestial coordinate transformations
/// - **Physics**: Wave mechanics, pendulum analysis
/// - **Computer Graphics**: Spherical coordinates, lighting
/// - **Robotics**: Inverse kinematics with constraints
///
/// # See Also
///
/// - [`atan_simd`]: Arctangent function
/// - [`acos_simd`]: Arccosine function
/// - [`sin_simd`]: Forward sine function
pub fn asin_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_asin(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.asin())
}

/// Compute the arccosine of each element (SIMD-accelerated).
///
/// Computes acos(x) for each element, returning values in the range [0, π].
///
/// # Arguments
///
/// * `x` - Input 1D array with values in [-1, 1]
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the arccosine
/// in radians. Returns NaN for |x| > 1.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Mathematical Properties
///
/// - Domain: [-1, 1]
/// - Range: [0, π]
/// - acos(1) = 0, acos(-1) = π
/// - acos(0) = π/2
/// - acos(-x) = π - acos(x)
/// - acos(cos(x)) = x for x ∈ [0, π]
/// - acos(x) + asin(x) = π/2
/// - Returns NaN for |x| > 1
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::acos_simd;
///
/// let x = array![1.0_f64, 0.5, 0.0, -1.0];
/// let result = acos_simd(&x.view());
/// // Result: [0.0, π/3, π/2, π]
/// ```
///
/// # Applications
///
/// - **3D Graphics**: Dot product angle calculations
/// - **Machine Learning**: Cosine similarity to angle conversion
/// - **Physics**: Angular momentum, rotation analysis
/// - **Astronomy**: Coordinate system transformations
/// - **Navigation**: Bearing and heading calculations
///
/// # See Also
///
/// - [`atan_simd`]: Arctangent function
/// - [`asin_simd`]: Arcsine function
/// - [`cos_simd`]: Forward cosine function
pub fn acos_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_acos(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.acos())
}

/// Compute the two-argument arctangent element-wise (SIMD-accelerated).
///
/// Computes atan2(y, x) for each pair of elements, returning values in the range
/// (-π, π]. This function correctly handles the signs of both arguments to
/// determine the quadrant.
///
/// # Arguments
///
/// * `y` - Y-coordinates (sine component)
/// * `x` - X-coordinates (cosine component)
///
/// # Returns
///
/// `Array1<F>` with the same length as inputs, where each element is the angle
/// in radians from the positive x-axis to the point (x, y).
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Mathematical Properties
///
/// - Range: (-π, π]
/// - atan2(0, 0) = 0 (by convention)
/// - atan2(y, x) = atan(y/x) when x > 0
/// - atan2(y, 0) = π/2 * sign(y) when x = 0
/// - atan2(-y, x) = -atan2(y, x)
/// - atan2(y, -x) = π - atan2(y, x) when y >= 0
/// - atan2(y, -x) = -π + atan2(y, x) when y < 0
///
/// # Quadrants
///
/// - Quadrant I (x > 0, y > 0): (0, π/2)
/// - Quadrant II (x < 0, y > 0): (π/2, π)
/// - Quadrant III (x < 0, y < 0): (-π, -π/2)
/// - Quadrant IV (x > 0, y < 0): (-π/2, 0)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::atan2_simd;
///
/// let y = array![1.0, 1.0, -1.0, -1.0];
/// let x = array![1.0_f64, -1.0, -1.0, 1.0];
/// let angles = atan2_simd(&y.view(), &x.view());
/// // Result: [π/4, 3π/4, -3π/4, -π/4]
/// ```
///
/// # Applications
///
/// - **Robotics**: Joint angle calculations, path planning
/// - **Computer Vision**: Feature orientation, optical flow
/// - **Navigation**: Heading and bearing calculations
/// - **Spatial Computing**: Coordinate transformations
/// - **Physics**: Vector angle calculations, force analysis
/// - **Computer Graphics**: Rotation angles, sprite orientation
/// - **Signal Processing**: Phase calculations, complex number arguments
///
/// # See Also
///
/// - [`atan_simd`]: Single-argument arctangent
/// - [`asin_simd`]: Arcsine function
/// - [`acos_simd`]: Arccosine function
pub fn atan2_simd<F>(y: &ArrayView1<F>, x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    assert_eq!(y.len(), x.len(), "y and x arrays must have the same length");

    if y.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(y.len()) {
        return F::simd_atan2(y, x);
    }

    // Scalar fallback for small arrays
    y.iter()
        .zip(x.iter())
        .map(|(&y_val, &x_val)| y_val.atan2(x_val))
        .collect::<Vec<_>>()
        .into()
}

/// Compute the base-10 logarithm of each element (SIMD-accelerated).
///
/// Computes log₁₀(x) for each element, where log₁₀(x) = ln(x) / ln(10).
///
/// # Arguments
///
/// * `x` - Input 1D array with positive values
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the base-10
/// logarithm. Returns NaN for x ≤ 0, and -∞ for x = 0.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Mathematical Properties
///
/// - Domain: (0, ∞)
/// - Range: (-∞, ∞)
/// - log₁₀(1) = 0
/// - log₁₀(10) = 1
/// - log₁₀(10ⁿ) = n
/// - log₁₀(x * y) = log₁₀(x) + log₁₀(y)
/// - log₁₀(x / y) = log₁₀(x) - log₁₀(y)
/// - log₁₀(xⁿ) = n * log₁₀(x)
/// - Returns NaN for x ≤ 0
/// - Returns -∞ for x = 0
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::log10_simd;
///
/// let x = array![1.0_f64, 10.0, 100.0, 1000.0];
/// let result = log10_simd(&x.view());
/// // Result: [0.0, 1.0, 2.0, 3.0]
/// ```
///
/// # Applications
///
/// - **Signal Processing**: Decibel scale (dB = 10 * log₁₀(P/P₀))
/// - **Chemistry**: pH scale (pH = -log₁₀[H⁺])
/// - **Astronomy**: Magnitude scale (apparent magnitude)
/// - **Scientific Computing**: Decades representation, log-log plots
/// - **Machine Learning**: Feature scaling, log-loss functions
/// - **Information Theory**: Hartley's entropy (base-10)
/// - **Physics**: Richter scale (earthquake magnitude)
/// - **Audio Processing**: Sound intensity level
///
/// # See Also
///
/// - [`ln_simd`]: Natural logarithm (base e)
/// - [`log2_simd`]: Base-2 logarithm
/// - [`exp_simd`]: Exponential function (inverse of ln)
pub fn log10_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_log10(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.log10())
}

/// Compute the base-2 logarithm of each element (SIMD-accelerated).
///
/// Computes log₂(x) for each element, where log₂(x) = ln(x) / ln(2).
///
/// # Arguments
///
/// * `x` - Input 1D array with positive values
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the base-2
/// logarithm. Returns NaN for x ≤ 0, and -∞ for x = 0.
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Mathematical Properties
///
/// - Domain: (0, ∞)
/// - Range: (-∞, ∞)
/// - log₂(1) = 0
/// - log₂(2) = 1
/// - log₂(2ⁿ) = n
/// - log₂(x * y) = log₂(x) + log₂(y)
/// - log₂(x / y) = log₂(x) - log₂(y)
/// - log₂(xⁿ) = n * log₂(x)
/// - Returns NaN for x ≤ 0
/// - Returns -∞ for x = 0
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::log2_simd;
///
/// let x = array![1.0_f64, 2.0, 4.0, 8.0, 16.0];
/// let result = log2_simd(&x.view());
/// // Result: [0.0, 1.0, 2.0, 3.0, 4.0]
/// ```
///
/// # Applications
///
/// - **Information Theory**: Shannon entropy (bits), channel capacity
/// - **Computer Science**: Binary tree depth, algorithm complexity analysis
/// - **Machine Learning**: Decision trees, information gain
/// - **Signal Processing**: Octave scales, frequency resolution
/// - **Data Compression**: Optimal coding, Huffman coding
/// - **Cryptography**: Key size representation, security levels
/// - **Scientific Computing**: Binary logarithmic plots
/// - **Digital Systems**: Bit depth calculations
///
/// # See Also
///
/// - [`ln_simd`]: Natural logarithm (base e)
/// - [`log10_simd`]: Base-10 logarithm
/// - [`exp_simd`]: Exponential function (inverse of ln)
pub fn log2_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_log2(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.log2())
}

/// Clamp each element to a specified range [min, max] (SIMD-accelerated).
///
/// Constrains each element x to satisfy min ≤ x ≤ max. Values below min are set
/// to min, values above max are set to max, and values within range are unchanged.
///
/// # Arguments
///
/// * `x` - Input 1D array
/// * `min` - Minimum value (lower bound)
/// * `max` - Maximum value (upper bound)
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is clamped to [min, max].
///
/// # Panics
///
/// Panics if min > max (invalid range).
///
/// # Performance
///
/// - **SIMD**: Automatically used for large arrays (1000+ elements)
/// - **Scalar**: Used for small arrays or when SIMD is unavailable
///
/// # Mathematical Properties
///
/// - clamp(x, min, max) = max(min, min(x, max))
/// - clamp(x, min, max) ∈ [min, max] for all x
/// - clamp(min, min, max) = min
/// - clamp(max, min, max) = max
/// - clamp is idempotent: clamp(clamp(x, a, b), a, b) = clamp(x, a, b)
/// - clamp preserves monotonicity: if x₁ ≤ x₂, then clamp(x₁) ≤ clamp(x₂)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::clamp_simd;
///
/// let x = array![-1.0_f64, 0.5, 1.5, 2.5, 3.5];
/// let result = clamp_simd(&x.view(), 0.0, 2.0);
/// // Result: [0.0, 0.5, 1.5, 2.0, 2.0]
/// ```
///
/// # Applications
///
/// - **Image Processing**: Pixel value normalization (0-255, 0.0-1.0)
/// - **Neural Networks**: Gradient clipping, activation bounding
/// - **Computer Vision**: Color space conversions, contrast limiting
/// - **Signal Processing**: Dynamic range compression, amplitude limiting
/// - **Data Normalization**: Feature scaling, outlier handling
/// - **Numerical Stability**: Preventing overflow/underflow in computations
/// - **Game Development**: Velocity limiting, position constraints
/// - **Robotics**: Joint angle limits, actuator bounds
/// - **Audio Processing**: Volume limiting, signal clipping prevention
/// - **Machine Learning**: Learning rate bounds, weight constraints
///
/// # See Also
///
/// - [`abs_simd`]: Absolute value (clamping negative values to positive)
/// - [`floor_simd`]: Lower bound only (ceiling at integers)
/// - [`ceil_simd`]: Upper bound only (floor at integers)
pub fn clamp_simd<F>(x: &ArrayView1<F>, min: F, max: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps + std::fmt::Debug,
{
    assert!(
        min <= max,
        "clamp_simd: min ({:?}) must be <= max ({:?})",
        min,
        max
    );

    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_clamp(x, min, max);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.clamp(min, max))
}

/// Compute 2^x for each element (SIMD-accelerated).
///
/// Computes the base-2 exponential function 2^x for each element.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is 2 raised to the
/// corresponding input power.
///
/// # Performance
///
/// Uses the identity 2^x = exp(x * ln(2)) with SIMD-accelerated exp and scalar multiply.
///
/// # Mathematical Properties
///
/// - 2^0 = 1
/// - 2^1 = 2
/// - 2^(-1) = 0.5
/// - 2^n for integer n is exact for small n
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::exp2_simd;
///
/// let x = array![0.0_f64, 1.0, 2.0, 3.0, -1.0];
/// let result = exp2_simd(&x.view());
/// // Result: [1.0, 2.0, 4.0, 8.0, 0.5]
/// ```
///
/// # Applications
///
/// - **Audio Processing**: Octave/semitone calculations
/// - **Computer Graphics**: Level of detail (LOD) calculations
/// - **Floating-Point**: Exponent manipulation
pub fn exp2_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_exp2(x);
    }

    // Scalar fallback for small arrays
    F::simd_exp2(x)
}

/// Compute the cube root of each element (SIMD-accelerated).
///
/// Computes cbrt(x) = x^(1/3) for each element, correctly handling negative values.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the cube root.
///
/// # Mathematical Properties
///
/// - cbrt(x^3) = x (exact inverse of cubing)
/// - cbrt(-x) = -cbrt(x) (handles negative numbers)
/// - cbrt(0) = 0
/// - cbrt(1) = 1, cbrt(8) = 2, cbrt(27) = 3
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::cbrt_simd;
///
/// let x = array![0.0_f64, 1.0, 8.0, 27.0, -8.0];
/// let result = cbrt_simd(&x.view());
/// // Result: [0.0, 1.0, 2.0, 3.0, -2.0]
/// ```
///
/// # Applications
///
/// - **Statistics**: Transforming skewed data
/// - **Physics**: Volume calculations from cubic dimensions
/// - **Numerical Analysis**: Root-finding algorithms
pub fn cbrt_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_cbrt(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.cbrt())
}

/// Compute ln(1+x) for each element (SIMD-accelerated, numerically stable).
///
/// Computes the natural logarithm of (1+x) with improved accuracy for small x values
/// where direct computation of ln(1+x) would lose precision.
///
/// # Arguments
///
/// * `x` - Input 1D array with values > -1
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is ln(1+x).
///
/// # Mathematical Properties
///
/// - ln_1p(0) = 0
/// - ln_1p(x) ≈ x for |x| << 1 (Taylor series first term)
/// - More accurate than ln(1+x) when |x| is small
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::ln_1p_simd;
///
/// let x = array![0.0_f64, 1.0, 1e-15, -0.5];
/// let result = ln_1p_simd(&x.view());
/// // Result: [0.0, ln(2), ≈1e-15, -ln(2)]
/// ```
///
/// # Applications
///
/// - **Finance**: Continuous compound interest rates
/// - **Statistics**: Log-likelihood computations
/// - **Numerical Analysis**: Avoiding catastrophic cancellation
pub fn ln_1p_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_ln_1p(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.ln_1p())
}

/// Compute exp(x)-1 for each element (SIMD-accelerated, numerically stable).
///
/// Computes e^x - 1 with improved accuracy for small x values where direct
/// computation would lose precision.
///
/// # Arguments
///
/// * `x` - Input 1D array
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is exp(x)-1.
///
/// # Mathematical Properties
///
/// - exp_m1(0) = 0
/// - exp_m1(x) ≈ x for |x| << 1 (Taylor series first term)
/// - exp_m1(-x) = -exp_m1(x) / (1 + exp_m1(x))
/// - More accurate than exp(x)-1 when |x| is small
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::exp_m1_simd;
///
/// let x = array![0.0_f64, 1.0, 1e-15, -1.0];
/// let result = exp_m1_simd(&x.view());
/// // Result: [0.0, e-1, ≈1e-15, 1/e - 1]
/// ```
///
/// # Applications
///
/// - **Finance**: Continuous compounding calculations
/// - **Physics**: Small perturbation calculations
/// - **Numerical Analysis**: Avoiding catastrophic cancellation
pub fn exp_m1_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_exp_m1(x);
    }

    // Scalar fallback for small arrays
    x.mapv(|val| val.exp_m1())
}

/// Convert degrees to radians for each element (SIMD-accelerated).
///
/// Computes x * π / 180 for each element, converting angle measurements
/// from degrees to radians.
///
/// # Arguments
///
/// * `x` - Input 1D array of angles in degrees
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the
/// corresponding angle in radians.
///
/// # Performance
///
/// Uses SIMD scalar multiplication for optimal performance.
///
/// # Mathematical Properties
///
/// - to_radians(0) = 0
/// - to_radians(90) = π/2
/// - to_radians(180) = π
/// - to_radians(360) = 2π
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::to_radians_simd;
///
/// let degrees = array![0.0, 90.0, 180.0, 360.0];
/// let radians = to_radians_simd(&degrees.view());
/// // Result: [0.0, π/2, π, 2π]
/// ```
///
/// # Applications
///
/// - **Trigonometry**: Converting human-readable angles to radian form
/// - **Graphics**: Rotation transformations
/// - **Physics**: Angular velocity and position calculations
/// - **Navigation**: Bearing and heading conversions
pub fn to_radians_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_to_radians(x);
    }

    // Scalar fallback for small arrays
    F::simd_to_radians(x)
}

/// Convert radians to degrees for each element (SIMD-accelerated).
///
/// Computes x * 180 / π for each element, converting angle measurements
/// from radians to degrees.
///
/// # Arguments
///
/// * `x` - Input 1D array of angles in radians
///
/// # Returns
///
/// `Array1<F>` with the same length as input, where each element is the
/// corresponding angle in degrees.
///
/// # Performance
///
/// Uses SIMD scalar multiplication for optimal performance.
///
/// # Mathematical Properties
///
/// - to_degrees(0) = 0
/// - to_degrees(π/2) = 90
/// - to_degrees(π) = 180
/// - to_degrees(2π) = 360
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::to_degrees_simd;
/// use std::f64::consts::PI;
///
/// let radians = array![0.0, PI/2.0, PI, 2.0*PI];
/// let degrees = to_degrees_simd(&radians.view());
/// // Result: [0.0, 90.0, 180.0, 360.0]
/// ```
///
/// # Applications
///
/// - **Trigonometry**: Converting calculation results to human-readable form
/// - **Graphics**: Displaying rotation angles
/// - **Physics**: Reporting angular measurements
/// - **Navigation**: Displaying compass headings
pub fn to_degrees_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_to_degrees(x);
    }

    // Scalar fallback for small arrays
    F::simd_to_degrees(x)
}

/// Computes the element-wise digamma function ψ(x) = d/dx ln(Γ(x)) using SIMD acceleration.
///
/// The digamma function is the logarithmic derivative of the gamma function.
/// It is essential for:
/// - Bayesian variational inference (Dirichlet, Beta priors)
/// - Maximum likelihood estimation for gamma distributions
/// - Statistical special function computations
///
/// # Implementation Details
///
/// The implementation uses three techniques for high accuracy:
/// 1. **Reflection formula**: For x < 0.5, uses ψ(1-x) - π/tan(πx)
/// 2. **Recurrence relation**: For small x, uses ψ(x+1) = ψ(x) + 1/x
/// 3. **Asymptotic expansion**: For large x, uses Bernoulli number series
///
/// # Arguments
///
/// * `x` - Input array of values. Should avoid non-positive integers where digamma has poles.
///
/// # Returns
///
/// Array of ψ(x) values, same shape as input.
///
/// # Mathematical Properties
///
/// - ψ(1) = -γ (Euler-Mascheroni constant ≈ -0.5772)
/// - ψ(n) = -γ + Σ(k=1 to n-1) 1/k for positive integers
/// - ψ(x+1) = ψ(x) + 1/x
/// - ψ(1-x) - ψ(x) = π·cot(πx)
///
/// # Example
///
/// ```ignore
/// use scirs2_core::ndarray_ext::elementwise::digamma_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![1.0f64, 2.0, 3.0, 4.0];
/// let result = digamma_simd(&x.view());
/// // ψ(1) ≈ -0.5772, ψ(2) ≈ 0.4228, ψ(3) ≈ 0.9228, ψ(4) ≈ 1.2561
/// ```
pub fn digamma_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_digamma(x);
    }

    // Scalar fallback for small arrays
    F::simd_digamma(x)
}

/// Computes the element-wise trigamma function ψ'(x) = d²/dx² ln(Γ(x)) using SIMD acceleration.
///
/// The trigamma function is the second derivative of the log-gamma function (first derivative
/// of digamma). It is critical for:
/// - Fisher information computation in Bayesian inference
/// - Variance of gamma and beta distribution parameters
/// - Maximum likelihood estimation for gamma distributions
/// - Sensitivity analysis in Bayesian variational inference
///
/// # Implementation Details
///
/// The implementation uses three techniques for high accuracy:
/// 1. **Reflection formula**: For x < 0, uses ψ'(1-x) + ψ'(x) = π²/sin²(πx)
/// 2. **Recurrence relation**: For small x, uses ψ'(x+1) = ψ'(x) - 1/x²
/// 3. **Asymptotic expansion**: For large x, uses ψ'(x) ≈ 1/x + 1/(2x²) + B₂/x³ - B₄/x⁵ + ...
///
/// # Arguments
///
/// * `x` - Input array of values. Should avoid non-positive integers where trigamma has poles.
///
/// # Returns
///
/// Array of ψ'(x) values, same shape as input.
///
/// # Mathematical Properties
///
/// - ψ'(1) = π²/6 ≈ 1.6449340668 (Basel problem)
/// - ψ'(n) = π²/6 - Σ(k=1 to n-1) 1/k² for positive integers
/// - ψ'(x+1) = ψ'(x) - 1/x²
/// - For large x: ψ'(x) → 1/x
///
/// # Example
///
/// ```ignore
/// use scirs2_core::ndarray_ext::elementwise::trigamma_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![1.0f64, 2.0, 3.0, 4.0];
/// let result = trigamma_simd(&x.view());
/// // ψ'(1) ≈ 1.6449, ψ'(2) ≈ 0.6449, ψ'(3) ≈ 0.3949, ψ'(4) ≈ 0.2838
/// ```
pub fn trigamma_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_trigamma(x);
    }

    // Scalar fallback for small arrays
    F::simd_trigamma(x)
}

/// Computes the element-wise log-gamma function ln(Γ(x)) using SIMD acceleration.
///
/// The log-gamma function is more numerically stable than computing `gamma(x).ln()`
/// since it avoids overflow for large arguments and handles the full range of inputs.
/// This function is extensively used in:
/// - Statistical distributions (gamma, beta, binomial, Poisson)
/// - Bayesian inference (prior/posterior computations)
/// - Maximum likelihood estimation
/// - Combinatorics (log of binomial coefficients)
///
/// # Implementation Details
///
/// Uses the Lanczos approximation with reflection formula:
/// - For x >= 0.5: ln(Γ(z)) = ln(√(2π)) + (z-0.5)·ln(t) - t + ln(sum) where t = z + g - 0.5
/// - For x < 0.5: ln(Γ(z)) = ln(π) - ln(|sin(πz)|) - ln(Γ(1-z))
///
/// # Arguments
///
/// * `x` - Input array of values. Poles at non-positive integers (returns +∞).
///
/// # Returns
///
/// Array of ln(Γ(x)) values, same shape as input.
///
/// # Mathematical Properties
///
/// - ln(Γ(1)) = ln(Γ(2)) = 0
/// - ln(Γ(n)) = ln((n-1)!) for positive integers
/// - ln(Γ(1/2)) = ln(√π) ≈ 0.5724
/// - For large x: ln(Γ(x)) ≈ (x-0.5)·ln(x) - x + ln(√(2π))
///
/// # Example
///
/// ```ignore
/// use scirs2_core::ndarray_ext::elementwise::ln_gamma_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![1.0f64, 2.0, 3.0, 4.0, 5.0];
/// let result = ln_gamma_simd(&x.view());
/// // ln(Γ(1)) = 0, ln(Γ(2)) = 0, ln(Γ(3)) = ln(2!) ≈ 0.693
/// // ln(Γ(4)) = ln(3!) ≈ 1.792, ln(Γ(5)) = ln(4!) ≈ 3.178
/// ```
pub fn ln_gamma_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_ln_gamma(x);
    }

    // Scalar fallback for small arrays
    F::simd_ln_gamma(x)
}

/// Element-wise error function erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
///
/// Uses SIMD acceleration when available for optimal performance.
/// Critical for normal distribution CDF: Φ(x) = 0.5 * (1 + erf(x/√2))
///
/// # Properties
/// - erf(0) = 0
/// - erf(∞) = 1, erf(-∞) = -1
/// - erf(-x) = -erf(x) (odd function)
/// - erf(1) ≈ 0.8427007929497148
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::erf_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0_f64, 1.0, 2.0, -1.0];
/// let result = erf_simd(&x.view());
/// assert!((result[0] - 0.0_f64).abs() < 1e-10);  // erf(0) = 0
/// assert!((result[1] - 0.8427007929497148_f64).abs() < 1e-6);  // erf(1)
/// assert!((result[3] + 0.8427007929497148_f64).abs() < 1e-6);  // erf(-1) = -erf(1)
/// ```
pub fn erf_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_erf(x);
    }

    // Scalar fallback for small arrays
    F::simd_erf(x)
}

/// Element-wise complementary error function erfc(x) = 1 - erf(x)
///
/// More numerically stable than computing 1 - erf(x) directly for large x.
/// Uses SIMD acceleration when available for optimal performance.
///
/// # Properties
/// - erfc(0) = 1
/// - erfc(∞) = 0, erfc(-∞) = 2
/// - erfc(x) = 1 - erf(x)
/// - erfc(-x) = 2 - erfc(x)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::erfc_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f64, 1.0, 2.0, 3.0];
/// let result = erfc_simd(&x.view());
/// assert!((result[0] - 1.0).abs() < 1e-10);  // erfc(0) = 1
/// assert!((result[1] - 0.1572992070502852).abs() < 1e-6);  // erfc(1)
/// ```
pub fn erfc_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_erfc(x);
    }

    // Scalar fallback for small arrays
    F::simd_erfc(x)
}

/// Element-wise inverse error function erfinv(y) = x such that erf(x) = y
///
/// Uses SIMD acceleration when available for optimal performance.
/// Critical for inverse normal CDF (probit function): Φ⁻¹(p) = √2 * erfinv(2p - 1)
///
/// # Domain and Range
/// - Domain: (-1, 1)
/// - Range: (-∞, ∞)
/// - erfinv(-1) = -∞, erfinv(1) = ∞
///
/// # Properties
/// - erfinv(0) = 0
/// - erfinv(-y) = -erfinv(y) (odd function)
/// - erf(erfinv(y)) = y
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::erfinv_simd;
/// use scirs2_core::ndarray::array;
///
/// let y = array![0.0f64, 0.5, -0.5, 0.9];
/// let result = erfinv_simd(&y.view());
/// assert!((result[0] - 0.0).abs() < 1e-10);  // erfinv(0) = 0
/// ```
pub fn erfinv_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_erfinv(x);
    }

    // Scalar fallback for small arrays
    F::simd_erfinv(x)
}

/// Element-wise inverse complementary error function erfcinv(y) = x such that erfc(x) = y
///
/// More numerically stable than erfinv(1 - y) for y close to 0.
/// Uses SIMD acceleration when available for optimal performance.
///
/// # Domain and Range
/// - Domain: (0, 2)
/// - Range: (-∞, ∞)
/// - erfcinv(0) = ∞, erfcinv(2) = -∞
///
/// # Properties
/// - erfcinv(1) = 0
/// - erfcinv(y) = erfinv(1 - y)
/// - erfc(erfcinv(y)) = y
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::erfcinv_simd;
/// use scirs2_core::ndarray::array;
///
/// let y = array![1.0f64, 0.5, 1.5];
/// let result = erfcinv_simd(&y.view());
/// assert!((result[0] - 0.0).abs() < 1e-10);  // erfcinv(1) = 0
/// ```
pub fn erfcinv_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_erfcinv(x);
    }

    // Scalar fallback for small arrays
    F::simd_erfcinv(x)
}

/// Compute the element-wise sigmoid (logistic) function of an array.
///
/// The sigmoid function is defined as:
/// σ(x) = 1 / (1 + exp(-x))
///
/// This is critical for neural networks, logistic regression, and probability modeling.
/// The implementation is numerically stable, avoiding overflow for large |x|.
///
/// # Properties
///
/// - Range: (0, 1)
/// - σ(0) = 0.5
/// - σ(-x) = 1 - σ(x)
/// - Derivative: σ'(x) = σ(x)(1 - σ(x))
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::sigmoid_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f64, 1.0, -1.0];
/// let result = sigmoid_simd(&x.view());
/// assert!((result[0] - 0.5).abs() < 1e-10);  // sigmoid(0) = 0.5
/// ```
pub fn sigmoid_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_sigmoid(x);
    }

    // Scalar fallback for small arrays
    F::simd_sigmoid(x)
}

/// Compute the element-wise GELU (Gaussian Error Linear Unit) of an array.
///
/// The GELU function is defined as:
/// GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / √2))
///
/// Where Φ(x) is the cumulative distribution function of the standard normal distribution.
/// GELU is critical for Transformer models (BERT, GPT, etc.) and provides a smooth
/// approximation of ReLU.
///
/// # Properties
///
/// - GELU(0) = 0
/// - GELU(x) ≈ x for large positive x
/// - GELU(x) ≈ 0 for large negative x
/// - Smooth and differentiable everywhere
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::gelu_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f64, 1.0, -1.0];
/// let result = gelu_simd(&x.view());
/// assert!(result[0].abs() < 1e-10);  // GELU(0) = 0
/// ```
pub fn gelu_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_gelu(x);
    }

    // Scalar fallback for small arrays
    F::simd_gelu(x)
}

/// SIMD-accelerated Swish (SiLU - Sigmoid Linear Unit) activation function
///
/// Computes the Swish activation function element-wise:
/// `Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))`
///
/// Swish is a self-gated activation function discovered through neural architecture
/// search that has become a popular choice for modern neural networks.
///
/// # Key Properties
///
/// - Swish(0) = 0
/// - Swish is smooth and non-monotonic
/// - Has a small negative region (unlike ReLU)
/// - Self-gating: x modulates its own activation via sigmoid
/// - Unbounded above, bounded below (minimum ≈ -0.278 at x ≈ -1.278)
///
/// # Usage in Deep Learning
///
/// Swish is used in:
/// - EfficientNet and EfficientNetV2
/// - GPT-NeoX and other large language models
/// - MobileNetV3
/// - Many modern vision and NLP architectures
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::swish_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f64, 1.0, -1.0];
/// let result = swish_simd(&x.view());
/// assert!(result[0].abs() < 1e-10);  // Swish(0) = 0
/// // Swish(1) ≈ 0.7311
/// assert!((result[1] - 0.7310585786).abs() < 1e-6);
/// // Swish(-1) ≈ -0.2689
/// assert!((result[2] - (-0.2689414214)).abs() < 1e-6);
/// ```
pub fn swish_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_swish(x);
    }

    // Scalar fallback for small arrays
    F::simd_swish(x)
}

/// SIMD-accelerated Softplus activation function
///
/// Computes the Softplus activation function element-wise:
/// `Softplus(x) = ln(1 + exp(x))`
///
/// Softplus is a smooth approximation of ReLU and is commonly used in
/// probabilistic models, Bayesian deep learning, and smooth counting.
///
/// # Key Properties
///
/// - Softplus(0) = ln(2) ≈ 0.693
/// - Always positive (> 0 for all x)
/// - Derivative: softplus'(x) = sigmoid(x)
/// - Approaches ReLU for x → +∞: softplus(x) ≈ x
/// - Approaches 0 for x → -∞: softplus(x) ≈ exp(x) ≈ 0
///
/// # Usage
///
/// Softplus is used in:
/// - Probabilistic models (output layer for positive quantities)
/// - Bayesian deep learning
/// - Smooth counting applications
/// - As a smooth ReLU replacement
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::softplus_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f64, 1.0, -1.0];
/// let result = softplus_simd(&x.view());
/// // Softplus(0) = ln(2) ≈ 0.693
/// assert!((result[0] - 0.6931471805599453).abs() < 1e-10);
/// // Softplus(1) = ln(1 + e) ≈ 1.3133
/// assert!((result[1] - 1.3132616875182228).abs() < 1e-10);
/// // Softplus(-1) = ln(1 + 1/e) ≈ 0.3133
/// assert!((result[2] - 0.31326168751822286).abs() < 1e-10);
/// ```
pub fn softplus_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_softplus(x);
    }

    // Scalar fallback for small arrays
    F::simd_softplus(x)
}

/// SIMD-accelerated Mish activation function
///
/// Computes the Mish activation function element-wise:
/// `Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))`
///
/// Mish is a self-regularized non-monotonic activation function that combines
/// the benefits of ReLU, Swish, and other modern activations.
///
/// # Key Properties
///
/// - Mish(0) = 0
/// - Smooth and non-monotonic
/// - Self-regularizing (bounded negative region)
/// - Unbounded above, bounded below (minimum ≈ -0.31 at x ≈ -1.2)
/// - Derivative: complex but well-behaved
///
/// # Usage in Deep Learning
///
/// Mish is used in:
/// - YOLOv4 and YOLOv5 object detection
/// - Modern convolutional neural networks
/// - Image classification and segmentation tasks
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray_ext::elementwise::mish_simd;
/// use scirs2_core::ndarray::array;
///
/// let x = array![0.0f64, 1.0, -1.0];
/// let result = mish_simd(&x.view());
/// // Mish(0) = 0
/// assert!(result[0].abs() < 1e-10);
/// // Mish(1) = 1 * tanh(ln(1 + e)) ≈ 0.8651
/// assert!((result[1] - 0.8650983882673103).abs() < 1e-10);
/// ```
pub fn mish_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_mish(x);
    }

    // Scalar fallback for small arrays
    F::simd_mish(x)
}

/// Apply ELU (Exponential Linear Unit) activation using SIMD operations
///
/// ELU is defined as:
/// - f(x) = x, if x >= 0
/// - f(x) = α * (exp(x) - 1), if x < 0
///
/// ELU is used in deep neural networks to:
/// - Push mean activations closer to zero (faster learning)
/// - Have negative values (unlike ReLU) for better gradient flow
/// - Have a smooth curve everywhere (unlike Leaky ReLU)
///
/// # Arguments
/// * `x` - Input array
/// * `alpha` - Scaling factor for negative inputs (commonly 1.0)
///
/// # Returns
/// * Array with ELU applied elementwise
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::elu_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![1.0_f32, 0.0, -1.0, -2.0];
/// let result = elu_simd(&x.view(), 1.0);
/// assert!((result[0] - 1.0).abs() < 1e-6);  // Positive: unchanged
/// assert!((result[1] - 0.0).abs() < 1e-6);  // Zero: unchanged
/// assert!(result[2] < 0.0);  // Negative: α * (exp(x) - 1) < 0
/// ```
pub fn elu_simd<F>(x: &ArrayView1<F>, alpha: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_elu(x, alpha);
    }

    // Scalar fallback for small arrays
    F::simd_elu(x, alpha)
}

/// Apply Leaky ReLU / PReLU activation using SIMD operations
///
/// Leaky ReLU (Parametric ReLU when alpha is learned) is defined as:
/// - f(x) = x, if x >= 0
/// - f(x) = alpha * x, if x < 0
///
/// Leaky ReLU addresses the "dying ReLU" problem by allowing
/// a small gradient for negative inputs, preventing neurons from
/// becoming permanently inactive.
///
/// # Arguments
/// * `x` - Input array
/// * `alpha` - Slope for negative inputs (commonly 0.01 for Leaky ReLU, learned for PReLU)
///
/// # Returns
/// * Array with Leaky ReLU applied elementwise
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::leaky_relu_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![1.0_f32, 0.0, -1.0, -2.0];
/// let result = leaky_relu_simd(&x.view(), 0.01);
/// assert!((result[0] - 1.0).abs() < 1e-6);    // Positive: unchanged
/// assert!((result[1] - 0.0).abs() < 1e-6);    // Zero: unchanged
/// assert!((result[2] - (-0.01)).abs() < 1e-6); // Negative: alpha * x
/// ```
pub fn leaky_relu_simd<F>(x: &ArrayView1<F>, alpha: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_leaky_relu(x, alpha);
    }

    // Scalar fallback for small arrays
    F::simd_leaky_relu(x, alpha)
}

/// Alias for leaky_relu_simd - PReLU (Parametric ReLU)
///
/// PReLU is mathematically identical to Leaky ReLU, but the alpha
/// parameter is learned during training rather than being fixed.
/// This function provides a convenience alias for neural network code
/// that uses PReLU terminology.
#[inline]
pub fn prelu_simd<F>(x: &ArrayView1<F>, alpha: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    leaky_relu_simd(x, alpha)
}

/// Apply SELU (Scaled Exponential Linear Unit) activation using SIMD operations
///
/// SELU is defined as:
/// - f(x) = λ * x, if x > 0
/// - f(x) = λ * α * (exp(x) - 1), if x <= 0
///
/// where λ ≈ 1.0507 and α ≈ 1.6733 are fixed constants.
///
/// SELU is the key activation for Self-Normalizing Neural Networks (SNNs):
/// - Automatically maintains mean ≈ 0 and variance ≈ 1 through layers
/// - Eliminates the need for Batch Normalization
/// - Requires LeCun Normal initialization for weights
/// - Works best with fully-connected networks
///
/// # Arguments
/// * `x` - Input array
///
/// # Returns
/// * Array with SELU applied elementwise
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::selu_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![1.0_f32, 0.0, -1.0, -2.0];
/// let result = selu_simd(&x.view());
/// assert!(result[0] > 1.0);  // Positive: scaled by λ ≈ 1.0507
/// assert!((result[1] - 0.0).abs() < 1e-6);  // Zero: unchanged
/// assert!(result[2] < 0.0);  // Negative: λ * α * (exp(x) - 1) < 0
/// ```
pub fn selu_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_selu(x);
    }

    // Scalar fallback for small arrays
    F::simd_selu(x)
}

/// Apply Hardsigmoid activation using SIMD operations
///
/// Hardsigmoid is defined as:
/// - f(x) = 0, if x <= -3
/// - f(x) = 1, if x >= 3
/// - f(x) = (x + 3) / 6, otherwise
///
/// Hardsigmoid is a piecewise linear approximation of sigmoid:
/// - Used in MobileNetV3 for efficient mobile inference
/// - Avoids expensive exp() computation
/// - Output always in range [0, 1]
///
/// # Arguments
/// * `x` - Input array
///
/// # Returns
/// * Array with Hardsigmoid applied elementwise
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::hardsigmoid_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![-4.0_f32, -3.0, 0.0, 3.0, 4.0];
/// let result = hardsigmoid_simd(&x.view());
/// assert!((result[0] - 0.0).abs() < 1e-6);   // Saturated at 0
/// assert!((result[1] - 0.0).abs() < 1e-6);   // Boundary
/// assert!((result[2] - 0.5).abs() < 1e-6);   // Linear region: (0+3)/6 = 0.5
/// assert!((result[3] - 1.0).abs() < 1e-6);   // Boundary
/// assert!((result[4] - 1.0).abs() < 1e-6);   // Saturated at 1
/// ```
pub fn hardsigmoid_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_hardsigmoid(x);
    }

    // Scalar fallback for small arrays
    F::simd_hardsigmoid(x)
}

/// Apply Hardswish activation using SIMD operations
///
/// Hardswish is defined as:
/// - f(x) = 0, if x <= -3
/// - f(x) = x, if x >= 3
/// - f(x) = x * (x + 3) / 6, otherwise
///
/// Hardswish is a piecewise linear approximation of Swish:
/// - Used in MobileNetV3 for efficient mobile inference
/// - Self-gating like Swish but without exp() computation
/// - Smooth at boundaries despite being piecewise
///
/// # Arguments
/// * `x` - Input array
///
/// # Returns
/// * Array with Hardswish applied elementwise
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::hardswish_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![-4.0_f32, -3.0, 0.0, 3.0, 4.0];
/// let result = hardswish_simd(&x.view());
/// assert!((result[0] - 0.0).abs() < 1e-6);   // Saturated at 0
/// assert!((result[1] - 0.0).abs() < 1e-6);   // Boundary: -3 * 0 / 6 = 0
/// assert!((result[2] - 0.0).abs() < 1e-6);   // Linear: 0 * 3 / 6 = 0
/// assert!((result[3] - 3.0).abs() < 1e-6);   // Boundary: identity
/// assert!((result[4] - 4.0).abs() < 1e-6);   // Identity region
/// ```
pub fn hardswish_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_hardswish(x);
    }

    // Scalar fallback for small arrays
    F::simd_hardswish(x)
}

/// Apply Sinc function using SIMD operations
///
/// The normalized sinc function is defined as:
/// - sinc(x) = sin(πx) / (πx) for x ≠ 0
/// - sinc(0) = 1 (by L'Hôpital's rule)
///
/// The sinc function is fundamental in signal processing:
/// - Ideal low-pass filter impulse response
/// - Whittaker-Shannon interpolation formula
/// - Windowing functions (Lanczos kernel)
/// - Sampling theory and Nyquist theorem
///
/// # Arguments
/// * `x` - Input array
///
/// # Returns
/// * Array with sinc applied elementwise
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::sinc_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![0.0_f64, 0.5, 1.0, 2.0];
/// let result = sinc_simd(&x.view());
/// assert!((result[0] - 1.0).abs() < 1e-10);   // sinc(0) = 1
/// assert!((result[2] - 0.0).abs() < 1e-10);   // sinc(1) = 0
/// assert!((result[3] - 0.0).abs() < 1e-10);   // sinc(2) = 0
/// ```
pub fn sinc_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_sinc(x);
    }

    // Scalar fallback for small arrays
    F::simd_sinc(x)
}

/// Apply Log-Softmax function using SIMD operations
///
/// The log-softmax function is defined as:
/// log_softmax(x_i) = x_i - log(Σ_j exp(x_j))
///
/// This is more numerically stable than computing log(softmax(x)) directly,
/// and is critical for neural network training:
/// - Cross-entropy loss computation
/// - Classification networks (output layer)
/// - Transformer language models
/// - Large language models (LLMs)
///
/// # Mathematical Properties
///
/// - log_softmax(x) = x - logsumexp(x)
/// - Σ exp(log_softmax(x)) = 1 (outputs are log-probabilities)
/// - log_softmax(x + c) = log_softmax(x) (invariant to constant shift)
/// - Maximum output element approaches 0 for peaked distributions
///
/// # Arguments
/// * `x` - Input array of logits (unnormalized log-probabilities)
///
/// # Returns
/// * Array with log-softmax applied elementwise (log-probabilities that sum to 0)
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::log_softmax_simd;
/// use ndarray::{array, ArrayView1};
///
/// let logits = array![1.0_f64, 2.0, 3.0];
/// let log_probs = log_softmax_simd(&logits.view());
///
/// // Verify: exp(log_probs) sums to 1
/// let probs: f64 = log_probs.mapv(|lp| lp.exp()).sum();
/// assert!((probs - 1.0).abs() < 1e-10);
///
/// // log_softmax values are always <= 0
/// for &lp in log_probs.iter() {
///     assert!(lp <= 0.0);
/// }
/// ```
///
/// # Applications
///
/// - **Cross-Entropy Loss**: -Σ target * log_softmax(logits)
/// - **Classification**: Converting logits to log-probabilities
/// - **Transformers**: Final output layer for token prediction
/// - **Language Models**: Computing perplexity and token probabilities
/// - **Numerical Stability**: Avoiding underflow in softmax computation
pub fn log_softmax_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_log_softmax(x);
    }

    // Scalar fallback for small arrays
    F::simd_log_softmax(x)
}

/// Apply inverse hyperbolic sine (asinh) using SIMD operations
///
/// The inverse hyperbolic sine is defined as:
/// asinh(x) = ln(x + √(x² + 1))
///
/// Domain: (-∞, +∞), Range: (-∞, +∞)
/// This is the inverse function of sinh.
///
/// # Mathematical Properties
///
/// - asinh(0) = 0
/// - asinh(-x) = -asinh(x) (odd function)
/// - asinh'(x) = 1/√(x² + 1)
/// - For large x: asinh(x) ≈ ln(2x)
///
/// # Arguments
/// * `x` - Input array
///
/// # Returns
/// * Array with asinh applied elementwise
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::asinh_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![0.0_f64, 1.0, -1.0, 10.0];
/// let result = asinh_simd(&x.view());
/// assert!((result[0] - 0.0).abs() < 1e-10);
/// assert!((result[1] - 0.881373587).abs() < 1e-6);  // asinh(1)
/// assert!((result[1] + result[2]).abs() < 1e-10);  // odd function
/// ```
///
/// # Applications
///
/// - **Hyperbolic Geometry**: Distance calculations in hyperbolic space
/// - **Special Relativity**: Rapidity in Lorentz transformations
/// - **Signal Processing**: Parametric representation of signals
/// - **Statistics**: Transformations for variance stabilization
pub fn asinh_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_asinh(x);
    }

    // Scalar fallback for small arrays
    F::simd_asinh(x)
}

/// Apply inverse hyperbolic cosine (acosh) using SIMD operations
///
/// The inverse hyperbolic cosine is defined as:
/// acosh(x) = ln(x + √(x² - 1))
///
/// Domain: [1, +∞), Range: [0, +∞)
/// Returns NaN for x < 1.
/// This is the inverse function of cosh.
///
/// # Mathematical Properties
///
/// - acosh(1) = 0
/// - acosh(x) is monotonically increasing for x ≥ 1
/// - acosh'(x) = 1/√(x² - 1)
/// - For large x: acosh(x) ≈ ln(2x)
///
/// # Arguments
/// * `x` - Input array (values should be ≥ 1 for valid results)
///
/// # Returns
/// * Array with acosh applied elementwise (NaN for values < 1)
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::acosh_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![1.0_f64, 2.0, 10.0];
/// let result = acosh_simd(&x.view());
/// assert!((result[0] - 0.0).abs() < 1e-10);         // acosh(1) = 0
/// assert!((result[1] - 1.316957897).abs() < 1e-6);  // acosh(2)
///
/// // Out of domain returns NaN
/// let x_invalid = array![0.5_f64];
/// let result_invalid = acosh_simd(&x_invalid.view());
/// assert!(result_invalid[0].is_nan());
/// ```
///
/// # Applications
///
/// - **Hyperbolic Geometry**: Distance in Poincaré disk model
/// - **Physics**: Catenary curves, suspension bridges
/// - **Electronics**: Transmission line analysis
/// - **Computer Graphics**: Hyperbolic tessellations
pub fn acosh_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_acosh(x);
    }

    // Scalar fallback for small arrays
    F::simd_acosh(x)
}

/// Apply inverse hyperbolic tangent (atanh) using SIMD operations
///
/// The inverse hyperbolic tangent is defined as:
/// atanh(x) = 0.5 * ln((1+x)/(1-x))
///
/// Domain: (-1, 1), Range: (-∞, +∞)
/// Returns ±∞ at x = ±1, NaN for |x| > 1.
/// This is the inverse function of tanh.
///
/// # Mathematical Properties
///
/// - atanh(0) = 0
/// - atanh(-x) = -atanh(x) (odd function)
/// - atanh(±1) = ±∞
/// - atanh'(x) = 1/(1 - x²)
///
/// # Arguments
/// * `x` - Input array (values should be in (-1, 1) for finite results)
///
/// # Returns
/// * Array with atanh applied elementwise
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::atanh_simd;
/// use ndarray::{array, ArrayView1};
///
/// let x = array![0.0_f64, 0.5, -0.5, 0.99];
/// let result = atanh_simd(&x.view());
/// assert!((result[0] - 0.0).abs() < 1e-10);          // atanh(0) = 0
/// assert!((result[1] - 0.5493061).abs() < 1e-6);     // atanh(0.5)
/// assert!((result[1] + result[2]).abs() < 1e-10);   // odd function
///
/// // Boundaries
/// let x_boundary = array![1.0_f64, -1.0];
/// let result_boundary = atanh_simd(&x_boundary.view());
/// assert!(result_boundary[0].is_infinite() && result_boundary[0] > 0.0);
/// assert!(result_boundary[1].is_infinite() && result_boundary[1] < 0.0);
/// ```
///
/// # Applications
///
/// - **Statistics**: Fisher's z-transformation for correlation coefficients
/// - **Signal Processing**: Parametric signal representation
/// - **Probability**: Logit function relationship (atanh(x) = 0.5*logit((1+x)/2))
/// - **Machine Learning**: Activation function transformations
pub fn atanh_simd<F>(x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_atanh(x);
    }

    // Scalar fallback for small arrays
    F::simd_atanh(x)
}

/// Compute the Beta function B(a, b) using SIMD operations
///
/// The Beta function is defined as:
/// B(a, b) = Γ(a)Γ(b) / Γ(a+b)
///
/// This function computes `B(a[i], b[i])` for each pair of elements.
/// The Beta function is fundamental in:
/// - Beta distribution (Bayesian priors for probabilities)
/// - Binomial coefficients: C(n,k) = 1/((n+1)·B(n-k+1, k+1))
/// - Statistical hypothesis testing
/// - Machine learning (Dirichlet processes)
///
/// # Mathematical Properties
///
/// - B(a, b) = B(b, a) (symmetric)
/// - B(1, 1) = 1
/// - B(a, 1) = 1/a
/// - B(1, b) = 1/b
/// - B(a, b) = B(a+1, b) + B(a, b+1)
///
/// # Arguments
/// * `a` - First parameter array (must be > 0)
/// * `b` - Second parameter array (must be > 0, same length as `a`)
///
/// # Returns
/// * Array with `B(a[i], b[i])` for each pair
///
/// # Example
/// ```
/// use scirs2_core::ndarray_ext::elementwise::beta_simd;
/// use ndarray::{array, ArrayView1};
///
/// let a = array![1.0_f64, 2.0, 0.5];
/// let b = array![1.0_f64, 2.0, 0.5];
/// let result = beta_simd(&a.view(), &b.view());
/// assert!((result[0] - 1.0).abs() < 1e-10);       // B(1,1) = 1
/// assert!((result[1] - 1.0/6.0).abs() < 1e-10);  // B(2,2) = 1/6
/// assert!((result[2] - std::f64::consts::PI).abs() < 1e-8);  // B(0.5,0.5) = π
/// ```
///
/// # Applications
///
/// - **Beta Distribution**: PDF = x^(a-1)(1-x)^(b-1) / B(a,b)
/// - **Bayesian Statistics**: Prior/posterior for probability parameters
/// - **A/B Testing**: Conversion rate analysis
/// - **Machine Learning**: Dirichlet processes, topic modeling
pub fn beta_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_beta(a, b);
    }

    // Scalar fallback for small arrays
    F::simd_beta(a, b)
}

/// Compute the Log-Beta function ln(B(a, b)) using SIMD operations
///
/// The Log-Beta function is defined as:
/// ln(B(a, b)) = ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b))
///
/// This is more numerically stable than computing B(a,b) directly,
/// especially for large arguments where Γ would overflow.
///
/// # Mathematical Properties
///
/// - ln(B(a, b)) = ln(B(b, a)) (symmetric)
/// - ln(B(1, 1)) = 0
/// - ln(B(a, 1)) = -ln(a)
/// - For large a,b: ln(B(a,b)) ≈ 0.5*ln(2π) - (a+b-0.5)*ln(a+b) + (a-0.5)*ln(a) + (b-0.5)*ln(b)
///
/// # Arguments
/// * `a` - First parameter array (must be > 0)
/// * `b` - Second parameter array (must be > 0, same length as `a`)
///
/// # Returns
/// * Array with `ln(B(a[i], b[i]))` for each pair
///
/// # Example
/// ```ignore
/// use scirs2_core::ndarray_ext::elementwise::ln_beta_simd;
/// use scirs2_core::ndarray::{array, ArrayView1};
///
/// let a = array![1.0_f64, 2.0, 10.0];
/// let b = array![1.0_f64, 2.0, 10.0];
/// let result = ln_beta_simd(&a.view(), &b.view());
/// assert!((result[0] - 0.0_f64).abs() < 1e-10);  // ln(B(1,1)) = ln(1) = 0
/// assert!((result[1] - (-6.0_f64).ln()).abs() < 1e-10);  // ln(B(2,2)) = ln(1/6)
/// ```
///
/// # Applications
///
/// - **Numerical Stability**: Avoiding overflow in Beta distribution computations
/// - **Log-likelihood**: Direct computation of log-probabilities
/// - **Monte Carlo Methods**: Log-probability computations
/// - **Variational Inference**: KL divergence between Beta distributions
pub fn ln_beta_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_ln_beta(a, b);
    }

    // Scalar fallback for small arrays
    F::simd_ln_beta(a, b)
}

/// SIMD-accelerated linear interpolation
///
/// Computes element-wise linear interpolation: lerp(a, b, t) = a + t * (b - a)
/// When t=0, returns a; when t=1, returns b.
///
/// # Arguments
/// * `a` - Start values
/// * `b` - End values
/// * `t` - Interpolation parameter (typically in [0, 1])
///
/// # Returns
/// Array of interpolated values
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::lerp_simd;
///
/// let a = array![0.0_f32, 0.0, 0.0];
/// let b = array![10.0_f32, 20.0, 30.0];
///
/// // t = 0: returns a
/// let result = lerp_simd::<f32>(&a.view(), &b.view(), 0.0);
/// assert!((result[0] - 0.0).abs() < 1e-6);
///
/// // t = 1: returns b
/// let result = lerp_simd::<f32>(&a.view(), &b.view(), 1.0);
/// assert!((result[0] - 10.0).abs() < 1e-6);
///
/// // t = 0.5: midpoint
/// let result = lerp_simd::<f32>(&a.view(), &b.view(), 0.5);
/// assert!((result[0] - 5.0).abs() < 1e-6);
/// ```
///
/// # Use Cases
/// - Animation blending
/// - Color interpolation
/// - Smooth parameter transitions
/// - Gradient computation in neural networks
pub fn lerp_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>, t: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_lerp(a, b, t);
    }

    // Scalar fallback for small arrays
    F::simd_lerp(a, b, t)
}

/// SIMD-accelerated smoothstep interpolation
///
/// Returns smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1:
/// - Returns 0 if x <= edge0
/// - Returns 1 if x >= edge1
/// - Returns smooth curve: 3t² - 2t³ where t = (x - edge0) / (edge1 - edge0)
///
/// # Arguments
/// * `edge0` - Lower edge of the transition
/// * `edge1` - Upper edge of the transition
/// * `x` - Input values
///
/// # Returns
/// Array of smoothstep values in [0, 1]
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::smoothstep_simd;
///
/// let x = array![0.0_f32, 0.25, 0.5, 0.75, 1.0];
///
/// let result = smoothstep_simd::<f32>(0.0, 1.0, &x.view());
/// assert!((result[0] - 0.0).abs() < 1e-6);  // x=0 -> 0
/// assert!((result[2] - 0.5).abs() < 1e-6);  // x=0.5 -> 0.5
/// assert!((result[4] - 1.0).abs() < 1e-6);  // x=1 -> 1
/// ```
///
/// # Use Cases
/// - Shader programming (smooth transitions)
/// - Activation function variants
/// - Anti-aliasing
/// - Soft thresholding
pub fn smoothstep_simd<F>(edge0: F, edge1: F, x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_smoothstep(edge0, edge1, x);
    }

    // Scalar fallback for small arrays
    F::simd_smoothstep(edge0, edge1, x)
}

/// SIMD-accelerated hypotenuse calculation
///
/// Computes element-wise hypotenuse: hypot(x, y) = sqrt(x² + y²)
/// Uses the standard library implementation which handles overflow/underflow correctly.
///
/// # Arguments
/// * `x` - First coordinate values
/// * `y` - Second coordinate values
///
/// # Returns
/// Array of hypotenuse values
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::hypot_simd;
///
/// let x = array![3.0_f64, 5.0, 8.0];
/// let y = array![4.0_f64, 12.0, 15.0];
///
/// let result = hypot_simd::<f64>(&x.view(), &y.view());
/// assert!((result[0] - 5.0).abs() < 1e-14);   // 3-4-5 triangle
/// assert!((result[1] - 13.0).abs() < 1e-14);  // 5-12-13 triangle
/// assert!((result[2] - 17.0).abs() < 1e-14);  // 8-15-17 triangle
/// ```
///
/// # Use Cases
/// - Distance calculations in 2D
/// - Computing vector magnitudes
/// - Complex number modulus: |a+bi| = hypot(a, b)
/// - Graphics and physics simulations
pub fn hypot_simd<F>(x: &ArrayView1<F>, y: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() || y.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_hypot(x, y);
    }

    // Scalar fallback for small arrays
    F::simd_hypot(x, y)
}

/// SIMD-accelerated copysign operation
///
/// Returns element-wise magnitude of x with the sign of y.
///
/// # Arguments
/// * `x` - Magnitude source values
/// * `y` - Sign source values
///
/// # Returns
/// Array where each element has the magnitude of x and sign of y
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::copysign_simd;
///
/// let x = array![1.0_f64, -2.0, 3.0, -4.0];
/// let y = array![-1.0_f64, 1.0, 1.0, -1.0];
///
/// let result = copysign_simd::<f64>(&x.view(), &y.view());
/// assert!((result[0] - (-1.0)).abs() < 1e-14);  // 1 with sign of -1 = -1
/// assert!((result[1] - 2.0).abs() < 1e-14);     // -2 with sign of 1 = 2
/// assert!((result[2] - 3.0).abs() < 1e-14);     // 3 with sign of 1 = 3
/// assert!((result[3] - (-4.0)).abs() < 1e-14); // -4 with sign of -1 = -4
/// ```
///
/// # Use Cases
/// - Sign manipulation in numerical algorithms
/// - Implementing special functions (reflection formula)
/// - Gradient sign propagation
pub fn copysign_simd<F>(x: &ArrayView1<F>, y: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() || y.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_copysign(x, y);
    }

    // Scalar fallback for small arrays
    F::simd_copysign(x, y)
}

/// SIMD-accelerated smootherstep interpolation (Ken Perlin's improved version)
///
/// Returns smooth Hermite interpolation with second-order continuity.
/// Formula: 6t⁵ - 15t⁴ + 10t³ where t = (x - edge0) / (edge1 - edge0)
///
/// Unlike smoothstep, both the first AND second derivatives are zero at the boundaries,
/// making it ideal for procedural generation and high-quality animations.
///
/// # Arguments
/// * `edge0` - Lower edge of the transition
/// * `edge1` - Upper edge of the transition
/// * `x` - Input values
///
/// # Returns
/// Array of smootherstep values in [0, 1]
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::smootherstep_simd;
///
/// let x = array![0.0_f64, 0.25, 0.5, 0.75, 1.0];
///
/// let result = smootherstep_simd::<f64>(0.0, 1.0, &x.view());
/// assert!((result[0] - 0.0).abs() < 1e-14);  // x=0 -> 0
/// assert!((result[2] - 0.5).abs() < 1e-14);  // x=0.5 -> 0.5
/// assert!((result[4] - 1.0).abs() < 1e-14);  // x=1 -> 1
/// ```
///
/// # Use Cases
/// - Perlin noise and procedural generation
/// - High-quality animation easing
/// - Shader programming (smooth lighting transitions)
/// - Gradient-based optimization (smoother loss landscapes)
pub fn smootherstep_simd<F>(edge0: F, edge1: F, x: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if x.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(x.len()) {
        return F::simd_smootherstep(edge0, edge1, x);
    }

    // Scalar fallback for small arrays
    F::simd_smootherstep(edge0, edge1, x)
}

/// SIMD-accelerated logaddexp: log(exp(a) + exp(b))
///
/// Computes the logarithm of the sum of exponentials in a numerically stable way.
/// Uses the identity: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
///
/// # Arguments
/// * `a` - First input values
/// * `b` - Second input values
///
/// # Returns
/// Array of log(exp(a) + exp(b)) values
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::logaddexp_simd;
///
/// let a = array![0.0_f64, 1.0, 2.0];
/// let b = array![0.0_f64, 1.0, 2.0];
///
/// let result = logaddexp_simd::<f64>(&a.view(), &b.view());
/// // log(exp(0) + exp(0)) = log(2) ≈ 0.693
/// assert!((result[0] - 2.0_f64.ln()).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Log-probability computations (Bayesian inference)
/// - Log-likelihood calculations in ML
/// - Hidden Markov Model algorithms
/// - Cross-entropy loss functions
pub fn logaddexp_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_logaddexp(a, b);
    }

    // Scalar fallback for small arrays
    F::simd_logaddexp(a, b)
}

/// SIMD-accelerated logit function: log(p / (1-p))
///
/// The logit function maps probabilities in (0, 1) to log-odds in (-∞, +∞).
/// It is the inverse of the sigmoid (logistic) function.
///
/// # Arguments
/// * `a` - Probability values in (0, 1)
///
/// # Returns
/// Array of log-odds values
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::logit_simd;
///
/// let p = array![0.5_f64, 0.1, 0.9];
///
/// let result = logit_simd::<f64>(&p.view());
/// // logit(0.5) = log(1) = 0
/// assert!((result[0] - 0.0).abs() < 1e-14);
/// // logit(0.1) = log(0.1/0.9) ≈ -2.197
/// assert!((result[1] - (0.1_f64 / 0.9).ln()).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Logistic regression
/// - Probability calibration
/// - Converting probabilities to unbounded space
/// - Statistical modeling (link functions)
pub fn logit_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_logit(a);
    }

    // Scalar fallback for small arrays
    F::simd_logit(a)
}

/// SIMD-accelerated element-wise square: x²
///
/// Computes the square of each element. This is more efficient than `pow(x, 2)`
/// since it avoids the overhead of general exponentiation.
///
/// # Arguments
/// * `a` - Input values
///
/// # Returns
/// Array of squared values
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::square_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0, -4.0];
///
/// let result = square_simd::<f64>(&x.view());
/// assert!((result[0] - 1.0).abs() < 1e-14);   // 1² = 1
/// assert!((result[1] - 4.0).abs() < 1e-14);   // 2² = 4
/// assert!((result[2] - 9.0).abs() < 1e-14);   // 3² = 9
/// assert!((result[3] - 16.0).abs() < 1e-14);  // (-4)² = 16
/// ```
///
/// # Use Cases
/// - Computing squared distances
/// - Variance calculations (sum of squared deviations)
/// - Loss functions (MSE, L2 regularization)
/// - Polynomial evaluation
pub fn square_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_square(a);
    }

    // Scalar fallback for small arrays
    F::simd_square(a)
}

/// SIMD-accelerated inverse square root: 1/sqrt(x)
///
/// Computes the reciprocal of the square root for each element.
/// This operation is fundamental in graphics and physics simulations.
///
/// # Arguments
/// * `a` - Input values (should be positive)
///
/// # Returns
/// Array of 1/sqrt(x) values
/// - Returns INFINITY for x = 0
/// - Returns NaN for x < 0
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::rsqrt_simd;
///
/// let x = array![1.0_f64, 4.0, 9.0, 16.0];
///
/// let result = rsqrt_simd::<f64>(&x.view());
/// assert!((result[0] - 1.0).abs() < 1e-14);     // 1/sqrt(1) = 1
/// assert!((result[1] - 0.5).abs() < 1e-14);     // 1/sqrt(4) = 0.5
/// assert!((result[2] - 1.0/3.0).abs() < 1e-14); // 1/sqrt(9) = 1/3
/// assert!((result[3] - 0.25).abs() < 1e-14);    // 1/sqrt(16) = 0.25
/// ```
///
/// # Use Cases
/// - Vector normalization: v_normalized = v * rsqrt(dot(v, v))
/// - Quaternion normalization in 3D graphics
/// - Inverse distance weighting
/// - Fast approximate physics simulations
pub fn rsqrt_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_rsqrt(a);
    }

    // Scalar fallback for small arrays
    F::simd_rsqrt(a)
}

/// SIMD-accelerated simultaneous sin and cos computation
///
/// Computes both sine and cosine of each element in a single pass.
/// This is more efficient than calling sin and cos separately because
/// many implementations can compute both with minimal additional overhead.
///
/// # Arguments
/// * `a` - Input angles in radians
///
/// # Returns
/// Tuple of (sin_array, cos_array)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::sincos_simd;
/// use std::f64::consts::PI;
///
/// let x = array![0.0_f64, PI/6.0, PI/4.0, PI/2.0];
///
/// let (sin_result, cos_result) = sincos_simd::<f64>(&x.view());
/// assert!((sin_result[0] - 0.0).abs() < 1e-14);        // sin(0) = 0
/// assert!((cos_result[0] - 1.0).abs() < 1e-14);        // cos(0) = 1
/// assert!((sin_result[1] - 0.5).abs() < 1e-14);        // sin(π/6) = 0.5
/// assert!((sin_result[3] - 1.0).abs() < 1e-14);        // sin(π/2) = 1
/// assert!(cos_result[3].abs() < 1e-14);                // cos(π/2) ≈ 0
/// ```
///
/// # Use Cases
/// - Rotation matrices (need both sin and cos)
/// - Complex number operations: e^(iθ) = cos(θ) + i·sin(θ)
/// - Fourier transform calculations
/// - Wave simulations
/// - Animation and interpolation (circular motion)
pub fn sincos_simd<F>(a: &ArrayView1<F>) -> (Array1<F>, Array1<F>)
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return (Array1::zeros(0), Array1::zeros(0));
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_sincos(a);
    }

    // Scalar fallback for small arrays
    F::simd_sincos(a)
}

/// SIMD-accelerated numerically stable exp(x) - 1
///
/// Computes exp(x) - 1 accurately for small x values where the direct
/// calculation `exp(x) - 1` would suffer from catastrophic cancellation.
/// For |x| < 1e-10, the result is approximately x (Taylor expansion).
///
/// # Arguments
/// * `a` - Input values
///
/// # Returns
/// Array of exp(x) - 1 values
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::expm1_simd;
///
/// let x = array![0.0_f64, 1e-15, 1.0, -1.0];
///
/// let result = expm1_simd::<f64>(&x.view());
/// // exp(0) - 1 = 0
/// assert!((result[0] - 0.0).abs() < 1e-14);
/// // For small x: exp(x) - 1 ≈ x
/// assert!((result[1] - 1e-15).abs() < 1e-29);
/// // exp(1) - 1 ≈ 1.718
/// assert!((result[2] - (1.0_f64.exp() - 1.0)).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Financial calculations (compound interest for small rates)
/// - Numerical integration (avoiding cancellation errors)
/// - Statistical distributions (Poisson, exponential)
/// - Machine learning (softplus: log(1 + exp(x)))
pub fn expm1_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_expm1(a);
    }

    // Scalar fallback for small arrays
    F::simd_expm1(a)
}

/// SIMD-accelerated numerically stable ln(1 + x)
///
/// Computes ln(1 + x) accurately for small x values where the direct
/// calculation `(1 + x).ln()` would suffer from catastrophic cancellation.
/// For |x| < 1e-10, the result is approximately x - x²/2 (Taylor expansion).
///
/// # Arguments
/// * `a` - Input values (should be > -1)
///
/// # Returns
/// Array of ln(1 + x) values
/// - Returns -∞ for x = -1
/// - Returns NaN for x < -1
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::log1p_simd;
///
/// let x = array![0.0_f64, 1e-15, 1.0, -0.5];
///
/// let result = log1p_simd::<f64>(&x.view());
/// // ln(1 + 0) = 0
/// assert!((result[0] - 0.0).abs() < 1e-14);
/// // For small x: ln(1 + x) ≈ x
/// assert!((result[1] - 1e-15).abs() < 1e-29);
/// // ln(2) ≈ 0.693
/// assert!((result[2] - 2.0_f64.ln()).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Log-probability calculations (log(1 - p) for small p)
/// - Numerical integration
/// - Statistical distributions
/// - Machine learning (binary cross-entropy: -y·log(p) - (1-y)·log(1-p))
pub fn log1p_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_log1p(a);
    }

    // Scalar fallback for small arrays
    F::simd_log1p(a)
}

/// SIMD-accelerated element-wise clipping (clamping)
///
/// Clips (limits) each element to be within [min_val, max_val].
/// Values below min_val become min_val, values above max_val become max_val.
///
/// # Arguments
/// * `a` - Input array
/// * `min_val` - Minimum value (lower bound)
/// * `max_val` - Maximum value (upper bound)
///
/// # Returns
/// Array with values clipped to [min_val, max_val]
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::clip_simd;
///
/// let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0, 3.0];
///
/// let result = clip_simd::<f64>(&x.view(), 0.0, 1.0);
/// assert!((result[0] - 0.0).abs() < 1e-14);  // -2 clipped to 0
/// assert!((result[2] - 0.0).abs() < 1e-14);  // 0 unchanged
/// assert!((result[3] - 1.0).abs() < 1e-14);  // 1 unchanged
/// assert!((result[5] - 1.0).abs() < 1e-14);  // 3 clipped to 1
/// ```
///
/// # Use Cases
/// - Gradient clipping in neural networks
/// - Image processing (pixel value clamping)
/// - Bounded optimization
/// - Data normalization preprocessing
pub fn clip_simd<F>(a: &ArrayView1<F>, min_val: F, max_val: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_clip(a, min_val, max_val);
    }

    // Scalar fallback for small arrays
    F::simd_clip(a, min_val, max_val)
}

/// SIMD-accelerated cumulative sum (prefix sum)
///
/// Computes the cumulative sum of elements. `result[i] = sum(a[0..=i])`
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Array of cumulative sums
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::cumsum_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
///
/// let result = cumsum_simd::<f64>(&x.view());
/// assert!((result[0] - 1.0).abs() < 1e-14);   // 1
/// assert!((result[1] - 3.0).abs() < 1e-14);   // 1+2
/// assert!((result[2] - 6.0).abs() < 1e-14);   // 1+2+3
/// assert!((result[3] - 10.0).abs() < 1e-14);  // 1+2+3+4
/// assert!((result[4] - 15.0).abs() < 1e-14);  // 1+2+3+4+5
/// ```
///
/// # Use Cases
/// - Computing CDF from PDF
/// - Running totals and statistics
/// - Parallel prefix algorithms
/// - Integral image computation
pub fn cumsum_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_cumsum(a);
    }

    // Scalar fallback for small arrays
    F::simd_cumsum(a)
}

/// SIMD-accelerated cumulative product
///
/// Computes the cumulative product of elements. `result[i] = product(a[0..=i])`
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Array of cumulative products
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::cumprod_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0, 4.0];
///
/// let result = cumprod_simd::<f64>(&x.view());
/// assert!((result[0] - 1.0).abs() < 1e-14);   // 1
/// assert!((result[1] - 2.0).abs() < 1e-14);   // 1*2
/// assert!((result[2] - 6.0).abs() < 1e-14);   // 1*2*3
/// assert!((result[3] - 24.0).abs() < 1e-14);  // 1*2*3*4 = 4!
/// ```
///
/// # Use Cases
/// - Computing factorials and permutations
/// - Product of survival probabilities
/// - Chain rule in automatic differentiation
/// - Compound interest calculations
pub fn cumprod_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_cumprod(a);
    }

    // Scalar fallback for small arrays
    F::simd_cumprod(a)
}

/// SIMD-accelerated first-order difference
///
/// Computes the first-order finite difference: `result[i] = a[i+1] - a[i]`
/// The output has length n-1 for input of length n.
///
/// # Arguments
/// * `a` - Input array (length >= 2)
///
/// # Returns
/// Array of differences (length n-1)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::{array, Array1};
/// use scirs2_core::ndarray_ext::elementwise::diff_simd;
///
/// let x = array![1.0_f64, 3.0, 6.0, 10.0, 15.0];
///
/// let result = diff_simd::<f64>(&x.view());
/// assert_eq!(result.len(), 4);  // n-1 elements
/// assert!((result[0] - 2.0).abs() < 1e-14);  // 3-1
/// assert!((result[1] - 3.0).abs() < 1e-14);  // 6-3
/// assert!((result[2] - 4.0).abs() < 1e-14);  // 10-6
/// assert!((result[3] - 5.0).abs() < 1e-14);  // 15-10
/// ```
///
/// # Use Cases
/// - Numerical differentiation
/// - Detecting changes in time series
/// - Computing gradients
/// - Edge detection in signal processing
pub fn diff_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.len() < 2 {
        return Array1::zeros(0);
    }

    let optimizer = AutoOptimizer::new();
    if optimizer.should_use_simd(a.len()) {
        return F::simd_diff(a);
    }

    // Scalar fallback for small arrays
    F::simd_diff(a)
}

// =============================================================================
// Phase 70: Statistical Functions
// =============================================================================

/// SIMD-accelerated variance computation
///
/// Computes the sample variance: Var(x) = sum((x - mean)²) / (n-1)
/// Uses Bessel's correction for unbiased estimation.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Sample variance of the array
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::variance_simd;
///
/// let x = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let var = variance_simd::<f64>(&x.view());
/// // Sample variance: sum of squared deviations = 32, n = 8
/// // var = 32 / (8-1) = 32/7 ≈ 4.571
/// let expected = 32.0 / 7.0;
/// assert!((var - expected).abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Statistical analysis
/// - Quality control (process capability)
/// - Risk assessment (financial variance)
/// - Feature scaling (standardization)
pub fn variance_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }

    F::simd_variance(a)
}

/// SIMD-accelerated standard deviation computation
///
/// Computes the sample standard deviation: std = sqrt(sample_variance)
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Sample standard deviation
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::std_simd;
///
/// let x = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let s = std_simd::<f64>(&x.view());
/// // Sample std = sqrt(sample variance) = sqrt(32/7) ≈ 2.138
/// let expected = (32.0_f64 / 7.0).sqrt();
/// assert!((s - expected).abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Volatility measurement in finance
/// - Error analysis in experiments
/// - Gaussian distribution parameters
/// - Confidence interval calculations
pub fn std_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }

    F::simd_std(a)
}

/// SIMD-accelerated array sum
///
/// Computes the sum of all elements in the array.
/// Uses Kahan summation for improved numerical accuracy.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Sum of all elements
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sum_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let s = sum_simd::<f64>(&x.view());
/// assert!((s - 15.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Computing totals
/// - Mean calculation component
/// - Probability mass functions
/// - Integration approximation
pub fn sum_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }

    F::simd_sum(a)
}

/// SIMD-accelerated array mean
///
/// Computes the arithmetic mean: mean = sum(x) / n
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Arithmetic mean of elements
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::mean_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let m = mean_simd::<f64>(&x.view());
/// assert!((m - 3.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Central tendency measurement
/// - Batch normalization (compute running mean)
/// - Signal averaging
/// - Expected value estimation
pub fn mean_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }

    F::simd_mean(a)
}

/// SIMD-accelerated weighted sum
///
/// Computes sum(values * weights).
///
/// # Arguments
/// * `values` - Values array
/// * `weights` - Weights array (same length as values)
///
/// # Returns
/// Weighted sum
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::weighted_sum_simd;
///
/// let values = array![10.0_f64, 20.0, 30.0];
/// let weights = array![0.2_f64, 0.3, 0.5];
/// let ws = weighted_sum_simd::<f64>(&values.view(), &weights.view());
/// // 10*0.2 + 20*0.3 + 30*0.5 = 2 + 6 + 15 = 23
/// assert!((ws - 23.0).abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Portfolio valuation
/// - Weighted regression
/// - Attention mechanism scores
/// - Signal filtering
pub fn weighted_sum_simd<F>(values: &ArrayView1<F>, weights: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if values.is_empty() || weights.is_empty() {
        return F::zero();
    }

    F::simd_weighted_sum(values, weights)
}

/// SIMD-accelerated weighted mean
///
/// Computes sum(values * weights) / sum(weights).
///
/// # Arguments
/// * `values` - Values array
/// * `weights` - Weights array (same length as values)
///
/// # Returns
/// Weighted mean
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::weighted_mean_simd;
///
/// let values = array![10.0_f64, 20.0, 30.0];
/// let weights = array![1.0_f64, 2.0, 2.0];
/// let wm = weighted_mean_simd::<f64>(&values.view(), &weights.view());
/// // (10*1 + 20*2 + 30*2) / (1+2+2) = (10+40+60)/5 = 22
/// assert!((wm - 22.0).abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Grade point average calculation
/// - Portfolio weighted return
/// - Weighted sentiment analysis
/// - Attention-weighted representations
pub fn weighted_mean_simd<F>(values: &ArrayView1<F>, weights: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if values.is_empty() || weights.is_empty() {
        return F::zero();
    }

    F::simd_weighted_mean(values, weights)
}

// =============================================================================
// Phase 71: Reduction Functions
// =============================================================================

/// SIMD-accelerated maximum element
///
/// Finds the maximum value in the array.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Maximum element value
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::max_element_simd;
///
/// let x = array![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
/// let max = max_element_simd::<f64>(&x.view());
/// assert!((max - 9.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Finding peak values
/// - Range calculation
/// - Normalization (min-max scaling)
/// - Softmax numerator stability
pub fn max_element_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::neg_infinity();
    }

    F::simd_max_element(a)
}

/// SIMD-accelerated minimum element
///
/// Finds the minimum value in the array.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Minimum element value
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::min_element_simd;
///
/// let x = array![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
/// let min = min_element_simd::<f64>(&x.view());
/// assert!((min - 1.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Finding minimum values
/// - Range calculation
/// - Clipping lower bounds
/// - Outlier detection
pub fn min_element_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::infinity();
    }

    F::simd_min_element(a)
}

/// SIMD-accelerated argmax (index of maximum)
///
/// Returns the index of the maximum element.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Some(index) of maximum element, or None if array is empty
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::argmax_simd;
///
/// let x = array![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
/// let idx = argmax_simd::<f64>(&x.view());
/// assert_eq!(idx, Some(5));  // x[5] = 9.0 is the maximum
/// ```
///
/// # Use Cases
/// - Classification prediction (class with highest probability)
/// - Finding optimal parameters
/// - Greedy selection algorithms
/// - Winner-take-all networks
pub fn argmax_simd<F>(a: &ArrayView1<F>) -> Option<usize>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return None;
    }

    F::simd_argmax(a)
}

/// SIMD-accelerated argmin (index of minimum)
///
/// Returns the index of the minimum element.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Some(index) of minimum element, or None if array is empty
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::argmin_simd;
///
/// let x = array![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
/// let idx = argmin_simd::<f64>(&x.view());
/// assert_eq!(idx, Some(1));  // x[1] = 1.0 is the minimum (first occurrence)
/// ```
///
/// # Use Cases
/// - Finding nearest neighbors
/// - Minimum loss selection
/// - Optimal path finding
/// - Resource allocation
pub fn argmin_simd<F>(a: &ArrayView1<F>) -> Option<usize>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return None;
    }

    F::simd_argmin(a)
}

/// SIMD-accelerated sum of squares
///
/// Computes sum(x²) efficiently.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Sum of squared elements
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sum_squares_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0, 4.0];
/// let ss = sum_squares_simd::<f64>(&x.view());
/// // 1 + 4 + 9 + 16 = 30
/// assert!((ss - 30.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - L2 norm calculation (sqrt of sum of squares)
/// - Mean squared error
/// - Variance computation
/// - Energy of signal
pub fn sum_squares_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }

    F::simd_sum_squares(a)
}

/// SIMD-accelerated sum of cubes
///
/// Computes sum(x³) efficiently.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Sum of cubed elements
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sum_cubes_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0];
/// let sc = sum_cubes_simd::<f64>(&x.view());
/// // 1 + 8 + 27 = 36
/// assert!((sc - 36.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Higher-order moment calculation
/// - Skewness estimation
/// - Cubic interpolation
/// - Power sum computation
pub fn sum_cubes_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }

    F::simd_sum_cubes(a)
}

/// SIMD-accelerated log-sum-exp
///
/// Computes log(sum(exp(x))) in a numerically stable way.
/// Uses the identity: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Log of sum of exponentials
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::log_sum_exp_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0];
/// let lse = log_sum_exp_simd::<f64>(&x.view());
/// // log(e^1 + e^2 + e^3) ≈ 3.407
/// let expected = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();
/// assert!((lse - expected).abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Softmax denominator
/// - Log-probability computations
/// - Partition function in statistical mechanics
/// - Evidence lower bound (ELBO) in VAEs
pub fn log_sum_exp_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::neg_infinity();
    }

    F::simd_log_sum_exp(a)
}

// =============================================================================
// Phase 72: Norm and Distance Functions
// =============================================================================

/// SIMD-accelerated L2 norm (Euclidean norm)
///
/// Computes ||x||₂ = sqrt(sum(x²))
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// L2 (Euclidean) norm
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::norm_simd;
///
/// let x = array![3.0_f64, 4.0];
/// let n = norm_simd::<f64>(&x.view());
/// // sqrt(9 + 16) = 5
/// assert!((n - 5.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Vector magnitude
/// - Normalization preprocessing
/// - Regularization penalty
/// - Gradient clipping
pub fn norm_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }

    F::simd_norm(a)
}

/// SIMD-accelerated L1 norm (Manhattan norm)
///
/// Computes ||x||₁ = sum(|x|)
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// L1 (Manhattan) norm
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::norm_l1_simd;
///
/// let x = array![1.0_f64, -2.0, 3.0, -4.0];
/// let n = norm_l1_simd::<f64>(&x.view());
/// // |1| + |-2| + |3| + |-4| = 10
/// assert!((n - 10.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Sparse regularization (LASSO)
/// - Robust statistics
/// - Compressed sensing
/// - Feature selection
pub fn norm_l1_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }

    F::simd_norm_l1(a)
}

/// SIMD-accelerated L-infinity norm (Chebyshev/max norm)
///
/// Computes ||x||∞ = max(|x|)
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// L-infinity (max) norm
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::norm_linf_simd;
///
/// let x = array![1.0_f64, -5.0, 3.0, -2.0];
/// let n = norm_linf_simd::<f64>(&x.view());
/// // max(|1|, |-5|, |3|, |-2|) = 5
/// assert!((n - 5.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Gradient clipping by max norm
/// - Adversarial robustness (L∞ perturbations)
/// - Convergence criteria
/// - Worst-case analysis
pub fn norm_linf_simd<F>(a: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return F::zero();
    }

    F::simd_norm_linf(a)
}

/// SIMD-accelerated Euclidean distance
///
/// Computes ||a - b||₂ = sqrt(sum((a - b)²))
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Euclidean distance between vectors
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::distance_euclidean_simd;
///
/// let a = array![0.0_f64, 0.0];
/// let b = array![3.0_f64, 4.0];
/// let d = distance_euclidean_simd::<f64>(&a.view(), &b.view());
/// assert!((d - 5.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - K-means clustering
/// - Nearest neighbor search
/// - Physical distance measurement
/// - Embedding similarity
pub fn distance_euclidean_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return F::zero();
    }

    F::simd_distance_euclidean(a, b)
}

/// SIMD-accelerated Manhattan distance
///
/// Computes sum(|a - b|)
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Manhattan (L1) distance between vectors
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::distance_manhattan_simd;
///
/// let a = array![1.0_f64, 2.0, 3.0];
/// let b = array![4.0_f64, 0.0, 3.0];
/// let d = distance_manhattan_simd::<f64>(&a.view(), &b.view());
/// // |1-4| + |2-0| + |3-3| = 3 + 2 + 0 = 5
/// assert!((d - 5.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Grid-based path finding
/// - Robust distance metric
/// - Taxicab geometry
/// - Feature comparison
pub fn distance_manhattan_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return F::zero();
    }

    F::simd_distance_manhattan(a, b)
}

/// SIMD-accelerated Chebyshev distance
///
/// Computes max(|a - b|)
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Chebyshev (L∞) distance between vectors
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::distance_chebyshev_simd;
///
/// let a = array![1.0_f64, 2.0, 3.0];
/// let b = array![4.0_f64, 0.0, 3.0];
/// let d = distance_chebyshev_simd::<f64>(&a.view(), &b.view());
/// // max(|1-4|, |2-0|, |3-3|) = max(3, 2, 0) = 3
/// assert!((d - 3.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Chess king distance
/// - Maximum deviation
/// - Robust outlier detection
/// - Image processing
pub fn distance_chebyshev_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return F::zero();
    }

    F::simd_distance_chebyshev(a, b)
}

/// SIMD-accelerated cosine distance
///
/// Computes 1 - cosine_similarity(a, b)
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Cosine distance (0 = identical direction, 2 = opposite direction)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::distance_cosine_simd;
///
/// let a = array![1.0_f64, 0.0, 0.0];
/// let b = array![0.0_f64, 1.0, 0.0];
/// let d = distance_cosine_simd::<f64>(&a.view(), &b.view());
/// // Orthogonal vectors have cosine similarity 0, distance 1
/// assert!((d - 1.0).abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Text similarity (TF-IDF vectors)
/// - Recommendation systems
/// - Document clustering
/// - Image retrieval
pub fn distance_cosine_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return F::one();
    }

    F::simd_distance_cosine(a, b)
}

/// SIMD-accelerated cosine similarity
///
/// Computes (a · b) / (||a|| * ||b||)
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Cosine similarity (-1 to 1, where 1 = identical direction)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::cosine_similarity_simd;
///
/// let a = array![1.0_f64, 2.0, 3.0];
/// let b = array![2.0_f64, 4.0, 6.0];
/// let sim = cosine_similarity_simd::<f64>(&a.view(), &b.view());
/// // Parallel vectors have cosine similarity 1
/// assert!((sim - 1.0).abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Word embedding similarity
/// - Document comparison
/// - Recommendation scoring
/// - Semantic search
pub fn cosine_similarity_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return F::zero();
    }

    F::simd_cosine_similarity(a, b)
}

// =============================================================================
// Phase 73: Binary Operations
// =============================================================================

/// SIMD-accelerated element-wise addition
///
/// Computes a + b element-wise.
///
/// # Arguments
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
/// Element-wise sum
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::add_simd;
///
/// let a = array![1.0_f64, 2.0, 3.0];
/// let b = array![4.0_f64, 5.0, 6.0];
/// let c = add_simd::<f64>(&a.view(), &b.view());
/// assert!((c[0] - 5.0).abs() < 1e-14);
/// assert!((c[1] - 7.0).abs() < 1e-14);
/// assert!((c[2] - 9.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Vector arithmetic
/// - Residual connections
/// - Bias addition
/// - Signal combination
pub fn add_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }

    F::simd_add(a, b)
}

/// SIMD-accelerated element-wise subtraction
///
/// Computes a - b element-wise.
///
/// # Arguments
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
/// Element-wise difference
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::sub_simd;
///
/// let a = array![5.0_f64, 7.0, 9.0];
/// let b = array![1.0_f64, 2.0, 3.0];
/// let c = sub_simd::<f64>(&a.view(), &b.view());
/// assert!((c[0] - 4.0).abs() < 1e-14);
/// assert!((c[1] - 5.0).abs() < 1e-14);
/// assert!((c[2] - 6.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Gradient computation
/// - Error calculation
/// - Differencing signals
/// - Relative positioning
pub fn sub_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }

    F::simd_sub(a, b)
}

/// SIMD-accelerated element-wise multiplication
///
/// Computes a * b element-wise (Hadamard product).
///
/// # Arguments
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
/// Element-wise product
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::mul_simd;
///
/// let a = array![2.0_f64, 3.0, 4.0];
/// let b = array![5.0_f64, 6.0, 7.0];
/// let c = mul_simd::<f64>(&a.view(), &b.view());
/// assert!((c[0] - 10.0).abs() < 1e-14);
/// assert!((c[1] - 18.0).abs() < 1e-14);
/// assert!((c[2] - 28.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Attention weights application
/// - Feature gating
/// - Gradient scaling
/// - Signal modulation
pub fn mul_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }

    F::simd_mul(a, b)
}

/// SIMD-accelerated element-wise division
///
/// Computes a / b element-wise.
///
/// # Arguments
/// * `a` - Numerator array
/// * `b` - Denominator array
///
/// # Returns
/// Element-wise quotient
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::div_simd;
///
/// let a = array![10.0_f64, 20.0, 30.0];
/// let b = array![2.0_f64, 4.0, 5.0];
/// let c = div_simd::<f64>(&a.view(), &b.view());
/// assert!((c[0] - 5.0).abs() < 1e-14);
/// assert!((c[1] - 5.0).abs() < 1e-14);
/// assert!((c[2] - 6.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Normalization
/// - Ratio computation
/// - Scaling by variable factors
/// - Probability normalization
pub fn div_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }

    F::simd_div(a, b)
}

/// SIMD-accelerated element-wise maximum
///
/// Computes max(a, b) element-wise.
///
/// # Arguments
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
/// Element-wise maximum
///
/// # Examples
/// ```ignore
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::max_simd;
///
/// let a = array![1.0_f64, 5.0, 3.0];
/// let b = array![4.0_f64, 2.0, 6.0];
/// let c = max_simd::<f64>(&a.view(), &b.view());
/// assert!((c[0] - 4.0_f64).abs() < 1e-14);
/// assert!((c[1] - 5.0_f64).abs() < 1e-14);
/// assert!((c[2] - 6.0_f64).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - ReLU activation: max(0, x)
/// - Soft clipping
/// - Envelope detection
/// - Upper bound enforcement
pub fn max_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }

    F::simd_max(a, b)
}

/// SIMD-accelerated element-wise minimum
///
/// Computes min(a, b) element-wise.
///
/// # Arguments
/// * `a` - First array
/// * `b` - Second array
///
/// # Returns
/// Element-wise minimum
///
/// # Examples
/// ```ignore
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::min_simd;
///
/// let a = array![1.0_f64, 5.0, 3.0];
/// let b = array![4.0_f64, 2.0, 6.0];
/// let c = min_simd::<f64>(&a.view(), &b.view());
/// assert!((c[0] - 1.0_f64).abs() < 1e-14);
/// assert!((c[1] - 2.0_f64).abs() < 1e-14);
/// assert!((c[2] - 3.0_f64).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Gradient clipping
/// - Soft clipping
/// - Lower bound enforcement
/// - Minimum filter in signal processing
pub fn min_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return Array1::zeros(0);
    }

    F::simd_min(a, b)
}

/// SIMD-accelerated scalar multiplication
///
/// Computes a * scalar for all elements.
///
/// # Arguments
/// * `a` - Input array
/// * `scalar` - Scalar multiplier
///
/// # Returns
/// Scaled array
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::scalar_mul_simd;
///
/// let a = array![1.0_f64, 2.0, 3.0];
/// let c = scalar_mul_simd::<f64>(&a.view(), 2.5);
/// assert!((c[0] - 2.5).abs() < 1e-14);
/// assert!((c[1] - 5.0).abs() < 1e-14);
/// assert!((c[2] - 7.5).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Learning rate scaling
/// - Normalization
/// - Unit conversion
/// - Signal amplification
pub fn scalar_mul_simd<F>(a: &ArrayView1<F>, scalar: F) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }

    F::simd_scalar_mul(a, scalar)
}

/// SIMD-accelerated fused multiply-add
///
/// Computes a * b + c element-wise in a single operation.
/// More accurate and efficient than separate multiply and add.
///
/// # Arguments
/// * `a` - First multiplicand
/// * `b` - Second multiplicand
/// * `c` - Addend
///
/// # Returns
/// Element-wise fused multiply-add result
///
/// # Examples
/// ```ignore
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::fma_simd;
///
/// let a = array![1.0_f64, 2.0, 3.0];
/// let b = array![4.0_f64, 5.0, 6.0];
/// let c = array![7.0_f64, 8.0, 9.0];
/// let result = fma_simd::<f64>(&a.view(), &b.view(), &c.view());
/// // 1*4+7=11, 2*5+8=18, 3*6+9=27
/// assert!((result[0] - 11.0_f64).abs() < 1e-14);
/// assert!((result[1] - 18.0_f64).abs() < 1e-14);
/// assert!((result[2] - 27.0_f64).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Matrix multiplication inner loop
/// - Polynomial evaluation (Horner's method)
/// - Linear combinations
/// - Neural network forward pass
pub fn fma_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>, c: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() || c.is_empty() {
        return Array1::zeros(0);
    }

    F::simd_fma(a, b, c)
}

/// SIMD-accelerated dot product
///
/// Computes sum(a * b).
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
/// Dot product (scalar)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::dot_simd;
///
/// let a = array![1.0_f64, 2.0, 3.0];
/// let b = array![4.0_f64, 5.0, 6.0];
/// let d = dot_simd::<f64>(&a.view(), &b.view());
/// // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
/// assert!((d - 32.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Projection calculations
/// - Cosine similarity numerator
/// - Attention scores
/// - Linear layer computation
pub fn dot_simd<F>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> F
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() || b.is_empty() {
        return F::zero();
    }

    F::simd_dot(a, b)
}

// =============================================================================
// Phase 74: Normalization and Activation Functions
// =============================================================================

/// SIMD-accelerated ReLU (Rectified Linear Unit)
///
/// Computes max(0, x) element-wise.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// ReLU-activated array
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::relu_simd;
///
/// let x = array![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
/// let result = relu_simd::<f64>(&x.view());
/// assert!((result[0] - 0.0).abs() < 1e-14);
/// assert!((result[1] - 0.0).abs() < 1e-14);
/// assert!((result[2] - 0.0).abs() < 1e-14);
/// assert!((result[3] - 1.0).abs() < 1e-14);
/// assert!((result[4] - 2.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Neural network activation
/// - Sparse representations
/// - Thresholding signals
/// - Feature rectification
pub fn relu_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }

    F::simd_relu(a)
}

/// SIMD-accelerated L2 normalization
///
/// Normalizes the vector to unit length: x / ||x||₂
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Unit-normalized array (or zero if input is zero)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::normalize_simd;
///
/// let x = array![3.0_f64, 4.0];
/// let result = normalize_simd::<f64>(&x.view());
/// // ||result|| = 1
/// let norm = (result[0]*result[0] + result[1]*result[1]).sqrt();
/// assert!((norm - 1.0).abs() < 1e-10);
/// // x/5 = [0.6, 0.8]
/// assert!((result[0] - 0.6).abs() < 1e-10);
/// assert!((result[1] - 0.8).abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Unit vector computation
/// - Cosine similarity preparation
/// - Gradient normalization
/// - Direction extraction
pub fn normalize_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }

    F::simd_normalize(a)
}

/// SIMD-accelerated standardization (z-score normalization)
///
/// Transforms to zero mean and unit variance: (x - mean) / std
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Standardized array
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::standardize_simd;
///
/// let x = array![2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let result = standardize_simd::<f64>(&x.view());
/// // Mean should be ~0, std should be ~1
/// let mean: f64 = result.iter().sum::<f64>() / result.len() as f64;
/// assert!(mean.abs() < 1e-10);
/// ```
///
/// # Use Cases
/// - Feature scaling for ML
/// - Batch normalization
/// - Statistical preprocessing
/// - Anomaly scoring
pub fn standardize_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }

    F::simd_standardize(a)
}

/// SIMD-accelerated softmax
///
/// Computes exp(x - max(x)) / sum(exp(x - max(x))) for numerical stability.
/// Output probabilities sum to 1.
///
/// # Arguments
/// * `a` - Input array (logits)
///
/// # Returns
/// Probability distribution (sums to 1)
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::softmax_simd;
///
/// let x = array![1.0_f64, 2.0, 3.0];
/// let result = softmax_simd::<f64>(&x.view());
/// // Probabilities should sum to 1
/// let sum: f64 = result.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-10);
/// // Higher logits should have higher probabilities
/// assert!(result[2] > result[1]);
/// assert!(result[1] > result[0]);
/// ```
///
/// # Use Cases
/// - Classification output layer
/// - Attention weights
/// - Policy probabilities in RL
/// - Mixture model weights
pub fn softmax_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }

    F::simd_softmax(a)
}

/// SIMD-accelerated truncation (round towards zero)
///
/// Removes the fractional part, rounding towards zero.
///
/// # Arguments
/// * `a` - Input array
///
/// # Returns
/// Truncated array
///
/// # Examples
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_core::ndarray_ext::elementwise::trunc_simd;
///
/// let x = array![2.7_f64, -2.7, 0.9, -0.9];
/// let result = trunc_simd::<f64>(&x.view());
/// assert!((result[0] - 2.0).abs() < 1e-14);
/// assert!((result[1] - (-2.0)).abs() < 1e-14);
/// assert!((result[2] - 0.0).abs() < 1e-14);
/// assert!((result[3] - 0.0).abs() < 1e-14);
/// ```
///
/// # Use Cases
/// - Integer conversion
/// - Discretization
/// - Quantization
/// - Index calculation
pub fn trunc_simd<F>(a: &ArrayView1<F>) -> Array1<F>
where
    F: Float + SimdUnifiedOps,
{
    if a.is_empty() {
        return Array1::zeros(0);
    }

    F::simd_trunc(a)
}

#[cfg(test)]
#[path = "elementwise_tests.rs"]
mod tests;
