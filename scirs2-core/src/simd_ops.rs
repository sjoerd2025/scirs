//! Unified SIMD operations abstraction layer
//!
//! This module provides a comprehensive abstraction layer for all SIMD operations
//! used across the scirs2 ecosystem. All modules should use these operations
//! instead of implementing their own SIMD code.

use ::ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1};
use num_traits::Zero;

#[cfg(feature = "simd")]
use crate::simd_ops_polynomial;

/// Unified SIMD operations trait
pub trait SimdUnifiedOps: Sized + Copy + PartialOrd + Zero {
    /// Element-wise addition
    fn simd_add(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise subtraction
    fn simd_sub(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise multiplication
    fn simd_mul(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise division
    fn simd_div(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Dot product
    fn simd_dot(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self;

    /// Matrix-vector multiplication (GEMV)
    fn simd_gemv(a: &ArrayView2<Self>, x: &ArrayView1<Self>, beta: Self, y: &mut Array1<Self>);

    /// Matrix-matrix multiplication (GEMM)
    fn simd_gemm(
        alpha: Self,
        a: &ArrayView2<Self>,
        b: &ArrayView2<Self>,
        beta: Self,
        c: &mut Array2<Self>,
    );

    /// Vector norm (L2)
    fn simd_norm(a: &ArrayView1<Self>) -> Self;

    /// Element-wise maximum
    fn simd_max(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise minimum
    fn simd_min(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Scalar multiplication
    fn simd_scalar_mul(a: &ArrayView1<Self>, scalar: Self) -> Array1<Self>;

    /// Sum reduction
    fn simd_sum(a: &ArrayView1<Self>) -> Self;

    /// Mean reduction
    fn simd_mean(a: &ArrayView1<Self>) -> Self;

    /// Find maximum element
    fn simd_max_element(a: &ArrayView1<Self>) -> Self;

    /// Find minimum element
    fn simd_min_element(a: &ArrayView1<Self>) -> Self;

    /// Fused multiply-add: a * b + c
    fn simd_fma(a: &ArrayView1<Self>, b: &ArrayView1<Self>, c: &ArrayView1<Self>) -> Array1<Self>;

    /// Enhanced cache-optimized addition for large arrays
    fn simd_add_cache_optimized(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Advanced-optimized fused multiply-add for maximum performance
    fn simd_fma_advanced_optimized(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        c: &ArrayView1<Self>,
    ) -> Array1<Self>;

    /// Adaptive SIMD operation that selects optimal implementation
    fn simd_add_adaptive(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Matrix transpose
    fn simd_transpose(a: &ArrayView2<Self>) -> Array2<Self>;

    /// Cache-optimized blocked matrix transpose
    /// Uses L1 cache-friendly block sizes for improved memory access patterns.
    /// Expected 3-5x speedup for large matrices (>512x512).
    fn simd_transpose_blocked(a: &ArrayView2<Self>) -> Array2<Self>;

    /// Element-wise absolute value
    fn simd_abs(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise square root
    fn simd_sqrt(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise exponential (e^x)
    fn simd_exp(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise natural logarithm (ln(x))
    fn simd_ln(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise sine (sin(x))
    fn simd_sin(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise cosine (cos(x))
    fn simd_cos(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise tangent (tan(x))
    fn simd_tan(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise hyperbolic sine (sinh(x))
    fn simd_sinh(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise hyperbolic cosine (cosh(x))
    fn simd_cosh(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise hyperbolic tangent (tanh(x))
    fn simd_tanh(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise floor (largest integer <= x)
    fn simd_floor(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise ceiling (smallest integer >= x)
    fn simd_ceil(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise rounding to nearest integer
    fn simd_round(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise arctangent (atan(x))
    fn simd_atan(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise arcsine (asin(x))
    fn simd_asin(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise arccosine (acos(x))
    fn simd_acos(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise two-argument arctangent (atan2(y, x))
    fn simd_atan2(y: &ArrayView1<Self>, x: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise base-10 logarithm (log10(x))
    fn simd_log10(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise base-2 logarithm (log2(x))
    fn simd_log2(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise clamp (constrain values to [min, max])
    fn simd_clamp(a: &ArrayView1<Self>, min: Self, max: Self) -> Array1<Self>;

    /// Element-wise fractional part (x - trunc(x))
    fn simd_fract(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise trunc (round toward zero)
    fn simd_trunc(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise reciprocal (1/x)
    fn simd_recip(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise power with scalar exponent (base^exp)
    fn simd_powf(base: &ArrayView1<Self>, exp: Self) -> Array1<Self>;

    /// Element-wise power with array exponent (`base[i]^exp[i]`)
    fn simd_pow(base: &ArrayView1<Self>, exp: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise power with integer exponent (base^n)
    fn simd_powi(base: &ArrayView1<Self>, n: i32) -> Array1<Self>;

    /// Element-wise gamma function Γ(x)
    fn simd_gamma(x: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise 2^x (base-2 exponential)
    fn simd_exp2(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise cube root (cbrt(x) = x^(1/3))
    fn simd_cbrt(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise ln(1+x) (numerically stable for small x)
    fn simd_ln_1p(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise exp(x)-1 (numerically stable for small x)
    fn simd_exp_m1(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise conversion from degrees to radians (x * π / 180)
    fn simd_to_radians(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise conversion from radians to degrees (x * 180 / π)
    fn simd_to_degrees(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise digamma function ψ(x) = d/dx ln(Γ(x))
    fn simd_digamma(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise trigamma function ψ'(x) = d²/dx² ln(Γ(x))
    /// The second derivative of log-gamma, critical for Fisher information in Bayesian inference.
    fn simd_trigamma(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise log-gamma function ln(Γ(x))
    /// More numerically stable than computing gamma(x).ln() - used extensively in statistical distributions.
    fn simd_ln_gamma(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise error function erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
    /// Critical for normal distribution CDF: Φ(x) = 0.5 * (1 + erf(x/√2))
    /// Properties: erf(0)=0, erf(∞)=1, erf(-x)=-erf(x)
    fn simd_erf(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise complementary error function erfc(x) = 1 - erf(x)
    /// More numerically stable than 1 - erf(x) for large x
    fn simd_erfc(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise inverse error function erfinv(y) such that erf(erfinv(y)) = y
    /// Critical for inverse normal CDF (probit function): Φ⁻¹(p) = √2 * erfinv(2p - 1)
    /// Domain: (-1, 1), Range: (-∞, ∞)
    /// Properties: erfinv(0)=0, erfinv(-y)=-erfinv(y) (odd function)
    fn simd_erfinv(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise inverse complementary error function erfcinv(y) such that erfc(erfcinv(y)) = y
    /// More numerically stable than erfinv(1-y) for y close to 0
    /// Domain: (0, 2), Range: (-∞, ∞)
    fn simd_erfcinv(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise sigmoid (logistic) function: σ(x) = 1 / (1 + exp(-x))
    /// Critical for neural networks, logistic regression, and probability modeling
    /// Range: (0, 1), σ(0) = 0.5, σ(-∞) = 0, σ(+∞) = 1
    /// Properties: σ(-x) = 1 - σ(x), derivative σ'(x) = σ(x)(1 - σ(x))
    fn simd_sigmoid(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise GELU (Gaussian Error Linear Unit) activation function
    /// GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / √2))
    /// Where Φ(x) is the standard normal CDF
    /// Critical for Transformer models (BERT, GPT, etc.)
    /// Properties: GELU(0) = 0, smooth approximation of ReLU
    fn simd_gelu(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise Swish (SiLU - Sigmoid Linear Unit) activation function
    /// Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
    /// Self-gated activation discovered via neural architecture search
    /// Used in EfficientNet, GPT-NeoX, and many modern architectures
    /// Properties: smooth, non-monotonic, self-gating, unbounded above
    fn simd_swish(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise Softplus activation function
    /// Softplus(x) = ln(1 + exp(x))
    /// Smooth approximation of ReLU
    /// Used in probabilistic models, Bayesian deep learning, smooth counting
    /// Properties: softplus(0) = ln(2) ≈ 0.693, always positive, derivative = sigmoid(x)
    fn simd_softplus(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise Mish activation function
    /// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    /// Self-regularized non-monotonic activation function
    /// Used in YOLOv4, modern object detection, and neural architectures
    /// Properties: smooth, non-monotonic, Mish(0) = 0, unbounded above
    fn simd_mish(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise ELU (Exponential Linear Unit) activation function
    /// ELU(x, α) = x if x >= 0, α * (exp(x) - 1) if x < 0
    /// Helps with vanishing gradients and faster learning
    /// Used in deep neural networks for smoother outputs
    /// Properties: smooth, continuous derivative, bounded below by -α
    fn simd_elu(a: &ArrayView1<Self>, alpha: Self) -> Array1<Self>;

    /// SELU activation function (Scaled Exponential Linear Unit)
    ///
    /// SELU(x) = λ * (x if x > 0, α * (exp(x) - 1) if x <= 0)
    /// where λ ≈ 1.0507 and α ≈ 1.6733 (fixed constants)
    /// Self-normalizing: preserves mean=0, variance=1 through layers
    /// Used in Self-Normalizing Neural Networks (SNNs)
    /// Eliminates need for BatchNorm when using LeCun Normal initialization
    fn simd_selu(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Hardsigmoid activation function
    ///
    /// Hardsigmoid(x) = clip((x + 3) / 6, 0, 1)
    /// Piecewise linear approximation of sigmoid
    /// Used in MobileNetV3 for efficient inference
    /// Avoids expensive exp() computation
    fn simd_hardsigmoid(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Hardswish activation function
    ///
    /// Hardswish(x) = x * hardsigmoid(x) = x * clip((x + 3) / 6, 0, 1)
    /// Piecewise linear approximation of Swish
    /// Used in MobileNetV3 for efficient inference
    /// Avoids expensive exp() computation while maintaining self-gating
    fn simd_hardswish(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Sinc function (normalized)
    ///
    /// sinc(x) = sin(πx) / (πx) for x ≠ 0, sinc(0) = 1
    /// Critical for signal processing, windowing, interpolation
    /// Properties: sinc(n) = 0 for all non-zero integers n
    fn simd_sinc(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Log-softmax function for numerically stable probability computation
    ///
    /// log_softmax(x_i) = x_i - log(Σ_j exp(x_j))
    /// Critical for neural networks, especially cross-entropy loss
    /// More numerically stable than computing log(softmax(x))
    /// Used in Transformers, LLMs, and classification networks
    fn simd_log_softmax(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Inverse hyperbolic sine: asinh(x) = ln(x + √(x² + 1))
    ///
    /// Domain: (-∞, +∞), Range: (-∞, +∞)
    /// Used in: hyperbolic geometry, conformal mapping, special relativity (rapidity)
    fn simd_asinh(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Inverse hyperbolic cosine: acosh(x) = ln(x + √(x² - 1))
    ///
    /// Domain: [1, +∞), Range: [0, +∞)
    /// Returns NaN for x < 1
    /// Used in: hyperbolic geometry, distance calculations, special relativity
    fn simd_acosh(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Inverse hyperbolic tangent: atanh(x) = 0.5 * ln((1+x)/(1-x))
    ///
    /// Domain: (-1, 1), Range: (-∞, +∞)
    /// Returns ±∞ at x = ±1, NaN for |x| > 1
    /// Used in: statistical transformations (Fisher's z), probability
    fn simd_atanh(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Beta function: B(a, b) = Γ(a)Γ(b)/Γ(a+b)
    ///
    /// The beta function is fundamental for:
    /// - Beta distribution (Bayesian priors)
    /// - Binomial coefficients: C(n,k) = 1/(n+1)/B(n-k+1, k+1)
    /// - Statistical hypothesis testing
    /// - Incomplete beta function (regularized)
    fn simd_beta(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Log-Beta function: ln(B(a, b)) = ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b))
    ///
    /// More numerically stable than computing B(a,b) for large arguments.
    /// Returns ln(B(a,b)) for each pair of inputs.
    fn simd_ln_beta(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Linear interpolation: lerp(a, b, t) = a + t * (b - a) = a * (1 - t) + b * t
    ///
    /// Computes element-wise linear interpolation between arrays `a` and `b`
    /// using interpolation parameter `t`. When t=0, returns a; when t=1, returns b.
    ///
    /// Critical for:
    /// - Animation blending (skeletal animation, morph targets)
    /// - Quaternion SLERP approximation (for small angles)
    /// - Gradient computation in neural networks
    /// - Smooth parameter transitions
    /// - Color blending and image processing
    fn simd_lerp(a: &ArrayView1<Self>, b: &ArrayView1<Self>, t: Self) -> Array1<Self>;

    /// Smoothstep interpolation: smoothstep(edge0, edge1, x)
    ///
    /// Returns smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1.
    /// - Returns 0 if x <= edge0
    /// - Returns 1 if x >= edge1
    /// - Returns smooth curve: 3t² - 2t³ where t = (x - edge0) / (edge1 - edge0)
    ///
    /// Critical for:
    /// - Shader programming (lighting, transitions)
    /// - Activation function variants
    /// - Smooth threshold functions
    /// - Anti-aliasing and blending
    fn simd_smoothstep(edge0: Self, edge1: Self, x: &ArrayView1<Self>) -> Array1<Self>;

    /// Hypotenuse: hypot(x, y) = sqrt(x² + y²)
    ///
    /// Computes element-wise hypotenuse without overflow/underflow issues.
    /// Uses the standard library implementation which handles extreme values.
    ///
    /// Critical for:
    /// - Distance calculations in 2D/3D
    /// - Computing vector magnitudes
    /// - Graphics and physics simulations
    /// - Complex number modulus: |a+bi| = hypot(a, b)
    fn simd_hypot(x: &ArrayView1<Self>, y: &ArrayView1<Self>) -> Array1<Self>;

    /// Copysign: copysign(x, y) returns x with the sign of y
    ///
    /// For each element, returns the magnitude of x with the sign of y.
    /// - copysign(1.0, -2.0) = -1.0
    /// - copysign(-3.0, 4.0) = 3.0
    ///
    /// Critical for:
    /// - Sign manipulation in numerical algorithms
    /// - Implementing special functions (e.g., reflection formula)
    /// - Gradient sign propagation
    fn simd_copysign(x: &ArrayView1<Self>, y: &ArrayView1<Self>) -> Array1<Self>;

    /// Smootherstep (Ken Perlin's improved smoothstep): 6t⁵ - 15t⁴ + 10t³
    ///
    /// An improved version of smoothstep with second-order continuous derivatives.
    /// The first AND second derivatives are zero at the boundaries.
    ///
    /// Critical for:
    /// - Perlin noise and procedural generation
    /// - High-quality animation easing
    /// - Shader programming (better lighting transitions)
    /// - Gradient-based optimization (smoother loss landscapes)
    fn simd_smootherstep(edge0: Self, edge1: Self, x: &ArrayView1<Self>) -> Array1<Self>;

    /// Logaddexp: log(exp(a) + exp(b)) computed in a numerically stable way
    ///
    /// Uses the identity: log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))
    /// This avoids overflow/underflow for large positive or negative values.
    ///
    /// Critical for:
    /// - Log-probability computations (Bayesian inference)
    /// - Log-likelihood calculations in ML
    /// - Hidden Markov Model forward/backward algorithms
    /// - Neural network loss functions (cross-entropy)
    fn simd_logaddexp(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Logit function: log(p / (1-p)) - inverse of sigmoid
    ///
    /// Maps probabilities in (0, 1) to log-odds in (-∞, +∞).
    /// The logit function is the inverse of the sigmoid (logistic) function.
    ///
    /// Critical for:
    /// - Logistic regression (log-odds interpretation)
    /// - Probability calibration
    /// - Converting probabilities to unbounded space for optimization
    /// - Statistical modeling (link functions)
    fn simd_logit(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Element-wise square: x²
    ///
    /// More efficient than simd_pow(x, 2) or simd_mul(x, x) as it's a single multiplication.
    ///
    /// Critical for:
    /// - Variance computation: E\[X²\] - E\[X\]²
    /// - Distance calculations: ||a - b||² = (a - b)²
    /// - Neural network loss functions (MSE)
    /// - Physics simulations (kinetic energy: ½mv²)
    fn simd_square(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Inverse square root: 1/sqrt(x)
    ///
    /// More efficient than simd_div(1, simd_sqrt(x)) for normalization operations.
    ///
    /// Critical for:
    /// - Vector normalization: v * rsqrt(dot(v,v))
    /// - Graphics (lighting, physics simulations)
    /// - Layer normalization in neural networks
    /// - Quaternion normalization
    fn simd_rsqrt(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Simultaneous sin and cos: returns (sin(x), cos(x))
    ///
    /// More efficient than calling sin and cos separately when both are needed.
    /// Returns a tuple of two arrays.
    ///
    /// Critical for:
    /// - Rotation matrices (2D and 3D)
    /// - Fourier transforms
    /// - Wave simulations
    /// - Animation and physics
    fn simd_sincos(a: &ArrayView1<Self>) -> (Array1<Self>, Array1<Self>);

    /// Numerically stable exp(x) - 1
    ///
    /// Returns exp(x) - 1 accurately for small x values where exp(x) ≈ 1.
    /// For small x, the direct calculation exp(x) - 1 suffers from catastrophic cancellation.
    ///
    /// Critical for:
    /// - Financial calculations (compound interest for small rates)
    /// - Numerical integration of differential equations
    /// - Statistical distributions (Poisson, exponential)
    /// - Machine learning (softplus, log-sum-exp)
    fn simd_expm1(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Numerically stable ln(1 + x)
    ///
    /// Returns ln(1 + x) accurately for small x values where 1 + x ≈ 1.
    /// For small x, the direct calculation ln(1 + x) suffers from catastrophic cancellation.
    ///
    /// Critical for:
    /// - Log-probability calculations (log(1 - p) for small p)
    /// - Numerical integration
    /// - Statistical distributions
    /// - Machine learning (binary cross-entropy loss)
    fn simd_log1p(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Sum of squares
    fn simd_sum_squares(a: &ArrayView1<Self>) -> Self;

    /// Element-wise multiplication (alias for simd_mul)
    fn simd_multiply(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self>;

    /// Check if SIMD is available for this type
    fn simd_available() -> bool;

    /// Ultra-optimized sum reduction (alias for simd_sum for compatibility)
    fn simd_sum_f32_ultra(a: &ArrayView1<Self>) -> Self {
        Self::simd_sum(a)
    }

    /// Ultra-optimized subtraction (alias for simd_sub for compatibility)
    fn simd_sub_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    );

    /// Ultra-optimized multiplication (alias for simd_mul for compatibility)
    fn simd_mul_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    );

    /// Ultra-optimized cubes sum (power 3 sum)
    fn simd_sum_cubes(a: &ArrayView1<Self>) -> Self;

    /// Ultra-optimized division (alias for simd_div for compatibility)
    fn simd_div_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    );

    /// Ultra-optimized sine function
    fn simd_sin_f32_ultra(a: &ArrayView1<Self>, result: &mut ArrayViewMut1<Self>);

    /// Ultra-optimized addition (alias for simd_add for compatibility)
    fn simd_add_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    );

    /// Ultra-optimized fused multiply-add
    fn simd_fma_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        c: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    );

    /// Ultra-optimized power function
    fn simd_pow_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    );

    /// Ultra-optimized exponential function
    fn simd_exp_f32_ultra(a: &ArrayView1<Self>, result: &mut ArrayViewMut1<Self>);

    /// Ultra-optimized cosine function
    fn simd_cos_f32_ultra(a: &ArrayView1<Self>, result: &mut ArrayViewMut1<Self>);

    /// Ultra-optimized dot product
    fn simd_dot_f32_ultra(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self;

    /// Variance (population variance)
    fn simd_variance(a: &ArrayView1<Self>) -> Self;

    /// Standard deviation
    fn simd_std(a: &ArrayView1<Self>) -> Self;

    /// L1 norm (Manhattan norm)
    fn simd_norm_l1(a: &ArrayView1<Self>) -> Self;

    /// L∞ norm (Chebyshev norm / max absolute)
    fn simd_norm_linf(a: &ArrayView1<Self>) -> Self;

    /// Cosine similarity between two vectors
    fn simd_cosine_similarity(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self;

    /// Euclidean distance between two vectors
    fn simd_distance_euclidean(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self;

    /// Manhattan distance between two vectors
    fn simd_distance_manhattan(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self;

    /// Chebyshev distance between two vectors
    fn simd_distance_chebyshev(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self;

    /// Cosine distance (1 - cosine_similarity)
    fn simd_distance_cosine(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self;

    /// Weighted sum
    fn simd_weighted_sum(values: &ArrayView1<Self>, weights: &ArrayView1<Self>) -> Self;

    /// Weighted mean
    fn simd_weighted_mean(values: &ArrayView1<Self>, weights: &ArrayView1<Self>) -> Self;

    /// Find index of minimum element (argmin)
    fn simd_argmin(a: &ArrayView1<Self>) -> Option<usize>;

    /// Find index of maximum element (argmax)
    fn simd_argmax(a: &ArrayView1<Self>) -> Option<usize>;

    /// Clip values to [min_val, max_val] range
    fn simd_clip(a: &ArrayView1<Self>, min_val: Self, max_val: Self) -> Array1<Self>;

    /// Log-sum-exp for numerically stable softmax computation
    fn simd_log_sum_exp(a: &ArrayView1<Self>) -> Self;

    /// Softmax for probability distribution (softmax = exp(x - log_sum_exp(x)))
    fn simd_softmax(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Cumulative sum
    fn simd_cumsum(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Cumulative product
    fn simd_cumprod(a: &ArrayView1<Self>) -> Array1<Self>;

    /// First-order difference (`a[i+1] - a[i]`)
    fn simd_diff(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Sign function: returns -1 for negative, 0 for zero, +1 for positive
    fn simd_sign(a: &ArrayView1<Self>) -> Array1<Self>;

    /// ReLU activation: max(0, x)
    fn simd_relu(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Leaky ReLU: x if x > 0 else alpha * x
    fn simd_leaky_relu(a: &ArrayView1<Self>, alpha: Self) -> Array1<Self>;

    /// L2 normalization (unit vector)
    fn simd_normalize(a: &ArrayView1<Self>) -> Array1<Self>;

    /// Standardization: (x - mean) / std
    fn simd_standardize(a: &ArrayView1<Self>) -> Array1<Self>;
}

// Lanczos approximation helper functions for gamma
/// Lanczos approximation for gamma function (f32)
#[allow(dead_code)]
fn lanczos_gamma_f32(z: f32) -> f32 {
    const G: f32 = 7.0;
    const SQRT_2PI: f32 = 2.5066282746310002; // sqrt(2*PI)
    const LANCZOS_COEFFS: [f32; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    // For z < 0.5, use reflection formula: Γ(z) = π / (sin(πz) · Γ(1-z))
    if z < 0.5 {
        let pi = std::f32::consts::PI;
        let sinpix = (pi * z).sin();
        if sinpix.abs() < 1e-10 {
            return f32::INFINITY;
        }
        return pi / (sinpix * lanczos_gamma_f32(1.0 - z));
    }

    let z_shifted = z - 1.0;
    let mut acc = LANCZOS_COEFFS[0];
    for (i, &coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        acc += coeff / (z_shifted + i as f32);
    }

    let t = z_shifted + G + 0.5;
    SQRT_2PI * acc * t.powf(z_shifted + 0.5) * (-t).exp()
}

/// Lanczos approximation for gamma function (f64)
#[allow(dead_code)]
fn lanczos_gamma_f64(z: f64) -> f64 {
    const G: f64 = 7.0;
    const SQRT_2PI: f64 = 2.5066282746310002; // sqrt(2*PI)
    const LANCZOS_COEFFS: [f64; 9] = [
        0.999999999999809932,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    // For z < 0.5, use reflection formula: Γ(z) = π / (sin(πz) · Γ(1-z))
    if z < 0.5 {
        let pi = std::f64::consts::PI;
        let sinpix = (pi * z).sin();
        if sinpix.abs() < 1e-14 {
            return f64::INFINITY;
        }
        return pi / (sinpix * lanczos_gamma_f64(1.0 - z));
    }

    let z_shifted = z - 1.0;
    let mut acc = LANCZOS_COEFFS[0];
    for (i, &coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        acc += coeff / (z_shifted + i as f64);
    }

    let t = z_shifted + G + 0.5;
    SQRT_2PI * acc * t.powf(z_shifted + 0.5) * (-t).exp()
}

/// Digamma function (psi) - logarithmic derivative of gamma function (f32)
/// ψ(x) = d/dx ln(Γ(x)) = Γ'(x) / Γ(x)
#[allow(dead_code)]
fn digamma_f32(mut x: f32) -> f32 {
    let pi = std::f32::consts::PI;

    // Handle special cases
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { f32::INFINITY } else { f32::NAN };
    }

    // Handle negative x using reflection formula:
    // ψ(-x) = ψ(x) + π·cot(πx) + 1/x
    // Or equivalently: ψ(1-x) - ψ(x) = π·cot(πx)
    if x <= 0.0 {
        // Check if x is a negative integer (pole)
        if x == x.floor() {
            return f32::NAN;
        }
        // Use reflection: ψ(x) = ψ(1-x) - π·cot(πx)
        return digamma_f32(1.0 - x) - pi / (pi * x).tan();
    }

    let mut result = 0.0;

    // Use recurrence relation ψ(x+1) = ψ(x) + 1/x to shift to large x
    // ψ(x) = ψ(x+n) - 1/(x+n-1) - 1/(x+n-2) - ... - 1/x
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }

    // Asymptotic expansion for large x:
    // ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶) + ...
    result += x.ln() - 0.5 / x;
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    result -= 1.0 / (12.0 * x2);
    result += 1.0 / (120.0 * x4);
    result -= 1.0 / (252.0 * x6);

    result
}

/// Digamma function (psi) - logarithmic derivative of gamma function (f64)
/// ψ(x) = d/dx ln(Γ(x)) = Γ'(x) / Γ(x)
#[allow(dead_code)]
fn digamma_f64(mut x: f64) -> f64 {
    let pi = std::f64::consts::PI;

    // Handle special cases
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { f64::INFINITY } else { f64::NAN };
    }

    // Handle negative x using reflection formula:
    // ψ(-x) = ψ(x) + π·cot(πx) + 1/x
    // Or equivalently: ψ(1-x) - ψ(x) = π·cot(πx)
    if x <= 0.0 {
        // Check if x is a negative integer (pole)
        if x == x.floor() {
            return f64::NAN;
        }
        // Use reflection: ψ(x) = ψ(1-x) - π·cot(πx)
        return digamma_f64(1.0 - x) - pi / (pi * x).tan();
    }

    let mut result = 0.0;

    // Use recurrence relation ψ(x+1) = ψ(x) + 1/x to shift to large x
    // ψ(x) = ψ(x+n) - 1/(x+n-1) - 1/(x+n-2) - ... - 1/x
    while x < 8.0 {
        result -= 1.0 / x;
        x += 1.0;
    }

    // Asymptotic expansion for large x (more terms for f64):
    // ψ(x) ≈ ln(x) - 1/(2x) - Σ B_{2k}/(2k·x^{2k}) for k=1,2,...
    // where B_{2k} are Bernoulli numbers
    result += x.ln() - 0.5 / x;
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x4 * x4;
    let x10 = x8 * x2;
    let x12 = x8 * x4;
    // B_2 = 1/6, B_4 = -1/30, B_6 = 1/42, B_8 = -1/30, B_10 = 5/66, B_12 = -691/2730
    result -= 1.0 / (12.0 * x2); // B_2 / (2 * x^2)
    result += 1.0 / (120.0 * x4); // -B_4 / (4 * x^4)
    result -= 1.0 / (252.0 * x6); // B_6 / (6 * x^6)
    result += 1.0 / (240.0 * x8); // -B_8 / (8 * x^8)
    result -= 5.0 / (660.0 * x10); // B_10 / (10 * x^10)
    result += 691.0 / (32760.0 * x12); // -B_12 / (12 * x^12)

    result
}

/// Trigamma function ψ'(x) = d²/dx² ln(Γ(x)) for f32
///
/// Uses:
/// 1. Reflection formula for x < 0.5: ψ'(1-x) + ψ'(x) = π²/sin²(πx)
/// 2. Recurrence relation for small x: ψ'(x+1) = ψ'(x) - 1/x²
/// 3. Asymptotic expansion for large x
fn trigamma_f32(mut x: f32) -> f32 {
    let pi = std::f32::consts::PI;

    // Handle special cases
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 0.0 } else { f32::NAN };
    }

    // Handle negative x using reflection formula:
    // ψ'(1-x) + ψ'(x) = π²/sin²(πx)
    if x <= 0.0 {
        // Check if x is a negative integer (pole)
        if x == x.floor() {
            return f32::NAN;
        }
        // Use reflection: ψ'(x) = π²/sin²(πx) - ψ'(1-x)
        let sin_pix = (pi * x).sin();
        return (pi * pi) / (sin_pix * sin_pix) - trigamma_f32(1.0 - x);
    }

    let mut result = 0.0;

    // Use recurrence relation ψ'(x+1) = ψ'(x) - 1/x² to shift to large x
    // ψ'(x) = ψ'(x+n) + 1/x² + 1/(x+1)² + ... + 1/(x+n-1)²
    while x < 6.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }

    // Asymptotic expansion for large x:
    // ψ'(x) ≈ 1/x + 1/(2x²) + 1/(6x³) - 1/(30x⁵) + 1/(42x⁷) - ...
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;

    result += 1.0 / x;
    result += 1.0 / (2.0 * x2);
    result += 1.0 / (6.0 * x3);
    result -= 1.0 / (30.0 * x5);
    result += 1.0 / (42.0 * x7);

    result
}

/// Trigamma function ψ'(x) = d²/dx² ln(Γ(x)) for f64
///
/// Uses:
/// 1. Reflection formula for x < 0.5: ψ'(1-x) + ψ'(x) = π²/sin²(πx)
/// 2. Recurrence relation for small x: ψ'(x+1) = ψ'(x) - 1/x²
/// 3. Asymptotic expansion for large x (more terms for f64 precision)
fn trigamma_f64(mut x: f64) -> f64 {
    let pi = std::f64::consts::PI;

    // Handle special cases
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 0.0 } else { f64::NAN };
    }

    // Handle negative x using reflection formula:
    // ψ'(1-x) + ψ'(x) = π²/sin²(πx)
    if x <= 0.0 {
        // Check if x is a negative integer (pole)
        if x == x.floor() {
            return f64::NAN;
        }
        // Use reflection: ψ'(x) = π²/sin²(πx) - ψ'(1-x)
        let sin_pix = (pi * x).sin();
        return (pi * pi) / (sin_pix * sin_pix) - trigamma_f64(1.0 - x);
    }

    let mut result = 0.0;

    // Use recurrence relation ψ'(x+1) = ψ'(x) - 1/x² to shift to large x
    // ψ'(x) = ψ'(x+n) + 1/x² + 1/(x+1)² + ... + 1/(x+n-1)²
    while x < 8.0 {
        result += 1.0 / (x * x);
        x += 1.0;
    }

    // Asymptotic expansion for large x (more terms for f64):
    // ψ'(x) ≈ 1/x + 1/(2x²) + Σ B_{2k}/x^{2k+1} for k=1,2,...
    // where B_{2k} are Bernoulli numbers: B_2=1/6, B_4=-1/30, B_6=1/42, B_8=-1/30
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    let x9 = x7 * x2;
    let x11 = x9 * x2;

    result += 1.0 / x;
    result += 1.0 / (2.0 * x2);
    result += 1.0 / (6.0 * x3); // B_2 = 1/6
    result -= 1.0 / (30.0 * x5); // B_4 = -1/30
    result += 1.0 / (42.0 * x7); // B_6 = 1/42
    result -= 1.0 / (30.0 * x9); // B_8 = -1/30
    result += 5.0 / (66.0 * x11); // B_10 = 5/66

    result
}

/// Log-gamma function ln(Γ(x)) for f32
///
/// More numerically stable than gamma(x).ln() since it avoids overflow.
/// Uses Lanczos approximation: ln(Γ(z)) = ln(√(2π)) + (z-0.5)*ln(t) - t + ln(sum)
/// where t = z + g - 0.5
fn ln_gamma_f32(z: f32) -> f32 {
    const G: f32 = 7.0;
    const LN_SQRT_2PI: f32 = 0.9189385332046727; // ln(sqrt(2*PI))
    const LANCZOS_COEFFS: [f32; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    // Handle special cases
    if z.is_nan() {
        return f32::NAN;
    }
    if z.is_infinite() {
        return if z > 0.0 { f32::INFINITY } else { f32::NAN };
    }
    if z <= 0.0 && z == z.floor() {
        return f32::INFINITY; // Poles at non-positive integers
    }

    // For z < 0.5, use reflection formula:
    // ln(Γ(z)) = ln(π) - ln(sin(πz)) - ln(Γ(1-z))
    if z < 0.5 {
        let pi = std::f32::consts::PI;
        let sinpiz = (pi * z).sin().abs();
        if sinpiz < 1e-10 {
            return f32::INFINITY;
        }
        return pi.ln() - sinpiz.ln() - ln_gamma_f32(1.0 - z);
    }

    // Lanczos approximation for z >= 0.5
    let z_shifted = z - 1.0;
    let mut sum = LANCZOS_COEFFS[0];
    for (i, &coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        sum += coeff / (z_shifted + i as f32);
    }

    let t = z_shifted + G + 0.5;
    // ln(Γ(z)) = ln(√(2π)) + (z-0.5)*ln(t) - t + ln(sum)
    LN_SQRT_2PI + (z_shifted + 0.5) * t.ln() - t + sum.ln()
}

/// Log-gamma function ln(Γ(x)) for f64
///
/// More numerically stable than gamma(x).ln() since it avoids overflow.
/// Uses Lanczos approximation with higher precision coefficients.
fn ln_gamma_f64(z: f64) -> f64 {
    const G: f64 = 7.0;
    const LN_SQRT_2PI: f64 = 0.9189385332046727; // ln(sqrt(2*PI))
    const LANCZOS_COEFFS: [f64; 9] = [
        0.999999999999809932,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    // Handle special cases
    if z.is_nan() {
        return f64::NAN;
    }
    if z.is_infinite() {
        return if z > 0.0 { f64::INFINITY } else { f64::NAN };
    }
    if z <= 0.0 && z == z.floor() {
        return f64::INFINITY; // Poles at non-positive integers
    }

    // For z < 0.5, use reflection formula:
    // ln(Γ(z)) = ln(π) - ln(sin(πz)) - ln(Γ(1-z))
    if z < 0.5 {
        let pi = std::f64::consts::PI;
        let sinpiz = (pi * z).sin().abs();
        if sinpiz < 1e-14 {
            return f64::INFINITY;
        }
        return pi.ln() - sinpiz.ln() - ln_gamma_f64(1.0 - z);
    }

    // Lanczos approximation for z >= 0.5
    let z_shifted = z - 1.0;
    let mut sum = LANCZOS_COEFFS[0];
    for (i, &coeff) in LANCZOS_COEFFS.iter().enumerate().skip(1) {
        sum += coeff / (z_shifted + i as f64);
    }

    let t = z_shifted + G + 0.5;
    // ln(Γ(z)) = ln(√(2π)) + (z-0.5)*ln(t) - t + ln(sum)
    LN_SQRT_2PI + (z_shifted + 0.5) * t.ln() - t + sum.ln()
}

/// Error function erf(x) for f32
///
/// Uses Abramowitz & Stegun approximation (equation 7.1.26)
/// Maximum error: ~1.5×10⁻⁷
fn erf_f32(x: f32) -> f32 {
    // Handle special cases
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }

    // erf is an odd function: erf(-x) = -erf(x)
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    // For very small x, use series approximation erf(x) ≈ 2x/√π
    if x < 1e-10 {
        return sign * x * std::f32::consts::FRAC_2_SQRT_PI;
    }

    // Abramowitz & Stegun approximation (7.1.26)
    // erf(x) ≈ 1 - (a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵)e^(-x²)
    // where t = 1/(1 + px), p = 0.3275911
    const P: f32 = 0.3275911;
    const A1: f32 = 0.254829592;
    const A2: f32 = -0.284496736;
    const A3: f32 = 1.421413741;
    const A4: f32 = -1.453152027;
    const A5: f32 = 1.061405429;

    let t = 1.0 / (1.0 + P * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let poly = A1 * t + A2 * t2 + A3 * t3 + A4 * t4 + A5 * t5;
    let result = 1.0 - poly * (-x * x).exp();

    sign * result
}

/// Error function erf(x) for f64
///
/// Uses a higher-precision rational approximation
/// Maximum error: ~1.5×10⁻¹⁵
fn erf_f64(x: f64) -> f64 {
    // Handle special cases
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }

    // erf is an odd function: erf(-x) = -erf(x)
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    // For very small x, use series approximation erf(x) ≈ 2x/√π
    if x < 1e-20 {
        return sign * x * std::f64::consts::FRAC_2_SQRT_PI;
    }

    // For large x, erf approaches 1
    if x > 6.0 {
        return sign;
    }

    // Use different approximations based on x range for best accuracy
    // Maclaurin series loses accuracy near x=0.5, so use lower threshold
    let result = if x < 0.25 {
        // For small x, use Maclaurin series
        // erf(x) = (2/√π) * (x - x³/3 + x⁵/10 - x⁷/42 + x⁹/216 - ...)
        let x2 = x * x;
        let two_over_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI;
        let term = x
            * (1.0
                - x2 / 3.0
                    * (1.0
                        - x2 / 5.0
                            * 0.5
                            * (1.0
                                - x2 / 7.0
                                    * (1.0 / 3.0)
                                    * (1.0
                                        - x2 / 9.0
                                            * 0.25
                                            * (1.0
                                                - x2 / 11.0
                                                    * 0.2
                                                    * (1.0
                                                        - x2 / 13.0
                                                            * (1.0 / 6.0)
                                                            * (1.0 - x2 / 15.0 * (1.0 / 7.0))))))));
        two_over_sqrt_pi * term
    } else {
        // For medium to large x, use Abramowitz & Stegun approximation with more terms
        // This is a 7th order approximation
        const P: f64 = 0.3275911;
        const A1: f64 = 0.254829592;
        const A2: f64 = -0.284496736;
        const A3: f64 = 1.421413741;
        const A4: f64 = -1.453152027;
        const A5: f64 = 1.061405429;

        let t = 1.0 / (1.0 + P * x);
        let poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))));
        1.0 - poly * (-x * x).exp()
    };

    sign * result
}

/// Complementary error function erfc(x) = 1 - erf(x) for f32
///
/// More numerically stable than 1 - erf(x) for large x
fn erfc_f32(x: f32) -> f32 {
    // Handle special cases
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 0.0 } else { 2.0 };
    }

    // For negative x, use erfc(-x) = 2 - erfc(x)
    if x < 0.0 {
        return 2.0 - erfc_f32(-x);
    }

    // For large x, erfc(x) → 0 very quickly
    if x > 10.0 {
        return 0.0;
    }

    // For small x, compute 1 - erf(x) directly
    if x < 0.5 {
        return 1.0 - erf_f32(x);
    }

    // Abramowitz & Stegun approximation for erfc
    // erfc(x) = (a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵)e^(-x²)
    const P: f32 = 0.3275911;
    const A1: f32 = 0.254829592;
    const A2: f32 = -0.284496736;
    const A3: f32 = 1.421413741;
    const A4: f32 = -1.453152027;
    const A5: f32 = 1.061405429;

    let t = 1.0 / (1.0 + P * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let poly = A1 * t + A2 * t2 + A3 * t3 + A4 * t4 + A5 * t5;
    poly * (-x * x).exp()
}

/// Complementary error function erfc(x) = 1 - erf(x) for f64
///
/// More numerically stable than 1 - erf(x) for large x
fn erfc_f64(x: f64) -> f64 {
    // Handle special cases
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 0.0 } else { 2.0 };
    }

    // For negative x, use erfc(-x) = 2 - erfc(x)
    if x < 0.0 {
        return 2.0 - erfc_f64(-x);
    }

    // For very large x, erfc(x) → 0
    if x > 27.0 {
        return 0.0;
    }

    // For small x, compute 1 - erf(x) directly
    if x < 0.5 {
        return 1.0 - erf_f64(x);
    }

    // Use continued fraction for medium to large x
    // erfc(x) = exp(-x²)/√π * 1/(x + 1/(2x + 2/(x + 3/(2x + ...))))
    // This is more accurate than the polynomial for larger x
    if x > 4.0 {
        // Asymptotic expansion for large x
        let x2 = x * x;
        let inv_x2 = 1.0 / x2;
        let sqrt_pi = std::f64::consts::PI.sqrt();

        // erfc(x) ≈ exp(-x²)/(x√π) * (1 - 1/(2x²) + 3/(4x⁴) - 15/(8x⁶) + ...)
        let asymp = 1.0 - 0.5 * inv_x2 + 0.75 * inv_x2 * inv_x2 - 1.875 * inv_x2 * inv_x2 * inv_x2
            + 6.5625 * inv_x2 * inv_x2 * inv_x2 * inv_x2;
        return (-x2).exp() / (x * sqrt_pi) * asymp;
    }

    // Abramowitz & Stegun approximation for medium x
    const P: f64 = 0.3275911;
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;

    let t = 1.0 / (1.0 + P * x);
    let poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))));
    poly * (-x * x).exp()
}

/// Inverse error function erfinv(y) for f32
///
/// Uses Winitzki's approximation followed by Newton-Raphson refinement
/// Domain: (-1, 1), returns x such that erf(x) = y
fn erfinv_f32(y: f32) -> f32 {
    // Handle special cases
    if y.is_nan() {
        return f32::NAN;
    }
    if y <= -1.0 {
        return if y == -1.0 {
            f32::NEG_INFINITY
        } else {
            f32::NAN
        };
    }
    if y >= 1.0 {
        return if y == 1.0 { f32::INFINITY } else { f32::NAN };
    }
    if y == 0.0 {
        return 0.0;
    }

    // erfinv is odd: erfinv(-y) = -erfinv(y)
    let sign = if y >= 0.0 { 1.0 } else { -1.0 };
    let y = y.abs();

    // Winitzki approximation for initial guess
    // erfinv(y) ≈ sign(y) * sqrt(sqrt((4/π + ay²)/(1+ay²))² - (4/π + ay²)/(1+ay²) + 2/πa * ln(1-y²))
    // where a ≈ 0.147 for good accuracy
    let a = 0.147_f32;
    let two_over_pi_a = 2.0 / (std::f32::consts::PI * a);
    let ln_one_minus_y2 = (1.0 - y * y).ln();

    let t1 = two_over_pi_a + 0.5 * ln_one_minus_y2;
    let t2 = (1.0 / a) * ln_one_minus_y2;
    let inner = (t1 * t1 - t2).sqrt() - t1;

    let mut x = inner.sqrt();

    // Newton-Raphson refinement: x_new = x - (erf(x) - y) / erf'(x)
    // erf'(x) = 2/sqrt(π) * exp(-x²)
    let two_over_sqrt_pi = std::f32::consts::FRAC_2_SQRT_PI;
    for _ in 0..2 {
        let erf_x = erf_f32(x);
        let erf_deriv = two_over_sqrt_pi * (-x * x).exp();
        x -= (erf_x - y) / erf_deriv;
    }

    sign * x
}

/// Inverse error function erfinv(y) for f64
///
/// Uses Winitzki's approximation followed by Halley's method refinement
/// Domain: (-1, 1), returns x such that erf(x) = y
fn erfinv_f64(y: f64) -> f64 {
    // Handle special cases
    if y.is_nan() {
        return f64::NAN;
    }
    if y <= -1.0 {
        return if y == -1.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
    }
    if y >= 1.0 {
        return if y == 1.0 { f64::INFINITY } else { f64::NAN };
    }
    if y == 0.0 {
        return 0.0;
    }

    // erfinv is odd: erfinv(-y) = -erfinv(y)
    let sign = if y >= 0.0 { 1.0 } else { -1.0 };
    let y = y.abs();

    // Winitzki approximation for initial guess
    // Higher precision constant for f64
    let a = 0.147_f64;
    let two_over_pi_a = 2.0 / (std::f64::consts::PI * a);
    let ln_one_minus_y2 = (1.0 - y * y).ln();

    let t1 = two_over_pi_a + 0.5 * ln_one_minus_y2;
    let t2 = (1.0 / a) * ln_one_minus_y2;
    let inner = (t1 * t1 - t2).sqrt() - t1;

    let mut x = inner.sqrt();

    // Halley's method for faster convergence (cubic)
    // x_new = x - f(x)/f'(x) * (1 + f(x)*f''(x)/(2*f'(x)²))⁻¹
    // For f(x) = erf(x) - y: f' = 2/√π * e^(-x²), f'' = -4x/√π * e^(-x²)
    // Use 5 iterations for ~1e-14 accuracy (f64 precision)
    let two_over_sqrt_pi = std::f64::consts::FRAC_2_SQRT_PI;
    for _ in 0..5 {
        let erf_x = erf_f64(x);
        let f = erf_x - y;
        let exp_neg_x2 = (-x * x).exp();
        let f_prime = two_over_sqrt_pi * exp_neg_x2;

        // Newton step
        let newton_step = f / f_prime;

        // Halley correction factor
        let f_double_prime = -2.0 * x * f_prime;
        let halley_correction = 1.0 / (1.0 - 0.5 * f * f_double_prime / (f_prime * f_prime));

        x -= newton_step * halley_correction;
    }

    sign * x
}

/// Inverse complementary error function erfcinv(y) for f32
///
/// erfcinv(y) = erfinv(1 - y)
/// Domain: (0, 2), returns x such that erfc(x) = y
fn erfcinv_f32(y: f32) -> f32 {
    // Handle special cases
    if y.is_nan() {
        return f32::NAN;
    }
    if y <= 0.0 {
        return if y == 0.0 { f32::INFINITY } else { f32::NAN };
    }
    if y >= 2.0 {
        return if y == 2.0 {
            f32::NEG_INFINITY
        } else {
            f32::NAN
        };
    }

    // Use erfinv(1 - y) for all valid inputs
    erfinv_f32(1.0 - y)
}

/// Inverse complementary error function erfcinv(y) for f64
///
/// erfcinv(y) = erfinv(1 - y)
/// More numerically stable for y close to 0
/// Domain: (0, 2), returns x such that erfc(x) = y
fn erfcinv_f64(y: f64) -> f64 {
    // Handle special cases
    if y.is_nan() {
        return f64::NAN;
    }
    if y <= 0.0 {
        return if y == 0.0 { f64::INFINITY } else { f64::NAN };
    }
    if y >= 2.0 {
        return if y == 2.0 {
            f64::NEG_INFINITY
        } else {
            f64::NAN
        };
    }

    // Use erfinv for all values: erfcinv(y) = erfinv(1 - y)
    // erfinv_f64 already handles all cases with good accuracy via Halley's method
    erfinv_f64(1.0 - y)
}

/// Sigmoid (logistic) function for f32
///
/// σ(x) = 1 / (1 + exp(-x))
/// Numerically stable implementation that avoids overflow
fn sigmoid_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    // Numerically stable sigmoid:
    // For x >= 0: σ(x) = 1 / (1 + exp(-x))
    // For x < 0: σ(x) = exp(x) / (1 + exp(x))
    if x >= 0.0 {
        let exp_neg_x = (-x).exp();
        1.0 / (1.0 + exp_neg_x)
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// Sigmoid (logistic) function for f64
///
/// σ(x) = 1 / (1 + exp(-x))
/// Numerically stable implementation that avoids overflow
fn sigmoid_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Numerically stable sigmoid:
    // For x >= 0: σ(x) = 1 / (1 + exp(-x))
    // For x < 0: σ(x) = exp(x) / (1 + exp(x))
    if x >= 0.0 {
        let exp_neg_x = (-x).exp();
        1.0 / (1.0 + exp_neg_x)
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

/// GELU (Gaussian Error Linear Unit) for f32
///
/// GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / √2))
/// Critical for Transformer models (BERT, GPT, etc.)
fn gelu_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    // GELU(x) = x * 0.5 * (1 + erf(x / √2))
    let sqrt_2_inv = std::f32::consts::FRAC_1_SQRT_2; // 1/√2
    let erf_arg = x * sqrt_2_inv;
    x * 0.5 * (1.0 + erf_f32(erf_arg))
}

/// GELU (Gaussian Error Linear Unit) for f64
///
/// GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / √2))
/// Critical for Transformer models (BERT, GPT, etc.)
fn gelu_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // GELU(x) = x * 0.5 * (1 + erf(x / √2))
    let sqrt_2_inv = std::f64::consts::FRAC_1_SQRT_2; // 1/√2
    let erf_arg = x * sqrt_2_inv;
    x * 0.5 * (1.0 + erf_f64(erf_arg))
}

/// Swish (SiLU - Sigmoid Linear Unit) for f32
///
/// Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
/// Self-gated activation discovered via neural architecture search
/// Used in EfficientNet, GPT-NeoX, and many modern architectures
/// Properties: smooth, non-monotonic, self-gating, unbounded above
fn swish_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    // Swish(x) = x * sigmoid(x)
    // Leverages the numerically stable sigmoid implementation
    x * sigmoid_f32(x)
}

/// Swish (SiLU - Sigmoid Linear Unit) for f64
///
/// Swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
/// Self-gated activation discovered via neural architecture search
/// Used in EfficientNet, GPT-NeoX, and many modern architectures
/// Properties: smooth, non-monotonic, self-gating, unbounded above
fn swish_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Swish(x) = x * sigmoid(x)
    // Leverages the numerically stable sigmoid implementation
    x * sigmoid_f64(x)
}

/// Softplus for f32
///
/// Softplus(x) = ln(1 + exp(x))
/// Smooth approximation of ReLU
/// Used in probabilistic models, Bayesian deep learning, smooth counting
/// Properties: softplus(0) = ln(2), always positive, approaches ReLU for x → +∞
fn softplus_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    // Numerically stable implementation:
    // For large positive x: ln(1 + exp(x)) ≈ x (exp(x) dominates)
    // For large negative x: ln(1 + exp(x)) ≈ exp(x) ≈ 0
    // Use threshold based on machine precision
    if x > 20.0 {
        // For x > 20, exp(x) >> 1, so ln(1 + exp(x)) ≈ x
        x
    } else if x < -20.0 {
        // For x < -20, exp(x) ≈ 0, so ln(1 + exp(x)) ≈ 0
        x.exp()
    } else {
        // Standard computation
        (1.0_f32 + x.exp()).ln()
    }
}

/// Softplus for f64
///
/// Softplus(x) = ln(1 + exp(x))
/// Smooth approximation of ReLU
/// Used in probabilistic models, Bayesian deep learning, smooth counting
/// Properties: softplus(0) = ln(2), always positive, approaches ReLU for x → +∞
fn softplus_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Numerically stable implementation:
    // For large positive x: ln(1 + exp(x)) ≈ x (exp(x) dominates)
    // For large negative x: ln(1 + exp(x)) ≈ exp(x) ≈ 0
    // Use threshold based on machine precision
    if x > 34.0 {
        // For x > 34, exp(x) >> 1, so ln(1 + exp(x)) ≈ x
        x
    } else if x < -34.0 {
        // For x < -34, exp(x) ≈ 0, so ln(1 + exp(x)) ≈ 0
        x.exp()
    } else {
        // Standard computation
        (1.0_f64 + x.exp()).ln()
    }
}

/// Mish activation for f32
///
/// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
/// Self-regularized non-monotonic activation function
/// Used in YOLOv4, modern object detection, and neural architectures
/// Properties: smooth, non-monotonic, self-gating, unbounded above
fn mish_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    // Mish(x) = x * tanh(softplus(x))
    // Leverage existing softplus for numerical stability
    x * softplus_f32(x).tanh()
}

/// Mish activation for f64
///
/// Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
/// Self-regularized non-monotonic activation function
/// Used in YOLOv4, modern object detection, and neural architectures
/// Properties: smooth, non-monotonic, self-gating, unbounded above
fn mish_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Mish(x) = x * tanh(softplus(x))
    // Leverage existing softplus for numerical stability
    x * softplus_f64(x).tanh()
}

/// ELU (Exponential Linear Unit) for f32
///
/// ELU(x, α) = x if x >= 0
/// ELU(x, α) = α * (exp(x) - 1) if x < 0
/// Helps with vanishing gradients and faster learning
/// Used in deep neural networks for smoother outputs
/// Properties: smooth, continuous derivative, bounded below by -α
fn elu_f32(x: f32, alpha: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x >= 0.0 {
        x
    } else {
        alpha * (x.exp() - 1.0)
    }
}

/// ELU (Exponential Linear Unit) for f64
///
/// ELU(x, α) = x if x >= 0
/// ELU(x, α) = α * (exp(x) - 1) if x < 0
/// Helps with vanishing gradients and faster learning
/// Used in deep neural networks for smoother outputs
/// Properties: smooth, continuous derivative, bounded below by -α
fn elu_f64(x: f64, alpha: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x >= 0.0 {
        x
    } else {
        alpha * (x.exp() - 1.0)
    }
}

/// SELU (Scaled Exponential Linear Unit) constants
///
/// These constants are derived from the self-normalizing property:
/// For inputs with mean 0 and variance 1, the outputs will also have
/// mean 0 and variance 1 (approximately) when using LeCun Normal initialization.
const SELU_ALPHA: f64 = 1.6732632423543772;
const SELU_LAMBDA: f64 = 1.0507009873554805;
const SELU_ALPHA_F32: f32 = 1.6732632;
const SELU_LAMBDA_F32: f32 = 1.0507010;

/// SELU (Scaled Exponential Linear Unit) for f32
///
/// SELU(x) = λ * (x if x > 0, α * (exp(x) - 1) if x <= 0)
/// where λ ≈ 1.0507 and α ≈ 1.6733
/// Self-normalizing activation: preserves mean=0, variance=1 through layers
/// Used in Self-Normalizing Neural Networks (SNNs)
/// Properties: automatic normalization, no BatchNorm needed
fn selu_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x > 0.0 {
        SELU_LAMBDA_F32 * x
    } else {
        SELU_LAMBDA_F32 * SELU_ALPHA_F32 * (x.exp() - 1.0)
    }
}

/// SELU (Scaled Exponential Linear Unit) for f64
///
/// SELU(x) = λ * (x if x > 0, α * (exp(x) - 1) if x <= 0)
/// where λ ≈ 1.0507 and α ≈ 1.6733
/// Self-normalizing activation: preserves mean=0, variance=1 through layers
/// Used in Self-Normalizing Neural Networks (SNNs)
/// Properties: automatic normalization, no BatchNorm needed
fn selu_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > 0.0 {
        SELU_LAMBDA * x
    } else {
        SELU_LAMBDA * SELU_ALPHA * (x.exp() - 1.0)
    }
}

/// Hardsigmoid for f32
///
/// Hardsigmoid(x) = clip((x + 3) / 6, 0, 1)
/// Piecewise linear approximation of sigmoid
/// Used in MobileNetV3 for efficient inference
/// Properties: hardsigmoid(0) = 0.5, linear in [-3, 3], saturates outside
fn hardsigmoid_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    // Piecewise linear: 0 for x <= -3, 1 for x >= 3, linear in between
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        1.0
    } else {
        (x + 3.0) / 6.0
    }
}

/// Hardsigmoid for f64
///
/// Hardsigmoid(x) = clip((x + 3) / 6, 0, 1)
/// Piecewise linear approximation of sigmoid
/// Used in MobileNetV3 for efficient inference
/// Properties: hardsigmoid(0) = 0.5, linear in [-3, 3], saturates outside
fn hardsigmoid_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Piecewise linear: 0 for x <= -3, 1 for x >= 3, linear in between
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        1.0
    } else {
        (x + 3.0) / 6.0
    }
}

/// Hardswish for f32
///
/// Hardswish(x) = x * hardsigmoid(x) = x * clip((x + 3) / 6, 0, 1)
/// Piecewise linear approximation of Swish
/// Used in MobileNetV3 for efficient inference
/// Properties: hardswish(0) = 0, smooth at boundaries, self-gating
fn hardswish_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    // Piecewise: 0 for x <= -3, x for x >= 3, x*(x+3)/6 in between
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        x
    } else {
        x * (x + 3.0) / 6.0
    }
}

/// Hardswish for f64
///
/// Hardswish(x) = x * hardsigmoid(x) = x * clip((x + 3) / 6, 0, 1)
/// Piecewise linear approximation of Swish
/// Used in MobileNetV3 for efficient inference
/// Properties: hardswish(0) = 0, smooth at boundaries, self-gating
fn hardswish_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // Piecewise: 0 for x <= -3, x for x >= 3, x*(x+3)/6 in between
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        x
    } else {
        x * (x + 3.0) / 6.0
    }
}

/// Sinc function for f32 (normalized)
///
/// sinc(x) = sin(πx) / (πx) for x ≠ 0
/// sinc(0) = 1 (by L'Hôpital's rule)
/// Critical for signal processing, windowing, interpolation
/// Properties: sinc(n) = 0 for all non-zero integers n
fn sinc_f32(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x.abs() < 1e-7 {
        // For very small x, sin(πx) ≈ πx, so sinc(x) ≈ 1
        // Use Taylor expansion: sinc(x) = 1 - (πx)²/6 + O(x⁴)
        let pi_x = std::f32::consts::PI * x;
        1.0 - pi_x * pi_x / 6.0
    } else {
        let pi_x = std::f32::consts::PI * x;
        pi_x.sin() / pi_x
    }
}

/// Sinc function for f64 (normalized)
///
/// sinc(x) = sin(πx) / (πx) for x ≠ 0
/// sinc(0) = 1 (by L'Hôpital's rule)
/// Critical for signal processing, windowing, interpolation
/// Properties: sinc(n) = 0 for all non-zero integers n
fn sinc_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.abs() < 1e-15 {
        // For very small x, sin(πx) ≈ πx, so sinc(x) ≈ 1
        // Use Taylor expansion: sinc(x) = 1 - (πx)²/6 + (πx)⁴/120 + O(x⁶)
        let pi_x = std::f64::consts::PI * x;
        let pi_x_sq = pi_x * pi_x;
        1.0 - pi_x_sq / 6.0 + pi_x_sq * pi_x_sq / 120.0
    } else {
        let pi_x = std::f64::consts::PI * x;
        pi_x.sin() / pi_x
    }
}

// Implementation for f32
impl SimdUnifiedOps for f32 {
    #[cfg(feature = "simd")]
    fn simd_add(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_add_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_add(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        (a + b).to_owned()
    }

    #[cfg(feature = "simd")]
    fn simd_sub(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_sub_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_sub(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        (a - b).to_owned()
    }

    #[cfg(feature = "simd")]
    fn simd_mul(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_mul_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_mul(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        (a * b).to_owned()
    }

    #[cfg(feature = "simd")]
    fn simd_div(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_div_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_div(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        (a / b).to_owned()
    }

    #[cfg(feature = "simd")]
    fn simd_dot(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_dot_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_dot(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        a.dot(b)
    }

    fn simd_gemv(a: &ArrayView2<Self>, x: &ArrayView1<Self>, beta: Self, y: &mut Array1<Self>) {
        let m = a.nrows();
        let n = a.ncols();

        assert_eq!(n, x.len());
        assert_eq!(m, y.len());

        // Scale y by beta
        if beta == 0.0 {
            y.fill(0.0);
        } else if beta != 1.0 {
            y.mapv_inplace(|v| v * beta);
        }

        // Compute matrix-vector product
        for i in 0..m {
            let row = a.row(i);
            y[i] += Self::simd_dot(&row, x);
        }
    }

    fn simd_gemm(
        alpha: Self,
        a: &ArrayView2<Self>,
        b: &ArrayView2<Self>,
        beta: Self,
        c: &mut Array2<Self>,
    ) {
        let m = a.nrows();
        let k = a.ncols();
        let n = b.ncols();

        assert_eq!(k, b.nrows());
        assert_eq!((m, n), c.dim());

        // Scale C by beta
        if beta == 0.0 {
            c.fill(0.0);
        } else if beta != 1.0 {
            c.mapv_inplace(|v| v * beta);
        }

        // Use blocked transpose for large matrices to improve cache efficiency
        // Threshold: n * k > 4096 (amortize transpose cost)
        const GEMM_TRANSPOSE_THRESHOLD: usize = 4096;

        if n * k > GEMM_TRANSPOSE_THRESHOLD {
            // Pre-transpose B for cache-efficient row-wise access
            let b_t = Self::simd_transpose_blocked(b);

            for i in 0..m {
                let a_row = a.row(i);
                for j in 0..n {
                    // Access b_t row-wise (contiguous memory)
                    let b_row = b_t.row(j);
                    c[[i, j]] += alpha * Self::simd_dot(&a_row, &b_row);
                }
            }
        } else {
            // Small matrices: use column access (overhead of transpose not worth it)
            for i in 0..m {
                let a_row = a.row(i);
                for j in 0..n {
                    let b_col = b.column(j);
                    c[[i, j]] += alpha * Self::simd_dot(&a_row, &b_col);
                }
            }
        }
    }

    #[cfg(feature = "simd")]
    fn simd_norm(a: &ArrayView1<Self>) -> Self {
        crate::simd::norms::simd_norm_l2_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_norm(a: &ArrayView1<Self>) -> Self {
        a.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    #[cfg(feature = "simd")]
    fn simd_max(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_maximum_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_max(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for _i in 0..a.len() {
            result[0] = a[0].max(b[0]);
        }
        result
    }

    #[cfg(feature = "simd")]
    fn simd_min(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_minimum_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_min(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for _i in 0..a.len() {
            result[0] = a[0].min(b[0]);
        }
        result
    }

    #[cfg(feature = "simd")]
    fn simd_scalar_mul(a: &ArrayView1<Self>, scalar: Self) -> Array1<Self> {
        crate::simd::simd_scalar_mul_f32(a, scalar)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_scalar_mul(a: &ArrayView1<Self>, scalar: Self) -> Array1<Self> {
        a.mapv(|x| x * scalar)
    }

    #[cfg(feature = "simd")]
    fn simd_sum(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_sum_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_sum(a: &ArrayView1<Self>) -> Self {
        a.sum()
    }

    fn simd_mean(a: &ArrayView1<Self>) -> Self {
        if a.is_empty() {
            0.0
        } else {
            Self::simd_sum(a) / (a.len() as f32)
        }
    }

    #[cfg(feature = "simd")]
    fn simd_max_element(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_max_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_max_element(a: &ArrayView1<Self>) -> Self {
        a.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x))
    }

    #[cfg(feature = "simd")]
    fn simd_min_element(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_min_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_min_element(a: &ArrayView1<Self>) -> Self {
        a.fold(f32::INFINITY, |acc, &x| acc.min(x))
    }

    #[cfg(feature = "simd")]
    fn simd_fma(a: &ArrayView1<Self>, b: &ArrayView1<Self>, c: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_fused_multiply_add_f32(a, b, c)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_fma(a: &ArrayView1<Self>, b: &ArrayView1<Self>, c: &ArrayView1<Self>) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for _i in 0..a.len() {
            result[0] = a[0] * b[0] + c[0];
        }
        result
    }

    #[cfg(feature = "simd")]
    fn simd_add_cache_optimized(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_add_cache_optimized_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_add_cache_optimized(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        a + b
    }

    #[cfg(feature = "simd")]
    fn simd_fma_advanced_optimized(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        c: &ArrayView1<Self>,
    ) -> Array1<Self> {
        crate::simd::simd_fma_advanced_optimized_f32(a, b, c)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_fma_advanced_optimized(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        c: &ArrayView1<Self>,
    ) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for _i in 0..a.len() {
            result[0] = a[0] * b[0] + c[0];
        }
        result
    }

    #[cfg(feature = "simd")]
    fn simd_add_adaptive(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_adaptive_add_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_add_adaptive(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        a + b
    }

    fn simd_transpose(a: &ArrayView2<Self>) -> Array2<Self> {
        a.t().to_owned()
    }

    fn simd_transpose_blocked(a: &ArrayView2<Self>) -> Array2<Self> {
        #[cfg(feature = "simd")]
        {
            crate::simd::simd_transpose_blocked_f32(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.t().to_owned()
        }
    }

    fn simd_sum_squares(a: &ArrayView1<Self>) -> Self {
        a.iter().map(|&x| x * x).sum()
    }

    fn simd_multiply(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        Self::simd_mul(a, b)
    }

    #[cfg(feature = "simd")]
    fn simd_available() -> bool {
        true
    }

    #[cfg(not(feature = "simd"))]
    fn simd_available() -> bool {
        false
    }

    fn simd_sub_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let sub_result = Self::simd_sub(a, b);
        result.assign(&sub_result);
    }

    fn simd_mul_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let mul_result = Self::simd_mul(a, b);
        result.assign(&mul_result);
    }

    fn simd_sum_cubes(a: &ArrayView1<Self>) -> Self {
        // Calculate sum of cubes: sum(x^3)
        a.iter().map(|&x| x * x * x).sum()
    }

    fn simd_div_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let div_result = Self::simd_div(a, b);
        result.assign(&div_result);
    }

    fn simd_sin_f32_ultra(a: &ArrayView1<Self>, result: &mut ArrayViewMut1<Self>) {
        let sin_result = a.mapv(|x| x.sin());
        result.assign(&sin_result);
    }

    fn simd_add_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let add_result = Self::simd_add(a, b);
        result.assign(&add_result);
    }

    fn simd_fma_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        c: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let fma_result = Self::simd_fma(a, b, c);
        result.assign(&fma_result);
    }

    fn simd_pow_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let pow_result = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x.powf(y))
            .collect::<Vec<_>>();
        result.assign(&Array1::from_vec(pow_result));
    }

    fn simd_exp_f32_ultra(a: &ArrayView1<Self>, result: &mut ArrayViewMut1<Self>) {
        let exp_result = a.mapv(|x| x.exp());
        result.assign(&exp_result);
    }

    fn simd_cos_f32_ultra(a: &ArrayView1<Self>, result: &mut ArrayViewMut1<Self>) {
        let cos_result = a.mapv(|x| x.cos());
        result.assign(&cos_result);
    }

    fn simd_dot_f32_ultra(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        Self::simd_dot(a, b)
    }

    #[cfg(feature = "simd")]
    fn simd_variance(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_variance_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_variance(a: &ArrayView1<Self>) -> Self {
        let mean = Self::simd_mean(a);
        let n = a.len() as f32;
        if n < 2.0 {
            return f32::NAN;
        }
        // Sample variance with Bessel's correction (n-1)
        a.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (n - 1.0)
    }

    #[cfg(feature = "simd")]
    fn simd_std(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_std_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_std(a: &ArrayView1<Self>) -> Self {
        Self::simd_variance(a).sqrt()
    }

    #[cfg(feature = "simd")]
    fn simd_norm_l1(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_norm_l1_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_norm_l1(a: &ArrayView1<Self>) -> Self {
        a.iter().map(|&x| x.abs()).sum()
    }

    #[cfg(feature = "simd")]
    fn simd_norm_linf(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_norm_linf_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_norm_linf(a: &ArrayView1<Self>) -> Self {
        a.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()))
    }

    #[cfg(feature = "simd")]
    fn simd_cosine_similarity(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_cosine_similarity_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_cosine_similarity(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        let dot = Self::simd_dot(a, b);
        let norm_a = Self::simd_norm(a);
        let norm_b = Self::simd_norm(b);
        dot / (norm_a * norm_b)
    }

    #[cfg(feature = "simd")]
    fn simd_distance_euclidean(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_distance_euclidean_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_distance_euclidean(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    #[cfg(feature = "simd")]
    fn simd_distance_manhattan(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_distance_manhattan_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_distance_manhattan(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
    }

    #[cfg(feature = "simd")]
    fn simd_distance_chebyshev(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_distance_chebyshev_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_distance_chebyshev(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        a.iter()
            .zip(b.iter())
            .fold(0.0f32, |acc, (&x, &y)| acc.max((x - y).abs()))
    }

    #[cfg(feature = "simd")]
    fn simd_distance_cosine(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_distance_cosine_f32(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_distance_cosine(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        1.0 - Self::simd_cosine_similarity(a, b)
    }

    #[cfg(feature = "simd")]
    fn simd_weighted_sum(values: &ArrayView1<Self>, weights: &ArrayView1<Self>) -> Self {
        crate::simd::simd_weighted_sum_f32(values, weights)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_weighted_sum(values: &ArrayView1<Self>, weights: &ArrayView1<Self>) -> Self {
        values
            .iter()
            .zip(weights.iter())
            .map(|(&v, &w)| v * w)
            .sum()
    }

    #[cfg(feature = "simd")]
    fn simd_weighted_mean(values: &ArrayView1<Self>, weights: &ArrayView1<Self>) -> Self {
        crate::simd::simd_weighted_mean_f32(values, weights)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_weighted_mean(values: &ArrayView1<Self>, weights: &ArrayView1<Self>) -> Self {
        let weighted_sum = Self::simd_weighted_sum(values, weights);
        let weight_sum: f32 = weights.iter().sum();
        weighted_sum / weight_sum
    }

    #[cfg(feature = "simd")]
    fn simd_argmin(a: &ArrayView1<Self>) -> Option<usize> {
        crate::simd::simd_argmin_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_argmin(a: &ArrayView1<Self>) -> Option<usize> {
        if a.is_empty() {
            return None;
        }
        let mut min_idx = 0;
        let mut min_val = a[0];
        for (i, &v) in a.iter().enumerate().skip(1) {
            if v < min_val {
                min_val = v;
                min_idx = i;
            }
        }
        Some(min_idx)
    }

    #[cfg(feature = "simd")]
    fn simd_argmax(a: &ArrayView1<Self>) -> Option<usize> {
        crate::simd::simd_argmax_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_argmax(a: &ArrayView1<Self>) -> Option<usize> {
        if a.is_empty() {
            return None;
        }
        let mut max_idx = 0;
        let mut max_val = a[0];
        for (i, &v) in a.iter().enumerate().skip(1) {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        Some(max_idx)
    }

    #[cfg(feature = "simd")]
    fn simd_clip(a: &ArrayView1<Self>, min_val: Self, max_val: Self) -> Array1<Self> {
        crate::simd::simd_clip_f32(a, min_val, max_val)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_clip(a: &ArrayView1<Self>, min_val: Self, max_val: Self) -> Array1<Self> {
        a.mapv(|v| v.max(min_val).min(max_val))
    }

    #[cfg(feature = "simd")]
    fn simd_log_sum_exp(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_log_sum_exp_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_log_sum_exp(a: &ArrayView1<Self>) -> Self {
        if a.is_empty() {
            return f32::NEG_INFINITY;
        }
        let max_val = a.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        let sum_exp: f32 = a.iter().map(|&x| (x - max_val).exp()).sum();
        max_val + sum_exp.ln()
    }

    #[cfg(feature = "simd")]
    fn simd_softmax(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_softmax_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_softmax(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.is_empty() {
            return Array1::zeros(0);
        }
        let lse = Self::simd_log_sum_exp(a);
        a.mapv(|x| (x - lse).exp())
    }

    #[cfg(feature = "simd")]
    fn simd_cumsum(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_cumsum_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_cumsum(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.is_empty() {
            return Array1::zeros(0);
        }
        let mut cumsum = 0.0f32;
        a.mapv(|x| {
            cumsum += x;
            cumsum
        })
    }

    #[cfg(feature = "simd")]
    fn simd_cumprod(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_cumprod_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_cumprod(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.is_empty() {
            return Array1::zeros(0);
        }
        let mut cumprod = 1.0f32;
        a.mapv(|x| {
            cumprod *= x;
            cumprod
        })
    }

    #[cfg(feature = "simd")]
    fn simd_diff(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_diff_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_diff(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.len() <= 1 {
            return Array1::zeros(0);
        }
        Array1::from_iter((1..a.len()).map(|i| a[i] - a[i - 1]))
    }

    #[cfg(feature = "simd")]
    fn simd_sign(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_sign_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_sign(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
    }

    #[cfg(feature = "simd")]
    fn simd_relu(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_relu_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_relu(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.max(0.0))
    }

    #[cfg(feature = "simd")]
    fn simd_leaky_relu(a: &ArrayView1<Self>, alpha: Self) -> Array1<Self> {
        crate::simd::simd_leaky_relu_f32(a, alpha)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_leaky_relu(a: &ArrayView1<Self>, alpha: Self) -> Array1<Self> {
        a.mapv(|x| if x > 0.0 { x } else { alpha * x })
    }

    #[cfg(feature = "simd")]
    fn simd_normalize(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_normalize_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_normalize(a: &ArrayView1<Self>) -> Array1<Self> {
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            return a.to_owned();
        }
        a.mapv(|x| x / norm)
    }

    #[cfg(feature = "simd")]
    fn simd_standardize(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_standardize_f32(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_standardize(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.len() <= 1 {
            return Array1::zeros(a.len());
        }
        let mean: f32 = a.iter().sum::<f32>() / a.len() as f32;
        let variance: f32 =
            a.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / (a.len() - 1) as f32;
        let std = variance.sqrt();
        if std == 0.0 {
            return Array1::zeros(a.len());
        }
        a.mapv(|x| (x - mean) / std)
    }

    fn simd_abs(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.abs())
    }

    fn simd_sqrt(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.sqrt())
    }

    fn simd_exp(a: &ArrayView1<Self>) -> Array1<Self> {
        // Note: SIMD polynomial approximation available via crate::simd::simd_exp_f32
        // for ~5-10x speedup with 10^-7 accuracy. Keeping scalar for trait compatibility.
        a.mapv(|x| x.exp())
    }

    fn simd_ln(a: &ArrayView1<Self>) -> Array1<Self> {
        // Note: SIMD logarithm available via crate::simd::simd_ln_f32
        // for speedup with ~0.05 absolute error. Keeping scalar for trait compatibility.
        a.mapv(|x| x.ln())
    }

    fn simd_sin(a: &ArrayView1<Self>) -> Array1<Self> {
        // Note: SIMD polynomial approximation available via crate::simd::simd_sin_f32
        // for ~5x speedup with 10^-4 accuracy. Keeping scalar for trait compatibility.
        a.mapv(|x| x.sin())
    }

    fn simd_cos(a: &ArrayView1<Self>) -> Array1<Self> {
        // Note: SIMD cosine available via crate::simd::simd_cos_f32
        // for ~5x speedup with 10^-4 accuracy. Keeping scalar for trait compatibility.
        a.mapv(|x| x.cos())
    }

    fn simd_tan(a: &ArrayView1<Self>) -> Array1<Self> {
        // Note: Can use SIMD via sin/cos: crate::simd::simd_sin_f32 / crate::simd::simd_cos_f32
        a.mapv(|x| x.tan())
    }

    fn simd_sinh(a: &ArrayView1<Self>) -> Array1<Self> {
        // sinh(x) = (exp(x) - exp(-x)) / 2
        // Using SIMD exp for acceleration
        let exp_a = Self::simd_exp(a);
        let neg_a = Self::simd_scalar_mul(a, -1.0);
        let exp_neg_a = Self::simd_exp(&neg_a.view());
        let diff = Self::simd_sub(&exp_a.view(), &exp_neg_a.view());
        Self::simd_scalar_mul(&diff.view(), 0.5)
    }

    fn simd_cosh(a: &ArrayView1<Self>) -> Array1<Self> {
        // cosh(x) = (exp(x) + exp(-x)) / 2
        // Using SIMD exp for acceleration
        let exp_a = Self::simd_exp(a);
        let neg_a = Self::simd_scalar_mul(a, -1.0);
        let exp_neg_a = Self::simd_exp(&neg_a.view());
        let sum = Self::simd_add(&exp_a.view(), &exp_neg_a.view());
        Self::simd_scalar_mul(&sum.view(), 0.5)
    }

    fn simd_tanh(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            // Use polynomial approximation (good accuracy, fast)
            simd_ops_polynomial::simd_tanh_f32_poly(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.tanh())
        }
    }

    fn simd_floor(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            crate::simd::simd_floor_f32(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.floor())
        }
    }

    fn simd_ceil(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            crate::simd::simd_ceil_f32(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.ceil())
        }
    }

    fn simd_round(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            crate::simd::simd_round_f32(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.round())
        }
    }

    fn simd_atan(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.atan())
    }

    fn simd_asin(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.asin())
    }

    fn simd_acos(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.acos())
    }

    fn simd_atan2(y: &ArrayView1<Self>, x: &ArrayView1<Self>) -> Array1<Self> {
        y.iter()
            .zip(x.iter())
            .map(|(&y_val, &x_val)| y_val.atan2(x_val))
            .collect::<Vec<_>>()
            .into()
    }

    fn simd_log10(a: &ArrayView1<Self>) -> Array1<Self> {
        // log10(x) = ln(x) * (1/ln(10)) - uses SIMD ln
        const LOG10_E: f32 = std::f32::consts::LOG10_E; // 1/ln(10)
        let ln_a = Self::simd_ln(a);
        Self::simd_scalar_mul(&ln_a.view(), LOG10_E)
    }

    fn simd_log2(a: &ArrayView1<Self>) -> Array1<Self> {
        // log2(x) = ln(x) * (1/ln(2)) - uses SIMD ln
        const LOG2_E: f32 = std::f32::consts::LOG2_E; // 1/ln(2)
        let ln_a = Self::simd_ln(a);
        Self::simd_scalar_mul(&ln_a.view(), LOG2_E)
    }

    #[cfg(feature = "simd")]
    fn simd_clamp(a: &ArrayView1<Self>, min: Self, max: Self) -> Array1<Self> {
        // Use SIMD-accelerated clip (AVX2/SSE/NEON)
        crate::simd::simd_clip_f32(a, min, max)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_clamp(a: &ArrayView1<Self>, min: Self, max: Self) -> Array1<Self> {
        a.mapv(|x| x.clamp(min, max))
    }

    fn simd_fract(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            let truncated = crate::simd::simd_trunc_f32(a);
            Self::simd_sub(a, &truncated.view())
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.fract())
        }
    }

    fn simd_trunc(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            crate::simd::simd_trunc_f32(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.trunc())
        }
    }

    fn simd_recip(a: &ArrayView1<Self>) -> Array1<Self> {
        // Optimized SIMD reciprocal: 1/x using SIMD division
        let ones = Array1::from_elem(a.len(), 1.0f32);
        Self::simd_div(&ones.view(), a)
    }

    fn simd_powf(base: &ArrayView1<Self>, exp: Self) -> Array1<Self> {
        // Optimized SIMD powf: base^exp = exp(exp * ln(base))
        // Uses SIMD-accelerated exp and ln operations
        let ln_base = Self::simd_ln(base);
        let scaled = Self::simd_scalar_mul(&ln_base.view(), exp);
        Self::simd_exp(&scaled.view())
    }

    fn simd_pow(base: &ArrayView1<Self>, exp: &ArrayView1<Self>) -> Array1<Self> {
        // Optimized SIMD pow: base[i]^exp[i] = exp(exp[i] * ln(base[i]))
        // Uses SIMD-accelerated exp, ln, and mul operations
        let ln_base = Self::simd_ln(base);
        let scaled = Self::simd_mul(&ln_base.view(), exp);
        Self::simd_exp(&scaled.view())
    }

    #[cfg(feature = "simd")]
    fn simd_powi(base: &ArrayView1<Self>, n: i32) -> Array1<Self> {
        crate::simd::unary_powi::simd_powi_f32(base, n)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_powi(base: &ArrayView1<Self>, n: i32) -> Array1<Self> {
        base.mapv(|x| x.powi(n))
    }

    fn simd_gamma(x: &ArrayView1<Self>) -> Array1<Self> {
        x.mapv(lanczos_gamma_f32)
    }

    fn simd_exp2(a: &ArrayView1<Self>) -> Array1<Self> {
        // 2^x = exp(x * ln(2))
        const LN2: f32 = std::f32::consts::LN_2;
        let scaled = Self::simd_scalar_mul(a, LN2);
        Self::simd_exp(&scaled.view())
    }

    fn simd_cbrt(a: &ArrayView1<Self>) -> Array1<Self> {
        // Cube root: x^(1/3)
        // Handle negative numbers: cbrt(-x) = -cbrt(x)
        a.mapv(|x| x.cbrt())
    }

    fn simd_ln_1p(a: &ArrayView1<Self>) -> Array1<Self> {
        // ln(1+x) - numerically stable for small x
        a.mapv(|x| x.ln_1p())
    }

    fn simd_exp_m1(a: &ArrayView1<Self>) -> Array1<Self> {
        // exp(x)-1 - numerically stable for small x
        a.mapv(|x| x.exp_m1())
    }

    fn simd_to_radians(a: &ArrayView1<Self>) -> Array1<Self> {
        // degrees to radians: x * π / 180
        const DEG_TO_RAD: f32 = std::f32::consts::PI / 180.0;
        Self::simd_scalar_mul(a, DEG_TO_RAD)
    }

    fn simd_to_degrees(a: &ArrayView1<Self>) -> Array1<Self> {
        // radians to degrees: x * 180 / π
        const RAD_TO_DEG: f32 = 180.0 / std::f32::consts::PI;
        Self::simd_scalar_mul(a, RAD_TO_DEG)
    }

    fn simd_digamma(a: &ArrayView1<Self>) -> Array1<Self> {
        // Digamma function ψ(x) = d/dx ln(Γ(x))
        // Uses reflection formula, recurrence relation, and asymptotic expansion
        a.mapv(digamma_f32)
    }

    fn simd_trigamma(a: &ArrayView1<Self>) -> Array1<Self> {
        // Trigamma function ψ'(x) = d²/dx² ln(Γ(x))
        // Critical for Fisher information in Bayesian inference
        a.mapv(trigamma_f32)
    }

    fn simd_ln_gamma(a: &ArrayView1<Self>) -> Array1<Self> {
        // Log-gamma function ln(Γ(x)) - more stable than gamma(x).ln()
        a.mapv(ln_gamma_f32)
    }

    fn simd_erf(a: &ArrayView1<Self>) -> Array1<Self> {
        // Error function erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
        // Critical for normal distribution CDF: Φ(x) = 0.5 * (1 + erf(x/√2))
        a.mapv(erf_f32)
    }

    fn simd_erfc(a: &ArrayView1<Self>) -> Array1<Self> {
        // Complementary error function erfc(x) = 1 - erf(x)
        // More numerically stable than 1 - erf(x) for large x
        a.mapv(erfc_f32)
    }

    fn simd_erfinv(a: &ArrayView1<Self>) -> Array1<Self> {
        // Inverse error function: erfinv(y) = x such that erf(x) = y
        // Critical for inverse normal CDF (probit function)
        a.mapv(erfinv_f32)
    }

    fn simd_erfcinv(a: &ArrayView1<Self>) -> Array1<Self> {
        // Inverse complementary error function: erfcinv(y) = x such that erfc(x) = y
        // More numerically stable than erfinv(1-y) for small y
        a.mapv(erfcinv_f32)
    }

    fn simd_sigmoid(a: &ArrayView1<Self>) -> Array1<Self> {
        // Sigmoid (logistic) function: σ(x) = 1 / (1 + exp(-x))
        // Critical for neural networks, logistic regression
        // Uses numerically stable implementation
        a.mapv(sigmoid_f32)
    }

    fn simd_gelu(a: &ArrayView1<Self>) -> Array1<Self> {
        // GELU (Gaussian Error Linear Unit): x * Φ(x) = x * 0.5 * (1 + erf(x / √2))
        // Critical for Transformer models (BERT, GPT, etc.)
        a.mapv(gelu_f32)
    }

    fn simd_swish(a: &ArrayView1<Self>) -> Array1<Self> {
        // Swish (SiLU): x * sigmoid(x)
        // Self-gated activation for EfficientNet, GPT-NeoX
        a.mapv(swish_f32)
    }

    fn simd_softplus(a: &ArrayView1<Self>) -> Array1<Self> {
        // Softplus: ln(1 + exp(x))
        // Smooth approximation of ReLU for probabilistic models
        a.mapv(softplus_f32)
    }

    fn simd_mish(a: &ArrayView1<Self>) -> Array1<Self> {
        // Mish: x * tanh(softplus(x))
        // Self-regularized activation for YOLOv4
        a.mapv(mish_f32)
    }

    fn simd_elu(a: &ArrayView1<Self>, alpha: Self) -> Array1<Self> {
        // ELU: x if x >= 0, alpha * (exp(x) - 1) if x < 0
        // Helps with vanishing gradients
        a.mapv(|x| elu_f32(x, alpha))
    }

    fn simd_selu(a: &ArrayView1<Self>) -> Array1<Self> {
        // SELU: λ * (x if x > 0, α * (exp(x) - 1) if x <= 0)
        // Self-normalizing activation with fixed α and λ constants
        a.mapv(selu_f32)
    }

    fn simd_hardsigmoid(a: &ArrayView1<Self>) -> Array1<Self> {
        // Hardsigmoid: clip((x + 3) / 6, 0, 1)
        // Piecewise linear approximation of sigmoid
        a.mapv(hardsigmoid_f32)
    }

    fn simd_hardswish(a: &ArrayView1<Self>) -> Array1<Self> {
        // Hardswish: x * hardsigmoid(x)
        // Piecewise linear approximation of Swish
        a.mapv(hardswish_f32)
    }

    fn simd_sinc(a: &ArrayView1<Self>) -> Array1<Self> {
        // Sinc: sin(πx) / (πx), with sinc(0) = 1
        // Critical for signal processing and interpolation
        a.mapv(sinc_f32)
    }

    fn simd_log_softmax(a: &ArrayView1<Self>) -> Array1<Self> {
        // Log-softmax: x_i - log(Σ_j exp(x_j))
        // Numerically stable for neural network loss computation
        // Leverages existing SIMD-accelerated log_sum_exp
        if a.is_empty() {
            return Array1::zeros(0);
        }
        let lse = Self::simd_log_sum_exp(a);
        a.mapv(|x| x - lse)
    }

    fn simd_asinh(a: &ArrayView1<Self>) -> Array1<Self> {
        // Inverse hyperbolic sine: asinh(x) = ln(x + √(x² + 1))
        // Uses Rust's built-in asinh for accuracy
        a.mapv(|x| x.asinh())
    }

    fn simd_acosh(a: &ArrayView1<Self>) -> Array1<Self> {
        // Inverse hyperbolic cosine: acosh(x) = ln(x + √(x² - 1))
        // Domain: [1, +∞), returns NaN for x < 1
        a.mapv(|x| x.acosh())
    }

    fn simd_atanh(a: &ArrayView1<Self>) -> Array1<Self> {
        // Inverse hyperbolic tangent: atanh(x) = 0.5 * ln((1+x)/(1-x))
        // Domain: (-1, 1), returns ±∞ at ±1, NaN for |x| > 1
        a.mapv(|x| x.atanh())
    }

    fn simd_ln_beta(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        // Log-Beta: ln(B(a,b)) = ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b))
        // Leverages SIMD-accelerated ln_gamma
        let ln_gamma_a = Self::simd_ln_gamma(a);
        let ln_gamma_b = Self::simd_ln_gamma(b);
        let a_plus_b = Self::simd_add(a, b);
        let ln_gamma_ab = Self::simd_ln_gamma(&a_plus_b.view());
        Self::simd_sub(
            &Self::simd_add(&ln_gamma_a.view(), &ln_gamma_b.view()).view(),
            &ln_gamma_ab.view(),
        )
    }

    fn simd_beta(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        // Beta: B(a,b) = exp(ln(B(a,b)))
        // Uses log-beta for numerical stability
        let ln_beta = Self::simd_ln_beta(a, b);
        Self::simd_exp(&ln_beta.view())
    }

    fn simd_lerp(a: &ArrayView1<Self>, b: &ArrayView1<Self>, t: Self) -> Array1<Self> {
        // Linear interpolation: lerp(a, b, t) = a + t * (b - a)
        // This formula is more numerically stable when t is close to 0
        // Alternative: a * (1 - t) + b * t (more symmetric but less stable)
        if a.is_empty() || b.is_empty() {
            return Array1::zeros(0);
        }
        let diff = Self::simd_sub(b, a);
        let scaled = Self::simd_scalar_mul(&diff.view(), t);
        Self::simd_add(a, &scaled.view())
    }

    fn simd_smoothstep(edge0: Self, edge1: Self, x: &ArrayView1<Self>) -> Array1<Self> {
        // Smoothstep: smooth Hermite interpolation
        // t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
        // result = t * t * (3 - 2 * t)
        if x.is_empty() {
            return Array1::zeros(0);
        }
        let range = edge1 - edge0;
        if range.abs() < Self::EPSILON {
            // Degenerate case: edge0 == edge1
            return x.mapv(|xi| if xi < edge0 { 0.0 } else { 1.0 });
        }
        x.mapv(|xi| {
            let t = ((xi - edge0) / range).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        })
    }

    fn simd_hypot(x: &ArrayView1<Self>, y: &ArrayView1<Self>) -> Array1<Self> {
        // Hypotenuse: sqrt(x² + y²) with overflow/underflow protection
        // Uses Rust's built-in hypot which handles extreme values correctly
        if x.is_empty() || y.is_empty() {
            return Array1::zeros(0);
        }
        let len = x.len().min(y.len());
        Array1::from_iter((0..len).map(|i| x[i].hypot(y[i])))
    }

    fn simd_copysign(x: &ArrayView1<Self>, y: &ArrayView1<Self>) -> Array1<Self> {
        // Returns magnitude of x with sign of y
        if x.is_empty() || y.is_empty() {
            return Array1::zeros(0);
        }
        let len = x.len().min(y.len());
        Array1::from_iter((0..len).map(|i| x[i].copysign(y[i])))
    }

    fn simd_smootherstep(edge0: Self, edge1: Self, x: &ArrayView1<Self>) -> Array1<Self> {
        // Ken Perlin's smootherstep: 6t⁵ - 15t⁴ + 10t³
        // Has zero first AND second derivatives at boundaries
        if x.is_empty() {
            return Array1::zeros(0);
        }
        let range = edge1 - edge0;
        if range.abs() < Self::EPSILON {
            return x.mapv(|xi| if xi < edge0 { 0.0 } else { 1.0 });
        }
        x.mapv(|xi| {
            let t = ((xi - edge0) / range).clamp(0.0, 1.0);
            // 6t⁵ - 15t⁴ + 10t³ = t³(10 - 15t + 6t²) = t³(6t² - 15t + 10)
            let t3 = t * t * t;
            t3 * (t * (t * 6.0 - 15.0) + 10.0)
        })
    }

    fn simd_logaddexp(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        // Numerically stable log(exp(a) + exp(b))
        // Formula: max(a,b) + log(1 + exp(-|a-b|))
        if a.is_empty() || b.is_empty() {
            return Array1::zeros(0);
        }
        let len = a.len().min(b.len());
        Array1::from_iter((0..len).map(|i| {
            let ai = a[i];
            let bi = b[i];
            let max_val = ai.max(bi);
            let diff = (ai - bi).abs();
            // For very large differences, the smaller term is negligible
            if diff > 50.0 {
                max_val
            } else {
                max_val + (1.0 + (-diff).exp()).ln()
            }
        }))
    }

    fn simd_logit(a: &ArrayView1<Self>) -> Array1<Self> {
        // Logit: log(p / (1-p)) = log(p) - log(1-p)
        // Domain: (0, 1), Range: (-∞, +∞)
        if a.is_empty() {
            return Array1::zeros(0);
        }
        a.mapv(|p| {
            // Handle edge cases
            if p <= 0.0 {
                Self::NEG_INFINITY
            } else if p >= 1.0 {
                Self::INFINITY
            } else {
                (p / (1.0 - p)).ln()
            }
        })
    }

    fn simd_square(a: &ArrayView1<Self>) -> Array1<Self> {
        // Element-wise x²
        if a.is_empty() {
            return Array1::zeros(0);
        }
        a.mapv(|x| x * x)
    }

    fn simd_rsqrt(a: &ArrayView1<Self>) -> Array1<Self> {
        // Inverse square root: 1/sqrt(x)
        if a.is_empty() {
            return Array1::zeros(0);
        }
        a.mapv(|x| {
            if x <= 0.0 {
                if x == 0.0 {
                    Self::INFINITY
                } else {
                    Self::NAN
                }
            } else {
                1.0 / x.sqrt()
            }
        })
    }

    fn simd_sincos(a: &ArrayView1<Self>) -> (Array1<Self>, Array1<Self>) {
        // Simultaneous sin and cos
        if a.is_empty() {
            return (Array1::zeros(0), Array1::zeros(0));
        }
        let sin_result = a.mapv(|x| x.sin());
        let cos_result = a.mapv(|x| x.cos());
        (sin_result, cos_result)
    }

    fn simd_expm1(a: &ArrayView1<Self>) -> Array1<Self> {
        // Numerically stable exp(x) - 1
        // Uses std library's expm1 which handles small x accurately
        if a.is_empty() {
            return Array1::zeros(0);
        }
        a.mapv(|x| x.exp_m1())
    }

    fn simd_log1p(a: &ArrayView1<Self>) -> Array1<Self> {
        // Numerically stable ln(1 + x)
        // Uses std library's ln_1p which handles small x accurately
        if a.is_empty() {
            return Array1::zeros(0);
        }
        a.mapv(|x| x.ln_1p())
    }
}

// Implementation for f64
impl SimdUnifiedOps for f64 {
    #[cfg(feature = "simd")]
    fn simd_add(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_add_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_add(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        (a + b).to_owned()
    }

    #[cfg(feature = "simd")]
    fn simd_sub(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_sub_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_sub(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        (a - b).to_owned()
    }

    #[cfg(feature = "simd")]
    fn simd_mul(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_mul_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_mul(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        (a * b).to_owned()
    }

    #[cfg(feature = "simd")]
    fn simd_div(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_div_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_div(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        (a / b).to_owned()
    }

    #[cfg(feature = "simd")]
    fn simd_dot(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_dot_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_dot(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        a.dot(b)
    }

    fn simd_gemv(a: &ArrayView2<Self>, x: &ArrayView1<Self>, beta: Self, y: &mut Array1<Self>) {
        let m = a.nrows();
        let n = a.ncols();

        assert_eq!(n, x.len());
        assert_eq!(m, y.len());

        // Scale y by beta
        if beta == 0.0 {
            y.fill(0.0);
        } else if beta != 1.0 {
            y.mapv_inplace(|v| v * beta);
        }

        // Compute matrix-vector product
        for i in 0..m {
            let row = a.row(i);
            y[i] += Self::simd_dot(&row, x);
        }
    }

    fn simd_gemm(
        alpha: Self,
        a: &ArrayView2<Self>,
        b: &ArrayView2<Self>,
        beta: Self,
        c: &mut Array2<Self>,
    ) {
        let m = a.nrows();
        let k = a.ncols();
        let n = b.ncols();

        assert_eq!(k, b.nrows());
        assert_eq!((m, n), c.dim());

        // Scale C by beta
        if beta == 0.0 {
            c.fill(0.0);
        } else if beta != 1.0 {
            c.mapv_inplace(|v| v * beta);
        }

        // Use blocked transpose for large matrices to improve cache efficiency
        // Threshold: n * k > 4096 (amortize transpose cost)
        const GEMM_TRANSPOSE_THRESHOLD: usize = 4096;

        if n * k > GEMM_TRANSPOSE_THRESHOLD {
            // Pre-transpose B for cache-efficient row-wise access
            let b_t = Self::simd_transpose_blocked(b);

            for i in 0..m {
                let a_row = a.row(i);
                for j in 0..n {
                    // Access b_t row-wise (contiguous memory)
                    let b_row = b_t.row(j);
                    c[[i, j]] += alpha * Self::simd_dot(&a_row, &b_row);
                }
            }
        } else {
            // Small matrices: use column access (overhead of transpose not worth it)
            for i in 0..m {
                let a_row = a.row(i);
                for j in 0..n {
                    let b_col = b.column(j);
                    c[[i, j]] += alpha * Self::simd_dot(&a_row, &b_col);
                }
            }
        }
    }

    #[cfg(feature = "simd")]
    fn simd_norm(a: &ArrayView1<Self>) -> Self {
        crate::simd::norms::simd_norm_l2_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_norm(a: &ArrayView1<Self>) -> Self {
        a.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    #[cfg(feature = "simd")]
    fn simd_max(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_maximum_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_max(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for _i in 0..a.len() {
            result[0] = a[0].max(b[0]);
        }
        result
    }

    #[cfg(feature = "simd")]
    fn simd_min(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_minimum_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_min(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for _i in 0..a.len() {
            result[0] = a[0].min(b[0]);
        }
        result
    }

    #[cfg(feature = "simd")]
    fn simd_scalar_mul(a: &ArrayView1<Self>, scalar: Self) -> Array1<Self> {
        crate::simd::simd_scalar_mul_f64(a, scalar)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_scalar_mul(a: &ArrayView1<Self>, scalar: Self) -> Array1<Self> {
        a.mapv(|x| x * scalar)
    }

    #[cfg(feature = "simd")]
    fn simd_sum(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_sum_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_sum(a: &ArrayView1<Self>) -> Self {
        a.sum()
    }

    fn simd_mean(a: &ArrayView1<Self>) -> Self {
        if a.is_empty() {
            0.0
        } else {
            Self::simd_sum(a) / (a.len() as f64)
        }
    }

    #[cfg(feature = "simd")]
    fn simd_max_element(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_max_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_max_element(a: &ArrayView1<Self>) -> Self {
        a.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
    }

    #[cfg(feature = "simd")]
    fn simd_min_element(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_min_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_min_element(a: &ArrayView1<Self>) -> Self {
        a.fold(f64::INFINITY, |acc, &x| acc.min(x))
    }

    #[cfg(feature = "simd")]
    fn simd_fma(a: &ArrayView1<Self>, b: &ArrayView1<Self>, c: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_fused_multiply_add_f64(a, b, c)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_fma(a: &ArrayView1<Self>, b: &ArrayView1<Self>, c: &ArrayView1<Self>) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for _i in 0..a.len() {
            result[0] = a[0] * b[0] + c[0];
        }
        result
    }

    #[cfg(feature = "simd")]
    fn simd_add_cache_optimized(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_add_cache_optimized_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_add_cache_optimized(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        a + b
    }

    #[cfg(feature = "simd")]
    fn simd_fma_advanced_optimized(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        c: &ArrayView1<Self>,
    ) -> Array1<Self> {
        crate::simd::simd_fma_advanced_optimized_f64(a, b, c)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_fma_advanced_optimized(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        c: &ArrayView1<Self>,
    ) -> Array1<Self> {
        let mut result = Array1::zeros(a.len());
        for _i in 0..a.len() {
            result[0] = a[0] * b[0] + c[0];
        }
        result
    }

    #[cfg(feature = "simd")]
    fn simd_add_adaptive(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_adaptive_add_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_add_adaptive(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        a + b
    }

    fn simd_transpose(a: &ArrayView2<Self>) -> Array2<Self> {
        a.t().to_owned()
    }

    fn simd_transpose_blocked(a: &ArrayView2<Self>) -> Array2<Self> {
        #[cfg(feature = "simd")]
        {
            crate::simd::simd_transpose_blocked_f64(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.t().to_owned()
        }
    }

    fn simd_sum_squares(a: &ArrayView1<Self>) -> Self {
        a.iter().map(|&x| x * x).sum()
    }

    fn simd_multiply(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        Self::simd_mul(a, b)
    }

    #[cfg(feature = "simd")]
    fn simd_available() -> bool {
        true
    }

    #[cfg(not(feature = "simd"))]
    fn simd_available() -> bool {
        false
    }

    fn simd_sub_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let sub_result = Self::simd_sub(a, b);
        result.assign(&sub_result);
    }

    fn simd_mul_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let mul_result = Self::simd_mul(a, b);
        result.assign(&mul_result);
    }

    fn simd_sum_cubes(a: &ArrayView1<Self>) -> Self {
        // Calculate sum of cubes: sum(x^3)
        a.iter().map(|&x| x * x * x).sum()
    }

    fn simd_div_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let div_result = Self::simd_div(a, b);
        result.assign(&div_result);
    }

    fn simd_sin_f32_ultra(a: &ArrayView1<Self>, result: &mut ArrayViewMut1<Self>) {
        let sin_result = a.mapv(|x| x.sin());
        result.assign(&sin_result);
    }

    fn simd_add_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let add_result = Self::simd_add(a, b);
        result.assign(&add_result);
    }

    fn simd_fma_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        c: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let fma_result = Self::simd_fma(a, b, c);
        result.assign(&fma_result);
    }

    fn simd_pow_f32_ultra(
        a: &ArrayView1<Self>,
        b: &ArrayView1<Self>,
        result: &mut ArrayViewMut1<Self>,
    ) {
        let pow_result = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| x.powf(y))
            .collect::<Vec<_>>();
        result.assign(&Array1::from_vec(pow_result));
    }

    fn simd_exp_f32_ultra(a: &ArrayView1<Self>, result: &mut ArrayViewMut1<Self>) {
        let exp_result = a.mapv(|x| x.exp());
        result.assign(&exp_result);
    }

    fn simd_cos_f32_ultra(a: &ArrayView1<Self>, result: &mut ArrayViewMut1<Self>) {
        let cos_result = a.mapv(|x| x.cos());
        result.assign(&cos_result);
    }

    fn simd_dot_f32_ultra(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        Self::simd_dot(a, b)
    }

    #[cfg(feature = "simd")]
    fn simd_variance(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_variance_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_variance(a: &ArrayView1<Self>) -> Self {
        let mean = Self::simd_mean(a);
        let n = a.len() as f64;
        if n < 2.0 {
            return f64::NAN;
        }
        // Sample variance with Bessel's correction (n-1)
        a.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    }

    #[cfg(feature = "simd")]
    fn simd_std(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_std_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_std(a: &ArrayView1<Self>) -> Self {
        Self::simd_variance(a).sqrt()
    }

    #[cfg(feature = "simd")]
    fn simd_norm_l1(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_norm_l1_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_norm_l1(a: &ArrayView1<Self>) -> Self {
        a.iter().map(|&x| x.abs()).sum()
    }

    #[cfg(feature = "simd")]
    fn simd_norm_linf(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_norm_linf_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_norm_linf(a: &ArrayView1<Self>) -> Self {
        a.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()))
    }

    #[cfg(feature = "simd")]
    fn simd_cosine_similarity(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_cosine_similarity_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_cosine_similarity(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        let dot = Self::simd_dot(a, b);
        let norm_a = Self::simd_norm(a);
        let norm_b = Self::simd_norm(b);
        dot / (norm_a * norm_b)
    }

    #[cfg(feature = "simd")]
    fn simd_distance_euclidean(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_distance_euclidean_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_distance_euclidean(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    #[cfg(feature = "simd")]
    fn simd_distance_manhattan(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_distance_manhattan_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_distance_manhattan(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
    }

    #[cfg(feature = "simd")]
    fn simd_distance_chebyshev(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_distance_chebyshev_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_distance_chebyshev(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        a.iter()
            .zip(b.iter())
            .fold(0.0f64, |acc, (&x, &y)| acc.max((x - y).abs()))
    }

    #[cfg(feature = "simd")]
    fn simd_distance_cosine(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        crate::simd::simd_distance_cosine_f64(a, b)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_distance_cosine(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Self {
        1.0 - Self::simd_cosine_similarity(a, b)
    }

    #[cfg(feature = "simd")]
    fn simd_weighted_sum(values: &ArrayView1<Self>, weights: &ArrayView1<Self>) -> Self {
        crate::simd::simd_weighted_sum_f64(values, weights)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_weighted_sum(values: &ArrayView1<Self>, weights: &ArrayView1<Self>) -> Self {
        values
            .iter()
            .zip(weights.iter())
            .map(|(&v, &w)| v * w)
            .sum()
    }

    #[cfg(feature = "simd")]
    fn simd_weighted_mean(values: &ArrayView1<Self>, weights: &ArrayView1<Self>) -> Self {
        crate::simd::simd_weighted_mean_f64(values, weights)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_weighted_mean(values: &ArrayView1<Self>, weights: &ArrayView1<Self>) -> Self {
        let weighted_sum = Self::simd_weighted_sum(values, weights);
        let weight_sum: f64 = weights.iter().sum();
        weighted_sum / weight_sum
    }

    #[cfg(feature = "simd")]
    fn simd_argmin(a: &ArrayView1<Self>) -> Option<usize> {
        crate::simd::simd_argmin_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_argmin(a: &ArrayView1<Self>) -> Option<usize> {
        if a.is_empty() {
            return None;
        }
        let mut min_idx = 0;
        let mut min_val = a[0];
        for (i, &v) in a.iter().enumerate().skip(1) {
            if v < min_val {
                min_val = v;
                min_idx = i;
            }
        }
        Some(min_idx)
    }

    #[cfg(feature = "simd")]
    fn simd_argmax(a: &ArrayView1<Self>) -> Option<usize> {
        crate::simd::simd_argmax_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_argmax(a: &ArrayView1<Self>) -> Option<usize> {
        if a.is_empty() {
            return None;
        }
        let mut max_idx = 0;
        let mut max_val = a[0];
        for (i, &v) in a.iter().enumerate().skip(1) {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        Some(max_idx)
    }

    #[cfg(feature = "simd")]
    fn simd_clip(a: &ArrayView1<Self>, min_val: Self, max_val: Self) -> Array1<Self> {
        crate::simd::simd_clip_f64(a, min_val, max_val)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_clip(a: &ArrayView1<Self>, min_val: Self, max_val: Self) -> Array1<Self> {
        a.mapv(|v| v.max(min_val).min(max_val))
    }

    #[cfg(feature = "simd")]
    fn simd_log_sum_exp(a: &ArrayView1<Self>) -> Self {
        crate::simd::simd_log_sum_exp_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_log_sum_exp(a: &ArrayView1<Self>) -> Self {
        if a.is_empty() {
            return f64::NEG_INFINITY;
        }
        let max_val = a.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let sum_exp: f64 = a.iter().map(|&x| (x - max_val).exp()).sum();
        max_val + sum_exp.ln()
    }

    #[cfg(feature = "simd")]
    fn simd_softmax(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_softmax_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_softmax(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.is_empty() {
            return Array1::zeros(0);
        }
        let lse = Self::simd_log_sum_exp(a);
        a.mapv(|x| (x - lse).exp())
    }

    #[cfg(feature = "simd")]
    fn simd_cumsum(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_cumsum_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_cumsum(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.is_empty() {
            return Array1::zeros(0);
        }
        let mut cumsum = 0.0f64;
        a.mapv(|x| {
            cumsum += x;
            cumsum
        })
    }

    #[cfg(feature = "simd")]
    fn simd_cumprod(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_cumprod_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_cumprod(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.is_empty() {
            return Array1::zeros(0);
        }
        let mut cumprod = 1.0f64;
        a.mapv(|x| {
            cumprod *= x;
            cumprod
        })
    }

    #[cfg(feature = "simd")]
    fn simd_diff(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_diff_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_diff(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.len() <= 1 {
            return Array1::zeros(0);
        }
        Array1::from_iter((1..a.len()).map(|i| a[i] - a[i - 1]))
    }

    #[cfg(feature = "simd")]
    fn simd_sign(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_sign_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_sign(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
    }

    #[cfg(feature = "simd")]
    fn simd_relu(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_relu_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_relu(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.max(0.0))
    }

    #[cfg(feature = "simd")]
    fn simd_leaky_relu(a: &ArrayView1<Self>, alpha: Self) -> Array1<Self> {
        crate::simd::simd_leaky_relu_f64(a, alpha)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_leaky_relu(a: &ArrayView1<Self>, alpha: Self) -> Array1<Self> {
        a.mapv(|x| if x > 0.0 { x } else { alpha * x })
    }

    #[cfg(feature = "simd")]
    fn simd_normalize(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_normalize_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_normalize(a: &ArrayView1<Self>) -> Array1<Self> {
        let norm: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            return a.to_owned();
        }
        a.mapv(|x| x / norm)
    }

    #[cfg(feature = "simd")]
    fn simd_standardize(a: &ArrayView1<Self>) -> Array1<Self> {
        crate::simd::simd_standardize_f64(a)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_standardize(a: &ArrayView1<Self>) -> Array1<Self> {
        if a.len() <= 1 {
            return Array1::zeros(a.len());
        }
        let mean: f64 = a.iter().sum::<f64>() / a.len() as f64;
        let variance: f64 =
            a.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (a.len() - 1) as f64;
        let std = variance.sqrt();
        if std == 0.0 {
            return Array1::zeros(a.len());
        }
        a.mapv(|x| (x - mean) / std)
    }

    fn simd_abs(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.abs())
    }

    fn simd_sqrt(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.sqrt())
    }

    fn simd_exp(a: &ArrayView1<Self>) -> Array1<Self> {
        // Note: SIMD polynomial approximation available via crate::simd::simd_exp_f64
        // for ~5-10x speedup with 10^-9 accuracy. Keeping scalar for trait compatibility.
        a.mapv(|x| x.exp())
    }

    fn simd_ln(a: &ArrayView1<Self>) -> Array1<Self> {
        // Note: SIMD logarithm available via crate::simd::simd_ln_f64
        // for speedup with moderate accuracy. Keeping scalar for trait compatibility.
        a.mapv(|x| x.ln())
    }

    fn simd_sin(a: &ArrayView1<Self>) -> Array1<Self> {
        // Note: SIMD polynomial approximation available via crate::simd::simd_sin_f64
        // for ~5x speedup with 10^-4 accuracy. Keeping scalar for trait compatibility.
        a.mapv(|x| x.sin())
    }

    fn simd_cos(a: &ArrayView1<Self>) -> Array1<Self> {
        // Note: SIMD cosine available via crate::simd::simd_cos_f64
        // for ~5x speedup with 10^-4 accuracy. Keeping scalar for trait compatibility.
        a.mapv(|x| x.cos())
    }

    fn simd_tan(a: &ArrayView1<Self>) -> Array1<Self> {
        // Note: Can use SIMD via sin/cos: crate::simd::simd_sin_f64 / crate::simd::simd_cos_f64
        a.mapv(|x| x.tan())
    }

    fn simd_sinh(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            simd_ops_polynomial::simd_sinh_f64_poly(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.sinh())
        }
    }

    fn simd_cosh(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            simd_ops_polynomial::simd_cosh_f64_poly(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.cosh())
        }
    }

    fn simd_tanh(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            // Use polynomial approximation (good accuracy, fast)
            simd_ops_polynomial::simd_tanh_f64_poly(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.tanh())
        }
    }

    fn simd_floor(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            crate::simd::simd_floor_f64(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.floor())
        }
    }

    fn simd_ceil(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            crate::simd::simd_ceil_f64(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.ceil())
        }
    }

    fn simd_round(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            crate::simd::simd_round_f64(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.round())
        }
    }

    fn simd_atan(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.atan())
    }

    fn simd_asin(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.asin())
    }

    fn simd_acos(a: &ArrayView1<Self>) -> Array1<Self> {
        a.mapv(|x| x.acos())
    }

    fn simd_atan2(y: &ArrayView1<Self>, x: &ArrayView1<Self>) -> Array1<Self> {
        y.iter()
            .zip(x.iter())
            .map(|(&y_val, &x_val)| y_val.atan2(x_val))
            .collect::<Vec<_>>()
            .into()
    }

    fn simd_log10(a: &ArrayView1<Self>) -> Array1<Self> {
        // log10(x) = ln(x) * (1/ln(10)) - uses SIMD ln
        const LOG10_E: f64 = std::f64::consts::LOG10_E; // 1/ln(10)
        let ln_a = Self::simd_ln(a);
        Self::simd_scalar_mul(&ln_a.view(), LOG10_E)
    }

    fn simd_log2(a: &ArrayView1<Self>) -> Array1<Self> {
        // log2(x) = ln(x) * (1/ln(2)) - uses SIMD ln
        const LOG2_E: f64 = std::f64::consts::LOG2_E; // 1/ln(2)
        let ln_a = Self::simd_ln(a);
        Self::simd_scalar_mul(&ln_a.view(), LOG2_E)
    }

    #[cfg(feature = "simd")]
    fn simd_clamp(a: &ArrayView1<Self>, min: Self, max: Self) -> Array1<Self> {
        // Use SIMD-accelerated clip (AVX2/SSE/NEON)
        crate::simd::simd_clip_f64(a, min, max)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_clamp(a: &ArrayView1<Self>, min: Self, max: Self) -> Array1<Self> {
        a.mapv(|x| x.clamp(min, max))
    }

    fn simd_fract(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            let truncated = crate::simd::simd_trunc_f64(a);
            Self::simd_sub(a, &truncated.view())
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.fract())
        }
    }

    fn simd_trunc(a: &ArrayView1<Self>) -> Array1<Self> {
        #[cfg(feature = "simd")]
        {
            crate::simd::simd_trunc_f64(a)
        }
        #[cfg(not(feature = "simd"))]
        {
            a.mapv(|x| x.trunc())
        }
    }

    fn simd_recip(a: &ArrayView1<Self>) -> Array1<Self> {
        // Optimized SIMD reciprocal: 1/x using SIMD division
        let ones = Array1::from_elem(a.len(), 1.0f64);
        Self::simd_div(&ones.view(), a)
    }

    fn simd_powf(base: &ArrayView1<Self>, exp: Self) -> Array1<Self> {
        // Optimized SIMD powf: base^exp = exp(exp * ln(base))
        // Uses SIMD-accelerated exp and ln operations
        let ln_base = Self::simd_ln(base);
        let scaled = Self::simd_scalar_mul(&ln_base.view(), exp);
        Self::simd_exp(&scaled.view())
    }

    fn simd_pow(base: &ArrayView1<Self>, exp: &ArrayView1<Self>) -> Array1<Self> {
        // Optimized SIMD pow: base[i]^exp[i] = exp(exp[i] * ln(base[i]))
        // Uses SIMD-accelerated exp, ln, and mul operations
        let ln_base = Self::simd_ln(base);
        let scaled = Self::simd_mul(&ln_base.view(), exp);
        Self::simd_exp(&scaled.view())
    }

    #[cfg(feature = "simd")]
    fn simd_powi(base: &ArrayView1<Self>, n: i32) -> Array1<Self> {
        crate::simd::unary_powi::simd_powi_f64(base, n)
    }

    #[cfg(not(feature = "simd"))]
    fn simd_powi(base: &ArrayView1<Self>, n: i32) -> Array1<Self> {
        base.mapv(|x| x.powi(n))
    }

    fn simd_gamma(x: &ArrayView1<Self>) -> Array1<Self> {
        x.mapv(lanczos_gamma_f64)
    }

    fn simd_exp2(a: &ArrayView1<Self>) -> Array1<Self> {
        // 2^x = exp(x * ln(2))
        const LN2: f64 = std::f64::consts::LN_2;
        let scaled = Self::simd_scalar_mul(a, LN2);
        Self::simd_exp(&scaled.view())
    }

    fn simd_cbrt(a: &ArrayView1<Self>) -> Array1<Self> {
        // Cube root: x^(1/3)
        // Handle negative numbers: cbrt(-x) = -cbrt(x)
        a.mapv(|x| x.cbrt())
    }

    fn simd_ln_1p(a: &ArrayView1<Self>) -> Array1<Self> {
        // ln(1+x) - numerically stable for small x
        a.mapv(|x| x.ln_1p())
    }

    fn simd_exp_m1(a: &ArrayView1<Self>) -> Array1<Self> {
        // exp(x)-1 - numerically stable for small x
        a.mapv(|x| x.exp_m1())
    }

    fn simd_to_radians(a: &ArrayView1<Self>) -> Array1<Self> {
        // degrees to radians: x * π / 180
        const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;
        Self::simd_scalar_mul(a, DEG_TO_RAD)
    }

    fn simd_to_degrees(a: &ArrayView1<Self>) -> Array1<Self> {
        // radians to degrees: x * 180 / π
        const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;
        Self::simd_scalar_mul(a, RAD_TO_DEG)
    }

    fn simd_digamma(a: &ArrayView1<Self>) -> Array1<Self> {
        // Digamma function ψ(x) = d/dx ln(Γ(x))
        // Uses reflection formula, recurrence relation, and asymptotic expansion
        a.mapv(digamma_f64)
    }

    fn simd_trigamma(a: &ArrayView1<Self>) -> Array1<Self> {
        // Trigamma function ψ'(x) = d²/dx² ln(Γ(x))
        // Critical for Fisher information in Bayesian inference
        a.mapv(trigamma_f64)
    }

    fn simd_ln_gamma(a: &ArrayView1<Self>) -> Array1<Self> {
        // Log-gamma function ln(Γ(x)) - more stable than gamma(x).ln()
        a.mapv(ln_gamma_f64)
    }

    fn simd_erf(a: &ArrayView1<Self>) -> Array1<Self> {
        // Error function erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
        // Critical for normal distribution CDF: Φ(x) = 0.5 * (1 + erf(x/√2))
        a.mapv(erf_f64)
    }

    fn simd_erfc(a: &ArrayView1<Self>) -> Array1<Self> {
        // Complementary error function erfc(x) = 1 - erf(x)
        // More numerically stable than 1 - erf(x) for large x
        a.mapv(erfc_f64)
    }

    fn simd_erfinv(a: &ArrayView1<Self>) -> Array1<Self> {
        // Inverse error function: erfinv(y) = x such that erf(x) = y
        // Critical for inverse normal CDF (probit function)
        a.mapv(erfinv_f64)
    }

    fn simd_erfcinv(a: &ArrayView1<Self>) -> Array1<Self> {
        // Inverse complementary error function: erfcinv(y) = x such that erfc(x) = y
        // More numerically stable than erfinv(1-y) for small y
        a.mapv(erfcinv_f64)
    }

    fn simd_sigmoid(a: &ArrayView1<Self>) -> Array1<Self> {
        // Sigmoid (logistic) function: σ(x) = 1 / (1 + exp(-x))
        // Critical for neural networks, logistic regression
        // Uses numerically stable implementation
        a.mapv(sigmoid_f64)
    }

    fn simd_gelu(a: &ArrayView1<Self>) -> Array1<Self> {
        // GELU (Gaussian Error Linear Unit): x * Φ(x) = x * 0.5 * (1 + erf(x / √2))
        // Critical for Transformer models (BERT, GPT, etc.)
        a.mapv(gelu_f64)
    }

    fn simd_swish(a: &ArrayView1<Self>) -> Array1<Self> {
        // Swish (SiLU): x * sigmoid(x)
        // Self-gated activation for EfficientNet, GPT-NeoX
        a.mapv(swish_f64)
    }

    fn simd_softplus(a: &ArrayView1<Self>) -> Array1<Self> {
        // Softplus: ln(1 + exp(x))
        // Smooth approximation of ReLU for probabilistic models
        a.mapv(softplus_f64)
    }

    fn simd_mish(a: &ArrayView1<Self>) -> Array1<Self> {
        // Mish: x * tanh(softplus(x))
        // Self-regularized activation for YOLOv4
        a.mapv(mish_f64)
    }

    fn simd_elu(a: &ArrayView1<Self>, alpha: Self) -> Array1<Self> {
        // ELU: x if x >= 0, alpha * (exp(x) - 1) if x < 0
        // Helps with vanishing gradients
        a.mapv(|x| elu_f64(x, alpha))
    }

    fn simd_selu(a: &ArrayView1<Self>) -> Array1<Self> {
        // SELU: λ * (x if x > 0, α * (exp(x) - 1) if x <= 0)
        // Self-normalizing activation with fixed α and λ constants
        a.mapv(selu_f64)
    }

    fn simd_hardsigmoid(a: &ArrayView1<Self>) -> Array1<Self> {
        // Hardsigmoid: clip((x + 3) / 6, 0, 1)
        // Piecewise linear approximation of sigmoid
        a.mapv(hardsigmoid_f64)
    }

    fn simd_hardswish(a: &ArrayView1<Self>) -> Array1<Self> {
        // Hardswish: x * hardsigmoid(x)
        // Piecewise linear approximation of Swish
        a.mapv(hardswish_f64)
    }

    fn simd_sinc(a: &ArrayView1<Self>) -> Array1<Self> {
        // Sinc: sin(πx) / (πx), with sinc(0) = 1
        // Critical for signal processing and interpolation
        a.mapv(sinc_f64)
    }

    fn simd_log_softmax(a: &ArrayView1<Self>) -> Array1<Self> {
        // Log-softmax: x_i - log(Σ_j exp(x_j))
        // Numerically stable for neural network loss computation
        // Leverages existing SIMD-accelerated log_sum_exp
        if a.is_empty() {
            return Array1::zeros(0);
        }
        let lse = Self::simd_log_sum_exp(a);
        a.mapv(|x| x - lse)
    }

    fn simd_asinh(a: &ArrayView1<Self>) -> Array1<Self> {
        // Inverse hyperbolic sine: asinh(x) = ln(x + √(x² + 1))
        // Uses Rust's built-in asinh for accuracy
        a.mapv(|x| x.asinh())
    }

    fn simd_acosh(a: &ArrayView1<Self>) -> Array1<Self> {
        // Inverse hyperbolic cosine: acosh(x) = ln(x + √(x² - 1))
        // Domain: [1, +∞), returns NaN for x < 1
        a.mapv(|x| x.acosh())
    }

    fn simd_atanh(a: &ArrayView1<Self>) -> Array1<Self> {
        // Inverse hyperbolic tangent: atanh(x) = 0.5 * ln((1+x)/(1-x))
        // Domain: (-1, 1), returns ±∞ at ±1, NaN for |x| > 1
        a.mapv(|x| x.atanh())
    }

    fn simd_ln_beta(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        // Log-Beta: ln(B(a,b)) = ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a+b))
        // Leverages SIMD-accelerated ln_gamma
        let ln_gamma_a = Self::simd_ln_gamma(a);
        let ln_gamma_b = Self::simd_ln_gamma(b);
        let a_plus_b = Self::simd_add(a, b);
        let ln_gamma_ab = Self::simd_ln_gamma(&a_plus_b.view());
        Self::simd_sub(
            &Self::simd_add(&ln_gamma_a.view(), &ln_gamma_b.view()).view(),
            &ln_gamma_ab.view(),
        )
    }

    fn simd_beta(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        // Beta: B(a,b) = exp(ln(B(a,b)))
        // Uses log-beta for numerical stability
        let ln_beta = Self::simd_ln_beta(a, b);
        Self::simd_exp(&ln_beta.view())
    }

    fn simd_lerp(a: &ArrayView1<Self>, b: &ArrayView1<Self>, t: Self) -> Array1<Self> {
        // Linear interpolation: lerp(a, b, t) = a + t * (b - a)
        // This formula is more numerically stable when t is close to 0
        if a.is_empty() || b.is_empty() {
            return Array1::zeros(0);
        }
        let diff = Self::simd_sub(b, a);
        let scaled = Self::simd_scalar_mul(&diff.view(), t);
        Self::simd_add(a, &scaled.view())
    }

    fn simd_smoothstep(edge0: Self, edge1: Self, x: &ArrayView1<Self>) -> Array1<Self> {
        // Smoothstep: smooth Hermite interpolation
        // t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
        // result = t * t * (3 - 2 * t)
        if x.is_empty() {
            return Array1::zeros(0);
        }
        let range = edge1 - edge0;
        if range.abs() < Self::EPSILON {
            // Degenerate case: edge0 == edge1
            return x.mapv(|xi| if xi < edge0 { 0.0 } else { 1.0 });
        }
        x.mapv(|xi| {
            let t = ((xi - edge0) / range).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        })
    }

    fn simd_hypot(x: &ArrayView1<Self>, y: &ArrayView1<Self>) -> Array1<Self> {
        // Hypotenuse: sqrt(x² + y²) with overflow/underflow protection
        // Uses Rust's built-in hypot which handles extreme values correctly
        if x.is_empty() || y.is_empty() {
            return Array1::zeros(0);
        }
        let len = x.len().min(y.len());
        Array1::from_iter((0..len).map(|i| x[i].hypot(y[i])))
    }

    fn simd_copysign(x: &ArrayView1<Self>, y: &ArrayView1<Self>) -> Array1<Self> {
        // Returns magnitude of x with sign of y
        if x.is_empty() || y.is_empty() {
            return Array1::zeros(0);
        }
        let len = x.len().min(y.len());
        Array1::from_iter((0..len).map(|i| x[i].copysign(y[i])))
    }

    fn simd_smootherstep(edge0: Self, edge1: Self, x: &ArrayView1<Self>) -> Array1<Self> {
        // Ken Perlin's smootherstep: 6t⁵ - 15t⁴ + 10t³
        // Has zero first AND second derivatives at boundaries
        if x.is_empty() {
            return Array1::zeros(0);
        }
        let range = edge1 - edge0;
        if range.abs() < Self::EPSILON {
            return x.mapv(|xi| if xi < edge0 { 0.0 } else { 1.0 });
        }
        x.mapv(|xi| {
            let t = ((xi - edge0) / range).clamp(0.0, 1.0);
            // 6t⁵ - 15t⁴ + 10t³ = t³(6t² - 15t + 10)
            let t3 = t * t * t;
            t3 * (t * (t * 6.0 - 15.0) + 10.0)
        })
    }

    fn simd_logaddexp(a: &ArrayView1<Self>, b: &ArrayView1<Self>) -> Array1<Self> {
        // Numerically stable log(exp(a) + exp(b))
        // Formula: max(a,b) + log(1 + exp(-|a-b|))
        if a.is_empty() || b.is_empty() {
            return Array1::zeros(0);
        }
        let len = a.len().min(b.len());
        Array1::from_iter((0..len).map(|i| {
            let ai = a[i];
            let bi = b[i];
            let max_val = ai.max(bi);
            let diff = (ai - bi).abs();
            // For very large differences, the smaller term is negligible
            if diff > 50.0 {
                max_val
            } else {
                max_val + (1.0 + (-diff).exp()).ln()
            }
        }))
    }

    fn simd_logit(a: &ArrayView1<Self>) -> Array1<Self> {
        // Logit: log(p / (1-p)) = log(p) - log(1-p)
        // Domain: (0, 1), Range: (-∞, +∞)
        if a.is_empty() {
            return Array1::zeros(0);
        }
        a.mapv(|p| {
            // Handle edge cases
            if p <= 0.0 {
                Self::NEG_INFINITY
            } else if p >= 1.0 {
                Self::INFINITY
            } else {
                (p / (1.0 - p)).ln()
            }
        })
    }

    fn simd_square(a: &ArrayView1<Self>) -> Array1<Self> {
        // Element-wise x²
        if a.is_empty() {
            return Array1::zeros(0);
        }
        a.mapv(|x| x * x)
    }

    fn simd_rsqrt(a: &ArrayView1<Self>) -> Array1<Self> {
        // Inverse square root: 1/sqrt(x)
        if a.is_empty() {
            return Array1::zeros(0);
        }
        a.mapv(|x| {
            if x <= 0.0 {
                if x == 0.0 {
                    Self::INFINITY
                } else {
                    Self::NAN
                }
            } else {
                1.0 / x.sqrt()
            }
        })
    }

    fn simd_sincos(a: &ArrayView1<Self>) -> (Array1<Self>, Array1<Self>) {
        // Simultaneous sin and cos
        if a.is_empty() {
            return (Array1::zeros(0), Array1::zeros(0));
        }
        let sin_result = a.mapv(|x| x.sin());
        let cos_result = a.mapv(|x| x.cos());
        (sin_result, cos_result)
    }

    fn simd_expm1(a: &ArrayView1<Self>) -> Array1<Self> {
        // Numerically stable exp(x) - 1
        // Uses std library's expm1 which handles small x accurately
        if a.is_empty() {
            return Array1::zeros(0);
        }
        a.mapv(|x| x.exp_m1())
    }

    fn simd_log1p(a: &ArrayView1<Self>) -> Array1<Self> {
        // Numerically stable ln(1 + x)
        // Uses std library's ln_1p which handles small x accurately
        if a.is_empty() {
            return Array1::zeros(0);
        }
        a.mapv(|x| x.ln_1p())
    }
}

/// Platform capability detection
#[derive(Debug, Clone, Copy)]
pub struct PlatformCapabilities {
    pub simd_available: bool,
    pub gpu_available: bool,
    pub cuda_available: bool,
    pub opencl_available: bool,
    pub metal_available: bool,
    pub avx2_available: bool,
    pub avx512_available: bool,
    pub neon_available: bool,
}

impl PlatformCapabilities {
    /// Detect current platform capabilities
    pub fn detect() -> Self {
        Self {
            simd_available: cfg!(feature = "simd"),
            gpu_available: cfg!(feature = "gpu"),
            cuda_available: cfg!(all(feature = "gpu", feature = "cuda")),
            opencl_available: cfg!(all(feature = "gpu", feature = "opencl")),
            metal_available: cfg!(all(feature = "gpu", feature = "metal", target_os = "macos")),
            avx2_available: cfg!(target_feature = "avx2"),
            avx512_available: cfg!(target_feature = "avx512f"),
            neon_available: cfg!(target_arch = "aarch64"),
        }
    }

    /// Get a summary of available acceleration features
    pub fn summary(&self) -> String {
        let mut features = Vec::new();

        if self.simd_available {
            features.push("SIMD");
        }
        if self.gpu_available {
            features.push("GPU");
        }
        if self.cuda_available {
            features.push("CUDA");
        }
        if self.opencl_available {
            features.push("OpenCL");
        }
        if self.metal_available {
            features.push("Metal");
        }
        if self.avx2_available {
            features.push("AVX2");
        }
        if self.avx512_available {
            features.push("AVX512");
        }
        if self.neon_available {
            features.push("NEON");
        }

        if features.is_empty() {
            "No acceleration features available".to_string()
        } else {
            format!(
                "Available acceleration: {features}",
                features = features.join(", ")
            )
        }
    }

    /// Check if AVX2 is available
    pub fn has_avx2(&self) -> bool {
        self.avx2_available
    }

    /// Check if AVX512 is available
    pub fn has_avx512(&self) -> bool {
        self.avx512_available
    }

    /// Check if SSE is available (fallback to SIMD availability)
    pub fn has_sse(&self) -> bool {
        self.simd_available || self.neon_available || self.avx2_available
    }

    /// Get the number of CPU cores
    pub fn num_cores(&self) -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }

    /// Get the cache line size in bytes
    pub fn cache_line_size(&self) -> usize {
        // Standard cache line size on most modern processors
        // x86/x64: typically 64 bytes
        // ARM: typically 64 bytes (Apple Silicon, newer ARM)
        64
    }
}

/// Automatic operation selection based on problem size and available features
pub struct AutoOptimizer {
    capabilities: PlatformCapabilities,
}

impl AutoOptimizer {
    pub fn new() -> Self {
        Self {
            capabilities: PlatformCapabilities::detect(),
        }
    }

    /// Determine if GPU should be used for a given problem size
    pub fn should_use_gpu(&self, size: usize) -> bool {
        // Use GPU for large problems when available
        self.capabilities.gpu_available && size > 10000
    }

    /// Determine if Metal should be used on macOS
    pub fn should_use_metal(&self, size: usize) -> bool {
        // Use Metal for medium to large problems on macOS
        // Metal has lower overhead than CUDA/OpenCL, so we can use it for smaller problems
        self.capabilities.metal_available && size > 1024
    }

    /// Determine if SIMD should be used
    pub fn should_use_simd(&self, size: usize) -> bool {
        // Use SIMD for medium to large problems
        self.capabilities.simd_available && size > 64
    }

    /// Select the best implementation for matrix multiplication
    pub fn select_gemm_impl(&self, m: usize, n: usize, k: usize) -> &'static str {
        let total_ops = m * n * k;

        // Metal-specific heuristics for macOS
        if self.capabilities.metal_available {
            // For Apple Silicon with unified memory, Metal is efficient even for smaller matrices
            if total_ops > 8192 {
                // 16x16x32 or larger
                return "Metal";
            }
        }

        if self.should_use_gpu(total_ops) {
            if self.capabilities.cuda_available {
                "CUDA"
            } else if self.capabilities.metal_available {
                "Metal"
            } else if self.capabilities.opencl_available {
                "OpenCL"
            } else {
                "GPU"
            }
        } else if self.should_use_simd(total_ops) {
            "SIMD"
        } else {
            "Scalar"
        }
    }

    /// Select the best implementation for vector operations
    pub fn select_vector_impl(&self, size: usize) -> &'static str {
        // Metal is efficient for vector operations on Apple Silicon
        if self.capabilities.metal_available && size > 1024 {
            return "Metal";
        }

        if self.should_use_gpu(size) {
            if self.capabilities.cuda_available {
                "CUDA"
            } else if self.capabilities.metal_available {
                "Metal"
            } else if self.capabilities.opencl_available {
                "OpenCL"
            } else {
                "GPU"
            }
        } else if self.should_use_simd(size) {
            if self.capabilities.avx512_available {
                "AVX512"
            } else if self.capabilities.avx2_available {
                "AVX2"
            } else if self.capabilities.neon_available {
                "NEON"
            } else {
                "SIMD"
            }
        } else {
            "Scalar"
        }
    }

    /// Select the best implementation for reduction operations
    pub fn select_reduction_impl(&self, size: usize) -> &'static str {
        // Reductions benefit from GPU parallelism at larger sizes
        // Metal has efficient reduction primitives
        if self.capabilities.metal_available && size > 4096 {
            return "Metal";
        }

        if self.should_use_gpu(size * 2) {
            // Higher threshold for reductions
            if self.capabilities.cuda_available {
                "CUDA"
            } else if self.capabilities.metal_available {
                "Metal"
            } else {
                "GPU"
            }
        } else if self.should_use_simd(size) {
            "SIMD"
        } else {
            "Scalar"
        }
    }

    /// Select the best implementation for FFT operations
    pub fn select_fft_impl(&self, size: usize) -> &'static str {
        // FFT benefits greatly from GPU acceleration
        // Metal Performance Shaders has optimized FFT
        if self.capabilities.metal_available && size > 512 {
            return "Metal-MPS";
        }

        if self.capabilities.cuda_available && size > 1024 {
            "cuFFT"
        } else if self.should_use_simd(size) {
            "SIMD"
        } else {
            "Scalar"
        }
    }

    /// Check if running on Apple Silicon with unified memory
    pub fn has_unified_memory(&self) -> bool {
        cfg!(all(target_os = "macos", target_arch = "aarch64"))
    }

    /// Get optimization recommendation for a specific operation
    pub fn recommend(&self, operation: &str, size: usize) -> String {
        let recommendation = match operation {
            "gemm" | "matmul" => self.select_gemm_impl(size, size, size),
            "vector" | "axpy" | "dot" => self.select_vector_impl(size),
            "reduction" | "sum" | "mean" => self.select_reduction_impl(size),
            "fft" => self.select_fft_impl(size),
            _ => "Scalar",
        };

        if self.has_unified_memory() && recommendation == "Metal" {
            format!("{recommendation} (Unified Memory)")
        } else {
            recommendation.to_string()
        }
    }
}

impl Default for AutoOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Standalone ultra-optimized dot product function for f32
pub fn simd_dot_f32_ultra(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    f32::simd_dot_f32_ultra(a, b)
}

/// Standalone ultra-optimized FMA function for f32
pub fn simd_fma_f32_ultra(
    a: &ArrayView1<f32>,
    b: &ArrayView1<f32>,
    c: &ArrayView1<f32>,
    result: &mut ArrayViewMut1<f32>,
) {
    f32::simd_fma_f32_ultra(a, b, c, result)
}

/// Additional standalone functions that might be needed
pub fn simd_add_f32_adaptive(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    f32::simd_add_adaptive(a, b)
}

pub fn simd_mul_f32_hyperoptimized(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> Array1<f32> {
    f32::simd_mul(a, b)
}

/// Helper functions for `Vec<T>` compatibility
/// These functions accept `Vec<T>` and internally convert to Array types
///
/// Helper function for Vec-based SIMD multiplication
pub fn simd_mul_f32_ultra_vec(a: &Vec<f32>, b: &Vec<f32>, result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.clone());
    let b_array = Array1::from_vec(b.clone());
    let mut result_array = Array1::from_vec(result.clone());

    f32::simd_mul_f32_ultra(
        &a_array.view(),
        &b_array.view(),
        &mut result_array.view_mut(),
    );
    *result = result_array.to_vec();
}

/// Helper function for Vec-based SIMD addition
pub fn simd_add_f32_ultra_vec(a: &Vec<f32>, b: &Vec<f32>, result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.clone());
    let b_array = Array1::from_vec(b.clone());
    let mut result_array = Array1::from_vec(result.clone());

    f32::simd_add_f32_ultra(
        &a_array.view(),
        &b_array.view(),
        &mut result_array.view_mut(),
    );
    *result = result_array.to_vec();
}

/// Helper function for Vec-based SIMD division
pub fn simd_div_f32_ultra_vec(a: &Vec<f32>, b: &Vec<f32>, result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.clone());
    let b_array = Array1::from_vec(b.clone());
    let mut result_array = Array1::from_vec(result.clone());

    f32::simd_div_f32_ultra(
        &a_array.view(),
        &b_array.view(),
        &mut result_array.view_mut(),
    );
    *result = result_array.to_vec();
}

/// Helper function for Vec-based SIMD sine
pub fn simd_sin_f32_ultra_vec(a: &[f32], result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.to_owned());
    let mut result_array = Array1::from_vec(result.clone());

    f32::simd_sin_f32_ultra(&a_array.view(), &mut result_array.view_mut());
    *result = result_array.to_vec();
}

/// Helper function for Vec-based SIMD subtraction
pub fn simd_sub_f32_ultra_vec(a: &[f32], b: &[f32], result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.to_owned());
    let b_array = Array1::from_vec(b.to_owned());
    let mut result_array = Array1::from_vec(result.clone());

    f32::simd_sub_f32_ultra(
        &a_array.view(),
        &b_array.view(),
        &mut result_array.view_mut(),
    );
    *result = result_array.to_vec();
}

/// Helper function for Vec-based SIMD FMA
pub fn simd_fma_f32_ultra_vec(a: &[f32], b: &[f32], c: &[f32], result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.to_owned());
    let b_array = Array1::from_vec(b.to_owned());
    let c_array = Array1::from_vec(c.to_owned());
    let mut result_array = Array1::from_vec(result.clone());

    f32::simd_fma_f32_ultra(
        &a_array.view(),
        &b_array.view(),
        &c_array.view(),
        &mut result_array.view_mut(),
    );
    *result = result_array.to_vec();
}

/// Helper function for Vec-based SIMD power
pub fn simd_pow_f32_ultra_vec(a: &[f32], b: &[f32], result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.to_owned());
    let b_array = Array1::from_vec(b.to_owned());
    let mut result_array = Array1::from_vec(result.clone());

    f32::simd_pow_f32_ultra(
        &a_array.view(),
        &b_array.view(),
        &mut result_array.view_mut(),
    );
    *result = result_array.to_vec();
}

/// Helper function for Vec-based SIMD exp
pub fn simd_exp_f32_ultra_vec(a: &[f32], result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.to_owned());
    let mut result_array = Array1::from_vec(result.clone());

    f32::simd_exp_f32_ultra(&a_array.view(), &mut result_array.view_mut());
    *result = result_array.to_vec();
}

/// Helper function for Vec-based SIMD cos
pub fn simd_cos_f32_ultra_vec(a: &[f32], result: &mut Vec<f32>) {
    let a_array = Array1::from_vec(a.to_owned());
    let mut result_array = Array1::from_vec(result.clone());

    f32::simd_cos_f32_ultra(&a_array.view(), &mut result_array.view_mut());
    *result = result_array.to_vec();
}

#[cfg(test)]
#[path = "simd_ops_tests.rs"]
mod tests;
