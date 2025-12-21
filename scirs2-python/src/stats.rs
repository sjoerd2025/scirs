//! Python bindings for scirs2-stats
//!
//! This module provides Python bindings for statistical functions.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

// NumPy types for Python array interface (scirs2-numpy with native ndarray 0.17)
use scirs2_numpy::{PyArray1, PyArray2, PyArrayMethods};

// ndarray types from scirs2-core
use scirs2_core::{Array1, Array2, ndarray::ArrayView1};

// Note: mean, std, var, median, percentile use optimized implementations below

// Import statistical test functions
use scirs2_stats::tests::ttest::{ttest_1samp, ttest_ind, ttest_rel, Alternative};
use scirs2_stats::tests::anova::{one_way_anova, tukey_hsd};
use scirs2_stats::tests::normality::{shapiro_wilk, anderson_darling, dagostino_k2, ks_2samp};
use scirs2_stats::tests::chi2_test::{chi2_gof, chi2_independence, chi2_yates};
use scirs2_stats::tests::nonparametric::{wilcoxon, mann_whitney, kruskal_wallis, friedman};
use scirs2_stats::tests::homogeneity::{levene, bartlett, brown_forsythe};
use scirs2_stats::{pearsonr, spearmanr, kendalltau};
use scirs2_stats::contingency::{fisher_exact, odds_ratio, relative_risk};
use scirs2_stats::regression::{linregress, polyfit};
use scirs2_stats::{mean_abs_deviation, median_abs_deviation, data_range, coef_variation, gini_coefficient};
use scirs2_stats::{boxplot_stats, quartiles, quintiles, deciles, winsorized_mean, winsorized_variance, QuantileInterpolation};
use scirs2_stats::{sem_simd, percentile_range_simd};
use scirs2_stats::{skewness_simd, kurtosis_simd};
use scirs2_stats::{pearson_r_simd, covariance_simd};
use scirs2_stats::{moment_simd, mean_simd, std_simd, variance_simd};
use scirs2_stats::{entropy, kl_divergence, cross_entropy};
use scirs2_stats::{weighted_mean, moment};
use scirs2_stats::distribution_characteristics::{skewness_ci, kurtosis_ci};

// Import distribution types
use scirs2_stats::distributions::normal::Normal as RustNormal;
use scirs2_stats::distributions::binomial::Binomial as RustBinomial;
use scirs2_stats::distributions::poisson::Poisson as RustPoisson;
use scirs2_stats::distributions::exponential::Exponential as RustExponential;
use scirs2_stats::distributions::uniform::Uniform as RustUniform;
use scirs2_stats::distributions::beta::Beta as RustBeta;
use scirs2_stats::distributions::gamma::Gamma as RustGamma;
use scirs2_stats::distributions::chi_square::ChiSquare as RustChiSquare;
use scirs2_stats::distributions::student_t::StudentT as RustStudentT;
use scirs2_stats::distributions::cauchy::Cauchy as RustCauchy;
use scirs2_stats::distributions::f::F as RustF;
use scirs2_stats::distributions::lognormal::Lognormal as RustLognormal;
use scirs2_stats::distributions::weibull::Weibull as RustWeibull;
use scirs2_stats::distributions::laplace::Laplace as RustLaplace;
use scirs2_stats::distributions::logistic::Logistic as RustLogistic;
use scirs2_stats::distributions::pareto::Pareto as RustPareto;
use scirs2_stats::distributions::geometric::Geometric as RustGeometric;
use scirs2_stats::traits::{DiscreteDistribution, ContinuousDistribution};

// ========================================
// DESCRIPTIVE STATISTICS
// ========================================

/// Compute descriptive statistics - uses optimized implementations
/// Returns dict with mean, std, var, min, max, median, count
#[pyfunction]
fn describe_py(py: Python, data: &Bound<'_, PyArray1<f64>>) -> PyResult<Py<PyAny>> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();

    if n == 0 {
        return Err(PyRuntimeError::new_err("Empty array provided"));
    }

    let slice = arr.as_slice().expect("Operation failed");

    // Optimized mean calculation
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;
    let chunks = slice.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        sum0 += chunk[0] + chunk[4];
        sum1 += chunk[1] + chunk[5];
        sum2 += chunk[2] + chunk[6];
        sum3 += chunk[3] + chunk[7];
    }
    let mut sum = sum0 + sum1 + sum2 + sum3;
    for &val in remainder {
        sum += val;
    }
    let mean_val = sum / n as f64;

    // Optimized variance/std calculation
    let mut sq0 = 0.0f64;
    let mut sq1 = 0.0f64;
    let mut sq2 = 0.0f64;
    let mut sq3 = 0.0f64;
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    let chunks = slice.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let d0 = chunk[0] - mean_val;
        let d1 = chunk[1] - mean_val;
        let d2 = chunk[2] - mean_val;
        let d3 = chunk[3] - mean_val;
        let d4 = chunk[4] - mean_val;
        let d5 = chunk[5] - mean_val;
        let d6 = chunk[6] - mean_val;
        let d7 = chunk[7] - mean_val;
        sq0 += d0 * d0 + d4 * d4;
        sq1 += d1 * d1 + d5 * d5;
        sq2 += d2 * d2 + d6 * d6;
        sq3 += d3 * d3 + d7 * d7;
        min_val = min_val.min(chunk[0]).min(chunk[1]).min(chunk[2]).min(chunk[3])
                         .min(chunk[4]).min(chunk[5]).min(chunk[6]).min(chunk[7]);
        max_val = max_val.max(chunk[0]).max(chunk[1]).max(chunk[2]).max(chunk[3])
                         .max(chunk[4]).max(chunk[5]).max(chunk[6]).max(chunk[7]);
    }
    for &val in remainder {
        let d = val - mean_val;
        sq0 += d * d;
        min_val = min_val.min(val);
        max_val = max_val.max(val);
    }
    let sq_sum = sq0 + sq1 + sq2 + sq3;
    let var_val = if n > 1 { sq_sum / (n - 1) as f64 } else { 0.0 };
    let std_val = var_val.sqrt();

    // Optimized median calculation
    let mut vec: Vec<f64> = arr.iter().cloned().collect();
    let median_val = if n % 2 == 1 {
        let mid = n / 2;
        let (_, val, _) = vec.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        *val
    } else {
        let mid = n / 2;
        let (lower, val_at_mid, _) = vec.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let val_mid = *val_at_mid;
        let val_mid_minus_1 = lower
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(val_mid);
        (val_mid_minus_1 + val_mid) / 2.0
    };

    let dict = PyDict::new(py);
    dict.set_item("mean", mean_val)?;
    dict.set_item("std", std_val)?;
    dict.set_item("var", var_val)?;
    dict.set_item("min", min_val)?;
    dict.set_item("max", max_val)?;
    dict.set_item("median", median_val)?;
    dict.set_item("count", n)?;

    Ok(dict.into())
}

/// Calculate mean - optimized with 8-way unrolling and multiple accumulators
#[pyfunction]
fn mean_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();

    if n == 0 {
        return Err(PyRuntimeError::new_err("Empty array provided"));
    }

    let slice = arr.as_slice().expect("Operation failed");

    // Use 4 independent accumulators to maximize instruction-level parallelism
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;

    let chunks = slice.chunks_exact(8);
    let remainder = chunks.remainder();

    // Process 8 elements at a time with 4 independent accumulators
    for chunk in chunks {
        sum0 += chunk[0] + chunk[4];
        sum1 += chunk[1] + chunk[5];
        sum2 += chunk[2] + chunk[6];
        sum3 += chunk[3] + chunk[7];
    }

    // Combine accumulators
    let mut sum = sum0 + sum1 + sum2 + sum3;

    // Handle remaining elements
    for &val in remainder {
        sum += val;
    }

    Ok(sum / n as f64)
}

/// Calculate standard deviation - optimized two-pass with multi-accumulator
#[pyfunction]
#[pyo3(signature = (data, ddof=0))]
fn std_py(data: &Bound<'_, PyArray1<f64>>, ddof: usize) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();

    if n == 0 {
        return Err(PyRuntimeError::new_err("Empty array provided"));
    }

    if n <= ddof {
        return Err(PyRuntimeError::new_err("Not enough data points for given ddof"));
    }

    let slice = arr.as_slice().expect("Operation failed");

    // Pass 1: Calculate mean with 4 independent accumulators
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;

    let chunks = slice.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        sum0 += chunk[0] + chunk[4];
        sum1 += chunk[1] + chunk[5];
        sum2 += chunk[2] + chunk[6];
        sum3 += chunk[3] + chunk[7];
    }
    let mut sum = sum0 + sum1 + sum2 + sum3;
    for &val in remainder {
        sum += val;
    }
    let mean = sum / n as f64;

    // Pass 2: Calculate sum of squared deviations with 4 independent accumulators
    let mut sq0 = 0.0f64;
    let mut sq1 = 0.0f64;
    let mut sq2 = 0.0f64;
    let mut sq3 = 0.0f64;

    let chunks = slice.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let d0 = chunk[0] - mean;
        let d1 = chunk[1] - mean;
        let d2 = chunk[2] - mean;
        let d3 = chunk[3] - mean;
        let d4 = chunk[4] - mean;
        let d5 = chunk[5] - mean;
        let d6 = chunk[6] - mean;
        let d7 = chunk[7] - mean;
        sq0 += d0 * d0 + d4 * d4;
        sq1 += d1 * d1 + d5 * d5;
        sq2 += d2 * d2 + d6 * d6;
        sq3 += d3 * d3 + d7 * d7;
    }
    let mut sq_sum = sq0 + sq1 + sq2 + sq3;
    for &val in remainder {
        let d = val - mean;
        sq_sum += d * d;
    }

    let variance = sq_sum / (n - ddof) as f64;
    Ok(variance.sqrt())
}

/// Calculate variance - optimized two-pass with multi-accumulator
#[pyfunction]
#[pyo3(signature = (data, ddof=0))]
fn var_py(data: &Bound<'_, PyArray1<f64>>, ddof: usize) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();

    if n == 0 {
        return Err(PyRuntimeError::new_err("Empty array provided"));
    }

    if n <= ddof {
        return Err(PyRuntimeError::new_err("Not enough data points for given ddof"));
    }

    let slice = arr.as_slice().expect("Operation failed");

    // Pass 1: Calculate mean with 4 independent accumulators
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;

    let chunks = slice.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        sum0 += chunk[0] + chunk[4];
        sum1 += chunk[1] + chunk[5];
        sum2 += chunk[2] + chunk[6];
        sum3 += chunk[3] + chunk[7];
    }
    let mut sum = sum0 + sum1 + sum2 + sum3;
    for &val in remainder {
        sum += val;
    }
    let mean = sum / n as f64;

    // Pass 2: Calculate sum of squared deviations with 4 independent accumulators
    let mut sq0 = 0.0f64;
    let mut sq1 = 0.0f64;
    let mut sq2 = 0.0f64;
    let mut sq3 = 0.0f64;

    let chunks = slice.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let d0 = chunk[0] - mean;
        let d1 = chunk[1] - mean;
        let d2 = chunk[2] - mean;
        let d3 = chunk[3] - mean;
        let d4 = chunk[4] - mean;
        let d5 = chunk[5] - mean;
        let d6 = chunk[6] - mean;
        let d7 = chunk[7] - mean;
        sq0 += d0 * d0 + d4 * d4;
        sq1 += d1 * d1 + d5 * d5;
        sq2 += d2 * d2 + d6 * d6;
        sq3 += d3 * d3 + d7 * d7;
    }
    let mut sq_sum = sq0 + sq1 + sq2 + sq3;
    for &val in remainder {
        let d = val - mean;
        sq_sum += d * d;
    }

    Ok(sq_sum / (n - ddof) as f64)
}

/// Calculate percentile using optimized partial sort
/// q: percentile value (0-100)
#[pyfunction]
fn percentile_py(data: &Bound<'_, PyArray1<f64>>, q: f64) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();

    if n == 0 {
        return Err(PyRuntimeError::new_err("Empty array provided"));
    }

    if !(0.0..=100.0).contains(&q) {
        return Err(PyRuntimeError::new_err("Percentile must be between 0 and 100"));
    }

    let mut vec: Vec<f64> = arr.iter().cloned().collect();

    // Calculate the index using linear interpolation (NumPy default method)
    let q_norm = q / 100.0;
    let virtual_index = q_norm * (n - 1) as f64;
    let i = virtual_index.floor() as usize;
    let fraction = virtual_index - i as f64;

    if fraction == 0.0 || i >= n - 1 {
        // Exact index or at the end
        let idx = i.min(n - 1);
        let (_, val, _) = vec.select_nth_unstable_by(idx, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(*val)
    } else {
        // Need linear interpolation between i and i+1
        // First get element at position i+1
        let (lower, val_i1, _) = vec.select_nth_unstable_by(i + 1, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let val_upper = *val_i1;

        // Now find max in lower partition (which is element at position i)
        let val_lower = lower
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(val_upper);

        // Linear interpolation
        Ok(val_lower + fraction * (val_upper - val_lower))
    }
}

/// Calculate correlation coefficient
#[pyfunction]
fn correlation_py(x: &Bound<'_, PyArray1<f64>>, y: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let x_binding = x.readonly();
    let x_arr = x_binding.as_array();
    let y_binding = y.readonly();
    let y_arr = y_binding.as_array();

    let (r, _p) = pearsonr(&x_arr, &y_arr, "two-sided")
        .map_err(|e| PyRuntimeError::new_err(format!("Correlation failed: {}", e)))?;

    Ok(r)
}

/// Calculate covariance
#[pyfunction]
#[pyo3(signature = (x, y, ddof=1))]
fn covariance_py(
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    ddof: usize,
) -> PyResult<f64> {
    let x_binding = x.readonly();
    let x_arr = x_binding.as_array();
    let y_binding = y.readonly();
    let y_arr = y_binding.as_array();

    covariance_simd(&x_arr, &y_arr, ddof)
        .map_err(|e| PyRuntimeError::new_err(format!("Covariance failed: {}", e)))
}

/// Calculate median using optimized partial sort (O(n) instead of O(n log n))
#[pyfunction]
fn median_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();

    if n == 0 {
        return Err(PyRuntimeError::new_err("Empty array provided"));
    }

    // Use partial sort (quickselect) for O(n) median calculation
    let mut vec: Vec<f64> = arr.iter().cloned().collect();

    if n % 2 == 1 {
        // Odd length: return the middle element
        let mid = n / 2;
        let (_, median_val, _) = vec.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(*median_val)
    } else {
        // Even length: return average of two middle elements
        let mid = n / 2;
        // First, find the element at position mid
        let (lower, val_at_mid, _) = vec.select_nth_unstable_by(mid, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let val_mid = *val_at_mid;
        // The lower partition contains elements <= val_mid, find the max (which is at mid-1)
        // We need to find the max in the lower partition
        let val_mid_minus_1 = lower
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(val_mid);
        Ok((val_mid_minus_1 + val_mid) / 2.0)
    }
}

/// Calculate interquartile range (IQR) using optimized partial sort
#[pyfunction]
fn iqr_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();

    if n == 0 {
        return Err(PyRuntimeError::new_err("Empty array provided"));
    }

    let mut vec: Vec<f64> = arr.iter().cloned().collect();

    // Helper function to get percentile value using partial sort
    let get_percentile = |vec: &mut [f64], q: f64| -> f64 {
        let virtual_index = q * (n - 1) as f64;
        let i = virtual_index.floor() as usize;
        let fraction = virtual_index - i as f64;

        if fraction == 0.0 || i >= n - 1 {
            let idx = i.min(n - 1);
            let (_, val, _) = vec.select_nth_unstable_by(idx, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            *val
        } else {
            let (lower, val_i1, _) = vec.select_nth_unstable_by(i + 1, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            let val_upper = *val_i1;
            let val_lower = lower
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .copied()
                .unwrap_or(val_upper);
            val_lower + fraction * (val_upper - val_lower)
        }
    };

    // Get Q75 first (modifies vec), then Q25
    // Since Q75 index > Q25 index, after Q75 select, the lower partition contains Q25
    let q75 = get_percentile(&mut vec, 0.75);

    // For Q25, we can use a fresh copy since the array is modified
    let mut vec2: Vec<f64> = arr.iter().cloned().collect();
    let q25 = get_percentile(&mut vec2, 0.25);

    Ok(q75 - q25)
}

// ========================================
// STATISTICAL TESTS
// ========================================

/// One-sample t-test
///
/// Test whether the mean of a sample is different from a given value.
///
/// Parameters:
/// - data: Input data
/// - popmean: Population mean for null hypothesis
/// - alternative: "two-sided", "less", or "greater"
#[pyfunction]
#[pyo3(signature = (data, popmean, alternative="two-sided"))]
fn ttest_1samp_py(
    py: Python,
    data: &Bound<'_, PyArray1<f64>>,
    popmean: f64,
    alternative: &str,
) -> PyResult<Py<PyAny>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    let alt = match alternative.to_lowercase().as_str() {
        "two-sided" | "two_sided" => Alternative::TwoSided,
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Invalid alternative: {}. Use 'two-sided', 'less', or 'greater'",
                alternative
            )));
        }
    };

    let result = ttest_1samp(&arr.view(), popmean, alt, "omit")
        .map_err(|e| PyRuntimeError::new_err(format!("t-test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("pvalue", result.pvalue)?;
    dict.set_item("df", result.df)?;

    Ok(dict.into())
}

/// Two-sample independent t-test
///
/// Test whether two independent samples have different means.
///
/// Parameters:
/// - a: First sample
/// - b: Second sample
/// - equal_var: If true, perform standard t-test assuming equal variance
/// - alternative: "two-sided", "less", or "greater"
#[pyfunction]
#[pyo3(signature = (a, b, equal_var=true, alternative="two-sided"))]
fn ttest_ind_py(
    py: Python,
    a: &Bound<'_, PyArray1<f64>>,
    b: &Bound<'_, PyArray1<f64>>,
    equal_var: bool,
    alternative: &str,
) -> PyResult<Py<PyAny>> {
    let a_binding = a.readonly();
    let a_arr = a_binding.as_array();
    let b_binding = b.readonly();
    let b_arr = b_binding.as_array();

    let alt = match alternative.to_lowercase().as_str() {
        "two-sided" | "two_sided" => Alternative::TwoSided,
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Invalid alternative: {}. Use 'two-sided', 'less', or 'greater'",
                alternative
            )));
        }
    };

    let result = ttest_ind(&a_arr.view(), &b_arr.view(), equal_var, alt, "omit")
        .map_err(|e| PyRuntimeError::new_err(format!("t-test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("pvalue", result.pvalue)?;
    dict.set_item("df", result.df)?;

    Ok(dict.into())
}

/// Paired (related samples) t-test.
///
/// Tests whether the means of two related/paired samples differ.
/// This is a parametric test for paired observations (e.g., before/after measurements).
///
/// Parameters:
///     a: First array of observations (e.g., before treatment)
///     b: Second array of observations (e.g., after treatment)
///         Must be same length as a (paired observations)
///     alternative: Alternative hypothesis: "two-sided" (default), "less", or "greater"
///
/// Returns:
///     Dictionary with 'statistic' (t), 'pvalue', and 'df' (degrees of freedom)
#[pyfunction]
#[pyo3(signature = (a, b, alternative="two-sided"))]
fn ttest_rel_py(
    py: Python,
    a: &Bound<'_, PyArray1<f64>>,
    b: &Bound<'_, PyArray1<f64>>,
    alternative: &str,
) -> PyResult<Py<PyAny>> {
    let a_binding = a.readonly();
    let a_arr = a_binding.as_array();
    let b_binding = b.readonly();
    let b_arr = b_binding.as_array();

    let alt = match alternative.to_lowercase().as_str() {
        "two-sided" | "two_sided" => Alternative::TwoSided,
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        _ => {
            return Err(PyRuntimeError::new_err(format!(
                "Invalid alternative: {}. Use 'two-sided', 'less', or 'greater'",
                alternative
            )));
        }
    };

    let result = ttest_rel(&a_arr.view(), &b_arr.view(), alt, "omit")
        .map_err(|e| PyRuntimeError::new_err(format!("Paired t-test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("pvalue", result.pvalue)?;
    dict.set_item("df", result.df)?;

    Ok(dict.into())
}

/// Skewness of the data
#[pyfunction]
fn skew_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len() as f64;

    if n < 3.0 {
        return Err(PyRuntimeError::new_err("Need at least 3 data points for skewness"));
    }

    let mean: f64 = arr.iter().sum::<f64>() / n;
    let m2: f64 = arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let m3: f64 = arr.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;

    if m2 == 0.0 {
        return Ok(0.0);
    }

    Ok(m3 / m2.powf(1.5))
}

/// Kurtosis of the data
#[pyfunction]
fn kurtosis_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len() as f64;

    if n < 4.0 {
        return Err(PyRuntimeError::new_err("Need at least 4 data points for kurtosis"));
    }

    let mean: f64 = arr.iter().sum::<f64>() / n;
    let m2: f64 = arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let m4: f64 = arr.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;

    if m2 == 0.0 {
        return Ok(0.0);
    }

    // Excess kurtosis (Fisher's definition)
    Ok(m4 / m2.powi(2) - 3.0)
}

/// Mode of the data (most frequent value)
#[pyfunction]
fn mode_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();

    if arr.is_empty() {
        return Err(PyRuntimeError::new_err("Cannot compute mode of empty array"));
    }

    // For continuous data, use a simple approach: find the most common rounded value
    // This is a simplified version; real mode for continuous data uses KDE
    let mut values: Vec<f64> = arr.iter().copied().collect();
    values.sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));

    // Find mode using histogram-like approach
    let mut max_count = 0;
    let mut mode = values[0];
    let mut current_count = 1;
    let mut current_val = values[0];

    for &v in values.iter().skip(1) {
        if (v - current_val).abs() < 1e-10 {
            current_count += 1;
        } else {
            if current_count > max_count {
                max_count = current_count;
                mode = current_val;
            }
            current_count = 1;
            current_val = v;
        }
    }

    if current_count > max_count {
        mode = current_val;
    }

    Ok(mode)
}

/// Geometric mean
#[pyfunction]
fn gmean_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();

    if n == 0 {
        return Err(PyRuntimeError::new_err("Cannot compute geometric mean of empty array"));
    }

    // Check for non-positive values
    if arr.iter().any(|&x| x <= 0.0) {
        return Err(PyRuntimeError::new_err("Geometric mean requires all positive values"));
    }

    let log_sum: f64 = arr.iter().map(|x| x.ln()).sum();
    Ok((log_sum / n as f64).exp())
}

/// Harmonic mean
#[pyfunction]
fn hmean_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len();

    if n == 0 {
        return Err(PyRuntimeError::new_err("Cannot compute harmonic mean of empty array"));
    }

    // Check for non-positive values
    if arr.iter().any(|&x| x <= 0.0) {
        return Err(PyRuntimeError::new_err("Harmonic mean requires all positive values"));
    }

    let reciprocal_sum: f64 = arr.iter().map(|x| 1.0 / x).sum();
    Ok(n as f64 / reciprocal_sum)
}

/// Z-score normalization
#[pyfunction]
fn zscore_py(
    py: Python,
    data: &Bound<'_, PyArray1<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();
    let n = arr.len() as f64;

    let mean: f64 = arr.iter().sum::<f64>() / n;
    let std: f64 = (arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt();

    if std == 0.0 {
        return Err(PyRuntimeError::new_err("Standard deviation is zero, cannot compute z-scores"));
    }

    let zscores: Vec<f64> = arr.iter().map(|x| (x - mean) / std).collect();

    Ok(PyArray1::from_vec(py, zscores).into())
}

// ========================================
// DISPERSION AND VARIABILITY MEASURES
// ========================================

/// Mean absolute deviation
///
/// Calculates the average absolute deviation from the mean (or specified center).
///
/// Parameters:
/// - data: Input data array
/// - center: Optional center value (default: None, uses mean)
///
/// Returns:
/// - Mean absolute deviation value
#[pyfunction]
#[pyo3(signature = (data, center=None))]
fn mean_abs_deviation_py(
    data: &Bound<'_, PyArray1<f64>>,
    center: Option<f64>,
) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();

    mean_abs_deviation(&arr.view(), center)
        .map_err(|e| PyRuntimeError::new_err(format!("Mean absolute deviation failed: {}", e)))
}

/// Median absolute deviation
///
/// Calculates the median absolute deviation from the median.
/// This is a robust measure of variability.
///
/// Parameters:
/// - data: Input data array
/// - center: Optional center value (default: None, uses median)
/// - scale: Optional scale factor (default: None, uses 1.0)
///
/// Returns:
/// - Median absolute deviation value
#[pyfunction]
#[pyo3(signature = (data, center=None, scale=None))]
fn median_abs_deviation_py(
    data: &Bound<'_, PyArray1<f64>>,
    center: Option<f64>,
    scale: Option<f64>,
) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();

    median_abs_deviation(&arr.view(), center, scale)
        .map_err(|e| PyRuntimeError::new_err(format!("Median absolute deviation failed: {}", e)))
}

/// Data range
///
/// Calculates the range of the data (maximum - minimum).
///
/// Parameters:
/// - data: Input data array
///
/// Returns:
/// - Range value (max - min)
#[pyfunction]
fn data_range_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();

    data_range(&arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Data range calculation failed: {}", e)))
}

/// Coefficient of variation
///
/// Calculates the coefficient of variation (standard deviation / mean).
/// This is a unitless measure of relative variability.
///
/// Parameters:
/// - data: Input data array
/// - ddof: Degrees of freedom for standard deviation (default: 1)
///
/// Returns:
/// - Coefficient of variation
#[pyfunction]
#[pyo3(signature = (data, ddof=1))]
fn coef_variation_py(
    data: &Bound<'_, PyArray1<f64>>,
    ddof: usize,
) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();

    coef_variation(&arr.view(), ddof)
        .map_err(|e| PyRuntimeError::new_err(format!("Coefficient of variation calculation failed: {}", e)))
}

/// Gini coefficient
///
/// Calculates the Gini coefficient, a measure of inequality.
/// Values range from 0 (perfect equality) to 1 (maximum inequality).
///
/// Parameters:
/// - data: Input data array (must be non-negative)
///
/// Returns:
/// - Gini coefficient
#[pyfunction]
fn gini_coefficient_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();

    gini_coefficient(&arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Gini coefficient calculation failed: {}", e)))
}

// ========================================
// QUANTILE AND ROBUST STATISTICS
// ========================================

/// Compute boxplot statistics (five-number summary + outliers)
#[pyfunction]
#[pyo3(signature = (data, whis=1.5))]
fn boxplot_stats_py(
    py: Python,
    data: &Bound<'_, PyArray1<f64>>,
    whis: f64,
) -> PyResult<Py<PyAny>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    let (q1, median, q3, whislo, whishi, outliers) = boxplot_stats::<f64>(
        &arr.view(),
        Some(whis),
        QuantileInterpolation::Linear,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Boxplot stats calculation failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("q1", q1)?;
    dict.set_item("median", median)?;
    dict.set_item("q3", q3)?;
    dict.set_item("whislo", whislo)?;
    dict.set_item("whishi", whishi)?;
    dict.set_item("outliers", outliers)?;

    Ok(dict.into())
}

/// Compute quartiles (Q1, Q2, Q3)
#[pyfunction]
fn quartiles_py(
    py: Python<'_>,
    data: &Bound<'_, PyArray1<f64>>,
) -> PyResult<Py<PyArray1<f64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    let result = quartiles::<f64>(&arr.view(), QuantileInterpolation::Linear)
        .map_err(|e| PyRuntimeError::new_err(format!("Quartiles calculation failed: {}", e)))?;

    Ok(PyArray1::from_vec(py, result.to_vec()).into())
}

/// Compute winsorized mean (robust mean)
#[pyfunction]
#[pyo3(signature = (data, limits=0.1))]
fn winsorized_mean_py(
    data: &Bound<'_, PyArray1<f64>>,
    limits: f64,
) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();

    winsorized_mean::<f64>(&arr.view(), limits)
        .map_err(|e| PyRuntimeError::new_err(format!("Winsorized mean calculation failed: {}", e)))
}

/// Compute winsorized variance (robust variance)
#[pyfunction]
#[pyo3(signature = (data, limits=0.1, ddof=1))]
fn winsorized_variance_py(
    data: &Bound<'_, PyArray1<f64>>,
    limits: f64,
    ddof: usize,
) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();

    winsorized_variance::<f64>(&arr.view(), limits, ddof)
        .map_err(|e| PyRuntimeError::new_err(format!("Winsorized variance calculation failed: {}", e)))
}

// ========================================
// INFORMATION THEORY AND ADVANCED STATISTICS
// ========================================

/// Compute Shannon entropy of discrete data
#[pyfunction]
#[pyo3(signature = (data, base=None))]
fn entropy_py(
    data: &Bound<'_, PyArray1<i64>>,
    base: Option<f64>,
) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();

    entropy::<i64>(&arr.view(), base)
        .map_err(|e| PyRuntimeError::new_err(format!("Entropy calculation failed: {}", e)))
}

/// Compute Kullback-Leibler divergence between two probability distributions
#[pyfunction]
fn kl_divergence_py(
    p: &Bound<'_, PyArray1<f64>>,
    q: &Bound<'_, PyArray1<f64>>,
) -> PyResult<f64> {
    let p_binding = p.readonly();
    let p_arr = p_binding.as_array();

    let q_binding = q.readonly();
    let q_arr = q_binding.as_array();

    kl_divergence::<f64>(&p_arr.view(), &q_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("KL divergence calculation failed: {}", e)))
}

/// Compute cross-entropy between two probability distributions
#[pyfunction]
fn cross_entropy_py(
    p: &Bound<'_, PyArray1<f64>>,
    q: &Bound<'_, PyArray1<f64>>,
) -> PyResult<f64> {
    let p_binding = p.readonly();
    let p_arr = p_binding.as_array();

    let q_binding = q.readonly();
    let q_arr = q_binding.as_array();

    cross_entropy::<f64>(&p_arr.view(), &q_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Cross-entropy calculation failed: {}", e)))
}

/// Compute weighted mean
#[pyfunction]
fn weighted_mean_py(
    data: &Bound<'_, PyArray1<f64>>,
    weights: &Bound<'_, PyArray1<f64>>,
) -> PyResult<f64> {
    let data_binding = data.readonly();
    let data_arr = data_binding.as_array();

    let weights_binding = weights.readonly();
    let weights_arr = weights_binding.as_array();

    weighted_mean::<f64>(&data_arr.view(), &weights_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Weighted mean calculation failed: {}", e)))
}

/// Compute statistical moment
#[pyfunction]
#[pyo3(signature = (data, order, center=true))]
fn moment_py(
    data: &Bound<'_, PyArray1<f64>>,
    order: usize,
    center: bool,
) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();

    moment::<f64>(&arr.view(), order, center, None)
        .map_err(|e| PyRuntimeError::new_err(format!("Moment calculation failed: {}", e)))
}

// ========================================
// QUANTILE ANALYSIS (ADDITIONAL FUNCTIONS)
// ========================================

/// Compute quintiles (20th, 40th, 60th, 80th percentiles) of a dataset
#[pyfunction]
fn quintiles_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<Py<PyArray1<f64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    let result = quintiles::<f64>(&arr.view(), QuantileInterpolation::Linear)
        .map_err(|e| PyRuntimeError::new_err(format!("Quintiles calculation failed: {}", e)))?;

    Ok(PyArray1::from_vec(data.py(), result.to_vec()).into())
}

/// Compute skewness with bootstrap confidence interval
#[pyfunction]
#[pyo3(signature = (data, bias=false, confidence=0.95, n_bootstrap=1000, seed=None))]
fn skewness_ci_py(
    py: Python,
    data: &Bound<'_, PyArray1<f64>>,
    bias: bool,
    confidence: f64,
    n_bootstrap: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    let result = skewness_ci::<f64>(
        &arr.view(),
        bias,
        Some(confidence),
        Some(n_bootstrap),
        seed,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Skewness CI calculation failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("estimate", result.estimate)?;
    dict.set_item("lower", result.lower)?;
    dict.set_item("upper", result.upper)?;
    dict.set_item("confidence", result.confidence)?;

    Ok(dict.into())
}

/// Compute kurtosis with bootstrap confidence interval
#[pyfunction]
#[pyo3(signature = (data, fisher=true, bias=false, confidence=0.95, n_bootstrap=1000, seed=None))]
fn kurtosis_ci_py(
    py: Python,
    data: &Bound<'_, PyArray1<f64>>,
    fisher: bool,
    bias: bool,
    confidence: f64,
    n_bootstrap: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    let result = kurtosis_ci::<f64>(
        &arr.view(),
        fisher,
        bias,
        Some(confidence),
        Some(n_bootstrap),
        seed,
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Kurtosis CI calculation failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("estimate", result.estimate)?;
    dict.set_item("lower", result.lower)?;
    dict.set_item("upper", result.upper)?;
    dict.set_item("confidence", result.confidence)?;

    Ok(dict.into())
}

/// Compute deciles (10th, 20th, ..., 90th percentiles) of a dataset
#[pyfunction]
fn deciles_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<Py<PyArray1<f64>>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    let result = deciles::<f64>(&arr.view(), QuantileInterpolation::Linear)
        .map_err(|e| PyRuntimeError::new_err(format!("Deciles calculation failed: {}", e)))?;

    Ok(PyArray1::from_vec(data.py(), result.to_vec()).into())
}

/// Compute standard error of the mean (SEM)
#[pyfunction]
#[pyo3(signature = (data, ddof=1))]
fn sem_py(data: &Bound<'_, PyArray1<f64>>, ddof: usize) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();

    sem_simd::<f64, _>(&arr, ddof)
        .map_err(|e| PyRuntimeError::new_err(format!("SEM calculation failed: {}", e)))
}

/// Compute the range between two percentiles
#[pyfunction]
#[pyo3(signature = (data, lower_pct, upper_pct, interpolation="linear"))]
fn percentile_range_py(
    data: &Bound<'_, PyArray1<f64>>,
    lower_pct: f64,
    upper_pct: f64,
    interpolation: &str,
) -> PyResult<f64> {
    let binding = data.readonly();
    let mut arr = binding.as_array().to_owned();

    percentile_range_simd::<f64, _>(&mut arr, lower_pct, upper_pct, interpolation)
        .map_err(|e| PyRuntimeError::new_err(format!("Percentile range calculation failed: {}", e)))
}

/// Compute SIMD-optimized skewness (third standardized moment)
#[pyfunction]
#[pyo3(signature = (data, bias=false))]
fn skewness_simd_py(data: &Bound<'_, PyArray1<f64>>, bias: bool) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();

    skewness_simd::<f64, _>(&arr.view(), bias)
        .map_err(|e| PyRuntimeError::new_err(format!("SIMD skewness calculation failed: {}", e)))
}

/// Compute SIMD-optimized kurtosis (fourth standardized moment)
#[pyfunction]
#[pyo3(signature = (data, fisher=true, bias=false))]
fn kurtosis_simd_py(data: &Bound<'_, PyArray1<f64>>, fisher: bool, bias: bool) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();

    kurtosis_simd::<f64, _>(&arr.view(), fisher, bias)
        .map_err(|e| PyRuntimeError::new_err(format!("SIMD kurtosis calculation failed: {}", e)))
}

/// Compute SIMD-optimized Pearson correlation coefficient
#[pyfunction]
fn pearson_r_simd_py(
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
) -> PyResult<f64> {
    let x_binding = x.readonly();
    let y_binding = y.readonly();
    let x_arr = x_binding.as_array();
    let y_arr = y_binding.as_array();

    pearson_r_simd::<f64, _>(&x_arr.view(), &y_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("SIMD Pearson correlation calculation failed: {}", e)))
}

/// Compute SIMD-optimized covariance
#[pyfunction]
#[pyo3(signature = (x, y, ddof=1))]
fn covariance_simd_py(
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    ddof: usize,
) -> PyResult<f64> {
    let x_binding = x.readonly();
    let y_binding = y.readonly();
    let x_arr = x_binding.as_array();
    let y_arr = y_binding.as_array();

    covariance_simd::<f64, _>(&x_arr.view(), &y_arr.view(), ddof)
        .map_err(|e| PyRuntimeError::new_err(format!("SIMD covariance calculation failed: {}", e)))
}

/// Compute SIMD-optimized nth statistical moment
#[pyfunction]
#[pyo3(signature = (data, moment_order, center=true))]
fn moment_simd_py(data: &Bound<'_, PyArray1<f64>>, moment_order: usize, center: bool) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();

    moment_simd::<f64, _>(&arr.view(), moment_order, center)
        .map_err(|e| PyRuntimeError::new_err(format!("SIMD moment calculation failed: {}", e)))
}

/// Compute SIMD-optimized mean
#[pyfunction]
fn mean_simd_py(data: &Bound<'_, PyArray1<f64>>) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();

    mean_simd::<f64, _>(&arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("SIMD mean calculation failed: {}", e)))
}

/// Compute SIMD-optimized standard deviation
#[pyfunction]
#[pyo3(signature = (data, ddof=1))]
fn std_simd_py(data: &Bound<'_, PyArray1<f64>>, ddof: usize) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();

    std_simd::<f64, _>(&arr.view(), ddof)
        .map_err(|e| PyRuntimeError::new_err(format!("SIMD standard deviation calculation failed: {}", e)))
}

/// Compute SIMD-optimized variance
#[pyfunction]
#[pyo3(signature = (data, ddof=1))]
fn variance_simd_py(data: &Bound<'_, PyArray1<f64>>, ddof: usize) -> PyResult<f64> {
    let binding = data.readonly();
    let arr = binding.as_array();

    variance_simd::<f64, _>(&arr.view(), ddof)
        .map_err(|e| PyRuntimeError::new_err(format!("SIMD variance calculation failed: {}", e)))
}

// ========================================
// NORMALITY AND GOODNESS-OF-FIT TESTS
// ========================================

/// Shapiro-Wilk test for normality
///
/// Tests the null hypothesis that the data was drawn from a normal distribution.
///
/// Parameters:
/// - data: Input data array
///
/// Returns:
/// - Dict with 'statistic' (W statistic) and 'pvalue'
#[pyfunction]
fn shapiro_py(py: Python, data: &Bound<'_, PyArray1<f64>>) -> PyResult<Py<PyAny>> {
    let binding = data.readonly();
    let arr = binding.as_array();

    let (statistic, pvalue) = shapiro_wilk(&arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Shapiro-Wilk test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;

    Ok(dict.into())
}

/// Chi-square goodness-of-fit test
///
/// Tests whether observed frequencies differ from expected frequencies.
///
/// Parameters:
/// - observed: Observed frequencies (integers or floats)
/// - expected: Expected frequencies (optional, defaults to uniform)
///
/// Returns:
/// - Dict with 'statistic', 'pvalue', 'dof' (degrees of freedom)
#[pyfunction]
#[pyo3(signature = (observed, expected=None))]
fn chisquare_py(
    py: Python,
    observed: &Bound<'_, PyArray1<f64>>,
    expected: Option<&Bound<'_, PyArray1<f64>>>,
) -> PyResult<Py<PyAny>> {
    let obs_binding = observed.readonly();
    let obs_arr = obs_binding.as_array();

    // Convert to integers
    let obs_int: Vec<i64> = obs_arr.iter().map(|&x| x.round() as i64).collect();
    let obs_int_arr = Array1::from_vec(obs_int);

    let exp_opt = expected.map(|e| {
        let e_binding = e.readonly();
        let e_arr = e_binding.as_array();
        e_arr.to_owned()
    });

    let result = chi2_gof(&obs_int_arr.view(), exp_opt.as_ref().map(|e| e.view()))
        .map_err(|e| PyRuntimeError::new_err(format!("Chi-square test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("pvalue", result.p_value)?;
    dict.set_item("dof", result.df)?;

    Ok(dict.into())
}

// ========================================
// ANOVA TESTS
// ========================================

/// One-way ANOVA (Analysis of Variance)
///
/// Tests whether the means of multiple groups are equal.
///
/// Parameters:
/// - *args: Variable number of arrays, each representing a group
///
/// Returns:
/// - Dict with 'f_statistic', 'pvalue', 'df_between', 'df_within',
///   'ss_between', 'ss_within', 'ms_between', 'ms_within'
#[pyfunction(signature = (*args))]
fn f_oneway_py(
    py: Python,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    if args.len() < 2 {
        return Err(PyRuntimeError::new_err("Need at least 2 groups for ANOVA"));
    }

    let mut group_arrays = Vec::new();
    for item in args.iter() {
        let arr: &Bound<'_, PyArray1<f64>> = item.cast()?;
        let binding = arr.readonly();
        group_arrays.push(binding.as_array().to_owned());
    }

    let group_views: Vec<ArrayView1<f64>> = group_arrays.iter().map(|a| a.view()).collect();
    let group_refs: Vec<&ArrayView1<f64>> = group_views.iter().collect();

    let result = one_way_anova(&group_refs)
        .map_err(|e| PyRuntimeError::new_err(format!("ANOVA failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("f_statistic", result.f_statistic)?;
    dict.set_item("pvalue", result.p_value)?;
    dict.set_item("df_between", result.df_treatment)?;
    dict.set_item("df_within", result.df_error)?;
    dict.set_item("ss_between", result.ss_treatment)?;
    dict.set_item("ss_within", result.ss_error)?;
    dict.set_item("ms_between", result.ms_treatment)?;
    dict.set_item("ms_within", result.ms_error)?;

    Ok(dict.into())
}

/// Wilcoxon signed-rank test for paired samples.
///
/// Parameters:
/// - x: First array of observations
/// - y: Second array of observations (paired with x)
/// - zero_method: How to handle zero differences: "wilcox" (default), "pratt"
/// - correction: Whether to apply continuity correction (default: True)
///
/// Returns:
/// - Dict with 'statistic', 'pvalue'
#[pyfunction]
#[pyo3(signature = (x, y, zero_method="wilcox", correction=true))]
fn wilcoxon_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    zero_method: &str,
    correction: bool,
) -> PyResult<Py<PyAny>> {
    let x_data = x.readonly();
    let x_arr = x_data.as_array();

    let y_data = y.readonly();
    let y_arr = y_data.as_array();

    let (statistic, pvalue) = wilcoxon(&x_arr.view(), &y_arr.view(), zero_method, correction)
        .map_err(|e| PyRuntimeError::new_err(format!("Wilcoxon test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;

    Ok(dict.into())
}

/// Mann-Whitney U test for independent samples.
///
/// Parameters:
/// - x: First array of observations
/// - y: Second array of observations
/// - alternative: Alternative hypothesis: "two-sided" (default), "less", or "greater"
/// - use_continuity: Whether to apply continuity correction (default: True)
///
/// Returns:
/// - Dict with 'statistic', 'pvalue'
#[pyfunction]
#[pyo3(signature = (x, y, alternative="two-sided", use_continuity=true))]
fn mannwhitneyu_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    alternative: &str,
    use_continuity: bool,
) -> PyResult<Py<PyAny>> {
    let x_data = x.readonly();
    let x_arr = x_data.as_array();

    let y_data = y.readonly();
    let y_arr = y_data.as_array();

    let (statistic, pvalue) = mann_whitney(&x_arr.view(), &y_arr.view(), alternative, use_continuity)
        .map_err(|e| PyRuntimeError::new_err(format!("Mann-Whitney U test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;

    Ok(dict.into())
}

/// Kruskal-Wallis H-test for independent samples.
///
/// Parameters:
/// - *args: Variable number of arrays, one for each group
///
/// Returns:
/// - Dict with 'statistic', 'pvalue'
#[pyfunction(signature = (*args))]
fn kruskal_py(
    py: Python,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    if args.len() < 2 {
        return Err(PyRuntimeError::new_err(
            "Need at least 2 groups for Kruskal-Wallis test",
        ));
    }

    // Convert all input arrays to scirs2_core Array1
    let mut arrays = Vec::new();
    for item in args.iter() {
        let arr: &Bound<'_, PyArray1<f64>> = item.cast()?;
        let readonly = arr.readonly();
        let owned = readonly.as_array().to_owned();
        arrays.push(owned);
    }

    // Create views from owned arrays
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();

    let (statistic, pvalue) = kruskal_wallis(&views)
        .map_err(|e| PyRuntimeError::new_err(format!("Kruskal-Wallis test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;

    Ok(dict.into())
}

/// Levene's test for homogeneity of variance.
///
/// Parameters:
/// - *args: Two or more arrays, each representing a group
/// - center: Which function to use: "mean", "median" (default), or "trimmed"
/// - proportion_to_cut: When using "trimmed", the proportion to cut from each end (default: 0.05)
///
/// Returns:
/// - Dict with 'statistic', 'pvalue'
#[pyfunction(signature = (*args, center="median", proportion_to_cut=0.05))]
fn levene_py(
    py: Python,
    args: &Bound<'_, pyo3::types::PyTuple>,
    center: &str,
    proportion_to_cut: f64,
) -> PyResult<Py<PyAny>> {
    if args.len() < 2 {
        return Err(PyRuntimeError::new_err(
            "Need at least 2 groups for Levene's test",
        ));
    }

    // Convert all input arrays to scirs2_core Array1
    let mut arrays = Vec::new();
    for item in args.iter() {
        let arr: &Bound<'_, PyArray1<f64>> = item.cast()?;
        let readonly = arr.readonly();
        let owned = readonly.as_array().to_owned();
        arrays.push(owned);
    }

    // Create views from owned arrays
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();

    let (statistic, pvalue) = levene(&views, center, proportion_to_cut)
        .map_err(|e| PyRuntimeError::new_err(format!("Levene's test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;

    Ok(dict.into())
}

/// Bartlett's test for homogeneity of variance.
///
/// Parameters:
/// - *args: Two or more arrays, each representing a group
///
/// Returns:
/// - Dict with 'statistic', 'pvalue'
#[pyfunction(signature = (*args))]
fn bartlett_test_py(
    py: Python,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    if args.len() < 2 {
        return Err(PyRuntimeError::new_err(
            "Need at least 2 groups for Bartlett's test",
        ));
    }

    // Convert all input arrays to scirs2_core Array1
    let mut arrays = Vec::new();
    for item in args.iter() {
        let arr: &Bound<'_, PyArray1<f64>> = item.cast()?;
        let readonly = arr.readonly();
        let owned = readonly.as_array().to_owned();
        arrays.push(owned);
    }

    // Create views from owned arrays
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();

    let (statistic, pvalue) = bartlett(&views)
        .map_err(|e| PyRuntimeError::new_err(format!("Bartlett's test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;

    Ok(dict.into())
}

/// Brown-Forsythe test for homogeneity of variance.
///
/// Parameters:
/// - *args: Two or more arrays, each representing a group
///
/// Returns:
/// - Dict with 'statistic', 'pvalue'
#[pyfunction(signature = (*args))]
fn brown_forsythe_py(
    py: Python,
    args: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<Py<PyAny>> {
    if args.len() < 2 {
        return Err(PyRuntimeError::new_err(
            "Need at least 2 groups for Brown-Forsythe test",
        ));
    }

    // Convert all input arrays to scirs2_core Array1
    let mut arrays = Vec::new();
    for item in args.iter() {
        let arr: &Bound<'_, PyArray1<f64>> = item.cast()?;
        let readonly = arr.readonly();
        let owned = readonly.as_array().to_owned();
        arrays.push(owned);
    }

    // Create views from owned arrays
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();

    let (statistic, pvalue) = brown_forsythe(&views)
        .map_err(|e| PyRuntimeError::new_err(format!("Brown-Forsythe test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;

    Ok(dict.into())
}

/// Anderson-Darling test for normality.
///
/// Tests whether a sample comes from a normal distribution using the
/// Anderson-Darling statistic. More sensitive to deviations in the tails
/// than the Shapiro-Wilk test.
///
/// Parameters:
///     x: Array of sample data
///
/// Returns:
///     Dictionary with 'statistic' and 'pvalue' keys
#[pyfunction]
fn anderson_darling_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
) -> PyResult<Py<PyAny>> {
    let x_data = x.readonly();
    let x_arr = x_data.as_array();

    let (statistic, pvalue) = anderson_darling(&x_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Anderson-Darling test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;

    Ok(dict.into())
}

/// D'Agostino's K-squared test for normality.
///
/// Tests whether a sample comes from a normal distribution using the
/// D'Agostino-Pearson K² test, which combines tests for skewness and kurtosis.
///
/// Parameters:
///     x: Array of sample data (minimum 20 observations)
///
/// Returns:
///     Dictionary with 'statistic' and 'pvalue' keys
#[pyfunction]
fn dagostino_k2_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
) -> PyResult<Py<PyAny>> {
    let x_data = x.readonly();
    let x_arr = x_data.as_array();

    let (statistic, pvalue) = dagostino_k2(&x_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("D'Agostino K² test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;

    Ok(dict.into())
}

/// Two-sample Kolmogorov-Smirnov test.
///
/// Tests whether two samples come from the same distribution using the
/// Kolmogorov-Smirnov statistic.
///
/// Parameters:
///     x: First sample array
///     y: Second sample array
///     alternative: Type of hypothesis test ("two-sided", "less", or "greater")
///
/// Returns:
///     Dictionary with 'statistic' and 'pvalue' keys
#[pyfunction]
#[pyo3(signature = (x, y, alternative="two-sided"))]
fn ks_2samp_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    alternative: &str,
) -> PyResult<Py<PyAny>> {
    let x_data = x.readonly();
    let x_arr = x_data.as_array();

    let y_data = y.readonly();
    let y_arr = y_data.as_array();

    let (statistic, pvalue) = ks_2samp(&x_arr.view(), &y_arr.view(), alternative)
        .map_err(|e| PyRuntimeError::new_err(format!("Two-sample KS test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;

    Ok(dict.into())
}

/// Friedman test for repeated measures.
///
/// Tests whether k treatments have different effects across n subjects.
/// This is a nonparametric alternative to repeated measures ANOVA.
///
/// Parameters:
///     data: 2D array with shape (n_subjects, n_treatments)
///
/// Returns:
///     Dictionary with 'statistic' and 'pvalue' keys
#[pyfunction]
fn friedman_py(
    py: Python,
    data: &Bound<'_, PyArray2<f64>>,
) -> PyResult<Py<PyAny>> {
    let data_readonly = data.readonly();
    let data_view = data_readonly.as_array();

    // Convert to owned array
    let data_arr = scirs2_core::ndarray::Array2::from_shape_vec(
        data_view.dim(),
        data_view.iter().copied().collect()
    ).map_err(|e| PyRuntimeError::new_err(format!("Array conversion failed: {}", e)))?;

    let (statistic, pvalue) = friedman(&data_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Friedman test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", statistic)?;
    dict.set_item("pvalue", pvalue)?;

    Ok(dict.into())
}

/// Chi-square test for independence (contingency table).
///
/// Tests whether two categorical variables are independent using a
/// contingency table of observed frequencies.
///
/// Parameters:
///     observed: 2D array of observed frequencies (contingency table)
///
/// Returns:
///     Dictionary with 'statistic', 'pvalue', 'df', and 'expected'
#[pyfunction]
fn chi2_independence_py(
    py: Python,
    observed: &Bound<'_, PyArray2<i64>>,
) -> PyResult<Py<PyAny>> {
    let data_readonly = observed.readonly();
    let data_view = data_readonly.as_array();

    // Convert to owned array
    let data_arr = scirs2_core::ndarray::Array2::from_shape_vec(
        data_view.dim(),
        data_view.iter().copied().collect()
    ).map_err(|e| PyRuntimeError::new_err(format!("Array conversion failed: {}", e)))?;

    let result = chi2_independence::<f64, i64>(&data_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Chi-square independence test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("pvalue", result.p_value)?;
    dict.set_item("df", result.df)?;

    // Convert expected frequencies to Python array
    let shape = result.expected.dim();
    let expected_vec: Vec<Vec<f64>> = (0..shape.0)
        .map(|i| {
            (0..shape.1)
                .map(|j| result.expected[(i, j)])
                .collect()
        })
        .collect();
    let expected_py = PyArray2::from_vec2(py, &expected_vec)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create expected array: {}", e)))?;
    dict.set_item("expected", expected_py)?;

    Ok(dict.into())
}

/// Chi-square test with Yates' continuity correction for 2x2 tables.
///
/// Applies Yates' correction to improve the chi-square approximation
/// for small sample sizes in 2x2 contingency tables.
///
/// Parameters:
///     observed: 2x2 array of observed frequencies
///
/// Returns:
///     Dictionary with 'statistic', 'pvalue', 'df', and 'expected'
#[pyfunction]
fn chi2_yates_py(
    py: Python,
    observed: &Bound<'_, PyArray2<i64>>,
) -> PyResult<Py<PyAny>> {
    let data_readonly = observed.readonly();
    let data_view = data_readonly.as_array();

    // Check dimensions
    let shape = data_view.dim();
    if shape.0 != 2 || shape.1 != 2 {
        return Err(PyRuntimeError::new_err(
            "Yates' correction requires a 2x2 contingency table"
        ));
    }

    // Convert to owned array
    let data_arr = scirs2_core::ndarray::Array2::from_shape_vec(
        data_view.dim(),
        data_view.iter().copied().collect()
    ).map_err(|e| PyRuntimeError::new_err(format!("Array conversion failed: {}", e)))?;

    let result = chi2_yates::<f64, i64>(&data_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Chi-square Yates' test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("statistic", result.statistic)?;
    dict.set_item("pvalue", result.p_value)?;
    dict.set_item("df", result.df)?;

    // Convert expected frequencies to Python array (2x2 for Yates)
    let expected_vec: Vec<f64> = result.expected.iter().copied().collect();
    let expected_py = PyArray2::from_vec2(py, &[
        expected_vec[0..2].to_vec(),
        expected_vec[2..4].to_vec(),
    ]).map_err(|e| PyRuntimeError::new_err(format!("Failed to create expected array: {}", e)))?;
    dict.set_item("expected", expected_py)?;

    Ok(dict.into())
}

// ========================================
// CONTINGENCY TABLE ANALYSIS
// ========================================

/// Fisher's exact test for 2x2 contingency tables.
///
/// Performs Fisher's exact test on a 2x2 contingency table. This test is
/// particularly useful for small sample sizes where the chi-square approximation
/// may not be valid.
///
/// Parameters:
///     table: 2x2 array of observed frequencies (must be 2x2)
///     alternative: Alternative hypothesis (default: "two-sided")
///                 - "two-sided": Test if association exists
///                 - "less": Test if odds ratio < 1
///                 - "greater": Test if odds ratio > 1
///
/// Returns:
///     Dictionary with:
///     - odds_ratio: Odds ratio (a*d)/(b*c)
///     - pvalue: P-value for the test
#[pyfunction]
#[pyo3(signature = (table, alternative="two-sided"))]
fn fisher_exact_py(
    py: Python,
    table: &Bound<'_, PyArray2<f64>>,
    alternative: &str,
) -> PyResult<Py<PyAny>> {
    let table_readonly = table.readonly();
    let table_arr = Array2::from_shape_vec(
        table_readonly.as_array().dim(),
        table_readonly.as_array().iter().copied().collect(),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Array conversion failed: {}", e)))?;

    let (odds_ratio, pvalue) = fisher_exact(&table_arr.view(), alternative)
        .map_err(|e| PyRuntimeError::new_err(format!("Fisher's exact test failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("odds_ratio", odds_ratio)?;
    dict.set_item("pvalue", pvalue)?;

    Ok(dict.into())
}

/// Calculate odds ratio for a 2x2 contingency table.
///
/// The odds ratio is a measure of association between exposure and outcome.
/// It represents the odds of outcome occurring in the exposed group relative
/// to the unexposed group.
///
/// For a 2x2 table:
///              Outcome+  Outcome-
///   Exposed+      a         b
///   Exposed-      c         d
///
/// Odds ratio = (a*d) / (b*c)
///
/// Parameters:
///     table: 2x2 array of observed frequencies
///
/// Returns:
///     Odds ratio value
#[pyfunction]
fn odds_ratio_py(
    table: &Bound<'_, PyArray2<f64>>,
) -> PyResult<f64> {
    let table_readonly = table.readonly();
    let table_arr = Array2::from_shape_vec(
        table_readonly.as_array().dim(),
        table_readonly.as_array().iter().copied().collect(),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Array conversion failed: {}", e)))?;

    let or = odds_ratio(&table_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Odds ratio calculation failed: {}", e)))?;

    Ok(or)
}

/// Calculate relative risk (risk ratio) for a 2x2 contingency table.
///
/// The relative risk is a measure of association between exposure and outcome
/// in cohort studies. It represents the risk of outcome in the exposed group
/// relative to the unexposed group.
///
/// For a 2x2 table:
///              Outcome+  Outcome-
///   Exposed+      a         b
///   Exposed-      c         d
///
/// Relative Risk = [a/(a+b)] / [c/(c+d)]
///
/// Parameters:
///     table: 2x2 array of observed frequencies
///
/// Returns:
///     Relative risk value
#[pyfunction]
fn relative_risk_py(
    table: &Bound<'_, PyArray2<f64>>,
) -> PyResult<f64> {
    let table_readonly = table.readonly();
    let table_arr = Array2::from_shape_vec(
        table_readonly.as_array().dim(),
        table_readonly.as_array().iter().copied().collect(),
    )
    .map_err(|e| PyRuntimeError::new_err(format!("Array conversion failed: {}", e)))?;

    let rr = relative_risk(&table_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Relative risk calculation failed: {}", e)))?;

    Ok(rr)
}

// ========================================
// LINEAR REGRESSION
// ========================================

/// Calculate a simple linear regression on two 1D arrays.
///
/// Performs ordinary least squares (OLS) linear regression to fit a line
/// y = slope * x + intercept to the data, and computes the correlation
/// coefficient, p-value, and standard error of the slope.
///
/// Parameters:
///     x: Independent variable (predictor)
///     y: Dependent variable (response)
///        Must be same length as x
///
/// Returns:
///     Dictionary with:
///     - slope: Slope of the regression line
///     - intercept: Y-intercept of the regression line
///     - rvalue: Correlation coefficient (Pearson's r)
///     - pvalue: Two-sided p-value for testing H₀: slope = 0
///     - stderr: Standard error of the slope estimate
#[pyfunction]
fn linregress_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
) -> PyResult<Py<PyAny>> {
    let x_readonly = x.readonly();
    let x_arr = x_readonly.as_array();

    let y_readonly = y.readonly();
    let y_arr = y_readonly.as_array();

    let (slope, intercept, rvalue, pvalue, stderr) = linregress(&x_arr.view(), &y_arr.view())
        .map_err(|e| PyRuntimeError::new_err(format!("Linear regression failed: {}", e)))?;

    let dict = PyDict::new(py);
    dict.set_item("slope", slope)?;
    dict.set_item("intercept", intercept)?;
    dict.set_item("rvalue", rvalue)?;
    dict.set_item("pvalue", pvalue)?;
    dict.set_item("stderr", stderr)?;

    Ok(dict.into())
}

/// Fit a polynomial of specified degree to data.
///
/// Fits a polynomial p(x) = c[0] + c[1]*x + c[2]*x^2 + ... + c[deg]*x^deg
/// using least squares regression.
///
/// Parameters:
///     x: Independent variable data (1D array)
///     y: Dependent variable data (1D array, same length as x)
///     deg: Degree of the fitting polynomial
///
/// Returns:
///     Dictionary containing:
///     - coefficients: Polynomial coefficients (c[0], c[1], ..., c[deg])
///     - r_squared: Coefficient of determination
///     - adj_r_squared: Adjusted R-squared
///     - residuals: Residual values (y - fitted)
///     - fitted_values: Fitted (predicted) y values
///
/// Example:
///     >>> import scirs2
///     >>> x = [0, 1, 2, 3, 4]
///     >>> y = [1, 3, 9, 19, 33]  # y ≈ 1 + 2x + x^2
///     >>> result = scirs2.polyfit(x, y, deg=2)
///     >>> print(result["coefficients"])  # Should be close to [1, 2, 1]
///
/// TODO: Registration issue - function compiles but doesn't register with PyO3
/// See /tmp/scirs2_session10_polyfit_issue.md for details
#[allow(dead_code)]
#[pyfunction]
fn polyfit_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    deg: usize,
) -> PyResult<Py<PyAny>> {
    let x_readonly = x.readonly();
    let x_arr = x_readonly.as_array();

    let y_readonly = y.readonly();
    let y_arr = y_readonly.as_array();

    let result = polyfit::<f64>(&x_arr.view(), &y_arr.view(), deg)
        .map_err(|e| PyRuntimeError::new_err(format!("Polynomial fit failed: {}", e)))?;

    let dict = PyDict::new(py);

    // Convert coefficients to Python array
    let coef_vec: Vec<f64> = result.coefficients.to_vec();
    dict.set_item("coefficients", coef_vec)?;

    dict.set_item("r_squared", result.r_squared)?;
    dict.set_item("adj_r_squared", result.adj_r_squared)?;

    // Convert residuals to Python array
    let residuals_vec: Vec<f64> = result.residuals.to_vec();
    dict.set_item("residuals", residuals_vec)?;

    // Convert fitted values to Python array
    let fitted_vec: Vec<f64> = result.fitted_values.to_vec();
    dict.set_item("fitted_values", fitted_vec)?;

    Ok(dict.into())
}

/// Tukey's Honestly Significant Difference (HSD) post-hoc test.
///
/// Performs pairwise comparisons between group means after a significant
/// ANOVA result. Controls the family-wise error rate.
///
/// Parameters:
///     *args: Variable number of group arrays (minimum 2 groups)
///     alpha: Significance level (default: 0.05)
///
/// Returns:
///     List of dictionaries, each containing:
///     - group1: Index of first group
///     - group2: Index of second group
///     - mean_diff: Mean difference between groups
///     - pvalue: P-value for the comparison
///     - significant: Whether the difference is significant at alpha level
#[pyfunction]
#[pyo3(signature = (*args, alpha=0.05))]
fn tukey_hsd_py(
    py: Python,
    args: &Bound<'_, pyo3::types::PyTuple>,
    alpha: f64,
) -> PyResult<Py<PyAny>> {
    if args.len() < 2 {
        return Err(PyRuntimeError::new_err(
            "Need at least 2 groups for Tukey's HSD",
        ));
    }

    // Convert all input arrays to scirs2_core Array1
    let mut arrays = Vec::new();
    for item in args.iter() {
        let arr: &Bound<'_, PyArray1<f64>> = item.cast()?;
        let readonly = arr.readonly();
        let owned = readonly.as_array().to_owned();
        arrays.push(owned);
    }

    // Create views from owned arrays
    let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
    let view_refs: Vec<&_> = views.iter().collect();

    let results = tukey_hsd(&view_refs, alpha)
        .map_err(|e| PyRuntimeError::new_err(format!("Tukey's HSD failed: {}", e)))?;

    // Convert results to list of dictionaries
    let result_list = pyo3::types::PyList::empty(py);
    for (group1, group2, mean_diff, pvalue, significant) in results {
        let dict = PyDict::new(py);
        dict.set_item("group1", group1)?;
        dict.set_item("group2", group2)?;
        dict.set_item("mean_diff", mean_diff)?;
        dict.set_item("pvalue", pvalue)?;
        dict.set_item("significant", significant)?;
        result_list.append(dict)?;
    }

    Ok(result_list.into())
}

// ========================================
// CORRELATION TESTS
// ========================================

/// Pearson correlation coefficient with significance test.
///
/// Calculates the Pearson correlation coefficient and tests for non-correlation.
///
/// Parameters:
///     x: First array of observations
///     y: Second array of observations (same length as x)
///     alternative: Type of test: "two-sided" (default), "less", or "greater"
///
/// Returns:
///     Dictionary containing:
///     - correlation: Pearson correlation coefficient (r)
///     - pvalue: P-value for testing non-correlation
#[pyfunction]
#[pyo3(signature = (x, y, alternative="two-sided"))]
fn pearsonr_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    alternative: &str,
) -> PyResult<Py<PyAny>> {
    // Convert Python arrays to Rust arrays
    let x_readonly = x.readonly();
    let x_arr = x_readonly.as_array();

    let y_readonly = y.readonly();
    let y_arr = y_readonly.as_array();

    // Call Rust function
    let (r, pvalue) = pearsonr(&x_arr.view(), &y_arr.view(), alternative)
        .map_err(|e| PyRuntimeError::new_err(format!("Pearson correlation test failed: {}", e)))?;

    // Create result dictionary
    let dict = PyDict::new(py);
    dict.set_item("correlation", r)?;
    dict.set_item("pvalue", pvalue)?;

    Ok(dict.into())
}

/// Spearman rank correlation coefficient with significance test.
///
/// Calculates the Spearman rank correlation coefficient and tests for non-correlation.
///
/// Parameters:
///     x: First array of observations
///     y: Second array of observations (same length as x)
///     alternative: Type of test: "two-sided" (default), "less", or "greater"
///
/// Returns:
///     Dictionary containing:
///     - correlation: Spearman rank correlation coefficient (rho)
///     - pvalue: P-value for testing non-correlation
#[pyfunction]
#[pyo3(signature = (x, y, alternative="two-sided"))]
fn spearmanr_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    alternative: &str,
) -> PyResult<Py<PyAny>> {
    // Convert Python arrays to Rust arrays
    let x_readonly = x.readonly();
    let x_arr = x_readonly.as_array();

    let y_readonly = y.readonly();
    let y_arr = y_readonly.as_array();

    // Call Rust function
    let (rho, pvalue) = spearmanr(&x_arr.view(), &y_arr.view(), alternative)
        .map_err(|e| PyRuntimeError::new_err(format!("Spearman correlation test failed: {}", e)))?;

    // Create result dictionary
    let dict = PyDict::new(py);
    dict.set_item("correlation", rho)?;
    dict.set_item("pvalue", pvalue)?;

    Ok(dict.into())
}

/// Kendall tau rank correlation coefficient with significance test.
///
/// Calculates the Kendall tau rank correlation coefficient and tests for non-correlation.
///
/// Parameters:
///     x: First array of observations
///     y: Second array of observations (same length as x)
///     method: Kendall tau variant: "b" (default) or "c"
///     alternative: Type of test: "two-sided" (default), "less", or "greater"
///
/// Returns:
///     Dictionary containing:
///     - correlation: Kendall tau correlation coefficient (tau)
///     - pvalue: P-value for testing non-correlation
#[pyfunction]
#[pyo3(signature = (x, y, method="b", alternative="two-sided"))]
fn kendalltau_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    method: &str,
    alternative: &str,
) -> PyResult<Py<PyAny>> {
    // Convert Python arrays to Rust arrays
    let x_readonly = x.readonly();
    let x_arr = x_readonly.as_array();

    let y_readonly = y.readonly();
    let y_arr = y_readonly.as_array();

    // Call Rust function
    let (tau, pvalue) = kendalltau(&x_arr.view(), &y_arr.view(), method, alternative)
        .map_err(|e| PyRuntimeError::new_err(format!("Kendall tau test failed: {}", e)))?;

    // Create result dictionary
    let dict = PyDict::new(py);
    dict.set_item("correlation", tau)?;
    dict.set_item("pvalue", pvalue)?;

    Ok(dict.into())
}

// ========================================
// STATISTICAL DISTRIBUTIONS
// ========================================

/// Normal (Gaussian) distribution
#[pyclass(name = "norm")]
pub struct PyNormal {
    dist: RustNormal<f64>,
}

#[pymethods]
impl PyNormal {
    /// Create a new Normal distribution
    ///
    /// Parameters:
    /// - loc: Mean (location) parameter (default: 0.0)
    /// - scale: Standard deviation (scale) parameter (default: 1.0)
    #[new]
    #[pyo3(signature = (loc=0.0, scale=1.0))]
    fn new(loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustNormal::new(loc, scale)
            .map_err(|e| PyRuntimeError::new_err(format!("Normal distribution creation failed: {}", e)))?;
        Ok(PyNormal { dist })
    }

    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }

    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }

    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist.ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }

    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self.dist.rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

/// Binomial distribution
#[pyclass(name = "binom")]
pub struct PyBinomial {
    dist: RustBinomial<f64>,
}

#[pymethods]
impl PyBinomial {
    /// Create a new Binomial distribution
    ///
    /// Parameters:
    /// - n: Number of trials
    /// - p: Probability of success
    #[new]
    fn new(n: usize, p: f64) -> PyResult<Self> {
        let dist = RustBinomial::new(n, p)
            .map_err(|e| PyRuntimeError::new_err(format!("Binomial distribution creation failed: {}", e)))?;
        Ok(PyBinomial { dist })
    }

    /// Probability mass function
    fn pmf(&self, k: f64) -> f64 {
        self.dist.pmf(k)
    }

    /// Cumulative distribution function
    fn cdf(&self, k: f64) -> f64 {
        self.dist.cdf(k)
    }

    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist.ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }

    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self.dist.rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

/// Poisson distribution
#[pyclass(name = "poisson")]
pub struct PyPoisson {
    dist: RustPoisson<f64>,
}

#[pymethods]
impl PyPoisson {
    /// Create a new Poisson distribution
    ///
    /// Parameters:
    /// - mu: Expected number of events (lambda parameter)
    #[new]
    fn new(mu: f64) -> PyResult<Self> {
        let dist = RustPoisson::new(mu, 0.0)  // loc parameter is 0 for standard Poisson
            .map_err(|e| PyRuntimeError::new_err(format!("Poisson distribution creation failed: {}", e)))?;
        Ok(PyPoisson { dist })
    }

    /// Probability mass function
    fn pmf(&self, k: f64) -> f64 {
        self.dist.pmf(k)
    }

    /// Cumulative distribution function
    fn cdf(&self, k: f64) -> f64 {
        self.dist.cdf(k)
    }

    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist.ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }

    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self.dist.rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

/// Exponential distribution
#[pyclass(name = "expon")]
pub struct PyExponential {
    dist: RustExponential<f64>,
}

#[pymethods]
impl PyExponential {
    /// Create a new Exponential distribution
    ///
    /// Parameters:
    /// - scale: Scale parameter (1/lambda) (default: 1.0)
    #[new]
    #[pyo3(signature = (scale=1.0))]
    fn new(scale: f64) -> PyResult<Self> {
        let rate = 1.0 / scale;  // Convert scale to rate
        let dist = RustExponential::new(rate, 0.0)  // rate and loc parameters
            .map_err(|e| PyRuntimeError::new_err(format!("Exponential distribution creation failed: {}", e)))?;
        Ok(PyExponential { dist })
    }

    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }

    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }

    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist.ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }

    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self.dist.rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

/// Uniform distribution
#[pyclass(name = "uniform")]
pub struct PyUniform {
    dist: RustUniform<f64>,
}

#[pymethods]
impl PyUniform {
    /// Create a new Uniform distribution
    ///
    /// Parameters:
    /// - loc: Lower bound (default: 0.0)
    /// - scale: Width (upper - lower) (default: 1.0)
    #[new]
    #[pyo3(signature = (loc=0.0, scale=1.0))]
    fn new(loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustUniform::new(loc, loc + scale)
            .map_err(|e| PyRuntimeError::new_err(format!("Uniform distribution creation failed: {}", e)))?;
        Ok(PyUniform { dist })
    }

    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }

    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }

    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist.ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }

    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self.dist.rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

// ========================================
// CONTINUOUS DISTRIBUTIONS (Extended)
// ========================================

/// Beta distribution
#[pyclass(name = "beta")]
pub struct PyBeta {
    dist: RustBeta<f64>,
}

#[pymethods]
impl PyBeta {
    /// Create a new Beta distribution
    ///
    /// Parameters:
    /// - alpha: Shape parameter alpha > 0
    /// - beta: Shape parameter beta > 0
    /// - loc: Location parameter (default: 0.0)
    /// - scale: Scale parameter (default: 1.0)
    #[new]
    #[pyo3(signature = (alpha, beta, loc=0.0, scale=1.0))]
    fn new(alpha: f64, beta: f64, loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustBeta::new(alpha, beta, loc, scale)
            .map_err(|e| PyRuntimeError::new_err(format!("Beta distribution creation failed: {}", e)))?;
        Ok(PyBeta { dist })
    }

    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }

    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }

    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist.ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }

    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self.dist.rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

/// Gamma distribution
#[pyclass(name = "gamma")]
pub struct PyGamma {
    dist: RustGamma<f64>,
}

#[pymethods]
impl PyGamma {
    /// Create a new Gamma distribution
    ///
    /// Parameters:
    /// - shape: Shape parameter k > 0
    /// - scale: Scale parameter theta > 0 (default: 1.0)
    /// - loc: Location parameter (default: 0.0)
    #[new]
    #[pyo3(signature = (shape, scale=1.0, loc=0.0))]
    fn new(shape: f64, scale: f64, loc: f64) -> PyResult<Self> {
        let dist = RustGamma::new(shape, scale, loc)
            .map_err(|e| PyRuntimeError::new_err(format!("Gamma distribution creation failed: {}", e)))?;
        Ok(PyGamma { dist })
    }

    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }

    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }

    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist.ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }

    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self.dist.rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

/// Chi-square distribution
#[pyclass(name = "chi2")]
pub struct PyChiSquare {
    dist: RustChiSquare<f64>,
}

#[pymethods]
impl PyChiSquare {
    /// Create a new Chi-square distribution
    ///
    /// Parameters:
    /// - df: Degrees of freedom > 0
    /// - loc: Location parameter (default: 0.0)
    /// - scale: Scale parameter (default: 1.0)
    #[new]
    #[pyo3(signature = (df, loc=0.0, scale=1.0))]
    fn new(df: f64, loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustChiSquare::new(df, loc, scale)
            .map_err(|e| PyRuntimeError::new_err(format!("Chi-square distribution creation failed: {}", e)))?;
        Ok(PyChiSquare { dist })
    }

    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }

    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }

    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist.ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }

    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self.dist.rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

/// Student's t distribution
#[pyclass(name = "t")]
pub struct PyStudentT {
    dist: RustStudentT<f64>,
}

#[pymethods]
impl PyStudentT {
    /// Create a new Student's t distribution
    ///
    /// Parameters:
    /// - df: Degrees of freedom > 0
    /// - loc: Location parameter (default: 0.0)
    /// - scale: Scale parameter (default: 1.0)
    #[new]
    #[pyo3(signature = (df, loc=0.0, scale=1.0))]
    fn new(df: f64, loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustStudentT::new(df, loc, scale)
            .map_err(|e| PyRuntimeError::new_err(format!("Student's t distribution creation failed: {}", e)))?;
        Ok(PyStudentT { dist })
    }

    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }

    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }

    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist.ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }

    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self.dist.rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

/// Cauchy distribution
#[pyclass(name = "cauchy")]
pub struct PyCauchy {
    dist: RustCauchy<f64>,
}

#[pymethods]
impl PyCauchy {
    /// Create a new Cauchy distribution
    ///
    /// Parameters:
    /// - loc: Location parameter (default: 0.0)
    /// - scale: Scale parameter > 0 (default: 1.0)
    #[new]
    #[pyo3(signature = (loc=0.0, scale=1.0))]
    fn new(loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustCauchy::new(loc, scale)
            .map_err(|e| PyRuntimeError::new_err(format!("Cauchy distribution creation failed: {}", e)))?;
        Ok(PyCauchy { dist })
    }

    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }

    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }

    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist.ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }

    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self.dist.rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

/// F distribution
#[pyclass(name = "f")]
pub struct PyF {
    dist: RustF<f64>,
}

#[pymethods]
impl PyF {
    /// Create a new F distribution
    ///
    /// Parameters:
    /// - dfn: Numerator degrees of freedom > 0
    /// - dfd: Denominator degrees of freedom > 0
    /// - loc: Location parameter (default: 0.0)
    /// - scale: Scale parameter (default: 1.0)
    #[new]
    #[pyo3(signature = (dfn, dfd, loc=0.0, scale=1.0))]
    fn new(dfn: f64, dfd: f64, loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustF::new(dfn, dfd, loc, scale)
            .map_err(|e| PyRuntimeError::new_err(format!("F distribution creation failed: {}", e)))?;
        Ok(PyF { dist })
    }

    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }

    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }

    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self.dist.rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

// ========================================
// CONTINUOUS DISTRIBUTIONS (Additional)
// ========================================

/// Lognormal distribution
#[pyclass(name = "lognorm")]
pub struct PyLognormal {
    dist: RustLognormal<f64>,
}

#[pymethods]
impl PyLognormal {
    /// Create a new Lognormal distribution
    ///
    /// Parameters:
    /// - mu: Mean of underlying normal distribution (default: 0.0)
    /// - sigma: Standard deviation of underlying normal distribution (default: 1.0)
    /// - loc: Location parameter (default: 0.0)
    #[new]
    #[pyo3(signature = (mu=0.0, sigma=1.0, loc=0.0))]
    fn new(mu: f64, sigma: f64, loc: f64) -> PyResult<Self> {
        let dist = RustLognormal::new(mu, sigma, loc)
            .map_err(|e| PyRuntimeError::new_err(format!("Lognormal distribution creation failed: {}", e)))?;
        Ok(PyLognormal { dist })
    }

    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }

    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }

    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist.ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }

    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self.dist.rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

/// Weibull distribution
#[pyclass(name = "weibull_min")]
pub struct PyWeibull {
    dist: RustWeibull<f64>,
}

#[pymethods]
impl PyWeibull {
    /// Create a new Weibull distribution
    ///
    /// Parameters:
    /// - shape: Shape parameter k > 0
    /// - scale: Scale parameter lambda > 0 (default: 1.0)
    /// - loc: Location parameter (default: 0.0)
    #[new]
    #[pyo3(signature = (shape, scale=1.0, loc=0.0))]
    fn new(shape: f64, scale: f64, loc: f64) -> PyResult<Self> {
        let dist = RustWeibull::new(shape, scale, loc)
            .map_err(|e| PyRuntimeError::new_err(format!("Weibull distribution creation failed: {}", e)))?;
        Ok(PyWeibull { dist })
    }

    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }

    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }

    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist.ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }

    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self.dist.rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

/// Laplace distribution
#[pyclass(name = "laplace")]
pub struct PyLaplace {
    dist: RustLaplace<f64>,
}

#[pymethods]
impl PyLaplace {
    /// Create a new Laplace distribution
    ///
    /// Parameters:
    /// - loc: Location parameter (default: 0.0)
    /// - scale: Scale parameter > 0 (default: 1.0)
    #[new]
    #[pyo3(signature = (loc=0.0, scale=1.0))]
    fn new(loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustLaplace::new(loc, scale)
            .map_err(|e| PyRuntimeError::new_err(format!("Laplace distribution creation failed: {}", e)))?;
        Ok(PyLaplace { dist })
    }

    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }

    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }

    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist.ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }

    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self.dist.rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

/// Logistic distribution
#[pyclass(name = "logistic")]
pub struct PyLogistic {
    dist: RustLogistic<f64>,
}

#[pymethods]
impl PyLogistic {
    /// Create a new Logistic distribution
    ///
    /// Parameters:
    /// - loc: Location parameter (default: 0.0)
    /// - scale: Scale parameter > 0 (default: 1.0)
    #[new]
    #[pyo3(signature = (loc=0.0, scale=1.0))]
    fn new(loc: f64, scale: f64) -> PyResult<Self> {
        let dist = RustLogistic::new(loc, scale)
            .map_err(|e| PyRuntimeError::new_err(format!("Logistic distribution creation failed: {}", e)))?;
        Ok(PyLogistic { dist })
    }

    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }

    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }

    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist.ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }

    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self.dist.rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

/// Pareto distribution
#[pyclass(name = "pareto")]
pub struct PyPareto {
    dist: RustPareto<f64>,
}

#[pymethods]
impl PyPareto {
    /// Create a new Pareto distribution
    ///
    /// Parameters:
    /// - shape: Shape parameter alpha > 0
    /// - scale: Scale parameter x_m > 0 (default: 1.0)
    /// - loc: Location parameter (default: 0.0)
    #[new]
    #[pyo3(signature = (shape, scale=1.0, loc=0.0))]
    fn new(shape: f64, scale: f64, loc: f64) -> PyResult<Self> {
        let dist = RustPareto::new(shape, scale, loc)
            .map_err(|e| PyRuntimeError::new_err(format!("Pareto distribution creation failed: {}", e)))?;
        Ok(PyPareto { dist })
    }

    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        self.dist.pdf(x)
    }

    /// Cumulative distribution function
    fn cdf(&self, x: f64) -> f64 {
        self.dist.cdf(x)
    }

    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist.ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }

    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self.dist.rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

/// Geometric distribution
#[pyclass(name = "geom")]
pub struct PyGeometric {
    dist: RustGeometric<f64>,
}

#[pymethods]
impl PyGeometric {
    /// Create a new Geometric distribution
    ///
    /// Parameters:
    /// - p: Success probability, 0 < p <= 1
    #[new]
    fn new(p: f64) -> PyResult<Self> {
        let dist = RustGeometric::new(p)
            .map_err(|e| PyRuntimeError::new_err(format!("Geometric distribution creation failed: {}", e)))?;
        Ok(PyGeometric { dist })
    }

    /// Probability mass function
    fn pmf(&self, k: f64) -> f64 {
        self.dist.pmf(k)
    }

    /// Cumulative distribution function
    fn cdf(&self, k: f64) -> f64 {
        self.dist.cdf(k)
    }

    /// Percent point function (inverse CDF)
    fn ppf(&self, q: f64) -> PyResult<f64> {
        self.dist.ppf(q)
            .map_err(|e| PyRuntimeError::new_err(format!("PPF failed: {}", e)))
    }

    /// Random variates
    fn rvs(&self, size: usize) -> PyResult<Vec<f64>> {
        let arr = self.dist.rvs(size)
            .map_err(|e| PyRuntimeError::new_err(format!("RVS failed: {}", e)))?;
        Ok(arr.to_vec())
    }
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Descriptive statistics
    m.add_function(wrap_pyfunction!(describe_py, m)?)?;
    m.add_function(wrap_pyfunction!(mean_py, m)?)?;
    m.add_function(wrap_pyfunction!(std_py, m)?)?;
    m.add_function(wrap_pyfunction!(var_py, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_py, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_py, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_py, m)?)?;
    m.add_function(wrap_pyfunction!(median_py, m)?)?;
    m.add_function(wrap_pyfunction!(iqr_py, m)?)?;

    // Statistical tests
    m.add_function(wrap_pyfunction!(ttest_1samp_py, m)?)?;
    m.add_function(wrap_pyfunction!(ttest_ind_py, m)?)?;
    m.add_function(wrap_pyfunction!(ttest_rel_py, m)?)?;
    m.add_function(wrap_pyfunction!(shapiro_py, m)?)?;
    m.add_function(wrap_pyfunction!(chisquare_py, m)?)?;
    m.add_function(wrap_pyfunction!(f_oneway_py, m)?)?;

    // Nonparametric tests
    m.add_function(wrap_pyfunction!(wilcoxon_py, m)?)?;
    m.add_function(wrap_pyfunction!(mannwhitneyu_py, m)?)?;
    m.add_function(wrap_pyfunction!(kruskal_py, m)?)?;

    // Homogeneity tests
    m.add_function(wrap_pyfunction!(levene_py, m)?)?;
    m.add_function(wrap_pyfunction!(bartlett_test_py, m)?)?;
    m.add_function(wrap_pyfunction!(brown_forsythe_py, m)?)?;

    // Additional normality tests
    m.add_function(wrap_pyfunction!(anderson_darling_py, m)?)?;
    m.add_function(wrap_pyfunction!(dagostino_k2_py, m)?)?;
    m.add_function(wrap_pyfunction!(ks_2samp_py, m)?)?;

    // Friedman test
    m.add_function(wrap_pyfunction!(friedman_py, m)?)?;

    // Additional chi-square tests
    m.add_function(wrap_pyfunction!(chi2_independence_py, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_yates_py, m)?)?;

    // Contingency table analysis
    m.add_function(wrap_pyfunction!(fisher_exact_py, m)?)?;
    m.add_function(wrap_pyfunction!(odds_ratio_py, m)?)?;
    m.add_function(wrap_pyfunction!(relative_risk_py, m)?)?;

    // Linear regression
    m.add_function(wrap_pyfunction!(linregress_py, m)?)?;

    // Tukey HSD post-hoc test
    m.add_function(wrap_pyfunction!(tukey_hsd_py, m)?)?;

    // Correlation tests with significance
    // m.add_function(wrap_pyfunction!(pearsonr_py, m)?)?;
    // m.add_function(wrap_pyfunction!(spearmanr_py, m)?)?;
    // m.add_function(wrap_pyfunction!(kendalltau_py, m)?)?;

    // Additional statistics
    m.add_function(wrap_pyfunction!(skew_py, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis_py, m)?)?;
    m.add_function(wrap_pyfunction!(mode_py, m)?)?;
    m.add_function(wrap_pyfunction!(gmean_py, m)?)?;
    m.add_function(wrap_pyfunction!(hmean_py, m)?)?;
    m.add_function(wrap_pyfunction!(zscore_py, m)?)?;

    // Dispersion and variability measures
    m.add_function(wrap_pyfunction!(mean_abs_deviation_py, m)?)?;
    m.add_function(wrap_pyfunction!(median_abs_deviation_py, m)?)?;
    m.add_function(wrap_pyfunction!(data_range_py, m)?)?;
    m.add_function(wrap_pyfunction!(coef_variation_py, m)?)?;
    m.add_function(wrap_pyfunction!(gini_coefficient_py, m)?)?;

    // Quantile and robust statistics
    m.add_function(wrap_pyfunction!(boxplot_stats_py, m)?)?;
    m.add_function(wrap_pyfunction!(quartiles_py, m)?)?;
    m.add_function(wrap_pyfunction!(quintiles_py, m)?)?;
    m.add_function(wrap_pyfunction!(deciles_py, m)?)?;
    m.add_function(wrap_pyfunction!(sem_py, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_range_py, m)?)?;
    m.add_function(wrap_pyfunction!(winsorized_mean_py, m)?)?;
    m.add_function(wrap_pyfunction!(winsorized_variance_py, m)?)?;

    // SIMD-optimized statistical functions
    m.add_function(wrap_pyfunction!(skewness_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(pearson_r_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(moment_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(mean_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(std_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(variance_simd_py, m)?)?;

    // Information theory and advanced statistics
    m.add_function(wrap_pyfunction!(entropy_py, m)?)?;
    m.add_function(wrap_pyfunction!(kl_divergence_py, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_py, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_mean_py, m)?)?;
    m.add_function(wrap_pyfunction!(moment_py, m)?)?;

    // Confidence intervals
    m.add_function(wrap_pyfunction!(skewness_ci_py, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis_ci_py, m)?)?;

    // Distribution classes
    m.add_class::<PyNormal>()?;
    m.add_class::<PyBinomial>()?;
    m.add_class::<PyPoisson>()?;
    m.add_class::<PyExponential>()?;
    m.add_class::<PyUniform>()?;
    m.add_class::<PyBeta>()?;
    m.add_class::<PyGamma>()?;
    m.add_class::<PyChiSquare>()?;
    m.add_class::<PyStudentT>()?;
    m.add_class::<PyCauchy>()?;
    m.add_class::<PyF>()?;
    m.add_class::<PyLognormal>()?;
    m.add_class::<PyWeibull>()?;
    m.add_class::<PyLaplace>()?;
    m.add_class::<PyLogistic>()?;
    m.add_class::<PyPareto>()?;
    m.add_class::<PyGeometric>()?;

    // Correlation tests with significance
    m.add_function(wrap_pyfunction!(pearsonr_py, m)?)?;
    m.add_function(wrap_pyfunction!(spearmanr_py, m)?)?;
    m.add_function(wrap_pyfunction!(kendalltau_py, m)?)?;

    // TODO: polyfit_py registration issue - see /tmp/scirs2_session10_polyfit_issue.md
    // m.add_function(wrap_pyfunction!(polyfit_py, m)?)?;

    Ok(())
}
