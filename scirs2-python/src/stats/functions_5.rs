//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use scirs2_numpy::{PyArray1, PyArray2, PyArrayMethods};

use super::types::{PyBeta, PyBinomial, PyCauchy, PyChiSquare, PyExponential, PyF, PyGamma, PyGeometric, PyLaplace, PyLogistic, PyLognormal, PyNormal, PyPareto, PyPoisson, PyStudentT, PyUniform, PyWeibull};

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
#[pyo3(signature = (x, y, method = "b", alternative = "two-sided"))]
fn kendalltau_py(
    py: Python,
    x: &Bound<'_, PyArray1<f64>>,
    y: &Bound<'_, PyArray1<f64>>,
    method: &str,
    alternative: &str,
) -> PyResult<Py<PyAny>> {
    let x_readonly = x.readonly();
    let x_arr = x_readonly.as_array();
    let y_readonly = y.readonly();
    let y_arr = y_readonly.as_array();
    let (tau, pvalue) = kendalltau(&x_arr.view(), &y_arr.view(), method, alternative)
        .map_err(|e| PyRuntimeError::new_err(
            format!("Kendall tau test failed: {}", e),
        ))?;
    let dict = PyDict::new(py);
    dict.set_item("correlation", tau)?;
    dict.set_item("pvalue", pvalue)?;
    Ok(dict.into())
}
/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(describe_py, m)?)?;
    m.add_function(wrap_pyfunction!(mean_py, m)?)?;
    m.add_function(wrap_pyfunction!(std_py, m)?)?;
    m.add_function(wrap_pyfunction!(var_py, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_py, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_py, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_py, m)?)?;
    m.add_function(wrap_pyfunction!(median_py, m)?)?;
    m.add_function(wrap_pyfunction!(iqr_py, m)?)?;
    m.add_function(wrap_pyfunction!(ttest_1samp_py, m)?)?;
    m.add_function(wrap_pyfunction!(ttest_ind_py, m)?)?;
    m.add_function(wrap_pyfunction!(ttest_rel_py, m)?)?;
    m.add_function(wrap_pyfunction!(shapiro_py, m)?)?;
    m.add_function(wrap_pyfunction!(chisquare_py, m)?)?;
    m.add_function(wrap_pyfunction!(f_oneway_py, m)?)?;
    m.add_function(wrap_pyfunction!(wilcoxon_py, m)?)?;
    m.add_function(wrap_pyfunction!(mannwhitneyu_py, m)?)?;
    m.add_function(wrap_pyfunction!(kruskal_py, m)?)?;
    m.add_function(wrap_pyfunction!(levene_py, m)?)?;
    m.add_function(wrap_pyfunction!(bartlett_test_py, m)?)?;
    m.add_function(wrap_pyfunction!(brown_forsythe_py, m)?)?;
    m.add_function(wrap_pyfunction!(anderson_darling_py, m)?)?;
    m.add_function(wrap_pyfunction!(dagostino_k2_py, m)?)?;
    m.add_function(wrap_pyfunction!(ks_2samp_py, m)?)?;
    m.add_function(wrap_pyfunction!(friedman_py, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_independence_py, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_yates_py, m)?)?;
    m.add_function(wrap_pyfunction!(fisher_exact_py, m)?)?;
    m.add_function(wrap_pyfunction!(odds_ratio_py, m)?)?;
    m.add_function(wrap_pyfunction!(relative_risk_py, m)?)?;
    m.add_function(wrap_pyfunction!(linregress_py, m)?)?;
    m.add_function(wrap_pyfunction!(tukey_hsd_py, m)?)?;
    m.add_function(wrap_pyfunction!(skew_py, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis_py, m)?)?;
    m.add_function(wrap_pyfunction!(mode_py, m)?)?;
    m.add_function(wrap_pyfunction!(gmean_py, m)?)?;
    m.add_function(wrap_pyfunction!(hmean_py, m)?)?;
    m.add_function(wrap_pyfunction!(zscore_py, m)?)?;
    m.add_function(wrap_pyfunction!(mean_abs_deviation_py, m)?)?;
    m.add_function(wrap_pyfunction!(median_abs_deviation_py, m)?)?;
    m.add_function(wrap_pyfunction!(data_range_py, m)?)?;
    m.add_function(wrap_pyfunction!(coef_variation_py, m)?)?;
    m.add_function(wrap_pyfunction!(gini_coefficient_py, m)?)?;
    m.add_function(wrap_pyfunction!(boxplot_stats_py, m)?)?;
    m.add_function(wrap_pyfunction!(quartiles_py, m)?)?;
    m.add_function(wrap_pyfunction!(quintiles_py, m)?)?;
    m.add_function(wrap_pyfunction!(deciles_py, m)?)?;
    m.add_function(wrap_pyfunction!(sem_py, m)?)?;
    m.add_function(wrap_pyfunction!(percentile_range_py, m)?)?;
    m.add_function(wrap_pyfunction!(winsorized_mean_py, m)?)?;
    m.add_function(wrap_pyfunction!(winsorized_variance_py, m)?)?;
    m.add_function(wrap_pyfunction!(skewness_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(pearson_r_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(moment_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(mean_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(std_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(variance_simd_py, m)?)?;
    m.add_function(wrap_pyfunction!(entropy_py, m)?)?;
    m.add_function(wrap_pyfunction!(kl_divergence_py, m)?)?;
    m.add_function(wrap_pyfunction!(cross_entropy_py, m)?)?;
    m.add_function(wrap_pyfunction!(weighted_mean_py, m)?)?;
    m.add_function(wrap_pyfunction!(moment_py, m)?)?;
    m.add_function(wrap_pyfunction!(skewness_ci_py, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis_ci_py, m)?)?;
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
    m.add_function(wrap_pyfunction!(pearsonr_py, m)?)?;
    m.add_function(wrap_pyfunction!(spearmanr_py, m)?)?;
    m.add_function(wrap_pyfunction!(kendalltau_py, m)?)?;
    Ok(())
}
