//! Unified distribution interface for the SciRS2 ecosystem
//!
//! This module provides a consistent interface for all statistical distributions,
//! ensuring compatibility across the entire SciRS2 ecosystem including ToRSh, SkleaRS, etc.
//!
//! ## Design Philosophy
//!
//! 1. **Zero Breaking Changes**: All existing code continues to work
//! 2. **Full Compatibility**: Direct access to all rand_distr distributions
//! 3. **Enhanced Functionality**: Additional scientific computing features
//! 4. **Type Safety**: Unified trait system for distribution operations
//!
//! ## Usage Examples
//!
//! ```rust
//! use scirs2_core::random::distributions_unified::*;
//! use scirs2_core::random::thread_rng;
//!
//! // Create distributions with unified interface
//! let normal = UnifiedNormal::new(0.0, 1.0).unwrap();
//! let beta = UnifiedBeta::new(2.0, 5.0).unwrap();
//! let student_t = UnifiedStudentT::new(10.0).unwrap();
//!
//! let mut rng = thread_rng();
//! let sample = normal.sample_unified(&mut rng);
//! ```

use crate::random::Random;
use ndarray::{Array1, ArrayD, IxDyn};
use rand::Rng;
use rand_distr::{Distribution, DistributionError};

/// Unified distribution trait for consistent interface across all distributions
pub trait UnifiedDistribution<T> {
    /// Sample a single value from the distribution
    fn sample_unified<R: Rng>(&self, rng: &mut Random<R>) -> T;

    /// Sample multiple values into a vector
    fn sample_vec<R: Rng>(&self, rng: &mut Random<R>, n: usize) -> Vec<T>;

    /// Sample into an ndarray
    fn sample_array<R: Rng>(&self, rng: &mut Random<R>, shape: IxDyn) -> ArrayD<T>
    where
        T: Clone;

    /// Get distribution parameters as a string (for debugging/logging)
    fn parameters_string(&self) -> String;

    /// Validate distribution parameters
    fn validate(&self) -> Result<(), DistributionError>;
}

/// Macro to implement unified wrapper for rand_distr distributions
macro_rules! impl_unified_distribution {
    ($name:ident, $inner:ty, $output:ty, $params:expr) => {
        #[derive(Debug, Clone)]
        pub struct $name {
            inner: $inner,
        }

        impl $name {
            /// Get reference to inner distribution
            pub fn inner(&self) -> &$inner {
                &self.inner
            }

            /// Get mutable reference to inner distribution
            pub fn inner_mut(&mut self) -> &mut $inner {
                &mut self.inner
            }
        }

        impl UnifiedDistribution<$output> for $name {
            fn sample_unified<R: Rng>(&self, rng: &mut Random<R>) -> $output {
                rng.sample(&self.inner)
            }

            fn sample_vec<R: Rng>(&self, rng: &mut Random<R>, n: usize) -> Vec<$output> {
                (0..n).map(|_| self.sample_unified(rng)).collect()
            }

            fn sample_array<R: Rng>(&self, rng: &mut Random<R>, shape: IxDyn) -> ArrayD<$output>
            where
                $output: Clone,
            {
                let size = shape.size();
                let values = self.sample_vec(rng, size);
                ArrayD::from_shape_vec(shape, values).unwrap()
            }

            fn parameters_string(&self) -> String {
                $params(&self.inner)
            }

            fn validate(&self) -> Result<(), DistributionError> {
                // Validation is done during construction
                Ok(())
            }
        }

        impl Distribution<$output> for $name {
            fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> $output {
                self.inner.sample(rng)
            }
        }
    };
}

// Continuous distributions

/// Unified Normal distribution
impl_unified_distribution!(UnifiedNormal, rand_distr::Normal<f64>, f64,
    |d: &rand_distr::Normal<f64>| format!("Normal(mean={}, std={})", d.mean(), d.std_dev()));

impl UnifiedNormal {
    pub fn new(mean: f64, std_dev: f64) -> Result<Self, DistributionError> {
        Ok(Self {
            inner: rand_distr::Normal::new(mean, std_dev)?,
        })
    }

    pub fn mean(&self) -> f64 {
        self.inner.mean()
    }

    pub fn std_dev(&self) -> f64 {
        self.inner.std_dev()
    }
}

/// Unified Beta distribution
impl_unified_distribution!(UnifiedBeta, rand_distr::Beta<f64>, f64,
    |_: &rand_distr::Beta<f64>| "Beta(alpha, beta)".to_string());

impl UnifiedBeta {
    pub fn new(alpha: f64, beta: f64) -> Result<Self, DistributionError> {
        Ok(Self {
            inner: rand_distr::Beta::new(alpha, beta)?,
        })
    }
}

/// Unified Cauchy distribution
impl_unified_distribution!(UnifiedCauchy, rand_distr::Cauchy<f64>, f64,
    |_: &rand_distr::Cauchy<f64>| "Cauchy(median, scale)".to_string());

impl UnifiedCauchy {
    pub fn new(median: f64, scale: f64) -> Result<Self, DistributionError> {
        Ok(Self {
            inner: rand_distr::Cauchy::new(median, scale)?,
        })
    }
}

/// Unified ChiSquared distribution
impl_unified_distribution!(UnifiedChiSquared, rand_distr::ChiSquared<f64>, f64,
    |_: &rand_distr::ChiSquared<f64>| "ChiSquared(k)".to_string());

impl UnifiedChiSquared {
    pub fn new(k: f64) -> Result<Self, DistributionError> {
        Ok(Self {
            inner: rand_distr::ChiSquared::new(k)?,
        })
    }
}

/// Unified FisherF distribution
impl_unified_distribution!(UnifiedFisherF, rand_distr::FisherF<f64>, f64,
    |_: &rand_distr::FisherF<f64>| "FisherF(m, n)".to_string());

impl UnifiedFisherF {
    pub fn new(m: f64, n: f64) -> Result<Self, DistributionError> {
        Ok(Self {
            inner: rand_distr::FisherF::new(m, n)?,
        })
    }
}

/// Unified LogNormal distribution
impl_unified_distribution!(UnifiedLogNormal, rand_distr::LogNormal<f64>, f64,
    |_: &rand_distr::LogNormal<f64>| "LogNormal(mean, std)".to_string());

impl UnifiedLogNormal {
    pub fn new(mean: f64, std_dev: f64) -> Result<Self, DistributionError> {
        Ok(Self {
            inner: rand_distr::LogNormal::new(mean, std_dev)?,
        })
    }
}

/// Unified StudentT distribution
impl_unified_distribution!(UnifiedStudentT, rand_distr::StudentT<f64>, f64,
    |_: &rand_distr::StudentT<f64>| "StudentT(n)".to_string());

impl UnifiedStudentT {
    pub fn new(n: f64) -> Result<Self, DistributionError> {
        Ok(Self {
            inner: rand_distr::StudentT::new(n)?,
        })
    }
}

/// Unified Weibull distribution
impl_unified_distribution!(UnifiedWeibull, rand_distr::Weibull<f64>, f64,
    |_: &rand_distr::Weibull<f64>| "Weibull(lambda, k)".to_string());

impl UnifiedWeibull {
    pub fn new(lambda: f64, k: f64) -> Result<Self, DistributionError> {
        Ok(Self {
            inner: rand_distr::Weibull::new(lambda, k)?,
        })
    }
}

/// Unified Gamma distribution
impl_unified_distribution!(UnifiedGamma, rand_distr::Gamma<f64>, f64,
    |_: &rand_distr::Gamma<f64>| "Gamma(shape, scale)".to_string());

impl UnifiedGamma {
    pub fn new(shape: f64, scale: f64) -> Result<Self, DistributionError> {
        Ok(Self {
            inner: rand_distr::Gamma::new(shape, scale)?,
        })
    }
}

/// Unified Exponential distribution
impl_unified_distribution!(UnifiedExp, rand_distr::Exp<f64>, f64,
    |_: &rand_distr::Exp<f64>| "Exp(lambda)".to_string());

impl UnifiedExp {
    pub fn new(lambda: f64) -> Result<Self, DistributionError> {
        Ok(Self {
            inner: rand_distr::Exp::new(lambda)?,
        })
    }
}

// Discrete distributions

/// Unified Binomial distribution
impl_unified_distribution!(UnifiedBinomial, rand_distr::Binomial, u64,
    |_: &rand_distr::Binomial| "Binomial(n, p)".to_string());

impl UnifiedBinomial {
    pub fn new(n: u64, p: f64) -> Result<Self, DistributionError> {
        Ok(Self {
            inner: rand_distr::Binomial::new(n, p)?,
        })
    }
}

/// Unified Poisson distribution
impl_unified_distribution!(UnifiedPoisson, rand_distr::Poisson<f64>, u64,
    |_: &rand_distr::Poisson<f64>| "Poisson(lambda)".to_string());

impl UnifiedPoisson {
    pub fn new(lambda: f64) -> Result<Self, DistributionError> {
        Ok(Self {
            inner: rand_distr::Poisson::new(lambda)?,
        })
    }
}

// Multivariate distributions

/// Unified Dirichlet distribution
#[derive(Debug, Clone)]
pub struct UnifiedDirichlet {
    inner: rand_distr::Dirichlet<f64>,
}

impl UnifiedDirichlet {
    pub fn new(alpha: Vec<f64>) -> Result<Self, DistributionError> {
        Ok(Self {
            inner: rand_distr::Dirichlet::new(alpha)?,
        })
    }

    /// Sample from Dirichlet distribution
    pub fn sample<R: Rng>(&self, rng: &mut Random<R>) -> Vec<f64> {
        rng.sample(&self.inner)
    }

    /// Sample into an Array1
    pub fn sample_array<R: Rng>(&self, rng: &mut Random<R>) -> Array1<f64> {
        Array1::from_vec(self.sample(rng))
    }

    /// Sample multiple times
    pub fn sample_multiple<R: Rng>(&self, rng: &mut Random<R>, n: usize) -> Vec<Vec<f64>> {
        (0..n).map(|_| self.sample(rng)).collect()
    }
}

impl Distribution<Vec<f64>> for UnifiedDirichlet {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        self.inner.sample(rng)
    }
}

/// Convenience functions for creating distributions with default parameters
pub mod defaults {
    use super::*;

    /// Create a standard normal distribution (mean=0, std=1)
    pub fn standard_normal() -> UnifiedNormal {
        UnifiedNormal::new(0.0, 1.0).unwrap()
    }

    /// Create a uniform distribution on [0, 1)
    pub fn uniform_01() -> rand_distr::Uniform<f64> {
        rand_distr::Uniform::new(0.0, 1.0).unwrap()
    }

    /// Create a symmetric beta distribution
    pub fn symmetric_beta(alpha: f64) -> UnifiedBeta {
        UnifiedBeta::new(alpha, alpha).unwrap()
    }

    /// Create a standard Cauchy distribution
    pub fn standard_cauchy() -> UnifiedCauchy {
        UnifiedCauchy::new(0.0, 1.0).unwrap()
    }

    /// Create an exponential distribution with rate 1
    pub fn standard_exponential() -> UnifiedExp {
        UnifiedExp::new(1.0).unwrap()
    }
}

/// Enhanced sampling utilities for scientific computing
pub mod sampling_utils {
    use super::*;
    use crate::random::thread_rng;

    /// Sample from any distribution and return statistics
    pub struct SampleStatistics {
        pub mean: f64,
        pub variance: f64,
        pub min: f64,
        pub max: f64,
        pub samples: Vec<f64>,
    }

    /// Generate samples and compute statistics
    pub fn sample_with_stats<D>(dist: &D, n: usize) -> SampleStatistics
    where
        D: UnifiedDistribution<f64>,
    {
        let mut rng = thread_rng();
        let samples = dist.sample_vec(&mut rng, n);

        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (n - 1) as f64;

        let min = samples.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = samples.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        SampleStatistics {
            mean,
            variance,
            min,
            max,
            samples,
        }
    }

    /// Generate samples for hypothesis testing
    pub fn generate_test_samples<D1, D2>(
        null_dist: &D1,
        alt_dist: &D2,
        n_samples: usize,
    ) -> (Vec<f64>, Vec<f64>)
    where
        D1: UnifiedDistribution<f64>,
        D2: UnifiedDistribution<f64>,
    {
        let mut rng = thread_rng();
        let null_samples = null_dist.sample_vec(&mut rng, n_samples);
        let alt_samples = alt_dist.sample_vec(&mut rng, n_samples);
        (null_samples, alt_samples)
    }
}

/// Compatibility layer for ToRSh and other projects
pub mod compat {
    pub use super::{
        UnifiedBeta as Beta,
        UnifiedCauchy as Cauchy,
        UnifiedChiSquared as ChiSquared,
        UnifiedFisherF as FisherF,
        UnifiedLogNormal as LogNormal,
        UnifiedStudentT as StudentT,
        UnifiedWeibull as Weibull,
        UnifiedDirichlet as Dirichlet,
        UnifiedGamma as Gamma,
        UnifiedNormal as Normal,
        UnifiedExp as Exponential,
        UnifiedBinomial as Binomial,
        UnifiedPoisson as Poisson,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::thread_rng;

    #[test]
    fn test_unified_distributions() {
        let mut rng = thread_rng();

        // Test continuous distributions
        let normal = UnifiedNormal::new(0.0, 1.0).unwrap();
        let _: f64 = normal.sample_unified(&mut rng);

        let beta = UnifiedBeta::new(2.0, 5.0).unwrap();
        let beta_sample = beta.sample_unified(&mut rng);
        assert!(beta_sample >= 0.0 && beta_sample <= 1.0);

        let student_t = UnifiedStudentT::new(10.0).unwrap();
        let _: f64 = student_t.sample_unified(&mut rng);

        // Test discrete distributions
        let binomial = UnifiedBinomial::new(10, 0.5).unwrap();
        let binom_sample = binomial.sample_unified(&mut rng);
        assert!(binom_sample <= 10);

        // Test multivariate
        let dirichlet = UnifiedDirichlet::new(vec![1.0, 2.0, 3.0]).unwrap();
        let dir_sample = dirichlet.sample(&mut rng);
        assert_eq!(dir_sample.len(), 3);
        let sum: f64 = dir_sample.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_array_sampling() {
        use ndarray::IxDyn;
        let mut rng = thread_rng();

        let normal = UnifiedNormal::new(0.0, 1.0).unwrap();
        let shape = IxDyn(&[10, 20]);
        let array = normal.sample_array(&mut rng, shape.clone());

        assert_eq!(array.shape(), shape.as_array_view().as_slice().unwrap());
        assert_eq!(array.len(), 200);
    }
}