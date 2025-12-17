# SciRS2 - Python Bindings

**SciRS2**: Type-safe scientific computing in Rust with Python bindings - Specialized for complex statistical analysis with exceptional performance for higher-order statistics.

[![PyPI](https://img.shields.io/pypi/v/scirs2)](https://pypi.org/project/scirs2/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/scirs2)](https://pypi.org/project/scirs2/)

## Overview

SciRS2-Python provides Python bindings for the SciRS2 scientific computing ecosystem, offering:

- **Exceptional Complex Statistics**: Up to **23x faster** than SciPy for skewness, kurtosis, and higher-order moments on small-medium datasets
- **Type Safety**: Rust's compile-time guarantees prevent many runtime errors
- **SciPy-Compatible APIs**: Familiar interface for Python scientists
- **Zero-Copy Integration**: Efficient NumPy array interoperability
- **Hybrid Approach**: Use alongside NumPy/SciPy for optimal performance

**Important**: SciRS2 is a **specialized tool** for type-safe complex statistical analysis, not a general-purpose NumPy/SciPy replacement. See [Performance Guide](#performance) for when to use scirs2 vs NumPy.

## Installation

```bash
pip install scirs2
```

For development:
```bash
pip install scirs2[dev]
```

## Quick Start

**Recommended Hybrid Approach** - Use scirs2 for complex statistics, NumPy/SciPy for everything else:

```python
import numpy as np
import scipy.linalg
import scirs2

# Generate data
data = np.random.randn(1000)

# ✅ Use NumPy for basic statistics (FAST)
mean = np.mean(data)
std = np.std(data)

# ✅ Use scirs2 for complex statistics (FASTER!)
skewness = scirs2.skew_py(data)      # 6x faster than SciPy!
kurtosis = scirs2.kurtosis_py(data)  # 5x faster than SciPy!

print(f"Mean: {mean:.4f}, Std: {std:.4f}")
print(f"Skewness: {skewness:.4f}, Kurtosis: {kurtosis:.4f}")

# ❌ AVOID: Linear algebra (200-700x slower than SciPy)
# Use SciPy/NumPy instead:
A = np.random.randn(100, 100)
det = scipy.linalg.det(A)  # NOT scirs2.det_py() - extremely slow!
```

## Modules (v0.2.0)

### Linear Algebra (`linalg`)

⚠️ **Performance Warning**: Linear algebra operations are currently **200-700x slower** than SciPy due to missing BLAS/LAPACK integration. **Use NumPy/SciPy for all linear algebra operations.**

```python
import numpy as np
import scirs2

A = np.array([[4.0, 2.0], [2.0, 3.0]])
b = np.array([1.0, 2.0])

# Basic operations
det = scirs2.det_py(A)           # Determinant
inv = scirs2.inv_py(A)           # Inverse
trace = scirs2.trace_py(A)       # Trace

# Decompositions
lu = scirs2.lu_py(A)             # LU: {'L', 'U', 'P'}
qr = scirs2.qr_py(A)             # QR: {'Q', 'R'}
svd = scirs2.svd_py(A)           # SVD: {'U', 'S', 'Vt'}
chol = scirs2.cholesky_py(A)     # Cholesky

# Eigenvalues
eig = scirs2.eig_py(A)           # {'eigenvalues_real', 'eigenvalues_imag', 'eigenvectors'}
eigh = scirs2.eigh_py(A)         # For symmetric matrices

# Linear systems
x = scirs2.solve_py(A, b)        # Solve Ax = b
lstsq = scirs2.lstsq_py(A, b)    # Least squares

# Norms
norm_fro = scirs2.matrix_norm_py(A, "fro")
norm_vec = scirs2.vector_norm_py(b, 2)
cond = scirs2.cond_py(A)
rank = scirs2.matrix_rank_py(A)
```

### Statistics (`stats`)

✅ **Performance Strength**: Complex statistics (skewness, kurtosis) are **4-23x faster** than SciPy on small-medium datasets (<10,000 elements). Use NumPy for basic operations (mean, std) on large data.

```python
import numpy as np
import scirs2

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

# Descriptive statistics
stats = scirs2.describe_py(data)  # Returns dict with all stats
mean = scirs2.mean_py(data)
std = scirs2.std_py(data, 0)      # ddof=0 for population
var = scirs2.var_py(data, 1)      # ddof=1 for sample

# Percentiles
median = scirs2.median_py(data)
p75 = scirs2.percentile_py(data, 75.0)
iqr = scirs2.iqr_py(data)

# Correlation
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
corr = scirs2.correlation_py(x, y)  # Returns 1.0
cov = scirs2.covariance_py(x, y, 1)
```

### FFT (`fft`)

⚠️ **Performance Warning**: FFT operations are **62x slower** than NumPy due to lack of FFTW integration. **Use NumPy/SciPy for all FFT operations.**

```python
import numpy as np
import scirs2

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

# FFT
result = scirs2.fft_py(data)      # {'real', 'imag'}
real, imag = result['real'], result['imag']

# Inverse FFT
reconstructed = scirs2.ifft_py(
    np.array(real), np.array(imag)
)

# Real FFT (for real-valued signals)
rfft = scirs2.rfft_py(data)
irfft = scirs2.irfft_py(
    np.array(rfft['real']),
    np.array(rfft['imag']),
    len(data)
)

# DCT
dct = scirs2.dct_py(data, 2)      # Type-II DCT
idct = scirs2.idct_py(np.array(dct), 2)

# Helper functions
freqs = scirs2.fftfreq_py(len(data), 1.0)
rfreqs = scirs2.rfftfreq_py(len(data), 1.0)
shifted = scirs2.fftshift_py(data)
fast_len = scirs2.next_fast_len_py(100, False)
```

### Clustering (`cluster`)

```python
import numpy as np
import scirs2

# Generate sample data
X = np.vstack([
    np.random.randn(50, 2) + [0, 0],
    np.random.randn(50, 2) + [5, 5],
])

# K-Means clustering
kmeans = scirs2.KMeans(n_clusters=2)
kmeans.fit(X)
labels = kmeans.labels
inertia = kmeans.inertia_

# Evaluation metrics
silhouette = scirs2.silhouette_score_py(X, labels)
davies_bouldin = scirs2.davies_bouldin_score_py(X, labels)
calinski = scirs2.calinski_harabasz_score_py(X, labels)

# Preprocessing
X_std = scirs2.standardize_py(X, True)   # Zero mean, unit variance
X_norm = scirs2.normalize_py(X, "l2")    # L2 normalization
```

### Time Series (`series`)

```python
import numpy as np
import scirs2

# Create time series
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
ts = scirs2.PyTimeSeries(data, None)

# Descriptive statistics
stats = ts.describe()

# Differencing
diff1 = scirs2.apply_differencing(ts, 1)        # First difference
seasonal = scirs2.apply_seasonal_differencing(ts, 4)  # Seasonal

# ARIMA modeling
arima = scirs2.PyARIMA(1, 1, 0)  # AR(1), I(1), MA(0)
arima.fit(ts)
forecast = arima.forecast(5)
params = arima.get_params()
print(arima.summary())

# Box-Cox transformation
result = scirs2.boxcox_transform(ts, None)  # Auto-select lambda
transformed = result['transformed']
lambda_val = result['lambda']
recovered = scirs2.boxcox_inverse(np.array(transformed), lambda_val)

# Stationarity test
adf = scirs2.adf_test(ts, None)
print(f"ADF statistic: {adf['statistic']}, p-value: {adf['p_value']}")

# STL decomposition
decomp = scirs2.stl_decomposition(ts, 4)
trend = decomp['trend']
seasonal = decomp['seasonal']
residual = decomp['residual']
```

## Performance

**Benchmark Summary** (79 tests across 3 modules):
- **Overall Average Speedup**: 1.19x
- **Win Rate**: 12.7% (10/79 tests)
- **Strength**: Complex statistics on small-medium data

### Where SciRS2 Excels 🏆

| Operation | Data Size | Speedup | Use Case |
|-----------|-----------|---------|----------|
| **Skewness** | 100-1,000 | **6-23x** | Distribution shape analysis |
| **Kurtosis** | 100-1,000 | **5-24x** | Distribution tail analysis |
| **Pearson correlation** | 100 | **3x** | Small dataset correlation |
| **IQR** | 100 | **1.7x** | Quartile calculations |

**Best Use Cases**:
- Complex statistical analysis on datasets <10,000 elements
- Higher-order moments (skewness, kurtosis)
- Distribution shape analysis
- Type-safe Rust integration

### Where NumPy/SciPy Win ⚠️

| Operation | Slowdown | Reason |
|-----------|----------|--------|
| **Linear algebra** (det, inv, solve) | **200-700x** | Missing BLAS/LAPACK |
| **FFT** operations | **62x** | Missing FFTW |
| **Basic stats** (mean, std) on large data | **10-50x** | NumPy C optimization |

**Use NumPy/SciPy for**:
- All linear algebra operations
- All FFT operations
- Basic statistics (mean, std, var, median)
- Large datasets (>10,000 elements)
- Performance-critical code paths

### Decision Matrix

| Your Use Case | Recommended Tool | Reason |
|---------------|------------------|--------|
| **Complex stats, small data** (<1K) | ✅ **scirs2** | 4-23x faster |
| **Basic stats** (mean/std/var) | ❌ NumPy | 10-50x faster |
| **Linear algebra** (any size) | ❌ SciPy/NumPy | 200-700x faster |
| **FFT operations** | ❌ NumPy/SciPy | 62x faster |
| **Large datasets** (>10K) | ❌ NumPy/SciPy | 30-100x faster |
| **Type-safe Rust integration** | ✅ **scirs2** | Native Rust types |

### Recommended Hybrid Approach

```python
import numpy as np
import scipy.stats
import scipy.linalg
import scirs2

# Generate data
data = np.random.randn(1000)

# ✅ Use NumPy for basics (FAST)
mean = np.mean(data)
std = np.std(data)

# ✅ Use scirs2 for complex stats (FASTER!)
skewness = scirs2.skew_py(data)      # 6x faster!
kurtosis = scirs2.kurtosis_py(data)  # 5x faster!

# ✅ Use SciPy for linalg (MUCH FASTER)
matrix = np.random.randn(100, 100)
det = scipy.linalg.det(matrix)       # 377x faster than scirs2!
inv = scipy.linalg.inv(matrix)       # 714x faster than scirs2!

# ✅ Use NumPy for FFT (MUCH FASTER)
signal = np.random.randn(1024)
spectrum = np.fft.rfft(signal)       # 62x faster than scirs2!
```

**Full benchmark report**: See [MASTER-PERFORMANCE-REPORT.md](/tmp/MASTER-PERFORMANCE-REPORT.md)

## Type Hints

SciRS2 includes type stubs (`.pyi` files) for better IDE support:

```python
# Your IDE will show type hints and autocompletion
import scirs2
result = scirs2.det_py(matrix)  # IDE knows this returns float
```

## Development

### Building from Source

```bash
# Install Rust and maturin
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin

# Build
cd scirs2-python
maturin develop --release

# Run tests
pip install pytest numpy
pytest tests/
```

### Project Structure

```
scirs2-python/
├── src/           # Rust source with PyO3 bindings
│   ├── lib.rs     # Module registration
│   ├── cluster.rs # Clustering bindings
│   ├── series.rs  # Time series bindings
│   ├── linalg.rs  # Linear algebra bindings
│   ├── stats.rs   # Statistics bindings
│   └── fft.rs     # FFT bindings
├── tests/         # Python tests
├── scirs2.pyi     # Type stubs
└── pyproject.toml
```

## Related Projects

- [SciRS2](https://github.com/cool-japan/scirs) - Core Rust library
- [NumPy](https://numpy.org/) - Array operations
- [SciPy](https://scipy.org/) - Scientific Python (API inspiration)

## License

Dual-licensed under MIT OR Apache-2.0
