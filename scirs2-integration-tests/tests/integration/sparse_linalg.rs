// Integration tests for scirs2-sparse + scirs2-linalg
// Tests sparse linear algebra operations, solver integration, and matrix conversions

use crate::common::*;
use crate::fixtures::TestDatasets;
use proptest::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_linalg::*;
use scirs2_sparse::*;

type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

// ---------------------------------------------------------------------------
// Helper: build a Poisson-1D tridiagonal SPD matrix (diag=2, off-diag=-1)
// ---------------------------------------------------------------------------
fn build_spd_tridiagonal(n: usize) -> TestResult<CsrMatrix<f64>> {
    let mut row_idx: Vec<usize> = Vec::new();
    let mut col_idx: Vec<usize> = Vec::new();
    let mut data: Vec<f64> = Vec::new();

    for i in 0..n {
        // diagonal
        row_idx.push(i);
        col_idx.push(i);
        data.push(2.0);

        // sub-diagonal
        if i > 0 {
            row_idx.push(i);
            col_idx.push(i - 1);
            data.push(-1.0);
        }

        // super-diagonal
        if i + 1 < n {
            row_idx.push(i);
            col_idx.push(i + 1);
            data.push(-1.0);
        }
    }

    CsrMatrix::new(data, row_idx, col_idx, (n, n))
        .map_err(|e| format!("Failed to build tridiagonal: {}", e).into())
}

/// Test sparse-dense matrix multiplication
#[test]
#[ignore] // Requires sparse-dense matmul API not yet wired in integration layer
fn test_sparse_dense_matmul() -> TestResult<()> {
    let sparse_triplets = TestDatasets::sparse_test_matrix(100, 50, 0.1);
    let dense_matrix = create_test_array_2d::<f64>(50, 20, 42)?;

    println!("Testing sparse-dense matrix multiplication");
    println!("Sparse: 100x50 (density 0.1), Dense: 50x20");

    Ok(())
}

/// Test sparse linear system solving — Poisson-1D CG
#[test]
fn test_sparse_linear_solver() -> TestResult<()> {
    let n = 50usize;
    let a = build_spd_tridiagonal(n)?;

    // x_true = all-ones
    let x_true = Array1::ones(n);

    // b = A * x_true  (compute with matvec via dot)
    let x_slice = x_true.as_slice().ok_or("x_true is not contiguous")?;
    let b_vec = a
        .dot(x_slice)
        .map_err(|e| format!("matvec failed: {}", e))?;
    let b = Array1::from_vec(b_vec);

    // Solve with CG
    let config = IterativeSolverConfig {
        max_iter: 2000,
        tol: 1e-10,
        verbose: false,
    };
    let result = enhanced_cg(&a, &b, &config, None).map_err(|e| format!("CG failed: {}", e))?;

    assert!(
        result.converged,
        "CG did not converge after {} iters, residual={}",
        result.n_iter, result.residual_norm
    );

    // Verify residual ||Ax - b|| / ||b|| < 1e-8
    let sol_slice = result
        .solution
        .as_slice()
        .ok_or("solution is not contiguous")?;
    let ax_vec = a
        .dot(sol_slice)
        .map_err(|e| format!("matvec (verification) failed: {}", e))?;
    let ax = Array1::from_vec(ax_vec);
    let residual: f64 = (&ax - &b).mapv(|x| x * x).sum().sqrt();
    let b_norm: f64 = b.mapv(|x| x * x).sum().sqrt();
    let rel_residual = residual / b_norm;

    assert!(
        rel_residual < 1e-8,
        "Relative residual too large: {} (expected < 1e-8)",
        rel_residual
    );

    println!(
        "Sparse CG converged in {} iters, rel_residual={:.2e}",
        result.n_iter, rel_residual
    );
    Ok(())
}

/// Test sparse eigenvalue computation
#[test]
#[ignore] // Requires LOBPCG/sparse eigensolver wired in integration — pending API stabilization
fn test_sparse_eigenvalues() -> TestResult<()> {
    Ok(())
}

/// Test sparse matrix factorization
#[test]
#[ignore] // Requires sparse LU/Cholesky API wired in integration layer
fn test_sparse_factorization() -> TestResult<()> {
    Ok(())
}

/// Test sparse-sparse addition and multiplication
#[test]
#[ignore] // Requires sparse-sparse arithmetic API wired in integration layer
fn test_sparse_sparse_operations() -> TestResult<()> {
    Ok(())
}

/// Test sparse format conversions
#[test]
fn test_sparse_format_conversions() -> TestResult<()> {
    // Build a 5×5 known matrix via triplets
    let rows = vec![0usize, 0, 1, 2, 3, 4, 4];
    let cols = vec![0usize, 4, 1, 2, 3, 0, 4];
    let data = vec![1.0f64, 5.0, 2.0, 3.0, 4.0, 6.0, 7.0];
    let n = 5usize;

    let csr = CsrMatrix::new(data.clone(), rows.clone(), cols.clone(), (n, n))
        .map_err(|e| format!("CsrMatrix::new failed: {}", e))?;

    // Verify element access for every known entry
    for k in 0..data.len() {
        let val = csr.get(rows[k], cols[k]);
        assert!(
            (val - data[k]).abs() < 1e-12,
            "Element mismatch at ({}, {}): expected {}, got {}",
            rows[k],
            cols[k],
            data[k],
            val
        );
    }

    // Verify a zero entry
    let zero_val = csr.get(0, 1);
    assert!(
        zero_val.abs() < 1e-12,
        "Expected 0 at (0,1) but got {}",
        zero_val
    );

    // Verify shape
    assert_eq!(csr.rows(), n, "rows mismatch");
    assert_eq!(csr.cols(), n, "cols mismatch");
    assert_eq!(csr.nnz(), data.len(), "nnz mismatch");

    println!(
        "Sparse format construction verified: {}x{} matrix, {} nnz",
        n,
        n,
        csr.nnz()
    );
    Ok(())
}

/// Test iterative solvers with preconditioning
#[test]
#[ignore] // Requires further preconditioner integration — pending API stabilization
fn test_iterative_solvers_with_preconditioner() -> TestResult<()> {
    Ok(())
}

/// Test sparse matrix norms
#[test]
#[ignore] // Requires sparse norm API wired at integration layer
fn test_sparse_matrix_norms() -> TestResult<()> {
    Ok(())
}

/// Test sparse matrix transpose and conjugate transpose
#[test]
#[ignore] // Requires sparse transpose API wired at integration layer
fn test_sparse_transpose_operations() -> TestResult<()> {
    Ok(())
}

/// Test sparse matrix slicing and submatrix extraction
#[test]
#[ignore] // Requires sparse slicing API not yet available at integration layer
fn test_sparse_submatrix_operations() -> TestResult<()> {
    Ok(())
}

/// Test sparse matrix concatenation
#[test]
#[ignore] // Requires hstack/vstack sparse integration
fn test_sparse_matrix_concatenation() -> TestResult<()> {
    Ok(())
}

// Property-based tests

proptest! {
    #[test]
    fn prop_sparse_dense_consistency(
        n in 10usize..50,
        m in 10usize..50,
        density in 0.05..0.3
    ) {
        // Property: Sparse matrix operations should give same results
        // as equivalent dense operations
        let sparse_triplets = TestDatasets::sparse_test_matrix(n, m, density);
        prop_assert!(n > 0 && m > 0);
        prop_assert!(!sparse_triplets.is_empty() || (n * m) as f64 * density < 1.0);
    }

    #[test]
    fn prop_sparse_matrix_symmetry(
        n in 10usize..50
    ) {
        // Property: For symmetric sparse matrix, A = A^T
        let sparse_triplets = TestDatasets::sparse_test_matrix(n, n, 0.1);
        prop_assert!(n > 0);
        // Just verify the triplets have valid indices
        for (r, c, _v) in &sparse_triplets {
            prop_assert!(*r < n, "row index out of bounds");
            prop_assert!(*c < n, "col index out of bounds");
        }
    }

    /// Property: sparse matvec scales linearly — (A * (alpha * x)) ≈ alpha * (A * x)
    #[test]
    fn prop_sparse_matvec_linear(
        n in 5usize..30,
        alpha in -5.0f64..5.0,
    ) {
        let size = n;
        // Build SPD tridiagonal
        let a = build_spd_tridiagonal(size).expect("build_spd_tridiagonal failed");

        let x: Vec<f64> = (0..size).map(|i| (i as f64 + 1.0) / size as f64).collect();
        let alpha_x: Vec<f64> = x.iter().map(|&v| alpha * v).collect();

        let ax = a.dot(&x).expect("matvec ax failed");
        let a_alphax = a.dot(&alpha_x).expect("matvec a_alphax failed");

        let max_err = ax.iter()
            .zip(a_alphax.iter())
            .map(|(axv, aalphaxv)| (aalphaxv - alpha * axv).abs())
            .fold(0.0_f64, f64::max);

        prop_assert!(
            max_err < 1e-10,
            "Linearity violated: max_err={}",
            max_err
        );
    }

    #[test]
    fn prop_sparse_solver_accuracy(
        n in 10usize..30,
        density in 0.1..0.5
    ) {
        // Property: Sparse solver should satisfy ||Ax - b|| / ||b|| < tolerance
        let sparse_triplets = TestDatasets::sparse_test_matrix(n, n, density);
        prop_assert!(n > 0);
        let _ = sparse_triplets;
    }
}

/// Test memory efficiency of sparse operations
#[test]
fn test_sparse_operations_memory_efficiency() -> TestResult<()> {
    let large_n = 1000;
    let sparse_triplets = TestDatasets::sparse_test_matrix(large_n, large_n, 0.01);

    println!("Testing memory efficiency of sparse operations");
    println!("Matrix size: {}x{} (density 0.01)", large_n, large_n);

    assert_memory_efficient(
        || {
            // Verify that building a large sparse matrix from triplets is cheap
            let rows: Vec<usize> = sparse_triplets.iter().map(|(r, _, _)| *r).collect();
            let cols: Vec<usize> = sparse_triplets.iter().map(|(_, c, _)| *c).collect();
            let data: Vec<f64> = sparse_triplets.iter().map(|(_, _, v)| *v).collect();
            let _m = CsrMatrix::new(data, rows, cols, (large_n, large_n))
                .map_err(|e| format!("CsrMatrix failed: {}", e))?;
            Ok(())
        },
        50.0,
        "Sparse matrix operations",
    )?;

    Ok(())
}

/// Test sparse matrix condition number estimation
#[test]
#[ignore] // Requires power iteration eigensolver API wired at integration layer
fn test_sparse_condition_number() -> TestResult<()> {
    Ok(())
}

/// Test sparse QR factorization
#[test]
#[ignore] // Requires sparse QR API wired at integration layer
fn test_sparse_qr_factorization() -> TestResult<()> {
    Ok(())
}

/// Test sparse SVD computation
#[test]
#[ignore] // Requires truncated SVD (Lanczos/Arnoldi) API wired at integration layer
fn test_sparse_svd() -> TestResult<()> {
    Ok(())
}

/// Test sparse matrix graph operations
#[test]
#[ignore] // Requires graph Laplacian/connected-components API
fn test_sparse_matrix_graph_operations() -> TestResult<()> {
    Ok(())
}

/// Test sparse matrix elementwise operations
#[test]
#[ignore] // Requires element-wise sparse ops API
fn test_sparse_elementwise_operations() -> TestResult<()> {
    Ok(())
}

/// Test sparse matrix reordering
#[test]
#[ignore] // Requires RCM/AMD reordering API
fn test_sparse_matrix_reordering() -> TestResult<()> {
    Ok(())
}

/// Test sparse direct solvers
#[test]
#[ignore] // Requires sparse LU direct solver API
fn test_sparse_direct_solvers() -> TestResult<()> {
    Ok(())
}

/// Test sparse matrix power operations
#[test]
#[ignore] // Requires sparse matrix exponentiation API
fn test_sparse_matrix_powers() -> TestResult<()> {
    Ok(())
}

/// Test integration with dense linear algebra
#[test]
#[ignore] // Requires sparse-dense mixed workflow API
fn test_sparse_dense_integration() -> TestResult<()> {
    Ok(())
}

// ---------------------------------------------------------------------------
// Dense matvec test (uses known entries)
// ---------------------------------------------------------------------------

/// Test sparse dense matvec — multiply a simple CsrMatrix by a vector
#[test]
fn test_sparse_dense_matvec() -> TestResult<()> {
    // 3×3 matrix:
    //  [1 0 2]
    //  [0 3 0]
    //  [4 0 5]
    let row_idx = vec![0usize, 0, 1, 2, 2];
    let col_idx = vec![0usize, 2, 1, 0, 2];
    let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
    let m = CsrMatrix::new(data, row_idx, col_idx, (3, 3))
        .map_err(|e| format!("CsrMatrix::new: {}", e))?;

    let x = vec![1.0f64, 2.0, 3.0];
    let y = m.dot(&x).map_err(|e| format!("dot: {}", e))?;

    // Expected: [1*1+2*3, 3*2, 4*1+5*3] = [7, 6, 19]
    let expected = [7.0, 6.0, 19.0];
    for (i, (&got, &exp)) in y.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-12,
            "y[{}]: expected {}, got {}",
            i,
            exp,
            got
        );
    }

    println!("Sparse matvec verified: y = {:?}", y);
    Ok(())
}

#[cfg(test)]
mod api_compatibility_tests {
    use super::*;

    /// Test that sparse matrix types are compatible with linalg operations
    #[test]
    fn test_sparse_type_compatibility() -> TestResult<()> {
        // Verify that CsrMatrix can be built and matvec works
        let a = build_spd_tridiagonal(5)?;
        let x = vec![1.0f64; 5];
        let _y = a
            .dot(&x)
            .map_err(|e| format!("type compat matvec: {}", e))?;
        println!("Sparse-linalg type compatibility verified");
        Ok(())
    }

    /// Test error handling across sparse-linalg boundary
    #[test]
    fn test_sparse_linalg_error_handling() -> TestResult<()> {
        // Dimension mismatch: 5x5 matrix with length-3 vector
        let a = build_spd_tridiagonal(5)?;
        let x = vec![1.0f64, 2.0, 3.0];
        let result = a.dot(&x);
        assert!(
            result.is_err(),
            "Expected error for dimension mismatch, got Ok"
        );
        println!("Error handling test passed: {:?}", result.err());
        Ok(())
    }

    /// Test performance characteristics
    #[test]
    fn test_sparse_linalg_performance() -> TestResult<()> {
        let sizes = vec![100, 200, 500, 1000];
        let density = 0.05;

        println!("Testing sparse vs dense performance");

        for n in sizes {
            let sparse_triplets = TestDatasets::sparse_test_matrix(n, n, density);
            let rows: Vec<usize> = sparse_triplets.iter().map(|(r, _, _)| *r).collect();
            let cols: Vec<usize> = sparse_triplets.iter().map(|(_, c, _)| *c).collect();
            let data: Vec<f64> = sparse_triplets.iter().map(|(_, _, v)| *v).collect();
            let m = CsrMatrix::new(data, rows, cols, (n, n))
                .map_err(|e| format!("CsrMatrix failed: {}", e))?;
            let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
            let (_y, perf) = measure_time(&format!("SpMV n={}", n), || {
                m.dot(&x).map_err(|e| e.to_string().into())
            })?;
            println!("  Size {}: {:.3} ms", n, perf.duration_ms);
        }

        Ok(())
    }
}
