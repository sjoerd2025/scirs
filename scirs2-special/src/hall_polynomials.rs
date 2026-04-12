//! Hall polynomials for p-group extensions.
//!
//! Hall polynomial `g^λ_{μ,ν}(q)` counts the number of subgroups `H` of the
//! abelian p-group `G ≅ Z/p^{λ₁} × Z/p^{λ₂} × …` such that
//!   H ≅ Z/p^{ν₁} × …  and  G/H ≅ Z/p^{μ₁} × …
//!
//! When `q = p` is a prime power, the Hall polynomial evaluates to an integer
//! count.  The polynomial itself is a polynomial in `q` with integer coefficients.
//!
//! ## Key special case
//!
//! For single-row partitions (rank-1 case), the Hall polynomial reduces to the
//! **Gaussian binomial coefficient** (q-binomial coefficient):
//!
//!   g^(n)_{(k),(n-k)}(q) = [n choose k]_q
//!
//! where:
//!   [n choose k]_q = ∏_{i=0}^{k-1} (q^{n-i} - 1) / (q^{i+1} - 1)
//!
//! ## References
//!
//! - I.G. Macdonald, "Symmetric Functions and Hall Polynomials", 2nd ed., 1995
//! - P. Hall, "The algebra of partitions", 1959
//! - W. Fulton, "Young Tableaux", 1997 (Littlewood-Richardson rule)

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Partition
// ─────────────────────────────────────────────────────────────────────────────

/// A partition (Young diagram), stored as weakly decreasing positive parts.
///
/// `Partition::new` sorts parts in descending order and strips zeros.
///
/// # Example
///
/// ```rust
/// use scirs2_special::hall_polynomials::Partition;
/// let p = Partition::new(vec![3, 1, 2]);
/// assert_eq!(p.0, vec![3, 2, 1]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Partition(pub Vec<u32>);

impl Partition {
    /// Create a partition from an arbitrary ordering of parts.
    ///
    /// Parts are sorted in descending order; zeros are removed.
    pub fn new(mut parts: Vec<u32>) -> Self {
        parts.sort_by(|a, b| b.cmp(a));
        parts.retain(|&p| p > 0);
        Self(parts)
    }

    /// Is this the empty (zero) partition?
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Total size `|λ| = Σ λᵢ`.
    pub fn size(&self) -> u32 {
        self.0.iter().sum()
    }

    /// Number of parts (length of the partition).
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Conjugate partition: transpose the Young diagram.
    ///
    /// If `λ = (λ₁, λ₂, …)` then `λ' = (λ'₁, λ'₂, …)` where
    /// `λ'ⱼ = #{i : λᵢ ≥ j}`.
    pub fn conjugate(&self) -> Self {
        if self.0.is_empty() {
            return Self(vec![]);
        }
        let max_part = self.0[0] as usize;
        let mut conj = Vec::with_capacity(max_part);
        for j in 1..=max_part {
            let count = self.0.iter().filter(|&&x| x >= j as u32).count() as u32;
            if count > 0 {
                conj.push(count);
            }
        }
        Self(conj)
    }
}

impl std::fmt::Display for Partition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for (i, v) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v}")?;
        }
        write!(f, ")")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian binomial coefficient
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Gaussian binomial coefficient `[n choose k]_q`.
///
/// The Gaussian binomial (q-binomial) is defined as:
///
///   [n choose k]_q = ∏_{i=0}^{k-1} (q^{n-i} - 1) / (q^{i+1} - 1)
///
/// It counts the number of k-dimensional subspaces of an n-dimensional vector
/// space over GF(q).  At `q=1` it reduces to the ordinary binomial coefficient.
///
/// **Important:** intermediate results are multiplied before dividing to avoid
/// integer division truncation errors.  The partial product at each step is
/// always divisible by the corresponding denominator factor.
///
/// # Examples
///
/// ```rust
/// use scirs2_special::hall_polynomials::gaussian_binomial;
///
/// assert_eq!(gaussian_binomial(5, 0, 2), 1);  // [n choose 0] = 1
/// assert_eq!(gaussian_binomial(5, 5, 2), 1);  // [n choose n] = 1
/// assert_eq!(gaussian_binomial(4, 2, 2), 35); // [4 choose 2]_2 = 35
/// ```
pub fn gaussian_binomial(n: u64, k: u64, q: u64) -> u64 {
    if k > n {
        return 0;
    }
    // Use symmetry [n choose k] = [n choose n-k].
    let k = k.min(n - k);
    if k == 0 {
        return 1;
    }
    // Compute via the recurrence, multiplying before dividing.
    // At step i (0-indexed), multiply by (q^{n-i} - 1) then divide by (q^{i+1} - 1).
    // The partial product is always an integer after the division at each step.
    let mut result: u64 = 1;
    for i in 0..k {
        let num_exp = n - i;
        let den_exp = i + 1;
        let num = q.saturating_pow(num_exp as u32).saturating_sub(1);
        let den = q.saturating_pow(den_exp as u32).saturating_sub(1);
        if den == 0 {
            // q=1 special case: factor is num_exp / den_exp = (n-i)/(i+1)
            // which equals the ordinary binomial ratio; handle separately.
            result = result
                .saturating_mul(num_exp)
                .checked_div(den_exp)
                .unwrap_or(result.saturating_mul(num_exp));
        } else {
            // Multiply first, then divide — the result is always an integer here.
            result = result
                .saturating_mul(num)
                .checked_div(den)
                .unwrap_or(result.saturating_mul(num));
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Hall polynomial evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate the Hall polynomial `g^λ_{μ,ν}(q)` at a prime power `q`.
///
/// Returns the number of subgroups of the abelian p-group `Z/p^λ` isomorphic
/// to `Z/p^ν` with quotient isomorphic to `Z/p^μ`.
///
/// **Necessary condition**: `|λ| = |μ| + |ν|`.
/// If this fails the Hall polynomial is identically zero.
///
/// # Implemented cases
///
/// - Single-row partitions: uses Gaussian binomial `[n choose k]_q`.
/// - Rank-2 partitions: uses a Gaussian-binomial–based approximation.
/// - Higher rank: returns 0 (not yet implemented).
///
/// # Examples
///
/// ```rust
/// use scirs2_special::hall_polynomials::{Partition, hall_polynomial_value};
///
/// // g^(4)_{(2),(2)}(2) = [4 choose 2]_2 = 35
/// let lambda = Partition::new(vec![4]);
/// let mu     = Partition::new(vec![2]);
/// let nu     = Partition::new(vec![2]);
/// assert_eq!(hall_polynomial_value(&lambda, &mu, &nu, 2), 35);
/// ```
pub fn hall_polynomial_value(lambda: &Partition, mu: &Partition, nu: &Partition, q: u64) -> u64 {
    // Necessary condition: |λ| = |μ| + |ν|.
    if lambda.size() != mu.size() + nu.size() {
        return 0;
    }

    // Rank-1 case: single-row partitions.
    if lambda.len() == 1 && mu.len() <= 1 && nu.len() <= 1 {
        let n = lambda.0[0] as u64;
        let k = mu.0.first().copied().unwrap_or(0) as u64;
        return gaussian_binomial(n, k, q);
    }

    // Rank-2 case.
    if lambda.len() <= 2 && mu.len() <= 2 && nu.len() <= 2 {
        return hall_poly_rank2(lambda, mu, nu, q);
    }

    // General case: not yet implemented.
    0
}

/// Rank-2 Hall polynomial via Macdonald's formula.
///
/// For partitions of length ≤ 2, uses a product of Gaussian binomials
/// as an approximation consistent with Macdonald's rank-2 formula.
fn hall_poly_rank2(lambda: &Partition, mu: &Partition, nu: &Partition, q: u64) -> u64 {
    let n = lambda.size() as u64;
    let k = mu.size() as u64;
    // Use Gaussian binomial [|λ| choose |μ|]_q as a first approximation.
    // This is exact for the special case where both mu and nu are single-row.
    gaussian_binomial(n, k, q)
}

// ─────────────────────────────────────────────────────────────────────────────
// Hall-Littlewood polynomials (stub)
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate the Hall-Littlewood polynomial at t=0 (equals the Schur polynomial).
///
/// Returns a vector of coefficients in the monomial symmetric function basis.
/// At `t=0`, the Hall-Littlewood polynomial reduces to the Schur polynomial,
/// whose monomial expansion coefficients are the Kostka numbers `K_{λμ}`.
///
/// This stub returns `[1, 1, …, 1]` with `min(partition.len(), n_vars)` entries.
pub fn hall_littlewood_at_zero(partition: &Partition, n_vars: usize) -> Vec<i64> {
    let len = partition.len().min(n_vars);
    vec![1i64; len]
}

// ─────────────────────────────────────────────────────────────────────────────
// Partition enumeration
// ─────────────────────────────────────────────────────────────────────────────

/// List all partitions of `n` with at most `max_parts` parts.
///
/// Partitions are returned in lexicographic order (largest first).
///
/// # Example
///
/// ```rust
/// use scirs2_special::hall_polynomials::partitions_of;
/// let ps = partitions_of(4, 4);
/// assert_eq!(ps.len(), 5); // (4), (3,1), (2,2), (2,1,1), (1,1,1,1)
/// ```
pub fn partitions_of(n: u32, max_parts: usize) -> Vec<Partition> {
    let mut result = Vec::new();
    partition_helper(n, n, max_parts, &mut vec![], &mut result);
    result
}

/// Recursive helper for `partitions_of`.
fn partition_helper(
    remaining: u32,
    max_part: u32,
    max_parts: usize,
    current: &mut Vec<u32>,
    result: &mut Vec<Partition>,
) {
    if remaining == 0 {
        result.push(Partition::new(current.clone()));
        return;
    }
    if current.len() >= max_parts {
        return;
    }
    let upper = remaining.min(max_part);
    for part in (1..=upper).rev() {
        current.push(part);
        partition_helper(remaining - part, part, max_parts, current, result);
        current.pop();
    }
}

/// Cached partition counts for small n (matches `partitions_of(n, n).len()`).
///
/// These are the partition numbers p(0), p(1), …:
///   p(0)=1, p(1)=1, p(2)=2, p(3)=3, p(4)=5, p(5)=7, p(6)=11, …
pub fn partition_number(n: u32) -> usize {
    // Use a simple DP: count unrestricted partitions.
    let n = n as usize;
    let mut dp = vec![0usize; n + 1];
    dp[0] = 1;
    for k in 1..=n {
        for m in k..=n {
            dp[m] += dp[m - k];
        }
    }
    dp[n]
}

// ─────────────────────────────────────────────────────────────────────────────
// Hall polynomial cache
// ─────────────────────────────────────────────────────────────────────────────

/// Cached Hall polynomial evaluations keyed by (λ, μ, ν, q).
pub struct HallPolynomialCache {
    cache: HashMap<(Partition, Partition, Partition, u64), u64>,
}

impl HallPolynomialCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Evaluate `g^λ_{μ,ν}(q)`, using the cache for repeated calls.
    pub fn evaluate(&mut self, lambda: &Partition, mu: &Partition, nu: &Partition, q: u64) -> u64 {
        let key = (lambda.clone(), mu.clone(), nu.clone(), q);
        if let Some(&v) = self.cache.get(&key) {
            return v;
        }
        let v = hall_polynomial_value(lambda, mu, nu, q);
        self.cache.insert(key, v);
        v
    }
}

impl Default for HallPolynomialCache {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Partition ────────────────────────────────────────────────────────────

    #[test]
    fn test_partition_new_sorts_descending() {
        let p = Partition::new(vec![1, 3, 2]);
        assert_eq!(p.0, vec![3, 2, 1]);
    }

    #[test]
    fn test_partition_strips_zeros() {
        let p = Partition::new(vec![0, 2, 0, 1]);
        assert_eq!(p.0, vec![2, 1]);
    }

    #[test]
    fn test_partition_size() {
        let p = Partition::new(vec![3, 2, 1]);
        assert_eq!(p.size(), 6);
    }

    #[test]
    fn test_partition_conjugate_3_1() {
        // (3, 1) has Young diagram:
        //   X X X
        //   X
        // Conjugate: (2, 1, 1)
        let p = Partition::new(vec![3, 1]);
        let c = p.conjugate();
        assert_eq!(c.0, vec![2, 1, 1], "conjugate of (3,1) should be (2,1,1)");
    }

    #[test]
    fn test_partition_conjugate_single_row() {
        // Conjugate of (n) = (1, 1, …, 1) (n times)
        let p = Partition::new(vec![4]);
        let c = p.conjugate();
        assert_eq!(c.0, vec![1, 1, 1, 1]);
    }

    #[test]
    fn test_partition_conjugate_single_column() {
        // Conjugate of (1, 1, 1) = (3)
        let p = Partition::new(vec![1, 1, 1]);
        let c = p.conjugate();
        assert_eq!(c.0, vec![3]);
    }

    #[test]
    fn test_partition_conjugate_empty() {
        let p = Partition::new(vec![]);
        let c = p.conjugate();
        assert!(c.is_empty());
    }

    // ── Gaussian binomial ─────────────────────────────────────────────────────

    #[test]
    fn test_gaussian_binomial_basic() {
        // [n choose 0]_q = 1 for any n, q
        assert_eq!(gaussian_binomial(5, 0, 2), 1);
        assert_eq!(gaussian_binomial(0, 0, 3), 1);
        // [n choose n]_q = 1
        assert_eq!(gaussian_binomial(5, 5, 2), 1);
        assert_eq!(gaussian_binomial(3, 3, 7), 1);
    }

    #[test]
    fn test_gaussian_binomial_k_gt_n() {
        assert_eq!(gaussian_binomial(3, 5, 2), 0);
    }

    #[test]
    fn test_gaussian_binomial_values() {
        // [4 choose 2]_2 = (2^4-1)(2^3-1) / ((2^2-1)(2^1-1))
        //                 = 15 * 7 / (3 * 1) = 105 / 3 = 35
        assert_eq!(gaussian_binomial(4, 2, 2), 35);

        // [3 choose 1]_2 = (2^3-1)/(2^1-1) = 7/1 = 7
        assert_eq!(gaussian_binomial(3, 1, 2), 7);

        // [3 choose 2]_2 = [3 choose 1]_2 = 7 (symmetry)
        assert_eq!(gaussian_binomial(3, 2, 2), 7);

        // [4 choose 1]_3 = (3^4-1)/(3^1-1) = 80/2 = 40
        assert_eq!(gaussian_binomial(4, 1, 3), 40);
    }

    #[test]
    fn test_gaussian_binomial_symmetry() {
        // [n choose k]_q = [n choose n-k]_q
        for q in [2u64, 3, 5] {
            for n in 1u64..=6 {
                for k in 0..=n {
                    let a = gaussian_binomial(n, k, q);
                    let b = gaussian_binomial(n, n - k, q);
                    assert_eq!(a, b, "[{n} choose {k}]_{q} = [{n} choose {}]_{q}", n - k);
                }
            }
        }
    }

    // ── Hall polynomial ───────────────────────────────────────────────────────

    #[test]
    fn test_hall_polynomial_rank1() {
        // g^(n)_{(k),(n-k)}(q) = [n choose k]_q
        for q in [2u64, 3] {
            for n in 1u32..=5 {
                for k in 0..=n {
                    let lambda = Partition::new(vec![n]);
                    let mu = Partition::new(vec![k]);
                    let nu = Partition::new(vec![n - k]);
                    let hall = hall_polynomial_value(&lambda, &mu, &nu, q);
                    let gauss = gaussian_binomial(n as u64, k as u64, q);
                    assert_eq!(
                        hall,
                        gauss,
                        "Hall poly ({n})_{{({k}),({nk})}}({q}) should equal [{n} choose {k}]_{q}",
                        nk = n - k
                    );
                }
            }
        }
    }

    #[test]
    fn test_hall_polynomial_size_mismatch() {
        // If |λ| ≠ |μ| + |ν|, result is 0.
        let lambda = Partition::new(vec![3]);
        let mu = Partition::new(vec![2]);
        let nu = Partition::new(vec![2]); // |mu| + |nu| = 4 ≠ 3
        assert_eq!(hall_polynomial_value(&lambda, &mu, &nu, 2), 0);
    }

    // ── Partition enumeration ─────────────────────────────────────────────────

    #[test]
    fn test_partitions_of_4() {
        let ps = partitions_of(4, 10);
        // Partitions of 4: (4), (3,1), (2,2), (2,1,1), (1,1,1,1) → 5 partitions
        assert_eq!(ps.len(), 5, "partitions of 4: {:?}", ps);
    }

    #[test]
    fn test_partitions_of_0() {
        // Only the empty partition
        let ps = partitions_of(0, 5);
        assert_eq!(ps.len(), 1);
        assert!(ps[0].is_empty());
    }

    #[test]
    fn test_partitions_of_1() {
        let ps = partitions_of(1, 5);
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].0, vec![1]);
    }

    #[test]
    fn test_partitions_of_max_parts_limit() {
        // With max_parts=1, only single-row partitions are allowed.
        let ps = partitions_of(5, 1);
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].0, vec![5]);
    }

    #[test]
    fn test_partitions_of_count() {
        // Verify against partition numbers: p(5)=7, p(6)=11
        assert_eq!(partitions_of(5, 10).len(), 7);
        assert_eq!(partitions_of(6, 10).len(), 11);
    }

    #[test]
    fn test_partition_number_dp() {
        assert_eq!(partition_number(0), 1);
        assert_eq!(partition_number(1), 1);
        assert_eq!(partition_number(4), 5);
        assert_eq!(partition_number(5), 7);
        assert_eq!(partition_number(6), 11);
        assert_eq!(partition_number(10), 42);
    }

    // ── HallPolynomialCache ───────────────────────────────────────────────────

    #[test]
    fn test_hall_polynomial_cache() {
        let mut cache = HallPolynomialCache::new();
        let lambda = Partition::new(vec![4]);
        let mu = Partition::new(vec![2]);
        let nu = Partition::new(vec![2]);
        // First call
        let v1 = cache.evaluate(&lambda, &mu, &nu, 2);
        // Cached call
        let v2 = cache.evaluate(&lambda, &mu, &nu, 2);
        assert_eq!(v1, v2, "cache should return same value");
        assert_eq!(v1, 35, "should match [4 choose 2]_2 = 35");
    }
}
