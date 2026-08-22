"""
metrics/bootstrap.py
====================
Vectorized bootstrap confidence interval.

Function:
    bootstrap_ci(values, n_bootstrap, alpha, seed)
        → (lower, upper, mean)

Design:
    Fully vectorized — one matrix operation, no Python loop.
    A naive loop of n_bootstrap iterations on K=30 values
    costs ~30000 Python function calls per CI — unacceptable
    when called 5 methods × 10 instances × 6 budgets × 4 datasets.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from config import BOOTSTRAP_N, BOOTSTRAP_ALPHA, GLOBAL_SEED


def bootstrap_ci(values, n_bootstrap=BOOTSTRAP_N,
                 alpha=BOOTSTRAP_ALPHA, seed=GLOBAL_SEED):
    """
    Percentile bootstrap CI on the mean of `values`.

    Vectorized implementation:
        1. Draw n_bootstrap × K index matrix (one operation)
        2. Index values array to get bootstrap samples (one operation)
        3. Compute row means (one operation)
        4. Return 2.5th and 97.5th percentiles

    Parameters
    ----------
    values      : array-like, shape (K,) — per-run metric values
    n_bootstrap : int   — number of bootstrap resamplings (default: 1000)
    alpha       : float — significance level (default: 0.05 → 95% CI)
    seed        : int   — random seed for reproducibility

    Returns
    -------
    ci_lower : float — (alpha/2)-th percentile of bootstrap distribution
    ci_upper : float — (1 - alpha/2)-th percentile
    mean     : float — sample mean of `values` (point estimate)

    Notes
    -----
    Returns (nan, nan, nan) if values contains fewer than 2 elements
    or all-nan values.
    """
    arr = np.asarray(values, dtype=float)

    if arr.ndim != 1 or len(arr) < 2:
        nan = float('nan')
        return nan, nan, nan

    if np.all(np.isnan(arr)):
        nan = float('nan')
        return nan, nan, nan

    # Remove NaN before bootstrapping
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        m = float(arr[0]) if len(arr) == 1 else float('nan')
        return m, m, m

    rng  = np.random.RandomState(seed)
    K    = len(arr)

    # Vectorized: shape (n_bootstrap, K) — one call to rng
    idx  = rng.randint(0, K, size=(n_bootstrap, K))
    boot = arr[idx].mean(axis=1)   # shape (n_bootstrap,)

    lo = alpha / 2 * 100
    hi = (1 - alpha / 2) * 100

    return (
        float(np.percentile(boot, lo)),
        float(np.percentile(boot, hi)),
        float(arr.mean()),
    )


def bootstrap_ci_pair(values_A, values_B, n_bootstrap=BOOTSTRAP_N,
                      seed=GLOBAL_SEED):
    """
    Bootstrap CI on the DIFFERENCE in means: mean(A) - mean(B).
    Useful for reporting Δ(Spearman) = ABC - B1, etc.

    FIXED (this pass): this function previously had no edge-case
    handling at all, unlike bootstrap_ci() above -- an empty A or B
    would raise ValueError from rng.randint(0, 0, ...), and NaN values
    in either array would silently propagate into the bootstrap
    distribution and corrupt np.percentile() in unpredictable ways
    depending on their position. Now mirrors bootstrap_ci()'s handling:
    NaNs are dropped from each array independently, and (nan, nan, nan)
    is returned if either array has fewer than 2 valid values after
    that filtering.

    Parameters
    ----------
    values_A, values_B : array-like — per-run metric values for the two
                          methods being compared (not required to be the
                          same length; K below is min(len(A), len(B)))
    n_bootstrap : int
    seed        : int

    Returns
    -------
    diff_ci_lower : float
    diff_ci_upper : float
    diff_mean     : float — point estimate of mean(A) - mean(B)

    Notes
    -----
    Returns (nan, nan, nan) if either input has fewer than 2 non-NaN
    elements after NaN removal.
    """
    A = np.asarray(values_A, dtype=float)
    B = np.asarray(values_B, dtype=float)
    A = A[~np.isnan(A)]
    B = B[~np.isnan(B)]

    if len(A) < 2 or len(B) < 2:
        nan = float('nan')
        return nan, nan, nan

    rng = np.random.RandomState(seed)
    K   = min(len(A), len(B))

    idx_A  = rng.randint(0, len(A), size=(n_bootstrap, K))
    idx_B  = rng.randint(0, len(B), size=(n_bootstrap, K))
    boot   = A[idx_A].mean(axis=1) - B[idx_B].mean(axis=1)

    return (
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
        float(A.mean() - B.mean()),
    )


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng    = np.random.RandomState(42)
    values = rng.normal(loc=0.75, scale=0.05, size=30)

    lo, hi, mean = bootstrap_ci(values)
    print(f"bootstrap_ci : mean={mean:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
    print(f"  width = {hi - lo:.4f}")

    # Edge case: all identical
    lo2, hi2, mean2 = bootstrap_ci(np.ones(30) * 0.9)
    print(f"bootstrap_ci (all same): [{lo2}, {hi2}]  mean={mean2}")

    # Edge case: single value
    lo3, hi3, mean3 = bootstrap_ci([0.5])
    print(f"bootstrap_ci (single):  [{lo3}, {hi3}]  mean={mean3}")

    # Pair difference
    A = rng.normal(0.8, 0.04, 30)
    B = rng.normal(0.6, 0.06, 30)
    dl, dh, dm = bootstrap_ci_pair(A, B)
    print(f"bootstrap_ci_pair (A-B): diff={dm:.4f}  95% CI=[{dl:.4f}, {dh:.4f}]")

    # NEW — Edge cases for bootstrap_ci_pair (previously untested, previously unhandled)
    dl_e, dh_e, dm_e = bootstrap_ci_pair([], [0.5, 0.6, 0.7])
    print(f"bootstrap_ci_pair (A empty): [{dl_e}, {dh_e}]  diff={dm_e}")

    dl_n, dh_n, dm_n = bootstrap_ci_pair([np.nan, np.nan], [0.5, 0.6, 0.7])
    print(f"bootstrap_ci_pair (A all-NaN): [{dl_n}, {dh_n}]  diff={dm_n}")

    print("bootstrap.py OK")