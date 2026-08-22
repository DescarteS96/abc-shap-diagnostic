"""
metrics/stability.py
====================
Ordinal stability metrics for Shapley value rankings.

Functions:
    compute_spearman_per_run(phi_runs, phi_ref)
        → (mean, std, per_run_array)

    compute_kendall_per_run(phi_runs, phi_ref)
        → (mean, std, per_run_array)

    compute_topk_overlap_per_run(phi_runs, phi_ref, k)
        → (mean, std, per_run_array)

Design principle [C1]:
    All rank correlations are computed on INDIVIDUAL runs φ̂_k vs φ_ref,
    never on the mean φ̄. Computing on the mean underestimates variability
    and produces artificially high stability scores.
"""

import numpy as np
from scipy.stats import spearmanr, kendalltau


def compute_spearman_per_run(phi_runs, phi_ref):
    """
    Spearman rank correlation between |φ̂_k| and |φ_ref| for each run k.
    Uses absolute values — feature importance is unsigned.

    Parameters
    ----------
    phi_runs : array-like, shape (K, M)
    phi_ref  : array-like, shape (M,)

    Returns
    -------
    mean    : float — mean Spearman over K runs
    std     : float — std of Spearman over K runs
    per_run : np.ndarray shape (K,)
    """
    phi_runs  = np.asarray(phi_runs,  dtype=float)
    phi_ref   = np.asarray(phi_ref,   dtype=float)
    abs_ref   = np.abs(phi_ref)

    K       = phi_runs.shape[0]
    per_run = np.zeros(K)

    for k in range(K):
        rho, _ = spearmanr(np.abs(phi_runs[k]), abs_ref)
        per_run[k] = float(rho) if not np.isnan(rho) else 0.0

    return float(per_run.mean()), float(per_run.std()), per_run


def compute_kendall_per_run(phi_runs, phi_ref):
    """
    Kendall τ between |φ̂_k| and |φ_ref| for each run k.
    Complementary to Spearman: measures concordant pair proportion.

    Parameters
    ----------
    phi_runs : array-like, shape (K, M)
    phi_ref  : array-like, shape (M,)

    Returns
    -------
    mean    : float
    std     : float
    per_run : np.ndarray shape (K,)
    """
    phi_runs = np.asarray(phi_runs, dtype=float)
    phi_ref  = np.asarray(phi_ref,  dtype=float)
    abs_ref  = np.abs(phi_ref)

    K       = phi_runs.shape[0]
    per_run = np.zeros(K)

    for k in range(K):
        tau, _ = kendalltau(np.abs(phi_runs[k]), abs_ref)
        per_run[k] = float(tau) if not np.isnan(tau) else 0.0

    return float(per_run.mean()), float(per_run.std()), per_run


def compute_topk_overlap_per_run(phi_runs, phi_ref, k):
    """
    Top-k overlap between estimated and reference feature importance rankings.
    Operationally relevant: practitioners use the top-k features, not full
    rank correlations.

    Score ∈ [0, 1]. This is a SET overlap, not an ordering measure --
    order among the top-k features is ignored (complementary to
    Spearman/Kendall, which are order-sensitive; see Section 2.4.2 for
    the rationale).
    Score = 1 : the same k features appear in both top-k sets,
                regardless of their relative order within that set
    Score = 0 : no common features in top-k

    Note: k is bounded by M if k > M.

    Parameters
    ----------
    phi_runs : array-like, shape (K, M)
    phi_ref  : array-like, shape (M,)
    k        : int

    Returns
    -------
    mean    : float
    std     : float
    per_run : np.ndarray shape (K,)
    """
    phi_runs = np.asarray(phi_runs, dtype=float)
    phi_ref  = np.asarray(phi_ref,  dtype=float)

    M       = phi_ref.shape[0]
    k       = min(k, M)
    ref_top = set(np.argsort(-np.abs(phi_ref))[:k])

    K       = phi_runs.shape[0]
    per_run = np.zeros(K)

    for i in range(K):
        est_top    = set(np.argsort(-np.abs(phi_runs[i]))[:k])
        per_run[i] = len(est_top & ref_top) / k

    return float(per_run.mean()), float(per_run.std()), per_run


def compute_interrun_variance(phi_runs):
    """
    Mean feature-wise variance across K runs.
    Captures magnitude instability independently of rank stability.

    Parameters
    ----------
    phi_runs : array-like, shape (K, M)

    Returns
    -------
    variance : float — mean(Var(φ̂_j)) over j=1..M
    """
    phi_runs = np.asarray(phi_runs, dtype=float)
    return float(phi_runs.var(axis=0).mean())


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng      = np.random.RandomState(42)
    M        = 14
    phi_ref  = rng.randn(M)
    phi_runs = phi_ref[np.newaxis, :] + rng.randn(30, M) * 0.05

    m, s, pr = compute_spearman_per_run(phi_runs, phi_ref)
    print(f"Spearman  : mean={m:.4f} std={s:.4f} min={pr.min():.4f} max={pr.max():.4f}")

    m, s, pr = compute_kendall_per_run(phi_runs, phi_ref)
    print(f"Kendall τ : mean={m:.4f} std={s:.4f}")

    m, s, pr = compute_topk_overlap_per_run(phi_runs, phi_ref, k=3)
    print(f"Top-3     : mean={m:.4f} std={s:.4f}")

    m, s, pr = compute_topk_overlap_per_run(phi_runs, phi_ref, k=5)
    print(f"Top-5     : mean={m:.4f} std={s:.4f}")

    v = compute_interrun_variance(phi_runs)
    print(f"Variance  : {v:.6f}")
    print("stability.py OK")