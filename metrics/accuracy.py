"""
metrics/accuracy.py
===================
Numerical accuracy metrics for Shapley value estimation.

Functions:
    compute_mse(phi_runs, phi_ref)
        → (mean, std) of MSE over K runs

    compute_mse_normalized(phi_runs, phi_ref, M)
        → (mean/M, std/M)

    compute_ratio_vs_b2(mse_method, mse_b2)
        → scalar ratio (nan-safe)

All functions accept phi_runs as list or np.ndarray of shape (K, M).
phi_ref is np.ndarray of shape (M,).
"""

import numpy as np


def compute_mse(phi_runs, phi_ref):
    """
    MSE between each individual run φ̂_k and the reference φ_ref.
    Correct implementation: one MSE per run, then aggregate.
    NOT computed on phi_mean (that was the original bug — C1/C5).

    Parameters
    ----------
    phi_runs : array-like, shape (K, M)
    phi_ref  : array-like, shape (M,)

    Returns
    -------
    mse_mean : float — E[||φ̂_k - φ_ref||²] over K runs
    mse_std  : float — std of per-run MSE
    mse_per_run : np.ndarray shape (K,)
    """
    phi_runs = np.asarray(phi_runs, dtype=float)   # (K, M)
    phi_ref  = np.asarray(phi_ref,  dtype=float)   # (M,)

    if phi_runs.ndim == 1:
        phi_runs = phi_runs[np.newaxis, :]          # handle single-run case

    diff         = phi_runs - phi_ref[np.newaxis, :]   # (K, M)
    mse_per_run  = np.sum(diff ** 2, axis=1)            # (K,) — L2 squared
    return float(mse_per_run.mean()), float(mse_per_run.std()), mse_per_run


def compute_mse_normalized(phi_runs, phi_ref, M):
    """
    MSE normalized by feature dimensionality M.
    Enables cross-dataset comparison (California M=8 vs Credit M=23).

    Returns
    -------
    mse_norm_mean : float — MSE/M
    mse_norm_std  : float — std(MSE)/M
    mse_per_run   : np.ndarray shape (K,) — raw per-run MSE (unnormalized)
    """
    if M <= 0:
        raise ValueError(f"M must be > 0, got {M}")

    mse_mean, mse_std, mse_per_run = compute_mse(phi_runs, phi_ref)
    return float(mse_mean / M), float(mse_std / M), mse_per_run


def compute_ratio_vs_b2(mse_method, mse_b2):
    """
    Ratio of method MSE to B2 (size-stratified, equal allocation) MSE.
    B2 is NOT the Neyman-optimal allocation (which would additionally
    require the within-stratum standard deviation of model evaluations,
    unknown a priori) -- it is the equal-allocation baseline used as the
    reference throughout this thesis for its MSE ratio computations.
    See Section 3.5.2/4.4.2 for the distinction.

    ratio > 1 : method is less accurate than B2
    ratio < 1 : method is more accurate than B2 (observed for B4 on Credit N=1000)
    ratio = 1 : reference (B2 itself)
    Parameters
    ----------
    mse_method : float
    mse_b2     : float

    Returns
    -------
    ratio : float (nan if mse_b2 == 0 or nan)
    """
    if mse_b2 is None or np.isnan(mse_b2) or mse_b2 == 0.0:
        return float('nan')
    return float(mse_method / mse_b2)


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng      = np.random.RandomState(42)
    phi_ref  = rng.randn(8)
    phi_runs = phi_ref[np.newaxis, :] + rng.randn(30, 8) * 0.1

    m, s, per_run = compute_mse(phi_runs, phi_ref)
    print(f"compute_mse          : mean={m:.6f} std={s:.6f} len={len(per_run)}")

    mn, sn, _ = compute_mse_normalized(phi_runs, phi_ref, M=8)
    print(f"compute_mse_normalized: mean/M={mn:.6f} std/M={sn:.6f}")

    ratio = compute_ratio_vs_b2(mse_method=0.002, mse_b2=0.0001)
    print(f"compute_ratio_vs_b2  : ratio={ratio:.2f}x")

    ratio_nan = compute_ratio_vs_b2(mse_method=0.001, mse_b2=0.0)
    print(f"compute_ratio_vs_b2 (b2=0): {ratio_nan}")
    print("accuracy.py OK")