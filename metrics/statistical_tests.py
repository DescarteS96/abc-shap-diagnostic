"""
metrics/statistical_tests.py
=============================
Statistical tests for comparing methods across K runs.

Function:
    wilcoxon_bonferroni(runs_A, runs_B, n_comparisons, alternative)
        → (statistic, p_raw, p_corrected, significant)

Design:
    Wilcoxon signed-rank test — appropriate for K=30 paired runs
    (same instance, same random seed sequence, different methods).
    Non-parametric: no normality assumption on MSE or Spearman distributions.

    Bonferroni correction applied over n_comparisons simultaneous tests.
    Standard grouping: one correction block per (dataset, budget) pair,
    with n_comparisons = number of method pairs tested simultaneously.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import wilcoxon
from config import BONFERRONI_ALPHA, N_COMPARISONS_PER_BLOCK


def wilcoxon_bonferroni(runs_A, runs_B,
                        n_comparisons=N_COMPARISONS_PER_BLOCK,
                        alternative='greater'):
    """
    Wilcoxon signed-rank test with Bonferroni correction.

    Tests H0: median(A - B) = 0  vs  H1 (alternative).

    Parameters
    ----------
    runs_A        : array-like, shape (K,) — per-run metric for method A
    runs_B        : array-like, shape (K,) — per-run metric for method B
    n_comparisons : int   — number of simultaneous tests (Bonferroni denominator)
                            Default: N_COMPARISONS_PER_BLOCK from config
    alternative   : str   — 'greater' | 'less' | 'two-sided'
                            'greater': tests A > B (e.g., Spearman_ABC > Spearman_B1)
                            'less'   : tests A < B (e.g., MSE_ABC < MSE_B2)

    Returns
    -------
    statistic   : float — Wilcoxon W statistic
    p_raw       : float — uncorrected p-value
    p_corrected : float — Bonferroni-corrected p-value = min(p_raw × n_comparisons, 1.0)
    significant : bool  — True if p_corrected < BONFERRONI_ALPHA (from config)

    Edge cases handled:
        - All differences zero    → returns W=0, p=1.0, not significant
        - Fewer than 2 pairs      → returns nan, not significant
        - Arrays of different lengths → truncated to min(len(A), len(B))
        - scipy ValueError (e.g. zero_method edge cases) → returns
          W=0.0, p=1.0, not significant, WITH a printed warning showing
          the actual scipy message (see FIXED note below)
    """
    A = np.asarray(runs_A, dtype=float)
    B = np.asarray(runs_B, dtype=float)

    # Align lengths
    K = min(len(A), len(B))
    A, B = A[:K], B[:K]

    # Remove pairs where either value is nan
    mask = ~(np.isnan(A) | np.isnan(B))
    A, B = A[mask], B[mask]

    if len(A) < 2:
        nan = float('nan')
        return nan, nan, nan, False

    diff = A - B

    # All differences zero → test undefined, p=1
    if np.all(diff == 0):
        return 0.0, 1.0, 1.0, False

    try:
        stat, p_raw = wilcoxon(A, B, alternative=alternative)
    except ValueError as e:
        # FIXED (this pass): previously caught silently with a comment
        # claiming this only catches scipy's zero_method edge cases,
        # but the actual exception message was never inspected or
        # logged -- any OTHER cause of ValueError here (e.g. an
        # unrelated bug, unexpected input shape) would be silently
        # treated as a "no difference" result (p=1.0), indistinguishable
        # from a genuine null finding in downstream tables. Now prints
        # the real scipy message so this can be told apart from a
        # genuine zero-differences edge case.
        print(f"  WARNING wilcoxon_bonferroni: scipy raised ValueError "
              f"({e}) -- treating as no-difference (p=1.0). Verify this "
              f"is a genuine zero_method edge case, not an unrelated bug.")
        return 0.0, 1.0, 1.0, False

    p_corrected = min(float(p_raw) * n_comparisons, 1.0)
    significant = p_corrected < BONFERRONI_ALPHA

    return float(stat), float(p_raw), float(p_corrected), significant


def significance_label(p_corrected):
    """
    LaTeX-friendly significance star label for tables.

    Parameters
    ----------
    p_corrected : float — Bonferroni-corrected p-value

    Returns
    -------
    str : '***' | '**' | '*' | 'ns' | 'ref'
    """
    if p_corrected is None or np.isnan(p_corrected):
        return 'ref'
    if p_corrected < 0.001:
        return '***'
    if p_corrected < 0.01:
        return '**'
    if p_corrected < BONFERRONI_ALPHA:
        return '*'
    return 'ns'


def run_pairwise_tests(metrics_dict, metric_key='spearman_per_run',
                       reference='b2_stratified', alternative='greater'):
    """
    Runs Wilcoxon + Bonferroni for all methods vs a reference method.

    Parameters
    ----------
    metrics_dict : dict {method_name: {metric_key: list_of_K_values, ...}}
    metric_key   : str — key to extract per-run values
    reference    : str — reference method name
    alternative  : str

    Returns
    -------
    results : dict {method_name: {
        'statistic', 'p_raw', 'p_corrected', 'significant', 'label'
    }}
    """
    if reference not in metrics_dict:
        raise KeyError(f"Reference method '{reference}' not found in metrics_dict.")

    ref_runs     = metrics_dict[reference][metric_key]
    n_comparisons = len(metrics_dict) - 1   # all methods except reference

    results = {}
    for method, m in metrics_dict.items():
        if method == reference:
            results[method] = {
                'statistic':   None,
                'p_raw':       None,
                'p_corrected': None,
                'significant': None,
                'label':       'ref',
            }
            continue

        stat, p_raw, p_corr, sig = wilcoxon_bonferroni(
            runs_A=m[metric_key],
            runs_B=ref_runs,
            n_comparisons=n_comparisons,
            alternative=alternative,
        )
        results[method] = {
            'statistic':   stat,
            'p_raw':       p_raw,
            'p_corrected': p_corr,
            'significant': sig,
            'label':       significance_label(p_corr),
        }

    return results


# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.RandomState(42)

    # Case 1: A clearly better than B
    A = rng.normal(0.85, 0.04, 30)
    B = rng.normal(0.65, 0.06, 30)
    stat, p_raw, p_corr, sig = wilcoxon_bonferroni(A, B, alternative='greater')
    print(f"Case A>B : W={stat:.1f}  p_raw={p_raw:.4f}  p_corr={p_corr:.4f}  sig={sig}")
    print(f"  Label  : {significance_label(p_corr)}")

    # Case 2: No difference
    A2 = rng.normal(0.75, 0.05, 30)
    B2 = rng.normal(0.75, 0.05, 30)
    stat2, p_raw2, p_corr2, sig2 = wilcoxon_bonferroni(A2, B2)
    print(f"Case A~B : W={stat2:.1f}  p_raw={p_raw2:.4f}  p_corr={p_corr2:.4f}  sig={sig2}")
    print(f"  Label  : {significance_label(p_corr2)}")

    # Case 3: All zeros
    stat3, p_raw3, p_corr3, sig3 = wilcoxon_bonferroni(
        np.zeros(30), np.zeros(30)
    )
    print(f"Case all 0: W={stat3}  p_raw={p_raw3}  sig={sig3}")

    print("statistical_tests.py OK")