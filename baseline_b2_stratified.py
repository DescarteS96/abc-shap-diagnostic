"""
BASELINE B2 — Kernel SHAP Stratified Sampling by |z|
=====================================================
Reference : Covert, I., & Lee, S. I. (2021).
            Improving KernelSHAP: Practical Shapley Value Estimation
            using Linear Regression.
            Proceedings of AISTATS 2021.
            https://github.com/iancovert/shapley-regression

Description
-----------
Kernel SHAP with stratified sampling: the coalition budget is
distributed uniformly across all coalition sizes |z| ∈ {1,...,M-1}.

Instead of sampling uniformly over {0,1}^M (which over-represents
mid-size coalitions), this method ensures each size stratum receives
equal representation. This is a static strategy — the same
distribution is used for all instances x and models f.

This is the TRUE competitive baseline for ABC-SHAP.
If ABC-SHAP does not outperform B2, the added complexity of ABC
is not justified.

Usage
-----
    from baseline_b2_stratified import KernelSHAPStratified
    explainer = KernelSHAPStratified(N_coalitions=1000)
    phi = explainer.explain(f, x, D)
"""

import numpy as np
from scipy.special import comb
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════
# SHARED UTILITIES (local copies for script autonomy)
# ══════════════════════════════════════════════════════════════════════

def shapley_kernel_weight(z, M):
    """
    Shapley kernel weight.
    Source: Lundberg & Lee (2017), Equation 9.
    """
    s = int(z.sum())
    if s == 0 or s == M:
        return 1e8
    return (M - 1) / (comb(M, s, exact=True) * s * (M - s))


def estimate_conditional(f, x, D, S_indices):
    """
    Independent marginalization over background D.
    Source: Lundberg & Lee (2017), Section 3.
    """
    inputs = D.copy().astype(float)
    if S_indices:
        inputs[:, S_indices] = x[S_indices]
    return f(inputs).mean()


def kernel_shap_regression(coalitions, f, x, D, M):
    """
    WLS under local accuracy constraint (A1).

    Problem:
        min_φ  (y - Zφ)ᵀ W (y - Zφ)
        s.t.   1ᵀφ = rhs   (local accuracy constraint A1)

    Solution via KKT conditions (exact, closed-form):
        A       = ZᵀWZ
        b       = ZᵀW y_centered
        φ_unc   = A⁻¹ b
        λ       = (1ᵀ φ_unc - rhs) / (1ᵀ A⁻¹ 1)
        φ*      = φ_unc - λ A⁻¹ 1

    Source: Lundberg & Lee (2017), Theorem 1.
    """
    y = np.zeros(len(coalitions))
    for i, z in enumerate(coalitions):
        S = np.where(z == 1)[0].tolist()
        if len(S) == 0:
            y[i] = estimate_conditional(f, x, D, [])
        elif len(S) == M:
            y[i] = f(x.reshape(1, -1))[0]
        else:
            y[i] = estimate_conditional(f, x, D, S)

    weights = np.array([shapley_kernel_weight(z, M) for z in coalitions])
    Z       = np.array(coalitions, dtype=float)
    phi0    = estimate_conditional(f, x, D, [])
    fx      = f(x.reshape(1, -1))[0]
    y_c     = y - phi0
    rhs     = fx - phi0

    # Weighted normal equations
    ZtW = Z.T * weights        # (M × n_coal)
    A   = ZtW @ Z              # (M × M)
    b   = ZtW @ y_c            # (M,)

    # Numerical regularization
    A += 1e-8 * np.eye(M)

    # Unconstrained WLS solution
    phi_unc = np.linalg.solve(A, b)

    # Lagrange multiplier (exact KKT)
    ones    = np.ones(M)
    A_inv_1 = np.linalg.solve(A, ones)
    lam     = (ones @ phi_unc - rhs) / (ones @ A_inv_1)

    # Constrained solution
    phi = phi_unc - lam * A_inv_1

    return phi


# ══════════════════════════════════════════════════════════════════════
# BASELINE B2 — Stratified Sampling by Coalition Size
# ══════════════════════════════════════════════════════════════════════

class KernelSHAPStratified:
    """
    Kernel SHAP with stratified sampling by coalition size |z|.

    Sampling distribution:
        π_k = 1 / (M-1)  for k ∈ {1,...,M-1}  (uniform over strata)

    Each stratum (coalition size) receives an equal share of the budget.
    Within each stratum, active features are chosen uniformly at random.

    This is a static policy — identical for all instances and models.
    It is the direct implementation of the stratification strategy
    described in Covert & Lee (2021), Section 3.

    Parameters
    ----------
    N_coalitions : int
        Total number of coalitions to sample (= N_budget).
    """

    def __init__(self, N_coalitions=1000):
        self.N_coalitions = N_coalitions

    def explain(self, f, x, D):
        """
        Compute SHAP values for instance x using stratified sampling.

        Parameters
        ----------
        f : callable, black-box model, f(X) -> array shape (n,)
        x : np.ndarray, shape (M,)
        D : np.ndarray, shape (n, M), background dataset

        Returns
        -------
        phi : np.ndarray, shape (M,), estimated SHAP values
        """
        M = len(x)

        # Trivial coalitions — always included (anchor A1)
        z_empty = np.zeros(M, dtype=int)
        z_full  = np.ones(M, dtype=int)
        coalitions = [z_empty, z_full]

        n_random = self.N_coalitions - 2
        n_strata = M - 1                         # sizes 1 to M-1
        # Budget per stratum — distribute as evenly as possible
        base     = n_random // n_strata
        extra    = n_random  % n_strata          # first 'extra' strata get +1

        # Stratified allocation
        for k_idx in range(n_strata):
            size  = k_idx + 1                    # coalition size ∈ {1,...,M-1}
            count = base + (1 if k_idx < extra else 0)

            for _ in range(count):
                z = np.zeros(M, dtype=int)
                active = np.random.choice(M, size=size, replace=False)
                z[active] = 1
                coalitions.append(z)

        phi = kernel_shap_regression(coalitions, f, x, D, M)
        return phi