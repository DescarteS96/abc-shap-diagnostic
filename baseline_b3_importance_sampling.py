"""
BASELINE B3 — Importance Sampling with q(z) ∝ κ(z)
===================================================
Theoretical reference:
    Owen, A. B. (2013). Monte Carlo theory, methods and examples.
    Chapter 9: Importance Sampling.
    https://artowen.su.domains/mc/

    Derivation from Shapley kernel:
    Lundberg, S. M., & Lee, S. I. (2017).
    A unified approach to interpreting model predictions.
    NeurIPS 2017. — Equation 9 (Shapley kernel definition).

Description
-----------
Importance Sampling replaces the uniform sampling distribution with
a proposal distribution q(z) proportional to the Shapley kernel
weight κ(z). This concentrates sampling on high-weight coalitions
(small and large sizes), which are the most informative for the
Shapley regression.

Theoretical motivation:
    Var_IS(φ̂) ≤ Var_uniform(φ̂)
    when q(z) ∝ κ(z)  (classical IS variance reduction result)

This is a NON-ADAPTIVE method — the same distribution q(z) ∝ κ(z)
is used for all instances x and models f. It represents the
theoretical lower bound on variance achievable without adaptation.

If ABC-SHAP does not outperform B3, instance-specific adaptation
provides no benefit beyond the static optimum.

Usage
-----
    from baseline_b3_importance_sampling import KernelSHAPImportanceSampling
    explainer = KernelSHAPImportanceSampling(N_coalitions=1000)
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
# BASELINE B3 — Importance Sampling q(z) ∝ κ(z)
# ══════════════════════════════════════════════════════════════════════

class KernelSHAPImportanceSampling:
    """
    Kernel SHAP with importance sampling proposal q(z) ∝ κ(z).

    Sampling strategy:
    1. Compute unnormalized weights: w_k = κ_k * C(M,k)
       where κ_k = (M-1) / [C(M,k) * k * (M-k)]  is the kernel weight
       for size k, and C(M,k) is the number of coalitions of size k.
       This gives w_k = (M-1) / [k * (M-k)]  — proportional to the
       total kernel mass at size k.

    2. Normalize: q_k = w_k / Σ_{k=1}^{M-1} w_k

    3. Sample coalition size k ~ q, then sample features uniformly
       within that size.

    This is the static optimal non-adaptive proposal distribution.
    Source: Owen (2013), Chapter 9 — importance sampling theory.
    Derivation: Lundberg & Lee (2017), Equation 9.

    Parameters
    ----------
    N_coalitions : int
        Total number of coalitions to sample (= N_budget).
    """

    def __init__(self, N_coalitions=1000):
        self.N_coalitions = N_coalitions

    def _compute_proposal_distribution(self, M):
        """
        Compute the IS proposal distribution q over coalition sizes.

        q_k ∝ κ_k * C(M,k) = total kernel mass at size k
             = (M-1) / [k * (M-k)]

        Returns
        -------
        q : np.ndarray, shape (M-1,)
            q[i] = P(|z| = i+1) under the IS proposal
        """
        sizes  = np.arange(1, M)
        unnorm = (M - 1) / (sizes * (M - sizes))
        q      = unnorm / unnorm.sum()
        return q

    def explain(self, f, x, D):
        """
        Compute SHAP values using importance sampling q(z) ∝ κ(z).

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

        q = self._compute_proposal_distribution(M)

        z_empty    = np.zeros(M, dtype=int)
        z_full     = np.ones(M, dtype=int)
        coalitions = [z_empty, z_full]

        sizes    = np.arange(1, M)
        n_random = self.N_coalitions - 2

        for _ in range(n_random):
            k      = np.random.choice(sizes, p=q)
            z      = np.zeros(M, dtype=int)
            active = np.random.choice(M, size=k, replace=False)
            z[active] = 1
            coalitions.append(z)

        phi = kernel_shap_regression(coalitions, f, x, D, M)
        return phi

    def get_proposal_distribution(self, M):
        """
        Return the IS proposal distribution for inspection.
        Useful for verifying that q concentrates on small/large sizes.
        """
        return self._compute_proposal_distribution(M)