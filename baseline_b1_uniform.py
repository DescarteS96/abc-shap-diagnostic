"""
BASELINE B1 — Kernel SHAP Uniform Sampling
==========================================
Reference : Lundberg, S. M., & Lee, S. I. (2017).
            A unified approach to interpreting model predictions.
            Advances in Neural Information Processing Systems, 30.
            https://github.com/shap/shap

Description
-----------
Standard Kernel SHAP with uniform coalition sampling.
This is the minimum baseline — ABC-SHAP must outperform this.

Coalitions z ∈ {0,1}^M are sampled uniformly over the binary
space, then weighted by the Shapley kernel κ(z) in the regression.
The mismatch between uniform sampling and kernel weights is the
primary source of estimation variance that ABC-SHAP targets.

Usage
-----
    from baseline_b1_uniform import KernelSHAPUniform
    explainer = KernelSHAPUniform(N_coalitions=1000)
    phi = explainer.explain(f, x, D)
"""

import numpy as np
from scipy.special import comb
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# (same formulas as abc_shap_en.py — kept local for script autonomy)
# ══════════════════════════════════════════════════════════════════════

def shapley_kernel_weight(z, M):
    """
    Shapley kernel weight.
    w(z) = (M-1) / [C(M,|z|) * |z| * (M-|z|)]
    Source: Lundberg & Lee (2017), Equation 9.
    """
    s = int(z.sum())
    if s == 0 or s == M:
        return 1e8  # hard constraint — trivial coalitions
    return (M - 1) / (comb(M, s, exact=True) * s * (M - s))


def estimate_conditional(f, x, D, S_indices):
    """
    Independent marginalization over background D.
    y_z = (1/|D|) Σ_{d ∈ D} f(x_S, d_{S̄})
    Source: Lundberg & Lee (2017), Section 3.
    """
    inputs = D.copy().astype(float)
    if S_indices:
        inputs[:, S_indices] = x[S_indices]
    return f(inputs).mean()


def kernel_shap_regression(coalitions, f, x, D, M):
    """
    Weighted least squares under local accuracy constraint (A1).

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
# BASELINE B1 — Kernel SHAP Uniform
# ══════════════════════════════════════════════════════════════════════

class KernelSHAPUniform:
    """
    Kernel SHAP with uniform coalition sampling.

    Sampling distribution: P(z) = 1 / 2^M  (uniform over {0,1}^M)
    This is the original sampling strategy from Lundberg & Lee (2017).

    Parameters
    ----------
    N_coalitions : int
        Total number of coalitions to sample (= N_budget).
    """

    def __init__(self, N_coalitions=1000):
        self.N_coalitions = N_coalitions

    def explain(self, f, x, D):
        """
        Compute SHAP values for instance x using uniform sampling.

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

        # Uniform sampling over {0,1}^M
        # Each bit is independently Bernoulli(0.5)
        # Excludes trivial coalitions from random draw
        coalitions = [z_empty, z_full]
        n_random   = self.N_coalitions - 2

        for _ in range(n_random):
            while True:
                z = np.random.randint(0, 2, size=M)
                if 0 < z.sum() < M:   # exclude trivials
                    break
            coalitions.append(z)

        phi = kernel_shap_regression(coalitions, f, x, D, M)
        return phi