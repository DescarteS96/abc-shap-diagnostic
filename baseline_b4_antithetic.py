"""
BASELINE B4 — Antithetic Variates
==================================
Reference:
    Hammersley, J. M., & Morton, K. W. (1956).
    A new Monte Carlo technique: antithetic variates.
    Mathematical Proceedings of the Cambridge Philosophical Society,
    52(3), 449-475.

    General treatment:
    Ross, S. M. (2012). Simulation (5th ed.).
    Academic Press. Chapter 9: Variance Reduction Techniques.

    Robert, C. P., & Casella, G. (2004).
    Monte Carlo Statistical Methods (2nd ed.).
    Springer. Chapter 4.

Description
-----------
Antithetic Variates is a classical variance reduction technique for
Monte Carlo estimators. For every coalition z sampled, its complement
z̄ = 1 - z is also included in the regression.

Key property: κ(z) = κ(z̄) by symmetry of the Shapley kernel.
    κ(z̄) = (M-1) / [C(M, M-|z|) * (M-|z|) * |z|]
           = κ(z)  ✓

Since z and z̄ have the same kernel weight, they contribute equally
to the regression. Their estimates tend to be negatively correlated
(when f is non-linear), reducing the variance of the estimator.

Variance reduction:
    Var_AV(φ̂) = Var_uniform(φ̂) * (1 + ρ(z, z̄)) / 2
    where ρ(z, z̄) ≤ 0 for monotone functions f.
    Source: Ross (2012), Theorem 9.2.

This is a NON-ADAPTIVE method — the pairing z ↔ z̄ is the same
for all instances and models.

Usage
-----
    from baseline_b4_antithetic import KernelSHAPAntithetic
    explainer = KernelSHAPAntithetic(N_coalitions=1000)
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
# BASELINE B4 — Antithetic Variates
# ══════════════════════════════════════════════════════════════════════

class KernelSHAPAntithetic:
    """
    Kernel SHAP with antithetic variates variance reduction.

    For every randomly sampled coalition z, its bitwise complement
    z̄ = 1 - z is automatically included in the regression set.

    This doubles the effective coverage per random draw:
    - N_coalitions // 2 random draws produce N_coalitions samples
      (each draw yields z and z̄)
    - The two trivial coalitions z_empty and z_full are always included

    Key theoretical property (Shapley kernel symmetry):
        κ(z) = κ(z̄)  for all z with 0 < |z| < M
    This means antithetic pairs have equal regression weights,
    preserving the unbiasedness of the estimator.

    Source: Hammersley & Morton (1956); Ross (2012) Ch. 9.

    Parameters
    ----------
    N_coalitions : int
        Total number of coalitions (including antithetic pairs).
        Must be even. If odd, rounded down to nearest even number.
    """

    def __init__(self, N_coalitions=1000):
        # Ensure even number — pairs require N/2 draws
        self.N_coalitions = N_coalitions if N_coalitions % 2 == 0 \
                            else N_coalitions - 1

    def explain(self, f, x, D):
        """
        Compute SHAP values using antithetic variates.

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

        # Number of antithetic pairs to draw
        n_pairs = (self.N_coalitions - 2) // 2

        for _ in range(n_pairs):
            while True:
                z = np.random.randint(0, 2, size=M)
                s = z.sum()
                if 0 < s < M:
                    break

            z_bar = 1 - z   # antithetic complement: z̄ = 1 - z
            coalitions.append(z)
            coalitions.append(z_bar)

        phi = kernel_shap_regression(coalitions, f, x, D, M)
        return phi

    def verify_kernel_symmetry(self, M, n_checks=5):
        """
        Verify that κ(z) = κ(z̄) for random coalitions.
        Diagnostic utility — not used in explain().
        """
        print(f"  Kernel symmetry check κ(z) = κ(z̄) :")
        for _ in range(n_checks):
            z = np.random.randint(0, 2, M)
            while z.sum() == 0 or z.sum() == M:
                z = np.random.randint(0, 2, M)
            z_bar = 1 - z
            kz    = shapley_kernel_weight(z, M)
            kzbar = shapley_kernel_weight(z_bar, M)
            ok    = "✓" if abs(kz - kzbar) < 1e-10 else "✗"
            print(f"    |z|={z.sum()}, |z̄|={z_bar.sum()} → "
                  f"κ(z)={kz:.6f}, κ(z̄)={kzbar:.6f}  {ok}")