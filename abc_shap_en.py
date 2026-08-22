"""
ABC-SHAP: Adaptive Coalition Sampling for Kernel SHAP
via Artificial Bee Colony Optimization

Implementation faithful to the master's thesis pseudo-code.
Author: Master's student in Statistics

Performance note (vs original version):
    `evaluate_policy` used to call `kernel_shap_regression` three times per
    policy evaluation — once each for the A-half, B-half, and full set of
    coalitions used in the split-half consistency test. Since the A-half and
    B-half are simple partitions of the SAME coalitions already drawn for the
    full set (no new coalitions are sampled for them), the conditional
    expectation y[i] for each coalition was being computed redundantly up to
    twice (once inside its half's regression call, once again inside the
    full-set regression call). This version computes y and the Shapley
    kernel weights ONCE per evaluate_policy call (on the full coalition set),
    then reuses array slices for the A-half and B-half regressions. The
    random draws (coalition sampling, split-half partition) are unchanged —
    only the redundant downstream computation is removed. Verified to
    produce bit-for-bit identical outputs to the original implementation on
    a controlled comparison (see thesis methodology note).
"""

import numpy as np
from scipy.special import comb
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════
# STANDALONE AUXILIARY FUNCTIONS
# (correspond to the auxiliary functions in the pseudo-code)
# ══════════════════════════════════════════════════════════════════════

def sample_dirichlet_policy(M):
    """
    Draws a policy π ~ Dirichlet(1,...,1) over Δ_{M-2}.
    M-1 components (sizes 1 to M-1), sum = 1.
    Dirichlet(1,...,1) = non-informative prior = uniform over the simplex.
    """
    alpha = np.ones(M - 1)
    pi = np.random.dirichlet(alpha)
    return pi  # shape: (M-1,), pi[k] = P(|z| = k+1)

def initial_population(SN, M, n_seeded=0):
    """
    Construit la population initiale de la colonie ABC.
    n_seeded=0 (defaut) reproduit exactement le comportement original :
    toutes les SN politiques tirees au hasard depuis Dirichlet(1,...,1).
    n_seeded>=1 ancre une partie de la population sur des points de
    depart informatifs (uniforme, proportionnel a la taille), le reste
    restant aleatoire -- teste si la convergence de la colonie est
    plus stable d'un run a l'autre avec un ancrage partagé.
    """
    policies = []
    if n_seeded >= 1:
        policies.append(np.full(M - 1, 1.0 / (M - 1)))          # uniforme (B1)
    if n_seeded >= 2:
        sizes = np.arange(1, M)
        prop = sizes.astype(float)
        policies.append(prop / prop.sum())                       # proportionnel a la taille
    while len(policies) < SN:
        policies.append(sample_dirichlet_policy(M))
    return policies[:SN]

def mutate_policy(pi, alpha=0.2):
    """
    Convex mutation on the simplex.
    π'_i = (1 - alpha) * π_i + alpha * δ_i
    with δ_i ~ Dirichlet(1,...,1)

    Guarantees π' ∈ Δ by construction (convex combination).
    alpha controls the amplitude of the perturbation.
    """
    M_minus_1 = len(pi)
    delta = np.random.dirichlet(np.ones(M_minus_1))
    pi_prime = (1 - alpha) * pi + alpha * delta
    # Safety renormalization (numerically π' is already on Δ)
    pi_prime = pi_prime / pi_prime.sum()
    return pi_prime


def sample_coalition(k, M):
    """
    Draws a coalition of size k from M features.
    Returns a binary vector z ∈ {0,1}^M with exactly k bits set to 1,
    chosen uniformly among C(M,k) possibilities.
    """
    z = np.zeros(M, dtype=int)
    active = np.random.choice(M, size=k, replace=False)
    z[active] = 1
    return z


def shapley_kernel_weight(z, M):
    """
    Shapley kernel weight for coalition z.
    w(z) = (M-1) / [C(M,|z|) * |z| * (M-|z|)]
    Trivial coalitions (|z|=0 or |z|=M) → infinite weight (hard constraint).
    """
    s = z.sum()
    if s == 0 or s == M:
        return 1e8  # +∞ in practice — hard constraint A1
    numerator = M - 1
    denominator = comb(M, s, exact=True) * s * (M - s)
    return numerator / denominator


def softmax(x):
    """
    Numerically stable softmax.
    Avoids winner-takes-all domination by a single bee.
    """
    x = np.array(x, dtype=float)
    x = x - x.max()  # numerical stabilization
    e = np.exp(x)
    return e / e.sum()


def weighted_mean(phi_list, weights):
    """
    Weighted mean of the colony's φ̂ estimates.
    φ̂_current = Σ_i p_i * φ̂_i with p_i = softmax(fitness)
    """
    w = softmax(np.array(weights))
    phi_array = np.array(phi_list)  # shape: (SN, M)
    return np.average(phi_array, axis=0, weights=w)


def estimate_conditional(f, x, D, S_indices):
    """
    Estimates the conditional expectation via independent marginalization.
    y_z = (1/|D|) Σ_{d ∈ D} f(x_S, d_{S̄})

    For each row d of D:
    - Active features (S) take the values of x
    - Inactive features (S̄) take the values of d
    """
    inputs = D.copy().astype(float)  # copy of D, shape: (n, M)
    if S_indices:
        inputs[:, S_indices] = x[S_indices]
    # Evaluate f on all rows and average
    predictions = f(inputs)
    return predictions.mean()


# ══════════════════════════════════════════════════════════════════════
# AUXILIARY FUNCTION — kernel_shap_regression
# (Block 6 of the pseudo-code) — UNCHANGED, kept for any external callers
# Solver: exact KKT analytical solution (replaces SLSQP)
# ══════════════════════════════════════════════════════════════════════

def kernel_shap_regression(coalitions, f, x, D, M):
    """
    Solves the Kernel SHAP regression on a set of coalitions.

    Problem:
        min_φ  (y - Zφ)ᵀ W (y - Zφ)
        s.t.   1ᵀφ = rhs   (local accuracy constraint A1)

    Solution via KKT conditions (exact, closed-form):
        A       = ZᵀWZ  (M × M)
        b       = ZᵀW y_centered  (M,)
        φ_unc   = A⁻¹ b  (unconstrained WLS solution)
        λ       = (1ᵀ φ_unc - rhs) / (1ᵀ A⁻¹ 1)  (Lagrange multiplier)
        φ*      = φ_unc - λ A⁻¹ 1  (constrained solution)

    Parameters
    ----------
    coalitions : list of np.ndarray, each z ∈ {0,1}^M
    f          : callable, black-box model
    x          : np.ndarray, shape (M,), instance to explain
    D          : np.ndarray, shape (n, M), background dataset
    M          : int, number of features

    Returns
    -------
    phi : np.ndarray, shape (M,), estimated SHAP values
    """
    y, weights = _compute_y_and_weights(coalitions, f, x, D, M)
    Z    = np.array(coalitions, dtype=float)
    phi0 = y[0] if (np.array(coalitions[0]) == 0).all() else estimate_conditional(f, x, D, [])
    fx   = f(x.reshape(1, -1))[0]
    return _solve_wls(Z, y, weights, phi0, fx, M)


def _compute_y_and_weights(coalitions, f, x, D, M):
    """
    Computes the conditional-expectation targets y and Shapley kernel
    weights for a list of coalitions. Factored out of
    `kernel_shap_regression` so that `evaluate_policy` can compute this
    ONCE on the full coalition set and reuse array slices for the
    split-half (A/B) regressions, instead of recomputing y for the same
    coalitions up to three times.
    """
    n_coal = len(coalitions)
    y = np.zeros(n_coal)
    for i, z in enumerate(coalitions):
        S = np.where(z == 1)[0].tolist()
        if len(S) == 0:
            y[i] = estimate_conditional(f, x, D, [])
        elif len(S) == M:
            y[i] = f(x.reshape(1, -1))[0]
        else:
            y[i] = estimate_conditional(f, x, D, S)

    weights = np.array([shapley_kernel_weight(z, M) for z in coalitions])
    return y, weights


def _solve_wls(Z, y, weights, phi0, fx, M):
    """
    Exact KKT closed-form solve of the weighted least squares Kernel SHAP
    regression, given precomputed y, weights and design matrix Z.
    Mathematically identical to the second half of the original
    `kernel_shap_regression` — factored out for reuse.
    """
    y_centered = y - phi0
    rhs        = fx - phi0

    ZtW = Z.T * weights          # (M × n_coal)
    A   = ZtW @ Z                # (M × M)
    b   = ZtW @ y_centered       # (M,)

    # Numerical regularization — prevents singular A (correlated features)
    A += 1e-8 * np.eye(M)

    try:
        phi_unc = np.linalg.solve(A, b)
        ones    = np.ones(M)
        A_inv_1 = np.linalg.solve(A, ones)
    except np.linalg.LinAlgError:
        # Fallback for near/exact-singular A (rare at very small N_inner
        # halves, e.g. M=8, N_budget=1000, especially with n_splits>1
        # which repeats the split-half partition and increases exposure
        # to degenerate splits). Pseudo-inverse gives the minimum-norm
        # least-squares solution instead of failing outright.
        phi_unc = np.linalg.pinv(A) @ b
        ones    = np.ones(M)
        A_inv_1 = np.linalg.pinv(A) @ ones

    lam = (ones @ phi_unc - rhs) / (ones @ A_inv_1)

    phi = phi_unc - lam * A_inv_1
    return phi


# ══════════════════════════════════════════════════════════════════════
# AUXILIARY FUNCTION — evaluate_policy
# (Block 5 of the pseudo-code)
# ══════════════════════════════════════════════════════════════════════

def evaluate_policy(pi, f, x, D, M, N_inner, n_splits=1):
    """
    Translates a policy π into fitness J(π) and estimate φ̂.

    1. Generates N_inner coalitions according to π (+ z_∅ and z_Ω mandatory)
    2. Random split-half — z_∅ and z_Ω in BOTH halves
    3. Computes φ̂_A and φ̂_B on each half
    4. J(π) = 1 / (1 + ‖φ̂_A - φ̂_B‖₂)
    5. φ̂ on the full set

    Parameters
    ----------
    pi      : np.ndarray, shape (M-1,), policy over Δ_{M-2}
    f       : callable
    x       : np.ndarray, shape (M,)
    D       : np.ndarray, shape (n, M)
    M       : int
    N_inner : int, number of coalitions to generate

    Returns
    -------
    J   : float, fitness ∈ (0, 1]
    phi : np.ndarray, shape (M,)
    """
    # ── Trivial coalitions (always included) ──────────────────────────
    z_empty = np.zeros(M, dtype=int)   # |z| = 0 → anchors φ₀
    z_full  = np.ones(M, dtype=int)    # |z| = M → anchors f(x)

    # ── Generate coalitions according to π ────────────────────────────
    sizes = np.arange(1, M)  # [1, 2, ..., M-1]

    sampled_coalitions = []
    for _ in range(N_inner - 2):  # -2 because z_∅ and z_Ω already count
        k = np.random.choice(sizes, p=pi)
        z = sample_coalition(k, M)
        sampled_coalitions.append(z)

    # ── Split-half (same random draws as the original implementation) ─
    n_sampled = len(sampled_coalitions)
    half      = n_sampled // 2

    coalitions_full = [z_empty, z_full] + sampled_coalitions

    # ── Compute y and weights ONCE on the full set ─────────────────────
    # coalitions_A and coalitions_B are partitions of THIS SAME set (no
    # new coalitions are drawn for them) — reusing the already-computed
    # y/weights avoids the 2-3x redundant computation of the original
    # version, with no change to the random draws or the math.
    y_full, w_full = _compute_y_and_weights(coalitions_full, f, x, D, M)
    Z_full = np.array(coalitions_full, dtype=float)
    phi0   = y_full[0]   # z_empty is always coalitions_full[0]
    fx     = y_full[1]   # z_full  is always coalitions_full[1]

    # ── Split-half fitness, averaged over n_splits independent partitions ──
    # (Section 3.4: J depends on the random split; averaged variants
    # recommended when budget allows — Pronk et al. 2022)
    J_values = []
    for _ in range(n_splits):
        indices = np.random.permutation(n_sampled)
        idx_A = np.concatenate(([0, 1], 2 + indices[:half]))
        idx_B = np.concatenate(([0, 1], 2 + indices[half:]))
        phi_A = _solve_wls(Z_full[idx_A], y_full[idx_A], w_full[idx_A], phi0, fx, M)
        phi_B = _solve_wls(Z_full[idx_B], y_full[idx_B], w_full[idx_B], phi0, fx, M)
        J_values.append(1.0 / (1.0 + np.linalg.norm(phi_A - phi_B)))
    J   = float(np.mean(J_values))
    phi = _solve_wls(Z_full, y_full, w_full, phi0, fx, M)

    return J, phi


# ══════════════════════════════════════════════════════════════════════
# MAIN CLASS — ABCShap
# ══════════════════════════════════════════════════════════════════════

class ABCShap:
    """
    ABC-SHAP: Adaptive Coalition Sampling for Kernel SHAP
    via Artificial Bee Colony Optimization.

    Parameters
    ----------
    SN          : int, colony size
    N_inner     : int, coalitions per policy evaluation
    limit       : int, scout abandonment threshold (default: SN*M)
    T_min       : int, minimum iterations before activating criterion C2
    T_max       : int, maximum iterations of the ABC loop
    epsilon_phi : float, convergence threshold on φ̂
    patience    : int, consecutive stable iterations for C2 stop
    alpha_mut   : float, mutation amplitude on the simplex (0,1)
    """
       
    def __init__(
        self,
        SN=30,
        N_inner=100,
        limit=None,
        T_min=10,
        T_max=50,
        epsilon_phi=1e-3,
        patience=5,
        alpha_mut=0.2,
        n_splits=1,
        n_seeded=0,
    ):
        self.SN          = SN
        self.N_inner     = N_inner
        self.limit       = limit
        self.T_min       = T_min
        self.T_max       = T_max
        self.epsilon_phi = epsilon_phi
        self.patience    = patience
        self.alpha_mut   = alpha_mut
        self.n_splits    = n_splits
        self.n_seeded    = n_seeded
    
        self.log_ = {
            'J_best':      [],
            'J_mean':      [],
            'delta_phi':   [],
            'eval_count':  [],
            'pi_best':     [],
            'phi_current': [],
        }

    def explain(self, f, x, D, N_budget):
        """
        Computes SHAP values for instance x.

        Parameters
        ----------
        f        : callable, black-box model. Signature: f(X) → array
        x        : np.ndarray, shape (M,), instance to explain
        D        : np.ndarray, shape (n, M), background dataset
        N_budget : int, total budget of coalition evaluations

        Returns
        -------
        phi_final : np.ndarray, shape (M,), estimated SHAP values
        """
        M     = len(x)
        limit = self.limit if self.limit is not None else self.SN * M

        # ── PHASE 1: INITIALIZATION ───────────────────────────────────
        eval_count   = 0
        trial        = np.zeros(self.SN, dtype=int)
        stable_count = 0
        phi_prev     = np.zeros(M)

        for key in self.log_:
            self.log_[key] = []

        policies = initial_population(self.SN, M, n_seeded=getattr(self, 'n_seeded', 0))
        fitness  = np.zeros(self.SN)
        phis     = [np.zeros(M) for _ in range(self.SN)]
        
        for i in range(self.SN):
            if eval_count >= N_budget:
                break
            fitness[i], phis[i] = evaluate_policy(
                policies[i], f, x, D, M, self.N_inner, n_splits=self.n_splits
            )
            eval_count += self.N_inner
        
        # Best bee estimate
        phi_current = phis[np.argmax(fitness)].copy()

        # ── PHASE 2: MAIN ABC LOOP ────────────────────────────────────
        for iter_num in range(1, self.T_max + 1):

            if eval_count >= N_budget:
                break

            # ── EMPLOYED BEES ─────────────────────────────────────────
            for i in range(self.SN):
                if eval_count >= N_budget:
                    break
                # Pre-check: avoid exceeding N_budget
                if eval_count + self.N_inner > N_budget:
                    break
       
                pi_prime = mutate_policy(policies[i], alpha=self.alpha_mut)
                fit_prime, phi_prime = evaluate_policy(
                    pi_prime, f, x, D, M, self.N_inner, n_splits=self.n_splits
                )
                eval_count += self.N_inner

                if fit_prime > fitness[i]:    
                    policies[i] = pi_prime
                    fitness[i]  = fit_prime
                    phis[i]     = phi_prime
                    trial[i]    = 0
                else:
                    trial[i] += 1

            # ── ONLOOKER BEES ─────────────────────────────────────────
            prob = softmax(fitness)

            for j in range(self.SN):
                if eval_count >= N_budget:
                    break
                # Pre-check: avoid exceeding N_budget
                if eval_count + self.N_inner > N_budget:
                    break

                i = np.random.choice(self.SN, p=prob)

                pi_prime = mutate_policy(policies[i], alpha=self.alpha_mut)
                fit_prime, phi_prime = evaluate_policy(
                    pi_prime, f, x, D, M, self.N_inner, n_splits=self.n_splits
                )
                eval_count += self.N_inner

                if fit_prime > fitness[i]:                                     
                    policies[i] = pi_prime
                    fitness[i]  = fit_prime
                    phis[i]     = phi_prime
                    trial[i]    = 0
                else:
                    trial[i] += 1

            # ── SCOUT BEES ────────────────────────────────────────────
            for i in range(self.SN):
                if trial[i] > limit:
                    policies[i] = sample_dirichlet_policy(M)
                    if eval_count < N_budget:
                        fitness[i], phis[i] = evaluate_policy(
                            policies[i], f, x, D, M, self.N_inner, n_splits=self.n_splits
                        )
                        eval_count += self.N_inner
                    trial[i] = 0

            # ── UPDATE φ̂_current — best bee ───────────────────────────
            phi_current = phis[np.argmax(fitness)].copy()

            # ── LOGGING ───────────────────────────────────────────────
            best_idx  = np.argmax(fitness)
            norm_prev = np.linalg.norm(phi_prev) + 1e-8
            delta_phi = np.linalg.norm(phi_current - phi_prev) / norm_prev

            self.log_['J_best'].append(float(fitness[best_idx]))
            self.log_['J_mean'].append(float(fitness.mean()))
            self.log_['delta_phi'].append(float(delta_phi))
            self.log_['eval_count'].append(int(eval_count))
            self.log_['pi_best'].append(policies[best_idx].copy())
            self.log_['phi_current'].append(phi_current.copy())

            # ── STOPPING CRITERION C2 ─────────────────────────────────
            if iter_num >= self.T_min:
                if delta_phi < self.epsilon_phi:
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count >= self.patience:
                    break

            phi_prev = phi_current.copy()

        # ── PHASE 3: FINAL ESTIMATION ─────────────────────────────────
        best_idx = int(np.argmax(fitness))
        pi_star  = policies[best_idx]

        N_final = N_budget - eval_count
        if N_final > self.N_inner:
            # Use full residual budget for final estimation
            _, phi_final = evaluate_policy(
                pi_star, f, x, D, M, N_final
            )
        else:
            phi_final = phi_current

        self.pi_star_    = pi_star
        self.fitness_    = fitness
        self.n_iters_    = iter_num if 'iter_num' in dir() else 0
        self.eval_count_ = eval_count

        return phi_final