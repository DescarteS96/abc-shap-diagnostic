"""
config_corrected.py (THESIS RE-ANALYSIS SCOPE)
=========
Central configuration for the ABC-SHAP thesis RE-ANALYSIS pipeline.
Imported by ALL other modules — never duplicate constants elsewhere.

RE-ANALYSIS SCOPE (agreed with consultant, week of [date]):
    - Datasets : california (M=8), adult (M=14), credit (M=23).
      Diabetes (M=10) removed to keep the grid tractable within the
      remaining time; the low/mid/high dimensionality spread is
      preserved by the three kept datasets.
    - Models   : ridge/logistic, rf, mlp. XGBoost removed — near-
      identical behavior to RF across all configurations in the
      original thesis grid, and Ridge/Logistic is scientifically more
      informative here (Section 5.5.2 identifies it as the "least
      disadvantaged" model family, a result worth re-testing under the
      corrected formula rather than dropping it).
    - Budgets  : {1000, 5000, 10000} — reduced from the six-point grid,
      matching the article's scope for consistency.
    - Formula  : CORRECTED formula ONLY. results_path() now always
      resolves to RESULTS_DIR_CORRECTED — this re-analysis exists
      specifically to answer the consultant's request that the
      analyses themselves (not just the narrative) be redone under the
      corrected allocation.

Structure:
    1. Reproducibility
    2. Experimental protocol
    3. ABC hyperparameters (defaults + sensitivity grids)
    4. Datasets
    5. Models
    6. Metrics
    7. Paths
"""

import os

# ══════════════════════════════════════════════════════════════════════════════
# 1. REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════

GLOBAL_SEED = 42


# ══════════════════════════════════════════════════════════════════════════════
# 2. EXPERIMENTAL PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════

K_RUNS            = 30    # Independent runs per instance
N_TEST_INSTANCES  = 10    # Test instances per dataset
N_BACKGROUND      = 100   # Background dataset size
REF_MULTIPLIER    = 10    # phi_ref = B2 at N_budget × REF_MULTIPLIER

# CHANGED — thesis re-analysis scope: budget levels reduced from 6 to 3,
# matching the article's grid for consistency across both documents.
# Was: [1000, 2000, 3000, 5000, 7000, 10000]
ALL_BUDGETS = [1000, 5000, 10000]

# Human-readable labels for tables and figures
# CHANGED — trimmed to match ALL_BUDGETS above.
BUDGET_LABELS = {
    1000:  "N=1000",
    5000:  "N=5000",
    10000: "N=10000",
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. ABC HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Default values
ABC_DEFAULT = {
    "SN":       30,     # Colony size
    "T_min":    10,     # Minimum iterations (convergence floor)
    "T_max":    50,     # Maximum iterations
    "patience":  5,     # Early-stopping patience (no-improvement cycles)
    "epsilon":  1e-3,   # Convergence threshold on fitness improvement
    "alpha_mut": 0.2,   # Convex mutation rate (simplex displacement)
}

# ── Sensitivity grids — Chapter 6 (NOT modified for the re-analysis;
# Adult (M=14) remains in scope this time, so SENSITIVITY_M's anchoring
# comment below is still accurate and needs no change) ─────────────────
SENSITIVITY_SN    = [5, 10, 20, 30]
SENSITIVITY_TMAX  = [10, 20, 50]
SENSITIVITY_ALPHA = [0.1, 0.2, 0.4]

SENSITIVITY_GRID = [
    {"SN": sn, "T_max": tmax, "alpha_mut": alpha}
    for sn    in SENSITIVITY_SN
    for tmax  in SENSITIVITY_TMAX
    for alpha in SENSITIVITY_ALPHA
]


# CHANGED — was [0.1, 0.3, 0.5, 0.7, 0.9]. Matches the article's triplet,
# itself taken directly from Table 6.3's original three points.
SENSITIVITY_RHO = [0.1, 0.5, 0.9]        

# M=8  : anchored to California Housing (kept in re-analysis scope)
# M=14 : anchored to Adult Income (kept in re-analysis scope)
# M=20 : extrapolation beyond real datasets (characterizes dimensionality trend)
SENSITIVITY_M = [8, 14, 20]


# CHANGED — was [500, 1000, 2000, 3000, 5000, 7000, 10000].
# Matches ALL_BUDGETS for consistency between the main grid and the
# sensitivity analysis.
SENSITIVITY_BUDGETS = [1000, 5000, 10000]  
# ── Corrected Budget Formula — N_final >= alpha * N ───────────────────────
# UNCHANGED. This is now the ONLY formula in use anywhere in the
# re-analysis pipeline — see results_path() below.
def compute_abc_budget_corrected(N, M, SN=30, T_max=50, alpha=0.5):
    """Coût réel d'exploration = (1 + 2*T_eff) * SN_eff * N_inner :
    une phase d'initialisation (SN_eff évaluations) puis T_eff cycles de
    [employed + onlooker] (2*SN_eff évaluations chacun).
    Vise N_final >= max(4M, alpha*N) ; si la cible est hors d'atteinte,
    applique le coût minimal et le signale via target_met=False."""
    N_inner      = max(2 * (M + 1), 4 * M)
    N_final_min  = max(4 * M, int(alpha * N))
    N_explor_bgt = N - N_final_min

    # budget trop faible pour toute exploration : ABC dégénère en estimation seule
    if N_explor_bgt < 9 * N_inner and N < 9 * N_inner + 4 * M:
        return {'N_inner': N_inner, 'SN_eff': 0, 'T_max_eff': 0,
                'N_explor': 0, 'N_final': N,
                'target_met': False, 'viable': N >= 4 * M}

    E = N_explor_bgt // N_inner          # évaluations de politique finançables
    if E < 9:                            # plancher : SN_eff=3, T_eff=1 -> 9
        SN_eff, T_max_eff = 3, 1
        N_explor = 9 * N_inner
        N_final  = N - N_explor
        return {'N_inner': N_inner, 'SN_eff': SN_eff, 'T_max_eff': T_max_eff,
                'N_explor': N_explor, 'N_final': N_final,
                'target_met': False, 'viable': N_final >= 4 * M}

    SN_eff    = min(SN, E // 3)
    T_max_eff = min(T_max, (E // SN_eff - 1) // 2)
    N_explor  = (1 + 2 * T_max_eff) * SN_eff * N_inner
    N_final   = N - N_explor
    assert N_final >= N_final_min, (N, M, N_final, N_final_min)
    return {'N_inner': N_inner, 'SN_eff': SN_eff, 'T_max_eff': T_max_eff,
            'N_explor': N_explor, 'N_final': N_final,
            'target_met': True, 'viable': True}


# ── Allocation à ratio fixé — expérience de sensibilité A1/A2/A3 ──────────
def compute_abc_budget_ratio(N, M, explor_share, SN_max=30, T_max=50):
    """Alloue une part cible du budget total à l'exploration.
       Coût réel = (1 + 2*T_eff) * SN_eff * N_inner.
       Retourne l'allocation dont le coût est le plus proche de la cible
       sans la dépasser. La granularité étant N_inner, la part réalisée
       peut différer légèrement de la part visée."""
    N_inner = max(2 * (M + 1), 4 * M)
    target  = int(explor_share * N)
    E = target // N_inner
    best = None
    for T in range(1, T_max + 1):
        SN = min(SN_max, E // (1 + 2 * T))
        if SN < 3:
            break
        cost = (1 + 2 * T) * SN * N_inner
        if cost <= target and (best is None or cost > best[2]):
            best = (SN, T, cost)
    if best is None:
        SN, T, cost = 3, 1, 9 * N_inner
    else:
        SN, T, cost = best
    return {'N_inner': N_inner, 'SN_eff': SN, 'T_max_eff': T,
            'N_explor': cost, 'N_final': N - cost,
            'explor_share_target': explor_share,
            'explor_share_realized': cost / N}

# ══════════════════════════════════════════════════════════════════════════════
# 4. DATASETS
# ══════════════════════════════════════════════════════════════════════════════

# CHANGED — thesis re-analysis scope: Diabetes (M=10) removed.
# Was: ["california", "diabetes", "adult", "credit"]
REAL_DATASETS = ["california", "adult", "credit"]

# CHANGED — trimmed to match REAL_DATASETS above.
DATASET_META = {
    "california": {"M": 8,  "task": "regression",     "N_min": 1600},
    "adult":      {"M": 14, "task": "classification", "N_min": 2800},
    "credit":     {"M": 23, "task": "classification", "N_min": 4600},
}

# CHANGED — Diabetes removed, so only California remains a
# regression-task dataset where Ridge is trivial (Spearman=1.0
# regardless of sampling policy). Adult and Credit Default use
# LogisticRegression (classification) and are NOT trivial — this was
# already the case in the original thesis pipeline (confirmed:
# "Adult and Credit Default use Logistic Regression rather than Ridge,
# and are not flagged as trivial in the pipeline").
# Was: ["california", "diabetes"]
RIDGE_TRIVIAL_DATASETS = ["california"]


# ══════════════════════════════════════════════════════════════════════════════
# 5. MODELS
# ══════════════════════════════════════════════════════════════════════════════

# CHANGED — thesis re-analysis scope: XGBoost removed (near-identical
# behavior to Random Forest across all configurations in the original
# thesis grid — redundant for this re-analysis). Ridge/Logistic is
# KEPT (unlike the article's scope): Section 5.5.2 of the thesis
# identifies it as the least-disadvantaged model family, a result
# specifically worth re-testing under the corrected formula.
# Was: ["ridge", "rf", "xgb", "mlp"]
MODEL_FAMILIES = ["ridge", "rf", "mlp"]

# CHANGED — trimmed to match MODEL_FAMILIES above.
MODEL_LABELS = {
    "ridge": "Ridge / LogReg",
    "rf":    "Random Forest",
    "mlp":   "MLP (64→32)",
}

# Methods compared in all experiments — UNCHANGED.
METHODS = ["abc_shap", "b1_uniform", "b2_stratified", "b3_is", "b4_antithetic"]

METHOD_LABELS = {
    "abc_shap":       "ABC-SHAP",
    "b1_uniform":     "B1 (Uniform)",
    "b2_stratified":  "B2 (Stratified)",
    "b3_is":          "B3 (Importance Sampling)",
    "b4_antithetic":  "B4 (Antithetic)",
}

REFERENCE_METHOD = "b2_stratified"


# ══════════════════════════════════════════════════════════════════════════════
# 6. METRICS
# ══════════════════════════════════════════════════════════════════════════════

TOP_K_VALUES = [3, 5]
BOOTSTRAP_N  = 1000
BOOTSTRAP_ALPHA = 0.05

# Wilcoxon + Bonferroni
# CHANGED — comment updated to reflect the re-analysis grid.
# 4 baselines x 3 datasets x 3 budgets = 36 blocks, corrected within
# each (dataset, budget) block — N_COMPARISONS_PER_BLOCK itself is
# unchanged (still 4: ABC vs each of B1-B4 within a block).
N_COMPARISONS_PER_BLOCK = 4
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS_PER_BLOCK   # = 0.0125


# ══════════════════════════════════════════════════════════════════════════════
# 7. PATHS
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Kept for reference only — NOT used by results_path() below.
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results_step3")
RESULTS_DIR_CORRECTED  = os.path.join(PROJECT_ROOT, "results_step3_corrected")
FIGURES_DIR  = os.path.join(PROJECT_ROOT, "figures")
TABLES_DIR   = os.path.join(PROJECT_ROOT, "tables")
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")

# CHANGED — thesis re-analysis scope: now resolves EXCLUSIVELY to the
# corrected-formula results directory. This is the central change that
# answers the consultant's request: no script in this pipeline can
# silently read or produce original-formula results anymore.
def results_path(dataset: str, budget: int) -> str:
    """Returns the path to the CORRECTED-formula results CSV for a given
    (dataset, budget). Re-analysis scope only reads/writes corrected-
    formula results."""
    return os.path.join(RESULTS_DIR_CORRECTED, f"results_{dataset}_N{budget}.csv")

SENSITIVITY_DIR = os.path.join(PROJECT_ROOT, "results_sensitivity")

def sensitivity_path(experiment: str) -> str:
    return os.path.join(SENSITIVITY_DIR, f"results_sensitivity_{experiment}.csv")

for _d in [RESULTS_DIR, RESULTS_DIR_CORRECTED, FIGURES_DIR, TABLES_DIR, OUTPUTS_DIR, SENSITIVITY_DIR]:
    os.makedirs(_d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK — run directly to verify config
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("CONFIG SANITY CHECK — THESIS RE-ANALYSIS SCOPE")
    print("=" * 60)
    print(f"GLOBAL_SEED       : {GLOBAL_SEED}")
    print(f"K_RUNS            : {K_RUNS}")
    print(f"N_TEST_INSTANCES  : {N_TEST_INSTANCES}")
    print(f"N_BACKGROUND      : {N_BACKGROUND}")
    print(f"REF_MULTIPLIER    : {REF_MULTIPLIER}×")
    print(f"ALL_BUDGETS       : {ALL_BUDGETS}")
    print()
    print(f"REAL_DATASETS     : {REAL_DATASETS}")
    print(f"DATASET_META      :")
    for ds, meta in DATASET_META.items():
        print(f"  {ds:12s} → M={meta['M']:2d} | {meta['task']:14s} | N_min={meta['N_min']}")
    print()
    print(f"MODEL_FAMILIES    : {MODEL_FAMILIES}")
    print(f"RIDGE_TRIVIAL_DATASETS : {RIDGE_TRIVIAL_DATASETS}")
    print(f"METHODS           : {METHODS}")
    print(f"REFERENCE_METHOD  : {REFERENCE_METHOD}")
    print()
    print(f"BONFERRONI_ALPHA  : {BONFERRONI_ALPHA:.4f}")
    print()
    print(f"RESULTS_DIR_CORRECTED (used by results_path()) : {RESULTS_DIR_CORRECTED}")
    print("=" * 60)

    # ── Expected-totals check, re-analysis scope ───────────────────────────
    n_datasets = len(REAL_DATASETS)
    n_models   = len(MODEL_FAMILIES)
    n_budgets  = len(ALL_BUDGETS)
    n_methods  = len(METHODS)

    # NOTE: Ridge is trivial (and typically excluded from rank-stability
    # analysis) only on california. The raw vector count below still
    # includes Ridge everywhere it was run, since exp_runner.py computes
    # it regardless — filtering happens downstream in analysis, not at
    # collection time.
    total_vectors = n_datasets * n_models * n_budgets * N_TEST_INSTANCES * n_methods * K_RUNS
    total_wilcoxon_blocks = n_datasets * n_budgets

    print()
    print("EXPECTED TOTALS FOR THIS SCOPE (sanity reference — compare against")
    print("actual script output; a mismatch signals missing or leftover")
    print("article-scale or thesis-original-scale data):")
    print(f"  Datasets x Models x Budgets                          : "
          f"{n_datasets} x {n_models} x {n_budgets} = {n_datasets*n_models*n_budgets}")
    print(f"  Total attribution vectors (x {N_TEST_INSTANCES} instances x {n_methods} methods x {K_RUNS} runs) "
          f": {total_vectors}")
    print(f"  Total Wilcoxon blocks (dataset x budget)             : {total_wilcoxon_blocks}")
    print(f"  Total Wilcoxon comparisons (x {N_COMPARISONS_PER_BLOCK} baselines/block)  : "
          f"{total_wilcoxon_blocks * N_COMPARISONS_PER_BLOCK}")
    print("=" * 60)

    print()
    print("COMPUTE_ABC_BUDGET_CORRECTED (re-analysis scope: california, adult, credit) :")
    for ds, M in [("california", 8), ("adult", 14), ("credit", 23)]:
        for N in ALL_BUDGETS:
            p = compute_abc_budget_corrected(N, M)
            print(f"  {ds:12s} M={M:2d} N={N:6d} → "
                  f"N_inner={p['N_inner']:3d} | SN={p['SN_eff']:2d} | "
                  f"T_max={p['T_max_eff']:2d} | N_final={p['N_final']:5d} "
                  f"({p['N_final']/N*100:.0f}%) | viable={p['viable']}")