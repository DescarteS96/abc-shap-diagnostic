"""
exp_runner.py (THESIS RE-ANALYSIS SCOPE + ALLOCATION EXPERIMENT)
=============
ABC-SHAP vs 4 baselines — re-analysis under the corrected budget formula,
plus the exploration/estimation allocation experiment (A0/A1/A2/A3).

Required files in the same directory (UNMODIFIED):
    abc_shap_en.py
    baseline_b1_uniform.py
    baseline_b2_stratified.py
    baseline_b3_importance_sampling.py
    baseline_b4_antithetic.py

Usage:
    # Grille principale (formule corrigee), inchangee
    python exp_runner.py --dataset california --budget 1000
    python exp_runner.py --dataset adult      --budget 5000

    # Experience d'allocation : A1 (25%) et A3 (75%), ABC seul
    python exp_runner.py --dataset california --budget 5000 \
        --explor-share 0.25 0.75 --abc-only

Outputs:
    results_step3_corrected/results_{dataset}_N{budget}.csv
        (grille principale, 5 methodes)
    results_step3_corrected/results_allocation_{dataset}_N{budget}.csv
        (experience d'allocation, colonnes 'allocation' et
         'explor_share_realized' en plus)
    Ecriture incrementale, reprise automatique apres interruption.
"""

import argparse
import numpy as np
import pandas as pd
import time
import warnings
import os
import sys

from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ── Method imports — original files UNMODIFIED ──────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from abc_shap_en import ABCShap
from baseline_b1_uniform import KernelSHAPUniform
from baseline_b2_stratified import KernelSHAPStratified
from baseline_b3_importance_sampling import KernelSHAPImportanceSampling
from baseline_b4_antithetic import KernelSHAPAntithetic
from config import compute_abc_budget_corrected, compute_abc_budget_ratio


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION — RE-ANALYSIS SCOPE
# ══════════════════════════════════════════════════════════════════════════════

ALL_BUDGETS      = [1000, 5000, 10000]
K_RUNS           = 30
N_TEST_INSTANCES = 10
N_BACKGROUND     = 100
REF_MULTIPLIER   = 10     # phi_ref = B2 at N_budget × REF_MULTIPLIER
BOOTSTRAP_N      = 1000

# ABC hyperparameters — fixed for the re-analysis
ABC_SN       = 30
ABC_T_MAX    = 50
ABC_T_MIN    = 10
ABC_PATIENCE = 5
ABC_EPSILON  = 1e-3
ABC_ALPHA    = 0.2

GLOBAL_SEED = 42

# Etiquettes d'allocation pour l'experience A1/A2/A3
SHARE_LABEL = {0.25: 'A1', 0.50: 'A2', 0.75: 'A3'}


def _alloc_label(share):
    """'A0' pour la formule corrigee, sinon A1/A2/A3."""
    if share is None:
        return 'A0'
    return SHARE_LABEL.get(round(float(share), 4), f"A_{share}")


# ══════════════════════════════════════════════════════════════════════════════
# DATASET LOADING — california, adult, credit only
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(name: str):
    """Loads (X, y, task). task in {'regression', 'classification'}."""
    if name == 'california':
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing()
        print(f"  California Housing — shape {d.data.shape}")
        return d.data.astype(float), d.target.astype(float), 'regression'

    elif name == 'adult':
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import LabelEncoder
        print("  Adult Income — loading OpenML...", end=" ", flush=True)
        data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
        df   = data.frame.dropna()
        y_raw = df['class'].values
        X_df  = df.drop('class', axis=1)
        le    = LabelEncoder()
        for col in X_df.select_dtypes(include='category').columns:
            X_df[col] = le.fit_transform(X_df[col].astype(str))
        X = X_df.values.astype(float)
        y = (y_raw == '>50K').astype(int)
        print(f"shape {X.shape}")
        return X, y, 'classification'

    elif name == 'credit':
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import LabelEncoder
        print("  Credit Default — loading OpenML...", end=" ", flush=True)
        _csv_path = os.path.join(SCRIPT_DIR, 'credit_default.csv')
        if os.path.exists(_csv_path):
            data_frame = pd.read_csv(_csv_path)
        else:
            data = fetch_openml('default-of-credit-card-clients', version=1,
                                as_frame=True, parser='auto')
            data_frame = data.frame
        df = data_frame.dropna()
        target_col = None
        for candidate in ['Y', 'default payment next month',
                          'default.payment.next.month']:
            if candidate in df.columns:
                target_col = candidate
                break
        if target_col is None:
            target_col = df.columns[-1]
        y    = df[target_col].values.astype(int)
        X_df = df.drop(columns=[target_col])
        le   = LabelEncoder()
        for col in X_df.select_dtypes(include=['object', 'category']).columns:
            X_df[col] = le.fit_transform(X_df[col].astype(str))
        X = X_df.values.astype(float)
        print(f"shape {X.shape}")
        return X, y, 'classification'

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Re-analysis scope choices: california, adult, credit"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING THE 3 MODELS — ridge, rf, mlp
# ══════════════════════════════════════════════════════════════════════════════

def train_models(X_train: np.ndarray, y_train: np.ndarray, task: str) -> dict:
    models = {}

    try:
        if task == 'regression':
            m = Ridge(alpha=1.0)
            m.fit(X_train, y_train)
            models['ridge'] = lambda X, _m=m: _m.predict(X)
        else:
            m = LogisticRegression(C=1.0, max_iter=1000,
                                   random_state=GLOBAL_SEED, n_jobs=1)
            m.fit(X_train, y_train)
            models['ridge'] = lambda X, _m=m: _m.predict_proba(X)[:, 1]
        print("    ridge : OK")
    except Exception as e:
        print(f"    ridge : SKIP ({e})")

    try:
        if task == 'regression':
            m = RandomForestRegressor(n_estimators=100,
                                      random_state=GLOBAL_SEED, n_jobs=1)
        else:
            m = RandomForestClassifier(n_estimators=100,
                                       random_state=GLOBAL_SEED, n_jobs=1)
        m.fit(X_train, y_train)
        if task == 'regression':
            models['rf'] = lambda X, _m=m: _m.predict(X)
        else:
            models['rf'] = lambda X, _m=m: _m.predict_proba(X)[:, 1]
        print("    rf    : OK")
    except Exception as e:
        print(f"    rf    : SKIP ({e})")

    try:
        mlp_kw = dict(hidden_layer_sizes=(64, 32), max_iter=500,
                      random_state=GLOBAL_SEED, early_stopping=True,
                      validation_fraction=0.1, n_iter_no_change=10)
        if task == 'regression':
            m = MLPRegressor(**mlp_kw)
        else:
            m = MLPClassifier(**mlp_kw)
        m.fit(X_train, y_train)
        if task == 'regression':
            models['mlp'] = lambda X, _m=m: _m.predict(X)
        else:
            models['mlp'] = lambda X, _m=m: _m.predict_proba(X)[:, 1]
        print("    mlp   : OK")
    except Exception as e:
        print(f"    mlp   : SKIP ({e})")

    return models


# ══════════════════════════════════════════════════════════════════════════════
# METRICS — UTILITY FUNCTIONS (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(values: np.ndarray, n_boot: int = BOOTSTRAP_N,
                 seed: int = GLOBAL_SEED):
    arr  = np.asarray(values, dtype=float)
    rng  = np.random.RandomState(seed)
    idx  = rng.randint(0, len(arr), size=(n_boot, len(arr)))
    boot = arr[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def topk_overlap(phi: np.ndarray, phi_ref: np.ndarray, k: int) -> float:
    M  = len(phi_ref)
    k  = min(k, M)
    tr = set(np.argsort(-np.abs(phi))[:k])
    rr = set(np.argsort(-np.abs(phi_ref))[:k])
    return len(tr & rr) / k


def compute_metrics(phis: list, times: list, n_coal_finals: list,
                    phi_ref: np.ndarray, N_budget: int, M: int) -> dict:
    K        = len(phis)
    phis_arr = np.array(phis)

    mse_per_run  = np.array([
        float(np.linalg.norm(phi - phi_ref) ** 2) for phi in phis
    ])
    mse_mean     = float(mse_per_run.mean())
    mse_std      = float(mse_per_run.std())
    mse_norm     = mse_mean / M
    mse_norm_std = mse_std  / M
    mse_ci_lo, mse_ci_hi = bootstrap_ci(mse_per_run)

    spearman_per_run = np.zeros(K)
    for k, phi in enumerate(phis):
        rho, _ = spearmanr(np.abs(phi), np.abs(phi_ref))
        spearman_per_run[k] = float(rho) if not np.isnan(rho) else 0.0
    spearman_mean = float(spearman_per_run.mean())
    spearman_std  = float(spearman_per_run.std())
    sp_ci_lo, sp_ci_hi = bootstrap_ci(spearman_per_run)

    kendall_per_run = np.zeros(K)
    for k, phi in enumerate(phis):
        tau, _ = kendalltau(np.abs(phi), np.abs(phi_ref))
        kendall_per_run[k] = float(tau) if not np.isnan(tau) else 0.0
    kendall_mean = float(kendall_per_run.mean())
    kendall_std  = float(kendall_per_run.std())

    top3_per_run = np.array([topk_overlap(phi, phi_ref, 3) for phi in phis])
    top5_per_run = np.array([topk_overlap(phi, phi_ref, 5) for phi in phis])

    variance = float(phis_arr.var(axis=0).mean())

    return {
        'mse':               mse_mean,
        'mse_std':           mse_std,
        'mse_norm':          mse_norm,
        'mse_norm_std':      mse_norm_std,
        'mse_ci_lower':      mse_ci_lo,
        'mse_ci_upper':      mse_ci_hi,
        'spearman_mean':     spearman_mean,
        'spearman_std':      spearman_std,
        'spearman_ci_lower': sp_ci_lo,
        'spearman_ci_upper': sp_ci_hi,
        'kendall_mean':      kendall_mean,
        'kendall_std':       kendall_std,
        'top3_mean':         float(top3_per_run.mean()),
        'top3_std':          float(top3_per_run.std()),
        'top5_mean':         float(top5_per_run.mean()),
        'top5_std':          float(top5_per_run.std()),
        'variance':          variance,
        'time_mean':         float(np.mean(times)),
        'time_std':          float(np.std(times)),
        'n_coal_final':      float(np.mean(n_coal_finals)),
        'mse_per_run':       mse_per_run.tolist(),
        'spearman_per_run':  spearman_per_run.tolist(),
        'kendall_per_run':   kendall_per_run.tolist(),
        'top3_per_run':      top3_per_run.tolist(),
        'top5_per_run':      top5_per_run.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HIGH-PRECISION REFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def compute_phi_ref(f, x: np.ndarray, D: np.ndarray, N_budget: int) -> np.ndarray:
    return KernelSHAPStratified(
        N_coalitions=N_budget * REF_MULTIPLIER
    ).explain(f, x, D)


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT PER INSTANCE
# ══════════════════════════════════════════════════════════════════════════════

def get_alloc_params(N_budget, M, explor_share):
    """Retourne le dictionnaire d'allocation ABC pour cette cellule."""
    if explor_share is None:
        return compute_abc_budget_corrected(N_budget, M,
                   SN=ABC_SN, T_max=ABC_T_MAX)
    return compute_abc_budget_ratio(N_budget, M, explor_share,
               SN_max=ABC_SN, T_max=ABC_T_MAX)


def run_single_instance(f, x: np.ndarray, D: np.ndarray,
                        N_budget: int, phi_ref: np.ndarray,
                        explor_share=None, abc_only: bool = False) -> dict:
    """
    K=30 runs. Si explor_share est None, utilise la formule corrigee.
    Sinon, alloue la part demandee a l'exploration (experience A1/A2/A3).
    Si abc_only, ne calcule qu'ABC-SHAP (les baselines sont reprises
    du run principal au moment de l'analyse).
    """
    M = len(x)

    _p        = get_alloc_params(N_budget, M, explor_share)
    N_inner   = _p['N_inner']
    _sn_eff   = _p['SN_eff']
    _tmax_eff = _p['T_max_eff']

    method_names = ['abc_shap'] if abc_only else [
        'abc_shap', 'b1_uniform', 'b2_stratified', 'b3_is', 'b4_antithetic']
    raw = {name: {'phis': [], 'times': [], 'n_coal': []}
           for name in method_names}

    baselines = {} if abc_only else {
        'b1_uniform':    KernelSHAPUniform(N_coalitions=N_budget),
        'b2_stratified': KernelSHAPStratified(N_coalitions=N_budget),
        'b3_is':         KernelSHAPImportanceSampling(N_coalitions=N_budget),
        'b4_antithetic': KernelSHAPAntithetic(N_coalitions=N_budget),
    }

    for _ in range(K_RUNS):
        abc = ABCShap(
            SN=_sn_eff, N_inner=N_inner,
            T_min=min(ABC_T_MIN, _tmax_eff), T_max=_tmax_eff,
            epsilon_phi=ABC_EPSILON, patience=ABC_PATIENCE,
            alpha_mut=ABC_ALPHA,
        )

        t0  = time.perf_counter()
        phi = abc.explain(f, x, D, N_budget)
        _elapsed = time.perf_counter() - t0

        _used = getattr(abc, 'eval_count_', None)
        if _used is None:
            raise RuntimeError(
                "ABCShap n'expose pas eval_count_ : le budget d'exploration "
                "consomme ne peut pas etre verifie."
            )
        _used = int(_used)
        assert _used == _p['N_explor'], (
            f"Budget d'exploration incoherent : consomme={_used}, "
            f"planifie={_p['N_explor']} (N={N_budget}, M={M}, "
            f"SN_eff={_p['SN_eff']}, T_eff={_p['T_max_eff']}, "
            f"N_inner={_p['N_inner']})"
        )

        raw['abc_shap']['phis'].append(phi)
        raw['abc_shap']['times'].append(_elapsed)
        raw['abc_shap']['n_coal'].append(_used)

        for name, explainer in baselines.items():
            t0  = time.perf_counter()
            phi = explainer.explain(f, x, D)
            raw[name]['phis'].append(phi)
            raw[name]['times'].append(time.perf_counter() - t0)
            raw[name]['n_coal'].append(N_budget)

    metrics = {
        name: compute_metrics(
            phis=res['phis'], times=res['times'],
            n_coal_finals=res['n_coal'],
            phi_ref=phi_ref, N_budget=N_budget, M=M,
        )
        for name, res in raw.items()
    }

    if abc_only:
        for m in metrics.values():
            m['mse_ratio_b2'] = float('nan')   # recalcule a l'analyse
    else:
        mse_b2 = metrics['b2_stratified']['mse']
        for m in metrics.values():
            m['mse_ratio_b2'] = (
                m['mse'] / mse_b2
                if (mse_b2 > 0 and not np.isnan(mse_b2))
                else float('nan')
            )

    _share_real = _p.get('explor_share_realized',
                         _p['N_explor'] / float(N_budget))
    for name, m in metrics.items():
        if name == 'abc_shap':
            m['n_exploration'] = float(_p['N_explor'])
            m['n_estimation']  = float(N_budget - _p['N_explor'])
            m['explor_share_realized'] = float(_share_real)
        else:
            m['n_exploration'] = 0.0
            m['n_estimation']  = float(N_budget)
            m['explor_share_realized'] = 0.0

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_dataset(ds_name: str, budgets: list,
                explor_shares=None, abc_only: bool = False):
    """
    Boucle complete : 3 modeles x budgets x instances x allocations.
    Reprise automatique, ecriture incrementale apres chaque cellule.
    """
    print(f"\n{'='*65}")
    print(f"DATASET : {ds_name.upper()} | Budgets : {budgets}")
    if explor_shares:
        print(f"ALLOCATION EXPERIMENT | shares : {explor_shares} | "
              f"abc_only={abc_only}")
    print(f"{'='*65}")

    np.random.seed(GLOBAL_SEED)

    X, y, task = load_dataset(ds_name)
    M = X.shape[1]
    print(f"  M={M} | task={task}")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    stratify = y if task == 'classification' else None
    X_train, X_test, y_train, _ = train_test_split(
        X_scaled, y, test_size=0.2, random_state=GLOBAL_SEED, stratify=stratify
    )

    rng        = np.random.RandomState(GLOBAL_SEED)
    n_inst     = min(N_TEST_INSTANCES, len(X_test))
    test_idx   = rng.choice(len(X_test), size=n_inst, replace=False)
    X_test_sel = X_test[test_idx]

    bg_idx = rng.choice(len(X_train),
                        size=min(N_BACKGROUND, len(X_train)), replace=False)
    D      = X_train[bg_idx]

    print("\n  Training the 3 re-analysis-scope models (ridge, rf, mlp)...")
    models = train_models(X_train, y_train, task)
    if not models:
        raise RuntimeError("No model available.")
    print(f"  Active models : {list(models.keys())}")

    out_dir = os.path.join(SCRIPT_DIR, "results_step3_corrected")
    os.makedirs(out_dir, exist_ok=True)

    # Liste des allocations a produire (None = formule corrigee, 'A0')
    shares = list(explor_shares) if explor_shares else [None]
    labels = [_alloc_label(s) for s in shares]

    for model_name, f in models.items():
        print(f"\n  ── Model : {model_name} ──")

        for N_budget in budgets:
            suffix   = "_allocation" if explor_shares else ""
            out_path = os.path.join(
                out_dir, f"results{suffix}_{ds_name}_N{N_budget}.csv")

            # ── Bloc d'information : une ligne par allocation ──────────
            for s, lab in zip(shares, labels):
                _pp = get_alloc_params(N_budget, M, s)
                print(f"\n    {lab} | N={N_budget} | N_inner={_pp['N_inner']} | "
                      f"SN_eff={_pp['SN_eff']} | T_eff={_pp['T_max_eff']} | "
                      f"explor={_pp['N_explor']} | "
                      f"final={N_budget - _pp['N_explor']} "
                      f"({100.0 * _pp['N_explor'] / N_budget:.1f}%)")

            # ── Reprise automatique ───────────────────────────────────
            done_keys = set()
            if os.path.exists(out_path):
                try:
                    _cols = ['model', 'instance_idx']
                    if explor_shares:
                        _cols.append('allocation')
                    df_ex = pd.read_csv(out_path, usecols=_cols,
                                        on_bad_lines='skip')
                    for _, row in df_ex.iterrows():
                        if row['model'] != row['model']:   # NaN guard
                            continue
                        try:
                            k = (row['model'], int(row['instance_idx']))
                            if explor_shares:
                                k = k + (str(row['allocation']),)
                            done_keys.add(k)
                        except (ValueError, TypeError):
                            continue
                    if done_keys:
                        print(f"\n    Resume: {len(done_keys)} cellules deja "
                              f"presentes dans le fichier")
                except Exception as e:
                    print(f"    WARNING resume: {e} — starting from scratch")

            # ── Boucle sur les instances ──────────────────────────────
            for inst_idx, x in enumerate(X_test_sel):

                # cles attendues pour cette (model, instance)
                keys = [
                    (model_name, inst_idx) if not explor_shares
                    else (model_name, inst_idx, lab)
                    for lab in labels
                ]
                if all(k in done_keys for k in keys):
                    print(f"      Inst {inst_idx+1:2d}/{n_inst} — skipped")
                    continue

                # phi_ref calcule UNE SEULE FOIS, reutilise par toutes
                # les allocations de cette instance
                try:
                    phi_ref = compute_phi_ref(f, x, D, N_budget)
                except Exception as e:
                    print(f"      Inst {inst_idx+1:2d}/{n_inst} "
                          f"ERROR phi_ref: {e} — skipped")
                    continue

                for share, lab, key in zip(shares, labels, keys):
                    if key in done_keys:
                        continue

                    print(f"      Inst {inst_idx+1:2d}/{n_inst} [{lab}]...",
                          end=" ", flush=True)

                    try:
                        metrics = run_single_instance(
                            f, x, D, N_budget, phi_ref,
                            explor_share=share, abc_only=abc_only)
                    except Exception as e:
                        print(f"ERROR run: {e} — skipped")
                        continue

                    abc = metrics['abc_shap']
                    if abc_only:
                        print(f"sp_ABC={abc['spearman_mean']:.3f}"
                              f"±{abc['spearman_std']:.3f} | "
                              f"MSE_ABC={abc['mse']:.5f} | "
                              f"N_expl={abc['n_exploration']:.0f} "
                              f"({100*abc['explor_share_realized']:.1f}%)")
                    else:
                        b2 = metrics['b2_stratified']
                        print(f"sp_ABC={abc['spearman_mean']:.3f}"
                              f"±{abc['spearman_std']:.3f} | "
                              f"MSE_ABC={abc['mse']:.5f} | "
                              f"MSE_B2={b2['mse']:.5f} | "
                              f"ratio={abc['mse_ratio_b2']:.2f} | "
                              f"N_expl={abc['n_exploration']:.0f} "
                              f"({100*abc['explor_share_realized']:.1f}%)")

                    new_rows = []
                    for method_name, m in metrics.items():
                        row = {
                            'dataset':           ds_name,
                            'model':             model_name,
                            'n_budget':          N_budget,
                            'instance_idx':      inst_idx,
                            'M':                 M,
                            'method':            method_name,
                            'allocation':        lab,
                            'explor_share_realized': m['explor_share_realized'],
                            'mse':               m['mse'],
                            'mse_std':           m['mse_std'],
                            'mse_norm':          m['mse_norm'],
                            'mse_norm_std':      m['mse_norm_std'],
                            'mse_ci_lower':      m['mse_ci_lower'],
                            'mse_ci_upper':      m['mse_ci_upper'],
                            'mse_ratio_b2':      m['mse_ratio_b2'],
                            'spearman_mean':     m['spearman_mean'],
                            'spearman_std':      m['spearman_std'],
                            'spearman_ci_lower': m['spearman_ci_lower'],
                            'spearman_ci_upper': m['spearman_ci_upper'],
                            'kendall_mean':      m['kendall_mean'],
                            'kendall_std':       m['kendall_std'],
                            'top3_mean':         m['top3_mean'],
                            'top3_std':          m['top3_std'],
                            'top5_mean':         m['top5_mean'],
                            'top5_std':          m['top5_std'],
                            'variance':          m['variance'],
                            'n_exploration':     m['n_exploration'],
                            'n_estimation':      m['n_estimation'],
                            'time_mean':         m['time_mean'],
                            'time_std':          m['time_std'],
                            'n_coal_final':      m['n_coal_final'],
                            'mse_per_run':       str(m['mse_per_run']),
                            'spearman_per_run':  str(m['spearman_per_run']),
                            'kendall_per_run':   str(m['kendall_per_run']),
                            'top3_per_run':      str(m['top3_per_run']),
                            'top5_per_run':      str(m['top5_per_run']),
                        }
                        new_rows.append(row)

                    df_new = pd.DataFrame(new_rows)
                    if os.path.exists(out_path):
                        # garde-fou : schema identique avant append
                        existing = pd.read_csv(out_path, nrows=0).columns.tolist()
                        if existing != list(df_new.columns):
                            df_old = pd.read_csv(out_path)
                            for c in df_new.columns:
                                if c not in df_old.columns:
                                    df_old[c] = pd.NA
                            df_old = df_old[df_new.columns]
                            df_old.to_csv(out_path, mode='w',
                                          header=True, index=False)
                            print("    (schema CSV migre vers le format courant)")
                        df_new.to_csv(out_path, mode='a',
                                      header=False, index=False)
                    else:
                        df_new.to_csv(out_path, mode='w',
                                      header=True, index=False)

    print(f"\n  Dataset {ds_name} — completed.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ABC-SHAP — re-analysis scope + allocation experiment"
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['california', 'adult', 'credit'],
        help="Dataset to process"
    )
    parser.add_argument(
        '--budget', type=int, default=None,
        choices=[1000, 5000, 10000],
        help="Single budget to process (optional)"
    )
    parser.add_argument(
        '--explor-share', type=float, nargs='+', default=None,
        help="Parts d'exploration pour l'experience d'allocation, "
             "ex. --explor-share 0.25 0.75"
    )
    parser.add_argument(
        '--abc-only', action='store_true',
        help="Ne calcule qu'ABC-SHAP (baselines reprises du run principal)"
    )
    args = parser.parse_args()

    budgets_to_run = [args.budget] if args.budget is not None else ALL_BUDGETS

    print("=" * 65)
    print("EXP RUNNER — THESIS RE-ANALYSIS SCOPE")
    print(f"Dataset  : {args.dataset}")
    print(f"Budgets  : {budgets_to_run}")
    print("Models   : ridge | rf | mlp   (3 families)")
    print(f"K_RUNS={K_RUNS} | N_inst={N_TEST_INSTANCES} | "
          f"N_bg={N_BACKGROUND} | REF×{REF_MULTIPLIER}")
    print(f"ABC : SN={ABC_SN} | T_max={ABC_T_MAX} | alpha={ABC_ALPHA}")
    print("Corrections : C1 C2 C3 C4 C5 C6 C7")
    if args.explor_share:
        print(f"Mode     : ALLOCATION EXPERIMENT "
              f"{[_alloc_label(s) for s in args.explor_share]}")
    else:
        print("Mode     : CORRECTED FORMULA (A0)")
    print(f"Methods  : {'ABC-SHAP only' if args.abc_only else 'ABC + B1-B4'}")
    print("=" * 65)

    run_dataset(ds_name=args.dataset, budgets=budgets_to_run,
                explor_shares=args.explor_share, abc_only=args.abc_only)