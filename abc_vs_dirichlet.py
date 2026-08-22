"""
abc_vs_dirichlet.py (THESIS RE-ANALYSIS SCOPE)
====================
Control experiment — ABC-SHAP versus a single random Dirichlet policy
draw, under the CORRECTED budget formula.

RE-ANALYSIS SCOPE — SINGLE VERSION ONLY (Version B from the original
thesis script). Version A (original-formula parity test) has been
removed entirely: this re-analysis exists specifically to redo the
analyses under the corrected formula, per the consultant's request —
no script in this pipeline should read or produce original-formula
results anymore.

Question this test answers: once the budget is properly allocated
between exploration and estimation (guaranteed >=50% reserved for the
final estimation step), does the ABC colony search still add value
over a single random Dirichlet policy draw?

Design:
    Datasets  : California (M=8, regression), Adult (M=14, classification),
                Credit Default (M=23, classification)
    Budget    : N=5000
    K         : 30 runs per method
    Instances : 10   (identical to the main grid: same seed, same split,
                      same rng.choice, so the instances are the same ones)
    Models    : ridge/logistic, rf, mlp
    Methods   : abc_corrected, dirichlet_strong, b1_uniform, b2_stratified

BUDGET GUARD
    run_abc_corrected asserts that the exploration budget actually
    consumed by the colony equals the planned one. Without this check a
    missing or wrong T_max silently multiplies the exploration cost and
    the resulting numbers look plausible.

INCREMENTAL SAVE + RESUME (INSTANCE-LEVEL)
    phi_ref is a stochastic quantity (B2 at 10 x N coalitions), so every
    method of a given instance must be measured against the SAME draw.
    Resume therefore works at instance granularity: a partially computed
    instance is deleted from the CSV and recomputed in full, rather than
    completed against a fresh reference.

Output:
    abc_vs_dirichlet_results.csv
    abc_vs_dirichlet_summary.txt
"""

import os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from abc_shap_en import ABCShap, evaluate_policy, sample_dirichlet_policy
from baseline_b1_uniform import KernelSHAPUniform
from baseline_b2_stratified import KernelSHAPStratified
from config import (
    GLOBAL_SEED, ABC_DEFAULT, REF_MULTIPLIER,
    compute_abc_budget_corrected,
)

RE_ANALYSIS_DATASETS = ['california', 'adult', 'credit']
N_BUDGET = 5000
K_RUNS   = 30
N_BG     = 100
N_INST   = 10
MODEL_FAMILIES = ['ridge', 'rf', 'mlp']

OUT_CSV = os.path.join(SCRIPT_DIR, 'abc_vs_dirichlet_results.csv')
OUT_TXT = os.path.join(SCRIPT_DIR, 'abc_vs_dirichlet_summary.txt')


# ══════════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════════

def load_dataset(name):
    """Returns (X, y, task). Same loaders as exp_runner.py (re-analysis scope)."""
    if name == 'california':
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing()
        return d.data.astype(float), d.target.astype(float), 'regression'

    elif name == 'adult':
        from sklearn.datasets import fetch_openml
        data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
        df   = data.frame.dropna()
        y_raw = df['class'].values
        X_df  = df.drop('class', axis=1)
        le    = LabelEncoder()
        for col in X_df.select_dtypes(include='category').columns:
            X_df[col] = le.fit_transform(X_df[col].astype(str))
        X = X_df.values.astype(float)
        y = (y_raw == '>50K').astype(int)
        return X, y, 'classification'

    elif name == 'credit':
        from sklearn.datasets import fetch_openml
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
        y = df[target_col].values.astype(int)
        X_df = df.drop(columns=[target_col])
        le = LabelEncoder()
        for col in X_df.select_dtypes(include=['object', 'category']).columns:
            X_df[col] = le.fit_transform(X_df[col].astype(str))
        X = X_df.values.astype(float)
        return X, y, 'classification'

    else:
        raise ValueError(f"Unknown dataset '{name}'. "
                         f"Re-analysis scope choices: california, adult, credit")


def load_split_and_instances(name):
    X, y, task = load_dataset(name)
    np.random.seed(GLOBAL_SEED)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    stratify = y if task == 'classification' else None
    X_train, X_test, y_train, _ = train_test_split(
        X_scaled, y, test_size=0.2, random_state=GLOBAL_SEED, stratify=stratify
    )
    rng = np.random.RandomState(GLOBAL_SEED)
    n_inst = min(N_INST, len(X_test))
    inst_idx = rng.choice(len(X_test), size=n_inst, replace=False)
    X_sel = X_test[inst_idx]
    bg_idx = rng.choice(len(X_train), size=min(N_BG, len(X_train)), replace=False)
    D = X_train[bg_idx]
    return X_train, y_train, X_sel, D, task


def train_models(X_train, y_train, task):
    models = {}
    if task == 'regression':
        m = Ridge(alpha=1.0)
        m.fit(X_train, y_train)
        models['ridge'] = lambda X, _m=m: _m.predict(X)
        m = RandomForestRegressor(n_estimators=100,
                                  random_state=GLOBAL_SEED, n_jobs=1)
        m.fit(X_train, y_train)
        models['rf'] = lambda X, _m=m: _m.predict(X)
        m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500,
                         random_state=GLOBAL_SEED, early_stopping=True,
                         validation_fraction=0.1, n_iter_no_change=10)
        m.fit(X_train, y_train)
        models['mlp'] = lambda X, _m=m: _m.predict(X)
    else:
        m = LogisticRegression(C=1.0, max_iter=1000,
                               random_state=GLOBAL_SEED, n_jobs=1)
        m.fit(X_train, y_train)
        models['ridge'] = lambda X, _m=m: _m.predict_proba(X)[:, 1]
        m = RandomForestClassifier(n_estimators=100,
                                   random_state=GLOBAL_SEED, n_jobs=1)
        m.fit(X_train, y_train)
        models['rf'] = lambda X, _m=m: _m.predict_proba(X)[:, 1]
        m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                          random_state=GLOBAL_SEED, early_stopping=True,
                          validation_fraction=0.1, n_iter_no_change=10)
        m.fit(X_train, y_train)
        models['mlp'] = lambda X, _m=m: _m.predict_proba(X)[:, 1]
    return models


# ══════════════════════════════════════════════════════════════════════════
# METRIQUES
# ══════════════════════════════════════════════════════════════════════════

def compute_phi_ref(f, x, D):
    return KernelSHAPStratified(
        N_coalitions=N_BUDGET * REF_MULTIPLIER).explain(f, x, D)


def sp(phi, ref):
    rho, _ = spearmanr(np.abs(phi), np.abs(ref))
    return float(rho) if not np.isnan(rho) else 0.0


def mse(phi, ref):
    return float(np.sum((phi - ref) ** 2))


# ══════════════════════════════════════════════════════════════════════════
# METHODES COMPAREES
# ══════════════════════════════════════════════════════════════════════════

def run_abc_corrected(f, x, D, M):
    """ABC-SHAP sous la formule corrigee : la colonie explore, puis le
    budget residuel sert a l'estimation finale."""
    p = compute_abc_budget_corrected(
        N_BUDGET, M, SN=ABC_DEFAULT['SN'], T_max=ABC_DEFAULT['T_max'])
    abc = ABCShap(
        SN=p['SN_eff'],
        N_inner=p['N_inner'],
        # T_min doit rester <= T_max_eff : sinon plancher > plafond.
        T_min=min(ABC_DEFAULT['T_min'], p['T_max_eff']),
        T_max=p['T_max_eff'],
        epsilon_phi=ABC_DEFAULT['epsilon'],
        patience=ABC_DEFAULT['patience'],
        alpha_mut=ABC_DEFAULT['alpha_mut'],
    )
    phi = abc.explain(f, x, D, N_BUDGET)

    # Garde-fou : sans lui, un T_max absent ou errone multiplie
    # silencieusement le cout d'exploration et les chiffres restent
    # plausibles a la lecture.
    used = getattr(abc, 'eval_count_', None)
    if used is None:
        raise RuntimeError(
            "ABCShap n'expose pas eval_count_ : le budget d'exploration "
            "consomme ne peut pas etre verifie.")
    used = int(used)
    assert used == p['N_explor'], (
        f"budget d'exploration incoherent : consomme={used}, "
        f"planifie={p['N_explor']} (N={N_BUDGET}, M={M}, "
        f"SN_eff={p['SN_eff']}, T_eff={p['T_max_eff']}, "
        f"N_inner={p['N_inner']})")
    return phi


def run_dirichlet_strong(f, x, D, M):
    """Un seul tirage aleatoire de politique sur le simplexe, budget
    complet N_BUDGET consacre directement a l'estimation : aucune
    exploration, aucune optimisation.

    evaluate_policy renvoie (J, phi) ou phi est l'estimation sur
    l'ECHANTILLON COMPLET, pas sur une moitie du split-half : les deux
    methodes comparees recoivent donc bien le meme budget."""
    pi = sample_dirichlet_policy(M)
    _, phi = evaluate_policy(pi, f, x, D, M, N_BUDGET)
    return phi


# ══════════════════════════════════════════════════════════════════════════
# INCREMENTAL SAVE + RESUME
# ══════════════════════════════════════════════════════════════════════════

def load_done_keys():
    """Retourne l'ensemble des cles (dataset, model, instance_idx, method)
    deja presentes dans OUT_CSV."""
    done = set()
    if os.path.exists(OUT_CSV):
        try:
            df = pd.read_csv(
                OUT_CSV,
                usecols=['dataset', 'model', 'instance_idx', 'method'],
                on_bad_lines='skip')
            done = set(df[['dataset', 'model', 'instance_idx', 'method']]
                       .itertuples(index=False, name=None))
        except Exception as e:
            print(f"  WARNING resume: {e} -- starting fresh")
    return done


def append_row(row):
    df_new = pd.DataFrame([row])
    if os.path.exists(OUT_CSV):
        df_new.to_csv(OUT_CSV, mode='a', header=False, index=False)
    else:
        df_new.to_csv(OUT_CSV, mode='w', header=True, index=False)


def drop_instance_rows(dataset, model_name, ii):
    """Supprime physiquement du CSV les lignes d'une instance partielle.
    Sans cela elles coexisteraient avec les nouvelles, mesurees contre une
    reference differente, et le fichier contiendrait deux valeurs pour la
    meme cle."""
    if not os.path.exists(OUT_CSV):
        return 0
    df = pd.read_csv(OUT_CSV)
    keep = ~((df['dataset'] == dataset) &
             (df['model'] == model_name) &
             (df['instance_idx'] == ii))
    n = int((~keep).sum())
    if n:
        df[keep].to_csv(OUT_CSV, mode='w', header=True, index=False)
    return n


# ══════════════════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 65)
    print("ABC-SHAP vs. RANDOM DIRICHLET — CORRECTED FORMULA (RE-ANALYSIS)")
    print("Question: does ABC search add value over a random policy draw")
    print("once the budget is properly allocated?")
    print(f"Datasets: {RE_ANALYSIS_DATASETS} | N={N_BUDGET} | "
          f"K={K_RUNS} | N_inst={N_INST}")
    print("=" * 65)

    done_keys = load_done_keys()
    if done_keys:
        print(f"Resume: {len(done_keys)} (dataset,model,instance,method) "
              f"rows already complete.\n")

    method_runners = {
        'abc_corrected':    run_abc_corrected,
        'dirichlet_strong': run_dirichlet_strong,
        'b1_uniform':       lambda f, x, D, M:
            KernelSHAPUniform(N_coalitions=N_BUDGET).explain(f, x, D),
        'b2_stratified':    lambda f, x, D, M:
            KernelSHAPStratified(N_coalitions=N_BUDGET).explain(f, x, D),
    }
    n_methods = len(method_runners)

    for dataset in RE_ANALYSIS_DATASETS:
        print(f"\n{'#'*65}\nDATASET: {dataset}\n{'#'*65}")
        X_train, y_train, X_sel, D, task = load_split_and_instances(dataset)
        models = train_models(X_train, y_train, task)
        M = X_sel.shape[1]

        for model_name, f in models.items():
            print(f"\n=== {dataset} | Model: {model_name} "
                  f"(M={M}, task={task}) ===")

            for ii, x in enumerate(X_sel):
                # ── Reprise au niveau de l'instance ────────────────────
                # phi_ref est stochastique : toutes les methodes d'une
                # meme instance doivent etre mesurees contre le meme
                # tirage, donc calculees dans la meme session.
                done_here = [m for m in method_runners
                             if (dataset, model_name, ii, m) in done_keys]

                if len(done_here) == n_methods:
                    print(f"  Instance {ii+1}/{len(X_sel)}: "
                          f"all methods already done -- skip")
                    continue

                if done_here:
                    n_removed = drop_instance_rows(dataset, model_name, ii)
                    print(f"  Instance {ii+1}/{len(X_sel)}: partielle "
                          f"({len(done_here)}/{n_methods}) — {n_removed} "
                          f"lignes supprimees, recalcul complet pour garder "
                          f"une reference unique")
                    done_keys -= {(dataset, model_name, ii, m)
                                  for m in method_runners}

                print(f"  Instance {ii+1}/{len(X_sel)}:")
                ref = compute_phi_ref(f, x, D)

                inst_results = {}
                for method_name, runner in method_runners.items():
                    t0 = time.perf_counter()
                    phis = [runner(f, x, D, M) for _ in range(K_RUNS)]
                    elapsed = time.perf_counter() - t0

                    sp_vals  = [sp(ph, ref) for ph in phis]
                    mse_vals = [mse(ph, ref) for ph in phis]
                    sp_mean  = float(np.mean(sp_vals))
                    ms_mean  = float(np.mean(mse_vals))
                    inst_results[method_name] = sp_mean

                    print(f"    {method_name:22s}: sp={sp_mean:.4f}  "
                          f"mse={ms_mean:.5f}  ({elapsed:.1f}s)")

                    append_row({
                        'dataset':       dataset,
                        'model':         model_name,
                        'instance_idx':  ii,
                        'method':        method_name,
                        'spearman_mean': sp_mean,
                        'spearman_std':  float(np.std(sp_vals)),
                        'mse_mean':      ms_mean,
                        'mse_std':       float(np.std(mse_vals)),
                        'n_budget':      N_BUDGET,
                        'k_runs':        K_RUNS,
                    })
                    done_keys.add((dataset, model_name, ii, method_name))

                if ('abc_corrected' in inst_results
                        and 'dirichlet_strong' in inst_results):
                    d = (inst_results['abc_corrected']
                         - inst_results['dirichlet_strong'])
                    print(f"    --> delta(ABC - Dirichlet): sp={d:+.4f}")

    # ══════════════════════════════════════════════════════════════════
    # SYNTHESE
    # ══════════════════════════════════════════════════════════════════
    df = pd.read_csv(OUT_CSV)
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    agg = (df.groupby(['dataset', 'model', 'method'])
             [['spearman_mean', 'mse_mean']].mean().round(5))
    print(agg.to_string())

    summary_lines = [f"\nCorrected-formula control test (re-analysis) — "
                     f"N={N_BUDGET} | K={K_RUNS} | N_inst={N_INST}\n"]
    for dataset in RE_ANALYSIS_DATASETS:
        for model in MODEL_FAMILIES:
            sub = df[(df['dataset'] == dataset) & (df['model'] == model)]
            if sub.empty:
                continue
            abc_sp  = sub[sub['method'] == 'abc_corrected']['spearman_mean'].mean()
            dir_sp  = sub[sub['method'] == 'dirichlet_strong']['spearman_mean'].mean()
            abc_mse = sub[sub['method'] == 'abc_corrected']['mse_mean'].mean()
            dir_mse = sub[sub['method'] == 'dirichlet_strong']['mse_mean'].mean()
            if abc_sp != abc_sp or dir_sp != dir_sp:
                continue
            dsp  = abc_sp - dir_sp
            dmse = abc_mse - dir_mse
            if   dsp >  0.005: verdict = "ABC BETTER on Spearman"
            elif dsp < -0.005: verdict = "DIRICHLET BETTER on Spearman"
            else:              verdict = "NO MEANINGFUL DIFFERENCE"
            line = (f"  {dataset.upper()} / {model.upper()}: "
                    f"delta_sp={dsp:+.4f}  delta_mse={dmse:+.6f}  [{verdict}]")
            print(line)
            summary_lines.append(line)

    with open(OUT_TXT, 'a', encoding='utf-8') as fh:
        fh.write("\n".join(summary_lines) + "\n")
    print(f"Appended -> {OUT_TXT}")


if __name__ == "__main__":
    run()