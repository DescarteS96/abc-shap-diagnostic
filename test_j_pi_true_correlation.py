"""
test_j_pi_true_correlation.py (THESIS RE-ANALYSIS SCOPE)
================================================
Teste si J(pi) (distance euclidienne split-half, telle qu'implementee
actuellement) correle avec la VRAIE stabilite inter-run sous une
politique FIXE -- pas la recherche adaptative elle-meme.

RE-ANALYSIS SCOPE: adapte aux 3 datasets reels de la reanalyse
(California, Adult, Credit Default) au lieu du synthetique pur de la
these originale. Meme protocole, applique separement sur chaque
dataset (M different : 8, 14, 23 -- espaces de politiques non
comparables entre eux).

K_STABILITY releve a 30 (etait 15 dans la these originale) pour rester
coherent avec K_RUNS=30 utilise partout ailleurs dans la reanalyse et
dans l'article.

Ce n'est PAS circulaire : on ne modifie pas J(pi), on le compare
apres coup a une mesure de stabilite independante (K=30 runs completes
sous chaque politique fixee). Verifie avec TROIS mesures de "vraie
stabilite" -- Spearman, Kendall et chevauchement top-3 -- pour eviter
toute dependance au choix specifique de metrique (Etape 2.10 de la
revision V6).

Pour chaque dataset : 30 politiques testees (10 aleatoires + 20 issues
d'une recherche ABC reelle, pour couvrir a la fois l'espace general et
les regions effectivement visitees par la colonie), modele RF, une
instance, N_inner ~ meme ordre de grandeur que le regime contraint reel
(derive de compute_abc_budget_corrected a N=5000 pour chaque M).

INCREMENTAL SAVE + RESUME:
    Chaque politique testee est ajoutee a j_pi_true_correlation.csv des
    qu'elle est evaluee (J(pi) + vraie stabilite calcules). Relancer le
    script saute toute (dataset, policy_idx) deja presente.

Output : j_pi_true_correlation.csv
    colonnes : dataset, M, policy_idx, source, J_pi,
               true_stability_spearman, true_stability_kendall,
               true_stability_top3
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from abc_shap_en import (
    ABCShap, sample_dirichlet_policy, sample_coalition,
    evaluate_policy, kernel_shap_regression
)
from config import GLOBAL_SEED, N_BACKGROUND, ABC_DEFAULT, compute_abc_budget_corrected

RE_ANALYSIS_DATASETS = ['california', 'adult', 'credit']
N_FIXED_EVAL_BUDGET = 5000   # budget de reference pour deriver N_inner realiste par dataset
K_STABILITY = 30             # etait 15 dans la these originale -- releve pour coherence K_RUNS=30
N_RANDOM_POLICIES = 10
N_ABC_VISITED = 20
TOP_K = 3                    # chevauchement top-k utilise comme 3e mesure de stabilite

OUT_PATH = os.path.join(SCRIPT_DIR, "j_pi_true_correlation.csv")


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
        for candidate in ['Y', 'default payment next month', 'default.payment.next.month']:
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


def sample_coalitions_under_policy(pi, M, n_coal):
    sizes = np.arange(1, M)
    z_empty = np.zeros(M, dtype=int)
    z_full = np.ones(M, dtype=int)
    coalitions = [z_empty, z_full]
    for _ in range(n_coal - 2):
        k = np.random.choice(sizes, p=pi)
        coalitions.append(sample_coalition(k, M))
    return coalitions


def true_stability(pi, f, x, D, M, n_coal, K):
    """K estimations independantes sous la politique FIXE pi.
    Retourne la stabilite moyenne inter-run selon trois mesures :
    Spearman, Kendall tau et chevauchement top-3."""
    phis = []
    for _ in range(K):
        coalitions = sample_coalitions_under_policy(pi, M, n_coal)
        phi = kernel_shap_regression(coalitions, f, x, D, M)
        phis.append(phi)
    phis = np.array(phis)

    k3 = min(TOP_K, M)
    top_sets = [set(np.argsort(-np.abs(phis[i]))[:k3]) for i in range(K)]

    sp_pairs, kt_pairs, t3_pairs = [], [], []
    for i in range(K):
        for j in range(i + 1, K):
            sp, _ = spearmanr(np.abs(phis[i]), np.abs(phis[j]))
            kt, _ = kendalltau(np.abs(phis[i]), np.abs(phis[j]))
            sp_pairs.append(sp if not np.isnan(sp) else 0.0)
            kt_pairs.append(kt if not np.isnan(kt) else 0.0)
            t3_pairs.append(len(top_sets[i] & top_sets[j]) / k3)

    return (float(np.mean(sp_pairs)),
            float(np.mean(kt_pairs)),
            float(np.mean(t3_pairs)))


def collect_abc_visited_policies(f, x, D, M, N_budget, n_needed):
    """Fait tourner ABC-SHAP une fois (formule corrigee) et recupere les
    politiques reellement visitees par la colonie (log_['pi_best'] a
    chaque iteration)."""
    p = compute_abc_budget_corrected(N_budget, M,
            SN=ABC_DEFAULT['SN'], T_max=ABC_DEFAULT['T_max'])
    # T_min doit rester <= T_max_eff : sinon plancher > plafond.
    abc = ABCShap(SN=p['SN_eff'], N_inner=p['N_inner'],
                  T_min=min(ABC_DEFAULT['T_min'], p['T_max_eff']),
                  T_max=p['T_max_eff'])
    abc.explain(f, x, D, N_budget)
    visited = list(abc.log_['pi_best'])
    if len(visited) >= n_needed:
        idx = np.random.choice(len(visited), size=n_needed, replace=False)
        return [visited[i] for i in idx]
    while len(visited) < n_needed:
        visited.append(sample_dirichlet_policy(M))
    return visited[:n_needed]


# ══════════════════════════════════════════════════════════════════════════
# INCREMENTAL SAVE + RESUME
# ══════════════════════════════════════════════════════════════════════════

def load_done_keys():
    """Returns the set of (dataset, policy_idx) keys already in OUT_PATH."""
    done = set()
    if os.path.exists(OUT_PATH):
        try:
            df = pd.read_csv(OUT_PATH, usecols=['dataset', 'policy_idx'],
                             on_bad_lines='skip')
            done = set(df[['dataset', 'policy_idx']].itertuples(index=False, name=None))
        except Exception as e:
            print(f"  WARNING resume: {e} -- starting fresh")
    return done


def append_row(row):
    df_new = pd.DataFrame([row])
    if os.path.exists(OUT_PATH):
        # garde-fou : si le fichier date d'une version sans top-3,
        # on le migre avant d'ajouter (evite un CSV a colonnes decalees).
        existing = pd.read_csv(OUT_PATH, nrows=0).columns.tolist()
        if existing != list(df_new.columns):
            df_old = pd.read_csv(OUT_PATH)
            for c in df_new.columns:
                if c not in df_old.columns:
                    df_old[c] = pd.NA
            df_old = df_old[df_new.columns]
            df_old.to_csv(OUT_PATH, mode='w', header=True, index=False)
            print("  (schema CSV migre vers le format courant)")
        df_new.to_csv(OUT_PATH, mode='a', header=False, index=False)
    else:
        df_new.to_csv(OUT_PATH, mode='w', header=True, index=False)


def run():
    done_keys = load_done_keys()
    if done_keys:
        print(f"Resume: {len(done_keys)} (dataset, policy_idx) deja calcules -- skip.\n")

    for dataset in RE_ANALYSIS_DATASETS:
        print(f"\n{'#'*65}\nDATASET: {dataset}\n{'#'*65}")

        X, y, task = load_dataset(dataset)
        M = X.shape[1]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        stratify = y if task == 'classification' else None
        X_train, X_test, y_train, _ = train_test_split(
            X_scaled, y, test_size=0.2, random_state=GLOBAL_SEED, stratify=stratify
        )
        rng = np.random.RandomState(GLOBAL_SEED)
        bg_idx = rng.choice(len(X_train), size=min(N_BACKGROUND, len(X_train)),
                            replace=False)
        D = X_train[bg_idx]
        x = X_test[rng.choice(len(X_test), size=1)[0]]

        if task == 'regression':
            model = RandomForestRegressor(n_estimators=100,
                                          random_state=GLOBAL_SEED, n_jobs=1)
            model.fit(X_train, y_train)
            f = lambda X, _m=model: _m.predict(X)
        else:
            model = RandomForestClassifier(n_estimators=100,
                                           random_state=GLOBAL_SEED, n_jobs=1)
            model.fit(X_train, y_train)
            f = lambda X, _m=model: _m.predict_proba(X)[:, 1]

        # N_inner realiste pour ce dataset, derive de la formule corrigee
        # a N=5000 (meme convention que le reste de la reanalyse).
        p_ref = compute_abc_budget_corrected(N_FIXED_EVAL_BUDGET, M,
                    SN=ABC_DEFAULT['SN'], T_max=ABC_DEFAULT['T_max'])
        n_inner_realistic = p_ref['N_inner']
        print(f"  M={M} | N_inner realiste (derive a N=5000, formule corrigee) "
              f"= {n_inner_realistic}")

        print("  Collecte des politiques a tester...")
        random_policies = [sample_dirichlet_policy(M) for _ in range(N_RANDOM_POLICIES)]
        abc_policies = collect_abc_visited_policies(
            f, x, D, M, N_budget=N_FIXED_EVAL_BUDGET, n_needed=N_ABC_VISITED
        )
        all_policies = random_policies + abc_policies
        labels = ["random"] * N_RANDOM_POLICIES + ["abc_visited"] * N_ABC_VISITED

        for idx, (pi, label) in enumerate(zip(all_policies, labels)):
            key = (dataset, idx)
            if key in done_keys:
                print(f"  [{idx+1}/{len(all_policies)}] {label}: deja fait -- skip")
                continue

            J, _ = evaluate_policy(pi, f, x, D, M, N_inner=n_inner_realistic)
            true_sp, true_kt, true_t3 = true_stability(
                pi, f, x, D, M, n_coal=n_inner_realistic, K=K_STABILITY)

            row = {
                "dataset": dataset,
                "M": M,
                "policy_idx": idx,
                "source": label,
                "J_pi": J,
                "true_stability_spearman": true_sp,
                "true_stability_kendall": true_kt,
                "true_stability_top3": true_t3,
            }
            append_row(row)
            print(f"  [{idx+1}/{len(all_policies)}] {label}: J={J:.4f}  "
                  f"true_sp={true_sp:.4f}  true_kt={true_kt:.4f}  "
                  f"true_t3={true_t3:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # SYNTHESE — tableau de l'Etape 2.10 (Pearson / Spearman / Kendall)
    # ══════════════════════════════════════════════════════════════════
    df = pd.read_csv(OUT_PATH)
    targets = [
        ("Split-half vs Spearman", "true_stability_spearman"),
        ("Split-half vs Kendall",  "true_stability_kendall"),
        ("Split-half vs top-3",    "true_stability_top3"),
    ]

    print("\n" + "=" * 78)
    print("CORRELATION J(pi) vs VRAIE STABILITE (mesuree independamment)")
    print("=" * 78)

    for dataset in RE_ANALYSIS_DATASETS:
        sub = df[df['dataset'] == dataset]
        if len(sub) < 5:
            print(f"\n{dataset}: pas assez d'observations ({len(sub)}), skip")
            continue
        print(f"\n{dataset.upper()} (n={len(sub)}):")
        print(f"  {'Relation':<26} {'Pearson':>9} {'p':>8} "
              f"{'Spearman':>10} {'p':>8} {'Kendall':>9} {'p':>8}")
        for lab, col in targets:
            xj, yv = sub["J_pi"].values, sub[col].values
            if np.std(yv) == 0 or np.std(xj) == 0:
                print(f"  {lab:<26} {'n/a':>9} {'n/a':>8} "
                      f"{'n/a':>10} {'n/a':>8} {'n/a':>9} {'n/a':>8}")
                continue
            r_p,  p_p  = pearsonr(xj, yv)
            r_sp, p_sp = spearmanr(xj, yv)
            r_kt, p_kt = kendalltau(xj, yv)
            print(f"  {lab:<26} {r_p:>9.4f} {p_p:>8.4f} "
                  f"{r_sp:>10.4f} {p_sp:>8.4f} {r_kt:>9.4f} {p_kt:>8.4f}")

    # Aggregat toutes cellules confondues (J(pi) standardise par dataset,
    # les echelles de J ne sont pas comparables entre M differents).
    print("\n" + "-" * 78)
    print("POOLED (J(pi) centre-reduit par dataset) :")
    d2 = df.copy()
    d2['J_z'] = d2.groupby('dataset')['J_pi'].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0.0)
    print(f"  {'Relation':<26} {'Pearson':>9} {'p':>8} "
          f"{'Spearman':>10} {'p':>8} {'Kendall':>9} {'p':>8}")
    for lab, col in targets:
        xj, yv = d2["J_z"].values, d2[col].values
        if np.std(yv) == 0 or np.std(xj) == 0:
            continue
        r_p,  p_p  = pearsonr(xj, yv)
        r_sp, p_sp = spearmanr(xj, yv)
        r_kt, p_kt = kendalltau(xj, yv)
        print(f"  {lab:<26} {r_p:>9.4f} {p_p:>8.4f} "
              f"{r_sp:>10.4f} {p_sp:>8.4f} {r_kt:>9.4f} {p_kt:>8.4f}")

    print("\nInterpretation : une correlation POSITIVE et significative confirmerait")
    print("que J(pi) est un bon proxy de la stabilite ordinale. Une correlation")
    print("faible, nulle ou negative indiquerait que le signal d'optimisation")
    print("interne est deconnecte de l'objectif cible.")
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    run()