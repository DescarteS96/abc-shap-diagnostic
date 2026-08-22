"""
check_sigma_rho.py
==================
Verification independante du tableau de dispersion inter-executions
(sigma_rho) utilise en Section 5.1 de l'article.

ORIGINE DE LA DONNEE
    La colonne 'spearman_std' de chaque CSV de la grille principale.
    Elle est produite par exp_runner.py :

        spearman_std = float(np.std(spearman_vals))

    ou spearman_vals est la liste des K = 30 correlations de Spearman
    obtenues aux K repetitions independantes d'une meme cellule
    (dataset, model, method, instance). C'est exactement la quantite
    definie en Section 4.6 de la these sous le nom sigma_rho, et
    utilisee par le critere R2 (seuil 0.15).

    La colonne 'spearman_per_run' contient les K valeurs elles-memes ;
    le script recalcule sigma_rho a partir d'elles et verifie que le
    resultat coincide avec la colonne stockee. Si l'ecart depasse 1e-9,
    le script s'arrete : cela signifierait que la colonne ne mesure pas
    ce qu'on croit.

PERIMETRE
    Ridge sur California Housing est exclu : les cinq methodes y
    retournent un Spearman de 1.0000 avec variance nulle, donc
    sigma_rho = 0 pour tout le monde et la cellule n'informe pas.
    C'est la meme exclusion que dans les Tables 1 et 2.

Usage :
    python check_sigma_rho.py \
        results_step3_corrected/results_california_N1000.csv \
        results_step3_corrected/results_california_N5000.csv \
        results_step3_corrected/results_adult_N1000.csv \
        results_step3_corrected/results_adult_N5000.csv \
        results_step3_corrected/results_credit_N1000.csv \
        results_step3_corrected/results_credit_N5000.csv
"""

import ast
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METHODS = {'abc_shap': 'ABC-SHAP', 'b1_uniform': 'B1',
           'b2_stratified': 'B2', 'b3_is': 'B3', 'b4_antithetic': 'B4'}
DS_LABEL = {'california': 'California', 'adult': 'Adult',
            'credit': 'Credit Default'}
RANK_DEGENERATE = {('california', 'ridge')}


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        sys.exit(1)

    rows, pooled = [], {k: [] for k in METHODS}

    for p in paths:
        if not os.path.exists(p):
            print(f"ERREUR : fichier introuvable : {p}")
            sys.exit(1)
        d = pd.read_csv(p)
        name = os.path.basename(p)

        # ── Controle 1 : la colonne stockee correspond-elle aux K valeurs ?
        runs = d['spearman_per_run'].apply(ast.literal_eval)
        k_vals = sorted(set(runs.apply(len)))
        recomputed = runs.apply(lambda v: float(np.std(v)))
        gap = float((recomputed - d['spearman_std']).abs().max())
        if gap > 1e-9:
            print(f"ERREUR : {name} — spearman_std ne correspond pas aux "
                  f"K valeurs par run (ecart max {gap:.2e}).")
            sys.exit(1)

        ds = str(d['dataset'].iloc[0])
        N = int(d['n_budget'].iloc[0])
        keep = d[~d.apply(lambda r: (r['dataset'], r['model'])
                          in RANK_DEGENERATE, axis=1)]
        w = keep.pivot_table(index=['model', 'instance_idx'],
                             columns='method', values='spearman_std')

        print(f"[check] {name:38s} K={k_vals} | ecart recalcul "
              f"{gap:.1e} | n={len(w)} instances informatives")

        rows.append((DS_LABEL.get(ds, ds), N, len(w),
                     {m: float(w[m].mean()) for m in METHODS}))
        for m in METHODS:
            pooled[m] += list(w[m].values)

    # ── Tableau ──────────────────────────────────────────────────────────
    print()
    print("sigma_rho — ecart-type inter-executions du Spearman (K = 30)")
    print(f"{'cellule':<22}{'n':>4}" + ''.join(f"{v:>11}" for v in METHODS.values()))
    print('-' * (26 + 11 * len(METHODS)))
    for lab, N, n, means in rows:
        print(f"{lab + ' N=' + str(N):<22}{n:>4}"
              + ''.join(f"{means[m]:>11.4f}" for m in METHODS))
    print('-' * (26 + 11 * len(METHODS)))
    n_tot = len(pooled['abc_shap'])
    print(f"{'AGREGE':<22}{n_tot:>4}"
          + ''.join(f"{np.mean(pooled[m]):>11.4f}" for m in METHODS))

    # ── Tests apparies ───────────────────────────────────────────────────
    a = np.array(pooled['abc_shap'])
    print()
    print(f"Tests de Wilcoxon apparies sur sigma_rho (n = {n_tot}).")
    print("Une valeur PLUS BASSE signifie une methode PLUS reproductible.")
    for m, lab in METHODS.items():
        if m == 'abc_shap':
            continue
        b = np.array(pooled[m])
        st, pv = wilcoxon(a, b)
        med = float(np.median(a - b))
        verdict = ("ABC-SHAP moins reproductible" if med > 0
                   else "ABC-SHAP plus reproductible")
        print(f"  ABC-SHAP vs {lab:<3} : mediane(diff) = {med:+.4f}   "
              f"p = {pv:.3e}   {verdict}")

    # ── Critere R2 de la these ───────────────────────────────────────────
    print()
    print("Critere R2 de la these (sigma_rho moyen < 0.15) :")
    for m, lab in METHODS.items():
        v = np.mean(pooled[m])
        print(f"  {lab:<10} {v:.4f}  {'OK' if v < 0.15 else 'ECHEC'}")
    worst = max(rows, key=lambda r: r[3]['abc_shap'])
    print(f"  cellule la plus dispersee pour ABC-SHAP : "
          f"{worst[0]} N={worst[1]} -> {worst[3]['abc_shap']:.4f}")


if __name__ == '__main__':
    main()