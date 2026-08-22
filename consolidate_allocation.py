"""
consolidate_allocation.py
==========================
Consolide les quatre sources de l'experience d'allocation (A0, A1, A2, A3)
en un seul fichier normalise, pret pour le tableau du 2.6 et les
graphiques du 2.7.

DECISION ACTEE : "Originale" (dans le tableau du consultant) designe la
formule D'ORIGINE, pas la formule corrigee. A0 est donc le point a
0,78 % d'estimation (99,22 % d'exploration), pas un doublon de A2.

Sources attendues :
    A0    : grille d'origine        — 200 lignes, 4 modeles dont xgb, 4961/39
    A2    : grille corrigee         — 150 lignes, 3 modeles, 2496/2504
    A1/A3 : results_allocation_*.csv —  60 lignes, abc_shap seul, 3 modeles

Les fichiers A0 et A2 portent souvent le MEME nom dans des dossiers
differents : passez les chemins complets, le script ne les confond pas
et verifie la nature de chacun avant de continuer.

Usage (chemins complets, guillemets si espaces dans le chemin) :
    python consolidate_allocation.py \
        --a0  "C:/.../original/results_california_N5000.csv" \
        --a2  "C:/.../results_step3_corrected/results_california_N5000.csv" \
        --a13 "C:/.../results_step3_corrected/results_allocation_california_N5000.csv" \
        --out "C:/.../results_step3_corrected/results_allocation_consolidated_california_N5000.csv"
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

N_BUDGET    = 5000
MODELS_KEPT = ['ridge', 'rf', 'mlp']

# Parts d'exploration attendues — arithmetique deja verifiee contre les
# en-tetes affiches par exp_runner.py. Aucun calcul experimental ici.
EXPLOR_SHARE_EXPECTED = {
    'A0': None,      # derive ligne a ligne : 4961/5000 = 0.9922
    'A1': 0.2496,
    'A3': 0.7488,
    'A2': 0.4992,
}
A0_SHARE_MIN, A0_SHARE_MAX = 0.98, 1.0

# Ordre de colonnes lisible a l'oeil ; le reste suit par ordre alphabetique.
LEAD_COLS = [
    'dataset', 'model', 'n_budget', 'instance_idx', 'M', 'method',
    'allocation', 'explor_share_realized', 'n_exploration', 'n_estimation',
    'spearman_mean', 'spearman_std', 'kendall_mean', 'top3_mean', 'top5_mean',
    'mse', 'mse_ratio_b2',
]


def log(msg):
    print(f"[consolidate] {msg}")


def die(msg):
    print(f"[consolidate] ERREUR — {msg}")
    sys.exit(1)


def _check_path(path, label):
    if not os.path.exists(path):
        die(f"{label} : fichier introuvable -> {path}")
    return os.path.abspath(path)


# ══════════════════════════════════════════════════════════════════════════
# A0 — formule d'origine
# ══════════════════════════════════════════════════════════════════════════

def load_a0(path):
    path = _check_path(path, 'A0')
    raw = pd.read_csv(path)
    log(f"A0 : {path}")
    log(f"A0 : {len(raw)} lignes, modeles presents = {sorted(raw['model'].unique())}")

    # Signature n.1 — la grille d'origine contient XGBoost.
    if 'xgb' not in set(raw['model'].astype(str)):
        die("A0 : aucun modele 'xgb' dans ce fichier. La grille d'origine en "
            "contient (200 lignes, 4 modeles). Le chemin --a0 pointe "
            "probablement vers le fichier de la formule CORRIGEE.")

    if 'n_exploration' not in raw.columns or 'n_estimation' not in raw.columns:
        die("A0 : colonnes n_exploration/n_estimation absentes.")

    df = raw[raw['model'].isin(MODELS_KEPT)].copy()
    log(f"A0 : {len(df)} lignes conservees apres filtrage xgb "
        f"(modeles gardes : {MODELS_KEPT})")

    df['allocation'] = 'A0'
    df['explor_share_realized'] = np.where(
        df['method'] == 'abc_shap',
        df['n_exploration'] / float(N_BUDGET),
        0.0,
    )

    # Signature n.2 — la part d'exploration ABC doit valoir ~0.9922.
    shares = df.loc[df['method'] == 'abc_shap', 'explor_share_realized'].unique()
    if len(shares) == 0:
        die("A0 : aucune ligne 'abc_shap' trouvee.")
    if not all(A0_SHARE_MIN <= float(s) <= A0_SHARE_MAX for s in shares):
        die(f"A0 : explor_share_realized attendu ~0.9922 (4961/5000), "
            f"trouve {np.round(shares, 4)}. Ce fichier n'est PAS la formule "
            f"d'origine. Verifiez le chemin --a0.")
    log(f"A0 : part d'exploration ABC = {float(shares[0]):.4f} — conforme")
    return df


# ══════════════════════════════════════════════════════════════════════════
# A2 — formule corrigee
# ══════════════════════════════════════════════════════════════════════════

def load_a2(path):
    path = _check_path(path, 'A2')
    raw = pd.read_csv(path)
    log(f"A2 : {path}")
    log(f"A2 : {len(raw)} lignes, modeles presents = {sorted(raw['model'].unique())}")

    if 'xgb' in set(raw['model'].astype(str)):
        die("A2 : ce fichier contient 'xgb', il s'agit de la grille "
            "D'ORIGINE. Le chemin --a2 doit pointer vers la grille CORRIGEE "
            "(150 lignes, 3 modeles).")

    df = raw[raw['model'].isin(MODELS_KEPT)].copy()
    log(f"A2 : {len(df)} lignes conservees")

    df['allocation'] = 'A2'
    df['explor_share_realized'] = np.where(
        df['method'] == 'abc_shap',
        df['n_exploration'] / float(N_BUDGET),
        0.0,
    )

    shares = df.loc[df['method'] == 'abc_shap', 'explor_share_realized'].unique()
    if len(shares) == 0:
        die("A2 : aucune ligne 'abc_shap' trouvee.")
    exp = EXPLOR_SHARE_EXPECTED['A2']
    if not all(abs(float(s) - exp) < 1e-4 for s in shares):
        die(f"A2 : explor_share_realized attendu {exp} (2496/5000), "
            f"trouve {np.round(shares, 4)}. Verifiez le chemin --a2.")
    log(f"A2 : part d'exploration ABC = {float(shares[0]):.4f} — conforme")
    return df


# ══════════════════════════════════════════════════════════════════════════
# A1 / A3 — experience d'allocation (abc_shap seul)
# ══════════════════════════════════════════════════════════════════════════

def load_a1_a3(path, b2_lookup):
    path = _check_path(path, 'A1/A3')
    df = pd.read_csv(path)
    log(f"A1/A3 : {path}")
    log(f"A1/A3 : {len(df)} lignes chargees "
        f"(attendu 60 = 3 modeles x 10 instances x 2 allocations)")

    if 'allocation' not in df.columns:
        die("A1/A3 : colonne 'allocation' absente. Ce fichier n'a pas ete "
            "produit par la version d'exp_runner.py avec --explor-share.")

    found = set(df['allocation'].astype(str).unique())
    if found - {'A1', 'A3'}:
        die(f"A1/A3 : valeurs inattendues dans 'allocation' : {sorted(found)}. "
            f"Attendu uniquement A1 et A3.")
    if found != {'A1', 'A3'}:
        log(f"A1/A3 : ATTENTION — allocations presentes : {sorted(found)}. "
            f"Le run n'est peut-etre pas termine.")

    # Controle des parts realisees contre les valeurs attendues.
    for al in sorted(found):
        exp = EXPLOR_SHARE_EXPECTED[al]
        got = df.loc[df['allocation'] == al, 'explor_share_realized'].unique()
        if len(got) != 1 or abs(float(got[0]) - exp) > 1e-6:
            die(f"{al} : explor_share_realized attendu {exp}, "
                f"trouve {np.round(got, 6)}.")
        log(f"A1/A3 : {al} — part d'exploration = {float(got[0]):.4f} — conforme")

    if set(df['method'].astype(str).unique()) != {'abc_shap'}:
        log(f"A1/A3 : ATTENTION — methodes presentes : "
            f"{sorted(set(df['method'].unique()))}. Le mode --abc-only "
            f"n'attend que 'abc_shap'.")

    # Recalcul de mse_ratio_b2 par jointure sur (model, instance_idx) avec
    # le MSE de b2_stratified extrait de A2. B2 utilise le budget complet
    # quelle que soit l'allocation ABC : la jointure est legitime.
    before = len(df)
    df = df.merge(b2_lookup, on=['model', 'instance_idx'], how='left')
    if len(df) != before:
        die(f"A1/A3 : la jointure B2 a change le nombre de lignes "
            f"({before} -> {len(df)}). Le fichier A2 contient probablement "
            f"des doublons sur (model, instance_idx).")

    missing = int(df['mse_b2_ref'].isna().sum())
    if missing > 0:
        log(f"A1/A3 : ATTENTION — {missing} lignes sans correspondance B2. "
            f"mse_ratio_b2 restera vide pour ces lignes.")
    df['mse_ratio_b2'] = df['mse'] / df['mse_b2_ref']
    df = df.drop(columns=['mse_b2_ref'])
    log(f"A1/A3 : mse_ratio_b2 recalcule depuis le B2 de A2 "
        f"({len(df) - missing}/{len(df)} lignes)")
    return df


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Consolide A0/A1/A2/A3 en un seul fichier normalise."
    )
    ap.add_argument('--a0', required=True,
                    help="Chemin COMPLET du fichier formule d'origine "
                         "(200 lignes, 4 modeles dont xgb)")
    ap.add_argument('--a2', required=True,
                    help="Chemin COMPLET du fichier formule corrigee "
                         "(150 lignes, 3 modeles) — sert aussi de reference B2")
    ap.add_argument('--a13', required=True,
                    help="Chemin COMPLET de results_allocation_*.csv (A1 + A3)")
    ap.add_argument('--out',
                    default='results_allocation_consolidated_california_N5000.csv',
                    help="Chemin du fichier consolide a ecrire")
    args = ap.parse_args()

    # A2 en premier : il fournit la table de correspondance B2.
    df_a2 = load_a2(args.a2)

    b2_lookup = (
        df_a2.loc[df_a2['method'] == 'b2_stratified',
                  ['model', 'instance_idx', 'mse']]
        .rename(columns={'mse': 'mse_b2_ref'})
    )
    if len(b2_lookup) == 0:
        die("aucune ligne 'b2_stratified' dans le fichier A2 : impossible de "
            "recalculer mse_ratio_b2 pour A1/A3.")
    if b2_lookup.duplicated(['model', 'instance_idx']).any():
        die("le fichier A2 contient des doublons sur (model, instance_idx) "
            "pour b2_stratified.")
    log(f"Reference B2 : {len(b2_lookup)} couples (model, instance_idx)")

    df_a0  = load_a0(args.a0)
    df_a13 = load_a1_a3(args.a13, b2_lookup)

    # ── Alignement des colonnes, ordre lisible ────────────────────────────
    union = set(df_a0.columns) | set(df_a2.columns) | set(df_a13.columns)
    all_cols = [c for c in LEAD_COLS if c in union] + sorted(union - set(LEAD_COLS))

    for name, d in [('A0', df_a0), ('A2', df_a2), ('A1/A3', df_a13)]:
        missing_cols = union - set(d.columns)
        if missing_cols:
            log(f"{name} : colonnes absentes ajoutees en vide -> "
                f"{sorted(missing_cols)}")

    df_all = pd.concat(
        [df_a0.reindex(columns=all_cols),
         df_a2.reindex(columns=all_cols),
         df_a13.reindex(columns=all_cols)],
        ignore_index=True,
    )

    # ── Controle de doublons ──────────────────────────────────────────────
    dup_key = ['model', 'instance_idx', 'method', 'allocation']
    dups = df_all.duplicated(subset=dup_key, keep=False)
    if dups.any():
        log(f"{int(dups.sum())} lignes dupliquees sur {dup_key} :")
        print(df_all.loc[dups, dup_key].to_string())
        die("doublons detectes, verifiez les fichiers sources.")

    # ── Resume de controle ────────────────────────────────────────────────
    print()
    log(f"Total consolide : {len(df_all)} lignes "
        f"(attendu 360 si le run A1/A3 est complet)")
    log("Repartition par (allocation, method) :")
    print(df_all.groupby(['allocation', 'method']).size()
          .unstack(fill_value=0).to_string())

    print()
    log("Part d'exploration realisee par allocation (abc_shap seul) :")
    summary = (df_all.loc[df_all['method'] == 'abc_shap']
               .groupby('allocation')['explor_share_realized']
               .agg(['mean', 'min', 'max', 'count'])
               .round(4))
    summary['part_estimation'] = (1.0 - summary['mean']).round(4)
    print(summary.to_string())

    expected = {'A0': 0.9922, 'A1': 0.2496, 'A2': 0.4992, 'A3': 0.7488}
    for al, exp in expected.items():
        if al in summary.index and abs(summary.loc[al, 'mean'] - exp) > 1e-3:
            die(f"{al} : part d'exploration moyenne {summary.loc[al, 'mean']} "
                f"!= attendu {exp}.")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    df_all.to_csv(out, index=False)
    print()
    log(f"Fichier consolide ecrit : {out}")


if __name__ == '__main__':
    main()