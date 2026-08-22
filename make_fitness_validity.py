"""
make_fitness_validity.py
=========================
Produit les livrables des ETAPES 2.9 a 2.11 de la revision V6 :

  - Le nuage de points aptitude interne J(pi) vs vraie stabilite ordinale,
    une facette par metrique de stabilite (Spearman, Kendall, top-3),
    couleurs par dataset, marqueurs par origine de la politique
    (tirage aleatoire vs politique visitee par la colonie ABC).
  - Le tableau recapitulatif du 2.10 : Pearson, Spearman, Kendall et
    p-values pour chaque relation, par dataset puis en agrege.

Entree :
    j_pi_true_correlation.csv
    (produit par test_j_pi_true_correlation.py — colonnes dataset, M,
     policy_idx, source, J_pi, true_stability_spearman,
     true_stability_kendall, true_stability_top3)

Sorties :
    --figdir  : fig5_7_fitness_validity.png / .pdf
    --outdir  : table_fitness_validity.csv / .tex

NOTE METHODOLOGIQUE — l'agregat "POOLED"
    Les echelles de J(pi) ne sont pas comparables entre dimensionnalites
    (M = 8, 14, 23) : une correlation calculee sur les valeurs brutes
    serait dominee par l'effet dataset. J(pi) est donc centre-reduit a
    l'interieur de chaque dataset avant l'agregation, et les mesures de
    stabilite le sont egalement pour la meme raison. L'agregat porte
    ainsi sur la relation INTRA-dataset, avec n = 90.

Usage :
    python make_fitness_validity.py \
        --input j_pi_true_correlation.csv \
        --figdir sekiller \
        --outdir tables
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("[fitness] ERREUR — matplotlib n'est pas installe. "
          "Lancez : pip install matplotlib")
    sys.exit(1)

DATASETS = ['california', 'adult', 'credit']
DATASET_LABEL = {'california': 'California ($M=8$)',
                 'adult': 'Adult ($M=14$)',
                 'credit': 'Credit Default ($M=23$)'}
DATASET_LABEL_PLAIN = {'california': 'California (M=8)',
                       'adult': 'Adult (M=14)',
                       'credit': 'Credit Default (M=23)'}
DATASET_COLOR = {'california': '#1f77b4',
                 'adult': '#2ca02c',
                 'credit': '#d62728'}
SOURCE_MARKER = {'random': 'o', 'abc_visited': '^'}
SOURCE_LABEL = {'random': 'Random policy',
                'abc_visited': 'ABC-visited policy'}

TARGETS = [
    ('true_stability_spearman', 'Spearman', 'Split-half vs Spearman'),
    ('true_stability_kendall',  'Kendall',  'Split-half vs Kendall'),
    ('true_stability_top3',     'Top-3',    'Split-half vs top-3'),
]


def log(m):
    print(f"[fitness] {m}")


def die(m):
    print(f"[fitness] ERREUR — {m}")
    sys.exit(1)


def corr_triplet(x, y):
    """Retourne (r_pearson, p, rho_spearman, p, tau_kendall, p)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return (np.nan,) * 6
    rp, pp = pearsonr(x, y)
    rs, ps = spearmanr(x, y)
    rk, pk = kendalltau(x, y)
    return float(rp), float(pp), float(rs), float(ps), float(rk), float(pk)


def zscore_within(df, cols, by='dataset'):
    """Centre-reduit chaque colonne a l'interieur de chaque dataset."""
    out = df.copy()
    for c in cols:
        out[c + '_z'] = out.groupby(by)[c].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0
            else 0.0)
    return out


# ══════════════════════════════════════════════════════════════════════════
# TABLEAU — ETAPE 2.10
# ══════════════════════════════════════════════════════════════════════════

def build_table(df):
    rows = []

    for ds in DATASETS:
        sub = df[df['dataset'] == ds]
        if len(sub) < 5:
            log(f"{ds} : {len(sub)} observations, ignore.")
            continue
        for col, short, label in TARGETS:
            rp, pp, rs, ps, rk, pk = corr_triplet(sub['J_pi'], sub[col])
            rows.append({
                'Scope': DATASET_LABEL_PLAIN[ds],
                'Relation': label,
                'n': int(len(sub)),
                'Pearson r': rp, 'p (Pearson)': pp,
                'Spearman rho': rs, 'p (Spearman)': ps,
                'Kendall tau': rk, 'p (Kendall)': pk,
            })

    # Agregat : tout centre-reduit a l'interieur de chaque dataset.
    cols = ['J_pi'] + [c for c, _, _ in TARGETS]
    dz = zscore_within(df, cols)
    for col, short, label in TARGETS:
        rp, pp, rs, ps, rk, pk = corr_triplet(dz['J_pi_z'], dz[col + '_z'])
        rows.append({
            'Scope': 'Pooled (within-dataset z-scores)',
            'Relation': label,
            'n': int(len(dz)),
            'Pearson r': rp, 'p (Pearson)': pp,
            'Spearman rho': rs, 'p (Spearman)': ps,
            'Kendall tau': rk, 'p (Kendall)': pk,
        })

    return pd.DataFrame(rows)


def tex_table(df, caption, label, note=None):
    body = df.to_latex(index=False, escape=True, float_format="%.4f",
                       na_rep='--')
    out = ["\\begin{table}[htbp]", "\\centering", "\\small",
           "\\begin{threeparttable}", f"\\caption{{{caption}}}",
           f"\\label{{{label}}}", body]
    if note:
        out += ["\\begin{tablenotes}[flushleft]\\footnotesize",
                f"\\item {note}", "\\end{tablenotes}"]
    out += ["\\end{threeparttable}", "\\end{table}", ""]
    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE — ETAPE 2.10
# ══════════════════════════════════════════════════════════════════════════

def make_scatter(df, figdir, stem, zscore_x=False):
    """Trois facettes cote a cote : J(pi) vs chaque mesure de stabilite."""
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))

    xcol = 'J_pi'
    xlabel = "Internal split-half fitness $J(\\pi)$"
    if zscore_x:
        df = zscore_within(df, ['J_pi'])
        xcol = 'J_pi_z'
        xlabel = "Internal split-half fitness $J(\\pi)$ (within-dataset z-score)"

    for ax, (col, short, _) in zip(axes, TARGETS):
        for ds in DATASETS:
            for src in ['random', 'abc_visited']:
                s = df[(df['dataset'] == ds) & (df['source'] == src)]
                if len(s) == 0:
                    continue
                ax.scatter(s[xcol], s[col],
                           s=34, alpha=0.80,
                           color=DATASET_COLOR[ds],
                           marker=SOURCE_MARKER[src],
                           edgecolors='white', linewidths=0.5,
                           label=None)
        # droite de regression sur l'ensemble des points de la facette
        x = df[xcol].values.astype(float)
        y = df[col].values.astype(float)
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() >= 3 and np.std(x[ok]) > 0:
            b, a = np.polyfit(x[ok], y[ok], 1)
            xx = np.linspace(np.min(x[ok]), np.max(x[ok]), 50)
            ax.plot(xx, a + b * xx, color='#555555', linewidth=1.2,
                    linestyle='--', zorder=1)
            rp, pp, _, _, _, _ = corr_triplet(x[ok], y[ok])
            ax.set_title(f"$r = {rp:.3f}$  ($p = {pp:.2f}$)", fontsize=10.5)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(f"True inter-run stability ({short})", fontsize=10)
        ax.grid(True, alpha=0.28, linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legende unique, horizontale, sous les trois facettes
    handles, labels = [], []
    for ds in DATASETS:
        handles.append(plt.Line2D([], [], marker='o', linestyle='none',
                                  color=DATASET_COLOR[ds], markersize=7))
        labels.append(DATASET_LABEL_PLAIN[ds])
    for src in ['random', 'abc_visited']:
        handles.append(plt.Line2D([], [], marker=SOURCE_MARKER[src],
                                  linestyle='none', color='#555555',
                                  markersize=7))
        labels.append(SOURCE_LABEL[src])

    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               frameon=True, fancybox=False, edgecolor='#999999',
               fontsize=9.5, bbox_to_anchor=(0.5, -0.012),
               columnspacing=1.9, handlelength=1.6)

    fig.tight_layout(rect=[0, 0.10, 1, 1])

    for ext in ('png', 'pdf'):
        p = os.path.join(figdir, f"{stem}.{ext}")
        fig.savefig(p, dpi=200 if ext == 'png' else None,
                    bbox_inches='tight', facecolor='white')
        log(f"ecrit : {p}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True,
                    help="j_pi_true_correlation.csv")
    ap.add_argument('--figdir', default='sekiller')
    ap.add_argument('--outdir', default='tables')
    ap.add_argument('--figname', default='fig5_7_fitness_validity')
    ap.add_argument('--zscore-x', action='store_true',
                    help="Trace J(pi) centre-reduit par dataset "
                         "(recommande si les echelles de J different)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        die(f"fichier introuvable : {args.input}")
    os.makedirs(args.figdir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    log(f"{len(df)} lignes chargees depuis {args.input}")

    required = ['dataset', 'source', 'J_pi'] + [c for c, _, _ in TARGETS]
    for c in required:
        if c not in df.columns:
            die(f"colonne '{c}' absente. Relancez "
                f"test_j_pi_true_correlation.py dans sa version corrigee "
                f"(celle qui produit true_stability_top3).")

    log(f"datasets : {sorted(df['dataset'].unique())}")
    log(f"origines des politiques : "
        f"{df.groupby('source').size().to_dict()}")

    # ── Tableau ──────────────────────────────────────────────────────────
    t = build_table(df)
    print()
    log("ETAPE 2.10 — tableau recapitulatif :")
    print(t.round(4).to_string(index=False))

    csv_path = os.path.join(args.outdir, 'table_fitness_validity.csv')
    tex_path = os.path.join(args.outdir, 'table_fitness_validity.tex')
    t.to_csv(csv_path, index=False)
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_table(
            t,
            caption="Relationship between the internal split-half fitness "
                    "$J(\\pi)$ and true inter-run ordinal stability, measured "
                    "independently under the same fixed policy",
            label="tab:fitness_validity",
            note="Thirty policies per dataset (ten drawn uniformly at random "
                 "on the simplex, twenty visited during an actual ABC search). "
                 "True stability is the mean pairwise agreement across $K = 30$ "
                 "independent full estimations under the same fixed policy. "
                 "$N_{\\text{inner}}$ is derived per dataset from the "
                 "allocation rule at $N = 5000$ (32, 56 and 92 respectively). "
                 "The pooled rows standardize $J(\\pi)$ and the stability "
                 "measures within each dataset, since their scales are not "
                 "comparable across dimensionalities."))
    log(f"ecrit : {csv_path}")
    log(f"ecrit : {tex_path}")

    # ── Figure ───────────────────────────────────────────────────────────
    print()
    make_scatter(df, args.figdir, args.figname, zscore_x=args.zscore_x)

    # ── Synthese ─────────────────────────────────────────────────────────
    print()
    log("SYNTHESE :")
    sig = t[(t['p (Pearson)'] <= 0.05) | (t['p (Spearman)'] <= 0.05) |
            (t['p (Kendall)'] <= 0.05)]
    log(f"  {len(sig)}/{len(t)} relations atteignent p <= 0.05 "
        f"sur au moins un des trois coefficients")
    if len(sig):
        print(sig[['Scope', 'Relation', 'Pearson r', 'p (Pearson)']]
              .round(4).to_string(index=False))
    amax = t[['Pearson r', 'Spearman rho', 'Kendall tau']].abs().max().max()
    log(f"  coefficient absolu maximal observe : {amax:.4f}")

    print()
    log("Pour LaTeX :")
    print(f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=\\textwidth]{{{args.figdir}/{args.figname}.png}}
\\caption{{Internal split-half fitness $J(\\pi)$ versus true inter-run ordinal
stability, measured independently under the same fixed policy. Each point is
one sampling policy; colours denote datasets and markers denote whether the
policy was drawn at random or visited during an actual ABC search. Dashed
lines are ordinary least-squares fits over all points in the panel, with the
corresponding Pearson coefficient reported above each panel.}}
\\label{{fig:fitness-validity}}
\\end{{figure}}
""")


if __name__ == '__main__':
    main()