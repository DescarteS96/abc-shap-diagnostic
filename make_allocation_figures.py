"""
make_allocation_figures.py
===========================
Produit les deux graphiques de l'ETAPE 2.7 de la revision V6.

Graphique 1 — part du budget d'exploration vs stabilite ordinale (Spearman)
Graphique 2 — part du budget d'exploration vs MSE (ratio relatif a B2)

Entree :
    results_allocation_consolidated_california_N5000.csv
    (produit par consolidate_allocation.py)

Sorties (dans --outdir, par defaut 'sekiller') :
    fig5_5_allocation_stability.png / .pdf
    fig5_6_allocation_mse.png / .pdf

MISE EN FORME (alignee sur les figures existantes de la these) :
    - Aucun titre dans l'image : la legende de figure est en LaTeX, sous
      le graphique.
    - Legende des modeles horizontale, centree SOUS le graphique.
    - Etiquettes A0..A3 placees au-dessus du sommet de moustache le plus
      haut de chaque abscisse, avec elargissement du haut de l'axe Y pour
      qu'elles ne chevauchent ni les points ni les barres d'erreur.

Conventions :
    - Abscisse : part du budget consacree a l'EXPLORATION, en pourcentage.
      Utilisez --xaxis estimation pour inverser.
    - Ridge (California) est exclu : ordre invariant a la politique.
    - Barres d'erreur : IC bootstrap 95 % sur la moyenne des 10 instances.

Usage :
    python make_allocation_figures.py \
        --input results_allocation_consolidated_california_N5000.csv \
        --outdir sekiller
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("[figures] ERREUR — matplotlib n'est pas installe. "
          "Lancez : pip install matplotlib")
    sys.exit(1)

ALLOC_ORDER = ['A0', 'A1', 'A2', 'A3']
INFORMATIVE_MODELS = ['rf', 'mlp']
MODEL_LABEL = {'rf': 'Random Forest', 'mlp': 'MLP'}
MODEL_STYLE = {'rf': dict(marker='o', linestyle='-', color='#1f77b4'),
               'mlp': dict(marker='s', linestyle='--', color='#d62728')}


def log(m):
    print(f"[figures] {m}")


def die(m):
    print(f"[figures] ERREUR — {m}")
    sys.exit(1)


def boot_ci(v, n_boot=2000, seed=42):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(v), size=(n_boot, len(v)))
    b = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(b, 2.5)), \
           float(np.percentile(b, 97.5))


def collect(df, metric, xaxis):
    """Retourne {model: (x, mean, lo, hi, labels)} trie par x croissant."""
    a = df[(df['method'] == 'abc_shap') &
           (df['model'].isin(INFORMATIVE_MODELS))]
    out = {}
    for m in INFORMATIVE_MODELS:
        xs, ms, los, his, labs = [], [], [], [], []
        for al in ALLOC_ORDER:
            s = a[(a['allocation'] == al) & (a['model'] == m)]
            if len(s) == 0:
                continue
            N = float(s['n_budget'].iloc[0])
            share = (s['n_exploration'].mean() / N if xaxis == 'exploration'
                     else s['n_estimation'].mean() / N)
            mean, lo, hi = boot_ci(s[metric].dropna().values)
            xs.append(100.0 * share)
            ms.append(mean); los.append(lo); his.append(hi); labs.append(al)
        order = np.argsort(xs)
        out[m] = (np.array(xs)[order], np.array(ms)[order],
                  np.array(los)[order], np.array(his)[order],
                  [labs[i] for i in order])
    return out


def make_figure(data, ylabel, outstem, outdir, xaxis, xscale, logy=False):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    # ── Courbes avec barres d'erreur ─────────────────────────────────────
    for m in INFORMATIVE_MODELS:
        if m not in data:
            continue
        x, mean, lo, hi, _ = data[m]
        if len(x) == 0:
            continue
        yerr = np.vstack([mean - lo, hi - mean])
        ax.errorbar(x, mean, yerr=yerr, capsize=4, linewidth=1.6,
                    markersize=6, label=MODEL_LABEL[m], **MODEL_STYLE[m])

    # ── Etendue verticale de toutes les series (points + moustaches) ─────
    tops, bottoms = [], []
    for m in INFORMATIVE_MODELS:
        if m in data and len(data[m][0]):
            tops.append(float(np.nanmax(data[m][3])))     # hi
            bottoms.append(float(np.nanmin(data[m][2])))  # lo
    if not tops:
        die("aucune donnee a tracer.")
    ymax_data, ymin_data = max(tops), min(bottoms)

    # Marge haute pour loger les etiquettes A0..A3 sans chevauchement.
    if logy:
        ax.set_yscale('log')
        ax.set_ylim(ymin_data * 0.80, ymax_data * 1.55)
    else:
        span = ymax_data - ymin_data
        if span <= 0:
            span = max(abs(ymax_data), 1.0) * 0.1
        ax.set_ylim(ymin_data - 0.10 * span, ymax_data + 0.22 * span)

    # ── Etiquettes A0..A3 : une par abscisse, au-dessus du sommet de
    #    moustache le plus haut tous modeles confondus ───────────────────
    per_x = {}
    for m in INFORMATIVE_MODELS:
        if m not in data:
            continue
        x, mean, lo, hi, labs = data[m]
        for xi, hii, lab in zip(x, hi, labs):
            key = round(float(xi), 3)
            prev = per_x.get(key)
            per_x[key] = (lab, float(hii) if prev is None
                          else max(prev[1], float(hii)))

    for xi, (lab, top) in sorted(per_x.items()):
        ax.annotate(lab, (xi, top), textcoords="offset points",
                    xytext=(0, 9), ha='center', va='bottom',
                    fontsize=9.5, color='#333333', fontweight='bold',
                    annotation_clip=False)

    # ── Axes et grille ───────────────────────────────────────────────────
    xlab = ("Exploration budget share (%)" if xaxis == 'exploration'
            else "Final-estimation budget share (%)")
    ax.set_xlabel(xlab, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if xscale == 'log':
        ax.set_xscale('log')
    ax.grid(True, alpha=0.28, linestyle=':')
    ax.margins(x=0.10)

    # Ne garder que les axes gauche et bas
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Legende horizontale, centree sous le graphique ──────────────────
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=max(1, len(labels)),
               frameon=True, fancybox=False, edgecolor='#999999',
               fontsize=10, bbox_to_anchor=(0.5, -0.01),
               columnspacing=2.4, handlelength=2.6)

    fig.tight_layout(rect=[0, 0.09, 1, 1])

    for ext in ('png', 'pdf'):
        p = os.path.join(outdir, f"{outstem}.{ext}")
        fig.savefig(p, dpi=200 if ext == 'png' else None,
                    bbox_inches='tight', facecolor='white')
        log(f"ecrit : {p}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--outdir', default='sekiller')
    ap.add_argument('--prefix1', default='fig5_5_allocation_stability')
    ap.add_argument('--prefix2', default='fig5_6_allocation_mse')
    ap.add_argument('--xaxis', choices=['exploration', 'estimation'],
                    default='exploration')
    ap.add_argument('--xscale', choices=['linear', 'log'], default='linear')
    ap.add_argument('--logy-mse', action='store_true',
                    help="Echelle log sur l'axe Y du graphique MSE")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        die(f"fichier introuvable : {args.input}")
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    log(f"{len(df)} lignes chargees")
    found = sorted(set(df['allocation'].astype(str)))
    log(f"allocations presentes : {found}")
    if len(found) < 2:
        die("moins de deux allocations : impossible de tracer une courbe.")

    d1 = collect(df, 'spearman_mean', args.xaxis)
    make_figure(d1, ylabel="Mean Spearman correlation",
                outstem=args.prefix1, outdir=args.outdir,
                xaxis=args.xaxis, xscale=args.xscale)

    d2 = collect(df, 'mse_ratio_b2', args.xaxis)
    make_figure(d2, ylabel="MSE ratio relative to B2",
                outstem=args.prefix2, outdir=args.outdir,
                xaxis=args.xaxis, xscale=args.xscale, logy=args.logy_mse)

    print()
    log("Valeurs tracees (moyenne [IC 95 %]) :")
    for name, d in [('Spearman', d1), ('MSE ratio vs B2', d2)]:
        print(f"\n  {name}")
        for m in INFORMATIVE_MODELS:
            if m not in d:
                continue
            x, mean, lo, hi, labs = d[m]
            for xi, mi, li, hi_, lab in zip(x, mean, lo, hi, labs):
                print(f"    {MODEL_LABEL[m]:<14} {lab}  x={xi:6.1f}%  "
                      f"{mi:8.4f}  [{li:.4f}, {hi_:.4f}]")

    print()
    log("Pour LaTeX :")
    print(f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.85\\textwidth]{{{args.outdir}/{args.prefix1}.png}}
\\caption{{Exploration budget share versus mean Spearman correlation, with 95\\%
bootstrap confidence intervals. California Housing, $N = 5000$, $K = 30$, ten
test instances per configuration. Ridge is excluded (attribution ordering
invariant to the sampling policy).}}
\\label{{fig:alloc-stability}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.85\\textwidth]{{{args.outdir}/{args.prefix2}.png}}
\\caption{{Exploration budget share versus MSE ratio relative to the B2
baseline, with 95\\% bootstrap confidence intervals. Same configurations as
Figure~\\ref{{fig:alloc-stability}}.}}
\\label{{fig:alloc-mse}}
\\end{{figure}}
""")


if __name__ == '__main__':
    main()