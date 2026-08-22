# Why Adaptive Coalition Sampling Fails to Improve Kernel SHAP Stability Under Finite Budgets

**Authors**
[René Fassou Ballamou](https://github.com/DescarteS96) ·
[İsmail Yenilmez](https://github.com/IsmailYenilmez)
Department of Statistics, Eskisehir Technical University, Eskişehir, Türkiye

Code and data accompanying the paper *"Why Adaptive Coalition Sampling Fails to
Improve Kernel SHAP Stability Under Finite Budgets: An Empirical Diagnostic
Study"*.

This repository contains everything needed to reproduce the four tables, the
three figures and the run-to-run dispersion analysis reported in the paper. It
does not contain material specific to the underlying master's thesis.

## What the study does

ABC-SHAP is an instance-specific adaptive coalition sampler for Kernel SHAP. It
searches the probability simplex of coalition sizes with an Artificial Bee
Colony metaheuristic, guided by an internal split-half consistency criterion
computable within the exploration budget. It is introduced as an experimental
instrument, not as a method proposed for adoption.

The study reports a negative result and separates its two candidate causes
experimentally: the share of the evaluation budget consumed by the search, and
the informational content of the criterion that guides it.

## Repository layout

| file | role |
| --- | --- |
| `config.py` | experimental constants, budget allocation rule |
| `abc_shap_en.py` | ABC-SHAP: policy search and split-half fitness |
| `baseline_b1_uniform.py` | B1 — uniform over coalition sizes |
| `baseline_b2_stratified.py` | B2 — size-stratified, equal allocation |
| `baseline_b3_importance_sampling.py` | B3 — kernel-proportional importance sampling |
| `baseline_b4_antithetic.py` | B4 — antithetic pairing |
| `metrics/` | stability, accuracy, bootstrap and test helpers |
| `exp_runner.py` | main grid and allocation experiment |
| `test_j_pi_true_correlation.py` | fitness-validity analysis |
| `abc_vs_dirichlet.py` | control: colony search vs random policy draw |
| `consolidate_allocation.py` | merges the four allocation runs |
| `make_article_tables.py` | Tables 1–4, Word output |
| `make_algorithm_box.py` | Algorithm 1, Word output |
| `make_allocation_figures.py` | Figures 1 and 2 |
| `make_fitness_validity.py` | Figure 3 |
| `check_sigma_rho.py` | run-to-run dispersion (Section 5.1) |
| `results/` | raw experimental output (CSV) |
| `figures/` | figures as they appear in the paper |

## Installation

```bash
git clone https://github.com/DescarteS96/abc-shap-diagnostic.git
cd abc-shap-diagnostic
pip install -r requirements.txt
```

Python 3.10 or later. The three datasets are downloaded automatically on first
use: California Housing and Adult Income through `scikit-learn`, Credit Default
through OpenML.

## Reproducing the results without re-running the experiments

Every table and figure in the paper is derived from the CSV files in
`results/`. The commands below regenerate them in a few seconds.

### Tables 1 to 4

```bash
python make_article_tables.py \
  --main results/results_california_N1000.csv \
         results/results_california_N5000.csv \
         results/results_adult_N1000.csv \
         results/results_adult_N5000.csv \
         results/results_credit_N1000.csv \
         results/results_credit_N5000.csv \
  --allocation results/results_allocation_consolidated_california_N5000.csv \
  --fitness results/j_pi_true_correlation.csv \
  --dirichlet results/abc_vs_dirichlet_results.csv \
  --outdir output
```

The script prints the full test detail to the console, so every number in the
tables can be checked against its source. Expect
`vs B2/B3 : 12/12 significativement defavorables` for Table 1 and a pooled
mean of `-0.0122` with a bootstrap interval of `[-0.0191, -0.0056]` for
Table 4.

### Figures 1 and 2

```bash
python make_allocation_figures.py \
  --input results/results_allocation_consolidated_california_N5000.csv \
  --outdir output
```

### Figure 3

The `--zscore-x` flag is required. Without it the three panels show a Simpson
effect caused by superimposing datasets whose fitness scales differ.

```bash
python make_fitness_validity.py \
  --input results/j_pi_true_correlation.csv \
  --figdir output --outdir output --zscore-x
```

### Run-to-run dispersion (Section 5.1)

This script recomputes the standard deviation from the K = 30 stored per-run
values and verifies that it matches the stored column before reporting
anything.

```bash
python check_sigma_rho.py \
  results/results_california_N1000.csv results/results_california_N5000.csv \
  results/results_adult_N1000.csv results/results_adult_N5000.csv \
  results/results_credit_N1000.csv results/results_credit_N5000.csv
```

Expected aggregate: `0.0843` for ABC-SHAP against `0.0558`, `0.0319`, `0.0222`
and `0.0267` for B1 to B4.

### Algorithm 1

```bash
python make_algorithm_box.py --outdir output
```

## Re-running the experiments from scratch

These commands regenerate the CSV files in `results/`. They are considerably
more expensive: the full set takes on the order of a hundred CPU-hours, almost
all of it spent on the random-forest configurations.

### Main grid

One command per (dataset, budget) cell.

```bash
python exp_runner.py --dataset california --budget 1000
python exp_runner.py --dataset california --budget 5000
python exp_runner.py --dataset adult      --budget 1000
python exp_runner.py --dataset adult      --budget 5000
python exp_runner.py --dataset credit     --budget 1000
python exp_runner.py --dataset credit     --budget 5000
```

### Allocation experiment

The A1 and A3 runs, then consolidation with A0 and A2.

```bash
python exp_runner.py --dataset california --budget 5000 \
       --explor-share 0.25 0.75 --abc-only

python consolidate_allocation.py \
  --a0  <original-allocation grid for california N=5000> \
  --a2  results/results_california_N5000.csv \
  --a13 results/results_allocation_california_N5000.csv \
  --out results/results_allocation_consolidated_california_N5000.csv
```

A0 is the initial allocation, which assigns 99.2% of the budget to
exploration. It comes from the grid produced before the allocation rule was
revised. `results/results_allocation_consolidated_california_N5000.csv` is
provided so that Table 2 can be reproduced without it.

### Fitness-validity analysis

Thirty candidate policies per dataset, K = 30 full estimations each.

```bash
python test_j_pi_true_correlation.py
```

### Dirichlet control

Four estimation procedures per instance, K = 30. The script saves
incrementally and resumes where it stopped.

```bash
python abc_vs_dirichlet.py
```

## Notes on the data files

Each row of a main-grid CSV is one (dataset, model, method, instance) cell.
`spearman_mean`, `kendall_mean`, `top3_mean`, `top5_mean` and `mse` are means
over K = 30 independent repetitions; `spearman_std` is the corresponding
run-to-run standard deviation; the `*_per_run` columns hold the K raw values,
so every aggregate can be recomputed from them. `n_exploration` and
`n_estimation` record how the total budget was actually split.

Ridge regression on California Housing is degenerate on the ordinal axis: as a
linear model on a regression task, its attribution ordering does not depend on
the sampling policy, and all five methods return a Spearman correlation of
exactly 1.0000 with zero variance. The analysis scripts exclude it from
rank-based comparisons and retain it for the MSE ratio.

## Scope

The paper evaluates three datasets at two budget levels. Results at N = 10000
exist for some cells but fall outside the reported scope and are not included
here. The Dirichlet control covers all three datasets; the block-level detail
is in `results/abc_vs_dirichlet_results.csv`.

## Citation

```bibtex
@article{ballamou2026abcshap,
  title   = {Why Adaptive Coalition Sampling Fails to Improve Kernel SHAP
             Stability Under Finite Budgets: An Empirical Diagnostic Study},
  author  = {Ballamou, Rene Fassou and Yenilmez, {\.I}smail},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code is released under the MIT License. The CSV files in `results/` are
released under CC BY 4.0.
