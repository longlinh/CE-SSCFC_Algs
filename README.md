# CE-SSCFC — reference implementation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22305371.svg)](https://doi.org/10.5281/zenodo.22305371)

**Archived on Zenodo:** concept DOI [10.5281/zenodo.22305371](https://doi.org/10.5281/zenodo.22305371) (all versions); version 1.0.0 DOI [10.5281/zenodo.22305372](https://doi.org/10.5281/zenodo.22305372). Licence MIT.

Companion source code for the manuscript

> **Classifier-embedded semi-supervised collaborative fuzzy clustering with
> validation-gated knowledge transfer for land-cover mapping across regions.**
> Xuan Hoang Nguyen, Quoc Binh Vuong, Viet Ha Tran, Chien Thang Nguyen, Thang Long Hoang.
> Submitted to *Neurocomputing* (under review, 2026).

A compact, NumPy-only implementation written to be read next to the equations of the
paper: one module, plain functions, no class hierarchy. The repository publishes the
**algorithm**, a self-contained synthetic multi-datasite demonstration, tests (including
numerical parity with the implementation used for the paper's experiments) and an example
that runs the paper's protocol on the companion Landsat-8 dataset.

**CE-SSCFC** is a semi-supervised fuzzy c-means in which a ridge classifier `W` on a fixed
quadratic feature map `φ` is an *optimisation variable* of the clustering objective, and the
only channel of collaboration between datasites is the exchange of per-class sufficient
statistics of the labelled pixels:

```
J_CE(U,V,W) = Σ_i Σ_k u_ik^m [ ‖x_i − v_k‖² + α ‖Wᵀφ_i − e_k‖² ] + μ Σ_{i∈L} ‖Wᵀφ_i − e_{y_i}‖² + λ ‖W‖²_F
J_col^(s)   = J_CE^(s) + ν_s Σ_{r≠s} Σ_k t_sk^(r) [ tr(Wᵀ G_k^(r) W) − 2 (Wᵀ h_k^(r))_k + n_k^(r) ]
                             G_k = Σ φφᵀ, h_k = Σ φ, n_k = |L_k|   (shipped only when n_k ≥ n_min)
U-step  u_ik ∝ D_ik^{−1/(m−1)},  D_ik = ‖x_i − v_k‖² + α ‖Wᵀφ_i − e_k‖²       closed form, u ≥ 0 automatically
V-step  v_k  = Σ_i u_ik^m x_i / Σ_i u_ik^m
W-step  (α ΦᵀSΦ + μ Φ_LᵀΦ_L + ν Σ t G_k + λI) W = α ΦᵀU^m + μ Φ_LᵀY + ν Σ t h_k e_kᵀ,  S = diag(Σ_k u_ik^m)
```

Every block is strictly convex, so the alternating optimisation is closed-form for any
`m > 1` and `J` is monotone non-increasing (Theorem 1). The two-phase protocol of
Algorithm 1 — Phase 1 alone, a **validation gate** that tries every received class block
with one `W`-step, Phase 2 with **backtracking** on the local validation risk `R_val` —
guarantees `R_val(Phase 2) ≤ R_val(Phase 1)` at every datasite by construction
(Theorem 2). Datasites that receive no block keep their Phase-1 solution unchanged.

## Install

Python 3.9 or newer, three pure-Python-wheel dependencies (no GPU):

```bash
pip install -r requirements.txt        # numpy, scipy, scikit-learn
python demo.py                         # three synthetic datasites, < 2 s, no data needed
pytest tests/ -q                       # 9 tests, ~2 s
```

`demo.py` prints, deterministically:

```
n_min = 35 (privacy threshold); gates opened: 5/30 -> site0<-site1:class2, site0<-site1:class3, ...
Phase 2: 3 rounds, backtracks [0, 0, 0], nu_eff [216.847, 0.0, 0.0]

datasite   R_val P1 -> P2        test ACC P1 -> P2   (F1 P2, NMI P2)
   0       0.0401 -> 0.0000        0.6476 -> 0.6773   (0.4731, 0.6137)
   1       0.2033 -> 0.2033        0.8118 -> 0.8118   (0.7467, 0.6182)
   2       0.1449 -> 0.1449        0.6274 -> 0.6274   (0.5879, 0.5148)
  mean                                   0.6956 -> 0.7055
```

This is the story of the paper in miniature: the label-poor datasite 0 opens gates towards
the two label-rich datasites, its validation risk and test accuracy improve, and the
label-rich datasites — for which no block passes the gate — are left exactly unchanged.

## Quick start

```python
import numpy as np
from cesscfc import ce_sscfc, region_blocked_split
from metrics import evaluate

# Xs[s]: (N_s, d) pixels of datasite s (standardised per datasite), row-major H_s x W_s
# ys[s]: (N_s,) partial labels, -1 = unlabelled, 0..C-1 = class (class-pure labelled regions)
val_idx = [region_blocked_split(y, shape, frac=0.2, seed=42)[1] for y, shape in zip(ys, shapes)]
res = ce_sscfc(Xs, ys, val_idx, n_clusters=C, m=3.0, alpha=32.0, mu=16.0, rho=0.1,
               eps_gate=0.005, seed=42)
res["labels"][s]          # (N_s,) hard labels = argmax_k u_ik after Phase 2
res["labels_phase1"][s]   # the same after Phase 1 (independent fit)
res["U"], res["V"], res["W"]                  # memberships (N_s,C), centroids (C,d), classifier (D,C)
res["gates"]              # (S, S, C): gates[s, r, k] = 1 if datasite s accepts class k from r
res["risk_phase1"], res["risk_phase2"]        # R_val per datasite (never increases)
res["nu_effective"], res["n_backtracks"], res["n_rounds"], res["n_min"]
print(evaluate(y_true[test], res["labels"][s][test], C))   # {'acc', 'f1', 'nmi'} (Hungarian-matched)
```

`ce_ssfcm(X, y, C, ...)` runs the single-datasite special case (Phase 1 only).
The validation pixels `val_idx[s]` must be whole labelled regions (not random pixels):
a random-pixel split leaks spatial context into the gate and switches it off
(Section 4.8 of the paper).

## Choosing the parameters

| Parameter | Paper | Meaning |
|---|---|---|
| `m` | 3 (VN3), 2 (ES2) | fuzzifier; also the confidence temperature of the pseudo-labels entering the `W`-step |
| `alpha` | 32 (VN3), 8 (ES2) | weight of the classifier term in `D_ik`; chosen on the validation accuracy |
| `mu` | 16 | supervision weight, scaled per datasite as `μ_s = μ·N_s/|L_s|` |
| `lam_ratio` | 1e-3 | `λ_s = lam_ratio · μ_s` (ridge) |
| `rho` | 0.1 | `ν_s = ρ·μ_s`: one peer labelled pixel weighs `ρ` of a local one in the `W`-step |
| `eps_gate` | 0.005 | validation-risk margin of the gate |
| `n_min` | `'auto'` (= 35 for d = 4) | privacy threshold: blocks with fewer labelled pixels are withheld |
| `max_iter`, `tol` | 60, 1e-7 | Phase 1 (`T_max`, `ε`) |
| `max_rounds`, `inner_iter`, `tau_min`, `tol_outer` | 10, 20, 1e-3, 1e-5 | Phase 2 (`T_2`, `T_in`, `τ_min`, `ε_out`) |

## API

| Function | Purpose |
|---|---|
| `phi_map(X, kind)` → `Φ` | fixed feature map (`'quad'`: 1, x, x_a x_b; `'linear'`: 1, x) |
| `label_stats(Φ, y, C, n_min)` → `{k: (G_k, h_k, n_k)}` | the per-class sufficient statistics a datasite ships |
| `fit_phase1(X, y_fit, C, m, alpha, mu, lam, ...)` → state | Phase 1 at one datasite (ridge-initialised `W`, class-mean `V`) |
| `gate_candidates(state, s, stats_all, nu_s, risk_fn, eps_gate, alpha, m)` → `(gates, Δrisk)` | validation gate: one `W`-step per received block |
| `fit_phase2(states, stats_all, gates, nus, risk_fn, alpha, m, ...)` → diagnostics | collaborative rounds with backtracking on `R_val` |
| `ce_sscfc(Xs, ys, val_idx, C, ...)` → dict | the full method (Algorithm 1) |
| `ce_ssfcm(X, y, C, ...)` → dict | single-datasite special case |
| `prior_weights`, `validation_risk`, `region_blocked_split` | `w_k = p̂_k / q_k`, weighted 0-1 risk of `argmax Wᵀφ`, whole-region validation split |
| `u_step`, `v_step`, `w_step`, `big_d`, `objective`, `peer_objective`, `alternating_optimisation` | the block updates and objectives |
| `metrics.evaluate / accuracy / macro_f1 / nmi` | Hungarian-matched evaluation |

## Running on the companion Landsat-8 dataset

The six Landsat-8 scenes of the paper (Hanoi, Thanh Hoa, Ho Chi Minh City and the held-out
confirmatory site Hai Phong, added in dataset v1.1.0; Valencia, Alicante) with their
WorldCover-derived reference maps are released separately:
<https://github.com/longlinh/CE-SSCFC_Dataset>, archived at
[doi:10.5281/zenodo.22304965](https://doi.org/10.5281/zenodo.22304965) (CC BY 4.0).

```bash
pip install rasterio
python examples/run_landsat.py --data-root /path/to/CE-SSCFC_Dataset --set vn3          # m=3, α=32
python examples/run_landsat.py --data-root /path/to/CE-SSCFC_Dataset --set es2 --seed 43 # m=2, α=8
python examples/run_landsat.py --data-root /path/to/CE-SSCFC_Dataset --set vn3c         # confirmatory set {HP, TH, HCM}
```

`--set vn3c` runs the pre-registered confirmatory protocol of the revised paper (Sec. 4.11): the
held-out site Hai Phong takes the role of Hanoi, and every parameter is the VN3 value.

The example samples class-pure circular labelled regions, a 32 × 32 block test hold-out
and a whole-region validation split with the paper's settings, then prints `R_val` and the
test accuracy of Phase 1 and Phase 2 per datasite and the opened gates. Its region
sampler is a compact re-implementation of the authors' loader, so it reproduces the
paper's tables in distribution (same protocol and parameters), not bit for bit. On a
laptop CPU the whole VN3 run takes about 20 s; typical output (seed 44):

```
[fit] 18s, n_min=35, gates opened 1/30: hn<-hcm2:2; rounds=2 backtracks=[0, 0, 0]
datasite    R_val P1  R_val P2    ACC P1    ACC P2    F1 P2   NMI P2
hn            0.1702    0.1626    0.6493    0.6490   0.5469   0.3118
th2           0.1057    0.1057    0.7719    0.7719   0.6197   0.4267
hcm2          0.1137    0.1137    0.6352    0.6352   0.5096   0.3903
```

As in the paper, few gates open (0-1 of 30 per seed), the validation risk of the
receiving datasite never increases, and the label-rich datasites are left unchanged.

## Repository layout

```
.
├── cesscfc.py         # the algorithm: feature map, statistics, closed-form steps, Phase 1, gate, Phase 2
├── metrics.py         # ACC (Hungarian), macro-F1, NMI
├── synthetic.py       # synthetic multi-datasite scenes (Voronoi classes, circular labels, block hold-out)
├── demo.py            # self-contained demonstration, no data needed
├── tests/test_cesscfc.py   # Theorems 1-2 on synthetic data, determinism, parity with the authors' library
├── examples/run_landsat.py # the paper's protocol on CE-SSCFC_Dataset
├── requirements.txt, CITATION.cff, LICENSE
```

## Computational requirements

Pure NumPy on the CPU. Per datasite and AO iteration the dominant costs are the
`N × C` distance matrix and the `D × D` normal equations of the `W`-step (`D = 15` for
`d = 4`), so a Landsat-8 datasite of 1.2 M pixels takes a few seconds per iteration;
Phase 1 converges in 6-8 iterations and Phase 2 needs at most `T_2 · (T_in + ⌊log₂ 1/τ_min⌋ + 1)`
inner solves per datasite. The GPU (CuPy, float32) variant used for the paper's multi-seed
runs lives in the authors' `ds` library; this repository keeps the CPU float64 path only.

## Citation

```bibtex
@article{nguyen2026cesscfc,
  title   = {Classifier-embedded semi-supervised collaborative fuzzy clustering with
             validation-gated knowledge transfer for land-cover mapping across regions},
  author  = {Nguyen, Xuan Hoang and Vuong, Quoc Binh and Tran, Viet Ha and Nguyen, Chien Thang and Hoang, Thang Long},
  journal = {Neurocomputing},
  year    = {2026},
  note    = {Under review. DOI to be assigned on acceptance.}
}
```

Software citation: X. H. Nguyen, *CE-SSCFC_Algs: reference implementation of classifier-embedded semi-supervised collaborative fuzzy clustering*, version 1.0.0, Zenodo, 2026, doi:10.5281/zenodo.22305372. See also [CITATION.cff](CITATION.cff). License: MIT.

## Changelog

- **1.0.0 (2026-09-04)** — archived release accompanying the manuscript; the algorithm, tests and API are unchanged since.
- **2026-09-20 (main, not re-archived)** — `examples/run_landsat.py` gains `--set vn3c`, the confirmatory set of the revised manuscript (held-out site Hai Phong, dataset v1.1.0). Example configuration only.
