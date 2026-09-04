#!/usr/bin/env python3
"""Self-contained demonstration of CE-SSCFC on three synthetic datasites (no data).

Datasite 0 is label-poor, datasites 1-2 are label-rich; all share the class spectra.
Phase 1 fits every datasite alone, the validation gate tests each received class
block, Phase 2 collaborates with backtracking on the validation risk.  Deterministic.

    python demo.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cesscfc import ce_sscfc, region_blocked_split          # noqa: E402
from metrics import evaluate                                 # noqa: E402
from synthetic import make_collab_scene                      # noqa: E402


def main(seed=42):
    sites, C = make_collab_scene(seed=seed)
    Xs = [s["X"] for s in sites]
    ys = [s["y_lab"] for s in sites]
    val_idx = [region_blocked_split(s["y_lab"], s["shape"], frac=0.2, seed=seed)[1]
               for s in sites]
    for i, s in enumerate(sites):
        n_lab = int((s["y_lab"] >= 0).sum())
        print(f"datasite {i}: {s['shape'][0]}x{s['shape'][1]} px, "
              f"{n_lab} labelled ({100 * n_lab / len(s['y_lab']):.1f} %), "
              f"{len(val_idx[i])} in the validation regions")

    res = ce_sscfc(Xs, ys, val_idx, n_clusters=C, m=2.0, alpha=8.0, mu=16.0,
                   rho=0.1, eps_gate=0.005, seed=seed)

    S = len(sites)
    opened = [(s, r, k) for s in range(S) for r in range(S) for k in range(C)
              if res["gates"][s, r, k] > 0]
    print(f"\nn_min = {res['n_min']} (privacy threshold); gates opened: "
          f"{len(opened)}/{S * (S - 1) * C} -> "
          + (", ".join(f"site{s}<-site{r}:class{k}" for s, r, k in opened) or "none"))
    print(f"Phase 2: {res['n_rounds']} rounds, backtracks {res['n_backtracks']}, "
          f"nu_eff {[round(v, 3) for v in res['nu_effective']]}\n")
    print("datasite   R_val P1 -> P2        test ACC P1 -> P2   (F1 P2, NMI P2)")
    accs = []
    for i, s in enumerate(sites):
        t = s["test_mask"]
        m1 = evaluate(s["y_true"][t], res["labels_phase1"][i][t], C)
        m2 = evaluate(s["y_true"][t], res["labels"][i][t], C)
        accs.append((m1["acc"], m2["acc"]))
        print(f"   {i}       {res['risk_phase1'][i]:.4f} -> {res['risk_phase2'][i]:.4f}"
              f"        {m1['acc']:.4f} -> {m2['acc']:.4f}   ({m2['f1']:.4f}, {m2['nmi']:.4f})")
    a1, a2 = np.mean([a for a, _ in accs]), np.mean([b for _, b in accs])
    print(f"  mean                                   {a1:.4f} -> {a2:.4f}")
    print(f"memberships U[0]: {res['U'][0].shape}, centroids V[0]: {res['V'][0].shape}, "
          f"classifier W[0]: {res['W'][0].shape}")


if __name__ == "__main__":
    main()
