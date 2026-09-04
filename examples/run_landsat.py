#!/usr/bin/env python3
"""Run CE-SSCFC on the companion Landsat-8 dataset (github.com/longlinh/CE-SSCFC_Dataset,
doi:10.5281/zenodo.22304965) with the paper's protocol, and print Phase-1 / Phase-2
test accuracy per datasite together with the opened gates.

    python examples/run_landsat.py --data-root /path/to/CE-SSCFC_Dataset --set vn3
    python examples/run_landsat.py --data-root /path/to/CE-SSCFC_Dataset --set es2 --seed 43

Protocol (Sec. 4.3 of the paper): class-pure circular labelled regions sampled from the
reference map (VN3: 10 / 14 / 24 regions per class, radius 30 px; ES2: Valencia 16 x r30,
Alicante 12 x r15), 32 x 32 spatial block hold-out of ~30 % as the test set, 20 % of the
labelled regions per class held out as whole regions for the validation gate, per-datasite
z-scoring.  The region sampler here is a compact re-implementation of the authors'
loader, so the numbers reproduce the paper's tables in distribution, not bit for bit.
Pure NumPy on the CPU: about 1-3 minutes per datasite of ~1.2 M pixels.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cesscfc import ce_sscfc, region_blocked_split                  # noqa: E402
from metrics import evaluate                                         # noqa: E402
from synthetic import block_holdout, circular_labels, standardize    # noqa: E402

SETS = {
    "vn3": dict(sites=["hn", "th2", "hcm2"], bands=(2, 3, 4, 5), gt="WorldCover2021_5class",
                C=5, n_regions={"hn": 10, "th2": 14, "hcm2": 24}, radius={"hn": 30, "th2": 30, "hcm2": 30},
                m=3.0, alpha=32.0),
    "es2": dict(sites=["valencia", "alicante"], bands=(2, 3, 4, 5, 6, 7), gt="WorldCover2021_6class",
                C=6, n_regions={"valencia": 16, "alicante": 12}, radius={"valencia": 30, "alicante": 15},
                m=2.0, alpha=8.0),
}


def load_site(root, name, bands, gt):
    import rasterio
    d = Path(root) / "landsat8" / name
    X = np.stack([rasterio.open(d / f"{name}_z50_SR_B{b}.tif").read(1) for b in bands], -1)
    with rasterio.open(d / f"{name}_z50_{gt}.tif") as src:
        y = src.read(1).astype(int)
    H, W = y.shape
    y = y.ravel()
    y[y == 255] = -1
    return standardize(X.reshape(H * W, len(bands))), y, (H, W)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", required=True, help="folder containing landsat8/<site>/")
    ap.add_argument("--set", choices=list(SETS), default="vn3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mu", type=float, default=16.0)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--eps-gate", type=float, default=0.005)
    args = ap.parse_args(argv)
    cfg = SETS[args.set]
    C = cfg["C"]

    Xs, ys, vals, sites = [], [], [], []
    for i, name in enumerate(cfg["sites"]):
        t0 = time.perf_counter()
        X, y_true, shape = load_site(args.data_root, name, cfg["bands"], cfg["gt"])
        rng = np.random.default_rng(args.seed + 100 * i)
        y_lab = circular_labels(y_true, shape, cfg["n_regions"][name], cfg["radius"][name], rng)
        test = block_holdout(shape, y_lab, rng, block=32, test_ratio=0.3, valid=y_true >= 0)
        _, val_idx = region_blocked_split(y_lab, shape, frac=0.2, seed=args.seed)
        Xs.append(X); ys.append(y_lab); vals.append(val_idx)
        sites.append(dict(name=name, y_true=y_true, test=test))
        print(f"[load] {name}: {shape[0]}x{shape[1]} px, {len(cfg['bands'])} bands, "
              f"{100 * (y_lab >= 0).mean():.1f} % labelled, {len(val_idx)} validation px, "
              f"{100 * test.mean():.1f} % test  ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    res = ce_sscfc(Xs, ys, vals, C, m=cfg["m"], alpha=cfg["alpha"], mu=args.mu, rho=args.rho,
                   eps_gate=args.eps_gate, seed=args.seed)
    S = len(sites)
    opened = [f"{sites[s]['name']}<-{sites[r]['name']}:{k}" for s in range(S) for r in range(S)
              for k in range(C) if res["gates"][s, r, k] > 0]
    print(f"[fit] {time.perf_counter() - t0:.0f}s, n_min={res['n_min']}, gates opened "
          f"{len(opened)}/{S * (S - 1) * C}: {', '.join(opened) or 'none'}; "
          f"rounds={res['n_rounds']} backtracks={res['n_backtracks']}")
    print(f"{'datasite':<10}{'R_val P1':>10}{'R_val P2':>10}{'ACC P1':>10}{'ACC P2':>10}"
          f"{'F1 P2':>9}{'NMI P2':>9}")
    for s, st in enumerate(sites):
        t = st["test"]
        m1 = evaluate(st["y_true"][t], res["labels_phase1"][s][t], C)
        m2 = evaluate(st["y_true"][t], res["labels"][s][t], C)
        print(f"{st['name']:<10}{res['risk_phase1'][s]:>10.4f}{res['risk_phase2'][s]:>10.4f}"
              f"{m1['acc']:>10.4f}{m2['acc']:>10.4f}{m2['f1']:>9.4f}{m2['nmi']:>9.4f}")


if __name__ == "__main__":
    main()
