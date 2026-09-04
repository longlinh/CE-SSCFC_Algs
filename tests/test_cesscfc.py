"""Tests: structural guarantees of the paper on synthetic datasites, determinism, and
(when the authors' `ds` library is importable) numerical parity with its CE_SSCFC class.

    pytest tests/ -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cesscfc import (NO_LABEL, ce_sscfc, ce_ssfcm, label_stats, n_min_auto, phi_map,  # noqa: E402
                     region_blocked_split, validation_risk)
from synthetic import make_collab_scene  # noqa: E402

SEED = 42
PARAMS = dict(m=2.0, alpha=8.0, mu=16.0, rho=0.1, eps_gate=0.005)


@pytest.fixture(scope="module")
def scene():
    sites, C = make_collab_scene(seed=SEED)
    val = [region_blocked_split(s["y_lab"], s["shape"], 0.2, SEED)[1] for s in sites]
    return sites, C, val


@pytest.fixture(scope="module")
def result(scene):
    sites, C, val = scene
    return ce_sscfc([s["X"] for s in sites], [s["y_lab"] for s in sites], val, C,
                    seed=SEED, **PARAMS)


def test_phi_map_dimension():
    X = np.random.default_rng(0).normal(size=(10, 4))
    assert phi_map(X, "quad").shape == (10, 1 + 4 + 10)
    assert phi_map(X, "linear").shape == (10, 5)
    assert n_min_auto(4, 15) == 35


def test_phase1_objective_non_increasing(result):
    """Theorem 1: J_CE is monotone non-increasing along the AO iterations."""
    for hist in result["J_hist_phase1"]:
        h = np.asarray(hist)
        assert np.all(np.diff(h) <= 1e-9 * (1.0 + np.abs(h[:-1])))


def test_validation_risk_never_increases(result):
    """Theorem 2: R_val(Phase 2) <= R_val(Phase 1) at every datasite."""
    for r1, r2 in zip(result["risk_phase1"], result["risk_phase2"]):
        assert r2 <= r1 + 1e-12


def test_datasite_without_open_gate_is_frozen(result):
    """A datasite that receives no block keeps its Phase-1 solution exactly."""
    S = len(result["U"])
    closed = [s for s in range(S) if result["gates"][s].sum() == 0]
    assert closed, "the synthetic scene should leave at least one datasite without gates"
    for s in closed:
        assert np.array_equal(result["U"][s], result["U1"][s])
        assert np.array_equal(result["W"][s], result["W1"][s])
        assert result["nu_effective"][s] == 0.0


def test_label_poor_datasite_receives_and_gains(scene, result):
    sites, C, _ = scene
    assert result["gates"][0].sum() >= 1
    t = sites[0]["test_mask"]
    from metrics import accuracy
    a1 = accuracy(sites[0]["y_true"][t], result["labels_phase1"][0][t], C)
    a2 = accuracy(sites[0]["y_true"][t], result["labels"][0][t], C)
    assert a2 > a1


def test_withheld_blocks_respect_n_min(result):
    for st in result["stats"]:
        for k, (G, h, n) in st.items():
            assert n >= result["n_min"]
            assert G.shape == (15, 15) and h.shape == (15,)


def test_deterministic(scene, result):
    sites, C, val = scene
    again = ce_sscfc([s["X"] for s in sites], [s["y_lab"] for s in sites], val, C,
                     seed=SEED, **PARAMS)
    for s in range(len(sites)):
        assert np.array_equal(again["W"][s], result["W"][s])
        assert np.array_equal(again["U"][s], result["U"][s])
    assert np.array_equal(again["gates"], result["gates"])


def test_single_datasite_equals_phase1(scene, result):
    sites, C, val = scene
    y = sites[1]["y_lab"].copy()
    y[val[1]] = NO_LABEL
    single = ce_ssfcm(sites[1]["X"], y, C, m=2.0, alpha=8.0, mu=16.0, seed=SEED)
    assert np.allclose(single["W"], result["W1"][1])


def test_parity_with_ds_library(scene, result):
    """Same numbers as ds.clustering.CE_SSCFC (the implementation used for the paper)
    when driven with the gates, weights and validation risk computed here."""
    ds = pytest.importorskip("ds.clustering")
    sites, C, val = scene
    Xs = [s["X"] for s in sites]
    ys = []
    for s, v in zip(sites, val):
        y = s["y_lab"].copy()
        y[v] = NO_LABEL
        ys.append(y)
    mus, lams, nus = result["mu"], [1e-3 * v for v in result["mu"]], result["nu"]
    common = dict(n_clusters=C, m=2.0, alpha=8.0, mu=mus, lam=lams, phi="quad",
                  w_init="ridge", max_iter=60, tol=1e-7, device="CPU")
    ref1 = ds.CE_SSCFC(nu=0.0, **common)
    ref1.fit(datas=Xs, labeled=ys, seed=SEED)
    for s in range(len(sites)):
        assert np.allclose(ref1.W_phase1_[s], result["W1"][s], atol=1e-8)
        assert np.allclose(ref1.membership_phase1_[s], result["U1"][s], atol=1e-8)

    Phi_val = [phi_map(Xs[s])[val[s]] for s in range(len(sites))]
    y_val = [sites[s]["y_lab"][val[s]] for s in range(len(sites))]

    def risk_fn(s, W):
        return validation_risk(Phi_val[s], y_val[s], np.asarray(W), result["weights"][s])

    ref2 = ds.CE_SSCFC(nu=nus, max_iter_collab=10, inner_iter_collab=20, backtrack=True,
                       backtrack_on="risk", **common)
    ref2.fit(datas=Xs, labeled=ys, seed=SEED, gates=result["gates"], risk_fn=risk_fn)
    for s in range(len(sites)):
        assert np.allclose(ref2.W_[s], result["W"][s], atol=1e-8)
        assert np.allclose(ref2.membership_[s], result["U"][s], atol=1e-8)
    assert ref2.n_backtracks_ == result["n_backtracks"]
    assert ref2.n_collab_rounds_ == result["n_rounds"]
    assert np.allclose(ref2.risk_phase2_, result["risk_phase2"])
    # the shipped statistics are identical too
    for s in range(len(sites)):
        mine = label_stats(phi_map(Xs[s]), ys[s], C, result["n_min"])
        assert set(mine) == set(ref1.label_stats_[s])
