"""CE-SSCFC — classifier-embedded semi-supervised collaborative fuzzy clustering.

NumPy-only reference implementation of the method described in

    X. H. Nguyen, Q. B. Vuong, V. H. Tran, C. T. Nguyen, T. L. Hoang,
    "Classifier-embedded semi-supervised collaborative fuzzy clustering with
    validation-gated knowledge transfer for multi-region land-cover classification",
    submitted to Neurocomputing (2026).

Written to be read next to the equations of the paper: plain functions, one module,
no class hierarchy.  Notation follows the paper (datasite index s, class/cluster k,
feature map phi, classifier W, fuzzifier m).

Objective at one datasite (Eq. J_CE)
    J_CE(U, V, W) = sum_ik u_ik^m [ ||x_i - v_k||^2 + alpha ||W^T phi_i - e_k||^2 ]
                    + mu sum_{i in L} ||W^T phi_i - e_{y_i}||^2 + lam ||W||_F^2

Collaboration channel (per-class sufficient statistics of the labelled pixels)
    a datasite ships only {G_k = sum phi phi^T, h_k = sum phi, n_k} for classes with
    n_k >= n_min; the receiving datasite adds
    nu sum_{r != s} sum_k t_sk^(r) [ tr(W^T G_k W) - 2 (W^T h_k)_k + n_k ]
    to its objective (J_col).  Gates t in {0, 1} are set by a validation test.

Closed-form alternating optimisation (every block strictly convex, m > 1)
    U-step  u_ik  ∝ D_ik^{-1/(m-1)},  D_ik = ||x_i - v_k||^2 + alpha ||W^T phi_i - e_k||^2
    V-step  v_k   = sum_i u_ik^m x_i / sum_i u_ik^m
    W-step  (alpha Phi^T S Phi + mu Phi_L^T Phi_L + nu sum t G_k + lam I) W
            = alpha Phi^T U^m + mu Phi_L^T Y + nu sum t h_k e_k^T,   S = diag(sum_k u_ik^m)

Two-phase protocol (Algorithm 1 of the paper)
    Phase 1   every datasite minimises J_CE alone (nu = 0)
    gating    each received block (r, k) is tried with one W-step; the gate opens when
              the weighted validation risk R_val drops by at least eps_gate
    Phase 2   at most T2 outer rounds x at most T_in AO iterations with strength
              tau_s nu_s; a candidate is accepted iff R_val does not increase, else
              tau_s <- tau_s / 2 and the round is redone from the last accepted
              solution; tau_s < tau_min freezes the datasite.  Datasites that receive
              no block keep the Phase-1 solution unchanged.
    =>  R_val(Phase 2) <= R_val(Phase 1) at every datasite by construction (Theorem 2).
"""

import numpy as np

NO_LABEL = -1
D_FLOOR = 1e-12          # floor for D_ik (x_i == v_k and a perfect classifier fit)
RISK_TOL = 1e-12         # rounding tolerance of the accept rule on R_val


# --------------------------------------------------------------------------- features
def phi_map(X, kind="quad"):
    """Fixed feature map phi.  'linear': (1, x);  'quad': (1, x, (x_a x_b)_{a<=b})."""
    X = np.asarray(X, dtype=float)
    N, d = X.shape
    cols = [np.ones((N, 1)), X]
    if kind == "quad":
        for a in range(d):
            for b in range(a, d):
                cols.append((X[:, a] * X[:, b])[:, None])
    elif kind != "linear":
        raise ValueError(f"kind must be 'quad' or 'linear', got {kind!r}")
    return np.concatenate(cols, axis=1)


def n_min_auto(d, D):
    """Smallest n_k for which recovering n_k pixels from (G_k, h_k, n_k) is
    under-determined: n_k * d > D(D+3)/2 + 1 (paper Sec. 3.3; d=4, D=15 -> 35)."""
    return (D * (D + 3) // 2 + 1) // d + 1


def one_hot(y, n_clusters):
    """(mask of labelled pixels, one-hot targets Y of the labelled pixels)."""
    y = np.asarray(y).astype(int).ravel()
    mask = y != NO_LABEL
    if mask.any() and (y[mask].min() < 0 or y[mask].max() >= n_clusters):
        raise ValueError("labels must be in {0..C-1} or NO_LABEL (-1)")
    Y = np.zeros((int(mask.sum()), n_clusters))
    Y[np.arange(int(mask.sum())), y[mask]] = 1.0
    return mask, Y


def label_stats(Phi, y, n_clusters, n_min=0):
    """Per-class sufficient statistics {k: (G_k, h_k, n_k)} of the labelled pixels.

    This is everything a datasite transmits.  Classes with n_k < n_min are withheld
    (absent from the dict) — the privacy threshold of Sec. 3.3."""
    y = np.asarray(y).astype(int).ravel()
    out = {}
    for k in range(n_clusters):
        idx = y == k
        n_k = int(idx.sum())
        if n_k > 0 and n_k >= n_min:
            P = Phi[idx]
            out[k] = (P.T @ P, P.sum(axis=0), float(n_k))
    return out


def peer_risk(W, G, h, n, k):
    """R_k(W) = tr(W^T G W) - 2 (W^T h)_k + n  (= sum_j ||W^T phi_j - e_k||^2 >= 0)."""
    return float(np.trace(W.T @ G @ W) - 2.0 * (W[:, k] @ h) + n)


def peer_terms(s, gates, stats_all, nu_s, D, n_clusters):
    """(nu_s sum_{r!=s,k} t G_k,  nu_s sum t h_k e_k^T) entering the W-step of datasite s."""
    lhs = np.zeros((D, D))
    rhs = np.zeros((D, n_clusters))
    if nu_s > 0.0:
        for r, stats in enumerate(stats_all):
            if r == s:
                continue
            for k, (G, h, n) in stats.items():
                t = float(gates[s, r, k])
                if t > 0.0:
                    lhs += nu_s * t * G
                    rhs[:, k] += nu_s * t * h
    return lhs, rhs


# --------------------------------------------------------------------------- block updates
def sq_distances(X, V):
    d2 = (X * X).sum(1)[:, None] + (V * V).sum(1)[None, :] - 2.0 * (X @ V.T)
    return np.maximum(d2, 0.0)


def big_d(X, Phi, V, W, alpha):
    """D_ik = ||x_i - v_k||^2 + alpha ||W^T phi_i - e_k||^2, floored."""
    P = Phi @ W
    q = (P * P).sum(1)[:, None] - 2.0 * P + 1.0
    return np.maximum(sq_distances(X, V) + alpha * q, D_FLOOR)


def u_step(D, m):
    """u_ik ∝ D_ik^{-1/(m-1)}, row-min normalised (no overflow)."""
    ratio = D / D.min(axis=1, keepdims=True)
    U = ratio ** (-1.0 / (m - 1.0))
    return U / U.sum(axis=1, keepdims=True)


def v_step(X, Um):
    return (Um.T @ X) / Um.sum(axis=0)[:, None]


def w_step(Phi, Um, PhiL, Y, alpha, mu, lam, peer_lhs=None, peer_rhs=None):
    """Solve the D x D linear system of the W-step (see module docstring)."""
    s = Um.sum(axis=1)
    A = alpha * ((Phi * s[:, None]).T @ Phi)
    B = alpha * (Phi.T @ Um)
    if PhiL is not None:
        A = A + mu * (PhiL.T @ PhiL)
        B = B + mu * (PhiL.T @ Y)
    if peer_lhs is not None:
        A = A + peer_lhs
        B = B + peer_rhs
    A[np.diag_indices_from(A)] += lam
    return np.linalg.solve(A, B)


def ridge_init(PhiL, Y, mu, lam, D, n_clusters):
    """W from the ridge solution on the labelled pixels (Phase-1 initialisation)."""
    if PhiL is None or mu == 0.0:
        return np.zeros((D, n_clusters))
    A = mu * (PhiL.T @ PhiL)
    A[np.diag_indices_from(A)] += lam
    return np.linalg.solve(A, mu * (PhiL.T @ Y))


def init_centroids(X, y, n_clusters, rng):
    """V from the class means of the labelled pixels (aligns clusters with classes
    across datasites); a class without labels gets a random pixel."""
    V = np.empty((n_clusters, X.shape[1]))
    for k in range(n_clusters):
        idx = np.where(y == k)[0]
        V[k] = X[idx].mean(axis=0) if idx.size else X[int(rng.integers(len(X)))]
    return V


# --------------------------------------------------------------------------- objectives
def objective(X, Phi, U, V, W, PhiL, Y, alpha, mu, lam, m):
    """Local J_CE (no collaboration term)."""
    Um = U ** m
    J = float((Um * big_d(X, Phi, V, W, alpha)).sum())
    if PhiL is not None:
        J += float(mu * ((PhiL @ W - Y) ** 2).sum())
    return J + float(lam * (W * W).sum())


def peer_objective(s, W, gates, stats_all, nu_s):
    """nu_s sum_{r!=s,k} t R_k^(r)(W)."""
    J = 0.0
    for r, stats in enumerate(stats_all):
        if r == s:
            continue
        for k, (G, h, n) in stats.items():
            t = float(gates[s, r, k])
            if t > 0.0:
                J += nu_s * t * peer_risk(W, G, h, n, k)
    return J


def alternating_optimisation(X, Phi, V, W, PhiL, Y, alpha, mu, lam, m, tol, max_iter,
                             s=0, gates=None, stats_all=None, nu_s=0.0):
    """U -> V -> W updates with fixed gates and strength nu_s (Theorem 1: J_col is
    non-increasing).  Stops early when |dJ| / (1 + |J|) < tol.  Returns (U, V, W, J history)."""
    C = V.shape[0]
    if gates is None or stats_all is None or nu_s == 0.0:
        peer_lhs = peer_rhs = None
        gates, stats_all, nu_s = np.zeros((1, 1, C)), [dict()], 0.0
    else:
        peer_lhs, peer_rhs = peer_terms(s, gates, stats_all, nu_s, Phi.shape[1], C)
    hist, J_prev, U = [], None, None
    for _ in range(max_iter):
        U = u_step(big_d(X, Phi, V, W, alpha), m)
        Um = U ** m
        V = v_step(X, Um)
        W = w_step(Phi, Um, PhiL, Y, alpha, mu, lam, peer_lhs, peer_rhs)
        J = objective(X, Phi, U, V, W, PhiL, Y, alpha, mu, lam, m) \
            + peer_objective(s, W, gates, stats_all, nu_s)
        hist.append(J)
        if J_prev is not None and abs(J_prev - J) / (1.0 + abs(J)) < tol:
            break
        J_prev = J
    return U, V, W, hist


# --------------------------------------------------------------------------- validation risk
def prior_weights(U_phase1, y_val, n_clusters):
    """w_k = p_hat_k / q_k: p_hat from the Phase-1 partition, q from the validation
    labels (Eq. prior-weight).  Falls back to w = 1 when p_hat vanishes on the
    validation classes.  Returned per validation pixel, normalised to mean 1."""
    pred = np.asarray(U_phase1).argmax(1)
    p_hat = np.bincount(pred, minlength=n_clusters) / len(pred)
    q = np.bincount(y_val, minlength=n_clusters) / len(y_val)
    w = p_hat[y_val] / np.maximum(q[y_val], 1e-12)
    if w.mean() <= 0:
        w = np.ones_like(w)
    return w / w.mean()


def validation_risk(Phi_val, y_val, W, weights=None):
    """R_val(W) = weighted 0-1 error of argmax_k (W^T phi) on the validation pixels."""
    pred = (Phi_val @ W).argmax(1)
    err = (pred != y_val).astype(float)
    return float(err.mean() if weights is None else (weights * err).mean())


def region_blocked_split(y, shape, frac=0.2, seed=42):
    """Hold out whole labelled regions (connected components per class) as the
    validation set V_s; returns (y_fit, val_idx).  Needs scipy."""
    from scipy import ndimage

    y = np.asarray(y).astype(int).ravel()
    H, W = shape
    rng = np.random.default_rng(seed)
    lab2d = y.reshape(H, W)
    y_fit = y.copy()
    val = []
    for k in np.unique(y[y != NO_LABEL]):
        comp, n_comp = ndimage.label(lab2d == k)
        n_val = max(1, int(round(frac * n_comp)))
        pick = rng.choice(np.arange(1, n_comp + 1), n_val, replace=False)
        idx = np.where(np.isin(comp, pick).ravel())[0]
        y_fit[idx] = NO_LABEL
        val.append(idx)
    return y_fit, np.sort(np.concatenate(val))


# --------------------------------------------------------------------------- the method
def fit_phase1(X, y_fit, n_clusters, m, alpha, mu, lam, tol=1e-7, max_iter=60,
               phi="quad", seed=42, rng=None):
    """Phase 1 at one datasite: independent minimisation of J_CE (nu = 0).
    `rng` (optional) is only used for classes without labels (random centroid);
    `ce_sscfc` shares one generator across datasites, as the authors' library does."""
    X = np.asarray(X, dtype=float)
    y_fit = np.asarray(y_fit).astype(int).ravel()
    Phi = phi_map(X, phi)
    mask, Y = one_hot(y_fit, n_clusters)
    PhiL = Phi[mask] if mask.any() else None
    Y = Y if mask.any() else None
    rng = np.random.default_rng(seed) if rng is None else rng
    V0 = init_centroids(X, y_fit, n_clusters, rng)
    W0 = ridge_init(PhiL, Y, mu, lam, Phi.shape[1], n_clusters)
    U, V, W, hist = alternating_optimisation(X, Phi, V0, W0, PhiL, Y, alpha, mu, lam, m,
                                             tol, max_iter)
    return dict(X=X, Phi=Phi, PhiL=PhiL, Y=Y, y_fit=y_fit, U=U, V=V, W=W, J_hist=hist,
                mu=mu, lam=lam)


def gate_candidates(state, s, stats_all, nu_s, risk_fn, eps_gate, alpha, m):
    """Validation gate of datasite s: try each received block (r, k) with ONE W-step at
    the Phase-1 solution; open iff R_val drops by at least eps_gate.
    Returns (gates row (S, C), risk deltas (S, C) with NaN for withheld blocks)."""
    S, C = len(stats_all), state["V"].shape[0]
    Phi, PhiL, Y = state["Phi"], state["PhiL"], state["Y"]
    Um = state["U"] ** m
    s_i = Um.sum(1)
    A0 = alpha * (Phi * s_i[:, None]).T @ Phi
    B0 = alpha * Phi.T @ Um
    if PhiL is not None:
        A0 = A0 + state["mu"] * (PhiL.T @ PhiL)
        B0 = B0 + state["mu"] * (PhiL.T @ Y)
    risk0 = risk_fn(s, state["W"])
    gates = np.zeros((S, C))
    delta = np.full((S, C), np.nan)
    for r in range(S):
        if r == s:
            continue
        for k, (G, h, n) in stats_all[r].items():
            A = A0 + nu_s * G
            A[np.diag_indices_from(A)] += state["lam"]
            B = B0.copy()
            B[:, k] += nu_s * h
            delta[r, k] = risk_fn(s, np.linalg.solve(A, B)) - risk0
            gates[r, k] = float(delta[r, k] <= -eps_gate - RISK_TOL)
    return gates, delta


def fit_phase2(states, stats_all, gates, nus, risk_fn, alpha, m, tol=1e-7,
               max_rounds=10, inner_iter=20, tau_init=1.0, tau_min=1e-3, tol_outer=1e-5):
    """Phase 2: collaborative rounds with per-datasite backtracking on R_val.
    Modifies `states` in place (U, V, W); returns a dict of diagnostics."""
    S = len(states)
    C = states[0]["V"].shape[0]
    D = states[0]["Phi"].shape[1]
    frozen = [not np.any(peer_terms(s, gates, stats_all, nus[s], D, C)[0]) for s in range(S)]
    risk = [risk_fn(s, states[s]["W"]) for s in range(S)]
    risk1 = list(risk)
    taus, taus_acc, n_bt, rounds = [tau_init] * S, [0.0] * S, [0] * S, 0
    if S > 1 and not all(frozen):
        for _ in range(max_rounds):
            W_prev = [st["W"].copy() for st in states]
            for s in range(S):
                if frozen[s]:
                    continue
                st = states[s]
                while True:      # accept / restore loop of Theorem 2
                    U, V, W, _ = alternating_optimisation(
                        st["X"], st["Phi"], st["V"].copy(), st["W"].copy(), st["PhiL"], st["Y"],
                        alpha, st["mu"], st["lam"], m, tol, inner_iter,
                        s=s, gates=gates, stats_all=stats_all, nu_s=taus[s] * nus[s])
                    r_new = risk_fn(s, W)
                    if r_new <= risk[s] + RISK_TOL:
                        st["U"], st["V"], st["W"], risk[s] = U, V, W, r_new
                        taus_acc[s] = taus[s]
                        break
                    taus[s] *= 0.5
                    n_bt[s] += 1
                    if taus[s] < tau_min:
                        frozen[s] = True          # keep the last accepted solution
                        break
            rounds += 1
            max_dw = max(np.abs(states[s]["W"] - W_prev[s]).max() for s in range(S))
            if max_dw < tol_outer or all(frozen):
                break
    return dict(risk_phase1=risk1, risk_phase2=risk, n_backtracks=n_bt, n_rounds=rounds,
                nu_effective=[taus_acc[s] * nus[s] for s in range(S)],
                n_receiving=int(S - sum(not np.any(peer_terms(s, gates, stats_all, nus[s], D, C)[0])
                                        for s in range(S))))


def ce_sscfc(Xs, ys, val_idx, n_clusters, m=2.0, alpha=8.0, mu=16.0, lam_ratio=1e-3,
             rho=0.1, eps_gate=0.005, n_min="auto", phi="quad", tol=1e-7, max_iter=60,
             max_rounds=10, inner_iter=20, tau_min=1e-3, tol_outer=1e-5,
             prior_weighting=True, seed=42):
    """The full method over S datasites (Algorithm 1).

    Parameters
    ----------
    Xs : list of (N_s, d) arrays, one per datasite (standardise per datasite first).
    ys : list of (N_s,) int arrays, ALL labelled pixels (-1 = unlabelled).
    val_idx : list of index arrays — the validation pixels V_s (a subset of the
        labelled pixels, held out as whole regions; see `region_blocked_split`).
        Their labels are removed from the fitting set L_s^fit.
    n_clusters : C.   m : fuzzifier (> 1).   alpha : classifier weight in D_ik.
    mu : supervision weight, scaled per datasite as mu_s = mu * N_s / |L_s^fit|.
    lam_ratio : lam_s = lam_ratio * mu_s.   rho : nu_s = rho * mu_s (peer pixel =
        rho x local labelled pixel in the W-step).
    eps_gate : validation-risk margin of the gate.   n_min : privacy threshold
        ('auto' = n_min_auto(d, D)).   prior_weighting : weight R_val by p_hat_k / q_k.

    Returns a dict with per-datasite lists: U, V, W (final), labels = argmax U,
    U1/V1/W1 (Phase 1), risk_phase1/2, gates (S, S, C), gate_delta (S, S, C),
    nu_effective, n_backtracks, n_rounds, J_hist_phase1, n_min, stats (sent blocks).
    """
    S = len(Xs)
    rng = np.random.default_rng(seed)
    states, y_vals, Phi_vals = [], [], []
    for s in range(S):
        y = np.asarray(ys[s]).astype(int).ravel()
        vi = np.asarray(val_idx[s]).astype(int)
        y_fit = y.copy()
        y_fit[vi] = NO_LABEL
        n_lab = int((y_fit != NO_LABEL).sum())
        mu_s = mu * len(y) / max(n_lab, 1)
        states.append(fit_phase1(Xs[s], y_fit, n_clusters, m, alpha, mu_s, lam_ratio * mu_s,
                                 tol, max_iter, phi, seed, rng))
        y_vals.append(y[vi])
        Phi_vals.append(states[s]["Phi"][vi])
    nus = [rho * st["mu"] for st in states]
    D = states[0]["Phi"].shape[1]
    n_min_eff = n_min_auto(np.asarray(Xs[0]).shape[1], D) if n_min == "auto" else int(n_min)
    stats_all = [label_stats(st["Phi"], st["y_fit"], n_clusters, n_min_eff) for st in states]

    weights = [prior_weights(states[s]["U"], y_vals[s], n_clusters) if prior_weighting
               else None for s in range(S)]

    def risk_fn(s, W):
        return validation_risk(Phi_vals[s], y_vals[s], W, weights[s])

    U1 = [st["U"].copy() for st in states]
    V1 = [st["V"].copy() for st in states]
    W1 = [st["W"].copy() for st in states]
    gates = np.zeros((S, S, n_clusters))
    delta = np.full((S, S, n_clusters), np.nan)
    for s in range(S):
        gates[s], delta[s] = gate_candidates(states[s], s, stats_all, nus[s], risk_fn,
                                             eps_gate, alpha, m)
    info = fit_phase2(states, stats_all, gates, nus, risk_fn, alpha, m, tol, max_rounds,
                      inner_iter, 1.0, tau_min, tol_outer)
    return dict(U=[st["U"] for st in states], V=[st["V"] for st in states],
                W=[st["W"] for st in states],
                labels=[st["U"].argmax(1) for st in states],
                labels_phase1=[u.argmax(1) for u in U1], U1=U1, V1=V1, W1=W1,
                gates=gates, gate_delta=delta, J_hist_phase1=[st["J_hist"] for st in states],
                n_min=n_min_eff, stats=stats_all, mu=[st["mu"] for st in states], nu=nus,
                weights=weights, **info)


def ce_ssfcm(X, y, n_clusters, m=2.0, alpha=8.0, mu=16.0, lam_ratio=1e-3, phi="quad",
             tol=1e-7, max_iter=60, seed=42):
    """Single-datasite special case (Phase 1 only, no collaboration)."""
    y = np.asarray(y).astype(int).ravel()
    mu_s = mu * len(y) / max(int((y != NO_LABEL).sum()), 1)
    st = fit_phase1(X, y, n_clusters, m, alpha, mu_s, lam_ratio * mu_s, tol, max_iter, phi, seed)
    return dict(U=st["U"], V=st["V"], W=st["W"], labels=st["U"].argmax(1),
                J_hist=st["J_hist"], mu=mu_s)


def classifier_labels(X, W, phi="quad"):
    """argmax_k (W^T phi(x)) — the embedded classifier's own prediction."""
    return (phi_map(X, phi) @ W).argmax(1)
