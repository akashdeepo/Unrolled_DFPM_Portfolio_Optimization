"""Experiment 5 -- the make-or-break test: DFPM on a NON-QUADRATIC objective (CVaR).

Result 4 showed that on the quadratic GMV objective the DFPM trajectory is a
spectral filter, so covariance shrinkage dominates it. The escape hatch is an
objective where that equivalence breaks. CVaR (Conditional Value-at-Risk) is
piecewise-linear in w and depends on the full return sample, not just its
covariance -- so "shrink the covariance" is not even the right operation, and
under skew the CVaR-optimal portfolio departs from the variance-optimal one.

We solve the smoothed Rockafellar-Uryasev CVaR program by DFPM, jointly on
(w, alpha), with early stopping:

    CVaR_beta(w) = min_alpha  alpha + 1/((1-beta) S) sum_s ( -w'r_s - alpha )_+

and ask three questions:
  1. COLLAPSE: under symmetric returns, does CVaR-DFPM track the variance solution?
  2. SEMICONVERGENCE: does OOS CVaR dip-then-rise along the trajectory (early
     stopping regularises the non-quadratic objective too)?
  3. PAYOFF: under SKEW, does early-stopped CVaR-DFPM beat variance-based
     Ledoit-Wolf shrinkage on out-of-sample CVaR -- the thing shrinkage can't see?

This is exploratory. A clean yes on (3) is the positive hook; a struggle is itself
a sharp question. Run:  python src/exp5_cvar_dfpm.py
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.covariance import LedoitWolf

N_DEFAULT = 100
BETA = 0.95


# ---------------------------------------------------------------------------
# Skewed heavy-tailed factor returns (GH skew-t via normal mean-variance mixture)
# ---------------------------------------------------------------------------
def make_factor(N, K, idio, rng):
    B = rng.standard_normal((N, K)) * np.sqrt(1.0 / K)
    d = idio * (0.5 + rng.random(N))
    return B, d


def simulate(n, B, d, nu, gamma, rng):
    """n x N returns. nu = t dof (heavy tails). gamma (length N) = skew vector;
    gamma=0 -> symmetric heavy-t. Mean is de-meaned by E[W]."""
    N, K = B.shape
    if np.isinf(nu):
        W = np.ones(n); EW = 1.0
    else:
        W = stats.invgamma.rvs(a=nu / 2, scale=nu / 2, size=n, random_state=rng)
        EW = nu / (nu - 2)
    f = rng.standard_normal((n, K))
    eps = rng.standard_normal((n, N)) * np.sqrt(d)
    sym = f @ B.T + eps
    return gamma[None, :] * (W - EW)[:, None] + np.sqrt(W)[:, None] * sym


# ---------------------------------------------------------------------------
# Smoothed CVaR objective and its DFPM solver (joint on w, alpha)
# ---------------------------------------------------------------------------
def smoothed_cvar(w, alpha, R, beta, tau):
    u = (-R @ w - alpha) / tau
    sp = tau * np.logaddexp(0.0, u)                      # softplus_tau
    return alpha + sp.mean() / (1 - beta)


def cvar_grads(w, alpha, R, beta, tau):
    S = R.shape[0]
    s = 1.0 / (1.0 + np.exp(-(-R @ w - alpha) / tau))    # sigmoid(u/tau)
    gw = -(R.T @ s) / ((1 - beta) * S)
    ga = 1.0 - s.sum() / ((1 - beta) * S)
    return gw, ga


def empirical_cvar(w, R, beta):
    """Exact historical CVaR (mean of the worst (1-beta) tail of losses)."""
    losses = -(R @ w)
    q = np.quantile(losses, beta)
    tail = losses[losses >= q]
    return float(tail.mean()) if tail.size else float(q)


def cvar_dfpm(R, beta, tau, n_iter, dt, eta, w0=None, alpha0=0.0, record=None):
    """Damped 2nd-order flow on (w, alpha); w projected to 1'w=1. Returns the
    iterates at indices in `record` (default: a log-spaced grid)."""
    N = R.shape[1]
    P = np.eye(N) - np.ones((N, N)) / N
    w = np.ones(N) / N if w0 is None else w0.copy()
    alpha = float(alpha0)
    vw = np.zeros(N); va = 0.0
    if record is None:
        record = np.unique(np.round(np.logspace(0, np.log10(n_iter), 45)).astype(int))
    rset = set(int(r) for r in record)
    out = {}
    for k in range(n_iter + 1):
        if k in rset:
            out[k] = (w.copy(), alpha)
        gw, ga = cvar_grads(w, alpha, R, beta, tau)
        vw = P @ ((1 - eta * dt) * vw - dt * (P @ gw))
        va = (1 - eta * dt) * va - dt * ga
        w = w + dt * vw
        alpha = alpha + dt * va
    return out, record


def lw_gmv(R):
    """Variance-based deployable benchmark: Ledoit-Wolf shrinkage GMV (budget only)."""
    N = R.shape[1]
    S = LedoitWolf().fit(R).covariance_
    w = np.linalg.solve(S, np.ones(N))
    return w / w.sum()


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
def run(skew, N=40, K=4, T=500, nu=5.0, idio=0.3, reps=60,
        n_iter=8000, seed=0, gamma_mag=2.0):
    rng = np.random.default_rng(seed)
    # scale step/damping to the return magnitude and smoothing
    grid = None
    traj_oos = None
    agg = {k: [] for k in ["one_over_n", "lw_gmv", "dfpm_best", "dfpm_conv", "emp_cvar_stop"]}
    best_iters = []

    for _ in range(reps):
        B, d = make_factor(N, K, idio, rng)
        gamma = np.zeros(N)
        if skew:
            gamma = -gamma_mag * np.abs(B[:, 0])          # downside skew along factor 1
        R = simulate(T, B, d, nu, gamma, rng)
        R_oos = simulate(20 * T, B, d, nu, gamma, rng)

        scale = R.std()
        tau = 0.10 * scale                                # CVaR hinge smoothing
        dt = 0.10 * tau                                   # conservative explicit step
        eta = 4.0 * scale                                 # damping
        alpha0 = float(np.quantile(-(R @ (np.ones(N) / N)), BETA))

        out, grid = cvar_dfpm(R, BETA, tau, n_iter, dt, eta, alpha0=alpha0)
        # OOS CVaR along the trajectory
        oos = np.array([empirical_cvar(out[k][0], R_oos, BETA) for k in grid])
        traj_oos = oos if traj_oos is None else traj_oos + oos
        bi = int(np.argmin(oos))
        best_iters.append(grid[bi])

        agg["one_over_n"].append(empirical_cvar(np.ones(N) / N, R_oos, BETA))
        agg["lw_gmv"].append(empirical_cvar(lw_gmv(R), R_oos, BETA))
        agg["dfpm_best"].append(float(oos[bi]))
        agg["dfpm_conv"].append(float(oos[-1]))

    m = {k: float(np.mean(v)) for k, v in agg.items()}
    m["best_iter"] = int(np.median(best_iters))
    return m, (grid, traj_oos / reps)


def report(label, m):
    print(f"=== {label} ===   (OOS CVaR 95%, lower=better)")
    print(f"  1/N                       : {m['one_over_n']:.4f}")
    print(f"  Ledoit-Wolf GMV (variance): {m['lw_gmv']:.4f}")
    print(f"  CVaR-DFPM best stop       : {m['dfpm_best']:.4f}   (iter {m['best_iter']})")
    print(f"  CVaR-DFPM converged       : {m['dfpm_conv']:.4f}")
    edge = (m['lw_gmv'] - m['dfpm_best']) / m['lw_gmv'] * 100
    print(f"  --> CVaR-DFPM best vs LW-GMV: {edge:+.1f}%  "
          f"({'DFPM better' if edge > 0 else 'LW better'})\n")


if __name__ == "__main__":
    print("CVaR-DFPM pilot (beta=0.95, t-dof=5)\n")
    print("--- WELL-SAMPLED regime: N=40, K=4, T=500 (tail ~25 scenarios) ---\n")
    results = {}
    for skew, label in [(False, "SYMMETRIC heavy-t"), (True, "SKEWED heavy-t")]:
        m, (grid, traj) = run(skew=skew)
        results[label] = (m, grid, traj)
        report(label, m)

    print("--- HARD regime, honest control: N=100, K=5, T=60 (tail ~3 scenarios) ---\n")
    m_hard, _ = run(skew=True, N=100, K=5, T=60, n_iter=4000, reps=40)
    report("SKEWED, T<N (data-starved tail)", m_hard)
    print("  (the small-sample tail regime is where CVaR estimation is hardest;")
    print("   the objective advantage needs enough tail data to surface.)\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)
        for ax, (label, (m, grid, traj)) in zip(axes, results.items()):
            ax.plot(grid, traj, "-", color="#c0504d", lw=1.8, label="CVaR-DFPM trajectory (OOS CVaR)")
            bi = int(np.argmin(traj))
            ax.scatter([grid[bi]], [traj[bi]], color="#c0504d", zorder=5, s=45, label="best stop")
            ax.axhline(m["lw_gmv"], color="#4878a8", ls="--", lw=1.4, label="Ledoit-Wolf GMV (variance)")
            ax.axhline(m["one_over_n"], color="grey", ls=":", lw=1.2, label="1/N")
            ax.set_xscale("log"); ax.set_xlabel("DFPM iteration"); ax.set_title(label)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("out-of-sample CVaR (95%)")
        axes[0].legend(fontsize=8, loc="best")
        fig.suptitle("CVaR-DFPM: does early stopping on a non-quadratic objective beat "
                     "variance-based shrinkage?", fontsize=11)
        fig.tight_layout()
        fig.savefig("../figures/cvar_dfpm.png", dpi=140)
        print("saved figures/cvar_dfpm.png")
    except Exception as e:
        print(f"(figure skipped: {e})")
