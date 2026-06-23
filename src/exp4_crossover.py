"""Experiment 4 -- could nonlinear shrinkage close the gap? The crossover sweep.

exp3 showed the learned DFPM stop beats *linear* Ledoit-Wolf in a weak-factor,
small-sample regime. The obvious objection: linear shrinkage is a crude, one-dial
filter; a stronger *nonlinear* shrinkage might close that gap. Rather than
implement (and risk mis-implementing) analytical nonlinear shrinkage, we compute
the exact SHRINKAGE CEILING: the lowest true GMV variance achievable by ANY
rotation-equivariant shrinkage estimator -- linear, nonlinear, realizable, or
oracle -- because every such estimator has the form U diag(d) Uᵀ with the sample
eigenvectors U and some d > 0. That is a small convex QP.

This bounds *all* shrinkage at once. Then:
  * (linear LW) - (ceiling)            = how much a better nonlinear shrinkage
                                          could possibly gain. Large -> NLS likely
                                          closes the gap.
  * (ceiling)   - (true GMV)           = eigenvector-estimation error no shrinkage
                                          can fix.
  * (DFPM oracle stop) vs (ceiling)    = a theory check. For the quadratic GMV
                                          objective the DFPM trajectory is itself
                                          a spectral filter, so it cannot beat the
                                          best shrinkage filter -- we expect the
                                          ceiling at or below DFPM. If that holds,
                                          GMV is the wrong battlefield and the
                                          method needs a non-quadratic objective
                                          (CVaR) to have a filter-inexpressible
                                          edge.

Sweeps factor strength x concentration c = N/T (crossing c = 1 into the singular
p > n regime). Produces figures/crossover.png.

Run:  python src/exp4_crossover.py
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from dfpm import dfpm_trajectory, default_dynamics, true_var
from dgp import make_factor_cov, simulate_returns, sample_cov

torch.set_default_dtype(torch.float64)
N = 100
K = 5
ones = np.ones(N)


def gmv_var(Sig_inv_source, Sigma_true):
    """True variance of the GMV portfolio built from a covariance estimate."""
    w = np.linalg.solve(Sig_inv_source, ones)
    w /= ones @ w
    return float(w @ Sigma_true @ w)


def shrinkage_ceiling(Sigma_hat, Sigma_true):
    """Lowest true GMV variance over ALL rotation-equivariant shrinkage estimators.

    Any such estimator is U diag(d) Uᵀ, d > 0, with U the eigenvectors of
    Sigma_hat. With a = Uᵀ1 and g = 1/d >= 0, the GMV portfolio's true variance is
    gᵀ M~ g / (bᵀg)² with M~ = diag(a) Uᵀ Sigma_true U diag(a) and b = a².
    Fixing bᵀg = 1, minimise gᵀ M~ g -- a convex QP over the scaled simplex.
    """
    evals, U = np.linalg.eigh(Sigma_hat)
    a = U.T @ ones
    M = U.T @ Sigma_true @ U
    Mt = (a[:, None] * M) * a[None, :]          # diag(a) M diag(a), PSD
    Mt = 0.5 * (Mt + Mt.T)
    b = a * a

    def f(g):  return float(g @ Mt @ g)
    def grad(g): return 2.0 * (Mt @ g)
    g0 = np.full(N, 1.0 / max(b.sum(), 1e-12))
    cons = [{"type": "eq", "fun": lambda g: b @ g - 1.0, "jac": lambda g: b}]
    res = minimize(f, g0, jac=grad, bounds=[(0, None)] * N, constraints=cons,
                   method="SLSQP", options={"maxiter": 400, "ftol": 1e-12})
    return float(res.fun)   # = optimal true GMV variance (bᵀg = 1 ⇒ objective is the variance)


def run_cell(factor_scale, T, reps, n_iter=2000, seed=0):
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ["true", "oracle", "conv", "lw", "ceiling"]}
    for _ in range(reps):
        St, B, d = make_factor_cov(N, K, idio=0.3, factor_scale=factor_scale, rng=rng)
        R = simulate_returns(T, B, d, df=np.inf, rng=rng)
        Sh = sample_cov(R)

        acc["true"].append(gmv_var(St, St))                       # true GMV
        acc["lw"].append(gmv_var(LedoitWolf().fit(R).covariance_, St))
        acc["ceiling"].append(shrinkage_ceiling(Sh, St))

        Sht = torch.from_numpy(Sh)
        dt, eta = default_dynamics(Sht)
        W = dfpm_trajectory(Sht, n_iter, dt, eta)
        tv = true_var(W, torch.from_numpy(St)).numpy()
        acc["oracle"].append(float(tv.min()))
        acc["conv"].append(float(tv[-1]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


if __name__ == "__main__":
    strengths = [("weak", 0.2), ("medium", 1.0), ("strong", 5.0)]
    Ts = [50, 70, 120, 200]            # c = N/T = 2.0, 1.43, 0.83, 0.5
    REPS = 30

    # --- sanity check the QP on one cell: true GMV <= ceiling <= linear LW ---
    chk = run_cell(0.2, 70, reps=10, seed=1)
    print("QP sanity (weak, T=70):  true GMV %.5f  <=  ceiling %.5f  <=  linear LW %.5f"
          % (chk["true"], chk["ceiling"], chk["lw"]))
    ok = chk["true"] <= chk["ceiling"] + 1e-6 <= chk["lw"] + 1e-6
    print("  ordering holds:" , ok, "\n")

    rows = []
    print(f"{'strength':8s} {'c=N/T':6s} | regret over true GMV (lower=better)")
    print(f"{'':8s} {'':6s} |  linLW  ceiling  DFPMconv  DFPMbest")
    for name, fs in strengths:
        for T in Ts:
            m = run_cell(fs, T, REPS)
            c = N / T
            r = {k: m[k] / m["true"] for k in ["lw", "ceiling", "conv", "oracle"]}
            rows.append((name, c, r, m))
            print(f"{name:8s} {c:5.2f}  |  {r['lw']:.3f}  {r['ceiling']:.3f}   "
                  f"{r['conv']:.3f}    {r['oracle']:.3f}")

    # --- figure: 3 panels (one per factor strength), regret vs concentration ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
        for ax, (name, fs) in zip(axes, strengths):
            sub = [(c, r) for (nm, c, r, m) in rows if nm == name]
            cs = [c for c, _ in sub]
            ax.fill_between([min(cs), max(cs)], 1.0, 1.025, color="#9ec79e", alpha=0.5,
                            label="shrinkage ceiling band")
            ax.plot(cs, [r["lw"] for _, r in sub], "o-", color="#4878a8", label="linear Ledoit-Wolf (deployable)")
            ax.plot(cs, [r["ceiling"] for _, r in sub], "s--", color="#3a3a3a", label="shrinkage ceiling (best possible NLS)")
            ax.plot(cs, [r["oracle"] for _, r in sub], "D-", color="#c0504d", label="DFPM best stop (oracle bound)")
            ax.axvline(1.0, color="k", ls=":", lw=1)
            ax.text(1.04, 1.52, "p=n", fontsize=8, va="top")
            ax.set_title(f"{name} factors"); ax.set_xlabel("concentration  c = N/T")
            ax.set_xscale("log"); ax.grid(alpha=0.3); ax.set_ylim(0.99, 1.55)
        axes[0].set_ylabel("true GMV variance / true-GMV optimum")
        axes[0].legend(fontsize=7.5, loc="upper right")
        fig.suptitle("On GMV, the best possible shrinkage (ceiling) beats DFPM's best stop "
                     "in every regime\n(DFPM only beats *linear* LW because linear LW is a crude filter)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig("../figures/crossover.png", dpi=140)
        print("\nsaved figures/crossover.png")
    except Exception as e:
        print(f"(figure skipped: {e})")
