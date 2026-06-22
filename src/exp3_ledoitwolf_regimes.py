"""Experiment 3 -- is the learned stopping rule actually useful? The honest test.

Ledoit-Wolf shrinkage is the natural deployable competitor: a closed-form,
tuning-free covariance estimator that needs no DFPM and no stopping rule. Whether
the learned DFPM stopping rule beats it depends entirely on the regime. This
script makes that explicit by running the same comparison in two regimes:

  A. STRONG factors, fixed T=60  -> Ledoit-Wolf nearly saturates the oracle
     frontier, leaving almost no headroom; any stopping rule is redundant.
  B. WEAK factors, small/heterogeneous T -> Ledoit-Wolf is one of the weaker
     regularised methods, and the DFPM stopping rule genuinely beats it.

Real equity returns are typically strongly factor-driven, so regime A is arguably
the more representative one -- which is why the worthwhile direction is an
objective (e.g. CVaR under heavy tails) where no closed-form shrinkage analogue
exists. Produces figures/regime_dependence.png.

Run:  python src/exp3_ledoitwolf_regimes.py
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from dfpm import dfpm_trajectory, default_dynamics, true_var
from dgp import make_task, make_factor_cov, simulate_returns, sample_cov

torch.set_default_dtype(torch.float64)
N = 100
ones = np.ones(N)

# Learned-gate result for the weak regime, carried over from exp2 (annotation
# only; reproduce it with `python src/exp2_learned_stopping.py`).
GATE_WEAK = 0.00375


def gmv(Sig):
    w = np.linalg.solve(Sig, ones)
    return w / (ones @ w)


def true_var_np(w, St):
    return float(w @ St @ w)


# ---------------------------------------------------------------------------
# Regime A: strong factors, fixed covariance, T = 60.
# ---------------------------------------------------------------------------
def regime_strong(reps=200, T=60, seed=20260622):
    rng0 = np.random.default_rng(seed)
    B = rng0.normal(0, 1.0, (N, 5)) * np.array([1.4, 1.0, 0.8, 0.6, 0.5])
    d = rng0.uniform(0.2, 0.6, N) ** 2
    St = B @ B.T + np.diag(d)
    w_t = gmv(St); true_gmv = true_var_np(w_t, St)

    out = {k: [] for k in ["lw", "dfpm_conv", "oracle", "plugin"]}
    for s in range(reps):
        rng = np.random.default_rng(7000 + s)
        R = simulate_returns(T, B, d, df=np.inf, rng=rng)
        Sh = sample_cov(R)
        # Ledoit-Wolf
        lw = LedoitWolf().fit(R).covariance_
        out["lw"].append(true_var_np(gmv(lw), St))
        # plug-in pseudo-inverse
        wp = np.linalg.pinv(Sh) @ ones; wp /= ones @ wp
        out["plugin"].append(true_var_np(wp, St))
        # DFPM trajectory
        Sht = torch.from_numpy(Sh)
        dt, eta = default_dynamics(Sht)
        W = dfpm_trajectory(Sht, 4000, dt, eta)
        tv = true_var(W, torch.from_numpy(St)).numpy()
        out["oracle"].append(float(tv.min()))
        out["dfpm_conv"].append(float(tv[-1]))
    m = {k: float(np.mean(v)) for k, v in out.items()}
    m["true_gmv"] = true_gmv
    m["gate"] = None  # gate not trained on this regime
    return m


# ---------------------------------------------------------------------------
# Regime B: weak factors, heterogeneous tasks (the exp2 distribution).
# ---------------------------------------------------------------------------
def regime_weak(reps=150, seed=0):
    Ts, Ks, dfs = [40, 50, 70, 100, 140], [2, 5, 10], [np.inf, 8.0, 5.0, 4.0]
    rng = np.random.default_rng(seed)

    def draw():
        T = int(Ts[rng.integers(len(Ts))]); K = int(Ks[rng.integers(len(Ks))])
        idio = float(rng.uniform(0.1, 0.5)); df = dfs[rng.integers(len(dfs))]
        return make_task(N=N, K=K, T=T, df=df, idio=idio, rng=rng)

    for _ in range(300):   # burn train draws -> land on exp2's held-out tasks
        draw()
    out = {k: [] for k in ["lw", "dfpm_conv", "oracle", "plugin"]}
    for _ in range(reps):
        task = draw(); St = task["Sigma_true"]
        lw = LedoitWolf().fit(task["R_train"]).covariance_
        out["lw"].append(true_var_np(gmv(lw), St))
        wp = np.linalg.pinv(task["Sigma_hat"]) @ ones; wp /= ones @ wp
        out["plugin"].append(true_var_np(wp, St))
        Sht = torch.from_numpy(task["Sigma_hat"])
        dt, eta = default_dynamics(Sht)
        W = dfpm_trajectory(Sht, 800, dt, eta)
        tv = true_var(W, torch.from_numpy(St)).numpy()
        out["oracle"].append(float(tv.min()))
        out["dfpm_conv"].append(float(tv[-1]))
    m = {k: float(np.mean(v)) for k, v in out.items()}
    m["gate"] = GATE_WEAK
    return m


def report(label, m):
    print(f"\n=== Regime {label} ===")
    print(f"  Ledoit-Wolf GMV (deployable) : {m['lw']:.5f}   ({m['lw']/m['oracle']:.3f}x oracle)")
    if m["gate"]:
        print(f"  learned gate (exp2)          : {m['gate']:.5f}   ({m['gate']/m['oracle']:.3f}x oracle)")
    print(f"  DFPM converged (deployable)  : {m['dfpm_conv']:.5f}   ({m['dfpm_conv']/m['oracle']:.3f}x oracle)")
    print(f"  DFPM oracle stop (bound)     : {m['oracle']:.5f}   (1.000x)")
    print(f"  plug-in pinv (deployable)    : {m['plugin']:.5f}")
    head = (m["lw"] / m["oracle"] - 1) * 100
    print(f"  --> headroom of shrinkage over oracle: {head:.1f}%  "
          f"({'negligible -> gate redundant' if head < 5 else 'real -> gate can help'})")


if __name__ == "__main__":
    print("Running regime comparison (this takes a minute) ...")
    A = regime_strong()
    B = regime_weak()
    report("A: STRONG factors, T=60", A)
    report("B: WEAK factors, small/heterogeneous T", B)

    # ---- figure: regret vs oracle in each regime ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        regimes = ["Strong factors\n(T=60)", "Weak factors\n(small/het. T)"]
        lw_reg = [A["lw"] / A["oracle"], B["lw"] / B["oracle"]]
        cv_reg = [A["dfpm_conv"] / A["oracle"], B["dfpm_conv"] / B["oracle"]]
        x = np.arange(2); w = 0.32
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.bar(x - w/2, lw_reg, w, label="Ledoit-Wolf shrinkage", color="#4878a8")
        ax.bar(x + w/2, cv_reg, w, label="DFPM converged", color="#a8c4a2")
        # learned gate marker (weak regime only)
        ax.scatter([1], [B["gate"] / B["oracle"]], color="#c0504d", zorder=5, s=70,
                   marker="D", label="learned gate (exp2)")
        ax.axhline(1.0, color="k", ls="--", lw=1.2, label="oracle stop (bound)")
        ax.set_xticks(x); ax.set_xticklabels(regimes)
        ax.set_ylabel("true OOS variance  /  oracle stop  (lower = better)")
        ax.set_title("Whether the stopping rule helps is regime-dependent")
        ax.set_ylim(0.95, max(lw_reg + cv_reg) * 1.12)
        for xi, v in zip(x - w/2, lw_reg):
            ax.text(xi, v + 0.005, f"{v:.2f}x", ha="center", fontsize=9)
        ax.legend(fontsize=8.5, loc="upper left")
        fig.tight_layout()
        fig.savefig("../figures/regime_dependence.png", dpi=140)
        print("\nsaved figures/regime_dependence.png")
    except Exception as e:
        print(f"(figure skipped: {e})")
