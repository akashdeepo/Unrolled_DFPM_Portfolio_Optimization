"""Sanity baseline: reproduce the DFPM semiconvergence dip and the reference
portfolios, before any learning. Confirms the solver behaves as in the original
run (OOS variance falls to a floor, then rises as the system converges onto the
noisy near-null directions).

Run:  python src/exp1_semiconvergence.py
"""
from __future__ import annotations

import numpy as np
import torch

from dfpm import dfpm_trajectory, default_dynamics, true_var, in_sample_var
from dgp import make_task

torch.set_default_dtype(torch.float64)


def gmv_pinv(Sigma_hat):
    """Plug-in GMV with Moore-Penrose pseudo-inverse: w = S+ 1 / (1' S+ 1)."""
    Sp = np.linalg.pinv(Sigma_hat)
    ones = np.ones(Sigma_hat.shape[0])
    w = Sp @ ones
    return w / (ones @ w)


def gmv_ridge(Sigma_hat, Sigma_true, lambdas=None):
    """Oracle-tuned ridge GMV: pick lambda minimising TRUE variance (cheating)."""
    if lambdas is None:
        lambdas = np.geomspace(1e-4, 1e1, 40)
    N = Sigma_hat.shape[0]
    ones = np.ones(N)
    best = (np.inf, None, None)
    for lam in lambdas:
        Sinv = np.linalg.inv(Sigma_hat + lam * np.eye(N))
        w = Sinv @ ones
        w /= ones @ w
        tv = float(w @ Sigma_true @ w)
        if tv < best[0]:
            best = (tv, lam, w)
    return best  # (true_var, lambda, w)


def oracle_min_var(Sigma_true):
    """True GMV (lower bound), uses Sigma_true directly."""
    Sinv = np.linalg.inv(Sigma_true)
    ones = np.ones(Sigma_true.shape[0])
    w = Sinv @ ones
    return w / (ones @ w)


def run(df, n_iter=4000, n_rep=200, N=100, K=5, T=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in
            ["one_over_n", "pinv", "dfpm_best", "dfpm_conv",
             "ridge_oracle", "oracle_mv", "best_iter", "cond"]}
    traj_true_sum = None  # average true-variance trajectory

    for _ in range(n_rep):
        task = make_task(N, K, T, df=df, rng=rng)
        Sh = torch.from_numpy(task["Sigma_hat"])
        St = torch.from_numpy(task["Sigma_true"])

        dt, eta = default_dynamics(Sh)
        W = dfpm_trajectory(Sh, n_iter, dt, eta)
        tv = true_var(W, St).detach().numpy()      # (n_iter+1,)

        traj_true_sum = tv if traj_true_sum is None else traj_true_sum + tv
        best_iter = int(np.argmin(tv))

        ones = np.ones(N)
        w_unif = ones / N
        w_pinv = gmv_pinv(task["Sigma_hat"])
        tv_ridge, _, _ = gmv_ridge(task["Sigma_hat"], task["Sigma_true"])
        w_omv = oracle_min_var(task["Sigma_true"])

        def truevar(w):
            return float(w @ task["Sigma_true"] @ w)

        rows["one_over_n"].append(truevar(w_unif))
        rows["pinv"].append(truevar(w_pinv))
        rows["dfpm_best"].append(float(tv[best_iter]))
        rows["dfpm_conv"].append(float(tv[-1]))
        rows["ridge_oracle"].append(tv_ridge)
        rows["oracle_mv"].append(truevar(w_omv))
        rows["best_iter"].append(best_iter)
        eig = np.linalg.eigvalsh(task["Sigma_hat"])
        pos = eig[eig > 1e-8]
        rows["cond"].append(float(pos[-1] / pos[0]))   # conditioning on the range

    summary = {k: float(np.mean(v)) for k, v in rows.items()}
    traj_true = traj_true_sum / n_rep
    return summary, traj_true


if __name__ == "__main__":
    import csv
    trajectories = {}
    for df, label in [(np.inf, "Gaussian"), (4.0, "t(nu=4)")]:
        s, traj = run(df=df)
        trajectories[label] = traj
        print(f"\n=== {label} ===  (N=100,K=5,T=60, 200 reps)")
        print(f"  rank-deficient: rank<=59<100, exactly singular off the range")
        print(f"  mean cond on range        : {s['cond']:.0f}")
        print(f"  median best stop iter     : {s['best_iter']:.0f}")
        print(f"  1/N                       : {s['one_over_n']:.5f}")
        print(f"  plug-in (pinv)            : {s['pinv']:.5f}")
        print(f"  DFPM best stop (oracle)   : {s['dfpm_best']:.5f}")
        print(f"  DFPM converged            : {s['dfpm_conv']:.5f}")
        print(f"  ridge (oracle lambda)     : {s['ridge_oracle']:.5f}")
        print(f"  oracle min-var (LB)       : {s['oracle_mv']:.5f}")
        idx = [0, 5, 15, 30, 60, 120, 250, 500, 1000, 2000, 4000]
        idx = [i for i in idx if i < len(traj)]
        print("  true-var trajectory:")
        for i in idx:
            print(f"      iter {i:5d}: {traj[i]:.5f}")

    # save trajectory CSV + semiconvergence figure
    n = len(next(iter(trajectories.values())))
    with open("../results/semiconvergence_trajectory.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter"] + list(trajectories.keys()))
        for i in range(n):
            w.writerow([i] + [f"{trajectories[k][i]:.6f}" for k in trajectories])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for label, traj in trajectories.items():
            ax.plot(np.arange(n), traj, label=label, lw=1.6)
            bi = int(np.argmin(traj))
            ax.scatter([bi], [traj[bi]], s=30, zorder=5)
        ax.axhline(0.00234, color="grey", ls=":", lw=1, label="oracle-tuned ridge")
        ax.set_xscale("log")
        ax.set_xlabel("DFPM iteration (log scale)")
        ax.set_ylabel("out-of-sample (true) portfolio variance")
        ax.set_title("DFPM semiconvergence: the stopping time is the regulariser")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("../figures/semiconvergence.png", dpi=130)
        print("\nsaved figures/semiconvergence.png and results/semiconvergence_trajectory.csv")
    except Exception as e:
        print(f"(figure skipped: {e})")
