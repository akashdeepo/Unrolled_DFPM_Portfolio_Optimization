"""Unrolled-DFPM: learn a *data-driven stopping rule* for the damped-dynamical
portfolio solver, and test whether the learned stop recovers the (cheating)
oracle stop.

Motivation
----------
The semiconvergence result says: an interior stopping time along the DFPM
trajectory beats the converged solution. But the "best stop" used in the
original run is chosen with oracle knowledge of Sigma_true, exactly like an
oracle-tuned ridge lambda. Neither is deployable. The open problem -- the actual
research hook, and the bridge to Mazur's estimation-error theory -- is a stopping
rule that uses only *observable* quantities.

Here we learn that rule by unrolling the (differentiable) solver. A small gate
network reads features available at run time (in-sample risk decay, weight norm,
velocity, progress) at each iterate and outputs a stopping distribution
alpha_k. The selected portfolio is w* = sum_k alpha_k w_k. The gate is trained to
minimise realised *true* risk w*' Sigma_true w* across many simulated tasks
(Sigma_true is used ONLY in the training loss; the gate never sees it as input),
so at test time it is fully deployable on observ­ables alone.

We compare, on held-out tasks:
  - oracle stop        : per-task argmin of true variance        (upper bound)
  - learned gate       : this method                              (deployable)
  - L-curve corner     : classical no-oracle rule                 (deployable)
  - best fixed iter    : single constant stop tuned on train      (deployable, non-adaptive)
  - converged          : run to steady state                      (the thing we beat)

Run:  python src/exp2_learned_stopping.py
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from dfpm import dfpm_trajectory, default_dynamics, in_sample_var, true_var
from dgp import make_task

torch.set_default_dtype(torch.float32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# Build one task's trajectory + observable features (no Sigma_true leakage).
# ----------------------------------------------------------------------------
def build_task_tensors(task, n_iter, device=DEVICE):
    Sh = torch.from_numpy(task["Sigma_hat"]).float().to(device)
    St = torch.from_numpy(task["Sigma_true"]).float().to(device)
    dt, eta = default_dynamics(Sh)

    with torch.no_grad():
        W = dfpm_trajectory(Sh, n_iter, dt, eta)          # (n_iter+1, N)
        isv = in_sample_var(W, Sh)                        # (n_iter+1,)
        tv = true_var(W, St)                              # (n_iter+1,)  ORACLE

        wn = (W * W).sum(dim=1)                           # ||w_k||^2
        dv = torch.zeros_like(wn)
        dv[1:] = ((W[1:] - W[:-1]) ** 2).sum(dim=1)       # ||w_k - w_{k-1}||^2
        k = torch.arange(n_iter + 1, device=device).float()

        eps = 1e-12
        feats = torch.stack([
            torch.log(isv / (isv[0] + eps) + eps),        # in-sample risk decay
            torch.log(wn + eps),                          # leverage / concentration
            torch.log(dv + eps),                          # velocity
            k / n_iter,                                   # progress
        ], dim=1)                                          # (n_iter+1, n_feat)

    return dict(W=W, St=St, isv=isv, tv=tv, wn=wn, feats=feats)


# ----------------------------------------------------------------------------
# Classical, deployable comparators (no oracle).
# ----------------------------------------------------------------------------
def lcurve_corner(isv, wn):
    """L-curve corner: max curvature of (log ||w||, log in-sample-var).

    Tikhonov's standard parameter-choice heuristic, transcribed to the iteration
    path. Uses only observables.
    """
    x = torch.log(wn + 1e-12).cpu().numpy()
    y = torch.log(isv + 1e-12).cpu().numpy()
    # discrete curvature kappa = |x' y'' - y' x''| / (x'^2 + y'^2)^{3/2}
    x1, y1 = np.gradient(x), np.gradient(y)
    x2, y2 = np.gradient(x1), np.gradient(y1)
    denom = (x1 ** 2 + y1 ** 2) ** 1.5 + 1e-12
    kappa = np.abs(x1 * y2 - y1 * x2) / denom
    kappa[:3] = 0.0   # ignore the very start (transient)
    return int(np.argmax(kappa))


# ----------------------------------------------------------------------------
# The stopping gate.
# ----------------------------------------------------------------------------
class StopGate(nn.Module):
    def __init__(self, n_feat=4, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats):
        """feats: (n_steps, n_feat) -> alpha: (n_steps,) stopping distribution."""
        scores = self.net(feats).squeeze(-1)              # (n_steps,)
        return torch.softmax(scores, dim=0)

    def soft_portfolio(self, task_t):
        alpha = self.forward(task_t["feats"])
        w_star = (alpha[:, None] * task_t["W"]).sum(dim=0)  # (N,)
        return w_star, alpha


def portfolio_true_var(w, St):
    return (w @ St @ w)


# ----------------------------------------------------------------------------
# Train / evaluate.
# ----------------------------------------------------------------------------
def run(n_iter=800, n_train=300, n_test=150, epochs=300, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    # HETEROGENEOUS task distribution: the whole point of an adaptive stopping
    # rule is to handle tasks whose optimal stop differs. We vary the sample size
    # T (concentration), the number of factors K, the idiosyncratic level, and
    # the tail index. With identical tasks a single constant stop is already
    # near-optimal and there is nothing to adapt to; varying difficulty is what
    # makes the oracle stop move and lets an adaptive gate beat a constant.
    Ts = [40, 50, 70, 100, 140]
    Ks = [2, 5, 10]
    dfs = [np.inf, 8.0, 5.0, 4.0]

    def gen(n):
        out = []
        for _ in range(n):
            T = int(Ts[rng.integers(len(Ts))])
            K = int(Ks[rng.integers(len(Ks))])
            idio = float(rng.uniform(0.1, 0.5))
            df = dfs[rng.integers(len(dfs))]
            task = make_task(N=100, K=K, T=T, df=df, idio=idio, rng=rng)
            t = build_task_tensors(task, n_iter)
            t["oracle_iter"] = int(torch.argmin(t["tv"]).item())
            t["oracle_var"] = float(t["tv"].min().item())
            t["df"], t["T"], t["K"] = df, T, K
            out.append(t)
        return out

    print(f"building {n_train} train + {n_test} test tasks (n_iter={n_iter}) ...")
    train = gen(n_train)
    test = gen(n_test)

    # Best fixed iteration (tuned on train): the non-adaptive deployable baseline.
    tv_stack = torch.stack([t["tv"] for t in train])       # (n_train, n_iter+1)
    fixed_iter = int(torch.argmin(tv_stack.mean(dim=0)).item())

    gate = StopGate().to(DEVICE)
    opt = torch.optim.Adam(gate.parameters(), lr=lr)

    print(f"training gate ({sum(p.numel() for p in gate.parameters())} params) ...")
    for ep in range(epochs):
        perm = rng.permutation(n_train)
        total = 0.0
        opt.zero_grad()
        for i in perm:
            t = train[i]
            w_star, _ = gate.soft_portfolio(t)
            # regret ratio vs the oracle stop (scale-free across tasks)
            loss = portfolio_true_var(w_star, t["St"]) / (t["oracle_var"] + 1e-12)
            (loss / n_train).backward()
            total += float(loss.detach())
        opt.step()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:4d}  mean train regret-ratio {total / n_train:.4f}")

    # ---- evaluation on held-out tasks ----
    def eval_split(split, label):
        agg = {k: [] for k in
               ["oracle", "gate_soft", "gate_hard", "lcurve", "fixed", "converged",
                "gate_iter", "oracle_iter"]}
        with torch.no_grad():
            for t in split:
                St, W, tv = t["St"], t["W"], t["tv"]
                alpha = gate(t["feats"])
                w_soft, _ = gate.soft_portfolio(t)
                ghard = int(torch.argmax(alpha).item())
                lc = lcurve_corner(t["isv"], t["wn"])

                agg["oracle"].append(t["oracle_var"])
                agg["gate_soft"].append(float(portfolio_true_var(w_soft, St)))
                agg["gate_hard"].append(float(tv[ghard]))
                agg["lcurve"].append(float(tv[lc]))
                agg["fixed"].append(float(tv[fixed_iter]))
                agg["converged"].append(float(tv[-1]))
                agg["gate_iter"].append(ghard)
                agg["oracle_iter"].append(t["oracle_iter"])
        m = {k: float(np.mean(v)) for k, v in agg.items()}
        gi = np.array(agg["gate_iter"]); oi = np.array(agg["oracle_iter"])
        corr = float(np.corrcoef(gi, oi)[0, 1]) if gi.std() > 0 and oi.std() > 0 else float("nan")
        mae = float(np.mean(np.abs(gi - oi)))
        print(f"\n=== {label} (n={len(split)}) ===")
        print(f"  mean true OOS variance (lower better):")
        print(f"    oracle stop (upper bound) : {m['oracle']:.5f}")
        print(f"    learned gate (soft)       : {m['gate_soft']:.5f}   <- deployable")
        print(f"    learned gate (hard argmax): {m['gate_hard']:.5f}")
        print(f"    L-curve corner            : {m['lcurve']:.5f}")
        print(f"    best fixed iter (={fixed_iter}) : {m['fixed']:.5f}")
        print(f"    converged                 : {m['converged']:.5f}")
        print(f"  regret vs oracle:  gate {m['gate_soft']/m['oracle']:.3f}x  "
              f"fixed {m['fixed']/m['oracle']:.3f}x  "
              f"lcurve {m['lcurve']/m['oracle']:.3f}x  conv {m['converged']/m['oracle']:.3f}x")
        print(f"  oracle stop spread: mean={oi.mean():.0f}  std={oi.std():.0f}  "
              f"range=[{oi.min()},{oi.max()}]   <- heterogeneity to adapt to")
        print(f"  stop-iter tracking (gate hard argmax): corr={corr:.3f}  MAE={mae:.0f} iters")
        return m, (gi, oi)

    m_all, (gi, oi) = eval_split(test, "TEST (heterogeneous)")
    # per-distribution transfer check
    for df, lab in [(np.inf, "TEST Gaussian only"), (4.0, "TEST t(nu=4) only")]:
        sub = [t for t in test if t["df"] == df]
        if sub:
            eval_split(sub, lab)

    # scatter: gate-chosen stop vs oracle stop (does the learned rule adapt?)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(oi, gi, s=18, alpha=0.6)
        lo, hi = 0, max(oi.max(), gi.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect tracking")
        ax.axhline(fixed_iter, color="r", ls=":", lw=1,
                   label=f"best constant stop ({fixed_iter})")
        ax.set_xlabel("oracle stop iteration (per task)")
        ax.set_ylabel("learned gate stop iteration")
        ax.set_title("Learned data-driven stop vs oracle stop")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("../figures/gate_vs_oracle.png", dpi=130)
        print("\nsaved figures/gate_vs_oracle.png")
    except Exception as e:
        print(f"(figure skipped: {e})")

    return gate


if __name__ == "__main__":
    run()
