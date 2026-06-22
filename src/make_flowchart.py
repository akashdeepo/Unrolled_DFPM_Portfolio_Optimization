"""Render the methodology flowchart for the unrolled-DFPM pilot.

Produces figures/methodology_flowchart.png. The layout encodes the central
distinction of the method: the RUN-TIME pipeline (left) uses only observable
quantities and is deployable, while the population covariance Sigma_true enters
ONLY the offline TRAINING signal (right) -- never the gate's input.

Run:  python src/make_flowchart.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- palette ----------------------------------------------------------------
C_DATA   = "#dfeaf5"   # data / simulation
C_SOLVE  = "#cfe8d8"   # solver
C_LEARN  = "#fde6c4"   # learned components
C_OUT    = "#e7dcf3"   # outputs / results
C_TRAIN  = "#f7d4d4"   # training-only (uses Sigma_true)
EDGE     = "#3a3a3a"
ARROW    = "#444444"
TRAINARR = "#b03030"

fig, ax = plt.subplots(figsize=(11.5, 13.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 142)
ax.axis("off")


def box(cx, cy, w, h, text, color, fontsize=10.5, bold=False, edge=EDGE):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.6,rounding_size=2.2",
                       linewidth=1.4, edgecolor=edge, facecolor=color, zorder=3)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            zorder=4, weight=("bold" if bold else "normal"))
    return dict(cx=cx, cy=cy, w=w, h=h)


def lane(x0, x1, y0, y1, label, color):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                       boxstyle="round,pad=0.2,rounding_size=2",
                       linewidth=1.2, linestyle="--",
                       edgecolor="#888888", facecolor=color, alpha=0.35, zorder=1)
    ax.add_patch(p)
    ax.text((x0 + x1) / 2, y1 - 2.4, label, ha="center", va="center",
            fontsize=11, style="italic", color="#444444", zorder=2)


def arrow(a, b, color=ARROW, style="-|>", dashed=False, label=None,
          lw=1.8, rad=0.0, lx=0, ly=0):
    """Arrow from edge of box a to edge of box b (auto vertical/horizontal)."""
    ax1, ay1 = a["cx"], a["cy"]
    bx, by = b["cx"], b["cy"]
    # choose anchor points on box edges
    if abs(by - ay1) >= abs(bx - ax1):           # mostly vertical
        p1 = (ax1, ay1 - a["h"] / 2 if by < ay1 else ay1 + a["h"] / 2)
        p2 = (bx, by + b["h"] / 2 if by < ay1 else by - b["h"] / 2)
    else:                                         # mostly horizontal
        p1 = (ax1 + a["w"] / 2 if bx > ax1 else ax1 - a["w"] / 2, ay1)
        p2 = (bx - b["w"] / 2 if bx > ax1 else bx + b["w"] / 2, by)
    ar = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=18,
                         linewidth=lw, color=color, zorder=2,
                         linestyle=("--" if dashed else "-"),
                         connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(ar)
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly
        ax.text(mx, my, label, ha="center", va="center", fontsize=8.8,
                color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9))


# ---- title ------------------------------------------------------------------
ax.text(50, 139.5, "Unrolled DFPM — methodology pipeline", ha="center",
        fontsize=15, weight="bold")

# ---- swimlanes --------------------------------------------------------------
lane(3, 64, 4, 116, "RUN-TIME PIPELINE  —  observables only (deployable)", C_DATA)
lane(67, 97, 4, 116, "TRAINING ONLY  —  uses Σ$_{true}$", C_TRAIN)

# ---- shared data generation (top, spanning) --------------------------------
data = box(50, 130, 70, 10,
           "Data-generating process (simulation)\n"
           r"factor model  $r_t = B f_t + \varepsilon_t$,   "
           r"$\Sigma_{true}=BB^\top+\mathrm{diag}(d)$,   draw $T<N$ returns",
           C_DATA, bold=True)

# ---- run-time pipeline (left lane) -----------------------------------------
cov = box(33, 110,
          50, 11,
          r"Sample covariance  $\widehat{\Sigma}$" + "\n"
          r"rank $\leq T{-}1 < N$  $\Rightarrow$  singular / ill-posed",
          C_SOLVE)

solver = box(33, 92, 50, 13,
             "Differentiable DFPM solver (PyTorch)\n"
             r"$\ddot w=-\eta\dot w-P\widehat{\Sigma}w$,   symplectic Euler" + "\n"
             r"budget-projected $P=I-\frac{1}{N}\mathbf{1}\mathbf{1}^\top$",
             C_SOLVE)

traj = box(33, 74, 50, 12,
           r"Trajectory  $W=(w_0,\dots,w_n)$" + "\n"
           r"per-iterate observable features $\phi_k$:" + "\n"
           "in-sample risk decay, $\\|w_k\\|$, velocity, progress",
           C_SOLVE)

gate = box(33, 56, 50, 12,
           "Gate network  (MLP, 1,249 params)\n"
           r"reads $\phi_k$  $\Rightarrow$  stopping distribution $\alpha_k$",
           C_LEARN, bold=True)

port = box(33, 40, 50, 9,
           r"Selected portfolio   $w^\star=\sum_k \alpha_k\, w_k$" + "\n"
           "(gate-weighted path average)",
           C_LEARN)

# ---- training-only signal (right lane) -------------------------------------
truerisk = box(82, 92, 26, 15,
               r"True risk" + "\n" + r"$\mathrm{TV}_k=w_k^\top\Sigma_{true}w_k$"
               + "\n" "dips then rises\n(semiconvergence)\n"
               r"oracle stop $k^\star=\arg\min_k$",
               C_TRAIN)

loss = box(82, 56, 26, 14,
           "Training loss\n"
           r"$\mathbb{E}\left[\frac{w^{\star\top}\Sigma_{true}w^\star}{\min_k \mathrm{TV}_k}\right]$"
           + "\n" "(Σ$_{true}$ only here)",
           C_TRAIN)

# ---- evaluation / result (bottom, spanning) --------------------------------
evalb = box(33, 22, 50, 10,
            "Evaluation — held-out heterogeneous tasks\n"
            "learned gate  vs  oracle / fixed / converged / L-curve",
            C_OUT)

result = box(50, 9, 74, 8.5,
             "Result:  learned gate = best deployable rule\n"
             r"$1.031\times$ oracle,  closes $\sim$80% of the converged$\to$oracle gap",
             C_OUT, bold=True)

# ---- arrows: main run-time flow --------------------------------------------
arrow(data, cov)
arrow(cov, solver)
arrow(solver, traj)
arrow(traj, gate)
arrow(gate, port)
arrow(port, evalb)
arrow(evalb, result)

# ---- arrows: training signal (red, dashed) ---------------------------------
arrow(data, truerisk, color=TRAINARR, dashed=True, label=r"$\Sigma_{true}$", rad=-0.15)
arrow(solver, truerisk, color=TRAINARR, dashed=True, label="trajectory", rad=0.0)
arrow(truerisk, loss, color=TRAINARR, dashed=True)
arrow(port, loss, color=TRAINARR, dashed=True, label=r"$w^\star$")
arrow(loss, gate, color=TRAINARR, dashed=True,
      label="backprop →\nupdate gate", rad=-0.32, ly=-5)
ax.text(82, 40, "backprop through the\nunrolled DFPM solver",
        ha="center", va="center", fontsize=8.5, style="italic", color=TRAINARR, zorder=5)

fig.tight_layout()
fig.savefig("../figures/methodology_flowchart.png", dpi=140, bbox_inches="tight")
print("saved figures/methodology_flowchart.png")
