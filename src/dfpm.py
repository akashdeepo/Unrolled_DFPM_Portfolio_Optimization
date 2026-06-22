"""DFPM (Damped Dynamical / Functional Particle Method) solver for the
minimum-variance portfolio, implemented in PyTorch so the whole trajectory is
differentiable.

Problem (global minimum-variance, GMV):

    min_w  (1/2) w' Sigma_hat w     s.t.   1' w = 1.

Gulliksson/Ogren's idea: don't invert the (here rank-deficient) sample
covariance. Instead let a *second-order damped dynamical system* roll to the
constrained minimiser and integrate it numerically:

    w_ddot = -eta * w_dot - P @ (Sigma_hat @ w),      P = I - (1 1') / N

P projects onto the tangent space of the budget constraint 1'w = 1, so a
trajectory started at the feasible point w0 = 1/N (with zero velocity) stays
feasible for all time. eta is the damping (friction).

The central empirical fact this code exists to expose: with Sigma_hat
rank-deficient (T < N), the *converged* steady state loads up the noisy
near-null directions, so out-of-sample risk along the trajectory falls to a
floor and then rises again (semiconvergence). The integration step index is an
implicit regularisation path; "when to stop" plays the role of a shrinkage
parameter.

Everything here is torch + autograd-friendly: dt, eta, and the trajectory are
differentiable, which is what lets us *learn* a stopping rule by unrolling
(see unroll.py).
"""
from __future__ import annotations

import torch


def projection_ones(N: int, device=None, dtype=torch.float64) -> torch.Tensor:
    """P = I - 1 1' / N : orthogonal projector onto {x : 1'x = 0}."""
    I = torch.eye(N, device=device, dtype=dtype)
    ones = torch.ones(N, N, device=device, dtype=dtype) / N
    return I - ones


def dfpm_trajectory(Sigma_hat: torch.Tensor,
                    n_iter: int,
                    dt: torch.Tensor | float,
                    eta: torch.Tensor | float,
                    w0: torch.Tensor | None = None):
    """Integrate the damped second-order GMV flow for `n_iter` steps.

    Symplectic (semi-implicit) Euler, velocity kept in the constraint tangent
    space so 1'w = 1 holds exactly throughout.

    Returns
    -------
    W : (n_iter+1, N) tensor
        The iterates w_0, w_1, ..., w_{n_iter} (row k = portfolio after k steps).
        Differentiable w.r.t. dt, eta and Sigma_hat.
    """
    N = Sigma_hat.shape[0]
    device, dtype = Sigma_hat.device, Sigma_hat.dtype
    P = projection_ones(N, device=device, dtype=dtype)

    if w0 is None:
        w = torch.ones(N, device=device, dtype=dtype) / N      # equal weight, feasible
    else:
        w = w0.clone()
    v = torch.zeros(N, device=device, dtype=dtype)

    dt = torch.as_tensor(dt, device=device, dtype=dtype)
    eta = torch.as_tensor(eta, device=device, dtype=dtype)

    W = [w]
    for _ in range(n_iter):
        grad = Sigma_hat @ w                # gradient of (1/2) w'Sigma w
        g = P @ grad                        # project onto feasible directions
        v = (1.0 - eta * dt) * v - dt * g   # damped velocity update
        v = P @ v                           # stay in the tangent space
        w = w + dt * v                      # symplectic position update
        W.append(w)
    return torch.stack(W, dim=0)


def stable_dt(Sigma_hat: torch.Tensor, safety: float = 0.5) -> float:
    """A conservative step size: safety / lambda_max(P Sigma_hat P).

    The fastest mode of the (undamped) flow has frequency ~ sqrt(lambda_max);
    the leapfrog stability limit is dt < 2/sqrt(lambda_max). We stay well under
    it. lambda_max of the projected operator <= lambda_max(Sigma_hat), which is
    cheap and good enough.
    """
    lam_max = torch.linalg.eigvalsh(Sigma_hat)[-1].item()
    return safety / max(lam_max, 1e-12)


def default_dynamics(Sigma_hat: torch.Tensor, dt_safety: float = 0.25,
                     damp_c: float = 0.5):
    """Pick a well-behaved (dt, eta) pair for the GMV flow.

    Damping is set relative to the *fastest* mode: eta = damp_c * sqrt(lambda_max).
    This critically-to-lightly damps the dominant (factor) directions so they
    settle quickly, while the near-null directions stay lightly damped and drift
    slowly -- which is exactly what produces the semiconvergence dip-then-rise.
    Tying eta to lambda_min instead leaves the high modes ringing forever.
    """
    lam_max = torch.linalg.eigvalsh(Sigma_hat)[-1].item()
    dt = dt_safety / max(lam_max, 1e-12)
    eta = damp_c * (max(lam_max, 1e-12) ** 0.5)
    return dt, eta


def in_sample_var(W: torch.Tensor, Sigma_hat: torch.Tensor) -> torch.Tensor:
    """Per-iterate in-sample variance w_k' Sigma_hat w_k  ->  (n_iter+1,) tensor."""
    return torch.einsum("ki,ij,kj->k", W, Sigma_hat, W)


def true_var(W: torch.Tensor, Sigma_true: torch.Tensor) -> torch.Tensor:
    """Per-iterate *true* (population) variance w_k' Sigma_true w_k.

    Oracle quantity: only available in simulation. The semiconvergence dip lives
    here, and the oracle stop is its argmin.
    """
    return torch.einsum("ki,ij,kj->k", W, Sigma_true, W)
