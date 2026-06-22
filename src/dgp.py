"""Factor-model data-generating process for the unrolled-DFPM pilot.

Reproduces the setting from the original DFPM semiconvergence run:
N assets driven by K << N common factors plus idiosyncratic noise, with only
T < N samples, so the sample covariance is genuinely rank-deficient
(rank <= T - 1 < N) and its condition number is large.

    r_t = B f_t + eps_t,     B in R^{N x K},  f_t in R^K,  eps_t in R^N
    Sigma_true = B B' + D,   D = diag(idio variances)

Returns are rows (each row = one period). Heavy tails are introduced with a
multivariate Student-t on the factor+idiosyncratic draws (a normal/chi2 scale
mixture), standardised so Cov stays Sigma_true regardless of the d.o.f.
"""
from __future__ import annotations

import numpy as np


def make_factor_cov(N=100, K=5, idio=0.2, factor_scale=1.0, rng=None):
    """Build (Sigma_true, B, d) for an N-asset, K-factor model.

    Parameters chosen so the population covariance is well-conditioned but the
    *sample* covariance at T < N is not.
    """
    rng = np.random.default_rng(rng)
    B = rng.standard_normal((N, K)) * np.sqrt(factor_scale / K)
    d = idio * (0.5 + rng.random(N))          # heterogeneous idiosyncratic var
    Sigma_true = B @ B.T + np.diag(d)
    return Sigma_true, B, d


def _scale_mixture_t(n, p, df, rng):
    """n x p iid-standardised draws with Student-t tails (unit-variance columns).

    Z ~ N(0, I); g ~ chi2(df); X = Z / sqrt(g/df) * sqrt((df-2)/df) -> Var=1.
    df = inf recovers Gaussian.
    """
    Z = rng.standard_normal((n, p))
    if np.isinf(df):
        return Z
    g = rng.chisquare(df, size=n)[:, None]
    return Z / np.sqrt(g / df) * np.sqrt((df - 2) / df)


def simulate_returns(n, B, d, df=np.inf, rng=None):
    """Draw n periods of returns from the factor model with optional heavy tails.

    df=np.inf -> Gaussian factors+noise; df finite (>2) -> Student-t tails with
    Cov preserved at B B' + diag(d).
    """
    rng = np.random.default_rng(rng)
    N, K = B.shape
    F = _scale_mixture_t(n, K, df, rng)       # n x K standardised factors
    E = _scale_mixture_t(n, N, df, rng) * np.sqrt(d)[None, :]
    return F @ B.T + E


def sample_cov(R):
    """Plain (MLE) sample covariance of return rows R (n x N), demeaned."""
    Rc = R - R.mean(axis=0, keepdims=True)
    n = R.shape[0]
    return (Rc.T @ Rc) / n


def make_task(N=100, K=5, T=60, df=np.inf, idio=0.2, rng=None):
    """One simulated estimation task.

    Returns a dict with the train-sample covariance Sigma_hat (rank-deficient),
    the population Sigma_true, the raw train returns, and a fresh OOS return
    block for empirical evaluation.
    """
    rng = np.random.default_rng(rng)
    Sigma_true, B, d = make_factor_cov(N, K, idio=idio, rng=rng)
    R_train = simulate_returns(T, B, d, df=df, rng=rng)
    R_oos = simulate_returns(10 * T, B, d, df=df, rng=rng)   # large OOS block
    Sigma_hat = sample_cov(R_train)
    return dict(Sigma_true=Sigma_true, Sigma_hat=Sigma_hat,
                R_train=R_train, R_oos=R_oos, B=B, d=d)
