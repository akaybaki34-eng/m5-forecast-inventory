"""Classical and intermittent-demand baselines, vectorised across SKUs.

Every function takes the history ``Y_hist`` of shape ``(n_items, origin)`` (NaN = inactive)
and returns a fixed-origin forecast of shape ``(n_items, horizon)``. Nothing after the
origin is ever touched.

Smoothing parameters are chosen per SKU by grid search on in-sample one-step-ahead squared
error; the grid is evaluated for all SKUs at once with ``(n_items, n_grid)`` state arrays,
so fitting 293 SKUs takes well under a second.
"""

from __future__ import annotations

import itertools

import numpy as np

FIT_WINDOW = 730  # days of history used to fit smoothing models


def seasonal_naive(Y_hist: np.ndarray, horizon: int, season: int = 7) -> np.ndarray:
    """y_hat(t+h) = y(t - season + 1 + (h-1) mod season): repeat the last observed week."""
    last = np.nan_to_num(Y_hist[:, -season:], nan=0.0)
    reps = int(np.ceil(horizon / season))
    return np.tile(last, reps)[:, :horizon]


def moving_average(Y_hist: np.ndarray, horizon: int, window: int = 28) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        level = np.nanmean(Y_hist[:, -window:], axis=1)
    return np.repeat(np.nan_to_num(level)[:, None], horizon, axis=1)


def _window(Y_hist: np.ndarray) -> tuple[np.ndarray, int]:
    lo = max(0, Y_hist.shape[1] - FIT_WINDOW)
    return Y_hist[:, lo:], lo


def _first_valid(Y: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(Y)
    return np.where(ok.any(axis=1), ok.argmax(axis=1), Y.shape[1])


def ets_ana(Y_hist: np.ndarray, horizon: int, season: int = 7,
            alphas=(0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5),
            gammas=(0.0, 0.01, 0.03, 0.05, 0.1, 0.2)) -> np.ndarray:
    """ETS(A,N,A): additive level + additive weekly seasonality, no trend.

    y_t = l_{t-1} + s_{t-m} + e_t ;  l_t = l_{t-1} + alpha e_t ;  s_t = s_{t-m} + gamma e_t
    (labelled "Holt-Winters" in the thesis tables).
    """
    Y, offset = _window(Y_hist)
    n, T = Y.shape
    grid = np.array([(a, g) for a, g in itertools.product(alphas, gammas) if g <= 1 - a])
    G = len(grid)
    alpha, gamma = grid[:, 0][None, :], grid[:, 1][None, :]

    start = _first_valid(Y)
    level0 = np.zeros(n)
    seas0 = np.zeros((n, season))
    for i in range(n):  # initialise from the first 8 active weeks of each SKU
        seg = Y[i, start[i]:start[i] + 8 * season]
        seg_t = np.arange(start[i], start[i] + len(seg)) + offset
        if np.isfinite(seg).sum() == 0:
            continue
        level0[i] = np.nanmean(seg)
        for p in range(season):
            v = seg[(seg_t % season) == p]
            if np.isfinite(v).any():
                seas0[i, p] = np.nanmean(v) - level0[i]

    level = np.repeat(level0[:, None], G, axis=1)
    seas = np.repeat(seas0[:, None, :], G, axis=1)  # (n, G, m)
    sse = np.zeros((n, G))
    rows = np.arange(n)
    for t in range(T):
        y = Y[:, t]
        ok = ~np.isnan(y)
        if not ok.any():
            continue
        p = (t + offset) % season
        e = np.where(ok[:, None], y[:, None] - (level + seas[:, :, p]), 0.0)
        sse += e * e
        level = level + alpha * e
        seas[:, :, p] = seas[:, :, p] + gamma * e
    best = sse.argmin(axis=1)
    lv = level[rows, best]
    sv = seas[rows, best]  # (n, m)
    T_abs = Y_hist.shape[1]
    phases = (T_abs + np.arange(horizon)) % season
    return np.maximum(lv[:, None] + sv[:, phases], 0.0)


def _intermittent_init(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yy = np.where(np.isnan(Y), 0.0, Y)
    pos = yy > 0
    n_pos = pos.sum(axis=1)
    n_act = (~np.isnan(Y)).sum(axis=1)
    z0 = np.where(n_pos > 0, yy.sum(axis=1) / np.maximum(n_pos, 1), 0.0)
    p0 = np.where(n_pos > 0, n_act / np.maximum(n_pos, 1), 1.0)
    return z0, p0


def croston_sba(Y_hist: np.ndarray, horizon: int,
                alphas=(0.05, 0.1, 0.15, 0.2, 0.3)) -> np.ndarray:
    """Croston with the Syntetos-Boylan Approximation bias correction: (1 - a/2) z / p."""
    Y, _ = _window(Y_hist)
    n, T = Y.shape
    a = np.asarray(alphas)[None, :]
    G = a.shape[1]
    z0, p0 = _intermittent_init(Y)
    z = np.repeat(z0[:, None], G, axis=1)
    p = np.repeat(p0[:, None], G, axis=1)
    q = np.ones((n, G))
    sse = np.zeros((n, G))
    for t in range(T):
        y = Y[:, t]
        ok = ~np.isnan(y)
        if not ok.any():
            continue
        f = (1 - a / 2) * z / p
        e = np.where(ok[:, None], y[:, None] - f, 0.0)
        sse += e * e
        hit = (ok & (np.nan_to_num(y) > 0))[:, None]
        z = np.where(hit, z + a * (y[:, None] - z), z)
        p = np.where(hit, p + a * (q - p), p)
        q = np.where(hit, 1.0, np.where(ok[:, None], q + 1.0, q))
    best = sse.argmin(axis=1)
    rows = np.arange(n)
    f = (1 - a[0, best] / 2) * z[rows, best] / p[rows, best]
    return np.repeat(np.maximum(f, 0.0)[:, None], horizon, axis=1)


def tsb(Y_hist: np.ndarray, horizon: int, alphas=(0.05, 0.1, 0.2, 0.3),
        betas=(0.01, 0.05, 0.1, 0.2)) -> np.ndarray:
    """Teunter-Syntetos-Babai: smooths demand probability every period, size on demand."""
    Y, _ = _window(Y_hist)
    n, T = Y.shape
    grid = np.array(list(itertools.product(alphas, betas)))
    a, b = grid[:, 0][None, :], grid[:, 1][None, :]
    G = len(grid)
    z0, p0 = _intermittent_init(Y)
    z = np.repeat(z0[:, None], G, axis=1)
    pr = np.repeat(np.minimum(1.0 / p0, 1.0)[:, None], G, axis=1)
    sse = np.zeros((n, G))
    for t in range(T):
        y = Y[:, t]
        ok = ~np.isnan(y)
        if not ok.any():
            continue
        f = pr * z
        e = np.where(ok[:, None], y[:, None] - f, 0.0)
        sse += e * e
        hit = (ok & (np.nan_to_num(y) > 0))[:, None]
        z = np.where(hit, z + a * (y[:, None] - z), z)
        pr = np.where(ok[:, None], pr + b * (hit.astype(float) - pr), pr)
    best = sse.argmin(axis=1)
    rows = np.arange(n)
    f = pr[rows, best] * z[rows, best]
    return np.repeat(np.maximum(f, 0.0)[:, None], horizon, axis=1)


CLASSICAL = {
    "SeasonalNaive": seasonal_naive,
    "MovingAverage28": moving_average,
    "ETS_ANA": ets_ana,
    "Croston_SBA": croston_sba,
    "TSB": tsb,
}
