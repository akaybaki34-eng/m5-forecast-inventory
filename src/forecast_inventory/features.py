"""Leakage-safe feature generation.

A single function, :func:`day_features`, computes the feature matrix for *one* day ``t``
from the demand history strictly before ``t`` plus exogenous information known in advance
(price, calendar, SNAP, events). It is used both to build the training set (on actual
history) and inside the recursive forecaster (on history where future days have been
replaced by the model's own forecasts). Because the same code path serves both, the
model can never see a feature at inference that was computed differently in training.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

LAGS = (1, 7, 14, 28)
WINDOWS = (7, 28)

BASE_FEATURES = (
    ["item_code"]
    + [f"lag_{k}" for k in LAGS]
    + [f"rolling_mean_{w}" for w in WINDOWS]
    + [f"rolling_std_{w}" for w in WINDOWS]
    + ["sell_price", "snap", "month", "day_of_week", "week_of_year", "is_weekend",
       "event_cultural", "event_national", "event_religious", "event_sporting", "time_idx"]
)
# Extension features (used only by the LightGBM_Tweedie variant)
PRICE_FEATURES = ["price_change_pct", "price_vs_28d_mean"]
EXTENDED_FEATURES = BASE_FEATURES + PRICE_FEATURES

_CAL_COLS = ["snap", "month", "day_of_week", "week_of_year", "is_weekend",
             "event_cultural", "event_national", "event_religious", "event_sporting"]


def _window(Y: np.ndarray, t: int, w: int) -> np.ndarray:
    lo = max(0, t - w)
    win = Y[:, lo:t]
    if win.shape[1] < w:  # pad the start of the series so short histories yield NaN
        win = np.concatenate([np.full((Y.shape[0], w - win.shape[1]), np.nan), win], axis=1)
    return win


def day_features(Y: np.ndarray, P: np.ndarray, cal: np.ndarray, t: int,
                 extended: bool = False) -> np.ndarray:
    """Feature matrix ``(n_items, n_features)`` for day ``t``.

    Uses only ``Y[:, :t]`` (strictly past demand). Rolling statistics follow
    ``series.shift(1).rolling(w)`` semantics: they require ``w`` non-missing values.
    """
    n = Y.shape[0]
    cols = [np.arange(n, dtype=float)]
    for k in LAGS:
        cols.append(Y[:, t - k] if t - k >= 0 else np.full(n, np.nan))
    wins = {w: _window(Y, t, w) for w in WINDOWS}
    for w in WINDOWS:
        win = wins[w]
        full = ~np.isnan(win).any(axis=1)
        m = np.full(n, np.nan)
        m[full] = win[full].mean(axis=1)
        cols.append(m)
    for w in WINDOWS:
        win = wins[w]
        full = ~np.isnan(win).any(axis=1)
        s = np.full(n, np.nan)
        s[full] = win[full].std(axis=1, ddof=1)
        cols.append(s)
    cols.append(P[:, t])
    cols.extend(np.repeat(cal[t][None, :], n, axis=0).T)
    cols.append(np.full(n, float(t)))
    if extended:
        prev = P[:, t - 7] if t >= 7 else np.full(n, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN price window before launch
            cols.append(P[:, t] / prev - 1.0)
            cols.append(P[:, t] / np.nanmean(_window(P, t, 28), axis=1))
    return np.column_stack(cols)


def calendar_matrix(calendar: pd.DataFrame) -> np.ndarray:
    return calendar[_CAL_COLS].to_numpy(dtype=float)


def build_training_tensor(Y: np.ndarray, P: np.ndarray, cal: np.ndarray, end: int,
                          extended: bool = False) -> np.ndarray:
    """Features for every day ``0..end-1`` -> array ``(end, n_items, n_features)``.

    Computed once on the *actual* history; training for an origin ``o`` uses days ``< o``,
    which only depend on demand before ``o``.
    """
    return np.stack([day_features(Y, P, cal, t, extended) for t in range(end)])


def training_rows(tensor: np.ndarray, Y: np.ndarray, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (day, item) rows with an observed target before day ``end``."""
    X = tensor[:end].reshape(-1, tensor.shape[-1])
    y = Y[:, :end].T.reshape(-1)
    keep = ~np.isnan(y)
    return X[keep], y[keep]
