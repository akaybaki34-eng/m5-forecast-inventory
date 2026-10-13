"""Global tree-ensemble models and the recursive multi-step forecaster."""

from __future__ import annotations

import warnings

import numpy as np

from ..features import day_features

# LightGBM names numpy columns internally; sklearn then warns on every predict call.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

ML_MODELS = ("RandomForest", "XGBoost", "LightGBM", "LightGBM_Tweedie")


def uses_extended_features(name: str) -> bool:
    return name == "LightGBM_Tweedie"


def make_model(name: str, cfg: dict):
    params = dict(cfg["models"].get(name, {}))
    seed = cfg["models"]["random_state"]
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    if name == "XGBoost":
        from xgboost import XGBRegressor

        return XGBRegressor(tree_method="hist", objective="reg:squarederror", random_state=seed,
                            n_jobs=-1, **params)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **params)
    if name == "LightGBM_Tweedie":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(objective="tweedie", random_state=seed, n_jobs=-1, verbose=-1, **params)
    raise ValueError(f"unknown ML model {name}")


def fit_model(name: str, X: np.ndarray, y: np.ndarray, cfg: dict):
    model = make_model(name, cfg)
    if name == "LightGBM_Tweedie":
        # item identity as a native categorical instead of an ordinal code (thesis future work #9)
        model.fit(X, y, categorical_feature=[0])
    else:
        model.fit(X, y)
    return model


def recursive_forecast(model, Y: np.ndarray, P: np.ndarray, cal: np.ndarray, origin: int,
                       horizon: int, extended: bool = False, keep_features: bool = False):
    """Fixed-origin recursive forecast for days ``origin .. origin+horizon-1``.

    Demand on and after ``origin`` is blanked out before the loop; each day's forecast is
    written back into the working history and feeds the lag/rolling features of later
    days. Only exogenous variables known in advance (price, calendar) are read for future
    days. Returns ``(n_items, horizon)``; days beyond the data end are NaN.
    """
    n, T = Y.shape
    h_eff = min(horizon, T - origin)
    work = Y[:, : origin + h_eff].copy()
    work[:, origin:] = np.nan
    out = np.full((n, horizon), np.nan)
    feats = []
    for k in range(h_eff):
        t = origin + k
        X = day_features(work, P, cal, t, extended)
        pred = np.clip(model.predict(X.astype(np.float32)), 0.0, None)
        active = ~np.isnan(P[:, t])
        pred = np.where(active, pred, 0.0)
        out[:, k] = pred
        work[:, t] = np.where(active, pred, np.nan)
        if keep_features:
            feats.append(X)
    return (out, np.stack(feats)) if keep_features else out
