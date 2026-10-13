"""Forecast accuracy metrics on SKU-day panels (NaN actuals = inactive, ignored)."""

from __future__ import annotations

import numpy as np


def _mask(actual: np.ndarray, forecast: np.ndarray):
    m = ~np.isnan(actual) & ~np.isnan(forecast)
    return actual[m], forecast[m]


def mae(actual, forecast) -> float:
    a, f = _mask(actual, forecast)
    return float(np.mean(np.abs(a - f)))


def rmse(actual, forecast) -> float:
    a, f = _mask(actual, forecast)
    return float(np.sqrt(np.mean((a - f) ** 2)))


def wape(actual, forecast) -> float:
    """sum|D - F| / sum D (portfolio-level, robust to zero days unlike MAPE)."""
    a, f = _mask(actual, forecast)
    return float(np.sum(np.abs(a - f)) / np.sum(a))


def bias(actual, forecast) -> float:
    """sum(F - D) / sum D ; positive = over-forecasting."""
    a, f = _mask(actual, forecast)
    return float(np.sum(f - a) / np.sum(a))


def summary(actual, forecast) -> dict:
    return {"MAE": mae(actual, forecast), "RMSE": rmse(actual, forecast),
            "WAPE": wape(actual, forecast), "Bias": bias(actual, forecast)}


def sku_wape(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Per-SKU WAPE (inf where the SKU had zero demand in the window)."""
    ok = ~np.isnan(actual)
    err = np.where(ok, np.abs(actual - forecast), 0.0).sum(axis=1)
    dem = np.where(ok, actual, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dem > 0, err / dem, np.inf)
