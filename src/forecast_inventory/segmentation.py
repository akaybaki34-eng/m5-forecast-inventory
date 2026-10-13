"""Economic prioritisation (ABC) and demand-pattern segmentation (ADI-CV^2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEMAND_CLASSES = ("smooth", "erratic", "intermittent", "lumpy")


def abc_classify(sales: np.ndarray, prices: np.ndarray, end: int, start: int = 0,
                 a_cut: float = 0.80, b_cut: float = 0.95) -> pd.DataFrame:
    """Revenue-based ABC on days ``[start, end)``.

    RV_i = sum_t D_it * P_it ; items sorted by RV, cumulative share CR_i decides the class:
    A if CR_i <= a_cut, B if CR_i <= b_cut, else C.
    """
    rev = np.nansum(sales[:, start:end] * prices[:, start:end], axis=1)
    order = np.argsort(-rev, kind="stable")
    cum = np.cumsum(rev[order]) / rev.sum()
    cls = np.empty(len(rev), dtype=object)
    cls[order] = np.where(cum <= a_cut, "A", np.where(cum <= b_cut, "B", "C"))
    cum_share = np.empty(len(rev))
    cum_share[order] = cum
    return pd.DataFrame({"revenue": rev, "cum_share": cum_share, "abc": cls})


def adi_cv2(sales: np.ndarray, end: int, start: int = 0) -> pd.DataFrame:
    """ADI = active periods / periods with positive demand ; CV^2 of positive demand sizes."""
    y = sales[:, start:end]
    active = ~np.isnan(y)
    pos = np.where(active, y, 0) > 0
    n_active = active.sum(axis=1)
    n_pos = pos.sum(axis=1)
    adi = np.where(n_pos > 0, n_active / np.maximum(n_pos, 1), np.inf)
    ypos = np.where(pos, y, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.nanmean(ypos, axis=1)
        sd = np.nanstd(ypos, axis=1)
        cv2 = (sd / mu) ** 2
    cv2 = np.nan_to_num(cv2, nan=0.0)
    return pd.DataFrame({"adi": adi, "cv2": cv2, "n_active": n_active, "n_pos": n_pos})


def classify_demand(adi: np.ndarray, cv2: np.ndarray, adi_cut: float = 1.32,
                    cv2_cut: float = 0.49) -> np.ndarray:
    """Syntetos-Boylan-Croston (2005) quadrants."""
    adi, cv2 = np.asarray(adi), np.asarray(cv2)
    return np.select(
        [(adi < adi_cut) & (cv2 < cv2_cut), (adi < adi_cut) & (cv2 >= cv2_cut),
         (adi >= adi_cut) & (cv2 < cv2_cut)],
        ["smooth", "erratic", "intermittent"],
        default="lumpy",
    )


def demand_classes(sales: np.ndarray, end: int, cfg: dict) -> np.ndarray:
    """Demand class of every SKU using only data strictly before day ``end``."""
    s = adi_cv2(sales, end)
    seg = cfg["segmentation"]
    return classify_demand(s["adi"].to_numpy(), s["cv2"].to_numpy(), seg["adi_cutoff"], seg["cv2_cutoff"])
