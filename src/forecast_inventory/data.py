"""M5 data loading and the wide SKU x day panel used everywhere else.

Design choice: all downstream code works on dense ``(n_sku, n_days)`` NumPy arrays rather
than long data frames. Feature generation, recursive forecasting and the inventory
simulator are then simple, fast, vectorised array operations, and the *same* feature
function is used for training and for recursive inference (no train/serve skew).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve

EVENT_TYPES = ("Cultural", "National", "Religious", "Sporting")


@dataclass
class Panel:
    """Daily demand panel for one store-department.

    sales : (n, T) float array, NaN before a SKU becomes active (structural zeros removed)
    prices: (n, T) float array, NaN where no selling price exists
    """

    item_ids: np.ndarray
    dates: pd.DatetimeIndex
    sales: np.ndarray
    prices: np.ndarray
    calendar: pd.DataFrame
    active_start: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_days(self) -> int:
        return len(self.dates)

    def day_index(self, date: str | pd.Timestamp) -> int:
        loc = self.dates.get_indexer([pd.Timestamp(date)])[0]
        if loc < 0:
            raise KeyError(f"{date} not in panel dates")
        return int(loc)

    def subset(self, mask_or_idx) -> "Panel":
        idx = np.asarray(mask_or_idx)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        return Panel(
            item_ids=self.item_ids[idx],
            dates=self.dates,
            sales=self.sales[idx],
            prices=self.prices[idx],
            calendar=self.calendar,
            active_start=self.active_start[idx],
            meta=dict(self.meta),
        )


def build_calendar(cal: pd.DataFrame, state: str) -> pd.DataFrame:
    """Known-in-advance calendar features (dates, SNAP, events)."""
    out = pd.DataFrame(index=pd.to_datetime(cal["date"]))
    out.index.name = "date"
    # Some M5 mirrors drop the "d" key; calendar rows are in d_1, d_2, ... order.
    out["d"] = cal["d"].to_numpy() if "d" in cal else [f"d_{i + 1}" for i in range(len(cal))]
    out["wm_yr_wk"] = cal["wm_yr_wk"].to_numpy()
    out["day_of_week"] = out.index.dayofweek
    out["month"] = out.index.month
    out["week_of_year"] = out.index.isocalendar().week.astype(int).to_numpy()
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["snap"] = cal[f"snap_{state}"].to_numpy()
    for et in EVENT_TYPES:
        hit = (cal["event_type_1"] == et) | (cal["event_type_2"] == et)
        out[f"event_{et.lower()}"] = hit.astype(int).to_numpy()
    return out


def load_store(cfg: dict, store_id: str | None = None, cat_prefix: str = "FOODS") -> Panel:
    """Load all SKUs of one store whose dept starts with ``cat_prefix`` (cached)."""
    dcfg = cfg["data"]
    store_id = store_id or dcfg["store_id"]
    raw = resolve(dcfg["raw_dir"])
    cache_dir = resolve(dcfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"panel_{store_id}_{cat_prefix}.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            return pickle.load(fh)

    cal = pd.read_csv(raw / "calendar.csv")
    state = store_id.split("_")[0]
    calendar = build_calendar(cal, state)

    train = pd.read_csv(raw / "sales_train_evaluation.csv")
    train = train[(train["store_id"] == store_id) & train["dept_id"].str.startswith(cat_prefix)]
    test_path = raw / "sales_test_evaluation.csv"
    day_cols = [c for c in train.columns if c.startswith("d_")]
    if test_path.exists():  # official M5 evaluation window d_1942..d_1969 (released after the competition)
        test = pd.read_csv(test_path)
        test = test[(test["store_id"] == store_id) & test["dept_id"].str.startswith(cat_prefix)]
        test_cols = [c for c in test.columns if c.startswith("d_")]
        train = train.merge(test[["item_id"] + test_cols], on="item_id", how="left")
        day_cols = day_cols + test_cols
    train = train.sort_values("item_id").reset_index(drop=True)
    n_days = len(day_cols)
    calendar = calendar.iloc[:n_days]

    prices = pd.read_csv(raw / "sell_prices.csv", dtype={"wm_yr_wk": np.int32, "sell_price": np.float32})
    prices = prices[prices["store_id"] == store_id]
    price_wide = prices.pivot(index="item_id", columns="wm_yr_wk", values="sell_price")
    price_wide = price_wide.reindex(train["item_id"])
    wk = calendar["wm_yr_wk"].to_numpy()
    P = price_wide.reindex(columns=np.unique(wk)).to_numpy(dtype=float)
    col_of_wk = {w: i for i, w in enumerate(np.unique(wk))}
    P = P[:, [col_of_wk[w] for w in wk]]

    Y = train[day_cols].to_numpy(dtype=float)
    # Activity proxy: first day with a valid selling price. Earlier zeros are structural.
    has_price = ~np.isnan(P)
    active_start = np.where(has_price.any(axis=1), has_price.argmax(axis=1), n_days)
    t_idx = np.arange(n_days)[None, :]
    Y[t_idx < active_start[:, None]] = np.nan

    panel = Panel(
        item_ids=train["item_id"].to_numpy(),
        dates=pd.DatetimeIndex(calendar.index),
        sales=Y,
        prices=P,
        calendar=calendar.reset_index(),
        active_start=active_start,
        meta={"store_id": store_id, "dept_ids": train["dept_id"].to_numpy(), "n_train_days": 1941},
    )
    with open(cache, "wb") as fh:
        pickle.dump(panel, fh)
    return panel


def department_summary(panel: Panel, end: int | None = None) -> pd.DataFrame:
    """SKU count, unit sales and median zero-share per department (scope-selection table)."""
    end = end or panel.meta.get("n_train_days", panel.n_days)
    Y = panel.sales[:, :end]
    rows = []
    for dept in np.unique(panel.meta["dept_ids"]):
        m = panel.meta["dept_ids"] == dept
        y = Y[m]
        active = ~np.isnan(y)
        zero_share = (y == 0).sum(axis=1) / np.maximum(active.sum(axis=1), 1)
        rows.append({"dept_id": dept, "n_sku": int(m.sum()), "unit_sales": float(np.nansum(y)),
                     "median_zero_share": float(np.median(zero_share))})
    return pd.DataFrame(rows).sort_values("unit_sales", ascending=False).reset_index(drop=True)
