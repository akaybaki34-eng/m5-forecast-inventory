"""Rolling-origin backtest: produce every model's forecast at every origin (cached).

Protocol (identical for every model and origin ``o``):
  * training / fitting data: days strictly before ``o``;
  * ML models are re-trained at each origin (global models over all SKUs);
  * forecasts are fixed-origin and recursive for ``horizon + extra_forecast_days`` days;
  * demand classes are recomputed per origin from data before ``o``.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve
from .data import Panel, load_store
from .features import (BASE_FEATURES, EXTENDED_FEATURES, build_training_tensor, calendar_matrix,
                       training_rows)
from .models.baselines import CLASSICAL
from .models.ml import fit_model, recursive_forecast, uses_extended_features
from .segmentation import abc_classify, demand_classes


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass
class BacktestResult:
    item_ids: np.ndarray
    origin_dates: list[pd.Timestamp]
    origins: list[int]
    horizon: int
    forecast_days: int
    forecasts: dict[str, np.ndarray]          # model -> (n_origins, n_items, forecast_days)
    classes: np.ndarray                       # (n_origins, n_items) demand class as of origin
    explain: dict = field(default_factory=dict)
    importances: dict = field(default_factory=dict)
    fit_seconds: dict = field(default_factory=dict)
    portfolio_mode: str = "expost"

    def origin_pos(self, date) -> int:
        return self.origin_dates.index(pd.Timestamp(date))


def select_portfolio(cfg: dict, mode: str | None = None) -> tuple[Panel, dict]:
    """Store -> department -> ABC class A portfolio.

    ``expost``: ABC on the full 1,941-day history (thesis; look-ahead in scope selection).
    ``prospective``: ABC only on days before the first backtest origin (no look-ahead).
    """
    dcfg = cfg["data"]
    mode = mode or dcfg["abc_mode"]
    store = load_store(cfg)
    dept = store.subset(store.meta["dept_ids"] == dcfg["dept_id"])
    if mode == "expost":
        end = dept.meta["n_train_days"]
    elif mode == "prospective":
        end = dept.day_index(cfg["backtest"]["first_rolling_origin"])
    else:
        raise ValueError(mode)
    abc = abc_classify(dept.sales, dept.prices, end, a_cut=dcfg["abc_a_cutoff"], b_cut=dcfg["abc_b_cutoff"])
    portfolio = dept.subset(abc["abc"].to_numpy() == "A")
    info = {"mode": mode, "abc_end_day": int(end), "abc_table": abc.assign(item_id=dept.item_ids)}
    return portfolio, info


def backtest_origins(panel: Panel, cfg: dict) -> list[pd.Timestamp]:
    b = cfg["backtest"]
    first = pd.Timestamp(b["first_rolling_origin"])
    dates = [first + pd.Timedelta(days=b["horizon"] * k) for k in range(b["n_rolling_origins"])]
    extra = pd.Timestamp(b["m5_eval_origin"])
    if extra not in dates and extra in panel.dates:
        dates.append(extra)
    return dates


def run_backtest(cfg: dict, models: list[str] | None = None, portfolio_mode: str | None = None,
                 origin_dates: list | None = None, cache_name: str | None = None,
                 explain_origin: str | None = None) -> BacktestResult:
    portfolio, info = select_portfolio(cfg, portfolio_mode)
    mode = info["mode"]
    models = models or (cfg["models"]["classical"] + cfg["models"]["ml"])
    origin_dates = [pd.Timestamp(d) for d in (origin_dates or backtest_origins(portfolio, cfg))]
    cache_dir = resolve(cfg["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / (cache_name or f"backtest_{mode}.pkl")
    if cache.exists():
        with open(cache, "rb") as fh:
            res: BacktestResult = pickle.load(fh)
        if set(models) <= set(res.forecasts) and [pd.Timestamp(d) for d in res.origin_dates] == origin_dates:
            log(f"loaded cached backtest {cache.name}")
            return res

    Y, P = portfolio.sales, portfolio.prices
    cal = calendar_matrix(portfolio.calendar)
    horizon = cfg["backtest"]["horizon"]
    fdays = horizon + cfg["backtest"]["extra_forecast_days"]
    origins = [portfolio.day_index(d) for d in origin_dates]
    n = portfolio.n_items
    explain_origin = pd.Timestamp(explain_origin or cfg["backtest"]["test_origin"])

    log(f"portfolio={mode} n_items={n} origins={len(origins)} models={models}")
    tensors = {}
    if any(m not in CLASSICAL for m in models):
        tensors[False] = build_training_tensor(Y, P, cal, portfolio.n_days, extended=False)
        if any(uses_extended_features(m) for m in models):
            tensors[True] = build_training_tensor(Y, P, cal, portfolio.n_days, extended=True)

    forecasts = {m: np.full((len(origins), n, fdays), np.nan) for m in models}
    classes = np.empty((len(origins), n), dtype=object)
    explain, importances, fit_seconds = {}, {}, {m: [] for m in models}
    for k, (o, od) in enumerate(zip(origins, origin_dates)):
        classes[k] = demand_classes(Y, o, cfg)
        for m in models:
            t0 = time.time()
            if m in CLASSICAL:
                forecasts[m][k] = CLASSICAL[m](Y[:, :o], fdays)
            else:
                ext = uses_extended_features(m)
                X, y = training_rows(tensors[ext], Y, o)
                model = fit_model(m, X.astype(np.float32), y, cfg)
                want = od == explain_origin
                out = recursive_forecast(model, Y, P, cal, o, fdays, extended=ext, keep_features=want)
                if want:
                    fc, feats = out
                    explain[m] = _explain(model, m, feats[:horizon], cfg, ext)
                    importances[m] = _native_importance(model, ext)
                else:
                    fc = out
                forecasts[m][k] = fc
            fit_seconds[m].append(time.time() - t0)
        log(f"origin {od.date()} done ({k + 1}/{len(origins)})")

    res = BacktestResult(portfolio.item_ids, origin_dates, origins, horizon, fdays, forecasts,
                         classes, explain, importances, fit_seconds, mode)
    with open(cache, "wb") as fh:
        pickle.dump(res, fh)
    return res


def _native_importance(model, extended: bool) -> dict:
    names = EXTENDED_FEATURES if extended else BASE_FEATURES
    imp = np.asarray(model.feature_importances_, dtype=float)
    imp = imp / imp.sum() if imp.sum() > 0 else imp
    return dict(zip(names, imp.tolist()))


def _explain(model, name: str, feats: np.ndarray, cfg: dict, extended: bool) -> dict:
    """TreeSHAP contributions on a random sample of the test-horizon feature rows.

    The rows are exactly the inputs the recursive forecaster used (lags/rollings partly
    built from its own forecasts). Uses the libraries' built-in exact TreeSHAP.
    """
    names = EXTENDED_FEATURES if extended else BASE_FEATURES
    X = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
    rng = np.random.default_rng(cfg["models"]["random_state"])
    idx = rng.choice(len(X), size=min(cfg["analysis"]["shap_sample"], len(X)), replace=False)
    Xs = X[idx]
    if name == "XGBoost":
        import xgboost as xgb

        contrib = model.get_booster().predict(xgb.DMatrix(Xs), pred_contribs=True)
    elif name.startswith("LightGBM"):
        contrib = model.predict(Xs, pred_contrib=True)
    else:
        return {}
    return {"features": names, "X": Xs, "shap": contrib[:, :-1], "base": contrib[:, -1]}
