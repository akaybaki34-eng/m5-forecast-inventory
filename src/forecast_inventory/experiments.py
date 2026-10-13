"""All experiments, computed from cached backtest forecasts.

Every function returns plain pandas/NumPy objects; :mod:`scripts.run_all` writes them to
``reports/``. Window ``k`` means the 28-day period starting at backtest origin ``k``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import metrics
from .backtest import BacktestResult
from .data import Panel
from .inventory import (Policy, cumulative_errors, order_up_to, portfolio_kpis, simulate,
                        ss_demand_std, ss_empirical_quantile, ss_error_std)
from .segmentation import DEMAND_CLASSES

ENSEMBLE_MEMBERS = ("ETS_ANA", "RandomForest", "XGBoost", "LightGBM", "LightGBM_Tweedie")
SS_METHODS = ("demand_std", "error_std", "empirical_quantile", "cost_optimal")


@dataclass
class Ctx:
    cfg: dict
    panel: Panel
    res: BacktestResult
    policy: Policy

    @property
    def Y(self):
        return self.panel.sales

    @property
    def P(self):
        return self.panel.prices

    @property
    def H(self):
        return self.res.horizon

    @property
    def models(self) -> list[str]:
        return list(self.res.forecasts)

    def k(self, date) -> int:
        return self.res.origin_pos(date)

    def actual(self, k: int) -> np.ndarray:
        o = self.res.origins[k]
        return self.Y[:, o: o + self.H]

    def fc(self, m: str, k: int, full: bool = False) -> np.ndarray:
        f = self.res.forecasts[m][k]
        return f if full else f[:, : self.H]


def add_ensemble(res: BacktestResult, members=ENSEMBLE_MEMBERS, name: str = "Ensemble") -> None:
    """Equal-weight forecast combination with members fixed a priori (no selection on test)."""
    members = [m for m in members if m in res.forecasts]
    res.forecasts[name] = np.mean([res.forecasts[m] for m in members], axis=0)


# ------------------------------------------------------------------ accuracy -----------

def accuracy_table(ctx: Ctx, k: int, models=None, baseline: str = "SeasonalNaive") -> pd.DataFrame:
    models = models or ctx.models
    a = ctx.actual(k)
    rows = [{"model": m, **metrics.summary(a, ctx.fc(m, k))} for m in models]
    df = pd.DataFrame(rows).set_index("model")
    if baseline in df.index:
        df["WAPE_gain_vs_SN"] = 1 - df["WAPE"] / df.loc[baseline, "WAPE"]
    return df.sort_values("WAPE")


def accuracy_by_class(ctx: Ctx, k: int, models=None) -> pd.DataFrame:
    models = models or ctx.models
    a = ctx.actual(k)
    cls = ctx.res.classes[k]
    out = {}
    for m in models:
        f = ctx.fc(m, k)
        out[m] = {c: metrics.wape(a[cls == c], f[cls == c]) for c in DEMAND_CLASSES}
    return pd.DataFrame(out).T[list(DEMAND_CLASSES)]


def rolling_wape(ctx: Ctx, models=None) -> pd.DataFrame:
    models = models or ctx.models
    rows = {}
    for k, d in enumerate(ctx.res.origin_dates):
        a = ctx.actual(k)
        rows[d.date()] = {m: metrics.wape(a, ctx.fc(m, k)) for m in models}
    return pd.DataFrame(rows).T


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def rank_tests(matrix: pd.DataFrame) -> dict:
    """Friedman test across models + Wilcoxon(best vs each, Holm) over windows/SKUs (rows)."""
    mat = matrix.dropna(axis=0, how="any")
    fr = stats.friedmanchisquare(*[mat[c].to_numpy() for c in mat.columns])
    ranks = mat.rank(axis=1).mean().sort_values()
    best = ranks.index[0]
    raw = {}
    for c in mat.columns:
        if c == best:
            continue
        diff = mat[c] - mat[best]
        raw[c] = 1.0 if np.allclose(diff, 0) else float(stats.wilcoxon(mat[best], mat[c]).pvalue)
    return {"friedman_stat": float(fr.statistic), "friedman_p": float(fr.pvalue), "best": best,
            "mean_rank": ranks.to_dict(), "wilcoxon_holm_vs_best": holm(raw), "n_blocks": len(mat)}


# ------------------------------------------------------------------ hybrids ------------

def hybrid_validation_rule(ctx: Ctx, k_val: int, k_test: int, candidates) -> tuple[np.ndarray, dict]:
    """Thesis rule: per demand class pick the validation-window winner, apply to test."""
    a = ctx.actual(k_val)
    cls_val = ctx.res.classes[k_val]
    choice = {}
    for c in DEMAND_CLASSES:
        m = cls_val == c
        choice[c] = min(candidates, key=lambda mod: metrics.wape(a[m], ctx.fc(mod, k_val)[m]))
    cls_test = ctx.res.classes[k_test]
    f = np.zeros_like(ctx.fc(candidates[0], k_test, full=True))
    for c, mod in choice.items():
        f[cls_test == c] = ctx.fc(mod, k_test, full=True)[cls_test == c]
    return f, choice


def hybrid_rolling_rule(ctx: Ctx, candidates, lookback: int = 3) -> pd.DataFrame:
    """Extension: per-class selection on the mean WAPE of the previous ``lookback`` windows."""
    rows = []
    n_win = len(ctx.res.origins)
    for k in range(lookback, n_win):
        cls_k = ctx.res.classes[k]
        f = np.zeros_like(ctx.fc(candidates[0], k))
        picks = {}
        for c in DEMAND_CLASSES:
            def score(mod):
                vals = []
                for j in range(k - lookback, k):
                    m = ctx.res.classes[j] == c
                    vals.append(metrics.wape(ctx.actual(j)[m], ctx.fc(mod, j)[m]))
                return np.mean(vals)
            picks[c] = min(candidates, key=score)
            f[cls_k == c] = ctx.fc(picks[c], k)[cls_k == c]
        rows.append({"origin": ctx.res.origin_dates[k].date(), "WAPE_hybrid": metrics.wape(ctx.actual(k), f),
                     **{f"pick_{c}": p for c, p in picks.items()}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ inventory ----------

def review_days(T: int, policy: Policy) -> list[int]:
    return list(range(0, T, policy.review_period))


def safety_stock(ctx: Ctx, m: str, k: int, policy: Policy, method: str) -> np.ndarray:
    o = ctx.res.origins[k]
    icfg = ctx.cfg["inventory"]
    if method == "demand_std":
        return ss_demand_std(ctx.Y, o, policy, icfg["demand_std_window"])
    if method == "cost_optimal":
        p2 = policy.with_(service_level=policy.cost_optimal_service_level())
        return ss_demand_std(ctx.Y, o, p2, icfg["demand_std_window"])
    if method == "error_std":
        js = range(max(0, k - icfg["error_std_origins"]), k)
        js = [j for j in js if ctx.res.origins[j] + ctx.H <= o]
        if not js:
            return ss_demand_std(ctx.Y, o, policy, icfg["demand_std_window"])
        return ss_error_std([ctx.actual(j) for j in js], [ctx.fc(m, j) for j in js], policy)
    if method == "empirical_quantile":
        H = policy.protection
        blocks = []
        for j in range(max(0, k - 6), k):
            oj = ctx.res.origins[j]
            rds = [r for r in range(0, ctx.res.forecast_days - H + 1, policy.review_period) if oj + r + H <= o]
            if rds:
                blocks.append(cumulative_errors(ctx.Y[:, oj: oj + ctx.res.forecast_days],
                                                ctx.fc(m, j, full=True), rds, H))
        if not blocks:
            return ss_demand_std(ctx.Y, o, policy, icfg["demand_std_window"])
        with np.errstate(invalid="ignore"):
            scale = np.nan_to_num(np.nanmean(ctx.Y[:, o - icfg["demand_std_window"]: o], axis=1))
        return ss_empirical_quantile(np.concatenate(blocks, axis=1), scale, ctx.res.classes[k], policy)
    raise ValueError(method)


def run_window(ctx: Ctx, m: str, k: int, policy: Policy, ss_method: str = "demand_std",
               init: str = "model", forecast: np.ndarray | None = None) -> dict:
    """Single 28-day window, thesis protocol (model-specific or common initial stock)."""
    o = ctx.res.origins[k]
    T = ctx.H
    rds = review_days(T, policy)
    fc = ctx.fc(m, k, full=True) if forecast is None else forecast
    ss = safety_stock(ctx, m, k, policy, ss_method)
    S = order_up_to(fc, rds, policy, ss)
    if init == "model":
        init_stock = S[:, 0]
    else:  # same starting stock for every model
        mu = np.nan_to_num(np.nanmean(ctx.Y[:, o - 28: o], axis=1))
        init_stock = mu * policy.protection + ss_demand_std(ctx.Y, o, policy, ctx.cfg["inventory"]["demand_std_window"])
    return simulate(ctx.Y[:, o: o + T], ctx.P[:, o: o + T], rds, S, policy, init_stock)


def inventory_table(ctx: Ctx, k: int, policy: Policy, models=None, ss_method="demand_std",
                    init="model", extra_forecasts: dict | None = None) -> pd.DataFrame:
    models = models or ctx.models
    rows = {}
    for m in models:
        rows[m] = portfolio_kpis(run_window(ctx, m, k, policy, ss_method, init))
    for name, f in (extra_forecasts or {}).items():
        rows[name] = portfolio_kpis(run_window(ctx, models[0], k, policy, ss_method, init, forecast=f))
    df = pd.DataFrame(rows).T.sort_values("total_cost")
    if "SeasonalNaive" in df.index:
        df["cost_gain_vs_SN"] = 1 - df["total_cost"] / df.loc["SeasonalNaive", "total_cost"]
    return df


def continuous_simulation(ctx: Ctx, m: str, k_start: int, k_end: int, policy: Policy,
                          ss_method: str = "demand_std", keep_trajectory: bool = False) -> dict:
    """Multi-window simulation with inventory carried over between windows.

    Window ``k_start`` is a warm-up (simulated, not counted). At each review day the policy
    uses the forecast issued at the most recent origin and the safety stock computed at
    that origin, so no information from after the decision date is used.
    """
    o0 = ctx.res.origins[k_start]
    ks = list(range(k_start, k_end + 1))
    T = ctx.res.origins[k_end] + ctx.H - o0
    rds = review_days(T, policy)
    fcs = {k: ctx.fc(m, k, full=True) for k in ks}
    sss = {k: safety_stock(ctx, m, k, policy, ss_method) for k in ks}
    S = np.zeros((ctx.panel.n_items, len(rds)))
    for j, r in enumerate(rds):
        abs_day = o0 + r
        k = max(kk for kk in ks if ctx.res.origins[kk] <= abs_day)
        off = abs_day - ctx.res.origins[k]
        S[:, j] = np.nansum(fcs[k][:, off: off + policy.protection], axis=1) + sss[k]
    mu = np.nan_to_num(np.nanmean(ctx.Y[:, o0 - 28: o0], axis=1))
    init_stock = mu * policy.protection + ss_demand_std(ctx.Y, o0, policy, ctx.cfg["inventory"]["demand_std_window"])
    return simulate(ctx.Y[:, o0: o0 + T], ctx.P[:, o0: o0 + T], rds, S, policy, init_stock,
                    keep_trajectory=keep_trajectory, account_from=ctx.H)


def scenario_grid(cfg: dict, base: Policy) -> dict[str, Policy]:
    """Thesis 12 unique environments (9 lead-time x service + 3 extra shortage costs) plus
    a full holding-rate x order-cost factorial (thesis future work #10)."""
    s = cfg["sensitivity"]
    grid = {}
    for L in s["lead_times"]:
        for sl in s["service_levels"]:
            grid[f"L={L}|SL={sl:.2f}"] = base.with_(lead_time=L, service_level=sl)
    for cs in s["shortage_multipliers"]:
        if cs != base.shortage_multiplier:
            grid[f"cs={cs}"] = base.with_(shortage_multiplier=cs)
    for h in s["holding_rates"]:
        for K in s["order_costs"]:
            if h == base.holding_rate and K == base.order_cost:
                continue
            grid[f"h={h}|K={K}"] = base.with_(holding_rate=h, order_cost=K)
    return grid


def sensitivity(ctx: Ctx, models, scenarios: dict[str, Policy], runner) -> pd.DataFrame:
    """Total cost for every (scenario, model); ``runner(model, policy) -> sim result``."""
    rows = {}
    for name, pol in scenarios.items():
        rows[name] = {m: float(runner(m, pol)["total_cost"].sum()) for m in models}
    return pd.DataFrame(rows).T


def regret(cost: pd.DataFrame) -> pd.DataFrame:
    """Relative regret_{m,s} = (C_ms - min_k C_ks) / min_k C_ks."""
    best = cost.min(axis=1)
    return cost.sub(best, axis=0).div(best, axis=0)


def pareto_front(df: pd.DataFrame, x: str, y: str) -> pd.Series:
    """True where no other row is at least as good on both objectives and better on one."""
    vals = df[[x, y]].to_numpy()
    eff = []
    for i, v in enumerate(vals):
        dominated = np.any(np.all(vals <= v, axis=1) & np.any(vals < v, axis=1))
        eff.append(not dominated)
    return pd.Series(eff, index=df.index)


# ------------------------------------------------------------------ SKU level ----------

def sku_level(ctx: Ctx, k: int, policy: Policy, models) -> dict:
    a = ctx.actual(k)
    per = {}
    for m in models:
        sim = run_window(ctx, m, k, policy)
        f = ctx.fc(m, k)
        ok = ~np.isnan(a)
        dem = np.where(ok, a, 0).sum(axis=1)
        per[m] = pd.DataFrame({
            "wape": metrics.sku_wape(a, f),
            "bias": np.where(dem > 0, (np.where(ok, f - a, 0)).sum(axis=1) / np.maximum(dem, 1e-9), np.nan),
            "cost": sim["total_cost"], "norm_cost": sim["total_cost"] / np.maximum(sim["revenue"], 1e-9),
            "fill": np.where(sim["demand"] > 0, sim["sold"] / np.maximum(sim["demand"], 1e-9), np.nan),
            "abs_err": np.where(ok, np.abs(a - f), 0).sum(axis=1), "demand": dem,
        })
    spearman = {}
    for m, d in per.items():
        fin = np.isfinite(d["wape"])
        r1 = stats.spearmanr(d.loc[fin, "wape"], d.loc[fin, "norm_cost"])
        fin2 = d["bias"].notna() & d["fill"].notna()
        r2 = stats.spearmanr(d.loc[fin2, "bias"], d.loc[fin2, "fill"])
        spearman[m] = {"wape_vs_norm_cost": float(r1.statistic), "p1": float(r1.pvalue),
                       "bias_vs_fill": float(r2.statistic), "p2": float(r2.pvalue)}
    W = pd.DataFrame({m: d["wape"] for m, d in per.items()})
    C = pd.DataFrame({m: d["cost"] for m, d in per.items()})
    fin = np.isfinite(W).all(axis=1)
    coincide = float((W[fin].idxmin(axis=1) == C[fin].idxmin(axis=1)).mean())
    return {"per_sku": per, "spearman": spearman, "winner_coincidence": coincide,
            "n_finite": int(fin.sum()), "cost_tests": rank_tests(C), "wape_tests": rank_tests(W[fin])}


def cluster_bootstrap(per: dict, a: str, b: str, reps: int, seed: int) -> dict:
    """SKU-cluster bootstrap of portfolio WAPE(a) - WAPE(b) and cost(a) - cost(b)."""
    rng = np.random.default_rng(seed)
    n = len(per[a])
    idx = rng.integers(0, n, size=(reps, n))
    ea, eb = per[a]["abs_err"].to_numpy(), per[b]["abs_err"].to_numpy()
    dem = per[a]["demand"].to_numpy()
    ca, cb = per[a]["cost"].to_numpy(), per[b]["cost"].to_numpy()
    D = dem[idx].sum(axis=1)
    dw = ea[idx].sum(axis=1) / D - eb[idx].sum(axis=1) / D
    dc = ca[idx].sum(axis=1) - cb[idx].sum(axis=1)

    def q(v):
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

    return {"a": a, "b": b, "wape_diff_point": float(ea.sum() / dem.sum() - eb.sum() / dem.sum()),
            "wape_diff_ci95": q(dw), "cost_diff_point": float(ca.sum() - cb.sum()),
            "cost_diff_ci95": q(dc), "p_cost_a_lower": float((dc < 0).mean()), "reps": reps}


# ------------------------------------------------------------------ explainability -----

def shap_summary(explain: dict) -> pd.DataFrame:
    if not explain:
        return pd.DataFrame()
    sh = np.abs(explain["shap"]).mean(axis=0)
    df = pd.DataFrame({"feature": explain["features"], "mean_abs_shap": sh})
    df["share"] = df["mean_abs_shap"] / df["mean_abs_shap"].sum()
    rho = []
    for j in range(len(explain["features"])):
        x, s = explain["X"][:, j], explain["shap"][:, j]
        ok = np.isfinite(x)
        rho.append(float(stats.spearmanr(x[ok], s[ok]).statistic) if ok.sum() > 10 and np.std(x[ok]) > 0 else np.nan)
    df["spearman_value_vs_shap"] = rho
    return df.sort_values("share", ascending=False).reset_index(drop=True)
