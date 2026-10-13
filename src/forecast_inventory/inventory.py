"""Forecast-driven periodic-review order-up-to policy (R, S_t) with lost sales.

Timeline of a simulated day ``t`` (identical for every SKU and model):

1. orders due on day ``t`` arrive and are added to on-hand stock;
2. if ``t`` is a review day, the order-up-to level ``S_t`` is compared with the inventory
   position ``IP_t = on-hand + on-order`` and ``Q_t = ceil(max(0, S_t - IP_t))`` is
   ordered, arriving on day ``t + L``;
3. demand is served from on-hand stock; unmet demand is lost (no backorders);
4. holding cost is charged on end-of-day stock.

``S_t = sum_{k=0}^{H-1} forecast_{t+k} + SS`` with protection period ``H = R + L``.

The simulator is policy-agnostic: it receives the order-up-to level of every review day,
so the same engine runs the main 28-day experiment, all sensitivity scenarios and the
multi-window continuous simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class Policy:
    review_period: int = 7
    lead_time: int = 5
    service_level: float = 0.95
    holding_rate: float = 0.25
    shortage_multiplier: float = 1.0
    order_cost: float = 5.0

    @property
    def protection(self) -> int:
        return self.review_period + self.lead_time

    @property
    def z(self) -> float:
        return float(norm.ppf(self.service_level))

    def with_(self, **kw) -> "Policy":
        return replace(self, **kw)

    @classmethod
    def from_config(cls, cfg: dict) -> "Policy":
        c = cfg["inventory"]
        return cls(c["review_period"], c["lead_time"], c["service_level"], c["holding_rate"],
                   c["shortage_multiplier"], c["order_cost"])

    def cost_optimal_service_level(self) -> float:
        """Critical ratio of a lost-sales cycle: p / (p + h R / 365), price cancels out.

        A newsvendor-style, cost-consistent target instead of a fixed nominal 95 %.
        """
        p = self.shortage_multiplier
        hold = self.holding_rate * self.review_period / 365.0
        return p / (p + hold)


# ---------------------------------------------------------------- safety stock ----------

def ss_demand_std(Y: np.ndarray, origin: int, policy: Policy, window: int = 90) -> np.ndarray:
    """Thesis main setting: SS = z * sigma_demand(last `window` days) * sqrt(H), common to all models."""
    with np.errstate(invalid="ignore"):
        sigma = np.nanstd(Y[:, origin - window: origin], axis=1, ddof=1)
    return policy.z * np.nan_to_num(sigma) * np.sqrt(policy.protection)


def ss_error_std(past_actuals: list[np.ndarray], past_forecasts: list[np.ndarray],
                 policy: Policy) -> np.ndarray:
    """Model-specific SS = z * sigma_e * sqrt(H) from out-of-sample errors of earlier origins."""
    err = np.concatenate([a - f for a, f in zip(past_actuals, past_forecasts)], axis=1)
    with np.errstate(invalid="ignore"):
        sigma = np.nanstd(err, axis=1, ddof=1)
    return policy.z * np.nan_to_num(sigma) * np.sqrt(policy.protection)


def cumulative_errors(actual: np.ndarray, forecast: np.ndarray, review_days, H: int) -> np.ndarray:
    """H-day cumulative errors (actual - forecast) at each review day of a past window.

    ``actual`` / ``forecast`` are aligned on the window's days (forecast may be longer).
    Returns ``(n_items, n_reviews)``; NaN when the H-day block is not fully observed.
    """
    out = []
    for r in review_days:
        a = actual[:, r: r + H]
        f = forecast[:, r: r + H]
        if a.shape[1] < H:
            out.append(np.full(actual.shape[0], np.nan))
            continue
        out.append(np.where(np.isnan(a).any(axis=1), np.nan, a.sum(axis=1) - f.sum(axis=1)))
    return np.column_stack(out)


def ss_empirical_quantile(cum_err: np.ndarray, scale: np.ndarray, groups: np.ndarray,
                          policy: Policy) -> np.ndarray:
    """Distribution-free SS: service-level quantile of past H-day cumulative errors.

    Removes the normality and sqrt(H) independence assumptions of the textbook formula.
    Per-SKU samples are few, so errors are scaled by each SKU's mean daily demand and
    pooled within its demand class; the class quantile is rescaled back to the SKU.
    """
    scale = np.maximum(scale, 1e-6)
    rel = cum_err / scale[:, None]
    ss = np.zeros(len(scale))
    for g in np.unique(groups):
        m = groups == g
        vals = rel[m].ravel()
        vals = vals[np.isfinite(vals)]
        q = np.quantile(vals, policy.service_level) if len(vals) else 0.0
        ss[m] = max(q, 0.0) * scale[m]
    return ss


# ---------------------------------------------------------------- order-up-to ----------

def order_up_to(forecast: np.ndarray, review_days, policy: Policy, ss: np.ndarray) -> np.ndarray:
    """S for each review day from a single-origin forecast ``(n, >= max(review)+H)``."""
    H = policy.protection
    cols = [np.nansum(forecast[:, r: r + H], axis=1) + ss for r in review_days]
    return np.column_stack(cols)


# ---------------------------------------------------------------- simulation -----------

def _ffill_prices(P: np.ndarray) -> np.ndarray:
    P = P.copy()
    for t in range(1, P.shape[1]):
        m = np.isnan(P[:, t])
        P[m, t] = P[m, t - 1]
    return np.nan_to_num(P, nan=0.0)


def simulate(demand: np.ndarray, prices: np.ndarray, review_days, S: np.ndarray,
             policy: Policy, initial_on_hand: np.ndarray, keep_trajectory: bool = False,
             account_from: int = 0) -> dict:
    """Run the lost-sales (R, S_t) system for ``T = demand.shape[1]`` days.

    demand : (n, T) actual demand (NaN treated as 0)
    S      : (n, n_reviews) order-up-to level at each review day
    account_from : days before this index are a warm-up: simulated but not counted in KPIs
    """
    n, T = demand.shape
    D = np.nan_to_num(demand, nan=0.0)
    price = _ffill_prices(prices)
    L = policy.lead_time
    arrivals = np.zeros((n, T + L + 1))
    on = np.ceil(np.maximum(initial_on_hand, 0.0))
    review_pos = {int(r): j for j, r in enumerate(review_days)}
    hold = np.zeros(n)
    short = np.zeros(n)
    n_orders = np.zeros(n)
    sold_tot = np.zeros(n)
    lost_tot = np.zeros(n)
    stockout_days = np.zeros(n)
    onhand_sum = np.zeros(n)
    traj = np.zeros((n, T)) if keep_trajectory else None
    for t in range(T):
        on += arrivals[:, t]
        if t in review_pos:
            ip = on + arrivals[:, t + 1:].sum(axis=1)
            q = np.ceil(np.maximum(S[:, review_pos[t]] - ip, 0.0))
            arrivals[:, t + L] += q
            if t >= account_from:
                n_orders += q > 0
        d = D[:, t]
        sold = np.minimum(on, d)
        lost = d - sold
        on -= sold
        if keep_trajectory:
            traj[:, t] = on
        if t < account_from:
            continue
        sold_tot += sold
        lost_tot += lost
        stockout_days += lost > 0
        onhand_sum += on
        hold += on * price[:, t] * policy.holding_rate / 365.0
        short += lost * price[:, t] * policy.shortage_multiplier
    T_acc = T - account_from
    order = n_orders * policy.order_cost
    res = {
        "demand": D[:, account_from:].sum(axis=1), "sold": sold_tot, "lost": lost_tot,
        "holding_cost": hold, "shortage_cost": short, "order_cost": order,
        "total_cost": hold + short + order, "n_orders": n_orders,
        "avg_on_hand": onhand_sum / T_acc, "stockout_days": stockout_days,
        "active_days": (~np.isnan(demand[:, account_from:])).sum(axis=1),
        "revenue": (D[:, account_from:] * price[:, account_from:]).sum(axis=1),
    }
    if keep_trajectory:
        res["on_hand"] = traj
    return res


def portfolio_kpis(res: dict) -> dict:
    """Aggregate per-SKU simulation output to portfolio KPIs."""
    dem = res["demand"].sum()
    return {
        "fill_rate": float(res["sold"].sum() / dem) if dem > 0 else np.nan,
        "stockout_rate": float(res["stockout_days"].sum() / max(res["active_days"].sum(), 1)),
        "lost_units": float(res["lost"].sum()),
        "avg_on_hand": float(res["avg_on_hand"].mean()),
        "n_orders": float(res["n_orders"].sum()),
        "holding_cost": float(res["holding_cost"].sum()),
        "shortage_cost": float(res["shortage_cost"].sum()),
        "order_cost": float(res["order_cost"].sum()),
        "total_cost": float(res["total_cost"].sum()),
    }
