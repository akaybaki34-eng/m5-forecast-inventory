"""Report figures (matplotlib, one consistent style)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LABEL = {
    "SeasonalNaive": "Seasonal Naive", "MovingAverage28": "Moving Avg 28", "ETS_ANA": "ETS(A,N,A)",
    "Croston_SBA": "Croston-SBA", "TSB": "TSB", "RandomForest": "Random Forest", "XGBoost": "XGBoost",
    "LightGBM": "LightGBM", "LightGBM_Tweedie": "LightGBM-Tweedie", "Ensemble": "Ensemble",
    "Hybrid_val": "Class hybrid (val.)",
}
COLOR = {
    "SeasonalNaive": "#9AA0A6", "MovingAverage28": "#BDC1C6", "ETS_ANA": "#5F6368",
    "Croston_SBA": "#C5B9A8", "TSB": "#A89A85", "RandomForest": "#1A7F64", "XGBoost": "#C2410C",
    "LightGBM": "#2563EB", "LightGBM_Tweedie": "#7C3AED", "Ensemble": "#0F172A",
    "Hybrid_val": "#D97706",
}
SS_LABEL = {"demand_std": "Demand σ (thesis)", "error_std": "Model error σ",
            "empirical_quantile": "Empirical quantile", "cost_optimal": "Cost-optimal SL"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "legend.frameon": False,
    "font.family": "DejaVu Sans",
})


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def bar_metric(df: pd.DataFrame, col: str, path: Path, title: str, pct: bool = True, xlabel: str = ""):
    d = df.sort_values(col, ascending=False)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(d) + 1.2))
    vals = d[col] * (100 if pct else 1)
    ax.barh([LABEL.get(i, i) for i in d.index], vals, color=[COLOR.get(i, "#888") for i in d.index])
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v:.2f}{'%' if pct else ''}" if pct else f" {v:,.0f}", va="center", fontsize=9)
    ax.set_xlim(vals.min() * 0.9 if pct else 0, vals.max() * 1.08)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0)
    _save(fig, path)


def rolling_lines(wape: pd.DataFrame, path: Path, models, highlight: str | None = None):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = [pd.Timestamp(i) for i in wape.index]
    for m in models:
        ax.plot(x, wape[m] * 100, marker="o", ms=3.5, lw=1.8 if m in ("XGBoost", "LightGBM_Tweedie", "ETS_ANA") else 1.2,
                color=COLOR.get(m), label=LABEL.get(m, m))
    if highlight:
        ax.axvspan(pd.Timestamp(highlight) - pd.Timedelta(days=6), pd.Timestamp(highlight) + pd.Timedelta(days=6),
                   color="#F59E0B", alpha=0.15, lw=0)
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Rolling-origin accuracy: 14 consecutive 28-day windows")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    _save(fig, path)


def pareto_scatter(df: pd.DataFrame, x: str, y: str, front: pd.Series, path: Path, title: str,
                   xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    xr = (df[x].max() - df[x].min()) * 100 or 1
    yr = (df[y].max() - df[y].min()) or 1
    placed = []
    for m, r in df.sort_values(x).iterrows():
        px, py = r[x] * 100, r[y]
        ax.scatter(px, py, s=70 if front[m] else 40, color=COLOR.get(m, "#888"),
                   edgecolor="black" if front[m] else "none", zorder=3)
        crowded = any(abs(px - qx) / xr < 0.12 and abs(py - qy) / yr < 0.06 for qx, qy in placed)
        ax.annotate(LABEL.get(m, m), (px, py), textcoords="offset points", xytext=(6, -12 if crowded else 4),
                    fontsize=8.5)
        placed.append((px, py))
    f = df[front].sort_values(x)
    ax.plot(f[x] * 100, f[y], ls="--", color="#111", lw=1, zorder=2, label="Pareto front")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    _save(fig, path)


def cost_breakdown(df: pd.DataFrame, path: Path, title: str):
    d = df.sort_values("total_cost")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = [LABEL.get(i, i) for i in d.index]
    bottom = np.zeros(len(d))
    for col, c, lab in [("holding_cost", "#60A5FA", "Holding"), ("shortage_cost", "#F87171", "Lost sales"),
                        ("order_cost", "#CBD5E1", "Ordering")]:
        ax.bar(names, d[col], bottom=bottom, color=c, label=lab)
        bottom += d[col].to_numpy()
    ax.set_ylabel("Cost units")
    ax.set_title(title)
    ax.legend(ncol=3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, path)


def ss_methods_plot(df: pd.DataFrame, path: Path, title: str):
    """df: rows (model, ss_method) with fill_rate and total_cost."""
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"demand_std": "o", "error_std": "s", "empirical_quantile": "D", "cost_optimal": "^"}
    for (m, s), r in df.iterrows():
        ax.scatter(r["fill_rate"] * 100, r["total_cost"], marker=markers[s], color=COLOR.get(m, "#888"), s=55,
                   edgecolor="white", lw=0.6)
    for s, mk in markers.items():
        ax.scatter([], [], marker=mk, color="#444", label=SS_LABEL[s])
    for m in df.index.get_level_values(0).unique():
        ax.scatter([], [], marker="o", color=COLOR.get(m, "#888"), label=LABEL.get(m, m))
    ax.set_xlabel("Fill rate (%)")
    ax.set_ylabel("Total relevant cost")
    ax.set_title(title)
    ax.legend(fontsize=7.5, ncol=2, loc="lower left")
    _save(fig, path)


def shap_bar(df: pd.DataFrame, path: Path, title: str, top: int = 12):
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(d["feature"], d["share"] * 100, color="#C2410C")
    for y, v in enumerate(d["share"] * 100):
        ax.text(v, y, f" {v:.1f}%", va="center", fontsize=8.5)
    ax.set_xlabel("Share of mean |SHAP| (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0)
    _save(fig, path)


def regret_heatmap(reg: pd.DataFrame, path: Path, title: str):
    cols = list(reg.mean().sort_values().index)
    d = reg[cols] * 100
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 2.5, 0.32 * len(d) + 1.6))
    im = ax.imshow(d.to_numpy(), cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(5, np.nanpercentile(d, 95)))
    ax.set_xticks(range(len(cols)), [LABEL.get(c, c) for c in cols], rotation=35, ha="right")
    ax.set_yticks(range(len(d)), d.index, fontsize=8)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.iat[i, j]
            ax.text(j, i, "0" if v < 0.005 else f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color="black" if v < 8 else "white")
    ax.grid(False)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Regret vs scenario best (%)", shrink=0.7)
    _save(fig, path)


def trajectory(dates, demand, on_hand, path: Path, title: str, warmup: int):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.axvspan(dates[0], dates[warmup - 1], color="#E2E8F0", alpha=0.6, zorder=0, lw=0, label="Warm-up (not counted)")
    ax.bar(dates, demand, color="#94A3B8", width=1.0, label="Daily demand")
    ax.plot(dates, on_hand, color="#1A7F64", lw=1.6, label="End-of-day stock")
    ax.set_ylim(0, max(np.max(on_hand), np.max(demand)) * 1.25)
    ax.set_title(title)
    ax.set_ylabel("Units")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, path)
