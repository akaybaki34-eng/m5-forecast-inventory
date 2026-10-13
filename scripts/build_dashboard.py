"""Build reports/dashboard.html: a self-contained interactive results page (data embedded)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "reports" / "tables"
TEMPLATE = ROOT / "scripts" / "dashboard_template.html"
OUT = ROOT / "reports" / "dashboard.html"

MODELS = ["SeasonalNaive", "MovingAverage28", "ETS_ANA", "Croston_SBA", "TSB", "RandomForest", "XGBoost",
          "LightGBM", "LightGBM_Tweedie", "Ensemble"]


def r(x, d=4):
    return None if pd.isna(x) else round(float(x), d)


def main():
    res = json.loads((ROOT / "reports" / "results.json").read_text())
    acc = pd.read_csv(T / "accuracy_test.csv", index_col=0)
    m5 = pd.read_csv(T / "accuracy_m5_eval.csv", index_col=0)
    roll = pd.read_csv(T / "accuracy_rolling_wape.csv", index_col=0)
    rsum = pd.read_csv(T / "accuracy_rolling_summary.csv", index_col=0)
    inv = pd.read_csv(T / "inventory_test_main.csv", index_col=0)
    cont = pd.read_csv(T / "continuous_sim.csv", index_col=[0, 1])
    sens = pd.read_csv(T / "sensitivity_continuous_cost.csv", index_col=0)
    shap = pd.read_csv(T / "shap_XGBoost.csv", index_col=0)

    data = {
        "models": MODELS,
        "accuracy": {m: {"wape_test": r(acc.loc[m, "WAPE"]), "bias_test": r(acc.loc[m, "Bias"]),
                         "mae_test": r(acc.loc[m, "MAE"], 3), "wape_m5": r(m5.loc[m, "WAPE"]),
                         "wape_roll": r(rsum.loc[m, "mean_WAPE"]), "wins": int(rsum.loc[m, "wins"])}
                     for m in MODELS},
        "inventory_test": {m: {k: r(inv.loc[m, k], 4) for k in
                               ["fill_rate", "lost_units", "avg_on_hand", "holding_cost", "shortage_cost",
                                "order_cost", "total_cost"]} for m in MODELS},
        "rolling": {"dates": list(roll.index), "series": {m: [r(v) for v in roll[m]] for m in MODELS}},
        "continuous": {m: {s: {"cost": r(cont.loc[(m, s), "total_cost"], 1), "fill": r(cont.loc[(m, s), "fill_rate"]),
                               "stock": r(cont.loc[(m, s), "avg_on_hand"], 2)}
                           for s in ["demand_std", "error_std", "cost_optimal", "empirical_quantile"]} for m in MODELS},
        "scenarios": {sc: {m: r(sens.loc[sc, m], 1) for m in MODELS} for sc in sens.index},
        "shap": [{"f": f, "s": r(s)} for f, s in zip(shap["feature"].head(10), shap["share"].head(10))],
        "meta": {"n_sku": res["scope"]["n_sku"], "rows": res["scope"]["active_rows_1941"],
                 "cont_start": res["continuous"]["start"], "cont_end": res["continuous"]["end"],
                 "days": res["continuous"]["days_counted"],
                 "boot": res["sku_level"]["bootstrap"][0],
                 "coincide": res["sku_level"]["winner_coincidence"],
                 "classes": res["scope"]["classes_at_test_origin"]},
    }
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    OUT.write_text(html)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
