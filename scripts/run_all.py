"""Run the full study end-to-end and write every table, figure and results.json.

    python scripts/run_all.py                # uses configs/default.toml
    python scripts/run_all.py --config my.toml --skip-prospective
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forecast_inventory import experiments as ex  # noqa: E402
from forecast_inventory import plots  # noqa: E402
from forecast_inventory.backtest import log, run_backtest, select_portfolio  # noqa: E402
from forecast_inventory.config import load_config  # noqa: E402
from forecast_inventory.data import department_summary, load_store  # noqa: E402
from forecast_inventory.inventory import Policy, portfolio_kpis  # noqa: E402
from forecast_inventory.segmentation import DEMAND_CLASSES  # noqa: E402

TABLES = ROOT / "reports" / "tables"
FIGS = ROOT / "reports" / "figures"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return {str(i): jsonable(r.to_dict()) for i, r in o.iterrows()}
    if isinstance(o, pd.Series):
        return jsonable(o.to_dict())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return o


def save(df: pd.DataFrame, name: str):
    df.to_csv(TABLES / f"{name}.csv", float_format="%.6g")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-prospective", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    out: dict = {}

    # ------------------------------------------------------------ scope ------------------
    store = load_store(cfg)
    first_origin = store.day_index(cfg["backtest"]["first_rolling_origin"])
    dept_full = department_summary(store)
    dept_pre = department_summary(store, end=first_origin)
    save(dept_full, "scope_departments_full_history")
    save(dept_pre, "scope_departments_pre_backtest")
    portfolio, info = select_portfolio(cfg, "expost")
    abc = info["abc_table"]
    abc_summary = abc.groupby("abc").agg(n_sku=("revenue", "size"), revenue=("revenue", "sum"))
    save(abc_summary, "scope_abc")
    res = run_backtest(cfg)
    ex.add_ensemble(res)
    policy = Policy.from_config(cfg)
    ctx = ex.Ctx(cfg, portfolio, res, policy)
    k_val, k_test, k_m5 = (ctx.k(cfg["backtest"][x]) for x in ("validation_origin", "test_origin", "m5_eval_origin"))
    thesis = cfg["models"]["thesis_models"]
    base_models = cfg["models"]["classical"] + cfg["models"]["ml"]
    all_models = base_models + ["Ensemble"]
    cls_test = pd.Series(res.classes[k_test]).value_counts().reindex(DEMAND_CLASSES)
    cls_val = pd.Series(res.classes[k_val]).value_counts().reindex(DEMAND_CLASSES)
    Y = portfolio.sales
    out["scope"] = {
        "store": cfg["data"]["store_id"], "dept": cfg["data"]["dept_id"],
        "departments_full": dept_full, "departments_pre_backtest": dept_pre,
        "abc": abc_summary, "n_sku": portfolio.n_items,
        "active_rows_1941": int(np.isfinite(Y[:, :1941]).sum()),
        "classes_at_validation_origin": cls_val, "classes_at_test_origin": cls_test,
        "origins": [d.date().isoformat() for d in res.origin_dates],
        "test_origin": cfg["backtest"]["test_origin"], "m5_eval_origin": cfg["backtest"]["m5_eval_origin"],
    }
    log("scope done")

    # ------------------------------------------------------------ accuracy ---------------
    f_hyb, hyb_choice = ex.hybrid_validation_rule(ctx, k_val, k_test, thesis)
    res.forecasts["Hybrid_val"] = np.zeros_like(res.forecasts["Ensemble"])
    res.forecasts["Hybrid_val"][k_test] = f_hyb
    acc_test = ex.accuracy_table(ctx, k_test, all_models + ["Hybrid_val"])
    acc_val = ex.accuracy_table(ctx, k_val, all_models)
    acc_m5 = ex.accuracy_table(ctx, k_m5, all_models)
    acc_cls = ex.accuracy_by_class(ctx, k_test, all_models)
    for df, n in [(acc_test, "accuracy_test"), (acc_val, "accuracy_validation"), (acc_m5, "accuracy_m5_eval"),
                  (acc_cls, "accuracy_test_by_class")]:
        save(df, n)
    roll = ex.rolling_wape(ctx, all_models)
    save(roll, "accuracy_rolling_wape")
    roll_summary = pd.DataFrame({"mean_WAPE": roll.mean(), "std_WAPE": roll.std(),
                                 "wins": roll.idxmin(axis=1).value_counts().reindex(roll.columns).fillna(0)}).sort_values("mean_WAPE")
    save(roll_summary, "accuracy_rolling_summary")
    roll_tests_all = ex.rank_tests(roll[base_models])
    roll_tests_thesis = ex.rank_tests(roll[thesis])
    hyb_roll = ex.hybrid_rolling_rule(ctx, base_models, lookback=3)
    save(hyb_roll, "hybrid_rolling_rule")
    best_single_same = roll.loc[[pd.Timestamp(d).date() for d in hyb_roll["origin"]], base_models].mean().sort_values()
    out["accuracy"] = {
        "test": acc_test, "validation": acc_val, "m5_eval": acc_m5, "test_by_class": acc_cls,
        "rolling": roll, "rolling_summary": roll_summary, "rank_tests_all": roll_tests_all,
        "rank_tests_thesis_models": roll_tests_thesis, "hybrid_validation_choice": hyb_choice,
        "hybrid_rolling_mean_wape": float(hyb_roll["WAPE_hybrid"].mean()),
        "best_single_same_windows": {best_single_same.index[0]: float(best_single_same.iloc[0])},
    }
    log("accuracy done")

    # ------------------------------------------------------------ inventory: main -------
    inv_models = all_models + ["Hybrid_val"]
    inv_main = ex.inventory_table(ctx, k_test, policy, inv_models)
    inv_common = ex.inventory_table(ctx, k_test, policy, inv_models, init="common")
    save(inv_main, "inventory_test_main")
    save(inv_common, "inventory_test_common_initial")
    pareto_df = acc_test[["WAPE"]].join(inv_main[["total_cost"]]).loc[[m for m in inv_models]]
    pareto_df["pareto_efficient"] = ex.pareto_front(pareto_df, "WAPE", "total_cost")
    save(pareto_df, "pareto_test")
    # inventory by demand class for the cheapest model
    best_cost_model = inv_main.index[0]
    sim_best = ex.run_window(ctx, best_cost_model, k_test, policy)
    cls_rows = {}
    for c in DEMAND_CLASSES:
        m = res.classes[k_test] == c
        cls_rows[c] = portfolio_kpis({k: v[m] for k, v in sim_best.items()})
    inv_cls = pd.DataFrame(cls_rows).T
    save(inv_cls, "inventory_test_by_class_best_model")

    # ------------------------------------------------------------ sensitivity (test) ----
    scen = ex.scenario_grid(cfg, policy)
    thesis_scen = {k: v for k, v in scen.items() if not k.startswith("h=")}
    sens_test = ex.sensitivity(ctx, all_models, {"main": policy, **scen},
                               lambda m, p: ex.run_window(ctx, m, k_test, p))
    save(sens_test, "sensitivity_test_cost")
    reg_test = ex.regret(sens_test.drop(index="main"))
    save(reg_test, "regret_test")
    out["inventory_test"] = {
        "main": inv_main, "common_initial": inv_common, "pareto": pareto_df, "by_class_best": inv_cls,
        "best_cost_model": best_cost_model,
        "sensitivity_best": sens_test.idxmin(axis=1),
        "thesis_env_best": sens_test.loc[list(thesis_scen)].idxmin(axis=1).value_counts(),
        "hk_env_best": sens_test.loc[[k for k in scen if k.startswith("h=")]].idxmin(axis=1).value_counts(),
        "regret_mean": reg_test.mean().sort_values(), "regret_max": reg_test.max().sort_values(),
        "cost_optimal_service_level": policy.cost_optimal_service_level(),
    }
    log("inventory test done")

    # ------------------------------------------------------------ safety-stock methods --
    ss_rows = {}
    for m in all_models:
        for s in ex.SS_METHODS:
            ss_rows[(m, s)] = portfolio_kpis(ex.run_window(ctx, m, k_test, policy, s))
    ss_test = pd.DataFrame(ss_rows).T
    ss_test.index.names = ["model", "ss_method"]
    save(ss_test, "ss_methods_test")

    # ------------------------------------------------------------ continuous simulation -
    k_start, k_end = 6, len(res.origins) - 1
    cont = {}
    for m in all_models:
        for s in ex.SS_METHODS:
            cont[(m, s)] = portfolio_kpis(ex.continuous_simulation(ctx, m, k_start, k_end, policy, s))
    cont = pd.DataFrame(cont).T
    cont.index.names = ["model", "ss_method"]
    save(cont, "continuous_sim")
    best_ss = cont.groupby(level="ss_method")["total_cost"].min().idxmin()
    sens_cont = ex.sensitivity(ctx, all_models, {"main": policy, **scen},
                               lambda m, p: ex.continuous_simulation(ctx, m, k_start, k_end, p, "demand_std"))
    save(sens_cont, "sensitivity_continuous_cost")
    reg_cont = ex.regret(sens_cont.drop(index="main"))
    save(reg_cont, "regret_continuous")
    roll_cont = roll.iloc[k_start + 1: k_end + 1][all_models].mean()
    pareto_cont = pd.DataFrame({"WAPE": roll_cont,
                                "total_cost": cont.xs("demand_std", level="ss_method")["total_cost"]})
    pareto_cont["pareto_efficient"] = ex.pareto_front(pareto_cont, "WAPE", "total_cost")
    save(pareto_cont, "pareto_continuous")
    o0, o_end = res.origins[k_start], res.origins[k_end] + res.horizon
    out["continuous"] = {
        "start": portfolio.dates[o0 + res.horizon].date().isoformat(),
        "end": portfolio.dates[o_end - 1].date().isoformat(),
        "warmup_start": portfolio.dates[o0].date().isoformat(),
        "days_counted": int(o_end - o0 - res.horizon),
        "kpis": cont, "best_ss_method": best_ss,
        "best_by_ss_method": cont.groupby(level="ss_method")["total_cost"].idxmin(),
        "sensitivity_best": sens_cont.idxmin(axis=1),
        "regret_mean": reg_cont.mean().sort_values(), "regret_max": reg_cont.max().sort_values(),
        "pareto": pareto_cont,
    }
    log("continuous simulation done")

    # ------------------------------------------------------------ SKU-level statistics --
    sku = ex.sku_level(ctx, k_test, policy, all_models)
    best_acc_model = acc_test.loc[all_models, "WAPE"].idxmin()
    boots = [ex.cluster_bootstrap(sku["per_sku"], "RandomForest", "XGBoost", cfg["analysis"]["bootstrap_reps"], 42)]
    if {best_acc_model, best_cost_model} - {"RandomForest", "XGBoost"} and best_acc_model != best_cost_model:
        boots.append(ex.cluster_bootstrap(sku["per_sku"], best_cost_model, best_acc_model,
                                          cfg["analysis"]["bootstrap_reps"], 42))
    out["sku_level"] = {"spearman": sku["spearman"], "winner_coincidence": sku["winner_coincidence"],
                        "n_finite": sku["n_finite"], "cost_tests": sku["cost_tests"],
                        "wape_tests": sku["wape_tests"], "bootstrap": boots,
                        "best_acc_model": best_acc_model}
    log("sku-level stats done")

    # ------------------------------------------------------------ explainability --------
    shap_tables = {}
    for m, e in res.explain.items():
        sh = ex.shap_summary(e)
        if len(sh):
            save(sh, f"shap_{m}")
            shap_tables[m] = sh
    out["explain"] = {m: {"top": t.head(10)[["feature", "share", "spearman_value_vs_shap"]].to_dict("records"),
                          "top2_share": float(t["share"].head(2).sum())} for m, t in shap_tables.items()}
    out["importances"] = res.importances
    out["fit_seconds_total"] = {m: float(np.sum(v)) for m, v in res.fit_seconds.items()}

    # ------------------------------------------------------------ prospective ABC -------
    if not args.skip_prospective:
        pport, pinfo = select_portfolio(cfg, "prospective")
        overlap = len(set(pport.item_ids) & set(portfolio.item_ids))
        pres = run_backtest(cfg, models=base_models, portfolio_mode="prospective",
                            origin_dates=[cfg["backtest"]["test_origin"]], cache_name="backtest_prospective_test.pkl")
        ex.add_ensemble(pres)
        pctx = ex.Ctx(cfg, pport, pres, policy)
        p_acc = ex.accuracy_table(pctx, 0, all_models)
        p_inv = ex.inventory_table(pctx, 0, policy, all_models)
        save(p_acc, "prospective_accuracy_test")
        save(p_inv, "prospective_inventory_test")
        out["prospective"] = {"n_sku": pport.n_items, "overlap_with_expost": overlap,
                              "jaccard": overlap / len(set(pport.item_ids) | set(portfolio.item_ids)),
                              "accuracy": p_acc, "inventory": p_inv}
        log("prospective portfolio done")

    # ------------------------------------------------------------ figures ---------------
    plots.bar_metric(acc_test.loc[all_models + ["Hybrid_val"]], "WAPE", FIGS / "accuracy_test_wape.png",
                     "Held-out test (25 Apr - 22 May 2016): WAPE", xlabel="WAPE (%)")
    plots.rolling_lines(roll, FIGS / "rolling_wape.png",
                        ["SeasonalNaive", "ETS_ANA", "TSB", "RandomForest", "XGBoost", "LightGBM_Tweedie", "Ensemble"],
                        highlight=cfg["backtest"]["test_origin"])
    plots.pareto_scatter(pareto_df.drop(index=["Hybrid_val"]), "WAPE", "total_cost",
                         pareto_df.drop(index=["Hybrid_val"])["pareto_efficient"], FIGS / "pareto_test.png",
                         "Accuracy vs. inventory cost (held-out test)", "WAPE (%)", "Total relevant cost")
    plots.pareto_scatter(pareto_cont, "WAPE", "total_cost", pareto_cont["pareto_efficient"],
                         FIGS / "pareto_continuous.png",
                         f"Accuracy vs. cost over {out['continuous']['days_counted']} simulated days",
                         "Mean rolling WAPE (%)", "Total relevant cost")
    plots.cost_breakdown(inv_main.loc[all_models], FIGS / "cost_breakdown_test.png",
                         "Cost decomposition, held-out test (demand-σ safety stock)")
    plots.ss_methods_plot(cont.loc[[m for m in all_models if m in ("ETS_ANA", "RandomForest", "XGBoost", "LightGBM",
                                                                     "LightGBM_Tweedie", "Ensemble", "SeasonalNaive")]],
                          FIGS / "ss_methods_continuous.png", "Safety-stock method x forecast model (continuous simulation)")
    if "XGBoost" in shap_tables:
        plots.shap_bar(shap_tables["XGBoost"], FIGS / "shap_xgboost.png", "XGBoost: global TreeSHAP importance")
    plots.regret_heatmap(reg_cont[all_models], FIGS / "regret_continuous.png",
                         "Cost regret across 20 inventory environments (continuous simulation)")
    # one SKU trajectory (highest-revenue SKU, cheapest continuous model)
    best_cont = cont.xs(best_ss, level="ss_method")["total_cost"].idxmin()
    tr = ex.continuous_simulation(ctx, best_cont, k_start, k_end, policy, best_ss, keep_trajectory=True)
    # example SKU: highest-revenue smooth SKU with no zero-sales run longer than 3 days in the period
    seg = np.nan_to_num(Y[:, o0:o_end], nan=0.0)
    run = np.zeros(len(seg))
    for i, row in enumerate(seg):
        z = np.concatenate([[0], (row == 0).astype(int), [0]])
        d = np.flatnonzero(np.diff(z))
        run[i] = (d[1::2] - d[::2]).max() if len(d) else 0
    rev = np.nansum(Y[:, :1941] * portfolio.prices[:, :1941], axis=1)
    ok = (run <= 3) & (res.classes[k_test] == "smooth")
    top = int(np.flatnonzero(ok)[np.argmax(rev[ok])])
    dates = portfolio.dates[o0:o_end]
    plots.trajectory(dates, np.nan_to_num(Y[top, o0:o_end]), tr["on_hand"][top], FIGS / "trajectory_top_sku.png",
                     f"{portfolio.item_ids[top]}: demand and stock, {plots.LABEL.get(best_cont, best_cont)} + "
                     f"{plots.SS_LABEL[best_ss]}", res.horizon)
    out["figures"] = sorted(p.name for p in FIGS.glob("*.png"))

    (ROOT / "reports" / "results.json").write_text(json.dumps(jsonable(out), indent=1, ensure_ascii=False))
    log("all done -> reports/")


if __name__ == "__main__":
    main()
