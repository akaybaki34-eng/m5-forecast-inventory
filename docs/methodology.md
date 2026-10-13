# Methodology

## 1. Data and scope

* Files: `calendar.csv`, `sales_train_evaluation.csv` (d_1–d_1941), `sales_test_evaluation.csv`
  (d_1942–d_1969, the official M5 evaluation window), `sell_prices.csv`.
* Store CA_3, department FOODS_3 (823 SKUs; the largest FOODS department by unit sales both on the
  full history and on data before the first backtest origin).
* **Activity:** a SKU is active from its first valid selling price. Earlier zeros are structural
  and set to missing (7.5 % of A-class SKU-days).
* **ABC:** `RV_i = Σ_t D_it · P_it`; A if cumulative share ≤ 0.80, B ≤ 0.95, else C.
  * `expost` (default, thesis): RV over all 1,941 days → 293 A-SKUs.
  * `prospective`: RV only before 2015-05-25 → 278 A-SKUs, 276 shared with the ex-post set.
* **Demand classes** (Syntetos-Boylan-Croston): `ADI = active days / days with demand`,
  `CV² = (σ⁺/μ⁺)²` of positive demand; cut-offs 1.32 and 0.49. Recomputed at every origin with
  data before that origin.

## 2. Backtest protocol

* 14 origins, 28 days apart: 13 consecutive windows from 2015-05-25 to 2016-04-25 (one full year,
  includes the thesis's validation 2016-03-28 and test 2016-04-25 origins) plus 2016-05-23, the
  official M5 evaluation window that the thesis never used.
* At each origin every model is (re)fitted on data strictly before the origin and produces a
  fixed-origin forecast for 42 days (28-day evaluation window + 14 days so the stock policy always
  sees a full protection period).
* Accuracy: MAE, RMSE, `WAPE = Σ|D−F| / ΣD`, `Bias = Σ(F−D) / ΣD` over active SKU-days.

## 3. Models

| Model | Notes |
|---|---|
| Seasonal Naive | last observed week repeated |
| Moving Average 28 | flat mean of the last 28 days |
| ETS(A,N,A) | additive level + weekly seasonality, no trend; α, γ by per-SKU grid search on one-step SSE over the last 730 days (vectorised over SKUs × grid) |
| Croston-SBA | Croston with `(1 − α/2)` bias correction; α by grid search |
| TSB | Teunter-Syntetos-Babai, probability updated every period; α, β by grid search |
| Random Forest | 50 trees, depth 14, min leaf 8, max_features 0.6, max_samples 0.2 |
| XGBoost | 300 trees, lr 0.05, depth 6, min_child_weight 20, subsample/colsample 0.8 |
| LightGBM | 300 trees, lr 0.05, 31 leaves, min_child_samples 50 |
| LightGBM-Tweedie | Tweedie loss (p = 1.2), native categorical item ID, 2 extra price features |
| Ensemble | equal-weight mean of ETS, RF, XGBoost, LightGBM, LightGBM-Tweedie (fixed a priori) |

ML models are global (one model for all SKUs) with 20 features: item code, lags 1/7/14/28,
rolling mean and std over 7 and 28 days (`shift(1).rolling(w)`), selling price, SNAP, month,
day of week, ISO week, weekend flag, four event-type flags, time index. Multi-step forecasts are
recursive: each day's forecast is written back into the history and feeds later lags.

## 4. Inventory policy and simulator

Periodic review, order-up-to, lost sales, daily time step. On day `t`:

1. orders due arrive;
2. on review days (`t mod R = 0`): `IP = on-hand + on-order`,
   `Q = ceil(max(0, S_t − IP))`, delivered on `t + L`;
3. demand is served from stock, the rest is lost;
4. holding cost on end-of-day stock.

`S_t = Σ_{k=0}^{H−1} F_{t+k} + SS`, `H = R + L`.

Costs: `HC = EI · P · h / 365`, `SC = lost · P · c_s`, `OC = K · orders`. Base case R = 7, L = 5,
nominal 95 %, h = 25 %/year, c_s = 1, K = 5. Costs are comparative scenario costs.

### Safety-stock rules

| Rule | Formula | Data used |
|---|---|---|
| Demand σ (thesis) | `z · σ_D · √H` | σ of daily demand, 90 days before the origin; common to all models |
| Model error σ | `z · σ_e · √H` | σ of daily forecast errors of the same model in the 4 previous windows |
| Cost-optimal SL | demand-σ rule with SL = `c_s / (c_s + h·R/365)` = 99.5 % | newsvendor critical ratio of one review cycle |
| Empirical quantile | `q_SL` of past H-day cumulative errors `Σ(D−F)` | errors from the 6 previous windows at every review offset whose H-day block ended before the current origin; scaled by each SKU's mean daily demand and pooled within its demand class |

The empirical rule drops the normality and the `√H` (independent daily errors) assumptions.
Forecast errors of recursive models are autocorrelated over the horizon, so `σ·√H`
understates H-day uncertainty. Measured on the six windows before the test origin, the standard
deviation of 12-day cumulative errors is 1.43–1.54 × `σ_daily · √12` (median over SKUs; Seasonal
Naive 1.43, Random Forest 1.52, XGBoost 1.54). This is why the textbook rule under-delivers its
target in the long simulation. Under the base cost structure (lost sale = full price, holding
25 %/year) extra stock is cheap relative to a lost sale, so the better-calibrated, larger safety
stock also lowers total cost, at the same nominal 95 % target.

### Experiments

* **Main window** (thesis protocol): 2016-04-25, initial stock = first order-up-to level of each
  model; robustness with a common initial stock for all models.
* **Continuous simulation:** windows 2015-11-09 → 2016-06-19; the first 28 days are warm-up and
  not counted (196 counted days). Inventory carries over; each review uses the forecast and the
  safety stock computed at the latest origin.
* **Sensitivity:** L ∈ {3,5,7} × SL ∈ {90,95,99 %}; c_s ∈ {0.25, 0.5, 2}; h ∈ {10,25,40 %} ×
  K ∈ {1,5,10} → 20 alternative environments + base case. Regret
  `(C_ms − min_k C_ks) / min_k C_ks`.

## 5. Statistics

* Friedman test across models over the 14 windows; Wilcoxon signed-rank of the best-ranked model
  against each other model with Holm correction.
* SKU level (main window): Spearman WAPE vs cost/revenue and bias vs fill rate; share of SKUs where
  the most accurate and the cheapest model coincide; Friedman/Wilcoxon-Holm on SKU costs.
* SKU-cluster bootstrap (10,000 resamples of SKUs, seed 42) for portfolio WAPE and cost
  differences. It measures uncertainty from portfolio composition, not from future time periods.
* TreeSHAP (XGBoost and LightGBM built-in) on 2,500 random rows of the recursive test-horizon
  feature matrix.

## 6. Hybrid model selection

* Validation rule (thesis): per demand class, the model with the lowest validation-window WAPE is
  applied to the test window.
* Rolling rule (new): per class, the model with the lowest mean WAPE over the previous three
  windows is applied to the next window, for every window from the fourth on.
