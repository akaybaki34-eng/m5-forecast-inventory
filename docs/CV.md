# CV, LinkedIn and interview material

Repository: https://github.com/akaybaki34-eng/m5-forecast-inventory

## CV – English (Projects section)

**Forecast-to-Inventory Decision Framework (M5 Walmart data)** · Python, scikit-learn, XGBoost, LightGBM · B.Sc. thesis, extended · github.com/akaybaki34-eng/m5-forecast-inventory

- Built a leakage-safe pipeline that scores 9 demand-forecasting models (ETS, Croston-SBA, TSB, Random Forest, XGBoost, LightGBM) on both accuracy and the inventory cost they produce in a lost-sales (R, S) policy; 293 SKUs, 526K SKU-days, 14-window rolling backtest.
- Showed that the most accurate model is not the cheapest: XGBoost had lower WAPE than Random Forest (42.9 % vs 44.1 %) but 12 % higher inventory cost; Random Forest was cheapest in 18 of 21 cost / lead-time scenarios.
- Replaced the textbook z·σ·√H safety stock with a distribution-free empirical-quantile rule: for the same Random Forest forecasts, fill rate rose from 94.3 % to 96.3 % and total cost fell 17 % in a 196-day simulation.
- Added a Tweedie-loss LightGBM that won 8 of 14 monthly backtest windows (Friedman p < 0.001); packaged the work with 25 unit tests (leakage, simulator), config-driven runs and CI.

*Short version (one line):* Forecast-to-inventory framework on M5 Walmart data: showed the most accurate model was 12 % costlier to run and cut inventory cost 17 % with empirical-quantile safety stock (Python, XGBoost, LightGBM).

## CV – Türkçe (Projeler bölümü)

**Talep Tahmininden Stok Kararına Karar Destek Çerçevesi (M5 Walmart verisi)** · Python, scikit-learn, XGBoost, LightGBM · Lisans bitirme tezi, genişletilmiş

- 9 talep tahmin modelini (ETS, Croston-SBA, TSB, Random Forest, XGBoost, LightGBM) hem doğruluk hem de kayıp satışlı (R, S) stok politikasında ürettikleri maliyet açısından değerlendiren, bilgi sızıntısına karşı test edilmiş bir pipeline kurdum; 293 SKU, 526 bin SKU-gün, 14 dönemlik kayan başlangıçlı backtest.
- En doğru modelin en ucuz model olmadığını gösterdim: XGBoost, Random Forest'tan daha düşük WAPE'ye sahipken (%42,9 vs %44,1) stok maliyeti %12 daha yüksekti; Random Forest 21 maliyet/tedarik süresi senaryosunun 18'inde en düşük maliyeti verdi.
- Klasik z·σ·√H emniyet stoğu yerine dağılımdan bağımsız ampirik kantil kuralı geliştirdim: aynı Random Forest tahminleriyle 196 günlük simülasyonda karşılama oranı %94,3 → %96,3, toplam maliyet %17 daha düşük.
- Tweedie kayıplı LightGBM modeli 14 aylık dönemin 8'inde en iyi sonucu verdi (Friedman p < 0,001); proje 25 birim testi, konfigürasyonla çalışan pipeline ve CI ile paketlendi.

## LinkedIn – project description

> Does a more accurate demand forecast lead to a cheaper inventory? In my Industrial Engineering
> thesis I tested this on Walmart's M5 data (293 high-value SKUs). Answer: not necessarily. XGBoost
> had the lowest forecast error but Random Forest produced 12 % lower inventory cost, because its
> slight over-forecasting prevented expensive lost sales. In the extended version I found that the
> way safety stock is calculated matters even more: replacing the textbook σ·√H formula with an
> empirical-quantile rule cut total cost by 17 % in a 196-day simulation. Code, tests and all
> results are open source.

## Interview talking points

1. **Why WAPE and not MAPE?** Many SKU-days have zero sales; MAPE divides by zero. WAPE weights
   errors by volume, which is what matters for inventory.
2. **How did you prevent leakage?** One feature function for training and inference; rolling
   windows exclude the target day; recursive forecasting after the origin; a unit test that
   overwrites all future actuals and asserts the forecast is unchanged; demand classes and
   safety stocks use only past data.
3. **Why did Random Forest win on cost?** +9.6 % bias → more stock (holding +71 cost units) but
   far fewer lost sales (−1,345 cost units). With lost sales priced at the selling price and
   holding at 25 %/year, mild over-forecasting pays.
4. **Why does σ·√H fail?** It assumes independent daily errors. Recursive forecast errors are
   autocorrelated, so the 12-day error spread is ~1.5× larger than the formula assumes and the
   policy under-stocks.
5. **What would you do with real company data?** Calibrate costs (margin, purchase cost, real lead
   times), treat stock-out days as censored demand, and select models on simulated cost rather
   than WAPE alone.
