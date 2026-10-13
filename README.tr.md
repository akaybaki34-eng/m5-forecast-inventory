# Tahmin → Stok: daha doğru tahmin, daha ucuz stok politikası mı demek?

*[English README](README.md)*

Bu depo, İstanbul Gedik Üniversitesi Endüstri Mühendisliği lisans bitirme tezimin
("Makine Öğrenmesi Tabanlı Talep Tahmini ve Stok Kontrol Politikalarının Entegrasyonu: M5
Perakende Veri Seti Üzerine Bir Uygulama", 2026) sıfırdan, yeniden üretilebilir ve genişletilmiş
yazılım sürümüdür.

Her talep tahmin modeli iki kez puanlanır: **istatistiksel doğruluk** (WAPE, MAE, RMSE, yanlılık)
ve aynı periyodik gözden geçirmeli (R, S_t) stok politikasına bağlandığında ürettiği
**operasyonel sonuç** (toplam ilgili maliyet, talep karşılama oranı, kayıp satış).

Kapsam: Walmart **CA_3** mağazası, **FOODS_3** departmanı, gelirin %80'ini oluşturan
**293 A-sınıfı SKU** (525.957 SKU-gün).

## Tezden bu sürüme ne eklendi?

| Tez (v1) | Bu depo (v2) |
|---|---|
| 5 kayan başlangıçlı dönem | **14 dönem** (1 tam yıl + hiç kullanılmamış M5 resmi değerlendirme penceresi) |
| 6 model | + **Croston-SBA, TSB** (kesikli talep), + **LightGBM-Tweedie** (kategorik ürün kimliği), + **Ensemble** |
| 28 günlük tek pencere stok simülasyonu | + **196 günlük sürekli simülasyon** (ısınma dönemi, stok pencereler arası taşınır) |
| z·σ·√H emniyet stoğu | + model hata σ, + **ampirik kantil (dağılımdan bağımsız)**, + **maliyet-optimal servis düzeyi** |
| 12 stok ortamı | **20 ortam** (+ h × K tam faktöriyel) |
| Ex-post ABC | + **prospektif ABC** (ileriye bakış yok) |
| Betik / notebook | Paket yapısı, TOML konfigürasyon, 25 birim testi, CI, önbellekli pipeline |

## Ana bulgular

1. **En doğru model en ucuz model değil.** Testte XGBoost, Random Forest'tan daha düşük WAPE
   üretiyor (%42,9 vs %44,1) ama stok maliyeti **%12 daha yüksek** (11.282 vs 10.058).
2. **Random Forest maliyet açısından en sağlam seçim:** uzun simülasyonda 21 ortamın 18'inde en
   düşük maliyet, ortalama pişmanlık (regret) %0,09.
3. **Bir yıllık backtest'te en doğru model LightGBM-Tweedie:** 14 pencerenin 8'ini kazanıyor
   (ortalama WAPE %46,7; Friedman p < 0,001).
4. **Emniyet stoğunun nasıl hesaplandığı, hangi modelin seçildiğinden daha önemli.** 196 günlük
   simülasyonda klasik z·σ·√H kuralı %95 hedefin altında kalıyor (%92,2–94,7 karşılama), çünkü
   12 günlük kümülatif tahmin hatası √H varsayımının öngördüğünün ~1,5 katı.
   Geçmiş H-günlük kümülatif hataların %95 kantiline dayanan kural karşılamayı %95,7–96,7'ye
   çıkarıyor ve **toplam maliyeti %18 düşürüyor** (94,7 bin → 77,3 bin).
5. Tek doğrulama penceresine göre talep sınıfı bazında model seçmek genellenmiyor (hibrit %45,1 WAPE).
6. ABC'yi yalnızca geçmiş veriyle seçmek portföyü neredeyse değiştirmiyor (278 SKU'dan 276'sı aynı).

## Çalıştırma

```bash
pip install -r requirements.txt && pip install -e .
python scripts/download_data.py
python -m pytest -q
python scripts/run_all.py      # 2 çekirdekte ~12 dk; tüm tablo ve şekiller reports/ altına
```

Ayrıntılı sonuç tabloları, şekiller ve yöntem açıklaması için [README.md](README.md) ve
[docs/methodology.md](docs/methodology.md).
