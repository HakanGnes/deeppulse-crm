# DeepPulse CRM

**Sekans bazlı müşteri davranışı ve dinamik fiyatlama motoru.** [AnalyzeWise-CRM](https://github.com/HakanGnes/analyzewise-crm)'in
RFM, BG/NBD ve XGBoost tabanlı modellerinin ötesine geçen derin öğrenme versiyonu.
Veri katmanı SAP SD/FI tablo yapısını izler.

Klasik RFM/CLTV modelleri müşterinin "toplamda kaç kez ve ne zaman" alışveriş yaptığına bakar.
DeepPulse olayların **sırasını ve bağlamını** modeller: sonraki sepette ne alınacak, müşteri ritmi
ne zaman kopacak, indirim bu ikisini nasıl değiştirir?

## Yol haritası

| Faz | İçerik | Durum |
|---|---|---|
| 0 | Veri katmanı (Dunnhumby → SAP tabloları), churn tanımı, sızıntısız zaman bölmesi, baseline'lar | ✅ |
| 1 | Sekans modelleri: GRU4Rec ve SASRec (sonraki sepet + kopma başlığı), PyTorch | 🔄 sonraki sepette kişisel tekrarı anlamlı geçiyor (kazanç tekrar teriminden); kopmada XGBoost'a eşit, dizi ek bilgi katmıyor. Zaman ağırlıklı baseline ve iki test günüyle doğrulama sürüyor |
| 2 | Deep survival: DeepHit, CoxTime (CoxPH ve BG/NBD'ye karşı) | ⏳ |
| 3 | İndirim etkisi: Double ML / uplift, ara modeller FT-Transformer ve TabNet | ⏳ |
| 4 | Karar destek paneli (FastAPI + React): attention görselleştirme, indirim → CLTV simülatörü | ⏳ |

Kararlar ve gerekçeleri: [`docs/decisions.md`](docs/decisions.md)

### Faz 0 bulgusu: bu veride "terk" yok, "kopma" var

Dunnhumby'de 90 günlük sessizlikten sonra bile müşterilerin %83'ü 180 gün içinde geri dönüyor.
Bu yüzden modellenen olay kalıcı terk değil **kopma (lapse)**: 60 gün hiç alışveriş yapmamak
(tüm alışveriş aralıklarının yalnızca %1'i). Yani geri kazanma kampanyasının tetikleyicisi.

| Görev | Baseline (gerçek veri) |
|---|---|
| 90 günde kopma | XGBoost ROC AUC 0.893 (sadece recency 0.834) |
| Kopmaya kadar süre | CoxPH C-index 0.878 |
| Sonraki sepet | kişisel tekrar Recall@10 0.157 (popülerlik 0.088) |

## Veri

[Dunnhumby — The Complete Journey](https://www.dunnhumby.com/source-files/): 2.500 hanenin
2 yıllık (711 gün) market alışverişleri, ~2,6 milyon satır. İndirim, kupon ve kampanya tabloları da var.

1. Veriyi şu kaynaklardan birinden indir:
   - Resmi sayfa: dunnhumby.com/source-files (form doldurmanı isteyebilir)
   - Kaggle kopyası: `frtgnn/dunnhumby-the-complete-journey`
2. CSV'leri `data/raw/` klasörüne koy. En az `transaction_data.csv` ve `product.csv` gerekli;
   `hh_demographic.csv` varsa demografi tablosu da üretilir.

> R'daki `completejourney` paketi farklı bir sürüm (1 yıl, 2.469 hane, farklı sütunlar).
> Bu proje orijinal CSV şemasını bekler.

`data/` klasörü `.gitignore`'da; veri depoya girmez.

## Kurulum (Windows / PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

Faz 1 PyTorch (CUDA) ister. Kurulum komutu pytorch.org'daki seçiciden alınır
(Stable → Windows → Pip → Python → CUDA); `pip install torch` tek başına CPU sürümünü kurabilir. Kontrol:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Çalıştırma

```powershell
# 1) Churn eşiğini seç (boşluk dağılımı + aday N'ler için churn oranı)
python scripts/suggest_threshold.py --raw data/raw

# 2) Baseline'lar → reports/baseline_metrics.json
python scripts/run_baselines.py --raw data/raw
python scripts/run_baselines.py --raw data/raw --churn-days 45 --rebuild

# Veri olmadan denemek için (sonuçların gerçek dünyayla ilgisi yoktur)
python scripts/run_baselines.py --synthetic
```

### Faz 1 — sepet dizisi modelleri

```powershell
python scripts/train_seq.py --raw data/raw --encoder gru
python scripts/train_seq.py --raw data/raw --encoder sasrec
python scripts/train_seq.py --raw data/raw --encoder sasrec --test-days 531 621   # iki test günü
python scripts/train_seq.py --raw data/raw --encoder sasrec --no-summary          # ablasyon: yalnızca dizi
python scripts/train_seq.py --raw data/raw --encoder sasrec --no-repeat           # ablasyon: tekrar terimi yok
```

Her çalıştırma modeli eğitir, aynı test kümesinde XGBoost (kopma), kişisel tekrar ve zaman ağırlıklı tekrar (sonraki sepet)
ile eşli bootstrap %95 güven aralığıyla karşılaştırır ve `reports/seq_<model>.json` yazar.
GPU belleği yetmezse `--batch-size 64` veya `--max-candidates 4096`.

İlk çalıştırma ham CSV'leri SAP tablolarına çevirip `data/sap/*.parquet` olarak kaydeder.
Sonraki çalıştırmalar doğrudan bu tablolardan okur (`--rebuild` ile yeniden üretilir).

## Proje yapısı

```
src/deeppulse/
  config.py          # churn eşiği, pencere, SAP ayarları
  data/dunnhumby.py  # ham CSV yükleyici + şema doğrulama
  data/canonical.py  # kaynaktan bağımsız işlem/ürün tabloları
  data/synthetic.py  # Dunnhumby şemasında sentetik veri (testler için)
  sap/schema.py      # SAP tablo/alan sözlüğü
  sap/adapter.py     # kanonik ⇄ KNA1, MARA, VBRK, VBRP, PRCD_ELEMENTS, BKPF, BSEG
  labels.py          # churn/survival etiketleri, sansürleme
  features.py        # snapshot özellikleri (baseline'lar için)
  splits.py          # sızıntısız zaman bazlı fold
  baselines.py       # XGBoost, CoxPH, popülerlik, kişisel tekrar
  metrics.py         # Recall/NDCG/Hit@k, eşli bootstrap
  pipeline.py        # uçtan uca hat
  seq/data.py        # vocab, sepet dizisi deposu, partiler (sızıntı güvenceli)
  seq/model.py       # sepet kodlayıcı, GRU / SASRec, tekrar-farkında sonraki sepet, kopma başlığı
  seq/train.py       # eğitim (AMP), doğrulama, başlık başına en iyi checkpoint
  seq/compare.py     # baseline'larla güven aralıklı karşılaştırma
scripts/             # suggest_threshold.py, run_baselines.py, train_seq.py
tests/               # 36 test (SAP round-trip, etiket örnekleri, sızıntı, nedensellik, model)
```

## Lisans

MIT
