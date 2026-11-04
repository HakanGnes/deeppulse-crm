"""Sızıntısız zaman bazlı katmanlar (fold).

AnalyzeWise-CRM'de öğrenilen ders: rolling snapshot'lar rastgele bölünürse aynı
müşterinin neredeyse aynı iki anlık görüntüsü eğitim ve teste dağılır ve
metrikler şişer (sinyalsiz veride bile AUC ~0.65 çıkmıştı).

Burada bölme zamana göre yapılır ve iki kural uygulanır:
1. Eğitim snapshot'ları T <= test_day - W  (etiket pencereleri test gününden
   önce kapanır).
2. Eğitim etiketleri `horizon = test_day` ile hesaplanır, yani test gününden
   sonraki hiçbir alışveriş (survival süreleri dahil) eğitime girmez.
Test snapshot'ı ise tüm veriyle (gerçek ufukla) etiketlenir.

Aynı müşteri hem eğitimde (geçmiş snapshot'lar) hem testte (son snapshot)
bulunabilir — bu gerçek kullanımın aynısıdır (mevcut müşterilerin geleceğini
tahmin ediyoruz) ve zaman ayrımı sayesinde sızıntı değildir.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from deeppulse.config import LabelConfig
from deeppulse.features import snapshot_features
from deeppulse.labels import label_snapshot, purchase_days, snapshot_days


@dataclass
class Fold:
    train: pd.DataFrame
    test: pd.DataFrame
    test_day: int
    train_days: list[int]


def default_test_day(tx: pd.DataFrame, cfg: LabelConfig) -> int:
    return int(tx["day"].max()) - cfg.prediction_window_days


def build_snapshot_table(
    tx: pd.DataFrame, days: list[int], horizon: int, cfg: LabelConfig
) -> pd.DataFrame:
    tx_h = tx[tx["day"] <= horizon]
    pdays = purchase_days(tx_h)
    parts = []
    for t in days:
        lab = label_snapshot(tx_h, t, horizon, cfg, pdays=pdays)
        if lab.empty:
            continue
        feats = snapshot_features(tx_h, t, customers=pd.Index(lab["customer_id"]))
        parts.append(lab.merge(feats, left_on="customer_id", right_index=True))
    if not parts:
        raise ValueError("Hiç etiketlenebilir snapshot yok — veri ufku çok kısa olabilir.")
    return pd.concat(parts, ignore_index=True)


def temporal_fold(tx: pd.DataFrame, cfg: LabelConfig, test_day: int | None = None) -> Fold:
    horizon = int(tx["day"].max())
    if test_day is None:
        test_day = default_test_day(tx, cfg)
    if test_day + cfg.prediction_window_days > horizon:
        raise ValueError("test_day + W veri ufkunu aşıyor; test etiketi tanımsız olur.")
    min_day = int(tx["day"].min())
    # snapshot_days(…, horizon=test_day) zaten d <= test_day - W olanları döndürür
    train_days = snapshot_days(min_day, test_day, cfg)
    train = build_snapshot_table(tx, train_days, horizon=test_day, cfg=cfg)
    test = build_snapshot_table(tx, [test_day], horizon=horizon, cfg=cfg)
    return Fold(train=train, test=test, test_day=test_day, train_days=train_days)
