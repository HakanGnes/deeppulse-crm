"""Churn eşiği N için karar tablosu.

    python scripts/suggest_threshold.py --raw data/raw
    python scripts/suggest_threshold.py --synthetic

Bakılacaklar:
- share_gaps_longer_than_N: alışverişler arası boşlukların ne kadarı N'den uzun.
  Çok yüksekse (ör. >%10) N, müşterinin normal ritmini "terk" sayıyor demektir.
- churn_rate_in_window: son snapshot'taki aktif müşterilerin W günde terk oranı.
  %5'in altı modellemeyi zorlaştırır (çok dengesiz), %50 üstü N'nin fazla kısa olduğunu gösterir.
"""

from __future__ import annotations

import argparse

import numpy as np

from deeppulse.config import DEFAULT_CONFIG
from deeppulse.labels import inter_purchase_gaps, suggest_threshold_table
from deeppulse.pipeline import load_canonical


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--sap-dir", default=None)
    args = ap.parse_args()
    sap_dir = args.sap_dir or ("data/sap_synthetic" if args.synthetic else "data/sap")
    tx, _ = load_canonical(args.raw, sap_dir, DEFAULT_CONFIG, synthetic=args.synthetic)

    gaps = inter_purchase_gaps(tx)
    qs = np.percentile(gaps, [50, 75, 90, 95, 99])
    print("Alışverişler arası boşluk (gün) — medyan/75/90/95/99. yüzdelik:", np.round(qs, 1))
    print()
    print(suggest_threshold_table(tx, prediction_window_days=DEFAULT_CONFIG.labels.prediction_window_days)
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
