"""Faz 0 baseline'larını çalıştırır ve reports/baseline_metrics.json yazar.

Gerçek veri:   python scripts/run_baselines.py --raw data/raw
Sentetik veri: python scripts/run_baselines.py --synthetic
Eşik değiştir: python scripts/run_baselines.py --raw data/raw --churn-days 45
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

from deeppulse.baselines import run_all
from deeppulse.config import DEFAULT_CONFIG
from deeppulse.pipeline import load_canonical, write_report
from deeppulse.splits import temporal_fold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None, help="Dunnhumby CSV klasörü")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--sap-dir", default=None, help="SAP parquet klasörü (varsayılan data/sap veya data/sap_synthetic)")
    ap.add_argument("--rebuild", action="store_true", help="SAP tablolarını yeniden üret")
    ap.add_argument("--churn-days", type=int, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--exclude-dept", nargs="*", default=[], help="Hariç tutulacak DEPARTMENT değerleri")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    labels = DEFAULT_CONFIG.labels
    if args.churn_days:
        labels = replace(labels, churn_inactivity_days=args.churn_days)
    if args.window:
        labels = replace(labels, prediction_window_days=args.window)
    cfg = replace(DEFAULT_CONFIG, labels=labels)

    sap_dir = args.sap_dir or ("data/sap_synthetic" if args.synthetic else "data/sap")
    t0 = time.time()
    tx, _ = load_canonical(args.raw, sap_dir, cfg, synthetic=args.synthetic,
                           rebuild=args.rebuild, exclude_departments=tuple(args.exclude_dept))
    print(f"[1/3] Veri yüklendi: {len(tx):,} satır, {tx['customer_id'].nunique():,} müşteri ({time.time() - t0:.1f}s)")

    fold = temporal_fold(tx, cfg.labels, cfg.split.test_snapshot_day)
    print(f"[2/3] Fold: {len(fold.train_days)} eğitim snapshot'ı, {len(fold.train):,} eğitim satırı, "
          f"test günü {fold.test_day} ({len(fold.test):,} aktif müşteri) ({time.time() - t0:.1f}s)")

    report = run_all(tx, fold, cfg.labels, cfg.random_state)
    out = args.out or ("reports/baseline_metrics_synthetic.json" if args.synthetic else "reports/baseline_metrics.json")
    write_report(report, out)
    print(f"[3/3] Baseline'lar tamam ({time.time() - t0:.1f}s) → {out}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
