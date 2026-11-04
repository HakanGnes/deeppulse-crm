"""Faz 1: sepet dizisi modelini eğitir, Faz 0 baseline'larıyla güven aralıklı karşılaştırır.

    python scripts/train_seq.py --raw data/raw --encoder gru
    python scripts/train_seq.py --raw data/raw --encoder sasrec
    python scripts/train_seq.py --raw data/raw --encoder sasrec --test-days 531 621   # iki test günü
    python scripts/train_seq.py --raw data/raw --encoder sasrec --no-summary          # ablasyon (--no-repeat, --no-logq)
    python scripts/train_seq.py --synthetic --epochs 3                                 # hızlı deneme

Çıktı: reports/seq_<encoder>.json (+ --save ile models/seq_<encoder>_<gün>.pt)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from deeppulse.config import DEFAULT_CONFIG
from deeppulse.pipeline import load_canonical, write_report
from deeppulse.seq.compare import compare_with_baselines, verdict
from deeppulse.seq.train import SeqConfig, train_and_evaluate
from deeppulse.splits import default_test_day, temporal_fold


def _fmt(ci: dict) -> str:
    return f"{ci['diff']:+.3f}  [%95: {ci['ci95_low']:+.3f}, {ci['ci95_high']:+.3f}]  → {verdict(ci)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--sap-dir", default=None)
    ap.add_argument("--encoder", choices=["gru", "sasrec"], default="sasrec")
    ap.add_argument("--test-days", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--no-summary", action="store_true",
                    help="ablasyon: kopma başlığı Faz 0 özet özelliklerini görmesin (yalnızca dizi + recency)")
    ap.add_argument("--no-repeat", action="store_true", help="ablasyon: sonraki sepette tekrar bonusu kapalı")
    ap.add_argument("--max-candidates", type=int, default=None, help="GPU belleği yetmezse düşür (ör. 4096)")
    ap.add_argument("--no-logq", action="store_true", help="ablasyon: logQ popülerlik düzeltmesi kapalı")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--save", action="store_true", help="en iyi modeli models/ altına kaydet")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = SeqConfig(encoder=args.encoder)
    overrides = {"epochs": args.epochs, "batch_size": args.batch_size, "d_model": args.d_model,
                 "max_candidates": args.max_candidates, "lr": args.lr}
    if args.no_logq:
        overrides["logq_correction"] = False
    if args.no_summary:
        overrides["lapse_summary_features"] = False
    if args.no_repeat:
        overrides["repeat_aware"] = False
    cfg = replace(cfg, **{k: v for k, v in overrides.items() if v is not None})

    tag = (cfg.encoder + ("_nosummary" if not cfg.lapse_summary_features else "")
           + ("_norepeat" if not cfg.repeat_aware else "") + ("_nologq" if not cfg.logq_correction else ""))
    sap_dir = args.sap_dir or ("data/sap_synthetic" if args.synthetic else "data/sap")
    t0 = time.time()
    tx, _ = load_canonical(args.raw, sap_dir, DEFAULT_CONFIG, synthetic=args.synthetic)
    label_cfg = DEFAULT_CONFIG.labels
    test_days = args.test_days or [default_test_day(tx, label_cfg)]
    print(f"Veri: {len(tx):,} satır ({time.time() - t0:.1f}s) | model={tag} | test günleri {test_days}")

    report = {"encoder": cfg.encoder, "lapse_inactivity_days": label_cfg.churn_inactivity_days,
              "prediction_window_days": label_cfg.prediction_window_days, "folds": []}
    for day in test_days:
        print(f"\n=== Test günü {day} ===")
        fold = temporal_fold(tx, label_cfg, test_day=day)
        result = train_and_evaluate(tx, fold, label_cfg, cfg, device=args.device)
        comp = compare_with_baselines(tx, fold, result, label_cfg, n_boot=args.n_boot)

        k = cfg.k
        la, ew = comp["lapse"]["all"], comp["lapse"]["early_warning_recency_le_14"]
        nb = comp["next_basket"]
        print(f"\n  Kopma (tümü, n={la['n']}):  model AUC {la['model_roc_auc']:.3f} vs XGBoost {la['xgboost_roc_auc']:.3f}")
        print(f"    ROC AUC farkı {_fmt(la['diff_roc_auc'])}")
        print(f"    PR AUC  farkı {_fmt(la['diff_pr_auc'])}")
        print(f"  Kopma erken uyarı (n={ew['n']}): model AUC {ew['model_roc_auc']:.3f} vs XGBoost {ew['xgboost_roc_auc']:.3f}")
        print(f"    ROC AUC farkı {_fmt(ew['diff_roc_auc'])}")
        print(f"    PR AUC  farkı {_fmt(ew['diff_pr_auc'])}")
        print(f"  Sonraki sepet (n={nb['n_customers']}): model R@{k} {nb[f'model_recall@{k}']:.3f} | "
              f"kişisel tekrar {nb[f'personal_repeat_recall@{k}']:.3f} | "
              f"zaman ağırlıklı tekrar (decay={nb['decay']}) {nb[f'decayed_repeat_recall@{k}']:.3f}")
        for base, label in (("personal_repeat", "kişisel tekrar"), ("decayed_repeat", "zaman ağırlıklı tekrar")):
            print(f"    vs {label}:")
            print(f"      Recall@{k} farkı {_fmt(nb[f'diff_vs_{base}_recall@{k}'])}")
            print(f"      NDCG@{k}   farkı {_fmt(nb[f'diff_vs_{base}_ndcg@{k}'])}")

        report["folds"].append({
            "test_day": day,
            "best_epoch": result["best_epoch"],
            "vocab_size": result["vocab_size"],
            "n_params": result["n_params"],
            "best_val": result["best_val"],
            "history": result["history"],
            "comparison": comp,
        })
        if args.save:
            path = Path("models") / f"seq_{tag}_{day}.pt"
            path.parent.mkdir(exist_ok=True)
            torch.save({"states": result["states"], "config": result["config"],
                        "vocab": result["vocab"].product_ids.tolist()}, path)
            print(f"  model kaydedildi → {path}")

    report["config"] = result["config"]
    out = args.out or (f"reports/seq_{tag}_synthetic.json" if args.synthetic else f"reports/seq_{tag}.json")
    write_report(report, out)
    print(f"\nRapor → {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
