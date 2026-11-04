"""Sekans modeli ↔ Faz 0 baseline'ları: aynı test kümesinde, eşli bootstrap ile."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from deeppulse.baselines import (
    EARLY_WARNING_MAX_RECENCY, decayed_repeat_predictions, fit_xgb_churn,
    next_basket_baseline_predictions, tune_decay,
)
from deeppulse.config import LabelConfig
from deeppulse.features import FEATURE_COLUMNS
from deeppulse.metrics import paired_bootstrap, paired_bootstrap_mean, per_customer_ranking
from deeppulse.splits import Fold


def compare_with_baselines(tx: pd.DataFrame, fold: Fold, result: dict, label_cfg: LabelConfig,
                           n_boot: int = 1000, random_state: int = 42) -> dict:
    k = result["config"]["k"]
    rows = result["test_rows"]
    test = fold.test.set_index("customer_id").loc[rows["customer_id"]].reset_index()
    y = rows["churn_w"].astype(int).to_numpy()
    p_model = result["test_lapse_prob"]
    p_xgb = fit_xgb_churn(fold.train, random_state).predict_proba(test[FEATURE_COLUMNS])[:, 1]
    ew = rows["recency"].to_numpy() <= EARLY_WARNING_MAX_RECENCY

    def block(mask: np.ndarray | None) -> dict:
        yy = y if mask is None else y[mask]
        pm = p_model if mask is None else p_model[mask]
        px = p_xgb if mask is None else p_xgb[mask]
        return {
            "n": int(len(yy)),
            "base_rate": float(yy.mean()),
            "model_roc_auc": float(roc_auc_score(yy, pm)),
            "xgboost_roc_auc": float(roc_auc_score(yy, px)),
            "model_pr_auc": float(average_precision_score(yy, pm)),
            "xgboost_pr_auc": float(average_precision_score(yy, px)),
            "diff_roc_auc": paired_bootstrap(roc_auc_score, y, p_model, p_xgb, n_boot, mask=mask),
            "diff_pr_auc": paired_bootstrap(average_precision_score, y, p_model, p_xgb, n_boot, mask=mask),
        }

    lapse = {"all": block(None), f"early_warning_recency_le_{EARLY_WARNING_MAX_RECENCY}": block(ew)}

    targets = result["test_next_basket_targets"]
    keys = [c for c in rows["customer_id"] if targets.get(c)]
    custs = pd.Index(rows["customer_id"])
    _, personal = next_basket_baseline_predictions(tx, custs, fold.test_day, k)
    decay, _ = tune_decay(tx, fold, label_cfg, k)
    decayed = decayed_repeat_predictions(tx, custs, fold.test_day, decay, k)
    m = per_customer_ranking(result["test_next_basket_preds"], targets, keys, k)
    nb = {"n_customers": len(keys), "decay": decay}
    for base_name, preds in (("personal_repeat", personal), ("decayed_repeat", decayed)):
        b = per_customer_ranking(preds, targets, keys, k)
        for name in (f"recall@{k}", f"ndcg@{k}"):
            nb[f"model_{name}"] = float(np.mean(m[name]))
            nb[f"{base_name}_{name}"] = float(np.mean(b[name]))
            nb[f"diff_vs_{base_name}_{name}"] = paired_bootstrap_mean(m[name], b[name], n_boot)
    return {"lapse": lapse, "next_basket": nb}


def verdict(ci: dict) -> str:
    if ci["ci95_low"] > 0:
        return "model daha iyi (aralık 0'ın üstünde)"
    if ci["ci95_high"] < 0:
        return "baseline daha iyi (aralık 0'ın altında)"
    return "fark anlamlı değil (aralık 0'ı içeriyor)"
