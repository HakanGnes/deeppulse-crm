"""Faz 0 baseline'ları — derin modellerin geçmesi gereken çıta.

1. Churn (W günde terk):   recency tek başına  |  XGBoost (özet özellikler)
2. Survival (terke süre):  CoxPH (lifelines)   →  C-index, ayrıca 1-S(W) ile churn AUC
3. Sonraki sepet:          global popülerlik   |  kişisel tekrar sıklığı
   Market verisinde kişisel tekrar baseline'ı genelde çok güçlüdür; sekans
   modeli (Faz 1) bunu geçemezse, bu dürüstçe raporlanacak bir sonuçtur.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from deeppulse.config import LabelConfig
from deeppulse.features import FEATURE_COLUMNS
from deeppulse.metrics import ranking_report
from deeppulse.splits import Fold


def _binary_report(y: np.ndarray, p: np.ndarray, is_prob: bool = True) -> dict[str, float]:
    out = {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "base_rate": float(np.mean(y)),
    }
    if is_prob:
        out["brier"] = float(brier_score_loss(y, p))
    return out


# Erken uyarı alt kümesi: hâlâ normal ritminde görünen müşteriler.
# Recency eşiğe yaklaştıkça kopma neredeyse kesinleşir (recency 55 olan biri
# 5 gün içinde gelmezse zaten "kopmuş" sayılır); genel AUC'yi bu kolay vakalar
# şişirir. İşe yarayan tahmin, sessizlik henüz başlamamışken yapılandır.
EARLY_WARNING_MAX_RECENCY = 14


def _early_warning(y: np.ndarray, scores: dict[str, np.ndarray], recency: np.ndarray) -> dict:
    mask = recency <= EARLY_WARNING_MAX_RECENCY
    out: dict = {
        "definition": f"recency <= {EARLY_WARNING_MAX_RECENCY}",
        "n": int(mask.sum()),
        "base_rate": float(y[mask].mean()) if mask.any() else np.nan,
    }
    two_classes = mask.any() and len(np.unique(y[mask])) == 2
    for name, s in scores.items():
        out[f"{name}_roc_auc"] = float(roc_auc_score(y[mask], s[mask])) if two_classes else np.nan
        out[f"{name}_pr_auc"] = float(average_precision_score(y[mask], s[mask])) if two_classes else np.nan
    return out


# --- 1. Churn (bu projede: kopma / lapse — docs/decisions.md K1) -------------

def fit_xgb_churn(train: pd.DataFrame, random_state: int = 42) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
        random_state=random_state, n_jobs=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train["churn_w"].astype(int).to_numpy())
    return model


def churn_baselines(fold: Fold, random_state: int = 42) -> dict[str, dict]:
    tr, te = fold.train, fold.test
    y_te = te["churn_w"].astype(int).to_numpy()

    results = {"recency_only": _binary_report(y_te, te["recency"].to_numpy(), is_prob=False)}

    model = fit_xgb_churn(tr, random_state)
    p = model.predict_proba(te[FEATURE_COLUMNS])[:, 1]
    results["xgboost"] = _binary_report(y_te, p)
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    results["xgboost"]["top_features"] = importance.head(5).round(4).to_dict()
    recency = te["recency"].to_numpy()
    results["early_warning"] = _early_warning(
        y_te, {"recency_only": recency.astype(float), "xgboost": p}, recency
    )
    return results


# --- 2. Survival ------------------------------------------------------------

def _prep_survival(df: pd.DataFrame, medians: pd.Series, mu: pd.Series, sd: pd.Series) -> pd.DataFrame:
    x = df[FEATURE_COLUMNS].astype(float).fillna(medians)
    # Ağır kuyruklu tutar/sayı özelliklerinde log ölçeği Cox için daha kararlı
    for c in ("net_total", "net_per_basket", "net_last_91", "n_baskets", "n_distinct_products"):
        x[c] = np.log1p(x[c].clip(lower=0))
    return (x - mu) / sd


def survival_baselines(fold: Fold, cfg: LabelConfig, penalizer: float = 0.05) -> dict[str, dict]:
    tr, te = fold.train, fold.test
    raw_tr = tr[FEATURE_COLUMNS].astype(float)
    medians = raw_tr.median()
    tmp = _prep_survival(tr, medians, pd.Series(0.0, index=FEATURE_COLUMNS), pd.Series(1.0, index=FEATURE_COLUMNS))
    mu, sd = tmp.mean(), tmp.std().replace(0, 1.0)

    x_tr = _prep_survival(tr, medians, mu, sd)
    x_te = _prep_survival(te, medians, mu, sd)
    df_tr = x_tr.assign(duration=tr["duration"].to_numpy(), event=tr["event"].to_numpy())

    cph = CoxPHFitter(penalizer=penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df_tr, duration_col="duration", event_col="event")

    risk = cph.predict_partial_hazard(x_te).to_numpy().ravel()
    c_index = concordance_index(te["duration"], -risk, te["event"])
    w = cfg.prediction_window_days
    surv_w = cph.predict_survival_function(x_te, times=[w]).to_numpy().ravel()
    y_te = te["churn_w"].astype(int).to_numpy()
    churn_from_surv = _binary_report(y_te, 1 - surv_w)
    ew = _early_warning(y_te, {"coxph": 1 - surv_w}, te["recency"].to_numpy())
    return {
        "coxph": {
            "c_index": float(c_index),
            "test_event_rate": float(te["event"].mean()),
            "test_max_followup_days": int(te["duration"].max()),
            f"churn_auc_from_1_minus_S({w})": churn_from_surv["roc_auc"],
            f"brier_from_1_minus_S({w})": churn_from_surv["brier"],
            "early_warning_churn_auc": ew["coxph_roc_auc"],
        }
    }


# --- 3. Sonraki sepet --------------------------------------------------------

def next_basket_targets(tx: pd.DataFrame, customers: pd.Index, snapshot_day: int, window: int) -> dict:
    """Snapshot'tan sonraki İLK sepetin ürün kümesi (pencere içinde alışveriş yoksa boş)."""
    fut = tx[(tx["day"] > snapshot_day) & (tx["day"] <= snapshot_day + window)
             & tx["customer_id"].isin(set(customers))]
    first = (
        fut.sort_values(["day", "trans_time"])
        .groupby("customer_id")["basket_id"].first()
    )
    rows = fut[fut["basket_id"].isin(set(first.to_numpy()))]
    return rows.groupby("customer_id")["product_id"].apply(set).to_dict()


def next_basket_baseline_predictions(
    tx: pd.DataFrame, customers: pd.Index, snapshot_day: int, k: int = 10
) -> tuple[dict, dict]:
    """(popülerlik, kişisel tekrar) top-k önerileri — yalnızca day <= snapshot_day verisiyle."""
    t = snapshot_day
    hist = tx[(tx["day"] <= t) & tx["customer_id"].isin(set(customers))]
    recent = tx[(tx["day"] > t - 91) & (tx["day"] <= t)]
    popular = (
        recent.drop_duplicates(["basket_id", "product_id"])["product_id"]
        .value_counts().index[: max(k * 5, 50)].tolist()
    )
    pop_preds = {c: popular[:k] for c in customers}

    # Kişisel tekrar: ürünü içeren sepet sayısı, eşitlikte en son alınan önce
    per = (
        hist.drop_duplicates(["customer_id", "basket_id", "product_id"])
        .groupby(["customer_id", "product_id"])
        .agg(n=("basket_id", "size"), last=("day", "max"))
        .reset_index()
        .sort_values(["customer_id", "n", "last"], ascending=[True, False, False])
    )
    top_personal = per.groupby("customer_id")["product_id"].apply(lambda s: s.head(k).tolist()).to_dict()
    personal_preds = {}
    for c in customers:
        own = top_personal.get(c, [])
        fill = [p for p in popular if p not in set(own)]
        personal_preds[c] = (own + fill)[:k]
    return pop_preds, personal_preds


DECAY_GRID = (1.0, 0.99, 0.97, 0.95, 0.9, 0.8, 0.7)


def decayed_repeat_predictions(
    tx: pd.DataFrame, customers: pd.Index, snapshot_day: int, decay: float, k: int = 10
) -> dict:
    """Zaman ağırlıklı kişisel tekrar: skor(i) = Σ_{i'yi içeren sepetler} decay^(o sepetten bu yana sepet sayısı).

    decay = 1 ⇒ düz kişisel tekrar (sayım). decay < 1 ⇒ son sepetler daha ağır basar.
    Sinir ağı içermeyen, ama "ne zaman aldı" bilgisini kullanan güçlü bir baseline;
    derin modelin kazancının sıralama/zaman bilgisinden mi yoksa öğrenmeden mi geldiğini ayırır.
    """
    t = snapshot_day
    hist = tx[(tx["day"] <= t) & tx["customer_id"].isin(set(customers))]
    lines = hist.drop_duplicates(["customer_id", "basket_id", "product_id"])[
        ["customer_id", "basket_id", "product_id", "day", "trans_time"]
    ]
    order = (
        lines[["customer_id", "basket_id", "day", "trans_time"]].drop_duplicates("basket_id")
        .sort_values(["customer_id", "day", "trans_time", "basket_id"], ascending=[True, False, False, False])
    )
    order["age"] = order.groupby("customer_id").cumcount()          # 0 = en son sepet
    lines = lines.merge(order[["basket_id", "age"]], on="basket_id")
    lines["w"] = np.power(decay, lines["age"].to_numpy(dtype=float))
    per = (
        lines.groupby(["customer_id", "product_id"])
        .agg(score=("w", "sum"), last=("day", "max")).reset_index()
        .sort_values(["customer_id", "score", "last"], ascending=[True, False, False])
    )
    top = per.groupby("customer_id")["product_id"].apply(lambda s: s.head(k).tolist()).to_dict()
    popular = _popular_items(tx, t, k)
    out = {}
    for c in customers:
        own = top.get(c, [])
        seen = set(own)
        out[c] = (own + [p for p in popular if p not in seen])[:k]
    return out


def _popular_items(tx: pd.DataFrame, t: int, k: int) -> list:
    recent = tx[(tx["day"] > t - 91) & (tx["day"] <= t)]
    return (
        recent.drop_duplicates(["basket_id", "product_id"])["product_id"]
        .value_counts().index[: max(k * 5, 50)].tolist()
    )


def tune_decay(tx: pd.DataFrame, fold: Fold, cfg: LabelConfig, k: int = 10) -> tuple[float, dict]:
    """decay'i son eğitim snapshot'ında seç (yalnızca test gününe kadarki veriyle; test sızmaz)."""
    t_val = max(fold.train_days)
    known = tx[tx["day"] <= fold.test_day]
    customers = pd.Index(fold.train.loc[fold.train["snapshot_day"] == t_val, "customer_id"])
    targets = next_basket_targets(known, customers, t_val, cfg.prediction_window_days)
    scores = {}
    for d in DECAY_GRID:
        preds = decayed_repeat_predictions(known, customers, t_val, d, k)
        scores[d] = ranking_report(preds, targets, k)[f"ndcg@{k}"]
    best = max(scores, key=scores.get)
    return best, scores


def next_basket_baselines(tx: pd.DataFrame, fold: Fold, cfg: LabelConfig, k: int = 10) -> dict[str, dict]:
    t = fold.test_day
    customers = pd.Index(fold.test["customer_id"])
    targets = next_basket_targets(tx, customers, t, cfg.prediction_window_days)
    pop_preds, personal_preds = next_basket_baseline_predictions(tx, customers, t, k)
    decay, grid = tune_decay(tx, fold, cfg, k)
    decayed = decayed_repeat_predictions(tx, customers, t, decay, k)
    return {
        "popularity": ranking_report(pop_preds, targets, k),
        "personal_repeat": ranking_report(personal_preds, targets, k),
        "decayed_repeat": {**ranking_report(decayed, targets, k), "decay": decay,
                           "val_ndcg_by_decay": {str(d): round(v, 4) for d, v in grid.items()}},
    }


def run_all(tx: pd.DataFrame, fold: Fold, cfg: LabelConfig, random_state: int = 42) -> dict:
    return {
        "setup": {
            "churn_inactivity_days": cfg.churn_inactivity_days,
            "prediction_window_days": cfg.prediction_window_days,
            "test_day": fold.test_day,
            "n_train_snapshots": len(fold.train_days),
            "n_train_rows": len(fold.train),
            "n_test_customers": len(fold.test),
        },
        "churn": churn_baselines(fold, random_state),
        "survival": survival_baselines(fold, cfg),
        "next_basket": next_basket_baselines(tx, fold, cfg),
    }
