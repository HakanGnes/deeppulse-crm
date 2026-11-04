"""Snapshot anındaki müşteri özellikleri (sadece day <= T verisiyle).

Bunlar baseline (XGBoost / CoxPH) özellikleri; AnalyzeWise-CRM'deki RFM+
davranış özelliklerinin Dunnhumby'ye uyarlanmış hali. Derin modeller (Faz 1+)
özet özellik yerine ham sekansı kullanacak, ama karşılaştırma tabanı bunlar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "recency", "tenure", "n_baskets", "n_purchase_days",
    "net_total", "net_per_basket", "items_per_basket",
    "baskets_last_28", "baskets_last_91", "net_last_91",
    "trend_91_vs_prev_91",
    "gap_mean", "gap_std", "gap_last", "recency_over_gap_mean",
    "n_distinct_products", "discount_share", "promo_item_share", "coupon_item_share",
]


HEAVY_TAILED = ("net_total", "net_per_basket", "net_last_91", "n_baskets", "n_distinct_products")


class FeatureScaler:
    """Medyanla doldurma + ağır kuyruklu sütunlarda log1p + z-skor. Yalnızca eğitimle fit edilir."""

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        x = df[FEATURE_COLUMNS].astype(float)
        self.medians_ = x.median()
        z = self._log(x.fillna(self.medians_))
        self.mu_, self.sd_ = z.mean(), z.std().replace(0, 1.0).fillna(1.0)
        return self

    @staticmethod
    def _log(x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        for c in HEAVY_TAILED:
            x[c] = np.log1p(x[c].clip(lower=0))
        return x

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = self._log(df[FEATURE_COLUMNS].astype(float).fillna(self.medians_))
        return ((x - self.mu_) / self.sd_).clip(-6, 6).to_numpy(dtype=np.float32)


def snapshot_features(tx: pd.DataFrame, snapshot_day: int, customers: pd.Index | None = None) -> pd.DataFrame:
    hist = tx[tx["day"] <= snapshot_day]
    if customers is not None:
        hist = hist[hist["customer_id"].isin(set(customers))]
    t = snapshot_day

    baskets = (
        hist.groupby(["customer_id", "basket_id"], sort=False)
        .agg(day=("day", "first"), net=("net_value", "sum"), n_items=("product_id", "size"))
        .reset_index()
    )
    g = baskets.groupby("customer_id")
    f = pd.DataFrame({
        "first_day": g["day"].min(),
        "last_day": g["day"].max(),
        "n_baskets": g.size(),
        "net_total": g["net"].sum(),
        "items_per_basket": g["n_items"].mean(),
    })
    f["recency"] = t - f["last_day"]
    f["tenure"] = t - f["first_day"]
    f["net_per_basket"] = f["net_total"] / f["n_baskets"]

    def window_count(lo: int, hi: int) -> pd.Series:
        m = (baskets["day"] > t - hi) & (baskets["day"] <= t - lo)
        return baskets[m].groupby("customer_id").size()

    f["baskets_last_28"] = window_count(0, 28)
    f["baskets_last_91"] = window_count(0, 91)
    prev_91 = window_count(91, 182)
    recent = baskets[baskets["day"] > t - 91]
    f["net_last_91"] = recent.groupby("customer_id")["net"].sum()
    f[["baskets_last_28", "baskets_last_91", "net_last_91"]] = (
        f[["baskets_last_28", "baskets_last_91", "net_last_91"]].fillna(0)
    )
    f["trend_91_vs_prev_91"] = (f["baskets_last_91"] + 1) / (prev_91.reindex(f.index).fillna(0) + 1)

    pdays = baskets[["customer_id", "day"]].drop_duplicates().sort_values(["customer_id", "day"])
    pdays["gap"] = pdays.groupby("customer_id")["day"].diff()
    gg = pdays.groupby("customer_id")
    f["n_purchase_days"] = gg.size()
    f["gap_mean"] = gg["gap"].mean()
    f["gap_std"] = gg["gap"].std()
    f["gap_last"] = gg["gap"].last()
    f["recency_over_gap_mean"] = f["recency"] / f["gap_mean"]

    h = hist.groupby("customer_id")
    gross = h["gross_value"].sum()
    f["n_distinct_products"] = h["product_id"].nunique()
    f["discount_share"] = (h["retail_disc"].sum() + h["coupon_match_disc"].sum()) / gross.replace(0, np.nan)
    flags = pd.DataFrame({
        "customer_id": hist["customer_id"],
        "promo": hist["retail_disc"] > 0,
        "coupon": hist["coupon_disc"] > 0,
    }).groupby("customer_id").mean()
    f["promo_item_share"] = flags["promo"]
    f["coupon_item_share"] = flags["coupon"]

    f = f[FEATURE_COLUMNS]
    if customers is not None:
        f = f.reindex(customers)
    f.index.name = "customer_id"
    return f
