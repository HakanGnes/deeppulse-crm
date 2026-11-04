"""Churn ve survival etiketleri.

Tanım (docs/decisions.md → K1):
    Bir müşteri, art arda N gün hiç alışveriş yapmazsa terk etmiş sayılır.
    Olay günü = son alışveriş günü + N  (eşiğin aşıldığı, terkin "kesinleştiği" gün).

Snapshot günü T için (özellikler sadece day <= T verisinden üretilir):
    - Sadece AKTİF müşteriler alınır: recency = T - son_alışveriş < N
      (zaten terk etmiş birini "terk edecek mi" diye sormak anlamsız).
    - duration = olay_günü - T  (her zaman >= 1)
    - event    = 1 olay veri ufku H içinde gözlendiyse, 0 ise sağdan sansürlü;
                 sansürlüyse duration = H - T.
    - churn_w  = event == 1 ve duration <= W  (W günlük pencerede terk)
      Bunun tanımlı olması için T + W <= H olmalı.

Böylece "kaç gün sonra terk eder?" sorusu "N günlük sessizlik eşiğine kaç günde
ulaşır?" sorusuna dönüşür; BG/NBD'deki gizli (gözlenmeyen) terk yerine
gözlenebilir bir olay elde edilir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from deeppulse.config import LabelConfig


def purchase_days(tx: pd.DataFrame) -> pd.DataFrame:
    """Müşteri başına benzersiz alışveriş günleri + bir sonraki alışverişe kadar boşluk."""
    pdays = tx[["customer_id", "day"]].drop_duplicates().sort_values(["customer_id", "day"])
    pdays["next_day"] = pdays.groupby("customer_id")["day"].shift(-1)
    pdays["gap_after"] = pdays["next_day"] - pdays["day"]
    return pdays.reset_index(drop=True)


def inter_purchase_gaps(tx: pd.DataFrame) -> np.ndarray:
    return purchase_days(tx)["gap_after"].dropna().to_numpy()


def snapshot_days(min_day: int, horizon: int, cfg: LabelConfig) -> list[int]:
    """Etiketlenebilir snapshot günleri, en yeniden geriye `snapshot_step_days` aralıkla."""
    last = horizon - cfg.prediction_window_days
    first = min_day + cfg.min_history_days
    days = list(range(last, first - 1, -cfg.snapshot_step_days))
    return sorted(days)


def label_snapshot(
    tx: pd.DataFrame,
    snapshot_day: int,
    horizon: int,
    cfg: LabelConfig,
    pdays: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Tek bir snapshot için aktif müşterileri ve etiketlerini döndürür.

    `horizon`: veri ufku (bu günden sonrası bilinmiyor sayılır). Eğitim
    katmanlarında test gününe eşitlenerek gelecekten sızıntı engellenir.
    """
    n = cfg.churn_inactivity_days
    if pdays is None:
        pdays = purchase_days(tx[tx["day"] <= horizon])
    else:
        pdays = pdays[pdays["day"] <= horizon].copy()
        # Ufuk ötesindeki bir sonraki alışveriş bilinmiyor sayılır
        beyond = pdays["next_day"] > horizon
        pdays.loc[beyond, ["next_day", "gap_after"]] = np.nan

    hist = pdays[pdays["day"] <= snapshot_day]
    stats = hist.groupby("customer_id")["day"].agg(first_day="min", last_day="max")
    baskets = (
        tx[(tx["day"] <= snapshot_day)].groupby("customer_id")["basket_id"].nunique().rename("n_baskets")
    )
    stats = stats.join(baskets, how="left")
    stats["recency"] = snapshot_day - stats["last_day"]
    eligible = stats[
        (stats["recency"] < n)
        & (stats["first_day"] <= snapshot_day - cfg.min_history_days)
        & (stats["n_baskets"] >= cfg.min_baskets)
    ]

    # Son alışverişten (dahil) itibaren ilk "N günden uzun boşluk" başlangıcını bul
    cand = pdays.merge(eligible[["last_day"]], left_on="customer_id", right_index=True)
    cand = cand[cand["day"] >= cand["last_day"]]
    spell = cand[(cand["gap_after"] > n) | cand["gap_after"].isna()]
    spell_start = spell.groupby("customer_id")["day"].min()

    out = pd.DataFrame(index=eligible.index)
    out["snapshot_day"] = snapshot_day
    event_day = spell_start.reindex(out.index) + n
    out["event"] = (event_day <= horizon).astype("int8")
    out["duration"] = np.where(out["event"] == 1, event_day - snapshot_day, horizon - snapshot_day)
    out["duration"] = out["duration"].astype("int32")
    w = cfg.prediction_window_days
    if snapshot_day + w <= horizon:
        out["churn_w"] = ((out["event"] == 1) & (out["duration"] <= w)).astype("int8")
    else:
        out["churn_w"] = pd.array([pd.NA] * len(out), dtype="Int8")
    return out.reset_index()


def suggest_threshold_table(
    tx: pd.DataFrame,
    candidates: tuple[int, ...] = (14, 21, 30, 45, 60, 90),
    prediction_window_days: int = 90,
) -> pd.DataFrame:
    """Aday N değerleri için boşluk dağılımı ve son snapshot'taki churn oranı."""
    gaps = inter_purchase_gaps(tx)
    horizon = int(tx["day"].max())
    rows = []
    pdays = purchase_days(tx)
    for n in candidates:
        cfg = LabelConfig(churn_inactivity_days=n, prediction_window_days=prediction_window_days)
        t = horizon - prediction_window_days
        lab = label_snapshot(tx, t, horizon, cfg, pdays=pdays)
        rows.append({
            "N": n,
            "share_gaps_longer_than_N": float((gaps > n).mean()),
            "customers_with_any_gap_over_N": float(
                pd.Series(gaps > n).groupby(
                    pdays.dropna(subset=["gap_after"])["customer_id"].to_numpy()
                ).any().mean()
            ),
            "active_at_snapshot": int(len(lab)),
            "churn_rate_in_window": float(lab["churn_w"].astype(float).mean()) if len(lab) else np.nan,
            **return_rate(pdays, n, horizon),
        })
    return pd.DataFrame(rows)


def return_rate(pdays: pd.DataFrame, n: int, horizon: int, follow_up_days: int = 180) -> dict[str, float]:
    """"Terk" olaylarının ne kadarı aslında geçici bir mola?

    Gerçekleşmiş her N-günlük sessizlik (olay günü = alışveriş + N) için, olaydan
    sonraki `follow_up_days` gün içinde müşteri geri döndü mü? Yalnızca bu kadar
    takip süresi veride bulunan olaylar sayılır.
    Oran yüksekse N "terk"i değil "mola"yı ölçüyor demektir.
    """
    event_day = pdays["day"] + n
    completed = (pdays["gap_after"] > n) | pdays["gap_after"].isna()
    followable = event_day + follow_up_days <= horizon
    spells = pdays[completed & followable]
    returned = (spells["next_day"] - (spells["day"] + n)) <= follow_up_days
    return {
        "n_lapse_events": int(len(spells)),
        f"returned_within_{follow_up_days}d": float(returned.mean()) if len(spells) else np.nan,
    }
