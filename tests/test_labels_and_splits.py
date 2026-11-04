import pandas as pd
import pytest

from deeppulse.config import LabelConfig
from deeppulse.features import snapshot_features
from deeppulse.labels import label_snapshot, purchase_days
from deeppulse.splits import temporal_fold

from conftest import make_tx

CFG = LabelConfig(churn_inactivity_days=30, prediction_window_days=30,
                  min_history_days=0, min_baskets=1, snapshot_step_days=14)


def _row(lab: pd.DataFrame, cust: str) -> pd.Series:
    return lab.set_index("customer_id").loc[cust]


def test_event_is_first_gap_longer_than_n():
    tx = make_tx({"A": [1, 5, 10, 50]})
    r = _row(label_snapshot(tx, 12, 100, CFG), "A")
    # 10 → 50 arası 40 gün > 30 ⇒ olay günü 10 + 30 = 40, süre 28
    assert (r["event"], r["duration"], r["churn_w"]) == (1, 28, 1)


def test_no_long_gap_event_after_last_purchase():
    tx = make_tx({"B": [1, 5, 10, 20, 35, 60]})
    r = _row(label_snapshot(tx, 12, 100, CFG), "B")
    # Hiçbir boşluk > 30 değil; son alışveriş 60 ⇒ olay 90, süre 78
    assert (r["event"], r["duration"], r["churn_w"]) == (1, 78, 0)


def test_right_censoring_when_event_beyond_horizon():
    tx = make_tx({"B": [1, 5, 10, 20, 35, 60]})
    r = _row(label_snapshot(tx, 12, 80, CFG), "B")
    assert (r["event"], r["duration"]) == (0, 68)


def test_inactive_customers_are_excluded():
    tx = make_tx({"A": [1, 3], "B": [1, 39]})
    lab = label_snapshot(tx, 40, 100, CFG)
    assert set(lab["customer_id"]) == {"B"}  # A'nın recency'si 37 >= 30


def test_purchases_beyond_horizon_are_unknown():
    tx = make_tx({"C": [1, 5, 10, 35]})
    # Ufuk 30: 35'teki alışveriş bilinmiyor; olay günü 40 > 30 ⇒ sansürlü
    r = _row(label_snapshot(tx, 12, 30, CFG), "C")
    assert (r["event"], r["duration"]) == (0, 18)
    # Ufuk 100: boşluk 25 < 30, son alışveriş 35 ⇒ olay 65
    r = _row(label_snapshot(tx, 12, 100, CFG), "C")
    assert (r["event"], r["duration"]) == (1, 53)


def test_return_rate_counts_only_followable_lapses():
    from deeppulse.labels import return_rate
    # A: 10'dan sonra 40 günlük boşluk, sonra geri dönüyor (olay günü 40, dönüş 50)
    # B: 5'ten sonra hiç gelmiyor (olay günü 35, dönüş yok)
    # C: 1 → 140 boşluğu (olay 31, dönüş 140) sayılır; 140'tan sonraki olay
    #    (170 + 180 > ufuk) takip edilemez, sayılmaz
    tx = make_tx({"A": [1, 10, 50, 60, 70, 80], "B": [1, 5], "C": [1, 140]})
    pdays = purchase_days(tx)
    r = return_rate(pdays, n=30, horizon=300, follow_up_days=180)
    # A'nın son alışverişi 80 → olay 110, 110+180=290 ≤ 300 ⇒ dönmeyen bir olay daha
    # Olaylar: A@40 (döndü), A@110 (dönmedi), B@35 (dönmedi), C@31 (döndü)
    assert r["n_lapse_events"] == 4
    assert r["returned_within_180d"] == 0.5


def test_precomputed_purchase_days_path_matches(synthetic_canonical):
    tx, _ = synthetic_canonical
    cfg = LabelConfig()
    a = label_snapshot(tx, 400, 500, cfg)
    b = label_snapshot(tx, 400, 500, cfg, pdays=purchase_days(tx))
    pd.testing.assert_frame_equal(a, b)


def test_features_use_only_past_data(synthetic_canonical):
    tx, _ = synthetic_canonical
    t = 300
    full = snapshot_features(tx, t)
    past_only = snapshot_features(tx[tx["day"] <= t], t)
    pd.testing.assert_frame_equal(full, past_only)


def test_fold_train_is_invariant_to_data_after_test_day(synthetic_canonical):
    """Sızıntı testi: test gününden sonraki veriyi değiştirmek eğitimi DEĞİŞTİRMEMELİ."""
    tx, _ = synthetic_canonical
    cfg = LabelConfig()
    fold = temporal_fold(tx, cfg)
    tampered = tx[tx["day"] <= fold.test_day]  # geleceği tamamen sil
    tampered = pd.concat([tampered, tx[tx["day"] > fold.test_day].sample(frac=0.3, random_state=0)])
    fold2 = temporal_fold(tampered, cfg, test_day=fold.test_day)
    pd.testing.assert_frame_equal(fold.train, fold2.train)


def test_train_label_windows_close_before_test_day(synthetic_canonical):
    tx, _ = synthetic_canonical
    cfg = LabelConfig()
    fold = temporal_fold(tx, cfg)
    assert max(fold.train_days) + cfg.prediction_window_days <= fold.test_day
    # Eğitimdeki hiçbir süre test gününü aşmaz
    assert (fold.train["snapshot_day"] + fold.train["duration"] <= fold.test_day).all()
    assert fold.train["churn_w"].notna().all() and fold.test["churn_w"].notna().all()


def test_fold_rejects_unlabelable_test_day(synthetic_canonical):
    tx, _ = synthetic_canonical
    with pytest.raises(ValueError):
        temporal_fold(tx, LabelConfig(), test_day=int(tx["day"].max()) - 10)
