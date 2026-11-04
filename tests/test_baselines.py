import math

import pytest

from deeppulse.baselines import next_basket_targets, run_all
from deeppulse.config import LabelConfig
from deeppulse.metrics import hit_at_k, ndcg_at_k, recall_at_k
from deeppulse.splits import temporal_fold

from conftest import make_tx


def test_ranking_metrics():
    pred = ["a", "b", "c"]
    assert recall_at_k(pred, {"a", "z"}, 3) == 0.5
    assert hit_at_k(pred, {"z"}, 3) == 0.0
    assert ndcg_at_k(pred, {"a"}, 3) == 1.0
    assert ndcg_at_k(pred, {"b"}, 3) == pytest.approx(1 / math.log2(3))
    assert math.isnan(recall_at_k(pred, set(), 3))


def test_decayed_repeat_prefers_recent_items():
    from deeppulse.baselines import decayed_repeat_predictions

    # "old" 3 eski sepette, "new" son 2 sepette
    tx = make_tx({"A": [1, 2, 3, 10, 11]})
    tx.loc[tx["day"] <= 3, "product_id"] = "old"
    tx.loc[tx["day"] >= 10, "product_id"] = "new"
    flat = decayed_repeat_predictions(tx, ["A"], 20, decay=1.0, k=2)["A"]
    decayed = decayed_repeat_predictions(tx, ["A"], 20, decay=0.5, k=2)["A"]
    assert flat[0] == "old"      # sayımda 3 > 2
    assert decayed[0] == "new"   # 1 + 0.5 = 1.5 > 0.25 + 0.125 + 0.0625


def test_paired_bootstrap_detects_real_and_null_differences():
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from deeppulse.metrics import paired_bootstrap

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = y + rng.normal(0, 0.5, 400)
    noise = rng.normal(0, 1, 400)
    same = paired_bootstrap(roc_auc_score, y, good, good, n_boot=200)
    assert same["diff"] == 0 and same["ci95_low"] <= 0 <= same["ci95_high"]
    better = paired_bootstrap(roc_auc_score, y, good, noise, n_boot=200)
    assert better["ci95_low"] > 0


def test_next_basket_target_is_first_basket_after_snapshot():
    tx = make_tx({"A": [5, 12, 15]})
    tx.loc[tx["day"] == 12, "product_id"] = "x"
    tx.loc[tx["day"] == 15, "product_id"] = "y"
    targets = next_basket_targets(tx, ["A"], snapshot_day=10, window=30)
    assert targets == {"A": {"x"}}


@pytest.fixture(scope="module")
def report(synthetic_canonical):
    tx, _ = synthetic_canonical
    cfg = LabelConfig()
    return run_all(tx, temporal_fold(tx, cfg), cfg)


def test_report_shape(report):
    assert set(report) == {"setup", "churn", "survival", "next_basket"}
    assert set(report["churn"]) == {"recency_only", "xgboost", "early_warning"}
    ew = report["churn"]["early_warning"]
    assert 0 < ew["n"] <= report["setup"]["n_test_customers"]
    assert "early_warning_churn_auc" in report["survival"]["coxph"]
    assert set(report["next_basket"]) == {"popularity", "personal_repeat", "decayed_repeat"}
    assert report["next_basket"]["decayed_repeat"]["decay"] in (1.0, 0.99, 0.97, 0.95, 0.9, 0.8, 0.7)


def test_baselines_find_signal_in_synthetic_data(report):
    # Sentetik veriye bilerek sinyal konuldu; baseline'lar şanstan iyi olmalı
    assert report["churn"]["xgboost"]["roc_auc"] > 0.6
    assert report["survival"]["coxph"]["c_index"] > 0.6
    nb = report["next_basket"]
    assert nb["personal_repeat"]["recall@10"] > nb["popularity"]["recall@10"]
