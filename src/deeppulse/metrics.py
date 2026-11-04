"""Öneri metrikleri (Recall@k, NDCG@k, HitRate@k)."""

from __future__ import annotations

import numpy as np


def recall_at_k(pred: list, target: set, k: int) -> float:
    if not target:
        return np.nan
    return len(set(pred[:k]) & target) / len(target)


def hit_at_k(pred: list, target: set, k: int) -> float:
    if not target:
        return np.nan
    return float(bool(set(pred[:k]) & target))


def ndcg_at_k(pred: list, target: set, k: int) -> float:
    if not target:
        return np.nan
    dcg = sum(1.0 / np.log2(i + 2) for i, p in enumerate(pred[:k]) if p in target)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(target))))
    return dcg / ideal


def per_customer_ranking(preds: dict, targets: dict, keys: list, k: int = 10) -> dict[str, np.ndarray]:
    return {
        f"recall@{k}": np.array([recall_at_k(preds.get(c, []), targets[c], k) for c in keys]),
        f"ndcg@{k}": np.array([ndcg_at_k(preds.get(c, []), targets[c], k) for c in keys]),
    }


def paired_bootstrap(
    metric, y: np.ndarray, a: np.ndarray, b: np.ndarray,
    n_boot: int = 1000, seed: int = 0, mask: np.ndarray | None = None,
) -> dict[str, float]:
    """metric(y, a) − metric(y, b) farkının eşli (aynı örneklemle) bootstrap %95 güven aralığı.

    Aynı müşteri yeniden örneklemesi iki modele birden uygulanır; böylece iki model
    arasındaki fark, test kümesinin şansına bağlı gürültüden ayrılır.
    Aralık 0'ı içeriyorsa "a, b'den iyidir" denemez.
    """
    if mask is not None:
        y, a, b = y[mask], a[mask], b[mask]
    rng = np.random.default_rng(seed)
    n = len(y)
    point = float(metric(y, a) - metric(y, b))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():  # tek sınıf kaldıysa AUC tanımsız
            continue
        diffs.append(metric(yy, a[idx]) - metric(yy, b[idx]))
    diffs = np.asarray(diffs)
    return {
        "diff": point,
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "p_a_not_better": float((diffs <= 0).mean()),
        "n": int(n),
    }


def paired_bootstrap_mean(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """Müşteri başına skorların (ör. recall@10) ortalama farkı için eşli bootstrap."""
    d = a - b
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {
        "diff": float(d.mean()),
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "p_a_not_better": float((boots <= 0).mean()),
        "n": int(len(d)),
    }


def ranking_report(preds: dict, targets: dict, k: int = 10) -> dict[str, float]:
    keys = [c for c in targets if targets[c]]
    return {
        f"recall@{k}": float(np.mean([recall_at_k(preds.get(c, []), targets[c], k) for c in keys])),
        f"ndcg@{k}": float(np.mean([ndcg_at_k(preds.get(c, []), targets[c], k) for c in keys])),
        f"hit@{k}": float(np.mean([hit_at_k(preds.get(c, []), targets[c], k) for c in keys])),
        "n_customers": len(keys),
    }
