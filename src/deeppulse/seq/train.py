"""Faz 1 eğitim ve değerlendirme.

Doğrulama: müşterilerin %10'u eğitimden tamamen ayrılır (ne sonraki sepet ne kopma
kaybına girer). Erken durdurma bu müşterilerde ölçülür:
    sonraki sepet — son eğitim snapshot'ında (T_val = max(train_days)) NDCG@10
    kopma         — doğrulama müşterilerinin tüm eğitim snapshot'larında PR AUC
İki başlık farklı hızda öğrendiği için (kopma birkaç epoch'ta doyuyor, sonraki
sepet daha yavaş) her başlığın kendi en iyi checkpoint'i ayrı tutulur ve testte
o checkpoint kullanılır. Hiçbiri `patience` epoch boyunca iyileşmezse eğitim durur.

Doğrulama hedefleri de test gününden önceki veriden gelir; test dönemine dair
hiçbir bilgi model seçimine girmez. Ayrılan müşteriler test kümesinde kalır
(geçmişleri yalnızca model seçiminde kullanıldı, eğitimde değil).
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from deeppulse.baselines import next_basket_targets
from deeppulse.config import LabelConfig
from deeppulse.features import FEATURE_COLUMNS, FeatureScaler
from deeppulse.metrics import ranking_report
from deeppulse.seq.data import (
    Batch, BasketStore, Vocab, next_basket_batch, sample_next_basket_ends, snapshot_batch,
)
from deeppulse.seq.model import DeepPulseSeq, ModelConfig, next_basket_loss
from deeppulse.splits import Fold


@dataclass(frozen=True)
class SeqConfig:
    encoder: str = "sasrec"
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    dropout: float = 0.2
    max_len: int = 50
    max_items: int = 32
    min_item_count: int = 5
    batch_size: int = 128
    lr: float = 3e-4                      # Faz 1.1: 1e-3'te sonraki sepet epoch 1'den sonra geriliyordu
    weight_decay: float = 0.01
    epochs: int = 30
    patience: int = 4
    n_negatives: int = 2048
    max_candidates: int = 8192          # 8 GB GPU için aday üst sınırı (OOM olursa düşür)
    windows_per_customer: int = 4
    lapse_weight: float = 1.0
    lapse_summary_features: bool = True   # kopma başlığı Faz 0 özelliklerini de görsün mü
    repeat_aware: bool = True             # sonraki sepette tekrar bonusu
    logq_correction: bool = True          # sampled softmax popülerlik yanlılığı düzeltmesi (Faz 1.1)
    val_frac: float = 0.1
    k: int = 10
    seed: int = 42


def _to_device(b: Batch, device: torch.device) -> dict[str, torch.Tensor]:
    out = {
        "items": torch.from_numpy(b.items).to(device),
        "feats": torch.from_numpy(b.feats).to(device),
        "weekday": torch.from_numpy(b.weekday).to(device),
        "lengths": torch.from_numpy(b.lengths).to(device),
    }
    for name in ("target_items", "recency", "label", "summary"):
        v = getattr(b, name)
        if v is not None:
            out[name] = torch.from_numpy(v).to(device)
    return out


@torch.no_grad()
def predict_snapshot(
    model: DeepPulseSeq, store: BasketStore, rows: pd.DataFrame, cfg: SeqConfig,
    device: torch.device, vocab: Vocab | None = None, summary: np.ndarray | None = None,
    chunk: int = 256,
) -> tuple[np.ndarray, dict]:
    """Kopma olasılıkları (rows sırasıyla) ve vocab verilirse top-k ürün önerileri."""
    model.eval()
    probs, preds = [], {}
    custs = rows["customer_id"].to_numpy()
    for i in range(0, len(rows), chunk):
        part = rows.iloc[i:i + chunk]
        summ = None if summary is None else summary[i:i + chunk]
        t = _to_device(snapshot_batch(store, part, cfg.max_len, summ), device)
        h = model.encode(t["items"], t["feats"], t["weekday"], t["lengths"])
        h_last = model.last_state(h, t["lengths"]).float()
        if summary is not None or not model.cfg.n_summary_features:
            probs.append(torch.sigmoid(model.lapse_logit(h_last, t["recency"], t.get("summary"))).cpu().numpy())
        if vocab is not None:
            top = model.score_all_items(h_last, t["items"]).topk(cfg.k, dim=1).indices.cpu().numpy()
            for c, row in zip(custs[i:i + chunk], top):
                preds[c] = vocab.product_ids[row].tolist()
    return np.concatenate(probs) if probs else np.array([]), preds


def _cycle(order: np.ndarray, step: int, size: int) -> np.ndarray:
    """`order` dizisinden step'inci partiyi al; sona gelince başa sar."""
    n = len(order)
    start = (step * size) % n
    idx = order[start:start + size]
    if len(idx) < size and n > size:
        idx = np.concatenate([idx, order[: size - len(idx)]])
    return idx


def _binary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}


def train_and_evaluate(
    tx: pd.DataFrame, fold: Fold, label_cfg: LabelConfig, cfg: SeqConfig,
    device: str | torch.device = "auto", log=print,
) -> dict:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    use_amp = device.type == "cuda"
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    t0 = time.time()

    test_day = fold.test_day
    tx_known = tx[tx["day"] <= test_day]
    vocab = Vocab.build(tx_known, cfg.min_item_count)
    store = BasketStore.build(tx_known, vocab, cutoff=test_day, max_items=cfg.max_items)
    log(f"  vocab: {vocab.size:,} ürün (min {cfg.min_item_count} sepet), "
        f"{len(store.days):,} sepet, {store.n_customers:,} müşteri ({time.time() - t0:.1f}s)")

    # --- müşteri ayrımı ---------------------------------------------------
    all_c = np.arange(store.n_customers)
    rng.shuffle(all_c)
    n_val = max(1, int(round(cfg.val_frac * len(all_c))))
    val_c, train_c = np.sort(all_c[:n_val]), np.sort(all_c[n_val:])
    val_ids = set(store.customers[val_c])

    lapse_train = fold.train[~fold.train["customer_id"].isin(val_ids)].reset_index(drop=True)
    lapse_val = fold.train[fold.train["customer_id"].isin(val_ids)].reset_index(drop=True)
    test_rows = fold.test[["customer_id", "snapshot_day", "churn_w", "recency"]].reset_index(drop=True)

    if cfg.lapse_summary_features:
        scaler = FeatureScaler().fit(lapse_train)
        s_train, s_val, s_test = (scaler.transform(d) for d in (lapse_train, lapse_val, fold.test))
    else:
        s_train = s_val = s_test = None

    t_val = max(fold.train_days)
    nb_val_rows = pd.DataFrame({"customer_id": store.customers[val_c], "snapshot_day": t_val})
    nb_val_rows = nb_val_rows[[store.end_before(store.cust_index[c], t_val) > store.span(store.cust_index[c])[0]
                               for c in nb_val_rows["customer_id"]]]
    nb_val_targets = next_basket_targets(tx_known, pd.Index(nb_val_rows["customer_id"]),
                                         t_val, label_cfg.prediction_window_days)

    # --- model ------------------------------------------------------------
    mcfg = ModelConfig(
        n_items=vocab.size, encoder=cfg.encoder, d_model=cfg.d_model, n_layers=cfg.n_layers,
        n_heads=cfg.n_heads, dropout=cfg.dropout, max_len=cfg.max_len,
        lapse_inactivity_days=label_cfg.churn_inactivity_days,
        n_summary_features=len(FEATURE_COLUMNS) if cfg.lapse_summary_features else 0,
        repeat_aware=cfg.repeat_aware,
    )
    model = DeepPulseSeq(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)
    bce = torch.nn.BCEWithLogitsLoss()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  model: {cfg.encoder}, {n_params:,} parametre, cihaz {device}"
        f"{', AMP fp16' if use_amp else ''}")

    log_q = None
    if cfg.logq_correction:
        # Hedef frekansı yalnızca eğitim müşterilerinin sepetlerinden (doğrulama sızmasın)
        rows = np.concatenate([np.arange(*store.span(int(c))) for c in train_c])
        it = store.items[rows].ravel()
        cnt = np.bincount(it[it > 1], minlength=vocab.size).astype(np.float64) + 1.0
        cnt[:2] = 1.0
        log_q = torch.tensor(np.log(cnt / cnt.sum()), dtype=torch.float32, device=device)

    best_nb = {"score": -np.inf, "epoch": 0, "state": None}
    best_lp = {"score": -np.inf, "epoch": 0, "state": None}
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ends = sample_next_basket_ends(store, train_c, cfg.windows_per_customer, rng)
        order = rng.permutation(len(ends))
        lapse_order = rng.permutation(len(lapse_train))
        # Bir epoch = iki görevden uzun olanın verisini bir kez dolaşmak; kısa olan döngüyle tekrarlanır
        n_steps = int(np.ceil(max(len(ends), len(lapse_train)) / cfg.batch_size))
        nb_sum = lp_sum = 0.0
        for step in range(n_steps):
            nb_idx = _cycle(order, step, cfg.batch_size)
            lp_idx = _cycle(lapse_order, step, cfg.batch_size)
            nb = _to_device(next_basket_batch(store, [ends[i] for i in nb_idx], cfg.max_len), device)
            lp = _to_device(snapshot_batch(store, lapse_train.iloc[lp_idx], cfg.max_len,
                                           None if s_train is None else s_train[lp_idx]), device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                h = model.encode(nb["items"], nb["feats"], nb["weekday"], nb["lengths"])
                loss_nb = next_basket_loss(model, h, nb["items"], nb["target_items"], nb["lengths"],
                                           cfg.n_negatives, max_candidates=cfg.max_candidates, log_q=log_q)
                hl = model.encode(lp["items"], lp["feats"], lp["weekday"], lp["lengths"])
                logit = model.lapse_logit(model.last_state(hl, lp["lengths"]), lp["recency"], lp.get("summary"))
                loss_lp = bce(logit.float(), lp["label"])
                loss = loss_nb + cfg.lapse_weight * loss_lp

            opt.zero_grad(set_to_none=True)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_amp.step(opt)
            scaler_amp.update()
            nb_sum += loss_nb.item()
            lp_sum += loss_lp.item()

        # --- doğrulama -----------------------------------------------------
        p_val, _ = predict_snapshot(model, store, lapse_val, cfg, device, summary=s_val)
        lapse_m = _binary(lapse_val["churn_w"].astype(int).to_numpy(), p_val)
        _, nb_preds = predict_snapshot(model, store, nb_val_rows, cfg, device, vocab)
        nb_m = ranking_report(nb_preds, nb_val_targets, cfg.k)
        rec = {
            "epoch": epoch, "loss_next_basket": nb_sum / n_steps, "loss_lapse": lp_sum / n_steps,
            "val_lapse_roc_auc": lapse_m["roc_auc"], "val_lapse_pr_auc": lapse_m["pr_auc"],
            f"val_recall@{cfg.k}": nb_m[f"recall@{cfg.k}"], f"val_ndcg@{cfg.k}": nb_m[f"ndcg@{cfg.k}"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        marks = ""
        if nb_m[f"ndcg@{cfg.k}"] > best_nb["score"]:
            best_nb = {"score": nb_m[f"ndcg@{cfg.k}"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            marks += " S"
        lp_score = lapse_m["pr_auc"] if np.isfinite(lapse_m["pr_auc"]) else -np.inf
        if lp_score > best_lp["score"]:
            best_lp = {"score": lp_score, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            marks += " K"
        log(f"  epoch {epoch:2d} | kayıp sepet {rec['loss_next_basket']:.3f} kopma {rec['loss_lapse']:.3f} | "
            f"val kopma AUC {lapse_m['roc_auc']:.3f} PR {lapse_m['pr_auc']:.3f} | "
            f"val NDCG@{cfg.k} {nb_m[f'ndcg@{cfg.k}']:.3f} R@{cfg.k} {nb_m[f'recall@{cfg.k}']:.3f}"
            f"{'  *' + marks if marks else ''} ({rec['seconds']}s)")
        if epoch - max(best_nb["epoch"], best_lp["epoch"]) >= cfg.patience:
            log(f"  erken durdurma: {cfg.patience} epoch iyileşme yok "
                f"(en iyi: sonraki sepet epoch {best_nb['epoch']}, kopma epoch {best_lp['epoch']})")
            break
    log("  (* S = sonraki sepet başlığının, K = kopma başlığının yeni en iyi checkpoint'i)")

    # --- test: her başlık kendi en iyi checkpoint'iyle ---------------------
    model.load_state_dict(best_lp["state"])
    p_test, _ = predict_snapshot(model, store, test_rows, cfg, device, summary=s_test)
    model.load_state_dict(best_nb["state"])
    _, nb_test = predict_snapshot(model, store, test_rows, cfg, device, vocab)
    nb_targets = next_basket_targets(tx, pd.Index(test_rows["customer_id"]), test_day,
                                     label_cfg.prediction_window_days)
    best_val = {
        "next_basket_epoch": best_nb["epoch"], f"next_basket_val_ndcg@{cfg.k}": best_nb["score"],
        "lapse_epoch": best_lp["epoch"], "lapse_val_pr_auc": best_lp["score"],
    }
    return {
        "config": asdict(cfg),
        "vocab_size": vocab.size,
        "n_params": n_params,
        "best_epoch": {"next_basket": best_nb["epoch"], "lapse": best_lp["epoch"]},
        "best_val": best_val,
        "history": history,
        "test_lapse_prob": p_test,
        "test_rows": test_rows,
        "test_next_basket_preds": nb_test,
        "test_next_basket_targets": nb_targets,
        "model": model,
        "states": {"next_basket": best_nb["state"], "lapse": best_lp["state"]},
        "vocab": vocab,
    }
