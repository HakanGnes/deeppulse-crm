import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from deeppulse.config import LabelConfig  # noqa: E402
from deeppulse.seq.data import (  # noqa: E402
    PAD, UNK, BasketStore, Vocab, next_basket_batch, snapshot_batch,
)
from deeppulse.seq.model import (  # noqa: E402
    DeepPulseSeq, ModelConfig, history_counts_cumulative, history_counts_total, next_basket_loss,
)
from deeppulse.seq.train import SeqConfig, train_and_evaluate  # noqa: E402
from deeppulse.splits import temporal_fold  # noqa: E402

from conftest import make_tx  # noqa: E402


@pytest.fixture(scope="module")
def small_store():
    tx = make_tx({"A": [1, 5, 9, 20, 40], "B": [2, 3, 50]})
    tx.loc[tx["day"] == 9, "product_id"] = "x"
    vocab = Vocab.build(tx, min_count=1)
    return tx, vocab, BasketStore.build(tx, vocab, cutoff=30, max_items=4)


def test_store_never_contains_baskets_after_cutoff(small_store):
    _, _, store = small_store
    assert store.days.max() <= 30
    assert store.span(store.cust_index["A"]) == (0, 4)       # 40. gün dışarıda
    assert store.span(store.cust_index["B"])[1] - store.span(store.cust_index["B"])[0] == 2


def test_snapshot_batch_uses_only_past_and_computes_recency(small_store):
    _, _, store = small_store
    rows = pd.DataFrame({"customer_id": ["A"], "snapshot_day": [12], "churn_w": [1]})
    b = snapshot_batch(store, rows, max_len=10)
    assert b.lengths[0] == 3            # 1, 5, 9
    assert b.recency[0] == 3            # 12 − 9
    assert b.label[0] == 1.0


def test_snapshot_beyond_cutoff_is_rejected(small_store):
    _, _, store = small_store
    with pytest.raises(ValueError, match="cutoff"):
        snapshot_batch(store, pd.DataFrame({"customer_id": ["A"], "snapshot_day": [35]}), 10)


def test_next_basket_targets_are_inputs_shifted_by_one(small_store):
    _, _, store = small_store
    c = store.cust_index["A"]
    b = next_basket_batch(store, [(c, store.span(c)[1])], max_len=10)
    n = b.lengths[0]
    assert n == 3
    np.testing.assert_array_equal(b.target_items[0, : n - 1], b.items[0, 1:n])
    np.testing.assert_array_equal(b.target_items[0, n - 1], store.items[store.span(c)[1] - 1])


def test_unknown_products_map_to_unk():
    tx = make_tx({"A": [1, 2, 3]})
    tx.loc[tx["day"] == 3, "product_id"] = "rare"
    vocab = Vocab.build(tx, min_count=2)
    assert vocab.encode(pd.Series(["1", "rare"])).tolist() == [2, UNK]


def test_history_counts():
    items = torch.tensor([[[5, 6, PAD], [5, PAD, PAD], [7, UNK, PAD]]])   # B=1, L=3, M=3
    cand = torch.tensor([5, 6, 7])
    cum = history_counts_cumulative(items, cand)[0]
    assert cum.tolist() == [[1, 1, 0], [2, 1, 0], [2, 1, 1]]
    tot = history_counts_total(items, n_items=8)[0]
    assert tot[5] == 2 and tot[6] == 1 and tot[7] == 1 and tot[UNK] == 0 and tot[PAD] == 0


@pytest.mark.parametrize("encoder", ["gru", "sasrec"])
def test_loss_is_finite_and_respects_candidate_cap(encoder):
    torch.manual_seed(0)
    v, b, l, m = 500, 8, 6, 5
    model = DeepPulseSeq(ModelConfig(n_items=v, encoder=encoder, d_model=16, max_len=l))
    items = torch.randint(2, v, (b, l, m))
    target = torch.randint(2, v, (b, l, m))
    lengths = torch.tensor([6, 5, 4, 3, 2, 1, 6, 6])
    h = model.encode(items, torch.randn(b, l, 5), torch.randint(0, 7, (b, l)), lengths)
    loss = next_basket_loss(model, h, items, target, lengths, n_negatives=16, max_candidates=64)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.basket.item_emb.weight.grad is not None


def test_causal_encoder_ignores_future_positions():
    """SASRec: t konumunun çıktısı t'den sonraki sepetlere bağlı olmamalı."""
    torch.manual_seed(0)
    model = DeepPulseSeq(ModelConfig(n_items=50, encoder="sasrec", d_model=16, max_len=6, dropout=0.0)).eval()
    items = torch.randint(2, 50, (1, 6, 3))
    feats, wd = torch.randn(1, 6, 5), torch.randint(0, 7, (1, 6))
    lengths = torch.tensor([6])
    h1 = model.encode(items, feats, wd, lengths)
    items2 = items.clone()
    items2[0, 4:] = torch.randint(2, 50, (2, 3))
    h2 = model.encode(items2, feats, wd, lengths)
    torch.testing.assert_close(h1[0, :4], h2[0, :4])


@pytest.mark.parametrize("encoder", ["gru", "sasrec"])
def test_train_and_evaluate_smoke(synthetic_canonical, encoder):
    tx, _ = synthetic_canonical
    label_cfg = LabelConfig()
    fold = temporal_fold(tx, label_cfg)
    cfg = SeqConfig(encoder=encoder, d_model=16, epochs=1, batch_size=64, n_negatives=64,
                    windows_per_customer=1, max_len=20)
    res = train_and_evaluate(tx, fold, label_cfg, cfg, device="cpu", log=lambda *_: None)
    assert len(res["test_lapse_prob"]) == len(fold.test)
    assert np.all((res["test_lapse_prob"] >= 0) & (res["test_lapse_prob"] <= 1))
    some = next(iter(res["test_next_basket_preds"].values()))
    assert len(some) == cfg.k and "<pad>" not in some and "<unk>" not in some


def test_logq_correction_changes_loss_but_stays_finite():
    torch.manual_seed(0)
    v, b, l, m = 300, 4, 5, 4
    model = DeepPulseSeq(ModelConfig(n_items=v, encoder="gru", d_model=16, max_len=l))
    items, target = torch.randint(2, v, (b, l, m)), torch.randint(2, v, (b, l, m))
    lengths = torch.tensor([5, 4, 3, 2])
    h = model.encode(items, torch.randn(b, l, 5), torch.randint(0, 7, (b, l)), lengths)
    log_q = torch.log(torch.softmax(torch.randn(v), 0))
    torch.manual_seed(1)
    a = next_basket_loss(model, h, items, target, lengths, 16)
    torch.manual_seed(1)
    c = next_basket_loss(model, h, items, target, lengths, 16, log_q=log_q)
    assert torch.isfinite(c) and not torch.isclose(a, c)
