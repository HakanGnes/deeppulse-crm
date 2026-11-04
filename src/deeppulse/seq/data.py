"""Sepet dizisi veri hattı.

Her müşteri, zamana göre sıralı bir sepet dizisidir. Bir sepet ("kelime"):
    - en fazla M ürün (vocab indeksi; brüt değere göre en büyükler),
    - sürekli özellikler: önceki sepetten bu yana gün (log), net tutar (log),
      indirim payı, ürün sayısı (log), kupon kullanıldı mı,
    - haftanın günü vekili (day % 7; Dunnhumby'de gerçek tarih yok).

Sızıntı güvencesi: BasketStore yalnızca `cutoff` (= test günü) ve öncesindeki
işlemlerle kurulur. Modelin gördüğü her girdi ve her eğitim hedefi bu tablodan
gelir; test gününden sonraki hiçbir sepet modele ulaşamaz. Test hedefleri
(sonraki sepet, kopma etiketi) ayrıca, tam veriden hesaplanır.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PAD, UNK = 0, 1
N_BASKET_FEATS = 5


@dataclass
class Vocab:
    product_ids: np.ndarray            # indeks → ürün (0: PAD, 1: UNK → "")
    index: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.product_ids)

    @classmethod
    def build(cls, tx: pd.DataFrame, min_count: int) -> "Vocab":
        counts = tx.drop_duplicates(["basket_id", "product_id"])["product_id"].value_counts()
        keep = counts[counts >= min_count].index.astype(str).tolist()
        ids = np.array(["<pad>", "<unk>", *keep], dtype=object)
        return cls(product_ids=ids, index={p: i + 2 for i, p in enumerate(keep)})

    def encode(self, products: pd.Series) -> np.ndarray:
        return products.map(self.index).fillna(UNK).astype("int64").to_numpy()


@dataclass
class BasketStore:
    customers: np.ndarray        # [n_cust] müşteri kimlikleri
    offsets: np.ndarray          # [n_cust + 1] her müşterinin sepet aralığı
    days: np.ndarray             # [n_baskets]
    items: np.ndarray            # [n_baskets, M] vocab indeksleri (0 = boş)
    feats: np.ndarray            # [n_baskets, F]
    weekday: np.ndarray          # [n_baskets]
    cutoff: int
    cust_index: dict[str, int]

    @property
    def n_customers(self) -> int:
        return len(self.customers)

    def span(self, c: int) -> tuple[int, int]:
        return int(self.offsets[c]), int(self.offsets[c + 1])

    def end_before(self, c: int, day: int) -> int:
        """`day` dahil, o güne kadarki son sepetten bir sonraki mutlak indeks."""
        s, e = self.span(c)
        return s + int(np.searchsorted(self.days[s:e], day, side="right"))

    @classmethod
    def build(cls, tx: pd.DataFrame, vocab: Vocab, cutoff: int, max_items: int) -> "BasketStore":
        tx = tx[tx["day"] <= cutoff]
        lines = (
            tx.groupby(["basket_id", "product_id"], sort=False)
            .agg(customer_id=("customer_id", "first"), day=("day", "first"),
                 trans_time=("trans_time", "first"), gross=("gross_value", "sum"))
            .reset_index()
        )
        baskets = (
            tx.groupby("basket_id", sort=False)
            .agg(customer_id=("customer_id", "first"), day=("day", "first"),
                 trans_time=("trans_time", "first"), net=("net_value", "sum"),
                 gross=("gross_value", "sum"), disc=("retail_disc", "sum"),
                 match=("coupon_match_disc", "sum"), coupon=("coupon_disc", "max"))
            .reset_index()
            .sort_values(["customer_id", "day", "trans_time", "basket_id"], kind="stable")
            .reset_index(drop=True)
        )
        n_items = lines.groupby("basket_id").size()
        baskets["n_items"] = baskets["basket_id"].map(n_items).to_numpy()
        pos = pd.Series(np.arange(len(baskets)), index=baskets["basket_id"])

        # Ürünler: sepet içinde brüt değere göre sırala, ilk M'yi tut
        lines = lines.sort_values(["basket_id", "gross"], ascending=[True, False], kind="stable")
        lines["rank"] = lines.groupby("basket_id").cumcount()
        lines = lines[lines["rank"] < max_items]
        items = np.zeros((len(baskets), max_items), dtype=np.int64)
        items[pos.loc[lines["basket_id"]].to_numpy(), lines["rank"].to_numpy()] = vocab.encode(lines["product_id"])

        cust = baskets["customer_id"].to_numpy()
        change = np.r_[True, cust[1:] != cust[:-1]]
        starts = np.flatnonzero(change)
        offsets = np.r_[starts, len(baskets)].astype(np.int64)
        customers = cust[starts]

        days = baskets["day"].to_numpy().astype(np.int64)
        gap = np.diff(days, prepend=days[0]).astype(float)
        gap[starts] = 0.0  # her müşterinin ilk sepeti
        gross = baskets["gross"].to_numpy()
        feats = np.stack([
            np.log1p(gap),
            np.log1p(baskets["net"].to_numpy().clip(min=0)),
            np.where(gross > 0, (baskets["disc"] + baskets["match"]).to_numpy() / np.maximum(gross, 1e-9), 0.0),
            np.log1p(baskets["n_items"].to_numpy()),
            (baskets["coupon"].to_numpy() > 0).astype(float),
        ], axis=1).astype(np.float32)

        return cls(
            customers=customers, offsets=offsets, days=days, items=items, feats=feats,
            weekday=(days % 7).astype(np.int64), cutoff=int(cutoff),
            cust_index={c: i for i, c in enumerate(customers)},
        )


@dataclass
class Batch:
    items: np.ndarray       # [B, L, M]
    feats: np.ndarray       # [B, L, F]
    weekday: np.ndarray     # [B, L]
    lengths: np.ndarray     # [B]
    target_items: np.ndarray | None = None   # [B, L, M]  sonraki sepetin ürünleri
    recency: np.ndarray | None = None        # [B]
    label: np.ndarray | None = None          # [B]
    summary: np.ndarray | None = None        # [B, n_feat] Faz 0 özet özellikleri (ölçeklenmiş)


def _gather(store: BasketStore, ranges: list[tuple[int, int]], max_len: int) -> Batch:
    b, m, f = len(ranges), store.items.shape[1], store.feats.shape[1]
    items = np.zeros((b, max_len, m), dtype=np.int64)
    feats = np.zeros((b, max_len, f), dtype=np.float32)
    wday = np.zeros((b, max_len), dtype=np.int64)
    lengths = np.zeros(b, dtype=np.int64)
    for i, (s, e) in enumerate(ranges):
        n = e - s
        items[i, :n] = store.items[s:e]
        feats[i, :n] = store.feats[s:e]
        wday[i, :n] = store.weekday[s:e]
        lengths[i] = n
    return Batch(items=items, feats=feats, weekday=wday, lengths=lengths)


def next_basket_batch(store: BasketStore, ends: list[tuple[int, int]], max_len: int) -> Batch:
    """Her (müşteri, bitiş) için pencere: girdi = sepetler[s : e-1], hedef = sepetler[s+1 : e]."""
    inp, tgt = [], []
    for c, e in ends:
        s0, _ = store.span(c)
        s = max(s0, e - 1 - max_len)
        inp.append((s, e - 1))
        tgt.append((s + 1, e))
    batch = _gather(store, inp, max_len)
    batch.target_items = _gather(store, tgt, max_len).items
    return batch


def snapshot_batch(store: BasketStore, rows: pd.DataFrame, max_len: int,
                   summary: np.ndarray | None = None) -> Batch:
    """rows: customer_id, snapshot_day (+ isteğe bağlı churn_w). Girdi = T'ye kadarki son L sepet.

    summary: rows ile aynı sırada, ölçeklenmiş Faz 0 özellikleri (isteğe bağlı).
    """
    ranges, rec = [], []
    for cust, t in zip(rows["customer_id"].to_numpy(), rows["snapshot_day"].to_numpy()):
        if t > store.cutoff:
            raise ValueError(f"snapshot {t} > store cutoff {store.cutoff}: gelecek verisi istenmiş")
        c = store.cust_index[cust]
        e = store.end_before(c, int(t))
        s0, _ = store.span(c)
        if e <= s0:
            raise ValueError(f"{cust} için {t} gününden önce sepet yok")
        ranges.append((max(s0, e - max_len), e))
        rec.append(int(t) - int(store.days[e - 1]))
    batch = _gather(store, ranges, max_len)
    batch.recency = np.asarray(rec, dtype=np.float32)
    if "churn_w" in rows:
        batch.label = rows["churn_w"].astype("float32").to_numpy()
    if summary is not None:
        batch.summary = np.asarray(summary, dtype=np.float32)
    return batch


def sample_next_basket_ends(
    store: BasketStore, customers: np.ndarray, per_customer: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Rastgele pencere bitişleri (en az 2 sepet: 1 girdi + 1 hedef)."""
    out = []
    for c in customers:
        s, e = store.span(int(c))
        if e - s < 2:
            continue
        for end in rng.integers(s + 2, e + 1, size=per_customer):
            out.append((int(c), int(end)))
    return out
