"""Sepet dizisi modeli: sepet kodlayıcı → dizi kodlayıcı (GRU ya da SASRec) → iki başlık.

    sepet token'ı = ortalama(ürün embedding'leri) + MLP(sürekli özellikler) + haftanın günü
    dizi kodlayıcı:
        "gru"    — GRU4Rec tarzı tekrarlayan ağ
        "sasrec" — SASRec tarzı nedensel (causal) self-attention Transformer
    başlıklar:
        sonraki sepet — h_t · ürün_embedding (ağırlık paylaşımlı)  +  tekrar bonusu
                        tekrar bonusu = g(h_t) · log(1 + ürünü içeren geçmiş sepet sayısı) + b·[hiç alındı]
                        g(h_t) ≥ 0, müşterinin o anki durumuna göre "tekrar mı keşif mi" ağırlığı.
                        Market verisinde tekrar alımlar baskın; bu terim olmadan ağ, kişisel
                        tekrar baseline'ının altında popülerliğe takılıyor (Faz 1 ilk deneme).
                        Sampled softmax ile eğitilir.
        kopma         — MLP([h_son, recency özellikleri, (isteğe bağlı) Faz 0 özet özellikleri])
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from deeppulse.seq.data import N_BASKET_FEATS, PAD, UNK


@dataclass(frozen=True)
class ModelConfig:
    n_items: int
    encoder: str = "sasrec"      # "gru" | "sasrec"
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 2
    dropout: float = 0.2
    max_len: int = 50
    lapse_inactivity_days: int = 60
    n_summary_features: int = 0   # >0 ise kopma başlığı Faz 0 özelliklerini de görür
    repeat_aware: bool = True


class BasketEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.item_emb = nn.Embedding(cfg.n_items, d, padding_idx=PAD)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        with torch.no_grad():
            self.item_emb.weight[PAD].zero_()
        self.feat_mlp = nn.Sequential(nn.Linear(N_BASKET_FEATS, d), nn.GELU(), nn.Linear(d, d))
        self.weekday = nn.Embedding(7, d)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, items: torch.Tensor, feats: torch.Tensor, weekday: torch.Tensor) -> torch.Tensor:
        e = self.item_emb(items)                         # B, L, M, d
        m = (items != PAD).unsqueeze(-1).to(e.dtype)
        pooled = (e * m).sum(-2) / m.sum(-2).clamp(min=1.0)
        return self.drop(self.norm(pooled + self.feat_mlp(feats) + self.weekday(weekday)))


class GRUSeqEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gru = nn.GRU(cfg.d_model, cfg.d_model, num_layers=cfg.n_layers, batch_first=True,
                          dropout=cfg.dropout if cfg.n_layers > 1 else 0.0)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, None]:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return out, None


class SASRecEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.pos = nn.Embedding(cfg.max_len, d)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, dim_feedforward=4 * d, dropout=cfg.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        L = x.size(1)
        pos = torch.arange(L, device=x.device)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
        # Sağdan dolgu: geçerli bir konum yalnızca kendisi ve öncesini görür, dolgu ona hiç
        # ulaşamaz. Dolgu konumlarının çıktıları kullanılmaz.
        h = self.encoder(x + self.pos(pos), mask=causal)
        return self.norm(h), causal


class DeepPulseSeq(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.basket = BasketEncoder(cfg)
        if cfg.encoder == "gru":
            self.seq = GRUSeqEncoder(cfg)
        elif cfg.encoder == "sasrec":
            self.seq = SASRecEncoder(cfg)
        else:
            raise ValueError(f"bilinmeyen encoder: {cfg.encoder}")
        d = cfg.d_model
        self.nb_proj = nn.Linear(d, d)
        self.repeat_gate = nn.Linear(d, 1)
        nn.init.zeros_(self.repeat_gate.weight)
        nn.init.constant_(self.repeat_gate.bias, 0.55)   # softplus(0.55) ≈ 1
        self.repeat_seen = nn.Parameter(torch.zeros(()))
        self.lapse_head = nn.Sequential(
            nn.Linear(d + 2 + cfg.n_summary_features, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, 1)
        )

    # --- gövde ------------------------------------------------------------
    def encode(self, items, feats, weekday, lengths) -> torch.Tensor:
        x = self.basket(items, feats, weekday)
        h, _ = self.seq(x, lengths)
        return h                                          # B, L, d

    @staticmethod
    def last_state(h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        return h.gather(1, idx).squeeze(1)

    # --- başlıklar --------------------------------------------------------
    def item_query(self, h: torch.Tensor) -> torch.Tensor:
        return self.nb_proj(h)

    def repeat_bonus(self, h: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """h: [..., d], counts: [..., C] → [..., C]"""
        if not self.cfg.repeat_aware:
            return torch.zeros_like(counts)
        gate = F.softplus(self.repeat_gate(h).float())
        return gate * torch.log1p(counts) + self.repeat_seen * (counts > 0).float()

    def lapse_logit(self, h_last: torch.Tensor, recency: torch.Tensor,
                    summary: torch.Tensor | None = None) -> torch.Tensor:
        r = torch.stack([torch.log1p(recency), recency / self.cfg.lapse_inactivity_days], dim=-1)
        parts = [h_last, r.to(h_last.dtype)]
        if self.cfg.n_summary_features:
            parts.append(summary.to(h_last.dtype))
        return self.lapse_head(torch.cat(parts, dim=-1)).squeeze(-1)

    def score_all_items(self, h_last: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Tüm vocab üzerinde skor; items: dizinin tüm sepetleri [B, L, M] (hepsi tahmin anından önce)."""
        s = (self.item_query(h_last) @ self.basket.item_emb.weight.T).float()
        s = s + self.repeat_bonus(h_last, history_counts_total(items, self.cfg.n_items))
        s[:, PAD] = float("-inf")
        s[:, UNK] = float("-inf")
        return s


def history_counts_total(items: torch.Tensor, n_items: int) -> torch.Tensor:
    """[B, L, M] → [B, V]: her ürünü içeren sepet sayısı (sepet içinde ürünler tekil)."""
    b = items.size(0)
    flat = items.reshape(b, -1)
    counts = torch.zeros(b, n_items, device=items.device, dtype=torch.float32)
    counts.scatter_add_(1, flat, (flat > UNK).float())
    return counts


def history_counts_cumulative(items: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
    """[B, L, M] → [B, L, C]: t konumuna kadar (dahil) aday ürünü içeren sepet sayısı."""
    b, l, _ = items.shape
    idx = torch.searchsorted(cand, items).clamp(max=cand.numel() - 1)
    hit = ((items > UNK) & (cand[idx] == items)).float()
    counts = torch.zeros(b, l, cand.numel(), device=items.device, dtype=torch.float32)
    counts.scatter_add_(2, idx, hit)
    return counts.cumsum(dim=1)


def next_basket_loss(
    model: DeepPulseSeq, h: torch.Tensor, items: torch.Tensor, target_items: torch.Tensor,
    lengths: torch.Tensor, n_negatives: int, generator: torch.Generator | None = None,
    max_candidates: int = 8192, log_q: torch.Tensor | None = None,
) -> torch.Tensor:
    """Çok etiketli sampled softmax.

    Aday küme = partideki tüm hedef ürünler (in-batch negatifler) + rastgele n_negatives ürün.
    Her konumun hedef dağılımı, o konumun sonraki sepetindeki ürünler üzerinde düzgün dağılım.
    Tekrar bonusu, t konumuna kadarki (dahil) girdi sepetlerinden hesaplanır.

    Bellek: en büyük tensörler P×C boyutunda (P = geçerli konum sayısı, C = aday sayısı).
    Partideki tekil hedef ürün sayısı `max_candidates − n_negatives`'i aşarsa hedef ürünlerin
    rastgele bir alt kümesi aday yapılır; kalan hedefler o adımda kayba girmez
    (8 GB GPU'da partiyi sığdırmak için yaklaştırma).

    log_q (isteğe bağlı, [V]): ürünlerin hedeflerdeki log frekansı. Verilirse logQ düzeltmesi
    uygulanır: in-batch negatifler popüler ürünleri orantısız sık "negatif" yaptığı için
    model popüler ürünleri fazla bastırmayı öğrenir; her aday logit'inden log(örneklenme
    olasılığı) çıkarılarak bu yanlılık giderilir (yalnızca eğitimde).
    """
    L = h.size(1)
    valid = torch.arange(L, device=h.device).unsqueeze(0) < lengths.unsqueeze(1)
    tp_all = target_items[valid]
    known_all = tp_all > UNK
    if not known_all.any():
        return h.sum() * 0.0
    n_items = model.cfg.n_items
    neg = torch.randint(UNK + 1, n_items, (n_negatives,), device=h.device, generator=generator)
    pos = torch.unique(tp_all[known_all])
    room = max(max_candidates - n_negatives, 1)
    if pos.numel() > room:
        keep = torch.randperm(pos.numel(), device=h.device, generator=generator)[:room]
        pos = pos[keep]
    cand = torch.unique(torch.cat([pos, neg]))                    # sıralı

    idx_all = torch.searchsorted(cand, tp_all.clamp(min=0)).clamp(max=cand.numel() - 1)
    hit_all = known_all & (cand[idx_all] == tp_all)
    rows = hit_all.any(dim=1)                                     # en az bir hedefi aday kümede olanlar
    hv, idx, hit = h[valid][rows], idx_all[rows], hit_all[rows]
    w = model.basket.item_emb.weight[cand]                        # C, d
    logits = (model.item_query(hv) @ w.T).float()                 # P, C
    if model.cfg.repeat_aware:
        counts = history_counts_cumulative(items, cand)[valid][rows]
        logits = logits + model.repeat_bonus(hv, counts)
    if log_q is not None:
        n_entries = float(known_all.sum())
        q = (n_entries * log_q[cand].exp()).clamp(max=1.0) + n_negatives / n_items
        logits = logits - torch.log(q.clamp(max=1.0))
    target = torch.zeros_like(logits)
    target.scatter_add_(1, idx, hit.float())
    target = target.clamp(max=1.0)
    target = target / target.sum(dim=1, keepdim=True)
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
