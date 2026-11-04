"""Proje genelindeki ayarlar.

Faz 0'ın kritik kararları (churn eşiği, tahmin penceresi, snapshot aralığı)
burada tek yerde durur. Gerekçeleri için: docs/decisions.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class LabelConfig:
    # N: bu kadar gün hiç alışveriş yoksa müşteri "koptu" (lapse) sayılır.
    # Kodda tarihsel olarak "churn" adı kullanılıyor; Dunnhumby'de bu olayların
    # ~%89'unda müşteri 180 gün içinde geri döndüğü için anlamı "kalıcı terk"
    # değil "kopma"dır. N=60 gerçek veride seçildi (docs/decisions.md → K1).
    churn_inactivity_days: int = 60
    # Sınıflandırma etiketi: snapshot'tan sonraki W gün içinde churn olayı gerçekleşti mi?
    prediction_window_days: int = 90
    # Rolling snapshot'lar arası gün sayısı (Dunnhumby'de hafta = 7 gün)
    snapshot_step_days: int = 28
    # Bir snapshot'ta müşterinin özellik üretilebilmesi için gereken asgari geçmiş
    min_history_days: int = 56
    # Snapshot'ta en az bu kadar sepeti olmayan müşteri dahil edilmez
    min_baskets: int = 2


@dataclass(frozen=True)
class SplitConfig:
    # Test snapshot'ı: bu günden itibaren alınan snapshot'lar test kümesidir.
    # None ise veri ufkuna göre otomatik seçilir (son geçerli snapshot).
    test_snapshot_day: int | None = None
    # Doğrulama kümesi için test'ten bir önceki geçerli snapshot kullanılır
    use_validation: bool = True


@dataclass(frozen=True)
class SapConfig:
    # Dunnhumby'de DAY 1..711 göreli gündür; SAP belgeleri gerçek tarih ister.
    # Seçilen tarih tamamen temsilidir (veri setinde gerçek tarih yoktur).
    base_date: date = date(2023, 1, 1)
    company_code: str = "1000"   # BUKRS
    sales_org: str = "1000"      # VKORG
    currency: str = "USD"        # WAERS
    revenue_gl_account: str = "0000800000"   # HKONT — satış gelirleri
    discount_gl_account: str = "0000880000"  # HKONT — satış indirimleri


@dataclass(frozen=True)
class Config:
    labels: LabelConfig = field(default_factory=LabelConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    sap: SapConfig = field(default_factory=SapConfig)
    random_state: int = 42


DEFAULT_CONFIG = Config()
