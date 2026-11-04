"""Dunnhumby şemasını birebir taklit eden sentetik veri üreteci.

Testler ve gerçek veri olmadan uçtan uca deneme için. İçine bilerek konan sinyaller:
- Her hane heterojen bir alışveriş sıklığına sahip (Gamma dağılımı).
- Bazı haneler bir "dropout" gününden sonra tamamen bırakıyor; bırakmadan önceki
  ~60 günde sıklıkları azalıyor (churn modellerinin yakalaması gereken sinyal).
- Haneler kişisel favori ürünlerini tekrar tekrar alıyor (next-basket'te
  kişisel tekrar baseline'ının güçlü çıkması beklenir — gerçek market verisi gibi).
- Ürünlerin bir kısmı rastgele haftalarda indirimde.

Sinyalsiz kontrol verisi için `signal=False`: dropout ve azalma kapatılır,
herkes aynı süreçten gelir → churn AUC'si ~0.5 çıkmalı (sızıntı testi).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEPARTMENTS = {
    "GROCERY": ["SOFT DRINKS", "CEREAL", "BAKED BREAD", "COFFEE", "PASTA"],
    "PRODUCE": ["VEGETABLES", "FRUIT", "SALAD MIX"],
    "MEAT": ["BEEF", "CHICKEN", "PORK"],
    "DRUG GM": ["DIAPERS", "VITAMINS", "COLD REMEDIES"],
    "PASTRY": ["DONUTS", "CAKES"],
}


def make_products(n_products: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    commodities = [(d, c) for d, cs in DEPARTMENTS.items() for c in cs]
    for i in range(n_products):
        dept, comm = commodities[rng.integers(len(commodities))]
        rows.append({
            "PRODUCT_ID": 100_000 + i,
            "MANUFACTURER": int(rng.integers(1, 60)),
            "DEPARTMENT": dept,
            "BRAND": "Private" if rng.random() < 0.3 else "National",
            "COMMODITY_DESC": comm,
            "SUB_COMMODITY_DESC": f"{comm} SUB{int(rng.integers(1, 4))}",
            "CURR_SIZE_OF_PRODUCT": f"{int(rng.integers(1, 64))} OZ",
        })
    return pd.DataFrame(rows)


def make_dataset(
    n_households: int = 300,
    n_products: int = 400,
    horizon_days: int = 711,
    signal: bool = True,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(transaction_data, product) döndürür — orijinal CSV sütun adlarıyla."""
    rng = np.random.default_rng(seed)
    products = make_products(n_products, rng)
    product_ids = products["PRODUCT_ID"].to_numpy()
    popularity = rng.zipf(1.6, n_products).astype(float)
    popularity /= popularity.sum()
    shelf_price = np.round(rng.lognormal(1.0, 0.6, n_products), 2)
    # Her ürün için indirimli haftalar (yaklaşık %15)
    n_weeks = horizon_days // 7 + 2
    on_promo = rng.random((n_products, n_weeks)) < 0.15
    promo_depth = rng.uniform(0.1, 0.35, n_products)

    rows = []
    basket_id = 26_000_000_000
    for hh in range(1, n_households + 1):
        rate = rng.gamma(2.0, 0.08)  # gün başına beklenen sepet (~ortalama 0.16)
        start = int(rng.integers(1, 120))
        if signal and rng.random() < 0.45:
            dropout = int(rng.integers(start + 90, horizon_days + 200))
        else:
            dropout = horizon_days + 10_000
        favourites = rng.choice(n_products, size=int(rng.integers(5, 20)), replace=False, p=popularity)
        coupon_user = rng.random() < 0.2

        day = start
        while day <= min(horizon_days, dropout):
            r = rate
            if signal and dropout - day < 60:  # bırakmadan önce yavaşlama
                r = rate * max(0.15, (dropout - day) / 60)
            gap = int(np.ceil(rng.exponential(1.0 / max(r, 1e-3))))
            day += max(gap, 1)
            if day > horizon_days or day > dropout:
                break
            basket_id += 1
            n_items = int(rng.integers(1, 12))
            n_fav = rng.binomial(n_items, 0.6)
            items = np.concatenate([
                rng.choice(favourites, size=n_fav),
                rng.choice(n_products, size=n_items - n_fav, p=popularity),
            ])
            week = (day - 1) // 7 + 1
            trans_time = int(rng.integers(6, 23)) * 100 + int(rng.integers(0, 60))
            store = int(rng.integers(300, 330))
            for idx in np.unique(items):
                qty = int(rng.integers(1, 4))
                gross = shelf_price[idx] * qty
                retail_disc = -round(gross * promo_depth[idx], 2) if on_promo[idx, week] else 0.0
                coupon_disc = -round(min(1.0, gross * 0.1), 2) if (coupon_user and rng.random() < 0.05) else 0.0
                coupon_match = round(coupon_disc * 0.5, 2) if coupon_disc else 0.0
                sales = round(gross + retail_disc + coupon_match, 2)
                rows.append((hh, basket_id, day, int(product_ids[idx]), qty, sales, store,
                             retail_disc, trans_time, week, coupon_disc, coupon_match))

    tx = pd.DataFrame(rows, columns=[
        "household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY", "SALES_VALUE",
        "STORE_ID", "RETAIL_DISC", "TRANS_TIME", "WEEK_NO", "COUPON_DISC", "COUPON_MATCH_DISC",
    ])
    return tx, products
