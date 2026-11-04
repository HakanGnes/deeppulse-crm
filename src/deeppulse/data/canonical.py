"""Kanonik (kaynaktan bağımsız) işlem ve ürün tabloları.

Modelleme kodunun tamamı bu iki tabloyla çalışır:

transactions: customer_id, basket_id, day, trans_time, product_id, quantity,
              net_value, gross_value, retail_disc, coupon_match_disc,
              coupon_disc, store_id
products:     product_id, department, commodity, sub_commodity, brand, manufacturer

Tutar tanımları (indirimler burada POZİTİF büyüklük olarak tutulur):
- net_value      = Dunnhumby SALES_VALUE (perakendecinin aldığı tutar)
- retail_disc    = -RETAIL_DISC (sadakat kartı / raf indirimi)
- coupon_match_disc = -COUPON_MATCH_DISC (perakendecinin kupon eşleştirmesi)
- coupon_disc    = -COUPON_DISC (üretici kuponu; perakendeci için istatistiksel)
- gross_value    = net_value + retail_disc + coupon_match_disc (indirimsiz raf değeri)
Bu tanımın gerçek veride tutarlılığı docs/decisions.md'de "doğrulanacak" olarak işaretli.
"""

from __future__ import annotations

import pandas as pd

from deeppulse.data.dunnhumby import DunnhumbyRaw

TRANSACTION_COLUMNS = [
    "customer_id", "basket_id", "day", "trans_time", "product_id", "quantity",
    "net_value", "gross_value", "retail_disc", "coupon_match_disc", "coupon_disc", "store_id",
]
PRODUCT_COLUMNS = ["product_id", "department", "commodity", "sub_commodity", "brand", "manufacturer"]


def canonical_from_raw(
    raw: DunnhumbyRaw,
    exclude_departments: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tx = raw.transactions
    # Miktarı ya da tutarı sıfır/negatif satırlar modellemeye katılmaz
    tx = tx[(tx["quantity"] > 0) & (tx["sales_value"] > 0)]

    products = pd.DataFrame({
        "product_id": raw.products["product_id"].astype(str),
        "department": raw.products["department"].astype(str).str.strip(),
        "commodity": raw.products["commodity_desc"].astype(str).str.strip(),
        "sub_commodity": raw.products["sub_commodity_desc"].astype(str).str.strip(),
        "brand": raw.products["brand"].astype(str).str.strip(),
        "manufacturer": raw.products["manufacturer"].astype(str),
    })

    out = pd.DataFrame({
        "customer_id": tx["household_key"].astype(str),
        "basket_id": tx["basket_id"].astype(str),
        "day": tx["day"].astype("int32"),
        "trans_time": tx["trans_time"].astype("int32"),
        "product_id": tx["product_id"].astype(str),
        "quantity": tx["quantity"].astype("int64"),
        "net_value": tx["sales_value"].astype(float),
        "retail_disc": (-tx["retail_disc"]).clip(lower=0).astype(float),
        "coupon_match_disc": (-tx["coupon_match_disc"]).clip(lower=0).astype(float),
        "coupon_disc": (-tx["coupon_disc"]).clip(lower=0).astype(float),
        "store_id": tx["store_id"].astype(str),
    })
    out["gross_value"] = out["net_value"] + out["retail_disc"] + out["coupon_match_disc"]

    if exclude_departments:
        keep = products.loc[~products["department"].isin(exclude_departments), "product_id"]
        out = out[out["product_id"].isin(set(keep))]

    out = out[TRANSACTION_COLUMNS].sort_values(["day", "trans_time", "basket_id", "product_id"])
    return out.reset_index(drop=True), products[PRODUCT_COLUMNS]
