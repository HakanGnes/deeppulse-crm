"""Dunnhumby "The Complete Journey" ham CSV yükleyici.

Beklenen klasör (orijinal indirme, CSV'ler tek klasörde):

    data/raw/
        transaction_data.csv   (zorunlu)
        product.csv            (zorunlu)
        hh_demographic.csv     (isteğe bağlı)
        campaign_table.csv     (isteğe bağlı)
        campaign_desc.csv      (isteğe bağlı)
        coupon.csv             (isteğe bağlı)
        coupon_redempt.csv     (isteğe bağlı)
        causal_data.csv        (isteğe bağlı, ~700 MB — Faz 0'da kullanılmıyor)

Sütun adları küçük harfe çevrilir (orijinalde BASKET_ID, household_key gibi karışık).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REQUIRED_FILES = ("transaction_data", "product")
OPTIONAL_FILES = (
    "hh_demographic",
    "campaign_table",
    "campaign_desc",
    "coupon",
    "coupon_redempt",
)

TRANSACTION_COLUMNS = (
    "household_key", "basket_id", "day", "product_id", "quantity", "sales_value",
    "store_id", "retail_disc", "trans_time", "week_no", "coupon_disc", "coupon_match_disc",
)
PRODUCT_COLUMNS = (
    "product_id", "manufacturer", "department", "brand",
    "commodity_desc", "sub_commodity_desc", "curr_size_of_product",
)

# Büyük tabloda bellek tasarrufu: kimlikler int64, tutarlar float32 yeterli
_TRANSACTION_DTYPES = {
    "household_key": "int64", "basket_id": "int64", "day": "int32",
    "product_id": "int64", "quantity": "int64", "sales_value": "float64",
    "store_id": "int64", "retail_disc": "float64", "trans_time": "int32",
    "week_no": "int32", "coupon_disc": "float64", "coupon_match_disc": "float64",
}


class SchemaError(ValueError):
    """Ham dosyada beklenen sütunlar yoksa."""


@dataclass
class DunnhumbyRaw:
    transactions: pd.DataFrame
    products: pd.DataFrame
    extras: dict[str, pd.DataFrame] = field(default_factory=dict)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def validate_columns(df: pd.DataFrame, required: tuple[str, ...], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{name}: eksik sütun(lar) {missing}. Mevcut: {list(df.columns)}")


def normalise_raw(transactions: pd.DataFrame, products: pd.DataFrame,
                  extras: dict[str, pd.DataFrame] | None = None) -> DunnhumbyRaw:
    """Bellekteki ham tabloları (CSV'den ya da sentetik üreteçten) doğrular ve tipler."""
    tx = _normalise_columns(transactions)
    pr = _normalise_columns(products)
    validate_columns(tx, TRANSACTION_COLUMNS, "transaction_data")
    validate_columns(pr, PRODUCT_COLUMNS, "product")
    tx = tx.astype({c: t for c, t in _TRANSACTION_DTYPES.items() if c in tx.columns})
    pr = pr.astype({"product_id": "int64"})
    return DunnhumbyRaw(
        transactions=tx,
        products=pr,
        extras={k: _normalise_columns(v) for k, v in (extras or {}).items()},
    )


def load_raw(raw_dir: str | Path) -> DunnhumbyRaw:
    raw_dir = Path(raw_dir)
    paths = {}
    for name in REQUIRED_FILES:
        p = raw_dir / f"{name}.csv"
        if not p.exists():
            raise FileNotFoundError(
                f"{p} bulunamadı. Dunnhumby CSV'lerini {raw_dir} içine koy (README → Veri)."
            )
        paths[name] = p
    tx = pd.read_csv(paths["transaction_data"])
    pr = pd.read_csv(paths["product"])
    extras = {}
    for name in OPTIONAL_FILES:
        p = raw_dir / f"{name}.csv"
        if p.exists():
            extras[name] = pd.read_csv(p)
    return normalise_raw(tx, pr, extras)


def summarise(raw: DunnhumbyRaw) -> dict[str, float]:
    tx = raw.transactions
    return {
        "rows": len(tx),
        "households": tx["household_key"].nunique(),
        "baskets": tx["basket_id"].nunique(),
        "products": tx["product_id"].nunique(),
        "day_min": int(tx["day"].min()),
        "day_max": int(tx["day"].max()),
        "share_rows_with_retail_disc": float((tx["retail_disc"] < 0).mean()),
        "share_rows_with_coupon": float((tx["coupon_disc"] < 0).mean()),
    }
