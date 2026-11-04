"""Kanonik tablolar ⇄ SAP tarzı tablolar.

to_sap   : kanonik işlem/ürün (+ isteğe bağlı demografi) → KNA1, ZKNA1_DEMO, MARA,
           T023T, TSPAT, VBRK, VBRP, PRCD_ELEMENTS, BKPF, BSEG
from_sap : SAP tabloları → kanonik işlem/ürün tabloları

Modelleme hattı her zaman from_sap çıktısını kullanır. Gerçek bir SAP
sisteminden aynı alanlarla export alınırsa, to_sap adımı atlanır.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from deeppulse.config import SapConfig
from deeppulse.data.canonical import PRODUCT_COLUMNS, TRANSACTION_COLUMNS
from deeppulse.sap.schema import (
    CONDITION_COUPON_MATCH,
    CONDITION_GROSS,
    CONDITION_MANUFACTURER_COUPON,
    CONDITION_RETAIL_DISC,
)

_DEMO_MAP = {
    "age_desc": "ZZAGE_DESC",
    "income_desc": "ZZINCOME_DESC",
    "hh_comp_desc": "ZZHH_COMP_DESC",
    "household_size_desc": "ZZHH_SIZE_DESC",
    "kid_category_desc": "ZZKID_CAT_DESC",
    "homeowner_desc": "ZZHOMEOWNER_DESC",
    "marital_status_code": "ZZMARITAL_CODE",
}


def _pad(s: pd.Series, width: int) -> pd.Series:
    return s.astype(str).str.zfill(width)


def _unpad(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.lstrip("0")
    return out.where(out != "", "0")


def _day_to_date(day: pd.Series, base: date) -> pd.Series:
    return pd.Timestamp(base) + pd.to_timedelta(day.astype("int64") - 1, unit="D")


def _date_to_day(d: pd.Series, base: date) -> pd.Series:
    return ((pd.to_datetime(d) - pd.Timestamp(base)).dt.days + 1).astype("int32")


def _code_map(values: pd.Series, prefix: str, width: int) -> dict[str, str]:
    uniq = sorted(values.dropna().astype(str).unique())
    return {v: f"{prefix}{i + 1:0{width}d}" for i, v in enumerate(uniq)}


def to_sap(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    demographics: pd.DataFrame | None = None,
    cfg: SapConfig = SapConfig(),
) -> dict[str, pd.DataFrame]:
    tx = transactions
    tables: dict[str, pd.DataFrame] = {}

    # --- Ana veriler -------------------------------------------------------
    customers = pd.Series(sorted(tx["customer_id"].unique(), key=lambda x: int(x)))
    kunnr = _pad(customers, 10)
    tables["KNA1"] = pd.DataFrame({
        "KUNNR": kunnr,
        "NAME1": "Household " + customers,
        "LAND1": "US",
        "KTOKD": "Z001",
    })
    if demographics is not None and len(demographics):
        demo = demographics.rename(columns=_DEMO_MAP)
        demo["KUNNR"] = _pad(demo["household_key"], 10)
        tables["ZKNA1_DEMO"] = demo[["KUNNR", *[c for c in _DEMO_MAP.values() if c in demo]]]

    matkl = _code_map(products["commodity"], "C", 5)
    spart = _code_map(products["department"], "", 2)
    tables["MARA"] = pd.DataFrame({
        "MATNR": _pad(products["product_id"], 18),
        "MATKL": products["commodity"].astype(str).map(matkl),
        "SPART": products["department"].astype(str).map(spart),
        "MFRNR": _pad(products["manufacturer"], 10),
        "ZZBRAND": products["brand"],
        "ZZSUBCOMM": products["sub_commodity"],
    })
    tables["T023T"] = pd.DataFrame({"MATKL": list(matkl.values()), "WGBEZ": list(matkl.keys())})
    tables["TSPAT"] = pd.DataFrame({"SPART": list(spart.values()), "VTEXT": list(spart.keys())})

    # --- SD faturaları -----------------------------------------------------
    baskets = (
        tx.groupby("basket_id", sort=False)
        .agg(customer_id=("customer_id", "first"), day=("day", "first"),
             trans_time=("trans_time", "first"), net=("net_value", "sum"),
             gross=("gross_value", "sum"),
             disc=("retail_disc", "sum"), match=("coupon_match_disc", "sum"))
        .reset_index()
        .sort_values(["day", "trans_time", "basket_id"], kind="stable")
        .reset_index(drop=True)
    )
    seq = np.arange(1, len(baskets) + 1)
    baskets["VBELN"] = pd.Series(9_000_000_000 + seq).astype(str).str.zfill(10)
    baskets["KNUMV"] = pd.Series(seq).astype(str).str.zfill(10)
    fkdat = _day_to_date(baskets["day"], cfg.base_date)
    tables["VBRK"] = pd.DataFrame({
        "VBELN": baskets["VBELN"],
        "FKART": "F2",
        "FKDAT": fkdat,
        "ERZET": (baskets["trans_time"].astype(int) * 100).astype(str).str.zfill(6),
        "VKORG": cfg.sales_org,
        "BUKRS": cfg.company_code,
        "KUNAG": _pad(baskets["customer_id"], 10),
        "NETWR": baskets["net"].round(2),
        "WAERK": cfg.currency,
        "KNUMV": baskets["KNUMV"],
        "XBLNR": baskets["basket_id"].astype(str),
    })

    items = tx.merge(baskets[["basket_id", "VBELN", "KNUMV"]], on="basket_id", how="left")
    items = items.sort_values(["VBELN", "product_id"], kind="stable")
    items["POSNR"] = ((items.groupby("VBELN").cumcount() + 1) * 10).astype(str).str.zfill(6)
    tables["VBRP"] = pd.DataFrame({
        "VBELN": items["VBELN"].to_numpy(),
        "POSNR": items["POSNR"].to_numpy(),
        "MATNR": _pad(items["product_id"], 18).to_numpy(),
        "FKIMG": items["quantity"].to_numpy(),
        "VRKME": "EA",
        "NETWR": items["net_value"].to_numpy(),
        "WERKS": _pad(items["store_id"], 4).to_numpy(),
    })

    conds = []
    for kschl, col, sign, kstat in (
        (CONDITION_GROSS, "gross_value", 1.0, ""),
        (CONDITION_RETAIL_DISC, "retail_disc", -1.0, ""),
        (CONDITION_COUPON_MATCH, "coupon_match_disc", -1.0, ""),
        (CONDITION_MANUFACTURER_COUPON, "coupon_disc", -1.0, "X"),
    ):
        part = items if kschl == CONDITION_GROSS else items[items[col] > 0]
        conds.append(pd.DataFrame({
            "KNUMV": part["KNUMV"].to_numpy(),
            "KPOSN": part["POSNR"].to_numpy(),
            "KSCHL": kschl,
            "KWERT": sign * part[col].to_numpy(),
            "KSTAT": kstat,
        }))
    tables["PRCD_ELEMENTS"] = pd.concat(conds, ignore_index=True)

    # --- FI muhasebe kayıtları (fatura → muhasebe aktarımı) -----------------
    belnr = pd.Series(100_000_000 + seq).astype(str).str.zfill(10)
    gjahr = fkdat.dt.year.astype(str)
    tables["BKPF"] = pd.DataFrame({
        "BUKRS": cfg.company_code,
        "BELNR": belnr,
        "GJAHR": gjahr,
        "BLART": "RV",
        "BUDAT": fkdat,
        "AWTYP": "VBRK",
        "AWKEY": baskets["VBELN"],
    })
    discount = (baskets["disc"] + baskets["match"]).round(2)
    lines = [
        # Müşteri borç: net tutar
        pd.DataFrame({"BELNR": belnr, "GJAHR": gjahr, "BUZEI": "001", "KOART": "D", "SHKZG": "S",
                      "KUNNR": _pad(baskets["customer_id"], 10), "HKONT": "",
                      "DMBTR": baskets["net"].round(2)}),
        # Satış geliri alacak: brüt tutar
        pd.DataFrame({"BELNR": belnr, "GJAHR": gjahr, "BUZEI": "002", "KOART": "S", "SHKZG": "H",
                      "KUNNR": "", "HKONT": cfg.revenue_gl_account,
                      "DMBTR": baskets["gross"].round(2)}),
    ]
    has_disc = discount > 0
    lines.append(pd.DataFrame({
        "BELNR": belnr[has_disc], "GJAHR": gjahr[has_disc], "BUZEI": "003", "KOART": "S",
        "SHKZG": "S", "KUNNR": "", "HKONT": cfg.discount_gl_account, "DMBTR": discount[has_disc],
    }))
    bseg = pd.concat(lines, ignore_index=True)
    bseg.insert(0, "BUKRS", cfg.company_code)
    tables["BSEG"] = bseg.sort_values(["BELNR", "BUZEI"]).reset_index(drop=True)
    return tables


def from_sap(tables: dict[str, pd.DataFrame], cfg: SapConfig = SapConfig()) -> tuple[pd.DataFrame, pd.DataFrame]:
    vbrk = tables["VBRK"]
    vbrp = tables["VBRP"]
    prcd = tables["PRCD_ELEMENTS"]

    items = vbrp.merge(
        vbrk[["VBELN", "KUNAG", "FKDAT", "ERZET", "KNUMV", "XBLNR"]], on="VBELN", how="left"
    )
    pivot = (
        prcd.pivot_table(index=["KNUMV", "KPOSN"], columns="KSCHL", values="KWERT",
                         aggfunc="sum", fill_value=0.0)
        .reset_index()
    )
    for k in (CONDITION_GROSS, CONDITION_RETAIL_DISC, CONDITION_COUPON_MATCH, CONDITION_MANUFACTURER_COUPON):
        if k not in pivot:
            pivot[k] = 0.0
    items = items.merge(pivot, left_on=["KNUMV", "POSNR"], right_on=["KNUMV", "KPOSN"], how="left")

    tx = pd.DataFrame({
        "customer_id": _unpad(items["KUNAG"]),
        "basket_id": items["XBLNR"].astype(str),
        "day": _date_to_day(items["FKDAT"], cfg.base_date),
        "trans_time": (items["ERZET"].astype(int) // 100).astype("int32"),
        "product_id": _unpad(items["MATNR"]),
        "quantity": items["FKIMG"].astype("int64"),
        "net_value": items["NETWR"].astype(float),
        "gross_value": items[CONDITION_GROSS].astype(float),
        "retail_disc": (-items[CONDITION_RETAIL_DISC]).astype(float),
        "coupon_match_disc": (-items[CONDITION_COUPON_MATCH]).astype(float),
        "coupon_disc": (-items[CONDITION_MANUFACTURER_COUPON]).astype(float),
        "store_id": _unpad(items["WERKS"]),
    })
    tx = tx[TRANSACTION_COLUMNS].sort_values(["day", "trans_time", "basket_id", "product_id"])

    mara = tables["MARA"]
    t023t = tables["T023T"].set_index("MATKL")["WGBEZ"]
    tspat = tables["TSPAT"].set_index("SPART")["VTEXT"]
    products = pd.DataFrame({
        "product_id": _unpad(mara["MATNR"]),
        "department": mara["SPART"].map(tspat),
        "commodity": mara["MATKL"].map(t023t),
        "sub_commodity": mara["ZZSUBCOMM"],
        "brand": mara["ZZBRAND"],
        "manufacturer": _unpad(mara["MFRNR"]),
    })
    return tx.reset_index(drop=True), products[PRODUCT_COLUMNS]


def save_tables(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)


def load_tables(in_dir: str | Path) -> dict[str, pd.DataFrame]:
    in_dir = Path(in_dir)
    return {p.stem: pd.read_parquet(p) for p in sorted(in_dir.glob("*.parquet"))}
