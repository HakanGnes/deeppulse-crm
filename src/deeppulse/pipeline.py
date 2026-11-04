"""Uçtan uca Faz 0 hattı: ham veri → SAP tabloları → kanonik → fold → baseline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from deeppulse.config import Config
from deeppulse.data.canonical import canonical_from_raw
from deeppulse.data.dunnhumby import DunnhumbyRaw, load_raw, normalise_raw
from deeppulse.data.synthetic import make_dataset
from deeppulse.sap.adapter import from_sap, load_tables, save_tables, to_sap


def build_sap_tables(raw: DunnhumbyRaw, cfg: Config, exclude_departments: tuple[str, ...] = ()) -> dict:
    tx, products = canonical_from_raw(raw, exclude_departments=exclude_departments)
    return to_sap(tx, products, raw.extras.get("hh_demographic"), cfg.sap)


def load_canonical(
    raw_dir: str | Path | None,
    sap_dir: str | Path,
    cfg: Config,
    synthetic: bool = False,
    rebuild: bool = False,
    exclude_departments: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SAP tabloları diskte varsa onları okur, yoksa ham veriden üretip kaydeder."""
    sap_dir = Path(sap_dir)
    if rebuild or not (sap_dir / "VBRK.parquet").exists():
        if synthetic:
            tx_raw, pr_raw = make_dataset(seed=cfg.random_state)
            raw = normalise_raw(tx_raw, pr_raw)
        else:
            if raw_dir is None:
                raise ValueError("raw_dir gerekli (ya da synthetic=True).")
            raw = load_raw(raw_dir)
        save_tables(build_sap_tables(raw, cfg, exclude_departments), sap_dir)
    return from_sap(load_tables(sap_dir), cfg.sap)


def write_report(report: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
