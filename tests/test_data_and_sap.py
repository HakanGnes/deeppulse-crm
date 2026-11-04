import pandas as pd
import pytest

from deeppulse.data.dunnhumby import SchemaError, load_raw, normalise_raw
from deeppulse.data.synthetic import make_dataset
from deeppulse.sap.adapter import from_sap, load_tables, save_tables, to_sap


def test_load_raw_from_csv_folder(tmp_path):
    tx, pr = make_dataset(n_households=20, n_products=50, seed=1)
    tx.to_csv(tmp_path / "transaction_data.csv", index=False)
    pr.to_csv(tmp_path / "product.csv", index=False)
    raw = load_raw(tmp_path)
    assert "basket_id" in raw.transactions.columns  # küçük harfe normalize
    assert len(raw.transactions) == len(tx)


def test_missing_required_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="transaction_data.csv"):
        load_raw(tmp_path)


def test_missing_column_raises_schema_error():
    tx, pr = make_dataset(n_households=5, n_products=20, seed=1)
    with pytest.raises(SchemaError, match="retail_disc"):
        normalise_raw(tx.drop(columns=["RETAIL_DISC"]), pr)


def test_canonical_discounts_are_positive_and_gross_consistent(synthetic_canonical):
    tx, _ = synthetic_canonical
    for c in ("retail_disc", "coupon_match_disc", "coupon_disc"):
        assert (tx[c] >= 0).all()
    diff = tx["gross_value"] - (tx["net_value"] + tx["retail_disc"] + tx["coupon_match_disc"])
    assert diff.abs().max() < 1e-9


def test_sap_round_trip_reproduces_canonical(synthetic_canonical):
    tx, products = synthetic_canonical
    tx2, products2 = from_sap(to_sap(tx, products))
    pd.testing.assert_frame_equal(tx.reset_index(drop=True), tx2, check_dtype=False)
    key = "product_id"
    pd.testing.assert_frame_equal(
        products.sort_values(key).reset_index(drop=True),
        products2.sort_values(key).reset_index(drop=True),
        check_dtype=False,
    )


def test_sap_round_trip_through_parquet(tmp_path, synthetic_canonical):
    tx, products = synthetic_canonical
    save_tables(to_sap(tx, products), tmp_path)
    tx2, _ = from_sap(load_tables(tmp_path))
    pd.testing.assert_frame_equal(tx.reset_index(drop=True), tx2, check_dtype=False)


def test_sap_documents_are_internally_consistent(synthetic_canonical):
    tx, products = synthetic_canonical
    t = to_sap(tx, products)
    # Fatura başlığı net değeri = kalemlerin toplamı
    item_sum = t["VBRP"].groupby("VBELN")["NETWR"].sum()
    header = t["VBRK"].set_index("VBELN")["NETWR"]
    assert (item_sum.reindex(header.index) - header).abs().max() < 0.011
    # Her FI belgesi borç = alacak
    b = t["BSEG"]
    signed = b["DMBTR"].where(b["SHKZG"] == "S", -b["DMBTR"])
    assert signed.groupby(b["BELNR"]).sum().abs().max() < 0.011
    # Her fatura bir muhasebe belgesine bağlı
    assert set(t["BKPF"]["AWKEY"]) == set(t["VBRK"]["VBELN"])
    # SAP alan uzunlukları
    assert t["KNA1"]["KUNNR"].str.len().eq(10).all()
    assert t["MARA"]["MATNR"].str.len().eq(18).all()
    assert t["VBRK"]["VBELN"].str.len().eq(10).all()
