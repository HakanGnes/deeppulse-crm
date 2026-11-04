import pandas as pd
import pytest

from deeppulse.data.canonical import TRANSACTION_COLUMNS, canonical_from_raw
from deeppulse.data.dunnhumby import normalise_raw
from deeppulse.data.synthetic import make_dataset


@pytest.fixture(scope="session")
def synthetic_raw():
    tx, pr = make_dataset(n_households=150, n_products=200, seed=7)
    return normalise_raw(tx, pr)


@pytest.fixture(scope="session")
def synthetic_canonical(synthetic_raw):
    return canonical_from_raw(synthetic_raw)


def make_tx(purchases: dict[str, list[int]]) -> pd.DataFrame:
    """El yapımı küçük işlem tablosu: {müşteri: [alışveriş günleri]} → her gün bir sepet."""
    rows = []
    b = 0
    for cust, days in purchases.items():
        for d in days:
            b += 1
            rows.append({
                "customer_id": cust, "basket_id": str(b), "day": d, "trans_time": 1200,
                "product_id": "1", "quantity": 1, "net_value": 1.0, "gross_value": 1.0,
                "retail_disc": 0.0, "coupon_match_disc": 0.0, "coupon_disc": 0.0, "store_id": "1",
            })
    return pd.DataFrame(rows, columns=TRANSACTION_COLUMNS)
