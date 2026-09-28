import pytest

from data import get_data


def test_get_data_returns_dataframe():
    data = get_data("^GSPC")

    assert not data.empty


def test_get_data_has_expected_columns():
    data = get_data("^GSPC")

    assert list(data.columns) == ["Adj Close"]
    assert data["Adj Close"].dtype == "float64"


def test_get_data_invalid_ticker():
    with pytest.raises(RuntimeError):
        get_data("INVALID_TICKER")
