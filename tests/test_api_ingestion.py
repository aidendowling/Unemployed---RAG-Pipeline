import pytest

from unemployed_rag_pipeline.ingestion import fred, qwi
from unemployed_rag_pipeline.ingestion.http import APIError

FRED_PAYLOAD = {
    "observations": [
        {"date": "2020-01-01", "value": "3.6"},
        {"date": "2020-04-01", "value": "14.8"},
        {"date": "2020-07-01", "value": "."},
    ]
}

QWI_PAYLOAD = [
    ["Emp", "HirA", "year", "quarter", "state"],
    ["4131234", "251234", "2020", "1", "13"],
    ["3987654", "198765", "2020", "2", "13"],
]


def test_fetch_fred_series_normalizes_dates_and_missing_values(monkeypatch):
    captured = {}

    def fake_get_json(url, params=None, **kwargs):
        captured["url"] = url
        captured["params"] = params
        return FRED_PAYLOAD

    monkeypatch.setattr(fred, "get_json", fake_get_json)

    frame = fred.fetch_fred_series("UNRATE", api_key="test-key", observation_start="2020-01-01")

    assert captured["url"].endswith("/series/observations")
    assert captured["params"]["series_id"] == "UNRATE"
    assert captured["params"]["observation_start"] == "2020-01-01"
    assert frame["year"].tolist() == [2020, 2020, 2020]
    assert frame["month"].tolist() == [1, 4, 7]
    assert frame["vintage_year"].tolist() == [2020, 2020, 2020]
    assert frame["value"].iloc[0] == pytest.approx(3.6)
    assert frame["value"].isna().iloc[2]
    assert frame["source"].iloc[0] == "FRED"


def test_save_fred_series_writes_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(fred, "get_json", lambda url, params=None, **kwargs: FRED_PAYLOAD)

    result = fred.save_fred_series("UNRATE", output_dir=tmp_path, api_key="test-key")

    assert result.row_count == 3
    assert result.start_date == "2020-01-01"
    assert result.end_date == "2020-07-01"
    assert result.output_path.exists()
    assert "vintage_year" in result.output_path.read_text(encoding="utf-8").splitlines()[0]


def test_fred_requires_api_key(monkeypatch):
    monkeypatch.setattr(fred.config, "FRED_API_KEY", None)

    with pytest.raises(fred.FredAPIError, match="FRED_API_KEY is not set"):
        fred.fetch_fred_series("UNRATE")


def test_fred_wraps_transport_errors(monkeypatch):
    def boom(url, params=None, **kwargs):
        raise APIError("HTTP 429 from https://api.stlouisfed.org/fred/series/observations?api_key=***")

    monkeypatch.setattr(fred, "get_json", boom)

    with pytest.raises(fred.FredAPIError, match="HTTP 429"):
        fred.fetch_fred_series("UNRATE", api_key="test-key")


def test_fetch_qwi_data_builds_request_and_tidies_response(monkeypatch):
    captured = {}

    def fake_get_json(url, params=None, **kwargs):
        captured["url"] = url
        captured["params"] = params
        return QWI_PAYLOAD

    monkeypatch.setattr(qwi, "get_json", fake_get_json)

    frame = qwi.fetch_qwi_data(
        state="13", years=[2020], quarters=[1, 2], variables=["Emp", "HirA"], api_key="census-key"
    )

    assert captured["url"].endswith("/sa")
    assert captured["params"]["get"] == "Emp,HirA"
    assert captured["params"]["for"] == "state:13"
    assert captured["params"]["quarter"] == "1,2"
    assert frame["vintage_year"].tolist() == [2020, 2020]
    assert frame["month"].tolist() == [1, 4]
    assert frame["Emp"].iloc[0] == 4131234
    assert frame["source"].iloc[0] == "Census QWI"


def test_qwi_county_query_sets_in_parameter(monkeypatch):
    captured = {}

    def fake_get_json(url, params=None, **kwargs):
        captured.update(params or {})
        return QWI_PAYLOAD

    monkeypatch.setattr(qwi, "get_json", fake_get_json)

    qwi.fetch_qwi_data(state="13", county="121", years=[2020], api_key="census-key")

    assert captured["for"] == "county:121"
    assert captured["in"] == "state:13"


def test_qwi_rejects_unknown_dataset():
    with pytest.raises(qwi.QWIAPIError, match="Unknown QWI dataset"):
        qwi.fetch_qwi_data(state="13", years=[2020], dataset="xx")


def test_qwi_requires_years():
    with pytest.raises(qwi.QWIAPIError, match="At least one year"):
        qwi.fetch_qwi_data(state="13", years=[])


def test_save_qwi_data_writes_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(qwi, "get_json", lambda url, params=None, **kwargs: QWI_PAYLOAD)

    result = qwi.save_qwi_data(state="13", years=[2020], output_dir=tmp_path, api_key="k")

    assert result.row_count == 2
    assert result.years == (2020,)
    assert result.output_path.exists()
