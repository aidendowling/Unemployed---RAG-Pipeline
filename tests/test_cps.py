import json

import pytest

from unemployed_rag_pipeline.ingestion import cps


def build_record(values: dict[str, str], layout: cps.CPSLayout) -> str:
    """Render one fixed-width CPS record from field name -> raw value."""
    record = [" "] * layout.record_length
    for field in layout.fields:
        raw = values.get(field.name, "")
        width = field.end - field.start + 1
        cell = str(raw).rjust(width)[:width]
        record[field.start - 1 : field.end] = list(cell)
    return "".join(record)


def write_dat(tmp_path, records, layout, name="cps_sample.dat"):
    path = tmp_path / name
    path.write_text("\n".join(build_record(values, layout) for values in records), encoding="utf-8")
    return path


@pytest.fixture
def builtin_layout():
    return cps.load_layout("cps-basic-monthly")


@pytest.fixture
def sample_records():
    return [
        {"HRHHID": "000000000000001", "HRMONTH": "6", "HRYEAR4": "2020", "GESTFIPS": "6",
         "PRTAGE": "34", "PESEX": "1", "PEMLR": "1", "PWCMPWGT": "12000000"},
        {"HRHHID": "000000000000002", "HRMONTH": "6", "HRYEAR4": "2020", "GESTFIPS": "6",
         "PRTAGE": "45", "PESEX": "2", "PEMLR": "4", "PWCMPWGT": "8000000"},
        {"HRHHID": "000000000000003", "HRMONTH": "6", "HRYEAR4": "2020", "GESTFIPS": "48",
         "PRTAGE": "29", "PESEX": "2", "PEMLR": "1", "PWCMPWGT": "10000000"},
        {"HRHHID": "000000000000004", "HRMONTH": "6", "HRYEAR4": "2020", "GESTFIPS": "48",
         "PRTAGE": "61", "PESEX": "1", "PEMLR": "5", "PWCMPWGT": "9000000"},
    ]


def test_builtin_layout_parses_and_decodes(tmp_path, builtin_layout, sample_records):
    dat_path = write_dat(tmp_path, sample_records, builtin_layout)

    frame = cps.read_cps_dat(dat_path, layout="cps-basic-monthly")

    assert len(frame) == 4
    assert frame["year"].tolist() == [2020] * 4
    assert frame["month"].tolist() == [6] * 4
    assert frame["vintage_year"].tolist() == [2020] * 4
    assert frame["state"].tolist() == ["CA", "CA", "TX", "TX"]
    assert frame["labor_force_status"].iloc[1] == "Unemployed - looking"
    assert frame["is_unemployed"].tolist() == [False, True, False, False]
    # PWCMPWGT carries four implied decimals
    assert frame["PWCMPWGT"].iloc[0] == pytest.approx(1200.0)


def test_summarize_cps_computes_weighted_unemployment_rate(tmp_path, builtin_layout, sample_records):
    dat_path = write_dat(tmp_path, sample_records, builtin_layout)

    summary = cps.summarize_cps(cps.read_cps_dat(dat_path, layout="cps-basic-monthly"))

    california = summary[summary["state"] == "CA"].iloc[0]
    assert california["labor_force"] == pytest.approx(2000.0)
    assert california["unemployment_rate"] == pytest.approx(40.0)

    texas = summary[summary["state"] == "TX"].iloc[0]
    # The retired person (PEMLR 5) is outside the labor force entirely
    assert texas["labor_force"] == pytest.approx(1000.0)
    assert texas["unemployment_rate"] == pytest.approx(0.0)
    assert summary["weighted"].iloc[0] == "PWCMPWGT"


def test_wrong_layout_raises_instead_of_producing_garbage(tmp_path, builtin_layout, sample_records):
    dat_path = write_dat(tmp_path, sample_records, builtin_layout)
    # Reads the household ID digits as if they were the survey year
    shifted = tmp_path / "shifted.json"
    shifted.write_text(
        json.dumps([{"name": "HRYEAR4", "start": 1, "end": 4, "dtype": "int"}]),
        encoding="utf-8",
    )

    with pytest.raises(cps.CPSLayoutError, match="implausible"):
        cps.read_cps_dat(dat_path, layout=shifted)


def test_json_sidecar_layout_is_auto_detected(tmp_path):
    layout = cps.CPSLayout(
        name="tiny",
        fields=(
            cps.FieldSpec(name="HRYEAR4", start=1, end=4),
            cps.FieldSpec(name="GESTFIPS", start=5, end=6),
            cps.FieldSpec(name="PEMLR", start=7, end=8),
        ),
        source="test",
    )
    dat_path = write_dat(tmp_path, [{"HRYEAR4": "2019", "GESTFIPS": "13", "PEMLR": "3"}], layout, "tiny.dat")
    (tmp_path / "tiny.json").write_text(
        json.dumps([
            {"name": "HRYEAR4", "start": 1, "end": 4},
            {"name": "GESTFIPS", "start": 5, "end": 6},
            {"name": "PEMLR", "start": 7, "end": 8},
        ]),
        encoding="utf-8",
    )

    frame = cps.read_cps_dat(dat_path)

    assert frame["state"].iloc[0] == "GA"
    assert frame["vintage_year"].iloc[0] == 2019
    assert bool(frame["is_unemployed"].iloc[0]) is True


def test_ipums_ddi_layout_is_parsed(tmp_path):
    ddi = tmp_path / "cps_00001.xml"
    ddi.write_text(
        """<?xml version="1.0"?>
        <codeBook xmlns="ddi:codebook:2_5">
          <dataDscr>
            <var ID="YEAR" name="YEAR" dcml="0">
              <location StartPos="1" EndPos="4" width="4"/>
              <labl>Survey year</labl>
            </var>
            <var ID="STATEFIP" name="GESTFIPS" dcml="0">
              <location StartPos="5" EndPos="6" width="2"/>
              <labl>State FIPS code</labl>
            </var>
            <var ID="EMPSTAT" name="PEMLR" dcml="0">
              <location StartPos="7" EndPos="8" width="2"/>
              <labl>Labor force status</labl>
            </var>
          </dataDscr>
        </codeBook>""",
        encoding="utf-8",
    )

    layout = cps.load_layout(ddi)

    assert [field.name for field in layout.fields] == ["YEAR", "GESTFIPS", "PEMLR"]
    assert layout.fields[0].description == "Survey year"
    assert layout.record_length == 8


def test_census_data_dictionary_layout_is_parsed(tmp_path):
    dictionary = tmp_path / "cps_dictionary.txt"
    dictionary.write_text(
        "\n".join(
            [
                "This is prose that should be ignored.",
                "D HRHHID       15      1",
                "D HRMONTH       2     16",
                "D HRYEAR4       4     18",
            ]
        ),
        encoding="utf-8",
    )

    layout = cps.load_layout(dictionary)

    assert [(f.name, f.start, f.end) for f in layout.fields] == [
        ("HRHHID", 1, 15),
        ("HRMONTH", 16, 17),
        ("HRYEAR4", 18, 21),
    ]


def test_missing_layout_reports_actionable_error(tmp_path):
    with pytest.raises(cps.CPSLayoutError, match="not found"):
        cps.load_layout(tmp_path / "nope.xml")
