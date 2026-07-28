"""Census QWI (Quarterly Workforce Indicators) API ingestion.

The QWI API is a Census timeseries endpoint, e.g.::

    https://api.census.gov/data/timeseries/qwi/sa?get=Emp,HirA&for=state:13&year=2021&quarter=1

An API key is optional for light use but recommended (https://api.census.gov/data/key_signup.html).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .. import config
from .http import APIError, get_json

DEFAULT_VARIABLES = ("Emp", "EmpEnd", "HirA", "Sep", "EarnBeg", "FrmJbGn")
DEFAULT_DATASET = "sa"
SUPPORTED_DATASETS = {"sa", "se", "rh"}


class QWIAPIError(RuntimeError):
    """Raised when the QWI API cannot be queried or returns an error."""


@dataclass(frozen=True, slots=True)
class QWIDownloadResult:
    """Summary of a single QWI download."""

    dataset: str
    geography: str
    output_path: Path
    row_count: int
    years: tuple[int, ...]


def _quarter_to_month(quarter: int) -> int:
    """Return the first calendar month of a quarter."""
    return (quarter - 1) * 3 + 1


def fetch_qwi_data(
    *,
    state: str,
    years: list[int],
    quarters: list[int] | None = None,
    variables: list[str] | None = None,
    dataset: str = DEFAULT_DATASET,
    county: str | None = None,
    industry: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> pd.DataFrame:
    """Download QWI observations as a tidy DataFrame.

    Args:
        state: Two-digit state FIPS code, e.g. ``13`` for Georgia.
        years: Calendar years to request.
        quarters: Quarters (1-4); defaults to all four.
        variables: QWI variable names; defaults to :data:`DEFAULT_VARIABLES`.
        dataset: ``sa`` (seasonally adjusted), ``se`` (worker sex/education),
            or ``rh`` (race/ethnicity).
        county: Optional three-digit county FIPS; requires ``state``.
        industry: Optional NAICS sector code, e.g. ``23``.
        api_key: Census API key; falls back to ``CENSUS_API_KEY``.
        base_url: Override for the QWI API root.

    Returns:
        DataFrame with one row per observation plus ``year``, ``quarter``,
        ``vintage_year``, and provenance columns.

    Raises:
        QWIAPIError: On invalid arguments or a failed request.
    """
    if dataset not in SUPPORTED_DATASETS:
        raise QWIAPIError(
            f"Unknown QWI dataset {dataset!r}; expected one of {sorted(SUPPORTED_DATASETS)}."
        )
    if not state:
        raise QWIAPIError("A two-digit state FIPS code is required (e.g. --state 13).")
    if not years:
        raise QWIAPIError("At least one year is required (e.g. --years 2019,2020).")

    quarters = quarters or [1, 2, 3, 4]
    variables = list(variables or DEFAULT_VARIABLES)
    base_url = (base_url or config.QWI_BASE_URL).rstrip("/")

    geography = f"county:{county}" if county else f"state:{state}"
    params = {
        "get": ",".join(variables),
        "for": geography,
        "in": f"state:{state}" if county else None,
        "year": ",".join(str(year) for year in years),
        "quarter": ",".join(str(quarter) for quarter in quarters),
        "industry": industry,
        "key": api_key or config.CENSUS_API_KEY,
    }

    try:
        payload = get_json(f"{base_url}/{dataset}", params)
    except APIError as error:
        raise QWIAPIError(str(error)) from error

    if not isinstance(payload, list) or len(payload) < 2:
        return pd.DataFrame()

    header, *rows = payload
    frame = pd.DataFrame(rows, columns=header)

    for column in (*variables, "year", "quarter"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "year" in frame.columns:
        frame["vintage_year"] = frame["year"].astype("Int64")
    if "quarter" in frame.columns:
        frame["month"] = frame["quarter"].map(_quarter_to_month, na_action="ignore")

    frame["dataset"] = dataset
    frame["geography"] = geography
    frame["source"] = "Census QWI"
    frame["retrieved_at"] = datetime.now(tz=timezone.utc).isoformat()
    return frame


def save_qwi_data(
    *,
    state: str,
    years: list[int],
    quarters: list[int] | None = None,
    variables: list[str] | None = None,
    dataset: str = DEFAULT_DATASET,
    county: str | None = None,
    industry: str | None = None,
    api_key: str | None = None,
    output_dir: Path | None = None,
    file_name: str | None = None,
    base_url: str | None = None,
) -> QWIDownloadResult:
    """Download QWI data and write it to CSV under ``output_dir``."""
    output_dir = Path(output_dir or config.RAW_DATA_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = fetch_qwi_data(
        state=state,
        years=years,
        quarters=quarters,
        variables=variables,
        dataset=dataset,
        county=county,
        industry=industry,
        api_key=api_key,
        base_url=base_url,
    )

    geography = f"county:{county}" if county else f"state:{state}"
    suffix = f"county{county}" if county else f"state{state}"
    output_path = output_dir / (file_name or f"qwi_{dataset}_{suffix}.csv")

    if frame.empty:
        return QWIDownloadResult(
            dataset=dataset,
            geography=geography,
            output_path=output_path,
            row_count=0,
            years=tuple(years),
        )

    frame.to_csv(output_path, index=False)
    return QWIDownloadResult(
        dataset=dataset,
        geography=geography,
        output_path=output_path,
        row_count=int(len(frame)),
        years=tuple(sorted({int(year) for year in frame["year"].dropna().tolist()})),
    )
