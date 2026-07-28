"""FRED (Federal Reserve Economic Data) API ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .. import config
from .http import APIError, get_json


class FredAPIError(RuntimeError):
    """Raised when the FRED API cannot be queried or returns an error."""


@dataclass(frozen=True, slots=True)
class FredDownloadResult:
    """Summary of a single FRED series download."""

    series_id: str
    output_path: Path
    row_count: int
    start_date: str | None
    end_date: str | None


def fetch_fred_series(
    series_id: str,
    *,
    api_key: str | None = None,
    observation_start: str | None = None,
    observation_end: str | None = None,
    base_url: str | None = None,
) -> pd.DataFrame:
    """Download one FRED series as a tidy DataFrame.

    Args:
        series_id: FRED series identifier, e.g. ``UNRATE`` or ``CAUR``.
        api_key: FRED API key; falls back to ``FRED_API_KEY``.
        observation_start: Inclusive start date as ``YYYY-MM-DD``.
        observation_end: Inclusive end date as ``YYYY-MM-DD``.
        base_url: Override for the FRED API root.

    Returns:
        DataFrame with ``date``, ``year``, ``month``, ``value``, ``series_id``,
        ``vintage_year``, and provenance columns.

    Raises:
        FredAPIError: If no API key is configured or the request fails.
    """
    if not series_id:
        raise FredAPIError("A FRED series ID is required (e.g. --series-id UNRATE).")

    api_key = api_key or config.FRED_API_KEY
    if not api_key:
        raise FredAPIError(
            "FRED_API_KEY is not set. Add it to .env or pass --api-key. "
            "Free keys: https://fredaccount.stlouisfed.org/apikeys"
        )

    base_url = (base_url or config.FRED_BASE_URL).rstrip("/")
    try:
        payload = get_json(
            f"{base_url}/series/observations",
            {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "observation_start": observation_start,
                "observation_end": observation_end,
            },
        )
    except APIError as error:
        raise FredAPIError(str(error)) from error

    observations = payload.get("observations", []) if isinstance(payload, dict) else []
    frame = pd.DataFrame(observations)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "year",
                "month",
                "value",
                "series_id",
                "vintage_year",
                "source",
                "retrieved_at",
            ]
        )

    dates = pd.to_datetime(frame["date"], errors="coerce")
    values = pd.to_numeric(frame["value"].replace(".", pd.NA), errors="coerce")

    tidy = pd.DataFrame(
        {
            "date": dates.dt.strftime("%Y-%m-%d"),
            "year": dates.dt.year.astype("Int64"),
            "month": dates.dt.month.astype("Int64"),
            "value": values,
            "series_id": series_id,
        }
    )
    tidy["vintage_year"] = tidy["year"]
    tidy["source"] = "FRED"
    tidy["retrieved_at"] = datetime.now(tz=timezone.utc).isoformat()
    return tidy.dropna(subset=["date"]).reset_index(drop=True)


def save_fred_series(
    series_id: str,
    *,
    output_dir: Path | None = None,
    api_key: str | None = None,
    observation_start: str | None = None,
    observation_end: str | None = None,
    file_name: str | None = None,
    base_url: str | None = None,
) -> FredDownloadResult:
    """Download a FRED series and write it to CSV under ``output_dir``.

    Returns:
        FredDownloadResult describing the written file. ``row_count`` is 0 when
        the series returned no observations, in which case no file is written.
    """
    output_dir = Path(output_dir or config.RAW_DATA_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = fetch_fred_series(
        series_id,
        api_key=api_key,
        observation_start=observation_start,
        observation_end=observation_end,
        base_url=base_url,
    )

    output_path = output_dir / (file_name or f"fred_{series_id.lower()}.csv")
    if frame.empty:
        return FredDownloadResult(
            series_id=series_id,
            output_path=output_path,
            row_count=0,
            start_date=None,
            end_date=None,
        )

    frame.to_csv(output_path, index=False)
    return FredDownloadResult(
        series_id=series_id,
        output_path=output_path,
        row_count=int(len(frame)),
        start_date=str(frame["date"].iloc[0]),
        end_date=str(frame["date"].iloc[-1]),
    )


def save_fred_series_batch(
    series_ids: list[str],
    *,
    output_dir: Path | None = None,
    api_key: str | None = None,
    observation_start: str | None = None,
    observation_end: str | None = None,
    base_url: str | None = None,
) -> list[FredDownloadResult]:
    """Download several FRED series, skipping (but reporting) individual failures."""
    results: list[FredDownloadResult] = []
    for series_id in series_ids:
        try:
            results.append(
                save_fred_series(
                    series_id,
                    output_dir=output_dir,
                    api_key=api_key,
                    observation_start=observation_start,
                    observation_end=observation_end,
                    base_url=base_url,
                )
            )
        except FredAPIError as error:
            print(f"Warning: FRED series {series_id} failed: {error}")
    return results
