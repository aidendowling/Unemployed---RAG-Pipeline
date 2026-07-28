"""CPS (Current Population Survey) fixed-width ``.dat`` ingestion.

CPS microdata ships as a fixed-width text file plus a separate layout, so the
column positions have to come from somewhere. This module resolves a layout in
the following order:

1. An explicit layout passed by the caller (``--layout``).
2. A sidecar layout next to the ``.dat`` file: an IPUMS DDI ``.xml``, a JSON
   field list, or a Census/BLS data-dictionary ``.txt``/``.dct``.
3. The bundled CPS basic-monthly layout, used only if the parsed record passes
   a sanity check.
"""

from __future__ import annotations

import gzip
import json
import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd

BUILTIN_LAYOUT_DIR = Path(__file__).parent / "layouts"
BUILTIN_LAYOUTS = {"cps-basic-monthly": BUILTIN_LAYOUT_DIR / "cps_basic_monthly.json"}
SIDECAR_SUFFIXES = (".xml", ".json", ".dct", ".txt")

WEIGHT_IMPLIED_DECIMALS = 4

LABOR_FORCE_STATUS = {
    1: "Employed - at work",
    2: "Employed - absent",
    3: "Unemployed - on layoff",
    4: "Unemployed - looking",
    5: "Not in labor force - retired",
    6: "Not in labor force - disabled",
    7: "Not in labor force - other",
}

SEX_LABELS = {1: "Male", 2: "Female"}

STATE_FIPS = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE",
    11: "DC", 12: "FL", 13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN",
    19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV",
    33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH",
    40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN",
    48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI",
    56: "WY",
}

YEAR_FIELDS = ("HRYEAR4", "YEAR", "HRYEAR", "SURVEY_YEAR")
MONTH_FIELDS = ("HRMONTH", "MONTH")


class CPSLayoutError(RuntimeError):
    """Raised when a CPS fixed-width layout is missing, unreadable, or wrong."""


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One fixed-width column in a CPS record (1-based, inclusive positions)."""

    name: str
    start: int
    end: int
    dtype: str = "int"
    description: str = ""

    @property
    def slice(self) -> tuple[int, int]:
        """Return 0-based half-open ``(start, end)`` bounds for pandas."""
        return (self.start - 1, self.end)


@dataclass(frozen=True, slots=True)
class CPSLayout:
    """A named collection of :class:`FieldSpec` entries."""

    name: str
    fields: tuple[FieldSpec, ...]
    source: str

    @property
    def record_length(self) -> int:
        """Return the last column position covered by the layout."""
        return max(field.end for field in self.fields)


def load_layout(layout_path: Path | str) -> CPSLayout:
    """Load a layout from a builtin name, JSON file, IPUMS DDI, or data dictionary.

    Raises:
        CPSLayoutError: If the layout cannot be found or contains no fields.
    """
    if isinstance(layout_path, str) and layout_path in BUILTIN_LAYOUTS:
        return _load_json_layout(BUILTIN_LAYOUTS[layout_path])

    path = Path(layout_path).expanduser()
    if not path.exists():
        raise CPSLayoutError(
            f"Layout {layout_path!r} not found. Pass a path to an IPUMS .xml DDI, a JSON "
            f"field list, a CPS data dictionary .txt, or one of: {sorted(BUILTIN_LAYOUTS)}."
        )

    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_json_layout(path)
    if suffix == ".xml":
        return _load_ddi_layout(path)
    if suffix in {".txt", ".dct"}:
        return _load_data_dictionary_layout(path)

    raise CPSLayoutError(f"Unsupported layout file type: {path.suffix}")


def _load_json_layout(path: Path) -> CPSLayout:
    """Parse a JSON layout: either a bare list of fields or ``{"fields": [...]}``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_fields = payload["fields"] if isinstance(payload, dict) else payload
    fields = tuple(
        FieldSpec(
            name=str(item["name"]).upper(),
            start=int(item["start"]),
            end=int(item["end"]),
            dtype=str(item.get("dtype", "int")),
            description=str(item.get("description", "")),
        )
        for item in raw_fields
    )
    if not fields:
        raise CPSLayoutError(f"Layout {path} contains no fields.")

    name = payload.get("name", path.stem) if isinstance(payload, dict) else path.stem
    return CPSLayout(name=str(name), fields=fields, source=str(path))


def _load_ddi_layout(path: Path) -> CPSLayout:
    """Parse an IPUMS DDI codebook (``.xml``) into a layout."""
    root = ElementTree.parse(path).getroot()
    namespace = {"ddi": "ddi:codebook:2_5"}

    variables = root.findall(".//ddi:var", namespace) or root.findall(".//var")
    fields: list[FieldSpec] = []
    for variable in variables:
        location = variable.find("ddi:location", namespace)
        if location is None:
            location = variable.find("location")
        if location is None:
            continue

        start = location.get("StartPos")
        end = location.get("EndPos")
        if not start or not end:
            continue

        label = variable.find("ddi:labl", namespace)
        if label is None:
            label = variable.find("labl")

        is_decimal = (variable.get("dcml") or "0") != "0"
        fields.append(
            FieldSpec(
                name=str(variable.get("name", variable.get("ID", ""))).upper(),
                start=int(start),
                end=int(end),
                dtype="float" if is_decimal else "int",
                description=(label.text or "").strip() if label is not None else "",
            )
        )

    if not fields:
        raise CPSLayoutError(f"No <var> locations found in DDI codebook {path}.")
    return CPSLayout(name=path.stem, fields=tuple(fields), source=str(path))


_DICTIONARY_LINE = re.compile(
    r"^D\s+([A-Z0-9_$]+)\s+(\d+)\s+(\d+)\s*(.*)$", re.IGNORECASE
)


def _load_data_dictionary_layout(path: Path) -> CPSLayout:
    """Parse a Census/BLS CPS data dictionary (``D NAME SIZE START`` lines)."""
    fields: list[FieldSpec] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _DICTIONARY_LINE.match(line.strip())
        if not match:
            continue
        name, size, start, description = match.groups()
        start_pos = int(start)
        fields.append(
            FieldSpec(
                name=name.upper(),
                start=start_pos,
                end=start_pos + int(size) - 1,
                dtype="int",
                description=description.strip(),
            )
        )

    if not fields:
        raise CPSLayoutError(
            f"No 'D NAME SIZE START' entries found in data dictionary {path}."
        )
    return CPSLayout(name=path.stem, fields=tuple(fields), source=str(path))


def find_sidecar_layout(dat_path: Path) -> Path | None:
    """Return a layout file sitting next to ``dat_path``, if one exists.

    Matches ``<stem>.xml``/``.json``/``.dct``/``.txt`` first, then any single
    ``.xml`` DDI in the same directory (the usual shape of an IPUMS extract).
    """
    stem = dat_path.name.removesuffix(".gz").removesuffix(".dat")
    for suffix in SIDECAR_SUFFIXES:
        candidate = dat_path.parent / f"{stem}{suffix}"
        if candidate.exists():
            return candidate

    xml_files = sorted(dat_path.parent.glob("*.xml"))
    return xml_files[0] if len(xml_files) == 1 else None


def _open_text(path: Path) -> str:
    """Read a ``.dat`` or ``.dat.gz`` file as text."""
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _coerce(frame: pd.DataFrame, layout: CPSLayout) -> pd.DataFrame:
    """Apply per-field dtypes, including CPS implied-decimal weights."""
    for field in layout.fields:
        column = frame[field.name].astype("string").str.strip()
        if field.dtype == "string":
            frame[field.name] = column
            continue

        numeric = pd.to_numeric(column, errors="coerce")
        if field.dtype == "weight":
            frame[field.name] = numeric / (10**WEIGHT_IMPLIED_DECIMALS)
        elif field.dtype == "float":
            frame[field.name] = numeric
        else:
            frame[field.name] = numeric.astype("Int64")
    return frame


def _first_present(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    """Return the first column name from ``names`` present in ``frame``."""
    return next((name for name in names if name in frame.columns), None)


def _validate(frame: pd.DataFrame, layout: CPSLayout, dat_path: Path) -> None:
    """Sanity-check a parsed CPS frame so a wrong layout fails loudly.

    Raises:
        CPSLayoutError: If no plausible survey year could be parsed.
    """
    year_column = _first_present(frame, YEAR_FIELDS)
    if year_column is None:
        raise CPSLayoutError(
            f"Layout {layout.name!r} produced no survey-year field "
            f"(expected one of {', '.join(YEAR_FIELDS)}) for {dat_path.name}. "
            "Pass the layout that shipped with your extract via --layout."
        )

    years = pd.to_numeric(frame[year_column], errors="coerce").dropna()
    plausible = years.between(1960, 2100)
    if years.empty or plausible.mean() < 0.9:
        sample = years.head(5).tolist()
        raise CPSLayoutError(
            f"Layout {layout.name!r} parsed implausible {year_column} values {sample} from "
            f"{dat_path.name}. The layout almost certainly does not match this file; pass the "
            "data dictionary or IPUMS DDI via --layout."
        )


def _annotate(frame: pd.DataFrame, dat_path: Path) -> pd.DataFrame:
    """Add decoded labels, vintage metadata, and provenance columns."""
    year_column = _first_present(frame, YEAR_FIELDS)
    if year_column is not None:
        frame["year"] = pd.to_numeric(frame[year_column], errors="coerce").astype("Int64")
        frame["vintage_year"] = frame["year"]

    month_column = _first_present(frame, MONTH_FIELDS)
    if month_column is not None:
        frame["month"] = pd.to_numeric(frame[month_column], errors="coerce").astype("Int64")

    if "GESTFIPS" in frame.columns:
        frame["state"] = frame["GESTFIPS"].map(STATE_FIPS)
    if "PEMLR" in frame.columns:
        frame["labor_force_status"] = frame["PEMLR"].map(LABOR_FORCE_STATUS)
        frame["is_unemployed"] = frame["PEMLR"].isin([3, 4])
        frame["is_employed"] = frame["PEMLR"].isin([1, 2])
    if "PESEX" in frame.columns:
        frame["sex"] = frame["PESEX"].map(SEX_LABELS)

    frame["source"] = "CPS"
    frame["source_file"] = dat_path.name
    frame["retrieved_at"] = datetime.now(tz=timezone.utc).isoformat()
    return frame


def read_cps_dat(
    dat_path: Path,
    *,
    layout: Path | str | CPSLayout | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Parse a fixed-width CPS ``.dat`` (or ``.dat.gz``) file into a DataFrame.

    Args:
        dat_path: Path to the CPS microdata file.
        layout: Explicit layout, builtin layout name, or ``None`` to auto-resolve.
        max_rows: Optional cap on the number of records read.

    Returns:
        DataFrame with the layout's raw fields plus decoded labels,
        ``year``/``month``/``vintage_year``, and provenance columns.

    Raises:
        CPSLayoutError: If no usable layout resolves or the parse looks wrong.
    """
    dat_path = Path(dat_path).expanduser()
    if not dat_path.exists():
        raise CPSLayoutError(f"CPS data file not found: {dat_path}")

    if isinstance(layout, CPSLayout):
        resolved = layout
    elif layout is not None:
        resolved = load_layout(layout)
    else:
        sidecar = find_sidecar_layout(dat_path)
        if sidecar is not None:
            resolved = load_layout(sidecar)
        else:
            print(
                f"Warning: no layout found next to {dat_path.name}; falling back to the bundled "
                "'cps-basic-monthly' layout. Pass --layout if your extract uses a different one."
            )
            resolved = load_layout("cps-basic-monthly")

    text = _open_text(dat_path)
    frame = pd.read_fwf(
        StringIO(text),
        colspecs=[field.slice for field in resolved.fields],
        names=[field.name for field in resolved.fields],
        dtype=str,
        header=None,
        nrows=max_rows,
    )
    if frame.empty:
        raise CPSLayoutError(
            f"Layout {resolved.name!r} parsed no records out of {dat_path.name}. The file may be "
            "empty, or the layout's column positions may fall outside the record length."
        )

    frame = _coerce(frame, resolved)
    _validate(frame, resolved, dat_path)
    return _annotate(frame, dat_path)


WEIGHT_FIELDS = ("PWCMPWGT", "PWSSWGT", "WTFINL", "ASECWT")


def summarize_cps(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse CPS person records into year/month/state labor-market aggregates.

    Person-level CPS files hold hundreds of thousands of records per month, which
    is far too granular to embed one document per row. This aggregates to the
    unit researchers actually query — a state-month unemployment rate — using the
    person weight when the layout provides one.

    Args:
        frame: Output of :func:`read_cps_dat`.

    Returns:
        One row per available year/month/state with weighted employed,
        unemployed, labor force counts and an unemployment rate.

    Raises:
        CPSLayoutError: If the frame lacks the labor-force recode needed to aggregate.
    """
    if "is_unemployed" not in frame.columns or "is_employed" not in frame.columns:
        raise CPSLayoutError(
            "Cannot aggregate CPS data without a PEMLR labor-force recode column. "
            "Ingest with --raw to load the person-level records instead."
        )

    weight_column = _first_present(frame, WEIGHT_FIELDS)
    weights = (
        pd.to_numeric(frame[weight_column], errors="coerce").fillna(0.0)
        if weight_column
        else pd.Series(1.0, index=frame.index)
    )

    working = pd.DataFrame(
        {
            "year": frame["year"] if "year" in frame.columns else pd.NA,
            "month": frame["month"] if "month" in frame.columns else pd.NA,
            "state": frame["state"] if "state" in frame.columns else "US",
            "employed": weights.where(frame["is_employed"], 0.0),
            "unemployed": weights.where(frame["is_unemployed"], 0.0),
        }
    )

    group_keys = [key for key in ("year", "month", "state") if working[key].notna().any()]
    grouped = working.groupby(group_keys, dropna=False)[["employed", "unemployed"]].sum()
    summary = grouped.reset_index()

    summary["labor_force"] = summary["employed"] + summary["unemployed"]
    summary["unemployment_rate"] = (
        (summary["unemployed"] / summary["labor_force"] * 100).round(2).where(
            summary["labor_force"] > 0
        )
    )
    if "year" in summary.columns:
        summary["vintage_year"] = summary["year"].astype("Int64")

    summary["source"] = "CPS"
    summary["weighted"] = weight_column or "unweighted"
    summary["retrieved_at"] = datetime.now(tz=timezone.utc).isoformat()
    return summary
