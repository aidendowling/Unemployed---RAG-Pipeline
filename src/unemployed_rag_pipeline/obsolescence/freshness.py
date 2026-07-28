import re
from datetime import datetime, timezone
from typing import Any, Dict


def _extract_years_from_query(query: str) -> tuple[int | None, int | None]:
    """Extract implied time range from a query text.
    
    Looks for patterns like:
    - "2020" or "in 2020" -> (2020, 2020)
    - "2015-2020" or "2015 to 2020" -> (2015, 2020)
    - "since 2015" or "after 2015" -> (2015, current_year)
    - "before 2020" or "until 2020" -> (None, 2020)
    
    Args:
        query: Query text to analyze
        
    Returns:
        Tuple of (start_year, end_year) or (None, None) if no years detected
    """
    current_year = datetime.now().year
    query_lower = query.lower()
    
    # Pattern 1: "YEAR - YEAR" or "YEAR to YEAR" (range)
    range_match = re.search(r'(\d{4})\s*(?:to|-|through)\s*(\d{4})', query_lower)
    if range_match:
        return (int(range_match.group(1)), int(range_match.group(2)))
    
    # Pattern 2: "since YEAR" or "after YEAR" (open-ended from)
    since_match = re.search(r'(?:since|after|from)\s+(\d{4})', query_lower)
    if since_match:
        return (int(since_match.group(1)), current_year)
    
    # Pattern 3: "before YEAR" or "until YEAR" (open-ended to)
    before_match = re.search(r'(?:before|until|by)\s+(\d{4})', query_lower)
    if before_match:
        return (1900, int(before_match.group(1)))
    
    # Pattern 4: "in YEAR" or just a standalone YEAR
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', query_lower)
    if year_match:
        year = int(year_match.group(1))
        return (year, year)
    
    return (None, None)


def query_intent_vintage_score(
    metadata: Dict[str, Any] | None,
    query: str,
    *,
    default_window_years: int = 5,
    current_year: int | None = None,
) -> float:
    """Score a document's relevance based on query-intent and vintage year alignment.
    
    If query implies a time range (e.g. "unemployment in 2020"), documents
    with vintage_year within or near that range score higher (1.0).
    Documents far from the implied range score lower, approaching 0.
    If no time intent is detected in query, returns 1.0 (fully relevant).
    
    Args:
        metadata: Document metadata (should contain 'vintage_year' if applicable)
        query: User query text
        default_window_years: If query specifies a single year, expand window by this many years
        current_year: Override current year (for testing); defaults to actual current year
        
    Returns:
        Score from 0.0 (document vintage far from query intent) to 1.0 (well-aligned)
    """
    if metadata is None or "vintage_year" not in metadata:
        # No vintage data available; assume relevant
        return 1.0
    
    current_year = current_year or datetime.now().year
    
    # Extract query-implied time range
    start_year, end_year = _extract_years_from_query(query)
    
    # If no time intent detected, document is fully relevant
    if start_year is None and end_year is None:
        return 1.0
    
    # Ensure we have a valid range
    if start_year is None:
        start_year = 1900
    if end_year is None:
        end_year = current_year
    
    # Expand single-year intent into a window
    if start_year == end_year:
        start_year = max(1900, start_year - default_window_years)
        end_year = min(current_year, end_year + default_window_years)
    
    try:
        doc_vintage = int(metadata.get("vintage_year"))
    except (ValueError, TypeError):
        return 1.0
    
    # Perfect match: within query-implied range
    if start_year <= doc_vintage <= end_year:
        return 1.0
    
    # Out of range: compute decay based on distance
    # Distance to nearest year in the range
    if doc_vintage < start_year:
        distance = start_year - doc_vintage
    else:
        distance = doc_vintage - end_year
    
    # Exponential decay: each year outside range reduces score
    score = 0.5 ** (distance / 10.0)  # Half-life of 10 years
    return float(max(0.0, min(1.0, score)))


class FreshnessScorer:
    """Simple freshness scoring using exponential decay based on document updated timestamp.

    The score is 0..1 where 1 is perfectly fresh. Uses half-life in days to compute decay.
    """

    def __init__(self, days_half_life: float = 180.0) -> None:
        self.days_half_life = float(days_half_life)

    def score(self, doc: Dict[str, Any]) -> float:
        updated = doc.get("updated_at")
        if not updated:
            return 1.0
        try:
            dt = datetime.fromisoformat(updated)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
            # Exponential decay: 0.5 ** (age / half_life)
            score = 0.5 ** (age_days / self.days_half_life)
            return float(max(0.0, min(1.0, score)))
        except Exception:
            return 0.0


def _parse_date(value: Any):
    from datetime import datetime, timezone

    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        # numeric epoch
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
            except Exception:
                if value.endswith("Z"):
                    try:
                        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    except Exception:
                        return None
                else:
                    return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
    except Exception:
        return None


def is_obsolete(metadata: Dict[str, Any] | None, *, threshold_days: int = 180) -> bool:
    """Return True if metadata indicates the document is obsolete.

    Checks `deprecated` flag or age > threshold_days from `updated_at`, `last_modified`, or `timestamp`.
    """
    if metadata is None:
        return False
    if metadata.get("deprecated"):
        return True
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc)
    for key in ("updated_at", "last_modified", "timestamp"):
        val = metadata.get(key)
        dt = _parse_date(val)
        if dt:
            age_days = (now - dt).total_seconds() / 86400.0
            return age_days > float(threshold_days)
    return False


def freshness_score(metadata: Dict[str, Any] | None, *, now: Any = None, max_days: int = 365) -> float:
    """Return a freshness score in [0.0, 1.0]. Newer documents -> score closer to 1.

    If no date is available, return 0.0.
    """
    if metadata is None:
        return 0.0
    from datetime import datetime, timezone

    now = now or datetime.now(tz=timezone.utc)
    for key in ("updated_at", "last_modified", "timestamp"):
        val = metadata.get(key)
        dt = _parse_date(val)
        if dt:
            delta_days = (now - dt).total_seconds() / 86400.0
            normalized = max(0.0, min(1.0, 1.0 - (delta_days / float(max_days))))
            return float(normalized)
    return 0.0
    
