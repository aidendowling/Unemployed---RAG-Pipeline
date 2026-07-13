from datetime import datetime, timezone
from typing import Dict, Any


class FreshnessChecker:
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
    
