"""Minimal JSON HTTP helper shared by the API ingestion clients."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 30.0


class APIError(RuntimeError):
    """Raised when a data provider API request fails."""


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Any:
    """Perform a GET request and decode the JSON response body.

    Args:
        url: Base URL without a query string.
        params: Query parameters; ``None`` values are dropped.
        timeout: Socket timeout in seconds.

    Returns:
        Decoded JSON payload.

    Raises:
        APIError: If the request fails or the response is not valid JSON.
    """
    query = {key: value for key, value in (params or {}).items() if value is not None}
    full_url = f"{url}?{urllib.parse.urlencode(query)}" if query else url

    try:
        with urllib.request.urlopen(full_url, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise APIError(f"HTTP {error.code} from {_redact(full_url)}: {detail}") from error
    except urllib.error.URLError as error:
        raise APIError(f"Could not reach {_redact(full_url)}: {error.reason}") from error

    try:
        return json.loads(payload)
    except json.JSONDecodeError as error:
        raise APIError(
            f"Non-JSON response from {_redact(full_url)}: {payload[:200]}"
        ) from error


def _redact(url: str) -> str:
    """Strip API keys out of a URL so they never reach logs or tracebacks."""
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    cleaned = [
        (key, "***" if key.lower() in {"api_key", "key"} else value) for key, value in query
    ]
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(cleaned)))
