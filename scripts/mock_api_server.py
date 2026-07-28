"""Local stand-in for the FRED and Census QWI APIs, for offline end-to-end runs.

Usage::

    python scripts/mock_api_server.py 8765 &
    FRED_BASE_URL=http://127.0.0.1:8765/fred \\
    QWI_BASE_URL=http://127.0.0.1:8765/qwi \\
    PYTHONPATH=src python -m unemployed_rag_pipeline.main fred --series-id UNRATE --api-key mock
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

FRED_OBSERVATIONS = {
    "observations": [
        {"date": f"{year}-{month:02d}-01", "value": f"{rate:.1f}"}
        for year, rates in {
            2018: [4.1, 3.9, 3.8, 3.7],
            2019: [4.0, 3.6, 3.7, 3.5],
            2020: [3.6, 14.8, 10.2, 6.7],
            2021: [6.4, 6.0, 5.4, 4.2],
            2022: [4.0, 3.6, 3.5, 3.6],
        }.items()
        for month, rate in zip((1, 4, 7, 10), rates)
    ]
}

QWI_ROWS = [
    ["Emp", "EmpEnd", "HirA", "Sep", "year", "quarter", "state"],
    ["4131234", "4145678", "251234", "240987", "2020", "1", "13"],
    ["3987654", "3901234", "198765", "260123", "2020", "2", "13"],
    ["4012345", "4055678", "232145", "215678", "2020", "3", "13"],
    ["4098765", "4110234", "245678", "220456", "2020", "4", "13"],
]


class MockHandler(BaseHTTPRequestHandler):
    """Serve canned FRED and QWI payloads."""

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if "/fred/series/observations" in self.path:
            payload: object = FRED_OBSERVATIONS
        elif "/qwi/" in self.path:
            payload = QWI_ROWS
        else:
            self.send_error(404, "Unknown mock endpoint")
            return

        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        """Silence per-request logging."""


def main() -> None:
    """Run the mock server until interrupted."""
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    HTTPServer(("127.0.0.1", port), MockHandler).serve_forever()


if __name__ == "__main__":
    main()
