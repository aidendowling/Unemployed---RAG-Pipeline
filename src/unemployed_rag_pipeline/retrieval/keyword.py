"""BM25 keyword search retrieval."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 retrieval.

    Args:
        text: Text to tokenize

    Returns:
        List of lowercase alphanumeric tokens
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    """Keyword-based retrieval using BM25 algorithm."""

    def __init__(self, bm25_path: Path) -> None:
        """Initialize retriever from a saved BM25 corpus.

        Args:
            bm25_path: Path to the BM25 corpus JSON file
        """
        self.bm25_path = bm25_path
        self._load_corpus()

    def _load_corpus(self) -> None:
        """Load the BM25 corpus from disk."""
        corpus_data = json.loads(self.bm25_path.read_text(encoding="utf-8"))
        self.documents = corpus_data["documents"]
        tokenized_corpus = [doc["tokens"] for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search documents using BM25.

        Args:
            query: Search query string
            k: Number of top results to return

        Returns:
            List of dicts with 'id', 'text', 'score', and 'metadata' keys
        """
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "score": float(scores[idx]),
                    "metadata": doc["metadata"],
                    "table_name": doc["table_name"],
                }
            )
        return results
