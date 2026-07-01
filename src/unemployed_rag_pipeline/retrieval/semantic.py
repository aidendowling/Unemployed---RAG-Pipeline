"""Vector-based semantic search retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb


class SemanticRetriever:
    """Semantic search using embeddings stored in Chroma."""

    def __init__(self, index_dir: Path, collection_name: str = "labor_market") -> None:
        """Initialize semantic retriever from a Chroma index.

        Args:
            index_dir: Directory containing the Chroma index
            collection_name: Name of the Chroma collection to use
        """
        self.index_dir = index_dir
        self.collection_name = collection_name
        self._load_collection()

    def _load_collection(self) -> None:
        """Load the Chroma collection."""
        chroma_dir = self.index_dir / "chroma"
        client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Search documents using semantic similarity.

        Args:
            query: Search query string
            k: Number of top results to return

        Returns:
            List of dicts with 'id', 'text', 'score', and 'metadata' keys
        """
        results = self.collection.query(query_texts=[query], n_results=k)

        # Format results consistently with BM25Retriever
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for doc_id, distance, metadata, text in zip(
                results["ids"][0],
                results["distances"][0],
                results["metadatas"][0],
                results["documents"][0],
            ):
                # Convert distance to similarity (closer = higher similarity)
                similarity_score = 1 - distance
                formatted_results.append(
                    {
                        "id": doc_id,
                        "text": text,
                        "score": float(similarity_score),
                        "metadata": metadata,
                        "table_name": metadata.get("table_name", "unknown"),
                    }
                )
        return formatted_results
