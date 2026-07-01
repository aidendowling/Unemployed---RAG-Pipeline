"""SentenceTransformer-based embedding generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer as SentenceTransformerType


class TextEmbedder:
    """Generate embeddings for text documents using SentenceTransformer."""

    def __init__(self, model_name: str) -> None:
        """Initialize the embedder with a specific model.

        Args:
            model_name: HuggingFace model identifier (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
        """
        self.model_name = model_name
        self.model: SentenceTransformerType = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
