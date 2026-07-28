"""RAG orchestration: retrieval fusion, obsolescence weighting, and generation."""

from typing import Any as _Any
from typing import Callable, Dict, Iterable, List, Optional, Protocol

from .obsolescence.freshness import query_intent_vintage_score


class GenerationError(RuntimeError):
    """Raised when an LLM generator fails and the caller should fall back."""


class Embedder(Protocol):
    def embed(self, text: str) -> List[float]: ...


class Retriever(Protocol):
    def retrieve(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, _Any]]: ...


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


class RAGPipeline:
    """Minimal RAG orchestrator.

    - embedder: provides `embed(text) -> List[float]`
    - retriever: provides `retrieve(embedding, top_k) -> List[dict]` (dicts include `id`, `content`, optional `updated_at`)
    - generator: provides `generate(prompt) -> str`
    - freshness_checker: optional object with `score(doc) -> float` (0..1)
    """

    def __init__(
        self,
        embedder: Embedder,
        retriever: Retriever,
        generator: Generator,
        freshness_checker: Optional[_Any] = None,
    ) -> None:
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.freshness_checker = freshness_checker

    def answer(self, query: str, top_k: int = 5) -> Dict[str, _Any]:
        emb = self.embedder.embed(query)
        docs = self.retriever.retrieve(emb, top_k=top_k)

        if self.freshness_checker is not None:
            scored: List[Dict[str, _Any]] = []
            for d in docs:
                score = float(self.freshness_checker.score(d) or 0.0)
                d["_freshness"] = score
                if score > 0.0:
                    scored.append(d)
            docs = sorted(scored, key=lambda x: x.get("_freshness", 0.0), reverse=True)

        contexts = "\n\n".join(
            f"Source: {d.get('id', '?')}\n{d.get('content', '')}" for d in docs
        )

        prompt = (
            "Use the following documents to answer the question. Provide a concise answer and cite sources.\n\n"
            f"{contexts}\n\nQuestion: {query}\nAnswer:"
        )

        answer = self.generator.generate(prompt)
        return {"answer": answer, "sources": [d.get("id") for d in docs], "docs": docs}


class OpenRouterGenerator:
    """LLM-based generator using OpenRouter API (OpenAI-compatible).
    
    Calls a remote LLM via OpenRouter to generate answers from contexts.
    Requires OPENROUTER_API_KEY to be set in environment.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/llama-3.1-8b-instruct:free",
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.7,
    ) -> None:
        """
        Args:
            api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var.
            model: Model identifier on OpenRouter.
            base_url: OpenRouter API base URL.
            temperature: Sampling temperature (0.0-1.0).
        """
        from openai import OpenAI
        
        if api_key is None:
            import os
            api_key = os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Set it as environment variable or pass as argument."
            )
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def __call__(self, query: str, contexts: Iterable[str]) -> str:
        """Generate answer from query and retrieved contexts.
        
        Args:
            query: User query
            contexts: Iterable of retrieved document contexts
            
        Returns:
            Generated answer string
        """
        contexts_list = list(contexts)
        if not contexts_list:
            return f"No retrieved context available. Query: {query}"

        contexts_text = "\n\n---\n\n".join(contexts_list)
        
        system_prompt = (
            "You are a helpful assistant that answers questions about U.S. labor market data. "
            "Use the provided documents to answer the question accurately. "
            "If the documents don't contain relevant information, say so clearly. "
            "Always cite your sources from the documents."
        )
        
        user_prompt = (
            f"Documents:\n\n{contexts_text}\n\n"
            f"Question: {query}\n\n"
            "Please provide a concise answer based on the documents above."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=1024,
            )
            return response.choices[0].message.content or "No response generated."
        except Exception as e:
            raise GenerationError(f"OpenRouter generation failed: {e}") from e


class DefaultGenerator:
    """Simple generator that formats retrieved contexts into a best-effort answer.

    This is intentionally minimal and does not call any external LLMs. It exists
    so the RAG orchestrator can be used without API keys while remaining
    replaceable by a real generator implementation (callable accepting
    `(query, contexts)` and returning a string).
    """

    def __call__(self, query: str, contexts: Iterable[str]) -> str:
        contexts_list = list(contexts)
        if not contexts_list:
            return f"No retrieved context available. Query: {query}"

        joined = "\n\n---\n\n".join(contexts_list)
        return f"Retrieved contexts:\n\n{joined}\n\nQuery: {query}\n\nAnswer: Based on the retrieved contexts above."


class RAGOrchestrator:
    """Orchestrates retrieval, obsolescence weighting, and answer generation.

    Args:
        retrievers: Iterable of objects exposing `search(query, k)` -> list[dict].
            Pass a single :class:`~.retrieval.hybrid.HybridRetriever` to get
            fused BM25 + semantic ranking with per-retriever provenance.
        generator: Callable `(query, contexts)` -> str used to produce the final
            answer. Failures fall back to :class:`DefaultGenerator` and add a warning.
        obsolescence_filter: Optional callable `(result) -> bool` returning True
            if a result should be kept. If omitted no hard filtering is applied.
        vintage_scoring: When True, every result is scored with
            :func:`query_intent_vintage_score` against the query's implied time
            range, and ranked by `retrieval score x vintage score`.
        min_vintage_score: Results scoring below this are dropped (only applies
            when `vintage_scoring` is enabled).
        warn_vintage_score: Results below this threshold are reported in the
            response `warnings` as temporally misaligned with the query.
    """

    def __init__(
        self,
        retrievers: Iterable[_Any],
        generator: Callable[[str, Iterable[str]], str] | None = None,
        obsolescence_filter: Callable[[dict], bool] | None = None,
        *,
        vintage_scoring: bool = False,
        min_vintage_score: float = 0.0,
        warn_vintage_score: float = 0.6,
    ) -> None:
        self.retrievers = list(retrievers)
        self.generator = generator or DefaultGenerator()
        self.obsolescence_filter = obsolescence_filter
        self.vintage_scoring = vintage_scoring
        self.min_vintage_score = min_vintage_score
        self.warn_vintage_score = warn_vintage_score

    def _aggregate(self, all_results: Iterable[dict], top_k: int, query: str) -> List[dict]:
        """Deduplicate by id, apply obsolescence, and rank by combined score."""
        best_by_id: dict[str, dict] = {}
        for res in all_results:
            rid = str(res.get("id"))
            if rid in best_by_id:
                if float(res.get("score", 0.0)) > float(best_by_id[rid].get("score", 0.0)):
                    best_by_id[rid] = res
            else:
                best_by_id[rid] = res

        results = list(best_by_id.values())
        if self.obsolescence_filter:
            results = [r for r in results if self.obsolescence_filter(r)]

        for result in results:
            retrieval_score = float(result.get("score", 0.0))
            if self.vintage_scoring:
                vintage_score = query_intent_vintage_score(result.get("metadata"), query)
            else:
                vintage_score = 1.0
            result["retrieval_score"] = retrieval_score
            result["vintage_score"] = vintage_score
            result["combined_score"] = retrieval_score * vintage_score

        if self.vintage_scoring:
            results = [
                r for r in results if float(r["vintage_score"]) >= self.min_vintage_score
            ]

        results.sort(key=lambda r: float(r.get("combined_score", 0.0)), reverse=True)
        return results[:top_k]

    @staticmethod
    def _provenance(result: dict) -> dict:
        """Extract the display-ready provenance fields for one result."""
        metadata = result.get("metadata") or {}
        return {
            "id": result.get("id"),
            "table": result.get("table_name") or metadata.get("table_name") or "unknown",
            "source_file": metadata.get("source_file"),
            "dataset": metadata.get("source"),
            "vintage_year": metadata.get("vintage_year"),
            "retrievers": result.get("retrievers", []),
            "component_scores": result.get("component_scores", {}),
            "retrieval_score": result.get("retrieval_score", result.get("score", 0.0)),
            "vintage_score": result.get("vintage_score", 1.0),
            "combined_score": result.get("combined_score", result.get("score", 0.0)),
            "fusion_method": result.get("fusion_method"),
        }

    def _collect_warnings(self, results: List[dict], retriever_errors: dict[str, str]) -> List[str]:
        """Build the user-facing warning list for one answered query."""
        warnings: list[str] = []
        for name, message in retriever_errors.items():
            warnings.append(f"Retriever '{name}' failed and returned no results: {message}")

        if not results:
            warnings.append("No documents were retrieved; the answer is not grounded in data.")
            return warnings

        missing_vintage = sum(
            1 for r in results if (r.get("metadata") or {}).get("vintage_year") is None
        )
        if missing_vintage:
            warnings.append(
                f"{missing_vintage} of {len(results)} retrieved document(s) have no vintage_year; "
                "obsolescence scoring treated them as temporally relevant."
            )

        if self.vintage_scoring:
            stale = [r for r in results if float(r.get("vintage_score", 1.0)) < self.warn_vintage_score]
            if stale:
                years = sorted(
                    {
                        str((r.get("metadata") or {}).get("vintage_year"))
                        for r in stale
                        if (r.get("metadata") or {}).get("vintage_year") is not None
                    }
                )
                detail = f" (vintages: {', '.join(years)})" if years else ""
                warnings.append(
                    f"{len(stale)} retrieved document(s) are poorly aligned with the query's "
                    f"time range{detail}; answer may be based on obsolete data."
                )
        return warnings

    def answer(self, query: str, k: int = 5, context_chars: int = 1500) -> dict:
        """Run retrieval and produce an answer structure.

        Returns:
            Dict with `answer`, `contexts`, `sources` (raw result dicts),
            `provenance` (per-source attribution) and `warnings`.
        """
        all_results: list[dict] = []
        retriever_errors: dict[str, str] = {}
        for retriever in self.retrievers:
            try:
                res = retriever.search(query, k)
            except Exception as error:  # noqa: BLE001 - one bad retriever must not kill the query
                retriever_errors[type(retriever).__name__] = str(error)
                res = []
            all_results.extend(res)
            retriever_errors.update(getattr(retriever, "errors", {}) or {})

        top_results = self._aggregate(all_results, k, query)

        # Build text contexts for the generator (truncate by chars to keep prompt small)
        contexts: list[str] = []
        for r in top_results:
            text = str(r.get("text", ""))
            metadata = r.get("metadata") or {}
            table = r.get("table_name") or metadata.get("table_name") or "unknown"
            vintage = metadata.get("vintage_year", "unknown")
            context_text = (
                f"[source={table} id={r.get('id')} vintage_year={vintage} "
                f"score={round(float(r.get('combined_score', r.get('score', 0.0))), 4)}\n]{text}"
            )
            if len(context_text) > context_chars:
                context_text = context_text[: context_chars - 3] + "..."
            contexts.append(context_text)

        warnings = self._collect_warnings(top_results, retriever_errors)

        try:
            answer_text = self.generator(query, contexts)
        except GenerationError as error:
            warnings.append(f"{error} Falling back to the non-LLM context formatter.")
            answer_text = DefaultGenerator()(query, contexts)

        return {
            "answer": answer_text,
            "contexts": contexts,
            "sources": top_results,
            "provenance": [self._provenance(r) for r in top_results],
            "warnings": warnings,
        }
