from typing import List, Optional, Protocol, Dict, Any, Any as _Any, Callable, Iterable


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
    """Orchestrates retrieval from one or more retrievers and generates an answer.

    Args:
        retrievers: Iterable of objects exposing `search(query, k)` -> list[dict]
        generator: Callable `(query, contexts)` -> str used to produce final answer
        obsolescence_filter: Optional callable `(result) -> bool` which returns
            True if a result should be kept (i.e. not obsolete). If omitted no
            obsolescence filtering is applied.
    """

    def __init__(
        self,
        retrievers: Iterable[_Any],
        generator: Callable[[str, Iterable[str]], str] | None = None,
        obsolescence_filter: Callable[[dict], bool] | None = None,
    ) -> None:
        self.retrievers = list(retrievers)
        self.generator = generator or DefaultGenerator()
        self.obsolescence_filter = obsolescence_filter

    def _aggregate(self, all_results: Iterable[dict], top_k: int) -> List[dict]:
        # Deduplicate by id keeping highest score
        best_by_id: dict[str, dict] = {}
        for res in all_results:
            rid = str(res.get("id"))
            if rid in best_by_id:
                if float(res.get("score", 0.0)) > float(best_by_id[rid].get("score", 0.0)):
                    best_by_id[rid] = res
            else:
                best_by_id[rid] = res

        results = list(best_by_id.values())
        # Optionally filter obsolete
        if self.obsolescence_filter:
            results = [r for r in results if self.obsolescence_filter(r)]

        # Sort by score desc and return top_k
        results.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
        return results[:top_k]

    def answer(self, query: str, k: int = 5, context_chars: int = 1500) -> dict:
        """Run retrieval and produce an answer structure.

        Returns a dict with keys: `answer` (str), `contexts` (list[str]),
        and `sources` (list[dict]) with the original result dicts included.
        """
        all_results: list[dict] = []
        for retriever in self.retrievers:
            try:
                res = retriever.search(query, k)
            except Exception:
                res = []
            all_results.extend(res)

        top_results = self._aggregate(all_results, k)

        # Build text contexts for the generator (truncate by chars to keep prompt small)
        contexts: list[str] = []
        for r in top_results:
            text = str(r.get("text", ""))
            metadata = r.get("metadata") or {}
            table = r.get("table_name") or metadata.get("table_name") or "unknown"
            context_text = f"[source={table} id={r.get('id')} score={r.get('score')}\n]{text}"
            if len(context_text) > context_chars:
                context_text = context_text[: context_chars - 3] + "..."
            contexts.append(context_text)

        answer_text = self.generator(query, contexts)

        return {"answer": answer_text, "contexts": contexts, "sources": top_results}
