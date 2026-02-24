"""
app/services/retrieval.py
──────────────────────────
Hybrid retrieval: dense vector search + BM25 keyword search,
fused via Reciprocal Rank Fusion (RRF), then cross-encoder reranked.

Why hybrid?
  Dense search (embeddings) excels at semantic/conceptual similarity.
  BM25 excels at exact keyword matches and rare terms.
  RRF fusion consistently outperforms either method alone by 5–15%.

Why rerank?
  Embedding similarity is a coarse score. Cross-encoders evaluate
  (query, chunk) jointly — much more accurate, but slower.
  We fetch top-K=20, rerank, return top-k=4 to the LLM.
"""

from __future__ import annotations

import logging
import math
from typing import Any

from langchain_core.documents import Document

from app.config import Settings

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Performs hybrid retrieval + reranking.
    Thread-safe: embedding models and Qdrant client are read-only at query time.
    """

    def __init__(self, settings: Settings, vector_store: Any) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._reranker = self._init_reranker()

    # ── Public API ──────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Full hybrid retrieval pipeline:
          1. Dense vector search (semantic)
          2. BM25 keyword search (over cached corpus)
          3. Reciprocal Rank Fusion
          4. Cross-encoder reranking
          5. Return top final_k documents
        """
        fetch_k = top_k or self._settings.retrieval_top_k
        final_k = self._settings.retrieval_final_k

        logger.debug("Starting hybrid retrieval", extra={"query": query[:80], "fetch_k": fetch_k})

        # Step 1: Dense retrieval
        dense_results = self._dense_search(query, fetch_k, filters)

        # Step 2: BM25 retrieval
        bm25_results = self._bm25_search(query, fetch_k)

        # Step 3: RRF fusion
        fused = self._reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            k=60,  # RRF constant — higher k = less aggressive fusion
        )

        # Step 4: Rerank (if enabled)
        if self._settings.reranker_enabled and self._reranker and len(fused) > final_k:
            fused = self._rerank(query, fused, final_k)
        else:
            fused = fused[:final_k]

        logger.debug(
            "Retrieval complete",
            extra={"returned": len(fused), "reranked": self._settings.reranker_enabled},
        )
        return fused

    # ── Dense search ────────────────────────────────────────────────────────

    def _dense_search(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """Qdrant ANN search using dense embeddings."""
        try:
            qdrant_filter = self._build_qdrant_filter(filters) if filters else None
            results = self._vector_store.similarity_search_with_score(
                query,
                k=k,
                filter=qdrant_filter,
            )
            logger.debug("Dense search returned", extra={"count": len(results)})
            return results  # List of (Document, score)
        except Exception as exc:
            logger.error("Dense search failed", extra={"error": str(exc)})
            return []

    # ── BM25 search ─────────────────────────────────────────────────────────

    def _bm25_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """
        BM25 keyword retrieval.
        We fetch all stored documents from Qdrant for BM25 indexing.
        For large corpora (>100k chunks), consider a dedicated BM25 store
        like Elasticsearch or a pre-built BM25 index loaded at startup.
        """
        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            # Fetch a representative sample for BM25
            # In production: maintain a BM25 index built during ingestion
            corpus_docs = self._vector_store.similarity_search(query, k=min(k * 3, 60))
            if not corpus_docs:
                return []

            tokenized_corpus = [doc.page_content.lower().split() for doc in corpus_docs]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.lower().split()
            scores = bm25.get_scores(tokenized_query)

            # Pair docs with BM25 scores, sort descending
            scored = sorted(zip(corpus_docs, scores.tolist()), key=lambda x: x[1], reverse=True)
            top_k = scored[:k]
            logger.debug("BM25 search returned", extra={"count": len(top_k)})
            return top_k
        except Exception as exc:
            logger.warning("BM25 search failed — using dense only", extra={"error": str(exc)})
            return []

    # ── RRF Fusion ──────────────────────────────────────────────────────────

    def _reciprocal_rank_fusion(
        self,
        dense: list[tuple[Document, float]],
        bm25: list[tuple[Document, float]],
        k: int = 60,
    ) -> list[Document]:
        """
        Reciprocal Rank Fusion: score = Σ (weight / (k + rank))
        Deduplicates by chunk_id, combines scores from both lists.
        Higher weight for dense (semantic) than BM25 (keyword) by default.
        """
        scores: dict[str, float] = {}
        docs_by_id: dict[str, Document] = {}

        def _fuse(ranked_list: list[tuple[Document, float]], weight: float) -> None:
            for rank, (doc, _score) in enumerate(ranked_list, start=1):
                chunk_id = doc.metadata.get("chunk_id", doc.page_content[:40])
                if chunk_id not in docs_by_id:
                    docs_by_id[chunk_id] = doc
                scores[chunk_id] = scores.get(chunk_id, 0.0) + weight * (1.0 / (k + rank))

        _fuse(dense, self._settings.retrieval_dense_weight)
        _fuse(bm25, self._settings.retrieval_bm25_weight)

        # Sort by RRF score descending
        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
        fused_docs = [docs_by_id[cid] for cid in sorted_ids]

        # Attach RRF score to metadata for debugging / observability
        for cid, doc in zip(sorted_ids, fused_docs):
            doc.metadata["rrf_score"] = round(scores[cid], 6)

        logger.debug("RRF fusion produced", extra={"count": len(fused_docs)})
        return fused_docs

    # ── Reranker ────────────────────────────────────────────────────────────

    def _init_reranker(self) -> Any | None:
        if not self._settings.reranker_enabled:
            return None
        try:
            from FlagEmbedding import FlagReranker  # type: ignore

            reranker = FlagReranker(
                self._settings.reranker_model_name,
                use_fp16=True,  # faster, minimal quality impact
            )
            logger.info("Reranker loaded", extra={"model": self._settings.reranker_model_name})
            return reranker
        except Exception as exc:
            logger.warning(
                "Reranker init failed — proceeding without reranking",
                extra={"error": str(exc)},
            )
            return None

    def _rerank(self, query: str, docs: list[Document], top_n: int) -> list[Document]:
        """
        Cross-encoder reranking: scores each (query, chunk) pair jointly.
        Returns top_n documents sorted by reranker score.
        """
        try:
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self._reranker.compute_score(pairs, normalize=True)

            # Attach reranker score to metadata
            scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            top_docs = []
            for doc, score in scored[:top_n]:
                doc.metadata["reranker_score"] = round(float(score), 4)
                top_docs.append(doc)

            logger.debug("Reranking complete", extra={"before": len(docs), "after": top_n})
            return top_docs
        except Exception as exc:
            logger.warning("Reranking failed — using RRF order", extra={"error": str(exc)})
            return docs[:top_n]

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_qdrant_filter(filters: dict[str, Any]) -> Any:
        """Convert a simple metadata filter dict to Qdrant Filter object."""
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore

            conditions = [
                FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            return Filter(must=conditions)
        except Exception:
            return None
