"""
Cross-encoder reranker for retrieved documents.
Uses sentence-transformers cross-encoder for high-quality reranking.
Falls back to BM25-based scoring if cross-encoder model is unavailable.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from loguru import logger

from config.settings import settings


class Reranker:
    """
    Cross-encoder reranker using sentence-transformers.
    Reranks retrieval results by computing query-document relevance scores.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self.model_name = model_name or settings.reranker.model_name
        self.max_length = max_length or settings.reranker.max_length
        self.batch_size = batch_size or settings.reranker.batch_size

        self._cross_encoder = None
        self._model_available = False

        self._load_model()

    def _load_model(self) -> None:
        """Load the cross-encoder model, gracefully handling unavailability."""
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
            )
            self._model_available = True
            logger.info(f"Cross-encoder reranker loaded: {self.model_name}")
        except Exception as e:
            logger.warning(
                f"Cross-encoder model '{self.model_name}' not available: {e}. "
                "Using BM25 score fallback."
            )
            self._model_available = False

    def rerank(
        self,
        query: str,
        documents: list[Any],
        top_k: Optional[int] = None,
    ) -> list[Any]:
        """
        Rerank documents by query-document relevance.

        Args:
            query: The search query string.
            documents: List of Document objects (with .text and .score attributes)
                       or dicts (with "text" and "score" keys).
            top_k: Number of top documents to return. If None, returns all reranked.

        Returns:
            Reranked documents sorted by relevance score, limited to top_k.
        """
        if not documents:
            return []

        if top_k is None:
            top_k = len(documents)
        top_k = min(top_k, len(documents))

        if self._model_available:
            return self._cross_encoder_rerank(query, documents, top_k)
        else:
            return self._bm25_fallback_rerank(query, documents, top_k)

    def _cross_encoder_rerank(
        self, query: str, documents: list[Any], top_k: int
    ) -> list[Any]:
        """Rerank using cross-encoder model."""
        try:
            texts = [self._get_doc_text(doc) for doc in documents]
            pairs = [[query, text] for text in texts]

            # Score in batches
            scores = []
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                batch_scores = self._cross_encoder.predict(
                    batch, show_progress_bar=False
                )
                scores.extend(batch_scores.tolist())

            # Apply scores to documents
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            result = []
            for doc, score in scored_docs[:top_k]:
                doc_copy = self._set_score(doc, float(score))
                result.append(doc_copy)

            logger.debug(
                f"Cross-encoder reranked {len(documents)} -> {top_k} documents. "
                f"Top score: {scores[0]:.4f} if scores else 0"
            )
            return result

        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}. Using fallback.")
            return self._bm25_fallback_rerank(query, documents, top_k)

    def _bm25_fallback_rerank(
        self, query: str, documents: list[Any], top_k: int
    ) -> list[Any]:
        """
        Fallback reranking using simple TF-IDF-like scoring.
        Used when the cross-encoder model is unavailable.
        """
        query_tokens = set(self._simple_tokenize(query))
        if not query_tokens:
            return documents[:top_k]

        scored_docs = []
        for doc in documents:
            text = self._get_doc_text(doc)
            doc_tokens = self._simple_tokenize(text)

            # TF matching score
            tf_score = sum(
                1.0 for token in doc_tokens if token in query_tokens
            ) / max(len(doc_tokens), 1)

            # Token overlap ratio
            doc_token_set = set(doc_tokens)
            overlap = len(query_tokens & doc_token_set) / max(len(query_tokens), 1)

            # Combine with original BM25/RRF score
            original_score = self._get_score(doc)
            combined_score = (
                0.4 * tf_score
                + 0.4 * overlap
                + 0.2 * self._normalize_score(original_score)
            )
            scored_docs.append((doc, combined_score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        result = []
        for doc, score in scored_docs[:top_k]:
            doc_copy = self._set_score(doc, score)
            result.append(doc_copy)

        logger.debug(f"BM25 fallback reranked {len(documents)} -> {top_k} documents")
        return result

    @staticmethod
    def _simple_tokenize(text: str) -> list[str]:
        """Simple whitespace + character tokenization for fallback."""
        import re
        # Split on whitespace and punctuation
        tokens = re.findall(r"[\w一-鿿]+", text.lower())
        return tokens

    @staticmethod
    def _get_doc_text(doc: Any) -> str:
        """Extract text from document (object or dict)."""
        if isinstance(doc, dict):
            return doc.get("text", doc.get("content", ""))
        return getattr(doc, "text", getattr(doc, "content", str(doc)))

    @staticmethod
    def _get_score(doc: Any) -> float:
        """Get current score from document."""
        if isinstance(doc, dict):
            return float(doc.get("score", 0.0))
        return float(getattr(doc, "score", 0.0))

    @staticmethod
    def _set_score(doc: Any, score: float) -> Any:
        """Set score on document (modifies in place, returns doc)."""
        if isinstance(doc, dict):
            doc["score"] = score
        else:
            doc.score = score
        return doc

    @staticmethod
    def _normalize_score(score: float, max_val: float = 10.0) -> float:
        """Normalize a score to [0, 1] range using sigmoid-like mapping."""
        if score <= 0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-score / max_val * 6 + 3))

    def batch_rerank(
        self,
        queries: list[str],
        document_sets: list[list[Any]],
        top_k: int = 5,
    ) -> list[list[Any]]:
        """
        Batch rerank multiple query-document set pairs.

        Args:
            queries: List of query strings.
            document_sets: Parallel list of document sets for each query.
            top_k: Top K results per query.

        Returns:
            List of reranked document lists.
        """
        results = []
        for query, docs in zip(queries, document_sets):
            reranked = self.rerank(query, docs, top_k)
            results.append(reranked)
        return results

    @property
    def is_model_available(self) -> bool:
        return self._model_available
