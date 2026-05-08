"""
Hybrid retriever combining Milvus dense+sparse search, rule filtering, and reranking.
Full pipeline: encode → hybrid search → rule filter → rerank → return.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from loguru import logger

from config.settings import settings
from retrieval.milvus_client import MilvusClient, Document
from retrieval.bm25_retriever import BM25Retriever
from retrieval.rule_filter import RuleFilter, FilterContext
from retrieval.reranker import Reranker


class EmbeddingModel:
    """
    Sentence-transformer embedding model wrapper.
    Produces dense vectors for query encoding.
    """

    def __init__(self, model_name: Optional[str] = None, dim: Optional[int] = None):
        self.model_name = model_name or settings.embedding.model_name
        self.dim = dim or settings.embedding.dim
        self._model = None
        self._available = False
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name,
                device=settings.embedding.device,
            )
            self._available = True
            logger.info(f"Embedding model loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"Embedding model not available: {e}. Using random vectors.")
            self._available = False

    def encode(
        self,
        texts: list[str] | str,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts to dense vectors.

        Args:
            texts: Single text or list of texts.
            normalize: Whether to L2-normalize embeddings.

        Returns:
            np.ndarray of shape (n_texts, dim) or (dim,) for single text.
        """
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        if self._available and self._model:
            try:
                embeddings = self._model.encode(
                    texts,
                    normalize_embeddings=normalize,
                    batch_size=settings.embedding.batch_size,
                    show_progress_bar=False,
                )
                return embeddings[0] if single else embeddings
            except Exception as e:
                logger.warning(f"Embedding encode failed: {e}. Using random fallback.")

        # Fallback: deterministic pseudo-random vectors based on text hash
        rng = np.random.default_rng(hash(texts[0]) % (2**32))
        vectors = rng.standard_normal((len(texts), self.dim)).astype(np.float32)
        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-10)
        return vectors[0] if single else vectors

    @property
    def is_available(self) -> bool:
        return self._available


class HybridRetriever:
    """
    Full hybrid retrieval pipeline:
    1. Encode query to dense embedding
    2. Encode query to sparse BM25 vector
    3. Milvus hybrid search (ANN + sparse, RRF fusion)
    4. Rule-based filtering
    5. Cross-encoder reranking
    """

    def __init__(
        self,
        milvus_client: MilvusClient,
        bm25_retriever: BM25Retriever,
        rule_filter: RuleFilter,
        reranker: Reranker,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        self.milvus = milvus_client
        self.bm25 = bm25_retriever
        self.rule_filter = rule_filter
        self.reranker = reranker
        self.embedding_model = embedding_model or EmbeddingModel()

        logger.info("HybridRetriever initialized")

    def retrieve(
        self,
        query: str,
        context: Optional[dict | FilterContext] = None,
        top_k: int = 10,
        pre_filter_k: int = 20,
    ) -> list[Document]:
        """
        Execute full hybrid retrieval pipeline.

        Args:
            query: Natural language query string.
            context: Enterprise/loan context for rule filtering.
            top_k: Final number of results to return after reranking.
            pre_filter_k: Number of results to fetch before filtering.

        Returns:
            List of Document objects sorted by final relevance score.
        """
        logger.info(f"HybridRetriever.retrieve: query='{query[:80]}...', top_k={top_k}")

        # Step 1: Encode query to dense vector
        dense_vector: list[float] = self._encode_dense(query)

        # Step 2: Encode query to sparse BM25 vector
        sparse_vector: dict[int, float] = self._encode_sparse(query)

        # Step 3: Milvus hybrid search
        raw_results = self._hybrid_search(dense_vector, sparse_vector, pre_filter_k)
        logger.debug(f"Hybrid search returned {len(raw_results)} raw results")

        # Step 4: Rule filtering
        if context is not None:
            filter_ctx = (
                context
                if isinstance(context, FilterContext)
                else FilterContext.from_enterprise_data(context)
            )
            filtered_results = self.rule_filter.filter(raw_results, filter_ctx)
            logger.debug(f"After rule filtering: {len(filtered_results)} results")
        else:
            filtered_results = raw_results

        # Ensure we have at least some results
        if not filtered_results:
            logger.warning("Rule filter removed all results; using unfiltered top results")
            filtered_results = raw_results[:top_k]

        # Step 5: Rerank top results
        final_results = self.reranker.rerank(query, filtered_results, top_k=top_k)
        logger.info(f"Final reranked results: {len(final_results)}")

        return final_results

    def index_documents(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        doc_ids: Optional[list[str]] = None,
    ) -> bool:
        """
        Index a list of documents into the vector store and BM25.

        Args:
            texts: Raw text content for each document.
            metadatas: Optional metadata dicts per document.
            doc_ids: Optional explicit document IDs.

        Returns:
            True if indexing succeeded.
        """
        if not texts:
            return True

        if metadatas is None:
            metadatas = [{}] * len(texts)
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(texts))]

        logger.info(f"Indexing {len(texts)} documents")

        # Fit/update BM25 on corpus
        self.bm25.fit(texts)

        # Encode dense vectors
        try:
            dense_vectors = self.embedding_model.encode(texts, normalize=True)
            if len(dense_vectors.shape) == 1:
                dense_vectors = dense_vectors.reshape(1, -1)
        except Exception as e:
            logger.warning(f"Dense encoding failed: {e}")
            import numpy as np
            rng = np.random.default_rng(42)
            dense_vectors = rng.standard_normal(
                (len(texts), settings.embedding.dim)
            ).astype(np.float32)

        # Build documents
        docs = []
        for i, (text, meta, doc_id) in enumerate(zip(texts, metadatas, doc_ids)):
            sparse_vec = self.bm25.get_sparse_vector(text)
            doc = Document(
                id=doc_id,
                text=text,
                dense_vector=dense_vectors[i].tolist(),
                sparse_vector=sparse_vec,
                metadata=meta,
            )
            docs.append(doc)

        # Insert into Milvus (or fallback)
        success = self.milvus.insert_documents(docs)
        if success:
            logger.info(f"Successfully indexed {len(docs)} documents")
        return success

    def _encode_dense(self, query: str) -> list[float]:
        """Encode query to dense vector."""
        try:
            vec = self.embedding_model.encode(query, normalize=True)
            return vec.tolist()
        except Exception as e:
            logger.warning(f"Dense encoding failed: {e}. Using zero vector.")
            return [0.0] * settings.embedding.dim

    def _encode_sparse(self, query: str) -> dict[int, float]:
        """Encode query to sparse BM25 vector."""
        try:
            return self.bm25.get_sparse_vector(query)
        except Exception as e:
            logger.warning(f"Sparse encoding failed: {e}. Using empty sparse vector.")
            return {}

    def _hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        top_k: int,
    ) -> list[Document]:
        """Execute hybrid search in Milvus."""
        try:
            return self.milvus.hybrid_search(
                query_dense=dense_vector,
                query_sparse=sparse_vector,
                top_k=top_k,
            )
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def get_stats(self) -> dict:
        """Return stats about the retrieval system."""
        return {
            "embedding_model_available": self.embedding_model.is_available,
            "bm25_fitted": self.bm25.is_fitted,
            "bm25_vocab_size": self.bm25.vocab_size,
            "reranker_available": self.reranker.is_model_available,
            "milvus_stats": self.milvus.get_collection_stats(),
        }


def create_hybrid_retriever() -> HybridRetriever:
    """
    Factory function to create a fully initialized HybridRetriever.
    All components fall back gracefully if external services are unavailable.
    """
    milvus_client = MilvusClient()
    bm25_retriever = BM25Retriever()
    rule_filter = RuleFilter()
    reranker = Reranker()
    embedding_model = EmbeddingModel()

    return HybridRetriever(
        milvus_client=milvus_client,
        bm25_retriever=bm25_retriever,
        rule_filter=rule_filter,
        reranker=reranker,
        embedding_model=embedding_model,
    )
