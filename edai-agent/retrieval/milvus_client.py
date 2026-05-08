"""
Milvus vector database client for hybrid dense+sparse retrieval.
Supports RRF (Reciprocal Rank Fusion) of ANN dense search and sparse BM25 search.
Falls back to in-memory storage if Milvus is unavailable.
"""
from __future__ import annotations

import json
import math
from typing import Any, Optional
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from config.settings import settings


@dataclass
class Document:
    """A document stored in the vector database."""
    id: str
    text: str
    dense_vector: list[float]
    sparse_vector: dict[int, float]  # {token_id: score}
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "score": self.score,
        }


class InMemoryFallback:
    """In-memory document store used when Milvus is unavailable."""

    def __init__(self):
        self._documents: list[Document] = []

    def insert(self, docs: list[Document]) -> None:
        existing_ids = {d.id for d in self._documents}
        for doc in docs:
            if doc.id not in existing_ids:
                self._documents.append(doc)
                existing_ids.add(doc.id)
        logger.debug(f"In-memory store now has {len(self._documents)} documents")

    def search_dense(self, query_vector: list[float], top_k: int) -> list[tuple[Document, float]]:
        """Cosine similarity search over dense vectors."""
        if not self._documents:
            return []
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            return []
        q = q / q_norm

        results = []
        for doc in self._documents:
            if not doc.dense_vector:
                continue
            d = np.array(doc.dense_vector, dtype=np.float32)
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-10:
                continue
            d = d / d_norm
            score = float(np.dot(q, d))
            results.append((doc, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_sparse(self, query_sparse: dict[int, float], top_k: int) -> list[tuple[Document, float]]:
        """Dot product over sparse BM25 vectors."""
        if not self._documents:
            return []
        results = []
        for doc in self._documents:
            if not doc.sparse_vector:
                continue
            score = 0.0
            for token_id, q_score in query_sparse.items():
                score += q_score * doc.sparse_vector.get(token_id, 0.0)
            if score > 0:
                results.append((doc, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def count(self) -> int:
        return len(self._documents)


def _rrf_fusion(
    dense_results: list[tuple[Document, float]],
    sparse_results: list[tuple[Document, float]],
    k: int = 60,
) -> list[tuple[Document, float]]:
    """
    Reciprocal Rank Fusion of dense and sparse retrieval results.
    RRF score = 1/(k + rank_dense) + 1/(k + rank_sparse)
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, (doc, _score) in enumerate(dense_results, start=1):
        rrf_scores[doc.id] = rrf_scores.get(doc.id, 0.0) + 1.0 / (k + rank)
        doc_map[doc.id] = doc

    for rank, (doc, _score) in enumerate(sparse_results, start=1):
        rrf_scores[doc.id] = rrf_scores.get(doc.id, 0.0) + 1.0 / (k + rank)
        doc_map[doc.id] = doc

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [(doc_map[doc_id], rrf_scores[doc_id]) for doc_id in sorted_ids]


class MilvusClient:
    """
    Milvus client with hybrid dense+sparse vector support.
    Automatically falls back to in-memory storage when Milvus is not available.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
    ):
        self.host = host or settings.milvus.host
        self.port = port or settings.milvus.port
        self.collection_name = collection_name or settings.milvus.collection_name
        self.dense_dim = settings.milvus.dense_dim

        self._milvus_available = False
        self._collection = None
        self._fallback = InMemoryFallback()

        self._try_connect()

    def _try_connect(self) -> None:
        """Attempt to connect to Milvus, gracefully fall back if unavailable."""
        try:
            from pymilvus import connections, utility
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                timeout=5,
            )
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            self._milvus_available = True
            self._ensure_collection()
        except Exception as e:
            logger.warning(
                f"Milvus not available ({e}). Using in-memory fallback storage."
            )
            self._milvus_available = False

    def _ensure_collection(self) -> None:
        """Create the Milvus collection with dense and sparse fields if it doesn't exist."""
        try:
            from pymilvus import (
                Collection, CollectionSchema, FieldSchema, DataType, utility
            )

            if utility.has_collection(self.collection_name):
                self._collection = Collection(self.collection_name)
                self._collection.load()
                logger.info(f"Loaded existing collection: {self.collection_name}")
                return

            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(
                    name="dense_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.dense_dim,
                ),
                FieldSchema(
                    name="sparse_vector",
                    dtype=DataType.SPARSE_FLOAT_VECTOR,
                ),
            ]

            schema = CollectionSchema(
                fields=fields,
                description="SME Financial Platform Document Store",
                enable_dynamic_field=True,
            )

            self._collection = Collection(
                name=self.collection_name,
                schema=schema,
                consistency_level="Strong",
            )

            # Create dense vector index
            self._collection.create_index(
                field_name="dense_vector",
                index_params={
                    "metric_type": settings.milvus.metric_type,
                    "index_type": settings.milvus.index_type,
                    "params": {"nlist": settings.milvus.nlist},
                },
            )

            # Create sparse vector index
            self._collection.create_index(
                field_name="sparse_vector",
                index_params={
                    "metric_type": "IP",
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "params": {"drop_ratio_build": 0.2},
                },
            )

            self._collection.load()
            logger.info(f"Created new collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to create Milvus collection: {e}")
            self._milvus_available = False

    def insert_documents(self, docs: list[Document]) -> bool:
        """
        Insert documents into Milvus (or fallback store).

        Args:
            docs: List of Document objects with dense and sparse vectors.

        Returns:
            True if insertion succeeded.
        """
        if not self._milvus_available:
            self._fallback.insert(docs)
            logger.info(f"Inserted {len(docs)} docs into in-memory fallback")
            return True

        try:
            data = {
                "id": [d.id for d in docs],
                "text": [d.text[:65000] for d in docs],
                "metadata_json": [json.dumps(d.metadata, ensure_ascii=False)[:65000] for d in docs],
                "dense_vector": [d.dense_vector for d in docs],
                "sparse_vector": [d.sparse_vector for d in docs],
            }
            self._collection.insert(data)
            self._collection.flush()
            logger.info(f"Inserted {len(docs)} documents into Milvus")
            return True
        except Exception as e:
            logger.error(f"Milvus insert failed: {e}. Falling back to in-memory.")
            self._fallback.insert(docs)
            return False

    def hybrid_search(
        self,
        query_dense: list[float],
        query_sparse: dict[int, float],
        top_k: int = 20,
        filters: Optional[dict] = None,
    ) -> list[Document]:
        """
        Hybrid search combining ANN dense search and sparse BM25 search via RRF.

        Args:
            query_dense: Dense embedding vector for the query.
            query_sparse: Sparse BM25 vector {token_id: score} for the query.
            top_k: Number of top results to return.
            filters: Optional metadata filters (applied post-retrieval in fallback mode).

        Returns:
            List of Documents sorted by RRF fusion score.
        """
        if not self._milvus_available:
            return self._fallback_hybrid_search(query_dense, query_sparse, top_k, filters)

        try:
            return self._milvus_hybrid_search(query_dense, query_sparse, top_k, filters)
        except Exception as e:
            logger.warning(f"Milvus hybrid search failed: {e}. Using fallback.")
            return self._fallback_hybrid_search(query_dense, query_sparse, top_k, filters)

    def _milvus_hybrid_search(
        self,
        query_dense: list[float],
        query_sparse: dict[int, float],
        top_k: int,
        filters: Optional[dict],
    ) -> list[Document]:
        """Execute hybrid search via Milvus AnnSearchRequest + RRFRanker."""
        from pymilvus import AnnSearchRequest, RRFRanker

        search_params_dense = {
            "metric_type": settings.milvus.metric_type,
            "params": {"nprobe": settings.milvus.nprobe},
        }
        search_params_sparse = {
            "metric_type": "IP",
            "params": {"drop_ratio_search": 0.2},
        }

        expr = self._build_milvus_filter(filters) if filters else None

        dense_req = AnnSearchRequest(
            data=[query_dense],
            anns_field="dense_vector",
            param=search_params_dense,
            limit=top_k * 2,
            expr=expr,
        )
        sparse_req = AnnSearchRequest(
            data=[query_sparse],
            anns_field="sparse_vector",
            param=search_params_sparse,
            limit=top_k * 2,
            expr=expr,
        )

        results = self._collection.hybrid_search(
            reqs=[dense_req, sparse_req],
            rerank=RRFRanker(k=60),
            limit=top_k,
            output_fields=["id", "text", "metadata_json"],
        )

        documents = []
        for hit in results[0]:
            try:
                metadata = json.loads(hit.entity.get("metadata_json", "{}"))
            except Exception:
                metadata = {}
            doc = Document(
                id=hit.entity.get("id", hit.id),
                text=hit.entity.get("text", ""),
                dense_vector=[],
                sparse_vector={},
                metadata=metadata,
                score=hit.score,
            )
            documents.append(doc)
        return documents

    def _fallback_hybrid_search(
        self,
        query_dense: list[float],
        query_sparse: dict[int, float],
        top_k: int,
        filters: Optional[dict],
    ) -> list[Document]:
        """In-memory hybrid search using cosine + dot product + RRF."""
        dense_results = self._fallback.search_dense(query_dense, top_k * 2)
        sparse_results = self._fallback.search_sparse(query_sparse, top_k * 2)

        # If no sparse results (empty sparse vector), use only dense
        if not sparse_results and dense_results:
            fused = [(doc, score) for doc, score in dense_results]
        else:
            fused = _rrf_fusion(dense_results, sparse_results)

        # Apply filters
        if filters:
            fused = self._apply_filters(fused, filters)

        docs = []
        for doc, score in fused[:top_k]:
            doc.score = score
            docs.append(doc)
        return docs

    def _apply_filters(
        self,
        results: list[tuple[Document, float]],
        filters: dict,
    ) -> list[tuple[Document, float]]:
        """Apply metadata filters to results."""
        filtered = []
        for doc, score in results:
            meta = doc.metadata
            include = True

            if "industry" in filters and meta.get("industry"):
                allowed = filters["industry"]
                if isinstance(allowed, list):
                    if meta["industry"] not in allowed:
                        include = False
                elif meta["industry"] != allowed:
                    include = False

            if "doc_type" in filters:
                allowed = filters["doc_type"]
                if isinstance(allowed, list):
                    if meta.get("doc_type") not in allowed:
                        include = False
                elif meta.get("doc_type") != allowed:
                    include = False

            if include:
                filtered.append((doc, score))
        return filtered

    @staticmethod
    def _build_milvus_filter(filters: dict) -> Optional[str]:
        """Build Milvus filter expression from dict."""
        clauses = []
        if "doc_type" in filters:
            val = filters["doc_type"]
            if isinstance(val, list):
                vals = ", ".join(f'"{v}"' for v in val)
                clauses.append(f"doc_type in [{vals}]")
            else:
                clauses.append(f'doc_type == "{val}"')
        return " && ".join(clauses) if clauses else None

    def get_collection_stats(self) -> dict:
        """Return stats about the document collection."""
        if self._milvus_available and self._collection:
            try:
                stats = self._collection.get_statistics()
                return {"backend": "milvus", "stats": stats}
            except Exception as e:
                return {"backend": "milvus", "error": str(e)}
        return {"backend": "in_memory", "count": self._fallback.count()}

    def close(self) -> None:
        """Close Milvus connection."""
        if self._milvus_available:
            try:
                from pymilvus import connections
                connections.disconnect("default")
            except Exception:
                pass
