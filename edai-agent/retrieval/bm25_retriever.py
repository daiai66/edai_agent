"""
BM25 retrieval module with Chinese + English mixed-language support.
Uses jieba for Chinese tokenization and rank_bm25 for scoring.
Produces sparse vectors compatible with Milvus SPARSE_FLOAT_VECTOR field.
"""
from __future__ import annotations

import re
import math
import hashlib
from typing import Optional

import numpy as np
from loguru import logger

from config.settings import settings


def _hash_token(token: str) -> int:
    """Convert a token string to a stable integer token_id via MD5 hash truncation."""
    h = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)  # Positive 31-bit integer


class BM25Retriever:
    """
    BM25-based sparse retrieval with jieba Chinese tokenization.
    Produces sparse vector representations suitable for Milvus.
    """

    def __init__(
        self,
        k1: Optional[float] = None,
        b: Optional[float] = None,
        min_token_length: int = 1,
    ):
        self.k1 = k1 or settings.bm25.k1
        self.b = b or settings.bm25.b
        self.min_token_length = min_token_length
        self.stop_words: set[str] = set(settings.bm25.stop_words)

        # BM25 internal state
        self._bm25 = None
        self._corpus_tokens: list[list[str]] = []
        self._vocab: dict[str, int] = {}  # token -> token_id
        self._idf: dict[str, float] = {}
        self._avg_doc_len: float = 0.0
        self._doc_lengths: list[int] = []
        self._fitted: bool = False

        self._init_jieba()

    def _init_jieba(self) -> None:
        """Initialize jieba tokenizer."""
        try:
            import jieba
            jieba.initialize()
            # Add financial domain vocabulary
            financial_terms = [
                "应收账款", "信用贷款", "供应链金融", "保理融资", "知识产权质押",
                "高新技术企业", "营业额", "经营年限", "信用评分", "中小微企业",
                "季节性周转贷款", "连锁餐饮", "农业产业链", "农产品收购",
                "农业科技", "电商", "供应链", "利率优惠", "专利", "认证",
                "还款能力", "税务申报", "财务数据", "POS流水", "应收账款",
                "核心企业", "上下游", "中小微",
            ]
            for term in financial_terms:
                jieba.add_word(term)
            self._jieba = jieba
            logger.debug("jieba initialized with financial domain vocabulary")
        except ImportError:
            logger.warning("jieba not available; falling back to character n-gram tokenization")
            self._jieba = None

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize mixed Chinese/English text.
        - Chinese text: jieba word segmentation
        - English text: lowercase whitespace split
        - Filters stop words and short tokens
        """
        if not text:
            return []

        tokens: list[str] = []

        if self._jieba:
            # Use jieba for full text (handles mixed CN/EN)
            raw_tokens = list(self._jieba.cut(text, cut_all=False))
        else:
            # Fallback: split English and character-bigram Chinese
            raw_tokens = self._fallback_tokenize(text)

        for token in raw_tokens:
            token = token.strip()
            # Skip whitespace
            if not token or token.isspace():
                continue
            # Lowercase English tokens
            if re.match(r"^[a-zA-Z0-9]+$", token):
                token = token.lower()
            # Skip stop words
            if token in self.stop_words:
                continue
            # Skip tokens shorter than minimum
            if len(token) < self.min_token_length:
                continue
            # Skip pure punctuation
            if re.match(r"^[^\w一-鿿]+$", token):
                continue
            tokens.append(token)

        return tokens

    def _fallback_tokenize(self, text: str) -> list[str]:
        """Fallback tokenizer: split on whitespace for English, bigrams for Chinese."""
        tokens = []
        # Split into segments: Chinese characters vs ASCII
        segments = re.split(r"([一-鿿]+)", text)
        for seg in segments:
            seg = seg.strip()
            if not seg:
                continue
            if re.search(r"[一-鿿]", seg):
                # Chinese: generate individual chars and bigrams
                tokens.extend(list(seg))
                tokens.extend([seg[i:i+2] for i in range(len(seg)-1)])
            else:
                # ASCII: split on whitespace and punctuation
                tokens.extend(re.split(r"[\s\W]+", seg))
        return [t for t in tokens if t]

    def fit(self, corpus: list[str]) -> "BM25Retriever":
        """
        Train BM25 on a corpus of documents.

        Args:
            corpus: List of document strings.

        Returns:
            Self for chaining.
        """
        logger.info(f"Fitting BM25 on {len(corpus)} documents")
        self._corpus_tokens = [self.tokenize(doc) for doc in corpus]
        self._doc_lengths = [len(tokens) for tokens in self._corpus_tokens]
        self._avg_doc_len = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 1.0
        )

        # Build vocabulary
        vocab_set: set[str] = set()
        for tokens in self._corpus_tokens:
            vocab_set.update(tokens)
        self._vocab = {token: _hash_token(token) for token in vocab_set}

        # Compute IDF
        n_docs = len(self._corpus_tokens)
        df: dict[str, int] = {}
        for tokens in self._corpus_tokens:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        self._idf = {}
        for token, freq in df.items():
            # BM25+ IDF formula
            self._idf[token] = math.log(
                (n_docs - freq + 0.5) / (freq + 0.5) + 1
            )

        # Initialize rank_bm25 for document retrieval
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(
                self._corpus_tokens,
                k1=self.k1,
                b=self.b,
            )
        except ImportError:
            logger.warning("rank_bm25 not available; using manual BM25 implementation")
            self._bm25 = None

        self._fitted = True
        logger.info(
            f"BM25 fitted: vocab_size={len(self._vocab)}, "
            f"avg_doc_len={self._avg_doc_len:.1f}"
        )
        return self

    def get_sparse_vector(self, text: str) -> dict[int, float]:
        """
        Convert text to a sparse vector representation for Milvus.
        Returns {token_id: bm25_score} dict.

        If BM25 is not fitted, falls back to raw TF-based scoring.
        """
        tokens = self.tokenize(text)
        if not tokens:
            return {}

        sparse: dict[int, float] = {}

        if self._fitted and self._idf:
            # Use BM25 term weights
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1

            doc_len = len(tokens)
            for token, count in tf.items():
                if token not in self._idf:
                    # Unknown token: use default IDF
                    idf = math.log(1 + 1.0)
                else:
                    idf = self._idf[token]

                # BM25 TF component
                numerator = count * (self.k1 + 1)
                denominator = count + self.k1 * (
                    1 - self.b + self.b * doc_len / max(self._avg_doc_len, 1)
                )
                tf_weight = numerator / denominator
                score = idf * tf_weight

                if score > 0:
                    token_id = self._vocab.get(token, _hash_token(token))
                    sparse[token_id] = max(sparse.get(token_id, 0.0), score)
        else:
            # Fallback: simple TF-based weighting
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            for token, count in tf.items():
                token_id = _hash_token(token)
                sparse[token_id] = float(count)

        # Normalize scores to [0, 1] range
        if sparse:
            max_score = max(sparse.values())
            if max_score > 0:
                sparse = {k: v / max_score for k, v in sparse.items()}

        return sparse

    def get_scores(self, query: str) -> Optional[np.ndarray]:
        """
        Get BM25 relevance scores for a query against all fitted documents.

        Returns:
            Array of shape (n_docs,) with BM25 scores, or None if not fitted.
        """
        if not self._fitted:
            logger.warning("BM25Retriever not fitted yet; call fit() first")
            return None

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return np.zeros(len(self._corpus_tokens))

        if self._bm25:
            return self._bm25.get_scores(query_tokens)

        # Manual BM25 implementation
        scores = np.zeros(len(self._corpus_tokens))
        for i, doc_tokens in enumerate(self._corpus_tokens):
            tf_map: dict[str, int] = {}
            for token in doc_tokens:
                tf_map[token] = tf_map.get(token, 0) + 1

            doc_len = self._doc_lengths[i]
            for query_token in set(query_tokens):
                if query_token not in self._idf:
                    continue
                tf = tf_map.get(query_token, 0)
                if tf == 0:
                    continue
                idf = self._idf[query_token]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / max(self._avg_doc_len, 1)
                )
                scores[i] += idf * (numerator / denominator)
        return scores

    def get_top_n(self, query: str, corpus: list[str], n: int = 5) -> list[tuple[int, float]]:
        """
        Get top N documents from corpus for query.

        Returns:
            List of (doc_index, score) sorted by score descending.
        """
        scores = self.get_scores(query)
        if scores is None:
            return []
        top_indices = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def is_fitted(self) -> bool:
        return self._fitted
