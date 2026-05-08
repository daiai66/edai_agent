"""
Configuration settings for the SME Financial Platform.
Reads all configuration from environment variables / .env file.
"""
import os
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class MilvusSettings(BaseSettings):
    """Milvus vector database settings."""
    host: str = Field(default="localhost", alias="MILVUS_HOST")
    port: int = Field(default=19530, alias="MILVUS_PORT")
    collection_name: str = Field(default="sme_financial_docs", alias="MILVUS_COLLECTION_NAME")
    user: Optional[str] = Field(default=None, alias="MILVUS_USER")
    password: Optional[str] = Field(default=None, alias="MILVUS_PASSWORD")
    dense_dim: int = Field(default=768, alias="EMBEDDING_DIM")
    index_type: str = "IVF_FLAT"
    metric_type: str = "COSINE"
    nlist: int = 128
    nprobe: int = 16

    model_config = {"populate_by_name": True, "env_file": ".env", "extra": "ignore"}


class AnthropicSettings(BaseSettings):
    """Anthropic Claude API settings."""
    api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    model: str = Field(default="claude-opus-4-7", alias="CLAUDE_MODEL")
    max_tokens: int = Field(default=16000, alias="CLAUDE_MAX_TOKENS")
    effort: str = Field(default="high", alias="CLAUDE_EFFORT")  # low|medium|high|max

    model_config = {"populate_by_name": True, "env_file": ".env", "extra": "ignore"}


class Neo4jSettings(BaseSettings):
    """Neo4j knowledge graph settings."""
    uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    user: str = Field(default="neo4j", alias="NEO4J_USER")
    password: str = Field(default="password", alias="NEO4J_PASSWORD")
    database: str = "neo4j"
    max_connection_pool_size: int = 50
    connection_timeout: float = 30.0

    model_config = {"populate_by_name": True, "env_file": ".env", "extra": "ignore"}


class EmbeddingSettings(BaseSettings):
    """Sentence transformer embedding model settings."""
    model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        alias="EMBEDDING_MODEL"
    )
    dim: int = Field(default=768, alias="EMBEDDING_DIM")
    batch_size: int = 32
    normalize_embeddings: bool = True
    device: str = "cpu"  # "cuda" if GPU available

    model_config = {"populate_by_name": True, "env_file": ".env", "extra": "ignore"}


class BM25Settings(BaseSettings):
    """BM25 retrieval parameters."""
    k1: float = Field(default=1.5, alias="BM25_K1")
    b: float = Field(default=0.75, alias="BM25_B")
    min_token_length: int = 1
    max_vocab_size: int = 50000
    stop_words: list[str] = [
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
        "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
        "你", "会", "着", "没有", "看", "好", "自己", "这",
    ]

    model_config = {"populate_by_name": True, "env_file": ".env", "extra": "ignore"}


class RerankerSettings(BaseSettings):
    """Cross-encoder reranker settings."""
    model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        alias="RERANKER_MODEL"
    )
    max_length: int = 512
    batch_size: int = 16

    model_config = {"populate_by_name": True, "env_file": ".env", "extra": "ignore"}


class RuleFilterSettings(BaseSettings):
    """Business rule filter settings."""
    # Credit scoring thresholds
    credit_score_min: int = 600
    credit_score_excellent: int = 750

    # Enterprise age requirements (years)
    enterprise_age_min: int = 1
    enterprise_age_preferred: int = 2

    # Revenue thresholds (CNY)
    revenue_threshold_micro: float = 1_000_000    # 100万
    revenue_threshold_small: float = 5_000_000    # 500万
    revenue_threshold_medium: float = 50_000_000  # 5000万

    # Loan amount limits (CNY)
    max_loan_amount: float = 10_000_000  # 1000万
    min_loan_amount: float = 50_000     # 5万

    # Industry whitelist
    industry_whitelist: list[str] = [
        "农业科技", "制造业", "零售", "电商", "科技", "餐饮",
        "供应链", "物流", "医疗", "教育", "建筑", "文化",
        "农业", "食品加工", "纺织", "汽车零部件", "家具制造",
    ]

    # Risk level mapping
    risk_levels: dict = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "very_high": 4,
    }
    max_risk_level: str = "high"

    model_config = {"populate_by_name": True, "env_file": ".env", "extra": "ignore"}


class AppSettings(BaseSettings):
    """Application-level settings."""
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    max_retrieval_results: int = Field(default=20, alias="MAX_RETRIEVAL_RESULTS")
    top_k_rerank: int = Field(default=5, alias="TOP_K_RERANK")
    debug: bool = False

    # Retry settings
    max_retries: int = 3
    retry_wait_min: float = 1.0
    retry_wait_max: float = 60.0
    retry_multiplier: float = 2.0

    # Async settings
    max_concurrent_agents: int = 3
    agent_timeout_seconds: float = 120.0

    model_config = {"populate_by_name": True, "env_file": ".env", "extra": "ignore"}


class PlatformSettings:
    """Unified platform settings container."""

    def __init__(self):
        self.milvus = MilvusSettings()
        self.anthropic = AnthropicSettings()
        self.neo4j = Neo4jSettings()
        self.embedding = EmbeddingSettings()
        self.bm25 = BM25Settings()
        self.reranker = RerankerSettings()
        self.rule_filter = RuleFilterSettings()
        self.app = AppSettings()

    def validate_api_keys(self) -> dict[str, bool]:
        """Check which API keys are configured."""
        return {
            "anthropic": bool(self.anthropic.api_key and self.anthropic.api_key != "your_anthropic_api_key_here"),
            "neo4j": bool(self.neo4j.password and self.neo4j.password != "your_neo4j_password_here"),
            "milvus": True,  # Milvus doesn't always need auth
        }

    def __repr__(self) -> str:
        keys = self.validate_api_keys()
        return (
            f"PlatformSettings("
            f"claude_model={self.anthropic.model}, "
            f"milvus={self.milvus.host}:{self.milvus.port}, "
            f"neo4j={self.neo4j.uri}, "
            f"api_keys_configured={keys}"
            f")"
        )


# Singleton instance
settings = PlatformSettings()
