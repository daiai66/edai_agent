"""
RAG-based Q&A Agent using HybridRetriever and Claude with adaptive thinking + streaming.
Answers financial questions by retrieving relevant documents and synthesizing responses.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, Field
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import settings
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.rule_filter import FilterContext


class Citation(BaseModel):
    """A document citation in the Q&A answer."""
    doc_id: str
    text_excerpt: str
    relevance_score: float
    source_type: str = "knowledge_base"


class QAResponse(BaseModel):
    """Structured response from the QA agent."""
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    retrieved_docs_count: int = 0
    thinking_summary: Optional[str] = None


class QAAgent:
    """
    Retrieval-Augmented Generation Q&A Agent.

    Combines HybridRetriever for relevant document retrieval with
    Claude LLM for answer synthesis. Supports Chinese and English queries.
    """

    SYSTEM_PROMPT = """你是一位专业的中小企业金融顾问助手，专门回答关于SME融资、信贷政策和金融产品的问题。

你的知识来源于内部知识库文档，请始终基于检索到的文档内容回答问题。

回答准则：
1. 基于文档内容回答，不要凭空捏造数据
2. 如果文档中没有足够信息，请明确说明
3. 对于金融政策和条件，给出精确的数字和条件
4. 回答要专业、简洁、易理解
5. 中文问题用中文回答，英文问题用英文回答
6. 在回答末尾列出引用的文档来源

回答格式：
- 直接回答核心问题
- 提供具体条件和要求
- 给出实际建议
- 注明信息来源"""

    def __init__(
        self,
        retriever: HybridRetriever,
        anthropic_client: Optional[Any] = None,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.top_k = top_k

        if anthropic_client is not None:
            self._client = anthropic_client
        else:
            self._client = self._init_client()

    def _init_client(self) -> Optional[Any]:
        """Initialize Anthropic client."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=settings.anthropic.api_key)
            logger.info("QAAgent: Anthropic client initialized")
            return client
        except Exception as e:
            logger.warning(f"QAAgent: Anthropic client init failed: {e}")
            return None

    @retry(
        stop=stop_after_attempt(settings.app.max_retries),
        wait=wait_exponential(
            multiplier=settings.app.retry_multiplier,
            min=settings.app.retry_wait_min,
            max=settings.app.retry_wait_max,
        ),
        reraise=True,
    )
    def answer(
        self,
        question: str,
        context: Optional[dict | FilterContext] = None,
        use_thinking: bool = True,
    ) -> QAResponse:
        """
        Answer a financial question using RAG.

        Args:
            question: Natural language question (Chinese or English).
            context: Optional enterprise/loan context for relevance filtering.
            use_thinking: Whether to enable Claude's adaptive thinking.

        Returns:
            QAResponse with answer, citations, and confidence score.
        """
        logger.info(f"QAAgent.answer: '{question[:80]}'")

        # Step 1: Retrieve relevant documents
        try:
            filter_ctx = (
                context
                if isinstance(context, FilterContext)
                else (FilterContext.from_enterprise_data(context) if context else None)
            )
            docs = self.retriever.retrieve(
                query=question,
                context=filter_ctx,
                top_k=self.top_k,
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            docs = []

        logger.debug(f"Retrieved {len(docs)} documents for Q&A")

        # Step 2: Build prompt with retrieved context
        context_text = self._format_retrieved_context(docs)
        user_message = self._build_user_message(question, context_text, context)

        # Step 3: Call Claude with streaming
        answer_text, thinking_text = self._call_claude(
            user_message=user_message,
            use_thinking=use_thinking,
        )

        # Step 4: Build citations
        citations = self._build_citations(docs)

        # Step 5: Estimate confidence
        confidence = self._estimate_confidence(docs, answer_text)

        return QAResponse(
            question=question,
            answer=answer_text,
            citations=citations,
            confidence=confidence,
            retrieved_docs_count=len(docs),
            thinking_summary=thinking_text[:500] if thinking_text else None,
        )

    def _format_retrieved_context(self, docs: list) -> str:
        """Format retrieved documents into context string for the prompt."""
        if not docs:
            return "暂无相关文档。"

        parts = []
        for i, doc in enumerate(docs, start=1):
            text = getattr(doc, "text", str(doc))
            score = getattr(doc, "score", 0.0)
            doc_id = getattr(doc, "id", f"doc_{i}")
            meta = getattr(doc, "metadata", {})
            doc_type = meta.get("doc_type", "知识文档")
            parts.append(
                f"【文档{i}】（ID: {doc_id}, 相关度: {score:.3f}, 类型: {doc_type}）\n{text}"
            )
        return "\n\n".join(parts)

    def _build_user_message(
        self,
        question: str,
        context_text: str,
        enterprise_context: Optional[dict | FilterContext],
    ) -> str:
        """Build the user message for Claude."""
        parts = []

        if enterprise_context:
            if isinstance(enterprise_context, dict):
                company = enterprise_context.get("company_name", "")
                industry = enterprise_context.get("industry", "")
                if company or industry:
                    parts.append(f"企业背景：{company}，行业：{industry}")

        parts.append(f"问题：{question}")
        parts.append(f"\n参考文档：\n{context_text}")
        parts.append("\n请基于以上参考文档，给出专业、准确的回答。")

        return "\n".join(parts)

    def _call_claude(
        self,
        user_message: str,
        use_thinking: bool = True,
    ) -> tuple[str, Optional[str]]:
        """Call Claude API with streaming and return (answer_text, thinking_text)."""
        if self._client is None:
            logger.warning("No Anthropic client; returning mock response")
            return self._mock_response(user_message), None

        try:
            kwargs = {
                "model": settings.anthropic.model,
                "max_tokens": settings.anthropic.max_tokens,
                "system": [
                    {
                        "type": "text",
                        "text": self.SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [{"role": "user", "content": user_message}],
            }

            if use_thinking:
                kwargs["thinking"] = {"type": "adaptive"}
            else:
                kwargs["output_config"] = {"effort": "medium"}

            with self._client.messages.stream(**kwargs) as stream:
                message = stream.get_final_message()

            text_parts = []
            thinking_parts = []

            for block in message.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
                elif hasattr(block, "thinking"):
                    thinking_parts.append(block.thinking)

            answer = "\n".join(text_parts)
            thinking = "\n".join(thinking_parts) if thinking_parts else None

            logger.info(
                f"Claude response: {len(answer)} chars, "
                f"thinking: {len(thinking) if thinking else 0} chars"
            )
            return answer, thinking

        except Exception as e:
            if "rate_limit" in str(e).lower():
                logger.warning(f"Rate limit hit: {e}")
                raise
            logger.warning(f"Claude API call failed: {e}")
            return self._mock_response(user_message), None

    @staticmethod
    def _mock_response(user_message: str) -> str:
        """Generate mock response when API is unavailable."""
        return (
            "根据知识库中的文档，中小微企业申请信用贷款的基本条件是：\n"
            "1. 营业额超过500万元\n"
            "2. 经营年限2年以上\n"
            "3. 信用评分700分以上\n\n"
            "（注：这是演示响应，实际分析需要配置Anthropic API Key）"
        )

    @staticmethod
    def _build_citations(docs: list) -> list[Citation]:
        """Build citation objects from retrieved documents."""
        citations = []
        for doc in docs:
            text = getattr(doc, "text", "")
            doc_id = getattr(doc, "id", "unknown")
            score = getattr(doc, "score", 0.0)
            meta = getattr(doc, "metadata", {})

            citation = Citation(
                doc_id=doc_id,
                text_excerpt=text[:200] + "..." if len(text) > 200 else text,
                relevance_score=float(score),
                source_type=meta.get("doc_type", "knowledge_base"),
            )
            citations.append(citation)
        return citations

    @staticmethod
    def _estimate_confidence(docs: list, answer: str) -> float:
        """Estimate answer confidence based on retrieval scores and answer quality."""
        if not docs:
            return 0.3

        # Average top-3 retrieval scores
        scores = [getattr(d, "score", 0.0) for d in docs[:3]]
        avg_score = sum(scores) / max(len(scores), 1)

        # Penalize very short answers
        length_factor = min(1.0, len(answer) / 200)

        # Boost for answers that reference specific numbers/conditions
        import re
        has_specifics = bool(re.search(r"\d+", answer))
        specifics_boost = 0.1 if has_specifics else 0.0

        confidence = min(1.0, avg_score * 0.6 + length_factor * 0.3 + specifics_boost + 0.3)
        return round(confidence, 2)

    def batch_answer(
        self,
        questions: list[str],
        context: Optional[dict] = None,
    ) -> list[QAResponse]:
        """Answer multiple questions sequentially."""
        responses = []
        for question in questions:
            try:
                response = self.answer(question, context)
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to answer question '{question[:50]}': {e}")
                responses.append(QAResponse(
                    question=question,
                    answer=f"Error: {str(e)}",
                    confidence=0.0,
                ))
        return responses
