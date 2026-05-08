"""
Knowledge Graph Agent for industry analysis and supply chain intelligence.
Queries Neo4j or in-memory graph to enrich enterprise data.
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
)

from config.settings import settings
from tools.graph_tools import (
    QueryIndustryKnowledge,
    FindSupplyChainRelations,
    GetCompetitorAnalysis,
    INDUSTRY_KNOWLEDGE_BASE,
)
from tools.financial_tools import _call_claude_streaming


# ─── Output Models ────────────────────────────────────────────────────────────

class IndustryRiskProfile(BaseModel):
    """Industry risk and characteristics profile."""
    industry: str
    risk_level: str = "medium"
    policy_support: str = "medium"
    seasonal_risk: bool = False
    avg_profit_margin: float = 0.10
    avg_credit_score: int = 680
    key_risks: list[str] = Field(default_factory=list)
    success_factors: list[str] = Field(default_factory=list)


class SupplyChainIntelligence(BaseModel):
    """Supply chain relationship intelligence."""
    has_core_enterprise_relations: bool = False
    core_enterprise_names: list[str] = Field(default_factory=list)
    core_enterprise_credit_ratings: dict[str, str] = Field(default_factory=dict)
    factoring_eligible: bool = False
    estimated_factoring_capacity: float = 0.0
    supply_chain_concentration: str = "medium"  # low/medium/high
    payment_cycle_assessment: str = "normal"


class CompetitivePosition(BaseModel):
    """Enterprise competitive position within industry."""
    position: str = "average"  # market_leader/strong_contender/average/weak
    revenue_percentile: float = 50.0
    competitive_advantages: list[str] = Field(default_factory=list)
    competitive_disadvantages: list[str] = Field(default_factory=list)
    market_opportunities: list[str] = Field(default_factory=list)


class IndustryAnalysis(BaseModel):
    """Complete industry analysis result."""
    company_name: str
    industry: str
    industry_risk_profile: IndustryRiskProfile
    supply_chain_intelligence: SupplyChainIntelligence
    competitive_position: CompetitivePosition
    industry_analysis_text: str = ""
    supply_chain_analysis_text: str = ""
    competitor_analysis_text: str = ""
    credit_implication: str = ""
    enriched_context: dict = Field(default_factory=dict)

    def to_summary(self) -> str:
        return (
            f"行业分析摘要\n"
            f"{'='*50}\n"
            f"企业：{self.company_name}\n"
            f"行业：{self.industry}\n"
            f"行业风险：{self.industry_risk_profile.risk_level}\n"
            f"政策支持：{self.industry_risk_profile.policy_support}\n"
            f"竞争地位：{self.competitive_position.position}\n"
            f"供应链融资资格：{'是' if self.supply_chain_intelligence.factoring_eligible else '否'}\n"
            f"信贷影响：{self.credit_implication}\n"
        )


# ─── Graph Agent ──────────────────────────────────────────────────────────────

class GraphAgent:
    """
    Knowledge Graph Agent for industry and supply chain intelligence.

    Queries the knowledge graph (Neo4j or in-memory fallback) to:
    1. Retrieve industry risk profiles
    2. Analyze supply chain relationships
    3. Benchmark competitive position
    4. Enrich enterprise data for credit assessment
    """

    SYNTHESIS_PROMPT = """你是一位行业研究分析师，请综合行业知识图谱数据，生成结构化分析报告。

分析要涵盖：
1. 行业风险特征（系统性风险、季节性风险、政策风险）
2. 供应链关系强度和融资机会
3. 竞争地位对信贷风险的影响
4. 对贷款审批的综合影响

请以JSON格式输出：
{
  "risk_level": "low/medium/high",
  "policy_support": "very_high/high/medium/low",
  "factoring_eligible": true/false,
  "estimated_factoring_capacity": 估计保理额度（元）,
  "supply_chain_concentration": "low/medium/high",
  "competitive_position": "market_leader/strong_contender/average/weak",
  "revenue_percentile": 百分位（0-100）,
  "competitive_advantages": ["优势"],
  "competitive_disadvantages": ["劣势"],
  "market_opportunities": ["机会"],
  "credit_implication": "信贷影响综述",
  "enrichment_notes": "其他增益信息"
}"""

    def __init__(self):
        self.industry_tool = QueryIndustryKnowledge()
        self.supply_chain_tool = FindSupplyChainRelations()
        self.competitor_tool = GetCompetitorAnalysis()
        logger.info("GraphAgent initialized")

    def analyze(self, enterprise_info: dict) -> IndustryAnalysis:
        """
        Perform comprehensive industry and supply chain analysis.

        Args:
            enterprise_info: Enterprise data dict with industry, supply_chain, financials.

        Returns:
            IndustryAnalysis with structured results.
        """
        company_name = enterprise_info.get("company_name", "Unknown")
        industry = enterprise_info.get("industry", "")
        logger.info(f"GraphAgent.analyze: {company_name} in {industry}")

        # Step 1: Industry knowledge query
        industry_analysis_text = self._query_industry(industry)

        # Step 2: Supply chain relations
        supply_chain_text = self._query_supply_chain(enterprise_info)

        # Step 3: Competitor analysis
        competitor_text = self._query_competitors(enterprise_info)

        # Step 4: Synthesize into structured output
        analysis = self._synthesize(
            enterprise_info=enterprise_info,
            industry_text=industry_analysis_text,
            supply_chain_text=supply_chain_text,
            competitor_text=competitor_text,
        )

        logger.info(f"GraphAgent analysis complete for: {company_name}")
        return analysis

    def _query_industry(self, industry: str) -> str:
        """Query industry knowledge graph."""
        if not industry:
            return json.dumps({"status": "skipped", "reason": "No industry specified"})
        try:
            result = self.industry_tool._run(industry=industry, query_type="full")
            logger.debug(f"Industry knowledge retrieved for: {industry}")
            return result
        except Exception as e:
            logger.error(f"Industry query failed: {e}")
            # Fallback to in-memory data
            data = {}
            for key, val in INDUSTRY_KNOWLEDGE_BASE.items():
                if key in industry or industry in key:
                    data = val
                    break
            return json.dumps(data or {"industry": industry, "status": "partial"}, ensure_ascii=False)

    def _query_supply_chain(self, enterprise_info: dict) -> str:
        """Query supply chain relations."""
        supply_chain = enterprise_info.get("supply_chain")
        if not supply_chain:
            return json.dumps({"status": "skipped", "reason": "No supply chain data"})
        try:
            result = self.supply_chain_tool._run(
                company_name=enterprise_info.get("company_name", ""),
                supply_chain_data=supply_chain,
                depth=2,
            )
            logger.debug("Supply chain relations queried")
            return result
        except Exception as e:
            logger.error(f"Supply chain query failed: {e}")
            return json.dumps({"error": str(e)})

    def _query_competitors(self, enterprise_info: dict) -> str:
        """Query competitor analysis."""
        try:
            # Extract key financial metrics for benchmarking
            financial_data = enterprise_info.get("financial_data", {})
            key_metrics = {}
            if financial_data:
                latest_year = max(financial_data.keys()) if financial_data else None
                if latest_year:
                    year_data = financial_data[latest_year]
                    revenue = year_data.get("revenue", 0)
                    profit = year_data.get("profit", 0)
                    assets = year_data.get("assets", 1)
                    liabilities = year_data.get("liabilities", 0)
                    key_metrics = {
                        "profit_margin": round(profit / max(revenue, 1), 3),
                        "asset_liability_ratio": round(liabilities / max(assets, 1), 3),
                        "roe": round(profit / max(assets - liabilities, 1), 3),
                    }

            result = self.competitor_tool._run(
                company_name=enterprise_info.get("company_name", ""),
                industry=enterprise_info.get("industry", ""),
                annual_revenue=enterprise_info.get("annual_revenue", 0),
                key_metrics=key_metrics,
            )
            logger.debug("Competitor analysis complete")
            return result
        except Exception as e:
            logger.error(f"Competitor analysis failed: {e}")
            return json.dumps({"error": str(e)})

    def _synthesize(
        self,
        enterprise_info: dict,
        industry_text: str,
        supply_chain_text: str,
        competitor_text: str,
    ) -> IndustryAnalysis:
        """Synthesize all graph analysis into structured IndustryAnalysis."""
        company_name = enterprise_info.get("company_name", "Unknown")
        industry = enterprise_info.get("industry", "")

        # Try to get structured synthesis from Claude
        try:
            synthesis_data = self._call_synthesis(
                enterprise_info, industry_text, supply_chain_text, competitor_text
            )
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}. Using heuristic fallback.")
            synthesis_data = self._heuristic_synthesis(enterprise_info, industry)

        # Build structured output
        industry_base = {}
        for key, val in INDUSTRY_KNOWLEDGE_BASE.items():
            if key in industry or industry in key:
                industry_base = val
                break

        industry_risk = IndustryRiskProfile(
            industry=industry,
            risk_level=synthesis_data.get("risk_level", industry_base.get("risk_profile", "medium")),
            policy_support=synthesis_data.get("policy_support", industry_base.get("policy_support", "medium")),
            seasonal_risk=industry_base.get("seasonal_risk", False),
            avg_profit_margin=industry_base.get("avg_profit_margin", 0.10),
            avg_credit_score=industry_base.get("avg_credit_score", 680),
            key_risks=industry_base.get("key_risks", []),
            success_factors=industry_base.get("success_factors", []),
        )

        supply_chain_data = enterprise_info.get("supply_chain", {})
        major_customers = supply_chain_data.get("major_customers", [])
        factoring_eligible = bool(
            synthesis_data.get("factoring_eligible", False)
            or any(c in ["京东农业", "盒马生鲜", "阿里巴巴"] for c in major_customers)
        )

        sc_intelligence = SupplyChainIntelligence(
            has_core_enterprise_relations=bool(major_customers),
            core_enterprise_names=major_customers[:3],
            factoring_eligible=factoring_eligible,
            estimated_factoring_capacity=float(
                synthesis_data.get("estimated_factoring_capacity", 0)
            ),
            supply_chain_concentration=synthesis_data.get("supply_chain_concentration", "medium"),
        )

        competitive = CompetitivePosition(
            position=synthesis_data.get("competitive_position", "average"),
            revenue_percentile=float(synthesis_data.get("revenue_percentile", 50)),
            competitive_advantages=synthesis_data.get("competitive_advantages", []),
            competitive_disadvantages=synthesis_data.get("competitive_disadvantages", []),
            market_opportunities=synthesis_data.get("market_opportunities", []),
        )

        return IndustryAnalysis(
            company_name=company_name,
            industry=industry,
            industry_risk_profile=industry_risk,
            supply_chain_intelligence=sc_intelligence,
            competitive_position=competitive,
            industry_analysis_text=industry_text,
            supply_chain_analysis_text=supply_chain_text,
            competitor_analysis_text=competitor_text,
            credit_implication=synthesis_data.get("credit_implication", ""),
            enriched_context=synthesis_data,
        )

    def _call_synthesis(
        self,
        enterprise_info: dict,
        industry_text: str,
        supply_chain_text: str,
        competitor_text: str,
    ) -> dict:
        """Call Claude to synthesize graph analysis."""
        company_name = enterprise_info.get("company_name", "")
        industry = enterprise_info.get("industry", "")
        annual_revenue = enterprise_info.get("annual_revenue", 0)
        loan_amount = enterprise_info.get("loan_request", {}).get("amount", 0)

        user_content = f"""请综合以下知识图谱分析结果，生成结构化评估：

企业：{company_name}
行业：{industry}
年营业额：{annual_revenue:,.0f}元
贷款申请：{loan_amount:,.0f}元

=== 行业知识图谱分析 ===
{industry_text[:2000]}

=== 供应链关系分析 ===
{supply_chain_text[:2000]}

=== 竞争分析 ===
{competitor_text[:2000]}

请给出综合评估。"""

        raw = _call_claude_streaming(
            system=self.SYNTHESIS_PROMPT,
            user_content=user_content,
            max_tokens=2000,
            use_thinking=False,
        )

        # Extract JSON
        import re
        for pattern in [r"```json\s*([\s\S]+?)\s*```", r"(\{[\s\S]+\})"]:
            match = re.search(pattern, raw)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    continue

        try:
            return json.loads(raw)
        except Exception:
            return {}

    @staticmethod
    def _heuristic_synthesis(enterprise_info: dict, industry: str) -> dict:
        """Heuristic synthesis when Claude is unavailable."""
        industry_data = {}
        for key, val in INDUSTRY_KNOWLEDGE_BASE.items():
            if key in industry or industry in key:
                industry_data = val
                break

        annual_revenue = enterprise_info.get("annual_revenue", 0)
        supply_chain = enterprise_info.get("supply_chain", {})
        major_customers = supply_chain.get("major_customers", [])
        avg_payment_days = supply_chain.get("avg_payment_days", 60)

        # Estimate receivables for factoring capacity
        monthly_revenue = annual_revenue / 12
        factoring_capacity = monthly_revenue * (avg_payment_days / 30) * 0.8

        return {
            "risk_level": industry_data.get("risk_profile", "medium"),
            "policy_support": industry_data.get("policy_support", "medium"),
            "factoring_eligible": bool(major_customers),
            "estimated_factoring_capacity": factoring_capacity,
            "supply_chain_concentration": "medium" if len(major_customers) >= 2 else "high",
            "competitive_position": "average",
            "revenue_percentile": 60.0,
            "competitive_advantages": industry_data.get("success_factors", [])[:2],
            "competitive_disadvantages": [],
            "market_opportunities": [],
            "credit_implication": f"行业风险{industry_data.get('risk_profile', 'medium')}，政策支持{industry_data.get('policy_support', 'medium')}",
        }
