"""
Knowledge graph tools for industry analysis, supply chain relations, and competitor analysis.
Supports Neo4j backend with in-memory graph fallback.
"""
from __future__ import annotations

import json
from typing import Any, ClassVar, Optional, Type

from pydantic import BaseModel, Field
from loguru import logger

from config.settings import settings
from tools.financial_tools import _call_claude_streaming

try:
    from crewai.tools import BaseTool
except ImportError:
    class BaseTool:
        name: str = ""
        description: str = ""
        def _run(self, *args, **kwargs): raise NotImplementedError
        def run(self, *args, **kwargs): return self._run(*args, **kwargs)


# ─── In-Memory Knowledge Graph ────────────────────────────────────────────────

INDUSTRY_KNOWLEDGE_BASE = {
    "农业科技": {
        "risk_profile": "medium",
        "avg_credit_score": 680,
        "seasonal_risk": True,
        "policy_support": "high",
        "typical_loan_products": ["农业季节性周转贷", "知识产权质押贷款", "供应链保理融资"],
        "key_risks": ["季节性收入波动", "自然灾害风险", "农产品价格波动", "政策变化风险"],
        "success_factors": ["核心技术壁垒", "稳定的收购渠道", "政府补贴支持", "品牌溢价能力"],
        "regulatory_notes": "享受农业扶持政策，利率可优惠20%",
        "avg_profit_margin": 0.14,
        "avg_asset_turnover": 1.2,
    },
    "制造业": {
        "risk_profile": "medium",
        "avg_credit_score": 700,
        "seasonal_risk": False,
        "policy_support": "medium",
        "typical_loan_products": ["固定资产抵押贷款", "供应链保理融资"],
        "key_risks": ["原材料价格上涨", "人工成本上升", "产能过剩", "出口订单波动"],
        "success_factors": ["自动化水平", "质量认证", "订单稳定性", "技术研发投入"],
        "avg_profit_margin": 0.08,
        "avg_asset_turnover": 0.9,
    },
    "餐饮": {
        "risk_profile": "high",
        "avg_credit_score": 650,
        "seasonal_risk": True,
        "policy_support": "low",
        "typical_loan_products": ["餐饮连锁经营贷", "中小微企业信用贷"],
        "key_risks": ["食品安全事件", "租金成本", "人员流动率", "消费降级风险"],
        "success_factors": ["品牌知名度", "标准化程度", "选址能力", "复购率"],
        "avg_profit_margin": 0.10,
        "avg_asset_turnover": 2.5,
    },
    "科技": {
        "risk_profile": "medium",
        "avg_credit_score": 720,
        "seasonal_risk": False,
        "policy_support": "very_high",
        "typical_loan_products": ["知识产权质押贷款", "中小微企业信用贷"],
        "key_risks": ["技术迭代风险", "人才流失", "知识产权侵权", "融资烧钱风险"],
        "success_factors": ["核心专利", "技术护城河", "商业化能力", "团队背景"],
        "avg_profit_margin": 0.18,
        "avg_asset_turnover": 0.8,
    },
    "零售": {
        "risk_profile": "medium",
        "avg_credit_score": 670,
        "seasonal_risk": True,
        "policy_support": "low",
        "typical_loan_products": ["供应链保理融资", "中小微企业信用贷"],
        "key_risks": ["库存积压", "电商冲击", "消费趋势变化", "租金上涨"],
        "success_factors": ["选品能力", "库存周转率", "会员体系", "全渠道布局"],
        "avg_profit_margin": 0.07,
        "avg_asset_turnover": 2.0,
    },
}

SUPPLY_CHAIN_GRAPH = {
    "京东农业": {
        "type": "core_enterprise",
        "credit_rating": "AAA",
        "industry": "电商",
        "avg_payment_days": 30,
        "suppliers_count": 5000,
        "supports_factoring": True,
    },
    "盒马生鲜": {
        "type": "core_enterprise",
        "credit_rating": "AAA",
        "industry": "新零售",
        "avg_payment_days": 45,
        "suppliers_count": 3000,
        "supports_factoring": True,
    },
    "农业合作社A": {
        "type": "supplier",
        "credit_rating": "A",
        "industry": "农业",
        "avg_delivery_days": 7,
    },
    "有机肥料厂B": {
        "type": "supplier",
        "credit_rating": "A",
        "industry": "农资",
        "avg_delivery_days": 5,
    },
}


# ─── Input Schemas ────────────────────────────────────────────────────────────

class IndustryKnowledgeInput(BaseModel):
    """Input for industry knowledge query."""
    industry: str = Field(description="Industry sector name")
    query_type: str = Field(
        default="full",
        description="Query type: 'full', 'risk_profile', 'loan_products', 'success_factors'"
    )


class SupplyChainRelationsInput(BaseModel):
    """Input for supply chain relations query."""
    company_name: str = Field(description="Target company name")
    supply_chain_data: dict = Field(description="Supply chain data with customers and suppliers")
    depth: int = Field(default=2, description="Graph traversal depth")


class CompetitorAnalysisInput(BaseModel):
    """Input for competitor analysis."""
    company_name: str = Field(description="Company name")
    industry: str = Field(description="Industry sector")
    annual_revenue: float = Field(description="Annual revenue for benchmarking")
    key_metrics: Optional[dict] = Field(default=None, description="Key financial metrics")


# ─── Graph Tools ──────────────────────────────────────────────────────────────

class Neo4jQueryExecutor:
    """Executes Neo4j Cypher queries with fallback to in-memory graph."""

    def __init__(self):
        self._driver = None
        self._available = False
        self._connect()

    def _connect(self) -> None:
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                settings.neo4j.uri,
                auth=(settings.neo4j.user, settings.neo4j.password),
                connection_timeout=5,
            )
            self._driver.verify_connectivity()
            self._available = True
            logger.info(f"Connected to Neo4j at {settings.neo4j.uri}")
        except Exception as e:
            logger.warning(f"Neo4j not available: {e}. Using in-memory graph fallback.")
            self._available = False

    def execute(self, query: str, params: Optional[dict] = None) -> list[dict]:
        if not self._available:
            return []
        try:
            with self._driver.session(database=settings.neo4j.database) as session:
                result = session.run(query, params or {})
                return [dict(record) for record in result]
        except Exception as e:
            logger.warning(f"Neo4j query failed: {e}")
            return []

    @property
    def is_available(self) -> bool:
        return self._available


# Singleton Neo4j executor
_neo4j_executor: Optional[Neo4jQueryExecutor] = None


def get_neo4j_executor() -> Neo4jQueryExecutor:
    global _neo4j_executor
    if _neo4j_executor is None:
        _neo4j_executor = Neo4jQueryExecutor()
    return _neo4j_executor


class QueryIndustryKnowledge(BaseTool):
    """
    Query industry risk patterns and knowledge from knowledge graph.
    """
    name: str = "query_industry_knowledge"
    description: str = (
        "Queries the industry knowledge graph to retrieve risk patterns, "
        "policy support levels, typical loan products, and success/failure factors "
        "for a specific industry. Returns structured industry intelligence."
    )
    args_schema: Type[BaseModel] = IndustryKnowledgeInput

    SYSTEM_PROMPT: ClassVar[str] = """你是一位行业研究专家，请基于提供的行业知识数据，给出该行业的深度分析报告。

分析内容：
1. 行业风险特征和主要风险因素
2. 行业信贷支持政策
3. 适合的金融产品
4. 行业成功关键因素
5. 当前行业趋势和机遇

请以JSON格式输出：
{
  "industry": "行业名称",
  "risk_profile": "low/medium/high",
  "industry_analysis": "行业综合分析",
  "credit_support_policy": "信贷支持政策",
  "recommended_products": ["产品列表"],
  "key_risks": ["风险因素"],
  "success_factors": ["成功因素"],
  "market_trend": "市场趋势描述",
  "lending_recommendation": "信贷建议"
}"""

    def _run(self, industry: str, query_type: str = "full") -> str:
        executor = get_neo4j_executor()

        # Try Neo4j first
        if executor.is_available:
            neo4j_result = executor.execute(
                """
                MATCH (i:Industry {name: $industry})
                OPTIONAL MATCH (i)-[:HAS_RISK]->(r:Risk)
                OPTIONAL MATCH (i)-[:SUPPORTED_BY]->(p:Policy)
                RETURN i, collect(r.name) as risks, collect(p.name) as policies
                """,
                {"industry": industry},
            )
            if neo4j_result:
                knowledge = neo4j_result[0]
                logger.info(f"Retrieved industry knowledge from Neo4j: {industry}")
            else:
                knowledge = INDUSTRY_KNOWLEDGE_BASE.get(industry, {})
        else:
            # Fuzzy match in-memory knowledge base
            knowledge = {}
            for key, val in INDUSTRY_KNOWLEDGE_BASE.items():
                if key in industry or industry in key:
                    knowledge = val
                    break
            if not knowledge:
                knowledge = INDUSTRY_KNOWLEDGE_BASE.get("科技", {})  # Default

        user_content = f"""请分析以下行业知识数据，生成行业分析报告：

行业：{industry}
查询类型：{query_type}

行业基础数据：
{json.dumps(knowledge, ensure_ascii=False, indent=2)}

请给出全面的行业分析和信贷建议。"""

        logger.info(f"Querying industry knowledge for: {industry}")
        result = _call_claude_streaming(
            system=self.SYSTEM_PROMPT,
            user_content=user_content,
            max_tokens=2000,
            use_thinking=False,
        )
        return result


class FindSupplyChainRelations(BaseTool):
    """
    Find and analyze enterprise supply chain relationships.
    """
    name: str = "find_supply_chain_relations"
    description: str = (
        "Finds enterprise supply chain relationships including core customers, "
        "suppliers, and credit ratings of supply chain partners. "
        "Identifies supply chain finance opportunities."
    )
    args_schema: Type[BaseModel] = SupplyChainRelationsInput

    SYSTEM_PROMPT: ClassVar[str] = """你是一位供应链关系分析专家，请分析企业的供应链关系网络。

分析内容：
1. 核心客户信用状况和合作深度
2. 供应商稳定性和依赖度
3. 供应链融资机会（保理、反向保理等）
4. 供应链集中度风险
5. 供应链金融产品推荐

请以JSON格式输出：
{
  "supply_chain_overview": "供应链概述",
  "core_customers_analysis": [{客户分析}],
  "suppliers_analysis": [{供应商分析}],
  "concentration_risk": "low/medium/high",
  "factoring_opportunities": [{保理机会}],
  "supply_chain_finance_eligibility": true/false,
  "estimated_financing_capacity": 估计融资能力（元）,
  "recommendations": ["建议"]
}"""

    def _run(
        self,
        company_name: str,
        supply_chain_data: dict,
        depth: int = 2,
    ) -> str:
        executor = get_neo4j_executor()

        # Enrich supply chain data with graph knowledge
        enriched_customers = []
        customers = supply_chain_data.get("major_customers", [])
        for customer in customers:
            graph_info = SUPPLY_CHAIN_GRAPH.get(customer, {})
            enriched_customers.append({
                "name": customer,
                "graph_data": graph_info,
            })

        enriched_suppliers = []
        suppliers = supply_chain_data.get("major_suppliers", [])
        for supplier in suppliers:
            graph_info = SUPPLY_CHAIN_GRAPH.get(supplier, {})
            enriched_suppliers.append({
                "name": supplier,
                "graph_data": graph_info,
            })

        user_content = f"""请分析以下企业的供应链关系：

企业名称：{company_name}
供应链数据：
{json.dumps(supply_chain_data, ensure_ascii=False, indent=2)}

主要客户（含知识图谱数据）：
{json.dumps(enriched_customers, ensure_ascii=False, indent=2)}

主要供应商（含知识图谱数据）：
{json.dumps(enriched_suppliers, ensure_ascii=False, indent=2)}

图谱遍历深度：{depth}层

请分析供应链关系和融资机会。"""

        logger.info(f"Finding supply chain relations for: {company_name}")
        result = _call_claude_streaming(
            system=self.SYSTEM_PROMPT,
            user_content=user_content,
            max_tokens=2500,
            use_thinking=False,
        )
        return result


class GetCompetitorAnalysis(BaseTool):
    """
    Analyze competitors and benchmark enterprise performance within its industry.
    """
    name: str = "get_competitor_analysis"
    description: str = (
        "Analyzes industry competitors and benchmarks the enterprise's performance "
        "against industry averages and top performers. "
        "Returns competitive positioning and relative risk assessment."
    )
    args_schema: Type[BaseModel] = CompetitorAnalysisInput

    SYSTEM_PROMPT: ClassVar[str] = """你是一位行业竞争分析专家，请对企业进行竞争态势分析。

分析内容：
1. 行业竞争格局（集中度、主要玩家）
2. 企业相对竞争位置
3. 与行业均值的关键指标对比
4. 竞争优势和劣势
5. 市场份额估计

请以JSON格式输出：
{
  "competitive_position": "market_leader/strong_contender/average/weak",
  "market_share_estimate": "估计市场份额",
  "vs_industry_average": {
    "revenue_percentile": 百分位,
    "profitability": "above/at/below_average",
    "growth_rate": "above/at/below_average"
  },
  "competitive_advantages": ["优势"],
  "competitive_disadvantages": ["劣势"],
  "industry_concentration": "low/medium/high",
  "threat_assessment": "low/medium/high",
  "opportunities": ["市场机会"],
  "credit_implication": "对信贷审批的影响"
}"""

    def _run(
        self,
        company_name: str,
        industry: str,
        annual_revenue: float,
        key_metrics: Optional[dict] = None,
    ) -> str:
        # Get industry benchmarks from knowledge base
        industry_data = {}
        for key, val in INDUSTRY_KNOWLEDGE_BASE.items():
            if key in industry or industry in key:
                industry_data = val
                break

        user_content = f"""请对以下企业进行竞争分析：

企业名称：{company_name}
行业：{industry}
年营业额：{annual_revenue:,.0f}元

关键财务指标：
{json.dumps(key_metrics or {}, ensure_ascii=False, indent=2)}

行业基准数据：
- 行业平均利润率：{industry_data.get('avg_profit_margin', 0.10):.1%}
- 行业平均资产周转率：{industry_data.get('avg_asset_turnover', 1.0):.1f}
- 行业风险等级：{industry_data.get('risk_profile', 'medium')}
- 行业政策支持：{industry_data.get('policy_support', 'medium')}

请分析该企业的竞争地位及对信贷申请的影响。"""

        logger.info(f"Getting competitor analysis for: {company_name} in {industry}")
        result = _call_claude_streaming(
            system=self.SYSTEM_PROMPT,
            user_content=user_content,
            max_tokens=2000,
            use_thinking=False,
        )
        return result
