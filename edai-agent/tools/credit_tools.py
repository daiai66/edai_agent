"""
Credit-specific CrewAI tools for loan product matching, collateral assessment,
and comprehensive credit report generation.
"""
from __future__ import annotations

import json
from typing import Any, ClassVar, Optional, Type

from pydantic import BaseModel, Field
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

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


# ─── Loan Products Database ──────────────────────────────────────────────────

LOAN_PRODUCTS = [
    {
        "product_id": "P001",
        "name": "中小微企业信用贷",
        "type": "credit_loan",
        "min_amount": 100_000,
        "max_amount": 3_000_000,
        "min_term_months": 6,
        "max_term_months": 36,
        "interest_rate_range": [0.04, 0.08],
        "requirements": {
            "min_registration_years": 2,
            "min_annual_revenue": 5_000_000,
            "min_credit_score": 700,
            "industries": ["all"],
        },
        "features": ["无抵押", "快速审批", "线上申请"],
    },
    {
        "product_id": "P002",
        "name": "供应链保理融资",
        "type": "factoring",
        "min_amount": 200_000,
        "max_amount": 10_000_000,
        "min_term_months": 1,
        "max_term_months": 12,
        "interest_rate_range": [0.035, 0.065],
        "requirements": {
            "min_registration_years": 1,
            "min_annual_revenue": 1_000_000,
            "min_credit_score": 600,
            "core_enterprise_required": True,
            "industries": ["all"],
        },
        "features": ["应收账款质押", "核心企业增信", "额度高"],
    },
    {
        "product_id": "P003",
        "name": "农业季节性周转贷",
        "type": "seasonal_loan",
        "min_amount": 100_000,
        "max_amount": 5_000_000,
        "min_term_months": 3,
        "max_term_months": 12,
        "interest_rate_range": [0.032, 0.055],
        "requirements": {
            "min_registration_years": 1,
            "min_annual_revenue": 500_000,
            "min_credit_score": 550,
            "industries": ["农业", "农业科技", "农产品加工", "食品加工"],
        },
        "features": ["农业专属", "利率优惠20%", "灵活还款"],
    },
    {
        "product_id": "P004",
        "name": "知识产权质押贷款",
        "type": "ip_pledge",
        "min_amount": 500_000,
        "max_amount": 10_000_000,
        "min_term_months": 12,
        "max_term_months": 60,
        "interest_rate_range": [0.04, 0.07],
        "requirements": {
            "min_registration_years": 2,
            "min_annual_revenue": 2_000_000,
            "min_credit_score": 650,
            "ip_required": True,
            "industries": ["科技", "农业科技", "制造业", "医疗", "教育"],
        },
        "features": ["专利质押", "高新企业优先", "额度大"],
    },
    {
        "product_id": "P005",
        "name": "固定资产抵押贷款",
        "type": "mortgage_loan",
        "min_amount": 500_000,
        "max_amount": 50_000_000,
        "min_term_months": 12,
        "max_term_months": 120,
        "interest_rate_range": [0.038, 0.065],
        "requirements": {
            "min_registration_years": 1,
            "min_annual_revenue": 1_000_000,
            "min_credit_score": 550,
            "collateral_required": True,
            "industries": ["all"],
        },
        "features": ["额度大", "期限长", "利率低"],
    },
    {
        "product_id": "P006",
        "name": "餐饮连锁经营贷",
        "type": "chain_business_loan",
        "min_amount": 200_000,
        "max_amount": 5_000_000,
        "min_term_months": 12,
        "max_term_months": 36,
        "interest_rate_range": [0.05, 0.085],
        "requirements": {
            "min_registration_years": 3,
            "min_annual_revenue": 3_000_000,
            "min_credit_score": 680,
            "industries": ["餐饮", "连锁餐饮"],
            "special_docs": ["3年财务数据", "税务申报", "POS流水"],
        },
        "features": ["连锁专属", "POS流水认可", "灵活授信"],
    },
]


# ─── Input Schemas ────────────────────────────────────────────────────────────

class CreditReportInput(BaseModel):
    """Input for comprehensive credit report generation."""
    enterprise_data: dict = Field(description="Full enterprise data dict")
    financial_analysis: Optional[str] = Field(
        default=None, description="Pre-computed financial analysis"
    )
    credit_score_report: Optional[str] = Field(
        default=None, description="Pre-computed credit score report"
    )
    supply_chain_assessment: Optional[str] = Field(
        default=None, description="Pre-computed supply chain assessment"
    )


class LoanMatchInput(BaseModel):
    """Input for loan product matching."""
    company_name: str = Field(description="Company name")
    industry: str = Field(description="Industry sector")
    registration_years: float = Field(description="Years in operation")
    annual_revenue: float = Field(description="Annual revenue in CNY")
    credit_score: int = Field(description="Credit score 0-1000")
    loan_amount: float = Field(description="Requested loan amount in CNY")
    loan_term_months: int = Field(description="Requested loan term in months")
    has_collateral: bool = Field(default=False, description="Has collateral")
    has_ip: bool = Field(default=False, description="Has patents/IP")
    has_core_enterprise: bool = Field(
        default=False, description="Has core enterprise in supply chain"
    )


class CollateralInput(BaseModel):
    """Input for collateral assessment."""
    company_name: str = Field(description="Company name")
    collateral_type: str = Field(
        description="Type: real_estate/equipment/inventory/ip/accounts_receivable"
    )
    estimated_value: float = Field(description="Estimated market value in CNY")
    collateral_details: dict = Field(
        default_factory=dict, description="Additional collateral details"
    )


# ─── Credit Tools ─────────────────────────────────────────────────────────────

class GenerateCreditReport(BaseTool):
    """
    Generate a comprehensive credit report synthesizing all assessment dimensions.
    """
    name: str = "generate_credit_report"
    description: str = (
        "Generates a comprehensive enterprise credit report combining financial analysis, "
        "credit scoring, supply chain assessment, and tax compliance. "
        "Output: Full credit report with approval recommendation."
    )
    args_schema: Type[BaseModel] = CreditReportInput

    SYSTEM_PROMPT: ClassVar[str] = """你是一位资深的企业信贷审查官，负责综合所有分析维度生成最终信贷报告。

报告结构：
1. 企业基本信息摘要
2. 财务健康状况（流动性/盈利/偿债）
3. 信用评分及等级
4. 供应链状况
5. 税务合规情况
6. 风险因素汇总
7. 贷款建议（批准/条件批准/拒绝）
8. 贷款条件建议（额度/期限/利率/担保要求）

请综合所有分析数据，以JSON格式输出完整信贷报告：
{
  "report_id": "唯一报告ID",
  "company_name": "企业名称",
  "report_date": "报告日期",
  "executive_summary": "执行摘要",
  "financial_health": {综合财务健康评估},
  "credit_score": 数字,
  "credit_grade": "等级",
  "risk_assessment": {
    "overall_risk": "low/medium/high",
    "key_risks": [],
    "mitigating_factors": []
  },
  "loan_recommendation": {
    "decision": "approved/conditional/rejected",
    "approved_amount": 数字,
    "recommended_term_months": 数字,
    "interest_rate_suggestion": 数字,
    "conditions": [],
    "rejection_reasons": []
  },
  "monitoring_requirements": []
}"""

    def _run(
        self,
        enterprise_data: dict,
        financial_analysis: Optional[str] = None,
        credit_score_report: Optional[str] = None,
        supply_chain_assessment: Optional[str] = None,
    ) -> str:
        company_name = enterprise_data.get("company_name", "Unknown")
        loan_request = enterprise_data.get("loan_request", {})

        context_parts = [
            f"企业数据：\n{json.dumps(enterprise_data, ensure_ascii=False, indent=2)}",
        ]
        if financial_analysis:
            context_parts.append(f"\n财务分析报告：\n{financial_analysis}")
        if credit_score_report:
            context_parts.append(f"\n信用评分报告：\n{credit_score_report}")
        if supply_chain_assessment:
            context_parts.append(f"\n供应链评估：\n{supply_chain_assessment}")

        user_content = f"""请为以下企业生成综合信贷报告：

{"".join(context_parts)}

贷款申请：
- 申请金额：{loan_request.get('amount', 0):,.0f}元
- 用途：{loan_request.get('purpose', '未说明')}
- 期限：{loan_request.get('term_months', 0)}个月

请综合所有信息出具完整信贷报告。"""

        logger.info(f"Generating credit report for: {company_name}")
        result = _call_claude_streaming(
            system=self.SYSTEM_PROMPT,
            user_content=user_content,
            max_tokens=6000,
            use_thinking=True,
        )
        return result


class MatchLoanProducts(BaseTool):
    """
    Match enterprise profile to available loan products.
    """
    name: str = "match_loan_products"
    description: str = (
        "Matches enterprise profile to available loan products based on eligibility criteria. "
        "Returns a ranked list of suitable products with estimated terms."
    )
    args_schema: Type[BaseModel] = LoanMatchInput

    SYSTEM_PROMPT: ClassVar[str] = """你是一位银行贷款产品专家，负责为中小企业匹配最合适的贷款产品。

请根据企业条件和可用产品，分析：
1. 企业符合哪些产品的申请资格
2. 每个产品的适配程度评分
3. 预计可获批额度和利率
4. 最推荐的产品及理由

请以JSON格式返回：
{
  "eligible_products": [
    {
      "product_id": "产品ID",
      "product_name": "产品名称",
      "match_score": 0-100,
      "eligible": true/false,
      "estimated_amount": 预计可获批额度,
      "estimated_rate": 预计利率,
      "advantages": [优势],
      "limitations": [限制],
      "application_tips": [申请建议]
    }
  ],
  "top_recommendation": "最推荐产品ID",
  "recommendation_reason": "推荐理由"
}"""

    def _run(
        self,
        company_name: str,
        industry: str,
        registration_years: float,
        annual_revenue: float,
        credit_score: int,
        loan_amount: float,
        loan_term_months: int,
        has_collateral: bool = False,
        has_ip: bool = False,
        has_core_enterprise: bool = False,
    ) -> str:
        user_content = f"""请为以下企业匹配合适的贷款产品：

企业信息：
- 企业名称：{company_name}
- 行业：{industry}
- 注册年限：{registration_years}年
- 年营业额：{annual_revenue:,.0f}元
- 信用评分：{credit_score}分
- 有无抵押品：{"有" if has_collateral else "无"}
- 有无知识产权：{"有" if has_ip else "无"}
- 有无核心企业关系：{"有" if has_core_enterprise else "无"}

贷款需求：
- 申请金额：{loan_amount:,.0f}元
- 申请期限：{loan_term_months}个月

可用贷款产品：
{json.dumps(LOAN_PRODUCTS, ensure_ascii=False, indent=2)}

请分析哪些产品适合该企业，并给出推荐排名。"""

        logger.info(f"Matching loan products for: {company_name}")
        result = _call_claude_streaming(
            system=self.SYSTEM_PROMPT,
            user_content=user_content,
            max_tokens=3000,
            use_thinking=False,
        )
        return result


class AssessCollateral(BaseTool):
    """
    Evaluate collateral value and eligibility for loan securing.
    """
    name: str = "assess_collateral"
    description: str = (
        "Evaluates collateral assets (real estate, equipment, IP, inventory, receivables) "
        "for loan security purposes. Returns LTV ratios and risk assessment."
    )
    args_schema: Type[BaseModel] = CollateralInput

    # LTV ratios by collateral type
    LTV_RATIOS: ClassVar[dict] = {
        "real_estate": 0.70,
        "equipment": 0.50,
        "inventory": 0.40,
        "ip": 0.40,
        "accounts_receivable": 0.80,
        "other": 0.30,
    }

    SYSTEM_PROMPT: ClassVar[str] = """你是一位专业的资产评估师，负责评估中小企业贷款担保资产的价值和风险。

评估维度：
1. 资产真实价值评估
2. 变现能力分析
3. 市场流动性
4. 法律合规性（是否可作为抵押品）
5. 贷款价值比（LTV）建议

请以JSON格式返回：
{
  "collateral_type": "抵押品类型",
  "estimated_market_value": 市场评估价值,
  "recommended_loan_value": 建议贷款价值,
  "ltv_ratio": LTV比率,
  "liquidity": "high/medium/low",
  "legal_compliance": true/false,
  "risk_factors": [风险因素],
  "assessment_notes": "评估备注",
  "validity_period_months": 评估有效期（月）
}"""

    def _run(
        self,
        company_name: str,
        collateral_type: str,
        estimated_value: float,
        collateral_details: Optional[dict] = None,
    ) -> str:
        ltv = self.LTV_RATIOS.get(collateral_type, 0.30)
        recommended_value = estimated_value * ltv

        details_str = json.dumps(collateral_details or {}, ensure_ascii=False, indent=2)

        user_content = f"""请评估以下担保资产：

企业名称：{company_name}
抵押品类型：{collateral_type}
估计市场价值：{estimated_value:,.0f}元
初步LTV参考：{ltv:.0%}（建议贷款价值：{recommended_value:,.0f}元）

抵押品详情：
{details_str}

请给出专业的资产评估报告。"""

        logger.info(f"Assessing collateral for: {company_name}, type: {collateral_type}")
        result = _call_claude_streaming(
            system=self.SYSTEM_PROMPT,
            user_content=user_content,
            max_tokens=2000,
            use_thinking=False,
        )
        return result
