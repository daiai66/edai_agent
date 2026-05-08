"""
CrewAI tools for financial analysis using Claude API with adaptive thinking and streaming.
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

try:
    from crewai.tools import BaseTool
except ImportError:
    # Fallback base class if crewai not installed
    class BaseTool:
        name: str = ""
        description: str = ""

        def _run(self, *args, **kwargs):
            raise NotImplementedError

        def run(self, *args, **kwargs):
            return self._run(*args, **kwargs)


def _get_anthropic_client():
    """Get or create Anthropic client."""
    try:
        import anthropic
        return anthropic.Anthropic(api_key=settings.anthropic.api_key)
    except Exception as e:
        logger.error(f"Failed to create Anthropic client: {e}")
        return None


@retry(
    stop=stop_after_attempt(settings.app.max_retries),
    wait=wait_exponential(
        multiplier=settings.app.retry_multiplier,
        min=settings.app.retry_wait_min,
        max=settings.app.retry_wait_max,
    ),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_claude_streaming(
    system: str,
    user_content: str,
    max_tokens: int = 4096,
    use_thinking: bool = True,
) -> str:
    """
    Call Claude API with adaptive thinking and streaming.
    Returns the final text response.
    """
    try:
        import anthropic
        client = _get_anthropic_client()
        if client is None:
            return _mock_claude_response(user_content)

        messages = [{"role": "user", "content": user_content}]

        kwargs = {
            "model": settings.anthropic.model,
            "max_tokens": max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": messages,
        }

        if use_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        else:
            kwargs["output_config"] = {"effort": "low"}

        with client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()

        # Extract text content from response
        text_parts = []
        for block in message.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "\n".join(text_parts)

    except Exception as e:
        if "RateLimitError" in type(e).__name__ or "rate_limit" in str(e).lower():
            logger.warning(f"Rate limit hit: {e}")
            raise
        logger.warning(f"Claude API call failed: {e}. Using mock response.")
        return _mock_claude_response(user_content)


def _mock_claude_response(prompt: str) -> str:
    """Generate a mock response when API is unavailable."""
    return json.dumps({
        "status": "mock_response",
        "note": "Claude API not available. This is a simulated response.",
        "analysis": "Based on the provided data, the financial indicators appear stable.",
        "recommendation": "Further manual review recommended.",
    }, ensure_ascii=False)


# ─── Input / Output Schemas ──────────────────────────────────────────────────

class FinancialStatementInput(BaseModel):
    """Input for financial statement analysis."""
    company_name: str = Field(description="Company name")
    financial_data: dict = Field(description="Financial data by year")
    analysis_type: str = Field(
        default="full",
        description="Type of analysis: 'full', 'liquidity', 'profitability', 'solvency'"
    )


class CreditScoreInput(BaseModel):
    """Input for credit score calculation."""
    company_name: str = Field(description="Company name")
    registration_years: float = Field(description="Years since registration")
    annual_revenue: float = Field(description="Annual revenue in CNY")
    financial_data: dict = Field(description="Financial data by year")
    tax_compliance: str = Field(default="良好", description="Tax compliance status")
    credit_history: str = Field(default="无不良记录", description="Credit history")
    industry: str = Field(default="", description="Industry sector")


class SupplyChainInput(BaseModel):
    """Input for supply chain assessment."""
    company_name: str = Field(description="Company name")
    supply_chain: dict = Field(description="Supply chain data")
    industry: str = Field(default="", description="Industry sector")


class TaxDataInput(BaseModel):
    """Input for tax data validation."""
    company_name: str = Field(description="Company name")
    tax_compliance: str = Field(description="Tax compliance status string")
    annual_revenue: Optional[float] = Field(default=None, description="Declared revenue")
    financial_data: Optional[dict] = Field(default=None, description="Financial statements")


# ─── Financial Analysis Tools ────────────────────────────────────────────────

class AnalyzeFinancialStatements(BaseTool):
    """
    Analyze enterprise financial statements to assess financial health.
    Uses Claude with adaptive thinking for deep financial analysis.
    """
    name: str = "analyze_financial_statements"
    description: str = (
        "Analyzes enterprise financial statements (balance sheet, income statement, cash flow) "
        "to assess financial health, liquidity, profitability, and solvency ratios. "
        "Input: JSON with company_name, financial_data (by year). "
        "Output: Comprehensive financial analysis with risk assessment."
    )
    args_schema: Type[BaseModel] = FinancialStatementInput

    SYSTEM_PROMPT: ClassVar[str] = """你是一位专业的中小企业财务分析师，拥有超过15年的企业信贷评估经验。
你的任务是对企业财务报表进行深度分析，评估企业的财务健康状况。

分析维度包括：
1. 流动性分析：流动比率、速动比率、现金比率
2. 盈利能力分析：毛利率、净利率、ROE、ROA
3. 偿债能力分析：资产负债率、利息保障倍数
4. 成长性分析：收入增长率、利润增长率
5. 综合风险评级

请以JSON格式返回分析结果，包含：
- financial_ratios: 各项财务比率
- risk_level: 风险等级 (low/medium/high/very_high)
- strengths: 财务优势列表
- weaknesses: 财务劣势列表
- recommendations: 建议事项
- overall_score: 综合评分 (0-100)"""

    def _run(self, company_name: str, financial_data: dict, analysis_type: str = "full") -> str:
        """Execute financial statement analysis."""
        user_content = f"""请分析以下企业的财务报表：

企业名称：{company_name}
分析类型：{analysis_type}

财务数据：
{json.dumps(financial_data, ensure_ascii=False, indent=2)}

请进行全面的财务分析，重点关注企业的信贷风险。"""

        logger.info(f"Analyzing financial statements for: {company_name}")
        result = _call_claude_streaming(
            system=self.SYSTEM_PROMPT,
            user_content=user_content,
            max_tokens=4096,
            use_thinking=True,
        )
        logger.info(f"Financial analysis complete for: {company_name}")
        return result


class CalculateCreditScore(BaseTool):
    """
    Calculate enterprise credit score (0-1000) based on comprehensive data.
    Uses Claude to synthesize multiple risk dimensions into a credit score.
    """
    name: str = "calculate_credit_score"
    description: str = (
        "Calculates enterprise credit score (0-1000 scale) based on financial data, "
        "registration history, tax compliance, and credit history. "
        "Output: Credit score with detailed scoring breakdown."
    )
    args_schema: Type[BaseModel] = CreditScoreInput

    SYSTEM_PROMPT: ClassVar[str] = """你是一位专业的企业信用评分专家，负责为中小微企业计算信用评分。

信用评分满分为1000分，评分维度及权重如下：
1. 财务健康度 (30%)：基于财务报表的盈利、流动性、偿债能力
2. 经营稳定性 (20%)：经营年限、收入稳定性
3. 税务合规性 (20%)：税务记录、申报准确性
4. 信用历史 (20%)：历史贷款记录、违约情况
5. 行业风险 (10%)：所在行业的系统性风险

评分标准：
- 900-1000: 优质 (AAA)
- 800-899: 良好 (AA)
- 700-799: 一般 (A)
- 600-699: 及格 (BBB)
- 500-599: 关注 (BB)
- <500: 较差 (B及以下)

请以JSON格式返回：
{
  "total_score": 0-1000的整数,
  "grade": "AAA/AA/A/BBB/BB/B",
  "dimension_scores": {各维度得分},
  "score_breakdown": {详细评分说明},
  "risk_factors": [主要风险因素],
  "positive_factors": [加分因素],
  "recommendation": "loan_approved/loan_conditional/loan_rejected"
}"""

    def _run(
        self,
        company_name: str,
        registration_years: float,
        annual_revenue: float,
        financial_data: dict,
        tax_compliance: str = "良好",
        credit_history: str = "无不良记录",
        industry: str = "",
    ) -> str:
        user_content = f"""请为以下企业计算信用评分：

企业名称：{company_name}
行业：{industry}
注册年限：{registration_years}年
年营业额：{annual_revenue:,.0f}元
税务合规：{tax_compliance}
信用历史：{credit_history}

财务数据：
{json.dumps(financial_data, ensure_ascii=False, indent=2)}

请依据评分体系给出完整的信用评分报告。"""

        logger.info(f"Calculating credit score for: {company_name}")
        result = _call_claude_streaming(
            system=self.SYSTEM_PROMPT,
            user_content=user_content,
            max_tokens=3000,
            use_thinking=True,
        )
        return result


class AssessSupplyChain(BaseTool):
    """
    Assess enterprise supply chain health and supply chain finance eligibility.
    """
    name: str = "assess_supply_chain"
    description: str = (
        "Evaluates enterprise supply chain health including customer/supplier concentration, "
        "payment cycles, and supply chain finance eligibility. "
        "Output: Supply chain health score and financing recommendations."
    )
    args_schema: Type[BaseModel] = SupplyChainInput

    SYSTEM_PROMPT: ClassVar[str] = """你是一位供应链金融专家，负责评估中小企业的供应链健康状况及供应链融资可行性。

评估维度：
1. 核心客户集中度：前5大客户占比
2. 应收账款质量：账期、逾期率
3. 供应商稳定性：供应商数量、合作年限
4. 供应链协议质量：是否有长期合作协议
5. 供应链融资资格：基于核心企业信用的保理融资可行性

请以JSON格式返回：
{
  "supply_chain_score": 0-100整数,
  "customer_concentration_risk": "low/medium/high",
  "supplier_stability": "stable/moderate/unstable",
  "factoring_eligibility": true/false,
  "max_factoring_amount": 最高保理额度（元）,
  "avg_payment_days_assessment": "优/良/一般/差",
  "key_risks": [关键风险],
  "opportunities": [融资机会],
  "recommendations": [建议]
}"""

    def _run(self, company_name: str, supply_chain: dict, industry: str = "") -> str:
        user_content = f"""请评估以下企业的供应链状况：

企业名称：{company_name}
行业：{industry}

供应链数据：
{json.dumps(supply_chain, ensure_ascii=False, indent=2)}

请给出供应链健康评估和供应链融资建议。"""

        logger.info(f"Assessing supply chain for: {company_name}")
        result = _call_claude_streaming(
            system=self.SYSTEM_PROMPT,
            user_content=user_content,
            max_tokens=2000,
            use_thinking=False,
        )
        return result


class ValidateTaxData(BaseTool):
    """
    Validate enterprise tax compliance and assess tax-related financial risk.
    """
    name: str = "validate_tax_data"
    description: str = (
        "Validates enterprise tax records and compliance, checking for consistency "
        "between declared revenues and financial statements. "
        "Output: Tax compliance assessment with risk flags."
    )
    args_schema: Type[BaseModel] = TaxDataInput

    SYSTEM_PROMPT: ClassVar[str] = """你是一位专业的税务合规审查员，负责评估企业的税务合规状况。

审查要点：
1. 税务申报记录完整性
2. 申报收入与财务报表一致性
3. 纳税信用等级
4. 是否存在欠税、逃税风险
5. 增值税、企业所得税合规性

风险标志（Red Flags）：
- 申报收入与银行流水严重不符
- 连续多年亏损但企业仍正常经营
- 税务局列为重点监控对象
- 历史逃税或欠税记录

请以JSON格式返回：
{
  "tax_compliance_score": 0-100整数,
  "compliance_level": "excellent/good/fair/poor",
  "red_flags": [风险标志列表],
  "revenue_consistency": "consistent/minor_discrepancy/major_discrepancy",
  "tax_credit_rating": "A/B/C/D/M",
  "recommendations": [建议],
  "overall_risk": "low/medium/high"
}"""

    def _run(
        self,
        company_name: str,
        tax_compliance: str,
        annual_revenue: Optional[float] = None,
        financial_data: Optional[dict] = None,
    ) -> str:
        revenue_info = f"{annual_revenue:,.0f}元" if annual_revenue else "未提供"
        fin_info = json.dumps(financial_data, ensure_ascii=False, indent=2) if financial_data else "未提供"

        user_content = f"""请评估以下企业的税务合规状况：

企业名称：{company_name}
税务合规描述：{tax_compliance}
年营业额：{revenue_info}

财务数据参考：
{fin_info}

请给出详细的税务合规评估报告。"""

        logger.info(f"Validating tax data for: {company_name}")
        result = _call_claude_streaming(
            system=self.SYSTEM_PROMPT,
            user_content=user_content,
            max_tokens=2000,
            use_thinking=False,
        )
        return result
