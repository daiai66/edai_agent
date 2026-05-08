"""
Credit Assessment Agent using CrewAI framework and Claude API.
Analyzes enterprise financial health and outputs structured credit reports.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from tools.financial_tools import (
    AnalyzeFinancialStatements,
    CalculateCreditScore,
    AssessSupplyChain,
    ValidateTaxData,
    _call_claude_streaming,
)
from tools.credit_tools import GenerateCreditReport, MatchLoanProducts


# ─── Output Models ────────────────────────────────────────────────────────────

class RiskAssessment(BaseModel):
    """Risk assessment component of credit report."""
    overall_risk: str = Field(description="low/medium/high/very_high")
    credit_score: int = Field(ge=0, le=1000)
    credit_grade: str = Field(description="AAA/AA/A/BBB/BB/B")
    key_risks: list[str] = Field(default_factory=list)
    mitigating_factors: list[str] = Field(default_factory=list)


class LoanRecommendation(BaseModel):
    """Loan approval recommendation."""
    decision: str = Field(description="approved/conditional/rejected")
    approved_amount: float = Field(ge=0)
    recommended_term_months: int = Field(ge=0)
    interest_rate_suggestion: float = Field(ge=0)
    conditions: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    recommended_product: Optional[str] = None


class CreditReport(BaseModel):
    """Complete credit assessment report."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8].upper())
    company_name: str
    report_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    executive_summary: str = ""

    # Financial analysis
    financial_health_score: float = Field(ge=0, le=100, default=0.0)
    financial_highlights: list[str] = Field(default_factory=list)

    # Risk
    risk_assessment: RiskAssessment

    # Supply chain
    supply_chain_score: float = Field(ge=0, le=100, default=0.0)
    supply_chain_notes: str = ""

    # Tax
    tax_compliance_score: float = Field(ge=0, le=100, default=0.0)

    # Loan recommendation
    loan_recommendation: LoanRecommendation

    # Detailed analysis (raw Claude output)
    financial_analysis_raw: str = ""
    credit_score_analysis_raw: str = ""
    supply_chain_analysis_raw: str = ""
    tax_analysis_raw: str = ""

    # Monitoring
    monitoring_requirements: list[str] = Field(default_factory=list)

    def is_approved(self) -> bool:
        return self.loan_recommendation.decision in ("approved", "conditional")

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        rec = self.loan_recommendation
        decision_cn = {
            "approved": "批准",
            "conditional": "条件批准",
            "rejected": "拒绝",
        }.get(rec.decision, rec.decision)

        return (
            f"信贷报告摘要\n"
            f"{'='*50}\n"
            f"企业: {self.company_name}\n"
            f"报告ID: {self.report_id}\n"
            f"报告日期: {self.report_date}\n"
            f"\n信用评分: {self.risk_assessment.credit_score}/1000 "
            f"({self.risk_assessment.credit_grade})\n"
            f"风险等级: {self.risk_assessment.overall_risk}\n"
            f"财务健康: {self.financial_health_score:.0f}/100\n"
            f"供应链评分: {self.supply_chain_score:.0f}/100\n"
            f"\n审批决定: {decision_cn}\n"
            f"批准额度: {rec.approved_amount:,.0f}元\n"
            f"建议期限: {rec.recommended_term_months}个月\n"
            f"建议利率: {rec.interest_rate_suggestion:.2%}\n"
            f"\n执行摘要: {self.executive_summary}\n"
        )


# ─── Credit Agent ─────────────────────────────────────────────────────────────

class CreditAgent:
    """
    Credit assessment agent using CrewAI and Claude.

    Orchestrates multiple financial analysis tools to produce
    comprehensive credit assessments for SME loan applications.
    """

    SYNTHESIS_PROMPT = """你是一位高级信贷审查官，请综合以下各项分析结果，生成最终信贷决策。

你需要考虑：
1. 财务健康度（流动性、盈利、偿债能力）
2. 信用评分及历史记录
3. 供应链稳定性和融资机会
4. 税务合规情况
5. 行业风险和市场前景

最终决策必须基于全面分析，以JSON格式输出：
{
  "executive_summary": "执行摘要（100字以内）",
  "financial_health_score": 0-100的数字,
  "financial_highlights": ["关键财务亮点"],
  "overall_risk": "low/medium/high/very_high",
  "credit_score": 整数（0-1000）,
  "credit_grade": "AAA/AA/A/BBB/BB/B",
  "key_risks": ["主要风险"],
  "mitigating_factors": ["减轻因素"],
  "supply_chain_score": 0-100的数字,
  "supply_chain_notes": "供应链评价",
  "tax_compliance_score": 0-100的数字,
  "decision": "approved/conditional/rejected",
  "approved_amount": 批准金额（元）,
  "recommended_term_months": 建议期限,
  "interest_rate_suggestion": 建议利率（小数，如0.05表示5%）,
  "recommended_product": "推荐产品名称",
  "conditions": ["批准条件（如有）"],
  "rejection_reasons": ["拒绝原因（如有）"],
  "monitoring_requirements": ["监控要求"]
}"""

    def __init__(self, use_crewai: bool = True):
        self.use_crewai = use_crewai

        # Initialize tools
        self.tools = {
            "financial": AnalyzeFinancialStatements(),
            "credit_score": CalculateCreditScore(),
            "supply_chain": AssessSupplyChain(),
            "tax": ValidateTaxData(),
            "report": GenerateCreditReport(),
            "loan_match": MatchLoanProducts(),
        }

        # Initialize CrewAI agent if available
        self._crew_agent = None
        if use_crewai:
            self._init_crew_agent()

    def _init_crew_agent(self) -> None:
        """Initialize CrewAI agent with credit analysis tools."""
        try:
            from crewai import Agent, LLM

            llm = LLM(
                model=f"anthropic/{settings.anthropic.model}",
                api_key=settings.anthropic.api_key,
                temperature=0.3,
            )

            self._crew_agent = Agent(
                role="高级信贷审查官",
                goal="全面评估中小企业信用状况，为贷款申请提供客观、专业的审查意见",
                backstory=(
                    "你是一位拥有15年银行信贷经验的资深审查官，"
                    "专注于中小微企业融资审查。"
                    "你擅长从财务报表、供应链关系和市场环境中发现风险和机会，"
                    "帮助银行做出明智的信贷决策。"
                ),
                tools=list(self.tools.values()),
                llm=llm,
                verbose=True,
                allow_delegation=False,
                max_iter=10,
            )
            logger.info("CrewAI CreditAgent initialized")
        except Exception as e:
            logger.warning(f"CrewAI agent init failed: {e}. Using direct tool calls.")
            self._crew_agent = None

    def assess(self, enterprise_data: dict) -> CreditReport:
        """
        Perform comprehensive credit assessment for an enterprise.

        Args:
            enterprise_data: Full enterprise data including financials, supply chain, etc.

        Returns:
            CreditReport with complete assessment and recommendation.
        """
        company_name = enterprise_data.get("company_name", "Unknown")
        logger.info(f"Starting credit assessment for: {company_name}")

        # Step 1: Financial analysis
        financial_analysis = self._run_financial_analysis(enterprise_data)

        # Step 2: Credit score
        credit_score_report = self._run_credit_score(enterprise_data)

        # Step 3: Supply chain assessment
        supply_chain_result = self._run_supply_chain_assessment(enterprise_data)

        # Step 4: Tax validation
        tax_result = self._run_tax_validation(enterprise_data)

        # Step 5: Synthesize final report
        final_report = self._synthesize_report(
            enterprise_data=enterprise_data,
            financial_analysis=financial_analysis,
            credit_score_report=credit_score_report,
            supply_chain_result=supply_chain_result,
            tax_result=tax_result,
        )

        logger.info(
            f"Credit assessment complete for {company_name}: "
            f"decision={final_report.loan_recommendation.decision}, "
            f"score={final_report.risk_assessment.credit_score}"
        )
        return final_report

    def _run_financial_analysis(self, enterprise_data: dict) -> str:
        """Run financial statement analysis."""
        try:
            tool = self.tools["financial"]
            result = tool._run(
                company_name=enterprise_data.get("company_name", ""),
                financial_data=enterprise_data.get("financial_data", {}),
                analysis_type="full",
            )
            logger.debug("Financial analysis complete")
            return result
        except Exception as e:
            logger.error(f"Financial analysis failed: {e}")
            return json.dumps({"error": str(e), "status": "failed"})

    def _run_credit_score(self, enterprise_data: dict) -> str:
        """Run credit score calculation."""
        try:
            tool = self.tools["credit_score"]
            result = tool._run(
                company_name=enterprise_data.get("company_name", ""),
                registration_years=enterprise_data.get("registration_years", 0),
                annual_revenue=enterprise_data.get("annual_revenue", 0),
                financial_data=enterprise_data.get("financial_data", {}),
                tax_compliance=enterprise_data.get("tax_compliance", "良好"),
                credit_history=enterprise_data.get("credit_history", "无不良记录"),
                industry=enterprise_data.get("industry", ""),
            )
            logger.debug("Credit score calculation complete")
            return result
        except Exception as e:
            logger.error(f"Credit score calculation failed: {e}")
            return json.dumps({"error": str(e), "status": "failed"})

    def _run_supply_chain_assessment(self, enterprise_data: dict) -> str:
        """Run supply chain assessment if data available."""
        supply_chain = enterprise_data.get("supply_chain")
        if not supply_chain:
            return json.dumps({"status": "skipped", "reason": "No supply chain data"})

        try:
            tool = self.tools["supply_chain"]
            result = tool._run(
                company_name=enterprise_data.get("company_name", ""),
                supply_chain=supply_chain,
                industry=enterprise_data.get("industry", ""),
            )
            logger.debug("Supply chain assessment complete")
            return result
        except Exception as e:
            logger.error(f"Supply chain assessment failed: {e}")
            return json.dumps({"error": str(e), "status": "failed"})

    def _run_tax_validation(self, enterprise_data: dict) -> str:
        """Run tax compliance validation."""
        try:
            tool = self.tools["tax"]
            result = tool._run(
                company_name=enterprise_data.get("company_name", ""),
                tax_compliance=enterprise_data.get("tax_compliance", "未知"),
                annual_revenue=enterprise_data.get("annual_revenue"),
                financial_data=enterprise_data.get("financial_data"),
            )
            logger.debug("Tax validation complete")
            return result
        except Exception as e:
            logger.error(f"Tax validation failed: {e}")
            return json.dumps({"error": str(e), "status": "failed"})

    def _synthesize_report(
        self,
        enterprise_data: dict,
        financial_analysis: str,
        credit_score_report: str,
        supply_chain_result: str,
        tax_result: str,
    ) -> CreditReport:
        """Synthesize all analysis results into a final credit report."""
        company_name = enterprise_data.get("company_name", "Unknown")
        loan_request = enterprise_data.get("loan_request", {})

        user_content = f"""请综合以下分析结果，生成最终信贷决策：

企业：{company_name}
贷款申请金额：{loan_request.get('amount', 0):,.0f}元
贷款用途：{loan_request.get('purpose', '未说明')}
申请期限：{loan_request.get('term_months', 0)}个月

=== 财务分析报告 ===
{financial_analysis}

=== 信用评分报告 ===
{credit_score_report}

=== 供应链评估 ===
{supply_chain_result}

=== 税务合规评估 ===
{tax_result}

请综合以上所有分析，给出最终信贷决策。"""

        try:
            synthesis_raw = _call_claude_streaming(
                system=self.SYNTHESIS_PROMPT,
                user_content=user_content,
                max_tokens=4000,
                use_thinking=True,
            )

            # Parse JSON from Claude response
            report_data = self._extract_json(synthesis_raw)
            return self._build_credit_report(
                company_name=company_name,
                report_data=report_data,
                financial_analysis=financial_analysis,
                credit_score_report=credit_score_report,
                supply_chain_result=supply_chain_result,
                tax_result=tax_result,
            )

        except Exception as e:
            logger.error(f"Report synthesis failed: {e}")
            return self._default_credit_report(company_name, loan_request)

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from Claude response text."""
        import re

        # Try direct JSON parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try extracting JSON block from markdown
        patterns = [
            r"```json\s*([\s\S]+?)\s*```",
            r"```\s*([\s\S]+?)\s*```",
            r"(\{[\s\S]+\})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    continue

        logger.warning("Could not parse JSON from synthesis response")
        return {}

    @staticmethod
    def _build_credit_report(
        company_name: str,
        report_data: dict,
        financial_analysis: str,
        credit_score_report: str,
        supply_chain_result: str,
        tax_result: str,
    ) -> CreditReport:
        """Build CreditReport from parsed data dict."""
        # Risk assessment
        risk = RiskAssessment(
            overall_risk=report_data.get("overall_risk", "medium"),
            credit_score=int(report_data.get("credit_score", 650)),
            credit_grade=report_data.get("credit_grade", "BBB"),
            key_risks=report_data.get("key_risks", []),
            mitigating_factors=report_data.get("mitigating_factors", []),
        )

        # Loan recommendation
        rec = LoanRecommendation(
            decision=report_data.get("decision", "conditional"),
            approved_amount=float(report_data.get("approved_amount", 0)),
            recommended_term_months=int(report_data.get("recommended_term_months", 0)),
            interest_rate_suggestion=float(report_data.get("interest_rate_suggestion", 0.05)),
            conditions=report_data.get("conditions", []),
            rejection_reasons=report_data.get("rejection_reasons", []),
            recommended_product=report_data.get("recommended_product"),
        )

        return CreditReport(
            company_name=company_name,
            executive_summary=report_data.get("executive_summary", ""),
            financial_health_score=float(report_data.get("financial_health_score", 70)),
            financial_highlights=report_data.get("financial_highlights", []),
            risk_assessment=risk,
            supply_chain_score=float(report_data.get("supply_chain_score", 70)),
            supply_chain_notes=report_data.get("supply_chain_notes", ""),
            tax_compliance_score=float(report_data.get("tax_compliance_score", 80)),
            loan_recommendation=rec,
            financial_analysis_raw=financial_analysis,
            credit_score_analysis_raw=credit_score_report,
            supply_chain_analysis_raw=supply_chain_result,
            tax_analysis_raw=tax_result,
            monitoring_requirements=report_data.get("monitoring_requirements", []),
        )

    @staticmethod
    def _default_credit_report(company_name: str, loan_request: dict) -> CreditReport:
        """Return a default conditional credit report when synthesis fails."""
        return CreditReport(
            company_name=company_name,
            executive_summary="由于技术原因，无法完成自动评估。请转人工审核。",
            financial_health_score=60.0,
            risk_assessment=RiskAssessment(
                overall_risk="medium",
                credit_score=650,
                credit_grade="BBB",
                key_risks=["评估系统故障"],
            ),
            supply_chain_score=60.0,
            tax_compliance_score=70.0,
            loan_recommendation=LoanRecommendation(
                decision="conditional",
                approved_amount=loan_request.get("amount", 0) * 0.5,
                recommended_term_months=loan_request.get("term_months", 12),
                interest_rate_suggestion=0.06,
                conditions=["需人工复核所有材料"],
            ),
        )
