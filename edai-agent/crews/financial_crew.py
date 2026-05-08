"""
CrewAI financial analysis crew with three specialized agents:
- Researcher: Gathers industry/market intelligence
- Credit Analyst: Performs credit scoring and risk assessment
- Report Writer: Synthesizes final approval report
"""
from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, Field
from loguru import logger

from config.settings import settings
from tools.financial_tools import (
    AnalyzeFinancialStatements,
    CalculateCreditScore,
    AssessSupplyChain,
    ValidateTaxData,
    _call_claude_streaming,
)
from tools.credit_tools import GenerateCreditReport, MatchLoanProducts
from tools.graph_tools import (
    QueryIndustryKnowledge,
    FindSupplyChainRelations,
    GetCompetitorAnalysis,
)


# ─── Output Model ─────────────────────────────────────────────────────────────

class ApprovalReport(BaseModel):
    """Final loan approval report from the financial crew."""
    company_name: str
    loan_amount_requested: float
    loan_amount_approved: float
    decision: str  # approved/conditional/rejected
    credit_score: int
    risk_level: str
    interest_rate: float
    term_months: int
    recommended_product: str = ""
    conditions: list[str] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    researcher_summary: str = ""
    analyst_summary: str = ""
    writer_summary: str = ""
    executive_summary: str = ""

    def is_approved(self) -> bool:
        return self.decision in ("approved", "conditional")

    def print_report(self) -> None:
        """Print formatted report to stdout."""
        decision_cn = {"approved": "批准", "conditional": "条件批准", "rejected": "拒绝"}.get(
            self.decision, self.decision
        )
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║              SME 贷款审批报告                                  ║
╠══════════════════════════════════════════════════════════════╣
║ 企业名称: {self.company_name:<50} ║
║ 申请金额: {self.loan_amount_requested:>15,.0f} 元                        ║
╠══════════════════════════════════════════════════════════════╣
║ 审批决定: {decision_cn:<51} ║
║ 批准金额: {self.loan_amount_approved:>15,.0f} 元                        ║
║ 信用评分: {self.credit_score}/1000                                    ║
║ 风险等级: {self.risk_level:<51} ║
║ 建议利率: {self.interest_rate:.2%}                                      ║
║ 贷款期限: {self.term_months}个月                                       ║
║ 推荐产品: {self.recommended_product:<50} ║
╠══════════════════════════════════════════════════════════════╣
║ 执行摘要:                                                     ║
╚══════════════════════════════════════════════════════════════╝
{self.executive_summary}

关键发现:
{chr(10).join(f'  • {f}' for f in self.key_findings)}

审批条件:
{chr(10).join(f'  • {c}' for c in self.conditions) if self.conditions else '  无附加条件'}
""")


# ─── Crew ─────────────────────────────────────────────────────────────────────

class FinancialAnalysisCrew:
    """
    CrewAI-based financial analysis crew.

    Three agents work collaboratively:
    1. **Researcher** (researcher): Gathers industry knowledge and supply chain info
    2. **Credit Analyst** (credit_analyst): Performs financial analysis and scoring
    3. **Report Writer** (report_writer): Synthesizes final approval report

    Falls back to direct tool calls if CrewAI is unavailable.
    """

    def __init__(
        self,
        use_crewai: bool = True,
        anthropic_client: Optional[Any] = None,
    ):
        self.use_crewai = use_crewai
        self._client = anthropic_client or self._init_client()

        # Initialize tools
        self._tools = {
            "financial": AnalyzeFinancialStatements(),
            "credit_score": CalculateCreditScore(),
            "supply_chain": AssessSupplyChain(),
            "tax": ValidateTaxData(),
            "report": GenerateCreditReport(),
            "loan_match": MatchLoanProducts(),
            "industry": QueryIndustryKnowledge(),
            "sc_relations": FindSupplyChainRelations(),
            "competitor": GetCompetitorAnalysis(),
        }

        # Initialize CrewAI if requested
        self._crew = None
        if use_crewai:
            self._init_crew()

    def _init_client(self) -> Optional[Any]:
        try:
            import anthropic
            return anthropic.Anthropic(api_key=settings.anthropic.api_key)
        except Exception as e:
            logger.warning(f"FinancialCrew: client init failed: {e}")
            return None

    def _init_crew(self) -> None:
        """Initialize CrewAI crew with 3 agents and their tasks."""
        try:
            from crewai import Agent, Crew, Task, Process, LLM

            llm = LLM(
                model=f"anthropic/{settings.anthropic.model}",
                api_key=settings.anthropic.api_key,
                temperature=0.3,
                max_tokens=4000,
            )

            # Agent 1: Researcher
            researcher = Agent(
                role="行业研究员",
                goal="收集目标企业所在行业的风险信息、供应链关系和竞争态势，为信贷决策提供行业背景",
                backstory=(
                    "你是一位专注于中小企业行业研究的分析师，"
                    "熟悉农业、制造业、科技等各行业的风险特征和发展趋势。"
                    "你能从行业数据中识别潜在风险和融资机会。"
                ),
                tools=[
                    self._tools["industry"],
                    self._tools["sc_relations"],
                    self._tools["competitor"],
                ],
                llm=llm,
                verbose=True,
                allow_delegation=False,
                max_iter=5,
            )

            # Agent 2: Credit Analyst
            credit_analyst = Agent(
                role="信贷分析师",
                goal="深入分析企业财务状况、信用历史和偿债能力，给出精准的信用评分和风险评估",
                backstory=(
                    "你是一位拥有10年信贷审查经验的专业分析师，"
                    "擅长从复杂的财务数据中发现风险信号。"
                    "你的评估客观公正，既保护银行利益，也支持优质中小企业融资。"
                ),
                tools=[
                    self._tools["financial"],
                    self._tools["credit_score"],
                    self._tools["supply_chain"],
                    self._tools["tax"],
                ],
                llm=llm,
                verbose=True,
                allow_delegation=False,
                max_iter=8,
            )

            # Agent 3: Report Writer
            report_writer = Agent(
                role="信贷报告撰写专家",
                goal="综合研究员和分析师的成果，撰写清晰、专业的信贷审批报告，给出最终审批建议",
                backstory=(
                    "你是一位资深信贷报告专家，"
                    "能够将复杂的金融分析转化为清晰易懂的审批报告。"
                    "你的报告既满足合规要求，又帮助决策者快速理解核心风险和建议。"
                ),
                tools=[
                    self._tools["report"],
                    self._tools["loan_match"],
                ],
                llm=llm,
                verbose=True,
                allow_delegation=False,
                max_iter=5,
            )

            self._agents = {
                "researcher": researcher,
                "credit_analyst": credit_analyst,
                "report_writer": report_writer,
            }

            logger.info("CrewAI crew agents initialized")

        except Exception as e:
            logger.warning(f"CrewAI init failed: {e}. Will use direct tool execution.")
            self._crew = None
            self._agents = {}

    def process_loan_application(self, enterprise_data: dict) -> ApprovalReport:
        """
        Process a complete SME loan application.

        Args:
            enterprise_data: Full enterprise data with financials, supply chain, loan request.

        Returns:
            ApprovalReport with final decision and detailed findings.
        """
        company_name = enterprise_data.get("company_name", "Unknown")
        logger.info(f"FinancialCrew: Processing loan application for {company_name}")

        # Phase 1: Research (industry + supply chain + competitors)
        researcher_output = self._run_research_phase(enterprise_data)
        logger.info("Research phase complete")

        # Phase 2: Credit Analysis
        analyst_output = self._run_analysis_phase(enterprise_data, researcher_output)
        logger.info("Analysis phase complete")

        # Phase 3: Report Writing
        report = self._run_report_phase(enterprise_data, researcher_output, analyst_output)
        logger.info(f"Report phase complete: decision={report.decision}")

        return report

    def _run_research_phase(self, enterprise_data: dict) -> dict:
        """Run the researcher agent to gather industry intelligence."""
        industry = enterprise_data.get("industry", "")
        supply_chain = enterprise_data.get("supply_chain", {})

        results = {}

        # Industry knowledge
        try:
            if self._agents.get("researcher") and self.use_crewai:
                results["industry"] = self._run_crewai_task(
                    agent=self._agents["researcher"],
                    description=f"分析{industry}行业的风险特征、政策支持和融资机会",
                    expected_output="行业风险报告JSON",
                    context={"enterprise_data": enterprise_data},
                )
            else:
                results["industry"] = self._tools["industry"]._run(
                    industry=industry, query_type="full"
                )
        except Exception as e:
            logger.warning(f"Research phase - industry failed: {e}")
            results["industry"] = f"行业分析失败: {e}"

        # Supply chain
        try:
            if supply_chain:
                results["supply_chain"] = self._tools["sc_relations"]._run(
                    company_name=enterprise_data.get("company_name", ""),
                    supply_chain_data=supply_chain,
                )
        except Exception as e:
            logger.warning(f"Research phase - supply chain failed: {e}")

        # Competitor
        try:
            results["competitor"] = self._tools["competitor"]._run(
                company_name=enterprise_data.get("company_name", ""),
                industry=industry,
                annual_revenue=enterprise_data.get("annual_revenue", 0),
            )
        except Exception as e:
            logger.warning(f"Research phase - competitor failed: {e}")

        return results

    def _run_analysis_phase(self, enterprise_data: dict, researcher_output: dict) -> dict:
        """Run the credit analyst to perform financial and credit analysis."""
        results = {}

        # Financial analysis
        try:
            if self._agents.get("credit_analyst") and self.use_crewai:
                results["financial"] = self._run_crewai_task(
                    agent=self._agents["credit_analyst"],
                    description="分析企业财务报表，评估财务健康状况",
                    expected_output="财务分析报告JSON",
                    context={"enterprise_data": enterprise_data},
                )
            else:
                results["financial"] = self._tools["financial"]._run(
                    company_name=enterprise_data.get("company_name", ""),
                    financial_data=enterprise_data.get("financial_data", {}),
                )
        except Exception as e:
            logger.warning(f"Analysis phase - financial failed: {e}")
            results["financial"] = f"财务分析失败: {e}"

        # Credit score
        try:
            results["credit_score"] = self._tools["credit_score"]._run(
                company_name=enterprise_data.get("company_name", ""),
                registration_years=enterprise_data.get("registration_years", 0),
                annual_revenue=enterprise_data.get("annual_revenue", 0),
                financial_data=enterprise_data.get("financial_data", {}),
                tax_compliance=enterprise_data.get("tax_compliance", "良好"),
                credit_history=enterprise_data.get("credit_history", "无不良记录"),
                industry=enterprise_data.get("industry", ""),
            )
        except Exception as e:
            logger.warning(f"Analysis phase - credit score failed: {e}")
            results["credit_score"] = f"信用评分失败: {e}"

        # Tax validation
        try:
            results["tax"] = self._tools["tax"]._run(
                company_name=enterprise_data.get("company_name", ""),
                tax_compliance=enterprise_data.get("tax_compliance", "良好"),
                annual_revenue=enterprise_data.get("annual_revenue"),
            )
        except Exception as e:
            logger.warning(f"Analysis phase - tax failed: {e}")

        # Supply chain
        if enterprise_data.get("supply_chain"):
            try:
                results["supply_chain"] = self._tools["supply_chain"]._run(
                    company_name=enterprise_data.get("company_name", ""),
                    supply_chain=enterprise_data.get("supply_chain", {}),
                    industry=enterprise_data.get("industry", ""),
                )
            except Exception as e:
                logger.warning(f"Analysis phase - supply chain failed: {e}")

        return results

    def _run_report_phase(
        self,
        enterprise_data: dict,
        researcher_output: dict,
        analyst_output: dict,
    ) -> ApprovalReport:
        """Synthesize all analyses into final approval report."""
        company_name = enterprise_data.get("company_name", "Unknown")
        loan_request = enterprise_data.get("loan_request", {})

        try:
            if self._agents.get("report_writer") and self.use_crewai:
                synthesis = self._run_crewai_task(
                    agent=self._agents["report_writer"],
                    description="综合所有分析结果，生成最终信贷审批报告",
                    expected_output="完整信贷报告JSON",
                    context={
                        "enterprise_data": enterprise_data,
                        "researcher_output": researcher_output,
                        "analyst_output": analyst_output,
                    },
                )
            else:
                synthesis = self._direct_synthesis(
                    enterprise_data, researcher_output, analyst_output
                )

            return self._build_approval_report(
                company_name=company_name,
                loan_request=loan_request,
                synthesis=synthesis,
                researcher_output=researcher_output,
                analyst_output=analyst_output,
            )

        except Exception as e:
            logger.error(f"Report phase failed: {e}")
            return self._default_approval_report(company_name, loan_request)

    def _direct_synthesis(
        self,
        enterprise_data: dict,
        researcher_output: dict,
        analyst_output: dict,
    ) -> str:
        """Direct synthesis using Claude when CrewAI is unavailable."""
        all_context = {
            "enterprise": enterprise_data,
            "research": researcher_output,
            "analysis": analyst_output,
        }

        user_content = f"""请综合以下所有分析结果，生成最终信贷审批报告：

{json.dumps(all_context, ensure_ascii=False, indent=2)[:6000]}

请以JSON格式输出审批报告，包含：
- decision (approved/conditional/rejected)
- approved_amount (元)
- credit_score (0-1000)
- risk_level (low/medium/high)
- interest_rate (小数)
- term_months
- recommended_product
- conditions (附加条件列表)
- key_findings (关键发现列表)
- executive_summary (执行摘要)"""

        return _call_claude_streaming(
            system="你是一位信贷审批专家，请基于综合分析给出审批报告。",
            user_content=user_content,
            max_tokens=3000,
            use_thinking=True,
        )

    @staticmethod
    def _run_crewai_task(
        agent: Any,
        description: str,
        expected_output: str,
        context: dict,
    ) -> str:
        """Run a single CrewAI task with a single agent."""
        try:
            from crewai import Task, Crew, Process

            task = Task(
                description=description + f"\n\n上下文：{json.dumps(context, ensure_ascii=False)[:2000]}",
                expected_output=expected_output,
                agent=agent,
            )
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
            result = crew.kickoff()
            return str(result)
        except Exception as e:
            logger.warning(f"CrewAI task failed: {e}")
            raise

    @staticmethod
    def _build_approval_report(
        company_name: str,
        loan_request: dict,
        synthesis: str,
        researcher_output: dict,
        analyst_output: dict,
    ) -> ApprovalReport:
        """Parse synthesis into ApprovalReport."""
        import re

        data = {}
        for pattern in [r"```json\s*([\s\S]+?)\s*```", r"(\{[\s\S]+\})"]:
            match = re.search(pattern, synthesis)
            if match:
                try:
                    data = json.loads(match.group(1))
                    break
                except Exception:
                    continue

        if not data:
            try:
                data = json.loads(synthesis)
            except Exception:
                data = {}

        requested_amount = loan_request.get("amount", 0)
        approved = float(data.get("approved_amount", requested_amount * 0.8))

        return ApprovalReport(
            company_name=company_name,
            loan_amount_requested=requested_amount,
            loan_amount_approved=approved,
            decision=data.get("decision", "conditional"),
            credit_score=int(data.get("credit_score", 700)),
            risk_level=data.get("risk_level", "medium"),
            interest_rate=float(data.get("interest_rate", 0.05)),
            term_months=int(data.get("term_months", loan_request.get("term_months", 24))),
            recommended_product=data.get("recommended_product", ""),
            conditions=data.get("conditions", []),
            key_findings=data.get("key_findings", []),
            researcher_summary=str(researcher_output.get("industry", ""))[:500],
            analyst_summary=str(analyst_output.get("financial", ""))[:500],
            executive_summary=data.get("executive_summary", ""),
        )

    @staticmethod
    def _default_approval_report(company_name: str, loan_request: dict) -> ApprovalReport:
        requested = loan_request.get("amount", 0)
        return ApprovalReport(
            company_name=company_name,
            loan_amount_requested=requested,
            loan_amount_approved=requested * 0.7,
            decision="conditional",
            credit_score=680,
            risk_level="medium",
            interest_rate=0.055,
            term_months=loan_request.get("term_months", 24),
            conditions=["需人工复核所有材料", "提供最近3个月银行流水"],
            key_findings=["系统自动评估（需人工确认）"],
            executive_summary="由于技术原因，本报告为系统自动生成，需人工复核。",
        )
