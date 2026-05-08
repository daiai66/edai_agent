"""
Dispatcher: Routes planner output to appropriate agents and aggregates results.
Supports parallel execution of independent steps and sequential dependency resolution.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

from pydantic import BaseModel, Field
from loguru import logger

from config.settings import settings
from planner.planner import DispatchPlan, DispatchStep


class StepResult(BaseModel):
    """Result from executing a single dispatch step."""
    step_id: int
    agent: str
    action: str
    result: Any
    duration_seconds: float
    success: bool
    error: Optional[str] = None


class DispatchResult(BaseModel):
    """Complete result from executing a dispatch plan."""
    query: str
    plan_summary: str
    step_results: list[StepResult] = Field(default_factory=list)
    final_response: str = ""
    total_duration_seconds: float = 0.0
    success: bool = True
    errors: list[str] = Field(default_factory=list)

    def get_agent_results(self) -> dict[str, list[Any]]:
        """Get results grouped by agent type."""
        grouped: dict[str, list[Any]] = {}
        for sr in self.step_results:
            if sr.success:
                grouped.setdefault(sr.agent, []).append(sr.result)
        return grouped


class Dispatcher:
    """
    Routes planner output to QA, Credit, and Graph agents.

    Features:
    - Dependency-aware execution ordering
    - Parallel execution for independent steps
    - Result aggregation and synthesis
    - Timeout and error handling
    """

    def __init__(
        self,
        qa_agent: Any,
        credit_agent: Any,
        graph_agent: Any,
        anthropic_client: Optional[Any] = None,
    ):
        self.qa_agent = qa_agent
        self.credit_agent = credit_agent
        self.graph_agent = graph_agent
        self._client = anthropic_client or self._init_client()
        logger.info("Dispatcher initialized with qa/credit/graph agents")

    def _init_client(self) -> Optional[Any]:
        try:
            import anthropic
            return anthropic.Anthropic(api_key=settings.anthropic.api_key)
        except Exception as e:
            logger.warning(f"Dispatcher: client init failed: {e}")
            return None

    async def execute(
        self,
        plan: DispatchPlan,
        context: Optional[dict] = None,
        query: str = "",
    ) -> DispatchResult:
        """
        Execute a dispatch plan asynchronously.

        Respects depends_on for ordering; runs independent steps in parallel.

        Args:
            plan: DispatchPlan from the planner.
            context: Enterprise/loan context dict.
            query: Original user query for final synthesis.

        Returns:
            DispatchResult with all step results and final synthesized response.
        """
        logger.info(f"Dispatcher.execute: {len(plan.steps)} steps")
        start_time = time.time()

        dispatch_result = DispatchResult(
            query=query,
            plan_summary=plan.summary(),
        )

        # Execute steps respecting dependencies
        completed: dict[int, StepResult] = {}
        remaining = list(plan.steps)
        max_rounds = len(plan.steps) + 1

        for _ in range(max_rounds):
            if not remaining:
                break

            # Find steps ready to execute (all deps completed)
            ready = [
                step for step in remaining
                if all(dep in completed for dep in step.depends_on)
            ]

            if not ready:
                logger.warning("No ready steps found — possible circular dependency")
                break

            # Execute ready steps in parallel
            tasks = [self._execute_step(step, context, completed) for step in ready]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for step, result in zip(ready, results):
                if isinstance(result, Exception):
                    step_result = StepResult(
                        step_id=step.step_id,
                        agent=step.agent,
                        action=step.action,
                        result=None,
                        duration_seconds=0.0,
                        success=False,
                        error=str(result),
                    )
                    dispatch_result.errors.append(
                        f"Step {step.step_id} [{step.agent}]: {result}"
                    )
                else:
                    step_result = result

                completed[step.step_id] = step_result
                dispatch_result.step_results.append(step_result)

                # Update plan step status
                plan_step = next((s for s in plan.steps if s.step_id == step.step_id), None)
                if plan_step:
                    plan_step.status = "done" if step_result.success else "failed"
                    plan_step.result = step_result.result

                remaining.remove(step)

        # Synthesize final response
        dispatch_result.final_response = self._synthesize_results(
            query=query,
            plan=plan,
            completed=completed,
        )

        dispatch_result.total_duration_seconds = time.time() - start_time
        dispatch_result.success = len(dispatch_result.errors) == 0

        logger.info(
            f"Dispatcher complete: {len(completed)} steps done, "
            f"{len(dispatch_result.errors)} errors, "
            f"{dispatch_result.total_duration_seconds:.1f}s total"
        )
        return dispatch_result

    def execute_sync(
        self,
        plan: DispatchPlan,
        context: Optional[dict] = None,
        query: str = "",
    ) -> DispatchResult:
        """
        Synchronous wrapper around execute().
        Creates an event loop if none exists.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.execute(plan, context, query))

    async def _execute_step(
        self,
        step: DispatchStep,
        context: Optional[dict],
        completed: dict[int, StepResult],
    ) -> StepResult:
        """Execute a single dispatch step, routing to the appropriate agent."""
        logger.info(f"Executing step {step.step_id}: [{step.agent}] {step.action}")
        start_time = time.time()

        try:
            # Inject results from dependency steps into inputs
            inputs = dict(step.inputs)
            for dep_id in step.depends_on:
                if dep_id in completed and completed[dep_id].success:
                    dep_result = completed[dep_id].result
                    inputs[f"dep_{dep_id}_result"] = dep_result

            # Route to agent
            result = await asyncio.wait_for(
                asyncio.to_thread(self._route_to_agent, step, inputs, context),
                timeout=settings.app.agent_timeout_seconds,
            )

            duration = time.time() - start_time
            logger.info(
                f"Step {step.step_id} [{step.agent}] completed in {duration:.1f}s"
            )

            return StepResult(
                step_id=step.step_id,
                agent=step.agent,
                action=step.action,
                result=result,
                duration_seconds=duration,
                success=True,
            )

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            error = f"Step {step.step_id} timed out after {settings.app.agent_timeout_seconds}s"
            logger.error(error)
            return StepResult(
                step_id=step.step_id,
                agent=step.agent,
                action=step.action,
                result=None,
                duration_seconds=duration,
                success=False,
                error=error,
            )
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Step {step.step_id} [{step.agent}] failed: {e}")
            return StepResult(
                step_id=step.step_id,
                agent=step.agent,
                action=step.action,
                result=None,
                duration_seconds=duration,
                success=False,
                error=str(e),
            )

    def _route_to_agent(
        self,
        step: DispatchStep,
        inputs: dict,
        context: Optional[dict],
    ) -> Any:
        """Route a step to the appropriate agent based on step.agent."""
        agent_type = step.agent.lower()

        if agent_type == "qa":
            return self._run_qa_agent(step, inputs, context)
        elif agent_type == "credit":
            return self._run_credit_agent(step, inputs, context)
        elif agent_type == "graph":
            return self._run_graph_agent(step, inputs, context)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def _run_qa_agent(
        self, step: DispatchStep, inputs: dict, context: Optional[dict]
    ) -> Any:
        """Execute QA agent step."""
        question = inputs.get("question", "")
        if not question and context:
            question = f"请根据企业背景回答：{inputs.get('topic', '信贷政策')}"

        response = self.qa_agent.answer(question=question, context=context)
        return {
            "question": question,
            "answer": response.answer,
            "citations": [c.model_dump() for c in response.citations],
            "confidence": response.confidence,
        }

    def _run_credit_agent(
        self, step: DispatchStep, inputs: dict, context: Optional[dict]
    ) -> Any:
        """Execute Credit agent step."""
        enterprise_data = inputs.get("enterprise_data", context or {})

        # Inject graph analysis results if available
        dep_result = inputs.get("dep_0_result", {})
        if dep_result and isinstance(dep_result, dict):
            enterprise_data = dict(enterprise_data)
            enterprise_data["_graph_enrichment"] = dep_result

        report = self.credit_agent.assess(enterprise_data)
        return report.model_dump()

    def _run_graph_agent(
        self, step: DispatchStep, inputs: dict, context: Optional[dict]
    ) -> Any:
        """Execute Graph agent step."""
        enterprise_info = inputs.get("enterprise_info", context or {})
        analysis = self.graph_agent.analyze(enterprise_info)
        return analysis.model_dump()

    def _synthesize_results(
        self,
        query: str,
        plan: DispatchPlan,
        completed: dict[int, StepResult],
    ) -> str:
        """Synthesize results from all completed steps into a final response."""
        if not completed:
            return "未能获取任何代理结果。"

        # Collect successful results by agent
        agent_results: dict[str, list] = {}
        for step_id, step_result in completed.items():
            if step_result.success and step_result.result is not None:
                agent_results.setdefault(step_result.agent, []).append(step_result.result)

        if not agent_results:
            return "所有代理执行均失败，请检查系统配置。"

        # Try Claude synthesis
        if self._client:
            try:
                return self._claude_synthesis(query, plan, agent_results)
            except Exception as e:
                logger.warning(f"Claude synthesis failed: {e}")

        # Fallback: structured text synthesis
        return self._text_synthesis(query, agent_results, plan)

    def _claude_synthesis(
        self,
        query: str,
        plan: DispatchPlan,
        agent_results: dict[str, list],
    ) -> str:
        """Use Claude to synthesize results into a coherent response."""
        results_text = []
        for agent, results in agent_results.items():
            agent_cn = {"qa": "问答", "credit": "信贷", "graph": "图谱"}.get(agent, agent)
            for i, result in enumerate(results):
                result_str = (
                    json.dumps(result, ensure_ascii=False, indent=2)[:3000]
                    if isinstance(result, dict)
                    else str(result)[:3000]
                )
                results_text.append(f"=== {agent_cn}代理结果 {i+1} ===\n{result_str}")

        user_content = (
            f"用户查询：{query}\n\n"
            f"执行计划综合指导：{plan.final_synthesis}\n\n"
            + "\n\n".join(results_text)
            + "\n\n请综合以上所有代理的分析结果，给出完整、清晰的最终回答。"
        )

        system = """你是一位综合金融分析报告撰写专家。
请基于各代理的分析结果，为用户提供清晰、专业、可操作的最终答案。
回答要结构清晰，包含：
1. 核心结论（一句话）
2. 详细分析
3. 具体建议
4. 注意事项"""

        kwargs = {
            "model": settings.anthropic.model,
            "max_tokens": 4000,
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": "user", "content": user_content}],
            "temperature": 0.3,
        }

        with self._client.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()

        return "\n".join(
            block.text for block in message.content if hasattr(block, "text")
        )

    @staticmethod
    def _text_synthesis(
        query: str,
        agent_results: dict[str, list],
        plan: DispatchPlan,
    ) -> str:
        """Fallback text-based synthesis without LLM."""
        parts = [f"## 查询：{query}\n"]

        if "graph" in agent_results:
            graph_result = agent_results["graph"][0]
            if isinstance(graph_result, dict):
                industry = graph_result.get("industry", "")
                risk = graph_result.get("industry_risk_profile", {}).get("risk_level", "medium")
                parts.append(f"### 行业分析\n- 行业：{industry}\n- 风险等级：{risk}")

        if "credit" in agent_results:
            credit_result = agent_results["credit"][0]
            if isinstance(credit_result, dict):
                rec = credit_result.get("loan_recommendation", {})
                risk = credit_result.get("risk_assessment", {})
                decision_cn = {"approved": "批准", "conditional": "条件批准", "rejected": "拒绝"}.get(
                    rec.get("decision", ""), rec.get("decision", "待定")
                )
                parts.append(
                    f"\n### 信贷评估\n"
                    f"- 信用评分：{risk.get('credit_score', 'N/A')}/1000\n"
                    f"- 风险等级：{risk.get('overall_risk', 'N/A')}\n"
                    f"- 审批决定：{decision_cn}\n"
                    f"- 批准金额：{rec.get('approved_amount', 0):,.0f}元\n"
                    f"- 建议期限：{rec.get('recommended_term_months', 0)}个月\n"
                    f"- 建议利率：{rec.get('interest_rate_suggestion', 0):.2%}"
                )

        if "qa" in agent_results:
            qa_result = agent_results["qa"][0]
            if isinstance(qa_result, dict):
                parts.append(f"\n### 政策参考\n{qa_result.get('answer', '')[:500]}")

        return "\n".join(parts)
