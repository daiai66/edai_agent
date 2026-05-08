"""
Rule-based post-filtering for SME financial document retrieval.
Applies business rules to filter and score retrieved documents.
"""
from __future__ import annotations

from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from config.settings import settings


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


RISK_RANK = {
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.VERY_HIGH: 4,
}


@dataclass
class FilterContext:
    """Context information used to apply business rules."""
    # Enterprise profile
    industry: Optional[str] = None
    registration_years: Optional[float] = None
    annual_revenue: Optional[float] = None
    credit_score: Optional[int] = None

    # Loan request
    loan_amount: Optional[float] = None
    loan_purpose: Optional[str] = None

    # Risk tolerance
    max_risk_level: str = "high"

    # Document filtering
    required_doc_types: list[str] = field(default_factory=list)
    preferred_industries: list[str] = field(default_factory=list)

    @classmethod
    def from_enterprise_data(cls, enterprise_data: dict) -> "FilterContext":
        """Build FilterContext from raw enterprise data dict."""
        loan = enterprise_data.get("loan_request", {})
        return cls(
            industry=enterprise_data.get("industry"),
            registration_years=enterprise_data.get("registration_years"),
            annual_revenue=enterprise_data.get("annual_revenue"),
            credit_score=enterprise_data.get("credit_score"),
            loan_amount=loan.get("amount"),
            loan_purpose=loan.get("purpose"),
        )


@dataclass
class FilterResult:
    """Result of applying a single rule."""
    rule_name: str
    passed: bool
    reason: str
    penalty_score: float = 0.0  # Score reduction applied when rule fails softly


class RuleFilter:
    """
    Business rule filter for retrieved documents.
    Rules include:
    - Industry whitelist check
    - Risk level threshold
    - Loan amount range validation
    - Enterprise credit score minimum
    - Enterprise age requirement
    - Revenue threshold
    """

    def __init__(self, config: Optional[Any] = None):
        cfg = config or settings.rule_filter
        self.industry_whitelist: set[str] = set(cfg.industry_whitelist)
        self.max_risk_level: str = cfg.max_risk_level
        self.credit_score_min: int = cfg.credit_score_min
        self.enterprise_age_min: float = cfg.enterprise_age_min
        self.revenue_threshold: float = cfg.revenue_threshold_micro
        self.max_loan_amount: float = cfg.max_loan_amount
        self.min_loan_amount: float = cfg.min_loan_amount

        # Custom rules registry
        self._custom_rules: list[callable] = []

        logger.debug(
            f"RuleFilter initialized: industry_whitelist={len(self.industry_whitelist)}, "
            f"max_risk={self.max_risk_level}"
        )

    def add_custom_rule(self, rule_fn: callable) -> None:
        """Add a custom rule function(doc, context) -> FilterResult."""
        self._custom_rules.append(rule_fn)

    def filter(
        self,
        results: list[Any],
        context: Optional[FilterContext | dict] = None,
    ) -> list[Any]:
        """
        Filter and re-score a list of retrieved documents based on business rules.

        Args:
            results: List of Document objects or dicts.
            context: FilterContext or raw dict with enterprise/loan data.

        Returns:
            Filtered list of documents, possibly with adjusted scores.
        """
        if context is None:
            context = FilterContext()
        elif isinstance(context, dict):
            context = FilterContext.from_enterprise_data(context)

        filtered = []
        for doc in results:
            rule_results = self._apply_all_rules(doc, context)
            passed, score_adj = self._evaluate_rules(rule_results)
            if passed:
                # Adjust score by rule penalties
                if hasattr(doc, "score"):
                    doc.score = max(0.0, doc.score + score_adj)
                filtered.append(doc)
            else:
                logger.debug(f"Document {getattr(doc, 'id', '?')} filtered out by rules")

        # Re-sort by adjusted score
        filtered.sort(key=lambda d: getattr(d, "score", 0.0), reverse=True)
        logger.debug(
            f"Rule filter: {len(results)} -> {len(filtered)} documents"
        )
        return filtered

    def _apply_all_rules(
        self, doc: Any, context: FilterContext
    ) -> list[FilterResult]:
        """Apply all rules to a single document."""
        results: list[FilterResult] = []
        metadata = getattr(doc, "metadata", {}) if not isinstance(doc, dict) else doc.get("metadata", {})

        results.append(self._check_industry(metadata, context))
        results.append(self._check_risk_level(metadata, context))
        results.append(self._check_credit_score(metadata, context))
        results.append(self._check_enterprise_age(metadata, context))
        results.append(self._check_revenue(metadata, context))
        results.append(self._check_loan_amount(metadata, context))

        for custom_rule in self._custom_rules:
            try:
                results.append(custom_rule(doc, context))
            except Exception as e:
                logger.warning(f"Custom rule failed: {e}")
                results.append(FilterResult(
                    rule_name="custom",
                    passed=True,
                    reason="Rule execution error (treated as pass)",
                ))

        return results

    def _evaluate_rules(
        self, rule_results: list[FilterResult]
    ) -> tuple[bool, float]:
        """
        Evaluate rule results to determine pass/fail and score adjustment.

        Returns:
            (passed, score_adjustment) where score_adjustment is negative on penalties.
        """
        score_adj = 0.0
        for result in rule_results:
            if not result.passed and result.penalty_score < 0:
                # Soft fail: apply penalty but don't exclude
                pass
            elif not result.passed and result.penalty_score == 0:
                # Hard fail: exclude document
                return False, 0.0
            score_adj += result.penalty_score
        return True, score_adj

    def _check_industry(
        self, metadata: dict, context: FilterContext
    ) -> FilterResult:
        """Check if the document's industry is in the whitelist."""
        if not context.industry:
            return FilterResult("industry_whitelist", True, "No industry context")

        doc_industry = metadata.get("industry", "")
        if not doc_industry:
            # Document has no industry restriction — apply to all
            return FilterResult("industry_whitelist", True, "No industry restriction on document")

        # Check for partial matches (e.g., "农业" matches "农业科技")
        context_industry = context.industry.lower()
        for allowed in self.industry_whitelist:
            if allowed.lower() in context_industry or context_industry in allowed.lower():
                return FilterResult("industry_whitelist", True, f"Industry match: {context_industry}")

        if doc_industry.lower() in context_industry or context_industry in doc_industry.lower():
            return FilterResult("industry_whitelist", True, f"Document industry match: {doc_industry}")

        # Soft penalty: industry mismatch reduces score
        return FilterResult(
            "industry_whitelist",
            True,  # Don't hard-exclude, just penalize
            f"Industry mismatch: context={context.industry}, doc={doc_industry}",
            penalty_score=-0.1,
        )

    def _check_risk_level(
        self, metadata: dict, context: FilterContext
    ) -> FilterResult:
        """Filter out documents exceeding max risk level."""
        doc_risk = metadata.get("risk_level", "medium")
        max_risk = context.max_risk_level or self.max_risk_level

        try:
            doc_rank = RISK_RANK.get(RiskLevel(doc_risk), 2)
            max_rank = RISK_RANK.get(RiskLevel(max_risk), 3)
        except ValueError:
            return FilterResult("risk_level", True, "Unknown risk level; treating as pass")

        if doc_rank > max_rank:
            return FilterResult(
                "risk_level",
                False,
                f"Risk level {doc_risk} exceeds maximum {max_risk}",
            )
        return FilterResult("risk_level", True, f"Risk level {doc_risk} acceptable")

    def _check_credit_score(
        self, metadata: dict, context: FilterContext
    ) -> FilterResult:
        """Check credit score requirement."""
        if context.credit_score is None:
            return FilterResult("credit_score", True, "No credit score in context")

        required_min = metadata.get("credit_score_min", self.credit_score_min)
        if context.credit_score < required_min:
            return FilterResult(
                "credit_score",
                False,
                f"Credit score {context.credit_score} < required {required_min}",
            )
        return FilterResult(
            "credit_score",
            True,
            f"Credit score {context.credit_score} meets requirement {required_min}",
        )

    def _check_enterprise_age(
        self, metadata: dict, context: FilterContext
    ) -> FilterResult:
        """Check enterprise registration age requirement."""
        if context.registration_years is None:
            return FilterResult("enterprise_age", True, "No age context")

        required_age = metadata.get("enterprise_age_min", self.enterprise_age_min)
        if context.registration_years < required_age:
            return FilterResult(
                "enterprise_age",
                False,
                f"Enterprise age {context.registration_years}y < required {required_age}y",
            )
        return FilterResult(
            "enterprise_age",
            True,
            f"Enterprise age {context.registration_years}y meets {required_age}y",
        )

    def _check_revenue(
        self, metadata: dict, context: FilterContext
    ) -> FilterResult:
        """Check minimum revenue requirement."""
        if context.annual_revenue is None:
            return FilterResult("revenue", True, "No revenue context")

        required_revenue = metadata.get("revenue_threshold", self.revenue_threshold)
        if context.annual_revenue < required_revenue:
            return FilterResult(
                "revenue",
                True,  # Soft: just penalize
                f"Revenue {context.annual_revenue:,.0f} below threshold {required_revenue:,.0f}",
                penalty_score=-0.05,
            )
        return FilterResult(
            "revenue",
            True,
            f"Revenue {context.annual_revenue:,.0f} meets threshold",
        )

    def _check_loan_amount(
        self, metadata: dict, context: FilterContext
    ) -> FilterResult:
        """Check if loan amount is within acceptable range."""
        if context.loan_amount is None:
            return FilterResult("loan_amount", True, "No loan amount context")

        amount_range = metadata.get("amount_range", {})
        min_amount = amount_range.get("min", self.min_loan_amount)
        max_amount = amount_range.get("max", self.max_loan_amount)

        if context.loan_amount < min_amount:
            return FilterResult(
                "loan_amount",
                True,
                f"Loan amount {context.loan_amount:,.0f} below doc minimum {min_amount:,.0f}",
                penalty_score=-0.05,
            )
        if context.loan_amount > max_amount:
            return FilterResult(
                "loan_amount",
                True,
                f"Loan amount {context.loan_amount:,.0f} above doc maximum {max_amount:,.0f}",
                penalty_score=-0.1,
            )
        return FilterResult(
            "loan_amount",
            True,
            f"Loan amount {context.loan_amount:,.0f} within range",
        )

    def generate_filter_report(
        self, results: list[Any], context: FilterContext
    ) -> dict:
        """Generate a detailed report of which rules affected which documents."""
        report = {
            "total_input": len(results),
            "document_reports": [],
        }
        for doc in results:
            doc_id = getattr(doc, "id", "unknown")
            metadata = getattr(doc, "metadata", {})
            rule_results = self._apply_all_rules(doc, context)
            passed, score_adj = self._evaluate_rules(rule_results)
            report["document_reports"].append({
                "id": doc_id,
                "passed": passed,
                "score_adjustment": score_adj,
                "rules": [
                    {
                        "rule": r.rule_name,
                        "passed": r.passed,
                        "reason": r.reason,
                        "penalty": r.penalty_score,
                    }
                    for r in rule_results
                ],
            })
        report["total_passed"] = sum(
            1 for d in report["document_reports"] if d["passed"]
        )
        return report
