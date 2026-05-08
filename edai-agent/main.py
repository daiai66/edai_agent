"""
SME Financial Platform — Main Demo Entry Point

Demonstrates the complete pipeline:
1. Initialize all components (retriever, agents, planner, dispatcher, crew)
2. Index sample knowledge base documents
3. Process a sample SME loan application
4. Show planner → dispatcher → multi-agent flow
5. Output credit report with approval decision
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger

# Configure loguru
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
    level="INFO",
    colorize=True,
)
logger.add(
    "logs/platform.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
)

from config.settings import settings


# ─── Sample Knowledge Base ────────────────────────────────────────────────────

SAMPLE_KB_DOCUMENTS = [
    {
        "id": "kb_001",
        "text": (
            "中小微企业贷款审批标准：营业额超过500万，经营年限2年以上，"
            "信用评分700分以上可申请信用贷款。贷款金额最高为年营业额的60%，"
            "最高不超过300万元。申请需提供最近2年完整财务报表。"
        ),
        "metadata": {
            "doc_type": "policy",
            "industry": "",
            "risk_level": "medium",
            "credit_score_min": 700,
            "enterprise_age_min": 2,
            "revenue_threshold": 5_000_000,
            "amount_range": {"min": 50_000, "max": 3_000_000},
        },
    },
    {
        "id": "kb_002",
        "text": (
            "供应链金融：核心企业上下游中小微企业可凭借应收账款申请保理融资，"
            "最高额度为应收账款的80%。核心企业需具有AAA级信用评级。"
            "融资期限最长12个月，年化利率3.5%-6.5%。"
        ),
        "metadata": {
            "doc_type": "product",
            "industry": "",
            "risk_level": "low",
            "credit_score_min": 600,
            "enterprise_age_min": 1,
            "amount_range": {"min": 200_000, "max": 10_000_000},
        },
    },
    {
        "id": "kb_003",
        "text": (
            "农业产业链金融：农产品收购企业可申请最高500万元的季节性周转贷款，"
            "利率优惠20%（低于同期LPR）。申请条件：经营年限1年以上，"
            "具有稳定的农产品购销合同，贷款期限3-12个月。"
        ),
        "metadata": {
            "doc_type": "product",
            "industry": "农业",
            "risk_level": "low",
            "credit_score_min": 550,
            "enterprise_age_min": 1,
            "amount_range": {"min": 100_000, "max": 5_000_000},
        },
    },
    {
        "id": "kb_004",
        "text": (
            "科技型中小企业：拥有专利或高新技术企业认证的企业，"
            "可申请知识产权质押贷款，最高贷款额度500万元。"
            "专利评估价值的40%可作为贷款质押。高新技术企业认证可提高信用评级。"
        ),
        "metadata": {
            "doc_type": "product",
            "industry": "科技",
            "risk_level": "medium",
            "credit_score_min": 650,
            "enterprise_age_min": 2,
            "amount_range": {"min": 500_000, "max": 10_000_000},
        },
    },
    {
        "id": "kb_005",
        "text": (
            "餐饮连锁企业贷款：连锁餐饮企业需提供3年完整财务数据，"
            "税务申报记录及POS流水作为还款能力证明。"
            "单店日均流水需超过1万元，整体门店数不少于3家。"
            "信用评分要求680分以上，贷款期限最长36个月。"
        ),
        "metadata": {
            "doc_type": "product",
            "industry": "餐饮",
            "risk_level": "medium",
            "credit_score_min": 680,
            "enterprise_age_min": 3,
            "amount_range": {"min": 200_000, "max": 5_000_000},
        },
    },
    {
        "id": "kb_006",
        "text": (
            "农业科技企业融资政策：农业科技企业结合了农业和科技双重政策支持，"
            "可同时享受农业贷款利率优惠（最高20%）和高新技术企业信贷支持。"
            "具备农业科技背景的企业，在信用评分相同情况下，审批通过率高出普通企业15%。"
        ),
        "metadata": {
            "doc_type": "policy",
            "industry": "农业科技",
            "risk_level": "low",
            "credit_score_min": 600,
            "enterprise_age_min": 1,
            "amount_range": {"min": 100_000, "max": 5_000_000},
        },
    },
    {
        "id": "kb_007",
        "text": (
            "中小企业信用贷款风险评级标准：AAA级（900-1000分）：最优质客户，可获批申请额度100%；"
            "AA级（800-899分）：优质客户，可获批80%；A级（700-799分）：良好客户，可获批60%；"
            "BBB级（600-699分）：一般客户，可获批40%，需提供担保；"
            "BB级以下（<600分）：高风险，通常不予批准。"
        ),
        "metadata": {
            "doc_type": "policy",
            "industry": "",
            "risk_level": "medium",
            "credit_score_min": 0,
        },
    },
    {
        "id": "kb_008",
        "text": (
            "供应链金融应收账款保理：京东、阿里巴巴、盒马等电商和新零售平台"
            "的供应商可直接通过供应链金融平台申请应收账款保理融资。"
            "平台会根据核心企业信用度自动授信，最快1个工作日放款。"
        ),
        "metadata": {
            "doc_type": "product",
            "industry": "",
            "risk_level": "low",
            "credit_score_min": 580,
            "enterprise_age_min": 1,
            "amount_range": {"min": 100_000, "max": 8_000_000},
        },
    },
]


# ─── Sample Enterprise ────────────────────────────────────────────────────────

SAMPLE_ENTERPRISE = {
    "company_name": "深圳绿色农业科技有限公司",
    "industry": "农业科技",
    "registration_years": 3,
    "annual_revenue": 8_500_000,  # 850万
    "financial_data": {
        "2023": {
            "revenue": 8_500_000,
            "profit": 1_200_000,
            "assets": 15_000_000,
            "liabilities": 6_000_000,
        },
        "2022": {
            "revenue": 6_800_000,
            "profit": 900_000,
            "assets": 12_000_000,
            "liabilities": 5_000_000,
        },
    },
    "tax_compliance": "良好",
    "credit_history": "无不良记录",
    "supply_chain": {
        "major_customers": ["京东农业", "盒马生鲜"],
        "major_suppliers": ["农业合作社A", "有机肥料厂B"],
        "avg_payment_days": 45,
    },
    "loan_request": {
        "amount": 3_000_000,  # 300万
        "purpose": "扩大生产线，购置农业设备",
        "term_months": 36,
    },
}


# ─── Initialization ───────────────────────────────────────────────────────────

def initialize_components():
    """Initialize all platform components with graceful degradation."""
    logger.info("Initializing SME Financial Platform components...")

    # Log API key status
    key_status = settings.validate_api_keys()
    logger.info(f"API key status: {key_status}")
    if not key_status["anthropic"]:
        logger.warning(
            "ANTHROPIC_API_KEY not configured. "
            "LLM features will use mock responses. "
            "Set ANTHROPIC_API_KEY in .env to enable full functionality."
        )

    # Initialize retrieval pipeline
    from retrieval.hybrid_retriever import create_hybrid_retriever
    logger.info("Creating hybrid retriever...")
    retriever = create_hybrid_retriever()

    # Initialize agents
    from agents.qa_agent import QAAgent
    from agents.credit_agent import CreditAgent
    from agents.graph_agent import GraphAgent

    logger.info("Creating QA agent...")
    qa_agent = QAAgent(retriever=retriever)

    logger.info("Creating Credit agent...")
    credit_agent = CreditAgent(use_crewai=False)  # Direct tools for stability

    logger.info("Creating Graph agent...")
    graph_agent = GraphAgent()

    # Initialize planner and dispatcher
    from planner.planner import FinancialPlanner
    from planner.dispatcher import Dispatcher

    logger.info("Creating Planner and Dispatcher...")
    planner = FinancialPlanner()
    dispatcher = Dispatcher(
        qa_agent=qa_agent,
        credit_agent=credit_agent,
        graph_agent=graph_agent,
    )

    # Initialize crew
    from crews.financial_crew import FinancialAnalysisCrew
    logger.info("Creating Financial Analysis Crew...")
    crew = FinancialAnalysisCrew(use_crewai=False)  # Direct tools for demo stability

    logger.info("All components initialized successfully")
    return {
        "retriever": retriever,
        "qa_agent": qa_agent,
        "credit_agent": credit_agent,
        "graph_agent": graph_agent,
        "planner": planner,
        "dispatcher": dispatcher,
        "crew": crew,
    }


def index_knowledge_base(retriever) -> None:
    """Index sample knowledge base documents."""
    logger.info(f"Indexing {len(SAMPLE_KB_DOCUMENTS)} knowledge base documents...")

    texts = [doc["text"] for doc in SAMPLE_KB_DOCUMENTS]
    metadatas = [doc["metadata"] for doc in SAMPLE_KB_DOCUMENTS]
    doc_ids = [doc["id"] for doc in SAMPLE_KB_DOCUMENTS]

    success = retriever.index_documents(texts=texts, metadatas=metadatas, doc_ids=doc_ids)

    if success:
        stats = retriever.get_stats()
        logger.info(f"Knowledge base indexed. Stats: {stats}")
    else:
        logger.warning("Knowledge base indexing failed (partial or complete)")


# ─── Demo Steps ───────────────────────────────────────────────────────────────

def demo_qa_agent(qa_agent, enterprise_data: dict) -> None:
    """Demonstrate Q&A agent with knowledge retrieval."""
    print("\n" + "="*60)
    print("DEMO 1: Q&A Agent (RAG)")
    print("="*60)

    questions = [
        "农业科技企业申请贷款需要满足哪些条件？",
        "供应链保理融资的最高额度是多少？",
    ]

    for question in questions:
        print(f"\n问题: {question}")
        print("-" * 40)
        try:
            response = qa_agent.answer(question=question, context=enterprise_data)
            print(f"回答: {response.answer}")
            print(f"置信度: {response.confidence:.0%}")
            if response.citations:
                print(f"引用文档: {[c.doc_id for c in response.citations[:3]]}")
        except Exception as e:
            logger.error(f"QA demo failed: {e}")
            print(f"Error: {e}")


def demo_graph_agent(graph_agent, enterprise_data: dict) -> None:
    """Demonstrate Graph agent with industry analysis."""
    print("\n" + "="*60)
    print("DEMO 2: Graph Agent (Industry Analysis)")
    print("="*60)

    try:
        analysis = graph_agent.analyze(enterprise_data)
        print(analysis.to_summary())
        print(f"供应链保理资格: {'是' if analysis.supply_chain_intelligence.factoring_eligible else '否'}")
        print(f"竞争优势: {analysis.competitive_position.competitive_advantages[:2]}")
    except Exception as e:
        logger.error(f"Graph demo failed: {e}")
        print(f"Error: {e}")


def demo_credit_agent(credit_agent, enterprise_data: dict) -> None:
    """Demonstrate Credit agent with full assessment."""
    print("\n" + "="*60)
    print("DEMO 3: Credit Agent (Full Assessment)")
    print("="*60)

    try:
        report = credit_agent.assess(enterprise_data)
        print(report.to_summary())
    except Exception as e:
        logger.error(f"Credit demo failed: {e}")
        print(f"Error: {e}")


def demo_planner_dispatcher(planner, dispatcher, enterprise_data: dict) -> None:
    """Demonstrate Planner → Dispatcher → Multi-Agent flow."""
    print("\n" + "="*60)
    print("DEMO 4: Planner → Dispatcher → Multi-Agent Flow")
    print("="*60)

    query = "请对深圳绿色农业科技有限公司的300万贷款申请进行全面评估"

    print(f"\n用户查询: {query}")
    print("\n[Planner] Generating dispatch plan...")
    try:
        plan = planner.plan(query=query, enterprise_context=enterprise_data)
        print(plan.summary())

        print(f"\n[Dispatcher] Executing {len(plan.steps)} steps...")
        result = dispatcher.execute_sync(
            plan=plan,
            context=enterprise_data,
            query=query,
        )

        print(f"\n[Results] Completed in {result.total_duration_seconds:.1f}s")
        print(f"Success: {result.success}")
        if result.errors:
            print(f"Errors: {result.errors}")

        print("\n[Final Response]")
        print("-" * 40)
        print(result.final_response[:2000] if len(result.final_response) > 2000
              else result.final_response)

    except Exception as e:
        logger.error(f"Planner/Dispatcher demo failed: {e}")
        print(f"Error: {e}")


def demo_financial_crew(crew, enterprise_data: dict) -> None:
    """Demonstrate full CrewAI financial analysis crew."""
    print("\n" + "="*60)
    print("DEMO 5: Financial Analysis Crew (Full Pipeline)")
    print("="*60)

    print(f"\nProcessing loan application for: {enterprise_data['company_name']}")
    print(f"Loan request: {enterprise_data['loan_request']['amount']:,.0f}元 / "
          f"{enterprise_data['loan_request']['term_months']}个月")
    print(f"Purpose: {enterprise_data['loan_request']['purpose']}")

    try:
        start = time.time()
        report = crew.process_loan_application(enterprise_data)
        elapsed = time.time() - start

        report.print_report()
        print(f"\n[Timing] Crew completed in {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"Crew demo failed: {e}")
        print(f"Error: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main demo entry point."""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    print("="*60)
    print("  SME Financial Platform - Demo")
    print(f"  Claude Model: {settings.anthropic.model}")
    print(f"  Milvus: {settings.milvus.host}:{settings.milvus.port}")
    print(f"  Neo4j: {settings.neo4j.uri}")
    print("="*60)

    # Step 1: Initialize components
    logger.info("Step 1: Initializing all components")
    components = initialize_components()

    # Step 2: Index knowledge base
    logger.info("Step 2: Indexing knowledge base")
    index_knowledge_base(components["retriever"])

    # Step 3: Run demos
    enterprise_data = SAMPLE_ENTERPRISE

    print("\n" + "#"*60)
    print("# STARTING DEMO SEQUENCE")
    print("#"*60)

    # Demo 1: Q&A with RAG
    demo_qa_agent(components["qa_agent"], enterprise_data)

    # Demo 2: Graph agent
    demo_graph_agent(components["graph_agent"], enterprise_data)

    # Demo 3: Credit agent
    demo_credit_agent(components["credit_agent"], enterprise_data)

    # Demo 4: Planner + Dispatcher
    demo_planner_dispatcher(
        components["planner"],
        components["dispatcher"],
        enterprise_data,
    )

    # Demo 5: Full crew
    demo_financial_crew(components["crew"], enterprise_data)

    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nTo run with full LLM capabilities:")
    print("  1. Copy .env.example to .env")
    print("  2. Set ANTHROPIC_API_KEY=your_key")
    print("  3. (Optional) Start Milvus: docker run -p 19530:19530 milvusdb/milvus:latest")
    print("  4. (Optional) Start Neo4j: docker run -p 7687:7687 neo4j:latest")
    print("  5. python main.py")


if __name__ == "__main__":
    main()
