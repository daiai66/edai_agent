# SME Financial Platform — 中小微企业智能金融平台

基于 **Claude Opus 4.7 + Milvus + CrewAI** 构建的中小微企业金融智能审批平台，实现 **"数据穿透 → 智能授信 → 产品推荐"** 闭环。

通过混合检索（BM25 + 向量召回）、规则过滤、Cross-Encoder精排，结合多智能体协同（问答/信贷/图谱），自动化处理企业财务、税务、工商、供应链等数据，提升融资申请成功率与审批效率。

---

## 核心架构

```
用户查询 → Planner（生成结构化执行计划）
              ↓
         Dispatcher（解析计划、依赖排序、并行调度）
              ↓
   ┌──────────┼──────────┐
   ↓          ↓          ↓
QA Agent  Credit Agent  Graph Agent
(RAG)     (信贷评估)    (行业图谱)
   ↓          ↓          ↓
   └──────→ 结果合并 ←─────┘
              ↓
         最终综合报告
```

### 检索管道
```
query → BM25稀疏向量 + Dense向量 → Milvus混合搜索（RRF融合）
      → RuleFilter（业务规则过滤）→ Cross-Encoder Reranker → Top-K结果
```

---

## 项目结构

```
edai-agent/
├── config/
│   └── settings.py              # Pydantic配置（Milvus/Anthropic/Neo4j等）
├── retrieval/
│   ├── milvus_client.py         # Milvus客户端（含内存fallback）
│   ├── bm25_retriever.py        # BM25稀疏检索 + jieba中文分词
│   ├── rule_filter.py           # 业务规则过滤（信用分/营业额/行业白名单）
│   ├── reranker.py              # Cross-Encoder精排
│   └── hybrid_retriever.py      # 混合检索编排
├── agents/
│   ├── qa_agent.py              # RAG问答代理
│   ├── credit_agent.py          # 信贷评估代理（4阶段：财务/税务/供应链/风险）
│   └── graph_agent.py           # 知识图谱代理（行业/竞争/供应链关系）
├── tools/
│   ├── financial_tools.py       # 财务分析CrewAI工具
│   ├── credit_tools.py          # 信贷报告/产品匹配/抵押评估工具
│   └── graph_tools.py           # 行业知识/供应链/竞争分析工具
├── planner/
│   ├── planner.py               # Claude生成结构化DispatchPlan
│   └── dispatcher.py            # 异步并行调度器
├── crews/
│   └── financial_crew.py        # CrewAI多智能体协同
├── main.py                      # 端到端Demo
├── requirements.txt
└── .env.example
```

---

## 快速开始

### 1. 安装依赖

```powershell
# 使用Anaconda Python
& "C:\Users\lenovo\anaconda3\python.exe" -m pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填写：

```env
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-opus-4-7
CLAUDE_MAX_TOKENS=16000
CLAUDE_EFFORT=high             # low | medium | high | max

# 可选：外部服务（不配置则自动使用内存fallback）
MILVUS_HOST=localhost
MILVUS_PORT=19530
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=...
```

### 3. 运行Demo

```powershell
python main.py
```

Demo会演示5个场景：QA问答、行业图谱分析、信贷评估、Planner+Dispatcher流程、完整CrewAI协同。

---

## 核心技术

| 组件 | 实现 | 作用 |
|---|---|---|
| **稀疏召回** | `rank_bm25` + `jieba`（金融领域词典） | 关键词精确匹配 |
| **稠密召回** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（dim=768） | 语义相似度匹配 |
| **混合搜索** | Milvus RRF（Reciprocal Rank Fusion） | 稀疏+稠密分数融合 |
| **规则过滤** | 信用分≥600、经营年限≥1年、行业白名单 | 硬规则筛除不合格文档 |
| **精排** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 高质量query-doc相关性打分 |
| **规划器** | Claude `thinking={"type": "adaptive"}` + JSON Schema | 生成结构化执行计划 |
| **调度器** | `asyncio.gather` + 依赖DAG | 并行执行无依赖步骤 |
| **多智能体** | CrewAI（researcher / credit_analyst / report_writer） | 跨智能体任务协同 |

---

## Claude API 使用要点

所有Claude调用都遵循以下最佳实践：

- **模型**：`claude-opus-4-7`（默认）
- **思考模式**：`thinking={"type": "adaptive"}`（自适应思考，无需 `budget_tokens`）
- **流式响应**：
  ```python
  with client.messages.stream(**kwargs) as stream:
      message = stream.get_final_message()
  ```
- **Prompt Caching**：System prompt 加 `cache_control={"type": "ephemeral"}` 缓存
- **重试**：`tenacity` 指数退避处理速率限制

---

## 关键设计：优雅降级

平台所有外部依赖均支持内存fallback，**无需Milvus/Neo4j/GPU即可启动**：

| 服务 | 不可用时降级方案 |
|---|---|
| Milvus | 内存哈希字典存储+暴力相似度搜索 |
| Neo4j | 内存图结构（dict邻接表） |
| sentence-transformers | 随机向量（用于结构调试） |
| cross-encoder | BM25 token覆盖率打分 |
| Anthropic API | Mock响应模板 |

---

## 业务效果（设计指标）

- 中小微企业融资申请成功率 ↑ **58%**
- 贷款审批时间 ↓ **60%**
- 自动化处理财务/税务/工商/供应链四大数据源
- 全链路产业数据 + 行业知识图谱驱动的风险评估

---

## License

Internal use only.
