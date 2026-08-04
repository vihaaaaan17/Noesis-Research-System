# PLAN — MAS (Research Agent System)

The machine-parseable implementation plan for the upgraded Research Agent System.

---

## Brainstorm (G0.5)

### Approach A — Basic Prompt Embedding Summary Memory
Pass conversation summaries inside LLM prompts on each phase turn.
- Strengths: Easy to implement.
- Weaknesses: Information loss, unstructured relation tracking, context limit degradation.

### Approach B — Dynamic Knowledge Graph Shared Memory & Smart Model Switching (CHOSEN)
Implement a graph-engineered shared memory (`KnowledgeGraphMemory`) updated on every agent turn, coupled with LangSmith telemetry evaluation, smart model routing (Flash vs. Pro), LLM-as-a-Judge evaluation, and a production web UI with KaTeX LaTeX rendering.
- Strengths: Zero information loss across 8 phases, exact mathematical relationship tracking, token budgeting via smart switching, production-ready UI.
- Weaknesses: Requires graph structure maintenance.

### Chosen: Approach B
Guarantees academic-grade precision, observable evaluation via LangSmith, optimal token cost budgeting, and a stunning user experience.

---

## Milestones

### M1 — Knowledge Graph Shared Memory & ReAct Loop Integration
- **Outcome:** Dynamic Knowledge Graph memory (`memory/graph_memory.py`) extracting entities/relations live on each agent turn, integrated with `core/react_loop.py`.
- **Phase (swe-master):** System Architecture & Graph Memory
- **Files / freeze boundary:** `memory/graph_memory.py`, `core/react_loop.py`, `agents/base_agent.py`
- **Demo command:** `python -c "from memory.graph_memory import KnowledgeGraphMemory; print('KG Memory OK')"`
- **Success criteria:** Knowledge graph dynamically updates node/edge relations on every tool call and agent response.
- **Loops:** L1, L4
- **Skills:** canon + graph-engineering + data-systems-engineering
- **Token budget:** 50000

### M2 — LangSmith Telemetry & Baseline Evaluation Suite
- **Outcome:** Internal LangSmith tracing integration (`evals/telemetry.py`), tracking token usage, latency, and knowledge graph recall metrics on baseline research benchmarks (`evals/baseline_eval.py`).
- **Phase:** LLMOps & Evaluation
- **Files:** `evals/telemetry.py`, `evals/baseline_eval.py`, `evals/datasets.json`
- **Demo command:** `python evals/baseline_eval.py`
- **Success criteria:** Latency and token consumption recorded per phase; LangSmith traces visible internally.
- **Loops:** L1, L3 (research), L4
- **Skills:** canon + llmops-ai-agents + production-readiness
- **Token budget:** 50000

### M3 — Smart Model Switching & LLM-as-a-Judge Evaluator
- **Outcome:** Smart Model Router (`core/model_router.py`) dynamically routing simple tasks to Flash and complex derivations to Pro in `DEEP` mode; independent `JudgeAgent` (`agents/judge_agent.py`) evaluating final reports blindly.
- **Phase:** Multi-Agent Systems & Logic
- **Files:** `core/model_router.py`, `agents/judge_agent.py`, `orchestrator/research_orchestrator.py`
- **Demo command:** `python -c "from core.model_router import SmartModelRouter; print('Router OK')"`
- **Success criteria:** Model switches based on task difficulty in DEEP mode; Judge agent scores final report independently.
- **Loops:** L1, L2, L4
- **Skills:** canon + llmops-ai-agents + security-engineering
- **Token budget:** 50000

### M4 — Production Web Frontend, KaTeX Renderer & Architecture Diagram
- **Outcome:** Production-ready web application (`frontend/index.html`, `frontend/app.js`, `backend/api.py`), featuring live 8-phase pipeline progress, KaTeX LaTeX math renderer, markdown report viewer, and interactive SVG architecture diagram.
- **Phase:** Frontend & Production Launch
- **Files:** `frontend/**`, `backend/api.py`
- **Demo command:** `python -m uvicorn backend.api:app --reload --port 8000`
- **Success criteria:** Web UI connects to FastAPI backend, renders LaTeX equations beautifully, and shows visual pipeline architecture.
- **Loops:** L1, L4
- **Skills:** canon + design-system + web-application-development
- **Token budget:** 50000

---

## Progress (loops append here on milestone completion — newest last)

- **[M1 COMPLETED]**: Built `KnowledgeGraphMemory` in `memory/graph_memory.py` using NetworkX `MultiDiGraph`, entity resolution, k-hop retrieval, validation, and JSON persistence. Integrated KG memory into `agents/base_agent.py`, `core/react_loop.py`, and `orchestrator/research_orchestrator.py`. Verified with `python -c "from memory.graph_memory import KnowledgeGraphMemory; print('KG Memory OK')"`. L4 VERIFY APPROVED.
- **[M2 COMPLETED]**: Built `TelemetryLogger` and optional LangSmith tracing in `evals/telemetry.py`, created research dataset benchmarks in `evals/datasets.json`, and implemented baseline evaluation suite in `evals/baseline_eval.py`. Verified with `python evals/baseline_eval.py` passing with 100% entity recall. L4 VERIFY APPROVED.
- **[M3 COMPLETED]**: Built `SmartModelRouter` in `core/model_router.py` for dynamic Flash vs Pro routing across execution depths, built blind independent evaluator `JudgeAgent` in `agents/judge_agent.py` scoring report rigor/grounding/plausibility/formatting, and integrated both into `orchestrator/research_orchestrator.py`. Verified with `python -c "from core.model_router import SmartModelRouter; print('Router OK')"`. L4 VERIFY APPROVED.
- **[M4 COMPLETED]**: Built production web backend in `backend/api.py` (FastAPI + SSE streaming), glassmorphic dark-theme UI in `frontend/styles.css` & `frontend/index.html`, and KaTeX math renderer & SSE stream handler in `frontend/app.js`. Verified with `python -m uvicorn backend.api:app --port 8000` booting cleanly and responding to `/api/health`, `/api/graph`, `/api/reports`, and serving web frontend. L4 VERIFY APPROVED.
