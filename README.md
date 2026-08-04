# Autonomous Multi-Agent Research System (MAS)

An academic-grade Multi-Agent System built in Python, designed to perform end-to-end scientific research, symbolic mathematical derivations, numerical simulations, and technical report writing. 

Powered by **Groq High-Throughput Llama-3.3-70B / Llama-3.1-8b-instant** with **Google Gemini fallback**, **GraphRAG Knowledge Graph Memory**, **Smart Model Routing**, and an **8-Phase ReAct Research Pipeline**.

---

## 🚀 Real Empirical Evaluation & Knowledge Graph Impact

### 📊 Empirical A/B Benchmark Results (`evals/baseline_eval.py`)

Evaluation results produced by running the **live multi-agent orchestrator** (`LiteratureScout`, `NumericalAnalyst`, `ReportWriter`) across 3 scientific domain benchmarks, comparing **WITH Knowledge Graph (GraphRAG Engine)** vs **WITHOUT Knowledge Graph (Raw Text Baseline)**:

| Evaluation ID | Scientific Domain | Entity Recall (WITH KG) | Entity Recall (NO KG) | Net Recall Impact | KG Graph Nodes | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `semiconductor_hemt_01` | Semiconductor Physics | **66.7%** | 0.0% | **+66.7%** | 42 | **PASS** |
| `physics_fourier_02` | Signal Processing | **50.0%** | 0.0% | **+50.0%** | 38 | **PASS** |
| `quantum_schrodinger_03` | Quantum Mechanics | **66.7%** | 16.7% | **+50.0%** | 31 | **PASS** |
| **EMPIRICAL AVERAGE** | **Multi-Domain Science** | **61.1%** | **5.6%** | **+55.5% Gain** | **37.0 Nodes / Run** | **PASSED** |

---

### 💡 Why the Knowledge Graph Matters (Empirical Analysis)

Without Knowledge Graph memory, agents relying solely on sliding-window context history suffer from **severe context decay**:
* **Entity Recall without KG**: Drops to **5.6%** because key technical terms, boundary conditions, and mathematical variables get truncated across multi-turn ReAct reasoning steps.
* **Entity Recall with KG**: Increases to **61.1%** (**+55.5% absolute gain**), as extracted triples are anchored in a persistent NetworkX GraphRAG structure, preventing entity loss across pipeline phase handoffs.
* **Graph Scale**: Extracted graphs average **37.0 nodes** per quick research run and scale beyond **150+ nodes / 110+ edges** on deep multi-phase investigations.

---

## 🏛️ System Architecture: The 8-Phase Pipeline

MAS decomposes scientific questions into structured sub-problems and executes them through a gated **8-Phase Pipeline**:

```
[1. UNDERSTAND] ──► [2. LITERATURE] ──► [3. MATHEMATICS] ──► [4. COMPUTATION]
                                                                  │
[8. REPORT]     ◄── [7. SYNTHESIZE] ◄── [6. PEER REVIEW] ◄── [5. ENGINEERING]
```

1. **Phase 1 — UNDERSTAND (Decomposition)**:
   * *Agent*: Research Planner (Orchestrator direct thinking).
   * Deconstructs the research prompt into 3-5 technical sub-problems, identifying physical variables and target outputs.
2. **Phase 2 — LITERATURE (Academic Search)**:
   * *Agent*: `LiteratureScout` (Tools: `ArxivSearch`, `Wikipedia`, `WebSearch`).
   * Identifies governing equations, historical background, and state-of-the-art results from peer-reviewed literature.
3. **Phase 3 — MATHEMATICS (Symbolic Derivation)**:
   * *Agent*: `Mathematician` (Tools: `SymPyTool`, `LatexFormatterTool`).
   * Performs analytical derivations, simplifies expressions, and formats mathematical proofs in LaTeX.
4. **Phase 4 — COMPUTATION (Numerical Analysis)**:
   * *Agent*: `NumericalAnalyst` (Tools: `NumericalTool`, `CalculatorTool`).
   * Solves linear systems ($Ax = b$), computes numerical integration, and evaluates models numerically.
5. **Phase 5 — ENGINEERING (Physical Bounds & Checks)**:
   * *Agent*: `Engineer` (Tools: `UnitConverterTool`, `NumericalTool`).
   * Applies dimensional consistency checks, identifies dominant terms, and validates physical plausibility.
6. **Phase 6 — PEER REVIEW (Refined Critique)**:
   * *Agent*: `PeerReviewer`.
   * Critiques collected findings as a journal referee, scoring scientific validity and clarity.
7. **Phase 7 — SYNTHESIZE (Coherence Gate)**:
   * *Agent*: `Synthesizer`.
   * Merges literature, derivations, calculations, and reviewer critiques into a unified summary.
8. **Phase 8 — REPORT (Academic Writeup)**:
   * *Agent*: `ReportWriter` (Tools: `LatexFormatterTool`).
   * Generates a publication-quality technical report complete with Abstract, Introduction, Theory, Analysis, Results, Discussion, and References in `./reports/`.

---

## ⚙️ Core Engineering & Capabilities

### ⚡ High-Throughput Groq Engine + Gemini Fallback
* **Primary LLM Engine**: Groq REST API utilizing `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`.
* **Thread-Safe Rate Pacing**: Enforces a `threading.Lock()` 3.5-second call spacer (`MIN_CALL_INTERVAL = 3.5s`) to remain strictly under Groq's 30 RPM rate limit across parallel backend worker threads.
* **Smart Model Router**: Matches phase complexity (`flash` / `pro` models) to optimize token budget and latency (`core/model_router.py`).

### 🔬 Academic Research Skills Toolkit (`tools/research_skills.py`)
* `literature_critique`: Identifies 5 key claims, steel-mans the weakest argument, and raises 5 critical objections.
* `gap_finder`: Extracts 7 unresolved limits and generates 10 novel research questions.
* `synthesis_drafter`: Cleans summaries, forms a central synthesis claim, and drafts LaTeX Related Works sections.
* `concept_mapper`: Explains complex concepts via analogy and maps logical argument chains.
* `academic_refinement`: Rewrites abstracts into a 4-sentence structure, plays devil's advocate, and drafts one-page briefs.

### 🛡️ Windows Console & Unicode Encoding Safety
* Fully resilient against Windows `cp1252` terminal encoding crashes via a custom `safe_print()` Unicode fallback layer (`config.py`).

---

## 🛠️ Project Structure

```
MAS/
├── backend/
│   └── api.py               # FastAPI Web Server (REST & SSE Stream Endpoints)
├── frontend/
│   └── index.html           # Real-Time Research Web Interface
├── core/
│   ├── model_router.py      # Smart Model Complexity Router
│   └── react_loop.py        # ReAct Reasoning Engine
├── agents/
│   ├── base_agent.py        # Base Agent Class with Tool & Memory Hooks
│   └── research_agents.py   # Specialized Agent Factories (Scout, Mathematician, etc.)
├── memory/
│   ├── graph_memory.py      # NetworkX Knowledge Graph Memory (GraphRAG)
│   ├── short_term.py        # Context Sliding Window & Summarizer
│   └── long_term.py         # Persistent JSON Memory Store
├── tools/
│   ├── builtin_tools.py     # Calculator & Web Search
│   ├── research_skills.py   # Academic Skills Toolkit (5 Core Skills)
│   └── research_tools.py    # SymPy, SciPy, Unit Converter, arXiv, Wikipedia
├── evals/
│   ├── baseline_eval.py     # Empirical Evaluation Benchmark Suite
│   ├── baseline_results.json# Real Empirical Results & Metrics
│   └── datasets.json        # Benchmark Dataset Queries
├── orchestrator/
│   └── research_orchestrator.py # 8-Phase Research Pipeline Engine
├── reports/                 # Output directory for Markdown Research Reports
├── config.py                # Global Configuration, Throttle Lock & LLM Dispatcher
├── research.py              # Interactive Terminal CLI Application
└── requirements.txt         # Dependencies
```

---

## 💻 Getting Started

### 1. Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/vihaaaaan17/Multiagent-research-pipeline.git
cd Multiagent-research-pipeline
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the root directory:
```env
GROQ_API_KEY="your_groq_api_key_here"
GEMINI_API_KEY="your_google_gemini_api_key_here"
```

### 3. Usage Options

#### Option A: Terminal CLI App
```bash
python research.py
```

#### Option B: Web Application (FastAPI + SSE Stream)
```bash
python -m uvicorn backend.api:app --reload --port 8000
```
Then open: `http://localhost:8000`

#### Option C: Run Empirical Benchmark Suite
```bash
python evals/baseline_eval.py
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.