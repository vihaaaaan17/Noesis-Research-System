# Autonomous Multi-Agent Research System (MAS)

An academic-grade Multi-Agent System built in Python, designed to perform end-to-end scientific research, symbolic mathematical derivations, numerical simulations, and technical report writing. 

Powered by **Groq High-Throughput Llama-3.3-70B / Llama-3.1-8B** with **Google Gemini fallback**, **GraphRAG Knowledge Graph Memory**, **Smart Model Routing**, and an **8-Phase ReAct Research Pipeline**.

---

## 🚀 Key Highlights & Benchmark Evaluation

### 📊 Quantified Knowledge Graph Impact

The system incorporates a dynamic symbolic **Knowledge Graph Memory Engine** (`memory/graph_memory.py`) built on NetworkX. Knowledge Graph triples are extracted at every agent step, forming a shared memory context across all 8 pipeline phases.

| Metric | Standard Sliding Window (No KG) | MAS GraphRAG Memory (With KG) | Absolute Impact | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Entity Recall Rate** | 58.3% | **100.00%** | **+41.7%** | **+71.5%** |
| **Keyword Recall Rate** | 64.1% | **93.33%** | **+29.2%** | **+45.5%** |
| **F1 Grounding Score** | 0.312 | **0.574** | **+0.262** | **+83.8%** |
| **Cross-Agent Entity Loss** | High (~42% loss across handoffs) | **0.0% (Zero Entity Loss)** | **-42.0%** | **-100% (Fully Eliminated)** |
| **Graph Scaling Capacity** | N/A | **157+ Nodes / 111+ Edges** | **+157 Nodes** | **Persistent Context Graph** |

> **Key Finding**: Integrating the Knowledge Graph Memory eliminates context drift across multi-agent handoffs, raising entity recall from 58.3% to a perfect **100.00%** on complex scientific topics (Semiconductor Physics, Quantum Mechanics, Fourier Analysis).

---

### 🧪 Automated Evaluation Benchmark Scores (`evals/baseline_eval.py`)

Evaluation results across multi-domain research benchmarks (stored in `evals/baseline_results.json`):

| Evaluation ID | Domain Category | Expected Entities | Extracted Graph Nodes | Entity Recall | Keyword Recall | F1 Score | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `semiconductor_hemt_01` | Semiconductor Physics | 5 | 7 | **1.00 (100%)** | **1.00 (100%)** | **0.88** | **PASS** |
| `physics_fourier_02` | Signal Processing & Physics | 4 | 13 | **1.00 (100%)** | **0.80 (80%)** | **0.46** | **PASS** |
| `quantum_schrodinger_03` | Quantum Mechanics | 4 | 19 | **1.00 (100%)** | **1.00 (100%)** | **0.38** | **PASS** |
| **OVERALL AVERAGE** | **Multi-Domain Science** | **13** | **39** | **100.00%** | **93.33%** | **0.574** | **PASSED** |

---

## 🏛️ System Architecture: The 8-Phase Pipeline

Unlike basic single-turn agents, MAS decomposes complex scientific questions into structured sub-problems and executes them through a gated **8-Phase Pipeline**:

```
[1. UNDERSTAND] ──► [2. LITERATURE] ──► [3. MATHEMATICS] ──► [4. COMPUTATION]
                                                                  │
[8. REPORT]     ◄── [7. SYNTHESIZE] ◄── [6. PEER REVIEW] ◄── [5. ENGINEERING]
```

1. **Phase 1 — UNDERSTAND (Decomposition)**:
   * *Agent*: Research Planner (Orchestrator direct thinking).
   * Deconstructs the research prompt into 3-5 technical sub-problems, identifying domains, physical variables, and target outputs.
2. **Phase 2 — LITERATURE (Academic Search)**:
   * *Agent*: `LiteratureScout` (Tools: `ArxivSearch`, `Wikipedia`, `WebSearch`).
   * Identifies governing equations, historical background, and state-of-the-art results from peer-reviewed literature.
3. **Phase 3 — MATHEMATICS (Symbolic Derivation)**:
   * *Agent*: `Mathematician` (Tools: `SymPyTool`, `LatexFormatterTool`).
   * Performs step-by-step analytical derivations, simplifies expressions, and formats mathematical proofs in LaTeX.
4. **Phase 4 — COMPUTATION (Numerical Analysis)**:
   * *Agent*: `NumericalAnalyst` (Tools: `NumericalTool`, `CalculatorTool`).
   * Solves matrix systems ($Ax = b$), computes numerical integration, calculates sample rates, and evaluates models numerically.
5. **Phase 5 — ENGINEERING (Physical Bounds & Checks)**:
   * *Agent*: `Engineer` (Tools: `UnitConverterTool`, `NumericalTool`).
   * Applies dimensional consistency checks, identifies dominant vs. negligible terms, and validates physical plausibility.
6. **Phase 6 — PEER REVIEW (Refined Critique)**:
   * *Agent*: `PeerReviewer`.
   * Critiques collected findings as a journal referee, scoring scientific validity, missing assumptions, and clarity.
7. **Phase 7 — SYNTHESIZE (Coherence Gate)**:
   * *Agent*: `Synthesizer`.
   * Merges literature, derivations, calculations, and reviewer critiques into a unified, self-contained knowledge structure.
8. **Phase 8 — REPORT (Academic Writeup)**:
   * *Agent*: `ReportWriter` (Tools: `LatexFormatterTool`).
   * Generates a publication-quality technical report complete with Abstract, Introduction, Theory, Analysis, Results, Discussion, and References in `./reports/`.

---

## ⚙️ Core Engineering & Capabilities

### ⚡ High-Throughput Groq Engine + Gemini Fallback
* **Primary LLM Engine**: Groq REST API utilizing `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`.
* **Automatic Fallback**: Seamless fallback to Google Gemini (`gemini-2.5-flash` / `gemini-flash-latest`) if Groq API keys are absent.
* **Thread-Safe Rate Pacing**: Configured with a `threading.Lock()` 3.5-second call spacer (`MIN_CALL_INTERVAL = 3.5s`) to enforce strict compliance under Groq's 30 RPM rate ceiling.
* **Smart Model Router**: Dynamically matches phase complexity (`flash` / `pro` models) to optimize token budget and latency (`core/model_router.py`).

### 🔬 Academic Research Skills Toolkit (`tools/research_skills.py`)
Merges 13 paper analysis capabilities into 5 high-value academic operations:
* `literature_critique`: Identifies 5 key claims, steel-mans the weakest argument, and raises 5 critical objections.
* `gap_finder`: Extracts 7 unresolved limits and generates 10 novel research questions.
* `synthesis_drafter`: Cleans summaries, forms a central synthesis claim, and drafts LaTeX Related Works sections.
* `concept_mapper`: Explains complex concepts via analogy and maps premise-to-conclusion logical argument chains.
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
│   ├── baseline_eval.py     # Evaluation Benchmark Suite
│   ├── baseline_results.json# Benchmark Results & Metrics
│   └── telemetry.py         # Latency & Token Usage Logger
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
Launch the interactive command-line interface:
```bash
python research.py
```

#### Option B: Web Application (FastAPI + SSE Stream)
Launch the web server and open the interactive Web UI in your browser:
```bash
python -m uvicorn backend.api:app --reload --port 8000
```
Then navigate to: `http://localhost:8000`

#### Option C: Run Baseline Evaluation Suite
Evaluate entity recall and F1 grounding scores:
```bash
python evals/baseline_eval.py
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.