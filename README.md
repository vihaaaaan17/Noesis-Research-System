# Research Agent System — Powered by Google Gemini

An academic-grade Multi-Agent System built in Python, designed to perform structured scientific research. The system coordinates specialized agents through an 8-phase pipeline mirroring real-world research workflows, backed by symbolic mathematics, scientific computing, unit conversion, and academic literature tools.

---

## 1. System Architecture: The 8-Phase Pipeline

Unlike basic agents that attempt to answer complex prompts in a single turn, this system decomposes research questions into structured sub-problems and runs them through a sequential, gated **8-Phase Pipeline**:

```
[1. UNDERSTAND] -> [2. LITERATURE] -> [3. MATHEMATICS] -> [4. COMPUTATION]
                                                                |
[8. REPORT]     <- [7. SYNTHESIZE] <- [6. PEER REVIEW]  <- [5. ENGINEERING]
```

1. **Phase 1 — UNDERSTAND (Decomposition)**:
   * *Agent*: Research Planner (Orchestrator-level direct thinking).
   * *Role*: Deconstructs the research question into 3-5 key sub-problems, identifying domains, physical variables, and expected outputs.
2. **Phase 2 — LITERATURE (Academic Search)**:
   * *Agent*: `LiteratureScout`
   * *Tools*: ArxivSearch, Wikipedia, WebSearch.
   * *Role*: Identifies governing equations, background information, and state-of-the-art results from peer-reviewed literature.
3. **Phase 3 — MATHEMATICS (Symbolic Derivation)**:
   * *Agent*: `Mathematician`
   * *Tools*: SymPy, LaTeX Formatter.
   * *Role*: Performs step-by-step analytical derivations, simplifies expressions, and structures mathematical output in LaTeX format.
4. **Phase 4 — COMPUTATION (Numerical Analysis)**:
   * *Agent*: `NumericalAnalyst`
   * *Tools*: NumPy/SciPy, Calculator.
   * *Role*: Evaluates mathematical models numerically, solves systems of equations, computes characteristic scales, and processes vectors/matrices.
5. **Phase 5 — ENGINEERING (Assessment & Checks)**:
   * *Agent*: `Engineer`
   * *Tools*: Unit Converter, Calculator, Numerical.
   * *Role*: Applies dimensional consistency checks, identifies negligible vs. dominant terms, maps physical constraints (material bounds, tolerances), and verifies physical plausibility.
6. **Phase 6 — PEER REVIEW (Refined Critique)**:
   * *Agent*: `PeerReviewer`
   * *Role*: Critiques the full body of collected findings as a journal referee, scoring it for scientific validity, missing bounds, and clarity.
7. **Phase 7 — SYNTHESIZE (Coherence Gate)**:
   * *Agent*: `Synthesizer`
   * *Role*: Merges literature, derivations, calculations, and reviewer critiques into a unified, self-contained summary, resolving any conflicting results.
8. **Phase 8 — REPORT (Academic Writeup)**:
   * *Agent*: `ReportWriter`
   * *Tools*: LaTeX Formatter.
   * *Role*: Generates a publication-quality technical report complete with Abstract, Introduction, Theory, Analysis, Results, Discussion, and Conclusion. Reports are saved directly to `./reports/` as markdown.

---

## 2. Key Technical Features

### A. Dedicated Research Tools
The agents utilize a suite of custom Python-wrapped tools to guarantee exact precision:
* **SymPyTool**: Performs symbolic computations (solving equations, derivatives, indefinite/definite integration, Taylor series, matrix inversions).
* **NumericalTool**: Harnesses NumPy and SciPy to solve matrix equations ($Ax = b$), compute polynomial roots, sample rates, stats, and convert ratios to decibels.
* **UnitConverterTool**: Handles conversions across engineering scales (length, energy, capacitance, pressure, temperature, power, charge, time) using SI base units.
* **ArxivSearchTool**: Queries the official arXiv API to retrieve paper titles, author lists, publication years, abstracts, and entry URLs.
* **WikipediaTool**: Connects to the Wikipedia API to retrieve structured article summaries.
* **LatexFormatterTool**: Translates SymPy expressions into clean LaTeX code.

### B. Robust ReAct Reasoning Loop
Agents requiring tools run within an decoupled, externally driven **ReAct Loop** (`core/react_loop.py`). 
* **Action Protocol**: The LLM outputs `TOOL_CALL: <tool_name> | <tool_input>`, halts generation, receives the tool's `Observation: <result>` response, and continues.
* **Finalization Protocol**: Once the agent has sufficient information, it returns `FINAL_ANSWER: <answer>`. If no tool is required, it returns a direct response, which is handled gracefully.

### C. Rate Limit & API Fault Tolerance
The system is built to operate reliably on standard and free-tier API endpoints:
* **Orchestrator & Agent Retries**: Exponential backoff loops automatically intercept rate limit issues (`ResourceExhausted` 429 warnings) and retry calls.
* **Phase Spacing Delays**: A 2.0-second delay is enforced between orchestrator phases to naturally stagger API requests and prevent burst blocks.
* **arXiv Cooldown Rate-Limiter**: Automatically spaces out arXiv API queries by a minimum of 3.0 seconds, including a 3-attempt backoff retry loop for HTTP 429/503 responses.

### D. Windows Console Encoding Safety (cp1252)
To prevent the common `UnicodeEncodeError` crashes on Windows terminals defaulting to legacy `cp1252` encoding, the codebase is **100% ASCII-compliant**. All box drawing (`─`), em-dashes (`—`), arrows (`→`), middle dots (`·`), and mathematical operators (`∫`, `Ω`) in printed strings and source comments are converted to clean ASCII equivalents.

### E. Memory Architecture
* **Short-Term Memory**: Conversation history inside the context window is managed via either a **Sliding Window** (dropping old messages) or **Summarization** (running the agent's LLM to compress the oldest half of the history into bullet points).
* **Long-Term Memory**: Backed by a local JSON file (`long_term_memory.json`). It persists direct facts (key-value) and searchable text notes, using keyword-overlap matching to rank and inject relevant context on initialization.

---

## 3. Why It is Better Than Basic Agent Systems

| Feature | Basic Agent / ReAct Loop | This Research System |
| :--- | :--- | :--- |
| **Logic Segmentation** | Single prompt trying to solve the problem in one turn. | Structured 8-phase gates separating research from writeup. |
| **Precision** | LLM attempts to do math and equations in its head. | SymPy and NumPy/SciPy execution environments. |
| **Verification** | Assumes the output is correct. | Peer review and engineering sanity check phases. |
| **Context Control** | History grows until context limit crashes the app. | Automated sliding window or LLM summarization. |
| **Stability** | Rapid burst calls trigger 429 API blocks. | Exponential backoffs, phase spacing, and tool cooldowns. |
| **Windows Support** | Unicode symbols crash standard terminals. | Strict ASCII-safe console layer. |

---

## 4. Project Structure

```
MAS/
│
├── core/
│   ├── __init__.py          # Core package initializer
│   └── react_loop.py        # External ReAct loop driving agent tools
│
├── agents/
│   ├── __init__.py          # Exports factories and agent instances
│   ├── base_agent.py        # Base Agent class with memory and tool attachment
│   └── research_agents.py   # Specialized agent factories (Scout, Mathematician, etc.)
│
├── tools/
│   ├── __init__.py          # Exports tools
│   ├── base_tool.py         # Abstract BaseTool template
│   ├── builtin_tools.py     # Calculator and Web Search tools
│   └── research_tools.py    # Math, SciPy, Converter, Arxiv, and Wikipedia tools
│
├── memory/
│   ├── short_term.py        # Context sliding window and LLM summarizer
│   └── long_term.py         # Persistent JSON fact and note store
│
├── reports/                 # Output folder for generated Markdown reports
├── research.py              # Interactive CLI Runner (Main Entrypoint)
├── config.py                # System settings (Model defaults, token limits)
├── requirements.txt         # Project dependencies
└── .env                     # Secrets file (ignored from version control)
```

---

## 5. Getting Started

### Installation
1. Clone the repository and navigate to the project directory:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY="your_google_gemini_api_key_here"
   ```
   *(Alternatively, you can set the key directly inside `config.py` on line 6).*

### Usage
Run the interactive terminal app:
```bash
python research.py
```

### Depth Settings
* **QUICK** (Phases: Understand -> Literature -> Computation -> Report): Best for fast overviews.
* **STANDARD** (Phases: Understand -> Literature -> Mathematics -> Computation -> Engineering -> Report): Balanced execution (Recommended).
* **DEEP** (All 8 phases): Exhaustive research including peer review and synthesis.