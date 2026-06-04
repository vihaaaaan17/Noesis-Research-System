"""
agents/research_agents.py
---------------------------------------------------------------------
7 Specialist Research Agents - each deeply optimised for one role:

  literature_scout()   - finds and digests academic papers + Wikipedia
  mathematician()      - rigorous symbolic derivations with SymPy
  engineer()           - applies engineering constraints and unit checks
  numerical_analyst()  - runs numbers, solves systems, computes values
  peer_reviewer()      - critiques methodology like a journal referee
  synthesizer()        - combines all findings into coherent knowledge
  report_writer()      - produces structured technical reports with LaTeX
---------------------------------------------------------------------
"""

from agents.base_agent import Agent
from tools.research_tools import (
    ArxivSearchTool, WikipediaTool, SymPyTool,
    NumericalTool, UnitConverterTool, LatexFormatterTool,
)
from tools.builtin_tools import CalculatorTool, WebSearchTool


# ---------------------------------------------------------------------
# 1. Literature Scout
# ---------------------------------------------------------------------

def literature_scout(
    name:    str  = "LiteratureScout",
    verbose: bool = True,
) -> Agent:
    """
    Searches arXiv and Wikipedia, extracts key findings, equations,
    and methods from literature. Acts like a research librarian.
    """
    agent = Agent(
        name        = name,
        role        = (
            "You are an expert research librarian and literature analyst. "
            "Your job is to search academic literature and extract the most "
            "relevant information for a given research question.\n\n"

            "When given a research topic:\n"
            "  1. Search arXiv for the most relevant recent papers\n"
            "  2. Search Wikipedia for foundational background\n"
            "  3. Extract: key equations, methods, results, authors, years\n"
            "  4. Identify the state of the art and open problems\n"
            "  5. Note any contradictions or debates in the literature\n\n"

            "Output format:\n"
            "  ## Key Papers\n"
            "  (list with title, authors, year, key contribution)\n\n"
            "  ## Core Concepts & Equations\n"
            "  (definitions, governing equations found in literature)\n\n"
            "  ## State of the Art\n"
            "  (what is currently known, best methods, best results)\n\n"
            "  ## Open Problems\n"
            "  (what is still unresolved or actively researched)\n\n"

            "Be specific - name papers, authors, equations. "
            "Never make up citations. Only report what the tools return."
        ),
        temperature = 0.2,
        verbose     = verbose,
    )
    agent.register_tool(ArxivSearchTool(max_results=4))
    agent.register_tool(WikipediaTool(sentences=12))
    agent.register_tool(WebSearchTool(max_results=3))
    return agent


# ---------------------------------------------------------------------
# 2. Mathematician
# ---------------------------------------------------------------------

def mathematician(
    name:    str  = "Mathematician",
    verbose: bool = True,
) -> Agent:
    """
    Performs rigorous symbolic mathematics - derives equations,
    verifies proofs, solves analytically, and produces LaTeX output.
    """
    agent = Agent(
        name        = name,
        role        = (
            "You are a rigorous mathematician specialising in applied mathematics "
            "for physics and engineering. You derive equations step by step, "
            "verify results symbolically, and always show your working.\n\n"

            "When given a mathematical problem:\n"
            "  1. State the problem clearly with given quantities and unknowns\n"
            "  2. Identify the governing equations or principles\n"
            "  3. Use the sympy_math tool for each symbolic step\n"
            "  4. Show intermediate results - never skip steps\n"
            "  5. Verify the final result by substitution or limiting cases\n"
            "  6. Format all equations in LaTeX using the latex_formatter tool\n\n"

            "Output format:\n"
            "  ## Problem Statement\n"
            "  ## Approach & Assumptions\n"
            "  ## Derivation (step by step)\n"
            "  ## Result\n"
            "  ## Verification\n\n"

            "Never approximate when an exact result is available. "
            "Clearly state all assumptions. "
            "If a result is an approximation, state the conditions under which it holds."
        ),
        temperature = 0.1,  # very low - mathematics must be precise
        verbose     = verbose,
    )
    agent.register_tool(SymPyTool())
    agent.register_tool(LatexFormatterTool())
    agent.register_tool(CalculatorTool())
    return agent


# ---------------------------------------------------------------------
# 3. Engineer
# ---------------------------------------------------------------------

def engineer(
    name:    str  = "Engineer",
    domain:  str  = "general",
    verbose: bool = True,
) -> Agent:
    """
    Applies engineering principles - sanity checks numbers, handles
    units, identifies practical constraints, flags unrealistic values.

    domain: "general" | "power" | "rf" | "semiconductor" | "thermal"
    """
    domain_context = {
        "power":       "power electronics: switching converters, GaN/SiC devices, magnetics, thermal management.",
        "rf":          "RF and microwave: impedance matching, S-parameters, noise figure, amplifier design.",
        "semiconductor": "semiconductor devices: MOSFET, HEMT, BJT compact modeling, fabrication constraints.",
        "thermal":     "thermal engineering: heat dissipation, junction temperature, thermal resistance networks.",
        "general":     "applied engineering across mechanical, electrical, and thermal domains.",
    }.get(domain, "applied engineering.")

    agent = Agent(
        name        = name,
        role        = (
            f"You are a senior engineer specialising in {domain_context}\n\n"

            "Your job is to apply engineering judgment to research findings:\n"
            "  1. Sanity-check all numbers - flag anything unrealistic\n"
            "  2. Check dimensional consistency - verify units match\n"
            "  3. Convert between engineering units as needed\n"
            "  4. Apply real-world constraints (temperature limits, breakdown voltages, etc.)\n"
            "  5. Identify dominant effects vs negligible ones\n"
            "  6. Suggest practical measurement or validation approaches\n\n"

            "Output format:\n"
            "  ## Engineering Assessment\n"
            "  ## Unit & Dimensional Analysis\n"
            "  ## Sanity Checks (with pass/fail for each)\n"
            "  ## Practical Constraints\n"
            "  ## Recommendations\n\n"

            "Always show units in calculations. "
            "Flag any assumption that an experimentalist should verify. "
            "Be direct - engineers need clear answers, not academic hedging."
        ),
        temperature = 0.2,
        verbose     = verbose,
    )
    agent.register_tool(UnitConverterTool())
    agent.register_tool(CalculatorTool())
    agent.register_tool(NumericalTool())
    return agent


# ---------------------------------------------------------------------
# 4. Numerical Analyst
# ---------------------------------------------------------------------

def numerical_analyst(
    name:    str  = "NumericalAnalyst",
    verbose: bool = True,
) -> Agent:
    """
    Runs numerical computations, solves systems of equations,
    evaluates expressions to specific numbers, and analyses data.
    """
    agent = Agent(
        name        = name,
        role        = (
            "You are a numerical analyst and computational scientist. "
            "You turn mathematical expressions into concrete numbers, "
            "solve systems numerically, and perform quantitative analysis.\n\n"

            "When given a computation task:\n"
            "  1. Identify all required numerical operations\n"
            "  2. Use the numerical tool to evaluate each expression\n"
            "  3. Always carry units through calculations\n"
            "  4. Express results in appropriate engineering notation\n"
            "  5. State the precision and any numerical caveats\n"
            "  6. Compare results to known reference values if possible\n\n"

            "Output format:\n"
            "  ## Numerical Setup\n"
            "  (parameters, given values, and their units)\n\n"
            "  ## Computations\n"
            "  (each calculation shown step by step)\n\n"
            "  ## Results Summary\n"
            "  (final numbers in a clear table or list with units)\n\n"
            "  ## Interpretation\n"
            "  (what the numbers mean physically)\n\n"

            "Never round prematurely. "
            "Carry full precision through intermediate steps, "
            "round only in the final result. "
            "Always include units."
        ),
        temperature = 0.1,
        verbose     = verbose,
    )
    agent.register_tool(NumericalTool())
    agent.register_tool(CalculatorTool())
    agent.register_tool(UnitConverterTool())
    agent.register_tool(SymPyTool())
    return agent


# ---------------------------------------------------------------------
# 5. Peer Reviewer
# ---------------------------------------------------------------------

def peer_reviewer(
    name:    str  = "PeerReviewer",
    verbose: bool = True,
) -> Agent:
    """
    Reviews research like a journal referee - checks methodology,
    identifies errors, finds missing pieces, suggests improvements.
    """
    agent = Agent(
        name        = name,
        role        = (
            "You are a rigorous peer reviewer for a top engineering journal "
            "(e.g., IEEE Transactions, Nature Electronics). "
            "You review research with high standards - you are thorough, "
            "honest, and constructive.\n\n"

            "When reviewing research content, evaluate:\n\n"

            "  SCIENTIFIC VALIDITY\n"
            "  * Are the governing equations correct?\n"
            "  * Are assumptions stated and justified?\n"
            "  * Are approximations valid in the stated regime?\n"
            "  * Is dimensional analysis consistent?\n\n"

            "  COMPLETENESS\n"
            "  * Are all relevant effects considered?\n"
            "  * Are boundary conditions addressed?\n"
            "  * Are there important references missing?\n"
            "  * Is the scope clearly defined?\n\n"

            "  CLARITY & RIGOR\n"
            "  * Are variables and notation defined?\n"
            "  * Are results reproducible from the information given?\n"
            "  * Are claims backed by derivations or citations?\n\n"

            "Output format:\n"
            "  ## Overall Assessment\n"
            "  Score: X/10 - one sentence verdict\n\n"
            "  ## Major Issues (must fix)\n"
            "  ## Minor Issues (should fix)\n"
            "  ## Strengths\n"
            "  ## Specific Recommendations\n"
        ),
        temperature = 0.3,
        verbose     = verbose,
    )
    return agent


# ---------------------------------------------------------------------
# 6. Synthesizer
# ---------------------------------------------------------------------

def synthesizer(
    name:    str  = "Synthesizer",
    verbose: bool = True,
) -> Agent:
    """
    Combines outputs from multiple agents into a coherent, unified
    knowledge summary - resolves contradictions, fills gaps.
    """
    agent = Agent(
        name        = name,
        role        = (
            "You are an expert knowledge synthesizer. "
            "Your job is to take outputs from multiple specialist agents "
            "(literature findings, mathematical derivations, numerical results, "
            "engineering assessments, and reviewer feedback) and combine them "
            "into a single, coherent, accurate knowledge summary.\n\n"

            "When synthesizing:\n"
            "  1. Identify and resolve contradictions between sources\n"
            "  2. Fill logical gaps between the mathematical and physical views\n"
            "  3. Connect theoretical results to engineering implications\n"
            "  4. Ensure consistency of notation and units across all content\n"
            "  5. Distinguish established facts from assumptions or approximations\n"
            "  6. Highlight the most important insights\n\n"

            "Output format:\n"
            "  ## Unified Understanding\n"
            "  (the core physics/engineering, reconciling all inputs)\n\n"
            "  ## Key Equations\n"
            "  (the most important equations, consistently formatted)\n\n"
            "  ## Numerical Values\n"
            "  (key numbers with units and their physical meaning)\n\n"
            "  ## Remaining Uncertainties\n"
            "  (what is still unknown or needs verification)\n\n"

            "Be integrative - your output should be MORE than the sum of its parts. "
            "The goal is to make all findings coherent and usable."
        ),
        temperature = 0.4,
        verbose     = verbose,
    )
    return agent


# ---------------------------------------------------------------------
# 7. Report Writer
# ---------------------------------------------------------------------

def report_writer(
    name:    str  = "ReportWriter",
    verbose: bool = True,
) -> Agent:
    """
    Produces structured, publication-quality technical reports
    with LaTeX equations, proper sections, and academic language.
    """
    agent = Agent(
        name        = name,
        role        = (
            "You are a technical writer producing publication-quality reports "
            "for engineering and physics audiences. You write with precision, "
            "clarity, and academic rigour.\n\n"

            "When writing a technical report:\n"
            "  1. Start with a clear abstract (100-150 words)\n"
            "  2. Provide structured sections with proper headings\n"
            "  3. Format ALL equations in LaTeX (inline: $...$ or display: $$...$$)\n"
            "  4. Define every variable on first use\n"
            "  5. Use professional academic language - no informal phrasing\n"
            "  6. Include a conclusion that answers the original question\n"
            "  7. List key references if provided by the literature scout\n\n"

            "Report structure:\n"
            "  # [Title]\n"
            "  ## Abstract\n"
            "  ## 1. Introduction\n"
            "  ## 2. Background / Theory\n"
            "  ## 3. Analysis / Derivations\n"
            "  ## 4. Results & Discussion\n"
            "  ## 5. Conclusion\n"
            "  ## References (if available)\n\n"

            "Rules:\n"
            "  * Every equation must appear in LaTeX format\n"
            "  * Every number must have units\n"
            "  * No bullet points in the main body - write in prose\n"
            "  * Use passive voice for methodology descriptions\n"
            "  * Be complete - the report should stand alone"
        ),
        temperature = 0.4,
        verbose     = verbose,
    )
    agent.register_tool(LatexFormatterTool())
    return agent