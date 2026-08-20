"""
agents/research_agents.py
---------------------------------------------------------------------
7 Specialist Research Agents optimized for Two-Memory Architecture:

  literature_scout()   - searches academic papers + Wikipedia
  mathematician()      - symbolic derivations with SymPy & LaTeX
  engineer()           - engineering constraints & unit checks
  numerical_analyst()  - numerical computation & data analysis
  peer_reviewer()      - structured methodology & validity critique
  synthesizer()        - combines findings from shared memory
  report_writer()      - section-based publication report generation
---------------------------------------------------------------------
"""

from agents.base_agent import Agent
from tools.research_tools import (
    ArxivSearchTool, WikipediaTool, SymPyTool,
    NumericalTool, UnitConverterTool, LatexFormatterTool,
)
from tools.builtin_tools import WebSearchTool


# ---------------------------------------------------------------------
# 1. Literature Scout
# ---------------------------------------------------------------------

def literature_scout(
    name: str = "LiteratureScout",
    verbose: bool = True,
) -> Agent:
    """
    Searches arXiv and Wikipedia for relevant papers, equations, and methods.
    Outputs concise, information-dense research facts with source provenance.
    """
    agent = Agent(
        name=name,
        role=(
            "You are an expert research analyst. Search academic literature and extract "
            "precise findings for the research query.\n\n"
            "Guidelines:\n"
            "  1. Use arXiv for recent peer-reviewed research papers.\n"
            "  2. Use Wikipedia for foundational concepts.\n"
            "  3. Extract core equations, key methods, numerical bounds, and author citations.\n"
            "  4. Note state of the art, contradictions, and open research gaps.\n"
            "  5. Be concise and information-dense. Avoid conversational filler.\n"
            "  6. Preserve source provenance. Never fabricate citations.\n\n"
            "Output format:\n"
            "  ## Key Papers & Sources\n"
            "  ## Core Equations & Definitions\n"
            "  ## State of the Art & Findings\n"
            "  ## Contradictions & Open Gaps"
        ),
        temperature=0.2,
        verbose=verbose,
    )
    agent.register_tool(ArxivSearchTool(max_results=3))
    agent.register_tool(WikipediaTool(sentences=8))
    agent.register_tool(WebSearchTool(max_results=3))
    return agent


# ---------------------------------------------------------------------
# 2. Mathematician
# ---------------------------------------------------------------------

def mathematician(
    name: str = "Mathematician",
    verbose: bool = True,
) -> Agent:
    """
    Performs symbolic mathematics, derives equations, and outputs LaTeX equations.
    """
    agent = Agent(
        name=name,
        role=(
            "You are an applied mathematician. Derive governing equations and perform "
            "symbolic mathematical analysis with exact precision.\n\n"
            "Guidelines:\n"
            "  1. State governing equations, key derivation steps, and core assumptions.\n"
            "  2. Use sympy_math for symbolic operations.\n"
            "  3. Format all equations in LaTeX using latex_formatter ($...$ or $$...$$).\n"
            "  4. Avoid verbose algebraic step-by-step prose. Focus on key steps and final results.\n"
            "  5. Verify results via limiting cases or physical boundary conditions.\n\n"
            "Output format:\n"
            "  ## Governing Equations & Assumptions\n"
            "  ## Key Derivations\n"
            "  ## Final Symbolic Result\n"
            "  ## Validity & Boundary Verification"
        ),
        temperature=0.1,
        verbose=verbose,
    )
    agent.register_tool(SymPyTool())
    agent.register_tool(LatexFormatterTool())
    return agent


# ---------------------------------------------------------------------
# 3. Engineer
# ---------------------------------------------------------------------

def engineer(
    name: str = "Engineer",
    domain: str = "general",
    verbose: bool = True,
) -> Agent:
    """
    Applies engineering judgment, sanity-checks numbers, and handles unit conversions.
    """
    domain_context = {
        "power": "power electronics and device thermal management.",
        "rf": "RF and microwave circuits.",
        "semiconductor": "semiconductor device physics and compact modeling.",
        "thermal": "thermal resistance networks and heat dissipation.",
        "general": "applied physics and engineering domains.",
    }.get(domain, "applied engineering.")

    agent = Agent(
        name=name,
        role=(
            f"You are a senior engineer specializing in {domain_context}\n\n"
            "Guidelines:\n"
            "  1. Sanity-check numbers and physical constraints (breakdown voltages, temperature limits).\n"
            "  2. Verify unit consistency using unit_converter.\n"
            "  3. Identify dominant physical mechanisms vs negligible effects.\n"
            "  4. Keep output concise, factual, and direct.\n\n"
            "Output format:\n"
            "  ## Engineering Assessment\n"
            "  ## Unit & Dimensional Analysis\n"
            "  ## Physical Constraints & Sanity Checks\n"
            "  ## Practical Recommendations"
        ),
        temperature=0.2,
        verbose=verbose,
    )
    agent.register_tool(UnitConverterTool())
    agent.register_tool(NumericalTool())
    return agent


# ---------------------------------------------------------------------
# 4. Numerical Analyst
# ---------------------------------------------------------------------

def numerical_analyst(
    name: str = "NumericalAnalyst",
    verbose: bool = True,
) -> Agent:
    """
    Runs numerical computations, evaluates expressions, and analyzes quantitative data.
    """
    agent = Agent(
        name=name,
        role=(
            "You are a numerical analyst. Evaluate equations to concrete numbers "
            "and perform quantitative computations.\n\n"
            "Guidelines:\n"
            "  1. Use numerical tool for matrix, polynomial root, and numeric evaluations.\n"
            "  2. Carry full precision through intermediate calculations and state units.\n"
            "  3. Avoid verbose step-by-step arithmetic commentary.\n"
            "  4. Present numerical results cleanly with engineering notation.\n\n"
            "Output format:\n"
            "  ## Numeric Parameters & Setup\n"
            "  ## Numerical Computation Results\n"
            "  ## Precision & Physical Interpretation"
        ),
        temperature=0.1,
        verbose=verbose,
    )
    agent.register_tool(NumericalTool())
    agent.register_tool(UnitConverterTool())
    return agent


# ---------------------------------------------------------------------
# 5. Peer Reviewer
# ---------------------------------------------------------------------

def peer_reviewer(
    name: str = "PeerReviewer",
    verbose: bool = True,
) -> Agent:
    """
    Critiques scientific findings like a journal referee.
    """
    agent = Agent(
        name=name,
        role=(
            "You are an expert peer reviewer for top scientific journals.\n\n"
            "Evaluate research findings on:\n"
            "  - Scientific validity and equation correctness\n"
            "  - Justification of physical assumptions\n"
            "  - Unit consistency and completeness\n"
            "  - Methodological rigor\n\n"
            "Output format:\n"
            "  ## Overall Rating & Summary\n"
            "  ## Major Concerns (Must Fix)\n"
            "  ## Minor Concerns & Strengths\n"
            "  ## Actionable Recommendations"
        ),
        temperature=0.2,
        verbose=verbose,
    )
    return agent


# ---------------------------------------------------------------------
# 6. Synthesizer
# ---------------------------------------------------------------------

def synthesizer(
    name: str = "Synthesizer",
    verbose: bool = True,
) -> Agent:
    """
    Synthesizes findings retrieved from Working Memory & Long-Term Memory into a unified summary.
    """
    agent = Agent(
        name=name,
        role=(
            "You are an expert knowledge synthesizer.\n\n"
            "Your task is to retrieve and combine findings from shared Working Memory and Long-Term Memory "
            "into an authoritative, unified research summary.\n\n"
            "Guidelines:\n"
            "  1. Resolve contradictions between literature, mathematical derivations, and numerical results.\n"
            "  2. Connect physical theory to engineering implications.\n"
            "  3. Distinguish established evidence from assumptions and remaining uncertainties.\n"
            "  4. Ensure unified notation, LaTeX equations, and SI units.\n\n"
            "Output format:\n"
            "  ## Unified Technical Understanding\n"
            "  ## Core Governing Equations\n"
            "  ## Validated Numerical Results & Engineering Bounds\n"
            "  ## Resolved Contradictions & Remaining Uncertainties"
        ),
        temperature=0.3,
        verbose=verbose,
    )
    return agent


# ---------------------------------------------------------------------
# 7. Report Writer
# ---------------------------------------------------------------------

def report_writer(
    name: str = "ReportWriter",
    verbose: bool = True,
) -> Agent:
    """
    Generates structured, publication-quality technical report sections from shared memory.
    """
    agent = Agent(
        name=name,
        role=(
            "You are a technical report writer preparing publication-quality academic report sections.\n\n"
            "Guidelines:\n"
            "  1. Retrieve required research findings from shared memory.\n"
            "  2. Write with academic precision, passive voice, and clear prose.\n"
            "  3. Format ALL equations in LaTeX ($...$ or $$...$$).\n"
            "  4. Include exact numbers, units, and source citations.\n"
            "  5. Prepare content suitable for section-by-section long-form report assembly.\n\n"
            "Output format:\n"
            "  # Section Title\n"
            "  [Prose content with inline LaTeX equations, data tables, and citations]"
        ),
        temperature=0.3,
        verbose=verbose,
    )
    agent.register_tool(LatexFormatterTool())
    return agent