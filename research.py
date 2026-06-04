"""
research.py
---------------------------------------------------------------------
The Research System - Interactive Terminal Interface

Run this to start the research system:
    python research.py

Modes:
  * Quick    - Literature + Computation + Report (~3-5 minutes)
  * Standard - + Mathematics + Engineering (~6-10 minutes)
  * Deep     - All 8 phases including Review + Synthesis (~10-15 min)

All reports are saved as markdown in the ./reports/ folder.
---------------------------------------------------------------------
"""

import os
from colorama import Fore, Style, init

from agents.research_agents import (
    literature_scout, mathematician, engineer,
    numerical_analyst, peer_reviewer, synthesizer, report_writer,
)
from orchestrator import ResearchOrchestrator

init(autoreset=True)


# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------

def print_banner():
    print(f"""
{Fore.CYAN}
+------------------------------------------------------------+
|              RESEARCH AGENT SYSTEM                         |
|   Literature * Mathematics * Numerics * Engineering         |
+------------------------------------------------------------+
{Style.RESET_ALL}""")


def get_input(prompt: str) -> str:
    try:
        return input(f"{Fore.CYAN}{prompt}{Style.RESET_ALL}").strip()
    except (KeyboardInterrupt, EOFError):
        return ""


def print_depth_menu():
    print(f"""
{Fore.YELLOW}Select research depth:{Style.RESET_ALL}

  {Fore.GREEN}[1] QUICK{Style.RESET_ALL}     - Literature + Computation + Report
              Best for: quick overviews, known topics
              Phases: Understand -> Literature -> Computation -> Report

  {Fore.GREEN}[2] STANDARD{Style.RESET_ALL}  - + Mathematics + Engineering  (recommended)
              Best for: technical topics needing derivations
              Phases: + Mathematics + Engineering

  {Fore.GREEN}[3] DEEP{Style.RESET_ALL}      - All 8 phases including peer review
              Best for: novel questions, publishable-quality output
              Phases: + Review + Synthesis

  {Fore.WHITE}[q] Quit{Style.RESET_ALL}
""")


def print_examples():
    print(f"""
{Fore.WHITE}Example research questions:{Style.RESET_ALL}

  Semiconductor / HEMT:
    * "Derive the 2DEG charge density equation in a GaN HEMT and
       approximate it using the smooth transition function"
    * "Explain and derive the subthreshold swing in MOSFETs from
       first principles and calculate it at 300K"
    * "What is the impact ionization model in GaN HEMTs and how
       does it affect breakdown voltage?"

  Mathematics / Physics:
    * "Derive the Fourier transform of a Gaussian pulse and compute
       its bandwidth for sigma = 1 ps"
    * "Solve the 1D Schrodinger equation for a finite potential well
       and find the quantised energy levels"
    * "Derive the small-signal equivalent circuit of a MOSFET from
       its large-signal model"

  Engineering:
    * "Design a GaN HEMT power amplifier for 2.4 GHz with 10W output
       - calculate key parameters"
    * "Compare thermal resistance of GaN-on-SiC vs GaN-on-Si for
       a 5mm x 5mm die at 10W dissipation"
""")


# ---------------------------------------------------------------------
# Build the research orchestrator
# ---------------------------------------------------------------------

def build_orchestrator(depth: str, verbose_agents: bool = False) -> ResearchOrchestrator:
    """
    Instantiate all research agents and wire them into the orchestrator.
    verbose_agents=False keeps agent output quiet so only
    the orchestrator's phase headers are visible.
    """
    orc = ResearchOrchestrator(
        depth      = depth,
        output_dir = "reports",
        verbose    = True,
    )

    orc.register_agents(
        scout         = literature_scout(verbose=verbose_agents),
        mathematician = mathematician(verbose=verbose_agents),
        engineer      = engineer(domain="semiconductor", verbose=verbose_agents),
        numerical     = numerical_analyst(verbose=verbose_agents),
        reviewer      = peer_reviewer(verbose=verbose_agents),
        synthesizer   = synthesizer(verbose=verbose_agents),
        writer        = report_writer(verbose=verbose_agents),
    )

    return orc


# ---------------------------------------------------------------------
# Individual tool demos (for testing without a full research run)
# ---------------------------------------------------------------------

def demo_tools():
    """Quick demo of every research tool in isolation."""
    print(f"\n{Fore.CYAN}-- Tool Demo ------------------------------------------{Style.RESET_ALL}")

    from tools.research_tools import (
        SymPyTool, NumericalTool, UnitConverterTool,
        LatexFormatterTool, WikipediaTool
    )

    print(f"\n{Fore.YELLOW}[SymPy - solve quadratic]{Style.RESET_ALL}")
    s = SymPyTool()
    print(s.run("solve | x**2 - 5*x + 6 | x"))

    print(f"\n{Fore.YELLOW}[SymPy - Taylor series of softplus]{Style.RESET_ALL}")
    print(s.run("series | log(1 + exp(x)) | x, 0, 6"))

    print(f"\n{Fore.YELLOW}[SymPy - integrate Gaussian]{Style.RESET_ALL}")
    print(s.run("integrate | exp(-x**2) | x, -oo, oo"))

    print(f"\n{Fore.YELLOW}[Numerical - evaluate 2*pi*f*C]{Style.RESET_ALL}")
    n = NumericalTool()
    print(n.run("evaluate | 2 * np.pi * 2.4e9 * 50e-12"))

    print(f"\n{Fore.YELLOW}[Numerical - polynomial roots]{Style.RESET_ALL}")
    print(n.run("roots | [1, -6, 11, -6]"))

    print(f"\n{Fore.YELLOW}[Unit Converter]{Style.RESET_ALL}")
    u = UnitConverterTool()
    for q in ["1 eV to J", "2.4 GHz to Hz", "300 K to C", "0.001 W to dBm", "100 nm to um"]:
        print(f"  {q}  ->  {u.run(q)}")

    print(f"\n{Fore.YELLOW}[LaTeX Formatter]{Style.RESET_ALL}")
    lt = LatexFormatterTool()
    print(lt.run("Integral(exp(-x**2), (x, -oo, oo))"))

    print(f"\n{Fore.YELLOW}[Wikipedia - GaN]{Style.RESET_ALL}")
    w = WikipediaTool(sentences=4)
    print(w.run("Gallium nitride"))


# ---------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------

def main():
    print_banner()

    while True:
        print_depth_menu()
        choice = get_input("Choose depth -> ")

        if not choice or choice.lower() == "q":
            print(f"\n{Fore.CYAN}Reports saved in ./reports/{Style.RESET_ALL}\n")
            break

        if choice == "0":
            demo_tools()
            continue

        depth_map = {"1": "quick", "2": "standard", "3": "deep"}
        if choice not in depth_map:
            print(f"{Fore.RED}Invalid choice. Enter 1, 2, 3, or q.{Style.RESET_ALL}")
            continue

        depth = depth_map[choice]

        # Show examples and get question
        print_examples()
        question = get_input("Research question -> ")
        if not question:
            continue

        # Optional: show agent verbose output
        show_agents = get_input(
            "Show detailed agent output? (y/n, default n) -> "
        ).lower() == "y"

        print(f"\n{Fore.CYAN}Building research team...{Style.RESET_ALL}")
        orc = build_orchestrator(depth, verbose_agents=show_agents)

        # Run
        report = orc.run(question)
        orc.print_log()

        again = get_input("\nResearch another question? (y/n) -> ").lower()

        if again != "y":
            print(f"\n{Fore.CYAN}All reports saved in ./reports/{Style.RESET_ALL}\n")
            break


if __name__ == "__main__":
    main()