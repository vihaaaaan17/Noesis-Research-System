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
{Fore.YELLOW}Select option:{Style.RESET_ALL}

  {Fore.GREEN}[1] QUICK{Style.RESET_ALL}     - Literature + Computation + Report
              Best for: quick overviews, known topics
              Phases: Understand -> Literature -> Computation -> Report

  {Fore.GREEN}[2] STANDARD{Style.RESET_ALL}  - + Mathematics + Engineering  (recommended)
              Best for: technical topics needing derivations
              Phases: + Mathematics + Engineering

  {Fore.GREEN}[3] DEEP{Style.RESET_ALL}      - All 8 phases including peer review
              Best for: novel questions, publishable-quality output
              Phases: + Review + Synthesis

  {Fore.CYAN}[4] UTILITIES{Style.RESET_ALL} - Research Skills Toolkit (Interactive 5-in-1 Tool)
              Run paper critique, gap analysis, synthesis, etc.

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


def run_skills_toolkit():
    import os
    import time
    from tools.research_skills import ResearchSkillsTool

    skills_map = {
        "1": ("literature_critique", "Literature Critique (Claims, Steel-man, Objections)"),
        "2": ("gap_finder", "Gap Finder (Unresolved questions, Study topics)"),
        "3": ("synthesis_drafter", "Synthesis Drafter (Notes cleaner, Related works section)"),
        "4": ("concept_mapper", "Concept Mapper (Concept explanation, Analogies, Argument chain)"),
        "5": ("academic_refinement", "Academic Refinement (Abstract rewrite, Devil's advocate, Brief)")
    }

    while True:
        print(f"\n{Fore.CYAN}=== RESEARCH SKILLS TOOLKIT ==={Style.RESET_ALL}")
        print("Select a consolidated academic skill to execute:")
        for k, v in skills_map.items():
            print(f"  {Fore.GREEN}[{k}]{Style.RESET_ALL} {v[1]}")
        print(f"  {Fore.WHITE}[b] Back to main menu{Style.RESET_ALL}\n")

        choice = get_input("Choose skill -> ")
        if choice.lower() == 'b' or not choice:
            break

        if choice not in skills_map:
            print(f"{Fore.RED}Invalid choice.{Style.RESET_ALL}")
            continue

        skill_key, skill_desc = skills_map[choice]
        print(f"\n{Fore.YELLOW}Selected: {skill_desc}{Style.RESET_ALL}")
        print("Please enter/paste your text below, or enter a valid file path to read from:")
        
        # Collect multi-line text input or a single path
        user_text = get_input("Text or File Path -> ")
        if not user_text:
            print(f"{Fore.RED}No input provided.{Style.RESET_ALL}")
            continue

        # Check if file path
        if os.path.isfile(user_text):
            try:
                print(f"{Fore.CYAN}Reading from file: {user_text}...{Style.RESET_ALL}")
                with open(user_text, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f"{Fore.RED}Error reading file: {e}{Style.RESET_ALL}")
                continue
        else:
            content = user_text

        if not content.strip():
            print(f"{Fore.RED}Content is empty.{Style.RESET_ALL}")
            continue

        print(f"\n{Fore.CYAN}Invoking academic intelligence loop...{Style.RESET_ALL}")
        tool = ResearchSkillsTool()
        result = tool.run(f"{skill_key} | {content}")

        # Display result beautifully
        print(f"\n{Fore.GREEN}{'='*60}")
        print(f"{Fore.GREEN}ANALYSIS RESULT - {skill_desc.upper()}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        print(result)
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")

        # Offer to save
        save_choice = get_input("\nSave this analysis to a report file? (y/n, default y) -> ").lower()
        if save_choice != 'n':
            os.makedirs("reports", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"reports/skill_analysis_{skill_key}_{timestamp}.md"
            
            report_content = f"""# Research Skills Toolkit Analysis
- **Skill Applied:** {skill_desc}
- **Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}

---

## Output Result

{result}

---
## Input Analyzed
```text
{content[:1000]}...
```
"""
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(report_content)
                print(f"{Fore.GREEN}Saved successfully to: {filename}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}Failed to save report: {e}{Style.RESET_ALL}")

        # Ask to run another skill
        again = get_input("\nRun another toolkit analysis? (y/n, default y) -> ").lower()
        if again == 'n':
            break


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

        if choice == "4":
            run_skills_toolkit()
            continue

        depth_map = {"1": "quick", "2": "standard", "3": "deep"}
        if choice not in depth_map:
            print(f"{Fore.RED}Invalid choice. Enter 1, 2, 3, 4, or q.{Style.RESET_ALL}")
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