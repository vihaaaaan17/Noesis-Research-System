"""
orchestrator/research_orchestrator.py
---------------------------------------------------------------------
The Research Orchestrator - 8-phase structured research pipeline

Unlike the general Orchestrator (which is fully dynamic), this one
has opinionated phases that mirror how real research actually works:

  Phase 1 - UNDERSTAND    : decompose the question
  Phase 2 - LITERATURE    : search papers and background
  Phase 3 - MATHEMATICS   : derive/verify equations
  Phase 4 - COMPUTATION   : evaluate numerically
  Phase 5 - ENGINEERING   : apply real-world constraints
  Phase 6 - REVIEW        : peer-review the findings
  Phase 7 - SYNTHESIZE    : combine everything coherently
  Phase 8 - REPORT        : produce the final document

Key design choices:
  * Each phase is driven by a specific agent
  * Outputs accumulate in a shared research_document dict
  * The peer reviewer can flag issues that trigger re-runs
  * Final output is a full markdown report saved to disk
---------------------------------------------------------------------
"""

import os
import time
from colorama import Fore, Style, init
import google.generativeai as genai
import config

init(autoreset=True)


class ResearchOrchestrator:
    """
    An 8-phase research orchestrator that drives specialist agents
    through a structured investigation of any research question.

    Parameters
    ----------
    question       : str  - the research question (set in run())
    depth          : str  - "quick" | "standard" | "deep"
                           quick    = phases 1,2,4,8     (~fast, surface-level)
                           standard = phases 1,2,3,4,5,8  (~balanced)
                           deep     = all 8 phases         (~exhaustive)
    output_dir     : str  - folder to save the final report
    verbose        : bool - print progress to console
    """

    PHASES = {
        1: ("UNDERSTAND",  "Decomposing the research question"),
        2: ("LITERATURE",  "Searching academic literature"),
        3: ("MATHEMATICS", "Deriving and verifying equations"),
        4: ("COMPUTATION", "Numerical evaluation and analysis"),
        5: ("ENGINEERING", "Engineering assessment and sanity checks"),
        6: ("REVIEW",      "Peer-reviewing the findings"),
        7: ("SYNTHESIZE",  "Synthesizing all findings"),
        8: ("REPORT",      "Writing the final report"),
    }

    DEPTH_PHASES = {
        "quick":    [1, 2, 4, 8],
        "standard": [1, 2, 3, 4, 5, 8],
        "deep":     [1, 2, 3, 4, 5, 6, 7, 8],
    }

    def __init__(
        self,
        depth:      str  = "standard",
        output_dir: str  = "reports",
        verbose:    bool = True,
    ):
        self.depth      = depth
        self.output_dir = output_dir
        self.verbose    = verbose

        # Research document - accumulates findings across phases
        self.doc: dict = {
            "question":    "",
            "understand":  "",
            "literature":  "",
            "mathematics": "",
            "computation": "",
            "engineering": "",
            "review":      "",
            "synthesis":   "",
            "report":      "",
        }

        # Agent registry - populated by register_agents()
        self._agents: dict = {}

        # Execution log
        self.log: list = []

        # Configure Gemini
        if config.GEMINI_API_KEY:
            genai.configure(api_key=config.GEMINI_API_KEY)

        os.makedirs(self.output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Agent setup
    # -------------------------------------------------------------

    def register_agents(
        self,
        scout     = None,
        mathematician = None,
        engineer  = None,
        numerical = None,
        reviewer  = None,
        synthesizer = None,
        writer    = None,
    ) -> "ResearchOrchestrator":
        """
        Register specialist agents for each phase.
        Any unregistered phase will use a fallback general agent.
        """
        if scout:        self._agents["literature"]   = scout
        if mathematician: self._agents["mathematics"] = mathematician
        if engineer:     self._agents["engineering"]  = engineer
        if numerical:    self._agents["computation"]  = numerical
        if reviewer:     self._agents["review"]       = reviewer
        if synthesizer:  self._agents["synthesis"]    = synthesizer
        if writer:       self._agents["report"]       = writer

        if self.verbose:
            print(f"{Fore.CYAN}[ResearchOrchestrator] Registered agents: "
                  f"{list(self._agents.keys())}{Style.RESET_ALL}")
        return self

    # -------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------

    def run(self, question: str) -> str:
        """
        Run the full research pipeline on a question.

        Returns the final report as a string and saves it to disk.
        """
        self.doc["question"] = question
        active_phases = self.DEPTH_PHASES.get(self.depth, self.DEPTH_PHASES["standard"])

        self._print_header(question, active_phases)

        for phase_num in active_phases:
            phase_name, phase_desc = self.PHASES[phase_num]
            self._print_phase(phase_num, phase_name, phase_desc)

            # Run the appropriate phase
            phase_method = getattr(self, f"_phase_{phase_name.lower()}", None)
            if phase_method:
                result = phase_method()
                self._log_phase(phase_name, result)
            else:
                if self.verbose:
                    print(f"{Fore.YELLOW}  [Phase {phase_num}] No method found "
                          f"for '{phase_name}' - skipping.{Style.RESET_ALL}")
            
            # Naturally space out phases to avoid triggering burst API rate limits
            time.sleep(2.0)

        # Save report to disk
        report  = self.doc["report"] or self.doc["synthesis"]
        outfile = self._save_report(question, report)

        if self.verbose:
            print(f"\n{Fore.GREEN}{'='*62}")
            print(f"  Research complete!")
            print(f"  Report saved to: {outfile}")
            print(f"{'='*62}{Style.RESET_ALL}")

        return report

    # -------------------------------------------------------------
    # Phase implementations
    # -------------------------------------------------------------

    def _phase_understand(self) -> str:
        """
        Phase 1: Use the LLM to decompose the research question
        into structured sub-problems before any research begins.
        """
        prompt = (
            f"You are a research planner. Decompose this research question "
            f"into 3-5 specific sub-problems that should be investigated:\n\n"
            f"Question: {self.doc['question']}\n\n"
            f"For each sub-problem state:\n"
            f"  - What needs to be found\n"
            f"  - What type of analysis is required (literature/math/numerical/engineering)\n"
            f"  - Why it matters for answering the main question\n\n"
            f"Also list: key variables, relevant physical domains, and expected output type."
        )
        result = self._llm_call(prompt)
        self.doc["understand"] = result
        return result

    def _phase_literature(self) -> str:
        """Phase 2: Literature scout searches arXiv + Wikipedia."""
        agent = self._get_agent("literature")
        agent.reset()

        task = (
            f"Research Question: {self.doc['question']}\n\n"
            f"Research Plan:\n{self.doc['understand']}\n\n"
            f"Search for papers and background on this topic. "
            f"Focus on: governing equations, key results, state of the art, "
            f"and the most relevant authors/papers."
        )
        result = self._run_agent(agent, task)
        self.doc["literature"] = result
        return result

    def _phase_mathematics(self) -> str:
        """Phase 3: Mathematician derives/verifies key equations."""
        agent = self._get_agent("mathematics")
        agent.reset()

        task = (
            f"Research Question: {self.doc['question']}\n\n"
            f"Literature Findings:\n{self.doc['literature']}\n\n"
            f"Based on the research plan and literature findings:\n"
            f"  1. Identify the key equations that govern this problem\n"
            f"  2. Derive or verify the most important ones step by step\n"
            f"  3. Show any simplifications or approximations used\n"
            f"  4. Format all results in LaTeX\n\n"
            f"Use the sympy_math tool to perform all symbolic operations."
        )
        result = self._run_agent(agent, task)
        self.doc["mathematics"] = result
        return result

    def _phase_computation(self) -> str:
        """Phase 4: Numerical analyst evaluates key expressions."""
        agent = self._get_agent("computation")
        agent.reset()

        context = "\n\n".join(filter(None, [
            f"Question: {self.doc['question']}",
            f"Literature:\n{self.doc['literature'][:800]}" if self.doc["literature"] else "",
            f"Mathematics:\n{self.doc['mathematics'][:800]}" if self.doc["mathematics"] else "",
        ]))

        task = (
            f"{context}\n\n"
            f"Perform all necessary numerical computations:\n"
            f"  1. Evaluate key expressions to specific numbers with units\n"
            f"  2. Compute any characteristic values (frequencies, voltages, etc.)\n"
            f"  3. Solve any numerical systems of equations\n"
            f"  4. Report all results in appropriate engineering notation\n\n"
            f"Use the numerical tool for all calculations."
        )
        result = self._run_agent(agent, task)
        self.doc["computation"] = result
        return result

    def _phase_engineering(self) -> str:
        """Phase 5: Engineer applies real-world constraints."""
        agent = self._get_agent("engineering")
        agent.reset()

        context = "\n\n".join(filter(None, [
            f"Question: {self.doc['question']}",
            f"Literature:\n{self.doc['literature'][:600]}" if self.doc["literature"] else "",
            f"Numerical Results:\n{self.doc['computation']}" if self.doc["computation"] else "",
        ]))

        task = (
            f"{context}\n\n"
            f"Apply engineering judgment:\n"
            f"  1. Sanity check all numerical results - are they physically plausible?\n"
            f"  2. Verify dimensional consistency of equations\n"
            f"  3. Identify dominant effects vs negligible terms\n"
            f"  4. Apply real-world constraints (material limits, process variations)\n"
            f"  5. Convert units as needed for practical engineering use\n\n"
            f"Flag anything that looks wrong or needs experimental verification."
        )
        result = self._run_agent(agent, task)
        self.doc["engineering"] = result
        return result

    def _phase_review(self) -> str:
        """Phase 6: Peer reviewer critiques the full body of work."""
        agent = self._get_agent("review")
        agent.reset()

        all_findings = "\n\n---\n\n".join(filter(None, [
            f"LITERATURE FINDINGS:\n{self.doc['literature']}",
            f"MATHEMATICAL DERIVATIONS:\n{self.doc['mathematics']}",
            f"NUMERICAL RESULTS:\n{self.doc['computation']}",
            f"ENGINEERING ASSESSMENT:\n{self.doc['engineering']}",
        ]))

        task = (
            f"Original research question: {self.doc['question']}\n\n"
            f"Review the following research findings as a journal peer reviewer:\n\n"
            f"{all_findings}\n\n"
            f"Provide a rigorous review covering: scientific validity, completeness, "
            f"dimensional consistency, missing considerations, and specific improvements."
        )
        result = self._run_agent(agent, task)
        self.doc["review"] = result
        return result

    def _phase_synthesize(self) -> str:
        """Phase 7: Synthesizer combines all findings coherently."""
        agent = self._get_agent("synthesis")
        agent.reset()

        all_content = "\n\n---\n\n".join(filter(None, [
            f"QUESTION:\n{self.doc['question']}",
            f"RESEARCH PLAN:\n{self.doc['understand']}",
            f"LITERATURE:\n{self.doc['literature']}",
            f"MATHEMATICS:\n{self.doc['mathematics']}",
            f"NUMERICS:\n{self.doc['computation']}",
            f"ENGINEERING:\n{self.doc['engineering']}",
            f"REVIEW FEEDBACK:\n{self.doc['review']}",
        ]))

        task = (
            f"Synthesize all the following research findings into a unified, "
            f"coherent knowledge summary that fully answers the research question.\n\n"
            f"{all_content}\n\n"
            f"Resolve any contradictions. Incorporate review feedback. "
            f"Make the synthesis self-contained and complete."
        )
        result = self._run_agent(agent, task)
        self.doc["synthesis"] = result
        return result

    def _phase_report(self) -> str:
        """Phase 8: Report writer produces the final document."""
        agent = self._get_agent("report")
        agent.reset()

        # Use the synthesis if available, otherwise compile all findings
        content_source = self.doc["synthesis"] or "\n\n".join(filter(None, [
            self.doc["literature"],
            self.doc["mathematics"],
            self.doc["computation"],
            self.doc["engineering"],
        ]))

        task = (
            f"Write a complete, publication-quality technical report that answers:\n\n"
            f"RESEARCH QUESTION: {self.doc['question']}\n\n"
            f"Based on these findings:\n{content_source}\n\n"
            f"Produce a full report with Abstract, Introduction, Theory/Background, "
            f"Analysis, Results, Discussion, and Conclusion. "
            f"Format all equations in LaTeX. Include units on every number. "
            f"Write in formal academic prose."
        )
        result = self._run_agent(agent, task)
        self.doc["report"] = result
        return result

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------

    def _get_agent(self, phase_key: str):
        """
        Get the agent for a phase. Falls back to a general agent
        if no specialist was registered for this phase.
        """
        if phase_key in self._agents:
            return self._agents[phase_key]

        # Fallback: create a minimal general agent
        from agents.base_agent import Agent
        fallback = Agent(
            name        = f"Fallback_{phase_key.capitalize()}",
            role        = f"You are a helpful expert assistant. Complete the task carefully.",
            temperature = 0.3,
            verbose     = False,
        )
        if self.verbose:
            print(f"{Fore.YELLOW}  [Warning] No agent for phase '{phase_key}'. "
                  f"Using fallback.{Style.RESET_ALL}")
        return fallback

    def _run_agent(self, agent, task: str) -> str:
        """Run an agent, using tools if it has any."""
        try:
            if agent.tools:
                return agent.chat_with_tools(task)
            return agent.chat(task)
        except Exception as e:
            return f"[Agent error in phase: {e}]"

    def _llm_call(self, prompt: str) -> str:
        """Direct LLM call for orchestrator-level thinking."""
        import google.generativeai as genai
        import time
        max_retries = 5
        base_delay = 2.0

        if not config.GEMINI_API_KEY:
            return "[Orchestrator LLM error: GEMINI_API_KEY is not set. Please add it to your .env file or config.py.]"
            
        genai.configure(api_key=config.GEMINI_API_KEY)
        
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(
                    model_name=config.DEFAULT_MODEL,
                    system_instruction="You are a research planning expert."
                )
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.4,
                        "max_output_tokens": config.DEFAULT_MAX_TOKENS,
                    }
                )
                return response.text
            except Exception as e:
                is_rate_limit = False
                err_str = str(e).lower()
                if "429" in err_str or "rate limit" in err_str or "exhausted" in err_str:
                    is_rate_limit = True
                    
                if is_rate_limit and attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    if self.verbose:
                        print(f"{Fore.YELLOW}[ResearchOrchestrator] Rate limit reached in direct call. Retrying in {sleep_time:.2f}s... (Attempt {attempt+1}/{max_retries}){Style.RESET_ALL}")
                    time.sleep(sleep_time)
                else:
                    return f"[Orchestrator LLM error: {e}]"
        return "[Orchestrator LLM error: Max retries exceeded due to rate limiting.]"

    def _save_report(self, question: str, report: str) -> str:
        """Save the final report to a markdown file."""
        timestamp  = time.strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() or c in "-_" else "_"
                             for c in question[:40])
        filename   = f"{self.output_dir}/report_{safe_title}_{timestamp}.md"

        full_content = (
            f"# Research Report\n\n"
            f"**Question:** {question}\n\n"
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"**Depth:** {self.depth}\n\n"
            f"---\n\n"
            f"{report}\n\n"
            f"---\n\n"
            f"## Research Phases\n\n"
        )

        if self.doc["literature"]:
            full_content += f"### Literature Findings\n{self.doc['literature']}\n\n"
        if self.doc["mathematics"]:
            full_content += f"### Mathematical Derivations\n{self.doc['mathematics']}\n\n"
        if self.doc["computation"]:
            full_content += f"### Numerical Results\n{self.doc['computation']}\n\n"
        if self.doc["engineering"]:
            full_content += f"### Engineering Assessment\n{self.doc['engineering']}\n\n"
        if self.doc["review"]:
            full_content += f"### Peer Review\n{self.doc['review']}\n\n"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(full_content)

        return filename

    def _log_phase(self, phase: str, result: str) -> None:
        self.log.append({
            "phase":     phase,
            "timestamp": time.strftime("%H:%M:%S"),
            "length":    len(result),
        })

    def print_log(self) -> None:
        """Print a summary of all phases that ran."""
        print(f"\n{'-'*62}")
        print(f"Research Orchestrator Log - {len(self.log)} phases")
        print(f"{'-'*62}")
        for entry in self.log:
            print(f"  [OK] {entry['timestamp']} | {entry['phase']:15s} "
                  f"| {entry['length']} chars output")
        print(f"{'-'*62}")

    def _print_header(self, question: str, phases: list) -> None:
        phase_names = [self.PHASES[p][0] for p in phases]
        print(f"\n{'='*62}")
        print(f"{Fore.CYAN}  Research Orchestrator - {self.depth.upper()} mode")
        print(f"  Phases: {' -> '.join(phase_names)}{Style.RESET_ALL}")
        print(f"{'-'*62}")
        print(f"  Question: {question[:80]}")
        print(f"{'='*62}")

    def _print_phase(self, num: int, name: str, desc: str) -> None:
        print(f"\n{Fore.YELLOW}  [{num}/{len(self.DEPTH_PHASES[self.depth])}] "
              f"{name} - {desc}{Style.RESET_ALL}")