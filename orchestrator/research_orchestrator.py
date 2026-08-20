"""
orchestrator/research_orchestrator.py
---------------------------------------------------------------------
The Research Orchestrator - Workflow Controller for MAS

Coordinates multi-agent research pipelines across 8 structured phases:
  Phase 1 - UNDERSTAND    : decompose the question
  Phase 2 - LITERATURE    : search papers and background
  Phase 3 - MATHEMATICS   : derive/verify equations
  Phase 4 - COMPUTATION   : evaluate numerically
  Phase 5 - ENGINEERING   : apply real-world constraints
  Phase 6 - REVIEW        : peer-review the findings
  Phase 7 - SYNTHESIZE    : combine everything coherently
  Phase 8 - REPORT        : produce the final document via LongFormGenerator

Features:
  * Durable checkpointing (reports/checkpoints/<run_id>.json)
  * Clean PAUSED_RATE_LIMIT handling when providers are exhausted
  * Resume from interrupted checkpoint without re-running completed phases
  * Memory integration via WorkingMemory and LongTermMemory
  * Optional Judge evaluation gate
---------------------------------------------------------------------
"""

import os
import time
import json
import re
from typing import Dict, Any, List, Optional
from colorama import Fore, Style, init

import config
from memory.working_memory import WorkingMemory
from memory.long_term import LongTermMemory
from core.report_generator import LongFormGenerator
from agents.judge_agent import JudgeAgent

init(autoreset=True)


class ResearchOrchestrator:
    """
    Workflow Controller driving specialist agents through structured research pipelines,
    managing memory ingestion, durable checkpointing, and rate-limit pause/resume states.
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
        depth: str = "standard",
        mode: str = "research_paper",
        output_dir: str = "reports",
        verbose: bool = True,
        working_memory: Optional[WorkingMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        enable_judge: bool = False,
    ):
        self.depth = depth
        self.mode = mode
        self.output_dir = output_dir
        self.verbose = verbose
        self.enable_judge = enable_judge

        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize Working Memory & Long-Term Memory
        self.working_memory = working_memory or WorkingMemory(verbose=self.verbose)
        self.long_term_memory = long_term_memory or LongTermMemory(verbose=self.verbose)
        self.graph_memory = self.working_memory.graph_memory

        # Initialize Centralized LLM Budget Manager
        from core.budget_manager import LLMBudgetManager
        self.budget = LLMBudgetManager(depth=self.depth)
        self.working_memory.budget = self.budget

        # Research document metadata dictionary (for backward compatibility)
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

        # Smart Model Router
        from core.model_router import SmartModelRouter
        self.router = SmartModelRouter(depth=self.depth, verbose=self.verbose)

        # Agent registry & execution status
        self._agents: dict = {}
        self.log: list = []
        self.status = "CREATED"
        self.run_id: Optional[str] = None
        self.completed_phases: List[str] = []

    # -------------------------------------------------------------
    # Agent Registration
    # -------------------------------------------------------------

    def register_agents(
        self,
        scout=None,
        mathematician=None,
        engineer=None,
        numerical=None,
        reviewer=None,
        synthesizer=None,
        writer=None,
    ) -> "ResearchOrchestrator":
        """Register specialist agents for each phase and route models."""
        provided = {
            "literature":  scout,
            "mathematics": mathematician,
            "engineering": engineer,
            "computation": numerical,
            "review":      reviewer,
            "synthesis":   synthesizer,
            "report":      writer,
        }

        for phase_key, agent in provided.items():
            if agent is not None:
                agent.working_memory = self.working_memory
                agent.long_term_memory = self.long_term_memory
                agent.budget = self.budget
                agent.model = self.router.get_model_for_phase(phase_key)
                self._agents[phase_key] = agent

        if self.verbose:
            config.safe_print(
                f"{Fore.CYAN}[ResearchOrchestrator] Registered specialist agents for phases: "
                f"{list(self._agents.keys())}{Style.RESET_ALL}"
            )
        return self

    # -------------------------------------------------------------
    # Execution & Resumption
    # -------------------------------------------------------------

    def run(self, question: str, resume_checkpoint_id: Optional[str] = None) -> str:
        """
        Run or resume the research pipeline for a question.
        Returns final generated document or report string.
        """
        self.doc["question"] = question
        active_phase_indices = self.DEPTH_PHASES.get(self.depth, self.DEPTH_PHASES["standard"])

        # Check for checkpoint resume
        if resume_checkpoint_id or not self.run_id:
            chk = self._load_checkpoint(resume_checkpoint_id or self._generate_run_id(question))
            if chk:
                self.run_id = chk.get("run_id")
                self.completed_phases = chk.get("completed_phases", [])
                self.status = chk.get("status", "RUNNING")
                if self.verbose:
                    config.safe_print(f"{Fore.YELLOW}[Orchestrator] Resuming run '{self.run_id}' | Completed: {self.completed_phases}{Style.RESET_ALL}")
            else:
                self.run_id = self._generate_run_id(question)

        self.status = "RUNNING"
        self._print_header(question, active_phase_indices)

        for phase_num in active_phase_indices:
            phase_name, phase_desc = self.PHASES[phase_num]

            # Skip if phase was already completed in checkpoint
            if phase_name in self.completed_phases:
                if self.verbose:
                    config.safe_print(f"{Fore.CYAN}  [Phase {phase_num}] {phase_name} already completed. Skipping.{Style.RESET_ALL}")
                continue

            self._print_phase(phase_num, phase_name, phase_desc)
            phase_method = getattr(self, f"_phase_{phase_name.lower()}", None)

            if phase_method:
                result, success = phase_method()

                if not success:
                    # Rate limit or unrecoverable provider failure -> PAUSED_RATE_LIMIT
                    self.status = "PAUSED_RATE_LIMIT"
                    self._save_checkpoint(failed_phase=phase_name, error_reason="Provider rate limit or failure")
                    if self.verbose:
                        config.safe_print(f"{Fore.RED}\n[Orchestrator] Pipeline PAUSED_RATE_LIMIT at phase '{phase_name}'. Checkpoint saved.{Style.RESET_ALL}")
                    return f"[Pipeline Paused]: Rate limit or provider failure encountered at phase '{phase_name}'. Run ID: {self.run_id}"

                self.doc[phase_name.lower()] = result
                self.completed_phases.append(phase_name)
                self._log_phase(phase_name, result, "SUCCESS")
                self._save_checkpoint()

        # Phase 8 / Report Assembly
        report = self.doc.get("report") or self.doc.get("synthesis") or ""

        # Optional Judge Evaluation Gate
        if self.enable_judge and report:
            if self.verbose:
                config.safe_print(f"\n{Fore.CYAN}[Orchestrator] Invoking Judge Agent evaluation...{Style.RESET_ALL}")
            judge = JudgeAgent(verbose=self.verbose)
            evidence_context = self.working_memory.get_context(question)
            eval_result = judge.evaluate(question=question, report_text=report, evidence_context=evidence_context)

            if eval_result.get("verdict") == "REVISE":
                if self.verbose:
                    config.safe_print(f"{Fore.YELLOW}[Orchestrator] Judge requested REVISE. Applying targeted report revision...{Style.RESET_ALL}")
                report = self._phase_report_revision(report, eval_result.get("weaknesses", []))

        # Save final report to disk
        outfile = self._save_report(question, report)
        self.status = "COMPLETED"
        self._save_checkpoint()

        if self.verbose:
            config.safe_print(f"\n{Fore.GREEN}{'='*62}")
            config.safe_print(f"  Research Pipeline Complete! Status: {self.status}")
            config.safe_print(f"  Report saved to: {outfile}")
            config.safe_print(f"{'='*62}{Style.RESET_ALL}")

        return report

    # -------------------------------------------------------------
    # Phase Executions
    # -------------------------------------------------------------

    def _phase_understand(self) -> (str, bool):
        """Phase 1: Planning & Sub-problem decomposition."""
        prompt = (
            f"Decompose this research question into 3-5 specific sub-problems:\n\n"
            f"Question: {self.doc['question']}\n\n"
            f"Specify required analysis types (literature/mathematics/numerical/engineering) "
            f"and key physical parameters."
        )
        messages = [
            {"role": "system", "content": "You are an expert research planner."},
            {"role": "user", "content": prompt}
        ]
        result = config.call_with_fallback(messages=messages, max_tokens=1000, budget=self.budget, category="understand")
        if self._is_error(result):
            return result, False

        self.working_memory.ingest(result, phase="UNDERSTAND", source_doc="Planner")
        return result, True

    def _phase_literature(self) -> (str, bool):
        """Phase 2: Literature Scout searches arXiv & Wikipedia."""
        agent = self._get_agent("literature")
        agent.reset()
        task = f"Research Question: {self.doc['question']}\nSearch literature for governing equations, methods, and papers."
        result = agent.chat_with_tools(task) if agent.tools else agent.chat(task)
        if self._is_error(result):
            return result, False
        return result, True

    def _phase_mathematics(self) -> (str, bool):
        """Phase 3: Mathematician derives/verifies key equations."""
        agent = self._get_agent("mathematics")
        agent.reset()
        context = self.working_memory.get_context("governing equations and math assumptions")
        task = f"Research Question: {self.doc['question']}\nContext:\n{context}\nDerive governing equations and output in LaTeX."
        result = agent.chat_with_tools(task) if agent.tools else agent.chat(task)
        if self._is_error(result):
            return result, False
        return result, True

    def _phase_computation(self) -> (str, bool):
        """Phase 4: Numerical Analyst performs quantitative calculations."""
        agent = self._get_agent("computation")
        agent.reset()
        context = self.working_memory.get_context("numerical parameters and equations")
        task = f"Question: {self.doc['question']}\nContext:\n{context}\nPerform numerical evaluations with engineering units."
        result = agent.chat_with_tools(task) if agent.tools else agent.chat(task)
        if self._is_error(result):
            return result, False
        return result, True

    def _phase_engineering(self) -> (str, bool):
        """Phase 5: Engineer applies real-world constraints."""
        agent = self._get_agent("engineering")
        agent.reset()
        context = self.working_memory.get_context("numerical values and physical bounds")
        task = f"Question: {self.doc['question']}\nContext:\n{context}\nSanity check numbers, physical constraints, and unit consistency."
        result = agent.chat_with_tools(task) if agent.tools else agent.chat(task)
        if self._is_error(result):
            return result, False
        return result, True

    def _phase_review(self) -> (str, bool):
        """Phase 6: Peer Reviewer critiques research findings."""
        agent = self._get_agent("review")
        agent.reset()
        context = self.working_memory.get_context("all research findings equations and results")
        task = f"Question: {self.doc['question']}\nContext:\n{context}\nCritique findings for scientific validity and missing evidence."
        result = agent.chat(task)
        if self._is_error(result):
            return result, False
        return result, True

    def _phase_synthesize(self) -> (str, bool):
        """Phase 7: Synthesizer combines findings from shared memory."""
        agent = self._get_agent("synthesis")
        agent.reset()
        context = self.working_memory.get_context(self.doc["question"])
        task = f"Question: {self.doc['question']}\nContext:\n{context}\nSynthesize all findings into a unified technical summary."
        result = agent.chat(task)
        if self._is_error(result):
            return result, False
        return result, True

    def _phase_report(self) -> (str, bool):
        """Phase 8: LongFormGenerator produces section-by-section document."""
        generator = LongFormGenerator(output_dir=self.output_dir, verbose=self.verbose)
        result = generator.generate(
            question=self.doc["question"],
            mode=self.mode,
            research_doc=self.doc,
            working_memory=self.working_memory,
            long_term_memory=self.long_term_memory,
            budget=self.budget
        )
        if self._is_error(result):
            return result, False
        return result, True

    def _phase_report_revision(self, current_report: str, weaknesses: List[str]) -> str:
        """Apply targeted Judge revision feedback to the final report."""
        feedback_str = "\n- ".join(weaknesses)
        prompt = (
            f"Original Report:\n{current_report[:4000]}\n\n"
            f"Judge Revision Feedback:\n- {feedback_str}\n\n"
            f"Revise the report addressing these specific weaknesses while maintaining academic structure and LaTeX equations."
        )
        messages = [
            {"role": "system", "content": "You are an expert scientific report editor."},
            {"role": "user", "content": prompt}
        ]
        revised = config.call_with_fallback(messages=messages, max_tokens=3000)
        return revised if not self._is_error(revised) else current_report

    # -------------------------------------------------------------
    # Checkpointing & State Persistence
    # -------------------------------------------------------------

    def _generate_run_id(self, question: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9]+", "_", question.lower()).strip("_")
        return f"{clean[:25]}_{int(time.time())}"

    def _save_checkpoint(self, failed_phase: Optional[str] = None, error_reason: Optional[str] = None):
        """Save durable pipeline checkpoint to disk."""
        chk_path = os.path.join(self.checkpoint_dir, f"{self.run_id}.json")
        payload = {
            "run_id": self.run_id,
            "question": self.doc["question"],
            "depth": self.depth,
            "mode": self.mode,
            "status": self.status,
            "completed_phases": self.completed_phases,
            "failed_phase": failed_phase,
            "error_reason": error_reason,
            "budget_state": self.budget.to_dict() if hasattr(self, "budget") and self.budget else None,
            "timestamp": time.time()
        }
        with open(chk_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _load_checkpoint(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load durable checkpoint from disk."""
        chk_path = os.path.join(self.checkpoint_dir, f"{run_id}.json")
        if os.path.exists(chk_path):
            try:
                with open(chk_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def _is_error(self, text: str) -> bool:
        """Check if output string represents a provider or tool error."""
        if not text or not isinstance(text, str):
            return True
        return any(err in text for err in ["[LLM", "[Groq API Error", "[Gemini API Error", "[Agent error", "[Generation Error]", "429 Too Many Requests"])

    # -------------------------------------------------------------
    # Helpers & Logging
    # -------------------------------------------------------------

    def _get_agent(self, phase_key: str):
        if phase_key in self._agents:
            return self._agents[phase_key]
        from agents.base_agent import Agent
        return Agent(
            name=f"Fallback_{phase_key.capitalize()}",
            role="You are a helpful expert research assistant.",
            working_memory=self.working_memory,
            long_term_memory=self.long_term_memory,
            verbose=False,
        )

    def _save_report(self, question: str, report: str) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in question[:40])
        filename = f"{self.output_dir}/report_{safe_title}_{timestamp}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        return filename

    def _log_phase(self, phase: str, result: str, status: str = "SUCCESS") -> None:
        self.log.append({
            "phase": phase,
            "status": status,
            "timestamp": time.strftime("%H:%M:%S"),
            "length": len(result) if result else 0,
        })

    def _print_header(self, question: str, phases: list) -> None:
        phase_names = [self.PHASES[p][0] for p in phases]
        config.safe_print(f"\n{'='*62}")
        config.safe_print(f"{Fore.CYAN}  Research Orchestrator - {self.depth.upper()} mode")
        config.safe_print(f"  Phases: {' -> '.join(phase_names)}{Style.RESET_ALL}")
        config.safe_print(f"{'-'*62}")
        config.safe_print(f"  Question: {question[:80]}")
        config.safe_print(f"{'='*62}")

    def _print_phase(self, num: int, name: str, desc: str) -> None:
        config.safe_print(f"\n{Fore.YELLOW}  [{num}/{len(self.DEPTH_PHASES[self.depth])}] {name} - {desc}{Style.RESET_ALL}")