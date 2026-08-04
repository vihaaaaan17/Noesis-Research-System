"""
evals/telemetry.py
---------------------------------------------------------------------
Internal Telemetry & LangSmith Tracing for MAS.

Tracks token consumption, latency, Knowledge Graph node/edge growth,
and agent execution turns across all 8 research pipeline phases.
Supports native LangSmith tracing integration when LANGSMITH_API_KEY is set.
---------------------------------------------------------------------
"""

import os
import time
import json
import datetime
from typing import Dict, Any, List, Optional
from colorama import Fore, Style, init

init(autoreset=True)

# Optional LangSmith integration
LANGSMITH_ACTIVE = False
try:
    if os.getenv("LANGCHAIN_TRACING_V2") == "true" or os.getenv("LANGSMITH_API_KEY"):
        from langsmith import Client, traceable
        LANGSMITH_ACTIVE = True
    else:
        def traceable(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
except Exception:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class TelemetryLogger:
    """
    Local and LangSmith telemetry tracker for MAS pipeline runs.
    """

    def __init__(self, run_name: str = "Research_Run", verbose: bool = True):
        self.run_name = run_name
        self.verbose = verbose
        self.start_time = time.time()

        self.phase_logs: Dict[str, Dict[str, Any]] = {}
        self.current_phase: Optional[str] = None
        self.phase_start_time: Optional[float] = None
        self.agent_steps: List[Dict[str, Any]] = []

        self.ls_run = None
        self.ls_phase_spans: Dict[str, Any] = {}

        if LANGSMITH_ACTIVE:
            try:
                from langsmith.run_trees import RunTree
                self.ls_run = RunTree(
                    name=self.run_name,
                    run_type="chain",
                    project_name=os.getenv("LANGCHAIN_PROJECT", "MAS-Research-Agent")
                )
                self.ls_run.post()
                if self.verbose:
                    print(f"{Fore.GREEN}[Telemetry] LangSmith Tracing Active | Project: {os.getenv('LANGCHAIN_PROJECT', 'MAS-Research-Agent')}{Style.RESET_ALL}")
            except Exception as e:
                if self.verbose:
                    print(f"{Fore.YELLOW}[Telemetry] LangSmith initialization skipped: {e}{Style.RESET_ALL}")

    def start_phase(self, phase_name: str) -> None:
        """Mark the beginning of a pipeline phase."""
        self.current_phase = phase_name
        self.phase_start_time = time.time()
        self.phase_logs[phase_name] = {
            "start_time": datetime.datetime.now().isoformat(),
            "duration_sec": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "kg_nodes": 0,
            "kg_edges": 0,
            "status": "RUNNING"
        }

        if self.ls_run:
            try:
                span = self.ls_run.create_child(
                    name=f"Phase_{phase_name}",
                    run_type="chain",
                    inputs={"phase": phase_name}
                )
                span.post()
                self.ls_phase_spans[phase_name] = span
            except Exception:
                pass

        if self.verbose:
            print(f"{Fore.CYAN}[Telemetry] Started phase: {phase_name}{Style.RESET_ALL}")

    def end_phase(
        self,
        phase_name: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        kg_nodes: int = 0,
        kg_edges: int = 0,
        status: str = "SUCCESS"
    ) -> None:
        """Mark the completion of a pipeline phase and record metrics."""
        duration = time.time() - (self.phase_start_time or time.time())
        total_tokens = prompt_tokens + completion_tokens

        if phase_name not in self.phase_logs:
            self.phase_logs[phase_name] = {}

        self.phase_logs[phase_name].update({
            "end_time": datetime.datetime.now().isoformat(),
            "duration_sec": round(duration, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "kg_nodes": kg_nodes,
            "kg_edges": kg_edges,
            "status": status
        })

        if phase_name in self.ls_phase_spans:
            try:
                span = self.ls_phase_spans[phase_name]
                span.end(
                    outputs={
                        "status": status,
                        "duration_sec": round(duration, 3),
                        "kg_nodes": kg_nodes,
                        "kg_edges": kg_edges,
                        "total_tokens": total_tokens
                    }
                )
                span.patch()
            except Exception:
                pass

        self.current_phase = None
        self.phase_start_time = None

        if self.verbose:
            print(f"{Fore.GREEN}[Telemetry] Completed phase: {phase_name} ({duration:.2f}s | {total_tokens} tokens | KG nodes: {kg_nodes}){Style.RESET_ALL}")

    def log_agent_step(
        self,
        agent_name: str,
        step_type: str,
        input_len: int,
        output_len: int,
        elapsed_sec: float,
        tool_used: Optional[str] = None
    ) -> None:
        """Record an individual agent reasoning/tool execution step."""
        step = {
            "timestamp": datetime.datetime.now().isoformat(),
            "phase": self.current_phase or "ORCHESTRATOR",
            "agent": agent_name,
            "step_type": step_type,
            "input_chars": input_len,
            "output_chars": output_len,
            "elapsed_sec": round(elapsed_sec, 3),
            "tool_used": tool_used
        }
        self.agent_steps.append(step)

    def get_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics across the entire run."""
        total_duration = round(time.time() - self.start_time, 3)
        total_prompt_tokens = sum(p.get("prompt_tokens", 0) for p in self.phase_logs.values())
        total_completion_tokens = sum(p.get("completion_tokens", 0) for p in self.phase_logs.values())
        total_tokens = total_prompt_tokens + total_completion_tokens
        max_kg_nodes = max((p.get("kg_nodes", 0) for p in self.phase_logs.values()), default=0)
        max_kg_edges = max((p.get("kg_edges", 0) for p in self.phase_logs.values()), default=0)

        return {
            "run_name": self.run_name,
            "total_duration_sec": total_duration,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "total_agent_steps": len(self.agent_steps),
            "max_kg_nodes": max_kg_nodes,
            "max_kg_edges": max_kg_edges,
            "phases_run": len(self.phase_logs),
            "phase_logs": self.phase_logs
        }

    def print_summary(self) -> None:
        """Print a clean telemetry report table to console."""
        metrics = self.get_metrics()
        print(f"\n{Fore.CYAN}{'='*64}")
        print(f"  TELEMETRY SUMMARY — {self.run_name}")
        print(f"{'='*64}{Style.RESET_ALL}")
        print(f"  Total Duration   : {metrics['total_duration_sec']} sec")
        print(f"  Total Tokens     : {metrics['total_tokens']} (Prompt: {metrics['total_prompt_tokens']} | Completion: {metrics['total_completion_tokens']})")
        print(f"  Total Agent Steps: {metrics['total_agent_steps']}")
        print(f"  Final KG Size    : {metrics['max_kg_nodes']} nodes, {metrics['max_kg_edges']} edges")
        print(f"{'-'*64}")
        print(f"  {'PHASE':<16} | {'TIME (s)':<10} | {'TOKENS':<10} | {'KG NODES':<10}")
        print(f"{'-'*64}")
        for phase, log in self.phase_logs.items():
            print(f"  {phase:<16} | {log.get('duration_sec', 0.0):<10.2f} | {log.get('total_tokens', 0):<10d} | {log.get('kg_nodes', 0):<10d}")
        print(f"{'='*64}\n")

    def export_json(self, filepath: str) -> None:
        """Export telemetry data to a JSON file and close LangSmith run tree."""
        metrics = self.get_metrics()
        if self.ls_run:
            try:
                self.ls_run.end(outputs=metrics)
                self.ls_run.patch()
            except Exception:
                pass

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
