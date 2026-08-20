"""
core/model_router.py
---------------------------------------------------------------------
Smart Provider-Aware Model Router for MAS.

Separates Provider and Model concepts into structured route dictionaries:
  {
      "provider": "groq" | "gemini",
      "model": "qwen/qwen3.6-27b",
      "fallback_provider": "gemini",
      "fallback_model": "gemini-3.6-flash"
  }
---------------------------------------------------------------------
"""

import os
from typing import Dict, Any, Optional
from colorama import Fore, Style, init

import config

init(autoreset=True)

# Model pricing ($ / 1M tokens) for telemetry estimation
MODEL_PRICING = {
    "llama-3.1-8b-instant":   {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "qwen/qwen3.6-27b":        {"input": 0.20, "output": 0.50},
    "gemini-2.5-flash":        {"input": 0.075, "output": 0.30},
    "gemini-3.6-flash":        {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro":          {"input": 1.25,  "output": 5.00},
}


class SmartModelRouter:
    """
    Dynamic provider-aware model router for multi-agent research pipelines.
    """

    PHASE_COMPLEXITY = {
        "understand":  "low",
        "literature":  "medium",
        "mathematics": "high",
        "computation": "medium",
        "engineering": "medium",
        "review":      "medium",
        "synthesis":   "medium",
        "report":      "medium",
        "judge":       "medium"
    }

    def __init__(
        self,
        depth: str = "standard",
        groq_model: str = None,
        gemini_research_model: str = None,
        gemini_final_model: str = None,
        verbose: bool = True
    ):
        self.depth = depth.lower()
        self.groq_model = groq_model or config.GROQ_MODEL
        self.gemini_research_model = gemini_research_model or config.GEMINI_RESEARCH_MODEL
        self.gemini_final_model = gemini_final_model or config.GEMINI_FINAL_MODEL
        self.verbose = verbose

    def get_route_for_phase(self, phase_name: str, override_complexity: Optional[str] = None) -> Dict[str, str]:
        """
        Determine provider and model route dictionary for a given phase.
        Returns:
          {"provider": ..., "model": ..., "fallback_provider": ..., "fallback_model": ...}
        """
        clean_phase = phase_name.lower().strip()
        complexity = override_complexity or self.PHASE_COMPLEXITY.get(clean_phase, "medium")

        # Stage 2: Final Report Phase is ALWAYS Gemini-only
        if clean_phase in ["report", "final"]:
            route = {
                "provider": "gemini",
                "model": self.gemini_final_model,
                "fallback_provider": "gemini",
                "fallback_model": self.gemini_final_model
            }
        else:
            # Stage 1: Research Phases default to Groq primary with Gemini fallback
            route = {
                "provider": "groq",
                "model": self.groq_model,
                "fallback_provider": "gemini",
                "fallback_model": self.gemini_research_model
            }

        if self.verbose:
            print(f"{Fore.CYAN}[ModelRouter] Phase '{phase_name}' (depth={self.depth}, complexity={complexity}) "
                  f"-> provider={route['provider']} ({route['model']}) | fallback={route['fallback_provider']}{Style.RESET_ALL}")

        return route

    def get_model_for_phase(self, phase_name: str, override_complexity: Optional[str] = None) -> str:
        """Backward compatible helper returning model string."""
        return self.get_route_for_phase(phase_name, override_complexity)["model"]

    def get_model_for_agent(self, agent_name: str) -> str:
        """Backward compatible helper for agents."""
        return self.get_route_for_phase(agent_name)["model"]

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """Estimate API cost in USD based on model pricing."""
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["gemini-3.6-flash"])
        input_cost = (prompt_tokens / 1_000_000.0) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000.0) * pricing["output"]
        return round(input_cost + output_cost, 6)

    def print_routing_table(self) -> None:
        """Print active provider routing table for all phases."""
        print(f"\n{Fore.CYAN}{'='*64}")
        print(f"  SMART MODEL ROUTER TABLE — Depth: {self.depth.upper()}")
        print(f"{'='*64}{Style.RESET_ALL}")
        print(f"  {'PHASE':<14} | {'PROVIDER':<10} | {'PRIMARY MODEL':<20} | {'FALLBACK':<10}")
        print(f"{'-'*64}")
        for phase, comp in self.PHASE_COMPLEXITY.items():
            route = self.get_route_for_phase(phase)
            print(f"  {phase:<14} | {route['provider']:<10} | {route['model']:<20} | {route['fallback_provider']:<10}")
        print(f"{'='*64}\n")
