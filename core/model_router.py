"""
core/model_router.py
---------------------------------------------------------------------
Smart Model Router for MAS.

Dynamically allocates Gemini models (Flash vs Pro) based on phase,
pipeline execution depth ('quick', 'standard', 'deep'), and task complexity.
Optimizes token budget, latency, and academic derivation precision.
---------------------------------------------------------------------
"""

import os
from typing import Dict, Any, Optional
from colorama import Fore, Style, init

import config

init(autoreset=True)

FLASH_MODEL = os.getenv("FLASH_MODEL", "llama-3.3-70b-versatile")
PRO_MODEL = os.getenv("PRO_MODEL", "llama-3.3-70b-versatile")

# Model pricing ($ / 1M tokens) for telemetry estimation
MODEL_PRICING = {
    "llama-3.1-8b-instant":  {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "gemini-2.5-flash":       {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro":         {"input": 1.25,  "output": 5.00},
}


class SmartModelRouter:
    """
    Dynamic model selection engine for multi-agent research pipelines.
    """

    # Default phase complexity mapping
    PHASE_COMPLEXITY = {
        "understand":  "low",
        "literature":  "medium",
        "mathematics": "high",
        "computation": "medium",
        "engineering": "medium",
        "review":      "high",
        "synthesis":   "high",
        "report":      "high",
        "judge":       "high"
    }

    def __init__(
        self,
        depth: str = "standard",
        flash_model: str = FLASH_MODEL,
        pro_model: str = PRO_MODEL,
        verbose: bool = True
    ):
        self.depth = depth.lower()
        self.flash_model = flash_model
        self.pro_model = pro_model
        self.verbose = verbose

    def get_model_for_phase(self, phase_name: str, override_complexity: Optional[str] = None) -> str:
        """
        Determine the optimal model for a given pipeline phase.
        """
        clean_phase = phase_name.lower().strip()
        complexity = override_complexity or self.PHASE_COMPLEXITY.get(clean_phase, "medium")

        if os.getenv("USE_FLASH_ONLY", "").lower() in ["true", "1"]:
            return self.flash_model

        if self.depth == "quick":
            # Quick mode: speed & low cost focus (uses Flash for all phases)
            selected = self.flash_model
        elif self.depth == "standard":
            # Standard mode: Pro for mathematics/report, Flash for others
            if clean_phase in ["mathematics", "report", "judge"] or complexity == "high":
                selected = self.pro_model
            else:
                selected = self.flash_model
        elif self.depth == "deep":
            # Deep mode: Pro for math, review, synthesis, report & high complexity
            if complexity in ["medium", "high"] or clean_phase in ["mathematics", "review", "synthesis", "report", "judge"]:
                selected = self.pro_model
            else:
                selected = self.flash_model
        else:
            selected = config.DEFAULT_MODEL or self.flash_model

        if self.verbose:
            print(f"{Fore.CYAN}[ModelRouter] Phase '{phase_name}' (depth={self.depth}, complexity={complexity}) -> {selected}{Style.RESET_ALL}")

        return selected

    def get_model_for_agent(self, agent_name: str) -> str:
        """
        Determine the optimal model based on agent specialization.
        """
        clean_name = agent_name.lower()
        if any(term in clean_name for term in ["mathematician", "reviewer", "judge", "writer"]):
            return self.get_model_for_phase(clean_name, override_complexity="high")
        return self.get_model_for_phase(clean_name, override_complexity="medium")

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        """
        Estimate API cost in USD based on model pricing.
        """
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["gemini-2.5-flash"])
        input_cost = (prompt_tokens / 1_000_000.0) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000.0) * pricing["output"]
        return round(input_cost + output_cost, 6)

    def print_routing_table(self) -> None:
        """Print the active model allocation table for all phases."""
        print(f"\n{Fore.CYAN}{'='*56}")
        print(f"  SMART MODEL ROUTER TABLE — Depth: {self.depth.upper()}")
        print(f"{'='*56}{Style.RESET_ALL}")
        print(f"  {'PHASE':<16} | {'COMPLEXITY':<12} | {'ALLOCATED MODEL':<20}")
        print(f"{'-'*56}")
        for phase, comp in self.PHASE_COMPLEXITY.items():
            model = self.get_model_for_phase(phase)
            print(f"  {phase:<16} | {comp:<12} | {model:<20}")
        print(f"{'='*56}\n")
