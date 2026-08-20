"""
core/budget_manager.py
---------------------------------------------------------------------
Centralized LLM Budget Manager for MAS.

Enforces bounded LLM calls, token limits, per-phase reservations,
hard safety ceilings, and telemetry accounting across all research agents.
---------------------------------------------------------------------
"""

import os
import threading
from typing import Dict, Any, Optional

import config


class LLMBudgetManager:
    """
    Centralized, thread-safe budget manager for multi-agent research pipelines.
    Enforces per-run limits on LLM calls, token budgets, and per-phase reservations.
    """

    PROFILE_DEFAULTS = {
        "quick":    {"max_llm_calls": getattr(config, "QUICK_MAX_LLM_CALLS", 15)},
        "standard": {"max_llm_calls": getattr(config, "STANDARD_MAX_LLM_CALLS", 30)},
        "deep":     {"max_llm_calls": getattr(config, "DEEP_MAX_LLM_CALLS", 40)},
    }

    MAX_ABSOLUTE_LLM_CALLS = getattr(config, "MAX_ABSOLUTE_LLM_CALLS", 45)

    def __init__(
        self,
        depth: str = "standard",
        max_llm_calls: Optional[int] = None,
        max_input_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        max_kg_extractions: int = 4,
        max_react_steps: int = 3,
    ):
        self.lock = threading.Lock()
        self.depth = depth.lower()

        default_calls = self.PROFILE_DEFAULTS.get(self.depth, {}).get("max_llm_calls", 30)
        configured_max = max_llm_calls or default_calls
        self.max_llm_calls = min(configured_max, self.MAX_ABSOLUTE_LLM_CALLS)

        self.max_input_tokens = max_input_tokens or getattr(config, "MAX_TOTAL_INPUT_TOKENS", 150000)
        self.max_output_tokens = max_output_tokens or getattr(config, "MAX_TOTAL_OUTPUT_TOKENS", 60000)
        self.max_total_tokens = self.max_input_tokens + self.max_output_tokens

        self.max_kg_extractions = max_kg_extractions
        self.max_react_steps = max_react_steps

        # Reservations
        self.generation_reservation = 8 if self.depth != "quick" else 4
        self.emergency_reservation = getattr(config, "EMERGENCY_RESERVE_CALLS", 2)

        # Counter metrics
        self.calls_used = 0
        self.input_tokens_used = 0
        self.output_tokens_used = 0
        self.retry_calls = 0
        self.fallback_calls = 0
        self.kg_extraction_calls = 0
        self.react_calls = 0
        self.generation_calls = 0
        self.judge_calls = 0

    def check_preflight(self, category: str = "general", estimated_tokens: int = 1500) -> Dict[str, Any]:
        """
        Check if sufficient budget remains before making an LLM call.
        """
        with self.lock:
            if self.calls_used >= self.max_llm_calls:
                return {
                    "allowed": False,
                    "reason": "BUDGET_EXHAUSTED",
                    "detail": f"Call limit reached ({self.calls_used}/{self.max_llm_calls})"
                }

            if self.calls_used >= self.MAX_ABSOLUTE_LLM_CALLS:
                return {
                    "allowed": False,
                    "reason": "SAFETY_CEILING_REACHED",
                    "detail": f"Safety ceiling reached ({self.calls_used}/{self.MAX_ABSOLUTE_LLM_CALLS})"
                }

            # Check Token Limits
            if self.input_tokens_used + self.output_tokens_used >= self.max_total_tokens:
                return {
                    "allowed": False,
                    "reason": "TOKEN_BUDGET_EXHAUSTED",
                    "detail": f"Token limit reached ({self.input_tokens_used + self.output_tokens_used}/{self.max_total_tokens})"
                }

            # Enforce 90% Warning Threshold: Protect generation & disable optional work
            remaining_calls = self.max_llm_calls - self.calls_used
            if category == "kg_extraction":
                if self.kg_extraction_calls >= self.max_kg_extractions:
                    return {
                        "allowed": False,
                        "reason": "KG_CAP_REACHED",
                        "detail": f"KG LLM extraction cap reached ({self.kg_extraction_calls}/{self.max_kg_extractions})"
                    }
                if remaining_calls <= self.generation_reservation + self.emergency_reservation:
                    return {
                        "allowed": False,
                        "reason": "RESERVED_FOR_GENERATION",
                        "detail": f"KG extraction skipped to protect final generation budget ({remaining_calls} calls remaining)"
                    }

            if category == "react" and remaining_calls <= self.generation_reservation:
                return {
                    "allowed": False,
                    "reason": "RESERVED_FOR_GENERATION",
                    "detail": f"ReAct loop truncated to protect final generation budget ({remaining_calls} calls remaining)"
                }

            return {"allowed": True, "remaining_calls": remaining_calls}

    def record_call(
        self,
        category: str = "general",
        input_tokens: int = 0,
        output_tokens: int = 0,
        is_retry: bool = False,
        is_fallback: bool = False
    ) -> None:
        """Record usage metrics for an executed LLM call."""
        with self.lock:
            self.calls_used += 1
            self.input_tokens_used += input_tokens
            self.output_tokens_used += output_tokens

            if is_retry:
                self.retry_calls += 1
            if is_fallback:
                self.fallback_calls += 1

            if category == "kg_extraction":
                self.kg_extraction_calls += 1
            elif category == "react":
                self.react_calls += 1
            elif category in ["report", "generation"]:
                self.generation_calls += 1
            elif category == "judge":
                self.judge_calls += 1

            pct = (self.calls_used / self.max_llm_calls) * 100.0
            if pct >= 90.0:
                config.safe_print(f"[LLMBudgetManager] CRITICAL WARNING: {pct:.1f}% budget used ({self.calls_used}/{self.max_llm_calls} calls). Protecting generation reserve.")
            elif pct >= 75.0:
                config.safe_print(f"[LLMBudgetManager] WARNING: {pct:.1f}% budget used ({self.calls_used}/{self.max_llm_calls} calls).")

    def get_summary(self) -> Dict[str, Any]:
        """Return a complete telemetry snapshot of current budget state."""
        with self.lock:
            return {
                "depth": self.depth,
                "max_llm_calls": self.max_llm_calls,
                "calls_used": self.calls_used,
                "remaining_calls": max(0, self.max_llm_calls - self.calls_used),
                "input_tokens_used": self.input_tokens_used,
                "output_tokens_used": self.output_tokens_used,
                "total_tokens_used": self.input_tokens_used + self.output_tokens_used,
                "retry_calls": self.retry_calls,
                "fallback_calls": self.fallback_calls,
                "kg_extraction_calls": self.kg_extraction_calls,
                "react_calls": self.react_calls,
                "generation_calls": self.generation_calls,
                "judge_calls": self.judge_calls
            }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize budget state for checkpointing."""
        return self.get_summary()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMBudgetManager":
        """Restore budget state from checkpoint."""
        bm = cls(
            depth=data.get("depth", "standard"),
            max_llm_calls=data.get("max_llm_calls", 30)
        )
        bm.calls_used = data.get("calls_used", 0)
        bm.input_tokens_used = data.get("input_tokens_used", 0)
        bm.output_tokens_used = data.get("output_tokens_used", 0)
        bm.retry_calls = data.get("retry_calls", 0)
        bm.fallback_calls = data.get("fallback_calls", 0)
        bm.kg_extraction_calls = data.get("kg_extraction_calls", 0)
        bm.react_calls = data.get("react_calls", 0)
        bm.generation_calls = data.get("generation_calls", 0)
        bm.judge_calls = data.get("judge_calls", 0)
        return bm
