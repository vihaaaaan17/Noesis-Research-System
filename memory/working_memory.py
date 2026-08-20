"""
memory/working_memory.py
---------------------------------------------------------------------
Working Memory System for MAS.

Combines ShortTermMemory (sliding-window context) and KnowledgeGraphMemory
(NetworkX graph state) into a single, unified interface for research agents.
---------------------------------------------------------------------
"""

import os
from typing import List, Dict, Any, Optional
from memory.short_term import ShortTermMemory
from memory.graph_memory import KnowledgeGraphMemory


class WorkingMemory:
    """
    Working Memory System.
    Provides a unified query-aware interface for recent conversation context,
    entities, equations, and Knowledge Graph relations.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        graph_storage_path: str = "long_term_graph.json",
        verbose: bool = False
    ):
        self.short_term = ShortTermMemory(max_tokens=max_tokens, strategy="window", verbose=verbose)
        self.graph_memory = KnowledgeGraphMemory(storage_path=graph_storage_path, verbose=verbose)
        self.verbose = verbose

    def add_message(self, role: str, content: str) -> None:
        """Add a conversation message to short-term working context."""
        self.short_term.add(role, content)

    def get_messages(self) -> List[Dict[str, str]]:
        """Return history from short-term context."""
        return self.short_term.get_messages()

    def ingest(self, text: str, phase: str = "literature", source_doc: str = "agent_turn") -> Any:
        """Run hybrid Knowledge Graph extraction and ingest into graph memory."""
        budget_obj = getattr(self, "budget", None)
        return self.graph_memory.extract_from_text(text, source_doc=source_doc, phase=phase, budget=budget_obj)

    def get_context(self, query: str = "") -> str:
        """
        Unified Working Memory retrieval:
        Retrieves relevant entities, relationships, equations, and recent context.
        """
        kg_context = self.graph_memory.get_context_for_prompt(query) if query else ""
        return kg_context

    def consolidate_graph(self) -> Dict[str, Any]:
        """Perform final graph consolidation pass."""
        return self.graph_memory.consolidate()

    def save(self) -> None:
        """Persist graph memory state to disk."""
        self.graph_memory.save_to_json()

    def clear(self) -> None:
        """Clear short-term context."""
        self.short_term.clear()
