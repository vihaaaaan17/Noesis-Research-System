"""
memory/context_builder.py
---------------------------------------------------------------------
Unified Context Builder for MAS.

Retrieves relevant context from Working Memory (Short-Term + Knowledge Graph)
and Long-Term Memory (Persistent Facts + Semantic Chunks) within token budgets.
---------------------------------------------------------------------
"""

from typing import List, Dict, Any, Optional
from memory.working_memory import WorkingMemory
from memory.long_term import LongTermMemory


class ContextBuilder:
    """
    Unified Context Builder layer.
    Assembles system prompt, tools, Working Memory, and Long-Term Memory
    into a bounded message list for LLM invocation.
    """

    def __init__(self, max_context_tokens: int = 4000):
        self.max_context_tokens = max_context_tokens

    def build_prompt_messages(
        self,
        role_prompt: str,
        tools_prompt: str,
        user_query: str,
        working_memory: Optional[WorkingMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Build clean bounded prompt messages for LLM generation.
        """
        system_content = role_prompt + tools_prompt

        # 1. Retrieve query-aware Long-Term Memory context
        if long_term_memory:
            lt_context = long_term_memory.get_context(query=user_query)
            if lt_context:
                system_content += f"\n\n{lt_context}"

        # 2. Retrieve query-aware Working Memory Knowledge Graph context
        if working_memory:
            wm_context = working_memory.get_context(query=user_query)
            if wm_context:
                system_content += f"\n\n{wm_context}"

        system_message = {"role": "system", "content": system_content}

        # 3. Retrieve Short-Term history
        msg_history = []
        if working_memory:
            msg_history = working_memory.get_messages()
        elif history:
            msg_history = list(history)

        return [system_message] + msg_history
