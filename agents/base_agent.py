"""
agents/base_agents.py
---------------------------------------------------------------------
The Base Agent - now with full memory integration (Step 4)

What's new vs Step 2:
  * Short-term memory  - auto-manages context window size
  * Long-term memory   - persist and recall facts across sessions
  * summarize_fn       - agent can compress its own old context
  * memory_stats()     - inspect what the agent remembers
---------------------------------------------------------------------
"""

import google.generativeai as genai
from colorama import Fore, Style, init
import config

init(autoreset=True)


class Agent:
    """
    Base Agent with tools + memory.

    Parameters
    ----------
    name            : str   - agent label
    role            : str   - system prompt / persona
    model           : str   - LLM model name
    max_tokens      : int   - max tokens per LLM response
    temperature     : float - 0=focused, 1=creative
    verbose         : bool  - print thinking to console
    memory_strategy : str   - "none" | "window" | "summarize"
                              "none"      -> raw list, no management (Step 1 behavior)
                              "window"    -> drop oldest messages when full
                              "summarize" -> compress old messages via LLM
    memory_limit    : int   - token budget for short-term memory
    long_term               - attach a persistent store (LongTermMemory)
    """

    def __init__(
        self,
        name:            str   = "Agent",
        role:            str   = "You are a helpful assistant.",
        model:           str   = None,
        max_tokens:      int   = None,
        temperature:     float = None,
        verbose:         bool  = None,
        memory_strategy: str   = "window",
        memory_limit:    int   = 2000,
        long_term               = None,
        graph_memory            = None,
    ):
        self.name        = name
        self.role        = role
        self.model       = model       or config.DEFAULT_MODEL
        self.max_tokens  = max_tokens  or config.DEFAULT_MAX_TOKENS
        self.temperature = temperature or config.DEFAULT_TEMPERATURE
        self.verbose     = verbose if verbose is not None else config.VERBOSE

        # Tool registry
        self.tools: dict = {}

        # Short-term memory
        self._setup_short_term(memory_strategy, memory_limit)

        # Long-term memory
        self.long_term = long_term

        # Knowledge Graph memory
        self.graph_memory = graph_memory

        # Configure Gemini
        if config.GEMINI_API_KEY:
            genai.configure(api_key=config.GEMINI_API_KEY)

        if self.verbose:
            print(f"{Fore.CYAN}[{self.name}] Initialized "
                  f"| model={self.model} "
                  f"| memory={memory_strategy} "
                  f"| graph_memory={'active' if self.graph_memory else 'none'}{Style.RESET_ALL}")

    # -------------------------------------------------------------
    # Short-term memory setup
    # -------------------------------------------------------------

    def _setup_short_term(self, strategy: str, limit: int):
        """Configure short-term memory based on strategy choice."""
        if strategy == "none":
            # Plain list - original Step 1 behavior
            self.history      = []
            self._short_term  = None
            self._mem_strategy = "none"
        else:
            from memory.short_term import ShortTermMemory
            # Pass self._summarize_text as the summarizer so the agent
            # can use its own LLM to compress its context
            self._short_term = ShortTermMemory(
                max_tokens   = limit,
                strategy     = strategy,
                summarize_fn = self._summarize_text if strategy == "summarize" else None,
                verbose      = self.verbose,
            )
            # Keep self.history pointing to the same list for compatibility
            self.history       = self._short_term.messages
            self._mem_strategy = strategy

    # -------------------------------------------------------------
    # Core chat method
    # -------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.
        Now routes through ShortTermMemory and KnowledgeGraphMemory if active.
        """
        if self.verbose:
            print(f"\n{Fore.YELLOW}[{self.name}] <- USER:{Style.RESET_ALL} "
                  f"{user_message}")

        # Add user message
        if self._short_term:
            self._short_term.add("user", user_message)
        else:
            self.history.append({"role": "user", "content": user_message})

        # Build messages for LLM
        messages      = self._build_messages(user_query=user_message)
        response_text = self._call_llm(messages)

        # Store assistant response
        if self._short_term:
            self._short_term.add("assistant", response_text)
        else:
            self.history.append({"role": "assistant", "content": response_text})

        # Auto-extract entities and relations to KnowledgeGraphMemory
        if self.graph_memory:
            self.graph_memory.extract_from_text(response_text, source_doc=self.name)

        if self.verbose:
            print(f"{Fore.GREEN}[{self.name}] -> RESPONSE:{Style.RESET_ALL} "
                  f"{response_text}")

        return response_text

    # -------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------

    def _build_messages(self, user_query: str = "") -> list[dict]:
        """
        Build the full message list sent to the LLM:
          [system prompt + tool list + long-term context + kg context] + [history]
        """
        system_content = self.role + self._build_tool_prompt()

        # Prepend any relevant long-term memory to system prompt
        if self.long_term:
            lt_context = self.long_term.build_context_string()
            if lt_context:
                system_content += f"\n\n{lt_context}"

        # Prepend any relevant Knowledge Graph shared memory context
        if self.graph_memory:
            query_text = user_query
            if not query_text and self.history:
                query_text = str(self.history[-1].get("content", ""))
            kg_context = self.graph_memory.get_context_for_prompt(query_text)
            if kg_context:
                system_content += f"\n\n{kg_context}"

        system_message = {"role": "system", "content": system_content}

        # Get history from short-term memory (includes summary if any)
        if self._short_term:
            history = self._short_term.get_messages()
        else:
            history = self.history

        return [system_message] + history

    def _call_llm(self, messages: list[dict]) -> str:
        """Make the API call to LLM via Groq / Gemini dispatcher."""
        return config.call_llm_api(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

    def _summarize_text(self, text: str) -> str:
        """
        Internal: use this agent's own LLM to summarize text.
        Passed into ShortTermMemory as the summarize_fn.
        Bypasses history - makes a raw LLM call without side effects.
        """
        messages = [
            {"role": "system",  "content": "You are a concise summarizer."},
            {"role": "user",    "content": text},
        ]
        return self._call_llm(messages)

    # -------------------------------------------------------------
    # Long-term memory helpers
    # -------------------------------------------------------------

    def remember(self, key: str, value: str) -> None:
        """Store a fact in long-term memory (requires long_term attached)."""
        if not self.long_term:
            print(f"{Fore.RED}[{self.name}] No long-term memory attached. "
                  f"Pass long_term=LongTermMemory() to constructor.{Style.RESET_ALL}")
            return
        self.long_term.remember(key, value)

    def recall(self, key: str) -> str | None:
        """Retrieve a fact from long-term memory by exact key."""
        if not self.long_term:
            return None
        return self.long_term.recall(key)

    def save_note(self, content: str, tags: list[str] = None) -> None:
        """Save a note to long-term memory with optional tags."""
        if not self.long_term:
            print(f"{Fore.RED}[{self.name}] No long-term memory attached.{Style.RESET_ALL}")
            return
        self.long_term.add_note(content, tags)

    def search_memory(self, query: str, top_k: int = 3) -> list[dict]:
        """Search long-term memory notes by keyword."""
        if not self.long_term:
            return []
        return self.long_term.search(query, top_k=top_k)

    def load_memory_context(self, query: str = None) -> None:
        """
        Pull relevant long-term memory into the current conversation
        as injected context, so the agent 'remembers' past sessions.
        """
        if not self.long_term:
            return
        context = self.long_term.build_context_string(query)
        if context:
            self.inject_context(context)

    # -------------------------------------------------------------
    # Tool management
    # -------------------------------------------------------------

    def register_tool(self, tool) -> None:
        """Register a tool so this agent can use it."""
        self.tools[tool.name] = tool
        if self.verbose:
            print(f"{Fore.CYAN}[{self.name}] Tool registered: "
                  f"'{tool.name}'{Style.RESET_ALL}")

    def use_tool(self, tool_name: str, tool_input: str) -> str:
        """Directly invoke a registered tool by name."""
        if tool_name not in self.tools:
            available = list(self.tools.keys())
            return f"Tool '{tool_name}' not found. Available: {available}"

        tool   = self.tools[tool_name]
        if self.verbose:
            print(f"{Fore.MAGENTA}[{self.name}] Using tool '{tool_name}' "
                  f"| input: {tool_input[:80]}{Style.RESET_ALL}")

        result = tool.run(tool_input)
        if self.verbose:
            print(f"{Fore.MAGENTA}[{self.name}] Tool result: "
                  f"{str(result)[:120]}{Style.RESET_ALL}")
        return result

    def list_tools(self) -> list[dict]:
        """Return all registered tools as a list of dicts."""
        return [t.to_dict() for t in self.tools.values()]

    def _build_tool_prompt(self) -> str:
        """Generate the AVAILABLE TOOLS section for the system prompt."""
        if not self.tools:
            return ""
        lines = ["\n\nAVAILABLE TOOLS:"]
        for tool in self.tools.values():
            lines.append(f"  - {tool.name}: {tool.description}")
        return "\n".join(lines)

    # -------------------------------------------------------------
    # ReAct-powered chat (tool-aware)
    # -------------------------------------------------------------

    def chat_with_tools(self, user_message: str, max_steps: int = 6) -> str:
        """Run the full ReAct loop - agent reasons and uses tools automatically."""
        from core.react_loop import ReActLoop
        loop = ReActLoop(agent=self, max_steps=max_steps, verbose=self.verbose)
        return loop.run(user_message)

    # -------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------

    def reset(self) -> None:
        """Clear short-term memory / conversation history."""
        if self._short_term:
            self._short_term.clear()
        else:
            self.history = []
        if self.verbose:
            print(f"{Fore.CYAN}[{self.name}] Memory cleared.{Style.RESET_ALL}")

    def inject_context(self, context: str) -> None:
        """
        Manually push content into the agent's history as an assistant message.
        Use this to pass another agent's output into this agent's context.
        """
        if self._short_term:
            self._short_term.add("assistant", context)
        else:
            self.history.append({"role": "assistant", "content": context})

        if self.verbose:
            print(f"{Fore.MAGENTA}[{self.name}] Context injected: "
                  f"{context[:80]}...{Style.RESET_ALL}")

    def get_history(self) -> list[dict]:
        """Return the current conversation history."""
        if self._short_term:
            return self._short_term.get_messages()
        return self.history

    def memory_stats(self) -> dict:
        """Return a summary of this agent's memory state."""
        stats = {
            "agent":           self.name,
            "memory_strategy": self._mem_strategy,
        }
        if self._short_term:
            stats["short_term"] = self._short_term.stats()
        if self.long_term:
            stats["long_term"] = self.long_term.stats()
        return stats

    def __repr__(self):
        return (f"Agent(name={self.name!r}, model={self.model!r}, "
                f"memory={self._mem_strategy!r}, "
                f"tools={list(self.tools.keys())})")