"""
agents/base_agent.py
---------------------------------------------------------------------
The Base Agent - unified with Working Memory & Long-Term Memory
---------------------------------------------------------------------
"""

from typing import Optional, List, Dict, Any
from colorama import Fore, Style, init
import config
from memory.working_memory import WorkingMemory
from memory.long_term import LongTermMemory
from memory.context_builder import ContextBuilder

init(autoreset=True)


class Agent:
    """
    Base Agent integrated with Working Memory, Long-Term Memory, ContextBuilder,
    and provider error validation gates.
    """

    def __init__(
        self,
        name: str = "Agent",
        role: str = "You are a helpful assistant.",
        provider: str = "gemini",
        fallback_provider: str = "gemini",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        verbose: Optional[bool] = None,
        memory_strategy: str = "window",
        memory_limit: int = 8000,
        long_term: Optional[LongTermMemory] = None,
        graph_memory=None,
        working_memory: Optional[WorkingMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
    ):
        self.name = name
        self.role = role
        self.provider = provider
        self.fallback_provider = fallback_provider
        self.model = model or config.DEFAULT_MODEL
        self.max_tokens = max_tokens if max_tokens is not None else config.DEFAULT_MAX_TOKENS
        self.temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        self.verbose = verbose if verbose is not None else config.VERBOSE

        # Tool registry
        self.tools: dict = {}

        # 1. Initialize Working Memory (combines short-term context & Knowledge Graph)
        if working_memory:
            self.working_memory = working_memory
        else:
            self.working_memory = WorkingMemory(
                max_tokens=memory_limit,
                verbose=self.verbose
            )
            if graph_memory:
                self.working_memory.graph_memory = graph_memory

        # Backward compatibility references
        self._short_term = self.working_memory.short_term
        self.history = self.working_memory.get_messages()
        self.graph_memory = self.working_memory.graph_memory

        # 2. Initialize Long-Term Memory (persistent facts & ChromaDB vector store)
        self.long_term_memory = long_term_memory or long_term
        self.long_term = self.long_term_memory

        # 3. Context Builder
        self.context_builder = ContextBuilder()

        if self.verbose:
            print(f"{Fore.CYAN}[{self.name}] Initialized | model={self.model} | working_memory=active | long_term_memory={'active' if self.long_term_memory else 'none'}{Style.RESET_ALL}")

    def chat(self, user_message: str, phase: str = "literature") -> str:
        """
        Send a message and get a response.
        Thin Execution Path:
          1. Store user query in Working Memory
          2. Retrieve query-aware memory context via ContextBuilder
          3. Call LLM provider with fallback
          4. Validate response (NEVER store API/provider errors in memory)
          5. Ingest valid output into Working Memory & Hybrid KG
        """
        if self.verbose:
            print(f"\n{Fore.YELLOW}[{self.name}] <- USER:{Style.RESET_ALL} {user_message}")

        # Step 1: Add user message to working memory context
        self.working_memory.add_message("user", user_message)

        # Step 2: Build bounded prompt messages via ContextBuilder
        messages = self._build_messages(user_query=user_message)

        # Step 3: Call LLM provider dispatcher
        response_text = self._call_llm(messages)

        # Step 4: Validate response - prevent provider/API errors from polluting memory
        if self._validate_response(response_text):
            # Record assistant response in working memory
            self.working_memory.add_message("assistant", response_text)

            # Step 5: Delegate hybrid KG extraction to working memory
            self.working_memory.ingest(response_text, phase=phase, source_doc=self.name)
        else:
            if self.verbose:
                print(f"{Fore.RED}[{self.name}] Provider error or invalid payload detected. Memory ingestion blocked.{Style.RESET_ALL}")

        if self.verbose:
            print(f"{Fore.GREEN}[{self.name}] -> RESPONSE:{Style.RESET_ALL} {response_text}")

        return response_text

    def _validate_response(self, text: Optional[str]) -> bool:
        """Defensive validation: Returns True if text is valid model output, False if provider error."""
        if not text or not isinstance(text, str):
            return False

        stripped = text.strip()
        if not stripped:
            return False

        # Obvious provider error markers
        error_indicators = [
            "[LLM", "[Groq API Error", "[Gemini API Error", "[Groq Error", "[Gemini Error",
            "429 Too Many Requests", "rate limit exceeded", "context length exceeded",
            "maximum context length", "authentication failed", "API error", "timeout"
        ]

        if any(indicator in stripped for indicator in error_indicators):
            return False

        return True

    def _build_messages(self, user_query: str = "") -> List[Dict[str, str]]:
        """Build query-aware message list sent to LLM using ContextBuilder."""
        tools_prompt = self._build_tool_prompt()
        return self.context_builder.build_prompt_messages(
            role_prompt=self.role,
            tools_prompt=tools_prompt,
            user_query=user_query,
            working_memory=self.working_memory,
            long_term_memory=self.long_term_memory
        )

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Make API call to LLM via Groq / Gemini dispatcher with fallback."""
        budget_obj = getattr(self, "budget", None)
        return config.call_with_fallback(
            messages=messages,
            primary_provider=self.provider,
            primary_model=self.model,
            fallback_provider=self.fallback_provider,
            fallback_model=config.GEMINI_RESEARCH_MODEL,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            budget=budget_obj,
            category=getattr(self, "name", "agent").lower()
        )

    def _summarize_text(self, text: str) -> str:
        """Internal summarization helper."""
        messages = [
            {"role": "system", "content": "You are a concise summarizer."},
            {"role": "user", "content": text},
        ]
        return self._call_llm(messages)

    # -------------------------------------------------------------
    # Memory Helpers
    # -------------------------------------------------------------

    def remember(self, key: str, value: str) -> None:
        """Store a fact in long-term memory."""
        if self.long_term_memory:
            self.long_term_memory.remember_fact(key, value)

    def recall(self, key: str) -> Optional[str]:
        """Retrieve a fact from long-term memory by key."""
        if self.long_term_memory:
            return self.long_term_memory.recall_fact(key)
        return None

    def save_note(self, content: str, tags: Optional[List[str]] = None) -> None:
        """Save a document chunk/note to long-term memory."""
        if self.long_term_memory:
            self.long_term_memory.store_document(content, metadata={"tags": tags or [], "agent": self.name})

    def search_memory(self, query: str, top_k: int = 3) -> List[dict]:
        """Search long-term memory semantically."""
        if self.long_term_memory:
            return self.long_term_memory.semantic_search(query, top_k=top_k)
        return []

    # -------------------------------------------------------------
    # Tool Management
    # -------------------------------------------------------------

    def register_tool(self, tool) -> None:
        """Register a tool so this agent can use it."""
        self.tools[tool.name] = tool
        if self.verbose:
            print(f"{Fore.CYAN}[{self.name}] Tool registered: '{tool.name}'{Style.RESET_ALL}")

    def use_tool(self, tool_name: str, tool_input: str) -> str:
        """Directly invoke a registered tool by name."""
        if tool_name not in self.tools:
            available = list(self.tools.keys())
            return f"Tool '{tool_name}' not found. Available: {available}"

        tool = self.tools[tool_name]
        if self.verbose:
            print(f"{Fore.MAGENTA}[{self.name}] Using tool '{tool_name}' | input: {tool_input[:80]}{Style.RESET_ALL}")

        result = tool.run(tool_input)

        if hasattr(result, "output") and hasattr(result, "success"):
            if not result.success:
                return f"[Tool Error]: {getattr(result, 'error', 'Tool execution failed')}"
            result = result.output

        if self.verbose:
            print(f"{Fore.MAGENTA}[{self.name}] Tool result: {str(result)[:120]}{Style.RESET_ALL}")
        return str(result)

    def list_tools(self) -> List[dict]:
        """Return all registered tools as a list of dicts."""
        return [t.to_dict() for t in self.tools.values()]

    def _build_tool_prompt(self) -> str:
        """Generate AVAILABLE TOOLS section for system prompt."""
        if not self.tools:
            return ""
        lines = ["\n\nAVAILABLE TOOLS:"]
        for tool in self.tools.values():
            lines.append(f"  - {tool.name}: {tool.description}")
        return "\n".join(lines)

    def chat_with_tools(self, user_message: str, max_steps: int = 6) -> str:
        """Run the full ReAct loop."""
        from core.react_loop import ReActLoop
        loop = ReActLoop(agent=self, max_steps=max_steps, verbose=self.verbose)
        return loop.run(user_message)

    def reset(self) -> None:
        """Clear working memory conversation history."""
        self.working_memory.clear()

    def inject_context(self, context: str, source: Optional[str] = None) -> None:
        """
        Inject external observation context into working memory.
        Formatted under user role to distinguish from this agent's own responses.
        """
        source_label = f" from {source}" if source else ""
        formatted = f"[EXTERNAL CONTEXT{source_label}]:\n{context}"
        self.working_memory.add_message("user", formatted)

        if self.verbose:
            print(f"{Fore.MAGENTA}[{self.name}] Context injected: {context[:80]}...{Style.RESET_ALL}")

    def get_history(self) -> List[Dict[str, str]]:
        """Return current working memory history."""
        return self.working_memory.get_messages()

    def memory_stats(self) -> dict:
        """Return summary of working and long-term memory state."""
        return {
            "agent": self.name,
            "working_memory": {
                "short_term_messages": len(self.working_memory.get_messages()),
                "graph_nodes": len(self.working_memory.graph_memory.graph.nodes) if self.working_memory.graph_memory else 0,
                "graph_edges": len(self.working_memory.graph_memory.graph.edges) if self.working_memory.graph_memory else 0,
            },
            "long_term_memory": self.long_term_memory.stats() if self.long_term_memory else None
        }

    def __repr__(self):
        return (f"Agent(name={self.name!r}, model={self.model!r}, "
                f"tools={list(self.tools.keys())})")