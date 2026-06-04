"""
memory/short_term.py
---------------------------------------------------------------------
Short-Term Memory

Manages the agent's conversation history inside the context window.

The problem it solves:
  LLMs have a token limit. If a conversation goes on too long, you
  either crash (too many tokens) or lose early context (just slice it).
  Neither is good.

This module handles it properly in two ways:

  1. SLIDING WINDOW  - keep only the last N messages (simple, fast)
  2. SUMMARIZATION   - compress old messages using the LLM itself,
                       keeping a dense summary instead of raw history
                       (smarter, preserves meaning)

The agent automatically uses this on every chat() call.
---------------------------------------------------------------------
"""

from colorama import Fore, Style, init

init(autoreset=True)

# Rough estimate: 1 token approx 4 characters in English
CHARS_PER_TOKEN = 4

class ShortTermMemory:
    """
    Manages the in-context conversation history for one agent.

    Parameters
    ----------
    max_tokens    : int   - max total tokens to keep in history
                            (not counting the system prompt)
    strategy      : str   - "window" or "summarize"
                            window    = keep last N messages only
                            summarize = compress old messages via LLM
    summarize_fn  : callable or None
                    A function that takes a string and returns a
                    summary string. Pass the agent's own LLM call
                    so it can summarize itself.
    verbose       : bool  - print memory events to console
    """

    STRATEGIES = ("window", "summarize")

    def __init__(
        self,
        max_tokens:   int      = 2000,
        strategy:     str      = "window",
        summarize_fn: callable = None,
        verbose:      bool     = True,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")

        self.max_tokens   = max_tokens
        self.strategy     = strategy
        self.summarize_fn = summarize_fn
        self.verbose      = verbose

        # The actual message list passed to the LLM
        self.messages: list[dict] = []

        # If summarization is used, the summary of old messages lives here
        # It gets prepended to the messages list as a "system" note
        self.summary: str = ""

    # -------------------------------------------------------------
    # Core interface
    # -------------------------------------------------------------

    def add(self, role: str, content: str) -> None:
        """
        Add a message to short-term memory.
        Automatically trims/summarizes if over the token limit.

        Parameters
        ----------
        role    : "user" or "assistant"
        content : the message text
        """
        self.messages.append({"role": role, "content": content})
        self._enforce_limit()

    def get_messages(self) -> list[dict]:
        """
        Return the current messages ready to be sent to the LLM.
        If a summary exists, it's prepended as a context note.
        """
        if not self.summary:
            return list(self.messages)

        # Prepend the summary of older messages as context
        summary_msg = {
            "role": "user",
            "content": (
                f"[Earlier conversation summary:\n{self.summary}\n"
                f"The conversation continues below.]"
            )
        }
        return [summary_msg] + list(self.messages)

    def clear(self) -> None:
        """Wipe all messages and summary."""
        self.messages = []
        self.summary  = ""

    def token_count(self) -> int:
        """Rough token estimate for all current messages."""
        total_chars = sum(len(m["content"]) for m in self.messages)
        return total_chars // CHARS_PER_TOKEN

    def message_count(self) -> int:
        return len(self.messages)

    # -------------------------------------------------------------
    # Limit enforcement
    # -------------------------------------------------------------

    def _enforce_limit(self) -> None:
        """
        Check if we're over the token budget.
        If yes, apply the chosen strategy.
        """
        if self.token_count() <= self.max_tokens:
            return   # we're within budget, nothing to do

        if self.strategy == "window":
            self._apply_window()
        elif self.strategy == "summarize":
            self._apply_summarize()

    def _apply_window(self) -> None:
        """
        Sliding window strategy.
        Drop oldest messages until we're back within the limit.
        Always keep at least the last 2 messages (user + assistant pair).
        """
        while self.token_count() > self.max_tokens and len(self.messages) > 2:
            dropped = self.messages.pop(0)
            if self.verbose:
                preview = dropped["content"][:50]
                print(f"{Fore.YELLOW}[ShortTermMemory] Dropped old message "
                      f"({dropped['role']}): '{preview}...'{Style.RESET_ALL}")

    def _apply_summarize(self) -> None:
        """
        Summarization strategy.
        Takes the oldest half of messages, compresses them into a summary,
        replaces them with the summary (keeping recent messages intact).
        """
        if not self.summarize_fn:
            # Fallback to window if no summarize function provided
            if self.verbose:
                print(f"{Fore.YELLOW}[ShortTermMemory] No summarize_fn provided. "
                      f"Falling back to window strategy.{Style.RESET_ALL}")
            self._apply_window()
            return

        # Split: compress the older half, keep the recent half
        split_point   = len(self.messages) // 2
        old_messages  = self.messages[:split_point]
        self.messages = self.messages[split_point:]

        # Build a text block of old messages for the LLM to summarize
        old_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in old_messages
        )

        # Include any existing summary in what we're compressing
        if self.summary:
            old_text = f"Previous summary:\n{self.summary}\n\nNew messages:\n{old_text}"

        # Call the summarize function (the agent's LLM)
        prompt = (
            f"Summarize this conversation history in 3-5 bullet points, "
            f"keeping all important facts, decisions, and results:\n\n{old_text}"
        )

        if self.verbose:
            print(f"{Fore.YELLOW}[ShortTermMemory] Context too long "
                  f"({self.token_count()} tokens). Summarizing "
                  f"{len(old_messages)} old messages...{Style.RESET_ALL}")

        new_summary  = self.summarize_fn(prompt)
        self.summary = new_summary

        if self.verbose:
            print(f"{Fore.YELLOW}[ShortTermMemory] Summary created: "
                  f"{new_summary[:100]}...{Style.RESET_ALL}")

    # -------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "messages":    self.message_count(),
            "tokens_used": self.token_count(),
            "token_limit": self.max_tokens,
            "strategy":    self.strategy,
            "has_summary": bool(self.summary),
        }

    def __repr__(self):
        return (f"ShortTermMemory(messages={self.message_count()}, "
                f"tokens approx {self.token_count()}, strategy={self.strategy!r})")
