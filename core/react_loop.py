"""
core/react_loop.py
---------------------------------------------------------------------
The ReAct Loop - Reason, Act, Observe

This is the brain of tool-using agents. It runs a loop where:
  1. REASON  - LLM reads the question + tool list, thinks about what to do
  2. ACT     - LLM decides to call a tool (or give a final answer)
  3. OBSERVE - we run the tool, feed the result back to the LLM
  4. REPEAT  - until the LLM gives a final answer (no more tool calls)

Protocol (what the LLM must output):
-------------------------------------
To call a tool:
    TOOL_CALL: <tool_name> | <input>

    Example:
    TOOL_CALL: calculator | (1024 * 1024) / 8

To give a final answer:
    FINAL_ANSWER: <the answer>

    Example:
    FINAL_ANSWER: The result is 131072 bytes.

Why this protocol?
  * Simple to parse - just split on ":" and "|"
  * LLMs follow it reliably when shown clear examples in the system prompt
  * No JSON needed - avoids formatting errors from smaller models

---------------------------------------------------------------------
"""

from colorama import Fore, Style, init

init(autoreset=True)


# -- Protocol tokens -----------------------------------------------
TOOL_CALL_TOKEN    = "TOOL_CALL:"
FINAL_ANSWER_TOKEN = "FINAL_ANSWER:"

# -- System prompt addon injected into every tool-using agent ------
REACT_SYSTEM_ADDON = """

INSTRUCTIONS FOR TOOL USE:
You have access to tools listed above. Use them when needed.

To call a tool, respond EXACTLY in this format (nothing before it):
    TOOL_CALL: <tool_name> | <input to the tool>

To give your final answer (when you don't need any more tools):
    FINAL_ANSWER: <your complete answer here>

Rules:
  - Only call ONE tool per response.
  - After seeing a tool result, decide: do you need another tool, or can you answer?
  - Always end with FINAL_ANSWER when you are done.
  - Never make up tool results - always actually call the tool.

Examples:
  User: What is 2 to the power of 16?
  You:  TOOL_CALL: calculator | 2 ** 16

  [Tool returns: 65536]

  You:  FINAL_ANSWER: 2 to the power of 16 is 65536.

  ---

  User: What day is it and what is 100 * 365?
  You:  TOOL_CALL: get_datetime | now

  [Tool returns: Monday, January 01, 2025...]

  You:  TOOL_CALL: calculator | 100 * 365

  [Tool returns: 36500]

  You:  FINAL_ANSWER: Today is Monday, January 01, 2025. 100 * 365 = 36500.
"""


class ReActLoop:
    """
    Runs the Reason-Act-Observe loop for a single agent.

    This class is decoupled from the Agent - it takes an agent as input
    and drives the loop externally. This makes it reusable across
    different agent types and easy to test.

    Parameters
    ----------
    agent       : Agent instance with tools registered
    max_steps   : max number of tool calls before forcing a stop
                  (prevents infinite loops)
    verbose     : print loop steps to console
    """

    def __init__(self, agent, max_steps: int = 3, verbose: bool = True):
        self.agent     = agent
        self.max_steps = max_steps
        self.verbose   = verbose

    # -------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------

    def run(self, user_message: str) -> str:
        """
        Run the full ReAct loop for a given user message.

        Returns the final answer string.
        """
        if self.verbose:
            self._print_header(user_message)

        # Inject the ReAct instructions into the agent's role
        # (only if not already injected - prevents duplicating on re-runs)
        if REACT_SYSTEM_ADDON not in self.agent.role:
            self.agent.role += REACT_SYSTEM_ADDON

        # Start the loop
        step          = 0
        current_input = user_message

        while step < self.max_steps:
            step += 1

            if self.verbose:
                self._print_step(step, "THINKING", current_input)

            # -- Step A: Ask the LLM -------------------------------
            llm_response = self.agent.chat(current_input)

            # -- Step B: Parse the response ------------------------
            parsed = self._parse_response(llm_response)

            # -- Step C: Branch on what the LLM decided ------------

            if parsed["type"] == "final_answer":
                # LLM is done - return the answer
                answer = parsed["content"]
                if self.verbose:
                    self._print_final(answer)
                return answer

            elif parsed["type"] == "tool_call":
                tool_name  = parsed["tool_name"]
                tool_input = parsed["tool_input"]

                if self.verbose:
                    self._print_tool_call(tool_name, tool_input)

                # -- Step D: Execute the tool ----------------------
                tool_result = self.agent.use_tool(tool_name, tool_input)

                if self.verbose:
                    self._print_observation(tool_result)

                # -- Step D2: Knowledge Graph Extraction ----------
                kg_subgraph_str = ""
                if getattr(self.agent, "graph_memory", None):
                    self.agent.graph_memory.extract_from_text(
                        str(tool_result),
                        source_doc=f"tool:{tool_name}"
                    )
                    kg_subgraph_str = self.agent.graph_memory.get_context_for_prompt(tool_input)

                # -- Step E: Feed result back to LLM --------------
                # The next message to the LLM includes the tool result and KG update
                current_input = (
                    f"Tool '{tool_name}' returned:\n{tool_result}\n\n"
                    f"{kg_subgraph_str}\n\n"
                    f"Now continue - either call another tool or give FINAL_ANSWER."
                )

            else:
                # LLM gave a direct response - treat as final answer
                if self.verbose:
                    print(f"{Fore.CYAN}[ReAct] Direct response received - treating as final answer.{Style.RESET_ALL}")
                return llm_response

        # -- Max steps reached -------------------------------------
        # Force the LLM to stop and summarize what it has so far
        if self.verbose:
            print(f"{Fore.RED}[ReAct] Max steps ({self.max_steps}) reached. "
                  f"Forcing final answer.{Style.RESET_ALL}")

        forced_response = self.agent.chat(
            "You have reached the maximum number of steps. "
            "Please give your FINAL_ANSWER now based on what you know so far."
        )
        parsed = self._parse_response(forced_response)
        return parsed.get("content", forced_response)

    # -------------------------------------------------------------
    # Response parser
    # -------------------------------------------------------------

    def _parse_response(self, response: str) -> dict:
        """
        Parse the LLM's response into a structured dict.

        Returns one of:
          {"type": "tool_call",    "tool_name": ..., "tool_input": ...}
          {"type": "final_answer", "content": ...}
          {"type": "unknown",      "content": ...}
        """
        response = response.strip()

        # Check for TOOL_CALL
        if TOOL_CALL_TOKEN in response:
            # Extract everything after "TOOL_CALL:"
            after_token = response.split(TOOL_CALL_TOKEN, 1)[1].strip()

            # Split on "|" to get tool_name and tool_input
            if "|" in after_token:
                parts      = after_token.split("|", 1)
                tool_name  = parts[0].strip()
                tool_input = parts[1].strip()
            else:
                # No "|" found - treat whole thing as tool name with empty input
                tool_name  = after_token.strip()
                tool_input = ""

            return {
                "type":       "tool_call",
                "tool_name":  tool_name,
                "tool_input": tool_input,
            }

        # Check for FINAL_ANSWER
        if FINAL_ANSWER_TOKEN in response:
            answer = response.split(FINAL_ANSWER_TOKEN, 1)[1].strip()
            return {
                "type":    "final_answer",
                "content": answer,
            }

        # Neither found - LLM didn't follow the protocol
        return {
            "type":    "unknown",
            "content": response,
        }

    # -------------------------------------------------------------
    # Pretty printers
    # -------------------------------------------------------------

    def _print_header(self, message: str):
        import config
        config.safe_print(f"\n{'='*60}")
        config.safe_print(f"{Fore.CYAN}ReAct Loop Started | Agent: {self.agent.name}{Style.RESET_ALL}")
        config.safe_print(f"{Fore.WHITE}User: {message}{Style.RESET_ALL}")
        config.safe_print(f"{'='*60}")

    def _print_step(self, step: int, label: str, content: str):
        import config
        config.safe_print(f"\n{Fore.YELLOW}[Step {step}] {label}{Style.RESET_ALL}")
        short = content[:120] + "..." if len(content) > 120 else content
        config.safe_print(f"  Input: {short}")

    def _print_tool_call(self, tool_name: str, tool_input: str):
        import config
        config.safe_print(f"{Fore.MAGENTA}  -> ACT: calling '{tool_name}' "
                          f"with '{tool_input}'{Style.RESET_ALL}")

    def _print_observation(self, result: str):
        import config
        short = result[:200] + "..." if len(result) > 200 else result
        config.safe_print(f"{Fore.BLUE}  <- OBSERVE: {short}{Style.RESET_ALL}")

    def _print_final(self, answer: str):
        import config
        config.safe_print(f"\n{Fore.GREEN}{'='*60}")
        config.safe_print(f"FINAL ANSWER:")
        config.safe_print(f"{answer}")
        config.safe_print(f"{'='*60}{Style.RESET_ALL}")