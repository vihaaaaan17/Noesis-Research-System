"""
core/react_loop.py
---------------------------------------------------------------------
The ReAct Loop - Reason, Act, Observe

Thin orchestrator for tool-using agents.
Driven purely by: Reason -> Act -> Observe -> Repeat.
---------------------------------------------------------------------
"""

from typing import Dict, Any, Optional
from colorama import Fore, Style, init

init(autoreset=True)


# -- Protocol tokens -----------------------------------------------
TOOL_CALL_TOKEN = "TOOL_CALL:"
FINAL_ANSWER_TOKEN = "FINAL_ANSWER:"

# Maximum characters allowed in an observation string fed back to LLM prompt
MAX_OBSERVATION_CHARS = 1500

# System prompt addon injected into tool-using agents
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
"""


class ReActLoop:
    """
    Runs the Reason-Act-Observe loop for a single agent.
    Thin orchestrator: handles tool validation, execution, and step control.
    Memory ingestion and graph traversal are delegated entirely to the Agent/WorkingMemory layer.
    """

    def __init__(self, agent, max_steps: int = 3, verbose: bool = True):
        self.agent = agent
        self.max_steps = max_steps
        self.verbose = verbose

    def run(self, user_message: str, phase: str = "computation") -> str:
        """
        Run the full ReAct loop for a given user message.
        Returns the clean final answer string.
        """
        budget = getattr(self.agent, "budget", None)

        if self.verbose:
            self._print_header(user_message)

        if REACT_SYSTEM_ADDON not in self.agent.role:
            self.agent.role += REACT_SYSTEM_ADDON

        step = 0
        current_input = user_message

        while step < self.max_steps:
            if budget is not None:
                check_res = budget.check_preflight(category="react")
                if not check_res["allowed"]:
                    if self.verbose:
                        print(f"{Fore.YELLOW}[ReAct] Budget check stopped loop: {check_res['detail']}{Style.RESET_ALL}")
                    return f"[ReAct Truncated]: {check_res['detail']}"

            step += 1

            if self.verbose:
                self._print_step(step, "THINKING", current_input)

            # Step A: Call Agent LLM
            llm_response = self.agent.chat(current_input, phase=phase)

            # Check if LLM call failed or returned an API error
            if hasattr(self.agent, "_validate_response") and not self.agent._validate_response(llm_response):
                if self.verbose:
                    print(f"{Fore.RED}[ReAct] Provider error encountered. Stopping loop.{Style.RESET_ALL}")
                return f"[Generation Error]: LLM call failed: {llm_response}"

            # Step B: Parse Response with strict protocol rules
            parsed = self._parse_response(llm_response)

            # Step C: Branch based on action
            if parsed["type"] == "final_answer":
                answer = parsed["content"]
                if self.verbose:
                    self._print_final(answer)
                return answer

            elif parsed["type"] == "tool_call":
                tool_name = parsed["tool_name"]
                tool_input = parsed["tool_input"]

                if self.verbose:
                    self._print_tool_call(tool_name, tool_input)

                # Step D1: Validate Tool Availability
                if tool_name not in self.agent.tools:
                    available = list(self.agent.tools.keys())
                    err_msg = f"Tool '{tool_name}' is not available. Available tools: {available}."
                    if self.verbose:
                        print(f"{Fore.RED}[ReAct] {err_msg}{Style.RESET_ALL}")
                    current_input = f"{err_msg}\nPlease choose an available tool or respond with FINAL_ANSWER."
                    continue

                # Step D2: Execute Tool
                tool_result_raw = self.agent.use_tool(tool_name, tool_input)
                tool_result_str = str(tool_result_raw)

                if self.verbose:
                    self._print_observation(tool_result_str)

                # Step D3: Tool Result Validation
                is_tool_error = (
                    tool_result_str.startswith("[Tool Error]") or
                    tool_result_str.startswith("Tool '") and "not found" in tool_result_str or
                    "429" in tool_result_str and "Rate" in tool_result_str or
                    "timeout" in tool_result_str.lower()
                )

                if is_tool_error:
                    if self.verbose:
                        print(f"{Fore.YELLOW}[ReAct] Tool call failed. Skipping memory ingestion.{Style.RESET_ALL}")
                    bounded_obs = tool_result_str[:MAX_OBSERVATION_CHARS]
                    current_input = (
                        f"Tool '{tool_name}' failed with observation:\n{bounded_obs}\n\n"
                        f"Decide whether to retry, try another tool, or provide FINAL_ANSWER."
                    )
                else:
                    # Ingest valid tool results into Working Memory
                    if hasattr(self.agent, "working_memory"):
                        self.agent.working_memory.ingest(
                            tool_result_str,
                            phase=phase,
                            source_doc=f"tool:{tool_name}"
                        )

                    # Create bounded observation for next LLM prompt
                    bounded_obs = tool_result_str
                    if len(bounded_obs) > MAX_OBSERVATION_CHARS:
                        bounded_obs = bounded_obs[:MAX_OBSERVATION_CHARS] + "\n... [Output truncated for prompt context]"

                    current_input = (
                        f"Tool '{tool_name}' returned:\n{bounded_obs}\n\n"
                        f"Now continue - either call another tool or give FINAL_ANSWER."
                    )

            else:
                # Direct model output without protocol token
                if self.verbose:
                    print(f"{Fore.CYAN}[ReAct] Direct model response received.{Style.RESET_ALL}")
                return llm_response

        # Max steps safety fallback
        if self.verbose:
            print(f"{Fore.RED}[ReAct] Max steps ({self.max_steps}) reached. Forcing final answer.{Style.RESET_ALL}")

        forced_response = self.agent.chat(
            "You have reached the maximum number of steps. Please give your FINAL_ANSWER now based on what you know so far.",
            phase=phase
        )
        parsed = self._parse_response(forced_response)
        return parsed.get("content", forced_response)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Strict protocol parser.
        Requires response to start with TOOL_CALL: or FINAL_ANSWER:
        """
        if not response or not isinstance(response, str):
            return {"type": "unknown", "content": ""}

        stripped = response.strip()

        # Strict start-of-response check for TOOL_CALL:
        if stripped.startswith(TOOL_CALL_TOKEN):
            after_token = stripped[len(TOOL_CALL_TOKEN):].strip()
            if "|" in after_token:
                parts = after_token.split("|", 1)
                tool_name = parts[0].strip()
                tool_input = parts[1].strip()
            else:
                tool_name = after_token.strip()
                tool_input = ""

            return {
                "type": "tool_call",
                "tool_name": tool_name,
                "tool_input": tool_input,
            }

        # Strict start-of-response check for FINAL_ANSWER:
        if stripped.startswith(FINAL_ANSWER_TOKEN):
            answer = stripped[len(FINAL_ANSWER_TOKEN):].strip()
            return {
                "type": "final_answer",
                "content": answer,
            }

        # Fallback for inline protocol markers if at start of line
        lines = stripped.splitlines()
        for line in lines:
            line_s = line.strip()
            if line_s.startswith(TOOL_CALL_TOKEN):
                after_token = line_s[len(TOOL_CALL_TOKEN):].strip()
                parts = after_token.split("|", 1) if "|" in after_token else [after_token, ""]
                return {
                    "type": "tool_call",
                    "tool_name": parts[0].strip(),
                    "tool_input": parts[1].strip() if len(parts) > 1 else "",
                }
            elif line_s.startswith(FINAL_ANSWER_TOKEN):
                answer = line_s[len(FINAL_ANSWER_TOKEN):].strip()
                return {
                    "type": "final_answer",
                    "content": answer,
                }

        return {
            "type": "unknown",
            "content": stripped,
        }

    # Pretty printers
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
        config.safe_print(f"{Fore.MAGENTA}  -> ACT: calling '{tool_name}' with '{tool_input}'{Style.RESET_ALL}")

    def _print_observation(self, result: str):
        import config
        short = result[:200] + "..." if len(result) > 200 else result
        config.safe_print(f"{Fore.BLUE}  <- OBSERVE: {short}{Style.RESET_ALL}")

    def _print_final(self, answer: str):
        import config
        config.safe_print(f"\n{Fore.GREEN}{'='*60}")
        config.safe_print("FINAL ANSWER:")
        config.safe_print(f"{answer}")
        config.safe_print(f"{'='*60}{Style.RESET_ALL}")