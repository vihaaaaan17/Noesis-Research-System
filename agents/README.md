# Agents Module Reference

This directory contains the core agent definitions and base classes for the Multi-Agent System (MAS).

## File: `base_agents.py`
The [base_agents.py](file:///d:/koding/codes/Machine%20Learning/projects/MAS/agents/base_agents.py) file defines a stateful **`Agent`** class backed by the **Groq API**. It manages conversation history, system roles, and registration/execution of tools.

---

## 1. Class: `Agent`

### Constructor: `__init__`
Initializes a new instance of an agent, setting its parameters, setting up its Groq client, and preparing its conversation history and tool dictionary.

#### Parameters & Instance Variables
| Variable | Scope | Type | Purpose |
| :--- | :--- | :--- | :--- |
| `name` | Parameter / Instance (`self.name`) | `str` | The unique name of the agent (e.g., `"CoderAgent"`). Useful for identifying which agent is active in console logs. |
| `role` | Parameter / Instance (`self.role`) | `str` | The system prompt/instructions setting the agent's persona. |
| `model` | Parameter / Instance (`self.model`) | `str` | The LLM model to use. If not provided, it falls back to the default model from the global `config` module. |
| `max_tokens` | Parameter / Instance (`self.max_tokens`) | `int` | The maximum length of the response the agent is allowed to generate. Falls back to config default if `None`. |
| `temperature` | Parameter / Instance (`self.temperature`) | `float` | Controls the randomness/creativity of the model's responses (ranges from 0.0 to 2.0). Falls back to config default if `None`. |
| `verbose` | Parameter / Instance (`self.verbose`) | `bool` | Toggle for printing colored debug logs to the terminal. Falls back to config default. |
| `self.history` | Instance | `list[dict]` | A list of dictionaries representing the ongoing conversation history (e.g., messages with `role` and `content`). |
| `self.tools` | Instance | `dict` | A dictionary mapping tool names to registered tool instances that this agent can use. |
| `self._client` | Instance | `Groq` | The Groq API client initialized using the API key from your config. |

---

### Core Chat & LLM Methods

#### `chat(user_message)`
* **Purpose:** The main entry point to talk to the agent. It takes a new message from the user, updates the log, requests a response from the LLM, logs the reply, and returns it.
* **Variables:**
  * `user_message` (Parameter - `str`): The text query sent by the user.
  * `response_text` (Local - `str`): Holds the text response returned after sending the full history to the LLM.

#### `_build_messages()`
* **Purpose:** Combines the static system role instructions with the running conversation history into a single list of messages formatted for the API.
* **Variables:**
  * Uses `self.role` and `self.history` directly to build and return a unified list of message dictionaries.

#### `_call_llm(messages)`
* **Purpose:** Performs the actual HTTP request to the Groq API endpoint to obtain a text completion and handles potential network/API errors.
* **Variables:**
  * `messages` (Parameter - `list[dict]`): The complete history of messages (prepended with the system prompt) to send to Groq.
  * `response` (Local - `ChatCompletion`): The raw response object returned by the Groq API.
  * `exc` (Local - `Exception`): Represents any error raised during the API call (e.g., rate limits, invalid API key, network issues).
  * `error_msg` (Local - `str`): A formatted error message containing the agent's name and the exception details.

---

### Utility & Memory Methods

#### `reset()`
* **Purpose:** Erases the conversational memory of the agent, letting you start a fresh discussion from scratch.
* **Variables:**
  * Clears `self.history` (re-initializes to `[]`) and uses `self.verbose` to decide whether to output a confirmation log.

#### `inject_context(context)`
* **Purpose:** Allows you to insert pre-existing context (like a pre-written answer or summary) directly into the agent's memory as if the assistant had said it.
* **Variables:**
  * `context` (Parameter - `str`): The text block to manually append to the chat logs.
  * `preview` (Local - `str`): A sliced preview of the context (first 80 characters) used for the debug print log.

#### `get_history()`
* **Purpose:** Returns the current list of conversation messages.
* **Variables:**
  * Returns `self.history`.

#### `__repr__`
* **Purpose:** Returns a clean string representation of the class object for debugging.

---

### Tool & Capability Management

#### `register_tool(tool)`
* **Purpose:** Registers an external capability (a tool object) to the agent.
* **Variables:**
  * `tool` (Parameter - `Tool` object): The tool instance you want to register. It must have a `.name` and `.description` attribute.

#### `use_tool(tool_name, tool_input)`
* **Purpose:** Allows the agent or coordinator to execute one of the registered tools by name with a specified input.
* **Variables:**
  * `tool_name` (Parameter - `str`): The name of the registered tool to call (e.g., `"web_search"`).
  * `tool_input` (Parameter - `str`): The parameter/input data string to pass to the tool (e.g., the search query).
  * `available` (Local - `list[str]`): A list of registered tool names, used to return a helpful error message if the requested tool isn't found.
  * `tool` (Local - `Tool` object): The retrieved tool object matched from `self.tools`.
  * `result` (Local - `str`): The output returned by running the tool via `tool.run(tool_input)`.

#### `list_tools()`
* **Purpose:** Returns metadata for all registered tools as a list of simple Python dictionaries.
* **Variables:**
  * `t` (Local loop - `Tool` object): Loop variable representing each tool stored in the values of `self.tools`.

#### `_build_tool_prompt()`
* **Purpose:** Generates a text string summarizing all the agent's available tools. This can be appended to the agent's instructions so the LLM understands what it has access to.
* **Variables:**
  * `lines` (Local - `list[str]`): A list of text lines initialized to build the formatted prompt string.
  * `tool` (Local loop - `Tool` object): Loop variable used to iterate through all tools in `self.tools.values()`.
