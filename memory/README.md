# Memory Module Reference

This directory contains the short-term and long-term memory management systems for the agents.

*   **Short-Term Memory (`short_term.py`)**: Manages in-context message history and applies context window limits (sliding window or LLM summarization).
*   **Long-Term Memory (`long_term.py`)**: Persists key-value facts and tagged search notes across restarts, backed by a simple JSON file.

---

## File: `short_term.py`
The [short_term.py](file:///d:/koding/codes/Machine%20Learning/projects/MAS/memory/short_term.py) file implements in-context message management.

### Class: `ShortTermMemory`

#### Constructor Parameters & Instance Variables
| Variable | Scope | Type | Purpose |
| :--- | :--- | :--- | :--- |
| `max_tokens` | Parameter / Instance (`self.max_tokens`) | `int` | Token threshold budget for short-term memory (not counting system prompt). |
| `strategy` | Parameter / Instance (`self.strategy`) | `str` | Trim strategy. E.g., `"window"` (drop oldest) or `"summarize"` (compress). |
| `summarize_fn` | Parameter / Instance (`self.summarize_fn`) | `callable` | Callback function pointing to the agent's LLM completion method, used to summarize context. |
| `verbose` | Parameter / Instance (`self.verbose`) | `bool` | Enables console print logs for memory actions (like dropping or summarizing). |
| `self.messages` | Instance | `list[dict]` | Chronological conversation logs sent to the LLM. |
| `self.summary` | Instance | `str` | Running summary of compressed historical turns. |

#### Methods

*   **`__init__(max_tokens=2000, strategy="window", summarize_fn=None, verbose=True)`**: Sets strategy constraints and initializes empty message lists and summary fields.
*   **`add(role, content)`**: Appends a new message dictionary `{"role": role, "content": content}` to history, then runs `_enforce_limit()`.
*   **`get_messages()`**: Returns current messages. If a summary exists, it formats and prepends it as a context system note.
*   **`clear()`**: Resets the conversation history and clears the summary text.
*   **`token_count()`**: Rough character-based token estimator (`chars // 4`).
*   **`message_count()`**: Returns the count of raw active messages.
*   **`_enforce_limit()`**: Inspects token usage and invokes `_apply_window()` or `_apply_summarize()` if memory limit is exceeded.
*   **`_apply_window()`**: Drops the oldest messages from `self.messages` one-by-one until token count fits inside the budget.
*   **`_apply_summarize()`**: Drops the older half of `self.messages`, converts them into a text block, and calls `self.summarize_fn()` to compress them.
*   **`stats()`**: Returns metrics on memory consumption.

---

### Deep Dive: How the LLM Summarizer Works
When strategy is `"summarize"`, the agent compresses old messages using its own LLM:

1. **Callback Binding:** The `Agent` passes its `_summarize_text` method to the `ShortTermMemory` constructor as `summarize_fn`.
2. **Context Compression:** When memory limit is breached, the older half of the conversation is bundled together.
3. **Live LLM Call:** Calling `self.summarize_fn(prompt)` invokes `Agent._summarize_text()`, which calls the Groq completion endpoint with the prompt:
   *"Summarize this conversation history in 3-5 bullet points..."*
4. **No Infinite Loops:** The summarization call bypasses the active history log (`self.history`) and directly executes a raw stateless `_call_llm()` call, avoiding recursive memory inflation.
5. **Context Injection:** The resulting summary text is stored in `self.summary` and automatically prepended to subsequent chats.

---

## File: `long_term.py`
The [long_term.py](file:///d:/koding/codes/Machine%20Learning/projects/MAS/memory/long_term.py) file implements persistent storage backed by a local JSON file.

### Class: `LongTermMemory`

#### Constructor Parameters & Instance Variables
| Variable | Scope | Type | Purpose |
| :--- | :--- | :--- | :--- |
| `filepath` | Parameter / Instance (`self.filepath`) | `str` | Destination path to the local JSON file on disk. |
| `verbose` | Parameter / Instance (`self.verbose`) | `bool` | Enables logging message statements on save, load, and wipe actions. |
| `self._data` | Instance | `dict` | Nested dictionary containing two stores: `"facts"` (key-value) and `"notes"` (list of dicts). |

#### Methods

*   **`__init__(filepath="long_term_memory.json", verbose=True)`**: Initializes the data layout and automatically loads any pre-existing JSON files.
*   **`remember(key, value)`**: Stores a key-value fact with a timestamp. Saves to disk immediately.
*   **`recall(key)`**: Retrieves a fact's value from memory. Returns `None` if not found.
*   **`forget(key)`**: Removes a fact from memory by key. Returns `True` if successful.
*   **`all_facts()`**: Returns all facts as a flat `{key: value}` dictionary.
*   **`add_note(content, tags)`**: Saves a larger text block (e.g., agent outputs, research findings) with a list of tags. Returns the note's integer ID.
*   **`search(query, top_k=3)`**: Searches notes by comparing overlapping words. Scores and returns the top `top_k` matches sorted descending by score.
*   **`get_note(note_id)`**: Fetches a note dictionary by its unique ID.
*   **`all_notes()`**: Returns the full list of saved note dictionaries.
*   **`build_context_string(query=None, max_facts=10)`**: Formats saved facts and matching notes into a clean instruction block to inject into an agent's system role.
*   **`_save()`**: Serializes `self._data` to a JSON file.
*   **`_load()`**: Deserializes JSON file contents into `self._data`.
*   **`wipe()`**: Wipes all facts and notes from memory and saves the empty state to disk.
*   **`stats()`**: Returns counts of saved facts, notes, and the file location.
