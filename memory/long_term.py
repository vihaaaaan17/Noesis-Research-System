"""
memory/long_term.py
---------------------------------------------------------------------
Long-Term Memory

Stores facts, results, and knowledge ACROSS sessions - survives
process restarts. Backed by a plain JSON file on disk (no database,
no vector store, no extra dependencies).

What it stores:
  * Key-value facts  ("user_name" -> "Arjun")
  * Tagged notes     ({"tag": "research", "content": "GaN HEMTs have..."})
  * Agent outputs    (results from previous runs you want to reuse)

Search:
  Simple keyword matching over stored content - no embeddings needed
  for most practical multi-agent tasks. Fast and transparent.

If you later want semantic search, just swap _keyword_search() for
a vector store call (Chroma, FAISS, etc.) without changing any other code.
---------------------------------------------------------------------
"""

import json
import os
import time
from colorama import Fore, Style, init

init(autoreset=True)

DEFAULT_MEMORY_FILE = "long_term_memory.json"


class LongTermMemory:
    """
    Persistent key-value + searchable note store.

    All data is saved to a JSON file after every write operation.
    On initialization, any existing file is loaded automatically.

    Parameters
    ----------
    filepath : str  - path to the JSON file (created if it doesn't exist)
    verbose  : bool - print memory events to console
    """

    def __init__(self, filepath: str = DEFAULT_MEMORY_FILE, verbose: bool = True):
        self.filepath = filepath
        self.verbose  = verbose

        # Two internal stores:
        #   facts  - simple key:value pairs  {"name": "Arjun"}
        #   notes  - list of tagged content entries for search
        self._data: dict = {
            "facts": {},
            "notes": [],
        }

        self._load()

    # -------------------------------------------------------------
    # Facts - simple key-value store
    # -------------------------------------------------------------

    def remember(self, key: str, value: str) -> None:
        """
        Store a fact under a key.
        Overwrites if the key already exists.

        Example:
            memory.remember("user_name", "Arjun")
            memory.remember("project", "GaN HEMT compact model")
        """
        self._data["facts"][key] = {
            "value":     value,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save()

        if self.verbose:
            print(f"{Fore.GREEN}[LongTermMemory] Stored: "
                  f"'{key}' = '{str(value)[:60]}'{Style.RESET_ALL}")

    def recall(self, key: str) -> str | None:
        """
        Retrieve a fact by exact key.
        Returns None if not found.

        Example:
            name = memory.recall("user_name")  # -> "Arjun"
        """
        entry = self._data["facts"].get(key)
        if entry:
            return entry["value"]
        return None

    def forget(self, key: str) -> bool:
        """Delete a fact by key. Returns True if it existed."""
        if key in self._data["facts"]:
            del self._data["facts"][key]
            self._save()
            if self.verbose:
                print(f"{Fore.YELLOW}[LongTermMemory] Deleted: '{key}'{Style.RESET_ALL}")
            return True
        return False

    def all_facts(self) -> dict:
        """Return all stored facts as {key: value} dict."""
        return {k: v["value"] for k, v in self._data["facts"].items()}

    # -------------------------------------------------------------
    # Notes - searchable tagged content
    # -------------------------------------------------------------

    def add_note(self, content: str, tags: list[str] = None) -> int:
        """
        Store a longer piece of content (a note) with optional tags.
        Returns the note's ID.

        Use this for:
          * Saving agent outputs to reuse later
          * Storing research findings
          * Logging tool results worth keeping

        Example:
            memory.add_note(
                "GaN HEMTs use a 2DEG at the AlGaN/GaN interface...",
                tags=["research", "hemt", "semiconductor"]
            )
        """
        note = {
            "id":        len(self._data["notes"]),
            "content":   content,
            "tags":      tags or [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._data["notes"].append(note)
        self._save()

        if self.verbose:
            print(f"{Fore.GREEN}[LongTermMemory] Note #{note['id']} saved "
                  f"| tags={tags} | '{content[:60]}...'{Style.RESET_ALL}")

        return note["id"]

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search notes by keyword relevance.

        Scores each note by how many query words appear in it,
        then returns the top_k most relevant notes.

        Parameters
        ----------
        query : str  - natural language or keyword query
        top_k : int  - max number of results to return

        Returns
        -------
        list of note dicts, sorted by relevance score (highest first)

        Example:
            results = memory.search("GaN HEMT 2DEG")
            for r in results:
                print(r["content"])
        """
        query_words = set(query.lower().split())
        scored      = []

        for note in self._data["notes"]:
            text       = (note["content"] + " " + " ".join(note["tags"])).lower()
            note_words = set(text.split())

            # Score = number of query words that appear in the note
            score = len(query_words & note_words)

            if score > 0:
                scored.append({**note, "_score": score})

        # Sort by score descending
        scored.sort(key=lambda x: x["_score"], reverse=True)
        results = scored[:top_k]

        if self.verbose:
            print(f"{Fore.CYAN}[LongTermMemory] Search '{query}' -> "
                  f"{len(results)} result(s){Style.RESET_ALL}")

        return results

    def get_note(self, note_id: int) -> dict | None:
        """Retrieve a specific note by ID."""
        for note in self._data["notes"]:
            if note["id"] == note_id:
                return note
        return None

    def all_notes(self) -> list[dict]:
        """Return all stored notes."""
        return list(self._data["notes"])

    # -------------------------------------------------------------
    # Context injection helper
    # -------------------------------------------------------------

    def build_context_string(self, query: str = None, max_facts: int = 10) -> str:
        """
        Build a formatted string of relevant memory to inject into
        an agent's context before a task.

        If query is given, searches notes for relevant ones.
        Always includes all stored facts.

        Example:
            context = memory.build_context_string("HEMT research")
            agent.inject_context(context)
        """
        lines = ["[Long-Term Memory Context]"]

        # Add facts
        facts = self.all_facts()
        if facts:
            lines.append("\nKnown facts:")
            for key, val in list(facts.items())[:max_facts]:
                lines.append(f"  * {key}: {val}")

        # Add relevant notes if query given
        if query:
            notes = self.search(query, top_k=3)
            if notes:
                lines.append(f"\nRelevant notes for '{query}':")
                for note in notes:
                    lines.append(f"  [Note #{note['id']}] {note['content'][:200]}")

        return "\n".join(lines)

    # -------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------

    def _save(self) -> None:
        """Write current memory to the JSON file."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        """Load memory from the JSON file if it exists."""
        if not os.path.exists(self.filepath):
            if self.verbose:
                print(f"{Fore.CYAN}[LongTermMemory] No existing file at "
                      f"'{self.filepath}'. Starting fresh.{Style.RESET_ALL}")
            return

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self._data = loaded
            if self.verbose:
                n_facts = len(self._data.get("facts", {}))
                n_notes = len(self._data.get("notes", []))
                print(f"{Fore.CYAN}[LongTermMemory] Loaded from '{self.filepath}': "
                      f"{n_facts} facts, {n_notes} notes.{Style.RESET_ALL}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"{Fore.RED}[LongTermMemory] Failed to load '{self.filepath}': "
                      f"{e}. Starting fresh.{Style.RESET_ALL}")

    def wipe(self) -> None:
        """Delete all memory (facts + notes) and clear the file."""
        self._data = {"facts": {}, "notes": []}
        self._save()
        if self.verbose:
            print(f"{Fore.RED}[LongTermMemory] All memory wiped.{Style.RESET_ALL}")

    # -------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "facts":    len(self._data["facts"]),
            "notes":    len(self._data["notes"]),
            "filepath": self.filepath,
        }

    def __repr__(self):
        s = self.stats()
        return (f"LongTermMemory(facts={s['facts']}, "
                f"notes={s['notes']}, file={s['filepath']!r})")