"""
memory/long_term.py
---------------------------------------------------------------------
Long-Term Semantic Memory

Combines persistent structured facts (JSON key-value store) and a local
ChromaDB Vector Database for semantic research chunk retrieval.

Architecture:
  LongTermMemory
  ├── semantic_search()  --> ChromaDB Vector Collection
  ├── recall_fact()      --> Structured JSON Facts
  └── store()            --> Facts & Research Chunks
---------------------------------------------------------------------
"""

import json
import os
import time
from typing import List, Dict, Any, Optional
from colorama import Fore, Style, init

init(autoreset=True)

DEFAULT_MEMORY_FILE = "long_term_memory.json"
DEFAULT_CHROMA_DIR = "chroma_db"


class LongTermMemory:
    """
    Long-Term Semantic Memory System.
    Exposes a unified interface over persistent JSON facts and a local ChromaDB vector store.
    """

    def __init__(
        self,
        filepath: str = DEFAULT_MEMORY_FILE,
        chroma_db_dir: str = DEFAULT_CHROMA_DIR,
        verbose: bool = True
    ):
        self.filepath = filepath
        self.chroma_db_dir = chroma_db_dir
        self.verbose = verbose

        self._data: dict = {
            "facts": {},
            "notes": [],
        }

        # 1. Initialize persistent JSON facts store
        self._load()

        # 2. Initialize local ChromaDB Vector Store
        self._chroma_client = None
        self._collection = None
        self._setup_chroma()

    def _setup_chroma(self) -> None:
        """Initialize ChromaDB local persistent client and collection."""
        try:
            import chromadb
            os.makedirs(self.chroma_db_dir, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=self.chroma_db_dir)
            self._collection = self._chroma_client.get_or_create_collection(name="mas_research_memory")
            if self.verbose:
                print(f"{Fore.CYAN}[LongTermMemory] ChromaDB vector store initialized at '{self.chroma_db_dir}'{Style.RESET_ALL}")
        except Exception as e:
            if self.verbose:
                print(f"{Fore.YELLOW}[LongTermMemory] ChromaDB setup warning: {e}. Falling back to local term search.{Style.RESET_ALL}")

    # -------------------------------------------------------------
    # Facts - Structured Key-Value Store
    # -------------------------------------------------------------

    def remember(self, key: str, value: str) -> None:
        """Store a fact under a key."""
        self._data["facts"][key] = {
            "value":     value,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._save()

        if self.verbose:
            print(f"{Fore.GREEN}[LongTermMemory] Stored fact: '{key}' = '{str(value)[:60]}'{Style.RESET_ALL}")

    def remember_fact(self, key: str, value: str) -> None:
        """Alias for remember()."""
        self.remember(key, value)

    def recall(self, key: str) -> Optional[str]:
        """Retrieve a fact by exact key."""
        entry = self._data["facts"].get(key)
        if entry:
            return entry["value"]
        return None

    def recall_fact(self, key: str) -> Optional[str]:
        """Alias for recall()."""
        return self.recall(key)

    def forget(self, key: str) -> bool:
        """Delete a fact by key."""
        if key in self._data["facts"]:
            del self._data["facts"][key]
            self._save()
            if self.verbose:
                print(f"{Fore.YELLOW}[LongTermMemory] Deleted fact: '{key}'{Style.RESET_ALL}")
            return True
        return False

    def all_facts(self) -> dict:
        """Return all stored facts as a dict."""
        return {k: v["value"] for k, v in self._data["facts"].items()}

    # -------------------------------------------------------------
    # Unified Store: Facts & Research Chunks
    # -------------------------------------------------------------

    def store(
        self,
        content: Optional[str] = None,
        key: Optional[str] = None,
        value: Optional[str] = None,
        metadata: Optional[dict] = None,
        tags: Optional[List[str]] = None
    ) -> Any:
        """
        Unified Store Interface:
        - If key & value provided: stores structured fact.
        - If content provided: stores research chunk into JSON notes & ChromaDB Vector store.
        """
        if key and value:
            self.remember_fact(key, value)
            return key

        if content:
            return self.store_document(content, metadata=metadata or {"tags": tags or []})

        return None

    def store_document(self, content: str, metadata: Optional[dict] = None) -> int:
        """
        Store reusable research document chunk/finding in JSON storage AND ChromaDB Vector collection.
        Metadata: (source, phase, topic, agent, timestamp, confidence, document_id).
        """
        meta = metadata or {}
        doc_id = len(self._data["notes"])
        doc_id_str = f"doc_{doc_id}_{int(time.time())}"

        note = {
            "id":          doc_id,
            "doc_id":      doc_id_str,
            "content":     content,
            "tags":        meta.get("tags", []),
            "metadata":    meta,
            "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._data["notes"].append(note)
        self._save()

        # Ingest into ChromaDB collection
        if self._collection:
            try:
                # Sanitize metadata values to simple primitives for ChromaDB
                clean_meta = {}
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    elif isinstance(v, list):
                        clean_meta[k] = ",".join(str(item) for item in v)
                clean_meta["timestamp"] = note["timestamp"]

                self._collection.upsert(
                    documents=[content],
                    metadatas=[clean_meta],
                    ids=[doc_id_str]
                )
            except Exception as e:
                if self.verbose:
                    print(f"{Fore.YELLOW}[LongTermMemory] ChromaDB upsert warning: {e}{Style.RESET_ALL}")

        if self.verbose:
            print(f"{Fore.GREEN}[LongTermMemory] Stored research chunk #{doc_id} in Vector DB | meta={meta} | '{content[:60]}...'{Style.RESET_ALL}")

        return doc_id

    def add_note(self, content: str, tags: Optional[List[str]] = None) -> int:
        """Alias for store_document."""
        return self.store_document(content, metadata={"tags": tags or []})

    # -------------------------------------------------------------
    # Semantic Retrieval (ChromaDB Vector Store)
    # -------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        """Search notes / research chunks using ChromaDB semantic search."""
        return self.semantic_search(query, top_k=top_k)

    def semantic_search(self, query: str, top_k: int = 3) -> List[dict]:
        """
        Perform semantic search across ChromaDB Vector Collection.
        Falls back to term frequency search if ChromaDB is empty or uninitialized.
        """
        if not query:
            return []

        # Step 1: Query ChromaDB Vector Collection
        if self._collection:
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(top_k, max(1, self._collection.count()))
                )
                formatted = []
                if results and "documents" in results and results["documents"]:
                    docs = results["documents"][0]
                    metas = results.get("metadatas", [[]])[0]
                    ids = results.get("ids", [[]])[0]
                    distances = results.get("distances", [[]])[0] if "distances" in results else [0.0]*len(docs)

                    for d, m, i, dist in zip(docs, metas, ids, distances):
                        formatted.append({
                            "doc_id": i,
                            "content": d,
                            "metadata": m,
                            "_score": round(1.0 / (1.0 + dist), 4) if dist is not None else 1.0
                        })

                if formatted:
                    if self.verbose:
                        print(f"{Fore.CYAN}[LongTermMemory] ChromaDB Vector Search '{query}' -> {len(formatted)} result(s){Style.RESET_ALL}")
                    return formatted
            except Exception as e:
                if self.verbose:
                    print(f"{Fore.YELLOW}[LongTermMemory] ChromaDB search error: {e}. Running local term search.{Style.RESET_ALL}")

        # Step 2: Fallback local term frequency search over JSON notes
        query_words = [w.lower() for w in query.split() if len(w) > 2]
        if not query_words or not self._data["notes"]:
            return []

        scored = []
        for note in self._data["notes"]:
            text = (note["content"] + " " + " ".join(note.get("tags", []))).lower()
            score = sum(text.count(w) for w in query_words)
            if score > 0:
                scored.append({**note, "_score": score})

        scored.sort(key=lambda x: x["_score"], reverse=True)
        return scored[:top_k]

    def get_note(self, note_id: int) -> Optional[dict]:
        """Retrieve a note by ID."""
        for note in self._data["notes"]:
            if note["id"] == note_id:
                return note
        return None

    def all_notes(self) -> List[dict]:
        """Return all stored notes."""
        return list(self._data["notes"])

    # -------------------------------------------------------------
    # Context Retrieval Interface
    # -------------------------------------------------------------

    def get_context(self, query: str = "", max_facts: int = 5) -> str:
        """Query-aware Long-Term Memory context retrieval."""
        return self.build_context_string(query=query, max_facts=max_facts)

    def build_context_string(self, query: Optional[str] = None, max_facts: int = 5) -> str:
        """Build formatted string of facts and semantic vector search chunks."""
        lines = ["[Long-Term Memory Context]"]

        # 1. Facts
        facts = self.all_facts()
        if facts:
            lines.append("\nKnown facts:")
            for key, val in list(facts.items())[:max_facts]:
                lines.append(f"  * {key}: {val}")

        # 2. Vector Semantic Research Chunks
        if query:
            chunks = self.semantic_search(query, top_k=3)
            if chunks:
                lines.append(f"\nSemantic Research Chunks for '{query}':")
                for chunk in chunks:
                    lines.append(f"  [Chunk #{chunk.get('doc_id', 'note')}] {chunk['content'][:200]}")

        if len(lines) == 1:
            return ""

        return "\n".join(lines)

    # -------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------

    def _save(self) -> None:
        """Write facts and notes to JSON file."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        """Load facts and notes from JSON file."""
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except Exception:
            pass

    def wipe(self) -> None:
        """Delete all memory (facts + vector database)."""
        self._data = {"facts": {}, "notes": []}
        self._save()
        if self._collection:
            try:
                self._chroma_client.delete_collection("mas_research_memory")
                self._collection = self._chroma_client.get_or_create_collection("mas_research_memory")
            except Exception:
                pass
        if self.verbose:
            print(f"{Fore.RED}[LongTermMemory] All memory wiped.{Style.RESET_ALL}")

    def stats(self) -> dict:
        vector_count = self._collection.count() if self._collection else 0
        return {
            "facts": len(self._data["facts"]),
            "notes": len(self._data["notes"]),
            "vector_chunks": vector_count,
            "filepath": self.filepath,
            "chroma_dir": self.chroma_db_dir,
        }

    def __repr__(self):
        s = self.stats()
        return (f"LongTermMemory(facts={s['facts']}, "
                f"vector_chunks={s['vector_chunks']}, file={s['filepath']!r})")