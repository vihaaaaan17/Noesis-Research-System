"""
memory/graph_memory.py
---------------------------------------------------------------------
Production Knowledge Graph Shared Memory for MAS.

Implements NetworkX MultiDiGraph storage, Pydantic entity/relation schemas,
entity resolution (alias mapping), k-hop neighborhood retrieval, graph
serialization for LLM prompts, structural validation, and JSON persistence.

Based on skills/knowledge_graph_engineering/SKILL.md.
---------------------------------------------------------------------
"""

import os
import re
import json
import datetime
from typing import Literal, Optional, List, Set, Dict, Any
import networkx as nx
from pydantic import BaseModel, Field
from colorama import Fore, Style, init

init(autoreset=True)

EntityType = Literal[
    "PERSON", "ORGANIZATION", "LOCATION", "CONCEPT",
    "EQUATION", "METHOD", "VARIABLE", "METRIC", "ARTIFACT", "UNKNOWN"
]

VAGUE_PREDICATES = {"related to", "associated with", "involved with", "connected to"}


class Entity(BaseModel):
    name: str
    type: str = "CONCEPT"
    description: str = ""
    key_facts: List[str] = Field(default_factory=list)


class Relation(BaseModel):
    source: str
    predicate: str
    target: str
    source_type: str = "CONCEPT"
    target_type: str = "CONCEPT"
    confidence: float = 1.0


class ExtractedGraph(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)


class KnowledgeGraphMemory:
    """
    Durable, queryable, provenance-carrying shared Knowledge Graph memory.
    Powered by NetworkX MultiDiGraph.
    """

    def __init__(self, storage_path: str = "long_term_graph.json", verbose: bool = False):
        self.storage_path = storage_path
        self.verbose = verbose
        self.graph = nx.MultiDiGraph()
        self.alias_map: Dict[str, str] = {}
        self.schema_version = "1.0"

        if os.path.exists(self.storage_path):
            self.load_from_json(self.storage_path)

    # -------------------------------------------------------------
    # Entity Resolution & Disambiguation
    # -------------------------------------------------------------

    def resolve_entity(self, name: str) -> str:
        """Resolve entity name to canonical form using alias map."""
        clean = name.strip()
        if not clean:
            return "UNKNOWN"
        lower_key = clean.lower()
        if lower_key in self.alias_map:
            return self.alias_map[lower_key]
        return clean

    def register_alias(self, alias: str, canonical: str) -> None:
        """Map an alias variant to a canonical entity name."""
        self.alias_map[alias.strip().lower()] = canonical.strip()

    # -------------------------------------------------------------
    # Node & Edge Assembly
    # -------------------------------------------------------------

    def add_entity(
        self,
        name: str,
        entity_type: str = "CONCEPT",
        description: str = "",
        key_facts: Optional[List[str]] = None,
        source_doc: str = "system"
    ) -> str:
        """Add or update an entity node in the graph."""
        canonical = self.resolve_entity(name)
        self.register_alias(name, canonical)

        if not self.graph.has_node(canonical):
            self.graph.add_node(
                canonical,
                entity_type=entity_type,
                description=description,
                key_facts=key_facts or [],
                source_documents={source_doc},
                mention_count=1,
                created_at=datetime.datetime.now().isoformat()
            )
            if self.verbose:
                print(f"{Fore.CYAN}[KG Memory] Added entity node: {canonical} ({entity_type}){Style.RESET_ALL}")
        else:
            node_data = self.graph.nodes[canonical]
            node_data["mention_count"] += 1
            if isinstance(node_data.get("source_documents"), set):
                node_data["source_documents"].add(source_doc)
            else:
                node_data["source_documents"] = {source_doc}

            if description and not node_data.get("description"):
                node_data["description"] = description

            if key_facts:
                existing_facts = set(node_data.get("key_facts", []))
                existing_facts.update(key_facts)
                node_data["key_facts"] = list(existing_facts)

        return canonical

    def add_relation(
        self,
        source: str,
        predicate: str,
        target: str,
        source_doc: str = "system",
        confidence: float = 1.0,
        schema_version: str = "1.0"
    ) -> None:
        """Add a directed, provenance-tracked relation edge."""
        s_canonical = self.add_entity(source, source_doc=source_doc)
        t_canonical = self.add_entity(target, source_doc=source_doc)

        clean_pred = predicate.strip().lower()
        if not clean_pred:
            clean_pred = "related_to"

        # Check duplicate edge
        for _, _, data in self.graph.out_edges(s_canonical, data=True):
            if self.graph.has_edge(s_canonical, t_canonical):
                if data.get("predicate") == clean_pred and data.get("source_document_id") == source_doc:
                    return

        self.graph.add_edge(
            s_canonical,
            t_canonical,
            predicate=clean_pred,
            source_document_id=source_doc,
            extracted_at=datetime.datetime.now().isoformat(),
            confidence=confidence,
            schema_version=schema_version
        )
        if self.verbose:
            print(f"{Fore.MAGENTA}[KG Memory] Added edge: ({s_canonical}) --[{clean_pred}]--> ({t_canonical}){Style.RESET_ALL}")

    def add_extracted_graph(self, extracted: ExtractedGraph, source_doc: str = "system") -> None:
        """Ingest a full ExtractedGraph (entities + relations)."""
        for entity in extracted.entities:
            self.add_entity(
                name=entity.name,
                entity_type=entity.type,
                description=entity.description,
                key_facts=entity.key_facts,
                source_doc=source_doc
            )
        for rel in extracted.relations:
            self.add_relation(
                source=rel.source,
                predicate=rel.predicate,
                target=rel.target,
                source_doc=source_doc,
                confidence=rel.confidence
            )

    # -------------------------------------------------------------
    # Heuristic & Regex Extraction
    # -------------------------------------------------------------

    def extract_from_text(self, text: str, source_doc: str = "agent_turn") -> ExtractedGraph:
        """
        Extract structured entities and relations from text using deterministic heuristics.
        Catches equations, definitions, physical variables, and tool outputs.
        """
        extracted = ExtractedGraph()

        # 1. Catch equations (e.g. E = mc^2, V = I * R, subthreshold swing = kT/q * ln(10))
        eq_matches = re.findall(r"([A-Za-z0-9_\s\\{\}\(\)]+)\s*=\s*([A-Za-z0-9_\s\+\-\*\/\^\(\)\.\\]+)", text)
        for lhs, rhs in eq_matches:
            lhs_clean = lhs.strip()
            rhs_clean = rhs.strip()
            if 1 < len(lhs_clean) < 40 and 1 < len(rhs_clean) < 100:
                extracted.entities.append(Entity(
                    name=lhs_clean,
                    type="EQUATION",
                    description=f"Defined as {rhs_clean}"
                ))
                extracted.relations.append(Relation(
                    source=lhs_clean,
                    predicate="equals",
                    target=rhs_clean,
                    source_type="EQUATION",
                    target_type="CONCEPT"
                ))

        # 2. Catch definitions ("X is defined as Y", "X is a Y")
        def_matches = re.findall(r"([A-Z][a-zA-Z0-9_\s]{2,30})\s+(is|are|defined as)\s+([^.\n]{5,80})", text)
        for term, verb, definition in def_matches:
            t_clean = term.strip()
            d_clean = definition.strip()
            extracted.entities.append(Entity(
                name=t_clean,
                type="CONCEPT",
                description=d_clean
            ))
            extracted.relations.append(Relation(
                source=t_clean,
                predicate=verb.strip().replace(" ", "_"),
                target=d_clean,
                source_type="CONCEPT",
                target_type="CONCEPT"
            ))

        # 3. Catch tool output markers
        if "SymPy" in text or "solve" in text:
            extracted.entities.append(Entity(name="SymPyTool", type="METHOD", description="Symbolic math solver"))
        if "Numerical" in text or "SciPy" in text or "NumPy" in text:
            extracted.entities.append(Entity(name="NumericalTool", type="METHOD", description="Numerical computation engine"))
        if "arXiv" in text:
            extracted.entities.append(Entity(name="ArxivSearchTool", type="METHOD", description="Academic paper lookup"))

        # 4. Catch explicit concept lists ("Key concepts include: A, B, C", "Entities: A, B, C")
        concept_list_matches = re.findall(r"(?:Key concepts|Entities|Focus on|include:)\s*([A-Za-z0-9_,\s\-\(\)]+)", text, re.IGNORECASE)
        for clist in concept_list_matches:
            items = [item.strip() for item in clist.split(",") if 1 < len(item.strip()) < 50]
            for item_name in items:
                extracted.entities.append(Entity(name=item_name, type="CONCEPT"))

        # Ingest directly into self
        self.add_extracted_graph(extracted, source_doc=source_doc)
        return extracted

    # -------------------------------------------------------------
    # Graph Retrieval & Serialization
    # -------------------------------------------------------------

    def k_hop_neighborhood(self, center_entities: List[str], k: int = 2) -> Set[str]:
        """BFS k-hop expansion starting from center entity nodes."""
        nodes: Set[str] = set()
        frontier: Set[str] = set()

        for c in center_entities:
            canonical = self.resolve_entity(c)
            if self.graph.has_node(canonical):
                nodes.add(canonical)
                frontier.add(canonical)

        for _ in range(k):
            nxt: Set[str] = set()
            for n in frontier:
                succ = set(self.graph.successors(n))
                pred = set(self.graph.predecessors(n))
                nxt |= succ | pred
            frontier = nxt - nodes
            nodes |= frontier

        return nodes

    def search_nodes(self, query: str, top_k: int = 5) -> List[str]:
        """Simple keyword matching node search."""
        query_terms = query.lower().split()
        matches = []
        for node, data in self.graph.nodes(data=True):
            score = 0
            node_str = f"{node} {data.get('description', '')} {' '.join(data.get('key_facts', []))}".lower()
            for term in query_terms:
                if term in node_str:
                    score += 1
            if score > 0:
                matches.append((score, node))
        matches.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in matches[:top_k]]

    def serialize_subgraph(self, nodes: Optional[Set[str]] = None) -> str:
        """Serialize a subgraph into clean triple text for LLM prompts."""
        if nodes is None:
            subgraph = self.graph
        else:
            subgraph = self.graph.subgraph(nodes)

        lines = []
        for s, t, d in subgraph.edges(data=True):
            pred = d.get("predicate", "related_to")
            prov = d.get("source_document_id", "system")
            lines.append(f"({s}) --[{pred}]--> ({t}) [source: {prov}]")

        if not lines:
            return "Knowledge Graph is empty."

        return "\n".join(sorted(set(lines)))

    def get_context_for_prompt(self, query: str, k: int = 2) -> str:
        """Search relevant seed entities, expand k-hops, and serialize to string."""
        seeds = self.search_nodes(query, top_k=3)
        if not seeds:
            # Fall back to top degree hub nodes
            degrees = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)
            seeds = [n for n, _ in degrees[:3]]

        if not seeds:
            return ""

        nodes = self.k_hop_neighborhood(seeds, k=k)
        triples = self.serialize_subgraph(nodes)
        return f"=== KNOWLEDGE GRAPH CONTEXT (Shared Memory) ===\n{triples}\n==============================================="

    # -------------------------------------------------------------
    # Graph Validation
    # -------------------------------------------------------------

    def validate(self) -> Dict[str, Any]:
        """Perform structural validation checks."""
        errors = []
        warnings = []

        # Orphan edges
        for s, t, _ in self.graph.edges(data=True):
            if s not in self.graph.nodes or t not in self.graph.nodes:
                errors.append(f"Orphan edge: ({s}) -> ({t})")

        # Self-loops
        for n in self.graph.nodes:
            if self.graph.has_edge(n, n):
                warnings.append(f"Self-loop: {n}")

        components = list(nx.weakly_connected_components(self.graph))
        density = self.graph.number_of_edges() / max(self.graph.number_of_nodes(), 1)

        predicates = {d["predicate"] for _, _, d in self.graph.edges(data=True)}
        vague = [p for p in predicates if p in VAGUE_PREDICATES]

        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_components": len(components),
            "density": density,
            "errors": errors,
            "warnings": warnings,
            "vague_predicates": vague
        }

    # -------------------------------------------------------------
    # JSON Persistence
    # -------------------------------------------------------------

    def save_to_json(self, filepath: Optional[str] = None) -> None:
        """Persist graph state to a JSON file."""
        target = filepath or self.storage_path
        nodes_data = []
        for n, data in self.graph.nodes(data=True):
            data_copy = dict(data)
            if isinstance(data_copy.get("source_documents"), set):
                data_copy["source_documents"] = list(data_copy["source_documents"])
            data_copy["name"] = n
            nodes_data.append(data_copy)

        edges_data = []
        for s, t, data in self.graph.edges(data=True):
            edge_dict = dict(data)
            edge_dict["source"] = s
            edge_dict["target"] = t
            edges_data.append(edge_dict)

        payload = {
            "schema_version": self.schema_version,
            "alias_map": self.alias_map,
            "nodes": nodes_data,
            "edges": edges_data
        }

        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        if self.verbose:
            print(f"{Fore.GREEN}[KG Memory] Graph saved to {target} ({len(nodes_data)} nodes, {len(edges_data)} edges){Style.RESET_ALL}")

    def load_from_json(self, filepath: Optional[str] = None) -> None:
        """Load graph state from a JSON file."""
        target = filepath or self.storage_path
        if not os.path.exists(target):
            return

        with open(target, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.alias_map = payload.get("alias_map", {})
        self.graph.clear()

        for node_info in payload.get("nodes", []):
            name = node_info.pop("name")
            if "source_documents" in node_info and isinstance(node_info["source_documents"], list):
                node_info["source_documents"] = set(node_info["source_documents"])
            self.graph.add_node(name, **node_info)

        for edge_info in payload.get("edges", []):
            source = edge_info.pop("source")
            target_node = edge_info.pop("target")
            self.graph.add_edge(source, target_node, **edge_info)

        if self.verbose:
            print(f"{Fore.GREEN}[KG Memory] Graph loaded from {target} ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges){Style.RESET_ALL}")
