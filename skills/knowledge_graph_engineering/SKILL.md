---
name: knowledge-graph-engineering
description: >-
  Production-grade knowledge graph construction, GraphRAG retrieval, and graph
  reasoning for multi-agent systems. Covers LLM-based extraction, entity
  resolution, graph assembly, hybrid retrieval, traversal algorithms, Neo4j
  migration, evaluation, and agent integration. Use when building knowledge
  graphs, GraphRAG pipelines, multi-hop reasoning, entity linking, graph
  databases, or agent memory layers from unstructured documents or codebases.
---

# Knowledge Graph Engineering

Complete engineering handbook for building production-grade knowledge graphs and GraphRAG systems. Synthesizes Anthropic's Knowledge Graph Cookbook, agent architecture patterns, and production extensions.

**Internal organization**: This single skill contains three capabilities — (1) Knowledge Graph Construction, (2) GraphRAG Retrieval, (3) Graph Reasoning — organized as sections below. Load this one skill; navigate to the relevant section.

**Supporting files**:
- Prompt templates: [templates/](templates/)
- Worked examples: [examples/apollo.md](examples/apollo.md), [examples/finance.md](examples/finance.md), [examples/codebase.md](examples/codebase.md)

---

## Table of Contents

1. [Philosophy & Design Principles](#1-philosophy--design-principles)
2. [Complete Pipeline Overview](#2-complete-pipeline-overview)
3. [Document Ingestion](#3-document-ingestion)
4. [Semantic Chunking](#4-semantic-chunking)
5. [Structured Entity Extraction](#5-structured-entity-extraction)
6. [Entity Resolution](#6-entity-resolution)
7. [Graph Assembly](#7-graph-assembly)
8. [Graph Validation](#8-graph-validation)
9. [Entity Summarization](#9-entity-summarization)
10. [Embedding Generation](#10-embedding-generation)
11. [Hybrid GraphRAG Retrieval](#11-hybrid-graphrag-retrieval)
12. [Graph Traversal Algorithms](#12-graph-traversal-algorithms)
13. [Multi-hop Reasoning](#13-multi-hop-reasoning)
14. [Agent Memory Integration](#14-agent-memory-integration)
15. [Incremental Graph Updates](#15-incremental-graph-updates)
16. [Graph Diagnostics](#16-graph-diagnostics)
17. [Evaluation Harness](#17-evaluation-harness)
18. [Performance Optimization](#18-performance-optimization)
19. [Storage: NetworkX → Neo4j](#19-storage-networkx--neo4j)
20. [Production Rules](#20-production-rules)
21. [Common Failure Modes](#21-common-failure-modes)
22. [Example Workflows](#22-example-workflows)
23. [Ready-to-use Prompt Templates](#23-ready-to-use-prompt-templates)
24. [Implementation Checklist](#24-implementation-checklist)

---

## 1. Philosophy & Design Principles

### Core Thesis

Multi-agent systems share a fundamental weakness: each agent's memory dies with its context window. Knowledge graphs are the **infrastructure layer** that provides durable, queryable, provenance-carrying shared state.

### Design Principles

| Principle | Rationale |
|-----------|-----------|
| **Schema is training data** | Pydantic schemas replace labeled NER/relation-classifier training sets |
| **Structured outputs everywhere** | Type-checked contracts between pipeline stages; no JSON parsing at scale |
| **Descriptions are first-class** | One-line entity descriptions enable resolution beyond string similarity |
| **Provenance on every edge** | Grounding requires traceable source document + extraction timestamp |
| **Model tiering** | Haiku for volume (extraction), Sonnet for judgment (resolution, query) |
| **Deterministic where possible** | Blocking, dedup, graph assembly — reserve LLM for judgment |
| **Evaluation or drift** | No gold set = blind prompt changes = production rot |
| **Simple composable patterns** | Prefer prompts + schema over frameworks; add complexity only when measured |

### What the Graph Replaces

- Trained NER models (domain-specific, brittle)
- Trained relation classifiers
- Hand-written entity-resolution heuristics
- Context-window passing between agents

### What the Graph Does NOT Replace

- Judgment (which facts matter, which actions to take)
- Human oversight (sample nodes daily, extend gold sets)
- Domain expertise (schema design, predicate vocabulary)

### RAG vs Knowledge Graph

| Capability | RAG | Knowledge Graph |
|------------|-----|-----------------|
| Single-hop QA | Excellent | Overkill |
| Multi-hop QA | Fails (no chaining) | Core strength |
| Cross-doc linking | Implicit (embedding proximity) | Explicit (entity nodes) |
| Provenance | Chunk-level | Edge-level |
| Agent shared memory | Poor (blob retrieval) | Excellent (structured store) |

**Use both**: RAG for direct retrieval, KG for structural reasoning.

### Decision Framework

```
Need to chain facts across documents?        → Knowledge Graph
Need shared state across agent context windows? → Knowledge Graph
Need evaluator ground truth with citations?  → Knowledge Graph
Need persistent memory across sessions?      → Knowledge Graph
Single-document QA?                          → RAG or direct context
Simple classification/routing?               → Single agent, no graph
```

---

## 2. Complete Pipeline Overview

### Architecture Diagram

```
Documents / Code / APIs
         │
         ▼
┌─────────────────┐
│ 1. INGESTION    │  Parse, normalize, metadata
└────────┬────────┘
         ▼
┌─────────────────┐
│ 2. CHUNKING     │  Semantic boundaries, overlap
└────────┬────────┘
         ▼
┌─────────────────┐
│ 3. EXTRACTION   │  Haiku · structured outputs · entities + SPO triples
└────────┬────────┘
         ▼
┌─────────────────┐
│ 4. RESOLUTION   │  Sonnet · blocking + clustering · alias map
└────────┬────────┘
         ▼
┌─────────────────┐
│ 5. ASSEMBLY     │  NetworkX MultiDiGraph · provenance · node attrs
└────────┬────────┘
         ▼
┌─────────────────┐
│ 6. VALIDATION   │  Connectivity, orphans, schema compliance
└────────┬────────┘
         ▼
┌─────────────────┐
│ 7. SUMMARIZE    │  Sonnet · hub nodes only · entity profiles
└────────┬────────┘
         ▼
┌─────────────────┐
│ 8. EMBED        │  Entity profiles + key_facts → vector index
└────────┬────────┘
         ▼
┌─────────────────┐
│ 9. STORE        │  NetworkX (dev) → Neo4j/Postgres (prod)
└────────┬────────┘
         ▼
┌─────────────────────────────────────────┐
│ QUERY LAYER                              │
│ Vector Search → Graph Expansion →        │
│ Rerank → Subgraph Serialize → LLM Answer │
└─────────────────────────────────────────┘
         │
         ▼
   Grounded Answers (edge citations)
         │
         ▼
   Evaluation Feedback Loop ← gold set F1
```

### Model Selection

| Stage | Model | Rationale |
|-------|-------|-----------|
| Extraction | Haiku | High volume, schema-constrained; cost dominates |
| Resolution | Sonnet | Weighing conflicting evidence |
| Summarization | Sonnet | Cross-document synthesis |
| Querying | Sonnet | Multi-hop reasoning over triples |
| Cypher generation | Sonnet | Complex graph query synthesis |

### Agent Pattern Integration

| Pattern | KG Role | Mechanism |
|---------|---------|-----------|
| Augmented LLM | Retrieval source | Graph traversal replaces vector-only search |
| Prompt chaining | Gate signal | Graph query checks entity conflicts between steps |
| Routing | Classifier input | Entity type + degree routes to specialist |
| Orchestrator–workers | Shared memory | Workers read/write graph; orchestrator stays small |
| Evaluator–optimizer | Grounding layer | Evaluator checks claims against graph edges |

---

## 3. Document Ingestion

### Supported Sources

| Source | Parser | Notes |
|--------|--------|-------|
| Plain text / Markdown | Direct | Preserve section headers as metadata |
| PDF | pdfplumber / PyMuPDF | Extract text + page numbers for provenance |
| HTML | BeautifulSoup / trafilatura | Strip nav/footer; keep article structure |
| DOCX | python-docx | Preserve heading hierarchy |
| Code | tree-sitter / AST | Structural chunking, not token chunking |
| API / JSON | Schema-aware parser | Map records to entity templates |
| Database | SQL export | Table → entity type mapping |

### Normalization Pipeline

```python
@dataclass
class Document:
    id: str                    # stable UUID or content hash
    text: str                  # normalized plain text
    source_uri: str            # file path, URL, or DB ref
    doc_type: str              # "legal", "news", "code", "filing"
    metadata: dict             # author, date, section, page_range
    ingested_at: datetime
    schema_version: str        # ties to extraction schema version

def ingest(raw_source) -> Document:
    text = normalize_whitespace(extract_text(raw_source))
    text = deduplicate_headers(text)
    doc_id = stable_hash(text[:1000] + source_uri)
    return Document(id=doc_id, text=text, ...)
```

### Metadata for Provenance

Every downstream edge must trace to:
- `source_document_id`
- `source_uri` (human-readable)
- `extraction_timestamp`
- `schema_version`
- `chunk_id` (if chunked)

### Deduplication

Before extraction, deduplicate documents by content hash. Duplicate ingestion produces duplicate edges and inflated entity counts.

---

## 4. Semantic Chunking

### Why Not Token Chunking

Naive fixed-token splits break entity–relation co-occurrence. An entity at chunk boundary loses its relational context.

### Chunking Rules

1. **Chunk at semantic boundaries**: section headers, paragraphs, function/class definitions
2. **Overlap one unit**: one paragraph (prose) or one import block (code)
3. **Keep entity + relations together**: if a relation spans sections, overlap must cover both
4. **Attach metadata**: `{doc_id, chunk_index, section_title, page_range}`

### Prose Chunking

```python
def chunk_document(doc: Document, max_tokens: int = 2000) -> list[Chunk]:
    sections = split_on_headers(doc.text)  # ## or numbered sections
    chunks = []
    for section in sections:
        if token_count(section) <= max_tokens:
            chunks.append(Chunk(text=section, ...))
        else:
            paragraphs = section.split("\n\n")
            buffer = []
            for para in paragraphs:
                if token_count(buffer + [para]) > max_tokens:
                    chunks.append(Chunk(text="\n\n".join(buffer), ...))
                    buffer = [buffer[-1], para]  # overlap last paragraph
                else:
                    buffer.append(para)
            if buffer:
                chunks.append(Chunk(text="\n\n".join(buffer), ...))
    return chunks
```

### Per-Document Dedup Before Resolution

After extracting all chunks of one document, run lightweight exact-match dedup on entity names within that document. Cross-document dedup is resolution's job.

### Code Chunking

Chunk by symbol (class, function, method), not token count. See [examples/codebase.md](examples/codebase.md).

---

## 5. Structured Entity Extraction

### Schema Design

The Pydantic schema IS the training data. Design entity types and predicate vocabulary for your domain upfront.

```python
from typing import Literal
from pydantic import BaseModel, Field

EntityType = Literal["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "ARTIFACT"]

class Entity(BaseModel):
    name: str
    type: EntityType
    description: str  # CRITICAL for resolution

class Relation(BaseModel):
    source: str
    predicate: str    # short verb phrase
    target: str

class ExtractedGraph(BaseModel):
    entities: list[Entity]
    relations: list[Relation]
```

### Extraction Prompt

See [templates/extraction_prompt.md](templates/extraction_prompt.md).

Four guidelines, four failure modes:
1. "Central only" → controls recall (precision vs noise)
2. "One-sentence description" → disambiguation signal for resolution
3. "Short verb phrases" → traversable predicates
4. "Every relation connects extracted entities" → prevents orphaned edges

### API Call

```python
def extract(text: str, doc_id: str) -> ExtractedGraph:
    response = client.messages.parse(
        model="claude-haiku-4-5",
        max_tokens=4096,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}],
        output_format=ExtractedGraph,
    )
    return response.parsed_output
```

### Precision/Recall Tuning

| Setting | Effect |
|---------|--------|
| "Central only" (default) | High precision, lower recall — preferred for production |
| "All mentioned entities" | Higher recall, noisier graph |
| Extended entity types | Domain coverage vs schema complexity tradeoff |

Production default: **prefer precision**. False entities spawn false relations that propagate through multi-hop reasoning. Missing entities produce incomplete but correct graphs.

---

## 6. Entity Resolution

### Problem

Raw extraction produces surface-form variants: "NASA" / "National Aeronautics and Space Administration", "Edwin Aldrin" / "Buzz Aldrin". String similarity fails on zero-overlap nicknames.

### Solution: Description-Guided LLM Clustering

Process **one entity type at a time**. Sonnet clusters using name + description + source documents.

See [templates/resolution_prompt.md](templates/resolution_prompt.md).

### Blocking at Scale

Never feed 10,000 entities to one prompt. Block first:

```python
def block_candidates(entities, block_size=75):
    """Cheap deterministic blocking before expensive LLM arbitration."""
    blocks = defaultdict(list)
    for e in entities:
        key = e.name.split()[-1].lower()  # surname / last token
        blocks[key].append(e)
    return [group[i:i+block_size] for group in blocks.values()
            for i in range(0, len(group), block_size)]
```

Hybrid: **deterministic blocking + LLM arbitration within blocks**.

### Alias Map Construction

```python
def build_alias_map(clusters, all_input_names) -> dict[str, str]:
    alias_map = {}
    matched = set()
    for c in clusters:
        for alias in c.aliases:
            alias_map[alias] = c.canonical
            matched.add(alias)
    # FALLBACK: unmatched → single-element cluster (prevents silent loss)
    for name in all_input_names - matched:
        alias_map[name] = name
    return alias_map
```

### Failure Modes

| Mode | Symptom | Fix |
|------|---------|-----|
| Silent loss | Entity vanishes | Fallback single-element clusters |
| Over-merging | "Gemini 12" → "Project Gemini" | Strengthen "genuinely distinct" instruction; spot-check |
| Under-merging | Duplicate nodes | Check descriptions are populated during extraction |

---

## 7. Graph Assembly

### Data Structure

Use `networkx.MultiDiGraph`:
- Multiple edges between same node pair (different predicates)
- Direction matters: `(Armstrong) --[commanded]--> (Apollo 11)` ≠ reverse

### Assembly Code

```python
import networkx as nx

G = nx.MultiDiGraph()

def assemble_graph(raw_relations, alias_map, doc_metadata):
    for rel in raw_relations:
        source = alias_map.get(rel.source, rel.source)
        target = alias_map.get(rel.target, rel.target)
        
        # Upsert nodes
        for name, etype in [(source, rel.source_type), (target, rel.target_type)]:
            if name not in G:
                G.add_node(name, entity_type=etype, source_documents=set(), mention_count=0)
            G.nodes[name]["mention_count"] += 1
            G.nodes[name]["source_documents"].add(rel.source_document_id)
        
        # Add edge with provenance
        G.add_edge(source, target,
            predicate=rel.predicate,
            source_document_id=rel.source_document_id,
            extracted_at=rel.timestamp,
            schema_version=rel.schema_version,
            confidence=rel.confidence,
        )
    return G
```

### Node Attributes

| Attribute | Purpose |
|-----------|---------|
| `entity_type` | Filtering, routing |
| `description` | Initial one-liner from extraction |
| `summary` | Post-summarization profile |
| `key_facts` | Atomic traceable facts |
| `source_documents` | Set of doc IDs |
| `mention_count` | Hub detection |
| `schema_version` | Migration tracking |

### Edge Attributes

| Attribute | Purpose |
|-----------|---------|
| `predicate` | Relation type |
| `source_document_id` | Provenance |
| `extracted_at` | Temporal tracking |
| `confidence` | Corroboration weighting |
| `valid_from` / `valid_to` | Temporal graphs |

---

## 8. Graph Validation

Run after assembly and after every incremental update.

### Structural Checks

```python
def validate_graph(G) -> ValidationReport:
    report = ValidationReport()
    
    # 1. Orphan edges (endpoints missing)
    for s, t, d in G.edges(data=True):
        if s not in G.nodes or t not in G.nodes:
            report.errors.append(f"Orphan edge: ({s}) -> ({t})")
    
    # 2. Self-loops (often extraction errors)
    for n in G.nodes:
        if G.has_edge(n, n):
            report.warnings.append(f"Self-loop: {n}")
    
    # 3. Dangling references in relations
    # (should not happen if extraction constraint enforced)
    
    # 4. Connectivity
    components = list(nx.weakly_connected_components(G))
    report.num_components = len(components)
    report.largest_component_size = max(len(c) for c in components)
    
    # 5. Density
    report.density = G.number_of_edges() / max(G.number_of_nodes(), 1)
    
    # 6. Predicate vocabulary audit
    predicates = {d["predicate"] for _, _, d in G.edges(data=True)}
    report.unique_predicates = len(predicates)
    report.vague_predicates = [p for p in predicates if p in VAGUE_PREDICATES]
    
    return report

VAGUE_PREDICATES = {"related to", "associated with", "involved with", "connected to"}
```

### Schema Compliance

Verify all nodes have required `entity_type`, all edges have `predicate` and `source_document_id`.

### Validation Gates

| Gate | Pass | Fail action |
|------|------|-------------|
| Orphan edges | 0 | Block deployment, fix extraction |
| Components | ≤ expected | Investigate resolution |
| Density | 0.5–3.0 | Tune extraction prompt |
| Vague predicates | <5% of edges | Refine extraction guidelines |

---

## 9. Entity Summarization

Turns a graph of labels into a graph of knowledge. Apply **selectively** to hub nodes.

### Trigger Criteria

- `degree ≥ 3` (mentioned across multiple documents/directions)
- OR `source_document` set changed since last summarization

### Profile Schema

```python
class EntityProfile(BaseModel):
    summary: str              # 2-3 paragraphs
    key_facts: list[str]      # 3-5 atomic, traceable facts
    time_range: TimeRange     # YYYY format
```

See [templates/summarization_prompt.md](templates/summarization_prompt.md).

### Context Assembly

Pool: all source excerpts mentioning entity + graph neighborhood (incoming/outgoing relations).

### Contradiction Resolution

Instruction: "prefer the most specific claim." Specific beats vague. Never invent unsupported facts.

---

## 10. Embedding Generation

Enable hybrid GraphRAG by embedding entity content for vector search seeding.

### What to Embed

| Content | Use |
|---------|-----|
| Entity `summary` | Primary search surface |
| `key_facts` (concatenated) | Fact-level retrieval |
| `description` (pre-summary) | Fallback for low-degree nodes |
| Subgraph serializations | Query-time cache |

### Implementation

```python
def embed_entities(G, embedding_model) -> dict[str, np.ndarray]:
    vectors = {}
    for node, data in G.nodes(data=True):
        text = data.get("summary") or data.get("description", "")
        if data.get("key_facts"):
            text += "\n" + "\n".join(data["key_facts"])
        vectors[node] = embedding_model.encode(text)
    return vectors

def build_vector_index(vectors) -> VectorIndex:
    index = VectorIndex(dimension=len(next(iter(vectors.values()))))
    for entity_id, vec in vectors.items():
        index.add(entity_id, vec)
    return index
```

### Graph Embeddings (Structural)

For link prediction, clustering, and similarity beyond text:

| Algorithm | Type | Use case |
|-----------|------|----------|
| **Node2Vec** | Random walk | Community detection, similarity |
| **GraphSAGE** | Inductive | New nodes without retraining |
| **GAT** | Attention | Weighted neighbor importance |

```python
# Node2Vec example
from node2vec import Node2Vec

node2vec = Node2Vec(G.to_undirected(), dimensions=128, walk_length=30, num_walks=200)
model = node2vec.fit(window=10, min_count=1)
structural_embeddings = {node: model.wv[node] for node in G.nodes}
```

Combine text + structural embeddings for hybrid entity similarity:
```
combined_emb = normalize(α · text_emb + (1-α) · structural_emb)
```

---

## 11. Hybrid GraphRAG Retrieval

### Pipeline

```
Query → Vector Search (seed entities) → Graph Expansion (k-hop)
      → Hybrid Rerank → Subgraph Serialization → LLM Synthesis
```

### Hybrid Scoring Formula

```
Score = α · VectorSim(q, entity) + β · GraphScore(entity, seeds) + γ · Recency(entity)
```

Where:
- `VectorSim`: cosine similarity between query embedding and entity embedding
- `GraphScore`: Personalized PageRank from seed entities, or inverse shortest-path distance
- `Recency`: exponential decay from `extracted_at` or document date
- `α + β + γ = 1` (typical: α=0.5, β=0.35, γ=0.15)

```python
def hybrid_score(query_emb, entity, seeds, G, alpha=0.5, beta=0.35, gamma=0.15):
    vec_sim = cosine(query_emb, entity.embedding)
    graph_score = personalized_pagerank_score(G, entity.name, seeds)
    recency = exp_decay(entity.last_updated, half_life_days=90)
    return alpha * vec_sim + beta * graph_score + gamma * recency
```

### GraphRAG Implementation

```python
def graphrag_query(question: str, G, vector_index, embeddings) -> Answer:
    # Stage 1: Vector search → seed entities
    query_emb = embed(question)
    seeds = vector_index.search(query_emb, top_k=5)
    
    # Stage 2: Graph expansion
    subgraph_nodes = set()
    for seed in seeds:
        subgraph_nodes |= k_hop_neighborhood(G, seed, k=2)
    subgraph = G.subgraph(subgraph_nodes)
    
    # Stage 3: Hybrid rerank
    ranked = sorted(subgraph.nodes, key=lambda n: hybrid_score(
        query_emb, G.nodes[n], seeds, G), reverse=True)
    top_entities = ranked[:20]
    filtered_subgraph = G.subgraph(top_entities)
    
    # Stage 4: Serialize + LLM
    context = serialize_subgraph(filtered_subgraph)
    answer = ask_grounded(question, context)
    
    return Answer(text=answer, subgraph=filtered_subgraph, seeds=seeds)
```

### When to Use GraphRAG vs Pure RAG vs Pure Graph

| Scenario | Approach |
|----------|----------|
| "What is X?" (single fact) | Pure RAG |
| "How is X connected to Y?" (multi-hop) | Pure graph traversal |
| "What do we know about X's ecosystem?" | Hybrid GraphRAG |
| Cross-domain synthesis | Hybrid GraphRAG with k=2+ |

---

## 12. Graph Traversal Algorithms

### Algorithm Selection Decision Tree

```
Start
  │
  ├─ Need shortest path? ──→ Dijkstra (weighted) or BFS (unweighted)
  │
  ├─ Need ALL paths? ──→ DFS with depth limit
  │
  ├─ Need optimal path with heuristic? ──→ A*
  │
  ├─ Need importance/centrality? ──→ PageRank or Personalized PageRank
  │
  ├─ Need community structure? ──→ Louvain / Leiden
  │
  └─ Need bounded exploration? ──→ BFS with k-hop limit
```

### BFS — k-Hop Neighborhood (Default for Querying)

```python
def k_hop_neighborhood(G, center: str, k: int = 2) -> set[str]:
    nodes = {center}
    frontier = {center}
    for _ in range(k):
        nxt = set()
        for n in frontier:
            nxt |= set(G.successors(n))
            nxt |= set(G.predecessors(n))
        frontier = nxt - nodes
        nodes |= frontier
    return nodes
```

Use for subgraph selection. k=2 is the sweet spot for most multi-hop questions.

### DFS — Path Enumeration

```python
def all_paths(G, source, target, max_depth=5):
    return list(nx.all_simple_paths(G, source, target, cutoff=max_depth))
```

Use for "how are X and Y connected?" with path listing.

### Dijkstra — Weighted Shortest Path

```python
def weighted_shortest_path(G, source, target):
    # Weight edges by inverse confidence or recency
    for s, t, d in G.edges(data=True):
        d["weight"] = 1.0 / d.get("confidence", 1.0)
    return nx.dijkstra_path(G, source, target, weight="weight")
```

### A* — Heuristic Path Search

```python
def astar_search(G, source, target, heuristic):
    return nx.astar_path(G, source, target, heuristic=heuristic)
```

Heuristic: embedding distance to target entity.

### Personalized PageRank

```python
def personalized_pagerank(G, seed_entities: list[str], alpha=0.85):
    personalization = {n: 0 for n in G.nodes}
    for s in seed_entities:
        personalization[s] = 1.0 / len(seed_entities)
    return nx.pagerank(G, alpha=alpha, personalization=personalization)
```

Use for ranking entities in GraphRAG reranking stage.

### Community Detection

```python
import community as community_louvain

partition = community_louvain.best_partition(G.to_undirected())
```

Use for: codebase module discovery, topic clustering, graph visualization coloring.

---

## 13. Multi-hop Reasoning

### Mechanism

1. Parse question → identify seed entity (or use vector search)
2. Extract k-hop subgraph
3. Serialize as triples
4. LLM reasons over triples with citation requirement

### Subgraph Serialization

```python
def serialize_subgraph(G_sub, center=None) -> str:
    lines = []
    for s, t, d in G_sub.edges(data=True):
        pred = d.get("predicate", "related_to")
        prov = d.get("source_document_id", "")
        lines.append(f"({s}) --[{pred}]--> ({t})  [source: {prov}]")
    return "\n".join(sorted(set(lines)))
```

### Query Prompt

```
Answer using ONLY the knowledge graph below. Cite the specific edges that support your answer.
If the graph does not contain sufficient information, say so explicitly.

<graph>
{serialized_subgraph}
</graph>

Question: {question}
```

### Grounded vs Ungrounded

| Aspect | Ungrounded | Grounded |
|--------|------------|----------|
| Source | Pretraining | Graph edges only |
| Citations | None | Edge-level |
| Private corpus | Fails/hallucinates | Works |
| Evaluator verification | Subjective | Programmatic string match |

### Return Subgraph with Answer

Production query functions must return `(answer, subgraph_triples, seed_entities)` for transparency and evaluator verification.

---

## 14. Agent Memory Integration

### Knowledge Graph as Shared Memory

```
Orchestrator ──delegates──→ Worker A, B, C
                                │   │   │
                                ▼   ▼   ▼
                           Knowledge Graph
                           (read + write)
                                │
                                ▼
                     Synthesizer queries subgraph
                     (orchestrator context stays small)
```

Workers write entities/relations after processing their slice. Synthesizer reads subgraph — not raw worker outputs.

### Evaluator–Optimizer with Graph Grounding

```
Generator → claims
     ↓
Graph Query → serialize subgraph for mentioned entities
     ↓
Evaluator → check each claim against edges
     ↓
Feedback: "Triple (X, works_at, Y) not found; graph has (X, left, Y) from doc Z"
     ↓
Generator revises → loop until pass or human escalation
```

### Persistent World Model

Graph survives context-window flushes. State file tracks:
- Processed document IDs
- Entities needing re-summarization
- Schema version
- Last validation report

### Multi-Framework Integration

| Framework | Integration pattern |
|-----------|---------------------|
| **LangGraph** | Graph store as ToolNode; agents call `query_graph` / `write_entities` |
| **CrewAI** | Shared memory backend pointing to graph API |
| **AG2** | Tool functions wrapping subgraph query + extraction |
| **Claude orchestration** | Tool use with structured output schemas |

```python
# LangGraph tool example
@tool
def query_knowledge_graph(question: str, seed_entity: str = None) -> str:
    """Query the knowledge graph for multi-hop reasoning with citations."""
    result = graphrag_query(question, G, vector_index, embeddings)
    return json.dumps({"answer": result.text, "triples": result.triples})
```

### Blackboard Architecture

In peer-to-peer multi-agent systems, the graph IS the blackboard. Agents communicate through graph reads/writes, not direct message passing.

---

## 15. Incremental Graph Updates

### Never Rebuild Full Graph on New Documents

```
New document arrives
  → Extract entities/relations
  → Resolve against EXISTING canonical set (not against each other only)
  → Add new edges (dedup by source+target+predicate+doc)
  → Re-summarize entities whose source_document set changed
  → Run validation
  → Update vector index for changed entities
```

### Incremental Resolution

```python
def incremental_resolve(new_entities, existing_alias_map, existing_canonical):
    for entity in new_entities:
        # Check exact match first (free)
        if entity.name in existing_alias_map:
            continue
        # Block against existing canonicals of same type
        candidates = block_against(entity, existing_canonical)
        if candidates:
            cluster = resolve_small_block([entity] + candidates)
            merge_cluster(existing_alias_map, cluster)
        else:
            existing_alias_map[entity.name] = entity.name  # new entity
```

### Summarization Invalidation

```python
def needs_resummary(node_data, new_doc_ids) -> bool:
    old_docs = node_data.get("source_documents", set())
    return bool(new_doc_ids - old_docs) and node_data.get("mention_count", 0) >= 3
```

### Edge Deduplication

Same `(source, target, predicate)` from same document → skip. From different documents → add with separate provenance (corroboration increases confidence).

### Confidence Propagation

```
confidence_edge = 1 - ∏(1 - confidence_i)  for independent sources i
```

Or simpler: `confidence = min(1.0, num_sources * 0.33)`.

---

## 16. Graph Diagnostics

### Key Metrics

| Metric | Formula | Healthy range | Interpretation |
|--------|---------|---------------|----------------|
| **Node count** | \|V\| | Domain-dependent | Growth rate matters more than absolute |
| **Edge count** | \|E\| | Domain-dependent | |
| **Density** | \|E\| / \|V\| | 0.5–2.5 | Below 0.5 = sparse/islands; above 3 = noisy extraction |
| **Connected components** | weakly connected | 1–few large | Many small = resolution failure |
| **Avg path length** | mean shortest path | 2–5 | Higher = disconnected or very deep |
| **Clustering coefficient** | local transitivity | 0.1–0.5 | Low = tree-like; high = clustered |
| **PageRank Gini** | inequality of PR scores | 0.4–0.7 | Identifies hub dominance |
| **Resolution compression** | raw_names / canonical | 1.2–2.5 | Near 1 = consistent naming |
| **Orphan rate** | orphan_edges / total | 0% | Any orphans = pipeline bug |

### Diagnostic Dashboard

```python
def graph_diagnostics(G) -> dict:
    U = G.to_undirected()
    components = list(nx.connected_components(U))
    degrees = [d for _, d in G.degree()]
    
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": G.number_of_edges() / max(G.number_of_nodes(), 1),
        "components": len(components),
        "largest_component_pct": max(len(c) for c in components) / max(G.number_of_nodes(), 1),
        "avg_degree": sum(degrees) / max(len(degrees), 1),
        "max_degree": max(degrees) if degrees else 0,
        "avg_path_length": nx.average_shortest_path_length(U) if nx.is_connected(U) else None,
        "clustering": nx.average_clustering(U),
        "top_hubs": sorted(G.degree, key=lambda x: x[1], reverse=True)[:10],
    }
```

### Production Monitoring (Four Signals)

1. **Extraction rate** per document — sudden drop = domain shift
2. **Resolution compression ratio** — sudden change = naming pattern shift
3. **Graph connectivity** — growing components = missed cross-doc links
4. **Query latency** — pre-compute hub subgraphs if needed

---

## 17. Evaluation Harness

### Gold Set Requirements

- Hand-labeled entities for ≥2 representative documents
- Alias map covering all canonical forms the resolver produces
- Optional: relation gold set on (source, target) pairs (ignore predicate wording)

### Metrics

```python
def score_extraction(predicted_entities, gold_entities, alias_map):
    pred_canonical = {alias_map.get(e.name, e.name) for e in predicted_entities}
    gold_canonical = {alias_map.get(e.name, e.name) for e in gold_entities}
    
    tp = len(pred_canonical & gold_canonical)
    precision = tp / len(pred_canonical) if pred_canonical else 0
    recall = tp / len(gold_canonical) if gold_canonical else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"precision": precision, "recall": recall, "f1": f1}
```

### Feedback Loop

```
Change extraction prompt → rerun pipeline on gold docs → score F1 → compare → document tradeoff
```

This is the same shape as self-improving agentic loops: act, observe, learn, repeat.

### Relation Scoring

Score on `(source, target)` pairs, ignoring predicate wording. "commanded" ≈ "led" ≈ "was commander of".

### Graph-Level Evaluation

| Metric | Measures |
|--------|----------|
| Coverage | % of gold entities reachable in graph |
| Structural recall | % of gold (source, target) pairs present |
| Query accuracy | % of eval questions answered correctly with valid citations |
| Citation validity | % of cited edges that exist in graph |

---

## 18. Performance Optimization

### Cost Structure

| Stage | Scaling | Dominant cost |
|-------|---------|---------------|
| Extraction | Linear with corpus | Haiku tokens |
| Resolution | Sublinear (blocked) | Sonnet calls per type |
| Summarization | Fixed (top-k hubs) | Sonnet large context |
| Querying | Per question | Subgraph size × Sonnet |

For large corpora: extraction dominates. For heavily-queried graphs: querying dominates.

### Cost Optimization Strategies

| Strategy | Savings | Applies to |
|----------|---------|------------|
| **Prompt caching** | ~90% on system/schema | Extraction |
| **Message Batches API** | 50% | Async extraction jobs |
| **Haiku for extraction** | 10-20× vs Sonnet | Extraction |
| **Selective summarization** | degree ≥ 3 only | Summarization |
| **Pre-computed subgraphs** | Avoid repeated traversal | Querying (hot entities) |
| **Blocking before resolution** | Fewer Sonnet calls | Resolution |
| **Embedding cache** | Skip re-embed unchanged nodes | GraphRAG |
| **Extraction cap per run** | Prevent runaway cost | Operations |

### Caching Layers

```python
class GraphCache:
    subgraph_cache: dict[str, str]     # entity → serialized k-hop subgraph
    embedding_cache: dict[str, array]  # entity → vector
    profile_cache: dict[str, Profile]  # entity → profile (invalidated on doc change)
    query_cache: dict[str, Answer]    # (question_hash) → answer (TTL-based)
```

### Parallelization

- Extraction: parallelize per document (embarrassingly parallel)
- Resolution: parallelize per entity type AND per block
- Summarization: parallelize per hub node
- Assembly/validation: single-threaded (fast, in-memory)

---

## 19. Storage: NetworkX → Neo4j

### When to Migrate

| Scale | Storage |
|-------|---------|
| <100K edges | NetworkX in memory |
| 100K–10M edges | Neo4j or Postgres + recursive CTEs |
| >10M edges | Neo4j cluster or Neptune |

Pipeline code (extraction, resolution) does NOT change — only persistence layer.

### Postgres Schema (Minimal)

```sql
CREATE TABLE entities (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    summary TEXT,
    schema_version TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE relations (
    id UUID PRIMARY KEY,
    source_id UUID REFERENCES entities(id),
    target_id UUID REFERENCES entities(id),
    predicate TEXT NOT NULL,
    source_document_id TEXT,
    confidence FLOAT DEFAULT 1.0,
    extracted_at TIMESTAMPTZ,
    valid_from TEXT,
    valid_to TEXT
);

CREATE TABLE aliases (
    entity_id UUID REFERENCES entities(id),
    alias TEXT NOT NULL,
    UNIQUE(alias)
);
```

### Neo4j Migration

```python
def export_to_neo4j(G, driver):
    with driver.session() as session:
        for node, data in G.nodes(data=True):
            session.run("""
                MERGE (n:Entity {name: $name})
                SET n.entity_type = $type, n.summary = $summary,
                    n.mention_count = $mentions
            """, name=node, type=data.get("entity_type"),
                summary=data.get("summary"), mentions=data.get("mention_count"))
        
        for s, t, d in G.edges(data=True):
            session.run("""
                MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
                CREATE (a)-[:REL {
                    predicate: $pred, source_doc: $doc,
                    confidence: $conf, extracted_at: $ts
                }]->(b)
            """, source=s, target=t, pred=d["predicate"],
                doc=d.get("source_document_id"), conf=d.get("confidence", 1.0),
                ts=d.get("extracted_at"))
```

### Neo4j Best Practices

1. **Constraints**: `CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE`
2. **Indexes**: On `entity_type`, `source_document_id`
3. **Relationship properties**: Store predicate as property, not relationship type (avoid schema explosion)
4. **Batch writes**: Use UNWIND for bulk imports (1000 per transaction)
5. **Query parameterization**: Never string-interpolate user input into Cypher

### Cypher Query Generation

For agent-driven graph queries, use LLM to generate Cypher with schema context:

```
Given this graph schema:
- Nodes: Entity(name, entity_type, summary)
- Edges: REL(predicate, source_doc, confidence)

Generate a Cypher query to answer: {question}

Rules:
- Use parameters for user-provided values
- LIMIT results to 50
- Return nodes AND relationships for visualization
```

### Schema Evolution and Versioning

```python
@dataclass
class SchemaVersion:
    version: str           # "1.2.0"
    entity_types: list[str]
    predicate_vocabulary: list[str]
    created_at: datetime
    migration_notes: str
```

When schema changes:
1. Bump version
2. Tag new extractions with new version
3. Optionally re-extract old documents
4. Never merge entities from incompatible schema versions without migration

---

## 20. Production Rules

### Operational Discipline

1. **Sample the graph daily**: Pick random node, verify edges against source docs
2. **Cap extraction volume per run**: Prevent unbounded cost from ingestion errors
3. **Version the schema**: Distinguish entities from different prompt versions
4. **Human escalation path**: Claims absent from graph → escalate, don't silently accept/reject

### Provenance Requirements

Every edge MUST carry:
- `source_document_id`
- `extracted_at`
- `schema_version`
- `confidence` (default 1.0, boost on corroboration)

### Security

- Sanitize document content before extraction (prompt injection defense)
- Parameterize all graph queries (Cypher injection defense)
- Access control on graph write operations
- Audit log for agent writes to graph

### Deployment Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Ingestion   │────→│ Extraction   │────→│ Resolution  │
│ Service     │     │ Workers (N)  │     │ Service     │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│ Query API   │←────│ Graph Store  │←────│ Assembly +  │
│ (GraphRAG)  │     │ (Neo4j)      │     │ Validation  │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │
       ▼                    ▼
┌─────────────┐     ┌──────────────┐
│ Vector Index│     │ Evaluation   │
│ (embeddings)│     │ Harness      │
└─────────────┘     └──────────────┘
```

---

## 21. Common Failure Modes

| # | Failure | Symptom | Root cause | Fix |
|---|---------|---------|------------|-----|
| 1 | Silent entity loss | Node count drops after resolution | Unmatched names not in clusters | Fallback single-element clusters |
| 2 | Orphaned edges | Edge references missing node | Extraction constraint violated | Enforce "relation connects extracted entities" |
| 3 | Over-merging | Distinct entities collapsed | Surface-form-only clustering | Require descriptions; per-type resolution |
| 4 | Fragmented graph | Many disconnected components | Resolution too conservative | Review blocking; check cross-doc entities |
| 5 | Vague predicates | Unusable for reasoning | Prompt allows "related to" | Constrain predicate vocabulary |
| 6 | Hallucinated summaries | Facts not in source docs | Missing "do not invent" instruction | Strengthen summarization guardrails |
| 7 | Context overflow | Query fails on large subgraph | k too high | Reduce k; filter/rerank before serialize |
| 8 | Schema drift | Incompatible entity types | Unversioned schema changes | Version schema; tag extractions |
| 9 | Duplicate edges | Inflated confidence | No dedup on incremental update | Dedup by (source, target, predicate, doc) |
| 10 | Evaluation artifacts | Recall appears worse than reality | Gold set missing canonical aliases | Extend alias map in scorer |
| 11 | Prompt injection | Adversarial entities extracted | Untrusted document content | Sanitize; separate system/user content |
| 12 | Cost runaway | Unexpected API bills | No extraction cap | Per-run document/entity limits |
| 13 | Stale embeddings | GraphRAG returns outdated info | No re-embed on update | Invalidate embedding cache on node change |
| 14 | Over-extraction | Noisy graph, low precision | "All entities" prompt on large corpus | Switch to "central only" |
| 15 | Under-extraction | Missing critical entities | Too aggressive filtering | Loosen central-only; check gold set recall |

---

## 22. Example Workflows

Detailed walkthroughs in separate files:

| Example | Domain | Key lesson |
|---------|--------|------------|
| [apollo.md](examples/apollo.md) | Historical documents | Baseline pipeline, resolution, grounded query |
| [finance.md](examples/finance.md) | Competitive intelligence | Multi-agent shared memory, cross-domain synthesis |
| [codebase.md](examples/codebase.md) | Software repository | AST + LLM hybrid, impact analysis, incremental updates |

---

## 23. Ready-to-use Prompt Templates

| Template | Stage | Model |
|----------|-------|-------|
| [extraction_prompt.md](templates/extraction_prompt.md) | Entity + relation extraction | Haiku |
| [resolution_prompt.md](templates/resolution_prompt.md) | Entity clustering | Sonnet |
| [summarization_prompt.md](templates/summarization_prompt.md) | Hub node profiles | Sonnet |

### Query Prompt (inline)

```
Answer using ONLY the knowledge graph below. Cite the specific edges that support your answer.
If the graph does not contain sufficient information, state what is missing.

<graph>
{serialized_subgraph}
</graph>

Question: {question}
```

### Cypher Generation Prompt (inline)

```
You are a graph query expert. Given this Neo4j schema:
- (:Entity {name, entity_type, summary})
- (:Entity)-[:REL {predicate, source_doc, confidence}]->(:Entity)

Write a parameterized Cypher query for: {question}
Return entity names, relationship predicates, and source_doc. LIMIT 50.
```

---

## 24. Implementation Checklist

### Phase 1: Foundation

- [ ] Define Pydantic extraction schema (entity types, relation structure)
- [ ] Define predicate vocabulary for domain
- [ ] Implement document ingestion + normalization
- [ ] Implement semantic chunking with overlap
- [ ] Implement extraction with structured outputs (Haiku)
- [ ] Create gold evaluation set (≥2 documents)
- [ ] Implement entity/resolution (Sonnet + blocking)
- [ ] Implement alias map with fallback for unmatched names
- [ ] Implement graph assembly (NetworkX MultiDiGraph)
- [ ] Implement subgraph serialization
- [ ] Implement grounded querying (Sonnet)

### Phase 2: Quality

- [ ] Implement validation checks (orphans, connectivity, density)
- [ ] Implement evaluation harness (precision, recall, F1)
- [ ] Run evaluation feedback loop (prompt tuning)
- [ ] Implement entity summarization for hub nodes (degree ≥ 3)
- [ ] Add provenance to all edges (doc_id, timestamp, schema_version)
- [ ] Version the extraction schema
- [ ] Set extraction cap per run

### Phase 3: Retrieval & Reasoning

- [ ] Generate embeddings for entity profiles
- [ ] Build vector index
- [ ] Implement k-hop subgraph extraction (BFS)
- [ ] Implement hybrid scoring (VectorSim + GraphScore + Recency)
- [ ] Implement GraphRAG pipeline (vector → expand → rerank → LLM)
- [ ] Implement multi-hop query with edge citations
- [ ] Return subgraph alongside answers

### Phase 4: Production

- [ ] Implement incremental update (no full rebuild)
- [ ] Implement summarization invalidation on doc change
- [ ] Implement graph diagnostics dashboard
- [ ] Set up monitoring (extraction rate, compression ratio, connectivity, latency)
- [ ] Migrate to Neo4j or Postgres (if >100K edges)
- [ ] Implement caching (subgraphs, embeddings, query results)
- [ ] Integrate with agent framework (LangGraph/CrewAI/AG2)
- [ ] Implement evaluator–optimizer graph grounding
- [ ] Daily human sampling of random node
- [ ] Document schema version migration process

### Phase 5: Advanced (Optional)

- [ ] Temporal edges (valid_from / valid_to)
- [ ] Confidence propagation across corroborating sources
- [ ] Graph embeddings (Node2Vec / GraphSAGE)
- [ ] Community detection for domain segmentation
- [ ] Personalized PageRank for GraphRAG reranking
- [ ] Cypher query generation for Neo4j
- [ ] Graph-of-graphs for multi-team federated graphs
- [ ] A*/Dijkstra for weighted path queries

---

## Appendix A: Temporal Knowledge Graphs

Extend relations with validity periods:

```python
class TemporalRelation(BaseModel):
    source: str
    predicate: str
    target: str
    valid_from: str | None = None  # YYYY-MM
    valid_to: str | None = None    # YYYY-MM or "ongoing"
```

Query: filter subgraph by time window before reasoning.

```
"Who was CEO of Acme Corp in Q3 2024?"
→ filter edges where predicate='held_role' AND valid_from <= '2024-09' AND valid_to >= '2024-07'
→ reason over filtered subgraph
```

EntityProfile `TimeRange` already supports entity-level temporality; extend to edges for full temporal graphs.

---

## Appendix B: Confidence Propagation

```python
def edge_confidence(edges_for_fact: list[dict]) -> float:
    """Multiple independent extractions corroborate a fact."""
    if len(edges_for_fact) == 1:
        return edges_for_fact[0].get("confidence", 0.7)
    # Independent corroboration: 1 - product(1 - ci)
    conf = 1.0
    for e in edges_for_fact:
        conf *= (1 - e.get("confidence", 0.7))
    return round(1 - conf, 3)
```

Evaluator weights: high-confidence edges are binding; low-confidence edges flagged for human review.

---

## Appendix C: Complete Query Implementation

```python
def serialize_subgraph(G, center: str, hops: int = 2) -> str:
    nodes = k_hop_neighborhood(G, center, hops)
    sub = G.subgraph(nodes)
    lines = [
        f"({s}) --[{d['predicate']}]--> ({t})  [source: {d.get('source_document_id', 'unknown')}]"
        for s, t, d in sub.edges(data=True)
    ]
    return "\n".join(sorted(set(lines)))

def ask(question: str, graph_context: str | None = None,
        return_subgraph: bool = True) -> dict:
    if graph_context is not None:
        prompt = f"""Answer using ONLY the knowledge graph below.
Cite the specific edges that support your answer.

<graph>
{graph_context}
</graph>

Question: {question}"""
    else:
        prompt = question
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = next(b.text for b in response.content if b.type == "text")
    return {"answer": answer, "graph_context": graph_context, "grounded": graph_context is not None}
```

---

## Appendix D: Glossary

| Term | Definition |
|------|------------|
| Knowledge graph | Entities (nodes) + typed relations (edges) with provenance |
| Structured outputs | API feature: response validates against Pydantic schema |
| Entity resolution | Merging surface-form variants into canonical nodes |
| Multi-hop reasoning | Answering questions requiring chained facts across edges |
| Subgraph serialization | Converting graph portion to text triples for LLM context |
| Ground truth | Verdict from environment (graph edge) not model estimation |
| Blocking | Grouping resolution candidates by cheap signals before LLM |
| Hub node | High-degree entity tying multiple documents together |
| Alias map | Dictionary: surface form → canonical name |
| Provenance | Source document + extraction context for a triple |
| GraphRAG | Hybrid retrieval: vector search + graph expansion + LLM synthesis |
| Personalized PageRank | PageRank biased toward seed entities from query |

---

## Appendix E: Failure Recovery and Validation Workflows

### Pipeline Failure Recovery

| Failure point | Detection | Recovery |
|---------------|-----------|----------|
| Extraction API error | Structured output validation failure | Retry with backoff; quarantine document |
| Partial extraction | Entity count = 0 | Log + skip; flag for manual review |
| Resolution timeout | Block too large | Split block in half; retry |
| Assembly conflict | Duplicate node attributes | Last-write-wins with merge strategy |
| Neo4j write failure | Transaction rollback | Retry batch; dead-letter queue |
| Embedding service down | Cache miss + API error | Serve graph-only retrieval (β=1.0, α=0) |

### Recovery State Machine

```
Document → [Extracting] → success → [Resolving]
                │ failure (retry ≤3)
                ▼
           [Quarantined] → manual review → re-ingest
```

### Validation Workflow (Post-Build)

```python
def post_build_validation(G, gold_set, config) -> bool:
    """Run all validation gates. Returns True if production-ready."""
    report = validate_graph(G)
    assert report.errors == [], f"Structural errors: {report.errors}"
    
    diagnostics = graph_diagnostics(G)
    assert diagnostics["density"] >= config.min_density, "Graph too sparse"
    assert diagnostics["largest_component_pct"] >= 0.8, "Too fragmented"
    
    if gold_set:
        scores = score_extraction(get_entities(G), gold_set.entities, gold_set.alias_map)
        assert scores["precision"] >= config.min_precision, f"Precision {scores['precision']} below threshold"
    
    assert all(d.get("source_document_id") for _, _, d in G.edges(data=True)), "Missing provenance"
    return True
```

### Rollback Strategy

Version graphs alongside schema versions:

```python
def snapshot_graph(G, version: str):
    nx.write_gpickle(G, f"graph_snapshots/graph_{version}.gpickle")
    export_metadata(version, G.number_of_nodes(), G.number_of_edges())

def rollback_to(version: str) -> nx.MultiDiGraph:
    return nx.read_gpickle(f"graph_snapshots/graph_{version}.gpickle")
```

---

## Appendix F: Multi-Agent Orchestration Patterns (Detailed)

### Pattern 1: Orchestrator–Workers with Graph Blackboard

```python
class GraphBlackboard:
    def __init__(self, G: nx.MultiDiGraph):
        self.G = G
        self.lock = threading.Lock()
    
    def write_extraction(self, worker_id: str, extracted: ExtractedGraph):
        with self.lock:
            for entity in extracted.entities:
                self._upsert_entity(entity, worker_id)
            for rel in extracted.relations:
                self._add_edge(rel, worker_id)
    
    def read_subgraph(self, entity: str, hops: int = 2) -> str:
        nodes = k_hop_neighborhood(self.G, entity, hops)
        return serialize_subgraph(self.G.subgraph(nodes))

# Orchestrator workflow
async def competitive_intelligence(competitor: str):
    blackboard = GraphBlackboard(nx.MultiDiGraph())
    
    # Parallel worker extraction
    workers = [pricing_agent, product_agent, financial_agent, marketing_agent]
    results = await asyncio.gather(*[w.extract(competitor) for w in workers])
    for worker, result in zip(workers, results):
        blackboard.write_extraction(worker.id, result)
    
    # Cross-worker resolution (sequential — needs all entities)
    resolve_and_merge(blackboard.G)
    
    # Strategic synthesis via graph query (not raw worker outputs)
    context = blackboard.read_subgraph(competitor, hops=2)
    return synthesize(f"Analyze {competitor} market position", context)
```

### Pattern 2: Evaluator–Optimizer with Graph Fact-Checking

```python
def evaluate_claims(generated_text: str, G: nx.MultiDiGraph) -> list[Feedback]:
    mentioned_entities = extract_entity_mentions(generated_text)
    subgraph = serialize_subgraph(G.subgraph(
        k_hop_neighborhood(G, mentioned_entities[0], 2)))
    
    response = client.messages.parse(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": f"""
            Check each factual claim in the generated text against the knowledge graph.
            For each unsupported or contradicted claim, provide specific graph evidence.
            
            <generated>{generated_text}</generated>
            <graph>{subgraph}</graph>
        """}],
        output_format=EvaluationResult,
    )
    return response.parsed_output.feedback_items
```

### Pattern 3: Routing via Graph Metadata

Route queries without LLM call when entity type and degree determine the specialist:

```python
def route_query(question: str, G: nx.MultiDiGraph) -> str:
    seeds = extract_seed_entities(question, G)
    if not seeds:
        return "general_agent"
    
    seed = seeds[0]
    etype = G.nodes[seed].get("entity_type")
    degree = G.degree(seed)
    
    routing_table = {
        ("PERSON", "high"): "biography_agent",
        ("ORGANIZATION", "high"): "company_agent",
        ("PRICING", "any"): "pricing_agent",
        ("PATENT", "any"): "ip_agent",
    }
    degree_class = "high" if degree >= 5 else "low"
    return routing_table.get((etype, degree_class), routing_table.get((etype, "any"), "general_agent"))
```

### Pattern 4: Prompt Chaining with Graph Gates

```python
def chain_with_graph_gate(step1_output, G):
    """Between chain steps, verify new entities don't conflict with existing graph."""
    new_entities = step1_output.entities
    conflicts = []
    for entity in new_entities:
        if entity.name in G.nodes:
            existing = G.nodes[entity.name]
            if existing.get("entity_type") != entity.type:
                conflicts.append(f"Type conflict: {entity.name} is {existing['entity_type']}, new says {entity.type}")
    if conflicts:
        return {"proceed": False, "conflicts": conflicts}
    return {"proceed": True}
```

### LangGraph Integration (Full Example)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    question: str
    seed_entities: list[str]
    subgraph: str
    answer: str

def vector_search_node(state: AgentState) -> AgentState:
    seeds = vector_index.search(embed(state["question"]), top_k=5)
    return {**state, "seed_entities": seeds}

def graph_expand_node(state: AgentState) -> AgentState:
    nodes = set()
    for seed in state["seed_entities"]:
        nodes |= k_hop_neighborhood(G, seed, k=2)
    subgraph = serialize_subgraph(G.subgraph(nodes))
    return {**state, "subgraph": subgraph}

def synthesize_node(state: AgentState) -> AgentState:
    answer = ask(state["question"], state["subgraph"])
    return {**state, "answer": answer}

workflow = StateGraph(AgentState)
workflow.add_node("vector_search", vector_search_node)
workflow.add_node("graph_expand", graph_expand_node)
workflow.add_node("synthesize", synthesize_node)
workflow.set_entry_point("vector_search")
workflow.add_edge("vector_search", "graph_expand")
workflow.add_edge("graph_expand", "synthesize")
workflow.add_edge("synthesize", END)
graph_agent = workflow.compile()
```

---

## Appendix G: Graph Evaluation Metrics (Detailed)

### Structural Metrics

```python
def compute_structural_metrics(G) -> dict:
    U = G.to_undirected()
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    metrics = {
        "order": n,
        "size": m,
        "density": m / (n * (n - 1)) if n > 1 else 0,  # directed density
        "avg_degree": sum(d for _, d in G.degree()) / max(n, 1),
    }
    
    if n > 0:
        pr = nx.pagerank(G)
        metrics["pagerank_max"] = max(pr.values())
        metrics["pagerank_gini"] = gini_coefficient(list(pr.values()))
    
    if nx.is_connected(U):
        metrics["avg_shortest_path"] = nx.average_shortest_path_length(U)
        metrics["diameter"] = nx.diameter(U)
    
    metrics["clustering_coefficient"] = nx.average_clustering(U)
    metrics["num_components"] = nx.number_connected_components(U)
    
    # Degree distribution shape
    degrees = [d for _, d in G.degree()]
    metrics["degree_std"] = np.std(degrees)
    metrics["max_degree"] = max(degrees) if degrees else 0
    
    return metrics

def gini_coefficient(values: list[float]) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumulative = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2 * cumulative) / (n * sum(sorted_vals)) - (n + 1) / n
```

### Quality Metrics (Against Gold Set)

| Metric | Formula | Target |
|--------|---------|--------|
| Entity precision | TP / (TP + FP) | ≥ 0.90 |
| Entity recall | TP / (TP + FN) | ≥ 0.50 (domain-dependent) |
| Entity F1 | 2PR / (P + R) | ≥ 0.65 |
| Relation structural recall | matched (s,t) pairs / gold pairs | ≥ 0.60 |
| Query answer accuracy | correct answers / eval questions | ≥ 0.80 |
| Citation validity | valid citations / total citations | ≥ 0.95 |
| Provenance coverage | edges with doc_id / total edges | 100% |

### Coverage Metrics

```python
def coverage_metrics(G, corpus_entity_mentions: dict) -> dict:
    """What fraction of known entity mentions appear in the graph?"""
    graph_entities = set(G.nodes)
    all_mentions = set(corpus_entity_mentions.keys())
    return {
        "entity_coverage": len(graph_entities & all_mentions) / max(len(all_mentions), 1),
        "relation_coverage": compute_relation_coverage(G, corpus_entity_mentions),
    }
```

---

## Appendix H: Anti-Patterns

| Anti-pattern | Why it fails | Do instead |
|--------------|--------------|------------|
| **Mega-prompt extraction** | One prompt for NER + relations + summarization | Separate stages with structured outputs |
| **String-only resolution** | Misses nicknames, abbreviations, cross-lingual | Description-guided LLM clustering |
| **Rebuild on every update** | O(corpus) cost per new document | Incremental extract + resolve against existing |
| **Summarize everything** | Sonnet cost scales with entity count | Hub nodes only (degree ≥ 3) |
| **k=5 subgraphs** | Context overflow, noise | k=2 default; increase only when needed |
| **No provenance** | Ungrounded answers, evaluator blind | source_document_id on every edge |
| **Vague predicates** | Graph unusable for reasoning | Constrained verb phrases |
| **No gold set** | Blind prompt tuning | Hand-label ≥2 docs; run F1 loop |
| **Context-window as agent memory** | Linear growth, coherence loss | Graph as shared blackboard |
| **Pure vector RAG for multi-hop** | Cannot chain cross-doc facts | GraphRAG or pure graph traversal |
| **Unversioned schema changes** | Incompatible entity types mixed | Version schema; tag extractions |
| **Trust ungrounded answers on private data** | Hallucination on unknown corpus | Require edge citations |
| **Skip validation gates** | Silent corruption in production | validate_graph() before every deploy |
| **No extraction cap** | Runaway cost on bad ingestion | Per-run document/entity limits |
| **Graph without evaluation loop** | Slow drift as corpus evolves | Weekly F1 check against gold set |

---

## Appendix I: Graph Schema Design Guide

### Entity Type Design

Start minimal, extend as needed:

```python
# Phase 1: Universal types
EntityType = Literal["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "ARTIFACT"]

# Phase 2: Domain extension (finance)
EntityType = Literal[..., "PRODUCT", "PRICING", "PATENT", "FILING", "METRIC"]

# Phase 3: Code domain
EntityType = Literal[..., "MODULE", "CLASS", "FUNCTION", "API_ENDPOINT"]
```

Rules:
- ≤15 entity types per domain (LLM extraction accuracy degrades beyond this)
- Each type must be distinguishable in descriptions
- Document each type with 2-3 examples in the extraction prompt

### Predicate Vocabulary Curation

Maintain an allowed predicate list:

```python
ALLOWED_PREDICATES = {
    "PERSON": ["works_at", "founded", "leads", "reports_to", "born_in", "educated_at"],
    "ORGANIZATION": ["acquired", "partnered_with", "located_in", "subsidiary_of", "competes_with"],
    "ARTIFACT": ["created_by", "used_in", "part_of", "version_of"],
    # ...
}
```

Reject vague predicates in validation: `related_to`, `associated_with`, `involved_with`, `connected_to`.

### Schema Migration Checklist

When adding entity types or predicates:
1. Bump `schema_version` (semver)
2. Update extraction prompt with new type definitions + examples
3. Update gold set to cover new types
4. Run evaluation — compare F1 before/after
5. Tag all new extractions with new version
6. Do NOT merge entities across incompatible schema versions
7. Optionally re-extract sample of old docs to measure delta

---

## Appendix J: Cost Estimation Worksheet

### Extraction Cost

```
cost_extract = num_documents × avg_doc_tokens × haiku_price_per_token
# With prompt caching:
cost_extract_cached = num_documents × avg_doc_tokens × haiku_price × (1 - cache_discount)
# With batch API (50% off):
cost_extract_batch = cost_extract × 0.5
```

Example: 10,000 docs × 2,000 tokens × $0.25/M = ~$5.00 (Haiku, no cache).

### Resolution Cost

```
num_blocks = sum(ceil(count_per_type / block_size) for each entity_type)
cost_resolve = num_blocks × avg_block_tokens × sonnet_price_per_token
```

Resolution is sublinear — one pass per entity type, blocked.

### Summarization Cost

```
hub_nodes = count(degree >= 3)
cost_summarize = hub_nodes × avg_context_tokens × sonnet_price_per_token
```

Typically 10-50 hub nodes regardless of corpus size.

### Query Cost

```
cost_query = num_queries × avg_subgraph_tokens × sonnet_price_per_token
```

Dominant for heavily-queried graphs. Pre-compute hub subgraphs to reduce tokens.

### Total Pipeline Cost (Typical)

| Corpus size | Extract | Resolve | Summarize | Total build |
|-------------|---------|---------|-----------|-------------|
| 100 docs | $0.05 | $0.10 | $0.05 | ~$0.20 |
| 1K docs | $0.50 | $0.50 | $0.10 | ~$1.10 |
| 10K docs | $5.00 | $2.00 | $0.20 | ~$7.20 |
| 100K docs | $50.00 | $15.00 | $0.50 | ~$65.50 |

Query costs are per-question and independent of corpus size (depends on subgraph size).

---

*Based on Anthropic's Knowledge Graph Cookbook, Building Effective AI Agents, and Claude API documentation. Independently compiled engineering specification.*
