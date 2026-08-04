# Example: Codebase Knowledge Graph

Map a software repository into a queryable graph: modules, functions, classes, dependencies, ownership, and cross-file relationships.

## Schema for Codebases

```python
EntityType = Literal[
    "MODULE", "CLASS", "FUNCTION", "METHOD", "INTERFACE",
    "FILE", "PACKAGE", "SERVICE", "API_ENDPOINT", "DATABASE",
    "PERSON", "TEAM", "CONCEPT"
]

# Relations
# (UserService) --[imports]--> (AuthModule)
# (create_user) --[defined_in]--> (user_service.py)
# (UserService) --[calls]--> (validate_token)
# (auth/router.py) --[owned_by]--> (Platform Team)
# (POST /users) --[handled_by]--> (create_user)
```

## Ingestion Pipeline

```
Source files → AST parse → Symbol extraction → Dependency edges → Graph
     │              │              │                  │
     │         tree-sitter    per-file entities    import/call graph
     │              │              │                  │
     └──────── Semantic docs (docstrings, README) ────┘
                    │
              LLM enrichment (optional)
              descriptions + conceptual relations
```

## Chunking Strategy

Unlike prose documents, code chunks by **structural boundaries**:

| Boundary | Rationale |
|----------|-----------|
| File | Natural module scope |
| Class/method | Preserves symbol + its relations |
| Directory | Package-level aggregation |

**Do not** chunk by token count — splits function from its callers.

```python
def chunk_codebase(repo_path: str) -> list[CodeChunk]:
    chunks = []
    for path in glob("**/*.py", root=repo_path):
        tree = parse_ast(path)
        for symbol in extract_symbols(tree):
            chunks.append(CodeChunk(
                id=f"{path}::{symbol.name}",
                content=symbol.source + symbol.docstring,
                symbol_type=symbol.kind,
                file_path=path,
            ))
    return chunks
```

## Extraction Prompt Adaptation

```
Extract a knowledge graph from this code artifact.

<artifact>
File: {file_path}
Symbol: {symbol_name} ({symbol_type})
{source_code_and_docstring}
</artifact>

Guidelines:
- Extract MODULE, CLASS, FUNCTION, METHOD entities central to this artifact.
- Relations: imports, calls, inherits, implements, defined_in, owned_by.
- Descriptions: one sentence on purpose/responsibility from docstrings and code.
- Skip standard library imports unless architecturally significant.
- Every relation connects two extracted entities.
```

## Deterministic + LLM Hybrid

Use AST for **structural facts** (imports, calls, inheritance). Use LLM for **semantic facts** (conceptual grouping, architectural role, team ownership from CODEOWNERS).

```python
def build_codebase_graph(repo):
    G = nx.MultiDiGraph()
    
    # Deterministic layer (fast, exact)
    for edge in ast_import_graph(repo):
        G.add_edge(edge.source, edge.target, predicate="imports", provenance="ast")
    for edge in call_graph(repo):
        G.add_edge(edge.caller, edge.callee, predicate="calls", provenance="ast")
    
    # LLM layer (semantic enrichment)
    for chunk in semantic_chunks(repo):
        extracted = extract(chunk)
        merge_into_graph(G, extracted, provenance="llm")
    
    return G
```

## Query Patterns

### Impact Analysis
```
Question: "What breaks if I change AuthModule.validate_token?"
Traversal: BFS reverse from validate_token, k=unlimited within repo
```

### Ownership Lookup
```
Question: "Who owns the payment processing flow?"
Path: (PaymentService) --[calls*]--> (*) filtered by owned_by edges
```

### Architecture Discovery
```
Question: "Show me all services that depend on the database layer"
Cypher: MATCH (s:SERVICE)-[:calls|imports*1..3]->(d:DATABASE) RETURN s, d
```

## GraphRAG for Code Questions

```python
def code_rag_query(question: str, G, embeddings):
    # 1. Vector search on function/class descriptions
    seeds = vector_search(question, embeddings, top_k=10)
    
    # 2. Graph expansion along call/import edges
    subgraph = expand(G, seeds, hops=2, edge_types=["calls", "imports", "defined_in"])
    
    # 3. Rerank by graph centrality (PageRank of seeds in subgraph)
    ranked = hybrid_rerank(subgraph, question, alpha=0.4, beta=0.45, gamma=0.15)
    
    # 4. LLM answer with file:line citations
    return synthesize(question, ranked, cite_format="file:line")
```

## Incremental Updates

On git push:
1. Diff changed files
2. Re-extract only affected symbols
3. Resolve against existing canonical set
4. Remove stale edges for deleted symbols
5. Re-summarize modules whose dependency set changed materially

```python
def incremental_update(G, git_diff):
    affected = {f for f in git_diff.changed_files if f.endswith('.py')}
    for path in affected:
        remove_edges_for_file(G, path)
        for chunk in chunk_file(path):
            merge_extraction(G, extract(chunk))
    re_summarize_changed_hubs(G, affected)
```

## Diagnostics for Code Graphs

| Metric | Healthy | Problem |
|--------|---------|---------|
| Orphan nodes | <5% | Missing import resolution |
| Cyclic import clusters | Documented | Hidden circular deps |
| Avg path length (module→module) | 2-4 | Over-coupled or under-connected |
| Hub modules | 3-8 high-degree | God modules need refactoring |
| Coverage | >90% of .py files | Incomplete ingestion |

## Neo4j Migration

```cypher
// Create constraints
CREATE CONSTRAINT module_name IF NOT EXISTS FOR (m:MODULE) REQUIRE m.name IS UNIQUE;
CREATE CONSTRAINT function_id IF NOT EXISTS FOR (f:FUNCTION) REQUIRE f.id IS UNIQUE;

// Load from NetworkX export
LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
MATCH (s {name: row.source}), (t {name: row.target})
CREATE (s)-[:REL {predicate: row.predicate, file: row.file}]->(t);

// Impact query
MATCH (f:FUNCTION {name: 'validate_token'})<-[:calls*1..5]-(caller)
RETURN caller.name, caller.file_path;
```

## Multi-Agent Code Review Integration

| Agent | Graph role |
|-------|------------|
| Security scanner | Writes (vuln) --[found_in]--> (function) edges |
| Reviewer | Reads call graph to assess blast radius |
| Evaluator | Checks claims against graph edges, not estimation |
| Architect | Queries community detection for module boundaries |

The graph is the **shared blackboard** — agents read/write without passing full file contents through the orchestrator.
