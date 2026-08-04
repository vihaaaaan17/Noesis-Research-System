# Example: Apollo Corpus Knowledge Graph

Reference implementation from Anthropic's Knowledge Graph Cookbook. Six Wikipedia summaries → 22 canonical nodes, 34 edges, 1 connected component.

## Corpus

| Document | Entities | Relations |
|----------|----------|-----------|
| Apollo program | 8 | 7 |
| Apollo 11 | 6 | 5 |
| Neil Armstrong | 3 | 2 |
| Saturn V | 5 | 4 |
| Buzz Aldrin | 6 | 6 |
| Kennedy Space Center | 8 | 10 |

**Total raw**: 36 entities, 34 relations → **22 canonical** after resolution.

## Resolution Wins

| Surface forms | Canonical |
|---------------|-----------|
| "Edwin Aldrin", "Buzz Aldrin" | Buzz Aldrin |
| "Neil Armstrong", "Neil Alden Armstrong" | Neil Alden Armstrong |
| "NASA", "National Aeronautics..." | NASA |

String similarity fails on Aldrin variants; description-based LLM clustering succeeds.

## Graph Structure

```
Hub nodes (degree 9): Apollo program, Apollo 11
Density: 34/22 = 1.55 (healthy)
Components: 1 (resolution worked)
```

## Sample Triples

```
(Neil Alden Armstrong) --[commanded]--> (Apollo 11)
(Apollo 11) --[landed on]--> (Moon)
(Apollo 11) --[launched from]--> (Kennedy Space Center)
(Saturn V) --[launched]--> (Apollo 11)
(Apollo program) --[used]--> (Saturn V)
```

## Multi-Hop Query Example

**Question**: "Which locations are connected to people who flew on Apollo 11?"

**Subgraph (k=2 from Apollo 11)**:
```
(Apollo 11) --[commanded by]--> (Neil Alden Armstrong)
(Apollo 11) --[landed on]--> (Moon)
(Neil Alden Armstrong) --[walked on]--> (Moon)
```

**Grounded answer**: "The only person-location relationship supported by the graph is Neil Armstrong → walked on → the Moon."

**Ungrounded answer** (without graph): Lists birthplaces, universities, military bases from pretraining — plausible but untraceable.

## Evaluation Results (Gold Set)

| Document | Raw F1 | Precision | Recall | Resolved R |
|----------|--------|-----------|--------|------------|
| Apollo 11 | 0.71 | 1.00 | 0.55 | 0.55 |
| Neil Armstrong | 0.55 | 1.00 | 0.38 | 0.38 |

Perfect precision (1.00) = conservative extractor. Lower recall = "central only" instruction filters peripheral mentions like "Purdue University."

## Pipeline Code Sketch

```python
import networkx as nx
from anthropic import Anthropic

client = Anthropic()
G = nx.MultiDiGraph()

# 1. Extract per document
for doc in apollo_corpus:
    graph = extract(doc.text, doc.id)
    raw_entities.extend(graph.entities)
    raw_relations.extend(graph.relations)

# 2. Resolve per entity type
for etype in ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "ARTIFACT"]:
    clusters = resolve([e for e in raw_entities if e.type == etype], etype)
    alias_map.update(build_alias_map(clusters))

# 3. Assemble
for rel in raw_relations:
    s, t = alias_map[rel.source], alias_map[rel.target]
    G.add_edge(s, t, predicate=rel.predicate, source_doc=rel.source_document_id)

# 4. Summarize hubs
for node, degree in G.degree():
    if degree >= 3:
        profile = summarize(G, node)

# 5. Query
context = serialize_subgraph(G, "Apollo 11", hops=2)
answer = ask("Which locations connect to Apollo 11 crew?", context)
```

## Lessons

1. Descriptions are first-class — not metadata — for resolution.
2. Single connected component validates resolution quality.
3. k=2 captures nearly entire Apollo graph (22 nodes).
4. Grounded answers cite edges; ungrounded answers cite pretraining.
