# Entity Summarization Prompt Template

Model: **Sonnet**. Apply selectively to hub nodes (degree ≥ 3).

## User Prompt

```
Generate a knowledge-graph profile for this entity.

Entity: {name} ({entity_type})

Source excerpts mentioning this entity:
{excerpts}

Known relations in the graph:
{relations}

Write a 2-3 paragraph factual summary synthesized from the excerpts, resolving any contradictions by preferring the most specific claim. Include 3-5 atomic key facts, each traceable to the sources. For the time range, use YYYY or YYYY-MM format. Do not invent facts not supported by the excerpts.
```

## Schema (Pydantic)

```python
class TimeRange(BaseModel):
    start: str  # YYYY or "unknown"
    end: str    # YYYY or "ongoing"

class EntityProfile(BaseModel):
    summary: str
    key_facts: list[str]
    time_range: TimeRange
    source_document_ids: list[str] = []
```

## When to Summarize

| Criterion | Action |
|-----------|--------|
| degree ≥ 3 | Summarize (hub node) |
| degree 1-2 | Keep single-document description |
| source_document set changed | Re-summarize |
| source_document set unchanged | Skip (cache hit) |

## Context Assembly

```python
def build_summarization_context(G, entity_name: str) -> tuple[str, str]:
    node = G.nodes[entity_name]
    excerpts = node.get("source_excerpts", [])
    relations = []
    for _, target, data in G.out_edges(entity_name, data=True):
        relations.append(f"({entity_name}) --[{data['predicate']}]--> ({target})")
    for source, _, data in G.in_edges(entity_name, data=True):
        relations.append(f"({source}) --[{data['predicate']}]--> ({entity_name})")
    return "\n---\n".join(excerpts), "\n".join(relations)
```

## Temporal Edge Extension (optional)

For temporal graphs, extend relations:

```python
class TemporalRelation(BaseModel):
    source: str
    predicate: str
    target: str
    valid_from: str | None = None  # YYYY-MM
    valid_to: str | None = None    # YYYY-MM or "ongoing"
```

Summarization prompt addition:
```
If temporal information is available in excerpts, note when relationships were active.
```
