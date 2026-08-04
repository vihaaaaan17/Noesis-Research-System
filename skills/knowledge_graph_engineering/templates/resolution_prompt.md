# Entity Resolution Prompt Template

Use with structured outputs. Model: **Sonnet** for reasoning over ambiguous clusters. Process **one entity type at a time**.

## User Prompt

```
Below are {entity_type} entities extracted from several documents. Some are different surface forms of the same real-world entity.

<entities>
{entity_list}
</entities>

Each entity is formatted as:
- name: <surface form>
- description: <one-line description from source document>
- source_documents: <list of doc IDs>

Cluster them. Rules:
1. Each input name must appear in exactly one cluster's aliases list.
2. Entities that are genuinely distinct get their own single-element cluster.
3. Use descriptions to avoid merging entities that merely share a name.
4. The canonical name should be the most complete, unambiguous form.
5. When descriptions conflict, prefer NOT merging unless evidence is strong.
6. Abbreviations and full names ("NASA" / "National Aeronautics and Space Administration") should merge.
7. Nicknames with zero string overlap ("Edwin Aldrin" / "Buzz Aldrin") should merge when descriptions align.
```

## Schema (Pydantic)

```python
class Cluster(BaseModel):
    canonical: str
    aliases: list[str]
    entity_type: str
    merge_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class ResolvedClusters(BaseModel):
    clusters: list[Cluster]
```

## Entity List Formatting

```python
def format_entity_list(entities: list[Entity]) -> str:
    lines = []
    for e in entities:
        docs = ", ".join(e.source_documents or ["unknown"])
        lines.append(f"- name: {e.name}\n  description: {e.description}\n  source_documents: {docs}")
    return "\n".join(lines)
```

## Blocking Pre-Filter (run before LLM)

```python
def block_candidates(entities: list[Entity], block_size: int = 75) -> list[list[Entity]]:
    """Group by cheap signals before Sonnet arbitration."""
    from collections import defaultdict
    blocks = defaultdict(list)
    for e in entities:
        # Block by last token (surname), first 3 chars, or embedding bucket
        key = e.name.split()[-1].lower() if " " in e.name else e.name[:3].lower()
        blocks[key].append(e)
    # Split oversized blocks by embedding similarity or token overlap
    result = []
    for group in blocks.values():
        for i in range(0, len(group), block_size):
            result.append(group[i:i + block_size])
    return result
```

## Fallback for Unmatched Names

```python
def build_alias_map(clusters: list[Cluster], all_names: set[str]) -> dict[str, str]:
    alias_map = {}
    matched = set()
    for c in clusters:
        for alias in c.aliases:
            alias_map[alias] = c.canonical
            matched.add(alias)
    # CRITICAL: unmatched names become single-element clusters
    for name in all_names - matched:
        alias_map[name] = name
    return alias_map
```

## Anti-Patterns

- **Silent loss**: Names not in any cluster vanish from the graph.
- **Over-merging**: "Gemini 12" merged into "Project Gemini" due to overlapping descriptions.
- **Cross-type merging**: Never resolve PERSON and ORGANIZATION in the same call.
