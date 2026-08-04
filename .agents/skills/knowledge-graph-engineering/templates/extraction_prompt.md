# Extraction Prompt Template

Use with structured outputs (`output_format=ExtractedGraph`). Model: **Haiku** for high-volume extraction.

## System Context (cache this)

```
You are a knowledge graph extractor. You produce typed entities and subject-predicate-object triples from documents. Output must validate against the provided schema. Never invent entities not supported by the text.
```

## User Prompt

```
Extract a knowledge graph from the document below.

<document>
{text}
</document>

Guidelines:
- Extract only entities that are central to what this document is about — skip incidental mentions.
- For each entity, write a one-sentence description grounded in this document. These descriptions are used later to disambiguate entities with similar names.
- Predicates should be short verb phrases ("commanded", "launched from", "part of", "works_at", "owns").
- Every relation must connect two entities you extracted.
- Include provenance: note the source document ID in metadata when the schema supports it.
- Prefer specific predicates over vague ones ("commanded" not "was involved with").
- Do not extract entities mentioned only in passing unless they are structurally important to the document's argument.
```

## Schema (Pydantic)

```python
from typing import Literal
from pydantic import BaseModel, Field

EntityType = Literal[
    "PERSON", "ORGANIZATION", "LOCATION", "EVENT",
    "ARTIFACT", "CONCEPT", "PRODUCT", "DOCUMENT"
]

class Entity(BaseModel):
    name: str
    type: EntityType
    description: str = Field(description="One-line description grounded in this document")

class Relation(BaseModel):
    source: str
    predicate: str
    target: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class ExtractedGraph(BaseModel):
    entities: list[Entity]
    relations: list[Relation]
    source_document_id: str | None = None
```

## Precision vs Recall Tuning

| Goal | Prompt adjustment |
|------|-------------------|
| Higher recall | Change "central only" → "extract all mentioned entities" |
| Higher precision | Add "skip entities mentioned fewer than twice" |
| Domain-specific | Extend `EntityType` Literal with domain types (PRICING, PATENT, FEATURE) |

## API Call Pattern

```python
def extract(text: str, doc_id: str) -> ExtractedGraph:
    response = client.messages.parse(
        model="claude-haiku-4-5",
        max_tokens=4096,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}],
        output_format=ExtractedGraph,
    )
    result = response.parsed_output
    result.source_document_id = doc_id
    return result
```
