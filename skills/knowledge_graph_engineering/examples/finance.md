# Example: Finance / Competitive Intelligence Knowledge Graph

Multi-agent system with orchestrator + 5 specialist workers. Graph serves as shared memory across pricing, product, financial, marketing, and strategic synthesis agents.

## Architecture

```
                    Orchestrator
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   Pricing Agent    Product Agent    Financial Agent
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                  Knowledge Graph
                  (shared memory)
                         │
                         ▼
              Strategic Synthesizer
              (graph traversal, not raw docs)
```

## Extended Schema

```python
EntityType = Literal[
    "PERSON", "ORGANIZATION", "LOCATION", "EVENT", "ARTIFACT",
    "PRODUCT", "PRICING", "PATENT", "FILING", "FEATURE", "METRIC"
]

# Domain-specific relations
# (ProductA) --[priced_at]--> ($99/mo)
# (CompetitorX) --[filed]--> (PatentFiling)
# (Acme Corp) --[reported]--> (Revenue $2.1B)
# (Acme Corp) --[reduced_pricing_by]--> (15%)
```

## Workflow

### Step 1: Parallel Extraction

Each worker extracts from its corpus slice using the **same schema** (extended types):

| Worker | Corpus | Sample entities |
|--------|--------|-----------------|
| Pricing | pricing pages, rate cards | ProductA, PriceTier, $99/mo |
| Product | announcements, patents | PatentFiling, NewFeatureX |
| Financial | 10-K, 10-Q filings | Revenue, R&D spend, Acme Corp |
| Marketing | campaigns, press releases | CampaignX, BrandY |

### Step 2: Cross-Worker Resolution

Critical merge: "Acme Corp" (financial) + "ACME Corporation" (product) + "acme" (pricing) → **Acme Corp**.

Without resolution, strategic synthesizer cannot connect pricing drop to patent filing.

### Step 3: Graph Assembly

Each edge carries provenance:
```python
G.add_edge("Acme Corp", "PatentFiling-2024-XXXX",
    predicate="filed",
    source_agent="product-agent",
    source_document="patent-filing.pdf",
    extracted_at="2026-03-15T10:00:00Z")
```

### Step 4: Strategic Synthesis

Synthesizer queries: `subgraph(Acme Corp, k=2)` → ~50 triples, not 5 workers' raw outputs.

**Output example**:
> Acme Corp reduced pricing by 15% (source: pricing-agent, document: pricing-page-q3.html) while simultaneously filing patent US-2024-XXXX for a new product category (source: product-agent, document: patent-filing.pdf), suggesting a strategy of undercutting incumbents before launching a differentiated offering.

Every claim cites a graph edge with provenance.

## Hybrid GraphRAG Query

For analyst questions spanning domains:

```python
def finance_query(question: str) -> Answer:
    # 1. Vector search on entity profiles + key_facts embeddings
    seed_entities = vector_search(question, top_k=5)
    
    # 2. Graph expansion (k=2 from each seed)
    subgraph = expand_graph(G, seed_entities, hops=2)
    
    # 3. Hybrid score rerank
    candidates = rerank(subgraph, question, alpha=0.5, beta=0.35, gamma=0.15)
    
    # 4. LLM synthesis with edge citations
    return ask(question, serialize_subgraph(candidates))
```

## Temporal Queries

With temporal edges:
```
(Acme Corp) --[held_role: CEO]--> (John Smith)  valid: 2020-01 to 2024-06
(Acme Corp) --[held_role: CEO]--> (Jane Doe)     valid: 2024-07 to ongoing
```

Query: "Who was CEO of Acme Corp in Q3 2024?" → filter edges by time window before reasoning.

## Monitoring Signals

| Signal | Healthy | Alert |
|--------|---------|-------|
| Extraction rate | Stable per doc type | Sudden drop = domain shift |
| Resolution ratio | 1.5-2.5x compression | Near 1.0 = consistent naming; >3.0 = heavy variation |
| Components | 1-3 large components | Growing islands = missed cross-doc links |
| Query latency | <2s for k=2 | Pre-compute hub subgraphs for hot entities |

## Cost Profile

| Stage | Model | Volume | Cost driver |
|-------|-------|--------|-------------|
| Extraction | Haiku | 10K docs | Linear with corpus |
| Resolution | Sonnet | ~50 blocks/type | Sublinear |
| Summarization | Sonnet | Top-20 hubs | Fixed per run |
| Querying | Sonnet | Per question | Subgraph size |

Use prompt caching on extraction schema; Message Batches API for 50% off async jobs.
