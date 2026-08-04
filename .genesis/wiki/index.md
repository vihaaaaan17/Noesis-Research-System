# Wiki Index — MAS

The project knowledge base. Same schema as the agentic-swe-kit wiki: concept pages in `concepts/`,
each with frontmatter and ≥2 `[[wikilinks]]`. The L3 RESEARCH loop writes here; G0 reads here first.

> **Read this file before any milestone (G0 step 1).** Pick candidate pages by name-matching the
> milestone's nouns, then drill in. The wiki is what prevents rebuilding work that already exists.

## Entities (the things this system has)
- [[concepts/research-agent]] — Specialized agent assigned to one of the 8 research pipeline phases.
- [[concepts/research-tool]] — Python-wrapped tool (SymPy, SciPy, arXiv, Wikipedia, LatexFormatter).
- [[concepts/react-loop]] — Decoupled ReAct action-observation loop driving agent tool calls.

## Concepts (how it works)
- [[concepts/8-phase-research-pipeline]] — Sequential gated execution from Understand to LaTeX Report.
- [[concepts/sliding-window-memory]] — Short-term context sliding window and LLM summarization.

## Sources (research distilled by L3)
- [[concepts/mas-readme-architecture]] — Architecture specification from README.md.

## Seeded from agentic-swe-kit
Relevant global concept pages for this project's phases (pointers only — read on demand):
- `$AGENTIC_SWE_WIKI_ROOT/modular-architecture/concepts/clean-architecture.md` — when deciding module boundaries
- `$AGENTIC_SWE_WIKI_ROOT/llmops-ai-agents/concepts/multi-agent-orchestration.md` — when expanding agent tools & prompts
- `$AGENTIC_SWE_WIKI_ROOT/production-readiness/concepts/circuit-breakers-retries.md` — when tuning rate limiters & backoffs

