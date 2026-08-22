# Open-Notebook Architecture Analysis & Noesis Production Integration Roadmap

## 📌 Executive Overview

This document provides a comprehensive technical breakdown of **`open-notebook`** (the open-source Google NotebookLM alternative) and establishes a structured integration roadmap for **`Noesis` (MAS)**.

While `open-notebook` excels at multi-modal source ingestion, multi-speaker podcast generation, and prompt transformations over static documents, **Noesis** possesses a superior autonomous multi-agent reasoning engine (8-phase ReAct pipeline), GraphRAG Knowledge Graph Memory (NetworkX `MultiDiGraph`), Centralized LLM Budget Manager, and a HIG Apple Liquid Glass Web UI.

By systematically adapting `open-notebook`'s core capabilities into Noesis, we will transform Noesis from an autonomous research engine into an **end-to-end production research & intelligence workspace**.

---

## 📑 Table of Contents
1. [Deep-Dive Architectural Breakdown of `open-notebook`](#1-deep-dive-architectural-breakdown-of-open-notebook)
2. [Noesis Current Capabilities & Architectural Gap Analysis](#2-noesis-current-capabilities--architectural-gap-analysis)
3. [Proposed Production Features for Noesis](#3-proposed-production-features-for-noesis)
   - [Feature 1: Multi-Speaker Audio Research Podcasts & Spoken Briefings](#feature-1-multi-speaker-audio-research-podcasts--spoken-briefings)
   - [Feature 2: Multi-Source Document Ingestion (PDFs, Web URLs, YouTube, Papers)](#feature-2-multi-source-document-ingestion-pdfs-web-urls-youtube-papers)
   - [Feature 3: 1-Click Research Transformation Cards](#feature-3-1-click-research-transformation-cards)
   - [Feature 4: Dynamic Encrypted Provider Vault & Local Model Manager](#feature-4-dynamic-encrypted-provider-vault--local-model-manager)
   - [Feature 5: Persistent Notebook Workspaces & Asset Library](#feature-5-persistent-notebook-workspaces--asset-library)
4. [System Architecture & Scalability Design](#4-system-architecture--scalability-design)
5. [Multi-Phase Research, Design & Implementation Blueprint](#5-multi-phase-research-design--implementation-blueprint)

---

## 1. Deep-Dive Architectural Breakdown of `open-notebook`

`open-notebook` is structured as a privacy-focused, self-hosted web system composed of **SurrealDB**, **FastAPI (Python backend)**, and **Next.js (React frontend)**. Its key architectural pillars include:

```
                          ┌────────────────────────────────────────┐
                          │         Open-Notebook System           │
                          └───────────────────┬────────────────────┘
                                              │
         ┌──────────────────┬─────────────────┼──────────────────┬──────────────────┐
         ▼                  ▼                 ▼                  ▼                  ▼
┌──────────────────┐┌──────────────┐┌──────────────────┐┌──────────────────┐┌──────────────────┐
│ Source Ingestion ││ Audio Podcast││ Transformations  ││ Encrypted Vault  ││ Notebook Storage │
│ (PDF/URL/YouTube)││ (TTS + Script││ (Prompt Templates││ (Fernet Crypto + ││ (SurrealDB Vector│
│                  ││  Multi-Voice)││  Study Guides)   ││  18+ LLMs)       ││  + Graph Store)  │
└──────────────────┘└──────────────┘└──────────────────┘└──────────────────┘└──────────────────┘
```

### 1.1 Source & Document Management System
- **Extractors**: Modules for extracting clean text from PDFs (`pypdf`), Web pages (`httpx` + HTML parsers), YouTube video transcripts (`youtube-transcript-api`), and audio transcriptions (`Whisper`).
- **Chunking & Vector Store**: Text is split into chunks, embedded via default/configured embedding models, and stored in SurrealDB vector tables for similarity retrieval.

### 1.2 Multi-Speaker Audio Podcast Engine
- **Dialogue Script Generator**: Prompts LLMs to generate structured dialogue between 1 to 4 speakers with distinct personas (e.g. *Host*, *Co-Host*, *Expert*, *Skeptic*).
- **TTS Synthesis Adapters**: Uses EdgeTTS (zero-cost Microsoft Edge neural voices), ElevenLabs, or OpenAI TTS to synthesize per-turn audio clips.
- **Audio Assembly**: Combines individual speaker audio clips into seamless `.mp3` podcasts with speaker timestamps.

### 1.3 AI Content Transformation Engine
- **Prompt Registry**: Maintains structured prompt templates that transform raw notebook sources or chat outputs into formatted analytical formats:
  - Executive Summaries
  - FAQs & Self-Quiz Questions
  - Study Guides
  - Action Plans / Bulleted Insights

### 1.4 Dynamic Encrypted Provider Vault
- **Encrypted Storage**: Uses Python `cryptography` (Fernet symmetric encryption) to store user API keys securely in the database (`OPEN_NOTEBOOK_ENCRYPTION_KEY`).
- **Model Discovery**: Dynamically queries provider endpoints (`/v1/models`) to sync available models across OpenAI, Anthropic, Gemini, Groq, Ollama, and LM Studio.

---

## 2. Noesis Current Capabilities & Architectural Gap Analysis

| Architectural Dimension | Noesis (Current System) | Open-Notebook | Target State after Integration |
| :--- | :--- | :--- | :--- |
| **Reasoning Engine** | 8-Phase ReAct Multi-Agent System (Literature, Math, Compute, Peer Review, Judge) | Single-turn RAG prompt | **Retain Noesis 8-Phase Engine** (Industry leading reasoning) |
| **Memory System** | GraphRAG Knowledge Graph (`MultiDiGraph`) + 3D WebGL Visualizer | Document Vector Chunking | **Hybrid GraphRAG + User Source Chunks** |
| **Audio Output** | None | Multi-Speaker Podcast Generation | **Add Audio Research Briefings & Dialogue Podcasts** |
| **Custom Inputs** | Query text only | PDFs, Web URLs, YouTube, Text | **Add Multi-Source Ingestion Pipeline** |
| **Post-Processing** | Raw Markdown + KaTeX rendering | 1-Click Transformation Cards | **Add 1-Click Transformation Cards** |
| **Key Governance** | Hardcoded/Env configuration | Encrypted UI Vault + Model Sync | **Add UI Encrypted Provider & Model Vault** |
| **Workspace Model** | Session chat history | Scoped Project Notebooks | **Add Persistent Notebook Workspaces** |

---

## 3. Proposed Production Features for Noesis

---

### Feature 1: Multi-Speaker Audio Research Podcasts & Spoken Briefings
* **Goal**: Allow users to click **"Listen to Podcast"** on any completed research report to hear an engaging 2-speaker audio discussion (e.g., *Lead Researcher* presenting findings + *Peer Judge* asking critical questions).
* **Key Components**:
  1. **Podcast Script Agent** (`agents/podcast_agent.py`): Takes the final synthesized markdown report and generates a clean, conversational JSON transcript:
     ```json
     [
       {"speaker": "Host", "text": "Welcome to Noesis Audio. Today we're breaking down expected value in game theory..."},
       {"speaker": "Expert", "text": "Exactly. At its core, expected value balances the probability of winning against the pot odds..."}
     ]
     ```
  2. **Audio Synthesis Engine** (`core/audio_synthesizer.py`): Asynchronously generates `.mp3` audio clips using `edge-tts` (zero-cost) or OpenAI TTS and stitches them with `pydub`.
  3. **Frontend Audio Player Widget**: Integrated HIG-styled audio player floating above composer with playback speed control, scrub bar, and speaker subtitle highlight.

---

### Feature 2: Multi-Source Document Ingestion (PDFs, Web URLs, YouTube, Papers)
* **Goal**: Enable users to attach PDFs, Web paper links, or YouTube transcripts directly to a prompt before triggering a research run.
* **Key Components**:
  1. **Source Extractor Service** (`core/source_ingestion.py`):
     - **PDF Extractor**: `pypdf` / `pdfplumber` to extract structured text & mathematical formulas.
     - **URL Extractor**: `httpx` + `trafilatura` for clean main-content extraction.
     - **YouTube Extractor**: `youtube-transcript-api` for automated transcript parsing.
  2. **WorkingMemory Integration**: Ingested content is summarized, entity-extracted into `KnowledgeGraphMemory`, and injected into `ContextBuilder` alongside live web search results.
  3. **Frontend File Attachment Bar**: Drag-and-drop file attachment area inside the bottom composer.

---

### Feature 3: 1-Click Research Transformation Cards
* **Goal**: Provide 1-click transformation cards at the end of a report to convert complex technical output into specialized formats.
* **Transformation Types**:
  - 📄 **Executive 1-Pager**: Dense 3-paragraph summary for executive stakeholders.
  - 🎓 **Study & Proof Guide**: Derivation steps, mathematical formulas, and definitions.
  - 📊 **Contradiction & Gap Matrix**: Detailed breakdown of conflicting literature & open research questions.
  - 💡 **Slide Deck Outline**: Bulleted presentation slides with speaker notes.
  - ❓ **Interactive FAQ**: Q&A pairs for quick revision.
* **Key Components**:
  1. **Transformation Agent** (`agents/transformation_agent.py`): Specialized prompt templates executing fast single-turn extractions.
  2. **UI Action Cards**: Floating action chips at the bottom of the response turn.

---

### Feature 4: Dynamic Encrypted Provider Vault & Local Model Manager
* **Goal**: Allow users to manage API keys (Gemini, Groq, OpenAI, Anthropic, Ollama, LM Studio) directly in the web UI.
* **Key Components**:
  1. **Fernet Cryptographic Vault** (`core/vault.py`): Encrypts sensitive keys stored locally on disk (`data/vault.json`) using `cryptography.fernet`.
  2. **Model Discovery & Testing Endpoint**: `/api/vault/test` and `/api/vault/models` to discover active model IDs dynamically.
  3. **UI Settings Modal**: HIG-styled modal for provider management and status check.

---

### Feature 5: Persistent Notebook Workspaces & Asset Library
* **Goal**: Group research runs, uploaded PDFs, Knowledge Graph snapshots, audio podcasts, and transformation cards into named persistent project folders ("Notebooks").
* **Key Components**:
  1. **Notebook Workspace Manager** (`core/workspace_manager.py`): Local JSON-backed workspace registry.
  2. **Sidebar Project Switcher**: Expanded sidebar section listing active Notebooks and their associated research assets.

---

## 4. System Architecture & Scalability Design

```
                                  +---------------------------------------+
                                  |         Noesis Web Frontend           |
                                  | (SPA / HIG Liquid Glass / KaTeX / SVG)|
                                  +-------------------+-------------------+
                                                      |
                                                      | REST / SSE API
                                                      v
                                  +-------------------+-------------------+
                                  |         FastAPI Backend Router        |
                                  |            (backend/api.py)           |
                                  +---------+-------------------+---------+
                                            |                   |
               +----------------------------+                   +----------------------------+
               |                                                                             |
               v                                                                             v
+--------------+---------------+                                              +--------------+---------------+
| Autonomous Multi-Agent Engine|                                              |    Ingestion & Asset Services |
|  - Orchestrator (8-Phase)    |                                              |  - Source Ingestion (PDF/URL)|
|  - WorkingMemory & GraphRAG  |                                              |  - Podcast Generator (TTS)   |
|  - LLM Budget Manager        |                                              |  - Transformation Agent      |
|  - Judge Agent Peer Review   |                                              |  - Fernet Provider Vault     |
+--------------+---------------+                                              +--------------+---------------+
               |                                                                             |
               +----------------------------+------------------------------------------------+
                                            |
                                            v
                               +------------+------------+
                               |     Local Storage &     |
                               |    Persistence Tier     |
                               | (JSON / NetworkX / MP3) |
                               +-------------------------+
```

### Scalability & Performance Strategy:
1. **Async Non-Blocking Tasks**: Heavy audio generation and PDF parsing run as background tasks via FastAPI `BackgroundTasks` with real-time SSE progress notifications.
2. **Zero-Cost Default Audio Stack**: Uses Microsoft `edge-tts` by default (no external API costs or limits) with option to fallback to OpenAI TTS.
3. **Strict Budget Integration**: Ingested sources and transformation tasks are tracked by `LLMBudgetManager` to prevent token budget exhaustion.

---

## 5. Multi-Phase Research, Design & Implementation Blueprint

We will execute this transformation systematically across **5 sequential phases**:

```
 ┌────────────────┐      ┌────────────────┐      ┌────────────────┐      ┌────────────────┐      ┌────────────────┐
 │    Phase 1     │      │    Phase 2     │      │    Phase 3     │      │    Phase 4     │      │    Phase 5     │
 │ Research &     │ ───► │ Source         │ ───► │ Transformation │ ───► │ Audio Podcast  │ ───► │ Encrypted      │
 │ Architecture   │      │ Ingestion      │      │ Cards          │      │ Synthesis      │      │ Provider Vault │
 └────────────────┘      └────────────────┘      └────────────────┘      └────────────────┘      └────────────────┘
```

### 💬 Next Steps:
For each feature phase, we will:
1. **Research & System Architecture Design**: Detail exact data structures, API endpoints, and component boundaries.
2. **Tech-Stack & Scalability Audit**: Verify dependencies (`pypdf`, `edge-tts`, `pydub`, `cryptography`, etc.).
3. **Implementation Plan (`implementation_plan.md`)**: Create a machine-parseable step-by-step plan for user approval.
4. **Execution & Empirical Verification**: Build the feature, run verification tests, and document in `walkthrough.md`.
