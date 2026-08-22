"""
backend/api.py
---------------------------------------------------------------------
Production FastAPI Backend Server for MAS.

Thin transport layer providing SSE real-time streaming endpoints over
ResearchOrchestrator, Knowledge Graph inspection, secured report lookup,
and static frontend file serving.
---------------------------------------------------------------------
"""

import os
import sys
import json
import asyncio
import time
from typing import Optional, AsyncGenerator, Dict, Any
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from memory.working_memory import WorkingMemory
from memory.long_term import LongTermMemory
from evals.telemetry import TelemetryLogger
from orchestrator import ResearchOrchestrator
from agents.research_agents import (
    literature_scout, mathematician, engineer,
    numerical_analyst, peer_reviewer, synthesizer, report_writer
)

app = FastAPI(
    title="MAS - Research Agent System API",
    description="Multi-Agent Research Pipeline powered by Google Gemini & Knowledge Graph Shared Memory",
    version="2.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    """Health check endpoint returning system status."""
    return {
        "status": "online",
        "system": "Research Agent System (MAS)",
        "default_model": config.DEFAULT_MODEL,
        "api_key_configured": bool(config.GEMINI_API_KEY),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }


REPORTS_DIR = os.path.abspath("reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


@app.get("/api/reports")
def list_reports():
    """List all generated report files."""
    if not os.path.exists(REPORTS_DIR):
        return {"reports": []}

    files = []
    for fname in os.listdir(REPORTS_DIR):
        if fname.endswith(".md"):
            fpath = os.path.join(REPORTS_DIR, fname)
            stat = os.stat(fpath)
            files.append({
                "filename": fname,
                "size_bytes": stat.st_size,
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
            })
    files.sort(key=lambda x: x["modified"], reverse=True)
    return {"reports": files}


@app.get("/api/reports/{filename}")
def get_report(filename: str):
    """Retrieve content of a specific report file with strict path-traversal protection."""
    target_path = os.path.abspath(os.path.join(REPORTS_DIR, filename))
    if os.path.commonpath([target_path, REPORTS_DIR]) != REPORTS_DIR:
        raise HTTPException(status_code=400, detail="Invalid report filename path traversal detected")

    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="Report file not found")

    with open(target_path, "r", encoding="utf-8") as f:
        content = f.read()
    return {"filename": filename, "content": content}


_ACTIVE_STREAM_LOCK = asyncio.Lock()


@app.get("/api/research/stream")
async def stream_research(
    question: str = Query(..., description="The research question to investigate"),
    depth: str = Query("standard", description="Execution depth: quick, standard, deep"),
    mode: str = Query("research_paper", description="Generation mode: research_paper, long_form, explanation, paragraph, answer")
):
    """
    Stream real-time research execution events via SSE (Server-Sent Events).
    Emits explicit structured event payloads with visibility and category.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        def send_event(event_type: str, data: dict) -> str:
            return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

        async with _ACTIVE_STREAM_LOCK:
            yield send_event("status", {
                "visibility": "system",
                "category": "PHASE_STATUS",
                "message": f"Initializing Research Orchestrator ({depth.upper()} mode, mode='{mode}')..."
            })
            await asyncio.sleep(0.1)

            working_mem = WorkingMemory(verbose=False)
            long_term_mem = LongTermMemory(verbose=False)
            telemetry = TelemetryLogger(run_name=f"Web_Run_{depth}", verbose=False)

            orc = ResearchOrchestrator(
                depth=depth,
                mode=mode,
                output_dir="reports",
                verbose=True,
                working_memory=working_mem,
                long_term_memory=long_term_mem,
                enable_judge=True
            )

            orc.register_agents(
                scout=literature_scout(verbose=False),
                mathematician=mathematician(verbose=False),
                engineer=engineer(domain="general", verbose=False),
                numerical=numerical_analyst(verbose=False),
                reviewer=peer_reviewer(verbose=False),
                synthesizer=synthesizer(verbose=False),
                writer=report_writer(verbose=False),
            )

            active_phases = orc.DEPTH_PHASES.get(depth, orc.DEPTH_PHASES["standard"])

            yield send_event("pipeline_start", {
                "visibility": "system",
                "category": "PHASE_STATUS",
                "question": question,
                "depth": depth,
                "mode": mode,
                "total_phases": len(active_phases),
                "active_phases": [orc.PHASES[p][0] for p in active_phases]
            })

            # Execute pipeline non-blockingly via run_in_executor
            loop = asyncio.get_event_loop()
            result_future = loop.run_in_executor(None, orc.run, question)

            # Stream thinking and status events while research runs
            while not result_future.done():
                completed = len(orc.completed_phases)
                if completed < len(active_phases):
                    current_p_num = active_phases[completed] if completed < len(active_phases) else active_phases[-1]
                    p_name, p_desc = orc.PHASES[current_p_num]
                    
                    yield send_event("thinking", {
                        "visibility": "internal",
                        "category": "PHASE_STATUS",
                        "phase_name": p_name,
                        "phase_num": current_p_num,
                        "total_phases": len(active_phases),
                        "message": f"Executing Phase {current_p_num}/{len(active_phases)} ({p_name}): {p_desc}..."
                    })

                    val_stats = working_mem.graph_memory.validate()
                    yield send_event("kg_update", {
                        "visibility": "internal",
                        "category": "MEMORY_ACTIVITY",
                        "phase": p_name,
                        "num_nodes": val_stats["num_nodes"],
                        "num_edges": val_stats["num_edges"]
                    })

                await asyncio.sleep(1.0)

            # Retrieve final result from future
            try:
                final_result = await result_future
            except Exception as ex:
                yield send_event("pipeline_error", {
                    "visibility": "system",
                    "category": "ERROR",
                    "error": str(ex)
                })
                return

            if orc.status == "PAUSED_RATE_LIMIT":
                yield send_event("pipeline_paused", {
                    "visibility": "system",
                    "category": "PROVIDER_ACTIVITY",
                    "status": "PAUSED_RATE_LIMIT",
                    "run_id": orc.run_id,
                    "message": "Research pipeline paused due to rate limits across all providers. Checkpoint saved."
                })
                return

            # Export active Knowledge Graph to disk for live visualizer
            try:
                kg_export = working_mem.graph_memory.to_dict()
                with open(os.path.join(REPORTS_DIR, "research_kg.json"), "w", encoding="utf-8") as kgf:
                    json.dump(kg_export, kgf, indent=2)
                working_mem.graph_memory.save_to_json("long_term_graph.json")
            except Exception:
                pass

            # Export full budget & LLM telemetry record to disk
            budget_metrics = orc.budget.get_summary()
            with open(os.path.join("reports", "telemetry_last_run.json"), "w", encoding="utf-8") as tf:
                json.dump(budget_metrics, tf, indent=2)

            yield send_event("research_complete", {
                "visibility": "user",
                "category": "FINAL_RESPONSE",
                "run_id": orc.run_id,
                "status": orc.status,
                "report_markdown": final_result,
                "telemetry": budget_metrics
            })

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class FollowupRequest(BaseModel):
    question: str
    report_context: Optional[str] = ""
    history: Optional[list] = []


@app.post("/api/chat/followup")
def chat_followup(req: FollowupRequest):
    """Interactive follow-up question endpoint using query-aware memory retrieval."""
    wm = WorkingMemory(verbose=False)
    query_context = wm.get_context(req.question)

    messages = [
        {"role": "system", "content": f"You are an AI research assistant. Answer the follow-up query using LaTeX for equations where appropriate.\n\n### RELEVANT MEMORY CONTEXT:\n{query_context}"}
    ]

    if req.history:
        for item in req.history:
            messages.append({"role": item.get("role", "user"), "content": item.get("content", "")})

    messages.append({"role": "user", "content": req.question})

    answer = config.call_with_fallback(
        messages=messages,
        primary_model=config.GEMINI_FINAL_MODEL,
        temperature=0.4,
        max_tokens=2500
    )
    return {"answer": answer}


@app.get("/api/graph")
def get_knowledge_graph():
    """Return complete Knowledge Graph nodes and edges for live visualizer."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    paths_to_check = [
        os.path.join(root_dir, "long_term_graph.json"),
        os.path.join(root_dir, "reports", "research_kg.json")
    ]

    data = {}
    for p in paths_to_check:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if content and (content.get("nodes") or content.get("edges")):
                        data = content
                        break
            except Exception:
                pass

    if not data:
        wm = WorkingMemory(verbose=False)
        data = wm.graph_memory.to_dict()

    nodes_raw = data.get("nodes", [])
    edges_raw = data.get("edges", []) or data.get("links", [])

    nodes = []
    node_set = set()
    for n in nodes_raw:
        nid = str(n.get("name") or n.get("id") or "")
        if nid and nid not in node_set:
            node_set.add(nid)
            nodes.append({
                "id": nid,
                "label": nid,
                "type": str(n.get("entity_type") or n.get("type") or "CONCEPT").upper(),
                "description": n.get("description", ""),
                "facts": n.get("key_facts") or n.get("facts") or []
            })

    edges = []
    for e in edges_raw:
        s = str(e.get("source") or "")
        t = str(e.get("target") or "")
        if s and t:
            edges.append({
                "source": s,
                "target": t,
                "label": str(e.get("predicate") or e.get("relation") or "relates_to")
            })

    return {
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "nodes": nodes,
        "edges": edges
    }


# Mount static frontend directory
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
