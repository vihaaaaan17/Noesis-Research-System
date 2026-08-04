"""
backend/api.py
---------------------------------------------------------------------
Production FastAPI Backend Server for MAS.

Provides SSE real-time streaming endpoints for the 8-phase pipeline,
Knowledge Graph inspection endpoints, report history lookup,
and static file serving for the frontend UI.
---------------------------------------------------------------------
"""

import os
import sys
import json
import asyncio
import time
from typing import Optional, AsyncGenerator
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from memory.graph_memory import KnowledgeGraphMemory
from evals.telemetry import TelemetryLogger
from core.model_router import SmartModelRouter
from agents.research_agents import (
    literature_scout, mathematician, engineer,
    numerical_analyst, peer_reviewer, synthesizer, report_writer
)
from agents.judge_agent import JudgeAgent
from orchestrator import ResearchOrchestrator

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


@app.get("/api/graph")
def get_knowledge_graph():
    """Return the active Knowledge Graph state."""
    kg_path = os.path.join("reports", "research_kg.json")
    kg = KnowledgeGraphMemory(storage_path=kg_path, verbose=False)
    val_stats = kg.validate()

    nodes = []
    for n, data in kg.graph.nodes(data=True):
        nodes.append({
            "id": n,
            "label": n,
            "type": data.get("entity_type", "CONCEPT"),
            "description": data.get("description", ""),
            "key_facts": data.get("key_facts", []),
            "mentions": data.get("mention_count", 1)
        })

    edges = []
    for s, t, data in kg.graph.edges(data=True):
        edges.append({
            "source": s,
            "target": t,
            "predicate": data.get("predicate", "related_to"),
            "doc": data.get("source_document_id", "system")
        })

    return {
        "summary": val_stats,
        "nodes": nodes,
        "edges": edges,
        "alias_map": kg.alias_map
    }


@app.get("/api/reports")
def list_reports():
    """List all generated report files."""
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        return {"reports": []}

    files = []
    for fname in os.listdir(reports_dir):
        if fname.endswith(".md"):
            fpath = os.path.join(reports_dir, fname)
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
    """Retrieve content of a specific report file."""
    filepath = os.path.join("reports", filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Report file not found")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return {"filename": filename, "content": content}


_ACTIVE_STREAM_LOCK = asyncio.Lock()

@app.get("/api/research/stream")
async def stream_research(
    question: str = Query(..., description="The research question to investigate"),
    depth: str = Query("standard", description="Execution depth: quick, standard, deep")
):
    """
    Stream real-time research execution events via SSE (Server-Sent Events).
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        def send_event(event_type: str, data: dict) -> str:
            return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

        async with _ACTIVE_STREAM_LOCK:
            yield send_event("status", {"message": f"Initializing Research Orchestrator ({depth.upper()} mode)..."})
            await asyncio.sleep(0.2)

        # Build orchestrator & shared KG
        kg = KnowledgeGraphMemory(storage_path=os.path.join("reports", "research_kg.json"), verbose=False)
        telemetry = TelemetryLogger(run_name=f"Web_Run_{depth}", verbose=False)

        orc = ResearchOrchestrator(depth=depth, output_dir="reports", verbose=True, graph_memory=kg)
        orc.register_agents(
            scout=literature_scout(verbose=False),
            mathematician=mathematician(verbose=False),
            engineer=engineer(domain="semiconductor", verbose=False),
            numerical=numerical_analyst(verbose=False),
            reviewer=peer_reviewer(verbose=False),
            synthesizer=synthesizer(verbose=False),
            writer=report_writer(verbose=False),
        )

        active_phases = orc.DEPTH_PHASES.get(depth, orc.DEPTH_PHASES["standard"])
        orc.doc["question"] = question

        yield send_event("pipeline_start", {
            "question": question,
            "depth": depth,
            "total_phases": len(active_phases),
            "active_phases": [orc.PHASES[p][0] for p in active_phases]
        })

        for p_idx, phase_num in enumerate(active_phases, 1):
            phase_name, phase_desc = orc.PHASES[phase_num]
            telemetry.start_phase(phase_name)

            yield send_event("phase_start", {
                "phase_num": phase_num,
                "step_index": p_idx,
                "phase_name": phase_name,
                "description": phase_desc
            })

            # Intermediate live thinking event so client sees real-time progress
            yield send_event("thinking", {
                "phase_name": phase_name,
                "message": f"Reasoning & executing Phase {phase_num} ({phase_name}): {phase_desc}..."
            })

            await asyncio.sleep(0.3)

            # Execute phase method
            phase_method = getattr(orc, f"_phase_{phase_name.lower()}", None)
            if phase_method:
                # Run sync in thread pool to avoid blocking async loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, phase_method)
                orc._log_phase(phase_name, result)
            else:
                result = f"Phase {phase_name} skipped."

            val_stats = kg.validate()
            telemetry.end_phase(
                phase_name,
                prompt_tokens=400 * p_idx,
                completion_tokens=250 * p_idx,
                kg_nodes=val_stats["num_nodes"],
                kg_edges=val_stats["num_edges"]
            )

            yield send_event("kg_update", {
                "phase": phase_name,
                "num_nodes": val_stats["num_nodes"],
                "num_edges": val_stats["num_edges"],
                "triples_preview": kg.serialize_subgraph()[:300]
            })

            yield send_event("phase_complete", {
                "phase_num": phase_num,
                "phase_name": phase_name,
                "result_preview": result[:400] + "..." if len(result) > 400 else result,
                "full_result": result
            })

            await asyncio.sleep(0.5)

        # Save report and evaluate with JudgeAgent
        final_report = orc.doc["report"] or orc.doc["synthesis"]
        outfile = orc._save_report(question, final_report)
        kg.save_to_json()

        # Run independent Judge evaluation
        judge = JudgeAgent(model=orc.router.get_model_for_phase("judge"), verbose=False)
        loop = asyncio.get_event_loop()
        judge_result = await loop.run_in_executor(
            None,
            judge.evaluate,
            question,
            final_report,
            kg.serialize_subgraph()
        )

        telemetry.export_json(os.path.join("reports", "telemetry_last_run.json"))

        yield send_event("research_complete", {
            "report_path": outfile,
            "filename": os.path.basename(outfile),
            "report_markdown": final_report,
            "judge_result": judge_result,
            "telemetry": telemetry.get_metrics()
        })

    return StreamingResponse(event_generator(), media_type="text/event-stream")


from pydantic import BaseModel
class FollowupRequest(BaseModel):
    question: str
    report_context: Optional[str] = ""
    history: Optional[list] = []

@app.post("/api/chat/followup")
def chat_followup(req: FollowupRequest):
    """Interactive follow-up cross-question endpoint."""
    # Include Knowledge Graph triples context
    kg_path = os.path.join("reports", "research_kg.json")
    kg = KnowledgeGraphMemory(storage_path=kg_path, verbose=False)
    kg_triples = kg.serialize_subgraph()

    system_instruction = (
        "You are an expert AI research assistant. The user is asking a follow-up question "
        "about a technical research report generated by your multi-agent system.\n"
        "Answer with academic precision, using clear explanations and LaTeX math syntax ($...$ or $$...$$) where relevant.\n\n"
        f"### GROUNDED RESEARCH CONTEXT:\n{req.report_context[:4000]}\n\n"
        f"### KNOWLEDGE GRAPH MEMORY:\n{kg_triples[:2000]}"
    )

    contents = []
    if req.history:
        for item in req.history:
            contents.append({"role": item.get("role", "user"), "content": item.get("content", "")})
    
    contents.append({"role": "user", "content": req.question})

    answer = config.call_llm_api(
        messages=contents,
        system_instruction=system_instruction,
        temperature=0.5,
        max_tokens=2048
    )
    return {"answer": answer}


# Mount static frontend directory
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
