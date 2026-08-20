"""
inspect_run.py
---------------------------------------------------------------------
Inspect durable checkpoints, telemetry counters, and phase execution
for your actual research runs.
---------------------------------------------------------------------
"""

import os
import glob
import json

CHECKPOINT_DIR = "reports/checkpoints"

def inspect_latest_run():
    files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.json"))
    if not files:
        print("No checkpoint records found.")
        return

    # Sort by timestamp
    latest_file = max(files, key=os.path.getmtime)
    print(f"\n==============================================================")
    print(f"  INSPECTING YOUR LATEST RESEARCH RUN RECORD")
    print(f"==============================================================")
    print(f"  Checkpoint File: {latest_file}\n")

    with open(latest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  Run ID:           {data.get('run_id')}")
    print(f"  Question:         '{data.get('question')}'")
    print(f"  Depth Strategy:   {data.get('depth')}")
    print(f"  Output Format:    {data.get('mode')}")
    print(f"  Pipeline Status:  {data.get('status')}")
    print(f"  Completed Phases: {data.get('completed_phases')}")

    # Load budget state from checkpoint or telemetry file
    budget = data.get("budget_state")
    telemetry_path = os.path.join("reports", "telemetry_last_run.json")
    if not budget and os.path.exists(telemetry_path):
        try:
            with open(telemetry_path, "r", encoding="utf-8") as tf:
                budget = json.load(tf)
        except Exception:
            pass

    if budget:
        print(f"\n--------------------------------------------------------------")
        print(f"  CENTRALIZED LLM BUDGET & TELEMETRY BREAKDOWN")
        print(f"--------------------------------------------------------------")
        print(f"  Depth Profile:        {budget.get('depth')}")
        print(f"  Max LLM Calls:        {budget.get('max_llm_calls')}")
        print(f"  LLM Calls Used:       {budget.get('calls_used')} ({budget.get('remaining_calls')} remaining)")
        print(f"  Input Tokens Used:    {budget.get('input_tokens_used')} tokens")
        print(f"  Output Tokens Used:   {budget.get('output_tokens_used')} tokens")
        print(f"  Total Tokens Used:    {budget.get('total_tokens_used')} tokens")
        print(f"  Provider Fallbacks:   {budget.get('fallback_calls')}")
        print(f"  KG Extractions:       {budget.get('kg_extraction_calls')}")
        print(f"  ReAct Tool Turns:     {budget.get('react_calls')}")

    # Inspect Knowledge Graph memory stats
    kg_path = os.path.join("reports", "research_kg.json")
    if os.path.exists(kg_path):
        try:
            with open(kg_path, "r", encoding="utf-8") as kgf:
                kg_data = json.load(kgf)
                num_nodes = len(kg_data.get("nodes", []))
                num_edges = len(kg_data.get("links", []) or kg_data.get("edges", []))
                print(f"\n--------------------------------------------------------------")
                print(f"  KNOWLEDGE GRAPH MEMORY STATE")
                print(f"--------------------------------------------------------------")
                print(f"  Extracted Nodes:      {num_nodes}")
                print(f"  Extracted Relations:  {num_edges}")
        except Exception:
            pass

    print(f"==============================================================\n")

if __name__ == "__main__":
    inspect_latest_run()
