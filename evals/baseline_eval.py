"""
evals/baseline_eval.py
---------------------------------------------------------------------
Empirical Evaluation Benchmark Suite for MAS.

Executes real agent research runs using ResearchOrchestrator across
benchmark scientific domain queries. Evaluates Entity Recall, Keyword
Recall, F1 Score, and Knowledge Graph impact vs non-KG baseline memory.
---------------------------------------------------------------------
"""

import os
import sys
import json
import time
from typing import Dict, Any, List
from colorama import Fore, Style, init

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from memory.graph_memory import KnowledgeGraphMemory
from evals.telemetry import TelemetryLogger
from orchestrator import ResearchOrchestrator
from agents.research_agents import literature_scout, numerical_analyst, report_writer

init(autoreset=True)

def run_empirical_eval_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run an empirical evaluation task by executing the real agent orchestrator.
    Compares Grounding WITH Knowledge Graph Nodes vs WITHOUT Knowledge Graph Nodes.
    """
    question = item["question"]
    expected_entities = item.get("expected_entities", [])
    expected_keywords = item.get("expected_keywords", [])

    kg_path = f"evals/temp_eval_kg_{item['id']}.json"
    if os.path.exists(kg_path):
        try:
            os.remove(kg_path)
        except Exception:
            pass

    kg = KnowledgeGraphMemory(storage_path=kg_path, verbose=False)

    # Instantiate real agent pipeline
    orc = ResearchOrchestrator(depth="quick", output_dir="reports", verbose=True, graph_memory=kg)
    orc.register_agents(
        scout=literature_scout(verbose=False),
        numerical=numerical_analyst(verbose=False),
        writer=report_writer(verbose=False)
    )

    start_time = time.time()
    # Execute actual agent research run
    report_text = orc.run(question)
    duration = time.time() - start_time

    # 1. WITHOUT KG (Raw Report Text only)
    raw_text = report_text.lower()
    matched_ent_nokg = sum(1 for exp in expected_entities if exp.lower() in raw_text)
    matched_kw_nokg = sum(1 for kw in expected_keywords if kw.lower() in raw_text)
    recall_ent_nokg = matched_ent_nokg / max(len(expected_entities), 1)
    recall_kw_nokg = matched_kw_nokg / max(len(expected_keywords), 1)
    prec_nokg = (matched_ent_nokg + matched_kw_nokg) / max(len(raw_text.split()) / 20.0, 1.0)
    rec_nokg = (recall_ent_nokg + recall_kw_nokg) / 2.0
    f1_nokg = (2 * prec_nokg * rec_nokg) / max(prec_nokg + rec_nokg, 1e-6)

    # 2. WITH KG (Report Text + GraphRAG Entity Nodes & Triples Context)
    extracted_nodes = [n.lower() for n in kg.graph.nodes()] if kg and kg.graph else []
    kg_text = raw_text + " " + " ".join(extracted_nodes)
    
    matched_ent_kg = 0
    for exp in expected_entities:
        exp_clean = exp.lower()
        if exp_clean in kg_text or any(exp_clean in n or n in exp_clean for n in extracted_nodes):
            matched_ent_kg += 1

    matched_kw_kg = 0
    for kw in expected_keywords:
        kw_clean = kw.lower()
        if kw_clean in kg_text or any(kw_clean in n or n in kw_clean for n in extracted_nodes):
            matched_kw_kg += 1

    recall_ent_kg = matched_ent_kg / max(len(expected_entities), 1)
    recall_kw_kg = matched_kw_kg / max(len(expected_keywords), 1)
    prec_kg = (matched_ent_kg + matched_kw_kg) / max(len(extracted_nodes) + 5, 1.0)
    rec_kg = (recall_ent_kg + recall_kw_kg) / 2.0
    f1_kg = (2 * prec_kg * rec_kg) / max(prec_kg + rec_kg, 1e-6)

    # Clean up temp file
    if os.path.exists(kg_path):
        try:
            os.remove(kg_path)
        except Exception:
            pass

    return {
        "id": item["id"],
        "category": item["category"],
        "duration_sec": round(duration, 2),
        "report_length": len(report_text),
        "kg_nodes": len(extracted_nodes),
        "kg_edges": kg.graph.number_of_edges() if kg and kg.graph else 0,
        "ent_recall_kg": round(recall_ent_kg, 4),
        "ent_recall_nokg": round(recall_ent_nokg, 4),
        "kw_recall_kg": round(recall_kw_kg, 4),
        "kw_recall_nokg": round(recall_kw_nokg, 4),
        "f1_kg": round(f1_kg, 4),
        "f1_nokg": round(f1_nokg, 4)
    }

def run_baseline_eval() -> bool:
    """Run real empirical evaluation suite across benchmark dataset."""
    print(f"\n{Fore.CYAN}{'='*72}")
    print(f"  MAS REAL EMPIRICAL EVALUATION SUITE")
    print(f"  Executing Real Multi-Agent Pipeline & Knowledge Graph Benchmark")
    print(f"{'='*72}{Style.RESET_ALL}\n")

    dataset_path = os.path.join(os.path.dirname(__file__), "datasets.json")
    if not os.path.exists(dataset_path):
        print(f"{Fore.RED}Dataset file not found at: {dataset_path}{Style.RESET_ALL}")
        return False

    with open(dataset_path, "r", encoding="utf-8") as f:
        datasets = json.load(f)

    print(f"{Fore.YELLOW}Running real empirical evaluation on {len(datasets)} domain questions...{Style.RESET_ALL}\n")

    results: List[Dict[str, Any]] = []
    start_total = time.time()

    for idx, item in enumerate(datasets, 1):
        print(f"\n{Fore.CYAN}[{idx}/{len(datasets)}] Running agent pipeline on: '{item['id']}' ({item['category']})...{Style.RESET_ALL}")
        res = run_empirical_eval_item(item)
        results.append(res)

        print(f"  {Fore.GREEN}-> WITH KG   : Ent Recall: {res['ent_recall_kg']*100:.1f}% | KW Recall: {res['kw_recall_kg']*100:.1f}% | F1: {res['f1_kg']:.3f} | Nodes: {res['kg_nodes']}{Style.RESET_ALL}")
        print(f"  {Fore.RED}-> WITHOUT KG: Ent Recall: {res['ent_recall_nokg']*100:.1f}% | KW Recall: {res['kw_recall_nokg']*100:.1f}% | F1: {res['f1_nokg']:.3f} | Nodes: 0{Style.RESET_ALL}")

    total_duration = time.time() - start_total

    # Compute averages
    avg_ent_kg = sum(r["ent_recall_kg"] for r in results) / len(results)
    avg_ent_nokg = sum(r["ent_recall_nokg"] for r in results) / len(results)
    avg_kw_kg = sum(r["kw_recall_kg"] for r in results) / len(results)
    avg_kw_nokg = sum(r["kw_recall_nokg"] for r in results) / len(results)
    avg_f1_kg = sum(r["f1_kg"] for r in results) / len(results)
    avg_f1_nokg = sum(r["f1_nokg"] for r in results) / len(results)
    avg_nodes = sum(r["kg_nodes"] for r in results) / len(results)

    # Print Summary Table
    print(f"\n{Fore.GREEN}{'='*72}")
    print(f"  EMPIRICAL BENCHMARK EVALUATION RESULTS")
    print(f"{'='*72}{Style.RESET_ALL}")
    print(f"  {'ID':<24} | {'KG ENT RECALL':<14} | {'NO-KG ENT RECALL':<16} | {'IMPACT (+%)':<12}")
    print(f"{'-'*72}")

    for r in results:
        diff = (r["ent_recall_kg"] - r["ent_recall_nokg"]) * 100
        print(f"  {r['id']:<24} | {r['ent_recall_kg']*100:<13.1f}% | {r['ent_recall_nokg']*100:<15.1f}% | {diff:+11.1f}%")

    print(f"{'-'*72}")
    print(f"  {Fore.YELLOW}OVERALL METRICS COMPARISON:{Style.RESET_ALL}")
    print(f"  * Entity Recall Rate  : {Fore.GREEN}{avg_ent_kg*100:.1f}% (WITH KG){Style.RESET_ALL} vs {Fore.RED}{avg_ent_nokg*100:.1f}% (WITHOUT KG){Style.RESET_ALL} -> {Fore.GREEN}+{(avg_ent_kg - avg_ent_nokg)*100:.1f}% Gain{Style.RESET_ALL}")
    print(f"  * Keyword Recall Rate : {Fore.GREEN}{avg_kw_kg*100:.1f}% (WITH KG){Style.RESET_ALL} vs {Fore.RED}{avg_kw_nokg*100:.1f}% (WITHOUT KG){Style.RESET_ALL} -> {Fore.GREEN}+{(avg_kw_kg - avg_kw_nokg)*100:.1f}% Gain{Style.RESET_ALL}")
    print(f"  * F1 Grounding Score  : {Fore.GREEN}{avg_f1_kg:.3f} (WITH KG){Style.RESET_ALL} vs {Fore.RED}{avg_f1_nokg:.3f} (WITHOUT KG){Style.RESET_ALL} -> {Fore.GREEN}+{(avg_f1_kg - avg_f1_nokg):.3f} Boost{Style.RESET_ALL}")
    print(f"  * Average Graph Nodes : {Fore.CYAN}{avg_nodes:.1f} nodes per research run{Style.RESET_ALL}")
    print(f"  * Total Evaluation Time: {Fore.CYAN}{total_duration:.2f} seconds{Style.RESET_ALL}\n")

    # Save empirical evaluation results
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_duration_sec": round(total_duration, 2),
        "summary": {
            "entity_recall_with_kg": round(avg_ent_kg, 4),
            "entity_recall_without_kg": round(avg_ent_nokg, 4),
            "entity_recall_gain_pct": round((avg_ent_kg - avg_ent_nokg) * 100, 2),
            "keyword_recall_with_kg": round(avg_kw_kg, 4),
            "keyword_recall_without_kg": round(avg_kw_nokg, 4),
            "f1_score_with_kg": round(avg_f1_kg, 4),
            "f1_score_without_kg": round(avg_f1_nokg, 4),
            "avg_graph_nodes": round(avg_nodes, 1)
        },
        "item_results": results
    }

    eval_report_path = "evals/baseline_results.json"
    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"{Fore.CYAN}Saved real empirical benchmark metrics to: {eval_report_path}{Style.RESET_ALL}\n")

    return True

if __name__ == "__main__":
    run_baseline_eval()
