"""
core/report_generator.py
---------------------------------------------------------------------
Generalized Long-Form Generation Engine for MAS.

Supports multiple generation modes:
  * "answer"           - single direct LLM response
  * "paragraph"        - single coherent paragraph
  * "explanation"      - multi-paragraph explanation
  * "long_form"        - chunked sequential response with compact continuation
  * "research_paper"   - section-by-section academic report with dynamic outline & checkpoints
---------------------------------------------------------------------
"""

import os
import re
import json
import time
from typing import Dict, Any, List, Optional
from colorama import Fore, Style, init
import config

init(autoreset=True)


class LongFormGenerator:
    """
    Generalized Long-Form Generation Engine executing mode-based document writing
    with chunked generation, compact continuation state, and checkpointing.
    """

    DEFAULT_SECTION_BUDGETS = {
        "abstract":     500,
        "introduction": 1200,
        "background":   1500,
        "methodology":  1800,
        "analysis":     2000,
        "results":      1500,
        "discussion":   1500,
        "conclusion":   700,
        "references":   500,
    }

    def __init__(
        self,
        output_dir: str = "reports",
        model: Optional[str] = None,
        verbose: bool = True
    ):
        self.output_dir = output_dir
        self.model = model or config.GEMINI_FINAL_MODEL
        self.verbose = verbose
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(
        self,
        question: str,
        mode: str = "research_paper",
        research_doc: Optional[Dict[str, str]] = None,
        working_memory=None,
        long_term_memory=None,
        budget=None
    ) -> str:
        """
        Main entry point for document and text generation.
        Routes to specific mode handlers: answer, paragraph, explanation, long_form, research_paper.
        """
        mode_clean = mode.lower().strip()

        if mode_clean == "answer":
            return self._generate_answer(question, budget=budget)
        elif mode_clean == "paragraph":
            return self._generate_paragraph(question, budget=budget)
        elif mode_clean == "explanation":
            return self._generate_explanation(question, budget=budget)
        elif mode_clean in ("long_form", "longform"):
            return self._generate_long_form(question, research_doc, working_memory, long_term_memory, budget=budget)
        else:
            # Default to research_paper / research_report
            return self.generate_report(question, research_doc or {}, working_memory, long_term_memory, budget=budget)

    def generate_report(
        self,
        question: str,
        research_doc: Dict[str, str],
        working_memory=None,
        long_term_memory=None
    ) -> str:
        """
        Execute full research paper generation pipeline with outline, sections, and checkpoints.
        """
        run_id = self._generate_run_id(question)
        checkpoint_dir = os.path.join(self.output_dir, "sections", run_id)
        os.makedirs(checkpoint_dir, exist_ok=True)

        if self.verbose:
            config.safe_print(f"\n{Fore.CYAN}{'='*60}")
            config.safe_print(f"[REPORT] Initializing Long-Form Document Generator (mode='research_paper')")
            config.safe_print(f"[REPORT] Model: {self.model} | Checkpoints: {checkpoint_dir}")
            config.safe_print(f"{'='*60}{Style.RESET_ALL}")

        # Step 1: Outline Generation
        outline = self._get_or_create_outline(question, research_doc, checkpoint_dir)
        sections_spec = outline.get("sections", [])

        # Step 2 & 3: Generate Sections Independently
        generated_sections: List[Dict[str, str]] = []
        continuation_state = ""

        for idx, sec_info in enumerate(sections_spec, 1):
            sec_id = sec_info.get("id", f"section_{idx}")
            sec_title = sec_info.get("title", f"Section {idx}")
            sec_budget = sec_info.get("budget", config.REPORT_SECTION_MAX_TOKENS)

            sec_filename = f"{idx:02d}_{sec_id}.md"
            sec_path = os.path.join(checkpoint_dir, sec_filename)

            if os.path.exists(sec_path):
                with open(sec_path, "r", encoding="utf-8") as f:
                    sec_content = f.read()
                if self.verbose:
                    config.safe_print(f"[REPORT] Loaded checkpoint {idx}/{len(sections_spec)}: {sec_title}")
            else:
                if self.verbose:
                    config.safe_print(f"[REPORT] Generating section {idx}/{len(sections_spec)}: {sec_title}...")

                sec_content = self._generate_single_section_with_retry(
                    question=question,
                    section_info=sec_info,
                    research_doc=research_doc,
                    working_memory=working_memory,
                    long_term_memory=long_term_memory,
                    continuation_state=continuation_state,
                    max_tokens=sec_budget
                )

                # Validate output before saving checkpoint
                if not sec_content or sec_content.startswith("[Generation Error]"):
                    if self.verbose:
                        config.safe_print(f"{Fore.RED}[REPORT] Generation failed on '{sec_title}'. Skipping invalid checkpoint.{Style.RESET_ALL}")
                    sec_content = f"*[Section '{sec_title}' generation encountered a provider error]*"
                else:
                    with open(sec_path, "w", encoding="utf-8") as f:
                        f.write(sec_content)

            generated_sections.append({"title": sec_title, "content": sec_content})
            # Build compact continuation state (title + last 3 lines / key terms)
            continuation_state += f"\n- {sec_title}: {sec_content[:200]}... [Ends with: '{sec_content[-150:]}']"

        # Step 4: Long-Form Document Assembly
        if self.verbose:
            config.safe_print(f"[REPORT] Assembling final document ({len(generated_sections)} sections)...")

        assembled_report = self._assemble_document(outline.get("title", question), generated_sections)

        # Write final output report to reports/report_<run_id>.md
        report_filename = f"report_{run_id}.md"
        report_file_path = os.path.join(self.output_dir, report_filename)
        with open(report_file_path, "w", encoding="utf-8") as f:
            f.write(assembled_report)

        if self.verbose:
            config.safe_print(f"{Fore.GREEN}[REPORT] Saved final report to: {report_file_path}{Style.RESET_ALL}\n")

        return assembled_report

    # -------------------------------------------------------------
    # Direct Mode Handlers (Short Circuits)
    # -------------------------------------------------------------

    def _generate_answer(self, question: str, budget=None) -> str:
        """Single LLM response for answer mode."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise scientific assistant. Give a clear, direct answer.\n"
                    "CONSTRAINTS:\n"
                    "1. DO NOT output any emojis.\n"
                    "2. DO NOT output internal thinking processes, scratchpads, or 'Here's a thinking process:' preambles.\n"
                    "3. Use LaTeX ($...$ or $$...$$) for equations."
                )
            },
            {"role": "user", "content": question}
        ]
        return config.call_with_fallback(messages=messages, primary_model=self.model, max_tokens=2048, budget=budget, category="generation")

    def _generate_paragraph(self, question: str, budget=None) -> str:
        """Single paragraph response mode."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Write exactly one clear, coherent, information-dense paragraph answering the prompt.\n"
                    "CONSTRAINTS:\n"
                    "1. DO NOT output any emojis.\n"
                    "2. DO NOT output internal thinking processes, scratchpads, or 'Here's a thinking process:' preambles.\n"
                    "3. Use LaTeX ($...$ or $$...$$) for equations."
                )
            },
            {"role": "user", "content": question}
        ]
        return config.call_with_fallback(messages=messages, primary_model=self.model, max_tokens=1000, budget=budget, category="generation")

    def _generate_explanation(self, question: str, budget=None) -> str:
        """Multi-paragraph explanation mode."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Provide a clear, multi-paragraph technical explanation with key equations and physical concepts.\n"
                    "CONSTRAINTS:\n"
                    "1. DO NOT output any emojis (e.g. 🔹, 🧠, 📌, ✅, ⚠️, 📝, etc.).\n"
                    "2. DO NOT output internal thinking processes, scratchpads, or 'Here's a thinking process:' preambles. Output ONLY clean text.\n"
                    "3. Use standard LaTeX ($...$ or $$...$$) for equations."
                )
            },
            {"role": "user", "content": question}
        ]
        return config.call_with_fallback(messages=messages, primary_model=self.model, max_tokens=4096, budget=budget, category="generation")

    def _generate_long_form(self, question: str, research_doc=None, working_memory=None, long_term_memory=None, budget=None) -> str:
        """Generic chunk-based long-form response mode."""
        chunks_plan = [
            {"title": "1. Overview & Fundamentals", "reqs": "Introduce problem and core concepts"},
            {"title": "2. Detailed Analysis", "reqs": "Explain key mechanisms, equations, and trade-offs"},
            {"title": "3. Conclusions & Summary", "reqs": "Synthesize results and practical conclusions"}
        ]
        results = []
        continuation_state = ""
        for chunk in chunks_plan:
            context = f"Previous Context:\n{continuation_state}" if continuation_state else ""
            messages = [
                {"role": "system", "content": f"Write chunk '{chunk['title']}' of a technical document. {chunk['reqs']}. Continue naturally without repeating introduction headers."},
                {"role": "user", "content": f"Question: {question}\n\n{context}"}
            ]
            chunk_text = config.call_with_fallback(messages=messages, primary_model=self.model, max_tokens=1000, budget=budget, category="generation")
            results.append(f"## {chunk['title']}\n\n{chunk_text}")
            continuation_state += f"\n- {chunk['title']}: {chunk_text[:150]}..."
        return "\n\n".join(results)

    # -------------------------------------------------------------
    # Helpers & Pipeline Utilities
    # -------------------------------------------------------------

    def _get_or_create_outline(
        self,
        question: str,
        research_doc: Dict[str, str],
        checkpoint_dir: str
    ) -> Dict[str, Any]:
        """Generate or load structured outline JSON."""
        outline_path = os.path.join(checkpoint_dir, "outline.json")
        if os.path.exists(outline_path):
            try:
                with open(outline_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        if self.verbose:
            config.safe_print("[REPORT] Generating dynamic report outline via Gemini...")

        synthesis_text = research_doc.get("synthesis") or research_doc.get("understand", "")
        prompt = (
            f"You are a scientific report architect. Create a structured outline for a publication-quality "
            f"research paper answering the following question:\n\n"
            f"Research Question: {question}\n\n"
            f"Research Findings Overview:\n{synthesis_text[:3000]}\n\n"
            f"Return ONLY a valid JSON object in this format:\n"
            f"{{\n"
            f'  "title": "Descriptive Academic Title",\n'
            f'  "sections": [\n'
            f'    {{"id": "abstract", "title": "Abstract", "requirements": "150-word summary", "budget": 500}},\n'
            f'    {{"id": "introduction", "title": "1. Introduction", "requirements": "Background and objectives", "budget": 1200}},\n'
            f'    {{"id": "background", "title": "2. Background & Related Work", "requirements": "Literature review and governing equations", "budget": 1500}},\n'
            f'    {{"id": "methodology", "title": "3. Methodology & Derivations", "requirements": "Rigorous step-by-step math derivations and LaTeX equations", "budget": 1800}},\n'
            f'    {{"id": "results", "title": "4. Numerical Results & Analysis", "requirements": "Calculated values with units and engineering notation", "budget": 1500}},\n'
            f'    {{"id": "discussion", "title": "5. Discussion & Engineering Assessment", "requirements": "Constraints, physical sanity checks, trade-offs", "budget": 1500}},\n'
            f'    {{"id": "conclusion", "title": "6. Conclusion", "requirements": "Summary of findings and future outlook", "budget": 700}},\n'
            f'    {{"id": "references", "title": "References", "requirements": "List papers, authors, years", "budget": 500}}\n'
            f'  ]\n'
            f"}}\n"
        )

        res_text = config.call_llm_api(
            prompt=prompt,
            provider="gemini",
            model=self.model,
            temperature=0.2,
            max_tokens=2048
        )

        outline_json = self._parse_json_from_response(res_text, question)
        with open(outline_path, "w", encoding="utf-8") as f:
            json.dump(outline_json, f, indent=2)

        return outline_json

    def build_section_context(self, section_id: str, research_doc: Dict[str, str]) -> str:
        """Construct targeted section-specific research context."""
        sec_clean = section_id.lower()

        if "abstract" in sec_clean:
            return f"QUESTION:\n{research_doc.get('question')}\n\nSYNTHESIS:\n{research_doc.get('synthesis')[:2500]}"
        elif "intro" in sec_clean:
            return f"QUESTION:\n{research_doc.get('question')}\n\nUNDERSTANDING:\n{research_doc.get('understand')[:2000]}\n\nLITERATURE:\n{research_doc.get('literature')[:2000]}"
        elif "background" in sec_clean or "related" in sec_clean:
            return f"LITERATURE FINDINGS:\n{research_doc.get('literature')[:4000]}"
        elif "method" in sec_clean or "math" in sec_clean or "analysis" in sec_clean:
            return f"MATHEMATICAL DERIVATIONS:\n{research_doc.get('mathematics')[:4500]}\n\nLITERATURE EQUATIONS:\n{research_doc.get('literature')[:1500]}"
        elif "result" in sec_clean or "num" in sec_clean or "compute" in sec_clean:
            return f"NUMERICAL RESULTS:\n{research_doc.get('computation')[:3500]}\n\nENGINEERING CHECKS:\n{research_doc.get('engineering')[:2000]}"
        elif "discuss" in sec_clean or "engine" in sec_clean:
            return f"ENGINEERING ASSESSMENT:\n{research_doc.get('engineering')[:3000]}\n\nREVIEW FEEDBACK:\n{research_doc.get('review')[:2500]}"
        elif "conclus" in sec_clean:
            return f"QUESTION:\n{research_doc.get('question')}\n\nSYNTHESIS:\n{research_doc.get('synthesis')[:3000]}"
        elif "refer" in sec_clean:
            return f"LITERATURE PAPERS:\n{research_doc.get('literature')[:3500]}"
        else:
            return f"RESEARCH SYNTHESIS:\n{research_doc.get('synthesis')[:4000]}"

    def _generate_single_section_with_retry(
        self,
        question: str,
        section_info: Dict[str, Any],
        research_doc: Dict[str, str],
        working_memory=None,
        long_term_memory=None,
        continuation_state: str = "",
        max_tokens: int = 1500,
        max_retries: int = 4
    ) -> str:
        """
        Generate a single section using Gemini Provider with parsed retry sleep,
        fallback, and zero fake content fallback.
        """
        from core.providers import extract_retry_delay

        sec_id = section_info.get("id", "section")
        sec_title = section_info.get("title", "Section")
        sec_reqs = section_info.get("requirements", "")

        sec_context = self.build_section_context(sec_id, research_doc)

        system_instruction = (
            "You are an expert technical writer and scientific researcher. "
            "Write the specified section of a publication-quality research paper in Markdown format.\n\n"
            "Rules:\n"
            "  * Format ALL mathematical equations in LaTeX (inline: $...$ or display: $$...$$)\n"
            "  * Define every variable on first use\n"
            "  * Preserve all numerical values, units, paper citations, author names, and equations\n"
            "  * Write in formal prose - do not use bullet points for body narrative\n"
            "  * Do NOT repeat the section title as a top heading - output the section body prose directly"
        )

        prompt = (
            f"Research Question: {question}\n\n"
            f"Target Section: {sec_title}\n"
            f"Section Requirements: {sec_reqs}\n\n"
            f"--- RELEVANT RESEARCH MATERIAL ---\n"
            f"{sec_context}\n\n"
            f"--- CONTINUATION CONTEXT ---\n"
            f"{continuation_state}\n\n"
            f"Write the complete, unabridged prose for section '{sec_title}' now."
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ]

        # 1. Try Gemini Provider with parsed retry sleep
        for attempt in range(max_retries):
            res = config.call_llm_api(
                messages=messages,
                provider="gemini",
                model=self.model,
                temperature=0.3,
                max_tokens=max_tokens
            )

            if res and not res.startswith("[") and not "API Error" in res and not "Exception" in res:
                return res.strip()

            retry_delay = extract_retry_delay(res) or (3.0 * (attempt + 1))
            if attempt < max_retries - 1:
                if self.verbose:
                    config.safe_print(f"{Fore.YELLOW}[REPORT] Gemini rate limit on '{sec_title}'. Waiting {retry_delay:.1f}s... (Attempt {attempt+1}/{max_retries}){Style.RESET_ALL}")
                time.sleep(retry_delay + 1.0)

        # 2. Emergency Safeguard Fallback to Gemini Flash Workhorse Cascade
        if self.verbose:
            config.safe_print(f"{Fore.YELLOW}[REPORT] Primary model unavailable on '{sec_title}'. Executing Gemini Flash cascade fallback...{Style.RESET_ALL}")

        gemini_fb_res = config.call_llm_api(
            messages=messages,
            provider="gemini",
            model="gemini-3.1-flash-lite",
            temperature=0.3,
            max_tokens=4096
        )

        if gemini_fb_res and not gemini_fb_res.startswith("[") and not "API Error" in gemini_fb_res:
            return gemini_fb_res.strip()

        # 3. Final Fallback to gemini-2.0-flash-lite Model
        fallback_res = config.call_llm_api(
            messages=messages,
            provider="gemini",
            model="gemini-2.0-flash-lite",
            temperature=0.3,
            max_tokens=4096
        )

        if fallback_res and not fallback_res.startswith("[") and not "API Error" in fallback_res:
            return fallback_res.strip()

        return f"[Generation Error]: Failed to generate section '{sec_title}' after all retries and fallbacks."

    def _assemble_document(self, title: str, sections: List[Dict[str, str]]) -> str:
        """Assemble generated sections into the final markdown document."""
        from core.providers import sanitize_scientific_markdown

        doc_parts = [f"# {title}\n"]
        for sec in sections:
            sec_title = sec["title"]
            sec_body = sec["content"]

            if not sec_body.lstrip().startswith("#"):
                doc_parts.append(f"## {sec_title}\n\n{sec_body}")
            else:
                doc_parts.append(sec_body)

        assembled = "\n\n".join(doc_parts)
        return sanitize_scientific_markdown(assembled)

    def _parse_json_from_response(self, text: str, fallback_title: str) -> Dict[str, Any]:
        """Extract clean JSON object from LLM response."""
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass

        return {
            "title": f"Research Report: {fallback_title}",
            "sections": [
                {"id": "abstract", "title": "Abstract", "requirements": "Summary", "budget": 500},
                {"id": "introduction", "title": "1. Introduction", "requirements": "Intro", "budget": 1200},
                {"id": "background", "title": "2. Background / Theory", "requirements": "Theory", "budget": 1500},
                {"id": "methodology", "title": "3. Methodology & Derivations", "requirements": "Derivations", "budget": 1800},
                {"id": "results", "title": "4. Results & Analysis", "requirements": "Results", "budget": 1500},
                {"id": "discussion", "title": "5. Discussion & Constraints", "requirements": "Discussion", "budget": 1500},
                {"id": "conclusion", "title": "6. Conclusion", "requirements": "Conclusion", "budget": 700},
                {"id": "references", "title": "References", "requirements": "Citations", "budget": 500}
            ]
        }

    def _generate_run_id(self, question: str) -> str:
        """Create a filesystem-friendly ID from prompt."""
        clean = re.sub(r"[^a-zA-Z0-9]+", "_", question.lower()).strip("_")
        return f"{clean[:30]}_{int(time.time())}"


# Backward compatibility class alias
class ReportGenerator(LongFormGenerator):
    """
    Subclass alias of LongFormGenerator for backward compatibility with existing imports.
    """
    pass
