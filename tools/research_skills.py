import time
from .base_tool import BaseTool
import google.generativeai as genai
import config
from colorama import Fore, Style

class ResearchSkillsTool(BaseTool):
    """
    Advanced Academic Research Skills Toolkit.
    Provides 5 consolidated scientific skills merging 13 paper analysis functions.
    
    Input format: <skill_name> | <text_to_analyze>
    Available skills:
      - literature_critique : Key claims, Steel-man weakest argument, critical objections.
      - gap_finder          : Unresolved issues, novel testable research questions.
      - synthesis_drafter   : Clean notes/summaries, formulate synthesis, write Related Works draft.
      - concept_mapper      : Conceptual explanation with analogies, logical argument matrix.
      - academic_refinement : 4-sentence abstract rewrite, data interpretation, devil's advocate, brief.
    """

    SKILLS = {
        "literature_critique": (
            "You are an elite academic peer reviewer and critic. Analyze the following paper text.\n"
            "Perform three tasks:\n"
            "1. CLAIMS ANALYSIS: Identify 5 key claims the author makes, mapping what prior research they build on and the gaps they claim to fill.\n"
            "2. STEEL-MAN ARGUMENT: Identify the weakest argument or limitation in the paper and steel-man it - making the absolute best possible case for it even if the data does not fully support it.\n"
            "3. CRITICAL OBJECTIONS: Acting as a skeptical reviewer, raise 5 serious objections to the methodology and conclusions, citing exactly what is missing or unsupported."
        ),
        "gap_finder": (
            "You are an academic advisor and research planner. Based on the limitations, conclusions, and context of the provided paper text:\n"
            "1. RESEARCH GAPS: List 7 unanswered questions the authors themselves admit they could not resolve in their limitations or conclusion.\n"
            "2. FEASIBLE STUDY TOPICS: Generate 10 specific, testable, novel, and feasible research questions that could form the basis of a follow-up study."
        ),
        "synthesis_drafter": (
            "You are a principal researcher writing a paper. Based on the provided notes, paper summaries, or text:\n"
            "1. SYNTHESIS ANALYSIS: Clean up the input, identifying the common threads, direct contradictions, and formulating one central synthesis claim.\n"
            "2. RELATED WORKS DRAFT: Write a structured Related Works section (formatted in LaTeX-compatible markdown) that positions our research as filling the gaps that none of these papers address."
        ),
        "concept_mapper": (
            "You are a senior academic explaining research. Analyze the provided text:\n"
            "1. CONCEPT EXPLANATION: Explain the core technical concepts and methods as if to a researcher who has never seen this specific method, using one clear analogy.\n"
            "2. ARGUMENT CHAIN: Map the logical structure of the paper as a numbered argument chain from the initial premises to the final conclusion, flagging any logical leaps or unsubstantiated claims."
        ),
        "academic_refinement": (
            "You are an editor for prestigious journals. Analyze the provided abstract, findings, or thesis:\n"
            "1. ABSTRACT REWRITE: Rewrite the abstract so it clearly states the problem, method, finding, and implication in exactly 4 sentences, removing jargon and keeping it clear.\n"
            "2. DEVIL'S ADVOCATE & INTERPRETATION: List 3 ways to interpret the findings (identifying which is most defensible). Then, play devil's advocate and ruthlessly argue why the core thesis or interpretation could be completely wrong using evidence and logic.\n"
            "3. ONE-PAGE BRIEF: Compile a summary of the paper containing the core argument, methodology, 3 key findings, 2 limitations, and 1 future implication."
        )
    }

    def __init__(self):
        super().__init__(
            name="research_skills",
            description=(
                "Academic Research Skills Toolkit. Merges 13 paper analysis functions into 5 core skills. "
                "Input format: 'skill_name | text_to_analyze'. "
                "Skills: 'literature_critique', 'gap_finder', 'synthesis_drafter', 'concept_mapper', 'academic_refinement'."
            )
        )

    def run(self, input_text: str) -> str:
        # Parse inputs
        if "|" not in input_text:
            return (
                "Invalid format. Use: '<skill_name> | <text_to_analyze>'\n"
                f"Available skills: {list(self.SKILLS.keys())}"
            )
            
        skill_name, content = input_text.split("|", 1)
        skill_name = skill_name.strip().lower()
        content = content.strip()

        if skill_name not in self.SKILLS:
            return f"Unknown skill '{skill_name}'. Available: {list(self.SKILLS.keys())}"

        if not content:
            return "Error: Content to analyze is empty."

        system_instruction = self.SKILLS[skill_name]
        
        # Invoke LLM with retries
        max_retries = 5
        base_delay = 2.0

        if not config.GEMINI_API_KEY:
            return "[ResearchSkillsTool error: GEMINI_API_KEY is not set. Please add it to your .env file.]"

        genai.configure(api_key=config.GEMINI_API_KEY)
        
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel(
                    model_name=config.DEFAULT_MODEL,
                    system_instruction=system_instruction
                )
                response = model.generate_content(
                    content,
                    generation_config={
                        "temperature": 0.3,
                        "max_output_tokens": config.DEFAULT_MAX_TOKENS,
                    }
                )
                # Clean Unicode box lines/em-dashes for console safety
                return response.text.replace("\u2014", "-").replace("\u2500", "-")
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "rate limit" in err_str or "exhausted" in err_str
                
                if is_rate_limit and attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    time.sleep(sleep_time)
                else:
                    return f"[ResearchSkillsTool error: {e}]"
        return "[ResearchSkillsTool error: Max retries exceeded due to rate limits.]"
