import os
import time
import threading
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model configuration (Provider and Model are separate concepts)
# Groq: llama-3.3-70b-versatile (14,000 TPM) / llama-3.1-8b-instant (100,000 TPM)
GROQ_MODEL = os.getenv("GROQ_MODEL", os.getenv("DEFAULT_MODEL", "llama-3.3-70b-versatile"))

# Gemini: gemini-3.1-flash-lite
GEMINI_RESEARCH_MODEL = os.getenv("GEMINI_RESEARCH_MODEL", "gemini-3.1-flash-lite")
GEMINI_FINAL_MODEL = os.getenv("GEMINI_FINAL_MODEL", "gemini-3.1-flash-lite")

DEFAULT_FLASH_MODEL = GROQ_MODEL
DEFAULT_PRO_MODEL = GROQ_MODEL
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", GROQ_MODEL)

VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"

# Agent default output token budgets
DEFAULT_MAX_TOKENS = 1500            # Controlled output budget for Stage 1 research steps
DEFAULT_TEMPERATURE = 0.7
REPORT_SECTION_MAX_TOKENS = 8000     # Gemini report section output budget

# DEFAULT_MAX_TOKENS = 1500
# SYNTHESIS_MAX_TOKENS = 2500
# REPORT_SECTION_MAX_TOKENS = {
#     "abstract": 500,
#     "introduction": 1200,
#     "related_work": 1500,
#     "methodology": 1800,
#     "analysis": 2200,
#     "results": 1600,
#     "discussion": 1800,
#     "conclusion": 700,
# }


# Import Provider Abstraction Layer
from core.providers import (
    call_llm_api,
    call_with_fallback,
    safe_print,
    build_canonical_messages,
    compress_canonical_messages
)

# Global API Call Throttling (enforces 3.5s thread-safe gap to stay under 17 RPM)
_LAST_CALL_TIME = 0.0
MIN_CALL_INTERVAL = 3.5  # seconds
_API_LOCK = threading.Lock()

def throttle_api_call():
    global _LAST_CALL_TIME
    with _API_LOCK:
        now = time.time()
        elapsed = now - _LAST_CALL_TIME
        if elapsed < MIN_CALL_INTERVAL:
            time.sleep(MIN_CALL_INTERVAL - elapsed)
        _LAST_CALL_TIME = time.time()
