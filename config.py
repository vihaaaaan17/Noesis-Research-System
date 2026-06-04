import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = "gemini-2.5-flash"
# Other current Gemini chat models:
#   "gemini-1.5-flash"
#   "gemini-2.5-flash"
#   "gemini-2.5-pro"
#   "gemini-3.1-pro-preview"

# Agent defaults
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7

# Debug
VERBOSE = True
