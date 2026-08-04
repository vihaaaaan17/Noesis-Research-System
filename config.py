import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DEFAULT_FLASH_MODEL = "llama-3.1-8b-instant"
DEFAULT_PRO_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", DEFAULT_PRO_MODEL)

# Agent defaults
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7

# Debug
import threading

# Global API Call Throttling (enforces 3.5s thread-safe gap to stay under 17 RPM)
_LAST_CALL_TIME = 0.0
MIN_CALL_INTERVAL = 3.5  # seconds (guarantees <18 RPM across all concurrent threads)
_API_LOCK = threading.Lock()

def throttle_api_call():
    global _LAST_CALL_TIME
    with _API_LOCK:
        now = time.time()
        elapsed = now - _LAST_CALL_TIME
        if elapsed < MIN_CALL_INTERVAL:
            time.sleep(MIN_CALL_INTERVAL - elapsed)
        _LAST_CALL_TIME = time.time()

def safe_print(*args, **kwargs):
    """Console print helper resilient to Windows cp1252 UnicodeEncodeErrors."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = []
        for arg in args:
            if isinstance(arg, str):
                safe_args.append(arg.encode("ascii", errors="backslashreplace").decode("ascii"))
            else:
                safe_args.append(arg)
        print(*safe_args, **kwargs)

def call_llm_api(
    messages: list[dict] = None,
    prompt: str = None,
    system_instruction: str = None,
    model: str = None,
    temperature: float = None,
    max_tokens: int = None
) -> str:
    """
    Universal LLM call dispatcher:
    Primary: Groq API (llama-3.3-70b-versatile / llama-3.1-8b-instant)
    Fallback: Google Gemini API (if GROQ_API_KEY not present)
    """
    import re
    temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
    max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
    target_model = model or DEFAULT_MODEL

    # Primary path: Groq API
    if GROQ_API_KEY:
        groq_model = target_model
        # Use llama-3.3-70b-versatile for high TPM capacity (30,000 TPM)
        if any(m in target_model.lower() for m in ["llama", "mixtral", "gemma"]):
            groq_model = target_model
        else:
            groq_model = "llama-3.3-70b-versatile"

        formatted_messages = []
        if system_instruction:
            formatted_messages.append({"role": "system", "content": system_instruction})

        if messages:
            for msg in messages:
                role = msg.get("role", "user")
                if role == "assistant":
                    role = "assistant"
                elif role == "system" and not system_instruction:
                    role = "system"
                elif role not in ["system", "user", "assistant"]:
                    role = "user"
                formatted_messages.append({"role": role, "content": str(msg.get("content", ""))})
        elif prompt:
            formatted_messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": groq_model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        max_retries = 6
        base_delay = 1.5

        for attempt in range(max_retries):
            try:
                throttle_api_call()
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    err_msg = resp.text
                    if (resp.status_code in [413, 429] or "rate_limit" in err_msg.lower() or "tpm" in err_msg.lower()) and attempt < max_retries - 1:
                        # Rotate model bucket on rate limit to utilize separate 30 RPM quotas
                        if payload["model"] == "llama-3.3-70b-versatile":
                            payload["model"] = "llama-3.1-8b-instant"
                        else:
                            payload["model"] = "llama-3.3-70b-versatile"

                        match = re.search(r"(?:try again in|retry in)\s*([0-9\.]+)s", err_msg, re.IGNORECASE)
                        if match:
                            sleep_time = float(match.group(1)) + 0.5
                        else:
                            sleep_time = max(base_delay * (2 ** attempt), 2.5)

                        if len(payload["messages"]) > 2 and (resp.status_code == 413 or "tokens" in err_msg.lower()):
                            # Trim context window to stay within TPM budget
                            payload["messages"] = [payload["messages"][0]] + payload["messages"][-2:]

                        if VERBOSE:
                            safe_print(f"[Groq API] Rate limit hit. Switched to '{payload['model']}', waiting {sleep_time:.1f}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(sleep_time)
                        continue
                    return f"[Groq API Error {resp.status_code}]: {err_msg}"
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                return f"[Groq API Exception]: {str(e)}"

    # Fallback path: Gemini API
    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model_name = "gemini-flash-latest" if "flash" in target_model.lower() else "gemini-2.5-flash"
        
        contents = []
        sys_inst = system_instruction
        if messages:
            for m in messages:
                if m["role"] == "system":
                    sys_inst = m["content"]
                else:
                    r = "model" if m["role"] == "assistant" else "user"
                    contents.append({"role": r, "parts": [m["content"]]})
        elif prompt:
            contents.append({"role": "user", "parts": [prompt]})

        try:
            mobj = genai.GenerativeModel(model_name=gemini_model_name, system_instruction=sys_inst)
            res = mobj.generate_content(contents, generation_config={"temperature": temperature, "max_output_tokens": max_tokens})
            return res.text
        except Exception as e:
            return f"[Gemini Fallback Error]: {str(e)}"

    return "[LLM Error]: Neither GROQ_API_KEY nor GEMINI_API_KEY is configured in your environment."
