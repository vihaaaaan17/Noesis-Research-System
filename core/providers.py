"""
core/providers.py
---------------------------------------------------------------------
Clean Provider Abstraction Layer for MAS.

Defines:
  - LLMProvider (Abstract Base Class)
  - GroqProvider (Groq OpenAI-compatible REST API)
  - GeminiProvider (Google Generative AI SDK)
  - ProviderRegistry (Central manager for provider lookup)
  - call_llm_api() (Direct provider dispatcher)
  - call_with_fallback() (Provider fallback router for Stage 1 research)
---------------------------------------------------------------------
"""

import os
import re
import time
import requests
import unicodedata
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from colorama import Fore, Style, init

init(autoreset=True)


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


def build_canonical_messages(
    messages: Optional[List[Dict[str, Any]]] = None,
    prompt: Optional[str] = None,
    system_instruction: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Build standard canonical message list:
    [{"role": "system"/"user"/"assistant", "content": "..."}]
    """
    canonical = []
    if system_instruction:
        canonical.append({"role": "system", "content": str(system_instruction)})

    if messages:
        for msg in messages:
            role = msg.get("role", "user")
            if role not in ["system", "user", "assistant"]:
                role = "user"
            canonical.append({"role": role, "content": str(msg.get("content", ""))})
    elif prompt:
        canonical.append({"role": "user", "content": str(prompt)})

    return canonical


def compress_canonical_messages(messages: List[Dict[str, str]], max_msg_chars: int = 3000) -> List[Dict[str, str]]:
    """
    Compress canonical message list by truncating oversized message bodies
    while retaining system instructions and recent conversation state.
    """
    compressed = []
    for msg in messages:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))

        if role == "system" or len(content) <= max_msg_chars:
            compressed.append({"role": role, "content": content})
        else:
            prefix = content[:1500]
            suffix = content[-1200:]
            truncated_content = (
                f"{prefix}\n\n"
                f"[... Content compressed ({len(content)} chars) ...]\n\n"
                f"{suffix}"
            )
            compressed.append({"role": role, "content": truncated_content})

    return compressed


class LLMProvider(ABC):
    """Abstract Base Class for LLM Providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier string ('groq', 'gemini')."""
        pass

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        Execute API generation request.
        Returns dict with keys:
          success: bool
          text: str (if success)
          error: str (if failure)
          error_type: 'rate_limit' | 'context_too_large' | 'server_error' | 'auth_error' | 'other'
          status_code: int (optional)
        """
        pass


def extract_retry_delay(error_str: str) -> float:
    """Extract retry delay seconds from Groq or Gemini rate limit error messages."""
    if not error_str:
        return 0.0
    m = re.search(r'(?:try again in|retry in|seconds:)\s*([\d\.]+)', error_str, re.IGNORECASE)
    if m:
        try:
            val = float(m.group(1))
            return max(val, 2.0)
        except ValueError:
            pass
    return 0.0


class GroqProvider(LLMProvider):
    """Groq API Provider implementation with rate-limit parsing and model cascade."""

    @property
    def name(self) -> str:
        return "groq"

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: int = 60
    ) -> Dict[str, Any]:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return {
                "success": False,
                "error": "[Groq Error]: GROQ_API_KEY not configured.",
                "error_type": "auth_error",
                "status_code": 401
            }

        target_model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        if "/" in target_model or "groq" in target_model.lower() or "qwen" in target_model.lower() or "llama" in target_model.lower():
            groq_model = target_model
        else:
            groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        formatted_messages = []
        for msg in messages:
            formatted_messages.append({"role": msg.get("role", "user"), "content": str(msg.get("content", ""))})

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Candidate models for Groq execution (automatically cascades if a model returns 404)
        model_candidates = [
            groq_model,
            "qwen/qwen3.6-27b",
            "groq/compound",
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ]
        candidate_list = []
        for m in model_candidates:
            if m and m not in candidate_list:
                candidate_list.append(m)

        for attempt, model_to_try in enumerate(candidate_list):
            payload = {
                "model": model_to_try,
                "messages": formatted_messages,
                "temperature": temperature if temperature is not None else 0.7,
                "max_tokens": max_tokens if max_tokens is not None else 1500
            }

            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )

                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    content = sanitize_scientific_markdown(content)
                    return {"success": True, "text": content}

                err_msg = resp.text
                status = resp.status_code

                if status == 404 or "model_not_found" in err_msg.lower() or "does not exist" in err_msg.lower():
                    safe_print(f"[LLM Groq] Model '{model_to_try}' not found on Groq API (404). Cascading to next model...")
                    continue

                if status == 413 or ("context" in err_msg.lower() and "too large" in err_msg.lower()) or "maximum context length" in err_msg.lower():
                    error_type = "context_too_large"
                elif status == 429 or "rate_limit" in err_msg.lower() or "tpm" in err_msg.lower() or "rate limit" in err_msg.lower():
                    error_type = "rate_limit"
                elif status >= 500:
                    error_type = "server_error"
                else:
                    error_type = "other"

                if error_type == "rate_limit" and attempt < len(candidate_list) - 1:
                    retry_delay = extract_retry_delay(err_msg) or 4.0
                    safe_print(f"[LLM Groq] Rate limit/TPM hit on {model_to_try}. Waiting {retry_delay:.1f}s before trying next candidate...")
                    time.sleep(retry_delay + 0.5)
                    continue

                return {
                    "success": False,
                    "error": f"[Groq API Error {status}]: {err_msg}",
                    "error_type": error_type,
                    "status_code": status
                }

            except requests.exceptions.RequestException as e:
                if attempt < max_attempts - 1:
                    time.sleep(3.0)
                    continue
                return {
                    "success": False,
                    "error": f"[Groq API Exception]: {str(e)}",
                    "error_type": "server_error",
                    "status_code": 503
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"[Groq API Exception]: {str(e)}",
                    "error_type": "other",
                    "status_code": 500
                }

        return {"success": False, "error": "[Groq Error]: Max retries exhausted.", "error_type": "rate_limit"}


class GeminiProvider(LLMProvider):
    """Google Gemini SDK Provider implementation with Zero-Sleep Multi-Model Failover Cascade."""

    @property
    def name(self) -> str:
        return "gemini"

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        model_candidates: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: int = 60
    ) -> Dict[str, Any]:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {
                "success": False,
                "error": "[Gemini Error]: GEMINI_API_KEY not configured.",
                "error_type": "auth_error",
                "status_code": 401
            }

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
        except Exception as e:
            return {
                "success": False,
                "error": f"[Gemini Configuration Error]: {str(e)}",
                "error_type": "auth_error"
            }

        # Build prioritized list of model candidates for instant 0ms failover
        candidates = []
        if model:
            candidates.append(model)
        if model_candidates:
            for c in model_candidates:
                if c not in candidates:
                    candidates.append(c)

        default_cascade = ["gemini-3.1-flash-lite", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-2.5-flash", "gemini-3.7-flash"]
        for c in default_cascade:
            if c not in candidates:
                candidates.append(c)

        sys_parts = []
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))

            if role == "system":
                sys_parts.append(content)
            else:
                gemini_role = "model" if role == "assistant" else "user"
                if contents and contents[-1]["role"] == gemini_role:
                    contents[-1]["parts"][0] += f"\n\n{content}"
                else:
                    contents.append({"role": gemini_role, "parts": [content]})

        sys_instruction = "\n\n".join(sys_parts) if sys_parts else None

        if not contents:
            contents.append({"role": "user", "parts": ["Hello"]})

        last_error = ""
        last_error_type = "other"

        for idx, current_model in enumerate(candidates):
            try:
                mobj = genai.GenerativeModel(model_name=current_model, system_instruction=sys_instruction)
                gen_config = {
                    "temperature": temperature if temperature is not None else 0.7,
                    "max_output_tokens": max_tokens if max_tokens is not None else 4096
                }
                res = mobj.generate_content(contents, generation_config=gen_config)

                if hasattr(res, "text") and res.text:
                    clean_text = sanitize_scientific_markdown(res.text)
                    return {"success": True, "text": clean_text, "model_used": current_model}
                else:
                    last_error = "[Gemini API Error]: Empty text returned in response."
                    last_error_type = "empty_response"
            except Exception as e:
                err_str = str(e)
                last_error = err_str

                if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                    last_error_type = "rate_limit"
                elif "413" in err_str or "token" in err_str.lower() or "context" in err_str.lower():
                    last_error_type = "context_too_large"
                elif "404" in err_str or "not found" in err_str.lower():
                    last_error_type = "not_found"
                else:
                    last_error_type = "server_error"

                if idx < len(candidates) - 1:
                    next_model = candidates[idx + 1]
                    safe_print(f"[LLM Gemini Cascade] {current_model} hit ({last_error_type}). Instant failover (0ms) to {next_model}...")
                    continue

        return {"success": False, "error": f"[Gemini Cascade Error]: Exhausted all models. Last error: {last_error}", "error_type": last_error_type}



class ProviderRegistry:
    """Registry maintaining active provider instances."""

    def __init__(self):
        self._providers: Dict[str, LLMProvider] = {
            "groq": GroqProvider(),
            "gemini": GeminiProvider()
        }

    def get(self, name: str) -> LLMProvider:
        clean_name = (name or "").lower().strip()
        if clean_name in self._providers:
            return self._providers[clean_name]
        raise ValueError(f"Unknown provider '{name}'. Available: {list(self._providers.keys())}")


REGISTRY = ProviderRegistry()


def call_llm_api(
    messages: Optional[List[Dict[str, Any]]] = None,
    prompt: Optional[str] = None,
    system_instruction: Optional[str] = None,
    provider: str = "groq",
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None
) -> str:
    """
    Direct LLM Provider Dispatcher.
    Respects explicit `provider` parameter ('groq' or 'gemini').
    Does NOT silently convert requested Gemini into Groq!
    """
    canonical_messages = build_canonical_messages(
        messages=messages,
        prompt=prompt,
        system_instruction=system_instruction
    )

    prov = REGISTRY.get(provider)
    res = prov.generate(
        messages=canonical_messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    if res["success"]:
        return res["text"]

    return res.get("error", f"[{provider.capitalize()} API Error]")


def call_with_fallback(
    messages: Optional[List[Dict[str, Any]]] = None,
    prompt: Optional[str] = None,
    system_instruction: Optional[str] = None,
    primary_provider: str = "gemini",
    primary_model: Optional[str] = None,
    fallback_provider: str = "gemini",
    fallback_model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    budget: Optional[Any] = None,
    category: str = "general"
) -> str:
    """
    Centralized 100% Gemini Zero-Sleep Multi-Model Router.
    Pre-flight checks LLMBudgetManager, attempts generation via primary Gemini model,
    and instantly fails over (0ms delay) across candidate Gemini model quotas.
    """
    if budget is not None:
        check_res = budget.check_preflight(category=category)
        if not check_res["allowed"]:
            safe_print(f"[LLM Router] Budget preflight rejected: {check_res.get('reason')} ({check_res.get('detail')})")
            return f"[LLM Provider Failure]: Preflight budget rejected - {check_res.get('detail')}"

    canonical_messages = build_canonical_messages(
        messages=messages,
        prompt=prompt,
        system_instruction=system_instruction
    )

    # Candidate cascade from config or parameters
    cascade_models = []
    if primary_model:
        cascade_models.append(primary_model)
    if fallback_model and fallback_model not in cascade_models:
        cascade_models.append(fallback_model)

    default_cascade = ["gemini-3.1-flash-lite", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-2.5-flash", "gemini-3.7-flash"]
    for m in default_cascade:
        if m not in cascade_models:
            cascade_models.append(m)

    g_prov = REGISTRY.get("gemini")
    res = g_prov.generate(
        messages=canonical_messages,
        model=primary_model,
        model_candidates=cascade_models,
        temperature=temperature,
        max_tokens=max_tokens
    )

    if res["success"]:
        text = sanitize_scientific_markdown(res["text"])
        if budget is not None:
            in_tok = len(str(canonical_messages)) // 4
            out_tok = len(text) // 4
            budget.record_call(category=category, input_tokens=in_tok, output_tokens=out_tok, is_retry=False, is_fallback=False)
        return text

    return res.get("error", "[Gemini Cascade Failure]")


def sanitize_scientific_markdown(text: str) -> str:
    """
    Sanitize LaTeX formulas, Markdown formatting, and strip internal LLM reasoning preambles:
    1. Normalize Mathematical Alphanumeric Unicode Symbols (U+1D400-U+1D7FF) to standard ASCII.
    2. Strip internal LLM reasoning/thinking preambles (e.g., <think>...</think>, "Here's a thinking process: ...").
    3. Fix invalid non-standard LaTeX macros like \\left\\round ... \\right\\round -> \\text{round}\\left( ... \\right)
    4. Fix double comma artifacts in clamp functions.
    5. Clean up unclosed math block formatting.
    """
    if not text or not isinstance(text, str):
        return text or ""

    # 0. Normalize Mathematical Alphanumeric Unicode Symbols (U+1D400-U+1D7FF) to standard ASCII
    try:
        text = unicodedata.normalize("NFKC", text)
    except Exception:
        pass

    # 1. Strip closed <think>...</think> and <thought>...</thought> blocks
    text = re.sub(r"(?si)<think>.*?</think>", "", text)
    text = re.sub(r"(?si)<thought>.*?</thought>", "", text)

    # 2. Strip unclosed <think> or leading thinking preambles up to actual content
    text = re.sub(r"(?si)^\s*<think>.*?(?=\n+#|\n+[A-Z0-9][a-zA-Z0-9\s–—\-]{2,}\n|\n+Node\.js|\n+Revised Report|\Z)", "", text)
    text = re.sub(r"(?si)^\s*Here'?s\s+(?:a\s+)?thinking\s+process:.*?(?=\n+#|\n+[A-Z0-9][a-zA-Z0-9\s–—\-]{2,}\n|\n+Node\.js|\n+Revised Report|\Z)", "", text)
    text = re.sub(r"(?si)^\s*Thinking\s+Process:.*?(?=\n+#|\n+[A-Z0-9][a-zA-Z0-9\s–—\-]{2,}\n|\n+Node\.js|\n+Revised Report|\Z)", "", text)
    text = re.sub(r"(?si)^\s*Analyze\s+User\s+Input:.*?(?=\n+#|\n+[A-Z0-9][a-zA-Z0-9\s–—\-]{2,}\n|\n+Node\.js|\n+Revised Report|\Z)", "", text)

    # 3. Fix invalid \left\round / \right\round macros
    text = re.sub(r'\\left\\round\s*', r'\\text{round}\\left(', text)
    text = re.sub(r'\\right\\round\s*', r'\\right)', text)
    text = re.sub(r'\\round\b', r'\\text{round}', text)

    # 4. Fix double commas in clamped formulas: clamp(..., , 0, , 2^n - 1)
    text = re.sub(r',\s*,', ',', text)

    # 5. Normalize malformed model-generated LaTeX (escaped underscores, asterisk subscripts)
    text = re.sub(r'\\_\{([^}]+)\}', r'_{\1}', text)
    text = re.sub(r'\\_([a-zA-Z0-9]+)', r'_\1', text)
    text = re.sub(r'([a-zA-Z0-9\}])\s*\*\{([^}]+)\}', r'\1_{\2}', text)
    text = re.sub(r'([a-zA-Z0-9\}])\s*\*([a-zA-Z0-9]+)', r'\1_\2', text)

    return text.strip()


def _is_truncated_response(text: str) -> bool:
    """Detect if response string was truncated mid-sentence or mid-formula."""
    if not text or len(text) < 400:
        return False
    stripped = text.rstrip()

    # If text ends with terminal punctuation or closing brackets, it's complete
    if stripped[-1] in ['.', '!', '?', '}', ']', '`', ')', '"', "'"]:
        return False

    # Check for unclosed LaTeX display math ($$ ... $$)
    if stripped.count("$$") % 2 != 0:
        return True

    return True


def call_llm_api_multichunk(
    messages: Optional[List[Dict[str, Any]]] = None,
    prompt: Optional[str] = None,
    system_instruction: Optional[str] = None,
    provider: str = "gemini",
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = 4096,
    max_chunks: int = 4
) -> str:
    """
    Multi-chunk continuation loop.
    If generation cuts off mid-sentence, automatically issues internal continuation calls
    (up to `max_chunks`), joins text seamlessly, and sanitizes scientific LaTeX output.
    """
    canonical_messages = build_canonical_messages(
        messages=messages,
        prompt=prompt,
        system_instruction=system_instruction
    )

    full_output = ""
    current_messages = list(canonical_messages)

    for chunk_idx in range(max_chunks):
        res = call_llm_api(
            messages=current_messages,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if not res or res.startswith("[LLM"):
            if not full_output:
                return res
            break

        full_output += res

        if not _is_truncated_response(res):
            break

        safe_print(f"[LLM Multi-Chunk] Truncation detected on chunk {chunk_idx+1}/{max_chunks}. Sending continuation prompt...")

        current_messages.append({"role": "assistant", "content": res})
        current_messages.append({
            "role": "user",
            "content": f"Continue writing the research report starting EXACTLY from where you stopped. Do NOT repeat prior text. Prior text ended with:\n...{res[-250:]}"
        })

    return sanitize_scientific_markdown(full_output)

