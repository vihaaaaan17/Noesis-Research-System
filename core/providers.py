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
    """Google Gemini SDK Provider implementation with rate limit auto-sleep."""

    @property
    def name(self) -> str:
        return "gemini"

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
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

        target_model = model or os.getenv("GEMINI_RESEARCH_MODEL", "gemini-3.1-flash-lite")
        gemini_model_name = target_model if "gemini" in target_model.lower() else "gemini-3.1-flash-lite"

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

        max_attempts = 3
        current_model = gemini_model_name

        for attempt in range(max_attempts):
            try:
                mobj = genai.GenerativeModel(model_name=current_model, system_instruction=sys_instruction)
                gen_config = {
                    "temperature": temperature if temperature is not None else 0.7,
                    "max_output_tokens": max_tokens if max_tokens is not None else 4096
                }
                res = mobj.generate_content(contents, generation_config=gen_config)

                if hasattr(res, "text") and res.text:
                    return {"success": True, "text": res.text}
                else:
                    return {
                        "success": False,
                        "error": "[Gemini API Error]: Empty text returned in response.",
                        "error_type": "other"
                    }
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                    err_type = "rate_limit"
                elif "413" in err_str or "token" in err_str.lower() or "context" in err_str.lower():
                    err_type = "context_too_large"
                else:
                    err_type = "server_error"

                if err_type == "rate_limit" and attempt < max_attempts - 1:
                    retry_delay = extract_retry_delay(err_str) or 4.0
                    safe_print(f"[LLM Gemini] Rate limit hit on {current_model}. Waiting {retry_delay:.1f}s before retry {attempt+1}/{max_attempts}...")
                    time.sleep(retry_delay)
                    if current_model != "gemini-3.6-flash":
                        current_model = "gemini-3.6-flash"
                    continue

                return {
                    "success": False,
                    "error": f"[Gemini API Exception]: {err_str}",
                    "error_type": err_type
                }


        return {"success": False, "error": "[Gemini Error]: Max retries exhausted.", "error_type": "rate_limit"}



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
    primary_provider: str = "groq",
    primary_model: Optional[str] = None,
    fallback_provider: str = "gemini",
    fallback_model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    budget: Optional[Any] = None,
    category: str = "general"
) -> str:
    """
    Stage 1 Research Centralized Fallback Router.
    Pre-flight checks LLMBudgetManager, tries primary provider, and fails over to
    fallback provider on rate limits while updating global run token and call telemetry.
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

    # 1. Try Primary Provider
    p_prov = REGISTRY.get(primary_provider)
    res = p_prov.generate(
        messages=canonical_messages,
        model=primary_model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    if res["success"]:
        if budget is not None:
            # Estimate token metrics if exact headers unavailable
            in_tok = len(str(canonical_messages)) // 4
            out_tok = len(res["text"]) // 4
            budget.record_call(category=category, input_tokens=in_tok, output_tokens=out_tok, is_retry=False, is_fallback=False)
        return res["text"]

    error_type = res.get("error_type", "other")
    p_err = res.get("error", f"Unknown {primary_provider} error")

    # If context too large, try compressed prompt with primary first
    if error_type == "context_too_large":
        compressed_msgs = compress_canonical_messages(canonical_messages, max_msg_chars=2500)
        res_trim = p_prov.generate(
            messages=compressed_msgs,
            model=primary_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        if res_trim["success"]:
            if budget is not None:
                in_tok = len(str(compressed_msgs)) // 4
                out_tok = len(res_trim["text"]) // 4
                budget.record_call(category=category, input_tokens=in_tok, output_tokens=out_tok, is_retry=True, is_fallback=False)
            return res_trim["text"]

    # 2. Check if Fallback Provider is available
    fallback_key = os.getenv("GEMINI_API_KEY") if fallback_provider == "gemini" else os.getenv("GROQ_API_KEY")
    if fallback_key:
        safe_print(f"[LLM Router] {primary_provider.capitalize()} unavailable ({error_type}: {p_err[:80]}...). Failing over to {fallback_provider.capitalize()}.")
        time.sleep(2.0)  # Brief delay ONLY when fallback is activated

        fb_prov = REGISTRY.get(fallback_provider)
        fb_payload = compress_canonical_messages(canonical_messages, max_msg_chars=3500)
        fb_res = fb_prov.generate(
            messages=fb_payload,
            model=fallback_model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if fb_res["success"]:
            if budget is not None:
                in_tok = len(str(fb_payload)) // 4
                out_tok = len(fb_res["text"]) // 4
                budget.record_call(category=category, input_tokens=in_tok, output_tokens=out_tok, is_retry=False, is_fallback=True)
            return fb_res["text"]

        safe_print(f"[LLM Router] {fallback_provider.capitalize()} fallback unavailable. Returning error.")
        return f"[LLM Provider Failure]: Primary ({primary_provider}) error: {p_err} | Fallback ({fallback_provider}) error: {fb_res.get('error')}"

    return p_err


def sanitize_scientific_markdown(text: str) -> str:
    """
    Sanitize LaTeX formulas and Markdown formatting:
    1. Fix invalid non-standard LaTeX macros like \\left\\round ... \\right\\round -> \\text{round}\\left( ... \\right)
    2. Fix double comma artifacts in clamp functions.
    3. Clean up unclosed math block formatting.
    """
    if not text:
        return text

    # Fix invalid \left\round / \right\round macros
    text = re.sub(r'\\left\\round\s*', r'\\text{round}\\left(', text)
    text = re.sub(r'\\right\\round\s*', r'\\right)', text)
    text = re.sub(r'\\round\b', r'\\text{round}', text)

    # Fix double commas in clamped formulas: clamp(..., , 0, , 2^n - 1)
    text = re.sub(r',\s*,', ',', text)

    return text


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

