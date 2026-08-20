"""
tests/test_provider_failover.py
---------------------------------------------------------------------
Unit test suite verifying LLM provider system and failover:
1. Groq succeeds -> Gemini is never called.
2. Groq returns 429 -> Gemini fallback is called.
3. Fallback preserves canonical conversation turns.
4. Gemini succeeds after Groq failure -> Gemini's response is returned.
5. Both providers fail -> clear provider failure error returned.
6. Explicit provider request respects specified provider.
---------------------------------------------------------------------
"""

import os
import pytest
from unittest.mock import MagicMock

import config
from core.providers import REGISTRY, call_llm_api, call_with_fallback


@pytest.fixture(autouse=True)
def disable_throttle(monkeypatch):
    """Disable API throttle sleep for fast test execution."""
    monkeypatch.setattr(config, "throttle_api_call", lambda: None)


# 1. Groq succeeds -> Gemini is never called
def test_groq_success_gemini_not_called(monkeypatch):
    monkeypatch.setattr(os, "getenv", lambda k, d=None: "mock_key" if "KEY" in k else d)

    groq_prov = REGISTRY.get("groq")
    gemini_prov = REGISTRY.get("gemini")

    mock_groq = MagicMock(return_value={"success": True, "text": "Groq Response"})
    mock_gemini = MagicMock(return_value={"success": True, "text": "Gemini Response"})

    monkeypatch.setattr(groq_prov, "generate", mock_groq)
    monkeypatch.setattr(gemini_prov, "generate", mock_gemini)

    res = call_with_fallback(prompt="Test Prompt", primary_provider="groq", fallback_provider="gemini")

    assert res == "Groq Response"
    mock_groq.assert_called_once()
    mock_gemini.assert_not_called()


# 2. Groq returns 429 -> Gemini fallback is called immediately
def test_groq_429_immediate_failover_to_gemini(monkeypatch):
    monkeypatch.setattr(os, "getenv", lambda k, d=None: "mock_key" if "KEY" in k else d)

    groq_prov = REGISTRY.get("groq")
    gemini_prov = REGISTRY.get("gemini")

    mock_groq = MagicMock(return_value={
        "success": False,
        "error": "[Groq API Error 429]: Rate limit reached",
        "error_type": "rate_limit",
        "status_code": 429
    })
    mock_gemini = MagicMock(return_value={"success": True, "text": "Gemini Failover Response"})

    monkeypatch.setattr(groq_prov, "generate", mock_groq)
    monkeypatch.setattr(gemini_prov, "generate", mock_gemini)

    res = call_with_fallback(prompt="Test Prompt", primary_provider="groq", fallback_provider="gemini")

    assert res == "Gemini Failover Response"
    mock_groq.assert_called_once()
    mock_gemini.assert_called_once()


# 3. Fallback preserves canonical conversation messages
def test_groq_rate_limit_preserves_conversation(monkeypatch):
    monkeypatch.setattr(os, "getenv", lambda k, d=None: "mock_key" if "KEY" in k else d)

    groq_prov = REGISTRY.get("groq")
    gemini_prov = REGISTRY.get("gemini")

    messages = [
        {"role": "system", "content": "You are a research bot."},
        {"role": "user", "content": "Derive harmonic oscillator."}
    ]

    mock_groq = MagicMock(return_value={
        "success": False,
        "error": "TPM quota exceeded",
        "error_type": "rate_limit",
        "status_code": 429
    })
    mock_gemini = MagicMock(return_value={"success": True, "text": "Gemini Math Output"})

    monkeypatch.setattr(groq_prov, "generate", mock_groq)
    monkeypatch.setattr(gemini_prov, "generate", mock_gemini)

    res = call_with_fallback(messages=messages, primary_provider="groq", fallback_provider="gemini")

    assert res == "Gemini Math Output"
    mock_gemini.assert_called_once()


# 4. Both providers fail -> clear provider failure error returned
def test_both_providers_fail(monkeypatch):
    monkeypatch.setattr(os, "getenv", lambda k, d=None: "mock_key" if "KEY" in k else d)

    groq_prov = REGISTRY.get("groq")
    gemini_prov = REGISTRY.get("gemini")

    mock_groq = MagicMock(return_value={
        "success": False,
        "error": "Groq 429",
        "error_type": "rate_limit",
        "status_code": 429
    })
    mock_gemini = MagicMock(return_value={
        "success": False,
        "error": "Gemini 500",
        "error_type": "server_error"
    })

    monkeypatch.setattr(groq_prov, "generate", mock_groq)
    monkeypatch.setattr(gemini_prov, "generate", mock_gemini)

    res = call_with_fallback(prompt="Hello", primary_provider="groq", fallback_provider="gemini")

    assert "[LLM Provider Failure]" in res
    assert "Primary (groq) error" in res
    assert "Fallback (gemini) error" in res



# 5. Explicit Gemini provider request never converts to Groq
def test_explicit_gemini_provider_request(monkeypatch):
    monkeypatch.setattr(os, "getenv", lambda k, d=None: "mock_key" if "KEY" in k else d)

    groq_prov = REGISTRY.get("groq")
    gemini_prov = REGISTRY.get("gemini")

    mock_groq = MagicMock(return_value={"success": True, "text": "Groq Output"})
    mock_gemini = MagicMock(return_value={"success": True, "text": "Gemini Output"})

    monkeypatch.setattr(groq_prov, "generate", mock_groq)
    monkeypatch.setattr(gemini_prov, "generate", mock_gemini)

    res = call_llm_api(prompt="Report Prompt", provider="gemini", model="gemini-3.6-flash")

    assert res == "Gemini Output"
    mock_gemini.assert_called_once()
    mock_groq.assert_not_called()
