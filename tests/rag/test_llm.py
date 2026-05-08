"""Tests for the :mod:`rag.llm` provider factory and HF adapters."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.huggingface_llm import HuggingFaceAPILLM, HuggingFaceLocalLLM
from rag.llm import LLM, SUPPORTED_PROVIDERS


def test_requires_api_key():
    with pytest.raises(ValueError):
        LLM(api_key="")


def test_openai_provider():
    with patch("rag.llm.ChatOpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        result = LLM(api_key="fake-key").get_llm("openai", "gpt-4", 0.0)

    mock_openai.assert_called_once_with(
        model="gpt-4",
        temperature=0.0,
        openai_api_key="fake-key",
    )
    assert result is mock_openai.return_value


def test_openai_provider_with_max_tokens():
    with patch("rag.llm.ChatOpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        LLM(api_key="fake-key").get_llm("openai", "gpt-4", 0.0, max_tokens=128)

    _, kwargs = mock_openai.call_args
    assert kwargs["max_tokens"] == 128


def test_groq_provider():
    with patch("rag.llm.ChatGroq") as mock_groq:
        mock_groq.return_value = MagicMock()
        result = LLM(api_key="fake-key").get_llm("groq", "llama-3.3-70b-versatile", 0.0)

    mock_groq.assert_called_once_with(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        groq_api_key="fake-key",
    )
    assert result is mock_groq.return_value


def test_google_provider():
    with patch("rag.llm.ChatGoogleGenerativeAI") as mock_google:
        mock_google.return_value = MagicMock()
        result = LLM(api_key="fake-key").get_llm("google", "gemini-2.0-flash", 0.0)

    mock_google.assert_called_once_with(
        model="gemini-2.0-flash",
        temperature=0.0,
        google_api_key="fake-key",
    )
    assert result is mock_google.return_value


def test_google_provider_with_max_tokens():
    with patch("rag.llm.ChatGoogleGenerativeAI") as mock_google:
        mock_google.return_value = MagicMock()
        LLM(api_key="fake-key").get_llm("google", "gemini-2.0-flash", 0.0, max_tokens=64)

    _, kwargs = mock_google.call_args
    assert kwargs["max_output_tokens"] == 64


def test_huggingface_api_provider():
    with patch("rag.huggingface_llm.InferenceClient") as mock_client:
        mock_client.return_value = MagicMock()
        result = LLM(api_key="hf-token").get_llm(
            "huggingface", "mistralai/Mistral-7B-Instruct-v0.2", 0.1, max_tokens=128
        )

    assert isinstance(result, HuggingFaceAPILLM)
    assert result.model_name == "mistralai/Mistral-7B-Instruct-v0.2"
    mock_client.assert_called_once()


def test_huggingface_local_provider():
    result = LLM(api_key="unused").get_llm(
        "huggingface_local", "HuggingFaceTB/SmolLM2-360M-Instruct"
    )
    assert isinstance(result, HuggingFaceLocalLLM)
    assert result.model_name == "HuggingFaceTB/SmolLM2-360M-Instruct"


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        LLM(api_key="fake-key").get_llm("anthropic", "claude-opus-4-7", 0.0)


def test_supported_providers():
    assert set(SUPPORTED_PROVIDERS) == {
        "openai",
        "groq",
        "google",
        "huggingface",
        "huggingface_local",
    }


class TestHuggingFaceAPILLM:
    def test_invoke_uses_chat_completion(self):
        with patch("rag.huggingface_llm.InferenceClient") as mock_client_cls:
            mock_client = MagicMock()
            choice = MagicMock()
            choice.message.content = "the answer"
            mock_client.chat_completion.return_value = MagicMock(choices=[choice])
            mock_client_cls.return_value = mock_client

            llm = HuggingFaceAPILLM(model_name="m", api_token="t")
            assert llm.invoke("hello") == "the answer"
            mock_client.chat_completion.assert_called_once()

    def test_requires_token(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
        with pytest.raises(ValueError):
            HuggingFaceAPILLM(model_name="m")
