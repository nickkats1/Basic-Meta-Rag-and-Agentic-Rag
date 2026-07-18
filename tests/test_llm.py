"""Tests for LLM providers and factory."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.llm import get_llm


class TestGetLLM:
    """Test LLM factory function."""

    def test_get_llm_invalid_provider(self):
        """get_llm raises ValueError for unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_llm(provider="unknown_provider")

    @patch("langchain_openai.ChatOpenAI")
    def test_get_llm_openai(self, mock_chat_openai):
        """get_llm returns OpenAILLM for openai provider."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        llm = get_llm(provider="openai", api_key="test_key")
        assert llm is not None
        assert hasattr(llm, "generate")

    @patch("langchain_groq.ChatGroq")
    def test_get_llm_groq(self, mock_chat_groq):
        """get_llm returns GroqLLM for groq provider."""
        mock_llm = MagicMock()
        mock_chat_groq.return_value = mock_llm

        llm = get_llm(provider="groq", api_key="test_key")
        assert llm is not None
        assert hasattr(llm, "generate")

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_get_llm_google(self, mock_google_llm):
        """get_llm returns GoogleLLM for google provider."""
        mock_llm = MagicMock()
        mock_google_llm.return_value = mock_llm

        llm = get_llm(provider="google", api_key="test_key")
        assert llm is not None
        assert hasattr(llm, "generate")

    @patch("transformers.pipeline")
    def test_get_llm_huggingface_local(self, mock_pipeline):
        """get_llm returns HuggingFaceLocalLLM for huggingface_local provider."""
        mock_pipe = MagicMock()
        mock_pipeline.return_value = mock_pipe

        llm = get_llm(provider="huggingface_local")
        assert llm is not None
        assert hasattr(llm, "generate")

    @patch("langchain_openai.ChatOpenAI")
    def test_get_llm_custom_model(self, mock_chat_openai):
        """get_llm accepts custom model parameter."""
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm

        llm = get_llm(provider="openai", model="gpt-4", api_key="test_key")
        assert llm is not None
