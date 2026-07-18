"""LLM adapters for different providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class LLM(ABC):
    """Base LLM interface."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""


class OpenAILLM(LLM):
    """OpenAI Chat Completions wrapper."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr

        self.llm = ChatOpenAI(
            model=model,
            api_key=SecretStr(api_key) if api_key else None,
            temperature=temperature,
        )

    def generate(self, prompt: str) -> str:
        return str(self.llm.invoke(prompt).content)


class GroqLLM(LLM):
    """Groq API wrapper."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        from langchain_groq import ChatGroq
        from pydantic import SecretStr

        self.llm = ChatGroq(
            model=model,
            api_key=SecretStr(api_key) if api_key else None,
            temperature=temperature,
        )

    def generate(self, prompt: str) -> str:
        return str(self.llm.invoke(prompt).content)


class GoogleLLM(LLM):
    """Google Gemini wrapper."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.llm = ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
        )

    def generate(self, prompt: str) -> str:
        return str(self.llm.invoke(prompt).content)


class HuggingFaceLocalLLM(LLM):
    """Local HuggingFace model using transformers."""

    def __init__(
        self,
        model: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        device: str = "cpu",
    ):
        from transformers import pipeline

        self.pipeline = pipeline("text-generation", model=model, device=device)
        self.model = model

    def generate(self, prompt: str) -> str:
        output = self.pipeline(
            prompt,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
        )
        return output[0]["generated_text"][len(prompt) :].strip()


def get_llm(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> LLM:
    """Factory to get LLM by provider."""
    if provider == "openai":
        model = model or "gpt-4o-mini"
        return OpenAILLM(model=model, api_key=api_key, temperature=temperature)
    elif provider == "groq":
        model = model or "llama-3.3-70b-versatile"
        return GroqLLM(model=model, api_key=api_key, temperature=temperature)
    elif provider == "google":
        model = model or "gemini-2.0-flash"
        return GoogleLLM(model=model, api_key=api_key, temperature=temperature)
    elif provider == "huggingface_local":
        model = model or "HuggingFaceTB/SmolLM2-360M-Instruct"
        return HuggingFaceLocalLLM(model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")
