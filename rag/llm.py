"""Multi-provider chat-model factory.

Cloud providers (OpenAI, Groq, Google) are served through their LangChain
wrappers; HuggingFace is served directly via :mod:`rag.huggingface_llm`. The
:class:`rag.pipeline.RAGPipeline` accepts either flavor.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional  # noqa: F401

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from rag.huggingface_llm import HuggingFaceAPILLM, HuggingFaceLocalLLM


Builder = Callable[[str, str, float, Optional[int]], Any]


def _maybe_token_kwarg(name: str, value: Optional[int]) -> Dict[str, int]:
    """Return ``{name: value}`` only if ``value`` is set, else ``{}``.

    Some LangChain provider wrappers reject ``max_tokens=None`` outright (the
    Google one in particular), so we only forward the kwarg when the user
    actually requested a cap.
    """
    return {name: value} if value is not None else {}


def _build_openai(api_key: str, model: str, temperature: float, max_tokens: Optional[int]) -> Any:
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
        **_maybe_token_kwarg("max_tokens", max_tokens),
    )


def _build_groq(api_key: str, model: str, temperature: float, max_tokens: Optional[int]) -> Any:
    return ChatGroq(
        model=model,
        temperature=temperature,
        groq_api_key=api_key,
        **_maybe_token_kwarg("max_tokens", max_tokens),
    )


def _build_google(api_key: str, model: str, temperature: float, max_tokens: Optional[int]) -> Any:
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
        **_maybe_token_kwarg("max_output_tokens", max_tokens),
    )


def _build_huggingface(api_key: str, model: str, temperature: float, max_tokens: Optional[int]) -> Any:
    return HuggingFaceAPILLM(
        model_name=model,
        api_token=api_key,
        temperature=temperature,
        max_new_tokens=max_tokens or 512,
    )


def _build_huggingface_local(api_key: str, model: str, temperature: float, max_tokens: Optional[int]) -> Any:
    del api_key  # local models don't need a token
    return HuggingFaceLocalLLM(
        model_name=model,
        temperature=temperature,
        max_new_tokens=max_tokens or 512,
    )


_BUILDERS: Dict[str, Builder] = {
    "openai": _build_openai,
    "groq": _build_groq,
    "google": _build_google,
    "huggingface": _build_huggingface,
    "huggingface_local": _build_huggingface_local,
}


SUPPORTED_PROVIDERS = tuple(_BUILDERS)


class LLM:
    """Factory for provider-specific chat models.

    Example:
        >>> llm = LLM(api_key=os.environ["GROQ_API_KEY"]).get_llm(
        ...     provider="groq",
        ...     model_name="llama-3.3-70b-versatile",
        ... )
    """

    def __init__(self, api_key: str) -> None:
        """Store the API key and load any ``.env`` file in the cwd.

        Args:
            api_key: API key for the chosen provider. Pass an empty-string
                sentinel like ``"unused"`` for ``huggingface_local`` if you
                don't have a token.

        Raises:
            ValueError: If ``api_key`` is falsy.
        """
        if not api_key:
            raise ValueError("Must provide an API key")
        self.api_key = api_key
        load_dotenv()

    def get_llm(
        self,
        provider: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """Build a chat model for the given provider.

        Args:
            provider: One of ``SUPPORTED_PROVIDERS`` (case-insensitive).
            model_name: Provider-specific model id.
            temperature: Sampling temperature.
            max_tokens: Optional output-token cap.

        Returns:
            A chat model instance compatible with :class:`rag.pipeline.RAGPipeline`.

        Raises:
            ValueError: If ``provider`` is not in :data:`SUPPORTED_PROVIDERS`.
        """
        builder = _BUILDERS.get(provider.lower())
        if builder is None:
            raise ValueError(
                f"Unsupported provider: {provider!r}. "
                f"Supported providers: {SUPPORTED_PROVIDERS}"
            )
        return builder(self.api_key, model_name, temperature, max_tokens)
