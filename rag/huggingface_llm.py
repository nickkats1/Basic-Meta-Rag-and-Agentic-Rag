"""Direct HuggingFace LLM adapters -- no LangChain wrapper.

Two adapters, both exposing a tiny ``.invoke(prompt: str) -> str`` interface
that :class:`rag.pipeline.RAGPipeline` knows how to call:

* :class:`HuggingFaceAPILLM` -- calls the HuggingFace serverless Inference API
  via ``huggingface_hub.InferenceClient``. Best when you don't want to host
  the weights yourself.
* :class:`HuggingFaceLocalLLM` -- runs a model locally with
  ``transformers.pipeline``. Best when you have a GPU and want zero network
  hops.

Neither depends on LangChain.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from huggingface_hub import InferenceClient
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)


class HuggingFaceAPILLM:
    """Chat-style LLM backed by the HuggingFace Inference API.

    Implements only the surface area needed by :class:`rag.pipeline.RAGPipeline`:
    a single ``invoke(prompt: str) -> str`` method.
    """

    def __init__(
        self,
        model_name: str,
        api_token: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        timeout: int = 120,
    ) -> None:
        """Build the adapter.

        Args:
            model_name: HuggingFace Hub repo id, e.g. ``"mistralai/Mistral-7B-Instruct-v0.2"``.
            api_token: HF API token. Falls back to ``$HF_TOKEN`` /
                ``$HUGGINGFACEHUB_API_TOKEN`` if omitted.
            temperature: Sampling temperature.
            max_new_tokens: Generation length cap.
            timeout: Request timeout in seconds.

        Raises:
            ValueError: If no token is found.
        """
        token = (
            api_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        )
        if not token:
            raise ValueError(
                "HuggingFaceAPILLM needs an API token (api_token or HF_TOKEN env var)"
            )

        self._model_name = model_name
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._client = InferenceClient(model=model_name, token=token, timeout=timeout)

    @property
    def model_name(self) -> str:
        """Return the configured model id."""
        return self._model_name

    def invoke(self, prompt: str) -> str:
        """Generate a completion for ``prompt`` and return the text.

        Uses the chat-completion endpoint when the model supports it (most
        instruction-tuned models do) and falls back to text-generation
        otherwise.
        """
        try:
            response = self._client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
                max_tokens=self._max_new_tokens,
            )
            return response.choices[0].message.content
        except Exception as exc:  # pragma: no cover - depends on HF backend
            logger.warning(
                "chat_completion failed (%s); falling back to text_generation", exc
            )
            return self._client.text_generation(
                prompt=prompt,
                temperature=self._temperature,
                max_new_tokens=self._max_new_tokens,
            )

    def __call__(self, prompt: str) -> str:
        return self.invoke(prompt)


class HuggingFaceLocalLLM:
    """Local LLM running through ``transformers.pipeline``.

    The pipeline is built lazily on the first call to :meth:`invoke` so that
    importing this module does not eagerly download model weights.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        device: Optional[str] = None,
        torch_dtype: Any = None,
    ) -> None:
        """Build the adapter.

        Args:
            model_name: HuggingFace Hub model id.
            temperature: Sampling temperature. ``0.0`` triggers greedy decoding.
            max_new_tokens: Generation length cap.
            device: ``"cuda"``, ``"cpu"``, or ``None`` for auto-selection.
            torch_dtype: Optional ``torch.dtype`` to load the weights in.
        """
        self._model_name = model_name
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._device = device
        self._torch_dtype = torch_dtype
        self._pipeline = None

    @property
    def model_name(self) -> str:
        """Return the configured model id."""
        return self._model_name

    def _ensure_pipeline(self):
        if self._pipeline is None:
            kwargs: dict = {"task": "text-generation", "model": self._model_name}
            if self._device is not None:
                kwargs["device"] = self._device
            if self._torch_dtype is not None:
                kwargs["torch_dtype"] = self._torch_dtype
            logger.info("Loading local HuggingFace model: %s", self._model_name)
            self._pipeline = hf_pipeline(**kwargs)
        return self._pipeline

    def invoke(self, prompt: str) -> str:
        """Generate a completion locally and return only the new tokens."""
        pipe = self._ensure_pipeline()
        do_sample = self._temperature > 0.0
        outputs = pipe(
            prompt,
            max_new_tokens=self._max_new_tokens,
            do_sample=do_sample,
            temperature=self._temperature if do_sample else None,
            return_full_text=False,
        )
        return outputs[0]["generated_text"]

    def __call__(self, prompt: str) -> str:
        return self.invoke(prompt)


__all__ = ["HuggingFaceAPILLM", "HuggingFaceLocalLLM"]
