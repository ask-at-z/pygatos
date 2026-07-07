"""OpenRouter LLM backend (OpenAI-compatible chat completions API)."""

import json
import logging
import os
import re
import time
from typing import Optional

from pygatos.llm.base import BaseLLM

logger = logging.getLogger(__name__)

_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_KEY_URL = "https://openrouter.ai/api/v1/key"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Default provider routing per model: pin to a specific provider/quantization for
# reproducible precision instead of OpenRouter's dynamic routing.
PROVIDER_PREFS = {
    "openai/gpt-oss-120b": {
        "order": ["DeepInfra"],
        "allow_fallbacks": False,
        "quantizations": ["bf16"],
    },
}


class OpenRouterBackend(BaseLLM):
    """
    OpenRouter backend for cloud inference over the OpenAI-compatible API.

    Mirrors the OllamaBackend LLM-IO hardening: reasoning is suppressed for
    structured-output calls (reasoning models like gpt-oss otherwise burn tokens
    on chain-of-thought that GATOS never uses), generate_json retries on parse
    failures, and <think> blocks are stripped before JSON extraction.

    Example:
        >>> llm = OpenRouterBackend(model="openai/gpt-oss-120b")
        >>> response = llm.generate("What is the capital of France?")

    Note:
        Set OPENROUTER_API_KEY environment variable or pass api_key directly.
        Provider pinning defaults to PROVIDER_PREFS (gpt-oss-120b -> DeepInfra bf16).
    """

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 240,
        debug: bool = False,
        rate_limit_delay: float = 0.0,
        provider: Optional[dict] = None,
        reasoning_effort: Optional[str] = "low",
        budget_usd: Optional[float] = None,
        max_retries: int = 4,
        retry_delay: float = 3.0,
    ):
        """
        Initialize the OpenRouter backend.

        Args:
            model: OpenRouter model id (e.g. "openai/gpt-oss-120b").
            api_key: OpenRouter API key. If None, reads OPENROUTER_API_KEY env var.
            temperature: Default sampling temperature.
            max_tokens: Default maximum completion tokens.
            timeout: Request timeout in seconds.
            debug: If True, log all prompts and responses.
            rate_limit_delay: Seconds to wait between API calls.
            provider: OpenRouter provider-routing dict; defaults to PROVIDER_PREFS[model].
            reasoning_effort: Reasoning effort for reasoning models ("low"/"medium"/"high"),
                or None to omit. "low" mirrors the Ollama think:false hardening — GATOS
                never consumes reasoning, so minimize its cost.
            budget_usd: Optional hard spend cap for THIS backend instance (a per-run
                runaway fuse). Refuses further calls once cumulative usage.cost exceeds it.
            max_retries: Retries for transient HTTP errors (429/5xx) and JSON parse failures.
            retry_delay: Base backoff delay between retries (seconds, linear escalation).
        """
        self._model = model
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.timeout = timeout
        self.debug = debug
        self.rate_limit_delay = rate_limit_delay
        self.provider = provider if provider is not None else PROVIDER_PREFS.get(model)
        self.reasoning_effort = reasoning_effort
        self.budget_usd = budget_usd
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.json_max_retries = 3
        # Usage accounting (recorded into run manifests by callers).
        self.total_cost_usd = 0.0
        self.n_calls = 0
        self.last_provider: Optional[str] = None

    @classmethod
    def from_config(cls, config) -> "OpenRouterBackend":
        """Create an OpenRouterBackend from an LLMConfig."""
        return cls(
            model=config.model,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            debug=config.debug,
            rate_limit_delay=config.rate_limit_delay,
        )

    @property
    def model_name(self) -> str:
        return self._model

    def _post(self, payload: dict) -> str:
        """POST a chat-completions payload with transient-error retries; returns content."""
        import requests

        if self.budget_usd is not None and self.total_cost_usd >= self.budget_usd:
            raise RuntimeError(
                f"OpenRouter budget ${self.budget_usd} reached for this backend instance "
                f"(spent ${self.total_cost_usd:.4f}); refusing further calls"
            )
        if self.rate_limit_delay:
            time.sleep(self.rate_limit_delay)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ask-at-z/pygatos",
            "X-Title": "pygatos",
        }
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(_API_URL, json=payload, headers=headers, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 520, 524):
                    last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))
                        continue
                    raise last_err
                r.raise_for_status()
                data = r.json()
                if "error" in data and not data.get("choices"):
                    raise RuntimeError(f"OpenRouter error: {data['error']}")
                usage = data.get("usage") or {}
                self.total_cost_usd += float(usage.get("cost") or 0.0)
                self.n_calls += 1
                self.last_provider = data.get("provider")
                return data["choices"][0]["message"].get("content", "") or ""
            except Exception as e:  # noqa: BLE001 - retry transient network errors
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                raise
        raise last_err  # pragma: no cover

    def _build_payload(
        self,
        prompt: str,
        system: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        json_mode: bool,
    ) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens or self.default_max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.provider:
            payload["provider"] = self.provider
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        return payload

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload = self._build_payload(prompt, system, temperature, max_tokens, json_mode=False)
        if self.debug:
            logger.info(f"OpenRouter prompt:\n{prompt}")
        response = _THINK_RE.sub("", self._post(payload)).strip()
        if self.debug:
            logger.info(f"OpenRouter response:\n{response}")
        return response

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        json_system = system or ""
        if json_system:
            json_system += "\n\n"
        json_system += "You must respond with valid JSON only. No additional text or explanation."

        # Retry on parse failure with a firmer instruction (mirrors OllamaBackend hardening).
        last_err: Optional[Exception] = None
        sys_prompt = json_system
        for attempt in range(self.json_max_retries):
            payload = self._build_payload(prompt, sys_prompt, temperature, max_tokens, json_mode=True)
            response = self._post(payload)
            try:
                return self._parse_json_response(response)
            except ValueError as e:
                last_err = e
                if self.debug:
                    logger.warning(
                        f"generate_json parse failure (attempt {attempt + 1}/"
                        f"{self.json_max_retries}); retrying"
                    )
                sys_prompt = (
                    json_system
                    + "\n\nCRITICAL: Output ONLY a single valid JSON value. Do NOT include "
                    "markdown code fences, <think> reasoning, or any prose before or after "
                    "the JSON."
                )
        raise last_err  # exhausted retries

    def _parse_json_response(self, response: str) -> dict:
        """Extract a JSON object from a model response (strips reasoning/fences)."""
        response = _THINK_RE.sub("", response).strip()
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        for pattern in (r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```", r"(\{[\s\S]*\})"):
            m = re.search(pattern, response)
            if m:
                try:
                    return json.loads(m.group(1))
                except (json.JSONDecodeError, IndexError):
                    continue
        raise ValueError(f"Could not parse JSON from response: {response[:200]}")

    def is_available(self) -> bool:
        """Check the API key is present and valid (cheap auth-check endpoint)."""
        if not self._api_key:
            return False
        try:
            import requests

            r = requests.get(
                _KEY_URL, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=10
            )
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def get_stats(self) -> dict:
        """Usage stats for provenance manifests."""
        return {
            "n_calls": self.n_calls,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "last_provider": self.last_provider,
            "provider_prefs": self.provider,
            "reasoning_effort": self.reasoning_effort,
        }
