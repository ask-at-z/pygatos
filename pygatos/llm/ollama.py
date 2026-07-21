"""Ollama LLM backend using HTTP API."""

import json
import logging
import re
from typing import Optional

import requests

from pygatos.llm.base import BaseLLM
from pygatos.config import LLMConfig

logger = logging.getLogger(__name__)


class OllamaBackend(BaseLLM):
    """
    Ollama LLM backend using direct HTTP requests.

    This backend communicates with a locally running Ollama server
    via its REST API.

    Example:
        >>> llm = OllamaBackend(model="qwen3:30b-a3b-instruct-2507-q4_K_M")
        >>> response = llm.generate("What is the capital of France?")
        >>> print(response)
        The capital of France is Paris.
    """

    def __init__(
        self,
        model: str = "qwen3:30b-a3b-instruct-2507-q4_K_M",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 120,
        debug: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Ollama backend.

        Args:
            model: The Ollama model name to use.
            base_url: Base URL for the Ollama API.
            temperature: Default sampling temperature.
            max_tokens: Default maximum tokens to generate.
            timeout: Request timeout in seconds.
            debug: If True, log all prompts and responses.
            seed: Optional sampling seed forwarded to Ollama as options.seed. Without it, sampling
                is unseeded regardless of any pipeline-level random_seed (which only governs
                numpy/UMAP/clustering, not the LLM).
        """
        self._model = model
        self.base_url = base_url.rstrip("/")
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.timeout = timeout
        self.debug = debug
        self.seed = seed
        # Disable "thinking" for structured-output calls. Thinking models (Qwen3.x, Gemma 4,
        # etc.) otherwise generate long reasoning blocks on EVERY call — 7-16x slower and the
        # source of most JSON parse failures. GATOS never uses the reasoning, so we turn it
        # off at generation time (Ollama ignores `think` for non-thinking models like the
        # qwen3-instruct baseline, so this is safe across the board).
        self.think = False
        # Retries for generate_json: modern "thinking"/verbose models (Qwen3.x, Gemma 4,
        # etc.) intermittently emit unparseable output at temperature>0 (reasoning blocks,
        # prose, malformed JSON). Without retries a single bad sample silently drops a
        # suggested code or a novelty decision, corrupting the codebook. 3 attempts makes
        # the LLM I/O robust across models without changing the GATOS algorithm.
        self.json_max_retries = 3

    def _options(self, temperature: Optional[float], max_tokens: Optional[int]) -> dict:
        """Resolve per-call sampling options.

        NOTE: uses ``is not None`` rather than ``or`` — ``0.0 or 0.7`` evaluates to 0.7 in Python,
        which silently discarded every explicitly-requested temperature of 0.0 (deterministic
        adjudication steps were in fact sampling at the default temperature).
        """
        opts = {
            "temperature": temperature if temperature is not None else self.default_temperature,
            "num_predict": max_tokens if max_tokens is not None else self.default_max_tokens,
        }
        if self.seed is not None:
            opts["seed"] = self.seed
        return opts

    @classmethod
    def from_config(cls, config: LLMConfig) -> "OllamaBackend":
        """Create an OllamaBackend from a configuration object."""
        return cls(
            model=config.model,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            debug=config.debug,
            seed=getattr(config, "seed", None),
        )

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The user prompt to send to the model.
            system: Optional system prompt.
            temperature: Optional sampling temperature (overrides default).
            max_tokens: Optional maximum tokens to generate (overrides default).

        Returns:
            The generated text response.

        Raises:
            requests.RequestException: If the API request fails.
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "think": self.think,
            "options": self._options(temperature, max_tokens),
        }

        if system:
            payload["system"] = system

        if self.debug:
            logger.debug("=" * 60)
            logger.debug("LLM REQUEST")
            logger.debug("=" * 60)
            if system:
                logger.debug(f"SYSTEM:\n{system}")
            logger.debug(f"PROMPT:\n{prompt}")
            logger.debug("-" * 60)

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        result = response.json()
        response_text = result.get("response", "")

        if self.debug:
            logger.debug(f"RESPONSE:\n{response_text}")
            logger.debug("=" * 60)

        return response_text

    def generate_chat(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text using the chat API format.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Optional sampling temperature.
            max_tokens: Optional maximum tokens to generate.

        Returns:
            The generated text response.
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": self._options(temperature, max_tokens),
        }

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        result = response.json()
        return result.get("message", {}).get("content", "")

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        Generate and parse a JSON response.

        The method attempts to extract JSON from the response, handling
        cases where the model wraps JSON in markdown code blocks.

        Args:
            prompt: The user prompt to send to the model.
            system: Optional system prompt.
            temperature: Optional sampling temperature (overrides default).
            max_tokens: Optional maximum tokens to generate (overrides default).

        Returns:
            Parsed JSON response as a dictionary.

        Raises:
            ValueError: If the response cannot be parsed as JSON.
        """
        # Add JSON formatting instruction to system prompt
        json_system = system or ""
        if json_system:
            json_system += "\n\n"
        json_system += "You must respond with valid JSON only. No additional text or explanation."

        # Retry on parse failure. Verbose/thinking models fail intermittently at temp>0;
        # a re-sample (with a firmer instruction after the first failure) usually recovers.
        last_err: Exception | None = None
        sys_prompt = json_system
        for attempt in range(self.json_max_retries):
            response = self.generate(
                prompt=prompt,
                system=sys_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            try:
                return self._parse_json_response(response)
            except ValueError as e:
                last_err = e
                if self.debug:
                    logger.warning(f"generate_json parse failure (attempt {attempt+1}/"
                                   f"{self.json_max_retries}); retrying")
                sys_prompt = (json_system + "\n\nCRITICAL: Output ONLY a single valid JSON "
                              "value. Do NOT include markdown code fences, <think> reasoning, "
                              "or any prose before or after the JSON.")
        raise last_err  # exhausted retries

    def _parse_json_response(self, response: str) -> dict:
        """
        Parse JSON from a response string.

        Handles cases where JSON is wrapped in markdown code blocks.

        Args:
            response: The raw response string.

        Returns:
            Parsed JSON as a dictionary.

        Raises:
            ValueError: If JSON cannot be extracted or parsed.
        """
        # Strip reasoning blocks that "thinking" models (Qwen3.x, etc.) emit before the
        # JSON payload; the braces inside them otherwise defeat the extraction regexes.
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        # Also drop a leading unclosed think block (truncated reasoning) up to </think>.
        response = re.sub(r"^\s*<think>.*?</think>", "", response, flags=re.DOTALL)
        response = response.strip()

        # Try direct parsing first
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        matches = re.findall(json_block_pattern, response)
        if matches:
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue

        # Try to find JSON object or array in the response
        json_patterns = [
            r"(\{[\s\S]*\})",  # JSON object
            r"(\[[\s\S]*\])",  # JSON array
        ]
        for pattern in json_patterns:
            matches = re.findall(pattern, response)
            if matches:
                # Try the longest match first (most likely to be complete)
                for match in sorted(matches, key=len, reverse=True):
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        continue

        raise ValueError(f"Could not parse JSON from response: {response[:500]}...")

    def is_available(self) -> bool:
        """
        Check if the Ollama server is available.

        Returns:
            True if the server is reachable, False otherwise.
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        """
        List available models on the Ollama server.

        Returns:
            List of model names.
        """
        response = requests.get(f"{self.base_url}/api/tags", timeout=10)
        response.raise_for_status()
        result = response.json()
        return [model["name"] for model in result.get("models", [])]
