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
        """
        self._model = model
        self.base_url = base_url.rstrip("/")
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.timeout = timeout
        self.debug = debug

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
            "options": {
                "temperature": temperature or self.default_temperature,
                "num_predict": max_tokens or self.default_max_tokens,
            },
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
            "options": {
                "temperature": temperature or self.default_temperature,
                "num_predict": max_tokens or self.default_max_tokens,
            },
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

        response = self.generate(
            prompt=prompt,
            system=json_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return self._parse_json_response(response)

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
