"""Abstract base class for LLM backends."""

from abc import ABC, abstractmethod
from typing import Optional, Any


class BaseLLM(ABC):
    """
    Abstract base class for LLM backends.

    All LLM implementations should inherit from this class and implement
    the required methods.
    """

    @abstractmethod
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
        """
        pass

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        Generate and parse a JSON response.

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
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
