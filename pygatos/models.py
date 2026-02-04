"""Core data models for pygatos.

This module contains dataclasses used throughout the pygatos library
for representing extracted information, results, and other structured data.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class InformationPoint:
    """
    An extracted information point with full lineage tracking.

    Represents an atomic piece of information extracted from text,
    along with metadata about where it came from. This enables
    full auditability - any code applied to this point can be
    traced back to the original source text and specific chunk.

    Attributes:
        text: The extracted atomic idea/information point.
        source_text: Original full text before any chunking.
        chunk_index: Which chunk this was extracted from (0-indexed).
                    None if the source text was short enough to not require chunking.
        chunk_text: The actual chunk text this point was extracted from.
                   None if no chunking was performed.

    Example:
        >>> # For a short text (no chunking)
        >>> point = InformationPoint(
        ...     text="Respondent supports candidate X",
        ...     source_text="I really like candidate X...",
        ...     chunk_index=None,
        ...     chunk_text=None,
        ... )

        >>> # For a long text (with chunking)
        >>> point = InformationPoint(
        ...     text="Respondent is concerned about inflation",
        ...     source_text="<full 500-word response>",
        ...     chunk_index=2,
        ...     chunk_text="...worried about rising prices...",
        ... )
    """

    text: str
    source_text: str
    chunk_index: Optional[int] = None
    chunk_text: Optional[str] = None

    def __str__(self) -> str:
        """Return the information point text."""
        return self.text

    def __repr__(self) -> str:
        """Return a detailed representation."""
        chunk_info = f", chunk={self.chunk_index}" if self.chunk_index is not None else ""
        return f"InformationPoint(text={self.text[:50]!r}...{chunk_info})"

    @property
    def was_chunked(self) -> bool:
        """Return True if this point came from a chunked text."""
        return self.chunk_index is not None

    @property
    def context_text(self) -> str:
        """
        Return the most relevant context text for this point.

        Returns the chunk text if available, otherwise the full source text.
        Useful when you need the immediate context without the full document.
        """
        return self.chunk_text if self.chunk_text else self.source_text
