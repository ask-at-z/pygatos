"""Utility functions for pygatos."""

import re
from typing import Optional


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.

    Args:
        text: Input text to clean.

    Returns:
        Cleaned text.
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Input text.
        max_length: Maximum length.
        suffix: Suffix to add if truncated.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def parse_bullet_points(text: str) -> list[str]:
    """
    Parse bullet points from LLM response text.

    Args:
        text: Text containing bullet points.

    Returns:
        List of bullet point texts (without the bullet markers).
    """
    lines = text.strip().split('\n')
    bullets = []

    for line in lines:
        line = line.strip()
        # Match various bullet formats: -, *, •, numbered
        match = re.match(r'^(?:[-*•]|\d+[.):])\s*(.+)$', line)
        if match:
            bullets.append(match.group(1).strip())
        elif line and not line.startswith('#'):
            # Include non-empty lines that aren't headers
            bullets.append(line)

    return bullets


def chunk_text(
    text: str,
    chunk_size: int = 250,
    overlap: int = 50,
    tokenizer: Optional[callable] = None,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text to chunk.
        chunk_size: Target chunk size (in words if no tokenizer, else tokens).
        overlap: Overlap between chunks.
        tokenizer: Optional tokenizer function.

    Returns:
        List of text chunks.
    """
    if tokenizer is not None:
        # Token-based chunking
        tokens = tokenizer(text)
        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(chunk_tokens)
            start += chunk_size - overlap

        return chunks

    # Word-based chunking (fallback)
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def deduplicate_texts(texts: list[str]) -> tuple[list[str], dict[str, list[int]]]:
    """
    Remove duplicate texts and track original indices.

    Args:
        texts: List of texts.

    Returns:
        Tuple of (unique_texts, mapping from unique text to original indices).
    """
    unique_texts = []
    text_to_indices = {}

    for i, text in enumerate(texts):
        normalized = clean_text(text.lower())
        if normalized not in text_to_indices:
            text_to_indices[normalized] = []
            unique_texts.append(text)
        text_to_indices[normalized].append(i)

    return unique_texts, text_to_indices


def cosine_similarity(a, b) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity (float between -1 and 1).
    """
    import numpy as np

    a = np.array(a).flatten()
    b = np.array(b).flatten()

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def batch_iterator(items: list, batch_size: int):
    """
    Iterate over items in batches.

    Args:
        items: List of items.
        batch_size: Size of each batch.

    Yields:
        Batches of items.
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Number of seconds.

    Returns:
        Formatted time string (e.g., "2m 30s").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
