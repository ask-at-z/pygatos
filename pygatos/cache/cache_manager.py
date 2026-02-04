"""Caching utilities for embeddings and LLM responses."""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Any
import pickle

import numpy as np

from pygatos.config import CacheConfig


class CacheManager:
    """
    Cache manager for storing embeddings and LLM responses.

    Uses a file-based cache with content-addressed storage (hashing keys).
    Embeddings are stored as numpy files, other data as JSON or pickle.

    Example:
        >>> cache = CacheManager(cache_dir=".pygatos_cache")
        >>> cache.set_embedding("text_hash", embeddings)
        >>> embeddings = cache.get_embedding("text_hash")
    """

    def __init__(
        self,
        cache_dir: str = ".pygatos_cache",
        enabled: bool = True,
        cache_embeddings: bool = True,
        cache_llm_responses: bool = True,
    ):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory for cache files.
            enabled: Whether caching is enabled.
            cache_embeddings: Whether to cache embeddings.
            cache_llm_responses: Whether to cache LLM responses.
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.cache_embeddings = cache_embeddings
        self.cache_llm_responses = cache_llm_responses

        # Create subdirectories
        self._embeddings_dir = self.cache_dir / "embeddings"
        self._llm_dir = self.cache_dir / "llm"
        self._data_dir = self.cache_dir / "data"

        if self.enabled:
            self._ensure_dirs()

    @classmethod
    def from_config(cls, config: CacheConfig) -> "CacheManager":
        """Create a CacheManager from a configuration object."""
        return cls(
            cache_dir=config.cache_dir,
            enabled=config.enabled,
            cache_embeddings=config.cache_embeddings,
            cache_llm_responses=config.cache_llm_responses,
        )

    def _ensure_dirs(self) -> None:
        """Create cache directories if they don't exist."""
        self._embeddings_dir.mkdir(parents=True, exist_ok=True)
        self._llm_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _hash_key(self, key: str) -> str:
        """Generate a hash for a cache key."""
        return hashlib.sha256(key.encode()).hexdigest()[:32]

    def _hash_texts(self, texts: list[str], model_name: str = "") -> str:
        """Generate a hash for a list of texts."""
        content = json.dumps(sorted(texts)) + model_name
        return self._hash_key(content)

    # ===== Embedding Cache =====

    def get_embedding(
        self,
        texts: list[str],
        model_name: str = "",
    ) -> Optional[np.ndarray]:
        """
        Get cached embeddings for texts.

        Args:
            texts: List of texts that were embedded.
            model_name: Name of the embedding model.

        Returns:
            Numpy array of embeddings if cached, None otherwise.
        """
        if not self.enabled or not self.cache_embeddings:
            return None

        cache_key = self._hash_texts(texts, model_name)
        cache_path = self._embeddings_dir / f"{cache_key}.npy"

        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception:
                return None

        return None

    def set_embedding(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        model_name: str = "",
    ) -> None:
        """
        Cache embeddings for texts.

        Args:
            texts: List of texts that were embedded.
            embeddings: The embedding array to cache.
            model_name: Name of the embedding model.
        """
        if not self.enabled or not self.cache_embeddings:
            return

        cache_key = self._hash_texts(texts, model_name)
        cache_path = self._embeddings_dir / f"{cache_key}.npy"

        try:
            np.save(cache_path, embeddings)
        except Exception:
            pass  # Silently fail on cache errors

    def get_embedding_single(
        self,
        text: str,
        model_name: str = "",
    ) -> Optional[np.ndarray]:
        """Get cached embedding for a single text."""
        return self.get_embedding([text], model_name)

    def set_embedding_single(
        self,
        text: str,
        embedding: np.ndarray,
        model_name: str = "",
    ) -> None:
        """Cache embedding for a single text."""
        self.set_embedding([text], embedding, model_name)

    # ===== LLM Response Cache =====

    def get_llm_response(
        self,
        prompt: str,
        system: Optional[str] = None,
        model_name: str = "",
    ) -> Optional[str]:
        """
        Get cached LLM response.

        Args:
            prompt: The prompt that was sent.
            system: The system prompt (if any).
            model_name: Name of the LLM model.

        Returns:
            Cached response string if available, None otherwise.
        """
        if not self.enabled or not self.cache_llm_responses:
            return None

        cache_key = self._hash_key(f"{model_name}:{system or ''}:{prompt}")
        cache_path = self._llm_dir / f"{cache_key}.json"

        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    return data.get("response")
            except Exception:
                return None

        return None

    def set_llm_response(
        self,
        prompt: str,
        response: str,
        system: Optional[str] = None,
        model_name: str = "",
    ) -> None:
        """
        Cache an LLM response.

        Args:
            prompt: The prompt that was sent.
            response: The response to cache.
            system: The system prompt (if any).
            model_name: Name of the LLM model.
        """
        if not self.enabled or not self.cache_llm_responses:
            return

        cache_key = self._hash_key(f"{model_name}:{system or ''}:{prompt}")
        cache_path = self._llm_dir / f"{cache_key}.json"

        try:
            with open(cache_path, "w") as f:
                json.dump({
                    "prompt": prompt[:500],  # Truncate for readability
                    "system": system[:500] if system else None,
                    "model": model_name,
                    "response": response,
                }, f)
        except Exception:
            pass  # Silently fail on cache errors

    # ===== Generic Data Cache =====

    def get_data(self, key: str) -> Optional[Any]:
        """
        Get cached data by key.

        Args:
            key: Cache key.

        Returns:
            Cached data if available, None otherwise.
        """
        if not self.enabled:
            return None

        cache_key = self._hash_key(key)
        cache_path = self._data_dir / f"{cache_key}.pkl"

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None

        return None

    def set_data(self, key: str, data: Any) -> None:
        """
        Cache arbitrary data.

        Args:
            key: Cache key.
            data: Data to cache (must be picklable).
        """
        if not self.enabled:
            return

        cache_key = self._hash_key(key)
        cache_path = self._data_dir / f"{cache_key}.pkl"

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception:
            pass  # Silently fail on cache errors

    # ===== Cache Management =====

    def clear(self, cache_type: Optional[str] = None) -> int:
        """
        Clear cached data.

        Args:
            cache_type: Type of cache to clear: 'embeddings', 'llm', 'data', or None for all.

        Returns:
            Number of files deleted.
        """
        deleted = 0

        if cache_type is None or cache_type == "embeddings":
            deleted += self._clear_dir(self._embeddings_dir)

        if cache_type is None or cache_type == "llm":
            deleted += self._clear_dir(self._llm_dir)

        if cache_type is None or cache_type == "data":
            deleted += self._clear_dir(self._data_dir)

        return deleted

    def _clear_dir(self, dir_path: Path) -> int:
        """Clear all files in a directory."""
        deleted = 0
        if dir_path.exists():
            for file in dir_path.iterdir():
                if file.is_file():
                    try:
                        file.unlink()
                        deleted += 1
                    except Exception:
                        pass
        return deleted

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        def dir_stats(dir_path: Path) -> dict:
            if not dir_path.exists():
                return {"files": 0, "size_mb": 0.0}

            files = list(dir_path.iterdir())
            total_size = sum(f.stat().st_size for f in files if f.is_file())

            return {
                "files": len(files),
                "size_mb": round(total_size / (1024 * 1024), 2),
            }

        return {
            "enabled": self.enabled,
            "cache_dir": str(self.cache_dir),
            "embeddings": dir_stats(self._embeddings_dir),
            "llm": dir_stats(self._llm_dir),
            "data": dir_stats(self._data_dir),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"CacheManager(enabled={self.enabled}, "
            f"embeddings={stats['embeddings']['files']} files, "
            f"llm={stats['llm']['files']} files)"
        )
