"""Text embedding using sentence-transformers."""

from typing import Union, Optional
import numpy as np
from tqdm import tqdm

from pygatos.config import EmbeddingConfig


class Embedder:
    """
    Text embedder using sentence-transformers.

    Generates dense vector representations of text that can be used
    for semantic similarity comparisons and clustering.

    Example:
        >>> embedder = Embedder(model_name="Qwen/Qwen3-Embedding-0.6B")
        >>> embeddings = embedder.embed(["Hello world", "Hi there"])
        >>> print(embeddings.shape)
        (2, 1024)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "auto",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        """
        Initialize the embedder.

        Args:
            model_name: HuggingFace model name for embeddings.
            device: Device to use: 'auto', 'cpu', 'cuda', 'mps'.
            batch_size: Batch size for embedding.
            normalize: Whether to L2-normalize embeddings.
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.normalize = normalize

        # Lazy load the model
        self._model = None

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> "Embedder":
        """Create an Embedder from a configuration object."""
        return cls(
            model_name=config.model_name,
            device=config.device,
            batch_size=config.batch_size,
            normalize=config.normalize,
        )

    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' device to the best available device."""
        if device != "auto":
            return device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    @property
    def model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension of the model."""
        return self.model.get_sentence_embedding_dimension()

    def embed(
        self,
        texts: Union[str, list[str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: A single text or list of texts to embed.
            show_progress: Whether to show a progress bar.

        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim).
            If a single text is provided, returns shape (1, embedding_dim).
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        # Empty input check
        if len(texts) == 0:
            return np.array([]).reshape(0, self.embedding_dimension)

        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        return embeddings

    def embed_batch(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed texts in batches with progress tracking.

        This is an alias for embed() with show_progress=True by default,
        provided for API clarity when processing large datasets.

        Args:
            texts: List of texts to embed.
            batch_size: Optional batch size override.
            show_progress: Whether to show a progress bar.

        Returns:
            Numpy array of embeddings with shape (n_texts, embedding_dim).
        """
        if batch_size is not None:
            original_batch_size = self.batch_size
            self.batch_size = batch_size
            try:
                return self.embed(texts, show_progress=show_progress)
            finally:
                self.batch_size = original_batch_size
        else:
            return self.embed(texts, show_progress=show_progress)

    def similarity(
        self,
        texts1: Union[str, list[str]],
        texts2: Union[str, list[str]],
    ) -> np.ndarray:
        """
        Compute cosine similarity between two sets of texts.

        Args:
            texts1: First text or list of texts.
            texts2: Second text or list of texts.

        Returns:
            Similarity matrix of shape (len(texts1), len(texts2)).
        """
        embeddings1 = self.embed(texts1)
        embeddings2 = self.embed(texts2)

        # Ensure 2D arrays
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)

        # Compute cosine similarity
        # If embeddings are normalized, this is just dot product
        if self.normalize:
            return np.dot(embeddings1, embeddings2.T)
        else:
            # Normalize on the fly
            norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
            norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
            return np.dot(embeddings1 / norm1, (embeddings2 / norm2).T)

    def find_most_similar(
        self,
        query: Union[str, np.ndarray],
        candidates: Union[list[str], np.ndarray],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Find the most similar candidates to a query.

        Args:
            query: Query text or pre-computed embedding.
            candidates: List of candidate texts or pre-computed embeddings.
            top_k: Number of top matches to return.

        Returns:
            List of (index, similarity) tuples, sorted by similarity descending.
        """
        # Get query embedding
        if isinstance(query, str):
            query_emb = self.embed(query)
        else:
            query_emb = query

        # Get candidate embeddings
        if isinstance(candidates, list) and len(candidates) > 0 and isinstance(candidates[0], str):
            candidate_embs = self.embed(candidates)
        else:
            candidate_embs = candidates

        # Ensure 2D
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        # Compute similarities
        if self.normalize:
            similarities = np.dot(query_emb, candidate_embs.T).flatten()
        else:
            norm_q = np.linalg.norm(query_emb, axis=1, keepdims=True)
            norm_c = np.linalg.norm(candidate_embs, axis=1, keepdims=True)
            similarities = np.dot(query_emb / norm_q, (candidate_embs / norm_c).T).flatten()

        # Get top-k indices
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def __repr__(self) -> str:
        return f"Embedder(model_name='{self.model_name}', device='{self.device}')"
