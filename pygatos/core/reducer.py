"""Dimensionality reduction for clustering embeddings."""

import logging
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA

from pygatos.config import DimensionalityReductionConfig

logger = logging.getLogger(__name__)


class DimensionalityReducer:
    """
    Reduces high-dimensional embeddings for better clustering.

    Uses a two-stage approach:
    1. PCA: Reduce to components explaining target variance (e.g., 90%)
    2. UMAP: Further reduce to low-dimensional space for clustering

    This approach balances:
    - Noise reduction (PCA removes low-variance dimensions)
    - Non-linear structure preservation (UMAP captures manifold structure)
    - Computational efficiency (PCA speeds up UMAP)

    Example:
        >>> reducer = DimensionalityReducer(pca_variance=0.9, umap_n_components=5)
        >>> reduced = reducer.fit_transform(embeddings)
        >>> print(f"Reduced from {embeddings.shape[1]} to {reduced.shape[1]} dims")
    """

    def __init__(
        self,
        pca_variance: float = 0.9,
        umap_n_components: int = 5,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        umap_metric: str = "cosine",
        random_state: Optional[int] = 42,
    ):
        """
        Initialize the dimensionality reducer.

        Args:
            pca_variance: Variance to retain in PCA (0.0-1.0).
            umap_n_components: Number of dimensions for UMAP output.
            umap_n_neighbors: Number of neighbors for UMAP (larger = more global structure).
            umap_min_dist: Minimum distance for UMAP (smaller = tighter clusters).
            umap_metric: Distance metric for UMAP.
            random_state: Random seed for reproducibility.
        """
        self.pca_variance = pca_variance
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_metric = umap_metric
        self.random_state = random_state

        self._pca = None
        self._umap = None
        self._is_fitted = False

    @classmethod
    def from_config(cls, config: DimensionalityReductionConfig, random_state: Optional[int] = 42) -> "DimensionalityReducer":
        """Create a DimensionalityReducer from configuration.

        Args:
            config: Dimensionality reduction configuration.
            random_state: Optional random seed.

        Returns:
            Configured DimensionalityReducer instance.
        """
        return cls(
            pca_variance=config.pca_variance,
            umap_n_components=config.umap_n_components,
            umap_n_neighbors=config.umap_n_neighbors,
            umap_min_dist=config.umap_min_dist,
            umap_metric=config.umap_metric,
            random_state=random_state,
        )

    def fit(self, embeddings: np.ndarray) -> "DimensionalityReducer":
        """
        Fit the reducer on embeddings.

        Args:
            embeddings: Array of shape (n_samples, n_features).

        Returns:
            self for chaining.
        """
        n_samples, n_features = embeddings.shape

        logger.info(f"Fitting reducer on {n_samples} samples with {n_features} features")

        # Stage 1: PCA
        self._fit_pca(embeddings)
        pca_embeddings = self._pca.transform(embeddings)

        logger.info(f"PCA reduced to {pca_embeddings.shape[1]} components")

        # Stage 2: UMAP (if needed and enough samples)
        if self._should_apply_umap(n_samples, pca_embeddings.shape[1]):
            self._fit_umap(pca_embeddings)
            logger.info(f"UMAP will reduce to {self.umap_n_components} dimensions")
        else:
            self._umap = None
            logger.info("Skipping UMAP (not enough samples or already low-dimensional)")

        self._is_fitted = True

        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using fitted reducer.

        Args:
            embeddings: Array of shape (n_samples, n_features).

        Returns:
            Reduced embeddings array.
        """
        if not self._is_fitted:
            raise ValueError("Reducer must be fitted before transform. Call fit() first.")

        # Apply PCA
        reduced = self._pca.transform(embeddings)

        # Apply UMAP if fitted
        if self._umap is not None:
            reduced = self._umap.transform(reduced)

        return reduced

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Args:
            embeddings: Array of shape (n_samples, n_features).

        Returns:
            Reduced embeddings array.
        """
        n_samples, n_features = embeddings.shape

        logger.info(f"Fitting and transforming {n_samples} samples with {n_features} features")

        # Stage 1: PCA
        self._fit_pca(embeddings)
        pca_embeddings = self._pca.transform(embeddings)

        logger.info(f"PCA reduced to {pca_embeddings.shape[1]} components "
                    f"(retained {self._pca.explained_variance_ratio_.sum():.1%} variance)")

        # Stage 2: UMAP (if needed and enough samples)
        if self._should_apply_umap(n_samples, pca_embeddings.shape[1]):
            self._fit_umap(pca_embeddings)
            reduced = self._umap.transform(pca_embeddings)
            logger.info(f"UMAP reduced to {reduced.shape[1]} dimensions")
        else:
            reduced = pca_embeddings
            logger.info("Skipped UMAP (not enough samples or already low-dimensional)")

        self._is_fitted = True

        return reduced

    def _fit_pca(self, embeddings: np.ndarray) -> None:
        """Fit PCA to retain target variance."""
        # Start with enough components to potentially explain target variance
        n_components = min(embeddings.shape[0] - 1, embeddings.shape[1])

        self._pca = PCA(
            n_components=n_components,
            random_state=self.random_state,
        )
        self._pca.fit(embeddings)

        # Find how many components needed for target variance
        cumulative_variance = np.cumsum(self._pca.explained_variance_ratio_)
        n_components_needed = np.searchsorted(cumulative_variance, self.pca_variance) + 1

        # Refit with exact number of components
        self._pca = PCA(
            n_components=min(n_components_needed, n_components),
            random_state=self.random_state,
        )
        self._pca.fit(embeddings)

    def _should_apply_umap(self, n_samples: int, n_pca_components: int) -> bool:
        """Determine if UMAP should be applied."""
        # Need enough samples for UMAP
        min_samples_for_umap = max(self.umap_n_neighbors + 1, 10)

        if n_samples < min_samples_for_umap:
            return False

        # No need for UMAP if PCA already reduced enough
        if n_pca_components <= self.umap_n_components:
            return False

        return True

    def _fit_umap(self, pca_embeddings: np.ndarray) -> None:
        """Fit UMAP on PCA-reduced embeddings."""
        try:
            import umap

            # Adjust n_neighbors if we have few samples
            n_neighbors = min(self.umap_n_neighbors, pca_embeddings.shape[0] - 1)

            self._umap = umap.UMAP(
                n_components=self.umap_n_components,
                n_neighbors=n_neighbors,
                min_dist=self.umap_min_dist,
                metric=self.umap_metric,
                random_state=self.random_state,
            )
            self._umap.fit(pca_embeddings)

        except ImportError:
            logger.warning("UMAP not installed. Install with: pip install umap-learn")
            self._umap = None

    @property
    def pca_components(self) -> Optional[int]:
        """Number of PCA components (after fitting)."""
        if self._pca is None:
            return None
        return self._pca.n_components_

    @property
    def explained_variance(self) -> Optional[float]:
        """Total variance explained by PCA (after fitting)."""
        if self._pca is None:
            return None
        return self._pca.explained_variance_ratio_.sum()

    @property
    def output_dimensions(self) -> Optional[int]:
        """Final output dimensions (after fitting)."""
        if not self._is_fitted:
            return None

        if self._umap is not None:
            return self.umap_n_components

        return self.pca_components

    def get_pca_only(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply only PCA transformation (skip UMAP).

        Useful when you want the higher-dimensional PCA representation
        for visualization or other purposes.

        Args:
            embeddings: Array of shape (n_samples, n_features).

        Returns:
            PCA-reduced embeddings.
        """
        if self._pca is None:
            raise ValueError("Reducer must be fitted first. Call fit() first.")

        return self._pca.transform(embeddings)

    def __repr__(self) -> str:
        if self._is_fitted:
            return (
                f"DimensionalityReducer(pca={self.pca_components}d, "
                f"output={self.output_dimensions}d, "
                f"variance={self.explained_variance:.1%})"
            )
        return (
            f"DimensionalityReducer(pca_variance={self.pca_variance}, "
            f"umap_n_components={self.umap_n_components})"
        )
