"""Clustering algorithms for grouping similar texts."""

from typing import Optional, Literal
import numpy as np

from pygatos.config import ClusteringConfig


class Clusterer:
    """
    Clustering implementation for grouping similar embeddings.

    Supports multiple clustering algorithms including agglomerative clustering,
    HDBSCAN, and K-means.

    Example:
        >>> clusterer = Clusterer(method="agglomerative", n_clusters=50)
        >>> labels = clusterer.fit(embeddings)
        >>> clusters = clusterer.get_cluster_contents(labels, texts)
    """

    def __init__(
        self,
        method: Literal["agglomerative", "hdbscan", "kmeans"] = "agglomerative",
        n_clusters: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        linkage: str = "ward",
        min_cluster_size: int = 5,
    ):
        """
        Initialize the clusterer.

        Args:
            method: Clustering method to use.
            n_clusters: Number of clusters (required for kmeans, optional for agglomerative).
            distance_threshold: Distance threshold for agglomerative (if n_clusters is None).
            linkage: Linkage method for agglomerative clustering.
            min_cluster_size: Minimum cluster size for HDBSCAN.
        """
        self.method = method
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.linkage = linkage
        self.min_cluster_size = min_cluster_size

        self._model = None
        self._labels = None

    @classmethod
    def from_config(cls, config: ClusteringConfig) -> "Clusterer":
        """Create a Clusterer from a configuration object."""
        return cls(
            method=config.method,
            n_clusters=config.n_clusters,
            distance_threshold=config.distance_threshold,
            linkage=config.linkage,
            min_cluster_size=config.min_cluster_size,
        )

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings and return cluster labels.

        Args:
            embeddings: Array of shape (n_samples, n_features).

        Returns:
            Array of cluster labels with shape (n_samples,).
        """
        if self.method == "agglomerative":
            return self._fit_agglomerative(embeddings)
        elif self.method == "hdbscan":
            return self._fit_hdbscan(embeddings)
        elif self.method == "kmeans":
            return self._fit_kmeans(embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

    def _fit_agglomerative(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit using agglomerative clustering."""
        from sklearn.cluster import AgglomerativeClustering

        # Determine n_clusters or distance_threshold
        if self.n_clusters is not None:
            self._model = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage=self.linkage,
            )
        elif self.distance_threshold is not None:
            self._model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                linkage=self.linkage,
            )
        else:
            # Default: use elbow method to determine n_clusters
            n_clusters = self._estimate_n_clusters(embeddings)
            self._model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=self.linkage,
            )

        self._labels = self._model.fit_predict(embeddings)
        return self._labels

    def _fit_hdbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit using HDBSCAN."""
        try:
            import hdbscan
        except ImportError:
            raise ImportError("hdbscan is required for HDBSCAN clustering. Install with: pip install hdbscan")

        self._model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",
        )
        self._labels = self._model.fit_predict(embeddings)
        return self._labels

    def _fit_kmeans(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit using K-means."""
        from sklearn.cluster import KMeans

        if self.n_clusters is None:
            n_clusters = self._estimate_n_clusters(embeddings)
        else:
            n_clusters = self.n_clusters

        self._model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )
        self._labels = self._model.fit_predict(embeddings)
        return self._labels

    def _estimate_n_clusters(self, embeddings: np.ndarray, max_clusters: int = 100) -> int:
        """
        Estimate the optimal number of clusters using the elbow method.

        Uses silhouette score to find a good number of clusters.

        Args:
            embeddings: The embedding matrix.
            max_clusters: Maximum number of clusters to try.

        Returns:
            Estimated optimal number of clusters.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        n_samples = len(embeddings)
        max_k = min(max_clusters, n_samples // 2, 100)
        min_k = 2

        if max_k < min_k:
            return min(n_samples, 10)

        # Try a range of cluster numbers
        k_range = list(range(min_k, max_k + 1, max(1, (max_k - min_k) // 10)))

        best_score = -1
        best_k = min_k

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = kmeans.fit_predict(embeddings)

            # Skip if only one cluster was found
            if len(set(labels)) < 2:
                continue

            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def get_cluster_contents(
        self,
        labels: np.ndarray,
        texts: list[str],
    ) -> dict[int, list[str]]:
        """
        Group texts by their cluster labels.

        Args:
            labels: Cluster labels from fit().
            texts: List of texts corresponding to each embedding.

        Returns:
            Dictionary mapping cluster ID to list of texts.
        """
        clusters = {}
        for i, label in enumerate(labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(texts[i])
        return clusters

    def get_cluster_indices(self, labels: np.ndarray) -> dict[int, list[int]]:
        """
        Group indices by their cluster labels.

        Args:
            labels: Cluster labels from fit().

        Returns:
            Dictionary mapping cluster ID to list of indices.
        """
        clusters = {}
        for i, label in enumerate(labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)
        return clusters

    def get_cluster_stats(self, labels: np.ndarray) -> dict:
        """
        Get statistics about the clustering results.

        Args:
            labels: Cluster labels from fit().

        Returns:
            Dictionary with clustering statistics.
        """
        unique_labels = set(labels)
        n_clusters = len(unique_labels)

        # Handle noise points (label = -1 in HDBSCAN)
        noise_points = sum(1 for l in labels if l == -1)

        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[label] = sum(1 for l in labels if l == label)

        return {
            "n_clusters": n_clusters - (1 if -1 in unique_labels else 0),
            "n_samples": len(labels),
            "noise_points": noise_points,
            "cluster_sizes": cluster_sizes,
            "min_cluster_size": min(cluster_sizes.values()) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes.values()) if cluster_sizes else 0,
            "mean_cluster_size": np.mean(list(cluster_sizes.values())) if cluster_sizes else 0,
        }

    @property
    def labels(self) -> Optional[np.ndarray]:
        """Return the cluster labels from the last fit."""
        return self._labels

    @property
    def n_clusters_found(self) -> Optional[int]:
        """Return the number of clusters found."""
        if self._labels is None:
            return None
        unique = set(self._labels)
        # Don't count noise (-1) as a cluster
        return len([l for l in unique if l != -1])

    def __repr__(self) -> str:
        return f"Clusterer(method='{self.method}', n_clusters={self.n_clusters})"
