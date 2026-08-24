"""Configuration dataclasses for pygatos."""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class SummarizationConfig:
    """Configuration for text summarization/chunking."""

    chunk_size: int = 250
    """Size of text chunks in tokens."""

    chunk_overlap: int = 50
    """Overlap between consecutive chunks in tokens."""

    max_context_bullets: int = 8
    """Maximum number of prior generic summary bullets to include as context."""

    bullets_per_chunk: int = 4
    """Target number of generic summary bullets per chunk."""

    hardened_prompts: bool = False
    """Use instruction/data-separated extraction prompts (extraction focus in the SYSTEM
    prompt; chunk wrapped in <source_text> tags with tag occurrences escaped). Recommended
    for text not authored by the researcher. Default False preserves existing behavior."""


@dataclass
class EmbeddingConfig:
    """Configuration for text embedding."""

    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    """HuggingFace model name for embeddings."""

    device: str = "auto"
    """Device for embedding model: 'auto', 'cpu', 'cuda', 'mps'."""

    batch_size: int = 32
    """Batch size for embedding."""

    normalize: bool = True
    """Whether to normalize embeddings to unit length."""


@dataclass
class DimensionalityReductionConfig:
    """Configuration for dimensionality reduction."""

    pca_variance: float = 0.9
    """Variance to retain in PCA (determines number of components)."""

    umap_n_components: int = 5
    """Number of dimensions for UMAP output."""

    umap_n_neighbors: int = 15
    """Number of neighbors for UMAP."""

    umap_min_dist: float = 0.1
    """Minimum distance for UMAP."""

    umap_metric: str = "cosine"
    """Distance metric for UMAP."""


@dataclass
class ClusteringConfig:
    """Configuration for clustering."""

    method: Literal["agglomerative", "hdbscan", "kmeans"] = "agglomerative"
    """Clustering method to use."""

    n_clusters: Optional[int] = None
    """Number of clusters (if None, determined automatically or via target_cluster_size)."""

    target_cluster_size: Optional[int] = 10
    """Target number of items per cluster. Used to auto-calculate n_clusters if n_clusters is None.
    Smaller values = more clusters = more specific codes. Default 10 gives ~1 cluster per 10 items."""

    distance_threshold: Optional[float] = None
    """Distance threshold for agglomerative clustering."""

    linkage: str = "ward"
    """Linkage method for agglomerative clustering."""

    min_cluster_size: int = 5
    """Minimum cluster size for HDBSCAN."""

    def compute_n_clusters(self, n_items: int) -> int:
        """Compute the number of clusters based on configuration and data size.

        Args:
            n_items: Number of items to cluster.

        Returns:
            Number of clusters to use.
        """
        if self.n_clusters is not None:
            return self.n_clusters
        if self.target_cluster_size is not None and self.target_cluster_size > 0:
            # Compute clusters based on target size, with a minimum of 10 clusters
            return max(10, n_items // self.target_cluster_size)
        # Fallback: 1 cluster per 15 items
        return max(10, n_items // 15)


@dataclass
class NoveltyConfig:
    """Configuration for novelty evaluation."""

    policy: Literal["reject-unless-distinct", "keep-unless-duplicate"] = "reject-unless-distinct"
    """Instructed novelty policy for the Stage 2 consolidation decision.

    - "reject-unless-distinct" (default): the published policy (prompt V2) -- reject a candidate
      that is a specific instance of a broader code; when in doubt, reject. This is the behavior
      the GATOS paper shipped and what its pinned reproductions expect; it is kept as the default
      so published runs stay reproducible.
    - "keep-unless-duplicate": the validated repair -- reject only true duplicates (same concept
      at the same granularity); a specific code under a broader one is novel. Multi-seed factorial
      evaluation on two model stacks showed this single instruction change moves ground-truth
      recall by +0.25 to +0.27 (~9 SE) with no overlap between policies; the other consolidation
      levers (threshold, comparison set, retrieval context, temperature) had no effect outside the
      noise floor. Known cost: the codebook grows to 3-8x the human codebook, and on weak
      generator models the judged duplicate rate rises -- measure it with
      pygatos.diagnostics.duplicate_rate (no ground truth needed).

    Mutually exclusive with the custom novelty prompt overrides on GATOSConfig
    (novelty_evaluation_system_prompt / novelty_evaluation_user_prompt): setting both raises."""

    similarity_threshold: float = 0.8
    """Cosine similarity threshold for auto-rejection in Stage 1."""

    top_k_rag: int = 8
    """Number of similar codes to retrieve for RAG in Stage 2."""

    include_rejected_in_rag: bool = True
    """Whether to include rejected codes in Stage 2 RAG context."""

    temperature: Optional[float] = None
    """Optional temperature override for Stage 2 LLM novelty decisions. None uses the
    backend default (typically 0.7); 0.0 makes decisions deterministic given identical
    context, removing sampling noise from accept/reject verdicts."""

    stage1_compare_accepted_only: bool = False
    """If True, Stage 1 auto-rejection compares only against ACCEPTED codes. The default
    (False, original behavior) also compares against previously REJECTED codes, which makes
    rejection contagious: one wrongly-rejected code auto-rejects every later similar code."""


@dataclass
class ApplicationConfig:
    """Configuration for codebook application."""

    top_k: int = 10
    """Number of candidate codes to retrieve for LLM judgment."""


@dataclass
class ThemeConfig:
    """Configuration for post-hoc code -> theme (re)assignment.

    After themes are generated (by clustering + naming), this optional stage revisits
    each accepted code and places it under its single best-fitting theme: rank themes by
    embedding cosine similarity, keep the top match when it is confident, and defer the
    low-confidence codes to the LLM, which chooses among the top-k candidate themes. This
    restores the per-code assignment + validation step from the original Auto-QDA design,
    which cluster-then-name alone does not provide (a code's theme is otherwise just the
    cluster it fell into, never re-checked against the other themes)."""

    assignment_top_k: int = 5
    """Number of candidate themes (by embedding similarity) shown to the LLM for a
    low-confidence code. Capped at the number of themes."""

    assignment_confidence_threshold: float = 0.8
    """If a code's top theme match has cosine similarity >= this, assign it directly
    (no LLM call). Below it, the LLM adjudicates among the top-k candidates."""

    validate_assignments: bool = True
    """If True, low-confidence codes are routed to the LLM to pick the best theme.
    If False, every code is assigned to its top embedding match (pure similarity)."""

    refine_after_generation: bool = False
    """If True, generate_themes() runs this reassignment as a final step. Default False so
    existing pipeline behavior (and reproducibility of prior runs) is unchanged; call
    pipeline.reassign_code_themes() explicitly, or opt in via this flag."""


@dataclass
class LLMConfig:
    """Configuration for LLM backend."""

    backend: Literal["ollama", "cerebras", "openrouter"] = "ollama"
    """Which LLM backend to use: 'ollama' for local inference, 'cerebras' or 'openrouter'
    for cloud APIs."""

    model: str = "qwen3:30b-a3b-instruct-2507-q4_K_M"
    """Model name. For Ollama, use local model names. For Cerebras, use 'llama-3.3-70b', etc."""

    base_url: str = "http://localhost:11434"
    """Base URL for Ollama API (only used when backend='ollama')."""

    api_key: Optional[str] = None
    """API key for cloud backends like Cerebras. If None, reads from environment variable."""

    temperature: float = 0.7
    """Sampling temperature."""

    max_tokens: int = 2048
    """Maximum tokens to generate."""

    timeout: int = 120
    """Request timeout in seconds."""

    debug: bool = False
    """If True, log all LLM prompts and responses."""

    rate_limit_delay: float = 0.1
    """Seconds to wait between API calls for rate-limited backends like Cerebras free tier (30 req/min)."""

    seed: Optional[int] = None
    """Sampling seed forwarded to the LLM backend (Ollama ``options.seed``).

    Distinct from GATOSConfig.random_seed, which seeds numpy/UMAP/clustering ONLY and never
    reaches the model. Set this to make LLM sampling reproducible; leave None for unseeded
    sampling. Note that a seed makes sampling *reproducible*, not deterministic across
    different server/model builds."""

    def create_backend(self):
        """
        Create and return the appropriate LLM backend based on configuration.

        Returns:
            BaseLLM: The configured LLM backend instance.

        Raises:
            ValueError: If an unknown backend is specified.
        """
        if self.backend == "ollama":
            from pygatos.llm.ollama import OllamaBackend

            return OllamaBackend.from_config(self)
        elif self.backend == "cerebras":
            from pygatos.llm.cerebras import CerebrasBackend

            return CerebrasBackend.from_config(self)
        elif self.backend == "openrouter":
            from pygatos.llm.openrouter import OpenRouterBackend

            return OpenRouterBackend.from_config(self)
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")


@dataclass
class CacheConfig:
    """Configuration for caching."""

    enabled: bool = True
    """Whether caching is enabled."""

    cache_dir: str = ".pygatos_cache"
    """Directory for cache files."""

    cache_embeddings: bool = True
    """Whether to cache embeddings."""

    cache_llm_responses: bool = True
    """Whether to cache LLM responses."""


@dataclass
class GATOSConfig:
    """
    Main configuration for the GATOS pipeline.

    This configuration class aggregates all sub-configurations and provides
    sensible defaults for the entire pipeline.

    Example:
        >>> config = GATOSConfig()
        >>> config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        >>> config.llm.model = "llama3:8b"
    """

    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    """Configuration for text summarization."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    """Configuration for text embedding."""

    dim_reduction: DimensionalityReductionConfig = field(
        default_factory=DimensionalityReductionConfig
    )
    """Configuration for dimensionality reduction."""

    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    """Configuration for clustering."""

    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    """Configuration for novelty evaluation."""

    application: ApplicationConfig = field(default_factory=ApplicationConfig)
    """Configuration for codebook application."""

    theme: ThemeConfig = field(default_factory=ThemeConfig)
    """Configuration for post-hoc code -> theme (re)assignment."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    """Configuration for LLM backend."""

    cache: CacheConfig = field(default_factory=CacheConfig)
    """Configuration for caching."""

    random_seed: Optional[int] = 42
    """Random seed for reproducibility."""

    verbose: bool = True
    """Whether to print progress information."""

    study_context: Optional[str] = None
    """Optional context about the study/dataset to include in LLM prompts.

    This helps the LLM understand the domain and produce more relevant codes.
    Example: "Survey responses from US adults about their concerns regarding inflation
    and its impact on their daily lives and financial decisions."
    """

    code_suggestion_system_prompt: Optional[str] = None
    """Optional custom system prompt for code suggestion. If None, uses default."""

    code_suggestion_user_prompt: Optional[str] = None
    """Optional custom user prompt for code suggestion. Must contain {texts} placeholder."""

    novelty_evaluation_system_prompt: Optional[str] = None
    """Optional custom system prompt for novelty evaluation. If None, uses default."""

    novelty_evaluation_user_prompt: Optional[str] = None
    """Optional custom user prompt for novelty evaluation. Must contain {code_name}, {code_definition}, {existing_codes} placeholders."""

    code_application_system_prompt: Optional[str] = None
    """Optional custom system prompt for code application. If None, uses default."""

    code_application_user_prompt: Optional[str] = None
    """Optional custom user prompt for code application. Must contain {information_point}, {source_text}, {codes} placeholders."""

    @classmethod
    def from_dict(cls, config_dict: dict) -> "GATOSConfig":
        """Create a GATOSConfig from a dictionary."""
        config = cls()

        # Map sub-config names to their classes
        sub_configs = {
            "summarization": SummarizationConfig,
            "embedding": EmbeddingConfig,
            "dim_reduction": DimensionalityReductionConfig,
            "clustering": ClusteringConfig,
            "novelty": NoveltyConfig,
            "application": ApplicationConfig,
            "llm": LLMConfig,
            "cache": CacheConfig,
        }

        for key, value in config_dict.items():
            if key in sub_configs and isinstance(value, dict):
                setattr(config, key, sub_configs[key](**value))
            elif hasattr(config, key):
                setattr(config, key, value)

        return config

    def to_dict(self) -> dict:
        """Convert the configuration to a dictionary."""
        from dataclasses import asdict

        return asdict(self)
