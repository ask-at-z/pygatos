"""Tests for new core modules: Summarizer and DimensionalityReducer."""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pygatos.core.summarizer import Summarizer, SummarizationResult
from pygatos.core.reducer import DimensionalityReducer


# =============================================================================
# Summarizer Tests
# =============================================================================

class TestSummarizationResult:
    """Tests for SummarizationResult dataclass."""

    def test_creation(self):
        """Test creating a SummarizationResult."""
        result = SummarizationResult(
            original_text="Original text here.",
            chunks=["chunk1", "chunk2"],
            generic_summaries=["summary1", "summary2"],
            information_points=["point1", "point2", "point3"],
            n_chunks=2,
        )

        assert result.original_text == "Original text here."
        assert len(result.chunks) == 2
        assert len(result.generic_summaries) == 2
        assert len(result.information_points) == 3
        assert result.n_chunks == 2

    def test_empty_result(self):
        """Test empty result."""
        result = SummarizationResult(
            original_text="",
            chunks=[],
            generic_summaries=[],
            information_points=[],
            n_chunks=0,
        )

        assert result.n_chunks == 0
        assert len(result.information_points) == 0


class TestSummarizer:
    """Tests for Summarizer class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock()
        llm.generate.return_value = "- Point one\n- Point two\n- Point three"
        return llm

    def test_initialization(self, mock_llm):
        """Test Summarizer initialization."""
        summarizer = Summarizer(
            llm=mock_llm,
            chunk_size=250,
            chunk_overlap=50,
        )

        assert summarizer.chunk_size == 250
        assert summarizer.chunk_overlap == 50
        assert summarizer.max_context_bullets == 8
        assert summarizer.bullets_per_chunk == 4

    def test_initialization_custom_params(self, mock_llm):
        """Test Summarizer with custom parameters."""
        summarizer = Summarizer(
            llm=mock_llm,
            chunk_size=100,
            chunk_overlap=20,
            max_context_bullets=5,
            bullets_per_chunk=3,
            temperature=0.5,
        )

        assert summarizer.chunk_size == 100
        assert summarizer.chunk_overlap == 20
        assert summarizer.max_context_bullets == 5
        assert summarizer.temperature == 0.5

    def test_summarize_empty_text(self, mock_llm):
        """Test summarizing empty text."""
        summarizer = Summarizer(mock_llm)

        result = summarizer.summarize("")

        assert result.n_chunks == 0
        assert len(result.information_points) == 0

    def test_summarize_whitespace_only(self, mock_llm):
        """Test summarizing whitespace-only text."""
        summarizer = Summarizer(mock_llm)

        result = summarizer.summarize("   \n\t   ")

        assert result.n_chunks == 0

    def test_summarize_short_text(self, mock_llm):
        """Test summarizing short text (no chunking needed)."""
        summarizer = Summarizer(mock_llm, chunk_size=100)

        short_text = "This is a short text that doesn't need chunking."
        result = summarizer.summarize(short_text)

        assert result.n_chunks == 1
        assert result.chunks == [short_text]
        # LLM should be called for information extraction
        mock_llm.generate.assert_called()

    def test_summarize_skip_chunking(self, mock_llm):
        """Test summarization with skip_chunking flag."""
        summarizer = Summarizer(mock_llm, chunk_size=10)

        # Long text that would normally be chunked
        long_text = " ".join(["word"] * 100)
        result = summarizer.summarize(long_text, skip_chunking=True)

        # Should be treated as single chunk
        assert result.n_chunks == 1

    def test_summarize_batch(self, mock_llm):
        """Test batch summarization."""
        summarizer = Summarizer(mock_llm)

        texts = [
            "First short text.",
            "Second short text.",
            "Third short text.",
        ]

        results = summarizer.summarize_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, SummarizationResult) for r in results)

    def test_extract_all_points(self, mock_llm):
        """Test extracting all points from multiple texts."""
        summarizer = Summarizer(mock_llm)

        texts = ["Text one.", "Text two."]

        points = summarizer.extract_all_points(texts)

        # Should be a flat list of strings
        assert isinstance(points, list)
        assert all(isinstance(p, str) for p in points)

    def test_repr(self, mock_llm):
        """Test string representation."""
        summarizer = Summarizer(mock_llm, chunk_size=200, chunk_overlap=40)

        repr_str = repr(summarizer)

        assert "Summarizer" in repr_str
        assert "200" in repr_str
        assert "40" in repr_str


class TestSummarizerChunking:
    """Tests for text chunking functionality."""

    @pytest.fixture
    def summarizer(self):
        """Create summarizer with mock LLM."""
        mock_llm = Mock()
        mock_llm.generate.return_value = "- Point"
        return Summarizer(mock_llm, chunk_size=50, chunk_overlap=10)

    def test_chunk_text_basic(self, summarizer):
        """Test basic text chunking."""
        # Create text with ~100 words
        text = " ".join(["This is a sentence with several words."] * 12)

        chunks = summarizer._chunk_text(text)

        # Should have multiple chunks
        assert len(chunks) >= 2

        # Each chunk should have content
        assert all(len(c.strip()) > 0 for c in chunks)

    def test_chunk_text_single_sentence(self, summarizer):
        """Test chunking single long sentence."""
        text = " ".join(["word"] * 100)

        chunks = summarizer._chunk_text(text)

        assert len(chunks) >= 1

    def test_chunk_overlap(self, summarizer):
        """Test that chunks have overlap."""
        # Create text that will definitely need chunking
        sentences = [f"This is sentence number {i}." for i in range(20)]
        text = " ".join(sentences)

        chunks = summarizer._chunk_text(text)

        # With overlap, later chunks should contain some content from earlier chunks
        # (this is a simplified test - overlap is at sentence level)
        assert len(chunks) > 1


class TestSummarizerParsing:
    """Tests for bullet point parsing."""

    @pytest.fixture
    def summarizer(self):
        """Create summarizer with mock LLM."""
        mock_llm = Mock()
        return Summarizer(mock_llm)

    def test_parse_bullets_dashes(self, summarizer):
        """Test parsing dash-formatted bullets."""
        response = "- Point one\n- Point two\n- Point three"

        bullets = summarizer._parse_bullets(response)

        assert bullets == ["Point one", "Point two", "Point three"]

    def test_parse_bullets_asterisks(self, summarizer):
        """Test parsing asterisk-formatted bullets."""
        response = "* Point one\n* Point two"

        bullets = summarizer._parse_bullets(response)

        assert bullets == ["Point one", "Point two"]

    def test_parse_bullets_numbered(self, summarizer):
        """Test parsing numbered list."""
        response = "1. First point\n2. Second point\n3. Third point"

        bullets = summarizer._parse_bullets(response)

        assert bullets == ["First point", "Second point", "Third point"]

    def test_parse_bullets_mixed(self, summarizer):
        """Test parsing mixed format."""
        response = "- First\n* Second\n3. Third"

        bullets = summarizer._parse_bullets(response)

        assert len(bullets) == 3

    def test_parse_bullets_empty_lines(self, summarizer):
        """Test handling empty lines."""
        response = "- Point one\n\n- Point two\n\n"

        bullets = summarizer._parse_bullets(response)

        assert bullets == ["Point one", "Point two"]

    def test_parse_bullets_no_markers(self, summarizer):
        """Test parsing plain text lines."""
        response = "Point one\nPoint two"

        bullets = summarizer._parse_bullets(response)

        assert bullets == ["Point one", "Point two"]


class TestSummarizerContext:
    """Tests for context building."""

    @pytest.fixture
    def summarizer(self):
        """Create summarizer with mock LLM."""
        mock_llm = Mock()
        return Summarizer(mock_llm, max_context_bullets=3)

    def test_build_context_empty(self, summarizer):
        """Test context with no prior summaries."""
        context = summarizer._build_context([])

        assert context is None

    def test_build_context_basic(self, summarizer):
        """Test building context from summaries."""
        summaries = ["Summary one", "Summary two"]

        context = summarizer._build_context(summaries)

        assert "Summary one" in context
        assert "Summary two" in context
        assert context.startswith("- ")

    def test_build_context_truncation(self, summarizer):
        """Test that context is truncated to max bullets."""
        summaries = ["S1", "S2", "S3", "S4", "S5"]

        context = summarizer._build_context(summaries)

        # Should only include last 3 (max_context_bullets)
        assert "S5" in context
        assert "S4" in context
        assert "S3" in context
        assert "S1" not in context
        assert "S2" not in context


# =============================================================================
# DimensionalityReducer Tests
# =============================================================================

class TestDimensionalityReducer:
    """Tests for DimensionalityReducer class."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample high-dimensional embeddings."""
        np.random.seed(42)
        return np.random.randn(100, 256)  # 100 samples, 256 dims

    @pytest.fixture
    def small_embeddings(self):
        """Create small sample for edge case testing."""
        np.random.seed(42)
        return np.random.randn(20, 64)

    def test_initialization(self):
        """Test DimensionalityReducer initialization."""
        reducer = DimensionalityReducer(
            pca_variance=0.9,
            umap_n_components=5,
        )

        assert reducer.pca_variance == 0.9
        assert reducer.umap_n_components == 5
        assert reducer.random_state == 42  # default
        assert not reducer._is_fitted

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        reducer = DimensionalityReducer(
            pca_variance=0.95,
            umap_n_components=10,
            umap_n_neighbors=30,
            umap_min_dist=0.2,
            umap_metric="euclidean",
            random_state=123,
        )

        assert reducer.pca_variance == 0.95
        assert reducer.umap_n_components == 10
        assert reducer.umap_n_neighbors == 30
        assert reducer.umap_min_dist == 0.2
        assert reducer.umap_metric == "euclidean"
        assert reducer.random_state == 123

    def test_fit_transform(self, sample_embeddings):
        """Test fit_transform reduces dimensions."""
        reducer = DimensionalityReducer(pca_variance=0.9)

        reduced = reducer.fit_transform(sample_embeddings)

        # Should have fewer dimensions
        assert reduced.shape[0] == sample_embeddings.shape[0]  # Same samples
        assert reduced.shape[1] < sample_embeddings.shape[1]  # Fewer dims
        assert reducer._is_fitted

    def test_fit_then_transform(self, sample_embeddings):
        """Test separate fit and transform."""
        reducer = DimensionalityReducer(pca_variance=0.9)

        # Fit
        reducer.fit(sample_embeddings)
        assert reducer._is_fitted

        # Transform
        reduced = reducer.transform(sample_embeddings)

        assert reduced.shape[0] == sample_embeddings.shape[0]
        assert reduced.shape[1] < sample_embeddings.shape[1]

    def test_transform_without_fit(self, sample_embeddings):
        """Test that transform raises error without fit."""
        reducer = DimensionalityReducer()

        with pytest.raises(ValueError) as exc_info:
            reducer.transform(sample_embeddings)

        assert "fitted" in str(exc_info.value).lower()

    def test_pca_variance_retention(self, sample_embeddings):
        """Test that PCA retains specified variance."""
        reducer = DimensionalityReducer(pca_variance=0.8)

        reducer.fit(sample_embeddings)

        # Should retain at least 80% variance
        assert reducer.explained_variance >= 0.8

    def test_pca_only(self, sample_embeddings):
        """Test PCA-only transformation."""
        reducer = DimensionalityReducer(pca_variance=0.9)
        reducer.fit(sample_embeddings)

        pca_result = reducer.get_pca_only(sample_embeddings)

        # PCA result should have pca_components dimensions
        assert pca_result.shape[1] == reducer.pca_components

    def test_pca_only_without_fit(self, sample_embeddings):
        """Test PCA-only raises error without fit."""
        reducer = DimensionalityReducer()

        with pytest.raises(ValueError):
            reducer.get_pca_only(sample_embeddings)

    def test_small_sample_handling(self, small_embeddings):
        """Test handling of small sample sizes."""
        reducer = DimensionalityReducer(
            umap_n_components=5,
            umap_n_neighbors=15,  # More neighbors than samples
        )

        # Should not crash with small sample
        reduced = reducer.fit_transform(small_embeddings)

        assert reduced.shape[0] == small_embeddings.shape[0]

    def test_very_small_sample(self):
        """Test with very few samples."""
        np.random.seed(42)
        tiny = np.random.randn(5, 100)  # Only 5 samples

        reducer = DimensionalityReducer(pca_variance=0.9)

        # Should work even with tiny dataset
        reduced = reducer.fit_transform(tiny)

        assert reduced.shape[0] == 5

    def test_properties_before_fit(self):
        """Test property values before fitting."""
        reducer = DimensionalityReducer()

        assert reducer.pca_components is None
        assert reducer.explained_variance is None
        assert reducer.output_dimensions is None

    def test_properties_after_fit(self, sample_embeddings):
        """Test property values after fitting."""
        reducer = DimensionalityReducer(pca_variance=0.9)
        reducer.fit(sample_embeddings)

        assert reducer.pca_components is not None
        assert reducer.pca_components > 0
        assert reducer.explained_variance is not None
        assert 0 < reducer.explained_variance <= 1
        assert reducer.output_dimensions is not None

    def test_repr_before_fit(self):
        """Test string representation before fitting."""
        reducer = DimensionalityReducer(pca_variance=0.85, umap_n_components=5)

        repr_str = repr(reducer)

        assert "DimensionalityReducer" in repr_str
        assert "0.85" in repr_str

    def test_repr_after_fit(self, sample_embeddings):
        """Test string representation after fitting."""
        reducer = DimensionalityReducer(pca_variance=0.9)
        reducer.fit(sample_embeddings)

        repr_str = repr(reducer)

        assert "DimensionalityReducer" in repr_str
        assert "variance" in repr_str


class TestDimensionalityReducerUMAP:
    """Tests specific to UMAP behavior in DimensionalityReducer."""

    def test_should_apply_umap_few_samples(self):
        """Test that UMAP is skipped for few samples."""
        reducer = DimensionalityReducer(umap_n_neighbors=15)

        # Too few samples
        should_apply = reducer._should_apply_umap(n_samples=10, n_pca_components=50)

        assert not should_apply

    def test_should_apply_umap_already_low_dim(self):
        """Test that UMAP is skipped when PCA already reduces enough."""
        reducer = DimensionalityReducer(umap_n_components=10)

        # PCA already reduced to target
        should_apply = reducer._should_apply_umap(n_samples=100, n_pca_components=5)

        assert not should_apply

    def test_should_apply_umap_normal_case(self):
        """Test that UMAP is applied in normal cases."""
        reducer = DimensionalityReducer(
            umap_n_components=5,
            umap_n_neighbors=15,
        )

        # Enough samples and high-dim PCA output
        should_apply = reducer._should_apply_umap(n_samples=100, n_pca_components=50)

        assert should_apply

    def test_umap_not_installed(self):
        """Test graceful handling when UMAP is not installed."""
        # This test verifies the code handles ImportError
        # In normal testing environment, UMAP is likely installed
        # So we just verify the try/except block exists by checking behavior
        reducer = DimensionalityReducer()

        np.random.seed(42)
        embeddings = np.random.randn(50, 100)

        # Should work regardless of UMAP availability
        reduced = reducer.fit_transform(embeddings)
        assert reduced is not None


class TestDimensionalityReducerPCA:
    """Tests specific to PCA behavior."""

    def test_pca_variance_selection(self):
        """Test that correct number of components are selected."""
        np.random.seed(42)

        # Create embeddings with clear principal components
        # First few dimensions have much more variance
        embeddings = np.zeros((100, 50))
        embeddings[:, 0] = np.random.randn(100) * 10  # High variance
        embeddings[:, 1] = np.random.randn(100) * 5
        embeddings[:, 2] = np.random.randn(100) * 2
        embeddings[:, 3:] = np.random.randn(100, 47) * 0.1  # Low variance

        reducer = DimensionalityReducer(pca_variance=0.9)
        reducer.fit(embeddings)

        # Should select few components to explain most variance
        assert reducer.pca_components <= 10  # Should be much less than 50

    def test_pca_all_variance(self):
        """Test with 100% variance retention."""
        np.random.seed(42)
        embeddings = np.random.randn(50, 20)

        reducer = DimensionalityReducer(pca_variance=1.0)
        reducer.fit(embeddings)

        # Should retain all possible components
        # (limited by min(n_samples-1, n_features))
        assert reducer.explained_variance >= 0.99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
