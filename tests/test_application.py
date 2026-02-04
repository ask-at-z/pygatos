"""Tests for code application module."""

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import directly to avoid triggering sklearn import through core/__init__.py
from pygatos.core.codebook import Codebook, Code
from pygatos.application.code_applier import CodeApplier, ApplicationResult


@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns predictable responses."""
    llm = Mock()
    llm.generate_json.return_value = {
        "applied_codes": ["Greed", "Corporate Greed"],
        "reasoning": "The text discusses corporate and systemic greed.",
    }
    return llm


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns random embeddings."""
    embedder = Mock()
    np.random.seed(42)

    def embed_func(texts):
        if isinstance(texts, str):
            return np.random.randn(1024)
        return np.random.randn(len(texts), 1024)

    embedder.embed.side_effect = embed_func
    return embedder


@pytest.fixture
def sample_codebook():
    """Create a sample codebook with codes."""
    codebook = Codebook()

    codes = [
        ("Greed", "General expression of greed as a cause."),
        ("Corporate Greed", "Corporations prioritizing profit over people."),
        ("Government Failure", "Government incompetence or mismanagement."),
        ("Supply Issues", "Supply chain disruptions."),
    ]

    np.random.seed(42)
    for name, definition in codes:
        code = Code(name=name, definition=definition)
        code.embedding = np.random.randn(1024)
        codebook.add_code(code, accepted=True)

    return codebook


class TestApplicationResult:
    """Tests for ApplicationResult dataclass."""

    def test_creation(self):
        """Test creating an ApplicationResult."""
        codes = [Code(name="Test", definition="Test def")]

        result = ApplicationResult(
            text="Test text",
            applied_codes=codes,
            reasoning="Test reasoning",
            candidate_codes=codes,
        )

        assert result.text == "Test text"
        assert len(result.applied_codes) == 1
        assert result.reasoning == "Test reasoning"
        assert result.text_embedding is None

    def test_with_embedding(self):
        """Test ApplicationResult with embedding."""
        embedding = np.random.randn(1024)

        result = ApplicationResult(
            text="Test",
            applied_codes=[],
            reasoning="",
            candidate_codes=[],
            text_embedding=embedding,
        )

        assert result.text_embedding is not None
        assert result.text_embedding.shape == (1024,)


class TestCodeApplier:
    """Tests for CodeApplier class."""

    def test_initialization(self, mock_llm, mock_embedder):
        """Test CodeApplier initialization."""
        applier = CodeApplier(
            llm=mock_llm,
            embedder=mock_embedder,
            top_k=5,
        )

        assert applier.top_k == 5
        assert applier.temperature is None

    def test_initialization_with_temperature(self, mock_llm, mock_embedder):
        """Test CodeApplier initialization with temperature."""
        applier = CodeApplier(
            llm=mock_llm,
            embedder=mock_embedder,
            top_k=10,
            temperature=0.3,
        )

        assert applier.temperature == 0.3

    def test_apply_empty_codebook(self, mock_llm, mock_embedder):
        """Test applying with empty codebook."""
        applier = CodeApplier(mock_llm, mock_embedder)
        codebook = Codebook()

        result = applier.apply("Test text", codebook)

        assert result.applied_codes == []
        assert "No codes" in result.reasoning

    def test_apply_empty_text(self, mock_llm, mock_embedder, sample_codebook):
        """Test applying to empty text."""
        applier = CodeApplier(mock_llm, mock_embedder)

        result = applier.apply("", sample_codebook)

        assert result.applied_codes == []
        assert "Empty text" in result.reasoning

    def test_apply_whitespace_only(self, mock_llm, mock_embedder, sample_codebook):
        """Test applying to whitespace-only text."""
        applier = CodeApplier(mock_llm, mock_embedder)

        result = applier.apply("   \n\t  ", sample_codebook)

        assert result.applied_codes == []
        assert "Empty text" in result.reasoning

    def test_apply_basic(self, mock_llm, mock_embedder, sample_codebook):
        """Test basic code application."""
        applier = CodeApplier(mock_llm, mock_embedder, top_k=3)

        result = applier.apply(
            "Companies are greedy and only care about profits.",
            sample_codebook,
        )

        # Should have called the embedder
        mock_embedder.embed.assert_called()

        # Should have called the LLM
        mock_llm.generate_json.assert_called_once()

        # Check result structure
        assert result.text is not None
        assert isinstance(result.applied_codes, list)
        assert result.text_embedding is not None

    def test_apply_with_verbose(self, mock_llm, mock_embedder, sample_codebook, caplog):
        """Test application with verbose logging."""
        applier = CodeApplier(mock_llm, mock_embedder)

        import logging
        caplog.set_level(logging.INFO)

        result = applier.apply(
            "Test text about corporate behavior.",
            sample_codebook,
            verbose=True,
        )

        assert result is not None

    def test_apply_batch(self, mock_llm, mock_embedder, sample_codebook):
        """Test batch application."""
        applier = CodeApplier(mock_llm, mock_embedder)

        texts = [
            "Companies are greedy.",
            "The government failed us.",
            "Supply chain issues everywhere.",
        ]

        results = applier.apply_batch(texts, sample_codebook)

        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    def test_apply_with_details(self, mock_llm, mock_embedder, sample_codebook):
        """Test batch application with full details."""
        applier = CodeApplier(mock_llm, mock_embedder)

        texts = [
            "Companies are greedy.",
            "The government failed us.",
        ]

        results = applier.apply_with_details(texts, sample_codebook)

        assert len(results) == 2
        assert all(isinstance(r, ApplicationResult) for r in results)

    def test_get_code_frequencies(self, mock_llm, mock_embedder):
        """Test frequency calculation from results."""
        applier = CodeApplier(mock_llm, mock_embedder)

        code1 = Code(name="Greed", definition="...")
        code2 = Code(name="Government", definition="...")

        results = [
            ApplicationResult(
                text="t1", applied_codes=[code1],
                reasoning="", candidate_codes=[]
            ),
            ApplicationResult(
                text="t2", applied_codes=[code1, code2],
                reasoning="", candidate_codes=[]
            ),
            ApplicationResult(
                text="t3", applied_codes=[code1],
                reasoning="", candidate_codes=[]
            ),
        ]

        frequencies = applier.get_code_frequencies(results)

        assert frequencies["Greed"] == 3
        assert frequencies["Government"] == 1

    def test_repr(self, mock_llm, mock_embedder):
        """Test string representation."""
        applier = CodeApplier(mock_llm, mock_embedder, top_k=15)

        repr_str = repr(applier)

        assert "CodeApplier" in repr_str
        assert "15" in repr_str


class TestCodeApplierRetrieval:
    """Tests for the retrieval stage of CodeApplier."""

    def test_retrieve_candidates_top_k(self, mock_llm, mock_embedder, sample_codebook):
        """Test that retrieval respects top_k."""
        applier = CodeApplier(mock_llm, mock_embedder, top_k=2)

        # Embed text
        text_embedding = mock_embedder.embed("Test text")

        # Call internal retrieval method
        candidates = applier._retrieve_candidates(text_embedding, sample_codebook)

        # Should return at most top_k codes
        assert len(candidates) <= 2

    def test_ensure_code_embeddings(self, mock_llm, mock_embedder):
        """Test that code embeddings are generated if missing."""
        applier = CodeApplier(mock_llm, mock_embedder)

        # Create codebook without embeddings
        codebook = Codebook()
        code = Code(name="Test", definition="Test definition")
        code.embedding = None
        codebook.add_code(code, accepted=True)

        # Ensure embeddings
        applier._ensure_code_embeddings(codebook)

        # Should have called embedder
        mock_embedder.embed.assert_called()


class TestCodeApplierJudgment:
    """Tests for the judgment stage of CodeApplier."""

    def test_judge_case_insensitive_match(self, mock_llm, mock_embedder):
        """Test that code name matching is case-insensitive."""
        # LLM returns lowercase name
        mock_llm.generate_json.return_value = {
            "applied_codes": ["greed"],  # lowercase
            "reasoning": "Test",
        }

        applier = CodeApplier(mock_llm, mock_embedder)

        candidates = [Code(name="Greed", definition="...")]  # Title case

        applied, reasoning, text_summary, analysis = applier._judge_codes("Test text", candidates)

        assert len(applied) == 1
        assert applied[0].name == "Greed"

    def test_judge_handles_llm_error(self, mock_llm, mock_embedder):
        """Test graceful handling of LLM errors."""
        mock_llm.generate_json.side_effect = Exception("LLM error")

        applier = CodeApplier(mock_llm, mock_embedder)

        candidates = [Code(name="Test", definition="...")]

        applied, reasoning, text_summary, analysis = applier._judge_codes("Test text", candidates)

        assert applied == []
        assert "Error" in reasoning
        assert text_summary is None
        assert analysis is None


class TestCodeApplierEdgeCases:
    """Tests for edge cases in CodeApplier."""

    def test_apply_no_similar_codes(self, mock_llm, mock_embedder):
        """Test when no similar codes are found."""
        # Create an embedder that returns very different embeddings
        embedder = Mock()
        embedder.embed.return_value = np.zeros(1024)  # Zero vector

        applier = CodeApplier(mock_llm, embedder, top_k=5)

        codebook = Codebook()
        code = Code(name="Test", definition="...")
        code.embedding = np.ones(1024)  # Different from zero
        codebook.add_code(code, accepted=True)

        # This should still work (just return empty or the available code)
        result = applier.apply("Test", codebook)
        assert result is not None

    def test_single_code_codebook(self, mock_llm, mock_embedder):
        """Test with codebook containing single code."""
        mock_llm.generate_json.return_value = {
            "applied_codes": ["Single"],
            "reasoning": "Only one code.",
        }

        applier = CodeApplier(mock_llm, mock_embedder, top_k=10)

        codebook = Codebook()
        code = Code(name="Single", definition="The only code")
        code.embedding = np.random.randn(1024)
        codebook.add_code(code, accepted=True)

        result = applier.apply("Test text", codebook)

        assert len(result.candidate_codes) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
