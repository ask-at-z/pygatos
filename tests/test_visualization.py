"""Tests for visualization modules."""

import sys
from pathlib import Path

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pygatos.visualization.frequencies import (
    plot_code_frequencies,
    plot_code_distribution,
    plot_pie_chart,
)
from pygatos.visualization.growth import (
    plot_codebook_growth,
    plot_acceptance_rate,
    plot_saturation_analysis,
)
from pygatos.visualization.clusters import (
    plot_clusters_2d,
    plot_cluster_sizes,
)
from pygatos.core.codebook import Codebook, Code


@pytest.fixture
def sample_frequencies():
    """Sample code frequency data."""
    return {
        "Greed": 25,
        "Supply Chain": 20,
        "Government": 15,
        "Inflation": 12,
        "Wages": 8,
        "Policy": 5,
    }


@pytest.fixture
def sample_codebook():
    """Create a sample codebook with evaluation order."""
    codebook = Codebook()

    # Add accepted codes
    for i, name in enumerate(["Greed", "Supply", "Government", "Wages"]):
        code = Code(name=name, definition=f"Definition for {name}")
        code.evaluation_order = i + 1
        codebook.add_code(code, accepted=True)

    # Add rejected codes
    for i, name in enumerate(["Duplicate1", "Duplicate2", "Duplicate3"]):
        code = Code(name=name, definition=f"Definition for {name}")
        code.evaluation_order = i + 5
        codebook.add_code(code, accepted=False)

    return codebook


@pytest.fixture
def sample_embeddings():
    """Sample 2D embeddings for cluster visualization."""
    np.random.seed(42)
    return np.random.randn(50, 2)


@pytest.fixture
def sample_labels():
    """Sample cluster labels."""
    return np.array([0] * 20 + [1] * 15 + [2] * 15)


class TestFrequencyCharts:
    """Tests for frequency visualization functions."""

    def test_plot_code_frequencies_basic(self, sample_frequencies):
        """Test basic frequency bar chart."""
        fig = plot_code_frequencies(sample_frequencies)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_code_frequencies_top_n(self, sample_frequencies):
        """Test frequency chart with top N limit."""
        fig = plot_code_frequencies(sample_frequencies, top_n=3)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_code_frequencies_vertical(self, sample_frequencies):
        """Test vertical bar chart."""
        fig = plot_code_frequencies(sample_frequencies, horizontal=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_code_frequencies_save(self, sample_frequencies, tmp_path):
        """Test saving frequency chart to file."""
        save_path = tmp_path / "frequencies.png"
        fig = plot_code_frequencies(sample_frequencies, save_path=save_path)
        assert save_path.exists()
        plt.close(fig)

    def test_plot_code_distribution(self, sample_frequencies):
        """Test frequency distribution histogram."""
        fig = plot_code_distribution(sample_frequencies)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_pie_chart(self, sample_frequencies):
        """Test pie chart generation."""
        fig = plot_pie_chart(sample_frequencies)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_pie_chart_top_n(self, sample_frequencies):
        """Test pie chart with top N and "Other" grouping."""
        fig = plot_pie_chart(sample_frequencies, top_n=3)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestGrowthCharts:
    """Tests for growth visualization functions."""

    def test_plot_codebook_growth(self, sample_codebook):
        """Test codebook growth chart."""
        fig = plot_codebook_growth(sample_codebook)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_codebook_growth_no_rejected(self, sample_codebook):
        """Test growth chart without rejected codes."""
        fig = plot_codebook_growth(sample_codebook, show_rejected=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_codebook_growth_save(self, sample_codebook, tmp_path):
        """Test saving growth chart to file."""
        save_path = tmp_path / "growth.png"
        fig = plot_codebook_growth(sample_codebook, save_path=save_path)
        assert save_path.exists()
        plt.close(fig)

    def test_plot_acceptance_rate(self, sample_codebook):
        """Test acceptance rate chart."""
        fig = plot_acceptance_rate(sample_codebook, window_size=3)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_saturation_analysis(self, sample_codebook):
        """Test multi-panel saturation analysis."""
        fig = plot_saturation_analysis(sample_codebook)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_saturation_analysis_save(self, sample_codebook, tmp_path):
        """Test saving saturation analysis to file."""
        save_path = tmp_path / "saturation.png"
        fig = plot_saturation_analysis(sample_codebook, save_path=save_path)
        assert save_path.exists()
        plt.close(fig)


class TestClusterCharts:
    """Tests for cluster visualization functions."""

    def test_plot_clusters_2d(self, sample_embeddings, sample_labels):
        """Test 2D cluster scatter plot."""
        fig = plot_clusters_2d(sample_embeddings, sample_labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_clusters_2d_no_legend(self, sample_embeddings, sample_labels):
        """Test cluster plot without legend."""
        fig = plot_clusters_2d(sample_embeddings, sample_labels, show_legend=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_clusters_2d_save(self, sample_embeddings, sample_labels, tmp_path):
        """Test saving cluster plot to file."""
        save_path = tmp_path / "clusters.png"
        fig = plot_clusters_2d(sample_embeddings, sample_labels, save_path=save_path)
        assert save_path.exists()
        plt.close(fig)

    def test_plot_clusters_2d_wrong_dimension(self, sample_labels):
        """Test error with non-2D embeddings."""
        embeddings_3d = np.random.randn(50, 3)
        with pytest.raises(ValueError) as exc_info:
            plot_clusters_2d(embeddings_3d, sample_labels)
        assert "2D" in str(exc_info.value)

    def test_plot_cluster_sizes(self, sample_labels):
        """Test cluster size bar chart."""
        fig = plot_cluster_sizes(sample_labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_cluster_sizes_save(self, sample_labels, tmp_path):
        """Test saving cluster size chart."""
        save_path = tmp_path / "cluster_sizes.png"
        fig = plot_cluster_sizes(sample_labels, save_path=save_path)
        assert save_path.exists()
        plt.close(fig)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_frequencies(self):
        """Test with empty frequency dict."""
        fig = plot_code_frequencies({})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_code_frequencies(self):
        """Test with single code."""
        fig = plot_code_frequencies({"Only One": 10})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_codebook_growth(self):
        """Test growth chart with empty codebook."""
        codebook = Codebook()
        fig = plot_codebook_growth(codebook)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_many_clusters(self):
        """Test cluster plot with many clusters."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 2)
        labels = np.array(list(range(25)) * 4)  # 25 clusters

        fig = plot_clusters_2d(embeddings, labels, max_legend_items=10)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
