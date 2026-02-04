"""Tests for I/O modules (loaders and exporters)."""

import sys
import tempfile
from pathlib import Path

import pytest
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pygatos.io.loaders import load_csv, load_data, load_texts, combine_csvs
from pygatos.io.exporters import (
    export_codebook_json,
    export_codebook_csv,
    export_code_frequencies,
    export_themes_hierarchy,
)
from pygatos.core.codebook import Codebook, Code, Theme


@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "text": [
            "This is the first response.",
            "Second response here.",
            "Third one is short.",
            "Fourth response about something else.",
            "Fifth and final response.",
        ],
        "category": ["A", "B", "A", "C", "B"],
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_codebook():
    """Create a sample codebook for testing."""
    codebook = Codebook()

    # Add accepted codes
    codes = [
        Code(name="Greed", definition="Expressions of greed or selfishness"),
        Code(name="Supply Issues", definition="Supply chain problems"),
        Code(name="Government", definition="References to government actions"),
    ]
    for code in codes:
        code.evaluation_order = codes.index(code) + 1
        codebook.add_code(code, accepted=True)

    # Add rejected code
    rejected = Code(name="Duplicate", definition="Similar to Greed")
    rejected.evaluation_order = 4
    codebook.add_code(rejected, accepted=False)

    # Add theme
    theme = Theme(
        name="Economic Factors",
        definition="Economic-related themes",
        codes=codes[:2],
    )
    codebook.add_theme(theme)

    return codebook


class TestLoaders:
    """Tests for data loading functions."""

    def test_load_csv_basic(self, sample_csv):
        """Test basic CSV loading."""
        df = load_csv(sample_csv, text_column="text")
        assert len(df) == 5
        assert "text" in df.columns

    def test_load_csv_with_id(self, sample_csv):
        """Test CSV loading with ID column."""
        df = load_csv(sample_csv, text_column="text", id_column="id")
        assert "id" in df.columns
        assert list(df["id"]) == [1, 2, 3, 4, 5]

    def test_load_csv_missing_column(self, sample_csv):
        """Test error when column is missing."""
        with pytest.raises(ValueError) as exc_info:
            load_csv(sample_csv, text_column="nonexistent")
        assert "not found" in str(exc_info.value)

    def test_load_csv_file_not_found(self, tmp_path):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "nonexistent.csv", text_column="text")

    def test_load_data_auto_detect(self, sample_csv):
        """Test auto-detection of file format."""
        df = load_data(sample_csv, text_column="text")
        assert len(df) == 5

    def test_load_texts(self, sample_csv):
        """Test extracting texts and IDs."""
        texts, ids = load_texts(sample_csv, text_column="text", id_column="id")
        assert len(texts) == 5
        assert len(ids) == 5
        assert texts[0] == "This is the first response."
        assert ids[0] == 1

    def test_load_texts_no_id(self, sample_csv):
        """Test loading texts without ID column."""
        texts, ids = load_texts(sample_csv, text_column="text")
        assert len(texts) == 5
        assert ids == list(range(5))

    def test_combine_csvs(self, tmp_path):
        """Test combining multiple CSV files."""
        # Create two CSV files
        csv1 = tmp_path / "data1.csv"
        csv2 = tmp_path / "data2.csv"

        pd.DataFrame({"text": ["a", "b"]}).to_csv(csv1, index=False)
        pd.DataFrame({"text": ["c", "d"]}).to_csv(csv2, index=False)

        combined = combine_csvs([csv1, csv2], text_column="text")
        assert len(combined) == 4
        assert "source_file" in combined.columns


class TestExporters:
    """Tests for data export functions."""

    def test_export_codebook_json(self, sample_codebook, tmp_path):
        """Test exporting codebook to JSON."""
        json_path = tmp_path / "codebook.json"
        result = export_codebook_json(sample_codebook, json_path)

        assert result.exists()
        assert result.suffix == ".json"

        # Verify content
        import json
        with open(json_path) as f:
            data = json.load(f)

        assert "accepted_codes" in data
        assert len(data["accepted_codes"]) == 3

    def test_export_codebook_json_with_rejected(self, sample_codebook, tmp_path):
        """Test exporting codebook with rejected codes."""
        json_path = tmp_path / "codebook.json"
        export_codebook_json(sample_codebook, json_path, include_rejected=True)

        import json
        with open(json_path) as f:
            data = json.load(f)

        assert "rejected_codes" in data
        assert len(data["rejected_codes"]) == 1

    def test_export_codebook_csv(self, sample_codebook, tmp_path):
        """Test exporting codebook to CSV."""
        csv_path = tmp_path / "codebook.csv"
        result = export_codebook_csv(sample_codebook, csv_path)

        assert result.exists()

        df = pd.read_csv(csv_path)
        assert len(df) >= 3  # At least accepted codes

    def test_export_code_frequencies(self, tmp_path):
        """Test exporting code frequencies."""
        frequencies = {"Greed": 10, "Supply": 5, "Government": 3}
        csv_path = tmp_path / "frequencies.csv"

        result = export_code_frequencies(frequencies, csv_path)

        assert result.exists()

        df = pd.read_csv(csv_path)
        assert "code" in df.columns
        assert "count" in df.columns
        assert "percentage" in df.columns

    def test_export_themes_hierarchy_markdown(self, sample_codebook, tmp_path):
        """Test exporting theme hierarchy as Markdown."""
        md_path = tmp_path / "themes.md"
        result = export_themes_hierarchy(sample_codebook, md_path, format="markdown")

        assert result.exists()

        content = md_path.read_text()
        assert "# Codebook Theme Hierarchy" in content
        assert "Economic Factors" in content

    def test_export_themes_hierarchy_json(self, sample_codebook, tmp_path):
        """Test exporting theme hierarchy as JSON."""
        json_path = tmp_path / "themes.json"
        result = export_themes_hierarchy(sample_codebook, json_path, format="json")

        assert result.exists()

        import json
        with open(json_path) as f:
            data = json.load(f)

        assert "themes" in data
        assert len(data["themes"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
