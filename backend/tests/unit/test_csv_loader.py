"""Tests for CSV document loader.

Tests CSV file loading, parsing, normalization, and document creation.
"""

import pytest
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

from src.csv_loader import CSVDocumentLoader, CSVConfig, load_csv_documents
from src.document_processor import Document, DocumentMetadata


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def csv_config(temp_data_dir):
    """Create test CSV configuration."""
    return CSVConfig(data_dir=temp_data_dir)


@pytest.fixture
def csv_loader(csv_config):
    """Create CSV loader with test config."""
    return CSVDocumentLoader(csv_config)


@pytest.fixture
def sample_csv(temp_data_dir):
    """Create sample CSV file."""
    csv_path = Path(temp_data_dir) / "test_data.csv"
    data = {
        "id": ["test_001", "test_002", "test_003"],
        "title": ["First Item", "Second Item", "Third Item"],
        "category": ["A", "B", "C"],
        "description": ["Description 1", "Description 2", "Description 3"],
        "tags": ["tag1|tag2", "tag3", "tag4|tag5|tag6"]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


# ============================================================================
# CSVDocumentLoader Initialization Tests
# ============================================================================


class TestCSVLoaderInitialization:
    """Tests for CSV loader initialization."""
    
    def test_loader_creation_with_default_config(self):
        """Test creating loader with default config."""
        loader = CSVDocumentLoader()
        assert loader.config is not None
        assert loader.data_dir == Path("data")
    
    def test_loader_creation_with_custom_config(self, csv_config):
        """Test creating loader with custom config."""
        loader = CSVDocumentLoader(csv_config)
        assert loader.config == csv_config
        assert loader.data_dir == Path(csv_config.data_dir)
    
    def test_loader_creates_data_directory(self):
        """Test that loader creates data directory if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_dir = Path(tmpdir) / "new_data"
            assert not nonexistent_dir.exists()
            
            config = CSVConfig(data_dir=str(nonexistent_dir))
            loader = CSVDocumentLoader(config)
            
            assert nonexistent_dir.exists()


# ============================================================================
# CSV File Discovery Tests
# ============================================================================


class TestCSVFileDiscovery:
    """Tests for finding CSV files."""
    
    def test_find_csv_files_empty_directory(self, csv_loader):
        """Test finding CSV files in empty directory."""
        files = csv_loader.find_csv_files()
        assert files == []
    
    def test_find_csv_files_single_file(self, csv_loader, sample_csv):
        """Test finding single CSV file."""
        files = csv_loader.find_csv_files()
        assert len(files) == 1
        assert files[0].name == "test_data.csv"
    
    def test_find_csv_files_multiple_files(self, temp_data_dir):
        """Test finding multiple CSV files."""
        # Create multiple CSV files
        for i in range(3):
            path = Path(temp_data_dir) / f"data_{i}.csv"
            path.write_text(f"id,name\n{i},name_{i}")
        
        config = CSVConfig(data_dir=temp_data_dir)
        loader = CSVDocumentLoader(config)
        files = loader.find_csv_files()
        
        assert len(files) == 3


# ============================================================================
# CSV Loading Tests
# ============================================================================


class TestCSVLoading:
    """Tests for loading CSV files."""
    
    def test_load_csv_safe_valid_file(self, csv_loader, sample_csv):
        """Test loading valid CSV file."""
        df = csv_loader.load_csv_safe(sample_csv)
        
        assert df is not None
        assert len(df) == 3
        assert list(df.columns) == ["id", "title", "category", "description", "tags"]
    
    def test_load_csv_safe_empty_file(self, temp_data_dir, csv_loader):
        """Test loading empty CSV file."""
        empty_csv = Path(temp_data_dir) / "empty.csv"
        empty_csv.write_text("id,name,description\n")
        
        df = csv_loader.load_csv_safe(empty_csv)
        assert df is None
    
    def test_load_csv_safe_nonexistent_file(self, csv_loader):
        """Test loading nonexistent file."""
        result = csv_loader.load_csv_safe(Path("nonexistent.csv"))
        assert result is None
    
    def test_load_csv_safe_with_utf8(self, temp_data_dir, csv_loader):
        """Test loading CSV with UTF-8 encoding."""
        csv_path = Path(temp_data_dir) / "utf8_data.csv"
        content = "id,name,description\n1,Test,Description with émojis 🚀"
        csv_path.write_text(content, encoding='utf-8')
        
        df = csv_loader.load_csv_safe(csv_path)
        assert df is not None
        assert len(df) == 1


# ============================================================================
# Column Normalization Tests
# ============================================================================


class TestColumnNormalization:
    """Tests for normalizing column names."""
    
    def test_normalize_columns_to_lowercase(self, csv_loader):
        """Test normalizing column names to lowercase."""
        df = pd.DataFrame({
            "ID": [1],
            "Title": ["test"],
            "DESCRIPTION": ["desc"],
            "Tags": ["tag1"]
        })
        
        normalized = csv_loader.normalize_columns(df)
        
        assert list(normalized.columns) == ["id", "title", "description", "tags"]
    
    def test_normalize_columns_with_spaces(self, csv_loader):
        """Test normalizing column names with spaces."""
        df = pd.DataFrame({
            "  ID  ": [1],
            "Item Title": ["test"],
            "  Description  ": ["desc"]
        })
        
        normalized = csv_loader.normalize_columns(df)
        
        assert list(normalized.columns) == ["id", "item title", "description"]


# ============================================================================
# Missing Values Handling Tests
# ============================================================================


class TestMissingValuesHandling:
    """Tests for handling missing values."""
    
    def test_handle_missing_values_fills_nan(self, csv_loader):
        """Test that NaN values are filled."""
        df = pd.DataFrame({
            "id": [1, 2, None],
            "name": ["A", None, "C"],
            "description": [None, None, "D"]
        })
        
        handled = csv_loader.handle_missing_values(df)
        
        # Should fill all NaN with empty string
        assert not handled.isnull().any().any()
        assert handled.loc[2, "id"] == ""
        assert handled.loc[1, "name"] == ""
    
    def test_handle_missing_values_preserves_data(self, csv_loader):
        """Test that non-null values are preserved."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"]
        })
        
        handled = csv_loader.handle_missing_values(df)
        
        assert handled.loc[0, "id"] == 1
        assert handled.loc[1, "name"] == "B"


# ============================================================================
# Searchable Text Generation Tests
# ============================================================================


class TestSearchableTextGeneration:
    """Tests for generating searchable text from rows."""
    
    def test_get_searchable_text_prioritizes_fields(self, csv_loader):
        """Test that searchable fields are prioritized."""
        row = pd.Series({
            "title": "Main Title",
            "description": "Main Description",
            "category": "Category",
            "other_field": "Other"
        })
        
        text = csv_loader.get_searchable_text(row, "test.csv")
        
        assert "Title: Main Title" in text
        assert "Description: Main Description" in text
    
    def test_get_searchable_text_empty_row(self, csv_loader):
        """Test with empty row."""
        row = pd.Series({"id": "", "name": "", "desc": ""})
        text = csv_loader.get_searchable_text(row, "test.csv")
        
        assert text is None
    
    def test_get_searchable_text_max_length(self, csv_loader):
        """Test that text is truncated to max length."""
        long_text = "A" * 3000
        row = pd.Series({"description": long_text})
        
        text = csv_loader.get_searchable_text(row, "test.csv")
        
        assert len(text) <= csv_loader.config.max_text_length + 3  # +3 for "..."


# ============================================================================
# Row ID Extraction Tests
# ============================================================================


class TestRowIDExtraction:
    """Tests for extracting row IDs."""
    
    def test_get_row_id_from_id_field(self, csv_loader):
        """Test extracting ID from id field."""
        row = pd.Series({"id": "custom_001", "name": "Test"})
        
        row_id = csv_loader.get_row_id(row, 0, "test.csv")
        
        assert row_id == "custom_001"
    
    def test_get_row_id_fallback_to_generated(self, csv_loader):
        """Test fallback to generated ID."""
        row = pd.Series({"id": "", "name": "Test"})
        
        row_id = csv_loader.get_row_id(row, 5, "test_data.csv")
        
        assert row_id == "test_data_0005"
    
    def test_get_row_id_no_id_field(self, csv_loader):
        """Test ID generation when no id field."""
        row = pd.Series({"name": "Test", "description": "Desc"})
        
        row_id = csv_loader.get_row_id(row, 3, "data.csv")
        
        assert row_id == "data_0003"


# ============================================================================
# Document Creation Tests
# ============================================================================


class TestDocumentCreation:
    """Tests for converting rows to documents."""
    
    def test_row_to_document_valid_row(self, csv_loader):
        """Test converting valid row to document."""
        row = pd.Series({
            "id": "item_001",
            "title": "Test Item",
            "description": "Test Description",
            "category": "Test Category"
        })
        
        doc = csv_loader.row_to_document(row, 0, "test.csv")
        
        assert doc is not None
        assert isinstance(doc, Document)
        assert "Title: Test Item" in doc.content
        assert doc.metadata.source == "test.csv"
        assert doc.metadata.type == "csv"
        assert doc.metadata.id == "item_001"
    
    def test_row_to_document_empty_row(self, csv_loader):
        """Test converting empty row returns None."""
        row = pd.Series({"id": "", "name": ""})
        
        doc = csv_loader.row_to_document(row, 0, "test.csv")
        
        assert doc is None
    
    def test_row_to_document_metadata(self, csv_loader):
        """Test document metadata creation."""
        row = pd.Series({"id": "doc_001", "title": "Test"})
        
        doc = csv_loader.row_to_document(row, 5, "source.csv")
        
        assert doc.metadata.id == "doc_001"
        assert doc.metadata.type == "csv"
        assert doc.metadata.source == "source.csv"
        assert doc.metadata.extra["row_index"] == 5
        assert doc.metadata.extra["csv_type"] == "source"


# ============================================================================
# CSV File Processing Tests
# ============================================================================


class TestCSVFileProcessing:
    """Tests for loading complete CSV files."""
    
    def test_load_csv_file(self, csv_loader, sample_csv):
        """Test loading complete CSV file."""
        docs = csv_loader.load_csv_file(sample_csv)
        
        assert len(docs) == 3
        assert all(isinstance(doc, Document) for doc in docs)
        assert all(doc.metadata.type == "csv" for doc in docs)
    
    def test_load_csv_file_creates_documents(self, temp_data_dir):
        """Test that documents are created correctly."""
        # Create structured CSV
        csv_path = Path(temp_data_dir) / "items.csv"
        data = {
            "id": ["item_1", "item_2"],
            "title": ["First", "Second"],
            "description": ["Desc 1", "Desc 2"]
        }
        pd.DataFrame(data).to_csv(csv_path, index=False)
        
        config = CSVConfig(data_dir=temp_data_dir)
        loader = CSVDocumentLoader(config)
        docs = loader.load_csv_file(csv_path)
        
        assert len(docs) == 2
        assert "First" in docs[0].content
        assert "Second" in docs[1].content


# ============================================================================
# Multiple CSV Loading Tests
# ============================================================================


class TestMultipleCSVLoading:
    """Tests for loading multiple CSV files."""
    
    def test_load_all_csvs_single_file(self, csv_loader, sample_csv):
        """Test loading all CSVs when only one exists."""
        docs = csv_loader.load_all_csvs()
        
        assert len(docs) == 3
    
    def test_load_all_csvs_multiple_files(self, temp_data_dir):
        """Test loading multiple CSV files."""
        # Create two CSV files
        for i in range(2):
            csv_path = Path(temp_data_dir) / f"data_{i}.csv"
            data = {
                "id": [f"item_{i}_1", f"item_{i}_2"],
                "title": ["Item 1", "Item 2"],
                "description": ["Desc 1", "Desc 2"]
            }
            pd.DataFrame(data).to_csv(csv_path, index=False)
        
        config = CSVConfig(data_dir=temp_data_dir)
        loader = CSVDocumentLoader(config)
        docs = loader.load_all_csvs()
        
        # Should have 2 files * 2 rows each = 4 documents
        assert len(docs) == 4
    
    def test_load_all_csvs_empty_directory(self, csv_loader):
        """Test loading from empty directory."""
        docs = csv_loader.load_all_csvs()
        
        assert docs == []


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_load_csv_documents_function(self, temp_data_dir, sample_csv):
        """Test load_csv_documents convenience function."""
        docs = load_csv_documents(temp_data_dir)
        
        assert len(docs) == 3
        assert all(isinstance(doc, Document) for doc in docs)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
