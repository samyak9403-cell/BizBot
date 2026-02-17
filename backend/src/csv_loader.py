"""CSV document loader for knowledge base indexing.

Handles loading, parsing, and normalizing CSV files into structured documents
with metadata. Supports safe pandas loading with error handling and graceful
handling of missing values.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from src.document_processor import Document, DocumentMetadata


logger = logging.getLogger(__name__)


@dataclass
class CSVConfig:
    """Configuration for CSV loading."""
    
    data_dir: str = "data"
    """Directory containing CSV files"""
    
    searchable_fields: List[str] = field(
        default_factory=lambda: ["title", "name", "tags", "category", "description", "content", "text"]
    )
    """Column names to prioritize in text generation"""
    
    id_field: str = "id"
    """Column name for unique identifier"""
    
    max_text_length: int = 2000
    """Maximum text length for a document"""
    
    allow_nan: bool = True
    """Allow NaN values (filled with empty string)"""


class CSVDocumentLoader:
    """Load and parse CSV files into documents for knowledge base indexing.
    
    Features:
    - Automatically scans for .csv files
    - Normalizes column names to lowercase
    - Handles missing values gracefully
    - Converts rows to structured natural language text
    - Prioritizes searchable fields
    - Skips invalid/empty rows
    - Attaches rich metadata
    """
    
    def __init__(self, config: Optional[CSVConfig] = None):
        """Initialize CSV loader.
        
        Args:
            config: CSVConfig instance or None for defaults
        """
        self.config = config or CSVConfig()
        self.data_dir = Path(self.config.data_dir)
        
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created data directory: {self.data_dir}")
    
    def find_csv_files(self) -> List[Path]:
        """Find all CSV files in data directory.
        
        Returns:
            List of Path objects for each CSV file found
        """
        csv_files = list(self.data_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV file(s)")
        return sorted(csv_files)
    
    def load_csv_safe(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Load CSV file safely with error handling.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame if successful, None if failed
        """
        try:
            # Try to read CSV with various settings
            df = pd.read_csv(
                filepath,
                encoding='utf-8',
                on_bad_lines='skip',  # Skip malformed lines
                dtype_backend='numpy_nullable'  # Better handling of nulls
            )
            
            if df.empty:
                logger.warning(f"CSV file is empty: {filepath}")
                return None
            
            logger.info(f"Loaded CSV: {filepath.name} ({len(df)} rows, {len(df.columns)} columns)")
            return df
            
        except UnicodeDecodeError:
            # Try alternate encoding
            try:
                df = pd.read_csv(filepath, encoding='latin-1', on_bad_lines='skip')
                logger.info(f"Loaded CSV with latin-1 encoding: {filepath.name}")
                return df
            except Exception as e:
                logger.error(f"Failed to load {filepath.name} (encoding error): {e}")
                return None
        except Exception as e:
            logger.error(f"Failed to load {filepath.name}: {e}")
            return None
    
    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase.
        
        Args:
            df: DataFrame to normalize
            
        Returns:
            DataFrame with normalized column names
        """
        df.columns = df.columns.str.lower().str.strip()
        logger.debug(f"Normalized columns: {list(df.columns)}")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values gracefully.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        # Fill NaN values with empty string
        df = df.fillna("")
        
        logger.debug(f"Handled missing values in {len(df)} rows")
        return df
    
    def get_searchable_text(self, row: pd.Series, filename: str) -> Optional[str]:
        """Convert row to structured natural language text.
        
        Prioritizes searchable fields (title, tags, category, description)
        and creates readable text representation.
        
        Args:
            row: Pandas Series (CSV row)
            filename: Source filename for context
            
        Returns:
            Structured text if row is valid, None if empty/invalid
        """
        # Collect searchable content
        content_parts = []
        
        # Add field-value pairs in priority order
        for field_name in self.config.searchable_fields:
            if field_name in row.index:
                value = str(row[field_name]).strip()
                if value and value.lower() != "nan":
                    # Format as "Field: value"
                    formatted_field = field_name.replace("_", " ").title()
                    content_parts.append(f"{formatted_field}: {value}")
        
        # If no searchable fields found, use all non-empty values
        if not content_parts:
            for col, value in row.items():
                value_str = str(value).strip()
                if value_str and value_str.lower() != "nan":
                    formatted_col = col.replace("_", " ").title()
                    content_parts.append(f"{formatted_col}: {value_str}")
        
        # If still empty, skip this row
        if not content_parts:
            logger.debug(f"Skipping empty row")
            return None
        
        # Combine all parts with newlines
        text = "\n".join(content_parts)
        
        # Enforce max length
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length] + "..."
        
        return text
    
    def get_row_id(self, row: pd.Series, row_index: int, filename: str) -> str:
        """Extract or generate unique ID for row.
        
        Args:
            row: Pandas Series (CSV row)
            row_index: Row index in CSV
            filename: Source filename
            
        Returns:
            Unique identifier string
        """
        # Try to get ID from configured id_field
        if self.config.id_field in row.index:
            row_id = str(row[self.config.id_field]).strip()
            if row_id and row_id.lower() != "nan":
                return row_id
        
        # Fallback: generate from filename and row index
        filename_base = filename.replace(".csv", "")
        return f"{filename_base}_{row_index:04d}"
    
    def row_to_document(
        self,
        row: pd.Series,
        row_index: int,
        csv_filename: str
    ) -> Optional[Document]:
        """Convert CSV row to Document object.
        
        Args:
            row: Pandas Series (CSV row)
            row_index: Row number in CSV
            csv_filename: Source CSV filename
            
        Returns:
            Document object or None if row is invalid
        """
        # Get searchable text
        text = self.get_searchable_text(row, csv_filename)
        if not text:
            return None
        
        # Get unique ID
        doc_id = self.get_row_id(row, row_index, csv_filename)
        
        # Create metadata
        metadata = DocumentMetadata(
            source=csv_filename,
            type="csv",  # Mark as CSV type
            id=doc_id,
            extra={
                "row_index": row_index,
                "filename": csv_filename,
                "csv_type": csv_filename.replace(".csv", "")
            }
        )
        
        # Create document
        document = Document(
            content=text,
            metadata=metadata
        )
        
        return document
    
    def load_csv_file(self, filepath: Path) -> List[Document]:
        """Load single CSV file and convert to documents.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            List of Document objects
        """
        logger.info(f"Loading CSV: {filepath.name}")
        
        # Load CSV
        df = self.load_csv_safe(filepath)
        if df is None:
            return []
        
        # Normalize and clean
        df = self.normalize_columns(df)
        df = self.handle_missing_values(df)
        
        # Convert rows to documents
        documents = []
        filename = filepath.name
        
        for idx, (_, row) in enumerate(df.iterrows()):
            try:
                doc = self.row_to_document(row, idx, filename)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.warning(f"Skipped row {idx} in {filename}: {e}")
                continue
        
        logger.info(f"Created {len(documents)} documents from {filename}")
        return documents
    
    def load_all_csvs(self) -> List[Document]:
        """Load all CSV files from data directory.
        
        Returns:
            Combined list of all Document objects from all CSVs
        """
        csv_files = self.find_csv_files()
        
        if not csv_files:
            logger.warning("No CSV files found in data directory")
            return []
        
        all_documents = []
        
        for filepath in csv_files:
            try:
                docs = self.load_csv_file(filepath)
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"Error loading {filepath.name}: {e}", exc_info=True)
                continue
        
        logger.info(f"Loaded {len(all_documents)} total documents from {len(csv_files)} CSV files")
        return all_documents


def load_csv_documents(data_dir: str = "data") -> List[Document]:
    """Convenience function to load all CSV documents.
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        List of Document objects
    """
    config = CSVConfig(data_dir=data_dir)
    loader = CSVDocumentLoader(config)
    return loader.load_all_csvs()
