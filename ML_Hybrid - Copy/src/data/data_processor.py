"""
Data processing module for the ML Hybrid Theme Analysis system.

This module provides functionality to process CSV files, extract summaries,
and manage data throughout the analysis pipeline.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import csv
from loguru import logger

from ..utils.config import config
from ..utils.logger import get_logger

# Initialize logger
log = get_logger(__name__)


class DataProcessor:
    """
    Processes CSV files and manages data throughout the analysis pipeline.

    This class provides functionality to load CSV files, detect summary columns,
    extract summaries, and manage data transformations.
    """

    def __init__(self):
        """Initialize the data processor."""
        self.supported_formats = [".csv", ".xlsx", ".xls"]
        self.summary_column_candidates = [
            "summary",
            "description",
            "issue",
            "problem",
            "title",
            "text",
            "content",
            "message",
            "details",
            "comment",
            "note",
        ]

        log.info("DataProcessor initialized successfully")

    def detect_summary_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Automatically detect the summary column in a DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Name of the summary column or None if not found
        """
        try:
            # Check for exact matches first
            for candidate in self.summary_column_candidates:
                if candidate in df.columns:
                    log.info(f"Found summary column: {candidate}")
                    return candidate

            # Check for partial matches
            for col in df.columns:
                col_lower = col.lower()
                for candidate in self.summary_column_candidates:
                    if candidate in col_lower or col_lower in candidate:
                        log.info(f"Found summary column (partial match): {col}")
                        return col

            # If no match found, try to infer from data characteristics
            text_columns = []
            for col in df.columns:
                if df[col].dtype == "object":
                    # Check if column contains text data
                    sample_values = df[col].dropna().head(10)
                    if len(sample_values) > 0:
                        avg_length = sample_values.str.len().mean()
                        if avg_length > 20:  # Likely to be summary text
                            text_columns.append((col, avg_length))

            if text_columns:
                # Return the column with the longest average text
                best_column = max(text_columns, key=lambda x: x[1])[0]
                log.info(f"Found summary column (inferred): {best_column}")
                return best_column

            log.warning("No summary column detected")
            return None

        except Exception as e:
            log.error(f"Error detecting summary column: {e}")
            return None

    def load_csv_file(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load a CSV file and extract metadata.

        Args:
            file_path: Path to the CSV file

        Returns:
            Tuple of (DataFrame, metadata)
        """
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if file_path.suffix.lower() not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")

            # Load the file
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            # Extract metadata
            metadata = {
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "column_names": list(df.columns),
                "missing_values": df.isnull().sum().to_dict(),
                "data_types": df.dtypes.to_dict(),
            }

            # Detect summary column
            summary_column = self.detect_summary_column(df)
            metadata["summary_column"] = summary_column

            if summary_column:
                metadata["summary_count"] = df[summary_column].notna().sum()
                metadata["empty_summaries"] = df[summary_column].isna().sum()
            else:
                metadata["summary_count"] = 0
                metadata["empty_summaries"] = len(df)

            log.info(f"Loaded file: {file_path.name} with {len(df)} rows")
            log.info(f"Summary column: {summary_column}")

            return df, metadata

        except Exception as e:
            log.error(f"Error loading file {file_path}: {e}")
            raise

    def extract_summaries(self, df: pd.DataFrame, summary_column: str) -> List[str]:
        """
        Extract summaries from the DataFrame.

        Args:
            df: Input DataFrame
            summary_column: Name of the summary column

        Returns:
            List of summary texts
        """
        try:
            if summary_column not in df.columns:
                raise ValueError(
                    f"Summary column '{summary_column}' not found in DataFrame"
                )

            # Extract summaries and convert to string
            summaries = df[summary_column].astype(str).tolist()

            # Remove empty or null summaries
            summaries = [
                summary
                for summary in summaries
                if summary and summary.lower() not in ["nan", "none", ""]
            ]

            log.info(
                f"Extracted {len(summaries)} summaries from column '{summary_column}'"
            )
            return summaries

        except Exception as e:
            log.error(f"Error extracting summaries: {e}")
            return []

    def validate_data(
        self, df: pd.DataFrame, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate the loaded data and provide quality metrics.

        Args:
            df: Input DataFrame
            metadata: Data metadata

        Returns:
            Dictionary containing validation results
        """
        try:
            validation_results = {
                "is_valid": True,
                "warnings": [],
                "errors": [],
                "quality_metrics": {},
            }

            # Check file size
            if metadata["file_size"] > 100 * 1024 * 1024:  # 100MB
                validation_results["warnings"].append("File size is large (>100MB)")

            # Check row count
            if metadata["total_rows"] == 0:
                validation_results["errors"].append("File contains no data rows")
                validation_results["is_valid"] = False

            if metadata["total_rows"] > 100000:
                validation_results["warnings"].append("File contains many rows (>100k)")

            # Check summary column
            if not metadata["summary_column"]:
                validation_results["errors"].append("No summary column detected")
                validation_results["is_valid"] = False

            # Check missing values
            missing_ratio = (
                metadata["empty_summaries"] / metadata["total_rows"]
                if metadata["total_rows"] > 0
                else 0
            )
            if missing_ratio > 0.5:
                validation_results["warnings"].append(
                    f"High missing data ratio: {missing_ratio:.1%}"
                )

            # Calculate quality metrics
            validation_results["quality_metrics"] = {
                "completeness": 1 - missing_ratio,
                "row_count": metadata["total_rows"],
                "summary_count": metadata["summary_count"],
                "column_count": metadata["total_columns"],
            }

            log.info(
                f"Data validation completed. Valid: {validation_results['is_valid']}"
            )
            return validation_results

        except Exception as e:
            log.error(f"Error validating data: {e}")
            return {
                "is_valid": False,
                "warnings": [],
                "errors": [f"Validation error: {e}"],
                "quality_metrics": {},
            }

    def create_analysis_dataset(
        self, df: pd.DataFrame, processed_summaries: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Create a dataset for analysis with processed summaries and metadata.

        Args:
            df: Original DataFrame
            processed_summaries: List of processed summary dictionaries

        Returns:
            DataFrame with analysis data
        """
        try:
            # Create analysis DataFrame
            analysis_data = []

            for i, processed_summary in enumerate(processed_summaries):
                row_data = {
                    "original_index": i,
                    "original_summary": processed_summary["original"],
                    "cleaned_summary": processed_summary["cleaned"],
                    "processed_summary": processed_summary["processed"],
                    "language": processed_summary["language"],
                    "summary_length": processed_summary["length"],
                }

                # Add original DataFrame columns if available
                if i < len(df):
                    for col in df.columns:
                        row_data[f"original_{col}"] = (
                            df.iloc[i][col] if i < len(df) else None
                        )

                analysis_data.append(row_data)

            analysis_df = pd.DataFrame(analysis_data)

            log.info(f"Created analysis dataset with {len(analysis_df)} rows")
            return analysis_df

        except Exception as e:
            log.error(f"Error creating analysis dataset: {e}")
            return pd.DataFrame()

    def save_analysis_results(self, results: Dict[str, Any], output_path: str) -> bool:
        """
        Save analysis results to files.

        Args:
            results: Analysis results dictionary
            output_path: Output directory path

        Returns:
            True if successful
        """
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save processed data
            if "processed_data" in results:
                processed_df = pd.DataFrame(results["processed_data"])
                processed_df.to_csv(output_dir / "processed_summaries.csv", index=False)

            # Save clustering results
            if "clustering_results" in results:
                clustering_data = results["clustering_results"]

                # Save 2D embeddings
                if "embeddings_2d" in clustering_data:
                    embeddings_df = pd.DataFrame(
                        clustering_data["embeddings_2d"], columns=["x", "y"]
                    )
                    embeddings_df["cluster"] = clustering_data["cluster_labels"]
                    embeddings_df.to_csv(
                        output_dir / "clustering_results.csv", index=False
                    )

            # Save theme analysis
            if "theme_analysis" in results:
                theme_data = []
                for cluster_id, analysis in results["theme_analysis"].items():
                    theme_data.append(
                        {
                            "cluster_id": cluster_id,
                            "theme_name": analysis["name"],
                            "description": analysis["description"],
                            "sample_count": analysis["sample_count"],
                            "data_quality_issues": analysis["data_quality_issues"],
                            "data_quality_percentage": analysis[
                                "data_quality_percentage"
                            ],
                        }
                    )

                theme_df = pd.DataFrame(theme_data)
                theme_df.to_csv(output_dir / "theme_analysis.csv", index=False)

            # Save summary report
            if "summary_report" in results:
                with open(output_dir / "analysis_report.txt", "w") as f:
                    f.write(results["summary_report"])

            log.info(f"Analysis results saved to {output_path}")
            return True

        except Exception as e:
            log.error(f"Error saving analysis results: {e}")
            return False

    def get_data_summary(
        self, metadata: Dict[str, Any], validation_results: Dict[str, Any]
    ) -> str:
        """
        Generate a human-readable data summary.

        Args:
            metadata: Data metadata
            validation_results: Validation results

        Returns:
            Formatted summary string
        """
        summary = f"""
Data Summary
===========

File Information:
- File Name: {metadata['file_name']}
- File Size: {metadata['file_size'] / 1024:.1f} KB
- Total Rows: {metadata['total_rows']:,}
- Total Columns: {metadata['total_columns']}

Summary Column:
- Column Name: {metadata['summary_column'] or 'Not detected'}
- Summary Count: {metadata['summary_count']:,}
- Empty Summaries: {metadata['empty_summaries']:,}

Data Quality:
- Completeness: {validation_results['quality_metrics'].get('completeness', 0):.1%}
- Valid: {validation_results['is_valid']}

Warnings: {len(validation_results['warnings'])}
Errors: {len(validation_results['errors'])}
"""

        if validation_results["warnings"]:
            summary += "\nWarnings:\n"
            for warning in validation_results["warnings"]:
                summary += f"- {warning}\n"

        if validation_results["errors"]:
            summary += "\nErrors:\n"
            for error in validation_results["errors"]:
                summary += f"- {error}\n"

        return summary
