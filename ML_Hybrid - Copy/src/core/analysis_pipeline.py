"""
Main analysis pipeline for ML Hybrid Theme Analysis.

This module orchestrates the complete end-to-end analysis pipeline including
data loading, preprocessing, embedding generation, clustering, and theme analysis.
"""

import os
import pandas as pd
from typing import Dict, Any, List, Optional
from loguru import logger

from ..utils.config import ConfigManager
from ..preprocessing.text_processor import TextProcessor
from ..clustering.theme_clusterer import ThemeClusterer
from ..llm.theme_analyzer import ThemeAnalyzer
from ..utils.logger import setup_logging

# Setup logging
setup_logging()
log = logger


class AnalysisPipeline:
    """
    Main orchestration class for the ML Hybrid Theme Analysis pipeline.

    This class coordinates all components of the analysis pipeline including
    data loading, preprocessing, embedding generation, clustering, and
    theme analysis using AWS Bedrock services.
    """

    def __init__(self):
        """Initialize the analysis pipeline with all components."""
        try:
            # Load configuration
            self.config_manager = ConfigManager()
            self.config = self.config_manager._config

            # Initialize components
            self.text_processor = TextProcessor(self.config)
            self.theme_clusterer = ThemeClusterer()
            self.theme_analyzer = ThemeAnalyzer()

            log.info("AnalysisPipeline initialized successfully")

        except Exception as e:
            log.error(f"Error initializing AnalysisPipeline: {str(e)}")
            raise

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and validate CSV data from file.

        Args:
            file_path: Path to the CSV file

        Returns:
            Loaded DataFrame with validated data
        """
        try:
            log.info(f"Loading data from: {file_path}")

            # Load CSV file
            df = pd.read_csv(file_path)

            # Validate data
            if df.empty:
                raise ValueError("CSV file is empty")

            # Detect summary column
            summary_column = self._detect_summary_column(df)
            if not summary_column:
                raise ValueError("No suitable summary column found in CSV")

            log.info(f"Data loaded successfully. Shape: {df.shape}")
            log.info(f"Summary column detected: {summary_column}")

            return df, summary_column

        except Exception as e:
            log.error(f"Error loading data: {str(e)}")
            raise

    def _detect_summary_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Automatically detect the summary column in the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Name of the summary column or None if not found
        """
        # Common column names for summaries
        summary_keywords = [
            "summary",
            "description",
            "text",
            "content",
            "message",
            "issue",
            "problem",
            "bug",
            "comment",
            "note",
            "details",
        ]

        # Check for exact matches first
        for col in df.columns:
            if col.lower() in summary_keywords:
                return col

        # Check for partial matches
        for col in df.columns:
            col_lower = col.lower()
            for keyword in summary_keywords:
                if keyword in col_lower:
                    return col

        # If no match found, return the first text-like column
        for col in df.columns:
            if df[col].dtype == "object":
                # Check if column contains text data
                sample_values = df[col].dropna().head(10)
                if len(sample_values) > 0:
                    avg_length = sample_values.str.len().mean()
                    if avg_length > 20:  # Assume it's text if average length > 20
                        return col

        return None

    def preprocess_data(self, df: pd.DataFrame, summary_column: str) -> Dict[str, Any]:
        """
        Preprocess the text data for analysis.

        Args:
            df: Input DataFrame
            summary_column: Name of the summary column

        Returns:
            Dictionary containing preprocessing results and statistics
        """
        try:
            log.info("Starting text preprocessing")

            # Extract summaries
            summaries = df[summary_column].fillna("").astype(str).tolist()

            # Get text statistics
            text_stats = self.text_processor.get_text_statistics(summaries)

            log.info(
                f"Preprocessing completed. Processed {text_stats['total_texts']} texts"
            )

            return {
                "processed_texts": text_stats["processed_texts"],
                "statistics": text_stats,
                "summaries": summaries,
            }

        except Exception as e:
            log.error(f"Error in preprocessing: {str(e)}")
            raise

    def generate_embeddings(
        self, processed_texts: List[Dict[str, Any]]
    ) -> List[List[float]]:
        """
        Generate embeddings for processed texts using AWS Bedrock.

        Args:
            processed_texts: List of processed text dictionaries

        Returns:
            List of embedding vectors
        """
        try:
            log.info("Generating embeddings using AWS Bedrock")

            # Extract cleaned text for embedding
            texts_for_embedding = [text["cleaned_text"] for text in processed_texts]

            # Generate embeddings using the theme clusterer
            embeddings = self.theme_clusterer.generate_embeddings(texts_for_embedding)

            log.info(f"Generated {len(embeddings)} embeddings")

            return embeddings

        except Exception as e:
            log.error(f"Error generating embeddings: {str(e)}")
            raise

    def perform_clustering(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Perform theme clustering on embeddings.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Dictionary containing clustering results
        """
        try:
            log.info("Performing theme clustering")

            # Perform clustering
            clustering_results = self.theme_clusterer.fit_clusters(embeddings)

            log.info(
                f"Clustering completed. Found {clustering_results['n_clusters']} clusters"
            )

            return clustering_results

        except Exception as e:
            log.error(f"Error in clustering: {str(e)}")
            raise

    def analyze_themes(
        self, clustering_results: Dict[str, Any], processed_texts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze themes using LLM-based analysis.

        Args:
            clustering_results: Results from clustering
            processed_texts: List of processed text dictionaries

        Returns:
            Dictionary containing theme analysis results
        """
        try:
            log.info("Starting theme analysis with LLM")

            # Extract cluster samples for analysis
            cluster_samples = clustering_results.get("cluster_samples", {})

            # Analyze themes
            theme_analysis = self.theme_analyzer.analyze_themes(cluster_samples)

            log.info("Theme analysis completed successfully")

            return theme_analysis

        except Exception as e:
            log.error(f"Error in theme analysis: {str(e)}")
            raise

    def run_analysis(
        self, file_path: str, summary_column: str = None
    ) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.

        Args:
            file_path: Path to the CSV file
            summary_column: Name of the column containing summary text (optional)

        Returns:
            Dictionary containing analysis results
        """
        try:
            log.info(f"Starting analysis for file: {file_path}")

            # Load and validate data
            log.info("Loading and validating data...")
            if summary_column:
                log.info(f"Using specified summary column: {summary_column}")
                df, detected_column = self._load_data_with_column(
                    file_path, summary_column
                )
            else:
                log.info("Auto-detecting summary column...")
                df, detected_column = self.load_data(file_path)

            summary_column = detected_column  # Use the detected/validated column

            # Preprocess data
            log.info("Preprocessing data...")
            preprocessing_results = self.preprocess_data(df, summary_column)
            processed_texts = preprocessing_results["processed_texts"]

            if not processed_texts:
                raise ValueError("No valid data after preprocessing")

            # Generate embeddings
            log.info("Generating embeddings...")
            embeddings = self.generate_embeddings(processed_texts)

            # Optimize hyperparameters if enabled
            if self.config.get("hyperparameter_tuning", {}).get("enabled", False):
                log.info("Optimizing hyperparameters...")
                self.theme_clusterer.optimize_hyperparameters(
                    embeddings, processed_texts
                )

            # Perform clustering with persistence
            log.info("Performing clustering...")
            clustering_results = self.theme_clusterer.fit_clusters_with_persistence(
                embeddings, processed_texts
            )

            # Analyze themes
            log.info("Analyzing themes...")
            theme_analysis = self.theme_analyzer.analyze_cluster_themes(
                clustering_results["cluster_samples"]
            )

            # Prepare results
            results = {
                "data_info": {
                    "total_records": len(processed_texts),
                    "summary_column": summary_column,
                    "file_path": file_path,
                    "preprocessing_stats": preprocessing_results.get("stats", {}),
                },
                "clustering_results": clustering_results,
                "theme_analysis": theme_analysis,
                "data_quality_analysis": self._analyze_data_quality(processed_texts),
                "analysis_id": f"analysis_{hash(file_path)}_{hash(summary_column)}",
            }

            log.info("Analysis completed successfully")
            return results

        except Exception as e:
            log.error(f"Analysis failed: {e}")
            raise

    def _load_data_with_column(self, file_path: str, summary_column: str) -> tuple:
        """
        Load data with a specified summary column.

        Args:
            file_path: Path to the CSV file
            summary_column: Name of the column containing summary text

        Returns:
            Tuple of (DataFrame, validated_column_name)
        """
        try:
            log.info(f"Loading data from: {file_path}")

            # Load CSV file
            df = pd.read_csv(file_path)

            # Validate data
            if df.empty:
                raise ValueError("CSV file is empty")

            # Validate the specified column exists
            if summary_column not in df.columns:
                available_columns = list(df.columns)
                raise ValueError(
                    f"Summary column '{summary_column}' not found. Available columns: {available_columns}"
                )

            # Check if the column has data
            if df[summary_column].isna().all():
                raise ValueError(f"Summary column '{summary_column}' contains no data")

            log.info(f"Data loaded successfully. Shape: {df.shape}")
            log.info(f"Using summary column: {summary_column}")

            return df, summary_column

        except Exception as e:
            log.error(f"Error loading data: {str(e)}")
            raise

    def _analyze_data_quality(
        self, processed_texts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze data quality of processed texts.

        Args:
            processed_texts: List of processed text dictionaries

        Returns:
            Dictionary containing data quality analysis results
        """
        try:
            log.info("Analyzing data quality...")

            total_texts = len(processed_texts)
            if total_texts == 0:
                return {
                    "data_quality_count": 0,
                    "data_quality_percentage": 0,
                    "total_texts": 0,
                }

            # Simple quality metrics
            quality_issues = 0
            for text_data in processed_texts:
                text = text_data.get("processed_text", "")
                if len(text.strip()) < 10:  # Very short texts
                    quality_issues += 1
                elif text.lower().count("error") > 0 or text.lower().count("null") > 0:
                    quality_issues += 1

            quality_percentage = (
                (quality_issues / total_texts) * 100 if total_texts > 0 else 0
            )

            return {
                "data_quality_count": quality_issues,
                "data_quality_percentage": quality_percentage,
                "total_texts": total_texts,
            }

        except Exception as e:
            log.error(f"Error in data quality analysis: {str(e)}")
            return {
                "data_quality_count": 0,
                "data_quality_percentage": 0,
                "total_texts": len(processed_texts),
            }

    def optimize_hyperparameters(self, file_path: str) -> Dict[str, Any]:
        """
        Optimize clustering hyperparameters.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary containing optimization results
        """
        try:
            log.info("Starting hyperparameter optimization")

            # Load and preprocess data
            df, summary_column = self.load_data(file_path)
            preprocessing_results = self.preprocess_data(df, summary_column)
            embeddings = self.generate_embeddings(
                preprocessing_results["processed_texts"]
            )

            # Optimize hyperparameters
            optimized_params = self.theme_clusterer.optimize_hyperparameters(embeddings)

            log.info("Hyperparameter optimization completed")

            return optimized_params

        except Exception as e:
            log.error(f"Error in hyperparameter optimization: {str(e)}")
            raise

    def predict_clusters(self, file_path: str) -> Dict[str, Any]:
        """
        Predict clusters for new data using saved models.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary containing prediction results
        """
        try:
            log.info("Predicting clusters for new data")

            # Load and preprocess data
            df, summary_column = self.load_data(file_path)
            preprocessing_results = self.preprocess_data(df, summary_column)
            embeddings = self.generate_embeddings(
                preprocessing_results["processed_texts"]
            )

            # Predict clusters
            predictions = self.theme_clusterer.predict_clusters_for_new_data(embeddings)

            log.info("Cluster prediction completed")

            return predictions

        except Exception as e:
            log.error(f"Error in cluster prediction: {str(e)}")
            raise

    def check_drift(self, file_path: str) -> Dict[str, Any]:
        """
        Check for data drift using saved models.

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary containing drift analysis results
        """
        try:
            log.info("Checking for data drift")

            # Load and preprocess data
            df, summary_column = self.load_data(file_path)
            preprocessing_results = self.preprocess_data(df, summary_column)
            embeddings = self.generate_embeddings(
                preprocessing_results["processed_texts"]
            )

            # Check for drift
            drift_info = self.theme_clusterer.check_model_drift(embeddings)

            log.info("Data drift check completed")

            return drift_info

        except Exception as e:
            log.error(f"Error in drift check: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about saved models.

        Returns:
            Dictionary containing model information
        """
        try:
            log.info("Getting model information")

            model_info = self.theme_clusterer.get_model_info()

            return model_info

        except Exception as e:
            log.error(f"Error getting model info: {str(e)}")
            raise
