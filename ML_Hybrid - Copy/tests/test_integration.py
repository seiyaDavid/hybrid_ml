"""
Integration tests for the complete ML Hybrid Theme Analysis system.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

# Import system components
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.analysis_pipeline import AnalysisPipeline
from preprocessing.text_processor import TextProcessor
from embeddings.embedding_manager import EmbeddingManager
from clustering.theme_clusterer import ThemeClusterer
from llm.theme_analyzer import ThemeAnalyzer
from data.data_processor import DataProcessor


class TestCompleteWorkflow:
    """Integration tests for the complete analysis workflow."""

    def test_end_to_end_analysis(self, sample_csv_data, temp_dir, mock_config):
        """Test complete end-to-end analysis workflow."""
        # Create a temporary CSV file
        csv_file_path = os.path.join(temp_dir, "test_data.csv")
        with open(csv_file_path, "w") as f:
            f.write(sample_csv_data)

        # Mock AWS services
        with patch("embeddings.embedding_manager.boto3.client") as mock_bedrock:
            mock_bedrock.return_value.invoke_model.return_value = {
                "body": Mock(
                    read=lambda: json.dumps({"embedding": [0.1] * 768}).encode()
                )
            }

            with patch("llm.theme_analyzer.boto3.client") as mock_bedrock_llm:
                mock_bedrock_llm.return_value.invoke_model.return_value = {
                    "body": Mock(
                        read=lambda: json.dumps(
                            {"completion": "Test theme name"}
                        ).encode()
                    )
                }

                # Mock configuration
                with patch(
                    "core.analysis_pipeline.config.get_aws_config",
                    return_value=mock_config["aws"],
                ):
                    with patch(
                        "core.analysis_pipeline.config.get_preprocessing_config",
                        return_value=mock_config["preprocessing"],
                    ):
                        with patch(
                            "core.analysis_pipeline.config.get_clustering_config",
                            return_value=mock_config["clustering"],
                        ):

                            # Initialize pipeline
                            pipeline = AnalysisPipeline()

                            # Run analysis
                            results = pipeline.run_analysis(csv_file_path)

                            # Check results structure
                            assert results["success"] == True
                            assert "file_path" in results
                            assert "metadata" in results
                            assert "validation_results" in results
                            assert "preprocessing_results" in results
                            assert "clustering_results" in results
                            assert "theme_analysis" in results
                            assert "reports" in results
                            assert "execution_time" in results

    def test_data_processing_integration(self, sample_csv_data, temp_dir, mock_config):
        """Test data processing integration."""
        # Create a temporary CSV file
        csv_file_path = os.path.join(temp_dir, "test_data.csv")
        with open(csv_file_path, "w") as f:
            f.write(sample_csv_data)

        # Mock configuration
        with patch(
            "data.data_processor.config.get_preprocessing_config",
            return_value=mock_config["preprocessing"],
        ):
            processor = DataProcessor()

            # Load and validate data
            metadata = processor.load_csv_file(csv_file_path)
            validation_results = processor.validate_data(metadata)
            summaries = processor.extract_summaries(metadata)

            # Check results
            assert metadata is not None
            assert validation_results["valid"] == True
            assert len(summaries) > 0

    def test_preprocessing_integration(self, sample_texts, mock_config):
        """Test preprocessing integration."""
        with patch(
            "preprocessing.text_processor.config.get_preprocessing_config",
            return_value=mock_config["preprocessing"],
        ):
            processor = TextProcessor()

            # Process summaries
            results = processor.process_summaries(sample_texts)

            # Check results
            assert len(results) == len(sample_texts)
            for result in results:
                assert "original" in result
                assert "processed" in result
                assert "valid" in result
                assert "language" in result

    def test_embedding_generation_integration(self, sample_texts, mock_config):
        """Test embedding generation integration."""
        with patch("embeddings.embedding_manager.boto3.client") as mock_bedrock:
            mock_bedrock.return_value.invoke_model.return_value = {
                "body": Mock(
                    read=lambda: json.dumps({"embedding": [0.1] * 768}).encode()
                )
            }

            with patch(
                "embeddings.embedding_manager.config.get_aws_config",
                return_value=mock_config["aws"],
            ):
                manager = EmbeddingManager()

                # Generate embeddings
                embeddings = manager.generate_embeddings(sample_texts)

                # Check results
                assert len(embeddings) == len(sample_texts)
                assert len(embeddings[0]) == 768

    def test_clustering_integration(self, sample_embeddings, mock_config):
        """Test clustering integration."""
        with patch(
            "clustering.theme_clusterer.config.get_clustering_config",
            return_value=mock_config["clustering"],
        ):
            clusterer = ThemeClusterer()

            # Perform clustering
            results = clusterer.fit_clusters(sample_embeddings)

            # Check results
            assert "embeddings_2d" in results
            assert "cluster_labels" in results
            assert "cluster_centroids" in results
            assert "statistics" in results

    def test_theme_analysis_integration(self, sample_texts, mock_config):
        """Test theme analysis integration."""
        with patch("llm.theme_analyzer.boto3.client") as mock_bedrock:
            mock_bedrock.return_value.invoke_model.return_value = {
                "body": Mock(
                    read=lambda: json.dumps({"completion": "Test theme name"}).encode()
                )
            }

            with patch(
                "llm.theme_analyzer.config.get_aws_config",
                return_value=mock_config["aws"],
            ):
                analyzer = ThemeAnalyzer()

                # Analyze themes
                cluster_samples = {0: sample_texts[:5], 1: sample_texts[5:10]}
                results = analyzer.analyze_cluster_themes(cluster_samples)

                # Check results
                assert isinstance(results, dict)
                for cluster_id, analysis in results.items():
                    assert "theme_name" in analysis
                    assert "description" in analysis
                    assert "sample_count" in analysis

    def test_hyperparameter_tuning_integration(self, sample_embeddings, mock_config):
        """Test hyperparameter tuning integration."""
        # Enable tuning in config
        test_config = mock_config.copy()
        test_config["clustering"]["hyperparameter_tuning"]["enabled"] = True
        test_config["clustering"]["hyperparameter_tuning"]["n_trials"] = 5

        with patch(
            "clustering.theme_clusterer.config.get_clustering_config",
            return_value=test_config["clustering"],
        ):
            clusterer = ThemeClusterer()

            # Run optimization
            results = clusterer.optimize_hyperparameters(sample_embeddings)

            # Check results
            assert "best_params" in results
            assert "best_score" in results
            assert "n_trials" in results

    def test_model_persistence_integration(
        self, sample_embeddings, temp_dir, mock_config
    ):
        """Test model persistence integration."""
        # Enable persistence in config
        test_config = mock_config.copy()
        test_config["clustering"]["model_persistence"]["enabled"] = True
        test_config["clustering"]["model_persistence"]["model_dir"] = os.path.join(
            temp_dir, "test_models"
        )

        with patch(
            "clustering.theme_clusterer.config.get_clustering_config",
            return_value=test_config["clustering"],
        ):
            clusterer = ThemeClusterer()

            # Train models with persistence
            results = clusterer.fit_clusters_with_persistence(sample_embeddings)

            # Check results
            assert "embeddings_2d" in results
            assert "cluster_labels" in results
            assert "cluster_centroids" in results
            assert "statistics" in results
            assert "model_persisted" in results

    def test_error_handling_integration(self, temp_dir):
        """Test error handling in integration scenarios."""
        # Test with non-existent file
        pipeline = AnalysisPipeline()
        results = pipeline.run_analysis("nonexistent.csv")

        # Should handle error gracefully
        assert results["success"] == False
        assert "error" in results

    def test_performance_integration(self, sample_csv_data, temp_dir, mock_config):
        """Test performance characteristics of the system."""
        # Create a larger dataset
        large_csv_data = (
            sample_csv_data + "\n" + sample_csv_data
        )  # Duplicate for larger dataset

        csv_file_path = os.path.join(temp_dir, "large_test_data.csv")
        with open(csv_file_path, "w") as f:
            f.write(large_csv_data)

        # Mock AWS services for performance test
        with patch("embeddings.embedding_manager.boto3.client") as mock_bedrock:
            mock_bedrock.return_value.invoke_model.return_value = {
                "body": Mock(
                    read=lambda: json.dumps({"embedding": [0.1] * 768}).encode()
                )
            }

            with patch("llm.theme_analyzer.boto3.client") as mock_bedrock_llm:
                mock_bedrock_llm.return_value.invoke_model.return_value = {
                    "body": Mock(
                        read=lambda: json.dumps(
                            {"completion": "Test theme name"}
                        ).encode()
                    )
                }

                # Mock configuration
                with patch(
                    "core.analysis_pipeline.config.get_aws_config",
                    return_value=mock_config["aws"],
                ):
                    with patch(
                        "core.analysis_pipeline.config.get_preprocessing_config",
                        return_value=mock_config["preprocessing"],
                    ):
                        with patch(
                            "core.analysis_pipeline.config.get_clustering_config",
                            return_value=mock_config["clustering"],
                        ):

                            # Initialize pipeline
                            pipeline = AnalysisPipeline()

                            # Run analysis and measure performance
                            import time

                            start_time = time.time()
                            results = pipeline.run_analysis(csv_file_path)
                            execution_time = time.time() - start_time

                            # Check performance
                            assert results["success"] == True
                            assert (
                                execution_time < 60
                            )  # Should complete within 60 seconds
                            assert results["execution_time"] > 0

    def test_memory_usage_integration(self, sample_csv_data, temp_dir, mock_config):
        """Test memory usage characteristics."""
        import psutil
        import gc

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Create pipeline and run analysis
        with patch("embeddings.embedding_manager.boto3.client") as mock_bedrock:
            mock_bedrock.return_value.invoke_model.return_value = {
                "body": Mock(
                    read=lambda: json.dumps({"embedding": [0.1] * 768}).encode()
                )
            }

            with patch("llm.theme_analyzer.boto3.client") as mock_bedrock_llm:
                mock_bedrock_llm.return_value.invoke_model.return_value = {
                    "body": Mock(
                        read=lambda: json.dumps(
                            {"completion": "Test theme name"}
                        ).encode()
                    )
                }

                # Mock configuration
                with patch(
                    "core.analysis_pipeline.config.get_aws_config",
                    return_value=mock_config["aws"],
                ):
                    with patch(
                        "core.analysis_pipeline.config.get_preprocessing_config",
                        return_value=mock_config["preprocessing"],
                    ):
                        with patch(
                            "core.analysis_pipeline.config.get_clustering_config",
                            return_value=mock_config["clustering"],
                        ):

                            # Create CSV file
                            csv_file_path = os.path.join(temp_dir, "memory_test.csv")
                            with open(csv_file_path, "w") as f:
                                f.write(sample_csv_data)

                            # Initialize pipeline
                            pipeline = AnalysisPipeline()

                            # Run analysis
                            results = pipeline.run_analysis(csv_file_path)

                            # Check memory usage
                            final_memory = process.memory_info().rss
                            memory_increase = final_memory - initial_memory

                            # Memory increase should be reasonable (less than 1GB)
                            assert memory_increase < 1024 * 1024 * 1024

                            # Clean up
                            del pipeline
                            gc.collect()

    def test_concurrent_processing_integration(
        self, sample_csv_data, temp_dir, mock_config
    ):
        """Test concurrent processing capabilities."""
        import threading
        import time

        # Create multiple CSV files
        csv_files = []
        for i in range(3):
            csv_file_path = os.path.join(temp_dir, f"concurrent_test_{i}.csv")
            with open(csv_file_path, "w") as f:
                f.write(sample_csv_data)
            csv_files.append(csv_file_path)

        # Mock AWS services
        with patch("embeddings.embedding_manager.boto3.client") as mock_bedrock:
            mock_bedrock.return_value.invoke_model.return_value = {
                "body": Mock(
                    read=lambda: json.dumps({"embedding": [0.1] * 768}).encode()
                )
            }

            with patch("llm.theme_analyzer.boto3.client") as mock_bedrock_llm:
                mock_bedrock_llm.return_value.invoke_model.return_value = {
                    "body": Mock(
                        read=lambda: json.dumps(
                            {"completion": "Test theme name"}
                        ).encode()
                    )
                }

                # Mock configuration
                with patch(
                    "core.analysis_pipeline.config.get_aws_config",
                    return_value=mock_config["aws"],
                ):
                    with patch(
                        "core.analysis_pipeline.config.get_preprocessing_config",
                        return_value=mock_config["preprocessing"],
                    ):
                        with patch(
                            "core.analysis_pipeline.config.get_clustering_config",
                            return_value=mock_config["clustering"],
                        ):

                            results = []
                            errors = []

                            def process_file(file_path):
                                try:
                                    pipeline = AnalysisPipeline()
                                    result = pipeline.run_analysis(file_path)
                                    results.append(result)
                                except Exception as e:
                                    errors.append(str(e))

                            # Run concurrent processing
                            threads = []
                            for file_path in csv_files:
                                thread = threading.Thread(
                                    target=process_file, args=(file_path,)
                                )
                                threads.append(thread)
                                thread.start()

                            # Wait for all threads to complete
                            for thread in threads:
                                thread.join()

                            # Check results
                            assert len(errors) == 0  # No errors should occur
                            assert len(results) == 3  # All files should be processed
                            for result in results:
                                assert result["success"] == True


class TestSystemReliability:
    """Tests for system reliability and robustness."""

    def test_system_recovery(self, sample_csv_data, temp_dir, mock_config):
        """Test system recovery after errors."""
        # Create CSV file
        csv_file_path = os.path.join(temp_dir, "recovery_test.csv")
        with open(csv_file_path, "w") as f:
            f.write(sample_csv_data)

        # Mock AWS services with intermittent failures
        call_count = 0

        def mock_bedrock_response():
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise Exception("AWS service temporarily unavailable")
            return {
                "body": Mock(
                    read=lambda: json.dumps({"embedding": [0.1] * 768}).encode()
                )
            }

        with patch("embeddings.embedding_manager.boto3.client") as mock_bedrock:
            mock_bedrock.return_value.invoke_model.side_effect = mock_bedrock_response

            with patch("llm.theme_analyzer.boto3.client") as mock_bedrock_llm:
                mock_bedrock_llm.return_value.invoke_model.return_value = {
                    "body": Mock(
                        read=lambda: json.dumps(
                            {"completion": "Test theme name"}
                        ).encode()
                    )
                }

                # Mock configuration
                with patch(
                    "core.analysis_pipeline.config.get_aws_config",
                    return_value=mock_config["aws"],
                ):
                    with patch(
                        "core.analysis_pipeline.config.get_preprocessing_config",
                        return_value=mock_config["preprocessing"],
                    ):
                        with patch(
                            "core.analysis_pipeline.config.get_clustering_config",
                            return_value=mock_config["clustering"],
                        ):

                            # Initialize pipeline
                            pipeline = AnalysisPipeline()

                            # Run analysis (should handle failures gracefully)
                            results = pipeline.run_analysis(csv_file_path)

                            # System should either succeed or fail gracefully
                            assert "success" in results
                            if not results["success"]:
                                assert "error" in results

    def test_data_validation_integration(self, temp_dir):
        """Test data validation in integration scenarios."""
        # Create invalid CSV file
        invalid_csv_data = "invalid,data,format\nno,summary,column"

        csv_file_path = os.path.join(temp_dir, "invalid_test.csv")
        with open(csv_file_path, "w") as f:
            f.write(invalid_csv_data)

        # Mock configuration
        with patch("core.analysis_pipeline.config.get_aws_config", return_value={}):
            with patch(
                "core.analysis_pipeline.config.get_preprocessing_config",
                return_value={},
            ):
                with patch(
                    "core.analysis_pipeline.config.get_clustering_config",
                    return_value={},
                ):

                    # Initialize pipeline
                    pipeline = AnalysisPipeline()

                    # Run analysis
                    results = pipeline.run_analysis(csv_file_path)

                    # Should handle invalid data gracefully
                    assert results["success"] == False
                    assert "error" in results
