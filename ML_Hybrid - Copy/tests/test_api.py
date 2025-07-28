"""
Tests for FastAPI backend endpoints.
"""

import pytest
import json
from unittest.mock import patch, Mock, MagicMock
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import os

# Import the FastAPI app
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "api"))
from main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Test cases for API endpoints."""

    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_upload_file_success(self, sample_csv_data, temp_dir):
        """Test successful file upload."""
        # Create a temporary CSV file
        csv_file_path = os.path.join(temp_dir, "test_data.csv")
        with open(csv_file_path, "w") as f:
            f.write(sample_csv_data)

        # Test file upload
        with open(csv_file_path, "rb") as f:
            response = client.post(
                "/upload", files={"file": ("test_data.csv", f, "text/csv")}
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "file_path" in data["data"]
        assert "file_name" in data["data"]

    def test_upload_file_invalid_type(self):
        """Test file upload with invalid file type."""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is not a CSV file")
            temp_file_path = f.name

        try:
            with open(temp_file_path, "rb") as f:
                response = client.post(
                    "/upload", files={"file": ("test.txt", f, "text/plain")}
                )

            assert response.status_code == 400
            data = response.json()
            assert "detail" in data
        finally:
            os.unlink(temp_file_path)

    def test_analyze_file_success(self, sample_csv_data, temp_dir):
        """Test successful file analysis."""
        # Create a temporary CSV file
        csv_file_path = os.path.join(temp_dir, "test_data.csv")
        with open(csv_file_path, "w") as f:
            f.write(sample_csv_data)

        # Mock the analysis pipeline
        with patch("api.main.pipeline") as mock_pipeline:
            mock_pipeline.run_analysis.return_value = {
                "success": True,
                "execution_time": 10.5,
                "preprocessing_results": {"statistics": {"processed_count": 10}},
                "clustering_results": {"statistics": {"clustering_quality": "Good"}},
                "theme_analysis": {0: {"theme_name": "Test Theme"}},
            }

            response = client.post("/analyze", json={"file_path": csv_file_path})

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "analysis_id" in data["data"]
            assert "execution_time" in data["data"]

    def test_analyze_file_failure(self, temp_dir):
        """Test file analysis failure."""
        # Create a temporary CSV file
        csv_file_path = os.path.join(temp_dir, "test_data.csv")
        with open(csv_file_path, "w") as f:
            f.write("invalid,csv,data")

        # Mock the analysis pipeline to return failure
        with patch("api.main.pipeline") as mock_pipeline:
            mock_pipeline.run_analysis.return_value = {
                "success": False,
                "error": "Analysis failed",
            }

            response = client.post("/analyze", json={"file_path": csv_file_path})

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == False
            assert "error" in data

    def test_get_analysis_results(self):
        """Test getting analysis results."""
        # Mock analysis results
        mock_results = {
            "file_path": "test.csv",
            "execution_time": 10.5,
            "preprocessing_results": {"statistics": {"processed_count": 10}},
            "clustering_results": {"statistics": {"num_clusters": 3}},
            "theme_analysis": {0: {"theme_name": "Test Theme"}},
        }

        with patch("api.main.analysis_results", {"test_analysis": mock_results}):
            response = client.get("/results/test_analysis")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "results" in data

    def test_get_analysis_results_not_found(self):
        """Test getting non-existent analysis results."""
        response = client.get("/results/nonexistent")
        assert response.status_code == 404

    def test_get_themes(self):
        """Test getting themes for analysis."""
        # Mock analysis results with themes
        mock_results = {
            "theme_analysis": {
                0: {
                    "theme_name": "Database Issues",
                    "description": "Database problems",
                },
                1: {"theme_name": "API Errors", "description": "API issues"},
            }
        }

        with patch("api.main.analysis_results", {"test_analysis": mock_results}):
            response = client.get("/themes/test_analysis")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "themes" in data
            assert len(data["themes"]) == 2

    def test_get_clusters(self):
        """Test getting clusters for analysis."""
        # Mock analysis results with clusters
        mock_results = {
            "clustering_results": {
                "embeddings_2d": [[0.1, 0.2], [0.3, 0.4]],
                "cluster_labels": [0, 1],
                "statistics": {"num_clusters": 2},
            }
        }

        with patch("api.main.analysis_results", {"test_analysis": mock_results}):
            response = client.get("/clusters/test_analysis")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "clusters" in data
            assert "statistics" in data

    def test_chat_endpoint(self):
        """Test chat endpoint."""
        response = client.post(
            "/chat", json={"message": "What are the main themes?", "context": {}}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "message" in data

    def test_compare_files(self):
        """Test file comparison endpoint."""
        file_paths = ["file1.csv", "file2.csv"]

        # Mock the comparison pipeline
        with patch("api.main.pipeline") as mock_pipeline:
            mock_pipeline.compare_files.return_value = {
                "success": True,
                "comparison_results": {"shared_themes": 2, "unique_themes": 3},
                "file_results": {"file1": {}, "file2": {}},
            }

            response = client.post("/compare", json=file_paths)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "comparison_results" in data

    def test_get_reports(self):
        """Test getting analysis reports."""
        # Mock analysis results with reports
        mock_results = {
            "reports": {
                "data_summary": "Summary report",
                "clustering_report": "Clustering report",
                "theme_analysis_report": "Theme analysis report",
            }
        }

        with patch("api.main.analysis_results", {"test_analysis": mock_results}):
            response = client.get("/reports/test_analysis")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "reports" in data

    def test_delete_analysis(self):
        """Test deleting analysis results."""
        # Mock analysis results
        with patch("api.main.analysis_results", {"test_analysis": {"data": "test"}}):
            response = client.delete("/results/test_analysis")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "message" in data

    def test_delete_analysis_not_found(self):
        """Test deleting non-existent analysis."""
        response = client.delete("/results/nonexistent")
        assert response.status_code == 404


class TestAdvancedEndpoints:
    """Test cases for advanced API endpoints."""

    def test_optimize_hyperparameters(self, sample_embeddings):
        """Test hyperparameter optimization endpoint."""
        with patch("api.main.pipeline") as mock_pipeline:
            mock_pipeline.theme_clusterer.optimize_hyperparameters.return_value = {
                "best_params": {"umap_n_neighbors": 20, "hdbscan_min_cluster_size": 10},
                "best_score": 0.85,
                "n_trials": 100,
            }

            response = client.post(
                "/optimize-hyperparameters", json={"embeddings": sample_embeddings}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "optimization_results" in data

    def test_predict_clusters(self, sample_embeddings):
        """Test cluster prediction endpoint."""
        with patch("api.main.pipeline") as mock_pipeline:
            mock_pipeline.theme_clusterer.predict_clusters_for_new_data.return_value = {
                "embeddings_2d": [[0.1, 0.2], [0.3, 0.4]],
                "cluster_labels": [0, 1],
                "statistics": {"num_clusters": 2},
            }

            response = client.post(
                "/predict-clusters", json={"embeddings": sample_embeddings}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "prediction_results" in data

    def test_check_drift(self, sample_embeddings):
        """Test drift detection endpoint."""
        with patch("api.main.pipeline") as mock_pipeline:
            mock_pipeline.theme_clusterer.check_model_drift.return_value = {
                "drift_score": 0.15,
                "threshold": 0.1,
                "needs_retraining": True,
                "drift_severity": "Medium",
            }

            response = client.post(
                "/check-drift", json={"embeddings": sample_embeddings}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "drift_analysis" in data

    def test_get_model_info(self):
        """Test model information endpoint."""
        with patch("api.main.pipeline") as mock_pipeline:
            mock_pipeline.theme_clusterer.get_model_info.return_value = {
                "n_training_samples": 1000,
                "n_clusters": 8,
                "cluster_ids": [0, 1, 2, 3, 4, 5, 6, 7],
                "model_metadata": {"version": "1.0"},
            }

            response = client.get("/model-info")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "model_info" in data

    def test_retrain_models(self, sample_embeddings):
        """Test model retraining endpoint."""
        with patch("api.main.pipeline") as mock_pipeline:
            mock_pipeline.theme_clusterer.fit_clusters_with_persistence.return_value = {
                "embeddings_2d": [[0.1, 0.2], [0.3, 0.4]],
                "cluster_labels": [0, 1],
                "cluster_centroids": {0: [0.5, 0.3], 1: [0.2, 0.8]},
                "statistics": {"num_clusters": 2},
                "model_persisted": True,
            }

            response = client.post(
                "/retrain-models", json={"embeddings": sample_embeddings}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert "retraining_results" in data

    def test_optimize_hyperparameters_no_embeddings(self):
        """Test hyperparameter optimization with no embeddings."""
        response = client.post("/optimize-hyperparameters", json={"embeddings": []})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == False
        assert "error" in data

    def test_predict_clusters_no_embeddings(self):
        """Test cluster prediction with no embeddings."""
        response = client.post("/predict-clusters", json={"embeddings": []})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] == False
        assert "error" in data


class TestErrorHandling:
    """Test cases for error handling in API endpoints."""

    def test_upload_file_error(self):
        """Test file upload error handling."""
        # Test with non-existent file
        response = client.post(
            "/upload", files={"file": ("nonexistent.csv", b"", "text/csv")}
        )

        # Should handle the error gracefully
        assert response.status_code in [400, 500]

    def test_analyze_file_error(self):
        """Test file analysis error handling."""
        response = client.post("/analyze", json={"file_path": "nonexistent.csv"})

        # Should handle the error gracefully
        assert response.status_code in [200, 500]

    def test_invalid_json(self):
        """Test invalid JSON handling."""
        response = client.post(
            "/analyze",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_missing_required_fields(self):
        """Test missing required fields."""
        response = client.post("/analyze", json={})  # Missing file_path

        assert response.status_code == 422
