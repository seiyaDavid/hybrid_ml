"""
Pytest configuration and fixtures for ML Hybrid Theme Analysis tests.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import ConfigManager
from utils.logger import setup_logging


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    return np.random.rand(100, 768).tolist()


@pytest.fixture
def sample_texts():
    """Generate sample texts for testing."""
    return [
        "Database connection failed due to timeout",
        "User authentication error occurred",
        "API endpoint returning 500 error",
        "Data validation failed for user input",
        "Memory usage exceeded limits",
        "Network connectivity issues detected",
        "File upload failed with size limit",
        "Configuration parsing error",
        "Service unavailable due to maintenance",
        "Permission denied for resource access",
    ] * 10  # 100 total texts


@pytest.fixture
def sample_csv_data():
    """Generate sample CSV data for testing."""
    return """id,summary,priority,status
1,Database connection failed due to timeout,high,open
2,User authentication error occurred,medium,closed
3,API endpoint returning 500 error,high,open
4,Data validation failed for user input,low,open
5,Memory usage exceeded limits,high,closed
6,Network connectivity issues detected,medium,open
7,File upload failed with size limit,low,closed
8,Configuration parsing error,medium,open
9,Service unavailable due to maintenance,high,closed
10,Permission denied for resource access,medium,open"""


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "aws": {
            "region": "us-east-1",
            "bedrock": {
                "model_id": "anthropic.claude-3-sonnet-20240229-v1:0",
                "embedding_model_id": "amazon.titan-embed-text-v1",
            },
            "credentials": {
                "access_key_id": "test_key",
                "secret_access_key": "test_secret",
            },
        },
        "app": {
            "name": "ML Hybrid Theme Analysis",
            "version": "1.0.0",
            "debug": True,
            "host": "0.0.0.0",
            "port": 8000,
        },
        "logging": {
            "level": "INFO",
            "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            "file": "logs/test.log",
            "rotation": "10 MB",
            "retention": "30 days",
        },
        "preprocessing": {
            "min_summary_length": 10,
            "max_summary_length": 2000,
            "supported_languages": ["en"],
            "remove_html": True,
            "normalize_case": True,
            "remove_duplicates": True,
            "lemmatization": True,
            "remove_boilerplate": True,
        },
        "clustering": {
            "umap": {
                "n_neighbors": 15,
                "n_components": 2,
                "min_dist": 0.1,
                "metric": "cosine",
            },
            "hdbscan": {
                "min_cluster_size": 5,
                "min_samples": 3,
                "cluster_selection_epsilon": 0.1,
            },
            "hyperparameter_tuning": {
                "enabled": False,
                "n_trials": 10,
                "study_name": "test_optimization",
                "save_optimization_results": False,
            },
            "model_persistence": {
                "enabled": False,
                "model_dir": "test_models",
                "retrain_threshold": 0.1,
                "drift_detection": False,
            },
        },
        "vector_store": {
            "type": "faiss",
            "index_path": "data/test_vector_store",
            "similarity_metric": "cosine",
        },
        "theme_analysis": {
            "max_samples_per_cluster": 10,
            "classification_threshold": 0.7,
        },
    }


@pytest.fixture
def mock_aws_credentials():
    """Mock AWS credentials for testing."""
    return {
        "access_key_id": "test_access_key",
        "secret_access_key": "test_secret_key",
        "region": "us-east-1",
    }


@pytest.fixture
def sample_cluster_results():
    """Sample clustering results for testing."""
    return {
        "embeddings_2d": np.random.rand(100, 2).tolist(),
        "cluster_labels": [0, 0, 1, 1, 2, 2, -1, -1, 0, 1] * 10,
        "cluster_centroids": {0: [0.5, 0.3], 1: [0.2, 0.8], 2: [0.8, 0.1]},
        "statistics": {
            "num_clusters": 3,
            "num_noise_points": 20,
            "total_points": 100,
            "cluster_sizes": {0: 30, 1: 30, 2: 20},
            "silhouette_score": 0.45,
            "clustering_quality": "Good",
        },
    }


@pytest.fixture
def sample_theme_analysis():
    """Sample theme analysis results for testing."""
    return {
        0: {
            "theme_name": "Database Issues",
            "description": "Issues related to database connectivity and performance",
            "sample_count": 30,
            "data_quality_issues": 5,
            "representative_examples": [
                "Database connection failed due to timeout",
                "Memory usage exceeded limits",
            ],
        },
        1: {
            "theme_name": "Authentication Problems",
            "description": "User authentication and authorization issues",
            "sample_count": 30,
            "data_quality_issues": 3,
            "representative_examples": [
                "User authentication error occurred",
                "Permission denied for resource access",
            ],
        },
        2: {
            "theme_name": "API Errors",
            "description": "API endpoint failures and service issues",
            "sample_count": 20,
            "data_quality_issues": 2,
            "representative_examples": [
                "API endpoint returning 500 error",
                "Service unavailable due to maintenance",
            ],
        },
    }


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup test logging."""
    setup_logging()
    yield


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Cleanup test files after tests."""
    yield
    # Cleanup any test files created during tests
    test_files = [
        "test_models",
        "logs/test.log",
        "data/test_vector_store",
        "test_optimization_results.joblib",
    ]
    for file_path in test_files:
        if os.path.exists(file_path):
            if os.path.isdir(file_path):
                import shutil

                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
