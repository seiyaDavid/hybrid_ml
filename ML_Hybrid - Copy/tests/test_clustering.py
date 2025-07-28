"""
Tests for clustering modules including hyperparameter tuning and model persistence.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

from clustering.theme_clusterer import ThemeClusterer
from clustering.hyperparameter_tuner import ClusteringHyperparameterTuner
from clustering.model_persistence import ClusteringModelPersistence


class TestThemeClusterer:
    """Test cases for ThemeClusterer class."""

    def test_theme_clusterer_initialization(self, mock_config):
        """Test ThemeClusterer initialization."""
        with patch(
            "clustering.theme_clusterer.config.get_clustering_config",
            return_value=mock_config["clustering"],
        ):
            clusterer = ThemeClusterer()
            assert clusterer is not None
            assert hasattr(clusterer, "umap_reducer")
            assert hasattr(clusterer, "hdbscan_clusterer")
            assert hasattr(clusterer, "fit_clusters")

    def test_fit_clusters_basic(self, sample_embeddings, mock_config):
        """Test basic clustering functionality."""
        with patch(
            "clustering.theme_clusterer.config.get_clustering_config",
            return_value=mock_config["clustering"],
        ):
            clusterer = ThemeClusterer()

            # Test clustering
            results = clusterer.fit_clusters(sample_embeddings)

            # Check results structure
            assert "embeddings_2d" in results
            assert "cluster_labels" in results
            assert "cluster_centroids" in results
            assert "statistics" in results

            # Check data types
            assert isinstance(results["embeddings_2d"], list)
            assert isinstance(results["cluster_labels"], list)
            assert isinstance(results["cluster_centroids"], dict)
            assert isinstance(results["statistics"], dict)

    def test_fit_clusters_with_persistence(self, sample_embeddings, mock_config):
        """Test clustering with persistence enabled."""
        # Enable persistence in config
        test_config = mock_config.copy()
        test_config["clustering"]["model_persistence"]["enabled"] = True

        with patch(
            "clustering.theme_clusterer.config.get_clustering_config",
            return_value=test_config["clustering"],
        ):
            clusterer = ThemeClusterer()

            # Test clustering with persistence
            results = clusterer.fit_clusters_with_persistence(sample_embeddings)

            # Check results
            assert "embeddings_2d" in results
            assert "cluster_labels" in results
            assert "cluster_centroids" in results
            assert "statistics" in results
            assert "model_persisted" in results

    def test_calculate_centroids(self, sample_embeddings, mock_config):
        """Test centroid calculation."""
        with patch(
            "clustering.theme_clusterer.config.get_clustering_config",
            return_value=mock_config["clustering"],
        ):
            clusterer = ThemeClusterer()

            # Fit clusters first
            clusterer.fit_clusters(sample_embeddings)

            # Calculate centroids
            centroids = clusterer._calculate_centroids()

            # Check centroids
            assert isinstance(centroids, dict)
            for cluster_id, centroid in centroids.items():
                assert isinstance(cluster_id, int)
                assert isinstance(centroid, list)
                assert len(centroid) == 2  # 2D coordinates

    def test_generate_clustering_statistics(self, sample_embeddings, mock_config):
        """Test clustering statistics generation."""
        with patch(
            "clustering.theme_clusterer.config.get_clustering_config",
            return_value=mock_config["clustering"],
        ):
            clusterer = ThemeClusterer()

            # Fit clusters first
            clusterer.fit_clusters(sample_embeddings)

            # Generate statistics
            stats = clusterer._generate_clustering_statistics()

            # Check statistics structure
            assert "num_clusters" in stats
            assert "num_noise_points" in stats
            assert "total_points" in stats
            assert "cluster_sizes" in stats
            assert "silhouette_score" in stats
            assert "clustering_quality" in stats

            # Check data types
            assert isinstance(stats["num_clusters"], int)
            assert isinstance(stats["num_noise_points"], int)
            assert isinstance(stats["total_points"], int)
            assert isinstance(stats["cluster_sizes"], dict)
            assert isinstance(stats["silhouette_score"], float)
            assert isinstance(stats["clustering_quality"], str)

    def test_get_cluster_samples(self, sample_texts, sample_embeddings, mock_config):
        """Test getting cluster samples."""
        with patch(
            "clustering.theme_clusterer.config.get_clustering_config",
            return_value=mock_config["clustering"],
        ):
            clusterer = ThemeClusterer()

            # Fit clusters first
            clusterer.fit_clusters(sample_embeddings)

            # Get cluster samples
            samples = clusterer.get_cluster_samples(sample_texts, max_samples=5)

            # Check samples
            assert isinstance(samples, dict)
            for cluster_id, cluster_samples in samples.items():
                assert isinstance(cluster_id, int)
                assert isinstance(cluster_samples, list)
                assert len(cluster_samples) <= 5

    def test_find_similar_clusters(self, sample_embeddings, mock_config):
        """Test finding similar clusters."""
        with patch(
            "clustering.theme_clusterer.config.get_clustering_config",
            return_value=mock_config["clustering"],
        ):
            clusterer = ThemeClusterer()

            # Fit clusters first
            clusterer.fit_clusters(sample_embeddings)

            # Find similar clusters
            similar = clusterer.find_similar_clusters(0, similarity_threshold=0.5)

            # Check results
            assert isinstance(similar, list)
            for cluster_info in similar:
                assert "cluster_id" in cluster_info
                assert "similarity" in cluster_info
                assert isinstance(cluster_info["similarity"], float)

    def test_error_handling(self, mock_config):
        """Test error handling in clustering."""
        with patch(
            "clustering.theme_clusterer.config.get_clustering_config",
            return_value=mock_config["clustering"],
        ):
            clusterer = ThemeClusterer()

            # Test with empty embeddings
            with pytest.raises(ValueError):
                clusterer.fit_clusters([])

            # Test with None embeddings
            with pytest.raises(ValueError):
                clusterer.fit_clusters(None)


class TestHyperparameterTuner:
    """Test cases for ClusteringHyperparameterTuner class."""

    def test_tuner_initialization(self, sample_embeddings):
        """Test hyperparameter tuner initialization."""
        tuner = ClusteringHyperparameterTuner(sample_embeddings, n_trials=10)
        assert tuner is not None
        assert tuner.n_trials == 10
        assert len(tuner.embeddings) == 100

    def test_objective_function(self, sample_embeddings):
        """Test objective function for optimization."""
        tuner = ClusteringHyperparameterTuner(sample_embeddings, n_trials=10)

        # Create a mock trial
        mock_trial = Mock()
        mock_trial.number = 1
        mock_trial.suggest_int = Mock(side_effect=[15, 2, 5, 3, 0.1])
        mock_trial.suggest_float = Mock(side_effect=[0.1, 0.1])

        # Test objective function
        score = tuner.objective(mock_trial)

        # Check that score is a float
        assert isinstance(score, float)
        assert score >= -1.0  # Error case returns -1.0

    def test_calculate_clustering_score(self, sample_embeddings):
        """Test clustering score calculation."""
        tuner = ClusteringHyperparameterTuner(sample_embeddings, n_trials=10)

        # Create sample 2D embeddings and cluster labels
        embeddings_2d = np.random.rand(100, 2)
        cluster_labels = np.random.randint(0, 5, 100)

        # Calculate score
        score = tuner._calculate_clustering_score(embeddings_2d, cluster_labels)

        # Check score
        assert isinstance(score, float)
        assert score >= 0.0

    def test_optimization(self, sample_embeddings):
        """Test hyperparameter optimization."""
        tuner = ClusteringHyperparameterTuner(sample_embeddings, n_trials=5)

        # Run optimization
        results = tuner.optimize("test_study")

        # Check results
        assert "best_params" in results
        assert "best_score" in results
        assert "n_trials" in results
        assert "study" in results

        # Check data types
        assert isinstance(results["best_params"], dict)
        assert isinstance(results["best_score"], float)
        assert results["n_trials"] == 5

    def test_get_optimized_clusterer(self, sample_embeddings):
        """Test getting optimized clusterer."""
        tuner = ClusteringHyperparameterTuner(sample_embeddings, n_trials=5)

        # Run optimization first
        tuner.optimize("test_study")

        # Get optimized clusterer
        umap_model, hdbscan_model = tuner.get_optimized_clusterer()

        # Check models
        assert umap_model is not None
        assert hdbscan_model is not None

    def test_save_load_optimization_results(self, sample_embeddings, temp_dir):
        """Test saving and loading optimization results."""
        tuner = ClusteringHyperparameterTuner(sample_embeddings, n_trials=5)

        # Run optimization
        tuner.optimize("test_study")

        # Save results
        results_path = os.path.join(temp_dir, "test_results.joblib")
        tuner.save_optimization_results(results_path)

        # Check file exists
        assert os.path.exists(results_path)

        # Load results
        new_tuner = ClusteringHyperparameterTuner(sample_embeddings, n_trials=5)
        new_tuner.load_optimization_results(results_path)

        # Check loaded results
        assert new_tuner.best_params is not None
        assert new_tuner.best_score is not None

    def test_get_optimization_report(self, sample_embeddings):
        """Test optimization report generation."""
        tuner = ClusteringHyperparameterTuner(sample_embeddings, n_trials=5)

        # Run optimization
        tuner.optimize("test_study")

        # Get report
        report = tuner.get_optimization_report()

        # Check report
        assert isinstance(report, str)
        assert "Optimization Report" in report
        assert "Best Parameters" in report


class TestModelPersistence:
    """Test cases for ClusteringModelPersistence class."""

    def test_persistence_initialization(self, temp_dir):
        """Test model persistence initialization."""
        model_dir = os.path.join(temp_dir, "test_models")
        persistence = ClusteringModelPersistence(model_dir)

        assert persistence.model_dir == Path(model_dir)
        assert persistence.model_dir.exists()

    def test_save_load_models(self, temp_dir, sample_embeddings, mock_config):
        """Test saving and loading models."""
        model_dir = os.path.join(temp_dir, "test_models")
        persistence = ClusteringModelPersistence(model_dir)

        # Create mock models
        mock_umap = Mock()
        mock_hdbscan = Mock()
        training_embeddings = np.random.rand(100, 768)
        cluster_centroids = {0: [0.5, 0.3], 1: [0.2, 0.8]}
        metadata = {"training_samples": 100, "n_clusters": 2}

        # Save models
        persistence.save_models(
            mock_umap, mock_hdbscan, training_embeddings, cluster_centroids, metadata
        )

        # Check files exist
        assert persistence.umap_model_path.exists()
        assert persistence.hdbscan_model_path.exists()
        assert persistence.embeddings_path.exists()
        assert persistence.cluster_centroids_path.exists()
        assert persistence.model_metadata_path.exists()

        # Load models
        (
            loaded_umap,
            loaded_hdbscan,
            loaded_embeddings,
            loaded_centroids,
            loaded_metadata,
        ) = persistence.load_models()

        # Check loaded models
        assert loaded_umap is not None
        assert loaded_hdbscan is not None
        assert loaded_embeddings is not None
        assert loaded_centroids is not None
        assert loaded_metadata is not None

    def test_predict_clusters(self, temp_dir, sample_embeddings):
        """Test cluster prediction with saved models."""
        model_dir = os.path.join(temp_dir, "test_models")
        persistence = ClusteringModelPersistence(model_dir)

        # Create and save mock models
        mock_umap = Mock()
        mock_umap.transform.return_value = np.random.rand(50, 2)
        mock_hdbscan = Mock()
        mock_hdbscan.fit_predict.return_value = np.random.randint(0, 3, 50)

        training_embeddings = np.random.rand(100, 768)
        cluster_centroids = {0: [0.5, 0.3], 1: [0.2, 0.8]}
        metadata = {"training_samples": 100, "n_clusters": 2}

        persistence.save_models(
            mock_umap, mock_hdbscan, training_embeddings, cluster_centroids, metadata
        )

        # Predict clusters
        new_embeddings = np.random.rand(50, 768)
        cluster_labels, embeddings_2d = persistence.predict_clusters(new_embeddings)

        # Check results
        assert len(cluster_labels) == 50
        assert len(embeddings_2d) == 50
        assert embeddings_2d.shape[1] == 2

    def test_find_nearest_cluster(self, temp_dir):
        """Test finding nearest cluster."""
        model_dir = os.path.join(temp_dir, "test_models")
        persistence = ClusteringModelPersistence(model_dir)

        # Create and save mock models
        mock_umap = Mock()
        mock_umap.transform.return_value = np.array([[0.3, 0.4]])
        mock_hdbscan = Mock()

        training_embeddings = np.random.rand(100, 768)
        cluster_centroids = {0: [0.5, 0.3], 1: [0.2, 0.8]}
        metadata = {"training_samples": 100, "n_clusters": 2}

        persistence.save_models(
            mock_umap, mock_hdbscan, training_embeddings, cluster_centroids, metadata
        )

        # Find nearest cluster
        new_embedding = np.random.rand(768)
        cluster_id, distance = persistence.find_nearest_cluster(new_embedding)

        # Check results
        assert isinstance(cluster_id, int)
        assert isinstance(distance, float)
        assert distance >= 0.0

    def test_get_cluster_statistics(self, temp_dir):
        """Test getting cluster statistics."""
        model_dir = os.path.join(temp_dir, "test_models")
        persistence = ClusteringModelPersistence(model_dir)

        # Create and save mock models
        mock_umap = Mock()
        mock_hdbscan = Mock()
        training_embeddings = np.random.rand(100, 768)
        cluster_centroids = {0: [0.5, 0.3], 1: [0.2, 0.8]}
        metadata = {"training_samples": 100, "n_clusters": 2}

        persistence.save_models(
            mock_umap, mock_hdbscan, training_embeddings, cluster_centroids, metadata
        )

        # Get statistics
        stats = persistence.get_cluster_statistics()

        # Check statistics
        assert "n_training_samples" in stats
        assert "n_clusters" in stats
        assert "cluster_ids" in stats
        assert "model_metadata" in stats

        assert stats["n_training_samples"] == 100
        assert stats["n_clusters"] == 2
        assert stats["cluster_ids"] == [0, 1]

    def test_calculate_drift_score(self, temp_dir):
        """Test drift score calculation."""
        model_dir = os.path.join(temp_dir, "test_models")
        persistence = ClusteringModelPersistence(model_dir)

        # Create sample embeddings
        training_embeddings = np.random.rand(100, 768)
        new_embeddings = np.random.rand(50, 768)

        # Calculate drift score
        drift_score = persistence._calculate_drift_score(
            new_embeddings, training_embeddings
        )

        # Check drift score
        assert isinstance(drift_score, float)
        assert 0.0 <= drift_score <= 1.0

    def test_models_exist_check(self, temp_dir):
        """Test checking if models exist."""
        model_dir = os.path.join(temp_dir, "test_models")
        persistence = ClusteringModelPersistence(model_dir)

        # Initially no models exist
        assert not persistence._models_exist()

        # Create mock models
        mock_umap = Mock()
        mock_hdbscan = Mock()
        training_embeddings = np.random.rand(100, 768)
        cluster_centroids = {0: [0.5, 0.3]}
        metadata = {"training_samples": 100}

        persistence.save_models(
            mock_umap, mock_hdbscan, training_embeddings, cluster_centroids, metadata
        )

        # Now models should exist
        assert persistence._models_exist()

    def test_error_handling(self, temp_dir):
        """Test error handling in model persistence."""
        model_dir = os.path.join(temp_dir, "test_models")
        persistence = ClusteringModelPersistence(model_dir)

        # Test loading non-existent models
        with pytest.raises(FileNotFoundError):
            persistence.load_models()

        # Test saving with None models
        with pytest.raises(Exception):
            persistence.save_models(None, None, None, None, None)
