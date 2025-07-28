"""
Model persistence system for unsupervised clustering.

This module provides functionality to save and load trained clustering models,
enabling inference without retraining for new data points.
"""

import joblib
import numpy as np
import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import umap
import hdbscan
from sklearn.neighbors import NearestNeighbors
from loguru import logger

from ..utils.config import config
from ..utils.logger import get_logger

# Initialize logger
log = get_logger(__name__)


class ClusteringModelPersistence:
    """
    Model persistence system for unsupervised clustering.

    This class handles saving and loading of trained clustering models,
    enabling efficient inference for new data points without retraining.
    """

    def __init__(self, model_dir: str = "models/clustering"):
        """
        Initialize the model persistence system.

        Args:
            model_dir: Directory to store model files
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Model file paths
        self.umap_model_path = self.model_dir / "umap_model.joblib"
        self.hdbscan_model_path = self.model_dir / "hdbscan_model.joblib"
        self.embeddings_path = self.model_dir / "training_embeddings.joblib"
        self.cluster_centroids_path = self.model_dir / "cluster_centroids.joblib"
        self.model_metadata_path = self.model_dir / "model_metadata.joblib"

        log.info(f"Model persistence initialized with directory: {self.model_dir}")

    def save_models(
        self,
        umap_model: umap.UMAP,
        hdbscan_model: hdbscan.HDBSCAN,
        training_embeddings: np.ndarray,
        cluster_centroids: Dict[int, np.ndarray],
        metadata: Dict[str, Any],
    ) -> None:
        """
        Save trained clustering models and metadata.

        Args:
            umap_model: Trained UMAP model
            hdbscan_model: Trained HDBSCAN model
            training_embeddings: Original training embeddings
            cluster_centroids: Cluster centroid coordinates
            metadata: Additional model metadata
        """
        try:
            # Save UMAP model
            joblib.dump(umap_model, self.umap_model_path)
            log.info(f"UMAP model saved to {self.umap_model_path}")

            # Save HDBSCAN model
            joblib.dump(hdbscan_model, self.hdbscan_model_path)
            log.info(f"HDBSCAN model saved to {self.hdbscan_model_path}")

            # Save training embeddings
            joblib.dump(training_embeddings, self.embeddings_path)
            log.info(f"Training embeddings saved to {self.embeddings_path}")

            # Save cluster centroids
            joblib.dump(cluster_centroids, self.cluster_centroids_path)
            log.info(f"Cluster centroids saved to {self.cluster_centroids_path}")

            # Save metadata
            joblib.dump(metadata, self.model_metadata_path)
            log.info(f"Model metadata saved to {self.model_metadata_path}")

        except Exception as e:
            log.error(f"Error saving models: {e}")
            raise

    def load_models(
        self,
    ) -> Tuple[
        umap.UMAP, hdbscan.HDBSCAN, np.ndarray, Dict[int, np.ndarray], Dict[str, Any]
    ]:
        """
        Load trained clustering models and metadata.

        Returns:
            Tuple of (umap_model, hdbscan_model, training_embeddings, cluster_centroids, metadata)
        """
        try:
            # Check if models exist
            if not self._models_exist():
                raise FileNotFoundError(
                    "No trained models found. Please train models first."
                )

            # Load UMAP model
            umap_model = joblib.load(self.umap_model_path)
            log.info(f"UMAP model loaded from {self.umap_model_path}")

            # Load HDBSCAN model
            hdbscan_model = joblib.load(self.hdbscan_model_path)
            log.info(f"HDBSCAN model loaded from {self.hdbscan_model_path}")

            # Load training embeddings
            training_embeddings = joblib.load(self.embeddings_path)
            log.info(f"Training embeddings loaded from {self.embeddings_path}")

            # Load cluster centroids
            cluster_centroids = joblib.load(self.cluster_centroids_path)
            log.info(f"Cluster centroids loaded from {self.cluster_centroids_path}")

            # Load metadata
            metadata = joblib.load(self.model_metadata_path)
            log.info(f"Model metadata loaded from {self.model_metadata_path}")

            return (
                umap_model,
                hdbscan_model,
                training_embeddings,
                cluster_centroids,
                metadata,
            )

        except Exception as e:
            log.error(f"Error loading models: {e}")
            raise

    def _models_exist(self) -> bool:
        """
        Check if all model files exist.

        Returns:
            True if all model files exist
        """
        required_files = [
            self.umap_model_path,
            self.hdbscan_model_path,
            self.embeddings_path,
            self.cluster_centroids_path,
            self.model_metadata_path,
        ]

        return all(file.exists() for file in required_files)

    def predict_clusters(
        self, new_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict clusters for new embeddings using saved models.

        Args:
            new_embeddings: New embeddings to cluster

        Returns:
            Tuple of (cluster_labels, embeddings_2d)
        """
        try:
            # Load trained models
            (
                umap_model,
                hdbscan_model,
                training_embeddings,
                cluster_centroids,
                metadata,
            ) = self.load_models()

            # Transform new embeddings using trained UMAP
            new_embeddings_2d = umap_model.transform(new_embeddings)

            # Predict clusters using trained HDBSCAN
            cluster_labels = hdbscan_model.fit_predict(new_embeddings_2d)

            log.info(f"Predicted clusters for {len(new_embeddings)} new embeddings")

            return cluster_labels, new_embeddings_2d

        except Exception as e:
            log.error(f"Error predicting clusters: {e}")
            raise

    def find_nearest_cluster(self, new_embedding: np.ndarray) -> Tuple[int, float]:
        """
        Find the nearest cluster for a single new embedding.

        Args:
            new_embedding: Single embedding vector

        Returns:
            Tuple of (cluster_id, distance)
        """
        try:
            # Load models and centroids
            umap_model, _, _, cluster_centroids, _ = self.load_models()

            # Transform new embedding
            new_embedding_2d = umap_model.transform(new_embedding.reshape(1, -1))

            # Find nearest cluster centroid
            min_distance = float("inf")
            nearest_cluster = -1

            for cluster_id, centroid in cluster_centroids.items():
                distance = np.linalg.norm(new_embedding_2d[0] - centroid)
                if distance < min_distance:
                    min_distance = distance
                    nearest_cluster = cluster_id

            return nearest_cluster, min_distance

        except Exception as e:
            log.error(f"Error finding nearest cluster: {e}")
            raise

    def get_cluster_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the trained clustering model.

        Returns:
            Dictionary containing cluster statistics
        """
        try:
            _, _, training_embeddings, cluster_centroids, metadata = self.load_models()

            stats = {
                "n_training_samples": len(training_embeddings),
                "n_clusters": len(cluster_centroids),
                "cluster_ids": list(cluster_centroids.keys()),
                "model_metadata": metadata,
            }

            return stats

        except Exception as e:
            log.error(f"Error getting cluster statistics: {e}")
            raise

    def update_models(
        self,
        new_embeddings: np.ndarray,
        new_cluster_labels: np.ndarray,
        retrain_threshold: float = 0.1,
    ) -> bool:
        """
        Update models with new data if significant changes detected.

        Args:
            new_embeddings: New embeddings to consider
            new_cluster_labels: Cluster labels for new embeddings
            retrain_threshold: Threshold for triggering retraining

        Returns:
            True if models were updated, False otherwise
        """
        try:
            # Load current models
            (
                umap_model,
                hdbscan_model,
                training_embeddings,
                cluster_centroids,
                metadata,
            ) = self.load_models()

            # Calculate drift metric
            drift_score = self._calculate_drift_score(
                new_embeddings, training_embeddings
            )

            if drift_score > retrain_threshold:
                log.info(
                    f"Significant drift detected ({drift_score:.3f}), triggering model update"
                )

                # Combine old and new data
                combined_embeddings = np.vstack([training_embeddings, new_embeddings])

                # Retrain models (this would be done in the main pipeline)
                # For now, just log the need for retraining
                log.warning("Model update required - significant data drift detected")
                return True
            else:
                log.info(
                    f"No significant drift detected ({drift_score:.3f}), keeping current models"
                )
                return False

        except Exception as e:
            log.error(f"Error updating models: {e}")
            raise

    def _calculate_drift_score(
        self, new_embeddings: np.ndarray, training_embeddings: np.ndarray
    ) -> float:
        """
        Calculate data drift score between new and training embeddings.

        Args:
            new_embeddings: New embedding vectors
            training_embeddings: Training embedding vectors

        Returns:
            Drift score (0-1, higher means more drift)
        """
        try:
            # Calculate mean embeddings
            new_mean = np.mean(new_embeddings, axis=0)
            training_mean = np.mean(training_embeddings, axis=0)

            # Calculate cosine distance between means
            cosine_distance = 1 - np.dot(new_mean, training_mean) / (
                np.linalg.norm(new_mean) * np.linalg.norm(training_mean)
            )

            return cosine_distance

        except Exception as e:
            log.error(f"Error calculating drift score: {e}")
            return 0.0

    def export_model_info(self, output_path: str) -> None:
        """
        Export model information to a readable format.

        Args:
            output_path: Path to save model information
        """
        try:
            stats = self.get_cluster_statistics()

            info = f"""
Clustering Model Information
===========================

Model Directory: {self.model_dir}
Training Samples: {stats['n_training_samples']}
Number of Clusters: {stats['n_clusters']}
Cluster IDs: {stats['cluster_ids']}

Model Files:
- UMAP Model: {self.umap_model_path}
- HDBSCAN Model: {self.hdbscan_model_path}
- Training Embeddings: {self.embeddings_path}
- Cluster Centroids: {self.cluster_centroids_path}
- Model Metadata: {self.model_metadata_path}

Metadata:
{stats['model_metadata']}
"""

            with open(output_path, "w") as f:
                f.write(info)

            log.info(f"Model information exported to {output_path}")

        except Exception as e:
            log.error(f"Error exporting model information: {e}")
            raise
