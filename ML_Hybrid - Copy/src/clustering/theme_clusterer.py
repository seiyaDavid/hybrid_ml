"""
Theme clustering module for the ML Hybrid Theme Analysis system.

This module provides functionality to cluster text embeddings using UMAP
for dimensionality reduction and HDBSCAN for clustering, enabling
automatic theme discovery from issue summaries.
"""

import numpy as np
import umap
import hdbscan
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import silhouette_score
from loguru import logger
from tqdm import tqdm

from ..utils.config import config
from ..utils.logger import get_logger
from .hyperparameter_tuner import ClusteringHyperparameterTuner
from .model_persistence import ClusteringModelPersistence

# Initialize logger
log = get_logger(__name__)


class ThemeClusterer:
    """
    Clusters text embeddings to discover themes using UMAP and HDBSCAN.

    This class provides functionality to reduce dimensionality of embeddings
    using UMAP and then cluster them using HDBSCAN to discover natural
    themes in the data.
    """

    def __init__(self):
        """Initialize the theme clusterer with configuration."""
        self.config = config.get_clustering_config()
        self.umap_config = self.config.get("umap", {})
        self.hdbscan_config = self.config.get("hdbscan", {})
        self.tuning_config = self.config.get("hyperparameter_tuning", {})
        self.persistence_config = self.config.get("model_persistence", {})

        # Initialize UMAP
        self.umap_reducer = umap.UMAP(
            n_neighbors=self.umap_config.get("n_neighbors", 15),
            n_components=self.umap_config.get("n_components", 2),
            min_dist=self.umap_config.get("min_dist", 0.1),
            metric=self.umap_config.get("metric", "cosine"),
            random_state=42,
        )

        # Initialize HDBSCAN
        self.hdbscan_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.hdbscan_config.get("min_cluster_size", 5),
            min_samples=self.hdbscan_config.get("min_samples", 3),
            cluster_selection_epsilon=self.hdbscan_config.get(
                "cluster_selection_epsilon", 0.1
            ),
            metric="euclidean",
        )

        # Initialize model persistence
        self.model_persistence = ClusteringModelPersistence(
            model_dir=self.persistence_config.get("model_dir", "models/clustering")
        )

        self.embeddings_2d = None
        self.cluster_labels = None
        self.cluster_centroids = None
        self.training_embeddings = None

        log.info("ThemeClusterer initialized successfully")

    def fit_clusters(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Fit clustering model to embeddings.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Dictionary containing clustering results and statistics
        """
        log.info(f"Fitting clustering model to {len(embeddings)} embeddings")

        try:
            # Convert to numpy array
            embeddings_array = np.array(embeddings)

            # Apply UMAP dimensionality reduction
            log.info("Applying UMAP dimensionality reduction")
            self.embeddings_2d = self.umap_reducer.fit_transform(embeddings_array)

            # Apply HDBSCAN clustering
            log.info("Applying HDBSCAN clustering")
            self.cluster_labels = self.hdbscan_clusterer.fit_predict(self.embeddings_2d)

            # Calculate cluster centroids
            self.cluster_centroids = self._calculate_centroids()

            # Generate clustering statistics
            statistics = self._generate_clustering_statistics()

            log.info(
                f"Clustering completed. Found {statistics['num_clusters']} clusters"
            )
            log.info(f"Clustering statistics: {statistics}")

            return {
                "embeddings_2d": self.embeddings_2d.tolist(),
                "cluster_labels": self.cluster_labels.tolist(),
                "cluster_centroids": self.cluster_centroids,
                "statistics": statistics,
            }

        except Exception as e:
            log.error(f"Error in clustering: {e}")
            raise

    def _calculate_centroids(self) -> Dict[int, List[float]]:
        """
        Calculate centroids for each cluster.

        Returns:
            Dictionary mapping cluster IDs to centroid coordinates
        """
        centroids = {}

        unique_labels = set(self.cluster_labels)

        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue

            # Get points belonging to this cluster
            cluster_points = self.embeddings_2d[self.cluster_labels == label]

            # Calculate centroid
            centroid = np.mean(cluster_points, axis=0)
            centroids[label] = centroid.tolist()

        return centroids

    def _generate_clustering_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive clustering statistics.

        Returns:
            Dictionary containing clustering statistics
        """
        unique_labels = set(self.cluster_labels)
        num_clusters = len([label for label in unique_labels if label != -1])
        num_noise = list(self.cluster_labels).count(-1)

        # Calculate cluster sizes
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[label] = int(np.sum(self.cluster_labels == label))

        # Calculate silhouette score (excluding noise points)
        valid_indices = self.cluster_labels != -1
        if (
            np.sum(valid_indices) > 1
            and len(set(self.cluster_labels[valid_indices])) > 1
        ):
            try:
                silhouette_avg = silhouette_score(
                    self.embeddings_2d[valid_indices],
                    self.cluster_labels[valid_indices],
                )
            except:
                silhouette_avg = 0.0
        else:
            silhouette_avg = 0.0

        return {
            "num_clusters": num_clusters,
            "num_noise_points": num_noise,
            "total_points": len(self.cluster_labels),
            "cluster_sizes": cluster_sizes,
            "silhouette_score": silhouette_avg,
            "clustering_quality": self._assess_clustering_quality(
                num_clusters, num_noise, silhouette_avg
            ),
        }

    def _assess_clustering_quality(
        self, num_clusters: int, num_noise: int, silhouette_score: float
    ) -> str:
        """
        Assess the quality of clustering results.

        Args:
            num_clusters: Number of clusters found
            num_noise: Number of noise points
            silhouette_score: Average silhouette score

        Returns:
            Quality assessment string
        """
        total_points = len(self.cluster_labels)
        noise_ratio = num_noise / total_points if total_points > 0 else 0

        if num_clusters == 0:
            return "Poor - No clusters found"
        elif noise_ratio > 0.5:
            return "Poor - Too many noise points"
        elif silhouette_score < 0.1:
            return "Fair - Low silhouette score"
        elif silhouette_score < 0.3:
            return "Good - Moderate silhouette score"
        else:
            return "Excellent - High silhouette score"

    def get_cluster_samples(
        self, texts: List[str], max_samples: int = 10
    ) -> Dict[int, List[str]]:
        """
        Get sample texts for each cluster.

        Args:
            texts: List of original texts
            max_samples: Maximum number of samples per cluster

        Returns:
            Dictionary mapping cluster IDs to sample texts
        """
        cluster_samples = {}

        for label in set(self.cluster_labels):
            if label == -1:  # Skip noise points
                continue

            # Get indices of points in this cluster
            cluster_indices = np.where(self.cluster_labels == label)[0]

            # Get sample texts
            samples = [texts[i] for i in cluster_indices[:max_samples]]
            cluster_samples[label] = samples

        return cluster_samples

    def get_cluster_summary(self, cluster_id: int, texts: List[str]) -> Dict[str, Any]:
        """
        Get detailed summary for a specific cluster.

        Args:
            cluster_id: ID of the cluster
            texts: List of original texts

        Returns:
            Dictionary containing cluster summary
        """
        if cluster_id == -1:
            return {"error": "Cannot summarize noise cluster"}

        # Get cluster points
        cluster_indices = np.where(self.cluster_labels == cluster_id)[0]
        cluster_texts = [texts[i] for i in cluster_indices]
        cluster_embeddings_2d = self.embeddings_2d[cluster_indices]

        # Calculate cluster statistics
        centroid = self.cluster_centroids.get(cluster_id, [0, 0])

        # Calculate average distance to centroid
        distances = np.linalg.norm(cluster_embeddings_2d - centroid, axis=1)
        avg_distance = np.mean(distances)

        return {
            "cluster_id": cluster_id,
            "size": len(cluster_texts),
            "centroid": centroid,
            "avg_distance_to_centroid": float(avg_distance),
            "sample_texts": cluster_texts[:5],  # First 5 samples
            "text_count": len(cluster_texts),
        }

    def find_similar_clusters(
        self, target_cluster_id: int, similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Find clusters similar to a target cluster.

        Args:
            target_cluster_id: ID of the target cluster
            similarity_threshold: Minimum similarity threshold

        Returns:
            List of similar clusters with similarity scores
        """
        if target_cluster_id not in self.cluster_centroids:
            return []

        target_centroid = np.array(self.cluster_centroids[target_cluster_id])
        similar_clusters = []

        for cluster_id, centroid in self.cluster_centroids.items():
            if cluster_id == target_cluster_id:
                continue

            centroid_array = np.array(centroid)
            similarity = self._compute_centroid_similarity(
                target_centroid, centroid_array
            )

            if similarity >= similarity_threshold:
                similar_clusters.append(
                    {"cluster_id": cluster_id, "similarity": similarity}
                )

        # Sort by similarity (descending)
        similar_clusters.sort(key=lambda x: x["similarity"], reverse=True)

        return similar_clusters

    def _compute_centroid_similarity(
        self, centroid1: np.ndarray, centroid2: np.ndarray
    ) -> float:
        """
        Compute similarity between two cluster centroids.

        Args:
            centroid1: First centroid
            centroid2: Second centroid

        Returns:
            Similarity score
        """
        try:
            # Compute cosine similarity
            dot_product = np.dot(centroid1, centroid2)
            norm1 = np.linalg.norm(centroid1)
            norm2 = np.linalg.norm(centroid2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            log.error(f"Error computing centroid similarity: {e}")
            return 0.0

    def get_clustering_report(self, statistics: Dict[str, Any]) -> str:
        """
        Generate a human-readable clustering report.

        Args:
            statistics: Clustering statistics

        Returns:
            Formatted report string
        """
        report = f"""
Theme Clustering Report
======================

Total Points: {statistics['total_points']}
Number of Clusters: {statistics['num_clusters']}
Noise Points: {statistics['num_noise_points']} ({statistics['num_noise_points']/statistics['total_points']*100:.1f}%)
Silhouette Score: {statistics['silhouette_score']:.3f}
Clustering Quality: {statistics['clustering_quality']}

Cluster Sizes:
"""

        for cluster_id, size in sorted(statistics["cluster_sizes"].items()):
            percentage = (size / statistics["total_points"]) * 100
            report += f"  Cluster {cluster_id}: {size} points ({percentage:.1f}%)\n"

        return report

    def optimize_hyperparameters(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Dictionary containing optimization results
        """
        if not self.tuning_config.get("enabled", False):
            log.info("Hyperparameter tuning is disabled")
            return {}

        log.info("Starting hyperparameter optimization")

        try:
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings)

            # Initialize tuner
            tuner = ClusteringHyperparameterTuner(
                embeddings=embeddings_array,
                n_trials=self.tuning_config.get("n_trials", 100),
            )

            # Run optimization
            results = tuner.optimize(
                study_name=self.tuning_config.get(
                    "study_name", "clustering_optimization"
                )
            )

            # Get optimized models
            optimized_umap, optimized_hdbscan = tuner.get_optimized_clusterer()

            # Update current models
            self.umap_reducer = optimized_umap
            self.hdbscan_clusterer = optimized_hdbscan

            # Save optimization results if enabled
            if self.tuning_config.get("save_optimization_results", False):
                results_path = self.tuning_config.get(
                    "optimization_results_path", "models/optimization_results.joblib"
                )
                tuner.save_optimization_results(results_path)

            log.info(
                f"Hyperparameter optimization completed with best score: {results['best_score']:.4f}"
            )

            return results

        except Exception as e:
            log.error(f"Error in hyperparameter optimization: {e}")
            raise

    def fit_clusters_with_persistence(
        self, embeddings: List[List[float]], force_retrain: bool = False
    ) -> Dict[str, Any]:
        """
        Fit clustering model with persistence support.

        Args:
            embeddings: List of embedding vectors
            force_retrain: Force retraining even if models exist

        Returns:
            Dictionary containing clustering results and statistics
        """
        log.info(
            f"Fitting clustering model to {len(embeddings)} embeddings with persistence"
        )

        try:
            # Convert to numpy array
            embeddings_array = np.array(embeddings)
            self.training_embeddings = embeddings_array

            # Check if we should use existing models
            if not force_retrain and self.persistence_config.get("enabled", False):
                try:
                    # Try to load existing models
                    umap_model, hdbscan_model, _, cluster_centroids, metadata = (
                        self.model_persistence.load_models()
                    )

                    # Use loaded models
                    self.umap_reducer = umap_model
                    self.hdbscan_clusterer = hdbscan_model

                    log.info("Using existing trained models")

                    # Apply models to new data
                    self.embeddings_2d = self.umap_reducer.transform(embeddings_array)
                    self.cluster_labels = self.hdbscan_clusterer.fit_predict(
                        self.embeddings_2d
                    )
                    self.cluster_centroids = cluster_centroids

                except FileNotFoundError:
                    log.info("No existing models found, training new models")
                    return self._train_new_models(embeddings_array)
                except Exception as e:
                    log.warning(
                        f"Error loading existing models: {e}, training new models"
                    )
                    return self._train_new_models(embeddings_array)
            else:
                return self._train_new_models(embeddings_array)

            # Generate clustering statistics
            statistics = self._generate_clustering_statistics()

            log.info(
                f"Clustering completed. Found {statistics['num_clusters']} clusters"
            )

            return {
                "embeddings_2d": self.embeddings_2d.tolist(),
                "cluster_labels": self.cluster_labels.tolist(),
                "cluster_centroids": self.cluster_centroids,
                "statistics": statistics,
                "model_persisted": True,
            }

        except Exception as e:
            log.error(f"Error in clustering with persistence: {e}")
            raise

    def _train_new_models(self, embeddings_array: np.ndarray) -> Dict[str, Any]:
        """
        Train new clustering models and save them.

        Args:
            embeddings_array: Numpy array of embeddings

        Returns:
            Dictionary containing clustering results
        """
        log.info("Training new clustering models")

        # Apply UMAP dimensionality reduction
        log.info("Applying UMAP dimensionality reduction")
        self.embeddings_2d = self.umap_reducer.fit_transform(embeddings_array)

        # Apply HDBSCAN clustering
        log.info("Applying HDBSCAN clustering")
        self.cluster_labels = self.hdbscan_clusterer.fit_predict(self.embeddings_2d)

        # Calculate cluster centroids
        self.cluster_centroids = self._calculate_centroids()

        # Save models if persistence is enabled
        if self.persistence_config.get("enabled", False):
            try:
                metadata = {
                    "training_samples": len(embeddings_array),
                    "n_clusters": len(
                        [label for label in set(self.cluster_labels) if label != -1]
                    ),
                    "training_date": str(np.datetime64("now")),
                    "model_version": "1.0",
                }

                self.model_persistence.save_models(
                    umap_model=self.umap_reducer,
                    hdbscan_model=self.hdbscan_clusterer,
                    training_embeddings=embeddings_array,
                    cluster_centroids=self.cluster_centroids,
                    metadata=metadata,
                )

                log.info("Models saved successfully")

            except Exception as e:
                log.error(f"Error saving models: {e}")

        # Generate clustering statistics
        statistics = self._generate_clustering_statistics()

        log.info(f"New models trained. Found {statistics['num_clusters']} clusters")

        return {
            "embeddings_2d": self.embeddings_2d.tolist(),
            "cluster_labels": self.cluster_labels.tolist(),
            "cluster_centroids": self.cluster_centroids,
            "statistics": statistics,
            "model_persisted": True,
        }

    def predict_clusters_for_new_data(
        self, new_embeddings: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Predict clusters for new data using trained models.

        Args:
            new_embeddings: New embedding vectors

        Returns:
            Dictionary containing prediction results
        """
        log.info(f"Predicting clusters for {len(new_embeddings)} new embeddings")

        try:
            new_embeddings_array = np.array(new_embeddings)

            # Use model persistence to predict
            cluster_labels, embeddings_2d = self.model_persistence.predict_clusters(
                new_embeddings_array
            )

            # Calculate statistics for new predictions
            unique_labels = set(cluster_labels)
            num_clusters = len([label for label in unique_labels if label != -1])
            num_noise = list(cluster_labels).count(-1)

            statistics = {
                "num_clusters": num_clusters,
                "num_noise_points": num_noise,
                "total_points": len(cluster_labels),
                "prediction_quality": (
                    "Good" if num_noise / len(cluster_labels) < 0.3 else "Fair"
                ),
            }

            return {
                "embeddings_2d": embeddings_2d.tolist(),
                "cluster_labels": cluster_labels.tolist(),
                "statistics": statistics,
                "prediction_type": "inference",
            }

        except Exception as e:
            log.error(f"Error predicting clusters for new data: {e}")
            raise

    def check_model_drift(self, new_embeddings: List[List[float]]) -> Dict[str, Any]:
        """
        Check for data drift in new embeddings compared to training data.

        Args:
            new_embeddings: New embedding vectors

        Returns:
            Dictionary containing drift analysis
        """
        if not self.persistence_config.get("drift_detection", False):
            return {"drift_detection_disabled": True}

        try:
            new_embeddings_array = np.array(new_embeddings)

            # Load training embeddings
            _, _, training_embeddings, _, _ = self.model_persistence.load_models()

            # Calculate drift score
            drift_score = self.model_persistence._calculate_drift_score(
                new_embeddings_array, training_embeddings
            )

            threshold = self.persistence_config.get("retrain_threshold", 0.1)
            needs_retraining = drift_score > threshold

            return {
                "drift_score": drift_score,
                "threshold": threshold,
                "needs_retraining": needs_retraining,
                "drift_severity": (
                    "High"
                    if drift_score > 0.2
                    else "Medium" if drift_score > 0.1 else "Low"
                ),
            }

        except Exception as e:
            log.error(f"Error checking model drift: {e}")
            return {"error": str(e)}

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current clustering model.

        Returns:
            Dictionary containing model information
        """
        try:
            if self.persistence_config.get("enabled", False):
                return self.model_persistence.get_cluster_statistics()
            else:
                return {"model_persistence_disabled": True}

        except Exception as e:
            log.error(f"Error getting model info: {e}")
            return {"error": str(e)}
