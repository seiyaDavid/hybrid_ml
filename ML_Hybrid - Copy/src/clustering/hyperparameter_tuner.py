"""
Advanced hyperparameter tuning module for UMAP and HDBSCAN clustering.

This module provides automated hyperparameter optimization using Optuna
for unsupervised clustering algorithms.
"""

import optuna
import numpy as np
import joblib
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
from loguru import logger

from ..utils.config import config
from ..utils.logger import get_logger

# Initialize logger
log = get_logger(__name__)


class ClusteringHyperparameterTuner:
    """
    Advanced hyperparameter tuner for UMAP and HDBSCAN clustering.

    This class provides automated optimization of clustering hyperparameters
    using Optuna with multiple objective functions for unsupervised learning.
    """

    def __init__(self, embeddings: np.ndarray, n_trials: int = 100):
        """
        Initialize the hyperparameter tuner.

        Args:
            embeddings: Input embeddings for clustering
            n_trials: Number of optimization trials
        """
        self.embeddings = embeddings
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = None
        self.study = None

        log.info(f"Initialized hyperparameter tuner with {len(embeddings)} embeddings")

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial: Optuna trial object

        Returns:
            Optimization score (higher is better)
        """
        try:
            # UMAP parameters
            n_neighbors = trial.suggest_int("umap_n_neighbors", 10, 50)
            min_dist = trial.suggest_float("umap_min_dist", 0.01, 0.3)
            n_components = trial.suggest_int("umap_n_components", 2, 10)

            # HDBSCAN parameters
            min_cluster_size = trial.suggest_int("hdbscan_min_cluster_size", 3, 30)
            min_samples = trial.suggest_int("hdbscan_min_samples", 2, 15)
            cluster_selection_epsilon = trial.suggest_float(
                "hdbscan_epsilon", 0.05, 0.3
            )

            # Apply UMAP
            umap_reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                metric="cosine",
                random_state=42,
            )

            embeddings_2d = umap_reducer.fit_transform(self.embeddings)

            # Apply HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=cluster_selection_epsilon,
                metric="euclidean",
            )

            cluster_labels = clusterer.fit_predict(embeddings_2d)

            # Calculate multiple metrics
            score = self._calculate_clustering_score(embeddings_2d, cluster_labels)

            # Log trial results
            log.debug(
                f"Trial {trial.number}: Score={score:.4f}, Clusters={len(set(cluster_labels))-1}"
            )

            return score

        except Exception as e:
            log.error(f"Error in trial {trial.number}: {e}")
            return -1.0  # Return worst score on error

    def _calculate_clustering_score(
        self, embeddings_2d: np.ndarray, cluster_labels: np.ndarray
    ) -> float:
        """
        Calculate comprehensive clustering score.

        Args:
            embeddings_2d: 2D embeddings from UMAP
            cluster_labels: Cluster labels from HDBSCAN

        Returns:
            Combined clustering score
        """
        try:
            # Filter out noise points (-1) for metric calculation
            valid_mask = cluster_labels != -1
            valid_embeddings = embeddings_2d[valid_mask]
            valid_labels = cluster_labels[valid_mask]

            if len(valid_labels) < 2 or len(set(valid_labels)) < 2:
                return 0.0

            # Calculate multiple metrics
            silhouette = silhouette_score(valid_embeddings, valid_labels)
            calinski_harabasz = calinski_harabasz_score(valid_embeddings, valid_labels)

            # Calculate cluster balance
            unique_labels, counts = np.unique(valid_labels, return_counts=True)
            cluster_balance = (
                1.0 - np.std(counts) / np.mean(counts) if len(counts) > 1 else 0.0
            )

            # Calculate noise ratio (lower is better)
            noise_ratio = np.sum(cluster_labels == -1) / len(cluster_labels)
            noise_penalty = 1.0 - noise_ratio

            # Calculate number of clusters (prefer moderate number)
            n_clusters = len(unique_labels)
            cluster_count_score = 1.0 / (
                1.0 + abs(n_clusters - 10)
            )  # Prefer around 10 clusters

            # Combined score (weighted average)
            score = (
                0.4 * silhouette
                + 0.2 * calinski_harabasz / 1000  # Normalize
                + 0.2 * cluster_balance
                + 0.1 * noise_penalty
                + 0.1 * cluster_count_score
            )

            return score

        except Exception as e:
            log.error(f"Error calculating clustering score: {e}")
            return 0.0

    def optimize(self, study_name: str = "clustering_optimization") -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            study_name: Name for the Optuna study

        Returns:
            Dictionary containing best parameters and results
        """
        log.info(f"Starting hyperparameter optimization with {self.n_trials} trials")

        # Create Optuna study
        self.study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=None,  # In-memory storage
        )

        # Run optimization
        self.study.optimize(self.objective, n_trials=self.n_trials)

        # Get best results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        log.info(f"Optimization completed. Best score: {self.best_score:.4f}")
        log.info(f"Best parameters: {self.best_params}")

        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "study": self.study,
        }

    def get_optimized_clusterer(self) -> Tuple[umap.UMAP, hdbscan.HDBSCAN]:
        """
        Get optimized UMAP and HDBSCAN models with best parameters.

        Returns:
            Tuple of (optimized_umap, optimized_hdbscan)
        """
        if self.best_params is None:
            raise ValueError("Must run optimize() before getting optimized clusterer")

        # Create optimized UMAP
        optimized_umap = umap.UMAP(
            n_neighbors=self.best_params["umap_n_neighbors"],
            min_dist=self.best_params["umap_min_dist"],
            n_components=self.best_params["umap_n_components"],
            metric="cosine",
            random_state=42,
        )

        # Create optimized HDBSCAN
        optimized_hdbscan = hdbscan.HDBSCAN(
            min_cluster_size=self.best_params["hdbscan_min_cluster_size"],
            min_samples=self.best_params["hdbscan_min_samples"],
            cluster_selection_epsilon=self.best_params["hdbscan_epsilon"],
            metric="euclidean",
        )

        return optimized_umap, optimized_hdbscan

    def save_optimization_results(self, filepath: str) -> None:
        """
        Save optimization results to file.

        Args:
            filepath: Path to save results
        """
        if self.study is None:
            raise ValueError("No optimization results to save")

        results = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "study": self.study,
        }

        joblib.dump(results, filepath)
        log.info(f"Optimization results saved to {filepath}")

    def load_optimization_results(self, filepath: str) -> None:
        """
        Load optimization results from file.

        Args:
            filepath: Path to load results from
        """
        results = joblib.load(filepath)
        self.best_params = results["best_params"]
        self.best_score = results["best_score"]
        self.study = results["study"]

        log.info(f"Optimization results loaded from {filepath}")

    def get_optimization_report(self) -> str:
        """
        Generate optimization report.

        Returns:
            Formatted optimization report
        """
        if self.study is None:
            return "No optimization results available"

        report = f"""
Hyperparameter Optimization Report
================================

Optimization Summary:
- Number of Trials: {self.n_trials}
- Best Score: {self.best_score:.4f}
- Optimization Direction: Maximize

Best Parameters:
"""

        for param, value in self.best_params.items():
            report += f"- {param}: {value}\n"

        # Add trial statistics
        trials_df = self.study.trials_dataframe()
        report += f"""
Trial Statistics:
- Mean Score: {trials_df['value'].mean():.4f}
- Std Score: {trials_df['value'].std():.4f}
- Min Score: {trials_df['value'].min():.4f}
- Max Score: {trials_df['value'].max():.4f}
"""

        return report
