# Advanced Features Documentation

## Overview

This document describes the advanced features implemented in the ML Hybrid Theme Analysis system, including hyperparameter tuning with Optuna and model persistence for unsupervised clustering.

## 🎯 Hyperparameter Tuning with Optuna

### What is Optuna?

Optuna is a hyperparameter optimization framework that provides efficient algorithms for finding optimal hyperparameters. In our system, it's used to automatically tune UMAP and HDBSCAN parameters for better clustering performance.

### Configuration

Enable hyperparameter tuning in `config/credentials.yaml`:

```yaml
clustering:
  hyperparameter_tuning:
    enabled: true
    n_trials: 100
    study_name: "clustering_optimization"
    save_optimization_results: true
    optimization_results_path: "models/optimization_results.joblib"
```

### Parameters Being Optimized

#### UMAP Parameters:
- `n_neighbors`: Number of neighbors for local manifold approximation (10-50)
- `min_dist`: Minimum distance between points (0.01-0.3)
- `n_components`: Output dimensionality (2-10)

#### HDBSCAN Parameters:
- `min_cluster_size`: Minimum size of clusters (3-30)
- `min_samples`: Number of samples in neighborhood (2-15)
- `cluster_selection_epsilon`: Distance threshold for cluster selection (0.05-0.3)

### Optimization Objective

The system uses a composite scoring function that considers:

1. **Silhouette Score** (40% weight): Measures cluster cohesion and separation
2. **Calinski-Harabasz Score** (20% weight): Ratio of between-cluster to within-cluster variance
3. **Cluster Balance** (20% weight): How evenly distributed cluster sizes are
4. **Noise Penalty** (10% weight): Penalizes high noise point ratios
5. **Cluster Count Score** (10% weight): Prefers moderate number of clusters (~10)

### Usage

#### Via API:
```bash
curl -X POST "http://localhost:8000/optimize-hyperparameters" \
  -H "Content-Type: application/json" \
  -d '{"embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]}'
```

#### Via Code:
```python
from src.clustering.theme_clusterer import ThemeClusterer

clusterer = ThemeClusterer()
optimization_results = clusterer.optimize_hyperparameters(embeddings)
print(f"Best score: {optimization_results['best_score']}")
print(f"Best parameters: {optimization_results['best_params']}")
```

### Optimization Results

The system saves optimization results including:
- Best hyperparameters found
- Optimization score
- Trial history
- Study statistics

## 🔄 Model Persistence

### Problem Solved

In unsupervised clustering, you typically need to retrain models for new data. Our persistence system allows you to:
- Save trained UMAP and HDBSCAN models
- Use saved models for inference on new data
- Avoid retraining when data distribution is similar
- Detect when retraining is necessary due to data drift

### Configuration

Enable model persistence in `config/credentials.yaml`:

```yaml
clustering:
  model_persistence:
    enabled: true
    model_dir: "models/clustering"
    retrain_threshold: 0.1
    drift_detection: true
```

### Saved Model Components

1. **UMAP Model**: Trained dimensionality reduction model
2. **HDBSCAN Model**: Trained clustering model
3. **Training Embeddings**: Original embeddings used for training
4. **Cluster Centroids**: Coordinates of cluster centers
5. **Model Metadata**: Training information and statistics

### Model Files Structure

```
models/clustering/
├── umap_model.joblib          # Trained UMAP model
├── hdbscan_model.joblib       # Trained HDBSCAN model
├── training_embeddings.joblib # Original training embeddings
├── cluster_centroids.joblib   # Cluster centroid coordinates
└── model_metadata.joblib      # Model metadata and statistics
```

### Usage

#### Automatic Persistence (Recommended):
```python
# Models are automatically saved during training
results = clusterer.fit_clusters_with_persistence(embeddings)

# For new data, models are automatically loaded if available
new_results = clusterer.fit_clusters_with_persistence(new_embeddings)
```

#### Manual Model Management:
```python
# Force retraining
results = clusterer.fit_clusters_with_persistence(embeddings, force_retrain=True)

# Predict clusters for new data
prediction_results = clusterer.predict_clusters_for_new_data(new_embeddings)

# Check for data drift
drift_analysis = clusterer.check_model_drift(new_embeddings)
```

### Data Drift Detection

The system automatically detects when new data differs significantly from training data:

```python
drift_analysis = clusterer.check_model_drift(new_embeddings)

# Returns:
{
    "drift_score": 0.15,           # 0-1, higher = more drift
    "threshold": 0.1,              # Retraining threshold
    "needs_retraining": True,      # Whether retraining is recommended
    "drift_severity": "Medium"     # Low/Medium/High
}
```

## 🚀 API Endpoints

### Hyperparameter Optimization

```bash
POST /optimize-hyperparameters
Content-Type: application/json

{
    "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
}

Response:
{
    "success": true,
    "optimization_results": {
        "best_params": {...},
        "best_score": 0.85,
        "n_trials": 100,
        "study": {...}
    }
}
```

### Cluster Prediction

```bash
POST /predict-clusters
Content-Type: application/json

{
    "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
}

Response:
{
    "success": true,
    "prediction_results": {
        "embeddings_2d": [[x1, y1], [x2, y2], ...],
        "cluster_labels": [0, 1, -1, 2, ...],
        "statistics": {...},
        "prediction_type": "inference"
    }
}
```

### Drift Detection

```bash
POST /check-drift
Content-Type: application/json

{
    "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
}

Response:
{
    "success": true,
    "drift_analysis": {
        "drift_score": 0.15,
        "threshold": 0.1,
        "needs_retraining": true,
        "drift_severity": "Medium"
    }
}
```

### Model Information

```bash
GET /model-info

Response:
{
    "success": true,
    "model_info": {
        "n_training_samples": 1000,
        "n_clusters": 8,
        "cluster_ids": [0, 1, 2, 3, 4, 5, 6, 7],
        "model_metadata": {...}
    }
}
```

### Force Retraining

```bash
POST /retrain-models
Content-Type: application/json

{
    "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
}

Response:
{
    "success": true,
    "retraining_results": {
        "embeddings_2d": [[x1, y1], [x2, y2], ...],
        "cluster_labels": [0, 1, 2, ...],
        "cluster_centroids": {...},
        "statistics": {...},
        "model_persisted": true
    }
}
```

## 📊 Performance Benefits

### Without Persistence (Traditional):
```
New data → Retrain UMAP → Retrain HDBSCAN → Get results
Time: ~30-60 seconds per analysis
```

### With Persistence (Our System):
```
New data → Load models → Transform → Predict → Get results
Time: ~2-5 seconds per analysis
```

### With Hyperparameter Tuning:
- **First run**: 100 trials × ~10 seconds = ~16 minutes
- **Subsequent runs**: Use optimized parameters, ~30-60 seconds
- **Long-term benefit**: Better clustering quality, more interpretable themes

## 🔧 Best Practices

### 1. Hyperparameter Tuning

- **For small datasets** (< 1k samples): Use 50-100 trials
- **For large datasets** (> 10k samples): Use 100-200 trials
- **For production**: Run optimization once, then use saved parameters
- **Monitor optimization**: Check logs for convergence and best scores

### 2. Model Persistence

- **Initial training**: Always train on representative data
- **Drift monitoring**: Check drift regularly (weekly/monthly)
- **Retraining triggers**: Retrain when drift_score > 0.2
- **Model versioning**: Keep multiple model versions for rollback

### 3. Configuration

```yaml
# For development/testing
clustering:
  hyperparameter_tuning:
    enabled: true
    n_trials: 50  # Fewer trials for faster iteration

# For production
clustering:
  hyperparameter_tuning:
    enabled: false  # Disable after optimization
  model_persistence:
    enabled: true
    retrain_threshold: 0.15  # More conservative
```

## 🐛 Troubleshooting

### Common Issues

1. **Optimization takes too long**
   - Reduce `n_trials` in configuration
   - Use smaller sample of data for optimization

2. **Models not loading**
   - Check if models exist in `models/clustering/`
   - Verify file permissions
   - Check logs for specific error messages

3. **Poor prediction quality**
   - Check drift analysis
   - Consider retraining with more data
   - Verify new data format matches training data

4. **Memory issues during optimization**
   - Reduce batch size
   - Use smaller subset of data for optimization
   - Increase system memory

### Debug Commands

```python
# Check if models exist
clusterer.model_persistence._models_exist()

# Get model statistics
clusterer.get_model_info()

# Check drift manually
drift = clusterer.check_model_drift(new_embeddings)
print(f"Drift score: {drift['drift_score']}")

# Force retraining
results = clusterer.fit_clusters_with_persistence(embeddings, force_retrain=True)
```

## 📈 Monitoring and Logging

The system provides comprehensive logging for all operations:

```python
# Optimization logs
2024-01-15 10:30:15 | INFO | hyperparameter_tuner:optimize - Starting hyperparameter optimization with 100 trials
2024-01-15 10:32:45 | INFO | hyperparameter_tuner:optimize - Optimization completed. Best score: 0.8234

# Persistence logs
2024-01-15 10:33:00 | INFO | model_persistence:save_models - UMAP model saved to models/clustering/umap_model.joblib
2024-01-15 10:33:01 | INFO | model_persistence:load_models - Using existing trained models

# Drift detection logs
2024-01-15 10:35:00 | INFO | model_persistence:update_models - Significant drift detected (0.25), triggering model update
```

## 🔮 Future Enhancements

1. **Distributed Optimization**: Use Optuna's distributed optimization for large datasets
2. **AutoML Integration**: Integrate with AutoML frameworks for end-to-end optimization
3. **Model Versioning**: Implement proper model versioning and rollback capabilities
4. **Real-time Drift Detection**: Continuous monitoring of data drift
5. **A/B Testing**: Compare different model versions in production
6. **Model Explainability**: Add SHAP or LIME for model interpretability

## 📚 References

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/)
- [Joblib Documentation](https://joblib.readthedocs.io/) 