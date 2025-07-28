"""
FastAPI backend for the ML Hybrid Theme Analysis system.

This module provides REST API endpoints for file upload, analysis,
theme exploration, and chat functionality.
"""

import os
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from loguru import logger

# Add the src directory to Python path
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from src.core.analysis_pipeline import AnalysisPipeline
from src.utils.config import config
from src.utils.logger import get_logger

# Initialize logger
log = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ML Hybrid Theme Analysis API",
    description="API for analyzing themes in CSV files using ML and LLM",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analysis pipeline
pipeline = None


# Pydantic models
class AnalysisRequest(BaseModel):
    file_path: str
    output_path: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None


class AnalysisResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


# Global storage for analysis results
analysis_results = {}


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global pipeline
    try:
        pipeline = AnalysisPipeline()
        log.info("FastAPI application started successfully")
    except Exception as e:
        log.error(f"Failed to initialize application: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "ML Hybrid Theme Analysis API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "pipeline_ready": pipeline is not None}


@app.post("/upload", response_model=AnalysisResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV file for analysis.

    Args:
        file: CSV file to upload

    Returns:
        Analysis response with file information
    """
    try:
        # Validate file type
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        # Create uploads directory
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)

        # Save uploaded file
        file_path = uploads_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        log.info(f"File uploaded: {file_path}")

        return AnalysisResponse(
            success=True,
            message=f"File uploaded successfully: {file.filename}",
            data={
                "file_path": str(file_path),
                "file_name": file.filename,
                "file_size": file_path.stat().st_size,
            },
        )

    except Exception as e:
        log.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_file(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Analyze a CSV file for themes.

    Args:
        request: Analysis request with file path
        background_tasks: Background tasks for async processing

    Returns:
        Analysis response with results
    """
    try:
        if not pipeline:
            raise HTTPException(
                status_code=500, detail="Analysis pipeline not initialized"
            )

        # Run analysis
        results = pipeline.run_analysis(request.file_path, request.output_path)

        if results["success"]:
            # Store results for later access
            analysis_id = f"analysis_{len(analysis_results)}"
            analysis_results[analysis_id] = results

            return AnalysisResponse(
                success=True,
                message="Analysis completed successfully",
                data={
                    "analysis_id": analysis_id,
                    "execution_time": results["execution_time"],
                    "summary": {
                        "total_themes": len(results["theme_analysis"]),
                        "total_summaries": results["preprocessing_results"][
                            "statistics"
                        ]["processed_count"],
                        "clustering_quality": results["clustering_results"][
                            "statistics"
                        ]["clustering_quality"],
                    },
                },
            )
        else:
            return AnalysisResponse(
                success=False,
                message=f"Analysis failed: {results.get('error', 'Unknown error')}",
                data=None,
            )

    except Exception as e:
        log.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{analysis_id}")
async def get_analysis_results(analysis_id: str):
    """
    Get analysis results by ID.

    Args:
        analysis_id: Analysis ID

    Returns:
        Analysis results
    """
    try:
        if analysis_id not in analysis_results:
            raise HTTPException(status_code=404, detail="Analysis not found")

        results = analysis_results[analysis_id]

        # Return a simplified version for the frontend
        return {
            "success": True,
            "analysis_id": analysis_id,
            "theme_analysis": results["theme_analysis"],
            "clustering_results": {
                "embeddings_2d": results["clustering_results"]["embeddings_2d"],
                "cluster_labels": results["clustering_results"]["cluster_labels"],
                "statistics": results["clustering_results"]["statistics"],
            },
            "preprocessing_results": results["preprocessing_results"]["statistics"],
            "reports": results["reports"],
        }

    except Exception as e:
        log.error(f"Error retrieving results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/themes/{analysis_id}")
async def get_themes(analysis_id: str):
    """
    Get themes for an analysis.

    Args:
        analysis_id: Analysis ID

    Returns:
        List of themes
    """
    try:
        if analysis_id not in analysis_results:
            raise HTTPException(status_code=404, detail="Analysis not found")

        results = analysis_results[analysis_id]
        themes = []

        for cluster_id, theme in results["theme_analysis"].items():
            themes.append(
                {
                    "cluster_id": cluster_id,
                    "name": theme["name"],
                    "description": theme["description"],
                    "sample_count": theme["sample_count"],
                    "data_quality_issues": theme["data_quality_issues"],
                    "data_quality_percentage": theme["data_quality_percentage"],
                    "samples": theme["samples"],
                }
            )

        return {"success": True, "themes": themes}

    except Exception as e:
        log.error(f"Error retrieving themes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clusters/{analysis_id}")
async def get_clusters(analysis_id: str):
    """
    Get clustering results for an analysis.

    Args:
        analysis_id: Analysis ID

    Returns:
        Clustering results
    """
    try:
        if analysis_id not in analysis_results:
            raise HTTPException(status_code=404, detail="Analysis not found")

        results = analysis_results[analysis_id]

        return {
            "success": True,
            "embeddings_2d": results["clustering_results"]["embeddings_2d"],
            "cluster_labels": results["clustering_results"]["cluster_labels"],
            "statistics": results["clustering_results"]["statistics"],
        }

    except Exception as e:
        log.error(f"Error retrieving clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_with_themes(request: ChatRequest):
    """
    Chat with themes using LangChain.

    Args:
        request: Chat request with message and context

    Returns:
        Chat response
    """
    try:
        # For now, return a simple response
        # In a full implementation, this would use LangChain's ConversationalRetrievalChain

        response = {
            "success": True,
            "message": f"I understand you're asking about: {request.message}. This is a placeholder response. In a full implementation, this would use LangChain to provide detailed theme analysis.",
            "context": request.context,
        }

        return response

    except Exception as e:
        log.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare")
async def compare_files(file_paths: List[str]):
    """
    Compare themes across multiple files.

    Args:
        file_paths: List of file paths to compare

    Returns:
        Comparison results
    """
    try:
        if not pipeline:
            raise HTTPException(
                status_code=500, detail="Analysis pipeline not initialized"
            )

        results = pipeline.compare_files(file_paths)

        return {
            "success": results["success"],
            "comparison_results": results.get("comparison_results", {}),
            "file_results": results.get("file_results", {}),
        }

    except Exception as e:
        log.error(f"File comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reports/{analysis_id}")
async def get_reports(analysis_id: str):
    """
    Get analysis reports.

    Args:
        analysis_id: Analysis ID

    Returns:
        Analysis reports
    """
    try:
        if analysis_id not in analysis_results:
            raise HTTPException(status_code=404, detail="Analysis not found")

        results = analysis_results[analysis_id]

        return {"success": True, "reports": results["reports"]}

    except Exception as e:
        log.error(f"Error retrieving reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/results/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """
    Delete analysis results.

    Args:
        analysis_id: Analysis ID

    Returns:
        Deletion confirmation
    """
    try:
        if analysis_id in analysis_results:
            del analysis_results[analysis_id]
            return {"success": True, "message": f"Analysis {analysis_id} deleted"}
        else:
            raise HTTPException(status_code=404, detail="Analysis not found")

    except Exception as e:
        log.error(f"Error deleting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize-hyperparameters")
async def optimize_hyperparameters(optimization_request: dict):
    """Optimize clustering hyperparameters using Optuna."""
    try:
        log.info("Starting hyperparameter optimization")

        # Extract embeddings from request
        embeddings = optimization_request.get("embeddings", [])
        if not embeddings:
            return {"success": False, "error": "No embeddings provided"}

        # Run hyperparameter optimization
        results = pipeline.theme_clusterer.optimize_hyperparameters(embeddings)

        log.info(
            f"Hyperparameter optimization completed with score: {results.get('best_score', 0):.4f}"
        )

        return {"success": True, "optimization_results": results}

    except Exception as e:
        log.error(f"Error in hyperparameter optimization: {e}")
        return {"success": False, "error": str(e)}


@app.post("/predict-clusters")
async def predict_clusters(prediction_request: dict):
    """Predict clusters for new data using trained models."""
    try:
        log.info("Starting cluster prediction for new data")

        # Extract embeddings from request
        new_embeddings = prediction_request.get("embeddings", [])
        if not new_embeddings:
            return {"success": False, "error": "No embeddings provided"}

        # Predict clusters
        results = pipeline.theme_clusterer.predict_clusters_for_new_data(new_embeddings)

        log.info(f"Cluster prediction completed for {len(new_embeddings)} embeddings")

        return {"success": True, "prediction_results": results}

    except Exception as e:
        log.error(f"Error in cluster prediction: {e}")
        return {"success": False, "error": str(e)}


@app.post("/check-drift")
async def check_model_drift(drift_request: dict):
    """Check for data drift in new embeddings."""
    try:
        log.info("Checking for model drift")

        # Extract embeddings from request
        new_embeddings = drift_request.get("embeddings", [])
        if not new_embeddings:
            return {"success": False, "error": "No embeddings provided"}

        # Check drift
        drift_analysis = pipeline.theme_clusterer.check_model_drift(new_embeddings)

        log.info(f"Drift analysis completed: {drift_analysis}")

        return {"success": True, "drift_analysis": drift_analysis}

    except Exception as e:
        log.error(f"Error in drift analysis: {e}")
        return {"success": False, "error": str(e)}


@app.get("/model-info")
async def get_model_info():
    """Get information about the current clustering model."""
    try:
        log.info("Retrieving model information")

        model_info = pipeline.theme_clusterer.get_model_info()

        return {"success": True, "model_info": model_info}

    except Exception as e:
        log.error(f"Error retrieving model info: {e}")
        return {"success": False, "error": str(e)}


@app.post("/retrain-models")
async def retrain_models(retrain_request: dict):
    """Force retraining of clustering models."""
    try:
        log.info("Starting forced model retraining")

        # Extract embeddings from request
        embeddings = retrain_request.get("embeddings", [])
        if not embeddings:
            return {"success": False, "error": "No embeddings provided"}

        # Retrain models
        results = pipeline.theme_clusterer.fit_clusters_with_persistence(
            embeddings, force_retrain=True
        )

        log.info("Model retraining completed successfully")

        return {"success": True, "retraining_results": results}

    except Exception as e:
        log.error(f"Error in model retraining: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    app_config = config.get_app_config()
    uvicorn.run(
        "main:app",
        host=app_config.get("host", "0.0.0.0"),
        port=app_config.get("port", 8000),
        reload=app_config.get("debug", False),
    )
