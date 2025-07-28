from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import pandas as pd
from werkzeug.utils import secure_filename
from src.utils.config import ConfigManager
from src.core.analysis_pipeline import AnalysisPipeline
from src.utils.logger import setup_logging

# Setup logging
setup_logging()

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB max file size

# Initialize components
config_manager = ConfigManager()
analysis_pipeline = AnalysisPipeline()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/preview-columns", methods=["POST"])
def preview_columns():
    """Preview CSV columns to allow user selection."""
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"})

        if file and allowed_file(file.filename):
            # Read CSV to get columns
            df = pd.read_csv(file)
            columns = df.columns.tolist()

            return jsonify({"success": True, "columns": columns, "total_rows": len(df)})
        else:
            return jsonify(
                {
                    "success": False,
                    "error": "Invalid file type. Please upload a CSV file.",
                }
            )

    except Exception as e:
        return jsonify(
            {"success": False, "error": f"Error previewing columns: {str(e)}"}
        )


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and analysis with column selection."""
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file provided"})

        file = request.files["file"]
        summary_column = request.form.get("summary_column", "summary")

        if file.filename == "":
            return jsonify({"success": False, "error": "No file selected"})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Run analysis with selected column
            analysis_result = analysis_pipeline.run_analysis(filepath, summary_column)

            # Clean up uploaded file
            os.remove(filepath)

            return jsonify({"success": True, "analysis": analysis_result})
        else:
            return jsonify(
                {
                    "success": False,
                    "error": "Invalid file type. Please upload a CSV file.",
                }
            )

    except Exception as e:
        return jsonify({"success": False, "error": f"Error processing file: {str(e)}"})


@app.route("/api/chat", methods=["POST"])
def chat():
    """Handle chat messages."""
    try:
        data = request.get_json()
        message = data.get("message", "")
        context = data.get("context", {})

        # Simple response for now - can be enhanced with actual chat functionality
        response = (
            f"I received your message: {message}. This is a placeholder response."
        )

        return jsonify({"success": True, "message": response})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/optimize-hyperparameters", methods=["POST"])
def optimize_hyperparameters():
    """Optimize clustering hyperparameters."""
    try:
        # Placeholder for hyperparameter optimization
        return jsonify(
            {"success": True, "message": "Hyperparameter optimization completed"}
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/predict-clusters", methods=["POST"])
def predict_clusters():
    """Predict clusters for new data."""
    try:
        # Placeholder for cluster prediction
        return jsonify({"success": True, "message": "Cluster prediction completed"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/check-drift", methods=["POST"])
def check_drift():
    """Check for data drift."""
    try:
        # Placeholder for drift detection
        return jsonify({"success": True, "message": "Drift detection completed"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """Get model information."""
    try:
        # Placeholder for model info
        return jsonify(
            {
                "success": True,
                "model_info": {
                    "last_trained": "2024-01-01",
                    "performance": "Good",
                    "version": "1.0.0",
                },
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
