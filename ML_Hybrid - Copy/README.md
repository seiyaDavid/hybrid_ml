# ML Hybrid Theme Analysis System

A comprehensive, modular, and scalable ML system for analyzing themes in CSV files containing issue summaries. This system integrates advanced text preprocessing, embedding generation, clustering, and LLM-based theme analysis using AWS Bedrock (Claude Sonnet and Titan Embeddings G1) orchestrated by LangChain.

## 🚀 Features

- **Flexible Column Selection**: Choose any column for analysis, not just "summary"
- **Advanced Text Preprocessing**: HTML cleaning, encoding fixes, lemmatization, language detection
- **Embedding Generation**: AWS Titan Embeddings G1 via LangChain
- **Intelligent Clustering**: UMAP + HDBSCAN with hyperparameter optimization
- **LLM Theme Analysis**: Claude Sonnet for theme naming and descriptions
- **Zero-Shot Classification**: Claude Sonnet for data quality analysis
- **Model Persistence**: Save and load trained models for faster inference
- **Professional UI**: Warm, earthy-themed Flask web interface
- **Comprehensive Testing**: Pytest suite with coverage reporting

## 📋 Prerequisites

- Python 3.8+
- AWS Account with Bedrock access
- Internet connection for AWS services

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ML_Hybrid
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure AWS Credentials
Create or update `config/credentials.yaml`:
```yaml
aws:
  access_key_id: "your-access-key"
  secret_access_key: "your-secret-key"
  region: "us-east-1"
  bedrock:
    model_id: "anthropic.claude-3-sonnet-20240229-v1:0"
```

## 🏃‍♂️ Running the Application

### 1. Start the Flask Web Application
```bash
python web_app.py
```

### 2. Access the Web Interface
Open your browser and navigate to:
```
http://localhost:5000
```

### 3. Usage Flow
1. **Upload CSV** → Select your CSV file
2. **Choose Column** → Select which column contains summary text
3. **Start Analysis** → Click "Start Analysis" button
4. **View Results** → Explore themes, clusters, and chat features

## 🔌 API Endpoints

### Core Analysis Endpoints

#### 1. Column Preview
```bash
POST /preview-columns
```
- **Purpose**: Preview CSV columns for selection
- **Input**: CSV file
- **Output**: List of available columns

#### 2. File Upload & Analysis
```bash
POST /upload
```
- **Purpose**: Upload CSV and run complete analysis
- **Input**: CSV file + `summary_column` parameter
- **Output**: Complete analysis results

### Advanced ML Endpoints

#### 3. Hyperparameter Optimization
```bash
POST /api/optimize-hyperparameters
```
- **Purpose**: Optimize UMAP/HDBSCAN parameters
- **Input**: File path
- **Output**: Optimized parameters

#### 4. Cluster Prediction
```bash
POST /api/predict-clusters
```
- **Purpose**: Predict clusters for new data
- **Input**: File path
- **Output**: Cluster predictions

#### 5. Data Drift Detection
```bash
POST /api/check-drift
```
- **Purpose**: Check for data drift
- **Input**: File path
- **Output**: Drift analysis results

#### 6. Model Information
```bash
GET /api/model-info
```
- **Purpose**: Get saved model details
- **Output**: Model metadata

### Chat Interface

#### 7. Interactive Chat
```bash
POST /api/chat
```
- **Purpose**: Chat about themes and analysis
- **Input**: Message + context
- **Output**: AI response

### Example API Usage

```bash
# Preview columns
curl -X POST -F "file=@data.csv" http://localhost:5000/preview-columns

# Run analysis
curl -X POST -F "file=@data.csv" -F "summary_column=description" http://localhost:5000/upload

# Chat about themes
curl -X POST -H "Content-Type: application/json" \
  -d '{"message": "What are the main themes?", "context": {"analysis_id": "123"}}' \
  http://localhost:5000/api/chat
```

### Response Format
```json
{
  "success": true,
  "analysis": {
    "data_info": {...},
    "clustering_results": {...},
    "theme_analysis": {...},
    "data_quality_analysis": {...}
  }
}
```

## 🧪 Testing

### 1. Install Test Dependencies
```bash
pip install pytest pytest-cov pytest-mock
```

### 2. Run All Tests
```bash
pytest
```

### 3. Run Tests with Coverage
```bash
pytest --cov=src --cov-report=html
```

### 4. Run Specific Test Files
```bash
# Run specific test file
pytest tests/test_analysis_pipeline.py

# Run tests matching pattern
pytest -k "test_upload"

# Run tests in specific directory
pytest tests/
```

### 5. Run Tests with Verbose Output
```bash
pytest -v
```

### 6. Run Tests with Detailed Output
```bash
pytest -vv --tb=long
```

### 7. Generate Coverage Report
```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Generate XML coverage report
pytest --cov=src --cov-report=xml

# Generate terminal coverage report
pytest --cov=src --cov-report=term
```

### 8. Run Tests in Parallel (if available)
```bash
pytest -n auto
```

### 9. Common Pytest Options
```bash
pytest -x          # Stop on first failure
pytest --maxfail=2 # Stop after 2 failures
pytest -s          # Show print statements
pytest -k "not slow" # Skip tests with "slow" in name
```

### 10. View Coverage Report
After running with HTML coverage:
```bash
# Open coverage report in browser
start htmlcov/index.html  # Windows
open htmlcov/index.html   # Mac
```

### Test Structure
```
tests/
├── test_analysis_pipeline.py
├── test_theme_analyzer.py
├── test_theme_clusterer.py
├── test_text_processor.py
├── test_config.py
└── test_web_app.py
```

### Example Test Run
```bash
# Quick test run
pytest tests/test_analysis_pipeline.py -v

# Full test suite with coverage
pytest --cov=src --cov-report=html --cov-report=term
```

## 📁 Project Structure

```
ML_Hybrid/
├── config/
│   └── credentials.yaml          # AWS and application configuration
├── src/
│   ├── core/
│   │   └── analysis_pipeline.py  # Main orchestration pipeline
│   ├── preprocessing/
│   │   └── text_processor.py     # Text preprocessing utilities
│   ├── clustering/
│   │   ├── theme_clusterer.py    # UMAP + HDBSCAN clustering
│   │   ├── hyperparameter_tuner.py # Optuna optimization
│   │   └── model_persistence.py  # Model save/load functionality
│   ├── llm/
│   │   └── theme_analyzer.py     # Claude Sonnet theme analysis
│   └── utils/
│       ├── config.py             # Configuration management
│       └── logger.py             # Logging setup
├── templates/
│   └── index.html               # Flask web interface template
├── static/
│   ├── css/
│   │   └── style.css            # Warm earthy theme styling
│   └── js/
│       └── app.js               # Frontend interactivity
├── tests/                       # Comprehensive test suite
├── web_app.py                   # Flask application entry point
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🔧 Configuration

### AWS Bedrock Configuration
Update `config/credentials.yaml`:
```yaml
aws:
  access_key_id: "your-access-key"
  secret_access_key: "your-secret-key"
  region: "us-east-1"
  bedrock:
    model_id: "anthropic.claude-3-sonnet-20240229-v1:0"

application:
  max_file_size: 10485760  # 10MB
  allowed_extensions: ["csv"]

preprocessing:
  max_text_length: 1000
  min_text_length: 10
  remove_html: true
  fix_encoding: true
  remove_duplicates: true

clustering:
  umap:
    n_neighbors: 15
    n_components: 2
    min_dist: 0.1
  hdbscan:
    min_cluster_size: 5
    min_samples: 5

hyperparameter_tuning:
  enabled: true
  n_trials: 50
  timeout: 300

model_persistence:
  enabled: true
  save_path: "models/"
  drift_threshold: 0.1
```

## 🎨 UI Customization

### Change Title and Logo
Edit `templates/index.html`:
```html
<div class="logo-text">
    <h1>Your Custom Title</h1>
    <p>Your Custom Subtitle</p>
</div>
```

### Modify Color Theme
Update `static/css/style.css` with your preferred colors.

## 🚨 Troubleshooting

### Common Issues

1. **AWS Credentials Error**
   - Ensure AWS credentials are properly configured
   - Check Bedrock access permissions

2. **Import Errors**
   - Run `pip install -r requirements.txt`
   - Check Python version (3.8+)

3. **File Upload Issues**
   - Ensure CSV file is valid
   - Check file size limits (10MB)

4. **Analysis Failures**
   - Verify selected column contains text data
   - Check internet connection for AWS services

### Debug Mode
Run Flask in debug mode for detailed error messages:
```bash
python web_app.py
```

## 📊 Performance

- **Scalability**: Handles 100k+ summaries efficiently
- **Processing Time**: <60s for 10k rows
- **Memory Usage**: Optimized for large datasets
- **Model Persistence**: Avoids retraining for new data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Run tests to verify functionality
4. Create an issue with detailed error information

---

**Built with ❤️ using Flask, AWS Bedrock, LangChain, and modern ML techniques** 