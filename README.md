# Heart Disease Detection Capstone Project

## Project Overview

This capstone project implements a machine learning solution for heart disease prediction using patient clinical features. The system uses various classification algorithms to predict the likelihood of heart disease based on 13 medical features including age, sex, chest pain type, blood pressure, cholesterol levels, and other cardiac indicators.

The project follows a complete ML pipeline from exploratory data analysis through model deployment, resulting in a production-ready FastAPI web service that can be run locally or containerized with Docker.

## Completed Phases

### 1. Exploratory Data Analysis (EDA)
- Comprehensive analysis of the heart disease dataset
- Feature distribution visualization (histograms, box plots)
- Correlation analysis and heatmap generation
- Outlier detection and data quality assessment
- Statistical summaries and insights

### 2. Baseline Models
- Implementation of multiple baseline classifiers:
  - Decision Tree
  - Random Forest
  - Logistic Regression
  - Support Vector Machine (SVM)
- Initial performance evaluation without hyperparameter tuning

### 3. Model Optimization
- Grid search hyperparameter optimization for all models
- Cross-validation performance evaluation
- Best model selection based on ROC-AUC metric
- Model persistence and artifact saving

### 4. Model Evaluation
- Comprehensive evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- Confusion matrix analysis
- Cross-validation results interpretation
- Model comparison and selection

### 5. Feature Analysis
- Feature importance analysis using Random Forest
- Correlation studies between features and target
- Key feature identification and visualization
- Insights into predictive feature relationships

### 6. Deployment
- FastAPI web service implementation
- Input validation and error handling
- Docker containerization
- API documentation and testing
- Production-ready deployment setup

## Installation and Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Docker (optional, for containerized deployment)

### Local Setup
1. Clone or download the project repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure model artifacts are present in the `artifacts/` directory:
   - `best_model.pkl`: Trained model
   - `scaler.pkl`: Feature scaler
   - `metadata.json`: Model metadata

## How to Run Scripts

### Exploratory Data Analysis
Run the EDA script to generate visualizations and statistical analysis:
```bash
python eda.py
```
This will create plots in the `plots/` directory including histograms, box plots, and correlation heatmaps.

### Model Training
Execute the training pipeline with hyperparameter optimization:
```bash
python train.py
```
This script performs:
- Data preprocessing and scaling
- Baseline model training
- Grid search optimization
- Model evaluation and artifact saving

### Feature Analysis
Analyze feature importance and relationships:
```bash
python feature_analysis.py
```
Generates feature importance plots and correlation analysis.

## Local API Execution

### Using the Run Script
Start the FastAPI server locally with auto-reload for development:
```bash
python run_api.py
```

### Manual Uvicorn Execution
Alternatively, run directly with uvicorn:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## Docker Build and Run Commands

### Build Docker Image
```bash
docker build -t heart-disease-api .
```

### Run Docker Container
```bash
docker run -p 8000:8000 heart-disease-api
```

### Build and Run in One Command
```bash
docker build -t heart-disease-api . && docker run -p 8000:8000 heart-disease-api
```

## API Endpoints

### GET /
Returns API status information.
**Response:**
```json
{
  "message": "Heart Disease Classification API",
  "status": "running"
}
```

### GET /docs
Access FastAPI auto-generated interactive API documentation (Swagger UI).

### GET /dashboard
Comprehensive interactive analysis dashboard with advanced visualizations and detailed model performance metrics.
**Features:**
- **Enhanced Performance Metrics**: Accuracy, precision, recall, F1-score, ROC-AUC, false positive/negative rates, balanced accuracy
- **Model Assessment Summary**: Classification strength, prediction reliability, clinical applicability ratings
- **Training vs Test Analysis**: Performance comparison with overfitting detection
- **Cross-Validation Results**: Stability scores and confidence intervals
- **Grid Search Optimization**: Hyperparameter tuning results across all models
- **Advanced Visualizations**: 15+ interactive plots including feature importance waterfall, violin distributions, box plots, histograms, outlier analysis, and model improvement summaries
- **Interactive Images**: Click any plot to zoom in and view it in full-size detail
- **Organized Sections**: Feature Analysis, Distribution Analysis, and Advanced Model Analysis

### GET /zoom/{image_name}
View individual plots in full-size zoom mode.
**Supported images:**
- dataset_overview.png
- enhanced_correlation_heatmap.png
- roc_curves_optimized.png
- model_performance_comparison.png
- feature_analysis_dashboard.png
- enhanced_confusion_matrices.png

**Example:** `GET /zoom/roc_curves_optimized.png`

### GET /health
Health check endpoint.
**Response:**
```json
{
  "status": "healthy"
}
```

### POST /predict
Predict heart disease probability from patient features.

**Request Body:**
```json
{
  "features": [
    58,  // age (29-77)
    1,   // sex (0-1)
    1,   // chest_pain_type (0-3)
    134, // resting_blood_pressure (94-200)
    246, // cholesterol (126-564)
    0,   // fasting_blood_sugar (0-1)
    0,   // resting_ecg (0-2)
    155, // max_heart_rate (71-202)
    0,   // exercise_induced_angina (0-1)
    0.4, // st_depression (0.0-6.2)
    1,   // st_slope (0-2)
    1,   // num_major_vessels (0-3)
    2    // thalassemia (0-3)
  ]
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.78
}
```

**Error Responses:**
- `400 Bad Request`: Invalid input features or out-of-range values
- `500 Internal Server Error`: Model loading or prediction failure

## Key Results and Findings

### Best Model Performance
- **Chosen Model**: Random Forest
- **Best Parameters**:
  - n_estimators: 50
  - max_depth: 7
  - max_features: sqrt
  - min_samples_split: 5
  - min_samples_leaf: 1
- **Cross-Validation ROC-AUC**: 0.732
- **Test Set Performance**:
  - Accuracy: 66.25%
  - ROC-AUC: 0.749
  - Precision (Class 0): 0.667
  - Recall (Class 0): 0.500
  - Precision (Class 1): 0.660
  - Recall (Class 1): 0.795

### Key Insights
1. **Important Features**: Maximum heart rate, ST depression, and chest pain type are among the most predictive features
2. **Model Comparison**: Random Forest outperformed other models in ROC-AUC metric
3. **Data Quality**: The dataset shows good balance and feature distributions suitable for classification
4. **Clinical Relevance**: The model provides probabilistic predictions useful for medical decision support

### Deployment Features
- **Input Validation**: Comprehensive range checking for all 13 features
- **Error Handling**: Robust error handling for model loading and prediction failures
- **Logging**: Structured logging for monitoring and debugging
- **Containerization**: Optimized Docker setup for production deployment
- **API Documentation**: Auto-generated interactive docs for easy integration

This project demonstrates a complete ML engineering workflow from data exploration to production deployment, resulting in a reliable heart disease prediction service.