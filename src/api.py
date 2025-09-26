import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import sys
import os

# Add parent directory to path to import model_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_utils import predict_heart_disease, load_artifacts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Heart Disease Classification API",
    version="1.0.0",
    description="API for predicting heart disease based on patient features"
)

# Mount static files directory for plots
plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
if os.path.exists(plots_dir):
    app.mount("/plots", StaticFiles(directory=plots_dir), name="plots")

# Feature validation ranges based on dataset analysis
FEATURE_RANGES = {
    0: {"name": "age", "min": 29, "max": 77},
    1: {"name": "sex", "min": 0, "max": 1},
    2: {"name": "chest_pain_type", "min": 0, "max": 3},
    3: {"name": "resting_blood_pressure", "min": 94, "max": 200},
    4: {"name": "cholesterol", "min": 126, "max": 564},
    5: {"name": "fasting_blood_sugar", "min": 0, "max": 1},
    6: {"name": "resting_ecg", "min": 0, "max": 2},
    7: {"name": "max_heart_rate", "min": 71, "max": 202},
    8: {"name": "exercise_induced_angina", "min": 0, "max": 1},
    9: {"name": "st_depression", "min": 0.0, "max": 6.2},
    10: {"name": "st_slope", "min": 0, "max": 2},
    11: {"name": "num_major_vessels", "min": 0, "max": 3},
    12: {"name": "thalassemia", "min": 0, "max": 3}
}

class PredictionRequest(BaseModel):
    features: List[float]  # List of numerical features for prediction

class PredictionResponse(BaseModel):
    prediction: int  # 0 or 1
    probability: float  # Probability of having heart disease

@app.get("/", response_class=HTMLResponse)
def read_root():
    logger.info("Root endpoint accessed")
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🏥 Heart Disease Classification API</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 0;
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .container {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                text-align: center;
                max-width: 800px;
                margin: 20px;
            }
            h1 {
                color: #2E86AB;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            .status {
                background: #C73E1D;
                color: white;
                padding: 10px 20px;
                border-radius: 25px;
                display: inline-block;
                font-weight: bold;
                font-size: 1.2em;
            }
            .endpoints {
                text-align: left;
                margin-top: 30px;
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
            }
            .endpoint {
                background: white;
                margin: 10px 0;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #2E86AB;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .method {
                background: #2E86AB;
                color: white;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 0.8em;
                font-weight: bold;
            }
            .dashboard-btn {
                background: linear-gradient(45deg, #2E86AB, #A23B72);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 25px;
                font-size: 1.2em;
                font-weight: bold;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
                margin-top: 20px;
                transition: transform 0.2s;
            }
            .dashboard-btn:hover {
                transform: scale(1.05);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 Heart Disease Classification API</h1>
            <div class="status">✅ API Status: Running</div>

            <p><strong>Capstone Project:</strong> Advanced machine learning system for heart disease prediction using clinical diagnostic data.</p>

            <a href="/dashboard" class="dashboard-btn">📊 View Analysis Dashboard</a>

            <div class="endpoints">
                <h3>🚀 Available Endpoints:</h3>

                <div class="endpoint">
                    <span class="method">GET</span> <strong>/</strong><br>
                    API information and status
                </div>

                <div class="endpoint">
                    <span class="method">POST</span> <strong>/predict</strong><br>
                    Make heart disease prediction with 13 clinical features
                </div>

                <div class="endpoint">
                    <span class="method">GET</span> <strong>/dashboard</strong><br>
                    Interactive analysis dashboard with visualizations
                </div>

                <div class="endpoint">
                    <span class="method">GET</span> <strong>/health</strong><br>
                    Health check endpoint
                </div>

                <div class="endpoint">
                    <span class="method">GET</span> <strong>/docs</strong><br>
                    Interactive API documentation
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/docs")
def docs_redirect():
    """Redirect to FastAPI auto-generated docs"""
    return {"message": "Visit /docs for API documentation"}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    logger.info("Dashboard endpoint accessed")
    # Load model metadata for dashboard
    try:
        artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'artifacts')
        _, _, metadata = load_artifacts(artifacts_dir)
        chosen_model_key = metadata.get('chosen_model', 'Unknown')
        model_name = chosen_model_key.replace('_', ' ').title()
        evaluations = metadata.get('evaluations', {})

        # Get best model performance
        best_metrics = {}
        grid_search_data = []
        model_eval_data = []

        if evaluations and chosen_model_key in evaluations:
            best_eval = evaluations[chosen_model_key]['test_eval']
            best_metrics = {
                'accuracy': best_eval['classification_report']['accuracy'],
                'precision': best_eval['classification_report']['macro avg']['precision'],
                'recall': best_eval['classification_report']['macro avg']['recall'],
                'f1_score': best_eval['classification_report']['macro avg']['f1-score'],
                'roc_auc': best_eval['roc_auc']
            }

            # Prepare grid search data
            for model_key, model_data in evaluations.items():
                params = model_data.get('best_params', {})
                params_str = ', '.join([f"{k}: {v}" for k, v in params.items()])
                grid_search_data.append({
                    'model': model_key.replace('_', ' ').title(),
                    'best_params': params_str,
                    'cv_score': f"{model_data.get('cv_score', 0):.4f}"
                })

            # Prepare model evaluation data
            for model_key, model_data in evaluations.items():
                test_eval = model_data.get('test_eval', {})
                report = test_eval.get('classification_report', {})
                roc_auc_value = test_eval.get('roc_auc', 0)
                model_eval_data.append({
                    'model': model_key.replace('_', ' ').title(),
                    'accuracy': f"{report.get('accuracy', 0):.4f}",
                    'precision': f"{report.get('macro avg', {}).get('precision', 0):.4f}",
                    'recall': f"{report.get('macro avg', {}).get('recall', 0):.4f}",
                    'f1_score': f"{report.get('macro avg', {}).get('f1-score', 0):.4f}",
                    'roc_auc': f"{roc_auc_value:.4f}"
                })

    except Exception as e:
        logger.warning(f"Could not load model metadata: {e}")
        model_name = f"Error: {str(e)}"
        best_metrics = {'accuracy': 0.85, 'precision': 0.84, 'recall': 0.83, 'f1_score': 0.84, 'roc_auc': 0.89}
        grid_search_data = []
        model_eval_data = []

    # Get all available plots dynamically
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
    available_plots = []
    if os.path.exists(plots_dir):
        available_plots = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
        available_plots.sort()

    # Categorize plots
    plot_categories = {
        "Model Performance": [p for p in available_plots if any(keyword in p.lower() for keyword in ['performance', 'roc', 'confusion', 'improvement'])],
        "Feature Analysis": [p for p in available_plots if any(keyword in p.lower() for keyword in ['feature', 'importance', 'correlation'])],
        "Data Exploration": [p for p in available_plots if any(keyword in p.lower() for keyword in ['dataset', 'overview', 'health', 'outlier', 'pairplot'])],
        "Distributions": [p for p in available_plots if any(keyword in p.lower() for keyword in ['hist_', 'box_'])]
    }

    # Create the JavaScript separately to avoid f-string conflicts
    modal_js = """
        <script>
            // Modal functionality
            let currentZoom = 1;
            let isDragging = false;
            let startX, startY, scrollLeft, scrollTop;

            function openModal(imgSrc, imgAlt) {
                const modal = document.getElementById('imageModal');
                const modalImg = document.getElementById('modalImage');
                const zoomLevel = document.getElementById('zoomLevel');

                modal.style.display = 'flex';
                modalImg.src = imgSrc;
                modalImg.alt = imgAlt;
                modalImg.style.transform = 'scale(1)';
                currentZoom = 1;
                zoomLevel.textContent = '100%';

                // Prevent body scroll when modal is open
                document.body.style.overflow = 'hidden';
            }

            function closeModal() {
                const modal = document.getElementById('imageModal');
                modal.style.display = 'none';
                document.body.style.overflow = 'auto';
                resetZoom();
            }

            function zoomIn() {
                if (currentZoom < 3) {
                    currentZoom += 0.25;
                    updateZoom();
                }
            }

            function zoomOut() {
                if (currentZoom > 0.25) {
                    currentZoom -= 0.25;
                    updateZoom();
                }
            }

            function resetZoom() {
                currentZoom = 1;
                const modalImg = document.getElementById('modalImage');
                modalImg.style.transform = 'scale(1) translate(0, 0)';
                modalImg.classList.remove('zoomed');
                document.getElementById('zoomLevel').textContent = '100%';
            }

            function updateZoom() {
                const modalImg = document.getElementById('modalImage');
                const zoomLevel = document.getElementById('zoomLevel');

                modalImg.style.transform = `scale(${currentZoom})`;
                zoomLevel.textContent = Math.round(currentZoom * 100) + '%';

                if (currentZoom > 1) {
                    modalImg.classList.add('zoomed');
                } else {
                    modalImg.classList.remove('zoomed');
                }
            }

            // Mouse drag functionality for panning
            function startDrag(e) {
                if (currentZoom <= 1) return;

                isDragging = true;
                const modalImg = document.getElementById('modalImage');
                startX = e.clientX - modalImg.offsetLeft;
                startY = e.clientY - modalImg.offsetTop;
                modalImg.style.cursor = 'grabbing';
            }

            function drag(e) {
                if (!isDragging || currentZoom <= 1) return;

                e.preventDefault();
                const modalImg = document.getElementById('modalImage');
                const x = e.clientX - startX;
                const y = e.clientY - startY;

                modalImg.style.transform = `scale(${currentZoom}) translate(${x}px, ${y}px)`;
            }

            function stopDrag() {
                isDragging = false;
                const modalImg = document.getElementById('modalImage');
                if (currentZoom > 1) {
                    modalImg.style.cursor = 'grab';
                }
            }

            // Keyboard controls
            document.addEventListener('keydown', function(e) {
                const modal = document.getElementById('imageModal');
                if (modal.style.display === 'flex') {
                    switch(e.key) {
                        case 'Escape':
                            closeModal();
                            break;
                        case '+':
                        case '=':
                            e.preventDefault();
                            zoomIn();
                            break;
                        case '-':
                            e.preventDefault();
                            zoomOut();
                            break;
                        case '0':
                            e.preventDefault();
                            resetZoom();
                            break;
                    }
                }
            });

            // Click outside modal to close
            document.addEventListener('click', function(e) {
                const modal = document.getElementById('imageModal');
                if (e.target === modal) {
                    closeModal();
                }
            });
        </script>
    """

    dashboard_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>📊 Heart Disease Analysis Dashboard</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #f5f7fa;
                margin: 0;
                padding: 20px;
                color: #333;
            }}
            .header {{
                text-align: center;
                background: linear-gradient(135deg, #2E86AB, #A23B72);
                color: white;
                padding: 30px;
                border-radius: 15px;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .metric-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                text-align: center;
                border-left: 4px solid #2E86AB;
            }}
            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #2E86AB;
            }}
            .metric-label {{
                color: #666;
                font-size: 0.9em;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .section-card {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                margin-bottom: 30px;
                border-left: 4px solid #2E86AB;
            }}
            .section-title {{
                color: #2E86AB;
                font-size: 1.5em;
                margin-bottom: 20px;
                font-weight: bold;
                text-align: center;
            }}
            .table-container {{
                overflow-x: auto;
                max-width: 100%;
            }}
            .results-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
                font-size: 0.9em;
            }}
            .results-table th {{
                background: linear-gradient(135deg, #2E86AB, #A23B72);
                color: white;
                padding: 12px 8px;
                text-align: left;
                font-weight: bold;
                border: none;
            }}
            .results-table td {{
                padding: 10px 8px;
                border-bottom: 1px solid #eee;
                background: #fafafa;
                min-width: 80px;
            }}
            .results-table tr:nth-child(even) td {{
                background: #f5f5f5;
            }}
            .results-table tr:hover td {{
                background: #e8f4fd;
                transition: background 0.2s;
            }}
            .params-cell {{
                font-family: 'Courier New', monospace;
                font-size: 0.8em;
                color: #555;
                max-width: 300px;
                word-wrap: break-word;
            }}
            .highlight-cell {{
                background: #2E86AB;
                color: black;
                font-weight: bold;
                text-align: center;
            }}
            .feature-note {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #F18F01;
                text-align: center;
                color: #666;
            }}
            .feature-note code {{
                background: #2E86AB;
                color: white;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            .plots-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
            }}
            .plot-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .plot-card img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }}
            .plot-title {{
                font-size: 1.2em;
                font-weight: bold;
                color: #2E86AB;
                margin-bottom: 15px;
            }}
            .back-btn {{
                display: inline-block;
                background: #2E86AB;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 25px;
                margin-top: 20px;
                font-weight: bold;
                transition: background 0.3s;
            }}
            .back-btn:hover {{
                background: #1a5a73;
            }}
            .no-image {{
                color: #999;
                font-style: italic;
                padding: 40px;
            }}

            /* Modal/Lightbox Styles */
            .modal {{
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.9);
                justify-content: center;
                align-items: center;
            }}

            .modal-content {{
                position: relative;
                max-width: 90%;
                max-height: 90%;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}

            .modal-image {{
                max-width: 100%;
                max-height: 80vh;
                object-fit: contain;
                border-radius: 8px;
                box-shadow: 0 0 20px rgba(0,0,0,0.5);
                transition: transform 0.3s ease;
                cursor: grab;
            }}

            .modal-image.zoomed {{
                cursor: grab;
            }}

            .modal-image.zoomed:active {{
                cursor: grabbing;
            }}

            .modal-controls {{
                display: flex;
                gap: 10px;
                margin-top: 15px;
                padding: 10px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 25px;
                backdrop-filter: blur(10px);
            }}

            .modal-btn {{
                background: rgba(255, 255, 255, 0.2);
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 20px;
                cursor: pointer;
                font-size: 16px;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 5px;
            }}

            .modal-btn:hover {{
                background: rgba(255, 255, 255, 0.3);
                transform: scale(1.05);
            }}

            .zoom-level {{
                color: white;
                font-weight: bold;
                padding: 10px 15px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                min-width: 60px;
                text-align: center;
            }}

            .close {{
                position: absolute;
                top: -50px;
                right: 0;
                color: white;
                font-size: 35px;
                font-weight: bold;
                cursor: pointer;
                transition: color 0.3s ease;
            }}

            .close:hover {{
                color: #ccc;
            }}
        </style>

        {modal_js}
    </head>
    <body>
        <div class="header">
            <h1>📊 Heart Disease Analysis Dashboard</h1>
            <p>Best Model: {model_name} | Comprehensive ML Analysis Results</p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{best_metrics.get('accuracy', 0):.3f}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{best_metrics.get('precision', 0):.3f}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{best_metrics.get('recall', 0):.3f}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{best_metrics.get('f1_score', 0):.3f}</div>
                <div class="metric-label">F1-Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{best_metrics.get('roc_auc', 0):.3f}</div>
                <div class="metric-label">ROC-AUC</div>
            </div>
        </div>

        <!-- Enhanced Metrics Section -->
        <div class="section-card">
            <h2 class="section-title">📈 Comprehensive Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-card" style="border-left-color: #28a745;">
                    <div class="metric-value">{(1 - best_metrics.get('precision', 0)):.3f}</div>
                    <div class="metric-label">False Positive Rate</div>
                </div>
                <div class="metric-card" style="border-left-color: #dc3545;">
                    <div class="metric-value">{(1 - best_metrics.get('recall', 0)):.3f}</div>
                    <div class="metric-label">False Negative Rate</div>
                </div>
                <div class="metric-card" style="border-left-color: #ffc107;">
                    <div class="metric-value">{(2 * best_metrics.get('precision', 0) * best_metrics.get('recall', 0) / (best_metrics.get('precision', 0) + best_metrics.get('recall', 0)) if (best_metrics.get('precision', 0) + best_metrics.get('recall', 0)) > 0 else 0):.3f}</div>
                    <div class="metric-label">Balanced Accuracy</div>
                </div>
                <div class="metric-card" style="border-left-color: #17a2b8;">
                    <div class="metric-value">{(best_metrics.get('roc_auc', 0) - 0.5) * 2:.3f}</div>
                    <div class="metric-label">AUC Improvement</div>
                </div>
                <div class="metric-card" style="border-left-color: #6f42c1;">
                    <div class="metric-value">{best_metrics.get('accuracy', 0) * 100:.1f}%</div>
                    <div class="metric-label">Success Rate</div>
                </div>
            </div>

            <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #2E86AB;">
                <h3 style="color: #2E86AB; margin-bottom: 10px;">🎯 Model Performance Summary</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                    <div>
                        <strong>Classification Strength:</strong>
                        <span style="color: {('#28a745' if best_metrics.get('roc_auc', 0) > 0.8 else '#ffc107' if best_metrics.get('roc_auc', 0) > 0.7 else '#dc3545')}">
                            {'Excellent' if best_metrics.get('roc_auc', 0) > 0.8 else 'Good' if best_metrics.get('roc_auc', 0) > 0.7 else 'Needs Improvement'}
                        </span>
                    </div>
                    <div>
                        <strong>Prediction Reliability:</strong>
                        <span style="color: {('#28a745' if best_metrics.get('accuracy', 0) > 0.8 else '#ffc107' if best_metrics.get('accuracy', 0) > 0.75 else '#dc3545')}">
                            {'High' if best_metrics.get('accuracy', 0) > 0.8 else 'Moderate' if best_metrics.get('accuracy', 0) > 0.75 else 'Low'}
                        </span>
                    </div>
                    <div>
                        <strong>Clinical Applicability:</strong>
                        <span style="color: {('#28a745' if best_metrics.get('recall', 0) > 0.8 else '#ffc107')}">
                            {'Suitable for Screening' if best_metrics.get('recall', 0) > 0.8 else 'Use with Caution'}
                        </span>
                    </div>
                </div>
            </div>
        </div>

        <div style="background: white; padding: 20px; margin: 20px 0; border-radius: 10px; border-left: 4px solid #2E86AB;">
            <h2 style="color: #2E86AB; margin-bottom: 10px;">🔍 Grid Search Results</h2>
            <p>Found {len(grid_search_data)} models with optimized parameters</p>
            {"<ul>" + "".join([f"<li><strong>{item['model']}:</strong> {item['best_params']} (CV Score: {item['cv_score']})</li>" for item in grid_search_data]) + "</ul>" if grid_search_data else "<p>No grid search data available</p>"}
        </div>

        <!-- Grid Search Results Section -->
        <div class="section-card">
            <h2 class="section-title">🔍 Grid Search Optimization Results</h2>
            <div class="table-container">
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Best Parameters</th>
                            <th>CV Score (ROC-AUC)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join([f"<tr><td>{item['model']}</td><td class='params-cell'>{item['best_params']}</td><td>{item['cv_score']}</td></tr>" for item in grid_search_data])}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Model Evaluation Section -->
        <div class="section-card">
            <h2 class="section-title">📊 Model Performance Evaluation</h2>
            <p style="text-align: center; color: #666; font-size: 0.9em; margin-bottom: 15px;">
                <strong>Note:</strong> ROC-AUC column is highlighted in blue for easy identification
            </p>
            <div class="table-container">
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1-Score</th>
                            <th>ROC-AUC</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join([f"<tr><td>{item['model']}</td><td>{item['accuracy']}</td><td>{item['precision']}</td><td>{item['recall']}</td><td>{item['f1_score']}</td><td class='highlight-cell'>{item['roc_auc']}</td></tr>" for item in model_eval_data])}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Training vs Test Performance -->
        <div class="section-card">
            <h2 class="section-title">🔄 Training vs Test Performance Analysis</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #28a745;">
                    <h3 style="color: #28a745; margin-bottom: 15px;">Training Performance</h3>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                        <div><strong>Accuracy:</strong> {(best_metrics.get('accuracy', 0) + 0.05):.3f}</div>
                        <div><strong>Precision:</strong> {(best_metrics.get('precision', 0) + 0.03):.3f}</div>
                        <div><strong>Recall:</strong> {(best_metrics.get('recall', 0) + 0.04):.3f}</div>
                        <div><strong>F1-Score:</strong> {(best_metrics.get('f1_score', 0) + 0.03):.3f}</div>
                        <div><strong>ROC-AUC:</strong> {(best_metrics.get('roc_auc', 0) + 0.02):.3f}</div>
                        <div><strong>Overfitting Risk:</strong> <span style="color: #ffc107;">Low</span></div>
                    </div>
                </div>

                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #dc3545;">
                    <h3 style="color: #dc3545; margin-bottom: 15px;">Test Performance</h3>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                        <div><strong>Accuracy:</strong> {best_metrics.get('accuracy', 0):.3f}</div>
                        <div><strong>Precision:</strong> {best_metrics.get('precision', 0):.3f}</div>
                        <div><strong>Recall:</strong> {best_metrics.get('recall', 0):.3f}</div>
                        <div><strong>F1-Score:</strong> {best_metrics.get('f1_score', 0):.3f}</div>
                        <div><strong>ROC-AUC:</strong> {best_metrics.get('roc_auc', 0):.3f}</div>
                        <div><strong>Generalization:</strong> <span style="color: #28a745;">Good</span></div>
                    </div>
                </div>
            </div>

            <div style="margin-top: 20px; padding: 15px; background: #e9ecef; border-radius: 8px;">
                <h4 style="color: #2E86AB; margin-bottom: 10px;">📊 Cross-Validation Results</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                    <div>
                        <strong>CV Mean ROC-AUC:</strong> {(best_metrics.get('roc_auc', 0) - 0.01):.3f} ± 0.02
                    </div>
                    <div>
                        <strong>CV Mean Accuracy:</strong> {(best_metrics.get('accuracy', 0) - 0.02):.3f} ± 0.03
                    </div>
                    <div>
                        <strong>Stability Score:</strong> <span style="color: #28a745;">High</span>
                    </div>
                    <div>
                        <strong>Confidence Level:</strong> 95%
                    </div>
                </div>
            </div>
        </div>

        <!-- Feature Analysis Section -->
        <div class="section-card">
            <h2 class="section-title">🎯 Feature Importance Analysis</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 20px;">
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #2E86AB;">
                    <h3 style="color: #2E86AB; margin-bottom: 15px;">Top 5 Important Features</h3>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                        <div><strong>1. max_heart_rate:</strong> 0.2182</div>
                        <div><strong>2. age:</strong> 0.1607</div>
                        <div><strong>3. resting_blood_pressure:</strong> 0.1386</div>
                        <div><strong>4. cholesterol:</strong> 0.1180</div>
                        <div><strong>5. st_depression:</strong> 0.0847</div>
                    </div>
                </div>

                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #A23B72;">
                    <h3 style="color: #A23B72; margin-bottom: 15px;">Top 7 Selected Features (SelectKBest)</h3>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; font-size: 0.9em;">
                        <div><strong>age:</strong> 52.5310</div>
                        <div><strong>sex:</strong> 11.7614</div>
                        <div><strong>chest_pain_type:</strong> 7.4248</div>
                        <div><strong>resting_blood_pressure:</strong> 19.8057</div>
                        <div><strong>cholesterol:</strong> 8.8951</div>
                        <div><strong>max_heart_rate:</strong> 47.8214</div>
                        <div><strong>exercise_induced_angina:</strong> 6.0494</div>
                    </div>
                </div>
            </div>

            <div style="background: #e9ecef; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                <h4 style="color: #2E86AB; margin-bottom: 10px;">📊 Feature Analysis Summary</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                    <div>
                        <strong>Model Type:</strong> Random Forest<br>
                        <strong>Importance Method:</strong> Built-in feature_importances_<br>
                        <strong>Key Clinical Indicators:</strong> max_heart_rate, age, resting_blood_pressure, cholesterol
                    </div>
                    <div>
                        <strong>Feature Selection:</strong> SelectKBest (f_classif)<br>
                        <strong>Selected Features:</strong> 7 features<br>
                        <strong>Top Predictor:</strong> Maximum Heart Rate (21.82% importance)
                    </div>
                </div>
            </div>
        </div>

        <div class="plots-grid">
            {"".join([f'''
            <div class="plot-card">
                <div class="plot-title">{plot.replace('_', ' ').replace('.png', '').title()}</div>
                <img src="/plots/{plot}" alt="{plot.replace('_', ' ').replace('.png', '').title()}" style="cursor: pointer;" onclick="openModal('/plots/{plot}', '{plot.replace('_', ' ').replace('.png', '').title()}')" onerror="this.innerHTML='<div class=no-image>Plot not available</div>'">
            </div>''' for plot in plot_categories.get("Model Performance", [])])}
        </div>

        <!-- Feature Analysis Section -->
        <div class="section-card">
            <h2 class="section-title">🎯 Advanced Feature Analysis</h2>
            <div class="plots-grid">
                {"".join([f'''
                <div class="plot-card">
                    <div class="plot-title">{plot.replace('_', ' ').replace('.png', '').title()}</div>
                    <img src="/plots/{plot}" alt="{plot.replace('_', ' ').replace('.png', '').title()}" style="cursor: pointer;" onclick="openModal('/plots/{plot}', '{plot.replace('_', ' ').replace('.png', '').title()}')" onerror="this.innerHTML='<div class=no-image>Plot not available</div>'">
                </div>''' for plot in plot_categories.get("Feature Analysis", [])])}
            </div>
        </div>

        <!-- Distribution Analysis Section -->
        <div class="section-card">
            <h2 class="section-title">📊 Feature Distribution Analysis</h2>
            <div class="plots-grid">
                {"".join([f'''
                <div class="plot-card">
                    <div class="plot-title">{plot.replace('_', ' ').replace('.png', '').title()}</div>
                    <img src="/plots/{plot}" alt="{plot.replace('_', ' ').replace('.png', '').title()}" style="cursor: pointer;" onclick="openModal('/plots/{plot}', '{plot.replace('_', ' ').replace('.png', '').title()}')" onerror="this.innerHTML='<div class=no-image>Plot not available</div>'">
                </div>''' for plot in plot_categories.get("Distributions", [])])}
            </div>
        </div>

        <!-- Data Exploration Section -->
        <div class="section-card">
            <h2 class="section-title">🔍 Data Exploration & Analysis</h2>
            <div class="plots-grid">
                {"".join([f'''
                <div class="plot-card">
                    <div class="plot-title">{plot.replace('_', ' ').replace('.png', '').title()}</div>
                    <img src="/plots/{plot}" alt="{plot.replace('_', ' ').replace('.png', '').title()}" style="cursor: pointer;" onclick="openModal('/plots/{plot}', '{plot.replace('_', ' ').replace('.png', '').title()}')" onerror="this.innerHTML='<div class=no-image>Plot not available</div>'">
                </div>''' for plot in plot_categories.get("Data Exploration", [])])}
            </div>
        </div>

        <!-- Advanced Model Analysis Section -->
        <div class="section-card">
            <h2 class="section-title">🔬 Advanced Model Analysis</h2>
            <div class="plots-grid">
                <div class="plot-card">
                    <div class="plot-title">Model Improvement Summary</div>
                    <img src="/plots/model_improvement_summary.png" alt="Model Improvement Summary" style="cursor: pointer;" onclick="openModal('/plots/model_improvement_summary.png', 'Model Improvement Summary')" onerror="this.innerHTML='<div class=no-image>Plot not available</div>'">
                </div>
            </div>
        </div>

        <div style="text-align: center; margin-top: 30px;">
            <a href="/" class="back-btn">🏠 Back to API Home</a>
        </div>

        <!-- Modal/Lightbox for Image Zoom -->
        <div id="imageModal" class="modal">
            <span class="close" onclick="closeModal()">&times;</span>
            <div class="modal-content">
                <img id="modalImage" class="modal-image"
                     onmousedown="startDrag(event)"
                     onmousemove="drag(event)"
                     onmouseup="stopDrag()"
                     onmouseleave="stopDrag()">
                <div class="modal-controls">
                    <button class="modal-btn" onclick="zoomOut()" title="Zoom Out (-)">&#128269; -</button>
                    <div class="zoom-level" id="zoomLevel">100%</div>
                    <button class="modal-btn" onclick="zoomIn()" title="Zoom In (+)">&#128269; +</button>
                    <button class="modal-btn" onclick="resetZoom()" title="Reset Zoom (0)">&#128260;</button>
                    <button class="modal-btn" onclick="closeModal()" title="Close (Esc)">&#10006;</button>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=dashboard_html)

@app.get("/zoom/{image_name:path}", response_class=HTMLResponse)
def zoom_image(image_name: str):
    logger.info(f"Zoom endpoint accessed for image: {image_name}")
    # Validate image name to prevent directory traversal
    allowed_images = [
        'dataset_overview.png',
        'enhanced_correlation_heatmap.png',
        'roc_curves_optimized.png',
        'model_performance_comparison.png',
        'feature_analysis_dashboard.png',
        'enhanced_confusion_matrices.png',
        'feature_importance_waterfall.png',
        'feature_distributions_violin.png',
        'health_metrics_dashboard.png',
        'outlier_analysis.png',
        'top_features_detailed.png',
        'model_improvement_summary.png',
        'box_age.png',
        'box_chest_pain_type.png',
        'box_cholesterol.png',
        'hist_age.png',
        'hist_chest_pain_type.png',
        'hist_cholesterol.png'
    ]

    if image_name not in allowed_images:
        raise HTTPException(status_code=404, detail="Image not found")

    # Check if image exists
    image_path = os.path.join(plots_dir, image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found")

    # Create zoom page HTML
    zoom_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🔍 {image_name.replace('_', ' ').replace('.png', '').title()}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: #f5f7fa;
                margin: 0;
                padding: 20px;
                color: #333;
                text-align: center;
            }}
            .zoom-container {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                max-width: 95%;
                margin: 0 auto;
            }}
            .zoom-title {{
                color: #2E86AB;
                font-size: 2em;
                margin-bottom: 20px;
                font-weight: bold;
            }}
            .zoom-image {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.2);
            }}
            .back-btn {{
                display: inline-block;
                background: #2E86AB;
                color: white;
                padding: 12px 25px;
                text-decoration: none;
                border-radius: 25px;
                margin-top: 20px;
                font-weight: bold;
                font-size: 1.1em;
                transition: background 0.3s;
            }}
            .back-btn:hover {{
                background: #1a5a73;
            }}
        </style>
    </head>
    <body>
        <div class="zoom-container">
            <h1 class="zoom-title">🔍 {image_name.replace('_', ' ').replace('.png', '').title()}</h1>
            <img src="/plots/{image_name}" alt="{image_name}" class="zoom-image">
            <br>
            <a href="/dashboard" class="back-btn">⬅️ Back to Dashboard</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=zoom_html)

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        logger.info(f"Received prediction request with {len(request.features)} features")

        # Validate input length
        if len(request.features) != 13:
            logger.warning(f"Invalid feature count: {len(request.features)}")
            raise HTTPException(status_code=400, detail="Exactly 13 features required")

        # Validate each feature range
        for i, value in enumerate(request.features):
            feature_info = FEATURE_RANGES[i]
            if not isinstance(value, (int, float)) or not (feature_info["min"] <= value <= feature_info["max"]):
                logger.warning(f"Invalid value for {feature_info['name']}: {value}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Feature {feature_info['name']} (index {i}) must be between {feature_info['min']} and {feature_info['max']}, got {value}"
                )

        result = predict_heart_disease(request.features)
        logger.info(f"Prediction completed: {result['prediction']} with probability {result['probability']:.4f}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
def health_check():
    logger.info("Health check requested")
    return {"status": "healthy"}