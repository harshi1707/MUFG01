import os
import json
import joblib
from typing import Dict

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.metrics import classification_report, roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')


# -----------------------------
# Data loading & preprocessing
# -----------------------------
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame, target_col: str, use_smote=True, feature_engineering=True):
    """Enhanced preprocessing with SMOTE and feature engineering"""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Feature engineering
    if feature_engineering:
        # Create interaction features
        X = create_interaction_features(X)

        # Add polynomial features for important numerical features
        num_cols = ['age', 'resting_blood_pressure', 'cholesterol', 'max_heart_rate', 'st_depression']
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        poly_features = poly.fit_transform(X[num_cols])
        poly_feature_names = [f'poly_{i}' for i in range(poly_features.shape[1])]

        # Add polynomial features to X
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=X.index)
        X = pd.concat([X, poly_df], axis=1)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply SMOTE for class balancing
    if use_smote:
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_scaled, y = smote.fit_resample(X_scaled, y)
        print(f"Applied SMOTE: {X_scaled.shape[0]} samples after balancing")

    return X_scaled, y, scaler

def create_interaction_features(X):
    """Create meaningful interaction features"""
    X = X.copy()

    # Age and heart rate interaction
    X['age_hr_interaction'] = X['age'] * X['max_heart_rate']

    # Blood pressure and cholesterol interaction
    X['bp_chol_interaction'] = X['resting_blood_pressure'] * X['cholesterol']

    # ST depression and slope interaction
    X['st_slope_interaction'] = X['st_depression'] * X['st_slope']

    # Age and exercise induced angina
    X['age_angina'] = X['age'] * X['exercise_induced_angina']

    return X


# -----------------------------
# Baseline models
# -----------------------------
def train_baselines(X, y) -> Dict:
    """Train enhanced baseline models with better defaults"""
    baselines = {
        "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "ada_boost": AdaBoostClassifier(random_state=42),
        "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
        "svm": SVC(probability=True, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": GaussianNB()
    }

    print(f"Training {len(baselines)} baseline models...")
    for name, model in baselines.items():
        print(f"  Training {name}...")
        model.fit(X, y)

    return baselines


# -----------------------------
# Grid Search optimization
# -----------------------------
def run_grid_search(X, y, random_state=42, cv=5) -> Dict:
    """Enhanced grid search with more models and better parameters"""

    # Define parameter grids for each model
    param_grids = {
        "decision_tree": {
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10, 15],
            "min_samples_leaf": [1, 2, 4, 8],
            "criterion": ["gini", "entropy"],
            "max_features": ["sqrt", "log2", None]
        },
        "random_forest": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        },
        "extra_trees": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None]
        },
        "gradient_boosting": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "subsample": [0.8, 0.9, 1.0]
        },
        "ada_boost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.5, 1.0],
            "algorithm": ["SAMME", "SAMME.R"]
        },
        "logistic_regression": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2", "elasticnet"],
            "solver": ["liblinear", "saga"],
            "max_iter": [1000, 2000],
            "l1_ratio": [0.1, 0.5, 0.9]  # For elasticnet
        },
        "svm": {
            "C": [0.1, 1, 10, 100, 1000],
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
            "degree": [2, 3, 4]  # For poly kernel
        },
        "knn": {
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree"],
            "p": [1, 2]  # Manhattan vs Euclidean
        }
    }

    # Define models
    models = {
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(random_state=random_state),
        "extra_trees": ExtraTreesClassifier(random_state=random_state),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "ada_boost": AdaBoostClassifier(random_state=random_state),
        "logistic_regression": LogisticRegression(random_state=random_state),
        "svm": SVC(probability=True, random_state=random_state),
        "knn": KNeighborsClassifier()
    }

    best_models = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

    for name, model in models.items():
        if name in param_grids:
            print(f"Grid searching: {name}")
            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                scoring=roc_auc_scorer,
                cv=skf,
                n_jobs=-1,
                verbose=0  # Reduce verbosity
            )
            gs.fit(X, y)
            best_models[name] = {
                "best_estimator": gs.best_estimator_,
                "best_params": gs.best_params_,
                "best_score": float(gs.best_score_)
            }
            print(f"    Best CV ROC-AUC: {gs.best_score_:.3f}")
    return best_models

def create_ensemble_models(best_models, X, y):
    """Create ensemble models using top performers"""
    print("Creating ensemble models...")

    # Get top 3 models by CV score
    sorted_models = sorted(best_models.items(), key=lambda x: x[1]['best_score'], reverse=True)
    top_models = sorted_models[:3]

    ensemble_models = {}

    # Voting Classifier (hard voting)
    estimators = [(name, info['best_estimator']) for name, info in top_models]
    voting_hard = VotingClassifier(estimators=estimators, voting='hard')
    voting_hard.fit(X, y)
    ensemble_models['voting_hard'] = voting_hard

    # Voting Classifier (soft voting)
    voting_soft = VotingClassifier(estimators=estimators, voting='soft')
    voting_soft.fit(X, y)
    ensemble_models['voting_soft'] = voting_soft

    # Stacking Classifier
    base_estimators = [(name, info['best_estimator']) for name, info in top_models[:2]]
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    stacking.fit(X, y)
    ensemble_models['stacking'] = stacking

    print(f"Created {len(ensemble_models)} ensemble models")
    return ensemble_models


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(estimator, X_test, y_test) -> Dict:
    y_pred = estimator.predict(X_test)
    try:
        y_prob = estimator.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = estimator.decision_function(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)

    return {
        "classification_report": report,
        "roc_auc": float(roc_auc)
    }


# -----------------------------
# Save artifacts
# -----------------------------
def save_artifacts(model, scaler, metadata: Dict, artifacts_dir: str = "artifacts") -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(model, os.path.join(artifacts_dir, "best_model.pkl"))
    if scaler:
        joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))
    with open(os.path.join(artifacts_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


# -----------------------------
# Load artifacts for prediction
# -----------------------------
def load_artifacts(artifacts_dir: str = "artifacts"):
    """Load the trained model, scaler, and metadata"""
    print(f"Loading artifacts from {artifacts_dir}...")
    model_path = os.path.join(artifacts_dir, "best_model.pkl")
    scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
    metadata_path = os.path.join(artifacts_dir, "metadata.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print(f"Loaded model: {metadata.get('chosen_model', 'unknown')}")
    return model, scaler, metadata


def predict_heart_disease(features: list, artifacts_dir: str = "artifacts"):
    """Predict heart disease probability from features"""
    try:
        print("Making prediction...")
        model, scaler, metadata = load_artifacts(artifacts_dir)
        print(f"Model type: {type(model)}")
        print(f"Model: {model}")

        # Assuming features are in order of the dataset columns
        import numpy as np
        features_scaled = scaler.transform(np.array([features]))
        print(f"Features scaled shape: {features_scaled.shape}")
        prob = model.predict_proba(features_scaled)[0][1]
        prediction = int(prob > 0.5)

        print(f"Prediction: {prediction} (probability: {prob:.4f})")
        return {"prediction": prediction, "probability": prob}
    except FileNotFoundError as e:
        raise RuntimeError(f"Model artifacts not found: {str(e)}")
    except Exception as e:
        print(f"Exception type: {type(e)}")
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Prediction error: {str(e)}")
