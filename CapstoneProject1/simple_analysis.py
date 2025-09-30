"""
Manufacturing Equipment Output Prediction with Linear Regression
Complete data science project for predicting manufacturing equipment output
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
import pickle
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def main():
    print("Starting Manufacturing Equipment Output Prediction Analysis")
    print("=" * 60)

    # Load dataset
    print("Step 1: Loading dataset...")
    try:
        df = pd.read_csv('manufacturing_dataset_1000_samples.csv')
        print(f"Dataset loaded successfully! Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: Dataset file not found!")
        return False

    # Select features
    features = [
        'Injection_Temperature', 'Injection_Pressure', 'Cycle_Time',
        'Cooling_Time', 'Material_Viscosity', 'Ambient_Temperature',
        'Machine_Age', 'Operator_Experience', 'Maintenance_Hours'
    ]
    target = 'Parts_Per_Hour'

    # Create clean dataset
    df_model = df[features + [target]].copy()
    df_model = df_model.dropna()

    print(f"Model dataset shape: {df_model.shape}")

    # Split data
    X = df_model[features]
    y = df_model[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("Step 2: Training model...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    print("Model trained successfully!")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_:.4f}")

    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Evaluate model
    def evaluate_predictions(y_true, y_pred, dataset_name):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"\n{dataset_name} Performance:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R2: {r2:.4f}")

        return r2, rmse

    train_r2, train_rmse = evaluate_predictions(y_train, y_train_pred, "Training")
    test_r2, test_rmse = evaluate_predictions(y_test, y_test_pred, "Testing")

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_,
        'Absolute_Coefficient': np.abs(model.coef_)
    })
    feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    # Save model
    print("Step 3: Saving model...")
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'r2_score': test_r2
    }

    with open('manufacturing_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("Model saved as 'manufacturing_model.pkl'")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("Ready to run FastAPI server: python fastapi_app.py")

    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\nProject completed successfully!")
    else:
        print("\nProject failed!")