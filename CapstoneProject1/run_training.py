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

print("Libraries imported successfully!")

# Load dataset
df = pd.read_csv('manufacturing_dataset_1000_samples.csv')
print(f"Dataset loaded! Shape: {df.shape}")
print("First 5 rows:")
print(df.head())

# Select features for modeling
features = [
    'Injection_Temperature', 'Injection_Pressure', 'Cycle_Time',
    'Cooling_Time', 'Material_Viscosity', 'Ambient_Temperature',
    'Machine_Age', 'Operator_Experience', 'Maintenance_Hours'
]
target = 'Parts_Per_Hour'

# Create a clean dataset
df_model = df[features + [target]].copy()
print(f"Model dataset shape: {df_model.shape}")

# Handle missing values (if any)
df_model = df_model.dropna()
print(f"After dropping missing values: {df_model.shape}")

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

print("Features scaled using StandardScaler")

# Build and train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Model trained successfully!")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Make predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Evaluate model
def evaluate_model(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{dataset_name} Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")

    return r2, rmse

train_r2, train_rmse = evaluate_model(y_train, y_train_pred, "Training")
test_r2, test_rmse = evaluate_model(y_test, y_test_pred, "Testing")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_,
    'Absolute_Coefficient': np.abs(model.coef_)
})

feature_importance = feature_importance.sort_values('Absolute_Coefficient', ascending=False)

print("Feature Importance (by absolute coefficient):")
print(feature_importance)

# Save model for deployment
with open('manufacturing_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'scaler': scaler,
        'features': features,
        'r2_score': test_r2
    }, f)

print("Model saved as 'manufacturing_model.pkl'")
print("Ready for deployment!")