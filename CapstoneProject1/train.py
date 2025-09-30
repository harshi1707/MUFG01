import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("manufacturing_dataset_1000_samples.csv")

# Drop unwanted columns
drop_cols = ['Machine_Utilization', 'Parts_Per_Hour', 'Total_Cycle_Time',
              'Efficiency_Score', 'Temperature_Pressure_Ratio', 'Timestamp',
              'Machine_Type', 'Material_Grade', 'Day_of_Week']
df = df.drop(columns=drop_cols, errors='ignore')

# Define features and target
X = df.drop(columns=['Shift'])   # Target column
y = df['Shift']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
pipeline = Pipeline([
    ('imputer', KNNImputer(n_neighbors=3)),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump((pipeline, X.columns), f)  # Save also the column order