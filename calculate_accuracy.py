# Script to calculate accuracy of the Heart Disease Prediction Model
import numpy as np
import pandas as pd
import joblib
import dill
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
print("Loading dataset...")
df = pd.read_csv("heart.csv")

# Remove duplicates (as done in training)
df = df.drop_duplicates()

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Split the data (same random state as training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the encoders
print("Loading encoders...")
with open("ohe_encoder.dill", "rb") as f:
    ohe_encoder = dill.load(f)

with open("standard_encoder.dill", "rb") as f:
    scaler = dill.load(f)

# Define column types
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Preprocess test data
X_test_encoded = ohe_encoder.transform(X_test[categorical_columns])
X_test_scaled = scaler.transform(X_test[numerical_columns])

# Get feature names and combine (handle older sklearn encoder versions)
try:
    encoded_columns = ohe_encoder.get_feature_names_out(categorical_columns)
except AttributeError:
    # For older versions, construct feature names manually
    encoded_columns = []
    for i, col in enumerate(categorical_columns):
        for cat in ohe_encoder.categories_[i]:
            encoded_columns.append(f"{col}_{cat}")

X_test_final = pd.concat([
    pd.DataFrame(X_test_scaled, columns=numerical_columns, index=X_test.index),
    pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)
], axis=1)

# Load the overall best model
print("Loading model...")
model = joblib.load("overall_best_model.pkl")

# Make predictions
y_pred = model.predict(X_test_final)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print("MODEL ACCURACY RESULTS")
print("="*50)
print(f"\nModel Type: {type(model).__name__}")
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"\nNumber of Test Samples: {len(y_test)}")

# Detailed classification report
print("\n" + "-"*50)
print("CLASSIFICATION REPORT")
print("-"*50)
print(classification_report(y_test, y_pred, target_names=['No Heart Disease', 'Heart Disease']))

# Confusion Matrix
print("\n" + "-"*50)
print("CONFUSION MATRIX")
print("-"*50)
cm = confusion_matrix(y_test, y_pred)
print(f"\n{cm}")
print(f"\nTrue Negatives: {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}")
print(f"True Positives: {cm[1][1]}")

# Also test all saved models for comparison
print("\n" + "="*50)
print("ACCURACY COMPARISON OF ALL MODELS")
print("="*50)

model_files = [
    ("Logistic Regression", "Logistic Regression_best_model.pkl"),
    ("Decision Tree", "Decision Tree_best_model.pkl"),
    ("Random Forest", "Random Forest_best_model.pkl"),
    ("SVC", "SVC_best_model.pkl"),
    ("KNN", "KNN_best_model.pkl"),
    ("XGBoost", "XGBoost_best_model.pkl")
]

for name, filename in model_files:
    try:
        model = joblib.load(filename)
        y_pred = model.predict(X_test_final)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name:25s}: {acc * 100:.2f}%")
    except Exception as e:
        print(f"{name:25s}: Error loading - {e}")
