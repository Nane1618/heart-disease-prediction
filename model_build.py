# Installation of required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action="ignore")
import dill
import os
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split

# 1) Exploratory Data Analysis

# Reading the dataset
df = pd.read_csv("heart.csv")


# Categorical columns
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']

# Numerical columns
numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Categorical columns to 'category' type
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
for col in categorical_columns:
    df[col] = df[col].astype('category')

# Ensure folders exist for saving plots
folders = ["Histograms", "Boxplots", "Correlation", "Countplots"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Create histograms for each numerical column
for col in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.savefig(f"Histograms/{col}_histogram.jpg", format="jpg")
    plt.close()

# Create boxplots for each numerical column
for col in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.savefig(f"Boxplots/{col}_boxplot.jpg", format="jpg")
    plt.close()

# Create a correlation matrix
correlation_matrix = df.corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig("Correlation/correlation_heatmap.jpg", format="jpg")
plt.close()

# Create a countplot for the 'target' column
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df, palette='coolwarm')
plt.title('Countplot of Target')
plt.xlabel('Target')
plt.ylabel('Count')
plt.savefig("Countplots/target_countplot.jpg", format="jpg")
plt.close()

print("No of null values in the data",df.isnull().sum().sum())


num_duplicates = df.duplicated().sum()

if num_duplicates > 0:
    print(f"Number of duplicate records: {num_duplicates}")
    # Remove duplicate rows
    df = df.drop_duplicates()
    print("\nShape of Data after removing duplicates:")
    print(df.shape)
else:
    print("No records of duplicate values.")

X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import dill
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import dill
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# One-Hot Encoder
ohe_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Specify categorical columns
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# One Hot Encoding
X_train_encoded = ohe_encoder.fit_transform(X_train[categorical_columns])
X_test_encoded = ohe_encoder.transform(X_test[categorical_columns])

# Save the encoder
with open("ohe_encoder.dill", "wb") as f:
    dill.dump(ohe_encoder, f)

# Initialize the StandardScaler (for standardization)
scaler = StandardScaler()

# Specify numerical columns
numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Scaling numerical columns using StandardScaler (Standardization)
X_train_scaled = scaler.fit_transform(X_train[numerical_columns])
X_test_scaled = scaler.transform(X_test[numerical_columns])

# Save the scaler
with open("standard_encoder.dill", "wb") as f:
    dill.dump(scaler, f)

# Get the one-hot encoded column names
encoded_columns = ohe_encoder.get_feature_names_out(categorical_columns)

# Combine encoded and scaled features for the training set
X_train_final_before_smote = pd.concat([
    pd.DataFrame(X_train_scaled, columns=numerical_columns, index=X_train.index),
    pd.DataFrame(X_train_encoded, columns=encoded_columns, index=X_train.index)
], axis=1)

# Combine encoded and scaled features for the test set
X_test_final = pd.concat([
    pd.DataFrame(X_test_scaled, columns=numerical_columns, index=X_test.index),
    pd.DataFrame(X_test_encoded, columns=encoded_columns, index=X_test.index)
], axis=1)



# Print the shapes of the data before SMOTE
print("X_train shape before SMOTE:", X_train_final_before_smote.shape)
print("X_test shape:", X_test_final.shape)

# Copy y_train before SMOTE
y_train_before_smote = y_train.copy()

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)

# Fit SMOTE on the training data and transform it
X_train_final, y_train = smote.fit_resample(X_train_final_before_smote, y_train_before_smote)

# Checking the new shape of the resampled data
print("Resampled X_train shape:", X_train_final.shape)
print("Resampled y_train shape:", y_train.shape)

import joblib
import dill
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVC': SVC(),
    'KNN': KNeighborsClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Define hyperparameters for each model
param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200]
    },
    'Decision Tree': {
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'Random Forest': {
        'n_estimators': [100, 200,500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['auto', 'sqrt']
    },
    'SVC': {
        'C': [0.1, 1, 10,0.01],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'KNN': {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'XGBoost': {
        'n_estimators': [100, 200,500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
}

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

# Initialize a dictionary to store the best models and their respective cross-validation scores
best_estimators = {}
best_cv_score = -1
overall_best_model = None

# Iterate over each model for hyperparameter tuning
for model_name, model in models.items():
    print(f"Performing GridSearchCV for {model_name}...")

    # Get the hyperparameter grid for the current model
    param_grid = param_grids[model_name]

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)

    # Fit GridSearchCV
    grid_search.fit(X_train_final, y_train)

    # Save the best estimator
    best_estimators[model_name] = grid_search.best_estimator_

    # Print the best parameters and best score
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best cross-validation score for {model_name}: {grid_search.best_score_}")

    # Test the best model on the test data
    y_pred = grid_search.best_estimator_.predict(X_test_final)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy for {model_name}: {test_accuracy}\n")

    # Save the best model to a file using joblib (you can also use pickle)
    joblib.dump(grid_search.best_estimator_, f"{model_name}_best_model.pkl")

    # Check if the current model has the highest cross-validation score
    if grid_search.best_score_ > best_cv_score:
        best_cv_score = grid_search.best_score_
        overall_best_model = grid_search.best_estimator_

# Print out best models for each classifier
for model_name, best_model in best_estimators.items():
    print(f"Best model for {model_name}: {best_model}")

# Print the overall best model based on cross-validation score
print(f"Overall best model based on cross-validation score: {overall_best_model}")

# Save the overall best model to a .pkl file
joblib.dump(overall_best_model, "overall_best_model.pkl")