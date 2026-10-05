
import os

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# Paths
DATA_DIR = "../data/processed"
MODEL_PATH = "../models/baseline_logistic_regression.pkl"
REPORT_DIR = "../reports"

os.makedirs(REPORT_DIR, exist_ok=True)


# Load test data
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv")

y_test = y_test["label_purchased"]


# Load trained model
model = joblib.load(MODEL_PATH)


# Make predictions
y_pred = model.predict(X_test)


# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)


# Display results
print("BASELINE MODEL EVALUATION")
print("=" * 40)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\nConfusion Matrix")
print("=" * 40)
print(cm)

print("\nClassification Report")
print("=" * 40)
print(report)


# Save evaluation results
report_path = f"{REPORT_DIR}/baseline_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("URBANCLAP DEMAND FORECASTING - BASELINE REPORT\n")
    file.write("=" * 55 + "\n\n")

    file.write("1. DATASET\n")
    file.write("-" * 30 + "\n")
    file.write("Dataset: clickstream.csv\n")
    file.write("Total records: 1000\n")
    file.write("Features used: 10\n")
    file.write("Target: label_purchased\n\n")

    file.write("2. PREPROCESSING\n")
    file.write("-" * 30 + "\n")
    file.write("- Missing search_query values replaced with 'No Search'\n")
    file.write("- Categorical features converted using One-Hot Encoding\n")
    file.write("- Dataset split into 80% training and 20% testing\n")
    file.write("- random_state = 42\n")
    file.write("- Stratified train-test split was used\n\n")

    file.write("3. BASELINE MODEL\n")
    file.write("-" * 30 + "\n")
    file.write("Model: Logistic Regression\n")
    file.write("max_iter: 1000\n")
    file.write("random_state: 42\n\n")

    file.write("4. EVALUATION RESULTS\n")
    file.write("-" * 30 + "\n")
    file.write(f"Accuracy : {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall   : {recall:.4f}\n")
    file.write(f"F1-score : {f1:.4f}\n\n")

    file.write("5. CONFUSION MATRIX\n")
    file.write("-" * 30 + "\n")
    file.write(str(cm))
    file.write("\n\n")

    file.write("6. CLASSIFICATION REPORT\n")
    file.write("-" * 30 + "\n")
    file.write(report)

    file.write("\n7. BASELINE CONCLUSION\n")
    file.write("-" * 30 + "\n")
    file.write(
        "The Logistic Regression model establishes the initial "
        "performance benchmark for purchase prediction. "
        "Future models can be compared against these baseline metrics.\n"
    )


print("\nBaseline report saved to:")
print(report_path)