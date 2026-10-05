import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Paths
DATA_DIR = "../data/processed"
MODEL_DIR = "../models"

os.makedirs(MODEL_DIR, exist_ok=True)


# Load processed training data
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv")


# Convert target to a one-dimensional Series
y_train = y_train["label_purchased"]


print("Training data loaded")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# Create baseline model
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
)


# Train model
model.fit(X_train, y_train)


print("\nBaseline model trained successfully!")


# Save model
model_path = f"{MODEL_DIR}/baseline_logistic_regression.pkl"

joblib.dump(model, model_path)

print("Model saved to:", model_path)