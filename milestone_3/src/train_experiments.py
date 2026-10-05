import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


DATA_DIR = "../../milestone_2/data/processed"
MODEL_DIR = "../models"
REPORT_DIR = "../reports"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")

y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv")["label_purchased"]
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv")["label_purchased"]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

mlflow.set_experiment("UrbanClap_Milestone_3")

configurations = [
    {"C": 0.1, "solver": "lbfgs", "max_iter": 1000},
    {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
    {"C": 10.0, "solver": "lbfgs", "max_iter": 1000},
]

results = []

best_f1 = -1
best_model = None
best_config = None

for config in configurations:

    with mlflow.start_run() as run:

        model = LogisticRegression(
            C=config["C"],
            solver=config["solver"],
            max_iter=config["max_iter"],
            random_state=42,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        mlflow.log_params(config)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
            model,
            name="logistic_regression_model",
        )

        results.append(
            {
                "run_id": run.info.run_id,
                "C": config["C"],
                "solver": config["solver"],
                "max_iter": config["max_iter"],
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }
        )

        print("\nExperiment completed")
        print("Run ID:", run.info.run_id)
        print("C:", config["C"])
        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1-score:", round(f1, 4))

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_config = config


results_df = pd.DataFrame(results)

results_path = f"{REPORT_DIR}/mlflow_experiment_results.csv"
results_df.to_csv(results_path, index=False)

best_model_path = f"{MODEL_DIR}/best_model.pkl"
joblib.dump(best_model, best_model_path)

report_path = f"{REPORT_DIR}/milestone_3_report.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("URBANCLAP DEMAND FORECASTING - MILESTONE 3\n")
    file.write("=" * 55 + "\n\n")

    file.write("1. OBJECTIVE\n")
    file.write("-" * 30 + "\n")
    file.write(
        "Train the primary Logistic Regression model using multiple "
        "hyperparameter configurations and track the experiments "
        "using MLflow.\n\n"
    )

    file.write("2. EXPERIMENT CONFIGURATIONS\n")
    file.write("-" * 30 + "\n")

    for config in configurations:
        file.write(
            f"C={config['C']}, "
            f"solver={config['solver']}, "
            f"max_iter={config['max_iter']}\n"
        )

    file.write("\n3. EXPERIMENT RESULTS\n")
    file.write("-" * 30 + "\n")
    file.write(results_df.to_string(index=False))

    file.write("\n\n4. BEST CONFIGURATION\n")
    file.write("-" * 30 + "\n")
    file.write(f"C: {best_config['C']}\n")
    file.write(f"Solver: {best_config['solver']}\n")
    file.write(f"Max iterations: {best_config['max_iter']}\n")
    file.write(f"Best F1-score: {best_f1:.4f}\n")

    file.write("\n5. CONCLUSION\n")
    file.write("-" * 30 + "\n")
    file.write(
        "Three Logistic Regression configurations were trained and "
        "tracked using MLflow. The configuration with the highest "
        "F1-score was selected as the best configuration. The "
        "experiment results provide evidence for comparing the "
        "hyperparameter configurations.\n"
    )

print("\n" + "=" * 50)
print("MILESTONE 3 EXPERIMENTS COMPLETED")
print("=" * 50)

print("\nBest configuration:")
print(best_config)
print("Best F1-score:", round(best_f1, 4))

print("\nResults saved to:")
print(results_path)

print("\nBest model saved to:")
print(best_model_path)

print("\nReport saved to:")
print(report_path)