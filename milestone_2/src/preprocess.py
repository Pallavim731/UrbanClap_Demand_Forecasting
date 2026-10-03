import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# Paths
DATA_PATH = "../data/clickstream.csv"
OUTPUT_DIR = "../data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Shape:", df.shape)


# Features and target
features = [
    "city",
    "device",
    "category",
    "product_views",
    "add_to_cart",
    "session_duration_mins",
    "return_visitor",
    "discount_applied",
    "search_query",
    "recommendation_clicked",
]

target = "label_purchased"

X = df[features].copy()
y = df[target].copy()


# Handle missing search queries
X["search_query"] = X["search_query"].fillna("No Search")


# Categorical and numerical columns
categorical_features = [
    "city",
    "device",
    "category",
    "search_query",
]

numerical_features = [
    "product_views",
    "add_to_cart",
    "session_duration_mins",
    "return_visitor",
    "discount_applied",
    "recommendation_clicked",
]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numerical",
            "passthrough",
            numerical_features,
        ),
    ]
)


# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# Convert processed data to DataFrames
feature_names = preprocessor.get_feature_names_out()

X_train_processed = pd.DataFrame(
    X_train_processed.toarray()
    if hasattr(X_train_processed, "toarray")
    else X_train_processed,
    columns=feature_names,
)

X_test_processed = pd.DataFrame(
    X_test_processed.toarray()
    if hasattr(X_test_processed, "toarray")
    else X_test_processed,
    columns=feature_names,
)


# Save processed datasets
X_train_processed.to_csv(
    f"{OUTPUT_DIR}/X_train.csv",
    index=False,
)

X_test_processed.to_csv(
    f"{OUTPUT_DIR}/X_test.csv",
    index=False,
)

y_train.to_csv(
    f"{OUTPUT_DIR}/y_train.csv",
    index=False,
)

y_test.to_csv(
    f"{OUTPUT_DIR}/y_test.csv",
    index=False,
)


print("\nPreprocessing completed successfully!")
print("Training samples:", len(X_train_processed))
print("Testing samples:", len(X_test_processed))
print("Processed features:", X_train_processed.shape[1])

print("\nSaved files:")
print("X_train.csv")
print("X_test.csv")
print("y_train.csv")
print("y_test.csv")