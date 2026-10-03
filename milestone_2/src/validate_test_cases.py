import json

import pandas as pd


TEST_CASES_PATH = "../data/test_cases.json"
DATA_PATH = "../data/clickstream.csv"


# Load test cases
with open(TEST_CASES_PATH, "r", encoding="utf-8") as file:
    test_cases = json.load(file)


# Load original dataset
df = pd.read_csv(DATA_PATH)


print("TEST CASE VALIDATION")
print("=" * 40)

print("Number of test cases:", len(test_cases))
print("Original dataset rows:", len(df))


# Validate each test case
required_input_fields = {
    "category",
    "user_device",
    "session_mins",
    "prior_purchases",
}


valid_cases = 0

for case in test_cases:
    test_id = case["test_id"]
    inputs = case["input"]
    expected_label = case["expected_label"]

    missing_fields = required_input_fields - set(inputs.keys())

    if missing_fields:
        print(
            f"{test_id}: FAILED - missing fields {missing_fields}"
        )
        continue

    if expected_label not in (0, 1):
        print(
            f"{test_id}: FAILED - invalid expected label"
        )
        continue

    valid_cases += 1


print("\nValidation Summary")
print("=" * 40)
print("Total cases:", len(test_cases))
print("Valid cases:", valid_cases)
print("Invalid cases:", len(test_cases) - valid_cases)


# Save validation result
result = {
    "total_test_cases": len(test_cases),
    "valid_test_cases": valid_cases,
    "invalid_test_cases": len(test_cases) - valid_cases,
    "model_target": "purchase_prediction",
    "note": (
        "The supplied test cases contain four input fields and "
        "do not contain all features required by the clickstream "
        "baseline model. Therefore they are validated separately "
        "rather than passed directly to the baseline model."
    ),
}


with open(
    "../reports/test_case_validation.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(result, file, indent=4)


print("\nValidation report saved to:")
print("../reports/test_case_validation.json")