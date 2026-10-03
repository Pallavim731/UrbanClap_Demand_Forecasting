\# UrbanClap Demand Forecasting — Milestone 2



\## Data Pipeline + Baseline Model



\### 1. Objective



The objective of Milestone 2 was to build a complete data ingestion and preprocessing pipeline, train a baseline machine-learning model, evaluate its performance, and establish a benchmark for future model improvements.



\---



\## 2. Dataset



The main dataset used for the baseline model is:



`data/clickstream.csv`



Dataset size:



\* 1,000 records

\* 16 columns

\* Target column: `label\_purchased`



Target distribution:



\* Purchased (`1`): 556

\* Not purchased (`0`): 444



The dataset contains information about user sessions, devices, categories, product interaction, session duration, discounts, searches, and recommendation clicks.



\---



\## 3. Data Preprocessing



The preprocessing pipeline is implemented in:



`src/preprocess.py`



The pipeline performs the following steps:



1\. Loads the clickstream dataset.

2\. Separates features and target.

3\. Removes identifier columns from the model features.

4\. Handles missing values in `search\_query`.

5\. Replaces missing search queries with `No Search`.

6\. Separates numerical and categorical features.

7\. Applies One-Hot Encoding to categorical features.

8\. Splits the dataset into training and testing sets.

9\. Uses an 80/20 train-test split.

10\. Uses stratification to preserve the target distribution.

11\. Saves the processed datasets.



Generated files:



\* `data/processed/X\_train.csv`

\* `data/processed/X\_test.csv`

\* `data/processed/y\_train.csv`

\* `data/processed/y\_test.csv`



\---



\## 4. Baseline Model



The baseline model is implemented in:



`src/train.py`



Model used:



\*\*Logistic Regression\*\*



Configuration:



\* `max\_iter = 1000`

\* `random\_state = 42`



The trained model is saved as:



`models/baseline\_logistic\_regression.pkl`



\---



\## 5. Model Evaluation



The evaluation is implemented in:



`src/evaluate.py`



The following metrics are calculated:



\* Accuracy

\* Precision

\* Recall

\* F1-score

\* Confusion Matrix

\* Classification Report



The baseline evaluation report is saved as:



`reports/baseline\_report.txt`



The metrics in this file are generated directly from the trained model and test dataset.



\---



\## 6. Testing



Automated tests are provided in:



`tests/test\_pipeline.py`



The tests verify:



\* Dataset availability

\* Dataset dimensions

\* Processed training data

\* Processed testing data

\* Valid target values

\* Trained model availability

\* Model prediction functionality



The supplied JSON test cases are validated using:



`src/validate\_test\_cases.py`



The test cases are stored in:



`data/test\_cases.json`



There are 50 test cases.



The test-case validation report is saved as:



`reports/test\_case\_validation.json`



The supplied JSON test cases use four input fields:



\* `category`

\* `user\_device`

\* `session\_mins`

\* `prior\_purchases`



These fields do not exactly match the complete feature set used by the clickstream baseline model. Therefore, the JSON cases are structurally validated separately rather than incorrectly passing incompatible inputs into the baseline model.



\---



\## 7. Project Structure



```text

UrbanClap\_Demand\_Forecasting

│

├── milestone\_1

│   └── existing milestone 1 files

│

└── milestone\_2

&#x20;   │

&#x20;   ├── data

&#x20;   │   ├── clickstream.csv

&#x20;   │   ├── product\_reviews.csv

&#x20;   │   ├── test\_cases.json

&#x20;   │   └── processed

&#x20;   │       ├── X\_train.csv

&#x20;   │       ├── X\_test.csv

&#x20;   │       ├── y\_train.csv

&#x20;   │       └── y\_test.csv

&#x20;   │

&#x20;   ├── src

&#x20;   │   ├── preprocess.py

&#x20;   │   ├── train.py

&#x20;   │   ├── evaluate.py

&#x20;   │   └── validate\_test\_cases.py

&#x20;   │

&#x20;   ├── models

&#x20;   │   └── baseline\_logistic\_regression.pkl

&#x20;   │

&#x20;   ├── reports

&#x20;   │   ├── baseline\_report.txt

&#x20;   │   └── test\_case\_validation.json

&#x20;   │

&#x20;   ├── tests

&#x20;   │   └── test\_pipeline.py

&#x20;   │

&#x20;   └── README.md

```



\---



\## 8. Tools and Technologies



\* Python

\* Pandas

\* Scikit-learn

\* Joblib

\* Pytest

\* JSON

\* CSV



\---



\## 9. Milestone Outcome



A complete baseline machine-learning pipeline has been established.



The pipeline now supports:



\*\*Data ingestion → preprocessing → train/test split → model training → evaluation → testing → reporting\*\*



The Logistic Regression results stored in `reports/baseline\_report.txt` provide the initial benchmark that can be used to compare future models.



