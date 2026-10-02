# UrbanClap Demand Forecasting

## Client Project

**Client:** UrbanClap (Urban Company)
**Project:** End-to-End Demand Forecasting Pipeline
**Duration:** 3 Months
**Level:** Advanced

## 1. Project Objective

The objective of this project is to build an end-to-end demand forecasting system for a home-services marketplace.

The system will use historical booking information such as:

* Service type
* City
* Date
* Time
* Weather information

The final system will forecast demand for the next 7 days for a selected city and service.

The target acceptance criterion is:

**MAPE ≤ 12% on a 4-week hold-out dataset.**

The final system will also include a Flask API, automated weekly retraining and basic data-drift monitoring.

---

## 2. Project Architecture

```text
Booking Data + Weather Data
            |
            v
     Data Ingestion
            |
            v
    Data Preprocessing
            |
            v
    Feature Engineering
            |
            v
    Time-Based Data Split
            |
       +----+----+
       |         |
       v         v
   Baseline    LSTM
     Model     Model
       |         |
       +----+----+
            |
            v
       Evaluation
   MAPE / MAE / RMSE
            |
            v
      MLflow Tracking
            |
            v
       Best Model
            |
            v
        Flask API
            |
            v
      7-Day Forecast
            |
            v
 Docker + CI/CD + Monitoring
```

---

## 3. Research Review

### Research 1 — LSTM Demand Forecasting

Research on LSTM and BiLSTM models for demand forecasting shows that recurrent neural networks can be used to learn patterns from sequential historical demand data.

**Relevance to this project:**
UrbanClap demand is also time-dependent, so historical booking patterns can be used to forecast future demand.

Reference:

https://www.sciencedirect.com/science/article/pii/S2212827121003711

---

### Research 2 — Booking Demand Forecasting

Research on booking-demand forecasting using LSTM-based approaches demonstrates the use of recurrent neural networks for predicting future booking demand.

**Relevance to this project:**
The UrbanClap system also needs to predict future service bookings.

Reference:

https://www.sciencedirect.com/science/article/pii/S0360835223007313

---

### Research 3 — Advanced LSTM Demand Forecasting

Recent research has explored attention-enhanced LSTM models for demand forecasting and compared neural-network approaches with other forecasting methods.

**Relevance to this project:**
It supports evaluating different model configurations experimentally rather than assuming that one configuration will always perform best.

Reference:

https://www.sciencedirect.com/science/article/pii/S0957417424012752

---

## 4. Planned LSTM Architecture

The initial LSTM architecture is:

```text
Input Sequence
      |
      v
LSTM Layer
64 Hidden Units
      |
      v
Dropout
0.2
      |
      v
LSTM Layer
64 Hidden Units
      |
      v
Fully Connected Layer
      |
      v
7 Output Values
      |
      v
7-Day Demand Forecast
```

The architecture is an initial configuration.

Hyperparameters will be tested and compared during the experimentation milestone.

---

## 5. Feature Engineering

The planned features include:

* Day of week
* Month
* Hour
* City
* Service type
* Lag-1 demand
* Lag-7 demand
* Lag-14 demand
* Lag-28 demand
* 7-day rolling mean
* 14-day rolling mean
* 28-day rolling mean
* Available weather variables

Lag and rolling features will use historical information only to prevent future-data leakage.

---

## 6. Data Splitting

Because this is a time-series forecasting problem, the dataset will be divided chronologically.

```text
Historical Data
      |
      +---- Training
      |
      +---- Validation
      |
      +---- 4-Week Hold-Out
```

Random train/test splitting will not be used because future information must not be used to train the model.

---

## 7. Baseline Model

A simple baseline model will be implemented before the LSTM.

The baseline will provide a reference point for measuring whether the LSTM improves forecasting performance.

The actual baseline algorithm and score will be established during Milestone 2 after inspecting the provided dataset.

---

## 8. Evaluation Metrics

### Primary Metric

**MAPE — Mean Absolute Percentage Error**

Target:

**MAPE ≤ 12%**

### Supporting Metrics

**MAE — Mean Absolute Error**

Measures the average absolute difference between actual and predicted demand.

**RMSE — Root Mean Squared Error**

Penalizes larger forecasting errors more strongly.

| Metric | Purpose                  |
| ------ | ------------------------ |
| MAPE   | Primary business metric  |
| MAE    | Average prediction error |
| RMSE   | Large-error sensitivity  |

---

## 9. Experiment Tracking

MLflow will be used during model experimentation.

The following information will be tracked:

* Learning rate
* Hidden size
* Number of LSTM layers
* Dropout
* Training loss
* Validation loss
* MAPE
* MAE
* RMSE

At least two model configurations will be evaluated during the core-model milestone.

---

## 10. Production API

The final model will be served through a Flask API.

Planned endpoints:

```text
GET /health
POST /forecast
```

Example request:

```json
{
  "city": "Bengaluru",
  "service": "Cleaning"
}
```

The API will return a 7-day demand forecast for the requested city and service.

---

## 11. MLOps Plan

The final project will include:

* Git/GitHub version control
* Automated testing
* Docker containerization
* CI/CD
* MLflow experiment tracking
* Weekly automated retraining
* Data drift monitoring
* Retraining triggers
* Production documentation

---

## 12. Repository Structure

```text
UrbanClap_Demand_Forecasting/
|
├── README.md
├── requirements.txt
├── .gitignore
|
├── data/
|   ├── raw/
|   └── processed/
|
├── notebooks/
|   └── 01_data_exploration.ipynb
|
├── src/
|   ├── __init__.py
|   ├── data_loader.py
|   ├── preprocessing.py
|   ├── features.py
|   ├── dataset.py
|   ├── model.py
|   ├── train.py
|   └── evaluate.py
|
├── models/
├── experiments/
|
├── api/
|   ├── __init__.py
|   └── app.py
|
├── tests/
|   └── test_pipeline.py
|
└── docs/
    └── architecture.md
```

---

## 13. Milestone Plan

### Milestone 1 — Research + Architecture

* Research three relevant references
* Define system architecture
* Define LSTM architecture
* Define evaluation metrics
* Create repository structure
* Create data pipeline skeleton

### Milestone 2 — Data Pipeline + Baseline

* Inspect and clean the dataset
* Build preprocessing pipeline
* Create features
* Train baseline model
* Establish benchmark

### Milestone 3 — Core Model + Experimentation

* Train LSTM
* Test at least two configurations
* Track experiments using MLflow
* Evaluate MAPE, MAE and RMSE

### Milestone 4 — Integration + Testing

* Build Flask API
* Add model inference
* Add 7-day forecasting
* Add tests
* Add Docker
* Add CI/CD

### Milestone 5 — Final Demo + Model Card

* Add weekly retraining
* Add drift monitoring
* Complete documentation
* Prepare model card
* Demonstrate complete production pipeline

---

## 14. Milestone 1 Status

**Status:** In Progress

Completed:

* Research plan
* Architecture design
* Initial LSTM architecture
* Evaluation metrics
* Repository skeleton
* Data pipeline skeleton

Next:

* Verify the actual provided dataset
* Complete data ingestion design
* Initialize Git repository
* Commit Milestone 1 work
* Submit Checkpoint 1
