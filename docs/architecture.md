\# UrbanClap Demand Forecasting — System Architecture



\## 1. Project Objective



The objective is to build an end-to-end demand forecasting system for a home-services marketplace.



The system will use historical booking information such as service type, city, date/time and weather-related information to forecast demand for the next 7 days for a selected city and service.



The target acceptance criterion is a MAPE of 12% or lower on a 4-week hold-out dataset.



\## 2. Data Pipeline



The planned pipeline is:



Raw booking data

→ Data cleaning

→ Date/time processing

→ Aggregation by city and service

→ Feature engineering

→ Time-based train/validation/test split

→ Baseline model

→ LSTM model

→ Evaluation

→ Model selection

→ Flask API



\## 3. Feature Engineering



The planned features include:



\* Day of week

\* Month

\* Hour/time information

\* Lag-1 demand

\* Lag-7 demand

\* Lag-14 demand

\* Lag-28 demand

\* 7-day rolling mean

\* 14-day rolling mean

\* 28-day rolling mean

\* Available weather variables

\* City

\* Service type



Lag and rolling features will be calculated using historical information only to avoid future-data leakage.



\## 4. Model Architecture



The primary model will be a PyTorch LSTM.



Initial architecture:



Input sequence

→ LSTM layer with 64 hidden units

→ Dropout 0.2

→ LSTM layer with 64 hidden units

→ Fully connected layer

→ 7 output values



The seven output values represent the predicted demand for the next seven days.



The initial architecture is a starting configuration. Hyperparameters will be experimentally evaluated during Milestone 3.



\## 5. Forecasting Strategy



The system will use direct multi-step forecasting.



For each city-service combination, historical observations will be converted into sliding sequences.



Example:



Historical window

→ LSTM

→ Day 1 forecast

→ Day 2 forecast

→ Day 3 forecast

→ Day 4 forecast

→ Day 5 forecast

→ Day 6 forecast

→ Day 7 forecast



\## 6. Data Splitting



Because this is a time-series problem, the data will be split chronologically.



The project will use:



\* Training period

\* Validation period

\* Final 4-week hold-out period



Future observations must not be used during training or preprocessing.



\## 7. Baseline



Before training the LSTM, a simple baseline model will be established.



The baseline will provide a reference point for evaluating whether the LSTM provides meaningful improvement.



\## 8. Evaluation Metrics



Primary metric:



\* MAPE



Supporting metrics:



\* MAE

\* RMSE



The final model will be evaluated on the 4-week hold-out period.



\## 9. Experiment Tracking



MLflow will be used during model experimentation to record:



\* Model configuration

\* Learning rate

\* Hidden size

\* Number of LSTM layers

\* Dropout

\* Training loss

\* Validation loss

\* MAPE

\* MAE

\* RMSE



\## 10. Production Architecture



After model development, the selected model will be saved and loaded by a Flask API.



The API will provide:



GET /health



POST /forecast



The forecast endpoint will accept a city and service and return a 7-day forecast.



\## 11. Future MLOps Components



The later milestones will add:



\* Docker containerization

\* Automated testing

\* CI/CD

\* Weekly retraining

\* Model monitoring

\* Data drift detection

\* Retraining triggers

\* Production documentation



\## 12. Expected Final Flow



Booking Data + Weather Data

→ Data Pipeline

→ Feature Engineering

→ Baseline

→ PyTorch LSTM

→ MLflow Experiment Tracking

→ Best Model

→ Flask API

→ 7-Day Forecast

→ Docker

→ Weekly Retraining

→ Drift Monitoring



