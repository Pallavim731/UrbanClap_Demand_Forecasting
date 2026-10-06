\# UrbanClap Demand Forecasting — Model Card



\## 1. Model Overview



This model is the primary machine learning model developed for the UrbanClap project. A Logistic Regression classifier was trained and evaluated during the experimentation stage.



The model was selected after comparing multiple hyperparameter configurations using MLflow.



\## 2. Intended Use



The model is intended to demonstrate an end-to-end machine learning workflow including:



\* Data preprocessing

\* Model training

\* Hyperparameter experimentation

\* Model evaluation

\* Model serving through a REST API

\* Integration testing

\* Load testing



The project is intended for internship evaluation and demonstration purposes.



\## 3. Data



The project uses the processed dataset prepared during the data pipeline stage.



The model training data contained 800 training samples and 53 input features. The test data contained 200 samples with the same 53 input features.



Data preprocessing and feature preparation were performed before model training.



\## 4. Model



The primary model is Logistic Regression.



The main hyperparameter evaluated was:



\* C = 0.1

\* C = 1.0

\* C = 10.0



The best observed configuration was:



\* Model: Logistic Regression

\* C: 0.1



\## 5. Evaluation Metrics



The best observed configuration produced the following results:



| Metric    |  Score |

| --------- | -----: |

| Accuracy  | 0.5300 |

| Precision | 0.5535 |

| Recall    | 0.7928 |

| F1 Score  | 0.6519 |



The model was selected based on the observed experimental performance.



\## 6. Experiment Tracking



MLflow was used to track the model experiments and hyperparameter configurations.



The experiments included different values of the Logistic Regression C parameter.



The tracked experiments helped compare configurations and identify the best observed configuration.



\## 7. Deployment



The trained model was integrated into a Flask REST API.



The API provides:



\* `/` — API status

\* `/health` — model health check

\* `/predict` — prediction endpoint



The prediction endpoint accepts a JSON payload containing the required feature values.



Example:



{

"features": \[0.0, 0.0, 0.0, "... 53 values ..."]

}



\## 8. Testing



Integration testing was performed using Pytest.



Results:



\* Total tests: 5

\* Passed: 5

\* Failed: 0



The tests covered:



\* Home endpoint

\* Health endpoint

\* Prediction endpoint

\* Missing feature validation

\* Invalid feature validation



\## 9. Load Testing



The API was tested with 50 concurrent requests.



Observed results:



| Metric              |                 Result |

| ------------------- | ---------------------: |

| Total requests      |                     50 |

| Concurrent requests |                     50 |

| Successful requests |                     50 |

| Failed requests     |                      0 |

| Average latency     |              164.29 ms |

| Minimum latency     |               24.37 ms |

| Maximum latency     |              268.44 ms |

| Throughput          | 149.57 requests/second |



The API successfully handled all 50 concurrent requests during the test.



\## 10. Bias Audit



A basic bias and fairness review was considered as part of the model documentation.



The available project dataset does not provide sufficient evidence for a complete demographic fairness analysis. Therefore, fairness across sensitive demographic groups cannot be conclusively measured from the current experiment.



Potential risks include:



\* Dataset imbalance

\* Differences in representation between groups

\* Historical patterns in the training data

\* Performance differences that may not be visible in aggregate metrics



For production deployment, group-level evaluation should be performed when appropriate and legally permitted.



\## 11. Limitations



The current model has several limitations:



1\. The observed accuracy is 53%, so the model should not be considered production-ready without further improvement.

2\. The evaluation dataset is limited in size.

3\. The current experiment does not provide a complete fairness evaluation across demographic groups.

4\. The model depends on the preprocessing and feature representation used during training.

5\. The Flask development server should not be used as the production serving infrastructure.

6\. Additional monitoring and retraining would be required for production use.

7\. The current model should be further validated on representative real-world data.



\## 12. Deployment Guide



\### Install dependencies



Create and activate a Python virtual environment and install the required project dependencies.



\### Start the API



From the project root:



python milestone\_4\\api\\app.py



The API runs at:



http://127.0.0.1:8000



\### Health check



Open:



http://127.0.0.1:8000/health



A healthy response indicates that the model has loaded successfully.



\### Prediction



Send a POST request to:



http://127.0.0.1:8000/predict



with the required feature vector.



\## 13. Monitoring Recommendations



For production deployment, the following should be monitored:



\* Prediction latency

\* Request throughput

\* API errors

\* Model prediction distribution

\* Data drift

\* Model performance

\* Retraining frequency



\## 14. Conclusion



The project demonstrates an end-to-end machine learning workflow from data preparation and model experimentation to API integration, testing, and performance evaluation.



The best observed Logistic Regression configuration used C=0.1 and achieved an F1 score of 0.6519.



The deployed API successfully handled 50 concurrent requests with zero failures, an average latency of 164.29 ms, and throughput of 149.57 requests/second.



Further model improvement and production-level validation are recommended before real-world deployment.



