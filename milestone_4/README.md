\# Milestone 4 - Integration + Testing



\## Project

UrbanClap Demand Forecasting



\## Objective



Integrate the trained Milestone 3 machine learning model into a serving layer and validate the API through integration testing and concurrent load testing.



\## Serving Layer



The trained model is served using a Flask REST API.



\### API URL



http://127.0.0.1:8000



\### Endpoints



\#### GET /



Checks whether the API is running.



\#### GET /health



Checks whether the machine learning model is loaded successfully.



Example response:



```json

{

&#x20;   "model\_loaded": true,

&#x20;   "status": "healthy"

}

