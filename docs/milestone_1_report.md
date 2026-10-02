\# Milestone 1 – Research and Architecture Design



\## Project



\*\*UrbanClap Demand Forecasting – Home Services Marketplace\*\*



\## Objective



The objective of this project is to build an end-to-end demand forecasting system for a home services marketplace.



The system will use historical booking information such as:



\* Service type

\* City

\* Date and time

\* Weather information

\* Historical booking demand



The final system is planned to generate a 7-day demand forecast for a selected city and service.



\---



\## 1. Research Completed



Relevant research on time-series and demand forecasting was reviewed.



The research focused on:



1\. LSTM/BiLSTM demand forecasting

2\. Booking demand forecasting using LSTM

3\. Attention-enhanced LSTM forecasting

4\. PyTorch-based LSTM time-series forecasting



The research indicates that LSTM-based models are suitable for learning patterns from sequential historical demand data.



Detailed references are available in:



`docs/research.md`



\---



\## 2. Proposed Architecture



The planned system architecture is:



```text

Booking Data + Weather

&#x20;         |

&#x20;         v

Data Loading

&#x20;         |

&#x20;         v

Data Cleaning

&#x20;         |

&#x20;         v

Feature Engineering

&#x20;         |

&#x20;         +----------------------+

&#x20;         |                      |

&#x20;         v                      v

&#x20;     Lag Features         Rolling Features

&#x20;         |                      |

&#x20;         +----------+-----------+

&#x20;                    |

&#x20;                    v

&#x20;             LSTM Sequences

&#x20;                    |

&#x20;                    v

&#x20;                LSTM Model

&#x20;                    |

&#x20;                    v

&#x20;             7-Day Forecast

&#x20;                    |

&#x20;                    v

&#x20;                Evaluation

```



The complete production architecture will later include MLflow, Flask API, Docker, weekly retraining, and drift monitoring.



\---



\## 3. Feature Engineering Plan



The planned features include:



\### Calendar Features



\* Day of week

\* Month

\* Hour



\### Historical Demand Features



\* Lag 1

\* Lag 7

\* Lag 14

\* Lag 28



\### Rolling Features



\* 7-period rolling mean

\* 14-period rolling mean

\* 28-period rolling mean



\### Additional Features



\* City

\* Service type

\* Weather variables



The exact feature columns will be finalized after the actual client dataset is inspected.



\---



\## 4. LSTM Architecture



The initial model architecture is:



```text

Input Sequence

&#x20;     |

&#x20;     v

LSTM - 64 Hidden Units

&#x20;     |

&#x20;     v

Dropout - 0.2

&#x20;     |

&#x20;     v

LSTM - 64 Hidden Units

&#x20;     |

&#x20;     v

Fully Connected Layer

&#x20;     |

&#x20;     v

7-Day Forecast

```



The model is implemented using PyTorch.



The implementation is available in:



`src/model.py`



\---



\## 5. Time-Series Splitting Strategy



The project will use chronological splitting rather than random splitting.



Planned structure:



```text

Historical Data

&#x20;     |

&#x20;     +---- Training Data

&#x20;     |

&#x20;     +---- Validation Data

&#x20;     |

&#x20;     +---- Final 4-Week Holdout

```



The final four-week period will be kept as an unseen hold-out set for final evaluation.



\---



\## 6. Evaluation Metrics



The primary metric specified by the client is:



\*\*MAPE ≤ 12%\*\*



Supporting metrics:



\* MAE

\* RMSE



The evaluation functions are prepared in:



`src/evaluate.py`



\---



\## 7. Data Pipeline Skeleton



The following modules have been created:



```text

src/

├── data\_loader.py

├── preprocessing.py

├── features.py

├── dataset.py

├── model.py

└── evaluate.py

```



Their responsibilities are:



\* `data\_loader.py` – load raw booking data

\* `preprocessing.py` – clean data and create datetime features

\* `features.py` – create lag and rolling features

\* `dataset.py` – create LSTM input sequences

\* `model.py` – define the PyTorch LSTM

\* `evaluate.py` – calculate forecasting metrics



\---



\## 8. Repository Structure



```text

UrbanClap\_Demand\_Forecasting/

│

├── README.md

├── requirements.txt

├── .gitignore

│

├── data/

│   ├── raw/

│   └── processed/

│

├── notebooks/

│

├── src/

│   ├── \_\_init\_\_.py

│   ├── data\_loader.py

│   ├── preprocessing.py

│   ├── features.py

│   ├── dataset.py

│   ├── model.py

│   ├── train.py

│   └── evaluate.py

│

├── models/

├── experiments/

│

├── api/

│   ├── \_\_init\_\_.py

│   └── app.py

│

├── tests/

│   └── test\_pipeline.py

│

└── docs/

&#x20;   ├── architecture.md

&#x20;   ├── research.md

&#x20;   └── milestone\_1\_report.md

```



\---



\## 9. Milestone 1 Status



| Requirement                       | Status    |

| --------------------------------- | --------- |

| Research relevant papers/repos    | Completed |

| Define model architecture         | Completed |

| Define evaluation metrics         | Completed |

| Create project repository         | Completed |

| Create README                     | Completed |

| Create data pipeline skeleton     | Completed |

| Create research documentation     | Completed |

| Create architecture documentation | Completed |

| Git commit                        | Completed |



\## Milestone 1 Conclusion



Milestone 1 has been completed with the research, architecture, evaluation strategy, repository structure, and initial data-pipeline/model skeleton prepared.



The next milestone will focus on the actual client dataset, complete preprocessing, baseline forecasting, and benchmark evaluation.



