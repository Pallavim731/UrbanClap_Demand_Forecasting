\# Milestone 1 Research – Demand Forecasting



\## 1. Research Objective



The objective of this research is to understand suitable time-series forecasting methods for predicting demand in a home-services marketplace.



The project requires forecasting future bookings using information such as:



\* Service type

\* City

\* Date

\* Time

\* Weather

\* Historical booking demand



LSTM networks were selected as the primary deep-learning approach because they are designed for sequential/time-series data and can learn relationships between previous observations and future demand.



\---



\## 2. Reference 1 – LSTM/BiLSTM Demand Forecasting



\*\*Source:\*\* ScienceDirect



\*\*Topic:\*\* LSTM and BiLSTM approaches for demand forecasting in supply-chain applications.



\*\*Key learning:\*\*



\* LSTM models can be applied to sequential demand data.

\* Historical demand patterns can be used to predict future demand.

\* Time-series forecasting requires careful handling of chronological data.

\* Different model configurations can be compared using forecasting metrics.



\*\*Relevance to this project:\*\*



UrbanClap booking demand is also time-dependent. Previous booking patterns can therefore be used as input sequences for the LSTM model.



\---



\## 3. Reference 2 – Booking Demand Forecasting Using LSTM



\*\*Source:\*\* ScienceDirect



\*\*Topic:\*\* Booking demand forecasting using LSTM-based models.



\*\*Key learning:\*\*



\* Booking data can contain strong temporal patterns.

\* LSTM models are suitable for learning patterns from historical booking sequences.

\* Forecasting performance should be evaluated on future/unseen periods.



\*\*Relevance to this project:\*\*



The UrbanClap project specifically requires forecasting future bookings. A chronological train/validation/test split will therefore be used instead of a random split.



\---



\## 4. Reference 3 – Attention-Enhanced LSTM Forecasting



\*\*Source:\*\* ScienceDirect



\*\*Topic:\*\* Demand forecasting using attention-enhanced LSTM models.



\*\*Key learning:\*\*



\* LSTM-based forecasting can be improved by focusing on



