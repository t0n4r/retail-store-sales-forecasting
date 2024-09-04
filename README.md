# Retail Store Sales Forecast
This project implements a time series forecasting solution for retail store sales using the "Store Sales - Time Series Forecasting" dataset. It includes data preprocessing, feature engineering, model training using Facebook's Prophet, and a Streamlit dashboard for visualizing the results.

# Project Structure
```
├── data/
│   ├── train.csv
│   ├── stores.csv
│   ├── preprocessed_data.csv
│   └── forecast_results.csv
├── diagrams/
│   ├── forecast_plot.png
│   └── forecast_components.png
├── dashboard/
│   └── app.py
├── src
│   ├──data_preprocessing.py
│   ├── feature_engineering.py
│   ├── main.py
│   └── model.py
└── README.md
```
# Files Description

```app.py```: Streamlit dashboard for visualizing the sales forecast.

```data_preprocessing.py```: Functions for loading and preprocessing the raw data.

```feature_engineering.py```: Functions for creating additional features from the data.

```main.py```: Main script to run the data preprocessing and feature engineering pipeline.

```model.py```: Script for training the Prophet model, making predictions, and evaluating the model.

# Dashboard Features
The Streamlit dashboard (app.py) provides the following features:

- Interactive date range selection
- Sales forecast visualization with confidence intervals
- Forecast statistics (average, highest, and lowest forecasted sales)
- Key insights and recommendations
- Option to view raw forecast data
- Trend and seasonality component visualizations

# Model Details
This project uses Facebook's Prophet for time series forecasting. The model is configured with the following parameters:

- Multiplicative seasonality
- Yearly and weekly seasonality enabled
- Daily seasonality disabled
- Changepoint prior scale of 0.05

The model is evaluated using cross-validation, Mean Absolute Error (MAE), and Root Mean Square Error (RMSE).


# License
This project is licensed under the MIT License - see the LICENSE file for details.