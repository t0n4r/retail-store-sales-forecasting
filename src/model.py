import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def prepare_data_for_prophet(df):
    # Aggregate sales data by date
    df_prophet = df.groupby('date')['sales'].sum().reset_index()
    df_prophet.columns = ['ds', 'y']
    return df_prophet

def train_prophet_model(df_prophet):
    model = Prophet(seasonality_mode='multiplicative',
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.05)
    model.fit(df_prophet)
    return model

def make_predictions(model, periods):
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def evaluate_model(model, df_prophet):
    # Perform cross-validation
    cv_results = cross_validation(model, initial='730 days', period='180 days', horizon='90 days')
    cv_metrics = performance_metrics(cv_results)
    
    # Calculate MAE and RMSE
    y_true = df_prophet['y']
    y_pred = model.predict(df_prophet[['ds']])['yhat']
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return cv_metrics, mae, rmse

def plot_forecast(model, forecast):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(model.history['ds'].values, model.history['y'].values, 'k.', label='Observed')
    ax.plot(forecast['ds'].values, forecast['yhat'].values, 'b-', label='Forecast')
    ax.fill_between(forecast['ds'].values, forecast['yhat_lower'].values, forecast['yhat_upper'].values, color='b', alpha=0.2)
    ax.set_title('Sales Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend()
    plt.savefig('diagrams/forecast_plot.png')
    plt.close()

def plot_components(model, forecast):
    components = ['trend', 'yearly', 'weekly']
    fig, axes = plt.subplots(len(components), 1, figsize=(12, 4*len(components)), sharex=True)
    
    for ax, component in zip(axes, components):
        if component == 'trend':
            ax.plot(forecast['ds'].values, forecast['trend'].values)
            ax.set_ylabel('Trend')
        elif component == 'yearly':
            yearly = forecast['yearly'].values
            ax.plot(forecast['ds'].values, yearly)
            ax.set_ylabel('Yearly')
        elif component == 'weekly':
            weekly = forecast['weekly'].values
            ax.plot(forecast['ds'].values, weekly)
            ax.set_ylabel('Weekly')
        
        ax.set_title(f'{component.capitalize()} Component')
    
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig('diagrams/forecast_components.png')
    plt.close()

def main():
    # Load preprocessed data
    df = pd.read_csv('data/preprocessed_data.csv', parse_dates=['date'])
    
    # Prepare data for Prophet
    df_prophet = prepare_data_for_prophet(df)
    
    # Train model
    model = train_prophet_model(df_prophet)
    
    # Make predictions
    forecast = make_predictions(model, periods=90)
    
    # Evaluate model
    cv_metrics, mae, rmse = evaluate_model(model, df_prophet)
    
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print("Cross-validation metrics:")
    print(cv_metrics)
    
    # Plot forecast
    plot_forecast(model, forecast)
    
    # Plot components
    plot_components(model, forecast)
    
    # Save forecast to CSV
    forecast.to_csv('forecast_results.csv', index=False)

if __name__ == "__main__":
    main()