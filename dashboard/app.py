import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Load data
@st.cache_data
def load_data():
    forecast = pd.read_csv('data/forecast_results.csv', parse_dates=['ds'])
    return forecast

def main():
    st.title('Retail Store Sales Forecast Dashboard')

    forecast = load_data()

    # Sidebar
    st.sidebar.header('Dashboard Controls')
    min_date = forecast['ds'].min().date()
    max_date = forecast['ds'].max().date()
    date_range = st.sidebar.date_input("Select Date Range", 
                                       [min_date, max_date],
                                       min_value=min_date,
                                       max_value=max_date)

    # Filter data based on selected date range
    start_date, end_date = date_range
    start_datetime = pd.Timestamp(start_date)
    end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    
    forecast_filtered = forecast[(forecast['ds'] >= start_datetime) & (forecast['ds'] <= end_datetime)]

    # Main content
    st.header('Sales Forecast')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_filtered['ds'], y=forecast_filtered['yhat'], mode='lines', name='Forecast', line=dict(color='blue')))
    if 'yhat_lower' in forecast_filtered.columns and 'yhat_upper' in forecast_filtered.columns:
        fig.add_trace(go.Scatter(x=forecast_filtered['ds'], y=forecast_filtered['yhat_upper'], mode='lines', line=dict(color='rgba(0,0,255,0.3)', width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_filtered['ds'], y=forecast_filtered['yhat_lower'], mode='lines', line=dict(color='rgba(0,0,255,0.3)', width=0), 
                                 fill='tonexty', fillcolor='rgba(255,99,71,0.8)', name='Confidence Interval'))
    fig.update_layout(title='Sales Forecast', xaxis_title='Date', yaxis_title='Sales')
    st.plotly_chart(fig)

    # Forecast Statistics
    st.header('Forecast Statistics')
    st.write(f"Average Forecasted Sales: ${forecast_filtered['yhat'].mean():,.2f}")
    st.write(f"Highest Forecasted Sales: ${forecast_filtered['yhat'].max():,.2f}")
    st.write(f"Lowest Forecasted Sales: ${forecast_filtered['yhat'].min():,.2f}")

    # Key Insights
    st.header('Key Insights')
    st.write("""
    1. The business is growing steadily over time, with accelerated growth in recent years.
    2. There are strong yearly seasonal patterns, likely due to holiday shopping.
    3. Weekly patterns are consistent and significant, suggesting certain days consistently outperform others.
    4. The forecast model captures these trends and patterns well, providing a reasonable prediction of future sales.
    """)

    # Recommendations
    st.header('Recommendations')
    st.write("""
    1. Plan for continued growth, but be prepared for the possibility of growth rate changes.
    2. Optimize inventory and staffing for peak seasons, especially during the annual high points.
    3. Adjust daily operations to account for weekly patterns, possibly increasing resources on high-performing days.
    4. Use the confidence intervals to prepare for best-case and worst-case scenarios in future planning.
    """)

    # Show raw forecast data
    if st.checkbox('Show raw forecast data'):
        st.subheader('Raw Forecast Data')
        st.write(forecast_filtered)

    # Additional components if available
    if 'trend' in forecast_filtered.columns:
        st.header('Trend Component')
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=forecast_filtered['ds'], y=forecast_filtered['trend'], mode='lines', name='Trend'))
        fig_trend.update_layout(title='Sales Trend', xaxis_title='Date', yaxis_title='Trend')
        st.plotly_chart(fig_trend)

    if 'yearly' in forecast_filtered.columns:
        st.header('Yearly Seasonality')
        fig_yearly = go.Figure()
        fig_yearly.add_trace(go.Scatter(x=forecast_filtered['ds'], y=forecast_filtered['yearly'], mode='lines', name='Yearly Seasonality'))
        fig_yearly.update_layout(xaxis_title='Date', yaxis_title='Yearly Effect')
        st.plotly_chart(fig_yearly)

    if 'weekly' in forecast_filtered.columns:
        st.header('Weekly Seasonality')
        fig_weekly = go.Figure()
        fig_weekly.add_trace(go.Scatter(x=forecast_filtered['ds'], y=forecast_filtered['weekly'], mode='lines', name='Weekly Seasonality'))
        fig_weekly.update_layout(xaxis_title='Date', yaxis_title='Weekly Effect')
        st.plotly_chart(fig_weekly)

if __name__ == "__main__":
    main()