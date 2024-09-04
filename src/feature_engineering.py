def engineer_features(df):
    # Extract date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Create lag features
    df['sales_lag_1'] = df.groupby(['store_nbr', 'family'])['sales'].shift(1)
    df['sales_lag_7'] = df.groupby(['store_nbr', 'family'])['sales'].shift(7)
    
    # Create rolling mean features
    df['sales_rolling_7'] = df.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['sales_rolling_30'] = df.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    
    return df