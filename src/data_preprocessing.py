# data_preprocessing.py
import pandas as pd
import numpy as np
from datetime import datetime

def load_data(train_path, stores_path):
    train = pd.read_csv(train_path)
    stores = pd.read_csv(stores_path)
    return train, stores

def preprocess_data(train, stores):
    # Merge train and stores data
    df = train.merge(stores, on='store_nbr', how='left')
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date and store
    df = df.sort_values(['date', 'store_nbr'])
    
    # Handle missing values
    df['sales'] = df['sales'].fillna(0)
    df['onpromotion'] = df['onpromotion'].fillna(0)
    
    return df