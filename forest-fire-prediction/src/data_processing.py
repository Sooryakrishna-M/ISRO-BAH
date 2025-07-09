# src/data_processing.py

import pandas as pd
import os

def load_data(path='data/raw/fire_archive_M6_96619.csv'):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df = df[['latitude', 'longitude', 'acq_date', 'brightness', 'confidence', 'frp']]
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    df = df.dropna()
    df['high_risk'] = df['confidence'].apply(lambda x: 1 if x > 70 else 0)
    return df

if __name__ == "__main__":
    df = load_data()
    df_cleaned = preprocess_data(df)
    os.makedirs('data/processed', exist_ok=True)
    df_cleaned.to_csv('data/processed/cleaned_data.csv', index=False)
