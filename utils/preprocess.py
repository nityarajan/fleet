import pandas as pd
import numpy as np
import re
import pickle
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load preprocessing objects and model
model = joblib.load("model/rf_fraud_model_final.pkl")
vehicle_encoder = pickle.load(open("model/vehicle_encoder.pkl", "rb"))
driver_encoder = pickle.load(open("model/driver_encoder.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

def clean_input_data(df):
    if 'driver_uid' in df.columns:
        df['driver_id'] = df['driver_uid'].str.extract(r'(^[^#]+)')

    for col in ['Account Number', 'Card Number', 'Vehicle Number', 'driver_id']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    if 'Transaction Date' in df.columns and 'Transaction Time' in df.columns:
        df['TransactionDateTime'] = pd.to_datetime(df['Transaction Date'] + ' ' + df['Transaction Time'])

    if 'Product Description' in df.columns:
        df['Product Description'] = df['Product Description'].str.replace(r'(?i)\bDIESEL\b.*', 'DIESEL', regex=True)
        df['Product Description'] = df['Product Description'].str.replace(r'(?i)\bUNLEADED\b.*', 'PETROL', regex=True)
        df['Product Description'] = df['Product Description'].str.replace(r'(?i)\bUNLD\b.*', 'PETROL', regex=True)
        df = df[df['Product Description'].isin(['PETROL', 'DIESEL'])]
        df = df.rename(columns={'Product Description': 'Fuel Type'})

    if 'Gallons Over' in df.columns:
        df['Gallons Over'] = df['Gallons Over'].fillna(0)

    expected_cols = ["transaction_uid", 'Vehicle Number', 'Units', 'driver_id', 'TransactionDateTime', 'Fuel Type',
                     'Amount', 'Actual Odometer', 'Gallons Over']
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[expected_cols]

def preprocess(dfx):
    df = dfx.copy()
    
    # Keep the UID column separately
    df['transaction_uid'] = dfx['transaction_uid']

    # Encoding
    df['Vehicle Number Encoded'] = df['Vehicle Number'].apply(
        lambda x: vehicle_encoder.transform([x])[0] if x in vehicle_encoder.classes_ else -1
    )
    df['Driver ID Encoded'] = df['driver_id'].apply(
        lambda x: driver_encoder.transform([x])[0] if x in driver_encoder.classes_ else -1
    )
    df['Fuel Type_PETROL'] = (df['Fuel Type'] == 'PETROL').astype(int)

    # Datetime features
    df['TransactionDateTime'] = pd.to_datetime(df['TransactionDateTime'])
    df['Transaction_Year'] = df['TransactionDateTime'].dt.year
    df['Transaction_Month'] = df['TransactionDateTime'].dt.month
    df['Transaction_Day'] = df['TransactionDateTime'].dt.day
    df['Transaction_Hour'] = df['TransactionDateTime'].dt.hour
    df['Transaction_Weekday'] = df['TransactionDateTime'].dt.weekday
    df['Is_Weekend'] = df['Transaction_Weekday'].isin([5, 6]).astype(int)

    # Cyclical encoding
    df['Transaction_Hour_sin'] = np.sin(2 * np.pi * df['Transaction_Hour'] / 24)
    df['Transaction_Hour_cos'] = np.cos(2 * np.pi * df['Transaction_Hour'] / 24)
    df['Transaction_Day_sin'] = np.sin(2 * np.pi * df['Transaction_Day'] / 31)
    df['Transaction_Day_cos'] = np.cos(2 * np.pi * df['Transaction_Day'] / 31)
    df['Transaction_Month_sin'] = np.sin(2 * np.pi * df['Transaction_Month'] / 12)
    df['Transaction_Month_cos'] = np.cos(2 * np.pi * df['Transaction_Month'] / 12)

    # Scaling
    features_to_scale = ['Amount', 'Actual Odometer', 'Units', 'Gallons Over']
    df[features_to_scale] = scaler.transform(df[features_to_scale])

    return df

def and_predict(df_preprocessed, dfx):
    df = df_preprocessed.copy()

    feature_cols = ['Units', 'Amount', 'Actual Odometer', 'Gallons Over',
                    'Vehicle Number Encoded', 'Driver ID Encoded', 'Fuel Type_PETROL',
                    'Transaction_Year', 'Transaction_Weekday', 'Is_Weekend',
                    'Transaction_Hour_sin', 'Transaction_Hour_cos',
                    'Transaction_Day_sin', 'Transaction_Day_cos',
                    'Transaction_Month_sin', 'Transaction_Month_cos']
    
    X = df[feature_cols]
    predictions = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    # Build result dataframe
    result_df = dfx.copy()
    
    result_df['Fraud_Prediction'] = predictions
    result_df['Fraud_Prediction_Probability'] = probs

    return result_df

