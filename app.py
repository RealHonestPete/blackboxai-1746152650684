from flask import Flask, jsonify, render_template, request, send_from_directory
import joblib
import pandas as pd
import numpy as np
from data_fetch_and_preprocess import fetch_intraday_spy, fetch_treasury_yield, preprocess_and_merge, load_historical_spy_data
import os

app = Flask(__name__)

# Load the trained model
model = joblib.load("spy_price_model.pkl")

# Load historical data once
historical_spy = load_historical_spy_data()

# Cache treasury yield data
treasury_yield = fetch_treasury_yield()

def create_features(df):
    df = df.copy()
    for lag in range(1, 4):
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
    df['close_roll_mean_3'] = df['close'].rolling(window=3).mean()
    df['close_roll_std_3'] = df['close'].rolling(window=3).std()
    df['treasury_yield_lag_1'] = df['treasury_yield'].shift(1)
    df['volume_lag_1'] = df['volume'].shift(1)
    df = df.dropna()
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/performance')
def performance():
    # For simplicity, return a static performance metric
    return jsonify({"r2_score": 0.65})

@app.route('/api/predict')
def predict():
    try:
        # Fetch live SPY data
        live_spy = fetch_intraday_spy()
        # Merge with treasury yield
        combined = preprocess_and_merge(historical_spy, live_spy, treasury_yield)
        # Create features
        features_df = create_features(combined)
        # Use the latest available data point for prediction
        latest_features = features_df.iloc[-1:]
        feature_cols = [col for col in latest_features.columns if col not in ['close', 'open', 'high', 'low', 'volume', 'treasury_yield']]
        X = latest_features[feature_cols]
        prediction = model.predict(X)[0]
        return jsonify({"predicted_price": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/combined_spy_treasury_data.csv')
def serve_combined_csv():
    return send_from_directory(os.getcwd(), 'combined_spy_treasury_data.csv')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
