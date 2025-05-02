import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import xgboost as xgb
import joblib

def create_features(df):
    """
    Create features for the model from the combined dataset.
    Features include lagged prices, volume, treasury yield, and rolling statistics.
    """
    df = df.copy()
    # Lag features for close price
    for lag in range(1, 4):
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
    # Rolling mean and std for close price
    df['close_roll_mean_3'] = df['close'].rolling(window=3).mean()
    df['close_roll_std_3'] = df['close'].rolling(window=3).std()
    # Treasury yield lag
    df['treasury_yield_lag_1'] = df['treasury_yield'].shift(1)
    # Volume lag
    df['volume_lag_1'] = df['volume'].shift(1)
    # Drop rows with NaN values created by shifting/rolling
    df = df.dropna()
    return df

def create_target(df):
    """
    Create target variable: SPY close price 10 minutes into the future.
    Since data is at 5-minute intervals, 10 minutes ahead is 2 steps ahead.
    """
    df = df.copy()
    df['target'] = df['close'].shift(-2)
    df = df.dropna()
    return df

def train_model(df):
    """
    Train an XGBoost regression model to predict future SPY price.
    """
    features = [col for col in df.columns if col not in ['target', 'timestamp', 'open', 'high', 'low', 'close']]
    X = df[features]
    y = df['target']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    print(f"Validation R2 score: {r2:.4f}")

    return model, r2

if __name__ == "__main__":
    print("Loading combined dataset...")
    df = pd.read_csv("combined_spy_treasury_data.csv", index_col=0, parse_dates=True)
    print(f"Dataset loaded with {len(df)} records")

    print("Creating features...")
    df_feat = create_features(df)

    print("Creating target variable...")
    df_target = create_target(df_feat)

    print("Training model...")
    model, r2 = train_model(df_target)

    if r2 >= 0.64:
        print(f"Model achieved required accuracy: {r2:.4f}")
    else:
        print(f"Model accuracy below target: {r2:.4f}. Consider further tuning or more data.")

    print("Saving model to spy_price_model.pkl")
    joblib.dump(model, "spy_price_model.pkl")
