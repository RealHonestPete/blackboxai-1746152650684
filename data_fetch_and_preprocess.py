import requests
import pandas as pd
from datetime import datetime, timedelta
import time

ALPHA_VANTAGE_API_KEY = "GZ755Y5DG4OE1809"

def fetch_treasury_yield():
    """
    Fetch daily 10-year US Treasury yield data from Alpha Vantage.
    Returns a DataFrame with date and yield.
    """
    url = f"https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity=10year&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if "data" in data:
        df = pd.DataFrame(data["data"])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'])
        return df[['date', 'value']]
    else:
        raise ValueError("Failed to fetch Treasury yield data: " + str(data))

def fetch_intraday_spy():
    """
    Fetch intraday SPY data from Alpha Vantage.
    Returns a DataFrame with timestamp and OHLCV data.
    """
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=SPY&interval=5min&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()
    time_series_key = "Time Series (5min)"
    if time_series_key in data:
        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume"
        })
        df = df.astype({
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": int
        })
        df = df.sort_index()
        return df
    else:
        raise ValueError("Failed to fetch intraday SPY data: " + str(data))

def load_historical_spy_data(filepath="SPY 4-2025 data.csv"):
    """
    Load historical SPY intraday data from CSV file.
    """
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df = df.set_index('timestamp')
    return df

def preprocess_and_merge(historical_df, live_spy_df, treasury_df):
    """
    Merge historical SPY, live SPY, and Treasury yield data on timestamps.
    Align Treasury yield data by date to intraday timestamps.
    """
    # Combine historical and live SPY data
    combined_spy = pd.concat([historical_df, live_spy_df])
    combined_spy = combined_spy[~combined_spy.index.duplicated(keep='last')]
    combined_spy = combined_spy.sort_index()

    # Map Treasury yield to each timestamp by date
    treasury_df['date'] = treasury_df['date'].dt.date
    combined_spy['date'] = combined_spy.index.date
    combined_spy = combined_spy.merge(treasury_df, how='left', left_on='date', right_on='date')
    combined_spy = combined_spy.rename(columns={'value': 'treasury_yield'})
    combined_spy = combined_spy.drop(columns=['date'])

    # Forward fill treasury yield for missing values
    combined_spy['treasury_yield'] = combined_spy['treasury_yield'].fillna(method='ffill')

    return combined_spy

if __name__ == "__main__":
    print("Loading historical SPY data...")
    historical_spy = load_historical_spy_data()
    print(f"Historical SPY data loaded: {len(historical_spy)} records")

    print("Fetching live intraday SPY data...")
    live_spy = fetch_intraday_spy()
    print(f"Live SPY data fetched: {len(live_spy)} records")

    print("Fetching US Treasury bond yield data...")
    treasury = fetch_treasury_yield()
    print(f"Treasury yield data fetched: {len(treasury)} records")

    print("Preprocessing and merging data...")
    combined_data = preprocess_and_merge(historical_spy, live_spy, treasury)
    print(f"Combined dataset size: {len(combined_data)}")

    # Save combined data for model training
    combined_data.to_csv("combined_spy_treasury_data.csv")
    print("Combined data saved to combined_spy_treasury_data.csv")
