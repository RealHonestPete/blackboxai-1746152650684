
Built by https://www.blackbox.ai

---

```markdown
# SPY Price Prediction and Analysis

## Project Overview
This project aims to fetch, preprocess, and analyze stock market data specifically focusing on the SPY (S&P 500 ETF) and US Treasury yields. It utilizes data from Alpha Vantage and builds an XGBoost regression model to predict future SPY prices based on historical and real-time data. Additionally, a Flask web application is created to provide an interface for retrieving predictions and performance metrics.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/spypredict.git
   cd spypredict
   ```

2. **Install Dependencies**:
   Ensure you have `pip` installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` could be generated based on the imports found in the project files:
   ```plaintext
   requests
   pandas
   scikit-learn
   xgboost
   joblib
   Flask
   ```

3. **Set Up Alpha Vantage API Key**:
   Replace the `ALPHA_VANTAGE_API_KEY` in `data_fetch_and_preprocess.py` with your own Alpha Vantage API key.

## Usage

1. **Fetch and Preprocess Data**:
   Before running the application, fetch and preprocess the data by running:
   ```bash
   python data_fetch_and_preprocess.py
   ```

   This will create a combined CSV file `combined_spy_treasury_data.csv`.

2. **Train the Model**:
   Train the model using the combined dataset:
   ```bash
   python train_spy_price_model.py
   ```

3. **Run the Flask Application**:
   Start the web application by executing:
   ```bash
   python app.py
   ```

   The application will be accessible at `http://127.0.0.1:8000`.

4. **API Endpoints**:
   - **Performance Metric**: Access the R² score via:
     ```
     GET /api/performance
     ```
   - **Price Prediction**: To get the predicted SPY price, make a request to:
     ```
     GET /api/predict
     ```

## Features
- Fetch daily and intraday SPY data along with US Treasury yield.
- Preprocess and merge data for model training.
- Train a prediction model using XGBoost.
- Web interface to make predictions and display model performance metrics.

## Dependencies
- `requests`
- `pandas`
- `scikit-learn`
- `xgboost`
- `joblib`
- `Flask`

## Project Structure
```
/spypredict
│
├── app.py                                 # Flask application
├── data_fetch_and_preprocess.py           # Data fetching and preprocessing logic
├── train_spy_price_model.py                # Model training script
├── combined_spy_treasury_data.csv         # Output file after preprocessing (generated)
├── spy_price_model.pkl                     # Trained model (generated)
├── requirements.txt                        # List of Python dependencies
└── README.md                               # Project documentation
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

For any questions or contributions, feel free to open an issue or submit a pull request.
```