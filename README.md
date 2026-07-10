# Crypto Oracle

A full-stack crypto price-direction prediction application using a pre-trained XGBoost model to predict next-day bullish/neutral/bearish trends for 8 cryptocurrencies.

## Setup

1. Copy `.env.example` to `.env` and fill in your keys.
2. Install Python dependencies:
   ```bash
   cd ml_server
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Install Node.js dependencies:
   ```bash
   cd server && npm install
   cd ../client && npm install
   ```

## Model Training Workflow

If you want to retrain the model on fresh historical data:

1. **Update Shared Cache**: Fetch the latest Fear & Greed, Macro, and On-chain data.
   ```bash
   cd ml_server
   python data_cache.py
   ```
2. **Generate Training Dataset**: This will fetch the full historical Binance OHLCV data, apply feature engineering, and merge it with the shared cache to produce `crypto_pooled_dataset.csv`.
   ```bash
   python data_pipeline.py
   ```
3. **Train**: Train the XGBoost model in your preferred environment (e.g. Google Colab) using the `crypto_pooled_dataset.csv` file. 
4. **Deploy**: Move your new `model_v1.json`, `feature_cols.json`, and `metadata.json` into `ml_server/model/current/`.
5. **Validate**: Verify the deployed model files before starting the server.
   ```bash
   python validate_model.py
   ```

## Running the Application

1. **Start the ML Server (Flask)**:
   ```bash
   cd ml_server
   python app.py
   ```
2. **Start the Backend (Express)**:
   ```bash
   cd server
   npm start
   ```
3. **Start the Frontend (React/Vite)**:
   ```bash
   cd client
   npm run dev
   ```
