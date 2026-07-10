"""
Crypto Oracle — ML Prediction Server
======================================
Flask server that serves predictions from a pre-trained XGBoost model.
The model is loaded ONCE at startup from model/current/ — no per-request
training occurs. Feature engineering uses the shared module
(feature_engineering.py) to guarantee consistency with the training pipeline.

Endpoints:
    POST /predict   — Generate a prediction for a single coin
    GET  /health    — Server and model status
"""

import os
import json
import time
import logging

import numpy as np
import pandas as pd
import requests as http_requests
import xgboost as xgb
from flask import Flask, request, jsonify
from flask_cors import CORS

from feature_engineering import engineer_features, merge_shared_daily_data

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml_server")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "current")

log.info("Loading model from %s …", MODEL_DIR)

model = xgb.XGBClassifier()
model.load_model(os.path.join(MODEL_DIR, "model_v1.json"))
log.info("  XGBoost model loaded (%d classes)", model.n_classes_)

with open(os.path.join(MODEL_DIR, "feature_cols.json")) as f:
    FEATURE_COLS = json.load(f)
log.info("  %d feature columns loaded", len(FEATURE_COLS))

with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
    METADATA = json.load(f)
log.info("  Model version: %s", METADATA.get("trained_at"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Label mapping from XGBoost class index → semantic label
# metadata.json stores {"0": -1, "1": 0, "2": 1}
LABEL_MAP = {int(k): v for k, v in METADATA.get("label_mapping", {}).items()}
LABEL_NAMES = {-1: "BEARISH", 0: "NEUTRAL", 1: "BULLISH"}

# Confidence threshold for high-confidence signals (raw probability, 0-1 scale)
CONFIDENCE_THRESHOLD = 0.5

# Coin registries
COIN_NAME_MAP = {
    "BTC": "Bitcoin",   "ETH": "Ethereum",  "SOL": "Solana",    "XRP": "Ripple",
    "DOGE": "Dogecoin", "ADA": "Cardano",   "LINK": "Chainlink", "DOT": "Polkadot",
}
CM_TICKER_MAP = {
    "BTC": "btc", "ETH": "eth", "SOL": "sol", "XRP": "xrp",
    "DOGE": "doge", "ADA": "ada", "LINK": "link", "DOT": "dot",
}
SUPPORTED_COINS = set(COIN_NAME_MAP.keys())

# On-chain columns the model expects
ONCHAIN_COLS = ["AdrActCnt", "TxCnt"]

# ---------------------------------------------------------------------------
# Shared daily data cache (FNG + macro)
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
SHARED_DATA_PATH = os.path.join(CACHE_DIR, "shared_data.json")


def load_shared_data() -> pd.DataFrame | None:
    """Load shared daily data (FNG, macro returns) from the local cache file."""
    if not os.path.exists(SHARED_DATA_PATH):
        log.warning("Shared data cache not found at %s", SHARED_DATA_PATH)
        return None
    try:
        with open(SHARED_DATA_PATH) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["Date"])
        log.info("  Shared data cache loaded: %d rows (%s → %s)",
                 len(df), df["Date"].min().date(), df["Date"].max().date())
        return df
    except Exception as exc:
        log.error("Failed to load shared data cache: %s", exc)
        return None


SHARED_DATA = load_shared_data()

# ---------------------------------------------------------------------------
# On-chain metrics — cached from data_cache.py
# ---------------------------------------------------------------------------
ONCHAIN_CACHE_PATH = os.path.join(CACHE_DIR, "onchain_data.json")

def load_onchain_data() -> dict | None:
    """Load on-chain data cache from file."""
    if not os.path.exists(ONCHAIN_CACHE_PATH):
        log.warning("On-chain data cache not found at %s", ONCHAIN_CACHE_PATH)
        return None
    try:
        with open(ONCHAIN_CACHE_PATH) as f:
            data = json.load(f)
        log.info("  On-chain data cache loaded for %d coins", len(data))
        return data
    except Exception as exc:
        log.error("Failed to load on-chain data cache: %s", exc)
        return None

ONCHAIN_DATA = load_onchain_data()

def get_onchain_metrics(symbol: str, target_date: pd.Timestamp) -> dict:
    """Get the latest cached on-chain metrics for the target date."""
    if not ONCHAIN_DATA or symbol not in ONCHAIN_DATA:
        return {}

    date_str = target_date.strftime("%Y-%m-%d")
    coin_data = ONCHAIN_DATA[symbol]

    # Try exact date
    if date_str in coin_data:
        return coin_data[date_str]

    # Otherwise get the most recent available date before target
    available_dates = sorted([d for d in coin_data.keys() if d <= date_str])
    if not available_dates:
        return {}

    latest_date = available_dates[-1]
    return coin_data[latest_date]


# ---------------------------------------------------------------------------
# OHLCV fetch — Binance SPOT API
# ---------------------------------------------------------------------------
def fetch_ohlcv(symbol: str, days: int = 150) -> pd.DataFrame:
    """
    Fetch recent daily OHLCV candles from Binance SPOT API.

    Uses api.binance.com (not futures — futures is geo-restricted).
    No pagination needed for ≤1000 days.

    Returns a DataFrame with columns [Date, open, high, low, close, volume],
    sorted chronologically.
    """
    pair = f"{symbol}USDT"
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": pair, "interval": "1d", "limit": days}

    resp = http_requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        raise ValueError(f"No OHLCV data returned for {pair}")

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "_q_vol", "_n_trades", "_tb_base", "_tb_quote", "_ignore",
    ])
    df["Date"] = (
        pd.to_datetime(df["open_time"], unit="ms", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    df = (
        df[["Date", "open", "high", "low", "close", "volume"]]
        .sort_values("Date")
        .reset_index(drop=True)
    )
    log.info("Fetched %d daily candles for %s", len(df), pair)
    return df


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Generate a prediction for a single cryptocurrency.

    Expects JSON body with at least ``{"coin": "BTC"}`` (or ``"cryptocurrency"``
    key for backward-compatibility with the Node backend).

    Returns JSON:
        {
            "prediction": "BULLISH" | "NEUTRAL" | "BEARISH",
            "confidence": <float 0-100>,
            "high_confidence_signal": <bool>,
            "model_version": "<ISO timestamp>"
        }
    """
    try:
        # ── 1. Extract coin identifier ────────────────────────────────────
        body = request.json or {}
        coin = (body.get("coin") or body.get("cryptocurrency") or "").upper()

        if not coin:
            return jsonify({"error": "Missing 'coin' parameter in request body"}), 400

        if coin not in SUPPORTED_COINS:
            return jsonify({
                "error": f"Unsupported coin: {coin}",
                "supported_coins": sorted(SUPPORTED_COINS),
            }), 400

        # ── 2. Fetch OHLCV from Binance ───────────────────────────────────
        try:
            ohlcv_df = fetch_ohlcv(coin, days=150)
        except Exception as exc:
            return jsonify({
                "error": (
                    f"Failed to fetch OHLCV data for {coin}USDT from Binance: "
                    f"{exc}"
                )
            }), 502

        if len(ohlcv_df) < 120:
            return jsonify({
                "error": (
                    f"Insufficient OHLCV history for {coin}: got {len(ohlcv_df)} "
                    f"days, need at least ~120 for rolling-window features"
                )
            }), 422

        # ── 3. Engineer features ──────────────────────────────────────────
        featured_df = engineer_features(ohlcv_df)

        # ── 4. Merge shared daily data (FNG + macro) ──────────────────────
        if SHARED_DATA is not None:
            featured_df = merge_shared_daily_data(featured_df, SHARED_DATA)
        else:
            return jsonify({
                "error": (
                    "Shared data cache not populated. "
                    "Run `python data_cache.py` to fetch FNG and macro data first."
                )
            }), 503

        # ── 5. Take the last row (today's / most recent prediction) ──────
        last_row = featured_df.iloc[[-1]].copy()
        prediction_date = last_row["Date"].iloc[0]

        # ── 6. Fetch on-chain metrics (from cache) ────────────────────────
        onchain = get_onchain_metrics(coin, prediction_date)
        for col in ONCHAIN_COLS:
            if col in onchain:
                last_row[col] = onchain[col]
                last_row[f"{col}_available"] = 1
            else:
                last_row[col] = 0.0
                last_row[f"{col}_available"] = 0

        # ── 7. Asset one-hot encoding ─────────────────────────────────────
        coin_name = COIN_NAME_MAP[coin]
        for name in COIN_NAME_MAP.values():
            last_row[f"asset_{name}"] = 1 if name == coin_name else 0

        # ── 8. Select features in exact model training order ──────────────
        missing_cols = [c for c in FEATURE_COLS if c not in last_row.columns]
        if missing_cols:
            return jsonify({
                "error": f"Missing expected feature columns: {missing_cols}",
            }), 500

        prediction_row = last_row[FEATURE_COLS]

        # ── 9. NaN guard ─────────────────────────────────────────────────
        nan_cols = prediction_row.columns[prediction_row.isna().any()].tolist()
        if nan_cols:
            return jsonify({
                "error": (
                    f"Insufficient history to generate a prediction for {coin} "
                    f"yet. Features with missing values: {nan_cols}"
                ),
            }), 422

        # ── 10. Predict ──────────────────────────────────────────────────
        proba = model.predict_proba(prediction_row)[0]
        pred_idx = int(np.argmax(proba))
        confidence = float(proba[pred_idx])
        predicted_class = LABEL_MAP.get(pred_idx, 0)

        return jsonify({
            "prediction": LABEL_NAMES.get(predicted_class, "UNKNOWN"),
            "confidence": round(confidence * 100, 2),
            "high_confidence_signal": confidence >= CONFIDENCE_THRESHOLD,
            "model_version": METADATA.get("trained_at"),
        })

    except Exception as exc:
        log.exception("Unhandled error in /predict")
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Return server and model status for operational visibility."""
    return jsonify({
        "status": "ok",
        "model_version": METADATA.get("trained_at"),
        "model_type": METADATA.get("model_type"),
        "supported_coins": sorted(SUPPORTED_COINS),
        "shared_data_loaded": SHARED_DATA is not None,
        "feature_count": len(FEATURE_COLS),
    })


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)
