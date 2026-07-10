#!/usr/bin/env python3
"""
Crypto-Oracle Training Data Pipeline
====================================
Fetches full historical cryptocurrency data for 8 coins, applies feature 
engineering (using the shared feature_engineering module), merges shared 
daily data (from data_cache.py), computes target labels, and exports 
a single merged CSV (crypto_pooled_dataset.csv) for model training.

Usage:
    python data_pipeline.py
"""

import sys
import time
import logging
import warnings
import requests
import pandas as pd
import numpy as np
import os

# Import our shared modules
from feature_engineering import engineer_features, merge_shared_daily_data
from data_cache import CACHE_FILE, ONCHAIN_CACHE_FILE, COINS

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

TARGET_THRESHOLD = 0.01  # ±1 % for 3-class target

# Feature columns we require before dropna
CORE_FEATURES = [
    "daily_return", "gap", "high_low_range", "close_to_high", "close_to_low",
    "close_sma7_ratio", "close_sma21_ratio", "close_sma50_ratio",
    "sma_cross", "regime_trending_up",
    "volatility_7", "volatility_14", "vol_regime_high",
    "atr_pct", "bb_width",
    "rsi_14", "macd_norm", "macd_signal_norm", "macd_diff_norm",
    "volume_change", "volume_ratio",
    "day_of_week",
    "target",
]

# ===================================================================
# 1. Historical OHLCV (Binance SPOT API paginated)
# ===================================================================
def fetch_ohlcv_historical(symbol: str, delay: float = 0.4) -> pd.DataFrame:
    """
    Fetch full daily OHLCV history for ``symbol``USDT from Binance SPOT API.
    Paginates backwards until the API returns no more data.
    """
    url = "https://api.binance.com/api/v3/klines"
    pair = f"{symbol}USDT"
    all_rows: list[list] = []
    end_time: int | None = None  # None means "latest"

    log.info("Fetching full historical OHLCV for %s from Binance SPOT …", pair)

    while True:
        params: dict = {"symbol": pair, "interval": "1d", "limit": 1000}
        if end_time is not None:
            params["endTime"] = end_time

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.error("Binance OHLCV request failed for %s: %s", pair, exc)
            break

        if not data:
            break

        all_rows.extend(data)
        # Move endTime to 1 ms before the earliest candle received
        earliest_open_ms = int(data[0][0])
        end_time = earliest_open_ms - 1

        if len(data) < 1000:
            break

        time.sleep(delay)

    if not all_rows:
        return pd.DataFrame()

    # De-duplicate
    seen_ts: set[int] = set()
    unique_rows = []
    for r in all_rows:
        ts = int(r[0])
        if ts not in seen_ts:
            seen_ts.add(ts)
            unique_rows.append(r)

    df = pd.DataFrame(unique_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "_q_vol", "_n_trades", "_tb_base", "_tb_quote", "_ignore",
    ])
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.date
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    df = df[["Date", "open", "high", "low", "close", "volume"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="last").reset_index(drop=True)
    return df

# ===================================================================
# 2. Main Pipeline
# ===================================================================
def run_pipeline(output_path: str = "crypto_pooled_dataset.csv") -> pd.DataFrame:
    # Load shared cached data
    if not os.path.exists(CACHE_FILE):
        log.error("Shared data cache not found. Run data_cache.py first.")
        sys.exit(1)
        
    with open(CACHE_FILE) as f:
        shared_data = pd.DataFrame(pd.read_json(f, orient="records"))
        shared_data["Date"] = pd.to_datetime(shared_data["Date"])
        
    onchain_data = {}
    if os.path.exists(ONCHAIN_CACHE_FILE):
        import json
        with open(ONCHAIN_CACHE_FILE) as f:
            onchain_data = json.load(f)

    coin_frames: list[pd.DataFrame] = []

    # Map for backwards compatibility with asset naming in training
    COIN_NAMES = {
        "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "XRP": "Ripple",
        "DOGE": "Dogecoin", "ADA": "Cardano", "LINK": "Chainlink", "DOT": "Polkadot"
    }

    for symbol, cm_ticker in COINS.items():
        coin_name = COIN_NAMES[symbol]
        log.info("=" * 60)
        log.info("Processing %s (%s)", coin_name, symbol)
        
        # 1. Full OHLCV history
        ohlcv = fetch_ohlcv_historical(symbol)
        if ohlcv.empty:
            log.error("Skipping %s — no OHLCV data", symbol)
            continue
            
        # 2. Add future target column (training only)
        c = ohlcv["close"]
        future_ret = (c.shift(-1) - c) / c
        ohlcv["target"] = 0
        ohlcv.loc[future_ret > TARGET_THRESHOLD, "target"] = 1
        ohlcv.loc[future_ret < -TARGET_THRESHOLD, "target"] = -1
        # Drop last row (no future value)
        ohlcv = ohlcv.iloc[:-1].copy()

        # 3. Base feature engineering
        feat = engineer_features(ohlcv)
        
        # We need to drop OHLCV cols here since engineer_features now preserves them 
        # (wait, feature_engineering.py's engineer_features drops open, high, low, close, volume)
        # Let's ensure target is retained by assigning it back
        feat["target"] = ohlcv["target"]

        # 4. Merge Shared Data (FNG & Macro)
        feat = merge_shared_daily_data(feat, shared_data)

        # 5. On-chain metrics merge
        if symbol in onchain_data:
            oc_df = pd.DataFrame.from_dict(onchain_data[symbol], orient="index")
            oc_df.index.name = "Date"
            oc_df = oc_df.reset_index()
            oc_df["Date"] = pd.to_datetime(oc_df["Date"])
            feat = feat.merge(oc_df, on="Date", how="left")
            
        for col in ["AdrActCnt", "TxCnt"]:
            if col in feat.columns:
                feat[f"{col}_available"] = feat[col].notna().astype(int)
                feat[col] = feat[col].fillna(0.0)
            else:
                feat[col] = 0.0
                feat[f"{col}_available"] = 0

        # 6. Add asset column
        feat["asset"] = coin_name
        coin_frames.append(feat)
        log.info("  %s: %d rows", coin_name, len(feat))

    if not coin_frames:
        log.error("No coins produced data — aborting.")
        sys.exit(1)

    # Concatenate all coins
    pooled = pd.concat(coin_frames, ignore_index=True)
    
    # Drop NaN in core features (caused by rolling windows)
    cols_to_check = [c for c in CORE_FEATURES if c in pooled.columns]
    pooled = pooled.dropna(subset=cols_to_check).reset_index(drop=True)

    # One-hot encode asset
    pooled = pd.get_dummies(pooled, columns=["asset"], dtype=int)

    # Export
    pooled.to_csv(output_path, index=False)
    log.info("Saved to %s (%d rows x %d cols)", output_path, *pooled.shape)
    print(f"\n[OK] Training dataset generated: {output_path}")

    return pooled

if __name__ == "__main__":
    run_pipeline()
