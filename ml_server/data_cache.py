"""
Crypto Oracle — Data Cache Script
===================================
Manually-invoked script to fetch daily FNG and macro (S&P500, Nasdaq, DXY)
data and cache it as JSON. The ML server (app.py) reads this cache at startup
and uses it to merge shared features onto per-coin data.

Usage:
    python data_cache.py

This script must be run periodically (e.g., daily) to keep the prediction
pipeline up to date with the latest shared market features.
"""

import os
import json
import logging
import time
import requests
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("data_cache")

# Paths
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "shared_data.json")
ONCHAIN_CACHE_FILE = os.path.join(CACHE_DIR, "onchain_data.json")

COINS = {
    "BTC": "btc", "ETH": "eth", "SOL": "sol", "XRP": "xrp",
    "DOGE": "doge", "ADA": "ada", "LINK": "link", "DOT": "dot",
}
ONCHAIN_COLS = ["AdrActCnt", "TxCnt"]

def fetch_fng() -> pd.DataFrame:
    """Fetch Fear & Greed Index from alternative.me."""
    log.info("Fetching FNG data...")
    url = "https://api.alternative.me/fng/?limit=0"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception as exc:
        log.error("Failed to fetch FNG: %s", exc)
        return pd.DataFrame()

    rows = []
    for d in data:
        rows.append({
            "Date": pd.to_datetime(int(d["timestamp"]), unit="s").date(),
            "fng": int(d["value"])
        })
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    log.info("FNG: %d days retrieved (%s to %s)", len(df), df["Date"].min().date(), df["Date"].max().date())
    return df

def fetch_macro() -> pd.DataFrame:
    """Fetch macro cross-asset returns via yfinance."""
    log.info("Fetching macro data (S&P500, Nasdaq, DXY)...")
    tickers = {"^GSPC": "sp500_ret", "^IXIC": "nasdaq_ret", "DX-Y.NYB": "dxy_ret"}
    frames = {}
    
    for symbol, col_name in tickers.items():
        try:
            raw = yf.download(symbol, start="2018-01-01", progress=False, auto_adjust=True)
            if raw.empty:
                log.warning("No data for %s", symbol)
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            frames[col_name] = raw[["Close"]].rename(columns={"Close": col_name})
        except Exception as exc:
            log.error("yfinance fetch failed for %s: %s", symbol, exc)
            
    if not frames:
        return pd.DataFrame()
        
    macro = pd.concat(frames.values(), axis=1)
    # Reindex to calendar days to fill weekends/holidays
    full_idx = pd.date_range(macro.index.min(), macro.index.max(), freq="D")
    macro = macro.reindex(full_idx).ffill()
    
    # Compute daily % returns
    macro = macro.pct_change()
    
    macro.index.name = "Date"
    macro = macro.reset_index()
    macro["Date"] = pd.to_datetime(macro["Date"].dt.date)
    
    log.info("Macro: %d days retrieved (%s to %s)", len(macro), macro["Date"].min().date(), macro["Date"].max().date())
    return macro

def fetch_onchain() -> dict:
    """Fetch on-chain metrics for all supported coins from CoinMetrics."""
    log.info("Fetching on-chain metrics from CoinMetrics...")
    onchain_data = {}
    
    for symbol, cm_ticker in COINS.items():
        url = f"https://raw.githubusercontent.com/coinmetrics/data/master/csv/{cm_ticker}.csv"
        try:
            log.info("  Fetching %s (%s)...", symbol, cm_ticker)
            df = pd.read_csv(url, low_memory=False)
            
            date_col = next((c for c in ("date", "Date", "time", "Time") if c in df.columns), None)
            if not date_col:
                log.warning("  No date column for %s", symbol)
                continue
                
            df = df.rename(columns={date_col: "Date"})
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date")
            
            # Keep only needed columns
            keep_cols = ["Date"] + [c for c in ONCHAIN_COLS if c in df.columns]
            df = df[keep_cols].copy()
            df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            
            # Convert to dictionary keyed by Date
            df = df.set_index("Date")
            # Convert values to float, dropna to keep JSON small
            coin_dict = {}
            for date_str, row in df.iterrows():
                row_dict = {}
                for col in ONCHAIN_COLS:
                    if col in row and pd.notna(row[col]):
                        row_dict[col] = float(row[col])
                if row_dict:
                    coin_dict[date_str] = row_dict
                    
            onchain_data[symbol] = coin_dict
            log.info("  %s: %d days of data", symbol, len(coin_dict))
            time.sleep(0.5)  # Rate limiting
        except Exception as exc:
            log.warning("  Failed to fetch %s: %s", symbol, exc)
            
    return onchain_data

def main():
    log.info("Starting data cache refresh...")
    
    fng_df = fetch_fng()
    macro_df = fetch_macro()
    
    if fng_df.empty:
        log.error("FNG data is empty. Aborting.")
        return
        
    # Merge
    merged = fng_df.copy()
    if not macro_df.empty:
        merged = merged.merge(macro_df, on="Date", how="left")
    else:
        log.warning("Macro data is empty, features will be missing.")
        
    # Date must be string for JSON serialization
    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")
    
    records = merged.to_dict(orient="records")
    
    with open(CACHE_FILE, "w") as f:
        json.dump(records, f, indent=2)
        
    log.info("Successfully wrote %d records to %s", len(records), CACHE_FILE)
    print(f"\n[OK] Shared data cache updated: {CACHE_FILE} ({len(records)} days)")
    
    # On-chain data
    onchain_data = fetch_onchain()
    if onchain_data:
        with open(ONCHAIN_CACHE_FILE, "w") as f:
            json.dump(onchain_data, f, indent=2)
        log.info("Successfully wrote on-chain data to %s", ONCHAIN_CACHE_FILE)
        print(f"[OK] On-chain cache updated: {ONCHAIN_CACHE_FILE} ({len(onchain_data)} coins)")
    else:
        log.error("No on-chain data retrieved.")

if __name__ == "__main__":
    main()
