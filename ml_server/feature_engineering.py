"""
Crypto Oracle — Shared Feature Engineering Module
===================================================
Single source of truth for all feature computation. Both the training pipeline
(data_pipeline.py / Colab notebook) and the prediction server (app.py) must
import and call these functions to ensure feature consistency.

Requirements:
    pip install pandas numpy ta
"""

import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator
from ta.trend import MACD


# ---------------------------------------------------------------------------
# FNG bucket boundaries (standard Fear & Greed interpretation)
# ---------------------------------------------------------------------------
_FNG_BINS = [-1, 24, 44, 55, 74, 100]
_FNG_LABELS = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]


def engineer_features(coin_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all engineered features from a single coin's raw OHLCV data.

    Parameters
    ----------
    coin_df : pd.DataFrame
        Must contain columns [Date, open, high, low, close, volume],
        sorted chronologically, for a SINGLE coin.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with all engineered feature columns appended.
        Raw OHLCV columns are preserved (caller may drop them if needed).
        Rows before ~120 days will have NaNs in some feature columns due to
        rolling-window warmup (vol_regime_high depends on a 14-day volatility
        rolled over a 90-day window).

    Notes
    -----
    This function does NOT compute targets — that is a training concern.
    This function does NOT drop any rows — the caller decides trimming.
    """
    df = coin_df.copy().sort_values("Date").reset_index(drop=True)
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # ── Returns / price-action ────────────────────────────────────────────
    df["daily_return"] = (c - o) / o
    df["gap"] = (o - c.shift(1)) / c.shift(1)
    df["high_low_range"] = (h - l) / c
    df["close_to_high"] = (h - c) / c
    df["close_to_low"] = (c - l) / c

    # ── Trend (SMA ratios) ────────────────────────────────────────────────
    sma7 = c.rolling(7).mean()
    sma21 = c.rolling(21).mean()
    sma50 = c.rolling(50).mean()

    df["close_sma7_ratio"] = c / sma7
    df["close_sma21_ratio"] = c / sma21
    df["close_sma50_ratio"] = c / sma50
    df["sma_cross"] = (sma7 > sma21).astype(int)
    df["regime_trending_up"] = (c > sma50).astype(int)

    # ── Volatility ────────────────────────────────────────────────────────
    df["volatility_7"] = df["daily_return"].rolling(7).std()
    df["volatility_14"] = df["daily_return"].rolling(14).std()
    vol14_med90 = df["volatility_14"].rolling(90).median()
    df["vol_regime_high"] = (df["volatility_14"] > vol14_med90).astype(int)

    atr = AverageTrueRange(high=h, low=l, close=c, window=14)
    df["atr_pct"] = atr.average_true_range() / c

    bb = BollingerBands(close=c, window=20, window_dev=2)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / c

    # ── Momentum ──────────────────────────────────────────────────────────
    rsi = RSIIndicator(close=c, window=14)
    df["rsi_14"] = rsi.rsi()

    macd_ind = MACD(close=c)
    df["macd_norm"] = macd_ind.macd() / c
    df["macd_signal_norm"] = macd_ind.macd_signal() / c
    df["macd_diff_norm"] = macd_ind.macd_diff() / c

    # ── Volume ────────────────────────────────────────────────────────────
    df["volume_change"] = v.pct_change()
    vol_rm20 = v.rolling(20).mean()
    df["volume_ratio"] = v / vol_rm20

    # ── Calendar ──────────────────────────────────────────────────────────
    df["day_of_week"] = df["Date"].dt.dayofweek

    return df


def merge_shared_daily_data(
    coin_df: pd.DataFrame, shared_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Left-merge shared daily features (FNG, macro returns) onto a per-coin
    engineered dataframe.

    Parameters
    ----------
    coin_df : pd.DataFrame
        Per-coin dataframe (output of ``engineer_features``).
        Must contain a ``Date`` column.
    shared_data : pd.DataFrame
        Shared daily data with at minimum columns:
        [Date, fng, sp500_ret, nasdaq_ret, dxy_ret].
        Provided by the caller — this function makes NO network calls
        and has NO side effects.

    Returns
    -------
    pd.DataFrame
        ``coin_df`` with the following columns merged / appended:
        - fng, fng_lag_1, fng_change, fng_sma_7
        - fng_Extreme Fear, fng_Extreme Greed, fng_Fear, fng_Greed, fng_Neutral
        - sp500_ret, nasdaq_ret, dxy_ret
    """
    # ------------------------------------------------------------------
    # Compute FNG dynamics on the full shared time-series BEFORE merging
    # (ensures lag/rolling are computed on the continuous daily series,
    #  not gaps introduced by a sparse coin date range)
    # ------------------------------------------------------------------
    sd = shared_data.copy().sort_values("Date").reset_index(drop=True)

    if "fng" in sd.columns:
        sd["fng_lag_1"] = sd["fng"].shift(1)
        sd["fng_change"] = sd["fng"] - sd["fng_lag_1"]
        sd["fng_sma_7"] = sd["fng"].rolling(7).mean()

        # FNG bucket one-hot dummies
        sd["_fng_bucket"] = pd.cut(
            sd["fng"], bins=_FNG_BINS, labels=_FNG_LABELS, right=True
        )
        bucket_dummies = pd.get_dummies(sd["_fng_bucket"], prefix="fng", dtype=int)
        # Guarantee all 5 bucket columns exist even if some buckets are empty
        for label in _FNG_LABELS:
            col = f"fng_{label}"
            if col not in bucket_dummies.columns:
                bucket_dummies[col] = 0
        sd = pd.concat([sd, bucket_dummies], axis=1)
        sd = sd.drop(columns=["_fng_bucket"])

    # ------------------------------------------------------------------
    # Select only the columns we need for the merge
    # ------------------------------------------------------------------
    merge_cols = ["Date"]
    desired = [
        "fng", "fng_lag_1", "fng_change", "fng_sma_7",
        "fng_Extreme Fear", "fng_Extreme Greed", "fng_Fear",
        "fng_Greed", "fng_Neutral",
        "sp500_ret", "nasdaq_ret", "dxy_ret",
    ]
    merge_cols += [c for c in desired if c in sd.columns]
    sd = sd[merge_cols]

    # ------------------------------------------------------------------
    # Left-merge onto coin dataframe
    # ------------------------------------------------------------------
    result = coin_df.merge(sd, on="Date", how="left")

    return result
