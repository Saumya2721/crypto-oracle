# Crypto Data Pipeline -- Walkthrough

## What was built

A single self-contained Python script ([data_pipeline.py](file:///c:/Users/LENOVO/Documents/Crypto-Oracle/ml_server/data_pipeline.py)) that fetches, merges, and feature-engineers daily cryptocurrency data for 8 coins into a single pooled CSV.

## Pipeline Results

**Output**: [crypto_pooled_dataset.csv](file:///c:/Users/LENOVO/Documents/Crypto-Oracle/ml_server/crypto_pooled_dataset.csv) -- **21,682 rows x 44 columns** (8.3 MB)

### Rows per coin

| Coin | Rows | Date Range |
|------|------|------------|
| Bitcoin | 3,197 | 2017-10-05 --> 2026-07-06 |
| Ethereum | 3,197 | 2017-10-05 --> 2026-07-06 |
| Cardano | 2,954 | 2018-06-05 --> 2026-07-06 |
| Ripple | 2,937 | 2018-06-22 --> 2026-07-06 |
| Chainlink | 2,680 | 2019-03-06 --> 2026-07-06 |
| Dogecoin | 2,510 | 2019-08-23 --> 2026-07-06 |
| Solana | 2,107 | 2020-09-29 --> 2026-07-06 |
| Polkadot | 2,100 | 2020-10-06 --> 2026-07-06 |

### Target distribution (3-class, threshold = +/-1%)

| Class | Count | % |
|-------|-------|---|
| Bearish (-1) | 8,074 | 37.2% |
| Neutral (0) | 5,593 | 25.8% |
| Bullish (1) | 8,015 | 37.0% |

## Data Sources -- Status

| Source | Status | Notes |
|--------|--------|-------|
| **Binance OHLCV** | All 8 coins | Full history via backward pagination |
| **Fear & Greed** | 3,075 days | 2018-02-01 to 2026-07-07 |
| **Macro (yfinance)** | 3,109 rows | S&P500, Nasdaq, DXY -- calendar-day reindexed, ffilled, pct_change |
| **Bybit Funding** | 3/8 coins | ETH, SOL, LINK got 67 daily values each; others geo-blocked |
| **CoinMetrics On-chain** | 7/8 coins partial | AdrActCnt + TxCnt for all except Solana; NVTAdj missing for all |

> [!NOTE]
> Bybit funding rates are intermittently geo-restricted. The `funding_rate_available` flag column correctly marks which rows have real data vs zero-filled. Same pattern for on-chain fields via `*_available` columns.

## 44 Final Columns

### Core Engineered Features (22)
`daily_return`, `gap`, `high_low_range`, `close_to_high`, `close_to_low`, `close_sma7_ratio`, `close_sma21_ratio`, `close_sma50_ratio`, `sma_cross`, `regime_trending_up`, `volatility_7`, `volatility_14`, `vol_regime_high`, `atr_pct`, `bb_width`, `rsi_14`, `macd_norm`, `macd_signal_norm`, `macd_diff_norm`, `volume_change`, `volume_ratio`, `day_of_week`

### Target
`target` (3-class: -1, 0, 1)

### Shared Features (4)
`fng`, `sp500_ret`, `nasdaq_ret`, `dxy_ret`

### Per-coin External (8)
`funding_rate_avg`, `funding_rate_available`, `AdrActCnt`, `TxCnt`, `NVTAdj`, `AdrActCnt_available`, `TxCnt_available`, `NVTAdj_available`

### Identity (9)
`Date` + 8 one-hot `asset_*` columns

## Key Design Decisions

- **All features are ratios/percentages** -- no raw price levels or dollar-denominated values, ensuring stationarity across BTC's $4k-$70k+ range
- **Raw OHLCV dropped** -- only engineered features remain in the final CSV
- **Availability flags** -- `funding_rate_available`, `AdrActCnt_available`, etc. prevent silent zero-fill from contaminating models
- **Chronological order preserved** -- no shuffling anywhere; Date column retained for train/test splitting
- **Graceful degradation** -- Bybit/CoinMetrics failures don't crash the pipeline; missing data is flagged, not silently dropped
