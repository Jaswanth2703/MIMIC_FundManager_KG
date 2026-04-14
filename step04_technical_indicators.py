"""
Step 04 -- Compute Technical Indicators from Daily OHLCV
========================================================

Computes per-stock daily technical indicators, then aggregates to monthly.

Input:  MARKET_DIR / kite_ohlcv_daily.csv
Output: MARKET_DIR / technical_indicators_monthly.csv

Indicators computed (daily, then aggregated to monthly last-day values):
- RSI (14-day)
- MACD (12, 26, 9) -- macd_line, signal_line, histogram
- Bollinger Bands (20, 2) -- %B position
- SMA_50, SMA_200, golden_cross / death_cross signals
- ATR (14-day)
- OBV (On-Balance Volume)
- monthly_return, monthly_volatility, volume_ratio
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MARKET_DIR
from utils import coverage_report

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Paths
DAILY_INPUT = os.path.join(MARKET_DIR, 'kite_ohlcv_daily.csv')
OUTPUT_CSV = os.path.join(MARKET_DIR, 'technical_indicators_monthly.csv')


# ============================================================
# Daily indicator calculations (manual -- no external ta lib needed)
# ============================================================

def compute_rsi(close, period=14):
    """Wilder's RSI using exponential smoothing of gains/losses."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.copy()
    avg_loss = loss.copy()

    avg_gain.iloc[:period] = np.nan
    avg_loss.iloc[:period] = np.nan

    first_avg_gain = gain.iloc[1:period + 1].mean()
    first_avg_loss = loss.iloc[1:period + 1].mean()

    avg_gain.iloc[period] = first_avg_gain
    avg_loss.iloc[period] = first_avg_loss

    for i in range(period + 1, len(close)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(close, fast=12, slow=26, signal=9):
    """MACD: macd_line, signal_line, histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_pband(close, period=20, num_std=2):
    """Bollinger %B = (price - lower) / (upper - lower)."""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_width = upper - lower
    pband = (close - lower) / band_width.replace(0, np.nan)
    return pband


def compute_sma(close, period):
    """Simple Moving Average."""
    return close.rolling(window=period).mean()


def compute_atr(high, low, close, period=14):
    """Average True Range using max(H-L, |H-Cprev|, |L-Cprev|)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def compute_obv(close, volume):
    """On-Balance Volume: cumulative sum of volume * sign(close change)."""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    obv = (volume * direction).cumsum()
    return obv


def compute_daily_indicators(df_stock):
    """Compute all daily technical indicators for a single stock.

    Parameters
    ----------
    df_stock : DataFrame
        Daily OHLCV data for one stock, sorted by date ascending.

    Returns
    -------
    DataFrame with daily indicator columns appended.
    """
    df = df_stock.copy()
    df = df.sort_values('date').reset_index(drop=True)

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # RSI(14)
    df['rsi'] = compute_rsi(close, period=14)

    # MACD (12, 26, 9)
    macd_line, signal_line, histogram = compute_macd(close, fast=12, slow=26, signal=9)
    df['macd_line'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = histogram

    # Bollinger %B (20, 2)
    df['bollinger_pband'] = compute_bollinger_pband(close, period=20, num_std=2)

    # SMA50, SMA200
    df['sma50'] = compute_sma(close, 50)
    df['sma200'] = compute_sma(close, 200)

    # Golden cross / death cross signals
    df['golden_cross'] = ((df['sma50'] > df['sma200']) &
                          (df['sma50'].shift(1) <= df['sma200'].shift(1))).astype(int)
    df['death_cross'] = ((df['sma50'] < df['sma200']) &
                         (df['sma50'].shift(1) >= df['sma200'].shift(1))).astype(int)

    # ATR(14)
    df['atr'] = compute_atr(high, low, close, period=14)

    # OBV
    df['obv'] = compute_obv(close, volume)

    # Daily returns (for monthly volatility calculation)
    df['daily_return'] = close.pct_change()

    return df


# ============================================================
# Monthly aggregation
# ============================================================

def aggregate_to_monthly(df_daily):
    """Aggregate daily indicators to monthly granularity.

    For each stock-month:
    - RSI, MACD, Bollinger, SMA, ATR, OBV: last trading day value
    - golden_cross / death_cross: max (1 if any signal in the month)
    - monthly_return: (last_close / first_close) - 1
    - monthly_volatility: std of daily returns within the month
    - volume_ratio: mean daily volume / 3-month rolling avg volume
    """
    df = df_daily.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year_month_str'] = df['date'].dt.to_period('M').astype(str)

    # Pre-compute 3-month rolling mean volume per stock for volume_ratio
    overall_mean_vol = df.groupby('symbol')['volume'].transform('mean')
    df['_vol_ratio_raw'] = df['volume'] / overall_mean_vol.replace(0, np.nan)

    grouped = df.groupby(['symbol', 'year_month_str'])

    # Last-day indicator values
    last_day = grouped.agg(
        rsi=('rsi', 'last'),
        macd_line=('macd_line', 'last'),
        macd_signal=('macd_signal', 'last'),
        macd_hist=('macd_hist', 'last'),
        bollinger_pband=('bollinger_pband', 'last'),
        sma50=('sma50', 'last'),
        sma200=('sma200', 'last'),
        golden_cross=('golden_cross', 'max'),
        death_cross=('death_cross', 'max'),
        obv=('obv', 'last'),
        atr=('atr', 'mean'),
        monthly_volatility=('daily_return', 'std'),
        volume_ratio=('_vol_ratio_raw', 'mean'),
    ).reset_index()

    # Monthly return: (last_close / first_close) - 1
    close_agg = grouped['close'].agg(['first', 'last']).reset_index()
    close_agg['monthly_return'] = (
        close_agg['last'] / close_agg['first'].replace(0, np.nan) - 1
    )

    last_day = last_day.merge(
        close_agg[['symbol', 'year_month_str', 'monthly_return']],
        on=['symbol', 'year_month_str'],
        how='left',
    )

    # Select final columns
    output_cols = [
        'symbol', 'year_month_str',
        'rsi', 'macd_line', 'macd_signal', 'macd_hist',
        'bollinger_pband', 'sma50', 'sma200',
        'golden_cross', 'death_cross',
        'atr', 'obv',
        'monthly_return', 'monthly_volatility', 'volume_ratio',
    ]
    return last_day[output_cols]


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("STEP 04: Technical Indicators from Daily OHLCV")
    print("=" * 70)

    # --- Cache check ---
    if os.path.isfile(OUTPUT_CSV):
        try:
            cached = pd.read_csv(OUTPUT_CSV, nrows=5)
            if len(cached) > 0:
                with open(OUTPUT_CSV, 'r') as f:
                    row_count = sum(1 for _ in f) - 1
                if row_count > 100:
                    print(f"[step04] Cache hit: {OUTPUT_CSV} ({row_count:,} rows)")
                    print("[step04] Skipping. Delete output file to recompute.")
                    return
        except Exception:
            pass

    # --- Load daily OHLCV ---
    if not os.path.exists(DAILY_INPUT):
        print(f"[step04] ERROR: Input file not found: {DAILY_INPUT}")
        print("         Run step03 (Kite OHLCV download) first.")
        return

    print(f"\n[step04] Loading daily OHLCV from: {DAILY_INPUT}")
    df = pd.read_csv(DAILY_INPUT, parse_dates=['date'])
    print(f"  Loaded {len(df):,} rows for {df['symbol'].nunique()} stocks")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    # --- Validate required columns ---
    required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"[step04] ERROR: Missing columns: {missing_cols}")
        return

    # Drop rows with missing price data
    df = df.dropna(subset=['close', 'high', 'low', 'volume'])
    print(f"  After dropping NaN prices: {len(df):,} rows")

    # --- Compute daily indicators per stock ---
    print("\n[step04] Computing daily technical indicators per stock ...")
    all_daily = []
    symbols = sorted(df['symbol'].unique())
    total = len(symbols)

    for idx, symbol in enumerate(symbols):
        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"  Processing {idx + 1}/{total}: {symbol}")

        df_stock = df[df['symbol'] == symbol].copy()
        if len(df_stock) < 30:
            continue  # skip stocks with too few data points

        df_daily = compute_daily_indicators(df_stock)
        all_daily.append(df_daily)

    if not all_daily:
        print("[step04] ERROR: No stocks had enough data for indicator calculation.")
        return

    df_all_daily = pd.concat(all_daily, ignore_index=True)
    print(f"\n  Daily indicators computed: {len(df_all_daily):,} rows")

    # --- Aggregate to monthly ---
    print("\n[step04] Aggregating to monthly granularity ...")
    df_monthly = aggregate_to_monthly(df_all_daily)
    print(f"  Monthly rows: {len(df_monthly):,}")
    print(f"  Unique stocks: {df_monthly['symbol'].nunique()}")
    print(f"  Unique months: {df_monthly['year_month_str'].nunique()}")

    # --- Coverage report ---
    coverage_report(df_monthly)

    # --- Save output ---
    df_monthly.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[step04] Saved: {OUTPUT_CSV}")
    print(f"         Shape: {df_monthly.shape}")

    # Print sample
    print("\n[step04] Sample output (first 5 rows):")
    print(df_monthly.head().to_string(index=False))

    print("\n[step04] Done.")


if __name__ == '__main__':
    main()
