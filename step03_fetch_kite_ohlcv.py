"""
Step 03 -- Fetch OHLCV Data from Kite Connect API
==================================================

Downloads daily OHLCV data for all portfolio stocks and key indices
from Zerodha Kite Connect, then aggregates to monthly granularity.

Inputs:
- unified_isin_symbol_map.csv (from MAPPINGS_DIR)
- Kite Connect API credentials (from config)

Outputs:
- MARKET_DIR / kite_ohlcv_daily.csv   : daily OHLCV for all stocks + indices
- MARKET_DIR / kite_ohlcv_monthly.csv : monthly aggregated OHLCV

Supports checkpoint-based resumption: per-stock daily CSVs are cached
in MARKET_DIR/checkpoints/ and skipped on re-run.

Rate-limiting is enforced at 1/KITE_RATE_LIMIT seconds between API calls.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    KITE_API_KEY, KITE_ACCESS_TOKEN, KITE_RATE_LIMIT,
    KITE_INDEX_MAP, MARKET_DIR, MAPPINGS_DIR, DATE_START, DATE_END,
)
from utils import rate_limit_sleep

import pandas as pd
import numpy as np
import traceback
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Output paths
DAILY_OUTPUT = os.path.join(MARKET_DIR, 'kite_ohlcv_daily.csv')
MONTHLY_OUTPUT = os.path.join(MARKET_DIR, 'kite_ohlcv_monthly.csv')

# Checkpoint directory for per-stock daily CSVs
CHECKPOINT_DIR = os.path.join(MARKET_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

UNIFIED_MAP_CSV = os.path.join(MAPPINGS_DIR, 'unified_isin_symbol_map.csv')


# ------------------------------------------------------------------
# Kite Connect initialisation
# ------------------------------------------------------------------

def init_kite():
    """Initialise and return a KiteConnect instance.

    Returns None if kiteconnect is not installed or credentials are placeholders.
    """
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        print("[step03] ERROR: kiteconnect package not installed.")
        print("         Install with: pip install kiteconnect")
        return None

    if KITE_API_KEY.startswith('YOUR_KITE') or KITE_ACCESS_TOKEN.startswith('YOUR_KITE'):
        print("[step03] WARNING: Kite API credentials not configured.")
        print("         Set KITE_API_KEY and KITE_ACCESS_TOKEN in config.py or env vars.")
        return None

    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(KITE_ACCESS_TOKEN)

    # Quick validation
    try:
        profile = kite.profile()
        print(f"[step03] Authenticated as: {profile.get('user_name', 'unknown')}"
              f" ({profile.get('user_id', '')})")
    except Exception as e:
        print(f"[step03] WARNING: Could not validate Kite session -- {e}")
        print("         Proceeding anyway (token may have expired).")

    return kite


# ------------------------------------------------------------------
# Instrument mapping
# ------------------------------------------------------------------

def build_symbol_token_map(kite):
    """Fetch NSE instrument list and build:
    - sym_map: {symbol -> instrument_token} for equities
    - idx_map: {index_name -> instrument_token} for indices in KITE_INDEX_MAP
    """
    print("[step03] Fetching NSE instrument list ...")
    instruments = kite.instruments("NSE")
    print(f"[step03] NSE instruments: {len(instruments)}")

    # Equity symbol -> token
    sym_map = {}
    for inst in instruments:
        sym = inst.get('tradingsymbol', '')
        seg = inst.get('segment', '')
        token = inst.get('instrument_token')
        if seg == 'NSE' and sym and token:
            if sym not in sym_map:
                sym_map[sym] = token

    print(f"[step03] Unique NSE equity symbols mapped: {len(sym_map)}")

    # Index tokens -- indices in Kite have segment="INDICES"
    idx_map = {}
    for inst in instruments:
        name = inst.get('name', '')
        seg = inst.get('segment', '')
        token = inst.get('instrument_token')
        if seg == 'INDICES' and name and token:
            if name.upper() in KITE_INDEX_MAP:
                idx_map[name.upper()] = token

    # Fallback: try NFO instrument list for missing indices
    if len(idx_map) < len(KITE_INDEX_MAP):
        try:
            nfo_instruments = kite.instruments("NFO")
            for inst in nfo_instruments:
                name = inst.get('name', '')
                token = inst.get('instrument_token')
                if name.upper() in KITE_INDEX_MAP and name.upper() not in idx_map:
                    idx_map[name.upper()] = token
        except Exception:
            pass

    print(f"[step03] Index tokens resolved: {len(idx_map)}/{len(KITE_INDEX_MAP)}")
    for idx_name in KITE_INDEX_MAP:
        status = "OK" if idx_name in idx_map else "MISSING"
        print(f"    {idx_name:20s} : {status}")

    return sym_map, idx_map


# ------------------------------------------------------------------
# Fetch daily OHLCV for a single instrument
# ------------------------------------------------------------------

def fetch_daily_ohlcv(kite, instrument_token, symbol, from_date, to_date):
    """Fetch daily OHLCV candles for a single instrument via Kite API.

    Returns pd.DataFrame with columns: date, open, high, low, close, volume
    """
    try:
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval="day",
        )
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
            df['date'] = pd.to_datetime(df['date'])

        expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = np.nan

        return df[expected_cols].copy()

    except Exception as e:
        print(f"    ERROR fetching {symbol} (token={instrument_token}): {e}")
        return pd.DataFrame()


# ------------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------------

def _checkpoint_path(symbol):
    """Return path for per-stock checkpoint CSV."""
    safe_sym = symbol.replace('&', '_AND_').replace(' ', '_').replace('/', '_')
    return os.path.join(CHECKPOINT_DIR, f"{safe_sym}_daily.csv")


def _load_checkpoint(symbol):
    """Load previously fetched daily data from checkpoint, or None."""
    path = _checkpoint_path(symbol)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, parse_dates=['date'])
            if len(df) > 0:
                return df
        except Exception:
            pass
    return None


def _save_checkpoint(symbol, df):
    """Save per-stock daily data to checkpoint CSV."""
    if df is not None and len(df) > 0:
        path = _checkpoint_path(symbol)
        df.to_csv(path, index=False)


# ------------------------------------------------------------------
# Main stock data fetch loop
# ------------------------------------------------------------------

def fetch_all_stocks(kite, sym_map, symbols, from_date, to_date):
    """Fetch daily OHLCV for every symbol. Supports checkpointing every 50 stocks.

    Returns pd.DataFrame (date, symbol, open, high, low, close, volume)
    """
    total = len(symbols)
    print(f"\n[step03] Fetching daily OHLCV for {total} stocks ...")
    print(f"         Date range: {from_date} to {to_date}")
    print(f"         Rate limit: {KITE_RATE_LIMIT} req/sec")

    all_frames = []
    fetched = 0
    skipped_checkpoint = 0
    skipped_no_token = 0
    errors = 0

    for i, symbol in enumerate(sorted(symbols), 1):
        if i % 25 == 0 or i == 1:
            print(f"\n  [{i}/{total}] Processing {symbol} ...")

        # Check checkpoint first
        cached = _load_checkpoint(symbol)
        if cached is not None:
            cached['symbol'] = symbol
            all_frames.append(cached)
            skipped_checkpoint += 1
            continue

        # Map symbol to instrument token
        token = sym_map.get(symbol)
        if token is None:
            for suffix in ['-EQ', '-BE', '-BZ']:
                alt = symbol + suffix
                if alt in sym_map:
                    token = sym_map[alt]
                    break
        if token is None:
            skipped_no_token += 1
            if skipped_no_token <= 20:
                print(f"    SKIP {symbol}: no instrument token found")
            continue

        # Fetch from Kite API
        df = fetch_daily_ohlcv(kite, token, symbol, from_date, to_date)
        if len(df) > 0:
            df['symbol'] = symbol
            _save_checkpoint(symbol, df)
            all_frames.append(df)
            fetched += 1
        else:
            errors += 1

        rate_limit_sleep(KITE_RATE_LIMIT)

        # Checkpoint every 50 stocks
        if i % 50 == 0:
            print(f"  Checkpoint: {fetched} fetched, {skipped_checkpoint} cached, "
                  f"{errors} errors so far")

    print(f"\n[step03] Fetch Summary:")
    print(f"    Total symbols:        {total}")
    print(f"    Fetched (API):        {fetched}")
    print(f"    Loaded (checkpoint):  {skipped_checkpoint}")
    print(f"    Skipped (no token):   {skipped_no_token}")
    print(f"    Errors:               {errors}")

    if all_frames:
        all_daily = pd.concat(all_frames, ignore_index=True)
        all_daily = all_daily.sort_values(['symbol', 'date']).reset_index(drop=True)
        return all_daily
    else:
        return pd.DataFrame(columns=['date', 'symbol', 'open', 'high', 'low',
                                     'close', 'volume'])


# ------------------------------------------------------------------
# Fetch index data
# ------------------------------------------------------------------

def fetch_index_data(kite, idx_map, from_date, to_date):
    """Fetch daily OHLCV for each index in KITE_INDEX_MAP.

    Returns pd.DataFrame with same schema as stock data.
    """
    print(f"\n[step03] Fetching index data for {len(idx_map)} indices ...")

    all_index_daily = []

    for idx_name, token in idx_map.items():
        col_key = KITE_INDEX_MAP.get(idx_name, idx_name.lower().replace(' ', '_'))
        ckpt_sym = f"INDEX_{col_key}"

        # Check checkpoint
        cached = _load_checkpoint(ckpt_sym)
        if cached is not None:
            cached['symbol'] = col_key
            all_index_daily.append(cached)
            print(f"    {idx_name}: loaded from checkpoint ({len(cached)} rows)")
            continue

        try:
            data = kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval="day",
            )
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date']).dt.date
                df['date'] = pd.to_datetime(df['date'])
                df['symbol'] = col_key

                expected_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']
                for col in expected_cols:
                    if col not in df.columns:
                        df[col] = np.nan
                df = df[expected_cols]

                _save_checkpoint(ckpt_sym, df)
                all_index_daily.append(df)
                print(f"    {idx_name}: {len(df)} daily candles")
            else:
                print(f"    {idx_name}: no data returned")
        except Exception as e:
            print(f"    {idx_name}: ERROR -- {e}")

        rate_limit_sleep(KITE_RATE_LIMIT)

    if all_index_daily:
        return pd.concat(all_index_daily, ignore_index=True)
    else:
        return pd.DataFrame()


# ------------------------------------------------------------------
# Aggregate daily -> monthly
# ------------------------------------------------------------------

def aggregate_to_monthly(daily_df):
    """Aggregate daily OHLCV to monthly frequency.

    For each symbol-month:
    - close: last trading day
    - high: max over month
    - low: min over month
    - open: first trading day
    - volume: sum of daily volumes
    - monthly_return: close pct_change month-over-month
    """
    if daily_df.empty:
        return pd.DataFrame()

    df = daily_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year_month_str'] = df['date'].dt.to_period('M').astype(str)

    grouped = df.groupby(['symbol', 'year_month_str'])

    monthly = grouped.agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
    ).reset_index()

    monthly = monthly.sort_values(['symbol', 'year_month_str']).reset_index(drop=True)

    # Monthly return: pct_change of close
    monthly['monthly_return'] = monthly.groupby('symbol')['close'].pct_change()

    return monthly


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STEP 03: Fetch Kite Connect OHLCV Data")
    print("=" * 70)

    # --- Cache check: skip if both outputs already exist ---
    if os.path.isfile(DAILY_OUTPUT) and os.path.isfile(MONTHLY_OUTPUT):
        try:
            daily_check = pd.read_csv(DAILY_OUTPUT, nrows=5)
            monthly_check = pd.read_csv(MONTHLY_OUTPUT, nrows=5)
            if len(daily_check) > 0 and len(monthly_check) > 0:
                # Count rows without loading full file
                with open(DAILY_OUTPUT, 'r') as f:
                    daily_rows = sum(1 for _ in f) - 1
                with open(MONTHLY_OUTPUT, 'r') as f:
                    monthly_rows = sum(1 for _ in f) - 1
                if daily_rows > 1000 and monthly_rows > 100:
                    print(f"[step03] Cache hit: daily ({daily_rows:,} rows), "
                          f"monthly ({monthly_rows:,} rows)")
                    print("[step03] Skipping fetch. Delete output files to re-fetch.")
                    return
        except Exception:
            pass

    # --- Load unified mapping to get symbol list ---
    if not os.path.exists(UNIFIED_MAP_CSV):
        print(f"[step03] ERROR: Unified mapping not found: {UNIFIED_MAP_CSV}")
        print("         Run step00_build_mapping.py first.")
        return

    unified_df = pd.read_csv(UNIFIED_MAP_CSV, low_memory=False)
    unified_df.columns = unified_df.columns.str.strip()
    print(f"\n[step03] Loaded unified mapping: {len(unified_df)} stocks")

    # Find symbol column
    sym_col = None
    for c in unified_df.columns:
        if c.upper() in ('SYMBOL', 'TICKER', 'NSE_SYMBOL'):
            sym_col = c
            break
    if sym_col is None:
        print("[step03] ERROR: No symbol column in unified mapping.")
        return

    symbols = unified_df[sym_col].dropna().astype(str).str.strip().str.upper().unique().tolist()
    symbols = [s for s in symbols if s and s != 'NAN' and len(s) <= 20]
    print(f"[step03] {len(symbols)} unique symbols to fetch")

    # --- Initialize Kite ---
    print("\n[step03] Initializing Kite Connect ...")
    kite = init_kite()
    if kite is None:
        print("[step03] Kite Connect not available. Skipping OHLCV fetch.")
        print("         To proceed without Kite, provide kite_ohlcv_daily.csv manually.")
        return

    # --- Build instrument maps ---
    sym_map, idx_map = build_symbol_token_map(kite)

    # --- Fetch stock OHLCV ---
    all_daily = fetch_all_stocks(kite, sym_map, symbols, DATE_START, DATE_END)

    # --- Fetch index data and append to daily ---
    if idx_map:
        index_daily = fetch_index_data(kite, idx_map, DATE_START, DATE_END)
        if not index_daily.empty:
            all_daily = pd.concat([all_daily, index_daily], ignore_index=True)
            print(f"[step03] Combined daily rows (stocks + indices): {len(all_daily):,}")

    # --- Save daily output ---
    if not all_daily.empty:
        all_daily = all_daily.drop_duplicates(subset=['symbol', 'date'], keep='last')
        all_daily.to_csv(DAILY_OUTPUT, index=False)
        print(f"\n[step03] Saved daily OHLCV: {DAILY_OUTPUT}")
        print(f"         {len(all_daily):,} rows, {all_daily['symbol'].nunique()} symbols")
        print(f"         Date range: {all_daily['date'].min()} to {all_daily['date'].max()}")
    else:
        print("[step03] WARNING: No daily data to save.")
        return

    # --- Aggregate to monthly ---
    print("\n[step03] Aggregating daily -> monthly ...")
    monthly_df = aggregate_to_monthly(all_daily)
    if not monthly_df.empty:
        monthly_df.to_csv(MONTHLY_OUTPUT, index=False)
        print(f"[step03] Saved monthly OHLCV: {MONTHLY_OUTPUT}")
        print(f"         {len(monthly_df):,} rows, {monthly_df['symbol'].nunique()} symbols")
    else:
        print("[step03] WARNING: No monthly data to save.")

    # --- Summary ---
    print(f"\n[step03] Summary:")
    print(f"  Daily:   {DAILY_OUTPUT}")
    print(f"  Monthly: {MONTHLY_OUTPUT}")
    print("[step03] Done.")


if __name__ == '__main__':
    main()
