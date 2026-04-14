"""
Step 06 -- Collect Macro Indicators (Monthly)
=============================================

Combines macro data from multiple sources into a single monthly DataFrame:
- Kite API: index OHLCV for KITE_INDEX_MAP indices -> monthly returns
- yfinance: YFINANCE_GLOBAL_MAP tickers (crude oil, brent, gold_usd, etc.)
- Local files: GOLD_USD_CSV, GOLD_INR_CSV, INTEREST_RATE_XLS, GDP_XLS, INFLATION_XLS
- Config hardcoded: REPO_RATE_MONTHLY, CPI_INFLATION_MONTHLY, GDP_GROWTH_QUARTERLY

Forward-fills quarterly/annual data to monthly.

Output: MACRO_DIR / macro_indicators_monthly.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MARKET_DIR, MACRO_DIR, DATE_START, DATE_END,
    GOLD_USD_CSV, GOLD_INR_CSV, INTEREST_RATE_XLS, GDP_XLS, INFLATION_XLS,
    REPO_RATE_MONTHLY, CPI_INFLATION_MONTHLY, GDP_GROWTH_QUARTERLY,
    KITE_INDEX_MAP, YFINANCE_GLOBAL_MAP,
)
from utils import coverage_report, quarter_from_month

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

OUTPUT_CSV = os.path.join(MACRO_DIR, 'macro_indicators_monthly.csv')


# ============================================================
# Data loaders
# ============================================================

def load_gold_data():
    """Load gold price CSVs (USD and INR) and return monthly DataFrames."""
    frames = {}

    for csv_path, col_name in [(GOLD_USD_CSV, 'gold_usd'), (GOLD_INR_CSV, 'gold_inr')]:
        if not os.path.exists(csv_path):
            print(f"  WARNING: {col_name} file not found: {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
            print(f"  {col_name}: {len(df)} rows from {os.path.basename(csv_path)}")

            # Auto-detect date and value columns
            date_col = [c for c in df.columns if 'date' in c.lower()]
            value_col = [c for c in df.columns
                         if any(kw in c.lower() for kw in ['gold', 'price', 'usd', 'inr'])
                         and 'date' not in c.lower()]

            if date_col and value_col:
                df = df.rename(columns={date_col[0]: 'Date', value_col[0]: col_name})
            elif len(df.columns) >= 2:
                df.columns = ['Date', col_name] + list(df.columns[2:])

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            df['year_month_str'] = df['Date'].dt.to_period('M').astype(str)

            monthly = df.groupby('year_month_str')[col_name].last().reset_index()
            frames[col_name] = monthly
        except Exception as e:
            print(f"  WARNING: Could not load {col_name}: {e}")

    return frames


def load_local_xls_data():
    """Load interest rate, GDP, and inflation from local XLS files."""
    frames = {}

    # Interest rate XLS
    if os.path.exists(INTEREST_RATE_XLS):
        try:
            df = pd.read_excel(INTEREST_RATE_XLS)
            print(f"  Interest rate XLS: {len(df)} rows")
            # Auto-detect columns
            date_col = [c for c in df.columns if 'date' in str(c).lower() or 'year' in str(c).lower()]
            val_col = [c for c in df.columns if 'rate' in str(c).lower() or 'interest' in str(c).lower()]
            if date_col and val_col:
                df = df.rename(columns={date_col[0]: 'Date', val_col[0]: 'interest_rate_xls'})
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])
                df['year_month_str'] = df['Date'].dt.to_period('M').astype(str)
                monthly = df.groupby('year_month_str')['interest_rate_xls'].last().reset_index()
                frames['interest_rate_xls'] = monthly
        except Exception as e:
            print(f"  WARNING: Could not load interest rate XLS: {e}")
    else:
        print(f"  Interest rate XLS not found: {INTEREST_RATE_XLS}")

    # GDP XLS
    if os.path.exists(GDP_XLS):
        try:
            df = pd.read_excel(GDP_XLS)
            print(f"  GDP XLS: {len(df)} rows")
            date_col = [c for c in df.columns if 'date' in str(c).lower() or 'year' in str(c).lower()]
            val_col = [c for c in df.columns if 'gdp' in str(c).lower() or 'growth' in str(c).lower()]
            if date_col and val_col:
                df = df.rename(columns={date_col[0]: 'Date', val_col[0]: 'gdp_xls'})
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])
                df['year_month_str'] = df['Date'].dt.to_period('M').astype(str)
                monthly = df.groupby('year_month_str')['gdp_xls'].last().reset_index()
                frames['gdp_xls'] = monthly
        except Exception as e:
            print(f"  WARNING: Could not load GDP XLS: {e}")
    else:
        print(f"  GDP XLS not found: {GDP_XLS}")

    # Inflation XLS
    if os.path.exists(INFLATION_XLS):
        try:
            df = pd.read_excel(INFLATION_XLS)
            print(f"  Inflation XLS: {len(df)} rows")
            date_col = [c for c in df.columns if 'date' in str(c).lower() or 'year' in str(c).lower()]
            val_col = [c for c in df.columns if 'inflation' in str(c).lower() or 'cpi' in str(c).lower()]
            if date_col and val_col:
                df = df.rename(columns={date_col[0]: 'Date', val_col[0]: 'inflation_xls'})
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.dropna(subset=['Date'])
                df['year_month_str'] = df['Date'].dt.to_period('M').astype(str)
                monthly = df.groupby('year_month_str')['inflation_xls'].last().reset_index()
                frames['inflation_xls'] = monthly
        except Exception as e:
            print(f"  WARNING: Could not load Inflation XLS: {e}")
    else:
        print(f"  Inflation XLS not found: {INFLATION_XLS}")

    return frames


def load_rbi_macro():
    """Load hardcoded RBI macro data from config dicts."""
    repo_df = pd.DataFrame(
        list(REPO_RATE_MONTHLY.items()),
        columns=['year_month_str', 'repo_rate'],
    )
    print(f"  Repo rate: {len(repo_df)} months")

    cpi_df = pd.DataFrame(
        list(CPI_INFLATION_MONTHLY.items()),
        columns=['year_month_str', 'cpi_inflation'],
    )
    print(f"  CPI inflation: {len(cpi_df)} months")

    return repo_df, cpi_df


def load_gdp_data():
    """Load GDP growth from GDP_GROWTH_QUARTERLY, forward-fill to monthly."""
    gdp_records = []
    for qkey, growth in GDP_GROWTH_QUARTERLY.items():
        parts = qkey.split('-')
        q_num = int(parts[0][1])
        year = int(parts[1])
        start_month = (q_num - 1) * 3 + 1
        for m_offset in range(3):
            month = start_month + m_offset
            ym = f"{year}-{month:02d}"
            gdp_records.append({'year_month_str': ym, 'gdp_growth': growth})

    gdp_df = pd.DataFrame(gdp_records)
    print(f"  GDP growth: {len(gdp_df)} months (quarterly forward-filled)")
    return gdp_df


def load_index_returns():
    """Load index monthly returns from step03 output (Kite indices)."""
    index_path = os.path.join(MARKET_DIR, 'kite_ohlcv_monthly.csv')
    if not os.path.exists(index_path):
        # Try alternative path
        index_path = os.path.join(MARKET_DIR, 'index_data_monthly.csv')
    if not os.path.exists(index_path):
        print(f"  WARNING: No index data found in {MARKET_DIR}")
        return pd.DataFrame()

    df = pd.read_csv(index_path, low_memory=False)
    print(f"  Index data loaded: {len(df)} rows")

    # Ensure year_month_str
    if 'year_month_str' not in df.columns and 'year_month' in df.columns:
        df['year_month_str'] = df['year_month'].astype(str)
    elif 'year_month_str' not in df.columns:
        for col in ['date', 'Date']:
            if col in df.columns:
                df['year_month_str'] = pd.to_datetime(df[col]).dt.to_period('M').astype(str)
                break

    if 'year_month_str' not in df.columns or 'symbol' not in df.columns:
        print("  WARNING: Index data missing required columns (year_month_str, symbol)")
        return pd.DataFrame()

    # Filter to only index symbols
    index_symbols = set(KITE_INDEX_MAP.values())
    idx_df = df[df['symbol'].isin(index_symbols)].copy()

    if idx_df.empty:
        print("  WARNING: No index symbols found in OHLCV monthly data")
        return pd.DataFrame()

    # Pivot: each index becomes a column (close and monthly_return)
    pivot_data = {}
    for sym in idx_df['symbol'].unique():
        sym_data = idx_df[idx_df['symbol'] == sym][['year_month_str', 'close', 'monthly_return']].copy()
        sym_data = sym_data.set_index('year_month_str')
        if 'close' in sym_data.columns:
            pivot_data[f"{sym}_close"] = sym_data['close']
        if 'monthly_return' in sym_data.columns:
            pivot_data[f"{sym}_return"] = sym_data['monthly_return']

    if not pivot_data:
        return pd.DataFrame()

    result = pd.DataFrame(pivot_data)
    result.index.name = 'year_month_str'
    result = result.reset_index()
    print(f"  Index features: {len(result)} months, {len(result.columns) - 1} columns")
    return result


def load_yfinance_global():
    """Download global macro data from yfinance. Uses YFINANCE_GLOBAL_MAP from config."""
    try:
        import yfinance as yf
    except ImportError:
        print("  WARNING: yfinance not installed. Skipping global macro data.")
        print("    Install with: pip install yfinance")
        return pd.DataFrame()

    print(f"\n  Downloading global macro data from yfinance ...")
    print(f"    Tickers: {list(YFINANCE_GLOBAL_MAP.keys())}")
    print(f"    Period: {DATE_START} to {DATE_END}")

    all_series = {}

    for name, ticker in YFINANCE_GLOBAL_MAP.items():
        try:
            data = yf.download(
                ticker,
                start=DATE_START,
                end=DATE_END,
                progress=False,
                auto_adjust=True,
            )
            if data.empty:
                print(f"    WARNING: No data for {name} ({ticker})")
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            monthly = data['Close'].resample('ME').last()
            monthly.name = name
            all_series[name] = monthly

            # Also compute monthly returns
            monthly_ret = monthly.pct_change()
            monthly_ret.name = f"{name}_return"
            all_series[f"{name}_return"] = monthly_ret

            print(f"    {name} ({ticker}): {len(monthly)} months")
        except Exception as e:
            print(f"    WARNING: Failed to download {name} ({ticker}): {e}")

    if not all_series:
        return pd.DataFrame()

    df = pd.DataFrame(all_series)
    df.index.name = 'Date'
    df = df.reset_index()
    df['year_month_str'] = pd.to_datetime(df['Date']).dt.to_period('M').astype(str)
    df = df.drop(columns=['Date'])

    return df


# ============================================================
# Combine all macro data
# ============================================================

def combine_macro_data(gold_frames, xls_frames, repo_df, cpi_df, gdp_df,
                       index_df, yf_df):
    """Merge all macro data sources on year_month_str.
    Forward-fill quarterly/annual data to monthly.
    """
    # Start with a complete month range
    all_months = pd.date_range(start=DATE_START, end=DATE_END, freq='MS')
    base = pd.DataFrame({
        'year_month_str': [d.strftime('%Y-%m') for d in all_months]
    })
    print(f"\n  Base month range: {base['year_month_str'].iloc[0]} to "
          f"{base['year_month_str'].iloc[-1]} ({len(base)} months)")

    # Merge gold data
    for key, gdf in gold_frames.items():
        base = base.merge(gdf, on='year_month_str', how='left')

    # Merge local XLS data
    for key, xdf in xls_frames.items():
        base = base.merge(xdf, on='year_month_str', how='left')

    # Merge RBI macro
    base = base.merge(repo_df, on='year_month_str', how='left')
    base = base.merge(cpi_df, on='year_month_str', how='left')

    # Merge GDP
    base = base.merge(gdp_df, on='year_month_str', how='left')

    # Merge index data
    if not index_df.empty and 'year_month_str' in index_df.columns:
        base = base.merge(index_df, on='year_month_str', how='left')

    # Merge yfinance global data
    if not yf_df.empty and 'year_month_str' in yf_df.columns:
        base = base.merge(yf_df, on='year_month_str', how='left')

    # Forward-fill gaps in all numeric columns
    numeric_cols = base.select_dtypes(include=[np.number]).columns.tolist()
    base[numeric_cols] = base[numeric_cols].ffill()

    # Compute derived features
    # Pct change for level variables (prices, indices)
    level_vars = [c for c in numeric_cols if c not in [
        'repo_rate', 'cpi_inflation', 'gdp_growth',
        'real_interest_rate', 'repo_rate_change',
    ] and '_return' not in c and '_pct' not in c]

    for col in level_vars:
        pct_col = f"{col}_pct"
        if pct_col not in base.columns:
            base[pct_col] = base[col].pct_change()

    # Real interest rate: repo_rate - cpi_inflation
    if 'repo_rate' in base.columns and 'cpi_inflation' in base.columns:
        base['real_interest_rate'] = base['repo_rate'] - base['cpi_inflation']

    # Repo rate change
    if 'repo_rate' in base.columns:
        base['repo_rate_change'] = base['repo_rate'].diff()

    return base


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("STEP 06: Macro Indicators (Monthly)")
    print("=" * 70)

    # --- Cache check ---
    if os.path.isfile(OUTPUT_CSV):
        try:
            cached = pd.read_csv(OUTPUT_CSV, nrows=5)
            if len(cached) > 0:
                with open(OUTPUT_CSV, 'r') as f:
                    row_count = sum(1 for _ in f) - 1
                if row_count > 10:
                    print(f"[step06] Cache hit: {OUTPUT_CSV} ({row_count:,} rows)")
                    print("[step06] Skipping. Delete output file to recompute.")
                    return
        except Exception:
            pass

    # --- Load gold data ---
    print("\n1. Loading gold price data ...")
    gold_frames = load_gold_data()

    # --- Load local XLS files ---
    print("\n2. Loading local XLS data (interest rate, GDP, inflation) ...")
    xls_frames = load_local_xls_data()

    # --- Load RBI macro (hardcoded) ---
    print("\n3. Loading RBI macro data (hardcoded) ...")
    repo_df, cpi_df = load_rbi_macro()

    # --- Load GDP ---
    print("\n4. Loading GDP growth data ...")
    gdp_df = load_gdp_data()

    # --- Load index data from step03 ---
    print("\n5. Loading index data (from step03) ...")
    index_df = load_index_returns()

    # --- Load yfinance global data ---
    print("\n6. Loading global macro data (yfinance) ...")
    yf_df = load_yfinance_global()

    # --- Combine all ---
    print("\n7. Combining all macro data sources ...")
    df_macro = combine_macro_data(
        gold_frames, xls_frames, repo_df, cpi_df, gdp_df, index_df, yf_df
    )

    print(f"\n  Combined macro dataset: {df_macro.shape}")
    print(f"  Columns ({len(df_macro.columns)}):")
    for col in df_macro.columns:
        non_null = df_macro[col].notna().sum()
        print(f"    {col:35s}: {non_null}/{len(df_macro)} non-null")

    # --- Coverage report ---
    coverage_report(df_macro)

    # --- Save output ---
    df_macro.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[step06] Saved: {OUTPUT_CSV}")
    print(f"         Shape: {df_macro.shape}")

    # Print sample
    print("\n[step06] Sample output (first 5 rows):")
    print(df_macro.head().to_string(index=False))

    print("\n[step06] Done.")


if __name__ == '__main__':
    main()
