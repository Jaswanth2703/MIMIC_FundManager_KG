"""
Step 0: Build Unified ISIN-Symbol Mapping
==========================================
Input:  isin_symbol_mapping.csv (200+ stocks) + Ticker_List_NSE_India.csv (1,664 stocks)
Output: data/mappings/unified_isin_symbol_map.csv

Merges both ticker mapping files on ISIN to create a single comprehensive
mapping used by all downstream steps for Kite API lookups.
"""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (ISIN_MAPPING_CSV, TICKER_LIST_CSV, MAPPINGS_DIR,
                    EXISTING_TEMPORAL_CSV)


def load_isin_mapping():
    """Load the existing isin_symbol_mapping.csv (built from bhavcopy)."""
    if not os.path.exists(ISIN_MAPPING_CSV):
        print(f"  WARNING: {ISIN_MAPPING_CSV} not found")
        return pd.DataFrame(columns=['ISIN', 'symbol', 'stock_name'])

    df = pd.read_csv(ISIN_MAPPING_CSV)
    print(f"  isin_symbol_mapping.csv: {len(df)} rows")
    print(f"    Columns: {list(df.columns)}")
    return df


def load_ticker_list():
    """Load the NSE Ticker_List_NSE_India.csv."""
    if not os.path.exists(TICKER_LIST_CSV):
        print(f"  WARNING: {TICKER_LIST_CSV} not found")
        return pd.DataFrame()

    df = pd.read_csv(TICKER_LIST_CSV)
    print(f"  Ticker_List_NSE_India.csv: {len(df)} rows")
    print(f"    Columns: {list(df.columns)}")

    # Standardize column names
    col_map = {
        'SYMBOL': 'nse_symbol',
        'NAME OF COMPANY': 'company_name',
        ' SERIES': 'nse_series',
        ' DATE OF LISTING': 'listing_date',
        ' FACE VALUE': 'face_value',
        ' ISIN NUMBER': 'ISIN',
        'YahooEquiv': 'yahoo_suffix',
        'Yahoo_Equivalent_Code': 'yahoo_code',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Clean ISIN column
    if 'ISIN' in df.columns:
        df['ISIN'] = df['ISIN'].astype(str).str.strip()
        df = df[df['ISIN'].str.startswith('INE')]

    # Clean series
    if 'nse_series' in df.columns:
        df['nse_series'] = df['nse_series'].astype(str).str.strip()

    return df


def build_unified_mapping():
    """Merge both sources into a single ISIN-symbol map."""
    print("\n--- Loading Sources ---")
    isin_df = load_isin_mapping()
    ticker_df = load_ticker_list()

    print("\n--- Building Unified Mapping ---")

    # Start with isin_symbol_mapping as base (has symbol + stock_name)
    unified = isin_df[['ISIN', 'symbol', 'stock_name']].copy()
    unified = unified.drop_duplicates(subset='ISIN', keep='first')

    # Merge with ticker list on ISIN for additional metadata
    if len(ticker_df) > 0 and 'ISIN' in ticker_df.columns:
        ticker_cols = ['ISIN']
        for col in ['nse_symbol', 'company_name', 'nse_series', 'listing_date',
                     'face_value', 'yahoo_code']:
            if col in ticker_df.columns:
                ticker_cols.append(col)

        ticker_dedup = ticker_df[ticker_cols].drop_duplicates(subset='ISIN', keep='first')

        unified = unified.merge(ticker_dedup, on='ISIN', how='outer')

        # Fill missing symbols: prefer isin_mapping symbol, fallback to ticker list
        if 'nse_symbol' in unified.columns:
            unified['symbol'] = unified['symbol'].fillna(unified['nse_symbol'])
        if 'company_name' in unified.columns:
            unified['stock_name'] = unified['stock_name'].fillna(unified['company_name'])

    # Clean up
    unified = unified.drop_duplicates(subset='ISIN', keep='first')
    unified = unified[unified['ISIN'].str.startswith('INE')]

    # Ensure symbol column is clean
    unified['symbol'] = unified['symbol'].astype(str).str.strip()
    unified = unified[unified['symbol'].notna() & (unified['symbol'] != '') & (unified['symbol'] != 'nan')]

    # Keep useful columns
    keep_cols = ['ISIN', 'symbol', 'stock_name']
    for col in ['nse_series', 'listing_date', 'face_value', 'yahoo_code']:
        if col in unified.columns:
            keep_cols.append(col)
    unified = unified[[c for c in keep_cols if c in unified.columns]]

    print(f"\n  Unified mapping: {len(unified)} stocks")
    print(f"  From isin_mapping only: ~{len(isin_df)} stocks")
    print(f"  From ticker_list only: ~{len(ticker_df)} stocks")
    print(f"  Union coverage: {len(unified)} unique ISINs with symbols")

    return unified


def check_portfolio_coverage(unified, portfolio_csv=None):
    """Check how many portfolio ISINs are covered by the mapping."""
    csv_path = portfolio_csv or EXISTING_TEMPORAL_CSV
    if not os.path.exists(csv_path):
        print("\n  Portfolio CSV not found — skipping coverage check")
        return

    portfolio = pd.read_csv(csv_path, usecols=['ISIN'])
    portfolio_isins = set(portfolio['ISIN'].unique())
    mapped_isins = set(unified['ISIN'].unique())

    covered = portfolio_isins & mapped_isins
    missing = portfolio_isins - mapped_isins

    print(f"\n  Portfolio Coverage:")
    print(f"    Portfolio unique ISINs: {len(portfolio_isins)}")
    print(f"    Mapped ISINs:          {len(covered)} ({100*len(covered)/len(portfolio_isins):.1f}%)")
    print(f"    Missing ISINs:         {len(missing)}")
    if missing and len(missing) <= 20:
        for isin in sorted(missing):
            print(f"      {isin}")

    # Return for downstream filtering
    return portfolio_isins


def filter_mapping_to_portfolio_isins(unified, portfolio_isins):
    """Filter unified mapping down to only ISINs present in the portfolio."""
    if portfolio_isins is None:
        return unified

    before = len(unified)
    filtered = unified[unified['ISIN'].isin(portfolio_isins)].copy()
    after = len(filtered)

    print(f"\n  Filtered mapping to portfolio universe:")
    print(f"    Before: {before} mapped ISINs")
    print(f"    After:  {after} mapped ISINs")
    return filtered


def main():
    print("=" * 70)
    print("STEP 0: BUILD UNIFIED ISIN-SYMBOL MAPPING")
    print("=" * 70)

    unified = build_unified_mapping()

    # Check portfolio coverage
    portfolio_isins = check_portfolio_coverage(unified)

    # IMPORTANT: shrink mapping to only stocks actually present in Step01 output
    # This prevents step03 from fetching OHLCV for ~1977 stocks.
    if portfolio_isins is not None:
        unified = filter_mapping_to_portfolio_isins(unified, portfolio_isins)

    # Save
    output_path = os.path.join(MAPPINGS_DIR, 'unified_isin_symbol_map.csv')
    unified.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")
    print(f"  {len(unified)} stocks mapped")

    print(f"\n  Next: Run step01_prepare_portfolio.py")
    return unified


if __name__ == '__main__':
    unified = main()
