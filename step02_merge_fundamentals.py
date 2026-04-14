"""
Step 02 -- Merge CMIE Fundamentals with Portfolio Data
======================================================

Joins CMIE monthly fundamental data onto the clean portfolio file using
ticker-based matching.  Builds a comprehensive ISIN->ticker map from the
unified mapping, then merges on (ticker, year_month_str).  Falls back to
fuzzy name matching for stocks that could not be resolved via ISIN.

Input:  PORTFOLIO_DIR / TEMPORAL_KG_READY.csv
Output: PORTFOLIO_DIR / portfolio_with_fundamentals.csv
"""

import sys
import os
import difflib

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FUNDAMENTALS_CSV, PORTFOLIO_DIR, MAPPINGS_DIR
from utils import coverage_report

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
FUNDAMENTAL_COLS = [
    'pe', 'pb', 'eps', 'beta', 'alpha', 'market_cap', 'roe',
    'debt_to_equity', 'profit_margin', 'bv_per_share',
    'total_returns', 'yield_pct',
    # Extended columns (may or may not exist)
    'industry_pe', 'industry_pb', 'profit_after_tax',
    'income_from_sales', 'total_assets', 'total_liabilities',
    'shares_traded',
]

INPUT_CSV = os.path.join(PORTFOLIO_DIR, 'TEMPORAL_KG_READY.csv')
OUTPUT_CSV = os.path.join(PORTFOLIO_DIR, 'portfolio_with_fundamentals.csv')
UNIFIED_MAP_CSV = os.path.join(MAPPINGS_DIR, 'unified_isin_symbol_map.csv')


# ------------------------------------------------------------------
# 1. Load portfolio
# ------------------------------------------------------------------
def load_portfolio(path):
    """Load the cleaned portfolio CSV from step01."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Portfolio file not found: {path}. Run step01 first.")
    df = pd.read_csv(path, low_memory=False)
    print(f"[step02] Loaded portfolio: {len(df):,} rows, {df.shape[1]} cols")
    # Ensure year_month_str exists
    if 'year_month_str' not in df.columns and 'date' in df.columns:
        df['year_month_str'] = pd.to_datetime(df['date']).dt.to_period('M').astype(str)
    return df


# ------------------------------------------------------------------
# 2. Build ISIN -> ticker mapping
# ------------------------------------------------------------------
def build_isin_ticker_map():
    """Build ISIN -> ticker mapping from unified_isin_symbol_map.csv.

    Returns dict {ISIN: ticker}
    """
    isin_to_ticker = {}

    if os.path.exists(UNIFIED_MAP_CSV):
        umap = pd.read_csv(UNIFIED_MAP_CSV, low_memory=False)
        umap.columns = umap.columns.str.strip()

        # Detect ISIN and symbol columns (flexible naming)
        isin_col = None
        sym_col = None
        for c in umap.columns:
            cu = c.upper()
            if cu == 'ISIN' or 'ISIN' in cu:
                isin_col = c
            if cu in ('SYMBOL', 'TICKER', 'NSE_SYMBOL'):
                sym_col = c

        if isin_col and sym_col:
            for _, row in umap.dropna(subset=[isin_col, sym_col]).iterrows():
                isin = str(row[isin_col]).strip().upper()
                sym = str(row[sym_col]).strip().upper()
                if isin and sym and isin != 'NAN' and sym != 'NAN':
                    isin_to_ticker[isin] = sym
            print(f"[step02] Unified mapping: {len(isin_to_ticker):,} ISIN->ticker entries")
        else:
            print(f"[step02] WARNING: Could not find ISIN/symbol columns in {UNIFIED_MAP_CSV}")
            print(f"         Available columns: {list(umap.columns)}")
    else:
        print(f"[step02] Unified mapping not found at {UNIFIED_MAP_CSV}")

    # Also try NSE ticker list from config
    try:
        from config import TICKER_LIST_CSV
        if os.path.exists(TICKER_LIST_CSV):
            tl = pd.read_csv(TICKER_LIST_CSV, low_memory=False)
            tl.columns = tl.columns.str.strip()
            isin_col = [c for c in tl.columns if 'ISIN' in c.upper()]
            sym_col = [c for c in tl.columns if c.upper() == 'SYMBOL']
            if isin_col and sym_col:
                before = len(isin_to_ticker)
                for _, row in tl.dropna(subset=[isin_col[0], sym_col[0]]).iterrows():
                    isin = str(row[isin_col[0]]).strip().upper()
                    sym = str(row[sym_col[0]]).strip().upper()
                    if isin and sym and isin != 'NAN':
                        isin_to_ticker.setdefault(isin, sym)
                print(f"[step02] NSE ticker list: added {len(isin_to_ticker) - before:,} "
                      f"new entries (total {len(isin_to_ticker):,})")
    except (ImportError, AttributeError):
        pass

    return isin_to_ticker


# ------------------------------------------------------------------
# 3. Load CMIE fundamentals
# ------------------------------------------------------------------
def load_fundamentals(path):
    """Load CMIE fundamentals CSV and derive year_month_str."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Fundamentals file not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    print(f"[step02] Loaded fundamentals: {len(df):,} rows, {df.shape[1]} cols")

    # Create year_month_str from the date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year_month_str'] = df['date'].dt.to_period('M').astype(str)
    elif 'year' in df.columns and 'month' in df.columns:
        df['year_month_str'] = (df['year'].astype(str) + '-'
                                + df['month'].astype(str).str.zfill(2))
    else:
        raise ValueError("Fundamentals CSV has neither 'date' nor 'year'+'month' columns")

    # Normalise ticker to uppercase
    if 'ticker' in df.columns:
        df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()

    print(f"[step02] Fundamentals date range: {df['year_month_str'].min()} to {df['year_month_str'].max()}")
    print(f"[step02] Unique tickers in fundamentals: {df['ticker'].nunique() if 'ticker' in df.columns else 'N/A'}")
    return df


# ------------------------------------------------------------------
# 4. Merge portfolio with fundamentals
# ------------------------------------------------------------------
def merge_fundamentals(portfolio, fundamentals, isin_to_ticker):
    """Map each portfolio row to its ticker via ISIN, then left-join
    fundamentals on (ticker, year_month_str)."""

    df = portfolio.copy()

    # Map ISIN -> ticker
    if 'ISIN' in df.columns:
        df['fund_ticker'] = (df['ISIN'].astype(str).str.strip().str.upper()
                             .map(isin_to_ticker))
    elif 'symbol' in df.columns:
        df['fund_ticker'] = df['symbol'].astype(str).str.strip().str.upper()
    else:
        print("[step02] WARNING: No ISIN or symbol column found in portfolio.")
        df['fund_ticker'] = np.nan

    mapped = df['fund_ticker'].notna().sum()
    total = len(df)
    print(f"[step02] Ticker mapping: {mapped:,}/{total:,} rows "
          f"({mapped / total * 100:.1f}%) have a ticker")

    # Prepare fundamentals for merge
    fund_cols_present = [c for c in FUNDAMENTAL_COLS if c in fundamentals.columns]
    if not fund_cols_present:
        print("[step02] WARNING: No fundamental columns found in CSV. "
              f"Available: {list(fundamentals.columns)}")
        return df

    fund_merge = fundamentals[['ticker', 'year_month_str'] + fund_cols_present].copy()
    fund_merge = fund_merge.drop_duplicates(subset=['ticker', 'year_month_str'], keep='first')

    # Left merge
    merged = df.merge(
        fund_merge,
        left_on=['fund_ticker', 'year_month_str'],
        right_on=['ticker', 'year_month_str'],
        how='left',
        suffixes=('', '_fund'),
    )

    # Drop the redundant ticker column
    if 'ticker' in merged.columns and 'fund_ticker' in merged.columns:
        merged.drop(columns=['ticker'], inplace=True, errors='ignore')

    matched = merged[fund_cols_present[0]].notna().sum()
    print(f"[step02] Fundamental merge: {matched:,}/{total:,} rows "
          f"({matched / total * 100:.1f}%) matched on (ticker, year_month)")

    return merged


# ------------------------------------------------------------------
# 5. Fuzzy matching for unmatched stocks
# ------------------------------------------------------------------
def fuzzy_fill(merged, fundamentals, cutoff=0.75, n=1):
    """For portfolio rows still lacking fundamentals, attempt fuzzy
    name matching against the fundamentals company column."""

    fund_cols_present = [c for c in FUNDAMENTAL_COLS if c in merged.columns]
    if not fund_cols_present:
        print("[step02] No fundamental columns to fuzzy-fill -- skipping.")
        return merged

    check_col = fund_cols_present[0]
    missing_mask = merged[check_col].isna()

    if not missing_mask.any():
        print("[step02] All rows already have fundamentals -- no fuzzy matching needed.")
        return merged

    # Determine name column in portfolio
    name_col = None
    for candidate in ['stock_name', 'stock_name_raw', 'NAME OF COMPANY']:
        if candidate in merged.columns:
            name_col = candidate
            break
    if name_col is None:
        print("[step02] No stock name column found -- skipping fuzzy match.")
        return merged

    # Build lookup: norm_name -> ticker
    name_to_ticker = {}
    fund_name_col = 'norm_name' if 'norm_name' in fundamentals.columns else 'company'
    if fund_name_col in fundamentals.columns:
        for _, row in fundamentals.drop_duplicates(
                subset=[fund_name_col]).dropna(subset=[fund_name_col]).iterrows():
            nm = str(row[fund_name_col]).strip().upper()
            tk = str(row.get('ticker', '')).strip().upper()
            if nm and tk and tk != 'NAN':
                name_to_ticker[nm] = tk

    if not name_to_ticker:
        print("[step02] No name->ticker map from fundamentals -- skipping fuzzy match.")
        return merged

    fund_names = list(name_to_ticker.keys())

    # Gather unique unmatched stock names
    unmatched_names = (merged.loc[missing_mask, name_col]
                       .dropna().astype(str).str.strip().str.upper().unique())
    print(f"[step02] Fuzzy matching {len(unmatched_names):,} unmatched stock names "
          f"against {len(fund_names):,} fundamental companies ...")

    fuzzy_map = {}
    match_count = 0
    for name in unmatched_names:
        matches = difflib.get_close_matches(name, fund_names, n=n, cutoff=cutoff)
        if matches:
            fuzzy_map[name] = name_to_ticker[matches[0]]
            match_count += 1

    print(f"[step02] Fuzzy matches found: {match_count:,}/{len(unmatched_names):,}")

    if not fuzzy_map:
        return merged

    # Apply fuzzy-matched tickers
    fuzzy_ticker = (merged[name_col].astype(str).str.strip().str.upper()
                    .map(fuzzy_map))
    fill_mask = merged['fund_ticker'].isna() & fuzzy_ticker.notna()
    merged.loc[fill_mask, 'fund_ticker'] = fuzzy_ticker[fill_mask]

    # Re-merge for fuzzy-matched rows
    fund_cols_keep = [c for c in FUNDAMENTAL_COLS if c in fundamentals.columns]
    fund_merge2 = fundamentals[['ticker', 'year_month_str'] + fund_cols_keep].copy()
    fund_merge2 = fund_merge2.drop_duplicates(subset=['ticker', 'year_month_str'], keep='first')

    rows_to_fill = merged.index[fill_mask & merged[check_col].isna()]
    if len(rows_to_fill) > 0:
        subset = merged.loc[rows_to_fill, ['fund_ticker', 'year_month_str']].copy()
        subset = subset.merge(
            fund_merge2,
            left_on=['fund_ticker', 'year_month_str'],
            right_on=['ticker', 'year_month_str'],
            how='left',
        )
        for col in fund_cols_keep:
            if col in subset.columns:
                merged.loc[rows_to_fill, col] = subset[col].values

    newly_filled = (~merged.loc[fill_mask, check_col].isna()).sum()
    print(f"[step02] Fuzzy fill: {newly_filled:,} additional rows got fundamentals")

    return merged


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("=" * 70)
    print("STEP 02: Merge CMIE Fundamentals with Portfolio")
    print("=" * 70)

    # 1. Load portfolio
    portfolio = load_portfolio(INPUT_CSV)

    # 2. Build ISIN -> ticker mapping
    print("\n[step02] Building ISIN -> ticker mapping ...")
    isin_to_ticker = build_isin_ticker_map()

    # 3. Load fundamentals
    print()
    if not os.path.isfile(FUNDAMENTALS_CSV):
        print(f"[step02] WARNING: Fundamentals CSV not found: {FUNDAMENTALS_CSV}")
        print("[step02] Saving portfolio as-is (no fundamentals merged).")
        portfolio.to_csv(OUTPUT_CSV, index=False)
        print(f"[step02] Saved: {OUTPUT_CSV}  ({len(portfolio):,} rows)")
        return

    fundamentals = load_fundamentals(FUNDAMENTALS_CSV)

    # 4. Merge
    print()
    merged = merge_fundamentals(portfolio, fundamentals, isin_to_ticker)

    # 5. Fuzzy matching fallback
    print()
    merged = fuzzy_fill(merged, fundamentals)

    # 6. Coverage report
    print("\n[step02] Coverage report after fundamental merge:")
    coverage_report(merged)

    # Final summary
    fund_cols_present = [c for c in FUNDAMENTAL_COLS if c in merged.columns]
    if fund_cols_present:
        any_fund = merged[fund_cols_present].notna().any(axis=1).sum()
        print(f"\n  Rows with at least one fundamental: "
              f"{any_fund:,}/{len(merged):,} ({any_fund / len(merged) * 100:.1f}%)")

    # 7. Save
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[step02] Saved: {OUTPUT_CSV}")
    print(f"         Shape: {merged.shape}")
    print("=" * 70)
    print("[step02] Done.")


if __name__ == '__main__':
    main()
