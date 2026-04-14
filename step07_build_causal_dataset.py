"""
Step 07 -- Build Causal Discovery Dataset
==========================================

Merges all data sources into a single panel dataset for causal discovery.

Input:
- PORTFOLIO_DIR / TEMPORAL_KG_READY.csv       (step01)
- PORTFOLIO_DIR / portfolio_with_fundamentals.csv  (step02)
- MARKET_DIR / technical_indicators_monthly.csv    (step04)
- SENTIMENT_DIR / finbert_monthly_sentiment.csv    (step05)
- MACRO_DIR / macro_indicators_monthly.csv         (step06)

Output:
- FEATURES_DIR / CAUSAL_DISCOVERY_DATASET.csv

Join keys:
- Portfolio base: (Fund_Name, ISIN, year_month_str)
- Fundamentals: on (ISIN, year_month_str) -- already merged in step02
- Technicals: on (symbol, year_month_str)
- Sentiment: on (ISIN, year_month_str) or (symbol, year_month_str)
- Macro: on (year_month_str) -- broadcast to all rows
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PORTFOLIO_DIR, MARKET_DIR, SENTIMENT_DIR, MACRO_DIR, FEATURES_DIR,
)
from utils import coverage_report

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

OUTPUT_CSV = os.path.join(FEATURES_DIR, 'CAUSAL_DISCOVERY_DATASET.csv')


# ============================================================
# Data loaders (with graceful fallback)
# ============================================================

def load_portfolio():
    """Load the portfolio base table (with fundamentals if available)."""
    # Prefer portfolio_with_fundamentals.csv (step02 output)
    path_fund = os.path.join(PORTFOLIO_DIR, 'portfolio_with_fundamentals.csv')
    path_base = os.path.join(PORTFOLIO_DIR, 'TEMPORAL_KG_READY.csv')

    if os.path.exists(path_fund):
        path = path_fund
        print(f"[step07] Loading portfolio (with fundamentals): {path}")
    elif os.path.exists(path_base):
        path = path_base
        print(f"[step07] Loading portfolio (base): {path}")
    else:
        print(f"[step07] ERROR: No portfolio file found in {PORTFOLIO_DIR}")
        print("         Run step01 and step02 first.")
        return None

    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Ensure year_month_str exists
    if 'year_month_str' not in df.columns:
        for col in ['Date', 'date', 'month', 'period']:
            if col in df.columns:
                df['date_parsed'] = pd.to_datetime(df[col], errors='coerce')
                df['year_month_str'] = df['date_parsed'].dt.to_period('M').astype(str)
                df = df.drop(columns=['date_parsed'])
                break

    if 'year_month_str' in df.columns:
        print(f"  Months: {df['year_month_str'].nunique()} unique")
    if 'ISIN' in df.columns:
        print(f"  ISINs: {df['ISIN'].nunique()} unique")
    if 'Fund_Name' in df.columns:
        print(f"  Funds: {df['Fund_Name'].nunique()}")

    return df


def load_technicals():
    """Load technical indicators (step04 output)."""
    path = os.path.join(MARKET_DIR, 'technical_indicators_monthly.csv')
    if not os.path.exists(path):
        print(f"[step07] WARNING: Technical indicators not found: {path}")
        print("         Run step04 first. Proceeding without technicals.")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    print(f"[step07] Technicals loaded: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    return df


def load_sentiment():
    path = os.path.join(SENTIMENT_DIR, 'finbert_monthly_sentiment.csv')
    if not os.path.exists(path):
        print(f"[step07] WARNING: Sentiment data not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    print(f"[step07] Sentiment loaded: {len(df):,} rows")

    # FIX: if ISIN is empty but symbol looks like ISIN, move it
    if 'ISIN' in df.columns and df['ISIN'].isna().all():
        if 'symbol' in df.columns:
            # If symbol looks like an ISIN (starts with INE...), treat it as ISIN
            isin_like = df['symbol'].astype(str).str.match(r'^INE[0-9A-Z]{9}$', na=False)
            if isin_like.mean() > 0.5:
                print("[step07] Fixing sentiment: symbol column contains ISINs. Moving -> ISIN.")
                df['ISIN'] = df['symbol']
                df.drop(columns=['symbol'], inplace=True)

    # Standardize keys
    if 'ISIN' in df.columns:
        df['ISIN'] = df['ISIN'].astype("string").str.strip()
    if 'year_month_str' in df.columns:
        df['year_month_str'] = df['year_month_str'].astype(str).str.strip()

    return df


def load_macro():
    """Load macro indicators (step06 output)."""
    path = os.path.join(MACRO_DIR, 'macro_indicators_monthly.csv')
    if not os.path.exists(path):
        print(f"[step07] WARNING: Macro indicators not found: {path}")
        print("         Run step06 first. Proceeding without macro.")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    print(f"[step07] Macro loaded: {len(df):,} rows, {len(df.columns)} columns")
    return df


# ============================================================
# Join logic
# ============================================================

def _safe_merge(left, right, on, how='left', suffix_tag=''):
    """Merge with automatic handling of overlapping column names."""
    overlap = set(right.columns) - set(on) - set(left.columns)
    rename_cols = set(right.columns) & set(left.columns) - set(on)

    if rename_cols:
        right = right.rename(columns={c: f"{c}_{suffix_tag}" for c in rename_cols})

    return left.merge(right, on=on, how=how)


def join_datasets(portfolio_df, tech_df, sent_df, macro_df):
    """Left-join all data sources onto the portfolio base.

    Join keys:
    - Technicals: (fund_ticker/symbol, year_month_str)
    - Sentiment:  (ISIN, year_month_str) or (symbol, year_month_str)
    - Macro:      (year_month_str) -- broadcast to all rows
    """
    df = portfolio_df.copy()
    initial_rows = len(df)
    join_report = []

    # Determine symbol column in portfolio
    sym_col = None
    for c in ['fund_ticker', 'symbol']:
        if c in df.columns:
            sym_col = c
            break

    # --- Join technicals on (symbol, year_month_str) ---
    if not tech_df.empty and sym_col and 'year_month_str' in df.columns:
        tech_cols_orig = [c for c in tech_df.columns if c not in ['symbol', 'year_month_str']]

        # Handle overlapping column names
        overlap = set(tech_cols_orig) & set(df.columns)
        if overlap:
            tech_df = tech_df.rename(columns={c: f"tech_{c}" for c in overlap})
        tech_cols = [c for c in tech_df.columns if c not in ['symbol', 'year_month_str']]

        df = df.merge(
            tech_df,
            left_on=[sym_col, 'year_month_str'],
            right_on=['symbol', 'year_month_str'],
            how='left',
            suffixes=('', '_tech'),
        )
        # Drop duplicate symbol column if created
        if 'symbol_tech' in df.columns:
            df.drop(columns=['symbol_tech'], inplace=True)
        if 'symbol' not in df.columns and sym_col != 'symbol':
            pass  # symbol from technicals side already dropped

        matched = df[tech_cols[0]].notna().sum() if tech_cols else 0
        pct = matched / len(df) * 100 if len(df) > 0 else 0
        join_report.append(f"  Technicals: {matched:,}/{len(df):,} rows ({pct:.1f}%)")
        print(f"[step07] Joined technicals: +{len(tech_cols)} columns, {matched:,} matches")
    else:
        join_report.append("  Technicals: SKIPPED")
        print("[step07] Technicals: SKIPPED")

    # --- Join sentiment on (ISIN, year_month_str) ---
    if not sent_df.empty and 'year_month_str' in df.columns:
        sent_cols_orig = [c for c in sent_df.columns
                          if c not in ['ISIN', 'symbol', 'year_month_str']]

        # Handle overlapping column names
        overlap = set(sent_cols_orig) & set(df.columns)
        if overlap:
            sent_df = sent_df.rename(columns={c: f"sent_{c}" for c in overlap})
        sent_cols = [c for c in sent_df.columns
                     if c not in ['ISIN', 'symbol', 'year_month_str']]

        # Try joining on ISIN first, then symbol
        if 'ISIN' in df.columns and 'ISIN' in sent_df.columns:
            df = df.merge(
                sent_df.drop(columns=['symbol'], errors='ignore'),
                on=['ISIN', 'year_month_str'],
                how='left',
                suffixes=('', '_sent'),
            )
        elif sym_col and 'symbol' in sent_df.columns:
            df = df.merge(
                sent_df.drop(columns=['ISIN'], errors='ignore'),
                left_on=[sym_col, 'year_month_str'],
                right_on=['symbol', 'year_month_str'],
                how='left',
                suffixes=('', '_sent'),
            )

        matched = df[sent_cols[0]].notna().sum() if sent_cols else 0
        pct = matched / len(df) * 100 if len(df) > 0 else 0
        join_report.append(f"  Sentiment:  {matched:,}/{len(df):,} rows ({pct:.1f}%)")
        print(f"[step07] Joined sentiment: +{len(sent_cols)} columns, {matched:,} matches")
    else:
        join_report.append("  Sentiment:  SKIPPED")
        print("[step07] Sentiment: SKIPPED")

    # --- Join macro on (year_month_str) -- broadcast ---
    if not macro_df.empty and 'year_month_str' in df.columns:
        macro_cols_orig = [c for c in macro_df.columns if c != 'year_month_str']

        overlap = set(macro_cols_orig) & set(df.columns)
        if overlap:
            macro_df = macro_df.rename(columns={c: f"macro_{c}" for c in overlap})
        macro_cols = [c for c in macro_df.columns if c != 'year_month_str']

        df = df.merge(macro_df, on='year_month_str', how='left')

        matched = df[macro_cols[0]].notna().sum() if macro_cols else 0
        pct = matched / len(df) * 100 if len(df) > 0 else 0
        join_report.append(f"  Macro:      {matched:,}/{len(df):,} rows ({pct:.1f}%)")
        print(f"[step07] Joined macro: +{len(macro_cols)} columns, {matched:,} matches")
    else:
        join_report.append("  Macro:      SKIPPED")
        print("[step07] Macro: SKIPPED")

    new_cols = len(df.columns) - len(portfolio_df.columns)
    print(f"\n[step07] Total new columns added: {new_cols}")
    assert len(df) == initial_rows, f"Row count changed: {initial_rows} -> {len(df)}"

    return df, join_report


# ============================================================
# Comprehensive coverage report
# ============================================================

def print_coverage_report(df, join_report):
    """Print comprehensive quality report."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE COVERAGE REPORT")
    print("=" * 60)

    print(f"\nDataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    if 'year_month_str' in df.columns:
        print(f"Month range: {df['year_month_str'].min()} to {df['year_month_str'].max()}")
        print(f"Unique months: {df['year_month_str'].nunique()}")
    if 'ISIN' in df.columns:
        print(f"Unique ISINs: {df['ISIN'].nunique()}")
    if 'Fund_Name' in df.columns:
        print(f"Unique funds: {df['Fund_Name'].nunique()}")

    print("\nJoin Results:")
    for line in join_report:
        print(line)

    # Group columns by coverage level
    coverage_data = []
    for col in sorted(df.columns):
        pct = df[col].notna().mean() * 100
        coverage_data.append((col, pct))

    high = [(c, p) for c, p in coverage_data if p >= 80]
    medium = [(c, p) for c, p in coverage_data if 50 <= p < 80]
    low = [(c, p) for c, p in coverage_data if 0 < p < 50]
    empty = [(c, p) for c, p in coverage_data if p == 0]

    print(f"\n  High coverage (>=80%): {len(high)} columns")
    for col, pct in high[:20]:
        print(f"    {col:40s}: {pct:5.1f}%")
    if len(high) > 20:
        print(f"    ... and {len(high) - 20} more")

    print(f"\n  Medium coverage (50-80%): {len(medium)} columns")
    for col, pct in medium:
        print(f"    {col:40s}: {pct:5.1f}%")

    print(f"\n  Low coverage (<50%): {len(low)} columns")
    for col, pct in low:
        print(f"    {col:40s}: {pct:5.1f}%")

    if empty:
        print(f"\n  Empty columns (0%): {len(empty)}")
        for col, pct in empty[:10]:
            print(f"    {col}")

    # Overall quality
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        overall_coverage = df[numeric_cols].notna().mean().mean() * 100
        print(f"\n  Overall numeric coverage: {overall_coverage:.1f}%")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("STEP 07: Build Causal Discovery Dataset")
    print("=" * 70)

    # --- Load all data sources ---
    print("\n1. Loading data sources ...")

    print("\n  [Portfolio - Base Table]")
    portfolio_df = load_portfolio()
    if portfolio_df is None:
        return

    print("\n  [Technical Indicators]")
    tech_df = load_technicals()

    print("\n  [Sentiment Data]")
    sent_df = load_sentiment()

    print("\n  [Macro Indicators]")
    macro_df = load_macro()

    # --- Join datasets ---
    print("\n2. Joining datasets ...")
    df_merged, join_report = join_datasets(portfolio_df, tech_df, sent_df, macro_df)

    # --- Coverage report ---
    print_coverage_report(df_merged, join_report)

    # --- Sort output ---
    sort_cols = []
    if 'year_month_str' in df_merged.columns:
        sort_cols.append('year_month_str')
    if 'Fund_Name' in df_merged.columns:
        sort_cols.append('Fund_Name')
    if 'ISIN' in df_merged.columns:
        sort_cols.append('ISIN')
    if sort_cols:
        df_merged = df_merged.sort_values(sort_cols).reset_index(drop=True)

    # --- Save output ---
    df_merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[step07] Saved: {OUTPUT_CSV}")
    print(f"         Shape: {df_merged.shape}")

    # Print column inventory
    print("\n[step07] Column inventory:")
    for i, col in enumerate(df_merged.columns):
        dtype = df_merged[col].dtype
        print(f"  [{i:3d}] {col:40s} ({dtype})")

    print("\n[step07] Done.")


if __name__ == '__main__':
    main()
