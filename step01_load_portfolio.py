"""
Step 01 -- Load, clean, classify, and feature-engineer the master portfolio.

Inputs
------
- MASTER_CSV  (primary)   : raw AMFI-style CSV with monthly fund holdings
- EXISTING_TEMPORAL_CSV   : pre-processed fallback (TEMPORAL_KG_READY.csv)

Outputs
-------
- PORTFOLIO_DIR / TEMPORAL_KG_READY.csv  : cleaned, classified, feature-rich holdings
- PORTFOLIO_DIR / EXIT_EVENTS.csv        : detected exit events per fund-stock pair
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MASTER_CSV, EXISTING_TEMPORAL_CSV, EXISTING_EXITS_CSV,
    PORTFOLIO_DIR, GAP_THRESHOLD,
)
from utils import standardize_industry, coverage_report

import pandas as pd
import numpy as np
import shutil
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_CSV = os.path.join(PORTFOLIO_DIR, 'TEMPORAL_KG_READY.csv')
EXITS_CSV = os.path.join(PORTFOLIO_DIR, 'EXIT_EVENTS.csv')


# ---------------------------------------------------------------------------
# 1.  PREPARE DATA
# ---------------------------------------------------------------------------

def prepare_data():
    """Load raw CSV, clean columns, standardize industries, normalise names.

    Tries MASTER_CSV first, falls back to copying EXISTING_TEMPORAL_CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned holdings DataFrame ready for classification.
    """
    # --- Resolve which file to load ---
    if os.path.isfile(MASTER_CSV):
        csv_path = MASTER_CSV
        source = "master"
    elif os.path.isfile(EXISTING_TEMPORAL_CSV):
        csv_path = EXISTING_TEMPORAL_CSV
        source = "temporal_fallback"
    else:
        raise FileNotFoundError(
            f"Neither MASTER_CSV ({MASTER_CSV}) nor "
            f"EXISTING_TEMPORAL_CSV ({EXISTING_TEMPORAL_CSV}) found."
        )

    print(f"[step01] Loading from: {csv_path}  (source={source})")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[step01] Raw rows: {len(df):,}")

    if source == "master":
        df = _clean_master(df)
    else:
        df = _validate_temporal(df)

    print(f"[step01] Cleaned rows: {len(df):,}")
    print(f"[step01] Funds: {df['Fund_Name'].nunique()}, "
          f"ISINs: {df['ISIN'].nunique()}, "
          f"Months: {df['year_month_str'].nunique()}")
    return df


def _clean_master(df):
    """Full cleaning pipeline for the raw MASTER_CONSOLIDATED CSV."""

    # --- Column rename ---
    rename_map = {
        'Name of the Instrument': 'stock_name_raw',
        '% to Net Assets': 'pct_nav',
        'Market Value': 'market_value',
    }
    df = df.rename(columns=rename_map)

    # --- Ensure required columns exist ---
    required = ['Fund_Name', 'ISIN', 'Date', 'pct_nav']
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found after rename. "
                           f"Available: {list(df.columns)}")

    # --- Parse dates ---
    df['date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['date'])

    # --- Filter valid ISINs (Indian equities start with INE) ---
    df['ISIN'] = df['ISIN'].astype(str).str.strip()
    df = df[df['ISIN'].str.startswith('INE')].copy()
    print(f"[step01]   After ISIN filter (INE*): {len(df):,}")

    # --- Numeric cleaning: remove zero/negative allocations ---
    df['pct_nav'] = pd.to_numeric(df['pct_nav'], errors='coerce')
    df = df[df['pct_nav'].notna() & (df['pct_nav'] > 0)].copy()
    print(f"[step01]   After pct_nav > 0 filter: {len(df):,}")

    if 'market_value' in df.columns:
        df['market_value'] = pd.to_numeric(df['market_value'], errors='coerce')

    if 'Quantity' in df.columns:
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')

    # --- Standardise industry -> sector ---
    industry_col = None
    for candidate in ['Industry', 'industry', 'Sector', 'sector']:
        if candidate in df.columns:
            industry_col = candidate
            break
    if industry_col is not None:
        df['sector'] = df[industry_col].apply(standardize_industry)
    else:
        df['sector'] = 'OTHERS'
        print("[step01]   WARNING: No Industry column found -- sector set to OTHERS")

    # --- Normalise stock names (most common name per ISIN) ---
    if 'stock_name_raw' in df.columns:
        name_map = (
            df.groupby('ISIN')['stock_name_raw']
            .agg(lambda x: x.value_counts().index[0])
            .to_dict()
        )
        df['stock_name'] = df['ISIN'].map(name_map)
    else:
        df['stock_name'] = df['ISIN']

    # --- Time helpers: normalize to year-month periods ---
    df['year_month'] = df['date'].dt.to_period('M')
    df['year_month_str'] = df['year_month'].astype(str)

    # --- Deduplicate: keep latest record per Fund / ISIN / month ---
    before = len(df)
    df = df.sort_values('date')
    df = df.drop_duplicates(subset=['Fund_Name', 'ISIN', 'year_month_str'],
                            keep='last')
    print(f"[step01]   After dedup: {len(df):,} (removed {before - len(df):,})")

    return df


def _validate_temporal(df):
    """Light validation for the pre-processed TEMPORAL_KG_READY fallback."""

    required = ['Fund_Name', 'ISIN', 'year_month_str']
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Temporal CSV missing required column: {col}")

    # Rebuild period column if missing
    if 'year_month' not in df.columns:
        df['year_month'] = pd.PeriodIndex(df['year_month_str'], freq='M')

    # Ensure date column
    if 'date' not in df.columns and 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'], errors='coerce')
    elif 'date' not in df.columns:
        df['date'] = df['year_month'].apply(lambda p: p.to_timestamp())

    # Ensure pct_nav is numeric
    if 'pct_nav' in df.columns:
        df['pct_nav'] = pd.to_numeric(df['pct_nav'], errors='coerce')

    # Ensure sector exists
    if 'sector' not in df.columns:
        if 'Industry' in df.columns:
            df['sector'] = df['Industry'].apply(standardize_industry)
        else:
            df['sector'] = 'OTHERS'

    # Ensure stock_name
    if 'stock_name' not in df.columns:
        if 'stock_name_raw' in df.columns:
            name_map = (
                df.groupby('ISIN')['stock_name_raw']
                .agg(lambda x: x.value_counts().index[0])
                .to_dict()
            )
            df['stock_name'] = df['ISIN'].map(name_map)
        else:
            df['stock_name'] = df['ISIN']

    return df


# ---------------------------------------------------------------------------
# 2.  CLASSIFY POSITIONS
# ---------------------------------------------------------------------------

def classify_positions(df, gap_threshold=3):
    """Classify each monthly holding row as INITIAL_POSITION / BUY / SELL / HOLD.

    Logic
    -----
    1. Sort by Fund_Name, ISIN, year_month.
    2. Compute month ordinal (year*12 + month) to find gaps.
    3. If first appearance or gap > threshold -> INITIAL_POSITION.
    4. BUY: quantity increased >0.5% or allocation change >0.1.
    5. SELL: quantity decreased >0.5% or allocation change <-0.1.
    6. HOLD: otherwise.
    """

    df = df.sort_values(['Fund_Name', 'ISIN', 'year_month_str']).copy()

    # Month ordinal for gap detection
    df['month_ordinal'] = df['year_month'].apply(
        lambda p: p.year * 12 + p.month
    )

    # Group-level lagged values
    grp = df.groupby(['Fund_Name', 'ISIN'])
    df['prev_month_ord'] = grp['month_ordinal'].shift(1)
    df['prev_pct_nav'] = grp['pct_nav'].shift(1)

    has_quantity = 'Quantity' in df.columns and df['Quantity'].notna().mean() > 0.3
    if has_quantity:
        df['prev_quantity'] = grp['Quantity'].shift(1)
    else:
        df['prev_quantity'] = np.nan

    # Gap in months
    df['month_gap'] = df['month_ordinal'] - df['prev_month_ord']

    # Allocation and quantity changes
    df['allocation_change'] = df['pct_nav'] - df['prev_pct_nav']

    if has_quantity:
        df['quantity_change'] = df['Quantity'] - df['prev_quantity']
    else:
        df['quantity_change'] = np.nan

    # --- Vectorised classification ---
    is_first = df['prev_month_ord'].isna()
    is_reentry = (~is_first) & (df['month_gap'] > gap_threshold)
    is_initial = is_first | is_reentry

    # Quantity-based signals
    if has_quantity:
        has_qty = (
            (~is_initial)
            & df['prev_quantity'].notna()
            & (df['prev_quantity'] > 0)
            & df['Quantity'].notna()
        )
        qty_pct = (df['quantity_change'].abs() / df['prev_quantity']).where(has_qty, other=0)

        is_buy_qty = has_qty & (df['quantity_change'] > 0) & (qty_pct > 0.005)
        is_sell_qty = has_qty & (df['quantity_change'] < 0) & (qty_pct > 0.005)
    else:
        is_buy_qty = pd.Series(False, index=df.index)
        is_sell_qty = pd.Series(False, index=df.index)

    # Fallback: allocation-based when quantity signal is absent
    no_qty_signal = (~is_initial) & (~is_buy_qty) & (~is_sell_qty)
    is_buy_alloc = no_qty_signal & (df['allocation_change'] > 0.1)
    is_sell_alloc = no_qty_signal & (df['allocation_change'] < -0.1)

    # Assign position actions
    df['position_action'] = 'HOLD'
    df.loc[is_initial, 'position_action'] = 'INITIAL_POSITION'
    df.loc[is_buy_qty | is_buy_alloc, 'position_action'] = 'BUY'
    df.loc[is_sell_qty | is_sell_alloc, 'position_action'] = 'SELL'

    # Size = magnitude of change
    df['size'] = np.where(is_initial, df['pct_nav'], df['allocation_change'].abs())
    df.loc[is_initial, 'allocation_change'] = df.loc[is_initial, 'pct_nav']

    # --- Holding-period ID (increments on INITIAL_POSITION) ---
    df['_is_initial'] = (df['position_action'] == 'INITIAL_POSITION').astype(int)
    df['holding_period_id'] = grp['_is_initial'].cumsum()

    # Make holding_period_id globally unique
    df['holding_period_id'] = (
        df['Fund_Name'].astype(str) + '|'
        + df['ISIN'].astype(str) + '|'
        + df['holding_period_id'].astype(str)
    )

    # --- Holding tenure (months since INITIAL_POSITION) ---
    df['holding_tenure'] = df.groupby(
        ['Fund_Name', 'ISIN', 'holding_period_id']
    ).cumcount() + 1

    # Cleanup temp columns
    df = df.drop(columns=['_is_initial', 'prev_month_ord', 'prev_pct_nav',
                           'prev_quantity'], errors='ignore')

    print(f"[step01] Position classification distribution:")
    for action, cnt in df['position_action'].value_counts().items():
        print(f"    {action:20s}: {cnt:>7,} ({100 * cnt / len(df):5.1f}%)")

    return df


# ---------------------------------------------------------------------------
# 3.  DETECT EXITS
# ---------------------------------------------------------------------------

def detect_exits(df, gap_threshold=3):
    """Detect exit events: stock disappears from a fund for > gap_threshold months
    while the fund continues reporting.

    Returns
    -------
    pd.DataFrame
        One row per exit event.
    """

    # Load existing exits as reference (if available)
    if os.path.isfile(EXISTING_EXITS_CSV):
        try:
            existing = pd.read_csv(EXISTING_EXITS_CSV, low_memory=False)
            print(f"[step01] Loaded existing exits reference: {len(existing):,} rows")
        except Exception:
            pass

    exit_records = []

    # Fund-level last reporting month
    fund_last_month = df.groupby('Fund_Name')['month_ordinal'].max().to_dict()

    for (fund, isin), grp in df.groupby(['Fund_Name', 'ISIN']):
        grp = grp.sort_values('month_ordinal')
        month_ords = grp['month_ordinal'].values
        fund_last = fund_last_month[fund]

        # Check for gaps within the series that exceed threshold
        for i in range(1, len(month_ords)):
            gap = month_ords[i] - month_ords[i - 1]
            if gap > gap_threshold:
                exit_row = grp.iloc[i - 1]
                exit_records.append({
                    'Fund_Name': fund,
                    'ISIN': isin,
                    'stock_name': exit_row.get('stock_name', isin),
                    'sector': exit_row.get('sector', 'OTHERS'),
                    'exit_month': exit_row['year_month_str'],
                    'exit_date': str(exit_row['date'].date()) if pd.notna(exit_row.get('date')) else exit_row['year_month_str'],
                    'last_pct_nav': exit_row.get('pct_nav', np.nan),
                    'last_quantity': exit_row.get('Quantity', np.nan),
                    'holding_tenure': exit_row.get('holding_tenure', np.nan),
                })

        # Check if stock's last appearance is before fund's last reporting month
        last_row = grp.iloc[-1]
        last_ord = int(last_row['month_ordinal'])
        if (fund_last - last_ord) > gap_threshold:
            exit_records.append({
                'Fund_Name': fund,
                'ISIN': isin,
                'stock_name': last_row.get('stock_name', isin),
                'sector': last_row.get('sector', 'OTHERS'),
                'exit_month': last_row['year_month_str'],
                'exit_date': str(last_row['date'].date()) if pd.notna(last_row.get('date')) else last_row['year_month_str'],
                'last_pct_nav': last_row.get('pct_nav', np.nan),
                'last_quantity': last_row.get('Quantity', np.nan),
                'holding_tenure': last_row.get('holding_tenure', np.nan),
            })

    exits_df = pd.DataFrame(exit_records)

    if len(exits_df) > 0:
        exits_df = exits_df.drop_duplicates(
            subset=['Fund_Name', 'ISIN', 'exit_month'], keep='last'
        )

    print(f"[step01] Exit events detected: {len(exits_df):,}")
    if len(exits_df) > 0:
        print(f"[step01]   Unique stocks exited: {exits_df['ISIN'].nunique()}")
        print(f"[step01]   Unique funds with exits: {exits_df['Fund_Name'].nunique()}")
    return exits_df


# ---------------------------------------------------------------------------
# 4.  FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_features(df):
    """Add portfolio-analytics features to the holdings DataFrame.

    New columns: rank_in_fund, fund_stock_count, sector_weight,
    consensus_count, is_top10, allocation_quintile
    """

    # rank_in_fund
    df['rank_in_fund'] = df.groupby(['Fund_Name', 'year_month_str'])['pct_nav'].rank(
        method='min', ascending=False
    ).astype(int)

    # fund_stock_count (portfolio breadth)
    df['fund_stock_count'] = df.groupby(
        ['Fund_Name', 'year_month_str']
    )['ISIN'].transform('count').astype(int)

    # sector_weight within fund
    df['sector_weight'] = df.groupby(
        ['Fund_Name', 'year_month_str', 'sector']
    )['pct_nav'].transform('sum')

    # consensus_count (how many funds hold this stock in this month)
    df['consensus_count'] = df.groupby(
        ['ISIN', 'year_month_str']
    )['Fund_Name'].transform('nunique').astype(int)

    # is_top10
    df['is_top10'] = (df['rank_in_fund'] <= 10).astype(int)

    # allocation_quintile
    def _quintile(s):
        try:
            return pd.qcut(s, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        except ValueError:
            return pd.Series(3, index=s.index)

    df['allocation_quintile'] = (
        df.groupby(['Fund_Name', 'year_month_str'])['pct_nav']
        .transform(_quintile)
    )
    df['allocation_quintile'] = pd.to_numeric(
        df['allocation_quintile'], errors='coerce'
    ).fillna(3).astype(int)

    print(f"[step01] Features added: rank_in_fund, fund_stock_count, "
          f"sector_weight, consensus_count, is_top10, allocation_quintile")
    print(f"[step01]   Avg consensus: {df['consensus_count'].mean():.1f} funds/stock")
    print(f"[step01]   Avg fund breadth: {df['fund_stock_count'].mean():.0f} stocks/fund")

    return df


# ---------------------------------------------------------------------------
# 5.  MAIN ORCHESTRATOR
# ---------------------------------------------------------------------------

def main():
    """Run the full Step 01 pipeline: load -> classify -> exits -> features -> save."""

    print("=" * 70)
    print("STEP 01: Load & Validate Portfolio Data")
    print("=" * 70)

    # If MASTER_CSV does not exist, try the fallback copy approach
    if not os.path.isfile(MASTER_CSV) and os.path.isfile(EXISTING_TEMPORAL_CSV):
        print(f"[step01] MASTER_CSV not found. Copying existing temporal data as fallback.")
        shutil.copy2(EXISTING_TEMPORAL_CSV, OUTPUT_CSV)
        print(f"[step01] Copied {EXISTING_TEMPORAL_CSV} -> {OUTPUT_CSV}")

        # Also copy exit events if available
        if os.path.isfile(EXISTING_EXITS_CSV):
            shutil.copy2(EXISTING_EXITS_CSV, EXITS_CSV)
            print(f"[step01] Copied {EXISTING_EXITS_CSV} -> {EXITS_CSV}")

        # Still validate the copied data
        df = pd.read_csv(OUTPUT_CSV, low_memory=False)
        print(f"[step01] Fallback data: {len(df):,} rows, "
              f"{df['Fund_Name'].nunique() if 'Fund_Name' in df.columns else '?'} funds")

        # Ensure essential columns exist for downstream steps
        if 'year_month_str' not in df.columns and 'year_month' in df.columns:
            df['year_month_str'] = df['year_month'].astype(str)
            df.to_csv(OUTPUT_CSV, index=False)

        print("[step01] Done (fallback mode).")
        return

    # --- Normal flow ---

    # 1. Load and clean
    df = prepare_data()

    # 2. Classify positions
    df = classify_positions(df, gap_threshold=GAP_THRESHOLD)

    # 3. Detect exit events
    exits_df = detect_exits(df, gap_threshold=GAP_THRESHOLD)

    # 4. Engineer features
    df = engineer_features(df)

    # --- Save outputs ---
    # Convert Period columns to string before saving
    save_df = df.copy()
    for col in save_df.columns:
        if hasattr(save_df[col].dtype, 'freq'):  # PeriodDtype
            save_df[col] = save_df[col].astype(str)

    save_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[step01] Saved TEMPORAL_KG_READY.csv -> {OUTPUT_CSV}")
    print(f"         Rows: {len(save_df):,}  Columns: {len(save_df.columns)}")

    if len(exits_df) > 0:
        exits_df.to_csv(EXITS_CSV, index=False)
        print(f"[step01] Saved EXIT_EVENTS.csv -> {EXITS_CSV}")
        print(f"         Exit rows: {len(exits_df):,}")
    else:
        print("[step01] No exit events detected -- skipping EXIT_EVENTS.csv")

    # --- Summary ---
    print("\n" + "-" * 50)
    print("STEP 01 SUMMARY")
    print("-" * 50)
    print(f"  Total holdings      : {len(save_df):,}")
    print(f"  Unique funds        : {save_df['Fund_Name'].nunique()}")
    print(f"  Unique ISINs        : {save_df['ISIN'].nunique()}")
    print(f"  Unique sectors      : {save_df['sector'].nunique()}")
    print(f"  Date range          : {save_df['year_month_str'].min()} to {save_df['year_month_str'].max()}")
    print(f"  Exit events         : {len(exits_df):,}")
    print(f"\n  Position action distribution:")
    for action, cnt in save_df['position_action'].value_counts().items():
        print(f"    {action:20s}: {cnt:>7,} ({100 * cnt / len(save_df):5.1f}%)")

    coverage_report(save_df)
    print("\n[step01] Done.")


if __name__ == '__main__':
    main()
