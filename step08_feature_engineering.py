"""
Step 08 -- Feature Engineering for LPCMCI Causal Discovery
==========================================================

Input:  FEATURES_DIR / CAUSAL_DISCOVERY_DATASET.csv  (step07 output)

Output:
- FEATURES_DIR / LPCMCI_READY.csv                 (full engineered dataset)
- FEATURES_DIR / macro_panel.csv                   (one row per month)
- FEATURES_DIR / sector_panel.csv                  (one row per sector/month)
- FEATURES_DIR / fund_panel.csv                    (one row per fund/month)
- FEATURES_DIR / stock_panel.csv                   (top consensus stocks per month)

Feature engineering:
- Lagged features (t-1, t-2) for: pct_nav, pe, pb, rsi, macd, sentiment_mean, monthly_return
- Analysis-level aggregation: macro, sector, fund, stock
- Standardisation (z-score per variable)
- Missing value handling: linear interpolation + forward/backward fill
- Zero-variance column removal
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES_DIR
from utils import coverage_report

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

INPUT_CSV = os.path.join(FEATURES_DIR, 'CAUSAL_DISCOVERY_DATASET.csv')
OUTPUT_CSV = os.path.join(FEATURES_DIR, 'LPCMCI_READY.csv')

# Key variables to create lags for (t-1, t-2)
LAG_VARIABLES = [
    'pct_nav', 'pe', 'pb', 'rsi', 'macd_hist', 'macd_line',
    'sentiment_mean', 'monthly_return', 'monthly_volatility',
    'eps', 'beta', 'market_cap', 'obv', 'atr', 'volume_ratio',
    'bollinger_pband', 'news_count',
]

# ID / categorical columns to exclude from standardisation
ID_COLS = {
    'year_month_str', 'symbol', 'ISIN', 'Fund_Name', 'Fund_Type',
    'sector', 'stock_name', 'stock_name_raw', 'Industry',
    'date', 'Date', 'holding_period_id', 'position_action', 'action',
    'golden_cross', 'death_cross', 'size', 'file_name', 'fund_ticker',
}


# ============================================================
# 1. Lagged feature creation
# ============================================================

def create_lagged_features(df, group_cols, feature_cols, lags=(1, 2)):
    """Create lagged features (t-1, t-2) for specified columns within groups.

    Parameters
    ----------
    df : DataFrame
    group_cols : list of str
    feature_cols : list of str
    lags : tuple of int

    Returns
    -------
    DataFrame with new lag columns appended.
    """
    df = df.sort_values(group_cols + ['year_month_str'])
    existing = [c for c in feature_cols if c in df.columns]

    if not existing:
        return df

    new_cols_count = 0
    for lag in lags:
        for col in existing:
            lag_name = f"{col}_lag{lag}"
            if lag_name not in df.columns:
                df[lag_name] = df.groupby(group_cols)[col].shift(lag)
                new_cols_count += 1

    if new_cols_count > 0:
        print(f"    Created {new_cols_count} lag columns for group {group_cols}")

    return df


# ============================================================
# 2. Derived features
# ============================================================

def create_derived_features(df):
    """Create derived features from existing columns.

    v3 improvements:
    - Removed sentiment_trend (redundant with sentiment_mean lags, ~95% corr)
    - Added interaction features (PE*momentum, VIX*sector_return, etc.)
    - Added nonlinear transforms (log market_cap, sqrt volatility)
    - Added extended lags (lag3-lag6) for key causal features
    """

    # pe_relative: PE ratio relative to sector median
    pe_col = None
    for c in ['pe', 'PE', 'pe_ratio']:
        if c in df.columns:
            pe_col = c
            break
    if pe_col and 'sector' in df.columns:
        sector_median_pe = df.groupby(['sector', 'year_month_str'])[pe_col].transform('median')
        df['pe_relative'] = df[pe_col] / sector_median_pe.replace(0, np.nan)
        print("    Created: pe_relative")

    # momentum_3m: rolling 3-month cumulative return
    if 'monthly_return' in df.columns and 'symbol' in df.columns:
        df = df.sort_values(['symbol', 'year_month_str'])
        df['momentum_3m'] = df.groupby('symbol')['monthly_return'].transform(
            lambda x: (1 + x).rolling(window=3, min_periods=2).apply(np.prod, raw=True) - 1
        )
        print("    Created: momentum_3m")

    # volatility_3m: rolling 3-month std of monthly returns
    if 'monthly_return' in df.columns and 'symbol' in df.columns:
        df['volatility_3m'] = df.groupby('symbol')['monthly_return'].transform(
            lambda x: x.rolling(window=3, min_periods=2).std()
        )
        print("    Created: volatility_3m")

    # allocation_momentum: month-over-month change in pct_nav
    pct_col = 'pct_nav' if 'pct_nav' in df.columns else None
    if pct_col:
        group_cols = ['Fund_Name', 'ISIN'] if 'Fund_Name' in df.columns and 'ISIN' in df.columns else (
            ['symbol'] if 'symbol' in df.columns else []
        )
        if group_cols:
            df = df.sort_values(group_cols + ['year_month_str'])
            df['allocation_momentum'] = df.groupby(group_cols)[pct_col].diff()
            print("    Created: allocation_momentum")

    # v3: REMOVED sentiment_trend — ~95% correlated with sentiment_mean lags
    # (was: 3-month rolling mean of sentiment_mean)

    # --- v3: Interaction features ---
    interactions_created = []

    # PE × momentum: value-momentum interaction
    if pe_col and 'momentum_3m' in df.columns:
        df['pe_x_momentum'] = df[pe_col] * df['momentum_3m']
        interactions_created.append('pe_x_momentum')

    # RSI × momentum: technical confluence
    if 'rsi' in df.columns and 'momentum_3m' in df.columns:
        df['rsi_x_momentum'] = df['rsi'] * df['momentum_3m']
        interactions_created.append('rsi_x_momentum')

    # VIX × monthly_return: fear-return interaction
    if 'india_vix_close' in df.columns and 'monthly_return' in df.columns:
        df['vix_x_return'] = df['india_vix_close'] * df['monthly_return']
        interactions_created.append('vix_x_return')

    # Sentiment × momentum: sentiment-price agreement
    if 'sentiment_mean' in df.columns and 'momentum_3m' in df.columns:
        df['sentiment_x_momentum'] = df['sentiment_mean'] * df['momentum_3m']
        interactions_created.append('sentiment_x_momentum')

    # Allocation × sentiment: conviction-sentiment interaction
    if pct_col and 'sentiment_mean' in df.columns:
        df['alloc_x_sentiment'] = df[pct_col] * df['sentiment_mean']
        interactions_created.append('alloc_x_sentiment')

    if interactions_created:
        print(f"    Created interaction features: {interactions_created}")

    # --- v3: Nonlinear transforms ---
    nonlinear_created = []

    if 'market_cap' in df.columns:
        df['log_market_cap'] = np.log1p(df['market_cap'].clip(lower=0))
        nonlinear_created.append('log_market_cap')

    if pe_col:
        df['log_pe'] = np.log1p(df[pe_col].clip(lower=0))
        nonlinear_created.append('log_pe')

    if 'monthly_volatility' in df.columns:
        df['sqrt_volatility'] = np.sqrt(df['monthly_volatility'].clip(lower=0))
        nonlinear_created.append('sqrt_volatility')

    if 'volume_ratio' in df.columns:
        df['log_volume_ratio'] = np.log1p(df['volume_ratio'].clip(lower=0))
        nonlinear_created.append('log_volume_ratio')

    if nonlinear_created:
        print(f"    Created nonlinear transforms: {nonlinear_created}")

    return df


# ============================================================
# 3. Analysis-level panel aggregations
# ============================================================

def build_macro_panel(df):
    """Macro level: single row per month with macro time series."""
    macro_candidates = [
        'repo_rate', 'cpi_inflation', 'gdp_growth', 'real_interest_rate',
        'repo_rate_change', 'gold_usd', 'gold_inr', 'crude_oil', 'brent_crude',
        'usd_inr', 'us_10y_yield', 'sp500', 'india_vix',
    ]
    # Include any column with index-like names
    for col in df.columns:
        if any(kw in col.lower() for kw in ['nifty', 'index', 'vix', '_return', '_pct']):
            if col not in macro_candidates:
                macro_candidates.append(col)

    available = [c for c in macro_candidates if c in df.columns]
    if not available or 'year_month_str' not in df.columns:
        print("    WARNING: Cannot build macro panel")
        return pd.DataFrame()

    panel = df.groupby('year_month_str')[available].first().reset_index()
    panel = panel.sort_values('year_month_str').reset_index(drop=True)
    print(f"    Macro panel: {panel.shape} ({len(available)} features)")
    return panel


def build_sector_panel(df):
    """Sector level: avg fundamentals/sentiment/allocation per sector per month."""
    if 'sector' not in df.columns or 'year_month_str' not in df.columns:
        print("    WARNING: Cannot build sector panel")
        return pd.DataFrame()

    agg_cols = [c for c in [
        'pct_nav', 'monthly_return', 'monthly_volatility', 'rsi', 'volume_ratio',
        'sentiment_mean', 'pe', 'pb', 'eps', 'market_cap', 'pe_relative',
        'momentum_3m', 'allocation_momentum',
    ] if c in df.columns]

    if not agg_cols:
        print("    WARNING: No aggregatable columns for sector panel")
        return pd.DataFrame()

    agg_dict = {col: 'mean' for col in agg_cols}
    if 'ISIN' in df.columns:
        agg_dict['ISIN'] = 'nunique'

    panel = df.groupby(['sector', 'year_month_str']).agg(agg_dict).reset_index()
    if 'ISIN' in panel.columns:
        panel = panel.rename(columns={'ISIN': 'stock_count'})

    panel = panel.sort_values(['sector', 'year_month_str']).reset_index(drop=True)
    print(f"    Sector panel: {panel.shape}")
    return panel


def build_fund_panel(df):
    """Fund level: fund-specific patterns per month."""
    if 'Fund_Name' not in df.columns or 'year_month_str' not in df.columns:
        print("    WARNING: Cannot build fund panel")
        return pd.DataFrame()

    agg_cols = [c for c in [
        'pct_nav', 'monthly_return', 'monthly_volatility', 'rsi',
        'sentiment_mean', 'pe', 'pb', 'pe_relative', 'allocation_momentum',
    ] if c in df.columns]

    if not agg_cols:
        print("    WARNING: No aggregatable columns for fund panel")
        return pd.DataFrame()

    agg_dict = {col: 'mean' for col in agg_cols}
    if 'ISIN' in df.columns:
        agg_dict['ISIN'] = 'nunique'

    panel = df.groupby(['Fund_Name', 'year_month_str']).agg(agg_dict).reset_index()
    if 'ISIN' in panel.columns:
        panel = panel.rename(columns={'ISIN': 'holdings_count'})

    panel = panel.sort_values(['Fund_Name', 'year_month_str']).reset_index(drop=True)
    print(f"    Fund panel: {panel.shape}")
    return panel


def build_stock_panel(df, top_n=50):
    """Stock level: per-stock features for top consensus stocks."""
    id_col = 'ISIN' if 'ISIN' in df.columns else ('symbol' if 'symbol' in df.columns else None)
    if id_col is None or 'year_month_str' not in df.columns:
        print("    WARNING: Cannot build stock panel")
        return pd.DataFrame()

    # Select top N by frequency (most held across funds)
    freq = df[id_col].value_counts()
    top_ids = freq.head(top_n).index.tolist()
    print(f"    Selecting top {top_n} consensus stocks by frequency")

    df_top = df[df[id_col].isin(top_ids)].copy()

    numeric_cols = df_top.select_dtypes(include=[np.number]).columns.tolist()
    agg_cols = [c for c in numeric_cols if c not in ID_COLS]

    if not agg_cols:
        print("    WARNING: No numeric columns for stock panel")
        return pd.DataFrame()

    panel = df_top.groupby([id_col, 'year_month_str'])[agg_cols].mean().reset_index()
    panel = panel.sort_values([id_col, 'year_month_str']).reset_index(drop=True)
    print(f"    Stock panel: {panel.shape} ({len(top_ids)} stocks)")
    return panel


# ============================================================
# 4. Standardisation and cleanup
# ============================================================

def standardize_features(df, train_months=None):
    """Z-score standardise all numeric features (exclude ID columns).

    CRITICAL: To prevent data leakage, compute mean/std ONLY from
    training months (pre-split).  If train_months is provided, stats
    are computed on that subset and then applied to the whole dataframe.
    The stats dict is returned so downstream code can apply the same
    transform to unseen data.

    Parameters
    ----------
    df : DataFrame
    train_months : list[str] or None
        List of year_month_str values to use as training set for
        computing statistics.  If None, uses temporal split: first 65%
        of unique months (matching step13b temporal split protocol).

    Returns
    -------
    (DataFrame, list[str], dict)  -- standardised df, column list, stats dict
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_std = [c for c in numeric_cols if c not in ID_COLS]

    # Determine training months for computing stats (prevent leakage)
    if train_months is None and 'year_month_str' in df.columns:
        all_months = sorted(df['year_month_str'].dropna().unique())
        split_idx = int(len(all_months) * 0.65)
        train_months = all_months[:split_idx]
        print(f"    [LEAKAGE GUARD] Using first {len(train_months)}/{len(all_months)} "
              f"months for standardisation stats ({train_months[0]}..{train_months[-1]})")

    if train_months is not None and 'year_month_str' in df.columns:
        train_mask = df['year_month_str'].isin(train_months)
        train_df = df.loc[train_mask]
    else:
        train_df = df
        print("    WARNING: No temporal split available; using full dataset for stats")

    stats = {}
    standardized_count = 0
    for col in cols_to_std:
        mean_val = train_df[col].mean()
        std_val = train_df[col].std()
        stats[col] = {'mean': mean_val, 'std': std_val}
        if std_val > 0:
            df[col] = (df[col] - mean_val) / std_val
            standardized_count += 1

    print(f"    Standardised {standardized_count} numeric columns (z-score, train-only stats)")
    return df, cols_to_std, stats


def interpolate_and_fill(df, group_col=None):
    """Handle missing values: linear interpolation + capped forward/backward fill.

    v3: Added limit=3 to prevent filling arbitrarily long gaps
    (e.g., 12+ months of NaN would create artificial persistence).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    before_na = df[numeric_cols].isna().sum().sum()

    if group_col and group_col in df.columns:
        df[numeric_cols] = df.groupby(group_col)[numeric_cols].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both',
                                     limit=3, limit_area='inside')
        )
    else:
        df[numeric_cols] = df[numeric_cols].interpolate(
            method='linear', limit_direction='both',
            limit=3, limit_area='inside'
        )

    # Forward/backward fill remaining NaN (capped at 3 periods)
    df[numeric_cols] = df[numeric_cols].ffill(limit=3).bfill(limit=3)

    after_na = df[numeric_cols].isna().sum().sum()
    print(f"    NaN reduction: {before_na:,} -> {after_na:,}")
    if after_na > 0:
        print(f"    WARNING: {after_na:,} NaN values remain (gaps > 3 months)")

    return df


def remove_zero_variance(df):
    """Remove columns with zero variance (constant values)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_check = [c for c in numeric_cols if c not in ID_COLS]

    zero_var_cols = []
    for col in cols_to_check:
        if df[col].dropna().std() == 0 or df[col].dropna().nunique() <= 1:
            zero_var_cols.append(col)

    if zero_var_cols:
        print(f"    Removing {len(zero_var_cols)} zero-variance columns: "
              f"{zero_var_cols[:10]}{'...' if len(zero_var_cols) > 10 else ''}")
        df = df.drop(columns=zero_var_cols)
    else:
        print("    No zero-variance columns found")

    return df


def prune_highly_correlated(df, threshold=0.95):
    """Remove one of each pair of features with |correlation| > threshold.

    Keeps the feature that has higher average absolute correlation with
    the remaining features (i.e., drops the more redundant one).
    Protects ID columns and key target variables from removal.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_check = [c for c in numeric_cols if c not in ID_COLS]

    # Protect target columns from removal
    protected = {'action_ordinal', 'is_buy', 'is_sell', 'pct_nav', 'buy_ratio',
                 'sell_ratio', 'holding_tenure', 'consensus_count'}

    if len(cols_to_check) < 2:
        return df

    corr_matrix = df[cols_to_check].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > threshold].tolist()
        for other in high_corr:
            if col in to_drop or other in to_drop:
                continue
            # Drop the one NOT in protected; if neither protected, drop the one
            # with higher mean correlation to everything else
            if col in protected:
                to_drop.add(other)
            elif other in protected:
                to_drop.add(col)
            else:
                mean_corr_col = corr_matrix[col].drop([col, other], errors='ignore').mean()
                mean_corr_other = corr_matrix[other].drop([col, other], errors='ignore').mean()
                to_drop.add(col if mean_corr_col > mean_corr_other else other)

    if to_drop:
        print(f"    Pruning {len(to_drop)} highly correlated features (|r|>{threshold}):")
        for c in sorted(to_drop)[:15]:
            print(f"      - {c}")
        if len(to_drop) > 15:
            print(f"      ... and {len(to_drop) - 15} more")
        df = df.drop(columns=list(to_drop))
    else:
        print(f"    No highly correlated feature pairs found (threshold={threshold})")

    return df


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("STEP 08: Feature Engineering for LPCMCI")
    print("=" * 70)

    # --- Load causal discovery dataset ---
    if not os.path.exists(INPUT_CSV):
        print(f"[step08] ERROR: Input file not found: {INPUT_CSV}")
        print("         Run step07 first.")
        return

    print(f"\n[step08] Loading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df = df[df['year_month_str'] >= '2022-09']
    print(f"  Shape: {df.shape}")
    # --- Winsorize extreme outliers before feature engineering ---
    print("\n[step08] Winsorizing extreme outliers ...")
    winsorize_cols = {
        'pe': (0, 99),           # cap PE at 99th percentile
        'market_cap': (1, 99),   # cap market_cap at 1st-99th
        'pe_relative': (0, 99),  # will be created later but add safety
        'eps': (1, 99),
        'atr': (1, 99),
        'obv': (1, 99),
    }
    for col, (lower_pct, upper_pct) in winsorize_cols.items():
        if col in df.columns:
            lo = np.nanpercentile(df[col].dropna(), lower_pct)
            hi = np.nanpercentile(df[col].dropna(), upper_pct)
            before_max = df[col].max()
            df[col] = df[col].clip(lower=lo, upper=hi)
            print(f"  {col:20s}: clipped [{lo:.1f}, {hi:.1f}]  (was max={before_max:.1f})")
        print(f"  Columns: {len(df.columns)}")

    # --- 1. Create lagged features ---
    print("\n1. Creating lagged features (t-1, t-2) ...")

    # Determine appropriate grouping for lags
    has_fund = 'Fund_Name' in df.columns
    has_isin = 'ISIN' in df.columns
    has_symbol = 'symbol' in df.columns

    # Stock-level lags (group by ISIN or symbol)
    stock_group = ['ISIN'] if has_isin else (['symbol'] if has_symbol else [])
    if stock_group:
        stock_lag_vars = [c for c in LAG_VARIABLES if c in df.columns]
        if stock_lag_vars:
            df = create_lagged_features(df, stock_group, stock_lag_vars, lags=(1, 2))

        # v3: Extended lags (3-6) for key causal features
        EXTENDED_LAG_VARS = [
            'sentiment_mean', 'monthly_return', 'rsi', 'pct_nav',
            'pe', 'momentum_3m', 'monthly_volatility',
        ]
        ext_vars = [c for c in EXTENDED_LAG_VARS if c in df.columns]
        if ext_vars:
            df = create_lagged_features(df, stock_group, ext_vars, lags=(3, 4, 5, 6))
            print(f"    Extended lags (3-6) for: {ext_vars}")

    # Fund-stock level lags (group by Fund_Name + ISIN)
    if has_fund and has_isin:
        fund_lag_vars = [c for c in ['pct_nav', 'allocation_change'] if c in df.columns]
        if fund_lag_vars:
            df = create_lagged_features(df, ['Fund_Name', 'ISIN'], fund_lag_vars, lags=(1, 2))

    # Macro-level lags (unique per month, no group)
    macro_lag_vars = [c for c in ['repo_rate', 'cpi_inflation', 'gdp_growth'] if c in df.columns]
    if macro_lag_vars and 'year_month_str' in df.columns:
        macro_sub = df[['year_month_str'] + macro_lag_vars].drop_duplicates('year_month_str')
        macro_sub = macro_sub.sort_values('year_month_str')
        for lag in [1, 2]:
            for col in macro_lag_vars:
                lag_name = f"{col}_lag{lag}"
                if lag_name not in df.columns:
                    macro_sub[lag_name] = macro_sub[col].shift(lag)
        new_lag_cols = [c for c in macro_sub.columns
                        if c.endswith('_lag1') or c.endswith('_lag2')]
        new_lag_cols = [c for c in new_lag_cols if c not in df.columns]
        if new_lag_cols:
            df = df.merge(
                macro_sub[['year_month_str'] + new_lag_cols],
                on='year_month_str', how='left',
            )
            print(f"    Macro lags: {new_lag_cols}")

    all_lag_cols = [c for c in df.columns if '_lag' in c]
    print(f"\n  Total lag features: {len(all_lag_cols)}")

    # --- 2. Create derived features ---
    print("\n2. Creating derived features ...")
    df = create_derived_features(df)

    derived_cols = [c for c in ['pe_relative', 'momentum_3m', 'volatility_3m',
                                'allocation_momentum',
                                'pe_x_momentum', 'rsi_x_momentum', 'vix_x_return',
                                'sentiment_x_momentum', 'alloc_x_sentiment',
                                'log_market_cap', 'log_pe', 'sqrt_volatility',
                                'log_volume_ratio']
                    if c in df.columns]
    print(f"  Derived features ({len(derived_cols)}): {derived_cols}")

    # --- 3. Build aggregated panels ---
    print("\n3. Building analysis-level panels ...")

    print("\n  [Macro Panel]")
    macro_panel = build_macro_panel(df)

    print("\n  [Sector Panel]")
    sector_panel = build_sector_panel(df)

    print("\n  [Fund Panel]")
    fund_panel = build_fund_panel(df)

    print("\n  [Stock Panel - Top Consensus]")
    stock_panel = build_stock_panel(df, top_n=50)

    # --- 4. Clean and standardise ---
    print("\n4. Handling missing values ...")

    # Interpolate NaN in full dataset
    group_for_interp = 'ISIN' if has_isin else ('symbol' if has_symbol else None)
    df = interpolate_and_fill(df, group_for_interp)

    # Clean panels too
    panels = {
        'macro_panel': (macro_panel, None),
        'sector_panel': (sector_panel, 'sector'),
        'fund_panel': (fund_panel, 'Fund_Name'),
        'stock_panel': (stock_panel, stock_group[0] if stock_group else None),
    }
    for name, (panel, grp) in panels.items():
        if not panel.empty:
            panel = interpolate_and_fill(panel, grp)
            panel = remove_zero_variance(panel)
            panels[name] = (panel, grp)

    # Remove zero-variance columns from main dataset
    print("\n5. Removing zero-variance columns ...")
    df = remove_zero_variance(df)

    # v3: Prune highly correlated features
    print("\n5b. Pruning highly correlated features ...")
    df = prune_highly_correlated(df, threshold=0.95)

    # Standardise
    print("\n6. Standardising features (z-score) ...")
    df_std, std_cols, std_stats = standardize_features(df.copy())

    # Save standardisation stats for downstream use (no leakage on re-apply)
    import json
    stats_path = os.path.join(FEATURES_DIR, 'standardisation_stats.json')
    json_stats = {k: {kk: float(vv) if not np.isnan(vv) else None
                       for kk, vv in v.items()} for k, v in std_stats.items()}
    with open(stats_path, 'w') as f:
        json.dump(json_stats, f, indent=2)
    print(f"  Saved standardisation stats: {stats_path}")

    # --- 5. Save outputs ---
    print("\n7. Saving outputs ...")

    # Full LPCMCI-ready dataset
    df_std.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved: {OUTPUT_CSV} ({df_std.shape})")

    # Panel CSVs
    panel_files = {
        'macro_panel.csv': panels['macro_panel'][0],
        'sector_panel.csv': panels['sector_panel'][0],
        'fund_panel.csv': panels['fund_panel'][0],
        'stock_panel.csv': panels['stock_panel'][0],
    }

    for filename, panel_df in panel_files.items():
        if panel_df.empty:
            print(f"  Skipped: {filename} (empty)")
            continue
        path = os.path.join(FEATURES_DIR, filename)
        panel_df.to_csv(path, index=False)
        print(f"  Saved: {path} ({panel_df.shape})")

    # --- 6. Final summary ---
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)

    print(f"\n  LPCMCI_READY dataset:")
    print(f"    Rows:    {df_std.shape[0]:,}")
    print(f"    Columns: {df_std.shape[1]}")

    # Count feature types
    lag_count = len([c for c in df_std.columns if '_lag' in c])
    derived_count = len([c for c in derived_cols if c in df_std.columns])
    original_count = df_std.shape[1] - lag_count - derived_count
    print(f"    Original features:  {original_count}")
    print(f"    Lag features:       {lag_count}")
    print(f"    Derived features:   {derived_count}")

    print(f"\n  Panel datasets:")
    for name, (panel, _) in panels.items():
        if not panel.empty:
            print(f"    {name:20s}: {panel.shape}")

    # Min observations per analysis level
    print(f"\n  Min observations per analysis level:")
    for name, (panel, grp) in panels.items():
        if panel.empty:
            continue
        if grp and grp in panel.columns:
            min_obs = panel.groupby(grp).size().min()
            print(f"    {name:20s}: {min_obs} months (min per group)")
        else:
            print(f"    {name:20s}: {len(panel)} months")

    # Coverage report
    coverage_report(df_std)

    print("\n[step08] Done.")


if __name__ == '__main__':
    main()
