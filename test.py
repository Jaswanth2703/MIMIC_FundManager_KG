import pandas as pd
import numpy as np
import os

# Paths to your specific project files
RAW_FILE = 'data/features/CAUSAL_DISCOVERY_DATASET.csv'
READY_FILE = 'data/features/LPCMCI_READY.csv'

def run_authenticity_check():
    if not os.path.exists(RAW_FILE) or not os.path.exists(READY_FILE):
        print("Error: Ensure both raw and ready CSVs exist in data/features/")
        return

    print("Loading datasets for comparison...")
    # Load raw data (unfilled) and ready data (filled/standardized)
    df_raw = pd.read_csv(RAW_FILE, low_memory=False)
    df_ready = pd.read_csv(READY_FILE, low_memory=False)

    # 1. Timeline Comparison
    raw_months = df_raw['year_month_str'].nunique()
    ready_months = df_ready['year_month_str'].nunique()
    print(f"\n[1] Timeline Integrity:")
    print(f"    - Original months in raw file: {raw_months}")
    print(f"    - Cleaned months (truncated):   {ready_months}")

    # 2. Identify "True" First Disclosure per Fund
    # We look for the first month where 'pct_nav' was not NaN in the raw data
    true_starts = df_raw.dropna(subset=['pct_nav']).groupby('Fund_Name')['year_month_str'].min()
    
    print(f"\n[2] Discovery of 'True' Disclosure Starts:")
    for fund, start in true_starts.items():
        # Check if the ready file has data BEFORE the true start
        ready_min = df_ready[df_ready['Fund_Name'] == fund]['year_month_str'].min()
        if pd.notna(ready_min) and ready_min < start:
            print(f"    - {fund[:25]:<25}: Real Start {start} | (Backfilled to {ready_min})")
        else:
            print(f"    - {fund[:25]:<25}: Real Start {start} | (Clean)")

    # 3. Calculate Imputation Ratio (Synthetic Density)
    # We compare a specific key column like 'rsi' or 'pe'
    target_col = 'rsi' if 'rsi' in df_raw.columns else df_raw.select_dtypes(include=[np.number]).columns[0]
    
    # Merge raw and ready on (Fund, ISIN, Month)
    check_df = df_ready[['Fund_Name', 'ISIN', 'year_month_str']].copy()
    check_df = check_df.merge(
        df_raw[['Fund_Name', 'ISIN', 'year_month_str', target_col]], 
        on=['Fund_Name', 'ISIN', 'year_month_str'], 
        how='left', suffixes=('', '_raw')
    )

    is_synthetic = check_df[target_col].isna() # It was NaN in raw but exists in ready
    synthetic_pct = (is_synthetic.sum() / len(check_df)) * 100

    print(f"\n[3] Synthetic Data Density (Column: {target_col}):")
    print(f"    - Total Rows: {len(check_df):,}")
    print(f"    - Real Data:  {len(check_df) - is_synthetic.sum():,}")
    print(f"    - Backfilled: {is_synthetic.sum():,}")
    print(f"    - Synthetic Ratio: {synthetic_pct:.2f}%")

if __name__ == "__main__":
    run_authenticity_check()