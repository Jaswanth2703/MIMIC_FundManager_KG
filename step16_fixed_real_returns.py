
"""
Step 16 FIXED -- Walk-Forward Backtest with Real Returns
=========================================================
Uses unstandardized monthly_return from CAUSAL_DISCOVERY_DATASET.csv
to calculate real financial performance.
"""

import sys, os
import json, traceback, warnings
import numpy as np, pandas as pd

# Standard paths relative to script location in causal_output
FEATURES_DIR = os.path.join('..', 'features')
FINAL_DIR = os.path.join('..', 'final')

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

ACTION_MAP = {'BUY':2,'INCREASE':2,'INITIAL_POSITION':2,'HOLD':1,'DECREASE':0,'SELL':0}
TRANSACTION_COST = 0.005  # 0.50% one-way

def run_fold(df, train_months, test_month, feature_cols, embargo=1):
    avail = [f for f in feature_cols if f in df.columns]
    if len(avail) < 3: return None

    # Apply embargo
    effective_train = train_months[:-embargo] if len(train_months) > embargo else train_months
    
    train = df[df['year_month_str'].isin(effective_train)].copy()
    test = df[df['year_month_str'] == test_month].copy()

    if len(train) < 50 or len(test) < 10: return None

    # Train XGBoost
    train_enc = train.dropna(subset=['position_action']).copy()
    train_enc['target'] = train_enc['position_action'].map(ACTION_MAP)
    train_enc = train_enc.dropna(subset=['target'])
    
    X_train = train_enc[avail].fillna(0)
    y_train = train_enc['target'].astype(int)

    model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                           use_label_encoder=False, eval_metric='mlogloss',
                           random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train)

    # Predict
    X_test = test[avail].fillna(0)
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)
    classes = list(model.classes_)
    
    buy_idx = classes.index(2) if 2 in classes else 0
    test = test.copy()
    test['buy_prob'] = probas[:, buy_idx]
    test['predicted'] = preds

    # Compute portfolio return using REAL monthly_return
    buy_mask = test['predicted'] == 2
    if buy_mask.sum() == 0:
        portfolio_return = 0.0
    else:
        buy_stocks = test[buy_mask]
        weights = buy_stocks['buy_prob'].values
        weights = weights / weights.sum()
        # Use 'real_return' column
        returns = buy_stocks['real_return'].fillna(0).values
        portfolio_return = float(np.dot(weights, returns))

    net_return = portfolio_return - (0.3 * TRANSACTION_COST * 2) # 30% turnover assumption
    return {'test_month': test_month, 'net_return': net_return, 'n_buy': int(buy_mask.sum())}

def compute_metrics(fold_results):
    if not fold_results: return {}
    returns = np.array([r['net_return'] for r in fold_results])
    rf = 0.005 # monthly 6% annual
    
    cumulative = np.prod(1 + returns) - 1
    ann_return = (1 + cumulative)**(12/len(returns)) - 1
    excess = returns - rf
    sharpe = (np.mean(excess) / np.std(excess)) * np.sqrt(12) if np.std(excess)>0 else 0
    
    cum_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_returns)
    dd = (cum_returns - peak) / peak
    max_dd = float(np.min(dd))
    
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    
    return {
        'cumulative_return': float(cumulative),
        'annualized_return': float(ann_return),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_dd),
        'calmar_ratio': float(calmar),
        'hit_rate': float(np.mean(returns > 0))
    }

def main():
    print("=" * 70)
    print("STEP 16 FIXED -- REAL RETURN BACKTEST")
    print("=" * 70)

    # 1. Load Data
    feat_df = pd.read_csv(os.path.join(FEATURES_DIR, 'LPCMCI_READY.csv'), low_memory=False)
    real_df = pd.read_csv(os.path.join(FEATURES_DIR, 'CAUSAL_DISCOVERY_DATASET.csv'), 
                          usecols=['ISIN', 'year_month_str', 'monthly_return'], low_memory=False)
    real_df = real_df.rename(columns={'monthly_return': 'real_return'})

    # 2. Merge Real Returns into Features
    df = feat_df.merge(real_df, on=['ISIN', 'year_month_str'], how='left')
    print(f"  Merged dataset: {df.shape}")

    # 3. Load MB Features
    with open(os.path.join(FINAL_DIR, 'markov_blanket_features.json')) as f:
        mb_data = json.load(f)
    
    # 4. Feature Exclusion (Prevent Leakage)
    exclude = {'year_month_str','ISIN','Fund_Name','Fund_Type','sector','stock_name',
               'position_action','allocation_momentum', 'pct_nav', 'allocation_change', 
               'quantity', 'market_value', 'real_return', 'target', 'date', 'Date',
               'quantity_change', 'month_gap', 'fund_stock_count', 'is_top10', 'allocation_quintile'}
    
    feature_sets = {
        'M0_all': [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude],
        'M1_causal_MB': [c for c in mb_data['mb_columns'] if c in df.columns],
        'M2_correlation': [c for c in mb_data['correlation_columns'] if c in df.columns]
    }

    all_months = sorted(df['year_month_str'].dropna().unique())
    test_months = all_months[24:] # start after 24 months training

    results = {}
    for name, feats in feature_sets.items():
        print(f"  Backtesting {name} ({len(feats)} features)...")
        folds = []
        for m in test_months:
            idx = all_months.index(m)
            res = run_fold(df, all_months[:idx], m, feats)
            if res: folds.append(res)
        
        results[name] = compute_metrics(folds)
        results[name]['n_features'] = len(feats)

    # Summary
    print("
" + "="*60)
    print(f"{'Model':<20} {'Sharpe':>8} {'AnnRet':>10} {'MaxDD':>10} {'Calmar':>8}")
    print("-"*60)
    for name, m in results.items():
        print(f"{name:<20} {m['sharpe_ratio']:>8.2f} {m['annualized_return']:>10.1%} "
              f"{m['max_drawdown']:>10.1%} {m['calmar_ratio']:>8.2f}")

    with open(os.path.join(FINAL_DIR, 'backtest_fixed_real.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
