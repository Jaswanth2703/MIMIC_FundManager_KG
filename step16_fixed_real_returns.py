
"""
Step 16 FIXED -- Walk-Forward Backtest with Real Returns
=========================================================
Uses unstandardized monthly_return from CAUSAL_DISCOVERY_DATASET.csv
to calculate real financial performance.
"""

import sys, os
import json, traceback, warnings
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES_DIR, FINAL_DIR

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False

ACTION_MAP = {'BUY':2,'INCREASE':2,'INITIAL_POSITION':2,'HOLD':1,'DECREASE':0,'SELL':0}
TRANSACTION_COST = 0.005  # 0.50% one-way

def run_fold(df, train_months, test_month, feature_cols, prev_buys=None, embargo=1):
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
    curr_buys = set(test.loc[buy_mask, 'ISIN'].values) if 'ISIN' in test.columns else set()

    if buy_mask.sum() == 0:
        portfolio_return = 0.0
    else:
        buy_stocks = test[buy_mask]
        weights = buy_stocks['buy_prob'].values
        weights = weights / weights.sum()
        returns = buy_stocks['real_return'].fillna(0).values
        portfolio_return = float(np.dot(weights, returns))

    # Calculate actual turnover vs previous month
    if prev_buys is not None and len(prev_buys) > 0:
        overlap = len(curr_buys & prev_buys)
        total = max(len(curr_buys | prev_buys), 1)
        turnover = 1.0 - (overlap / total)
    else:
        turnover = 1.0  # first month: full turnover

    net_return = portfolio_return - (turnover * TRANSACTION_COST * 2)

    # Benchmark: equally weighted return of all stocks this month
    benchmark_return = float(test['real_return'].fillna(0).mean())

    return {
        'test_month': test_month,
        'net_return': net_return,
        'gross_return': portfolio_return,
        'benchmark_return': benchmark_return,
        'turnover': turnover,
        'n_buy': int(buy_mask.sum()),
        'curr_buys': curr_buys,
    }

def compute_metrics(fold_results):
    if not fold_results: return {}
    returns = np.array([r['net_return'] for r in fold_results])
    bench_returns = np.array([r['benchmark_return'] for r in fold_results])
    rf = 0.005 # monthly 6% annual
    
    cumulative = np.prod(1 + returns) - 1
    ann_return = (1 + cumulative)**(12/len(returns)) - 1
    excess = returns - rf
    sharpe = (np.mean(excess) / np.std(excess)) * np.sqrt(12) if np.std(excess)>0 else 0
    
    # Sortino ratio (penalizes downside volatility only)
    downside = excess[excess < 0]
    downside_std = np.std(downside) if len(downside) > 0 else np.std(excess)
    sortino = (np.mean(excess) / downside_std) * np.sqrt(12) if downside_std > 0 else 0

    # Information ratio (vs benchmark)
    active_returns = returns - bench_returns
    tracking_error = np.std(active_returns)
    info_ratio = (np.mean(active_returns) / tracking_error) * np.sqrt(12) \
                 if tracking_error > 0 else 0

    cum_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_returns)
    dd = (cum_returns - peak) / peak
    max_dd = float(np.min(dd))
    
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    avg_turnover = float(np.mean([r['turnover'] for r in fold_results]))

    return {
        'cumulative_return': float(cumulative),
        'annualized_return': float(ann_return),
        'sharpe_ratio': float(sharpe),
        'sortino_ratio': float(sortino),
        'information_ratio': float(info_ratio),
        'max_drawdown': float(max_dd),
        'calmar_ratio': float(calmar),
        'hit_rate': float(np.mean(returns > 0)),
        'avg_turnover': avg_turnover,
        'benchmark_cumulative': float(np.prod(1 + bench_returns) - 1),
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
    mb_path = os.path.join(FINAL_DIR, 'markov_blanket_features.json')
    if os.path.exists(mb_path):
        with open(mb_path) as f:
            mb_data = json.load(f)
    else:
        mb_data = {'mb_columns': [], 'correlation_columns': []}
        print("  WARNING: markov_blanket_features.json not found, using M0 only")
    
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
        prev_buys = None
        for m in test_months:
            idx = all_months.index(m)
            res = run_fold(df, all_months[:idx], m, feats, prev_buys=prev_buys)
            if res:
                prev_buys = res.pop('curr_buys', set())
                folds.append(res)
        
        results[name] = compute_metrics(folds)
        results[name]['n_features'] = len(feats)

    # 5. Backtest KG-based model predictions (CBR, PathTransformer, HGT, CI-HGT)
    print("\n  --- KG-Based Model Backtests ---")
    kg_pred_files = {
        'M4_CBR_KG': (os.path.join(FINAL_DIR, 'cbr_decision_predictions.csv'), 'cbr_predicted'),
        'M5_PathTransformer': (os.path.join(FINAL_DIR, 'path_decision_predictions.csv'), 'path_predicted'),
        'M6_HGT': (os.path.join(FINAL_DIR, 'hgt_decision_predictions.csv'), 'hgt_predicted'),
        'M7_CI_HGT': (os.path.join(FINAL_DIR, 'ci_hgt_decision_predictions.csv'), 'ci_hgt_predicted'),
    }

    kg_predictions = {}
    for model_name, (pred_file, pred_col) in kg_pred_files.items():
        if not os.path.exists(pred_file):
            print(f"  {model_name}: predictions not found, skipping")
            continue
        pred_df = pd.read_csv(pred_file)
        if pred_col not in pred_df.columns:
            print(f"  {model_name}: column '{pred_col}' not in predictions, skipping")
            continue
        kg_predictions[model_name] = (pred_df, pred_col)
        print(f"  {model_name}: loaded {len(pred_df)} predictions")

    for model_name, (pred_df, pred_col) in kg_predictions.items():
        # Merge KG predictions with real returns
        merged = pred_df.merge(
            df[['ISIN', 'year_month_str', 'real_return']].drop_duplicates(),
            on=['ISIN', 'year_month_str'], how='left')
        merged['real_return'] = merged['real_return'].fillna(0)

        # Backtest month by month
        kg_months = sorted(merged['year_month_str'].unique())
        folds = []
        prev_buys = None
        for m in kg_months:
            month_data = merged[merged['year_month_str'] == m]
            buy_mask = month_data[pred_col] == 2
            curr_buys = set(month_data.loc[buy_mask, 'ISIN'].values) \
                if 'ISIN' in month_data.columns else set()

            if buy_mask.sum() > 0:
                returns = month_data.loc[buy_mask, 'real_return'].values
                portfolio_return = float(np.mean(returns))  # equal weight
            else:
                portfolio_return = 0.0

            if prev_buys is not None and len(prev_buys) > 0:
                overlap = len(curr_buys & prev_buys)
                total = max(len(curr_buys | prev_buys), 1)
                turnover = 1.0 - (overlap / total)
            else:
                turnover = 1.0

            net_return = portfolio_return - (turnover * TRANSACTION_COST * 2)
            benchmark_return = float(month_data['real_return'].mean())

            folds.append({
                'test_month': m,
                'net_return': net_return,
                'gross_return': portfolio_return,
                'benchmark_return': benchmark_return,
                'turnover': turnover,
                'n_buy': int(buy_mask.sum()),
            })
            prev_buys = curr_buys

        if folds:
            results[model_name] = compute_metrics(folds)
            results[model_name]['n_features'] = 0
            results[model_name]['source'] = 'KG_model_predictions'
            print(f"  {model_name}: Sharpe={results[model_name]['sharpe_ratio']:.2f} "
                  f"AnnRet={results[model_name]['annualized_return']:.1%}")

    # 6. Ensemble: majority vote across available KG models
    if len(kg_predictions) >= 2:
        print("\n  Building KG Ensemble (majority vote)...")
        # Merge all predictions on (ISIN, year_month_str)
        ensemble_base = None
        for model_name, (pred_df, pred_col) in kg_predictions.items():
            sub = pred_df[['ISIN', 'year_month_str', pred_col]].copy()
            sub = sub.rename(columns={pred_col: f'pred_{model_name}'})
            if ensemble_base is None:
                ensemble_base = sub
            else:
                ensemble_base = ensemble_base.merge(sub, on=['ISIN', 'year_month_str'], how='inner')

        if ensemble_base is not None and len(ensemble_base) > 0:
            pred_cols = [c for c in ensemble_base.columns if c.startswith('pred_')]
            # Majority vote
            ensemble_base['ensemble_pred'] = ensemble_base[pred_cols].mode(axis=1)[0].astype(int)

            # Merge real returns
            ens_merged = ensemble_base.merge(
                df[['ISIN', 'year_month_str', 'real_return']].drop_duplicates(),
                on=['ISIN', 'year_month_str'], how='left')
            ens_merged['real_return'] = ens_merged['real_return'].fillna(0)

            # Backtest ensemble
            ens_months = sorted(ens_merged['year_month_str'].unique())
            folds = []
            prev_buys = None
            for m in ens_months:
                month_data = ens_merged[ens_merged['year_month_str'] == m]
                buy_mask = month_data['ensemble_pred'] == 2
                curr_buys = set(month_data.loc[buy_mask, 'ISIN'].values)

                portfolio_return = float(month_data.loc[buy_mask, 'real_return'].mean()) \
                    if buy_mask.sum() > 0 else 0.0

                if prev_buys is not None and len(prev_buys) > 0:
                    overlap = len(curr_buys & prev_buys)
                    turnover = 1.0 - (overlap / max(len(curr_buys | prev_buys), 1))
                else:
                    turnover = 1.0

                net_return = portfolio_return - (turnover * TRANSACTION_COST * 2)
                benchmark_return = float(month_data['real_return'].mean())
                folds.append({
                    'test_month': m, 'net_return': net_return,
                    'gross_return': portfolio_return,
                    'benchmark_return': benchmark_return,
                    'turnover': turnover, 'n_buy': int(buy_mask.sum()),
                })
                prev_buys = curr_buys

            if folds:
                results['M8_KG_Ensemble'] = compute_metrics(folds)
                results['M8_KG_Ensemble']['n_features'] = 0
                results['M8_KG_Ensemble']['source'] = 'KG_ensemble_majority_vote'
                results['M8_KG_Ensemble']['n_models'] = len(kg_predictions)
                print(f"  Ensemble: Sharpe={results['M8_KG_Ensemble']['sharpe_ratio']:.2f} "
                      f"AnnRet={results['M8_KG_Ensemble']['annualized_return']:.1%}")

    # Summary
    print("\n" + "="*85)
    print(f"{'Model':<22} {'Sharpe':>8} {'Sortino':>8} {'IR':>8} {'AnnRet':>10} "
          f"{'MaxDD':>8} {'Calmar':>8} {'Turn%':>7}")
    print("-"*85)
    for name, m in results.items():
        print(f"{name:<22} {m.get('sharpe_ratio',0):>8.2f} {m.get('sortino_ratio',0):>8.2f} "
              f"{m.get('information_ratio',0):>8.2f} {m.get('annualized_return',0):>10.1%} "
              f"{m.get('max_drawdown',0):>8.1%} {m.get('calmar_ratio',0):>8.2f} "
              f"{m.get('avg_turnover',0):>6.0%}")

    with open(os.path.join(FINAL_DIR, 'backtest_fixed_real.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
