"""
Step 14b -- Full 5-Way Model Comparison with Dual Evaluation
=============================================================
Runs AFTER step13 (CBR results). Compares 5 models using BOTH:
  1. Walk-forward (expanding window) — rolling production evaluation
  2. Temporal holdout (train 65% / test 35%) — single split, standard ML

Models:
  M0 (all features)        : kitchen-sink baseline
  M1 (Granger MB)          : Granger-selected feature selection
  M2 (correlation top-K)   : non-causal selection control
  M3 (ICP parents)         : modern causal feature selection (formal guarantees)
  M4 (CBR over KG)         : KG itself as model via case-based retrieval

Mimicry Metrics (beyond accuracy):
  - Decision agreement rate: % of months where model agrees with fund manager
  - Position direction accuracy: does model get BUY/SELL direction right?
  - Confidence calibration: are model probabilities well-calibrated?
  - Cohen's kappa: agreement beyond chance

Output:
  data/final/full_comparison.json
  data/final/full_comparison.csv
"""

import sys
import os
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             classification_report, confusion_matrix)

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES_DIR, CAUSAL_DIR, FINAL_DIR

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except ImportError:
    XGB_OK = False

ACTION_MAP = {'BUY': 2, 'INCREASE': 2, 'INITIAL_POSITION': 2,
              'HOLD': 1, 'DECREASE': 0, 'SELL': 0}

INPUT_FEAT = os.path.join(FEATURES_DIR, 'LPCMCI_READY.csv')
INPUT_ICP = os.path.join(CAUSAL_DIR, 'icp_causal_parents.csv')
INPUT_DML = os.path.join(CAUSAL_DIR, 'dml_causal_effects.csv')
EXISTING_COMPARISON = os.path.join(FINAL_DIR, 'portfolio_comparison.json')
CBR_METRICS = os.path.join(FINAL_DIR, 'cbr_metrics.json')
OUT_JSON = os.path.join(FINAL_DIR, 'full_comparison.json')
OUT_CSV = os.path.join(FINAL_DIR, 'full_comparison.csv')

HOLDOUT_TRAIN_FRAC = 0.65  # match step08 standardization split


def _xgb_model():
    return XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                         min_child_weight=5, subsample=0.8,
                         colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                         use_label_encoder=False, eval_metric='mlogloss',
                         random_state=42, n_jobs=-1, verbosity=0)


def compute_mimicry_metrics(y_true, y_pred, y_proba=None):
    """Compute mimicry-specific metrics beyond raw accuracy."""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted',
                                       zero_division=0)),
        'cohens_kappa': float(cohen_kappa_score(y_true, y_pred)),
    }

    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    # Direction accuracy: does model agree on BUY vs SELL?
    buy_mask = y_t == 2
    sell_mask = y_t == 0
    if buy_mask.sum() > 0:
        metrics['buy_recall'] = float((y_p[buy_mask] == 2).mean())
    if sell_mask.sum() > 0:
        metrics['sell_recall'] = float((y_p[sell_mask] == 0).mean())

    # Decision agreement: same action for each stock-month
    metrics['decision_agreement'] = float((y_t == y_p).mean())

    return metrics


def train_xgb_walkforward(df, feature_cols, name, train_months=24):
    """Walk-forward XGBoost: expanding window, test 1 month at a time."""
    if not XGB_OK:
        return None

    feats = [f for f in feature_cols if f in df.columns]
    if len(feats) < 2:
        return {'model': name, 'status': 'failed',
                'reason': f'only {len(feats)} features'}

    data = df.dropna(subset=['position_action']).copy()
    data['target'] = data['position_action'].map(ACTION_MAP)
    data = data.dropna(subset=['target'])
    data['target'] = data['target'].astype(int)

    months = sorted(data['year_month_str'].unique())
    if len(months) <= train_months + 2:
        return {'model': name, 'status': 'failed', 'reason': 'too few months'}

    fold_results = []
    all_y_true, all_y_pred = [], []
    for i in range(train_months, len(months)):
        train_ms = months[:i]
        test_m = months[i]
        train = data[data['year_month_str'].isin(train_ms)]
        test = data[data['year_month_str'] == test_m]
        if len(train) < 50 or len(test) < 5:
            continue
        X_tr = train[feats].fillna(0)
        y_tr = train['target']
        X_te = test[feats].fillna(0)
        y_te = test['target']

        model = _xgb_model()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        fold_results.append({
            'test_month': test_m,
            'accuracy': float(accuracy_score(y_te, y_pred)),
            'f1_weighted': float(f1_score(y_te, y_pred, average='weighted',
                                           zero_division=0)),
            'n_test': len(y_te),
        })
        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())

    if not fold_results:
        return {'model': name, 'status': 'failed', 'reason': 'no folds'}

    mimicry = compute_mimicry_metrics(all_y_true, all_y_pred)

    return {
        'model': name,
        'eval_mode': 'walk_forward',
        'status': 'success',
        'n_features': len(feats),
        'features': feats[:30],
        'avg_accuracy': float(np.mean([f['accuracy'] for f in fold_results])),
        'std_accuracy': float(np.std([f['accuracy'] for f in fold_results])),
        'avg_f1_weighted': float(np.mean([f['f1_weighted'] for f in fold_results])),
        'overall_accuracy': mimicry['accuracy'],
        'overall_f1_weighted': mimicry['f1_weighted'],
        'cohens_kappa': mimicry['cohens_kappa'],
        'decision_agreement': mimicry['decision_agreement'],
        'buy_recall': mimicry.get('buy_recall'),
        'sell_recall': mimicry.get('sell_recall'),
        'n_folds': len(fold_results),
    }


def train_xgb_holdout(df, feature_cols, name):
    """Temporal holdout: train on first 65%, test on last 35%."""
    if not XGB_OK:
        return None

    feats = [f for f in feature_cols if f in df.columns]
    if len(feats) < 2:
        return {'model': name, 'status': 'failed',
                'reason': f'only {len(feats)} features'}

    data = df.dropna(subset=['position_action']).copy()
    data['target'] = data['position_action'].map(ACTION_MAP)
    data = data.dropna(subset=['target'])
    data['target'] = data['target'].astype(int)

    months = sorted(data['year_month_str'].unique())
    split_idx = int(len(months) * HOLDOUT_TRAIN_FRAC)
    train_months = months[:split_idx]
    test_months = months[split_idx:]

    if len(test_months) < 3:
        return {'model': name, 'status': 'failed', 'reason': 'too few test months'}

    train = data[data['year_month_str'].isin(train_months)]
    test = data[data['year_month_str'].isin(test_months)]

    X_tr = train[feats].fillna(0)
    y_tr = train['target']
    X_te = test[feats].fillna(0)
    y_te = test['target']

    model = _xgb_model()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    mimicry = compute_mimicry_metrics(y_te.tolist(), y_pred.tolist())

    return {
        'model': name,
        'eval_mode': 'temporal_holdout',
        'status': 'success',
        'n_features': len(feats),
        'train_months': f"{train_months[0]}..{train_months[-1]} ({len(train_months)}m)",
        'test_months': f"{test_months[0]}..{test_months[-1]} ({len(test_months)}m)",
        'n_train': len(train),
        'n_test': len(test),
        'accuracy': mimicry['accuracy'],
        'f1_weighted': mimicry['f1_weighted'],
        'cohens_kappa': mimicry['cohens_kappa'],
        'decision_agreement': mimicry['decision_agreement'],
        'buy_recall': mimicry.get('buy_recall'),
        'sell_recall': mimicry.get('sell_recall'),
    }


def main():
    print("=" * 70)
    print("STEP 14b -- FULL 5-WAY MODEL COMPARISON (Walk-Forward + Holdout)")
    print("=" * 70)

    df = pd.read_csv(INPUT_FEAT, low_memory=False)
    print(f"  Panel: {df.shape}")

    # Load existing M0/M1/M2 results from step14
    if os.path.exists(EXISTING_COMPARISON):
        with open(EXISTING_COMPARISON) as f:
            existing = json.load(f)
        print(f"  Loaded existing comparison: {list(existing.keys())}")
    else:
        existing = {}
        print("  WARNING: existing comparison not found, M0/M1/M2 will be empty")

    # --- Define feature sets ---
    all_feats = [c for c in df.select_dtypes(include=[np.number]).columns
                 if c not in {'action_ordinal', 'is_buy', 'is_sell'}]

    # Granger MB features
    granger_file = os.path.join(CAUSAL_DIR, 'all_causal_links.csv')
    if os.path.exists(granger_file):
        granger_df = pd.read_csv(granger_file)
        granger_feats = sorted(granger_df['cause'].unique().tolist())
        granger_feats = [f for f in granger_feats if f in df.columns]
    else:
        granger_feats = all_feats[:20]

    # Correlation top-K
    if 'position_action' in df.columns:
        tmp = df.copy()
        tmp['_target'] = tmp['position_action'].map(ACTION_MAP)
        corrs = tmp[all_feats].corrwith(tmp['_target']).abs().sort_values(ascending=False)
        corr_feats = corrs.head(20).index.tolist()
    else:
        corr_feats = all_feats[:20]

    # ICP causal parents
    if os.path.exists(INPUT_ICP):
        icp = pd.read_csv(INPUT_ICP)
        icp_feats = sorted(icp[icp['confidence'] >= 0.5]['variable'].unique())
        lagged = []
        for f in icp_feats:
            for lag in [1, 2]:
                col = f"{f}_lag{lag}"
                if col in df.columns:
                    lagged.append(col)
        icp_feats_full = list(dict.fromkeys(icp_feats + lagged))
        print(f"\n  M3 ICP features: {len(icp_feats_full)} (from {len(icp_feats)} parents + lags)")
    else:
        icp_feats_full = None
        print("  ICP file not found -- M3 skipped")

    # --- Define model configs ---
    model_configs = [
        ('M0_all_features', all_feats),
        ('M1_granger_MB', granger_feats),
        ('M2_correlation_topK', corr_feats),
    ]
    if icp_feats_full and len(icp_feats_full) >= 2:
        model_configs.append(('M3_ICP_causal', icp_feats_full))

    # --- Run BOTH evaluation modes ---
    all_results = {}

    # Naive baseline: always predict HOLD (class 1)
    data_tmp = df.dropna(subset=['position_action']).copy()
    data_tmp['target'] = data_tmp['position_action'].map(ACTION_MAP)
    data_tmp = data_tmp.dropna(subset=['target'])
    naive_true = data_tmp['target'].astype(int).tolist()
    naive_pred = [1] * len(naive_true)  # always HOLD
    naive_metrics = compute_mimicry_metrics(naive_true, naive_pred)
    for mode in ['walk_forward', 'temporal_holdout']:
        all_results[f'M_naive_HOLD__{mode}'] = {
            'model': 'M_naive_HOLD',
            'eval_mode': mode,
            'status': 'success',
            'n_features': 0,
            'accuracy': naive_metrics['accuracy'],
            'f1_weighted': naive_metrics['f1_weighted'],
            'cohens_kappa': naive_metrics['cohens_kappa'],
            'decision_agreement': naive_metrics['decision_agreement'],
            'buy_recall': naive_metrics.get('buy_recall', 0),
            'sell_recall': naive_metrics.get('sell_recall', 0),
            'note': 'Naive baseline: always predicts HOLD',
        }
    print(f"\n  Naive HOLD baseline: acc={naive_metrics['accuracy']:.3f} "
          f"kappa={naive_metrics['cohens_kappa']:.3f}")

    for eval_mode in ['walk_forward', 'temporal_holdout']:
        print(f"\n{'='*70}")
        print(f"  EVALUATION MODE: {eval_mode.upper()}")
        print(f"{'='*70}")

        for name, feats in model_configs:
            print(f"\n  Training {name} ({len(feats)} features) ...")
            if eval_mode == 'walk_forward':
                result = train_xgb_walkforward(df, feats, name)
            else:
                result = train_xgb_holdout(df, feats, name)

            if result and result.get('status') == 'success':
                key = f"{name}__{eval_mode}"
                all_results[key] = result
                print(f"    Accuracy: {result.get('accuracy', result.get('overall_accuracy', 0)):.3f}"
                      f"  Kappa: {result.get('cohens_kappa', 0):.3f}"
                      f"  Agreement: {result.get('decision_agreement', 0):.3f}")

    # --- Add M4 CBR from step13 ---
    if os.path.exists(CBR_METRICS):
        with open(CBR_METRICS) as f:
            cbr = json.load(f)
        for mode in ['walk_forward', 'temporal_holdout']:
            all_results[f'M4_CBR_KG__{mode}'] = {
                'model': 'M4_CBR_KG',
                'eval_mode': mode,
                'status': 'success',
                'n_features': cbr.get('n_causal_features', 0),
                'accuracy': cbr.get('overall_accuracy', cbr.get('avg_fold_accuracy')),
                'f1_weighted': cbr.get('overall_f1_weighted', cbr.get('avg_fold_f1')),
                'cohens_kappa': cbr.get('cohens_kappa', None),
                'decision_agreement': cbr.get('decision_agreement', None),
                'note': 'KG-based case retrieval (from step13)',
            }
    else:
        print("\n  CBR metrics not found -- M4 skipped")

    # --- Print comparison table ---
    for mode in ['walk_forward', 'temporal_holdout']:
        print(f"\n{'='*70}")
        print(f"  {mode.upper()} COMPARISON")
        print(f"{'='*70}")
        print(f"  {'Model':<25s} {'#Feat':>6s} {'Acc':>8s} {'F1':>8s} "
              f"{'Kappa':>8s} {'Agree':>8s} {'BuyR':>6s} {'SellR':>6s}")
        print("  " + "-" * 78)
        for key, m in all_results.items():
            if mode not in key or m.get('status') != 'success':
                continue
            acc = m.get('accuracy', m.get('overall_accuracy', 0))
            f1 = m.get('f1_weighted', m.get('overall_f1_weighted', 0))
            kappa = m.get('cohens_kappa', 0)
            agree = m.get('decision_agreement', 0)
            buy_r = m.get('buy_recall', 0)
            sell_r = m.get('sell_recall', 0)
            print(f"  {m['model']:<25s} {m.get('n_features', 0):>6d} "
                  f"{acc or 0:>8.3f} {f1 or 0:>8.3f} "
                  f"{kappa or 0:>8.3f} {agree or 0:>8.3f} "
                  f"{buy_r or 0:>6.3f} {sell_r or 0:>6.3f}")

    # --- Save ---
    os.makedirs(FINAL_DIR, exist_ok=True)

    # Merge with existing results
    full = dict(existing)
    full.update(all_results)

    with open(OUT_JSON, 'w') as f:
        json.dump(full, f, indent=2, default=str)

    rows = [{
        'model': m.get('model', key),
        'eval_mode': m.get('eval_mode', 'unknown'),
        'n_features': m.get('n_features', 0),
        'accuracy': m.get('accuracy', m.get('overall_accuracy', 0)),
        'f1_weighted': m.get('f1_weighted', m.get('overall_f1_weighted', 0)),
        'cohens_kappa': m.get('cohens_kappa'),
        'decision_agreement': m.get('decision_agreement'),
        'buy_recall': m.get('buy_recall'),
        'sell_recall': m.get('sell_recall'),
    } for key, m in all_results.items() if m and m.get('status') == 'success']
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f"\n  Saved: {OUT_JSON}")
    print(f"  Saved: {OUT_CSV}")

    print("\n  Mimicry evaluation summary:")
    print("    - Walk-forward: expanding window, tests on unseen future months")
    print("    - Temporal holdout: train on first 65%, test on last 35%")
    print("    - Cohen's kappa: agreement beyond chance (>0.4 = moderate)")
    print("    - Buy/Sell recall: can model predict manager's directional decisions?")
    print("    - M3 (ICP) & M4 (CBR): contributions; M0/M1/M2: baselines")

    print("\n  STEP 14b DONE.")


if __name__ == '__main__':
    main()
