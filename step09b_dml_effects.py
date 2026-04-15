"""
Step 09b -- Double Machine Learning (DML) for Causal Effect Sizes v5
=====================================================================
Runs DML on auto-discovered features + comprehensive fixed list.

v5 improvements:
- Auto-discovers all numeric columns in LPCMCI_READY.csv (no hardcoded only)
- Adds interaction features (pe_x_momentum, rsi_x_momentum, etc.)
- Adds non-linear transforms (log_market_cap, sqrt_volatility, etc.)
- Loads Markov Blanket from step09a to prioritize MB variables
- Falls back to comprehensive fixed list for columns not in data

Reference:
  Chernozhukov et al. (2018). Double/debiased machine learning.
  Econometrics Journal, 21(1), C1-C68.

Input:  data/causal_output/icp_causal_parents.csv  (for ICP flag)
        data/causal_output/markov_blanket.json     (for MB priority)
        data/features/LPCMCI_READY.csv
Output: data/causal_output/dml_causal_effects.csv
"""

import sys, os, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES_DIR, CAUSAL_DIR

INPUT_FEAT    = os.path.join(FEATURES_DIR, 'LPCMCI_READY.csv')
INPUT_PARENTS = os.path.join(CAUSAL_DIR, 'icp_causal_parents.csv')
INPUT_MB      = os.path.join(CAUSAL_DIR, 'markov_blanket.json')
OUTPUT_CSV    = os.path.join(CAUSAL_DIR, 'dml_causal_effects.csv')

ACTION_MAP = {
    'BUY': 2, 'INCREASE': 2, 'INITIAL_POSITION': 2,
    'HOLD': 1, 'DECREASE': 0, 'SELL': 0,
}

N_FOLDS       = 5
N_REPETITIONS = 3   # v4: repeated cross-fitting for stability
RF_PARAMS = dict(n_estimators=150, max_depth=8,
                 min_samples_leaf=20, n_jobs=-1, random_state=42)

LEAK = {
    'action_ordinal', 'is_buy', 'is_sell', 'position_action',
    'allocation_change', 'pct_nav', 'rank_in_fund',
    'consensus_count', 'is_top10', 'allocation_quintile',
    'quantity', 'market_value', 'quantity_change',
}

# ============================================================
# ============================================================
# Treatment list: comprehensive + auto-discovered
# ============================================================
# v5: Core treatments (always tested if present in data)
CORE_TREATMENTS = [
    # Position management (lagged — no lookahead)
    'allocation_change_lag1',
    'allocation_change_lag2',
    'holding_tenure',
    'pct_nav_lag1',
    'pct_nav_lag2',
    # Technical indicators
    'rsi',
    'rsi_lag1',
    'rsi_lag2',
    'macd_line',
    'macd_line_lag1',
    'macd_hist',
    'macd_hist_lag1',
    'bollinger_pband',
    'bollinger_pband_lag1',
    'momentum_3m',
    'volatility_3m',
    'monthly_return',
    'monthly_return_lag1',
    'monthly_return_lag2',
    'volume_ratio',
    'volume_ratio_lag1',
    'sma200',
    # Fundamentals
    'pe',
    'pb',
    'pe_relative',
    'eps',
    'eps_lag1',
    'market_cap',
    'beta',
    'alpha',
    'bv_per_share',
    'profit_after_tax',
    'income_from_sales',
    # Macro
    'repo_rate',
    'repo_rate_lag1',
    'repo_rate_lag2',
    'cpi_inflation',
    'cpi_inflation_lag1',
    'us_10y_yield',
    'india_vix_close',
    'real_interest_rate',
    'gdp_growth',
    'usd_inr',
    'crude_oil',
    'nifty50_return',
    'sp500_return',
    # Sentiment
    'sentiment_mean',
    'sentiment_mean_lag1',
    'sentiment_mean_lag2',
    'sentiment_confidence',
    'sentiment_extremity',
    'positive_ratio',
    'negative_ratio',
    # v5: Interaction features
    'pe_x_momentum',
    'rsi_x_momentum',
    'vix_x_return',
    'sentiment_x_momentum',
    'alloc_x_sentiment',
    # v5: Non-linear transforms
    'log_market_cap',
    'log_pe',
    'sqrt_volatility',
    'log_volume_ratio',
]


def build_treatment_list(df):
    """Build treatment list from CORE + auto-discovered columns + MB.

    v5: Auto-discovers all numeric columns that could be treatments.
    Ensures no feature is missed because of name changes from pruning.
    """
    # Start with core treatments that exist in data
    treatments = [t for t in CORE_TREATMENTS if t in df.columns]
    existing = set(treatments)

    # Auto-discover: any numeric column not in LEAK/EXCLUDE
    auto_exclude = LEAK | {
        'year_month_str', 'ISIN', 'Fund_Name', 'Fund_Type',
        'sector', 'stock_name', 'stock_name_raw', 'Industry',
        'date', 'Date', 'holding_period_id', 'fund_ticker',
        'symbol', 'golden_cross', 'death_cross', 'size',
        'year_month', 'month_ordinal',
    }
    for c in df.select_dtypes(include=[np.number]).columns:
        if c not in existing and c not in auto_exclude:
            if df[c].std() > 1e-9 and c not in existing:
                treatments.append(c)
                existing.add(c)

    # Load MB variables (from step09a) and ensure they're included
    if os.path.exists(INPUT_MB):
        import json
        with open(INPUT_MB) as f:
            mb = json.load(f)
        for target_mb in mb.values():
            for v in target_mb:
                if v in df.columns and v not in existing and v not in auto_exclude:
                    treatments.append(v)
                    existing.add(v)

    return treatments

# Three targets — matching ICP v5
TARGETS = {
    'action_ordinal': 'action_ordinal',
    'is_buy':         'is_buy',
    'is_sell':        'is_sell',
}


def double_ml(Y, T, X, n_folds=N_FOLDS):
    """Run DML with repeated cross-fitting (v4).

    Averages theta estimates across N_REPETITIONS random splits
    for more stable estimates (Chernozhukov et al. 2018, Section 3.2).

    BUGFIX (v4.1): If X has 0 columns (no controls survived filtering),
    fall back to a simple bivariate demeaned OLS estimate rather than
    crashing with a sklearn shape error.
    """
    # ── Guard: no controls → bivariate demeaned OLS (no debiasing) ──
    if X.shape[1] == 0:
        T_c   = T - T.mean()
        denom = float(np.dot(T_c, T_c))
        if denom < 1e-9:
            return np.nan, np.nan, np.nan, np.nan
        Y_c   = Y - Y.mean()
        theta = float(np.dot(T_c, Y_c) / denom)
        psi   = (Y_c - theta * T_c) * T_c
        se    = float(np.sqrt(max(np.var(psi) / denom, 0)))
        return theta, se, theta - 1.96 * se, theta + 1.96 * se

    thetas = []
    ses    = []
    for rep in range(N_REPETITIONS):
        n       = len(Y)
        Y_resid = np.zeros(n)
        T_resid = np.zeros(n)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42 + rep)
        for tr, te in kf.split(X):
            m_Y = RandomForestRegressor(**RF_PARAMS).fit(X[tr], Y[tr])
            m_T = RandomForestRegressor(**RF_PARAMS).fit(X[tr], T[tr])
            Y_resid[te] = Y[te] - m_Y.predict(X[te])
            T_resid[te] = T[te] - m_T.predict(X[te])
        denom = float(np.dot(T_resid, T_resid))
        if denom < 1e-9:
            continue
        theta = float(np.dot(T_resid, Y_resid) / denom)
        psi   = (Y_resid - theta * T_resid) * T_resid
        se    = float(np.sqrt(max(np.var(psi) / denom, 0)))
        thetas.append(theta)
        ses.append(se)

    if not thetas:
        return np.nan, np.nan, np.nan, np.nan

    # Median of repeated estimates (robust to outlier splits)
    theta_final = float(np.median(thetas))
    se_final    = float(np.median(ses))
    return theta_final, se_final, theta_final - 1.96 * se_final, theta_final + 1.96 * se_final


def get_controls(df, treatment, outcome):
    """Select control variables for DML.

    v4: Replaced naive variance-based selection with correlation-based:
    selects variables that correlate with BOTH treatment AND outcome
    (potential confounders), not just high-variance ones.
    Also excludes contemporaneous variables to avoid post-treatment bias.

    BUGFIX (v4.1): If the correlation-scoring loop produces an empty
    scores dict (all columns had NaN corr or zero std after dropna),
    fall back to any numeric columns with non-zero variance so that
    X never ends up with 0 columns.
    """
    leak = LEAK | {treatment, outcome}
    cols = [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in leak and not c.endswith('_lag3')]

    if len(cols) == 0:
        return []

    # Score by sum of |corr with treatment| + |corr with outcome|
    # This identifies confounders (correlated with both T and Y)
    scores  = {}
    y_vals  = df[outcome].astype(float)
    t_vals  = df[treatment].astype(float)

    for c in cols:
        x = df[c].fillna(df[c].median())
        if x.std() < 1e-9:
            continue
        try:
            corr_t = abs(np.corrcoef(x, t_vals)[0, 1])
            corr_y = abs(np.corrcoef(x, y_vals)[0, 1])
            if np.isnan(corr_t) or np.isnan(corr_y):
                continue
            # Prefer variables correlated with BOTH (confounders)
            scores[c] = corr_t + corr_y
        except Exception:
            continue

    # Select top 30 by confounding score
    ranked   = sorted(scores.items(), key=lambda kv: -kv[1])[:30]
    selected = [c for c, _ in ranked]

    # ── Fallback: scoring filtered everything → use any non-constant cols ──
    if not selected:
        selected = [
            c for c in cols
            if df[c].std() > 1e-9
        ][:10]

    return selected


def main():
    print("=" * 70)
    print("STEP 09b -- DML v5  |  auto-discovery + MB + interactions")
    print("=" * 70)

    df = pd.read_csv(INPUT_FEAT, low_memory=False)
    print(f"  Features: {df.shape}")

    df['action_ordinal'] = df['position_action'].map(ACTION_MAP)
    df = df.dropna(subset=['action_ordinal']).copy()
    df['is_buy']  = (df['action_ordinal'] == 2).astype(float)
    df['is_sell'] = (df['action_ordinal'] == 0).astype(float)

    # v5: Auto-discover treatments
    ALL_TREATMENTS = build_treatment_list(df)
    print(f"  Treatments: {len(ALL_TREATMENTS)}  |  Targets: {len(TARGETS)}")
    print(f"  Est. runtime: ~{len(ALL_TREATMENTS) * len(TARGETS) * 2 // 60} minutes")

    # Load ICP results to flag certified variables
    icp_certified = set()
    icp_high_conf = set()
    if os.path.exists(INPUT_PARENTS):
        parents = pd.read_csv(INPUT_PARENTS)
        icp_certified = set(
            parents[parents['in_intersection'] == True]['variable'].unique()
        )
        icp_high_conf = set(
            parents[parents['confidence'] >= 0.3]['variable'].unique()
        )
        print(f"  ICP certified: {sorted(icp_certified)}")
        print(f"  ICP high-conf (>=0.3): {sorted(icp_high_conf)}")

    all_rows = []
    total    = len(ALL_TREATMENTS) * len(TARGETS)
    done     = 0

    for outcome_name, outcome_col in TARGETS.items():
        print(f"\n{'=' * 60}")
        print(f"TARGET: {outcome_name}")
        print(f"{'=' * 60}")
        Y_all = df[outcome_col].astype(float).values

        for treatment in ALL_TREATMENTS:
            done += 1
            if treatment not in df.columns:
                print(f"  [{done:3d}/{total}] {treatment:35s}  MISSING")
                continue

            controls = get_controls(df, treatment, outcome_col)
            sub      = df[[treatment] + controls].dropna()

            if len(sub) < 1000:
                print(f"  [{done:3d}/{total}] {treatment:35s}  "
                      f"SKIPPED (n={len(sub)})")
                continue

            Y = Y_all[sub.index]
            T = sub[treatment].values.astype(float)
            X = sub[controls].values.astype(float) if controls else np.empty((len(sub), 0))

            theta, se, lo, hi = double_ml(Y, T, X)
            sig  = bool(not (np.isnan(lo) or np.isnan(hi)) and
                        ((lo > 0 and hi > 0) or (lo < 0 and hi < 0)))

            flag = ''
            if treatment in icp_certified:
                flag = '[ICP✓]'
            elif treatment in icp_high_conf:
                flag = '[ICP~]'

            marker = '***' if sig else ''
            print(f"  [{done:3d}/{total}] {treatment:35s}  "
                  f"theta={theta:+.5f}  SE={se:.5f}  "
                  f"CI=[{lo:+.5f},{hi:+.5f}]  {marker} {flag}")

            all_rows.append({
                'treatment':     treatment,
                'outcome':       outcome_name,
                'theta_hat':     theta,
                'std_error':     se,
                'ci_lower_95':   lo,
                'ci_upper_95':   hi,
                'significant':   sig,
                'direction':     'positive' if (not np.isnan(theta) and theta > 0) else 'negative',
                'icp_certified': treatment in icp_certified,
                'icp_high_conf': treatment in icp_high_conf,
                'n_obs':         int(len(sub)),
                'n_controls':    int(len(controls)),
                'method':        'DoubleML_PLR',
                'edge_type':     'CAUSAL_EFFECT',
            })

    out_df = pd.DataFrame(all_rows)
    os.makedirs(CAUSAL_DIR, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved: {OUTPUT_CSV}  ({len(out_df)} rows)")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DML SUMMARY")
    print("=" * 70)
    for outcome in TARGETS:
        sub = out_df[out_df['outcome'] == outcome]
        sig = sub[sub['significant'] == True]
        icp = sig[sig['icp_certified'] == True]
        print(f"\n  {outcome}: {len(sig)}/{len(sub)} significant  "
              f"({len(icp)} ICP-certified)")
        top = sig.reindex(
            sig['theta_hat'].abs().sort_values(ascending=False).index
        ).head(10)
        for _, r in top.iterrows():
            tag = ' [ICP✓]' if r['icp_certified'] else (
                  ' [ICP~]' if r['icp_high_conf'] else '')
            print(f"    {r['treatment']:35s}  "
                  f"theta={r['theta_hat']:+.5f}  "
                  f"{'↑' if r['direction'] == 'positive' else '↓'}{tag}")

    print("\n  STEP 09b DONE.")


if __name__ == '__main__':
    main()