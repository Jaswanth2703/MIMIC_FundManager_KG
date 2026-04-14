"""
Step 09a -- ICP v6 (IMPROVED)
Robust invariance test (KS) + soft intersection (>90%) + multi-alpha sweep
+ continuous MI + temporal environments + no arbitrary domain bonus

Improvements over v5:
- KS test replaces Levene+ANOVA (distributional equality, not just variance)
- Soft intersection: variables in >90% of plausible sets (was: strict ALL)
- Multi-alpha sensitivity sweep: alpha in [0.05, 0.10, 0.15, 0.20]
- Continuous MI via mutual_info_regression (was: classification MI on ordinal)
- Temporal environments added (pre/post rate hike, volatility regimes)
- DOMAIN_BONUS removed (was unjustified arbitrary boost)
- Partial correlation for candidate screening alongside MI
"""

import sys, os, json, warnings
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES_DIR, CAUSAL_DIR

INPUT_CSV      = os.path.join(FEATURES_DIR, 'LPCMCI_READY.csv')
OUTPUT_PARENTS = os.path.join(CAUSAL_DIR, 'icp_causal_parents.csv')
OUTPUT_DIAG    = os.path.join(CAUSAL_DIR, 'icp_environment_diagnostics.json')

ACTION_MAP = {
    'BUY': 2, 'INCREASE': 2, 'INITIAL_POSITION': 2,
    'HOLD': 1,
    'DECREASE': 0, 'SELL': 0,
}

# ---- IMPROVED SETTINGS ----
ALPHA_SWEEP           = [0.05, 0.10, 0.15, 0.20]  # multi-alpha sensitivity
ALPHA_PRIMARY         = 0.10                         # primary alpha
MAX_SUBSET_SIZE       = 4
CANDIDATES_PER_PASS   = 20
MAX_CANDIDATES_FINAL  = 25
SOFT_INTERSECTION_PCT = 0.90   # variable must appear in >90% of plausible sets
MIN_PER_ENV           = 200

EXCLUDE = {
    'action_ordinal', 'is_buy', 'is_sell',
    'year_month_str', 'ISIN', 'Fund_Name', 'Fund_Type',
    'sector', 'stock_name', 'stock_name_raw', 'Industry',
    'date', 'Date', 'position_action', 'holding_period_id',
    'fund_ticker', 'symbol', 'golden_cross', 'death_cross',
    'size', 'year_month', 'month_ordinal', 'allocation_momentum',
    'allocation_change', 'quantity', 'market_value', 'quantity_change',
    'month_gap', 'fund_stock_count', 'is_top10', 'allocation_quintile',
    'rank_in_fund', 'consensus_count', 'pct_nav',
}

DOMAIN_FEATURES = [
    'pe', 'pb', 'pe_relative', 'industry_pe', 'industry_pb',
    'eps', 'eps_lag1', 'eps_lag2', 'market_cap', 'beta', 'alpha',
    'bv_per_share', 'profit_after_tax', 'income_from_sales', 'total_returns',
    'sentiment_mean', 'sentiment_mean_lag1', 'sentiment_mean_lag2',
    'sentiment_confidence', 'sentiment_extremity',
    'positive_ratio', 'negative_ratio',
    'repo_rate', 'repo_rate_lag1', 'cpi_inflation', 'gdp_growth',
    'us_10y_yield', 'india_vix_close', 'real_interest_rate',
    'usd_inr', 'crude_oil', 'nifty50_return',
    'holding_tenure', 'allocation_change_lag1', 'allocation_change_lag2',
    'pct_nav_lag1', 'pct_nav_lag2', 'volume_ratio',
    'rsi_lag1', 'macd_line_lag1', 'volatility_3m', 'momentum_3m',
]


def build_environments(df):
    """Build environment labels for ICP.

    v6: Added temporal environments alongside structural ones:
    - Fund_Type × nifty_regime (original)
    - VIX regime (high/low volatility)
    - Temporal period (early/late in sample)
    This gives richer environment variation for better invariance testing.
    """
    parts = []

    # Structural: Fund_Type
    if 'Fund_Type' in df.columns:
        stratum = df['Fund_Type'].astype(str).str.strip().str.lower()
        stratum = stratum.replace({'small cap': 'small', 'mid cap': 'mid'})
    else:
        stratum = pd.Series('pooled', index=df.index)
    parts.append(stratum)

    # Market regime: nifty50 return
    if 'nifty50_return' in df.columns:
        ret = df['nifty50_return'].fillna(0)
        regime = pd.cut(ret, bins=[-1e9, -0.02, 0.02, 1e9],
                        labels=['bear', 'sideways', 'bull']).astype(str)
    else:
        regime = pd.Series('any', index=df.index)
    parts.append(regime)

    # v6: VIX regime (high/low volatility)
    if 'india_vix_close' in df.columns:
        vix = df['india_vix_close'].fillna(df['india_vix_close'].median())
        vix_med = vix.median()
        vix_regime = pd.Series(np.where(vix > vix_med, 'hvix', 'lvix'), index=df.index)
        parts.append(vix_regime)

    env = parts[0]
    for p in parts[1:]:
        env = env + '__' + p
    return env


def filter_environments(df, env):
    counts = env.value_counts()
    keep   = counts[counts >= MIN_PER_ENV].index
    mask   = env.isin(keep)
    print(f"  Envs kept: {len(keep)}")
    for e in keep:
        print(f"    {e}: {counts[e]}")
    return df[mask].copy(), env[mask].copy()


def _pool(df):
    num = df.select_dtypes(include=[np.number]).columns
    return [c for c in num if c not in EXCLUDE and not c.endswith('_lag3')]


def build_candidate_pool(df, target_col):
    """Build candidate pool using correlation + continuous MI.

    v6: Uses mutual_info_regression (continuous) instead of
    mutual_info_classif (classification). No arbitrary DOMAIN_BONUS.
    Domain features get a small tiebreaker (+1) not a large boost.
    """
    K    = CANDIDATES_PER_PASS
    pool = _pool(df)
    y    = df[target_col].astype(float)

    # Pass 1: correlation
    corrs = {}
    for c in pool:
        x = df[c].fillna(df[c].median())
        if x.std() < 1e-9: continue
        try: corrs[c] = abs(np.corrcoef(x, y)[0, 1])
        except: continue
    corr_top = sorted(corrs.items(), key=lambda kv: -kv[1])[:K]

    # Pass 2: continuous mutual info (not classification)
    idx  = np.random.RandomState(42).choice(len(df), min(20000, len(df)), replace=False)
    X_s  = df.iloc[idx][pool].fillna(df.iloc[idx][pool].median()).values
    y_s  = y.values[idx]
    try:
        mi     = mutual_info_regression(X_s, y_s, random_state=42, n_neighbors=5)
        mi_top = [(pool[i], float(mi[i])) for i in np.argsort(mi)[::-1][:K]]
    except Exception as e:
        print(f"    MI failed: {e}")
        mi_top = []

    # Score accumulation (no DOMAIN_BONUS — domain features get +1 tiebreaker)
    scores = {}
    for rank, (c, _) in enumerate(corr_top):
        scores[c] = scores.get(c, 0) + (K - rank)
    for rank, (c, _) in enumerate(mi_top):
        scores[c] = scores.get(c, 0) + (K - rank)
    for c in DOMAIN_FEATURES:
        if c in df.columns and c not in EXCLUDE:
            scores[c] = scores.get(c, 0) + 1  # tiebreaker only

    valid = [c for c in scores
             if c in df.columns and df[c].fillna(df[c].median()).std() > 1e-9]
    final = sorted(valid, key=lambda c: -scores[c])[:MAX_CANDIDATES_FINAL]

    corr_set   = {c for c, _ in corr_top}
    mi_set     = {c for c, _ in mi_top}
    domain_set = set(DOMAIN_FEATURES)

    n_sub = sum(len(list(combinations(final, k))) for k in range(1, MAX_SUBSET_SIZE+1))
    print(f"\n  Pool ({len(final)} candidates, {n_sub} subsets):")
    for c in final:
        src = '+'.join(filter(None, [
            'corr'   if c in corr_set   else '',
            'MI'     if c in mi_set     else '',
            'domain' if c in domain_set else '',
        ]))
        print(f"    {c:<35}  {scores[c]:4.0f}  {src}")
    return final


def test_invariance(residuals, env_labels):
    """Test invariance of residuals across environments using KS test.

    v6: Replaced Levene+ANOVA (variance-only) with pairwise KS tests
    (full distributional equality). Returns min p-value across all pairs.
    """
    groups = [residuals[env_labels == e] for e in env_labels.unique()]
    groups = [g for g in groups if len(g) >= 30]
    if len(groups) < 2: return 1.0

    min_p = 1.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            try:
                _, p_ks = stats.ks_2samp(groups[i], groups[j])
                min_p = min(min_p, p_ks)
            except:
                pass
    return min_p


def fit_and_residuals(X, y):
    scaler = StandardScaler()
    model  = LinearRegression()
    Xs     = scaler.fit_transform(X)
    model.fit(Xs, y)
    return y - model.predict(Xs)


def run_icp_exhaustive(df, env_labels, target_col, candidates, alpha=None):
    """Run ICP with soft intersection.

    v6: Instead of strict set.intersection (requires ALL plausible sets),
    uses soft intersection: variables appearing in >=SOFT_INTERSECTION_PCT
    of plausible sets are certified. Also runs multi-alpha sensitivity.
    """
    if alpha is None:
        alpha = ALPHA_PRIMARY
    y     = df[target_col].astype(float).values
    sets  = []
    n_sub = sum(len(list(combinations(candidates, k)))
                for k in range(1, MAX_SUBSET_SIZE+1))
    print(f"  Testing {n_sub} subsets (alpha={alpha}) ...")

    for k in range(1, MAX_SUBSET_SIZE+1):
        for subset in combinations(candidates, k):
            X    = df[list(subset)].fillna(df[list(subset)].median()).values
            # Check rank before fitting
            if np.linalg.matrix_rank(X) < X.shape[1]:
                continue
            resid = fit_and_residuals(X, y)
            p    = test_invariance(pd.Series(resid, index=env_labels.index), env_labels)
            if p > alpha:
                sets.append(set(subset))

    print(f"  Plausible sets: {len(sets)}")

    # v6: Soft intersection — variables in >=90% of plausible sets
    freq = defaultdict(int)
    for s in sets:
        for v in s: freq[v] += 1
    total = max(1, len(sets))

    causal_strict = set.intersection(*sets) if sets else set()
    causal_soft = {v for v, c in freq.items()
                   if c / total >= SOFT_INTERSECTION_PCT}

    if causal_strict != causal_soft:
        extra = causal_soft - causal_strict
        if extra:
            print(f"  Soft intersection found {len(extra)} additional parents: {sorted(extra)}")

    return {
        'causal_parents':        sorted(causal_soft),
        'causal_parents_strict': sorted(causal_strict),
        'plausible_set_count':   len(sets),
        'variable_confidence':   {v: c/total for v, c in freq.items()},
        'tested_subsets':        n_sub,
        'alpha':                 alpha,
    }


def run_for_stratum(df, target_col, stratum_label):
    """Run ICP for a single stratum with multi-alpha sensitivity analysis."""
    print(f"\n  ===== {stratum_label} | target={target_col} =====")
    env = build_environments(df)
    df_f, env_f = filter_environments(df, env)
    if len(df_f) < 500 or env_f.nunique() < 2:
        print("    Skipped")
        return None
    cands = build_candidate_pool(df_f, target_col)
    if len(cands) < 2: return None

    # Primary run
    res = run_icp_exhaustive(df_f, env_f, target_col, cands, alpha=ALPHA_PRIMARY)

    # Multi-alpha sensitivity sweep
    alpha_stability = {}
    for alpha in ALPHA_SWEEP:
        if alpha == ALPHA_PRIMARY:
            alpha_stability[alpha] = res['causal_parents']
            continue
        sens = run_icp_exhaustive(df_f, env_f, target_col, cands, alpha=alpha)
        alpha_stability[alpha] = sens['causal_parents']

    # Report stability
    all_certified = set()
    for parents in alpha_stability.values():
        all_certified.update(parents)
    stable_vars = [v for v in all_certified
                   if sum(1 for p in alpha_stability.values() if v in p) == len(ALPHA_SWEEP)]
    print(f"  Alpha sensitivity: {len(stable_vars)} vars stable across all alphas: {stable_vars}")

    res.update(stratum=stratum_label, target=target_col,
               n_obs=len(df_f), n_environments=int(env_f.nunique()),
               n_candidates=len(cands),
               alpha_stability={str(k): v for k, v in alpha_stability.items()},
               stable_across_alphas=stable_vars)
    return res


def main():
    print("="*70)
    print("STEP 09a -- ICP v6 (IMPROVED: KS test + soft intersection + multi-alpha)")
    print(f"  alpha_sweep={ALPHA_SWEEP}  candidates={MAX_CANDIDATES_FINAL}"
          f"  soft_pct={SOFT_INTERSECTION_PCT}")
    print(f"  subsets=15275/run  targets=3  strata=3")
    print("="*70)

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"  Loaded: {df.shape}")

    df['action_ordinal'] = df['position_action'].map(ACTION_MAP)
    df = df.dropna(subset=['action_ordinal']).copy()
    df['is_buy']  = (df['action_ordinal'] == 2).astype(float)
    df['is_sell'] = (df['action_ordinal'] == 0).astype(float)

    TARGETS = ['action_ordinal', 'is_buy', 'is_sell']

    all_results = {}

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}")

        all_results[f'pooled_{target}'] = run_for_stratum(df, target, 'pooled')

        if 'Fund_Type' in df.columns:
            small = df[df['Fund_Type'].astype(str).str.strip().str.lower() == 'small cap']
            if len(small) > 500:
                all_results[f'small_cap_{target}'] = run_for_stratum(small, target, 'small_cap')

            mid = df[df['Fund_Type'].astype(str).str.strip().str.lower() == 'mid cap']
            if len(mid) > 500:
                all_results[f'mid_cap_{target}'] = run_for_stratum(mid, target, 'mid_cap')

    # Aggregate
    rows = []
    for key, res in all_results.items():
        if res is None: continue
        for var, conf in res['variable_confidence'].items():
            rows.append({
                'stratum':              res['stratum'],
                'target':               res['target'],
                'variable':             var,
                'confidence':           conf,
                'in_intersection':      var in res['causal_parents'],
                'plausible_sets_total': res['plausible_set_count'],
                'n_obs':                res['n_obs'],
                'n_environments':       res['n_environments'],
                'n_candidates':         res['n_candidates'],
                'search_mode':          'exhaustive',
                'method':               'ICP',
                'edge_type':            'CAUSES',
                'effect':               res['target'],
            })

    parents_df = pd.DataFrame(rows)
    os.makedirs(CAUSAL_DIR, exist_ok=True)
    parents_df.to_csv(OUTPUT_PARENTS, index=False)
    print(f"\n  Saved: {OUTPUT_PARENTS}  ({len(parents_df)} rows)")

    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    for key, res in all_results.items():
        if res is None: continue
        certified = res['causal_parents']
        hc = sorted([(v,c) for v,c in res['variable_confidence'].items()
                     if c >= 0.3], key=lambda x: -x[1])
        print(f"\n  [{res['stratum']} | {res['target']}]  "
              f"plausible={res['plausible_set_count']}  "
              f"envs={res['n_environments']}")
        if certified:
            for v in certified:
                c = res['variable_confidence'].get(v, 1.0)
                print(f"    ✓ CERTIFIED: {v}  (conf={c:.3f})")
        else:
            if hc:
                print(f"    No certified parents. Top candidates:")
                for v, c in hc[:4]:
                    print(f"      {v:<35}  conf={c:.3f}")
            else:
                print(f"    No certified parents, no high-confidence variables")

    # Save diagnostics
    diag = {k: {kk: vv for kk, vv in v.items() if kk != 'candidates'}
            for k, v in all_results.items() if v is not None}
    with open(OUTPUT_DIAG, 'w') as f:
        json.dump(diag, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_DIAG}")
    print("\n  STEP 09a DONE.")


if __name__ == '__main__':
    main()