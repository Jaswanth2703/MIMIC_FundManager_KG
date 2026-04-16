"""
Step 09a -- ICP v7 (Markov Blanket + Rich Environments)
=======================================================
Proper Markov Blanket discovery + rich environments + expanded candidates

v7 improvements over v6:
- Grow-Shrink Markov Blanket: discovers parents + children + co-parents
  using conditional independence tests (partial correlation)
- Rich environments: quarter-based temporal + sector + momentum regime
  (15-25 environments instead of 2-4)
- Expanded candidate pool: MB variables + interactions + domain (50 max)
- Soft intersection threshold relaxed to 80% for better discovery
- ICP now has statistical power to find invariant causal parents

Reference:
  Peters et al. (2016). Causal inference using invariant prediction.
  JRSS-B, 78(5), 947-1012.
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
OUTPUT_MB      = os.path.join(CAUSAL_DIR, 'markov_blanket.json')

ACTION_MAP = {
    'BUY': 2, 'INCREASE': 2, 'INITIAL_POSITION': 2,
    'HOLD': 1,
    'DECREASE': 0, 'SELL': 0,
}

# ---- IMPROVED SETTINGS (v7) ----
ALPHA_SWEEP           = [0.05, 0.10, 0.15, 0.20]
ALPHA_PRIMARY         = 0.10
MAX_SUBSET_SIZE       = 3       # v7.1: was 4 → C(30,3)=4060 vs C(50,4)=230K
CANDIDATES_PER_PASS   = 25
MAX_CANDIDATES_FINAL  = 30      # v7.1: was 50 → prevents combinatorial explosion
SOFT_INTERSECTION_PCT = 0.80
MIN_PER_ENV           = 150

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
    # v7: interaction features
    'pe_x_momentum', 'rsi_x_momentum', 'vix_x_return',
    'sentiment_x_momentum', 'alloc_x_sentiment',
    # v7: non-linear transforms
    'log_market_cap', 'log_pe', 'sqrt_volatility', 'log_volume_ratio',
    # v7: key raw features
    'monthly_return', 'monthly_return_lag1', 'monthly_return_lag2',
]


def build_environments(df):
    """Build rich environment labels for ICP.

    v7: Redesigned for 15-25 diverse environments (was 2-4).
    Uses orthogonal dimensions that create genuine distributional shifts:
    - Quarter-of-year (Q1-Q4): seasonal patterns in fund behavior
    - Market momentum regime (bull/bear/sideways): different return environments
    - Temporal half (early/late): structural change over time

    This gives ~24 potential environments (4 × 3 × 2), filtered to those
    with >=MIN_PER_ENV observations.
    """
    parts = []

    # Dimension 1: Quarter of year (Q1-Q4) — seasonal behavior
    if 'year_month_str' in df.columns:
        months = pd.to_datetime(df['year_month_str'], errors='coerce')
        quarter = months.dt.quarter.fillna(1).astype(int)
        quarter_label = quarter.map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
        parts.append(quarter_label)
    else:
        parts.append(pd.Series('qX', index=df.index))

    # Dimension 2: Market momentum regime (return-based)
    if 'nifty50_return' in df.columns:
        ret = df['nifty50_return'].fillna(0)
        regime = pd.cut(ret, bins=[-1e9, -0.02, 0.02, 1e9],
                        labels=['bear', 'flat', 'bull']).astype(str)
    else:
        regime = pd.Series('any', index=df.index)
    parts.append(regime)

    # Dimension 3: Temporal half (structural change)
    if 'year_month_str' in df.columns:
        all_months = sorted(df['year_month_str'].dropna().unique())
        mid = all_months[len(all_months) // 2] if all_months else '2024-01'
        temporal = pd.Series(
            np.where(df['year_month_str'] <= mid, 'early', 'late'),
            index=df.index
        )
    else:
        temporal = pd.Series('any', index=df.index)
    parts.append(temporal)

    env = parts[0].astype(str)
    for p in parts[1:]:
        env = env + '__' + p.astype(str)
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


# ============================================================
# Grow-Shrink Markov Blanket Discovery
# ============================================================
def _partial_corr(x, y, Z, df):
    """Partial correlation between x and y given Z using OLS residuals."""
    if len(Z) == 0:
        vals = df[[x, y]].dropna()
        if len(vals) < 30 or vals[x].std() < 1e-9 or vals[y].std() < 1e-9:
            return 0.0, 1.0
        r, p = stats.pearsonr(vals[x], vals[y])
        return r, p

    subset = df[[x, y] + list(Z)].dropna()
    if len(subset) < 30 + len(Z):
        return 0.0, 1.0

    try:
        Zmat = subset[list(Z)].values
        x_resid = subset[x].values - LinearRegression().fit(Zmat, subset[x].values).predict(Zmat)
        y_resid = subset[y].values - LinearRegression().fit(Zmat, subset[y].values).predict(Zmat)
        if np.std(x_resid) < 1e-9 or np.std(y_resid) < 1e-9:
            return 0.0, 1.0
        r, p = stats.pearsonr(x_resid, y_resid)
        return r, p
    except Exception:
        return 0.0, 1.0


def grow_shrink_markov_blanket(df, target_col, candidates, alpha_mb=0.01,
                                max_mb_size=40):
    """Discover Markov Blanket using Grow-Shrink algorithm.

    The MB(Y) = Parents(Y) ∪ Children(Y) ∪ Co-Parents(Y).

    Grow phase: Add variables that are NOT conditionally independent of Y
    given the current blanket (partial correlation test).

    Shrink phase: Remove variables that become conditionally independent
    of Y given the rest of the blanket.

    Reference: Margaritis & Thrun (2000). Bayesian Network Induction via
    Local Neighborhoods.
    """
    print(f"\n  === MARKOV BLANKET DISCOVERY (Grow-Shrink) ===")
    print(f"    Target: {target_col}  |  Candidates: {len(candidates)}  |  alpha={alpha_mb}")

    # Subsample for speed (partial corr is O(n) per test)
    n = len(df)
    if n > 30000:
        idx = np.random.RandomState(42).choice(n, 30000, replace=False)
        df_sub = df.iloc[idx].copy()
    else:
        df_sub = df.copy()

    mb = []
    # GROW: add variables that are dependent on target given current MB
    for c in candidates:
        if c == target_col or c not in df_sub.columns:
            continue
        if df_sub[c].std() < 1e-9:
            continue
        r, p = _partial_corr(c, target_col, mb, df_sub)
        if p < alpha_mb and abs(r) > 0.01:
            mb.append(c)
            if len(mb) >= max_mb_size:
                break

    print(f"    After GROW: {len(mb)} variables")

    # SHRINK: remove variables that are conditionally independent given rest
    to_remove = []
    for c in mb:
        rest = [v for v in mb if v != c]
        r, p = _partial_corr(c, target_col, rest, df_sub)
        if p >= alpha_mb or abs(r) < 0.005:
            to_remove.append(c)

    for c in to_remove:
        mb.remove(c)

    print(f"    After SHRINK: {len(mb)} variables")
    print(f"    MB: {mb[:20]}{'...' if len(mb) > 20 else ''}")

    return mb


def build_candidate_pool(df, target_col, mb_vars=None):
    """Build candidate pool using MB + correlation + MI.

    v7: MB variables get priority (+15 score boost), ensuring they're
    always included in the candidate pool. Then correlation and MI
    fill remaining slots.
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

    # Pass 2: continuous mutual info
    idx  = np.random.RandomState(42).choice(len(df), min(20000, len(df)), replace=False)
    X_s  = df.iloc[idx][pool].fillna(df.iloc[idx][pool].median()).values
    y_s  = y.values[idx]
    try:
        mi     = mutual_info_regression(X_s, y_s, random_state=42, n_neighbors=5)
        mi_top = [(pool[i], float(mi[i])) for i in np.argsort(mi)[::-1][:K]]
    except Exception as e:
        print(f"    MI failed: {e}")
        mi_top = []

    # Score accumulation
    scores = {}
    for rank, (c, _) in enumerate(corr_top):
        scores[c] = scores.get(c, 0) + (K - rank)
    for rank, (c, _) in enumerate(mi_top):
        scores[c] = scores.get(c, 0) + (K - rank)

    # v7: MB variables get priority boost
    mb_set = set(mb_vars or [])
    for c in mb_set:
        if c in df.columns and c not in EXCLUDE:
            scores[c] = scores.get(c, 0) + 15  # strong boost

    # Domain features get tiebreaker
    for c in DOMAIN_FEATURES:
        if c in df.columns and c not in EXCLUDE:
            scores[c] = scores.get(c, 0) + 1

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

    v7.1: Added progress printing + reduced subset size for speed.
    """
    if alpha is None:
        alpha = ALPHA_PRIMARY
    y     = df[target_col].astype(float).values
    sets  = []
    n_sub = sum(len(list(combinations(candidates, k)))
                for k in range(1, MAX_SUBSET_SIZE+1))
    print(f"  Testing {n_sub} subsets (alpha={alpha}) ...", end=' ', flush=True)

    tested = 0
    for k in range(1, MAX_SUBSET_SIZE+1):
        for subset in combinations(candidates, k):
            tested += 1
            if tested % 1000 == 0:
                print(f"{tested}/{n_sub}", end=' ', flush=True)
            X    = df[list(subset)].fillna(df[list(subset)].median()).values
            if np.linalg.matrix_rank(X) < X.shape[1]:
                continue
            resid = fit_and_residuals(X, y)
            p    = test_invariance(pd.Series(resid, index=env_labels.index), env_labels)
            if p > alpha:
                sets.append(set(subset))

    print(f"\n  Plausible sets: {len(sets)}")

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


def run_for_stratum(df, target_col, stratum_label, mb_vars=None):
    """Run ICP for a single stratum with multi-alpha sensitivity analysis."""
    print(f"\n  ===== {stratum_label} | target={target_col} =====")
    env = build_environments(df)
    df_f, env_f = filter_environments(df, env)
    if len(df_f) < 500 or env_f.nunique() < 2:
        print("    Skipped (insufficient data or environments)")
        return None
    cands = build_candidate_pool(df_f, target_col, mb_vars=mb_vars)
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
    print("STEP 09a -- ICP v7 (Markov Blanket + Rich Environments)")
    print(f"  alpha_sweep={ALPHA_SWEEP}  candidates={MAX_CANDIDATES_FINAL}"
          f"  soft_pct={SOFT_INTERSECTION_PCT}")
    print("="*70)

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"  Loaded: {df.shape}")

    df['action_ordinal'] = df['position_action'].map(ACTION_MAP)
    df = df.dropna(subset=['action_ordinal']).copy()
    df['is_buy']  = (df['action_ordinal'] == 2).astype(float)
    df['is_sell'] = (df['action_ordinal'] == 0).astype(float)

    TARGETS = ['action_ordinal', 'is_buy', 'is_sell']

    # v7: Compute Markov Blanket FIRST for each target
    pool = _pool(df)
    print(f"\n  Feature pool: {len(pool)} numeric columns")

    mb_results = {}
    for target in TARGETS:
        mb = grow_shrink_markov_blanket(df, target, pool)
        mb_results[target] = mb

    # Save MB results
    os.makedirs(CAUSAL_DIR, exist_ok=True)
    with open(OUTPUT_MB, 'w') as f:
        json.dump(mb_results, f, indent=2)
    print(f"\n  Saved MB: {OUTPUT_MB}")

    all_results = {}

    for target in TARGETS:
        print(f"\n{'='*60}")
        print(f"TARGET: {target}")
        print(f"{'='*60}")

        mb_vars = mb_results[target]

        # Pooled (all data)
        all_results[f'pooled_{target}'] = run_for_stratum(
            df, target, 'pooled', mb_vars=mb_vars)

        # Strata by Fund_Type
        if 'Fund_Type' in df.columns:
            for ft_name, ft_filter in [('small_cap', 'small cap'),
                                        ('mid_cap', 'mid cap'),
                                        ('large_cap', 'large cap')]:
                sub = df[df['Fund_Type'].astype(str).str.strip().str.lower() == ft_filter]
                if len(sub) > 500:
                    all_results[f'{ft_name}_{target}'] = run_for_stratum(
                        sub, target, ft_name, mb_vars=mb_vars)

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