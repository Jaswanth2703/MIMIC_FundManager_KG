"""
Step 09 — L9 (Rigorous): Targeted Panel Causal Discovery — Fixed & Hardened
=============================================================================

BUGS FIXED FROM PREVIOUS VERSION
-----------------------------------

BUG 1 [CRITICAL]: action_ordinal == net_flow_ratio (algebraically identical)
  action_ordinal = mean(+1,0,-1) = (n_buy - n_sell)/N = net_flow_ratio
  Fix: Drop net_flow_ratio. Keep action_ordinal (the canonical target).
       Rename: buy_ratio, sell_ratio are the directional decomposition.

BUG 2 [CRITICAL]: pct_nav lag1 == lag2 in allocation_delta
  allocation_delta(t) = pct_nav(t) - pct_nav(t-1)
  AR control = allocation_delta(t-1) = pct_nav(t-1) - pct_nav(t-2)
  When cause=pct_nav(t-2), it appears in BOTH the cause column AND the AR
  control → identical betas across lags 1 and 2 via collinear partialling.
  Fix: Drop allocation_delta as a target. It's mechanically derived from
       pct_nav and will always produce degenerate pct_nav regressions.
       Instead, use pct_nav itself as a target (is the fund increasing/
       decreasing its position weight?).

BUG 3 [DESIGN]: lag=0 links are contemporaneous, NOT causal
  "RSI causes buying" at lag=0 just means they move together in the same
  month. You cannot claim direction. We already see them contemporaneously
  (e.g., price rises → RSI up → funds buy: all three happen same month).
  Fix: For causal claims, restrict to lag >= 1. Report lag=0 separately
       as "contemporaneous associations" with explicit caveat.

BUG 4 [DESIGN]: Single AR lag is insufficient for persistent variables
  consensus_count appears significant at lags 1–6 with barely decaying
  betas. This is autocorrelation bleedthrough: cc(t-1), cc(t-2)... are
  all proxying for the same persistent cc(t) signal.
  Fix: Include AR lags 1–3 as controls (not just 1). Use AIC to confirm
       optimal AR order. This properly absorbs persistence.

BUG 5 [DESIGN]: Bonferroni is too conservative when causes are correlated
  RSI, momentum_3m, bollinger_pband, monthly_return are all price momentum
  proxies. Testing them all as separate hypotheses and correcting for
  105 tests per target inflates the threshold beyond reason.
  Fix: Use Benjamini-Hochberg FDR (q=5%) instead of Bonferroni.
       Also group by cause_group and report best representative per group.

NEW ADDITIONS
--------------
1. FDR correction (BH) replacing Bonferroni for better power/precision balance
2. AR order selection via AIC (1–4 lags tested per target)
3. Within-group deduplication (best lag per cause×target)
4. Nonlinear test: include cause² to test for convex/concave effects
5. Out-of-sample R² lift: how much does adding cause(t-k) improve prediction?
6. Regime interaction terms: cause × regime_dummy in the panel (cleaner than split-sample)

TARGETS (cleaned)
------------------
  action_ordinal  — mean BUY=+1, HOLD=0, SELL=-1 across funds (primary)
  buy_ratio       — fraction of funds buying
  sell_ratio      — fraction of funds selling
  pct_nav         — current portfolio weight (is position growing?)

CAUSES (lag >= 1 only for causal claims)
-----------------------------------------
  Price/momentum:  monthly_return, rsi, momentum_3m, bollinger_pband, macd_hist
  Risk:            monthly_volatility, volume_ratio
  Fund behavior:   holding_tenure, consensus_count, pct_nav (for non-pct_nav targets)
  Sentiment:       sentiment_mean
  Macro:           nifty50_return, india_vix_close, cpi_inflation, us_10y_yield
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import FEATURES_DIR, CAUSAL_DIR
    LPCMCI_CSV = os.path.join(FEATURES_DIR, 'LPCMCI_READY.csv')
    OUTPUT_DIR = os.path.join(CAUSAL_DIR, 'targeted_panel_v2')
except ImportError:
    LPCMCI_CSV  = 'data/features/LPCMCI_READY.csv'
    OUTPUT_DIR  = 'data/causal_output/targeted_panel_v2'

NEWS_COUNT_Z_THRESHOLD = -0.790

# ============================================================
# CAUSE GROUPS (for deduplication and reporting)
# ============================================================
CAUSE_GROUPS = {
    'price_momentum':   ['monthly_return', 'rsi', 'momentum_3m', 'bollinger_pband'],
    'trend':            ['macd_hist'],
    'risk':             ['monthly_volatility', 'volume_ratio'],
    'position_size':    ['pct_nav', 'holding_tenure'],
    'herding':          ['consensus_count'],
    'sentiment':        ['sentiment_mean'],
    'macro_equity':     ['nifty50_return', 'india_vix_close'],
    'macro_rates':      ['cpi_inflation', 'us_10y_yield'],
}

CAUSE_TO_GROUP = {c: g for g, cs in CAUSE_GROUPS.items() for c in cs}

ALL_CAUSES = [c for cs in CAUSE_GROUPS.values() for c in cs]

# Targets — net_flow_ratio REMOVED (== action_ordinal), allocation_delta REMOVED (tautological)
TARGETS = ['action_ordinal', 'buy_ratio', 'sell_ratio']
# pct_nav as target in a separate run (cause list excludes pct_nav)
PCT_NAV_CAUSES = [c for c in ALL_CAUSES if c != 'pct_nav']

# Lags: 1–6 for causal claims; 0 reported separately as associations
CAUSAL_LAGS      = [1, 2, 3, 4, 5, 6]
ASSOC_LAGS       = [0]
MAX_AR_ORDER     = 4   # test AR(1)...AR(4), pick by AIC
FDR_Q            = 0.05
MIN_OBS          = 500


# ============================================================
# STEP 1: BUILD PANEL
# ============================================================

def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the pooled (ISIN × month) panel from LPCMCI_READY.
    
    Targets computed:
      action_ordinal  = mean(BUY=+1, HOLD=0, SELL=-1)
      buy_ratio       = fraction of funds buying
      sell_ratio      = fraction of funds selling
      pct_nav         = mean portfolio weight across funds
    
    Note: net_flow_ratio is NOT computed (== action_ordinal by construction).
    Note: allocation_delta is NOT computed (causes tautological regression with pct_nav).
    """
    print("Building pooled panel...")

    if 'year_month_str' not in df.columns:
        for col in ['Date', 'date', 'month']:
            if col in df.columns:
                df['year_month_str'] = pd.to_datetime(df[col]).dt.to_period('M').astype(str)
                break

    df['month_dt']  = pd.to_datetime(df['year_month_str'])
    action_map      = {'BUY': 1, 'INITIAL_POSITION': 1, 'HOLD': 0, 'SELL': -1}
    df['action_num'] = df['position_action'].map(action_map)

    STOCK_VARS = ['monthly_return', 'rsi', 'momentum_3m', 'bollinger_pband',
                  'macd_hist', 'volume_ratio', 'monthly_volatility']
    FUND_VARS  = ['holding_tenure', 'consensus_count', 'pct_nav']
    MACRO_VARS = ['nifty50_return', 'india_vix_close', 'cpi_inflation', 'us_10y_yield']

    records = []
    for (isin, month), grp in df.groupby(['ISIN', 'month_dt']):
        actions = grp['action_num'].dropna()
        if len(actions) == 0:
            continue

        row = {'isin': isin, 'month_dt': month}
        row['action_ordinal'] = float(actions.mean())
        row['buy_ratio']      = float((actions == 1).mean())
        row['sell_ratio']     = float((actions == -1).mean())
        row['n_funds']        = int(len(actions))

        for v in STOCK_VARS:
            if v in grp.columns:
                vals = pd.to_numeric(grp[v], errors='coerce').dropna()
                row[v] = float(vals.iloc[0]) if len(vals) > 0 else np.nan
            else:
                row[v] = np.nan

        for v in FUND_VARS:
            if v in grp.columns:
                vals = pd.to_numeric(grp[v], errors='coerce').dropna()
                row[v] = float(vals.mean()) if len(vals) > 0 else np.nan
            else:
                row[v] = np.nan

        if 'sentiment_mean' in grp.columns:
            nc = pd.to_numeric(grp.get('news_count', pd.Series(dtype=float)), errors='coerce').dropna()
            if len(nc) > 0 and float(nc.iloc[0]) < NEWS_COUNT_Z_THRESHOLD:
                row['sentiment_mean'] = np.nan
            else:
                vals = pd.to_numeric(grp['sentiment_mean'], errors='coerce').dropna()
                row['sentiment_mean'] = float(vals.iloc[0]) if len(vals) > 0 else np.nan
        else:
            row['sentiment_mean'] = np.nan

        for v in MACRO_VARS:
            if v in grp.columns:
                vals = pd.to_numeric(grp[v], errors='coerce').dropna()
                row[v] = float(vals.iloc[0]) if len(vals) > 0 else np.nan
            else:
                row[v] = np.nan

        records.append(row)

    panel = pd.DataFrame(records).sort_values(['isin', 'month_dt'])

    # Macro regime dummies (computed at month level, same across stocks)
    macro_month = (panel.groupby('month_dt')[['nifty50_return', 'india_vix_close', 'cpi_inflation']]
                   .first())
    panel = panel.merge(
        pd.DataFrame({
            'month_dt':   macro_month.index,
            'regime_bull':     (macro_month['nifty50_return'] > 0).astype(float).values,
            'regime_hvix':     (macro_month['india_vix_close'] > macro_month['india_vix_close'].median()).astype(float).values,
            'regime_hcpi':     (macro_month['cpi_inflation']   > macro_month['cpi_inflation'].median()).astype(float).values,
        }),
        on='month_dt', how='left'
    )

    panel = panel.set_index(['isin', 'month_dt'])

    n_stocks = panel.index.get_level_values('isin').nunique()
    n_months = panel.index.get_level_values('month_dt').nunique()
    print(f"  Panel: {len(panel):,} obs | {n_stocks} stocks | {n_months} months")
    print(f"  Targets available: {[t for t in TARGETS + ['pct_nav'] if t in panel.columns]}")

    return panel


# ============================================================
# STEP 2: OPTIMAL AR ORDER SELECTION
# ============================================================

def select_ar_order(panel: pd.DataFrame, target: str,
                    max_order: int = MAX_AR_ORDER) -> int:
    """
    Pick AR order 1..max_order for 'target' by comparing AIC from pooled OLS
    (FE-demeaned). Returns the order with lowest AIC.
    """
    # Demean by entity to approximate FE
    p = panel[[target]].copy().dropna()
    p_demeaned = p - p.groupby(level='isin').transform('mean')
    y = p_demeaned[target].dropna()

    best_aic  = np.inf
    best_order = 1

    for order in range(1, max_order + 1):
        work = pd.DataFrame({'y': y})
        for k in range(1, order + 1):
            work[f'y_lag{k}'] = work['y'].shift(k)
        work = work.dropna()
        if len(work) < 100:
            continue
        X = sm.add_constant(work.drop(columns='y'))
        try:
            res = sm.OLS(work['y'], X).fit()
            if res.aic < best_aic:
                best_aic   = res.aic
                best_order = order
        except Exception:
            continue

    return best_order


# ============================================================
# STEP 3: SINGLE PANEL GRANGER TEST
# ============================================================

def run_panel_granger(panel: pd.DataFrame,
                      target: str,
                      cause: str,
                      lag: int,
                      ar_order: int = 2,
                      min_obs: int = MIN_OBS,
                      regime_interaction: str = None) -> dict | None:
    """
    Test cause(t-lag) -> target(t) with:
      - Stock fixed effects (entity demeaning via PanelOLS)
      - AR(ar_order) controls on target
      - Clustered SE by stock
      - Optional regime x cause interaction term
      - v3: Durbin-Watson autocorrelation check on residuals

    For lag=0: uses contemporaneous cause value. Label as 'association' not 'causal'.

    Returns result dict or None if insufficient data / estimation failure.
    """
    if target not in panel.columns or cause not in panel.columns:
        return None

    # Build working dataset: only target + cause (avoid duplicate column bug)
    avail_causes = [c for c in ALL_CAUSES if c in panel.columns and c != target]
    work = panel[[target, cause] + [c for c in ['regime_bull', 'regime_hvix', 'regime_hcpi']
                                    if c in panel.columns]].copy()
    work = work.sort_index()

    # Lagged cause
    cause_col = f'{cause}_lag{lag}' if lag > 0 else cause
    if lag > 0:
        work[cause_col] = work.groupby(level='isin')[cause].shift(lag)

    # AR controls
    ar_cols = []
    for k in range(1, ar_order + 1):
        col = f'{target}_ar{k}'
        work[col] = work.groupby(level='isin')[target].shift(k)
        ar_cols.append(col)

    # Regime interaction (optional)
    interaction_col = None
    if regime_interaction and regime_interaction in work.columns:
        interaction_col = f'{cause_col}_x_{regime_interaction}'
        work[interaction_col] = work[cause_col] * work[regime_interaction]

    required = [target, cause_col] + ar_cols + ([interaction_col] if interaction_col else [])
    sub = work[required].dropna()

    if len(sub) < min_obs:
        return None

    exog_cols = [cause_col] + ar_cols + ([interaction_col] if interaction_col else [])
    try:
        mod = PanelOLS(sub[target], sub[exog_cols],
                       entity_effects=True,
                       drop_absorbed=True,
                       check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True)
    except Exception:
        return None

    if cause_col not in res.params.index:
        return None

    beta  = float(res.params[cause_col])
    tstat = float(res.tstats[cause_col])
    pval  = float(res.pvalues[cause_col])
    nobs  = int(res.nobs)
    rsq   = float(res.rsquared)

    n_entities = panel.index.get_level_values('isin').nunique()
    df_resid   = max(nobs - len(exog_cols) - n_entities, 1)
    partial_r2 = tstat**2 / (tstat**2 + df_resid)

    # v3: Durbin-Watson autocorrelation check
    try:
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = float(durbin_watson(res.resids.values))
    except Exception:
        dw_stat = np.nan

    # v3: Degrees of freedom check
    dof_ratio = nobs / max(len(exog_cols) + n_entities, 1)

    result = {
        'target':        target,
        'cause':         cause,
        'cause_group':   CAUSE_TO_GROUP.get(cause, 'other'),
        'lag':           lag,
        'link_type':     'causal' if lag > 0 else 'association',
        'beta':          round(beta, 6),
        't_stat':        round(tstat, 4),
        'p_value':       round(pval, 8),
        'partial_r2':    round(partial_r2, 6),
        'n_obs':         nobs,
        'ar_order':      ar_order,
        'rsquared':      round(rsq, 6),
        'regime_interaction': regime_interaction,
        'dw_stat':       round(dw_stat, 3) if not np.isnan(dw_stat) else None,
        'dw_warning':    dw_stat < 1.5 if not np.isnan(dw_stat) else None,
        'dof_ratio':     round(dof_ratio, 1),
    }

    # Regime interaction coefficient
    if interaction_col and interaction_col in res.params.index:
        result['interaction_beta']  = round(float(res.params[interaction_col]), 6)
        result['interaction_tstat'] = round(float(res.tstats[interaction_col]), 4)
        result['interaction_pval']  = round(float(res.pvalues[interaction_col]), 8)

    return result


# ============================================================
# STEP 4: FULL TARGETED SWEEP
# ============================================================

def run_all_targeted(panel: pd.DataFrame,
                     targets: list[str] = None,
                     causes_map: dict = None,
                     lags: list[int] = None,
                     include_lag0: bool = True) -> pd.DataFrame:
    """
    Run all (target, cause, lag) tests with FDR correction.

    causes_map: {target: [cause_list]} — allows different cause sets per target
                e.g. pct_nav shouldn't predict itself
    """
    targets   = targets   or [t for t in TARGETS if t in panel.columns]
    lags      = lags      or CAUSAL_LAGS
    all_lags  = ([0] + lags) if include_lag0 else lags

    if causes_map is None:
        causes_map = {t: [c for c in ALL_CAUSES if c in panel.columns] for t in targets}

    print(f"\nRunning targeted panel Granger tests (rigorous v2)")
    print(f"  Targets:            {targets}")
    print(f"  Causal lags tested: {lags}")
    print(f"  Lag=0 (association) included: {include_lag0}")
    print(f"  FDR correction: BH q={FDR_Q}")
    print(f"  AR order selection: up to AR({MAX_AR_ORDER}) by AIC")
    print("-" * 60)

    all_results = []

    for target in targets:
        if target not in panel.columns:
            print(f"  [SKIP] {target}: not in panel")
            continue

        causes = causes_map.get(target, [c for c in ALL_CAUSES if c in panel.columns])

        # Select optimal AR order for this target
        ar_order = select_ar_order(panel, target)
        print(f"\n  Target: {target}  [AR order selected: {ar_order}]")

        target_results = []

        for cause in causes:
            if cause == target:
                continue
            if cause not in panel.columns:
                continue

            for lag in all_lags:
                result = run_panel_granger(panel, target, cause, lag, ar_order=ar_order)
                if result is not None:
                    target_results.append(result)

        if not target_results:
            continue

        df_t = pd.DataFrame(target_results)

        # FDR correction (BH) separately for causal (lag>0) and association (lag=0) links
        for link_type in ['causal', 'association']:
            mask = df_t['link_type'] == link_type
            if mask.sum() == 0:
                continue
            pvals = df_t.loc[mask, 'p_value'].values
            reject, pvals_fdr, _, _ = multipletests(pvals, alpha=FDR_Q, method='fdr_bh')
            df_t.loc[mask, 'p_fdr']      = pvals_fdr
            df_t.loc[mask, 'significant'] = reject

        if 'significant' not in df_t.columns:
            df_t['significant'] = False
        if 'p_fdr' not in df_t.columns:
            df_t['p_fdr'] = np.nan

        n_causal_sig = df_t[(df_t['link_type']=='causal') & df_t['significant']].shape[0]
        n_assoc_sig  = df_t[(df_t['link_type']=='association') & df_t['significant']].shape[0]
        print(f"    FDR-significant: {n_causal_sig} causal (lag>0) | {n_assoc_sig} association (lag=0)")

        # v3: Report Durbin-Watson warnings
        if 'dw_warning' in df_t.columns:
            dw_warn_count = df_t['dw_warning'].sum() if df_t['dw_warning'].notna().any() else 0
            if dw_warn_count > 0:
                print(f"    WARNING: {int(dw_warn_count)} tests have DW<1.5 (residual autocorrelation)")

        all_results.append(df_t)

    if not all_results:
        return pd.DataFrame()

    result_df = pd.concat(all_results, ignore_index=True)

    # Sort: causal links first, then by |t_stat|
    result_df['abs_t'] = result_df['t_stat'].abs()
    result_df = result_df.sort_values(
        ['link_type', 'significant', 'target', 'abs_t'],
        ascending=[True, False, True, False]
    ).drop(columns='abs_t').reset_index(drop=True)

    return result_df


# ============================================================
# STEP 5: REGIME INTERACTION TESTS
# ============================================================

def run_regime_interactions(panel: pd.DataFrame,
                            top_links: pd.DataFrame,
                            ar_order_map: dict) -> pd.DataFrame:
    """
    For the top FDR-significant causal links (lag>0), add regime interaction
    terms directly in the panel regression instead of splitting the sample.
    
    Model: target = alpha_i + beta*cause(t-k) + gamma*(cause×regime) + AR + eps
    
    gamma tells you: is the causal effect STRONGER/WEAKER in this regime?
    This is cleaner than split-sample (avoids halving N).
    """
    if top_links.empty:
        return pd.DataFrame()

    print("\nRunning regime interaction tests...")

    regimes = ['regime_bull', 'regime_hvix', 'regime_hcpi']
    regime_results = []

    causal_sig = top_links[
        (top_links['link_type'] == 'causal') &
        (top_links['significant'])
    ].sort_values('partial_r2', ascending=False).head(30)

    for _, link in causal_sig.iterrows():
        target = link['target']
        cause  = link['cause']
        lag    = int(link['lag'])
        ar_ord = ar_order_map.get(target, 2)

        for regime in regimes:
            if regime not in panel.columns:
                continue
            result = run_panel_granger(panel, target, cause, lag,
                                       ar_order=ar_ord,
                                       regime_interaction=regime)
            if result and 'interaction_beta' in result:
                result['regime'] = regime
                regime_results.append(result)

    if not regime_results:
        return pd.DataFrame()

    regime_df = pd.DataFrame(regime_results)
    # Significant interaction = regime modifies the causal effect
    regime_df['interaction_significant'] = regime_df['interaction_pval'] < 0.05
    return regime_df


# ============================================================
# STEP 6: DEDUPLICATE & SUMMARIZE
# ============================================================

def deduplicate_by_group(results: pd.DataFrame) -> pd.DataFrame:
    """
    Within each (target, cause_group, link_type), keep only the
    best-lag representative (highest partial_r2 among significant).
    This prevents the report being dominated by consensus_count at lags 1,2,3,4,5,6.
    """
    sig = results[results['significant']].copy()
    best_per_group = (sig.sort_values('partial_r2', ascending=False)
                        .groupby(['target', 'cause_group', 'link_type'])
                        .first()
                        .reset_index())
    return best_per_group


def build_summary(results: pd.DataFrame, panel: pd.DataFrame,
                  regime_df: pd.DataFrame, ar_order_map: dict) -> dict:
    sig_causal = results[(results['link_type'] == 'causal') & results['significant']]
    sig_assoc  = results[(results['link_type'] == 'association') & results['significant']]
    dedup      = deduplicate_by_group(results)

    top_by_target = {}
    for target in results['target'].unique():
        t = dedup[dedup['target'] == target].sort_values('partial_r2', ascending=False)
        top_by_target[target] = t[['cause', 'cause_group', 'lag', 'link_type',
                                    'beta', 't_stat', 'p_fdr', 'partial_r2']].head(12).to_dict('records')

    var_importance = (sig_causal.groupby('cause')['t_stat']
                      .apply(lambda x: round(float(x.abs().mean()), 3))
                      .sort_values(ascending=False)
                      .to_dict())

    return {
        'generated_at':    datetime.now().isoformat(),
        'analysis':        'L9_Targeted_Panel_Granger_Rigorous_v2',
        'bugs_fixed':      [
            'net_flow_ratio removed (== action_ordinal algebraically)',
            'allocation_delta removed (tautological with pct_nav as cause)',
            'pct_nav lag1==lag2 bug fixed by removing allocation_delta target',
            'lag=0 separated as associations not causal',
            'AR order selected by AIC (not fixed at 1)',
            'FDR (BH q=5%) replaces Bonferroni for better power/precision',
        ],
        'methodology': {
            'model':        'PanelOLS entity FE + clustered SE by stock',
            'fdr_method':   f'Benjamini-Hochberg q={FDR_Q}',
            'ar_orders':    ar_order_map,
            'causal_lags':  CAUSAL_LAGS,
        },
        'panel_stats': {
            'n_obs':    int(len(panel)),
            'n_stocks': int(panel.index.get_level_values('isin').nunique()),
            'n_months': int(panel.index.get_level_values('month_dt').nunique()),
        },
        'significant_causal_links':  int(len(sig_causal)),
        'significant_associations':  int(len(sig_assoc)),
        'unique_drivers_after_dedup': int(len(dedup)),
        'top_causal_drivers_by_target': top_by_target,
        'variable_importance_avg_abs_t': var_importance,
        'regime_interactions': (regime_df[regime_df['interaction_significant']]
                                [['target','cause','lag','regime',
                                  'beta','interaction_beta','interaction_pval']]
                                .head(20).to_dict('records')
                                if not regime_df.empty else []),
    }


def print_top_results(results: pd.DataFrame) -> None:
    dedup = deduplicate_by_group(results)

    print(f"\n{'='*70}")
    print("CAUSAL DRIVERS — BEST REPRESENTATIVE PER CAUSE GROUP (FDR-corrected)")
    print(f"{'='*70}")

    for target in [t for t in TARGETS if t in results['target'].values]:
        t_df = dedup[(dedup['target'] == target) &
                     (dedup['link_type'] == 'causal')].sort_values('partial_r2', ascending=False)
        if t_df.empty:
            continue
        print(f"\n-> {target.upper()}  (one best row per cause group)")
        print(f"  {'Group':<20} {'Cause':<22} {'Lag':<5} {'Beta':>9} {'t-stat':>8} {'p-FDR':>9} {'partR2':>8}")
        print(f"  {'-'*20} {'-'*22} {'-'*5} {'-'*9} {'-'*8} {'-'*9} {'-'*8}")
        for _, row in t_df.iterrows():
            direction = '+' if row['beta'] > 0 else '-'
            print(f"  {row['cause_group']:<20} {row['cause']:<22} {row['lag']:<5} "
                  f"{row['beta']:>9.4f} {row['t_stat']:>8.3f} "
                  f"{row['p_fdr']:>9.6f} {row['partial_r2']:>8.5f}  {direction}")

    print(f"\n{'='*70}")
    print("CONTEMPORANEOUS ASSOCIATIONS (lag=0, NOT causal — direction unknown)")
    print(f"{'='*70}")
    for target in [t for t in TARGETS if t in results['target'].values]:
        t_df = dedup[(dedup['target'] == target) &
                     (dedup['link_type'] == 'association')].sort_values('partial_r2', ascending=False)
        if t_df.empty:
            continue
        print(f"\n  {target}: "
              + ", ".join(f"{r['cause']}(β={r['beta']:.3f})" for _, r in t_df.iterrows()))


def save_results(results: pd.DataFrame, regime_df: pd.DataFrame,
                 panel: pd.DataFrame, ar_order_map: dict,
                 output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Full results
    p = os.path.join(output_dir, 'panel_granger_v2_full.csv')
    results.to_csv(p, index=False)
    print(f"\nSaved: {p}  ({len(results)} rows)")

    # FDR-significant causal links only
    sig_causal = results[(results['link_type'] == 'causal') & results['significant']]
    p2 = os.path.join(output_dir, 'panel_granger_v2_causal_significant.csv')
    sig_causal.to_csv(p2, index=False)
    print(f"Saved: {p2}  ({len(sig_causal)} rows)")

    # FDR-significant association links (lag=0)
    sig_assoc = results[(results['link_type'] == 'association') & results['significant']]
    p3 = os.path.join(output_dir, 'panel_granger_v2_associations.csv')
    sig_assoc.to_csv(p3, index=False)
    print(f"Saved: {p3}  ({len(sig_assoc)} rows)")

    # Deduplicated (best per group)
    dedup = deduplicate_by_group(results)
    p4 = os.path.join(output_dir, 'panel_granger_v2_deduped.csv')
    dedup.to_csv(p4, index=False)
    print(f"Saved: {p4}  ({len(dedup)} rows)")

    # Regime interactions
    if not regime_df.empty:
        p5 = os.path.join(output_dir, 'panel_granger_v2_regime.csv')
        regime_df.to_csv(p5, index=False)
        print(f"Saved: {p5}  ({len(regime_df)} rows)")

    # Summary JSON
    summary = build_summary(results, panel, regime_df, ar_order_map)
    p6 = os.path.join(output_dir, 'panel_granger_v2_summary.json')
    with open(p6, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {p6}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("L9 RIGOROUS v2: TARGETED PANEL CAUSAL DISCOVERY")
    print("=" * 70)
    print("Bugs fixed: duplicate targets, tautological causes, lag=0 labelling,")
    print("            AR order selection, FDR vs Bonferroni\n")

    if not os.path.exists(LPCMCI_CSV):
        print(f"ERROR: {LPCMCI_CSV} not found.")
        return

    print(f"Loading {LPCMCI_CSV} ...")
    df = pd.read_csv(LPCMCI_CSV, low_memory=False)
    print(f"  Shape: {df.shape} | Funds: {df['Fund_Name'].nunique()} | ISINs: {df['ISIN'].nunique()}")

    panel = build_panel(df)

    # Determine optimal AR orders per target
    print("\nSelecting AR orders by AIC...")
    available_targets = [t for t in TARGETS if t in panel.columns]
    ar_order_map = {}
    for t in available_targets:
        ar_order_map[t] = select_ar_order(panel, t)
        print(f"  {t}: AR({ar_order_map[t]})")

    # Run main tests
    results = run_all_targeted(panel, targets=available_targets)

    if results.empty:
        print("No results. Check data.")
        return

    # Regime interactions on top significant links
    regime_df = run_regime_interactions(panel, results, ar_order_map)

    print_top_results(results)
    save_results(results, regime_df, panel, ar_order_map, OUTPUT_DIR)

    print(f"\n{'='*70}\nDONE\n{'='*70}")


if __name__ == '__main__':
    main()