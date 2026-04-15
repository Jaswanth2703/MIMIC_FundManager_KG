"""
Step 13b -- Export KG to portable format for GPU training
==========================================================
Run this on your MAIN machine (Neo4j available).
Output: data/rgcn/kg_export.pkl  (~50MB)

Copy data/rgcn/ folder to GPU machine, then run step13b_rgcn.py

What gets exported:
  - Node feature matrices for Fund, Stock, Sector, TimePeriod, CausalVariable
  - Edge indices for all 8 relation types
  - HOLDS edge features (the causal features identified by ICP/DML)
  - HOLDS labels (position_action)
  - Train/val/test split by month (no lookahead)
  - Node ID mappings
"""

import sys, os, json, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES_DIR, CAUSAL_DIR, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: pip install neo4j")
    sys.exit(1)

INPUT_FEAT = os.path.join(FEATURES_DIR, 'LPCMCI_READY.csv')
INPUT_ICP  = os.path.join(CAUSAL_DIR,   'icp_causal_parents.csv')
INPUT_DML  = os.path.join(CAUSAL_DIR,   'dml_causal_effects.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data', 'rgcn')
OUTPUT_PKL = os.path.join(OUTPUT_DIR, 'kg_export.pkl')

ACTION_MAP = {'BUY':2,'INCREASE':2,'INITIAL_POSITION':2,'HOLD':1,'DECREASE':0,'SELL':0}

# HOLDS edge features — chosen from ICP certified + DML significant
HOLDS_FEATURES = [
    'pct_nav', 'holding_tenure', 'allocation_change_lag1',
    'rsi', 'bollinger_pband', 'momentum_3m', 'monthly_return',
    'sentiment_mean', 'market_cap', 'beta', 'pe', 'pb',
    'india_vix_close', 'repo_rate', 'cpi_inflation',
]


def query_neo4j(driver, cypher):
    with driver.session() as s:
        return s.run(cypher).data()


def build_node_maps(driver):
    """Build str→int index maps for each node type."""
    maps = {}

    funds = query_neo4j(driver, "MATCH (n:Fund) RETURN n.name AS name ORDER BY n.name")
    maps['Fund'] = {r['name']: i for i, r in enumerate(funds)}

    stocks = query_neo4j(driver, "MATCH (n:Stock) RETURN n.isin AS isin ORDER BY n.isin")
    maps['Stock'] = {r['isin']: i for i, r in enumerate(stocks) if r['isin']}

    sectors = query_neo4j(driver, "MATCH (n:Sector) RETURN n.name AS name ORDER BY n.name")
    maps['Sector'] = {r['name']: i for i, r in enumerate(sectors)}

    months = query_neo4j(driver, "MATCH (n:TimePeriod) RETURN n.id AS month ORDER BY n.id")
    valid_months = [r['month'] for r in months if r['month']]
    maps['TimePeriod'] = {m: i for i, m in enumerate(valid_months)}

    cvars = query_neo4j(driver, "MATCH (n:CausalVariable) RETURN n.name AS name ORDER BY n.name")
    maps['CausalVariable'] = {r['name']: i for i, r in enumerate(cvars)}

    print(f"  Node counts: Fund={len(maps['Fund'])}, Stock={len(maps['Stock'])}, "
          f"Sector={len(maps['Sector'])}, TimePeriod={len(maps['TimePeriod'])}, "
          f"CausalVariable={len(maps['CausalVariable'])}")
    return maps


def build_node_features(maps, df, icp_df, dml_df):
    """Construct feature vectors for each node type."""
    feats = {}

    # Fund features: fund type (small=0, mid=1) + aggregate stats
    n_funds = len(maps['Fund'])
    fund_feat = np.zeros((n_funds, 4), dtype=np.float32)
    if 'Fund_Type' in df.columns and 'Fund_Name' in df.columns:
        for fname, fidx in maps['Fund'].items():
            rows = df[df['Fund_Name'] == fname]
            if len(rows) == 0:
                continue
            ft = str(rows['Fund_Type'].iloc[0]).lower()
            fund_feat[fidx, 0] = 1.0 if 'small' in ft else 0.0
            fund_feat[fidx, 1] = 1.0 if 'mid' in ft else 0.0
            if 'pct_nav' in df.columns:
                fund_feat[fidx, 2] = float(rows['pct_nav'].mean())
            if 'holding_tenure' in df.columns:
                fund_feat[fidx, 3] = float(rows['holding_tenure'].mean())
    feats['Fund'] = fund_feat
    print(f"  Fund features: {fund_feat.shape}")

    # Stock features: mean of ICP-relevant features across all months
    n_stocks = len(maps['Stock'])
    stock_cols = ['pct_nav','holding_tenure','rsi','monthly_return',
                  'sentiment_mean','pe','pb','market_cap','beta','bollinger_pband']
    stock_cols = [c for c in stock_cols if c in df.columns]
    stock_feat = np.zeros((n_stocks, len(stock_cols)), dtype=np.float32)
    if 'ISIN' in df.columns:
        for isin, sidx in maps['Stock'].items():
            rows = df[df['ISIN'] == isin]
            if len(rows) == 0:
                continue
            for j, col in enumerate(stock_cols):
                stock_feat[sidx, j] = float(rows[col].mean())
    feats['Stock'] = stock_feat
    print(f"  Stock features: {stock_feat.shape}")

    # Sector features: one-hot
    n_sectors = len(maps['Sector'])
    sector_feat = np.eye(n_sectors, dtype=np.float32)
    feats['Sector'] = sector_feat
    print(f"  Sector features: {sector_feat.shape}")

    # TimePeriod features: normalized month index + month-of-year + year
    n_months = len(maps['TimePeriod'])
    n_tp_feats = 3  # normalized_idx, month_of_year_sin, month_of_year_cos
    tp_feat = np.zeros((n_months, n_tp_feats), dtype=np.float32)
    for month_str, midx in maps['TimePeriod'].items():
        tp_feat[midx, 0] = midx / max(n_months - 1, 1)
        try:
            mo = int(month_str.split('-')[1])
            tp_feat[midx, 1] = np.sin(2 * np.pi * mo / 12)
            tp_feat[midx, 2] = np.cos(2 * np.pi * mo / 12)
        except (IndexError, ValueError):
            pass
    feats['TimePeriod'] = tp_feat
    print(f"  TimePeriod features: {tp_feat.shape}")

    # CausalVariable features: category one-hot + DML theta
    n_cvars = len(maps['CausalVariable'])
    categories = ['price_momentum','position_size','herding','risk',
                  'macro_rates','macro_equity','sentiment','fundamentals','other']
    cv_feat = np.zeros((n_cvars, len(categories) + 1), dtype=np.float32)

    # DML theta lookup
    dml_theta = {}
    if dml_df is not None:
        ao = dml_df[dml_df['outcome'] == 'action_ordinal']
        for _, row in ao.iterrows():
            dml_theta[row['treatment']] = float(row['theta_hat'])

    for cname, cidx in maps['CausalVariable'].items():
        # Infer category from name
        n = cname.lower()
        cat = 'other'
        if any(s in n for s in ['rsi','macd','bollinger','momentum','monthly_return','sma','alpha']):
            cat = 'price_momentum'
        elif any(s in n for s in ['pct_nav','holding_tenure','allocation_change','rank','sector_weight']):
            cat = 'position_size'
        elif 'consensus' in n:
            cat = 'herding'
        elif any(s in n for s in ['volatility','volume_ratio','beta']):
            cat = 'risk'
        elif any(s in n for s in ['repo_rate','cpi','us_10y','real_interest']):
            cat = 'macro_rates'
        elif any(s in n for s in ['nifty','vix','india']):
            cat = 'macro_equity'
        elif 'sentiment' in n or 'ratio' in n:
            cat = 'sentiment'
        elif any(s in n for s in ['pe','pb','eps','market_cap','bv_per','profit','income']):
            cat = 'fundamentals'
        if cat in categories:
            cv_feat[cidx, categories.index(cat)] = 1.0
        cv_feat[cidx, -1] = dml_theta.get(cname, 0.0)
    feats['CausalVariable'] = cv_feat
    print(f"  CausalVariable features: {cv_feat.shape}")

    return feats, stock_cols


def build_edge_indices(driver, maps):
    """Extract all edge types from Neo4j as (src_idx, dst_idx) arrays."""
    edges = {}

    # HOLDS edges + features + labels
    print("  Querying HOLDS edges (83K+ rows, ~30s)...")
    holds_data = query_neo4j(driver, """
        MATCH (f:Fund)-[h:HOLDS]->(s:Stock)
        RETURN f.name AS fund, s.isin AS isin,
               h.month AS month,
               h.pct_nav AS pct_nav,
               h.holding_tenure AS tenure,
               h.allocation_change AS alloc_change,
               h.rsi AS rsi,
               h.sentiment_score AS sentiment,
               COALESCE(h.monthly_return, 0) AS m_return,
               h.position_action AS action
        ORDER BY h.month
    """)
    print(f"  HOLDS: {len(holds_data)} rows")
    edges['HOLDS'] = holds_data

    # BELONGS_TO
    bt_data = query_neo4j(driver, """
        MATCH (s:Stock)-[:BELONGS_TO]->(sec:Sector)
        RETURN s.isin AS isin, sec.name AS sector
    """)
    bt_src = [maps['Stock'].get(r['isin'], -1) for r in bt_data]
    bt_dst = [maps['Sector'].get(r['sector'], -1) for r in bt_data]
    valid = [(s, d) for s, d in zip(bt_src, bt_dst) if s >= 0 and d >= 0]
    edges['BELONGS_TO'] = np.array(valid, dtype=np.int64).T if valid else np.zeros((2, 0), dtype=np.int64)
    print(f"  BELONGS_TO: {edges['BELONGS_TO'].shape[1]} edges")

    # GRANGER_CAUSES
    gc_data = query_neo4j(driver, """
        MATCH (c:CausalVariable)-[r:GRANGER_CAUSES]->(t:CausalVariable)
        RETURN c.name AS src, t.name AS dst,
               r.beta AS beta, r.partial_r2 AS r2
    """)
    gc_src = [maps['CausalVariable'].get(r['src'], -1) for r in gc_data]
    gc_dst = [maps['CausalVariable'].get(r['dst'], -1) for r in gc_data]
    gc_feat = [[r.get('beta', 0.0) or 0.0, r.get('r2', 0.0) or 0.0] for r in gc_data]
    valid_mask = [i for i, (s, d) in enumerate(zip(gc_src, gc_dst)) if s >= 0 and d >= 0]
    edges['GRANGER_CAUSES'] = {
        'index': np.array([[gc_src[i], gc_dst[i]] for i in valid_mask], dtype=np.int64).T,
        'attr':  np.array([gc_feat[i] for i in valid_mask], dtype=np.float32),
    }
    print(f"  GRANGER_CAUSES: {len(valid_mask)} edges")

    # ICP CAUSES
    icp_data = query_neo4j(driver, """
        MATCH (c:CausalVariable)-[r:CAUSES]->(t:CausalVariable)
        RETURN c.name AS src, t.name AS dst,
               r.confidence AS conf, r.in_intersection AS cert
    """)
    icp_src = [maps['CausalVariable'].get(r['src'], -1) for r in icp_data]
    icp_dst = [maps['CausalVariable'].get(r['dst'], -1) for r in icp_data]
    icp_feat = [[r.get('conf', 0.0) or 0.0, 1.0 if r.get('cert') else 0.0] for r in icp_data]
    valid_mask = [i for i, (s, d) in enumerate(zip(icp_src, icp_dst)) if s >= 0 and d >= 0]
    edges['CAUSES'] = {
        'index': np.array([[icp_src[i], icp_dst[i]] for i in valid_mask], dtype=np.int64).T if valid_mask else np.zeros((2,0),dtype=np.int64),
        'attr':  np.array([icp_feat[i] for i in valid_mask], dtype=np.float32) if valid_mask else np.zeros((0,2),dtype=np.float32),
    }
    print(f"  CAUSES (ICP): {len(valid_mask)} edges")

    # DML CAUSAL_EFFECT
    dml_data = query_neo4j(driver, """
        MATCH (c:CausalVariable)-[r:CAUSAL_EFFECT]->(t:CausalVariable)
        WHERE r.significant = true
        RETURN c.name AS src, t.name AS dst,
               r.theta_hat AS theta, r.icp_certified AS cert
    """)
    dml_src = [maps['CausalVariable'].get(r['src'], -1) for r in dml_data]
    dml_dst = [maps['CausalVariable'].get(r['dst'], -1) for r in dml_data]
    dml_feat = [[r.get('theta', 0.0) or 0.0, 1.0 if r.get('cert') else 0.0] for r in dml_data]
    valid_mask = [i for i, (s, d) in enumerate(zip(dml_src, dml_dst)) if s >= 0 and d >= 0]
    edges['CAUSAL_EFFECT'] = {
        'index': np.array([[dml_src[i], dml_dst[i]] for i in valid_mask], dtype=np.int64).T if valid_mask else np.zeros((2,0),dtype=np.int64),
        'attr':  np.array([dml_feat[i] for i in valid_mask], dtype=np.float32) if valid_mask else np.zeros((0,2),dtype=np.float32),
    }
    print(f"  CAUSAL_EFFECT (DML): {len(valid_mask)} edges")

    return edges


def process_holds(holds_data, maps, df):
    """Convert HOLDS query results to edge index + features + labels."""
    fund_idx, stock_idx, month_idx, labels = [], [], [], []
    feat_rows = []
    # Track identifiers for downstream prediction mapping
    fund_names_list, stock_isins_list, month_strs_list = [], [], []

    # Build month→months_sorted index
    all_months = sorted(maps['TimePeriod'].keys())
    month_to_sorted_idx = {m: i for i, m in enumerate(all_months)}

    # Feature lookup from df
    df_lookup = {}
    if 'ISIN' in df.columns and 'year_month_str' in df.columns:
        for _, row in df.iterrows():
            key = (str(row['ISIN']), str(row['year_month_str']))
            df_lookup[key] = row

    for r in holds_data:
        fi = maps['Fund'].get(r['fund'], -1)
        si = maps['Stock'].get(r['isin'], -1)
        if fi < 0 or si < 0:
            continue
        action = ACTION_MAP.get(str(r.get('action', '')), -1)
        if action < 0:
            continue

        month = str(r.get('month', ''))
        mi = month_to_sorted_idx.get(month, -1)

        # Get full features from df lookup
        key = (r['isin'], month)
        row = df_lookup.get(key)
        if row is not None:
            feat = [float(row.get(c, 0.0) or 0.0) for c in HOLDS_FEATURES if c in df.columns]
        else:
            # Fallback: use what Neo4j returned
            feat = [
                float(r.get('pct_nav', 0) or 0),
                float(r.get('tenure', 0) or 0),
                0.0,  # allocation_change_lag1
                float(r.get('rsi', 0) or 0),
                0.0, 0.0,  # bollinger, momentum
                float(r.get('m_return', 0) or 0),
                float(r.get('sentiment', 0) or 0),
            ] + [0.0] * (len(HOLDS_FEATURES) - 8)

        fund_idx.append(fi)
        stock_idx.append(si)
        month_idx.append(mi)
        labels.append(action)
        feat_rows.append(feat)
        fund_names_list.append(str(r.get('fund', '')))
        stock_isins_list.append(str(r.get('isin', '')))
        month_strs_list.append(month)

    n_feats = len([c for c in HOLDS_FEATURES if c in df.columns]) if df is not None else 8
    feat_arr = np.array(feat_rows, dtype=np.float32)
    if feat_arr.shape[1] < n_feats:
        pad = np.zeros((len(feat_rows), n_feats - feat_arr.shape[1]), dtype=np.float32)
        feat_arr = np.hstack([feat_arr, pad])

    return {
        'edge_index': np.array([fund_idx, stock_idx], dtype=np.int64),
        'edge_attr':  feat_arr,
        'labels':     np.array(labels, dtype=np.int64),
        'month_idx':  np.array(month_idx, dtype=np.int64),
        'fund_names': fund_names_list,
        'stock_isins': stock_isins_list,
        'month_strs': month_strs_list,
    }


def make_train_val_test_split(month_idx, all_months, train_frac=0.65, val_frac=0.17):
    """Split HOLDS edges by month for proper temporal evaluation."""
    n_months = len(all_months)
    train_cutoff = int(n_months * train_frac)
    val_cutoff   = int(n_months * (train_frac + val_frac))

    train_mask = month_idx < train_cutoff
    val_mask   = (month_idx >= train_cutoff) & (month_idx < val_cutoff)
    test_mask  = month_idx >= val_cutoff

    print(f"  Split: train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}")
    print(f"  Train months: {all_months[:train_cutoff][0]} → {all_months[:train_cutoff][-1]}")
    print(f"  Val months:   {all_months[train_cutoff:val_cutoff][0]} → {all_months[train_cutoff:val_cutoff][-1]}")
    print(f"  Test months:  {all_months[val_cutoff:][0]} → {all_months[val_cutoff:][-1]}")

    return train_mask, val_mask, test_mask


def main():
    print("="*70)
    print("STEP 13b -- EXPORT KG FOR GPU TRAINING")
    print("="*70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FEAT, low_memory=False)
    print(f"  Loaded panel: {df.shape}")

    icp_df = pd.read_csv(INPUT_ICP) if os.path.exists(INPUT_ICP) else None
    dml_df = pd.read_csv(INPUT_DML) if os.path.exists(INPUT_DML) else None

    # Normalize numeric features
    num_cols = [c for c in HOLDS_FEATURES if c in df.columns]
    for c in num_cols:
        mean, std = df[c].mean(), df[c].std()
        if std > 0:
            df[c] = (df[c] - mean) / std
    df[num_cols] = df[num_cols].fillna(0)
    print(f"  Normalized {len(num_cols)} feature columns")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as s:
        s.run("RETURN 1").single()
    print("  Neo4j connected")

    print("\n  Building node maps...")
    maps = build_node_maps(driver)

    print("\n  Building node features...")
    node_features, stock_cols = build_node_features(maps, df, icp_df, dml_df)

    print("\n  Extracting edges from Neo4j...")
    edges = build_edge_indices(driver, maps)
    driver.close()

    print("\n  Processing HOLDS edges...")
    holds = process_holds(edges.pop('HOLDS'), maps, df)
    print(f"  HOLDS processed: {holds['edge_index'].shape[1]} edges, "
          f"{holds['edge_attr'].shape[1]} features")

    all_months = sorted(maps['TimePeriod'].keys())
    train_mask, val_mask, test_mask = make_train_val_test_split(
        holds['month_idx'], all_months)

    export = {
        'node_features':  node_features,
        'edges':          edges,
        'holds':          holds,
        'train_mask':     train_mask,
        'val_mask':       val_mask,
        'test_mask':      test_mask,
        'node_maps':      maps,
        'holds_features': [c for c in HOLDS_FEATURES if c in df.columns],
        'all_months':     all_months,
        'metadata': {
            'n_holds':       holds['edge_index'].shape[1],
            'n_train':       int(train_mask.sum()),
            'n_val':         int(val_mask.sum()),
            'n_test':        int(test_mask.sum()),
            'n_fund':        len(maps['Fund']),
            'n_stock':       len(maps['Stock']),
            'n_sector':      len(maps['Sector']),
            'n_timeperiod':  len(maps['TimePeriod']),
            'n_causalvar':   len(maps['CausalVariable']),
        }
    }

    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(export, f, protocol=4)

    size_mb = os.path.getsize(OUTPUT_PKL) / 1e6
    print(f"\n  Saved: {OUTPUT_PKL}  ({size_mb:.1f} MB)")
    print("\n  Copy data/rgcn/ to GPU machine and run step13b_rgcn.py")
    print("  EXPORT DONE.")


if __name__ == '__main__':
    main()
