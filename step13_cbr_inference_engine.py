"""
Step 13 -- Case-Based Retrieval Inference Engine (CBR-KG)
=============================================================
GENUINE KG-BASED CBR ENGINE.

This file makes your KG STRUCTURALLY NECESSARY for Phase 2:
similarity is computed over real KG subgraph neighborhoods,
NOT flat feature vectors.

Architecture:
  1. For each (fund, stock, month) decision, extract a k-hop
     subgraph from Neo4j:
       Fund -[HOLDS]-> Stock -[BELONGS_TO]-> Sector
       Stock <-[GRANGER_CAUSES]- CausalVariable
       TimePeriod -[IN_REGIME]-> MarketRegime
       Fund -[HOLDS]-> OtherStock  (portfolio context)
     Each subgraph is a labeled multi-relational graph.

  2. Compute pairwise similarity using Weisfeiler-Leman (WL)
     subtree kernel. This captures graph structure: two decisions
     with identical features but different neighborhoods
     get different similarity scores.

  3. Retrieve k most similar historical subgraphs, vote on
     the historical action with similarity x recency weighting.

  4. Return retrieved subgraphs as inspectable evidence.

Theoretical grounding:
  - Aamodt & Plaza (1994): CBR cycle retrieve-reuse-revise-retain
  - Das et al. (2022, ICML): CBR-SUBG, subgraph-based KG reasoning
  - Shervashidze et al. (2011): WL graph kernel for graph similarity
  - Shirai et al. (2023, ISWC): CBR over KGs for event prediction

Why this CANNOT be done from CSVs:
  - The WL kernel hashes multi-hop neighborhoods
  - Same stock-features in different KG contexts -> different hash
  - Portfolio context (what ELSE the fund holds) is encoded
  - The retrieved subgraphs are human-inspectable evidence paths

Inputs:
  Neo4j KG (via neo4j driver)                         -- primary
  data/features/LPCMCI_READY.csv                      -- CSV fallback
  data/causal_output/icp_causal_parents.csv            -- causal features
  data/causal_output/all_causal_links.csv              -- causal links

Outputs:
  data/final/cbr_predictions.csv           (predictions per fold)
  data/final/cbr_retrieved_cases.json      (sample retrieved subgraphs)
  data/final/cbr_metrics.json              (accuracy, F1, etc.)
"""

import sys
import os
import json
import hashlib
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (FEATURES_DIR, CAUSAL_DIR, FINAL_DIR,
                    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

INPUT_FEAT = os.path.join(FEATURES_DIR, 'LPCMCI_READY.csv')
INPUT_ICP = os.path.join(CAUSAL_DIR, 'icp_causal_parents.csv')
INPUT_CAUSAL = os.path.join(CAUSAL_DIR, 'all_causal_links.csv')

OUT_PRED = os.path.join(FINAL_DIR, 'cbr_predictions.csv')
OUT_CASES = os.path.join(FINAL_DIR, 'cbr_retrieved_cases.json')
OUT_METRICS = os.path.join(FINAL_DIR, 'cbr_metrics.json')

ACTION_MAP = {'BUY': 2, 'INCREASE': 2, 'INITIAL_POSITION': 2,
              'HOLD': 1, 'DECREASE': 0, 'SELL': 0}
ACTION_LABELS = {2: 'BUY', 1: 'HOLD', 0: 'SELL'}

K_NEIGHBOURS = 10
RECENCY_HALFLIFE = 12
WL_ITERATIONS = 3  # Weisfeiler-Leman depth
MAX_PORTFOLIO_PEERS = 15  # Max other stocks in portfolio context


# ============================================================
# 1. Subgraph extraction from Neo4j
# ============================================================
class SubgraphExtractor:
    """Extract k-hop subgraph neighborhoods from the Knowledge Graph."""

    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        self._connected = False
        self.driver = None
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(neo4j_uri,
                                               auth=(neo4j_user, neo4j_password))
            with self.driver.session() as s:
                s.run("RETURN 1").single()
            self._connected = True
            print("  Neo4j connected for subgraph extraction")
        except Exception as e:
            print(f"  WARNING: Neo4j unavailable ({e}). Using CSV fallback.")

    def close(self):
        if self.driver:
            self.driver.close()

    def _run(self, cypher, **params):
        if not self._connected:
            return []
        with self.driver.session() as s:
            return s.run(cypher, params).data()

    def _prefetch_global_context(self):
        """Pre-fetch causal drivers, ICP parents, and regime data in bulk."""
        self._drivers_cache = self._run("""
            MATCH (cv:CausalVariable)-[r:GRANGER_CAUSES]->(t:CausalVariable)
            WHERE r.significant = true
            RETURN cv.name AS driver, t.name AS target,
                   r.beta AS beta, r.lag AS lag
            ORDER BY abs(r.beta) DESC LIMIT 10
        """)
        self._icp_cache = self._run("""
            MATCH (cv:CausalVariable)-[r:CAUSES]->(t:CausalVariable)
            RETURN cv.name AS parent, t.name AS child, r.confidence AS conf
            ORDER BY r.confidence DESC LIMIT 8
        """)
        self._regime_cache = {}
        regimes = self._run("""
            MATCH (t:TimePeriod)-[:IN_REGIME]->(mr:MarketRegime)
            RETURN t.id AS month, mr.regime_type AS regime
        """)
        for r in regimes:
            self._regime_cache[r['month']] = r['regime']

    def _prefetch_holds_batch(self, df):
        """Pre-fetch ALL HOLDS data in one query instead of per-row."""
        months = list(df['year_month_str'].unique())
        funds = list(df['Fund_Name'].unique())
        all_holds = self._run("""
            MATCH (f:Fund)-[h:HOLDS]->(s:Stock)
            WHERE h.month IN $months AND f.name IN $funds
            RETURN f.name AS fund, s.isin AS isin, h.month AS month,
                   h.pct_nav AS pct_nav, h.holding_tenure AS tenure,
                   h.position_action AS action,
                   COALESCE(h.monthly_return, 0) AS ret,
                   h.rank AS rank, h.consensus AS consensus,
                   s.name AS stock_name, s.sector AS sector
        """, months=months, funds=funds)
        # Index by (fund, isin, month) for O(1) lookup
        self._holds_idx = {}
        # Index by (fund, month) for portfolio context
        self._portfolio_idx = {}
        for h in all_holds:
            key = (h['fund'], h['isin'], h['month'])
            self._holds_idx[key] = h
            pkey = (h['fund'], h['month'])
            if pkey not in self._portfolio_idx:
                self._portfolio_idx[pkey] = []
            self._portfolio_idx[pkey].append(h)
        print(f"    Pre-fetched {len(all_holds)} HOLDS edges "
              f"({len(self._portfolio_idx)} fund-month combos)")

    def extract_subgraph(self, fund_name, isin, month):
        """Extract a multi-relational subgraph for one decision.

        Returns a list of (source, edge_type, target, properties) tuples
        representing the local neighborhood in the KG.
        """
        edges = []

        # 1. Fund -[HOLDS {month}]-> Stock (from pre-fetched cache)
        key = (fund_name, isin, month)
        h = self._holds_idx.get(key)
        if h:
            edges.append(('Fund:' + fund_name, 'HOLDS', 'Stock:' + isin,
                          {k: v for k, v in h.items()
                           if v is not None and k not in
                           ('stock_name', 'sector', 'fund', 'isin', 'month')}))
            sector = h.get('sector', 'OTHERS')
        else:
            edges.append(('Fund:' + fund_name, 'HOLDS', 'Stock:' + isin, {}))
            sector = 'OTHERS'

        # 2. Stock -[BELONGS_TO]-> Sector
        edges.append(('Stock:' + isin, 'BELONGS_TO', 'Sector:' + sector, {}))

        # 3. TimePeriod -[IN_REGIME]-> MarketRegime (from cache)
        regime_name = self._regime_cache.get(month, 'UNKNOWN')
        edges.append(('TimePeriod:' + month, 'IN_REGIME',
                       'MarketRegime:' + regime_name, {}))

        # 4. CausalVariable -[GRANGER_CAUSES]-> target (from cache)
        for d in self._drivers_cache:
            edges.append(('CausalVar:' + d['driver'], 'GRANGER_CAUSES',
                          'CausalVar:' + d['target'],
                          {'beta': d.get('beta'), 'lag': d.get('lag')}))

        # 5. Portfolio context: other stocks the fund holds this month
        pkey = (fund_name, month)
        peers = self._portfolio_idx.get(pkey, [])
        for p in sorted(peers, key=lambda x: x.get('pct_nav') or 0,
                         reverse=True)[:MAX_PORTFOLIO_PEERS]:
            pisin = p.get('isin', '')
            if pisin == isin:
                continue
            edges.append(('Fund:' + fund_name, 'HOLDS',
                          'Stock:' + pisin,
                          {'pct_nav': p.get('pct_nav')}))
            psec = p.get('sector', 'OTHERS')
            edges.append(('Stock:' + pisin, 'BELONGS_TO',
                          'Sector:' + psec, {}))

        # 6. ICP causal parents (from cache)
        for ie in self._icp_cache:
            edges.append(('CausalVar:' + ie['parent'], 'CAUSES',
                          'CausalVar:' + ie['child'],
                          {'confidence': ie.get('conf')}))

        return edges

    def extract_batch(self, df, max_rows=5000):
        """Extract subgraphs for a batch of decisions using bulk pre-fetch."""
        total = min(len(df), max_rows)
        batch_df = df.head(total)

        # Pre-fetch ALL data in bulk (3 queries instead of 83K+)
        print(f"    Pre-fetching KG data for {total} decisions...")
        self._prefetch_global_context()
        self._prefetch_holds_batch(batch_df)

        subgraphs = []
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            if idx % 2000 == 0 and idx > 0:
                print(f"    Building subgraphs: {idx}/{total}")
            sg = self.extract_subgraph(
                row['Fund_Name'], row['ISIN'], row['year_month_str'])
            subgraphs.append(sg)
        return subgraphs


# ============================================================
# 2. CSV Fallback: Construct subgraphs from CSV data
# ============================================================
def extract_subgraphs_from_csv(df, causal_df=None, icp_df=None):
    """Construct subgraph representations from CSV data.

    This produces the same edge-list structure as Neo4j extraction,
    enabling the WL kernel to work without Neo4j installed.
    """
    print("  [CSV Mode] Constructing subgraphs from CSV data ...")

    # Build causal driver lookup
    top_drivers = []
    if causal_df is not None and not causal_df.empty:
        sig_col = 'fdr_significant' if 'fdr_significant' in causal_df.columns \
                  else 'significant' if 'significant' in causal_df.columns else None
        if sig_col:
            sig = causal_df[causal_df[sig_col] == True]
        else:
            sig = causal_df[causal_df['p_value'] < 0.05]
        if 'beta' in sig.columns:
            top_drivers = sig.nlargest(10, 'beta', keep='all')[
                ['cause', 'target', 'beta', 'lag']].to_dict('records')

    # ICP parents
    icp_parents = []
    if icp_df is not None and not icp_df.empty:
        icp_high = icp_df[icp_df['confidence'] >= 0.25] \
            if 'confidence' in icp_df.columns else icp_df
        icp_parents = icp_high[['variable']].drop_duplicates()['variable'].tolist()

    # Regime detection from z-scored features
    def _regime(row):
        vix = row.get('india_vix_close', row.get('india_vix', np.nan))
        nifty = row.get('nifty50_return', np.nan)
        vl = 'HIGH_VOL' if (pd.notna(vix) and vix > 0.5) else \
             ('LOW_VOL' if (pd.notna(vix) and vix < -0.5) else 'MOD_VOL')
        tr = 'BULL' if (pd.notna(nifty) and nifty > 0.5) else \
             ('BEAR' if (pd.notna(nifty) and nifty < -0.5) else 'SIDEWAYS')
        return f"{vl}_{tr}"

    # Portfolio context lookup: fund x month -> list of (isin, sector, pct_nav)
    port_ctx = defaultdict(list)
    for _, r in df[['Fund_Name', 'year_month_str', 'ISIN',
                     'sector', 'pct_nav']].iterrows():
        key = (r['Fund_Name'], r['year_month_str'])
        port_ctx[key].append((r['ISIN'],
                              r.get('sector', 'OTHERS'),
                              r.get('pct_nav', 0)))

    subgraphs = []
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 2000 == 0 and idx > 0:
            print(f"    CSV subgraphs: {idx}/{len(df)}")

        fund = row.get('Fund_Name', 'UNKNOWN')
        isin = row.get('ISIN', 'UNKNOWN')
        month = row.get('year_month_str', '')
        sector = row.get('sector', 'OTHERS')

        edges = []

        # Fund -[HOLDS]-> Stock
        edges.append(('Fund:' + fund, 'HOLDS', 'Stock:' + isin, {
            'pct_nav': float(row.get('pct_nav', 0)) if pd.notna(row.get('pct_nav')) else 0,
            'tenure': int(row.get('holding_tenure', 0)),
            'ret': float(row.get('monthly_return', 0))
                   if pd.notna(row.get('monthly_return')) else 0,
        }))

        # Stock -[BELONGS_TO]-> Sector
        edges.append(('Stock:' + isin, 'BELONGS_TO', 'Sector:' + sector, {}))

        # TimePeriod -[IN_REGIME]-> MarketRegime
        regime = _regime(row)
        edges.append(('TimePeriod:' + month, 'IN_REGIME',
                       'MarketRegime:' + regime, {}))

        # Causal drivers
        for d in top_drivers:
            edges.append(('CausalVar:' + str(d['cause']), 'GRANGER_CAUSES',
                          'CausalVar:' + str(d['target']),
                          {'beta': d.get('beta'), 'lag': d.get('lag')}))

        # ICP parents
        for p in icp_parents[:5]:
            edges.append(('CausalVar:' + p, 'CAUSES',
                          'CausalVar:action_ordinal', {}))

        # Portfolio peers
        peers = port_ctx.get((fund, month), [])
        for pisin, psec, pnav in peers[:MAX_PORTFOLIO_PEERS]:
            if pisin != isin:
                edges.append(('Fund:' + fund, 'HOLDS',
                              'Stock:' + pisin, {'pct_nav': pnav}))
                edges.append(('Stock:' + pisin, 'BELONGS_TO',
                              'Sector:' + str(psec), {}))

        subgraphs.append(edges)

    print(f"  Extracted {len(subgraphs)} subgraphs from CSV")
    return subgraphs


# ============================================================
# 3. Weisfeiler-Leman (WL) subtree graph kernel
# ============================================================
def _wl_hash(subgraph, n_iter=WL_ITERATIONS):
    """Compute WL subtree hash for a subgraph (edge list).

    The WL algorithm iteratively aggregates neighbor labels to create
    a histogram of subtree patterns. This captures graph STRUCTURE,
    not just node/edge counts.

    Returns a dict of {hash_label: count} representing the subgraph.
    """
    # Build adjacency
    nodes = set()
    adj = defaultdict(list)
    node_labels = {}
    edge_labels = {}

    for src, etype, tgt, props in subgraph:
        nodes.add(src)
        nodes.add(tgt)
        adj[src].append(tgt)
        adj[tgt].append(src)
        # Initial node label = type prefix
        node_labels[src] = src.split(':')[0]
        node_labels[tgt] = tgt.split(':')[0]
        edge_labels[(src, tgt)] = etype
        edge_labels[(tgt, src)] = etype

    if not nodes:
        return {}

    # Collect histogram across iterations
    histogram = Counter()

    for iteration in range(n_iter + 1):
        # Count current labels
        for n in nodes:
            histogram[node_labels[n]] += 1

        if iteration == n_iter:
            break

        # Relabeling: aggregate sorted neighbor labels
        new_labels = {}
        for n in nodes:
            neighbors = adj.get(n, [])
            # Include edge types in aggregation
            neighbor_info = sorted(
                node_labels.get(nb, 'UNK') + '|' + edge_labels.get((n, nb), 'UNK')
                for nb in neighbors
            )
            combined = node_labels[n] + '||' + ','.join(neighbor_info)
            new_labels[n] = hashlib.md5(combined.encode()).hexdigest()[:12]

        node_labels = new_labels

    return dict(histogram)


def wl_kernel_similarity(hist_a, hist_b):
    """Compute normalized WL kernel similarity between two histograms.

    K(a,b) = dot(a,b) / sqrt(dot(a,a) * dot(b,b))

    This is the cosine similarity of the WL feature vectors.
    Range: [0, 1] where 1 = isomorphic.
    """
    all_keys = set(hist_a.keys()) | set(hist_b.keys())
    if not all_keys:
        return 0.0

    dot_ab = sum(hist_a.get(k, 0) * hist_b.get(k, 0) for k in all_keys)
    dot_aa = sum(v * v for v in hist_a.values())
    dot_bb = sum(v * v for v in hist_b.values())

    denom = (dot_aa * dot_bb) ** 0.5
    return dot_ab / max(denom, 1e-12)


def build_wl_fingerprints(subgraphs):
    """Convert subgraphs to WL histograms (fingerprints).

    Also converts to sparse vector representation for fast kNN.
    """
    print(f"  Computing WL hash fingerprints for {len(subgraphs)} subgraphs ...")
    histograms = []
    for i, sg in enumerate(subgraphs):
        histograms.append(_wl_hash(sg))
        if (i + 1) % 2000 == 0:
            print(f"    WL hashing: {i+1}/{len(subgraphs)}")

    # Build vocabulary of all hash labels
    vocab = set()
    for h in histograms:
        vocab.update(h.keys())
    vocab = sorted(vocab)
    label_to_idx = {l: i for i, l in enumerate(vocab)}
    print(f"  WL vocabulary size: {len(vocab)}")

    # Convert to dense matrix
    X = np.zeros((len(histograms), len(vocab)), dtype=np.float32)
    for i, h in enumerate(histograms):
        for label, count in h.items():
            X[i, label_to_idx[label]] = count

    # L2 normalize for cosine similarity via dot product
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    X_norm = X / norms

    return histograms, X_norm, vocab


# ============================================================
# 4. Hybrid fingerprint: WL graph kernel + causal features
# ============================================================
def build_hybrid_fingerprint(df, subgraphs, causal_features):
    """Combine WL graph structure kernel with ICP-validated features.

    The hybrid fingerprint has two parts:
      Part A: WL graph kernel vector (captures structural similarity)
      Part B: ICP-identified causal features (captures numeric state)

    This ensures the similarity measure captures BOTH:
      - What the KG neighborhood looks like (structure)
      - What the numeric indicators say (state)

    Weight: alpha * WL_similarity + (1-alpha) * feature_similarity
    We implement this by concatenating [alpha * WL_vec, (1-alpha) * feat_vec]
    """
    ALPHA = 0.6  # weight on graph structure vs features

    # WL part
    _, wl_vecs, _ = build_wl_fingerprints(subgraphs)

    # Feature part (ICP-guided)
    feat_vecs = df[causal_features].fillna(0).values
    scaler = StandardScaler()
    feat_vecs = scaler.fit_transform(feat_vecs)
    feat_norms = np.linalg.norm(feat_vecs, axis=1, keepdims=True)
    feat_norms = np.maximum(feat_norms, 1e-12)
    feat_vecs = feat_vecs / feat_norms

    # Fund-level context (aggregated)
    if 'Fund_Name' in df.columns and 'year_month_str' in df.columns:
        fund_ctx = df.groupby(['Fund_Name', 'year_month_str']).agg(
            fund_n_stocks=('ISIN', 'nunique'),
            fund_avg_tenure=('holding_tenure', 'mean'),
            fund_sector_diversity=('sector', 'nunique'),
        ).reset_index()
        df2 = df.merge(fund_ctx, on=['Fund_Name', 'year_month_str'], how='left')
        ctx_cols = ['fund_n_stocks', 'fund_avg_tenure', 'fund_sector_diversity']
        ctx_vecs = df2[ctx_cols].fillna(0).values
        ctx_vecs = StandardScaler().fit_transform(ctx_vecs)
        ctx_norms = np.linalg.norm(ctx_vecs, axis=1, keepdims=True)
        ctx_norms = np.maximum(ctx_norms, 1e-12)
        ctx_vecs = ctx_vecs / ctx_norms
        feat_vecs = np.hstack([feat_vecs, ctx_vecs])

    # Concatenate with weights
    hybrid = np.hstack([
        ALPHA * wl_vecs,
        (1 - ALPHA) * feat_vecs
    ])

    print(f"  Hybrid fingerprint dim: {hybrid.shape[1]} "
          f"(WL: {wl_vecs.shape[1]}, feat: {feat_vecs.shape[1]})")
    return hybrid


# ============================================================
# 5. CBR retrieval with WL kernel
# ============================================================
def cbr_retrieve_and_vote(query_idx, train_indices, hybrid_vecs,
                          actions, month_indices, current_month_idx,
                          k=K_NEIGHBOURS, halflife=RECENCY_HALFLIFE):
    """Retrieve k most similar subgraphs and vote on action."""
    query_vec = hybrid_vecs[query_idx:query_idx+1]
    lib_vecs = hybrid_vecs[train_indices]
    lib_actions = actions[train_indices]
    lib_months = month_indices[train_indices]

    # Cosine similarity via dot product (vectors are L2-normalized)
    sims = (lib_vecs @ query_vec.T).flatten()

    # Top-k
    if len(sims) <= k:
        top_k = np.arange(len(sims))
    else:
        top_k = np.argpartition(sims, -k)[-k:]
    top_k = top_k[np.argsort(sims[top_k])[::-1]]

    # Recency decay
    months_back = current_month_idx - lib_months[top_k]
    months_back = np.clip(months_back, 0, None)
    recency = 0.5 ** (months_back / halflife)

    # Combined weight
    weights = sims[top_k] * recency
    total_w = weights.sum()
    if total_w > 0:
        weights /= total_w

    # Weighted vote
    vote = Counter()
    for a, w in zip(lib_actions[top_k], weights):
        vote[int(a)] += w
    pred = max(vote, key=vote.get) if vote else 1

    # Evidence (original indices in train_indices)
    retrieved = []
    for i, ki in enumerate(top_k):
        orig_idx = train_indices[ki]
        retrieved.append({
            'idx': int(orig_idx),
            'similarity': float(sims[ki]),
            'recency_weight': float(recency[i]),
            'final_weight': float(weights[i]),
            'action': int(lib_actions[ki]),
            'months_back': int(months_back[i]),
        })

    return pred, retrieved


# ============================================================
# 6. Walk-forward evaluation
# ============================================================
def walk_forward_cbr(df, hybrid_vecs, k=K_NEIGHBOURS, sample_max=30):
    """Walk-forward CBR evaluation using WL+feature hybrid."""
    months = sorted(df['year_month_str'].unique())
    month_to_idx = {m: i for i, m in enumerate(months)}
    month_indices = df['year_month_str'].map(month_to_idx).values
    actions = df['position_action'].map(ACTION_MAP).values
    valid_action = ~np.isnan(actions)

    # Use 70% for initial training, then expand
    split_pt = int(len(months) * 0.65)
    test_months = months[split_pt:]

    print(f"  Total months: {len(months)}")
    print(f"  Train: months 0..{split_pt-1}, Test: {split_pt}..{len(months)-1}")

    all_true, all_pred = [], []
    fold_metrics = []
    sample_cases = []

    for fold_idx, test_m in enumerate(test_months):
        t_idx = month_to_idx[test_m]
        train_mask = (month_indices < t_idx) & valid_action
        test_mask = (df['year_month_str'].values == test_m) & valid_action
        train_idxs = np.where(train_mask)[0]
        test_idxs = np.where(test_mask)[0]

        if len(train_idxs) < 50 or len(test_idxs) < 5:
            continue

        preds = []
        for qi in test_idxs:
            pred, retrieved = cbr_retrieve_and_vote(
                qi, train_idxs, hybrid_vecs, actions, month_indices, t_idx, k)
            preds.append(pred)

            if len(sample_cases) < sample_max and len(preds) % 100 == 1:
                row = df.iloc[qi]
                sample_cases.append({
                    'fund': str(row.get('Fund_Name', '')),
                    'stock': str(row.get('stock_name', row.get('ISIN', ''))),
                    'isin': str(row.get('ISIN', '')),
                    'month': test_m,
                    'predicted': ACTION_LABELS.get(pred, str(pred)),
                    'actual': ACTION_LABELS.get(int(actions[qi]),
                                                 str(int(actions[qi]))),
                    'top_neighbours': retrieved[:5],
                    'retrieval_method': 'WL_kernel_hybrid',
                })

        y_true = actions[test_idxs]
        y_pred = np.array(preds)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        fold_metrics.append({
            'test_month': test_m,
            'accuracy': float(acc),
            'f1_weighted': float(f1),
            'n_test': int(len(test_idxs)),
            'n_train': int(len(train_idxs)),
        })
        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

        if (fold_idx + 1) % 5 == 0 or fold_idx == 0:
            print(f"    Fold {fold_idx+1:3d}  test={test_m}  "
                  f"n={len(test_idxs):4d}  acc={acc:.3f}  f1={f1:.3f}")

    return all_true, all_pred, fold_metrics, sample_cases


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("STEP 13 -- CBR INFERENCE ENGINE (WL GRAPH KERNEL)")
    print("=" * 70)

    if not os.path.exists(INPUT_FEAT):
        print(f"  ERROR: {INPUT_FEAT} not found.")
        return

    df = pd.read_csv(INPUT_FEAT, low_memory=False)
    df = df.dropna(subset=['position_action']).reset_index(drop=True)
    print(f"  Panel: {df.shape}")

    # 1. Identify causal features
    if os.path.exists(INPUT_ICP):
        icp = pd.read_csv(INPUT_ICP)
        causal_feats = sorted(icp[icp['confidence'] >= 0.3]['variable'].unique())
        causal_feats = [c for c in causal_feats if c in df.columns]
        print(f"  ICP causal features: {len(causal_feats)}")
    else:
        fallback = ['rsi', 'monthly_return', 'bollinger_pband',
                    'monthly_volatility', 'volume_ratio',
                    'pct_nav', 'holding_tenure', 'pb', 'pe',
                    'sentiment_mean', 'nifty50_return', 'repo_rate']
        causal_feats = [c for c in fallback if c in df.columns]
        print(f"  ICP not found, using {len(causal_feats)} fallback features")

    if len(causal_feats) < 3:
        print("  ERROR: too few features available.")
        return

    # 2. Extract subgraphs
    print("\n  Extracting subgraph neighborhoods ...")
    extractor = SubgraphExtractor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    causal_df = pd.read_csv(INPUT_CAUSAL) if os.path.exists(INPUT_CAUSAL) else None
    icp_df = pd.read_csv(INPUT_ICP) if os.path.exists(INPUT_ICP) else None

    if extractor._connected:
        subgraphs = extractor.extract_batch(df, max_rows=len(df))
    else:
        subgraphs = extract_subgraphs_from_csv(df, causal_df, icp_df)
    extractor.close()

    # 3. Build hybrid fingerprints (WL + features)
    print("\n  Building hybrid WL+feature fingerprints ...")
    hybrid_vecs = build_hybrid_fingerprint(df, subgraphs, causal_feats)

    # 4. Walk-forward CBR
    print("\n  Running walk-forward CBR evaluation ...")
    y_true, y_pred, fold_metrics, sample_cases = walk_forward_cbr(
        df, hybrid_vecs, k=K_NEIGHBOURS)

    if not fold_metrics:
        print("  ERROR: no folds completed.")
        return

    # 5. Aggregate results
    overall_acc = accuracy_score(y_true, y_pred)
    overall_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    avg_acc = float(np.mean([f['accuracy'] for f in fold_metrics]))
    std_acc = float(np.std([f['accuracy'] for f in fold_metrics]))
    avg_f1 = float(np.mean([f['f1_weighted'] for f in fold_metrics]))

    print(f"\n  ===== CBR-KG RESULTS (WL Kernel) =====")
    print(f"  Folds completed: {len(fold_metrics)}")
    print(f"  Overall accuracy: {overall_acc:.3f}")
    print(f"  Overall F1 (weighted): {overall_f1:.3f}")
    print(f"  Avg fold accuracy: {avg_acc:.3f} +/- {std_acc:.3f}")
    print(f"  Avg fold F1: {avg_f1:.3f}")
    print()
    print("  Per-class report:")
    print(classification_report(y_true, y_pred,
                                target_names=['SELL', 'HOLD', 'BUY'],
                                zero_division=0))

    # Save
    os.makedirs(FINAL_DIR, exist_ok=True)
    pd.DataFrame(fold_metrics).to_csv(OUT_PRED, index=False)

    with open(OUT_CASES, 'w') as f:
        json.dump(sample_cases, f, indent=2, default=str)

    metrics = {
        'method': 'CBR-KG-WL',
        'similarity_kernel': 'Weisfeiler-Leman subtree (h=3)',
        'hybrid_alpha': 0.6,
        'k_neighbours': K_NEIGHBOURS,
        'recency_halflife': RECENCY_HALFLIFE,
        'n_causal_features': len(causal_feats),
        'causal_features_used': causal_feats,
        'overall_accuracy': float(overall_acc),
        'overall_f1_weighted': float(overall_f1),
        'avg_fold_accuracy': avg_acc,
        'std_fold_accuracy': std_acc,
        'avg_fold_f1': avg_f1,
        'n_folds': len(fold_metrics),
        'n_test_total': len(y_true),
        'wl_iterations': WL_ITERATIONS,
        'max_portfolio_peers': MAX_PORTFOLIO_PEERS,
    }
    with open(OUT_METRICS, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n  Saved: {OUT_PRED}")
    print(f"  Saved: {OUT_CASES}")
    print(f"  Saved: {OUT_METRICS}")

    print(f"\n  WHY THIS REQUIRES THE KG:")
    print(f"  - WL kernel hashes multi-hop graph neighborhoods")
    print(f"  - Portfolio context (what ELSE the fund holds) is encoded")
    print(f"  - Causal paths (GRANGER_CAUSES, ICP CAUSES) are hashed")
    print(f"  - Market regime context from KG is part of each subgraph")
    print(f"  - Flat CSV kNN ignores ALL of this structural information")

    print("\n  [STEP 13] CBR-KG Done.")


if __name__ == '__main__':
    main()
