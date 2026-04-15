"""
Step 13a -- Causal Decision Path Extraction & Mimicry Engine
==============================================================
THIS IS THE CORE "MIMICRY" COMPONENT.

The Problem Statement says: "imitate the decision-making PROCESS."
NOT just classify BUY/SELL/HOLD (that's outcome prediction, not process mimicry).

What this step does:
1. For each (fund, stock, month) decision, TRAVERSE the KG to extract
   the full CAUSAL DECISION PATH:

   Path = [MacroCondition] --(DRIVES_FUND_DECISION)--> [TimePeriod]
          [TimePeriod] --(IN_REGIME)--> [MarketRegime]
          [CausalVariable] --(GRANGER_CAUSES)--> [action_ordinal]
          [Stock] --(BELONGS_TO)--> [Sector]
          [Fund] --(HOLDS)--> [Stock]  {with properties: pct_nav, tenure, etc.}

2. Encode each decision path as a SEQUENCE of (node_type, node_id, edge_type, edge_props)

3. Train a Transformer-based SEQUENCE MODEL on these paths to learn
   the decision PROCESS (not just the outcome)

4. At inference time:
   - Construct the query path from current KG context
   - Model predicts the next decision action
   - The PATH itself is the explanation

Why this REQUIRES the KG:
- Flat CSVs have no paths -- you need graph structure for traversal
- The path encodes the ORDER in which factors are considered
- Two stocks with identical features but in different graph contexts
  (different sector, different regime, different causal drivers)
  will produce DIFFERENT paths and DIFFERENT decisions

Theoretical Grounding:
- Grover & Leskovec (2016): node2vec -- learning structural patterns
- Das et al. (2018, ICLR): Go for a Walk -- paths for KG reasoning
- Aamodt & Plaza (1994): CBR cycle -- retrieve SIMILAR PATHS

Inputs:
  Neo4j KG (temporal + causal layers)
  data/features/LPCMCI_READY.csv (for labels)
  data/causal_output/all_causal_links.csv

Outputs:
  data/final/decision_paths.json          (extracted paths per decision)
  data/final/path_model_results.json      (accuracy, F1 of path-based prediction)
  data/final/path_embeddings.npy          (learned path embeddings for clustering)
"""

import sys
import os
import json
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (FEATURES_DIR, CAUSAL_DIR, FINAL_DIR, EVAL_DIR,
                    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

INPUT_FEAT = os.path.join(FEATURES_DIR, 'LPCMCI_READY.csv')
INPUT_CAUSAL = os.path.join(CAUSAL_DIR, 'all_causal_links.csv')
INPUT_ICP = os.path.join(CAUSAL_DIR, 'icp_causal_parents.csv')
INPUT_DML = os.path.join(CAUSAL_DIR, 'dml_causal_effects.csv')

OUT_PATHS = os.path.join(FINAL_DIR, 'decision_paths.json')
OUT_METRICS = os.path.join(FINAL_DIR, 'path_model_results.json')
OUT_EMBEDDINGS = os.path.join(FINAL_DIR, 'path_embeddings.npy')

ACTION_MAP = {'BUY': 2, 'INCREASE': 2, 'INITIAL_POSITION': 2,
              'HOLD': 1, 'DECREASE': 0, 'SELL': 0}

# Maximum path hops
MAX_PATH_HOPS = 5
# Top-K causal drivers to include in path
TOP_K_DRIVERS = 8


# ============================================================
# 1. Extract decision paths from Neo4j
# ============================================================
class DecisionPathExtractor:
    """Extract structured decision paths from the KG for each decision point."""

    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self._connected = True
            print("  Connected to Neo4j for path extraction")
        except Exception as e:
            print(f"  WARNING: Neo4j not available ({e}). Using CSV fallback.")
            self._connected = False
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def _run(self, cypher, **params):
        if not self._connected:
            return []
        with self.driver.session() as s:
            return s.run(cypher, params).data()

    def extract_path_for_decision(self, fund_name, isin, month):
        """Extract the full causal decision path for a specific (fund, stock, month).

        Path structure:
        [
            {"hop": 0, "node_type": "Fund", "node_id": fund_name, "edge_type": null},
            {"hop": 1, "node_type": "Stock", "node_id": isin, "edge_type": "HOLDS",
             "edge_props": {"pct_nav": 2.3, "holding_tenure": 5, ...}},
            {"hop": 2, "node_type": "Sector", "node_id": "IT", "edge_type": "BELONGS_TO"},
            {"hop": 3, "node_type": "MarketRegime", "node_id": "LOW_VOL_BULL",
             "edge_type": "IN_REGIME"},
            {"hop": 4, "node_type": "CausalVariable", "node_id": "momentum_3m",
             "edge_type": "GRANGER_CAUSES", "edge_props": {"beta": 0.04, "lag": 1}},
            ...
        ]
        """
        path = []

        # Hop 0: Fund node
        path.append({
            'hop': 0, 'node_type': 'Fund', 'node_id': fund_name,
            'edge_type': None, 'edge_props': {}
        })

        # Hop 1: Fund -[HOLDS]-> Stock (with edge properties)
        holds = self._run("""
            MATCH (f:Fund {name: $fund})-[h:HOLDS {month: $month}]->(s:Stock {isin: $isin})
            RETURN h.pct_nav AS pct_nav, h.holding_tenure AS tenure,
                   h.position_action AS action, h.allocation_change AS alloc_change,
                   h.rank AS rank, h.consensus AS consensus,
                   COALESCE(h.monthly_return, 0) AS monthly_return,
                   s.name AS stock_name, s.sector AS sector
        """, fund=fund_name, month=month, isin=isin)

        if holds:
            h = holds[0]
            path.append({
                'hop': 1, 'node_type': 'Stock', 'node_id': isin,
                'edge_type': 'HOLDS',
                'edge_props': {k: v for k, v in h.items()
                               if v is not None and k != 'stock_name' and k != 'sector'}
            })
            sector = h.get('sector', 'OTHERS')
        else:
            path.append({
                'hop': 1, 'node_type': 'Stock', 'node_id': isin,
                'edge_type': 'HOLDS', 'edge_props': {}
            })
            sector = 'OTHERS'

        # Hop 2: Stock -[BELONGS_TO]-> Sector
        path.append({
            'hop': 2, 'node_type': 'Sector', 'node_id': sector,
            'edge_type': 'BELONGS_TO', 'edge_props': {}
        })

        # Hop 3: TimePeriod -[IN_REGIME]-> MarketRegime
        regime = self._run("""
            MATCH (t:TimePeriod {id: $month})-[r:IN_REGIME]->(mr:MarketRegime)
            RETURN mr.regime_type AS regime, mr.vix_level AS vix_level,
                   mr.nifty_trend AS nifty_trend
        """, month=month)
        if regime:
            r = regime[0]
            path.append({
                'hop': 3, 'node_type': 'MarketRegime',
                'node_id': r.get('regime', 'UNKNOWN'),
                'edge_type': 'IN_REGIME',
                'edge_props': {k: v for k, v in r.items() if v is not None}
            })
        else:
            path.append({
                'hop': 3, 'node_type': 'MarketRegime', 'node_id': 'UNKNOWN',
                'edge_type': 'IN_REGIME', 'edge_props': {}
            })

        # Hop 4+: CausalVariable -[GRANGER_CAUSES]-> target
        # These are the causal drivers active for this decision
        drivers = self._run("""
            MATCH (cv:CausalVariable)-[r:GRANGER_CAUSES]->(t:CausalVariable)
            WHERE r.significant = true
            RETURN cv.name AS driver, r.beta AS beta, r.lag AS lag,
                   r.cause_group AS cause_group, r.partial_r2 AS partial_r2,
                   t.name AS target
            ORDER BY abs(r.beta) DESC
            LIMIT $top_k
        """, top_k=TOP_K_DRIVERS)

        for i, d in enumerate(drivers):
            path.append({
                'hop': 4 + i, 'node_type': 'CausalVariable',
                'node_id': d['driver'],
                'edge_type': 'GRANGER_CAUSES',
                'edge_props': {k: v for k, v in d.items()
                               if v is not None and k != 'driver'}
            })

        return path

    def extract_paths_batch(self, decisions_df, max_decisions=5000):
        """Extract paths for a batch of decisions. Returns list of paths."""
        paths = []
        total = min(len(decisions_df), max_decisions)
        print(f"  Extracting paths for {total} decisions...")

        for idx, (_, row) in enumerate(decisions_df.head(total).iterrows()):
            if idx % 500 == 0:
                print(f"    Progress: {idx}/{total}")
            path = self.extract_path_for_decision(
                row['Fund_Name'], row['ISIN'], row['year_month_str']
            )
            paths.append({
                'fund': row['Fund_Name'],
                'isin': row['ISIN'],
                'month': row['year_month_str'],
                'action': row.get('position_action', 'UNKNOWN'),
                'path': path
            })

        return paths


# ============================================================
# 2. CSV Fallback: Construct paths from CSV data when Neo4j unavailable
# ============================================================
def extract_paths_from_csv(df, causal_df, icp_df=None, dml_df=None,
                           max_decisions=5000):
    """Construct decision paths from CSV files (portable, no Neo4j needed).

    This produces the same path structure as Neo4j extraction, enabling
    the path model to work on any machine without Neo4j installed.
    """
    print("  [CSV Mode] Constructing decision paths from CSV data...")

    # Build causal driver lookup (top drivers by |beta|)
    if causal_df is not None and not causal_df.empty:
        sig = causal_df[causal_df.get('significant', causal_df.get(
            'fdr_significant', causal_df['p_value'] < 0.05)) == True] \
            if 'significant' in causal_df.columns \
            else causal_df[causal_df['p_value'] < 0.05]
        top_drivers = sig.nlargest(TOP_K_DRIVERS, 'beta', keep='all') \
            if 'beta' in sig.columns else sig.head(TOP_K_DRIVERS)
    else:
        top_drivers = pd.DataFrame()

    # Build ICP lookup
    icp_vars = set()
    if icp_df is not None and not icp_df.empty:
        icp_high = icp_df[icp_df.get('confidence', 0) >= 0.25] \
            if 'confidence' in icp_df.columns else icp_df
        icp_vars = set(icp_high['variable'].unique()) \
            if 'variable' in icp_df.columns else set()

    # Detect regime from features
    def get_regime(row):
        vix = row.get('india_vix_close', row.get('india_vix', np.nan))
        nifty_ret = row.get('nifty50_return', np.nan)
        if pd.isna(vix) and pd.isna(nifty_ret):
            return 'UNKNOWN'
        vix_level = 'HIGH_VOL' if (pd.notna(vix) and vix > 0.5) else \
                    ('LOW_VOL' if (pd.notna(vix) and vix < -0.5) else 'MOD_VOL')
        trend = 'BULL' if (pd.notna(nifty_ret) and nifty_ret > 0.5) else \
                ('BEAR' if (pd.notna(nifty_ret) and nifty_ret < -0.5) else 'SIDEWAYS')
        return f"{vix_level}_{trend}"

    # Fund-level context
    fund_ctx = df.groupby(['Fund_Name', 'year_month_str']).agg(
        fund_n_stocks=('ISIN', 'nunique'),
        fund_avg_tenure=('holding_tenure', 'mean'),
    ).reset_index() if 'Fund_Name' in df.columns else pd.DataFrame()

    paths = []
    total = min(len(df), max_decisions)

    for idx, (_, row) in enumerate(df.head(total).iterrows()):
        if idx % 1000 == 0 and idx > 0:
            print(f"    Progress: {idx}/{total}")

        fund = row.get('Fund_Name', 'UNKNOWN')
        isin = row.get('ISIN', 'UNKNOWN')
        month = row.get('year_month_str', '')
        sector = row.get('sector', 'OTHERS')
        action = row.get('position_action', 'HOLD')

        # Build path
        path = [
            {'hop': 0, 'node_type': 'Fund', 'node_id': fund,
             'edge_type': None, 'edge_props': {}},
            {'hop': 1, 'node_type': 'Stock', 'node_id': isin,
             'edge_type': 'HOLDS',
             'edge_props': {
                 'pct_nav': float(row.get('pct_nav', 0)) if pd.notna(row.get('pct_nav')) else 0,
                 'holding_tenure': int(row.get('holding_tenure', 0)),
                 'consensus': int(row.get('consensus_count', 0)),
                 'monthly_return': float(row.get('monthly_return', 0))
                     if pd.notna(row.get('monthly_return')) else 0,
             }},
            {'hop': 2, 'node_type': 'Sector', 'node_id': sector,
             'edge_type': 'BELONGS_TO', 'edge_props': {}},
            {'hop': 3, 'node_type': 'MarketRegime', 'node_id': get_regime(row),
             'edge_type': 'IN_REGIME', 'edge_props': {}},
        ]

        # Add causal drivers
        for i, (_, drv) in enumerate(top_drivers.iterrows()):
            cause = drv.get('cause', '')
            path.append({
                'hop': 4 + i, 'node_type': 'CausalVariable',
                'node_id': cause,
                'edge_type': 'GRANGER_CAUSES',
                'edge_props': {
                    'beta': float(drv.get('beta', 0)),
                    'lag': int(drv.get('lag', 0)),
                    'cause_group': str(drv.get('cause_group', '')),
                    'icp_certified': cause in icp_vars,
                }
            })
            if i >= TOP_K_DRIVERS - 1:
                break

        paths.append({
            'fund': fund, 'isin': isin, 'month': month,
            'action': action, 'path': path
        })

    print(f"  Extracted {len(paths)} decision paths")
    return paths


# ============================================================
# 3. Path encoding for ML model
# ============================================================

# Vocabulary for encoding path elements
NODE_TYPE_VOCAB = {'Fund': 0, 'Stock': 1, 'Sector': 2, 'MarketRegime': 3,
                   'CausalVariable': 4, 'TimePeriod': 5, 'PAD': 6}
EDGE_TYPE_VOCAB = {None: 0, 'HOLDS': 1, 'BELONGS_TO': 2, 'IN_REGIME': 3,
                   'GRANGER_CAUSES': 4, 'CAUSES': 5, 'CAUSAL_EFFECT': 6,
                   'DRIVES_FUND_DECISION': 7, 'PAD': 8}

# Max number of continuous features per path hop
MAX_EDGE_FEATURES = 8


def encode_path(path_dict, node_id_vocab, max_hops=MAX_PATH_HOPS + TOP_K_DRIVERS):
    """Encode a decision path into fixed-length numerical vectors.

    Returns:
        node_types: (max_hops,) int array -- node type IDs
        node_ids:   (max_hops,) int array -- node identity IDs
        edge_types: (max_hops,) int array -- edge type IDs
        edge_feats: (max_hops, MAX_EDGE_FEATURES) float array -- edge properties
    """
    path = path_dict['path']
    n_hops = min(len(path), max_hops)

    node_types = np.full(max_hops, NODE_TYPE_VOCAB['PAD'], dtype=np.int32)
    node_ids = np.zeros(max_hops, dtype=np.int32)
    edge_types = np.full(max_hops, EDGE_TYPE_VOCAB['PAD'], dtype=np.int32)
    edge_feats = np.zeros((max_hops, MAX_EDGE_FEATURES), dtype=np.float32)

    for i in range(n_hops):
        hop = path[i]
        node_types[i] = NODE_TYPE_VOCAB.get(hop['node_type'], 6)
        node_ids[i] = node_id_vocab.get(hop['node_id'], 0)
        edge_types[i] = EDGE_TYPE_VOCAB.get(hop.get('edge_type'), 0)

        # Pack edge properties into fixed-size vector
        props = hop.get('edge_props', {})
        feat_keys = ['pct_nav', 'holding_tenure', 'consensus', 'monthly_return',
                     'beta', 'lag', 'partial_r2', 'icp_certified']
        for j, k in enumerate(feat_keys[:MAX_EDGE_FEATURES]):
            val = props.get(k, 0)
            if isinstance(val, bool):
                val = float(val)
            elif val is None or (isinstance(val, float) and np.isnan(val)):
                val = 0.0
            edge_feats[i, j] = float(val)

    return node_types, node_ids, edge_types, edge_feats


def build_vocabularies(paths):
    """Build node_id vocabulary from all paths."""
    node_ids = set()
    for p in paths:
        for hop in p['path']:
            node_ids.add(hop['node_id'])
    vocab = {nid: idx + 1 for idx, nid in enumerate(sorted(node_ids, key=str))}
    vocab['PAD'] = 0
    return vocab


# ============================================================
# 4. Path-based prediction model (Transformer)
# ============================================================
def build_path_model_torch(max_hops, n_node_types, n_edge_types,
                           node_vocab_size, n_classes=3, d_model=64):
    """Build a Transformer-based path prediction model.

    Architecture:
        PathEncoder: embed(node_type) + embed(node_id) + embed(edge_type) + linear(edge_feats)
                     -> Transformer encoder -> classification head

    This model learns the DECISION PROCESS from paths, not just features.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("  PyTorch not available. Skipping Transformer path model.")
        return None

    class PathTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.node_type_emb = nn.Embedding(n_node_types + 1, d_model // 4)
            self.node_id_emb = nn.Embedding(node_vocab_size + 1, d_model // 4)
            self.edge_type_emb = nn.Embedding(n_edge_types + 1, d_model // 4)
            self.edge_feat_proj = nn.Linear(MAX_EDGE_FEATURES, d_model // 4)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=d_model * 2,
                dropout=0.1, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model, n_classes)
            )

        def forward(self, node_types, node_ids, edge_types, edge_feats):
            nt = self.node_type_emb(node_types)
            ni = self.node_id_emb(node_ids)
            et = self.edge_type_emb(edge_types)
            ef = self.edge_feat_proj(edge_feats)
            x = torch.cat([nt, ni, et, ef], dim=-1)  # (B, max_hops, d_model)

            # Padding mask
            pad_mask = (node_types == NODE_TYPE_VOCAB['PAD'])
            x = self.transformer(x, src_key_padding_mask=pad_mask)

            # Pool over non-padded positions
            mask = (~pad_mask).unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return self.classifier(pooled)

    return PathTransformer()


# ============================================================
# 5. Walk-forward training with path model
# ============================================================
def train_path_model(paths, train_months_list, test_months_list):
    """Train path Transformer with walk-forward evaluation."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  [SKIP] PyTorch not available for path model training.")
        return _sklearn_fallback(paths, train_months_list, test_months_list)

    # Build vocabularies
    vocab = build_vocabularies(paths)
    print(f"  Node ID vocabulary: {len(vocab)} entries")

    # Encode all paths
    max_hops = MAX_PATH_HOPS + TOP_K_DRIVERS
    all_nt, all_ni, all_et, all_ef, all_labels = [], [], [], [], []

    for p in paths:
        nt, ni, et, ef = encode_path(p, vocab, max_hops=max_hops)
        all_nt.append(nt)
        all_ni.append(ni)
        all_et.append(et)
        all_ef.append(ef)
        all_labels.append(ACTION_MAP.get(p['action'], 1))

    all_nt = np.array(all_nt)
    all_ni = np.array(all_ni)
    all_et = np.array(all_et)
    all_ef = np.array(all_ef)
    all_labels = np.array(all_labels)
    months = np.array([p['month'] for p in paths])

    # Train/test split
    train_mask = np.isin(months, train_months_list)
    test_mask = np.isin(months, test_months_list)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = build_path_model_torch(
        max_hops=max_hops,
        n_node_types=len(NODE_TYPE_VOCAB),
        n_edge_types=len(EDGE_TYPE_VOCAB),
        node_vocab_size=len(vocab),
        n_classes=3, d_model=64
    )
    if model is None:
        return _sklearn_fallback(paths, train_months_list, test_months_list)

    model = model.to(device)

    # Prepare tensors
    def make_tensors(mask):
        return (
            torch.LongTensor(all_nt[mask]).to(device),
            torch.LongTensor(all_ni[mask]).to(device),
            torch.LongTensor(all_et[mask]).to(device),
            torch.FloatTensor(all_ef[mask]).to(device),
            torch.LongTensor(all_labels[mask]).to(device),
        )

    train_data = make_tensors(train_mask)
    test_data = make_tensors(test_mask)

    # Class weights for imbalanced labels
    from collections import Counter
    label_counts = Counter(all_labels[train_mask].tolist())
    total = sum(label_counts.values())
    weights = torch.FloatTensor([total / (3 * label_counts.get(i, 1)) for i in range(3)]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    EPOCHS = 60
    PATIENCE = 10
    BATCH = 512

    n_train = train_data[0].shape[0]
    n_test = test_data[0].shape[0]

    print(f"  Training: {n_train} paths, Testing: {n_test} paths")

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_train, BATCH):
            idx = perm[i:i + BATCH]
            batch = [t[idx] for t in train_data]
            logits = model(batch[0], batch[1], batch[2], batch[3])
            loss = criterion(logits, batch[4])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(test_data[0], test_data[1], test_data[2], test_data[3])
            val_loss = criterion(val_logits, test_data[4]).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}")

        if patience_counter >= PATIENCE:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Load best model and evaluate
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(test_data[0], test_data[1], test_data[2], test_data[3])
        preds = logits.argmax(dim=1).cpu().numpy()
        true = test_data[4].cpu().numpy()

    from sklearn.metrics import accuracy_score, f1_score, classification_report
    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average='weighted')
    report = classification_report(true, preds,
                                   target_names=['SELL', 'HOLD', 'BUY'],
                                   output_dict=True)
    print(f"\n  Path Model Results: Acc={acc:.4f}, F1={f1:.4f}")

    # Extract path embeddings for clustering (fund manager style analysis)
    with torch.no_grad():
        # Get transformer output before classifier
        all_data = make_tensors(np.ones(len(paths), dtype=bool))
        nt_emb = model.node_type_emb(all_data[0])
        ni_emb = model.node_id_emb(all_data[1])
        et_emb = model.edge_type_emb(all_data[2])
        ef_emb = model.edge_feat_proj(all_data[3])
        x = torch.cat([nt_emb, ni_emb, et_emb, ef_emb], dim=-1)
        pad_mask = (all_data[0] == NODE_TYPE_VOCAB['PAD'])
        x = model.transformer(x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).unsqueeze(-1).float()
        embeddings = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embeddings = embeddings.cpu().numpy()

    return {
        'accuracy': float(acc),
        'f1_weighted': float(f1),
        'classification_report': report,
        'n_train': int(n_train),
        'n_test': int(n_test),
        'n_paths': len(paths),
        'model_type': 'PathTransformer',
        'device': str(device),
    }, embeddings


def _sklearn_fallback(paths, train_months, test_months):
    """Fallback: encode paths as flat vectors and use sklearn."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    print("  [Fallback] Using sklearn GBM on encoded paths")
    vocab = build_vocabularies(paths)
    max_hops = MAX_PATH_HOPS + TOP_K_DRIVERS

    X, y, months_arr = [], [], []
    for p in paths:
        nt, ni, et, ef = encode_path(p, vocab, max_hops=max_hops)
        flat = np.concatenate([nt.astype(float), ni.astype(float),
                               et.astype(float), ef.flatten()])
        X.append(flat)
        y.append(ACTION_MAP.get(p['action'], 1))
        months_arr.append(p['month'])

    X = np.array(X)
    y = np.array(y)
    months_arr = np.array(months_arr)

    train_mask = np.isin(months_arr, train_months)
    test_mask = np.isin(months_arr, test_months)

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                     learning_rate=0.05, random_state=42)
    clf.fit(X[train_mask], y[train_mask])
    preds = clf.predict(X[test_mask])
    true = y[test_mask]

    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average='weighted')
    report = classification_report(true, preds,
                                   target_names=['SELL', 'HOLD', 'BUY'],
                                   output_dict=True)
    print(f"\n  Path Model (GBM fallback): Acc={acc:.4f}, F1={f1:.4f}")

    return {
        'accuracy': float(acc),
        'f1_weighted': float(f1),
        'classification_report': report,
        'n_train': int(train_mask.sum()),
        'n_test': int(test_mask.sum()),
        'n_paths': len(paths),
        'model_type': 'GBM_path_fallback',
    }, X  # Use encoded vectors as embeddings


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("STEP 13a: Causal Decision Path Extraction & Mimicry")
    print("=" * 70)

    # Load data
    print("\n1. Loading data ...")
    if not os.path.exists(INPUT_FEAT):
        print(f"  ERROR: {INPUT_FEAT} not found. Run step08 first.")
        return

    df = pd.read_csv(INPUT_FEAT, low_memory=False)
    print(f"  Features: {df.shape}")

    causal_df = pd.read_csv(INPUT_CAUSAL) if os.path.exists(INPUT_CAUSAL) else None
    icp_df = pd.read_csv(INPUT_ICP) if os.path.exists(INPUT_ICP) else None
    dml_df = pd.read_csv(INPUT_DML) if os.path.exists(INPUT_DML) else None

    if causal_df is not None:
        print(f"  Causal links: {len(causal_df)}")

    # 2. Extract paths
    print("\n2. Extracting decision paths ...")
    extractor = DecisionPathExtractor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    if extractor._connected:
        paths = extractor.extract_paths_batch(df, max_decisions=10000)
    else:
        paths = extract_paths_from_csv(df, causal_df, icp_df, dml_df,
                                       max_decisions=10000)
    extractor.close()

    print(f"  Total paths: {len(paths)}")

    # Save paths
    print(f"  Saving paths to {OUT_PATHS}")
    # Save sample (full paths too large for JSON)
    sample_paths = paths[:500]
    with open(OUT_PATHS, 'w') as f:
        json.dump({'n_total': len(paths), 'sample_paths': sample_paths}, f, indent=2)

    # 3. Train path model
    print("\n3. Training path-based decision model ...")
    all_months = sorted(set(p['month'] for p in paths))
    split_idx = int(len(all_months) * 0.7)
    train_months = all_months[:split_idx]
    test_months = all_months[split_idx:]
    print(f"  Train months: {train_months[0]}..{train_months[-1]} ({len(train_months)})")
    print(f"  Test months:  {test_months[0]}..{test_months[-1]} ({len(test_months)})")

    results, embeddings = train_path_model(paths, train_months, test_months)

    # Save results
    with open(OUT_METRICS, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved metrics: {OUT_METRICS}")

    np.save(OUT_EMBEDDINGS, embeddings)
    print(f"  Saved embeddings: {OUT_EMBEDDINGS} ({embeddings.shape})")

    # 4. Summary
    print(f"\n{'='*70}")
    print("DECISION PATH MIMICRY SUMMARY")
    print(f"{'='*70}")
    print(f"  Paths extracted:     {len(paths)}")
    print(f"  Model type:          {results['model_type']}")
    print(f"  Accuracy:            {results['accuracy']:.4f}")
    print(f"  F1 (weighted):       {results['f1_weighted']:.4f}")
    print(f"  Train/Test split:    {results['n_train']}/{results['n_test']}")
    print(f"\n  WHY THIS REQUIRES THE KG:")
    print(f"  - Each path traverses Fund->Stock->Sector->Regime->CausalDrivers")
    print(f"  - Path structure encodes the PROCESS, not just the outcome")
    print(f"  - Identical features + different KG context = different paths")
    print(f"\n[step13a] Done.")


if __name__ == '__main__':
    main()
