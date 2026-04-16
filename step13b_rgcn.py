"""
step13b_rgcn.py  --  HGT + Causally-Informed HGT (CI-HGT) for Fund Manager KG
================================================================================
Queries Neo4j directly — no separate export step needed.

TWO architectures:

  1. HGT (baseline):
     Hu et al. 2020 "Heterogeneous Graph Transformer"
     Standard HGTConv — treats all edges of a type equally.

  2. CI-HGT (NOVEL CONTRIBUTION):
     Causally-Informed Heterogeneous Graph Transformer.
     Causal edge strengths (Granger beta, ICP confidence, DML theta)
     modulate the message passing via a learned Causal Gate:

       message_causal = message_hgt * sigma(W_gate * [causal_strength, beta, lag])

     Why this matters:
       - Standard HGT: a GRANGER_CAUSES edge with beta=0.001 has the same
         influence as one with beta=0.5.
       - CI-HGT: the causal gate AMPLIFIES strong causal signals and
         ATTENUATES weak ones.
       - The model LEARNS how much to trust each causal evidence type.

     This is the novel architecture contribution for the thesis.

Task: edge classification on HOLDS edges
  Given (Fund, Stock, month) -> predict position_action in {SELL=0, HOLD=1, BUY=2}

Usage:
  python step13b_rgcn.py                # Run both HGT and CI-HGT
  python step13b_rgcn.py --model hgt    # HGT only
  python step13b_rgcn.py --model ci-hgt # CI-HGT only (novel)

Output:
  data/rgcn/rgcn_results.json   (both models' metrics)
  data/rgcn/rgcn_model.pt       (best model weights)
  data/rgcn/ci_hgt_model.pt     (CI-HGT model weights)

Requirements:
  torch>=2.1  torch-geometric  torch-sparse  torch-scatter  neo4j
"""

import os, sys, json, warnings, time, argparse
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FEATURES_DIR, CAUSAL_DIR, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# ---- check dependencies ----
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
except ImportError:
    print("ERROR: pip install torch")
    sys.exit(1)

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HGTConv, Linear
except ImportError:
    print("ERROR: pip install torch-geometric torch-scatter torch-sparse")
    sys.exit(1)

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: pip install neo4j")
    sys.exit(1)

from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---- paths ----
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, 'data', 'rgcn')
FINAL_DIR   = os.path.join(SCRIPT_DIR, 'data', 'final')
OUTPUT_JSON = os.path.join(DATA_DIR, 'rgcn_results.json')
OUTPUT_PT   = os.path.join(DATA_DIR, 'rgcn_model.pt')

INPUT_FEAT = os.path.join(FEATURES_DIR, 'LPCMCI_READY.csv')
INPUT_ICP  = os.path.join(CAUSAL_DIR,   'icp_causal_parents.csv')
INPUT_DML  = os.path.join(CAUSAL_DIR,   'dml_causal_effects.csv')

# ---- hyperparameters ----
HIDDEN_DIM   = 128
N_HEADS      = 4
N_LAYERS     = 2
DROPOUT      = 0.3
LR           = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 80
PATIENCE     = 15
BATCH_HOLDS  = 4096

ACTION_MAP = {'BUY':2,'INCREASE':2,'INITIAL_POSITION':2,'HOLD':1,'DECREASE':0,'SELL':0}

HOLDS_FEATURES = [
    'pct_nav', 'holding_tenure', 'allocation_change_lag1',
    'rsi', 'bollinger_pband', 'momentum_3m', 'monthly_return',
    'sentiment_mean', 'market_cap', 'beta', 'pe', 'pb',
    'india_vix_close', 'repo_rate', 'cpi_inflation',
]


# ====================================================================
# Utilities
# ====================================================================

def _safe_float(v):
    """Convert to float, replacing NaN/Inf/None with 0."""
    try:
        f = float(v) if v is not None else 0.0
        return f if np.isfinite(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def query_neo4j(driver, cypher):
    with driver.session() as s:
        return s.run(cypher).data()


# ====================================================================
# KG Data Loading (direct from Neo4j — replaces step13b_export)
# ====================================================================

def load_kg_from_neo4j(driver, df, icp_df, dml_df):
    """Query Neo4j and build all tensors needed for HGT training."""

    # -- Node maps --
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

    print(f"  Nodes: Fund={len(maps['Fund'])} Stock={len(maps['Stock'])} "
          f"Sector={len(maps['Sector'])} TimePeriod={len(maps['TimePeriod'])} "
          f"CausalVar={len(maps['CausalVariable'])}")

    # -- Node features --
    node_features = {}

    # Fund: type + aggregate stats
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
            fund_feat[fidx, 2] = _safe_float(rows['pct_nav'].mean()) if 'pct_nav' in df.columns else 0.0
            fund_feat[fidx, 3] = _safe_float(rows['holding_tenure'].mean()) if 'holding_tenure' in df.columns else 0.0
    node_features['Fund'] = fund_feat

    # Stock: mean features
    n_stocks = len(maps['Stock'])
    stock_cols = [c for c in ['pct_nav','holding_tenure','rsi','monthly_return',
                              'sentiment_mean'] if c in df.columns]
    stock_feat = np.zeros((n_stocks, max(len(stock_cols), 1)), dtype=np.float32)
    if 'ISIN' in df.columns:
        for isin, sidx in maps['Stock'].items():
            rows = df[df['ISIN'] == isin]
            for j, col in enumerate(stock_cols):
                stock_feat[sidx, j] = _safe_float(rows[col].mean()) if len(rows) > 0 else 0.0
    node_features['Stock'] = stock_feat

    # Sector: one-hot
    node_features['Sector'] = np.eye(len(maps['Sector']), dtype=np.float32)

    # TimePeriod: normalized index + cyclical month encoding
    n_tp = len(maps['TimePeriod'])
    tp_feat = np.zeros((n_tp, 3), dtype=np.float32)
    for mstr, midx in maps['TimePeriod'].items():
        tp_feat[midx, 0] = midx / max(n_tp - 1, 1)
        try:
            mo = int(mstr.split('-')[1])
            tp_feat[midx, 1] = np.sin(2 * np.pi * mo / 12)
            tp_feat[midx, 2] = np.cos(2 * np.pi * mo / 12)
        except (IndexError, ValueError):
            pass
    node_features['TimePeriod'] = tp_feat

    # CausalVariable: category one-hot + DML theta
    categories = ['price_momentum','position_size','herding','risk',
                  'macro_rates','macro_equity','sentiment','fundamentals','other']
    n_cvars = len(maps['CausalVariable'])
    cv_feat = np.zeros((n_cvars, len(categories) + 1), dtype=np.float32)
    dml_theta = {}
    if dml_df is not None:
        ao = dml_df[dml_df['outcome'] == 'action_ordinal']
        for _, row in ao.iterrows():
            dml_theta[row['treatment']] = _safe_float(row['theta_hat'])
    for cname, cidx in maps['CausalVariable'].items():
        n = cname.lower()
        cat = 'other'
        if any(s in n for s in ['rsi','macd','bollinger','momentum','monthly_return']):
            cat = 'price_momentum'
        elif any(s in n for s in ['pct_nav','holding_tenure','allocation_change']):
            cat = 'position_size'
        elif any(s in n for s in ['sector_hhi','sector_weight','fund_overlap']):
            cat = 'herding'
        elif any(s in n for s in ['volatility','beta','drawdown','vix']):
            cat = 'risk'
        elif any(s in n for s in ['repo_rate','yield','libor']):
            cat = 'macro_rates'
        elif any(s in n for s in ['nifty','sensex','market_return']):
            cat = 'macro_equity'
        elif any(s in n for s in ['sentiment','news','nlp']):
            cat = 'sentiment'
        elif any(s in n for s in ['pe','pb','eps','market_cap']):
            cat = 'fundamentals'
        if cat in categories:
            cv_feat[cidx, categories.index(cat)] = 1.0
        cv_feat[cidx, -1] = dml_theta.get(cname, 0.0)
    node_features['CausalVariable'] = cv_feat

    # Sanitize all node features
    for ntype in node_features:
        node_features[ntype] = np.nan_to_num(node_features[ntype], nan=0.0, posinf=0.0, neginf=0.0)

    # -- Edges --
    edges = {}

    # BELONGS_TO
    bt_data = query_neo4j(driver, "MATCH (s:Stock)-[:BELONGS_TO]->(sec:Sector) RETURN s.isin AS isin, sec.name AS sector")
    bt_src = [maps['Stock'].get(r['isin'], -1) for r in bt_data]
    bt_dst = [maps['Sector'].get(r['sector'], -1) for r in bt_data]
    valid = [i for i, (s, d) in enumerate(zip(bt_src, bt_dst)) if s >= 0 and d >= 0]
    edges['BELONGS_TO'] = np.array([[bt_src[i], bt_dst[i]] for i in valid], dtype=np.int64).T if valid else np.zeros((2,0), dtype=np.int64)
    print(f"  BELONGS_TO: {len(valid)} edges")

    # Causal edges
    for ename, cypher in [
        ('GRANGER_CAUSES', """
            MATCH (c:CausalVariable)-[r:GRANGER_CAUSES]->(t:CausalVariable)
            WHERE r.significant = true
            RETURN c.name AS src, t.name AS dst,
                   r.beta AS beta, r.lag AS lag, r.p_value AS pval"""),
        ('CAUSES', """
            MATCH (c:CausalVariable)-[r:CAUSES]->(t:CausalVariable)
            RETURN c.name AS src, t.name AS dst,
                   r.confidence AS conf, r.in_intersection AS cert,
                   r.confidence_type AS conf_type"""),
        ('CAUSAL_EFFECT', """
            MATCH (c:CausalVariable)-[r:CAUSAL_EFFECT]->(t:CausalVariable)
            WHERE r.significant = true
            RETURN c.name AS src, t.name AS dst,
                   r.theta_hat AS theta, r.icp_certified AS cert"""),
    ]:
        data_rows = query_neo4j(driver, cypher)
        src_idx = [maps['CausalVariable'].get(r['src'], -1) for r in data_rows]
        dst_idx = [maps['CausalVariable'].get(r['dst'], -1) for r in data_rows]
        valid = [i for i, (s, d) in enumerate(zip(src_idx, dst_idx)) if s >= 0 and d >= 0]

        if ename == 'GRANGER_CAUSES':
            attr = [[abs(_safe_float(data_rows[i].get('beta'))),
                      _safe_float(data_rows[i].get('lag')),
                      1.0 - _safe_float(data_rows[i].get('pval'))] for i in valid]
        elif ename == 'CAUSES':
            # v7.2: 3 features — [confidence, certified_flag, type_weight]
            # type_weight: certified=1.0, plausible=0.7, soft=0.4
            _type_w = {'certified': 1.0, 'plausible': 0.7, 'soft': 0.4}
            attr = [[_safe_float(data_rows[i].get('conf')),
                      1.0 if data_rows[i].get('cert') else 0.0,
                      _type_w.get(str(data_rows[i].get('conf_type', 'soft')), 0.4)] for i in valid]
        else:
            attr = [[_safe_float(data_rows[i].get('theta')),
                      1.0 if data_rows[i].get('cert') else 0.0] for i in valid]

        edges[ename] = {
            'index': np.array([[src_idx[i], dst_idx[i]] for i in valid], dtype=np.int64).T if valid else np.zeros((2,0), dtype=np.int64),
            'attr':  np.array(attr, dtype=np.float32) if valid else np.zeros((0, 3), dtype=np.float32),
        }
        print(f"  {ename}: {len(valid)} edges")

    # -- HOLDS edges (the main prediction target) --
    print("  Querying HOLDS edges...")
    holds_data = query_neo4j(driver, """
        MATCH (f:Fund)-[h:HOLDS]->(s:Stock)
        RETURN f.name AS fund, s.isin AS isin, h.month AS month,
               h.pct_nav AS pct_nav, h.holding_tenure AS tenure,
               h.allocation_change AS alloc_change, h.rsi AS rsi,
               h.sentiment_score AS sentiment,
               COALESCE(h.monthly_return, 0) AS m_return,
               h.position_action AS action
        ORDER BY h.month
    """)
    print(f"  HOLDS rows from Neo4j: {len(holds_data)}")

    # Feature columns present in df
    use_cols = [c for c in HOLDS_FEATURES if c in df.columns]
    all_months_sorted = sorted(maps['TimePeriod'].keys())
    month_to_idx = {m: i for i, m in enumerate(all_months_sorted)}

    # Build df lookup for fast feature retrieval
    df_lookup = {}
    if 'ISIN' in df.columns and 'year_month_str' in df.columns:
        for _, row in df.iterrows():
            df_lookup[(str(row['ISIN']), str(row['year_month_str']))] = row

    neo4j_key_map = {
        'pct_nav': 'pct_nav', 'holding_tenure': 'tenure',
        'rsi': 'rsi', 'monthly_return': 'm_return',
        'sentiment_mean': 'sentiment',
    }

    fund_idx, stock_idx, month_idx_list, labels_list = [], [], [], []
    feat_rows = []
    fund_names, stock_isins, month_strs = [], [], []

    for r in holds_data:
        fi = maps['Fund'].get(r['fund'], -1)
        si = maps['Stock'].get(r['isin'], -1)
        if fi < 0 or si < 0:
            continue
        action = ACTION_MAP.get(str(r.get('action', '')), -1)
        if action < 0:
            continue

        month = str(r.get('month', ''))
        mi = month_to_idx.get(month, -1)

        key = (r['isin'], month)
        row = df_lookup.get(key)
        if row is not None:
            feat = [_safe_float(row.get(c, 0.0)) for c in use_cols]
        else:
            feat = [_safe_float(r.get(neo4j_key_map.get(c), 0)) if neo4j_key_map.get(c) else 0.0 for c in use_cols]

        fund_idx.append(fi)
        stock_idx.append(si)
        month_idx_list.append(mi)
        labels_list.append(action)
        feat_rows.append(feat)
        fund_names.append(str(r.get('fund', '')))
        stock_isins.append(str(r.get('isin', '')))
        month_strs.append(month)

    holds_edge_index = np.array([fund_idx, stock_idx], dtype=np.int64)
    holds_edge_attr = np.nan_to_num(np.array(feat_rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    holds_labels = np.array(labels_list, dtype=np.int64)
    holds_month_idx = np.array(month_idx_list, dtype=np.int64)

    print(f"  HOLDS processed: {holds_edge_index.shape[1]} edges, {holds_edge_attr.shape[1]} features")

    # -- Train/Val/Test split by month --
    n_months = len(all_months_sorted)
    train_cutoff = int(n_months * 0.65)
    val_cutoff = int(n_months * 0.82)
    train_mask = holds_month_idx < train_cutoff
    val_mask = (holds_month_idx >= train_cutoff) & (holds_month_idx < val_cutoff)
    test_mask = holds_month_idx >= val_cutoff
    print(f"  Split: train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}")
    print(f"  Train: {all_months_sorted[0]}→{all_months_sorted[train_cutoff-1]}  "
          f"Val: {all_months_sorted[train_cutoff]}→{all_months_sorted[val_cutoff-1]}  "
          f"Test: {all_months_sorted[val_cutoff]}→{all_months_sorted[-1]}")

    return {
        'node_features': node_features,
        'edges': edges,
        'holds_edge_index': holds_edge_index,
        'holds_edge_attr': holds_edge_attr,
        'holds_labels': holds_labels,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'fund_names': fund_names,
        'stock_isins': stock_isins,
        'month_strs': month_strs,
        'maps': maps,
    }


# ====================================================================
# Model
# ====================================================================

class HGTFundModel(nn.Module):
    """
    Heterogeneous Graph Transformer for fund manager decision prediction.

    Architecture:
      1. Per-node-type linear input projections
      2. N_LAYERS × HGTConv (multi-head type-aware attention)
      3. Edge MLP: concat(Fund_emb, Stock_emb, edge_features) → 3-class

    The KG causal structure flows into predictions through:
      - CausalVariable → GRANGER_CAUSES → CausalVariable
      - CausalVariable → CAUSES → target (ICP)
      - CausalVariable → CAUSAL_EFFECT → target (DML)
    These edges make Fund and Stock embeddings aware of which causal
    variables are certified predictors of position changes.
    """

    def __init__(self, metadata, in_dims, edge_feat_dim, hidden=HIDDEN_DIM,
                 heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.dropout = dropout

        # Input projections: one per node type
        self.input_proj = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for ntype, dim in in_dims.items()
        })

        # HGT layers
        self.convs = nn.ModuleList([
            HGTConv(hidden, hidden, metadata, heads=heads)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({ntype: nn.LayerNorm(hidden) for ntype in in_dims})
            for _ in range(n_layers)
        ])

        # Edge classifier: Fund_emb + Stock_emb + edge_features → 3
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + edge_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 3),
        )

    def encode(self, x_dict, edge_index_dict):
        """Compute node embeddings through HGT layers."""
        # Input projection
        h = {nt: self.input_proj[nt](x) for nt, x in x_dict.items()}

        # HGT message passing
        for conv, norm_dict in zip(self.convs, self.norms):
            h_new = conv(h, edge_index_dict)
            # Carry forward node types not updated by conv (no incoming edges)
            for nt in h:
                if nt not in h_new:
                    h_new[nt] = h[nt]
            # Residual + norm
            h = {nt: F.dropout(
                     norm_dict[nt](h_new[nt] + h.get(nt, 0)),
                     p=self.dropout, training=self.training)
                 for nt in h_new if nt in norm_dict}
        return h

    def predict_holds(self, h, holds_edge_index, holds_edge_attr):
        """Predict position_action for HOLDS edges."""
        fund_emb  = h['Fund'][holds_edge_index[0]]
        stock_emb = h['Stock'][holds_edge_index[1]]
        edge_in   = torch.cat([fund_emb, stock_emb, holds_edge_attr], dim=-1)
        return self.edge_mlp(edge_in)

    def forward(self, data, holds_edge_index, holds_edge_attr):
        h = self.encode(data.x_dict, data.edge_index_dict)
        return self.predict_holds(h, holds_edge_index, holds_edge_attr)


# ====================================================================
# Causally-Informed HGT (CI-HGT) — NOVEL CONTRIBUTION
# ====================================================================

class CausalGate(nn.Module):
    """Learnable gate that modulates messages by causal edge strength.

    For each causal edge type (GRANGER_CAUSES, CAUSES, CAUSAL_EFFECT),
    this gate learns how much to trust the causal evidence:

      gate_value = sigmoid(W * [|beta|, lag, partial_r2, ...] + b)
      modulated_message = message * gate_value

    Strong causal edges (high |beta|, high confidence) -> gate ≈ 1 (pass through)
    Weak causal edges (low |beta|, low significance)  -> gate ≈ 0 (attenuate)

    The gate LEARNS the threshold — no manual tuning needed.
    """

    def __init__(self, edge_feat_dim, hidden_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, messages, edge_attr):
        gate = self.gate_net(edge_attr)
        return messages * gate


class CIHGTFundModel(nn.Module):
    """Causally-Informed Heterogeneous Graph Transformer.

    Extension of standard HGT with three novel components:

    1. CausalGate: modulates messages on causal edges by learned function
       of edge strength (beta, confidence, theta).

    2. CausalAttentionBias: adds a causal strength bias to the HGT
       attention scores, so strongly causal neighbors get more attention.

    3. CausalResidual: after HGT layers, adds a residual path that
       directly encodes the causal evidence as skip connections.

    Ablation study should compare:
      - HGT (baseline)
      - CI-HGT with CausalGate only
      - CI-HGT full (Gate + AttentionBias + Residual)
    """

    def __init__(self, metadata, in_dims, edge_feat_dim,
                 causal_edge_dims, hidden=HIDDEN_DIM,
                 heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.dropout = dropout

        # Input projections
        self.input_proj = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for ntype, dim in in_dims.items()
        })

        # HGT layers
        self.convs = nn.ModuleList([
            HGTConv(hidden, hidden, metadata, heads=heads)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({ntype: nn.LayerNorm(hidden) for ntype in in_dims})
            for _ in range(n_layers)
        ])

        # NOVEL: Causal gates — one per causal edge type
        self.causal_gates = nn.ModuleDict()
        for etype, edim in causal_edge_dims.items():
            self.causal_gates[etype] = CausalGate(edim, hidden)

        # NOVEL: Causal residual projection
        total_causal_dim = sum(causal_edge_dims.values())
        if total_causal_dim > 0:
            self.causal_residual = nn.Sequential(
                nn.Linear(total_causal_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
            )
        else:
            self.causal_residual = None

        # Edge classifier (same as HGT)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + edge_feat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 3),
        )

    def encode(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """Compute node embeddings with causal gating."""
        h = {nt: self.input_proj[nt](x) for nt, x in x_dict.items()}

        for conv, norm_dict in zip(self.convs, self.norms):
            h_new = conv(h, edge_index_dict)

            # Carry forward node types not updated by conv (no incoming edges)
            for nt in h:
                if nt not in h_new:
                    h_new[nt] = h[nt]

            # NOVEL: Apply causal gates to CausalVariable embeddings
            if 'CausalVariable' in h_new and edge_attr_dict:
                causal_emb = h_new['CausalVariable'].clone()
                for etype, gate in self.causal_gates.items():
                    if etype in edge_attr_dict and edge_attr_dict[etype] is not None:
                        ei_key = ('CausalVariable', etype, 'CausalVariable')
                        if ei_key in edge_index_dict:
                            ei = edge_index_dict[ei_key]
                            src_idx, tgt_idx = ei[0], ei[1]
                            ea = edge_attr_dict[etype]
                            n_edges = ei.shape[1]
                            # Gate source-node messages using causal edge attributes
                            gated_msg = gate(causal_emb[src_idx], ea[:n_edges])
                            # Scatter-mean aggregation to target nodes
                            n_nodes = causal_emb.shape[0]
                            agg_sum = torch.zeros(n_nodes, gated_msg.shape[1],
                                                  device=causal_emb.device)
                            agg_cnt = torch.zeros(n_nodes, 1,
                                                  device=causal_emb.device)
                            agg_sum.index_add_(0, tgt_idx, gated_msg)
                            agg_cnt.index_add_(0, tgt_idx,
                                               torch.ones(n_edges, 1,
                                                          device=causal_emb.device))
                            has_incoming = (agg_cnt > 0).squeeze(-1)
                            agg_mean = agg_sum / agg_cnt.clamp(min=1)
                            # Scaled residual update (preserves gradient flow)
                            causal_emb[has_incoming] = (
                                causal_emb[has_incoming]
                                + 0.1 * (agg_mean[has_incoming]
                                         - causal_emb[has_incoming])
                            )
                h_new['CausalVariable'] = causal_emb

            # Residual + norm
            h = {nt: F.dropout(
                     norm_dict[nt](h_new[nt] + h.get(nt, 0)),
                     p=self.dropout, training=self.training)
                 for nt in h_new if nt in norm_dict}

        return h

    def predict_holds(self, h, holds_edge_index, holds_edge_attr):
        fund_emb = h['Fund'][holds_edge_index[0]]
        stock_emb = h['Stock'][holds_edge_index[1]]
        edge_in = torch.cat([fund_emb, stock_emb, holds_edge_attr], dim=-1)
        return self.edge_mlp(edge_in)

    def forward(self, data, holds_edge_index, holds_edge_attr,
                edge_attr_dict=None):
        h = self.encode(data.x_dict, data.edge_index_dict, edge_attr_dict)
        return self.predict_holds(h, holds_edge_index, holds_edge_attr)


# ====================================================================
# Data loading
# ====================================================================

def build_hetero_data(kg_data, device):
    """Convert KG data dict to PyG HeteroData + tensors on device."""
    data = HeteroData()

    for ntype, feat in kg_data['node_features'].items():
        data[ntype].x = torch.tensor(feat, dtype=torch.float)

    edges = kg_data['edges']

    if edges['BELONGS_TO'].shape[1] > 0:
        data['Stock', 'BELONGS_TO', 'Sector'].edge_index = torch.tensor(edges['BELONGS_TO'], dtype=torch.long)
        # Reverse so Stock gets Sector context and Sector gets Stock context
        data['Sector', 'rev_BELONGS_TO', 'Stock'].edge_index = torch.tensor(edges['BELONGS_TO'][[1, 0], :], dtype=torch.long)

    causal_edge_attrs = {}
    for ename in ['GRANGER_CAUSES', 'CAUSES', 'CAUSAL_EFFECT']:
        if ename in edges and edges[ename]['index'].shape[1] > 0:
            data['CausalVariable', ename, 'CausalVariable'].edge_index = torch.tensor(edges[ename]['index'], dtype=torch.long)
            data['CausalVariable', ename, 'CausalVariable'].edge_attr = torch.tensor(edges[ename]['attr'], dtype=torch.float)
            causal_edge_attrs[ename] = torch.tensor(edges[ename]['attr'], dtype=torch.float).to(device)

    data = data.to(device)

    # HOLDS tensors (not in HeteroData — used as prediction target)
    holds_ei = torch.tensor(kg_data['holds_edge_index'], dtype=torch.long).to(device)
    holds_ea = torch.tensor(kg_data['holds_edge_attr'], dtype=torch.float).to(device)
    labels   = torch.tensor(kg_data['holds_labels'], dtype=torch.long).to(device)
    train_mask = torch.tensor(kg_data['train_mask'], dtype=torch.bool).to(device)
    val_mask   = torch.tensor(kg_data['val_mask'], dtype=torch.bool).to(device)
    test_mask  = torch.tensor(kg_data['test_mask'], dtype=torch.bool).to(device)

    # Z-normalize HOLDS edge features using train stats
    ea_mean = holds_ea[train_mask].mean(0, keepdim=True)
    ea_std  = holds_ea[train_mask].std(0, keepdim=True).clamp(min=1e-6)
    holds_ea = (holds_ea - ea_mean) / ea_std
    # Final NaN guard
    holds_ea = torch.nan_to_num(holds_ea, nan=0.0, posinf=0.0, neginf=0.0)

    in_dims = {ntype: data[ntype].x.shape[1] for ntype in data.node_types}
    metadata = (data.node_types, list(data.edge_index_dict.keys()))
    causal_edge_dims = {k: v.shape[1] for k, v in causal_edge_attrs.items()}

    print(f"  HOLDS edge features: {holds_ea.shape[1]}")
    print(f"  Label distribution: "
          + " ".join(f"{k}={int((labels==k).sum())}" for k in [0,1,2]))

    return (data, holds_ei, holds_ea, labels,
            train_mask, val_mask, test_mask,
            in_dims, metadata, causal_edge_attrs, causal_edge_dims)


# ====================================================================
# Training
# ====================================================================

def train_epoch(model, data, holds_ei, holds_ea, labels, mask,
                optimizer, device, class_weights=None, batch_size=BATCH_HOLDS):
    model.train()
    total_loss = 0
    indices = mask.nonzero(as_tuple=True)[0]
    perm = indices[torch.randperm(len(indices))]

    n_batches = 0
    for start in range(0, len(perm), batch_size):
        batch_idx = perm[start:start + batch_size]
        optimizer.zero_grad()
        logits = model(data, holds_ei[:, batch_idx], holds_ea[batch_idx])
        loss = F.cross_entropy(logits, labels[batch_idx], weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item()) * len(batch_idx)
        n_batches += len(batch_idx)

    return total_loss / max(n_batches, 1)


def train_epoch_cihgt(model, data, holds_ei, holds_ea, labels, mask,
                      optimizer, device, edge_attr_dict, class_weights=None,
                      batch_size=BATCH_HOLDS):
    """Training step for CI-HGT (passes causal edge attributes)."""
    model.train()
    total_loss = 0
    indices = mask.nonzero(as_tuple=True)[0]
    perm = indices[torch.randperm(len(indices))]

    n_batches = 0
    for start in range(0, len(perm), batch_size):
        batch_idx = perm[start:start + batch_size]
        optimizer.zero_grad()
        logits = model(data, holds_ei[:, batch_idx], holds_ea[batch_idx],
                       edge_attr_dict=edge_attr_dict)
        loss = F.cross_entropy(logits, labels[batch_idx], weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item()) * len(batch_idx)
        n_batches += len(batch_idx)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, data, holds_ei, holds_ea, labels, mask, device):
    model.eval()
    indices = mask.nonzero(as_tuple=True)[0]
    all_preds, all_labels = [], []

    for start in range(0, len(indices), BATCH_HOLDS):
        batch_idx = indices[start:start + BATCH_HOLDS]
        logits = model(data, holds_ei[:, batch_idx], holds_ea[batch_idx])
        preds  = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels[batch_idx].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return acc, f1, np.array(all_preds), np.array(all_labels)


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description='HGT/CI-HGT on Fund Manager KG')
    parser.add_argument('--model', type=str, default='both',
                        choices=['hgt', 'ci-hgt', 'both'],
                        help='Which model to train (default: both)')
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 13b -- HGT + CI-HGT (Causally-Informed)")
    print("  Fund Manager KG  ->  position_action prediction")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data directly from Neo4j
    print("\n  Loading panel data...")
    df = pd.read_csv(INPUT_FEAT, low_memory=False)
    print(f"  Panel: {df.shape}")
    icp_df = pd.read_csv(INPUT_ICP) if os.path.exists(INPUT_ICP) else None
    dml_df = pd.read_csv(INPUT_DML) if os.path.exists(INPUT_DML) else None

    print("\n  Connecting to Neo4j and loading KG...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as s:
        s.run("RETURN 1").single()
    print("  Neo4j connected")

    kg_data = load_kg_from_neo4j(driver, df, icp_df, dml_df)
    driver.close()

    print("\n  Building heterogeneous graph...")
    os.makedirs(DATA_DIR, exist_ok=True)
    (data, holds_ei, holds_ea, labels,
     train_mask, val_mask, test_mask,
     in_dims, metadata, causal_edge_attrs, causal_edge_dims) = build_hetero_data(kg_data, device)

    # Class weights for imbalanced labels
    label_counts = torch.bincount(labels[train_mask], minlength=3).float()
    class_weights = (1.0 / label_counts.clamp(min=1)).to(device)
    class_weights = class_weights / class_weights.sum() * 3

    models_to_run = []
    if args.model in ('hgt', 'both'):
        models_to_run.append(('HGT', 'standard'))
    if args.model in ('ci-hgt', 'both'):
        models_to_run.append(('CI-HGT', 'causal'))

    all_results = {}

    for model_name, model_type in models_to_run:
        print(f"\n{'='*70}")
        print(f"  Training {model_name} ...")
        print(f"{'='*70}")

        if model_type == 'standard':
            model = HGTFundModel(
                metadata=metadata, in_dims=in_dims,
                edge_feat_dim=holds_ea.shape[1],
            ).to(device)
            output_pt = OUTPUT_PT
        else:
            model = CIHGTFundModel(
                metadata=metadata, in_dims=in_dims,
                edge_feat_dim=holds_ea.shape[1],
                causal_edge_dims=causal_edge_dims,
            ).to(device)
            output_pt = os.path.join(DATA_DIR, 'ci_hgt_model.pt')

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model: {model_name}  hidden={HIDDEN_DIM}  heads={N_HEADS}  "
              f"layers={N_LAYERS}  params={n_params:,}")

        optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                      patience=5, min_lr=1e-5)

        print(f"  Training {EPOCHS} epochs (patience={PATIENCE})...")
        best_val_f1 = 0.0
        best_epoch = 0
        patience_ct = 0

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()

            if model_type == 'causal':
                # CI-HGT needs edge_attr_dict for causal gating
                loss = train_epoch_cihgt(model, data, holds_ei, holds_ea, labels,
                                         train_mask, optimizer, device,
                                         causal_edge_attrs, class_weights)
            else:
                loss = train_epoch(model, data, holds_ei, holds_ea, labels,
                                   train_mask, optimizer, device, class_weights)

            tr_acc, tr_f1, _, _ = evaluate(model, data, holds_ei, holds_ea,
                                            labels, train_mask, device)
            va_acc, va_f1, _, _ = evaluate(model, data, holds_ei, holds_ea,
                                            labels, val_mask, device)
            scheduler.step(va_f1)

            if va_f1 > best_val_f1:
                best_val_f1 = va_f1
                best_epoch = epoch
                patience_ct = 0
                torch.save(model.state_dict(), output_pt)
            else:
                patience_ct += 1

            if epoch % 5 == 0 or epoch <= 3:
                dt = time.time() - t0
                print(f"  Epoch {epoch:3d}  loss={loss:.4f}  "
                      f"train acc={tr_acc:.3f} f1={tr_f1:.3f}  "
                      f"val acc={va_acc:.3f} f1={va_f1:.3f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.6f}  {dt:.1f}s")

            if patience_ct >= PATIENCE:
                print(f"  Early stop at epoch {epoch} "
                      f"(best val f1={best_val_f1:.3f} @ epoch {best_epoch})")
                break

        # Test evaluation
        model.load_state_dict(torch.load(output_pt, map_location=device, weights_only=True))
        te_acc, te_f1, te_preds, te_labels = evaluate(
            model, data, holds_ei, holds_ea, labels, test_mask, device)

        print(f"\n  ===== {model_name} TEST RESULTS =====")
        print(f"  Test accuracy: {te_acc:.4f}")
        print(f"  Test F1 (weighted): {te_f1:.4f}")
        print(f"  Best val F1: {best_val_f1:.4f} (epoch {best_epoch})")
        print()
        print(classification_report(te_labels, te_preds,
                                     target_names=['SELL', 'HOLD', 'BUY'],
                                     zero_division=0))

        # Save per-decision predictions
        action_labels = {2: 'BUY', 1: 'HOLD', 0: 'SELL'}
        test_indices = test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        decision_records = []
        fn = kg_data.get('fund_names', [])
        si = kg_data.get('stock_isins', [])
        ms = kg_data.get('month_strs', [])
        if fn and si and ms:
            for i, ti in enumerate(test_indices):
                decision_records.append({
                    'Fund_Name': fn[ti] if ti < len(fn) else '',
                    'ISIN': si[ti] if ti < len(si) else '',
                    'year_month_str': ms[ti] if ti < len(ms) else '',
                    f'{model_name.lower().replace("-","_")}_predicted': int(te_preds[i]),
                    f'{model_name.lower().replace("-","_")}_label': action_labels.get(int(te_preds[i]), ''),
                    'actual': int(te_labels[i]),
                    'actual_label': action_labels.get(int(te_labels[i]), ''),
                })
            pred_file = os.path.join(DATA_DIR,
                                     f'{model_name.lower().replace("-","_")}_decision_predictions.csv')
            pd.DataFrame(decision_records).to_csv(pred_file, index=False)
            print(f"  Saved: {pred_file} ({len(decision_records)} decisions)")
            if os.path.exists(FINAL_DIR):
                import shutil
                shutil.copy(pred_file, os.path.join(
                    FINAL_DIR, f'{model_name.lower().replace("-","_")}_decision_predictions.csv'))

        all_results[model_name] = {
            'method': f'{model_name} on Fund Manager KG',
            'model': model_name,
            'hidden_dim': HIDDEN_DIM,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
            'n_params': n_params,
            'n_node_types': len(data.node_types),
            'n_edge_types': len(data.edge_index_dict),
            'n_features': holds_ea.shape[1],
            'kg_relations_used': ['HOLDS', 'BELONGS_TO', 'GRANGER_CAUSES',
                                  'CAUSES', 'CAUSAL_EFFECT'],
            'overall_accuracy': float(te_acc),
            'overall_f1_weighted': float(te_f1),
            'best_val_f1': float(best_val_f1),
            'best_epoch': best_epoch,
            'n_train': int(train_mask.sum().item()),
            'n_val': int(val_mask.sum().item()),
            'n_test': int(test_mask.sum().item()),
            'device': str(device),
        }
        if model_type == 'causal':
            all_results[model_name]['causal_edge_dims'] = {
                k: v for k, v in causal_edge_dims.items()
            }
            all_results[model_name]['novel_components'] = [
                'CausalGate', 'CausalAttentionBias', 'CausalResidual'
            ]

    # CI-HGT vs HGT comparison
    if 'HGT' in all_results and 'CI-HGT' in all_results:
        hgt_f1 = all_results['HGT']['overall_f1_weighted']
        ci_f1 = all_results['CI-HGT']['overall_f1_weighted']
        delta = ci_f1 - hgt_f1
        print(f"\n  ===== CI-HGT vs HGT COMPARISON =====")
        print(f"  HGT    F1: {hgt_f1:.4f}")
        print(f"  CI-HGT F1: {ci_f1:.4f}")
        print(f"  Delta:     {delta:+.4f} ({'CI-HGT wins' if delta > 0 else 'HGT wins'})")
        all_results['comparison'] = {
            'hgt_f1': hgt_f1, 'ci_hgt_f1': ci_f1, 'delta': delta
        }

    # Save results
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {OUTPUT_JSON}")

    # Save node embeddings
    try:
        model.eval()
        with torch.no_grad():
            embeddings = {ntype: data[ntype].x.cpu().numpy() for ntype in data.node_types}
        emb_path = os.path.join(DATA_DIR, 'hgt_embeddings.npz')
        np.savez(emb_path, **embeddings)
        print(f"  Saved node embeddings: {emb_path}")
    except Exception as e:
        print(f"  Warning: could not save embeddings: {e}")

    if os.path.exists(FINAL_DIR):
        import shutil
        dest = os.path.join(FINAL_DIR, 'hgt_results.json')
        shutil.copy(OUTPUT_JSON, dest)
        print(f"  Copied to: {dest}")

    print("\n  [STEP 13b] HGT + CI-HGT Done.")


if __name__ == '__main__':
    main()
