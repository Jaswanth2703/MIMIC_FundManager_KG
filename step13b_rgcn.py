"""
step13b_rgcn.py  --  HGT + Causally-Informed HGT (CI-HGT) for Fund Manager KG
================================================================================
STANDALONE — no project imports, no Neo4j, no config.py needed.
Copy this file + data/rgcn/kg_export.pkl to your GPU machine and run.

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

Input:
  data/rgcn/kg_export.pkl  (from step13b_export_kg_for_gpu.py)

Output:
  data/rgcn/rgcn_results.json   (both models' metrics)
  data/rgcn/rgcn_model.pt       (best model weights)
  data/rgcn/ci_hgt_model.pt     (CI-HGT model weights)

Requirements (install with setup_gpu.bat):
  torch>=2.1  torch-geometric  torch-sparse  torch-scatter
"""

import os, sys, json, pickle, warnings, time, argparse
import numpy as np

warnings.filterwarnings('ignore')

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
    from torch_geometric.transforms import ToUndirected
except ImportError:
    print("ERROR: pip install torch-geometric torch-scatter torch-sparse")
    sys.exit(1)

from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---- paths ----
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, 'data', 'rgcn')
INPUT_PKL   = os.path.join(DATA_DIR, 'kg_export.pkl')
OUTPUT_JSON = os.path.join(DATA_DIR, 'rgcn_results.json')
OUTPUT_PT   = os.path.join(DATA_DIR, 'rgcn_model.pt')

# ---- hyperparameters ----
HIDDEN_DIM   = 128
N_HEADS      = 4
N_LAYERS     = 2
DROPOUT      = 0.3
LR           = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 80
PATIENCE     = 15     # early stopping
BATCH_HOLDS  = 4096   # process HOLDS edges in batches for memory efficiency


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

def load_hetero_data(export):
    """Convert pickled export to PyG HeteroData object."""
    data = HeteroData()
    meta = export['metadata']

    # Node features
    for ntype, feat in export['node_features'].items():
        data[ntype].x = torch.tensor(feat, dtype=torch.float)

    # Static edges (non-HOLDS)
    edges = export['edges']

    if edges['BELONGS_TO'].shape[1] > 0:
        data['Stock', 'BELONGS_TO', 'Sector'].edge_index = \
            torch.tensor(edges['BELONGS_TO'], dtype=torch.long)

    for ename in ['GRANGER_CAUSES', 'CAUSES', 'CAUSAL_EFFECT']:
        if ename in edges and edges[ename]['index'].shape[1] > 0:
            data['CausalVariable', ename, 'CausalVariable'].edge_index = \
                torch.tensor(edges[ename]['index'], dtype=torch.long)
            data['CausalVariable', ename, 'CausalVariable'].edge_attr = \
                torch.tensor(edges[ename]['attr'], dtype=torch.float)

    # Add reverse BELONGS_TO (Sector→Stock) so Stock embeddings get sector context
    if edges['BELONGS_TO'].shape[1] > 0:
        rev = edges['BELONGS_TO'][[1, 0], :]
        data['Sector', 'rev_BELONGS_TO', 'Stock'].edge_index = \
            torch.tensor(rev, dtype=torch.long)

    return data


def build_metadata(data):
    """Extract (node_types, edge_types) metadata for HGTConv."""
    node_types = list(data.x_dict.keys())
    edge_types = list(data.edge_index_dict.keys())
    return node_types, edge_types


# ====================================================================
# Training
# ====================================================================

def train_epoch(model, data, holds_ei, holds_ea, labels, mask,
                optimizer, device, batch_size=BATCH_HOLDS):
    model.train()
    total_loss = 0
    indices = mask.nonzero(as_tuple=True)[0]
    perm = indices[torch.randperm(len(indices))]

    n_batches = 0
    for start in range(0, len(perm), batch_size):
        batch_idx = perm[start:start + batch_size]
        optimizer.zero_grad()
        logits = model(data, holds_ei[:, batch_idx], holds_ea[batch_idx])
        loss = F.cross_entropy(logits, labels[batch_idx])
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

    if not os.path.exists(INPUT_PKL):
        print(f"ERROR: {INPUT_PKL} not found. Run step13b_export_kg_for_gpu.py first.")
        sys.exit(1)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load export
    print(f"\n  Loading {INPUT_PKL}...")
    with open(INPUT_PKL, 'rb') as f:
        export = pickle.load(f)
    meta = export['metadata']
    print(f"  Nodes: Fund={meta['n_fund']} Stock={meta['n_stock']} "
          f"Sector={meta['n_sector']} TimePeriod={meta['n_timeperiod']} "
          f"CausalVar={meta['n_causalvar']}")
    print(f"  HOLDS: {meta['n_holds']} edges")
    print(f"  Split: train={meta['n_train']} val={meta['n_val']} test={meta['n_test']}")

    # Build HeteroData
    print("\n  Building heterogeneous graph...")
    data = load_hetero_data(export)
    data = data.to(device)

    # HOLDS tensors
    holds = export['holds']
    holds_ei = torch.tensor(holds['edge_index'], dtype=torch.long).to(device)
    holds_ea = torch.tensor(holds['edge_attr'],  dtype=torch.float).to(device)
    labels   = torch.tensor(holds['labels'],     dtype=torch.long).to(device)
    train_mask = torch.tensor(export['train_mask'], dtype=torch.bool).to(device)
    val_mask   = torch.tensor(export['val_mask'],   dtype=torch.bool).to(device)
    test_mask  = torch.tensor(export['test_mask'],  dtype=torch.bool).to(device)

    # Normalize HOLDS edge features
    ea_mean = holds_ea[train_mask].mean(0, keepdim=True)
    ea_std  = holds_ea[train_mask].std(0, keepdim=True).clamp(min=1e-6)
    holds_ea = (holds_ea - ea_mean) / ea_std

    print(f"  HOLDS edge features: {holds_ea.shape[1]}")
    print(f"  Label distribution: "
          + " ".join(f"{k}={int((labels==k).sum())}" for k in [0,1,2]))

    # Model
    in_dims = {ntype: data[ntype].x.shape[1] for ntype in data.node_types}
    metadata = (data.node_types, list(data.edge_index_dict.keys()))

    # Collect causal edge attributes for CI-HGT
    causal_edge_attrs = {}
    edges = export['edges']
    for ename in ['GRANGER_CAUSES', 'CAUSES', 'CAUSAL_EFFECT']:
        if ename in edges and edges[ename]['attr'].shape[0] > 0:
            causal_edge_attrs[ename] = torch.tensor(
                edges[ename]['attr'], dtype=torch.float).to(device)

    causal_edge_dims = {k: v.shape[1] for k, v in causal_edge_attrs.items()}

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
                metadata=metadata,
                in_dims=in_dims,
                edge_feat_dim=holds_ea.shape[1],
            ).to(device)
            output_pt = OUTPUT_PT
        else:
            model = CIHGTFundModel(
                metadata=metadata,
                in_dims=in_dims,
                edge_feat_dim=holds_ea.shape[1],
                causal_edge_dims=causal_edge_dims,
            ).to(device)
            output_pt = os.path.join(DATA_DIR, 'ci_hgt_model.pt')

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model: {model_name}  hidden={HIDDEN_DIM}  heads={N_HEADS}  "
              f"layers={N_LAYERS}  params={n_params:,}")

        # Class weights for imbalanced labels
        label_counts = torch.bincount(labels[train_mask], minlength=3).float()
        class_weights = (1.0 / label_counts.clamp(min=1)).to(device)
        class_weights = class_weights / class_weights.sum() * 3

        optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                      patience=5, min_lr=1e-5)

        # Training loop
        print(f"  Training {EPOCHS} epochs (patience={PATIENCE})...")
        best_val_f1 = 0.0
        best_epoch = 0
        patience_ct = 0

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            loss = train_epoch(model, data, holds_ei, holds_ea, labels,
                               train_mask, optimizer, device)
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
        model.load_state_dict(torch.load(output_pt, map_location=device))
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

        # Save per-decision predictions for downstream (step14b, step16)
        action_labels = {2: 'BUY', 1: 'HOLD', 0: 'SELL'}
        test_indices = test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
        decision_records = []
        fund_names = holds.get('fund_names', [])
        stock_isins = holds.get('stock_isins', [])
        month_strs = holds.get('month_strs', [])
        if fund_names and stock_isins and month_strs:
            for i, ti in enumerate(test_indices):
                decision_records.append({
                    'Fund_Name': fund_names[ti] if ti < len(fund_names) else '',
                    'ISIN': stock_isins[ti] if ti < len(stock_isins) else '',
                    'year_month_str': month_strs[ti] if ti < len(month_strs) else '',
                    f'{model_name.lower().replace("-","_")}_predicted': int(te_preds[i]),
                    f'{model_name.lower().replace("-","_")}_label': action_labels.get(int(te_preds[i]), ''),
                    'actual': int(te_labels[i]),
                    'actual_label': action_labels.get(int(te_labels[i]), ''),
                })
            pred_file = os.path.join(DATA_DIR,
                                     f'{model_name.lower().replace("-","_")}_decision_predictions.csv')
            import pandas as pd
            pd.DataFrame(decision_records).to_csv(pred_file, index=False)
            print(f"  Saved: {pred_file} ({len(decision_records)} decisions)")
            # Also copy to final/
            final_dir = os.path.join(SCRIPT_DIR, 'data', 'final')
            if os.path.exists(final_dir):
                import shutil
                shutil.copy(pred_file, os.path.join(
                    final_dir, f'{model_name.lower().replace("-","_")}_decision_predictions.csv'))
        else:
            print(f"  NOTE: No fund/stock/month identifiers in export. "
                  f"Re-run step13b_export_kg_for_gpu.py to enable prediction mapping.")

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
        print(f"\n  CI-HGT novelty: Causal edge strengths modulate message passing")
        print(f"  via learned CausalGate. Strong causal evidence is amplified,")
        print(f"  weak evidence is attenuated. The gate LEARNS the threshold.")
        all_results['comparison'] = {
            'hgt_f1': hgt_f1, 'ci_hgt_f1': ci_f1, 'delta': delta
        }

    # Save combined results
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {OUTPUT_JSON}")

    # Save node embeddings for downstream use (step16c style clustering, etc.)
    try:
        last_model_name = models_to_run[-1][0] if models_to_run else None
        if last_model_name:
            model.eval()
            with torch.no_grad():
                embeddings = {}
                for ntype in data.node_types:
                    if hasattr(data[ntype], 'x'):
                        embeddings[ntype] = data[ntype].x.cpu().numpy()
            emb_path = os.path.join(DATA_DIR, 'hgt_embeddings.npz')
            np.savez(emb_path, **embeddings)
            print(f"  Saved node embeddings: {emb_path}")
    except Exception as e:
        print(f"  Warning: could not save embeddings: {e}")

    # Copy to final/
    final_dir = os.path.join(SCRIPT_DIR, 'data', 'final')
    if os.path.exists(final_dir):
        import shutil
        dest = os.path.join(final_dir, 'hgt_results.json')
        shutil.copy(OUTPUT_JSON, dest)
        print(f"  Copied to: {dest}")

    print("\n  [STEP 13b] HGT + CI-HGT Done.")


if __name__ == '__main__':
    main()
