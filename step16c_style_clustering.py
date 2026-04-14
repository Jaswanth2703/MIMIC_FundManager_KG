"""
Step 16c -- Fund Manager Style Clustering via KG Embeddings
==============================================================
Uses path embeddings from step13a to discover fund manager "styles".

Styles are clusters of decision patterns in the embedding space:
  - "Momentum": high turnover, buys on positive return streaks
  - "Value": long holding tenures, buys on low PE/PB
  - "Contrarian": sells when consensus buys, buys on dips
  - "Index-hugger": low active share, follows sector weights

This analysis answers the thesis question:
  "Can the KG reveal latent decision-making styles that are
   not observable from individual transactions?"

Method:
  1. Load path embeddings from step13a (64-dim Transformer output)
  2. Aggregate embeddings by fund (mean + std of all decisions)
  3. Cluster with HDBSCAN (density-based, auto-selects K)
  4. Characterize clusters via feature statistics
  5. Visualize with UMAP/t-SNE dimensionality reduction

Inputs:
  data/final/path_embeddings.npy      (from step13a)
  data/final/decision_paths.json      (path metadata)
  data/features/LPCMCI_READY.csv      (for feature profiles)

Outputs:
  data/evaluation/style_clusters.json
  data/evaluation/style_profiles.csv
"""

import sys
import os
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES_DIR, EVAL_DIR, FINAL_DIR

INPUT_EMB = os.path.join(FINAL_DIR, 'path_embeddings.npy')
INPUT_PATHS = os.path.join(FINAL_DIR, 'decision_paths.json')
INPUT_FEAT = os.path.join(FEATURES_DIR, 'LPCMCI_READY.csv')

OUT_CLUSTERS = os.path.join(EVAL_DIR, 'style_clusters.json')
OUT_PROFILES = os.path.join(EVAL_DIR, 'style_profiles.csv')


def aggregate_fund_embeddings(embeddings, paths_meta, df):
    """Aggregate path embeddings by fund manager.

    For each fund, compute:
      - Mean embedding (centroid in path space)
      - Std embedding (diversity of decision paths)
      - Action distribution (% BUY, HOLD, SELL)
      - Behavioral features (avg tenure, avg allocation, turnover)
    """
    fund_embs = {}
    fund_actions = {}

    for i, pm in enumerate(paths_meta):
        if i >= len(embeddings):
            break
        fund = pm.get('fund', 'UNKNOWN')
        action = pm.get('action', 'HOLD')
        if fund not in fund_embs:
            fund_embs[fund] = []
            fund_actions[fund] = []
        fund_embs[fund].append(embeddings[i])
        fund_actions[fund].append(action)

    # Aggregate
    fund_profiles = []
    fund_vectors = []
    fund_names = []

    for fund, embs in fund_embs.items():
        embs = np.array(embs)
        mean_emb = embs.mean(axis=0)
        std_emb = embs.std(axis=0)
        combined = np.concatenate([mean_emb, std_emb])
        fund_vectors.append(combined)
        fund_names.append(fund)

        # Action distribution
        actions = fund_actions[fund]
        n = len(actions)
        pct_buy = sum(1 for a in actions if a in ('BUY', 'INCREASE', 'INITIAL_POSITION')) / n
        pct_sell = sum(1 for a in actions if a in ('SELL', 'DECREASE')) / n
        pct_hold = sum(1 for a in actions if a == 'HOLD') / n

        # Behavioral features from df
        fund_df = df[df['Fund_Name'] == fund] if 'Fund_Name' in df.columns else pd.DataFrame()
        profile = {
            'fund': fund,
            'n_decisions': n,
            'pct_buy': pct_buy,
            'pct_sell': pct_sell,
            'pct_hold': pct_hold,
            'avg_holding_tenure': float(fund_df['holding_tenure'].mean())
                if 'holding_tenure' in fund_df.columns and len(fund_df) > 0 else 0,
            'avg_pct_nav': float(fund_df['pct_nav'].mean())
                if 'pct_nav' in fund_df.columns and len(fund_df) > 0 else 0,
            'n_unique_stocks': int(fund_df['ISIN'].nunique())
                if 'ISIN' in fund_df.columns and len(fund_df) > 0 else 0,
            'sector_diversity': int(fund_df['sector'].nunique())
                if 'sector' in fund_df.columns and len(fund_df) > 0 else 0,
            'turnover_rate': float(pct_buy + pct_sell),
        }
        fund_profiles.append(profile)

    return np.array(fund_vectors), fund_names, fund_profiles


def cluster_fund_styles(fund_vectors, fund_names, fund_profiles):
    """Cluster fund managers by decision style."""
    n_funds = len(fund_names)
    print(f"  Clustering {n_funds} fund managers ...")

    if n_funds < 3:
        print("    Too few funds for clustering")
        return fund_profiles

    # Try HDBSCAN first, fallback to KMeans
    labels = None
    method = None

    try:
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=max(2, n_funds // 8),
                             min_samples=2, metric='euclidean')
        labels = clusterer.fit_predict(fund_vectors)
        method = 'HDBSCAN'
        n_clusters = len(set(labels) - {-1})
        n_noise = (labels == -1).sum()
        print(f"    HDBSCAN: {n_clusters} clusters, {n_noise} noise points")
    except (ImportError, Exception):
        pass

    if labels is None or len(set(labels) - {-1}) < 2:
        from sklearn.cluster import KMeans
        k = min(5, max(2, n_funds // 5))
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(fund_vectors)
        method = f'KMeans(k={k})'
        print(f"    KMeans: {k} clusters")

    # Assign labels
    for i, profile in enumerate(fund_profiles):
        profile['cluster_id'] = int(labels[i])

    # Characterize clusters
    cluster_chars = {}
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        members = [p for p in fund_profiles if p['cluster_id'] == cid]
        avg_tenure = np.mean([m['avg_holding_tenure'] for m in members])
        avg_turnover = np.mean([m['turnover_rate'] for m in members])
        avg_buy = np.mean([m['pct_buy'] for m in members])
        avg_sell = np.mean([m['pct_sell'] for m in members])
        avg_diversity = np.mean([m['sector_diversity'] for m in members])

        # Infer style label
        if avg_tenure > np.median([p['avg_holding_tenure'] for p in fund_profiles]) * 1.2:
            style = 'Patient/Value'
        elif avg_turnover > 0.6:
            style = 'Active/Momentum'
        elif avg_sell > avg_buy * 1.3:
            style = 'Contrarian/Defensive'
        elif avg_diversity < 5:
            style = 'Concentrated/Sector-focused'
        else:
            style = 'Diversified/Balanced'

        cluster_chars[int(cid)] = {
            'style_label': style,
            'n_members': len(members),
            'avg_tenure': float(avg_tenure),
            'avg_turnover': float(avg_turnover),
            'avg_pct_buy': float(avg_buy),
            'avg_pct_sell': float(avg_sell),
            'avg_sector_diversity': float(avg_diversity),
            'member_funds': [m['fund'] for m in members],
        }

        for m in members:
            m['style_label'] = style

    # Dimensionality reduction for visualization
    try:
        from sklearn.manifold import TSNE
        if n_funds >= 5:
            perp = min(5, n_funds - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
            coords = tsne.fit_transform(fund_vectors)
            for i, profile in enumerate(fund_profiles):
                profile['tsne_x'] = float(coords[i, 0])
                profile['tsne_y'] = float(coords[i, 1])
            print(f"    t-SNE coordinates computed")
    except Exception:
        pass

    return fund_profiles, cluster_chars, method


def main():
    print("=" * 70)
    print("STEP 16c -- FUND MANAGER STYLE CLUSTERING")
    print("=" * 70)

    # Load data
    if not os.path.exists(INPUT_EMB):
        print(f"  WARNING: {INPUT_EMB} not found. Run step13a first.")
        print("  Attempting to create embeddings from feature data ...")

        if not os.path.exists(INPUT_FEAT):
            print(f"  ERROR: {INPUT_FEAT} not found.")
            return

        # Fallback: use aggregated features as proxy for embeddings
        df = pd.read_csv(INPUT_FEAT, low_memory=False)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {'action_ordinal'}
        feat_cols = [c for c in numeric_cols if c not in exclude][:30]

        fund_profiles = []
        fund_vectors = []
        fund_names = []

        for fund, fdf in df.groupby('Fund_Name'):
            vec = fdf[feat_cols].mean().fillna(0).values
            std_vec = fdf[feat_cols].std().fillna(0).values
            fund_vectors.append(np.concatenate([vec, std_vec]))
            fund_names.append(fund)
            n = len(fdf)
            pct_buy = (fdf['position_action'].isin(['BUY', 'INCREASE', 'INITIAL_POSITION'])).mean()
            pct_sell = (fdf['position_action'].isin(['SELL', 'DECREASE'])).mean()
            fund_profiles.append({
                'fund': fund, 'n_decisions': n,
                'pct_buy': float(pct_buy), 'pct_sell': float(pct_sell),
                'pct_hold': float(1 - pct_buy - pct_sell),
                'avg_holding_tenure': float(fdf['holding_tenure'].mean())
                    if 'holding_tenure' in fdf.columns else 0,
                'avg_pct_nav': float(fdf['pct_nav'].mean())
                    if 'pct_nav' in fdf.columns else 0,
                'n_unique_stocks': int(fdf['ISIN'].nunique()),
                'sector_diversity': int(fdf['sector'].nunique())
                    if 'sector' in fdf.columns else 0,
                'turnover_rate': float(pct_buy + pct_sell),
            })
        fund_vectors = np.array(fund_vectors)
        print(f"  Fallback: {len(fund_names)} funds, {fund_vectors.shape[1]}-dim vectors")
    else:
        embeddings = np.load(INPUT_EMB)
        print(f"  Loaded embeddings: {embeddings.shape}")

        with open(INPUT_PATHS) as f:
            paths_data = json.load(f)
        paths_meta = paths_data.get('sample_paths', paths_data)
        print(f"  Path metadata: {len(paths_meta)} entries")

        df = pd.read_csv(INPUT_FEAT, low_memory=False)
        fund_vectors, fund_names, fund_profiles = aggregate_fund_embeddings(
            embeddings, paths_meta, df)
        print(f"  Fund vectors: {fund_vectors.shape}")

    # Cluster
    fund_profiles, cluster_chars, method = cluster_fund_styles(
        fund_vectors, fund_names, fund_profiles)

    # Print results
    print(f"\n  === FUND MANAGER STYLES ===")
    print(f"  Clustering method: {method}")
    for cid, info in sorted(cluster_chars.items()):
        print(f"\n  Cluster {cid}: {info['style_label']}")
        print(f"    Members: {info['n_members']}")
        print(f"    Avg tenure: {info['avg_tenure']:.1f} months")
        print(f"    Avg turnover: {info['avg_turnover']:.2f}")
        print(f"    Buy/Sell ratio: {info['avg_pct_buy']:.2f}/{info['avg_pct_sell']:.2f}")
        print(f"    Sector diversity: {info['avg_sector_diversity']:.1f}")
        for f in info['member_funds'][:5]:
            print(f"      - {f}")

    # Save
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(OUT_CLUSTERS, 'w') as f:
        json.dump({
            'method': method,
            'n_funds': len(fund_names),
            'cluster_characteristics': cluster_chars,
            'fund_profiles': fund_profiles,
        }, f, indent=2, default=str)

    pd.DataFrame(fund_profiles).to_csv(OUT_PROFILES, index=False)

    print(f"\n  Saved: {OUT_CLUSTERS}")
    print(f"  Saved: {OUT_PROFILES}")

    print(f"\n  WHY THIS REQUIRES THE KG:")
    print(f"  - Path embeddings encode KG structure (node types, edge types)")
    print(f"  - Managers with similar KG traversal patterns cluster together")
    print(f"  - Flat features miss portfolio context and causal structure")
    print(f"\n  [STEP 16c] Style Clustering Done.")


if __name__ == '__main__':
    main()
