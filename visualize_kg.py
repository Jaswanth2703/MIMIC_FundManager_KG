"""
Knowledge Graph Visualization
===============================
Generates publication-quality visualizations of both the Temporal KG
and Causal KG from Neo4j for research paper / presentation.

Output: explanations/ folder
  - temporal_kg_schema.png     : Schema diagram of Temporal KG
  - causal_kg_schema.png       : Schema diagram of Causal KG
  - causal_network.png         : Top causal links as directed network
  - causal_heatmap.png         : Cause-effect strength heatmap
  - lag_distribution.png       : Distribution of causal lags
  - edge_type_distribution.png : Pie chart of edge types
  - sector_causal_drivers.png  : Which macro factors drive which sectors
  - portfolio_allocation.png   : Final portfolio treemap
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from collections import defaultdict, Counter

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAUSAL_DIR = os.path.join(BASE_DIR, 'data', 'causal_output')
FINAL_DIR = os.path.join(BASE_DIR, 'data', 'final')
EVAL_DIR = os.path.join(BASE_DIR, 'data', 'evaluation')
OUT_DIR = os.path.join(BASE_DIR, 'explanations')
os.makedirs(OUT_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.facecolor': 'white',
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

COLORS = {
    'macro': '#2196F3',
    'technical': '#FF9800',
    'sentiment': '#4CAF50',
    'fund_action': '#F44336',
    'fundamental': '#9C27B0',
    'sector': '#00BCD4',
    'unknown': '#607D8B',
}

EDGE_COLORS = {
    '-->': '#1B5E20',
    'o->': '#4CAF50',
    '<->': '#F44336',
    'o-o': '#9E9E9E',
    '<--': '#FF9800',
    '<-o': '#FFC107',
    'x-x': '#BDBDBD',
}


def categorize_variable(name):
    """Categorize a causal variable by its name."""
    name_l = name.lower()
    macro_kw = ['nifty', 'vix', 'crude', 'gold', 'usd', 'sp500', 'yield',
                'cpi', 'repo', 'gdp', 'brent', 'inflation']
    tech_kw = ['rsi', 'macd', 'momentum', 'volatility', 'return', 'beta',
               'relative_strength']
    sent_kw = ['sentiment', 'positive', 'negative', 'news']
    action_kw = ['buy', 'sell', 'net_action', 'alloc', 'wt_', 'flow_',
                 'stock_count', 'avg_tenure', 'avg_consensus', 'fund_count']
    fund_kw = ['pe', 'pb', 'eps', 'market_cap', 'roe', 'debt']

    for kw in macro_kw:
        if kw in name_l:
            return 'macro'
    for kw in tech_kw:
        if kw in name_l:
            return 'technical'
    for kw in sent_kw:
        if kw in name_l:
            return 'sentiment'
    for kw in action_kw:
        if kw in name_l:
            return 'fund_action'
    for kw in fund_kw:
        if kw in name_l:
            return 'fundamental'
    return 'unknown'


# ====================================================================
# 1. Temporal KG Schema Diagram
# ====================================================================
def draw_temporal_kg_schema():
    """Draw the Temporal KG schema as a publication-quality diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 11)
    ax.axis('off')
    ax.set_title('Temporal Knowledge Graph Schema', fontsize=18, fontweight='bold', pad=20)

    # Node definitions: (label, x, y, color, properties)
    nodes = [
        ('Fund', 2, 9, '#2196F3', 'name, type, aum_avg'),
        ('Stock', 8, 9, '#4CAF50', 'isin, name, symbol, sector'),
        ('Sector', 14, 9, '#FF9800', 'name'),
        ('TimePeriod', 2, 5, '#9C27B0', 'id, year, month, quarter'),
        ('MarketRegime', 2, 1, '#F44336', 'id, regime_type, vix_level'),
        ('FundSnapshot', 5, 5, '#00BCD4', 'fund_name, month, total_stocks'),
        ('StockSnapshot', 11, 5, '#795548', 'isin, month, pe, pb, rsi...'),
    ]

    for label, x, y, color, props in nodes:
        box = FancyBboxPatch((x-1.3, y-0.5), 2.6, 1.8, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', alpha=0.85, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y+0.35, f':{label}', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
        ax.text(x, y-0.15, props, ha='center', va='center',
                fontsize=7, color='white', style='italic',
                wrap=True)

    # Relationships: (from_xy, to_xy, label, color, style)
    rels = [
        ((3.3, 9.3), (6.7, 9.3), 'HOLDS\n{month, pct_nav,\nposition_action...}', '#1B5E20'),
        ((3.3, 8.7), (6.7, 8.7), 'EXITED\n{month, last_pct_nav}', '#B71C1C'),
        ((9.3, 9), (12.7, 9), 'BELONGS_TO', '#E65100'),
        ((2, 8.2), (2, 6.3), 'ACTIVE_IN', '#6A1B9A'),
        ((2, 4.2), (2, 2.3), 'IN_REGIME', '#C62828'),
        ((5, 6.3), (3.0, 8.2), 'OF_FUND', '#00838F'),
        ((5, 4.5), (2.7, 5.3), 'AT_TIME', '#00838F'),
        ((11, 6.3), (8.7, 8.2), 'OF_STOCK', '#4E342E'),
        ((11, 4.5), (3.3, 5.0), 'AT_TIME', '#4E342E'),
    ]

    for (x1, y1), (x2, y2), label, color in rels:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.15, label, ha='center', va='center',
                fontsize=7, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=color, alpha=0.9))

    # TimePeriod NEXT chain
    ax.annotate('', xy=(3.3, 5.8), xytext=(3.3, 5.2),
                arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=1.5,
                                connectionstyle='arc3,rad=-0.5'))
    ax.text(3.9, 5.5, 'NEXT', fontsize=7, color='#6A1B9A', fontweight='bold')

    path = os.path.join(OUT_DIR, 'temporal_kg_schema.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ====================================================================
# 2. Causal KG Schema Diagram
# ====================================================================
def draw_causal_kg_schema():
    """Draw the Causal KG schema diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-1, 11)
    ax.axis('off')
    ax.set_title('Causal Knowledge Graph Schema', fontsize=18, fontweight='bold', pad=20)

    nodes = [
        ('CausalVariable', 4, 8, '#2196F3', 'name, category,\ndescription, unit'),
        ('CausalVariable', 12, 8, '#2196F3', 'name, category,\ndescription, unit'),
        ('CausalAnalysis', 8, 4, '#FF9800', 'name, level, alpha,\nn_variables, n_links'),
        ('DomainConcept', 4, 1, '#4CAF50', 'name, type'),
        ('Stock / Sector', 12, 1, '#9C27B0', 'isin / name'),
    ]

    for label, x, y, color, props in nodes:
        box = FancyBboxPatch((x-1.5, y-0.6), 3.0, 2.0, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', alpha=0.85, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y+0.4, f':{label}', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
        ax.text(x, y-0.2, props, ha='center', va='center',
                fontsize=8, color='white', style='italic')

    rels = [
        ((5.5, 8.5), (10.5, 8.5), 'CAUSES\n{lag, strength, p_value,\nedge_type, confidence}', '#1B5E20'),
        ((5.5, 7.5), (10.5, 7.5), 'LATENT_COMMON_CAUSE\n{lag, strength}', '#F44336'),
        ((4, 7.0), (7, 5.5), 'DISCOVERED_IN', '#E65100'),
        ((4, 3.5), (4, 1.8), 'REPRESENTS', '#2E7D32'),
        ((5.5, 1), (10.5, 1), 'INFLUENCES', '#1B5E20'),
        ((12, 7.0), (12, 2.5), 'MEASURED_FOR', '#6A1B9A'),
    ]

    for (x1, y1), (x2, y2), label, color in rels:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.2, label, ha='center', va='center',
                fontsize=8, color=color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=color, alpha=0.9))

    # Edge type legend
    legend_x, legend_y = 0.5, 5
    ax.text(legend_x, legend_y+1, 'PAG Edge Types:', fontsize=10, fontweight='bold')
    edge_types = [
        ('-->  Direct causal', '#1B5E20'),
        ('<->  Latent confounder', '#F44336'),
        ('o->  Probable causal', '#4CAF50'),
        ('o-o  Uncertain', '#9E9E9E'),
    ]
    for i, (txt, col) in enumerate(edge_types):
        ax.text(legend_x, legend_y - i*0.5, txt, fontsize=9, color=col, fontweight='bold')

    path = os.path.join(OUT_DIR, 'causal_kg_schema.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ====================================================================
# 3. Causal Network Graph (Top Links)
# ====================================================================
def draw_causal_network():
    """Draw top causal links as a directed network."""
    try:
        import networkx as nx
    except ImportError:
        print("  networkx not installed, skipping causal network.")
        return

    csv_path = os.path.join(CAUSAL_DIR, 'causal_links_lpcmci.csv')
    if not os.path.exists(csv_path):
        print(f"  {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    # Filter: strong links, no self-loops
    strong = df[(df['cause'] != df['effect']) &
                (df['strength'] > 0.2) &
                (df['edge_type'].isin(['-->', 'o->']))].copy()
    strong = strong.nlargest(60, 'strength')

    if strong.empty:
        print("  No strong links to plot.")
        return

    G = nx.DiGraph()
    for _, row in strong.iterrows():
        G.add_edge(row['cause'], row['effect'],
                   weight=row['strength'],
                   edge_type=row['edge_type'],
                   lag=row['lag'])

    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    ax.set_title('Top 60 Causal Links (strength > 0.2, directed)',
                 fontsize=16, fontweight='bold')

    pos = nx.spring_layout(G, k=2.5, iterations=80, seed=42)

    # Node colors by category
    node_colors = []
    for node in G.nodes():
        cat = categorize_variable(node)
        node_colors.append(COLORS.get(cat, COLORS['unknown']))

    # Edge widths by strength
    edge_widths = [G[u][v]['weight'] * 4 for u, v in G.edges()]
    edge_colors = [EDGE_COLORS.get(G[u][v]['edge_type'], '#333333') for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=800, alpha=0.9, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors,
                           width=edge_widths, alpha=0.7,
                           arrows=True, arrowsize=20,
                           connectionstyle='arc3,rad=0.1')

    # Labels
    labels = {}
    for node in G.nodes():
        short = node[:18] + '..' if len(node) > 20 else node
        labels[node] = short
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7,
                            font_weight='bold')

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=l.replace('_', ' ').title())
                      for l, c in COLORS.items() if l != 'unknown']
    ax.legend(handles=legend_patches, loc='upper left', fontsize=10,
              title='Variable Category', title_fontsize=11)

    ax.axis('off')
    path = os.path.join(OUT_DIR, 'causal_network.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ====================================================================
# 4. Causal Heatmap (Macro -> Sector)
# ====================================================================
def draw_causal_heatmap():
    """Draw cause-effect strength heatmap for macro -> sector links."""
    csv_path = os.path.join(CAUSAL_DIR, 'causal_links_lpcmci.csv')
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    # Filter to L1 (macro -> sector) analysis
    l1 = df[df['analysis'].str.contains('L1', na=False)].copy()
    if l1.empty:
        l1 = df[df['analysis'].str.contains('macro', case=False, na=False)]
    if l1.empty:
        print("  No L1/macro links for heatmap.")
        return

    # Identify macro causes and sector effects
    macro_vars = [v for v in l1['cause'].unique() if categorize_variable(v) == 'macro']
    sector_vars = [v for v in l1['effect'].unique()
                   if 'alloc' in v.lower() or 'buy' in v.lower() or 'sell' in v.lower()]

    if not macro_vars or not sector_vars:
        print("  Not enough macro->sector links for heatmap.")
        return

    macro_vars = macro_vars[:12]
    sector_vars = sector_vars[:15]

    # Build matrix
    matrix = pd.DataFrame(0.0, index=macro_vars, columns=sector_vars)
    for _, row in l1.iterrows():
        if row['cause'] in macro_vars and row['effect'] in sector_vars:
            matrix.loc[row['cause'], row['effect']] = max(
                matrix.loc[row['cause'], row['effect']], row['strength']
            )

    # Remove empty rows/cols
    matrix = matrix.loc[matrix.sum(axis=1) > 0, matrix.sum(axis=0) > 0]
    if matrix.empty:
        return

    fig, ax = plt.subplots(1, 1, figsize=(max(12, len(matrix.columns)*0.9),
                                           max(6, len(matrix.index)*0.6)))
    im = ax.imshow(matrix.values, cmap='YlOrRd', aspect='auto', vmin=0)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([c[:20] for c in matrix.columns], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)
    ax.set_title('Macro -> Sector Causal Strength Heatmap', fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Causal Strength', shrink=0.8)

    # Annotate cells
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            val = matrix.values[i, j]
            if val > 0.05:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color='white' if val > 0.4 else 'black')

    path = os.path.join(OUT_DIR, 'causal_heatmap.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ====================================================================
# 5. Lag Distribution
# ====================================================================
def draw_lag_distribution():
    """Bar chart of causal link lag distribution."""
    csv_path = os.path.join(CAUSAL_DIR, 'causal_links_lpcmci.csv')
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    lag_counts = df['lag'].value_counts().sort_index()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bars = ax.bar(lag_counts.index, lag_counts.values,
                  color=['#1565C0' if l == 0 else '#42A5F5' for l in lag_counts.index],
                  edgecolor='black', linewidth=0.5)

    for bar, count in zip(bars, lag_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{count}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Time Lag (months)', fontsize=13)
    ax.set_ylabel('Number of Causal Links', fontsize=13)
    ax.set_title('Distribution of Causal Time Lags', fontsize=15, fontweight='bold')
    ax.set_xticks(range(7))
    ax.set_xticklabels(['0\n(Contemp.)', '1', '2', '3', '4', '5', '6'])

    path = os.path.join(OUT_DIR, 'lag_distribution.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ====================================================================
# 6. Edge Type Distribution
# ====================================================================
def draw_edge_type_distribution():
    """Pie chart of PAG edge type distribution."""
    csv_path = os.path.join(CAUSAL_DIR, 'causal_links_lpcmci.csv')
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    et_counts = df['edge_type'].value_counts()

    labels_map = {
        '-->': 'Direct Causal (-->)',
        'o->': 'Probable Causal (o->)',
        '<->': 'Latent Confounder (<->)',
        'o-o': 'Uncertain (o-o)',
        '<--': 'Reverse Causal (<--)',
        '<-o': 'Probable Reverse (<-o)',
        'x-x': 'Weak (x-x)',
    }

    labels = [labels_map.get(et, et) for et in et_counts.index]
    colors = [EDGE_COLORS.get(et, '#999999') for et in et_counts.index]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        et_counts.values, labels=labels, colors=colors,
        autopct='%1.1f%%', pctdistance=0.85, startangle=90,
        textprops={'fontsize': 10}
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight('bold')

    ax.set_title('PAG Edge Type Distribution\n(LPCMCI Causal Discovery Results)',
                 fontsize=15, fontweight='bold')

    path = os.path.join(OUT_DIR, 'edge_type_distribution.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ====================================================================
# 7. Portfolio Allocation Chart
# ====================================================================
def draw_portfolio_allocation():
    """Bar chart of final portfolio allocations."""
    csv_path = os.path.join(FINAL_DIR, 'constructed_portfolio.csv')
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    if 'weight' not in df.columns:
        return

    df = df.sort_values('weight', ascending=True)

    name_col = 'stock_name' if 'stock_name' in df.columns else 'isin'
    names = df[name_col].apply(lambda x: str(x)[:25])
    weights = df['weight'] * 100  # percentage

    # Color by sector
    sector_colors = {}
    palette = plt.cm.Set3(np.linspace(0, 1, 12))
    if 'sector' in df.columns:
        unique_sectors = df['sector'].unique()
        for i, sec in enumerate(unique_sectors):
            sector_colors[sec] = palette[i % len(palette)]
        bar_colors = [sector_colors.get(s, '#999999') for s in df['sector']]
    else:
        bar_colors = '#2196F3'

    fig, ax = plt.subplots(1, 1, figsize=(12, max(6, len(df)*0.4)))
    bars = ax.barh(range(len(df)), weights.values, color=bar_colors,
                   edgecolor='black', linewidth=0.5, height=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(names.values, fontsize=9)
    ax.set_xlabel('Portfolio Weight (%)', fontsize=13)
    ax.set_title('Constructed Portfolio Allocation', fontsize=15, fontweight='bold')

    # Annotate with scores
    if 'buy_score' in df.columns:
        for i, (w, s) in enumerate(zip(weights.values, df['buy_score'].values)):
            ax.text(w + 0.2, i, f'{w:.1f}% (score={s:.3f})',
                    va='center', fontsize=8)

    # Sector legend
    if sector_colors:
        legend_patches = [mpatches.Patch(color=c, label=s[:20])
                          for s, c in sector_colors.items()]
        ax.legend(handles=legend_patches, loc='lower right', fontsize=8,
                  title='Sector', title_fontsize=9)

    path = os.path.join(OUT_DIR, 'portfolio_allocation.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ====================================================================
# 8. Architecture / Methodology Diagram
# ====================================================================
def draw_methodology_diagram():
    """Draw the full pipeline methodology diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(22, 14))
    ax.set_xlim(-0.5, 22)
    ax.set_ylim(-0.5, 14.5)
    ax.axis('off')
    ax.set_title('Fund Manager Knowledge Graph: System Architecture',
                 fontsize=20, fontweight='bold', pad=15)

    def draw_box(x, y, w, h, text, color, fontsize=9, subtext=None):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor='#333333', alpha=0.9, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2 + (0.15 if subtext else 0), text,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='white')
        if subtext:
            ax.text(x + w/2, y + h/2 - 0.25, subtext,
                    ha='center', va='center', fontsize=7,
                    color='white', style='italic')

    def draw_arrow(x1, y1, x2, y2, color='#333333', text=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        if text:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.15, text, ha='center', fontsize=7,
                    color=color, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor=color, alpha=0.8, pad=1))

    # Phase headers
    phase_bg = [('#E3F2FD', 0, 10, 22, 4), ('#E8F5E9', 0, 5.5, 22, 4.2), ('#FFF3E0', 0, 0.5, 22, 4.8)]
    for color, px, py, pw, ph in phase_bg:
        rect = plt.Rectangle((px, py), pw, ph, facecolor=color, alpha=0.3, edgecolor='#999', linewidth=1)
        ax.add_patch(rect)
    ax.text(0.3, 13.5, 'PHASE 1: Data Acquisition & Feature Engineering', fontsize=13, fontweight='bold', color='#1565C0')
    ax.text(0.3, 9.3, 'PHASE 2: Causal Discovery & Knowledge Graph', fontsize=13, fontweight='bold', color='#2E7D32')
    ax.text(0.3, 4.9, 'PHASE 3: Evaluation, Reasoning & Decisions', fontsize=13, fontweight='bold', color='#E65100')

    # Phase 1 boxes
    draw_box(0.5, 11.5, 2.5, 1.2, 'Step 0-1', '#1565C0', subtext='Portfolio Load\nISIN Mapping')
    draw_box(3.5, 11.5, 2.5, 1.2, 'Step 2-3', '#1976D2', subtext='Fundamentals\nOHLCV Fetch')
    draw_box(6.5, 11.5, 2.5, 1.2, 'Step 4', '#1E88E5', subtext='Technical\nIndicators')
    draw_box(9.5, 11.5, 2.5, 1.2, 'Step 5', '#42A5F5', subtext='FinBERT\nSentiment')
    draw_box(12.5, 11.5, 2.5, 1.2, 'Step 6', '#64B5F6', subtext='Macro\nIndicators')
    draw_box(15.5, 11.5, 2.5, 1.2, 'Step 7-8', '#1565C0', subtext='Feature Eng.\nLPCMCI_READY')

    # Phase 1 arrows
    for x in [3.0, 6.0, 9.0, 12.0, 15.0]:
        draw_arrow(x, 12.1, x+0.5, 12.1)

    # Data output
    draw_box(18.5, 11.5, 3, 1.2, 'LPCMCI_READY.csv', '#0D47A1', subtext='83K rows, 174 cols')

    draw_arrow(18.0, 12.1, 18.5, 12.1)

    # Phase 2 boxes
    draw_box(2, 7.5, 4, 1.5, 'Step 9: LPCMCI', '#2E7D32', 11, subtext='8 Analysis Levels\n1,055 Causal Links')
    draw_box(8, 7.5, 4, 1.5, 'Step 10: Temporal KG', '#388E3C', 11, subtext='Fund/Stock/Sector\nHOLDS/EXITED')
    draw_box(14, 7.5, 4, 1.5, 'Step 11: Causal KG', '#43A047', 11, subtext='CausalVariable\nCAUSES edges')

    draw_arrow(19, 11.5, 4, 9.0, '#0D47A1', 'features')
    draw_arrow(6, 8.25, 8, 8.25, '#2E7D32', 'links')
    draw_arrow(6, 7.8, 14, 7.8, '#2E7D32', 'causal links')
    draw_arrow(12, 8.25, 14, 8.25, '#388E3C')

    # Neo4j
    draw_box(19, 7, 2.5, 2.5, 'Neo4j', '#1B5E20', 14, subtext='Temporal +\nCausal KG')
    draw_arrow(12, 7.8, 19, 8.0, '#388E3C')
    draw_arrow(18, 8.0, 19, 8.0, '#43A047')

    # Phase 3 boxes
    draw_box(0.5, 2.5, 3.5, 1.5, 'Step 12:\nIntrinsic Eval', '#E65100', 10, subtext='Structural/Semantic\nFDR Correction')
    draw_box(4.5, 2.5, 3.5, 1.5, 'Step 13:\nGraphRAG', '#F57C00', 10, subtext='Query Classification\nCypher Retrieval')
    draw_box(8.5, 2.5, 3.5, 1.5, 'Step 14:\nPortfolio Engine', '#FF9800', 10, subtext='XGBoost + Causal\nRules + Constraints')
    draw_box(12.5, 2.5, 3.5, 1.5, 'Step 15:\nExplainable AI', '#FFA726', 10, subtext='Causal Chains\nSHAP + Gemini')
    draw_box(16.5, 2.5, 3.5, 1.5, 'Step 16:\nBacktest', '#FFB74D', 10, subtext='Walk-Forward\nSharpe/Accuracy')

    draw_arrow(20, 7.0, 2, 4.0, '#1B5E20', 'Neo4j')
    draw_arrow(20, 7.0, 6, 4.0, '#1B5E20')
    draw_arrow(20, 7.0, 10, 4.0, '#1B5E20')
    draw_arrow(20, 7.0, 14, 4.0, '#1B5E20')

    draw_arrow(4.0, 3.25, 4.5, 3.25)
    draw_arrow(8.0, 3.25, 8.5, 3.25)
    draw_arrow(12.0, 3.25, 12.5, 3.25)
    draw_arrow(16.0, 3.25, 16.5, 3.25)

    # Final outputs
    draw_box(5, 0.7, 4, 1.0, 'Portfolio CSV', '#BF360C', 10, subtext='15 stocks, weights')
    draw_box(10, 0.7, 4, 1.0, 'Explanations', '#D84315', 10, subtext='Causal + SHAP + NL')
    draw_box(15.5, 0.7, 4.5, 1.0, 'Backtest Results', '#E64A19', 10, subtext='Sharpe, Returns, Accuracy')

    draw_arrow(10, 2.5, 7, 1.7)
    draw_arrow(14, 2.5, 12, 1.7)
    draw_arrow(18, 2.5, 17.5, 1.7)

    path = os.path.join(OUT_DIR, 'methodology_diagram.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ====================================================================
# 9. LPCMCI Decision Flowchart
# ====================================================================
def draw_lpcmci_decision_flowchart():
    """Draw the Runge et al. decision flowchart for choosing LPCMCI."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 20)
    ax.axis('off')
    ax.set_title('Causal Method Selection Flowchart\n(Runge et al., 2023)',
                 fontsize=16, fontweight='bold', pad=15)

    def decision(x, y, text, answer):
        diamond_x = [x, x+1.5, x, x-1.5, x]
        diamond_y = [y+1.0, y, y-1.0, y, y+1.0]
        ax.fill(diamond_x, diamond_y, color='#E3F2FD', edgecolor='#1565C0', linewidth=2)
        ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold',
                wrap=True)
        # YES arrow going down
        ax.annotate('', xy=(x, y-1.3), xytext=(x, y-1.0),
                    arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2))
        ax.text(x+0.3, y-1.15, 'YES', fontsize=8, color='#2E7D32', fontweight='bold')

    def result_box(x, y, text, color='#4CAF50'):
        box = FancyBboxPatch((x-2, y-0.4), 4, 0.8, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')

    def reason_box(x, y, text):
        ax.text(x, y, text, ha='left', va='center', fontsize=7.5,
                color='#37474F', style='italic',
                bbox=dict(facecolor='#FAFAFA', edgecolor='#BDBDBD', pad=3))

    # Q1
    decision(7, 18.5, 'Causal\nQuestion?', 'YES')
    reason_box(9.5, 18.5, '"What CAUSES fund managers\nto change allocations?"')

    # Q2
    decision(7, 16.5, 'Causal Nodes\nDefined?', 'YES')
    reason_box(9.5, 16.5, 'Macro, Technical, Sentiment,\nFund Action variables defined')

    # Q3
    decision(7, 14.5, 'Time Series\nData?', 'YES')
    reason_box(9.5, 14.5, '45 monthly observations\n(Jan 2022 - Oct 2025)')

    # Q4
    decision(7, 12.5, 'Causal\nDiscovery?', 'YES')
    reason_box(9.5, 12.5, 'Structure unknown a priori.\nAlgorithm must DISCOVER it.')

    # Q5
    decision(7, 10.5, 'Multiple\nDistributions?', 'NO')
    ax.text(7.3, 9.85, 'NO', fontsize=8, color='#C62828', fontweight='bold')
    reason_box(9.5, 10.5, 'Single continuous time series.\nNot multi-environment.')

    # Q6
    decision(7, 8.5, 'Deterministic\nSystem?', 'NO')
    ax.text(7.3, 7.85, 'NO', fontsize=8, color='#C62828', fontweight='bold')
    reason_box(9.5, 8.5, 'Financial markets are\ninherently stochastic.')

    # Q7 - KEY
    decision(7, 6.5, 'Hidden\nConfounders?', 'YES')
    reason_box(9.5, 6.5, 'RBI decisions, geopolitical events,\nfund manager private info (unobserved)')

    # Q8
    decision(7, 4.5, 'Contemporaneous\nEffects?', 'YES')
    reason_box(9.5, 4.5, '46% of links are lag=0.\nSame-month market reactions.')

    # Result
    result_box(7, 2.5, 'LPCMCI RECOMMENDED', '#1B5E20')

    # Highlight Q7 as the key decision
    highlight = plt.Rectangle((5, 5.3), 4, 2.4, fill=False,
                              edgecolor='#F44336', linewidth=3, linestyle='--')
    ax.add_patch(highlight)
    ax.text(4.8, 7.8, 'KEY DECISION', fontsize=10, color='#F44336',
            fontweight='bold', rotation=0)

    path = os.path.join(OUT_DIR, 'lpcmci_decision_flowchart.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ====================================================================
# Main
# ====================================================================
def main():
    print("=" * 60)
    print("KNOWLEDGE GRAPH VISUALIZATION")
    print(f"Output: {OUT_DIR}")
    print("=" * 60)

    draw_temporal_kg_schema()
    draw_causal_kg_schema()
    draw_causal_network()
    draw_causal_heatmap()
    draw_lag_distribution()
    draw_edge_type_distribution()
    draw_portfolio_allocation()
    draw_methodology_diagram()
    draw_lpcmci_decision_flowchart()

    print("\n  All visualizations saved.")
    print("=" * 60)


if __name__ == '__main__':
    main()
