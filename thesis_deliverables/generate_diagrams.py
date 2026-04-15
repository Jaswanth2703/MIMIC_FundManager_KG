"""
Thesis Diagram Generator — Publication Quality
================================================
Generates all diagrams for:
  "Imitating Fund Manager Decisions: A Causally-Informed
   Knowledge Graph Approach for Portfolio Construction"

All diagrams: 300 DPI, no overlapping, paper-ready.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diagrams')
os.makedirs(OUT, exist_ok=True)

# ── Consistent Professional Color Palette ──
C = {
    'phase1_data':   '#2196F3',   # Blue — data pipeline
    'phase1_causal': '#1565C0',   # Dark blue — causal discovery
    'phase1_kg':     '#0D47A1',   # Deeper blue — KG construction
    'phase2_model':  '#2E7D32',   # Green — mimicry models
    'phase2_eval':   '#1B5E20',   # Dark green — evaluation
    'novel':         '#E65100',   # Orange — novel contributions
    'highlight':     '#C62828',   # Red — key findings
    'bg_light':      '#F5F5F5',   # Light gray background
    'bg_box':        '#FAFAFA',   # Very light box fill
    'text':          '#212121',   # Near-black text
    'text_light':    '#616161',   # Gray text
    'arrow':         '#424242',   # Dark gray arrows
    'border':        '#BDBDBD',   # Light border
    'white':         '#FFFFFF',
    'fund':          '#1976D2',   # Fund node
    'stock':         '#388E3C',   # Stock node
    'sector':        '#F57C00',   # Sector node
    'time':          '#7B1FA2',   # TimePeriod node
    'regime':        '#C62828',   # MarketRegime node
    'causal':        '#00838F',   # CausalVariable node
    'snapshot':      '#5D4037',   # Snapshot nodes
    'domain':        '#455A64',   # DomainConcept
}

FONT = 'DejaVu Sans'

def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white',
                edgecolor='none', pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 1. END-TO-END METHODOLOGY FLOW
# ═══════════════════════════════════════════════════════════════
def diagram_methodology_flow():
    """Complete 16-step pipeline in a clean snake layout."""
    fig, ax = plt.subplots(figsize=(22, 16))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 16)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # Title
    ax.text(11, 15.4, 'End-to-End System Architecture',
            ha='center', va='center', fontsize=18, fontweight='bold',
            fontfamily=FONT, color=C['text'])
    ax.text(11, 15.0,
            'Imitating Fund Manager Decisions via Causally-Informed Knowledge Graph',
            ha='center', va='center', fontsize=11, fontfamily=FONT,
            color=C['text_light'], style='italic')

    # Phase labels on left
    phase_labels = [
        (14.0, 'PHASE 1A', 'Data Acquisition\n& Feature Engineering',
         C['phase1_data']),
        (11.0, 'PHASE 1B', 'Causal Discovery\n& KG Construction',
         C['phase1_causal']),
        (7.6,  'PHASE 1C', 'KG Evaluation\n& Novel Metrics',
         C['phase1_kg']),
        (4.5,  'PHASE 2A', 'KG-Based Mimicry\nModels',
         C['phase2_model']),
        (1.8,  'PHASE 2B', 'Evaluation, Backtest\n& Ablation',
         C['phase2_eval']),
    ]
    for y, label, desc, color in phase_labels:
        ax.add_patch(FancyBboxPatch(
            (0.2, y - 0.55), 2.6, 1.1,
            boxstyle="round,pad=0.15", facecolor=color, alpha=0.12,
            edgecolor=color, linewidth=1.5))
        ax.text(1.5, y + 0.15, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color, fontfamily=FONT)
        ax.text(1.5, y - 0.25, desc, ha='center', va='center',
                fontsize=7.5, color=C['text_light'], fontfamily=FONT)

    # Step definitions: (x, y, step_num, short_name, color)
    steps = [
        # Row 1: Data Pipeline (steps 0-6)
        (4.0, 14.0, '00', 'Build ISIN\nMapping',       C['phase1_data']),
        (6.5, 14.0, '01', 'Load Portfolio\nHoldings',   C['phase1_data']),
        (9.0, 14.0, '02', 'Merge\nFundamentals',       C['phase1_data']),
        (11.5,14.0, '03', 'Fetch OHLCV\nPrice Data',   C['phase1_data']),
        (14.0,14.0, '04', 'Technical\nIndicators',      C['phase1_data']),
        (16.5,14.0, '05', 'FinBERT\nSentiment',        C['phase1_data']),
        (19.0,14.0, '06', 'Macro\nIndicators',         C['phase1_data']),

        # Row 2: Causal Discovery (steps 7-9)
        (19.0,11.0, '07', 'Build Causal\nDataset',     C['phase1_causal']),
        (16.5,11.0, '08', 'Feature\nEngineering',      C['phase1_causal']),
        (14.0,11.0, '09', 'Targeted Panel\nGranger',   C['phase1_causal']),
        (11.5,11.0, '09a','ICP\nDiscovery',            C['phase1_causal']),
        (9.0, 11.0, '09b','DML\nEffects',              C['phase1_causal']),

        # Row 3: KG Construction & Evaluation (steps 10-12)
        (4.0, 8.0,  '10', 'Build Temporal\nKG (Neo4j)', C['phase1_kg']),
        (6.5, 8.0,  '11', 'Build Causal\nKG Edges',    C['phase1_kg']),
        (9.0, 8.0, '11b','Add ICP/DML\nto KG',         C['phase1_kg']),
        (11.5,8.0,  '12', 'Intrinsic\nEvaluation',     C['phase1_kg']),
        (14.0,8.0, '12b','Novel Metrics\nCSCS/SCSI/DMF',C['novel']),

        # Row 4: Mimicry Models (steps 13-13b)
        (14.0,4.5, '13', 'CBR Inference\nEngine',       C['phase2_model']),
        (11.5,4.5, '13a','Causal Path\nTransformer',    C['phase2_model']),
        (9.0, 4.5, '13b\nexp','Export KG\nfor GPU',     C['phase2_model']),
        (6.5, 4.5, '13b','HGT +\nCI-HGT (Novel)',      C['novel']),

        # Row 5: Evaluation & Backtest (steps 14-16)
        (6.5, 1.8, '14b','7-Way Model\nComparison',    C['phase2_eval']),
        (9.0, 1.8, '15', 'Explainable\nAI (XAI)',      C['phase2_eval']),
        (11.5,1.8, '16', 'Walk-Forward\nBacktest',     C['phase2_eval']),
        (14.0,1.8, '16b','Ablation\nStudy',            C['phase2_eval']),
        (16.5,1.8, '16c','Style\nClustering',          C['phase2_eval']),
    ]

    # Draw steps
    for x, y, num, label, color in steps:
        ax.add_patch(FancyBboxPatch(
            (x - 1.05, y - 0.55), 2.1, 1.1,
            boxstyle="round,pad=0.12", facecolor=color, alpha=0.08,
            edgecolor=color, linewidth=1.8))
        # Step number badge
        ax.add_patch(plt.Circle((x - 0.7, y + 0.35), 0.22,
                     facecolor=color, edgecolor='white', linewidth=1.5, zorder=5))
        ax.text(x - 0.7, y + 0.35, num.split('\n')[0], ha='center', va='center',
                fontsize=6.5, fontweight='bold', color='white', fontfamily=FONT, zorder=6)
        ax.text(x, y - 0.1, label, ha='center', va='center',
                fontsize=8, color=C['text'], fontfamily=FONT, linespacing=1.3)

    # Arrows — Row 1 (left to right)
    arrow_kw = dict(arrowstyle='->', color=C['arrow'], lw=1.2,
                    connectionstyle='arc3,rad=0', mutation_scale=12)
    for i in range(6):
        x1 = 4.0 + i * 2.5 + 1.05
        x2 = 4.0 + (i+1) * 2.5 - 1.05
        ax.annotate('', xy=(x2, 14.0), xytext=(x1, 14.0),
                    arrowprops=arrow_kw)

    # Arrow: Row 1 → Row 2 (step 06 → step 07)
    ax.annotate('', xy=(19.0, 11.55), xytext=(19.0, 13.45),
                arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.2))

    # Arrows — Row 2 (right to left)
    row2_xs = [19.0, 16.5, 14.0, 11.5, 9.0]
    for i in range(4):
        ax.annotate('', xy=(row2_xs[i+1] + 1.05, 11.0),
                    xytext=(row2_xs[i] - 1.05, 11.0),
                    arrowprops=arrow_kw)

    # Arrow: Row 2 → Row 3 (step 09b → step 10)
    ax.annotate('', xy=(4.0, 8.55), xytext=(9.0, 10.45),
                arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.2,
                                connectionstyle='arc3,rad=0.3'))

    # Arrows — Row 3 (left to right)
    row3_xs = [4.0, 6.5, 9.0, 11.5, 14.0]
    for i in range(4):
        ax.annotate('', xy=(row3_xs[i+1] - 1.05, 8.0),
                    xytext=(row3_xs[i] + 1.05, 8.0),
                    arrowprops=arrow_kw)

    # Arrow: Row 3 → Row 4 (step 12b → step 13)
    ax.annotate('', xy=(14.0, 5.05), xytext=(14.0, 7.45),
                arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.2))

    # Arrows — Row 4 (right to left)
    row4_xs = [14.0, 11.5, 9.0, 6.5]
    for i in range(3):
        ax.annotate('', xy=(row4_xs[i+1] + 1.05, 4.5),
                    xytext=(row4_xs[i] - 1.05, 4.5),
                    arrowprops=arrow_kw)

    # Arrow: Row 4 → Row 5 (step 13b → step 14b)
    ax.annotate('', xy=(6.5, 2.35), xytext=(6.5, 3.95),
                arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.2))

    # Arrows — Row 5 (left to right)
    row5_xs = [6.5, 9.0, 11.5, 14.0, 16.5]
    for i in range(4):
        ax.annotate('', xy=(row5_xs[i+1] - 1.05, 1.8),
                    xytext=(row5_xs[i] + 1.05, 1.8),
                    arrowprops=arrow_kw)

    # KG symbol in center
    ax.add_patch(FancyBboxPatch(
        (16.5, 7.2), 4.5, 1.6,
        boxstyle="round,pad=0.2", facecolor=C['novel'], alpha=0.06,
        edgecolor=C['novel'], linewidth=1.5, linestyle='--'))
    ax.text(18.75, 8.3, 'Neo4j Knowledge Graph', ha='center', va='center',
            fontsize=10, fontweight='bold', color=C['novel'], fontfamily=FONT)
    ax.text(18.75, 7.7, '27,842 Nodes • 143,814 Edges\n10 Node Types • 13 Relation Types',
            ha='center', va='center', fontsize=8, color=C['text_light'], fontfamily=FONT)

    # Legend
    legend_items = [
        ('Data Acquisition', C['phase1_data']),
        ('Causal Discovery', C['phase1_causal']),
        ('KG Construction', C['phase1_kg']),
        ('Mimicry Models', C['phase2_model']),
        ('Evaluation', C['phase2_eval']),
        ('Novel Contribution', C['novel']),
    ]
    for i, (label, color) in enumerate(legend_items):
        lx = 17.0 + (i % 3) * 1.8
        ly = 0.4 if i >= 3 else 0.8
        ax.add_patch(plt.Rectangle((lx - 0.12, ly - 0.1), 0.24, 0.2,
                     facecolor=color, alpha=0.7))
        ax.text(lx + 0.22, ly, label, va='center', fontsize=7,
                color=C['text'], fontfamily=FONT)

    _save(fig, '01_methodology_flow.png')


# ═══════════════════════════════════════════════════════════════
# 2. KNOWLEDGE GRAPH SCHEMA
# ═══════════════════════════════════════════════════════════════
def diagram_kg_schema():
    """KG schema with all 10 node types and key relationships."""
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(9, 13.5, 'Knowledge Graph Schema',
            ha='center', va='center', fontsize=18, fontweight='bold',
            fontfamily=FONT, color=C['text'])
    ax.text(9, 13.1, '27,842 Nodes  •  143,814 Edges  •  10 Node Types  •  13 Relationship Types',
            ha='center', va='center', fontsize=10, fontfamily=FONT,
            color=C['text_light'])

    # Node positions and definitions
    nodes = {
        'Fund':           (3.5,  10.5, C['fund'],     32,    'Fund portfolios'),
        'Stock':          (9.0,  10.5, C['stock'],    1057,  'Equity instruments'),
        'Sector':         (14.5, 10.5, C['sector'],    21,   'Industry sectors'),
        'TimePeriod':     (3.5,   7.0, C['time'],      46,   'Monthly periods'),
        'MarketRegime':   (9.0,   7.0, C['regime'],     4,   'VIX-based regimes'),
        'FundSnapshot':   (14.5,  7.0, C['snapshot'], 1076,  'Monthly fund state'),
        'StockSnapshot':  (14.5,  3.5, C['snapshot'],25538,  'Monthly stock state'),
        'CausalVariable': (3.5,   3.5, C['causal'],    58,   'Causal features'),
        'CausalAnalysis': (9.0,   3.5, C['domain'],     1,   'Analysis metadata'),
        'DomainConcept':  (9.0,   1.0, C['domain'],     9,   'Financial concepts'),
    }

    # Draw nodes
    node_w, node_h = 3.2, 1.6
    for name, (x, y, color, count, desc) in nodes.items():
        ax.add_patch(FancyBboxPatch(
            (x - node_w/2, y - node_h/2), node_w, node_h,
            boxstyle="round,pad=0.15", facecolor=color, alpha=0.12,
            edgecolor=color, linewidth=2.0))
        ax.text(x, y + 0.25, name, ha='center', va='center',
                fontsize=11, fontweight='bold', color=color, fontfamily=FONT)
        ax.text(x, y - 0.15, f'n = {count:,}', ha='center', va='center',
                fontsize=9, color=C['text'], fontfamily=FONT)
        ax.text(x, y - 0.5, desc, ha='center', va='center',
                fontsize=8, color=C['text_light'], fontfamily=FONT, style='italic')

    # Relationships: (from_node, to_node, label, count, color, rad)
    rels = [
        ('Fund', 'Stock',          'HOLDS\n83,643',     C['fund'],    0.0),
        ('Fund', 'Stock',          'EXITED\n4,567',     C['highlight'], -0.25),
        ('Stock', 'Sector',        'BELONGS_TO\n1,057', C['stock'],   0.0),
        ('TimePeriod','TimePeriod', 'NEXT\n45',          C['time'],    0.4),
        ('TimePeriod','MarketRegime','IN_REGIME\n46',    C['regime'],  0.0),
        ('Fund', 'FundSnapshot',   'OF_FUND\n1,076',    C['fund'],    0.15),
        ('FundSnapshot','TimePeriod','ACTIVE_IN\n1,076', C['snapshot'],0.15),
        ('Stock','StockSnapshot',  'OF_STOCK\n25,538',  C['stock'],   0.15),
        ('StockSnapshot','TimePeriod','AT_TIME\n26,614', C['snapshot'],0.15),
        ('CausalVariable','CausalVariable','GRANGER_CAUSES\n106', C['causal'], 0.35),
        ('CausalVariable','DomainConcept','REPRESENTS\n12', C['causal'], 0.0),
        ('DomainConcept','DomainConcept','INFLUENCES\n7', C['domain'], 0.4),
        ('CausalVariable','CausalAnalysis','ASSOCIATED_WITH\n27', C['causal'], 0.0),
    ]

    for from_n, to_n, label, color, rad in rels:
        x1, y1 = nodes[from_n][0], nodes[from_n][1]
        x2, y2 = nodes[to_n][0], nodes[to_n][1]
        if from_n == to_n:
            # Self-loop: draw a small arc above/beside the node
            ax.annotate('', xy=(x1 + 0.8, y1 + node_h/2),
                        xytext=(x1 - 0.8, y1 + node_h/2),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.3,
                                        connectionstyle=f'arc3,rad={rad}'))
            ax.text(x1, y1 + node_h/2 + 0.55, label.split('\n')[0],
                    ha='center', va='center', fontsize=7, fontweight='bold',
                    color=color, fontfamily=FONT,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor=color, alpha=0.9, linewidth=0.5))
        else:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.3,
                                        connectionstyle=f'arc3,rad={rad}'))
            mx, my = (x1+x2)/2, (y1+y2)/2
            # Offset label for curved arrows
            if abs(rad) > 0.1:
                mx += rad * 1.2
                my += abs(rad) * 0.3
            parts = label.split('\n')
            ax.text(mx, my + 0.15, parts[0], ha='center', va='center',
                    fontsize=7, fontweight='bold', color=color, fontfamily=FONT,
                    bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                              edgecolor=color, alpha=0.9, linewidth=0.5))
            if len(parts) > 1:
                ax.text(mx, my - 0.15, parts[1], ha='center', va='center',
                        fontsize=6.5, color=C['text_light'], fontfamily=FONT)

    _save(fig, '02_kg_schema.png')


# ═══════════════════════════════════════════════════════════════
# 3. CAUSAL DISCOVERY FRAMEWORK
# ═══════════════════════════════════════════════════════════════
def diagram_causal_discovery():
    """Three causal methods: Granger, ICP, DML — their inputs, outputs, KG integration."""
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(9, 10.5, 'Causal Discovery Framework — Three Complementary Methods',
            ha='center', va='center', fontsize=16, fontweight='bold',
            fontfamily=FONT, color=C['text'])

    # Input box
    ax.add_patch(FancyBboxPatch(
        (6.5, 8.8), 5, 1.0,
        boxstyle="round,pad=0.15", facecolor=C['phase1_data'], alpha=0.1,
        edgecolor=C['phase1_data'], linewidth=1.5))
    ax.text(9, 9.5, 'CAUSAL_DISCOVERY_DATASET.csv', ha='center', va='center',
            fontsize=10, fontweight='bold', color=C['phase1_data'], fontfamily=FONT)
    ax.text(9, 9.1, 'Panel data: 83,643 fund×stock×month observations, 58 features',
            ha='center', va='center', fontsize=8, color=C['text_light'], fontfamily=FONT)

    # Three method boxes
    methods = [
        (3.0, 6.0, 'Panel Granger\nCausality',
         'Step 09',
         ['Panel OLS with entity FE',
          'FDR correction (α = 0.05)',
          'Lags 1–6 months',
          'Stratified by fund type'],
         ['β coefficient (effect direction)',
          'p_fdr (significance)',
          'partial_r² (effect size)',
          '106 significant edges'],
         C['phase1_causal'],
         'GRANGER_CAUSES'),

        (9.0, 6.0, 'Invariant Causal\nPrediction (ICP)',
         'Step 09a',
         ['Tests invariance across',
          '  environments (Fund_Type,',
          '  regime, VIX level)',
          'Confidence ∈ [0, 1]'],
         ['confidence (causal strength)',
          'in_intersection (boolean)',
          'plausible_sets per var',
          '1 confirmed variable'],
         C['novel'],
         'CAUSES'),

        (15.0, 6.0, 'Double Machine\nLearning (DML)',
         'Step 09b',
         ['Debiased treatment effects',
          'RF residualization',
          '5-fold cross-fitting',
          'Robust to confounders'],
         ['θ̂ (treatment effect)',
          '95% CI bounds',
          't-statistic, p-value',
          'Heterogeneous effects'],
         C['phase1_causal'],
         'CAUSAL_EFFECT'),
    ]

    for x, y, title, step, details, outputs, color, rel in methods:
        bw, bh = 4.8, 5.2
        ax.add_patch(FancyBboxPatch(
            (x - bw/2, y - bh/2 - 0.8), bw, bh,
            boxstyle="round,pad=0.15", facecolor=color, alpha=0.06,
            edgecolor=color, linewidth=1.8))

        # Title
        ax.text(x, y + 1.6, title, ha='center', va='center',
                fontsize=12, fontweight='bold', color=color, fontfamily=FONT)
        ax.text(x, y + 1.1, f'({step})', ha='center', va='center',
                fontsize=9, color=C['text_light'], fontfamily=FONT)

        # Method details
        ax.text(x - bw/2 + 0.3, y + 0.6, 'Method:', ha='left', va='center',
                fontsize=8, fontweight='bold', color=C['text'], fontfamily=FONT)
        for i, d in enumerate(details):
            ax.text(x - bw/2 + 0.3, y + 0.2 - i*0.32, f'• {d}',
                    ha='left', va='center', fontsize=7.5, color=C['text'],
                    fontfamily=FONT)

        # Outputs
        ax.text(x - bw/2 + 0.3, y - 1.2, 'Outputs:', ha='left', va='center',
                fontsize=8, fontweight='bold', color=C['text'], fontfamily=FONT)
        for i, o in enumerate(outputs):
            ax.text(x - bw/2 + 0.3, y - 1.55 - i*0.32, f'→ {o}',
                    ha='left', va='center', fontsize=7.5, color=C['text'],
                    fontfamily=FONT)

        # KG relation badge
        ax.add_patch(FancyBboxPatch(
            (x - 1.3, y - bh/2 - 0.65), 2.6, 0.45,
            boxstyle="round,pad=0.1", facecolor=color, alpha=0.15,
            edgecolor=color, linewidth=1.0))
        ax.text(x, y - bh/2 - 0.43, f'→ KG: {rel}',
                ha='center', va='center', fontsize=8, fontweight='bold',
                color=color, fontfamily=FONT)

        # Arrow from input
        ax.annotate('', xy=(x, y + bh/2 - 0.8),
                    xytext=(x, 8.8),
                    arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.2))

    # Convergence arrow to KG
    ax.add_patch(FancyBboxPatch(
        (6.0, 0.2), 6.0, 1.0,
        boxstyle="round,pad=0.15", facecolor=C['phase1_kg'], alpha=0.1,
        edgecolor=C['phase1_kg'], linewidth=1.5))
    ax.text(9, 0.9, 'Neo4j Knowledge Graph', ha='center', va='center',
            fontsize=11, fontweight='bold', color=C['phase1_kg'], fontfamily=FONT)
    ax.text(9, 0.5, 'Unified causal layer: 133 causal edges from 3 complementary methods',
            ha='center', va='center', fontsize=8, color=C['text_light'], fontfamily=FONT)

    for x in [3.0, 9.0, 15.0]:
        ax.annotate('', xy=(9, 1.2), xytext=(x, 2.8),
                    arrowprops=dict(arrowstyle='->', color=C['phase1_kg'],
                                    lw=1.5, connectionstyle='arc3,rad=0'))

    _save(fig, '03_causal_discovery_framework.png')


# ═══════════════════════════════════════════════════════════════
# 4. PHASE 2 ARCHITECTURE — MIMICRY MODELS
# ═══════════════════════════════════════════════════════════════
def diagram_phase2_architecture():
    """Phase 2: CBR → PathTransformer → HGT/CI-HGT → Comparison → Backtest."""
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(10, 11.5, 'Phase 2: KG-Based Fund Manager Mimicry Architecture',
            ha='center', va='center', fontsize=16, fontweight='bold',
            fontfamily=FONT, color=C['text'])

    # KG Input (top)
    ax.add_patch(FancyBboxPatch(
        (7.5, 10.0), 5.0, 1.0,
        boxstyle="round,pad=0.15", facecolor=C['phase1_kg'], alpha=0.12,
        edgecolor=C['phase1_kg'], linewidth=2.0))
    ax.text(10, 10.7, 'Neo4j Knowledge Graph', ha='center', va='center',
            fontsize=12, fontweight='bold', color=C['phase1_kg'], fontfamily=FONT)
    ax.text(10, 10.3, '27,842 nodes • 143,814 edges • Causal + Temporal layers',
            ha='center', va='center', fontsize=8, color=C['text_light'], fontfamily=FONT)

    # Three mimicry models
    models = [
        (3.5, 7.0, 'CBR Inference Engine',
         'Step 13',
         ['Extract KG subgraphs per decision',
          'WL graph kernel similarity',
          'k-NN retrieval from historical',
          'Walk-forward cross-validation'],
         'cbr_decision_predictions.csv',
         C['phase2_model']),

        (10.0, 7.0, 'Causal Path Transformer',
         'Step 13a',
         ['Extract causal paths from KG',
          'd_model=128, 8 heads, 3 layers',
          'Warmup + cosine annealing LR',
          'Path-level attention over KG'],
         'path_decision_predictions.csv',
         C['phase2_model']),

        (16.5, 7.0, 'HGT + CI-HGT (Novel)',
         'Step 13b',
         ['Heterogeneous Graph Transformer',
          'CI-HGT: CausalGate modulates',
          '  message passing via causal',
          '  edge strengths (Novel)'],
         'hgt/ci_hgt_predictions.csv',
         C['novel']),
    ]

    for x, y, title, step, details, output, color in models:
        bw, bh = 5.0, 4.0
        ax.add_patch(FancyBboxPatch(
            (x - bw/2, y - bh/2), bw, bh,
            boxstyle="round,pad=0.15", facecolor=color, alpha=0.07,
            edgecolor=color, linewidth=1.8))
        ax.text(x, y + 1.5, title, ha='center', va='center',
                fontsize=11, fontweight='bold', color=color, fontfamily=FONT)
        ax.text(x, y + 1.1, f'({step})', ha='center', va='center',
                fontsize=8.5, color=C['text_light'], fontfamily=FONT)
        for i, d in enumerate(details):
            ax.text(x - bw/2 + 0.3, y + 0.5 - i*0.35, f'• {d}',
                    ha='left', va='center', fontsize=8, color=C['text'],
                    fontfamily=FONT)
        # Output file
        ax.add_patch(FancyBboxPatch(
            (x - 1.8, y - bh/2 + 0.1), 3.6, 0.4,
            boxstyle="round,pad=0.08", facecolor=color, alpha=0.15,
            edgecolor=color, linewidth=0.8))
        ax.text(x, y - bh/2 + 0.3, f'→ {output}',
                ha='center', va='center', fontsize=7, fontweight='bold',
                color=color, fontfamily=FONT)
        # Arrow from KG
        ax.annotate('', xy=(x, y + bh/2),
                    xytext=(10, 10.0),
                    arrowprops=dict(arrowstyle='->', color=C['arrow'],
                                    lw=1.3, connectionstyle='arc3,rad=0'))

    # Downstream: Comparison, XAI, Backtest, Ablation
    downstream = [
        (3.5, 2.0, '7-Way Model\nComparison', 'Step 14b',
         'M0–M7 + Ensemble', C['phase2_eval']),
        (8.0, 2.0, 'Explainable\nAI (XAI)', 'Step 15',
         'KG-grounded explanations', C['phase2_eval']),
        (12.5, 2.0, 'Walk-Forward\nBacktest', 'Step 16',
         'Real returns + Sharpe/Sortino', C['phase2_eval']),
        (17.0, 2.0, 'Ablation\nStudy', 'Step 16b',
         'Feature/Method/Architecture', C['phase2_eval']),
    ]

    for x, y, title, step, desc, color in downstream:
        bw, bh = 3.5, 2.2
        ax.add_patch(FancyBboxPatch(
            (x - bw/2, y - bh/2), bw, bh,
            boxstyle="round,pad=0.12", facecolor=color, alpha=0.07,
            edgecolor=color, linewidth=1.5))
        ax.text(x, y + 0.5, title, ha='center', va='center',
                fontsize=10, fontweight='bold', color=color, fontfamily=FONT)
        ax.text(x, y + 0.0, f'({step})', ha='center', va='center',
                fontsize=8, color=C['text_light'], fontfamily=FONT)
        ax.text(x, y - 0.4, desc, ha='center', va='center',
                fontsize=7.5, color=C['text'], fontfamily=FONT, style='italic')

    # Arrows from models to downstream
    for mx in [3.5, 10.0, 16.5]:
        for dx in [3.5, 8.0, 12.5, 17.0]:
            ax.annotate('', xy=(dx, 3.1),
                        xytext=(mx, 5.0),
                        arrowprops=dict(arrowstyle='->', color=C['border'],
                                        lw=0.6, alpha=0.4))

    # Data flow label
    ax.text(10, 4.2, 'Per-Decision Predictions Flow',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=C['novel'], fontfamily=FONT,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=C['novel'],
                      alpha=0.08, edgecolor=C['novel']))

    _save(fig, '04_phase2_architecture.png')


# ═══════════════════════════════════════════════════════════════
# 5. CI-HGT NOVEL ARCHITECTURE (Detailed)
# ═══════════════════════════════════════════════════════════════
def diagram_cihgt_architecture():
    """Detailed CI-HGT architecture showing CausalGate innovation."""
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(9, 10.5, 'CI-HGT: Causally-Informed Heterogeneous Graph Transformer',
            ha='center', va='center', fontsize=16, fontweight='bold',
            fontfamily=FONT, color=C['novel'])
    ax.text(9, 10.1, 'Novel Contribution — Causal Edge Strengths Modulate GNN Message Passing',
            ha='center', va='center', fontsize=10, fontfamily=FONT,
            color=C['text_light'], style='italic')

    # Input layer
    inputs = [
        (2.5, 8.5, 'Fund\nNodes', f'n=32\ndim=4', C['fund']),
        (5.5, 8.5, 'Stock\nNodes', f'n=1,057\ndim=6', C['stock']),
        (8.5, 8.5, 'Sector\nNodes', f'n=21\ndim=3', C['sector']),
        (11.5,8.5, 'TimePeriod\nNodes', f'n=46\ndim=5', C['time']),
        (14.5,8.5, 'CausalVar\nNodes', f'n=58\ndim=4', C['causal']),
    ]
    ax.text(0.5, 8.5, 'Input\nLayer', ha='center', va='center',
            fontsize=9, fontweight='bold', color=C['text'], fontfamily=FONT,
            rotation=0)
    for x, y, label, dim, color in inputs:
        ax.add_patch(FancyBboxPatch(
            (x - 1.1, y - 0.6), 2.2, 1.2,
            boxstyle="round,pad=0.1", facecolor=color, alpha=0.12,
            edgecolor=color, linewidth=1.5))
        ax.text(x, y + 0.2, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color, fontfamily=FONT)
        ax.text(x, y - 0.3, dim, ha='center', va='center',
                fontsize=7.5, color=C['text_light'], fontfamily=FONT)

    # HGT Layers
    ax.text(0.5, 6.3, 'HGT\nLayers', ha='center', va='center',
            fontsize=9, fontweight='bold', color=C['text'], fontfamily=FONT)
    for i in range(3):
        lx = 2.0 + i * 5.0
        ly = 6.3
        ax.add_patch(FancyBboxPatch(
            (lx, ly - 0.5), 4.0, 1.0,
            boxstyle="round,pad=0.12", facecolor=C['phase2_model'], alpha=0.08,
            edgecolor=C['phase2_model'], linewidth=1.5))
        ax.text(lx + 2.0, ly + 0.15, f'HGT Layer {i+1}',
                ha='center', va='center', fontsize=10, fontweight='bold',
                color=C['phase2_model'], fontfamily=FONT)
        ax.text(lx + 2.0, ly - 0.2, f'hidden=128, heads=4, attention per edge type',
                ha='center', va='center', fontsize=7.5, color=C['text_light'],
                fontfamily=FONT)
        if i < 2:
            ax.annotate('', xy=(lx + 5.0, ly),
                        xytext=(lx + 4.0, ly),
                        arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.2))

    # Arrows from inputs to HGT
    for x, _, _, _, _ in inputs:
        ax.annotate('', xy=(4.0, 6.8), xytext=(x, 7.9),
                    arrowprops=dict(arrowstyle='->', color=C['border'], lw=0.7, alpha=0.5))

    # CausalGate (THE NOVEL PART)
    ax.add_patch(FancyBboxPatch(
        (1.5, 3.5), 15.0, 2.0,
        boxstyle="round,pad=0.2", facecolor=C['novel'], alpha=0.06,
        edgecolor=C['novel'], linewidth=2.5, linestyle='-'))
    ax.text(9, 5.15, '★  CausalGate Module (Novel Contribution)  ★',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color=C['novel'], fontfamily=FONT)

    gate_steps = [
        (3.0, 4.2, '①  Source Gating',
         'gate = σ(W · causal_edge_attr)\nh_src_gated = h_src ⊙ gate',
         'Modulate source node\nembeddings by causal strength'),
        (9.0, 4.2, '②  Scatter-Mean',
         'h_agg = scatter_mean(\n  h_src_gated, target_idx)',
         'Aggregate gated messages\nto target nodes'),
        (15.0, 4.2, '③  Residual Update',
         'h_target += 0.1 × h_agg',
         'Scaled residual preserves\noriginal embeddings'),
    ]
    for x, y, title, formula, desc in gate_steps:
        ax.text(x, y + 0.45, title, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=C['novel'], fontfamily=FONT)
        ax.text(x, y - 0.05, formula, ha='center', va='center',
                fontsize=8, color=C['text'], fontfamily='monospace')
        ax.text(x, y - 0.65, desc, ha='center', va='center',
                fontsize=7.5, color=C['text_light'], fontfamily=FONT, style='italic')
    # Arrows between gate steps
    ax.annotate('', xy=(6.0, 4.2), xytext=(5.0, 4.2),
                arrowprops=dict(arrowstyle='->', color=C['novel'], lw=1.3))
    ax.annotate('', xy=(12.0, 4.2), xytext=(11.0, 4.2),
                arrowprops=dict(arrowstyle='->', color=C['novel'], lw=1.3))

    # Arrow from HGT to CausalGate
    ax.annotate('', xy=(9, 5.5), xytext=(9, 5.8),
                arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.5))

    # Edge classifier
    ax.add_patch(FancyBboxPatch(
        (5.5, 1.0), 7.0, 1.5,
        boxstyle="round,pad=0.15", facecolor=C['phase2_eval'], alpha=0.08,
        edgecolor=C['phase2_eval'], linewidth=1.5))
    ax.text(9, 2.1, 'Edge Classifier (HOLDS)', ha='center', va='center',
            fontsize=11, fontweight='bold', color=C['phase2_eval'], fontfamily=FONT)
    ax.text(9, 1.65, 'concat(h_fund, h_stock, edge_features) → MLP → softmax',
            ha='center', va='center', fontsize=8.5, color=C['text'], fontfamily=FONT)
    ax.text(9, 1.3, 'Output: P(BUY) | P(HOLD) | P(SELL)  per fund×stock×month',
            ha='center', va='center', fontsize=8, color=C['text_light'],
            fontfamily=FONT, style='italic')

    ax.annotate('', xy=(9, 2.5), xytext=(9, 3.5),
                arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.5))

    # Key insight box
    ax.add_patch(FancyBboxPatch(
        (1.0, 0.1), 16.0, 0.7,
        boxstyle="round,pad=0.1", facecolor=C['novel'], alpha=0.08,
        edgecolor=C['novel'], linewidth=1.0))
    ax.text(9, 0.45, 'Key Insight: Strong causal evidence (high ICP confidence, low Granger p-value) '
            'is amplified; weak evidence is attenuated. The gate LEARNS the threshold.',
            ha='center', va='center', fontsize=8.5, color=C['novel'],
            fontfamily=FONT, fontweight='bold')

    _save(fig, '05_cihgt_architecture.png')


# ═══════════════════════════════════════════════════════════════
# 6. NOVEL EVALUATION METRICS (CSCS, SCSI, DMF)
# ═══════════════════════════════════════════════════════════════
def diagram_novel_metrics():
    """Three novel KG evaluation metrics with formulas and results."""
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(9, 9.5, 'Novel KG Evaluation Metrics',
            ha='center', va='center', fontsize=16, fontweight='bold',
            fontfamily=FONT, color=C['text'])
    ax.text(9, 9.1, 'Proposed metrics to evaluate causally-informed Knowledge Graphs '
            '(no prior standard exists)',
            ha='center', va='center', fontsize=9, fontfamily=FONT,
            color=C['text_light'], style='italic')

    metrics = [
        (3.0, 5.5, 'CSCS', 'Causal-Semantic\nCoherence Score',
         ['Measures alignment between',
          'discovered causal edges and',
          'established financial theory.',
          '',
          'CSCS = Σ(theory_aligned) / |E|',
          'CSCS_W = Σ(|β| × aligned) / Σ|β|',
          '',
          'Interpretation:',
          '> 0.50 → Good (majority grounded)',
          '< 0.30 → Poor (mostly spurious)'],
         'CSCS = 0.549\nCSCS_W = 0.535',
         '✓ Good: majority of edges\ngrounded in financial theory',
         C['novel']),

        (9.0, 5.5, 'SCSI', 'Stratified Causal\nStability Index',
         ['Measures stability of causal',
          'structure across fund segments',
          '(e.g., small-cap vs mid-cap).',
          '',
          'SCSI = Jaccard(S₁, S₂)',
          'SCSI_comp = stability⁰·⁷ ×',
          '            lag_persist⁰·³',
          'Interpretation:',
          '> 0.40 → Stable across segments',
          '< 0.20 → Segment-specific only'],
         'SCSI = 0.391\nSCSI_comp = 0.500',
         '≈ Moderate: partial overlap,\nsome segment-specific drivers',
         C['novel']),

        (15.0, 5.5, 'DMF', 'Decision-Mimicry\nFaithfulness',
         ['Measures alignment between',
          'KG causal features and ML',
          'feature importance rankings.',
          '',
          'DMF = grounding_ratio × rank_align',
          'grounding = |KG ∩ ML_top_K| / K',
          '',
          'Interpretation:',
          '> 0.50 → Strong KG-ML alignment',
          '< 0.25 → KG not used by ML'],
         'DMF = 0.600',
         '✓ Moderate: substantial\nKG–classifier alignment',
         C['novel']),
    ]

    for x, y, abbrev, full_name, details, result, interpretation, color in metrics:
        bw, bh = 4.8, 7.2
        ax.add_patch(FancyBboxPatch(
            (x - bw/2, y - bh/2), bw, bh,
            boxstyle="round,pad=0.15", facecolor=color, alpha=0.05,
            edgecolor=color, linewidth=1.8))

        # Title
        ax.text(x, y + 3.1, abbrev, ha='center', va='center',
                fontsize=16, fontweight='bold', color=color, fontfamily=FONT)
        ax.text(x, y + 2.5, full_name, ha='center', va='center',
                fontsize=10, color=C['text'], fontfamily=FONT)

        # Details
        for i, d in enumerate(details):
            ax.text(x - bw/2 + 0.3, y + 1.8 - i*0.36, d,
                    ha='left', va='center', fontsize=8,
                    color=C['text'] if d and not d.startswith('>') and not d.startswith('<') else C['text_light'],
                    fontfamily='monospace' if '=' in d or '>' in d or '<' in d else FONT)

        # Result box
        ax.add_patch(FancyBboxPatch(
            (x - 1.8, y - 2.6), 3.6, 0.8,
            boxstyle="round,pad=0.1", facecolor=color, alpha=0.12,
            edgecolor=color, linewidth=1.2))
        ax.text(x, y - 2.2, result, ha='center', va='center',
                fontsize=10, fontweight='bold', color=color, fontfamily=FONT)

        # Interpretation
        ax.text(x, y - 3.2, interpretation, ha='center', va='center',
                fontsize=8, color=C['text'], fontfamily=FONT, style='italic')

    _save(fig, '06_novel_evaluation_metrics.png')


# ═══════════════════════════════════════════════════════════════
# 7. KG STATISTICS & EVALUATION RESULTS TABLE
# ═══════════════════════════════════════════════════════════════
def diagram_results_table():
    """Phase 1 evaluation results as a clean table image."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(0.5, 0.96, 'Phase 1: Knowledge Graph Evaluation Results',
            ha='center', va='center', fontsize=16, fontweight='bold',
            fontfamily=FONT, color=C['text'], transform=ax.transAxes)

    # Table 1: KG Structural Statistics
    t1_data = [
        ['Metric', 'Value', 'Assessment'],
        ['Total Nodes', '27,842', '—'],
        ['Total Edges', '143,814', '—'],
        ['Node Types', '10', '—'],
        ['Relationship Types', '13', '—'],
        ['Schema Node Coverage', '87.5%', 'Good'],
        ['Schema Relation Coverage', '100.0%', 'Excellent'],
        ['Connected Ratio', '99.99%', 'Excellent'],
        ['Graph Density', '0.000186', 'Sparse (expected)'],
        ['Average Degree', '10.33', 'Well-connected'],
        ['Temporal Completeness', '100.0%', 'Complete'],
        ['Causal Variable Utilization', '24.1%', 'Selective'],
    ]

    t1 = ax.table(cellText=t1_data[1:], colLabels=t1_data[0],
                  cellLoc='center', loc='upper left',
                  bbox=[0.02, 0.48, 0.46, 0.43])
    t1.auto_set_font_size(False)
    t1.set_fontsize(8.5)
    for (row, col), cell in t1.get_celld().items():
        cell.set_edgecolor(C['border'])
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(C['phase1_kg'])
            cell.set_text_props(color='white', fontweight='bold', fontsize=9)
        elif row % 2 == 0:
            cell.set_facecolor('#F8F9FA')
        else:
            cell.set_facecolor('white')

    # Table 2: Causal Discovery Results
    t2_data = [
        ['Method', 'Edges', 'Key Output', 'Significance'],
        ['Panel Granger', '106', 'β, p_fdr, partial_r²', '100% FDR significant'],
        ['ICP', '1 confirmed', 'confidence ∈ [0,1]', 'volume_ratio confirmed'],
        ['DML', '27 effects', 'θ̂, 95% CI', 'Debiased estimates'],
        ['Total Causal', '133', '3 complementary layers', 'Multi-method validated'],
    ]

    t2 = ax.table(cellText=t2_data[1:], colLabels=t2_data[0],
                  cellLoc='center', loc='upper right',
                  bbox=[0.52, 0.66, 0.46, 0.24])
    t2.auto_set_font_size(False)
    t2.set_fontsize(8.5)
    for (row, col), cell in t2.get_celld().items():
        cell.set_edgecolor(C['border'])
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(C['phase1_causal'])
            cell.set_text_props(color='white', fontweight='bold', fontsize=9)
        elif row % 2 == 0:
            cell.set_facecolor('#F8F9FA')
        else:
            cell.set_facecolor('white')

    # Table 3: Novel Metrics
    t3_data = [
        ['Metric', 'Score', 'Interpretation'],
        ['CSCS', '0.549', 'Good: majority theory-aligned'],
        ['CSCS_W (weighted)', '0.535', 'Good: β-weighted alignment'],
        ['SCSI', '0.391', 'Moderate: partial stability'],
        ['SCSI_composite', '0.500', 'Moderate: spatial + temporal'],
        ['DMF', '0.600', 'Moderate: KG-ML alignment'],
        ['Overall Quality', '0.740', 'Good: comprehensive score'],
        ['Inferential Utility', '80.0%', '8/10 questions answered'],
    ]

    t3 = ax.table(cellText=t3_data[1:], colLabels=t3_data[0],
                  cellLoc='center', loc='center',
                  bbox=[0.52, 0.32, 0.46, 0.30])
    t3.auto_set_font_size(False)
    t3.set_fontsize(8.5)
    for (row, col), cell in t3.get_celld().items():
        cell.set_edgecolor(C['border'])
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(C['novel'])
            cell.set_text_props(color='white', fontweight='bold', fontsize=9)
        elif row % 2 == 0:
            cell.set_facecolor('#FFF3E0')
        else:
            cell.set_facecolor('white')

    # Table 4: Position Action Distribution
    t4_data = [
        ['Action', 'Count', 'Percentage'],
        ['HOLD', '40,762', '48.7%'],
        ['BUY', '20,032', '23.9%'],
        ['SELL', '15,298', '18.3%'],
        ['INITIAL_POSITION', '7,551', '9.0%'],
        ['Total', '83,643', '100.0%'],
    ]
    t4 = ax.table(cellText=t4_data[1:], colLabels=t4_data[0],
                  cellLoc='center', loc='lower left',
                  bbox=[0.02, 0.05, 0.35, 0.24])
    t4.auto_set_font_size(False)
    t4.set_fontsize(8.5)
    for (row, col), cell in t4.get_celld().items():
        cell.set_edgecolor(C['border'])
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(C['phase1_data'])
            cell.set_text_props(color='white', fontweight='bold', fontsize=9)
        elif row % 2 == 0:
            cell.set_facecolor('#E3F2FD')
        else:
            cell.set_facecolor('white')

    # Section labels
    ax.text(0.25, 0.92, 'KG Structural Statistics',
            ha='center', va='center', fontsize=11, fontweight='bold',
            fontfamily=FONT, color=C['phase1_kg'], transform=ax.transAxes)
    ax.text(0.75, 0.92, 'Causal Discovery Results',
            ha='center', va='center', fontsize=11, fontweight='bold',
            fontfamily=FONT, color=C['phase1_causal'], transform=ax.transAxes)
    ax.text(0.75, 0.63, 'Novel Evaluation Metrics',
            ha='center', va='center', fontsize=11, fontweight='bold',
            fontfamily=FONT, color=C['novel'], transform=ax.transAxes)
    ax.text(0.19, 0.30, 'Decision Distribution',
            ha='center', va='center', fontsize=11, fontweight='bold',
            fontfamily=FONT, color=C['phase1_data'], transform=ax.transAxes)

    _save(fig, '07_phase1_results_table.png')


# ═══════════════════════════════════════════════════════════════
# 8. PHASE 2 MODEL COMPARISON TABLE (Placeholder)
# ═══════════════════════════════════════════════════════════════
def diagram_phase2_comparison():
    """Phase 2 model comparison table — fill after running pipeline."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(0.5, 0.94, 'Phase 2: 7-Way Model Comparison (Mimicry Metrics)',
            ha='center', va='center', fontsize=16, fontweight='bold',
            fontfamily=FONT, color=C['text'], transform=ax.transAxes)
    ax.text(0.5, 0.88, 'All models evaluated on same test months — walk-forward cross-validation',
            ha='center', va='center', fontsize=10, fontfamily=FONT,
            color=C['text_light'], transform=ax.transAxes, style='italic')

    data = [
        ['Model', 'KG?', 'Accuracy', 'F1 (Wt)', 'κ (Kappa)', 'Agreement', 'BUY Recall', 'SELL Recall'],
        ['M0: XGBoost (All Features)', 'No', '—', '—', '—', '—', '—', '—'],
        ['M1: XGBoost (Causal MB)', 'Indirect', '—', '—', '—', '—', '—', '—'],
        ['M2: XGBoost (Correlation)', 'No', '—', '—', '—', '—', '—', '—'],
        ['M3: Naïve HOLD Baseline', 'No', '—', '—', '—', '—', '—', '—'],
        ['M4: CBR-KG (WL Kernel)', '✓ Direct', '—', '—', '—', '—', '—', '—'],
        ['M5: Path Transformer', '✓ Direct', '—', '—', '—', '—', '—', '—'],
        ['M6: HGT', '✓ Direct', '—', '—', '—', '—', '—', '—'],
        ['M7: CI-HGT (Novel)', '✓ Direct', '—', '—', '—', '—', '—', '—'],
        ['M8: KG Ensemble', '✓ Direct', '—', '—', '—', '—', '—', '—'],
    ]

    table = ax.table(cellText=data[1:], colLabels=data[0],
                     cellLoc='center', loc='center',
                     bbox=[0.02, 0.08, 0.96, 0.75])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(C['border'])
        cell.set_linewidth(0.5)
        cell.set_height(0.08)
        if row == 0:
            cell.set_facecolor(C['phase2_model'])
            cell.set_text_props(color='white', fontweight='bold', fontsize=9.5)
        elif row in [5, 6, 7, 8, 9]:
            cell.set_facecolor('#E8F5E9')  # Green tint for KG models
        elif row % 2 == 0:
            cell.set_facecolor('#F5F5F5')
        else:
            cell.set_facecolor('white')
        if col == 0:
            cell.set_text_props(ha='left')
            cell._loc = 'left'

    ax.text(0.5, 0.03,
            '— = Fill after running Phase 2 on personal PC  |  '
            'κ = Cohen\'s Kappa (Cohen, 1960)  |  '
            'F1 Wt = Weighted F1 Score',
            ha='center', va='center', fontsize=8.5, fontfamily=FONT,
            color=C['text_light'], transform=ax.transAxes, style='italic')

    _save(fig, '08_phase2_comparison_table.png')


# ═══════════════════════════════════════════════════════════════
# 9. BACKTEST COMPARISON TABLE (Placeholder)
# ═══════════════════════════════════════════════════════════════
def diagram_backtest_table():
    """Backtest results table — Sharpe, Sortino, IR, etc."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(0.5, 0.94, 'Walk-Forward Backtest Results (Real Monthly Returns)',
            ha='center', va='center', fontsize=16, fontweight='bold',
            fontfamily=FONT, color=C['text'], transform=ax.transAxes)
    ax.text(0.5, 0.87, 'Transaction cost: 0.5% one-way  |  Benchmark: equal-weighted universe  |  '
            'Monthly risk-free: 0.5%',
            ha='center', va='center', fontsize=9, fontfamily=FONT,
            color=C['text_light'], transform=ax.transAxes, style='italic')

    data = [
        ['Strategy', 'Sharpe', 'Sortino', 'Info Ratio', 'Ann. Return', 'Max DD', 'Calmar', 'Turnover'],
        ['M0: XGBoost All', '—', '—', '—', '—', '—', '—', '—'],
        ['M1: XGBoost Causal', '—', '—', '—', '—', '—', '—', '—'],
        ['M4: CBR-KG', '—', '—', '—', '—', '—', '—', '—'],
        ['M5: PathTransformer', '—', '—', '—', '—', '—', '—', '—'],
        ['M6: HGT', '—', '—', '—', '—', '—', '—', '—'],
        ['M7: CI-HGT (Novel)', '—', '—', '—', '—', '—', '—', '—'],
        ['M8: KG Ensemble', '—', '—', '—', '—', '—', '—', '—'],
        ['Benchmark (EW)', '—', '—', '—', '—', '—', '—', '—'],
    ]

    table = ax.table(cellText=data[1:], colLabels=data[0],
                     cellLoc='center', loc='center',
                     bbox=[0.02, 0.08, 0.96, 0.72])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(C['border'])
        cell.set_linewidth(0.5)
        cell.set_height(0.09)
        if row == 0:
            cell.set_facecolor(C['phase2_eval'])
            cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        elif row in [3, 4, 5, 6, 7]:
            cell.set_facecolor('#E8F5E9')
        elif row % 2 == 0:
            cell.set_facecolor('#F5F5F5')
        else:
            cell.set_facecolor('white')

    ax.text(0.5, 0.03,
            '— = Fill after running  |  '
            'Sharpe ratio (Sharpe, 1966)  |  '
            'Sortino ratio (Sortino & van der Meer, 1991)  |  '
            'Info Ratio (Grinold & Kahn, 2000)',
            ha='center', va='center', fontsize=8, fontfamily=FONT,
            color=C['text_light'], transform=ax.transAxes, style='italic')

    _save(fig, '09_backtest_results_table.png')


# ═══════════════════════════════════════════════════════════════
# 10. DATA PIPELINE FLOW
# ═══════════════════════════════════════════════════════════════
def diagram_data_pipeline():
    """Steps 0-6: Data sources and feature engineering."""
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(9, 7.5, 'Data Acquisition & Feature Engineering Pipeline',
            ha='center', va='center', fontsize=16, fontweight='bold',
            fontfamily=FONT, color=C['text'])

    # Data Sources (top row)
    sources = [
        (2.0, 6.0, 'AMFI India\nPortfolios', 'Monthly holdings\n32 funds × 46 months', C['phase1_data']),
        (5.5, 6.0, 'Screener.in\nFundamentals', 'EPS, P/E, Debt/Equity\nper stock per quarter', C['phase1_data']),
        (9.0, 6.0, 'Kite/NSE\nOHLCV', 'Daily prices → Monthly\naggregation', C['phase1_data']),
        (12.5,6.0, 'FinBERT\nNLP', 'News sentiment\nper stock per month', C['novel']),
        (16.0,6.0, 'RBI/FRED\nMacro', 'GDP, VIX, Crude Oil\nRepo Rate, US10Y', C['phase1_data']),
    ]

    for x, y, title, desc, color in sources:
        bw, bh = 2.8, 1.5
        ax.add_patch(FancyBboxPatch(
            (x - bw/2, y - bh/2), bw, bh,
            boxstyle="round,pad=0.12", facecolor=color, alpha=0.1,
            edgecolor=color, linewidth=1.5))
        ax.text(x, y + 0.3, title, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=color, fontfamily=FONT)
        ax.text(x, y - 0.35, desc, ha='center', va='center',
                fontsize=7.5, color=C['text_light'], fontfamily=FONT)

    # Processing steps (middle)
    ax.add_patch(FancyBboxPatch(
        (1.0, 3.3), 16, 1.2,
        boxstyle="round,pad=0.15", facecolor=C['phase1_causal'], alpha=0.06,
        edgecolor=C['phase1_causal'], linewidth=1.5))
    ax.text(9, 4.15, 'Feature Engineering (Step 08)', ha='center', va='center',
            fontsize=12, fontweight='bold', color=C['phase1_causal'], fontfamily=FONT)
    ax.text(9, 3.7, 'Lag features (1–6 months)  •  Momentum (3/6/12 month)  •  '
            'Rolling statistics  •  Cross-sectional ranks  •  Prune correlated (r > 0.95)',
            ha='center', va='center', fontsize=8.5, color=C['text'], fontfamily=FONT)

    # Arrows from sources to processing
    for x, _, _, _, _ in sources:
        ax.annotate('', xy=(x, 4.5), xytext=(x, 5.25),
                    arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.0))

    # Output datasets (bottom)
    outputs = [
        (4.5, 1.5, 'CAUSAL_DISCOVERY\n_DATASET.csv',
         '83,643 rows × 58 cols\nRaw (unstandardized)', C['phase1_causal']),
        (9.0, 1.5, 'LPCMCI_READY.csv',
         '83,643 rows × 45 cols\nStandardized + pruned', C['phase1_causal']),
        (13.5,1.5, 'Technical Indicators',
         'RSI, Bollinger, SMA\nMACD, Volume Ratio', C['phase1_data']),
    ]
    for x, y, title, desc, color in outputs:
        bw, bh = 3.5, 1.5
        ax.add_patch(FancyBboxPatch(
            (x - bw/2, y - bh/2), bw, bh,
            boxstyle="round,pad=0.12", facecolor=color, alpha=0.08,
            edgecolor=color, linewidth=1.5, linestyle='--'))
        ax.text(x, y + 0.3, title, ha='center', va='center',
                fontsize=9, fontweight='bold', color=color, fontfamily=FONT)
        ax.text(x, y - 0.3, desc, ha='center', va='center',
                fontsize=7.5, color=C['text_light'], fontfamily=FONT)

    # Arrows from processing to outputs
    for x, _, _, _, _ in outputs:
        ax.annotate('', xy=(x, 2.25), xytext=(x, 3.3),
                    arrowprops=dict(arrowstyle='->', color=C['arrow'], lw=1.0))

    _save(fig, '10_data_pipeline.png')


# ═══════════════════════════════════════════════════════════════
# 11. POSITION ACTION DISTRIBUTION (Pie + Bar)
# ═══════════════════════════════════════════════════════════════
def diagram_action_distribution():
    """Decision distribution with bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle('Fund Manager Decision Distribution (83,643 HOLDS edges)',
                 fontsize=14, fontweight='bold', fontfamily=FONT, color=C['text'])

    labels = ['HOLD', 'BUY', 'SELL', 'INITIAL_POSITION']
    values = [40762, 20032, 15298, 7551]
    colors = ['#42A5F5', '#66BB6A', '#EF5350', '#FFA726']
    explode = (0.03, 0.03, 0.03, 0.03)

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        values, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90, pctdistance=0.75,
        textprops={'fontsize': 10, 'fontfamily': FONT})
    for at in autotexts:
        at.set_fontweight('bold')
        at.set_fontsize(9)
    ax1.set_title('Proportional Distribution', fontsize=12, fontfamily=FONT, pad=15)

    # Bar chart
    bars = ax2.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5)
    ax2.set_ylabel('Number of Decisions', fontsize=11, fontfamily=FONT)
    ax2.set_title('Absolute Counts', fontsize=12, fontfamily=FONT, pad=15)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f'{val:,}', ha='center', va='bottom', fontsize=10,
                 fontweight='bold', fontfamily=FONT)
    ax2.set_ylim(0, max(values) * 1.15)
    ax2.tick_params(axis='x', labelsize=9)
    ax2.tick_params(axis='y', labelsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, '11_action_distribution.png')


# ═══════════════════════════════════════════════════════════════
# 12. MULTI-METHOD CONSENSUS DIAGRAM
# ═══════════════════════════════════════════════════════════════
def diagram_multi_method_consensus():
    """Venn-style showing overlap of three causal methods."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(6, 7.5, 'Multi-Method Causal Consensus',
            ha='center', va='center', fontsize=16, fontweight='bold',
            fontfamily=FONT, color=C['text'])
    ax.text(6, 7.1, 'Variables confirmed by multiple causal discovery methods',
            ha='center', va='center', fontsize=10, fontfamily=FONT,
            color=C['text_light'], style='italic')

    # Three circles (Venn-like)
    from matplotlib.patches import Circle
    c1 = Circle((4.5, 4.0), 2.2, facecolor=C['phase1_causal'], alpha=0.08,
                edgecolor=C['phase1_causal'], linewidth=2.0)
    c2 = Circle((7.5, 4.0), 2.2, facecolor=C['novel'], alpha=0.08,
                edgecolor=C['novel'], linewidth=2.0)
    c3 = Circle((6.0, 2.0), 2.2, facecolor=C['phase2_model'], alpha=0.08,
                edgecolor=C['phase2_model'], linewidth=2.0)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.add_patch(c3)

    # Labels
    ax.text(3.3, 5.0, 'Granger\nCausality', ha='center', va='center',
            fontsize=12, fontweight='bold', color=C['phase1_causal'], fontfamily=FONT)
    ax.text(3.3, 4.2, '106 edges', ha='center', va='center',
            fontsize=10, color=C['text_light'], fontfamily=FONT)

    ax.text(8.7, 5.0, 'ICP', ha='center', va='center',
            fontsize=12, fontweight='bold', color=C['novel'], fontfamily=FONT)
    ax.text(8.7, 4.2, '~12 parents', ha='center', va='center',
            fontsize=10, color=C['text_light'], fontfamily=FONT)

    ax.text(6.0, 0.8, 'DML', ha='center', va='center',
            fontsize=12, fontweight='bold', color=C['phase2_model'], fontfamily=FONT)
    ax.text(6.0, 0.2, '27 effects', ha='center', va='center',
            fontsize=10, color=C['text_light'], fontfamily=FONT)

    # Intersection labels
    ax.text(5.5, 4.5, 'Granger\n∩ ICP\n1 var', ha='center', va='center',
            fontsize=9, fontweight='bold', color=C['text'], fontfamily=FONT)
    ax.text(5.0, 2.8, 'Granger\n∩ DML\n6 vars', ha='center', va='center',
            fontsize=9, fontweight='bold', color=C['text'], fontfamily=FONT)

    # Center: all three
    ax.add_patch(FancyBboxPatch(
        (4.8, 3.0), 2.4, 0.8,
        boxstyle="round,pad=0.1", facecolor=C['highlight'], alpha=0.15,
        edgecolor=C['highlight'], linewidth=1.5))
    ax.text(6.0, 3.4, 'All Three: volume_ratio', ha='center', va='center',
            fontsize=10, fontweight='bold', color=C['highlight'], fontfamily=FONT)

    _save(fig, '12_multi_method_consensus.png')


# ═══════════════════════════════════════════════════════════════
# 13. COMPETENCY QUESTIONS TABLE
# ═══════════════════════════════════════════════════════════════
def diagram_competency_questions():
    """Inferential utility — 10 competency questions."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(0.5, 0.95, 'KG Inferential Utility — Competency Questions',
            ha='center', va='center', fontsize=15, fontweight='bold',
            fontfamily=FONT, color=C['text'], transform=ax.transAxes)
    ax.text(0.5, 0.90, 'Score: 8/10 answered (80% inferential utility)',
            ha='center', va='center', fontsize=10, fontfamily=FONT,
            color=C['text_light'], transform=ax.transAxes, style='italic')

    data = [
        ['#', 'Competency Question', 'Answered?', 'Result'],
        ['Q1', 'What variables Granger-cause BUY decisions at lag-1 in mid-cap?', '✗', '—'],
        ['Q2', 'What variables Granger-cause SELL decisions (negative β)?', '✓', '30 variables'],
        ['Q3', 'Is the causal structure different between strata?', '✓', 'Yes (1 stratum diff)'],
        ['Q4', 'What macro factors influence fund decisions?', '✓', '2 macro factors'],
        ['Q5', 'Does sentiment causally predict decisions?', '✓', '3 sentiment vars'],
        ['Q6', 'What is the strongest causal predictor overall?', '✓', 'pct_nav'],
        ['Q7', 'Which domain concepts influence fund decisions?', '✗', '—'],
        ['Q8', 'Are there variables significant in only one stratum?', '✓', '10 unique vars'],
        ['Q9', 'What is the lag distribution of causal effects?', '✓', 'Avg lag = 3.29'],
        ['Q10', 'How many total causal + temporal edges exist?', '✓', '144,469'],
    ]

    table = ax.table(cellText=data[1:], colLabels=data[0],
                     cellLoc='center', loc='center',
                     bbox=[0.02, 0.05, 0.96, 0.80])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(C['border'])
        cell.set_linewidth(0.5)
        cell.set_height(0.075)
        if row == 0:
            cell.set_facecolor(C['phase1_kg'])
            cell.set_text_props(color='white', fontweight='bold', fontsize=9.5)
        elif col == 2:
            if cell.get_text().get_text() == '✓':
                cell.set_facecolor('#E8F5E9')
                cell.set_text_props(color='#2E7D32', fontweight='bold')
            elif cell.get_text().get_text() == '✗':
                cell.set_facecolor('#FFEBEE')
                cell.set_text_props(color='#C62828', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#F5F5F5')
        else:
            cell.set_facecolor('white')
        if col == 1:
            cell.set_text_props(ha='left')
            cell._loc = 'left'

    _save(fig, '13_competency_questions.png')


# ═══════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("Generating Publication-Quality Thesis Diagrams")
    print("=" * 60)

    diagram_methodology_flow()
    diagram_kg_schema()
    diagram_causal_discovery()
    diagram_phase2_architecture()
    diagram_cihgt_architecture()
    diagram_novel_metrics()
    diagram_results_table()
    diagram_phase2_comparison()
    diagram_backtest_table()
    diagram_data_pipeline()
    diagram_action_distribution()
    diagram_multi_method_consensus()
    diagram_competency_questions()

    print(f"\n  All diagrams saved to: {OUT}")
    print("  Done!")
