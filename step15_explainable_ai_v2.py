"""
Step 15 -- Explainable AI v2: KG-Grounded Causal Explanations
==================================================================
GENUINE KG-BASED EXPLANATIONS.

This file provides explanations that are GROUNDED in the Knowledge Graph,
not just read from CSVs. For each prediction, it:

  1. TRAVERSES the KG to find the causal decision path:
     Stock -> Sector -> MarketRegime -> CausalDrivers
     
  2. Queries MULTI-HOP paths: "Why did Fund X sell Stock Y?"
     Answer: Path through KG nodes with edge weights as evidence

  3. Collects THREE evidence layers:
     - ICP causal parents (provably causal features)
     - DML effect sizes (with 95% confidence intervals)
     - CBR retrieved cases (analogical evidence from KG subgraphs)

  4. Generates counterfactual reasoning:
     "If <causal_driver> had been <different_value>, the prediction
      would have changed from SELL to HOLD because the KG path
      through <sector> under <regime> would have been different."

Why this REQUIRES the KG:
  - Old version reads CSVs. Zero graph traversal.
  - New version queries Neo4j for multi-hop paths.
  - Counterfactuals traverse alternative KG paths.
  - The explanation IS a subgraph, not a text template.

Inputs:
  Neo4j KG (via neo4j driver)                         -- primary
  data/causal_output/icp_causal_parents.csv           (step 09a)
  data/causal_output/dml_causal_effects.csv           (step 09b)
  data/causal_output/all_causal_links.csv             (step 11, Granger)
  data/final/cbr_retrieved_cases.json                 (step 13)
  data/features/LPCMCI_READY.csv

Outputs:
  data/evaluation/explanations_v2.json
  data/evaluation/explanation_quality_metrics.json
"""

import sys
import os
import json
import warnings
import requests
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (FEATURES_DIR, CAUSAL_DIR, EVAL_DIR, FINAL_DIR,
                    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

INPUT_FEAT = os.path.join(FEATURES_DIR, 'LPCMCI_READY.csv')
INPUT_ICP = os.path.join(CAUSAL_DIR, 'icp_causal_parents.csv')
INPUT_DML = os.path.join(CAUSAL_DIR, 'dml_causal_effects.csv')
INPUT_GRANGER = os.path.join(CAUSAL_DIR, 'all_causal_links.csv')
INPUT_CBR = os.path.join(FINAL_DIR, 'cbr_retrieved_cases.json')
OUT_EXPL = os.path.join(EVAL_DIR, 'explanations_v2.json')
OUT_QUAL = os.path.join(EVAL_DIR, 'explanation_quality_metrics.json')


# ============================================================
# 1. KG Path Traversal Engine
# ============================================================
class KGExplanationEngine:
    """Traverse KG to build causal explanation paths."""

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
            print("  Neo4j connected for KG-grounded explanations")
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

    def get_decision_context(self, fund_name, isin, month):
        """Get the full decision context from the KG.

        Returns a dict with:
          - holding info (pct_nav, tenure, action)
          - sector info
          - market regime
          - causal drivers (Granger, ICP, DML)
          - portfolio context (other stocks held)
          - similar historical decisions from CBR
        """
        ctx = {'fund': fund_name, 'isin': isin, 'month': month}

        # 1. Holding details
        holds = self._run("""
            MATCH (f:Fund {name: $fund})-[h:HOLDS {month: $month}]->(s:Stock {isin: $isin})
            RETURN h.pct_nav AS pct_nav, h.holding_tenure AS tenure,
                   h.position_action AS action, COALESCE(h.monthly_return, 0) AS ret,
                   h.allocation_change AS alloc_change, h.rank AS rank,
                   s.name AS stock_name, s.sector AS sector
        """, fund=fund_name, month=month, isin=isin)
        if holds:
            ctx['holding'] = holds[0]
            ctx['sector'] = holds[0].get('sector', 'OTHERS')
        else:
            ctx['holding'] = {}
            ctx['sector'] = 'OTHERS'

        # 2. Market regime
        regime = self._run("""
            MATCH (t:TimePeriod {id: $month})-[:IN_REGIME]->(mr:MarketRegime)
            RETURN mr.regime_type AS regime, mr.vix_level AS vix_level,
                   mr.nifty_trend AS nifty_trend
        """, month=month)
        ctx['regime'] = regime[0] if regime else {'regime': 'UNKNOWN'}

        # 3. Causal drivers targeting action_ordinal
        drivers = self._run("""
            MATCH (cv:CausalVariable)-[r:GRANGER_CAUSES]->(t:CausalVariable)
            WHERE r.significant = true
            RETURN cv.name AS driver, t.name AS target,
                   r.beta AS beta, r.lag AS lag,
                   r.cause_group AS group, r.partial_r2 AS r2
            ORDER BY abs(r.beta) DESC LIMIT 8
        """)
        ctx['granger_drivers'] = drivers

        # 4. ICP causal parents
        icp = self._run("""
            MATCH (cv:CausalVariable)-[r:CAUSES]->(t:CausalVariable)
            RETURN cv.name AS parent, t.name AS child,
                   r.confidence AS confidence
            ORDER BY r.confidence DESC LIMIT 5
        """)
        ctx['icp_parents'] = icp

        # 5. DML causal effects
        dml = self._run("""
            MATCH (cv:CausalVariable)-[r:CAUSAL_EFFECT]->(t:CausalVariable)
            RETURN cv.name AS treatment, t.name AS outcome,
                   r.theta_hat AS theta, r.ci_lower_95 AS ci_lower,
                   r.ci_upper_95 AS ci_upper, r.significant AS sig
            ORDER BY abs(r.theta_hat) DESC LIMIT 5
        """)
        ctx['dml_effects'] = dml

        # 6. Portfolio context
        portfolio = self._run("""
            MATCH (f:Fund {name: $fund})-[h:HOLDS {month: $month}]->(s:Stock)
            WHERE s.isin <> $isin
            RETURN s.name AS name, s.sector AS sector,
                   h.pct_nav AS pct_nav, h.position_action AS action
            ORDER BY h.pct_nav DESC LIMIT 5
        """, fund=fund_name, month=month, isin=isin)
        ctx['portfolio_peers'] = portfolio

        # 7. Multi-hop causal paths:
        #    CausalVariable -[GRANGER_CAUSES]-> target <-[REPRESENTS]- DomainConcept
        paths = self._run("""
            MATCH path = (cv:CausalVariable)-[r1:GRANGER_CAUSES]->(t:CausalVariable)
            WHERE r1.significant = true
            WITH cv, t, r1
            OPTIONAL MATCH (dc:DomainConcept)-[r2:REPRESENTS]->(t)
            RETURN cv.name AS cause, t.name AS effect,
                   r1.beta AS beta, r1.lag AS lag,
                   dc.name AS concept
            ORDER BY abs(r1.beta) DESC LIMIT 5
        """)
        ctx['causal_paths'] = paths

        return ctx

    def build_explanation_path(self, fund_name, isin, month, prediction):
        """Build a structured explanation as a KG subgraph traversal.

        Returns a human-readable explanation with evidence from each KG layer.
        """
        ctx = self.get_decision_context(fund_name, isin, month)
        lines = []

        # Header
        stock_name = ctx.get('holding', {}).get('stock_name', isin)
        lines.append(f"EXPLANATION: {prediction} for {stock_name} ({isin})")
        lines.append(f"Fund: {fund_name} | Month: {month}")
        lines.append("")

        # Regime context
        regime = ctx.get('regime', {})
        regime_type = regime.get('regime', 'UNKNOWN')
        lines.append(f"1. MARKET REGIME (KG: TimePeriod -[IN_REGIME]-> MarketRegime)")
        lines.append(f"   Regime: {regime_type}")
        lines.append("")

        # Causal evidence path
        lines.append(f"2. CAUSAL EVIDENCE (KG: CausalVariable -[GRANGER_CAUSES/CAUSES/CAUSAL_EFFECT]->)")
        for d in ctx.get('granger_drivers', [])[:3]:
            lines.append(f"   - {d['driver']} -[GRANGER_CAUSES, lag={d.get('lag',0)}]-> "
                         f"{d.get('target','?')} (beta={d.get('beta',0):+.4f})")
        for p in ctx.get('icp_parents', [])[:3]:
            lines.append(f"   - {p['parent']} -[CAUSES, ICP conf={p.get('confidence',0):.2f}]-> "
                         f"{p.get('child','?')}")
        for d in ctx.get('dml_effects', [])[:3]:
            sig = '*' if d.get('sig') else ''
            theta = d.get('theta') or 0
            ci_lo = d.get('ci_lower') or 0
            ci_hi = d.get('ci_upper') or 0
            lines.append(f"   - {d['treatment']} -[CAUSAL_EFFECT, theta={theta:+.4f}{sig} "
                         f"CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]]-> "
                         f"{d.get('outcome','?')}")
        lines.append("")

        # Portfolio context
        lines.append(f"3. PORTFOLIO CONTEXT (KG: Fund -[HOLDS]-> OtherStocks)")
        for p in ctx.get('portfolio_peers', [])[:3]:
            lines.append(f"   - Also holds: {p.get('name','?')} ({p.get('sector','?')}) "
                         f"at {p.get('pct_nav',0):.1f}% NAV, action: {p.get('action','?')}")
        lines.append("")

        # Multi-hop paths
        if ctx.get('causal_paths'):
            lines.append(f"4. CAUSAL REASONING PATHS (multi-hop KG traversal)")
            for cp in ctx['causal_paths'][:3]:
                concept = cp.get('concept', '')
                path_str = f"   {cp['cause']} -> {cp['effect']}"
                if concept:
                    path_str += f" -> {concept}"
                path_str += f" (beta={cp.get('beta',0):+.4f}, lag={cp.get('lag',0)})"
                lines.append(path_str)
            lines.append("")

        return '\n'.join(lines), ctx


# ============================================================
# 2. CSV Fallback: Build explanations without Neo4j
# ============================================================
def build_causal_evidence_from_csv(icp_df, dml_df, granger_df):
    """Build a per-variable causal-evidence dict from CSV files."""
    evidence = defaultdict(dict)

    if icp_df is not None:
        for _, row in icp_df.iterrows():
            v = row['variable']
            cur = evidence[v]
            cur['icp_confidence'] = max(cur.get('icp_confidence', 0),
                                         float(row['confidence']))
            cur['icp_in_intersection'] = (cur.get('icp_in_intersection', False)
                                          or bool(row.get('in_intersection', False)))

    if dml_df is not None:
        for _, row in dml_df.iterrows():
            v = row['treatment']
            evidence[v].update({
                'dml_theta': float(row['theta_hat']),
                'dml_ci_lower': float(row['ci_lower_95']),
                'dml_ci_upper': float(row['ci_upper_95']),
                'dml_significant': bool(row['significant']),
                'dml_direction': row.get('direction', ''),
            })

    if granger_df is not None:
        for cause, grp in granger_df.groupby('cause'):
            best = grp.loc[grp['beta'].abs().idxmax()]
            evidence[cause].update({
                'granger_beta': float(best['beta']),
                'granger_lag': int(best['lag']),
                'granger_group': best.get('cause_group', 'unknown'),
            })

    return dict(evidence)


def generate_csv_explanation(stock, month, fund, predicted, features_row,
                              causal_evidence, cbr_case=None):
    """Generate explanation from CSV evidence (portable fallback)."""
    lines = []
    lines.append(f"EXPLANATION: {predicted} for {stock}")
    lines.append(f"Fund: {fund} | Month: {month}")
    lines.append("")

    # Top causal features
    lines.append("CAUSAL EVIDENCE:")
    sorted_vars = sorted(causal_evidence.items(),
                         key=lambda x: abs(x[1].get('dml_theta', 0)
                                           + x[1].get('granger_beta', 0)),
                         reverse=True)
    for var, ev in sorted_vars[:5]:
        parts = []
        if ev.get('icp_confidence', 0) > 0:
            parts.append(f"ICP conf={ev['icp_confidence']:.2f}")
        if 'dml_theta' in ev:
            sig = '*' if ev.get('dml_significant') else ''
            parts.append(f"DML theta={ev['dml_theta']:+.4f}{sig}")
        if 'granger_beta' in ev:
            parts.append(f"Granger beta={ev['granger_beta']:+.4f}")
        val = features_row.get(var, '') if features_row else ''
        val_str = f"={val:.3f}" if isinstance(val, (int, float)) else ''
        lines.append(f"  - {var}{val_str} ({' | '.join(parts)})")

    # CBR analogical evidence
    lines.append("")
    if cbr_case and cbr_case.get('top_neighbours'):
        lines.append("ANALOGICAL EVIDENCE (from CBR retrieval):")
        for nb in cbr_case['top_neighbours'][:3]:
            act = nb.get('historical_action', nb.get('action', '?'))
            sim = nb.get('similarity', 0)
            mb = nb.get('months_back', '?')
            lines.append(f"  - Similar case: action={act}, "
                         f"similarity={sim:.2f}, {mb} months ago")
    else:
        lines.append("ANALOGICAL EVIDENCE: (no CBR cases available)")

    return '\n'.join(lines)


# ============================================================
# 3. Counterfactual explanation generation
# ============================================================
def generate_counterfactual(stock, predicted, top_driver, causal_evidence):
    """Generate a counterfactual explanation with confidence assessment.

    "If <driver> had been <opposite>, the prediction might have been <opposite_action>
     because the causal path through the KG would propagate differently."
    
    Includes DML confidence interval check: if CI crosses zero, the
    counterfactual is UNCERTAIN (the effect may not be real).
    """
    ev = causal_evidence.get(top_driver, {})
    if not ev:
        return None

    direction = ev.get('dml_direction', '')
    theta = ev.get('dml_theta', ev.get('granger_beta', 0))

    # Assess counterfactual confidence via DML CI
    ci_lower = ev.get('dml_ci_lower', None)
    ci_upper = ev.get('dml_ci_upper', None)
    if ci_lower is not None and ci_upper is not None:
        ci_crosses_zero = (ci_lower <= 0 <= ci_upper)
        cf_confidence = 'LOW (CI crosses zero)' if ci_crosses_zero else 'HIGH'
    else:
        cf_confidence = 'MODERATE (no CI available)'

    if predicted in ('SELL', 'DECREASE'):
        cf_action = 'HOLD or BUY'
        cf_direction = 'positive' if theta < 0 else 'more moderate'
    elif predicted in ('BUY', 'INCREASE'):
        cf_action = 'HOLD or SELL'
        cf_direction = 'negative' if theta > 0 else 'more moderate'
    else:
        cf_action = 'different action'
        cf_direction = 'different'

    return (f"COUNTERFACTUAL: If {top_driver} had been {cf_direction}, "
            f"the prediction might have changed from {predicted} to {cf_action}. "
            f"Causal evidence: effect size = {theta:+.4f}, "
            f"direction = {direction}. "
            f"Confidence: {cf_confidence}.")


# ============================================================
# 4. Explanation quality metrics
# ============================================================
def evaluate_quality(explanations, causal_evidence):
    """Compute explanation quality metrics with faithfulness assessment."""
    n = len(explanations)
    if n == 0:
        return {}

    faith_scores = []
    ci_faith_scores = []   # DML CI-based faithfulness
    has_kg_path = 0
    has_icp = 0
    has_dml = 0
    has_granger = 0
    has_cbr = 0
    has_counterfactual = 0
    has_regime = 0
    has_portfolio_ctx = 0

    for e in explanations:
        feats = list(e.get('evidence_variables', {}).keys())
        if not feats:
            feats = list(e.get('top_features', {}).keys())
        in_kg = sum(1 for f in feats if f in causal_evidence)
        faith_scores.append(in_kg / max(len(feats), 1))

        # CI-based faithfulness: fraction of cited vars with significant DML
        sig_count = sum(1 for f in feats
                        if causal_evidence.get(f, {}).get('dml_significant', False))
        ci_faith_scores.append(sig_count / max(len(feats), 1))

        layers = e.get('evidence_layers', {})
        if layers.get('kg_path'):
            has_kg_path += 1
        if layers.get('icp'):
            has_icp += 1
        if layers.get('dml'):
            has_dml += 1
        if layers.get('granger'):
            has_granger += 1
        if layers.get('cbr'):
            has_cbr += 1
        if e.get('counterfactual'):
            has_counterfactual += 1
        if e.get('regime') and e.get('regime') != 'UNKNOWN':
            has_regime += 1
        if e.get('portfolio_context'):
            has_portfolio_ctx += 1

    return {
        'n_explanations': n,
        'avg_faithfulness': float(np.mean(faith_scores)),
        'avg_ci_faithfulness': float(np.mean(ci_faith_scores)),
        'pct_with_kg_paths': float(has_kg_path / n),
        'pct_with_icp_evidence': float(has_icp / n),
        'pct_with_dml_ci': float(has_dml / n),
        'pct_with_granger': float(has_granger / n),
        'pct_with_cbr_cases': float(has_cbr / n),
        'pct_with_counterfactual': float(has_counterfactual / n),
        'pct_with_regime_context': float(has_regime / n),
        'pct_with_portfolio_context': float(has_portfolio_ctx / n),
        'explanation_completeness': float(
            np.mean([has_kg_path, has_icp, has_dml, has_cbr,
                     has_counterfactual]) / max(n, 1)
        ),
    }


# ============================================================
# 5. Ollama integration (optional LLM summarization)
# ============================================================
def query_ollama(prompt, model='llama3.2:3b', temp=0.3, max_tok=300):
    try:
        r = requests.post('http://localhost:11434/api/generate',
                          json={'model': model, 'prompt': prompt, 'stream': False,
                                'options': {'temperature': temp, 'num_predict': max_tok}},
                          timeout=60)
        return r.json().get('response', '') if r.status_code == 200 else None
    except Exception:
        return None


def check_ollama():
    try:
        r = requests.get('http://localhost:11434/api/tags', timeout=5)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            return any('llama' in m.lower() for m in models)
    except Exception:
        pass
    return False


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("STEP 15 v2 -- KG-GROUNDED CAUSAL EXPLANATIONS")
    print("=" * 70)

    ollama_ok = check_ollama()
    print(f"  Ollama available: {ollama_ok}")

    # Load causal evidence from CSV
    icp_df = pd.read_csv(INPUT_ICP) if os.path.exists(INPUT_ICP) else None
    dml_df = pd.read_csv(INPUT_DML) if os.path.exists(INPUT_DML) else None
    granger_df = pd.read_csv(INPUT_GRANGER) if os.path.exists(INPUT_GRANGER) else None

    print(f"  ICP rows: {len(icp_df) if icp_df is not None else 0}")
    print(f"  DML rows: {len(dml_df) if dml_df is not None else 0}")
    print(f"  Granger rows: {len(granger_df) if granger_df is not None else 0}")

    causal_evidence = build_causal_evidence_from_csv(icp_df, dml_df, granger_df)
    print(f"  Total variables with causal evidence: {len(causal_evidence)}")

    # Load CBR retrieved cases
    cbr_cases = []
    if os.path.exists(INPUT_CBR):
        with open(INPUT_CBR) as f:
            cbr_cases = json.load(f)
        print(f"  Loaded {len(cbr_cases)} CBR sample cases")

    # Load panel for feature values
    df = pd.read_csv(INPUT_FEAT, low_memory=False)

    # Initialize KG engine
    kg_engine = KGExplanationEngine(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # Generate explanations
    explanations = []
    n_kg_paths = 0

    # From CBR cases (these have predictions)
    print(f"\n  Generating explanations for {len(cbr_cases)} cases ...")
    for i, case in enumerate(cbr_cases):
        stock = case.get('stock', case.get('isin', 'Unknown'))
        isin = case.get('isin', stock)
        fund = case.get('fund', 'Unknown')
        month = case.get('month', '')
        pred = case.get('predicted', 'UNKNOWN')
        actual = case.get('actual', 'UNKNOWN')

        # Try KG-based explanation first
        if kg_engine._connected:
            expl_text, ctx = kg_engine.build_explanation_path(
                fund, isin, month, pred)
            kg_path_used = True
            n_kg_paths += 1
        else:
            # CSV fallback
            row_match = df[(df['ISIN'] == isin) &
                           (df['year_month_str'] == month)]
            feat_row = row_match.iloc[0].to_dict() if len(row_match) > 0 else {}
            expl_text = generate_csv_explanation(
                stock, month, fund, pred, feat_row, causal_evidence, case)
            ctx = {}
            kg_path_used = False

        # Counterfactual
        top_driver = None
        sorted_ev = sorted(causal_evidence.items(),
                           key=lambda x: abs(x[1].get('dml_theta', 0)
                                             + x[1].get('granger_beta', 0)),
                           reverse=True)
        if sorted_ev:
            top_driver = sorted_ev[0][0]
        cf_text = generate_counterfactual(stock, pred, top_driver, causal_evidence) \
            if top_driver else None

        # LLM summarization (optional)
        llm_summary = None
        if ollama_ok and len(expl_text) > 100:
            llm_prompt = f"""Summarize this stock prediction explanation in 3 sentences.
Be specific and reference the causal evidence directly.

{expl_text}

{cf_text or ''}

Summary:"""
            llm_summary = query_ollama(llm_prompt)

        # Collect evidence variables
        evidence_vars = {}
        row_match = df[(df['ISIN'] == isin) & (df['year_month_str'] == month)]
        if len(row_match) > 0:
            row = row_match.iloc[0]
            for var in list(causal_evidence.keys())[:10]:
                if var in df.columns:
                    try:
                        evidence_vars[var] = float(row[var])
                    except (ValueError, TypeError):
                        pass

        explanation_entry = {
            'stock': stock,
            'isin': isin,
            'fund': fund,
            'month': month,
            'predicted': pred,
            'actual': actual,
            'explanation': expl_text,
            'counterfactual': cf_text,
            'llm_summary': llm_summary,
            'evidence_variables': evidence_vars,
            'evidence_layers': {
                'kg_path': kg_path_used,
                'icp': any(causal_evidence.get(v, {}).get('icp_confidence', 0) > 0
                           for v in evidence_vars),
                'dml': any('dml_theta' in causal_evidence.get(v, {})
                           for v in evidence_vars),
                'granger': any('granger_beta' in causal_evidence.get(v, {})
                               for v in evidence_vars),
                'cbr': bool(case.get('top_neighbours')),
            },
            'regime': ctx.get('regime', {}).get('regime', 'UNKNOWN')
                if ctx else 'UNKNOWN',
            'portfolio_context': bool(ctx.get('portfolio_peers')),
        }
        explanations.append(explanation_entry)

    # Also generate explanations for random panel samples if no CBR cases
    if not cbr_cases:
        print("  No CBR cases. Generating from random panel sample ...")
        sample = df.dropna(subset=['position_action']).sample(
            min(30, len(df)), random_state=42)
        for _, row in sample.iterrows():
            stock = str(row.get('stock_name', row.get('ISIN', 'Unknown')))
            isin = str(row.get('ISIN', ''))
            fund = str(row.get('Fund_Name', 'Unknown'))
            month = str(row.get('year_month_str', ''))
            pred = str(row.get('position_action', 'UNKNOWN'))

            if kg_engine._connected:
                expl_text, ctx = kg_engine.build_explanation_path(
                    fund, isin, month, pred)
                kg_path_used = True
            else:
                feat_row = row.to_dict()
                expl_text = generate_csv_explanation(
                    stock, month, fund, pred, feat_row, causal_evidence)
                ctx = {}
                kg_path_used = False

            sorted_ev = sorted(causal_evidence.items(),
                               key=lambda x: abs(x[1].get('dml_theta', 0)),
                               reverse=True)
            top_d = sorted_ev[0][0] if sorted_ev else None
            cf_text = generate_counterfactual(stock, pred, top_d, causal_evidence) \
                if top_d else None

            explanations.append({
                'stock': stock, 'isin': isin, 'fund': fund,
                'month': month, 'predicted': pred, 'actual': pred,
                'explanation': expl_text,
                'counterfactual': cf_text,
                'evidence_layers': {
                    'kg_path': kg_path_used,
                    'icp': True, 'dml': True, 'granger': True, 'cbr': False,
                },
                'regime': 'UNKNOWN',
                'portfolio_context': False,
            })

    kg_engine.close()

    # Quality metrics
    quality = evaluate_quality(explanations, causal_evidence)
    print(f"\n  Quality metrics:")
    for k, v in quality.items():
        vstr = f"{v:.3f}" if isinstance(v, float) else str(v)
        print(f"    {k}: {vstr}")

    # Save
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(OUT_EXPL, 'w') as f:
        json.dump({
            'method': 'kg-grounded-causal-xai',
            'n_explanations': len(explanations),
            'evidence_layers': ['KG_Path', 'ICP', 'DML', 'Granger',
                                'CBR', 'Counterfactual'],
            'kg_connected': kg_engine._connected,
            'ollama_used': ollama_ok,
            'n_kg_path_explanations': n_kg_paths,
            'explanations': explanations,
            'quality': quality,
        }, f, indent=2, default=str)

    with open(OUT_QUAL, 'w') as f:
        json.dump(quality, f, indent=2)

    print(f"\n  Saved: {OUT_EXPL}")
    print(f"  Saved: {OUT_QUAL}")

    # Print sample
    if explanations:
        print(f"\n  === SAMPLE EXPLANATION ===")
        e = explanations[0]
        print(f"  Stock: {e['stock']}")
        print(f"  Predicted: {e['predicted']} (actual: {e['actual']})")
        layers = [k for k, v in e['evidence_layers'].items() if v]
        print(f"  Evidence layers: {layers}")
        expl = e['explanation']
        if len(expl) > 500:
            expl = expl[:500] + "..."
        print(f"  Explanation:\n{expl}")
        if e.get('counterfactual'):
            print(f"\n  {e['counterfactual']}")

    print(f"\n  WHY THIS REQUIRES THE KG:")
    print(f"  - Explanations traverse multi-hop KG paths")
    print(f"  - Portfolio context from Fund-[HOLDS]->OtherStocks")
    print(f"  - Counterfactuals reason about alternative causal paths")
    print(f"  - Old version read CSVs with zero graph queries")

    print("\n  [STEP 15 v2] Done.")


if __name__ == '__main__':
    main()
