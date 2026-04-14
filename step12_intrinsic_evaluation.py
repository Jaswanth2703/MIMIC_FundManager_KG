"""
Step 12 -- Intrinsic Evaluation of Knowledge Graphs (Granger Edition)
=======================================================================
Computes structural, semantic, causal, and temporal metrics for both
the temporal KG and the Granger-based causal KG in Neo4j.

Evaluation Framework (cite these in thesis):
  1. Structural Completeness  - Zaveri et al. (2016), J. Web Semantics
  2. Consistency & Integrity  - Hogan et al. (2021), ACM Computing Surveys
  3. Semantic Coherence        - Paulheim (2017), Semantic Web Journal
  4. Informativeness           - Shannon entropy of edge distributions
  5. Inferential Utility       - Competency question coverage + link prediction

Formulas used:
  Density = |E| / (|V| * (|V|-1))
  Completeness = |E_actual| / |E_possible|
  SignConsistency = |edges_same_sign_across_strata| / |edges_in_multiple_strata|
  Informativeness(H) = -Σ p(type) * log₂(p(type))
  InferentialUtility = |answerable_competency_questions| / |total_questions|

Input:  Neo4j (steps 10, 11)
        data/causal_output/all_causal_links.csv
Output: data/evaluation/evaluation_report.json
        data/evaluation/evaluation_summary.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FEATURES_DIR, CAUSAL_DIR, PORTFOLIO_DIR, EVAL_DIR,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    ALPHA_STRICT, ALPHA_EXPLORE,
)

import json
import traceback
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from neo4j import GraphDatabase


CAUSAL_LINKS_CSV = os.path.join(CAUSAL_DIR, 'all_causal_links.csv')
PORTFOLIO_CLEAN_CSV = os.path.join(PORTFOLIO_DIR, 'portfolio_clean.csv')


def benjamini_hochberg(p_values, alpha=0.05):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return {'rejected': np.array([], dtype=bool), 'adjusted_p': np.array([]),
                'n_rejected': 0, 'fdr_threshold': 0.0}
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    ranks = np.arange(1, n + 1)
    adjusted_p_sorted = np.minimum(1.0, sorted_p * n / ranks)
    for i in range(n - 2, -1, -1):
        adjusted_p_sorted[i] = min(adjusted_p_sorted[i], adjusted_p_sorted[i + 1])
    adjusted_p = np.zeros(n)
    adjusted_p[sorted_idx] = adjusted_p_sorted
    final_rejected = adjusted_p <= alpha
    fdr_threshold = float(np.max(p_values[final_rejected])) if final_rejected.any() else 0.0
    return {'rejected': final_rejected, 'adjusted_p': adjusted_p,
            'n_rejected': int(final_rejected.sum()), 'fdr_threshold': fdr_threshold}


class KGEvaluator:
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        with self.driver.session() as s:
            s.run("RETURN 1").single()
        print("  Neo4j connected.")
        self.metrics = {}

    def close(self):
        if self.driver:
            self.driver.close()

    def _run(self, cypher, params=None):
        with self.driver.session() as s:
            return s.run(cypher, params or {}).data()

    def _val(self, cypher, key='cnt'):
        r = self._run(cypher)
        if r and key in r[0] and r[0][key] is not None:
            return r[0][key]
        return None  # distinguish missing from zero

    # ================================================================
    # 1. STRUCTURAL COMPLETENESS
    # ================================================================
    def evaluate_structural(self):
        """
        Structural Completeness (Zaveri et al., 2016):
        Measures graph topology, schema conformance, and coverage.
        """
        print("\n  === 1. STRUCTURAL COMPLETENESS ===")
        s = {}

        # Node counts
        node_labels = ['Fund', 'Stock', 'Sector', 'TimePeriod', 'MarketRegime',
                       'FundSnapshot', 'StockSnapshot',
                       'CausalVariable', 'CausalAnalysis', 'DomainConcept', 'MarketCapStratum']
        nc = {}
        total_nodes = 0
        for label in node_labels:
            c = self._val(f"MATCH (n:{label}) RETURN COUNT(n) AS cnt")
            if c > 0:
                nc[label] = c
                total_nodes += c
        s['node_counts'] = nc
        s['total_nodes'] = total_nodes
        print(f"    Total nodes: {total_nodes:,}")
        for l, c in nc.items():
            print(f"      {l}: {c:,}")

        # Edge counts
        rel_types = ['HOLDS', 'EXITED', 'BELONGS_TO', 'ACTIVE_IN', 'NEXT',
                     'OF_FUND', 'AT_TIME', 'OF_STOCK', 'IN_REGIME',
                     'GRANGER_CAUSES', 'ASSOCIATED_WITH',
                     'DISCOVERED_IN', 'HAS_VARIABLE', 'REPRESENTS',
                     'INFLUENCES', 'ACTIVE_IN_STRATUM']
        ec = {}
        total_edges = 0
        for rt in rel_types:
            c = self._val(f"MATCH ()-[r:{rt}]->() RETURN COUNT(r) AS cnt")
            if c > 0:
                ec[rt] = c
                total_edges += c
        s['edge_counts'] = ec
        s['total_edges'] = total_edges
        print(f"    Total edges: {total_edges:,}")
        for rt, c in ec.items():
            print(f"      {rt}: {c:,}")

        # Density = |E| / (|V| * (|V|-1))
        if total_nodes > 1:
            s['density'] = total_edges / (total_nodes * (total_nodes - 1))
        else:
            s['density'] = 0
        print(f"    Density: {s['density']:.6f}")

        # Average degree
        if total_nodes > 0:
            s['average_degree'] = (2 * total_edges) / total_nodes
        else:
            s['average_degree'] = 0
        print(f"    Average Degree: {s['average_degree']:.2f}")

        # Schema completeness
        expected_nodes = {'Fund', 'Stock', 'Sector', 'TimePeriod',
                          'CausalVariable', 'CausalAnalysis', 'DomainConcept', 'MarketCapStratum'}
        present_nodes = set(nc.keys())
        s['schema_node_coverage'] = len(present_nodes & expected_nodes) / len(expected_nodes)
        print(f"    Schema node coverage: {s['schema_node_coverage']:.1%}")

        expected_rels = {'HOLDS', 'BELONGS_TO', 'ACTIVE_IN', 'NEXT',
                         'GRANGER_CAUSES', 'REPRESENTS', 'INFLUENCES'}
        present_rels = set(ec.keys())
        s['schema_rel_coverage'] = len(present_rels & expected_rels) / len(expected_rels)
        print(f"    Schema rel coverage: {s['schema_rel_coverage']:.1%}")

        # Connected components approximation
        comp_result = self._run("""
            MATCH (n)
            WHERE NOT (n)--()
            RETURN COUNT(n) AS isolated
        """)
        isolated = comp_result[0]['isolated'] if comp_result else 0
        s['isolated_nodes'] = isolated
        s['connected_ratio'] = 1.0 - (isolated / total_nodes) if total_nodes > 0 else 0
        print(f"    Connected ratio: {s['connected_ratio']:.1%} ({isolated} isolated)")

        self.metrics['structural'] = s
        return s

    # ================================================================
    # 2. CONSISTENCY & INTEGRITY
    # ================================================================
    def evaluate_consistency(self, causal_df=None):
        """
        Consistency & Integrity (Hogan et al., 2021):
        - Sign consistency across strata
        - Statistical integrity (FDR-corrected significance)
        - Schema conformance (no orphan edges)
        """
        print("\n  === 2. CONSISTENCY & INTEGRITY ===")
        c = {}

        # Sign consistency across strata
        if causal_df is not None and not causal_df.empty and 'stratum' in causal_df.columns:
            gc = causal_df[causal_df['edge_type'] == 'GRANGER_CAUSES']
            # Group by (cause, lag) and check sign across strata
            multi_strata = gc.groupby(['cause']).filter(
                lambda g: g['stratum'].nunique() >= 2
            )
            if not multi_strata.empty:
                sign_checks = []
                for cause, grp in multi_strata.groupby('cause'):
                    betas_by_stratum = grp.groupby('stratum')['beta'].mean()
                    signs = np.sign(betas_by_stratum.values)
                    same_sign = len(set(signs)) == 1
                    sign_checks.append(same_sign)
                c['sign_consistency'] = float(np.mean(sign_checks))
                c['variables_in_multiple_strata'] = len(sign_checks)
                print(f"    Sign consistency: {c['sign_consistency']:.1%} "
                      f"({sum(sign_checks)}/{len(sign_checks)} variables)")
            else:
                c['sign_consistency'] = None
                print(f"    Sign consistency: N/A (no multi-strata variables)")
        else:
            c['sign_consistency'] = None

        # Statistical integrity: check all significant edges have |t| > 1.96
        if causal_df is not None and 'strength' in causal_df.columns:
            gc = causal_df[causal_df['edge_type'] == 'GRANGER_CAUSES']
            if not gc.empty:
                valid_sig = (gc['strength'].abs() > 1.96).mean()
                c['t_stat_integrity'] = float(valid_sig)
                print(f"    t-stat integrity (|t|>1.96): {valid_sig:.1%}")

        # FDR analysis
        if causal_df is not None and 'p_value' in causal_df.columns:
            gc = causal_df[causal_df['edge_type'] == 'GRANGER_CAUSES']
            for stratum in gc['stratum'].unique():
                p_vals = gc[gc['stratum'] == stratum]['p_value'].dropna().values
                if len(p_vals) > 0:
                    bh = benjamini_hochberg(p_vals, alpha=0.05)
                    c[f'fdr_{stratum}'] = {
                        'tested': len(p_vals),
                        'significant': bh['n_rejected'],
                        'pct': float(bh['n_rejected'] / len(p_vals) * 100),
                    }
                    print(f"    FDR ({stratum}): {bh['n_rejected']}/{len(p_vals)} "
                          f"({bh['n_rejected']/len(p_vals)*100:.1f}%)")

        # Orphan check: CausalVariables without any edges
        orphan_vars = self._val("""
            MATCH (cv:CausalVariable)
            WHERE NOT (cv)-[:GRANGER_CAUSES|ASSOCIATED_WITH]-()
            RETURN COUNT(cv) AS cnt
        """)
        total_cv = self._val("MATCH (cv:CausalVariable) RETURN COUNT(cv) AS cnt")
        c['orphan_causal_variables'] = orphan_vars
        c['causal_variable_utilization'] = 1.0 - (orphan_vars / total_cv) if total_cv > 0 else 0
        print(f"    Causal variable utilization: {c['causal_variable_utilization']:.1%}")

        self.metrics['consistency'] = c
        return c

    # ================================================================
    # 3. SEMANTIC COHERENCE
    # ================================================================
    def evaluate_semantic(self, portfolio_df=None):
        """
        Semantic Coherence (Paulheim, 2017):
        - Entity coverage vs portfolio
        - Domain concept representation balance
        - Temporal completeness
        """
        print("\n  === 3. SEMANTIC COHERENCE ===")
        sem = {}

        # Stock coverage in KG
        total_stocks = self._val("MATCH (s:Stock) RETURN COUNT(s) AS cnt")
        stocks_with_sector = self._val(
            "MATCH (s:Stock)-[:BELONGS_TO]->(:Sector) RETURN COUNT(DISTINCT s) AS cnt")
        sem['total_stocks'] = total_stocks
        sem['stock_sector_coverage'] = stocks_with_sector / total_stocks if total_stocks else 0
        print(f"    Stocks: {total_stocks}, with sector: {stocks_with_sector} "
              f"({sem['stock_sector_coverage']:.1%})")

        # Entity overlap with portfolio
        if portfolio_df is not None and 'ISIN' in portfolio_df.columns:
            p_isins = set(portfolio_df['ISIN'].dropna().unique())
            kg_isins = {r['isin'] for r in self._run("MATCH (s:Stock) RETURN s.isin AS isin") if r['isin']}
            overlap = len(p_isins & kg_isins)
            sem['entity_coverage_ratio'] = overlap / len(p_isins) if p_isins else 0
            print(f"    Entity coverage: {overlap}/{len(p_isins)} ({sem['entity_coverage_ratio']:.1%})")

        # Domain concept balance
        concept_dist = self._run("""
            MATCH (cv:CausalVariable)-[:REPRESENTS]->(dc:DomainConcept)
            RETURN dc.type AS type, COUNT(cv) AS cnt ORDER BY cnt DESC
        """)
        if concept_dist:
            types = {r['type']: r['cnt'] for r in concept_dist}
            total = sum(types.values())
            sem['concept_distribution'] = types
            # Shannon entropy
            probs = np.array(list(types.values())) / total
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(len(types))
            sem['concept_entropy'] = float(entropy)
            sem['concept_entropy_normalized'] = float(entropy / max_entropy) if max_entropy > 0 else 0
            print(f"    Concept entropy: {entropy:.3f} (normalized: {sem['concept_entropy_normalized']:.3f})")

        # Temporal completeness
        total_tp = self._val("MATCH (t:TimePeriod) RETURN COUNT(t) AS cnt")
        active_tp = self._val(
            "MATCH (:Fund)-[:ACTIVE_IN]->(t:TimePeriod) RETURN COUNT(DISTINCT t) AS cnt")
        sem['temporal_completeness'] = active_tp / total_tp if total_tp else 0
        sem['total_timepoints'] = total_tp
        print(f"    Temporal completeness: {sem['temporal_completeness']:.1%}")

        # Fund and sector coverage
        sem['total_funds'] = self._val("MATCH (f:Fund) RETURN COUNT(f) AS cnt")
        sem['total_sectors'] = self._val("MATCH (s:Sector) RETURN COUNT(s) AS cnt")
        sem['total_regimes'] = self._val("MATCH (mr:MarketRegime) RETURN COUNT(mr) AS cnt")
        print(f"    Funds: {sem['total_funds']}, Sectors: {sem['total_sectors']}, "
              f"Regimes: {sem['total_regimes']}")

        self.metrics['semantic'] = sem
        return sem

    # ================================================================
    # 4. INFORMATIVENESS
    # ================================================================
    def evaluate_informativeness(self, causal_df=None):
        """
        Informativeness: How much useful knowledge does the KG contain?
        - Edge type distribution entropy
        - Effect size distribution
        - Lag distribution
        - Stratum-specific findings
        """
        print("\n  === 4. INFORMATIVENESS ===")
        info = {}

        # Causal edge statistics from Neo4j
        gc_count = self._val("MATCH ()-[r:GRANGER_CAUSES]->() RETURN COUNT(r) AS cnt")
        assoc_count = self._val("MATCH ()-[r:ASSOCIATED_WITH]->() RETURN COUNT(r) AS cnt")
        info['granger_causes_count'] = gc_count
        info['associations_count'] = assoc_count
        info['total_causal_edges'] = gc_count + assoc_count
        print(f"    GRANGER_CAUSES: {gc_count}")
        print(f"    ASSOCIATED_WITH: {assoc_count}")

        # Effect size distribution from CSV
        if causal_df is not None and 'partial_r2' in causal_df.columns:
            gc = causal_df[causal_df['edge_type'] == 'GRANGER_CAUSES']
            pr2 = gc['partial_r_squared'].dropna() if 'partial_r_squared' in gc.columns else gc['partial_r2'].dropna()
            if len(pr2) > 0:
                info['effect_size_stats'] = {
                    'mean': float(pr2.mean()),
                    'median': float(pr2.median()),
                    'max': float(pr2.max()),
                    'above_small': int((pr2 >= 0.02).sum()),
                    'above_medium': int((pr2 >= 0.06).sum()),
                    'above_large': int((pr2 >= 0.14).sum()),
                }
                print(f"    Effect sizes: mean={pr2.mean():.4f}, median={pr2.median():.4f}")
                print(f"      ≥small(0.02): {(pr2>=0.02).sum()}, "
                      f"≥medium(0.06): {(pr2>=0.06).sum()}, "
                      f"≥large(0.14): {(pr2>=0.14).sum()}")

        # Lag distribution
        lag_dist = self._run("""
            MATCH ()-[r:GRANGER_CAUSES]->()
            RETURN r.lag AS lag, COUNT(r) AS cnt ORDER BY r.lag
        """)
        if lag_dist:
            info['lag_distribution'] = {r['lag']: r['cnt'] for r in lag_dist}
            avg_lag = sum(r['lag'] * r['cnt'] for r in lag_dist) / sum(r['cnt'] for r in lag_dist)
            info['avg_lag'] = float(avg_lag)
            print(f"    Average lag: {avg_lag:.2f}")
            for r in lag_dist:
                print(f"      lag={r['lag']}: {r['cnt']}")

        # Stratum-specific counts
        strata_dist = self._run("""
            MATCH ()-[r:GRANGER_CAUSES]->()
            RETURN r.stratum AS stratum, COUNT(r) AS cnt ORDER BY cnt DESC
        """)
        if strata_dist:
            info['strata_distribution'] = {r['stratum']: r['cnt'] for r in strata_dist}
            print(f"    By stratum: {info['strata_distribution']}")

        # Unique causal variables
        info['unique_causes'] = self._val("""
            MATCH (cv:CausalVariable)-[:GRANGER_CAUSES]->()
            RETURN COUNT(DISTINCT cv) AS cnt
        """)
        print(f"    Unique causes: {info['unique_causes']}")

        # Category distribution
        cat_dist = self._run("""
            MATCH (cv:CausalVariable)-[:GRANGER_CAUSES]->()
            RETURN cv.category AS cat, COUNT(DISTINCT cv) AS cnt ORDER BY cnt DESC
        """)
        if cat_dist:
            info['category_distribution'] = {r['cat']: r['cnt'] for r in cat_dist}
            print(f"    By category: {info['category_distribution']}")

        self.metrics['informativeness'] = info
        return info

    # ================================================================
    # 5. INFERENTIAL UTILITY
    # ================================================================
    def evaluate_inferential(self):
        """
        Inferential Utility: Can the KG answer meaningful questions?
        Define competency questions and test if KG can answer them.
        """
        print("\n  === 5. INFERENTIAL UTILITY ===")
        infer = {}

        competency_questions = [
            ("Q1: What variables Granger-cause BUY decisions at lag-1 in mid-cap?",
             """MATCH (cv)-[r:GRANGER_CAUSES]->(t {name:'action_ordinal'})
                WHERE r.lag = 1 AND r.beta > 0
                RETURN COUNT(cv) AS cnt"""),
            ("Q2: What variables Granger-cause SELL decisions (negative beta)?",
             """MATCH (cv)-[r:GRANGER_CAUSES]->(t {name:'action_ordinal'})
                WHERE r.beta < 0 AND r.fdr_p_value < 0.01
                RETURN COUNT(cv) AS cnt"""),
            ("Q3: Is the causal structure different between strata?",
             """MATCH ()-[r:GRANGER_CAUSES]->()
                WITH r.stratum AS stratum, COUNT(r) AS cnt
                RETURN COUNT(DISTINCT stratum) AS cnt"""),
            ("Q4: What macro factors influence fund decisions?",
             """MATCH (cv:CausalVariable)-[r:GRANGER_CAUSES]->(t)
                WHERE cv.name STARTS WITH 'cpi' OR cv.name STARTS WITH 'repo'
                   OR cv.name STARTS WITH 'us_10y' OR cv.name STARTS WITH 'nifty'
                   OR cv.name STARTS WITH 'india_vix'
                RETURN COUNT(DISTINCT cv) AS cnt"""),
            ("Q5: Does sentiment causally predict decisions?",
             """MATCH (cv:CausalVariable)-[r:GRANGER_CAUSES]->(t)
                WHERE cv.name STARTS WITH 'sentiment'
                RETURN COUNT(cv) AS cnt"""),
            ("Q6: What is the strongest causal predictor overall?",
             """MATCH (cv)-[r:GRANGER_CAUSES]->()
                RETURN cv.name AS name, abs(r.t_statistic) AS t ORDER BY t DESC LIMIT 1"""),
            ("Q7: Which domain concepts influence fund decisions?",
             """MATCH (dc:DomainConcept)-[r:INFLUENCES]->(:DomainConcept {name:'Fund Decision'})
                RETURN COUNT(dc) AS cnt"""),
            ("Q8: Are there variables only significant in one stratum?",
             """MATCH (cv)-[r:GRANGER_CAUSES]->()
                WITH cv.name AS var, COLLECT(DISTINCT r.stratum) AS strata
                WHERE SIZE(strata) = 1
                RETURN COUNT(var) AS cnt"""),
            ("Q9: What is the lag distribution?",
             """MATCH ()-[r:GRANGER_CAUSES]->()
                RETURN r.lag AS lag, COUNT(r) AS cnt ORDER BY lag"""),
            ("Q10: How many total causal + temporal edges exist?",
             """MATCH ()-[r]->() RETURN COUNT(r) AS cnt"""),
        ]

        answered = 0
        results = []
        for question, cypher in competency_questions:
            try:
                result = self._run(cypher)
                val = result[0].get('cnt', result[0].get('name', 0)) if result else 0
                is_answered = (val is not None and val != 0 and val != '' and
                               (not isinstance(val, (int, float)) or val > 0))
                answered += is_answered
                status = "✓" if is_answered else "✗"
                results.append({'question': question, 'answered': is_answered, 'value': val})
                print(f"    {status} {question}: {val}")
            except Exception as e:
                results.append({'question': question, 'answered': False, 'value': str(e)})
                print(f"    ✗ {question}: ERROR - {e}")

        infer['competency_questions'] = results
        infer['total_questions'] = len(competency_questions)
        infer['answered'] = answered
        infer['inferential_utility'] = answered / len(competency_questions)
        print(f"\n    Inferential Utility: {answered}/{len(competency_questions)} = "
              f"{infer['inferential_utility']:.1%}")

        # Causal path depth (longest chain through INFLUENCES)
        try:
            path_result = self._run("""
                MATCH path = (s:DomainConcept)-[:INFLUENCES*1..4]->(t:DomainConcept)
                WHERE s <> t
                RETURN [n IN nodes(path) | n.name] AS chain, length(path) AS len
                ORDER BY len DESC LIMIT 3
            """)
            if path_result:
                infer['max_domain_chain_length'] = path_result[0]['len']
                infer['top_chains'] = [{'chain': p['chain'], 'length': p['len']} for p in path_result]
                print(f"    Longest domain chain: {path_result[0]['len']}")
                print(f"      {' → '.join(path_result[0]['chain'])}")
        except Exception:
            pass

        self.metrics['inferential'] = infer
        return infer

    # ================================================================
    # 6. TEMPORAL CONSISTENCY
    # ================================================================
    def evaluate_temporal(self):
        print("\n  === 6. TEMPORAL CONSISTENCY ===")
        t = {}

        # Position action distribution
        action_dist = self._run("""
            MATCH ()-[h:HOLDS]->()
            WHERE h.position_action IS NOT NULL
            RETURN h.position_action AS action, COUNT(h) AS cnt ORDER BY cnt DESC
        """)
        t['position_action_distribution'] = {r['action']: r['cnt'] for r in action_dist}
        total_acts = sum(r['cnt'] for r in action_dist)
        print(f"    Position actions: {total_acts:,}")
        for r in action_dist:
            pct = r['cnt'] / total_acts * 100 if total_acts else 0
            print(f"      {r['action']}: {r['cnt']:,} ({pct:.1f}%)")

        # TimePeriod chain
        chain_count = self._val(
            "MATCH (:TimePeriod)-[:NEXT]->(:TimePeriod) RETURN COUNT(*) AS cnt")
        total_tp = self._val("MATCH (t:TimePeriod) RETURN COUNT(t) AS cnt")
        t['chain_complete'] = chain_count == total_tp - 1 if total_tp > 1 else True
        print(f"    NEXT chain: {chain_count}/{total_tp - 1} ({'complete' if t['chain_complete'] else 'gaps!'})")

        # Average holdings per fund per month
        avg_result = self._run("""
            MATCH (f:Fund)-[h:HOLDS]->(s:Stock)
            WITH f.name AS fund, h.month AS month, COUNT(s) AS n
            RETURN AVG(n) AS avg, MIN(n) AS min, MAX(n) AS max
        """)
        if avg_result and avg_result[0]['avg']:
            t['avg_stocks_per_fund_month'] = float(avg_result[0]['avg'])
            print(f"    Avg stocks/fund/month: {avg_result[0]['avg']:.1f}")

        t['total_holds'] = self._val("MATCH ()-[h:HOLDS]->() RETURN COUNT(h) AS cnt")
        t['total_exits'] = self._val("MATCH ()-[e:EXITED]->() RETURN COUNT(e) AS cnt")
        print(f"    Total HOLDS: {t['total_holds']:,}, EXITED: {t['total_exits']:,}")

        # Market regime distribution
        regime_dist = self._run("""
            MATCH (t:TimePeriod)-[:IN_REGIME]->(mr:MarketRegime)
            RETURN mr.regime_type AS regime, COUNT(t) AS cnt ORDER BY cnt DESC
        """)
        if regime_dist:
            t['regime_distribution'] = {r['regime']: r['cnt'] for r in regime_dist}
            print(f"    Regimes: {t['regime_distribution']}")

        self.metrics['temporal'] = t
        return t

    # ================================================================
    # GENERATE REPORT
    # ================================================================
    def generate_report(self, portfolio_df=None, causal_links_df=None):
        print("\n" + "=" * 70)
        print("  INTRINSIC EVALUATION REPORT")
        print("=" * 70)

        self.evaluate_structural()
        self.evaluate_consistency(causal_links_df)
        self.evaluate_semantic(portfolio_df)
        self.evaluate_informativeness(causal_links_df)
        self.evaluate_inferential()
        self.evaluate_temporal()

        # Overall quality score
        scores = []
        s = self.metrics.get('structural', {})
        if s.get('total_nodes', 0) > 100:
            scores.append(('struct_size', min(1.0, s['total_nodes'] / 10000)))
        if 'schema_node_coverage' in s:
            scores.append(('schema_nodes', s['schema_node_coverage']))
        if 'schema_rel_coverage' in s:
            scores.append(('schema_rels', s['schema_rel_coverage']))
        if 'connected_ratio' in s:
            scores.append(('connected', s['connected_ratio']))

        c = self.metrics.get('consistency', {})
        if c.get('sign_consistency') is not None:
            scores.append(('sign_consistency', c['sign_consistency']))
        if c.get('causal_variable_utilization'):
            scores.append(('var_utilization', c['causal_variable_utilization']))

        sem = self.metrics.get('semantic', {})
        if 'temporal_completeness' in sem:
            scores.append(('temporal_complete', sem['temporal_completeness']))
        if sem.get('entity_coverage_ratio') is not None:
            scores.append(('entity_coverage', sem['entity_coverage_ratio']))
        if 'concept_entropy_normalized' in sem:
            scores.append(('concept_diversity', sem['concept_entropy_normalized']))

        inf = self.metrics.get('inferential', {})
        if 'inferential_utility' in inf:
            scores.append(('inferential', inf['inferential_utility']))

        overall = sum(v for _, v in scores) / len(scores) if scores else 0
        self.metrics['overall_quality_score'] = overall
        self.metrics['component_scores'] = dict(scores)
        self.metrics['generated_at'] = datetime.now().isoformat()

        print(f"\n  {'='*50}")
        print(f"  OVERALL QUALITY SCORE: {overall:.3f}")
        print(f"  {'='*50}")
        for name, score in scores:
            print(f"    {name:25s}: {score:.3f}")

        return self.metrics

    def save_report(self):
        os.makedirs(EVAL_DIR, exist_ok=True)

        path = os.path.join(EVAL_DIR, 'evaluation_report.json')
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        print(f"\n  Saved: {path}")

        rows = []
        for cat, block in self.metrics.items():
            if isinstance(block, dict):
                for k, v in block.items():
                    if isinstance(v, (int, float, str, bool)):
                        rows.append({'category': cat, 'metric': k, 'value': v})
                    elif isinstance(v, dict):
                        for sk, sv in v.items():
                            if isinstance(sv, (int, float, str, bool)):
                                rows.append({'category': cat, 'metric': f'{k}.{sk}', 'value': sv})
            elif isinstance(block, (int, float, str, bool)):
                rows.append({'category': 'overall', 'metric': cat, 'value': block})

        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(EVAL_DIR, 'evaluation_summary.csv'), index=False)
            print(f"  Saved: evaluation_summary.csv")


def main():
    print("=" * 70)
    print("STEP 12 -- INTRINSIC EVALUATION (Granger Edition)")
    print("=" * 70)

    portfolio_df = None
    if os.path.exists(PORTFOLIO_CLEAN_CSV):
        portfolio_df = pd.read_csv(PORTFOLIO_CLEAN_CSV, usecols=['ISIN'], low_memory=False)
        print(f"  Portfolio: {portfolio_df['ISIN'].nunique()} unique ISINs")

    causal_df = None
    if os.path.exists(CAUSAL_LINKS_CSV):
        causal_df = pd.read_csv(CAUSAL_LINKS_CSV)
        print(f"  Causal links: {len(causal_df)}")
    else:
        # Try alternate path
        alt = os.path.join(CAUSAL_DIR, 'causal_links_lpcmci.csv')
        if os.path.exists(alt):
            causal_df = pd.read_csv(alt)
            print(f"  Causal links (compat): {len(causal_df)}")

    evaluator = KGEvaluator()
    try:
        report = evaluator.generate_report(portfolio_df, causal_df)
        evaluator.save_report()
        print(f"\n{'='*70}")
        print(f"  STEP 12 COMPLETE -- Score: {report.get('overall_quality_score', 0):.3f}")
        print(f"{'='*70}")
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
    finally:
        evaluator.close()


if __name__ == '__main__':
    main()