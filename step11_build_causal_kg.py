"""
Step 11 -- Build Causal KG from Panel Granger v2 Results
=========================================================
Reads panel_granger_v2_* files produced by step09_targeted_pannel.py
and builds the Granger causal layer in Neo4j on top of the temporal KG.

Schema added to Neo4j:
  (:CausalVariable) nodes for each cause + target
  (:CausalAnalysis) node for the panel Granger analysis
  (:DomainConcept)  nodes per cause_group
  (:MarketCapStratum) nodes

  (CausalVariable)-[:GRANGER_CAUSES {...}]->(CausalVariable)
  (CausalVariable)-[:ASSOCIATED_WITH {...}]->(CausalVariable)
  (CausalVariable)-[:REPRESENTS]->(DomainConcept)
  (DomainConcept)-[:INFLUENCES]->(DomainConcept)

Also saves:
  data/causal_output/all_causal_links.csv  ← used by step12, step12b, step11b
"""

import sys, os, json, warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAUSAL_DIR, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, BATCH_SIZE

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j package not installed.")
    sys.exit(1)

# Input files from step09_targeted_pannel.py
GRANGER_DIR   = os.path.join(CAUSAL_DIR, 'targeted_panel_v2')
CAUSAL_CSV    = os.path.join(GRANGER_DIR, 'panel_granger_v2_causal_significant.csv')
ASSOC_CSV     = os.path.join(GRANGER_DIR, 'panel_granger_v2_associations.csv')
SUMMARY_JSON  = os.path.join(GRANGER_DIR, 'panel_granger_v2_summary.json')
REGIME_CSV    = os.path.join(GRANGER_DIR, 'panel_granger_v2_regime.csv')

# Output
ALL_LINKS_CSV = os.path.join(CAUSAL_DIR, 'all_causal_links.csv')

CAUSE_GROUP_DESCRIPTIONS = {
    'price_momentum': 'Price & Momentum Signal',
    'herding':        'Cross-Fund Herding Behavior',
    'position_size':  'Position Size & Allocation',
    'risk':           'Risk & Volatility Signal',
    'macro_rates':    'Macro Rate Environment',
    'macro_equity':   'Equity Market Condition',
    'sentiment':      'News Sentiment Signal',
    'other':          'Other Signal',
}


def _effect_label(partial_r2):
    if partial_r2 is None or (isinstance(partial_r2, float) and np.isnan(partial_r2)):
        return 'unknown'
    pr2 = abs(partial_r2)
    if pr2 >= 0.014: return 'large'
    elif pr2 >= 0.006: return 'medium'
    elif pr2 >= 0.002: return 'small'
    return 'negligible'


class CausalKGBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with self.driver.session() as s:
            s.run("RETURN 1").single()
        print("  Neo4j connected.")
        self.stats = defaultdict(int)

    def close(self):
        self.driver.close()

    def _run(self, cypher, params=None):
        with self.driver.session() as s:
            return s.run(cypher, params or {}).data()

    def _batch(self, cypher, records, bs=BATCH_SIZE):
        total = 0
        for i in range(0, len(records), bs):
            with self.driver.session() as s:
                s.run(cypher, {'batch': records[i:i+bs]})
            total += len(records[i:i+bs])
        return total

    def create_constraints(self):
        for stmt in [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (cv:CausalVariable) REQUIRE cv.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ca:CausalAnalysis) REQUIRE ca.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (dc:DomainConcept) REQUIRE dc.name IS UNIQUE",
        ]:
            try: self._run(stmt)
            except: pass

    def clear_causal_layer(self):
        print("  Clearing old causal layer...")
        for label in ['CausalVariable', 'CausalAnalysis', 'DomainConcept']:
            self._run(f"MATCH (n:{label}) DETACH DELETE n")
        for rel in ['GRANGER_CAUSES','ASSOCIATED_WITH','REPRESENTS','INFLUENCES',
                    'DISCOVERED_IN','CAUSES','CAUSAL_EFFECT']:
            try: self._run(f"MATCH ()-[r:{rel}]->() DELETE r")
            except: pass

    def create_nodes(self, all_vars, cause_groups):
        # CausalVariable nodes
        records = [{'name': v} for v in list(all_vars) + ['position_action']]
        n = self._batch("""
            UNWIND $batch AS row
            MERGE (cv:CausalVariable {name: row.name})
        """, records)
        self.stats['CausalVariable'] = n
        print(f"  Created {n} CausalVariable nodes")

        # DomainConcept nodes
        dc_records = [
            {'name': grp, 'description': CAUSE_GROUP_DESCRIPTIONS.get(grp, grp)}
            for grp in cause_groups
        ]
        dc_records.append({'name': 'fund_decision', 'description': 'Fund Decision'})
        n = self._batch("""
            UNWIND $batch AS row
            MERGE (dc:DomainConcept {name: row.name})
            SET dc.description = row.description
        """, dc_records)
        self.stats['DomainConcept'] = n
        print(f"  Created {n} DomainConcept nodes")

        # CausalAnalysis node
        self._run("""
            MERGE (ca:CausalAnalysis {name: 'panel_granger_v2'})
            SET ca.method = 'PanelOLS + BH-FDR',
                ca.model  = 'Entity FE + clustered SE by stock',
                ca.targets = ['action_ordinal','buy_ratio','sell_ratio'],
                ca.lags   = [1,2,3,4,5,6]
        """)
        self.stats['CausalAnalysis'] = 1
        print(f"  Created 1 CausalAnalysis node")

    def create_granger_causes(self, df):
        print(f"\n  Creating GRANGER_CAUSES ({len(df)} links)...")
        records = []
        for _, r in df.iterrows():
            records.append({
                'cause':       r['cause'],
                'target':      r['target'],
                'lag':         int(r['lag']),
                'beta':        float(r['beta']),
                't_stat':      float(r['t_stat']),
                'p_value':     float(r['p_value']),
                'p_fdr':       float(r['p_fdr']),
                'partial_r2':  float(r['partial_r2']),
                'direction':   'positive' if r['beta'] > 0 else 'negative',
                'effect_size': _effect_label(r['partial_r2']),
                'cause_group': str(r.get('cause_group', 'other')),
                'significant': bool(r.get('significant', True)),
                'n_obs':       int(r.get('n_obs', 0)),
            })
        n = self._batch("""
            UNWIND $batch AS row
            MERGE (c:CausalVariable {name: row.cause})
            MERGE (t:CausalVariable {name: row.target})
            MERGE (c)-[r:GRANGER_CAUSES {lag: row.lag, cause: row.cause, target: row.target}]->(t)
            SET r.beta =        row.beta,
                r.t_statistic = row.t_stat,
                r.p_value =     row.p_value,
                r.fdr_p_value = row.p_fdr,
                r.partial_r2 =  row.partial_r2,
                r.direction =   row.direction,
                r.effect_size = row.effect_size,
                r.cause_group = row.cause_group,
                r.significant = row.significant,
                r.n_obs =       row.n_obs,
                r.stratum =     'pooled'
        """, records)
        self.stats['GRANGER_CAUSES'] = n
        print(f"  Created {n} GRANGER_CAUSES edges")

    def create_associations(self, df):
        print(f"\n  Creating ASSOCIATED_WITH ({len(df)} links)...")
        records = []
        for _, r in df.iterrows():
            records.append({
                'cause':      r['cause'],
                'target':     r['target'],
                'beta':       float(r['beta']),
                't_stat':     float(r['t_stat']),
                'p_value':    float(r['p_value']),
                'partial_r2': float(r['partial_r2']),
                'cause_group': str(r.get('cause_group', 'other')),
            })
        n = self._batch("""
            UNWIND $batch AS row
            MERGE (c:CausalVariable {name: row.cause})
            MERGE (t:CausalVariable {name: row.target})
            MERGE (c)-[r:ASSOCIATED_WITH {cause: row.cause, target: row.target}]->(t)
            SET r.lag =         0,
                r.beta =        row.beta,
                r.t_statistic = row.t_stat,
                r.p_value =     row.p_value,
                r.partial_r2 =  row.partial_r2,
                r.cause_group = row.cause_group,
                r.stratum =     'pooled'
        """, records)
        self.stats['ASSOCIATED_WITH'] = n
        print(f"  Created {n} ASSOCIATED_WITH edges")

    def create_represents(self, df_all):
        print("\n  Creating REPRESENTS relationships...")
        pairs = set()
        for _, r in df_all.iterrows():
            pairs.add((r['cause'], str(r.get('cause_group', 'other'))))
        pairs.add(('position_action', 'fund_decision'))
        records = [{'var': v, 'concept': c} for v, c in pairs]
        n = self._batch("""
            UNWIND $batch AS row
            MATCH (cv:CausalVariable {name: row.var})
            MATCH (dc:DomainConcept {name: row.concept})
            MERGE (cv)-[:REPRESENTS]->(dc)
        """, records)
        self.stats['REPRESENTS'] = n
        print(f"  Created {n} REPRESENTS relationships")

    def create_influences(self, df_causal):
        print("  Creating INFLUENCES (domain-level) relationships...")
        pairs = defaultdict(list)
        for _, r in df_causal.iterrows():
            grp = str(r.get('cause_group', 'other'))
            pairs[(grp, 'fund_decision')].append(float(r['partial_r2']))
        records = [
            {'src': src, 'tgt': tgt,
             'count': len(vals), 'max_r2': float(max(vals))}
            for (src, tgt), vals in pairs.items()
        ]
        n = self._batch("""
            UNWIND $batch AS row
            MATCH (src:DomainConcept {name: row.src})
            MATCH (tgt:DomainConcept {name: row.tgt})
            MERGE (src)-[r:INFLUENCES]->(tgt)
            SET r.link_count = row.count, r.max_partial_r2 = row.max_r2
        """, records)
        self.stats['INFLUENCES'] = n
        print(f"  Created {n} INFLUENCES relationships")

    def verify(self):
        print("\n  Verification:")
        for label, cypher in [
            ("CausalVariables",  "MATCH (n:CausalVariable) RETURN COUNT(n) AS c"),
            ("DomainConcepts",   "MATCH (n:DomainConcept) RETURN COUNT(n) AS c"),
            ("GRANGER_CAUSES",   "MATCH ()-[r:GRANGER_CAUSES]->() RETURN COUNT(r) AS c"),
            ("ASSOCIATED_WITH",  "MATCH ()-[r:ASSOCIATED_WITH]->() RETURN COUNT(r) AS c"),
            ("REPRESENTS",       "MATCH ()-[r:REPRESENTS]->() RETURN COUNT(r) AS c"),
            ("INFLUENCES",       "MATCH ()-[r:INFLUENCES]->() RETURN COUNT(r) AS c"),
        ]:
            cnt = self._run(cypher)[0]['c']
            print(f"    {label:25s}: {cnt:>6,}")


def main():
    print("="*70)
    print("STEP 11 -- BUILD CAUSAL KG FROM PANEL GRANGER V2")
    print("="*70)

    # Load files
    for path, label in [(CAUSAL_CSV,'causal'), (ASSOC_CSV,'associations')]:
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found. Run step09_targeted_pannel.py first.")
            return

    causal = pd.read_csv(CAUSAL_CSV)
    assoc  = pd.read_csv(ASSOC_CSV)
    print(f"  Causal links:       {len(causal)}")
    print(f"  Associations:       {len(assoc)}")

    # Combine all for node creation
    df_all   = pd.concat([causal, assoc], ignore_index=True)
    all_vars = set(df_all['cause'].unique()) | set(df_all['target'].unique())
    all_groups = set(df_all['cause_group'].dropna().unique())
    print(f"  Unique variables:   {len(all_vars)}")
    print(f"  Cause groups:       {sorted(all_groups)}")

    # Save combined links CSV for step12, step12b, step11b
    combined = df_all.copy()
    combined['stratum']   = 'pooled'
    combined['effect']    = combined['target']
    combined['edge_type'] = combined['link_type'].map(
        {'causal': 'GRANGER_CAUSES', 'association': 'ASSOCIATED_WITH'})
    combined['strength']  = combined['t_stat']
    combined.to_csv(ALL_LINKS_CSV, index=False)
    print(f"  Saved: {ALL_LINKS_CSV}  ({len(combined)} rows)")

    # Build KG
    builder = CausalKGBuilder()
    try:
        builder.create_constraints()
        builder.clear_causal_layer()

        print("\n--- Nodes ---")
        builder.create_nodes(all_vars, all_groups)

        print("\n--- Relationships ---")
        builder.create_granger_causes(causal)
        builder.create_associations(assoc)
        builder.create_represents(df_all)
        builder.create_influences(causal)

        builder.verify()

        print(f"\n{'='*70}")
        print("  STEP 11 COMPLETE")
        print(f"{'='*70}")

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
    finally:
        builder.close()


if __name__ == '__main__':
    main()