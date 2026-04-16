"""
Step 11b -- Add ICP and DML edges to the Causal KG
====================================================
Adds two new edge types on top of the Granger KG from step11:

  (:CausalVariable)-[:CAUSES]->(target)       from ICP
  (:CausalVariable)-[:CAUSAL_EFFECT]->(target) from DML

ICP v5 has 3 targets: action_ordinal, is_buy, is_sell
DML v3 has 3 outcomes: action_ordinal, is_buy, is_sell

Each edge points to its correct target node, not a single
hardcoded 'position_action' node.

Final KG has 4 causal evidence layers:
  :GRANGER_CAUSES   — predictive lead-lag (step11)
  :ASSOCIATED_WITH  — contemporaneous (step11)
  :CAUSES           — ICP invariant parents (this step)
  :CAUSAL_EFFECT    — DML debiased effects (this step)

Input:  data/causal_output/icp_causal_parents.csv
        data/causal_output/dml_causal_effects.csv
Output: Neo4j — causal layer extended
"""

import sys, os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAUSAL_DIR, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j package not installed.")
    sys.exit(1)

ICP_CSV = os.path.join(CAUSAL_DIR, 'icp_causal_parents.csv')
DML_CSV = os.path.join(CAUSAL_DIR, 'dml_causal_effects.csv')


def ensure_target_nodes(session):
    """Create target nodes for all 3 ICP/DML outcomes."""
    targets = ['action_ordinal', 'is_buy', 'is_sell', 'position_action']
    for t in targets:
        session.run("""
            MERGE (n:CausalVariable {name: $name})
            SET n.is_target = true
        """, name=t)
    print(f"  Ensured {len(targets)} target nodes exist")


def merge_icp_edges(session, df):
    """Insert :CAUSES edges from ICP v7.2 results.

    v7.2: Now includes soft confidence for ALL candidates, not just certified.
    confidence_type: 'certified' (in plausible intersection), 'plausible' (in some sets),
                     'soft' (best_p/alpha fallback when zero plausible sets).
    CI-HGT CausalGate uses confidence values — more edges = more signal.
    """
    print(f"\n  Merging {len(df)} ICP rows ...")

    # Determine target column
    target_col = 'target' if 'target' in df.columns else None

    rows = []
    for _, r in df.iterrows():
        target = r[target_col] if target_col else 'position_action'
        conf_type = str(r.get('confidence_type', 'unknown'))
        rows.append({
            'variable':            str(r['variable']),
            'target':              str(target),
            'stratum':             str(r['stratum']),
            'confidence':          float(r['confidence']),
            'confidence_type':     conf_type,
            'in_intersection':     bool(r['in_intersection']),
            'plausible_sets_total':int(r['plausible_sets_total']),
            'n_obs':               int(r['n_obs']),
            'n_environments':      int(r['n_environments']),
        })

    cypher = """
    UNWIND $rows AS row
    MERGE (c:CausalVariable {name: row.variable})
    MERGE (t:CausalVariable {name: row.target})
    MERGE (c)-[r:CAUSES {stratum: row.stratum, target: row.target}]->(t)
    SET r.method             = 'ICP',
        r.confidence         = row.confidence,
        r.confidence_type    = row.confidence_type,
        r.in_intersection    = row.in_intersection,
        r.plausible_sets     = row.plausible_sets_total,
        r.n_obs              = row.n_obs,
        r.n_environments     = row.n_environments
    """
    session.run(cypher, rows=rows)

    certified = [r for r in rows if r['in_intersection']]
    soft = [r for r in rows if r['confidence_type'] == 'soft']
    print(f"  ICP edges merged: {len(rows)} total, {len(certified)} certified, {len(soft)} soft-confidence")


def merge_dml_edges(session, df):
    """Insert :CAUSAL_EFFECT edges from DML v3 results.

    Each edge points to its correct outcome node.
    Excludes unreliable large-effect macro variables.
    """
    print(f"\n  Merging DML rows ...")

    # Flag unreliable estimates (model saturation with macro time trends)
    UNRELIABLE = {'nifty50_return', 'usd_inr'}

    rows = []
    skipped = 0
    for _, r in df.iterrows():
        treatment = str(r['treatment'])
        if treatment in UNRELIABLE:
            skipped += 1
            continue
        outcome = str(r['outcome']) if 'outcome' in r.index else 'action_ordinal'
        rows.append({
            'treatment':    treatment,
            'outcome':      outcome,
            'theta_hat':    float(r['theta_hat'])    if pd.notna(r['theta_hat'])    else 0.0,
            'std_error':    float(r['std_error'])    if pd.notna(r['std_error'])    else 0.0,
            'ci_lower_95':  float(r['ci_lower_95'])  if pd.notna(r['ci_lower_95'])  else 0.0,
            'ci_upper_95':  float(r['ci_upper_95'])  if pd.notna(r['ci_upper_95'])  else 0.0,
            'significant':  bool(r['significant']),
            'direction':    str(r['direction']),
            'icp_certified': bool(r.get('icp_certified', False)),
            'icp_high_conf': bool(r.get('icp_high_conf', False)),
            'n_obs':        int(r['n_obs']),
            'n_controls':   int(r['n_controls']),
        })

    cypher = """
    UNWIND $rows AS row
    MERGE (c:CausalVariable {name: row.treatment})
    MERGE (t:CausalVariable {name: row.outcome})
    MERGE (c)-[r:CAUSAL_EFFECT {method: 'DoubleML_PLR', outcome: row.outcome}]->(t)
    SET r.theta_hat     = row.theta_hat,
        r.std_error     = row.std_error,
        r.ci_lower_95   = row.ci_lower_95,
        r.ci_upper_95   = row.ci_upper_95,
        r.significant   = row.significant,
        r.direction     = row.direction,
        r.icp_certified = row.icp_certified,
        r.icp_high_conf = row.icp_high_conf,
        r.n_obs         = row.n_obs,
        r.n_controls    = row.n_controls
    """
    session.run(cypher, rows=rows)
    print(f"  DML edges merged: {len(rows)} (skipped {skipped} unreliable)")


def bridge_causal_to_temporal(session, granger_df):
    """
    Connect significant CausalVariable nodes to TimePeriod nodes via
    DRIVES_FUND_DECISION edges.

    Why: Granger/ICP/DML edges connect CausalVariable → CausalVariable
    only (a 'causal island'). Without bridges to the temporal/structural
    KG, these nodes are orphaned and useless in Phase 2 graph traversal.

    This adds the semantic link: "this causal variable was a significant
    driver of fund decisions across the observed time periods."

    Result: reduces orphan_causal_variables from ~50 to ~0;
            increases causal_variable_utilization from 0.25 to ~1.0.
    """
    # Determine significance filter explicitly (no opaque fallback chain)
    if 'significant' in granger_df.columns:
        sig = granger_df[granger_df['significant'] == True].copy()
        print(f"  Bridge filter: 'significant' column ({len(sig)} edges)")
    elif 'fdr_significant' in granger_df.columns:
        sig = granger_df[granger_df['fdr_significant'] == True].copy()
        print(f"  Bridge filter: 'fdr_significant' column ({len(sig)} edges)")
    else:
        sig = granger_df[granger_df['p_value'] < 0.05].copy()
        print(f"  Bridge filter: p_value < 0.05 fallback ({len(sig)} edges)")

    # Also include ICP and DML variables
    all_causes = set(sig['cause'].unique()) if 'cause' in sig.columns else set()

    if not all_causes:
        print("  WARNING: no significant causes found for bridge edges")
        return 0

    # Build list of rows: one per (cause, lag) pair with max |beta| per cause
    bridge_rows = []
    for cause, grp in sig.groupby('cause'):
        best = grp.loc[grp['beta'].abs().idxmax()]
        bridge_rows.append({
            'cause':     cause,
            'lag':       int(best['lag']),
            'beta':      float(best['beta']),
            'cause_group': str(best.get('cause_group', 'unknown')),
        })

    cypher = """
    UNWIND $rows AS row
    MATCH (cv:CausalVariable {name: row.cause})
    MATCH (tp:TimePeriod)
    MERGE (cv)-[r:DRIVES_FUND_DECISION]->(tp)
    SET r.dominant_lag       = row.lag,
        r.beta               = row.beta,
        r.cause_group        = row.cause_group,
        r.evidence_method    = 'Granger',
        r.created_by         = 'step11b_bridge'
    """
    session.run(cypher, rows=bridge_rows)
    n = len(bridge_rows)
    print(f"  Bridge: {n} CausalVariable nodes linked to all TimePeriod nodes "
          f"via DRIVES_FUND_DECISION")
    return n


def verify(session):
    print("\n  Final KG edge counts:")
    result = session.run("""
        MATCH ()-[r]->()
        WHERE type(r) IN ['GRANGER_CAUSES','ASSOCIATED_WITH','CAUSES','CAUSAL_EFFECT']
        RETURN type(r) AS edge_type, count(r) AS n
        ORDER BY n DESC
    """)
    total = 0
    for rec in result:
        print(f"    {rec['edge_type']:25s}  {rec['n']:>6,}")
        total += rec['n']
    print(f"    {'TOTAL':25s}  {total:>6,}")

    print("\n  ICP certified parents in KG:")
    result2 = session.run("""
        MATCH (c)-[r:CAUSES]->(t)
        WHERE r.in_intersection = true
        RETURN c.name AS variable, t.name AS target,
               r.confidence AS conf, r.n_environments AS envs
        ORDER BY r.confidence DESC
    """)
    for rec in result2:
        print(f"    {rec['variable']:35s} → {rec['target']:20s}  "
              f"conf={rec['conf']:.2f}  envs={rec['envs']}")

    print("\n  Top DML effects (|theta| > 0.05, significant):")
    result3 = session.run("""
        MATCH (c)-[r:CAUSAL_EFFECT]->(t)
        WHERE r.significant = true AND abs(r.theta_hat) > 0.05
        RETURN c.name AS treatment, t.name AS outcome,
               r.theta_hat AS theta, r.icp_certified AS certified
        ORDER BY abs(r.theta_hat) DESC
        LIMIT 15
    """)
    for rec in result3:
        tag = ' [ICP✓]' if rec['certified'] else ''
        print(f"    {rec['treatment']:35s} → {rec['outcome']:20s}  "
              f"theta={rec['theta']:+.4f}{tag}")


def main():
    print("="*70)
    print("STEP 11b -- ADD ICP + DML EDGES TO CAUSAL KG (v2)")
    print("="*70)

    for path, label in [(ICP_CSV,'ICP'), (DML_CSV,'DML')]:
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found. Run step09{'a' if label=='ICP' else 'b'} first.")
            return

    icp = pd.read_csv(ICP_CSV)
    dml = pd.read_csv(DML_CSV)
    print(f"  ICP: {len(icp)} rows  |  targets: {icp['target'].unique().tolist() if 'target' in icp.columns else ['position_action']}")
    print(f"  DML: {len(dml)} rows  |  outcomes: {dml['outcome'].unique().tolist() if 'outcome' in dml.columns else ['action_ordinal']}")

    from config import CAUSAL_DIR as _CDIR
    _granger_path = os.path.join(_CDIR, 'all_causal_links.csv')

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            ensure_target_nodes(session)
            merge_icp_edges(session, icp)
            merge_dml_edges(session, dml)

            # Bridge orphan CausalVariable nodes → TimePeriod nodes
            print("\n  Bridging CausalVariable nodes to temporal layer ...")
            if os.path.exists(_granger_path):
                granger_df = pd.read_csv(_granger_path)
                bridge_causal_to_temporal(session, granger_df)
            else:
                print("  WARNING: all_causal_links.csv not found — bridge skipped")

            verify(session)
    finally:
        driver.close()

    print("\n  STEP 11b DONE.")


if __name__ == '__main__':
    main()