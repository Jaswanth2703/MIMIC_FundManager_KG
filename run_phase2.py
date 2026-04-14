"""
Phase 2 Orchestrator: Fund Manager Imitation from Phase 1 KG
=============================================================
Runs Phase 2 steps in dependency order.

How Phase 1 KG feeds Phase 2:
  step13  (CBR):  reads data/causal_output/icp_causal_parents.csv
                  to select ICP-certified causal features for
                  subgraph fingerprints. The KG's causal structure
                  defines WHAT is similar between two manager decisions.

  step13b (export): queries Neo4j (Phase 1 KG) to build kg_export.pkl.
                    Exports node features, HOLDS edges, GRANGER_CAUSES,
                    CAUSES (ICP), CAUSAL_EFFECT (DML) edge attributes.

  step13b (HGT):  runs Heterogeneous Graph Transformer directly on
                  the exported KG. Phase 1 KG structure IS the model.
                  Predictions flow through: Fund→HOLDS→Stock edges,
                  informed by causal edges from CausalVariable nodes.

  step14b (compare): 5-way comparison. M3 uses ICP causal parents
                     (from Phase 1) as feature selector. M4 is CBR.
                     Shows KG-informed methods vs baselines.

  step15  (XAI):  3-layer explanations grounded in Phase 1 KG:
                  ICP evidence + DML confidence intervals + CBR cases.

  step16  (backtest): walk-forward returns using Phase 1 causal
                      features for portfolio construction.

Usage:
    python run_phase2.py               # Run all Phase 2 steps
    python run_phase2.py --from 14     # Resume from step 14b
    python run_phase2.py --only 13     # Run only CBR (step 13)
    python run_phase2.py --only 13b    # Run only HGT export+train
    python run_phase2.py --skip-hgt    # Skip GPU-required HGT steps
"""

import argparse
import importlib.util
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = os.path.dirname(os.path.abspath(__file__))


def load_and_run(script_path, description):
    """Load a flat .py file and call its main()."""
    spec = importlib.util.spec_from_file_location("_step_mod", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, 'main'):
        mod.main()
    else:
        raise AttributeError(f"No main() in {script_path}")


# Step registry: (id, filename, description, requires_gpu, requires_neo4j)
STEPS = [
    ('13',   'step13_cbr_inference_engine.py',
     'CBR-KG Inference (WL graph kernel + subgraph retrieval)',
     False, False),

    ('13a',  'step13a_causal_path_engine.py',
     'Causal Decision Path Extraction & Mimicry Engine',
     False, False),

    ('13b-export', 'step13b_export_kg_for_gpu.py',
     'Export Phase 1 KG to pkl for HGT',
     False, True),   # requires Neo4j

    ('13b-hgt', 'step13b_rgcn.py',
     'Heterogeneous Graph Transformer on KG',
     True,  False),  # requires GPU + kg_export.pkl

    ('14b',  'step14b_full_comparison.py',
     'Full 5-way model comparison (M0-M4)',
     False, False),

    ('15',   'step15_explainable_ai_v2.py',
     'KG-Grounded Causal Explanations (multi-hop + counterfactual)',
     False, False),

    ('16',   'step16_fixed_real_returns.py',
     'Walk-forward backtest with real returns',
     False, False),

    ('16b',  'step16b_ablation_study.py',
     'Ablation Study Framework (feature/causal/model ablations)',
     False, False),

    ('16c',  'step16c_style_clustering.py',
     'Fund Manager Style Clustering via KG Embeddings',
     False, False),
]


def check_phase1_outputs():
    """Verify Phase 1 KG outputs exist before running Phase 2."""
    from config import CAUSAL_DIR, FEATURES_DIR

    required = {
        'ICP parents': os.path.join(CAUSAL_DIR, 'icp_causal_parents.csv'),
        'DML effects': os.path.join(CAUSAL_DIR, 'dml_causal_effects.csv'),
        'Granger links': os.path.join(CAUSAL_DIR, 'all_causal_links.csv'),
        'Feature panel': os.path.join(FEATURES_DIR, 'LPCMCI_READY.csv'),
    }
    all_ok = True
    for name, path in required.items():
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"    [{status:7s}] {name}: {os.path.basename(path)}")
        if not exists:
            all_ok = False
    return all_ok


def run_step(step_id, filename, description, skip_gpu=False, skip_neo4j=False):
    script_path = os.path.join(BASE, filename)

    if not os.path.exists(script_path):
        print(f"  ERROR: {filename} not found — skipping.")
        return False

    t0 = time.time()
    try:
        load_and_run(script_path, description)
        elapsed = time.time() - t0
        print(f"\n  Step {step_id} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  Step {step_id} FAILED after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Run Phase 2 pipeline')
    parser.add_argument('--from', type=str, default='13', dest='from_step',
                        help='Start from this step id (13, 13b-export, 13b-hgt, 14b, 15, 16)')
    parser.add_argument('--only', type=str, nargs='+', default=None,
                        help='Run only these step ids')
    parser.add_argument('--skip-hgt', action='store_true',
                        help='Skip GPU-required HGT steps (13b-export and 13b-hgt)')
    parser.add_argument('--stop-on-error', action='store_true',
                        help='Stop pipeline on first error')
    args = parser.parse_args()

    print("=" * 70)
    print("FUND MANAGER KG -- PHASE 2: IMITATION ENGINE")
    print("=" * 70)
    print("\n  Phase 1 → Phase 2 connection check:")
    ok = check_phase1_outputs()
    if not ok:
        print("\n  WARNING: Some Phase 1 outputs missing. Run phase 1 first.")
    print()

    step_ids = [s[0] for s in STEPS]
    from_idx = 0
    if args.from_step in step_ids:
        from_idx = step_ids.index(args.from_step)

    t_start = time.time()
    results = {}

    for step_id, filename, description, requires_gpu, requires_neo4j in STEPS:
        # Skip checks
        if args.only and step_id not in args.only:
            continue
        if step_ids.index(step_id) < from_idx:
            continue
        if args.skip_hgt and step_id in ('13b-export', '13b-hgt'):
            print(f"\n  Skipping step {step_id} (--skip-hgt)")
            continue

        print(f"\n{'='*70}")
        print(f"STEP {step_id}: {description}")
        if requires_gpu:
            print(f"  [NOTE: GPU recommended for this step]")
        if requires_neo4j:
            print(f"  [NOTE: Neo4j must be running (Phase 1 KG)]")
        print(f"{'='*70}")

        success = run_step(step_id, filename, description,
                           skip_gpu=(args.skip_hgt and requires_gpu),
                           skip_neo4j=False)
        results[step_id] = success

        if not success and args.stop_on_error:
            print(f"\n  Stopping: step {step_id} failed.")
            break

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print("PHASE 2 PIPELINE SUMMARY")
    print(f"{'='*70}")
    for step_id, _, description, _, _ in STEPS:
        if step_id not in results:
            print(f"  Step {step_id:12s}: [SKIPPED ] {description}")
        else:
            status = "OK     " if results[step_id] else "FAILED "
            print(f"  Step {step_id:12s}: [{status}] {description}")

    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"\n  WARNING: Steps {failed} failed.")
    elif results:
        print(f"\n  All Phase 2 steps completed successfully!")

    print("\n  Phase 2 KG utilisation summary:")
    print("    CBR (13):     WL graph kernel on Neo4j subgraphs + ICP features")
    print("    Paths(13a):   KG path traversal -> Transformer mimicry model")
    print("    HGT (13b):    HGT + CI-HGT (causal-gated attention) on KG")
    print("    Compare(14b): M3/M4 use KG causal knowledge vs flat baselines")
    print("    XAI  (15):    Multi-hop KG paths + counterfactuals + CBR analogies")
    print("    Backtest(16): Causal feature portfolio construction")
    print("    Ablation(16b):Proves each KG layer contributes measurably")
    print("    Styles(16c):  KG embedding clusters reveal manager styles")


if __name__ == '__main__':
    main()
