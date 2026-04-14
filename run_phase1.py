"""
Phase 1 Orchestrator: Data Pipeline + Knowledge Graph Construction
=================================================================
Runs steps 00-12 in dependency order.

Usage:
    python run_phase1.py                # Run all steps
    python run_phase1.py --from 5       # Resume from step 5
    python run_phase1.py --only 3 4     # Run only steps 3 and 4
    python run_phase1.py --skip-kite    # Skip Kite API steps (use existing data)
"""

import argparse
import importlib
import os
import sys
import time
import traceback

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STEP_REGISTRY = [
    # (step_num, module_path, description, requires_api)
    (0,  'step00_build_mapping',               'Build ISIN-Symbol Mapping',          False),
    (1,  'phase1_data.step01_load_portfolio',   'Load & Validate Portfolio',          False),
    (2,  'phase1_data.step02_merge_fundamentals','Merge CMIE Fundamentals',           False),
    (3,  'phase1_data.step03_fetch_kite_ohlcv', 'Fetch Kite OHLCV',                  True),
    (4,  'phase1_data.step04_technical_indicators','Compute Technical Indicators',     False),
    (5,  'phase1_data.step05_finbert_sentiment', 'FinBERT Sentiment Analysis',        False),
    (6,  'phase1_data.step06_macro_indicators',  'Load Macro Indicators',             False),
    (7,  'phase1_data.step07_build_causal_dataset','Build Causal Discovery Dataset',  False),
    (8,  'phase1_data.step08_feature_engineering','Feature Engineering for LPCMCI',   False),
    (9,  'phase1_kg.step09_lpcmci_discovery',    'LPCMCI Causal Discovery',           False),
    (10, 'phase1_kg.step10_build_temporal_kg',   'Build Temporal KG in Neo4j',        False),
    (11, 'phase1_kg.step11_build_causal_kg',     'Build Causal KG in Neo4j',          False),
    (12, 'phase1_kg.step12_intrinsic_evaluation','Intrinsic KG Evaluation',           False),
]


def run_step(step_num, module_path, description, skip_kite=False, requires_api=False):
    """Import and run a single step's main() function."""
    if skip_kite and requires_api:
        print(f"\n{'='*70}")
        print(f"STEP {step_num:02d}: {description} — SKIPPED (--skip-kite)")
        print(f"{'='*70}")
        return True

    print(f"\n{'='*70}")
    print(f"STEP {step_num:02d}: {description}")
    print(f"{'='*70}")

    t0 = time.time()
    try:
        module = importlib.import_module(module_path)
        module.main()
        elapsed = time.time() - t0
        print(f"\n  Step {step_num:02d} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  Step {step_num:02d} FAILED after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Run Phase 1 pipeline')
    parser.add_argument('--from', type=int, default=0, dest='from_step',
                        help='Start from this step number')
    parser.add_argument('--only', type=int, nargs='+', default=None,
                        help='Run only these step numbers')
    parser.add_argument('--skip-kite', action='store_true',
                        help='Skip steps requiring Kite API')
    parser.add_argument('--stop-on-error', action='store_true',
                        help='Stop pipeline on first error')
    args = parser.parse_args()

    print("=" * 70)
    print("FUND MANAGER KG — PHASE 1 PIPELINE")
    print("=" * 70)
    t_start = time.time()

    results = {}
    for step_num, module_path, description, requires_api in STEP_REGISTRY:
        # Filter steps
        if args.only and step_num not in args.only:
            continue
        if step_num < args.from_step:
            continue

        success = run_step(step_num, module_path, description,
                          skip_kite=args.skip_kite, requires_api=requires_api)
        results[step_num] = success

        if not success and args.stop_on_error:
            print(f"\nStopping: step {step_num} failed (--stop-on-error)")
            break

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"PHASE 1 PIPELINE SUMMARY")
    print(f"{'='*70}")
    for step_num, success in results.items():
        status = "OK" if success else "FAILED"
        desc = next(d for n, _, d, _ in STEP_REGISTRY if n == step_num)
        print(f"  Step {step_num:02d}: [{status:6s}] {desc}")
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"\n  WARNING: Steps {failed} failed. Check logs above.")
    else:
        print(f"\n  All steps completed successfully!")
        print(f"  Next: Run run_phase2.py for GraphRAG + Portfolio Construction")


if __name__ == '__main__':
    main()
