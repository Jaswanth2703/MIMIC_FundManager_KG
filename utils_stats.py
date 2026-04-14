"""
Statistical Rigor Module for MIMIC_FundManager_KG
===================================================
Provides research-grade statistical testing for all model comparisons.

This module implements:
  1. Bootstrap confidence intervals (N=200, BCa corrected)
  2. McNemar's test for pairwise binary classifier comparison
  3. Friedman test + Nemenyi post-hoc for multi-model ranking
  4. Cohen's d for effect size
  5. Critical difference diagrams (text-based)

Usage:
  from utils_stats import StatisticalTester
  tester = StatisticalTester()
  ci = tester.bootstrap_ci(y_true, y_pred, metric='f1_weighted')
  p = tester.mcnemar_test(preds_a, preds_b, y_true)
  ranking = tester.friedman_nemenyi(fold_results_dict)

Can also be run standalone to test all existing results:
  python utils_stats.py
"""

import os
import sys
import json
import warnings

import numpy as np
from scipy import stats
from collections import defaultdict

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BOOTSTRAP_N = 200
BOOTSTRAP_SEED = 42
CONFIDENCE_LEVEL = 0.95


class StatisticalTester:
    """Statistical testing suite for model comparison."""

    def __init__(self, n_bootstrap=BOOTSTRAP_N, seed=BOOTSTRAP_SEED,
                 alpha=1 - CONFIDENCE_LEVEL):
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)

    # ================================================================
    # 1. Bootstrap Confidence Intervals
    # ================================================================
    def bootstrap_ci(self, y_true, y_pred, metric='accuracy',
                     n_bootstrap=None):
        """Compute BCa bootstrap confidence interval for a metric.

        Args:
            y_true: ground truth labels
            y_pred: predicted labels
            metric: 'accuracy', 'f1_weighted', 'f1_macro', 'kappa'
            n_bootstrap: override default bootstrap samples

        Returns:
            dict with point_estimate, ci_lower, ci_upper, std, n_bootstrap
        """
        from sklearn.metrics import (accuracy_score, f1_score,
                                      cohen_kappa_score)

        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = len(y_true)

        metric_funcs = {
            'accuracy': lambda yt, yp: accuracy_score(yt, yp),
            'f1_weighted': lambda yt, yp: f1_score(yt, yp, average='weighted',
                                                     zero_division=0),
            'f1_macro': lambda yt, yp: f1_score(yt, yp, average='macro',
                                                  zero_division=0),
            'kappa': lambda yt, yp: cohen_kappa_score(yt, yp),
        }
        func = metric_funcs.get(metric, metric_funcs['accuracy'])

        # Point estimate
        point = func(y_true, y_pred)

        # Bootstrap samples
        boot_vals = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            idx = self.rng.randint(0, n, size=n)
            boot_vals[b] = func(y_true[idx], y_pred[idx])

        # BCa correction
        alpha_lo = self.alpha / 2
        alpha_hi = 1 - self.alpha / 2

        # Bias correction
        z0 = stats.norm.ppf(np.mean(boot_vals < point))
        if np.isinf(z0):
            z0 = 0.0

        # Acceleration (jackknife)
        jackknife = np.zeros(n)
        for i in range(n):
            idx = np.concatenate([np.arange(i), np.arange(i + 1, n)])
            jackknife[i] = func(y_true[idx], y_pred[idx])
        jack_mean = jackknife.mean()
        a_num = np.sum((jack_mean - jackknife) ** 3)
        a_den = 6 * (np.sum((jack_mean - jackknife) ** 2) ** 1.5)
        a = a_num / max(a_den, 1e-12)

        # Adjusted percentiles
        z_lo = stats.norm.ppf(alpha_lo)
        z_hi = stats.norm.ppf(alpha_hi)
        adj_lo = stats.norm.cdf(z0 + (z0 + z_lo) / max(1 - a * (z0 + z_lo), 1e-12))
        adj_hi = stats.norm.cdf(z0 + (z0 + z_hi) / max(1 - a * (z0 + z_hi), 1e-12))

        ci_lower = float(np.percentile(boot_vals, 100 * adj_lo))
        ci_upper = float(np.percentile(boot_vals, 100 * adj_hi))

        return {
            'metric': metric,
            'point_estimate': float(point),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'std': float(boot_vals.std()),
            'n_bootstrap': n_bootstrap,
            'confidence_level': 1 - self.alpha,
            'method': 'BCa bootstrap',
        }

    # ================================================================
    # 2. McNemar's Test
    # ================================================================
    def mcnemar_test(self, preds_a, preds_b, y_true):
        """McNemar's test: are two classifiers significantly different?

        Compares the number of cases where:
          - A is correct but B is wrong (b)
          - B is correct but A is wrong (c)

        H0: both models have the same error rate.

        Returns:
            dict with statistic, p_value, significant, interpretation
        """
        preds_a = np.asarray(preds_a)
        preds_b = np.asarray(preds_b)
        y_true = np.asarray(y_true)

        a_correct = (preds_a == y_true)
        b_correct = (preds_b == y_true)

        # Contingency table
        b = np.sum(a_correct & ~b_correct)  # A right, B wrong
        c = np.sum(~a_correct & b_correct)  # A wrong, B right

        # McNemar's with continuity correction
        if b + c == 0:
            return {'statistic': 0, 'p_value': 1.0, 'significant': False,
                    'interpretation': 'No disagreement between models'}

        if b + c < 25:
            # Exact binomial test for small samples
            p_value = float(stats.binom_test(b, b + c, 0.5))
            stat = float(b)
            method = 'exact_binomial'
        else:
            stat = float((abs(b - c) - 1) ** 2 / (b + c))
            p_value = float(1 - stats.chi2.cdf(stat, df=1))
            method = 'chi2_continuity_corrected'

        return {
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'a_correct_b_wrong': int(b),
            'a_wrong_b_correct': int(c),
            'method': method,
            'interpretation': (
                f"Model A is {'significantly' if p_value < self.alpha else 'NOT significantly'} "
                f"different from Model B (p={p_value:.4f})"
            ),
        }

    # ================================================================
    # 3. Friedman + Nemenyi Post-hoc
    # ================================================================
    def friedman_nemenyi(self, fold_results, metric='accuracy'):
        """Friedman test for multi-model comparison across folds.

        Args:
            fold_results: dict of {model_name: [fold_score_1, fold_score_2, ...]}
            metric: name of the metric (for reporting)

        Returns:
            dict with friedman_stat, friedman_p, model_ranks, nemenyi results
        """
        model_names = sorted(fold_results.keys())
        if len(model_names) < 3:
            return {'error': 'Need at least 3 models for Friedman test'}

        # Align fold counts
        min_folds = min(len(v) for v in fold_results.values())
        if min_folds < 3:
            return {'error': f'Need at least 3 folds (got {min_folds})'}

        data = np.array([fold_results[m][:min_folds] for m in model_names])
        n_models, n_folds = data.shape

        # Friedman test
        try:
            f_stat, f_pval = stats.friedmanchisquare(*data)
        except Exception as e:
            return {'error': str(e)}

        # Average ranks per model
        ranks = np.zeros_like(data)
        for j in range(n_folds):
            col_ranks = stats.rankdata(-data[:, j])  # higher is better → rank 1
            ranks[:, j] = col_ranks
        avg_ranks = ranks.mean(axis=1)

        model_ranks = {m: float(r) for m, r in zip(model_names, avg_ranks)}

        # Nemenyi post-hoc (if Friedman is significant)
        nemenyi = {}
        if f_pval < self.alpha:
            # Critical difference
            q_alpha = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
                       7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
            q = q_alpha.get(n_models, 2.343)
            cd = q * np.sqrt(n_models * (n_models + 1) / (6 * n_folds))

            for i in range(n_models):
                for j in range(i + 1, n_models):
                    diff = abs(avg_ranks[i] - avg_ranks[j])
                    sig = diff > cd
                    nemenyi[f"{model_names[i]} vs {model_names[j]}"] = {
                        'rank_diff': float(diff),
                        'critical_diff': float(cd),
                        'significant': bool(sig),
                    }

        # Text-based critical difference diagram
        cd_diagram = self._text_cd_diagram(model_names, avg_ranks)

        return {
            'metric': metric,
            'friedman_statistic': float(f_stat),
            'friedman_p_value': float(f_pval),
            'friedman_significant': bool(f_pval < self.alpha),
            'model_ranks': model_ranks,
            'n_folds': n_folds,
            'n_models': n_models,
            'nemenyi_comparisons': nemenyi,
            'cd_diagram': cd_diagram,
        }

    def _text_cd_diagram(self, names, ranks):
        """Generate a text-based critical difference diagram."""
        pairs = sorted(zip(ranks, names))
        lines = ["Critical Difference Diagram (text):"]
        lines.append("-" * 50)
        for rank, name in pairs:
            bar = "#" * int(rank * 5)
            lines.append(f"  {rank:.2f}  {bar:20s}  {name}")
        lines.append("-" * 50)
        return '\n'.join(lines)

    # ================================================================
    # 4. Cohen's d
    # ================================================================
    @staticmethod
    def cohens_d(group_a, group_b):
        """Compute Cohen's d effect size.

        Interpretation:
          |d| < 0.2: negligible
          0.2 <= |d| < 0.5: small
          0.5 <= |d| < 0.8: medium
          |d| >= 0.8: large
        """
        a = np.asarray(group_a, dtype=float)
        b = np.asarray(group_b, dtype=float)
        na, nb = len(a), len(b)
        pooled_std = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 +
                               (nb - 1) * b.std(ddof=1) ** 2) /
                              (na + nb - 2))
        d = (a.mean() - b.mean()) / max(pooled_std, 1e-12)
        magnitude = ('negligible' if abs(d) < 0.2 else
                      'small' if abs(d) < 0.5 else
                      'medium' if abs(d) < 0.8 else 'large')
        return {
            'cohens_d': float(d),
            'magnitude': magnitude,
            'mean_a': float(a.mean()),
            'mean_b': float(b.mean()),
        }

    # ================================================================
    # 5. Paired t-test with Bonferroni correction
    # ================================================================
    def paired_ttest(self, scores_a, scores_b, n_comparisons=1):
        """Paired t-test with optional Bonferroni correction."""
        a = np.asarray(scores_a, dtype=float)
        b = np.asarray(scores_b, dtype=float)
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        if n < 3:
            return {'error': 'Need at least 3 paired observations'}
        t_stat, p_val = stats.ttest_rel(a, b)
        p_corrected = min(p_val * n_comparisons, 1.0)
        d = self.cohens_d(a, b)
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'p_corrected': float(p_corrected),
            'significant_raw': bool(p_val < self.alpha),
            'significant_corrected': bool(p_corrected < self.alpha),
            'n_comparisons': n_comparisons,
            'cohens_d': d['cohens_d'],
            'effect_magnitude': d['magnitude'],
        }

    # ================================================================
    # 6. Full comparison report
    # ================================================================
    def full_comparison_report(self, results_dict):
        """Generate a complete statistical comparison report.

        Args:
            results_dict: {model_name: {'y_true': [...], 'y_pred': [...],
                                        'fold_accs': [...], 'fold_f1s': [...]}}
        Returns:
            Complete report dict
        """
        report = {'models': list(results_dict.keys())}

        # Bootstrap CIs for each model
        report['bootstrap_ci'] = {}
        for name, r in results_dict.items():
            if 'y_true' in r and 'y_pred' in r:
                for metric in ['accuracy', 'f1_weighted', 'kappa']:
                    key = f"{name}_{metric}"
                    report['bootstrap_ci'][key] = self.bootstrap_ci(
                        r['y_true'], r['y_pred'], metric=metric)

        # Pairwise McNemar tests
        names = list(results_dict.keys())
        report['mcnemar'] = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if ('y_pred' in results_dict[a] and
                    'y_pred' in results_dict[b] and
                    'y_true' in results_dict[a]):
                    key = f"{a}_vs_{b}"
                    report['mcnemar'][key] = self.mcnemar_test(
                        results_dict[a]['y_pred'],
                        results_dict[b]['y_pred'],
                        results_dict[a]['y_true'])

        # Friedman + Nemenyi (if 3+ models with fold results)
        fold_data = {}
        for name, r in results_dict.items():
            if 'fold_accs' in r and len(r['fold_accs']) >= 3:
                fold_data[name] = r['fold_accs']
        if len(fold_data) >= 3:
            report['friedman_nemenyi'] = self.friedman_nemenyi(fold_data)

        # Pairwise effect sizes
        report['effect_sizes'] = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if 'fold_accs' in results_dict[a] and 'fold_accs' in results_dict[b]:
                    report['effect_sizes'][f"{a}_vs_{b}"] = self.paired_ttest(
                        results_dict[a]['fold_accs'],
                        results_dict[b]['fold_accs'],
                        n_comparisons=len(names) * (len(names) - 1) // 2)

        return report


# ============================================================
# Standalone: test all existing results
# ============================================================
def main():
    print("=" * 70)
    print("STATISTICAL RIGOR MODULE — Testing Existing Results")
    print("=" * 70)

    try:
        from config import FINAL_DIR, EVAL_DIR
    except ImportError:
        FINAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'data', 'final')
        EVAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'data', 'evaluation')

    tester = StatisticalTester()

    # Load any existing results
    results = {}
    for name, fname in [('CBR-KG', 'cbr_metrics.json'),
                        ('PathModel', 'path_model_results.json'),
                        ('HGT', 'hgt_results.json')]:
        path = os.path.join(FINAL_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
            print(f"  Loaded {name}: {path}")

    if not results:
        print("  No existing results found. Run step13, step13a, step13b first.")
        print("  Module is ready for import: from utils_stats import StatisticalTester")
        return

    # Demo: bootstrap CI on available results
    print("\n  === Bootstrap Confidence Intervals ===")
    for name, r in results.items():
        acc = r.get('overall_accuracy', r.get('accuracy', None))
        if acc is not None:
            print(f"  {name}: accuracy = {acc:.4f}")

    # Load ablation results if available
    ablation_path = os.path.join(EVAL_DIR, 'ablation_results.json')
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            ablation = json.load(f)
        print(f"\n  Loaded ablation results")

        # Friedman test on model architectures
        if 'model_architectures' in ablation:
            fold_data = {}
            for model_name, m in ablation['model_architectures'].items():
                if isinstance(m, dict) and 'fold_accs' in m:
                    fold_data[model_name] = m['fold_accs']
            if len(fold_data) >= 3:
                friedman = tester.friedman_nemenyi(fold_data)
                print(f"\n  === Friedman Test (Model Architectures) ===")
                print(f"  Statistic: {friedman.get('friedman_statistic', '?'):.4f}")
                print(f"  p-value: {friedman.get('friedman_p_value', '?'):.6f}")
                print(f"  Significant: {friedman.get('friedman_significant', '?')}")
                if 'model_ranks' in friedman:
                    print(f"\n  Model Rankings:")
                    for m, r in sorted(friedman['model_ranks'].items(),
                                       key=lambda x: x[1]):
                        print(f"    {m:20s}: rank {r:.2f}")
                if 'cd_diagram' in friedman:
                    print(f"\n{friedman['cd_diagram']}")

    # Save statistical report
    report_path = os.path.join(EVAL_DIR, 'statistical_rigor_report.json')
    os.makedirs(EVAL_DIR, exist_ok=True)
    report = {
        'module': 'utils_stats.StatisticalTester',
        'n_bootstrap': BOOTSTRAP_N,
        'confidence_level': CONFIDENCE_LEVEL,
        'tests_available': ['bootstrap_ci', 'mcnemar', 'friedman_nemenyi',
                            'cohens_d', 'paired_ttest', 'full_comparison_report'],
        'models_found': list(results.keys()),
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {report_path}")
    print("\n  [utils_stats] Done.")


if __name__ == '__main__':
    main()
