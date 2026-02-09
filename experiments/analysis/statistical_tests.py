"""
Statistical Tests for ECR Experiments

Provides statistical significance testing and effect size computation
for comparing ECR against baseline methods.

Tests included:
- Paired t-test
- McNemar's test (for binary outcomes)
- One-way ANOVA
- Cohen's d effect size
- Confidence intervals
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from scipy import stats


class StatisticalAnalyzer:
    """Statistical analysis utilities for experiment results."""

    @staticmethod
    def paired_t_test(
        scores1: List[Union[int, float]],
        scores2: List[Union[int, float]]
    ) -> Dict:
        """
        Paired t-test for within-subject comparison.

        Use this when comparing two methods on the same set of questions.

        Args:
            scores1: Scores for method 1 (e.g., ECR)
            scores2: Scores for method 2 (e.g., Self-Consistency)

        Returns:
            Dictionary with t_statistic, p_value, significant flags
        """
        if len(scores1) != len(scores2):
            raise ValueError("Score lists must have same length")

        if len(scores1) < 2:
            return {
                't_statistic': 0.0,
                'p_value': 1.0,
                'significant_005': False,
                'significant_001': False,
                'n': len(scores1)
            }

        scores1 = np.array(scores1, dtype=float)
        scores2 = np.array(scores2, dtype=float)

        t_stat, p_value = stats.ttest_rel(scores1, scores2)

        # Handle NaN from constant arrays
        if np.isnan(t_stat):
            t_stat = 0.0
            p_value = 1.0

        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_005': bool(p_value < 0.05),
            'significant_001': bool(p_value < 0.01),
            'n': int(len(scores1)),
            'mean_diff': float(np.mean(scores1) - np.mean(scores2))
        }

    @staticmethod
    def cohens_d(
        group1: List[Union[int, float]],
        group2: List[Union[int, float]]
    ) -> float:
        """
        Compute Cohen's d effect size.

        Interpretation:
        - Small: d = 0.2
        - Medium: d = 0.5
        - Large: d = 0.8

        Args:
            group1: Scores for group 1
            group2: Scores for group 2

        Returns:
            Cohen's d value
        """
        group1 = np.array(group1, dtype=float)
        group2 = np.array(group2, dtype=float)

        n1, n2 = len(group1), len(group2)

        if n1 < 2 or n2 < 2:
            return 0.0

        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        d = (np.mean(group1) - np.mean(group2)) / pooled_std

        return float(d)

    @staticmethod
    def mcnemar_test(
        correct1: List[bool],
        correct2: List[bool]
    ) -> Dict:
        """
        McNemar's test for paired binary outcomes.

        Better suited than t-test for comparing binary accuracy on same items.

        Args:
            correct1: Boolean correctness for method 1
            correct2: Boolean correctness for method 2

        Returns:
            Dictionary with chi_square, p_value, contingency table
        """
        if len(correct1) != len(correct2):
            raise ValueError("Lists must have same length")

        # Build contingency table
        # b: method1 correct, method2 wrong
        # c: method1 wrong, method2 correct
        b = sum(1 for c1, c2 in zip(correct1, correct2) if c1 and not c2)
        c = sum(1 for c1, c2 in zip(correct1, correct2) if not c1 and c2)

        # Also compute a and d for completeness
        a = sum(1 for c1, c2 in zip(correct1, correct2) if c1 and c2)
        d = sum(1 for c1, c2 in zip(correct1, correct2) if not c1 and not c2)

        # McNemar statistic with continuity correction
        if b + c == 0:
            chi_square = 0.0
            p_value = 1.0
        else:
            chi_square = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = 1 - stats.chi2.cdf(chi_square, df=1)

        return {
            'chi_square': float(chi_square),
            'p_value': float(p_value),
            'significant_005': bool(p_value < 0.05),
            'significant_001': bool(p_value < 0.01),
            'contingency_table': {
                'both_correct': int(a),
                'method1_only': int(b),
                'method2_only': int(c),
                'both_wrong': int(d)
            },
            'n': int(len(correct1))
        }

    @staticmethod
    def one_way_anova(*groups: List[Union[int, float]]) -> Dict:
        """
        One-way ANOVA for comparing multiple groups.

        Use this for ablation study comparing multiple configurations.

        Args:
            *groups: Variable number of score lists

        Returns:
            Dictionary with f_statistic, p_value, group means
        """
        if len(groups) < 2:
            raise ValueError("Need at least 2 groups for ANOVA")

        # Filter out empty groups
        valid_groups = [np.array(g, dtype=float) for g in groups if len(g) > 0]

        if len(valid_groups) < 2:
            return {
                'f_statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'group_means': [],
                'n_groups': 0
            }

        f_stat, p_value = stats.f_oneway(*valid_groups)

        # Handle NaN
        if np.isnan(f_stat):
            f_stat = 0.0
            p_value = 1.0

        return {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'group_means': [float(np.mean(g)) for g in valid_groups],
            'n_groups': int(len(valid_groups))
        }

    @staticmethod
    def confidence_interval(
        data: List[Union[int, float]],
        confidence: float = 0.95
    ) -> Dict:
        """
        Compute confidence interval for a sample.

        Args:
            data: Sample data
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Dictionary with mean, lower, upper bounds
        """
        data = np.array(data, dtype=float)
        n = len(data)

        if n < 2:
            mean = float(data[0]) if n == 1 else 0.0
            return {
                'mean': mean,
                'lower': mean,
                'upper': mean,
                'std_error': 0.0,
                'n': n,
                'confidence': confidence
            }

        mean = np.mean(data)
        std_error = stats.sem(data)

        # t-distribution critical value
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_crit * std_error

        return {
            'mean': float(mean),
            'lower': float(mean - margin),
            'upper': float(mean + margin),
            'std_error': float(std_error),
            'n': n,
            'confidence': confidence
        }

    @staticmethod
    def bootstrap_ci(
        data: List[Union[int, float]],
        statistic_fn=np.mean,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int = 42
    ) -> Dict:
        """
        Bootstrap confidence interval.

        More robust for small samples or non-normal distributions.

        Args:
            data: Sample data
            statistic_fn: Function to compute statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            seed: Random seed

        Returns:
            Dictionary with point_estimate, lower, upper
        """
        np.random.seed(seed)
        data = np.array(data, dtype=float)
        n = len(data)

        if n < 2:
            point = float(statistic_fn(data)) if n == 1 else 0.0
            return {
                'point_estimate': point,
                'lower': point,
                'upper': point,
                'n_bootstrap': 0
            }

        # Generate bootstrap samples
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_fn(sample))

        bootstrap_stats = np.array(bootstrap_stats)

        # Percentile method for CI
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        return {
            'point_estimate': float(statistic_fn(data)),
            'lower': float(lower),
            'upper': float(upper),
            'n_bootstrap': n_bootstrap,
            'confidence': confidence
        }

    @staticmethod
    def compute_all_comparisons(
        results: Dict[str, List[Union[int, float]]]
    ) -> Dict:
        """
        Compute all pairwise comparisons between methods.

        Args:
            results: Dictionary mapping method names to score lists

        Returns:
            Dictionary with all pairwise comparison results
        """
        methods = list(results.keys())
        comparisons = {}

        for i, method1 in enumerate(methods):
            for method2 in methods[i + 1:]:
                key = f"{method1}_vs_{method2}"
                scores1 = results[method1]
                scores2 = results[method2]

                comparisons[key] = {
                    't_test': StatisticalAnalyzer.paired_t_test(scores1, scores2),
                    'cohens_d': StatisticalAnalyzer.cohens_d(scores1, scores2),
                    'method1_mean': float(np.mean(scores1)),
                    'method2_mean': float(np.mean(scores2))
                }

                # If binary, also do McNemar
                if all(s in [0, 1, True, False] for s in scores1 + scores2):
                    correct1 = [bool(s) for s in scores1]
                    correct2 = [bool(s) for s in scores2]
                    comparisons[key]['mcnemar'] = StatisticalAnalyzer.mcnemar_test(
                        correct1, correct2
                    )

        return comparisons
