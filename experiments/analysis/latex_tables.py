"""
LaTeX Table Generator

Generates publication-ready LaTeX tables from experiment results.

Tables follow journal formatting conventions with proper statistical notation.
"""

import json
from pathlib import Path
from typing import Dict, Optional


class LaTeXTableGenerator:
    """Generate publication-ready LaTeX tables from experiment results."""

    @staticmethod
    def _format_pvalue(p: float) -> str:
        """Format p-value with significance markers."""
        if p < 0.001:
            return "$<$0.001***"
        elif p < 0.01:
            return f"{p:.3f}**"
        elif p < 0.05:
            return f"{p:.3f}*"
        else:
            return f"{p:.3f}"

    @staticmethod
    def _format_ci(lower: float, upper: float) -> str:
        """Format confidence interval."""
        return f"[{lower:.3f}, {upper:.3f}]"

    @staticmethod
    def generate_baseline_table(
        results_path: str,
        stats_path: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate LaTeX table for baseline comparison (Table 1).

        Args:
            results_path: Path to summary.json from baseline comparison
            stats_path: Path to statistics.json (optional)
            output_path: Path to save .tex file (optional)

        Returns:
            LaTeX table string
        """
        with open(results_path, 'r') as f:
            summary = json.load(f)

        results = summary['results']

        # Load statistics if available
        stats = {}
        if stats_path and Path(stats_path).exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)

        # Extract values
        ecr = results['ecr']
        sc = results['self_consistency']
        vanilla = results['vanilla']

        # Get p-values and effect sizes
        ecr_vs_sc_p = stats.get('ecr_vs_sc', {}).get('p_value', 1.0)
        ecr_vs_sc_d = stats.get('ecr_vs_sc', {}).get('cohens_d', 0.0)
        ecr_vs_vanilla_p = stats.get('ecr_vs_vanilla', {}).get('p_value', 1.0)
        ecr_vs_vanilla_d = stats.get('ecr_vs_vanilla', {}).get('cohens_d', 0.0)

        # Get confidence intervals
        ecr_ci = stats.get('ecr_ci', {})
        sc_ci = stats.get('sc_ci', {})
        vanilla_ci = stats.get('vanilla_ci', {})

        table = r"""
\begin{table}[t]
\centering
\caption{Baseline Comparison on TruthfulQA (MC1 Accuracy)}
\label{tab:baseline_comparison}
\begin{tabular}{lcccc}
\toprule
Method & Accuracy & 95\% CI & Cohen's $d$ & $p$-value \\
\midrule
Vanilla (Single-shot) & %.1f\%% & %s & -- & -- \\
Self-Consistency & %.1f\%% & %s & %.2f & %s \\
\textbf{ECR (Ours)} & \textbf{%.1f\%%} & %s & %.2f & %s \\
\bottomrule
\end{tabular}
\vspace{0.5em}

\footnotesize{$n=%d$ questions. Statistical significance: *$p<0.05$, **$p<0.01$, ***$p<0.001$. \\
Cohen's $d$ and $p$-values compare against ECR using paired $t$-test.}
\end{table}
""" % (
            vanilla['accuracy'] * 100,
            LaTeXTableGenerator._format_ci(
                vanilla_ci.get('lower', 0) * 100,
                vanilla_ci.get('upper', 1) * 100
            ) if vanilla_ci else "--",
            sc['accuracy'] * 100,
            LaTeXTableGenerator._format_ci(
                sc_ci.get('lower', 0) * 100,
                sc_ci.get('upper', 1) * 100
            ) if sc_ci else "--",
            ecr_vs_sc_d,
            LaTeXTableGenerator._format_pvalue(ecr_vs_sc_p),
            ecr['accuracy'] * 100,
            LaTeXTableGenerator._format_ci(
                ecr_ci.get('lower', 0) * 100,
                ecr_ci.get('upper', 1) * 100
            ) if ecr_ci else "--",
            ecr_vs_vanilla_d,
            LaTeXTableGenerator._format_pvalue(ecr_vs_vanilla_p),
            ecr.get('total', summary.get('metadata', {}).get('n_questions', 100))
        )

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(table)

        return table

    @staticmethod
    def generate_ablation_table(
        results_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate LaTeX table for ablation study (Table 2).

        Args:
            results_path: Path to results.json from ablation study
            output_path: Path to save .tex file (optional)

        Returns:
            LaTeX table string
        """
        with open(results_path, 'r') as f:
            data = json.load(f)

        configs = data['configurations']
        ranking = data.get('ranking', list(configs.keys()))
        stats = data.get('statistics', {}).get('vs_full', {})

        # Build table rows
        rows = []
        for name in ranking:
            cfg = configs[name]
            weights = cfg['weights']

            # Format weight string
            weight_str = (f"({weights['alpha']:.2f}, {weights['beta']:.2f}, "
                         f"{weights['gamma']:.2f}, {weights['delta']:.2f}, "
                         f"{weights['epsilon']:.2f})")

            # Get statistics vs full
            cfg_stats = stats.get(name, {})
            p_value = cfg_stats.get('t_test', {}).get('p_value', 1.0)
            cohens_d = cfg_stats.get('cohens_d', 0.0)

            # Delta vs full
            delta = cfg.get('delta_vs_full')
            delta_str = f"{delta:+.1f}\\%" if delta is not None else "--"

            # Bold if this is full or best
            if name == 'full':
                acc_str = f"\\textbf{{{cfg['accuracy']*100:.1f}\\%}}"
            else:
                acc_str = f"{cfg['accuracy']*100:.1f}\\%"

            rows.append(f"{cfg['description']:<35} & {acc_str} & {delta_str} & "
                       f"{cfg['mean_cci']:.3f} & {LaTeXTableGenerator._format_pvalue(p_value)}")

        rows_str = " \\\\\n".join(rows)

        table = r"""
\begin{table}[t]
\centering
\caption{Ablation Study: Individual Metric Contributions}
\label{tab:ablation}
\begin{tabular}{lcccc}
\toprule
Configuration & Accuracy & $\Delta$ Full & Mean CCI & $p$-value \\
\midrule
%s \\
\bottomrule
\end{tabular}
\vspace{0.5em}

\footnotesize{$p$-values from paired $t$-test vs. Full ECR. CCI weights: $(\alpha, \beta, \gamma, \delta, \epsilon)$.}
\end{table}
""" % rows_str

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(table)

        return table

    @staticmethod
    def generate_domain_table(
        results_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate LaTeX table for domain calibration (Table 3).

        Args:
            results_path: Path to results.json from domain calibration
            output_path: Path to save .tex file (optional)

        Returns:
            LaTeX table string
        """
        with open(results_path, 'r') as f:
            data = json.load(f)

        domains = data['domains']
        stats = data.get('statistics', {})

        # Build table rows
        rows = []
        total_default = 0
        total_calibrated = 0
        total_n = 0

        for name, domain in domains.items():
            weights = domain['weights']

            # Weight emphasis (show which weights are different from 0.20)
            weight_notes = []
            if weights['beta'] > 0.25:
                weight_notes.append(f"$\\beta$={weights['beta']:.2f}")
            if weights['gamma'] > 0.22:
                weight_notes.append(f"$\\gamma$={weights['gamma']:.2f}")
            weight_str = ", ".join(weight_notes) if weight_notes else "equal"

            # Get p-value
            domain_stats = stats.get(name, {})
            p_value = domain_stats.get('t_test', {}).get('p_value', 1.0)

            # Improvement
            improvement = domain['improvement']
            if improvement > 0:
                imp_str = f"\\textbf{{+{improvement*100:.1f}\\%}}"
            else:
                imp_str = f"{improvement*100:+.1f}\\%"

            rows.append(f"{name.capitalize():<12} & {domain['default_accuracy']*100:.1f}\\% & "
                       f"{domain['calibrated_accuracy']*100:.1f}\\% & {imp_str} & "
                       f"{weight_str} & {LaTeXTableGenerator._format_pvalue(p_value)}")

            total_default += domain['default_accuracy'] * domain['n_questions']
            total_calibrated += domain['calibrated_accuracy'] * domain['n_questions']
            total_n += domain['n_questions']

        # Add overall row
        if total_n > 0:
            overall_default = total_default / total_n
            overall_calibrated = total_calibrated / total_n
            overall_improvement = overall_calibrated - overall_default
            imp_str = f"+{overall_improvement*100:.1f}\\%" if overall_improvement > 0 else f"{overall_improvement*100:.1f}\\%"
            rows.append(f"\\midrule\n\\textbf{{Overall}} & {overall_default*100:.1f}\\% & "
                       f"{overall_calibrated*100:.1f}\\% & {imp_str} & -- & --")

        rows_str = " \\\\\n".join(rows)

        table = r"""
\begin{table}[t]
\centering
\caption{Domain-Specific Calibration Results}
\label{tab:domain_calibration}
\begin{tabular}{lccccc}
\toprule
Domain & Default & Calibrated & Improvement & Key Weights & $p$-value \\
\midrule
%s \\
\bottomrule
\end{tabular}
\vspace{0.5em}

\footnotesize{Default uses equal weights ($\alpha=\beta=\gamma=\delta=\epsilon=0.20$). \\
$\beta$: CR weight (contradiction rate), $\gamma$: TS weight (trajectory smoothness).}
\end{table}
""" % rows_str

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(table)

        return table

    @staticmethod
    def generate_metrics_table(
        results_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate LaTeX table showing ECR metric statistics.

        Args:
            results_path: Path to summary.json with metric means
            output_path: Path to save .tex file (optional)

        Returns:
            LaTeX table string
        """
        with open(results_path, 'r') as f:
            summary = json.load(f)

        ecr = summary['results']['ecr']
        metric_means = ecr.get('metric_means', {})

        table = r"""
\begin{table}[t]
\centering
\caption{ECR Coherence Metrics Summary}
\label{tab:ecr_metrics}
\begin{tabular}{lcc}
\toprule
Metric & Mean & Interpretation \\
\midrule
EVB (Evaluative Variance) & %.3f & Lower is better \\
CR (Contradiction Rate) & %.3f & Lower is better \\
TS (Trajectory Smoothness) & %.3f & Higher is better \\
ES (Expectation Stability) & %.3f & Higher is better \\
PD (Policy Divergence) & %.3f & Lower is better \\
\midrule
\textbf{CCI (Composite Index)} & \textbf{%.3f} & Higher is better \\
\bottomrule
\end{tabular}
\end{table}
""" % (
            metric_means.get('evb', 0),
            metric_means.get('cr', 0),
            metric_means.get('ts', 0),
            metric_means.get('es', 0),
            metric_means.get('pd', 0),
            ecr.get('mean_cci', 0)
        )

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(table)

        return table

    @staticmethod
    def generate_all_tables(
        baseline_dir: str,
        ablation_dir: str,
        domain_dir: str,
        output_dir: str
    ):
        """
        Generate all tables from experiment results.

        Args:
            baseline_dir: Directory with baseline comparison results
            ablation_dir: Directory with ablation study results
            domain_dir: Directory with domain calibration results
            output_dir: Directory to save .tex files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Baseline comparison table
        baseline_summary = Path(baseline_dir) / "summary.json"
        baseline_stats = Path(baseline_dir) / "statistics.json"
        if baseline_summary.exists():
            table = LaTeXTableGenerator.generate_baseline_table(
                str(baseline_summary),
                str(baseline_stats) if baseline_stats.exists() else None,
                str(output_path / "baseline_comparison.tex")
            )
            print(f"Generated: {output_path / 'baseline_comparison.tex'}")

            # Also generate metrics table
            LaTeXTableGenerator.generate_metrics_table(
                str(baseline_summary),
                str(output_path / "ecr_metrics.tex")
            )
            print(f"Generated: {output_path / 'ecr_metrics.tex'}")

        # Ablation table
        ablation_results = Path(ablation_dir) / "results.json"
        if ablation_results.exists():
            LaTeXTableGenerator.generate_ablation_table(
                str(ablation_results),
                str(output_path / "ablation_study.tex")
            )
            print(f"Generated: {output_path / 'ablation_study.tex'}")

        # Domain calibration table
        domain_results = Path(domain_dir) / "results.json"
        if domain_results.exists():
            LaTeXTableGenerator.generate_domain_table(
                str(domain_results),
                str(output_path / "domain_calibration.tex")
            )
            print(f"Generated: {output_path / 'domain_calibration.tex'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate LaTeX tables from experiment results")
    parser.add_argument("--baseline-dir", type=str, required=True,
                        help="Directory with baseline comparison results")
    parser.add_argument("--ablation-dir", type=str, required=True,
                        help="Directory with ablation study results")
    parser.add_argument("--domain-dir", type=str, required=True,
                        help="Directory with domain calibration results")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for .tex files")

    args = parser.parse_args()

    LaTeXTableGenerator.generate_all_tables(
        args.baseline_dir,
        args.ablation_dir,
        args.domain_dir,
        args.output_dir
    )
