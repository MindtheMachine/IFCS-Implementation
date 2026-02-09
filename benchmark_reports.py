"""
Benchmark Report Generator
Generate evaluation reports in CSV, JSON, and HTML formats
"""

import json
import csv
from typing import List, Dict, Any
from datetime import datetime


class BenchmarkReportGenerator:
    """Generate evaluation reports in multiple formats"""

    @staticmethod
    def generate_csv_report(
        results: List[Any],  # List[BenchmarkEvaluationResult]
        output_path: str
    ):
        """Generate CSV with per-example results

        CSV Schema:
            - example_id
            - prompt (truncated to 100 chars)
            - baseline_[metric_name]
            - regulated_[metric_name]
            - metric_improvement
            - ecr_fired, cp_type1_fired, ifcs_fired, cp_type2_fired
            - ifcs_risk_reduction_pct (if available)
            - processing_time_s
            - error

        Args:
            results: List of BenchmarkEvaluationResult
            output_path: Path to save CSV file
        """
        if not results:
            print(f"[Warning] No results to write to CSV")
            return

        # Determine metric columns based on first result
        metric_columns = []
        if results[0].baseline_metrics:
            metric_columns = list(results[0].baseline_metrics.scores.keys())

        # Define CSV columns
        columns = ['example_id', 'prompt_preview']

        # Add baseline metrics columns
        for metric in metric_columns:
            columns.append(f'baseline_{metric}')

        # Add regulated metrics columns
        for metric in metric_columns:
            columns.append(f'regulated_{metric}')

        # Add improvement columns
        for metric in metric_columns:
            columns.append(f'{metric}_improvement')

        # Add mechanism columns
        columns.extend(['ecr_fired', 'cp_type1_fired', 'ifcs_fired', 'cp_type2_fired'])

        # Add other columns
        columns.extend(['processing_time_s', 'error'])

        # Write CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for result in results:
                row = {
                    'example_id': result.example_id,
                    'prompt_preview': result.prompt[:100] if result.prompt else '',
                    'processing_time_s': f"{result.processing_time_s:.2f}",
                    'error': result.error or ''
                }

                # Add baseline metrics
                if result.baseline_metrics:
                    for metric, value in result.baseline_metrics.scores.items():
                        row[f'baseline_{metric}'] = f"{value:.4f}"

                # Add regulated metrics
                if result.regulated_metrics:
                    for metric, value in result.regulated_metrics.scores.items():
                        row[f'regulated_{metric}'] = f"{value:.4f}"

                # Add improvements
                if result.baseline_metrics and result.regulated_metrics:
                    for metric in metric_columns:
                        baseline_val = result.baseline_metrics.scores.get(metric, 0)
                        regulated_val = result.regulated_metrics.scores.get(metric, 0)
                        improvement = regulated_val - baseline_val
                        row[f'{metric}_improvement'] = f"{improvement:.4f}"

                # Add mechanism firing info
                if result.regulated_result:
                    row['ecr_fired'] = result.regulated_result.ecr_fired
                    row['cp_type1_fired'] = result.regulated_result.cp_type1_fired
                    row['ifcs_fired'] = result.regulated_result.ifcs_fired
                    row['cp_type2_fired'] = result.regulated_result.cp_type2_fired

                writer.writerow(row)

        print(f"[Report] CSV saved to: {output_path}")

    @staticmethod
    def generate_summary_json(
        results: List[Any],  # List[BenchmarkEvaluationResult]
        aggregated_stats: Dict,
        output_path: str,
        config: Any = None  # BenchmarkConfig
    ):
        """Generate JSON summary with aggregated statistics

        Args:
            results: List of BenchmarkEvaluationResult
            aggregated_stats: Aggregated statistics dictionary
            output_path: Path to save JSON file
            config: Optional BenchmarkConfig
        """
        successful = sum(1 for r in results if not r.error)
        failed = sum(1 for r in results if r.error)

        # Collect errors
        errors = [
            {'example_id': r.example_id, 'error': r.error}
            for r in results if r.error
        ]

        # Build summary
        summary = {
            'metadata': {
                'benchmark_name': results[0].benchmark_name if results else 'unknown',
                'total_examples': len(results),
                'successful': successful,
                'failed': failed,
                'evaluation_date': datetime.now().isoformat()
            },
            'overall_metrics': BenchmarkReportGenerator._format_overall_metrics(aggregated_stats),
            'mechanism_analysis': aggregated_stats.get('mechanism_firing_rates', {}),
            'statistical_summary': {
                'baseline': aggregated_stats.get('baseline', {}),
                'regulated': aggregated_stats.get('regulated', {})
            },
            'processing_stats': {
                'total_time_s': sum(r.processing_time_s for r in results),
                'avg_time_per_example_s': sum(r.processing_time_s for r in results) / len(results) if results else 0,
                'median_time_s': sorted([r.processing_time_s for r in results])[len(results)//2] if results else 0
            },
            'errors': errors
        }

        # Add config info if available
        if config:
            summary['metadata']['config'] = {
                'batch_size': config.batch_size,
                'rate_limit_delay_s': config.rate_limit_delay_s,
                'max_retries': config.max_retries
            }

        # Write JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"[Report] JSON summary saved to: {output_path}")

    @staticmethod
    def generate_comparison_report(
        results: List[Any],  # List[BenchmarkEvaluationResult]
        output_path: str,
        max_examples: int = 20
    ):
        """Generate detailed baseline vs regulated comparison

        Args:
            results: List of BenchmarkEvaluationResult
            output_path: Path to save text file
            max_examples: Maximum number of examples to include
        """
        lines = []

        lines.append("="*80)
        lines.append("BENCHMARK EVALUATION: BASELINE VS REGULATED COMPARISON")
        lines.append("="*80)
        lines.append("")

        # Show first N examples
        for i, result in enumerate(results[:max_examples]):
            lines.append(f"\n{'='*80}")
            lines.append(f"EXAMPLE {i+1}: {result.example_id}")
            lines.append(f"{'='*80}")
            lines.append(f"\nPrompt: {result.prompt}")
            lines.append("")

            # Baseline
            lines.append("-" * 40)
            lines.append("BASELINE (Unregulated)")
            lines.append("-" * 40)
            if result.baseline_response:
                lines.append(result.baseline_response[:500])
                if len(result.baseline_response) > 500:
                    lines.append("... (truncated)")

            if result.baseline_metrics:
                lines.append("\nBaseline Metrics:")
                for metric, value in result.baseline_metrics.scores.items():
                    lines.append(f"  {metric}: {value:.4f}")
            lines.append("")

            # Regulated
            lines.append("-" * 40)
            lines.append("REGULATED (Trilogy)")
            lines.append("-" * 40)
            if result.regulated_result:
                lines.append(result.regulated_result.final_response[:500])
                if len(result.regulated_result.final_response) > 500:
                    lines.append("... (truncated)")

                lines.append("\nMechanisms Fired:")
                lines.append(f"  ECR: {result.regulated_result.ecr_fired}")
                lines.append(f"  CP Type-1: {result.regulated_result.cp_type1_fired}")
                lines.append(f"  IFCS: {result.regulated_result.ifcs_fired}")
                lines.append(f"  CP Type-2: {result.regulated_result.cp_type2_fired}")

            if result.regulated_metrics:
                lines.append("\nRegulated Metrics:")
                for metric, value in result.regulated_metrics.scores.items():
                    lines.append(f"  {metric}: {value:.4f}")
            lines.append("")

            # Comparison
            if result.comparison and 'metric_improvements' in result.comparison:
                lines.append("-" * 40)
                lines.append("IMPROVEMENT")
                lines.append("-" * 40)
                for metric, improvement in result.comparison['metric_improvements'].items():
                    sign = "+" if improvement >= 0 else ""
                    lines.append(f"  {metric}: {sign}{improvement:.4f}")
            lines.append("")

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"[Report] Comparison report saved to: {output_path}")

    @staticmethod
    def generate_html_visualization(
        results: List[Any],  # List[BenchmarkEvaluationResult]
        aggregated_stats: Dict,
        output_path: str
    ):
        """Generate HTML report with visualizations

        Args:
            results: List of BenchmarkEvaluationResult
            aggregated_stats: Aggregated statistics
            output_path: Path to save HTML file
        """
        # Extract overall metrics
        overall_metrics = BenchmarkReportGenerator._format_overall_metrics(aggregated_stats)

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Trilogy Benchmark Evaluation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-card {{
            display: inline-block;
            width: 200px;
            margin: 10px;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .improvement {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .degradation {{
            color: #f44336;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Trilogy Benchmark Evaluation Report</h1>

        <h2>Summary Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Baseline</th>
                <th>Regulated</th>
                <th>Improvement</th>
            </tr>
"""

        # Add metric rows
        if overall_metrics:
            baseline = overall_metrics.get('baseline', {})
            regulated = overall_metrics.get('regulated', {})
            improvements = overall_metrics.get('improvement', {})

            for metric in baseline.keys():
                baseline_val = baseline.get(metric, 0)
                regulated_val = regulated.get(metric, 0)
                improvement = improvements.get(metric, 0)

                improvement_class = 'improvement' if improvement >= 0 else 'degradation'
                sign = '+' if improvement >= 0 else ''

                html_content += f"""
            <tr>
                <td>{metric.replace('_', ' ').title()}</td>
                <td>{baseline_val:.4f}</td>
                <td>{regulated_val:.4f}</td>
                <td class="{improvement_class}">{sign}{improvement:.4f} ({sign}{improvement/baseline_val*100:.1f}%)</td>
            </tr>
"""

        html_content += """
        </table>

        <h2>Mechanism Firing Rates</h2>
        <table>
            <tr>
                <th>Mechanism</th>
                <th>Firing Rate</th>
            </tr>
"""

        # Add mechanism firing rates
        mechanism_rates = aggregated_stats.get('mechanism_firing_rates', {})
        for mechanism, rate in mechanism_rates.items():
            html_content += f"""
            <tr>
                <td>{mechanism.replace('_', ' ').title()}</td>
                <td>{rate:.1%}</td>
            </tr>
"""

        html_content += f"""
        </table>

        <h2>Processing Statistics</h2>
        <p>Total Examples: {len(results)}</p>
        <p>Successful: {sum(1 for r in results if not r.error)}</p>
        <p>Failed: {sum(1 for r in results if r.error)}</p>
        <p>Total Processing Time: {sum(r.processing_time_s for r in results):.1f}s</p>
        <p>Average Time per Example: {sum(r.processing_time_s for r in results)/len(results) if results else 0:.2f}s</p>

        <footer style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #888;">
            <p>Generated with Trilogy Benchmark Evaluation System</p>
            <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>
</body>
</html>
"""

        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"[Report] HTML visualization saved to: {output_path}")

    @staticmethod
    def _format_overall_metrics(aggregated_stats: Dict) -> Dict:
        """Format overall metrics for display

        Args:
            aggregated_stats: Aggregated statistics

        Returns:
            Formatted metrics dictionary
        """
        baseline_stats = aggregated_stats.get('baseline', {})
        regulated_stats = aggregated_stats.get('regulated', {})
        improvements = aggregated_stats.get('improvements', {})

        formatted = {
            'baseline': {},
            'regulated': {},
            'improvement': improvements
        }

        # Extract mean values
        for metric, stats in baseline_stats.items():
            if isinstance(stats, dict) and 'mean' in stats:
                formatted['baseline'][metric] = stats['mean']

        for metric, stats in regulated_stats.items():
            if isinstance(stats, dict) and 'mean' in stats:
                formatted['regulated'][metric] = stats['mean']

        return formatted
