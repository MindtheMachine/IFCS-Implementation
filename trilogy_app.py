"""
Main Trilogy Application
Supports multiple LLM providers (Anthropic, OpenAI, HuggingFace, Ollama)
"""

import os
import time
from typing import Dict, Tuple, List, Optional

from trilogy_config import TrilogyConfig
from trilogy_orchestrator import (
    TrilogyOrchestrator,
    BaselineAgent,
    ComparisonEngine,
)
from llm_provider import LLMProviderFactory

# Benchmark evaluation imports (optional)
try:
    from benchmark_loader import BenchmarkLoader
    from benchmark_adapters import TruthfulQAAdapter, ASQAAdapter
    from benchmark_metrics import (
        TruthfulQAMetrics,
        ASQAMetrics,
        BenchmarkMetricsAggregator,
    )
    from benchmark_orchestrator import BenchmarkOrchestrator
    from benchmark_reports import BenchmarkReportGenerator
    from benchmark_config import (
        BenchmarkConfig,
        DEFAULT_TRUTHFULQA_CONFIG,
        DEFAULT_ASQA_CONFIG,
    )
    _BENCHMARKS_AVAILABLE = True
except ImportError:
    BenchmarkLoader = None
    TruthfulQAAdapter = None
    ASQAAdapter = None
    TruthfulQAMetrics = None
    ASQAMetrics = None
    BenchmarkMetricsAggregator = None
    BenchmarkOrchestrator = None
    BenchmarkReportGenerator = None
    BenchmarkConfig = None
    DEFAULT_TRUTHFULQA_CONFIG = None
    DEFAULT_ASQA_CONFIG = None
    _BENCHMARKS_AVAILABLE = False


class TrilogyApp:
    """Main application for ECR-Control Probe-IFCS system"""
    
    def __init__(self, api_key: str = None, config: TrilogyConfig = None):
        """Initialize trilogy application

        Args:
            api_key: LLM API key (or uses environment variable)
            config: TrilogyConfig instance (or uses defaults)
        """
        # Ensure .env is loaded before reading any env-dependent config
        LLMProviderFactory._reload_env()

        # Initialize config
        if config is None:
            config = TrilogyConfig(api_key=api_key)

        self.config = config

        # Initialize LLM provider (auto-detects from .env)
        self.llm_provider = LLMProviderFactory.create_from_env()

        # Initialize agents
        self.trilogy = TrilogyOrchestrator(
            config, self.call_llm, self.llm_provider
        )
        self.baseline = BaselineAgent(self.call_llm)

        provider_name = os.getenv("LLM_PROVIDER", "anthropic")
        print(
            f"[App] Initialized with provider: {provider_name}, "
            f"model: {config.model}"
        )
    
    def call_llm(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Call LLM API (works with any configured provider)

        Args:
            prompt: Prompt to send
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Response text
        """
        if max_tokens is None:
            max_tokens = self.config.max_tokens
        if temperature is None:
            temperature = self.config.temperature
        if top_p is None:
            top_p = self.config.top_p

        try:
            # Use the provider abstraction
            response = self.llm_provider.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                seed=self.config.seed,
            )
            return response
        except (RuntimeError, TimeoutError, OSError, ValueError) as e:
            print(f"[Error] API call failed: {e}")
            return f"Error calling LLM: {str(e)}"
    
    def process_single(self, prompt: str, context: str = "") -> Tuple[
        str, str, Dict
    ]:
        """Process single prompt through both baseline and trilogy
        
        Args:
            prompt: User prompt
            context: Optional context
            
        Returns:
            (baseline_response, regulated_response, comparison)
        """
        print("\n" + "*" * 40)
        print("PROCESSING QUERY")
        print("*" * 40)
        
        # Run baseline
        print("\n" + "-" * 80)
        print("BASELINE AGENT (Unregulated LLM)")
        print("-" * 80)
        baseline_start = time.time()
        baseline_response = self.baseline.process(prompt)
        baseline_time = time.time() - baseline_start
        print(f"[Baseline] Completed in {baseline_time:.2f}s")
        
        # Run trilogy
        print("\n" + "-" * 80)
        print("TRILOGY AGENT (ECR-Control Probe-IFCS)")
        print("-" * 80)
        trilogy_start = time.time()
        regulated_result = self.trilogy.process(prompt, context)
        trilogy_time = time.time() - trilogy_start
        print(f"[Trilogy] Completed in {trilogy_time:.2f}s")
        
        # Compare
        comparison = ComparisonEngine.compare(
            prompt,
            baseline_response,
            regulated_result,
        )
        
        comparison['baseline_time_s'] = baseline_time
        comparison['trilogy_time_s'] = trilogy_time
        comparison['delta_time_s'] = trilogy_time - baseline_time
        
        return baseline_response, regulated_result.final_response, comparison
    
    def save_outputs(
        self,
        prompt: str,
        baseline_response: str,
        regulated_response: str,
        comparison: Dict,
    ):
        """Save outputs to files
        
        Args:
            prompt: Original prompt
            baseline_response: Baseline output
            regulated_response: Regulated output
            comparison: Comparison analysis
        """
        # Save baseline
        baseline_path = self.config.baseline_output_path
        with open(baseline_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BASELINE OUTPUT (Unregulated LLM)\n")
            f.write("="*80 + "\n\n")
            f.write(f"PROMPT:\n{prompt}\n\n")
            f.write("-"*80 + "\n\n")
            f.write(f"RESPONSE:\n{baseline_response}\n")
        
        print(
            f"[App] Saved baseline output to: {self.config.baseline_output_path}"
        )
        
        # Save regulated
        regulated_path = self.config.regulated_output_path
        with open(regulated_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("REGULATED OUTPUT (ECR-Control Probe-IFCS)\n")
            f.write("="*80 + "\n\n")
            f.write(f"PROMPT:\n{prompt}\n\n")
            f.write("-"*80 + "\n\n")
            f.write(f"RESPONSE:\n{regulated_response}\n")
        
        print(
            f"[App] Saved regulated output to: {self.config.regulated_output_path}"
        )
        
        # Save comparison
        with open(self.config.comparison_output_path, 'w', encoding='utf-8') as f:
            # Side-by-side comparison
            side_by_side = ComparisonEngine.format_side_by_side(
                prompt,
                baseline_response,
                regulated_response,
                comparison,
                width=60,
            )
            f.write(side_by_side)
            f.write("\n\n")
            
            # Detailed metrics
            f.write("="*80 + "\n")
            f.write("DETAILED METRICS\n")
            f.write("="*80 + "\n\n")
            
            import json
            f.write(json.dumps(comparison, indent=2, default=str))
        
        print(f"[App] Saved comparison analysis to: {self.config.comparison_output_path}")
    
    def process_test_case(self, test_case: Dict) -> Dict:
        """Process a test case from the taxonomy
        
        Args:
            test_case: Test case dictionary with 'id', 'category', 'prompt', etc.
            
        Returns:
            Results dictionary
        """
        print("\n" + "*" * 40)
        print(f"TEST CASE: {test_case['id']} - {test_case['category']}")
        print("*" * 40)
        
        if test_case.get('multi_turn') and test_case.get('turns'):
            return self.process_multi_turn_case(test_case)
        
        self.trilogy.reset_interaction()
        
        prompt = test_case['prompt']
        baseline_response, regulated_response, comparison = self.process_single(prompt)
        
        # Check if expected mechanism fired
        expected = test_case.get('expected_mechanism_impl', test_case.get('expected_mechanism'))
        expected_paper = test_case.get('expected_mechanism_paper', test_case.get('expected_mechanism'))
        mechanisms_fired = comparison['mechanisms_fired']
        
        # Map expected to actual
        mechanism_map = {
            'ECR': mechanisms_fired['ECR'],
            'CP-Type-1': mechanisms_fired['CP_Type1'],
            'IFCS': mechanisms_fired['IFCS'],
            'CP-Type-2': mechanisms_fired['CP_Type2']
        }
        
        result = {
            'test_id': test_case['id'],
            'category': test_case['category'],
            'prompt': prompt,
            'expected_mechanism': expected,
            'expected_mechanism_paper': expected_paper,
            'mechanisms_fired': mechanisms_fired,
            'expected_responsible': expected,
            'expected_to_fire': mechanism_map.get(expected, False) if expected else None,
            'expected_fired': mechanism_map.get(expected, False) if expected else None,
            'baseline_time_s': comparison.get('baseline_time_s'),
            'regulated_time_s': comparison.get('trilogy_time_s'),
            'delta_time_s': comparison.get('delta_time_s'),
            'baseline_response': baseline_response,
            'regulated_response': regulated_response,
            'comparison': comparison
        }
        
        return result

    def process_multi_turn_case(self, test_case: Dict) -> Dict:
        """Process a multi-turn test case from the taxonomy.
        
        Args:
            test_case: Test case dictionary with 'turns' array
        
        Returns:
            Results dictionary with per-turn outputs
        """
        self.trilogy.reset_interaction()
        
        turns = test_case.get('turns', [])
        per_turn_results = []
        
        mechanisms_agg = {
            'ECR': False,
            'CP_Type1': False,
            'IFCS': False,
            'CP_Type2': False
        }
        
        baseline_time_total = 0.0
        regulated_time_total = 0.0
        
        for idx, prompt in enumerate(turns, 1):
            print(f"\n[Turn {idx}/{len(turns)}] {prompt[:80]}...")
            baseline_response, regulated_response, comparison = self.process_single(prompt)
            
            per_turn_results.append({
                'turn_index': idx,
                'prompt': prompt,
                'baseline_response': baseline_response,
                'regulated_response': regulated_response,
                'comparison': comparison
            })
            
            baseline_time_total += comparison.get('baseline_time_s') or 0.0
            regulated_time_total += comparison.get('trilogy_time_s') or 0.0
            
            mech = comparison['mechanisms_fired']
            mechanisms_agg['ECR'] = mechanisms_agg['ECR'] or mech.get('ECR', False)
            mechanisms_agg['CP_Type1'] = mechanisms_agg['CP_Type1'] or mech.get('CP_Type1', False)
            mechanisms_agg['IFCS'] = mechanisms_agg['IFCS'] or mech.get('IFCS', False)
            mechanisms_agg['CP_Type2'] = mechanisms_agg['CP_Type2'] or mech.get('CP_Type2', False)
        
        expected = test_case.get('expected_mechanism_impl', test_case.get('expected_mechanism'))
        expected_paper = test_case.get('expected_mechanism_paper', test_case.get('expected_mechanism'))
        mechanism_map = {
            'ECR': mechanisms_agg['ECR'],
            'CP-Type-1': mechanisms_agg['CP_Type1'],
            'IFCS': mechanisms_agg['IFCS'],
            'CP-Type-2': mechanisms_agg['CP_Type2']
        }
        
        return {
            'test_id': test_case['id'],
            'category': test_case['category'],
            'prompt': test_case.get('prompt', ''),
            'expected_mechanism': expected,
            'expected_mechanism_paper': expected_paper,
            'mechanisms_fired': mechanisms_agg,
            'expected_responsible': expected,
            'expected_to_fire': mechanism_map.get(expected, False) if expected else None,
            'expected_fired': mechanism_map.get(expected, False) if expected else None,
            'baseline_time_s': baseline_time_total,
            'regulated_time_s': regulated_time_total,
            'delta_time_s': regulated_time_total - baseline_time_total,
            'multi_turn': True,
            'turns': per_turn_results
        }
    
    def run_test_suite(self, test_cases: list) -> list:
        """Run full test suite
        
        Args:
            test_cases: List of test case dictionaries
            
        Returns:
            List of results
        """
        print("\n" + "*" * 40)
        print(f"RUNNING TEST SUITE ({len(test_cases)} tests)")
        print("*" * 40)
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] Processing test case {test_case['id']}...")
            
            result = self.process_test_case(test_case)
            results.append(result)
            
            # Brief pause to avoid rate limits
            time.sleep(1)
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUITE SUMMARY")
        print("="*80)
        
        for result in results:
            status = "OK" if result['expected_fired'] else "--"
            print(f"{status} {result['test_id']:6s} {result['category']:40s} "
                  f"Expected: {result['expected_mechanism']}")

        return results

    def process_benchmark(
        self,
        benchmark_name: str,
        config: BenchmarkConfig = None,
    ) -> Tuple[List, Dict]:
        """Process benchmark evaluation

        Args:
            benchmark_name: 'truthfulqa' or 'asqa'
            config: Optional benchmark configuration

        Returns:
            (results, aggregated_stats)
        """
        if not _BENCHMARKS_AVAILABLE:
            raise RuntimeError(
                "Benchmark modules are not available. The benchmark subsystem was removed in cleanup. "
                "Use the core pipeline or reintroduce benchmark modules if needed."
            )
        print("\n" + "*" * 40)
        print(f"BENCHMARK EVALUATION: {benchmark_name.upper()}")
        print("*" * 40)

        # Use default config if none provided
        if config is None:
            if benchmark_name == 'truthfulqa':
                config = DEFAULT_TRUTHFULQA_CONFIG
            elif benchmark_name == 'asqa':
                config = DEFAULT_ASQA_CONFIG
            else:
                raise ValueError(f"Unknown benchmark: {benchmark_name}")

        # 1. Load dataset
        print(f"\n[1/5] Loading {benchmark_name} dataset...")
        loader = BenchmarkLoader(cache_dir=config.cache_dir)

        if benchmark_name == 'truthfulqa':
            examples = loader.load_truthfulqa(split=config.split)
            adapter = TruthfulQAAdapter(prompt_strategy=config.truthfulqa_prompt_strategy)
            metrics_computer = TruthfulQAMetrics()
        elif benchmark_name == 'asqa':
            examples = loader.load_asqa(split=config.split)
            adapter = ASQAAdapter()
            metrics_computer = ASQAMetrics()
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        # 2. Apply batch configuration
        if config.batch_size > 0:
            examples = examples[config.batch_start_idx:config.batch_start_idx + config.batch_size]
            print(
                f"[Info] Processing subset: {len(examples)} examples "
                f"(indices {config.batch_start_idx} to {config.batch_start_idx + len(examples)})"
            )
        else:
            print(f"[Info] Processing full dataset: {len(examples)} examples")

        # 3. Run evaluation
        print("\n[2/5] Running batch evaluation...")
        orchestrator = BenchmarkOrchestrator(self, adapter, metrics_computer, config)
        results = orchestrator.evaluate_batch(examples)

        # 4. Aggregate statistics
        print("\n[3/5] Aggregating statistics...")
        aggregated = BenchmarkMetricsAggregator.aggregate_scores(results)

        # 5. Generate reports
        print("\n[4/5] Generating reports...")
        BenchmarkReportGenerator.generate_csv_report(results, config.results_csv_path)
        BenchmarkReportGenerator.generate_summary_json(results, aggregated, config.summary_json_path, config)

        if config.comparison_path:
            BenchmarkReportGenerator.generate_comparison_report(results, config.comparison_path)

        if config.html_report_path:
            try:
                BenchmarkReportGenerator.generate_html_visualization(
                    results, aggregated, config.html_report_path
                )
            except (OSError, ValueError, RuntimeError) as e:
                print(f"[Warning] Could not generate HTML report: {e}")

        # 6. Print summary
        print("\n[5/5] Evaluation complete!")
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Benchmark: {benchmark_name}")
        print(f"Examples evaluated: {len(results)}")
        print(f"Successful: {sum(1 for r in results if not r.error)}")
        print(f"Failed: {sum(1 for r in results if r.error)}")

        # Print overall metrics if available
        if aggregated and 'baseline' in aggregated and 'regulated' in aggregated:
            print("\nOverall Metrics:")
            baseline_stats = aggregated['baseline']
            regulated_stats = aggregated['regulated']
            improvements = aggregated.get('improvements', {})

            for metric in baseline_stats.keys():
                baseline_val = (
                    baseline_stats[metric].get('mean', 0)
                    if isinstance(baseline_stats[metric], dict)
                    else baseline_stats[metric]
                )
                regulated_val = (
                    regulated_stats[metric].get('mean', 0)
                    if isinstance(regulated_stats[metric], dict)
                    else regulated_stats[metric]
                )
                improvement = improvements.get(metric, 0)

                sign = "+" if improvement >= 0 else ""
                print(
                    f"  {metric}: {baseline_val:.4f} - {regulated_val:.4f} ("
                    f"{sign}{improvement:.4f})"
                )

        print("\nResults saved to:")
        print(f"  CSV: {config.results_csv_path}")
        print(f"  JSON: {config.summary_json_path}")
        if config.comparison_path:
            print(f"  Comparison: {config.comparison_path}")
        if config.html_report_path:
            print(f"  HTML: {config.html_report_path}")

        return results, aggregated


def main():
    """Main entry point for command-line usage"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description='ECR-Control Probe-IFCS Trilogy System'
    )
    parser.add_argument('--prompt', type=str, help='Single prompt to process')
    parser.add_argument('--test-suite', action='store_true', help='Run full test suite')
    parser.add_argument(
        '--test-ids', type=str, help='Comma-separated test case IDs to run (e.g., 2.2,2.3)'
    )
    parser.add_argument('--api-key', type=str, help='LLM API key (or use LLM_API_KEY env var)')

    # Benchmark evaluation arguments
    parser.add_argument('--benchmark', choices=['truthfulqa', 'asqa'],
                       help='Run benchmark evaluation')
    parser.add_argument('--batch-size', type=int, default=0,
                       help='Number of examples to evaluate (0=all)')
    parser.add_argument('--batch-start', type=int, default=0,
                       help='Starting index for batch processing')
    parser.add_argument('--rate-limit', type=float, default=1.0,
                       help='Delay between API calls (seconds)')

    args = parser.parse_args()
    
    # Initialize app
    app = TrilogyApp(api_key=args.api_key)

    if args.benchmark:
        # Run benchmark evaluation
        config = BenchmarkConfig(
            benchmark_name=args.benchmark,
            batch_size=args.batch_size,
            batch_start_idx=args.batch_start,
            rate_limit_delay_s=args.rate_limit,
        )

        results, _ = app.process_benchmark(args.benchmark, config)

        print("\n" + "="*80)
        print("BENCHMARK EVALUATION COMPLETE")
        print("="*80)
        print(f"Benchmark: {args.benchmark}")
        print(f"Examples evaluated: {len(results)}")
        print(f"Results saved to: {config.results_csv_path}")
        print(f"Summary saved to: {config.summary_json_path}")

    elif args.test_ids:
        # Run specific test cases
        from trilogy_config import TEST_CASES_36_TAXONOMY
        
        requested = [t.strip() for t in args.test_ids.split(',') if t.strip()]
        test_cases = [tc for tc in TEST_CASES_36_TAXONOMY if tc['id'] in requested]
        
        if not test_cases:
            print(f"No matching test cases found for IDs: {args.test_ids}")
            return
        
        results = app.run_test_suite(test_cases)
        
        # Save results
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\nResults saved to: test_results.json")
    
    elif args.test_suite:
        # Run test suite
        from trilogy_config import TEST_CASES_36_TAXONOMY
        
        # Run subset (first 10 for demo)
        test_cases = TEST_CASES_36_TAXONOMY[:10]
        results = app.run_test_suite(test_cases)
        
        # Save results
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\nResults saved to: test_results.json")
    
    elif args.prompt:
        # Process single prompt
        baseline, regulated, comparison = app.process_single(args.prompt)
        
        # Save outputs
        app.save_outputs(args.prompt, baseline, regulated, comparison)
        
        print("\n" + "="*80)
        print("OUTPUTS SAVED")
        print("="*80)
        print(f"Baseline:   {app.config.baseline_output_path}")
        print(f"Regulated:  {app.config.regulated_output_path}")
        print(f"Comparison: {app.config.comparison_output_path}")
    
    else:
        print("Usage:")
        print("  --prompt 'Your question here'  : Process single prompt")
        print("  --test-suite                   : Run test suite")
        print("  --benchmark truthfulqa|asqa    : Run benchmark evaluation")
        print("\nExamples:")
        print("  python trilogy_app.py --prompt 'What is the best programming language-'")
        print("  python trilogy_app.py --benchmark truthfulqa --batch-size 100")
        print("  python trilogy_app.py --benchmark asqa --batch-size 50 --rate-limit 2.0")


if __name__ == '__main__':
    main()
