"""
Domain Calibration Experiment

Tests domain-specific ECR weight configurations on different question categories.

This is Experiment 3 for the ECR empirical validation paper.
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_provider import LLMProviderFactory, LLMProvider
from ecr_engine import ECREngine
from benchmark_loader import BenchmarkLoader, BenchmarkExample
from benchmark_adapters import TruthfulQAAdapter
from trilogy_config import ECRConfig

from experiments.configs.experiment_config import (
    ExperimentConfig,
    DomainConfig,
    DOMAIN_CONFIGS
)
from experiments.analysis.statistical_tests import StatisticalAnalyzer


@dataclass
class DomainResult:
    """Result for a single domain."""
    domain: str
    default_accuracy: float
    calibrated_accuracy: float
    n_questions: int
    improvement: float
    default_per_question: List[bool]
    calibrated_per_question: List[bool]


class DomainCalibrationExperiment:
    """
    Domain-specific calibration experiment.

    Tests whether domain-specific weight configurations improve accuracy
    compared to default (equal) weights.
    """

    def __init__(
        self,
        config: ExperimentConfig = None,
        domain_configs: Dict[str, DomainConfig] = None
    ):
        """
        Initialize domain calibration experiment.

        Args:
            config: Experiment configuration
            domain_configs: Domain configurations (uses defaults if None)
        """
        self.config = config or ExperimentConfig()
        self.domain_configs = domain_configs or DOMAIN_CONFIGS

        # Initialize LLM provider
        print("[Domain] Initializing LLM provider from .env...")
        self.llm_provider = LLMProviderFactory.create_from_env()
        self.model_name = self.llm_provider.get_model_name()
        print(f"[Domain] Using model: {self.model_name}")

        # Initialize adapter
        self.adapter = TruthfulQAAdapter(prompt_strategy="choices_in_prompt")

        # Output directory
        # Sanitize model name for filesystem
        safe_model_name = self.model_name.replace(':', '_').replace('/', '_').replace('\\', '_')
        self.output_path = Path(self.config.output_dir) / "domain_calibration" / safe_model_name
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _llm_call_fn(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 2000
    ) -> str:
        """Wrapper for LLM calls."""
        return self.llm_provider.generate(
            prompt=prompt,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens
        )

    def _is_correct(self, response: str, example: BenchmarkExample) -> bool:
        """Check if response is correct."""
        selected_idx = self.adapter._parse_choice_selection(response)
        if selected_idx is None:
            choices = example.ground_truth['mc1_choices']
            selected_idx = self.adapter._compute_best_match(response, choices)

        labels = example.ground_truth['mc1_labels']
        if selected_idx < len(labels):
            return labels[selected_idx] == 1
        return False

    def _filter_examples_by_category(
        self,
        examples: List[BenchmarkExample],
        category_filter: Optional[str]
    ) -> List[BenchmarkExample]:
        """Filter examples by category."""
        if category_filter is None:
            return examples

        # TruthfulQA categories (case-insensitive matching)
        filtered = [
            ex for ex in examples
            if category_filter.lower() in ex.metadata.get('category', '').lower()
        ]

        return filtered

    def _evaluate_domain(
        self,
        domain_name: str,
        domain_config: DomainConfig,
        examples: List[BenchmarkExample]
    ) -> DomainResult:
        """Evaluate a single domain with default and calibrated weights."""

        # Create default ECR engine (equal weights)
        default_ecr_config = ECRConfig(
            K=self.config.k_candidates,
            H=self.config.ecr_h,
            tau_CCI=self.config.ecr_tau_cci,
            alpha=0.20, beta=0.20, gamma=0.20, delta=0.20, epsilon=0.20
        )
        default_engine = ECREngine(default_ecr_config)

        # Create calibrated ECR engine (domain-specific weights)
        calibrated_ecr_config = ECRConfig(
            K=self.config.k_candidates,
            H=self.config.ecr_h,
            tau_CCI=self.config.ecr_tau_cci,
            alpha=domain_config.alpha,
            beta=domain_config.beta,
            gamma=domain_config.gamma,
            delta=domain_config.delta,
            epsilon=domain_config.epsilon
        )
        calibrated_engine = ECREngine(calibrated_ecr_config)

        default_correct = []
        calibrated_correct = []

        for example in examples:
            prompt, _ = self.adapter.prepare_prompt(example)

            # Generate candidates once
            candidates = default_engine.generate_candidates(
                prompt=prompt,
                llm_call_fn=self._llm_call_fn,
                num_candidates=self.config.k_candidates,
                llm_provider=self.llm_provider
            )

            # Evaluate with default weights
            default_selected, _, _ = default_engine.select_best_candidate(
                candidates=candidates,
                prompt=prompt,
                llm_call_fn=self._llm_call_fn
            )
            default_is_correct = self._is_correct(default_selected, example)
            default_correct.append(default_is_correct)

            # Evaluate with calibrated weights
            calibrated_selected, _, _ = calibrated_engine.select_best_candidate(
                candidates=candidates,
                prompt=prompt,
                llm_call_fn=self._llm_call_fn
            )
            calibrated_is_correct = self._is_correct(calibrated_selected, example)
            calibrated_correct.append(calibrated_is_correct)

        default_accuracy = sum(default_correct) / len(examples) if examples else 0
        calibrated_accuracy = sum(calibrated_correct) / len(examples) if examples else 0
        improvement = calibrated_accuracy - default_accuracy

        return DomainResult(
            domain=domain_name,
            default_accuracy=default_accuracy,
            calibrated_accuracy=calibrated_accuracy,
            n_questions=len(examples),
            improvement=improvement,
            default_per_question=default_correct,
            calibrated_per_question=calibrated_correct
        )

    def run(
        self,
        domains_to_run: Optional[List[str]] = None
    ) -> Dict:
        """
        Run domain calibration experiment.

        Args:
            domains_to_run: List of domain names to run (None = all)

        Returns:
            Summary dictionary
        """
        # Select domains to run
        if domains_to_run:
            domains = {k: v for k, v in self.domain_configs.items() if k in domains_to_run}
        else:
            domains = self.domain_configs

        print(f"[Domain] Running {len(domains)} domains")

        # Load TruthfulQA dataset
        print("[Domain] Loading TruthfulQA dataset...")
        loader = BenchmarkLoader()
        all_examples = loader.load_truthfulqa()

        results: Dict[str, DomainResult] = {}

        for domain_name, domain_config in tqdm(domains.items(), desc="Domains"):
            print(f"\n[Domain] Evaluating domain: {domain_name}")
            print(f"  Category filter: {domain_config.category_filter or 'None (all)'}")
            print(f"  Weights: alpha={domain_config.alpha:.2f}, beta={domain_config.beta:.2f}, "
                  f"gamma={domain_config.gamma:.2f}, delta={domain_config.delta:.2f}, "
                  f"epsilon={domain_config.epsilon:.2f}")

            # Filter examples by category
            filtered = self._filter_examples_by_category(
                all_examples, domain_config.category_filter
            )

            if len(filtered) == 0:
                print(f"  WARNING: No examples found for category filter '{domain_config.category_filter}'")
                # Fall back to random sample
                filtered = all_examples

            # Sample questions
            random.seed(self.config.seed)
            n_questions = min(domain_config.n_questions, len(filtered))
            examples = random.sample(filtered, n_questions)

            print(f"  Using {len(examples)} questions")

            # Evaluate domain
            result = self._evaluate_domain(domain_name, domain_config, examples)
            results[domain_name] = result

            print(f"  Default accuracy:    {result.default_accuracy:.1%}")
            print(f"  Calibrated accuracy: {result.calibrated_accuracy:.1%}")
            print(f"  Improvement:         {result.improvement:+.1%}")

        # Save results
        summary = self._save_results(results, domains)

        # Print summary
        print("\n[Domain] Results Summary:")
        print("-" * 70)
        print(f"{'Domain':<15} {'Default':>12} {'Calibrated':>12} {'Improvement':>12} {'N':>6}")
        print("-" * 70)
        for name, result in results.items():
            print(f"{name:<15} {result.default_accuracy:>12.1%} "
                  f"{result.calibrated_accuracy:>12.1%} "
                  f"{result.improvement:>+12.1%} {result.n_questions:>6}")
        print("-" * 70)

        # Overall improvement
        total_default = sum(sum(r.default_per_question) for r in results.values())
        total_calibrated = sum(sum(r.calibrated_per_question) for r in results.values())
        total_n = sum(r.n_questions for r in results.values())

        if total_n > 0:
            overall_default = total_default / total_n
            overall_calibrated = total_calibrated / total_n
            overall_improvement = overall_calibrated - overall_default
            print(f"{'OVERALL':<15} {overall_default:>12.1%} "
                  f"{overall_calibrated:>12.1%} {overall_improvement:>+12.1%} {total_n:>6}")

        return summary

    def _save_results(
        self,
        results: Dict[str, DomainResult],
        configs: Dict[str, DomainConfig]
    ) -> Dict:
        """Save results to JSON."""
        summary = {
            'experiment': 'domain_calibration',
            'model': self.model_name,
            'seed': self.config.seed,
            'domains': {},
            'statistics': {}
        }

        for name, result in results.items():
            summary['domains'][name] = {
                'description': configs[name].description,
                'category_filter': configs[name].category_filter,
                'weights': {
                    'alpha': configs[name].alpha,
                    'beta': configs[name].beta,
                    'gamma': configs[name].gamma,
                    'delta': configs[name].delta,
                    'epsilon': configs[name].epsilon
                },
                'default_accuracy': result.default_accuracy,
                'calibrated_accuracy': result.calibrated_accuracy,
                'improvement': result.improvement,
                'n_questions': result.n_questions
            }

            # Statistical test for improvement
            default_scores = [1 if c else 0 for c in result.default_per_question]
            calibrated_scores = [1 if c else 0 for c in result.calibrated_per_question]

            if len(default_scores) > 1:
                summary['statistics'][name] = {
                    't_test': StatisticalAnalyzer.paired_t_test(
                        calibrated_scores, default_scores
                    ),
                    'mcnemar': StatisticalAnalyzer.mcnemar_test(
                        result.calibrated_per_question,
                        result.default_per_question
                    )
                }

        # Save to file
        results_path = self.output_path / "results.json"
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"[Domain] Results saved to {results_path}")

        return summary


def run_domain_calibration(
    n_per_domain: int = 10,
    k_candidates: int = 5,
    seed: int = 42,
    domains: Optional[List[str]] = None
) -> Dict:
    """
    Convenience function to run domain calibration.

    Args:
        n_per_domain: Number of questions per domain
        k_candidates: Number of candidates per question
        seed: Random seed
        domains: List of domains to run (None = all)

    Returns:
        Summary dictionary
    """
    config = ExperimentConfig(
        k_candidates=k_candidates,
        seed=seed
    )

    # Update domain configs with n_per_domain
    domain_configs = {}
    for name, cfg in DOMAIN_CONFIGS.items():
        domain_configs[name] = DomainConfig(
            name=cfg.name,
            description=cfg.description,
            category_filter=cfg.category_filter,
            alpha=cfg.alpha,
            beta=cfg.beta,
            gamma=cfg.gamma,
            delta=cfg.delta,
            epsilon=cfg.epsilon,
            n_questions=n_per_domain
        )

    experiment = DomainCalibrationExperiment(config, domain_configs)
    return experiment.run(domains_to_run=domains)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ECR domain calibration experiment")
    parser.add_argument("--n-per-domain", type=int, default=10,
                        help="Questions per domain")
    parser.add_argument("--k-candidates", type=int, default=5,
                        help="Candidates per question")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--domains", nargs="*",
                        help="Specific domains to run (default: all)")

    args = parser.parse_args()

    run_domain_calibration(
        n_per_domain=args.n_per_domain,
        k_candidates=args.k_candidates,
        seed=args.seed,
        domains=args.domains
    )
