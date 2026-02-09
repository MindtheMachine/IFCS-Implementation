"""
Ablation Study Experiment

Tests individual metric contributions by varying ECR weight configurations.

This is Experiment 2 for the ECR empirical validation paper.
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
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
    AblationConfig,
    ABLATION_CONFIGS
)
from experiments.analysis.statistical_tests import StatisticalAnalyzer


@dataclass
class AblationResult:
    """Result for a single ablation configuration."""
    config_name: str
    accuracy: float
    n_correct: int
    n_total: int
    mean_cci: float
    per_question: List[bool]
    delta_vs_full: Optional[float] = None


class AblationStudy:
    """
    Ablation study for ECR metric contributions.

    Tests how removing or isolating individual metrics affects performance.
    Uses pre-generated candidates for fair comparison across configurations.
    """

    def __init__(
        self,
        config: ExperimentConfig = None,
        ablation_configs: Dict[str, AblationConfig] = None
    ):
        """
        Initialize ablation study.

        Args:
            config: Experiment configuration
            ablation_configs: Ablation configurations to test (uses defaults if None)
        """
        self.config = config or ExperimentConfig()
        self.ablation_configs = ablation_configs or ABLATION_CONFIGS

        # Initialize LLM provider
        print("[Ablation] Initializing LLM provider from .env...")
        self.llm_provider = LLMProviderFactory.create_from_env()
        self.model_name = self.llm_provider.get_model_name()
        print(f"[Ablation] Using model: {self.model_name}")

        # Initialize adapter
        self.adapter = TruthfulQAAdapter(prompt_strategy="choices_in_prompt")

        # Output directory
        # Sanitize model name for filesystem
        safe_model_name = self.model_name.replace(':', '_').replace('/', '_').replace('\\', '_')
        self.output_path = Path(self.config.output_dir) / "ablation_study" / safe_model_name
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

    def _generate_candidates_for_question(
        self,
        example: BenchmarkExample,
        ecr_engine: ECREngine
    ) -> List[str]:
        """Generate candidates for a question."""
        prompt, _ = self.adapter.prepare_prompt(example)
        return ecr_engine.generate_candidates(
            prompt=prompt,
            llm_call_fn=self._llm_call_fn,
            num_candidates=self.config.k_candidates,
            llm_provider=self.llm_provider
        )

    def _evaluate_config(
        self,
        config_name: str,
        ablation_config: AblationConfig,
        examples: List[BenchmarkExample],
        pregenerated_candidates: Dict[int, List[str]]
    ) -> AblationResult:
        """Evaluate a single ablation configuration."""

        # Create ECR engine with ablation weights
        ecr_config = ECRConfig(
            K=self.config.k_candidates,
            H=self.config.ecr_h,
            tau_CCI=self.config.ecr_tau_cci,
            **ablation_config.to_dict()
        )
        ecr_engine = ECREngine(ecr_config)

        correct_count = 0
        per_question = []
        cci_scores = []

        for i, example in enumerate(examples):
            prompt, _ = self.adapter.prepare_prompt(example)
            candidates = pregenerated_candidates[i]

            # Select using this configuration's weights
            selected, metrics, _ = ecr_engine.select_best_candidate(
                candidates=candidates,
                prompt=prompt,
                llm_call_fn=self._llm_call_fn
            )

            is_correct = self._is_correct(selected, example)
            per_question.append(is_correct)
            cci_scores.append(metrics.CCI)

            if is_correct:
                correct_count += 1

        accuracy = correct_count / len(examples) if examples else 0
        mean_cci = sum(cci_scores) / len(cci_scores) if cci_scores else 0

        return AblationResult(
            config_name=config_name,
            accuracy=accuracy,
            n_correct=correct_count,
            n_total=len(examples),
            mean_cci=mean_cci,
            per_question=per_question
        )

    def run(
        self,
        n_questions: Optional[int] = None,
        configs_to_run: Optional[List[str]] = None
    ) -> Dict:
        """
        Run ablation study.

        Args:
            n_questions: Number of questions (None = use config)
            configs_to_run: List of config names to run (None = all)

        Returns:
            Summary dictionary with all results
        """
        n_questions = n_questions or self.config.n_questions

        # Select configurations to run
        if configs_to_run:
            configs = {k: v for k, v in self.ablation_configs.items() if k in configs_to_run}
        else:
            configs = self.ablation_configs

        print(f"[Ablation] Running {len(configs)} configurations")

        # Load TruthfulQA dataset
        print("[Ablation] Loading TruthfulQA dataset...")
        loader = BenchmarkLoader()
        all_examples = loader.load_truthfulqa()

        # Sample questions
        random.seed(self.config.seed)
        if n_questions < len(all_examples):
            examples = random.sample(all_examples, n_questions)
        else:
            examples = all_examples[:n_questions]

        print(f"[Ablation] Using {len(examples)} questions")

        # Pre-generate candidates once (shared across all configurations)
        print(f"[Ablation] Pre-generating candidates for all questions...")
        base_ecr_config = ECRConfig(K=self.config.k_candidates, H=self.config.ecr_h)
        base_ecr = ECREngine(base_ecr_config)

        pregenerated = {}
        for i, example in enumerate(tqdm(examples, desc="Generating candidates")):
            pregenerated[i] = self._generate_candidates_for_question(example, base_ecr)

        # Run each configuration
        results: Dict[str, AblationResult] = {}
        for config_name, ablation_config in tqdm(configs.items(), desc="Configurations"):
            print(f"\n[Ablation] Running configuration: {config_name}")
            print(f"  Weights: alpha={ablation_config.alpha:.2f}, beta={ablation_config.beta:.2f}, "
                  f"gamma={ablation_config.gamma:.2f}, delta={ablation_config.delta:.2f}, "
                  f"epsilon={ablation_config.epsilon:.2f}")

            result = self._evaluate_config(
                config_name=config_name,
                ablation_config=ablation_config,
                examples=examples,
                pregenerated_candidates=pregenerated
            )
            results[config_name] = result

            print(f"  Accuracy: {result.accuracy:.1%} ({result.n_correct}/{result.n_total})")

        # Compute delta vs full configuration
        if 'full' in results:
            full_accuracy = results['full'].accuracy
            for name, result in results.items():
                result.delta_vs_full = result.accuracy - full_accuracy

        # Save results
        summary = self._save_results(results, configs)

        # Print summary
        print("\n[Ablation] Results Summary:")
        print("-" * 60)
        print(f"{'Configuration':<20} {'Accuracy':>10} {'Delta':>12} {'Mean CCI':>10}")
        print("-" * 60)
        for name, result in sorted(results.items(), key=lambda x: -x[1].accuracy):
            delta_str = f"{result.delta_vs_full:+.1%}" if result.delta_vs_full is not None else "N/A"
            print(f"{name:<20} {result.accuracy:>10.1%} {delta_str:>12} {result.mean_cci:>10.3f}")
        print("-" * 60)

        return summary

    def _save_results(
        self,
        results: Dict[str, AblationResult],
        configs: Dict[str, AblationConfig]
    ) -> Dict:
        """Save results to JSON."""
        summary = {
            'experiment': 'ablation_study',
            'model': self.model_name,
            'n_questions': self.config.n_questions,
            'k_candidates': self.config.k_candidates,
            'seed': self.config.seed,
            'configurations': {},
            'ranking': []
        }

        for name, result in results.items():
            summary['configurations'][name] = {
                'description': configs[name].description,
                'weights': configs[name].to_dict(),
                'accuracy': result.accuracy,
                'n_correct': result.n_correct,
                'n_total': result.n_total,
                'mean_cci': result.mean_cci,
                'delta_vs_full': result.delta_vs_full
            }

        # Create ranking
        summary['ranking'] = sorted(
            results.keys(),
            key=lambda x: -results[x].accuracy
        )

        # Compute statistics
        per_question_scores = {
            name: [int(1 if c else 0) for c in result.per_question]
            for name, result in results.items()
        }

        if len(per_question_scores) > 1:
            # ANOVA across configurations
            groups = list(per_question_scores.values())
            anova_result = StatisticalAnalyzer.one_way_anova(*groups)
            # Convert numpy types to native Python
            summary['statistics'] = {
                'anova': {k: float(v) if hasattr(v, 'item') else v for k, v in anova_result.items()}
            }

            # Pairwise comparisons vs full
            if 'full' in per_question_scores:
                summary['statistics']['vs_full'] = {}
                full_scores = per_question_scores['full']
                for name, scores in per_question_scores.items():
                    if name != 'full':
                        t_test = StatisticalAnalyzer.paired_t_test(full_scores, scores)
                        cohens_d = StatisticalAnalyzer.cohens_d(full_scores, scores)
                        summary['statistics']['vs_full'][name] = {
                            't_test': {k: (float(v) if hasattr(v, 'item') else bool(v) if isinstance(v, (bool, type(True))) else v) for k, v in t_test.items()},
                            'cohens_d': float(cohens_d) if hasattr(cohens_d, 'item') else cohens_d
                        }

        # Save to file
        results_path = self.output_path / "results.json"
        with open(results_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"[Ablation] Results saved to {results_path}")

        return summary


def run_ablation_study(
    n_questions: int = 100,
    k_candidates: int = 5,
    seed: int = 42,
    configs: Optional[List[str]] = None
) -> Dict:
    """
    Convenience function to run ablation study.

    Args:
        n_questions: Number of questions to evaluate
        k_candidates: Number of candidates per question
        seed: Random seed
        configs: List of configuration names to run (None = all)

    Returns:
        Summary dictionary
    """
    config = ExperimentConfig(
        n_questions=n_questions,
        k_candidates=k_candidates,
        seed=seed
    )

    study = AblationStudy(config)
    return study.run(configs_to_run=configs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ECR ablation study")
    parser.add_argument("--n-questions", type=int, default=100,
                        help="Number of questions")
    parser.add_argument("--k-candidates", type=int, default=5,
                        help="Candidates per question")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--configs", nargs="*",
                        help="Specific configs to run (default: all)")

    args = parser.parse_args()

    run_ablation_study(
        n_questions=args.n_questions,
        k_candidates=args.k_candidates,
        seed=args.seed,
        configs=args.configs
    )
