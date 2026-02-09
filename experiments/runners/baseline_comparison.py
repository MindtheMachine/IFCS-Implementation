"""
Baseline Comparison Experiment

Compares ECR against Self-Consistency and Vanilla baselines on TruthfulQA.

This is Experiment 1 for the ECR empirical validation paper.
"""

import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import asdict
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_provider import LLMProviderFactory, LLMProvider
from ecr_engine import ECREngine, CoherenceMetrics
from benchmark_loader import BenchmarkLoader, BenchmarkExample
from benchmark_adapters import TruthfulQAAdapter
from trilogy_config import ECRConfig

from experiments.baselines.self_consistency import SelfConsistencyBaseline
from experiments.baselines.vanilla import VanillaBaseline
from experiments.configs.experiment_config import ExperimentConfig
from experiments.utils.result_logger import ResultLogger, QuestionResult
from experiments.analysis.statistical_tests import StatisticalAnalyzer


class BaselineComparisonExperiment:
    """
    Compare ECR against baseline methods on TruthfulQA.

    Methods compared:
    1. ECR (Evaluative Coherence Regulation) - our method
    2. Self-Consistency - semantic majority voting (Wang et al., 2022)
    3. Vanilla - single-shot LLM response
    """

    def __init__(self, config: ExperimentConfig = None):
        """
        Initialize experiment.

        Args:
            config: Experiment configuration (uses defaults if None)
        """
        self.config = config or ExperimentConfig()

        # Initialize LLM provider from environment
        print("[Experiment] Initializing LLM provider from .env...")
        self.llm_provider = LLMProviderFactory.create_from_env()
        self.model_name = self.llm_provider.get_model_name()
        print(f"[Experiment] Using model: {self.model_name}")

        # Initialize ECR engine
        ecr_config = ECRConfig(
            K=self.config.k_candidates,
            H=self.config.ecr_h,
            tau_CCI=self.config.ecr_tau_cci
        )
        self.ecr_engine = ECREngine(ecr_config)

        # Initialize baselines
        self.sc_baseline = SelfConsistencyBaseline(
            llm_provider=self.llm_provider,
            num_samples=self.config.k_candidates,
            similarity_threshold=self.config.sc_similarity_threshold,
            temperature=self.config.temperature,
            embedding_model=self.config.sc_embedding_model
        )
        self.vanilla_baseline = VanillaBaseline(
            llm_provider=self.llm_provider,
            temperature=self.config.temperature
        )

        # Initialize adapter for TruthfulQA
        self.adapter = TruthfulQAAdapter(prompt_strategy="choices_in_prompt")

        # Initialize result logger
        self.logger = ResultLogger(
            experiment_name="baseline_comparison",
            model_name=self.model_name,
            output_dir=self.config.output_dir,
            checkpoint_every=self.config.checkpoint_every
        )

    def _llm_call_fn(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 2000
    ) -> str:
        """Wrapper for LLM calls compatible with ECR engine."""
        return self.llm_provider.generate(
            prompt=prompt,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens
        )

    def _is_correct(
        self,
        response: str,
        example: BenchmarkExample
    ) -> bool:
        """
        Check if response is correct for TruthfulQA.

        For multiple choice, we parse the choice letter and check against labels.
        """
        # Parse choice from response
        selected_idx = self.adapter._parse_choice_selection(response)

        if selected_idx is None:
            # Fallback to semantic matching
            choices = example.ground_truth['mc1_choices']
            selected_idx = self.adapter._compute_best_match(response, choices)

        # Check if selected choice is correct
        labels = example.ground_truth['mc1_labels']
        if selected_idx < len(labels):
            return labels[selected_idx] == 1

        return False

    def _get_correct_answer_text(self, example: BenchmarkExample) -> List[str]:
        """Get the text of correct answers for an example."""
        choices = example.ground_truth['mc1_choices']
        labels = example.ground_truth['mc1_labels']
        return [choices[i] for i, label in enumerate(labels) if label == 1]

    def run(
        self,
        n_questions: Optional[int] = None,
        resume: bool = True
    ) -> Dict:
        """
        Run the baseline comparison experiment.

        Args:
            n_questions: Number of questions to evaluate (None = use config)
            resume: Whether to resume from checkpoint if available

        Returns:
            Summary dictionary with results
        """
        n_questions = n_questions or self.config.n_questions

        # Load TruthfulQA dataset
        print("[Experiment] Loading TruthfulQA dataset...")
        loader = BenchmarkLoader()
        all_examples = loader.load_truthfulqa()

        # Sample questions randomly
        random.seed(self.config.seed)
        if n_questions < len(all_examples):
            examples = random.sample(all_examples, n_questions)
        else:
            examples = all_examples[:n_questions]

        print(f"[Experiment] Selected {len(examples)} questions for evaluation")

        # Check for checkpoint
        start_idx = 0
        if resume:
            start_idx = self.logger.load_checkpoint()
            if start_idx > 0:
                print(f"[Experiment] Resuming from question {start_idx + 1}")

        # Run evaluation
        for i, example in enumerate(tqdm(examples[start_idx:], desc="Evaluating")):
            question_idx = start_idx + i

            try:
                result = self._evaluate_question(question_idx, example)
                self.logger.add_result(result)
            except Exception as e:
                print(f"\n[ERROR] Failed on question {question_idx}: {e}")
                # Log partial result
                result = QuestionResult(
                    question_id=question_idx,
                    question=example.prompt,
                    correct_answers=self._get_correct_answer_text(example),
                    candidates=[]
                )
                self.logger.add_result(result)

        # Save final results
        config_dict = asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else vars(self.config)
        summary = self.logger.save_final_results(config=config_dict)

        # Compute and save statistics
        self._compute_statistics(summary)

        print("\n[Experiment] Complete!")
        print(f"  ECR Accuracy:              {summary['results']['ecr']['accuracy']:.1%}")
        print(f"  Self-Consistency Accuracy: {summary['results']['self_consistency']['accuracy']:.1%}")
        print(f"  Vanilla Accuracy:          {summary['results']['vanilla']['accuracy']:.1%}")

        return summary

    def _evaluate_question(
        self,
        question_idx: int,
        example: BenchmarkExample
    ) -> QuestionResult:
        """Evaluate a single question with all methods."""

        # Prepare prompt
        prompt, _ = self.adapter.prepare_prompt(example)

        # Generate candidates once (shared across all methods for fair comparison)
        print(f"\n[Q{question_idx}] Generating {self.config.k_candidates} candidates...")
        candidates = self.ecr_engine.generate_candidates(
            prompt=prompt,
            llm_call_fn=self._llm_call_fn,
            num_candidates=self.config.k_candidates,
            llm_provider=self.llm_provider
        )

        # Method 1: ECR selection
        print(f"[Q{question_idx}] Running ECR selection...")
        ecr_selected, ecr_metrics, ecr_debug = self.ecr_engine.select_best_candidate(
            candidates=candidates,
            prompt=prompt,
            llm_call_fn=self._llm_call_fn
        )
        ecr_correct = self._is_correct(ecr_selected, example)

        # Method 2: Self-Consistency selection
        print(f"[Q{question_idx}] Running Self-Consistency selection...")
        sc_selected, sc_debug = self.sc_baseline.select_from_existing(candidates)
        sc_correct = self._is_correct(sc_selected, example)

        # Method 3: Vanilla selection (first candidate)
        print(f"[Q{question_idx}] Running Vanilla selection...")
        vanilla_selected, vanilla_debug = self.vanilla_baseline.select_from_existing(candidates)
        vanilla_correct = self._is_correct(vanilla_selected, example)

        # Create result
        result = QuestionResult(
            question_id=question_idx,
            question=example.prompt,
            correct_answers=self._get_correct_answer_text(example),
            candidates=candidates,
            ecr_selected=ecr_selected,
            ecr_cci=ecr_metrics.CCI,
            ecr_metrics={
                'evb': ecr_metrics.EVB,
                'cr': ecr_metrics.CR,
                'ts': ecr_metrics.TS,
                'es': ecr_metrics.ES,
                'pd': ecr_metrics.PD
            },
            ecr_correct=ecr_correct,
            sc_selected=sc_selected,
            sc_cluster_info=sc_debug,
            sc_correct=sc_correct,
            vanilla_selected=vanilla_selected,
            vanilla_correct=vanilla_correct
        )

        return result

    def _compute_statistics(self, summary: Dict):
        """Compute and save statistical tests."""
        # Get per-question correctness
        ecr_correct = summary['per_question_correct']['ecr']
        sc_correct = summary['per_question_correct']['self_consistency']
        vanilla_correct = summary['per_question_correct']['vanilla']

        # Convert to numeric (True=1, False=0)
        ecr_scores = [1 if c else 0 for c in ecr_correct]
        sc_scores = [1 if c else 0 for c in sc_correct]
        vanilla_scores = [1 if c else 0 for c in vanilla_correct]

        stats = {}

        # Paired t-test: ECR vs Self-Consistency
        if len(ecr_scores) > 1:
            stats['ecr_vs_sc'] = StatisticalAnalyzer.paired_t_test(ecr_scores, sc_scores)
            stats['ecr_vs_sc']['cohens_d'] = StatisticalAnalyzer.cohens_d(ecr_scores, sc_scores)

        # Paired t-test: ECR vs Vanilla
        if len(ecr_scores) > 1:
            stats['ecr_vs_vanilla'] = StatisticalAnalyzer.paired_t_test(ecr_scores, vanilla_scores)
            stats['ecr_vs_vanilla']['cohens_d'] = StatisticalAnalyzer.cohens_d(ecr_scores, vanilla_scores)

        # McNemar's test (better for binary outcomes)
        stats['mcnemar_ecr_vs_sc'] = StatisticalAnalyzer.mcnemar_test(ecr_correct, sc_correct)
        stats['mcnemar_ecr_vs_vanilla'] = StatisticalAnalyzer.mcnemar_test(ecr_correct, vanilla_correct)

        # Confidence intervals
        stats['ecr_ci'] = StatisticalAnalyzer.confidence_interval(ecr_scores)
        stats['sc_ci'] = StatisticalAnalyzer.confidence_interval(sc_scores)
        stats['vanilla_ci'] = StatisticalAnalyzer.confidence_interval(vanilla_scores)

        # Save statistics
        import json
        stats_path = self.logger.output_path / "statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n[Statistics]")
        if 'ecr_vs_sc' in stats:
            print(f"  ECR vs SC: p={stats['ecr_vs_sc']['p_value']:.4f}, "
                  f"d={stats['ecr_vs_sc']['cohens_d']:.3f}")
        if 'ecr_vs_vanilla' in stats:
            print(f"  ECR vs Vanilla: p={stats['ecr_vs_vanilla']['p_value']:.4f}, "
                  f"d={stats['ecr_vs_vanilla']['cohens_d']:.3f}")


def run_baseline_comparison(
    n_questions: int = 100,
    k_candidates: int = 5,
    seed: int = 42,
    resume: bool = True
) -> Dict:
    """
    Convenience function to run baseline comparison experiment.

    Args:
        n_questions: Number of TruthfulQA questions to evaluate
        k_candidates: Number of candidates to generate per question
        seed: Random seed for reproducibility
        resume: Whether to resume from checkpoint

    Returns:
        Summary dictionary with results
    """
    config = ExperimentConfig(
        n_questions=n_questions,
        k_candidates=k_candidates,
        seed=seed
    )

    experiment = BaselineComparisonExperiment(config)
    return experiment.run(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ECR baseline comparison experiment")
    parser.add_argument("--n-questions", type=int, default=100,
                        help="Number of questions to evaluate")
    parser.add_argument("--k-candidates", type=int, default=5,
                        help="Number of candidates per question")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh (don't resume from checkpoint)")

    args = parser.parse_args()

    run_baseline_comparison(
        n_questions=args.n_questions,
        k_candidates=args.k_candidates,
        seed=args.seed,
        resume=not args.no_resume
    )
