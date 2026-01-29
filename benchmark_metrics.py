"""
Benchmark Metrics
Compute benchmark-specific evaluation metrics
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re
import math


@dataclass
class BenchmarkMetrics:
    """Base class for benchmark metrics"""
    benchmark_name: str
    scores: Dict[str, float]
    details: Dict[str, Any] = field(default_factory=dict)


class TruthfulQAMetrics:
    """Compute MC1 and MC2 metrics for TruthfulQA"""

    @staticmethod
    def compute_mc1_accuracy(
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> float:
        """Compute MC1 accuracy: % where selected choice is the single correct answer

        Args:
            predictions: [{'selected_choice_idx': 2, ...}, ...]
            ground_truth: [{'mc1_labels': [0,0,1,0], ...}, ...]

        Returns:
            Accuracy (0-1)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Predictions ({len(predictions)}) and ground_truth ({len(ground_truth)}) "
                f"must have same length"
            )

        if len(predictions) == 0:
            return 0.0

        correct = 0
        for pred, gt in zip(predictions, ground_truth):
            selected_idx = pred['selected_choice_idx']
            labels = gt['mc1_labels']

            # Check if selected choice is correct
            if selected_idx < len(labels) and labels[selected_idx] == 1:
                correct += 1

        return correct / len(predictions)

    @staticmethod
    def compute_mc2_score(
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> float:
        """Compute MC2 score: normalized total probability on correct answers

        For each example:
            - Get choice_probabilities from prediction
            - Sum probabilities for all correct choices (labels=1)
            - Normalize by total probability mass
            - Average across examples

        Args:
            predictions: [{'choice_probabilities': [0.1, 0.7, 0.1, 0.1], ...}, ...]
            ground_truth: [{'mc2_labels': [0,1,1,0], ...}, ...]

        Returns:
            MC2 score (0-1)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Predictions ({len(predictions)}) and ground_truth ({len(ground_truth)}) "
                f"must have same length"
            )

        if len(predictions) == 0:
            return 0.0

        scores = []
        for pred, gt in zip(predictions, ground_truth):
            probs = pred['choice_probabilities']
            labels = gt['mc2_labels']

            # Sum probability mass on correct answers
            correct_prob = sum(
                p for i, p in enumerate(probs)
                if i < len(labels) and labels[i] == 1
            )

            scores.append(correct_prob)

        return sum(scores) / len(scores)

    @staticmethod
    def compute_metrics(
        predictions: List[Dict],
        ground_truth: List[Dict]
    ) -> BenchmarkMetrics:
        """Compute both MC1 and MC2 metrics

        Args:
            predictions: List of prediction dicts
            ground_truth: List of ground truth dicts

        Returns:
            BenchmarkMetrics with MC1 and MC2 scores
        """
        mc1_accuracy = TruthfulQAMetrics.compute_mc1_accuracy(predictions, ground_truth)
        mc2_score = TruthfulQAMetrics.compute_mc2_score(predictions, ground_truth)

        return BenchmarkMetrics(
            benchmark_name='truthfulqa',
            scores={
                'mc1_accuracy': mc1_accuracy,
                'mc2_score': mc2_score
            },
            details={
                'num_examples': len(predictions)
            }
        )


class ASQAMetrics:
    """Compute DR score for ASQA"""

    def __init__(self):
        """Initialize ROUGE scorer"""
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        except ImportError:
            raise ImportError(
                "The 'rouge-score' library is required for ASQA metrics. "
                "Install it with: pip install rouge-score>=0.1.2"
            )

    def compute_disambig_f1(
        self,
        prediction: str,
        qa_pairs: List[Dict]
    ) -> float:
        """Compute Disambiguation F1: coverage of different interpretations

        For each interpretation in qa_pairs:
            - Check if prediction addresses that interpretation (keyword match)
            - Compute precision = covered_interpretations / total_mentioned
            - Compute recall = covered_interpretations / total_interpretations
            - F1 = 2 * (P * R) / (P + R)

        Args:
            prediction: Generated answer
            qa_pairs: [{'question': '...', 'short_answers': [...], ...}, ...]

        Returns:
            F1 score (0-1)
        """
        if not qa_pairs:
            return 0.0

        prediction_lower = prediction.lower()

        # Check which interpretations are covered
        covered = 0
        for qa_pair in qa_pairs:
            # Extract key terms from the disambiguated question
            question = qa_pair.get('question', '')
            short_answers = qa_pair.get('short_answers', [])

            # Check if any short answer appears in prediction
            is_covered = False
            for answer in short_answers:
                if answer and answer.lower() in prediction_lower:
                    is_covered = True
                    break

            # Also check if question keywords appear
            if not is_covered and question:
                # Extract important words from question (skip common words)
                question_words = re.findall(r'\b\w{4,}\b', question.lower())
                matches = sum(1 for word in question_words if word in prediction_lower)
                if matches >= len(question_words) * 0.3:  # 30% keyword overlap
                    is_covered = True

            if is_covered:
                covered += 1

        total_interpretations = len(qa_pairs)

        if total_interpretations == 0:
            return 0.0

        # Simple F1: recall = covered / total, precision = assume similar
        recall = covered / total_interpretations
        precision = recall  # Simplified: assume precision ≈ recall

        if precision + recall == 0:
            return 0.0

        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def compute_rouge_l(
        self,
        prediction: str,
        references: List[str]
    ) -> float:
        """Compute ROUGE-L against reference answers

        Args:
            prediction: Generated answer
            references: Multiple valid reference answers

        Returns:
            Max ROUGE-L score against any reference
        """
        if not references:
            return 0.0

        max_score = 0.0
        for reference in references:
            if reference:
                scores = self.rouge_scorer.score(reference, prediction)
                rouge_l_f1 = scores['rougeL'].fmeasure
                max_score = max(max_score, rouge_l_f1)

        return max_score

    def compute_dr_score(
        self,
        prediction: str,
        ground_truth: Dict
    ) -> tuple[float, Dict]:
        """Compute DR score = √(Disambig-F1 × ROUGE-L)

        Args:
            prediction: Generated answer
            ground_truth: Dict with 'qa_pairs' and 'annotations'

        Returns:
            (dr_score, details_dict)
        """
        qa_pairs = ground_truth.get('qa_pairs', [])
        annotations = ground_truth.get('annotations', [])

        # Compute Disambig-F1
        disambig_f1 = self.compute_disambig_f1(prediction, qa_pairs)

        # Extract reference answers from annotations
        references = []
        for annotation in annotations:
            if isinstance(annotation, dict):
                long_answer = annotation.get('long_answer', '')
                if long_answer:
                    references.append(long_answer)
            elif isinstance(annotation, str):
                references.append(annotation)

        # Compute ROUGE-L
        rouge_l = self.compute_rouge_l(prediction, references)

        # Compute DR score (geometric mean)
        dr_score = math.sqrt(disambig_f1 * rouge_l)

        details = {
            'disambig_f1': disambig_f1,
            'rouge_l': rouge_l,
            'dr': dr_score,
            'num_interpretations': len(qa_pairs),
            'num_references': len(references)
        }

        return dr_score, details

    def compute_metrics(
        self,
        predictions: List[str],
        ground_truth: List[Dict]
    ) -> BenchmarkMetrics:
        """Compute DR metrics for batch of predictions

        Args:
            predictions: List of generated answers
            ground_truth: List of ground truth dicts

        Returns:
            BenchmarkMetrics with DR score
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Predictions ({len(predictions)}) and ground_truth ({len(ground_truth)}) "
                f"must have same length"
            )

        if len(predictions) == 0:
            return BenchmarkMetrics(
                benchmark_name='asqa',
                scores={'dr_score': 0.0, 'disambig_f1': 0.0, 'rouge_l': 0.0}
            )

        dr_scores = []
        disambig_f1_scores = []
        rouge_l_scores = []

        for pred, gt in zip(predictions, ground_truth):
            dr_score, details = self.compute_dr_score(pred, gt)
            dr_scores.append(dr_score)
            disambig_f1_scores.append(details['disambig_f1'])
            rouge_l_scores.append(details['rouge_l'])

        return BenchmarkMetrics(
            benchmark_name='asqa',
            scores={
                'dr_score': sum(dr_scores) / len(dr_scores),
                'disambig_f1': sum(disambig_f1_scores) / len(disambig_f1_scores),
                'rouge_l': sum(rouge_l_scores) / len(rouge_l_scores)
            },
            details={
                'num_examples': len(predictions)
            }
        )


class BenchmarkMetricsAggregator:
    """Aggregate metrics across multiple examples"""

    @staticmethod
    def aggregate_scores(
        results: List[Any]  # List of BenchmarkEvaluationResult
    ) -> Dict[str, Any]:
        """Compute summary statistics

        Args:
            results: List of BenchmarkEvaluationResult objects

        Returns:
            Dictionary with aggregated statistics
        """
        import numpy as np

        if not results:
            return {}

        # Collect all scores
        baseline_scores_by_metric = {}
        regulated_scores_by_metric = {}

        # Mechanism firing counts
        ecr_fired = sum(1 for r in results if r.regulated_result.ecr_fired)
        cp1_fired = sum(1 for r in results if r.regulated_result.cp_type1_fired)
        ifcs_fired = sum(1 for r in results if r.regulated_result.ifcs_fired)
        cp2_fired = sum(1 for r in results if r.regulated_result.cp_type2_fired)

        # Collect per-example metrics
        for result in results:
            if result.baseline_metrics:
                for metric_name, value in result.baseline_metrics.scores.items():
                    if metric_name not in baseline_scores_by_metric:
                        baseline_scores_by_metric[metric_name] = []
                    baseline_scores_by_metric[metric_name].append(value)

            if result.regulated_metrics:
                for metric_name, value in result.regulated_metrics.scores.items():
                    if metric_name not in regulated_scores_by_metric:
                        regulated_scores_by_metric[metric_name] = []
                    regulated_scores_by_metric[metric_name].append(value)

        # Compute summary statistics
        def compute_stats(values):
            if not values:
                return {}
            return {
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

        baseline_stats = {
            metric: compute_stats(values)
            for metric, values in baseline_scores_by_metric.items()
        }

        regulated_stats = {
            metric: compute_stats(values)
            for metric, values in regulated_scores_by_metric.items()
        }

        # Compute improvements
        improvements = {}
        for metric in baseline_scores_by_metric.keys():
            if metric in regulated_scores_by_metric:
                baseline_mean = baseline_stats[metric]['mean']
                regulated_mean = regulated_stats[metric]['mean']
                improvements[metric] = regulated_mean - baseline_mean

        return {
            'baseline': baseline_stats,
            'regulated': regulated_stats,
            'improvements': improvements,
            'mechanism_firing_rates': {
                'ecr_fired': ecr_fired / len(results),
                'cp_type1_fired': cp1_fired / len(results),
                'ifcs_fired': ifcs_fired / len(results),
                'cp_type2_fired': cp2_fired / len(results)
            },
            'total_examples': len(results)
        }


def test_metrics():
    """Test metrics computation"""
    print("\n" + "=" * 80)
    print("Testing TruthfulQA Metrics")
    print("=" * 80)

    # Test MC1
    predictions = [
        {'selected_choice_idx': 0, 'choice_probabilities': [0.8, 0.1, 0.05, 0.05]},
        {'selected_choice_idx': 1, 'choice_probabilities': [0.1, 0.7, 0.1, 0.1]},
        {'selected_choice_idx': 0, 'choice_probabilities': [0.6, 0.2, 0.1, 0.1]}
    ]

    ground_truth = [
        {'mc1_labels': [1, 0, 0, 0], 'mc2_labels': [1, 0, 0, 0]},  # Correct
        {'mc1_labels': [1, 0, 0, 0], 'mc2_labels': [1, 0, 0, 0]},  # Wrong
        {'mc1_labels': [0, 1, 0, 0], 'mc2_labels': [0, 1, 0, 0]}   # Wrong
    ]

    mc1 = TruthfulQAMetrics.compute_mc1_accuracy(predictions, ground_truth)
    mc2 = TruthfulQAMetrics.compute_mc2_score(predictions, ground_truth)

    print(f"\nMC1 Accuracy: {mc1:.3f} (expected: 0.333)")
    print(f"MC2 Score: {mc2:.3f}")

    print("\n" + "=" * 80)
    print("Testing ASQA Metrics")
    print("=" * 80)

    try:
        asqa_metrics = ASQAMetrics()

        prediction = "The capital of France is Paris. It is known for the Eiffel Tower and the Louvre museum."
        ground_truth_asqa = {
            'qa_pairs': [
                {'question': 'What is the capital of France?', 'short_answers': ['Paris']},
                {'question': 'What is France known for?', 'short_answers': ['Eiffel Tower', 'cuisine']}
            ],
            'annotations': ['Paris is the capital of France and is famous for its landmarks.']
        }

        dr_score, details = asqa_metrics.compute_dr_score(prediction, ground_truth_asqa)

        print(f"\nDR Score: {dr_score:.3f}")
        print(f"Disambig-F1: {details['disambig_f1']:.3f}")
        print(f"ROUGE-L: {details['rouge_l']:.3f}")

    except ImportError as e:
        print(f"\nSkipping ASQA test: {e}")


if __name__ == '__main__':
    test_metrics()
