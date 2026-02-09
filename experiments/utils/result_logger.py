"""
Result Logger

Utilities for logging, saving, and loading experiment results.
Supports checkpointing for resumable experiments.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class QuestionResult:
    """Result for a single question."""
    question_id: int
    question: str
    correct_answers: List[str]
    candidates: List[str]

    # Method results
    ecr_selected: Optional[str] = None
    ecr_cci: Optional[float] = None
    ecr_metrics: Optional[Dict[str, float]] = None
    ecr_correct: Optional[bool] = None

    sc_selected: Optional[str] = None
    sc_cluster_info: Optional[Dict] = None
    sc_correct: Optional[bool] = None

    vanilla_selected: Optional[str] = None
    vanilla_correct: Optional[bool] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class ResultLogger:
    """
    Logger for experiment results with checkpointing support.

    Handles:
    - Saving results incrementally
    - Checkpointing for resume
    - Final aggregation and export
    """

    def __init__(
        self,
        experiment_name: str,
        model_name: str,
        output_dir: str = "Results/experiments",
        checkpoint_every: int = 10
    ):
        """
        Initialize result logger.

        Args:
            experiment_name: Name of the experiment (e.g., "baseline_comparison")
            model_name: Name of the model being evaluated
            output_dir: Base output directory
            checkpoint_every: Save checkpoint every N questions
        """
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.checkpoint_every = checkpoint_every

        # Sanitize model name for filesystem (replace invalid chars)
        safe_model_name = model_name.replace(':', '_').replace('/', '_').replace('\\', '_')

        # Create output directory
        self.output_path = Path(output_dir) / experiment_name / safe_model_name
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results: List[QuestionResult] = []
        self.metadata: Dict[str, Any] = {
            'experiment': experiment_name,
            'model': model_name,
            'started_at': datetime.now().isoformat(),
            'completed_at': None
        }

    def add_result(self, result: QuestionResult):
        """Add a single question result."""
        self.results.append(result)

        # Checkpoint if needed
        if len(self.results) % self.checkpoint_every == 0:
            self.save_checkpoint()

    def save_checkpoint(self):
        """Save current results as checkpoint."""
        checkpoint_path = self.output_path / "checkpoint.json"
        data = {
            'metadata': self.metadata,
            'results': [r.to_dict() for r in self.results],
            'checkpoint_at': datetime.now().isoformat()
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_checkpoint(self) -> int:
        """
        Load results from checkpoint if exists.

        Returns:
            Number of questions already processed (for resume)
        """
        checkpoint_path = self.output_path / "checkpoint.json"
        if not checkpoint_path.exists():
            return 0

        with open(checkpoint_path, 'r') as f:
            data = json.load(f)

        self.metadata = data.get('metadata', self.metadata)
        self.results = [
            QuestionResult(**r) for r in data.get('results', [])
        ]
        return len(self.results)

    def save_final_results(self, config: Dict[str, Any] = None):
        """
        Save final results and summary.

        Args:
            config: Experiment configuration to save
        """
        self.metadata['completed_at'] = datetime.now().isoformat()
        self.metadata['n_questions'] = len(self.results)

        if config:
            self.metadata['config'] = config

        # Save full results
        results_path = self.output_path / "results.json"
        data = {
            'metadata': self.metadata,
            'questions': [r.to_dict() for r in self.results]
        }
        with open(results_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Compute and save summary
        summary = self._compute_summary()
        summary_path = self.output_path / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Remove checkpoint after successful save
        checkpoint_path = self.output_path / "checkpoint.json"
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        return summary

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute aggregate summary statistics."""
        if not self.results:
            return {'error': 'No results to summarize'}

        # Count correct answers per method
        ecr_correct = sum(1 for r in self.results if r.ecr_correct)
        sc_correct = sum(1 for r in self.results if r.sc_correct)
        vanilla_correct = sum(1 for r in self.results if r.vanilla_correct)
        total = len(self.results)

        # Collect CCI scores
        cci_scores = [r.ecr_cci for r in self.results if r.ecr_cci is not None]

        # Collect individual metric scores
        metric_scores = {
            'evb': [], 'cr': [], 'ts': [], 'es': [], 'pd': []
        }
        for r in self.results:
            if r.ecr_metrics:
                for metric, scores in metric_scores.items():
                    if metric in r.ecr_metrics:
                        scores.append(r.ecr_metrics[metric])

        # Collect cluster sizes for self-consistency
        cluster_sizes = [
            r.sc_cluster_info.get('majority_cluster_size', 0)
            for r in self.results
            if r.sc_cluster_info
        ]

        summary = {
            'metadata': self.metadata,
            'results': {
                'ecr': {
                    'accuracy': ecr_correct / total if total > 0 else 0,
                    'correct': ecr_correct,
                    'total': total,
                    'mean_cci': sum(cci_scores) / len(cci_scores) if cci_scores else 0,
                    'cci_scores': cci_scores,
                    'metric_means': {
                        k: sum(v) / len(v) if v else 0
                        for k, v in metric_scores.items()
                    }
                },
                'self_consistency': {
                    'accuracy': sc_correct / total if total > 0 else 0,
                    'correct': sc_correct,
                    'total': total,
                    'mean_cluster_size': sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
                },
                'vanilla': {
                    'accuracy': vanilla_correct / total if total > 0 else 0,
                    'correct': vanilla_correct,
                    'total': total
                }
            },
            'per_question_correct': {
                'ecr': [r.ecr_correct for r in self.results],
                'self_consistency': [r.sc_correct for r in self.results],
                'vanilla': [r.vanilla_correct for r in self.results]
            }
        }

        return summary

    @staticmethod
    def load_results(results_path: str) -> Dict[str, Any]:
        """Load results from a saved file."""
        with open(results_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def load_summary(summary_path: str) -> Dict[str, Any]:
        """Load summary from a saved file."""
        with open(summary_path, 'r') as f:
            return json.load(f)
