"""
Benchmark Orchestrator
Orchestrates batch processing of benchmark datasets with error handling and checkpointing
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable, Union
import time
import json
import os
from datetime import datetime

from benchmark_loader import BenchmarkExample
from benchmark_adapters import BaseBenchmarkAdapter, TruthfulQAAdapter, ASQAAdapter
from benchmark_metrics import TruthfulQAMetrics, ASQAMetrics, BenchmarkMetrics
from benchmark_config import BenchmarkConfig
from trilogy_orchestrator import TrilogyResult


@dataclass
class BenchmarkEvaluationResult:
    """Result from evaluating single benchmark example"""
    example_id: str
    benchmark_name: str

    # Original data
    prompt: str
    ground_truth: Any

    # Trilogy outputs
    baseline_response: str
    regulated_result: TrilogyResult

    # Benchmark metrics
    baseline_metrics: Optional[BenchmarkMetrics] = None
    regulated_metrics: Optional[BenchmarkMetrics] = None

    # Trilogy analysis
    comparison: Optional[Dict] = None

    # Processing info
    processing_time_s: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert TrilogyResult to dict
        if self.regulated_result:
            data['regulated_result'] = self.regulated_result.to_dict()
        return data


class BenchmarkOrchestrator:
    """Orchestrates benchmark evaluation with trilogy system"""

    def __init__(
        self,
        trilogy_app: Any,  # TrilogyApp instance
        adapter: BaseBenchmarkAdapter,
        metrics_computer: Union[TruthfulQAMetrics, ASQAMetrics],
        config: BenchmarkConfig
    ):
        """Initialize orchestrator

        Args:
            trilogy_app: Initialized TrilogyApp instance
            adapter: Benchmark-specific adapter
            metrics_computer: Benchmark-specific metrics
            config: BenchmarkConfig with batch settings
        """
        self.trilogy_app = trilogy_app
        self.adapter = adapter
        self.metrics_computer = metrics_computer
        self.config = config
        self.last_api_call_time = 0

        # Create checkpoint directory if needed
        if config.enable_checkpoints:
            os.makedirs(config.checkpoint_dir, exist_ok=True)

    def evaluate_single(
        self,
        example: BenchmarkExample
    ) -> BenchmarkEvaluationResult:
        """Evaluate single example through trilogy system

        Pipeline:
            1. Adapter prepares prompt/context
            2. Run baseline (unregulated LLM)
            3. Run trilogy (regulated)
            4. Extract answers from both
            5. Compute benchmark metrics for both
            6. Compare baseline vs regulated

        Args:
            example: Benchmark example

        Returns:
            BenchmarkEvaluationResult
        """
        start_time = time.time()
        max_retries = self.config.max_retries
        retry_delay = self.config.retry_delay_s

        for attempt in range(max_retries + 1):
            try:
                # Each benchmark item is an independent sample; avoid cross-example
                # interaction carryover in CP Type-2 history.
                if hasattr(self.trilogy_app, "trilogy") and hasattr(self.trilogy_app.trilogy, "reset_interaction"):
                    self.trilogy_app.trilogy.reset_interaction()

                # 1. Prepare prompt
                prompt, context = self.adapter.prepare_prompt(example)

                # 2. Run baseline
                baseline_response = self.trilogy_app.baseline.process(prompt)
                if self._is_llm_error_response(baseline_response):
                    raise RuntimeError(f"Baseline LLM error: {baseline_response}")

                # 3. Run trilogy
                regulated_result = self.trilogy_app.trilogy.process(prompt, context)
                if regulated_result and (
                    self._is_llm_error_response(regulated_result.final_response)
                    or self._is_llm_error_response(regulated_result.selected_response)
                    or self._is_llm_error_response(regulated_result.shaped_response)
                ):
                    raise RuntimeError(
                        "Regulated LLM error: "
                        f"{regulated_result.final_response}"
                    )

                # 4. Extract answers
                # Create pseudo TrilogyResult for baseline
                baseline_pseudo_result = TrilogyResult(
                    final_response=baseline_response,
                    selected_response=baseline_response,
                    shaped_response=baseline_response,
                    ecr_fired=False,
                    cp_type1_fired=False,
                    cp_type1_decision="not_evaluated",
                    ifcs_fired=False,
                    cp_type2_fired=False,
                    cp_type2_decision="not_evaluated",
                    ecr_metrics=None,
                    cp_type1_metrics=None,
                    ifcs_metrics=None,
                    cp_type2_metrics=None,
                    num_candidates=0,
                    selected_candidate_idx=0,
                    processing_time_ms=0.0
                )

                baseline_answer = self.adapter.extract_answer(baseline_pseudo_result, example)
                regulated_answer = self.adapter.extract_answer(regulated_result, example)

                # 5. Compute metrics
                baseline_metrics = self._compute_single_metrics(
                    baseline_answer,
                    example.ground_truth,
                    self.config.benchmark_name
                )

                regulated_metrics = self._compute_single_metrics(
                    regulated_answer,
                    example.ground_truth,
                    self.config.benchmark_name
                )

                # 6. Success!
                processing_time = time.time() - start_time

                return BenchmarkEvaluationResult(
                    example_id=example.id,
                    benchmark_name=self.config.benchmark_name,
                    prompt=example.prompt,
                    ground_truth=example.ground_truth,
                    baseline_response=baseline_response,
                    regulated_result=regulated_result,
                    baseline_metrics=baseline_metrics,
                    regulated_metrics=regulated_metrics,
                    comparison=self._create_comparison(baseline_metrics, regulated_metrics, regulated_result),
                    processing_time_s=processing_time,
                    error=None
                )

            except Exception as e:
                error_type = type(e).__name__

                # Check for rate limit errors
                if 'rate' in str(e).lower() or 'RateLimitError' in error_type:
                    # Exponential backoff
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"[Warning] Rate limit hit on {example.id}, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue

                # Other API errors
                elif 'APIError' in error_type or 'API' in error_type:
                    print(f"[Warning] API error on {example.id} (attempt {attempt+1}/{max_retries+1}): {e}")
                    if attempt < max_retries:
                        time.sleep(retry_delay)
                        continue

                # Unexpected error
                print(f"[Error] Failed to process {example.id}: {e}")

                if not self.config.continue_on_error:
                    raise

                # Return partial result with error
                processing_time = time.time() - start_time
                return BenchmarkEvaluationResult(
                    example_id=example.id,
                    benchmark_name=self.config.benchmark_name,
                    prompt=example.prompt,
                    ground_truth=example.ground_truth,
                    baseline_response="",
                    regulated_result=None,
                    error=f"{error_type}: {str(e)}",
                    processing_time_s=processing_time
                )

        # All retries exhausted
        processing_time = time.time() - start_time
        return BenchmarkEvaluationResult(
            example_id=example.id,
            benchmark_name=self.config.benchmark_name,
            prompt=example.prompt,
            ground_truth=example.ground_truth,
            baseline_response="",
            regulated_result=None,
            error=f"Max retries ({max_retries}) exhausted",
            processing_time_s=processing_time
        )

    def evaluate_batch(
        self,
        examples: List[BenchmarkExample],
        progress_callback: Optional[Callable] = None
    ) -> List[BenchmarkEvaluationResult]:
        """Evaluate batch of examples with progress tracking

        Features:
            - Progress bar / callback updates
            - Rate limiting (configurable delay between examples)
            - Error handling (continue on failure, log errors)
            - Checkpoint saving (resume from interruption)

        Args:
            examples: List of benchmark examples
            progress_callback: Optional callback(current, total, result)

        Returns:
            List of results (may be partial if errors occurred)
        """
        # Try to resume from checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, "latest.json")
        results = self._load_checkpoint(checkpoint_path)
        start_idx = len(results)

        if start_idx > 0:
            print(f"\n[Resume] Loaded {start_idx} results from checkpoint, resuming...")

        # Setup progress bar
        try:
            from tqdm import tqdm
            use_tqdm = True
            pbar = tqdm(total=len(examples), initial=start_idx, desc="Evaluating")
        except ImportError:
            use_tqdm = False
            pbar = None

        # Process remaining examples
        for i in range(start_idx, len(examples)):
            # Apply rate limiting (except for first call)
            if i > start_idx:
                self._apply_rate_limiting()

            example = examples[i]

            # Evaluate
            result = self.evaluate_single(example)
            results.append(result)

            # Progress callback
            if progress_callback:
                progress_callback(i + 1, len(examples), result)

            # Update progress bar
            if use_tqdm and pbar:
                pbar.update(1)
                if result.error:
                    pbar.set_postfix({"errors": sum(1 for r in results if r.error)})

            # Print progress (if no tqdm)
            if not use_tqdm:
                print(f"[Progress] {i+1}/{len(examples)} ({(i+1)/len(examples)*100:.1f}%)")

            # Checkpoint periodically
            if self.config.enable_checkpoints and (i + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint(results, checkpoint_path)

        # Close progress bar
        if use_tqdm and pbar:
            pbar.close()

        # Final checkpoint
        if self.config.enable_checkpoints:
            self._save_checkpoint(results, checkpoint_path)

        return results

    def _apply_rate_limiting(self):
        """Ensure minimum delay between API calls"""
        elapsed = time.time() - self.last_api_call_time
        if elapsed < self.config.rate_limit_delay_s:
            sleep_time = self.config.rate_limit_delay_s - elapsed
            time.sleep(sleep_time)
        self.last_api_call_time = time.time()

    def _save_checkpoint(
        self,
        results: List[BenchmarkEvaluationResult],
        path: str
    ):
        """Save intermediate results to checkpoint file"""
        try:
            # Convert results to serializable format
            serializable_results = []
            for r in results:
                r_dict = r.to_dict()
                # Handle non-serializable objects
                serializable_results.append(r_dict)

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, default=str)

            print(f"[Checkpoint] Saved {len(results)} results to {path}")

        except Exception as e:
            print(f"[Warning] Failed to save checkpoint: {e}")

    def _load_checkpoint(self, path: str) -> List[BenchmarkEvaluationResult]:
        """Load results from checkpoint file"""
        if not os.path.exists(path):
            return []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                print(f"[Warning] Invalid checkpoint format at {path}: expected list")
                return []

            results: List[BenchmarkEvaluationResult] = []
            for item in data:
                regulated_result = None
                if isinstance(item.get("regulated_result"), dict):
                    regulated_result = TrilogyResult(**item["regulated_result"])

                baseline_metrics = None
                if isinstance(item.get("baseline_metrics"), dict):
                    baseline_metrics = BenchmarkMetrics(**item["baseline_metrics"])

                regulated_metrics = None
                if isinstance(item.get("regulated_metrics"), dict):
                    regulated_metrics = BenchmarkMetrics(**item["regulated_metrics"])

                result = BenchmarkEvaluationResult(
                    example_id=item.get("example_id", ""),
                    benchmark_name=item.get("benchmark_name", self.config.benchmark_name),
                    prompt=item.get("prompt", ""),
                    ground_truth=item.get("ground_truth"),
                    baseline_response=item.get("baseline_response", ""),
                    regulated_result=regulated_result,
                    baseline_metrics=baseline_metrics,
                    regulated_metrics=regulated_metrics,
                    comparison=item.get("comparison"),
                    processing_time_s=item.get("processing_time_s", 0.0),
                    error=item.get("error")
                )
                results.append(result)

            print(f"[Resume] Loaded {len(results)} results from checkpoint")
            return results

        except Exception as e:
            print(f"[Warning] Failed to load checkpoint: {e}")
            return []

    @staticmethod
    def _is_llm_error_response(text: Any) -> bool:
        """Detect provider-level failures that should not be scored as valid answers."""
        if not isinstance(text, str):
            return False
        lowered = text.lower()
        return (
            "error calling llm" in lowered
            or "failed to connect to ollama" in lowered
            or "status code: 404" in lowered
        )

    def _compute_single_metrics(
        self,
        answer: Any,
        ground_truth: Any,
        benchmark_name: str
    ) -> BenchmarkMetrics:
        """Compute metrics for single example

        Args:
            answer: Extracted answer from adapter
            ground_truth: Ground truth data
            benchmark_name: 'truthfulqa' or 'asqa'

        Returns:
            BenchmarkMetrics
        """
        if benchmark_name == 'truthfulqa':
            # TruthfulQA: answer is a dict with selected_choice_idx and probabilities
            predictions = [answer]
            gt = [ground_truth]
            return TruthfulQAMetrics.compute_metrics(predictions, gt)

        elif benchmark_name == 'asqa':
            # ASQA: answer is a string (long-form answer)
            predictions = [answer]
            gt = [ground_truth]
            return self.metrics_computer.compute_metrics(predictions, gt)

        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

    def _create_comparison(
        self,
        baseline_metrics: BenchmarkMetrics,
        regulated_metrics: BenchmarkMetrics,
        regulated_result: TrilogyResult
    ) -> Dict:
        """Create comparison summary

        Args:
            baseline_metrics: Baseline metrics
            regulated_metrics: Regulated metrics
            regulated_result: Regulated trilogy result

        Returns:
            Comparison dictionary
        """
        comparison = {
            'mechanisms_fired': {
                'ecr': regulated_result.ecr_fired if regulated_result else False,
                'cp_type1': regulated_result.cp_type1_fired if regulated_result else False,
                'ifcs': regulated_result.ifcs_fired if regulated_result else False,
                'cp_type2': regulated_result.cp_type2_fired if regulated_result else False
            },
            'metric_improvements': {}
        }

        # Calculate improvements
        if baseline_metrics and regulated_metrics:
            for metric_name in baseline_metrics.scores.keys():
                if metric_name in regulated_metrics.scores:
                    baseline_val = baseline_metrics.scores[metric_name]
                    regulated_val = regulated_metrics.scores[metric_name]
                    improvement = regulated_val - baseline_val
                    comparison['metric_improvements'][metric_name] = improvement

        return comparison
