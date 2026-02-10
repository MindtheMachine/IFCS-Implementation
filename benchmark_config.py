"""
Benchmark Configuration Module
Configuration classes for TruthfulQA and ASQA benchmark evaluation
"""

from dataclasses import dataclass
from typing import Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip loading .env file
    pass

# Import default model from trilogy configuration
try:
    from trilogy_config import TrilogyConfig
    DEFAULT_MODEL_NAME = TrilogyConfig().model
except (ImportError, ValueError, Exception):
    # Fallback if import fails or API key not configured yet
    # Try to get model from environment first
    import os
    DEFAULT_MODEL_NAME = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation

    Output Organization:
        All outputs are organized by LLM model name in the Results folder:
        Results/
          └── claude-sonnet-4-20250514/
              ├── truthfulqa_results.csv
              ├── truthfulqa_summary.json
              ├── truthfulqa_comparison.txt
              ├── truthfulqa_report.html
              └── .checkpoints/
    """

    # Dataset selection
    benchmark_name: str  # 'truthfulqa' or 'asqa'
    split: str = 'validation'  # or 'dev', 'test'

    # LLM model name for organizing outputs (auto-detected)
    model_name: str = DEFAULT_MODEL_NAME
    results_dir: str = "Results"  # Base directory for all results

    # Batch processing
    batch_size: int = 0  # 0 = process all
    batch_start_idx: int = 0  # For resuming/subsetting
    rate_limit_delay_s: float = 1.0  # Delay between API calls

    # Checkpointing
    enable_checkpoints: bool = True
    checkpoint_interval: int = 10  # Save every N examples
    # Auto-configured to Results/{model_name}/.checkpoints
    checkpoint_dir: Optional[str] = None

    # Output options
    include_full_text_in_csv: bool = False  # Include full text responses in CSV (makes files large)

    # Output paths (auto-configured to Results/{model_name}/ if None)
    results_csv_path: Optional[str] = None
    summary_json_path: Optional[str] = None
    comparison_path: Optional[str] = None
    html_report_path: Optional[str] = None

    # TruthfulQA specific
    truthfulqa_prompt_strategy: str = "choices_in_prompt"  # or 'freeform_then_match'

    # ASQA specific
    asqa_max_answer_length: int = 500  # Truncate long answers for processing

    # Error handling
    continue_on_error: bool = True
    max_retries: int = 3
    retry_delay_s: float = 5.0

    # Cache settings
    cache_dir: str = ".benchmark_cache"

    def __post_init__(self):
        """Validate configuration and auto-configure output paths"""
        import os

        # Validate inputs
        if self.benchmark_name not in ['truthfulqa', 'asqa']:
            raise ValueError(
                f"Invalid benchmark_name: {self.benchmark_name}. "
                f"Must be 'truthfulqa' or 'asqa'"
            )

        if self.batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {self.batch_size}")

        if self.rate_limit_delay_s < 0:
            raise ValueError(f"rate_limit_delay_s must be >= 0, got {self.rate_limit_delay_s}")

        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")

        # Create model-specific output directory (sanitize for filesystem)
        safe_model_name = self.model_name.replace("/", "-").replace("\\", "-").replace(":", "-")
        output_dir = os.path.join(self.results_dir, safe_model_name)
        os.makedirs(output_dir, exist_ok=True)

        # Auto-configure output paths if not explicitly set
        if self.results_csv_path is None:
            self.results_csv_path = os.path.join(
                output_dir, f"{self.benchmark_name}_results.csv"
            )
        if self.summary_json_path is None:
            self.summary_json_path = os.path.join(
                output_dir, f"{self.benchmark_name}_summary.json"
            )
        if self.comparison_path is None:
            self.comparison_path = os.path.join(
                output_dir, f"{self.benchmark_name}_comparison.txt"
            )
        if self.html_report_path is None:
            self.html_report_path = os.path.join(
                output_dir, f"{self.benchmark_name}_report.html"
            )

        # Auto-configure checkpoint directory under model folder
        if self.checkpoint_dir is None:
            self.checkpoint_dir = os.path.join(output_dir, ".checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)


# Default configurations for each benchmark
# Output paths are auto-configured to: Results/{model_name}/{benchmark}_*.{ext}
DEFAULT_TRUTHFULQA_CONFIG = BenchmarkConfig(
    benchmark_name='truthfulqa',
    split='validation',
    batch_size=0,  # Process all 817 examples
    rate_limit_delay_s=1.0
    # Paths auto-configured: Results/claude-sonnet-4-20250514/truthfulqa_results.csv, etc.
)

DEFAULT_ASQA_CONFIG = BenchmarkConfig(
    benchmark_name='asqa',
    split='dev',
    batch_size=100,  # Process subset by default (ASQA has 5000+ examples)
    rate_limit_delay_s=1.0
    # Paths auto-configured: Results/claude-sonnet-4-20250514/asqa_results.csv, etc.
)
