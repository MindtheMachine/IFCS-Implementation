"""
Benchmark Dataset Loader
Loads TruthfulQA and ASQA datasets from Hugging Face and normalizes formats
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os


@dataclass
class BenchmarkExample:
    """Unified benchmark example format"""
    id: str
    prompt: str
    context: str = ""
    ground_truth: Any = None  # Type varies by benchmark
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkLoader:
    """Loads benchmark datasets from Hugging Face"""

    def __init__(self, cache_dir: str = ".benchmark_cache"):
        """Initialize loader with cache directory

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

    def load_truthfulqa(self, split: str = "validation") -> List[BenchmarkExample]:
        """Load TruthfulQA dataset

        Args:
            split: Dataset split (validation is the primary split)

        Returns:
            List of BenchmarkExample with ground_truth containing:
            {
                'mc1_choices': [str, ...],  # Multiple choice options
                'mc1_labels': [int, ...],    # Binary labels (0 or 1)
                'mc2_choices': [str, ...],   # Same as mc1
                'mc2_labels': [int, ...]     # Binary labels (0 or 1)
            }
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required. Install it with: pip install datasets>=2.14.0"
            )

        print(f"[BenchmarkLoader] Loading TruthfulQA dataset (split={split})...")

        # Load dataset from Hugging Face
        dataset = load_dataset(
            "truthfulqa/truthful_qa",
            "multiple_choice",
            split=split,
            cache_dir=self.cache_dir
        )

        print(f"[BenchmarkLoader] Loaded {len(dataset)} examples from TruthfulQA")

        # Normalize to BenchmarkExample format
        examples = []
        for i, item in enumerate(dataset):
            example = BenchmarkExample(
                id=f"truthfulqa_{i}",
                prompt=item['question'],
                context="",  # TruthfulQA doesn't provide context
                ground_truth={
                    'mc1_choices': item['mc1_targets']['choices'],
                    'mc1_labels': item['mc1_targets']['labels'],
                    'mc2_choices': item['mc2_targets']['choices'],
                    'mc2_labels': item['mc2_targets']['labels']
                },
                metadata={
                    'category': item.get('category', 'unknown'),
                    'source': item.get('source', 'unknown')
                }
            )
            examples.append(example)

        return examples

    def load_asqa(self, split: str = "dev") -> List[BenchmarkExample]:
        """Load ASQA dataset

        Args:
            split: Dataset split (dev or train)

        Returns:
            List of BenchmarkExample with ground_truth containing:
            {
                'annotations': List[Dict],  # Multiple valid answer annotations
                'qa_pairs': List[Dict]      # Question-answer pairs for each interpretation
            }
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "The 'datasets' library is required. Install it with: pip install datasets>=2.14.0"
            )

        print(f"[BenchmarkLoader] Loading ASQA dataset (split={split})...")

        # Load dataset from Hugging Face
        dataset = load_dataset(
            "din0s/asqa",
            split=split,
            cache_dir=self.cache_dir
        )

        print(f"[BenchmarkLoader] Loaded {len(dataset)} examples from ASQA")

        # Normalize to BenchmarkExample format
        examples = []
        for i, item in enumerate(dataset):
            # ASQA has ambiguous questions with multiple valid interpretations
            example = BenchmarkExample(
                id=f"asqa_{i}",
                prompt=item['ambiguous_question'],
                context="",  # ASQA doesn't provide context at query time
                ground_truth={
                    'annotations': item.get('annotations', []),
                    'qa_pairs': item.get('qa_pairs', []),
                    'sample_id': item.get('sample_id', f'asqa_{i}')
                },
                metadata={
                    'disambiguated_questions': item.get('disambiguated_questions', [])
                }
            )
            examples.append(example)

        return examples

    def get_dataset_info(self, benchmark_name: str) -> Dict[str, Any]:
        """Get metadata about benchmark dataset

        Args:
            benchmark_name: 'truthfulqa' or 'asqa'

        Returns:
            Dictionary with dataset information
        """
        if benchmark_name == 'truthfulqa':
            return {
                'name': 'TruthfulQA',
                'description': 'Benchmark for measuring truthfulness in LLM responses',
                'paper': 'Lin et al. (2021)',
                'size': 817,
                'tasks': ['multiple_choice'],
                'metrics': ['MC1 (accuracy)', 'MC2 (calibrated probability)'],
                'license': 'Apache 2.0',
                'url': 'https://huggingface.co/datasets/truthfulqa/truthful_qa'
            }
        elif benchmark_name == 'asqa':
            return {
                'name': 'ASQA',
                'description': 'Answer Sentence Question Answering - ambiguous factoid questions',
                'paper': 'Stelmakh et al. (2022)',
                'size': 5300,
                'tasks': ['long_form_qa'],
                'metrics': ['DR score (Disambig-F1 × ROUGE-L)'],
                'license': 'Apache 2.0',
                'url': 'https://huggingface.co/datasets/din0s/asqa'
            }
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")


def test_loader():
    """Test function to verify dataset loading"""
    loader = BenchmarkLoader()

    # Test TruthfulQA
    print("\n" + "=" * 80)
    print("Testing TruthfulQA Loading")
    print("=" * 80)

    truthfulqa_examples = loader.load_truthfulqa()
    print(f"\nLoaded {len(truthfulqa_examples)} TruthfulQA examples")

    # Show first example
    if truthfulqa_examples:
        ex = truthfulqa_examples[0]
        print(f"\nExample 1:")
        print(f"  ID: {ex.id}")
        print(f"  Prompt: {ex.prompt}")
        print(f"  MC1 Choices: {len(ex.ground_truth['mc1_choices'])} options")
        print(f"  MC1 Labels: {ex.ground_truth['mc1_labels']}")
        print(f"  Category: {ex.metadata.get('category', 'N/A')}")

    # Test ASQA
    print("\n" + "=" * 80)
    print("Testing ASQA Loading")
    print("=" * 80)

    asqa_examples = loader.load_asqa()
    print(f"\nLoaded {len(asqa_examples)} ASQA examples")

    # Show first example
    if asqa_examples:
        ex = asqa_examples[0]
        print(f"\nExample 1:")
        print(f"  ID: {ex.id}")
        print(f"  Prompt: {ex.prompt}")
        print(f"  Annotations: {len(ex.ground_truth.get('annotations', []))}")
        print(f"  QA Pairs: {len(ex.ground_truth.get('qa_pairs', []))}")


if __name__ == '__main__':
    # Run test if executed directly
    test_loader()
