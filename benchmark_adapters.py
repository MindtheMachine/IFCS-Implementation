"""
Benchmark Adapters
Adapt trilogy system to handle benchmark-specific requirements
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional, List
import re

from benchmark_loader import BenchmarkExample
from trilogy_orchestrator import TrilogyResult


class BaseBenchmarkAdapter(ABC):
    """Abstract base adapter for benchmarks"""

    @abstractmethod
    def prepare_prompt(self, example: BenchmarkExample) -> Tuple[str, str]:
        """Prepare prompt and context for trilogy system

        Args:
            example: Benchmark example

        Returns:
            (prompt, context) tuple
        """
        pass

    @abstractmethod
    def extract_answer(
        self,
        trilogy_result: TrilogyResult,
        example: BenchmarkExample
    ) -> Any:
        """Extract answer from trilogy output for metric computation

        Args:
            trilogy_result: Result from trilogy system
            example: Original benchmark example

        Returns:
            Extracted answer in benchmark-specific format
        """
        pass


class TruthfulQAAdapter(BaseBenchmarkAdapter):
    """Adapter for TruthfulQA multiple choice format"""

    def __init__(self, prompt_strategy: str = "choices_in_prompt"):
        """Initialize adapter

        Args:
            prompt_strategy: How to present choices
                - 'choices_in_prompt': Include choices A, B, C, D in prompt
                - 'freeform_then_match': Generate answer, then match to choices
        """
        if prompt_strategy not in ['choices_in_prompt', 'freeform_then_match']:
            raise ValueError(
                f"Invalid prompt_strategy: {prompt_strategy}. "
                f"Must be 'choices_in_prompt' or 'freeform_then_match'"
            )

        self.prompt_strategy = prompt_strategy

    def prepare_prompt(self, example: BenchmarkExample) -> Tuple[str, str]:
        """Format question with multiple choice options

        Args:
            example: TruthfulQA example

        Returns:
            (prompt, context) tuple
        """
        question = example.prompt

        if self.prompt_strategy == 'choices_in_prompt':
            # Strategy 1: Include choices A, B, C, D in prompt
            choices = example.ground_truth['mc1_choices']

            choice_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            formatted_choices = []

            for i, choice in enumerate(choices[:len(choice_letters)]):
                formatted_choices.append(f"{choice_letters[i]}) {choice}")

            prompt = f"""{question}

{chr(10).join(formatted_choices)}

Select the most accurate answer."""

            return prompt, ""

        else:
            # Strategy 2: Free-form answer, will match later
            return question, ""

    def extract_answer(
        self,
        trilogy_result: TrilogyResult,
        example: BenchmarkExample
    ) -> Dict[str, Any]:
        """Extract selected choice and compute probability distribution

        Args:
            trilogy_result: Result from trilogy
            example: Original example

        Returns:
            {
                'selected_choice_idx': int,  # For MC1
                'choice_probabilities': [float, ...],  # For MC2
                'raw_response': str
            }
        """
        response = trilogy_result.final_response
        choices = example.ground_truth['mc1_choices']

        if self.prompt_strategy == 'choices_in_prompt':
            # Parse choice letter from response
            selected_idx = self._parse_choice_selection(response)

            if selected_idx is None:
                # Fallback: try semantic matching
                selected_idx = self._compute_best_match(response, choices)

        else:
            # Semantic matching strategy
            selected_idx = self._compute_best_match(response, choices)

        # Compute probability distribution (simple approach: 1.0 for selected, distribute rest)
        num_choices = len(choices)
        probabilities = [0.1 / (num_choices - 1) if i != selected_idx else 0.9
                        for i in range(num_choices)]

        # Normalize
        prob_sum = sum(probabilities)
        probabilities = [p / prob_sum for p in probabilities]

        return {
            'selected_choice_idx': selected_idx if selected_idx is not None else 0,
            'choice_probabilities': probabilities,
            'raw_response': response
        }

    def _parse_choice_selection(self, response: str) -> Optional[int]:
        """Parse response to extract selected choice (A/B/C/D → 0/1/2/3)

        Args:
            response: LLM response

        Returns:
            Choice index or None if parsing failed
        """
        # Try multiple patterns to extract choice letter
        patterns = [
            r'\b([A-H])\)',  # "A)", "B)"
            r'\b([A-H])\.',  # "A.", "B."
            r'answer is ([A-H])',  # "answer is A"
            r'select ([A-H])',  # "select A"
            r'choice ([A-H])',  # "choice A"
            r'^([A-H])\b',  # "A" at start
            r'\(([A-H])\)',  # "(A)", "(B)"
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                letter = match.group(1).upper()
                # Convert letter to index (A→0, B→1, etc.)
                return ord(letter) - ord('A')

        return None

    def _compute_best_match(self, response: str, choices: List[str]) -> int:
        """Compute best matching choice using word overlap

        Args:
            response: LLM response
            choices: List of choice texts

        Returns:
            Index of best matching choice
        """
        response_words = set(re.findall(r'\b\w+\b', response.lower()))

        best_score = -1
        best_idx = 0

        for i, choice in enumerate(choices):
            choice_words = set(re.findall(r'\b\w+\b', choice.lower()))
            overlap = len(response_words & choice_words)
            score = overlap / max(len(choice_words), 1)

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx


class ASQAAdapter(BaseBenchmarkAdapter):
    """Adapter for ASQA long-form question answering"""

    def prepare_prompt(self, example: BenchmarkExample) -> Tuple[str, str]:
        """Keep question as-is (no modification needed)

        Args:
            example: ASQA example

        Returns:
            (prompt, context) tuple
        """
        # ASQA questions are ambiguous by design - no need to modify
        return example.prompt, ""

    def extract_answer(
        self,
        trilogy_result: TrilogyResult,
        example: BenchmarkExample
    ) -> str:
        """Extract long-form answer directly

        Args:
            trilogy_result: Result from trilogy
            example: Original example

        Returns:
            Answer text
        """
        # For ASQA, the entire response is the answer
        return trilogy_result.final_response


def test_adapters():
    """Test function for adapters"""
    from benchmark_loader import BenchmarkExample

    print("\n" + "=" * 80)
    print("Testing TruthfulQA Adapter")
    print("=" * 80)

    # Create test example
    truthfulqa_example = BenchmarkExample(
        id="test_1",
        prompt="What color is the sky?",
        ground_truth={
            'mc1_choices': ['Blue', 'Red', 'Green', 'Yellow'],
            'mc1_labels': [1, 0, 0, 0],
            'mc2_choices': ['Blue', 'Red', 'Green', 'Yellow'],
            'mc2_labels': [1, 0, 0, 0]
        }
    )

    adapter = TruthfulQAAdapter(prompt_strategy="choices_in_prompt")
    prompt, context = adapter.prepare_prompt(truthfulqa_example)

    print("\nFormatted Prompt:")
    print(prompt)
    print(f"\nContext: {context if context else '(empty)'}")

    # Test choice parsing
    test_responses = [
        "The answer is A) Blue",
        "I select B.",
        "Choice C is correct",
        "D",
        "The sky is blue, which matches the first option"
    ]

    print("\nTesting Choice Extraction:")
    for resp in test_responses:
        idx = adapter._parse_choice_selection(resp)
        print(f"  Response: '{resp[:50]}...' → Choice: {idx}")

    print("\n" + "=" * 80)
    print("Testing ASQA Adapter")
    print("=" * 80)

    asqa_example = BenchmarkExample(
        id="test_2",
        prompt="Who is the president?",
        ground_truth={
            'annotations': [],
            'qa_pairs': []
        }
    )

    asqa_adapter = ASQAAdapter()
    prompt, context = asqa_adapter.prepare_prompt(asqa_example)

    print(f"\nPrompt: {prompt}")
    print(f"Context: {context if context else '(empty)'}")


if __name__ == '__main__':
    test_adapters()
