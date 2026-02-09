"""
Vanilla Baseline

Simple single-shot LLM response without any selection mechanism.
Used as a lower bound baseline for comparison.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_provider import LLMProvider


class VanillaBaseline:
    """
    Vanilla baseline - single-shot LLM response.

    This baseline simply returns the first/only response from the LLM
    without any candidate generation or selection mechanism.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        temperature: float = 0.7
    ):
        """
        Initialize vanilla baseline.

        Args:
            llm_provider: LLM provider instance
            temperature: Sampling temperature
        """
        self.llm_provider = llm_provider
        self.temperature = temperature

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None
    ) -> str:
        """
        Generate a single response.

        Args:
            prompt: The input prompt
            system: Optional system prompt

        Returns:
            Single response string
        """
        return self.llm_provider.generate(
            prompt=prompt,
            temperature=self.temperature,
            system=system
        )

    def select_from_existing(
        self,
        candidates: List[str]
    ) -> Tuple[str, Dict]:
        """
        Select from pre-generated candidates (returns first one).

        For fair comparison, when candidates are pre-generated, vanilla
        simply returns the first candidate (simulating random selection).

        Args:
            candidates: Pre-generated candidate responses

        Returns:
            Tuple of (first_candidate, debug_info)
        """
        if not candidates:
            raise ValueError("Cannot select from empty candidate list")

        return candidates[0], {
            'num_candidates': len(candidates),
            'selected_idx': 0,
            'method': 'first'
        }
