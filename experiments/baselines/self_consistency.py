"""
Self-Consistency Baseline

Implements self-consistency sampling (Wang et al., 2022) as a baseline for
comparing against ECR.

Self-consistency generates multiple candidate responses and selects the most
common answer through semantic clustering and majority voting.

Reference:
Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., ... & Zhou, D. (2022).
Self-consistency improves chain of thought reasoning in language models.
arXiv preprint arXiv:2203.11171.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_provider import LLMProvider
from experiments.utils.semantic_similarity import select_by_semantic_majority


class SelfConsistencyBaseline:
    """
    Self-Consistency baseline for response selection.

    Generates multiple candidate responses from an LLM and selects the most
    representative one using semantic similarity-based majority voting.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        num_samples: int = 5,
        similarity_threshold: float = 0.85,
        temperature: float = 0.7,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize self-consistency baseline.

        Args:
            llm_provider: LLM provider instance for generating candidates
            num_samples: Number of candidate responses to generate (K)
            similarity_threshold: Threshold for clustering similar responses (0-1)
            temperature: Sampling temperature for candidate generation
            embedding_model: Sentence transformer model for semantic similarity
        """
        self.llm_provider = llm_provider
        self.num_samples = num_samples
        self.similarity_threshold = similarity_threshold
        self.temperature = temperature
        self.embedding_model = embedding_model

    def generate_candidates(
        self,
        prompt: str,
        system: Optional[str] = None
    ) -> List[str]:
        """
        Generate K candidate responses for a prompt.

        Args:
            prompt: The input prompt
            system: Optional system prompt

        Returns:
            List of K candidate response strings
        """
        candidates = []
        for _ in range(self.num_samples):
            response = self.llm_provider.generate(
                prompt=prompt,
                temperature=self.temperature,
                system=system
            )
            candidates.append(response)
        return candidates

    def select(
        self,
        candidates: List[str]
    ) -> Tuple[str, Dict]:
        """
        Select the best candidate using semantic majority voting.

        Args:
            candidates: List of candidate response strings

        Returns:
            Tuple of (selected_response, debug_info) where debug_info contains:
            - num_candidates: Total number of candidates
            - num_clusters: Number of distinct semantic clusters
            - majority_cluster_size: Size of the largest cluster
            - cluster_distribution: List of all cluster sizes
            - selected_idx: Index of the selected candidate
        """
        selected, debug_info = select_by_semantic_majority(
            candidates=candidates,
            similarity_threshold=self.similarity_threshold,
            model_name=self.embedding_model
        )

        # Add total candidates to debug info
        debug_info['num_candidates'] = len(candidates)

        return selected, debug_info

    def generate_and_select(
        self,
        prompt: str,
        system: Optional[str] = None
    ) -> Tuple[str, List[str], Dict]:
        """
        Generate candidates and select the best one.

        Convenience method that combines generate_candidates and select.

        Args:
            prompt: The input prompt
            system: Optional system prompt

        Returns:
            Tuple of (selected_response, all_candidates, debug_info)
        """
        candidates = self.generate_candidates(prompt, system)
        selected, debug_info = self.select(candidates)
        return selected, candidates, debug_info

    def select_from_existing(
        self,
        candidates: List[str]
    ) -> Tuple[str, Dict]:
        """
        Select from pre-generated candidates (for fair comparison with ECR).

        Use this method when comparing against ECR to ensure both methods
        evaluate the same set of candidates.

        Args:
            candidates: Pre-generated candidate responses

        Returns:
            Tuple of (selected_response, debug_info)
        """
        return self.select(candidates)
