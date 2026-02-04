"""
Semantic Signal Framework for System-Wide Signal Estimation
Provides unified semantic similarity and fuzzy logic estimators without text-matching heuristics
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
import numpy as np

from enhanced_signal_estimator import enhanced_signal_estimator
from fuzzy_membership import TriangularMF
from signal_estimation import compute_text_stats


@dataclass
class SemanticSignals:
    """Unified semantic signal container"""
    intent: float
    risk_specificity: float
    polarity: float
    disagreement: float
    confidence: float
    authority: float
    grounding: float
    coherence: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy access"""
        return {
            "intent": self.intent,
            "risk_specificity": self.risk_specificity,
            "polarity": self.polarity,
            "disagreement": self.disagreement,
            "confidence": self.confidence,
            "authority": self.authority,
            "grounding": self.grounding,
            "coherence": self.coherence,
        }

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for matrix operations"""
        return np.array([
            self.intent,
            self.risk_specificity,
            self.polarity,
            self.disagreement,
            self.confidence,
            self.authority,
            self.grounding,
            self.coherence,
        ])


class SemanticSimilarityEngine:
    """Semantic similarity using statistical feature vectors (no pattern matching)."""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.feature_weights = self._load_feature_weights(config.get("feature_weights"))

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity based on statistical feature vectors."""
        if not text1 or not text2:
            return 0.0

        vec1 = self._extract_feature_vector(text1)
        vec2 = self._extract_feature_vector(text2)
        return self._cosine_similarity(vec1, vec2)

    def _load_feature_weights(self, weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        defaults = {
            "length_norm": 0.20,
            "type_token_ratio": 0.15,
            "avg_word_len": 0.10,
            "digit_ratio": 0.10,
            "uppercase_ratio": 0.05,
            "question_ratio": 0.10,
            "exclam_ratio": 0.05,
            "list_density": 0.10,
            "sentence_count_norm": 0.15,
        }
        if not weights:
            return defaults

        merged = {**defaults, **weights}
        total = sum(merged.values()) or 1.0
        return {k: v / total for k, v in merged.items()}

    def _extract_feature_vector(self, text: str) -> np.ndarray:
        stats = compute_text_stats(text)

        sentence_count_norm = min(1.0, math.log1p(stats["sentence_count"]) / math.log1p(50))
        avg_word_len_norm = min(1.0, stats["avg_word_len"] / 10.0)

        vec = np.array([
            stats["length_norm"] * self.feature_weights["length_norm"],
            stats["type_token_ratio"] * self.feature_weights["type_token_ratio"],
            avg_word_len_norm * self.feature_weights["avg_word_len"],
            stats["digit_ratio"] * self.feature_weights["digit_ratio"],
            stats["uppercase_ratio"] * self.feature_weights["uppercase_ratio"],
            stats["question_ratio"] * self.feature_weights["question_ratio"],
            stats["exclam_ratio"] * self.feature_weights["exclam_ratio"],
            stats["list_density"] * self.feature_weights["list_density"],
            sentence_count_norm * self.feature_weights["sentence_count_norm"],
        ])

        return vec

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        if denom == 0:
            return 0.0
        return float(max(0.0, min(1.0, np.dot(vec1, vec2) / denom)))

    def _extract_polarity(self, text: str) -> float:
        """Extract polarity score [-1, 1] using statistical stance proxies."""
        stats = compute_text_stats(text)
        strength = stats["exclam_ratio"] - stats["question_ratio"]
        return max(-1.0, min(1.0, strength))


class UnifiedSemanticSignalEstimator:
    """Unified semantic signal estimator for system-wide use"""

    def __init__(self, config: Optional[Dict] = None):
        config = config or self._load_config()
        self.similarity_engine = SemanticSimilarityEngine(config=config)
        self._fuzzy_memberships = self._init_fuzzy_memberships(config.get("fuzzy_memberships"))

    def estimate_semantic_signals(self, text: str, context: str = "") -> SemanticSignals:
        """Estimate comprehensive semantic signals for any text."""
        if not text or len(text.strip()) < 3:
            return SemanticSignals(
                intent=0.0,
                risk_specificity=0.0,
                polarity=0.0,
                disagreement=0.0,
                confidence=0.0,
                authority=0.0,
                grounding=0.0,
                coherence=0.0,
            )

        intent = self._fuzzify_signal("intent", self._estimate_intent_clarity(text))
        risk_specificity = self._fuzzify_signal("risk_specificity", self._estimate_risk_specificity(text))
        polarity = self._fuzzify_signal("polarity", self._estimate_polarity_strength(text))
        disagreement = self._fuzzify_signal("disagreement", self._estimate_disagreement_signals(text, context))

        confidence = enhanced_signal_estimator.estimate_epistemic_certainty(text)
        authority = enhanced_signal_estimator.estimate_authority_posture(text)
        grounding = self._estimate_grounding_strength(text, context)
        coherence = self._estimate_coherence_signals(text)

        return SemanticSignals(
            intent=intent,
            risk_specificity=risk_specificity,
            polarity=polarity,
            disagreement=disagreement,
            confidence=confidence,
            authority=authority,
            grounding=grounding,
            coherence=coherence,
        )

    def estimate_explicit_polarity_signal(self, text: str, context: str = "") -> float:
        return self._fuzzify_signal("polarity", self._estimate_polarity_strength(text))

    def estimate_explicit_disagreement_signal(self, text: str, context: str = "") -> float:
        base_disagreement = self._estimate_disagreement_signals(text, context)
        return self._fuzzify_signal("disagreement", base_disagreement)

    def estimate_risk_pattern_scores(self, text: str) -> Dict[str, float]:
        """Estimate risk pattern scores using statistical specificity only."""
        if not text or len(text.strip()) < 3:
            return {"pattern_1": 0.0, "pattern_2": 0.0, "pattern_3": 0.0}

        base = self._estimate_risk_specificity(text)
        return {
            "pattern_1": base,
            "pattern_2": min(1.0, base * 0.9),
            "pattern_3": min(1.0, base * 0.8),
        }

    def _init_fuzzy_memberships(self, config: Optional[Dict]) -> Dict[str, Dict[str, TriangularMF]]:
        defaults = {
            "intent": {
                "LOW": (0.0, 0.0, 0.4),
                "MEDIUM": (0.3, 0.55, 0.8),
                "HIGH": (0.7, 1.0, 1.0),
            },
            "risk_specificity": {
                "LOW": (0.0, 0.0, 0.4),
                "MEDIUM": (0.3, 0.6, 0.9),
                "HIGH": (0.8, 1.0, 1.0),
            },
            "polarity": {
                "LOW": (0.0, 0.0, 0.35),
                "MEDIUM": (0.25, 0.55, 0.85),
                "HIGH": (0.75, 1.0, 1.0),
            },
            "disagreement": {
                "LOW": (0.0, 0.0, 0.35),
                "MEDIUM": (0.25, 0.55, 0.85),
                "HIGH": (0.75, 1.0, 1.0),
            },
        }
        config = config or {}
        memberships = {}
        for signal, levels in defaults.items():
            signal_config = config.get(signal, {})
            memberships[signal] = {}
            for level, params in levels.items():
                tuned = signal_config.get(level, params)
                memberships[signal][level] = TriangularMF(*tuned)
        return memberships

    def _fuzzify_signal(self, signal_type: str, raw_score: float) -> float:
        memberships = self._fuzzy_memberships.get(signal_type)
        if not memberships:
            return max(0.0, min(1.0, raw_score))

        low = memberships["LOW"].membership(raw_score)
        med = memberships["MEDIUM"].membership(raw_score)
        high = memberships["HIGH"].membership(raw_score)

        weighted_sum = low * 0.2 + med * 0.55 + high * 0.9
        total = low + med + high
        if total <= 0:
            return 0.0
        return min(1.0, max(0.0, weighted_sum / total))

    def _estimate_intent_clarity(self, text: str) -> float:
        stats = compute_text_stats(text)
        score = (
            0.5 * stats["question_ratio"] +
            0.3 * stats["length_norm"] +
            0.2 * stats["type_token_ratio"]
        )
        return max(0.0, min(1.0, score))

    def _estimate_risk_specificity(self, text: str) -> float:
        stats = compute_text_stats(text)
        score = (
            0.4 * stats["length_norm"] +
            0.3 * stats["list_density"] +
            0.3 * min(1.0, stats["digit_ratio"] * 6.0)
        )
        return max(0.0, min(1.0, score))

    def _estimate_polarity_strength(self, text: str) -> float:
        polarity_value = self.similarity_engine._extract_polarity(text)
        return max(0.0, min(1.0, abs(polarity_value)))

    def _estimate_disagreement_signals(self, text: str, context: str = "") -> float:
        if not context:
            return 0.0
        similarity = self.similarity_engine.compute_semantic_similarity(text, context)
        return max(0.0, min(1.0, 1.0 - similarity))

    def _estimate_grounding_strength(self, text: str, context: str = "") -> float:
        evidential_risk = enhanced_signal_estimator.estimate_evidential_risk(text, context)
        base_grounding = 1.0 - evidential_risk
        stats = compute_text_stats(text, context)
        structure_boost = min(0.2, stats["list_density"] * 0.2 + stats["digit_ratio"] * 2.0)
        return min(1.0, max(0.0, base_grounding + structure_boost))

    def _estimate_coherence_signals(self, text: str) -> float:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if len(sentences) < 2:
            return 0.8

        lengths = [len(s.split()) for s in sentences if s]
        if not lengths:
            return 0.5

        avg_len = sum(lengths) / len(lengths)
        variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
        variance_norm = min(1.0, variance / (avg_len + 1.0))

        stats = compute_text_stats(text)
        consistency = stats["type_token_ratio"]
        coherence = 1.0 - variance_norm * 0.5 + consistency * 0.2
        return max(0.0, min(1.0, coherence))

    def _load_config(self) -> Dict:
        """Load semantic signal configuration from disk."""
        import os
        import json

        default_path = os.path.join(os.path.dirname(__file__), "config", "semantic_signal_config.json")
        config_path = os.getenv("SEMANTIC_SIGNAL_CONFIG", default_path)
        try:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
        except Exception:
            pass
        return {}


unified_semantic_estimator = UnifiedSemanticSignalEstimator()


def test_semantic_framework():
    """Test the semantic signal framework"""
    estimator = unified_semantic_estimator

    test_texts = [
        "Can I access the confidential database for this project?",
        "The research clearly shows that this approach is the most effective solution.",
        "I'm experiencing severe symptoms and need professional advice.",
        "What are the current market trends for cryptocurrency investments?",
        "This policy is wrong, but I understand the reasoning behind it.",
    ]

    print("Testing Unified Semantic Signal Framework...")
    print("=" * 60)

    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text}")
        print("-" * 50)

        signals = estimator.estimate_semantic_signals(text)

        print("Semantic signals:")
        for signal_name, value in signals.to_dict().items():
            if value > 0.1:
                print(f"  {signal_name}: {value:.3f}")

        if i > 1:
            prev_text = test_texts[i - 2]
            similarity = estimator.similarity_engine.compute_semantic_similarity(text, prev_text)
            print(f"  Similarity to previous: {similarity:.3f}")

    print("\n" + "=" * 60)
    print("Semantic framework test completed successfully!")


if __name__ == "__main__":
    test_semantic_framework()
