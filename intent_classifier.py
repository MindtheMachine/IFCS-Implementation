"""
Intent Classifier for Enhanced Signal Strength Computation
Implements statistical semantic analysis for prompt intent detection
"""

from typing import Dict

from signal_estimation import signal_estimator, compute_text_stats


class IntentClassifier:
    """Lightweight intent classification using statistical semantic proxies"""

    def __init__(self):
        """Initialize intent classifier (no pattern lists or keyword matching)."""
        pass

    def analyze_prompt(self, prompt: str) -> Dict[str, float]:
        """Analyze prompt intent using statistical features.

        Args:
            prompt: Input prompt to analyze

        Returns:
            Dictionary mapping signal types to intent scores [0,1]
        """
        if not prompt or len(prompt.strip()) < 3:
            return {
                "jurisdictional": 0.0,
                "policy": 0.0,
                "binary": 0.0,
                "personal_data": 0.0,
                "consequence": 0.0,
                "temporal": 0.0,
            }

        stats = compute_text_stats(prompt)

        question_ratio = stats["question_ratio"]
        length_norm = stats["length_norm"]
        list_density = stats["list_density"]
        digit_ratio = stats["digit_ratio"]
        type_token_ratio = stats["type_token_ratio"]
        exclam_ratio = stats["exclam_ratio"]
        uppercase_ratio = stats["uppercase_ratio"]

        # Statistical proxies (no keyword/pattern matching)
        jurisdictional = min(1.0, 0.7 * question_ratio + 0.3 * (1.0 - length_norm))
        policy = min(1.0, 0.5 * length_norm + 0.3 * list_density + 0.2 * digit_ratio * 5.0)
        binary = min(1.0, 0.6 * question_ratio + 0.2 * (1.0 - type_token_ratio) + 0.2 * (1.0 - length_norm))
        personal_data = min(1.0, 0.5 * (1.0 - type_token_ratio) + 0.3 * digit_ratio * 5.0 + 0.2 * (1.0 - length_norm))
        consequence = min(1.0, 0.5 * exclam_ratio + 0.3 * question_ratio + 0.2 * uppercase_ratio)
        temporal = signal_estimator.estimate_temporal_risk("", prompt)

        return {
            "jurisdictional": jurisdictional,
            "policy": policy,
            "binary": binary,
            "personal_data": personal_data,
            "consequence": consequence,
            "temporal": temporal,
        }


# Global instance for system-wide use
intent_classifier = IntentClassifier()


def test_intent_classifier():
    """Test intent classifier with sample prompts"""
    classifier = IntentClassifier()

    test_prompts = [
        "Can I access this confidential database?",
        "What are the company policies regarding remote work?",
        "Should I invest in this stock or not?",
        "I am experiencing chest pain after exercise",
        "What are the risks of using this medication?",
        "What are the current trends in AI development?",
    ]

    print("Testing Intent Classifier (statistical)...")
    for prompt in test_prompts:
        scores = classifier.analyze_prompt(prompt)
        print(f"\nPrompt: {prompt}")
        for signal_type, score in scores.items():
            if score > 0.1:
                print(f"  {signal_type}: {score:.3f}")

    return True


if __name__ == "__main__":
    test_intent_classifier()
