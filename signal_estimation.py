"""
Simple Signal Estimation for IFCS
Industry-standard approach: Statistical signal estimation without text-matching heuristics
"""

from typing import Dict
import math
import re


def compute_text_stats(text: str, prompt: str = "") -> Dict[str, float]:
    """Compute statistical text features with no keyword or pattern matching."""
    combined = f"{prompt} {text}".strip()
    if not combined:
        return {
            "word_count": 0.0,
            "sentence_count": 0.0,
            "avg_word_len": 0.0,
            "type_token_ratio": 0.0,
            "length_norm": 0.0,
            "digit_ratio": 0.0,
            "uppercase_ratio": 0.0,
            "question_ratio": 0.0,
            "exclam_ratio": 0.0,
            "period_ratio": 0.0,
            "comma_ratio": 0.0,
            "colon_ratio": 0.0,
            "semicolon_ratio": 0.0,
            "ellipsis_ratio": 0.0,
            "list_density": 0.0,
            "token_with_digit_ratio": 0.0,
        }

    words = re.findall(r"[A-Za-z0-9']+", combined)
    word_count = len(words)
    sentence_count = len([s for s in re.split(r"[.!?]+", combined) if s.strip()])
    sentence_count = max(1, sentence_count)

    avg_word_len = (sum(len(w) for w in words) / word_count) if word_count else 0.0
    type_token_ratio = (len({w.lower() for w in words}) / word_count) if word_count else 0.0
    length_norm = min(1.0, math.log1p(word_count) / math.log1p(200))

    char_count = len(combined)
    digit_count = sum(1 for c in combined if c.isdigit())
    alpha_count = sum(1 for c in combined if c.isalpha())
    upper_count = sum(1 for c in combined if c.isupper())

    digit_ratio = digit_count / max(char_count, 1)
    uppercase_ratio = upper_count / max(alpha_count, 1)

    question_ratio = combined.count("?") / sentence_count
    exclam_ratio = combined.count("!") / sentence_count
    period_ratio = combined.count(".") / max(char_count, 1)
    comma_ratio = combined.count(",") / max(char_count, 1)
    colon_ratio = combined.count(":") / max(char_count, 1)
    semicolon_ratio = combined.count(";") / max(char_count, 1)
    ellipsis_ratio = combined.count("...") / sentence_count

    lines = combined.splitlines() or [""]
    list_item_count = sum(1 for line in lines if re.match(r"\s*(\d+\.|[-*])\s+\S", line))
    list_density = list_item_count / max(len(lines), 1)

    tokens_with_digits = sum(1 for w in words if any(ch.isdigit() for ch in w))
    token_with_digit_ratio = tokens_with_digits / max(word_count, 1)

    return {
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "avg_word_len": float(avg_word_len),
        "type_token_ratio": float(type_token_ratio),
        "length_norm": float(length_norm),
        "digit_ratio": float(digit_ratio),
        "uppercase_ratio": float(uppercase_ratio),
        "question_ratio": float(min(1.0, question_ratio)),
        "exclam_ratio": float(min(1.0, exclam_ratio)),
        "period_ratio": float(period_ratio),
        "comma_ratio": float(comma_ratio),
        "colon_ratio": float(colon_ratio),
        "semicolon_ratio": float(semicolon_ratio),
        "ellipsis_ratio": float(min(1.0, ellipsis_ratio)),
        "list_density": float(min(1.0, list_density)),
        "token_with_digit_ratio": float(min(1.0, token_with_digit_ratio)),
    }


class TrueSignalEstimator:
    """Industry-standard signal estimation without text-matching heuristics"""

    def __init__(self):
        """Initialize signal estimation parameters"""
        pass

    def estimate_assertion_strength(self, text: str) -> float:
        """Estimate assertion strength using statistical properties"""
        if not text or len(text.strip()) < 5:
            return 0.0

        stats = compute_text_stats(text)
        score = (
            0.45 * (1.0 - stats["question_ratio"]) +
            0.25 * stats["exclam_ratio"] +
            0.20 * stats["uppercase_ratio"] +
            0.10 * stats["length_norm"]
        )
        return max(0.0, min(1.0, score))

    def estimate_epistemic_certainty(self, text: str) -> float:
        """Estimate epistemic certainty using statistical markers"""
        if not text or len(text.strip()) < 5:
            return 0.0

        stats = compute_text_stats(text)
        score = (
            0.60 * (1.0 - stats["question_ratio"]) +
            0.25 * (1.0 - stats["ellipsis_ratio"]) +
            0.15 * stats["length_norm"]
        )
        return max(0.0, min(1.0, score))

    def estimate_scope_breadth(self, text: str) -> float:
        """Estimate scope breadth using distributional features"""
        if not text or len(text.strip()) < 5:
            return 0.0

        stats = compute_text_stats(text)
        score = 0.6 * stats["length_norm"] + 0.4 * stats["type_token_ratio"]
        return max(0.0, min(1.0, score))

    def estimate_authority_posture(self, text: str) -> float:
        """Estimate authority posture using structural cues"""
        if not text or len(text.strip()) < 5:
            return 0.0

        stats = compute_text_stats(text)
        score = (
            0.45 * stats["exclam_ratio"] +
            0.30 * stats["uppercase_ratio"] +
            0.25 * (1.0 - stats["question_ratio"])
        )
        return max(0.0, min(1.0, score))

    def estimate_evidential_risk(self, text: str, context: str = "") -> float:
        """Estimate evidential insufficiency using statistical grounding proxies"""
        if not text or len(text.strip()) < 5:
            return 0.5  # Neutral

        stats = compute_text_stats(text, context)
        evidence_strength = min(
            1.0,
            stats["digit_ratio"] * 6.0 +
            stats["colon_ratio"] * 20.0 +
            stats["list_density"] * 0.5
        )
        evidential_risk = 1.0 - evidence_strength
        return max(0.0, min(1.0, evidential_risk))

    def estimate_temporal_risk(self, text: str, prompt: str = "") -> float:
        """Estimate temporal grounding risk using structural time proxies"""
        if not (text and text.strip()) and not (prompt and prompt.strip()):
            return 0.0

        stats = compute_text_stats(text, prompt)
        score = (
            stats["token_with_digit_ratio"] * 1.2 +
            stats["digit_ratio"] * 2.0 +
            stats["question_ratio"] * 0.1
        )
        return max(0.0, min(1.0, score))


# Global instance for use throughout the system
signal_estimator = TrueSignalEstimator()
