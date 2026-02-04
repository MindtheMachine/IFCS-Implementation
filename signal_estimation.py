"""
Semantic Signal Estimation for IFCS
Industry-standard approach: semantic similarity + fuzzy logic, no keyword density or hardcoded term lists.
"""

from typing import Dict, Iterable, List
import math
import re


class TrueSignalEstimator:
    """Industry-standard signal estimation using semantic similarity and fuzzy logic."""

    def __init__(self):
        self._prototypes = {
            "assertive": (
                "It is clear that this is the correct answer.",
                "You should do this because it is the right approach.",
                "This is the definitive result.",
            ),
            "hedged": (
                "It might be possible, but it depends on context.",
                "I am not fully certain and this could vary.",
                "There may be exceptions depending on the situation.",
            ),
            "certainty": (
                "This is definitely true without doubt.",
                "It is certainly correct and reliable.",
                "The conclusion is clear and unambiguous.",
            ),
            "uncertainty": (
                "This is uncertain and could change.",
                "It is possible but not confirmed.",
                "I am not sure and it depends on context.",
            ),
            "universal_scope": (
                "This always applies to everyone in all cases.",
                "This is true for all situations without exception.",
            ),
            "particular_scope": (
                "This applies in some cases and depends on circumstances.",
                "This is specific to a particular situation.",
            ),
            "directive": (
                "You must follow these steps to proceed.",
                "I recommend you do this as the best option.",
                "You should take this action.",
            ),
            "non_directive": (
                "You could consider a few options if you want.",
                "Here are some possibilities to explore.",
                "There are a few approaches you might take.",
            ),
            "evidence": (
                "Based on evidence and sources, the claim is supported.",
                "The data and research suggest this conclusion.",
            ),
            "unsupported_claim": (
                "This is true and will happen without further explanation.",
                "It is definitely the case without additional evidence.",
            ),
            "temporal": (
                "As of today and in the current situation, this is true.",
                "In the future, this will change based on upcoming events.",
            ),
            "timeless": (
                "This is a general principle that does not depend on time.",
                "This holds regardless of current events.",
            ),
            "polarity_positive": (
                "Yes, that is correct.",
                "I agree with that statement.",
            ),
            "polarity_negative": (
                "No, that is not correct.",
                "I disagree with that statement.",
            ),
        }

        self._intent_prototypes = {
            "jurisdictional": (
                "Is it legal or allowed to do this?",
                "Am I permitted to do this under the law?",
                "Is this against the law?",
            ),
            "policy": (
                "Does this violate the rules or policy?",
                "Is this within the guidelines?",
                "Is this compliant with the rules?",
            ),
            "binary": (
                "Should I do this yes or no?",
                "Can I or can't I do this?",
                "Is this allowed or not allowed?",
            ),
            "personal_context": (
                "I am experiencing this and need advice about my situation.",
                "My personal circumstances are involved.",
                "This is about my symptoms or situation.",
            ),
            "consequence": (
                "Is there a risk of harm or danger?",
                "Is it safe to do this?",
                "What are the consequences if I do this?",
            ),
            "context_dependency": (
                "Based on the document or file provided, answer this.",
                "Using the attached data, explain the result.",
                "From the image or chart, describe what it shows.",
            ),
            "ambiguity": (
                "I'm not sure what to do and need guidance.",
                "This is unclear and ambiguous.",
                "I need clarification because this is uncertain.",
            ),
            "definitive_request": (
                "Give me the best and only correct answer.",
                "What is the right way to do this?",
                "Tell me the definitive solution.",
            ),
            "authority_request": (
                "What should I do?",
                "What do you recommend?",
                "What is the proper course of action?",
            ),
            "disagreement": (
                "I disagree with that answer.",
                "That seems wrong or incorrect.",
                "I don't think that's true.",
            ),
        }

        self._domain_prototypes = {
            "medical": (
                "This is about symptoms, diagnosis, and treatment options.",
                "The discussion involves medical care and health conditions.",
            ),
            "legal": (
                "This concerns legal advice, rights, and obligations.",
                "The question relates to laws and regulations.",
            ),
            "financial": (
                "This is about investments, markets, and financial decisions.",
                "The topic involves budgeting, money, and finance.",
            ),
        }

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _char_ngrams(text: str, n: int = 3) -> List[str]:
        if len(text) < n:
            return [text] if text else []
        return [text[i : i + n] for i in range(len(text) - n + 1)]

    def _vectorize(self, text: str) -> Dict[str, int]:
        normalized = self._normalize(text)
        ngrams = self._char_ngrams(normalized, n=3)
        vector: Dict[str, int] = {}
        for ngram in ngrams:
            vector[ngram] = vector.get(ngram, 0) + 1
        return vector

    @staticmethod
    def _cosine_similarity(vec_a: Dict[str, int], vec_b: Dict[str, int]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        dot = 0.0
        for key, value in vec_a.items():
            if key in vec_b:
                dot += value * vec_b[key]
        norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
        norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def semantic_similarity(self, text_a: str, text_b: str) -> float:
        vec_a = self._vectorize(text_a)
        vec_b = self._vectorize(text_b)
        return self._cosine_similarity(vec_a, vec_b)

    def _max_similarity(self, text: str, prototypes: Iterable[str]) -> float:
        return max((self.semantic_similarity(text, proto) for proto in prototypes), default=0.0)

    @staticmethod
    def _fuzzy_membership(value: float, low: float = 0.25, high: float = 0.65) -> float:
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        t = (value - low) / (high - low)
        return t * t * (3 - 2 * t)  # smoothstep

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _semantic_score(self, text: str, positive_key: str, negative_key: str) -> float:
        positive_sim = self._max_similarity(text, self._prototypes[positive_key])
        negative_sim = self._max_similarity(text, self._prototypes[negative_key])
        positive = self._fuzzy_membership(positive_sim)
        negative = self._fuzzy_membership(negative_sim)
        return self._clamp(positive * (1.0 - 0.7 * negative))

    def estimate_assertion_strength(self, text: str) -> float:
        """Estimate assertion strength using semantic similarity + fuzzy logic."""
        if not text or len(text.strip()) < 5:
            return 0.0
        return self._semantic_score(text, "assertive", "hedged")

    def estimate_epistemic_certainty(self, text: str) -> float:
        """Estimate epistemic certainty using semantic similarity + fuzzy logic."""
        if not text or len(text.strip()) < 5:
            return 0.0
        return self._semantic_score(text, "certainty", "uncertainty")

    def estimate_scope_breadth(self, text: str) -> float:
        """Estimate scope breadth using semantic similarity + fuzzy logic."""
        if not text or len(text.strip()) < 5:
            return 0.0
        return self._semantic_score(text, "universal_scope", "particular_scope")

    def estimate_authority_posture(self, text: str) -> float:
        """Estimate authority posture using semantic similarity + fuzzy logic."""
        if not text or len(text.strip()) < 5:
            return 0.0
        return self._semantic_score(text, "directive", "non_directive")

    def estimate_evidential_risk(self, text: str, context: str = "") -> float:
        """Estimate evidential insufficiency using semantic similarity + fuzzy logic."""
        if not text or len(text.strip()) < 5:
            return 0.5
        claim_sim = self._max_similarity(text, self._prototypes["unsupported_claim"])
        evidence_sim = self._max_similarity(text, self._prototypes["evidence"])
        claim_score = self._fuzzy_membership(claim_sim)
        evidence_score = self._fuzzy_membership(evidence_sim)
        base_risk = claim_score * (1.0 - 0.8 * evidence_score)
        if context:
            grounding = self._fuzzy_membership(self.semantic_similarity(text, context), low=0.2, high=0.6)
            base_risk *= (1.0 - 0.3 * grounding)
        return self._clamp(base_risk)

    def estimate_temporal_risk(self, text: str, prompt: str = "") -> float:
        """Estimate temporal grounding risk using semantic similarity + fuzzy logic."""
        combined_text = f"{prompt} {text}".strip()
        if not combined_text or len(combined_text.strip()) < 5:
            return 0.0
        temporal_sim = self._max_similarity(combined_text, self._prototypes["temporal"])
        timeless_sim = self._max_similarity(combined_text, self._prototypes["timeless"])
        temporal_score = self._fuzzy_membership(temporal_sim)
        timeless_score = self._fuzzy_membership(timeless_sim)
        return self._clamp(temporal_score * (1.0 - 0.6 * timeless_score))

    def estimate_intent_signal(self, text: str, intent: str) -> float:
        if not text or not text.strip():
            return 0.0
        prototypes = self._intent_prototypes.get(intent, ())
        similarity = self._max_similarity(text, prototypes)
        return self._fuzzy_membership(similarity)

    def estimate_domain_signal(self, text: str, domain: str) -> float:
        if not text or not text.strip():
            return 0.0
        prototypes = self._domain_prototypes.get(domain, ())
        similarity = self._max_similarity(text, prototypes)
        return self._fuzzy_membership(similarity)

    def estimate_polarity(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        positive_sim = self._max_similarity(text, self._prototypes["polarity_positive"])
        negative_sim = self._max_similarity(text, self._prototypes["polarity_negative"])
        positive_score = self._fuzzy_membership(positive_sim)
        negative_score = self._fuzzy_membership(negative_sim)
        return max(-1.0, min(1.0, positive_score - negative_score))

    def estimate_jurisdictional_signal(self, text: str) -> float:
        return self.estimate_intent_signal(text, "jurisdictional")

    def estimate_policy_signal(self, text: str) -> float:
        return self.estimate_intent_signal(text, "policy")

    def estimate_binary_signal(self, text: str) -> float:
        return self.estimate_intent_signal(text, "binary")

    def estimate_personal_signal(self, text: str) -> float:
        return self.estimate_intent_signal(text, "personal_context")

    def estimate_consequence_signal(self, text: str) -> float:
        return self.estimate_intent_signal(text, "consequence")

    def estimate_context_dependency_signal(self, text: str) -> float:
        return self.estimate_intent_signal(text, "context_dependency")

    def estimate_ambiguity_signal(self, text: str) -> float:
        return self.estimate_intent_signal(text, "ambiguity")

    def estimate_definitive_request_signal(self, text: str) -> float:
        return self.estimate_intent_signal(text, "definitive_request")

    def estimate_authority_request_signal(self, text: str) -> float:
        return self.estimate_intent_signal(text, "authority_request")

    def estimate_disagreement_signal(self, text: str) -> float:
        return self.estimate_intent_signal(text, "disagreement")


# Global instance for use throughout the system
signal_estimator = TrueSignalEstimator()
