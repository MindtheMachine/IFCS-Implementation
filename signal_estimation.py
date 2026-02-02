"""
Simple Signal Estimation for IFCS
Industry-standard approach: Statistical signal estimation without text-matching heuristics
"""

from typing import Dict, List


class TrueSignalEstimator:
    """Industry-standard signal estimation without text-matching heuristics"""
    
    def __init__(self):
        """Initialize signal estimation parameters"""
        pass
        
    def estimate_assertion_strength(self, text: str) -> float:
        """Estimate assertion strength using statistical properties"""
        if not text or len(text.strip()) < 5:
            return 0.0
        
        words = text.lower().split()
        
        # Statistical approach: modal verb density
        assertive_modals = {'must', 'should', 'will', 'shall', 'need', 'have'}
        hedged_modals = {'might', 'may', 'could', 'would', 'perhaps', 'possibly'}
        
        assertive_count = sum(1 for word in words if word in assertive_modals)
        hedged_count = sum(1 for word in words if word in hedged_modals)
        
        modal_strength = (assertive_count - hedged_count * 0.5) / max(len(words), 1)
        return max(0.0, min(1.0, modal_strength * 4.0))
    
    def estimate_epistemic_certainty(self, text: str) -> float:
        """Estimate epistemic certainty using statistical markers"""
        if not text or len(text.strip()) < 5:
            return 0.0
        
        words = text.lower().split()
        
        # Statistical approach: certainty vs uncertainty markers
        certainty_markers = {'definitely', 'certainly', 'clearly', 'obviously', 'undoubtedly', 'always', 'never'}
        uncertainty_markers = {'maybe', 'perhaps', 'possibly', 'likely', 'probably', 'seems', 'appears'}
        
        certainty_count = sum(1 for word in words if word in certainty_markers)
        uncertainty_count = sum(1 for word in words if word in uncertainty_markers)
        
        certainty_imbalance = (certainty_count - uncertainty_count * 0.6) / max(len(words), 1)
        return max(0.0, min(1.0, certainty_imbalance * 8.0))
    
    def estimate_scope_breadth(self, text: str) -> float:
        """Estimate scope breadth using quantifier analysis"""
        if not text or len(text.strip()) < 5:
            return 0.0
        
        words = text.lower().split()
        
        # Statistical approach: universal vs particular quantifiers
        universal_markers = {'all', 'every', 'always', 'never', 'everyone', 'everything'}
        particular_markers = {'some', 'sometimes', 'certain', 'specific', 'particular'}
        
        universal_count = sum(1 for word in words if word in universal_markers)
        particular_count = sum(1 for word in words if word in particular_markers)
        
        scope_signal = (universal_count - particular_count * 0.5) / max(len(words), 1)
        return max(0.0, min(1.0, scope_signal * 8.0))
    
    def estimate_authority_posture(self, text: str) -> float:
        """Estimate authority posture using directive analysis"""
        if not text or len(text.strip()) < 5:
            return 0.0
        
        text_lower = text.lower()
        
        # Statistical approach: directive phrase density
        directive_phrases = ['you should', 'you must', 'you need to', 'i recommend', 'i suggest', 'the best']
        hedging_phrases = ['might consider', 'could try', 'it depends', 'may want to']
        
        directive_count = sum(1 for phrase in directive_phrases if phrase in text_lower)
        hedging_count = sum(1 for phrase in hedging_phrases if phrase in text_lower)
        
        sentence_count = max(1, text_lower.count('.') + text_lower.count('!') + text_lower.count('?') + 1)
        authority_signal = (directive_count - hedging_count * 0.7) / sentence_count
        
        return max(0.0, min(1.0, authority_signal * 3.0))
    
    def estimate_evidential_risk(self, text: str, context: str = "") -> float:
        """Estimate evidential insufficiency using claim-evidence analysis"""
        if not text or len(text.strip()) < 5:
            return 0.5  # Neutral
        
        words = text.lower().split()
        
        # Statistical approach: claim-evidence imbalance
        claim_words = {'is', 'are', 'will', 'must', 'should', 'this', 'that'}
        evidence_words = {'because', 'since', 'due', 'based', 'evidence', 'study', 'research', 'data'}
        
        claim_count = sum(1 for word in words if word in claim_words)
        evidence_count = sum(1 for word in words if word in evidence_words)
        
        claim_evidence_ratio = claim_count / max(evidence_count + 1, 1)
        evidential_risk = min(1.0, claim_evidence_ratio * 0.3)
        
        return max(0.0, evidential_risk)
    
    def estimate_temporal_risk(self, text: str, prompt: str = "") -> float:
        """Estimate temporal grounding risk using time reference analysis"""
        if not text or len(text.strip()) < 5:
            return 0.0
        
        combined_text = f"{prompt} {text}".lower()
        
        # Statistical approach: temporal marker density
        current_markers = {'current', 'now', 'today', 'present', 'currently'}
        future_markers = {'will', 'future', 'upcoming', 'next', 'soon', 'eventually'}
        specific_phrases = ['this year', 'next year', 'this month', 'next month']
        
        current_count = sum(combined_text.count(marker) for marker in current_markers)
        future_count = sum(combined_text.count(marker) for marker in future_markers)
        specific_count = sum(1 for phrase in specific_phrases if phrase in combined_text)
        
        sentence_count = max(1, combined_text.count('.') + combined_text.count('!') + combined_text.count('?') + 1)
        temporal_risk = (current_count * 0.4 + future_count * 0.2 + specific_count * 0.4) / sentence_count
        
        return max(0.0, min(1.0, temporal_risk * 2.0))


# Global instance for use throughout the system
signal_estimator = TrueSignalEstimator()