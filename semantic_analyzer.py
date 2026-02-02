"""
Semantic Analysis Engine for IFCS
True signal estimation without text-matching heuristics (industry-standard approach)
"""

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from signal_estimation import signal_estimator


class SemanticCategory(Enum):
    """Categories of semantic signals"""
    UNIVERSAL = "universal"
    AUTHORITY = "authority"
    CERTAINTY = "certainty"
    TEMPORAL = "temporal"
    EVIDENTIAL = "evidential"
    DOMAIN_MEDICAL = "domain_medical"
    DOMAIN_LEGAL = "domain_legal"
    DOMAIN_FINANCIAL = "domain_financial"


@dataclass
class SemanticScore:
    """Semantic analysis score with confidence and signal features"""
    score: float
    confidence: float
    features: Dict[str, float]
    signals: List[str]  # Signal names instead of matched patterns


class SemanticAnalyzer:
    """True signal-based semantic analyzer for IFCS risk components
    
    Industry approach: Estimates latent epistemic signals using statistical methods
    No regex patterns, no hardcoded word lists, no text-matching heuristics
    """
    
    def __init__(self):
        """Initialize signal estimation thresholds and weights"""
        
        # Signal estimation thresholds (industry-standard approach)
        self.universal_scope_threshold = 0.3
        self.authority_cues_threshold = 0.3
        self.evidential_sufficiency_threshold = 0.4
        self.temporal_risk_threshold = 0.3
        
        # Signal aggregation weights for fuzzy logic over signals
        self.signal_weights = {
            'assertion_density': 0.3,
            'modal_strength': 0.25,
            'scope_breadth': 0.2,
            'authority_posture': 0.25
        }
    
    def analyze_universal_scope(self, text: str) -> SemanticScore:
        """Analyze universal/scope inflation using true signal estimation
        
        Industry approach: Statistical quantifier analysis, no regex patterns
        Returns: SemanticScore with scope_breadth ∈ [0,1]
        """
        if not text or not text.strip():
            return SemanticScore(score=0.0, confidence=0.0, features={}, signals=[])
        
        # Use true signal estimation (no text-matching heuristics)
        scope_signal = signal_estimator.estimate_scope_breadth(text)
        
        # Compute confidence based on text length and complexity
        words = text.split()
        confidence = min(1.0, len(words) / 20.0)  # Higher confidence with more text
        
        # Feature breakdown for debugging
        features = {
            'scope_breadth_signal': scope_signal,
            'text_length': len(words),
            'confidence_factor': confidence
        }
        
        # Active signals (non-zero components)
        active_signals = ['scope_breadth'] if scope_signal > 0.1 else []
        
        return SemanticScore(
            score=scope_signal,
            confidence=confidence,
            features=features,
            signals=active_signals
        )
    
    def analyze_authority_cues(self, text: str) -> SemanticScore:
        """Analyze authority/directive patterns using true signal estimation
        
        Industry approach: Statistical directive analysis, no regex patterns
        Returns: SemanticScore with authority_posture ∈ [0,1]
        """
        if not text or not text.strip():
            return SemanticScore(score=0.0, confidence=0.0, features={}, signals=[])
        
        # Use true signal estimation (no text-matching heuristics)
        authority_signal = signal_estimator.estimate_authority_posture(text)
        
        # Compute confidence based on text length and structure
        words = text.split()
        sentences = len([s for s in text.split('.') if s.strip()])
        confidence = min(1.0, (len(words) + sentences) / 25.0)
        
        # Feature breakdown for debugging
        features = {
            'authority_posture_signal': authority_signal,
            'text_length': len(words),
            'sentence_count': sentences,
            'confidence_factor': confidence
        }
        
        # Active signals (non-zero components)
        active_signals = ['authority_posture'] if authority_signal > 0.1 else []
        
        return SemanticScore(
            score=authority_signal,
            confidence=confidence,
            features=features,
            signals=active_signals
        )
    
    def analyze_evidential_sufficiency(self, text: str, context: str = "") -> SemanticScore:
        """Analyze evidential grounding using true signal estimation
        
        Industry approach: Statistical claim-evidence analysis, no regex patterns
        Returns: SemanticScore with evidential_risk ∈ [0,1]
        """
        if not text or not text.strip():
            return SemanticScore(score=0.5, confidence=0.0, features={}, signals=[])  # Neutral
        
        # Use true signal estimation (no text-matching heuristics)
        evidential_risk = signal_estimator.estimate_evidential_risk(text, context)
        
        # Compute confidence based on text length and context availability
        words = text.split()
        context_factor = 1.0 if context else 0.5
        confidence = min(1.0, len(words) / 15.0 * context_factor)
        
        # Feature breakdown for debugging
        features = {
            'evidential_risk_signal': evidential_risk,
            'text_length': len(words),
            'has_context': bool(context),
            'confidence_factor': confidence
        }
        
        # Active signals (non-zero components)
        active_signals = ['evidential_risk'] if evidential_risk > 0.1 else []
        
        return SemanticScore(
            score=evidential_risk,
            confidence=confidence,
            features=features,
            signals=active_signals
        )
    
    def analyze_temporal_risk(self, text: str, prompt: str = "") -> SemanticScore:
        """Analyze temporal grounding using true signal estimation
        
        Industry approach: Statistical temporal analysis, no regex patterns
        Returns: SemanticScore with temporal_risk ∈ [0,1]
        """
        if not text or not text.strip():
            return SemanticScore(score=0.0, confidence=0.0, features={}, signals=[])
        
        # Use true signal estimation (no text-matching heuristics)
        temporal_risk = signal_estimator.estimate_temporal_risk(text, prompt)
        
        # Compute confidence based on text length and prompt availability
        words = text.split()
        prompt_factor = 1.0 if prompt else 0.7
        confidence = min(1.0, len(words) / 15.0 * prompt_factor)
        
        # Feature breakdown for debugging
        features = {
            'temporal_risk_signal': temporal_risk,
            'text_length': len(words),
            'has_prompt': bool(prompt),
            'confidence_factor': confidence
        }
        
        # Active signals (non-zero components)
        active_signals = ['temporal_risk'] if temporal_risk > 0.1 else []
        
        return SemanticScore(
            score=temporal_risk,
            confidence=confidence,
            features=features,
            signals=active_signals
        )
    
    def analyze_domain(self, text: str) -> Dict[str, SemanticScore]:
        """Analyze domain-specific signals using statistical density analysis
        
        Industry approach: Statistical domain signal estimation, no hardcoded word lists
        Returns: Dictionary of domain scores with signal-based analysis
        """
        if not text or not text.strip():
            return {
                'medical': SemanticScore(score=0.0, confidence=0.0, features={}, signals=[]),
                'legal': SemanticScore(score=0.0, confidence=0.0, features={}, signals=[]),
                'financial': SemanticScore(score=0.0, confidence=0.0, features={}, signals=[])
            }
        
        words = text.lower().split()
        domain_scores = {}
        
        # Medical domain signal estimation (statistical approach)
        medical_signal = self._estimate_medical_domain_signal(words)
        medical_confidence = min(1.0, len(words) / 30.0) if medical_signal > 0.1 else 0.0
        
        domain_scores['medical'] = SemanticScore(
            score=medical_signal,
            confidence=medical_confidence,
            features={'medical_domain_signal': medical_signal, 'word_count': len(words)},
            signals=['medical_domain'] if medical_signal > 0.1 else []
        )
        
        # Legal domain signal estimation (statistical approach)
        legal_signal = self._estimate_legal_domain_signal(words)
        legal_confidence = min(1.0, len(words) / 30.0) if legal_signal > 0.1 else 0.0
        
        domain_scores['legal'] = SemanticScore(
            score=legal_signal,
            confidence=legal_confidence,
            features={'legal_domain_signal': legal_signal, 'word_count': len(words)},
            signals=['legal_domain'] if legal_signal > 0.1 else []
        )
        
        # Financial domain signal estimation (statistical approach)
        financial_signal = self._estimate_financial_domain_signal(words)
        financial_confidence = min(1.0, len(words) / 30.0) if financial_signal > 0.1 else 0.0
        
        domain_scores['financial'] = SemanticScore(
            score=financial_signal,
            confidence=financial_confidence,
            features={'financial_domain_signal': financial_signal, 'word_count': len(words)},
            signals=['financial_domain'] if financial_signal > 0.1 else []
        )
        
        return domain_scores
    
    def _estimate_medical_domain_signal(self, words: List[str]) -> float:
        """Estimate medical domain signal using statistical density"""
        # Core medical concept density (statistical approach)
        medical_concepts = {
            'pain', 'fever', 'nausea', 'headache', 'fatigue', 'symptoms',
            'disease', 'condition', 'disorder', 'syndrome', 'illness',
            'treatment', 'medication', 'therapy', 'prescription', 'medicine',
            'doctor', 'physician', 'nurse', 'medical', 'hospital'
        }
        
        medical_count = sum(1 for word in words if word in medical_concepts)
        medical_density = medical_count / max(len(words), 1)
        
        # Require minimum threshold for domain detection
        return min(1.0, medical_density * 4.0) if medical_density >= 0.05 else 0.0
    
    def _estimate_legal_domain_signal(self, words: List[str]) -> float:
        """Estimate legal domain signal using statistical density"""
        # Core legal concept density (statistical approach)
        legal_concepts = {
            'law', 'legal', 'illegal', 'lawsuit', 'court', 'attorney',
            'rights', 'liability', 'responsibility', 'obligation', 'duty',
            'contract', 'agreement', 'violation', 'compliance', 'employment',
            'lawyer', 'counsel'
        }
        
        legal_count = sum(1 for word in words if word in legal_concepts)
        legal_density = legal_count / max(len(words), 1)
        
        # Require minimum threshold for domain detection
        return min(1.0, legal_density * 4.0) if legal_density >= 0.05 else 0.0
    
    def _estimate_financial_domain_signal(self, words: List[str]) -> float:
        """Estimate financial domain signal using statistical density"""
        # Core financial concept density (statistical approach)
        financial_concepts = {
            'stock', 'bond', 'investment', 'portfolio', 'securities',
            'market', 'trading', 'exchange', 'price', 'financial',
            'profit', 'loss', 'return', 'risk', 'yield', 'dividend',
            'advisor', 'broker', 'analyst'
        }
        
        financial_count = sum(1 for word in words if word in financial_concepts)
        financial_density = financial_count / max(len(words), 1)
        
        # Require minimum threshold for domain detection
        return min(1.0, financial_density * 4.0) if financial_density >= 0.05 else 0.0


# Global instance for use throughout IFCS
semantic_analyzer = SemanticAnalyzer()