"""
Semantic Analysis Engine for IFCS
Replaces brittle text matching with robust semantic pattern detection
"""

import re
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class SemanticCategory(Enum):
    """Categories of semantic patterns"""
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
    """Semantic analysis score with confidence and features"""
    score: float
    confidence: float
    features: Dict[str, float]
    matched_patterns: List[str]


class SemanticAnalyzer:
    """Advanced semantic analyzer for IFCS risk components"""
    
    def __init__(self):
        """Initialize semantic pattern definitions"""
        
        # Universal/scope inflation patterns
        self.universal_patterns = {
            'absolute_quantifiers': ['all', 'every', 'always', 'never', 'none', 'everything', 'nothing'],
            'definitive_articles': ['the answer', 'the solution', 'the way', 'the method', 'the approach'],
            'certainty_adverbs': ['definitely', 'certainly', 'absolutely', 'undoubtedly', 'clearly', 'obviously'],
            'exclusivity_markers': ['only', 'sole', 'single', 'exclusive', 'unique'],
            'universality_phrases': ['without exception', 'invariably', 'in all cases', 'universally']
        }
        
        # Authority/directive patterns
        self.authority_patterns = {
            'modal_verbs': ['must', 'should', 'ought', 'need', 'have to', 'required to'],
            'imperative_verbs': ['do', 'use', 'implement', 'follow', 'apply', 'ensure'],
            'recommendation_verbs': ['recommend', 'suggest', 'advise', 'propose', 'urge'],
            'obligation_phrases': ['you must', 'you should', 'you need to', 'you have to'],
            'authority_claims': ['the best', 'the right way', 'the correct', 'essential', 'critical', 'imperative']
        }
        
        # Evidential patterns
        self.evidential_patterns = {
            'evidence_markers': ['evidence', 'research', 'studies', 'data', 'findings', 'proof'],
            'source_references': ['according to', 'based on', 'research shows', 'studies indicate'],
            'uncertainty_markers': ['unclear', 'unknown', 'uncertain', 'ambiguous', 'disputed'],
            'qualification_markers': ['may', 'might', 'could', 'possibly', 'potentially', 'appears to']
        }
        
        # Temporal patterns
        self.temporal_patterns = {
            'current_markers': ['current', 'now', 'today', 'present', 'currently', 'at present'],
            'recent_markers': ['recent', 'lately', 'recently', 'just', 'latest', 'new'],
            'future_markers': ['will', 'future', 'upcoming', 'next', 'soon', 'eventually'],
            'date_patterns': [r'\b202[4-9]\b', r'\b203\d\b', r'\bthis year\b', r'\bnext year\b']
        }
        
        # Domain-specific patterns
        self.domain_patterns = {
            'medical': {
                'symptoms': ['pain', 'fever', 'nausea', 'nauseous', 'headache', 'fatigue', 'symptoms', 'ache', 'hurt', 'sick'],
                'conditions': ['disease', 'condition', 'disorder', 'syndrome', 'illness', 'infection', 'injury'],
                'treatments': ['treatment', 'medication', 'therapy', 'surgery', 'prescription', 'medicine', 'drug'],
                'professionals': ['doctor', 'physician', 'nurse', 'specialist', 'medical', 'hospital', 'clinic']
            },
            'legal': {
                'legal_terms': ['law', 'legal', 'legally', 'illegal', 'lawsuit', 'court', 'judge', 'attorney', 'lawyer'],
                'rights': ['rights', 'liability', 'responsibility', 'obligation', 'duty', 'claim', 'compensation'],
                'processes': ['sue', 'contract', 'agreement', 'violation', 'compliance', 'terminate', 'employment'],
                'professionals': ['lawyer', 'attorney', 'counsel', 'legal advisor', 'employer', 'employee']
            },
            'financial': {
                'instruments': ['stock', 'bond', 'investment', 'portfolio', 'securities', 'shares', 'equity'],
                'markets': ['market', 'trading', 'exchange', 'price', 'value', 'financial', 'money'],
                'concepts': ['profit', 'loss', 'return', 'risk', 'yield', 'dividend', 'interest', 'loan'],
                'professionals': ['advisor', 'broker', 'analyst', 'financial planner', 'investor']
            }
        }
    
    def analyze_universal_scope(self, text: str) -> SemanticScore:
        """Analyze universal/scope inflation patterns"""
        text_lower = text.lower()
        words = text_lower.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        features = {}
        matched_patterns = []
        total_score = 0.0
        
        # Analyze each pattern category
        for category, patterns in self.universal_patterns.items():
            count = 0
            for pattern in patterns:
                if ' ' in pattern:  # Multi-word phrase
                    if pattern in text_lower:
                        count += 1
                        matched_patterns.append(pattern)
                else:  # Single word
                    count += sum(1 for word in words if word == pattern)
                    if count > 0:
                        matched_patterns.append(pattern)
            
            features[category] = count
            
            # Weight different categories
            if category == 'absolute_quantifiers':
                total_score += count * 0.3
            elif category == 'definitive_articles':
                total_score += count * 0.4
            elif category == 'certainty_adverbs':
                total_score += count * 0.2
            elif category == 'exclusivity_markers':
                total_score += count * 0.3
            elif category == 'universality_phrases':
                total_score += count * 0.5
        
        # Normalize by sentence count
        if sentences:
            normalized_score = min(1.0, total_score / len(sentences))
        else:
            normalized_score = 0.0
        
        # Calculate confidence based on pattern diversity
        confidence = min(1.0, len(set(matched_patterns)) / 5.0)
        
        return SemanticScore(
            score=normalized_score,
            confidence=confidence,
            features=features,
            matched_patterns=matched_patterns
        )
    
    def analyze_authority_cues(self, text: str) -> SemanticScore:
        """Analyze authority/directive patterns"""
        text_lower = text.lower()
        words = text_lower.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        features = {}
        matched_patterns = []
        total_score = 0.0
        
        # Analyze each pattern category
        for category, patterns in self.authority_patterns.items():
            count = 0
            for pattern in patterns:
                if ' ' in pattern:  # Multi-word phrase
                    if pattern in text_lower:
                        count += 1
                        matched_patterns.append(pattern)
                else:  # Single word
                    count += sum(1 for word in words if word == pattern)
                    if count > 0:
                        matched_patterns.append(pattern)
            
            features[category] = count
            
            # Weight different categories
            if category == 'modal_verbs':
                total_score += count * 0.3
            elif category == 'imperative_verbs':
                total_score += count * 0.2
            elif category == 'recommendation_verbs':
                total_score += count * 0.4
            elif category == 'obligation_phrases':
                total_score += count * 0.5
            elif category == 'authority_claims':
                total_score += count * 0.4
        
        # Check for imperative mood (sentences starting with verbs)
        imperative_count = 0
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if sentence and any(sentence.startswith(verb) for verb in ['use', 'do', 'try', 'follow', 'implement']):
                imperative_count += 1
        
        features['imperative_mood'] = imperative_count
        total_score += imperative_count * 0.3
        
        # Normalize by sentence count
        if sentences:
            normalized_score = min(1.0, total_score / len(sentences))
        else:
            normalized_score = 0.0
        
        # Calculate confidence
        confidence = min(1.0, len(set(matched_patterns)) / 5.0)
        
        return SemanticScore(
            score=normalized_score,
            confidence=confidence,
            features=features,
            matched_patterns=matched_patterns
        )
    
    def analyze_evidential_sufficiency(self, text: str, context: str = "") -> SemanticScore:
        """Analyze evidential grounding patterns"""
        text_lower = text.lower()
        context_lower = context.lower() if context else ""
        words = text_lower.split()
        
        features = {}
        matched_patterns = []
        evidential_score = 0.0
        
        # Check for evidence markers
        evidence_count = sum(1 for word in words 
                           if word in self.evidential_patterns['evidence_markers'])
        features['evidence_markers'] = evidence_count
        evidential_score += evidence_count * 0.3
        
        # Check for source references
        source_count = sum(1 for phrase in self.evidential_patterns['source_references']
                          if phrase in text_lower)
        features['source_references'] = source_count
        evidential_score += source_count * 0.4
        
        # Check for uncertainty markers (reduce evidential confidence)
        uncertainty_count = sum(1 for word in words
                              if word in self.evidential_patterns['uncertainty_markers'])
        features['uncertainty_markers'] = uncertainty_count
        
        # Check for qualification markers
        qualification_count = sum(1 for word in words
                                if word in self.evidential_patterns['qualification_markers'])
        features['qualification_markers'] = qualification_count
        
        # Context overlap analysis (if context provided)
        context_overlap = 0.0
        if context:
            context_words = set(re.findall(r'\b[a-z]{4,}\b', context_lower))
            text_words = set(re.findall(r'\b[a-z]{4,}\b', text_lower))
            if text_words:
                context_overlap = len(context_words & text_words) / len(text_words)
        
        features['context_overlap'] = context_overlap
        evidential_score += context_overlap * 0.5
        
        # Penalize uncertainty and qualification
        evidential_score -= (uncertainty_count + qualification_count) * 0.1
        
        # Invert score (higher uncertainty = higher evidential risk)
        evidential_risk = max(0.0, 1.0 - evidential_score)
        
        confidence = min(1.0, (evidence_count + source_count + 1) / 5.0)
        
        return SemanticScore(
            score=evidential_risk,
            confidence=confidence,
            features=features,
            matched_patterns=matched_patterns
        )
    
    def analyze_temporal_risk(self, text: str, prompt: str = "") -> SemanticScore:
        """Analyze temporal grounding patterns"""
        combined_text = f"{prompt} {text}".lower()
        words = combined_text.split()
        
        features = {}
        matched_patterns = []
        temporal_score = 0.0
        
        # Check for current/present markers
        current_count = sum(1 for word in words
                          if word in self.temporal_patterns['current_markers'])
        features['current_markers'] = current_count
        temporal_score += current_count * 0.4
        
        # Check for recent markers
        recent_count = sum(1 for word in words
                         if word in self.temporal_patterns['recent_markers'])
        features['recent_markers'] = recent_count
        temporal_score += recent_count * 0.3
        
        # Check for future markers
        future_count = sum(1 for word in words
                         if word in self.temporal_patterns['future_markers'])
        features['future_markers'] = future_count
        temporal_score += future_count * 0.2
        
        # Check for date patterns
        date_matches = 0
        for pattern in self.temporal_patterns['date_patterns']:
            if re.search(pattern, combined_text):
                date_matches += 1
                matched_patterns.append(pattern)
        
        features['date_patterns'] = date_matches
        temporal_score += date_matches * 0.5
        
        # Normalize and cap
        normalized_score = min(1.0, temporal_score / 3.0)
        
        confidence = min(1.0, (current_count + recent_count + date_matches + 1) / 5.0)
        
        return SemanticScore(
            score=normalized_score,
            confidence=confidence,
            features=features,
            matched_patterns=matched_patterns
        )
    
    def analyze_domain(self, text: str) -> Dict[str, SemanticScore]:
        """Analyze domain-specific patterns"""
        text_lower = text.lower()
        words = text_lower.split()
        
        domain_scores = {}
        
        for domain, categories in self.domain_patterns.items():
            features = {}
            matched_patterns = []
            total_score = 0.0
            
            for category, patterns in categories.items():
                count = sum(1 for word in words if word in patterns)
                features[category] = count
                total_score += count
                
                if count > 0:
                    matched_patterns.extend([p for p in patterns if p in text_lower])
            
            # Require higher threshold for domain detection (C6 compliance)
            if total_score >= 2:  # Require at least 2 domain-specific terms
                normalized_score = min(1.0, total_score / 8.0)  # More conservative normalization
                confidence = min(1.0, len(set(matched_patterns)) / 4.0)  # Higher confidence threshold
            else:
                normalized_score = 0.0
                confidence = 0.0
            
            domain_scores[domain] = SemanticScore(
                score=normalized_score,
                confidence=confidence,
                features=features,
                matched_patterns=matched_patterns
            )
        
        return domain_scores


# Global instance for use throughout IFCS
semantic_analyzer = SemanticAnalyzer()