"""
Semantic Signal Framework for System-Wide Signal Estimation
Provides unified semantic similarity and fuzzy logic estimators for ECR, Control Probes, and domain analysis
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
import numpy as np
from enhanced_signal_estimator import enhanced_signal_estimator


@dataclass
class SemanticSignals:
    """Unified semantic signal container"""
    intent: float           # Intent clarity and directness
    domain: float          # Domain specificity and expertise level
    polarity: float        # Sentiment and stance polarity
    disagreement: float    # Conflict and contradiction signals
    confidence: float      # Epistemic certainty
    authority: float       # Authority posture and directiveness
    grounding: float       # Evidential support and factual basis
    coherence: float       # Internal consistency and logical flow
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy access"""
        return {
            'intent': self.intent,
            'domain': self.domain,
            'polarity': self.polarity,
            'disagreement': self.disagreement,
            'confidence': self.confidence,
            'authority': self.authority,
            'grounding': self.grounding,
            'coherence': self.coherence
        }
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for matrix operations"""
        return np.array([
            self.intent, self.domain, self.polarity, self.disagreement,
            self.confidence, self.authority, self.grounding, self.coherence
        ])


class SemanticSimilarityEngine:
    """Advanced semantic similarity computation using multiple methods"""
    
    def __init__(self):
        """Initialize semantic similarity engine"""
        self.word_importance_weights = {
            # High importance words
            'must': 2.0, 'should': 2.0, 'will': 2.0, 'cannot': 2.0, 'never': 2.0, 'always': 2.0,
            'definitely': 1.8, 'certainly': 1.8, 'clearly': 1.8, 'obviously': 1.8,
            'recommend': 1.6, 'suggest': 1.6, 'advise': 1.6, 'propose': 1.6,
            'important': 1.5, 'critical': 1.5, 'essential': 1.5, 'necessary': 1.5,
            # Medium importance words
            'might': 1.2, 'could': 1.2, 'possibly': 1.2, 'perhaps': 1.2, 'maybe': 1.2,
            'likely': 1.1, 'probably': 1.1, 'seems': 1.1, 'appears': 1.1,
            # Low importance (default weight = 1.0)
        }
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts using multiple methods
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score [0,1] where 1 is identical meaning
        """
        if not text1 or not text2:
            return 0.0
        
        # Method 1: Weighted word overlap
        word_overlap = self._weighted_word_overlap(text1, text2)
        
        # Method 2: Structural similarity (sentence patterns)
        structural_sim = self._structural_similarity(text1, text2)
        
        # Method 3: Semantic role similarity (subject-verb-object patterns)
        role_sim = self._semantic_role_similarity(text1, text2)
        
        # Method 4: Negation and polarity alignment
        polarity_sim = self._polarity_similarity(text1, text2)
        
        # Weighted combination
        similarity = (
            word_overlap * 0.4 +
            structural_sim * 0.25 +
            role_sim * 0.2 +
            polarity_sim * 0.15
        )
        
        return min(1.0, max(0.0, similarity))
    
    def _weighted_word_overlap(self, text1: str, text2: str) -> float:
        """Compute weighted word overlap with importance weighting"""
        words1 = self._extract_significant_words(text1)
        words2 = self._extract_significant_words(text2)
        
        if not words1 or not words2:
            return 0.0
        
        # Compute weighted intersection
        intersection_weight = 0.0
        union_weight = 0.0
        
        all_words = set(words1.keys()) | set(words2.keys())
        
        for word in all_words:
            weight = self.word_importance_weights.get(word, 1.0)
            
            count1 = words1.get(word, 0)
            count2 = words2.get(word, 0)
            
            intersection_weight += min(count1, count2) * weight
            union_weight += max(count1, count2) * weight
        
        return intersection_weight / max(union_weight, 1.0)
    
    def _extract_significant_words(self, text: str) -> Dict[str, int]:
        """Extract significant words with frequency counts"""
        # Remove punctuation and convert to lowercase
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean_text.split()
        
        # Filter out stop words and short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        significant_words = {}
        for word in words:
            if len(word) > 2 and word not in stop_words:
                significant_words[word] = significant_words.get(word, 0) + 1
        
        return significant_words
    
    def _structural_similarity(self, text1: str, text2: str) -> float:
        """Compute structural similarity based on sentence patterns"""
        # Extract sentence structures
        struct1 = self._extract_sentence_structure(text1)
        struct2 = self._extract_sentence_structure(text2)
        
        if not struct1 or not struct2:
            return 0.0
        
        # Compare structures
        common_patterns = len(set(struct1) & set(struct2))
        total_patterns = len(set(struct1) | set(struct2))
        
        return common_patterns / max(total_patterns, 1)
    
    def _extract_sentence_structure(self, text: str) -> List[str]:
        """Extract sentence structure patterns"""
        sentences = re.split(r'[.!?]+', text)
        structures = []
        
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if len(sentence) < 5:
                continue
            
            # Identify question vs statement
            if sentence.startswith(('what', 'how', 'why', 'when', 'where', 'who')):
                structures.append('wh_question')
            elif sentence.startswith(('can', 'could', 'would', 'should', 'may', 'might')):
                structures.append('modal_question')
            elif sentence.startswith(('is', 'are', 'was', 'were', 'do', 'does', 'did')):
                structures.append('yes_no_question')
            elif any(word in sentence for word in ['must', 'should', 'need to', 'have to']):
                structures.append('imperative')
            elif any(word in sentence for word in ['recommend', 'suggest', 'advise']):
                structures.append('recommendation')
            else:
                structures.append('statement')
        
        return structures
    
    def _semantic_role_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity based on semantic roles (simplified)"""
        # Extract action words (verbs) and entities (nouns)
        actions1 = self._extract_actions(text1)
        actions2 = self._extract_actions(text2)
        
        entities1 = self._extract_entities(text1)
        entities2 = self._extract_entities(text2)
        
        # Compute action similarity
        action_sim = len(set(actions1) & set(actions2)) / max(len(set(actions1) | set(actions2)), 1)
        
        # Compute entity similarity
        entity_sim = len(set(entities1) & set(entities2)) / max(len(set(entities1) | set(entities2)), 1)
        
        return (action_sim + entity_sim) / 2.0
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract action words (simplified verb detection)"""
        # Common action patterns
        action_patterns = [
            r'\b(\w+ing)\b',  # -ing verbs
            r'\b(access|use|create|delete|modify|update|install|run|execute|implement)\b',
            r'\b(recommend|suggest|advise|propose|consider|evaluate|analyze)\b',
            r'\b(is|are|was|were|will|would|could|should|must|can|may)\b'
        ]
        
        actions = []
        text_lower = text.lower()
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text_lower)
            actions.extend(matches)
        
        return actions
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entity words (simplified noun detection)"""
        # Look for capitalized words and common entity patterns
        entity_patterns = [
            r'\b[A-Z][a-z]+\b',  # Capitalized words
            r'\b(database|system|server|application|user|data|file|document)\b',
            r'\b(policy|rule|regulation|guideline|procedure|process)\b',
            r'\b(risk|safety|security|privacy|confidential|sensitive)\b'
        ]
        
        entities = []
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, text)
            entities.extend([match.lower() for match in matches])
        
        return entities
    
    def _polarity_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity based on polarity and sentiment alignment"""
        polarity1 = self._extract_polarity(text1)
        polarity2 = self._extract_polarity(text2)
        
        # If both are neutral, high similarity
        if abs(polarity1) < 0.1 and abs(polarity2) < 0.1:
            return 1.0
        
        # If same sign, compute distance
        if polarity1 * polarity2 >= 0:
            return 1.0 - abs(polarity1 - polarity2)
        else:
            # Opposite polarities
            return max(0.0, 0.5 - abs(polarity1 - polarity2) / 2.0)
    
    def _extract_polarity(self, text: str) -> float:
        """Extract polarity score [-1, 1] where -1 is negative, 1 is positive"""
        positive_words = {
            'good', 'great', 'excellent', 'best', 'better', 'safe', 'secure',
            'recommend', 'should', 'yes', 'correct', 'right', 'proper', 'effective'
        }
        
        negative_words = {
            'bad', 'poor', 'worst', 'worse', 'unsafe', 'insecure', 'dangerous',
            'avoid', 'should not', 'no', 'incorrect', 'wrong', 'improper', 'ineffective'
        }
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Check for negation patterns
        negation_patterns = ['not', 'never', 'no', "don't", "won't", "can't", "shouldn't"]
        negation_count = sum(1 for pattern in negation_patterns if pattern in text_lower)
        
        # Adjust for negations (simplified)
        if negation_count > 0:
            positive_count, negative_count = negative_count, positive_count
        
        total_words = len(text_lower.split())
        if total_words == 0:
            return 0.0
        
        polarity = (positive_count - negative_count) / max(total_words, 1)
        return max(-1.0, min(1.0, polarity * 10))  # Scale and clamp


class UnifiedSemanticSignalEstimator:
    """Unified semantic signal estimator for system-wide use"""
    
    def __init__(self):
        """Initialize unified semantic signal estimator"""
        self.similarity_engine = SemanticSimilarityEngine()
        
        # Domain classification patterns
        self.domain_patterns = {
            'technical': [
                'system', 'database', 'server', 'application', 'software', 'code',
                'implementation', 'configuration', 'deployment', 'architecture'
            ],
            'legal': [
                'law', 'legal', 'regulation', 'compliance', 'policy', 'rule',
                'jurisdiction', 'court', 'statute', 'contract'
            ],
            'medical': [
                'health', 'medical', 'doctor', 'patient', 'diagnosis', 'treatment',
                'symptom', 'medication', 'therapy', 'clinical'
            ],
            'financial': [
                'money', 'investment', 'financial', 'bank', 'loan', 'credit',
                'market', 'stock', 'portfolio', 'risk'
            ],
            'personal': [
                'i feel', 'i have', 'my', 'personal', 'private', 'confidential',
                'individual', 'specific situation', 'my case'
            ]
        }
    
    def estimate_semantic_signals(self, text: str, context: str = "") -> SemanticSignals:
        """Estimate comprehensive semantic signals for any text
        
        Args:
            text: Text to analyze
            context: Optional context for grounding analysis
            
        Returns:
            SemanticSignals object with all signal types
        """
        if not text or len(text.strip()) < 3:
            return SemanticSignals(
                intent=0.0, domain=0.0, polarity=0.0, disagreement=0.0,
                confidence=0.0, authority=0.0, grounding=0.0, coherence=0.0
            )
        
        # Use enhanced signal estimator for base signals
        intent = self._estimate_intent_clarity(text)
        domain = self._estimate_domain_specificity(text)
        polarity = self._estimate_polarity_strength(text)
        disagreement = self._estimate_disagreement_signals(text)
        
        # Use existing signal estimator for established signals
        confidence = enhanced_signal_estimator.estimate_epistemic_certainty(text)
        authority = enhanced_signal_estimator.estimate_authority_posture(text)
        grounding = self._estimate_grounding_strength(text, context)
        coherence = self._estimate_coherence_signals(text)
        
        return SemanticSignals(
            intent=intent,
            domain=domain,
            polarity=polarity,
            disagreement=disagreement,
            confidence=confidence,
            authority=authority,
            grounding=grounding,
            coherence=coherence
        )
    
    def _estimate_intent_clarity(self, text: str) -> float:
        """Estimate intent clarity and directness"""
        text_lower = text.lower()
        
        # Clear intent markers
        clear_intent_patterns = [
            r'\b(what|how|why|when|where|who)\b',  # Questions
            r'\b(please|can you|could you|would you)\b',  # Requests
            r'\b(i want|i need|i would like)\b',  # Desires
            r'\b(recommend|suggest|advise|help)\b'  # Seeking guidance
        ]
        
        clarity_score = 0.0
        for pattern in clear_intent_patterns:
            matches = len(re.findall(pattern, text_lower))
            clarity_score += matches * 0.2
        
        # Sentence structure clarity
        sentences = re.split(r'[.!?]+', text)
        clear_sentences = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 5:
                # Check for clear structure
                if (sentence.lower().startswith(('what', 'how', 'can', 'should', 'is', 'are')) or
                    any(word in sentence.lower() for word in ['recommend', 'suggest', 'help'])):
                    clear_sentences += 1
        
        structure_clarity = clear_sentences / max(len([s for s in sentences if s.strip()]), 1)
        
        # Combine scores
        intent_clarity = min(1.0, (clarity_score + structure_clarity) / 2.0)
        return intent_clarity
    
    def _estimate_domain_specificity(self, text: str) -> float:
        """Estimate domain specificity and expertise level"""
        text_lower = text.lower()
        
        domain_scores = {}
        for domain, patterns in self.domain_patterns.items():
            score = 0.0
            for pattern in patterns:
                if pattern in text_lower:
                    score += 1.0
            
            # Normalize by pattern count
            domain_scores[domain] = score / len(patterns)
        
        # Return highest domain specificity
        max_domain_score = max(domain_scores.values()) if domain_scores else 0.0
        
        # Boost for technical terminology
        technical_terms = len(re.findall(r'\b[A-Z]{2,}\b', text))  # Acronyms
        jargon_boost = min(0.3, technical_terms * 0.1)
        
        return min(1.0, max_domain_score + jargon_boost)
    
    def _estimate_polarity_strength(self, text: str) -> float:
        """Estimate polarity and stance strength"""
        polarity_value = self.similarity_engine._extract_polarity(text)
        
        # Return absolute polarity strength
        return abs(polarity_value)
    
    def _estimate_disagreement_signals(self, text: str) -> float:
        """Estimate conflict and contradiction signals"""
        text_lower = text.lower()
        
        # Disagreement markers
        disagreement_patterns = [
            r'\b(but|however|although|though|nevertheless|nonetheless)\b',
            r'\b(disagree|dispute|contest|challenge|oppose)\b',
            r'\b(wrong|incorrect|false|mistaken|error)\b',
            r'\b(not|never|no|none|neither)\b'
        ]
        
        disagreement_score = 0.0
        for pattern in disagreement_patterns:
            matches = len(re.findall(pattern, text_lower))
            disagreement_score += matches * 0.15
        
        # Internal contradiction detection
        contradiction_patterns = [
            (r'\b(yes|true|correct)\b', r'\b(no|false|incorrect)\b'),
            (r'\b(safe|secure)\b', r'\b(unsafe|dangerous|risky)\b'),
            (r'\b(recommend|should)\b', r'\b(avoid|should not)\b')
        ]
        
        contradiction_score = 0.0
        for pos_pattern, neg_pattern in contradiction_patterns:
            pos_matches = len(re.findall(pos_pattern, text_lower))
            neg_matches = len(re.findall(neg_pattern, text_lower))
            
            if pos_matches > 0 and neg_matches > 0:
                contradiction_score += 0.3
        
        return min(1.0, disagreement_score + contradiction_score)
    
    def _estimate_grounding_strength(self, text: str, context: str = "") -> float:
        """Estimate evidential grounding strength"""
        # Use existing evidential risk estimation (inverted)
        evidential_risk = enhanced_signal_estimator.estimate_evidential_risk(text, context)
        base_grounding = 1.0 - evidential_risk
        
        # Boost for explicit evidence markers
        evidence_markers = [
            'research shows', 'studies indicate', 'data suggests', 'evidence shows',
            'according to', 'based on', 'research', 'study', 'data', 'evidence',
            'source', 'reference', 'citation', 'documented', 'published'
        ]
        
        text_lower = text.lower()
        evidence_boost = 0.0
        
        for marker in evidence_markers:
            if marker in text_lower:
                evidence_boost += 0.1
        
        return min(1.0, base_grounding + evidence_boost)
    
    def _estimate_coherence_signals(self, text: str) -> float:
        """Estimate internal coherence and logical flow"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if len(sentences) < 2:
            return 0.8  # Single sentence assumed coherent
        
        # Check for logical connectors
        connectors = [
            'therefore', 'thus', 'consequently', 'as a result', 'because',
            'since', 'due to', 'first', 'second', 'finally', 'in conclusion',
            'furthermore', 'moreover', 'additionally', 'also', 'similarly'
        ]
        
        text_lower = text.lower()
        connector_count = sum(1 for connector in connectors if connector in text_lower)
        connector_score = min(0.4, connector_count * 0.1)
        
        # Check for topic consistency (simplified)
        # Extract key topics from each sentence
        sentence_topics = []
        for sentence in sentences:
            topics = self.similarity_engine._extract_significant_words(sentence)
            sentence_topics.append(set(topics.keys()))
        
        # Compute topic overlap between consecutive sentences
        topic_consistency = 0.0
        for i in range(1, len(sentence_topics)):
            overlap = len(sentence_topics[i] & sentence_topics[i-1])
            union = len(sentence_topics[i] | sentence_topics[i-1])
            if union > 0:
                topic_consistency += overlap / union
        
        if len(sentences) > 1:
            topic_consistency /= (len(sentences) - 1)
        
        # Combine scores
        coherence = min(1.0, 0.6 + connector_score + topic_consistency * 0.3)
        return coherence


# Global instance for system-wide use
unified_semantic_estimator = UnifiedSemanticSignalEstimator()


# Test function
def test_semantic_framework():
    """Test the semantic signal framework"""
    estimator = unified_semantic_estimator
    
    test_texts = [
        "Can I access the confidential database for this project?",
        "The research clearly shows that this approach is the most effective solution.",
        "I'm experiencing severe headaches and need medical advice.",
        "What are the current market trends for cryptocurrency investments?",
        "This policy is wrong, but I understand the reasoning behind it."
    ]
    
    print("Testing Unified Semantic Signal Framework...")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: {text}")
        print("-" * 50)
        
        signals = estimator.estimate_semantic_signals(text)
        
        print("Semantic signals:")
        for signal_name, value in signals.to_dict().items():
            if value > 0.1:  # Only show significant signals
                print(f"  {signal_name}: {value:.3f}")
        
        # Test semantic similarity
        if i > 1:
            prev_text = test_texts[i-2]
            similarity = estimator.similarity_engine.compute_semantic_similarity(text, prev_text)
            print(f"  Similarity to previous: {similarity:.3f}")
    
    print("\n" + "=" * 60)
    print("Semantic framework test completed successfully!")


if __name__ == "__main__":
    test_semantic_framework()