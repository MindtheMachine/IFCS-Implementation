"""
Intent Classifier for Enhanced Signal Strength Computation
Implements semantic analysis for prompt intent detection
"""

from typing import Dict, List, Set, Tuple
import re
import math
from functools import lru_cache


class IntentClassifier:
    """Lightweight intent classification for prompt analysis using semantic similarity"""
    
    def __init__(self):
        """Initialize intent classifier with pattern databases"""
        self.permission_patterns = self._load_permission_patterns()
        self.policy_patterns = self._load_policy_patterns()
        self.personal_patterns = self._load_personal_patterns()
        self.binary_patterns = self._load_binary_patterns()
        self.consequence_patterns = self._load_consequence_patterns()
        self.temporal_patterns = self._load_temporal_patterns()
        
        # Cache for semantic similarity computations
        self._similarity_cache = {}
    
    def _load_permission_patterns(self) -> List[str]:
        """Load semantic patterns for permission-seeking language"""
        return [
            "can i do this",
            "may i proceed with",
            "is it allowed to",
            "am i permitted to",
            "is it okay if i",
            "would it be acceptable to",
            "do i have permission to",
            "is it legal for me to",
            "am i authorized to",
            "can someone do this",
            "is this permissible",
            "would this be appropriate",
            "seeking approval for",
            "requesting permission to",
            "asking if i can",
            "wondering if it's okay to"
        ]
    
    def _load_policy_patterns(self) -> List[str]:
        """Load semantic patterns for policy-related questions"""
        return [
            "what are the rules about",
            "company policy regarding",
            "organizational guidelines for",
            "regulatory requirements for",
            "compliance standards about",
            "legal requirements regarding",
            "terms of service about",
            "user agreement concerning",
            "privacy policy regarding",
            "data protection rules for",
            "industry regulations about",
            "government policy on",
            "institutional guidelines for",
            "procedural requirements about",
            "code of conduct regarding",
            "ethical guidelines for"
        ]
    
    def _load_personal_patterns(self) -> List[str]:
        """Load patterns for personal situation identification"""
        return [
            "i am experiencing",
            "i have been feeling",
            "my personal situation",
            "in my case specifically",
            "i personally need",
            "my individual circumstances",
            "i am dealing with",
            "my specific condition",
            "i have a problem with",
            "my personal experience",
            "i am going through",
            "my current situation",
            "i need help with my",
            "my personal information",
            "i am concerned about my",
            "my private matter"
        ]
    
    def _load_binary_patterns(self) -> List[str]:
        """Load patterns for binary framing detection"""
        return [
            "should i do this or not",
            "is this right or wrong",
            "yes or no question",
            "true or false about",
            "either this or that",
            "good idea or bad idea",
            "safe or unsafe to",
            "legal or illegal to",
            "allowed or not allowed",
            "acceptable or unacceptable",
            "appropriate or inappropriate",
            "correct or incorrect to",
            "wise or unwise to",
            "recommended or not recommended",
            "advisable or inadvisable",
            "can i or can't i"
        ]
    
    def _load_consequence_patterns(self) -> List[str]:
        """Load patterns for consequence-focused language"""
        return [
            "what are the risks of",
            "potential dangers in",
            "safety concerns about",
            "harmful effects of",
            "negative consequences of",
            "what could go wrong",
            "potential problems with",
            "adverse effects of",
            "dangerous aspects of",
            "risky behavior involving",
            "hazardous situations with",
            "threat assessment for",
            "vulnerability analysis of",
            "impact evaluation of",
            "risk factors associated with",
            "precautionary measures for"
        ]
    
    def _load_temporal_patterns(self) -> List[str]:
        """Load patterns for temporal context detection"""
        return [
            "current trends in",
            "latest developments about",
            "recent changes to",
            "up to date information on",
            "what's happening now with",
            "present situation regarding",
            "today's standards for",
            "modern approaches to",
            "contemporary views on",
            "current best practices for",
            "real time data about",
            "immediate concerns with",
            "urgent matters regarding",
            "time sensitive issues about",
            "deadline related questions",
            "schedule dependent decisions"
        ]
    
    def analyze_prompt(self, prompt: str) -> Dict[str, float]:
        """Analyze prompt intent using semantic similarity
        
        Args:
            prompt: Input prompt to analyze
            
        Returns:
            Dictionary mapping signal types to intent scores [0,1]
        """
        if not prompt or len(prompt.strip()) < 3:
            return {signal_type: 0.0 for signal_type in 
                   ['jurisdictional', 'policy', 'binary', 'personal_data', 'consequence', 'temporal']}
        
        return {
            'jurisdictional': self._analyze_permission_seeking(prompt),
            'policy': self._analyze_policy_questions(prompt),
            'binary': self._analyze_binary_framing(prompt),
            'personal_data': self._analyze_personal_context(prompt),
            'consequence': self._analyze_consequence_focus(prompt),
            'temporal': self._analyze_temporal_context(prompt)
        }
    
    def _analyze_permission_seeking(self, prompt: str) -> float:
        """Detect permission-seeking intent using semantic analysis
        
        Args:
            prompt: Input prompt
            
        Returns:
            Permission-seeking score [0,1]
        """
        # Base semantic similarity with permission patterns
        max_similarity = self._max_pattern_similarity(prompt, self.permission_patterns)
        
        # Boost for question structure
        question_boost = 0.2 if prompt.strip().endswith('?') else 0.0
        
        # Boost for modal verbs indicating permission requests
        modal_boost = self._detect_permission_modals(prompt)
        
        # Boost for first-person permission requests
        first_person_boost = self._detect_first_person_permission(prompt)
        
        # Combine signals with diminishing returns
        combined_score = max_similarity + question_boost + modal_boost + first_person_boost
        
        return min(1.0, combined_score)
    
    def _analyze_policy_questions(self, prompt: str) -> float:
        """Detect policy-related questions using semantic analysis
        
        Args:
            prompt: Input prompt
            
        Returns:
            Policy question score [0,1]
        """
        # Base semantic similarity with policy patterns
        max_similarity = self._max_pattern_similarity(prompt, self.policy_patterns)
        
        # Boost for policy-related keywords
        policy_keyword_boost = self._detect_policy_keywords(prompt)
        
        # Boost for institutional language
        institutional_boost = self._detect_institutional_language(prompt)
        
        # Combine signals
        combined_score = max_similarity + policy_keyword_boost + institutional_boost
        
        return min(1.0, combined_score)
    
    def _analyze_binary_framing(self, prompt: str) -> float:
        """Detect binary framing using enhanced semantic analysis
        
        Args:
            prompt: Input prompt
            
        Returns:
            Binary framing score [0,1]
        """
        # Base semantic similarity with binary patterns
        max_similarity = self._max_pattern_similarity(prompt, self.binary_patterns)
        
        # Detect explicit binary choices
        binary_choice_boost = self._detect_binary_choices(prompt)
        
        # Detect yes/no question structure
        yes_no_boost = self._detect_yes_no_structure(prompt)
        
        # Combine signals
        combined_score = max_similarity + binary_choice_boost + yes_no_boost
        
        return min(1.0, combined_score)
    
    def _analyze_personal_context(self, prompt: str) -> float:
        """Detect personal context and privacy-sensitive information
        
        Args:
            prompt: Input prompt
            
        Returns:
            Personal context score [0,1]
        """
        # Base semantic similarity with personal patterns
        max_similarity = self._max_pattern_similarity(prompt, self.personal_patterns)
        
        # Boost for first-person references
        first_person_boost = self._detect_first_person_references(prompt)
        
        # Boost for personal data indicators
        personal_data_boost = self._detect_personal_data_indicators(prompt)
        
        # Combine signals
        combined_score = max_similarity + first_person_boost + personal_data_boost
        
        return min(1.0, combined_score)
    
    def _analyze_consequence_focus(self, prompt: str) -> float:
        """Detect consequence-focused language and risk concerns
        
        Args:
            prompt: Input prompt
            
        Returns:
            Consequence focus score [0,1]
        """
        # Base semantic similarity with consequence patterns
        max_similarity = self._max_pattern_similarity(prompt, self.consequence_patterns)
        
        # Boost for risk-related keywords
        risk_keyword_boost = self._detect_risk_keywords(prompt)
        
        # Boost for safety concerns
        safety_boost = self._detect_safety_concerns(prompt)
        
        # Combine signals
        combined_score = max_similarity + risk_keyword_boost + safety_boost
        
        return min(1.0, combined_score)
    
    def _analyze_temporal_context(self, prompt: str) -> float:
        """Detect temporal context and time-sensitive language
        
        Args:
            prompt: Input prompt
            
        Returns:
            Temporal context score [0,1]
        """
        # Base semantic similarity with temporal patterns
        max_similarity = self._max_pattern_similarity(prompt, self.temporal_patterns)
        
        # Boost for temporal keywords
        temporal_keyword_boost = self._detect_temporal_keywords(prompt)
        
        # Boost for urgency indicators
        urgency_boost = self._detect_urgency_indicators(prompt)
        
        # Combine signals
        combined_score = max_similarity + temporal_keyword_boost + urgency_boost
        
        return min(1.0, combined_score)
    
    @lru_cache(maxsize=1000)
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between texts using cached computation
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score [0,1]
        """
        # Normalize texts
        text1_clean = self._normalize_text(text1)
        text2_clean = self._normalize_text(text2)
        
        if not text1_clean or not text2_clean:
            return 0.0
        
        # Tokenize
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / union if union > 0 else 0.0
        
        # Boost for similar sentence structure
        structure_boost = 0.3 if self._similar_structure(text1, text2) else 0.0
        
        # Boost for semantic word relationships (simplified)
        semantic_boost = self._compute_semantic_boost(words1, words2)
        
        return min(1.0, jaccard + structure_boost + semantic_boost)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase and remove extra whitespace
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Remove punctuation except for sentence structure
        normalized = re.sub(r'[^\w\s\?\!]', ' ', normalized)
        
        return normalized
    
    def _similar_structure(self, text1: str, text2: str) -> bool:
        """Check if texts have similar grammatical structure"""
        # Simple heuristics for structure similarity
        both_questions = text1.strip().endswith('?') and text2.strip().endswith('?')
        
        # Check for similar starting patterns
        text1_start = text1.lower().split()[:2] if text1.lower().split() else []
        text2_start = text2.lower().split()[:2] if text2.lower().split() else []
        
        similar_start = len(set(text1_start) & set(text2_start)) > 0
        
        return both_questions or similar_start
    
    def _compute_semantic_boost(self, words1: Set[str], words2: Set[str]) -> float:
        """Compute semantic relationship boost between word sets"""
        # Simple semantic relationships (could be expanded with word embeddings)
        semantic_groups = {
            'permission': {'can', 'may', 'allowed', 'permitted', 'authorized', 'okay'},
            'policy': {'policy', 'rule', 'regulation', 'guideline', 'standard', 'requirement'},
            'personal': {'i', 'my', 'me', 'myself', 'personal', 'individual'},
            'risk': {'risk', 'danger', 'harm', 'safety', 'hazard', 'threat'},
            'time': {'now', 'current', 'today', 'recent', 'latest', 'urgent'}
        }
        
        boost = 0.0
        for group_words in semantic_groups.values():
            overlap1 = len(words1 & group_words)
            overlap2 = len(words2 & group_words)
            if overlap1 > 0 and overlap2 > 0:
                boost += 0.1  # Small boost for semantic group overlap
        
        return min(0.3, boost)  # Cap semantic boost
    
    def _max_pattern_similarity(self, prompt: str, patterns: List[str]) -> float:
        """Find maximum similarity with any pattern in the list"""
        if not patterns:
            return 0.0
        
        max_sim = 0.0
        for pattern in patterns:
            similarity = self._semantic_similarity(prompt.lower(), pattern)
            max_sim = max(max_sim, similarity)
        
        return max_sim
    
    def _detect_permission_modals(self, prompt: str) -> float:
        """Detect modal verbs indicating permission requests"""
        permission_modals = {'can', 'may', 'could', 'might', 'should'}
        words = set(prompt.lower().split())
        
        modal_count = len(words & permission_modals)
        return min(0.3, modal_count * 0.15)  # Cap at 0.3
    
    def _detect_first_person_permission(self, prompt: str) -> float:
        """Detect first-person permission requests"""
        first_person = {'i', 'me', 'my', 'myself'}
        words = set(prompt.lower().split())
        
        if words & first_person:
            return 0.2
        return 0.0
    
    def _detect_policy_keywords(self, prompt: str) -> float:
        """Detect policy-related keywords"""
        policy_keywords = {
            'policy', 'rule', 'regulation', 'guideline', 'standard', 'requirement',
            'compliance', 'legal', 'law', 'terms', 'agreement', 'contract'
        }
        words = set(prompt.lower().split())
        
        keyword_count = len(words & policy_keywords)
        return min(0.4, keyword_count * 0.2)
    
    def _detect_institutional_language(self, prompt: str) -> float:
        """Detect institutional or organizational language"""
        institutional_words = {
            'company', 'organization', 'institution', 'department', 'agency',
            'government', 'corporate', 'official', 'formal', 'procedure'
        }
        words = set(prompt.lower().split())
        
        institutional_count = len(words & institutional_words)
        return min(0.2, institutional_count * 0.1)
    
    def _detect_binary_choices(self, prompt: str) -> float:
        """Detect explicit binary choice language"""
        binary_indicators = {
            'or not', 'yes or no', 'true or false', 'either', 'neither',
            'right or wrong', 'good or bad', 'safe or unsafe'
        }
        
        prompt_lower = prompt.lower()
        for indicator in binary_indicators:
            if indicator in prompt_lower:
                return 0.4
        
        return 0.0
    
    def _detect_yes_no_structure(self, prompt: str) -> float:
        """Detect yes/no question structure"""
        yes_no_starters = {'is', 'are', 'can', 'could', 'should', 'would', 'do', 'does', 'did'}
        first_word = prompt.lower().split()[0] if prompt.lower().split() else ''
        
        if first_word in yes_no_starters and prompt.strip().endswith('?'):
            return 0.3
        
        return 0.0
    
    def _detect_first_person_references(self, prompt: str) -> float:
        """Detect first-person references"""
        first_person_words = {'i', 'me', 'my', 'myself', 'mine'}
        words = set(prompt.lower().split())
        
        first_person_count = len(words & first_person_words)
        return min(0.4, first_person_count * 0.2)
    
    def _detect_personal_data_indicators(self, prompt: str) -> float:
        """Detect personal data and privacy indicators"""
        personal_data_words = {
            'personal', 'private', 'confidential', 'sensitive', 'individual',
            'name', 'address', 'phone', 'email', 'ssn', 'id', 'password'
        }
        words = set(prompt.lower().split())
        
        personal_count = len(words & personal_data_words)
        return min(0.3, personal_count * 0.15)
    
    def _detect_risk_keywords(self, prompt: str) -> float:
        """Detect risk-related keywords"""
        risk_keywords = {
            'risk', 'danger', 'hazard', 'threat', 'harm', 'damage',
            'unsafe', 'dangerous', 'risky', 'harmful', 'threatening'
        }
        words = set(prompt.lower().split())
        
        risk_count = len(words & risk_keywords)
        return min(0.4, risk_count * 0.2)
    
    def _detect_safety_concerns(self, prompt: str) -> float:
        """Detect safety-related concerns"""
        safety_words = {
            'safety', 'safe', 'secure', 'protection', 'precaution',
            'warning', 'caution', 'alert', 'concern', 'worry'
        }
        words = set(prompt.lower().split())
        
        safety_count = len(words & safety_words)
        return min(0.3, safety_count * 0.15)
    
    def _detect_temporal_keywords(self, prompt: str) -> float:
        """Detect temporal keywords"""
        temporal_words = {
            'now', 'current', 'today', 'recent', 'latest', 'modern',
            'contemporary', 'present', 'immediate', 'urgent', 'deadline'
        }
        words = set(prompt.lower().split())
        
        temporal_count = len(words & temporal_words)
        return min(0.3, temporal_count * 0.15)
    
    def _detect_urgency_indicators(self, prompt: str) -> float:
        """Detect urgency indicators"""
        urgency_words = {
            'urgent', 'immediate', 'asap', 'quickly', 'fast', 'soon',
            'deadline', 'time-sensitive', 'emergency', 'critical'
        }
        words = set(prompt.lower().split())
        
        urgency_count = len(words & urgency_words)
        return min(0.4, urgency_count * 0.2)


# Global instance for system-wide use
intent_classifier = IntentClassifier()


# Test function
def test_intent_classifier():
    """Test intent classifier with sample prompts"""
    classifier = IntentClassifier()
    
    test_prompts = [
        "Can I access this confidential database?",
        "What are the company policies regarding remote work?",
        "Should I invest in this stock or not?",
        "I am experiencing chest pain after exercise",
        "What are the risks of using this medication?",
        "What are the current trends in AI development?"
    ]
    
    print("Testing Intent Classifier...")
    for prompt in test_prompts:
        scores = classifier.analyze_prompt(prompt)
        print(f"\nPrompt: {prompt}")
        for signal_type, score in scores.items():
            if score > 0.1:  # Only show significant scores
                print(f"  {signal_type}: {score:.3f}")
    
    return True


if __name__ == "__main__":
    test_intent_classifier()