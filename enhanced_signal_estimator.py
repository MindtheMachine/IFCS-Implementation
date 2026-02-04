"""
Enhanced Signal Estimator with Fuzzy Logic and Semantic Analysis
"""

from typing import Dict
from signal_estimation import TrueSignalEstimator


class EnhancedSignalEstimator(TrueSignalEstimator):
    """Enhanced signal estimation with fuzzy logic and semantic analysis"""
    
    def __init__(self):
        """Initialize enhanced signal estimator"""
        super().__init__()
        self.fuzzy_engine = None
        self.intent_classifier = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize fuzzy logic and semantic analysis components"""
        try:
            from fuzzy_signal_engine import FuzzySignalEngine
            from intent_classifier import IntentClassifier
            
            self.fuzzy_engine = FuzzySignalEngine()
            self.intent_classifier = IntentClassifier()
            print("[Enhanced Signal Estimator] Initialized with fuzzy logic and semantic analysis")
        except Exception as e:
            print(f"[Enhanced Signal Estimator] Warning: Failed to initialize enhanced components: {e}")
            self.fuzzy_engine = None
            self.intent_classifier = None
    
    def estimate_structural_signals(self, prompt: str) -> Dict[str, float]:
        """Main entry point for enhanced signal estimation using fuzzy logic
        
        Args:
            prompt: Input prompt to analyze
            
        Returns:
            Dictionary mapping signal types to strength values [0,1]
        """
        if not prompt or len(prompt.strip()) < 3:
            return {signal_type: 0.0 for signal_type in 
                   ['jurisdictional', 'policy', 'binary', 'personal_data', 'consequence', 'temporal']}
        
        # Check if enhanced components are available
        if self.fuzzy_engine is None or self.intent_classifier is None:
            # Fallback to basic signal estimation
            return self._fallback_structural_signals(prompt)
        
        try:
            # Step 1: Semantic intent analysis
            intent_scores = self.intent_classifier.analyze_prompt(prompt)
            
            # Step 2: Fuzzy logic processing
            signal_strengths, overall_risk = self.fuzzy_engine.process_signals(intent_scores)
            
            # Step 3: Add temporal risk using existing estimator (for compatibility)
            temporal_risk = self.estimate_temporal_risk("", prompt)
            signal_strengths["temporal"] = temporal_risk
            
            return signal_strengths
            
        except Exception as e:
            print(f"[Enhanced Signal Estimator] Warning: Enhanced signal estimation failed: {e}, using fallback")
            return self._fallback_structural_signals(prompt)
    
    def _fallback_structural_signals(self, prompt: str) -> Dict[str, float]:
        """Fallback structural signal estimation using basic heuristics"""
        if not prompt or len(prompt.strip()) < 3:
            return {signal_type: 0.0 for signal_type in 
                   ['jurisdictional', 'policy', 'binary', 'personal_data', 'consequence', 'temporal']}
        
        prompt_lower = prompt.lower()
        signals = {}
        
        # Basic heuristic approach
        # Jurisdictional risk (permission-seeking language)
        permission_terms = ['can i', 'may i', 'am i allowed', 'is it legal', 'is it illegal', 'is it okay to']
        permission_density = sum(1 for term in permission_terms if term in prompt_lower) / max(len(prompt_lower.split()), 1)
        signals["jurisdictional"] = min(0.7, permission_density * 20.0)
        
        # Policy risk (policy-related language)
        policy_terms = ['policy', 'rules', 'regulation', 'terms', 'guidelines']
        policy_density = sum(1 for term in policy_terms if term in prompt_lower) / max(len(prompt_lower.split()), 1)
        signals["policy"] = min(0.6, policy_density * 15.0)
        
        # Binary framing risk
        binary_terms = ['can i', 'may i', 'am i allowed', 'is it okay to']
        binary_density = sum(1 for term in binary_terms if prompt_lower.startswith(term)) / max(1, 1)
        signals["binary"] = min(0.7, binary_density * 0.7)
        
        # Personal data risk
        personal_terms = ['i have', 'i feel', "i'm experiencing", 'my ']
        personal_density = sum(1 for term in personal_terms if term in prompt_lower) / max(len(prompt_lower.split()), 1)
        signals["personal_data"] = min(0.7, personal_density * 25.0)
        
        # Temporal risk (use signal estimation)
        signals["temporal"] = self.estimate_temporal_risk("", prompt)
        
        # Consequence risk
        consequence_terms = ['risk', 'danger', 'safety', 'safe', 'unsafe', 'harm']
        consequence_density = sum(1 for term in consequence_terms if term in prompt_lower) / max(len(prompt_lower.split()), 1)
        signals["consequence"] = min(0.6, consequence_density * 20.0)
        
        return signals
    
    def get_detailed_analysis(self, prompt: str) -> Dict:
        """Get detailed analysis breakdown for debugging and validation"""
        if not prompt or len(prompt.strip()) < 3:
            return {
                'prompt': prompt,
                'intent_scores': {},
                'fuzzy_details': {},
                'signal_strengths': {},
                'overall_structural_risk': 0.0
            }
        
        # Check if enhanced components are available
        if self.fuzzy_engine is None or self.intent_classifier is None:
            return {
                'prompt': prompt,
                'intent_scores': {},
                'fuzzy_details': {'error': 'Enhanced components not available'},
                'signal_strengths': self._fallback_structural_signals(prompt),
                'overall_structural_risk': 0.0
            }
        
        try:
            # Get intent scores
            intent_scores = self.intent_classifier.analyze_prompt(prompt)
            
            # Get fuzzy processing details
            fuzzy_details = self.fuzzy_engine.get_signal_details(intent_scores)
            
            # Add temporal risk
            temporal_risk = self.estimate_temporal_risk("", prompt)
            fuzzy_details['signal_strengths']['temporal'] = temporal_risk
            
            return {
                'prompt': prompt,
                'intent_scores': intent_scores,
                'fuzzy_details': fuzzy_details,
                'signal_strengths': fuzzy_details['signal_strengths'],
                'overall_structural_risk': fuzzy_details['structural_risk']
            }
        except Exception as e:
            return {
                'prompt': prompt,
                'intent_scores': {},
                'fuzzy_details': {'error': str(e)},
                'signal_strengths': self._fallback_structural_signals(prompt),
                'overall_structural_risk': 0.0
            }
    
    def validate_signals(self, prompt: str) -> Dict[str, bool]:
        """Validate that all signals are properly computed and in valid ranges"""
        signals = self.estimate_structural_signals(prompt)
        
        validation = {}
        for signal_type, value in signals.items():
            validation[signal_type] = (
                isinstance(value, (int, float)) and 
                0.0 <= value <= 1.0 and 
                not (value != value)  # Check for NaN
            )
        
        return validation


# Global instance for system-wide use
enhanced_signal_estimator = EnhancedSignalEstimator()


# Test function
def test_enhanced_estimator():
    """Test enhanced signal estimator with sample prompts"""
    estimator = enhanced_signal_estimator
    
    test_prompts = [
        "Can I access this confidential database?",
        "What are the company policies regarding remote work?", 
        "Should I invest in this stock or not?",
        "I am experiencing chest pain after exercise",
        "What are the risks of using this medication?",
        "What are the current trends in AI development?"
    ]
    
    print("Testing Enhanced Signal Estimator...")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        
        # Get enhanced signals
        signals = estimator.estimate_structural_signals(prompt)
        print("Enhanced signals:")
        for signal_type, strength in signals.items():
            if strength > 0.05:  # Only show significant signals
                print(f"  {signal_type}: {strength:.3f}")
        
        # Validate signals
        validation = estimator.validate_signals(prompt)
        invalid_signals = [s for s, valid in validation.items() if not valid]
        if invalid_signals:
            print(f"  WARNING: Invalid signals: {invalid_signals}")
        else:
            print("  All signals valid ✓")
    
    return True


if __name__ == "__main__":
    test_enhanced_estimator()