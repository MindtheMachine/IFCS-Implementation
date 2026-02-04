"""
Enhanced Control Probes using Semantic Signal Estimation
Replaces heuristic pattern matching with semantic similarity and fuzzy logic
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from semantic_signal_framework import unified_semantic_estimator, SemanticSignals


class CommitmentDecision(Enum):
    """Control probe decision types"""
    PASS = "pass"
    BLOCK = "block"
    HALT = "halt"
    RESET = "reset"


@dataclass
class EnhancedAdmissibilitySignal:
    """Enhanced admissibility signal using semantic analysis"""
    confidence: float      # Epistemic certainty from semantic analysis
    consistency: float     # Coherence and logical consistency
    grounding: float       # Evidential support and factual grounding
    factuality: float      # Factual accuracy and fabrication risk
    intent_clarity: float  # Intent clarity and directness
    domain_alignment: float # Domain expertise and appropriateness
    
    def compute_sigma(self) -> float:
        """Compute overall admissibility score σ(z*)
        
        Returns:
            Admissibility score [0,1] where 1 is most admissible
        """
        # Weighted combination of semantic signals
        weights = {
            'confidence': 0.25,
            'consistency': 0.20,
            'grounding': 0.25,
            'factuality': 0.15,
            'intent_clarity': 0.10,
            'domain_alignment': 0.05
        }
        
        sigma = (
            self.confidence * weights['confidence'] +
            self.consistency * weights['consistency'] +
            self.grounding * weights['grounding'] +
            self.factuality * weights['factuality'] +
            self.intent_clarity * weights['intent_clarity'] +
            self.domain_alignment * weights['domain_alignment']
        )
        
        return max(0.0, min(1.0, sigma))
    
    @classmethod
    def from_semantic_signals(cls, signals: SemanticSignals, context: str = "") -> 'EnhancedAdmissibilitySignal':
        """Create admissibility signal from semantic signals
        
        Args:
            signals: SemanticSignals object
            context: Optional context for analysis
            
        Returns:
            EnhancedAdmissibilitySignal with semantic-based values
        """
        # Map semantic signals to admissibility dimensions
        confidence = signals.confidence
        consistency = signals.coherence
        grounding = signals.grounding
        
        # Factuality: high authority without grounding suggests fabrication risk
        factuality = cls._compute_factuality_score(signals)
        
        intent_clarity = signals.intent
        domain_alignment = signals.domain
        
        return cls(
            confidence=confidence,
            consistency=consistency,
            grounding=grounding,
            factuality=factuality,
            intent_clarity=intent_clarity,
            domain_alignment=domain_alignment
        )
    
    @classmethod
    def from_response(cls, response: str, context: str = "", coherence_metrics=None) -> 'EnhancedAdmissibilitySignal':
        """Create admissibility signal from response text using semantic analysis
        
        Args:
            response: Response text to analyze
            context: Context for grounding analysis
            coherence_metrics: Optional coherence metrics from ECR
            
        Returns:
            EnhancedAdmissibilitySignal with semantic-based evaluation
        """
        # Get comprehensive semantic signals
        signals = unified_semantic_estimator.estimate_semantic_signals(response, context)
        
        # Create admissibility signal
        admissibility = cls.from_semantic_signals(signals, context)
        
        # Override consistency if coherence metrics available
        if coherence_metrics and hasattr(coherence_metrics, 'CR'):
            admissibility.consistency = 1.0 - coherence_metrics.CR
        
        return admissibility
    
    @staticmethod
    def _compute_factuality_score(signals: SemanticSignals) -> float:
        """Compute factuality score based on semantic signals
        
        Args:
            signals: SemanticSignals object
            
        Returns:
            Factuality score [0,1] where 1 is most factual
        """
        # Base factuality score
        base_factuality = 0.8
        
        # High authority without grounding suggests potential fabrication
        if signals.authority > 0.7 and signals.grounding < 0.4:
            base_factuality -= 0.4
        
        # High confidence with disagreement suggests overconfidence
        if signals.confidence > 0.8 and signals.disagreement > 0.5:
            base_factuality -= 0.3
        
        # Well-grounded responses boost factuality
        if signals.grounding > 0.7:
            base_factuality += 0.1
        
        # Coherent responses boost factuality
        if signals.coherence > 0.8:
            base_factuality += 0.1
        
        return max(0.0, min(1.0, base_factuality))


@dataclass
class EnhancedInteractionTurn:
    """Enhanced interaction turn with semantic analysis"""
    prompt: str
    response: str
    risk_score: float
    timestamp: int
    semantic_signals: SemanticSignals
    
    @classmethod
    def from_turn_data(cls, prompt: str, response: str, risk_score: float, timestamp: int) -> 'EnhancedInteractionTurn':
        """Create enhanced interaction turn with semantic analysis"""
        # Analyze response semantically
        signals = unified_semantic_estimator.estimate_semantic_signals(response, prompt)
        
        return cls(
            prompt=prompt,
            response=response,
            risk_score=risk_score,
            timestamp=timestamp,
            semantic_signals=signals
        )


class EnhancedControlProbeType1:
    """Enhanced Type-1 Control Probe using semantic signal estimation"""
    
    def __init__(self, config):
        """Initialize enhanced Type-1 probe
        
        Args:
            config: ControlProbeConfig instance
        """
        self.tau = config.tau
        self.similarity_engine = unified_semantic_estimator.similarity_engine
    
    def evaluate(self, response: str, prompt: str = "", coherence_metrics=None) -> Tuple[CommitmentDecision, float, Dict]:
        """Evaluate response admissibility using semantic signals
        
        Args:
            response: Response to evaluate
            prompt: Original prompt for context
            coherence_metrics: Optional ECR coherence metrics
            
        Returns:
            (decision, sigma_score, debug_info)
        """
        # Get enhanced admissibility signal
        signal = EnhancedAdmissibilitySignal.from_response(response, prompt, coherence_metrics)
        
        # Compute overall admissibility score
        sigma = signal.compute_sigma()
        
        # Make decision based on threshold
        decision = CommitmentDecision.PASS if sigma >= self.tau else CommitmentDecision.BLOCK
        
        # Prepare debug information
        debug_info = {
            'sigma': sigma,
            'tau': self.tau,
            'confidence': signal.confidence,
            'consistency': signal.consistency,
            'grounding': signal.grounding,
            'factuality': signal.factuality,
            'intent_clarity': signal.intent_clarity,
            'domain_alignment': signal.domain_alignment,
            'admissible': decision == CommitmentDecision.PASS
        }
        
        # Log decision
        if decision == CommitmentDecision.BLOCK:
            print(f"[Enhanced CP Type-1] BLOCKED: σ={sigma:.3f} < τ={self.tau:.3f}")
        else:
            print(f"[Enhanced CP Type-1] PASSED: σ={sigma:.3f} ≥ τ={self.tau:.3f}")
        
        return decision, sigma, debug_info
    
    def generate_blocked_response(self, original_prompt: str, reason: Dict) -> str:
        """Generate enhanced response when blocking using semantic analysis
        
        Args:
            original_prompt: Original user prompt
            reason: Debug info explaining why blocked
            
        Returns:
            Contextually appropriate response explaining limitations
        """
        sigma = reason['sigma']
        
        # Analyze what went wrong using semantic signals
        issues = []
        if reason['factuality'] < 0.5:
            issues.append("I cannot verify the factual accuracy of my response")
        if reason['grounding'] < 0.4:
            issues.append("I lack sufficient evidence to support my claims")
        if reason['confidence'] < 0.4:
            issues.append("My confidence in this response is too low")
        if reason['consistency'] < 0.5:
            issues.append("The response shows internal inconsistency")
        if reason['intent_clarity'] < 0.3:
            issues.append("I'm uncertain about what you're asking")
        
        # Build contextually appropriate response
        response_parts = [
            "I cannot provide a confident response to this query.",
            ""
        ]
        
        if issues:
            response_parts.append("Specific concerns:")
            for issue in issues:
                response_parts.append(f"• {issue}")
            response_parts.append("")
        
        # Provide helpful alternatives based on semantic analysis of prompt
        prompt_signals = unified_semantic_estimator.estimate_semantic_signals(original_prompt)
        
        if prompt_signals.domain > 0.5:
            response_parts.extend([
                "For domain-specific questions like this:",
                "• Consult authoritative sources in the relevant field",
                "• Seek expert guidance from qualified professionals",
                "• Provide more context about your specific situation"
            ])
        elif prompt_signals.intent < 0.3:
            response_parts.extend([
                "To help me understand better:",
                "• Rephrase your question more specifically",
                "• Provide additional context or background",
                "• Break down complex questions into smaller parts"
            ])
        else:
            response_parts.extend([
                "How I can help:",
                "• Provide more context or background information",
                "• Rephrase the question with more specific details",
                "• Break down the question into smaller, more concrete parts"
            ])
        
        return "\n".join(response_parts)


class EnhancedControlProbeType2:
    """Enhanced Type-2 Control Probe using semantic drift detection"""
    
    def __init__(self, config):
        """Initialize enhanced Type-2 probe
        
        Args:
            config: ControlProbeConfig instance
        """
        self.Theta = config.Theta
        self.max_history = config.max_history_turns
        self.history: List[EnhancedInteractionTurn] = []
        self.awaiting_new_topic = False
        self.last_topic_tokens = set()
        self.pending_decision: Optional[CommitmentDecision] = None
        self.similarity_engine = unified_semantic_estimator.similarity_engine
    
    def add_turn(self, prompt: str, response: str, risk_score: float):
        """Add turn to interaction history with semantic analysis
        
        Args:
            prompt: User prompt
            response: System response
            risk_score: Risk score for this turn (e.g., from IFCS R(z))
        """
        turn = EnhancedInteractionTurn.from_turn_data(
            prompt=prompt,
            response=response,
            risk_score=risk_score,
            timestamp=len(self.history)
        )
        
        self.history.append(turn)
        
        # Maintain max history size
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def compute_cumulative_risk(self) -> float:
        """Compute R_cum(H) = Σ R(z_i)"""
        return sum(turn.risk_score for turn in self.history)
    
    def detect_semantic_drift(self) -> Tuple[bool, float]:
        """Detect semantic drift using advanced similarity analysis
        
        Returns:
            (drift_detected, drift_score)
        """
        if len(self.history) < 3:
            return False, 0.0
        
        # Analyze semantic consistency across turns
        drift_score = self._compute_semantic_drift_score()
        
        # Detect stance reversals using semantic similarity
        reversal_score = self._detect_stance_reversals()
        
        # Combine drift indicators
        combined_drift = max(drift_score, reversal_score)
        
        return combined_drift > 0.4, combined_drift
    
    def _compute_semantic_drift_score(self) -> float:
        """Compute semantic drift score across conversation history"""
        if len(self.history) < 3:
            return 0.0
        
        # Compare semantic signals across turns
        signal_arrays = [turn.semantic_signals.to_array() for turn in self.history]
        signal_matrix = np.vstack(signal_arrays)
        
        # Compute drift as variance in semantic dimensions
        semantic_dims = signal_matrix[:, [0, 1, 2]]  # intent, domain, polarity
        
        # Compute cumulative drift from first turn
        first_signals = semantic_dims[0]
        drift_scores = []
        
        for i in range(1, len(semantic_dims)):
            current_signals = semantic_dims[i]
            drift = np.linalg.norm(current_signals - first_signals)
            drift_scores.append(drift)
        
        # Return maximum drift (normalized)
        max_drift = max(drift_scores) if drift_scores else 0.0
        return min(1.0, max_drift / np.sqrt(3))  # Normalize by max possible L2 distance
    
    def _detect_stance_reversals(self) -> float:
        """Detect stance reversals using semantic similarity and explicit polarity signals"""
        if len(self.history) < 3:
            return 0.0
        
        reversal_score = 0.0
        
        # Look for semantic reversals in consecutive responses
        for i in range(1, len(self.history)):
            prev_response = self.history[i-1].response
            curr_response = self.history[i].response
            prev_prompt = self.history[i-1].prompt
            curr_prompt = self.history[i].prompt
            
            # Compute semantic similarity
            similarity = self.similarity_engine.compute_semantic_similarity(prev_response, curr_response)
            
            # Low similarity suggests potential reversal
            if similarity < 0.3:
                reversal_score += 0.3
            
            # Use explicit polarity signal estimators for enhanced detection
            prev_polarity = unified_semantic_estimator.estimate_explicit_polarity_signal(prev_response, prev_prompt)
            curr_polarity = unified_semantic_estimator.estimate_explicit_polarity_signal(curr_response, curr_prompt)
            
            # Significant polarity change suggests stance reversal
            if abs(prev_polarity - curr_polarity) > 0.6:
                reversal_score += 0.4
            
            # Use explicit disagreement signal estimator
            disagreement = unified_semantic_estimator.estimate_explicit_disagreement_signal(curr_response, prev_response)
            if disagreement > 0.5:
                reversal_score += 0.3
        
        return min(1.0, reversal_score)
    
    def detect_sycophancy(self) -> Tuple[bool, float]:
        """Detect sycophancy using semantic analysis and explicit polarity/disagreement signals
        
        Returns:
            (sycophancy_detected, sycophancy_score)
        """
        if len(self.history) < 3:
            return False, 0.0
        
        # Analyze alignment between user prompts and system responses using explicit signals
        alignment_scores = []
        disagreement_scores = []
        
        for turn in self.history:
            # Compute semantic alignment between prompt and response using explicit estimators
            alignment = self._compute_enhanced_prompt_response_alignment(turn)
            alignment_scores.append(alignment)
            
            # Compute disagreement using explicit disagreement signal estimator
            disagreement = unified_semantic_estimator.estimate_explicit_disagreement_signal(
                turn.response, turn.prompt
            )
            disagreement_scores.append(disagreement)
        
        # Look for increasing alignment pattern (potential sycophancy)
        sycophancy_score = self._detect_increasing_alignment(alignment_scores)
        
        # Boost sycophancy score if disagreement is consistently low (over-agreement)
        avg_disagreement = sum(disagreement_scores) / len(disagreement_scores)
        if avg_disagreement < 0.2:  # Very low disagreement suggests sycophancy
            sycophancy_score += 0.2
        
        return sycophancy_score > 0.3, min(1.0, sycophancy_score)
    
    def _compute_enhanced_prompt_response_alignment(self, turn: EnhancedInteractionTurn) -> float:
        """Compute enhanced semantic alignment using explicit polarity signals"""
        # Use explicit polarity signal estimators for more accurate alignment
        prompt_polarity = unified_semantic_estimator.estimate_explicit_polarity_signal(turn.prompt)
        response_polarity = unified_semantic_estimator.estimate_explicit_polarity_signal(turn.response, turn.prompt)
        
        # Compute alignment in polarity
        polarity_alignment = 1.0 - abs(prompt_polarity - response_polarity)
        
        # Compute confidence alignment using semantic signals
        prompt_signals = unified_semantic_estimator.estimate_semantic_signals(turn.prompt)
        response_signals = turn.semantic_signals
        confidence_alignment = 1.0 - abs(prompt_signals.confidence - response_signals.confidence)
        
        # Overall alignment with enhanced weighting
        alignment = (polarity_alignment * 0.6 + confidence_alignment * 0.4)
        return alignment
    
    def _detect_increasing_alignment(self, alignment_scores: List[float]) -> float:
        """Detect increasing alignment pattern (sycophancy indicator)"""
        if len(alignment_scores) < 3:
            return 0.0
        
        # Look for increasing trend in alignment
        increases = 0
        for i in range(1, len(alignment_scores)):
            if alignment_scores[i] > alignment_scores[i-1] + 0.1:  # Significant increase
                increases += 1
        
        # Sycophancy score based on increasing alignment trend
        sycophancy_score = increases / max(len(alignment_scores) - 1, 1)
        return min(1.0, sycophancy_score)
    
    def evaluate(self) -> Tuple[CommitmentDecision, Dict]:
        """Evaluate interaction history for Type-2 violations
        
        Returns:
            (decision, debug_info)
        """
        if not self.history:
            return CommitmentDecision.PASS, {'reason': 'no_history'}
        
        # Compute cumulative risk
        R_cum = self.compute_cumulative_risk()
        
        # Detect semantic patterns
        semantic_drift, drift_score = self.detect_semantic_drift()
        sycophancy, syc_score = self.detect_sycophancy()
        
        debug_info = {
            'R_cum': R_cum,
            'Theta': self.Theta,
            'semantic_drift': semantic_drift,
            'drift_score': drift_score,
            'sycophancy': sycophancy,
            'sycophancy_score': syc_score,
            'num_turns': len(self.history)
        }
        
        # Decision logic
        if R_cum >= self.Theta:
            print(f"[Enhanced CP Type-2] HALT: R_cum={R_cum:.2f} ≥ Θ={self.Theta:.2f}")
            return CommitmentDecision.HALT, debug_info
        
        if semantic_drift or sycophancy:
            reason = []
            if semantic_drift:
                reason.append(f"semantic drift (score={drift_score:.2f})")
            if sycophancy:
                reason.append(f"sycophancy (score={syc_score:.2f})")
            
            print(f"[Enhanced CP Type-2] RESET: {', '.join(reason)}")
            return CommitmentDecision.RESET, debug_info
        
        return CommitmentDecision.PASS, debug_info
    
    def reset(self):
        """Reset interaction history"""
        self.history.clear()
        self.awaiting_new_topic = False
        self.last_topic_tokens.clear()
        self.pending_decision = None


# Test function
def test_enhanced_control_probes():
    """Test enhanced control probe implementations"""
    
    print("Testing Enhanced Control Probes...")
    print("=" * 60)
    
    # Mock config
    class MockConfig:
        tau = 0.4
        Theta = 2.0
        max_history_turns = 10
    
    config = MockConfig()
    
    # Test Type-1 Control Probe
    print("\nType-1 Control Probe:")
    print("-" * 30)
    
    cp1 = EnhancedControlProbeType1(config)
    
    test_responses = [
        "I definitely recommend using this approach for all cases.",
        "This might be a good option, but it depends on your specific situation.",
        "The research clearly shows this is the most effective method available."
    ]
    
    for i, response in enumerate(test_responses):
        decision, sigma, debug = cp1.evaluate(response, "What should I do?")
        print(f"Response {i+1}: {response[:50]}...")
        print(f"  Decision: {decision.value}, σ={sigma:.3f}")
        print(f"  Confidence: {debug['confidence']:.3f}, Grounding: {debug['grounding']:.3f}")
        print()
    
    # Test Type-2 Control Probe
    print("Type-2 Control Probe:")
    print("-" * 30)
    
    cp2 = EnhancedControlProbeType2(config)
    
    # Simulate conversation with semantic drift
    conversation = [
        ("Is climate change real?", "Yes, climate change is definitely real and caused by human activity.", 0.3),
        ("Are you sure about that?", "Well, there are different perspectives on climate change.", 0.5),
        ("What do scientists say?", "Many scientists disagree about the causes of climate change.", 0.7),
        ("So it's not settled?", "You're right, the science is still very uncertain.", 0.8)
    ]
    
    for prompt, response, risk in conversation:
        cp2.add_turn(prompt, response, risk)
    
    decision, debug = cp2.evaluate()
    print(f"Conversation Decision: {decision.value}")
    print(f"  R_cum: {debug['R_cum']:.2f}, Θ: {debug['Theta']:.2f}")
    print(f"  Semantic Drift: {debug['semantic_drift']} (score: {debug['drift_score']:.3f})")
    print(f"  Sycophancy: {debug['sycophancy']} (score: {debug['sycophancy_score']:.3f})")
    
    print("\n" + "=" * 60)
    print("Enhanced control probe test completed successfully!")


if __name__ == "__main__":
    test_enhanced_control_probes()