"""
Control Probe Implementation (Type-1 and Type-2)
Based on: Chatterjee, A. (2026a). Control Probe: Inference-time commitment control
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class CommitmentDecision(Enum):
    """Control Probe decisions"""
    PASS = "pass"  # Commitment allowed
    BLOCK = "block"  # Type-1: Block output
    HALT = "halt"  # Type-2: Halt interaction
    RESET = "reset"  # Type-2: Reset interaction


@dataclass
class AdmissibilitySignal:
    """Evaluative support signal σ(z)"""
    confidence: float  # 0-1
    consistency: float  # 0-1
    grounding: float  # 0-1
    factuality: float  # 0-1
    
    def compute_sigma(self) -> float:
        """Compute overall evaluative support"""
        return (self.confidence + self.consistency + self.grounding + self.factuality) / 4.0
    
    @classmethod
    def from_response(cls, response: str, context: str = "", coherence_metrics=None) -> 'AdmissibilitySignal':
        """Estimate admissibility signal from response"""
        
        # Confidence: inverse of uncertainty markers
        uncertainty_markers = ['might', 'could', 'possibly', 'perhaps', 'maybe', 
                              'not sure', 'unclear', "don't know"]
        uncert_count = sum(1 for marker in uncertainty_markers if marker in response.lower())
        confidence = max(0.0, 1.0 - (uncert_count * 0.15))
        
        # Consistency: from coherence metrics if available
        if coherence_metrics:
            consistency = 1.0 - coherence_metrics.CR  # Inverse of contradiction rate
        else:
            consistency = 0.7  # Default
        
        # Grounding: check for specific claims vs generic statements
        specific_markers = re.findall(r'\b\d+\b|[A-Z][a-z]+ [A-Z][a-z]+|\b[12][0-9]{3}\b', response)
        grounding = min(1.0, len(specific_markers) * 0.1 + 0.3)
        
        # Factuality: detect fabrication patterns
        fabrication_patterns = [
            r'as (?:an|a) expert',
            r'I (?:have|had) (?:analyzed|examined|reviewed)',
            r'the (?:study|research|report) shows',
            r'(?:recent|latest) (?:findings|research|studies)',
        ]
        fabrication_count = sum(1 for pattern in fabrication_patterns 
                               if re.search(pattern, response, re.IGNORECASE))
        
        # High fabrication indicators reduce factuality
        factuality = max(0.0, 0.8 - (fabrication_count * 0.2))
        
        return cls(
            confidence=confidence,
            consistency=consistency,
            grounding=grounding,
            factuality=factuality
        )


@dataclass
class InteractionTurn:
    """Single turn in interaction history"""
    prompt: str
    response: str
    risk_score: float
    timestamp: int


class ControlProbeType1:
    """Type-1 Control Probe: Inference-local admissibility gating"""
    
    def __init__(self, config):
        """Initialize Type-1 probe
        
        Args:
            config: ControlProbeConfig instance
        """
        self.tau = config.tau
    
    @staticmethod
    def _estimate_prompt_risk(prompt: str) -> float:
        """Estimate prompt-only risk for inadmissible commitments."""
        if not prompt:
            return 0.0
        
        prompt_lower = prompt.lower()
        risk = 0.0
        
        # Future or time-sensitive factual requests
        if re.search(r'\b(202[4-9]|203\d)\b', prompt_lower):
            risk = max(risk, 0.7)
        if any(term in prompt_lower for term in ["current", "latest", "today", "right now", "just announced"]):
            risk = max(risk, 0.5)
        
        # Missing context references
        missing_context_terms = [
            "this code", "the code", "the file", "the document", "the image",
            "the chart", "the table", "the dataset", "the log", "uploaded"
        ]
        if any(term in prompt_lower for term in missing_context_terms):
            risk = max(risk, 0.6)
        
        # Ambiguous directives
        if re.search(r'\bwhat (?:should|do) i do\b', prompt_lower):
            risk = max(risk, 0.6)
        if "ambiguous" in prompt_lower:
            risk = max(risk, 0.7)
        
        return min(1.0, risk)
        
    def evaluate(
        self, 
        response: str, 
        context: str = "",
        coherence_metrics=None
    ) -> Tuple[CommitmentDecision, float, Dict]:
        """Evaluate if commitment is admissible
        
        Args:
            response: Candidate response
            context: Original context/prompt
            coherence_metrics: CoherenceMetrics from ECR (optional)
            
        Returns:
            (decision, sigma, debug_info)
        """
        # Compute evaluative support
        signal = AdmissibilitySignal.from_response(response, context, coherence_metrics)
        sigma_raw = signal.compute_sigma()
        
        # Adjust for prompt-only risk signals (inadmissible or underspecified prompts)
        prompt_risk = self._estimate_prompt_risk(context)
        sigma = min(sigma_raw, 1.0 - prompt_risk) if prompt_risk > 0 else sigma_raw
        
        # Decision: PASS if σ(z) ≥ τ, else BLOCK
        decision = CommitmentDecision.PASS if sigma >= self.tau else CommitmentDecision.BLOCK
        
        debug_info = {
            'sigma': sigma,
            'sigma_raw': sigma_raw,
            'tau': self.tau,
            'prompt_risk': prompt_risk,
            'confidence': signal.confidence,
            'consistency': signal.consistency,
            'grounding': signal.grounding,
            'factuality': signal.factuality,
            'admissible': decision == CommitmentDecision.PASS
        }
        
        if decision == CommitmentDecision.BLOCK:
            print(f"[CP Type-1] BLOCKED: σ={sigma:.3f} < τ={self.tau:.3f}")
        else:
            print(f"[CP Type-1] PASSED: σ={sigma:.3f} ≥ τ={self.tau:.3f}")
        
        return decision, sigma, debug_info
    
    def generate_blocked_response(self, original_prompt: str, reason: Dict) -> str:
        """Generate appropriate response when blocking
        
        Args:
            original_prompt: Original user prompt
            reason: Debug info explaining why blocked
            
        Returns:
            Honest, helpful response explaining limitations
        """
        sigma = reason['sigma']
        
        # Analyze what went wrong
        issues = []
        if reason['factuality'] < 0.5:
            issues.append("I cannot verify the factual accuracy of my response")
        if reason['grounding'] < 0.4:
            issues.append("I lack sufficient grounding to commit to an answer")
        if reason['confidence'] < 0.4:
            issues.append("My confidence in this response is too low")
        if reason['consistency'] < 0.5:
            issues.append("The response shows internal inconsistency")
        
        # Build response
        response_parts = [
            "I cannot provide a confident response to this query.",
            ""
        ]
        
        if issues:
            response_parts.append("Specific concerns:")
            for issue in issues:
                response_parts.append(f"• {issue}")
            response_parts.append("")
        
        # Provide helpful alternative
        response_parts.extend([
            "How I can help:",
            "• Provide more context or background information",
            "• Rephrase the question with more specific details",
            "• Break down the question into smaller, more concrete parts",
            "• Clarify what you're looking to understand or accomplish"
        ])
        
        return "\n".join(response_parts)


class ControlProbeType2:
    """Type-2 Control Probe: Interaction-level monitoring"""
    
    def __init__(self, config):
        """Initialize Type-2 probe
        
        Args:
            config: ControlProbeConfig instance
        """
        self.Theta = config.Theta
        self.max_history = config.max_history_turns
        self.history: List[InteractionTurn] = []
        
    def add_turn(self, prompt: str, response: str, risk_score: float):
        """Add turn to interaction history
        
        Args:
            prompt: User prompt
            response: System response
            risk_score: Risk score for this turn (e.g., from IFCS R(z))
        """
        turn = InteractionTurn(
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
        """Detect if positions shift without new evidence
        
        Returns:
            (drift_detected, drift_score)
        """
        if len(self.history) < 3:
            return False, 0.0
        
        # Check for stance reversals
        # Simplified: look for negation patterns between turns
        reversals = 0
        for i in range(1, len(self.history)):
            prev_response = self.history[i-1].response.lower()
            curr_response = self.history[i].response.lower()
            
            # Check for opposite positions
            if ('yes' in prev_response and 'no' in curr_response) or \
               ('no' in prev_response and 'yes' in curr_response) or \
               ('correct' in prev_response and 'incorrect' in curr_response):
                reversals += 1
        
        drift_score = reversals / len(self.history)
        return drift_score > 0.3, drift_score
    
    def detect_sycophancy(self) -> Tuple[bool, float]:
        """Detect if positions align with user pressure
        
        Returns:
            (sycophancy_detected, sycophancy_score)
        """
        if len(self.history) < 3:
            return False, 0.0
        
        # Check if system progressively weakens stance in response to user disagreement
        disagreement_markers = ['but', 'however', 'disagree', 'wrong', 'incorrect', 
                               'not true', "don't think"]
        
        weakening_pattern = 0
        for i in range(1, len(self.history)):
            user_disagreed = any(marker in self.history[i].prompt.lower() 
                                for marker in disagreement_markers)
            
            if user_disagreed:
                # Check if system weakened stance
                prev_certainty = self._estimate_certainty(self.history[i-1].response)
                curr_certainty = self._estimate_certainty(self.history[i].response)
                
                if curr_certainty < prev_certainty - 0.15:
                    weakening_pattern += 1
        
        sycophancy_score = weakening_pattern / len(self.history)
        return sycophancy_score > 0.2, sycophancy_score
    
    def _estimate_certainty(self, response: str) -> float:
        """Estimate certainty level of response"""
        certainty_markers = ['definitely', 'certainly', 'clearly', 'obviously', 'always']
        uncertainty_markers = ['might', 'could', 'possibly', 'perhaps', 'maybe']
        
        cert_count = sum(1 for marker in certainty_markers if marker in response.lower())
        uncert_count = sum(1 for marker in uncertainty_markers if marker in response.lower())
        
        return 0.5 + (cert_count * 0.1) - (uncert_count * 0.1)
    
    def evaluate(self) -> Tuple[CommitmentDecision, Dict]:
        """Evaluate if interaction should halt or reset
        
        Returns:
            (decision, debug_info)
        """
        if len(self.history) < 2:
            # Need at least 2 turns to detect patterns
            return CommitmentDecision.PASS, {'reason': 'insufficient_history'}
        
        # Compute cumulative risk
        R_cum = self.compute_cumulative_risk()
        
        # Detect drift patterns
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
            print(f"[CP Type-2] HALT: R_cum={R_cum:.3f} ≥ Θ={self.Theta:.3f}")
            return CommitmentDecision.HALT, debug_info
        
        if semantic_drift or sycophancy:
            reason = []
            if semantic_drift:
                reason.append(f"semantic drift (score={drift_score:.2f})")
            if sycophancy:
                reason.append(f"sycophancy (score={syc_score:.2f})")
            
            print(f"[CP Type-2] RESET: {', '.join(reason)}")
            return CommitmentDecision.RESET, debug_info
        
        return CommitmentDecision.PASS, debug_info
    
    def generate_halt_response(self, reason: Dict) -> str:
        """Generate response when halting interaction
        
        Args:
            reason: Debug info explaining why halted
            
        Returns:
            Explanation of halt
        """
        response_parts = [
            "⚠ I need to pause our conversation.",
            ""
        ]

        # Check if R_cum exists (might be insufficient history scenario)
        if 'R_cum' in reason and 'Theta' in reason and reason['R_cum'] >= reason['Theta']:
            response_parts.extend([
                f"The cumulative commitment risk across our conversation has exceeded safe thresholds "
                f"(R_cum={reason['R_cum']:.2f} ≥ Θ={reason['Theta']:.2f}).",
                ""
            ])
        
        if reason.get('semantic_drift'):
            response_parts.extend([
                "I've noticed my positions have shifted across turns without new evidence. "
                "This suggests I may be providing inconsistent guidance.",
                ""
            ])
        
        if reason.get('sycophancy'):
            response_parts.extend([
                "I've been adjusting my responses based on conversational pressure rather than evidence. "
                "This is inappropriate.",
                ""
            ])
        
        response_parts.extend([
            "To maintain reliability, I should:",
            "• Reset to evidence-based positions",
            "• Clarify what I can and cannot confidently claim",
            "• Acknowledge areas of uncertainty honestly"
        ])
        
        return "\n".join(response_parts)
    
    def reset(self):
        """Reset interaction history"""
        print(f"[CP Type-2] Resetting interaction history ({len(self.history)} turns cleared)")
        self.history.clear()
