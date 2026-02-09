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
    """Evaluative support signal -(z)"""
    confidence: float  # 0-1
    consistency: float  # 0-1
    grounding: float  # 0-1
    factuality: float  # 0-1
    
    def compute_sigma(self) -> float:
        """Compute overall evaluative support"""
        return (self.confidence + self.consistency + self.grounding + self.factuality) / 4.0
    
    @classmethod
    def from_response(cls, response: str, context: str = "", coherence_metrics=None) -> 'AdmissibilitySignal':
        """Estimate admissibility signal from response using signal estimation"""
        
        # Use signal estimation instead of text-matching heuristics
        from signal_estimation import signal_estimator
        
        # Confidence: inverse of uncertainty (using signal estimation)
        certainty_signal = signal_estimator.estimate_epistemic_certainty(response)
        confidence = certainty_signal  # Direct mapping
        
        # Consistency: from coherence metrics if available
        if coherence_metrics:
            consistency = 1.0 - coherence_metrics.CR  # Inverse of contradiction rate
        else:
            consistency = 0.7  # Default
        
        # Grounding: use signal estimation for evidential grounding
        evidential_risk = signal_estimator.estimate_evidential_risk(response, context)
        grounding = 1.0 - evidential_risk  # Inverse of evidential risk
        
        # Factuality: use authority signal as proxy for fabrication risk
        authority_signal = signal_estimator.estimate_authority_posture(response)
        # High authority without evidence suggests potential fabrication
        factuality = max(0.0, 0.8 - (authority_signal * evidential_risk))
        
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
        """Estimate prompt-only risk using signal estimation."""
        if not prompt:
            return 0.0
        
        # Use semantic signal estimation (no pattern matching)
        from signal_estimation import signal_estimator, compute_text_stats
        from enhanced_signal_estimator import enhanced_signal_estimator
        from semantic_signal_framework import unified_semantic_estimator
        
        # Estimate temporal risk from prompt
        temporal_risk = signal_estimator.estimate_temporal_risk("", prompt)
        
        # Structural risk using semantic intent + fuzzy weighting
        structural_signals = enhanced_signal_estimator.estimate_structural_signals(prompt)
        structural_risk = max(structural_signals.values()) if structural_signals else 0.0
        
        stats = compute_text_stats(prompt)
        specificity = unified_semantic_estimator.estimate_semantic_signals(prompt).risk_specificity
        underspec_score = max(0.0, (1.0 - specificity) * (1.0 - stats["length_norm"]))
        ambiguity_score = max(0.0, (1.0 - specificity) * stats["question_ratio"])
        
        risk = max(temporal_risk, structural_risk, underspec_score * 0.7, ambiguity_score * 0.7)
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
        
        # Decision: PASS if -(z) - -, else BLOCK
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
            print(f"[CP Type-1] BLOCKED: -={sigma:.3f} < -={self.tau:.3f}")
        else:
            print(f"[CP Type-1] PASSED: -={sigma:.3f} - -={self.tau:.3f}")
        
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
                response_parts.append(f"- {issue}")
            response_parts.append("")
        
        # Provide helpful alternative
        response_parts.extend([
            "How I can help:",
            "- Provide more context or background information",
            "- Rephrase the question with more specific details",
            "- Break down the question into smaller, more concrete parts",
            "- Clarify what you're looking to understand or accomplish"
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
        self.awaiting_new_topic = False
        self.last_topic_prompt = ""
        self.pending_decision: Optional[CommitmentDecision] = None
        
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
        """Compute R_cum(H) = - R(z_i)"""
        return sum(turn.risk_score for turn in self.history)
    
    def detect_semantic_drift(self) -> Tuple[bool, float]:
        """Detect if positions shift without new evidence
        
        Returns:
            (drift_detected, drift_score)
        """
        if len(self.history) < 3:
            return False, 0.0
        
        from semantic_signal_framework import unified_semantic_estimator
        
        similarity_engine = unified_semantic_estimator.similarity_engine
        drift_events = 0
        
        for i in range(1, len(self.history)):
            prev_turn = self.history[i-1]
            curr_turn = self.history[i]
            
            # Same topic if prompts are semantically similar
            prompt_similarity = similarity_engine.compute_semantic_similarity(
                prev_turn.prompt, curr_turn.prompt
            )
            response_similarity = similarity_engine.compute_semantic_similarity(
                prev_turn.response, curr_turn.response
            )
            
            # Drift if prompt is same topic but response semantics shift
            if prompt_similarity >= 0.5 and response_similarity <= 0.35:
                drift_events += 1
        
        drift_score = drift_events / len(self.history)
        return drift_score > 0.25, drift_score
    
    def detect_sycophancy(self) -> Tuple[bool, float]:
        """Detect if positions align with user pressure
        
        Returns:
            (sycophancy_detected, sycophancy_score)
        """
        if len(self.history) < 3:
            return False, 0.0
        
        from semantic_signal_framework import unified_semantic_estimator
        
        weakening_pattern = 0
        for i in range(1, len(self.history)):
            user_prompt = self.history[i].prompt
            disagreement_score = unified_semantic_estimator.estimate_explicit_disagreement_signal(user_prompt)
            
            if disagreement_score >= 0.35:
                prev_certainty = self._estimate_certainty(self.history[i-1].response)
                curr_certainty = self._estimate_certainty(self.history[i].response)
                
                if curr_certainty < prev_certainty - 0.15:
                    weakening_pattern += 1
        
        sycophancy_score = weakening_pattern / len(self.history)
        return sycophancy_score > 0.2, sycophancy_score
    
    def _estimate_certainty(self, response: str) -> float:
        """Estimate certainty level of response"""
        from signal_estimation import signal_estimator
        
        return signal_estimator.estimate_epistemic_certainty(response)
    
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
            print(f"[CP Type-2] HALT: R_cum={R_cum:.3f} - -={self.Theta:.3f}")
            self._activate_topic_gate(self.history[-1].prompt, CommitmentDecision.HALT)
            return CommitmentDecision.HALT, debug_info
        
        if semantic_drift or sycophancy:
            reason = []
            if semantic_drift:
                reason.append(f"semantic drift (score={drift_score:.2f})")
            if sycophancy:
                reason.append(f"sycophancy (score={syc_score:.2f})")
            
            print(f"[CP Type-2] RESET: {', '.join(reason)}")
            self._activate_topic_gate(self.history[-1].prompt, CommitmentDecision.RESET)
            return CommitmentDecision.RESET, debug_info
        
        return CommitmentDecision.PASS, debug_info

    def should_block_prompt(self, prompt: str) -> Tuple[bool, Optional[str], Optional[CommitmentDecision]]:
        """Block if continuing the same line of thought after a RESET/HALT."""
        if not self.awaiting_new_topic:
            return False, None, None

        if self._is_new_topic(prompt):
            self.awaiting_new_topic = False
            self.pending_decision = None
            return False, None, None

        message = self.generate_reset_response(self.pending_decision or CommitmentDecision.RESET)
        return True, message, (self.pending_decision or CommitmentDecision.RESET)

    def _activate_topic_gate(self, prompt: str, decision: CommitmentDecision):
        self.awaiting_new_topic = True
        self.pending_decision = decision
        self.last_topic_prompt = prompt or ""

    def _is_new_topic(self, prompt: str) -> bool:
        if not self.last_topic_prompt:
            return True

        from semantic_signal_framework import unified_semantic_estimator
        similarity = unified_semantic_estimator.similarity_engine.compute_semantic_similarity(
            prompt, self.last_topic_prompt
        )
        return similarity < 0.5
    
    def generate_halt_response(self, reason: Dict) -> str:
        """Generate response when halting interaction
        
        Args:
            reason: Debug info explaining why halted
            
        Returns:
            Explanation of halt
        """
        response_parts = [
            "- I need to pause our conversation.",
            ""
        ]

        # Check if R_cum exists (might be insufficient history scenario)
        if 'R_cum' in reason and 'Theta' in reason and reason['R_cum'] >= reason['Theta']:
            response_parts.extend([
                f"The cumulative commitment risk across our conversation has exceeded safe thresholds "
                f"(R_cum={reason['R_cum']:.2f} - -={reason['Theta']:.2f}).",
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
            "- Reset to evidence-based positions",
            "- Clarify what I can and cannot confidently claim",
            "- Acknowledge areas of uncertainty honestly"
        ])
        
        return "\n".join(response_parts)

    def generate_reset_response(self, decision: CommitmentDecision) -> str:
        """Generate response when repeating the same line of thought."""
        response_parts = [
            "-- There is no point in continuing this line of thought.",
            "Please start a new line of inquiry or provide a different angle.",
            "",
            "I will not continue the current thread until you change topics."
        ]
        return "\n".join(response_parts)
    
    def reset(self):
        """Reset interaction history"""
        print(f"[CP Type-2] Resetting interaction history ({len(self.history)} turns cleared)")
        self.history.clear()
        # Also clear topic-gate state so new sessions are not spuriously blocked.
        self.awaiting_new_topic = False
        self.pending_decision = None
        self.last_topic_prompt = ""
