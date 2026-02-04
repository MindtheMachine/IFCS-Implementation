"""
Three-Gate Inference-Time Governance Pipeline - CORRECTED ARCHITECTURE
ECR - CP-1 - IFCS (with CP-2 parallel monitoring)

- ARCHITECTURAL INVARIANTS:
- ECR: Pure selection (no blocking)
- CP-1: Admissibility gate (binary pass/block) 
- IFCS: Commitment shaping (non-blocking)
- CP-2: PARALLEL interaction monitoring (cumulative risk tracking)

- PROHIBITIONS:
- No learning/training
- No benchmarks  
- No global confidence
- No regex/keyword matching
- No cross-gate signal leakage
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time


class PipelineDecision(Enum):
    """Pipeline decisions"""
    PASS = "pass"
    BLOCK = "block"  
    HALT = "halt"
    RESET = "reset"


@dataclass
class PipelineResult:
    """Complete pipeline result"""
    final_response: str
    decision: PipelineDecision
    ecr_metrics: Dict
    cp1_metrics: Dict
    ifcs_metrics: Dict
    cp2_metrics: Dict
    processing_time_ms: float


# ============================================================================
# STAGE 1: ECR - Evaluative Coherence Regulation (Pure Selection)
# ============================================================================

def ecr_select(candidates: List[str], prompt: str) -> Tuple[str, Dict]:
    """
    ECR Stage: Select most internally coherent candidate
    
    ECR Signals (ECR-only):
    - Internal contradiction indicators
    - Paraphrase invariance  
    - Summary-original agreement
    - Conclusion stability
    
    Invariant: ECR is pure selection, never blocks
    """
    if not candidates:
        return "", {'error': 'no_candidates'}
    
    if len(candidates) == 1:
        return candidates[0], {'selected_idx': 0, 'coherence_score': 0.8}
    
    candidate_scores = []
    
    for i, candidate in enumerate(candidates):
        from semantic_signal_framework import unified_semantic_estimator

        signals = unified_semantic_estimator.estimate_semantic_signals(candidate, prompt)
        coherence_score = signals.coherence
        candidate_scores.append((i, coherence_score))
    
    # Select candidate with maximum coherence
    best_idx, best_score = max(candidate_scores, key=lambda x: x[1])
    
    metrics = {
        'selected_idx': best_idx,
        'coherence_score': best_score,
        'num_candidates': len(candidates),
        'all_scores': [score for _, score in candidate_scores]
    }
    
    return candidates[best_idx], metrics


# ============================================================================
# STAGE 2: Control Probe Type-1 - Admissibility Gate (Binary)
# ============================================================================

def control_probe_pre(response: str, prompt: str = "", tau: float = 0.4) -> Tuple[PipelineDecision, float, Dict]:
    """
    Control Probe Type-1: Admissibility gate

    CP-1 Signals (CP-1-only):
    - Claim count vs prompt-supported facts
    - Unsupported entity density
    - Evidence presence ratio
    - Self-reported uncertainty vs asserted claims

    Invariant: CP-1 blocks only on groundability, never on tone/authority
    """
    if not response or len(response.strip()) < 5:
        return PipelineDecision.BLOCK, 0.0, {'reason': 'empty_response'}

    from signal_estimation import signal_estimator, compute_text_stats

    response_stats = compute_text_stats(response)
    prompt_stats = compute_text_stats(prompt)

    confidence = signal_estimator.estimate_epistemic_certainty(response)
    grounding = 1.0 - signal_estimator.estimate_evidential_risk(response, prompt)
    consistency = 1.0 - min(1.0, abs(response_stats["length_norm"] - prompt_stats["length_norm"]))
    scope = signal_estimator.estimate_scope_breadth(response)

    sigma = (
        grounding * 0.35 +
        confidence * 0.30 +
        consistency * 0.20 +
        (1.0 - scope) * 0.15
    )

    # Decision: PASS if ?(z*) ? ?, else BLOCK
    decision = PipelineDecision.PASS if sigma >= tau else PipelineDecision.BLOCK

    debug_info = {
        'sigma': sigma,
        'tau': tau,
        'signals': {
            'grounding': grounding,
            'confidence': confidence,
            'consistency': consistency,
            'scope_inverse': 1.0 - scope
        },
        'admissible': decision == PipelineDecision.PASS
    }

    return decision, sigma, debug_info


# ============================================================================
# STAGE 3: IFCS - Commitment Shaping (Non-blocking, Fuzzy Logic)
# ============================================================================

def fuzzify(value: float) -> Dict[str, float]:
    """Fuzzify numeric signal into LOW/MEDIUM/HIGH membership"""
    low = max(0.0, min(1.0, (0.5 - value) / 0.5)) if value <= 0.5 else 0.0
    high = max(0.0, min(1.0, (value - 0.5) / 0.5)) if value >= 0.5 else 0.0
    medium = 1.0 - max(low, high)
    
    return {'LOW': low, 'MEDIUM': medium, 'HIGH': high}


def apply_fuzzy_rules(A: Dict, E: Dict, S: Dict, U: Dict) -> Dict[str, float]:
    """
    Apply IFCS fuzzy rules
    
    Fuzzy Rules:
    - IF A is HIGH and E is LOW - risk HIGH
    - IF S is HIGH and U is HIGH - risk HIGH  
    - IF E is HIGH - risk LOW
    """
    # Rule 1: IF A is HIGH and E is LOW - risk HIGH
    rule1_strength = min(A['HIGH'], E['HIGH'])  # E inverted: HIGH E = LOW risk
    
    # Rule 2: IF S is HIGH and U is HIGH - risk HIGH
    rule2_strength = min(S['HIGH'], U['HIGH'])
    
    # Rule 3: IF E is HIGH - risk LOW  
    rule3_strength = E['HIGH']
    
    # Aggregate rules
    risk_high = max(rule1_strength, rule2_strength)
    risk_low = rule3_strength
    risk_medium = 1.0 - max(risk_high, risk_low)
    
    return {
        'LOW': max(0.0, risk_low),
        'MEDIUM': max(0.0, risk_medium), 
        'HIGH': max(0.0, risk_high)
    }


def defuzzify(risk_membership: Dict[str, float]) -> float:
    """Defuzzify risk membership to scalar R(z*) - [0,1]"""
    numerator = (
        risk_membership['LOW'] * 0.25 +
        risk_membership['MEDIUM'] * 0.5 +
        risk_membership['HIGH'] * 0.75
    )
    denominator = sum(risk_membership.values())
    
    return numerator / denominator if denominator > 0 else 0.0


def compute_R_ifcs(response: str, sigma: float, rho: float = 0.4, kappa: int = 1) -> Tuple[str, float, Dict]:
    """
    IFCS Stage: Compute R(z*) and apply commitment shaping
    
    IFCS Signals (IFCS-only):
    - Assertion strength: Declarative claims / total sentences
    - Evidence sufficiency: Evidence units per claim  
    - Scope breadth: Generalized vs qualified claims
    - Authority posture: Prescriptive stance density
    
    Fixed Firing Condition: -(z*) - - - R(z*) > - - -(z*) = 1
    
    Invariant: IFCS never blocks, only shapes
    """
    if not response or len(response.strip()) < 5:
        return response, 0.0, {'reason': 'empty_response'}
    
    # Extract IFCS signals (statistical)
    from signal_estimation import signal_estimator

    assertion_strength = signal_estimator.estimate_assertion_strength(response)
    evidence_sufficiency = 1.0 - signal_estimator.estimate_evidential_risk(response)
    scope_breadth = signal_estimator.estimate_scope_breadth(response)
    authority_posture = signal_estimator.estimate_authority_posture(response)

    signals = {
        'assertion_strength': assertion_strength,
        'evidence_sufficiency': evidence_sufficiency,
        'scope_breadth': scope_breadth,
        'authority_posture': authority_posture
    }
    
    # Fuzzification (IFCS ONLY)
    A = fuzzify(assertion_strength)
    E_inverted = fuzzify(1.0 - evidence_sufficiency)  # Invert for risk
    S = fuzzify(scope_breadth)
    U = fuzzify(authority_posture)
    
    # Apply fuzzy rules
    risk_membership = apply_fuzzy_rules(A, E_inverted, S, U)
    
    # Defuzzification
    R_score = defuzzify(risk_membership)
    
    # Fixed Firing Condition: -(z*) - - - R(z*) > - - -(z*) = 1
    tau = 0.4  # Default CP-1 threshold
    should_shape = sigma >= tau and R_score > rho and kappa == 1
    
    shaped_response = response
    if should_shape:
        # Apply commitment shaping transformations
        shaped_response = apply_commitment_shaping(response, signals)
    
    debug_info = {
        'signals': signals,
        'fuzzy_values': {'assertion': A, 'evidence_inverted': E_inverted, 'scope': S, 'authority': U},
        'risk_membership': risk_membership,
        'R_score': R_score,
        'sigma': sigma,
        'rho': rho,
        'kappa': kappa,
        'should_shape': should_shape,
        'firing_condition': f"-={sigma:.3f} - -={tau:.3f} - R={R_score:.3f} > -={rho:.3f} - -={kappa}"
    }
    
    return shaped_response, R_score, debug_info


def apply_commitment_shaping(text: str, signals: Dict[str, float]) -> str:
    """
    Apply commitment shaping transformations

    Invariant: Only attenuate commitment markers, never add/remove propositions
    """
    shaped = text

    # Rule 1: Weaken universal claims if scope_breadth is high
    if signals['scope_breadth'] >= 0.4:
        prefix = "In many cases, "
        stripped = shaped.lstrip()
        if not stripped.lower().startswith(prefix.lower()):
            leading_ws = shaped[: len(shaped) - len(stripped)]
            shaped = f"{leading_ws}{prefix}{stripped}"

    # Rule 2: Attenuate authority if authority_posture is high
    if signals['authority_posture'] >= 0.4:
        prefix = "One possible approach is: "
        stripped = shaped.lstrip()
        if not stripped.lower().startswith(prefix.lower()):
            leading_ws = shaped[: len(shaped) - len(stripped)]
            shaped = f"{leading_ws}{prefix}{stripped}"

    # Rule 3: Add conditional framing if evidence_sufficiency is low
    if signals['evidence_sufficiency'] < 0.3:
        suffix = " Details can vary by context."
        if "details can vary by context" not in shaped.lower():
            shaped = f"{shaped.rstrip()}{suffix}"

    return shaped


# ============================================================================
# CONTROL PROBE TYPE-2: Parallel Interaction Monitor
# ============================================================================

@dataclass
class InteractionTurn:
    """Single interaction turn"""
    prompt: str
    response: str
    risk_score: float
    timestamp: float


class ControlProbeType2:
    """
    Control Probe Type-2: PARALLEL interaction monitoring
    
    Runs in parallel to track cumulative risk across conversation
    When CP-2 fires (HALT/RESET), it gates subsequent requests until user changes topic
    """
    
    def __init__(self, theta: float = 2.0, max_history: int = 10):
        self.theta = theta
        self.max_history = max_history
        self.history: List[InteractionTurn] = []
        
        # Topic gating state (activated when CP-2 fires)
        self.awaiting_new_topic = False
        self.last_topic_prompt = ""
        self.pending_decision: Optional[PipelineDecision] = None
        
    def add_turn(self, prompt: str, response: str, risk_score: float):
        """Add turn to interaction history"""
        turn = InteractionTurn(
            prompt=prompt,
            response=response, 
            risk_score=risk_score,
            timestamp=time.time()
        )
        
        self.history.append(turn)
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def compute_cumulative_risk(self) -> float:
        """Compute R_cum(H) = - R(z_i)"""
        return sum(turn.risk_score for turn in self.history)
    
    def evaluate(self) -> Tuple[PipelineDecision, Dict]:
        """Evaluate interaction-level risk"""
        if len(self.history) < 2:
            return PipelineDecision.PASS, {'reason': 'insufficient_history'}
        
        R_cum = self.compute_cumulative_risk()
        
        debug_info = {
            'R_cum': R_cum,
            'theta': self.theta,
            'num_turns': len(self.history),
            'recent_risks': [turn.risk_score for turn in self.history[-3:]]
        }
        
        if R_cum >= self.theta:
            # Activate topic gate when halting
            if self.history:
                self._activate_topic_gate(self.history[-1].prompt, PipelineDecision.HALT)
            return PipelineDecision.HALT, debug_info
        
        return PipelineDecision.PASS, debug_info
    
    def should_block_prompt(self, prompt: str) -> Tuple[bool, Optional[str], Optional[PipelineDecision]]:
        """
        Check if prompt should be blocked due to topic gating
        
        When CP-2 fires (HALT/RESET), subsequent prompts are blocked until user changes topic
        
        Returns:
            (should_block, block_message, decision_type)
        """
        if not self.awaiting_new_topic:
            return False, None, None

        if self._is_new_topic(prompt):
            # User changed topic - clear the gate AND reset cumulative risk
            self.awaiting_new_topic = False
            self.pending_decision = None
            self.last_topic_prompt = ""
            # Reset history to allow fresh start on new topic
            self.history = []
            return False, None, None

        # Still on same topic - block with appropriate message
        message = self._generate_topic_gate_message(self.pending_decision or PipelineDecision.HALT)
        return True, message, (self.pending_decision or PipelineDecision.HALT)
    
    def _activate_topic_gate(self, prompt: str, decision: PipelineDecision):
        """Activate topic gating after CP-2 fires"""
        self.awaiting_new_topic = True
        self.pending_decision = decision
        self.last_topic_prompt = prompt or ""
    
    def _is_new_topic(self, prompt: str) -> bool:
        """Check if prompt represents a new topic using semantic similarity"""
        if not self.last_topic_prompt:
            return True

        from semantic_signal_framework import unified_semantic_estimator
        similarity = unified_semantic_estimator.similarity_engine.compute_semantic_similarity(
            prompt, self.last_topic_prompt
        )

        # Consider it a new topic if similarity is low (more sensitive threshold)
        return similarity < 0.2
    
    def _generate_topic_gate_message(self, decision: PipelineDecision) -> str:
        """Generate appropriate message when topic gate is active"""
        if decision == PipelineDecision.HALT:
            return (
                "-- I've reached my limit for commitment-heavy responses in this conversation thread. "
                "To continue, please start a new line of inquiry or change the topic. "
                "I'm happy to help with different questions or a fresh perspective on other subjects."
            )
        else:  # RESET
            return (
                "-- I need to step back from this line of discussion. "
                "Please try a different topic or approach. "
                "I'm available to help with other questions or subjects."
            )


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

class CorrectedGovernancePipeline:
    """
    Corrected Three-Gate Pipeline
    
    Sequential: ECR - CP-1 - IFCS
    Parallel: CP-2 (interaction monitoring)
    """
    
    def __init__(self, cp2_theta: float = 2.0):
        self.cp2 = ControlProbeType2(theta=cp2_theta)
        
    def process(
        self,
        prompt: str,
        candidates: List[str],
        context: str = ""
    ) -> PipelineResult:
        """Process through corrected three-gate pipeline"""
        start_time = time.perf_counter()
        
        # Check CP-2 topic gate first (if active)
        should_block, block_message, gate_decision = self.cp2.should_block_prompt(prompt)
        if should_block:
            return PipelineResult(
                final_response=block_message,
                decision=gate_decision,
                ecr_metrics={},
                cp1_metrics={},
                ifcs_metrics={},
                cp2_metrics={'topic_gate_active': True, 'awaiting_new_topic': True},
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )
        
        # Check CP-2 parallel monitor for cumulative risk
        cp2_decision, cp2_metrics = self.cp2.evaluate()
        if cp2_decision == PipelineDecision.HALT:
            return PipelineResult(
                final_response=(
                    "-- I've reached my limit for commitment-heavy responses in this conversation. "
                    f"Cumulative risk: {cp2_metrics['R_cum']:.2f} - {cp2_metrics['theta']:.2f}. "
                    "Please start a new line of inquiry or change the topic."
                ),
                decision=cp2_decision,
                ecr_metrics={},
                cp1_metrics={},
                ifcs_metrics={},
                cp2_metrics=cp2_metrics,
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )
        
        # STAGE 1: ECR - Selection
        selected_candidate, ecr_metrics = ecr_select(candidates, prompt)
        
        # STAGE 2: CP-1 - Admissibility
        cp1_decision, sigma, cp1_metrics = control_probe_pre(selected_candidate, prompt)
        
        if cp1_decision == PipelineDecision.BLOCK:
            return PipelineResult(
                final_response="I cannot provide a confident response to this query due to insufficient grounding.",
                decision=cp1_decision,
                ecr_metrics=ecr_metrics,
                cp1_metrics=cp1_metrics,
                ifcs_metrics={},
                cp2_metrics=cp2_metrics,
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )
        
        # STAGE 3: IFCS - Commitment Shaping
        shaped_response, R_score, ifcs_metrics = compute_R_ifcs(selected_candidate, sigma)
        
        # Update CP-2 parallel monitor
        self.cp2.add_turn(prompt, shaped_response, R_score)
        
        return PipelineResult(
            final_response=shaped_response,
            decision=PipelineDecision.PASS,
            ecr_metrics=ecr_metrics,
            cp1_metrics=cp1_metrics,
            ifcs_metrics=ifcs_metrics,
            cp2_metrics=cp2_metrics,
            processing_time_ms=(time.perf_counter() - start_time) * 1000
        )


# ============================================================================
# DEMO USAGE
# ============================================================================

def demo_corrected_pipeline():
    """Demonstrate the corrected pipeline architecture"""
    
    print("="*80)
    print("CORRECTED THREE-GATE GOVERNANCE PIPELINE DEMO")
    print("Architecture: ECR - CP-1 - IFCS (with CP-2 parallel)")
    print("="*80)
    
    pipeline = CorrectedGovernancePipeline()
    
    # Test cases
    test_cases = [
        {
            'prompt': 'What framework should I use for my web project?',
            'candidates': [
                'You should definitely use React because it is the best framework available.',
                'Common frameworks include React, Vue, and Angular, each with different strengths.',
                'React is popular for component-based development and has good community support.'
            ]
        },
        {
            'prompt': 'What are the current best practices for web development?',
            'candidates': [
                'Best practices include responsive design, accessibility, performance optimization, and security.',
                'You must follow these practices: use HTTPS, validate inputs, optimize images.',
                'Common practices involve using modern frameworks, testing, and version control.'
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- TEST CASE {i} ---")
        print(f"Prompt: {test_case['prompt']}")
        print(f"Candidates: {len(test_case['candidates'])}")
        
        result = pipeline.process(
            prompt=test_case['prompt'],
            candidates=test_case['candidates']
        )
        
        print(f"\nResult:")
        print(f"Decision: {result.decision.value}")
        print(f"Final Response: {result.final_response}")
        print(f"Processing Time: {result.processing_time_ms:.2f}ms")
        
        print(f"\nPipeline Metrics:")
        print(f"ECR: Selected candidate {result.ecr_metrics.get('selected_idx', 'N/A')} "
              f"(coherence: {result.ecr_metrics.get('coherence_score', 0):.3f})")
        print(f"CP-1: -={result.cp1_metrics.get('sigma', 0):.3f}, "
              f"admissible={result.cp1_metrics.get('admissible', False)}")
        print(f"IFCS: R={result.ifcs_metrics.get('R_score', 0):.3f}, "
              f"shaped={result.ifcs_metrics.get('should_shape', False)}")
        print(f"CP-2: R_cum={result.cp2_metrics.get('R_cum', 0):.3f}")


if __name__ == "__main__":
    demo_corrected_pipeline()
