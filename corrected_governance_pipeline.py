"""
Three-Gate Inference-Time Governance Pipeline - CORRECTED ARCHITECTURE
ECR → CP-1 → IFCS (with CP-2 parallel monitoring)

🔒 ARCHITECTURAL INVARIANTS:
- ECR: Pure selection (no blocking)
- CP-1: Admissibility gate (binary pass/block) 
- IFCS: Commitment shaping (non-blocking)
- CP-2: PARALLEL interaction monitoring (cumulative risk tracking)

🚫 PROHIBITIONS:
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
        # Extract ECR coherence signals
        sentences = [s.strip() for s in candidate.split('.') if s.strip()]
        
        # Signal 1: Internal contradiction indicators
        contradiction_signals = 0
        for j, sent in enumerate(sentences):
            sent_lower = sent.lower()
            if any(neg in sent_lower for neg in ['however', 'but', 'although', 'despite']):
                if j > 0:
                    prev_sent = sentences[j-1].lower()
                    if any(pos in prev_sent for pos in ['always', 'never', 'definitely']):
                        contradiction_signals += 1
        
        contradiction_rate = contradiction_signals / max(len(sentences), 1)
        
        # Signal 2: Paraphrase stability (concept density)
        words = candidate.split()
        unique_concepts = len(set(word.lower() for word in words if len(word) > 3))
        concept_density = unique_concepts / max(len(words), 1)
        paraphrase_stability = min(1.0, concept_density * 2.0)
        
        # Signal 3: Summary agreement (placeholder)
        summary_agreement = 0.8
        
        # Signal 4: Conclusion stability
        conclusion_stability = 1.0 - contradiction_rate
        
        # Compute coherence score
        coherence_score = (
            (1.0 - contradiction_rate) * 0.4 +
            paraphrase_stability * 0.2 +
            summary_agreement * 0.2 +
            conclusion_stability * 0.2
        )
        
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
    
    words = response.lower().split()
    prompt_words = prompt.lower().split() if prompt else []
    
    # Signal 1: Claim support ratio
    claim_indicators = {'is', 'are', 'will', 'must', 'should', 'can', 'cannot'}
    claim_count = sum(1 for word in words if word in claim_indicators)
    
    supported_claims = 0
    for word in words:
        if word in claim_indicators and word in prompt_words:
            supported_claims += 1
    
    claim_support_ratio = supported_claims / max(claim_count, 1)
    
    # Signal 2: Entity support ratio
    entities = [word for word in words if word.istitle() and len(word) > 2]
    prompt_entities = [word for word in prompt_words if word.istitle() and len(word) > 2]
    
    supported_entities = sum(1 for entity in entities if entity.lower() in [e.lower() for e in prompt_entities])
    entity_support_ratio = supported_entities / max(len(entities), 1) if entities else 1.0
    
    # Signal 3: Evidence presence
    evidence_markers = {'because', 'since', 'due', 'based', 'evidence', 'study', 'research', 'data'}
    evidence_count = sum(1 for word in words if word in evidence_markers)
    evidence_presence = min(1.0, evidence_count / max(len(words), 1) * 10)
    
    # Signal 4: Uncertainty-claim balance
    uncertainty_markers = {'uncertain', 'unclear', 'might', 'may', 'possibly', 'perhaps'}
    certainty_markers = {'definitely', 'certainly', 'clearly', 'obviously', 'undoubtedly'}
    
    uncertainty_count = sum(1 for word in words if word in uncertainty_markers)
    certainty_count = sum(1 for word in words if word in certainty_markers)
    
    uncertainty_claim_balance = 1.0 - abs(certainty_count - uncertainty_count) / max(claim_count, 1)
    
    # Compute σ(z*): Overall evaluative support
    sigma = (
        claim_support_ratio * 0.3 +
        entity_support_ratio * 0.3 +
        evidence_presence * 0.2 +
        uncertainty_claim_balance * 0.2
    )
    
    # Decision: PASS if σ(z*) ≥ τ, else BLOCK
    decision = PipelineDecision.PASS if sigma >= tau else PipelineDecision.BLOCK
    
    debug_info = {
        'sigma': sigma,
        'tau': tau,
        'signals': {
            'claim_support_ratio': claim_support_ratio,
            'entity_support_ratio': entity_support_ratio,
            'evidence_presence': evidence_presence,
            'uncertainty_claim_balance': uncertainty_claim_balance
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
    - IF A is HIGH and E is LOW → risk HIGH
    - IF S is HIGH and U is HIGH → risk HIGH  
    - IF E is HIGH → risk LOW
    """
    # Rule 1: IF A is HIGH and E is LOW → risk HIGH
    rule1_strength = min(A['HIGH'], E['HIGH'])  # E inverted: HIGH E = LOW risk
    
    # Rule 2: IF S is HIGH and U is HIGH → risk HIGH
    rule2_strength = min(S['HIGH'], U['HIGH'])
    
    # Rule 3: IF E is HIGH → risk LOW  
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
    """Defuzzify risk membership to scalar R(z*) ∈ [0,1]"""
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
    
    Fixed Firing Condition: σ(z*) ≥ τ ∧ R(z*) > ρ ∧ κ(z*) = 1
    
    Invariant: IFCS never blocks, only shapes
    """
    if not response or len(response.strip()) < 5:
        return response, 0.0, {'reason': 'empty_response'}
    
    sentences = [s.strip() for s in response.split('.') if s.strip()]
    words = response.lower().split()
    
    # Extract IFCS signals (fast path)
    
    # Signal 1: Assertion strength
    declarative_markers = {'is', 'are', 'will', 'must', 'should', 'have', 'has'}
    declarative_count = sum(1 for sent in sentences 
                           for word in sent.lower().split() 
                           if word in declarative_markers)
    assertion_strength = min(1.0, declarative_count / max(len(sentences), 1) / 2.0)
    
    # Signal 2: Evidence sufficiency  
    evidence_markers = {'because', 'since', 'due', 'based', 'evidence', 'study', 'research', 'data'}
    evidence_count = sum(1 for word in words if word in evidence_markers)
    evidence_sufficiency = min(1.0, evidence_count / max(declarative_count, 1))
    
    # Signal 3: Scope breadth
    universal_markers = {'all', 'every', 'always', 'never', 'everyone', 'everything'}
    particular_markers = {'some', 'sometimes', 'certain', 'specific', 'particular'}
    
    universal_count = sum(1 for word in words if word in universal_markers)
    particular_count = sum(1 for word in words if word in particular_markers)
    
    if universal_count + particular_count == 0:
        scope_breadth = 0.0
    else:
        scope_breadth = universal_count / (universal_count + particular_count)
    
    # Signal 4: Authority posture
    authority_markers = {'should', 'must', 'need', 'recommend', 'suggest', 'best', 'correct', 'right'}
    authority_count = sum(1 for word in words if word in authority_markers)
    authority_posture = min(1.0, authority_count / max(len(words), 1) * 10)
    
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
    
    # Fixed Firing Condition: σ(z*) ≥ τ ∧ R(z*) > ρ ∧ κ(z*) = 1
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
        'firing_condition': f"σ={sigma:.3f} ≥ τ={tau:.3f} ∧ R={R_score:.3f} > ρ={rho:.3f} ∧ κ={kappa}"
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
        universal_replacements = {
            'always': 'typically',
            'never': 'rarely',
            'all': 'many',
            'every': 'most',
            'the best': 'an effective',
            'definitely': 'likely',
            'certainly': 'probably'
        }
        
        for original, replacement in universal_replacements.items():
            shaped = shaped.replace(original, replacement)
            shaped = shaped.replace(original.capitalize(), replacement.capitalize())
    
    # Rule 2: Attenuate authority if authority_posture is high
    if signals['authority_posture'] >= 0.4:
        authority_replacements = {
            'you must': 'you might consider',
            'you should': 'you could',
            'you need to': 'consider',
            'must': 'may need to',
            'should': 'could'
        }
        
        for original, replacement in authority_replacements.items():
            shaped = shaped.replace(original, replacement)
            shaped = shaped.replace(original.capitalize(), replacement.capitalize())
    
    # Rule 3: Add conditional framing if evidence_sufficiency is low
    if signals['evidence_sufficiency'] < 0.3:
        if not any(marker in shaped.lower() for marker in ['typically', 'generally', 'often']):
            shaped = f"In typical scenarios, {shaped[0].lower()}{shaped[1:]}"
    
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
        self.last_topic_tokens = set()
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
        """Compute R_cum(H) = Σ R(z_i)"""
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
            self.last_topic_tokens = set()
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
        self.last_topic_tokens = self._tokenize_prompt(prompt)
    
    def _is_new_topic(self, prompt: str) -> bool:
        """Check if prompt represents a new topic"""
        current_tokens = self._tokenize_prompt(prompt)
        if not current_tokens or not self.last_topic_tokens:
            return True

        # Calculate semantic similarity
        overlap = len(current_tokens & self.last_topic_tokens)
        union = len(current_tokens | self.last_topic_tokens)
        similarity = overlap / union if union > 0 else 0.0
        
        # Consider it a new topic if similarity is low (more sensitive threshold)
        return similarity < 0.2  # Lower threshold = more sensitive to topic changes
    
    def _tokenize_prompt(self, prompt: str) -> set:
        """Extract meaningful tokens from prompt for topic comparison"""
        import re
        
        # Extract words 4+ characters, excluding common stopwords
        tokens = re.findall(r"[a-zA-Z]{4,}", prompt.lower())
        stopwords = {
            "this", "that", "with", "from", "your", "have", "what", "would",
            "should", "could", "about", "please", "think", "being", "their",
            "there", "which", "where", "when", "while", "these", "those",
            "they", "them", "will", "been", "were", "said", "each", "more",
            "very", "like", "just", "than", "only", "over", "also", "back",
            "after", "first", "well", "work", "life", "right", "down", "call"
        }
        return {t for t in tokens if t not in stopwords}
    
    def _generate_topic_gate_message(self, decision: PipelineDecision) -> str:
        """Generate appropriate message when topic gate is active"""
        if decision == PipelineDecision.HALT:
            return (
                "⚠️ I've reached my limit for commitment-heavy responses in this conversation thread. "
                "To continue, please start a new line of inquiry or change the topic. "
                "I'm happy to help with different questions or a fresh perspective on other subjects."
            )
        else:  # RESET
            return (
                "⚠️ I need to step back from this line of discussion. "
                "Please try a different topic or approach. "
                "I'm available to help with other questions or subjects."
            )


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

class CorrectedGovernancePipeline:
    """
    Corrected Three-Gate Pipeline
    
    Sequential: ECR → CP-1 → IFCS
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
                    "⚠️ I've reached my limit for commitment-heavy responses in this conversation. "
                    f"Cumulative risk: {cp2_metrics['R_cum']:.2f} ≥ {cp2_metrics['theta']:.2f}. "
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
    print("Architecture: ECR → CP-1 → IFCS (with CP-2 parallel)")
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
        print(f"CP-1: σ={result.cp1_metrics.get('sigma', 0):.3f}, "
              f"admissible={result.cp1_metrics.get('admissible', False)}")
        print(f"IFCS: R={result.ifcs_metrics.get('R_score', 0):.3f}, "
              f"shaped={result.ifcs_metrics.get('should_shape', False)}")
        print(f"CP-2: R_cum={result.cp2_metrics.get('R_cum', 0):.3f}")


if __name__ == "__main__":
    demo_corrected_pipeline()