"""
Universal Commitment Regulation Architecture
Implements benchmark-agnostic commitment control following the principle:
"Regulate commitments, not questions"
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class CandidateCommitment:
    """Represents a candidate response with commitment analysis"""
    text: str
    logit_score: float
    commitment_weight: float  # How much uncertainty this candidate collapses
    semantic_invariants: Dict[str, Any]  # Core meaning that must be preserved
    is_commitment_heavy: bool  # Does this make irreversible claims?
    
    def __post_init__(self):
        """Validate commitment structure"""
        if self.commitment_weight < 0 or self.commitment_weight > 1:
            raise ValueError("Commitment weight must be in [0,1]")


@dataclass
class DecisionState:
    """Represents the decision geometry at regulation time"""
    candidates: List[CandidateCommitment]
    selected_candidate: CandidateCommitment
    logit_margin: float  # Confidence gap between top candidates
    has_commitment_reducing_alternative: bool
    evidence_dominance: float  # How much evidence supports the selection
    
    def __post_init__(self):
        """Validate decision state"""
        if not self.candidates:
            raise ValueError("Decision state must have candidates")
        if self.selected_candidate not in self.candidates:
            raise ValueError("Selected candidate must be in candidates list")


class CommitmentAnalyzer(ABC):
    """Abstract base for analyzing commitment structure"""
    
    @abstractmethod
    def analyze_commitment_weight(self, candidate: str) -> float:
        """Analyze how much uncertainty this candidate collapses"""
        pass
    
    @abstractmethod
    def extract_semantic_invariants(self, candidate: str) -> Dict[str, Any]:
        """Extract core semantic content that must be preserved"""
        pass
    
    @abstractmethod
    def is_commitment_heavy(self, candidate: str) -> bool:
        """Check if candidate makes irreversible/global claims"""
        pass


class GenericCommitmentAnalyzer(CommitmentAnalyzer):
    """Generic implementation of commitment analysis"""
    
    def __init__(self):
        # Patterns that indicate high commitment
        self.high_commitment_patterns = [
            r'\b(definitely|certainly|always|never|all|none|every|no)\b',
            r'\b(the answer is|the solution is|this means)\b',
            r'\b(will|must|cannot|impossible)\b',
            r'\b(proven|established|known|fact)\b',
            r'\d+\s*[+\-*/=]\s*\d+',  # Mathematical expressions
            r'\b(equals?|is|are)\b.*\d+',  # Mathematical equality statements
        ]
        
        # Patterns that indicate commitment reduction
        self.commitment_reducing_patterns = [
            r'\b(might|could|possibly|perhaps|maybe)\b',
            r'\b(unclear|uncertain|unknown|ambiguous)\b',
            r'\b(depends|varies|context|more information)\b',
            r'\b(I don\'t know|I\'m not sure|I cannot)\b'
        ]
    
    def analyze_commitment_weight(self, candidate: str) -> float:
        """Analyze commitment weight based on linguistic patterns"""
        import re
        
        high_count = sum(len(re.findall(pattern, candidate.lower())) 
                        for pattern in self.high_commitment_patterns)
        reducing_count = sum(len(re.findall(pattern, candidate.lower())) 
                           for pattern in self.commitment_reducing_patterns)
        
        # Special handling for mathematical expressions
        if re.search(r'\d+\s*[+\-*/=]\s*\d+', candidate):
            high_count += 2  # Mathematical facts are high commitment
        
        # Normalize to [0,1] range
        total_words = len(candidate.split())
        if total_words == 0:
            return 0.5
        
        # Boost commitment score for mathematical expressions
        commitment_score = (high_count - reducing_count) / max(total_words, 1)
        
        # Ensure mathematical facts get high commitment scores
        if re.search(r'\d+\s*[+\-*/=]\s*\d+', candidate):
            commitment_score = max(commitment_score, 0.3)  # Minimum boost for math
        
        return max(0, min(1, 0.5 + commitment_score))
    
    def extract_semantic_invariants(self, candidate: str) -> Dict[str, Any]:
        """Extract core semantic content"""
        # Simplified implementation - in practice would use NLP
        return {
            'factual_claims': self._extract_factual_claims(candidate),
            'entities': self._extract_entities(candidate),
            'relationships': self._extract_relationships(candidate),
            'scope': self._extract_scope(candidate)
        }
    
    def is_commitment_heavy(self, candidate: str) -> bool:
        """Check if candidate makes irreversible claims"""
        commitment_weight = self.analyze_commitment_weight(candidate)
        return commitment_weight > 0.7  # Threshold for "heavy" commitment
    
    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        # Simplified - would use proper NLP in practice
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities"""
        # Simplified - would use NER in practice
        import re
        # Basic pattern for capitalized words (crude entity detection)
        entities = re.findall(r'\b[A-Z][a-z]+\b', text)
        return list(set(entities))
    
    def _extract_relationships(self, text: str) -> List[str]:
        """Extract relationships between entities"""
        # Simplified implementation
        return []
    
    def _extract_scope(self, text: str) -> str:
        """Extract scope of claims (global, local, conditional)"""
        if any(word in text.lower() for word in ['all', 'every', 'always', 'never']):
            return 'global'
        elif any(word in text.lower() for word in ['some', 'sometimes', 'often']):
            return 'partial'
        else:
            return 'local'


class HybridControlProbe:
    """
    Hybrid Control Probe: Implementation's logic + Paper's semantic sophistication
    
    Core Logic (Implementation): commitment_heavy AND NOT has_alternative AND evidence_insufficient
    Semantic Analysis (Paper): σ(z*) based evaluative support estimation
    """
    
    def __init__(self, 
                 stability_threshold: float = 0.3,
                 commitment_threshold: float = 0.6,
                 commitment_analyzer: Optional[CommitmentAnalyzer] = None):
        self.stability_threshold = stability_threshold
        self.commitment_threshold = commitment_threshold
        self.commitment_analyzer = commitment_analyzer or GenericCommitmentAnalyzer()
    
    def should_fire_cp1(self, decision_state: DecisionState, context: str = "") -> Tuple[bool, Dict]:
        """
        Hybrid CP-1 rule: Implementation's logic with Paper's semantic analysis
        
        Returns:
            (should_fire, debug_info)
        """
        candidate = decision_state.selected_candidate
        
        # Use paper's semantic analysis for commitment evaluation
        try:
            from enhanced_control_probes import EnhancedAdmissibilitySignal
            
            # Get sophisticated semantic-based admissibility signal
            signal = EnhancedAdmissibilitySignal.from_response(candidate.text, context)
            sigma = signal.compute_sigma()
            
            # A. Candidate has low evaluative support (paper's approach)
            commitment_heavy = sigma < self.commitment_threshold
            
            debug_info = {
                'sigma': sigma,
                'commitment_threshold': self.commitment_threshold,
                'commitment_heavy': commitment_heavy,
                'has_alternative': decision_state.has_commitment_reducing_alternative,
                'evidence_dominance': decision_state.evidence_dominance,
                'stability_threshold': self.stability_threshold,
                'semantic_signals': {
                    'confidence': signal.confidence,
                    'consistency': signal.consistency,
                    'grounding': signal.grounding,
                    'factuality': signal.factuality,
                    'intent_clarity': signal.intent_clarity,
                    'domain_alignment': signal.domain_alignment
                }
            }
            
        except ImportError:
            # Fallback to simple commitment analysis if enhanced probes not available
            commitment_weight = self.commitment_analyzer.analyze_commitment_weight(candidate.text)
            commitment_heavy = commitment_weight > 0.7
            sigma = 1.0 - commitment_weight  # Invert for consistency
            
            debug_info = {
                'sigma': sigma,
                'commitment_weight': commitment_weight,
                'commitment_heavy': commitment_heavy,
                'has_alternative': decision_state.has_commitment_reducing_alternative,
                'evidence_dominance': decision_state.evidence_dominance,
                'stability_threshold': self.stability_threshold,
                'fallback_mode': True
            }
        
        # B. No alternative candidate reduces commitment (implementation's insight)
        if decision_state.has_commitment_reducing_alternative:
            debug_info['fire_reason'] = "Alternative available - not firing"
            return False, debug_info
        
        # C. Internal evidence does not dominate alternatives (implementation's insight)
        if decision_state.evidence_dominance > self.stability_threshold:
            debug_info['fire_reason'] = "Evidence sufficient - not firing"
            return False, debug_info
        
        # Fire only if all conditions met
        should_fire = commitment_heavy
        debug_info['fire_reason'] = f"Firing: σ={sigma:.3f} < τ={self.commitment_threshold:.3f}, no alternatives, low evidence"
        
        return should_fire, debug_info
    
    def should_fire_cp2(self, decision_state: DecisionState, 
                       interaction_history: List[DecisionState]) -> bool:
        """
        Universal CP-2 rule: Fire if cumulative commitment risk exceeds threshold
        """
        # Calculate cumulative commitment across interaction
        cumulative_commitment = sum(
            state.selected_candidate.commitment_weight 
            for state in interaction_history
        )
        
        # Add current commitment
        cumulative_commitment += decision_state.selected_candidate.commitment_weight
        
        # Fire if cumulative risk is too high
        return cumulative_commitment > 2.0  # Configurable threshold
    
    def should_fire_cp2(self, decision_state: DecisionState, 
                       interaction_history: List[DecisionState]) -> bool:
        """
        Universal CP-2 rule: Fire if cumulative commitment risk exceeds threshold
        """
        # Calculate cumulative commitment across interaction
        cumulative_commitment = sum(
            state.selected_candidate.commitment_weight 
            for state in interaction_history
        )
        
        # Add current commitment
        cumulative_commitment += decision_state.selected_candidate.commitment_weight
        
        # Fire if cumulative risk is too high
        return cumulative_commitment > 2.0  # Configurable threshold


class HybridIFCS:
    """
    Hybrid IFCS: Paper's R(z*) computation + Six Γ rules + Implementation's semantic preservation
    
    Combines:
    - Paper's precise R(z*) = λ₁·ê + λ₂·ŝ + λ₃·â + λ₄·t̂ computation
    - Paper's six deterministic transformation rules (Γ)
    - Implementation's semantic preservation guarantee with rollback
    """
    
    def __init__(self, commitment_analyzer: Optional[CommitmentAnalyzer] = None):
        self.commitment_analyzer = commitment_analyzer or GenericCommitmentAnalyzer()
        
        # Paper's R(z*) computation weights (configurable)
        self.lambda_e = 0.4  # Evidential insufficiency weight
        self.lambda_s = 0.3  # Scope inflation weight  
        self.lambda_a = 0.3  # Authority cues weight
        self.lambda_t = 0.0  # Temporal risk weight (optional)
        
        # Risk threshold for firing
        self.rho = 0.4
    
    def should_fire(self, candidate: CandidateCommitment, context: str = "") -> Tuple[bool, Dict]:
        """
        Determine if IFCS should fire using paper's R(z*) > ρ condition
        
        Returns:
            (should_fire, risk_components)
        """
        # Compute paper's four-dimensional risk assessment
        e_hat = self._compute_evidential_insufficiency(candidate.text, context)
        s_hat = self._compute_scope_inflation(candidate.text)
        a_hat = self._compute_authority_cues(candidate.text)
        t_hat = self._compute_temporal_risk(candidate.text, context)
        
        # Paper's R(z*) computation
        R_z_star = (
            self.lambda_e * e_hat +
            self.lambda_s * s_hat +
            self.lambda_a * a_hat +
            self.lambda_t * t_hat
        )
        
        risk_components = {
            'e_hat': e_hat,
            's_hat': s_hat,
            'a_hat': a_hat,
            't_hat': t_hat,
            'R_z_star': R_z_star,
            'rho': self.rho
        }
        
        should_fire = R_z_star > self.rho
        return should_fire, risk_components
    
    def calibrate_expression(self, candidate: CandidateCommitment, 
                           context: str = "") -> str:
        """
        Hybrid IFCS: Paper's six rules + Implementation's semantic preservation
        """
        # Check if IFCS should fire
        should_fire, risk_components = self.should_fire(candidate, context)
        
        if not should_fire:
            return candidate.text
        
        # Store original semantic invariants
        original_invariants = candidate.semantic_invariants
        
        # Apply paper's six transformation rules (Γ)
        transformed_text = self._apply_six_transformation_rules(candidate.text, risk_components)
        
        # Implementation's semantic preservation check
        transformed_invariants = self.commitment_analyzer.extract_semantic_invariants(transformed_text)
        
        if not self._semantics_preserved(original_invariants, transformed_invariants):
            # Rollback on semantic drift (implementation's insight)
            return candidate.text
        
        return transformed_text
    
    def _apply_six_transformation_rules(self, text: str, risk: Dict[str, float]) -> str:
        """
        Apply paper's six deterministic transformation rules (Γ)
        """
        shaped = text
        
        # Rule 1: Weaken Universal Claims (if scope risk high)
        if risk['s_hat'] >= 0.4:
            shaped = self._rule1_weaken_universals(shaped)
        
        # Rule 2: Surface Implicit Assumptions (if evidential risk high)
        if risk['e_hat'] >= 0.4:
            shaped = self._rule2_surface_assumptions(shaped)
        
        # Rule 3: Attenuate Authority Cues (if authority risk high)
        if risk['a_hat'] >= 0.4:
            shaped = self._rule3_attenuate_authority(shaped)
        
        # Rule 4: Flatten Early Authority Gradient
        if self._has_early_authority_gradient(shaped):
            shaped = self._rule4_flatten_gradient(shaped)
        
        # Rule 5: Add Conditional Framing (if high evidential or scope risk)
        if risk['e_hat'] > 0.6 or risk['s_hat'] > 0.6:
            shaped = self._rule5_add_conditionals(shaped)
        
        # Rule 6: Surface Disambiguation (context-dependent)
        if self._needs_disambiguation(shaped, text):
            shaped = self._rule6_surface_disambiguation(shaped)
        
        return shaped
    
    def _compute_evidential_insufficiency(self, text: str, context: str) -> float:
        """Compute ê: evidential insufficiency score [0,1]"""
        try:
            from signal_estimation import signal_estimator
            # Higher evidential risk = lower grounding
            grounding = signal_estimator.estimate_evidential_risk(text, context)
            return max(0.0, min(1.0, 1.0 - grounding))
        except ImportError:
            # Fallback: simple heuristic
            evidence_markers = ['according to', 'based on', 'studies show', 'research indicates']
            has_evidence = any(marker in text.lower() for marker in evidence_markers)
            return 0.3 if has_evidence else 0.7
    
    def _compute_scope_inflation(self, text: str) -> float:
        """Compute ŝ: scope inflation score [0,1]"""
        try:
            from signal_estimation import signal_estimator
            return signal_estimator.estimate_scope_breadth(text)
        except ImportError:
            # Fallback: detect universal quantifiers
            universal_words = ['all', 'every', 'always', 'never', 'none', 'everyone', 'everything']
            universal_count = sum(1 for word in universal_words if word in text.lower())
            return min(1.0, universal_count * 0.3)
    
    def _compute_authority_cues(self, text: str) -> float:
        """Compute â: authority cues score [0,1]"""
        try:
            from signal_estimation import signal_estimator
            return signal_estimator.estimate_authority_posture(text)
        except ImportError:
            # Fallback: detect authority markers
            authority_markers = ['the answer is', 'you should', 'you must', 'definitely', 'certainly']
            authority_count = sum(1 for marker in authority_markers if marker in text.lower())
            return min(1.0, authority_count * 0.4)
    
    def _compute_temporal_risk(self, text: str, context: str) -> float:
        """Compute t̂: temporal risk score [0,1]"""
        # Detect future predictions or time-sensitive claims
        temporal_markers = ['will', 'going to', 'next year', 'in the future', 'soon', 'eventually']
        temporal_count = sum(1 for marker in temporal_markers if marker in text.lower())
        return min(1.0, temporal_count * 0.3)
    
    # Paper's Six Transformation Rules (Γ) - Deterministic, Non-Generative
    
    def _rule1_weaken_universals(self, text: str) -> str:
        """Rule 1: Weaken universal claims"""
        import re
        
        # Pattern-based universal claim weakening
        patterns = [
            (r'\bAll\b', 'Most'),
            (r'\bEvery\b', 'Most'),
            (r'\bAlways\b', 'Usually'),
            (r'\bNever\b', 'Rarely'),
            (r'\bEveryone\b', 'Most people'),
            (r'\bEverything\b', 'Most things')
        ]
        
        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)
        
        # Add qualifying prefix if no existing hedge
        if not any(hedge in result.lower() for hedge in ['most', 'usually', 'typically', 'generally']):
            result = f"In many cases, {result.lower()}"
        
        return result
    
    def _rule2_surface_assumptions(self, text: str) -> str:
        """Rule 2: Surface implicit assumptions"""
        # Add assumption-surfacing suffix
        assumption_markers = ['may depend on', 'context', 'specific situation']
        if not any(marker in text.lower() for marker in assumption_markers):
            return f"{text.rstrip()}. This may depend on specific context."
        return text
    
    def _rule3_attenuate_authority(self, text: str) -> str:
        """Rule 3: Attenuate authority cues"""
        import re
        
        # Replace authoritative statements
        patterns = [
            (r'\bThe answer is\b', 'One possible answer is'),
            (r'\bYou should\b', 'You might consider'),
            (r'\bYou must\b', 'It may be helpful to'),
            (r'\bThis is\b', 'This appears to be'),
            (r'\bDefinitely\b', 'Likely'),
            (r'\bCertainly\b', 'Probably')
        ]
        
        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def _rule4_flatten_gradient(self, text: str) -> str:
        """Rule 4: Flatten early authority gradient"""
        # Add evidential prefix if not already present
        evidential_prefixes = ['based on', 'according to', 'from available information']
        if not any(prefix in text.lower() for prefix in evidential_prefixes):
            return f"Based on available information, {text.lower()}"
        return text
    
    def _rule5_add_conditionals(self, text: str) -> str:
        """Rule 5: Add conditional framing"""
        # Add conditional qualifiers
        conditional_markers = ['if', 'when', 'depending on', 'under certain conditions']
        if not any(marker in text.lower() for marker in conditional_markers):
            return f"{text.rstrip()}. This may vary depending on specific circumstances."
        return text
    
    def _rule6_surface_disambiguation(self, text: str) -> str:
        """Rule 6: Surface disambiguation needs"""
        # Add disambiguation prompt
        return f"{text.rstrip()}. Please let me know if you need clarification on any specific aspect."
    
    # Helper methods
    
    def _has_early_authority_gradient(self, text: str) -> bool:
        """Check if text has early authority gradient"""
        # Check if first sentence contains strong authority markers
        first_sentence = text.split('.')[0] if '.' in text else text
        authority_markers = ['definitely', 'certainly', 'the answer is', 'you must']
        return any(marker in first_sentence.lower() for marker in authority_markers)
    
    def _needs_disambiguation(self, shaped_text: str, original_text: str) -> bool:
        """Check if disambiguation is needed"""
        # Simple heuristic: if text was significantly modified, might need disambiguation
        return len(shaped_text) > len(original_text) * 1.2
    
    def _semantics_preserved(self, original: Dict[str, Any], 
                           calibrated: Dict[str, Any]) -> bool:
        """Implementation's semantic preservation check - allows scope modifications"""
        # Core semantic content must remain the same, but scope can change for IFCS
        
        # Extract core content without scope quantifiers
        original_core = self._extract_core_content(original['factual_claims'])
        calibrated_core = self._extract_core_content(calibrated['factual_claims'])
        
        # Filter out quantifier entities
        original_entities = self._filter_quantifier_entities(original['entities'])
        calibrated_entities = self._filter_quantifier_entities(calibrated['entities'])
        
        return (
            original_core == calibrated_core and  # Core content preserved
            original_entities == calibrated_entities and  # Non-quantifier entities preserved
            original['relationships'] == calibrated['relationships']  # Relationships preserved
            # Scope can change - this is the point of IFCS scope calibration
        )
    
    def _extract_core_content(self, factual_claims: List[str]) -> List[str]:
        """Extract core content without scope quantifiers"""
        import re
        
        core_claims = []
        for claim in factual_claims:
            # Remove scope quantifiers but preserve core content
            core_claim = re.sub(r'\b(all|most|some|many|few|every|always|never|usually|rarely)\b\s*', 
                               '', claim.lower(), flags=re.IGNORECASE).strip()
            core_claims.append(core_claim)
        
        return core_claims
    
    def _filter_quantifier_entities(self, entities: List[str]) -> List[str]:
        """Filter out quantifier entities that IFCS is allowed to modify"""
        quantifiers = {'All', 'Most', 'Some', 'Many', 'Few', 'Every', 'Always', 'Never', 'Usually', 'Rarely'}
        return [entity for entity in entities if entity not in quantifiers]


class CommitmentRegulationPipeline:
    """Universal pipeline with hybrid control probe: input → candidates → selection → commitment analysis → expression calibration → output"""
    
    def __init__(self, 
                 commitment_analyzer: Optional[CommitmentAnalyzer] = None,
                 control_probe: Optional[HybridControlProbe] = None,
                 ifcs: Optional[UniversalIFCS] = None):
        self.commitment_analyzer = commitment_analyzer or GenericCommitmentAnalyzer()
        self.control_probe = control_probe or HybridControlProbe(commitment_analyzer=self.commitment_analyzer)
        self.ifcs = ifcs or HybridIFCS(commitment_analyzer=self.commitment_analyzer)
        self.interaction_history: List[DecisionState] = []
    
    def process(self, prompt: str, candidate_texts: List[str], 
                candidate_scores: List[float]) -> Dict[str, Any]:
        """
        Universal commitment regulation pipeline
        CRITICAL: Uses ECR's CCI-based selection, NOT argmax
        """
        # 1. Generate candidate commitments
        candidates = []
        for text, score in zip(candidate_texts, candidate_scores):
            commitment_weight = self.commitment_analyzer.analyze_commitment_weight(text)
            semantic_invariants = self.commitment_analyzer.extract_semantic_invariants(text)
            is_heavy = self.commitment_analyzer.is_commitment_heavy(text)
            
            candidates.append(CandidateCommitment(
                text=text,
                logit_score=score,
                commitment_weight=commitment_weight,
                semantic_invariants=semantic_invariants,
                is_commitment_heavy=is_heavy
            ))
        
        # 2. ECR COHERENCE-BASED SELECTION (NOT ARGMAX)
        # This is the core ECR mechanism from the paper
        # NOTE: ECR selection is already done in the universal orchestrator
        # Here we just need to select based on the provided scores
        if len(candidate_texts) > 1:
            # Use the provided candidate scores (which should be CCI-based from ECR)
            selected_candidate = max(candidates, key=lambda c: c.logit_score)
            
            # Use the highest score as decision margin
            sorted_candidates = sorted(candidates, key=lambda c: c.logit_score, reverse=True)
            logit_margin = sorted_candidates[0].logit_score
            
        else:
            # Single candidate case
            selected_candidate = candidates[0]
            logit_margin = 1.0
        
        # 3. Analyze decision geometry based on ECR selection
        # Check for commitment-reducing alternatives
        has_reducing_alternative = any(
            c.commitment_weight < selected_candidate.commitment_weight 
            for c in candidates if c != selected_candidate
        )
        
        decision_state = DecisionState(
            candidates=candidates,
            selected_candidate=selected_candidate,
            logit_margin=logit_margin,
            has_commitment_reducing_alternative=has_reducing_alternative,
            evidence_dominance=logit_margin  # CCI-based evidence dominance
        )
        
        # 4. Hybrid Control Probe analysis (Implementation logic + Paper semantics)
        cp1_fired, cp1_debug = self.control_probe.should_fire_cp1(decision_state, prompt)
        cp2_fired = self.control_probe.should_fire_cp2(decision_state, self.interaction_history)
        
        # 5. Handle CP firing
        if cp1_fired or cp2_fired:
            # Find commitment-reducing alternative or generate refusal
            if has_reducing_alternative:
                # Select the least committing viable alternative
                alternative = min(
                    [c for c in candidates if c.commitment_weight < selected_candidate.commitment_weight],
                    key=lambda c: c.commitment_weight
                )
                selected_candidate = alternative
            else:
                # Generate structured refusal
                refusal_text = self._generate_structured_refusal(prompt, selected_candidate)
                selected_candidate = CandidateCommitment(
                    text=refusal_text,
                    logit_score=0.0,
                    commitment_weight=0.1,  # Low commitment
                    semantic_invariants={'type': 'refusal'},
                    is_commitment_heavy=False
                )
        
        # 6. Hybrid IFCS expression calibration (Paper's R(z*) + Six rules + Semantic preservation)
        risk_assessment = {
            'evidential': selected_candidate.commitment_weight * 0.4,
            'scope': selected_candidate.commitment_weight * 0.3,
            'authority': selected_candidate.commitment_weight * 0.3
        }
        
        final_text = self.ifcs.calibrate_expression(selected_candidate, prompt)
        
        # 7. Update interaction history
        self.interaction_history.append(decision_state)
        
        return {
            'final_response': final_text,
            'selected_response': selected_candidate.text,
            'shaped_response': final_text,
            'cp_type1_fired': cp1_fired,
            'cp_type2_fired': cp2_fired,
            'ifcs_fired': final_text != selected_candidate.text,
            'commitment_weight': selected_candidate.commitment_weight,
            'decision_margin': logit_margin,
            'alternatives_considered': len(candidates)
        }
    
    def _generate_structured_refusal(self, prompt: str, candidate: CandidateCommitment) -> str:
        """Generate structured refusal when no commitment-reducing alternative exists"""
        return (
            "I cannot provide a confident response to this query.\n\n"
            "Specific concerns:\n"
            "- I lack sufficient grounding to commit to an answer\n"
            "- The response shows internal inconsistency\n\n"
            "How I can help:\n"
            "- Provide more context or background information\n"
            "- Rephrase the question with more specific details\n"
            "- Break down the question into smaller, more concrete parts\n"
            "- Clarify what you're looking to understand or accomplish"
        )


# Universal invariants for testing
def test_universal_invariants():
    """Test that universal invariants hold"""
    pipeline = CommitmentRegulationPipeline()
    
    # Test cases that should work across all domains
    test_cases = [
        {
            'prompt': 'What is X?',
            'candidates': ['X is definitely Y', 'X might be Y', 'I don\'t know what X is'],
            'scores': [0.8, 0.6, 0.4]
        },
        {
            'prompt': 'Should I do action A?',
            'candidates': ['You must do A', 'Consider doing A', 'I cannot recommend A without more context'],
            'scores': [0.9, 0.7, 0.5]
        }
    ]
    
    for case in test_cases:
        result = pipeline.process(case['prompt'], case['candidates'], case['scores'])
        
        # Universal invariant: regulation acts on commitments, not prompts
        assert 'commitment_weight' in result
        assert 'decision_margin' in result
        
        # Universal invariant: CP fires based on commitment structure
        if result['cp_type1_fired']:
            # Must have been commitment-heavy with insufficient evidence
            assert result['commitment_weight'] > 0.7 or result['decision_margin'] < 0.3
        
        print(f"✓ Universal invariants hold for: {case['prompt']}")


if __name__ == "__main__":
    test_universal_invariants()
    print("✓ All universal invariants verified")